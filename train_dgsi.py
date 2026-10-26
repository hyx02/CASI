"""Stage 2: category-shared dual-branch diffusion with frozen CODD and trainable DGSI."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--codd_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dataset", required=True, choices=["mvtec", "visa"])
    parser.add_argument("--category", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_manifest", default=None, help="Defaults to the CODD training manifest")
    parser.add_argument("--train_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lambda_defect", type=float, default=5.0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="Use CPU for small smoke tests")
    return parser.parse_args()


def main(args):
    import torch
    import torch.nn.functional as F
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from transformers import CLIPTextModel, CLIPTokenizer

    from casi.codd import CODD
    from casi.data import CASIDataset, load_manifest
    from casi.dgsi import install_dgsi, save_dgsi, load_dgsi

    if min(args.train_steps, args.batch_size, args.gradient_accumulation_steps) < 1:
        raise ValueError("Training steps, batch size and accumulation must be positive")
    set_seed(args.seed)
    accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.gradient_accumulation_steps)
    device = accelerator.device
    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]
    output = Path(args.output_dir)
    if (output / "pytorch_lora_weights.bin").exists() and not args.resume_from_checkpoint:
        raise FileExistsError(f"DGSI weights already exist: {output}. Use another output directory.")
    manifest_path = Path(args.split_manifest) if args.split_manifest else Path(args.codd_path) / "training_samples.json"
    samples = load_manifest(manifest_path, args.dataset, args.category)
    dataset = CASIDataset(args.data_root, args.category, samples, args.resolution, augment=True, center_crop=args.center_crop)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        (output / "training_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        (output / "training_samples.json").write_bytes(manifest_path.read_bytes())

    load_kwargs = {"pretrained_model_name_or_path": args.model_path, "local_files_only": True}
    tokenizer = CLIPTokenizer.from_pretrained(**load_kwargs, subfolder="tokenizer")
    encoder = CLIPTextModel.from_pretrained(**load_kwargs, subfolder="text_encoder")
    codd = CODD.from_pretrained(encoder, args.codd_path).requires_grad_(False).to(device=device, dtype=dtype).eval()
    vae = AutoencoderKL.from_pretrained(**load_kwargs, subfolder="vae").requires_grad_(False).to(device=device, dtype=dtype).eval()
    scheduler = DDPMScheduler.from_pretrained(**load_kwargs, subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(**load_kwargs, subfolder="unet").requires_grad_(False).to(device=device, dtype=dtype)
    # Keep trainable parameters in fp32 when the frozen backbone uses fp16.
    trainable = install_dgsi(unet, args.rank).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(trainable.parameters(), lr=args.learning_rate, weight_decay=0.01)
    unet, optimizer, loader = accelerator.prepare(unet, optimizer, loader)

    def save_model_hook(models, weights, directory):
        if accelerator.is_main_process:
            save_dgsi(accelerator.unwrap_model(models[0]), directory)
        weights.clear()  # Do not copy the frozen SD backbone into every checkpoint.

    def load_model_hook(models, directory):
        for model in models:
            load_dgsi(accelerator.unwrap_model(model), directory, replace=False)
        models.clear()

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    global_step = 0
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        global_step = json.loads((Path(args.resume_from_checkpoint) / "training_state.json").read_text())["global_step"]

    def batches():
        while True:
            yield from loader

    iterator = batches()
    progress = tqdm(total=args.train_steps, initial=global_step, desc="DGSI", disable=not accelerator.is_local_main_process)
    unet.train()
    while global_step < args.train_steps:
        batch = next(iterator)
        with accelerator.accumulate(unet):
            pixels = batch["pixel_values"].to(device=device, dtype=dtype)
            mask = batch["mask"].to(device=device, dtype=dtype)
            batch_size = pixels.shape[0]
            with torch.no_grad():
                global_latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
                defect_latents = vae.encode(pixels * mask).latent_dist.sample() * vae.config.scaling_factor
                global_hidden = codd.encode_texts(tokenizer, [args.category] * batch_size, list(batch["defect"]))
                defect_hidden = codd.encode_texts(tokenizer, [args.category] * batch_size, list(batch["defect"]), branch="defect")
            latents = torch.stack((global_latents, defect_latents), dim=1).flatten(0, 1)
            hidden = torch.stack((global_hidden, defect_hidden), dim=1).flatten(0, 1)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device).repeat_interleave(2)
            noisy = scheduler.add_noise(latents, noise, timesteps)
            predicted = unet(noisy, timesteps, encoder_hidden_states=hidden).sample
            if scheduler.config.prediction_type == "epsilon":
                target = noise
            elif scheduler.config.prediction_type == "v_prediction":
                target = scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unsupported prediction type: {scheduler.config.prediction_type}")
            weights = F.interpolate(mask.float(), size=latents.shape[-2:], mode="nearest")
            weights = (1 + (args.lambda_defect - 1) * weights).repeat_interleave(2, dim=0)
            loss = (weights * (predicted.float() - target.float()).square()).mean()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if accelerator.sync_gradients:
            global_step += 1
            progress.update()
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0:
                checkpoint = output / f"checkpoint-{global_step}"
                accelerator.save_state(str(checkpoint))
                if accelerator.is_main_process:
                    (checkpoint / "training_state.json").write_text(json.dumps({"global_step": global_step}), encoding="utf-8")
    progress.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_dgsi(accelerator.unwrap_model(unet), output)
        print(f"DGSI saved: {output}")
    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
