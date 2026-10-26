"""Stage 1: train Category-Shared Object-Defect Disentanglement (CODD)."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dataset", choices=["mvtec", "visa"], required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--defects", default="", help="Comma-separated; empty means all defects of the category")
    parser.add_argument("--split_file", default=None, help="Optional VisA split_csv/1cls.csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_fraction", type=int, default=3, help="Use ceil(N/data_fraction) samples per defect")
    parser.add_argument("--num_adapter_layers", type=int, default=4)
    parser.add_argument("--bottleneck", type=int, default=192)
    parser.add_argument("--train_steps", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lambda_dis1", type=float, default=0.1)
    parser.add_argument("--lambda_dis2", type=float, default=0.05)
    parser.add_argument("--lambda_obj", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main(args):
    import torch
    import torch.nn.functional as F
    from accelerate.utils import set_seed
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from transformers import CLIPTextModel, CLIPTokenizer

    from casi.codd import CODD, semantic_losses
    from casi.data import CASIDataset, collect_samples, select_samples, save_manifest

    if args.train_steps < 1 or args.batch_size < 1:
        raise ValueError("train_steps and batch_size must be positive")
    set_seed(args.seed)
    output = Path(args.output_dir)
    if (output / "adapters.pt").exists():
        raise FileExistsError(f"CODD weights already exist: {output}. Use another output directory.")
    defects = [name.strip() for name in args.defects.split(",") if name.strip()] or None
    samples = select_samples(collect_samples(
        args.data_root, args.dataset, args.category, defects, args.split_file,
    ), args.data_fraction)
    dataset = CASIDataset(args.data_root, args.category, samples, args.resolution)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    save_manifest(output / "training_samples.json", samples, args.dataset, args.category, args.data_fraction)
    (output / "training_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    load_kwargs = {"pretrained_model_name_or_path": args.model_path, "local_files_only": True}
    tokenizer = CLIPTokenizer.from_pretrained(**load_kwargs, subfolder="tokenizer")
    encoder = CLIPTextModel.from_pretrained(**load_kwargs, subfolder="text_encoder")
    codd = CODD(encoder, args.num_adapter_layers, args.bottleneck).to(args.device)
    unet = UNet2DConditionModel.from_pretrained(**load_kwargs, subfolder="unet").requires_grad_(False).to(args.device).eval()
    vae = AutoencoderKL.from_pretrained(**load_kwargs, subfolder="vae").requires_grad_(False).to(args.device).eval()
    scheduler = DDPMScheduler.from_pretrained(**load_kwargs, subfolder="scheduler")
    optimizer = torch.optim.AdamW(codd.adapter_parameters(), lr=args.learning_rate, weight_decay=0.01)
    codd.train()
    codd.text_encoder.eval()
    iterator = iter(loader)
    for step in (progress := tqdm(range(args.train_steps), desc="CODD")):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        pixels = batch["pixel_values"].to(args.device)
        with torch.no_grad():
            latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (pixels.shape[0],), device=args.device)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        hidden = codd.encode_texts(tokenizer, [args.category] * pixels.shape[0], list(batch["defect"]))
        predicted = unet(noisy_latents, timesteps, encoder_hidden_states=hidden).sample
        if scheduler.config.prediction_type == "epsilon":
            target = noise
        elif scheduler.config.prediction_type == "v_prediction":
            target = scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction type: {scheduler.config.prediction_type}")
        loss_diff = F.mse_loss(predicted.float(), target.float())
        loss_dis1, loss_dis2, loss_obj = semantic_losses(
            codd, tokenizer, args.category, dataset.defects, batch["defect"], args.margin,
        )
        loss = loss_diff + args.lambda_dis1 * loss_dis1 + args.lambda_dis2 * loss_dis2 + args.lambda_obj * loss_obj
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        progress.set_postfix(loss=f"{loss.item():.4f}", diffusion=f"{loss_diff.item():.4f}")
    codd.save_pretrained(output)
    print(f"CODD saved: {output}")


if __name__ == "__main__":
    main(parse_args())
