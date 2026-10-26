"""Generate images and defect foregrounds from one category-shared CASI model."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--codd_path", required=True)
    parser.add_argument("--dgsi_path", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--defects", default="", help="Comma-separated; defaults to defects in the CODD manifest")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_images", type=int, default=1000, help="Number per defect, including existing complete pairs")
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--seed", type=int, default=32)
    return parser.parse_args()


def main(args):
    import torch
    from casi.codd import CODD
    from casi.dgsi import load_dgsi
    from casi.pipeline import CASIPipeline

    if args.num_images < 1:
        raise ValueError("num_images must be positive")
    defects = [name.strip() for name in args.defects.split(",") if name.strip()]
    if not defects:
        manifest = json.loads((Path(args.codd_path) / "training_samples.json").read_text(encoding="utf-8"))
        if manifest["category"] != args.category:
            raise ValueError("CODD manifest category does not match --category")
        defects = sorted({sample["defect"] for sample in manifest["samples"]})
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    pipe = CASIPipeline.from_pretrained(
        args.model_path, safety_checker=None, requires_safety_checker=False,
        torch_dtype=dtype, local_files_only=True,
    ).to(args.device)
    load_dgsi(pipe.unet, args.dgsi_path)
    codd = CODD.from_pretrained(pipe.text_encoder, args.codd_path).to(device=args.device, dtype=dtype).eval()
    for defect_index, defect in enumerate(defects):
        with torch.no_grad():
            global_hidden = codd.encode_texts(pipe.tokenizer, [args.category], [defect])
            defect_hidden = codd.encode_texts(pipe.tokenizer, [args.category], [defect], branch="defect")
        directory = Path(args.output_dir) / args.category / defect
        image_dir, foreground_dir = directory / "image", directory / "fg"
        image_dir.mkdir(parents=True, exist_ok=True)
        foreground_dir.mkdir(parents=True, exist_ok=True)
        for index in range(args.num_images):
            image_path = image_dir / f"{index}.png"
            foreground_path = foreground_dir / f"{index}.png"
            if image_path.exists() and foreground_path.exists():
                continue
            # Each index has a fixed seed, including after interrupted generation.
            generator = torch.Generator(device=args.device).manual_seed(args.seed + defect_index * 1_000_000 + index)
            images = pipe(
                global_hidden, defect_hidden, num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale, generator=generator,
                height=args.resolution, width=args.resolution,
            ).images
            images[0].save(image_path)
            images[1].save(foreground_path)
            print(f"{args.category}/{defect}: {index + 1}/{args.num_images}")


if __name__ == "__main__":
    main(parse_args())
