"""Shared MVTec AD / VisA data loading for both CASI stages."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VISA_ID2CLASS = json.loads(Path(__file__).with_name("visa_defects.json").read_text(encoding="utf-8"))
VISA_ID2CLASS = {category: {int(key): value for key, value in names.items()} for category, names in VISA_ID2CLASS.items()}


def image_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def read_mask(path):
    with Image.open(path) as image:
        array = np.array(ImageOps.exif_transpose(image))
    return array[..., 0] if array.ndim == 3 else array


def collect_samples(data_root, dataset, category, defects=None, split_file=None):
    """Collect source samples without silently treating missing masks as normal data."""
    root = Path(data_root).resolve()
    samples = []
    if dataset == "mvtec":
        test_root = root / category / "test"
        if not test_root.is_dir():
            raise FileNotFoundError(test_root)
        names = sorted(path.name for path in test_root.iterdir() if path.is_dir() and path.name != "good")
        if defects:
            missing = set(defects) - set(names)
            if missing:
                raise ValueError(f"Unknown defects: {sorted(missing)}")
        for name in names:
            if defects and name not in defects:
                continue
            for image in image_files(test_root / name):
                mask_root = root / category / "ground_truth" / name
                candidates = [mask_root / f"{image.stem}_mask.png", mask_root / image.name]
                mask = next((path for path in candidates if path.is_file()), None)
                if mask is None:
                    raise FileNotFoundError(f"No mask for {image}")
                samples.append({"image": image.relative_to(root).as_posix(), "mask": mask.relative_to(root).as_posix(),
                                "defect": name, "mask_id": None})
    elif dataset == "visa":
        if category not in VISA_ID2CLASS:
            raise ValueError(f"Unknown VisA category: {category}")
        pairs = []
        if split_file:
            with Path(split_file).open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    if row.get("object", "").strip() != category:
                        continue
                    split = (row.get("split") or row.get("set") or "").strip().lower()
                    if split == "test" and row.get("label", "").strip().lower() == "anomaly":
                        image = row.get("image") or row.get("image_path")
                        mask = row.get("mask") or row.get("mask_path")
                        if not image or not mask:
                            raise ValueError(f"Missing image/mask in split row: {row}")
                        pairs.append((root / image, root / mask))
        else:
            image_root = root / category / "Data" / "Images" / "Anomaly"
            masks = {path.stem.lower(): path for path in image_files(root / category / "Data" / "Masks" / "Anomaly")}
            for image in image_files(image_root):
                if image.stem.lower() not in masks:
                    raise FileNotFoundError(f"No mask for {image}")
                pairs.append((image, masks[image.stem.lower()]))
        for image, mask in pairs:
            if not image.is_file() or not mask.is_file():
                raise FileNotFoundError(f"Missing image/mask: {image}, {mask}")
            ids = sorted(int(value) for value in np.unique(read_mask(mask)) if value != 0)
            for mask_id in ids:
                if mask_id not in VISA_ID2CLASS[category]:
                    raise ValueError(f"Unknown VisA semantic mask ID {mask_id}: {mask}")
                name = VISA_ID2CLASS[category][mask_id]
                if not defects or name in defects:
                    samples.append({"image": image.relative_to(root).as_posix(), "mask": mask.relative_to(root).as_posix(),
                                    "defect": name, "mask_id": mask_id})
        if defects:
            missing = set(defects) - {sample["defect"] for sample in samples}
            if missing:
                raise ValueError(f"No images found for VisA defects: {sorted(missing)}")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    if not samples:
        raise ValueError(f"No valid anomalous samples for {dataset}/{category}")
    return samples


def select_samples(samples, data_fraction=3):
    if data_fraction < 1:
        raise ValueError("data_fraction must be >= 1")
    groups = defaultdict(list)
    for sample in samples:
        groups[sample["defect"]].append(sample)
    selected = []
    for defect in sorted(groups):
        group = sorted(groups[defect], key=lambda sample: sample["image"])
        selected.extend(group[:max(1, math.ceil(len(group) / data_fraction))])
    return selected


def save_manifest(path, samples, dataset, category, data_fraction):
    payload = {"dataset": dataset, "category": category, "data_fraction": data_fraction, "samples": samples}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest(path, dataset, category):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest["dataset"] != dataset or manifest["category"] != category:
        raise ValueError("Training manifest does not match the requested dataset/category")
    if not manifest["samples"]:
        raise ValueError("Empty training manifest")
    return manifest["samples"]


def paired_transform(image, mask, size, augment=False, flip=True, center_crop=False):
    """One geometry for both image and mask; masks always use nearest-neighbor resize."""
    if image.size != mask.size:
        raise ValueError(f"Image/mask sizes differ: {image.size}, {mask.size}")
    if size <= 0 or size % 8:
        raise ValueError("resolution must be a positive multiple of 8")
    width, height = image.size
    scale = size / min(width, height)
    resized = (max(size, round(width * scale)), max(size, round(height * scale)))
    image = image.resize(resized, Image.Resampling.BILINEAR)
    mask = mask.resize(resized, Image.Resampling.NEAREST)
    if augment and flip:
        if torch.rand(()).item() < 0.5:
            image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
        if torch.rand(()).item() < 0.5:
            image, mask = ImageOps.flip(image), ImageOps.flip(mask)
    width, height = resized
    if augment and not center_crop:
        left = int(torch.randint(width - size + 1, ()).item())
        top = int(torch.randint(height - size + 1, ()).item())
    else:
        left, top = (width - size) // 2, (height - size) // 2
    box = left, top, left + size, top + size
    pixels = torch.from_numpy(np.array(image.crop(box), dtype=np.float32).copy()).permute(2, 0, 1) / 127.5 - 1
    binary_mask = torch.from_numpy((np.array(mask.crop(box)) > 0).astype(np.float32)).unsqueeze(0)
    return pixels, binary_mask


class CASIDataset(Dataset):
    def __init__(self, data_root, category, samples, size=512, augment=False, center_crop=False):
        if not samples:
            raise ValueError("CASIDataset requires at least one sample")
        self.root = Path(data_root)
        self.category = category
        self.samples = samples
        self.size = size
        self.augment = augment
        self.center_crop = center_crop
        self.defects = sorted({sample["defect"] for sample in samples})
        for sample in samples:
            for key in ("image", "mask"):
                if not (self.root / sample[key]).is_file():
                    raise FileNotFoundError(self.root / sample[key])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(self.root / sample["image"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        mask = read_mask(self.root / sample["mask"])
        mask = mask > 0 if sample["mask_id"] is None else mask == sample["mask_id"]
        mask = Image.fromarray(mask.astype(np.uint8) * 255)
        pixels, mask = paired_transform(
            image, mask, self.size, self.augment,
            flip=self.category != "cable", center_crop=self.center_crop,
        )
        return {"pixel_values": pixels, "mask": mask, "defect": sample["defect"]}
