"""
Generate a JSON description of the Neuron3d_192_192_192 dataset splits.

The script reads train/test/val TXT files and emits absolute paths for the
image, mask, and swc files in each split so downstream training code can
consume a single JSON manifest.

Example:
    python export_dataset_json.py \
        --dataset-root E:/NeuronOpenSource/Neuron3d_192_192_192 \
        --images-dir images/images-8bit \
        --output dataset_splits.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a JSON file containing absolute paths for each split."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=r"E:\NeuronOpenSource\Neuron3d_192_192_192",
        help="Root directory that stores images/, mask/, swc/ and TXT split files.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("images") / "images-8bit",
        help="Directory containing image *.tif files (relative to dataset root).",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=Path("mask") / "mask",
        help="Directory containing mask *.tif files (relative to dataset root).",
    )
    parser.add_argument(
        "--swc-dir",
        type=Path,
        default=Path("swc") / "swc",
        help="Directory containing *.swc annotation files (relative to dataset root).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Split names to export. Each requires a matching <name>.txt file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination JSON path. Defaults to <dataset-root>/dataset_splits.json.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation level for pretty-printing JSON (set 0 for compact).",
    )
    return parser.parse_args()


def ensure_suffix(name: str, suffix: str) -> str:
    return name if name.endswith(suffix) else f"{name}{suffix}"


def read_names(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_split_records(
    names: List[str], image_dir: Path, mask_dir: Path, swc_dir: Path
) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for name in names:
        tif_name = ensure_suffix(name, ".tif")
        swc_name = Path(tif_name).with_suffix(".swc").name
        record = {
            "image": str((image_dir / tif_name).resolve()),
            "mask": str((mask_dir / tif_name).resolve()),
            "swc": str((swc_dir / swc_name).resolve()),
            "id": Path(tif_name).stem,
        }
        records.append(record)
    return records


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    image_dir = (
        args.images_dir
        if args.images_dir.is_absolute()
        else dataset_root / args.images_dir
    )
    masks_dir = (
        args.masks_dir if args.masks_dir.is_absolute() else dataset_root / args.masks_dir
    )
    swc_dir = args.swc_dir if args.swc_dir.is_absolute() else dataset_root / args.swc_dir
    output = (
        args.output.expanduser().resolve()
        if args.output
        else dataset_root / "dataset_splits.json"
    )

    manifest: Dict[str, List[Dict[str, str]]] = {}

    for split in args.splits:
        split_file = dataset_root / f"{split}.txt"
        if not split_file.exists():
            print(f"[WARN] Missing split file: {split_file}, skipping.")
            continue
        names = read_names(split_file)
        manifest[split] = build_split_records(names, image_dir, masks_dir, swc_dir)
        print(f"{split}: {len(manifest[split])} entries")

    json_kwargs = {"indent": args.indent} if args.indent > 0 else {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, **json_kwargs)
    print(f"Wrote JSON manifest to {output}")


if __name__ == "__main__":
    main()
