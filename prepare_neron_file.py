import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import tifffile as tiff
from tqdm import tqdm


VALID_TIFF_SUFFIXES = {".tif", ".tiff"}
DEFAULT_SPLITS = ("train", "test", "val")


@dataclass(frozen=True)
class SampleRecord:
    stem: str
    image: Path
    mask: Path
    swc: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据 split_dataset.py 的逻辑，使用 train/test/eval TXT 文件划分数据集。"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"E:\NeuronOpenSource\Neuron3d_192_192_192"),
        help="原始数据所在目录（包含 images/, mask/, swc/ 以及 train.txt 等划分文件）。",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("images") / "images-8bit",
        help="相对 dataset-root 的图像目录，默认为 images/images-8bit。",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=Path("mask") / "mask",
        help="相对 dataset-root 的 mask 目录，默认为 mask/mask。",
    )
    parser.add_argument(
        "--swc-dir",
        type=Path,
        default=Path("swc") / "swc",
        help="相对 dataset-root 的 swc 目录，默认为 swc/swc。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"E:\Project\NeuronTracing\data"),
        help="整理后的数据将保存在该目录下，默认是 E:/Project/NeuronTracing/data。",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="Neuron3d_192_dataset",
        help="output-root 下用于保存本数据集的子目录名称。",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="需要处理的划分名称（默认 train test eval），将按照 <name>.txt 列表复制数据。",
    )
    return parser.parse_args()


def resolve_dir(base: Path, maybe_relative: Path) -> Path:
    return maybe_relative if maybe_relative.is_absolute() else base / maybe_relative


def collect_tiff_files(directory: Path) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"找不到目录：{directory}")
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_TIFF_SUFFIXES
        ]
    )


def collect_swc_files(directory: Path) -> Dict[str, Path]:
    if not directory.exists():
        raise FileNotFoundError(f"找不到 SWC 目录：{directory}")
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".swc"
    }


def build_records(image_dir: Path, mask_dir: Path, swc_dir: Path) -> List[SampleRecord]:
    image_paths = collect_tiff_files(image_dir)
    mask_map = {path.stem: path for path in collect_tiff_files(mask_dir)}
    swc_map = collect_swc_files(swc_dir)
    records: List[SampleRecord] = []

    for image_path in image_paths:
        stem = image_path.stem
        mask_path = mask_map.get(stem)
        swc_path = swc_map.get(stem)
        if mask_path is None:
            raise FileNotFoundError(
                f"缺少 mask：{mask_dir} 中未找到 {stem}.tif 或 {stem}.tiff"
            )
        if swc_path is None:
            raise FileNotFoundError(f"缺少 swc：{swc_dir}/{stem}.swc")
        records.append(SampleRecord(stem=stem, image=image_path, mask=mask_path, swc=swc_path))

    if not records:
        raise RuntimeError("未在给定目录中找到任何图像。")
    return records


def read_split_names(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [Path(line.strip()).stem for line in f if line.strip()]


def build_split_partitions(
    records: List[SampleRecord],
    dataset_root: Path,
    splits: Sequence[str],
) -> Dict[str, List[SampleRecord]]:
    record_map = {record.stem: record for record in records}
    partitions: Dict[str, List[SampleRecord]] = {}

    for split in splits:
        split_file = dataset_root / f"{split}.txt"
        if not split_file.exists():
            print(f"[WARN] Missing split file: {split_file}, skipping.")
            continue
        names = read_split_names(split_file)
        missing = [name for name in names if name not in record_map]
        if missing:
            preview = ", ".join(missing[:5])
            suffix = " ..." if len(missing) > 5 else ""
            raise FileNotFoundError(
                f"{split_file} 中存在未匹配的样本：{preview}{suffix}"
            )
        partitions[split] = [record_map[name] for name in names]
        print(f"{split}: {len(partitions[split])} entries")

    if not partitions:
        raise RuntimeError("未找到任何有效的划分文件，请确认 train/test/eval TXT 是否存在。")
    return partitions


def prepare_output_dirs(target_dir: Path, splits: Sequence[str]) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    subfolders = ("images", "labels", "swc")
    for split in splits:
        for folder in subfolders:
            (target_dir / split / folder).mkdir(parents=True, exist_ok=True)
    (target_dir / "dataset").mkdir(parents=True, exist_ok=True)


def binarize_and_save_mask(src: Path, dst: Path) -> None:
    lb = tiff.imread(src)
    label_max = int(lb.max())
    if label_max == 0:
        raise ValueError(f"{src} 中的像素值全为 0，无法二值化。")
    threshold = label_max * 0.5
    lb[lb < threshold] = 0
    lb[lb >= threshold] = 255
    tiff.imwrite(dst, lb.astype("uint8"))


def copy_partition(records: Iterable[SampleRecord], subset_dir: Path, desc: str) -> None:
    images_dir = subset_dir / "images"
    labels_dir = subset_dir / "labels"
    swc_dir = subset_dir / "swc"

    for record in tqdm(records, desc=desc):
        shutil.copy2(record.image, images_dir / record.image.name)
        binarize_and_save_mask(record.mask, labels_dir / f"{record.stem}.tif")
        shutil.copy2(record.swc, swc_dir / record.swc.name)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    image_dir = resolve_dir(dataset_root, args.images_dir)
    mask_dir = resolve_dir(dataset_root, args.masks_dir)
    swc_dir = resolve_dir(dataset_root, args.swc_dir)
    output_root = args.output_root.expanduser().resolve()
    target_dir = output_root / args.dataset_name

    records = build_records(image_dir, mask_dir, swc_dir)
    partitions = build_split_partitions(records, dataset_root, args.splits)

    split_names = list(partitions.keys())
    prepare_output_dirs(target_dir, split_names)
    for split_name, split_records in partitions.items():
        copy_partition(split_records, target_dir / split_name, f"{split_name}_dataset")
    print(f"数据已保存到 {target_dir}")


if __name__ == "__main__":
    main()
