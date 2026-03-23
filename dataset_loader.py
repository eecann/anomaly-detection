from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

SPLIT_ALIASES = {
    "train": {"train"},
    "val": {"val", "valid", "validation", "dev"},
    "test": {"test"},
}


@dataclass(frozen=True)
class Sample:
    path: Path
    class_name: str
    class_idx: int


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _sorted_subdirs(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_dir()], key=lambda p: p.name.lower())


def discover_split_dirs(data_root: Path) -> Dict[str, Path]:
    name_to_dir = {d.name.lower(): d for d in _sorted_subdirs(data_root)}
    split_dirs: Dict[str, Path] = {}

    for canonical_split, aliases in SPLIT_ALIASES.items():
        for alias in aliases:
            if alias in name_to_dir:
                split_dirs[canonical_split] = name_to_dir[alias]
                break

    return split_dirs


def discover_class_dirs(folder: Path) -> Dict[str, Path]:
    class_dirs = {}
    for subdir in _sorted_subdirs(folder):
        class_dirs[subdir.name] = subdir
    return class_dirs


def build_class_to_idx(class_names: Iterable[str]) -> Dict[str, int]:
    sorted_names = sorted(set(class_names), key=lambda x: x.lower())
    return {name: idx for idx, name in enumerate(sorted_names)}


def collect_samples_from_class_dirs(
    class_dirs: Dict[str, Path], class_to_idx: Dict[str, int]
) -> List[Sample]:
    samples: List[Sample] = []

    for class_name in sorted(class_dirs, key=lambda x: x.lower()):
        class_dir = class_dirs[class_name]
        image_paths = sorted(
            [p for p in class_dir.rglob("*") if is_image_file(p)],
            key=lambda p: str(p).lower(),
        )
        for image_path in image_paths:
            samples.append(
                Sample(path=image_path, class_name=class_name, class_idx=class_to_idx[class_name])
            )

    return samples


def class_counts(samples: Iterable[Sample]) -> Counter:
    return Counter(s.class_name for s in samples)


def _ratio_counts(n_items: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    if n_items == 0:
        return 0, 0, 0

    if any(r < 0 for r in ratios):
        raise ValueError("Split ratios must be non-negative.")

    total = sum(ratios)
    if total <= 0:
        raise ValueError("At least one split ratio must be positive.")

    normalized = [r / total for r in ratios]
    raw = [n_items * r for r in normalized]
    counts = [int(x) for x in raw]

    remaining = n_items - sum(counts)
    remainder_order = sorted(
        range(3), key=lambda i: (raw[i] - counts[i], normalized[i]), reverse=True
    )
    for i in remainder_order:
        if remaining == 0:
            break
        counts[i] += 1
        remaining -= 1

    positive_targets = [i for i, r in enumerate(normalized) if r > 0]
    if n_items >= len(positive_targets):
        for i in positive_targets:
            if counts[i] == 0:
                donor = max(
                    (j for j in range(3) if counts[j] > 1),
                    key=lambda j: counts[j],
                    default=None,
                )
                if donor is None:
                    break
                counts[donor] -= 1
                counts[i] += 1

    return counts[0], counts[1], counts[2]


def stratified_split(
    samples: List[Sample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Sample]]:
    by_class: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        by_class[sample.class_name].append(sample)

    train_split: List[Sample] = []
    val_split: List[Sample] = []
    test_split: List[Sample] = []

    for class_name in sorted(by_class, key=lambda x: x.lower()):
        class_samples = sorted(by_class[class_name], key=lambda s: str(s.path).lower())
        class_rng = random.Random(f"{seed}:{class_name}")
        class_rng.shuffle(class_samples)

        n_train, n_val, n_test = _ratio_counts(
            len(class_samples), (train_ratio, val_ratio, test_ratio)
        )

        train_split.extend(class_samples[:n_train])
        val_split.extend(class_samples[n_train : n_train + n_val])
        test_split.extend(class_samples[n_train + n_val : n_train + n_val + n_test])

    for split_name, split_samples in (
        ("train", train_split),
        ("val", val_split),
        ("test", test_split),
    ):
        split_samples.sort(key=lambda s: str(s.path).lower())
        split_rng = random.Random(f"{seed}:{split_name}")
        split_rng.shuffle(split_samples)

    return {"train": train_split, "val": val_split, "test": test_split}


def inspect_dataset_layout(data_root: Path) -> None:
    split_dirs = discover_split_dirs(data_root)
    if split_dirs:
        print("Detected split-style dataset layout:")
        for split_name in ("train", "val", "test"):
            if split_name in split_dirs:
                class_dirs = discover_class_dirs(split_dirs[split_name])
                print(
                    f"  {split_name}: {split_dirs[split_name]} "
                    f"({len(class_dirs)} class folders)"
                )
    else:
        class_dirs = discover_class_dirs(data_root)
        print("Detected class-folder dataset layout:")
        print(f"  root: {data_root} ({len(class_dirs)} class folders)")


def prepare_splits(
    data_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[Dict[str, int], Dict[str, List[Sample]]]:
    split_dirs = discover_split_dirs(data_root)

    if split_dirs:
        split_class_dirs = {
            split_name: discover_class_dirs(split_path)
            for split_name, split_path in split_dirs.items()
        }
        all_class_names = set()
        for class_dirs in split_class_dirs.values():
            all_class_names.update(class_dirs.keys())
        class_to_idx = build_class_to_idx(all_class_names)

        split_to_samples = {
            split_name: collect_samples_from_class_dirs(class_dirs, class_to_idx)
            for split_name, class_dirs in split_class_dirs.items()
        }

        has_train = "train" in split_to_samples
        has_val = "val" in split_to_samples
        has_test = "test" in split_to_samples

        if has_train and has_test and not has_val:
            tv_total = train_ratio + val_ratio
            if tv_total <= 0:
                raise ValueError("train_ratio + val_ratio must be > 0 when creating val from train.")

            train_rel = train_ratio / tv_total
            val_rel = val_ratio / tv_total
            train_and_val = stratified_split(
                split_to_samples["train"], train_rel, val_rel, 0.0, seed
            )
            split_to_samples["train"] = train_and_val["train"]
            split_to_samples["val"] = train_and_val["val"]

        elif has_train and not has_val and not has_test:
            split_to_samples = stratified_split(
                split_to_samples["train"], train_ratio, val_ratio, test_ratio, seed
            )
        else:
            split_to_samples.setdefault("train", [])
            split_to_samples.setdefault("val", [])
            split_to_samples.setdefault("test", [])

        return class_to_idx, split_to_samples

    class_dirs = discover_class_dirs(data_root)
    if not class_dirs:
        raise FileNotFoundError(
            f"No class folders found in: {data_root}. "
            "Expected folders like <root>/<class_name>/image.png"
        )

    class_to_idx = build_class_to_idx(class_dirs.keys())
    all_samples = collect_samples_from_class_dirs(class_dirs, class_to_idx)
    split_to_samples = stratified_split(all_samples, train_ratio, val_ratio, test_ratio, seed)
    return class_to_idx, split_to_samples


def print_class_distributions(
    class_to_idx: Dict[str, int], split_to_samples: Dict[str, List[Sample]]
) -> None:
    ordered_classes = sorted(class_to_idx, key=lambda x: class_to_idx[x])

    print("\nClass-to-index mapping:")
    print(json.dumps(class_to_idx, indent=2))

    for split_name in ("train", "val", "test"):
        samples = split_to_samples.get(split_name, [])
        counts = class_counts(samples)
        print(f"\n{split_name.upper()} ({len(samples)} images)")
        for class_name in ordered_classes:
            print(f"  {class_name:15s} -> {counts.get(class_name, 0)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load image dataset from class folders, build class mapping, "
            "print class counts, and create reproducible train/val/test splits."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "Dataset root. Supports either:\n"
            "1) <root>/<class_name>/<images>\n"
            "2) <root>/Train|Test|Val/<class_name>/<images>"
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"Path does not exist: {args.data_root}")

    inspect_dataset_layout(args.data_root)
    class_to_idx, split_to_samples = prepare_splits(
        data_root=args.data_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print_class_distributions(class_to_idx, split_to_samples)


if __name__ == "__main__":
    main()
