from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SPLIT_ALIASES = {
    "train": {"train"},
    "val": {"val", "valid", "validation", "dev"},
    "test": {"test"},
}


@dataclass(frozen=True)
class FrameSample:
    path: Path
    class_name: str
    class_idx: int
    video_id: str
    frame_idx: int


@dataclass(frozen=True)
class SequenceSample:
    frame_paths: Tuple[Path, ...]
    class_name: str
    class_idx: int
    video_key: str
    start_frame: int
    end_frame: int


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


def build_class_to_idx(class_names: Iterable[str]) -> Dict[str, int]:
    ordered: List[str] = []
    seen = set()
    for class_name in class_names:
        if class_name in seen:
            continue
        seen.add(class_name)
        ordered.append(class_name)
    return {name: idx for idx, name in enumerate(ordered)}


def parse_video_and_frame(stem: str) -> Tuple[str, int]:
    match = re.match(r"^(?P<video_id>.+)_(?P<frame_idx>\d+)$", stem)
    if not match:
        raise ValueError(
            f"Dosya ismi video/frame formatina uymuyor: {stem}. "
            "Beklenen kalip: <video_id>_<frame_index>.png"
        )
    return match.group("video_id"), int(match.group("frame_idx"))


def _collect_frame_samples_from_class_dir(
    class_dir: Path,
    class_name: str,
    class_idx: int,
    image_extensions: Tuple[str, ...],
) -> List[FrameSample]:
    records: List[FrameSample] = []
    for path in sorted(class_dir.rglob("*"), key=lambda p: str(p).lower()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in image_extensions:
            continue
        video_id, frame_idx = parse_video_and_frame(path.stem)
        records.append(
            FrameSample(
                path=path,
                class_name=class_name,
                class_idx=class_idx,
                video_id=video_id,
                frame_idx=frame_idx,
            )
        )
    return records


def _collect_from_split_dir(
    split_dir: Path,
    class_to_idx: Dict[str, int],
    image_extensions: Tuple[str, ...],
) -> List[FrameSample]:
    samples: List[FrameSample] = []
    for class_dir in _sorted_subdirs(split_dir):
        class_name = class_dir.name
        if class_name not in class_to_idx:
            continue
        class_idx = class_to_idx[class_name]
        samples.extend(
            _collect_frame_samples_from_class_dir(
                class_dir=class_dir,
                class_name=class_name,
                class_idx=class_idx,
                image_extensions=image_extensions,
            )
        )
    return samples


def _video_key(sample: FrameSample) -> str:
    return f"{sample.class_name}/{sample.video_id}"


def _split_video_keys_by_class(
    samples: List[FrameSample],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, set[str]]:
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("Split oranlari negatif olamaz.")

    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Split oranlarinin toplami sifirdan buyuk olmali.")

    nr_train = train_ratio / total
    nr_val = val_ratio / total

    class_to_video_keys: Dict[str, List[str]] = defaultdict(list)
    for sample in samples:
        class_to_video_keys[sample.class_name].append(_video_key(sample))

    split_keys = {"train": set(), "val": set(), "test": set()}

    for class_name, keys in class_to_video_keys.items():
        unique_keys = sorted(set(keys))
        rng = random.Random(f"{seed}:{class_name}:video_split")
        rng.shuffle(unique_keys)

        n = len(unique_keys)
        n_train = int(n * nr_train)
        n_val = int(n * nr_val)
        n_test = n - n_train - n_val

        if n >= 3:
            if n_train == 0:
                n_train = 1
            if n_val == 0:
                n_val = 1
            n_test = max(1, n - n_train - n_val)
            while n_train + n_val + n_test > n:
                if n_train >= n_val and n_train > 1:
                    n_train -= 1
                elif n_val > 1:
                    n_val -= 1
                else:
                    n_test -= 1

        train_keys = unique_keys[:n_train]
        val_keys = unique_keys[n_train : n_train + n_val]
        test_keys = unique_keys[n_train + n_val :]
        if len(test_keys) != n_test:
            test_keys = unique_keys[n_train + n_val : n_train + n_val + n_test]

        split_keys["train"].update(train_keys)
        split_keys["val"].update(val_keys)
        split_keys["test"].update(test_keys)

    return split_keys


def _samples_by_video_keys(samples: List[FrameSample], selected_keys: set[str]) -> List[FrameSample]:
    return [s for s in samples if _video_key(s) in selected_keys]


def _split_train_into_train_val_by_video(
    train_samples: List[FrameSample],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[FrameSample], List[FrameSample]]:
    if train_ratio < 0 or val_ratio < 0:
        raise ValueError("train_ratio ve val_ratio negatif olamaz.")
    if train_ratio + val_ratio <= 0:
        raise ValueError("train_ratio + val_ratio sifirdan buyuk olmali.")

    by_class_video_keys: Dict[str, List[str]] = defaultdict(list)
    for sample in train_samples:
        by_class_video_keys[sample.class_name].append(_video_key(sample))

    train_keys: set[str] = set()
    val_keys: set[str] = set()
    val_share = val_ratio / (train_ratio + val_ratio)

    for class_name, keys in by_class_video_keys.items():
        unique_keys = sorted(set(keys))
        rng = random.Random(f"{seed}:{class_name}:train_val_video_split")
        rng.shuffle(unique_keys)
        n_val = int(len(unique_keys) * val_share)
        if len(unique_keys) >= 2 and n_val == 0:
            n_val = 1
        val_set = set(unique_keys[:n_val])
        train_set = set(unique_keys[n_val:])
        if not train_set and val_set:
            one_key = next(iter(val_set))
            val_set.remove(one_key)
            train_set.add(one_key)
        train_keys.update(train_set)
        val_keys.update(val_set)

    return _samples_by_video_keys(train_samples, train_keys), _samples_by_video_keys(train_samples, val_keys)


def prepare_frame_splits(
    data_root: Path,
    class_names: List[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    image_format: str = "png",
) -> Tuple[Dict[str, int], Dict[str, List[FrameSample]]]:
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root bulunamadi: {data_root}")

    ext = image_format.lower().strip().lstrip(".")
    image_extensions = (f".{ext}",)
    class_to_idx = build_class_to_idx(class_names)
    split_dirs = discover_split_dirs(data_root)

    if split_dirs:
        split_to_samples: Dict[str, List[FrameSample]] = {}
        for split_name, split_dir in split_dirs.items():
            split_to_samples[split_name] = _collect_from_split_dir(
                split_dir=split_dir,
                class_to_idx=class_to_idx,
                image_extensions=image_extensions,
            )

        has_train = "train" in split_to_samples and bool(split_to_samples["train"])
        has_val = "val" in split_to_samples and bool(split_to_samples["val"])
        has_test = "test" in split_to_samples and bool(split_to_samples["test"])

        if has_train and has_test and not has_val:
            new_train, new_val = _split_train_into_train_val_by_video(
                train_samples=split_to_samples["train"],
                seed=seed,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
            )
            split_to_samples["train"] = new_train
            split_to_samples["val"] = new_val

        elif has_train and not has_val and not has_test:
            split_video_keys = _split_video_keys_by_class(
                samples=split_to_samples["train"],
                seed=seed,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )
            all_train = split_to_samples["train"]
            split_to_samples["train"] = _samples_by_video_keys(all_train, split_video_keys["train"])
            split_to_samples["val"] = _samples_by_video_keys(all_train, split_video_keys["val"])
            split_to_samples["test"] = _samples_by_video_keys(all_train, split_video_keys["test"])
        else:
            split_to_samples.setdefault("train", [])
            split_to_samples.setdefault("val", [])
            split_to_samples.setdefault("test", [])

        for split_name in ("train", "val", "test"):
            split_to_samples[split_name].sort(key=lambda s: str(s.path).lower())

        return class_to_idx, split_to_samples

    all_samples: List[FrameSample] = []
    for class_dir in _sorted_subdirs(data_root):
        class_name = class_dir.name
        if class_name not in class_to_idx:
            continue
        all_samples.extend(
            _collect_frame_samples_from_class_dir(
                class_dir=class_dir,
                class_name=class_name,
                class_idx=class_to_idx[class_name],
                image_extensions=image_extensions,
            )
        )

    split_video_keys = _split_video_keys_by_class(
        samples=all_samples,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    split_to_samples = {
        "train": _samples_by_video_keys(all_samples, split_video_keys["train"]),
        "val": _samples_by_video_keys(all_samples, split_video_keys["val"]),
        "test": _samples_by_video_keys(all_samples, split_video_keys["test"]),
    }
    for split_name in ("train", "val", "test"):
        split_to_samples[split_name].sort(key=lambda s: str(s.path).lower())
    return class_to_idx, split_to_samples


def build_sequence_samples(
    frame_samples: List[FrameSample],
    sequence_length: int,
    stride: int,
) -> List[SequenceSample]:
    if sequence_length <= 0:
        raise ValueError("sequence_length sifirdan buyuk olmali.")
    if stride <= 0:
        raise ValueError("stride sifirdan buyuk olmali.")

    by_video: Dict[str, List[FrameSample]] = defaultdict(list)
    for sample in frame_samples:
        by_video[_video_key(sample)].append(sample)

    sequence_samples: List[SequenceSample] = []
    for video_key, samples in by_video.items():
        ordered = sorted(samples, key=lambda s: s.frame_idx)
        n = len(ordered)
        if n < sequence_length:
            continue

        class_name = ordered[0].class_name
        class_idx = ordered[0].class_idx
        for start in range(0, n - sequence_length + 1, stride):
            chunk = ordered[start : start + sequence_length]
            sequence_samples.append(
                SequenceSample(
                    frame_paths=tuple(s.path for s in chunk),
                    class_name=class_name,
                    class_idx=class_idx,
                    video_key=video_key,
                    start_frame=chunk[0].frame_idx,
                    end_frame=chunk[-1].frame_idx,
                )
            )

    sequence_samples.sort(key=lambda s: (s.video_key, s.start_frame))
    return sequence_samples


def class_counts(samples: List[FrameSample] | List[SequenceSample]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for sample in samples:
        counts[sample.class_name] += 1
    return dict(sorted(counts.items(), key=lambda x: x[0].lower()))
