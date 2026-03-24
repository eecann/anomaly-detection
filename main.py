from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

from dataset_loader import prepare_frame_splits
from evaluate_multiclass import EvalConfig, run_evaluation
from train_multiclass import TrainConfig, run_training_experiment


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / "configs"
DATASET_CONFIG_PATH = CONFIG_DIR / "dataset.yaml"
EXPERIMENTS_CONFIG_PATH = CONFIG_DIR / "experiments.yaml"


IMAGE_MODELS = {"mobilenetv2_image", "efficientnetb0_image"}
TEMPORAL_MODELS = {"cnn_lstm_temporal", "convlstm_temporal"}


@dataclass
class DatasetSettings:
    data_root: Path
    classes: List[str]
    normal_class_name: str
    image_size: int
    image_format: str
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML bulunamadi: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML dict olmali: {path}")
    return data


def resolve_path(raw_path: str, base_dir: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def load_dataset_settings(path: Path) -> DatasetSettings:
    cfg = load_yaml(path).get("dataset", {})
    if not cfg:
        raise ValueError("dataset.yaml icindeki 'dataset' bolumu bos.")

    split_cfg = cfg.get("split", {})
    classes = list(cfg.get("classes", []))
    if not classes:
        raise ValueError("dataset.classes bos olamaz.")
    normal_class_name = str(cfg.get("normal_class_name", "NormalVideos"))
    if normal_class_name not in classes:
        raise ValueError("normal_class_name classes icinde bulunmuyor.")

    return DatasetSettings(
        data_root=resolve_path(str(cfg["data_root"]), path.parent),
        classes=classes,
        normal_class_name=normal_class_name,
        image_size=int(cfg.get("image_size", 64)),
        image_format=str(cfg.get("image_format", "png")).lower(),
        seed=int(cfg.get("seed", 42)),
        train_ratio=float(split_cfg.get("train_ratio", 0.8)),
        val_ratio=float(split_cfg.get("val_ratio", 0.1)),
        test_ratio=float(split_cfg.get("test_ratio", 0.1)),
    )


def _model_kind(model_key: str) -> str:
    if model_key in IMAGE_MODELS:
        return "image"
    if model_key in TEMPORAL_MODELS:
        return "temporal"
    raise ValueError(f"Bilinmeyen model_key: {model_key}")


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_csv_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _merge_summary_rows(existing: List[Dict[str, str]], new_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for row in existing:
        merged[str(row["experiment"])] = dict(row)
    for row in new_rows:
        merged[str(row["experiment"])] = dict(row)
    return [merged[k] for k in sorted(merged.keys())]


def _merge_per_class_rows(existing: List[Dict[str, str]], new_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in existing:
        key = (str(row["experiment"]), str(row["class_name"]))
        merged[key] = dict(row)
    for row in new_rows:
        key = (str(row["experiment"]), str(row["class_name"]))
        merged[key] = dict(row)
    return [merged[k] for k in sorted(merged.keys(), key=lambda x: (x[0], x[1]))]


def _save_markdown_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("# Experiment Summary\n\nNo experiment rows.\n", encoding="utf-8")
        return

    headers = list(rows[0].keys())
    lines = [
        "# Experiment Summary",
        "",
        "|" + "|".join(headers) + "|",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("|" + "|".join(str(row[h]) for h in headers) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_curve_methodology(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "# Multiclass ROC/PR Methodology\n\n"
            "One-vs-rest stratejisi kullanilir. Her sinif icin y_true binary hale getirilir "
            "(sinif=1, digerleri=0) ve modelin ilgili sinif olasiliklari kullanilir.\n"
            "ROC-AUC: TPR-FPR alani, PR-AUC: Precision-Recall alani.\n"
            "Macro ROC-AUC: sinif ROC-AUC degerlerinin ortalamasi.\n"
        ),
        encoding="utf-8",
    )


def run_all_experiments() -> None:
    dataset = load_dataset_settings(DATASET_CONFIG_PATH)
    exp_cfg = load_yaml(EXPERIMENTS_CONFIG_PATH)

    run_cfg = exp_cfg.get("run", {})
    training_defaults = exp_cfg.get("training_defaults", {})
    evaluation_defaults = exp_cfg.get("evaluation_defaults", {})
    experiments = [e for e in exp_cfg.get("experiments", []) if e.get("enabled", True)]
    if not experiments:
        raise ValueError("experiments.yaml icinde enabled=true en az bir deney olmali.")

    run_name = str(run_cfg.get("name", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    output_root = resolve_path(str(run_cfg.get("output_root", "../outputs")), EXPERIMENTS_CONFIG_PATH.parent)
    run_dir = output_root / run_name
    experiments_dir = run_dir / "experiments"
    reports_dir = run_dir / "reports"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    selection_metric = str(run_cfg.get("selection_metric", "test_macro_f1"))

    print("\n" + "=" * 100)
    print("Dataset splitleri bir kez hazirlaniyor...")
    print("=" * 100)
    precomputed_frame_splits = prepare_frame_splits(
        data_root=dataset.data_root,
        class_names=dataset.classes,
        seed=dataset.seed,
        train_ratio=dataset.train_ratio,
        val_ratio=dataset.val_ratio,
        test_ratio=dataset.test_ratio,
        image_format=dataset.image_format,
    )

    all_summary_rows: List[Dict[str, Any]] = []
    all_per_class_rows: List[Dict[str, Any]] = []

    for experiment in experiments:
        exp_name = str(experiment["name"])
        model_key = str(experiment["model_key"]).strip().lower()
        model_kind = _model_kind(model_key)

        exp_dir = experiments_dir / exp_name
        train_dir = exp_dir / "train"
        eval_dir = exp_dir / "eval"
        train_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        image_size = int(experiment.get("image_size", training_defaults.get("image_size", dataset.image_size)))
        batch_size = int(experiment.get("batch_size", training_defaults.get("batch_size", 32)))
        epochs = int(experiment.get("epochs", training_defaults.get("epochs", 10)))
        learning_rate = float(experiment.get("learning_rate", training_defaults.get("learning_rate", 1e-4)))
        weight_decay = float(experiment.get("weight_decay", training_defaults.get("weight_decay", 1e-4)))
        num_workers = int(experiment.get("num_workers", training_defaults.get("num_workers", 2)))
        max_train_samples = int(experiment.get("max_train_samples", training_defaults.get("max_train_samples", 0)))
        max_val_samples = int(experiment.get("max_val_samples", training_defaults.get("max_val_samples", 0)))
        sequence_length = int(experiment.get("sequence_length", training_defaults.get("sequence_length", 20)))
        train_stride = int(experiment.get("train_stride", training_defaults.get("train_stride", 5)))
        eval_stride = int(experiment.get("eval_stride", training_defaults.get("eval_stride", 10)))
        seed = int(experiment.get("seed", dataset.seed))

        train_config = TrainConfig(
            experiment_name=exp_name,
            model_key=model_key,
            model_kind=model_kind,
            data_root=dataset.data_root,
            class_names=dataset.classes,
            normal_class_name=dataset.normal_class_name,
            image_format=dataset.image_format,
            image_size=image_size,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_workers=num_workers,
            seed=seed,
            train_ratio=dataset.train_ratio,
            val_ratio=dataset.val_ratio,
            test_ratio=dataset.test_ratio,
            sequence_length=sequence_length,
            train_stride=train_stride,
            eval_stride=eval_stride,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            checkpoint_path=train_dir / "best_model.keras",
            history_csv_path=train_dir / "history.csv",
            history_json_path=train_dir / "history.json",
            metadata_json_path=train_dir / "model_metadata.json",
        )

        print("\n" + "=" * 100)
        print(f"[TRAIN] {exp_name} ({model_key})")
        print("=" * 100)
        train_result = run_training_experiment(
            train_config,
            precomputed_frame_splits=precomputed_frame_splits,
        )

        eval_batch_size = int(experiment.get("eval_batch_size", evaluation_defaults.get("batch_size", 64)))
        eval_num_workers = int(experiment.get("eval_num_workers", evaluation_defaults.get("num_workers", 2)))
        eval_max_test_samples = int(experiment.get("max_test_samples", evaluation_defaults.get("max_test_samples", 0)))

        eval_config = EvalConfig(
            experiment_name=exp_name,
            model_key=model_key,
            model_kind=model_kind,
            data_root=dataset.data_root,
            class_names=dataset.classes,
            normal_class_name=dataset.normal_class_name,
            image_format=dataset.image_format,
            image_size=image_size,
            batch_size=eval_batch_size,
            num_workers=eval_num_workers,
            seed=seed,
            train_ratio=dataset.train_ratio,
            val_ratio=dataset.val_ratio,
            test_ratio=dataset.test_ratio,
            sequence_length=sequence_length,
            eval_stride=eval_stride,
            max_test_samples=eval_max_test_samples,
            checkpoint_path=train_result.checkpoint_path,
            metrics_dir=eval_dir,
        )

        print("\n" + "=" * 100)
        print(f"[EVAL] {exp_name} ({model_key})")
        print("=" * 100)
        eval_result = run_evaluation(
            eval_config,
            precomputed_frame_splits=precomputed_frame_splits,
        )

        summary_row = {
            "experiment": exp_name,
            "model_key": model_key,
            "model_kind": model_kind,
            "epochs": epochs,
            "batch_size": batch_size,
            "image_size": image_size,
            "sequence_length": sequence_length if model_kind == "temporal" else "",
            "best_epoch": train_result.best_epoch,
            "best_val_macro_f1": f"{train_result.best_val_macro_f1:.6f}",
            "test_accuracy": f"{eval_result.summary_metrics['accuracy']:.6f}",
            "test_macro_f1": f"{eval_result.summary_metrics['macro_f1']:.6f}",
            "test_weighted_f1": f"{eval_result.summary_metrics['weighted_f1']:.6f}",
            "test_macro_roc_auc": f"{eval_result.summary_metrics['macro_roc_auc']:.6f}",
            "test_macro_precision": f"{eval_result.summary_metrics['macro_precision']:.6f}",
            "test_macro_recall": f"{eval_result.summary_metrics['macro_recall']:.6f}",
            "test_sample_count": int(eval_result.test_sample_count),
            "model_path": str(train_result.checkpoint_path),
            "model_metadata_path": str(train_result.metadata_json_path),
        }
        all_summary_rows.append(summary_row)

        for row in eval_result.per_class_rows:
            all_per_class_rows.append(
                {
                    "experiment": exp_name,
                    "model_key": model_key,
                    "model_kind": model_kind,
                    "class_idx": row["class_idx"],
                    "class_name": row["class_name"],
                    "precision": f"{float(row['precision']):.6f}",
                    "recall": f"{float(row['recall']):.6f}",
                    "f1_score": f"{float(row['f1_score']):.6f}",
                    "support": int(row["support"]),
                    "roc_auc": f"{float(row['roc_auc']):.6f}",
                    "pr_auc": f"{float(row['pr_auc']):.6f}",
                }
            )

    summary_csv = reports_dir / "experiment_summary.csv"
    per_class_csv = reports_dir / "per_class_comparison.csv"
    summary_md = reports_dir / "experiment_summary.md"
    methodology_md = reports_dir / "curve_methodology.md"
    best_model_json = reports_dir / "best_model.json"

    existing_summary = _read_csv_rows(summary_csv)
    existing_per_class = _read_csv_rows(per_class_csv)
    merged_summary = _merge_summary_rows(existing_summary, all_summary_rows)
    merged_per_class = _merge_per_class_rows(existing_per_class, all_per_class_rows)

    _write_csv_rows(summary_csv, merged_summary)
    _write_csv_rows(per_class_csv, merged_per_class)
    _save_markdown_summary(summary_md, merged_summary)
    _save_curve_methodology(methodology_md)

    candidate_rows = [r for r in merged_summary if selection_metric in r]
    if not candidate_rows:
        raise ValueError(f"selection_metric bulunamadi: {selection_metric}")
    best_row = max(candidate_rows, key=lambda r: float(r[selection_metric]))

    best_payload = {
        "selection_metric": selection_metric,
        "selected_experiment": best_row["experiment"],
        "model_key": best_row["model_key"],
        "model_kind": best_row["model_kind"],
        "model_path": best_row["model_path"],
        "model_metadata_path": best_row["model_metadata_path"],
        "score": float(best_row[selection_metric]),
        "normal_class_name": dataset.normal_class_name,
        "class_names": dataset.classes,
    }
    best_model_json.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

    run_snapshot = {
        "dataset": {
            "data_root": str(dataset.data_root),
            "classes": dataset.classes,
            "normal_class_name": dataset.normal_class_name,
            "image_size": dataset.image_size,
            "image_format": dataset.image_format,
            "seed": dataset.seed,
            "split": {
                "train_ratio": dataset.train_ratio,
                "val_ratio": dataset.val_ratio,
                "test_ratio": dataset.test_ratio,
            },
        },
        "experiments_config": exp_cfg,
    }
    (run_dir / "config_snapshot.json").write_text(json.dumps(run_snapshot, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    print("Tum deneyler tamamlandi.")
    print(f"Summary CSV        : {summary_csv}")
    print(f"Per-class CSV      : {per_class_csv}")
    print(f"Summary Markdown   : {summary_md}")
    print(f"Best model         : {best_model_json}")
    print("=" * 100)


if __name__ == "__main__":
    run_all_experiments()
