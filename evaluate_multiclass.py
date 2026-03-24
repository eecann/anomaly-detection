from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)

from dataset_loader import FrameSample, build_sequence_samples, prepare_frame_splits
from train_multiclass import create_image_dataset, create_sequence_dataset

T = TypeVar("T")


@dataclass
class EvalConfig:
    experiment_name: str
    model_key: str
    model_kind: str
    data_root: Path
    class_names: List[str]
    normal_class_name: str
    image_format: str
    image_size: int
    batch_size: int
    num_workers: int
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    sequence_length: int
    eval_stride: int
    max_test_samples: int
    checkpoint_path: Path
    metrics_dir: Path


@dataclass
class EvalResult:
    experiment_name: str
    summary_metrics: Dict[str, float]
    per_class_rows: List[Dict[str, float | int | str]]
    summary_csv_path: Path
    per_class_csv_path: Path
    confusion_csv_path: Path
    confusion_plot_path: Path
    roc_plot_path: Path
    pr_plot_path: Path
    metadata_json_path: Path
    test_sample_count: int


def _limit_samples(items: List[T], max_count: int, seed: int, name: str) -> List[T]:
    if max_count <= 0 or len(items) <= max_count:
        return items
    rng = random.Random(f"{seed}:{name}:subset")
    limited = items.copy()
    rng.shuffle(limited)
    return limited[:max_count]


def _prepare_test_dataset(
    config: EvalConfig,
    frame_splits: Dict[str, List[FrameSample]],
) -> tuple[tf.data.Dataset, np.ndarray]:
    test_frame_samples = frame_splits["test"]
    if config.model_kind == "image":
        test_frame_samples = _limit_samples(
            test_frame_samples,
            config.max_test_samples,
            config.seed,
            f"{config.experiment_name}:test",
        )
        ds = create_image_dataset(
            samples=test_frame_samples,
            image_size=config.image_size,
            image_format=config.image_format,
            batch_size=config.batch_size,
            training=False,
            seed=config.seed,
        )
        y_true = np.array([s.class_idx for s in test_frame_samples], dtype=np.int32)
        return ds, y_true

    test_sequences = build_sequence_samples(
        frame_samples=test_frame_samples,
        sequence_length=config.sequence_length,
        stride=config.eval_stride,
    )
    test_sequences = _limit_samples(
        test_sequences,
        config.max_test_samples,
        config.seed,
        f"{config.experiment_name}:test_seq",
    )
    ds = create_sequence_dataset(
        samples=test_sequences,
        image_size=config.image_size,
        image_format=config.image_format,
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
        training=False,
        seed=config.seed,
    )
    y_true = np.array([s.class_idx for s in test_sequences], dtype=np.int32)
    return ds, y_true


def _predict_probabilities(model: tf.keras.Model, dataset: tf.data.Dataset) -> np.ndarray:
    probs = model.predict(dataset, verbose=0)
    return probs.astype(np.float64)


def _compute_summary_and_per_class(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> tuple[Dict[str, float], List[Dict[str, float | int | str]], np.ndarray, Dict[str, object]]:
    y_pred = np.argmax(y_prob, axis=1)
    labels = np.arange(len(class_names), dtype=np.int32)

    acc = float(accuracy_score(y_true, y_pred))
    per_precision, per_recall, per_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    conf = confusion_matrix(y_true, y_pred, labels=labels)

    curve_rows: List[Dict[str, object]] = []
    roc_auc_values: List[float] = []
    macro_fpr_parts: List[np.ndarray] = []
    macro_tpr_parts: List[np.ndarray] = []

    for idx, class_name in enumerate(class_names):
        y_true_bin = (y_true == idx).astype(np.int32)
        y_score = y_prob[:, idx]
        positives = int(y_true_bin.sum())
        negatives = int(len(y_true_bin) - positives)

        if positives > 0 and negatives > 0:
            fpr, tpr, _ = roc_curve(y_true_bin, y_score)
            roc_auc = float(auc(fpr, tpr))
            roc_auc_values.append(roc_auc)
            macro_fpr_parts.append(fpr)
            macro_tpr_parts.append(tpr)
        else:
            fpr = np.array([0.0, 1.0], dtype=np.float64)
            tpr = np.array([0.0, 1.0], dtype=np.float64)
            roc_auc = float("nan")

        if positives > 0:
            pr_precision, pr_recall, _ = precision_recall_curve(y_true_bin, y_score)
            pr_auc = float(auc(pr_recall, pr_precision))
        else:
            pr_precision = np.array([1.0, 0.0], dtype=np.float64)
            pr_recall = np.array([0.0, 1.0], dtype=np.float64)
            pr_auc = float("nan")

        curve_rows.append(
            {
                "class_idx": idx,
                "class_name": class_name,
                "roc_fpr": fpr,
                "roc_tpr": tpr,
                "roc_auc": roc_auc,
                "pr_precision": pr_precision,
                "pr_recall": pr_recall,
                "pr_auc": pr_auc,
            }
        )

    if macro_fpr_parts:
        all_fpr = np.unique(np.concatenate(macro_fpr_parts))
        mean_tpr = np.zeros_like(all_fpr)
        for fpr, tpr in zip(macro_fpr_parts, macro_tpr_parts):
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr /= len(macro_fpr_parts)
        macro_roc_curve_auc = float(auc(all_fpr, mean_tpr))
    else:
        all_fpr = np.array([0.0, 1.0], dtype=np.float64)
        mean_tpr = np.array([0.0, 1.0], dtype=np.float64)
        macro_roc_curve_auc = float("nan")

    macro_roc_auc = float(np.mean(roc_auc_values)) if roc_auc_values else float("nan")

    per_class_rows: List[Dict[str, float | int | str]] = []
    for idx, class_name in enumerate(class_names):
        curve = curve_rows[idx]
        per_class_rows.append(
            {
                "class_idx": idx,
                "class_name": class_name,
                "precision": float(per_precision[idx]),
                "recall": float(per_recall[idx]),
                "f1_score": float(per_f1[idx]),
                "support": int(support[idx]),
                "roc_auc": float(curve["roc_auc"]),
                "pr_auc": float(curve["pr_auc"]),
            }
        )

    summary = {
        "accuracy": acc,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "macro_roc_auc": macro_roc_auc,
        "macro_roc_curve_auc": macro_roc_curve_auc,
    }
    curves = {
        "rows": curve_rows,
        "macro_fpr": all_fpr,
        "macro_tpr": mean_tpr,
        "macro_roc_auc": macro_roc_auc,
    }
    return summary, per_class_rows, conf, curves


def _save_summary_csv(summary: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in summary.items():
            writer.writerow([k, f"{float(v):.8f}"])


def _save_per_class_csv(rows: List[Dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["class_idx", "class_name", "precision", "recall", "f1_score", "support", "roc_auc", "pr_auc"]
        )
        for row in rows:
            writer.writerow(
                [
                    row["class_idx"],
                    row["class_name"],
                    f"{float(row['precision']):.8f}",
                    f"{float(row['recall']):.8f}",
                    f"{float(row['f1_score']):.8f}",
                    int(row["support"]),
                    f"{float(row['roc_auc']):.8f}",
                    f"{float(row['pr_auc']):.8f}",
                ]
            )


def _save_confusion(confusion: np.ndarray, class_names: List[str], csv_path: Path, plot_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + class_names)
        for idx, class_name in enumerate(class_names):
            writer.writerow([class_name] + confusion[idx].tolist())

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_roc_pr_plots(curves: Dict[str, object], roc_path: Path, pr_path: Path) -> None:
    curve_rows = curves["rows"]
    macro_fpr = curves["macro_fpr"]
    macro_tpr = curves["macro_tpr"]
    macro_auc = curves["macro_roc_auc"]

    roc_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 10))
    cmap = plt.get_cmap("tab20")
    for idx, row in enumerate(curve_rows):
        ax.plot(
            row["roc_fpr"],
            row["roc_tpr"],
            lw=1.8,
            color=cmap(idx % 20),
            label=f"{row['class_name']} (AUC={float(row['roc_auc']):.3f})",
        )
    ax.plot(macro_fpr, macro_tpr, color="black", lw=3.0, label=f"Macro-average (AUC={macro_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.4, label="Chance")
    ax.set_title("Multiclass ROC Curves (One-vs-Rest)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pr_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 10))
    for idx, row in enumerate(curve_rows):
        ax.plot(
            row["pr_recall"],
            row["pr_precision"],
            lw=1.8,
            color=cmap(idx % 20),
            label=f"{row['class_name']} (PR-AUC={float(row['pr_auc']):.3f})",
        )
    ax.set_title("Multiclass Precision-Recall Curves (One-vs-Rest)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_evaluation(
    config: EvalConfig,
    precomputed_frame_splits: tuple[Dict[str, int], Dict[str, List[FrameSample]]] | None = None,
) -> EvalResult:
    print(f"\n[Test Evaluation] {config.experiment_name} ({config.model_key})")
    model = tf.keras.models.load_model(config.checkpoint_path, compile=False)

    if precomputed_frame_splits is None:
        _, frame_splits = prepare_frame_splits(
            data_root=config.data_root,
            class_names=config.class_names,
            seed=config.seed,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            image_format=config.image_format,
        )
    else:
        _, frame_splits = precomputed_frame_splits

    test_dataset, y_true = _prepare_test_dataset(config, frame_splits=frame_splits)
    print(f"Test sample sayisi: {len(y_true)}")
    y_prob = _predict_probabilities(model, test_dataset)

    summary, per_class_rows, conf, curves = _compute_summary_and_per_class(
        y_true=y_true,
        y_prob=y_prob,
        class_names=config.class_names,
    )

    summary_csv = config.metrics_dir / "summary_metrics.csv"
    per_class_csv = config.metrics_dir / "per_class_metrics.csv"
    conf_csv = config.metrics_dir / "confusion_matrix.csv"
    conf_plot = config.metrics_dir / "confusion_matrix.png"
    roc_plot = config.metrics_dir / "roc_ovr_curves.png"
    pr_plot = config.metrics_dir / "pr_ovr_curves.png"
    metadata_json = config.metrics_dir / "evaluation_metadata.json"

    _save_summary_csv(summary, summary_csv)
    _save_per_class_csv(per_class_rows, per_class_csv)
    _save_confusion(conf, config.class_names, conf_csv, conf_plot)
    _save_roc_pr_plots(curves, roc_plot, pr_plot)

    metadata_json.write_text(
        json.dumps(
            {
                "experiment_name": config.experiment_name,
                "model_key": config.model_key,
                "model_kind": config.model_kind,
                "checkpoint_path": str(config.checkpoint_path),
                "test_sample_count": int(len(y_true)),
                "summary_metrics": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Accuracy: {summary['accuracy']:.4f} | Macro F1: {summary['macro_f1']:.4f} | Macro ROC-AUC: {summary['macro_roc_auc']:.4f}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Per-class CSV: {per_class_csv}")
    print(f"ROC plot: {roc_plot}")

    return EvalResult(
        experiment_name=config.experiment_name,
        summary_metrics=summary,
        per_class_rows=per_class_rows,
        summary_csv_path=summary_csv,
        per_class_csv_path=per_class_csv,
        confusion_csv_path=conf_csv,
        confusion_plot_path=conf_plot,
        roc_plot_path=roc_plot,
        pr_plot_path=pr_plot,
        metadata_json_path=metadata_json,
        test_sample_count=int(len(y_true)),
    )
