from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, TypeVar

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from dataset_loader import (
    FrameSample,
    SequenceSample,
    build_sequence_samples,
    class_counts,
    prepare_frame_splits,
)


AUTOTUNE = tf.data.AUTOTUNE
ModelKind = Literal["image", "temporal"]
T = TypeVar("T")


@dataclass
class TrainConfig:
    experiment_name: str
    model_key: str
    model_kind: ModelKind
    data_root: Path
    class_names: List[str]
    normal_class_name: str
    image_format: str
    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    sequence_length: int
    train_stride: int
    eval_stride: int
    max_train_samples: int
    max_val_samples: int
    checkpoint_path: Path
    history_csv_path: Path
    history_json_path: Path
    metadata_json_path: Path


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float
    val_macro_f1: float


@dataclass
class TrainResult:
    experiment_name: str
    model_key: str
    model_kind: ModelKind
    class_names: List[str]
    class_to_idx: Dict[str, int]
    normal_class_idx: int
    best_epoch: int
    best_val_macro_f1: float
    train_sample_count: int
    val_sample_count: int
    history_rows: List[EpochMetrics]
    checkpoint_path: Path
    history_csv_path: Path
    history_json_path: Path
    metadata_json_path: Path


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _limit_samples(items: List[T], max_count: int, seed: int, name: str) -> List[T]:
    if max_count <= 0 or len(items) <= max_count:
        return items
    rng = random.Random(f"{seed}:{name}:subset")
    limited = items.copy()
    rng.shuffle(limited)
    return limited[:max_count]


def _decode_image(path: tf.Tensor, image_size: int, image_format: str) -> tf.Tensor:
    raw = tf.io.read_file(path)
    fmt = image_format.lower()
    if fmt == "png":
        image = tf.io.decode_png(raw, channels=3)
    elif fmt in {"jpg", "jpeg"}:
        image = tf.io.decode_jpeg(raw, channels=3)
    else:
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, [image_size, image_size], method="bilinear")
    return tf.cast(image, tf.float32)


def _augment_image(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.random_flip_left_right(image)
    return image


def create_image_dataset(
    samples: List[FrameSample],
    image_size: int,
    image_format: str,
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    paths = np.array([str(s.path) for s in samples], dtype=np.str_)
    labels = np.array([s.class_idx for s in samples], dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(buffer_size=min(len(samples), 100_000), seed=seed, reshuffle_each_iteration=True)

    def _map_fn(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = _decode_image(path, image_size=image_size, image_format=image_format)
        if training:
            image = _augment_image(image)
        return image, label

    return ds.map(_map_fn, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)


def create_sequence_dataset(
    samples: List[SequenceSample],
    image_size: int,
    image_format: str,
    sequence_length: int,
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    paths = np.array([[str(p) for p in s.frame_paths] for s in samples], dtype=np.str_)
    labels = np.array([s.class_idx for s in samples], dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(buffer_size=min(len(samples), 50_000), seed=seed, reshuffle_each_iteration=True)

    def _map_fn(sequence_paths: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        def _load_one(path: tf.Tensor) -> tf.Tensor:
            return _decode_image(path, image_size=image_size, image_format=image_format)

        frames = tf.map_fn(
            _load_one,
            sequence_paths,
            fn_output_signature=tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
        )
        frames = tf.ensure_shape(frames, (sequence_length, image_size, image_size, 3))
        return frames, label

    return ds.map(_map_fn, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)


def build_model(model_key: str, image_size: int, sequence_length: int, num_classes: int) -> tf.keras.Model:
    key = model_key.lower().strip()

    if key == "mobilenetv2_image":
        inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="input_image")
        x = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="preprocess")(inputs)
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(image_size, image_size, 3),
        )
        x = backbone(x)
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
        x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_image")
        return model

    if key == "efficientnetb0_image":
        inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="input_image")
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(image_size, image_size, 3),
        )
        x = backbone(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
        x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="efficientnetb0_image")
        return model

    if key == "cnn_lstm_temporal":
        inputs = tf.keras.Input(shape=(sequence_length, image_size, image_size, 3), name="input_sequence")
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Rescaling(1.0 / 255.0), name="rescale")(inputs)
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")
        )(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2)))(x)
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")
        )(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2)))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        x = tf.keras.layers.LSTM(128, return_sequences=False)(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_lstm_temporal")

    if key == "convlstm_temporal":
        inputs = tf.keras.Input(shape=(sequence_length, image_size, image_size, 3), name="input_sequence")
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Rescaling(1.0 / 255.0), name="rescale")(inputs)
        x = tf.keras.layers.ConvLSTM2D(
            filters=16,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="tanh",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
        x = tf.keras.layers.ConvLSTM2D(
            filters=24,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=False,
            activation="tanh",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(96, activation="relu")(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="convlstm_temporal")

    raise ValueError(
        "Bilinmeyen model_key. Beklenen: "
        "mobilenetv2_image, efficientnetb0_image, cnn_lstm_temporal, convlstm_temporal"
    )


class MacroF1CheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_dataset: tf.data.Dataset, save_path: Path, verbose: bool = True) -> None:
        super().__init__()
        self.val_dataset = val_dataset
        self.save_path = save_path
        self.verbose = verbose
        self.best_macro_f1 = -1.0
        self.best_epoch = 0

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        y_true: List[int] = []
        y_pred: List[int] = []
        for features, labels in self.val_dataset:
            probs = self.model.predict(features, verbose=0)
            preds = np.argmax(probs, axis=1)
            y_true.extend(labels.numpy().astype(int).tolist())
            y_pred.extend(preds.astype(int).tolist())

        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        logs = logs if logs is not None else {}
        logs["val_macro_f1"] = macro_f1

        if macro_f1 > self.best_macro_f1:
            self.best_macro_f1 = macro_f1
            self.best_epoch = epoch + 1
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.save_path)
            if self.verbose:
                print(f"  -> Yeni en iyi model kaydedildi (val_macro_f1={macro_f1:.4f})")


class EpochLogCallback(tf.keras.callbacks.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.rows: List[EpochMetrics] = []

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        row = EpochMetrics(
            epoch=epoch + 1,
            train_loss=float(logs.get("loss", 0.0)),
            val_loss=float(logs.get("val_loss", 0.0)),
            val_acc=float(logs.get("val_accuracy", 0.0)),
            val_macro_f1=float(logs.get("val_macro_f1", 0.0)),
        )
        self.rows.append(row)
        print(
            f"Epoch {row.epoch:03d} | "
            f"train_loss={row.train_loss:.4f} | "
            f"val_loss={row.val_loss:.4f} | "
            f"val_acc={row.val_acc:.4f} | "
            f"val_macro_f1={row.val_macro_f1:.4f}"
        )


def _save_history(rows: List[EpochMetrics], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1"])
        for row in rows:
            writer.writerow(
                [
                    row.epoch,
                    f"{row.train_loss:.8f}",
                    f"{row.val_loss:.8f}",
                    f"{row.val_acc:.8f}",
                    f"{row.val_macro_f1:.8f}",
                ]
            )

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_payload = {
        "history": [asdict(r) for r in rows],
        "best_epoch": max(rows, key=lambda r: r.val_macro_f1).epoch if rows else 0,
        "best_val_macro_f1": max((r.val_macro_f1 for r in rows), default=0.0),
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")


def _save_metadata(config: TrainConfig, class_to_idx: Dict[str, int], normal_class_idx: int) -> None:
    payload = {
        "experiment_name": config.experiment_name,
        "model_key": config.model_key,
        "model_kind": config.model_kind,
        "class_names": config.class_names,
        "class_to_idx": class_to_idx,
        "normal_class_name": config.normal_class_name,
        "normal_class_idx": normal_class_idx,
        "image_size": config.image_size,
        "image_format": config.image_format,
        "sequence_length": config.sequence_length if config.model_kind == "temporal" else None,
        "train_stride": config.train_stride if config.model_kind == "temporal" else None,
        "eval_stride": config.eval_stride if config.model_kind == "temporal" else None,
    }
    config.metadata_json_path.parent.mkdir(parents=True, exist_ok=True)
    config.metadata_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_training_experiment(
    config: TrainConfig,
    precomputed_frame_splits: tuple[Dict[str, int], Dict[str, List[FrameSample]]] | None = None,
) -> TrainResult:
    set_global_seed(config.seed)

    if precomputed_frame_splits is None:
        class_to_idx, frame_splits = prepare_frame_splits(
            data_root=config.data_root,
            class_names=config.class_names,
            seed=config.seed,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            image_format=config.image_format,
        )
    else:
        class_to_idx, frame_splits = precomputed_frame_splits

    if config.normal_class_name not in class_to_idx:
        raise ValueError(f"normal_class_name bulunamadi: {config.normal_class_name}")
    normal_class_idx = class_to_idx[config.normal_class_name]

    train_frame_samples = frame_splits["train"]
    val_frame_samples = frame_splits["val"]

    if config.model_kind == "image":
        train_samples = _limit_samples(
            train_frame_samples, config.max_train_samples, config.seed, f"{config.experiment_name}:train"
        )
        val_samples = _limit_samples(
            val_frame_samples, config.max_val_samples, config.seed, f"{config.experiment_name}:val"
        )
        train_dataset = create_image_dataset(
            samples=train_samples,
            image_size=config.image_size,
            image_format=config.image_format,
            batch_size=config.batch_size,
            training=True,
            seed=config.seed,
        )
        val_dataset = create_image_dataset(
            samples=val_samples,
            image_size=config.image_size,
            image_format=config.image_format,
            batch_size=config.batch_size,
            training=False,
            seed=config.seed,
        )
    else:
        train_seq_all = build_sequence_samples(
            frame_samples=train_frame_samples,
            sequence_length=config.sequence_length,
            stride=config.train_stride,
        )
        val_seq_all = build_sequence_samples(
            frame_samples=val_frame_samples,
            sequence_length=config.sequence_length,
            stride=config.eval_stride,
        )
        train_samples = _limit_samples(
            train_seq_all, config.max_train_samples, config.seed, f"{config.experiment_name}:train_seq"
        )
        val_samples = _limit_samples(
            val_seq_all, config.max_val_samples, config.seed, f"{config.experiment_name}:val_seq"
        )
        train_dataset = create_sequence_dataset(
            samples=train_samples,
            image_size=config.image_size,
            image_format=config.image_format,
            sequence_length=config.sequence_length,
            batch_size=config.batch_size,
            training=True,
            seed=config.seed,
        )
        val_dataset = create_sequence_dataset(
            samples=val_samples,
            image_size=config.image_size,
            image_format=config.image_format,
            sequence_length=config.sequence_length,
            batch_size=config.batch_size,
            training=False,
            seed=config.seed,
        )

    print("\nClass dagilimi (train):", class_counts(train_samples))
    print("Class dagilimi (val):", class_counts(val_samples))
    print(f"Train sample sayisi: {len(train_samples)}")
    print(f"Val sample sayisi  : {len(val_samples)}")

    model = build_model(
        model_key=config.model_key,
        image_size=config.image_size,
        sequence_length=config.sequence_length,
        num_classes=len(config.class_names),
    )
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    macro_f1_callback = MacroF1CheckpointCallback(
        val_dataset=val_dataset,
        save_path=config.checkpoint_path,
        verbose=True,
    )
    epoch_log_callback = EpochLogCallback()
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=False,
        verbose=1,
    )

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.epochs,
        callbacks=[macro_f1_callback, epoch_log_callback, early_stop],
        verbose=0,
    )

    _save_history(epoch_log_callback.rows, config.history_csv_path, config.history_json_path)
    _save_metadata(config, class_to_idx=class_to_idx, normal_class_idx=normal_class_idx)

    best_val_macro_f1 = macro_f1_callback.best_macro_f1
    best_epoch = macro_f1_callback.best_epoch
    print(f"\nBest epoch: {best_epoch}, best val_macro_f1: {best_val_macro_f1:.4f}")
    print(f"Best model: {config.checkpoint_path}")

    return TrainResult(
        experiment_name=config.experiment_name,
        model_key=config.model_key,
        model_kind=config.model_kind,
        class_names=config.class_names,
        class_to_idx=class_to_idx,
        normal_class_idx=normal_class_idx,
        best_epoch=best_epoch,
        best_val_macro_f1=best_val_macro_f1,
        train_sample_count=len(train_samples),
        val_sample_count=len(val_samples),
        history_rows=epoch_log_callback.rows,
        checkpoint_path=config.checkpoint_path,
        history_csv_path=config.history_csv_path,
        history_json_path=config.history_json_path,
        metadata_json_path=config.metadata_json_path,
    )
