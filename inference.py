from __future__ import annotations

import csv
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import requests
import tensorflow as tf
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / "configs"
INFERENCE_CONFIG_PATH = CONFIG_DIR / "inference.yaml"
ALERTS_CONFIG_PATH = CONFIG_DIR / "alerts.yaml"


@dataclass
class InferenceConfig:
    best_model_info_path: Path
    input_video_path: Path
    output_dir: Path
    sample_fps: float
    threshold: float
    min_consecutive: int
    cooldown_sec: float
    sequence_stride: int
    save_annotated_video: bool
    annotated_video_name: str
    event_log_name: str
    summary_name: str


@dataclass
class TelegramConfig:
    enabled: bool
    token_env: str
    chat_id_env: str


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


def load_inference_config(path: Path) -> InferenceConfig:
    cfg = load_yaml(path).get("inference", {})
    if not cfg:
        raise ValueError("inference.yaml icinde 'inference' bolumu bos.")
    return InferenceConfig(
        best_model_info_path=resolve_path(str(cfg["best_model_info_path"]), path.parent),
        input_video_path=resolve_path(str(cfg["input_video_path"]), path.parent),
        output_dir=resolve_path(str(cfg.get("output_dir", "../outputs/inference")), path.parent),
        sample_fps=float(cfg.get("sample_fps", 5.0)),
        threshold=float(cfg.get("threshold", 0.6)),
        min_consecutive=int(cfg.get("min_consecutive", 3)),
        cooldown_sec=float(cfg.get("cooldown_sec", 60)),
        sequence_stride=int(cfg.get("sequence_stride", 1)),
        save_annotated_video=bool(cfg.get("save_annotated_video", False)),
        annotated_video_name=str(cfg.get("annotated_video_name", "annotated_output.mp4")),
        event_log_name=str(cfg.get("event_log_name", "anomaly_events.csv")),
        summary_name=str(cfg.get("summary_name", "inference_summary.json")),
    )


def load_telegram_config(path: Path) -> TelegramConfig:
    cfg = load_yaml(path).get("telegram", {})
    if not cfg:
        return TelegramConfig(enabled=False, token_env="TELEGRAM_BOT_TOKEN", chat_id_env="TELEGRAM_CHAT_ID")
    return TelegramConfig(
        enabled=bool(cfg.get("enabled", False)),
        token_env=str(cfg.get("token_env", "TELEGRAM_BOT_TOKEN")),
        chat_id_env=str(cfg.get("chat_id_env", "TELEGRAM_CHAT_ID")),
    )


def _send_telegram_message(telegram_cfg: TelegramConfig, message: str) -> bool:
    if not telegram_cfg.enabled:
        return False
    token = os.getenv(telegram_cfg.token_env, "")
    chat_id = os.getenv(telegram_cfg.chat_id_env, "")
    if not token or not chat_id:
        print(
            f"[WARN] Telegram aktif ama env eksik. Gerekli: {telegram_cfg.token_env}, {telegram_cfg.chat_id_env}"
        )
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code == 200:
            return True
        print(f"[WARN] Telegram mesaj hatasi: {r.status_code} {r.text}")
        return False
    except requests.RequestException as exc:
        print(f"[WARN] Telegram baglanti hatasi: {exc}")
        return False


def _preprocess_frame(frame_bgr: np.ndarray, image_size: int) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return frame_resized.astype(np.float32)


def _format_alert_message(
    timestamp_sec: float,
    predicted_class: str,
    predicted_prob: float,
    anomaly_score: float,
    threshold: float,
) -> str:
    ts_str = f"{timestamp_sec:.2f}s"
    return (
        "Anomali tespit edildi.\n"
        f"Zaman: {ts_str}\n"
        f"Sinif: {predicted_class}\n"
        f"Sinif olasiligi: {predicted_prob:.4f}\n"
        f"Anomaly score: {anomaly_score:.4f}\n"
        f"Threshold: {threshold:.4f}"
    )


def run_video_inference() -> None:
    inference_cfg = load_inference_config(INFERENCE_CONFIG_PATH)
    telegram_cfg = load_telegram_config(ALERTS_CONFIG_PATH)

    if not inference_cfg.input_video_path.exists():
        raise FileNotFoundError(f"Video bulunamadi: {inference_cfg.input_video_path}")
    if not inference_cfg.best_model_info_path.exists():
        raise FileNotFoundError(f"best_model.json bulunamadi: {inference_cfg.best_model_info_path}")

    model_info = json.loads(inference_cfg.best_model_info_path.read_text(encoding="utf-8"))
    model_path = Path(model_info["model_path"])
    model_metadata_path = Path(model_info["model_metadata_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model bulunamadi: {model_path}")
    if not model_metadata_path.exists():
        raise FileNotFoundError(f"Model metadata bulunamadi: {model_metadata_path}")

    model_meta = json.loads(model_metadata_path.read_text(encoding="utf-8"))
    class_names = list(model_meta["class_names"])
    normal_class_idx = int(model_meta["normal_class_idx"])
    model_kind = str(model_meta["model_kind"])
    image_size = int(model_meta["image_size"])
    sequence_length = int(model_meta["sequence_length"] or 1)

    inference_cfg.output_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(model_path, compile=False)

    cap = cv2.VideoCapture(str(inference_cfg.input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {inference_cfg.input_video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 25.0
    sample_step = max(int(round(source_fps / max(inference_cfg.sample_fps, 0.1))), 1)

    writer = None
    if inference_cfg.save_annotated_video:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = inference_cfg.output_dir / inference_cfg.annotated_video_name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, source_fps / sample_step, (width, height))

    sequence_buffer: deque[np.ndarray] = deque(maxlen=sequence_length)
    event_rows: List[Dict[str, Any]] = []
    consecutive = 0
    last_alert_walltime = -1e12
    total_predictions = 0
    total_alerts = 0
    model_inference_index = 0

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        do_sample = (frame_idx % sample_step) == 0
        prediction = None
        if do_sample:
            processed = _preprocess_frame(frame, image_size=image_size)
            if model_kind == "image":
                input_tensor = np.expand_dims(processed, axis=0)
                prediction = model.predict(input_tensor, verbose=0)[0]
            else:
                sequence_buffer.append(processed)
                if len(sequence_buffer) == sequence_length and (model_inference_index % inference_cfg.sequence_stride == 0):
                    input_tensor = np.expand_dims(np.stack(sequence_buffer, axis=0), axis=0)
                    prediction = model.predict(input_tensor, verbose=0)[0]
                model_inference_index += 1

        alert_sent = False
        if prediction is not None:
            total_predictions += 1
            probs = prediction.astype(np.float64)
            pred_idx = int(np.argmax(probs))
            pred_class = class_names[pred_idx]
            pred_prob = float(probs[pred_idx])
            anomaly_score = float(1.0 - probs[normal_class_idx])
            above_threshold = anomaly_score >= inference_cfg.threshold

            if above_threshold:
                consecutive += 1
            else:
                consecutive = 0

            timestamp_sec = frame_idx / source_fps
            now_wall = time.time()
            if (
                above_threshold
                and consecutive >= inference_cfg.min_consecutive
                and (now_wall - last_alert_walltime) >= inference_cfg.cooldown_sec
            ):
                msg = _format_alert_message(
                    timestamp_sec=timestamp_sec,
                    predicted_class=pred_class,
                    predicted_prob=pred_prob,
                    anomaly_score=anomaly_score,
                    threshold=inference_cfg.threshold,
                )
                alert_sent = _send_telegram_message(telegram_cfg, msg)
                last_alert_walltime = now_wall
                if alert_sent:
                    total_alerts += 1

            event_rows.append(
                {
                    "frame_index": frame_idx,
                    "timestamp_sec": f"{timestamp_sec:.4f}",
                    "predicted_class": pred_class,
                    "predicted_prob": f"{pred_prob:.6f}",
                    "anomaly_score": f"{anomaly_score:.6f}",
                    "threshold": f"{inference_cfg.threshold:.6f}",
                    "above_threshold": int(above_threshold),
                    "consecutive_count": consecutive,
                    "alert_sent": int(alert_sent),
                }
            )

            if writer is not None:
                display_text = (
                    f"{pred_class} p={pred_prob:.2f} | anomaly={anomaly_score:.2f} "
                    f"| cons={consecutive} | alert={int(alert_sent)}"
                )
                cv2.putText(frame, display_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                writer.write(frame)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    event_log_path = inference_cfg.output_dir / inference_cfg.event_log_name
    with event_log_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(
            [
                "frame_index",
                "timestamp_sec",
                "predicted_class",
                "predicted_prob",
                "anomaly_score",
                "threshold",
                "above_threshold",
                "consecutive_count",
                "alert_sent",
            ]
        )
        for row in event_rows:
            writer_csv.writerow(
                [
                    row["frame_index"],
                    row["timestamp_sec"],
                    row["predicted_class"],
                    row["predicted_prob"],
                    row["anomaly_score"],
                    row["threshold"],
                    row["above_threshold"],
                    row["consecutive_count"],
                    row["alert_sent"],
                ]
            )

    summary_payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "input_video_path": str(inference_cfg.input_video_path),
        "model_path": str(model_path),
        "model_key": model_info.get("model_key"),
        "model_kind": model_kind,
        "sample_fps": inference_cfg.sample_fps,
        "threshold": inference_cfg.threshold,
        "min_consecutive": inference_cfg.min_consecutive,
        "cooldown_sec": inference_cfg.cooldown_sec,
        "total_frames_read": frame_idx,
        "total_predictions": total_predictions,
        "total_alerts_sent": total_alerts,
        "event_log_path": str(event_log_path),
        "annotated_video_path": (
            str(inference_cfg.output_dir / inference_cfg.annotated_video_name)
            if inference_cfg.save_annotated_video
            else None
        ),
    }
    summary_path = inference_cfg.output_dir / inference_cfg.summary_name
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("\nInference tamamlandi.")
    print(f"Event log : {event_log_path}")
    print(f"Summary   : {summary_path}")
    if inference_cfg.save_annotated_video:
        print(f"Video     : {inference_cfg.output_dir / inference_cfg.annotated_video_name}")


if __name__ == "__main__":
    run_video_inference()
