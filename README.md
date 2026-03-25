# UCF-Crime 14-Class Keras Pipeline

Bu proje, UCF-Crime frame dataseti icin tez odakli bir multiclass pipeline sunar:

- 4 modeli ayni veri kosulunda egitir ve karsilastirir:
  - `mobilenetv2_image`
  - `efficientnetb0_image`
  - `cnn_lstm_temporal`
  - `convlstm_temporal`
- Test metrikleri uretir:
  - Accuracy
  - Per-class Precision/Recall/F1
  - Macro/Weighted Precision/Recall/F1
  - Confusion Matrix
  - One-vs-Rest ROC-AUC ve PR-AUC
- En iyi modeli secer (`selection_metric`, varsayilan `test_macro_f1`)
- Video uzerinde anomaly score (`1 - P(NormalVideos)`) ile tespit yapar
- Anomali durumunda Telegram bildirimi gonderir

## 1) Kurulum

```bash
pip install -r requirements.txt
```

## 2) Konfig dosyalari

- `configs/dataset.yaml`: dataset yolu, 14 sinif, split, normal class, image format/size
- `configs/experiments.yaml`: deney listesi, model ayarlari, output klasoru, secim metrigi
- `configs/inference.yaml`: video inference parametreleri
- `configs/alerts.yaml`: Telegram ayarlari

## 3) Egitim + Karsilastirma

```bash
python main.py
```

Hizli smoke test icin `configs/experiments.yaml` icerisinde gecici olarak su degerleri kullanabilirsin:
- `epochs: 1`
- `max_train_samples: 1000`
- `max_val_samples: 300`
- `max_test_samples: 800`

Uretilen ana rapor dosyalari:

- `outputs/<run_name>/reports/experiment_summary.csv`
- `outputs/<run_name>/reports/per_class_comparison.csv`
- `outputs/<run_name>/reports/experiment_summary.md`
- `outputs/<run_name>/reports/best_model.json`

Her deneyin kendi klasorunde:

- train history (`history.csv`, `history.json`)
- best model (`best_model.keras`)
- test grafik/tablolari (ROC, PR, confusion)

## 4) Video Inference + Telegram

### Ortam degiskenleri

`.env` (onerilen):

```bash
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

Not: `inference.py` calisirken proje kokundeki `.env` dosyasini otomatik yukler.

ya da terminalden:

```bash
set TELEGRAM_BOT_TOKEN=...
set TELEGRAM_CHAT_ID=...
```

### Calistirma

```bash
python inference.py
```

Ciktilar:

- `anomaly_events.csv`
- `inference_summary.json`
- opsiyonel `annotated_output.mp4`

## 5) Colab Onerisi

- Runtime: GPU
- Deneyleri tek tek kos: `experiments.yaml` icinde bir deneyi `enabled: true`, digerlerini `false`
- Her calistirma sonunda rapor dosyalarini Drive'a kaydet
