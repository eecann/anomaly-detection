# Google Colab Calistirma Rehberi

## 1) Runtime
- `Runtime > Change runtime type > GPU`

## 2) Drive baglama
```python
from google.colab import drive
drive.mount('/content/drive')
```

## 3) Proje klasoru
```bash
%cd /content
!git clone <REPO_URL> anomalydetection
%cd anomalydetection
!pip install -r requirements.txt
```

## 4) Dataset yolu
- `configs/dataset.yaml` icinde `data_root` alanini Colab/Drive yoluna gore guncelle:
  - ornek: `/content/drive/MyDrive/ucfcrime/archive`

## 5) Deneyleri tek tek kosma
- `configs/experiments.yaml` icinde sadece bir modeli `enabled: true` yap.
- Sonra:
```bash
!python main.py
```

## 6) Ciktilar
- `outputs/<run_name>/reports/` klasorunde:
  - `experiment_summary.csv`
  - `per_class_comparison.csv`
  - `best_model.json`

## 7) Video inference (opsiyonel)
- `configs/inference.yaml` video yolu ve output yolunu guncelle.
- Telegram icin ortam degiskenleri:
```python
import os
os.environ["TELEGRAM_BOT_TOKEN"] = "..."
os.environ["TELEGRAM_CHAT_ID"] = "..."
```
- Ardindan:
```bash
!python inference.py
```
