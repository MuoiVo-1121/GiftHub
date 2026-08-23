# UVC Photodetector Signal Classification
## ML Pipeline: 1D CNN vs LSTM Comparison

A complete machine-learning pipeline that classifies time-resolved photocurrent
signals from a self-powered UVC photodetector into three source categories.

---

## Problem Definition

| Label | Class | Signal Characteristics |
|-------|-------|------------------------|
| `0` | **Dark / No Source** | Near-zero current, very low variance |
| `1` | **UVC Lamp** | Stable DC plateau, low variance |
| `2` | **Methanol Flame** | High amplitude, high variance, random bursts |

**Input:** Photocurrent time series `I(t) = [i₁, i₂, ..., i₁₀₀]` in nA (1-second window at 100 Hz)
**Output:** Predicted class label ∈ {0, 1, 2}

---

## Project Structure

```
signal_classifier/
├── data/
│   ├── raw/                        # CSV recordings (one file per class)
│   └── processed/                  # Windowed, normalized arrays + global stats
│
├── models/
│   ├── __init__.py
│   ├── cnn_model.py                # 1D CNN architecture  (10,755 params)
│   └── lstm_model.py               # Stacked LSTM architecture (30,467 params)
│
├── weights/                        # Saved best model weights (.h5)
│
├── generate_dummy_data.py          # Synthetic data generator (use if no real data)
├── preprocess.py                   # Windowing + normalization + train/val/test split
├── train.py                        # Model training with early stopping
├── evaluate.py                     # Per-class metrics + confusion matrix PNG
├── compare.py                      # Side-by-side comparison table + bar chart PNG
├── predict.py                      # Single-window inference
└── venv/                           # Python virtual environment
```

---

## Architecture Summary

### 1D CNN
```
Input (100, 1)
  → Conv1D(32, k=5, ReLU) → MaxPool(2)
  → Conv1D(64, k=3, ReLU) → GlobalAvgPool
  → Dense(64, ReLU) → Dropout(0.3)
  → Dense(3, Softmax)
Total params: 10,755
```

### LSTM
```
Input (100, 1)
  → LSTM(64, return_sequences=True) → Dropout(0.3)
  → LSTM(32) → Dropout(0.3)
  → Dense(32, ReLU)
  → Dense(3, Softmax)
Total params: 30,467
```

---

## Critical Design Note — Normalization

**Global min-max normalization is used (NOT per-window mean subtraction).**

```python
# CORRECT — preserves absolute current level
X = (X - global_min) / (global_max - global_min)

# WRONG — erases the DC offset that separates Dark from UVC Lamp
X = X - X.mean(axis=1, keepdims=True)
```

Dark (~0 nA) and UVC Lamp (~10 nA) are primarily distinguished by their
absolute amplitude. Stripping the mean makes them look identical and causes
models — especially LSTM — to completely fail on the UVC Lamp class (F1 = 0.00).

---

## Quick Start

### Step 0 — Environment Setup (first time only)

```bash
cd /Users/macos/Downloads/signal_classifier

python3 -m venv venv
source venv/bin/activate

pip install "tensorflow-macos==2.13.*" scikit-learn numpy pandas matplotlib
```

**Verify:**
```bash
python -c "import tensorflow as tf; print('TF', tf.__version__)"
# Expected: TF 2.13.1
```

---

### Step 1 — Prepare Data

**Option A — Use real photocurrent recordings (recommended)**

Place CSV files in `data/raw/`. Each file must contain:
```
time_s,current_nA,label
0.00,0.12,0
0.01,0.15,0
...
```
Label convention: `0` = Dark, `1` = UVC Lamp, `2` = Methanol Flame.
Minimum recommended: **200 windows per class**.

**Option B — Generate synthetic data (for testing the pipeline)**

```bash
source venv/bin/activate
python generate_dummy_data.py
```

**Verify:**
```bash
ls data/raw/
# Expected: dark_recording.csv  lamp_recording.csv  flame_recording.csv
```

---

### Step 2 — Preprocess

Windows the signal, applies global min-max normalization, and splits into
train (70%) / val (15%) / test (15%) sets.

```bash
source venv/bin/activate
python preprocess.py
```

**Verify:**
```bash
ls data/processed/
# Expected: X_train.npy  X_val.npy  X_test.npy
#           y_train.npy  y_val.npy  y_test.npy
#           global_stats.npy

python -c "
import numpy as np
X = np.load('data/processed/X_train.npy')
y = np.load('data/processed/y_train.npy')
print('Shape:', X.shape)          # (N, 100, 1)
print('Classes:', set(y.tolist()))  # {0, 1, 2}
print('Range: [{:.3f}, {:.3f}]'.format(X.min(), X.max()))  # [0.0, 1.0]
"
```

---

### Step 3 — Train Models

Train each model independently. Best weights are saved automatically.

```bash
source venv/bin/activate

python train.py --model cnn    # trains 1D CNN, saves weights/cnn_best.h5
python train.py --model lstm   # trains LSTM,   saves weights/lstm_best.h5
```

Optional arguments:
```bash
python train.py --model cnn --epochs 80 --batch_size 64 --lr 0.0005
```

**Verify:**
```bash
ls weights/
# Expected: cnn_best.h5  lstm_best.h5

# Check best val accuracy printed at end of each run:
# CNN  — Test accuracy: 1.0000
# LSTM — Test accuracy: 1.0000
```

---

### Step 4 — Evaluate

Loads saved weights, runs on the held-out test set, prints classification
report, and saves confusion matrix PNGs.

```bash
source venv/bin/activate
python evaluate.py --model both
```

Or evaluate a single model:
```bash
python evaluate.py --model cnn
python evaluate.py --model lstm
```

**Expected output (with clean data):**
```
==================================================
  Model : CNN  |  Test Accuracy : 100.00%
==================================================
              precision  recall  f1-score  support
        Dark       1.00    1.00      1.00        9
    UVC Lamp       1.00    1.00      1.00        9
       Flame       1.00    1.00      1.00        9
    accuracy                         1.00       27
```

**Verify outputs:**
```bash
ls confusion_matrix_cnn.png confusion_matrix_lstm.png
# Open them to inspect the 3x3 confusion matrix — diagonal should be all dark
```

---

### Step 5 — Compare

Runs both models on the same test set and produces a side-by-side table and
a bar chart.

```bash
source venv/bin/activate
python compare.py
```

**Expected output:**
```
==============================================================
  Metric                        1D CNN          LSTM
==============================================================
  Accuracy                     1.0000       1.0000  <-- winner
  F1 macro                     1.0000       1.0000  <-- winner
  Precision                    1.0000       1.0000  <-- winner
  Recall                       1.0000       1.0000  <-- winner
--------------------------------------------------------------
  F1 Dark                     1.0000       1.0000
  F1 UVC Lamp                 1.0000       1.0000
  F1 Flame                    1.0000       1.0000
--------------------------------------------------------------
  Parameters                    10,755        30,467
  Inference ms/window           0.754        0.992
==============================================================
```

**Verify:**
```bash
ls comparison_bar.png
# Open to see grouped bar chart — both models at 1.000 across all metrics
```

---

### Step 6 — Predict on New Signal

Run inference on a single window from the command line.

```bash
source venv/bin/activate

# From raw values
python predict.py --model cnn --signal 12.1 12.3 11.9 12.5 12.0 11.8 12.2

# From a CSV file
python predict.py --model lstm --input data/raw/lamp_recording.csv
```

**Expected output:**
```
Prediction : UVC Lamp
  Dark                   0.1%
  UVC Lamp               99.7%
  Methanol Flame         0.2%
```

---

## Full Pipeline — One Command Check

Run all steps end-to-end to verify the entire pipeline works:

```bash
cd /Users/macos/Downloads/signal_classifier
source venv/bin/activate

python generate_dummy_data.py && \
python preprocess.py && \
python train.py --model cnn && \
python train.py --model lstm && \
python evaluate.py --model both && \
python compare.py && \
echo "=== PIPELINE COMPLETE ==="
```

---

## Experimental Results

### With Synthetic Data (current run)

| | 1D CNN | LSTM |
|-|--------|------|
| Test Accuracy | **100%** | **100%** |
| F1 Macro | **1.000** | **1.000** |
| Parameters | **10,755** | 30,467 |
| Inference | **0.75 ms** | 0.99 ms |
| Epochs to best val | ~17 | ~6 |

### Confusion Matrices
Both models produce a perfect diagonal (zero misclassifications):

```
              Predicted
              Dark  Lamp  Flame
Actual  Dark  [ 9    0     0  ]
        Lamp  [ 0    9     0  ]
       Flame  [ 0    0     9  ]
```

---

## Deployment Recommendation

**Use 1D CNN** for production / embedded use:
- Same accuracy as LSTM
- 3× fewer parameters → smaller binary for microcontrollers
- 25% faster inference
- Straightforward to export via TFLite:

```python
import tensorflow as tf
model = tf.keras.models.load_model("weights/cnn_best.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("cnn_classifier.tflite", "wb").write(tflite_model)
```

Use **LSTM** if:
- Signal windows are longer than 5 seconds (long-range temporal context matters)
- Gradual drift or warm-up behaviour needs to be captured

---

## Dependencies

```
tensorflow-macos==2.13.*   # or tensorflow==2.13.* on non-Apple hardware
scikit-learn
numpy
pandas
matplotlib
```

Install:
```bash
pip install "tensorflow-macos==2.13.*" scikit-learn numpy pandas matplotlib
```
