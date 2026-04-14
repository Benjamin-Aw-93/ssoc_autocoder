# BGE SSOC Classifier Artifacts

Trained components for the BGE-based SSOC classifier head. The base encoder is **not** included here — download it from Hugging Face.

## Base encoder

[`BAAI/bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5) — 1024-dim, 24 layers, 16 heads, BertModel architecture. See `bge_architecture.txt` for the full module tree dumped from the extracted weights.

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
encoder = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
```

## Trained artifacts

| File | Shape | Dtype | Purpose |
|------|-------|-------|---------|
| `scaler_mean.npy`   | `(768,)`      | float64 | `StandardScaler.mean_` — per-feature mean |
| `scaler_scale.npy`  | `(768,)`      | float64 | `StandardScaler.scale_` — per-feature std |
| `logreg_weights.npy`| `(554, 768)`  | float64 | Logistic regression `coef_` — 554 SSOC classes × 768 features |
| `logreg_bias.npy`   | `(554,)`      | float64 | Logistic regression `intercept_` |

### Dimension note

The classifier operates on **768-dim** feature vectors, while `bge-large-en-v1.5` produces **1024-dim** embeddings. A dimensionality reduction step (PCA, projection, or similar) is applied between the encoder and the scaler — see the training pipeline for the exact transform.

## Loading

```python
import numpy as np

scaler_mean   = np.load("artifacts/bge_ssoc_classifier/scaler_mean.npy")
scaler_scale  = np.load("artifacts/bge_ssoc_classifier/scaler_scale.npy")
logreg_W      = np.load("artifacts/bge_ssoc_classifier/logreg_weights.npy")
logreg_b      = np.load("artifacts/bge_ssoc_classifier/logreg_bias.npy")

def predict_logits(features_768):
    x = (features_768 - scaler_mean) / scaler_scale
    return x @ logreg_W.T + logreg_b
```
