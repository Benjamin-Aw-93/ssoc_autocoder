# ssoc_autocoder

Core Python package for SSOC prediction.

## Modules

| Module | Purpose |
|--------|---------|
| `processing.py` | HTML job description parsing, verb extraction, text cleaning |
| `model_training.py` | Dataset class, HierarchicalSSOCClassifier (V1/V2), training loop |
| `model_prediction.py` | Single and batch prediction generation |
| `augmentation.py` | Data augmentation (word embeddings, back translation, synonym, contextual, sentence, summarization) |
| `converting_json.py` | MCF JSON extraction and weekly CSV splitting |
| `utils.py` | Verbose printing, raw data processing |
| `masked_language_model.py` | TensorFlow MLM pre-training pipeline |
| `run_mlm.py` | PyTorch MLM pre-training (HuggingFace Trainer) |
| `predict.py` | Standalone prediction script |

## Key Dependencies

- PyTorch, HuggingFace Transformers (DistilBERT)
- spaCy (`en_core_web_lg`)
- BeautifulSoup4, nlpaug, pandas, numpy
