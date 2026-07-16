# Industrial Defect Detection Using Deep Learning
# Deep Learning-Based Automated Visual Inspection
## Hybrid CNN + Vision Transformer Model

**Research Review Project | 2024–2025**  
Saiguru Nagula · Labdhi Shah · Vinayak Pandit · Solomon

---

## Architecture

```
Input Image (224×224×3)
        │
        ▼
┌───────────────────────────┐
│   CNN Backbone            │  ResNet-50 (ImageNet pretrained)
│   Feature Pyramid Network │  Scales: P2 (H/8), P3 (H/16), P4 (H/32)
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│   Transformer Attention   │  Multi-Head Self-Attention (8 heads)
│   Layer (P3 + P4)         │  Positional Encoding (2D Sinusoidal)
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│   Domain Adaptation       │  Gradient Reversal Layer
│   Module                  │  Adversarial discriminator
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│   Multi-Scale Defect      │  Shared detection towers
│   Detection Head          │  Classification + Regression
└───────────────────────────┘
        │
        ▼
    Output: Class Logits + Detection Maps + Confidence
```

---

## Project Structure

```
AVI_Project/
│
├── models/
│   └── hybrid_avi_model.py    # Full Hybrid CNN+Transformer architecture
│
├── datasets/
│   └── avi_datasets.py        # MVTec AD | NEU-DET | Synthetic loaders
│
├── utils/
│   ├── training_engine.py     # Focal Loss | Optimizer | LR Scheduler
│   ├── evaluation.py          # Metrics | Confusion Matrix | ROC | PR curves
│   └── explainability.py      # GradCAM | GradCAM++ | Attention Rollout
│
├── outputs/                   # Generated plots, checkpoints, reports
│   └── xai/                   # XAI explanation images
│
├── AVI_Complete_Pipeline.ipynb  # Step-by-step Jupyter notebook
├── main.py                    # CLI entry point
├── requirements.txt
└── README.md
```

---

## Quick Start

### Option 1: Jupyter Notebook (Recommended)
```bash
pip install -r requirements.txt
jupyter notebook AVI_Complete_Pipeline.ipynb
```

### Option 2: Command Line
```bash
# Full pipeline (train + eval + XAI)
python main.py --mode all --epochs 50 --batch_size 32

# Train only
python main.py --mode train --epochs 50

# Evaluate only (loads best checkpoint)
python main.py --mode eval

# Demo on a single image
python main.py --mode demo --image path/to/defect.jpg
```

### With real datasets
```bash
python main.py --mode all \
    --mvtec_root data/mvtec_anomaly_detection \
    --mvtec_cat  bottle \
    --neu_root   data/NEU-DET \
    --epochs 50
```

---

## Datasets

| Dataset | Classes | Task | Download |
|---------|---------|------|----------|
| **MVTec AD** | 15 categories, normal/anomaly | Anomaly detection | [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad) |
| **NEU-DET** | 6 defect types | Object detection | [neu.edu.cn](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html) |
| **Synthetic** | 2–6 classes | Classification | Auto-generated (no download) |

> **Note:** If real dataset paths are not provided, all datasets automatically fall back to procedurally-generated synthetic data with the same interface. The pipeline runs completely without downloading anything.

---

## Key Features

| Feature | Implementation |
|---------|---------------|
| **Multi-scale features** | ResNet-50 + Feature Pyramid Network (P2, P3, P4) |
| **Attention mechanism** | Multi-Head Self-Attention (8 heads, 256d) |
| **Domain adaptation** | Gradient Reversal Layer + adversarial discriminator |
| **Loss function** | Focal Loss (α=0.25, γ=2.0) + Domain CE Loss |
| **Optimizer** | AdamW with differential LR (backbone = 10× lower) |
| **LR schedule** | Cosine Annealing with Linear Warmup |
| **Training tricks** | Mixed precision (AMP), gradient clipping, early stopping |
| **Explainability** | GradCAM, GradCAM++, Attention Rollout, SHAP-compatible |
| **Metrics** | Accuracy, Precision, Recall, F1, AUC-ROC, mAP |

---

## Outputs

After running the pipeline, `outputs/` contains:

```
outputs/
├── best_model.pth           # Best checkpoint (by Val F1)
├── training_history.json    # Epoch-by-epoch metric log
├── training_curves.png      # Loss / F1 / AUC-ROC / LR plots
├── confusion_matrix.png     # Normalized confusion matrix
├── roc_curve.png            # ROC curve with AUC scores
├── pr_curve.png             # Precision-Recall curve
├── evaluation_report.json   # Full classification report
└── xai/
    ├── sample_1_xai.png     # GradCAM | GradCAM++ | Attn Rollout
    ├── sample_2_xai.png
    ├── sample_3_xai.png
    └── batch_xai_grid.png   # 8-sample explanation grid
```

---

## References

1. X. Zhu et al., "Automated assembly quality inspection by deep learning with 2D and 3D synthetic CAD data," *Journal of Intelligent Manufacturing*, 2024.
2. K. Wonggasem et al., "Automated quality inspection of baby corn using image processing and deep learning," *Artificial Intelligence in Agriculture*, 2024.
3. N. Hütten et al., "Deep learning for automated visual inspection in manufacturing and maintenance: A survey," *Applied System Innovation*, 2024.
4. Y. Ma et al., "Surface defect inspection of industrial products with object detection deep networks: A systematic review," *Artificial Intelligence Review*, 2024.
