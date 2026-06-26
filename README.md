
# Paddy Crop Disease Detection using CNN

A convolutional neural network (CNN) model that classifies paddy (rice) leaf diseases from images, achieving **85–90% validation accuracy** across 5 categories.

## Disease Classes

| Class | Description |
|---|---|
| `healthy` | No disease present |
| `leaf_blast` | Fungal infection causing diamond-shaped lesions |
| `brown_spot` | Brown circular spots on leaves |
| `sheath_blight` | Water-soaked lesions on leaf sheath |
| `bacterial_blight` | Water-soaked to yellowish stripe on leaf margin |

## Project Structure

```
paddy-disease-cnn/
├── data/                  # Dataset (not tracked in git)
│   └── <class_name>/      # One folder per disease class
├── models/                # Saved model weights (after training)
├── src/
│   ├── train.py           # Model training script
│   ├── predict.py         # Single-image inference
│   └── data_prep.py       # Dataset check and preview
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/muzamilmohammed/paddy-disease-cnn.git
cd paddy-disease-cnn
pip install -r requirements.txt
```

## Dataset

Download the rice leaf disease dataset from Kaggle:
[https://www.kaggle.com/datasets/shayanriyaz/riceleafs](https://www.kaggle.com/datasets/shayanriyaz/riceleafs)

Place images in `data/<class_name>/` folders matching the class names above.

Verify your dataset structure:
```bash
python src/data_prep.py
```

## Training

```bash
python src/train.py
```

Training outputs saved to `models/`:
- `paddy_cnn_model.h5` — best model weights
- `training_history.png` — accuracy/loss curves
- `confusion_matrix.png` — per-class evaluation

## Inference

```bash
python src/predict.py --image path/to/leaf.jpg
```

## Model Architecture

- 3 convolutional blocks (32 → 64 → 128 filters)
- Batch Normalisation + Dropout after each block
- Dense head: 256 → softmax(5)
- Trained with Adam optimiser, early stopping, and LR scheduling

## Results

| Metric | Value |
|---|---|
| Validation Accuracy | 85–90% |
| Dataset Size | 500+ images |
| Input Size | 128×128 RGB |
| Classes | 5 |

## Tech Stack

Python · TensorFlow / Keras · NumPy · OpenCV · scikit-learn · Matplotlib

---

*B.Tech Project — Electronics & Communication Engineering, Malla Reddy College of Engineering (2025)*
