# Classical CNN Baseline for Brain Tumor Classification

## Overview

This project implements a **classical Convolutional Neural Network (CNN)** as a baseline model for brain tumor classification using MRI images. The model classifies brain MRI scans into four categories:
- **Glioma**
- **Meningioma**
- **Pituitary tumor**
- **No tumor**

This baseline model serves as a reference point for comparison with quantum-enhanced models (QCNN and HQCNN).

---

## Dataset

**Dataset Name:** Brain Tumor Classification (MRI)  
**Author:** Sartaj Bhuvaji  
**Source:** Kaggle

### Dataset Structure
```
Brain-Tumor-Classification/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── no_tumor/
```

### Download Instructions
1. Visit: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
2. Download the dataset
3. Extract the files
4. Update the `TRAIN_DIR` and `TEST_DIR` paths in the script

---

## Requirements

### Python Version
- Python 3.8 or higher

### Dependencies
```bash
tensorflow>=2.10.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

### Installation

1. **Clone or download this project:**
```bash
git clone <your-repo-url>
cd brain-tumor-cnn-baseline
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages:**
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

---

## Model Architecture

### CNN Structure

```
Input (128×128×3)
    ↓
[Conv2D(32, 3×3) + ReLU] → MaxPooling(2×2)
    ↓
[Conv2D(64, 3×3) + ReLU] → MaxPooling(2×2)
    ↓
[Conv2D(128, 3×3) + ReLU] → MaxPooling(2×2)
    ↓
Flatten
    ↓
Dense(256) + ReLU
    ↓
Dropout(0.5)
    ↓
Dense(4) + Softmax
    ↓
Output (4 classes)
```

### Model Specifications

| Component | Details |
|-----------|---------|
| **Input Shape** | 128 × 128 × 3 |
| **Convolutional Layers** | 3 blocks with 32, 64, 128 filters |
| **Kernel Size** | 3 × 3 |
| **Pooling** | 2 × 2 Max Pooling |
| **Activation** | ReLU (hidden), Softmax (output) |
| **Dense Layer** | 256 neurons |
| **Dropout** | 0.5 |
| **Total Parameters** | ~1.5 million |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| **Image Size** | 128 × 128 |
| **Batch Size** | 32 |
| **Epochs** | 20 (with early stopping) |
| **Learning Rate** | 0.001 |
| **Optimizer** | Adam |
| **Loss Function** | Categorical Cross-Entropy |

---

## Usage

### Basic Usage

1. **Update dataset paths in the script:**
```python
class Config:
    TRAIN_DIR = 'path/to/Training'  # Update this
    TEST_DIR = 'path/to/Testing'    # Update this
```

2. **Run the training script:**
```bash
python cnn_brain_tumor_classifier.py
```

### Expected Output

The script will:
1. Load and preprocess the dataset
2. Build the CNN model
3. Train the model with validation
4. Evaluate on test data
5. Generate visualizations:
   - Training/validation accuracy curves
   - Training/validation loss curves
   - Confusion matrix
6. Print classification report
7. Save the trained model as `cnn_baseline_model.h5`

### Output Files

- `cnn_baseline_model.h5` - Trained model weights
- `training_history.png` - Training curves
- `confusion_matrix.png` - Confusion matrix visualization

---

## Data Preprocessing

### Training Data Augmentation
- **Rescaling:** Pixel values normalized to [0, 1]
- **Rotation:** Random rotation up to 15°
- **Shift:** Random horizontal/vertical shift (10%)
- **Flip:** Random horizontal flip
- **Zoom:** Random zoom (10%)

### Test Data
- **Rescaling only:** Pixel values normalized to [0, 1]
- **No augmentation** (preserves original test distribution)

---

## Evaluation Metrics

The model is evaluated using:

1. **Accuracy:** Overall classification accuracy
2. **Precision:** Positive predictive value per class
3. **Recall:** Sensitivity per class
4. **F1-Score:** Harmonic mean of precision and recall
5. **Confusion Matrix:** Visual representation of predictions

### Sample Output
```
Classification Report:
                precision    recall  f1-score   support

      glioma       0.8765    0.9012    0.8887       300
  meningioma       0.8932    0.8723    0.8826       306
    no_tumor       0.9654    0.9512    0.9582       405
   pituitary       0.9234    0.9456    0.9344       300

    accuracy                           0.9175      1311
   macro avg       0.9146    0.9176    0.9160      1311
weighted avg       0.9178    0.9175    0.9176      1311
```

---

## Model Interpretation

### Architecture Rationale

1. **Shallow Network:** 3 convolutional blocks are sufficient for this task
   - Deeper networks risk overfitting on limited medical data
   - Captures hierarchical features from edges to complex patterns

2. **Progressive Filter Increase:** 32 → 64 → 128
   - Lower layers capture basic features (edges, textures)
   - Higher layers capture complex patterns (tumor shapes, structures)

3. **Regularization:**
   - Dropout (0.5) prevents overfitting
   - Data augmentation improves generalization

### Expected Performance

- **Test Accuracy:** 85-95% (depends on dataset split)
- **Training Time:** 10-20 minutes on GPU, 1-2 hours on CPU
- **Inference Time:** <10ms per image

### Baseline Purpose

This CNN serves as a **classical baseline** for:
- Comparing with quantum convolutional neural networks (QCNN)
- Comparing with hybrid quantum-classical neural networks (HQCNN)
- Understanding quantum advantage in medical image classification
- Establishing performance benchmarks

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM) Error
**Solution:** Reduce batch size
```python
BATCH_SIZE = 16  # or even 8
```

#### 2. Dataset Not Found
**Solution:** Verify paths
```python
# Use absolute paths
TRAIN_DIR = '/absolute/path/to/Training'
TEST_DIR = '/absolute/path/to/Testing'
```

#### 3. Low Accuracy
**Possible causes:**
- Incorrect data paths
- Corrupted images
- Imbalanced classes

**Solutions:**
- Verify dataset integrity
- Check class distribution
- Increase training epochs

#### 4. Slow Training
**Solutions:**
- Use GPU acceleration
- Reduce image size (not recommended)
- Enable mixed precision training

---

## Advanced Configuration

### GPU Acceleration

```python
# Check GPU availability
import tensorflow as tf
print(f"GPUs Available: {tf.config.list_physical_devices('GPU')}")

# Enable memory growth (prevents OOM)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

### Custom Callbacks

```python
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

# Save best model
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# TensorBoard logging
tensorboard = TensorBoard(log_dir='./logs')

# Add to training
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=config.EPOCHS,
    callbacks=[checkpoint, tensorboard, early_stopping, reduce_lr]
)
```

---

## Citation

If you use this code, please cite:

**Dataset:**
```
Bhuvaji, S., Kadam, A., Bhumkar, P., Dedge, S., & Kanchan, S. (2020).
Brain Tumor Classification (MRI). Kaggle.
https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
```

**TensorFlow:**
```
Abadi, M., et al. (2016). TensorFlow: A system for large-scale machine learning.
In OSDI (Vol. 16, pp. 265-283).
```

---

## Future Enhancements

Potential improvements for this baseline:
1. **Transfer Learning:** Use pre-trained models (VGG16, ResNet50)
2. **Ensemble Methods:** Combine multiple models
3. **Class Balancing:** Handle imbalanced datasets
4. **Hyperparameter Tuning:** Grid search or Bayesian optimization
5. **Explainability:** Grad-CAM visualizations

---

## Comparison with Quantum Models

### Metrics for Comparison

| Model | Accuracy | Parameters | Training Time | Inference Time |
|-------|----------|------------|---------------|----------------|
| **CNN (Baseline)** | ~90% | 1.5M | 15-20 min | <10ms |
| **QCNN** | TBD | Lower | TBD | TBD |
| **HQCNN** | TBD | Medium | TBD | TBD |

The quantum models (QCNN, HQCNN) should demonstrate:
- Comparable or better accuracy
- Reduced parameter count
- Potential quantum advantage in specific scenarios

---

## License

This project is provided for educational and research purposes.

---

## Contact

For questions or issues:
- Open an issue in the repository
- Contact: [Your contact information]

---

## Acknowledgments

- **Dataset:** Sartaj Bhuvaji and team
- **Framework:** TensorFlow/Keras team
- **Inspiration:** Medical image analysis community

---

**Last Updated:** January 2026
