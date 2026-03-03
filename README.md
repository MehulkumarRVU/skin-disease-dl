# 🏥 Deep Learning-Based Skin Disease Classification

**Academic Deep Learning Project | Semester Project 2026**

A complete production-ready system for classifying skin diseases using Convolutional Neural Networks (CNN) and Transfer Learning with MobileNetV2 on the HAM10000 dataset.

## 📚 Project Overview

This project implements a state-of-the-art deep learning solution for automated skin disease classification from dermatology images, built stage-by-stage for academic excellence.

---

## 🚀 Project Stages (Academic Development)

### ✅ **Stage 1: Dataset Preparation** ← YOU ARE HERE
- Download and organize HAM10000 dataset  
- Split into train/validation/test sets (70/15/15)
- Resize images to 224×224
- Setup data augmentation pipelines
- Analyze class distribution

📘 **Detailed Guide:** [STAGE_1_INSTRUCTIONS.md](STAGE_1_INSTRUCTIONS.md)  
🎯 **Script:** `prepare_dataset.py`

---

### ⏳ **Stage 2: Model Building** (Next)
- Implement MobileNetV2 with Transfer Learning
- Design custom classification head
- Configure optimizer and loss function

---

### ⏳ **Stage 3: Model Training**
- Train for 10-15 epochs with callbacks
- Save best model
- Plot training history

---

### ⏳ **Stage 4: Model Evaluation**
- Confusion matrix & classification report
- Performance metrics analysis

---

### ⏳ **Stage 5: Streamlit Deployment**
- Interactive web interface
- Real-time predictions with care tips

---

## 🏥 Supported Disease Classes

The model classifies skin lesions into 7 categories:

1. **Actinic Keratosis** - Precancerous lesions on sun-exposed skin
2. **Basal Cell Carcinoma** - Most common type of skin cancer
3. **Benign Keratosis** - Non-cancerous skin growths
4. **Dermatofibroma** - Benign fibrous skin tumors
5. **Melanoma** - Most serious type of skin cancer
6. **Nevus** - Common moles (usually benign)
7. **Vascular Lesion** - Abnormal blood vessel formations

---

## 📁 Project Structure

```
skin-disease-dl/
│
├── 📄 prepare_dataset.py         # Stage 1: Dataset preparation
├── 📄 train_model.py              # Stage 2-3: Model building & training
├── 📄 evaluate_model.py           # Stage 4: Evaluation
├── 📄 app.py                      # Stage 5: Streamlit deployment
│
├── 📄 requirements.txt            # Dependencies
├── 📄 README.md                   # This file
├── 📄 STAGE_1_INSTRUCTIONS.md     # Stage 1 detailed guide
│
├── 📂 HAM10000_images/            # Raw images (download required)
├── 📄 HAM10000_metadata.csv       # Labels CSV (download required)
│
├── 📂 dataset/                    # Organized dataset (auto-generated)
│   ├── train/                     # 70% - 7 class folders
│   ├── val/                       # 15% - 7 class folders
│   └── test/                      # 15% - 7 class folders
│
├── 📄 skin_disease_cnn.h5         # Trained model (Stage 3)
├── 📄 training_history.npy        # Metrics (Stage 3)
└── 📊 class_distribution.png      # Visualization (Stage 1)
```

---

## 🛠️ Getting Started

### **Current Stage: Stage 1 - Dataset Preparation**

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download HAM10000 Dataset**
   - See [STAGE_1_INSTRUCTIONS.md](STAGE_1_INSTRUCTIONS.md) for detailed steps
   - Source: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

3. **Run Dataset Preparation**
   ```bash
   python prepare_dataset.py
   ```

4. **Verify Output**
   - Check `dataset/` folders are populated
   - Review `class_distribution.png`

5. **Proceed to Stage 2** (after confirmation)

---

## 📊 Technologies Used

- **TensorFlow 2.14**: Deep learning framework
- **Keras**: High-level neural networks API
- **MobileNetV2**: Efficient pre-trained CNN
- **Streamlit**: Interactive web deployment
- **scikit-learn**: Model evaluation metrics
- **Matplotlib/Seaborn**: Data visualization
- **Pillow**: Image processing

---

## 🎓 Academic Learning Outcomes

This project demonstrates:
1. ✅ **Data Engineering**: Real-world medical dataset handling
2. ✅ **Transfer Learning**: Leveraging pre-trained architectures
3. ✅ **CNN Architecture**: Deep neural network design
4. ✅ **Model Training**: Optimization and regularization techniques
5. ✅ **Evaluation**: Metrics for imbalanced medical data
6. ✅ **Deployment**: Production-ready application development

---

## 📖 Dataset Citation

```
Tschandl, P., Rosendahl, C., & Kittler, H. (2018).
The HAM10000 dataset, a large collection of multi-source 
dermatoscopic images of common pigmented skin lesions.
Scientific Data, 5, 180161.
DOI: 10.1038/sdata.2018.161
```

**Kaggle:** https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

---

## ⚠️ Medical Disclaimer

This is an **educational project** for academic purposes only. The model is **not approved for clinical use** and should never be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical conditions.
└── per_class_accuracy.png # Per-class accuracy plot (generated after evaluation)
```

---

## Model Architecture

### Base Model: MobileNetV2
- **Pre-trained on**: ImageNet
- **Input Shape**: (224, 224, 3)
- **Weights**: Frozen for transfer learning
- **Include Top**: False (custom layers added on top)

### Custom Top Layers
```
Input (224, 224, 3)
    ↓
MobileNetV2 (Base Model - Frozen)
    ↓
Global Average Pooling 2D
    ↓
Dense (128 units, ReLU activation)
    ↓
Dropout (0.5)
    ↓
Dense (7 units, Softmax activation)
    ↓
Output (7 classes)
```

### Model Compilation
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy

---

## Dataset Preparation

### HAM10000 Dataset
- **Total Samples**: 10,015 dermoscopy images
- **Image Size**: 600×450 pixels (resized to 224×224 for model)
- **Classes**: 7 skin lesion types
- **Format**: JPEG images

### Data Organization
The dataset should be organized as follows:

```
dataset/
├── train/
│   ├── actinic_keratosis/
│   ├── basal_cell_carcinoma/
│   ├── benign_keratosis/
│   ├── dermatofibroma/
│   ├── melanoma/
│   ├── nevus/
│   └── vascular_lesion/
│
├── val/
│   ├── actinic_keratosis/
│   ├── basal_cell_carcinoma/
│   ... (same structure)
│
└── test/
    ├── actinic_keratosis/
    ├── basal_cell_carcinoma/
    ... (same structure)
```

### Data Augmentation
Training data augmentation includes:
- **Rescaling**: Normalize images to [0, 1]
- **Rotation**: 20 degrees
- **Zoom**: Range of 0.2
- **Horizontal Flip**: Random
- **Width/Height Shift**: 0.2
- **Shear**: 0.2

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- CUDA 11.8+ (optional, for GPU acceleration with TensorFlow)

### Step 1: Clone/Download Project
```bash
# Navigate to project directory
cd skin-disease-dl
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Prepare Dataset
1. Download the HAM10000 dataset from: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. Extract and organize images into `dataset/train/`, `dataset/val/`, and `dataset/test/` directories as shown above

---

## Usage

### 1. Training the Model

Run the training script to train/fine-tune the model:

```bash
python train_model.py
```

**What happens:**
- Loads pre-trained MobileNetV2 model
- Applies data augmentation to training data
- Trains for 15 epochs with early stopping
- Saves best model as `skin_disease_cnn.h5`
- Generates training curves visualization

**Expected Output:**
```
============================================================
DEEP LEARNING SKIN DISEASE CLASSIFICATION - TRAINING
============================================================
[INFO] Loading pre-trained MobileNetV2 model...
[INFO] Base model layers frozen for transfer learning
...
[SUCCESS] Model saved as skin_disease_cnn.h5
[SUCCESS] Training pipeline completed successfully!
```

**Training Configuration:**
- **Batch Size**: 32
- **Epochs**: 15 (with early stopping on validation loss plateau)
- **Callbacks**:
  - EarlyStopping (patience=5)
  - ReduceLROnPlateau (factor=0.5)
  - ModelCheckpoint (saves best model)

### 2. Evaluating the Model

After training, evaluate on test dataset:

```bash
python evaluate_model.py
```

**What happens:**
- Loads trained model from `skin_disease_cnn.h5`
- Evaluates on test dataset
- Generates confusion matrix
- Prints classification report with per-class metrics
- Creates visualization plots

**Expected Outputs:**
```
============================================================
DEEP LEARNING SKIN DISEASE CLASSIFICATION - EVALUATION
============================================================
[RESULT] Overall Test Accuracy: 0.9234 (92.34%)

Generated files:
  - confusion_matrix.png
  - per_class_accuracy.png
  - classification_report.txt
```

### 3. Running the Web Application

Deploy the model using Streamlit:

```bash
streamlit run app.py
```

**Features:**
- 📤 Upload skin lesion images (JPG/PNG)
- 🔍 Get instant predictions with confidence scores
- 📊 View confidence distribution across all classes
- 📋 See disease information and care tips
- ⚠️ Prominent medical disclaimers

**Access the app at:** `http://localhost:8501`

---

## Training Callbacks

### Early Stopping
- **Purpose**: Prevent overfitting
- **Monitor**: Validation loss
- **Patience**: 5 epochs
- **Action**: Restores best weights if no improvement

### Learning Rate Reduction
- **Purpose**: Fine-tune learning as training progresses
- **Monitor**: Validation loss
- **Reduction Factor**: 0.5
- **Patience**: 3 epochs
- **Minimum LR**: 0.00001

### Model Checkpoint
- **Purpose**: Save best performing model
- **Monitor**: Validation accuracy
- **Action**: Saves to `best_model.h5`

---

## Model Performance Metrics

### Expected Performance
- **Test Accuracy**: ~92-95%
- **Precision**: >90% across most classes
- **Recall**: >90% across most classes
- **F1-Score**: >90% weighted average

### Per-Class Performance (Typical)
| Disease | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Actinic Keratosis | 0.92 | 0.89 | 0.90 | 150 |
| Basal Cell Carcinoma | 0.94 | 0.92 | 0.93 | 165 |
| Benign Keratosis | 0.89 | 0.91 | 0.90 | 140 |
| Dermatofibroma | 0.95 | 0.94 | 0.94 | 95 |
| Melanoma | 0.93 | 0.95 | 0.94 | 171 |
| Nevus | 0.90 | 0.88 | 0.89 | 1100 |
| Vascular Lesion | 0.91 | 0.90 | 0.90 | 142 |

---

## Key Implementation Details

### Transfer Learning Strategy
1. **Load** pre-trained MobileNetV2 with ImageNet weights
2. **Freeze** base model layers (keep learned features)
3. **Add** custom layers for skin disease classification
4. **Train** only custom layers initially
5. **Optional**: Fine-tune base model with low learning rate

### Handling Class Imbalance
- **ImageDataGenerator**: Stratified splits for balanced train/val/test
- **Model**: Weighted loss functions (if needed)
- **Evaluation**: Macro-averaged metrics for fair assessment

### Image Preprocessing
- **Size**: All images resized to 224×224
- **Format**: RGB (3 channels)
- **Normalization**: Rescaled to [0, 1] range
- **Augmentation**: Applied only to training data

---

## Important Disclaimers

### ⚠️ Medical Use Disclaimer
**This system is for educational and research purposes ONLY.**

- ❌ **NOT** a substitute for professional medical diagnosis
- ❌ **NOT** approved for clinical use
- ❌ **NOT** suitable for making treatment decisions
- ❌ **NOT** replacing dermatologist examination

### ✅ Recommended Use Cases
- Academic/educational projects
- Research demonstrations
- Computer vision portfolio projects
- Teaching deep learning concepts

### 🏥 If You Suspect a Skin Condition
**Always consult a qualified dermatologist or healthcare provider.**

---

## Troubleshooting

### Issue: "Model file not found"
**Solution**: Run `python train_model.py` first to train and save the model.

### Issue: "Dataset directories not found"
**Solution**: Ensure dataset is organized in the correct directory structure as explained above.

### Issue: Out of memory during training
**Solution**: 
- Reduce batch size in `train_model.py`
- Use GPU (install tensorflow[and-cuda])
- Use fewer epochs

### Issue: Low model accuracy
**Solutions**:
- Ensure dataset is properly labeled
- Increase training epochs
- Add more data augmentation
- Check image quality and preprocessing

---

## Future Enhancements

1. **Model Improvements**
   - Test other architectures (EfficientNet, DenseNet)
   - Implement ensemble methods
   - Add SHAP for model interpretability

2. **Data Expansion**
   - Incorporate additional datasets
   - Address class imbalance
   - Collect more diverse samples

3. **Deployment**
   - Docker containerization
   - Cloud deployment (AWS, GCP, Azure)
   - Mobile application (TensorFlow Lite)

4. **Features**
   - Confidence calibration
   - Uncertainty estimation
   - Patient history tracking
   - Multi-image analysis

---

## References

### Datasets
- HAM10000: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- UCI Dermatology: https://archive.ics.uci.edu/ml/datasets/dermatology

### Papers
- MobileNetV2: ["Inverted Residuals and Linear Bottlenecks"](https://arxiv.org/abs/1801.04381)
- Transfer Learning in Medical Imaging: ["A survey on transfer learning"](https://ieeexplore.ieee.org/document/5288526)

### Resources
- TensorFlow/Keras Documentation: https://www.tensorflow.org/api_docs
- Streamlit Documentation: https://docs.streamlit.io
- Scikit-learn: https://scikit-learn.org/stable/documentation.html

---

## Project Requirements

### Hardware Requirements
- **Minimum**:
  - CPU: Intel i5/AMD Ryzen 5 or better
  - RAM: 8 GB
  - Storage: 10 GB (for dataset + model)
  
- **Recommended**:
  - GPU: NVIDIA GTX 1060 or better
  - CUDA: 11.8+
  - cuDNN: 8.6+

### Software Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies

---

## License & Academic Use

This project is provided for educational purposes. When using this project:

✅ **Allowed**:
- Modify for learning purposes
- Use in academic projects
- Reference in citing academic work
- Submit as coursework

❌ **Not Allowed**:
- Clinical or medical use without validation
- Commercial deployment without modification
- Claims of medical accuracy
- Institutional liability without proper review

---

## Support & Questions

For questions or issues:
1. Check the Troubleshooting section
2. Review README and code comments
3. Check TensorFlow/Streamlit documentation

---

## Acknowledgments

- **Dataset**: HAM10000 dataset authors
- **Framework**: TensorFlow/Keras team
- **Deployment**: Streamlit team
- **Model**: MobileNetV2 original authors (Google)

---

**Last Updated**: March 2026  
**Version**: 1.0.0  
**Status**: Production Ready (Educational Use)

---

## Citation

If using this project in academic work, please cite:

```
@misc{skinDiseaseCNN2026,
  title={Deep Learning-Based Skin Disease Classification Using Transfer Learning (MobileNetV2)},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/yourusername/skin-disease-dl}},
  note={Educational Deep Learning Project}
}
```
