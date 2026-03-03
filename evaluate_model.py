"""
Stage 4: Model Evaluation and Metrics Analysis
===============================================

This script evaluates the trained MobileNetV2 Transfer Learning model
on the test dataset. It generates comprehensive metrics including:

- Test accuracy
- Confusion matrix
- Classification report (precision, recall, F1-score)
- Per-class accuracy analysis

Evaluation Approach:
- Load test set using ImageDataGenerator
- Generate predictions for all test images
- Compute evaluation metrics using scikit-learn
- Visualize confusion matrix with heatmap

Author: Academic Deep Learning Project
Date: March 2026
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============================================================
# CONFIGURATION
# ============================================================

# Model path
MODEL_PATH = "skin_disease_cnn.h5"

# Dataset paths
DATASET_PATH = "dataset"
TEST_PATH = os.path.join(DATASET_PATH, "test")

# Image settings
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Class names (must match training)
CLASS_NAMES = [
    'actinic_keratosis',
    'basal_cell_carcinoma',
    'benign_keratosis',
    'dermatofibroma',
    'melanoma',
    'melanocytic_nevus',
    'vascular_lesion'
]

# Prettier display names
CLASS_DISPLAY_NAMES = {
    'actinic_keratosis': 'Actinic Keratosis',
    'basal_cell_carcinoma': 'Basal Cell Carcinoma',
    'benign_keratosis': 'Benign Keratosis',
    'dermatofibroma': 'Dermatofibroma',
    'melanoma': 'Melanoma',
    'melanocytic_nevus': 'Melanocytic Nevus',
    'vascular_lesion': 'Vascular Lesion'
}

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_test_data():
    """
    Load test dataset using ImageDataGenerator.
    
    Preprocessing:
    - Rescale to [0, 1] range
    - Resize to 224×224
    - Batch size: 32
    - No augmentation (preserve original images)
    
    Returns:
        ImageDataGenerator: Test data generator
    """
    print("[INFO] Loading test dataset...")
    
    # Create test data generator (no augmentation)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load test data
    test_generator = test_datagen.flow_from_directory(
        TEST_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # Important: don't shuffle for evaluation
    )
    
    print(f"[INFO] Test dataset loaded:")
    print(f"       Total samples: {test_generator.samples}")
    print(f"       Batch size: {test_generator.batch_size}")
    print(f"       Steps per epoch: {len(test_generator)}")
    
    return test_generator


def load_model_safe(model_path):
    """
    Load trained model safely with error handling.
    
    Args:
        model_path: Path to model file
    
    Returns:
        tensorflow.keras.Model: Loaded model
    """
    print("[INFO] Loading trained model...")
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        print("[INFO] Please run 'python train_model.py' first.")
        return None
    
    try:
        model = load_model(model_path)
        print(f"[SUCCESS] Model loaded successfully")
        print(f"          Parameters: {model.count_params():,}")
        return model
    
    except Exception as e:
        print(f"[ERROR] Failed to load model: {str(e)}")
        return None


def generate_predictions(model, test_generator):
    """
    Generate predictions for all test samples.
    
    Args:
        model: Trained Keras model
        test_generator: Test data generator
    
    Returns:
        tuple: (y_true, y_pred_proba, y_pred_classes)
    """
    print("[INFO] Generating predictions on test set...")
    
    # Get true labels
    y_true = test_generator.classes
    
    # Generate predictions
    predictions = model.predict(test_generator, verbose=1)
    
    # Get predicted classes (argmax of probabilities)
    y_pred_classes = np.argmax(predictions, axis=1)
    
    print("[SUCCESS] Predictions generated")
    print(f"          Total predictions: {len(y_pred_classes)}")
    
    return y_true, predictions, y_pred_classes


def evaluate_model_metrics(y_true, y_pred_classes):
    """
    Compute evaluation metrics.
    
    Metrics:
    - Overall accuracy
    - Per-class accuracy
    - Confusion matrix
    - Classification report
    
    Args:
        y_true: True class labels
        y_pred_classes: Predicted class labels
    
    Returns:
        dict: Dictionary containing all metrics
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION RESULTS")
    print("="*70 + "\n")
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred_classes)
    print(f"[METRIC] Overall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Classification report
    class_report = classification_report(
        y_true, y_pred_classes,
        target_names=[CLASS_DISPLAY_NAMES[cn] for cn in CLASS_NAMES],
        digits=4,
        zero_division=0
    )
    
    # Per-class accuracy
    print("\n" + "-"*70)
    print("PER-CLASS ACCURACY")
    print("-"*70)
    print(f"{'Disease':<30} {'Accuracy':>15} {'Total Samples':>15}")
    print("-"*70)
    
    per_class_acc = {}
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = y_true == i
        if class_mask.sum() > 0:
            class_acc = (y_pred_classes[class_mask] == i).sum() / class_mask.sum()
            per_class_acc[class_name] = {
                'accuracy': class_acc,
                'samples': class_mask.sum()
            }
            
            display_name = CLASS_DISPLAY_NAMES[class_name]
            print(f"{display_name:<30} {class_acc:>14.2%} {class_mask.sum():>15}")
    
    print("-"*70 + "\n")
    
    # Classification report
    print("DETAILED CLASSIFICATION REPORT")
    print("-"*70)
    print(class_report)
    print("-"*70 + "\n")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'per_class_accuracy': per_class_acc
    }


def plot_confusion_matrix(cm, class_names, display_names):
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        display_names: Dictionary mapping names to display names
    """
    print("[INFO] Generating confusion matrix visualization...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Convert to DataFrame for better labels
    cm_df = pd.DataFrame(
        cm,
        index=[display_names[name] for name in class_names],
        columns=[display_names[name] for name in class_names]
    )
    
    # Plot heatmap
    sns.heatmap(
        cm_df,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar_kws={'label': 'Number of Samples'},
        ax=ax,
        square=True,
        linewidths=1,
        linecolor='gray'
    )
    
    # Labels and title
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Skin Disease Classification\nTest Set Evaluation',
                 fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("[SUCCESS] Confusion matrix saved as 'confusion_matrix.png'")
    
    plt.show()


def print_summary(eval_metrics):
    """
    Print summary of evaluation results.
    
    Args:
        eval_metrics: Dictionary of evaluation metrics
    """
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    accuracy = eval_metrics['accuracy']
    
    print(f"\n✓ Test Accuracy: {accuracy*100:.2f}%")
    print(f"✓ Model Performance: {'Excellent' if accuracy > 0.85 else 'Good' if accuracy > 0.75 else 'Fair' if accuracy > 0.65 else 'Poor'}")
    print(f"✓ Test Samples: {sum([v['samples'] for v in eval_metrics['per_class_accuracy'].values()])}")
    
    # Best and worst performing classes
    per_class = eval_metrics['per_class_accuracy']
    best_class = max(per_class.items(), key=lambda x: x[1]['accuracy'])
    worst_class = min(per_class.items(), key=lambda x: x[1]['accuracy'])
    
    print(f"\n📊 Best Performing Class: {CLASS_DISPLAY_NAMES[best_class[0]]} ({best_class[1]['accuracy']*100:.2f}%)")
    print(f"📊 Worst Performing Class: {CLASS_DISPLAY_NAMES[worst_class[0]]} ({worst_class[1]['accuracy']*100:.2f}%)")
    
    print("\n" + "="*70 + "\n")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """
    Main evaluation pipeline.
    """
    print("\n" + "="*70)
    print("STAGE 4: MODEL EVALUATION")
    print("Skin Disease Classification - Test Set Analysis")
    print("="*70 + "\n")
    
    # Verify dataset exists
    if not os.path.exists(TEST_PATH):
        print(f"[ERROR] Test dataset not found at: {TEST_PATH}")
        print("[INFO] Please run Stage 1 (prepare_dataset.py) first.")
        return
    
    # Load model
    model = load_model_safe(MODEL_PATH)
    if model is None:
        return
    
    # Load test data
    test_generator = load_test_data()
    
    # Generate predictions
    y_true, y_pred_proba, y_pred_classes = generate_predictions(model, test_generator)
    
    # Evaluate model
    eval_metrics = evaluate_model_metrics(y_true, y_pred_classes)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        eval_metrics['confusion_matrix'],
        CLASS_NAMES,
        CLASS_DISPLAY_NAMES
    )
    
    # Print summary
    print_summary(eval_metrics)
    
    print("[SUCCESS] STAGE 4 COMPLETE: Model Evaluation Finished!")
    print("[INFO] Files generated:")
    print("       - confusion_matrix.png (visualization)")
    print("       - Console output (detailed metrics)")
    print("="*70 + "\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
