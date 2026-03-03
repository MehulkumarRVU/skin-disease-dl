"""
Stage 2 & 3: Model Building & Training for Skin Disease Classification
=======================================================================

This script implements the complete training pipeline using Transfer Learning
with MobileNetV2 on the HAM10000 dataset.

Key Features:
- Transfer Learning with pre-trained ImageNet weights
- Frozen base model layers for efficient training
- Custom classification head
- Data augmentation during training
- Callbacks: EarlyStopping and ReduceLROnPlateau
- Model checkpoint saving
- Training history visualization

Architecture:
    MobileNetV2 (frozen) → GlobalAveragePooling2D → Dense(128) → 
    Dropout(0.5) → Dense(7, softmax)

Author: Academic Deep Learning Project
Date: March 2026
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ============================================================
# CONFIGURATION
# ============================================================

# Dataset paths
DATASET_PATH = "dataset"  # Organized dataset from Stage 1
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "val")
TEST_PATH = os.path.join(DATASET_PATH, "test")

# Model configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 7
BATCH_SIZE = 32
EPOCHS = 15
INITIAL_LEARNING_RATE = 0.001

# Model save paths
MODEL_SAVE_PATH = "skin_disease_cnn.h5"
CHECKPOINT_PATH = "models/best_model.h5"
HISTORY_SAVE_PATH = "training_history.npy"

# Class names for HAM10000 dataset
CLASS_NAMES = [
    'actinic_keratosis',
    'basal_cell_carcinoma',
    'benign_keratosis',
    'dermatofibroma',
    'melanoma',
    'melanocytic_nevus',
    'vascular_lesion'
]


# ============================================================
# STEP 1: DATA GENERATORS
# ============================================================

def create_data_generators():
    """
    Create data generators with augmentation for training, validation, and test sets.
    
    AUGMENTATION STRATEGY:
    - Training: Full augmentation to increase dataset diversity
      * Rotation ±20°, width/height shifts, shear, zoom, brightness adjustments
      * Prevents overfitting by presenting diverse variations
    - Validation/Test: Only rescaling (no augmentation)
      * Ensures fair model evaluation on unaltered images
    
    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """
    print("[INFO] Creating data generators with augmentation...")
    
    # Training augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Validation/Test (no augmentation)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators from directories
    train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        VAL_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        TEST_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"[INFO] Data generators created:")
    print(f"       Training samples: {train_generator.samples}")
    print(f"       Validation samples: {val_generator.samples}")
    print(f"       Test samples: {test_generator.samples}")
    print(f"       Classes: {list(train_generator.class_indices.keys())}")
    
    return train_generator, val_generator, test_generator


# ============================================================
# STEP 2: TRANSFER LEARNING MODEL
# ============================================================

def build_transfer_learning_model():
    """
    Build MobileNetV2 transfer learning model.
    
    TRANSFER LEARNING RATIONALE:
    - MobileNetV2: Lightweight architecture (3.5M parameters)
    - Pre-trained on ImageNet: 1.4M diverse images across 1000 classes
    - Frozen base layers: Leverages learned features, prevents catastrophic forgetting
    - Fine-tuning not used: Limited training data, risk of overfitting
    
    ARCHITECTURE:
                          Input (224×224×3)
                                ↓
                    MobileNetV2 Base Model (frozen)
                                ↓
                        GlobalAveragePooling2D
                    (Reduces 7×7×1280 → 1280)
                                ↓
                    Dense(128, ReLU activation)
                                ↓
                          Dropout(0.5)
                    (Regularization, prevents overfitting)
                                ↓
                    Dense(7, Softmax activation)
                   (Output: 7 disease classes)
    
    Returns:
        keras.Model: Compiled model
    """
    print("[INFO] Building transfer learning model with MobileNetV2...")
    
    # Load MobileNetV2 with ImageNet weights
    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,  # Remove default classification head
        weights='imagenet'   # Use pre-trained ImageNet weights
    )
    
    # Freeze all base layers (transfer learning)
    base_model.trainable = False
    print(f"[INFO] Frozen {len(base_model.layers)} base model layers")
    
    # Build custom classification head
    model = keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu', name='dense_1'),
        Dropout(0.5, name='dropout_1'),
        Dense(NUM_CLASSES, activation='softmax', name='output')
    ], name='skin_disease_classifier')
    
    print(f"[INFO] Model architecture created successfully")
    
    return model


# ============================================================
# STEP 3: COMPILE MODEL
# ============================================================

def compile_model(model):
    """
    Compile model with optimizer, loss, and metrics.
    
    OPTIMIZATION CHOICES:
    - Optimizer: Adam (learning_rate=0.001)
      * Adaptive moment estimation
      * Per-parameter learning rate adjustment
      * Efficient for transfer learning
    
    - Loss: Categorical crossentropy
      * Standard for multi-class classification
      * Measures probability distribution difference
    
    - Metrics: Accuracy
      * Primary performance metric for classification
    
    Args:
        model: Uncompiled Keras model
        
    Returns:
        model: Compiled Keras model
    """
    print("[INFO] Compiling model...")
    
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"[INFO] Model compiled with Adam optimizer (lr={INITIAL_LEARNING_RATE})")
    print(f"[INFO] Total trainable parameters: {model.count_params():,}")
    
    return model


# ============================================================
# STEP 4: SETUP CALLBACKS
# ============================================================

def create_callbacks():
    """
    Setup training callbacks for regularization and monitoring.
    
    CALLBACK DESCRIPTIONS:
    
    1. EarlyStopping:
       - Monitors validation loss for improvement
       - Stops training if no improvement for 5 consecutive epochs
       - Restores weights to best epoch (prevents overfitting)
    
    2. ReduceLROnPlateau:
       - Monitors validation loss
       - Reduces learning rate by 50% when loss plateaus
       - Allows fine-tuning of weights in later training
       - Minimum learning rate: 0.00001 (prevents excessive reduction)
    
    3. ModelCheckpoint:
       - Saves model weights when validation accuracy improves
       - Preserves best model during training
    
    Returns:
        list: Configured callback objects
    """
    print("[INFO] Setting up training callbacks...")
    
    # Create models directory for checkpoints
    os.makedirs('models', exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1,
            mode='min'
        ),
        ModelCheckpoint(
            os.path.join('models', 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        )
    ]
    
    print("[INFO] Callbacks configured:")
    print("       - EarlyStopping (patience=5)")
    print("       - ReduceLROnPlateau (factor=0.5)")
    print("       - ModelCheckpoint (monitors val_accuracy)")
    
    return callbacks


# ============================================================
# STEP 5: TRAIN MODEL
# ============================================================

def train_model(model, train_gen, val_gen, callbacks):
    """
    Train the model using data generators with callbacks.
    
    TRAINING CONFIGURATION:
    - Batch size: 32 (balance between speed and stability)
    - Epochs: 15 (initial estimate, may stop early)
    - Steps per epoch: Automatic (calculated from dataset size)
    - Validation: Per epoch on validation set
    
    Args:
        model: Compiled Keras model
        train_gen: Training data generator
        val_gen: Validation data generator
        callbacks: List of callbacks
        
    Returns:
        history: Training history object
    """
    print("[INFO] Starting model training...")
    print("="*70)
    print(f"Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | Initial LR: {INITIAL_LEARNING_RATE}")
    print("="*70)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    print("="*70)
    print("[INFO] Training completed!")
    print("="*70)
    
    return history


# ============================================================
# STEP 6: SAVE MODEL AND HISTORY
# ============================================================

def save_model_and_history(model, history):
    """
    Save trained model and training history to disk.
    
    FILES SAVED:
    - skin_disease_cnn.h5: Complete trained model (architecture + weights)
      * Used for predictions in Stage 3 (evaluate_model.py)
      * Used for deployment in Stage 5 (app.py)
    
    - models/best_model.h5: Best checkpoint (saved during training)
      * Alternative model with highest validation accuracy
    
    - training_history.npy: NumPy array of training history
      * Metrics: accuracy, loss, val_accuracy, val_loss per epoch
      * Used for visualization and analysis
    
    Args:
        model: Trained Keras model
        history: Training history object
    """
    print("[INFO] Saving model and history...")
    
    # Save full model
    model.save(MODEL_SAVE_PATH)
    print(f"[SUCCESS] Model saved to {MODEL_SAVE_PATH}")
    
    # Save training history
    np.save(HISTORY_SAVE_PATH, history.history)
    print(f"[SUCCESS] History saved to {HISTORY_SAVE_PATH}")
    
    print("[INFO] Files ready for Stage 3 (Evaluation)")


# ============================================================
# STEP 7: VISUALIZE TRAINING
# ============================================================

def plot_training_metrics(history):
    """
    Visualize training and validation metrics.
    
    PLOTS GENERATED:
    - Accuracy over epochs: Shows how model improves on train vs val sets
    - Loss over epochs: Shows overfitting (val loss > train loss)
    - Saved as: training_history.png (300 DPI, high quality)
    
    INTERPRETATION GUIDE:
    - Ideal: Both curves smooth and following similar trends
    - Overfitting: Val loss increases while train loss decreases
    - Underfitting: Both losses high and not improving
    
    Args:
        history: Training history object
    """
    print("[INFO] Generating training visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', 
                linewidth=2.5, marker='o', markersize=4)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', 
                linewidth=2.5, marker='s', markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11, loc='lower right')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_ylim([0, 1.0])
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss', 
                linewidth=2.5, marker='o', markersize=4)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', 
                linewidth=2.5, marker='s', markersize=4)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11, loc='upper right')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Training plot saved to training_history.png")
    
    # Print summary statistics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Total Epochs Trained: {len(history.history['accuracy'])}")
    print("="*70 + "\n")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """
    Main execution function orchestrating all 7 training steps.
    """
    print("\n" + "="*70)
    print("STAGE 2 & 3: MODEL BUILDING & TRAINING")
    print("Deep Learning Skin Disease Classification")
    print("="*70 + "\n")
    
    # Verify dataset exists
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(VAL_PATH):
        print("[ERROR] Dataset directories not found!")
        print(f"[INFO] Expected structure:")
        print(f"       {DATASET_PATH}/train/<class_name>/")
        print(f"       {DATASET_PATH}/val/<class_name>/")
        print(f"       {DATASET_PATH}/test/<class_name>/")
        print("\n[INFO] Please run Stage 1 (prepare_dataset.py) first to organize images.")
        return
    
    try:
        # STEP 1: Create data generators
        train_gen, val_gen, test_gen = create_data_generators()
        print()
        
        # STEP 2: Build model
        model = build_transfer_learning_model()
        print()
        
        # STEP 3: Compile model
        model = compile_model(model)
        print()
        
        # STEP 4: Setup callbacks
        callbacks = create_callbacks()
        print()
        
        # STEP 5: Train model
        history = train_model(model, train_gen, val_gen, callbacks)
        print()
        
        # STEP 6: Save model and history
        save_model_and_history(model, history)
        print()
        
        # STEP 7: Visualize training
        plot_training_metrics(history)
        
        print("[SUCCESS] STAGES 2 & 3 COMPLETE!")
        print("[INFO] Next: Run evaluate_model.py for Stage 4 (Model Evaluation)")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {str(e)}")
        print(f"[DEBUG] Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
    
    # Save model
    print(f"[INFO] Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    print(f"[SUCCESS] Model saved as {MODEL_SAVE_PATH}")
    
    # Save training history
    np.save(HISTORY_SAVE_PATH, history.history)
    print(f"[INFO] Training history saved as {HISTORY_SAVE_PATH}")
    
    # Plot training history
    plot_training_history(history)
    
    print("\n[SUCCESS] Training pipeline completed successfully!")
    print(f"[INFO] Model is ready for evaluation at {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
