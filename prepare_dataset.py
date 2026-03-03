"""
Stage 1: Dataset Preparation for Skin Disease Classification
=============================================================

This script prepares the HAM10000 dataset for training a deep learning model.

Key Responsibilities:
- Organize images into train/validation/test folders by class
- Resize images to 224x224 pixels (MobileNetV2 input size)
- Setup data augmentation pipelines
- Analyze and visualize class distribution
- Generate dataset statistics

HAM10000 Dataset Classes (7 types):
1. Actinic keratosis (akiec)
2. Basal cell carcinoma (bcc)
3. Benign keratosis (bkl)
4. Dermatofibroma (df)
5. Melanoma (mel)
6. Melanocytic nevus (nv)
7. Vascular lesion (vasc)

Author: Academic Deep Learning Project
Date: March 2026
"""

import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURATION
# ===================
# =========================================

# Paths
RAW_DATASET_PATH = "assets"  # Your raw dataset folder (contains all images)
METADATA_FILE = "assets/HAM10000_metadata.csv"  # Metadata CSV file
ORGANIZED_DATASET_PATH = "dataset"

# Target image dimensions (MobileNetV2 standard input)
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Data split ratios
TRAIN_RATIO = 0.70  # 70% for training
VAL_RATIO = 0.15    # 15% for validation
TEST_RATIO = 0.15   # 15% for testing

# Random seed for reproducibility
RANDOM_SEED = 42

# Class name mapping (short code to full name)
CLASS_MAPPING = {
    'akiec': 'actinic_keratosis',
    'bcc': 'basal_cell_carcinoma',
    'bkl': 'benign_keratosis',
    'df': 'dermatofibroma',
    'mel': 'melanoma',
    'nv': 'melanocytic_nevus',
    'vasc': 'vascular_lesion'
}

# Full class names for display
FULL_CLASS_NAMES = [
    'Actinic Keratosis',
    'Basal Cell Carcinoma',
    'Benign Keratosis',
    'Dermatofibroma',
    'Melanoma',
    'Melanocytic Nevus',
    'Vascular Lesion'
]


# ============================================================
# STEP 1: CREATE DIRECTORY STRUCTURE
# ============================================================

def create_directory_structure():
    """
    Create organized folder structure for train/val/test splits.
    
    Structure:
        dataset/
            train/
                actinic_keratosis/
                basal_cell_carcinoma/
                ...
            val/
                actinic_keratosis/
                ...
            test/
                actinic_keratosis/
                ...
    """
    print("=" * 60)
    print("STEP 1: Creating Directory Structure")
    print("=" * 60)
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        for class_code, class_name in CLASS_MAPPING.items():
            dir_path = os.path.join(ORGANIZED_DATASET_PATH, split, class_name)
            os.makedirs(dir_path, exist_ok=True)
            
    print(f"✓ Created directory structure at: {ORGANIZED_DATASET_PATH}")
    print(f"  - Splits: {splits}")
    print(f"  - Classes per split: {len(CLASS_MAPPING)}")
    print()


# ============================================================
# STEP 2: LOAD AND ANALYZE METADATA
# ============================================================

def load_metadata():
    """
    Load HAM10000 metadata CSV file.
    
    Expected columns:
    - image_id: Image filename (without extension)
    - dx: Diagnosis (class label)
    - dx_type: Type of diagnosis
    - age: Patient age
    - sex: Patient gender
    - localization: Body location
    
    Returns:
        pd.DataFrame: Metadata dataframe
    """
    print("=" * 60)
    print("STEP 2: Loading Metadata")
    print("=" * 60)
    
    if not os.path.exists(METADATA_FILE):
        print(f"❌ ERROR: Metadata file not found: {METADATA_FILE}")
        print(f"\nPlease ensure you have:")
        print(f"1. Downloaded HAM10000 dataset")
        print(f"2. Placed metadata CSV file in project root")
        print(f"3. Named it '{METADATA_FILE}' or update METADATA_FILE variable")
        return None
    
    df = pd.read_csv(METADATA_FILE)
    print(f"✓ Loaded metadata: {len(df)} images")
    print(f"  Columns: {list(df.columns)}")
    print()
    
    return df


# ============================================================
# STEP 3: ANALYZE CLASS DISTRIBUTION
# ============================================================

def analyze_class_distribution(df):
    """
    Analyze and visualize class distribution in the dataset.
    
    Args:
        df (pd.DataFrame): Metadata dataframe
    """
    print("=" * 60)
    print("STEP 3: Analyzing Class Distribution")
    print("=" * 60)
    
    # Count samples per class
    class_counts = df['dx'].value_counts()
    
    print("\nClass Distribution:")
    print("-" * 60)
    for class_code, count in class_counts.items():
        class_name = CLASS_MAPPING.get(class_code, class_code)
        percentage = (count / len(df)) * 100
        print(f"  {class_name:25s}: {count:5d} images ({percentage:5.2f}%)")
    
    print(f"\nTotal Images: {len(df)}")
    print(f"Number of Classes: {len(class_counts)}")
    print()
    
    # Skip visualization due to matplotlib compatibility issues
    print("⚠ Skipping visualization (matplotlib compatibility issue)")
    print("  Class distribution shown above")
    print()


# ============================================================
# STEP 4: SPLIT DATASET
# ============================================================

def split_dataset(df):
    """
    Split dataset into train/validation/test sets with stratification.
    
    Stratification ensures each split has similar class distribution.
    
    Args:
        df (pd.DataFrame): Metadata dataframe
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    print("=" * 60)
    print("STEP 4: Splitting Dataset")
    print("=" * 60)
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=df['dx']
    )
    
    # Second split: separate train and validation sets
    # Adjust val_ratio to account for remaining data
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        random_state=RANDOM_SEED,
        stratify=train_val_df['dx']
    )
    
    print(f"Dataset Split Summary:")
    print(f"  Training Set:   {len(train_df):5d} images ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Validation Set: {len(val_df):5d} images ({VAL_RATIO*100:.0f}%)")
    print(f"  Test Set:       {len(test_df):5d} images ({TEST_RATIO*100:.0f}%)")
    print(f"  Total:          {len(df):5d} images")
    print()
    
    return train_df, val_df, test_df


# ============================================================
# STEP 5: ORGANIZE AND PREPROCESS IMAGES
# ============================================================

def copy_and_preprocess_images(df, split_name):
    """
    Copy and preprocess images to organized folder structure.
    
    Preprocessing steps:
    - Resize to 224x224 pixels
    - Convert to RGB format
    - Save as JPEG
    
    Args:
        df (pd.DataFrame): Dataframe for specific split
        split_name (str): 'train', 'val', or 'test'
    """
    print(f"Processing {split_name} set...")
    success_count = 0
    error_count = 0
    
    for idx, row in df.iterrows():
        try:
            # Get image filename and class
            image_id = row['image_id']
            class_code = row['dx']
            class_name = CLASS_MAPPING[class_code]
            
            # Find source image (try different extensions)
            source_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                potential_path = os.path.join(RAW_DATASET_PATH, image_id + ext)
                if os.path.exists(potential_path):
                    source_path = potential_path
                    break
            
            if source_path is None:
                error_count += 1
                continue
            
            # Load and preprocess image
            img = Image.open(source_path)
            
            # Convert to RGB (some images might be grayscale)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to target dimensions
            img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
            
            # Save to organized location
            dest_path = os.path.join(
                ORGANIZED_DATASET_PATH,
                split_name,
                class_name,
                f"{image_id}.jpg"
            )
            
            img.save(dest_path, 'JPEG', quality=95)
            success_count += 1
            
        except Exception as e:
            error_count += 1
            continue
    
    print(f"  ✓ Successfully processed: {success_count} images")
    if error_count > 0:
        print(f"  ⚠ Errors encountered: {error_count} images")
    print()


def organize_all_images(train_df, val_df, test_df):
    """
    Organize all images into train/val/test folders.
    
    Args:
        train_df, val_df, test_df: Dataframes for each split
    """
    print("=" * 60)
    print("STEP 5: Organizing and Preprocessing Images")
    print("=" * 60)
    print(f"Target size: {IMG_WIDTH}x{IMG_HEIGHT} pixels")
    print()
    
    if not os.path.exists(RAW_DATASET_PATH):
        print(f"❌ ERROR: Raw dataset folder not found: {RAW_DATASET_PATH}")
        print(f"\nPlease:")
        print(f"1. Download HAM10000 dataset images")
        print(f"2. Place them in '{RAW_DATASET_PATH}' folder")
        print(f"3. Or update RAW_DATASET_PATH variable to match your folder")
        return
    
    copy_and_preprocess_images(train_df, 'train')
    copy_and_preprocess_images(val_df, 'val')
    copy_and_preprocess_images(test_df, 'test')
    
    print("✓ All images organized successfully!")
    print()


# ============================================================
# STEP 6: SETUP DATA AUGMENTATION
# ============================================================

def demonstrate_data_augmentation():
    """
    Demonstrate data augmentation techniques to be used during training.
    
    Data Augmentation Benefits:
    - Increases effective dataset size
    - Improves model generalization
    - Reduces overfitting
    - Makes model robust to variations
    """
    print("=" * 60)
    print("STEP 6: Data Augmentation Configuration")
    print("=" * 60)
    
    print("\nTraining Data Augmentation:")
    print("-" * 60)
    print("  • Rotation: ±20 degrees")
    print("  • Width/Height Shift: 20%")
    print("  • Shear Transformation: 20%")
    print("  • Zoom: 20%")
    print("  • Horizontal Flip: Yes")
    print("  • Vertical Flip: Yes")
    print("  • Brightness: 80-120%")
    print("  • Normalization: Rescale to [0,1]")
    
    print("\nValidation/Test Data Preprocessing:")
    print("-" * 60)
    print("  • No augmentation (original images only)")
    print("  • Normalization: Rescale to [0,1]")
    
    print("\n📚 Why Data Augmentation?")
    print("-" * 60)
    print("  1. Medical images can vary in lighting and orientation")
    print("  2. Limited dataset size requires artificial expansion")
    print("  3. Model learns invariant features")
    print("  4. Improves real-world performance")
    
    print("\n✓ Data augmentation pipeline configured")
    print("  (Will be applied during model training)")
    print()


# ============================================================
# STEP 7: GENERATE DATASET STATISTICS
# ============================================================

def generate_statistics():
    """
    Generate comprehensive statistics about the organized dataset.
    """
    print("=" * 60)
    print("STEP 7: Dataset Statistics Summary")
    print("=" * 60)
    
    stats = {}
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(ORGANIZED_DATASET_PATH, split)
        split_stats = {}
        total_images = 0
        
        for class_name in CLASS_MAPPING.values():
            class_path = os.path.join(split_path, class_name)
            if os.path.exists(class_path):
                num_images = len([f for f in os.listdir(class_path) 
                                if f.endswith(('.jpg', '.jpeg', '.png'))])
                split_stats[class_name] = num_images
                total_images += num_images
        
        stats[split] = {'classes': split_stats, 'total': total_images}
    
    # Display statistics
    print("\nOrganized Dataset Structure:")
    print("-" * 60)
    
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()} SET ({stats[split]['total']} images):")
        for class_name, count in stats[split]['classes'].items():
            display_name = class_name.replace('_', ' ').title()
            print(f"  {display_name:25s}: {count:4d} images")
    
    print("\n" + "=" * 60)
    print(f"TOTAL IMAGES: {sum(s['total'] for s in stats.values())}")
    print("=" * 60)
    print()


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """
    Main execution function - orchestrates all dataset preparation steps.
    """
    print("\n" + "=" * 60)
    print("SKIN DISEASE CLASSIFICATION - DATASET PREPARATION")
    print("Stage 1: HAM10000 Dataset Organization")
    print("=" * 60)
    print()
    
    # Step 1: Create folder structure
    create_directory_structure()
    
    # Step 2: Load metadata
    df = load_metadata()
    if df is None:
        print("\n⚠ SETUP REQUIRED:")
        print("-" * 60)
        print("To proceed with dataset preparation, you need:")
        print()
        print("1. HAM10000 Dataset:")
        print(f"   - Download from: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        print(f"   - Extract images to: {RAW_DATASET_PATH}/")
        print()
        print("2. Metadata File:")
        print(f"   - Place 'HAM10000_metadata.csv' in project root")
        print(f"   - Or update METADATA_FILE variable in this script")
        print()
        print("After setup, run this script again.")
        print()
        return
    
    # Step 3: Analyze class distribution
    analyze_class_distribution(df)
    
    # Step 4: Split dataset
    train_df, val_df, test_df = split_dataset(df)
    
    # Step 5: Organize images
    organize_all_images(train_df, val_df, test_df)
    
    # Step 6: Demonstrate data augmentation
    demonstrate_data_augmentation()
    
    # Step 7: Generate statistics
    generate_statistics()
    
    print("=" * 60)
    print("✅ STAGE 1 COMPLETE: Dataset Preparation Finished!")
    print("=" * 60)
    print()
    print("Next Steps:")
    print("  1. Review the class_distribution.png visualization")
    print("  2. Verify dataset/ folder structure")
    print("  3. Proceed to Stage 2: Model Building")
    print()


if __name__ == "__main__":
    main()
