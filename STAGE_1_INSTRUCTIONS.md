# STAGE 1: Dataset Preparation - Instructions

## 📥 Step 1: Download HAM10000 Dataset

### Option A: Kaggle (Recommended)
1. Go to: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. Click "Download" (requires Kaggle account - free)
3. You'll get a ZIP file (~5GB)

### Option B: Official Source
- Visit: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

---

## 📂 Step 2: Extract and Organize

After downloading, extract the files. You should have:
- `HAM10000_images/` (or two folders with images split)
- `HAM10000_metadata.csv` (contains labels)

### Required Structure:
```
skin-disease-dl/
│
├── prepare_dataset.py          ← Our script
├── HAM10000_metadata.csv        ← Place here!
├── HAM10000_images/             ← Place images here!
│   ├── ISIC_0024306.jpg
│   ├── ISIC_0024307.jpg
│   └── ... (10,015 images)
│
└── dataset/                     ← Will be populated by script
    ├── train/
    ├── val/
    └── test/
```

**Note:** If your images are in two folders (`HAM10000_images_part_1/` and `HAM10000_images_part_2/`), merge them into one `HAM10000_images/` folder.

---

## ▶️ Step 3: Run the Script

```bash
python prepare_dataset.py
```

### What Happens:
1. ✅ Creates organized folder structure
2. ✅ Loads metadata and validates
3. ✅ Shows class distribution (saves visualization)
4. ✅ Splits data 70/15/15
5. ✅ Resizes and organizes 10,000+ images (takes 5-10 minutes)
6. ✅ Generates statistics

---

## 📊 Expected Output

### Console Output:
```
============================================================
SKIN DISEASE CLASSIFICATION - DATASET PREPARATION
Stage 1: HAM10000 Dataset Organization
============================================================

STEP 1: Creating Directory Structure
============================================================
✓ Created directory structure at: dataset
  - Splits: ['train', 'val', 'test']
  - Classes per split: 7

STEP 2: Loading Metadata
============================================================
✓ Loaded metadata: 10015 images
  Columns: ['lesion_id', 'image_id', 'dx', 'dx_type', ...]

STEP 3: Analyzing Class Distribution
============================================================

Class Distribution:
------------------------------------------------------------
  melanocytic_nevus          :  6705 images (66.95%)
  melanoma                   :  1113 images (11.11%)
  benign_keratosis           :  1099 images (10.97%)
  basal_cell_carcinoma       :   514 images ( 5.13%)
  actinic_keratosis          :   327 images ( 3.26%)
  vascular_lesion            :   142 images ( 1.42%)
  dermatofibroma             :   115 images ( 1.15%)

Total Images: 10015
Number of Classes: 7

✓ Class distribution plot saved: class_distribution.png

...
```

### Generated Files:
- `class_distribution.png` - Bar chart showing class imbalance
- `dataset/` populated with organized images

---

## ⚠️ Important Notes

### Class Imbalance:
Notice that **melanocytic nevus** (benign moles) makes up ~67% of data while **dermatofibroma** is only ~1%. This is realistic medical data - we'll handle this with:
- Data augmentation (Stage 1) ✓
- Class weights (Stage 3)
- Balanced evaluation metrics (Stage 4)

### Storage Requirements:
- Original dataset: ~5 GB
- Organized dataset: ~5 GB (additional)
- **Total needed: ~10 GB free space**

### Time Estimate:
- Download: 5-15 minutes (depends on internet)
- Script execution: 5-10 minutes
- **Total: ~20-30 minutes**

---

## ✅ Verification

After running, verify:

1. **Check folder structure:**
   ```bash
   dataset/
   ├── train/
   │   ├── actinic_keratosis/      (~229 images)
   │   ├── basal_cell_carcinoma/   (~360 images)
   │   ├── benign_keratosis/       (~769 images)
   │   ├── dermatofibroma/         (~81 images)
   │   ├── melanoma/               (~779 images)
   │   ├── melanocytic_nevus/      (~4694 images)
   │   └── vascular_lesion/        (~99 images)
   ├── val/ (similar structure)
   └── test/ (similar structure)
   ```

2. **Open class_distribution.png** - should see a clear bar chart

3. **Check console output** shows "✅ STAGE 1 COMPLETE"

---

## 🐛 Troubleshooting

### Error: "Metadata file not found"
- Ensure `HAM10000_metadata.csv` is in project root
- Check exact filename (case-sensitive)

### Error: "Raw dataset folder not found"
- Ensure folder is named `HAM10000_images`
- Check images are extracted (not still in .zip)
- Update `RAW_DATASET_PATH` variable if using different name

### Images not copying:
- Verify images have extensions: `.jpg`, `.jpeg`, or `.png`
- Check file permissions
- Ensure enough disk space

---

## 📚 Academic Context

### Why This Organization?
- **Separate test set**: Ensures unbiased final evaluation
- **Validation set**: Prevents overfitting during training
- **Class folders**: Standard format for Keras ImageDataGenerator
- **Standardized size**: Required by MobileNetV2 architecture

### Key Concepts Demonstrated:
1. **Data preprocessing pipeline**
2. **Stratified splitting** (preserves class distribution)
3. **Data augmentation** (synthetic data generation)
4. **Reproducibility** (random seed = 42)

---

## ➡️ Next Stage

Once Stage 1 completes successfully:
- ✅ Dataset is ready
- ✅ You understand the data distribution
- ✅ Ready for Stage 2: Model Building

**Reply with "Ready for Stage 2"** when complete!
