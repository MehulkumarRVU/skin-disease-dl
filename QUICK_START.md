# 🚀 QUICK START - Stage 1

## What You Need to Do Now:

### Step 1: Download HAM10000 Dataset 
**Kaggle (Easiest):**
1. Go to: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. Sign in (free account)
3. Click "Download" button
4. Extract the ZIP file

### Step 2: Organize Files
Place in your project folder:
```
skin-disease-dl/
├── HAM10000_images/           ← All 10,015 .jpg images here
└── HAM10000_metadata.csv      ← The CSV file here
```

**Note:** If images are split into `part_1` and `part_2` folders, merge them into one `HAM10000_images/` folder.

### Step 3: Run Preparation Script
```bash
python prepare_dataset.py
```

⏱️ **Time:** 5-10 minutes  
💾 **Output:** ~10,000 organized images in `dataset/` folder

### Step 4: Verify Success
✅ See "STAGE 1 COMPLETE" message  
✅ Check `class_distribution.png` created  
✅ Verify `dataset/train/`, `dataset/val/`, `dataset/test/` have images

---

## 📊 What Happens During Execution:

```
[Step 1] Creating directory structure...        ✓
[Step 2] Loading metadata...                    ✓
[Step 3] Analyzing class distribution...        ✓
[Step 4] Splitting dataset (70/15/15)...        ✓
[Step 5] Organizing 10,015 images...            ✓
        - Resizing to 224x224
        - Converting to RGB
        - Organizing by class
[Step 6] Configuring data augmentation...       ✓
[Step 7] Generating statistics...               ✓
```

---

## 🔍 Expected Results:

### Dataset Distribution:
- **Training:** ~7,010 images
- **Validation:** ~1,502 images
- **Testing:** ~1,503 images

### Class Breakdown (approximate):
- Melanocytic Nevus: ~6,700 (66%)
- Melanoma: ~1,100 (11%)
- Benign Keratosis: ~1,100 (11%)
- Basal Cell Carcinoma: ~510 (5%)
- Actinic Keratosis: ~330 (3%)
- Vascular Lesion: ~140 (1%)
- Dermatofibroma: ~115 (1%)

---

## ❓ Troubleshooting:

**Problem:** "Metadata file not found"
- **Fix:** Ensure `HAM10000_metadata.csv` is in project root folder

**Problem:** "Raw dataset folder not found"  
- **Fix:** Create folder named `HAM10000_images` with all images

**Problem:** Script runs but says "0 images processed"
- **Fix:** Check image files have `.jpg` extension

---

## ✅ When Stage 1 is Complete:

**Reply: "Ready for Stage 2"**

I'll then provide:
- Model architecture code
- Transfer learning implementation  
- Explanation of design choices

---

## 📚 Key Concepts Learned in Stage 1:

1. ✅ **Dataset Organization**: Structured folders for Keras
2. ✅ **Data Splitting**: Train/Val/Test separation
3. ✅ **Stratification**: Maintaining class balance
4. ✅ **Image Preprocessing**: Standardization (224x224 RGB)
5. ✅ **Data Augmentation**: Synthetic data generation
6. ✅ **Class Imbalance**: Understanding real-world medical data

---

**Need help?** Check [STAGE_1_INSTRUCTIONS.md](STAGE_1_INSTRUCTIONS.md) for detailed troubleshooting!
