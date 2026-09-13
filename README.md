# Breast Cancer Classification (Wisconsin Diagnostic Dataset)

A neural network that classifies breast tumors as **benign** or **malignant** from digitized fine needle aspirate (FNA) features, using the Breast Cancer Wisconsin (Diagnostic) Data Set.

## Overview

This project builds a small feedforward neural network in TensorFlow/Keras to perform binary classification on 30 numeric features describing cell nuclei (radius, texture, perimeter, smoothness, concavity, etc.). The pipeline covers data cleaning, a proper train/validation/test split, feature scaling, model training with early stopping, and evaluation with a full set of classification metrics.

## Dataset

- **Source:** [Breast Cancer Wisconsin (Diagnostic) Data Set on Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Samples:** 569
- **Features:** 30 numeric features (mean, standard error, and "worst" values for 10 measurements per cell nucleus)
- **Target:** `diagnosis` — Malignant (M) or Benign (B), encoded as 1/0
- **Class balance:** 357 benign (62.7%) / 212 malignant (37.3%)

The dataset is not included in this repository — see [Reproducing this project](#reproducing-this-project) below for how to get it.

## Approach

1. **Cleaning** — dropped the `id` column and an empty `Unnamed: 32` column; encoded `diagnosis` as binary (0 = benign, 1 = malignant).
2. **Split** — stratified 60% train / 20% validation / 20% test split, so class balance is preserved across all three sets.
3. **Scaling** — `StandardScaler` fit only on the training set, then applied to validation and test to avoid data leakage.
4. **Model** — a small fully connected network:
   - Dense(16, ReLU) → Dense(8, ReLU) → Dense(1, sigmoid)
   - ~641 trainable parameters
5. **Training** — Adam optimizer, binary cross-entropy loss, up to 100 epochs with early stopping on validation loss (patience 10, best weights restored).
6. **Evaluation** — accuracy, confusion matrix, precision/recall/F1 per class, and ROC-AUC.

## Results

| Metric | Score |
|---|---|
| Test accuracy | 97.4% |
| ROC-AUC | see notebook |
| Precision (malignant) | 1.00 |
| Recall (malignant) | 0.93 |
| Precision (benign) | 0.96 |
| Recall (benign) | 1.00 |

**Note on false negatives:** recall on the malignant class is 0.93, meaning a small fraction of malignant cases in the test set were predicted benign. In a real diagnostic setting, this is the error that matters most — a missed malignant case is far costlier than a false alarm. This is flagged here rather than hidden behind the headline accuracy number, and it's the main direction for future improvement (e.g. class weighting, threshold tuning, or cross-validation to check how stable this number is across different splits).


## Reproducing this project

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) and place `breast_cancer_data.csv` in the project root.
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib tensorflow scikit-learn joblib
   ```
3. Run the notebook top to bottom.

## Limitations / next steps

- Single train/val/test split — with only 569 samples, a k-fold cross-validation would give a more reliable estimate of generalization.
- No baseline model (e.g. logistic regression) is included for comparison; given the small feature/sample size, simpler models may perform comparably.
- Class weighting or threshold adjustment could be explored to reduce false negatives on the malignant class.
