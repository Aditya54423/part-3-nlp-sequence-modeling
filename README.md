# Part 1: Customer Churn Prediction using Neural Networks

Dataset Folder Source:- https://drive.google.com/drive/folders/1Aihn49cUYMjCgeCTFBTyprjrgZO3UY6r?usp=drive_link
customer_churn_nn.csv:https://drive.google.com/file/d/1VS4UAHiS25bCLSkOAtttfn-_S-6BvL0o/view?usp=drive_link
data_dictionary.md:https://drive.google.com/file/d/1u75TF3jq0AHPH7o4CFuu3FAOoUCWU7Du/view?usp=drive_link
## Project Overview
This project focuses on building, tuning, and evaluating a Feed-Forward Neural Network (FFNN) to predict customer churn. The primary challenge addressed in this notebook is **severe class imbalance** (98.5% retained vs. 1.5% churned), requiring specialized preprocessing, class weighting, and evaluation metrics (AUC-ROC and Recall) to build a commercially viable retention model.

## Dataset & Preprocessing
* **Data Retrieval:** Data is fetched directly into memory from Google Drive via HTTP requests (avoiding local disk clutter).
* **Shape:** 2,000 rows and 16 predictive features (after dropping `customer_id`).
* **Encoding:** Categorical features (`region`, `plan_type`, `contract_type`, `payment_method`) were transformed using Label Encoding to manage feature space effectively.
* **Scaling:** `StandardScaler` was applied to numerical features (fit on the training set to prevent data leakage) to prevent features like `monthly_charges_inr` from dominating gradient updates.
* **Imbalance Handling:** Computed balanced class weights, making the network treat each minority churn case as ~63x more important during backpropagation.

## Neural Network Architecture
A dynamic, modular factory function was used to build the Sequential model. The baseline architecture consists of:
1. **Input Layer:** 15 preprocessed features.
2. **Hidden Layer 1:** Dense (64 units, ReLU) + BatchNormalization + Dropout (0.3).
3. **Hidden Layer 2:** Dense (32 units, ReLU) + BatchNormalization + Dropout (0.3).
4. **Output Layer:** Dense (1 unit, Sigmoid) for probability mapping.

**Callbacks Configured:** * `EarlyStopping` (monitor='val_auc', patience=15, restore_best_weights=True)
* `ReduceLROnPlateau` (factor=0.5, patience=8)

## Hyperparameter Experimentation
To systematically improve the model, 5 distinct experiments were conducted by isolating variables against the baseline:

| Exp | Description | AUC | Recall | Observation |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Baseline (64-32, LR=0.001, Batch=32) | 0.8473 | 0.8333 | Steady learning, safe calibration. |
| **2** | Deeper Network (128-64-32) | 0.8572 | 1.0000 | Perfect recall but sharp drop in precision (too many false alarms). |
| **3** | High Learning Rate (LR=0.01) | 0.7618 | 0.3333 | Worst result; overshot optimal weights due to imbalanced loss landscape. |
| **4** | **Small Batch Size (Batch=8)** | **0.9107** | **1.0000** | **Best overall configuration.** More frequent updates helped the model learn the minority class reliably. |
| **5** | Tanh Activation | 0.8964 | 1.0000 | Zero-centered outputs performed well, but slightly lower AUC than Exp 4. |

## Reproducibility Note (Apple Silicon)
This notebook was executed on an Apple M2 machine utilizing the TensorFlow Metal backend. Due to hardware-level floating-point non-determinism, slight variations in recall (±0.1667) may occur between runs, given the extremely small number of test churners (n=6). The AUC metric remains highly stable.

##  How to Run
1. Ensure dependencies are installed: `pip install -r requirements.txt`
2. Open the notebook: `jupyter notebook notebook.ipynb`
3. Run all cells. The dataset is fetched automatically via URL.

---
