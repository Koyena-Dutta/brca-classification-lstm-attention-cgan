# 🧬 Breast Cancer Classification using CGAN + LSTM with Attention

> Leveraging Conditional Generative Adversarial Networks and attention-based sequence modeling to tackle class imbalance in clinical cancer diagnostics.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=flat-square&logo=keras)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellowgreen?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 📌 Table of Contents

- [Why This Project Matters](#-why-this-project-matters)
- [Project Overview](#-project-overview)
- [Architecture & Pipeline](#-architecture--pipeline)
- [Repository Structure](#-repository-structure)
- [Tech Stack](#-tech-stack)
- [Setup & Installation](#-setup--installation)
- [Running the Project](#-running-the-project)
- [Results & Evaluation](#-results--evaluation)
- [Future Improvements](#-future-improvements)
- [Skills Demonstrated](#-skills-demonstrated)

---

## ❤️ Why This Project Matters

Breast cancer is one of the most common cancers worldwide, and early, accurate classification — benign vs. malignant — can be the difference between life and death. Machine learning models trained on real-world clinical datasets often struggle with **class imbalance**, where one class (typically malignant) has significantly fewer samples. This imbalance causes models to become biased, predicting the majority class at the expense of catching critical cases.

This project directly addresses that problem by:

- Using a **Conditional GAN (CGAN)** to generate realistic synthetic samples for underrepresented classes — without simply duplicating existing data.
- Feeding the augmented dataset into an **LSTM with a custom Attention mechanism**, allowing the model to focus on the most diagnostically relevant features.

The result is a pipeline that is more **robust, fair, and clinically meaningful** than a standard classifier trained on imbalanced data.

---

## 🔍 Project Overview

This project builds an end-to-end machine learning pipeline for binary classification of breast cancer tumors (benign/malignant) using the **Breast Cancer Wisconsin Dataset** from the UCI Machine Learning Repository.

The core idea is a two-stage approach:

1. **Data Augmentation via CGAN** — Train a conditional generative model to synthesize new, class-conditioned samples that address the imbalance in the original dataset.
2. **Sequence Classification via LSTM + Attention** — Train a recurrent model with an attention layer on the augmented dataset, enabling the model to learn both temporal dependencies and feature importance.

This combination pushes beyond the limitations of simple oversampling techniques like SMOTE while demonstrating the power of deep generative models in structured/tabular healthcare data.

---

## 🏗️ Architecture & Pipeline

The project follows a clean, modular 5-step pipeline:

```
Raw Dataset (UCI)
      │
      ▼
┌─────────────────────┐
│  1. Preprocessing   │  Label encoding + MinMaxScaler normalization
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  2. CGAN Training   │  Generator + Discriminator trained on class-conditioned data
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  3. Data Augmentation│ Synthetic samples added to balance the training set
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  4. LSTM + Attention │  Custom attention layer highlights key diagnostic features
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  5. Evaluation       │  Accuracy, Classification Report, Confusion Matrix
└─────────────────────┘
```

### How Each Component Works

| Stage | Description |
|---|---|
| **Preprocessing** | Categorical labels are encoded numerically; all features are scaled to [0, 1] using `MinMaxScaler` to stabilize training |
| **CGAN** | The Generator learns to create synthetic feature vectors conditioned on a class label; the Discriminator learns to distinguish real from fake samples |
| **Augmentation** | Synthetic samples from the CGAN are appended to the training set to achieve a balanced class distribution |
| **LSTM + Attention** | Input features are reshaped into sequences; the LSTM captures dependencies across features, and the custom Attention layer assigns learned weights to time steps |
| **Evaluation** | Model is assessed on held-out test data using standard classification metrics |

---

## 📁 Repository Structure

```
breast-cancer-cgan-lstm/
│
├── notebooks/
│   └── full_experiment.ipynb       # End-to-end experimentation notebook
│
├── src/
│   ├── preprocessing.py            # Data loading, encoding, and scaling
│   ├── cgan.py                     # CGAN architecture (Generator + Discriminator)
│   └── model.py                    # LSTM model with custom Attention layer
│
├── data/
│   └── dataset_link.txt            # Link to UCI Breast Cancer Wisconsin Dataset
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Libraries / Tools |
|---|---|
| **Language** | Python 3.8+ |
| **Deep Learning** | TensorFlow, Keras |
| **Data Handling** | NumPy, Pandas |
| **ML Utilities** | Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/breast-cancer-cgan-lstm.git
cd breast-cancer-cgan-lstm
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

The dataset used is the **Breast Cancer Wisconsin (Diagnostic) Dataset** from the UCI ML Repository.

👉 See `data/dataset_link.txt` for the download link.

Place the downloaded `.csv` file in the `data/` directory before running.

---

## ▶️ Running the Project

### Option A — Jupyter Notebook (Recommended for exploration)

```bash
jupyter notebook notebooks/full_experiment.ipynb
```

### Option B — Run Modular Scripts

```bash
# Step 1: Preprocess data
python src/preprocessing.py

# Step 2: Train the CGAN
python src/cgan.py

# Step 3: Train and evaluate LSTM + Attention
python src/model.py
```

---

## 📊 Results & Evaluation

The model is evaluated on a held-out test set using the following metrics:

| Metric | Description |
|---|---|
| **Accuracy** | Overall percentage of correct predictions |
| **Precision** | Of all predicted positives, how many were truly positive |
| **Recall** | Of all actual positives, how many were correctly identified |
| **F1-Score** | Harmonic mean of precision and recall |
| **Confusion Matrix** | Visual breakdown of true vs. predicted class labels |

> 📌 Detailed results, training curves, and confusion matrix visualizations are available inside the notebook: `notebooks/full_experiment.ipynb`

---

## 🚀 Future Improvements

- [ ] **Hyperparameter tuning** — Systematic search (e.g., Optuna or Keras Tuner) for optimal CGAN and LSTM configurations
- [ ] **WGAN-GP** — Upgrade the CGAN to a Wasserstein GAN with Gradient Penalty for more stable training
- [ ] **Transformer-based classifier** — Replace the LSTM with a lightweight Transformer encoder for potentially stronger performance
- [ ] **Cross-validation** — Use k-fold cross-validation for more reliable performance estimates
- [ ] **Explainability (XAI)** — Integrate SHAP or LIME to explain individual predictions for clinical interpretability
- [ ] **Deployment** — Wrap the trained model in a Flask/FastAPI service for real-time inference

---

## 💡 Skills Demonstrated

This project showcases a range of practical and advanced skills relevant to **machine learning engineering**, **healthcare AI**, and **research**:

- ✅ **Generative Adversarial Networks** — Designed and trained a CGAN for structured/tabular data augmentation
- ✅ **Recurrent Neural Networks** — Built and trained an LSTM model on augmented sequential feature data
- ✅ **Custom Keras Layers** — Implemented a custom Attention mechanism from scratch using the Keras API
- ✅ **Handling Class Imbalance** — Applied a deep generative approach rather than naive oversampling
- ✅ **End-to-End ML Pipeline** — Covered preprocessing, model training, augmentation, and evaluation
- ✅ **Data Preprocessing** — Applied label encoding and MinMax scaling for neural network compatibility
- ✅ **Model Evaluation** — Used accuracy, classification reports, and confusion matrices for comprehensive assessment
- ✅ **Modular Code Design** — Structured project with separation of concerns across `src/` modules
- ✅ **Healthcare Domain Awareness** — Framed and approached the problem with clinical relevance in mind

---

## 📄 Dataset Reference

**Breast Cancer Wisconsin (Diagnostic) Data Set**
UCI Machine Learning Repository
[https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

> W.N. Street, W.H. Wolberg, and O.L. Mangasarian. Nuclear feature extraction for breast tumor diagnosis. *IS&T/SPIE 1993 International Symposium on Electronic Imaging*, 1993.

---

## 🤝 Contributing

Contributions, suggestions, and feedback are welcome. Feel free to open an issue or submit a pull request.

---

<p align="center">Made with curiosity, code, and a bit of adversarial training ⚡</p>
