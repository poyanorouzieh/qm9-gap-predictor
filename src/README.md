# 🧪 QM9 GAP Predictor

![Language](https://img.shields.io/badge/language-Python%203.10%2B-blue?style=for-the-badge&logo=python)
![Domain](https://img.shields.io/badge/domain-Machine%20Learning%20%7C%20Cheminformatics-purple?style=for-the-badge)
![Dataset](https://img.shields.io/badge/dataset-QM9-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/status-active-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

A machine learning pipeline for predicting the **HOMO–LUMO energy gap** of organic molecules using the **QM9 quantum chemistry dataset**. The project explores multiple regression models — from polynomial Ridge regression to ensemble methods — to find the best predictor of molecular electronic properties.

---

## 📖 Overview

The **HOMO–LUMO gap** is a fundamental quantum-chemical property that determines a molecule's reactivity, stability, and optical behavior. Computing it requires expensive DFT calculations, making fast and accurate ML surrogates highly valuable in drug discovery and materials science.

This project:

1. **Parses** the raw QM9 `.xyz` files into a clean structured CSV dataset
2. **Trains** multiple regression models to predict the gap
3. **Evaluates** them using industry-standard regression metrics (MAE, RMSE, R²)
4. **Saves** the trained models for later inference

---

## 📊 Dataset — QM9

The **QM9 dataset** is a benchmark collection of **133,885 small organic molecules** (up to 9 heavy atoms: C, H, O, N, F) with their DFT-computed properties.

- **Subset used:** 133,660 curated molecules
- **Features available per molecule:** rotational constants (A, B, C), dipole moment (μ), polarizability (α), HOMO/LUMO energies, internal energies (U0, U, H, G), heat capacity (Cv), zero-point vibrational energy (zpve), etc.

> **Source:** Ramakrishnan et al., *Scientific Data* (2014). More info: [figshare.com](https://figshare.com/articles/dataset/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904)

---

## ✨ Features

- 🧹 **Automated data extraction** — parses raw QM9 `.xyz` files into a structured CSV
- 🎯 **Feature engineering** — polynomial features (degree 3) for linear models
- 🤖 **Two trained models** with different trade-offs:
  - **Ridge Regression** (interpretable, fast, R² = 0.60)
  - **Random Forest** (non-linear, high-accuracy, R² ≈ 0.81)
- 📈 **Comprehensive evaluation** — MAE, RMSE, R² on held-out test set
- 💾 **Model persistence** — trained models saved as `.pkl` via `joblib`
- 🔀 **Reproducible** — fixed random seeds for train/test splitting and sampling

---

## 🤖 Models & Results

| Model | Features | Test R² | Notes |
|-------|----------|---------|-------|
| **Ridge Regression** (α=1, poly degree=3) | Polynomial features | **0.5968** | Interpretable, fast to train |
| **Random Forest Regressor** | Raw numeric features | **~0.81** | Best accuracy, non-linear |

### Evaluation Metrics

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of Determination

---

## 📂 Project Structure

```
ML_based_on_QM9/
├── src/
│   ├── CreatDataFrame.py        # Parse QM9 .xyz files into a DataFrame
│   ├── RegressionModel.py       # Ridge + Polynomial pipeline
│   └── RandomForestModel.py     # Random Forest pipeline
├── data/
│   ├── qm9_dataset.csv          # Small sample dataset
│   └── qm9_dataset_full_data_v1.csv   # Full parsed dataset (not tracked)
├── models/
│   ├── LinearRegression(Ridge_alpha=1,degree=3,R=0.5968).pkl
│   └── RandomForestModel.pkl    # (not tracked — too large)
├── docs/                        # Documentation (WIP)
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **pip**

### Installation

```
git clone https://github.com/poyanorouzieh/qm9-gap-predictor.git
cd qm9-gap-predictor
pip install -r requirements.txt
```

### Step 1 — Prepare the Dataset

Place the raw QM9 `.xyz` files under `data/133660_curatedQM9_outof_133885/`, then run:

```
python src/CreatDataFrame.py
```

This will produce `qm9_dataset_full_data_v1.csv`.

### Step 2 — Train a Model

**Option A — Ridge Regression (fast):**
```
python src/RegressionModel.py
```

**Option B — Random Forest (higher accuracy):**
```
python src/RandomForestModel.py
```

### Step 3 — Inspect Results

Trained models are saved to the `models/` folder as `.pkl` files and can be loaded with:

```
import joblib
model = joblib.load("models/RandomForestModel.pkl")
predictions = model.predict(X_new)
```

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| **pandas** | Data loading and manipulation |
| **NumPy** | Numerical operations |
| **scikit-learn** | ML models, pipelines, metrics |
| **joblib** | Model serialization (`.pkl`) |
| **tqdm** | Progress bars for large file parsing |
| **pathlib** | Cross-platform file paths |

---

## 🧠 Technical Highlights

- **Modular pipeline design** — data extraction decoupled from training
- **Reproducibility** — fixed `random_state=42` for consistent splits
- **Memory-conscious** — 50% sampling for faster experimentation
- **Feature leakage prevention** — HOMO/LUMO columns dropped before training
- **Model persistence** — production-ready saved pipelines
- **Path-robust** — uses `pathlib` to avoid hardcoded absolute paths

---

## 🗺️ Future Work

- [ ] Add **XGBoost** and **Gradient Boosting** models
- [ ] Hyperparameter tuning with `GridSearchCV` / `Optuna`
- [ ] Feature importance visualization for Random Forest
- [ ] **SHAP** analysis for model interpretability
- [ ] Cross-validation instead of a single train/test split
- [ ] Deep learning approach with **SchNet** or **DimeNet** on molecular graphs
- [ ] Deploy model as a REST API (FastAPI)
- [ ] Add automated tests with `pytest`

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Poya Norouzieh**
- GitHub: [@poyanorouzieh](https://github.com/poyanorouzieh)
- Email: poyanorouzieh@gmail.com

---

⭐ **If you find this project useful, feel free to star it!**