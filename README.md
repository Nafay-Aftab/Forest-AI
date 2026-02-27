# 🌲 ForestAI: Spatial Cover Intelligence Engine

## 📌 Executive Summary

ForestAI is an end-to-end Machine Learning pipeline and decoupled microservices web application designed to predict the dominant tree species (7 distinct classes) across the Roosevelt National Forest based on 54 topographical, hydrological, and radiometric features.

Trained on a massive, highly imbalanced dataset of ~581,000 spatial records, the system acts as a high-speed inference engine for geospatial analysis. It identifies hyper-rare vegetation zones (comprising less than 0.5% of the forest) with extreme precision, achieving a **0.947 Matthews Correlation Coefficient (MCC)** with a sub-second inference latency of **3.8 microseconds per prediction**.

📁 **Project Assets (Screenshots, Artifacts & Outputs):** [Google Drive Folder](https://drive.google.com/drive/folders/1lPpY4i4n15HuY8z5h1cDx63TErgKfHV3?usp=sharing)

![ForestAI UI](<img width="1917" height="865" alt="image" src="https://github.com/user-attachments/assets/22e6dc55-e350-470a-a39a-e9b2f1f7e853" />
)

---

## 🧠 Architectural Decisions & Engineering Process

This project prioritizes mathematical rigor, latency optimization, and robust ML engineering over blind algorithm application.

**Data Leakage Shielding:** Built an impermeable scikit-learn `Pipeline` utilizing `PowerTransformer` (Yeo-Johnson) to stabilize severe skewness in continuous distance metrics, followed by a strict `StandardScaler`. `fit_transform()` was strictly isolated to the 80% training fold to mathematically guarantee zero data leakage into the test set.

**Domain Feature Engineering:** Engineered custom spatial features (`Euclidean_Distance_To_Hydrology`, `Water_Elevation`). Explainable AI (XAI) extraction later proved that `Water_Elevation` organically outranked the raw `Elevation` data in XGBoost's Gain metrics, validating the physical intuition behind the engineering.

**Solving Extreme Class Imbalance (No SMOTE):** The dataset contained a severe imbalance, with Class 4 (Cottonwood/Willow) representing only 0.47% of the data. Instead of generating noisy synthetic data via SMOTE, the pipeline dynamically injected algorithmic sample weights (`compute_sample_weight('balanced')`) into the loss function, mathematically forcing the algorithm to penalize minority class errors heavily. This rescued Class 4 Recall to an elite **0.92**.

**Bayesian Optimization (Optuna):** Executed a highly constrained Bayesian search to tune the GPU-accelerated XGBoost champion. Enforced a `colsample_bytree=0.60` (feature dropout) to prevent the model from lazily overfitting to `Elevation`, forcing it to build deep, 14-level logic trees across wilderness and soil binary flags.

**Microservices Architecture:** Scrapped the monolithic script approach. The project is deployed as a two-tier microservice: a FastAPI backend that safely isolates the `.joblib` artifacts and memory footprint, communicating via REST with a lightweight Streamlit React frontend.

---

## 📊 Production Metrics (20% Stratified Holdout)

| Metric | Value | Business Translation |
|---|---|---|
| MCC (Matthews Correlation) | 0.9474 | Flawless performance across both dominant and minority classes. |
| ROC-AUC (Multi-Class) | 0.9989 | Near-perfect probability ranking and class separation. |
| Log Loss | 0.0993 | High mathematical confidence; the model is rarely "guessing". |
| Class 4 (0.47%) Recall | 0.9200 | Successfully hunts down the rarest trees in the forest. |
| Batch Inference Latency | 0.44 sec | Processed 116,203 test rows in under half a second on GPU. |

> The model successfully proved an exceptionally low generalization gap (**Train MCC: 0.9906** vs **Test MCC: 0.9474**), proving zero catastrophic overfitting.

---

## 💻 The Microservices Architecture

The system is split into two robust components:

### 1. The Inference Engine (FastAPI)
- Strictly typed payload validation via **Pydantic** guarantees the model never crashes due to invalid frontend inputs.
- Calculates engineered features dynamically on the fly before passing the tensor through the Scikit-Learn preprocessing pipeline.

### 2. The Presentation Layer (Streamlit)
- Premium, custom-injected CSS featuring a glassmorphic **"Dark Forest Biome"** UI.
- Displays dynamic probability distributions, live metric derivations, and intuitive confidence scoring.
- Operates statelessly, safely catching API connection errors without exposing traceback logic to the user.

---

## 🚀 Installation & Usage

**1. Clone the repository:**
```bash
git clone https://github.com/Nafay-Aftab/Forest-AI.git
cd Forest-AI
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Boot the FastAPI Backend (Terminal 1):**
```bash
uvicorn fast_api:app --reload
```
The API will boot locally at `http://127.0.0.1:8000`

**4. Boot the Streamlit Frontend (Terminal 2):**
```bash
streamlit run app.py
```
The UI will launch locally at `http://localhost:8501`

---

## 📬 Contact & Author

**Muhammad Nafay Aftab**
Aspiring AI Engineer | BSCS
[LinkedIn Profile](#)
