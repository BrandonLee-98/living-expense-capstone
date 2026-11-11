# 🏙️ Non-Mortgage Living Expense Estimation Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-F37626)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57)
![License](https://img.shields.io/badge/License-MIT-green)

An interactive, AI-driven web application designed to predict and break down non-mortgage homeownership costs across **Harris, Fort Bend, and Galveston Counties**. Built as the final Capstone project for the Houston City College Artificial Intelligence program (Fall 2026).

---

## 👥 Team Members (Group 5)
* **Brandon Lee Matias** – Data & Machine Learning Lead / Full-Stack Developer
* **Jonah Joseph** – ML Engineering & Spatial Arbitrage Analytics
* **Khadijah Patience** – Data Engineering & Baseline Development
* **Taki Eddine Boubekri** – UI Integration & Pipeline Engineering

---

## 🚀 Project Overview

Standard mortgage calculators severely underestimate the true cost of homeownership by ignoring geographical tax step-functions, utility grid delivery rates, and actuarial disaster risk. 

The **Living Expense Estimation Engine** solves this by fusing **Machine Learning (XGBoost)** with **Deterministic Actuarial Rules** and **Spatial Proximity Math**. Users input a ZIP code and household parameters, and the engine outputs a highly accurate, audit-defensible monthly budget for Groceries, Electricity, Water, Natural Gas, HOA fees, and Mandatory Windstorm Insurance (TWIA).

### ✨ Core Features
* **Multi-Target Predictive ML Engine:** Uses a hyper-tuned XGBoost `MultiOutputRegressor` to simultaneously predict 5 continuous expense targets with an average Mean Absolute Error (MAE) of just **\$0.30/month**.
* **Spatial Arbitrage Engine:** Employs the Haversine Great-Circle distance algorithm to scan neighboring ZIP codes. If a user selects a home in a heavy tax zone (e.g., MUD water taxes or coastal TWIA surcharges), the app automatically recommends cheaper, nearby alternative ZIP codes.
* **Hybrid Deterministic Governance:** Ensures regulatory financial accuracy by hardcoding strict policies—such as the exact \$125/mo TWIA coastal surcharge and official USDA grocery scaling formulas—on top of the ML predictions.
* **Explainable AI (XAI):** Integrated SHAP (SHapley Additive exPlanations) visualizations to open the "black box," proving exactly how many dollars each variable (like square footage or occupant count) adds to the baseline cost.
* **Lightning-Fast Backend:** Driven by a parameterized **SQLite database**, ensuring secure, sub-150ms inference and data retrieval.

---

## 🧠 System Architecture & Tech Stack

| Component | Technology Used | Description |
| :--- | :--- | :--- |
| **Frontend UI** | Streamlit | Responsive, 7-tab web dashboard with dynamic data visualization. |
| **Backend Data** | SQLite & Pandas | Fast, parameterized SQL queries pulling from `living_costs.db`. |
| **Machine Learning** | XGBoost & Scikit-Learn | Tree-based ensemble modeling optimized via `GridSearchCV`. |
| **Interpretability** | SHAP (`TreeExplainer`) | Game-theoretic metric breakdown of feature dollar impacts. |
| **Serialization** | Joblib | Lightweight model caching (`xgboost_cost_model.joblib`). |

---

## 📂 Repository Structure

```text
living-expense-capstone/
│
├── app.py                            # Main Streamlit application and UI routing
├── living_costs.db                   # SQLite database (Contains 279 ZIP code profiles)
├── xgboost_cost_model.joblib         # Serialized MultiOutputRegressor model pipeline
├── requirements.txt                  # Python dependencies for cloud deployment
├── README.md                         # Project documentation
│
└── figures/                          # Directory for XAI and diagnostic images
    ├── metric_summary.png
    ├── shap_importance_all_targets.png
    ├── shap_beeswarm_food.png
    ├── residual_histograms.png
    └── ... (other diagnostic PNGs)
```

## 📊 Model Performance & Tuning
The core XGBoost model was hyper-tuned across 2,790 simulated testing scenarios.

* **Final Architecture:** 150 Estimators, Max Depth of 8, Learning Rate of 0.08, Subsample 1.0.

* **Validation Score:** Achieved a Cross-Validation MAE of $0.30/month (reduced from a default of $14.50).

## 🎓 Academic Context
This repository represents Phase 04 (Integrated System Prototype) of the AI Capstone lifecycle at Houston City College. It satisfies all requirements for backend database implementation, component development, multi-target predictive modeling, and cloud deployment preparation.
