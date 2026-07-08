# Differential Privacy vs Membership-Inference Attacks in Fraud Detection

**MSc thesis code — JADS (Jheronimus Academy of Data Science), Data Science & Entrepreneurship, 2022.**

This project quantifies the **privacy–utility tradeoff of differential privacy (DP)** applied to machine-learning fraud detection: how much predictive performance is lost, and how much protection against membership-inference attacks is gained, as the privacy budget ε varies.

## Research setup

**Datasets (two parallel pipelines):**
- `au_notebook.ipynb` — a proprietary equipment-finance fraud dataset ("AU"/internal), including feature engineering with address geocoding (Nominatim / Bing / HERE) to derive residential-vs-commercial address fraud signals.
- `ieee_fraud_dataset.ipynb` — the public [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset from Kaggle.

**Method:**
1. Train six classifiers (Random Forest, AdaBoost, Gaussian NB, Decision Tree, kNN, RBF SVM) on each dataset.
2. Apply the **Laplace mechanism** to the data across a sweep of privacy budgets (ε from 0.01 to 10,000), retraining and re-evaluating at each level (ROC-AUC, PR-AUC, F1, precision, recall, accuracy + feature-importance drift). IBM's [diffprivlib](https://github.com/IBM/differential-privacy-library) DP models are included for comparison.
3. Run **Salem** and **Yeom** membership-inference attacks against the trained models at each ε to measure how DP noise degrades an attacker's ability to tell whether a record was in the training set.
4. Compare classifiers across datasets with **Friedman/Nemenyi critical-difference diagrams** (Demšar, 2006).

## Key results

- On IEEE-CIS, Random Forest performs best (ROC-AUC 0.828, F1 0.755); on the internal dataset, AdaBoost leads (ROC-AUC 0.785). See `res_ieee.csv` / `res_au.csv`.
- The ε sweep artifacts (`df_dp_<ε>_*.csv`, `epsilon_*` prediction tables) concretely show the tradeoff: at ε = 0.1 the injected noise destroys feature utility, while at ε = 1000 the data is barely perturbed — and attack success moves accordingly.

## Repository layout

| Files | Contents |
|---|---|
| `au_notebook.ipynb`, `ieee_fraud_dataset.ipynb` | the two experiment pipelines |
| `res_*.csv` | classifier performance tables and critical-difference inputs |
| `df_dp_<ε>_{au,ieee}.csv` | sample data after DP noise at each ε |
| `epsilon_*`, `int_epsilon_*`, `sample_data_*` | per-ε prediction tables (thesis figures) |
| `au_unique_asset_address_info_dict.pickle` | cached geocoding responses for the AU dataset |

## Running

```bash
pip install -r requirements.txt
```

The IEEE pipeline needs `train_transaction.csv` from the [Kaggle competition](https://www.kaggle.com/c/ieee-fraud-detection/data); the AU pipeline requires the proprietary source spreadsheet (not distributed). Update the data paths at the top of each notebook.

## Reproducible v2 experiment + interactive dashboard

The `src/` package re-implements the thesis experiment on a fully public dataset (the [ULB credit-card fraud dataset](https://www.openml.org/d/1597) via OpenML — no credentials needed), so the whole privacy–utility sweep is reproducible end-to-end:

- `src/data.py` — dataset loading with thesis-style majority downsampling (synthetic offline fallback included)
- `src/dp.py` — Laplace mechanism with the budget split across features
- `src/models.py` — the classifier zoo (Random Forest, AdaBoost, GaussianNB, Decision Tree, kNN)
- `src/attacks.py` — Salem and Yeom membership-inference attacks, scored by attack ROC-AUC
- `src/experiment.py` — the full ε sweep, writing `results/dp_sweep_results.csv`

```bash
python -m src.experiment          # regenerate results (~1 min)
streamlit run dp_dashboard.py     # explore them interactively
```

The dashboard lets you slide ε and watch classifier utility and membership-attack success move against each other, including a privacy–utility Pareto view. Headline from the included run: at ε ≤ 10 the Laplace noise reduces every model to chance (F1 = 0) while attacks drop to ~0.5 AUC; the clean baseline reaches ROC-AUC 0.977 — with the highest membership leakage (Yeom attack AUC 0.546).

## Docker

```bash
docker compose up --build -d      # dashboard at http://localhost:8501
docker compose --profile tools run --rm experiment   # regenerate results/
docker compose down --rmi local -v                   # full teardown
```
