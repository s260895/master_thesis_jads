"""Run the full DP privacy-utility sweep and write results/dp_sweep_results.csv.

For each privacy budget epsilon: noise the train AND test sets with the
Laplace mechanism, retrain every classifier, record utility metrics on the
noised test set, and score Salem/Yeom membership-inference attacks against
the retrained model. eps = inf is the clean-data baseline.

Usage:
    python -m src.experiment [--synthetic]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .attacks import salem_attack_auc, yeom_attack_auc
from .data import load_creditcard, load_synthetic, train_test
from .dp import add_laplace_noise
from .models import get_classifiers

EPSILONS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, float("inf")]
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_sweep(X, y, epsilons=EPSILONS) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test(X, y)
    rows = []
    for eps in epsilons:
        if np.isinf(eps):
            Xtr, Xte = X_train, X_test
        else:
            rng = np.random.default_rng(42)
            Xtr = add_laplace_noise(X_train, eps, rng)
            Xte = add_laplace_noise(X_test, eps, rng)

        for name, clf in get_classifiers().items():
            t0 = time.time()
            clf.fit(Xtr, y_train)
            y_pred = clf.predict(Xte)
            y_proba = clf.predict_proba(Xte)[:, 1]
            rows.append(
                {
                    "epsilon": eps,
                    "classifier": name,
                    "roc_auc": roc_auc_score(y_test, y_proba),
                    "pr_auc": average_precision_score(y_test, y_proba),
                    "f1": f1_score(y_test, y_pred, zero_division=0),
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall": recall_score(y_test, y_pred, zero_division=0),
                    "accuracy": accuracy_score(y_test, y_pred),
                    "salem_attack_auc": salem_attack_auc(clf, Xtr, y_train, Xte, y_test),
                    "yeom_attack_auc": yeom_attack_auc(clf, Xtr, y_train, Xte, y_test),
                    "fit_seconds": round(time.time() - t0, 2),
                }
            )
            print(f"eps={eps:>8} {name:<14} done ({rows[-1]['fit_seconds']}s)")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="skip the OpenML download")
    args = parser.parse_args()

    if args.synthetic:
        X, y = load_synthetic()
        source = "synthetic"
    else:
        try:
            X, y = load_creditcard()
            source = "creditcard-openml"
        except Exception as exc:  # offline fallback
            print(f"OpenML fetch failed ({exc}); using synthetic data")
            X, y = load_synthetic()
            source = "synthetic"

    results = run_sweep(X, y)
    results["dataset"] = source
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "dp_sweep_results.csv"
    results.to_csv(out, index=False)
    print(f"\nWrote {len(results)} rows to {out}")


if __name__ == "__main__":
    main()
