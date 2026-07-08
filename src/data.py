"""Dataset loading for the DP fraud-detection experiments.

Uses the public ULB credit-card fraud dataset from OpenML (no credentials
needed), mirroring the class-imbalance handling of the thesis pipelines:
majority class downsampled, minority kept, stratified train/test split.
Falls back to a synthetic imbalanced dataset when offline.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
MAJORITY_N = 20_000  # matches the thesis IEEE pipeline downsampling


def load_creditcard(majority_n: int = MAJORITY_N, data_home: str | None = None):
    """Load the ULB credit-card fraud dataset, downsampled like the thesis."""
    ds = fetch_openml("creditcard", version=1, as_frame=True, data_home=data_home)
    df = ds.frame
    df["Class"] = df["Class"].astype(int)

    majority = df[df["Class"] == 0]
    minority = df[df["Class"] == 1]
    majority = majority.sample(
        n=min(majority_n, len(majority)), replace=False, random_state=RANDOM_STATE
    )
    df = pd.concat([majority, minority]).sample(frac=1.0, random_state=RANDOM_STATE)

    X = df.drop(columns=["Class"])
    y = df["Class"].to_numpy()
    return X, y


def load_synthetic(n_samples: int = 20_492):
    """Offline fallback with a similar shape/imbalance to the credit-card data."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=30,
        n_informative=12,
        n_redundant=6,
        weights=[0.976, 0.024],
        flip_y=0.01,
        class_sep=0.8,
        random_state=RANDOM_STATE,
    )
    cols = [f"V{i}" for i in range(1, X.shape[1] + 1)]
    return pd.DataFrame(X, columns=cols), y


def train_test(X, y, test_size: float = 0.2):
    """Stratified split + scaling (scaler fit on train only)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )
    return X_train, X_test, np.asarray(y_train), np.asarray(y_test)
