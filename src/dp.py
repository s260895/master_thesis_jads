"""Laplace-mechanism noise injection for the differential-privacy sweep.

The total privacy budget epsilon is split across features (basic sequential
composition: each of the d per-feature queries gets eps/d), with per-feature
sensitivity taken as the observed value range. Noise scale per feature:

    b_j = sensitivity_j / (eps / d)
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42


def add_laplace_noise(X: pd.DataFrame, eps: float, rng: np.random.Generator | None = None) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)
    d = X.shape[1]
    eps_per_feature = eps / d
    sensitivity = (X.max() - X.min()).to_numpy()
    scale = np.where(sensitivity > 0, sensitivity / eps_per_feature, 0.0)
    noise = rng.laplace(0.0, scale, size=X.shape)
    return pd.DataFrame(X.to_numpy() + noise, columns=X.columns, index=X.index)
