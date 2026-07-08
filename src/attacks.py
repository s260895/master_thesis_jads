"""Membership-inference attacks used in the thesis (Salem & Yeom variants).

Both attacks receive a fitted classifier plus equal-sized member (train) and
non-member (test) samples, and are scored by ROC-AUC of the attacker's
membership score: 0.5 = no leakage, 1.0 = full leakage.

- Salem et al. (2019), attack 3: membership score = max posterior confidence.
- Yeom et al. (2018): membership score = -(per-sample log-loss).
"""

import numpy as np
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42
_EPS = 1e-12


def _membership_sets(X_train, y_train, X_test, y_test, rng):
    """Equal-sized member/non-member samples for a balanced attack evaluation."""
    n = min(len(X_train), len(X_test))
    train_idx = rng.choice(len(X_train), size=n, replace=False)
    test_idx = rng.choice(len(X_test), size=n, replace=False)
    return (
        X_train.iloc[train_idx],
        np.asarray(y_train)[train_idx],
        X_test.iloc[test_idx],
        np.asarray(y_test)[test_idx],
    )


def salem_attack_auc(clf, X_train, y_train, X_test, y_test, rng=None) -> float:
    rng = rng or np.random.default_rng(RANDOM_STATE)
    Xm, _, Xn, _ = _membership_sets(X_train, y_train, X_test, y_test, rng)
    score_member = clf.predict_proba(Xm).max(axis=1)
    score_nonmember = clf.predict_proba(Xn).max(axis=1)
    scores = np.concatenate([score_member, score_nonmember])
    labels = np.concatenate([np.ones(len(Xm)), np.zeros(len(Xn))])
    return roc_auc_score(labels, scores)


def yeom_attack_auc(clf, X_train, y_train, X_test, y_test, rng=None) -> float:
    rng = rng or np.random.default_rng(RANDOM_STATE)
    Xm, ym, Xn, yn = _membership_sets(X_train, y_train, X_test, y_test, rng)

    def neg_log_loss(X, y):
        proba = np.clip(clf.predict_proba(X), _EPS, 1 - _EPS)
        return np.log(proba[np.arange(len(y)), y.astype(int)])

    scores = np.concatenate([neg_log_loss(Xm, ym), neg_log_loss(Xn, yn)])
    labels = np.concatenate([np.ones(len(Xm)), np.zeros(len(Xn))])
    return roc_auc_score(labels, scores)
