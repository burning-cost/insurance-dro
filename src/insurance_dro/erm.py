"""
Deterministic ERM (Empirical Risk Minimisation) baseline premium.

Standard least-squares regression — minimise expected squared loss on training data.
No robustness. The reference model against which DRO is compared.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class DeterministicERM(BaseEstimator):
    """Standard ERM premium estimator (OLS baseline).

    Parameters
    ----------
    fit_intercept : bool
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self._scaler = StandardScaler()
        self._model: LinearRegression | None = None

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
    ) -> "DeterministicERM":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        X_scaled = self._scaler.fit_transform(X)
        self._model = LinearRegression(fit_intercept=self.fit_intercept)
        self._model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        X = np.asarray(X, dtype=float)
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)

    def worst_case_loss(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
        perturbation_std: float = 0.1,
        n_samples: int = 200,
        seed: int = 42,
    ) -> float:
        """Estimate worst-case loss under random perturbations (same interface as DRO)."""
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(seed)

        worst_loss = -np.inf
        for _ in range(n_samples):
            X_pert = X + rng.normal(0, perturbation_std * X.std(axis=0) + 1e-8, X.shape)
            pred   = self.predict(X_pert)
            loss   = float(np.mean((y - pred) ** 2))
            worst_loss = max(worst_loss, loss)
        return worst_loss
