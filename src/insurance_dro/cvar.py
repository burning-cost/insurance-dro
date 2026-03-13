"""
CVaR-based robust premium estimator.

The CVaR (Conditional Value at Risk) premium principle:
    pi_CVaR(Y; alpha) = E[Y | Y >= VaR_alpha(Y)] * (1 + loading)

This is the expected loss given that we're in the worst (1-alpha) fraction of outcomes.
As a premium principle it provides robustness against tail events: if the underlying
loss distribution shifts adversarially, the CVaR premium provides a buffer.

Connection to DRO: minimising CVaR loss is equivalent to DRO with a specific
uncertainty set (the variation distance ball). CVaRPremium is included as a
complementary approach to WassersteinDRO.

The fitted premium for individual risks is:
    premium_i = E[Y_i] * (1 + cvar_loading_i)
where the CVaR loading depends on the tail of the individual residual distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class CVaRPremium(BaseEstimator):
    """CVaR-based robust premium estimator.

    Parameters
    ----------
    alpha : float
        Tail probability level. alpha=0.9 means the premium is the expected
        loss in the worst 10% of scenarios. Higher alpha = more conservative.
    loading : float
        Additional multiplicative loading on top of CVaR premium. Default 0.0.
    fit_intercept : bool

    Examples
    --------
    >>> from insurance_dro import CVaRPremium
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> X = rng.normal(size=(1000, 3))
    >>> y = np.maximum(rng.normal(100, 30, 1000), 0)
    >>> model = CVaRPremium(alpha=0.90)
    >>> model.fit(X, y)
    >>> pred = model.predict(X)
    """

    def __init__(
        self,
        alpha: float = 0.90,
        loading: float = 0.0,
        fit_intercept: bool = True,
    ):
        self.alpha = alpha
        self.loading = loading
        self.fit_intercept = fit_intercept
        self._scaler = StandardScaler()
        self._model: LinearRegression | None = None
        self._cvar_multiplier: float = 1.0

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
    ) -> "CVaRPremium":
        """Fit CVaR premium model.

        Parameters
        ----------
        X : array-like (n, p)
        y : array-like (n,)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        X_scaled = self._scaler.fit_transform(X)
        self._model = LinearRegression(fit_intercept=self.fit_intercept)
        self._model.fit(X_scaled, y)

        # Compute CVaR multiplier from training residuals
        y_pred    = self._model.predict(X_scaled)
        residuals = y - y_pred
        var_level = np.quantile(residuals, self.alpha)
        tail_mask = residuals >= var_level

        if tail_mask.sum() > 0:
            cvar_resid     = residuals[tail_mask].mean()
            mean_y         = y.mean()
            # CVaR multiplier: 1 + CVaR_loading/mean
            self._cvar_multiplier = float(1.0 + max(cvar_resid, 0.0) / max(mean_y, 1e-8))
        else:
            self._cvar_multiplier = 1.0

        self._cvar_multiplier *= (1.0 + self.loading)
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict CVaR-loaded premium."""
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        X = np.asarray(X, dtype=float)
        X_scaled = self._scaler.transform(X)
        base_pred = self._model.predict(X_scaled)
        return base_pred * self._cvar_multiplier
