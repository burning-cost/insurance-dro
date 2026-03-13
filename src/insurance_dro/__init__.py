"""
insurance-dro: Distributionally Robust Optimisation for insurance premium setting.

DRO finds a premium that minimises the worst-case expected loss over a set of
plausible data distributions near the training distribution. The uncertainty set
is a Wasserstein ball of radius rho around the empirical distribution.

In insurance pricing, DRO is used when:
- The test distribution may differ from training (new segments, climate shift)
- You need a premium that is stable under distributional perturbation
- Standard ERM produces premiums that degrade badly on stressed scenarios

Key classes:
    WassersteinDRO     — DRO premium via Wasserstein uncertainty set
    CVaRPremium        — CVaR-based robust premium (tail risk aware)
    DeterministicERM   — Standard ERM premium (baseline)
    DROResult          — Result container

Typical usage::

    from insurance_dro import WassersteinDRO

    dro = WassersteinDRO(rho=0.1, loss_function='quadratic')
    result = dro.fit(X, y)
    premium = dro.predict(X_new)
"""

from insurance_dro.wasserstein import WassersteinDRO, DROResult
from insurance_dro.cvar import CVaRPremium
from insurance_dro.erm import DeterministicERM

__version__ = "0.1.0"

__all__ = [
    "WassersteinDRO",
    "DROResult",
    "CVaRPremium",
    "DeterministicERM",
    "__version__",
]
