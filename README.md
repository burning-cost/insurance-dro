# insurance-dro

Distributionally Robust Optimisation (DRO) for insurance premium setting.

## The problem

Your severity model was trained on 2022-2024 motor data. In 2025, claims inflation
hits 15%, a litigation funding firm enters the market, and a new risk segment
(EVs) makes up 20% of new business. Your ERM-fitted model is now pricing a
distribution it was never trained on.

Standard ERM (OLS, GLM) minimises expected loss on the training distribution.
When the test distribution shifts, it can produce premiums that are significantly
off — either underpricing tails (claims inflation) or overpricing standard risks
(new market entry creates adverse selection).

DRO finds a premium that minimises the *worst-case* expected loss over a set of
plausible distributions near the training distribution. The uncertainty set is a
Wasserstein ball of radius rho — a formal way of saying "allow the test distribution
to differ from training by at most this much".

## The solution

`WassersteinDRO` solves the Esfahani-Kuhn (2018) Wasserstein DRO problem for
quadratic loss. For this loss function, the DRO problem reduces to a regularised
regression with an L2 penalty calibrated to rho and the operator norm of the
feature matrix. No inner optimisation loop required — it's a clean, fast fit.

Also included: `CVaRPremium` (tail-risk-aware premium via Conditional Value at Risk)
and `DeterministicERM` (standard OLS baseline).

## Installation

```bash
pip install git+https://github.com/burning-cost/insurance-dro.git
```

## Usage

```python
from insurance_dro import WassersteinDRO

dro = WassersteinDRO(
    rho=0.2,             # uncertainty radius (tune to expected shift magnitude)
    loss="quadratic",    # quadratic loss (MSE)
)
dro.fit(X_train, y_train)
premiums = dro.predict(X_new)

# Evaluate worst-case loss under distributional perturbation
wc_loss = dro.worst_case_loss(X_test, y_test, perturbation_std=0.2)
```

## Choosing rho

rho controls the size of the uncertainty set. A practical approach:
- Fit on recent years, evaluate on older years under each rho
- Choose rho where the loss ratio is stable across the historical periods
- For soft robustness, rho=0.05-0.10 is a starting point
- For explicit regulatory stress, rho should match the stated shock magnitude

## Performance

Benchmarked against deterministic ERM (OLS) on synthetic motor severity data
(20,000 training policies) under 5 distribution shift scenarios. See
`notebooks/benchmark_dro.py` for full methodology.

- **DRO is 15-30% more stable** (lower loss ratio variance) under distribution shift
  scenarios (covariate shift, variance shift, tail shift, mixed) than ERM.
- **ERM outperforms DRO in-distribution** — this is expected. DRO trades
  in-sample accuracy for robustness. The tradeoff is governed by rho.
- **Tail shift is the most dramatic scenario**: ERM loss ratio can drift 10-20%
  from target under heavy-tailed perturbations; DRO stays within 3-5%.
- **CVaR premium** is simpler and more interpretable but assumes a fixed
  multiplicative loading rather than adapting to the feature distribution.
- **Fit time is negligible**: DRO reduces to regularised regression and fits
  in under 0.2s for n=20k. The main overhead is tuning rho.

## References

- Esfahani, P. M., & Kuhn, D. (2018). Data-driven DRO using the Wasserstein metric. *Math. Programming*, 171(1).
- Blanchet, J., & Murthy, K. (2019). Quantifying distributional model risk. *Math. Operations Research*, 44(2).
- Rahimian, H., & Mehrotra, S. (2019). Distributionally robust optimization: A review. *arXiv:1908.05659*.
