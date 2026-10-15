"""Dynamic Pricing — elasticity estimation + revenue-maximising price finder.

Two tasks:
1. Estimate price elasticity via log-log regression: ln(Q) = α + β × ln(P)
2. Find the revenue-maximising price under a known cost constraint:
   max  R(p) = p × Q(p) = p × exp(α + β × ln(p))  =  exp(α) × p^(1 + β)
   s.t. p >= unit_cost × (1 + min_markup)

   Closed-form solution: optimal_price = -β / (1 + β) × unit_cost  (when β < -1)
   (derived from setting dR/dp = 0)

The code falls back to scipy.optimize for the constrained case.
"""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression

RANDOM_SEED = 42


def estimate_elasticity(df: pd.DataFrame, price_col: str = "price",
                        qty_col: str = "quantity") -> dict:
    """Estimate constant price elasticity via log-log OLS regression.

    Returns:
        {elasticity: float, intercept: float, r_squared: float, n: int, std_err: float}
    """
    df = df.copy()
    df["log_price"] = np.log(df[price_col])
    df["log_qty"]   = np.log(df[qty_col])

    X = df[["log_price"]].values
    y = df["log_qty"].values

    model = LinearRegression()
    model.fit(X, y)
    beta  = float(model.coef_[0])         # elasticity
    alpha = float(model.intercept_)
    r2    = float(model.score(X, y))

    # Standard error of the slope (1-D OLS, simple formula)
    n = len(X)
    residuals = y - model.predict(X)
    sse = float(np.sum(residuals ** 2))
    sxx = float(np.sum((X - X.mean()) ** 2))
    std_err = float(np.sqrt(sse / (n - 2) / sxx)) if sxx > 0 else float("nan")

    return {
        "elasticity": beta,
        "intercept": alpha,
        "r_squared": r2,
        "n": n,
        "std_err": std_err,
    }


def estimate_elasticity_by_tier(df: pd.DataFrame, tier_col: str = "product_tier") -> pd.DataFrame:
    """Estimate elasticity separately for each product tier."""
    rows = []
    for tier in df[tier_col].unique():
        sub = df[df[tier_col] == tier]
        if len(sub) < 30:
            continue
        est = estimate_elasticity(sub)
        est["tier"] = tier
        rows.append(est)
    return pd.DataFrame(rows)[["tier", "elasticity", "r_squared", "n", "std_err"]]


def revenue_function(price: float, alpha: float, beta: float) -> float:
    """Predicted revenue at a given price, given the log-log demand model."""
    if price <= 0:
        return 0.0
    log_qty = alpha + beta * np.log(price)
    return float(price * np.exp(log_qty))


def profit_function(price: float, alpha: float, beta: float, unit_cost: float) -> float:
    """Predicted profit at a given price."""
    if price <= 0:
        return 0.0
    log_qty = alpha + beta * np.log(price)
    qty = float(np.exp(log_qty))
    return float((price - unit_cost) * qty)


def find_optimal_price(alpha: float, beta: float, unit_cost: float,
                      min_markup: float = 0.10, max_markup: float = 2.0) -> dict:
    """Find the revenue-maximising price subject to a cost constraint.

    Constraint: unit_cost × (1 + min_markup) <= price <= unit_cost × (1 + max_markup)

    Returns:
        {optimal_price: float, optimal_revenue: float, optimal_qty: float,
         closed_form_price: float}
    """
    price_low  = unit_cost * (1 + min_markup)
    price_high = unit_cost * (1 + max_markup)

    # Closed-form (only valid when elasticity < -1)
    if beta < -1:
        closed_form = -beta / (1 + beta) * unit_cost
    else:
        closed_form = float("nan")

    # Constrained optimisation
    result = minimize_scalar(
        lambda p: -revenue_function(p, alpha, beta),
        bounds=(price_low, price_high),
        method="bounded",
        options={"xatol": 1e-3},
    )
    optimal_price = float(result.x)
    optimal_revenue = -float(result.fun)
    optimal_qty = float(np.exp(alpha + beta * np.log(optimal_price)))

    return {
        "optimal_price": optimal_price,
        "optimal_revenue": optimal_revenue,
        "optimal_qty": optimal_qty,
        "closed_form_price": closed_form,
        "unit_cost": unit_cost,
        "min_markup": min_markup,
        "max_markup": max_markup,
        "price_low": price_low,
        "price_high": price_high,
    }


def find_optimal_profit_price(alpha: float, beta: float, unit_cost: float,
                              min_markup: float = 0.10, max_markup: float = 2.0) -> dict:
    """Find the profit-maximising price subject to the same constraint."""
    price_low  = unit_cost * (1 + min_markup)
    price_high = unit_cost * (1 + max_markup)

    result = minimize_scalar(
        lambda p: -profit_function(p, alpha, beta, unit_cost),
        bounds=(price_low, price_high),
        method="bounded",
        options={"xatol": 1e-3},
    )
    optimal_price = float(result.x)
    optimal_profit = -float(result.fun)
    optimal_qty = float(np.exp(alpha + beta * np.log(optimal_price)))

    return {
        "optimal_price": optimal_price,
        "optimal_profit": optimal_profit,
        "optimal_qty": optimal_qty,
        "unit_cost": unit_cost,
        "min_markup": min_markup,
        "max_markup": max_markup,
    }


def save_model(model, name: str, models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name: str, models_dir: Path):
    return joblib.load(models_dir / f"{name}.joblib")
