"""Dynamic Pricing — data generation with a known price-elasticity curve.

Generates a synthetic transaction-level dataset where demand follows a
log-log relationship with price (constant elasticity), so the elasticity
estimation code can be verified against the ground truth.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SAMPLE_PATH = RAW_DIR / "transactions_sample.csv"
FULL_PATH = RAW_DIR / "transactions_full.csv"

RANDOM_SEED = 42

# Ground-truth parameters used by the generator (so we can validate estimation)
TRUE_ELASTICITY = -1.8        # demand falls 1.8% for every 1% price increase
TRUE_INTERCEPT  = 10.0        # ln(demand) at price=1
TRUE_BASE_COST = 12.0         # constant unit cost
UNIT_COSTS = {"standard": 12.0, "premium": 18.0, "budget": 8.0}


def generate_sample_data(n_rows: int = 2000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate synthetic transactions with a known constant-elasticity demand curve.

    Demand model (log-log):
        ln(quantity) = α + β × ln(price) + ε
    where β = TRUE_ELASTICITY = -1.8 (constant elasticity of demand).

    Adds realistic features:
    - Three product tiers (budget / standard / premium) with different base costs
    - Day-of-week seasonality (weekend uplift)
    - December peak (Christmas)
    - Promo days (random 20% lift)
    """
    rng = np.random.default_rng(seed)

    product_tier = rng.choice(["standard", "premium", "budget"], size=n_rows, p=[0.55, 0.20, 0.25])
    base_cost = np.array([UNIT_COSTS[t] for t in product_tier])
    tier_multiplier = np.where(product_tier == "premium", 1.5,
                       np.where(product_tier == "budget", 0.7, 1.0))

    # Price = cost × tier multiplier × (1 + random markup between 10% and 90%)
    markup = rng.uniform(0.10, 0.90, size=n_rows)
    price = base_cost * tier_multiplier * (1 + markup)

    # Day-of-week and seasonal multipliers
    days = rng.integers(0, 7, size=n_rows)
    months = rng.integers(1, 13, size=n_rows)
    dow_mult = np.array([1.0, 1.0, 1.0, 1.0, 1.1, 1.25, 1.30])[days]
    month_mult = 1.0 + 0.20 * np.sin(2 * np.pi * (months - 1) / 12)
    month_mult[months == 12] *= 1.30

    # Promo flag (random 25% of transactions)
    promo = rng.binomial(1, 0.25, size=n_rows)
    promo_mult = np.where(promo == 1, 1.20, 1.0)

    # Quantity demanded — log-log model
    log_quantity = (
        TRUE_INTERCEPT
        + TRUE_ELASTICITY * np.log(price)
        + np.log(dow_mult * month_mult * promo_mult * tier_multiplier)
        + rng.normal(loc=0, scale=0.20, size=n_rows)
    )
    quantity = np.maximum(1, np.exp(log_quantity)).round().astype(int)

    revenue = price * quantity
    cost = base_cost * quantity
    profit = revenue - cost

    df = pd.DataFrame({
        "transaction_id":   range(n_rows),
        "date":             pd.date_range("2023-01-01", periods=n_rows, freq="h").floor("D"),
        "product_tier":     product_tier,
        "price":            price.round(2),
        "quantity":         quantity,
        "unit_cost":        base_cost,
        "revenue":          revenue.round(2),
        "cost":             cost.round(2),
        "profit":           profit.round(2),
        "promo":            promo,
        "day_of_week":      days,
        "month":            months,
    })
    return df


def load_data() -> pd.DataFrame:
    """Load transactions: prefer full CSV, fall back to bundled sample."""
    if FULL_PATH.exists():
        return pd.read_csv(FULL_PATH, parse_dates=["date"])
    if SAMPLE_PATH.exists():
        return pd.read_csv(SAMPLE_PATH, parse_dates=["date"])
    df = generate_sample_data()
    df.to_csv(SAMPLE_PATH, index=False)
    return df


def save_processed(df: pd.DataFrame, name: str = "processed.csv") -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / name
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    df = generate_sample_data()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAMPLE_PATH, index=False)
    print(f"Wrote sample dataset: {SAMPLE_PATH} ({len(df)} rows)")
    print(f"True elasticity: {TRUE_ELASTICITY}")
    print(f"Total revenue: ${df.revenue.sum():,.0f}")
    print(df.head())
