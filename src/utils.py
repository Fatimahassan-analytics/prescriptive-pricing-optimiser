"""Shared utilities for the Dynamic Pricing project."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.rcParams.update({
    "figure.figsize": (9, 5),
    "figure.dpi": 100,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
sns.set_style("whitegrid")


def save_fig(name: str, fig_dir: Path = Path("artefacts")) -> Path:
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    return path


def save_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def plot_price_vs_quantity(df: pd.DataFrame, price_col="price", qty_col="quantity",
                            tier_col="product_tier"):
    """Scatter of price vs quantity, log-log, coloured by product tier."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for tier in df[tier_col].unique():
        sub = df[df[tier_col] == tier]
        ax.scatter(sub[price_col], sub[qty_col], alpha=0.3, s=10, label=tier)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Price ($)"); ax.set_ylabel("Quantity sold")
    ax.set_title("Price vs quantity (log-log) by product tier")
    ax.legend(title=tier_col)
    return ax


def plot_demand_curve(alpha: float, beta: float, price_range=(10, 100), n=100):
    """Plot the estimated demand curve Q(P) = exp(α + β × ln(P))."""
    prices = np.linspace(*price_range, n)
    qty = np.exp(alpha + beta * np.log(prices))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(prices, qty, color="#216b91", linewidth=2)
    ax.set_xlabel("Price ($)"); ax.set_ylabel("Predicted quantity")
    ax.set_title(f"Estimated demand curve  (elasticity = {beta:.2f})")
    return ax


def plot_revenue_vs_price(alpha: float, beta: float, unit_cost: float,
                           optimal_price: float = None,
                           price_range=(10, 100), n=200):
    """Plot revenue and profit as functions of price; mark the optimum."""
    prices = np.linspace(*price_range, n)
    revenue = prices * np.exp(alpha + beta * np.log(prices))
    profit  = (prices - unit_cost) * np.exp(alpha + beta * np.log(prices))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(prices, revenue, color="#216b91", linewidth=2, label="Revenue")
    ax.plot(prices, profit,  color="#b05b3e", linewidth=2, label="Profit")
    if optimal_price is not None:
        ax.axvline(optimal_price, color="#374952", linestyle="--", alpha=0.7,
                   label=f"Optimal price = ${optimal_price:.2f}")
    ax.axvline(unit_cost, color="#747a7d", linestyle=":", alpha=0.7,
               label=f"Unit cost = ${unit_cost:.2f}")
    ax.set_xlabel("Price ($)"); ax.set_ylabel("Amount ($)")
    ax.set_title("Revenue and profit vs price")
    ax.legend()
    return ax


def plot_elasticity_by_tier(elasticity_df: pd.DataFrame):
    """Bar chart of estimated elasticity per product tier."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=elasticity_df, x="tier", y="elasticity", ax=ax, palette="Set2")
    ax.axhline(-1, color="#b05b3e", linestyle="--", alpha=0.7, label="Elasticity = -1 (unit elastic)")
    ax.set_title("Estimated price elasticity by product tier")
    ax.set_ylabel("Elasticity")
    ax.legend()
    return ax
