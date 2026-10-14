"""Streamlit app — Dynamic Pricing Optimiser.

Run locally:
    streamlit run app.py

The app lets the user:
- Pick a product tier (budget / standard / premium)
- Adjust the cost constraint (min/max markup over cost)
- See the estimated elasticity and the optimal price
- Visualise revenue and profit as functions of price
"""
from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data import load_data, UNIT_COSTS, TRUE_ELASTICITY  # noqa: E402
from model import (  # noqa: E402
    estimate_elasticity, estimate_elasticity_by_tier,
    find_optimal_price, find_optimal_profit_price,
    revenue_function, profit_function,
)
from utils import plot_revenue_vs_price  # noqa: E402


@st.cache_data(show_spinner="Fitting elasticity on first run ...")
def fit_elasticity(df: pd.DataFrame, tier: str | None = None) -> dict:
    if tier is not None:
        df = df[df["product_tier"] == tier]
    return estimate_elasticity(df)


def main():
    st.set_page_config(page_title="Dynamic Pricing Optimiser", page_icon=":moneybag:",
                       layout="wide")
    st.title("Dynamic Pricing Optimiser")
    st.caption("Estimate price elasticity from historical transactions and find the revenue-maximising price under a cost constraint.")

    df = load_data()
    st.write(f"**Loaded {len(df)} transactions**")

    # Sidebar
    st.sidebar.header("Pricing parameters")
    tier = st.sidebar.selectbox("Product tier", ["all", "budget", "standard", "premium"])
    unit_cost = st.sidebar.slider("Unit cost ($)", 5.0, 30.0, 12.0, 0.5)
    min_markup = st.sidebar.slider("Min markup over cost", 0.05, 0.50, 0.10, 0.01)
    max_markup = st.sidebar.slider("Max markup over cost", 0.50, 3.00, 2.00, 0.05)

    # Fit elasticity
    if tier == "all":
        est = fit_elasticity(df)
    else:
        est = fit_elasticity(df[df["product_tier"] == tier])
        # Override unit cost with the tier's true cost for the optimisation
        unit_cost = UNIT_COSTS.get(tier, unit_cost)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Estimated elasticity", f"{est['elasticity']:.3f}")
    col2.metric("R² of fit",            f"{est['r_squared']:.3f}")
    col3.metric("Std. error",           f"{est['std_err']:.3f}")
    col4.metric("Sample size",          f"{est['n']:,}")

    # Optimise
    optimum = find_optimal_price(est["intercept"], est["elasticity"], unit_cost,
                                  min_markup=min_markup, max_markup=max_markup)
    profit_opt = find_optimal_profit_price(est["intercept"], est["elasticity"], unit_cost,
                                            min_markup=min_markup, max_markup=max_markup)

    st.divider()
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Revenue-maximising price", f"${optimum['optimal_price']:.2f}",
                delta=f"${optimum['optimal_revenue']:,.0f} predicted revenue")
    col_b.metric("Profit-maximising price",  f"${profit_opt['optimal_price']:.2f}",
                delta=f"${profit_opt['optimal_profit']:,.0f} predicted profit")
    col_c.metric("Closed-form price (elasticity < -1)",
                f"${optimum['closed_form_price']:.2f}" if not np.isnan(optimum['closed_form_price']) else "N/A")

    # Plot
    st.divider()
    st.subheader("Revenue and profit vs price")
    fig = plot_revenue_vs_price(est["intercept"], est["elasticity"], unit_cost,
                                 optimal_price=optimum["optimal_price"],
                                 price_range=(unit_cost * (1 + min_markup * 0.2),
                                              unit_cost * (1 + max_markup)))
    st.pyplot(fig, use_container_width=True)

    # Per-tier comparison
    st.divider()
    st.subheader("Elasticity by product tier")
    el_by_tier = estimate_elasticity_by_tier(df)
    st.dataframe(el_by_tier, use_container_width=True)

    st.caption(f"Ground-truth elasticity in the synthetic data generator: {TRUE_ELASTICITY}")


if __name__ == "__main__":
    main()
