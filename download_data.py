"""No external download for the Dynamic Pricing project — data is synthetic.

Run `python download_data.py` to regenerate the bundled sample with a fresh seed.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

RAW_DIR = Path(__file__).parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the synthetic generator.")
    parser.add_argument("--rows", type=int, default=2000,
                        help="Number of synthetic transactions to generate.")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from data import generate_sample_data
    df = generate_sample_data(n_rows=args.rows, seed=args.seed)
    out = RAW_DIR / "transactions_sample.csv"
    df.to_csv(out, index=False)
    print(f"[OK] Wrote synthetic transactions → {out} ({len(df)} rows)")
    print(f"Ground-truth elasticity: -1.80 (constant-elasticity log-log model)")


if __name__ == "__main__":
    main()
