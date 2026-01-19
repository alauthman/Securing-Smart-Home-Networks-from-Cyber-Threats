#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make(rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    proto = rng.choice(["TCP", "UDP", "ICMP"], size=rows, p=[0.7, 0.28, 0.02])

    # Normal traffic distribution
    pkt_size_mean = rng.normal(loc=350, scale=80, size=rows).clip(40, 1500)
    conn_rate = rng.gamma(shape=2.0, scale=0.6, size=rows)  # connections / sec
    dns_qps = rng.gamma(shape=1.5, scale=0.2, size=rows)

    # Inject anomalies
    y = np.array(["normal"] * rows, dtype=object)
    anom_idx = rng.choice(np.arange(rows), size=max(1, rows // 20), replace=False)  # 5% anomalies
    y[anom_idx] = "attack"

    pkt_size_mean[anom_idx] = rng.normal(loc=1100, scale=120, size=len(anom_idx)).clip(200, 1500)
    conn_rate[anom_idx] = rng.gamma(shape=6.0, scale=1.2, size=len(anom_idx))
    dns_qps[anom_idx] = rng.gamma(shape=7.0, scale=0.3, size=len(anom_idx))
    proto[anom_idx] = rng.choice(["TCP", "UDP"], size=len(anom_idx), p=[0.5, 0.5])

    df = pd.DataFrame(
        {
            "pkt_size_mean": pkt_size_mean,
            "conn_rate": conn_rate,
            "dns_qps": dns_qps,
            "proto": proto,
            "label": y,
        }
    )
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic smart-home network dataset (CSV).")
    ap.add_argument("--out", required=True, help="Output CSV path.")
    ap.add_argument("--rows", type=int, default=10000, help="Number of rows.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = make(rows=args.rows, seed=args.seed)
    df.to_csv(out, index=False)
    print(f"[OK] Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
