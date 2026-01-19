#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from shnet_ml.data.pcap_features import PcapFeatureConfig, pcap_to_features


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert PCAP into time-window tabular features using tshark.")
    ap.add_argument("--pcap", required=True, help="Path to a .pcap/.pcapng file.")
    ap.add_argument("--out", required=True, help="Output CSV path.")
    ap.add_argument("--window", type=float, default=1.0, help="Window size in seconds.")
    args = ap.parse_args()

    out = pcap_to_features(Path(args.pcap), Path(args.out), PcapFeatureConfig(window_seconds=args.window))
    print(f"[OK] Wrote features: {out}")


if __name__ == "__main__":
    main()
