#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from shnet_ml.system.monitor import MonitorConfig, run_monitor


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect system monitoring metrics into JSONL.")
    ap.add_argument("--out", required=True, help="Output JSONL path.")
    ap.add_argument("--seconds", type=int, default=60, help="Total duration in seconds.")
    ap.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds.")
    ap.add_argument("--pid", type=int, default=None, help="Optional PID to monitor.")
    args = ap.parse_args()

    cfg = MonitorConfig(out=Path(args.out), seconds=args.seconds, interval=args.interval, pid=args.pid)
    out = run_monitor(cfg)
    print(f"[OK] Wrote monitor data: {out}")


if __name__ == "__main__":
    main()
