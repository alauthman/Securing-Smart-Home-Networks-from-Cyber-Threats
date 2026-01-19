#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from shnet_ml.system.report import build_system_report


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a Markdown system report from JSONL monitor output.")
    ap.add_argument("--monitor", required=True, help="Path to monitor JSONL.")
    ap.add_argument("--out", required=True, help="Output Markdown path.")
    ap.add_argument(
        "--out-json",
        default=None,
        help="Output summary JSON path (default: alongside markdown).",
    )
    args = ap.parse_args()

    monitor = Path(args.monitor)
    out_md = Path(args.out)
    out_json = Path(args.out_json) if args.out_json else out_md.with_suffix(".json")

    out = build_system_report(monitor, out_md, out_json)
    print(f"[OK] Report: {out.markdown_path}")
    print(f"[OK] Summary JSON: {out.summary_json_path}")


if __name__ == "__main__":
    main()
