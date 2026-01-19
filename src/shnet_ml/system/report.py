from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class SystemReportOutput:
    markdown_path: Path
    summary_json_path: Path


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _pct(x: List[float], p: float) -> float:
    if not x:
        return float("nan")
    return float(np.percentile(np.array(x, dtype=float), p))


def build_system_report(monitor_jsonl: Path, out_md: Path, out_json: Path) -> SystemReportOutput:
    rows = _load_jsonl(monitor_jsonl)
    if not rows:
        raise ValueError(f"No rows found in {monitor_jsonl}")

    cpu = [r.get("cpu_percent", 0.0) for r in rows]
    mem = [r.get("mem_percent", 0.0) for r in rows]

    proc_cpu = [r.get("proc_cpu_percent") for r in rows if "proc_cpu_percent" in r]
    proc_rss = [r.get("proc_rss") for r in rows if "proc_rss" in r]

    summary: Dict[str, Any] = {
        "samples": len(rows),
        "cpu_mean": float(mean(cpu)),
        "cpu_p95": _pct(cpu, 95),
        "mem_mean": float(mean(mem)),
        "mem_p95": _pct(mem, 95),
    }

    if proc_cpu:
        summary["proc_cpu_mean"] = float(mean([float(x) for x in proc_cpu if x is not None]))
        summary["proc_cpu_p95"] = _pct([float(x) for x in proc_cpu if x is not None], 95)
    if proc_rss:
        summary["proc_rss_mean_bytes"] = float(mean([float(x) for x in proc_rss if x is not None]))
        summary["proc_rss_p95_bytes"] = _pct([float(x) for x in proc_rss if x is not None], 95)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = f"""# System Monitoring Report

**Input:** `{monitor_jsonl}`  
**Samples:** {summary['samples']}

## System utilization
- CPU mean: {summary['cpu_mean']:.2f} %
- CPU p95: {summary['cpu_p95']:.2f} %
- Memory mean: {summary['mem_mean']:.2f} %
- Memory p95: {summary['mem_p95']:.2f} %

"""

    if "proc_cpu_mean" in summary:
        md += f"""## Process utilization
- Process CPU mean: {summary['proc_cpu_mean']:.2f} %
- Process CPU p95: {summary['proc_cpu_p95']:.2f} %
"""
    if "proc_rss_mean_bytes" in summary:
        md += f"""- Process RSS mean: {summary['proc_rss_mean_bytes'] / (1024**2):.2f} MB
- Process RSS p95: {summary['proc_rss_p95_bytes'] / (1024**2):.2f} MB
"""

    md += "\n## Notes\n- Use this report to support the deployment/performance section of your paper.\n"
    out_md.write_text(md, encoding="utf-8")

    return SystemReportOutput(markdown_path=out_md, summary_json_path=out_json)
