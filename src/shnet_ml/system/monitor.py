from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import psutil


@dataclass(frozen=True)
class MonitorConfig:
    out: Path
    seconds: int = 60
    interval: float = 1.0
    pid: Optional[int] = None


def _snapshot(pid: Optional[int] = None) -> Dict[str, Any]:
    ts = time.time()
    vm = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=None)
    disk = psutil.disk_usage("/")
    net = psutil.net_io_counters()

    snap: Dict[str, Any] = {
        "ts": ts,
        "cpu_percent": cpu,
        "mem_used": vm.used,
        "mem_available": vm.available,
        "mem_percent": vm.percent,
        "disk_used": disk.used,
        "disk_free": disk.free,
        "disk_percent": disk.percent,
        "net_bytes_sent": net.bytes_sent,
        "net_bytes_recv": net.bytes_recv,
        "net_packets_sent": net.packets_sent,
        "net_packets_recv": net.packets_recv,
    }

    if pid is not None:
        try:
            p = psutil.Process(pid)
            with p.oneshot():
                snap["proc_cpu_percent"] = p.cpu_percent(interval=None)
                snap["proc_rss"] = p.memory_info().rss
                snap["proc_vms"] = p.memory_info().vms
                snap["proc_num_threads"] = p.num_threads()
        except Exception:
            snap["proc_error"] = "process_not_available"

    return snap


def run_monitor(cfg: MonitorConfig) -> Path:
    cfg.out.parent.mkdir(parents=True, exist_ok=True)

    end = time.time() + cfg.seconds
    with cfg.out.open("w", encoding="utf-8") as f:
        # Prime CPU percent calculation
        _ = psutil.cpu_percent(interval=None)
        if cfg.pid is not None:
            try:
                _ = psutil.Process(cfg.pid).cpu_percent(interval=None)
            except Exception:
                pass

        while time.time() < end:
            snap = _snapshot(cfg.pid)
            f.write(json.dumps(snap) + "\n")
            f.flush()
            time.sleep(cfg.interval)

    return cfg.out
