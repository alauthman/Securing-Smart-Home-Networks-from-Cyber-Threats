from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class PcapFeatureConfig:
    window_seconds: float = 1.0


def _require_tshark() -> str:
    exe = shutil.which("tshark")
    if not exe:
        raise RuntimeError(
            "tshark is required for PCAP conversion but was not found in PATH. "
            "Install Wireshark/tshark (e.g., `sudo apt-get install tshark`)."
        )
    return exe


def pcap_to_features(pcap_path: Path, out_csv: Path, cfg: Optional[PcapFeatureConfig] = None) -> Path:
    """Convert PCAP to simple time-window features using tshark.

    Output columns:
    - ts: window start (epoch seconds)
    - pkt_size_mean: average frame length in window
    - conn_rate: approximate connection rate (unique 5-tuple count / window)
    - dns_qps: DNS packets per second (within window)
    - proto: most frequent protocol label in window

    This is a *baseline* extractor meant for research prototyping.
    For production-grade flow features, consider exporting flows (e.g., via Zeek) or richer extraction.
    """
    cfg = cfg or PcapFeatureConfig()
    tshark = _require_tshark()

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Extract per-packet fields. We rely on tshark to parse pcap efficiently.
    # We use -E header=y and comma separator to get a CSV-like output.
    cmd = [
        tshark,
        "-r",
        str(pcap_path),
        "-T",
        "fields",
        "-E",
        "header=y",
        "-E",
        "separator=,",
        "-e",
        "frame.time_epoch",
        "-e",
        "frame.len",
        "-e",
        "_ws.col.Protocol",
        "-e",
        "ip.src",
        "-e",
        "ip.dst",
        "-e",
        "tcp.srcport",
        "-e",
        "tcp.dstport",
        "-e",
        "udp.srcport",
        "-e",
        "udp.dstport",
        "-e",
        "dns.qry.name",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"tshark failed: {proc.stderr[:4000]}")

    # Read tshark output
    from io import StringIO

    df = pd.read_csv(StringIO(proc.stdout))

    if df.empty:
        df_out = pd.DataFrame(
            columns=["ts", "pkt_size_mean", "conn_rate", "dns_qps", "proto"]
        )
        df_out.to_csv(out_csv, index=False)
        return out_csv

    # Clean and prepare
    df["frame.time_epoch"] = pd.to_numeric(df["frame.time_epoch"], errors="coerce")
    df["frame.len"] = pd.to_numeric(df["frame.len"], errors="coerce")
    df = df.dropna(subset=["frame.time_epoch", "frame.len"])

    # Windowing
    df["ts"] = (df["frame.time_epoch"] // cfg.window_seconds) * cfg.window_seconds

    # Approximate 5-tuple
    df["srcport"] = df["tcp.srcport"].fillna(df["udp.srcport"])
    df["dstport"] = df["tcp.dstport"].fillna(df["udp.dstport"])
    df["five_tuple"] = (
        df["ip.src"].astype(str)
        + "-"
        + df["ip.dst"].astype(str)
        + "-"
        + df["srcport"].astype(str)
        + "-"
        + df["dstport"].astype(str)
        + "-"
        + df["_ws.col.Protocol"].astype(str)
    )

    df["is_dns"] = df["dns.qry.name"].notna() & (df["dns.qry.name"].astype(str).str.len() > 0)

    agg = df.groupby("ts").agg(
        pkt_size_mean=("frame.len", "mean"),
        conn_rate=("five_tuple", "nunique"),
        dns_packets=("is_dns", "sum"),
    )
    agg["dns_qps"] = agg["dns_packets"] / cfg.window_seconds
    agg = agg.drop(columns=["dns_packets"])

    # Most frequent protocol in the window
    proto = (
        df.groupby(["ts", "_ws.col.Protocol"])
        .size()
        .reset_index(name="n")
        .sort_values(["ts", "n"], ascending=[True, False])
        .drop_duplicates(subset=["ts"])
        .set_index("ts")["_ws.col.Protocol"]
    )
    agg["proto"] = proto

    agg = agg.reset_index()
    agg.to_csv(out_csv, index=False)
    return out_csv
