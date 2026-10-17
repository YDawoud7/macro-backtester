"""Download and read FRED series (https://fred.stlouisfed.org)."""

from __future__ import annotations

import io
import time
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def parse_fred_csv(text: str) -> pd.DataFrame:
    """Parse fredgraph CSV text. Missing values may be blank or '.'."""
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    date_col = df.columns[0]  # 'observation_date' (current) or 'DATE' (older files)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df.index.name = "date"
    return df.apply(pd.to_numeric, errors="coerce")


def fetch_series(sid: str, timeout: float = 30) -> pd.DataFrame:
    req = urllib.request.Request(URL.format(sid=sid), headers={"User-Agent": "macrobt/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return parse_fred_csv(resp.read().decode("utf-8"))


def download(series: list[str], out_dir: str | Path, pause: float = 0.5) -> None:
    """Save each series to ``out_dir/<ID>.csv`` (one request per series)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for sid in series:
        df = fetch_series(sid)
        df.to_csv(out / f"{sid}.csv")
        time.sleep(pause)


def load(series: list[str], data_dir: str | Path) -> pd.DataFrame:
    """Read saved series into one frame (outer join on date)."""
    frames = []
    for sid in series:
        path = Path(data_dir) / f"{sid}.csv"
        frames.append(parse_fred_csv(path.read_text())[[sid]])
    return pd.concat(frames, axis=1).sort_index()
