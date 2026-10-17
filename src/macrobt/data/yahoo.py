"""Adjusted close prices from Yahoo Finance's chart endpoint.

Adjusted closes include dividends and splits, so their returns are total
returns. Yahoo's terms don't allow redistribution, which is why price data
is downloaded locally and never committed to the repository.
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone

import pandas as pd

URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    "?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit&includeAdjustedClose=true"
)


def parse_chart_json(payload: dict) -> pd.Series:
    res = payload["chart"]["result"][0]
    offset = res["meta"].get("gmtoffset", 0)
    ts = pd.to_datetime([t + offset for t in res["timestamp"]], unit="s").normalize()
    adj = res["indicators"]["adjclose"][0]["adjclose"]
    s = pd.Series(adj, index=ts, dtype=float).dropna()
    return s[~s.index.duplicated(keep="last")]


def fetch_adjclose(tickers: list[str], start: str = "2000-01-01", pause: float = 0.5, timeout: float = 30) -> pd.DataFrame:
    p1 = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp())
    p2 = int(time.time())
    cols = {}
    for tk in tickers:
        req = urllib.request.Request(URL.format(ticker=tk, p1=p1, p2=p2), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            cols[tk] = parse_chart_json(json.loads(resp.read().decode("utf-8")))
        time.sleep(pause)
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df
