"""Download everything the study needs into ./data (not committed).

    python scripts/download_data.py
"""

from __future__ import annotations

from pathlib import Path

from macrobt.data import fred, yahoo
from macrobt.data.panel import ETFS, FRED_SERIES


def main() -> None:
    out = Path("data")
    out.mkdir(exist_ok=True)
    print(f"FRED: {len(FRED_SERIES)} series")
    fred.download(FRED_SERIES, out / "fred")
    print(f"Yahoo Finance: {', '.join(ETFS)}")
    yahoo.fetch_adjclose(list(ETFS)).to_csv(out / "etf_prices.csv")
    print("Saved to", out.resolve())


if __name__ == "__main__":
    main()
