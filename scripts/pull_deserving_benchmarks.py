"""Cache the outside data the Most Deserving backtest compares against.

Writes into data/raw (git-ignored), one file per season, skipping files that exist:
  postseason_YYYY.json  CFBD postseason games (identifies CFP selections and results)
  sp_YYYY.json          CFBD SP+ ratings (final, after bowls)
  fpi_YYYY.json         CFBD FPI ratings, including ESPN's résumé ranks (Strength of Record)

Usage:  CFBD_API_KEY=... python3 scripts/pull_deserving_benchmarks.py
        (also reads CFBD_API_KEY from .env at the repository root)
"""
import json
import os
import pathlib
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = pathlib.Path(os.environ.get("CFB_RAW_DIR", ROOT / "data" / "raw"))
BASE = "https://api.collegefootballdata.com"
SEASONS = [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]


def api_key():
    key = os.environ.get("CFBD_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("CFBD_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
    if not key:
        sys.exit("set CFBD_API_KEY (environment or .env)")
    return key


def get(path, key):
    req = urllib.request.Request(BASE + path, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main():
    key = api_key()
    RAW.mkdir(parents=True, exist_ok=True)
    jobs = []
    for y in SEASONS:
        jobs += [(f"postseason_{y}.json", f"/games?year={y}&seasonType=postseason"),
                 (f"sp_{y}.json", f"/ratings/sp?year={y}"),
                 (f"fpi_{y}.json", f"/ratings/fpi?year={y}")]
    for name, path in jobs:
        out = RAW / name
        if out.exists():
            continue
        data = get(path, key)
        out.write_text(json.dumps(data))
        print(f"{name}: {len(data)} rows", file=sys.stderr)
        time.sleep(0.4)


if __name__ == "__main__":
    main()
