"""Score a Fourth & Jev football ledger against stored outcomes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def _p(answer):
    if not isinstance(answer, dict):
        return None
    value = answer.get("noul")
    return float(value) if value is not None else None


def load_rows(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            answers = record.get("answers", {})
            outcome = record.get("metadata", {}).get("outcome", {})
            base = {
                "game_id": record.get("game_id"),
                "season": record.get("metadata", {}).get("season"),
                "model": record.get("metadata", {}).get("resolved_model", record.get("jev_model")),
            }
            for name in ("home_win","home_by_7_plus","home_by_14_plus","home_by_21_plus","away_by_7_plus","within_3"):
                p = _p(answers.get(name))
                y = outcome.get(name)
                if p is not None and y is not None:
                    rows.append({**base, "target": name, "p": p, "y": int(bool(y))})
    return pd.DataFrame(rows)


def metric(group: pd.DataFrame) -> dict:
    p = group.p.clip(1e-8, 1 - 1e-8)
    y = group.y.astype(float)
    return {
        "n": int(len(group)),
        "brier": float(((p-y)**2).mean()),
        "logloss": float((-(y*p.map(math.log) + (1-y)*(1-p).map(math.log))).mean()),
        "mean_p": float(p.mean()),
        "event_rate": float(y.mean()),
        "calibration_bias": float(p.mean() - y.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=Path, default=Path("artifacts/fourth_jev/football_ledger.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/fourth_jev/calibration.csv"))
    args = ap.parse_args()
    frame = load_rows(args.ledger)
    if frame.empty:
        raise SystemExit("no scored Noul answers found in ledger")
    rows=[]
    for (target,season), g in frame.groupby(["target","season"], dropna=False):
        rows.append({"target":target,"season":season,**metric(g)})
    for target,g in frame.groupby("target"):
        rows.append({"target":target,"season":"ALL",**metric(g)})
    out=pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(args.out,index=False)
    print(out.to_string(index=False))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
