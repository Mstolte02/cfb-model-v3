"""Build opponent-adjusted QB values from CFBD PPA and save them.

Run: ./venv/bin/python -m scripts.qb_war [year ...]   (default 2020-2025)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ARTIFACTS
from src.data import load
from src import qbwar


def main(years):
    load.require_key()
    print(f"Building QB values for {list(years)} (pulling game PPA by week; cached) ...")
    df = qbwar.build_qb_values(years)

    out = ARTIFACTS / "qb_values.csv"
    df.to_csv(out, index=False)

    for y in years:
        sub = df[(df["season"] == y) & (df["att"] >= 200)].head(12)
        if sub.empty:
            continue
        alpha = sub["alpha"].iloc[0]
        print(f"\n=== {y} top QBs (opp-adjusted value, >=200 att; ridge alpha={alpha:g}) ===")
        for _, r in sub.iterrows():
            print(f"  {r['qb_value']:+.3f}  {r['name']:<20} {r['team']:<18} "
                  f"att={r['att']:>3} g={int(r['n_games'])}")
    print(f"\nSaved {len(df)} QB-seasons -> {out}")


if __name__ == "__main__":
    yrs = [int(a) for a in sys.argv[1:]] or [2020, 2021, 2022, 2023, 2024, 2025]
    main(yrs)
