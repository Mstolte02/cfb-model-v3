"""2026 power ratings using the matchup-adjusted model (with uncertainty index).

Run: ./venv/bin/python -m scripts.rank [pff|roster]
  (no arg)  balanced default frame
  pff       PFF-ONLY variant: no EA anywhere. Needs CFB_WAR_VARIANT=pff set, or it
            refuses -> artifacts/2026_power_ratings_pff.csv
  roster    ROSTER-WEIGHTED variant (70% two-deep PFF talent, full uncertainty
            shrinkage) -> artifacts/2026_power_ratings_roster.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PROJECTION_YEAR, ARTIFACTS, ROSTER_VARIANT, SITE_VARIANTS, \
    variant_suffix
from src.data import load
from src import matchup as MU
from src.model import CFBModel
from scripts.train import build_projection_frame


def main(variant=""):
    load.require_key()
    kw = ROSTER_VARIANT if variant == "roster" else {}
    suffix = variant_suffix(variant)
    frame = build_projection_frame(**kw)
    model = CFBModel.load()

    ratings = MU.power_ratings(model, frame)
    out_csv = ARTIFACTS / f"{PROJECTION_YEAR}_power_ratings{suffix}.csv"
    ratings.to_csv(out_csv, index=False)

    label = SITE_VARIANTS[variant]
    print(f"\n=== {PROJECTION_YEAR} Power Ratings (top 25, {label}) ===")
    for _, r in ratings.head(25).iterrows():
        print(f"  {int(r['rank']):>3}. {r['team']:<24} power={r['power']:.3f}")
    print(f"\nFull table -> {out_csv}")

    a, b = ratings.iloc[0]["team"], ratings.iloc[1]["team"]
    p = model.win_prob(MU.matchup_vector(frame, a, b), is_home=0.0)
    print(f"\nExample neutral-site matchup: {a} vs {b}")
    print(f"   P({a}) = {p:.3f}   P({b}) = {1-p:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
