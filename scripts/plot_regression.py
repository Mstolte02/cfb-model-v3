"""Visualize the offseason regression to the mean.

Left:  each team's END-of-(N-1) rating (recency-weighted "finishing form") vs the
       model's PRESEASON-N projected rating. The fitted slope < 1 (vs the y=x
       line) is the regression: extreme teams get pulled toward the mean.
Right: rolling game-by-game rating for a few teams across seasons, with the
       preseason reset marked at each season boundary.

Saves artifacts/regression_plot.png.  Run: ./venv/bin/python -m scripts.plot_regression
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import GAME_YEARS, STAT_YEARS, UNCERTAINTY_LAMBDA, ARTIFACTS
from src.data import load
from src import matchup as MU
from src import ewma as E
from scripts.train import load_bundle, raw_returning

HL = 3.0  # finishing-form half-life for the "end of season" rating


def main():
    load.require_key()
    print("Loading data + building ratings ...")
    std, talent, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()
    od_recent = E.build_od_by_year(STAT_YEARS, halflife=HL)
    b_o, b_d = MU.fit_talent_od_slopes(GAME_YEARS, std, talent)

    # ---- Scatter: end-of-(N-1) finishing form vs preseason-N projection ----
    xs, ys = [], []
    for N in GAME_YEARS:
        prior = N - 1
        if prior not in od_recent or N not in ret_raw:
            continue
        end_prev = E._z(od_recent[prior]["O"] + od_recent[prior]["D"])
        frame = MU.team_frame(N, std, pyth, talent, ret,
                              uncertainty=(UNCERTAINTY_LAMBDA, b_o, b_d, ret_raw[N]))
        if frame is None:
            continue
        pre = E._z(frame["O"] + frame["D"])
        common = end_prev.index.intersection(pre.index)
        xs.extend(end_prev.loc[common].values)
        ys.extend(pre.loc[common].values)
    xs, ys = np.array(xs), np.array(ys)
    slope, intercept = np.polyfit(xs, ys, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.scatter(xs, ys, s=14, alpha=0.4, color="#1E407C")
    lim = [min(xs.min(), ys.min()), max(xs.max(), ys.max())]
    ax1.plot(lim, lim, "--", color="gray", label="y = x (no regression)")
    xx = np.linspace(lim[0], lim[1], 50)
    ax1.plot(xx, slope * xx + intercept, "-", color="#D2492A", lw=2,
             label=f"fit: slope = {slope:.2f}")
    ax1.set_xlabel(f"End of season N-1 rating (recency-weighted, HL={HL:g})")
    ax1.set_ylabel("Preseason N projected rating (after regression)")
    ax1.set_title(f"Offseason regression to the mean\nslope {slope:.2f} "
                  f"=> ~{(1-slope)*100:.0f}% pulled toward average")
    ax1.axhline(0, color="k", lw=0.5); ax1.axvline(0, color="k", lw=0.5)
    ax1.legend()

    # ---- Time series: rolling rating across seasons for a few teams ----
    teams = ["Ohio State", "Alabama", "Indiana", "Oregon"]
    colors = {"Ohio State": "#1f77b4", "Alabama": "#d62728",   # blue, red,
              "Indiana": "#ff7f0e", "Oregon": "#2ca02c"}        # orange, green
    seasons = [s for s in (2023, 2024, 2025) if s in STAT_YEARS]
    offset, boundaries = 0, []
    for si, s in enumerate(seasons):
        rs = E.rolling_series(s, HL, teams=teams)
        boundaries.append(offset)
        for t in teams:
            d = rs[rs["team"] == t].sort_values("game_no")
            if d.empty:
                continue
            x = offset + np.arange(len(d))
            ax2.plot(x, d["rating"].values, color=colors[t], lw=1.6,
                     label=t if si == 0 else None)
        offset += 18
    for bnd, s in zip(boundaries, seasons):
        ax2.axvline(bnd, color="gray", ls=":", lw=1)
        ax2.text(bnd + 0.3, 0.96, str(s), transform=ax2.get_xaxis_transform(),
                 fontsize=9, color="gray", va="top")
    ax2.set_xlabel("Game index across seasons (dotted = season start)")
    ax2.set_ylabel("Rolling rating (O + D, recency-weighted)")
    ax2.set_title("Rating over time — note the reset each season boundary")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.legend(loc="lower left", fontsize=8)

    out = ARTIFACTS / "regression_plot.png"
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"slope of preseason-on-end-of-prior = {slope:.3f} "
          f"(1.0 = pure persistence, lower = stronger regression)")
    print(f"correlation r = {np.corrcoef(xs, ys)[0,1]:.3f}, n = {len(xs)}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
