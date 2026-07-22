"""THE GATE: does the opponent-adjusted QB feature improve the model?

LOSO 2022-25, production logistic+margin ensemble, baseline (6 features) vs
+qb_edge (7th feature = home QB value - away QB value). Keep the QB layer only
if Brier/log-loss improve.

Run: ./venv/bin/python -m scripts.loso_qb
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import (GAME_YEARS, UNCERTAINTY_LAMBDA, OPP_ADJ_ALPHA, TALENT_BLEND,
                    ARTIFACTS)
from src.data import load, pff
from src import matchup as MU
from src import oppadj as OA
from src import model as M
from src import qbwar
from scripts.train import load_bundle, raw_returning, blended_talent

FOLDS = [2022, 2023, 2024, 2025]


def frames_by_year(std, talent, ret, games, pyth, ret_raw, od, train_years):
    b_o, b_d = MU.fit_talent_od_slopes(train_years, std, talent, od_by_year=od)
    frames = {}
    for N in GAME_YEARS:
        u = (1.0 - ret_raw[N]).clip(lower=0, upper=1) if N in ret_raw else None
        unc = (UNCERTAINTY_LAMBDA, b_o, b_d, u) if u is not None else None
        f = MU.team_frame(N, std, pyth, talent, ret, uncertainty=unc, od_by_year=od)
        if f is not None:
            frames[N] = f
    return frames


def build_year(frame, games_df, qb_series, use_qb):
    teams = set(frame.index)
    X, y, hf, mg = [], [], [], []
    for _, g in games_df.iterrows():
        h, a = g["home_team"], g["away_team"]
        if h not in teams or a not in teams or g["home_points"] == g["away_points"]:
            continue
        fh, fa = frame.loc[h], frame.loc[a]
        row = [fh.O - fa.D, fh.D - fa.O, fh.fp_margin - fa.fp_margin,
               fh.pythag - fa.pythag, fh.talent - fa.talent, fh.returning - fa.returning]
        if use_qb:
            row.append(float(qb_series.get(h, 0.0)) - float(qb_series.get(a, 0.0)))
        X.append(row)
        y.append(1 if g["home_points"] > g["away_points"] else 0)
        hf.append(0 if g.get("neutral_site", False) else 1)
        mg.append(float(g["home_points"] - g["away_points"]))
    return np.array(X, float), np.array(y), np.array(hf, float), np.array(mg)


def run(use_qb, qb_feat, std, talent, ret, games, pyth, ret_raw):
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    agg = {"brier": [], "log_loss": [], "accuracy": []}
    for ty in FOLDS:
        tr = [g for g in GAME_YEARS if g != ty]
        frames = frames_by_year(std, talent, ret, games, pyth, ret_raw, od, tr)
        parts = {N: build_year(frames[N], games[N], qb_feat.get(N, {}), use_qb)
                 for N in frames}
        Xtr = np.vstack([parts[g][0] for g in tr if g in parts])
        ytr = np.concatenate([parts[g][1] for g in tr if g in parts])
        hf = np.concatenate([parts[g][2] for g in tr if g in parts])
        mg = np.concatenate([parts[g][3] for g in tr if g in parts])
        model, _ = M.train(Xtr, ytr, hf, margins=mg)
        m = M.evaluate(model, parts[ty][0], parts[ty][1], parts[ty][2])
        for k in agg:
            agg[k].append(m[k])
        if use_qb and ty == FOLDS[-1]:
            print(f"    qb_edge coef (last fold) = {model.coef[-1]:+.3f}")
    return {k: float(np.mean(v)) for k, v in agg.items()}


def main():
    load.require_key()
    print("Loading bundle + QB feature ...")
    std, cfbd_tal, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()
    talent = blended_talent(cfbd_tal, pff.build_roster_talent(), w=TALENT_BLEND)
    qb_feat = qbwar.build_team_qb_feature(ARTIFACTS / "qb_values.csv", GAME_YEARS)
    cover = {N: len(s) for N, s in qb_feat.items()}
    print(f"  QB feature teams/season: {cover}")

    print(f"\n{'variant':<24}{'Brier':>9}{'LogLoss':>10}{'Acc':>8}   (LOSO {FOLDS})")
    print("-" * 55)
    qb_delta = qbwar.build_team_qb_delta(ARTIFACTS / "qb_values.csv", GAME_YEARS)

    base = run(False, qb_feat, std, talent, ret, games, pyth, ret_raw)
    print(f"{'baseline (6 feat)':<24}{base['brier']:>9.4f}{base['log_loss']:>10.4f}{base['accuracy']:>8.3f}")
    qb = run(True, qb_feat, std, talent, ret, games, pyth, ret_raw)
    print(f"{'+ qb_edge (level)':<24}{qb['brier']:>9.4f}{qb['log_loss']:>10.4f}{qb['accuracy']:>8.3f}")
    qd = run(True, qb_delta, std, talent, ret, games, pyth, ret_raw)
    print(f"{'+ qb_delta (change)':<24}{qd['brier']:>9.4f}{qd['log_loss']:>10.4f}{qd['accuracy']:>8.3f}")

    for name, r in [("qb_edge", qb), ("qb_delta", qd)]:
        dB = r["brier"] - base["brier"]
        dL = r["log_loss"] - base["log_loss"]
        print(f"\n  {name}: Brier {dB:+.4f} ({100*dB/base['brier']:+.2f}%), "
              f"log-loss {dL:+.4f} ({100*dL/base['log_loss']:+.2f}%) -> "
              f"{'HELPS' if dB < -0.0002 else 'no meaningful gain'}")


if __name__ == "__main__":
    main()
