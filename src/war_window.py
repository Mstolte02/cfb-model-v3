"""Player WAR for any window of weeks, on the production facet path and weights.

The shipped WAR (war_model/build_hybrid.py) is a full-season object: every PFF report
it reads is a season total. An in-season update needs the same number for weeks 1-W
and for the rest of the season, so this re-runs the production facet grammar on a
window of PFF reports and scores it with the production weights HELD FIXED:

  1. the five legacy facet reports (passing, rushing, receiving, blocking, defense)
     and the five API position-report families (pass/run blocking, pass rush, run
     defense, coverage) for the window, merged exactly as candidates.load_players
     merges them;
  2. candidates.facet_values - same catalogue, same denominators, same role-relative
     z (standardised within the window's own population);
  3. the consolidated composites, rebuilt from their members with coefficients
     recovered from the production build (a composite is linear in its members, so a
     per-season least-squares fit of composite value on member values returns each
     member's loading / sigma exactly - checked by `recover_composites`);
  4. f_contrib = w * value / sigma with production w and the season's production
     sigma.

Every position is covered, because the PFF reports cover every position - offensive
line from blocking/pass_blocking/run_blocking, defenders from defense/pass_rush/
run_defense/coverage. What is NOT reproduced: the 12 CFBD play-value facets (no
week-window source is wired in), and the Massey schedule share and team sensitivity
that turn f_contrib into WAA. So the output is "facet WAR": the player's own measured
contribution in production weights, before schedule allocation. The full-season
version of it is computed by the same code, so comparisons are like with like.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WAR = ROOT / "war_model"
if str(WAR) not in sys.path:
    sys.path.insert(0, str(WAR))

import candidates as cand  # noqa: E402  (war_model module; checks staged API files)

LEGACY = {"pass": "passing", "rush": "rushing", "recv": "receiving",
          "blk": "blocking", "def": "defense"}
API = dict(cand.API_SOURCES)                      # pblk/rblk/prsh/rdef/cov
SCALE = 1000.0


def load_players_from(files: dict[str, Path], season: int) -> pd.DataFrame:
    """candidates.load_players for one season, from an explicit {prefix: csv} map."""
    frames = []
    for pref, path in files.items():
        d = pd.read_csv(path, low_memory=False)
        d["season"] = season
        d = d.drop_duplicates(subset=["season", "player_id", "team_name"], keep="first")
        idx = ["season", "player_id", "team_name"]
        keep = [c for c in d.columns if c not in idx]
        d = d[idx + keep].rename(columns={c: f"{pref}__{c}" for c in keep})
        frames.append(d.set_index(idx))
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how="outer")
    out = out.reset_index()
    for base in ("player", "position"):
        cols = [f"{p}__{base}" for p in files if f"{p}__{base}" in out.columns]
        out[base] = out[cols].bfill(axis=1).iloc[:, 0]
        out = out.drop(columns=cols)
    return out


def season_files(season: int) -> dict[str, Path]:
    """The full-season files the production build itself reads."""
    files = {p: Path(cand.PFF_DIR) / f"{n}_{season}.csv" for p, n in LEGACY.items()}
    files.update({p: cand._position_dir / f"{n}_{season}.csv" for p, n in API.items()})
    return files


@lru_cache(maxsize=1)
def _production():
    f = pd.read_parquet(WAR / "hybrid_facet_war.parquet",
                        columns=["season", "player_id", "facet", "value", "sigma", "w",
                                 "f_contrib", "source", "snaps"])
    f["player_id"] = f.player_id.astype(str)
    return f


@lru_cache(maxsize=1)
def recover_composites():
    """{(season, composite): {member: coef}} with composite value = sum coef * member.

    Members are the pre-consolidation facet values in candidate_values.parquet. CFBD
    members are dropped - they have no window source - and reported, so a composite
    that leaned on one is known to be partial rather than silently rescaled.
    """
    import json
    prod = _production()
    report = json.load(open(WAR / "consolidated_facets.json"))
    raw = pd.read_parquet(WAR / "candidate_values.parquet",
                          columns=["season", "player_id", "facet", "value"])
    raw["player_id"] = raw.player_id.astype(str)
    coefs, fit = {}, {}
    for comp, info in report.items():
        members = [m for m in info["members"] if not m.startswith("cfbd_")]
        cy = prod[prod.facet == comp]
        for season, g in cy.groupby("season"):
            X = (raw[(raw.season == season) & raw.facet.isin(members)]
                 .pivot_table(index="player_id", columns="facet", values="value",
                              aggfunc="sum").reindex(columns=members).fillna(0.0))
            y = g.groupby("player_id").value.sum().reindex(X.index)
            ok = y.notna()
            if ok.sum() < 20:
                continue
            # cfbd members, if any, are left in the residual
            b, *_ = np.linalg.lstsq(X[ok].to_numpy(), y[ok].to_numpy(), rcond=None)
            pred = X[ok].to_numpy() @ b
            fit[(season, comp)] = float(np.corrcoef(pred, y[ok])[0, 1])
            coefs[(season, comp)] = dict(zip(members, b))
    return coefs, fit


def facet_contrib(players: pd.DataFrame, season: int,
                  weight_season: int | None = None) -> pd.DataFrame:
    """Per player: facet-WAR (sum of production w * value / sigma), snaps, group.

    `weight_season` picks whose sigma and composite coefficients to use; defaults to
    `season`. For the live season, which has no production sigma yet, pass the last
    completed season.
    """
    ws = weight_season or season
    # candidates.facet_values drops a facet with fewer than 200 qualifying
    # player-seasons, counted over every season in the frame. A three-week window has
    # ~110-220 qualifying quarterbacks, so without this the QB facets vanish silently
    # (weeks 1-3 correlated .17 with the full season for QBs). Padding with a full
    # season under a different season label passes the count; z is taken per season,
    # so the padding never touches the window's standardisation, and it is dropped.
    PAD = -1
    pad = load_players_from(season_files(ws), PAD)
    fv = cand.facet_values(pd.concat([players, pad], ignore_index=True), verbose=False)
    fv = fv[fv.season != PAD].copy()
    fv["player_id"] = fv.player_id.astype(str)
    prod = _production()
    pw = prod[prod.season == ws].drop_duplicates("facet").set_index("facet")
    w, sigma = pw.w, pw.sigma
    coefs, _ = recover_composites()

    rows = [fv[fv.facet.isin(w.index)][["player_id", "facet", "value", "snaps"]]]
    for (s, comp), cf in coefs.items():
        if s != ws:
            continue
        m = fv[fv.facet.isin(cf)].copy()
        if m.empty:
            continue
        m["value"] = m.value * m.facet.map(cf)
        rows.append(m.groupby("player_id", as_index=False)
                    .agg(value=("value", "sum"), snaps=("snaps", "max"))
                    .assign(facet=comp))
    f = pd.concat(rows, ignore_index=True)
    f = f[f.facet.isin(w.index)]
    f["fc"] = w.reindex(f.facet).to_numpy() * f.value / sigma.reindex(f.facet).to_numpy()
    info = fv.drop_duplicates("player_id").set_index("player_id")[
        ["player", "position", "team_name"]]
    out = f.groupby("player_id").agg(fc=("fc", "sum"), snaps=("snaps", "max"))
    out = out.join(info)
    out["group"] = out.position.map(GROUP)
    out["season"] = season
    return out.reset_index()


GROUP = {"QB": "QB", "HB": "RB", "FB": "RB", "RB": "RB", "WR": "WR", "TE": "TE",
         "T": "OT", "G": "IOL", "C": "IOL", "DI": "DT", "DL": "DT", "DT": "DT",
         "ED": "EDGE", "DE": "EDGE", "EDGE": "EDGE", "LB": "LB", "CB": "CB",
         "DB": "CB", "S": "SAF"}
