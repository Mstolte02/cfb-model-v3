"""Positional recruiting and rated transfer-portal talent features.

The shipping talent blend has three axes - the CFBD recruiting composite, PFF
roster-aware grades and roster WAR - and none of them prices two facts a roster
carries: who it just gained or lost through the transfer portal, and how its
recent recruiting classes are distributed across position groups. These builders
supply both, and `attach` puts them onto a v4 team frame.

Both families are preseason facts for season N - the portal class that arrived
for N and recruiting classes through N - so they satisfy v4's temporal contract.
Measured against the clean core in audit/TALENT_SOURCES_EXPERIMENTS.md (-.00632
static / -.00330 online Brier, both intervals excluding zero) and adopted as
separate model features beside O/D rather than axes of config.TALENT_BLEND,
which is the form the measurement validated.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.data import cfbd_client, fbs


PORTAL = ["portal_in", "portal_out", "portal_net"]
# Rated-only: summing every entry at an imputed rating turns portal_in into a head
# count wearing a quality label. These count only players the recruiting services
# actually rated, plus the blue-chip tally separately, so quality and volume are
# not the same column.
PORTAL_RATED = ["portal_in_rated", "portal_out_rated", "portal_net_rated",
                "portal_blue_in", "portal_blue_out"]
GROUPS = ["rec_qb", "rec_ol", "rec_skill", "rec_front7", "rec_secondary"]

# The five GROUPS correlate .80-.94 with each other and .90 with talent: one construct
# entered five times. Fitting them separately splits a single signal across five
# coefficients that are not individually identifiable - two of the five flip sign in
# 11-23% of bootstrap refits, and the published rec_qb came out NEGATIVE, which is not
# a finding about quarterback recruiting but wherever a degenerate basis happened to
# land. REDUCED replaces them with their first principal component (83.7% of the
# block) and drops the three rated-portal columns, of which portal_net_rated is a
# construction identity - it is built as in - out, and regresses on those two at
# R^2 .964. See audit/STANDARDISATION_AND_COLLINEARITY.md.
REDUCED = ["O", "D", "talent_resid", "returning", "war_projected",
           "portal_blue_in", "rec_pc1"]

GROUP_MAP = {
    "Quarterback": "rec_qb",
    "Offensive Line": "rec_ol",
    "Receiver": "rec_skill", "Running Back": "rec_skill",
    "Defensive Line": "rec_front7", "Linebacker": "rec_front7",
    "Defensive Back": "rec_secondary",
}


def portal_features(years) -> dict[int, pd.DataFrame]:
    """Incoming, outgoing and net rated portal talent per team, per season.

    A portal entry is dated to the season it is listed under, which is the season the
    player arrives for. Unrated entries are counted at the league's rated mean rather
    than dropped: a walk-on transfer is not a zero-talent player, and dropping them
    would make a team that took ten unrated transfers look like it took none.
    """
    out = {}
    for year in years:
        rows = cfbd_client.transfer_portal(year)
        # The feed covers every division. A move between two FCS programmes is not an
        # FBS roster event and must not set the imputed rating that FBS arrivals are
        # scored against, so the fallback is the mean of moves landing in FBS.
        members = fbs.teams(year)
        rated = [float(r["rating"]) for r in rows
                 if r.get("rating") and r.get("destination") in members]
        fallback = float(np.mean(rated)) if rated else 0.85
        incoming, outgoing = defaultdict(float), defaultdict(float)
        in_rated, out_rated = defaultdict(float), defaultdict(float)
        blue_in, blue_out = defaultdict(float), defaultdict(float)
        # A blue-chip portal move is the one everybody actually notices. Four stars
        # and up, which is roughly the top decile of rated entries.
        for entry in rows:
            rating = float(entry["rating"]) if entry.get("rating") else None
            stars = float(entry["stars"]) if entry.get("stars") else 0.0
            value = rating if rating is not None else fallback
            dest, origin = entry.get("destination"), entry.get("origin")
            if dest:
                incoming[dest] += value
                if rating is not None:
                    in_rated[dest] += rating
                if stars >= 4:
                    blue_in[dest] += 1.0
            if origin:
                outgoing[origin] += value
                if rating is not None:
                    out_rated[origin] += rating
                if stars >= 4:
                    blue_out[origin] += 1.0
        teams = sorted(set(incoming) | set(outgoing))
        frame = pd.DataFrame({
            "portal_in": [incoming.get(t, 0.0) for t in teams],
            "portal_out": [outgoing.get(t, 0.0) for t in teams],
            "portal_in_rated": [in_rated.get(t, 0.0) for t in teams],
            "portal_out_rated": [out_rated.get(t, 0.0) for t in teams],
            "portal_blue_in": [blue_in.get(t, 0.0) for t in teams],
            "portal_blue_out": [blue_out.get(t, 0.0) for t in teams],
        }, index=teams)
        frame["portal_net"] = frame.portal_in - frame.portal_out
        frame["portal_net_rated"] = frame.portal_in_rated - frame.portal_out_rated
        out[year] = frame
    return out


def group_features(years) -> dict[int, pd.DataFrame]:
    """Three-year rolling recruiting rating per position group, per team.

    One class is a small sample and most of a roster is not freshmen, so each season
    averages the three classes that make up the bulk of it.
    """
    raw = cfbd_client.recruiting_groups(min(years) - 4, max(years))
    frame = pd.DataFrame(raw)
    frame["column"] = frame.positionGroup.map(GROUP_MAP)
    frame = frame.dropna(subset=["column", "averageRating"])
    frame["year"] = pd.to_numeric(frame.year, errors="coerce") if "year" in frame \
        else np.nan
    if frame.year.isna().all():
        # The endpoint omits the year when a range is requested; fall back to the
        # single-season call so each class keeps its date.
        blocks = []
        for year in range(min(years) - 3, max(years) + 1):
            block = pd.DataFrame(cfbd_client.recruiting_groups(year, year))
            block["year"] = year
            blocks.append(block)
        frame = pd.concat(blocks, ignore_index=True)
        frame["column"] = frame.positionGroup.map(GROUP_MAP)
        frame = frame.dropna(subset=["column", "averageRating"])
    frame["averageRating"] = pd.to_numeric(frame.averageRating, errors="coerce")

    out = {}
    for year in years:
        window = frame[(frame.year <= year) & (frame.year >= year - 2)]
        pivot = window.pivot_table(index="team", columns="column",
                                   values="averageRating", aggfunc="mean")
        for column in GROUPS:
            if column not in pivot:
                pivot[column] = np.nan
        out[year] = pivot[GROUPS]
    return out


def attach(frame: pd.DataFrame, year_portal: pd.DataFrame,
           year_groups: pd.DataFrame) -> pd.DataFrame:
    """Standardize the talent-source columns onto one team frame, within season.

    A portal sum and a recruit rating enter on the same scale as the rest of the
    frame; missing means average, not zero talent, after standardization. Coverage
    lands in attrs for diagnostics instead of becoming a football coefficient.
    """
    p = year_portal.reindex(frame.index)
    g = year_groups.reindex(frame.index)
    for column in [*PORTAL, *PORTAL_RATED]:
        values = p[column]
        frame[column] = ((values - values.mean()) /
                         (values.std(ddof=0) or 1.0)).fillna(0.0)
    for column in GROUPS:
        values = g[column]
        frame[column] = ((values - values.mean()) /
                         (values.std(ddof=0) or 1.0)).fillna(0.0)
    frame.attrs["portal_coverage"] = float(p.portal_in.notna().mean())
    frame.attrs["groups_coverage"] = float(g[GROUPS[0]].notna().mean())
    return derived(frame)


def _z(values: np.ndarray) -> np.ndarray:
    sd = values.std(ddof=0)
    return (values - values.mean()) / (sd or 1.0)


def derived(frame: pd.DataFrame) -> pd.DataFrame:
    """Add rec_pc1 and talent_resid, both within season.

    rec_pc1 is the first principal component of the five recruiting groups, signed to
    point the same way they do so a positive coefficient still means "recruits well".
    Season by season rather than pooled, because every other column in the frame is
    standardised within season and a pooled component would smuggle cross-season
    drift into a feature the model reads as a within-season z.

    talent_resid is talent with rec_pc1 projected out. The two correlate .85, and
    fitting both leaves talent carrying a NEGATIVE coefficient - true as a partial
    effect and badly misleading read alone, because talent on its own is +0.677 and
    a 1sd talent edge is worth +0.483 logits once the correlated inputs follow.
    Orthogonalising costs 0.0001 Brier and reports the shared axis on rec_pc1, where
    it belongs, leaving talent_resid as the small remainder it actually is.
    """
    if not set(GROUPS) <= set(frame.columns):
        return frame
    M = frame[GROUPS].astype(float).values
    M = (M - M.mean(axis=0)) / (M.std(axis=0, ddof=0) + 1e-12)
    u, s, _ = np.linalg.svd(M, full_matrices=False)
    pc = u[:, 0] * s[0]
    # SVD fixes the component only up to sign; anchor it to the block's own mean so
    # the sign cannot flip between seasons and silently negate the coefficient.
    if np.corrcoef(pc, M.mean(axis=1))[0, 1] < 0:
        pc = -pc
    frame["rec_pc1"] = _z(pc)

    if "talent" in frame.columns:
        t = frame["talent"].astype(float).values
        r = frame["rec_pc1"].values
        slope, intercept = np.polyfit(r, t, 1)
        frame["talent_resid"] = _z(t - (slope * r + intercept))
    return frame
