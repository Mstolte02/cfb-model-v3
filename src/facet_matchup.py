"""Facet x position WAR, and the unit-against-unit matchup features built from it.

The model already knows a team's WAR. It does not know *where* that WAR sits, and it
never compares one team's rooms against the specific rooms that will line up opposite
them. This module builds both halves of that.

Two feature bases, deliberately kept apart, because they obey different contracts and
carry different amounts of information:

**Realized, lagged, facet x position.** `hybrid_player_war_by_facet.csv` splits every
player-season's WAR across 87 facets, and the split is exact: the facet columns sum to
`war` to machine precision. Each facet already carries a predeclared football concept
in `war_model/concepts.json` and `consolidated_facets.json`, and each player carries a
position, so (concept, position group) is a well-defined cell that costs no new
judgement. Season N reads the completed N-1 team totals only, exactly as
`src.data.war.lagged_team_talent` does.

**Preseason, projected, position group.** `war_model/preseason_group_war.py` stops the
leakage-safe roster projection one level earlier than the shipping team total, so
season N gets a projected WAR per position room from N-1 and earlier. This is the base
the production selector actually rewards (`war_projected`), but it is only as granular
as the room.

The matchup terms are the odd contrast already used by `V4.MATCHUP_PAIRS`:

    edge_ha * |edge_ha| - edge_ah * |edge_ah|

which is exactly zero when the two teams' rooms are level, grows faster than linear in
a real mismatch, and negates when the teams are swapped. That last property is not
decoration: the whole v4 architecture rests on the game vector being antisymmetric, so
a matchup term that is not antisymmetric is not admissible at all.

A purely *linear* cross term needs no new machinery, because
`b(O_h - D_a) + b(D_h - O_a)` is algebraically the same expression as
`b(O_h - O_a) + b(D_h - D_a)` - see `audit/RATING_ARCHITECTURE_EXPERIMENTS.md`. The
only thing a cross term can add is the nonlinearity.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import v4 as V4

WAR_DIR = Path(__file__).resolve().parents[1] / "war_model"
PLAYER_FACET_WAR = WAR_DIR / "hybrid_player_war_by_facet.csv"
GROUP_WAR = WAR_DIR / "preseason_group_war.csv"
CONCEPTS = WAR_DIR / "concepts.json"
CONSOLIDATED = WAR_DIR / "consolidated_facets.json"

META = ["season", "player_id", "player", "position", "team"]

# PFF position -> the room the model already reports in. Same map as
# war_model/build_roster_2026.PFF_TO_GROUP, extended with the CFBD-side spellings that
# also appear in the facet export.
POSITION_GROUP = {
    "QB": "QB", "HB": "RB", "FB": "RB", "RB": "RB", "WR": "WR", "TE": "TE",
    "T": "OT", "G": "IOL", "C": "IOL",
    "DI": "DT", "DT": "DT", "DL": "DT", "ED": "EDGE", "DE": "EDGE", "EDGE": "EDGE",
    "LB": "LB", "CB": "CB", "DB": "CB", "S": "SAF",
}

# The nine CFBD-sourced facets carry no concept label in concepts.json because they
# were added after it. Their names state the job. Havoc spans run and pass disruption,
# so it is its own concept rather than being forced into either.
CFBD_CONCEPTS = {
    "cfbd_havoc_dl": "havoc", "cfbd_havoc_lb": "havoc",
    "cfbd_recv_wr": "receiving", "cfbd_recv_te": "receiving",
    "cfbd_recv_rb": "receiving", "cfbd_run_qb": "qb_rushing",
    "cfbd_run_rb": "rushing", "cfbd_tackle_db": "tackling",
    "cfbd_tackle_lb": "tackling",
}

# unit -> (side, [(concept, position group), ...]). Cells with negligible mass are
# folded into the neighbouring room that does the same job, so no unit rests on a
# handful of players. Every populated cell in the export lands in exactly one unit.
UNITS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    # ---------------------------------------------------------------- offense
    "o_pass_qb":  ("off", [("passing", "QB"), ("qb_pressure", "QB")]),
    "o_run_qb":   ("off", [("qb_rushing", "QB")]),
    "o_run_rb":   ("off", [("rushing", "RB"), ("rushing", "WR"), ("rushing", "TE")]),
    "o_recv_wr":  ("off", [("receiving", "WR"), ("passing", "WR"),
                           ("yac", "WR"), ("hands", "WR")]),
    "o_recv_te":  ("off", [("receiving", "TE"), ("yac", "TE"), ("hands", "TE")]),
    "o_recv_rb":  ("off", [("receiving", "RB"), ("yac", "RB")]),
    "o_pblk_ot":  ("off", [("pass_protection", "OT")]),
    "o_pblk_iol": ("off", [("pass_protection", "IOL"), ("pass_protection", "TE")]),
    "o_rblk_ot":  ("off", [("run_blocking", "OT")]),
    "o_rblk_iol": ("off", [("run_blocking", "IOL"), ("run_blocking", "TE"),
                           ("run_blocking", "RB")]),
    "o_ballsec":  ("off", [("ball_security", "RB"), ("ball_security", "QB")]),
    "o_pen":      ("off", [("discipline", g) for g in
                           ("QB", "RB", "WR", "TE", "OT", "IOL")]),
    # ---------------------------------------------------------------- defense
    "d_prsh_edge": ("def", [("pass_rush", "EDGE")]),
    "d_prsh_dt":   ("def", [("pass_rush", "DT")]),
    "d_prsh_2nd":  ("def", [("pass_rush", "LB"), ("pass_rush", "CB"),
                            ("pass_rush", "SAF")]),
    "d_rdef_dt":   ("def", [("run_defense", "DT")]),
    "d_rdef_edge": ("def", [("run_defense", "EDGE")]),
    "d_rdef_lb":   ("def", [("run_defense", "LB")]),
    "d_rdef_sec":  ("def", [("run_defense", "CB"), ("run_defense", "SAF")]),
    "d_cov_cb":    ("def", [("coverage", "CB")]),
    "d_cov_saf":   ("def", [("coverage", "SAF")]),
    "d_cov_lb":    ("def", [("coverage", "LB")]),
    "d_havoc":     ("def", [("havoc", g) for g in ("EDGE", "LB", "DT", "SAF", "CB")]),
    "d_tackle":    ("def", [("tackling", g) for g in
                            ("DT", "EDGE", "LB", "CB", "SAF")]),
    "d_pen":       ("def", [("discipline", g) for g in
                            ("DT", "EDGE", "LB", "CB", "SAF")]),
}
OFF_UNITS = [u for u, (side, _) in UNITS.items() if side == "off"]
DEF_UNITS = [u for u, (side, _) in UNITS.items() if side == "def"]

# Predeclared, before any number was looked at: who lines up opposite whom.
UNIT_PAIRS: dict[str, tuple[str, str]] = {
    "fx_ot_vs_edge":    ("o_pblk_ot", "d_prsh_edge"),
    "fx_iol_vs_dt":     ("o_pblk_iol", "d_prsh_dt"),
    "fx_wr_vs_cb":      ("o_recv_wr", "d_cov_cb"),
    "fx_te_vs_lb":      ("o_recv_te", "d_cov_lb"),
    "fx_rb_vs_lb":      ("o_recv_rb", "d_cov_lb"),
    "fx_qb_vs_saf":     ("o_pass_qb", "d_cov_saf"),
    "fx_rbrun_vs_dt":   ("o_run_rb", "d_rdef_dt"),
    "fx_rblk_vs_dt":    ("o_rblk_iol", "d_rdef_dt"),
    "fx_rblkot_vs_ed":  ("o_rblk_ot", "d_rdef_edge"),
    "fx_qbrun_vs_lb":   ("o_run_qb", "d_rdef_lb"),
    "fx_ball_vs_havoc": ("o_ballsec", "d_havoc"),
    "fx_yac_vs_tackle": ("o_recv_wr", "d_tackle"),
    "fx_pen_vs_pen":    ("o_pen", "d_pen"),
}

# The projected base is only as granular as the room, so its pairs are rooms.
GROUPS = ["QB", "RB", "WR", "TE", "OT", "IOL", "DT", "EDGE", "LB", "CB", "SAF"]
OFF_GROUPS = ["QB", "RB", "WR", "TE", "OT", "IOL"]
DEF_GROUPS = ["DT", "EDGE", "LB", "CB", "SAF"]
GROUP_PAIRS: dict[str, tuple[str, str]] = {
    "gx_ot_vs_edge": ("OT", "EDGE"),
    "gx_iol_vs_dt":  ("IOL", "DT"),
    "gx_wr_vs_cb":   ("WR", "CB"),
    "gx_te_vs_lb":   ("TE", "LB"),
    "gx_qb_vs_saf":  ("QB", "SAF"),
    "gx_rb_vs_dt":   ("RB", "DT"),
    "gx_qb_vs_cb":   ("QB", "CB"),
    "gx_rb_vs_lb":   ("RB", "LB"),
}

UNIT_PREFIX = "fu_"
GROUP_PREFIX = "gu_"


# --------------------------------------------------------------- facet -> concept
def concept_map() -> dict[str, str]:
    concepts = json.loads(CONCEPTS.read_text())["concepts"]
    consolidated = json.loads(CONSOLIDATED.read_text())
    out = {f: name for name, members in concepts.items() for f in members}
    out.update({name: meta["concept"] for name, meta in consolidated.items()})
    out.update(CFBD_CONCEPTS)
    return out


def _cell_to_unit() -> dict[tuple[str, str], str]:
    return {cell: unit for unit, (_, cells) in UNITS.items() for cell in cells}


# ------------------------------------------------------------ realized unit WAR
def realized_unit_war() -> pd.DataFrame:
    """(season, team) x unit table of realized WAR, in wins.

    The facet columns sum to the player's WAR exactly, so the unit columns sum to the
    team's WAR exactly apart from cells the taxonomy declines to carry. How much of
    the total survives is recorded in `attrs["coverage_share"]` rather than assumed.
    """
    raw = pd.read_csv(PLAYER_FACET_WAR)
    facets = [c for c in raw.columns if c not in META]
    mapping, cells = concept_map(), _cell_to_unit()

    group = raw.position.map(POSITION_GROUP)
    if group.isna().any():
        missing = sorted(raw.loc[group.isna(), "position"].unique())
        raise KeyError(f"positions outside POSITION_GROUP: {missing}")
    group = group.to_numpy()
    masks = {g: (group == g) for g in np.unique(group)}

    unit_names = list(UNITS)
    column_of = {u: i for i, u in enumerate(unit_names)}
    matrix = np.zeros((len(raw), len(unit_names)), dtype=float)
    total = 0.0
    for facet in facets:
        concept = mapping.get(facet)
        if concept is None:
            raise KeyError(f"facet {facet!r} carries no concept label")
        values = raw[facet].to_numpy(float)
        total += float(values.sum())
        for g, mask in masks.items():
            unit = cells.get((concept, g))
            if unit is None:
                continue
            matrix[mask, column_of[unit]] += values[mask]

    frame = pd.DataFrame(matrix, columns=unit_names, index=raw.index)
    frame[["season", "team"]] = raw[["season", "team"]]
    table = frame.groupby(["season", "team"]).sum()
    table.attrs["coverage_share"] = (float(matrix.sum() / total) if total
                                     else float("nan"))
    return table


def _z_frame(frame: pd.DataFrame) -> pd.DataFrame:
    mu, sd = frame.mean(), frame.std(ddof=0)
    sd = sd.where(np.isfinite(sd) & (sd > 1e-12), 1.0)
    return (frame - mu) / sd


def lagged_unit_war(index_by_year: dict[int, pd.Index]) -> dict[int, pd.DataFrame]:
    """Season N reads the completed N-1 unit totals, standardized within season.

    Identical discipline to `src.data.war.lagged_team_talent`: the season-N participant
    table is never consulted, so a team that loses its whole two-deep still enters N
    carrying what it produced in N-1. 2020 is absent from the WAR build, so 2021 has no
    prior and is simply missing; the caller decides the imputation, as it already does
    for `war_lag`.
    """
    table = realized_unit_war()
    by_year = {int(season) + 1: g.droplevel("season")
               for season, g in table.groupby("season")}
    out = {}
    for year, index in index_by_year.items():
        prior = by_year.get(int(year))
        if prior is None:
            continue
        out[int(year)] = _z_frame(prior.reindex(index))
    return out


# ----------------------------------------------------------- projected group WAR
def projected_group_war(index_by_year: dict[int, pd.Index]) -> dict[int, pd.DataFrame]:
    """Preseason projected WAR per position room, standardized within season.

    Sourced from `war_model/preseason_group_war.py`, whose room totals reconcile
    exactly with the `war_projected` the production model already selects.

    A room with no row contributed nothing to that team's projected total, so it is
    zero WAR before standardization rather than a missing value afterwards - that
    keeps the room columns summing to the team total. It is zero for a reason worth
    stating: CFBD's historical rosters list linemen as `OL` and `DL`, neither of
    which appears in `project_2026_v2.CFBD_TO_GROUP`, so the historical projection
    never sees most of them. See the coverage table in the audit note.
    """
    if not GROUP_WAR.exists():
        return {}
    raw = pd.read_csv(GROUP_WAR)
    wide = raw.pivot_table(index=["season", "team"], columns="group",
                           values="proj_war", aggfunc="sum")
    wide = wide.reindex(columns=GROUPS).fillna(0.0)
    seasons = set(wide.index.get_level_values("season"))
    out = {}
    for year, index in index_by_year.items():
        if int(year) not in seasons:
            continue
        out[int(year)] = _z_frame(wide.xs(int(year), level="season").reindex(index))
    return out


# ------------------------------------------------------------------ frame joining
def attach(frame: pd.DataFrame, units: pd.DataFrame | None,
           groups: pd.DataFrame | None) -> pd.DataFrame:
    """Add unit and room columns to a v4 team frame.

    A missing player source is unknown, not evidence of a weak room, so absent rows
    become the within-season mean (zero after standardization) and the coverage is
    recorded in attrs rather than turned into a football coefficient - the same policy
    `V4.build_frame` applies to `pff_lag` and `war_lag`.
    """
    for prefix, table, names in ((UNIT_PREFIX, units, list(UNITS)),
                                 (GROUP_PREFIX, groups, GROUPS)):
        columns = [f"{prefix}{n}" for n in names]
        if table is None:
            for column in columns:
                frame[column] = 0.0
            frame.attrs[f"{prefix}coverage"] = 0.0
            continue
        piece = table.reindex(frame.index)
        for name, column in zip(names, columns):
            frame[column] = (piece[name].fillna(0.0) if name in piece.columns
                             else 0.0)
        frame.attrs[f"{prefix}coverage"] = float(piece.notna().all(axis=1).mean())
    return frame


def unit_columns() -> list[str]:
    return [f"{UNIT_PREFIX}{u}" for u in UNITS]


def group_columns() -> list[str]:
    return [f"{GROUP_PREFIX}{g}" for g in GROUPS]


# ------------------------------------------------------------- cross registration
def register_pairs(pairs: dict[str, tuple[str, str]], prefix: str) -> list[str]:
    """Register cross terms with V4 so `matchup_vector` builds them.

    `V4.MATCHUP_PAIRS` is the model's own extension point for a nonlinear,
    antisymmetric offence-against-defence term, and
    `scripts/rating_architecture_backtest` registers into it the same way.
    """
    for name, (off, deff) in pairs.items():
        V4.MATCHUP_PAIRS[name] = (f"{prefix}{off}", f"{prefix}{deff}")
    return list(pairs)


def register_unit_pairs() -> list[str]:
    return register_pairs(UNIT_PAIRS, UNIT_PREFIX)


def register_group_pairs() -> list[str]:
    return register_pairs(GROUP_PAIRS, GROUP_PREFIX)


def all_unit_pairs() -> dict[str, tuple[str, str]]:
    """Every offence unit against every defence unit, for the discovery scan."""
    return {f"fs_{o}__{d}": (o, d) for o in OFF_UNITS for d in DEF_UNITS}


def all_group_pairs() -> dict[str, tuple[str, str]]:
    return {f"gs_{o}__{d}": (o, d) for o in OFF_GROUPS for d in DEF_GROUPS}
