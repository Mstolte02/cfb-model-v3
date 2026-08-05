"""Facet definitions for the PFF Massey model.

Facets reference columns in the merged player-season frame built by build_massey.py,
where each export contributes columns under a prefix: rush__, pass__, recv__, blk__, def__.

The paper splits facets whose grades carry different value at different positions
("receiving by wide receivers" and "coverage by defensive backs" are called out as
separate high-value facets), so the position splits below follow that logic.
"""

OL = ("T", "G", "C")
DL = ("DI", "ED")
DB = ("CB", "S")
SKILL = ("QB", "HB", "FB", "WR", "TE")

# name -> (grade_col, snap_col, positions, side)
FACETS = {
    # ---- offense ----
    "pass_qb":    ("pass__grades_pass",              "pass__passing_snaps",          ("QB",),                  "off"),
    # attempts, not run_plays: grades_run scores the ball carrier, and run_plays counts
    # every snap on a running play, which would hand rushing credit to blocking backs
    "run_rb":     ("rush__grades_run",               "rush__attempts",               ("HB", "FB"),             "off"),
    "run_qb":     ("rush__grades_run",               "rush__attempts",               ("QB",),                  "off"),
    "recv_wr":    ("recv__grades_pass_route",        "recv__routes",                 ("WR",),                  "off"),
    "recv_te":    ("recv__grades_pass_route",        "recv__routes",                 ("TE",),                  "off"),
    "recv_rb":    ("recv__grades_pass_route",        "recv__routes",                 ("HB", "FB"),             "off"),
    "pblk_ol":    ("blk__grades_pass_block",         "blk__snap_counts_pass_block",  OL,                       "off"),
    "pblk_skill": ("blk__grades_pass_block",         "blk__snap_counts_pass_block",  ("TE", "HB", "FB"),       "off"),
    "rblk_ol":    ("blk__grades_run_block",          "blk__snap_counts_run_block",   OL,                       "off"),
    "rblk_skill": ("blk__grades_run_block",          "blk__snap_counts_run_block",   ("TE", "HB", "FB", "WR"), "off"),
    "fumble":     ("rush__grades_hands_fumble",      "rush__total_touches",          SKILL,                    "off"),
    "pen_off":    ("rush__grades_offense_penalty",   "blk__snap_counts_offense",     SKILL + OL,               "off"),
    # ---- defense ----
    "cov_db":     ("def__grades_coverage_defense",   "def__snap_counts_coverage",    DB,                       "def"),
    "cov_lb":     ("def__grades_coverage_defense",   "def__snap_counts_coverage",    ("LB",),                  "def"),
    "prsh_dl":    ("def__grades_pass_rush_defense",  "def__snap_counts_pass_rush",   DL,                       "def"),
    "prsh_sec":   ("def__grades_pass_rush_defense",  "def__snap_counts_pass_rush",   ("LB",) + DB,             "def"),
    "rdef_dl":    ("def__grades_run_defense",        "def__snap_counts_run_defense", DL,                       "def"),
    "rdef_sec":   ("def__grades_run_defense",        "def__snap_counts_run_defense", ("LB",) + DB,             "def"),
    "tackle":     ("def__grades_tackle",             "def__snap_counts_defense",     DL + DB + ("LB",),        "def"),
    "pen_def":    ("def__grades_defense_penalty",    "def__snap_counts_defense",     DL + DB + ("LB",),        "def"),
}

# which export each prefix comes from
SOURCES = {
    "rush": "rushing", "pass": "passing", "recv": "receiving",
    "blk": "blocking", "def": "defense",
}

# position -> broad group used for reporting
POS_GROUP = {
    "QB": "QB", "HB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
    "T": "T", "G": "G", "C": "C",
    "DI": "DI", "ED": "ED", "LB": "LB", "CB": "CB", "S": "S",
}

# PFF exports cover 2014-2025. 2020 is dropped: conferences played in isolation that
# year, which leaves the Massey system barely connected between them, and teams played
# 8.4 games on average against a normal 11.9 - one played 3. The ratings show it.
# Massey rating vs adjusted win pct by season: .58 .79 .62 .65 .71 .72 | .22 | .72 .72
# .71 .64 .64 - every season lands between .58 and .79 except 2020 at .22. Including it
# dragged the pooled figure from .67 to .48.
EXCLUDE_SEASONS = {2020}
YEARS = [y for y in range(2014, 2026) if y not in EXCLUDE_SEASONS]
from paths import PFF_DIR, require  # noqa: E402  (single definition, see paths.py)

require(PFF_DIR, "the PFF exports", "PFF_DIR")
