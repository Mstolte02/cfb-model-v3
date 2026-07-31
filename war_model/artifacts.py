"""Which WAR build the downstream stages read from.

Two builds coexist: the PFF-only one (stages 2-3, build_ratings + build_war) and the
PFF+CFBD hybrid (build_hybrid). They write the same artifacts under different names so
that either can be rebuilt and compared without clobbering the other.

Set WAR_BUILD=pff to pin the original. Hybrid is the default because it measurably
predicts better - +0.017 CV r on both the same-season and next-season targets.
"""
import os

BUILD = os.environ.get("WAR_BUILD", "hybrid").lower()
_PREFIX = "hybrid_" if BUILD == "hybrid" else ""

PLAYER_WAR = f"{_PREFIX}player_war.csv"
TEAM_RATINGS = f"{_PREFIX}team_ratings.csv"
FACET_WEIGHTS = f"{_PREFIX}facet_weights.csv"
WINS_MAP = f"{_PREFIX}wins_map.json"
FACET_WAR = f"{_PREFIX}facet_war.parquet"
PLAYER_WAR_BY_FACET = f"{_PREFIX}player_war_by_facet.csv"
