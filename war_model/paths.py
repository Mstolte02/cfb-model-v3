"""External data locations for the WAR build, re-exported from the repo's config.py.

war_model/ runs with its own directory as the working directory, so the repo root is
not importable by default. Rather than keep a second copy of the paths here - which is
how build_massey, build_wins, facets, build_roster_2026 and src/data/pff.py each came
to carry their own absolute string, and how a machine that moved one of them got four
modules agreeing and one not - this puts the root on sys.path and takes the single
definition from config.

`require` fails with the variable name that moves the file. Nothing in this build is
allowed to shrug off a missing input any more.
"""
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (  # noqa: E402
    CFB_EXTERNAL, GAMES_CSV, LOGO_DIR, PFF_DIR, PFSN_MASTER, TWODEEP_2026, require,
)

# Where this build's own artifacts live. WAR_DIR is honoured for the same reason
# src/data/war.py honours it: so a build made elsewhere can be pointed at.
WAR_DIR = Path(os.environ.get("WAR_DIR", Path(__file__).resolve().parent))

__all__ = ["CFB_EXTERNAL", "GAMES_CSV", "LOGO_DIR", "PFF_DIR", "PFSN_MASTER",
           "TWODEEP_2026", "WAR_DIR", "require"]
