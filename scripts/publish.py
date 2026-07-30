"""Build the publishable copy of the site into dist/.

viz/ is the working copy the local server renders. dist/ is what goes to GitHub
Pages. They are almost the same thing, which is the point - the fewer differences
between what you look at while building and what other people get, the fewer ways
the published site can be wrong in a way you never see.

Two differences, both deliberate:

  war_diagnostics.html   the WAR build's internal report - facet weights, model
                         comparisons, per-player tables. A working document.
  --no-players           OPTIONAL. Strips the per-player rows out of players.json,
                         leaving team and position-group totals. Off by default: the
                         site ships the full roster view. It exists because those rows
                         are derived from licensed PFF grades, so if that ever needs
                         to change it is one flag rather than a rewrite.

When --no-players is used the stripping happens HERE, at build time, rather than by
asking the app to hide things: a hidden field is still a downloaded field. A number
that is not in dist/ cannot be read out of the page.

Deliberately stdlib-only, and it does NOT import config: the GitHub Action that
publishes the site runs it on a bare Python with none of the model's dependencies
installed. Everything it needs is already committed under viz/.

Run: python3 scripts/publish.py [--no-players]
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIZ = ROOT / "viz"
DIST = ROOT / "dist"

# What survives --no-players. Everything else in a players.json entry is the
# per-player array itself.
TEAM_KEYS = ("total", "winsTotal", "projWins", "replWins", "context", "scale",
             "byGroup")


def strip_players(path: Path) -> tuple[int, int]:
    data = json.loads(path.read_text())
    dropped = sum(len(v.get("players", [])) for v in data.values())
    out = {t: {k: v[k] for k in TEAM_KEYS if k in v} for t, v in data.items()}
    path.write_text(json.dumps(out, allow_nan=False))
    return len(out), dropped


def main(no_players: bool = False):
    if DIST.exists():
        shutil.rmtree(DIST)
    shutil.copytree(VIZ, DIST, ignore=shutil.ignore_patterns(
        "war_diagnostics.html", ".DS_Store"))

    # GitHub Pages runs everything through Jekyll otherwise, which drops files and
    # folders beginning with an underscore.
    (DIST / ".nojekyll").write_text("")

    players = DIST / "data" / "players.json"
    if no_players:
        teams, dropped = strip_players(players)
        # The one failure that would matter, so check rather than trust.
        leaked = [t for t, v in json.loads(players.read_text()).items()
                  if "players" in v]
        if leaked:
            raise SystemExit(f"ABORT: player rows survived for {len(leaked)} teams")
        print(f"  players.json: {teams} teams, {dropped:,} player rows removed")
    else:
        data = json.loads(players.read_text())
        n = sum(len(v.get("players", [])) for v in data.values())
        print(f"  players.json: {len(data)} teams, {n:,} players included")

    total = sum(f.stat().st_size for f in DIST.rglob("*") if f.is_file())
    print(f"dist/ built: {total / 1e6:.1f} MB")
    print(f"-> {DIST}")


if __name__ == "__main__":
    main(no_players="--no-players" in sys.argv[1:])
