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
import hashlib
import re
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


def fingerprint_assets(site: Path):
    """Make deployed HTML request the exact JS/CSS copied in this build.

    A cached six-feature v3 app running against a four-feature v4 model turns missing
    array slots into NaN probabilities and scores. Put the hash in the filename, not
    only the query string: GitHub Pages/Fastly can reuse the path-level object while
    ignoring a query-string revision. A new path makes each build unambiguous and
    also recovers cleanly if one cached asset is ever corrupted.
    """
    index = site / "index.html"
    html = index.read_text()
    versions = {}
    for name in ("app.js", "style.css", "team-card.js"):
        source = site / name
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        target_name = f"{source.stem}.{digest}{source.suffix}"
        source.rename(site / target_name)
        pattern = re.escape(name) + r'(?:\?v=[^"\']*)?'
        html, count = re.subn(pattern, target_name, html)
        if count != 1:
            raise RuntimeError(f"expected exactly one {name} reference; found {count}")
        versions[name] = target_name
    index.write_text(html)
    return versions


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
    versions = fingerprint_assets(DIST)
    print("  assets: " + ", ".join(versions.values()))

    # Globbed rather than named. There is one players.json now - the site shipped
    # several complete builds at one point, each with its own players*.json, and
    # stripping only the first published the licensed rows under the other name while
    # reporting success. The glob costs nothing and cannot forget a file.
    rosters = sorted((DIST / "data").glob("players*.json"))
    if not rosters:
        print("  [warn] no players*.json in dist/")
    for players in rosters:
        if no_players:
            teams, dropped = strip_players(players)
            # The one failure that would matter, so check rather than trust.
            leaked = [t for t, v in json.loads(players.read_text()).items()
                      if "players" in v]
            if leaked:
                raise SystemExit(
                    f"ABORT: player rows survived in {players.name} for "
                    f"{len(leaked)} teams")
            print(f"  {players.name}: {teams} teams, {dropped:,} player rows removed")
        else:
            data = json.loads(players.read_text())
            n = sum(len(v.get("players", [])) for v in data.values())
            print(f"  {players.name}: {len(data)} teams, {n:,} players included")

    total = sum(f.stat().st_size for f in DIST.rglob("*") if f.is_file())
    print(f"dist/ built: {total / 1e6:.1f} MB")
    print(f"-> {DIST}")


if __name__ == "__main__":
    main(no_players="--no-players" in sys.argv[1:])
