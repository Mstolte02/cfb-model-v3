"""Pull the CFBD player-season data needed to build a non-PFF facet set.

Three endpoints carry everything usable at the player level:
  ppa/players/season  - predicted points added, split all/pass/rush and by down
  player/usage        - share of the team's plays the player was involved in
  stats/player/season - raw counting stats, one row per (player, category, statType)

Cached to disk as JSON so the rest of the pipeline never re-hits the API.
"""
import json, os, sys, time
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = f"{HERE}/cfbd_cache"
BASE = "https://api.collegefootballdata.com"
# PFF only covers 2021-25, but CFBD's PPA goes back to 2014. The extra seasons are
# useless for the facet horse race (which needs both sources) and essential for the
# projection model, which currently trains on four season-to-season transitions.
YEARS = [int(y) for y in os.environ.get(
    "CFBD_YEARS", "2021,2022,2023,2024,2025").split(",")]

# categories that exist for player season stats
STAT_CATEGORIES = ["passing", "rushing", "receiving", "defensive",
                   "fumbles", "interceptions", "kicking", "punting"]


def key():
    p = os.environ.get("CFBD_KEY_FILE")
    if p and os.path.exists(p):
        return open(p).read().strip()
    k = os.environ.get("CFBD_API_KEY")
    if k:
        return k.strip()
    sys.exit("set CFBD_API_KEY or CFBD_KEY_FILE")


def get(sess, path, params, dest):
    """Fetch and cache; a present file short-circuits the request."""
    if os.path.exists(dest):
        return json.load(open(dest))
    for attempt in range(4):
        r = sess.get(f"{BASE}/{path}", params=params, timeout=90)
        if r.status_code == 200:
            d = r.json()
            json.dump(d, open(dest, "w"))
            return d
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        sys.exit(f"{path} {params} -> {r.status_code}: {r.text[:200]}")
    sys.exit(f"{path} {params}: gave up after retries")


def main():
    os.makedirs(CACHE, exist_ok=True)
    sess = requests.Session()
    sess.headers["Authorization"] = f"Bearer {key()}"

    for y in YEARS:
        d = get(sess, "ppa/players/season", {"year": y, "excludeGarbageTime": "false"},
                f"{CACHE}/ppa_{y}.json")
        print(f"ppa {y}: {len(d)} players")

        d = get(sess, "player/usage", {"year": y}, f"{CACHE}/usage_{y}.json")
        print(f"usage {y}: {len(d)} players")

        for cat in STAT_CATEGORIES:
            d = get(sess, "stats/player/season", {"year": y, "category": cat},
                    f"{CACHE}/stats_{y}_{cat}.json")
            print(f"  stats {y} {cat}: {len(d)} rows")

        # team-level advanced stats, for a sanity ceiling on what team data explains
        d = get(sess, "stats/season/advanced", {"year": y}, f"{CACHE}/team_adv_{y}.json")
        print(f"team advanced {y}: {len(d)} teams")

    # roster with class/height/weight, useful for the freshman diagnostics
    for y in YEARS + [2026]:
        try:
            d = get(sess, "roster", {"year": y}, f"{CACHE}/roster_{y}.json")
            print(f"roster {y}: {len(d)} players")
        except SystemExit as e:
            print(f"roster {y}: skipped ({e})")


if __name__ == "__main__":
    main()
