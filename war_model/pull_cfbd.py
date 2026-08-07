"""Pull the CFBD player-season data needed to build a non-PFF facet set.

Three endpoints carry everything usable at the player level:
  ppa/players/season  - predicted points added, split all/pass/rush and by down
  player/usage        - share of the team's plays the player was involved in
  stats/player/season - raw counting stats, one row per (player, category, statType)

Plus the two inputs the roster and recruiting stages need, which used to be pulled by
hand and so had no record of WHEN or with WHAT parameters:
  recruiting/players  - high school classes, the prior for players with no snaps
  player/portal       - transfers, which supply a rating when the HS class does not

Cached to disk as JSON so the rest of the pipeline never re-hits the API.

AN EMPTY PAYLOAD IS NOT A CACHE HIT. CFBD publishes a season's data when it has it,
so asking early returns `[]` - and the first version of this file wrote that `[]` to
disk and then short-circuited on it forever. The 2026 roster was cached empty on
25-Jul, went live some time after, and the pipeline kept reading two bytes and falling
back to the two-deep's class years for a fortnight. Nothing reported an error; the
projection just quietly served a variable it was not trained on. So an empty response
is never written, an empty cache file is re-fetched, and a year that is genuinely not
published yet costs one request per run instead of silently freezing.

Run: ../venv/bin/python pull_cfbd.py [--refresh [PATTERN]]
     --refresh with no PATTERN re-pulls everything; with one, only cache files whose
     name contains it (e.g. --refresh rec_ for the recruiting classes).
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

# The season being projected. Its roster, recruiting class and portal are the three
# inputs that go live partway through the off-season rather than all at once.
PROJECTION_YEAR = 2026

# A player on a 2015 roster was recruited around 2011-14, so the classes reach back
# four years before the earliest WAR season; build_recruiting.py declares the same
# range and this list is what fills it.
REC_YEARS = list(range(2011, PROJECTION_YEAR + 1))

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


def cached(dest, refresh):
    """The cached payload, or None if it should be re-fetched.

    Missing, unparseable, empty, and explicitly refreshed all mean the same thing to
    the caller; only a file with rows in it counts as a hit. See the module docstring
    on why empty does not.
    """
    if refresh is not None and refresh in os.path.basename(dest):
        return None
    if not os.path.exists(dest):
        return None
    try:
        d = json.load(open(dest))
    except json.JSONDecodeError:
        return None
    return d or None


def get(sess, path, params, dest, refresh=None):
    """Fetch and cache; a non-empty cache file short-circuits the request."""
    d = cached(dest, refresh)
    if d is not None:
        return d
    for attempt in range(4):
        r = sess.get(f"{BASE}/{path}", params=params, timeout=90)
        if r.status_code == 200:
            d = r.json()
            if d:
                json.dump(d, open(dest, "w"))
            return d
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        sys.exit(f"{path} {params} -> {r.status_code}: {r.text[:200]}")
    sys.exit(f"{path} {params}: gave up after retries")


def main(refresh=None):
    os.makedirs(CACHE, exist_ok=True)
    sess = requests.Session()
    sess.headers["Authorization"] = f"Bearer {key()}"
    pending = []   # (label, what it is waiting for) - reported at the end, not buried

    for y in YEARS:
        d = get(sess, "ppa/players/season", {"year": y, "excludeGarbageTime": "false"},
                f"{CACHE}/ppa_{y}.json", refresh)
        print(f"ppa {y}: {len(d)} players")

        d = get(sess, "player/usage", {"year": y}, f"{CACHE}/usage_{y}.json", refresh)
        print(f"usage {y}: {len(d)} players")

        for cat in STAT_CATEGORIES:
            d = get(sess, "stats/player/season", {"year": y, "category": cat},
                    f"{CACHE}/stats_{y}_{cat}.json", refresh)
            print(f"  stats {y} {cat}: {len(d)} rows")

        # team-level advanced stats, for a sanity ceiling on what team data explains
        d = get(sess, "stats/season/advanced", {"year": y},
                f"{CACHE}/team_adv_{y}.json", refresh)
        print(f"team advanced {y}: {len(d)} teams")

    # roster with class/height/weight. The projection year's is the one that arrives
    # late, and it is the source of class on BOTH sides of project_2026_v2 - so when
    # it is empty the model serves a class variable it was not trained on.
    for y in YEARS + [PROJECTION_YEAR]:
        d = get(sess, "roster", {"year": y}, f"{CACHE}/roster_{y}.json", refresh)
        print(f"roster {y}: {len(d)} players")
        if not d:
            pending.append((f"roster {y}", "CFBD has not published it yet"))

    # recruiting classes and the portal live beside the build, not in cfbd_cache,
    # because build_recruiting.py reads them from there.
    for y in REC_YEARS:
        d = get(sess, "recruiting/players", {"year": y}, f"{HERE}/rec_{y}.json", refresh)
        print(f"recruiting {y}: {len(d)} recruits")
        if not d:
            pending.append((f"recruiting {y}", "CFBD has not published it yet"))

    d = get(sess, "player/portal", {"year": PROJECTION_YEAR},
            f"{HERE}/portal_{PROJECTION_YEAR}.json", refresh)
    print(f"portal {PROJECTION_YEAR}: {len(d)} transfers")

    if pending:
        print("\nstill unpublished (re-run to pick these up; nothing was cached):")
        for label, why in pending:
            print(f"  {label}: {why}")


if __name__ == "__main__":
    a = sys.argv[1:]
    # --refresh alone means everything; --refresh PATTERN narrows it to matching files
    r = None
    if "--refresh" in a:
        i = a.index("--refresh")
        r = a[i + 1] if len(a) > i + 1 else ""
    main(refresh=r)
