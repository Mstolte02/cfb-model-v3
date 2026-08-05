"""Build FBS team-season outcome table (2014-2025) and map PFF team names onto it."""
import csv, os, re, json
from collections import defaultdict

from paths import GAMES_CSV, PFF_DIR, require

GAMES = str(require(GAMES_CSV, "the CFB games table", "CFB_GAMES_CSV"))
GAMES_2025 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "games_2025.json")
PFF = str(require(PFF_DIR, "the PFF exports", "PFF_DIR"))
OUT = os.path.dirname(os.path.abspath(__file__))
from facets import YEARS   # 2014-2025 less the COVID season; see facets.py

# ---------- team-season records ----------
# Local games.csv was pulled Jul-2025, so it only carries finals through 2024;
# 2025 finals come from the CFBD API pull in games_2025.json.
rec = defaultdict(lambda: dict(g=0, w=0, pf=0, pa=0, conf=None))


def add_game(season, hteam, hpts, hcls, hconf, ateam, apts, acls, aconf):
    if hpts is None or apts is None or hpts == apts:
        return
    for team, pts, opts, cls, conf in (
        (hteam, hpts, apts, hcls, hconf),
        (ateam, apts, hpts, acls, aconf),
    ):
        if cls != "fbs":
            continue
        d = rec[(team, season)]
        d["g"] += 1
        d["w"] += 1 if pts > opts else 0
        d["pf"] += pts
        d["pa"] += opts
        d["conf"] = conf


for r in csv.DictReader(open(GAMES)):
    # games.csv carries finals through 2024; 2025 comes from the CFBD pull below
    if int(r["season"]) not in set(YEARS) - {2025}:
        continue
    if r["status"] != "completed" or not r["home_points"] or not r["away_points"]:
        continue
    add_game(int(r["season"]), r["home_team"], float(r["home_points"]),
             r["home_classification"], r["home_conference"],
             r["away_team"], float(r["away_points"]),
             r["away_classification"], r["away_conference"])

for g in json.load(open(GAMES_2025)):
    if not g.get("completed"):
        continue
    add_game(2025, g["homeTeam"], g["homePoints"], g["homeClassification"], g["homeConference"],
             g["awayTeam"], g["awayPoints"], g["awayClassification"], g["awayConference"])

wins = {}
for (team, yr), d in rec.items():
    if d["g"] < 8:  # incomplete/partial FBS seasons
        continue
    wins[(team, yr)] = dict(
        team=team, season=yr, games=d["g"], wins=d["w"],
        win_pct=d["w"] / d["g"],
        margin_pg=(d["pf"] - d["pa"]) / d["g"],
        pf_pg=d["pf"] / d["g"], pa_pg=d["pa"] / d["g"],
        conference=d["conf"],
    )

# ---------- PFF -> CFBD name mapping ----------
pff_teams = set()
for y in YEARS:
    for r in csv.DictReader(open(f"{PFF}/rushing_{y}.csv")):
        pff_teams.add(r["team_name"])

cfbd_teams = sorted({t for (t, _) in wins})


def norm(s):
    s = s.lower()
    s = s.replace("&", "and").replace("'", "").replace("-", " ").replace("(", " ").replace(")", " ")
    s = re.sub(r"\bst\b|\bstate\b", "state", s)
    s = re.sub(r"\buniv\b|\buniversity\b", "", s)
    s = re.sub(r"[^a-z ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


# hand overrides for PFF abbreviations that normalization can't recover
OVERRIDES = {
    "ARK STATE": "Arkansas State", "BOWL GREEN": "Bowling Green", "BOSTON COL": "Boston College",
    "C MICHIGAN": "Central Michigan", "CAL": "California", "COAST CAR": "Coastal Carolina",
    "COLO STATE": "Colorado State", "DOMINION": "Old Dominion", "E CAROLINA": "East Carolina",
    "E MICHIGAN": "Eastern Michigan", "FAU": "Florida Atlantic", "FIU": "Florida International",
    "GA SOUTHRN": "Georgia Southern", "GA STATE": "Georgia State", "GA TECH": "Georgia Tech",
    "HAWAII": "Hawai'i", "JAX STATE": "Jacksonville State", "JMU": "James Madison",
    "KENN STATE": "Kennesaw State", "LA LAFAYET": "Louisiana", "LA MONROE": "UL Monroe",
    "LA TECH": "Louisiana Tech", "MASS": "Massachusetts", "MIAMI FL": "Miami",
    "MIAMI OH": "Miami (OH)", "MID TENN": "Middle Tennessee", "MISS STATE": "Mississippi State",
    "MO STATE": "Missouri State", "N ILLINOIS": "Northern Illinois", "N MEXICO": "New Mexico",
    "N MEXICO ST": "New Mexico State", "NM STATE": "New Mexico State", "NC STATE": "NC State",
    "NORTH TEX": "North Texas", "NTHWESTERN": "Northwestern", "OKLA STATE": "Oklahoma State",
    "OREGON ST": "Oregon State", "PENN STATE": "Penn State", "PITT": "Pittsburgh",
    "S ALABAMA": "South Alabama", "S CAROLINA": "South Carolina", "S DIEGO ST": "San Diego State",
    "S FLORIDA": "South Florida", "S HOUSTON": "Sam Houston", "S JOSE ST": "San José State",
    "S MISS": "Southern Miss", "SAN JOSE ST": "San José State", "TEXAS AM": "Texas A&M",
    "TEXAS ST": "Texas State", "TEXAS TECH": "Texas Tech", "UCONN": "UConn",
    "UNC": "North Carolina", "VA TECH": "Virginia Tech", "W KENTUCKY": "Western Kentucky",
    "W MICHIGAN": "Western Michigan", "W VIRGINIA": "West Virginia", "WAKE": "Wake Forest",
    "WASH STATE": "Washington State", "APP STATE": "App State", "ARIZONA ST": "Arizona State",
    "BALL ST": "Ball State", "BOISE ST": "Boise State", "FLORIDA ST": "Florida State",
    "FRESNO ST": "Fresno State", "IOWA STATE": "Iowa State", "KANSAS ST": "Kansas State",
    "KENT STATE": "Kent State", "MICH STATE": "Michigan State", "UTAH STATE": "Utah State",
    "JVILLE ST": "Jacksonville State", "MIDDLE TN": "Middle Tennessee",
    "N CAROLINA": "North Carolina", "N TEXAS": "North Texas",
    "NEW MEX ST": "New Mexico State", "NWESTERN": "Northwestern",
    "SM HOUSTON": "Sam Houston", "SO MISS": "Southern Miss", "UMASS": "Massachusetts",
    "USF": "South Florida", "W GEORGIA": "West Georgia", "KENNESAW": "Kennesaw State",
    "JAMES MAD": "James Madison", "UTAH ST": "Utah State",
}

cfbd_by_norm = {norm(t): t for t in cfbd_teams}
mapping, unmatched = {}, []
for p in sorted(pff_teams):
    if p in OVERRIDES:
        mapping[p] = OVERRIDES[p]
        continue
    n = norm(p)
    if n in cfbd_by_norm:
        mapping[p] = cfbd_by_norm[n]
        continue
    # unique prefix match as a last resort
    cands = [t for t in cfbd_teams if norm(t).startswith(n) or n.startswith(norm(t))]
    if len(cands) == 1:
        mapping[p] = cands[0]
    else:
        unmatched.append((p, cands))

bad = [(p, m) for p, m in mapping.items() if m not in cfbd_teams]
print(f"PFF teams: {len(pff_teams)} | mapped: {len(mapping)} | unmatched: {len(unmatched)}")
for p, c in unmatched:
    print("  UNMATCHED:", p, "->", c)
for p, m in bad:
    print("  BAD TARGET:", p, "->", m)

# every FBS team should be claimed by exactly one PFF name
dupes = defaultdict(list)
for p, m in mapping.items():
    dupes[m].append(p)
for m, ps in dupes.items():
    if len(ps) > 1:
        print("  COLLISION:", m, ps)

with open(f"{OUT}/team_map.json", "w") as f:
    json.dump(mapping, f, indent=1, sort_keys=True)

with open(f"{OUT}/team_seasons.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(next(iter(wins.values())).keys()))
    w.writeheader()
    for k in sorted(wins):
        w.writerow(wins[k])
print(f"team-seasons written: {len(wins)}")

# coverage check: PFF team-seasons that find no record
miss = 0
for y in YEARS:
    ts = {r["team_name"] for r in csv.DictReader(open(f"{PFF}/rushing_{y}.csv"))}
    for t in ts:
        if (mapping.get(t), y) not in wins:
            print(f"  NO RECORD: {t} {y} -> {mapping.get(t)}")
            miss += 1
print("missing team-seasons:", miss)
