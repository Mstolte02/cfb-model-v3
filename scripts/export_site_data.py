"""Export editorial/site-only data that is not part of the trained model.

The prediction pipeline remains reproducible and sportsbook-independent.  This
module adds dated market snapshots, CFBD weekly lines, the prior final AP poll, and
ESPN athlete ids for presentation.  Every market row carries its source and date so
stale prices cannot masquerade as live model inputs.

Run after ``scripts.export_viz``::

    .\\venv\\Scripts\\python.exe -m scripts.export_site_data
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ROOT, PROJECTION_YEAR
from src.data.cfbd_client import _get

VIZ = ROOT / "viz" / "data"
RAW = ROOT / "data" / "raw"


def _rows(text: str, fields: tuple[str, ...]) -> list[dict]:
    """Parse pipe-delimited constants while keeping the source snapshot readable."""
    out = []
    for raw in text.strip().splitlines():
        vals = [v.strip() for v in raw.split("|")]
        out.append(dict(zip(fields, vals)))
    return out


TITLE_BETMGM = _rows("""
Ohio State|+600
Texas|+650
Notre Dame|+650
Indiana|+800
Oregon|+800
Georgia|+900
Miami|+1100
Alabama|+1500
LSU|+1500
Texas A&M|+1500
Texas Tech|+2200
Ole Miss|+2500
Michigan|+2500
Oklahoma|+3000
USC|+3500
Clemson|+5000
Tennessee|+5000
Penn State|+5000
""", ("team", "odds"))

TITLE_FANDUEL = _rows("""
Ohio State|+650
Notre Dame|+650
Indiana|+700
Texas|+750
Oregon|+950
Georgia|+1100
LSU|+1400
Texas A&M|+1500
Texas Tech|+1600
Miami|+1900
""", ("team", "odds"))

CFP_FANDUEL = _rows("""
Notre Dame|-750
Miami|-320
Indiana|-320
Oregon|-310
Ohio State|-260
Georgia|-230
Texas Tech|-185
Texas|-185
Texas A&M|+150
Alabama|+150
LSU|+155
Ole Miss|+170
Oklahoma|+200
USC|+250
Michigan|+310
BYU|+340
Tennessee|+350
Penn State|+350
Florida|+390
SMU|+450
Louisville|+460
Utah|+500
Washington|+500
Boise State|+600
Missouri|+700
Clemson|+700
Auburn|+700
Iowa|+700
Kansas State|+800
Houston|+800
South Carolina|+850
Pittsburgh|+900
""", ("team", "odds"))

CONFERENCE_DRAFTKINGS = _rows("""
Miami|ACC|-135
SMU|ACC|+700
Louisville|ACC|+1000
Clemson|ACC|+1600
Texas Tech|Big 12|-105
BYU|Big 12|+550
Utah|Big 12|+650
Kansas State|Big 12|+1400
Houston|Big 12|+1600
Ohio State|Big Ten|+180
Indiana|Big Ten|+250
Oregon|Big Ten|+260
USC|Big Ten|+1400
Michigan|Big Ten|+1500
Georgia|SEC|+260
Texas|SEC|+310
Texas A&M|SEC|+850
Alabama|SEC|+850
LSU|SEC|+900
Navy|American Athletic|+350
UTSA|American Athletic|+500
East Carolina|American Athletic|+550
Memphis|American Athletic|+600
Tulane|American Athletic|+680
Liberty|Conference USA|+260
Western Kentucky|Conference USA|+280
Jacksonville State|Conference USA|+450
Delaware|Conference USA|+700
Kennesaw State|Conference USA|+1000
Western Michigan|Mid-American|+360
Miami (OH)|Mid-American|+360
Toledo|Mid-American|+400
Ohio|Mid-American|+650
Central Michigan|Mid-American|+750
UNLV|Mountain West|+260
New Mexico|Mountain West|+265
North Dakota State|Mountain West|+300
Hawai'i|Mountain West|+450
Air Force|Mountain West|+600
Boise State|Pac-12|+170
San Diego State|Pac-12|+370
Texas State|Pac-12|+550
Fresno State|Pac-12|+650
Washington State|Pac-12|+800
James Madison|Sun Belt|+270
Old Dominion|Sun Belt|+460
Louisiana|Sun Belt|+700
Troy|Sun Belt|+800
Marshall|Sun Belt|+900
""", ("team", "conference", "odds"))

WIN_TOTALS_BETMGM = _rows("""
Notre Dame|11.5|+150|-200
Texas Tech|10.5|-220|+170
Miami|10.5|-125|+100
Indiana|10.5|-105|-120
Oregon|10.5|+110|-145
Georgia|9.5|-190|+145
Ohio State|9.5|-165|+130
Texas|9.5|+110|-145
SMU|8.5|-155|+120
Utah|8.5|-135|+105
Penn State|8.5|-135|+105
Texas A&M|8.5|-125|-105
Alabama|8.5|-125|+100
BYU|8.5|-125|+100
Kansas State|8.5|-110|-120
LSU|8.5|-110|-118
USC|8.5|+105|-125
Michigan|8.5|+120|-155
Ole Miss|7.5|-175|+135
Houston|7.5|-165|+130
Louisville|7.5|-165|+130
Washington|7.5|-160|+125
Pittsburgh|7.5|-155|+120
Virginia|7.5|-150|+115
Iowa|7.5|-145|+110
Oklahoma|7.5|-145|+115
Clemson|7.5|-135|+105
Tennessee|7.5|-115|-115
NC State|7.5|-115|-115
Florida|7.5|+110|-145
Arizona|7.5|+115|-145
Illinois|7.5|+130|-170
Virginia Tech|6.5|-160|+125
TCU|6.5|-155|+120
Wisconsin|6.5|-135|+105
Missouri|6.5|-135|+105
Arizona State|6.5|-130|+100
Auburn|6.5|-120|-110
UCLA|6.5|+100|-125
Georgia Tech|6.5|+105|-135
California|6.5|+110|-145
South Carolina|6.5|+115|-145
Nebraska|6.5|+120|-155
Minnesota|6.5|+130|-170
Florida State|6.5|+135|-175
Baylor|6.5|+135|-175
West Virginia|5.5|-175|+135
Oklahoma State|5.5|-175|+135
UCF|5.5|-170|+130
Kansas|5.5|-160|+125
Vanderbilt|5.5|-155|+120
Wake Forest|5.5|-125|+100
Duke|5.5|-115|-110
Iowa State|5.5|+110|-145
Cincinnati|5.5|+115|-150
Maryland|4.5|-185|+140
Kentucky|4.5|-155|+120
Rutgers|4.5|-155|+120
North Carolina|4.5|-135|+105
Mississippi State|4.5|-115|-115
Syracuse|4.5|-115|-110
Arkansas|4.5|+130|-170
Colorado|4.5|+135|-175
Michigan State|4.5|+145|-190
Stanford|3.5|-125|+100
Purdue|3.5|+115|-145
Boston College|3.5|+120|-155
""", ("team", "line", "over", "under"))

HEISMAN_BETMGM = _rows("""
CJ Carr|Notre Dame|QB|+750
Arch Manning|Texas|QB|+750
Trinidad Chambliss|Ole Miss|QB|+900
Dante Moore|Oregon|QB|+1000
Julian Sayin|Ohio State|QB|+1100
Darian Mensah|Miami|QB|+1100
Josh Hoover|Indiana|QB|+1200
Jeremiah Smith|Ohio State|WR|+1200
Gunner Stockton|Georgia|QB|+1600
Sam Leavitt|LSU|QB|+2000
John Mateer|Oklahoma|QB|+2000
Jayden Maiava|USC|QB|+2500
Marcel Reed|Texas A&M|QB|+2500
Keelon Russell|Alabama|QB|+2500
Will Hammond|Texas Tech|QB|+3000
Bryce Underwood|Michigan|QB|+3500
Malachi Toney|Miami|WR|+3500
LaNorris Sellers|South Carolina|QB|+3500
Byrum Brown|Auburn|QB|+4000
Austin Mack|Alabama|QB|+4000
""", ("player", "team", "position", "odds"))


SOURCES = {
    "betmgm_title": {"book": "BetMGM", "as_of": "2026-08-10",
        "url": "https://sports.betmgm.com/en/blog/college-football/college-football-national-championship-odds-bm06/"},
    "fanduel_title": {"book": "FanDuel", "as_of": "2026-07-22",
        "url": "https://www.al.com/betting/college-football-national-championship-odds/"},
    "fanduel_cfp": {"book": "FanDuel", "as_of": "2026-07-20",
        "url": "https://www.si.com/betting/2026-college-football-playoff-odds-notre-dame-will-return-to-cfb-playoff"},
    "draftkings_conference": {"book": "DraftKings", "as_of": "2026-07-24 (ACC 2026-06-12)",
        "url": "https://www.sportsbettingdime.com/college-football/futures/conference-title-odds/"},
    "betmgm_wins": {"book": "BetMGM", "as_of": "2026-07-31",
        "url": "https://sports.betmgm.com/en/blog/college-football/college-football-win-totals-odds-ncaaf-futures-bets-analysis-bm06/"},
    "betmgm_heisman": {"book": "BetMGM", "as_of": "2026-08-10",
        "url": "https://sports.betmgm.com/en/blog/college-football/heisman-trohpy-odds-favorites-to-win-bm06/"},
    "cfbd_lines": {"book": "Multiple", "as_of": "2026-08-12",
        "url": "https://api.collegefootballdata.com/"},
}


def american(value: str) -> int:
    return int(value.replace("+", ""))


def normalize_market(rows, numeric=("odds", "over", "under"), floats=("line",)):
    out = []
    for row in rows:
        row = dict(row)
        for key in numeric:
            if key in row:
                row[key] = american(row[key])
        for key in floats:
            if key in row:
                row[key] = float(row[key])
        out.append(row)
    return out


def export_weekly_lines() -> list[dict]:
    # Prefer the append-only tracker once it exists. It records exactly when we
    # observed every provider price; the ordinary cache has no timestamp semantics.
    from scripts.capture_market_snapshot import latest_weekly
    tracked = latest_weekly()
    if tracked:
        return tracked
    raw = _get("/lines", {"year": PROJECTION_YEAR, "seasonType": "regular"},
               f"lines_{PROJECTION_YEAR}.json")
    rows = []
    for game in raw:
        books = {}
        for line in game.get("lines") or []:
            provider = line.get("provider")
            if not provider:
                continue
            books[provider] = {k: line.get(k) for k in (
                "spread", "spreadOpen", "overUnder", "overUnderOpen",
                "homeMoneyline", "awayMoneyline")}
        if books:
            rows.append({"id": game["id"], "week": game.get("week"),
                         "start": game.get("startDate"),
                         "home": game.get("homeTeam"), "away": game.get("awayTeam"),
                         "books": books})
    return rows


def export_poll() -> list[dict]:
    polls = json.loads((RAW / "rankings_2025_post.json").read_text())
    ap = next(p for week in polls for p in week["polls"] if p["poll"] == "AP Top 25")
    return [{"rank": r["rank"], "team": r["school"], "points": r.get("points")}
            for r in ap["ranks"]]


def norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower().replace("jr", "").replace("iii", ""))


def export_headshots() -> dict[str, str]:
    roster = _get("/roster", {"year": PROJECTION_YEAR}, f"roster_{PROJECTION_YEAR}.json")
    wanted = json.loads((VIZ / "players.json").read_text())
    roster_map = {}
    for p in roster:
        key = (p.get("team"), norm_name(f"{p.get('firstName', '')} {p.get('lastName', '')}"))
        roster_map[key] = str(p.get("id"))
    out = {}
    for team, group in wanted.items():
        for player in group.get("players", []):
            pid = roster_map.get((team, norm_name(player["n"])))
            if pid:
                out[f"{team}\0{player['n']}"] = (
                    f"https://a.espncdn.com/i/headshots/college-football/players/full/{pid}.png")
    return out


def main() -> None:
    VIZ.mkdir(parents=True, exist_ok=True)
    sources = json.loads(json.dumps(SOURCES))
    tracking_path = VIZ / "market_tracking.json"
    if tracking_path.exists():
        tracking = json.loads(tracking_path.read_text())
        sources["cfbd_lines"].update({
            "as_of": tracking.get("checked_at"),
            "timestamp_semantics": tracking.get("timestamp_semantics")})
    odds = {
        "season": PROJECTION_YEAR,
        "sources": sources,
        "markets": {
            "national_title": {
                "BetMGM": normalize_market(TITLE_BETMGM),
                "FanDuel": normalize_market(TITLE_FANDUEL),
            },
            "make_cfp": {"FanDuel": normalize_market(CFP_FANDUEL)},
            "conference_title": {"DraftKings": normalize_market(CONFERENCE_DRAFTKINGS)},
            "win_totals": {"BetMGM": normalize_market(WIN_TOTALS_BETMGM)},
            "heisman": {"BetMGM": normalize_market(HEISMAN_BETMGM)},
        },
        "weekly": export_weekly_lines(),
    }
    editorial = {
        "prior_final_ap": export_poll(),
        "headshots": export_headshots(),
        "headshot_source": "ESPN athlete headshots matched through CFBD roster ids",
    }
    (VIZ / "odds.json").write_text(json.dumps(odds, indent=1, allow_nan=False))
    (VIZ / "editorial.json").write_text(json.dumps(editorial, indent=1, allow_nan=False))
    print(f"odds.json: {len(odds['weekly'])} games with lines")
    print(f"editorial.json: {len(editorial['headshots'])} player headshots")


if __name__ == "__main__":
    main()
