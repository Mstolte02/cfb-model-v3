"""Pull public college football injury reports into the availability event stream.

Three public reports are read and merged:

  * MyGameSim  - every FBS team, full names, status and injury type. The backbone.
  * Covers     - first initial + last name, a dated status and a short note. Used to
                 corroborate MyGameSim and to add players it does not list.
  * Rotowire   - full names and an "OFS" (out for season) flag, but the free table
                 stops at seven rows, so it only ever adds a handful of players.

Each run is a snapshot. A player whose merged status changed gets a new event in
``war_model/availability_events_2026.csv`` (source_type ``injury_report``); a player
an earlier report listed who has since dropped off every report gets a ``clear``
event. Events from other sources (team press conferences, manual audits) are never
cleared or overridden here, because those are direct statements and a public report
missing a name is weaker evidence than the school saying it.

Statuses keep their report wording (out, doubtful, questionable). The live WAR build
turns them into an expected availability share - see src/inseason_war.AVAIL_SHARE.
Probable players are expected to play and are not recorded.

    python -m scripts.sync_injury_reports            # fetch, merge, append events
    python -m scripts.sync_injury_reports --dry-run  # print the diff only
"""
from __future__ import annotations

import argparse
import csv
import html as H
import json
import re
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "war_model"))
from build_roster_2026 import norm_name  # noqa: E402

EVENTS = ROOT / "war_model" / "availability_events_2026.csv"
SNAPSHOT = ROOT / "data" / "live" / "injury_report_2026.json"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/129 Safari/537.36")
SOURCES = {
    "mygamesim": "https://www.mygamesim.com/cfb/college-football-injuries.asp",
    "covers": "https://www.covers.com/sport/football/ncaaf/injuries",
    "rotowire": "https://www.rotowire.com/cfootball/tables/injury-report.php?pos=ALL",
}
# Expected share of the next game each report status implies. Matches
# src/inseason_war.AVAIL_SHARE; a player listed as probable is expected to play.
SHARE = {"out": 0.0, "doubtful": 0.25, "questionable": 0.5, "probable": 1.0}
# Report spellings that differ from the model's CFBD school names.
TEAM_ALIAS = {
    "UNC": "North Carolina", "TX-San Antonio": "UTSA", "Miami (Ohio)": "Miami (OH)",
    "Connecticut": "UConn", "ULM": "UL Monroe", "Hawaii": "Hawai'i",
    "San Jose State": "San José State", "Southern Miss": "Southern Miss",
    "Louisiana-Lafayette": "Louisiana", "Louisiana-Monroe": "UL Monroe",
    "Miami (FL)": "Miami", "Miami FL": "Miami", "Ole Miss": "Ole Miss",
    "FIU": "Florida International", "UMass": "Massachusetts",
    "Sam Houston State": "Sam Houston", "Appalachian State": "App State",
    "Appalachian St": "App State", "Mississippi": "Ole Miss",
    "North Carolina State": "NC State",
}


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": url})
    with urllib.request.urlopen(req, timeout=45) as r:
        return r.read().decode("utf-8", "replace")


def norm_status(report: str) -> str | None:
    """Status from a report's headline (never its prose, which mentions "out" freely)."""
    t = report.lower().strip()
    if re.match(r"(ofs|ir)\b", t) or re.search(r"\bout\b|for (the )?season", t):
        return "out"
    for s in ("doubtful", "questionable", "probable"):
        if re.search(rf"\b{s}\b", t):
            return s
    return None


def status_of_share(share: float) -> str | None:
    """Nearest report status to an averaged share; None means expected to play."""
    if share <= .125:
        return "out"
    if share <= .375:
        return "doubtful"
    if share <= .75:
        return "questionable"
    return None


# ---------------------------------------------------------------- parsers
def parse_mygamesim(page: str) -> list[dict]:
    rows = []
    for head, block in re.findall(r"<h3>(.*?)</h3>(.*?)</table>", page, re.S):
        cells = [H.unescape(re.sub(r"<[^>]+>", " ", c)).strip()
                 for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", block, re.S)]
        cells = [re.sub(r"\s+", " ", c) for c in cells]
        if "Injury" not in cells[:4]:
            continue
        team = H.unescape(re.sub(r"<[^>]+>", "", head)).strip()
        body = cells[4:]
        for i in range(0, len(body) - 3, 4):
            name, injury, status, details = body[i:i + 4]
            if name:
                rows.append({"source": "mygamesim", "team": team, "player": name,
                             "injury": injury, "report": status, "note": details})
    return rows


def parse_covers(page: str) -> list[dict]:
    rows = []
    for part in re.split(r'class="covers-CoversMatchups-imgLink"', page)[1:]:
        m = re.search(r">\s*([^<]+?)\s*<br>", part)
        if not m:
            continue
        team = H.unescape(m.group(1)).strip()
        table = part.split("covers-CoversMatchups-imgLink")[0]
        for tr in re.finditer(
                r"<span class='player-link'>(.*?)</span>\s*</td>\s*<td>(.*?)</td>\s*"
                r"<td><b>(.*?)</b><br>\((.*?)\)</td>(.*?)injuryCopy\">(.*?)</div>",
                table, re.S):
            name = re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", "", tr.group(1)))).strip()
            rows.append({"source": "covers", "team": team, "player": name,
                         "pos": tr.group(2).strip(),
                         "report": re.sub(r"\s+", " ", tr.group(3)).strip(),
                         "date": re.sub(r"\s+", " ", tr.group(4)).strip(),
                         "note": re.sub(r"\s+", " ", H.unescape(tr.group(6))).strip()})
    return rows


def parse_rotowire(page: str) -> list[dict]:
    try:
        data = json.loads(page)
    except ValueError:
        return []
    return [{"source": "rotowire", "team": r["team"], "player": r["player"],
             "injury": r.get("injury_type", ""),
             "report": "Out for season" if r.get("IR") == "OFS" else r.get("IR", ""),
             "note": f"Return {r.get('ReturnDate', '')}".strip()} for r in data]


PARSERS = {"mygamesim": parse_mygamesim, "covers": parse_covers,
           "rotowire": parse_rotowire}


# ---------------------------------------------------------------- roster matching
def roster() -> pd.DataFrame:
    from src.data import war
    r = war.player_contributions()
    r["key"] = r.player.map(norm_name)
    r["last"] = r.key.str.split().str[-1]
    r["initial"] = r.key.str[0]
    return r


def resolve_team(name: str, teams: set[str]) -> str | None:
    name = TEAM_ALIAS.get(name, name)
    if name in teams:
        return name
    st = name.replace(" St.", " State").replace(" St", " State")
    return st if st in teams else None


def match(row: dict, R: pd.DataFrame, teams: set[str]) -> str | None:
    team = resolve_team(row["team"], teams)
    if not team:
        return None
    room = R[R.team == team]
    key = norm_name(row["player"])
    hit = room[room.key == key]
    if len(hit) == 1:
        return hit.player.iloc[0]
    parts = key.split()
    if len(parts) >= 2:   # Covers "N. Prongos" form, or a nickname first name
        cand = room[(room["last"] == parts[-1])]
        if len(parts[0]) == 1:
            cand = cand[cand.initial == parts[0]]
        if len(cand) == 1:
            return cand.player.iloc[0]
    return None


# ---------------------------------------------------------------- event diff
def read_events() -> list[dict]:
    with EVENTS.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latest_by_player(events: list[dict]) -> dict[tuple[str, str], dict]:
    out = {}
    for e in sorted(events, key=lambda r: (r["observed_at"], r["event_id"])):
        out[(e["team"], e["player"])] = e
    return out


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", norm_name(s)).strip("-")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    raw, fetched = [], {}
    for src, url in SOURCES.items():
        try:
            rows = PARSERS[src](fetch(url))
        except Exception as exc:   # one dead source must not stop the others
            print(f"[warn] {src}: {exc}")
            rows = []
        fetched[src] = len(rows)
        raw += rows
    print("fetched", fetched)
    if fetched["mygamesim"] < 100:
        print("[stop] MyGameSim returned too few rows to treat as a full report")
        return 1

    R = roster()
    teams = set(R.team)
    merged: dict[tuple[str, str], dict] = {}
    unmatched = Counter()
    for r in raw:
        status = norm_status(r["report"])
        # A headline of "Out" whose note says the season is over is still just out;
        # the note matters only when the headline is softer than the prose.
        if r.get("note") and re.search(r"remainder of the season|season[- ]ending",
                                       r["note"].lower()):
            status = "out"
        player = match(r, R, teams)
        if player is None:
            unmatched[r["source"]] += 1
            continue
        team = resolve_team(r["team"], teams)
        m = merged.setdefault((team, player), {"team": team, "player": player,
                                               "sources": {}, "shares": []})
        m["sources"][r["source"]] = {k: r[k] for k in ("report", "note") if r.get(k)}
        if status:
            m["shares"].append(SHARE[status])
    # Sources disagree on about one shared player in four; averaging their implied
    # availability keeps the contribution continuous instead of trusting the harshest.
    for m in merged.values():
        shares = m.pop("shares")
        m["share"] = round(sum(shares) / len(shares), 3) if shares else 1.0
        m["status"] = status_of_share(m["share"])
    print("unmatched (not on the projected two-deep)", dict(unmatched))

    listed = {k: v for k, v in merged.items() if v["status"]}
    SNAPSHOT.write_text(json.dumps({
        "season": 2026, "fetched_at": stamp, "rows": fetched,
        "players": sorted(listed.values(), key=lambda v: (v["team"], v["player"])),
    }, indent=1, ensure_ascii=False))

    events = read_events()
    latest = latest_by_player(events)
    new = []
    for (team, player), v in sorted(listed.items()):
        prev = latest.get((team, player))
        if prev and prev["status"] == v["status"]:
            continue
        if prev and prev["source_type"] != "injury_report" and prev["status"] == "out":
            continue   # a direct team statement outranks a public report
        detail = "; ".join(f"{s}: {' - '.join(d.values())}" for s, d in v["sources"].items())
        new.append({"event_id": f"rpt-{slug(team)}-{slug(player)}-{now:%Y%m%d%H%M}",
                    "observed_at": stamp, "effective_at": f"{now:%Y-%m-%d}",
                    "team": team, "player": player, "status": v["status"],
                    "source_type": "injury_report",
                    "source_ref": "+".join(sorted(v["sources"])), "note": detail[:240]})
    for (team, player), prev in sorted(latest.items()):
        if (prev["source_type"] == "injury_report" and prev["status"] != "clear"
                and (team, player) not in listed):
            new.append({"event_id": f"rpt-{slug(team)}-{slug(player)}-{now:%Y%m%d%H%M}-clear",
                        "observed_at": stamp, "effective_at": f"{now:%Y-%m-%d}",
                        "team": team, "player": player, "status": "clear",
                        "source_type": "injury_report", "source_ref": "absent",
                        "note": "No longer on any injury report"})

    counts = Counter(e["status"] for e in new)
    print(f"{len(listed)} players on reports; {len(new)} new events {dict(counts)}")
    if args.dry_run or not new:
        return 0
    with EVENTS.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(events[0].keys()))
        w.writerows(new)
    print(f"-> {EVENTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
