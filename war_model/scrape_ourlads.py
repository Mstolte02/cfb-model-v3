"""Scrape the 2026 Ourlads two-deeps. FALLBACK ROSTER SOURCE.

scrape_twodeep.py is the primary now - it charts all 138 FBS programs where this has
136, missing North Dakota State and Sacramento State in their first FBS season. This
is kept because it is a second independent read of the same charts and its position
labels are equally granular, so build_roster_2026 can fall back to it without losing
the OT/IOL split. It is only consulted when twodeep_2026.csv is absent.


The workbook the pipeline shipped with was exported on 27 June; Ourlads keeps its
charts current through camp, and a spot check found LSU's starting back had already
changed. This pulls every FBS chart directly.

Polite by construction: one request every 1.5s, everything cached to disk so a re-run
costs nothing, and a real User-Agent so the traffic is identifiable rather than
disguised. Run with --refresh to force a re-fetch.

Output matches the columns build_roster_2026.py already expects from the workbook, so
the rest of the pipeline is unchanged.

Run: ./rbenv/bin/python scrape_ourlads.py [--refresh]
"""
import json, os, re, sys, time, unicodedata

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = f"{HERE}/ourlads_cache"
INDEX = "https://www.ourlads.com/ncaa-football-depth-charts/"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
DELAY = 1.5

# Ourlads' school names -> the CFBD names everything else in the pipeline uses.
TEAM_ALIAS = {
    "Appalachian State": "App State", "Central Florida": "UCF",
    "Connecticut": "UConn", "Hawaii": "Hawai'i", "Louisiana-Monroe": "UL Monroe",
    "Miami (Ohio)": "Miami (OH)", "Mississippi": "Ole Miss",
    "North Carolina State": "NC State", "San Jose State": "San José State",
    "Brigham Young": "BYU", "Massachusetts": "Massachusetts",
    "Southern Mississippi": "Southern Miss", "Texas-San Antonio": "UTSA",
    "Alabama-Birmingham": "UAB", "Nevada-Las Vegas": "UNLV",
    "Texas-El Paso": "UTEP", "Florida International": "Florida International",
}

# Ourlads marks class and transfer status inside the name cell: "Brown, Jayce SR/TR"
CLASS_RE = re.compile(r"\b(RS\s+)?(FR|SO|JR|SR|GR)\b(/TR)?\s*$", re.I)
TAG_RE = re.compile(r"<[^>]+>")

# Specialists and placeholders: real roster spots, but not part of a lineup and not
# valued by the WAR model. H is the holder, not an H-back (H-backs appear as TE-H).
SKIP_POS = {"PK", "PT", "P", "KO", "LS", "KR", "PR", "H", "INJ", "SUS", "OUT"}

# Every alignment name in the 2026 charts, collapsed onto the ten groups the rest of
# the pipeline uses. Programs name the same job a dozen ways - the edge rusher alone
# shows up as JACK, RUSH, BAN, BUCK, LEO, STUD, VIPER, WOLF, STING, JOKER, CAT, DOG
# and SPEAR - so the mapping is explicit rather than pattern-matched.
POS_GROUP = {
    "QB": "QB",
    "RB": "RB", "RB-A": "RB", "RB-B": "RB", "FB": "RB",
    "SB": "RB", "SB-A": "RB", "SB-Z": "RB",
    "WR": "WR", "WR-X": "WR", "WR-Z": "WR", "WR-SL": "WR", "WR-H": "WR",
    "WR-F": "WR", "WR-Y": "WR",
    "TE": "TE", "TE-Y": "TE", "TE-H": "TE", "LTE": "TE", "RTE": "TE",
    # the line is two groups; see candidates.GROUPS. QT/QG/SG/ST are the
    # quick-side/strong-side naming some charts use in place of left/right.
    "LT": "OT", "RT": "OT", "OT": "OT", "QT": "OT", "ST": "OT",
    "LG": "IOL", "RG": "IOL", "C": "IOL", "OC": "IOL", "OG": "IOL",
    "QG": "IOL", "SG": "IOL",
    "NT": "DT", "DT": "DT", "LDT": "DT", "RDT": "DT", "DL": "DT",
    "DE": "EDGE", "LDE": "EDGE", "RDE": "EDGE", "EDGE": "EDGE", "JACK": "EDGE",
    "RUSH": "EDGE", "BAN": "EDGE", "BUCK": "EDGE", "LEO": "EDGE", "STUD": "EDGE",
    "VIPER": "EDGE", "WOLF": "EDGE", "STING": "EDGE", "JOKER": "EDGE",
    "CAT": "EDGE", "DOG": "EDGE", "SPEAR": "EDGE",
    "MLB": "LB", "WLB": "LB", "SLB": "LB", "OLB": "LB", "LOLB": "LB",
    "ROLB": "LB", "LILB": "LB", "RILB": "LB", "ILB": "LB", "LB": "LB",
    "MAC": "LB", "MIKE": "LB", "WILL": "LB", "SAM": "LB",
    "LCB": "CB", "RCB": "CB", "CB": "CB", "FCB": "CB", "BCB": "CB",
    "NB": "CB", "STAR": "CB", "CASH": "CB", "MONEY": "CB", "HUSKY": "CB",
    "CHEET": "CB", "SPUR": "CB", "NICKEL": "CB",
    "SS": "SAF", "FS": "SAF", "S": "SAF", "BS": "SAF", "ROVER": "SAF",
    "SAF": "SAF", "FLD": "SAF",
}


def clean(s):
    s = TAG_RE.sub("", s)
    s = (s.replace("&nbsp;", " ").replace("&amp;", "&")
          .replace("&#39;", "'").replace("&quot;", '"'))
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()


def fetch(url, dest, sess, refresh=False):
    if os.path.exists(dest) and not refresh:
        return open(dest, encoding="utf-8", errors="replace").read()
    for attempt in range(4):
        r = sess.get(url, timeout=40)
        if r.status_code == 200:
            open(dest, "w", encoding="utf-8").write(r.text)
            time.sleep(DELAY)
            return r.text
        if r.status_code in (429, 500, 502, 503):
            time.sleep(3 * (attempt + 1))
            continue
        print(f"  [warn] {url} -> {r.status_code}")
        return None
    return None


def team_links(html):
    """{school: url} from the index page.

    The anchors all read "Depth Chart" - the school name lives in a sibling div
    just before them - so the pair has to be matched across the gap, and the
    hrefs are single-quoted relative .aspx links.
    """
    pat = re.compile(
        r"nfl-dc-mm-team-name'>\s*(.*?)\s*</div>.*?"
        r"href='(depth-chart\.aspx\?s=[^']+)'", re.S | re.I)
    out = {}
    for name, href in pat.findall(html):
        school = clean(name)
        if not school or len(school) < 2:
            continue
        out.setdefault(school, "https://www.ourlads.com/ncaa-football-depth-charts/"
                       + href.replace("&amp;", "&"))
    return out


def parse_chart(html):
    """[(position, depth, name, class, is_transfer)] from one team's page."""
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        cells = [clean(c) for c in
                 re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
        if len(cells) < 3 or cells[0].lower() in ("pos", "position", ""):
            continue
        pos = cells[0].upper()
        if not re.fullmatch(r"[A-Z0-9\-/ ]{1,8}", pos):
            continue
        # cells alternate: pos, no, player1, no, player2, no, player3, ...
        for depth, ci in enumerate((2, 4, 6), start=1):
            if ci >= len(cells):
                break
            raw = cells[ci]
            if not raw or raw in ("-", "--"):
                continue
            transfer = bool(re.search(r"/TR\b", raw, re.I))
            m = CLASS_RE.search(raw)
            cls = m.group(2).upper() if m else None
            redshirt = bool(m and m.group(1))
            name = CLASS_RE.sub("", raw).strip().rstrip(",")
            # Ourlads writes "Last, First"; the rest of the pipeline wants "First Last"
            if "," in name:
                last, first = name.split(",", 1)
                name = f"{first.strip()} {last.strip()}"
            name = re.sub(r"\s+", " ", name).strip()
            if len(name) < 3:
                continue
            rows.append((pos, depth, name, cls, transfer, redshirt))
    return rows


def main(refresh=False):
    os.makedirs(CACHE, exist_ok=True)
    sess = requests.Session()
    sess.headers["User-Agent"] = UA

    idx = fetch(INDEX, f"{CACHE}/_index.html", sess, refresh)
    if not idx:
        sys.exit("could not fetch the Ourlads index")
    links = team_links(idx)
    print(f"teams on the index: {len(links)}")

    fbs = set(pd.read_csv(f"{HERE}/records.csv").team)
    out, missing, empty = [], [], []
    unknown = set()
    for school, url in sorted(links.items()):
        team = TEAM_ALIAS.get(school, school)
        if team not in fbs:
            missing.append(school)
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", school.lower()).strip("-")
        html = fetch(url, f"{CACHE}/{slug}.html", sess, refresh)
        if not html:
            empty.append(school)
            continue
        rows = parse_chart(html)
        if len(rows) < 20:
            empty.append(f"{school} ({len(rows)} rows)")
            continue
        kept = 0
        for pos, depth, name, cls, tr, rs in rows:
            if pos in SKIP_POS:
                continue
            grp = POS_GROUP.get(pos)
            if grp is None:
                unknown.add(pos)
                continue
            out.append({"team": team, "roster_position": pos, "broad_group": grp,
                        "unit": "OFF" if grp in ("QB", "RB", "WR", "TE", "OT", "IOL")
                                else "DEF",
                        "depth": depth, "player": name, "class": cls,
                        "is_transfer": tr, "redshirt": rs})
            kept += 1
        print(f"  {team:<24} {kept:>3} players")

    d = pd.DataFrame(out)
    d.to_csv(f"{HERE}/ourlads_2026.csv", index=False)
    print(f"\nscraped {d.team.nunique()} teams, {len(d)} player-slots")
    if missing:
        print(f"  index names not in our FBS set ({len(missing)}): {', '.join(missing[:12])}")
    covered = set(d.team)
    gaps = sorted(fbs - covered)
    if gaps:
        print(f"  FBS teams with NO chart ({len(gaps)}): {', '.join(gaps)}")
    if empty:
        print(f"  fetched but unparsed: {', '.join(empty)}")
    if unknown:
        print(f"  UNMAPPED position labels ({len(unknown)}): {', '.join(sorted(unknown))}")
    print("  depth-1 per team:",
          d[d.depth == 1].groupby("team").size().describe()[["min", "50%", "max"]].to_dict())
    print(f"-> {HERE}/ourlads_2026.csv")


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
