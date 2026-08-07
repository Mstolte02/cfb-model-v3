"""Scrape the 2026 TWO-DEEP college charts and emit the roster the projection uses.

Replaces scrape_ourlads.py as the roster source. Three reasons it is the better one:

  - It charts all 138 FBS programs. Ourlads had 136 and was missing North Dakota
    State and Sacramento State, both in their first FBS season in 2026, so those two
    rosters were simply absent from the model.
  - Its position labels are granular for every CHARTED slot - LT/LG/C/RG/RT rather
    than a bare "OL" - which is what lets broad_group carry the OT/IOL split. The
    generic "OL" label does appear, but only on players with depth_slot = null, who
    are roster members not on the depth chart at all and are dropped here.
  - It publishes `projected_slot` beside the listed `depth_slot`: its own read of who
    will actually play. We take the LISTED order, for the same reason
    build_roster_2026 always has - the projection is ours to make - but the
    disagreement is carried through as a column so it can be looked at.

    The two are NOT on the same scale and must not be compared row to row.
    `depth_slot` is depth within one alignment (both starting guards are depth 1);
    `projected_slot` is a rank across the entire position room (1..10 for a line). The
    comparison that means something is per room, which is what main() reports: whether
    the men listed first are the men projected to play.

WHAT IT DOES NOT HAVE IS INJURY STATUS. The `status` field is provenance
(returning / true_freshman / transfer_in), not availability, and injuries appear only
as prose inside the player bios. So availability_2026.csv remains the mechanism for
benching a hurt player, and depth_correction still applies it. Nothing here overrides
it. cfbdepth.com does publish a structured injury report and would be the source for
that, but its terms of service prohibit automated access and it 403s scripted
requests, so it is not used.

Polite by construction: the team list comes from the site's own sitemap rather than
being crawled, one request every 1.5s, everything cached to disk so a re-run costs
nothing, and a real User-Agent so the traffic is identifiable rather than disguised.
robots.txt is `Allow: /` with no restriction on any path used here. Run with --refresh
to force a re-fetch.

Output matches the columns build_roster_2026.py already expects, so the rest of the
pipeline is unchanged.

Run: ../venv/bin/python scrape_twodeep.py [--refresh]
"""
import json, os, re, sys, time, unicodedata

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = f"{HERE}/twodeep_cache"
SITE = "https://www.thetwodeep.com"
SITEMAP = f"{SITE}/sitemap.xml"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
DELAY = 1.5

# Slugs are derived from the CFBD school name, so the table only has to carry the one
# the rule cannot produce. Asserted in main(): every slug on the sitemap must resolve.
TEAM_ALIAS = {"texas-am": "Texas A&M"}

# The player records sit in the Next.js flight payload rather than in the markup, which
# is inline-styled and hostile to parse. The payload carries a `depth_chart` array of
# {position, players[]} which is extracted whole and json-parsed.
#
# IT IS PARSED AS JSON RATHER THAN REGEXED. A regex over the flat player objects looks
# like it works and quietly loses rows: the optional fields really are optional, so
# anchoring on one - headshot_url was the first attempt - drops every player who
# happens not to have it. That cost Toledo 59 of its 86 players and Old Dominion 43 of
# 57, and it failed silently, as a team with a thin-looking chart rather than an error.
FLIGHT_RE = re.compile(r'self\.__next_f\.push\(\[1,(".*?")\]\)</script>', re.S)
DEPTH_CHART_KEY = '"depth_chart":'

# Specialists: real roster spots, but not part of a lineup and not valued by the WAR
# model. H is the holder, not an H-back (H-backs appear as TE-H).
SKIP_POS = {"PK", "PT", "P", "K", "KO", "LS", "KR", "PR", "H"}

# Every alignment name in the 2026 charts, collapsed onto the ELEVEN groups the rest of
# the pipeline uses. Programs name the same job a dozen ways - the edge rusher alone
# shows up as JACK, RUSH, BAN, BANDIT, BUCK, LEO, STUD, VIPER, WOLF, STING, JOKER, CAT,
# DOG and SPEAR - so the mapping is explicit rather than pattern-matched.
#
# THE LINE IS TWO GROUPS, NOT ONE. Tackles and interior linemen do different jobs, are
# paid differently for them, and the facet weights are fitted per (job x group), so
# folding them together was making the regression answer one question for both. A bare
# "OL" cannot be assigned to either and is not mapped; those rows are all off-chart
# reserves and are dropped before this table is consulted.
POS_GROUP = {
    "QB": "QB",
    "RB": "RB", "RB-A": "RB", "RB-B": "RB", "FB": "RB",
    "SB": "RB", "SB-A": "RB", "SB-Z": "RB",
    "WR": "WR", "WR-X": "WR", "WR-Z": "WR", "WR-SL": "WR", "WR-H": "WR",
    "WR-F": "WR", "WR-Y": "WR", "X": "WR", "Z": "WR", "SLOT": "WR",
    "TE": "TE", "TE-Y": "TE", "TE-H": "TE", "LTE": "TE", "RTE": "TE", "Y": "TE",
    # tackles
    "LT": "OT", "RT": "OT", "OT": "OT", "QT": "OT", "ST": "OT", "T": "OT",
    # interior: guards and centres
    "LG": "IOL", "RG": "IOL", "C": "IOL", "OC": "IOL", "OG": "IOL",
    "QG": "IOL", "SG": "IOL", "G": "IOL",
    "NT": "DT", "DT": "DT", "LDT": "DT", "RDT": "DT", "DL": "DT", "DI": "DT",
    "DE": "EDGE", "LDE": "EDGE", "RDE": "EDGE", "EDGE": "EDGE", "JACK": "EDGE",
    "RUSH": "EDGE", "BAN": "EDGE", "BANDIT": "EDGE", "BUCK": "EDGE", "LEO": "EDGE",
    "STUD": "EDGE", "VIPER": "EDGE", "WOLF": "EDGE", "STING": "EDGE",
    "JOKER": "EDGE", "CAT": "EDGE", "DOG": "EDGE", "SPEAR": "EDGE",
    "MLB": "LB", "WLB": "LB", "SLB": "LB", "OLB": "LB", "LOLB": "LB",
    "ROLB": "LB", "LILB": "LB", "RILB": "LB", "ILB": "LB", "LB": "LB",
    "MAC": "LB", "MIKE": "LB", "WILL": "LB", "SAM": "LB",
    "LCB": "CB", "RCB": "CB", "CB": "CB", "FCB": "CB", "BCB": "CB",
    "NB": "CB", "STAR": "CB", "CASH": "CB", "MONEY": "CB", "HUSKY": "CB",
    "CHEET": "CB", "CHEETAH": "CB", "SPUR": "CB", "NICKEL": "CB",
    "SS": "SAF", "FS": "SAF", "S": "SAF", "BS": "SAF", "ROVER": "SAF",
    "SAF": "SAF", "FLD": "SAF", "DB": "SAF",
}

OFFENSE = {"QB", "RB", "WR", "TE", "OT", "IOL"}

# "RS JR/TR" -> class JR, redshirt True, transfer True. Identical in shape to what
# Ourlads wrote and to what build_roster_2026.parse_eligibility already reads, which is
# why swapping the source needs no change downstream.
CLASS_RE = re.compile(r"^(RS\s+)?(FR|SO|JR|SR|GR)(/TR)?$", re.I)


def slugify(team):
    t = unicodedata.normalize("NFKD", team).encode("ascii", "ignore").decode()
    # apostrophes CLOSE rather than separate: Hawai'i is hawaii, not hawai-i. The
    # accent is already gone by here, so San José has arrived as San Jose.
    t = t.replace("'", "").replace("’", "")
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")


def parse_class(raw):
    """('JR', redshirt, transfer) from the site's class_2026 string."""
    s = re.sub(r"\s+", " ", str(raw or "")).strip().upper()
    m = CLASS_RE.match(s)
    if not m:
        return None, False, bool(re.search(r"/TR\b", s))
    return m.group(2).upper(), bool(m.group(1)), bool(m.group(3))


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


def flight_payload(html):
    """The RSC payload, reassembled from the chunks the page pushes into __next_f.

    The chunks are JSON string literals that concatenate into one document; a record
    can straddle a chunk boundary, so they are joined before anything is matched.
    """
    return "".join(json.loads(c) for c in FLIGHT_RE.findall(html))


def json_value(buf, key):
    """The bracketed value following `key` in `buf`, matched on balanced brackets.

    Quoted sections are skipped so a bracket inside a player's hometown or bio does not
    close the array early.
    """
    i = buf.find(key)
    if i < 0:
        return None
    start = buf.find("[", i)
    if start < 0:
        return None
    depth, k = 0, start
    while k < len(buf):
        c = buf[k]
        if c == '"':
            k += 1
            while k < len(buf) and buf[k] != '"':
                k += 2 if buf[k] == "\\" else 1
        elif c in "[{":
            depth += 1
        elif c in "]}":
            depth -= 1
            if depth == 0:
                return buf[start:k + 1]
        k += 1
    return None


def parse_chart(html):
    """[dict] of charted players from one team's depth-chart page.

    Players with depth_slot = null are on the roster but not on the chart. They are
    dropped: the pipeline wants a depth chart, and they are also the only rows whose
    position label is the unassignable generic "OL".
    """
    raw = json_value(flight_payload(html), DEPTH_CHART_KEY)
    if raw is None:
        return []
    try:
        chart = json.loads(raw)
    except json.JSONDecodeError:
        return []

    out, seen = [], set()
    for grp in chart:
        for r in grp.get("players", []):
            depth = r.get("depth_slot")
            if depth is None:
                continue
            name = re.sub(r"\s+", " ", str(r.get("name") or "")).strip()
            # the group's own label is the fallback for a player listed generically
            pos = str(r.get("pos") or grp.get("position") or "").strip().upper()
            if len(name) < 3 or not pos:
                continue
            if (name, pos, depth) in seen:
                continue
            seen.add((name, pos, depth))
            out.append({"pos": pos, "depth": int(depth), "player": name,
                        "class_raw": r.get("class_2026"),
                        "projected_slot": r.get("projected_slot")})
    return out


def main(refresh=False):
    os.makedirs(CACHE, exist_ok=True)
    sess = requests.Session()
    sess.headers["User-Agent"] = UA

    teams = [t["school"] for t in
             json.load(open(f"{HERE}/cfbd_cache/teams_2026.json"))]
    by_slug = {slugify(t): t for t in teams}
    by_slug.update(TEAM_ALIAS)

    sm = fetch(SITEMAP, f"{CACHE}/_sitemap.xml", sess, refresh)
    if not sm:
        sys.exit("could not fetch the TWO-DEEP sitemap")
    slugs = sorted(set(re.findall(
        r"/college/([^/<]+)/depth-chart<", sm)))
    print(f"depth charts on the sitemap: {len(slugs)}")

    unresolved = [s for s in slugs if s not in by_slug]
    if unresolved:
        sys.exit(f"slugs that match no 2026 FBS team, add to TEAM_ALIAS: {unresolved}")

    out, empty, unknown = [], [], {}
    for slug in slugs:
        team = by_slug[slug]
        html = fetch(f"{SITE}/college/{slug}/depth-chart",
                     f"{CACHE}/{slug}.html", sess, refresh)
        if not html:
            empty.append(team)
            continue
        rows = parse_chart(html)
        if len(rows) < 20:
            empty.append(f"{team} ({len(rows)} rows)")
            continue
        kept = 0
        for r in rows:
            if r["pos"] in SKIP_POS:
                continue
            grp = POS_GROUP.get(r["pos"])
            if grp is None:
                unknown.setdefault(r["pos"], set()).add(team)
                continue
            cls, rs, tr = parse_class(r["class_raw"])
            out.append({"team": team, "roster_position": r["pos"],
                        "broad_group": grp,
                        "unit": "OFF" if grp in OFFENSE else "DEF",
                        "depth": r["depth"], "player": r["player"],
                        "class": cls, "is_transfer": tr, "redshirt": rs,
                        "projected_slot": r["projected_slot"]})
            kept += 1
        print(f"  {team:<24} {kept:>3} players")

    d = pd.DataFrame(out)
    # one row per team-player-position; the page lists a man once per view
    d = d.sort_values("depth").drop_duplicates(["team", "player", "roster_position"])
    d.to_csv(f"{HERE}/twodeep_2026.csv", index=False)

    print(f"\nscraped {d.team.nunique()} of {len(teams)} teams, "
          f"{len(d)} player-slots")
    gaps = sorted(set(teams) - set(d.team))
    if gaps:
        print(f"  FBS teams with NO chart ({len(gaps)}): {', '.join(gaps)}")
    if empty:
        print(f"  fetched but unparsed: {', '.join(empty)}")
    if unknown:
        print(f"  UNMAPPED position labels ({len(unknown)}):")
        for p, ts in sorted(unknown.items()):
            print(f"    {p:<8} {len(ts)} teams, e.g. {sorted(ts)[0]}")
    print(f"\n  depth: {dict(d.depth.value_counts().sort_index())}")
    print(f"  class: {dict(d['class'].value_counts(dropna=False))}")
    print(f"  redshirts: {int(d.redshirt.sum())}   transfers: {int(d.is_transfer.sum())}")
    print(f"  line split: OT {int((d.broad_group == 'OT').sum())}, "
          f"IOL {int((d.broad_group == 'IOL').sum())}")

    two = d[d.depth <= 2]
    print("  depth-1 slots per team:",
          {k: round(v, 1) for k, v in
           two[two.depth == 1].groupby("team").size()
           .describe()[["min", "50%", "max"]].to_dict().items()})

    # Where the site's own projection disagrees with the listed chart, per ROOM: are
    # the men listed first the men it projects to play? We keep the listed order; this
    # is reported so the disagreement stays visible rather than being lost.
    rooms = disagree = 0
    for _, g in d.groupby(["team", "broad_group"]):
        g = g[g.projected_slot.notna()]
        listed = set(g.loc[g.depth == 1, "player"])
        if not listed or len(g) <= len(listed):
            continue
        rooms += 1
        proj = set(g.nsmallest(len(listed), "projected_slot").player)
        disagree += listed != proj
    print(f"  rooms where their projected starters differ from the listed ones: "
          f"{disagree} of {rooms}")
    print(f"-> {HERE}/twodeep_2026.csv")


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
