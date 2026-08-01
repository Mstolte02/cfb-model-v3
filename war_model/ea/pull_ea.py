"""Pull EA College Football launch ratings into a flat CSV.

EA's ratings page is a Next.js app, so the table is not scraped from rendered HTML -
the same payload the page renders is served as JSON at

    /_next/data/<buildId>/games/<franchiseSlug>/ratings.json?franchiseSlug=...&page=N

100 players a page, 9,013 players over 91 pages for CFB 27. The buildId changes every
time EA redeploys the site, so it is read out of the live page rather than hardcoded.

WHAT THIS CAN AND CANNOT GET. EA serves only the CURRENT game. There is no franchise
slug for College Football 25 or 26 - every candidate returns 500 - and the Wayback
Machine's earliest capture of this page is 2025-07-01, which is CFB 26's launch window.
So:

    CFB 27  entering 2026   this script, clean
    CFB 26  entering 2025   ovrbase.com only, server-rendered, one page per player
    CFB 25  entering 2024   not archived anywhere found

`iteration` is carried through and should be "Launch Ratings". EA updates ratings
weekly in season, and an in-season snapshot of game G has seen part of season G-1,
which would put results into what is supposed to be a preseason signal.

Run: ../../venv/bin/python pull_ea.py [--out ea_cfb27.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.request

HERE = __file__.rsplit("/", 1)[0]
RATINGS_PAGE = "https://www.ea.com/games/ea-sports-college-football/ratings"
SLUG = "ea-sports-college-football"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

META = ["id", "player", "team", "conference", "position", "positionLong", "jersey",
        "classYear", "height", "weight", "redshirt", "overall", "iteration"]


def _get(url: str, tries: int = 4) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "x-nextjs-data": "1"})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.read()
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
    raise RuntimeError("unreachable")


def build_id() -> str:
    """The deploy hash in the page's __NEXT_DATA__, which the data URL is keyed on."""
    html = _get(RATINGS_PAGE).decode("utf-8", "replace")
    m = re.search(r'"buildId"\s*:\s*"([^"]+)"', html)
    if not m:
        sys.exit("could not find buildId - EA changed the page shape")
    return m.group(1)


def page(bid: str, n: int) -> dict:
    url = (f"https://www.ea.com/_next/data/{bid}/games/{SLUG}/ratings.json"
           f"?franchiseSlug={SLUG}&page={n}")
    return json.loads(_get(url))["pageProps"]["ratingDetails"]


def flatten(p: dict, attrs: list[str]) -> dict:
    lbl = lambda d, k="label": (d or {}).get(k)
    row = {
        "id": p.get("id"),
        "player": f"{p.get('firstName','')} {p.get('lastName','')}".strip(),
        "team": lbl(p.get("team")), "conference": lbl(p.get("conference")),
        "position": lbl(p.get("position"), "shortLabel"),
        "positionLong": lbl(p.get("position")),
        "jersey": p.get("jerseyNum"), "classYear": p.get("schoolYear"),
        "height": p.get("height"), "weight": p.get("weight"),
        "redshirt": p.get("redShirtStatus"), "overall": p.get("overallRating"),
        "iteration": lbl(p.get("iteration")),
    }
    stats = p.get("stats") or {}
    for a in attrs:
        v = stats.get(a)
        row[a] = v.get("value") if isinstance(v, dict) else v
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{HERE}/ea_cfb27.csv")
    ap.add_argument("--sleep", type=float, default=0.25)
    args = ap.parse_args()

    bid = build_id()
    first = page(bid, 1)
    total = first["totalItems"]
    n_pages = -(-total // 100)
    attrs = sorted((first["items"][0].get("stats") or {}).keys())
    print(f"buildId {bid}   {total} players over {n_pages} pages   {len(attrs)} attributes")

    rows = [flatten(p, attrs) for p in first["items"]]
    for n in range(2, n_pages + 1):
        rows.extend(flatten(p, attrs) for p in page(bid, n)["items"])
        if n % 10 == 0 or n == n_pages:
            print(f"  page {n}/{n_pages}  {len(rows)} players")
        time.sleep(args.sleep)

    seen, uniq = set(), []
    for r in rows:                        # pagination can repeat a row if EA reorders
        if r["id"] in seen:
            continue
        seen.add(r["id"]); uniq.append(r)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=META + attrs)
        w.writeheader(); w.writerows(uniq)

    its = sorted({r["iteration"] for r in uniq})
    print(f"\n-> {args.out}   {len(uniq)} players, {len(META) + len(attrs)} columns")
    print(f"   iterations present: {its}")
    if its != ["Launch Ratings"]:
        print("   [warn] not launch ratings - this snapshot has seen live results")


if __name__ == "__main__":
    main()
