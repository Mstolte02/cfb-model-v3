"""Download the basic ESPN team logos and emit the app's team metadata.

Earlier versions pointed at ~/Downloads/cfb_alt_logos, which holds alternate and
retired marks - visually inconsistent, and in several cases not the logo the school
currently uses. This takes ESPN's primary 500px mark for every FBS team instead
(CFBD hands us the URL in /teams/fbs as logos[0]; logos[1] is the dark variant),
writes them into viz/logos so the app is self-contained, and verifies that every
team resolves to a file that is actually a PNG.

Also emits viz/data/teams.json with each team's conference, abbreviation and colors,
plus a readable foreground for that color so the UI can tint surfaces without
guessing at contrast.

Run: ./venv/bin/python -m scripts.prepare_logos [--force]
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests

from config import ROOT

LOGO_DIR = ROOT / "viz" / "logos"
VIZ_DATA = ROOT / "viz" / "data"
# last-resort local source for a team ESPN has no usable mark for
FALLBACK_DIR = Path("/Users/markstolte/Downloads/cfb_logos")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def slug(school: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", school.lower()).strip("-")


def readable_on(hex_color: str) -> str:
    """Black or white, whichever stays legible on the given background."""
    h = (hex_color or "#666666").lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    except ValueError:
        return "#ffffff"
    f = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    lum = 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)   # WCAG relative luminance
    return "#000000" if lum > 0.45 else "#ffffff"


def norm_hex(c, default="#666666"):
    if not c:
        return default
    c = str(c).strip()
    if not c.startswith("#"):
        c = "#" + c
    return c if re.fullmatch(r"#[0-9a-fA-F]{6}", c) else default


def fetch(url: str) -> bytes | None:
    try:
        r = requests.get(url.replace("http://", "https://"), timeout=25)
        r.raise_for_status()
        return r.content if r.content.startswith(PNG_MAGIC) else None
    except Exception:
        return None


def main(force=False):
    LOGO_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DATA.mkdir(parents=True, exist_ok=True)
    teams = json.load(open(ROOT / "data" / "raw" / "teams_2026.json"))

    out, got, kept, fell_back, missing = {}, 0, 0, [], []
    for t in teams:
        school = t["school"]
        fname = f"{slug(school)}.png"
        dest = LOGO_DIR / fname

        if dest.exists() and dest.stat().st_size > 0 and not force:
            kept += 1
        else:
            data = None
            for url in (t.get("logos") or [])[:1]:      # logos[0] = the basic mark
                data = fetch(url)
                if data:
                    break
            if data is None:
                local = FALLBACK_DIR / f"{school.replace(' ', '_')}.png"
                if local.exists():
                    data = local.read_bytes()
                    fell_back.append(school)
            if data is None:
                missing.append(school)
                continue
            dest.write_bytes(data)
            got += 1
            time.sleep(0.05)

        color = norm_hex(t.get("color"))
        out[school] = {
            "logo": f"logos/{fname}",
            "conference": t.get("conference"),
            "abbreviation": t.get("abbreviation") or school[:4].upper(),
            "mascot": t.get("mascot"),
            "color": color,
            "altColor": norm_hex(t.get("alternateColor"), "#ffffff"),
            "onColor": readable_on(color),
        }

    json.dump(out, open(VIZ_DATA / "teams.json", "w"), indent=1)

    # verification: every emitted entry must point at a real, non-empty PNG
    bad = [s for s, v in out.items()
           if not (LOGO_DIR / Path(v["logo"]).name).exists()
           or (LOGO_DIR / Path(v["logo"]).name).stat().st_size == 0]
    print(f"teams: {len(teams)}   downloaded: {got}   already present: {kept}")
    if fell_back:
        print(f"  local fallback used for {len(fell_back)}: {', '.join(sorted(fell_back))}")
    if missing:
        print(f"  NO LOGO for {len(missing)}: {', '.join(sorted(missing))}")
    if bad:
        print(f"  BROKEN entries: {', '.join(sorted(bad))}")
    if not missing and not bad:
        print(f"  all {len(out)} teams resolve to a valid PNG")
    print(f"teams.json -> {VIZ_DATA / 'teams.json'}")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
