"""Map CFBD team names to the cfb_alt_logos directory; download any missing
FBS team logos from the ESPN CDN (URLs provided by CFBD /teams/fbs).

Existing files are kept untouched (they're the preferred alt marks); new files
are written as <school_slug>_alt.png in the same directory. Also emits
viz/data/teams.json with per-team logo path, conference, and colors.

Run: ./venv/bin/python -m scripts.prepare_logos
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests

from config import ROOT

LOGO_DIR = Path("/Users/markstolte/Downloads/cfb_alt_logos")
VIZ_DATA = ROOT / "viz" / "data"


def slug(school: str) -> str:
    return re.sub(r"[^a-z0-9&]+", "_", school.lower()).strip("_")


def main():
    teams = json.load(open(ROOT / "data" / "raw" / "teams_2026.json"))
    existing = {p.name.lower(): p.name for p in LOGO_DIR.glob("*.png")}

    out, downloaded, missing = {}, 0, []
    for t in teams:
        school = t["school"]
        fname = f"{slug(school)}_alt.png"
        if fname.lower() in existing:
            fname = existing[fname.lower()]          # keep original casing
        else:
            urls = t.get("logos") or []
            if not urls:
                missing.append(school)
                continue
            try:
                r = requests.get(urls[0].replace("http://", "https://"), timeout=20)
                r.raise_for_status()
                (LOGO_DIR / fname).write_bytes(r.content)
                downloaded += 1
                time.sleep(0.1)
            except Exception as e:
                print(f"  [warn] {school}: {e}")
                missing.append(school)
                continue
        out[school] = {
            "logo": fname,
            "conference": t.get("conference"),
            "abbreviation": t.get("abbreviation"),
            "color": t.get("color"),
            "altColor": t.get("alternateColor"),
            "mascot": t.get("mascot"),
        }

    VIZ_DATA.mkdir(parents=True, exist_ok=True)
    (VIZ_DATA / "teams.json").write_text(json.dumps(out, indent=1))
    print(f"teams mapped: {len(out)}  downloaded: {downloaded}  "
          f"still missing: {missing or 'none'}")
    print(f"-> {VIZ_DATA / 'teams.json'}")


if __name__ == "__main__":
    main()
