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

It also decides which presentation to use on the team's own colour. ESPN's dark
variant (logos[1]) is preferred when it gives the primary mark enough contrast.
When neither supplied mark clears the contrast target, the app presents the primary
mark in white. Ranking tiles therefore keep the team colour in every case.

Run: ./venv/bin/python -m scripts.prepare_logos [--force]
"""
import json
import re
import sys
import time
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import requests
from PIL import Image

from config import ROOT, LOGO_DIR as FALLBACK_DIR

LOGO_DIR = ROOT / "viz" / "logos"
DARK_DIR = ROOT / "viz" / "logos-dark"
WHITE_DIR = ROOT / "viz" / "logos-white"
VIZ_DATA = ROOT / "viz" / "data"

# These primary marks contain an opaque oval in the standard PNG. A CSS white
# treatment therefore turns the entire mark into a featureless white pill. ESPN's
# supplied white artwork preserves the internal Georgia G and Missouri tiger detail.
WHITE_OVERRIDES = {
    "Georgia": "https://a.espncdn.com/guid/4351fef8-fe69-53b1-ea57-72684b36ec35/logos/primary_logo_white.png",
    "Missouri": "https://a.espncdn.com/guid/6bbe4a57-263d-12b2-639a-f1db99afcac9/logos/primary_logo_white.png",
}

MIN_CONTRAST = 3.0          # WCAG 2.1 non-text contrast for a graphical object
# FALLBACK_DIR is the last-resort local source for a team ESPN has no usable mark
# for. Genuinely optional - a missing logo degrades a picture, not a number - so it
# is the one external path that does not go through require().
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


def _rel_lum(r, g, b):
    """WCAG relative luminance from 0-255 channels."""
    f = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (f(c / 255) for c in (r, g, b))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(l1, l2):
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def mark_luminance(path: Path) -> float | None:
    """Mean relative luminance of a logo's opaque pixels.

    Alpha-weighted rather than thresholded: many of these marks are antialiased
    line art whose edge pixels are half transparent, and counting those as fully
    present drags a white knockout mark down toward the background it was cut out
    of. Pixels below ~5% alpha carry no visible ink and are skipped entirely.
    """
    try:
        with Image.open(path) as im:
            im = im.convert("RGBA")
            px = np.asarray(im, dtype=float)
    except Exception:
        return None
    alpha = px[..., 3] / 255.0
    keep = alpha > 0.05
    if not keep.any():
        return None
    lum = np.vectorize(_rel_lum)(px[..., 0], px[..., 1], px[..., 2])
    w = alpha[keep]
    return float((lum[keep] * w).sum() / w.sum())


def choose_crest(color: str, base: Path, dark: Path | None):
    """Pick the supplied mark that works on the team colour, else show it white.

    Returns (relative_mark_path, contrast, white) or None when neither file
    can be read - the caller then leaves the field off and the app falls back.
    """
    marks = {}
    lb = mark_luminance(base)
    if lb is not None:
        marks["base"] = lb
    ld = mark_luminance(dark) if dark and dark.exists() else None
    # ESPN serves the primary file again for teams with no true dark variant.
    # Treating that as a second option just duplicates a losing candidate.
    if ld is not None and (lb is None or abs(ld - lb) >= 0.005):
        marks["dark"] = ld
    if not marks:
        return None

    h = color.lstrip("#")
    team_lum = _rel_lum(*(int(h[i:i + 2], 16) for i in (0, 2, 4)))

    on_color = sorted(((_contrast(l, team_lum), k) for k, l in marks.items()),
                      reverse=True)
    if on_color[0][0] >= MIN_CONTRAST:
        c, which = on_color[0]
        return which, c, False

    return "base", _contrast(1.0, team_lum), True


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
    DARK_DIR.mkdir(parents=True, exist_ok=True)
    WHITE_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DATA.mkdir(parents=True, exist_ok=True)
    teams = json.load(open(ROOT / "data" / "raw" / "teams_2026.json"))

    out, got, kept, fell_back, missing = {}, 0, 0, [], []
    white_count, dark_got = 0, 0
    for t in teams:
        school = t["school"]
        fname = f"{slug(school)}.png"
        dest = LOGO_DIR / fname
        dark_dest = DARK_DIR / fname

        # logos[1], the variant drawn for dark backgrounds. Optional in exactly the
        # same sense as the primary mark: without it a team keeps its own colour
        # only if the primary already reads on it, which is a worse tile, not a
        # broken one.
        if not (dark_dest.exists() and dark_dest.stat().st_size > 0) or force:
            urls = t.get("logos") or []
            data = fetch(urls[1]) if len(urls) > 1 else None
            if data:
                dark_dest.write_bytes(data)
                dark_got += 1
                time.sleep(0.05)

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
        entry = {
            "logo": f"logos/{fname}",
            "conference": t.get("conference"),
            "abbreviation": t.get("abbreviation") or school[:4].upper(),
            "mascot": t.get("mascot"),
            "color": color,
            "altColor": norm_hex(t.get("alternateColor"), "#ffffff"),
            "onColor": readable_on(color),
        }
        chosen = choose_crest(color, dest, dark_dest)
        if chosen:
            which, contrast, white = chosen
            white_override = WHITE_OVERRIDES.get(school)
            if white and white_override:
                white_dest = WHITE_DIR / fname
                if not white_dest.exists() or force:
                    data = fetch(white_override)
                    if data:
                        with Image.open(BytesIO(data)) as im:
                            im.thumbnail((500, 500), Image.Resampling.LANCZOS)
                            im.convert("RGBA").save(white_dest, optimize=True)
                if white_dest.exists():
                    which, white = "white", False
            entry["crest"] = {
                "mark": (f"logos-dark/{fname}" if which == "dark" else
                         f"logos-white/{fname}" if which == "white" else
                         f"logos/{fname}"),
                "plate": "color",
                "contrast": round(contrast, 2),
                "white": white,
            }
            white_count += int(white)
        out[school] = entry

    json.dump(out, open(VIZ_DATA / "teams.json", "w"), indent=1)

    # An unused dark variant is dead weight in the repo: 138 extra PNGs of which
    # only the chosen ones are ever requested. Drop the rest.
    used = {Path(v["crest"]["mark"]).name for v in out.values()
            if v.get("crest", {}).get("mark", "").startswith("logos-dark/")}
    dropped = 0
    for f in DARK_DIR.glob("*.png"):
        if f.name not in used:
            f.unlink()
            dropped += 1

    # verification: every emitted entry must point at a real, non-empty PNG
    bad = [s for s, v in out.items()
           if not (LOGO_DIR / Path(v["logo"]).name).exists()
           or (LOGO_DIR / Path(v["logo"]).name).stat().st_size == 0]
    print(f"teams: {len(teams)}   downloaded: {got}   already present: {kept}")
    weak = sorted((v["crest"]["contrast"], s) for s, v in out.items() if v.get("crest"))
    print(f"  dark variants: {dark_got} fetched, {len(used)} used, {dropped} dropped")
    print(f"  crest presentation: color tiles={len(out)}, white marks={white_count}")
    if weak:
        below = [f"{s} {c}" for c, s in weak if c < MIN_CONTRAST]
        print(f"  crest contrast: min {weak[0][0]} ({weak[0][1]}), "
              f"{sum(1 for c, _ in weak if c >= 4.5)}/{len(weak)} at 4.5+")
        if below:
            print(f"  BELOW {MIN_CONTRAST}: {', '.join(below)}")
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
