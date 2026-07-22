"""Local roster-editing server: serves the viz app AND a roster API.

Static files come from viz/ (identical to `python -m http.server`), plus:

  GET  /api/rosters            -> {team: [{name, group, depth, grade}]}
                                  the 2026 two-deep (Ourlads) with each player's
                                  2025 PFF grade — the roster the model reads.
  POST /api/recompute          -> body {lens, edits: {team: [players...]}}
                                  re-derives talent[2026] from the edited depth
                                  charts and rebuilds the frame with the FROZEN
                                  model math; returns {teams: {team: [6-vector]}}.

Only talent[2026] depends on the roster, so the opponent-adjusted O/D, the
talent->O/D slopes, Pythagorean and returning inputs are computed ONCE at startup
and reused — each recompute is milliseconds.

Run:  ./venv/bin/python -m scripts.serve [port]     (default 8642)
"""
import json
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import (GAME_YEARS, TEST_GAME_YEAR, UNCERTAINTY_LAMBDA, OPP_ADJ_ALPHA,
                    TALENT_BLEND, ROSTER_VARIANT, PROJECTION_YEAR, ROOT)
from src.data import load, pff
from src import matchup as MU
from src import oppadj as OA
from src.projection import _z
from scripts.train import load_bundle, raw_returning, blended_talent

VIZ = ROOT / "viz"
YEAR = PROJECTION_YEAR
STATE = {}
LOCK = threading.Lock()

# Vocabulary the two-deep 'broad_group' uses (matches pff.PFF_OPT_WEIGHTS keys).
GROUPS = list(pff.PFF_OPT_WEIGHTS.keys())


def _depth_weight(depth):
    return 1.0 if (pd.to_numeric(depth, errors="coerce") or 2) <= 1 else 0.45


def team_raw_score(roster, weights=pff.PFF_OPT_WEIGHTS):
    """Depth- and position-weighted 2025-grade score for one team's roster.
    roster: [{group, depth, grade}]. Mirrors pff.build_2026_roster_talent."""
    num = den = 0.0
    by_grp = {}
    for p in roster:
        g = p.get("group")
        if g in weights and p.get("grade") is not None:
            by_grp.setdefault(g, []).append(p)
    for grp, players in by_grp.items():
        dwsum = sum(_depth_weight(p.get("depth", 2)) for p in players)
        if dwsum == 0:
            continue
        gs = sum(_depth_weight(p.get("depth", 2)) * float(p["grade"]) for p in players) / dwsum
        pw = weights.get(grp, 0)
        num += pw * gs
        den += pw
    return (num / den) if den > 0 else None


def load_rosters():
    """{team: [{name, group, depth, grade, source}]} from the 2026 two-deep. QBs
    carry opponent-adjusted CFBD WAR (rescaled to the PFF grade scale); every other
    position keeps its 2025 PFF grade. `source` is "EPA-WAR" or "PFF"."""
    from src import qbwar
    from config import ARTIFACTS
    g = pff.load_player_grades()
    grades25 = g[g["season"] == 2025].groupby("pname", as_index=False)["grade"].max()
    gmap = dict(zip(grades25["pname"], grades25["grade"]))
    war = {}
    qbv = ARTIFACTS / "qb_values.csv"
    if qbv.exists():
        try:
            war = qbwar.war_qb_grades(qbv, 2025)
        except Exception as e:
            print(f"  [warn] QB WAR grades unavailable: {e}")
    td = pd.read_excel(pff.TWODEEP_2026, sheet_name="Weighted Two Deep")
    td = td[["team", "broad_group", "player_display", "depth"]].dropna(
        subset=["player_display", "broad_group"])
    rosters = {}
    for team, tg in td.groupby("team"):
        players = []
        for _, r in tg.iterrows():
            pname = pff._norm(r["player_display"])
            grp = str(r["broad_group"])
            source = "PFF"
            grade = gmap.get(pname)
            if grp == "QB" and pname in war:            # WAR replaces PFF at QB
                grade = war[pname]; source = "EPA-WAR"
            players.append({
                "name": str(r["player_display"]),
                "group": grp,
                "depth": int(pd.to_numeric(r["depth"], errors="coerce") or 2),
                "grade": None if grade is None or pd.isna(grade) else round(float(grade), 1),
                "source": source,
            })
        rosters[str(team)] = players
    return rosters


def lens_params(lens):
    if lens == "roster":
        return ROSTER_VARIANT["talent_blend"], ROSTER_VARIANT["unc_lambda"]
    return TALENT_BLEND, UNCERTAINTY_LAMBDA


def load_state():
    """Load everything once; cache the lens-independent pieces + per-lens slopes."""
    print("Loading pipeline (this takes ~20s) ...", flush=True)
    std, cfbd_tal, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()

    # 2026 returning + CFBD talent proxy, exactly as build_projection_frame does.
    rp_csv = ROOT / "data" / f"returning_{YEAR}.csv"
    if YEAR not in ret and rp_csv.exists():
        df = pd.read_csv(rp_csv).set_index("team")["ret_prod"]
        ret[YEAR] = _z(df); ret_raw[YEAR] = df
    tal_csv = ROOT / "data" / f"talent_{YEAR}.csv"
    if YEAR not in cfbd_tal and tal_csv.exists():
        cfbd_tal[YEAR] = _z(pd.read_csv(tal_csv).set_index("team")["talent"])
    elif YEAR not in cfbd_tal:
        cfbd_tal[YEAR] = cfbd_tal[2025]

    rosters = load_rosters()
    baseline_raw = {t: team_raw_score(r) for t, r in rosters.items()}
    baseline_raw = {t: s for t, s in baseline_raw.items() if s is not None}

    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    train_years = [g for g in GAME_YEARS if g != TEST_GAME_YEAR]

    # Per-lens: historical blended talent -> talent/O-D slopes (roster edits to
    # 2026 don't touch these, so cache once per lens).
    pff_hist = pff.build_roster_talent()
    slopes = {}
    for lens in ("", "roster"):
        blend, _ = lens_params(lens)
        tal_blend_hist = blended_talent(cfbd_tal, pff_hist, w=blend)
        b_o, b_d = MU.fit_talent_od_slopes(train_years, std, tal_blend_hist, od_by_year=od)
        slopes[lens] = (b_o, b_d)

    STATE.update(std=std, cfbd_tal=cfbd_tal, ret=ret, ret_raw=ret_raw, pyth=pyth,
                 od=od, rosters=rosters, baseline_raw=baseline_raw, slopes=slopes)
    print(f"Ready. Rosters for {len(rosters)} teams; "
          f"baseline talent teams: {len(baseline_raw)}.", flush=True)


def recompute(lens, edits):
    """Rebuild the 2026 frame with edited depth charts; return {team: [6-vector]}."""
    blend, lam = lens_params(lens)
    base2026 = STATE["cfbd_tal"][YEAR]

    raw = dict(STATE["baseline_raw"])
    for team, roster in (edits or {}).items():
        sc = team_raw_score(roster)
        if sc is None:
            raw.pop(team, None)
        else:
            raw[team] = sc
    s = pd.Series(raw)
    pff_z = (s - s.mean()) / (s.std(ddof=0) or 1.0)

    r = pff_z.reindex(base2026.index).fillna(base2026)
    talent = blend * r + (1 - blend) * base2026

    # Service-academy / no-composite fallback (mirrors build_projection_frame).
    missing = STATE["ret"][YEAR].index.difference(talent.index)
    if len(missing):
        floor = float(talent.quantile(0.10))
        vals = {t: (blend * float(pff_z[t]) + (1 - blend) * floor
                    if t in pff_z.index else floor) for t in missing}
        talent = pd.concat([talent, pd.Series(vals)])

    b_o, b_d = STATE["slopes"][lens]
    unc = (lam, b_o, b_d, STATE["ret_raw"][YEAR])
    frame = MU.team_frame(YEAR, STATE["std"], STATE["pyth"], {YEAR: talent},
                          STATE["ret"], uncertainty=unc, od_by_year=STATE["od"])
    cols = ["O", "D", "fp_margin", "pythag", "talent", "returning"]
    return {t: [round(float(frame.loc[t, c]), 4) for c in cols] for t in frame.index}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(VIZ), **k)

    def log_message(self, *a):
        pass  # quiet

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] == "/api/rosters":
            return self._json(STATE["rosters"])
        return super().do_GET()

    def do_POST(self):
        if self.path.split("?")[0] != "/api/recompute":
            return self.send_error(404)
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            with LOCK:
                teams = recompute(payload.get("lens", ""), payload.get("edits", {}))
            self._json({"teams": teams})
        except Exception as e:
            import traceback; traceback.print_exc()
            self._json({"error": str(e)}, code=500)


def main(port=8642):
    load_state()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"\nRoster server on http://localhost:{port}  (Ctrl+C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8642)
