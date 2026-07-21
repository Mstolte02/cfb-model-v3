"""Monte Carlo simulation of the 2026 season -> 2026-27 College Football Playoff.

Rules implemented (2026-27 format, confirmed via CFP/NCAA announcements):
  - 12 teams.
  - Auto bids: champions of the ACC, Big 12, Big Ten and SEC regardless of
    ranking, PLUS the highest-ranked Group of 6 team (champion or not).
    (New for 2026: the G6 bid is no longer tied to winning the conference.)
  - Remaining 7 spots at-large by committee ranking.
  - STRAIGHT seeding: seeds 1-12 in committee-ranking order; top 4 get byes.
  - First round hosted by higher seed (5v12, 6v11, 7v10, 8v9); quarters on,
    fixed bracket (no reseeding), QF/SF/title at neutral sites.

Committee ranking proxy: score = 10*win_pct + 1.0*rating_z + 0.75*sos_z,
weights fit on the final 2025 committee ranking (Spearman 0.923).

Run: ./venv/bin/python -m scripts.simulate_playoff [n_sims]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import ROOT, ARTIFACTS
from src.model import CFBModel
from scripts.train import build_projection_frame

P4 = {"ACC", "Big 12", "Big Ten", "SEC"}
G6 = {"American Athletic", "Conference USA", "Mid-American",
      "Mountain West", "Pac-12", "Sun Belt"}
CCG_CONFS = P4 | G6                      # every conference stages a title game

# Committee proxy weights (fit vs final 2025 CFP ranking, see repo notes).
W_WINPCT, W_RATING, W_SOS = 10.0, 1.0, 0.75
FCS_WIN_P = 0.95                         # FBS team over a non-FBS opponent
FCS_OPP_RATING = -2.0                    # z-rating charged to SOS for FCS games


def pairwise_probs(model, comp):
    """Vectorized win-prob + margin matrices for every ordered team pair."""
    O, D, fp, py, tal, ret = (comp[c].values[:, None] for c in
                              ("O", "D", "fp_margin", "pythag", "talent", "returning"))
    X = np.stack([O - D.T, D - O.T, fp - fp.T, py - py.T,
                  tal - tal.T, ret - ret.T])                 # (6, n, n)
    z0 = model.intercept + np.tensordot(model.coef, X, axes=1)
    m0 = model.margin_intercept + np.tensordot(model.margin_coef, X, axes=1)

    def blend(z, m):
        p_lg = 1.0 / (1.0 + np.exp(-z))
        from scipy.stats import norm
        p_mg = norm.cdf(m / model.margin_sigma)
        return model.ens_w * p_lg + (1 - model.ens_w) * p_mg

    p_neutral = blend(z0, m0)
    p_home = blend(z0 + model.hfa_coef, m0 + model.margin_hfa)
    return p_neutral, p_home, m0


def main(n_sims=20000, seed=2026, variant=""):
    from config import ROSTER_VARIANT
    kw = ROSTER_VARIANT if variant == "roster" else {}
    suffix = "_roster" if variant == "roster" else ""
    rng = np.random.default_rng(seed)
    frame = build_projection_frame(**kw)
    model = CFBModel.load()

    teams_meta = json.load(open(ROOT / "data" / "raw" / "teams_2026.json"))
    conf = {t["school"]: t["conference"] for t in teams_meta}
    fbs = [t["school"] for t in teams_meta]

    # Teams missing from the frame (FBS newcomers with no FBS priors) get a
    # 5th-percentile rating row: weak, but present for schedules/standings.
    comp = frame.copy()
    for t in fbs:
        if t not in comp.index:
            comp.loc[t] = frame.quantile(0.05)
            print(f"  [info] frame fallback (5th pct) for newcomer: {t}")
    comp = comp.loc[fbs]
    idx = {t: i for i, t in enumerate(fbs)}
    n = len(fbs)

    rating = ((comp["O"] + comp["D"]) / 2.0).values
    rating = (rating - rating.mean()) / rating.std()

    p_neutral, p_home, _ = pairwise_probs(model, comp)

    # ---- schedule ------------------------------------------------------------
    sched = json.load(open(ROOT / "data" / "raw" / "schedule_2026.json"))
    gh, ga, gp, gconf = [], [], [], []          # FBS-vs-FBS games
    fcs_count = np.zeros(n)                     # buy games vs non-FBS
    sos_sum, sos_n = np.zeros(n), np.zeros(n)
    for g in sched:
        h, a = g["homeTeam"], g["awayTeam"]
        hi, ai = idx.get(h), idx.get(a)
        if hi is not None and ai is not None:
            gh.append(hi); ga.append(ai)
            gp.append(p_neutral[hi, ai] if g.get("neutralSite") else p_home[hi, ai])
            gconf.append(bool(g.get("conferenceGame")) and conf[h] == conf[a])
            sos_sum[hi] += rating[ai]; sos_n[hi] += 1
            sos_sum[ai] += rating[hi]; sos_n[ai] += 1
        elif hi is not None or ai is not None:
            t = hi if hi is not None else ai
            fcs_count[t] += 1
            sos_sum[t] += FCS_OPP_RATING; sos_n[t] += 1
    gh, ga = np.array(gh), np.array(ga)
    gp = np.array(gp)
    gconf = np.array(gconf)
    ng = len(gh)
    print(f"  schedule: {ng} FBS-vs-FBS games, {int(fcs_count.sum())} buy games")

    sos = sos_sum / np.maximum(sos_n, 1)
    sos_z = (sos - sos.mean()) / sos.std()
    static_score = W_RATING * rating + W_SOS * sos_z

    # Incidence matrices for vectorized win totals.
    H = np.zeros((ng, n), np.float32); H[np.arange(ng), gh] = 1
    A = np.zeros((ng, n), np.float32); A[np.arange(ng), ga] = 1
    Hc, Ac = H[gconf], A[gconf]
    conf_games = Hc.sum(0) + Ac.sum(0)          # per-team conference game count

    conf_teams = {c: np.array([idx[t] for t in fbs if conf[t] == c])
                  for c in sorted(CCG_CONFS)}

    # ---- simulate all regular-season games at once ---------------------------
    O = (rng.random((n_sims, ng)) < gp).astype(np.float32)
    wins = O @ H + (1 - O) @ A
    cwins = O[:, gconf] @ Hc + (1 - O[:, gconf]) @ Ac
    fcs_w = rng.binomial(fcs_count.astype(int), FCS_WIN_P, size=(n_sims, n))
    wins += fcs_w
    games_played = (H.sum(0) + A.sum(0) + fcs_count)[None, :].repeat(1, 0)
    base_games = H.sum(0) + A.sum(0) + fcs_count

    cpct = cwins / np.maximum(conf_games, 1)
    ccg_rand = rng.random((n_sims, len(conf_teams)))
    po_rand = rng.random((n_sims, 11))          # 4 R1 + 4 QF + 2 SF + 1 F

    # ---- per-sim: CCGs, selection, bracket ------------------------------------
    stats = {k: np.zeros(n) for k in
             ("playoff", "bye", "conf_champ", "qf", "sf", "final", "champ")}
    seed_counts = np.zeros((n, 12))
    win_tot = np.zeros(n); loss_tot = np.zeros(n)

    conf_list = list(conf_teams.items())
    for s in range(n_sims):
        w = wins[s].copy()
        gp_played = base_games.copy()
        champs = {}
        for ci, (c, members) in enumerate(conf_list):
            sub = cpct[s, members] + 1e-4 * rating[members]
            top2 = members[np.argsort(-sub)[:2]]
            t1, t2 = top2
            p = p_neutral[t1, t2]
            winner, loser = (t1, t2) if ccg_rand[s, ci] < p else (t2, t1)
            champs[c] = winner
            w[winner] += 1
            gp_played[t1] += 1; gp_played[t2] += 1
        score = W_WINPCT * (w / gp_played) + static_score

        for t in champs.values():
            stats["conf_champ"][t] += 1

        # --- selection: 4 P4 champs + best G6 + at-larges, straight seeding ----
        autos = [champs[c] for c in P4]
        g6_pool = np.concatenate([m for c, m in conf_list if c in G6])
        autos.append(int(g6_pool[np.argmax(score[g6_pool])]))
        field = set(autos)
        for t in np.argsort(-score):
            if len(field) >= 12:
                break
            field.add(int(t))
        seeds = sorted(field, key=lambda t: -score[t])   # seeds[0] = 1-seed

        for rk, t in enumerate(seeds):
            stats["playoff"][t] += 1
            seed_counts[t, rk] += 1
            if rk < 4:
                stats["bye"][t] += 1

        # --- bracket (straight seeding, fixed) ---------------------------------
        r = po_rand[s]
        def play(t1, t2, pr, home=False):
            p = p_home[t1, t2] if home else p_neutral[t1, t2]
            return t1 if pr < p else t2
        w89 = play(seeds[7], seeds[8], r[0], home=True)
        w512 = play(seeds[4], seeds[11], r[1], home=True)
        w611 = play(seeds[5], seeds[10], r[2], home=True)
        w710 = play(seeds[6], seeds[9], r[3], home=True)
        q1 = play(seeds[0], w89, r[4])
        q2 = play(seeds[3], w512, r[5])
        q3 = play(seeds[2], w611, r[6])
        q4 = play(seeds[1], w710, r[7])
        for t in (seeds[0], seeds[3], seeds[2], seeds[1], w89, w512, w611, w710):
            stats["qf"][t] += 1
        s1 = play(q1, q2, r[8])
        s2 = play(q4, q3, r[9])
        for t in (q1, q2, q3, q4):
            stats["sf"][t] += 1
        champ = play(s1, s2, r[10])
        for t in (s1, s2):
            stats["final"][t] += 1
        stats["champ"][champ] += 1

        win_tot += w
        loss_tot += gp_played - w

    # ---- output ---------------------------------------------------------------
    power = pd.read_csv(ARTIFACTS / f"2026_power_ratings{suffix}.csv").set_index("team")
    out = []
    for t, i in idx.items():
        if stats["playoff"][i] == 0 and stats["conf_champ"][i] == 0:
            continue
        out.append({
            "team": t,
            "conference": conf[t],
            "power_rank": int(power.loc[t, "rank"]) if t in power.index else None,
            "rating": round(float(rating[i]), 3),
            "avg_wins": round(float(win_tot[i] / n_sims), 2),
            "avg_losses": round(float(loss_tot[i] / n_sims), 2),
            "conf_champ": round(float(stats["conf_champ"][i] / n_sims), 4),
            "playoff": round(float(stats["playoff"][i] / n_sims), 4),
            "bye": round(float(stats["bye"][i] / n_sims), 4),
            "qf": round(float(stats["qf"][i] / n_sims), 4),
            "sf": round(float(stats["sf"][i] / n_sims), 4),
            "final": round(float(stats["final"][i] / n_sims), 4),
            "champ": round(float(stats["champ"][i] / n_sims), 4),
            "seeds": [round(float(c / n_sims), 4) for c in seed_counts[i]],
        })
    out.sort(key=lambda r: (-r["playoff"], -r["champ"]))

    result = {
        "n_sims": n_sims,
        "season": 2026,
        "rules": ("12 teams; auto: ACC/Big12/BigTen/SEC champs + highest-ranked "
                  "G6 team; 7 at-large; straight seeding, top-4 byes; first round "
                  "at higher seed; fixed bracket"),
        "committee_proxy": f"{W_WINPCT}*win_pct + {W_RATING}*rating_z + {W_SOS}*sos_z "
                           "(Spearman 0.923 vs final 2025 CFP ranking)",
        "teams": out,
    }
    result["variant"] = variant or "balanced"
    viz = ROOT / "viz" / "data"
    viz.mkdir(parents=True, exist_ok=True)
    (viz / f"playoff{suffix}.json").write_text(json.dumps(result, indent=1))

    print(f"\n=== 2026-27 CFP odds ({n_sims:,} sims) — top 20 ===")
    print(f"{'team':<20}{'conf':<18}{'W':>5}{'CC%':>7}{'CFP%':>7}"
          f"{'Bye%':>7}{'Champ%':>8}")
    for r in out[:20]:
        print(f"{r['team']:<20}{r['conference']:<18}{r['avg_wins']:>5.1f}"
              f"{100*r['conf_champ']:>6.1f}%{100*r['playoff']:>6.1f}%"
              f"{100*r['bye']:>6.1f}%{100*r['champ']:>7.1f}%")
    print(f"\n-> {viz / f'playoff{suffix}.json'}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    n = next((int(a) for a in args if a.isdigit()), 20000)
    variant = "roster" if "roster" in args else ""
    main(n, variant=variant)
