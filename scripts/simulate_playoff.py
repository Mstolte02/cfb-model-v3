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

Committee ranking proxy: weights fitted by scripts/fit_committee.py on every
published committee ranking from 2014 to 2025 (leave-one-season-out Spearman 0.908
against the real final ranking, against 0.885 for the old hand-set weights).

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

# Committee proxy. Fitted on all 12 seasons of published committee rankings by
# scripts/fit_committee.py; the literals below are the pre-2026 hand-set fallback
# used only if that artifact is missing. The fit is worth having: leave-one-season-out
# Spearman against the real final ranking goes 0.885 -> 0.908, and it says the
# committee leans on schedule strength and power-conference membership far more than
# on raw team quality (rating weight 1.00 -> 0.22, SOS 0.75 -> 0.71, plus a +1.05
# bump simply for playing in a power league).
W_WINPCT, W_RATING, W_SOS, W_P4 = 10.0, 1.0, 0.75, 0.0
_cm = ARTIFACTS / "committee_model.json"
if _cm.exists():
    _w = json.loads(_cm.read_text())["weights"]
    W_WINPCT = _w.get("win_pct", W_WINPCT)
    W_RATING = _w.get("rating_z", W_RATING)
    W_SOS = _w.get("sos_z", W_SOS)
    W_P4 = _w.get("p4", 0.0)

FCS_WIN_P = 0.95                         # FBS team over a non-FBS opponent
FCS_OPP_RATING = -2.0                    # z-rating charged to SOS for FCS games


FEATURE_ORDER = ("O", "D", "fp_margin", "pythag", "talent", "returning")
TALENT_IX = FEATURE_ORDER.index("talent")


def _talent_noise_sd():
    """Extra sd carried by the 2026 talent feature. 0 disables the perturbation."""
    from src.data import war
    return war.talent_noise_sd()


def _blend(model, z, m):
    from scipy.stats import norm
    return (model.ens_w / (1.0 + np.exp(-z))
            + (1 - model.ens_w) * norm.cdf(m / model.margin_sigma))


def pairwise_probs(model, comp):
    """Vectorized win-prob + margin matrices for every ordered team pair.

    Also returns the pre-link scores, because the simulation perturbs talent and needs
    somewhere to add the shift that is still linear.
    """
    O, D, fp, py, tal, ret = (comp[c].values[:, None] for c in FEATURE_ORDER)
    X = np.stack([O - D.T, D - O.T, fp - fp.T, py - py.T,
                  tal - tal.T, ret - ret.T])                 # (6, n, n)
    z0 = model.intercept + np.tensordot(model.coef, X, axes=1)
    m0 = model.margin_intercept + np.tensordot(model.margin_coef, X, axes=1)

    p_neutral = _blend(model, z0, m0)
    p_home = _blend(model, z0 + model.hfa_coef, m0 + model.margin_hfa)
    return p_neutral, p_home, m0, z0


TOSSUP_LO, TOSSUP_HI = 0.45, 0.55
MAX_TOSSUPS = 8                          # 2^8 = 256 cells, ~78 sims each at 20k


def _tossup_table(fbs, idx, sched, gh, ga, gp, O, po_bits, wins, n_sims):
    """Per team: playoff odds conditional on how its coin-flip games actually break.

    The season average answers "how good is this team"; it does not answer "what does
    this team have to do". A team at 40% to make the field with four true toss-ups is
    a different proposition depending on whether it needs three of them or one, and
    that is the question a fan actually has.

    Answered by FILTERING the simulations rather than re-running them, so every cell
    is internally consistent with the bracket, the conference title games and the
    committee proxy - conditioning cannot break a rule the simulation enforces. Each
    team's cell table is indexed by the bit pattern of its toss-up results, capped at
    the 8 closest to a coin flip so the thinnest cell still holds ~78 sims.
    """
    close = np.where((gp >= TOSSUP_LO) & (gp <= TOSSUP_HI))[0]
    meta = {}
    for g in sched:                       # week/date, keyed by the unordered pairing
        meta[frozenset((g["homeTeam"], g["awayTeam"]))] = g
    out = {}
    for t, i in idx.items():
        mine = [j for j in close if gh[j] == i or ga[j] == i]
        mine.sort(key=lambda j: abs(gp[j] - 0.5))
        mine = mine[:MAX_TOSSUPS]
        if not mine:
            continue
        games, key = [], np.zeros(n_sims, np.int32)
        for b, j in enumerate(mine):
            home = gh[j] == i
            opp = fbs[ga[j]] if home else fbs[gh[j]]
            won = O[:, j].astype(bool) if home else ~O[:, j].astype(bool)
            key |= won.astype(np.int32) << b
            g = meta.get(frozenset((fbs[gh[j]], fbs[ga[j]])), {})
            games.append({"opp": opp, "home": bool(home),
                          "p": round(float(gp[j] if home else 1 - gp[j]), 4),
                          "w": g.get("week"), "d": (g.get("startDate") or "")[:10]})
        cells = 1 << len(mine)
        cnt = np.bincount(key, minlength=cells)
        po = np.bincount(key, weights=po_bits[:, i], minlength=cells)
        wn = np.bincount(key, weights=wins[:, i], minlength=cells)
        out[t] = {"games": games,
                  "n": [int(c) for c in cnt],
                  "playoff": [round(float(p / c), 4) if c else None
                              for p, c in zip(po, cnt)],
                  "wins": [round(float(w / c), 2) if c else None
                           for w, c in zip(wn, cnt)]}
    thin = min((min(v["n"]) for v in out.values()), default=0)
    print(f"  toss-up games ({TOSSUP_LO:.0%}-{TOSSUP_HI:.0%}): {len(close)} of "
          f"{len(gp)}; {len(out)} teams have at least one, thinnest cell {thin} sims")
    return out


def main(n_sims=20000, seed=2026, variant=""):
    from config import ROSTER_VARIANT
    kw = ROSTER_VARIANT if variant == "roster" else {}
    suffix = "_roster" if variant == "roster" else ""
    rng = np.random.default_rng(seed)
    frame, b_o, b_d, shrink = build_projection_frame(return_params=True, **kw)
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

    p_neutral, p_home, m_pair, z_pair = pairwise_probs(model, comp)

    # ---- schedule ------------------------------------------------------------
    sched = json.load(open(ROOT / "data" / "raw" / "schedule_2026.json"))
    gh, ga, gp, gconf = [], [], [], []          # FBS-vs-FBS games
    gz, gm = [], []                             # pre-link scores, for the talent shift
    fcs_count = np.zeros(n)                     # buy games vs non-FBS
    sos_sum, sos_n = np.zeros(n), np.zeros(n)
    for g in sched:
        h, a = g["homeTeam"], g["awayTeam"]
        hi, ai = idx.get(h), idx.get(a)
        if hi is not None and ai is not None:
            neutral = bool(g.get("neutralSite"))
            gh.append(hi); ga.append(ai)
            gp.append(p_neutral[hi, ai] if neutral else p_home[hi, ai])
            gz.append(z_pair[hi, ai] + (0.0 if neutral else model.hfa_coef))
            gm.append(m_pair[hi, ai] + (0.0 if neutral else model.margin_hfa))
            gconf.append(bool(g.get("conferenceGame")) and conf[h] == conf[a])
            sos_sum[hi] += rating[ai]; sos_n[hi] += 1
            sos_sum[ai] += rating[hi]; sos_n[ai] += 1
        elif hi is not None or ai is not None:
            t = hi if hi is not None else ai
            fcs_count[t] += 1
            sos_sum[t] += FCS_OPP_RATING; sos_n[t] += 1
    gh, ga = np.array(gh), np.array(ga)
    gp, gz, gm = np.array(gp), np.array(gz), np.array(gm)
    gconf = np.array(gconf)
    ng = len(gh)
    print(f"  schedule: {ng} FBS-vs-FBS games, {int(fcs_count.sum())} buy games")

    sos = sos_sum / np.maximum(sos_n, 1)
    sos_z = (sos - sos.mean()) / sos.std()
    # Notre Dame carries the power flag despite being independent - it has a CFP
    # contract slot and the committee has never treated it as a mid-major. Matches
    # scripts/fit_committee.POWER_INDEPENDENTS, where including it moved
    # leave-one-season-out Spearman from 0.898 to 0.908.
    is_p4 = np.array([1.0 if (conf[t] in P4 or t == "Notre Dame") else 0.0
                      for t in fbs])
    static_score = W_RATING * rating + W_SOS * sos_z + W_P4 * is_p4

    # Incidence matrices for vectorized win totals.
    H = np.zeros((ng, n), np.float32); H[np.arange(ng), gh] = 1
    A = np.zeros((ng, n), np.float32); A[np.arange(ng), ga] = 1
    Hc, Ac = H[gconf], A[gconf]
    conf_games = Hc.sum(0) + Ac.sum(0)          # per-team conference game count

    conf_teams = {c: np.array([idx[t] for t in fbs if conf[t] == c])
                  for c in sorted(CCG_CONFS)}

    # ---- roster uncertainty ---------------------------------------------------
    # Two different things can make a season turn out differently, and only one of
    # them was being simulated. Coin-flip games are one; the other is that the talent
    # estimate itself may be wrong, and for 2026 it is built from PROJECTED player WAR
    # rather than an observed prior season. rb-win-model/uncertainty.talent_noise()
    # measures what that substitution costs by building 2025 talent both ways: the
    # residual is 0.505 sd, which is what gets drawn here.
    #
    # ONE sd for every team, and that is a finding rather than a shortcut. The obvious
    # per-team story - a roster full of unproven players should be less predictable -
    # does not survive: correlation between a team's predicted uncertainty and how far
    # its talent estimate actually missed in 2025 is +0.01. See that function's
    # docstring. Drawing one delta per team per sim (not per game) is what makes this
    # different from noise: a team that draws badly is worse all season, which is how
    # a season actually goes wrong.
    talent_sd = _talent_noise_sd()
    if talent_sd > 0:
        # Talent is a RETIRED column (config.DROPPED_FEATURES), so model.coef[TALENT_IX]
        # is exactly zero and scaling the noise by it would silently turn this whole
        # perturbation into a no-op while still printing that it ran. Talent now reaches
        # the model through the shrink instead: a talent delta d moves O by
        # shrink*b_o*d and D by shrink*b_d*d, and those land on off_edge and def_edge
        # with opposite signs for the two teams -
        #     off_edge = O_home - D_away,  def_edge = D_home - O_away
        # so the home and away multipliers are NOT equal and are carried separately.
        cO, cD = float(model.coef[0]), float(model.coef[1])
        mO, mD = float(model.margin_coef[0]), float(model.margin_coef[1])
        cz_h = shrink * (cO * b_o + cD * b_d) * talent_sd
        cz_a = shrink * (cO * b_d + cD * b_o) * talent_sd
        cm_h = shrink * (mO * b_o + mD * b_d) * talent_sd
        cm_a = shrink * (mO * b_d + mD * b_o) * talent_sd
        from scipy.stats import norm
        O = np.empty((n_sims, ng), np.float32)
        for lo in range(0, n_sims, 4000):        # chunked: (sims x games) is large
            hi_ = min(lo + 4000, n_sims)
            d = rng.standard_normal((hi_ - lo, n))
            dz = cz_h * d[:, gh] - cz_a * d[:, ga]
            dm = cm_h * d[:, gh] - cm_a * d[:, ga]
            p = (model.ens_w / (1.0 + np.exp(-(gz + dz)))
                 + (1 - model.ens_w) * norm.cdf((gm + dm) / model.margin_sigma))
            O[lo:hi_] = rng.random((hi_ - lo, ng)) < p
        print(f"  roster uncertainty: talent perturbed by {talent_sd:.3f} sd per team "
              f"per sim (uniform - per-team spread did not validate)")
    else:
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
             ("playoff", "bye", "conf_champ", "qf", "sf", "final", "champ", "g6_bid")}
    seed_counts = np.zeros((n, 12))
    win_tot = np.zeros(n); loss_tot = np.zeros(n)
    # Full win distribution, not just the mean: an 8.5-win average built from a
    # narrow 7-10 spread is a different team from one built on 5-12, and the team
    # page is where that difference is worth showing.
    MAX_W = 17
    win_hist = np.zeros((n, MAX_W + 1))

    # Which sims did each team make the field in? Needed for the toss-up conditional
    # below, which is a filter over sims rather than a summary of them.
    po_bits = np.zeros((n_sims, n), bool)

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
        g6_bid = int(g6_pool[np.argmax(score[g6_pool])])
        stats["g6_bid"][g6_bid] += 1
        autos.append(g6_bid)
        field = set(autos)
        for t in np.argsort(-score):
            if len(field) >= 12:
                break
            field.add(int(t))
        seeds = sorted(field, key=lambda t: -score[t])   # seeds[0] = 1-seed

        po_bits[s, seeds] = True
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
        win_hist[np.arange(n), np.clip(w.astype(int), 0, MAX_W)] += 1

    tossups = _tossup_table(fbs, idx, sched, gh, ga, gp, O, po_bits, wins, n_sims)

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
            "g6_bid": round(float(stats["g6_bid"][i] / n_sims), 4),
            "seeds": [round(float(c / n_sims), 4) for c in seed_counts[i]],
        })
    out.sort(key=lambda r: (-r["playoff"], -r["champ"]))

    # ---- modal bracket -------------------------------------------------------
    # The field has to obey the format, not just take the twelve highest playoff
    # probabilities. Every simulated season sends exactly one Group of 6 team, but
    # that bid rotates across a dozen candidates, so no single G6 team accumulates
    # a high enough probability to make a naive top-12 - which left the displayed
    # bracket with no G6 representative at all, and five SEC at-larges instead.
    # Build it the way the selection works: each power conference's most likely
    # champion, then the most likely G6 bid, then at-larges.
    by_team = {r["team"]: r for r in out}
    field, chosen = [], set()

    def take(team):
        if team and team not in chosen and team in by_team:
            chosen.add(team)
            field.append(by_team[team])

    for c in sorted(P4):
        pool = [r for r in out if r["conference"] == c]
        if pool:
            take(max(pool, key=lambda r: r["conf_champ"])["team"])
    g6 = [r for r in out if r["conference"] in G6]
    if g6:
        take(max(g6, key=lambda r: r.get("g6_bid", 0.0))["team"])
    for r in sorted(out, key=lambda r: -r["playoff"]):
        if len(field) >= 12:
            break
        take(r["team"])
    field = field[:12]
    taken, placed = set(), {}
    for r in field:
        pref = sorted(range(12), key=lambda k: -r["seeds"][k])
        for k in pref:
            if k not in taken:
                taken.add(k)
                placed[k + 1] = r["team"]
                break
    seeds = [placed.get(k) for k in range(1, 13)]

    def gm(rd, a, b, site=None):
        return {"round": rd, "top": a, "bottom": b, "site": site}

    bracket = {
        "seeds": seeds,
        "games": [
            gm("R1", seeds[7], seeds[8], seeds[7]),     # 8 v 9, higher seed hosts
            gm("R1", seeds[4], seeds[11], seeds[4]),    # 5 v 12
            gm("R1", seeds[5], seeds[10], seeds[5]),    # 6 v 11
            gm("R1", seeds[6], seeds[9], seeds[6]),     # 7 v 10
            gm("QF", seeds[0], None), gm("QF", seeds[3], None),
            gm("QF", seeds[2], None), gm("QF", seeds[1], None),
            gm("SF", None, None), gm("SF", None, None),
            gm("F", None, None),
        ],
        # which earlier game feeds each open slot, so the app can fill the bracket
        # forward once it has decided each game
        "feeds": {"4": [0], "5": [1], "6": [2], "7": [3],
                  "8": [4, 5], "9": [7, 6], "10": [8, 9]},
    }

    result = {
        "n_sims": n_sims,
        "season": 2026,
        "rules": ("12 teams; auto: ACC/Big12/BigTen/SEC champs + highest-ranked "
                  "G6 team; 7 at-large; straight seeding, top-4 byes; first round "
                  "at higher seed; fixed bracket"),
        "committee_proxy": (
            f"{W_WINPCT:.0f}*win_pct + {W_RATING:.2f}*rating_z + {W_SOS:.2f}*sos_z "
            f"+ {W_P4:.2f}*power_conf, fitted on all 12 seasons of published "
            "committee rankings (leave-one-season-out Spearman 0.908)"),
        "committee_weights": {"win_pct": W_WINPCT, "rating": W_RATING,
                              "sos": W_SOS, "power_conf": W_P4},
        "talent_noise_sd": round(talent_sd, 4),
        "teams": out,
        "bracket": bracket,
        "tossups": tossups,
        "win_dist": {t: [int(c) for c in win_hist[i]] for t, i in idx.items()
                     if win_hist[i].sum() > 0},
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
