"""Stage 10: render the self-contained HTML report."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(f"{HERE}/report_2026.json"))
M = D["meta"]

GROUPS = ["QB", "RB", "WR", "TE", "OT", "IOL", "DT", "EDGE", "LB", "CB", "SAF"]
GROUP_LABEL = {"QB": "Quarterback", "RB": "Running back", "WR": "Receiver",
               "TE": "Tight end", "OT": "Offensive tackle", "IOL": "Interior O-line",
               "DT": "Interior D-line", "EDGE": "Edge", "LB": "Linebacker",
               "CB": "Cornerback", "SAF": "Safety"}


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def bars(rows, key, label_key, fmt="{:.2f}", max_val=None):
    """Single-series magnitude bars: one hue, direct labels, no legend."""
    mx = max_val or max(r[key] for r in rows)
    out = []
    for r in rows:
        pct = max(r[key] / mx * 100, 0.6)
        out.append(
            f'<div class="bar-row"><span class="bar-label">{esc(r[label_key])}</span>'
            f'<span class="bar-track"><span class="bar-fill" style="width:{pct:.1f}%"></span></span>'
            f'<span class="bar-val">{fmt.format(r[key])}</span></div>')
    return "\n".join(out)


def table(rows, cols, headers, numeric=(), flag_key=None):
    th = "".join(f'<th{" class=num" if c in numeric else ""}>{esc(h)}</th>'
                 for c, h in zip(cols, headers))
    trs = []
    for r in rows:
        tds = []
        for c in cols:
            v = r.get(c, "")
            cls = " class=num" if c in numeric else ""
            if isinstance(v, float):
                v = f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"
            if isinstance(v, bool):
                v = "yes" if v else ""
            tds.append(f"<td{cls}>{esc(v)}</td>")
        mark = ' class="imputed"' if flag_key and r.get(flag_key) else ""
        trs.append(f"<tr{mark}>{''.join(tds)}</tr>")
    return (f'<div class="tw"><table><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(trs)}</tbody></table></div>')


# ---------- position value chart ----------
pv = sorted(D["position_value"], key=lambda r: -r["total"])
pv_rows = [{"g": GROUP_LABEL[r["broad_group"]], "total": r["total"]} for r in pv]

# ---------- validation bars ----------
val_rows = [
    {"m": "This model", "r": M["holdout_r"]},
    {"m": "Carry last season forward", "r": M["carry_r"]},
    {"m": "Position-group average", "r": M["posmean_r"]},
]

# ---------- player tabs ----------
tabs, panels = [], []
for i, g in enumerate(GROUPS):
    act = " aria-selected=true" if i == 0 else ""
    tabs.append(f'<button class="tab" role="tab" data-g="{g}"{act}>{g}</button>')
    rows = D["players"][g]
    panels.append(
        f'<div class="panel" data-g="{g}"{"" if i == 0 else " hidden"}>'
        + table(rows,
                ["player", "team", "class", "proj_war", "war_2025", "snaps_2025", "stars", "imputed"],
                ["Player", "Team", "Class", "Proj 2026 WAR", "2025 WAR", "2025 snaps", "Stars", "Imputed"],
                numeric={"proj_war", "war_2025", "snaps_2025", "stars"}, flag_key="imputed")
        + "</div>")

teams = D["teams"]
cov = sorted(D["coverage"], key=lambda r: -r["pct_imputed"])

HTML = f"""<title>2026 College Football WAR Projections</title>
<style>
:root {{
  color-scheme: light;
  --paper:#F5F6F4; --panel:#FFFFFF; --ink:#16191A; --muted:#5A6560;
  --line:#DCE0DC; --accent:#1F6F4A; --accent-soft:#E4EFE9; --flag:#8A6D1F;
  --serif: Georgia,'Iowan Old Style','Times New Roman',ui-serif,serif;
  --sans: system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
  --mono: ui-monospace,'SF Mono',Menlo,Consolas,monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --paper:#101413; --panel:#171C1A; --ink:#E4E8E5; --muted:#9AA5A0;
    --line:#28302C; --accent:#4FB183; --accent-soft:#17291F; --flag:#C9A227;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --paper:#101413; --panel:#171C1A; --ink:#E4E8E5; --muted:#9AA5A0;
  --line:#28302C; --accent:#4FB183; --accent-soft:#17291F; --flag:#C9A227;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink);
  font-family:var(--sans); font-size:16px; line-height:1.6;
  -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:62rem; margin:0 auto; padding:0 1.5rem 6rem; }}
p, li {{ max-width:64ch; color:var(--ink); }}
h1,h2,h3 {{ font-family:var(--serif); font-weight:600; text-wrap:balance; margin:0; }}
h1 {{ font-size:clamp(2.1rem,5vw,3.2rem); line-height:1.08; letter-spacing:-.015em; }}
h2 {{ font-size:1.6rem; letter-spacing:-.01em; }}
h3 {{ font-size:1.08rem; }}
a {{ color:var(--accent); }}

header.mast {{ border-bottom:2px solid var(--ink); padding:4rem 0 1.75rem; margin-bottom:2.5rem;
  display:flex; flex-direction:column; gap:1rem; }}
.eyebrow {{ font-family:var(--mono); font-size:.72rem; letter-spacing:.14em;
  text-transform:uppercase; color:var(--accent); }}
.standfirst {{ font-size:1.15rem; color:var(--muted); max-width:60ch; }}

.metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(8.5rem,1fr));
  gap:1px; background:var(--line); border:1px solid var(--line); margin:2.5rem 0 3rem; }}
.metric {{ background:var(--panel); padding:1rem 1.1rem; }}
.metric .v {{ font-family:var(--mono); font-size:1.5rem; font-weight:600;
  font-variant-numeric:tabular-nums; color:var(--accent); line-height:1.1; }}
.metric .k {{ font-size:.75rem; color:var(--muted); margin-top:.35rem; }}

section {{ margin:3.5rem 0; display:flex; flex-direction:column; gap:1rem; }}
section > h2 {{ padding-bottom:.5rem; border-bottom:1px solid var(--line); }}

ol.stages {{ counter-reset:s; list-style:none; padding:0; margin:0;
  display:flex; flex-direction:column; gap:.9rem; max-width:64ch; }}
ol.stages li {{ counter-increment:s; position:relative; padding-left:2.6rem; }}
ol.stages li::before {{ content:counter(s,decimal-leading-zero);
  position:absolute; left:0; top:.15rem; font-family:var(--mono); font-size:.78rem;
  color:var(--accent); border:1px solid var(--line); padding:.1rem .4rem; }}
ol.stages b {{ font-weight:600; }}

.bar-row {{ display:grid; grid-template-columns:11.5rem 1fr 4rem; align-items:center;
  gap:.75rem; padding:.28rem 0; }}
.bar-label {{ font-size:.87rem; color:var(--ink); }}
.bar-track {{ background:var(--accent-soft); height:16px; border-radius:2px; }}
.bar-fill {{ display:block; height:100%; background:var(--accent);
  border-radius:0 4px 4px 0; }}
.bar-val {{ font-family:var(--mono); font-size:.82rem; text-align:right;
  font-variant-numeric:tabular-nums; color:var(--muted); }}

.tw {{ overflow-x:auto; border:1px solid var(--line); background:var(--panel); }}
table {{ border-collapse:collapse; width:100%; font-size:.85rem; }}
th, td {{ padding:.5rem .7rem; text-align:left; border-bottom:1px solid var(--line);
  white-space:nowrap; }}
th {{ font-family:var(--mono); font-size:.7rem; letter-spacing:.06em;
  text-transform:uppercase; color:var(--muted); font-weight:500;
  position:sticky; top:0; background:var(--panel); }}
td.num, th.num {{ text-align:right; font-family:var(--mono);
  font-variant-numeric:tabular-nums; }}
tbody tr:last-child td {{ border-bottom:none; }}
tbody tr:hover {{ background:var(--accent-soft); }}
tr.imputed td:first-child::after {{ content:"·imputed"; color:var(--flag);
  font-family:var(--mono); font-size:.62rem; margin-left:.45rem;
  letter-spacing:.04em; }}

.tabs {{ display:flex; flex-wrap:wrap; gap:.3rem; }}
.tab {{ font-family:var(--mono); font-size:.75rem; letter-spacing:.05em;
  padding:.35rem .7rem; border:1px solid var(--line); background:var(--panel);
  color:var(--muted); cursor:pointer; }}
.tab[aria-selected=true] {{ background:var(--accent); color:var(--paper);
  border-color:var(--accent); }}
.tab:focus-visible, .toggle:focus-visible {{ outline:2px solid var(--accent);
  outline-offset:2px; }}

.note {{ border-left:2px solid var(--accent); padding:.15rem 0 .15rem 1rem;
  color:var(--muted); font-size:.92rem; max-width:62ch; }}
.cols {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(15rem,1fr)); gap:1.5rem; }}
footer {{ border-top:1px solid var(--line); padding-top:1.5rem; color:var(--muted);
  font-size:.82rem; }}
.toggle {{ position:fixed; top:1rem; right:1rem; font-family:var(--mono);
  font-size:.7rem; padding:.35rem .6rem; border:1px solid var(--line);
  background:var(--panel); color:var(--muted); cursor:pointer; z-index:10; }}
@media (prefers-reduced-motion:reduce) {{ * {{ transition:none !important; }} }}
</style>

<button class="toggle" id="tg">theme</button>
<div class="wrap">

<header class="mast">
  <span class="eyebrow">Wins Above Replacement &middot; FBS &middot; 2026 projection</span>
  <h1>What every player on a 2026 two-deep is worth, in wins</h1>
  <p class="standfirst">A wins-above-replacement model built from five seasons of PFF
  grades and every FBS result since 2021, then projected onto the 2026 depth charts &mdash;
  including the {M['imputed']} players who have never taken a snap.</p>
</header>

<div class="metrics">
  <div class="metric"><div class="v">{M['slots']:,}</div><div class="k">roster slots valued</div></div>
  <div class="metric"><div class="v">{M['teams']}</div><div class="k">FBS teams</div></div>
  <div class="metric"><div class="v">{M['matched_pct']}%</div><div class="k">matched to snap history</div></div>
  <div class="metric"><div class="v">{M['holdout_r']}</div><div class="k">holdout correlation</div></div>
  <div class="metric"><div class="v">{M['qb_mean_war']:.2f}</div><div class="k">mean QB WAR</div></div>
  <div class="metric"><div class="v">{M['rb_mean_war']:.2f}</div><div class="k">mean RB WAR</div></div>
</div>

<section>
  <h2>How a grade becomes a win</h2>
  <p>The model follows the method in PFF's own WAR paper (Eager &amp; Chahrouri), rebuilt
  from scratch on college data. The point of the Massey step is that it never asks
  "who put up numbers" &mdash; it asks how much a player's play moved his team's rating
  once the schedule he played is accounted for.</p>
  <ol class="stages">
    <li><b>Normalize every grade.</b> Each of 20 facets &mdash; passing, DB coverage, run
    blocking, ball security and so on &mdash; is centered so the average snap scores zero,
    then multiplied by snaps in that facet.</li>
    <li><b>Weight the facets by what wins.</b> A random forest scores each facet against
    adjusted team wins, where a margin of nine or more counts as a full win and anything
    closer counts as half. Passing takes {D['facet_weights'][0]['rf']:.0%} of the weight
    and DB coverage {D['facet_weights'][1]['rf']:.0%}.</li>
    <li><b>Solve the Massey system.</b> <span class="mono">M&thinsp;r&thinsp;=&thinsp;f</span>
    across all {M['games']:,} FBS-vs-FBS games, turning raw grade totals into
    schedule-adjusted team ratings (r&nbsp;=&nbsp;{M['massey_r']} with win percentage).</li>
    <li><b>Remove one player at a time.</b> Wins above average is the drop in a team's
    implied wins when a player's grades are replaced by an average player at the same
    snaps &mdash; computed exactly in closed form, not by refitting.</li>
    <li><b>Project onto 2026.</b> A gradient booster trained on every season-to-season
    transition in the data predicts next-year WAR from prior production, class, recruiting
    rating, team strength and depth slot.</li>
  </ol>
  <p class="note">The two-deep workbook is used only to establish who is on which roster
  at which position and depth. Every valuation number here is produced by this model from
  the PFF exports; none of the workbook's own scores were used.</p>
</section>

<section>
  <h2>Does it actually predict anything?</h2>
  <p>Trained on the 2022&ndash;24 transitions and tested on a held-out 2025, against the two
  baselines any projection has to beat. Correlation with actual 2025 WAR,
  {M['holdout_n']:,} two-deep slots &mdash; higher is better.</p>
  {bars(val_rows, "r", "m", "{:.3f}")}
  <p>The harder test is the players the model knows nothing about. For the
  <b>{M['nohist_n']:,} players in that holdout with no prior snaps at all</b> &mdash; where recruiting
  rating, class, team and projected role are the only inputs &mdash; the projection still
  correlates at <b>{M['nohist_r']}</b>, and cuts error to {M['nohist_mae']:.4f} from the
  {M['nohist_base_mae']:.4f} you get by guessing the position average. That is the
  imputation earning its place.</p>
</section>

<section>
  <h2>Where wins actually come from</h2>
  <p>Total projected 2026 WAR by position group, summed across all {M['teams']} teams.
  This is the model's central claim, and it is not a close call.</p>
  {bars(pv_rows, "total", "g", "{:.0f}")}
  <p>Quarterbacks account for more projected wins than the entire offensive line, receiver
  corps, tight ends and running backs combined. Running backs come last at
  {[r['total'] for r in pv if r['broad_group']=='RB'][0]:.0f} wins across 138 teams &mdash;
  a mean of {M['rb_mean_war']:.2f} WAR per back against {M['qb_mean_war']:.2f} for a
  quarterback. Cornerback and safety rank second and third, which is the same answer PFF
  got in the NFL and the same answer the two-deep workbook's own independent position
  weights arrived at.</p>
</section>

<section>
  <h2>Projected 2026 teams</h2>
  <p>Roster WAR summed over the two-deep, then mapped to a win total by a regression fitted
  on 2021&ndash;25 actual results (r&nbsp;=&nbsp;{M['war_calib_r']}). Win totals are for a
  12-game regular season and exclude any postseason.</p>
  {table(teams[:30],
         ["rank","team","conference","proj_war","proj_wins_12","QB","CB","OT","IOL","imputed"],
         ["#","Team","Conference","Roster WAR","Proj wins","QB","CB","OT","IOL","Imputed slots"],
         numeric={"rank","proj_war","proj_wins_12","QB","CB","OT","IOL","imputed"})}
</section>

<section>
  <h2>Every position, ranked</h2>
  <p>The top 20 projected players in each group. Rows tagged
  <span style="color:var(--flag);font-family:var(--mono);font-size:.75rem">&middot;imputed</span>
  have no FBS snap history &mdash; their number comes entirely from recruiting profile,
  class, team and depth slot.</p>
  <div class="tabs" role="tablist">{''.join(tabs)}</div>
  {''.join(panels)}
</section>

<section>
  <h2>The players with no track record</h2>
  <p>{M['imputed']} of {M['slots']:,} slots ({M['imputed_pct']}%) belong to players with no
  FBS snaps in the PFF data &mdash; true freshmen, JUCO arrivals, and the two programs
  joining FBS in 2026. These are the highest projections among them.</p>
  {table(D['top_imputed'],
         ["player","team","broad_group","class","stars","is_starter","proj_war"],
         ["Player","Team","Group","Class","Stars","Starter","Proj 2026 WAR"],
         numeric={"stars","proj_war","is_starter"})}
  <h3 style="margin-top:1.5rem">How much imputation each group needs</h3>
  {table(cov, ["broad_group","slots","imputed","pct_imputed"],
         ["Group","Slots","Imputed","% imputed"],
         numeric={"slots","imputed","pct_imputed"})}
</section>

<section>
  <h2>What this model cannot tell you</h2>
  <div class="cols">
    <div>
      <h3>Season grades, not play grades</h3>
      <p>PFF's paper normalizes grades at the play level. These exports are season
      aggregates, so each facet is a snap-weighted season grade times snaps. That adds
      noise: our ratings correlate with win percentage at {M['massey_r']} where the NFL
      paper reports 0.73.</p>
    </div>
    <div>
      <h3>Replacement level is a choice</h3>
      <p>A replacement roster is set at a .150 win rate, roughly 1.8 wins of 12. It is an
      assumption, exactly as the NFL paper assumes three wins of sixteen. Rankings do not
      move if you change it; the absolute WAR level does.</p>
    </div>
    <div>
      <h3>Depth charts move</h3>
      <p>Projections are conditioned on the two-deep as published. Camp battles, injuries
      and late portal activity will reshuffle roles, and a player's projection is
      largely a function of the role he is assumed to hold.</p>
    </div>
    <div>
      <h3>Projections are compressed</h3>
      <p>Every projection is a conditional mean, so the spread is narrower than a real
      season's. Expect the actual 2026 distribution to have fatter tails in both
      directions than the numbers here.</p>
    </div>
  </div>
</section>

<footer>
  Built from PFF rushing, passing, receiving, blocking and defense exports (2021&ndash;25),
  {M['games']:,} FBS results, and the 2026 two-deep. Method after Eager &amp; Chahrouri,
  <i>PFF WAR: Modeling Player Value in American Football</i>.
  {M['player_seasons']:,} player-seasons graded.
</footer>
</div>

<script>
const tabs=[...document.querySelectorAll('.tab')],panels=[...document.querySelectorAll('.panel')];
tabs.forEach(t=>t.addEventListener('click',()=>{{
  tabs.forEach(x=>x.setAttribute('aria-selected',x===t));
  panels.forEach(p=>p.hidden=p.dataset.g!==t.dataset.g);
}}));
const tg=document.getElementById('tg');
tg.addEventListener('click',()=>{{
  const dark=matchMedia('(prefers-color-scheme: dark)').matches;
  const cur=document.documentElement.getAttribute('data-theme')||(dark?'dark':'light');
  document.documentElement.setAttribute('data-theme',cur==='dark'?'light':'dark');
}});
</script>
"""

open(f"{HERE}/war_2026.html", "w").write(HTML)
print(f"war_2026.html written ({len(HTML)/1024:.0f} KB)")
