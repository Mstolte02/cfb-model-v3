"""Are the pipeline artifacts consistent with the inputs they were built from?

Several rounds of this project produced numbers that looked authoritative and were
stale - a report showing a different model's positional values, a Method tab showing
retired facet weights, a browser running cached JavaScript against fresh JSON. Each
was found by accident. This finds them on purpose.

Dependencies are declared as a graph rather than a linear chain, because the pipeline
is not linear: diagnostics.py and rb_analysis.py hang off the WAR build as side
branches and are not upstream of the roster.

MTIMES ARE NOT ENOUGH, and this file learned that the hard way. Every artifact can be
newer than everything it was built from and still disagree with it: four metadata files
described different facet sets while all being freshly written, and viz/app.js carried
a hardcoded x1.64 rescale in its prose long after build_hybrid started solving the
de-attenuation internally. A timestamp says WHEN something was written, not WHAT it
says.

So there are two passes now. The first is the dependency graph below. The second reads
the artifacts and checks they are describing the same model as each other, and that no
number published in prose has drifted from the artifact it was copied out of.

Run: ./rbenv/bin/python staleness_check.py
"""
import json
import os
import re

# artifact -> (stage that writes it, artifacts it is built from)
DEPS = {
    "records.csv":               ("build_wins", []),
    "schedule.csv":              ("build_massey", []),
    "facet_values.parquet":      ("build_massey", ["records.csv"]),
    "cfbd_facet_values.parquet": ("cfbd_facets", ["records.csv"]),
    "hybrid_facet_weights.csv":  ("build_hybrid", ["facet_values.parquet",
                                                   "cfbd_facet_values.parquet",
                                                   "records.csv"]),
    "hybrid_player_war.csv":     ("build_hybrid", ["facet_values.parquet",
                                                   "cfbd_facet_values.parquet"]),
    "hybrid_team_ratings.csv":   ("build_hybrid", ["facet_values.parquet"]),
    "position_table.csv":        ("diagnostics", ["hybrid_player_war.csv"]),
    "rb_qualified.csv":          ("rb_analysis", ["hybrid_player_war.csv"]),
    "twodeep_2026.csv":          ("scrape_twodeep", []),
    "roster_2026.csv":           ("build_roster_2026", ["hybrid_player_war.csv",
                                                        "twodeep_2026.csv"]),
    "projections_2026_v2.csv":   ("project_2026_v2", ["roster_2026.csv",
                                                      "hybrid_player_war.csv",
                                                      "hybrid_team_ratings.csv"]),
    "report_2026.json":          ("final_report", ["projections_2026_v2.csv",
                                                   "position_table.csv",
                                                   "hybrid_facet_weights.csv"]),
    "war_2026.html":             ("make_html", ["report_2026.json"]),
}

SLACK = 2  # seconds, so a same-run write order does not register as stale

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---------------------------------------------------------------- content checks
# A number that appears in prose is a COPY, and copies rot. Each entry says where a
# number is published, how to find it in the text, and what it is supposed to equal.
# `live` is called with a dict of loaded artifacts and returns the truth.
#
# CHECKED NUMBERS ARE TAGGED, not guessed at. The first attempt matched any "<n>
# facets" in the prose and immediately flagged a sentence describing what the OLD flat
# fit did to 98 facets - which is true, historical, and exactly the kind of false
# positive that gets a checker switched off. So a number that is meant to track an
# artifact says so, in a marker the reader does not see:
#
#     ...consolidated to 82<!--live:n_facets--> facets...
#
# A claim with no marker is prose about the past and is left alone; a marker whose
# value has drifted is a real defect, every time.
CLAIMS = [
    ("../README.md", r"(\d+)<!--live:n_facets-->", lambda a: a["n_facets"], 0,
     "facet count in the root README"),
    ("README.md", r"(\d+)<!--live:n_facets-->", lambda a: a["n_facets"], 0,
     "facet count in war_model/README.md"),
    ("../viz/app.js", r"(?:&times;|×)\s*(\d\.\d+)", lambda a: None, None,
     "a hardcoded WAR rescale in the app. build_hybrid solves the de-attenuation "
     "internally now, so ANY such factor in the viz is stale by construction"),
]


def _load_live():
    """The artifacts that content checks are measured against."""
    live = {}
    fw = os.path.join(HERE, "hybrid_facet_weights.csv")
    if os.path.exists(fw):
        with open(fw) as f:
            live["n_facets"] = sum(1 for _ in f) - 1
    for name in ("projection_metrics.json", "two_level_meta.json", "model_meta.json",
                 "consolidated_facets.json", "interval_coverage.json"):
        p = os.path.join(HERE, name)
        if os.path.exists(p):
            try:
                live[name] = json.load(open(p))
            except json.JSONDecodeError:
                live[name] = None
    return live


def check_contents(live):
    """Do the artifacts describe the SAME model, and does the prose still match?"""
    problems = []

    # 1. every metadata file that names a facet set must name the same one
    sets = {}
    fw = os.path.join(HERE, "hybrid_facet_weights.csv")
    if os.path.exists(fw):
        with open(fw) as f:
            sets["hybrid_facet_weights.csv"] = {
                ln.split(",")[0] for ln in f.read().splitlines()[1:] if ln.strip()}
    tl = live.get("two_level_meta.json")
    if isinstance(tl, dict) and isinstance(tl.get("concept_weights"), dict):
        sets["two_level_meta.json"] = set(tl["concept_weights"])
    mm = live.get("model_meta.json")
    if isinstance(mm, dict) and isinstance(mm.get("features"), list):
        sets["model_meta.json"] = set(mm["features"])

    if len(sets) > 1:
        ref_name, ref = next(iter(sets.items()))
        for name, s in list(sets.items())[1:]:
            # concept files legitimately hold concepts rather than facets; only
            # complain when the two overlap enough to be describing the same thing
            if s & ref and s != ref:
                problems.append(
                    f"{name} and {ref_name} disagree: "
                    f"{len(s - ref)} only in the first, {len(ref - s)} only in the second")

    # 2. the projection metrics must describe the build that is actually installed
    pm = live.get("projection_metrics.json")
    if isinstance(pm, dict):
        import artifacts as A
        if pm.get("build") not in (None, A.BUILD):
            problems.append(
                f"projection_metrics.json was written for build "
                f"{pm['build']!r}, installed build is {A.BUILD!r}")
        if pm.get("ex_ante_only") is not True:
            problems.append(
                "projection_metrics.json does not record ex_ante_only=true; it may "
                "predate the removal of the is_starter leak")

    # 3. numbers copied into prose
    for rel, pattern, live_fn, tol, what in CLAIMS:
        path = os.path.normpath(os.path.join(HERE, rel))
        if not os.path.exists(path):
            continue
        text = open(path, errors="ignore").read()
        found = re.findall(pattern, text)
        want = live_fn(live)
        if want is None:
            if found:
                problems.append(f"{os.path.relpath(path, ROOT)}: {what} "
                                f"(found {', '.join(sorted(set(found))[:3])})")
            continue
        vals = {float(v) for v in found}
        off = {v for v in vals if abs(v - want) > (tol or 0)}
        if off and vals:
            problems.append(
                f"{os.path.relpath(path, ROOT)}: {what} says "
                f"{', '.join(str(v) for v in sorted(off))}, artifacts say {want}")
    return problems


def main():
    stale, missing = [], []
    print(f"{'artifact':<28}{'stage':<20}{'status'}")
    print("-" * 66)
    for art, (stage, srcs) in DEPS.items():
        if not os.path.exists(art):
            missing.append(art)
            print(f"{art:<28}{stage:<20}MISSING")
            continue
        m = os.path.getmtime(art)
        older = [s for s in srcs
                 if os.path.exists(s) and os.path.getmtime(s) > m + SLACK]
        if older:
            stale.append((art, stage, older))
            print(f"{art:<28}{stage:<20}STALE - older than {', '.join(older)}")
        else:
            print(f"{art:<28}{stage:<20}ok")

    print()
    if missing:
        print(f"missing: {', '.join(missing)}")
    if stale:
        print("rebuild these stages, in this order:")
        seen = []
        for _, stage, _ in stale:
            if stage not in seen:
                seen.append(stage)
        for s in seen:
            print(f"  ./rbenv/bin/python {s}.py")
    if not stale and not missing:
        print("every artifact is newer than everything it was built from")

    print(f"\n{'content checks':<28}")
    print("-" * 66)
    problems = check_contents(_load_live())
    for p in problems:
        print(f"  MISMATCH  {p}")
    if not problems:
        print("  artifacts agree with each other, and the prose agrees with them")
    return len(stale) + len(missing) + len(problems)


if __name__ == "__main__":
    main()
