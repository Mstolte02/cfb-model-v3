"""Are the pipeline artifacts consistent with the inputs they were built from?

Several rounds of this project produced numbers that looked authoritative and were
stale - a report showing a different model's positional values, a Method tab showing
retired facet weights, a browser running cached JavaScript against fresh JSON. Each
was found by accident. This finds them on purpose.

Dependencies are declared as a graph rather than a linear chain, because the pipeline
is not linear: diagnostics.py and rb_analysis.py hang off the WAR build as side
branches and are not upstream of the roster.

Run: ./rbenv/bin/python staleness_check.py
"""
import os

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
    "ourlads_2026.csv":          ("scrape_ourlads", []),
    "roster_2026.csv":           ("build_roster_2026", ["hybrid_player_war.csv",
                                                        "ourlads_2026.csv"]),
    "projections_2026_v2.csv":   ("project_2026_v2", ["roster_2026.csv",
                                                      "hybrid_player_war.csv",
                                                      "hybrid_team_ratings.csv"]),
    "report_2026.json":          ("final_report", ["projections_2026_v2.csv",
                                                   "position_table.csv",
                                                   "hybrid_facet_weights.csv"]),
    "war_2026.html":             ("make_html", ["report_2026.json"]),
}

SLACK = 2  # seconds, so a same-run write order does not register as stale


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


if __name__ == "__main__":
    main()
