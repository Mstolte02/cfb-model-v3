"""Stage 7: recruiting priors.

Every player on a 2026 roster arrived with a high school evaluation. For players with
little or no PFF snap history, that evaluation plus their class year, team, and depth
slot is the only signal available, so it carries the imputation.
"""
import json, os
import numpy as np
import pandas as pd

from build_roster_2026 import norm_name

HERE = os.path.dirname(os.path.abspath(__file__))
# A player on a 2015 roster was recruited around 2011-2014, so the classes have
# to reach back roughly four years before the earliest WAR season or the older
# training rows carry no recruiting rating at all - it was 4% coverage before
# these were pulled, against 72% for recent seasons.
REC_YEARS = range(2011, 2027)


def load_recruits():
    rows = []
    for y in REC_YEARS:
        d = pd.DataFrame(json.load(open(f"{HERE}/rec_{y}.json")))
        rows.append(d[["year", "name", "position", "committedTo", "stars", "rating", "ranking"]])
    r = pd.concat(rows, ignore_index=True)
    r["key"] = r.name.map(norm_name)
    r = r[r.key != ""]
    # a recruit can appear once per class; keep the highest-rated record per name
    r = r.sort_values("rating", ascending=False).drop_duplicates("key", keep="first")
    return r


def load_portal():
    p = pd.DataFrame(json.load(open(f"{HERE}/portal_2026.json")))
    p["key"] = (p.firstName.fillna("") + " " + p.lastName.fillna("")).map(norm_name)
    p = p[p.key != ""]
    p = p.sort_values("rating", ascending=False).drop_duplicates("key", keep="first")
    return p[["key", "origin", "destination", "stars", "rating", "eligibility"]].rename(
        columns={"stars": "portal_stars", "rating": "portal_rating"})


def main():
    ros = pd.read_csv(f"{HERE}/roster_2026.csv")
    rec = load_recruits()
    por = load_portal()

    ros = ros.merge(rec[["key", "stars", "rating", "ranking", "year"]].rename(
        columns={"year": "recruit_year", "ranking": "recruit_rank"}), on="key", how="left")
    ros = ros.merge(por[["key", "portal_stars", "portal_rating"]], on="key", how="left")

    ros["stars"] = ros.stars.fillna(ros.portal_stars)
    ros["rating"] = ros.rating.fillna(ros.portal_rating)

    print(f"roster slots: {len(ros)}")
    print(f"  with a recruiting rating: {ros.rating.notna().sum()} "
          f"({ros.rating.notna().mean()*100:.1f}%)")
    print(f"  with a star grade:        {ros.stars.notna().sum()} "
          f"({ros.stars.notna().mean()*100:.1f}%)")
    print(f"\nstars: {dict(ros.stars.value_counts().sort_index())}")

    print("\nrecruiting coverage by whether we have PFF snaps:")
    print(ros.groupby("has_history").apply(
        lambda g: pd.Series({"slots": len(g), "rated": g.rating.notna().sum(),
                             "pct": round(g.rating.notna().mean()*100, 1)}),
        include_groups=False).to_string())

    # Team recruiting strength: the average rating a program signs, which stands in
    # for development environment and surrounding talent when a player has no snaps.
    tal = ros.groupby("team").rating.mean().rename("team_rating")
    ros = ros.merge(tal, on="team", how="left")

    # blue-chip share is the conventional summary of a roster's talent level
    ros["blue_chip"] = (ros.stars >= 4).astype(float)
    bc = ros.groupby("team").blue_chip.mean().rename("team_blue_chip")
    ros = ros.merge(bc, on="team", how="left")

    print("\ntop 10 teams by mean recruiting rating on the 2026 two-deep:")
    print(tal.sort_values(ascending=False).head(10).round(4).to_string())

    ros.to_csv(f"{HERE}/roster_2026_features.csv", index=False)
    print("\nroster_2026_features.csv written")


if __name__ == "__main__":
    main()
