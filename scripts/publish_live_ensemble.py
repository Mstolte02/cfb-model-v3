"""Publish the fitted v5 ensemble into viz/data/model_v4.json, or refresh it.

    python -m scripts.publish_live_ensemble            # attach the fitted ensemble
    python -m scripts.publish_live_ensemble --refresh  # re-pull finals + form, replay
    python -m scripts.publish_live_ensemble --war-only # rewrite ensemble.war_projected only

Attaching adds an ``ensemble`` block beside the frozen v4 blocks and replays every
published 2026 final through it with the same stdlib code the scheduled capture
runs (scripts/ensemble_replay.py).  The v4 ``logistic``/``margin``/``teams`` blocks
and ``dynamic.preseason_ratings`` are left byte-for-byte alone: they are the basis
Week 1 was graded on and the week-0 numbers the futures board is held to.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import ROOT
from scripts import capture_market_snapshot as CAP
from src import live_ensemble as LE

MODEL = ROOT / "viz" / "data" / "model_v4.json"


def comp_frame(frame: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """The fitted frame plus fifth-percentile rows for FBS newcomers."""
    comp = frame.copy()
    fallback = frame.quantile(.05)
    for team in teams:
        if team not in comp.index:
            comp.loc[team] = fallback
    return comp.loc[teams]


def attach(model_path: Path = MODEL) -> dict:
    manifest = LE.load_manifest()
    frame = pd.read_csv(LE.FRAME_PATH, index_col="team")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    teams = list(model["teams"])
    missing = set(frame.index) - set(teams)
    if missing:
        raise ValueError(f"fitted frame has teams the site does not: {sorted(missing)}")
    model["ensemble"] = LE.ensemble_block(manifest, frame, comp_frame(frame, teams))
    model["schema_version"] = 6
    model["architecture"] = model["ensemble"]["architecture"]
    dynamic = model.get("dynamic") or {}
    dynamic["role"] = ("frozen v4 reference: preseason_ratings are the futures "
                       "board's week-0 basis; live probabilities come from "
                       "ensemble")
    model["dynamic"] = dynamic
    model_path.write_text(json.dumps(model, indent=1, allow_nan=False), encoding="utf-8")
    return model


def attach_war(model_path: Path = MODEL, frame_path: Path = LE.FRAME_PATH) -> dict:
    """Write ``ensemble.war_projected`` into the published block and touch nothing else.

    ``attach`` rebuilds the whole block and drops the live state, so a payload that only
    lacks the display WAR would otherwise need a full re-attach and replay. This keeps
    ``members`` and ``state`` byte-for-byte and places the field where ``ensemble_block``
    puts it. The scheduled capture carries it forward unchanged.
    """
    frame = pd.read_csv(frame_path, index_col="team")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    ensemble = model["ensemble"]
    teams = list(ensemble["members"][0]["initial"])
    war = LE.team_war(comp_frame(frame, teams))
    rebuilt = {}
    for key, value in ensemble.items():
        if key == "war_projected":
            continue
        if key == "members":
            rebuilt["war_projected"] = war
        rebuilt[key] = value
    model["ensemble"] = rebuilt
    model_path.write_text(json.dumps(model, indent=1, allow_nan=False), encoding="utf-8")
    return model


def refresh() -> int:
    """Pull finals and form from CFBD, then replay - the capture without lines."""
    CAP.load_env()
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        raise RuntimeError("CFBD_API_KEY is not configured")
    finals = CAP.publish_finals(CAP.fetch_games(key))
    rows = CAP.publish_form(CAP.fetch_advanced(key))
    replayed = CAP.replay_published_results()
    print(f"{finals} new finals, {rows} form rows, {replayed} games replayed")
    return replayed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true",
                        help="re-pull finals and form and replay the live ensemble")
    parser.add_argument("--war-only", action="store_true",
                        help="rewrite ensemble.war_projected from the fitted frame only")
    args = parser.parse_args()
    if args.war_only:
        attach_war()
        print(f"ensemble.war_projected written -> {MODEL}")
        return
    if not args.refresh:
        attach()
        print(f"ensemble attached -> {MODEL}")
    refresh()


if __name__ == "__main__":
    main()
