import pandas as pd

from fourth_jev.historical import cfb_distribution, team_frame_block


def test_team_frame_block_serializes_numpy_scalars():
    frame = pd.DataFrame(
        {"O": [1.25], "D": [-0.5], "label": ["x"]},
        index=["A"],
    )
    block = team_frame_block(frame, "A")
    assert block == {"O": 1.25, "D": -0.5, "label": "x"}


def test_cfb_distribution_is_coherent_and_centred():
    d = cfb_distribution(.5, 17.0)
    p = d["probabilities"]
    assert abs(d["predicted_margin"]) < 1e-12
    assert p["home_by_21_plus"] <= p["home_by_14_plus"] <= p["home_by_7_plus"] <= p["home_win"]
    assert p["away_by_7_plus"] <= 1 - p["home_win"]
    assert 0 < p["within_3"] < 1
