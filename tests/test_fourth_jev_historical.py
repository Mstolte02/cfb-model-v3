import pandas as pd

from fourth_jev.historical import team_frame_block


def test_team_frame_block_serializes_numpy_scalars():
    frame = pd.DataFrame(
        {"O": [1.25], "D": [-0.5], "label": ["x"]},
        index=["A"],
    )
    block = team_frame_block(frame, "A")
    assert block == {"O": 1.25, "D": -0.5, "label": "x"}
