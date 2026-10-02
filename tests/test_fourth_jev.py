from fourth_jev.client import JevClient
from fourth_jev.questions import football_questions
from fourth_jev.state import build_game_state, build_market_state


def test_football_state_is_market_blind():
    state = build_game_state(game={"home_team": "A", "away_team": "B"}, home={"rating": 1.2}, away={"rating": -0.3}, model={"p": 0.64})
    assert state["contract"]["market_blind"] is True
    assert "market" not in state


def test_question_set_has_core_distribution_queries():
    q = football_questions()
    assert q["home_win"]["type"] == "noul"
    assert q["home_by_21_plus"]["type"] == "noul"
    assert q["tail_shape"]["type"] == "choice"
    assert q["variance_level"]["type"] == "score"


def test_payload_matches_typesafe_shape():
    client = JevClient(api_key="test", model="jev-latest")
    state = build_game_state(game={"home_team": "A", "away_team": "B"}, home={}, away={})
    payload = client.payload(state, football_questions())
    assert set(payload) == {"state", "model", "questions"}
    assert payload["model"] == "jev-latest"


def test_market_stage_is_separate():
    football = build_game_state(game={"home_team": "A", "away_team": "B"}, home={}, away={})
    market = build_market_state(football, {"home_win": {"type": "noul", "noul": .6}}, {"spread": -3.5})
    assert market["contract"]["football_forecast_generated_without_market"] is True
    assert market["market"]["spread"] == -3.5
