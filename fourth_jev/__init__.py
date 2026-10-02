"""Fourth & Jev probabilistic decision layer.

Import submodules explicitly so stdlib state export does not require an HTTP
client or the historical model's scientific dependencies.
"""

__all__ = ["JevClient", "football_questions", "market_questions", "build_game_state", "build_market_state"]


def __getattr__(name):
    if name == "JevClient":
        from .client import JevClient
        return JevClient
    if name in {"football_questions", "market_questions"}:
        from . import questions
        return getattr(questions, name)
    if name in {"build_game_state", "build_market_state"}:
        from . import state
        return getattr(state, name)
    raise AttributeError(name)
