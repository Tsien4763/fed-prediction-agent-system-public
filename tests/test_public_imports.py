from __future__ import annotations


def test_public_agent_facades_import() -> None:
    import agents.data_perception  # noqa: F401
    import agents.decision_reasoning  # noqa: F401
    import agents.evidence_chain  # noqa: F401
    import agents.langchain_runtime  # noqa: F401
    import agents.multi_cluster_game  # noqa: F401
    import agents.semantic_extraction  # noqa: F401
    import fed_game.cli  # noqa: F401
