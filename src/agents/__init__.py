"""Public agent facades and optional LangChain runtime.

Import facade packages directly, for example:

    import agents.data_perception

The LangChain orchestration layer lives in ``agents.langchain_runtime`` and
requires installing the ``agent`` extra.
"""

__all__ = [
    "data_perception",
    "semantic_extraction",
    "decision_reasoning",
    "multi_cluster_game",
    "evidence_chain",
    "langchain_runtime",
]

