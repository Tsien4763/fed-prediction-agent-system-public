"""Agent 4: Multi-Cluster Game Simulation — LLM-powered strategic equilibrium.

Responsible for:
  - In-cluster political decision simulation (Warsh, FOMC Hawk, FOMC Dove, Treasury, etc.)
  - Between-cluster strategic game (US, CN, RU, UK, FR/Euro)
  - Self-play with best-response iteration, LLM payoff judging, and equilibrium auditing
  - Nuwa-style policy persona loading for Warsh-specific reaction-function priors
  - External shock variable generation (energy risk, geopolitical risk, tariff risk)
  - Teacher-student distillation (DeepSeek-compatible teacher → Qwen3.5-0.8B)
  - VAR/VECM economic context injection into agent belief states

Entry points:
  from agents.multi_cluster_game import run_game, get_latest_fed_prediction, get_latest_shock_variables

Implementation:
  Core logic in fed_game/self_play.py (RollingSelfPlayEngine)
  Agent definitions in fed_game/clusters.py (11 agents across 5 clusters)
  Belief state with VAR/VECM injection in fed_game/schemas.py
  Training pipeline in fed_game/training/ (SFT, DAPT, GRPO)
"""
from fed_game.self_play import RollingSelfPlayEngine
from fed_game.config import RuntimeConfig, default_self_play_trace_path, load_config
from fed_game.clusters import CLUSTER_MEMBERS, ROLE_CARDS
from fed_game.persona import PolicyPersona, load_configured_personas, load_policy_persona
from fed_game.skills import LLMEquilibriumJudge, LLMBestResponseSkill
from typing import Any

from agents.runtime_support import append_audit


class MultiClusterGameAgent:
    """Agent boundary for role self-play, payoff judging, and equilibrium audit."""

    name = "multi_cluster_game"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        runtime_inputs = state.get("runtime", {}).get("extra_inputs", {})
        run_self_play = bool(runtime_inputs.get("run_self_play", False))
        quarter = runtime_inputs.get("quarter") or state.get("game_context", {}).get("quarter")
        game_context: dict[str, Any] = {
            "status": "prepared",
            "facade": __name__,
            "role_count": len(ROLE_CARDS),
            "cluster_count": len(CLUSTER_MEMBERS),
            "quarter": quarter,
            "self_play_engine": "fed_game.self_play.RollingSelfPlayEngine",
            "llm_payoff_judge": "fed_game.skills.LLMBestResponseSkill",
            "equilibrium_judge": "fed_game.skills.LLMEquilibriumJudge",
        }
        if run_self_play:
            config = runtime_inputs.get("runtime_config") or load_config()
            engine = RollingSelfPlayEngine(config)
            if not quarter:
                raise ValueError("run_self_play=True requires a quarter in runtime inputs.")
            trace = engine.run_quarter(str(quarter), max_context_docs=int(runtime_inputs.get("max_context_docs", 4)))
            game_context.update({"status": "ran_self_play", "trace": trace.to_dict()})

        state["game_context"] = game_context
        return append_audit(state, self.name, game_context)

def run_game(config_path: str = None):
    """Run a full multi-cluster game simulation."""
    import json
    from pathlib import Path
    if config_path is None:
        from fed_game.config import DEFAULT_CONFIG_PATH
        config_path = DEFAULT_CONFIG_PATH
    cfg = json.loads(Path(config_path).read_text())
    engine = RollingSelfPlayEngine(RuntimeConfig(raw=cfg))
    return engine.run()

def get_latest_fed_prediction():
    """Get the latest FOMC prediction from the most recent game trace."""
    latest = _latest_trace()
    if latest is None:
        return None
    return latest.get("fed_prediction")


def get_latest_shock_variables():
    """Get structured external shock variables from the most recent game trace."""
    latest = _latest_trace()
    if latest is None:
        return None
    return latest.get("shock_variables", [])


def _latest_trace():
    import json
    trace_path = default_self_play_trace_path(load_config())
    if not trace_path.exists():
        return None
    with open(trace_path, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    if not lines:
        return None
    return lines[-1]

__all__ = [
    "MultiClusterGameAgent",
    "RollingSelfPlayEngine",
    "RuntimeConfig",
    "CLUSTER_MEMBERS",
    "ROLE_CARDS",
    "PolicyPersona",
    "load_configured_personas",
    "load_policy_persona",
    "LLMBestResponseSkill",
    "LLMEquilibriumJudge",
    "run_game",
    "get_latest_fed_prediction",
    "get_latest_shock_variables",
]
