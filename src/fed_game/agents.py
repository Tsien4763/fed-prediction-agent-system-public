from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .clusters import RoleCard
from .schemas import AgentMemory, AgentProposal, StrategyVector
from .skills import LLMBestResponseSkill


@dataclass
class PolicyAgent:
    role: RoleCard
    memory: AgentMemory
    best_response_skill: LLMBestResponseSkill

    def propose(
        self,
        *,
        context: dict[str, Any],
        opponent_strategies: dict[str, StrategyVector],
        evidence: list[dict[str, Any]],
        round_id: int,
    ) -> AgentProposal:
        proposal = self.best_response_skill.run(
            role=self.role,
            memory=self.memory,
            context=context,
            opponent_strategies=opponent_strategies,
            evidence=evidence,
            round_id=round_id,
        )
        previous_distance = self.memory.previous_strategy.distance(proposal.strategy)
        self.memory.stable_rounds = self.memory.stable_rounds + 1 if previous_distance < 0.025 else 0
        self.memory.previous_strategy = proposal.strategy
        self.memory.payoff_history.append(proposal.payoff_estimate)
        if proposal.regret_estimate > 0.25:
            self.memory.failed_strategies.append(
                {
                    "round_id": round_id,
                    "strategy": proposal.strategy.to_dict(),
                    "regret_estimate": proposal.regret_estimate,
                }
            )
        return proposal
