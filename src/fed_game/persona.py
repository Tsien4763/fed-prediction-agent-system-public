from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import repo_path
from .schemas import StrategyVector


PERSONA_BLOCK_RE = re.compile(r"```json\s+policy_persona\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass(frozen=True)
class PersonaEvidence:
    source_id: str
    title: str
    url: str
    source_type: str
    reliability: str
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersonaEvidence":
        return cls(
            source_id=str(data.get("source_id", "")),
            title=str(data.get("title", "")),
            url=str(data.get("url", "")),
            source_type=str(data.get("source_type", "unknown")),
            reliability=str(data.get("reliability", "unknown")),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class PersonaMentalModel:
    name: str
    one_liner: str
    evidence_ids: list[str] = field(default_factory=list)
    application: str = ""
    limitation: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersonaMentalModel":
        return cls(
            name=str(data.get("name", "")),
            one_liner=str(data.get("one_liner", "")),
            evidence_ids=[str(item) for item in data.get("evidence_ids", [])],
            application=str(data.get("application", "")),
            limitation=str(data.get("limitation", "")),
        )


@dataclass(frozen=True)
class PersonaHeuristic:
    rule: str
    trigger: str
    action: str
    evidence_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersonaHeuristic":
        return cls(
            rule=str(data.get("rule", "")),
            trigger=str(data.get("trigger", "")),
            action=str(data.get("action", "")),
            evidence_ids=[str(item) for item in data.get("evidence_ids", [])],
        )


@dataclass(frozen=True)
class PolicyPersona:
    persona_id: str
    role_id: str
    name: str
    research_cutoff: str
    skill_path: str
    nuwa_dimensions: dict[str, list[str]]
    mental_models: list[PersonaMentalModel]
    decision_heuristics: list[PersonaHeuristic]
    expression_dna: dict[str, Any]
    value_ordering: list[str]
    anti_patterns: list[str]
    honest_boundaries: list[str]
    priors: dict[str, float]
    evidence_sources: list[PersonaEvidence]
    raw_markdown: str = ""

    @classmethod
    def from_skill_md(cls, path: str | Path) -> "PolicyPersona":
        resolved = repo_path(path)
        text = resolved.read_text(encoding="utf-8")
        match = PERSONA_BLOCK_RE.search(text)
        if not match:
            raise ValueError(f"No ```json policy_persona block found in {resolved}")
        payload = json.loads(match.group(1))
        try:
            skill_path = str(resolved.relative_to(repo_path(".")))
        except ValueError:
            skill_path = str(resolved)
        persona = cls.from_dict(payload, skill_path=skill_path, raw_markdown=text)
        persona.validate()
        return persona

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, skill_path: str = "", raw_markdown: str = "") -> "PolicyPersona":
        return cls(
            persona_id=str(data["persona_id"]),
            role_id=str(data["role_id"]),
            name=str(data["name"]),
            research_cutoff=str(data.get("research_cutoff", "")),
            skill_path=skill_path or str(data.get("skill_path", "")),
            nuwa_dimensions={
                str(key): [str(item) for item in value]
                for key, value in dict(data.get("nuwa_dimensions", {})).items()
            },
            mental_models=[PersonaMentalModel.from_dict(item) for item in data.get("mental_models", [])],
            decision_heuristics=[PersonaHeuristic.from_dict(item) for item in data.get("decision_heuristics", [])],
            expression_dna=dict(data.get("expression_dna", {})),
            value_ordering=[str(item) for item in data.get("value_ordering", [])],
            anti_patterns=[str(item) for item in data.get("anti_patterns", [])],
            honest_boundaries=[str(item) for item in data.get("honest_boundaries", [])],
            priors={str(key): float(value) for key, value in dict(data.get("priors", {})).items()},
            evidence_sources=[PersonaEvidence.from_dict(item) for item in data.get("evidence_sources", [])],
            raw_markdown=raw_markdown,
        )

    def validate(self) -> None:
        if not self.persona_id or not self.role_id or not self.name:
            raise ValueError("Policy persona requires persona_id, role_id, and name.")
        if len(self.nuwa_dimensions) < 6:
            raise ValueError(f"{self.persona_id} must include six Nuwa research dimensions.")
        if len(self.mental_models) < 3:
            raise ValueError(f"{self.persona_id} needs at least three mental models.")
        if len(self.decision_heuristics) < 5:
            raise ValueError(f"{self.persona_id} needs at least five decision heuristics.")
        if not self.evidence_sources:
            raise ValueError(f"{self.persona_id} needs traceable evidence sources.")

    def source_map(self) -> dict[str, PersonaEvidence]:
        return {item.source_id: item for item in self.evidence_sources}

    def to_prompt_payload(self, *, max_markdown_chars: int = 3600) -> dict[str, Any]:
        sources = self.source_map()
        return {
            "persona_id": self.persona_id,
            "role_id": self.role_id,
            "name": self.name,
            "research_cutoff": self.research_cutoff,
            "mental_models": [
                {
                    **asdict(model),
                    "evidence": [
                        {
                            "source_id": source_id,
                            "title": sources[source_id].title,
                            "url": sources[source_id].url,
                        }
                        for source_id in model.evidence_ids
                        if source_id in sources
                    ],
                }
                for model in self.mental_models
            ],
            "decision_heuristics": [asdict(item) for item in self.decision_heuristics],
            "expression_dna": self.expression_dna,
            "value_ordering": self.value_ordering,
            "anti_patterns": self.anti_patterns,
            "honest_boundaries": self.honest_boundaries,
            "priors": self.priors,
            "skill_excerpt": self.raw_markdown[:max_markdown_chars],
        }

    def consistency_score(self, strategy: StrategyVector) -> dict[str, Any]:
        data = strategy.to_dict()
        distances = {
            key: abs(float(data.get(key, 0.0)) - float(value))
            for key, value in self.priors.items()
            if key in data
        }
        avg_distance = sum(distances.values()) / max(1, len(distances))
        # Public persona priors are intentionally soft. A 0.30 average distance
        # is treated as the edge of reasonable consistency.
        score = max(0.0, 1.0 - avg_distance / 0.30)
        return {
            "score": round(score, 6),
            "distance": round(avg_distance, 6),
            "feature_distances": {key: round(value, 6) for key, value in distances.items()},
            "persona_id": self.persona_id,
            "skill_path": self.skill_path,
            "mental_models": [item.name for item in self.mental_models],
            "evidence_source_count": len(self.evidence_sources),
            "honest_boundaries": self.honest_boundaries,
        }


def load_policy_persona(path: str | Path) -> PolicyPersona:
    return PolicyPersona.from_skill_md(path)


def load_configured_personas(config: Any) -> dict[str, PolicyPersona]:
    persona_paths = dict(getattr(config, "raw", {}).get("personas", {}))
    personas: dict[str, PolicyPersona] = {}
    for role_id, path in persona_paths.items():
        try:
            persona = load_policy_persona(path)
        except FileNotFoundError:
            continue
        personas[str(role_id)] = persona
    return personas
