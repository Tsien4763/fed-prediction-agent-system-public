# Codex Project Instructions

## No Fallback Contract

- Semantic extraction must run as TF-IDF policy-context selection followed by DeepSeek scoring.
- Multi-agent self-play must use DeepSeek for role strategy generation, payoff judgement, and equilibrium auditing.
- Missing API keys, malformed LLM JSON, failed LLM calls, or disabled LLM judges must raise explicit errors.
- Do not add silent rule, mock, heuristic, keyword, or cached-output fallback paths for semantic scoring or strategic self-play.
- Offline tests should use explicit fake DeepSeek clients, not production fallbacks.
