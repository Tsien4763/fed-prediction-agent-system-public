"""Policy source monitoring and LLM semantic hawkish-dovish scoring.

Agent 1 (Data Perception) + Agent 2 (Semantic Extraction):
  1. registry-based policy source monitoring
  2. LLM hawkish-dovish scoring of FOMC statements using a teacher API

Usage:
    python -m models.semantic_pipeline
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_engineering.config import REPO_ROOT
from data_engineering.source_monitor import PolicySourceSpec, fetch_source, incremental_fetch


def generate_crawler_with_llm(
    target_url: str,
    description: str,
    api_key: str | None = None,
) -> str:
    """Return a registry-backed source spec instead of generated crawler code.

    The name is kept for backward compatibility with earlier notebooks. New
    code should call data_engineering.source_monitor directly.
    """
    source_id = (
        target_url.replace("https://", "")
        .replace("http://", "")
        .replace("/", "_")
        .replace(".", "_")
        .strip("_")
    )[:80]
    spec = PolicySourceSpec(
        source_id=source_id,
        url=target_url,
        parser="generic_html",
        cadence="daily",
        metadata={"description": description, "generated_by": "source_registry"},
    )
    return json.dumps({"mode": "source_registry", "source": spec.to_dict()}, indent=2)


def score_hawkish_dovish(
    fomc_text: str,
    meeting_date: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    require_llm: bool | None = None,
) -> dict[str, Any]:
    """DeepSeek semantic scoring after TF-IDF policy-context selection."""
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    _ = require_llm
    if not api_key:
        raise RuntimeError("DeepSeek semantic extraction is required, but no API key was provided.")
    try:
        return _llm_score(fomc_text, api_key, base_url, meeting_date=meeting_date)
    except Exception as exc:
        raise RuntimeError("DeepSeek semantic extraction failed; fallback is disabled.") from exc


def _llm_score(text: str, api_key: str, base_url: str, meeting_date: str | None = None) -> dict[str, Any]:
    """Call DeepSeek API after selecting policy-relevant excerpts with TF-IDF."""
    selected = select_tfidf_policy_context(text, max_chars=6000, top_k=12)
    prompt = f"""You are a Federal Reserve policy analyst. Score the selected FOMC/policy excerpts on these dimensions.

Return ONLY a JSON object with these exact float fields (no explanation):

{{
  "hawkish_dovish_score": float between -1.0 (max dovish) and +1.0 (max hawkish),
  "inflation_concern": float 0-1,
  "labor_market_assessment": float -1 (weak) to +1 (strong),
  "growth_outlook": float -1 (weak) to +1 (strong),
  "forward_guidance_strength": float 0-1,
  "uncertainty_index": float 0-1,
  "rate_hike_signal": float 0-1,
  "rate_cut_signal": float 0-1,
  "policy_flexibility": float 0-1,
  "inflation_commitment_credibility": float 0-1
}}

Calibration anchors:
- "considerable time" before liftoff (2014): hawkish_dovish_score ≈ -0.8
- "act as appropriate to sustain expansion" (2019): ≈ -0.5
- "ongoing increases will be appropriate" (2022): ≈ +0.9
- "Committee is strongly committed to returning inflation to 2 percent" (2023): ≈ +0.5

Meeting date, if known: {meeting_date or "unknown"}

The excerpts below were selected by a deterministic TF-IDF policy-context filter.
Use only these excerpts and do not infer facts not present in them.

Selected policy excerpts:
{selected["selected_text"]}
"""
    
    payload = {
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        "messages": [
            {"role": "system", "content": "You are an FOMC policy analyst. Output only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 500,
    }

    result = _post_chat_completion(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        payload=payload,
        api_key=api_key,
        timeout_seconds=60,
        max_attempts=3,
    )
    
    content = result["choices"][0]["message"]["content"]
    # Extract JSON from response (may be wrapped in markdown)
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    
    scores = json.loads(content.strip())
    scores["_method"] = "LLM (DeepSeek)"
    scores["_semantic_filter"] = selected["diagnostics"]
    return scores


def _post_chat_completion(
    url: str,
    *,
    payload: dict[str, Any],
    api_key: str,
    timeout_seconds: int,
    max_attempts: int,
) -> dict[str, Any]:
    """POST to a chat-completions endpoint with timeout and bounded retry."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout_seconds)
            response.raise_for_status()
            return response.json()
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError, ValueError) as exc:
            last_error = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            retryable_status = status is None or status in {408, 429, 500, 502, 503, 504}
            if attempt >= max_attempts or not retryable_status:
                break
            time.sleep(min(2 ** (attempt - 1), 4))
    raise RuntimeError(f"LLM chat-completion request failed after {max_attempts} attempts") from last_error


FOMC_TFIDF_CONTEXT_ANCHORS: tuple[str, ...] = (
    "federal funds rate target range monetary policy stance committee decision",
    "inflation remains elevated two percent objective price stability expectations",
    "labor market maximum employment unemployment job gains wage growth",
    "economic activity growth spending production financial conditions credit conditions",
    "ongoing increases additional policy firming sufficiently restrictive",
    "reduce target range rate cuts easing policy restraint",
    "data dependent incoming information balance of risks meeting by meeting",
    "forward guidance considerable time act as appropriate greater confidence",
)


def select_tfidf_policy_context(text: str, *, max_chars: int = 6000, top_k: int = 12) -> dict[str, Any]:
    """Select policy-relevant excerpts with TF-IDF before DeepSeek scoring."""
    normalized = _normalize_text(text)
    sentences = _split_sentences(normalized)
    if not sentences:
        return {
            "selected_text": "",
            "diagnostics": {
                "filter": "tfidf_policy_context",
                "input_sentence_count": 0,
                "selected_sentence_count": 0,
                "top_sentences": [],
            },
        }

    query_text = " ".join(FOMC_TFIDF_CONTEXT_ANCHORS)
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words="english", min_df=1)
    matrix = vectorizer.fit_transform([query_text, *sentences])
    scores = cosine_similarity(matrix[1:], matrix[0]).ravel()
    ranked = sorted(enumerate(scores), key=lambda item: (float(item[1]), -item[0]), reverse=True)

    selected_indices: list[int] = []
    total_chars = 0
    for index, score in ranked[: max(top_k, 1)]:
        sentence = sentences[index]
        if total_chars and total_chars + len(sentence) + 1 > max_chars:
            continue
        selected_indices.append(index)
        total_chars += len(sentence) + 1
        if total_chars >= max_chars:
            break

    if not selected_indices:
        selected_indices = [ranked[0][0]]

    selected_indices = sorted(set(selected_indices))
    selected_text = "\n".join(sentences[index] for index in selected_indices)
    if len(selected_text) > max_chars:
        selected_text = selected_text[:max_chars]

    top_sentences = [
        {
            "rank": rank + 1,
            "sentence_index": index,
            "score": round(float(score), 6),
            "text_preview": sentences[index][:180],
        }
        for rank, (index, score) in enumerate(ranked[: min(top_k, 8)])
    ]
    return {
        "selected_text": selected_text,
        "diagnostics": {
            "filter": "tfidf_policy_context",
            "input_sentence_count": len(sentences),
            "selected_sentence_count": len(selected_indices),
            "top_sentences": top_sentences,
        },
    }


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _split_sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?;])\s+", text) if part.strip()]
    return sentences or ([text] if text else [])


def demo_semantic_pipeline() -> None:
    """Demonstrate the full Agent 1 + Agent 2 pipeline."""
    print("=" * 60)
    print("Agent 1 + 2: source monitoring + semantic extraction")
    print("=" * 60 + "\n")
    
    # --- Agent 1: registry-backed source monitoring ---
    print("[Agent 1] source registry dry-run")
    print("-" * 40)
    
    fomc_url = "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260617a.htm"
    source_spec = PolicySourceSpec(
        source_id="fed_fomc_statement_demo",
        url=fomc_url,
        parser="generic_html",
        cadence="on_demand",
        metadata={"institution": "Federal Reserve", "document_type": "fomc_statement"},
    )
    print(f"  Source: {source_spec.source_id}")
    print(f"  Parser: {source_spec.parser}")
    
    # --- Run the source monitor ---
    print("[Agent 1] Fetching normalized document...")
    try:
        docs = fetch_source(source_spec)
        result = docs[0].to_dict()
        print(f"  Title: {result.get('title', 'N/A')}")
        print(f"  Hash: {result.get('text_hash', '')[:12]}")
        print(f"  Words: {len(result.get('text', '').split())}")
        incremental = incremental_fetch([source_spec], dry_run=True)
        print(f"  Incremental dry-run new documents: {incremental['new_documents']}")

        fomc_text = result.get("text", "")
        
        # --- Agent 2: Semantic Scoring ---
        print(f"\n[Agent 2] LLM Semantic Hawkish-Dovish Scoring")
        print("-" * 40)
        
        scores = score_hawkish_dovish(fomc_text[:6000], result.get("meeting_date"))
        
        print(f"  Method: {scores.pop('_method', 'unknown')}")
        print(f"  Hawkish-Dovish Score: {scores.get('hawkish_dovish_score', 0):+.3f}")
        print(f"  Inflation Concern:    {scores.get('inflation_concern', 0):.3f}")
        print(f"  Rate Hike Signal:     {scores.get('rate_hike_signal', 0):.3f}")
        print(f"  Rate Cut Signal:      {scores.get('rate_cut_signal', 0):.3f}")
        print(f"  Uncertainty:          {scores.get('uncertainty_index', 0):.3f}")
        print(f"  Forward Guidance:     {scores.get('forward_guidance_strength', 0):.3f}")
        
        # Save
        semantic_path = REPO_ROOT / "data" / "processed" / "semantic_scores.json"
        semantic_path.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "source_url": fomc_url,
            "published_at": result.get("published_at"),
            "text_hash": result.get("text_hash"),
            "scores": scores,
            "extracted_at": datetime.now().isoformat(),
        }
        semantic_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"\n  Saved → {semantic_path.relative_to(REPO_ROOT)}")
        
    except Exception as e:
        print(f"  Source monitor failed: {e}")
        print("  (Expected in offline or blocked-network environments. Use a test http_get for dry runs.)")
    
    print("\nAgent 1 + 2 pipeline demo complete.")


if __name__ == "__main__":
    demo_semantic_pipeline()
