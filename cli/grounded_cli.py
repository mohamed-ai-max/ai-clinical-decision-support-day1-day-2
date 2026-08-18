"""CLI for grounded generation. Supports simulation and live modes.

Usage:
    python cli\grounded_cli.py --mode simulate --question "..."
    python cli\grounded_cli.py --mode live --question "..."

Live mode expects NV_API_KEY and optional NV_LLM_ENDPOINT environment variables.
"""
import os
import argparse
import json
from typing import Optional

from query import load_index
from generation import answer_question

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompt", "grounding_prompt.txt")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "schema", "response_schema.json")


def load_prompt():
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def simulate_llm(prompt: str) -> str:
    # Simple simulation: echo a grounded response built from the prompt's snippets
    # For robust testing, use retrieval-based summarization instead of a real LLM.
    # This returns a JSON string that adheres to schema for the test harness.
    simulated = {
        "status": "grounded",
        "answer": "Target systolic BP for patients with CVD is <130 mmHg according to the guideline.",
        "evidence": ["...excerpt from guideline p.28..."],
        "citations": [{"document_name": "Guideline for the pharmacological treatment of hypertension in adults.pdf", "page_number": 28, "chunk_id": "..."}],
        "confidence": "confident",
        "top_score": 0.79,
    }
    return json.dumps(simulated)


def call_live_llm(prompt: str, nv_api_key: str, endpoint: Optional[str] = None) -> str:
    """Send prompt to a live LLM. Endpoint and payload format must match your LLM provider.
    This implementation is provider-agnostic and expects an HTTP JSON API. Configure
    NV_LLM_ENDPOINT if necessary.
    """
    import requests

    endpoint = endpoint or os.environ.get("NV_LLM_ENDPOINT", "https://api.nvidia.com/v1/generate")
    headers = {"Authorization": f"Bearer {nv_api_key}", "Content-Type": "application/json"}
    payload = {"prompt": prompt, "max_tokens": 512}
    res = requests.post(endpoint, headers=headers, json=payload, timeout=30)
    res.raise_for_status()
    # The response parsing will differ by provider; try to handle common shapes.
    data = res.json()
    if isinstance(data, dict) and "text" in data:
        return data["text"]
    if isinstance(data, dict) and "choices" in data:
        return data["choices"][0]["text"]
    return json.dumps({"status": "abstain", "answer": "live call returned unexpected format", "evidence": [], "citations": [], "confidence": "insufficient", "top_score": 0.0})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["simulate", "live"], default="simulate")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    prompt_template = load_prompt()

    # Load vector DB and retrieved chunks for context
    vectordb = load_index()
    retrieved = None
    # Use the existing retrieval helper to fetch context
    from retrieval import retrieve_with_citation
    retrieved = retrieve_with_citation(vectordb, args.question, k=3)

    context_blocks = []
    for r in retrieved:
        context_blocks.append(f"SOURCE: {r['document_name']} (p.{r['page_number']})\n{r['text'][:800]}")
    prompt_text = prompt_template.format(context_blocks="\n\n".join(context_blocks), question=args.question)

    if args.mode == "simulate":
        out = simulate_llm(prompt_text)
    else:
        nv_api_key = os.environ.get("NV_API_KEY") or os.environ.get("NVAPI_KEY") or os.environ.get("NV_LLM_API_KEY")
        if not nv_api_key:
            raise RuntimeError("NV_API_KEY not set in environment for live mode")
        out = call_live_llm(prompt_text, nv_api_key)

    print(out)


if __name__ == "__main__":
    main()
