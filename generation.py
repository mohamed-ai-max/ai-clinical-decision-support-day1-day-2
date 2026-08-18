"""Day 3 grounded generation and abstain logic."""

import re

import config
from dense_reranker import DenseRerankRetriever
from retrieval import retrieve_with_citation


def _tokenize(text: str):
    return re.findall(r"[a-zA-Z]{3,}", text.lower())


def _select_best_sentence(question: str, chunks: list[dict]) -> str:
    text_blocks = []
    for chunk in chunks:
        text_blocks.extend(
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", chunk.get("text", ""))
            if s.strip()
        )

    if not text_blocks:
        return "The retrieved evidence does not contain a complete answer to this question."

    question_tokens = set(_tokenize(question))
    scored = []
    for sentence in text_blocks:
        sentence_tokens = set(_tokenize(sentence))
        overlap = len(question_tokens & sentence_tokens)
        sentence_len = len(sentence)
        score = overlap * 2 + min(sentence_len, 220) / 80
        scored.append((score, sentence))

    _, best = max(scored, key=lambda item: item[0])
    return best[:500].strip()


def answer_question(vectordb, question: str, k: int | None = None, use_reranker: bool = False, reranker=None):
    """Return a grounded answer and citation metadata, or abstain if evidence is weak."""
    retrieved = retrieve_with_citation(vectordb, question, k=k, use_reranker=use_reranker, reranker=reranker)
    if not retrieved:
        return {
            "status": "abstain",
            "answer": "I can’t answer this from the hypertension guideline corpus because no relevant evidence was retrieved.",
            "citations": [],
            "top_score": 0.0,
            "confidence": "uncertain",
            "grounded_chunks": [],
        }

    top = retrieved[0]
    top_score = float(top.get("score", top.get("dense_score", 0.0)))
    if top_score < config.OUT_OF_SCOPE_MAX_SCORE:
        return {
            "status": "abstain",
            "answer": "I can’t answer this confidently because the retrieved guideline evidence is below the acceptable threshold.",
            "citations": [],
            "top_score": top_score,
            "confidence": "uncertain",
            "grounded_chunks": retrieved[:min(3, len(retrieved))],
        }

    used_chunks = retrieved[: min(3, len(retrieved))]
    answer_text = _select_best_sentence(question, used_chunks)
    citations = []
    seen = set()
    for chunk in used_chunks:
        key = (chunk.get("document_name"), chunk.get("page_number"))
        if key in seen:
            continue
        seen.add(key)
        citations.append({
            "document_name": chunk.get("document_name"),
            "page_number": chunk.get("page_number"),
            "chunk_id": chunk.get("chunk_id"),
        })

    return {
        "status": "grounded",
        "answer": answer_text,
        "citations": citations,
        "top_score": top_score,
        "confidence": top.get("confidence", "confident"),
        "grounded_chunks": used_chunks,
    }


def answer_question_cli(question: str, k: int | None = None, use_reranker: bool = False):
    """Convenience wrapper for the project CLI."""
    from query import load_index

    vectordb = load_index()
    return answer_question(vectordb, question, k=k, use_reranker=use_reranker)


if __name__ == "__main__":
    test_question = "What is the target blood pressure for a patient with cardiovascular disease?"
    result = answer_question_cli(test_question)
    print(result["status"].upper())
    print(result["answer"])
    print("Citations:")
    for c in result["citations"]:
        print(f"  - {c['document_name']} (p.{c['page_number']})")
