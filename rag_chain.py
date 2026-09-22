"""Fast vector-only retrieval and grounded answer generation."""

from __future__ import annotations

from pathlib import Path

from ingest import get_vector_store

RETRIEVAL_VERSION = "relevance-plus-mmr-v1"

ANSWER_PROMPT = """Answer using only the provided context. If the answer is not in
the context, say: "I don't have enough information in the provided documents to answer that."

Write in a warm, lightly conversational tone, as if explaining the answer directly
to the user. Start with a clear direct answer, then briefly explain the supporting
details from the context. Use simple language and connect ideas naturally; do not
sound like a list of extracted notes. Use Markdown bullets only when they genuinely
improve clarity. Do not return JSON, Python dictionaries, curly-brace objects, or
meta-commentary.

Context:
{context}

Question: {question}

Answer:"""


def _as_readable_text(content) -> str:
    """Gemini may return text blocks rather than a plain string."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        if parts:
            return "\n".join(parts).strip()
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"].strip()
    return str(content).strip()


class VectorRAGChain:
    def __init__(self, api_key: str, persist_directory: str | Path, k: int = 8, fetch_k: int = 20):
        from langchain_google_genai import ChatGoogleGenerativeAI
        self.k, self.fetch_k = k, max(fetch_k, k)
        self.vectordb = get_vector_store(persist_directory)
        self.llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=api_key, temperature=0.2)

    def ask(self, question: str):
        # Keep the strongest direct matches for precise questions, then add MMR
        # results for coverage elsewhere in a longer document.
        direct_results = self.vectordb.similarity_search_with_relevance_scores(
            question, k=self.fetch_k
        )
        direct_sources = [
            {"text": doc.page_content, "metadata": doc.metadata, "score": score}
            for doc, score in direct_results
        ]
        direct_count = min(4, self.k)
        sources = direct_sources[:direct_count]
        seen = {item["text"] for item in sources}
        try:
            documents = self.vectordb.max_marginal_relevance_search(
                question,
                k=self.k,
                fetch_k=self.fetch_k,
                lambda_mult=0.65,
            )
            for document in documents:
                if document.page_content not in seen and len(sources) < self.k:
                    seen.add(document.page_content)
                    sources.append({"text": document.page_content, "metadata": document.metadata, "score": None})
        except AttributeError:
            pass  # Direct results below are a compatible fallback.

        # Ensure the requested count even when MMR is unavailable or overlaps.
        for item in direct_sources[direct_count:]:
            if item["text"] not in seen and len(sources) < self.k:
                seen.add(item["text"])
                sources.append(item)
        if not sources:
            return "I don't have enough information in the provided documents to answer that.", []
        context = "\n\n---\n\n".join(item["text"] for item in sources)
        response = self.llm.invoke(ANSWER_PROMPT.format(context=context, question=question))
        return _as_readable_text(response.content), sources


# Compatibility for any external imports; retrieval is now vector-only.
HybridRAGChain = VectorRAGChain


def load_qa_chain(api_key: str, persist_directory: str | Path, k: int = 4):
    return VectorRAGChain(api_key=api_key, persist_directory=persist_directory, k=k)


def ask(qa_chain: VectorRAGChain, question: str):
    return qa_chain.ask(question)
