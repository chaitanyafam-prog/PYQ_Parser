"""Fast vector-only retrieval and grounded answer generation."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ingest import BLOOM_LEVELS, get_embeddings, get_vector_store, parse_model_object

RETRIEVAL_VERSION = "question-bank-generation-v1"
DUPLICATE_SIMILARITY_THRESHOLD = 0.92

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

GENERATION_PROMPT = """Create one new exam question from the examples below.
The new question must assess the requested topic, mark value, student year, and
Bloom's level. It must be meaningfully different from every example: do not copy,
paraphrase, or change only names/numbers. Return only valid JSON with these keys:
"question", "topic", "mark_value", "bloom_level".

Topic: {topic}
Mark value: {mark_value}
Student year: {student_year}
Bloom's level: {bloom_level}

Examples:
{examples}
"""


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


def _as_json_object(content) -> dict:
    return parse_model_object(_as_readable_text(content))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _mark_slots(constraints: dict) -> list[dict]:
    distribution = constraints.get("mark_distribution", [])
    if isinstance(distribution, str):
        distribution = [
            {"count": int(count), "marks": int(marks)}
            for count, marks in re.findall(r"(\d+)\s*(?:questions?\s*x?|x)\s*(\d+)\s*marks?", distribution, re.IGNORECASE)
        ]
    slots = []
    for item in distribution:
        count = int(item.get("count", item.get("quantity", 0)))
        marks = int(item.get("marks", item.get("mark_value", 0)))
        slots.extend({"mark_value": marks} for _ in range(count))
    if not slots and constraints.get("total_marks"):
        slots.append({"mark_value": int(constraints["total_marks"])})
    return slots


def _bloom_slots(constraints: dict, count: int) -> list[str | None]:
    target = constraints.get("bloom_targets", constraints.get("bloom_levels", {}))
    if isinstance(target, str):
        target = [level.strip().title() for level in target.split(",") if level.strip()]
    if isinstance(target, list) and target:
        return (target * count)[:count]
    if isinstance(target, list):
        return [None] * count
    target = target or {}
    levels = []
    for level in BLOOM_LEVELS:
        levels.extend([level] * int(target.get(level, target.get(level.lower(), 0))))
    return (levels + [None] * count)[:count]


def _topics(constraints: dict) -> list[str]:
    topics = constraints.get("topics", constraints.get("units", []))
    if isinstance(topics, str):
        topics = [topic.strip() for topic in topics.split(",") if topic.strip()]
    return [str(topic) for topic in topics]


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

    def _metadata_filter(self, topic: str | None, mark_value: int, bloom_level: str | None, topics: list[str]):
        filters = [{"mark_value": {"$eq": mark_value}}]
        if topic:
            filters.append({"topic": {"$eq": topic}})
        elif topics:
            filters.append({"topic": {"$in": topics}})
        if bloom_level:
            filters.append({"bloom_level": {"$eq": bloom_level}})
        return filters[0] if len(filters) == 1 else {"$and": filters}

    def retrieve_candidates(
        self,
        topic: str | None,
        mark_value: int,
        bloom_level: str | None,
        topics: list[str],
        query: str,
        k: int = 4,
    ) -> list[dict]:
        """Retrieve question-bank examples using metadata and semantic relevance."""
        metadata_filter = self._metadata_filter(topic, mark_value, bloom_level, topics)
        results = self.vectordb.similarity_search_with_relevance_scores(
            query,
            k=k,
            filter=metadata_filter,
        )
        return [
            {"text": document.page_content, "metadata": document.metadata, "score": score, "generated": False}
            for document, score in results
        ]

    def _is_duplicate(self, question: str, topic: str | None, topics: list[str], draft: list[dict]) -> bool:
        question_embedding = get_embeddings().embed_query(question)
        if any(_cosine_similarity(question_embedding, get_embeddings().embed_query(item["text"])) >= DUPLICATE_SIMILARITY_THRESHOLD for item in draft):
            return True
        collection = getattr(self.vectordb, "_collection", None)
        if collection is None:
            return False
        where = {"topic": {"$eq": topic}} if topic else ({"topic": {"$in": topics}} if topics else None)
        stored = collection.get(where=where, include=["embeddings"])
        return any(
            _cosine_similarity(question_embedding, embedding) >= DUPLICATE_SIMILARITY_THRESHOLD
            for embedding in (stored.get("embeddings") or [])
            if embedding
        )

    def _generate_question(self, slot: dict, examples: list[dict], constraints: dict, draft: list[dict], topics: list[str]):
        topic = slot.get("topic") or (topics[0] if topics else "the syllabus")
        bloom_level = slot.get("bloom_level") or "Understand"
        prompt = GENERATION_PROMPT.format(
            topic=topic,
            mark_value=slot["mark_value"],
            student_year=constraints.get("student_year", "the target student year"),
            bloom_level=bloom_level,
            examples="\n\n".join(item["text"] for item in examples[:4]),
        )
        for _ in range(3):
            response = self.llm.invoke(prompt)
            generated = _as_json_object(getattr(response, "content", response))
            question = str(generated.get("question", "")).strip()
            if not question:
                continue
            if self._is_duplicate(question, topic, topics, draft):
                prompt += "\nThe previous output was too similar. Generate a substantially different question."
                continue
            metadata = {
                "topic": str(generated.get("topic", topic)),
                "mark_value": int(generated.get("mark_value", slot["mark_value"])),
                "bloom_level": str(generated.get("bloom_level", bloom_level)).title(),
                "source_year": "Generated",
                "source_paper": "Generated",
            }
            return {"text": question, "metadata": metadata, "score": None, "generated": True}
        return None

    def generate_paper(self, constraints: dict) -> list[dict]:
        """Assemble a constrained paper from stored questions and new questions."""
        slots = _mark_slots(constraints)
        topics = _topics(constraints)
        bloom_levels = _bloom_slots(constraints, len(slots))
        draft = []
        for index, slot in enumerate(slots):
            slot["bloom_level"] = bloom_levels[index]
            slot["topic"] = topics[index % len(topics)] if topics else None
            query = f"{slot['topic'] or 'exam'} question for {constraints.get('student_year', 'students')}"
            candidates = self.retrieve_candidates(
                slot["topic"], slot["mark_value"], slot["bloom_level"], topics, query, k=4
            )
            candidates = [candidate for candidate in candidates if candidate["text"] not in {item["text"] for item in draft}]
            if candidates:
                draft.append(candidates[0])
                continue
            examples = self.retrieve_candidates(
                slot["topic"], slot["mark_value"], None, topics, query, k=4
            )
            generated = self._generate_question(slot, examples, constraints, draft, topics)
            if generated:
                draft.append(generated)
        return draft


# Compatibility for any external imports; retrieval is now vector-only.
HybridRAGChain = VectorRAGChain


def load_qa_chain(api_key: str, persist_directory: str | Path, k: int = 4):
    return VectorRAGChain(api_key=api_key, persist_directory=persist_directory, k=k)


def ask(qa_chain: VectorRAGChain, question: str):
    return qa_chain.ask(question)
