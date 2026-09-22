"""Rubric validation for generated question papers."""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Callable

DUPLICATE_SIMILARITY_THRESHOLD = 0.92


def _question_text(question: Any) -> str:
    if isinstance(question, dict):
        return str(question.get("text", question.get("question", ""))).strip()
    return str(getattr(question, "page_content", question)).strip()


def _metadata(question: Any) -> dict:
    if isinstance(question, dict):
        metadata = question.get("metadata", {})
        if isinstance(question.get("tags"), dict):
            metadata = {**metadata, **question["tags"]}
        return metadata
    return getattr(question, "metadata", {}) or {}


def _mark_distribution(constraints: dict) -> Counter:
    distribution = constraints.get("mark_distribution", [])
    if isinstance(distribution, str):
        distribution = [
            {"count": int(count), "marks": int(marks)}
            for count, marks in re.findall(
                r"(\d+)\s*(?:questions?\s*x?|x)\s*(\d+)\s*marks?",
                distribution,
                re.IGNORECASE,
            )
        ]
    return Counter(
        {
            int(item.get("marks", item.get("mark_value", 0))): int(
                item.get("count", item.get("quantity", 0))
            )
            for item in distribution
        }
    )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _violation(kind: str, message: str, expected=None, actual=None) -> dict:
    return {
        "type": kind,
        "message": message,
        "expected": expected,
        "actual": actual,
        "severity": "error",
    }


def validate_paper(
    paper: list[Any],
    constraints: dict,
    embedding_function: Callable[[list[str]], list[list[float]]] | None = None,
    duplicate_threshold: float = DUPLICATE_SIMILARITY_THRESHOLD,
) -> dict:
    """Return a structured rubric report for a draft paper.

    ``embedding_function`` should accept a list of question texts and return one
    vector per text. If omitted, lexical similarity is still checked; callers can
    pass ``get_embeddings().embed_documents`` for semantic near-duplicate checks.
    """
    records = [{"text": _question_text(item), "metadata": _metadata(item)} for item in paper]
    marks = [int(record["metadata"].get("mark_value", 0) or 0) for record in records]
    topics = [str(record["metadata"].get("topic", "")).strip() for record in records]
    bloom_levels = [str(record["metadata"].get("bloom_level", "")).title() for record in records]
    expected_mark_counts = _mark_distribution(constraints)
    expected_total = constraints.get("total_marks")
    if expected_total is None and expected_mark_counts:
        expected_total = sum(mark * count for mark, count in expected_mark_counts.items())
    actual_mark_counts = Counter(marks)
    actual_total = sum(marks)

    violations = []
    if expected_total is not None and actual_total != int(expected_total):
        violations.append(_violation("total_marks", "Total marks do not match the rubric.", int(expected_total), actual_total))
    if expected_mark_counts and actual_mark_counts != expected_mark_counts:
        violations.append(
            _violation(
                "mark_distribution",
                "The number of questions at one or more mark values does not match the rubric.",
                dict(expected_mark_counts),
                dict(actual_mark_counts),
            )
        )

    required_topics = constraints.get("topics", constraints.get("units", []))
    if isinstance(required_topics, str):
        required_topics = [topic.strip() for topic in required_topics.split(",") if topic.strip()]
    required_topics = [str(topic) for topic in (required_topics or [])]
    missing_topics = [topic for topic in required_topics if topic not in topics]
    if missing_topics:
        violations.append(_violation("topic_coverage", "Required topics are missing.", required_topics, sorted(set(topics))))

    bloom_targets = constraints.get("bloom_targets", constraints.get("bloom_levels", {})) or {}
    if isinstance(bloom_targets, dict):
        actual_bloom = Counter(bloom_levels)
        missing_bloom = {
            level: int(count) - actual_bloom.get(level.title(), 0)
            for level, count in bloom_targets.items()
            if int(count) > actual_bloom.get(level.title(), 0)
        }
        if missing_bloom:
            violations.append(_violation("bloom_distribution", "Bloom's level targets are underrepresented.", bloom_targets, dict(actual_bloom)))
    else:
        actual_bloom = Counter(bloom_levels)

    duplicate_pairs = []
    embeddings = embedding_function([record["text"] for record in records]) if embedding_function and records else []
    for left_index, left in enumerate(records):
        for right_index in range(left_index + 1, len(records)):
            lexical_similarity = SequenceMatcher(None, _normalized(left["text"]), _normalized(records[right_index]["text"])).ratio()
            semantic_similarity = (
                _cosine_similarity(embeddings[left_index], embeddings[right_index])
                if embeddings
                else 0.0
            )
            similarity = max(lexical_similarity, semantic_similarity)
            if similarity >= duplicate_threshold:
                duplicate_pairs.append({"indexes": [left_index, right_index], "similarity": round(similarity, 4)})
    if duplicate_pairs:
        violations.append(_violation("duplicates", "The paper contains duplicate or near-duplicate questions.", [], duplicate_pairs))

    return {
        "valid": not violations,
        "violations": violations,
        "checks": {
            "total_marks": {"expected": expected_total, "actual": actual_total, "passed": expected_total is None or actual_total == int(expected_total)},
            "mark_distribution": {"expected": dict(expected_mark_counts), "actual": dict(actual_mark_counts), "passed": not expected_mark_counts or actual_mark_counts == expected_mark_counts},
            "topic_coverage": {"required": required_topics, "missing": missing_topics, "passed": not missing_topics},
            "bloom_distribution": {"targets": bloom_targets, "actual": dict(actual_bloom), "passed": not any(item["type"] == "bloom_distribution" for item in violations)},
            "duplicates": {"pairs": duplicate_pairs, "passed": not duplicate_pairs},
        },
        "actual": {
            "question_count": len(records),
            "total_marks": actual_total,
            "mark_distribution": dict(actual_mark_counts),
            "topics": sorted(set(topic for topic in topics if topic)),
            "bloom_distribution": dict(actual_bloom),
        },
    }
