"""PDF ingestion helpers. Heavy ML dependencies are imported only when used."""

from __future__ import annotations

import hashlib
import json
import re
import string
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "langchain"


@lru_cache(maxsize=1)
def get_embeddings():
    """Load MiniLM once per Python process (the slowest local startup step)."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def user_storage_paths(user_id: int) -> tuple[Path, Path]:
    """Return isolated PDF and Chroma directories for one authenticated user."""
    safe_user_id = str(user_id)
    if not safe_user_id or any(char not in string.ascii_letters + string.digits + "_-" for char in safe_user_id):
        raise ValueError("Invalid user storage identifier.")
    documents_dir = DATA_DIR / "users" / safe_user_id
    vector_db_dir = DB_DIR / "users" / safe_user_id
    documents_dir.mkdir(parents=True, exist_ok=True)
    vector_db_dir.mkdir(parents=True, exist_ok=True)
    return documents_dir, vector_db_dir


@lru_cache(maxsize=32)
def get_vector_store(persist_directory: str | Path = DB_DIR):
    from langchain_chroma import Chroma
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(Path(persist_directory).resolve()),
        embedding_function=get_embeddings(),
    )


def load_pdf(path: str | Path):
    from langchain_community.document_loaders import PyPDFLoader
    return PyPDFLoader(str(path)).load()


def load_documents():
    documents = []
    for path in DATA_DIR.glob("*.pdf"):
        documents.extend(load_pdf(path))
    print(f"Loaded {len(documents)} pages from {DATA_DIR}")
    return documents


QUESTION_START_PATTERN = re.compile(
    r"^\s*(?:(?:question|ques|q)\s*\d+\s*[:.)]?|\d+\s*[.)]|\(?[a-z]\)\s*)",
    re.IGNORECASE,
)
MARK_PATTERN = re.compile(r"(?:\[|\(|\b)\s*(\d+)\s*(?:marks?|mks?)\s*(?:\]|\))?", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
BLOOM_LEVELS = ("Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create")


def _source_year(metadata: dict) -> str:
    for key in ("source_year", "year", "paper_year"):
        if metadata.get(key):
            return str(metadata[key])
    match = YEAR_PATTERN.search(str(metadata.get("source", "")))
    return match.group(0) if match else "Unknown"


def _question_blocks(text: str) -> list[tuple[str, str]]:
    """Extract numbered question blocks while preserving their visible labels."""
    normalized = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()
    if not normalized:
        return []
    lines = normalized.splitlines()
    starts = [index for index, line in enumerate(lines) if QUESTION_START_PATTERN.match(line)]
    if not starts:
        return [("", normalized)]
    blocks = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        block = " ".join(line.strip() for line in lines[start:end] if line.strip())
        label_match = QUESTION_START_PATTERN.match(block)
        label = label_match.group(0).strip() if label_match else ""
        if block:
            blocks.append((label, block))
    return blocks


def extract_questions(documents):
    """Create one LangChain Document per numbered question in the PDFs."""
    from langchain_core.documents import Document

    questions = []
    for document in documents:
        for question_index, (label, text) in enumerate(_question_blocks(document.page_content)):
            metadata = deepcopy(document.metadata)
            metadata.update({
                "question_index": question_index,
                "question_label": label,
                "source_year": _source_year(metadata),
                "source_paper": Path(str(metadata.get("source", "unknown"))).stem,
            })
            questions.append(Document(page_content=text, metadata=metadata))
    print(f"Extracted {len(questions)} questions")
    return questions


def _tagging_prompt(question: str, syllabus: str, source_year: str) -> str:
    return f'''Tag this exam question. Return only valid JSON with exactly these keys:
"topic", "mark_value", "bloom_level", "source_year".
Use one topic/unit name from the syllabus when possible. mark_value must be an integer.
bloom_level must be exactly one of {", ".join(BLOOM_LEVELS)}. Preserve the supplied source year.

Syllabus:
{syllabus or "No syllabus was supplied; infer a concise topic from the question."}

Source year: {source_year}
Question:
{question}'''


def _json_object(content) -> dict:
    text = content if isinstance(content, str) else str(content)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("The tagging model did not return a JSON object.")
    return json.loads(match.group(0))


def _fallback_tags(question: str, metadata: dict) -> dict:
    mark_match = MARK_PATTERN.search(question)
    return {
        "topic": "Unclassified",
        "mark_value": int(mark_match.group(1)) if mark_match else 0,
        "bloom_level": "Understand",
        "source_year": _source_year(metadata),
    }


def tag_questions(questions, syllabus_text: str = "", llm=None):
    """Attach syllabus-aligned LLM tags to question Documents.

    Passing ``llm=None`` keeps parsing and local ingestion usable for tests and
    callers that want to defer model tagging; production ingestion should pass
    a Gemini chat model.
    """
    from langchain_core.documents import Document

    tagged = []
    for question in questions:
        metadata = deepcopy(question.metadata)
        tags = _fallback_tags(question.page_content, metadata)
        if llm is not None:
            response = llm.invoke(_tagging_prompt(question.page_content, syllabus_text, tags["source_year"]))
            model_tags = _json_object(getattr(response, "content", response))
            tags.update(model_tags)
        bloom = str(tags.get("bloom_level", "Understand")).title()
        metadata.update({
            "topic": str(tags.get("topic", "Unclassified")),
            "mark_value": int(tags.get("mark_value", 0) or 0),
            "bloom_level": bloom if bloom in BLOOM_LEVELS else "Understand",
            "source_year": str(tags.get("source_year") or metadata["source_year"]),
        })
        tagged.append(Document(page_content=question.page_content, metadata=metadata))
    return tagged


def chunk_documents(documents, syllabus_text: str = "", llm=None):
    """Compatibility wrapper: ingestion units are now individual questions."""
    return tag_questions(extract_questions(documents), syllabus_text=syllabus_text, llm=llm)


def _chunk_id(chunk) -> str:
    """Stable IDs make question ingestion idempotent, including after reruns."""
    source = str(chunk.metadata.get("source", ""))
    question_index = str(chunk.metadata.get("question_index", ""))
    return hashlib.sha256(f"{source}\0{question_index}\0{chunk.page_content}".encode("utf-8")).hexdigest()


def embed_and_store(questions, persist_directory: str | Path = DB_DIR):
    """Embed only questions Chroma does not already contain; return the number added."""
    if not questions:
        return 0
    vectordb = get_vector_store(persist_directory)
    ids = [_chunk_id(question) for question in questions]
    existing = set(vectordb.get(ids=ids, include=[])["ids"])
    new_pairs = [(question, doc_id) for question, doc_id in zip(questions, ids) if doc_id not in existing]
    if new_pairs:
        new_chunks, new_ids = zip(*new_pairs)
        vectordb.add_documents(list(new_chunks), ids=list(new_ids))
    print(f"Added {len(new_pairs)} new questions to Chroma at {persist_directory}")
    return len(new_pairs)


def remove_document(file_path: str | Path, documents_directory: str | Path, persist_directory: str | Path):
    """Remove one user's PDF and all vector chunks created from that exact file."""
    document_path = Path(file_path).resolve()
    user_documents = Path(documents_directory).resolve()
    try:
        document_path.relative_to(user_documents)
    except ValueError as exc:
        raise ValueError("The selected file is outside this user's document folder.") from exc
    if document_path.suffix.lower() != ".pdf" or not document_path.is_file():
        raise ValueError("The selected document does not exist.")

    # PyPDFLoader stores the absolute file path in each chunk's source metadata.
    get_vector_store(persist_directory).delete(where={"source": str(document_path)})
    document_path.unlink()


if __name__ == "__main__":
    docs = load_documents()
    if not docs:
        print(f"No PDFs found in {DATA_DIR}. Add at least one PDF and re-run.")
    else:
        print(f"Ingestion complete: {embed_and_store(chunk_documents(docs))} new chunks added.")
