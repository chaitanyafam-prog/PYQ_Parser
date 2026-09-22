"""PDF ingestion helpers. Heavy ML dependencies are imported only when used."""

from __future__ import annotations

import hashlib
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


SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
MAX_CHUNK_CHARS = 1_200
MIN_CHUNK_CHARS = 350


def _sentences(text: str) -> list[str]:
    """Split into sentence-like units without an additional NLP download."""
    return [sentence.strip() for sentence in SENTENCE_PATTERN.split(text.replace("\n", " ")) if sentence.strip()]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    return dot_product / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _semantic_text_chunks(text: str) -> list[str]:
    """Group adjacent sentences until their meaning changes or a chunk is full."""
    sentences = _sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [sentences[0]]

    embeddings = get_embeddings().embed_documents(sentences)
    similarities = [_cosine_similarity(embeddings[index], embeddings[index + 1]) for index in range(len(embeddings) - 1)]
    # The lowest-similarity 20% of sentence transitions are likely topic changes.
    topic_shift_threshold = _percentile(similarities, 20)
    chunks, current = [], [sentences[0]]
    for index, sentence in enumerate(sentences[1:]):
        current_text = " ".join(current)
        is_topic_shift = similarities[index] <= topic_shift_threshold
        would_exceed_limit = len(current_text) + len(sentence) + 1 > MAX_CHUNK_CHARS
        if (would_exceed_limit or is_topic_shift) and len(current_text) >= MIN_CHUNK_CHARS:
            chunks.append(current_text)
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_documents(documents):
    """Create semantic chunks by grouping sentences with related embeddings."""
    from langchain_core.documents import Document

    chunks = []
    for document in documents:
        for chunk_index, text in enumerate(_semantic_text_chunks(document.page_content)):
            metadata = deepcopy(document.metadata)
            metadata["chunk_index"] = chunk_index
            chunks.append(Document(page_content=text, metadata=metadata))
    print(f"Split into {len(chunks)} semantic chunks")
    return chunks


def _chunk_id(chunk) -> str:
    """Stable IDs make ingestion idempotent, including after an app rerun."""
    source, page = str(chunk.metadata.get("source", "")), str(chunk.metadata.get("page", ""))
    return hashlib.sha256(f"{source}\0{page}\0{chunk.page_content}".encode("utf-8")).hexdigest()


def embed_and_store(chunks, persist_directory: str | Path = DB_DIR):
    """Embed only chunks Chroma does not already contain; return the number added."""
    if not chunks:
        return 0
    vectordb = get_vector_store(persist_directory)
    ids = [_chunk_id(chunk) for chunk in chunks]
    existing = set(vectordb.get(ids=ids, include=[])["ids"])
    new_pairs = [(chunk, doc_id) for chunk, doc_id in zip(chunks, ids) if doc_id not in existing]
    if new_pairs:
        new_chunks, new_ids = zip(*new_pairs)
        vectordb.add_documents(list(new_chunks), ids=list(new_ids))
    print(f"Added {len(new_pairs)} new chunks to Chroma at {persist_directory}")
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
