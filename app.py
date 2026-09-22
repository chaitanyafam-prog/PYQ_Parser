"""Streamlit interface for the vector-only knowledge assistant."""

from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from database import authenticate_user, create_user, is_configured, load_messages, save_message
from ingest import DATA_DIR, chunk_documents, embed_and_store, load_pdf, remove_document, user_storage_paths
from rag_chain import RETRIEVAL_VERSION, VectorRAGChain


load_dotenv()
st.set_page_config(page_title="Mini AI Knowledge Assistant", page_icon="📚", layout="wide")
st.title("📚 Mini AI Knowledge Assistant")
st.caption("Upload documents, then ask questions grounded in their content.")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def show_authentication():
    st.subheader("Sign in to your knowledge assistant")
    st.caption("Sign in with your email. Accounts and chat memory are managed by Supabase.")
    login_tab, signup_tab = st.tabs(["Sign in", "Create account"])
    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email address", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Sign in", type="primary")
        if submitted:
            authenticated, result = authenticate_user(email, password)
            if authenticated:
                st.session_state.user = result
                st.rerun()
            st.error(result)
    with signup_tab:
        with st.form("signup_form", clear_on_submit=True):
            email = st.text_input("Email address")
            password = st.text_input("Choose a password", type="password", help="At least 8 characters.")
            confirm_password = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create account", type="primary")
        if submitted:
            if password != confirm_password:
                st.error("The passwords do not match.")
            else:
                created, result = create_user(email, password)
                if created:
                    st.session_state.user = result
                    st.rerun()
                st.error(result)


if not is_configured():
    st.error("Supabase is not configured. Copy `.env.example` to `.env` and add your project credentials.")
    st.stop()

# A previous SQLite session does not have a Supabase access token.
if "user" in st.session_state and "access_token" not in st.session_state.user:
    del st.session_state.user
if "user" not in st.session_state:
    show_authentication()
    st.stop()

user_documents_dir, user_vector_db_dir = user_storage_paths(st.session_state.user["id"])

with st.sidebar:
    st.caption(f"Signed in as **{st.session_state.user['email']}**")
    if st.button("Log out"):
        del st.session_state.user
        st.rerun()
    st.divider()

api_key = os.getenv("GOOGLE_API_KEY") or st.sidebar.text_input(
    "Google API Key", type="password", help="Get a free key at https://aistudio.google.com/apikey"
)
if not api_key:
    st.warning("Enter your Google API key in the sidebar to continue.")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_chain(key: str, user_id: int, vector_db_directory: str, retrieval_version: str):
    """Load the embedding model and vector database only for a submitted question."""
    return VectorRAGChain(api_key=key, persist_directory=vector_db_directory)


st.sidebar.header("📄 Documents")
uploaded_files = st.sidebar.file_uploader(
    "Drag and drop PDF(s) here, or browse files",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can drag one or more PDF files into this area, or click Browse files.",
)
if uploaded_files and st.sidebar.button("Process documents", type="primary"):
    with st.spinner("Extracting, chunking, and embedding new content..."):
        documents = []
        for uploaded_file in uploaded_files:
            save_path = user_documents_dir / Path(uploaded_file.name).name
            save_path.write_bytes(uploaded_file.getbuffer())
            documents.extend(load_pdf(save_path))
        added = embed_and_store(chunk_documents(documents), str(user_vector_db_dir))
    get_chain.clear()
    st.sidebar.success(f"Processed {len(uploaded_files)} file(s): {added} new chunk(s) embedded.")

existing_files = sorted(path.name for path in user_documents_dir.glob("*.pdf"))
st.sidebar.caption(f"Currently indexed: {', '.join(existing_files)}" if existing_files else "No documents uploaded yet.")
if existing_files:
    with st.sidebar.expander("Remove a document"):
        document_to_remove = st.selectbox("Document", existing_files, key="document_to_remove")
        st.caption("This removes the PDF and its indexed chunks from your account.")
        confirmed_removal = st.checkbox("I understand this cannot be undone.", key="confirmed_document_removal")
        if st.button("Remove document", type="secondary", disabled=not confirmed_removal):
            try:
                remove_document(
                    user_documents_dir / document_to_remove,
                    user_documents_dir,
                    str(user_vector_db_dir),
                )
                get_chain.clear()
                st.rerun()
            except (OSError, ValueError) as exc:
                st.error(f"Could not remove the document: {exc}")
if not existing_files:
    st.info("👈 Upload a PDF and click **Process documents** in the sidebar to get started.")
    st.stop()


def show_sources(sources):
    if not sources:
        return
    with st.expander(f"View {len(sources)} source chunk(s)"):
        for number, item in enumerate(sources, 1):
            metadata = item["metadata"]
            source_name = Path(str(metadata.get("source", "unknown"))).name
            page = metadata.get("page")
            page_label = f"Page {int(page) + 1}" if isinstance(page, int) else "Page unknown"
            chunk_label = metadata.get("chunk_index")
            details = page_label + (f" · Section {int(chunk_label) + 1}" if isinstance(chunk_label, int) else "")
            score = f" · Relevance {item['score']:.0%}" if item["score"] is not None else ""
            # Semantic chunks are short enough to show in full. Normalize the PDF's
            # line wrapping so they read like a natural document excerpt.
            excerpt = re.sub(r"\s+", " ", item["text"]).strip()
            with st.container(border=True):
                st.markdown(f"**{number}. {source_name}**")
                st.caption(f"{details}{score}")
                st.write(excerpt)


# Load only the authenticated user's saved chat memory.
for message in load_messages(st.session_state.user):
    with st.chat_message("user"):
        st.write(message["question"])
    with st.chat_message("assistant"):
        st.markdown(message["answer"])
        show_sources(message["sources"])

question = st.chat_input("Ask a question about your documents...")
if question:
    # This delta is sent before the slower retrieval/model request begins.
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving and generating answer..."):
                answer, sources = get_chain(
                    api_key,
                    st.session_state.user["id"],
                    str(user_vector_db_dir),
                    RETRIEVAL_VERSION,
                ).ask(question)
            st.markdown(answer)
            show_sources(sources)
            save_message(st.session_state.user, question, answer, sources)
        except Exception as exc:
            st.error(f"Could not query the knowledge base: {exc}")
