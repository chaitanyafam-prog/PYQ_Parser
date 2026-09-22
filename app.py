"""Streamlit interface for the Teacher's Question Bank Assistant."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from database import authenticate_user, create_user, is_configured
from ingest import (
    DATA_DIR,
    chunk_documents,
    embed_and_store,
    load_pdf,
    remove_document,
    user_storage_paths,
)
from rag_chain import BLOOM_LEVELS, RETRIEVAL_VERSION, VectorRAGChain
from rubric_checker import validate_paper


load_dotenv()
st.set_page_config(page_title="Teacher's Question Bank Assistant", page_icon="📝", layout="wide")
st.title("Teacher's Question Bank Assistant")
st.caption("Build a fresh exam paper from previous papers, guided by your rubric.")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def show_authentication():
    st.subheader("Sign in to your question bank")
    st.caption("Accounts and document storage are isolated per teacher.")
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
            password = st.text_input("Choose a password", type="password")
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
    api_key = os.getenv("GOOGLE_API_KEY") or st.text_input("Google API Key", type="password")

if not api_key:
    st.warning("Enter a Google API key to tag questions and generate papers.")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_chain(key: str, user_id: str, vector_db_directory: str, retrieval_version: str):
    return VectorRAGChain(api_key=key, persist_directory=vector_db_directory)


chain = get_chain(api_key, str(st.session_state.user["id"]), str(user_vector_db_dir), RETRIEVAL_VERSION)
syllabus_path = user_documents_dir / "_syllabus.pdf"


def syllabus_text() -> str:
    if not syllabus_path.is_file():
        return ""
    return "\n\n".join(page.page_content for page in load_pdf(syllabus_path))


def parse_bloom_targets(values: dict[str, int]) -> dict[str, int]:
    return {level: count for level, count in values.items() if count > 0}


def paper_constraints() -> tuple[dict, bool]:
    with st.form("constraints_form"):
        st.subheader("Paper constraints")
        unit_text = st.text_input("Required units/topics", placeholder="Unit 1, Unit 3")
        total_marks = st.number_input("Total marks", min_value=1, value=16, step=1)
        distribution = st.text_input(
            "Mark distribution",
            value="3 questions x 2 marks, 2 questions x 5 marks",
            help="Use entries such as: 3 questions x 2 marks, 2 questions x 5 marks.",
        )
        student_year = st.text_input("Student year", value="Year 10")
        st.caption("Minimum Bloom's targets")
        bloom_values = {
            level: st.number_input(level, min_value=0, value=0, step=1, key=f"bloom_{level}")
            for level in BLOOM_LEVELS
        }
        submitted = st.form_submit_button("Generate draft paper", type="primary")
    constraints = {
        "units": [unit.strip() for unit in unit_text.split(",") if unit.strip()],
        "total_marks": int(total_marks),
        "mark_distribution": distribution,
        "student_year": student_year,
        "bloom_targets": parse_bloom_targets(bloom_values),
    }
    return constraints, submitted


st.sidebar.header("Question bank")
syllabus_upload = st.sidebar.file_uploader("Syllabus PDF", type=["pdf"], key="syllabus_upload")
question_uploads = st.sidebar.file_uploader(
    "Previous question papers",
    type=["pdf"],
    accept_multiple_files=True,
    key="question_uploads",
)
if st.sidebar.button("Process question papers", type="primary"):
    if not question_uploads:
        st.sidebar.error("Upload at least one previous question paper.")
    else:
        with st.spinner("Extracting questions, tagging metadata, and embedding the bank..."):
            if syllabus_upload:
                syllabus_path.write_bytes(syllabus_upload.getbuffer())
            documents = []
            for uploaded_file in question_uploads:
                save_path = user_documents_dir / Path(uploaded_file.name).name
                save_path.write_bytes(uploaded_file.getbuffer())
                documents.extend(load_pdf(save_path))
            questions = chunk_documents(documents, syllabus_text=syllabus_text(), llm=chain.llm)
            added = embed_and_store(questions, str(user_vector_db_dir))
        get_chain.clear()
        st.sidebar.success(f"Indexed {added} new question(s).")

question_files = sorted(path for path in user_documents_dir.glob("*.pdf") if path.name != syllabus_path.name)
st.sidebar.caption(f"Indexed papers: {len(question_files)}")
if question_files:
    with st.sidebar.expander("Remove a question paper"):
        selected_file = st.selectbox("Paper", [path.name for path in question_files])
        confirmed = st.checkbox("I understand this removes its indexed questions.")
        if st.button("Remove paper", disabled=not confirmed):
            remove_document(user_documents_dir / selected_file, user_documents_dir, str(user_vector_db_dir))
            get_chain.clear()
            st.rerun()

constraints, generate_submitted = paper_constraints()
if generate_submitted:
    with st.spinner("Assembling and checking the paper..."):
        st.session_state.paper_constraints = constraints
        st.session_state.paper = chain.generate_paper(constraints)

paper = st.session_state.get("paper", [])
active_constraints = st.session_state.get("paper_constraints", constraints)
if not paper:
    st.info("Upload and process previous papers, then define a rubric to generate a draft.")
    st.stop()


def current_report() -> dict:
    return validate_paper(paper, active_constraints)


st.subheader("Draft paper")
for index, item in enumerate(paper):
    metadata = item.setdefault("metadata", {})
    source = "newly generated" if item.get("generated") else f"from {metadata.get('source_year', 'unknown')} paper"
    with st.container(border=True):
        st.markdown(f"**Question {index + 1}**")
        edited_text = st.text_area("Question text", value=item["text"], key=f"question_text_{index}")
        tags = st.columns(4)
        tags[0].caption(f"Topic\n{metadata.get('topic', 'Unclassified')}")
        tags[1].caption(f"Marks\n{metadata.get('mark_value', 0)}")
        tags[2].caption(f"Bloom\n{metadata.get('bloom_level', 'Understand')}")
        tags[3].caption(f"Source\n{source}")
        actions = st.columns(3)
        if actions[0].button("Save edit", key=f"edit_{index}"):
            item["text"] = edited_text.strip()
            st.rerun()
        if actions[1].button("Regenerate", key=f"regenerate_{index}"):
            replacement_constraints = {
                **active_constraints,
                "units": [metadata.get("topic", "")],
                "mark_distribution": f"1 question x {int(metadata.get('mark_value', 0))} marks",
                "bloom_targets": {metadata.get("bloom_level", "Understand"): 1},
            }
            replacement = chain.generate_paper(replacement_constraints)
            if replacement:
                paper[index] = replacement[0]
            st.rerun()
        if actions[2].button("Remove", key=f"remove_{index}"):
            paper.pop(index)
            st.rerun()

report = current_report()
st.subheader("Rubric compliance")
if report["valid"]:
    st.success("This draft satisfies the current rubric.")
else:
    st.warning(f"{len(report['violations'])} rubric issue(s) need review before export.")
    for violation in report["violations"]:
        st.error(violation["message"])
with st.expander("Validation details"):
    st.json(report)

approved = st.checkbox("I approve this reviewed paper for export.", disabled=not report["valid"])
export_lines = [f"EXAM PAPER - {active_constraints.get('student_year', '')}", ""]
for index, item in enumerate(paper, 1):
    metadata = item["metadata"]
    export_lines.extend([
        f"{index}. {item['text']}",
        f"[{metadata.get('mark_value', 0)} marks | {metadata.get('topic', 'Unclassified')} | {metadata.get('bloom_level', 'Understand')} ]",
        "",
    ])
st.download_button(
    "Export approved paper",
    "\n".join(export_lines),
    file_name="exam_paper.txt",
    mime="text/plain",
    disabled=not approved,
)
