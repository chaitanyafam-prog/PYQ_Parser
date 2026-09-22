"""Supabase Auth and per-user chat-memory client for the Streamlit app."""
from __future__ import annotations
import json
import os
from dotenv import load_dotenv
from supabase import create_client

try:
    import streamlit as st
except ImportError:
    st = None

load_dotenv()


def _get_config(*keys: str) -> str | None:
    """Return the first non-empty value found for any of the given key names,
    checking Streamlit Secrets first, then environment variables (.env locally).

    Streamlit Cloud accepts root-level secrets, while this project stores them
    locally under ``[connections.supabase]``. Support both layouts.
    """
    for key in keys:
        if st is not None:
            try:
                value = st.secrets.get(key)
                if value:
                    return value
            except Exception:
                pass
            try:
                connection = st.secrets.get("connections", {}).get("supabase", {})
                value = connection.get(key)
                if value:
                    return value
            except Exception:
                pass
        value = os.getenv(key)
        if value:
            return value
    return None


SUPABASE_URL = _get_config("SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = _get_config("SUPABASE_PUBLISHABLE_KEY", "NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY", "SUPABASE_KEY")

def is_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)

def _client(session: dict | None = None):
    if not is_configured():
        raise RuntimeError("Supabase is not configured. Set SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY.")
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    if session:
        client.auth.set_session(session["access_token"], session["refresh_token"])
    return client

def _user_session(response) -> dict:
    if response.user is None or response.session is None:
        raise RuntimeError("No active session was returned. Confirm the account email, then sign in.")
    return {"id": response.user.id, "email": response.user.email,
            "access_token": response.session.access_token, "refresh_token": response.session.refresh_token}

def create_user(email: str, password: str) -> tuple[bool, str | dict]:
    if len(password) < 8:
        return False, "Password must contain at least 8 characters."
    try:
        response = _client().auth.sign_up({"email": email.strip(), "password": password})
        if response.session is None:
            return False, "Check your email to confirm the account, then sign in."
        return True, _user_session(response)
    except Exception as exc:
        return False, str(exc)

def authenticate_user(email: str, password: str) -> tuple[bool, str | dict]:
    try:
        response = _client().auth.sign_in_with_password({"email": email.strip(), "password": password})
        return True, _user_session(response)
    except Exception:
        return False, "Incorrect email or password."

def save_message(user: dict, question: str, answer: str, sources: list[dict]):
    safe_sources = json.loads(json.dumps(sources, ensure_ascii=False, default=str))
    _client(user).table("chat_messages").insert({"user_id": user["id"], "question": question,
        "answer": answer, "sources": safe_sources}).execute()

def load_messages(user: dict) -> list[dict]:
    response = _client(user).table("chat_messages").select("question,answer,sources").order("created_at").execute()
    messages = []
    for row in response.data:
        try:
            sources = json.loads(row["sources"]) if isinstance(row["sources"], str) else row["sources"]
        except (TypeError, json.JSONDecodeError):
            sources = []
        messages.append({"question": row["question"], "answer": row["answer"], "sources": sources})
    return messages
