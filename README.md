# Mini AI Knowledge Assistant

A Streamlit-based RAG application for asking grounded questions about uploaded PDF documents. Each person signs in with Supabase, receives isolated local document/vector storage, and has their chat history saved to their own Supabase account.

## What it does

- Creates accounts and signs users in with Supabase Auth.
- Lets each signed-in user upload one or more PDFs from the sidebar.
- Extracts PDF pages and creates semantic chunks (roughly 350-1,200 characters) using local MiniLM embeddings.
- Stores embeddings in a persistent, user-specific Chroma collection.
- Retrieves direct matches plus diverse MMR results, then asks Gemini to answer only from the retrieved context.
- Shows the source file, page, section, relevance score (when available), and excerpt for every answer.
- Persists questions, answers, and source metadata in Supabase so a user's history is restored after they sign in again.
- Allows users to remove an uploaded PDF and its corresponding indexed chunks.

## Architecture

| Component | Role |
| --- | --- |
| `app.py` | Streamlit UI, authentication flow, upload/remove controls, and chat display. |
| `database.py` | Supabase Auth plus per-user chat-message reads and writes. |
| `ingest.py` | PDF loading, semantic chunking, MiniLM embeddings, and persistent Chroma storage. |
| `rag_chain.py` | Vector retrieval, context construction, and Gemini answer generation. |
| `supabase_schema.sql` | `chat_messages` table, index, RLS, and per-user policies. |

Documents and Chroma data are stored locally in `data/users/<user-id>/` and `chroma_db/users/<user-id>/`. Chat history is stored remotely in Supabase.

## Prerequisites

- Python 3.10 or newer
- A [Supabase](https://supabase.com/) project
- A Google AI Studio API key for Gemini

## Setup

1. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

2. Create the Supabase database table.

   In the Supabase dashboard, open **SQL Editor**, paste the contents of `supabase_schema.sql`, and run it. This creates `chat_messages`, enables Row Level Security, and grants signed-in users access only to their own messages.

3. Configure Supabase Auth.

   In the Supabase dashboard, enable the Email provider under **Authentication -> Providers**. If email confirmation is enabled, new users must confirm their email before signing in.

4. Create a `.env` file from the example.

   ```bash
   cp .env.example .env
   ```

   On Windows PowerShell:

   ```powershell
   Copy-Item .env.example .env
   ```

   Set these values in `.env`:

   ```dotenv
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_PUBLISHABLE_KEY=your-publishable-key
   GOOGLE_API_KEY=your-google-ai-studio-key
   ```

   Use the project's **publishable** key (or legacy `anon` key), never a Supabase `service_role` or secret key.

   Alternatively, configure Supabase in Streamlit secrets using either root-level keys or the local connection layout:

   ```toml
   [connections.supabase]
   SUPABASE_URL = "https://your-project.supabase.co"
   SUPABASE_PUBLISHABLE_KEY = "your-publishable-key"
   ```

5. Start the app.

   ```bash
   streamlit run app.py
   ```

6. Create an account, sign in, add PDFs in the sidebar, select **Process documents**, and start asking questions. If `GOOGLE_API_KEY` is not in `.env`, enter it in the sidebar for the current session.

## Retrieval behavior

The app uses `sentence-transformers/all-MiniLM-L6-v2` locally for both indexing and querying. It keeps chunk IDs stable, so reprocessing the same content does not add duplicate vectors. For each question it retrieves up to 20 direct matches, retains the best four, then adds diverse results using maximal marginal relevance, up to eight source chunks total. Gemini (`gemini-3.6-flash`) generates the final response with a prompt that requires it to stay within that context.

## Notes and limitations

- Document vectors and PDFs remain on the machine running the app; they are not stored in Supabase.
- Initial local embedding-model download and indexing can take time, especially for large PDFs.
- The current ingestion work happens during the Streamlit request. Large-document background processing is not implemented yet.
- Keep `.env` and `.streamlit/secrets.toml` private; both are ignored by Git.

## Development checks

```bash
python -m py_compile app.py database.py ingest.py rag_chain.py
```

## AI-use disclosure

AI tools were used to assist with code scaffolding, architecture discussions, debugging, and documentation. The application code, configuration, and project behavior were reviewed and adapted for this repository.
