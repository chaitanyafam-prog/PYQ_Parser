# PYQ Parser

A Streamlit app that turns a teacher's previous question papers into a searchable question bank and builds rubric-constrained draft exam papers. Each teacher signs in with Supabase and receives isolated local document and vector storage.

## What it does

- Creates accounts and signs users in with Supabase Auth.
- Lets each signed-in teacher upload a syllabus and previous question papers.
- Extracts individual questions and tags them with topic, marks, Bloom's level, source year, and source paper.
- Stores embeddings in a persistent, user-specific Chroma collection.
- Retrieves questions using mark, topic, and Bloom's metadata filters combined with semantic similarity.
- Generates new questions with Gemini when the bank lacks a suitable match, then rejects near-duplicates.
- Validates generated papers against marks, topic coverage, Bloom's targets, and duplicate checks.
- Supports per-question editing, regeneration, removal, rubric review, and approved plain-text export.
- Allows users to remove an uploaded paper and its corresponding indexed questions.

## Typical workflow

1. Create an account and sign in.
2. Upload an optional syllabus PDF and one or more previous question papers, then select **Process question papers**.
3. Enter the required topics, mark distribution, student year, and minimum Bloom's-level counts.
4. Generate a draft, review or edit individual questions, and resolve any rubric warnings.
5. Approve a valid draft and download it as a plain-text exam paper.

For a mark distribution, use comma-separated entries such as `3 questions x 2 marks, 2 questions x 5 marks`. The number of requested questions is determined by this distribution; `Total marks` is checked by the rubric validator.

## Architecture

| Component | Role |
| --- | --- |
| `app.py` | Streamlit UI, authentication, ingestion, paper generation, review, and export. |
| `database.py` | Supabase Auth and per-user configuration. |
| `ingest.py` | Question extraction, LLM tagging, embeddings, and persistent Chroma storage. |
| `rag_chain.py` | Metadata-filtered retrieval, few-shot question generation, and duplicate checks. |
| `rubric_checker.py` | Structured validation of draft papers against teacher constraints. |
| `supabase_schema.sql` | `chat_messages` table, index, RLS, and per-user policies. |

Documents and Chroma data are stored locally in `data/users/<user-id>/` and `chroma_db/users/<user-id>/`. Supabase provides authentication; the supplied schema also creates a per-user message table for integrations that use the database helpers.

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

6. Create an account, sign in, upload a syllabus and previous papers, select **Process question papers**, define the rubric, and generate a draft. If `GOOGLE_API_KEY` is not in `.env`, enter it in the sidebar for the current session.

## Retrieval behavior

The app uses `sentence-transformers/all-MiniLM-L6-v2` locally for indexing and querying. It keeps question IDs stable, so reprocessing the same content does not add duplicate vectors. For each requested paper slot it filters Chroma by marks, topic, and Bloom's level, then retrieves up to four semantically relevant examples. If no matching stored question is available, Gemini (`gemini-3.6-flash`) generates a new question from those examples. Generated questions are checked against both the draft and the stored questions in the relevant topic using a 0.92 embedding-similarity threshold.

## Notes and limitations

- Question vectors and PDFs remain on the machine running the app; they are not stored in Supabase.
- Initial local embedding-model download and indexing can take time, especially for large PDFs.
- The current ingestion work happens during the Streamlit request. Large-document background processing is not implemented yet.
- Keep `.env` and `.streamlit/secrets.toml` private; both are ignored by Git.

## Development checks

```bash
python -m py_compile app.py database.py ingest.py rag_chain.py rubric_checker.py
```

## AI-use disclosure

AI tools were used to assist with code scaffolding, architecture discussions, debugging, and documentation. The application code, configuration, and project behavior were reviewed and adapted for this repository.
