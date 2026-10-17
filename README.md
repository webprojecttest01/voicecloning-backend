# Voice Cloning backend

## Deploy to Railway

1. Create a Railway project from the backend repository. If deploying from a
   monorepo, set the service root directory to `voicecloning-backend`; if this
   folder is the repository root, leave the root directory as `.`.
2. Railway uses `railway.toml` and `Dockerfile` to build and start the FastAPI
   service. Generate a public domain for the service.
3. Add a Railway PostgreSQL database and expose its `DATABASE_URL` to the
   backend service (for example, `${{Postgres.DATABASE_URL}}`). The backend
   creates its tables and applies the balance-column migration during startup.
   Production startup fails if the database URL is missing or the database
   cannot be initialized; `/health` also checks the live database connection.
4. Add a Railway volume mounted at `/data` for generated audio, uploads, and
   voice samples. These files are separate from the PostgreSQL database.
5. Set these backend service variables:
   - `FLASK_SECRET_KEY`: a long, randomly generated secret.
   - `ENVIRONMENT`: `production`.
   - `CORS_ORIGINS`: the exact Vercel production origin, such as
     `https://your-app.vercel.app` (no trailing slash). Add additional
     comma-separated origins if required.
   - `SESSION_COOKIE_SAMESITE`: `none`.
   - `SESSION_COOKIE_SECURE`: `true`.
   - `DATA_DIR`: `/data`.
   - Optional first-run account: set `DEFAULT_USER_EMAIL` and
     `DEFAULT_USER_PASSWORD` (at least 12 characters), and optionally
     `DEFAULT_USER_NAME`. The password is stored as a hash in PostgreSQL. The
     account is created only if that email does not already exist; changing
     these variables does not reset an existing password.
   - `ELEVENLABS_API_KEY`: required only for ElevenLabs TTS and voice listing.
6. In Vercel, set `VITE_API_URL` to the Railway public URL, then redeploy the
   frontend. Set the corresponding Vercel origin in `CORS_ORIGINS` on Railway
   and redeploy the backend if the frontend URL changes.

For local development, omit `DATABASE_URL` to use SQLite in `DATA_DIR`. Do not
put real credentials or API keys in `.env.example`; configure secrets through
your local `.env` or Railway variables.

The Whisper and Chatterbox models are downloaded when first used and need
significant memory, disk space, and startup time. Ensure the selected Railway
plan has sufficient resources. The health check verifies the API and database
connection but does not load these models.

For local development, install the requirements and run:

```sh
uvicorn app:app --reload --host 0.0.0.0 --port 5000
```
