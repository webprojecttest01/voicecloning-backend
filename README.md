# Voice Cloning backend

## Deploy to Railway

1. Create a Railway project from the backend repository. If deploying from a
   monorepo, set the service root directory to `voicecloning-backend`; if this
   folder is the repository root, leave the root directory as `.`.
2. Railway uses `railway.toml` and `Dockerfile` to build and start the FastAPI
   service. Generate a public domain for the service.
3. Add a Railway volume mounted at `/data`. The SQLite database, generated
   audio, uploads, and voice samples are stored there; without a volume,
   Railway's ephemeral filesystem will not retain them across deployments.
4. Set these service variables:
   - `FLASK_SECRET_KEY`: a long, randomly generated secret.
   - `ENVIRONMENT`: `production`.
   - `CORS_ORIGINS`: the exact Vercel production origin, such as
     `https://your-app.vercel.app` (no trailing slash). Add additional
     comma-separated origins if required.
   - `SESSION_COOKIE_SAMESITE`: `none`.
   - `SESSION_COOKIE_SECURE`: `true`.
   - `DATA_DIR`: `/data`.
   - `ELEVENLABS_API_KEY`: required only for ElevenLabs TTS and voice listing.
5. In Vercel, set `VITE_API_URL` to the Railway public URL, then redeploy the
   frontend. Set the corresponding Vercel origin in `CORS_ORIGINS` on Railway
   and redeploy the backend if the frontend URL changes.

The Whisper and Chatterbox models are downloaded when first used and need
significant memory, disk space, and startup time. Ensure the selected Railway
plan has sufficient resources. The app health check only verifies that the API
is running; it does not load these models.

For local development, install the requirements and run:

```sh
uvicorn app:app --reload --host 0.0.0.0 --port 5000
```
