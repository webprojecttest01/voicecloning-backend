from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI()

# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Whisper Model
# =========================

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

# =========================
# STT Endpoint
# =========================

@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):

    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(await file.read())
        temp_path = temp_file.name

    try:

        segments, info = model.transcribe(temp_path)

        text = " ".join(
            segment.text.strip()
            for segment in segments
        )

        return {
            "success": True,
            "language": info.language,
            "text": text
        }

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/")
def home():
    return {
        "message": "STT API is running"
    }