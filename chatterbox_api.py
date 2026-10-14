import os
import uuid
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio as ta
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from chatterbox.tts import ChatterboxTTS

app = FastAPI(
    title="Chatterbox TTS & Voice Cloning API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5176",
    "http://127.0.0.1:5176"
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "generated_audio"
VOICE_DIR = "voice_samples"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VOICE_DIR, exist_ok=True)


# --------------------------------------------------
# Load Chatterbox model once when the server starts
# --------------------------------------------------

print("Loading Chatterbox model...")
print("Using CPU...")

model = ChatterboxTTS.from_pretrained(device="cpu")

print("Chatterbox model loaded successfully!")


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "Chatterbox backend is running"
    }


# --------------------------------------------------
# Text-to-Speech + Voice Cloning
# --------------------------------------------------

@app.post("/tts")
async def generate_speech(
    text: str = Form(...),
    voice: UploadFile = File(...)
):
    try:
        if not text.strip():
            return {
                "error": "Text cannot be empty"
            }

        # Create unique filenames
        file_id = str(uuid.uuid4())

        voice_extension = os.path.splitext(voice.filename or ".wav")[1]

        voice_path = os.path.join(
            VOICE_DIR,
            f"{file_id}{voice_extension}"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{file_id}.wav"
        )

        # Save uploaded voice sample
        voice_data = await voice.read()

        with open(voice_path, "wb") as f:
            f.write(voice_data)

        print("--------------------------------")
        print("Generating speech...")
        print("Text:", text)
        print("Voice:", voice.filename)

        # Generate cloned speech
        wav = model.generate(
            text,
            audio_prompt_path=voice_path
        )

        # Save generated audio
        ta.save(
            output_path,
            wav,
            model.sr
        )

        print("Speech generated successfully!")
        print("Output:", output_path)
        print("--------------------------------")

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="generated_voice.wav"
        )

    except Exception as e:
        print("ERROR:", str(e))

        return {
            "error": str(e)
        }


# --------------------------------------------------
# Run server
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )