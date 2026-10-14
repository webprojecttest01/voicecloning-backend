import os
import uuid
import torchaudio as ta

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from faster_whisper import WhisperModel
import argostranslate.translate
from chatterbox.mtl_tts import ChatterboxMultilingualTTS


app = FastAPI(
    title="Multilingual Voice Dubbing API",
    version="1.0.0"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5176",
        "http://127.0.0.1:5176",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Folders
# --------------------------------------------------

UPLOAD_DIR = "dubbing_uploads"
OUTPUT_DIR = "dubbing_output"
VOICE_DIR = "dubbing_voices"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VOICE_DIR, exist_ok=True)


# --------------------------------------------------
# Supported Chatterbox languages
# --------------------------------------------------

SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}


# --------------------------------------------------
# Load Whisper
# --------------------------------------------------

print("--------------------------------")
print("Loading Whisper model...")

whisper_model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

print("Whisper model loaded successfully!")


# --------------------------------------------------
# Load Chatterbox Multilingual
# --------------------------------------------------

print("--------------------------------")
print("Loading Chatterbox Multilingual model...")
print("Using CPU...")

chatterbox_model = ChatterboxMultilingualTTS.from_pretrained(
    device="cpu"
)

print("Chatterbox Multilingual model loaded successfully!")
print("--------------------------------")


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "message": "Multilingual Voice Dubbing API is running",
        "supported_languages": SUPPORTED_LANGUAGES
    }


# --------------------------------------------------
# Languages
# --------------------------------------------------

@app.get("/languages")
def get_languages():

    return {
        "languages": SUPPORTED_LANGUAGES
    }


# --------------------------------------------------
# Voice Dubbing
# --------------------------------------------------

@app.post("/dub")
async def voice_dubbing(
    audio: UploadFile = File(...),
    voice: UploadFile = File(...),
    target_language: str = Form(...)
):

    file_id = str(uuid.uuid4())

    audio_path = None
    voice_path = None
    output_path = None

    try:

        # ------------------------------------------
        # Validate target language
        # ------------------------------------------

        target_language = target_language.lower().strip()

        if target_language not in SUPPORTED_LANGUAGES:

            return {
                "success": False,
                "error": (
                    f"Unsupported target language: "
                    f"{target_language}"
                )
            }


        # ------------------------------------------
        # Validate files
        # ------------------------------------------

        if not audio.filename:

            return {
                "success": False,
                "error": "Source audio file is required"
            }

        if not voice.filename:

            return {
                "success": False,
                "error": "Reference voice file is required"
            }


        # ------------------------------------------
        # Save source audio
        # ------------------------------------------

        audio_extension = (
            os.path.splitext(audio.filename)[1]
            or ".wav"
        )

        audio_path = os.path.join(
            UPLOAD_DIR,
            f"{file_id}{audio_extension}"
        )

        audio_data = await audio.read()

        with open(audio_path, "wb") as file:
            file.write(audio_data)


        # ------------------------------------------
        # Save reference voice
        # ------------------------------------------

        voice_extension = (
            os.path.splitext(voice.filename)[1]
            or ".wav"
        )

        voice_path = os.path.join(
            VOICE_DIR,
            f"{file_id}{voice_extension}"
        )

        voice_data = await voice.read()

        with open(voice_path, "wb") as file:
            file.write(voice_data)


        print("--------------------------------")
        print("MULTILINGUAL VOICE DUBBING")
        print("--------------------------------")

        print("Source audio:", audio.filename)
        print("Reference voice:", voice.filename)
        print(
            "Target language:",
            SUPPORTED_LANGUAGES[target_language]
        )


        # ==========================================
        # STEP 1: SPEECH → TEXT
        # ==========================================

        print("--------------------------------")
        print("STEP 1: Transcribing audio...")

        segments, info = whisper_model.transcribe(
            audio_path
        )

        transcript = " ".join(
            segment.text.strip()
            for segment in segments
        )

        source_language = info.language.lower()

        print("Detected language:", source_language)
        print("Original text:", transcript)


        if not transcript.strip():

            return {
                "success": False,
                "error": "No speech detected in source audio"
            }


        # ==========================================
        # STEP 2: LOCAL TRANSLATION
        # ==========================================

        print("--------------------------------")
        print("STEP 2: Local translation...")

        if source_language == target_language:

            translated_text = transcript

            print(
                "Source and target languages are the same."
            )

            print("Translation skipped.")

        else:

            print(
                f"Translating: "
                f"{source_language} → "
                f"{target_language}"
            )

            translated_text = (
                argostranslate.translate.translate(
                    transcript,
                    source_language,
                    target_language
                )
            )


        print("Translated text:", translated_text)


        if not translated_text.strip():

            return {
                "success": False,
                "error": "Translation returned empty text"
            }


        # ==========================================
        # STEP 3: TEXT → TARGET LANGUAGE SPEECH
        # ==========================================

        print("--------------------------------")
        print("STEP 3: Generating dubbed speech...")

        wav = chatterbox_model.generate(
            translated_text,
            language_id=target_language,
            audio_prompt_path=voice_path
        )


        # ==========================================
        # STEP 4: SAVE OUTPUT
        # ==========================================

        output_path = os.path.join(
            OUTPUT_DIR,
            f"dubbed_{target_language}_{file_id}.wav"
        )

        ta.save(
            output_path,
            wav,
            chatterbox_model.sr
        )


        print("--------------------------------")
        print("VOICE DUBBING COMPLETED")
        print("--------------------------------")

        print("Source language:", source_language)
        print("Target language:", target_language)
        print("Original text:", transcript)
        print("Translated text:", translated_text)
        print("Output:", output_path)

        print("--------------------------------")


        # ==========================================
        # RETURN AUDIO
        # ==========================================

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="dubbed_voice.wav"
        )


    except Exception as error:

        print("--------------------------------")
        print("DUBBING ERROR")
        print("--------------------------------")
        print(str(error))
        print("--------------------------------")

        return {
            "success": False,
            "error": str(error)
        }


# --------------------------------------------------
# Run server
# --------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8002
    )