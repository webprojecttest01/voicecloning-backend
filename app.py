import sys
import os

# Auto-detect chatterbox venv python if running with standard system python missing dependencies
_venv_python = os.path.join(os.path.dirname(__file__), "chatterbox-env", "Scripts", "python.exe")
if os.path.exists(_venv_python) and sys.executable.lower() != os.path.abspath(_venv_python).lower():
    try:
        import torchaudio
    except ImportError:
        print(f"Auto-switching to virtual environment Python: {_venv_python}")
        os.execv(_venv_python, [_venv_python] + sys.argv)

import sqlite3
import tempfile
import uuid
import requests
from dotenv import load_dotenv

from fastapi import FastAPI, Request, Response, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from starlette.middleware.sessions import SessionMiddleware
from werkzeug.security import generate_password_hash, check_password_hash


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

FLASK_SECRET_KEY = os.getenv(
    "FLASK_SECRET_KEY",
    "dev-secret-key"
)

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")


# ============================================================
# FOLDERS SETUP
# ============================================================

OUTPUT_DIR = "generated_audio"
VOICE_DIR = "voice_samples"
UPLOAD_DIR = "dubbing_uploads"
DUBBING_OUTPUT_DIR = "dubbing_output"
DUBBING_VOICE_DIR = "dubbing_voices"

for folder in [OUTPUT_DIR, VOICE_DIR, UPLOAD_DIR, DUBBING_OUTPUT_DIR, DUBBING_VOICE_DIR]:
    os.makedirs(folder, exist_ok=True)


# ============================================================
# DATABASE SETUP
# ============================================================

DATABASE = "users.db"


def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    conn = get_db_connection()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            text_snippet TEXT NOT NULL,
            char_count INTEGER NOT NULL,
            token_count INTEGER NOT NULL,
            model_used TEXT NOT NULL,
            cost_usd REAL NOT NULL,
            cost_inr REAL NOT NULL,
            status TEXT DEFAULT 'COMPLETED',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_balances (
            user_id INTEGER PRIMARY KEY,
            balance_tokens INTEGER DEFAULT 5553333,
            balance_usd REAL DEFAULT 49.98,
            balance_inr REAL DEFAULT 4173.24
        )
        """
    )

    try:
        cols = [c[1] for c in conn.execute("PRAGMA table_info(user_balances)").fetchall()]
        if "balance_tokens" not in cols:
            conn.execute("ALTER TABLE user_balances ADD COLUMN balance_tokens INTEGER DEFAULT 5553333")
            conn.execute("UPDATE user_balances SET balance_tokens = CAST(ROUND((balance_usd / 0.000009)) AS INTEGER)")
            conn.commit()
    except Exception as e:
        print("Database balance_tokens migration notice:", e)

    conn.commit()
    conn.close()

    print("Database initialized successfully with token balance support")


init_database()


# ============================================================
# LAZY MODEL LOADERS
# ============================================================

_whisper_model = None
_chatterbox_model = None
_chatterbox_multilingual_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print("--------------------------------")
        print("Loading Whisper model...")
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8"
        )
        print("Whisper model loaded successfully!")
        print("--------------------------------")
    return _whisper_model


def get_chatterbox_model():
    global _chatterbox_model
    if _chatterbox_model is None:
        print("--------------------------------")
        print("Loading Chatterbox model...")
        from chatterbox.tts import ChatterboxTTS
        _chatterbox_model = ChatterboxTTS.from_pretrained(device="cpu")
        print("Chatterbox model loaded successfully!")
        print("--------------------------------")
    return _chatterbox_model


def get_chatterbox_multilingual_model():
    global _chatterbox_multilingual_model
    if _chatterbox_multilingual_model is None:
        print("--------------------------------")
        print("Loading Chatterbox Multilingual model...")
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        _chatterbox_multilingual_model = ChatterboxMultilingualTTS.from_pretrained(
            device="cpu"
        )
        print("Chatterbox Multilingual model loaded successfully!")
        print("--------------------------------")
    return _chatterbox_multilingual_model


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


# ============================================================
# FASTAPI APP CREATION & MIDDLEWARE
# ============================================================

app = FastAPI(
    title="Unified Voice AI Backend API",
    version="1.0.0"
)

app.add_middleware(
    SessionMiddleware,
    secret_key=FLASK_SECRET_KEY,
    same_site="lax",
    https_only=False
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5176",
        "http://127.0.0.1:5176",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Unified Voice AI master backend is running",
        "supported_languages": SUPPORTED_LANGUAGES
    }


# ============================================================
# AUTHENTICATION ENDPOINTS
# ============================================================

@app.post("/api/auth/signup")
async def signup(request: Request):
    try:
        data = await request.json()

        if not data:
            return JSONResponse({"error": "Request body is required"}, status_code=400)

        name = data.get("name")
        email = data.get("email")
        password = data.get("password")

        if not name or not email or not password:
            return JSONResponse({"error": "Name, email and password are required"}, status_code=400)

        name = name.strip()
        email = email.strip().lower()

        if len(password) < 6:
            return JSONResponse({"error": "Password must be at least 6 characters"}, status_code=400)

        conn = get_db_connection()

        existing_user = conn.execute(
            "SELECT id FROM users WHERE email = ?",
            (email,)
        ).fetchone()

        if existing_user:
            conn.close()
            return JSONResponse({"error": "Email already registered"}, status_code=409)

        hashed_password = generate_password_hash(password)

        cursor = conn.execute(
            """
            INSERT INTO users (name, email, password)
            VALUES (?, ?, ?)
            """,
            (name, email, hashed_password)
        )

        conn.commit()
        user_id = cursor.lastrowid
        conn.close()

        request.session["user_id"] = user_id
        request.session["user_name"] = name
        request.session["user_email"] = email

        return JSONResponse({
            "success": True,
            "message": "Account created successfully",
            "user": {
                "id": user_id,
                "name": name,
                "email": email
            }
        }, status_code=201)

    except Exception as error:
        print("Signup error:", error)
        return JSONResponse({"error": "Signup failed"}, status_code=500)


@app.post("/api/auth/login")
async def login(request: Request):
    try:
        data = await request.json()

        if not data:
            return JSONResponse({"error": "Request body is required"}, status_code=400)

        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return JSONResponse({"error": "Email and password are required"}, status_code=400)

        email = email.strip().lower()

        conn = get_db_connection()

        user = conn.execute(
            """
            SELECT id, name, email, password
            FROM users
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

        conn.close()

        if not user or not check_password_hash(user["password"], password):
            return JSONResponse({"error": "Invalid email or password"}, status_code=401)

        request.session["user_id"] = user["id"]
        request.session["user_name"] = user["name"]
        request.session["user_email"] = user["email"]

        return JSONResponse({
            "success": True,
            "message": "Login successful",
            "user": {
                "id": user["id"],
                "name": user["name"],
                "email": user["email"]
            }
        }, status_code=200)

    except Exception as error:
        print("Login error:", error)
        return JSONResponse({"error": "Login failed"}, status_code=500)


@app.get("/api/auth/me")
def current_user(request: Request):
    if "user_id" not in request.session:
        return JSONResponse({"authenticated": False}, status_code=401)

    user_id = request.session["user_id"]
    b_tokens, b_usd, b_inr = get_user_balance_and_ensure(user_id)

    return JSONResponse({
        "authenticated": True,
        "user": {
            "id": user_id,
            "name": request.session["user_name"],
            "email": request.session["user_email"],
            "balance_tokens": b_tokens,
            "balance_usd": round(b_usd, 2),
            "balance_inr": round(b_inr, 2)
        }
    }, status_code=200)


@app.post("/api/auth/logout")
def logout(request: Request):
    request.session.clear()

    return JSONResponse({
        "success": True,
        "message": "Logged out successfully"
    }, status_code=200)


# ============================================================
# ELEVENLABS TTS ENDPOINTS
# ============================================================

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"


@app.get("/api/voices")
def get_voices():
    if not ELEVENLABS_API_KEY:
        return JSONResponse({"error": "ELEVENLABS_API_KEY is missing"}, status_code=500)

    try:
        response = requests.get(
            f"{ELEVENLABS_BASE_URL}/voices",
            headers={
                "xi-api-key": ELEVENLABS_API_KEY
            },
            timeout=30
        )

        try:
            data = response.json()
        except ValueError:
            data = {"error": response.text}

        if not response.ok:
            return JSONResponse(data, status_code=response.status_code)

        return data

    except requests.RequestException as error:
        print("ElevenLabs voices error:", error)
        return JSONResponse({"error": "Failed to connect to ElevenLabs"}, status_code=500)


@app.post("/api/tts")
async def text_to_speech(request: Request):
    if not ELEVENLABS_API_KEY:
        return JSONResponse({"error": "ELEVENLABS_API_KEY is missing"}, status_code=500)

    try:
        data = await request.json()

        if not data:
            return JSONResponse({"error": "Request body is required"}, status_code=400)

        text = data.get("text")
        voice_id = data.get("voiceId")

        if not text or not voice_id:
            return JSONResponse({"error": "text and voiceId are required"}, status_code=400)

        user_id = request.session.get("user_id", 1)
        calc = calculate_token_cost(text, "eleven_multilingual_v2")
        ok, err_data = check_sufficient_tokens(user_id, calc["token_count"])
        if not ok:
            return JSONResponse(err_data, status_code=402)

        response = requests.post(
            f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}",
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg"
            },
            json={
                "text": text,
                "model_id": "eleven_multilingual_v2"
            },
            timeout=120
        )

        if not response.ok:
            print("ElevenLabs TTS error:", response.status_code, response.text)
            return JSONResponse({"error": response.text}, status_code=response.status_code)

        # Log payment transaction automatically for TTS generation
        try:
            trans_id, new_tokens, new_usd, new_inr = record_transaction_in_db(
                user_id=user_id,
                text_snippet=f"Text to Speech: {text}",
                char_count=calc["char_count"],
                token_count=calc["token_count"],
                model_used="eleven_multilingual_v2",
                cost_usd=calc["cost_usd"],
                cost_inr=calc["cost_inr"]
            )
            headers = {
                "X-Token-Count": str(calc["token_count"]),
                "X-Balance-Tokens": str(new_tokens),
                "X-Cost-USD": str(calc["cost_usd"]),
                "X-Cost-INR": str(calc["cost_inr"]),
                "X-Transaction-ID": str(trans_id),
                "X-Char-Count": str(calc["char_count"])
            }
            return Response(content=response.content, media_type="audio/mpeg", headers=headers)
        except Exception as log_err:
            print("Failed to auto-log transaction:", log_err)
            return Response(content=response.content, media_type="audio/mpeg")

    except requests.RequestException as error:
        print("ElevenLabs TTS request error:", error)
        return JSONResponse({"error": "Failed to connect to ElevenLabs"}, status_code=500)
    except Exception as error:
        print("Unexpected TTS error:", error)
        return JSONResponse({"error": "Text-to-speech failed"}, status_code=500)


# ============================================================
# PAYMENT & TOKEN COST CALCULATION ENDPOINTS
# ============================================================

import math

# UNIFIED TOKEN PRICING CONFIGURATION
# $9.00 per 1 million tokens | $0.009 for 1,000 tokens (covers ~40 seconds of voice)
INR_PER_USD = 83.50
TOKEN_PRICE_PER_MILLION_USD = 9.00
TOKEN_PRICE_PER_MILLION_INR = round(TOKEN_PRICE_PER_MILLION_USD * INR_PER_USD, 2)  # 751.50
TOKEN_PRICE_PER_1K_USD = 0.009
TOKEN_PRICE_PER_1K_INR = round(TOKEN_PRICE_PER_1K_USD * INR_PER_USD, 4)           # 0.7515 (~0.75)
USD_PER_TOKEN = 0.000009
INR_PER_TOKEN = round(USD_PER_TOKEN * INR_PER_USD, 7)
VOICE_SECONDS_PER_1K_TOKENS = 40.0  # 40 seconds covers 1,000 tokens

MODEL_RATES = {
    "chatterbox_voice_cloning": {
        "name": "Chatterbox Voice Cloning",
        "usd_per_1k_tokens": TOKEN_PRICE_PER_1K_USD,
        "inr_per_1k_tokens": 0.75,
        "usd_per_1m_tokens": TOKEN_PRICE_PER_MILLION_USD,
        "inr_per_1m_tokens": TOKEN_PRICE_PER_MILLION_INR,
        "voice_coverage": "~40s voice per 1k tokens",
        "usd_per_1k_chars": TOKEN_PRICE_PER_1K_USD,
        "inr_per_1k_chars": 0.75
    },
    "eleven_multilingual_v2": {
        "name": "ElevenLabs Multilingual v2 (TTS)",
        "usd_per_1k_tokens": TOKEN_PRICE_PER_1K_USD,
        "inr_per_1k_tokens": 0.75,
        "usd_per_1m_tokens": TOKEN_PRICE_PER_MILLION_USD,
        "inr_per_1m_tokens": TOKEN_PRICE_PER_MILLION_INR,
        "voice_coverage": "~40s voice per 1k tokens",
        "usd_per_1k_chars": TOKEN_PRICE_PER_1K_USD,
        "inr_per_1k_chars": 0.75
    },
    "eleven_turbo_v2_5": {
        "name": "ElevenLabs Turbo v2.5 (Fast TTS)",
        "usd_per_1k_tokens": TOKEN_PRICE_PER_1K_USD,
        "inr_per_1k_tokens": 0.75,
        "usd_per_1m_tokens": TOKEN_PRICE_PER_MILLION_USD,
        "inr_per_1m_tokens": TOKEN_PRICE_PER_MILLION_INR,
        "voice_coverage": "~40s voice per 1k tokens",
        "usd_per_1k_chars": TOKEN_PRICE_PER_1K_USD,
        "inr_per_1k_chars": 0.75
    },
    "whisper_speech_to_text": {
        "name": "Whisper Speech-to-Text (STT)",
        "usd_per_1k_tokens": TOKEN_PRICE_PER_1K_USD,
        "inr_per_1k_tokens": 0.75,
        "usd_per_1m_tokens": TOKEN_PRICE_PER_MILLION_USD,
        "inr_per_1m_tokens": TOKEN_PRICE_PER_MILLION_INR,
        "voice_coverage": "~40s voice per 1k tokens",
        "usd_per_1k_chars": TOKEN_PRICE_PER_1K_USD,
        "inr_per_1k_chars": 0.75
    },
    "chatterbox_voice_dubbing": {
        "name": "Chatterbox Voice Dubbing",
        "usd_per_1k_tokens": TOKEN_PRICE_PER_1K_USD,
        "inr_per_1k_tokens": 0.75,
        "usd_per_1m_tokens": TOKEN_PRICE_PER_MILLION_USD,
        "inr_per_1m_tokens": TOKEN_PRICE_PER_MILLION_INR,
        "voice_coverage": "~40s voice per 1k tokens",
        "usd_per_1k_chars": TOKEN_PRICE_PER_1K_USD,
        "inr_per_1k_chars": 0.75
    },
    "chatterbox_local": {
        "name": "Chatterbox Local AI",
        "usd_per_1k_tokens": TOKEN_PRICE_PER_1K_USD,
        "inr_per_1k_tokens": 0.75,
        "usd_per_1m_tokens": TOKEN_PRICE_PER_MILLION_USD,
        "inr_per_1m_tokens": TOKEN_PRICE_PER_MILLION_INR,
        "voice_coverage": "~40s voice per 1k tokens",
        "usd_per_1k_chars": TOKEN_PRICE_PER_1K_USD,
        "inr_per_1k_chars": 0.75
    },
    "standard_tts": {
        "name": "Standard Speech Engine",
        "usd_per_1k_tokens": TOKEN_PRICE_PER_1K_USD,
        "inr_per_1k_tokens": 0.75,
        "usd_per_1m_tokens": TOKEN_PRICE_PER_MILLION_USD,
        "inr_per_1m_tokens": TOKEN_PRICE_PER_MILLION_INR,
        "voice_coverage": "~40s voice per 1k tokens",
        "usd_per_1k_chars": TOKEN_PRICE_PER_1K_USD,
        "inr_per_1k_chars": 0.75
    }
}


def calculate_token_cost(text: str, model_key: str = "eleven_multilingual_v2"):
    text_clean = text if text else ""
    char_count = len(text_clean)
    words = text_clean.split()
    word_count = len(words)
    # Token calculation: approx 1 token per 3.8 characters (subword tokenization)
    token_count = math.ceil(char_count / 3.8) if char_count > 0 else 0

    rate_info = MODEL_RATES.get(model_key, MODEL_RATES["eleven_multilingual_v2"])

    # Pricing formula: $9.00 per 1M tokens ($0.009 for 1,000 tokens)
    cost_usd = (token_count / 1_000_000.0) * TOKEN_PRICE_PER_MILLION_USD
    cost_inr = cost_usd * INR_PER_USD

    cost_usd = round(cost_usd, 6)
    cost_inr = round(cost_inr, 4)

    # Voice duration estimation: 1,000 tokens covers ~40 seconds of voice
    est_voice_seconds = round((token_count / 1000.0) * VOICE_SECONDS_PER_1K_TOKENS, 2)

    return {
        "char_count": char_count,
        "word_count": word_count,
        "token_count": token_count,
        "model_key": model_key,
        "model_name": rate_info["name"],
        "cost_usd": cost_usd,
        "cost_inr": cost_inr,
        "usd_per_1k_chars": TOKEN_PRICE_PER_1K_USD,
        "inr_per_1k_chars": 0.75,
        "usd_per_1k_tokens": TOKEN_PRICE_PER_1K_USD,
        "inr_per_1k_tokens": 0.75,
        "usd_per_1m_tokens": TOKEN_PRICE_PER_MILLION_USD,
        "inr_per_1m_tokens": TOKEN_PRICE_PER_MILLION_INR,
        "voice_coverage_sec": est_voice_seconds,
        "estimated_time_sec": max(0.5, est_voice_seconds)
    }


def get_user_balance_and_ensure(user_id: int):
    conn = get_db_connection()
    row = conn.execute("SELECT balance_tokens, balance_usd, balance_inr FROM user_balances WHERE user_id = ?", (user_id,)).fetchone()
    if not row:
        default_tokens = 5553333
        default_usd = 49.98
        default_inr = 4173.24
        conn.execute(
            "INSERT INTO user_balances (user_id, balance_tokens, balance_usd, balance_inr) VALUES (?, ?, ?, ?)",
            (user_id, default_tokens, default_usd, default_inr)
        )
        conn.commit()
        balance_tokens = default_tokens
        balance_usd = default_usd
        balance_inr = default_inr
    else:
        balance_tokens = row["balance_tokens"]
        balance_usd = row["balance_usd"]
        balance_inr = row["balance_inr"]
        if balance_tokens is None:
            balance_tokens = int(round((balance_usd or 0) / USD_PER_TOKEN))
            conn.execute("UPDATE user_balances SET balance_tokens = ? WHERE user_id = ?", (balance_tokens, user_id))
            conn.commit()
    conn.close()
    return int(balance_tokens), float(balance_usd), float(balance_inr)


def check_sufficient_tokens(user_id: int, required_tokens: int):
    b_tokens, b_usd, b_inr = get_user_balance_and_ensure(user_id)
    if b_tokens < required_tokens:
        voice_sec = round((required_tokens / 1000.0) * VOICE_SECONDS_PER_1K_TOKENS, 1)
        return False, {
            "error": f"Insufficient token balance! You have {b_tokens:,} tokens remaining, but {required_tokens:,} tokens are required (covers ~{voice_sec}s of voice). Please top up your wallet with tokens to continue.",
            "needs_topup": True,
            "available_tokens": b_tokens,
            "required_tokens": required_tokens,
            "balance_usd": round(b_usd, 2),
            "balance_inr": round(b_inr, 2)
        }
    return True, None


def record_transaction_in_db(user_id: int, text_snippet: str, char_count: int, token_count: int, model_used: str, cost_usd: float, cost_inr: float):
    conn = get_db_connection()
    b_tokens, b_usd, b_inr = get_user_balance_and_ensure(user_id)
    new_tokens = max(0, b_tokens - token_count)
    new_usd = round(new_tokens * USD_PER_TOKEN, 2)
    new_inr = round(new_usd * INR_PER_USD, 2)

    conn.execute(
        "UPDATE user_balances SET balance_tokens = ?, balance_usd = ?, balance_inr = ? WHERE user_id = ?",
        (new_tokens, new_usd, new_inr, user_id)
    )

    cursor = conn.execute(
        """
        INSERT INTO transactions (user_id, text_snippet, char_count, token_count, model_used, cost_usd, cost_inr, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'COMPLETED')
        """,
        (user_id, text_snippet[:120], char_count, token_count, model_used, cost_usd, cost_inr)
    )
    conn.commit()
    trans_id = cursor.lastrowid
    conn.close()
    return trans_id, new_tokens, new_usd, new_inr


@app.post("/api/payments/calculate")
async def api_calculate_payment(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        model_key = data.get("model", "eleven_multilingual_v2")
        res = calculate_token_cost(text, model_key)
        return JSONResponse(res)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/payments/history")
def api_payment_history(request: Request):
    try:
        user_id = request.session.get("user_id", 1)
        b_tokens, b_usd, b_inr = get_user_balance_and_ensure(user_id)

        conn = get_db_connection()
        rows = conn.execute(
            """
            SELECT id, text_snippet, char_count, token_count, model_used, cost_usd, cost_inr, status, created_at
            FROM transactions
            WHERE user_id = ? OR user_id IS NULL
            ORDER BY id DESC
            LIMIT 50
            """,
            (user_id,)
        ).fetchall()

        conn.close()

        transactions = [dict(r) for r in rows]

        # Calculate totals
        total_tokens = sum(t["token_count"] for t in transactions if t.get("status") != "TOPUP")
        total_cost_usd = round(sum(t["cost_usd"] for t in transactions if t.get("status") != "TOPUP"), 4)
        total_cost_inr = round(sum(t["cost_inr"] for t in transactions if t.get("status") != "TOPUP"), 2)
        total_conversions = len([t for t in transactions if t.get("status") != "TOPUP"])

        return JSONResponse({
            "balance_tokens": b_tokens,
            "balance_usd": round(b_usd, 2),
            "balance_inr": round(b_inr, 2),
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost_usd,
            "total_cost_inr": total_cost_inr,
            "total_conversions": total_conversions,
            "token_pricing": {
                "usd_per_1m": TOKEN_PRICE_PER_MILLION_USD,
                "inr_per_1m": TOKEN_PRICE_PER_MILLION_INR,
                "usd_per_1k": TOKEN_PRICE_PER_1K_USD,
                "inr_per_1k": 0.75,
                "voice_coverage_sec_per_1k": VOICE_SECONDS_PER_1K_TOKENS
            },
            "model_rates": MODEL_RATES,
            "transactions": transactions
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/payments/record")
async def api_record_payment(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        model_key = data.get("model", "eleven_multilingual_v2")

        calc = calculate_token_cost(text, model_key)
        user_id = request.session.get("user_id", 1)

        ok, err_data = check_sufficient_tokens(user_id, calc["token_count"])
        if not ok:
            return JSONResponse(err_data, status_code=402)

        trans_id, new_tokens, new_usd, new_inr = record_transaction_in_db(
            user_id=user_id,
            text_snippet=text,
            char_count=calc["char_count"],
            token_count=calc["token_count"],
            model_used=model_key,
            cost_usd=calc["cost_usd"],
            cost_inr=calc["cost_inr"]
        )

        return JSONResponse({
            "success": True,
            "transaction_id": trans_id,
            "calculation": calc,
            "new_balance_tokens": new_tokens,
            "new_balance_usd": round(new_usd, 2),
            "new_balance_inr": round(new_inr, 2)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
# WALLET TOP-UP ENDPOINT (BUY TOKENS WITH MONEY)
# ============================================================

@app.post("/api/payments/topup")
async def api_topup_wallet(request: Request):
    """Add tokens to the user's wallet balance using money (USD and INR)."""
    try:
        data = await request.json()
        amount_usd = float(data.get("amount_usd", 0))
        tokens_requested = data.get("tokens")

        if tokens_requested:
            tokens_to_add = int(tokens_requested)
            amount_usd = round((tokens_to_add / 1_000_000.0) * TOKEN_PRICE_PER_MILLION_USD, 2)
        else:
            if amount_usd <= 0 or amount_usd > 10000:
                return JSONResponse({"error": "Invalid top-up amount. Must be between $0.01 and $10,000."}, status_code=400)
            tokens_to_add = int(round((amount_usd / TOKEN_PRICE_PER_MILLION_USD) * 1_000_000))

        amount_inr = round(amount_usd * INR_PER_USD, 2)

        user_id = request.session.get("user_id", 1)
        conn = get_db_connection()

        b_tokens, b_usd, b_inr = get_user_balance_and_ensure(user_id)

        new_tokens = b_tokens + tokens_to_add
        new_usd = round(new_tokens * USD_PER_TOKEN, 2)
        new_inr = round(new_usd * INR_PER_USD, 2)

        conn.execute(
            "UPDATE user_balances SET balance_tokens = ?, balance_usd = ?, balance_inr = ? WHERE user_id = ?",
            (new_tokens, new_usd, new_inr, user_id)
        )

        # Log as special TOPUP transaction
        cursor = conn.execute(
            """
            INSERT INTO transactions (user_id, text_snippet, char_count, token_count, model_used, cost_usd, cost_inr, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'TOPUP')
            """,
            (user_id, f"Wallet Top-Up: +{tokens_to_add:,} Tokens (+${amount_usd:.2f} [INR {amount_inr:.2f}])", 0, tokens_to_add, "wallet_topup", -amount_usd, -amount_inr)
        )

        conn.commit()
        trans_id = cursor.lastrowid
        conn.close()

        print("TOPUP: user={} +{} tokens +${} (INR {}) new_balance={} tokens".format(user_id, tokens_to_add, amount_usd, amount_inr, new_tokens))

        return JSONResponse({
            "success": True,
            "transaction_id": trans_id,
            "added_tokens": tokens_to_add,
            "added_usd": amount_usd,
            "added_inr": amount_inr,
            "new_balance_tokens": new_tokens,
            "new_balance_usd": new_usd,
            "new_balance_inr": new_inr
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": "Top-up failed: " + str(e)}, status_code=500)


# ============================================================
# CHATTERBOX VOICE CLONING ENDPOINT
# ============================================================

@app.post("/tts")
async def generate_speech(
    request: Request,
    text: str = Form(...),
    voice: UploadFile = File(...)
):
    try:
        if not text.strip():
            return JSONResponse({"error": "Text cannot be empty"}, status_code=400)

        user_id = request.session.get("user_id", 1)
        calc = calculate_token_cost(text, "chatterbox_voice_cloning")
        ok, err_data = check_sufficient_tokens(user_id, calc["token_count"])
        if not ok:
            return JSONResponse(err_data, status_code=402)

        file_id = str(uuid.uuid4())
        voice_extension = os.path.splitext(voice.filename or ".wav")[1]

        voice_path = os.path.join(VOICE_DIR, f"{file_id}{voice_extension}")
        output_path = os.path.join(OUTPUT_DIR, f"{file_id}.wav")

        voice_data = await voice.read()
        with open(voice_path, "wb") as f:
            f.write(voice_data)

        print("--------------------------------")
        print("Generating Chatterbox cloned speech...")
        print("Text:", text)
        print("Voice:", voice.filename)

        model = get_chatterbox_model()
        import torchaudio as ta

        wav = model.generate(
            text,
            audio_prompt_path=voice_path
        )

        ta.save(
            output_path,
            wav,
            model.sr
        )

        print("Speech generated successfully!")
        print("Output:", output_path)
        print("--------------------------------")

        trans_id = 0
        new_tokens = 0
        try:
            trans_id, new_tokens, new_usd, new_inr = record_transaction_in_db(
                user_id=user_id,
                text_snippet=f"Voice Cloning: {text}",
                char_count=calc["char_count"],
                token_count=calc["token_count"],
                model_used="chatterbox_voice_cloning",
                cost_usd=calc["cost_usd"],
                cost_inr=calc["cost_inr"]
            )
        except Exception as log_err:
            print("Failed to auto-log Voice Cloning transaction:", log_err)

        headers = {
            "X-Token-Count": str(calc["token_count"]),
            "X-Balance-Tokens": str(new_tokens),
            "X-Cost-USD": str(calc["cost_usd"]),
            "X-Cost-INR": str(calc["cost_inr"]),
            "X-Transaction-ID": str(trans_id),
            "X-Char-Count": str(calc["char_count"])
        }

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="generated_voice.wav",
            headers=headers
        )

    except Exception as e:
        print("ERROR:", str(e))
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
# SPEECH TO TEXT (STT - WHISPER) ENDPOINT
# ============================================================

@app.post("/stt")
async def speech_to_text(request: Request, file: UploadFile = File(...)):
    temp_path = None

    try:
        # ── Normalize file extension ─────────────────────────────────────
        # Files like "audio.mp3.mpeg" → use .mp3; unknown/empty → .wav
        filename = file.filename or "audio.wav"
        # Walk backwards through dots to find a known audio extension
        KNOWN_AUDIO_EXT = {".mp3", ".wav", ".ogg", ".flac", ".m4a",
                           ".aac", ".webm", ".opus", ".mp4", ".mpeg",
                           ".mpga", ".wma", ".aiff"}

        parts = filename.split(".")
        suffix = ".wav"  # default
        for part in reversed(parts[1:]):          # skip the basename
            candidate = "." + part.lower()
            if candidate in KNOWN_AUDIO_EXT:
                # Prefer actual decodable formats; map .mpeg → .mp3
                suffix = ".mp3" if candidate == ".mpeg" else candidate
                break

        print(f"STT: incoming file='{filename}'  resolved suffix='{suffix}'")

        user_id = request.session.get("user_id", 1)
        ok, err_data = check_sufficient_tokens(user_id, 25)
        if not ok:
            return JSONResponse(err_data, status_code=402)

        file_data = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_data)
            temp_path = tmp.name

        print(f"STT: temp file written -> {temp_path}  size={len(file_data)} bytes")

        whisper_model = get_whisper_model()
        segments, info = whisper_model.transcribe(temp_path)

        # Force the generator to evaluate fully so errors surface here
        text = " ".join(seg.text.strip() for seg in list(segments))

        print(f"STT: transcription done  lang={info.language}  chars={len(text)}")

        trans_id = 0
        new_tokens = 0
        calc = calculate_token_cost(text, "whisper_speech_to_text")
        try:
            trans_id, new_tokens, new_usd, new_inr = record_transaction_in_db(
                user_id=user_id,
                text_snippet=f"Speech to Text: {text}",
                char_count=calc["char_count"],
                token_count=calc["token_count"],
                model_used="whisper_speech_to_text",
                cost_usd=calc["cost_usd"],
                cost_inr=calc["cost_inr"]
            )
        except Exception as log_err:
            print("Failed to auto-log STT transaction:", log_err)

        return {
            "success": True,
            "language": info.language,
            "text": text,
            "token_count": calc["token_count"],
            "balance_tokens": new_tokens,
            "cost_usd": calc["cost_usd"],
            "cost_inr": calc["cost_inr"],
            "transaction_id": trans_id,
            "char_count": calc["char_count"]
        }

    except Exception as e:
        import traceback
        print("STT Error:", str(e))
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# MULTILINGUAL VOICE DUBBING ENDPOINTS
# ============================================================

@app.get("/languages")
def get_languages():
    return {
        "languages": SUPPORTED_LANGUAGES
    }


@app.post("/dub")
async def voice_dubbing(
    request: Request,
    audio: UploadFile = File(...),
    voice: UploadFile = File(...),
    target_language: str = Form(...)
):
    file_id = str(uuid.uuid4())

    audio_path = None
    voice_path = None
    output_path = None

    try:
        target_language = target_language.lower().strip()

        if target_language not in SUPPORTED_LANGUAGES:
            return JSONResponse({
                "success": False,
                "error": f"Unsupported target language: {target_language}"
            }, status_code=400)

        if not audio.filename:
            return JSONResponse({
                "success": False,
                "error": "Source audio file is required"
            }, status_code=400)

        if not voice.filename:
            return JSONResponse({
                "success": False,
                "error": "Reference voice file is required"
            }, status_code=400)

        audio_extension = os.path.splitext(audio.filename)[1] or ".wav"
        audio_path = os.path.join(UPLOAD_DIR, f"{file_id}{audio_extension}")
        audio_data = await audio.read()
        with open(audio_path, "wb") as file:
            file.write(audio_data)

        voice_extension = os.path.splitext(voice.filename)[1] or ".wav"
        voice_path = os.path.join(DUBBING_VOICE_DIR, f"{file_id}{voice_extension}")
        voice_data = await voice.read()
        with open(voice_path, "wb") as file:
            file.write(voice_data)

        print("--------------------------------")
        print("MULTILINGUAL VOICE DUBBING")
        print("Source audio:", audio.filename)
        print("Reference voice:", voice.filename)
        print("Target language:", SUPPORTED_LANGUAGES[target_language])

        # Step 1: Transcribe audio
        whisper_model = get_whisper_model()
        segments, info = whisper_model.transcribe(audio_path)
        transcript = " ".join(segment.text.strip() for segment in segments)
        source_language = info.language.lower()

        if not transcript.strip():
            return JSONResponse({
                "success": False,
                "error": "No speech detected in source audio"
            }, status_code=400)

        # Step 2: Translate text
        if source_language == target_language:
            translated_text = transcript
        else:
            import argostranslate.translate
            translated_text = argostranslate.translate.translate(
                transcript,
                source_language,
                target_language
            )

        if not translated_text.strip():
            return JSONResponse({
                "success": False,
                "error": "Translation returned empty text"
            }, status_code=400)

        user_id = request.session.get("user_id", 1)
        calc = calculate_token_cost(translated_text, "chatterbox_voice_dubbing")
        ok, err_data = check_sufficient_tokens(user_id, calc["token_count"])
        if not ok:
            return JSONResponse(err_data, status_code=402)

        # Step 3: Synthesize dubbed speech
        chatterbox_multilingual_model = get_chatterbox_multilingual_model()
        import torchaudio as ta

        wav = chatterbox_multilingual_model.generate(
            translated_text,
            language_id=target_language,
            audio_prompt_path=voice_path
        )

        output_path = os.path.join(
            DUBBING_OUTPUT_DIR,
            f"dubbed_{target_language}_{file_id}.wav"
        )

        ta.save(
            output_path,
            wav,
            chatterbox_multilingual_model.sr
        )

        print("VOICE DUBBING COMPLETED:", output_path)

        trans_id = 0
        new_tokens = 0
        try:
            lang_name = SUPPORTED_LANGUAGES.get(target_language, target_language)
            trans_id, new_tokens, new_usd, new_inr = record_transaction_in_db(
                user_id=user_id,
                text_snippet=f"Voice Dubbing ({lang_name}): {translated_text}",
                char_count=calc["char_count"],
                token_count=calc["token_count"],
                model_used="chatterbox_voice_dubbing",
                cost_usd=calc["cost_usd"],
                cost_inr=calc["cost_inr"]
            )
        except Exception as log_err:
            print("Failed to auto-log Dubbing transaction:", log_err)

        headers = {
            "X-Token-Count": str(calc["token_count"]),
            "X-Balance-Tokens": str(new_tokens),
            "X-Cost-USD": str(calc["cost_usd"]),
            "X-Cost-INR": str(calc["cost_inr"]),
            "X-Transaction-ID": str(trans_id),
            "X-Char-Count": str(calc["char_count"])
        }

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="dubbed_voice.wav",
            headers=headers
        )

    except Exception as error:
        print("DUBBING ERROR:", str(error))
        return JSONResponse({
            "success": False,
            "error": str(error)
        }, status_code=500)


# ============================================================
# START UNIFIED SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("--------------------------------------------------")
    print("Unified Voice AI Master Backend server running on http://localhost:5000")
    print("--------------------------------------------------")
    uvicorn.run(app, host="0.0.0.0", port=5000)