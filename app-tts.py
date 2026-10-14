import os
import requests

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")



print("ElevenLabs API Key loaded:", bool(ELEVENLABS_API_KEY))



# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)

CORS(app)


# ==========================================
# ELEVENLABS
# ==========================================

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"

# ==========================================
# HEALTH CHECK
# ==========================================

@app.route("/", methods=["GET"])
def health_check():

    return jsonify({
        "message": "Voice AI Python backend is running"
    })


# ==========================================
# ELEVENLABS - GET VOICES
# ==========================================

@app.route("/api/voices", methods=["GET"])
def get_voices():

    if not ELEVENLABS_API_KEY:
        return jsonify({
            "error": "ELEVENLABS_API_KEY is missing"
        }), 500

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
            data = {
                "error": response.text
            }

        if not response.ok:
            return jsonify(data), response.status_code

        return jsonify(data)

    except requests.RequestException as error:

        print("ElevenLabs voices error:", error)

        return jsonify({
            "error": "Failed to connect to ElevenLabs"
        }), 500


# ==========================================
# ELEVENLABS - TEXT TO SPEECH
# ==========================================

@app.route("/api/tts", methods=["POST"])
def text_to_speech():

    if not ELEVENLABS_API_KEY:
        return jsonify({
            "error": "ELEVENLABS_API_KEY is missing"
        }), 500

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Request body is required"
            }), 400

        text = data.get("text")
        voice_id = data.get("voiceId")

        if not text or not voice_id:
            return jsonify({
                "error": "text and voiceId are required"
            }), 400

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

            print(
                "ElevenLabs TTS error:",
                response.status_code,
                response.text
            )

            return jsonify({
                "error": response.text
            }), response.status_code

        return Response(
            response.content,
            mimetype="audio/mpeg"
        )

    except requests.RequestException as error:

        print("ElevenLabs TTS request error:", error)

        return jsonify({
            "error": "Failed to connect to ElevenLabs"
        }), 500

    except Exception as error:

        print("Unexpected TTS error:", error)

        return jsonify({
            "error": "Text-to-speech failed"
        }), 500


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":

    print(
    "Python TTS server running on "
    "http://localhost:5001"
)

app.run(
    host="0.0.0.0",
    port=5001,
    debug=True
)