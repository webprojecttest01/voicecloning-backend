require("dotenv").config();

console.log(
  "API Key loaded:",
  !!process.env.ELEVENLABS_API_KEY
);

const express = require("express");
const cors = require("cors");
const multer = require("multer");

const upload = multer({
  storage: multer.memoryStorage(),
});

const app = express();

app.use(cors());
app.use(express.json());

// ==========================================
// TEXT-TO-SPEECH
// ==========================================

app.post("/api/tts", async (req, res) => {
  try {
    const { text, voiceId } = req.body;

    if (!text || !voiceId) {
      return res.status(400).json({
        error: "text and voiceId are required",
      });
    }

    const response = await fetch(
      `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`,
      {
        method: "POST",
        headers: {
          "xi-api-key": process.env.ELEVENLABS_API_KEY,
          "Content-Type": "application/json",
          Accept: "audio/mpeg",
        },
        body: JSON.stringify({
          text,
          model_id: "eleven_multilingual_v2",
        }),
      }
    );

    if (!response.ok) {
      const error = await response.text();

      return res.status(response.status).json({
        error,
      });
    }

    const audioBuffer = Buffer.from(
      await response.arrayBuffer()
    );

    res.set("Content-Type", "audio/mpeg");
    res.send(audioBuffer);
  } catch (error) {
    console.error(error);

    res.status(500).json({
      error: "Text-to-speech failed",
    });
  }
});

// ==========================================
// GET AVAILABLE ELEVENLABS VOICES
// ==========================================

app.get("/api/voices", async (req, res) => {
  try {
    const response = await fetch(
      "https://api.elevenlabs.io/v1/voices",
      {
        headers: {
          "xi-api-key":
            process.env.ELEVENLABS_API_KEY,
        },
      }
    );

    const data = await response.json();

    if (!response.ok) {
      return res.status(response.status).json(data);
    }

    res.json(data);
  } catch (error) {
    console.error(error);

    res.status(500).json({
      error: "Failed to fetch voices",
    });
  }
});

// ==========================================
// HEALTH CHECK
// ==========================================

app.get("/", (req, res) => {
  res.json({
    message: "Voice AI backend is running",
  });
});

// ==========================================
// START SERVER
// ==========================================

const PORT = 5001;

app.listen(PORT, () => {
  console.log(
    `Server running on http://localhost:${PORT}`
  );
});