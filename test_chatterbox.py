import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

print("Loading Chatterbox model...")
print("Using CPU...")

model = ChatterboxTTS.from_pretrained(device="cpu")

print("Model loaded!")

text = "Hello, this is a test of my cloned voice using Chatterbox."

print("Generating speech...")

wav = model.generate(
    text,
    audio_prompt_path="myvoice.ogg"
)

ta.save(
    "chatterbox-test.wav",
    wav,
    model.sr
)

print("Done!")
print("Audio saved as chatterbox-test.wav")