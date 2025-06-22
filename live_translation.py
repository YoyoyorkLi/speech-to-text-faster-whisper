import pyaudio
import numpy as np
import time
from faster_whisper import WhisperModel
from googletrans import Translator  # online translation
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#Define output language
TARGET_OUTPUT = input('Please enter en for English, ht for Haitian Creole: ')

# CONFIGS
#TARGET_OUTPUT = "en"  # set to "ht" for Haitian Creole
MODEL_SIZE = "small"   # small for middle ground between speed and accuracy
CHUNK_DURATION = 3   # seconds of audio per chunk (shorter = faster)
BEAM_SIZE = 2

# Initialize Faster-Whisper (run on int8 for speed)
model = WhisperModel(MODEL_SIZE, compute_type="int8", device="cpu")

# Initialize Translator
translator = Translator()

# Audio stream config
CHUNK = 1024
RATE = 14500
CHANNELS = 1
FORMAT = pyaudio.paInt16

# Setup PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                input=True, frames_per_buffer=CHUNK)

buffer = []
start_time = time.time()

print("🎙️ Listening (Spanish input only)...")

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        buffer.append(np.frombuffer(data, dtype=np.int16))

        if time.time() - start_time > CHUNK_DURATION:
            # Convert audio buffer to normalized float32 array
            audio_np = np.concatenate(buffer).astype(np.float32) / 32768.0

            # Transcribe (low latency mode: beam_size=1, fp16=False)
            segments, info = model.transcribe(audio_np, language="es", beam_size=BEAM_SIZE)

            for segment in segments:
                original_text = segment.text.strip()
                if not original_text:
                    continue

                print(f"\n📝 Transcript (ES): {original_text}")

                # Translate to English or Haitian Creole
                translated = translator.translate(original_text, src="es", dest=TARGET_OUTPUT)
                print(f"🌐 Translation ({TARGET_OUTPUT.upper()}): {translated.text}")

            # Reset buffer & timer
            buffer = []
            start_time = time.time()

except KeyboardInterrupt:
    print("🛑 Stopping...")
    stream.stop_stream()
    stream.close()
    p.terminate()

