import pyaudio
import numpy as np
from faster_whisper import WhisperModel
from googletrans import Translator
import os
import time
import threading

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Define output language
# Spanish input, English or Haitian Creole output
print('This program takes in Spanish audio input and outputs its Spanish transcript and English or Haitian Creole translation.')
TARGET_OUTPUT = input('Please enter en for English, ht for Haitian Creole: ').lower()
while TARGET_OUTPUT not in ['en', 'ht']:
    TARGET_OUTPUT = input('Invalid input. Please enter en for English, ht for Haitian Creole: ').lower()

# CONFIGS
MODEL_SIZE = "small" #Small model side for faster processing
BEAM_SIZE = 2
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = pyaudio.paInt16 # This is correct for 16-bit audio
CHUNK = 1024

# Run on int8 and cpu (optimized fot laptop)
model = WhisperModel(MODEL_SIZE, compute_type="int8", device="cpu")

# Initialize Translator
translator = Translator()

def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=AUDIO_FORMAT,
                    channels=AUDIO_CHANNELS,
                    rate=AUDIO_SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording... Press Enter to stop.")
    audio_frames = []

    stop_recording = False

    def wait_for_enter():
        nonlocal stop_recording
        input()
        stop_recording = True

    # Start thread to listen for Enter key
    threading.Thread(target=wait_for_enter, daemon=True).start()

    while not stop_recording:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_frames.append(data)

    print("Recording stopped.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Combine frames and convert to numpy array
    # **THIS IS THE CRITICAL CHANGE**
    # Interpret the bytes as int16 directly
    combined_audio_data = np.concatenate([np.frombuffer(frame, dtype=np.int16) for frame in audio_frames])
    return combined_audio_data

def transcribe_and_translate(audio_data, target_language):
    if not audio_data.any():
        print("No audio data to process.")
        return

    print("Transcribing...")
    # Now that audio_data is np.int16, this normalization is correct
    audio_float32 = audio_data.astype(np.float32) / 32768.0
    full_transcript = ""
    try:
        segments, info = model.transcribe(audio_float32, language="es", beam_size=BEAM_SIZE)
        for segment in segments:
            full_transcript += segment.text.strip() + " "
        full_transcript = full_transcript.strip()
    except Exception as e:
        print(f"Transcription error: {e}")
        return

    if full_transcript:
        print(f"Transcript (ES): {full_transcript}")
        print("Translating...")
        try:
            translated = translator.translate(full_transcript, src="es", dest=target_language)
            print(f"Translation ({target_language.upper()}): {translated.text}")
        except Exception as e:
            print(f"Translation error: {e}")
    else:
        print("No Spanish speech detected in the audio.")

if __name__ == "__main__":
    print("Ready to record. Press Enter to start.")
    input()
    recorded_audio = record_audio()
    transcribe_and_translate(recorded_audio, TARGET_OUTPUT)
    print("Done.")