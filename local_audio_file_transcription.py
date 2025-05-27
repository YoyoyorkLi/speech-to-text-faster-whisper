import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from faster_whisper import WhisperModel

model_size = "small"

# Run on CPU with int8(Macbook Air)
model = WhisperModel(model_size, device="cpu", compute_type="int8")

#Enter local audio file pathname
segments, info = model.transcribe("/Users/lirunhe/Desktop/Speach to text bot/Colab-Lab takeaways.m4a", beam_size=1)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

full_text = ''.join([segment.text for segment in segments])
print(full_text)