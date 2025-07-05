import gradio as gr
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import os

# Mapping from language name to ISO code for Whisper and translation
LANGUAGE_CODES = {
    "English": "en",
    "Spanish": "es",
    "Haitian Creole": "ht"
}

# Load the Whisper model using faster-whisper on CPU (small model for speed)
model_size = os.getenv("WHISPER_MODEL_SIZE", "small")  # allow override via env if needed
model = WhisperModel(model_size, device="cpu", compute_type="int8")

def transcribe_and_translate(audio_path, input_language, history):
    """
    Transcribe the given audio file with Whisper (faster-whisper) and translate the transcription
    into the other two languages. Maintains a history of transcripts and translations.
    """
    # Ensure history is a list
    if history is None:
        history = []
    # Determine source language code and target language codes
    src_code = LANGUAGE_CODES.get(input_language, "en")
    # The other two languages for translation
    target_langs = [lang for lang in LANGUAGE_CODES.keys() if lang != input_language]
    # Codes for the two target languages
    tgt_code1 = LANGUAGE_CODES[target_langs[0]]
    tgt_code2 = LANGUAGE_CODES[target_langs[1]]
    # Prepare current displayed history strings
    orig_history = [entry[0] for entry in history]
    trans1_history = [entry[1] for entry in history]
    trans2_history = [entry[2] for entry in history]
    orig_display = "\n\n".join(orig_history)
    trans1_display = "\n\n".join(trans1_history)
    trans2_display = "\n\n".join(trans2_history)
    # Show a processing indicator in the UI while transcription/translation is in progress
    yield (
        (orig_display + ("\n\n" if orig_display else "") + "... (Transcribing audio)") if audio_path else orig_display,
        (trans1_display + ("\n\n" if trans1_display else "") + "... (Translating...)") if audio_path else trans1_display,
        (trans2_display + ("\n\n" if trans2_display else "") + "... (Translating...)") if audio_path else trans2_display,
        history
    )
    # If no audio or invalid input, stop further processing
    if not audio_path:
        return (orig_display, trans1_display, trans2_display, history)
    # Transcribe the audio using faster-whisper (with known source language to skip detection)
    segments, info = model.transcribe(audio_path, language=src_code)
    # Combine all segment texts into the full transcription
    transcript = "".join([segment.text for segment in segments]).strip()
    # If transcription is empty (no speech detected), just return without adding to history
    if transcript == "":
        return (orig_display, trans1_display, trans2_display, history)
    # Translate the transcription into the two target languages using deep_translator (GoogleTranslator)
    translation1 = GoogleTranslator(source=src_code, target=tgt_code1).translate(transcript)
    translation2 = GoogleTranslator(source=src_code, target=tgt_code2).translate(transcript)
    # Append the new results to history
    history.append((transcript, translation1, translation2))
    # Update display strings with the new entries
    orig_history.append(transcript)
    trans1_history.append(translation1)
    trans2_history.append(translation2)
    orig_display = "\n\n".join(orig_history)
    trans1_display = "\n\n".join(trans1_history)
    trans2_display = "\n\n".join(trans2_history)
    # Return the updated text columns and history state
    yield (orig_display, trans1_display, trans2_display, history)

def clear_history():
    """Clear the transcript history and reset all text displays."""
    return "", "", "", []

def switch_language(new_lang):
    """Update output labels based on a new input language selection, and clear current history."""
    # Determine the other two languages aside from the new input language
    other_langs = [lang for lang in LANGUAGE_CODES.keys() if lang != new_lang]
    # Prepare updated labels and clear outputs and history
    orig_label = f"Original ({new_lang})"
    trans1_label = other_langs[0]
    trans2_label = other_langs[1]
    return (gr.update(label=orig_label, value=""),
            gr.update(label=trans1_label, value=""),
            gr.update(label=trans2_label, value=""),
            [])

css = """
.output_text textarea {
    font-size: 2em;
    line-height: 1.4;
    color: #000000;
    background-color: #FFFFFF;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("## Multilingual Classroom Translator", elem_id="title")
    # Top row: input language selection and clear button
    with gr.Row():
        input_lang = gr.Dropdown(label="Input Language", choices=list(LANGUAGE_CODES.keys()), value="English", scale=1)
        clear_btn = gr.Button("Clear All", scale=0)
    # Audio input (microphone or file) - compact display without waveform
    audio_input = gr.Audio(label="Speak or Upload Audio", type="filepath",
                           sources=["microphone", "upload"],
                           waveform_options={"show_recording_waveform": False})
    # Three columns for original transcript and two translations (side by side)
    with gr.Row():
        text_orig = gr.Textbox(label="Original (English)", lines=8, interactive=False, elem_classes="output_text")
        text_trans1 = gr.Textbox(label="Spanish", lines=8, interactive=False, elem_classes="output_text")
        text_trans2 = gr.Textbox(label="Haitian Creole", lines=8, interactive=False, elem_classes="output_text")
    # State to hold history of transcripts
    state = gr.State([])
    # Define interactions
    audio_input.change(fn=transcribe_and_translate,
                       inputs=[audio_input, input_lang, state],
                       outputs=[text_orig, text_trans1, text_trans2, state])
    clear_btn.click(fn=clear_history, inputs=None, outputs=[text_orig, text_trans1, text_trans2, state])
    input_lang.change(fn=switch_language, inputs=input_lang, outputs=[text_orig, text_trans1, text_trans2, state])

if __name__ == "__main__":
    demo.queue().launch()