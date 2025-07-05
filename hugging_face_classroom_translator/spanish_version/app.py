import gradio as gr
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import os

LANGUAGE_CODES = {
    "Inglés": "en",
    "Español": "es",
    "Criollo Haitiano": "ht"
}

model_size = os.getenv("WHISPER_MODEL_SIZE", "small")
model = WhisperModel(model_size, device="cpu", compute_type="int8")

def transcribe_and_translate(audio_path, input_language, history):
    if history is None:
        history = []

    src_code = LANGUAGE_CODES.get(input_language, "en")
    target_langs = [lang for lang in LANGUAGE_CODES.keys() if lang != input_language]
    tgt_code1 = LANGUAGE_CODES[target_langs[0]]
    tgt_code2 = LANGUAGE_CODES[target_langs[1]]

    orig_history = [entry[0] for entry in history]
    trans1_history = [entry[1] for entry in history]
    trans2_history = [entry[2] for entry in history]
    orig_display = "\n\n".join(orig_history)
    trans1_display = "\n\n".join(trans1_history)
    trans2_display = "\n\n".join(trans2_history)

    yield (
        (orig_display + ("\n\n" if orig_display else "") + "... (Transcribiendo audio)") if audio_path else orig_display,
        (trans1_display + ("\n\n" if trans1_display else "") + "... (Traduciendo...)") if audio_path else trans1_display,
        (trans2_display + ("\n\n" if trans2_display else "") + "... (Traduciendo...)") if audio_path else trans2_display,
        history
    )

    if not audio_path:
        return (orig_display, trans1_display, trans2_display, history)

    segments, info = model.transcribe(audio_path, language=src_code)
    transcript = "".join([segment.text for segment in segments]).strip()
    if transcript == "":
        return (orig_display, trans1_display, trans2_display, history)

    translation1 = GoogleTranslator(source=src_code, target=tgt_code1).translate(transcript)
    translation2 = GoogleTranslator(source=src_code, target=tgt_code2).translate(transcript)

    history.append((transcript, translation1, translation2))
    orig_history.append(transcript)
    trans1_history.append(translation1)
    trans2_history.append(translation2)
    orig_display = "\n\n".join(orig_history)
    trans1_display = "\n\n".join(trans1_history)
    trans2_display = "\n\n".join(trans2_history)

    yield (orig_display, trans1_display, trans2_display, history)

def clear_history():
    return "", "", "", []

def switch_language(new_lang):
    other_langs = [lang for lang in LANGUAGE_CODES.keys() if lang != new_lang]
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
    gr.Markdown("## Traductor Multilingüe para el Aula", elem_id="title")
    with gr.Row():
        input_lang = gr.Dropdown(label="Idioma de entrada", choices=list(LANGUAGE_CODES.keys()), value="Inglés", scale=1)
        clear_btn = gr.Button("Borrar todo", scale=0)
    audio_input = gr.Audio(label="Habla o sube un audio", type="filepath",
                           sources=["microphone", "upload"],
                           waveform_options={"show_recording_waveform": False})
    with gr.Row():
        text_orig = gr.Textbox(label="Original (Inglés)", lines=8, interactive=False, elem_classes="output_text")
        text_trans1 = gr.Textbox(label="Español", lines=8, interactive=False, elem_classes="output_text")
        text_trans2 = gr.Textbox(label="Criollo Haitiano", lines=8, interactive=False, elem_classes="output_text")
    state = gr.State([])

    audio_input.change(fn=transcribe_and_translate,
                       inputs=[audio_input, input_lang, state],
                       outputs=[text_orig, text_trans1, text_trans2, state])
    clear_btn.click(fn=clear_history, inputs=None, outputs=[text_orig, text_trans1, text_trans2, state])
    input_lang.change(fn=switch_language, inputs=input_lang, outputs=[text_orig, text_trans1, text_trans2, state])

if __name__ == "__main__":
    demo.queue().launch()
