import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import soundfile as sf
import os
import torch
from TTS.api import TTS
import whisper


def load_model(model_name):
    model_path = f"inputs/text/llm_models/{model_name}"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return tokenizer, model


def transcribe_audio(audio_file_path):
    model = whisper.load_model("medium")
    result = model.transcribe(audio_file_path)
    return result["text"]


def generate_text_and_speech(input_text, input_audio, llm_model_name, avatar_name, enable_tts,
                             speaker_wav, language):
    prompt = transcribe_audio(input_audio) if input_audio else input_text
    tokenizer, llm_model = load_model(llm_model_name)
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = llm_model.generate(inputs, max_length=512, pad_token_id=tokenizer.eos_token_id)
    generated_sequence = outputs[0][inputs.shape[-1]:]
    text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
    avatar_path = f"inputs/image/avatars/{avatar_name}" if avatar_name else None
    audio_path = None
    if enable_tts:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts = TTS("xtts_v2").to(device)
        wav = tts.tts(text=text, speaker_wav=f"inputs/audio/voices/{speaker_wav}", language=language)
        sf.write('outputs/audio/output.wav', wav, 22050)
        audio_path = 'outputs/audio/output.wav'
    return text, avatar_path, audio_path


llm_models_list = [model for model in os.listdir("inputs/text/llm_models") if not model.endswith(".txt")]
avatars_list = [None] + [avatar for avatar in os.listdir("inputs/image/avatars") if not avatar.endswith(".txt")]
speaker_wavs_list = [wav for wav in os.listdir("inputs/audio/voices") if not wav.endswith(".txt")]


iface = gr.Interface(
    fn=generate_text_and_speech,
    inputs=[
        gr.Textbox(label="Введите ваш запрос"),
        gr.Audio(type="filepath", label="Запишите ваш запрос"),
        gr.Dropdown(choices=llm_models_list, label="Выберите языковую модель"),
        gr.Dropdown(choices=avatars_list, label="Выберите аватар", value=None),
        gr.Checkbox(label="Включить TTS", value=False),
        gr.Dropdown(choices=speaker_wavs_list, label="Выберите голос", interactive=True),
        gr.Dropdown(choices=["en", "ru"], label="Выберите язык голоса", interactive=True),
    ],
    outputs=[
        gr.Textbox(label="Текстовый ответ от LLM", type="text"),
        gr.Image(type="filepath", label="Аватар"),
        gr.Audio(label="Аудио ответ от LLM", type="filepath")
    ],
    title="НейроЧатWebUI",
    description="Этот пользовательский интерфейс позволяет вам вводить любой текст или аудио и получать "
                "сгенерированный ответ. Вы можете выбрать модель, "
                "аватар, голос и язык из раскрывающихся списков. Попробуйте и посмотрите, что произойдет!",
    allow_flagging="never"
)
iface.launch()
