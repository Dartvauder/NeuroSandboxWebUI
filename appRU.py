import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import soundfile as sf
import os
import torch
from TTS.api import TTS
import whisper
from datetime import datetime
import warnings
import logging
from diffusers import StableDiffusionPipeline
from git import Repo

warnings.filterwarnings("ignore")
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('TTS').setLevel(logging.ERROR)


def load_model(model_name):
    if model_name:
        model_path = f"inputs/text/llm_models/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(model_path)
        return tokenizer, model.to(device)
    return None, None


def transcribe_audio(audio_file_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model_path = "models/whisper"
    whisper_repo_path = os.path.join(whisper_model_path, "whisper-medium")
    if not os.path.exists(whisper_repo_path):
        os.makedirs(whisper_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/openai/whisper-medium", whisper_repo_path)
    else:
        repo = Repo(whisper_repo_path)
        repo.remotes.origin.pull()

    model_file = os.path.join(whisper_repo_path, "pytorch_model.bin")
    model = whisper.load_model(model_file, device=device)
    result = model.transcribe(audio_file_path)
    return result["text"]


def generate_text_and_speech(input_text, input_audio, llm_model_name, max_tokens, temperature, top_p, top_k,
                             avatar_name, enable_tts, speaker_wav, language, enable_stable_diffusion,
                             stable_diffusion_steps, stable_diffusion_cfg, stable_diffusion_width,
                             stable_diffusion_height, stable_diffusion_clip_skip, chat_dir=None):
    prompt = transcribe_audio(input_audio) if input_audio else input_text
    tokenizer, llm_model = load_model(llm_model_name)
    tts_model = None
    whisper_model = None
    stable_diffusion_model = None

    try:
        if enable_tts:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tts_model_path = "models/tts"
            tts_repo_path = os.path.join(tts_model_path, "XTTS-v2")

            if not os.path.exists(tts_repo_path):
                os.makedirs(tts_model_path, exist_ok=True)
                Repo.clone_from("https://huggingface.co/coqui/XTTS-v2", tts_repo_path)
            else:
                repo = Repo(tts_repo_path)
                repo.remotes.origin.pull()

            tts_model = TTS(model_path="models/tts/XTTS-v2", config_path="models/tts/XTTS-v2/config.json").to(device)

        if input_audio:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            whisper_model_path = "models/whisper"
            whisper_repo_path = os.path.join(whisper_model_path, "whisper-medium")
            if not os.path.exists(whisper_repo_path):
                os.makedirs(whisper_model_path, exist_ok=True)
                Repo.clone_from("https://huggingface.co/openai/whisper-medium", whisper_repo_path)
            else:
                repo = Repo(whisper_repo_path)
                repo.remotes.origin.pull()
            whisper_model = whisper.load_model(os.path.join(whisper_repo_path, "pytorch_model.bin"), device=device)

        if enable_stable_diffusion:
            stable_diffusion_model_path = "models/stable-diffusion"
            os.makedirs(stable_diffusion_model_path, exist_ok=True)
            stable_diffusion_model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                                             cache_dir=stable_diffusion_model_path)
            stable_diffusion_model.to("cuda")

        text = None
        if llm_model:
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            device = llm_model.device
            inputs = inputs.to(device)
            outputs = llm_model.generate(inputs, max_length=max_tokens, top_p=top_p, top_k=top_k,
                                         temperature=temperature, pad_token_id=tokenizer.eos_token_id)
            generated_sequence = outputs[0][inputs.shape[-1]:]
            text = tokenizer.decode(generated_sequence, skip_special_tokens=True)

        avatar_path = f"inputs/image/avatars/{avatar_name}" if avatar_name else None
        audio_path = None
        if enable_tts and text:
            wav = tts_model.tts(text=text, speaker_wav=f"inputs/audio/voices/{speaker_wav}", language=language)
            if not chat_dir:
                now = datetime.now()
                chat_dir = os.path.join('outputs', f"chat_{now.strftime('%Y%m%d_%H%M%S')}")
                os.makedirs(chat_dir)
                os.makedirs(os.path.join(chat_dir, 'text'))
                os.makedirs(os.path.join(chat_dir, 'audio'))
                os.makedirs(os.path.join(chat_dir, 'image'))
            now = datetime.now()
            audio_filename = f"output_{now.strftime('%Y%m%d_%H%M%S')}.wav"
            audio_path = os.path.join(chat_dir, 'audio', audio_filename)
            sf.write(audio_path, wav, 22050)

        image_path = None
        if enable_stable_diffusion:
            images = stable_diffusion_model(prompt, num_inference_steps=stable_diffusion_steps,
                                            guidance_scale=stable_diffusion_cfg, height=stable_diffusion_height,
                                            width=stable_diffusion_width, clip_skip=stable_diffusion_clip_skip)
            image = images["images"][0]
            if not chat_dir:
                now = datetime.now()
                chat_dir = os.path.join('outputs', f"chat_{now.strftime('%Y%m%d_%H%M%S')}")
                os.makedirs(chat_dir)
                os.makedirs(os.path.join(chat_dir, 'text'))
                os.makedirs(os.path.join(chat_dir, 'audio'))
                os.makedirs(os.path.join(chat_dir, 'image'))
            now = datetime.now()
            image_filename = f"output_{now.strftime('%Y%m%d_%H%M%S')}.png"
            image_path = os.path.join(chat_dir, 'image', image_filename)
            image.save(image_path, format="PNG")

        if not chat_dir:
            now = datetime.now()
            chat_dir = os.path.join('outputs', f"chat_{now.strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(chat_dir)
            os.makedirs(os.path.join(chat_dir, 'text'))
            os.makedirs(os.path.join(chat_dir, 'audio'))
            os.makedirs(os.path.join(chat_dir, 'image'))

        chat_history_path = os.path.join(chat_dir, 'text', 'chat_history.txt')
        with open(chat_history_path, "a", encoding="utf-8") as f:
            f.write(f"Human: {prompt}\n")
            if text:
                f.write(f"AI: {text}\n\n")
    finally:
        if tokenizer is not None:
            del tokenizer
        if llm_model is not None:
            del llm_model
        if tts_model is not None:
            del tts_model
        if whisper_model is not None:
            del whisper_model
        if stable_diffusion_model is not None:
            del stable_diffusion_model
        torch.cuda.empty_cache()

    return text, avatar_path, audio_path, image_path, chat_dir


llm_models_list = [None] + [model for model in os.listdir("inputs/text/llm_models") if not model.endswith(".txt")]
avatars_list = [None] + [avatar for avatar in os.listdir("inputs/image/avatars") if not avatar.endswith(".txt")]
speaker_wavs_list = [None] + [wav for wav in os.listdir("inputs/audio/voices") if not wav.endswith(".txt")]

iface = gr.Interface(
    fn=generate_text_and_speech,
    inputs=[
        gr.Textbox(label="Введите ваш запрос"),
        gr.Audio(type="filepath", label="Запишите ваш запрос"),
        gr.Dropdown(choices=llm_models_list, label="Выберите языковую модель"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Максимум токенов"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, label="Температура"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.1, label="Top P"),
        gr.Slider(minimum=0, maximum=100, value=30, step=1, label="Top K"),
        gr.Dropdown(choices=avatars_list, label="Выберите аватар", value=None),
        gr.Checkbox(label="Включить TTS", value=False),
        gr.Dropdown(choices=speaker_wavs_list, label="Выберите голос", interactive=True),
        gr.Dropdown(choices=["en", "ru"], label="Выберите язык голоса", interactive=True),
        gr.Checkbox(label="Включить Stable Diffusion", value=False),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Шаги"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label="CFG"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Ширина"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Высота"),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Пропуск клипа"),
        gr.State()
    ],
    outputs=[
        gr.Textbox(label="Текстовый ответ от LLM", type="text"),
        gr.Image(type="filepath", label="Аватар"),
        gr.Audio(label="Аудио ответ от LLM", type="filepath"),
        gr.Image(type="filepath", label="Сгенерированное изображение"),
        gr.State()
    ],
    title="NeuroChatWebUI (ALPHA)",
    description="Этот пользовательский интерфейс позволяет вам вводить любой текст или аудио и получать "
                "сгенерированный ответ или изображение. Вы можете выбрать модель, "
                "аватар, голос и язык из раскрывающихся списков. А так же вы можете настраивать настройки моделей с "
                "помощью слайдеров. Попробуйте и посмотрите, что произойдет!",
    allow_flagging="never"
)

iface.launch()
