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
from llama_cpp import Llama
import requests

warnings.filterwarnings("ignore")
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('llama_cpp').setLevel(logging.ERROR)
logging.getLogger('whisper').setLevel(logging.ERROR)
logging.getLogger('TTS').setLevel(logging.ERROR)
logging.getLogger('diffusers').setLevel(logging.ERROR)
logging.getLogger('h11').setLevel(logging.ERROR)
logging.getLogger('uvicorn').setLevel(logging.ERROR)


def load_model(model_name, model_type):
    if model_type == "transformers":
        if model_name:
            model_path = f"inputs/text/llm_models/{model_name}"
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = AutoModelForCausalLM.from_pretrained(model_path)
            return tokenizer, model.to(device)
    elif model_type == "llama":
        if model_name:
            model_path = f"inputs/text/llm_models/{model_name}"
            model = Llama(model_path)
            return model, None
    return None, None


def transcribe_audio(audio_file_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model_path = "inputs/text/whisper-medium"

    # Download the medium.pt file from the provided URL
    if not os.path.exists(whisper_model_path):
        os.makedirs(whisper_model_path, exist_ok=True)
        url = ("https://openaipublic.azureedge.net/main/whisper/models"
               "/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt")
        r = requests.get(url, allow_redirects=True)
        open(os.path.join(whisper_model_path, "medium.pt"), "wb").write(r.content)

    model_file = os.path.join(whisper_model_path, "medium.pt")
    model = whisper.load_model(model_file, device=device)
    result = model.transcribe(audio_file_path)
    return result["text"]


def generate_text_and_speech(input_text, input_audio, llm_model_name, llm_model_type, max_tokens, temperature, top_p,
                             top_k, avatar_name, enable_tts, speaker_wav, language, enable_stable_diffusion,
                             stable_diffusion_model_name, stable_diffusion_steps, stable_diffusion_cfg,
                             stable_diffusion_width, stable_diffusion_height, stable_diffusion_clip_skip,
                             chat_dir=None):
    prompt = transcribe_audio(input_audio) if input_audio else input_text

    if not llm_model_name:
        return "Please select a language model.", None, None, None, None

    tokenizer, llm_model = load_model(llm_model_name, llm_model_type)

    tts_model = None
    whisper_model = None
    stable_diffusion_model = None

    try:
        if enable_tts:
            if not speaker_wav or not language:
                return "Please select a voice and language for TTS.", None, None, None, None

            device = "cuda" if torch.cuda.is_available() else "cpu"
            tts_model_path = "inputs/audio/XTTS-v2"
            if not os.path.exists(tts_model_path):
                os.makedirs(tts_model_path, exist_ok=True)
                Repo.clone_from("https://huggingface.co/coqui/XTTS-v2", tts_model_path)
            else:
                repo = Repo(tts_model_path)
                repo.remotes.origin.pull()
            tts_model = TTS(model_path=tts_model_path, config_path=f"{tts_model_path}/config.json").to(device)

        if input_audio:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            whisper_model_path = "inputs/text/whisper-medium"

            # Download the medium.pt file from the provided URL
            if not os.path.exists(whisper_model_path):
                os.makedirs(whisper_model_path, exist_ok=True)
                url = ("https://openaipublic.azureedge.net/main/whisper/models"
                       "/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt")
                r = requests.get(url, allow_redirects=True)
                open(os.path.join(whisper_model_path, "medium.pt"), "wb").write(r.content)

            model_file = os.path.join(whisper_model_path, "medium.pt")
            whisper_model = whisper.load_model(model_file, device=device)

        if enable_stable_diffusion:
            if not stable_diffusion_model_name:
                return "Please select a Stable Diffusion model.", None, None, None, None

            stable_diffusion_model_path = os.path.join("inputs", "image", "sd_models",
                                                       f"{stable_diffusion_model_name}.safetensors")
            if os.path.exists(stable_diffusion_model_path):
                stable_diffusion_model = StableDiffusionPipeline.from_single_file(
                    stable_diffusion_model_path, use_safetensors=True, device_map="auto"
                )
            else:
                print(f"Stable Diffusion model not found: {stable_diffusion_model_path}")
                stable_diffusion_model = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5", use_safetensors=True, device_map="auto"
                )
            device = "cuda" if torch.cuda.is_available() else "cpu"

            stable_diffusion_model.to(device)
            stable_diffusion_model.text_encoder.to(device)
            stable_diffusion_model.vae.to(device)
            stable_diffusion_model.unet.to(device)
            stable_diffusion_model.safety_checker = None

        text = None
        if llm_model:
            if llm_model_type == "transformers":
                inputs = tokenizer.encode(prompt, return_tensors="pt")
                device = llm_model.device
                inputs = inputs.to(device)
                outputs = llm_model.generate(inputs, max_length=max_tokens, top_p=top_p, top_k=top_k,
                                             temperature=temperature, pad_token_id=tokenizer.eos_token_id)
                generated_sequence = outputs[0][inputs.shape[-1]:]
                text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
            elif llm_model_type == "llama":
                text = llm_model(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                                 top_k=top_k)

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
stable_diffusion_models_list = [None] + [model.replace(".safetensors", "") for model in
                                         os.listdir("inputs/image/sd_models")
                                         if (model.endswith(".safetensors") or not model.endswith(".txt"))
                                         and os.path.isfile(os.path.join("inputs/image/sd_models", model))]


chat_interface = gr.Interface(
    fn=generate_text_and_speech,
    inputs=[
        gr.Textbox(label="Введите ваш запрос"),
        gr.Audio(type="filepath", label="Запишите ваш запрос"),
        gr.Dropdown(choices=llm_models_list, label="Выберите LLM модель", value=None),
        gr.Dropdown(choices=["transformers", "llama"], label="Выберите тип модели", value="transformers"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Максимум токенов"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, label="Температура"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.1, label="Top P"),
        gr.Slider(minimum=0, maximum=100, value=30, step=1, label="Top K"),
        gr.Dropdown(choices=avatars_list, label="Выберите аватар", value=None),
        gr.Checkbox(label="Включить TTS", value=False),
        gr.Dropdown(choices=speaker_wavs_list, label="Выберите голос", interactive=True),
        gr.Dropdown(choices=["en", "ru"], label="Выберите язык голоса", interactive=True),
    ],
    outputs=[
        gr.Textbox(label="Текстовый ответ от LLM", type="text"),
        gr.Image(type="filepath", label="Аватар"),
        gr.Audio(label="Аудио ответ от LLM", type="filepath"),
    ],
    title="NeuroChatWebUI (ALPHA) - Chat",
    description="Этот пользовательский интерфейс позволяет вам вводить любой текст или аудио и получать "
                "сгенерирован ответ. Вы можете выбрать модель LLM, "
                "аватар, голос и язык из раскрывающихся списков. Вы также можете настроить параметры модели с "
                "помощью слайдеров. Попробуйте и посмотрите, что получится!",
    allow_flagging="never"
)

image_interface = gr.Interface(
    fn=generate_text_and_speech,
    inputs=[
        gr.Textbox(label="Введите ваш запрос"),
        gr.Checkbox(label="Включить Stable Diffusion", value=False),
        gr.Dropdown(choices=stable_diffusion_models_list, label="Выберите модель Stable Diffusion", interactive=True),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Шаги"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label="CFG"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Ширина"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Высота"),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Пропуск клипа"),
    ],
    outputs=[
        gr.Image(type="filepath", label="Сгенерированное изображение"),
    ],
    title="NeuroChatWebUI (ALPHA) - Image",
    description="Этот пользовательский интерфейс позволяет вам вводить любой текст и генерировать изображение с помощью Stable Diffusion. "
                "Вы можете выбрать модель Stable Diffusion и настроить параметры генерации с помощью ползунков. "
                "Попробуйте и посмотрите, что получится!",
    allow_flagging="never"
)

with gr.TabbedInterface([chat_interface, image_interface]) as app:
    app.launch()
