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
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from git import Repo
from llama_cpp import Llama
import requests
import torchaudio
from audiocraft.models import MusicGen, AudioGen
from audiocraft.data.audio import audio_write

try:
    import xformers
    import xformers.ops

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

warnings.filterwarnings("ignore")
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('llama_cpp').setLevel(logging.ERROR)
logging.getLogger('whisper').setLevel(logging.ERROR)
logging.getLogger('TTS').setLevel(logging.ERROR)
logging.getLogger('diffusers').setLevel(logging.ERROR)
logging.getLogger('audiocraft').setLevel(logging.ERROR)

chat_dir = None
tts_model = None
whisper_model = None
audiocraft_model_path = None


def load_model(model_name, model_type):
    if model_type == "transformers":
        if model_name:
            model_path = f"inputs/text/llm_models/{model_name}"
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = AutoModelForCausalLM.from_pretrained(model_path)
            if XFORMERS_AVAILABLE:
                try:
                    model = model.with_xformers()
                except (AttributeError, ImportError):
                    try:
                        model.decoder.enable_xformers_memory_efficient_attention()
                        model.encoder.enable_xformers_memory_efficient_attention()
                    except AttributeError:
                        pass
            return tokenizer, model.to(device)
    elif model_type == "llama":
        if model_name:
            model_path = os.path.join("inputs/text/llm_models", model_name)
            model = Llama(model_path)
            return None, model
    return None, None


def transcribe_audio(audio_file_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model_path = "inputs/text/whisper-medium"

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


def load_tts_model():
    print("Скачивание TTS...")
    tts_model_path = "inputs/audio/XTTS-v2"
    if not os.path.exists(tts_model_path):
        os.makedirs(tts_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/coqui/XTTS-v2", tts_model_path)
    print("TTS установлен")
    return TTS(model_path=tts_model_path, config_path=f"{tts_model_path}/config.json")


def load_whisper_model():
    print("Скачивание Whisper...")
    whisper_model_path = "inputs/text/whisper-medium"
    if not os.path.exists(whisper_model_path):
        os.makedirs(whisper_model_path, exist_ok=True)
        url = ("https://openaipublic.azureedge.net/main/whisper/models"
               "/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt")
        r = requests.get(url, allow_redirects=True)
        open(os.path.join(whisper_model_path, "medium.pt"), "wb").write(r.content)
    print("Whisper установлен")
    model_file = os.path.join(whisper_model_path, "medium.pt")
    return whisper.load_model(model_file)


def load_audiocraft_model(model_name):
    print(f"Скачивание AudioCraft модели: {model_name}...")
    audiocraft_model_path = os.path.join("inputs", "audio", "audiocraft", model_name)
    if not os.path.exists(audiocraft_model_path):
        os.makedirs(audiocraft_model_path, exist_ok=True)
        if model_name == "musicgen-stereo-medium":
            Repo.clone_from("https://huggingface.co/facebook/musicgen-stereo-medium", audiocraft_model_path)
        elif model_name == "audiogen-medium":
            Repo.clone_from("https://huggingface.co/facebook/audiogen-medium", audiocraft_model_path)
        elif model_name == "musicgen-stereo-melody":
            Repo.clone_from("https://huggingface.co/facebook/musicgen-stereo-melody", audiocraft_model_path)
    print(f"AudioCraft модель {model_name} установлена")
    return audiocraft_model_path


def generate_text_and_speech(input_text, input_audio, llm_model_name, llm_model_type, max_tokens, temperature, top_p,
                             top_k, avatar_name, enable_tts, speaker_wav, language):
    global chat_dir, tts_model, whisper_model

    if not input_text and not input_audio:
        return "Пожалуйста, введите ваш запрос!", None, None, None, None

    prompt = transcribe_audio(input_audio) if input_audio else input_text

    if not llm_model_name:
        return "Пожалуйста, выберите модель LLM!", None, None, None, None

    tokenizer, llm_model = load_model(llm_model_name, llm_model_type)

    tts_model = None
    whisper_model = None
    text = None
    audio_path = None
    avatar_path = None

    try:
        if enable_tts:
            if not tts_model:
                tts_model = load_tts_model()
            if not speaker_wav or not language:
                return "Пожалуйста выберите голос и язык для TTS!", None, None, None, None

            device = "cuda" if torch.cuda.is_available() else "cpu"
            tts_model = tts_model.to(device)

        if input_audio:
            if not whisper_model:
                whisper_model = load_whisper_model()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            whisper_model = whisper_model

        if llm_model:
            if llm_model_type == "transformers":
                inputs = tokenizer.encode(prompt, return_tensors="pt")
                device = llm_model.device
                inputs = inputs.to(device)
                outputs = llm_model.generate(inputs, max_new_tokens=max_tokens, top_p=top_p, top_k=top_k,
                                             temperature=temperature, pad_token_id=tokenizer.eos_token_id)
                generated_sequence = outputs[0][inputs.shape[-1]:]
                text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
            elif llm_model_type == "llama":
                outputs = llm_model(prompt)
                text = outputs['choices'][0]['text']

        if not chat_dir:
            now = datetime.now()
            chat_dir = os.path.join('outputs', f"chat_{now.strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(chat_dir)
            os.makedirs(os.path.join(chat_dir, 'text'))
            os.makedirs(os.path.join(chat_dir, 'audio'))

        chat_history_path = os.path.join(chat_dir, 'text', 'chat_history.txt')
        with open(chat_history_path, "a", encoding="utf-8") as f:
            f.write(f"Human: {prompt}\n")
            if text:
                f.write(f"AI: {text}\n\n")

        avatar_path = f"inputs/image/avatars/{avatar_name}" if avatar_name else None
        if enable_tts and text:
            wav = tts_model.tts(text=text, speaker_wav=f"inputs/audio/voices/{speaker_wav}", language=language)
            now = datetime.now()
            audio_filename = f"output_{now.strftime('%Y%m%d_%H%M%S')}.wav"
            audio_path = os.path.join(chat_dir, 'audio', audio_filename)
            sf.write(audio_path, wav, 22050)

    finally:
        if tokenizer is not None:
            del tokenizer
        if llm_model is not None:
            del llm_model
        if tts_model is not None:
            del tts_model
        if whisper_model is not None:
            del whisper_model
        torch.cuda.empty_cache()

    return text, audio_path, avatar_path, chat_dir


def generate_image(prompt, negative_prompt, stable_diffusion_model_name, stable_diffusion_sampler, stable_diffusion_steps, stable_diffusion_cfg,
                   stable_diffusion_width, stable_diffusion_height, stable_diffusion_clip_skip):
    if not stable_diffusion_model_name:
        return None, "Пожалуйста, выберите модель Stable Diffusion!"

    stable_diffusion_model_path = os.path.join("inputs", "image", "sd_models", f"{stable_diffusion_model_name}.safetensors")

    is_sdxl = "SDXL" in stable_diffusion_model_name

    if os.path.exists(stable_diffusion_model_path):
        if is_sdxl:
            stable_diffusion_model = StableDiffusionXLPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto", attention_slice=1
            )
        else:
            stable_diffusion_model = StableDiffusionPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto"
            )
    else:
        print(f"Модель Stable Diffusion не найдена: {stable_diffusion_model_path}")
        if is_sdxl:
            stable_diffusion_model = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True, device_map="auto", attention_slice=1
            )
        else:
            stable_diffusion_model = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", use_safetensors=True, device_map="auto"
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if XFORMERS_AVAILABLE:
        stable_diffusion_model.unet.enable_xformers_memory_efficient_attention()

    stable_diffusion_model.to(device)
    stable_diffusion_model.text_encoder.to(device)
    stable_diffusion_model.vae.to(device)
    stable_diffusion_model.unet.to(device)
    stable_diffusion_model.safety_checker = None

    try:
        images = stable_diffusion_model(prompt, negative_prompt=negative_prompt,
                                        num_inference_steps=stable_diffusion_steps,
                                        guidance_scale=stable_diffusion_cfg, height=stable_diffusion_height,
                                        width=stable_diffusion_width, clip_skip=stable_diffusion_clip_skip,
                                        sampler=stable_diffusion_sampler)
        image = images["images"][0]
        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"images_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format="PNG")
        return image_path, None
    finally:
        del stable_diffusion_model
        torch.cuda.empty_cache()


def generate_audio(prompt, input_audio=None, model_name=None, model_type="musicgen", duration=10, temperature=0.7, top_p=0.9, top_k=30, cfg=8):
    global audiocraft_model_path

    if not model_name:
        return None, "Пожалуйста, выберите модель AudioCraft!"

    if not audiocraft_model_path:
        audiocraft_model_path = load_audiocraft_model(model_name)

    if model_type == "musicgen":
        model = MusicGen.get_pretrained(audiocraft_model_path)
        model.set_generation_params(duration=duration)
    elif model_type == "audiogen":
        model = AudioGen.get_pretrained(audiocraft_model_path)
        model.set_generation_params(duration=duration)
    else:
        return None, "Неверный тип модели!"

    audio_paths = []

    try:
        if input_audio and model_type == "musicgen":
            audio_path = input_audio
            melody, sr = torchaudio.load(audio_path)
            wav = model.generate_with_chroma([prompt], melody[None].expand(1, -1, -1), sr, temperature=temperature, top_p=top_p, top_k=top_k, cfg=cfg)
        else:
            descriptions = [prompt]
            wav = model.generate(descriptions, temperature=temperature, top_p=top_p, top_k=top_k, cfg=cfg)

        today = datetime.now().date()
        audio_dir = os.path.join('outputs', f"audio_{today.strftime('%Y%m%d')}")
        os.makedirs(audio_dir, exist_ok=True)

        for idx, one_wav in enumerate(wav):
            audio_filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}"
            audio_path = os.path.join(audio_dir, audio_filename)
            audio_write(audio_path, one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
            audio_paths.append(audio_path)

        if audio_paths:
            return audio_paths[0], None
        else:
            return None, "Не получилось сгенерировать аудио!"

    finally:
        del model
        torch.cuda.empty_cache()


llm_models_list = [None] + [model for model in os.listdir("inputs/text/llm_models") if not model.endswith(".txt")]
avatars_list = [None] + [avatar for avatar in os.listdir("inputs/image/avatars") if not avatar.endswith(".txt")]
speaker_wavs_list = [None] + [wav for wav in os.listdir("inputs/audio/voices") if not wav.endswith(".txt")]
stable_diffusion_models_list = [None] + [model.replace(".safetensors", "") for model in
                                         os.listdir("inputs/image/sd_models")
                                         if (model.endswith(".safetensors") or not model.endswith(".txt"))]
audiocraft_models_list = ["musicgen-stereo-medium", "audiogen-medium", "musicgen-stereo-melody"]

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
        gr.Audio(label="Аудио ответ от LLM", type="filepath"),
        gr.Image(type="filepath", label="Аватар"),
    ],
    title="НейроЧатWebUI (АЛЬФА) - LLM",
    description="Этот пользовательский интерфейс позволяет вам вводить любой текст или аудио и получать "
                "сгенерированный ответ. Вы можете выбрать модель LLM, "
                "аватар, голос и язык из раскрывающихся списков. Вы также можете настроить параметры модели с "
                "помощью ползунков. Попробуйте и посмотрите, что получится!",
    allow_flagging="never"
)

image_interface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Введите ваш промт"),
        gr.Textbox(label="Введите ваш отрицательный промт", value=""),
        gr.Dropdown(choices=stable_diffusion_models_list, label="Выберите модель Stable Diffusion", value=None),
        gr.Dropdown(choices=["euler_ancestral", "euler", "lms", "heun", "dpm", "dpm_solver", "dpm_solver++"],
                    label="Выберите Sampler", value="euler_ancestral"),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Шаги"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label="CFG"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Ширина"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Высота"),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Пропуск клипа"),
    ],
    outputs=[
        gr.Image(type="filepath", label="Сгенерированное изображение"),
        gr.Textbox(label="Сообщение", type="text"),
    ],
    title="НейроЧатWebUI (АЛЬФА) - Stable Diffusion",
    description="Этот пользовательский интерфейс позволяет вам вводить любой текст и генерировать изображение с помощью Stable Diffusion. "
                "Вы можете выбрать модель Stable Diffusion и настроить параметры генерации с помощью ползунков. "
                "Попробуйте и посмотрите, что получится!",
    allow_flagging="never"
)

audiocraft_interface = gr.Interface(
    fn=generate_audio,
    inputs=[
        gr.Textbox(label="Введите ваш промт"),
        gr.Audio(type="filepath", label="Аудио мелодия (опционально)", interactive=True),
        gr.Dropdown(choices=audiocraft_models_list, label="Выберите модель AudioCraft", value=None),
        gr.Dropdown(choices=["musicgen", "audiogen"], label="Выберите тип модели", value="musicgen"),
        gr.Slider(minimum=1, maximum=120, value=10, step=1, label="Длительность (секунды)"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, label="Температура"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.1, label="Top P"),
        gr.Slider(minimum=0, maximum=100, value=30, step=1, label="Top K"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label="CFG"),
    ],
    outputs=[
        gr.Audio(label="Сгенерированное аудио", type="filepath"),
        gr.Textbox(label="Сообщение", type="text"),
    ],
    title="НейроЧатWebUI (АЛЬФА) - AudioCraft",
    description="Этот пользовательский интерфейс позволяет вам вводить любой текст и генерировать аудио с помощью AudioCraft. "
                "Вы можете выбрать модель AudioCraft и настроить параметры генерации с помощью ползунков. "
                "Попробуйте и посмотрите, что получится!",
    allow_flagging="never"
)

with gr.TabbedInterface([chat_interface, image_interface, audiocraft_interface],
                        tab_names=["LLM", "Stable Diffusion", "AudioCraft"]) as app:
    app.launch()
