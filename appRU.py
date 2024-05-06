import gradio as gr
import langdetect
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, BarkModel
from libretranslatepy import LibreTranslateAPI
import urllib.error
import soundfile as sf
import os
import subprocess
import json
import torch
from einops import rearrange
from TTS.api import TTS
import whisper
from datetime import datetime
import warnings
import logging
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionDepth2ImgPipeline, AutoencoderKL, StableDiffusionLatentUpscalePipeline, StableDiffusionUpscalePipeline, StableDiffusionInpaintPipeline, StableDiffusionGLIGENPipeline, AnimateDiffPipeline, DDIMScheduler, MotionAdapter, StableVideoDiffusionPipeline, I2VGenXLPipeline, StableCascadePriorPipeline, StableCascadeDecoderPipeline, DiffusionPipeline, DPMSolverMultistepScheduler, ShapEPipeline, ShapEImg2ImgPipeline, AudioLDM2Pipeline
from diffusers.utils import load_image, export_to_video, export_to_gif, export_to_ply
from compel import Compel
import trimesh
from tsr.system import TSR
from tsr.utils import to_gradio_3d_orientation, resize_foreground
from git import Repo
import numpy as np
import scipy
import imageio
from PIL import Image
from tqdm import tqdm
from llama_cpp import Llama
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from rembg import remove
import torchaudio
from audiocraft.models import MusicGen, AudioGen, MultiBandDiffusion  # MAGNeT
from audiocraft.data.audio import audio_write
import psutil
import GPUtil
from cpuinfo import get_cpu_info
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetTemperature, NVML_TEMPERATURE_GPU

XFORMERS_AVAILABLE = False
torch.cuda.is_available()
try:
    import xformers
    import xformers.ops

    XFORMERS_AVAILABLE = True
except ImportError:
    print("Xformers is not installed. Proceeding without it")

warnings.filterwarnings("ignore")
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('llama_cpp').setLevel(logging.ERROR)
logging.getLogger('whisper').setLevel(logging.ERROR)
logging.getLogger('TTS').setLevel(logging.ERROR)
logging.getLogger('diffusers').setLevel(logging.ERROR)
logging.getLogger('audiocraft').setLevel(logging.ERROR)
logging.getLogger('xformers').setLevel(logging.ERROR)

chat_dir = None
tts_model = None
whisper_model = None
audiocraft_model_path = None


def remove_bg(src_img_path, out_img_path):
    model_path = "inputs/image/sd_models/rembg"
    os.makedirs(model_path, exist_ok=True)

    os.environ["U2NET_HOME"] = model_path

    with open(src_img_path, "rb") as input_file:
        input_data = input_file.read()

    output_data = remove(input_data)

    with open(out_img_path, "wb") as output_file:
        output_file.write(output_data)


def load_model(model_name, model_type, n_ctx=None):
    global stop_signal
    if stop_signal:
        return None, None, "Generation stopped"
    if model_name:
        model_path = f"inputs/text/llm_models/{model_name}"
        if model_type == "transformers":
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device,
                    load_in_4bit=True,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                return tokenizer, model, None
            except (OSError, RuntimeError):
                return None, None, "The selected model is not compatible with the 'transformers' model type"
        elif model_type == "llama":
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = Llama(model_path, n_gpu_layers=-1 if device == "cuda" else 0)
                model.n_ctx = n_ctx
                tokenizer = None
                return tokenizer, model, None
            except (ValueError, RuntimeError):
                return None, None, "The selected model is not compatible with the 'llama' model type"
    return None, None, None


def load_moondream2_model(model_id, revision):
    global stop_signal
    if stop_signal:
        return "Generation stopped"
    print(f"Downloading MoonDream2 model...")
    moondream2_model_path = os.path.join("inputs", "text", "llm_models", model_id)
    if not os.path.exists(moondream2_model_path):
        os.makedirs(moondream2_model_path, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, revision=revision
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model.save_pretrained(moondream2_model_path)
        tokenizer.save_pretrained(moondream2_model_path)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(moondream2_model_path, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(moondream2_model_path)
    print("MoonDream2 model downloaded")
    return model, tokenizer


def transcribe_audio(audio_file_path):
    global stop_signal
    if stop_signal:
        return "Generation stopped"
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
    global stop_signal
    if stop_signal:
        return "Generation stopped"
    print("Downloading TTS...")
    tts_model_path = "inputs/audio/XTTS-v2"
    if not os.path.exists(tts_model_path):
        os.makedirs(tts_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/coqui/XTTS-v2", tts_model_path)
    print("TTS model downloaded")
    return TTS(model_path=tts_model_path, config_path=f"{tts_model_path}/config.json")


def load_whisper_model():
    global stop_signal
    if stop_signal:
        return "Generation stopped"
    print("Downloading Whisper...")
    whisper_model_path = "inputs/text/whisper-medium"
    if not os.path.exists(whisper_model_path):
        os.makedirs(whisper_model_path, exist_ok=True)
        url = ("https://openaipublic.azureedge.net/main/whisper/models"
               "/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt")
        r = requests.get(url, allow_redirects=True)
        open(os.path.join(whisper_model_path, "medium.pt"), "wb").write(r.content)
    print("Whisper downloaded")
    model_file = os.path.join(whisper_model_path, "medium.pt")
    return whisper.load_model(model_file)


def load_audiocraft_model(model_name):
    global stop_signal
    if stop_signal:
        return "Generation stopped"
    global audiocraft_model_path
    print(f"Downloading AudioCraft model: {model_name}...")
    audiocraft_model_path = os.path.join("inputs", "audio", "audiocraft", model_name)
    if not os.path.exists(audiocraft_model_path):
        os.makedirs(audiocraft_model_path, exist_ok=True)
        if model_name == "musicgen-stereo-medium":
            Repo.clone_from("https://huggingface.co/facebook/musicgen-stereo-medium", audiocraft_model_path)
        elif model_name == "audiogen-medium":
            Repo.clone_from("https://huggingface.co/facebook/audiogen-medium", audiocraft_model_path)
        elif model_name == "musicgen-stereo-melody":
            Repo.clone_from("https://huggingface.co/facebook/musicgen-stereo-melody", audiocraft_model_path)
        elif model_name == "musicgen-medium":
            Repo.clone_from("https://huggingface.co/facebook/musicgen-medium", audiocraft_model_path)
        elif model_name == "musicgen-melody":
            Repo.clone_from("https://huggingface.co/facebook/musicgen-melody", audiocraft_model_path)
        elif model_name == "musicgen-large":
            Repo.clone_from("https://huggingface.co/facebook/musicgen-large", audiocraft_model_path)
        elif model_name == "hybrid-magnet-medium":
            Repo.clone_from("https://huggingface.co/facebook/hybrid-magnet-medium", audiocraft_model_path)
        elif model_name == "magnet-medium-30sec":
            Repo.clone_from("https://huggingface.co/facebook/magnet-medium-30secs", audiocraft_model_path)
        elif model_name == "magnet-medium-10sec":
            Repo.clone_from("https://huggingface.co/facebook/magnet-medium-10secs", audiocraft_model_path)
        elif model_name == "audio-magnet-medium":
            Repo.clone_from("https://huggingface.co/facebook/audio-magnet-medium", audiocraft_model_path)
    print(f"AudioCraft model {model_name} downloaded")
    return audiocraft_model_path


def load_multiband_diffusion_model():
    global stop_signal
    if stop_signal:
        return "Generation stopped"
    print(f"Downloading Multiband Diffusion model")
    multiband_diffusion_path = os.path.join("inputs", "audio", "audiocraft", "multiband-diffusion")
    if not os.path.exists(multiband_diffusion_path):
        os.makedirs(multiband_diffusion_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/facebook/multiband-diffusion", multiband_diffusion_path)
        print("Multiband Diffusion model downloaded")
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_upscale_model(upscale_factor):
    global stop_signal
    if stop_signal:
        return None, "Generation stopped"
    original_config_file = None

    if upscale_factor == 2:
        upscale_model_name = "stabilityai/sd-x2-latent-upscaler"
        upscale_model_path = os.path.join("inputs", "image", "sd_models", "upscale", "x2-upscaler")
    else:
        upscale_model_name = "stabilityai/stable-diffusion-x4-upscaler"
        upscale_model_path = os.path.join("inputs", "image", "sd_models", "upscale", "x4-upscaler")
        original_config_file = "configs/sd/x4-upscaling.yaml"

    print(f"Downloading Upscale model: {upscale_model_name}")

    if not os.path.exists(upscale_model_path):
        os.makedirs(upscale_model_path, exist_ok=True)
        Repo.clone_from(f"https://huggingface.co/{upscale_model_name}", upscale_model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if upscale_factor == 2:
        upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
            upscale_model_path,
            revision="fp16",
            torch_dtype=torch.float16
        )
    else:
        upscaler = StableDiffusionUpscalePipeline.from_pretrained(
            upscale_model_path,
            original_config_file=original_config_file,
            revision="fp16",
            torch_dtype=torch.float16
        )

    upscaler.to(device)
    upscaler.enable_attention_slicing()
    if XFORMERS_AVAILABLE:
        upscaler.enable_xformers_memory_efficient_attention(attention_op=None)

    print(f"Upscale model {upscale_model_name} downloaded")

    upscaler.upscale_factor = upscale_factor
    return upscaler


stop_signal = False

chat_history = []


def generate_text_and_speech(input_text, input_audio, input_image, llm_model_name, llm_settings_html, llm_model_type, max_length, max_tokens,
                             temperature, top_p, top_k, chat_history_format, enable_web_search, enable_libretranslate, target_lang, enable_multimodal, enable_tts, tts_settings_html,
                             speaker_wav, language, tts_temperature, tts_top_p, tts_top_k, tts_speed, output_format, stop_generation):
    global chat_history, chat_dir, tts_model, whisper_model, stop_signal
    stop_signal = False
    if not input_text and not input_audio:
        chat_history.append(["Please, enter your request!", None])
        return chat_history, None, None, None
    prompt = transcribe_audio(input_audio) if input_audio else input_text
    if not llm_model_name:
        chat_history.append([None, "Please, select a LLM model!"])
        return chat_history, None, None, None
    if enable_multimodal and llm_model_name == "moondream2":
        if llm_model_type == "llama":
            chat_history.append([None, "Multimodal with 'llama' model type is not supported yet!"])
            return chat_history, None, None, None
        model_id = "vikhyatk/moondream2"
        revision = "2024-04-02"
        model, tokenizer = load_moondream2_model(model_id, revision)

        try:
            image = Image.open(input_image)
            enc_image = model.encode_image(image)

            detect_lang = langdetect.detect(prompt)
            if detect_lang == "en":
                bot_instruction = "You are a friendly chatbot who always provides useful and meaningful answers based on the given image and text input."
            else:
                bot_instruction = "Вы дружелюбный чат-бот, который всегда дает полезные и содержательные ответы на основе данного изображения и текстового ввода."

            context = ""
            for human_text, ai_text in chat_history[-5:]:
                if human_text:
                    context += f"Human: {human_text}\n"
                if ai_text:
                    context += f"AI: {ai_text}\n"

            prompt_with_context = f"{bot_instruction}\n\n{context}Human: {prompt}\nAI:"

            text = model.answer_question(enc_image, prompt_with_context, tokenizer)
        finally:
            del model
            del tokenizer
            torch.cuda.empty_cache()

        if not chat_dir:
            now = datetime.now()
            chat_dir = os.path.join('outputs', f"LLM_{now.strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(chat_dir)
            os.makedirs(os.path.join(chat_dir, 'text'))
            os.makedirs(os.path.join(chat_dir, 'audio'))
        chat_history_path = os.path.join(chat_dir, 'text', f'chat_history.{chat_history_format}')
        if chat_history_format == "txt":
            with open(chat_history_path, "a", encoding="utf-8") as f:
                f.write(f"Human: {prompt}\n")
                if text:
                    f.write(f"AI: {text}\n\n")
        elif chat_history_format == "json":
            chat_history = []
            if os.path.exists(chat_history_path):
                with open(chat_history_path, "r", encoding="utf-8") as f:
                    chat_history = json.load(f)
            chat_history.append(["Human: " + prompt, "AI: " + (text if text else "")])
            with open(chat_history_path, "w", encoding="utf-8") as f:
                json.dump(chat_history, f, ensure_ascii=False, indent=4)

        chat_history.append([prompt, text])
        return chat_history, None, chat_dir, None
    else:
        tokenizer, llm_model, error_message = load_model(llm_model_name, llm_model_type)
        if error_message:
            chat_history.append([None, error_message])
            return chat_history, None, None, None
        tts_model = None
        whisper_model = None
        text = None
        audio_path = None

        try:
            if enable_tts:
                if not tts_model:
                    tts_model = load_tts_model()
                if not speaker_wav or not language:
                    chat_history.append([None, "Please, select a voice and language for TTS!"])
                    return chat_history, None, None, None
                device = "cuda" if torch.cuda.is_available() else "cpu"
                tts_model = tts_model.to(device)
            if input_audio:
                if not whisper_model:
                    whisper_model = load_whisper_model()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                whisper_model = whisper_model.to(device)
            if enable_web_search:
                chat_history.append([None, "This feature doesn't work yet. Please, turn it off"])
                return chat_history, None, None, None
            if llm_model:
                if llm_model_type == "transformers":
                    detect_lang = langdetect.detect(prompt)
                    if detect_lang == "en":
                        bot_instruction = "You are a friendly chatbot who always provides useful and meaningful answers in any language"
                    else:
                        bot_instruction = "Вы дружелюбный чат-бот, который всегда дает полезные и содержательные ответы на любом языке"

                    context = ""
                    for human_text, ai_text in chat_history[-10:]:
                        if human_text:
                            context += f"Human: {human_text}\n"
                        if ai_text:
                            context += f"AI: {ai_text}\n"

                    messages = [
                        {
                            "role": "system",
                            "content": bot_instruction,
                        },
                        {"role": "user", "content": context + prompt},
                    ]

                    tokenizer.padding_side = "left"
                    tokenizer.pad_token = tokenizer.eos_token
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, padding=True,
                                                                 return_tensors="pt").to(device)
                    input_length = model_inputs.shape[1]

                    progress_bar = tqdm(total=max_length, desc="Generating text")
                    progress_tokens = 0

                    generated_ids = llm_model.generate(
                        model_inputs,
                        do_sample=True,
                        max_new_tokens=max_length,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        repetition_penalty=1.1,
                        num_beams=5,
                        no_repeat_ngram_size=2,
                    )

                    progress_tokens = max_length
                    progress_bar.update(progress_tokens - progress_bar.n)

                    if stop_signal:
                        chat_history.append([prompt, "Generation stopped"])
                        return chat_history, None, None

                    progress_bar.close()

                    text = tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0]

                elif llm_model_type == "llama":
                    detect_lang = langdetect.detect(prompt)
                    if detect_lang == "en":
                        instruction = "I am a chatbot created to help with any questions. I use my knowledge and abilities to provide useful and meaningful answers in any language\n\n"
                    else:
                        instruction = "Я чат-бот, созданный для помощи по любым вопросам. Я использую свои знания и способности, чтобы давать полезные и содержательные ответы на любом языке\n\n"

                    context = ""
                    for human_text, ai_text in chat_history[-10:]:
                        if human_text:
                            context += f"Human: {human_text}\n"
                        if ai_text:
                            context += f"Assistant: {ai_text}\n"

                    prompt_with_context = instruction + context + "Human: " + prompt + "\nAssistant: "

                    progress_bar = tqdm(total=max_tokens, desc="Generating text")
                    progress_tokens = 0

                    output = llm_model(
                        prompt_with_context,
                        max_tokens=max_tokens,
                        stop=["Human:", "\n"],
                        echo=False,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repeat_penalty=1.1,
                    )

                    progress_tokens = max_tokens
                    progress_bar.update(progress_tokens - progress_bar.n)

                    if stop_signal:
                        chat_history.append([prompt, "Generation stopped"])
                        return chat_history, None, None

                    progress_bar.close()

                    text = output['choices'][0]['text']

                if enable_libretranslate:
                    try:
                        translator = LibreTranslateAPI("http://127.0.0.1:5000")
                        translation = translator.translate(text, detect_lang, target_lang)
                        text = translation
                    except urllib.error.URLError:
                        chat_history.append([None, "LibreTranslate is not running. Please start the LibreTranslate server."])
                        return chat_history, None, None, None

            if not chat_dir:
                now = datetime.now()
                chat_dir = os.path.join('outputs', f"LLM_{now.strftime('%Y%m%d_%H%M%S')}")
                os.makedirs(chat_dir)
                os.makedirs(os.path.join(chat_dir, 'text'))
                os.makedirs(os.path.join(chat_dir, 'audio'))
            chat_history_path = os.path.join(chat_dir, 'text', f'chat_history.{chat_history_format}')
            if chat_history_format == "txt":
                with open(chat_history_path, "a", encoding="utf-8") as f:
                    f.write(f"Human: {prompt}\n")
                    if text:
                        f.write(f"AI: {text}\n\n")
            elif chat_history_format == "json":
                chat_history = []
                if os.path.exists(chat_history_path):
                    with open(chat_history_path, "r", encoding="utf-8") as f:
                        chat_history = json.load(f)
                chat_history.append(["Human: " + prompt, "AI: " + (text if text else "")])
                with open(chat_history_path, "w", encoding="utf-8") as f:
                    json.dump(chat_history, f, ensure_ascii=False, indent=4)
            if enable_tts and text:
                if stop_signal:
                    chat_history.append([prompt, text])
                    return chat_history, None, chat_dir, "Generation stopped"
                enable_text_splitting = False
                repetition_penalty = 2.0
                length_penalty = 1.0
                if enable_text_splitting:
                    text_parts = text.split(".")
                    for part in text_parts:
                        wav = tts_model.tts(text=part.strip(), speaker_wav=f"inputs/audio/voices/{speaker_wav}",
                                            language=language,
                                            temperature=tts_temperature, top_p=tts_top_p, top_k=tts_top_k, speed=tts_speed,
                                            repetition_penalty=repetition_penalty, length_penalty=length_penalty)
                        now = datetime.now()
                        audio_filename = f"TTS_{now.strftime('%Y%m%d_%H%M%S')}.{output_format}"
                        audio_path = os.path.join(chat_dir, 'audio', audio_filename)
                        if output_format == "mp3":
                            sf.write(audio_path, wav, 22050, format='mp3')
                        elif output_format == "ogg":
                            sf.write(audio_path, wav, 22050, format='ogg')
                        else:
                            sf.write(audio_path, wav, 22050)
                else:
                    wav = tts_model.tts(text=text, speaker_wav=f"inputs/audio/voices/{speaker_wav}", language=language,
                                        temperature=tts_temperature, top_p=tts_top_p, top_k=tts_top_k, speed=tts_speed,
                                        repetition_penalty=repetition_penalty, length_penalty=length_penalty)
                    now = datetime.now()
                    audio_filename = f"TTS_{now.strftime('%Y%m%d_%H%M%S')}.{output_format}"
                    audio_path = os.path.join(chat_dir, 'audio', audio_filename)
                    if output_format == "mp3":
                        sf.write(audio_path, wav, 22050, format='mp3')
                    elif output_format == "ogg":
                        sf.write(audio_path, wav, 22050, format='ogg')
                    else:
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

    chat_history.append([prompt, text])
    return chat_history, audio_path, chat_dir, None


def generate_tts_stt(text, audio, tts_settings_html, speaker_wav, language, tts_temperature, tts_top_p, tts_top_k, tts_speed, tts_output_format, stt_output_format):
    global tts_model, whisper_model

    tts_output = None
    stt_output = None

    if not text and not audio:
        return None, "Please enter text for TTS or record audio for STT!"

    if text:
        if not tts_model:
            tts_model = load_tts_model()
        if not speaker_wav or not language:
            return None, "Please select a voice and language for TTS!"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts_model = tts_model.to(device)

        wav = tts_model.tts(text=text, speaker_wav=f"inputs/audio/voices/{speaker_wav}", language=language,
                            temperature=tts_temperature, top_p=tts_top_p, top_k=tts_top_k, speed=tts_speed)

        today = datetime.now().date()
        audio_dir = os.path.join('outputs', f"TTS_{today.strftime('%Y%m%d')}")
        os.makedirs(audio_dir, exist_ok=True)
        audio_filename = f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{tts_output_format}"
        tts_output = os.path.join(audio_dir, audio_filename)

        if tts_output_format == "mp3":
            sf.write(tts_output, wav, 22050, format='mp3')
        elif tts_output_format == "ogg":
            sf.write(tts_output, wav, 22050, format='ogg')
        else:
            sf.write(tts_output, wav, 22050)

    if audio:
        if not whisper_model:
            whisper_model = load_whisper_model()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        whisper_model = whisper_model.to(device)

        stt_output = transcribe_audio(audio)

        if stt_output:
            today = datetime.now().date()
            stt_dir = os.path.join('outputs', f"STT_{today.strftime('%Y%m%d')}")
            os.makedirs(stt_dir, exist_ok=True)

            if stt_output_format == "txt":
                stt_filename = f"stt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                stt_file_path = os.path.join(stt_dir, stt_filename)
                with open(stt_file_path, 'w', encoding='utf-8') as f:
                    f.write(stt_output)
            elif stt_output_format == "json":
                stt_filename = f"stt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                stt_file_path = os.path.join(stt_dir, stt_filename)
                stt_history = []
                if os.path.exists(stt_file_path):
                    with open(stt_file_path, "r", encoding="utf-8") as f:
                        stt_history = json.load(f)
                stt_history.append(stt_output)
                with open(stt_file_path, "w", encoding="utf-8") as f:
                    json.dump(stt_history, f, ensure_ascii=False, indent=4)

    return tts_output, stt_output


def generate_bark_audio(text, voice_preset, max_length, fine_temperature, coarse_temperature, output_format, stop_generation):
    global stop_signal
    stop_signal = False

    if not text:
        return None, "Please enter text for the request!"

    bark_model_path = os.path.join("inputs", "audio", "bark")

    if not os.path.exists(bark_model_path):
        print("Downloading Bark model...")
        os.makedirs(bark_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/suno/bark", bark_model_path)
        print("Bark model downloaded")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_default_tensor_type(torch.cuda.FloatTensor if device == "cuda" else torch.FloatTensor)

        processor = AutoProcessor.from_pretrained(bark_model_path)
        model = BarkModel.from_pretrained(bark_model_path, torch_dtype=torch.float32)

        if voice_preset:
            inputs = processor(text, voice_preset=voice_preset, return_tensors="pt")
        else:
            inputs = processor(text, return_tensors="pt")

        audio_array = model.generate(**inputs, max_length=max_length, do_sample=True, fine_temperature=fine_temperature, coarse_temperature=coarse_temperature)
        model.enable_cpu_offload()

        if stop_signal:
            return None, "Generation stopped"

        today = datetime.now().date()
        audio_dir = os.path.join('outputs', f"Bark_{today.strftime('%Y%m%d')}")
        os.makedirs(audio_dir, exist_ok=True)

        audio_array = audio_array.cpu().numpy().squeeze()

        audio_filename = f"bark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        audio_path = os.path.join(audio_dir, audio_filename)

        if output_format == "mp3":
            sf.write(audio_path, audio_array, 24000)
        elif output_format == "ogg":
            sf.write(audio_path, audio_array, 24000)
        else:
            sf.write(audio_path, audio_array, 24000)

        return audio_path, None

    except Exception as e:
        return None, str(e)


def translate_text(text, source_lang, target_lang, enable_translate_history, translate_history_format, file=None):
    try:
        translator = LibreTranslateAPI("http://127.0.0.1:5000")
        if file:
            with open(file.name, "r", encoding="utf-8") as f:
                text = f.read()
        translation = translator.translate(text, source_lang, target_lang)

        if enable_translate_history:
            today = datetime.now().date()
            translate_dir = os.path.join('outputs', f"Translate_{today.strftime('%Y%m%d')}")
            os.makedirs(translate_dir, exist_ok=True)

            translate_history_path = os.path.join(translate_dir, f'translate_history.{translate_history_format}')
            if translate_history_format == "txt":
                with open(translate_history_path, "a", encoding="utf-8") as f:
                    f.write(f"Source ({source_lang}): {text}\n")
                    f.write(f"Translation ({target_lang}): {translation}\n\n")
            elif translate_history_format == "json":
                translate_history = []
                if os.path.exists(translate_history_path):
                    with open(translate_history_path, "r", encoding="utf-8") as f:
                        translate_history = json.load(f)
                translate_history.append({
                    "source": {
                        "language": source_lang,
                        "text": text
                    },
                    "translation": {
                        "language": target_lang,
                        "text": translation
                    }
                })
                with open(translate_history_path, "w", encoding="utf-8") as f:
                    json.dump(translate_history, f, ensure_ascii=False, indent=4)

        return translation

    except urllib.error.URLError as e:
        error_message = "LibreTranslate is not running. Please start the LibreTranslate server."
        return error_message


def generate_wav2lip(image_path, audio_path, fps, pads, face_det_batch_size, wav2lip_batch_size, resize_factor, crop):
    global stop_signal
    stop_signal = False

    if not image_path or not audio_path:
        return None, "Please upload an image and an audio file!"

    try:
        wav2lip_path = os.path.join("inputs", "image", "Wav2Lip")

        checkpoint_path = os.path.join(wav2lip_path, "checkpoints", "wav2lip_gan.pth")

        if not os.path.exists(checkpoint_path):
            print("Downloading Wav2Lip GAN checkpoint...")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            url = "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth"
            response = requests.get(url, allow_redirects=True)
            with open(checkpoint_path, "wb") as file:
                file.write(response.content)
            print("Wav2Lip GAN checkpoint downloaded")

        today = datetime.now().date()
        output_dir = os.path.join("outputs", f"FaceAnimation_{today.strftime('%Y%m%d')}")
        os.makedirs(output_dir, exist_ok=True)

        output_filename = f"face_animation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_path = os.path.join(output_dir, output_filename)

        command = f"py {os.path.join(wav2lip_path, 'inference.py')} --checkpoint_path {checkpoint_path} --face {image_path} --audio {audio_path} --outfile {output_path} --fps {fps} --pads {pads} --face_det_batch_size {face_det_batch_size} --wav2lip_batch_size {wav2lip_batch_size} --resize_factor {resize_factor} --crop {crop} --box {-1}"

        subprocess.run(command, shell=True, check=True)

        if stop_signal:
            return None, "Generation stopped"

        return output_path, None

    except Exception as e:
        return None, str(e)


def generate_image_txt2img(prompt, negative_prompt, stable_diffusion_model_name, vae_model_name, lora_model_names, textual_inversion_model_names, stable_diffusion_settings_html,
                           stable_diffusion_model_type, stable_diffusion_sampler, stable_diffusion_steps,
                           stable_diffusion_cfg, stable_diffusion_width, stable_diffusion_height,
                           stable_diffusion_clip_skip, enable_upscale=False, upscale_factor="x2", upscale_steps=50, upscale_cfg=6, output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    if not stable_diffusion_model_name:
        return None, "Please, select a StableDiffusion model!"

    stable_diffusion_model_path = os.path.join("inputs", "image", "sd_models",
                                               f"{stable_diffusion_model_name}.safetensors")

    if not os.path.exists(stable_diffusion_model_path):
        return None, f"StableDiffusion model not found: {stable_diffusion_model_path}"

    try:
        if stable_diffusion_model_type == "SD":
            original_config_file = "configs/sd/v1-inference.yaml"
            vae_config_file = "configs/sd/v1-inference.yaml"
            stable_diffusion_model = StableDiffusionPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                original_config_file=original_config_file, torch_dtype=torch.float16, variant="fp16"
            )
        elif stable_diffusion_model_type == "SD2":
            original_config_file = "configs/sd/v2-inference.yaml"
            vae_config_file = "configs/sd/v2-inference.yaml"
            stable_diffusion_model = StableDiffusionPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                original_config_file=original_config_file, torch_dtype=torch.float16, variant="fp16"
            )
        elif stable_diffusion_model_type == "SDXL":
            original_config_file = "configs/sd/sd_xl_base.yaml"
            vae_config_file = "configs/sd/sd_xl_base.yaml"
            stable_diffusion_model = StableDiffusionXLPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto", attention_slice=1,
                original_config_file=original_config_file, torch_dtype=torch.float16, variant="fp16"
            )
        else:
            return None, "Invalid StableDiffusion model type!"
    except (ValueError, KeyError):
        return None, "The selected model is not compatible with the chosen model type"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if XFORMERS_AVAILABLE:
        stable_diffusion_model.enable_xformers_memory_efficient_attention(attention_op=None)
        stable_diffusion_model.vae.enable_xformers_memory_efficient_attention(attention_op=None)
        stable_diffusion_model.unet.enable_xformers_memory_efficient_attention(attention_op=None)

    stable_diffusion_model.to(device)
    stable_diffusion_model.text_encoder.to(device)
    stable_diffusion_model.vae.to(device)
    stable_diffusion_model.unet.to(device)

    stable_diffusion_model.safety_checker = None

    if vae_model_name is not None:
        vae_model_path = os.path.join("inputs", "image", "sd_models", "vae", f"{vae_model_name}.safetensors")
        if os.path.exists(vae_model_path):
            vae = AutoencoderKL.from_single_file(vae_model_path, device_map="auto",
                                                 original_config_file=vae_config_file, torch_dtype=torch.float16,
                                                 variant="fp16")
            stable_diffusion_model.vae = vae.to(device)

    if lora_model_names is not None:
        for lora_model_name in lora_model_names:
            lora_model_path = os.path.join("inputs", "image", "sd_models", "lora", lora_model_name)
            stable_diffusion_model.load_lora_weights(lora_model_path)

    if textual_inversion_model_names is not None:
        for textual_inversion_model_name in textual_inversion_model_names:
            textual_inversion_model_path = os.path.join("inputs", "image", "sd_models", "embedding",
                                                        textual_inversion_model_name)
            if os.path.exists(textual_inversion_model_path):
                stable_diffusion_model.load_textual_inversion(textual_inversion_model_path)

    try:
        compel_proc = Compel(tokenizer=stable_diffusion_model.tokenizer,
                             text_encoder=stable_diffusion_model.text_encoder)
        prompt_embeds = compel_proc(prompt)

        images = stable_diffusion_model(prompt_embeds=prompt_embeds, negative_prompt=negative_prompt,
                                        num_inference_steps=stable_diffusion_steps,
                                        guidance_scale=stable_diffusion_cfg, height=stable_diffusion_height,
                                        width=stable_diffusion_width, clip_skip=stable_diffusion_clip_skip,
                                        sampler=stable_diffusion_sampler)
        if stop_signal:
            return None, "Generation stopped"
        image = images["images"][0]

        if enable_upscale:
            upscale_factor_value = 2 if upscale_factor == "x2" else 4
            upscaler = load_upscale_model(upscale_factor_value)
            if upscaler:
                if upscale_factor == "x2":
                    upscaled_image = upscaler(prompt=prompt, image=image, num_inference_steps=upscale_steps, guidance_scale=upscale_cfg).images[0]
                else:
                    upscaled_image = upscaler(prompt=prompt, image=image, num_inference_steps=upscale_steps, guidance_scale=upscale_cfg)["images"][0]
                image = upscaled_image

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"txt2img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        return image_path, None

    finally:
        del stable_diffusion_model
        torch.cuda.empty_cache()


def generate_image_img2img(prompt, negative_prompt, init_image,
                           strength, stable_diffusion_model_name, vae_model_name, stable_diffusion_settings_html,
                           stable_diffusion_model_type,
                           stable_diffusion_sampler, stable_diffusion_steps, stable_diffusion_cfg,
                           stable_diffusion_clip_skip, output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    if not stable_diffusion_model_name:
        return None, "Please, select a StableDiffusion model!"

    if not init_image:
        return None, "Please, upload an initial image!"

    stable_diffusion_model_path = os.path.join("inputs", "image", "sd_models",
                                               f"{stable_diffusion_model_name}.safetensors")

    if not os.path.exists(stable_diffusion_model_path):
        return None, f"StableDiffusion model not found: {stable_diffusion_model_path}"

    try:
        if stable_diffusion_model_type == "SD":
            original_config_file = "configs/sd/v1-inference.yaml"
            vae_config_file = "configs/sd/v1-inference.yaml"
            stable_diffusion_model = StableDiffusionImg2ImgPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                original_config_file=original_config_file, torch_dtype=torch.float16, variant="fp16"
            )
        elif stable_diffusion_model_type == "SD2":
            original_config_file = "configs/sd/v2-inference.yaml"
            vae_config_file = "configs/sd/v2-inference.yaml"
            stable_diffusion_model = StableDiffusionPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                original_config_file=original_config_file, torch_dtype=torch.float16, variant="fp16"
            )
        elif stable_diffusion_model_type == "SDXL":
            original_config_file = "configs/sd/sd_xl_base.yaml"
            vae_config_file = "configs/sd/sd_xl_base.yaml"
            stable_diffusion_model = StableDiffusionImg2ImgPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto", attention_slice=1,
                original_config_file=original_config_file, torch_dtype=torch.float16, variant="fp16"
            )
        else:
            return None, "Invalid StableDiffusion model type!"
    except (ValueError, KeyError):
        return None, "The selected model is not compatible with the chosen model type"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if XFORMERS_AVAILABLE:
        stable_diffusion_model.enable_xformers_memory_efficient_attention(attention_op=None)
        stable_diffusion_model.vae.enable_xformers_memory_efficient_attention(attention_op=None)
        stable_diffusion_model.unet.enable_xformers_memory_efficient_attention(attention_op=None)

    stable_diffusion_model.to(device)
    stable_diffusion_model.text_encoder.to(device)
    stable_diffusion_model.vae.to(device)
    stable_diffusion_model.unet.to(device)

    stable_diffusion_model.safety_checker = None

    if vae_model_name is not None:
        vae_model_path = os.path.join("inputs", "image", "sd_models", "vae", f"{vae_model_name}.safetensors")
        if os.path.exists(vae_model_path):
            vae = AutoencoderKL.from_single_file(vae_model_path, device_map=device,
                                                 original_config_file=vae_config_file, torch_dtype=torch.float16,
                                                 variant="fp16")
            stable_diffusion_model.vae = vae.to(device)

    try:
        init_image = Image.open(init_image).convert("RGB")
        init_image = stable_diffusion_model.image_processor.preprocess(init_image)

        compel_proc = Compel(tokenizer=stable_diffusion_model.tokenizer,
                             text_encoder=stable_diffusion_model.text_encoder)
        prompt_embeds = compel_proc(prompt)

        images = stable_diffusion_model(prompt_embeds=prompt_embeds, negative_prompt=negative_prompt,
                                        num_inference_steps=stable_diffusion_steps,
                                        guidance_scale=stable_diffusion_cfg, clip_skip=stable_diffusion_clip_skip,
                                        sampler=stable_diffusion_sampler, image=init_image, strength=strength)
        if stop_signal:
            return None, "Generation stopped"
        image = images["images"][0]

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"img2img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        return image_path, None

    except Exception as e:
        return None, str(e)

    finally:
        del stable_diffusion_model
        torch.cuda.empty_cache()


def generate_image_depth2img(prompt, negative_prompt, init_image, stable_diffusion_settings_html, strength,
                             output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    if not init_image:
        return None, "Please, upload an initial image!"

    stable_diffusion_model_path = os.path.join("inputs", "image", "sd_models", "depth")

    if not os.path.exists(stable_diffusion_model_path):
        print("Downloading depth2img model...")
        os.makedirs(stable_diffusion_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/stabilityai/stable-diffusion-2-depth", stable_diffusion_model_path)
        print("Depth2img model downloaded")

    try:
        original_config_file = "configs/sd/v2-inference.yaml"
        stable_diffusion_model = StableDiffusionDepth2ImgPipeline.from_pretrained(
            stable_diffusion_model_path, use_safetensors=True,
            original_config_file=original_config_file, torch_dtype=torch.float16, variant="fp16",
        )
    except (ValueError, KeyError):
        return None, "Failed to load the depth2img model"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if XFORMERS_AVAILABLE:
        stable_diffusion_model.enable_xformers_memory_efficient_attention(attention_op=None)
        stable_diffusion_model.vae.enable_xformers_memory_efficient_attention(attention_op=None)
        stable_diffusion_model.unet.enable_xformers_memory_efficient_attention(attention_op=None)

    stable_diffusion_model.to(device)
    stable_diffusion_model.text_encoder.to(device)
    stable_diffusion_model.vae.to(device)
    stable_diffusion_model.unet.to(device)

    stable_diffusion_model.safety_checker = None

    try:
        init_image = Image.open(init_image).convert("RGB")
        image = stable_diffusion_model(prompt=prompt, negative_prompt=negative_prompt, image=init_image, strength=strength).images[0]

        if stop_signal:
            return None, "Generation stopped"

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"depth2img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        return image_path, None

    finally:
        del stable_diffusion_model
        torch.cuda.empty_cache()


def generate_image_upscale(image_path, num_inference_steps, guidance_scale, output_format="png", stop_generation=None):
    global stop_signal
    if stop_signal:
        return None, "Generation stopped"

    if not image_path:
        return None, "Please, upload an initial image!"

    upscale_factor = 2
    upscaler = load_upscale_model(upscale_factor)
    if upscaler:
        image = Image.open(image_path).convert("RGB")
        upscaled_image = upscaler(prompt="", image=image, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"upscaled_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        upscaled_image.save(image_path, format=output_format.upper())

        return image_path, None
    else:
        return None, "Failed to load upscale model"


def generate_image_inpaint(prompt, negative_prompt, init_image, mask_image, stable_diffusion_model_name, vae_model_name,
                           stable_diffusion_settings_html, stable_diffusion_model_type, stable_diffusion_sampler,
                           stable_diffusion_steps, stable_diffusion_cfg, width, height, output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    if not stable_diffusion_model_name:
        return None, "Please, select a StableDiffusion model!"

    if not init_image or not mask_image:
        return None, "Please, upload an initial image and a mask image!"

    stable_diffusion_model_path = os.path.join("inputs", "image", "sd_models", "inpaint",
                                               f"{stable_diffusion_model_name}.safetensors")

    if not os.path.exists(stable_diffusion_model_path):
        return None, f"StableDiffusion model not found: {stable_diffusion_model_path}"

    try:
        if stable_diffusion_model_type == "SD":
            original_config_file = "configs/sd/v1-inference.yaml"
            vae_config_file = "configs/sd/v1-inference.yaml"
            stable_diffusion_model = StableDiffusionInpaintPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                original_config_file=original_config_file, torch_dtype=torch.float16, variant="fp16"
            )
        elif stable_diffusion_model_type == "SD2":
            original_config_file = "configs/sd/v2-inference.yaml"
            vae_config_file = "configs/sd/v2-inference.yaml"
            stable_diffusion_model = StableDiffusionInpaintPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                original_config_file=original_config_file, torch_dtype=torch.float16, variant="fp16"
            )
        elif stable_diffusion_model_type == "SDXL":
            original_config_file = "configs/sd/sd_xl_base.yaml"
            vae_config_file = "configs/sd/sd_xl_base.yaml"
            stable_diffusion_model = StableDiffusionInpaintPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto", attention_slice=1,
                original_config_file=original_config_file, torch_dtype=torch.float16, variant="fp16"
            )
        else:
            return None, "Invalid StableDiffusion model type!"
    except (ValueError, KeyError):
        return None, "The selected model is not compatible with the chosen model type"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if XFORMERS_AVAILABLE:
        stable_diffusion_model.enable_xformers_memory_efficient_attention(attention_op=None)
        stable_diffusion_model.vae.enable_xformers_memory_efficient_attention(attention_op=None)
        stable_diffusion_model.unet.enable_xformers_memory_efficient_attention(attention_op=None)

    stable_diffusion_model.to(device)
    stable_diffusion_model.text_encoder.to(device)
    stable_diffusion_model.vae.to(device)
    stable_diffusion_model.unet.to(device)

    stable_diffusion_model.safety_checker = None

    if vae_model_name is not None:
        vae_model_path = os.path.join("inputs", "image", "sd_models", "vae", f"{vae_model_name}.safetensors")
        if os.path.exists(vae_model_path):
            vae = AutoencoderKL.from_single_file(vae_model_path, device_map="auto",
                                                 original_config_file=vae_config_file, torch_dtype=torch.float16,
                                                 variant="fp16")
            stable_diffusion_model.vae = vae.to(device)

    try:
        if isinstance(mask_image, dict):
            composite_path = mask_image.get('composite', None)
            if composite_path is None:
                raise ValueError("Invalid mask image data: missing 'composite' key")

            mask_image = Image.open(composite_path).convert("L")
        elif isinstance(mask_image, str):
            mask_image = Image.open(mask_image).convert("L")
        else:
            raise ValueError("Invalid mask image format")

        init_image = Image.open(init_image).convert("RGB")

        mask_array = np.array(mask_image)
        mask_array = np.where(mask_array < 255, 0, 255).astype(np.uint8)

        mask_array = Image.fromarray(mask_array).resize(init_image.size, resample=Image.NEAREST)

        images = stable_diffusion_model(prompt=prompt, negative_prompt=negative_prompt, image=init_image,
                                        mask_image=mask_array, width=width, height=height,
                                        num_inference_steps=stable_diffusion_steps,
                                        guidance_scale=stable_diffusion_cfg, sampler=stable_diffusion_sampler)

        if stop_signal:
            return None, "Generation stopped"
        image = images["images"][0]

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"inpaint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        return image_path, None

    finally:
        del stable_diffusion_model
        torch.cuda.empty_cache()


def generate_image_gligen(prompt, gligen_phrases, gligen_boxes, stable_diffusion_model_name, stable_diffusion_settings_html,
                          stable_diffusion_model_type, stable_diffusion_sampler, stable_diffusion_steps,
                          stable_diffusion_cfg, stable_diffusion_width, stable_diffusion_height,
                          stable_diffusion_clip_skip, output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    if not stable_diffusion_model_name:
        return None, "Please, select a StableDiffusion model!"

    stable_diffusion_model_path = os.path.join("inputs", "image", "sd_models",
                                               f"{stable_diffusion_model_name}.safetensors")

    if not os.path.exists(stable_diffusion_model_path):
        return None, f"StableDiffusion model not found: {stable_diffusion_model_path}"

    try:
        if stable_diffusion_model_type == "SD":
            original_config_file = "configs/sd/v1-inference.yaml"
            stable_diffusion_model = StableDiffusionPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                original_config_file=original_config_file, torch_dtype=torch.float16, variant="fp16"
            )
        elif stable_diffusion_model_type == "SD2":
            original_config_file = "configs/sd/v2-inference.yaml"
            stable_diffusion_model = StableDiffusionPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                original_config_file=original_config_file, torch_dtype=torch.float16, variant="fp16"
            )
        elif stable_diffusion_model_type == "SDXL":
            original_config_file = "configs/sd/sd_xl_base.yaml"
            stable_diffusion_model = StableDiffusionXLPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto", attention_slice=1,
                original_config_file=original_config_file, torch_dtype=torch.float16, variant="fp16"
            )
        else:
            return None, "Invalid StableDiffusion model type!"
    except (ValueError, KeyError):
        return None, "The selected model is not compatible with the chosen model type"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if XFORMERS_AVAILABLE:
        stable_diffusion_model.enable_xformers_memory_efficient_attention(attention_op=None)
        stable_diffusion_model.vae.enable_xformers_memory_efficient_attention(attention_op=None)
        stable_diffusion_model.unet.enable_xformers_memory_efficient_attention(attention_op=None)

    stable_diffusion_model.to(device)
    stable_diffusion_model.text_encoder.to(device)
    stable_diffusion_model.vae.to(device)
    stable_diffusion_model.unet.to(device)

    stable_diffusion_model.safety_checker = None

    try:
        image = stable_diffusion_model(prompt, num_inference_steps=stable_diffusion_steps,
                                        guidance_scale=stable_diffusion_cfg, height=stable_diffusion_height,
                                        width=stable_diffusion_width, clip_skip=stable_diffusion_clip_skip,
                                        sampler=stable_diffusion_sampler)["images"][0]

        if stop_signal:
            return None, "Generation stopped"

        gligen_model_path = os.path.join("inputs", "image", "sd_models", "gligen")

        if not os.path.exists(gligen_model_path):
            print("Downloading GLIGEN model...")
            os.makedirs(gligen_model_path, exist_ok=True)
            Repo.clone_from("https://huggingface.co/masterful/gligen-1-4-inpainting-text-box", os.path.join(gligen_model_path, "inpainting"))
            print("GLIGEN model downloaded")

        gligen_boxes = json.loads(gligen_boxes)

        pipe = StableDiffusionGLIGENPipeline.from_pretrained(
            os.path.join(gligen_model_path, "inpainting"), variant="fp16", torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda")

        images = pipe(
            prompt=prompt,
            gligen_phrases=gligen_phrases,
            gligen_inpaint_image=image,
            gligen_boxes=[gligen_boxes],
            gligen_scheduled_sampling_beta=1,
            output_type="pil",
            num_inference_steps=stable_diffusion_steps,
        ).images

        if stop_signal:
            return None, "Generation stopped"

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"gligen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        images[0].save(image_path)

        return image_path, None

    finally:
        del stable_diffusion_model
        del pipe
        torch.cuda.empty_cache()


def generate_animation_animatediff(prompt, negative_prompt, stable_diffusion_model_name, num_frames, num_inference_steps,
                                   guidance_scale, width, height, stop_generation):
    global stop_signal
    stop_signal = False

    if not stable_diffusion_model_name:
        return None, "Please, select a StableDiffusion model!"

    if ValueError:
        return None, "You are using the wrong model. Use StableDiffusion 1.5 model"

    stable_diffusion_model_path = os.path.join("inputs", "image", "sd_models", f"{stable_diffusion_model_name}.safetensors")

    if not os.path.exists(stable_diffusion_model_path):
        return None, f"StableDiffusion model not found: {stable_diffusion_model_path}"

    motion_adapter_path = os.path.join("inputs", "image", "sd_models", "motion_adapter")
    if not os.path.exists(motion_adapter_path):
        print("Downloading motion adapter...")
        os.makedirs(motion_adapter_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-2", motion_adapter_path)
        print("Motion adapter downloaded")

    try:
        adapter = MotionAdapter.from_pretrained(motion_adapter_path, torch_dtype=torch.float16)
        original_config_file = "configs/sd/v1-inference.yaml"
        stable_diffusion_model = StableDiffusionPipeline.from_single_file(
            stable_diffusion_model_path,
            torch_dtype=torch.float16,
            variant="fp16",
            original_config_file=original_config_file,
            device_map="auto",
        )

        scheduler_config_path = "configs/sd/animatediff/scheduler_config.json"
        scheduler = DDIMScheduler.from_pretrained(
            scheduler_config_path,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )

        pipe = AnimateDiffPipeline(
            unet=stable_diffusion_model.unet,
            text_encoder=stable_diffusion_model.text_encoder,
            vae=stable_diffusion_model.vae,
            motion_adapter=adapter,
            tokenizer=stable_diffusion_model.tokenizer,
            feature_extractor=stable_diffusion_model.feature_extractor,
            scheduler=scheduler,
        )

        pipe.enable_vae_slicing()
        pipe.enable_model_cpu_offload()

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(-1),
            width=width,
            height=height,
        )

        if stop_signal:
            return None, "Generation stopped"

        frames = output.frames[0]

        today = datetime.now().date()
        output_dir = os.path.join('outputs', f"AnimateDiff_{today.strftime('%Y%m%d')}")
        os.makedirs(output_dir, exist_ok=True)

        gif_filename = f"animatediff_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
        gif_path = os.path.join(output_dir, gif_filename)
        export_to_gif(frames, gif_path)

        return gif_path, None

    finally:
        try:
            del pipe
            del adapter
            del stable_diffusion_model
        except UnboundLocalError:
            pass
        torch.cuda.empty_cache()


def generate_video(init_image, output_format, video_settings_html, motion_bucket_id, noise_aug_strength, fps, num_frames, decode_chunk_size,
                   iv2gen_xl_settings_html, prompt, negative_prompt, num_inference_steps, guidance_scale, stop_generation):
    global stop_signal
    stop_signal = False

    if not init_image:
        return None, None, "Please upload an initial image!"

    today = datetime.now().date()
    video_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
    os.makedirs(video_dir, exist_ok=True)

    if output_format == "mp4":
        video_model_name = "vdo/stable-video-diffusion-img2vid-xt-1-1"
        video_model_path = os.path.join("inputs", "image", "sd_models", "video", "SVD")

        print(f"Downloading StableVideoDiffusion model")

        if not os.path.exists(video_model_path):
            os.makedirs(video_model_path, exist_ok=True)
            Repo.clone_from(f"https://huggingface.co/{video_model_name}", video_model_path)

        print(f"StableVideoDiffusion model downloaded")

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path=video_model_path,
                torch_dtype=torch.float16,
                variant="fp16"
            )
            pipe.to(device)
            pipe.enable_model_cpu_offload()
            pipe.unet.enable_forward_chunking()

            image = load_image(init_image)
            image = image.resize((1024, 576))

            generator = torch.manual_seed(42)
            frames = pipe(image, decode_chunk_size=decode_chunk_size, generator=generator,
                          motion_bucket_id=motion_bucket_id, noise_aug_strength=noise_aug_strength, num_frames=num_frames).frames[0]

            if stop_signal:
                return None, None, "Generation stopped"

            video_filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            video_path = os.path.join(video_dir, video_filename)
            export_to_video(frames, video_path, fps=fps)

            return video_path, None, None

        finally:
            try:
                del pipe
            except UnboundLocalError:
                pass
            torch.cuda.empty_cache()

    elif output_format == "gif":
        video_model_name = "ali-vilab/i2vgen-xl"
        video_model_path = os.path.join("inputs", "image", "sd_models", "video", "i2vgenxl")

        print(f"Downloading i2vgen-xl model")

        if not os.path.exists(video_model_path):
            os.makedirs(video_model_path, exist_ok=True)
            Repo.clone_from(f"https://huggingface.co/{video_model_name}", video_model_path)

        print(f"i2vgen-xl model downloaded")

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipeline = I2VGenXLPipeline.from_pretrained(video_model_path, torch_dtype=torch.float16, variant="fp16")
            pipeline.to(device)
            pipeline.enable_model_cpu_offload()

            image = load_image(init_image).convert("RGB")

            generator = torch.manual_seed(8888)

            frames = pipeline(
                prompt=prompt,
                image=image,
                num_inference_steps=num_inference_steps,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                generator=generator
            ).frames[0]

            if stop_signal:
                return None, None, "Generation stopped"

            video_filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
            video_path = os.path.join(video_dir, video_filename)
            export_to_gif(frames, video_path)

            return None, video_path, None

        finally:
            try:
                del pipeline
            except UnboundLocalError:
                pass
            torch.cuda.empty_cache()


def generate_image_cascade(prompt, negative_prompt, stable_cascade_settings_html, width, height, prior_steps, prior_guidance_scale,
                           decoder_steps, decoder_guidance_scale, output_format="png",
                           stop_generation=None):
    global stop_signal
    stop_signal = False

    if not prompt:
        return None, "Please enter a prompt!"

    stable_cascade_model_path = os.path.join("inputs", "image", "sd_models", "cascade")

    if not os.path.exists(stable_cascade_model_path):
        print("Downloading Stable Cascade models...")
        os.makedirs(stable_cascade_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/stabilityai/stable-cascade-prior",
                        os.path.join(stable_cascade_model_path, "prior"))
        Repo.clone_from("https://huggingface.co/stabilityai/stable-cascade",
                        os.path.join(stable_cascade_model_path, "decoder"))
        print("Stable Cascade models downloaded")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        prior = StableCascadePriorPipeline.from_pretrained(os.path.join(stable_cascade_model_path, "prior"),
                                                           variant="bf16", torch_dtype=torch.bfloat16).to(device)
        decoder = StableCascadeDecoderPipeline.from_pretrained(os.path.join(stable_cascade_model_path, "decoder"),
                                                               variant="bf16", torch_dtype=torch.float16).to(device)
    except (ValueError, OSError):
        return None, "Failed to load the Stable Cascade models"

    prior.enable_model_cpu_offload()
    decoder.enable_model_cpu_offload()

    try:
        prior_output = prior(
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            guidance_scale=prior_guidance_scale,
            num_images_per_prompt=1,
            num_inference_steps=prior_steps
        )

        if stop_signal:
            return None, "Generation stopped"

        decoder_output = decoder(
            image_embeddings=prior_output.image_embeddings.to(torch.float16),
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=decoder_guidance_scale,
            output_type="pil",
            num_inference_steps=decoder_steps
        ).images[0]

        if stop_signal:
            return None, "Generation stopped"

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"cascade_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        decoder_output.save(image_path)

        return image_path, None

    finally:
        del prior
        del decoder
        torch.cuda.empty_cache()


def generate_image_extras(input_image, image_output_format, remove_background, stop_generation):
    if not input_image:
        return None, "Please upload an image file!"

    if not remove_background:
        return None, "Please choose the option to modify the image"

    today = datetime.now().date()
    output_dir = os.path.join('outputs', f"Extras_{today.strftime('%Y%m%d')}")
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"background_removed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{image_output_format}"
    output_path = os.path.join(output_dir, output_filename)

    try:
        remove_bg(input_image, output_path)
        return output_path, None

    except Exception as e:
        return None, str(e)


def generate_video_zeroscope2(prompt, video_to_enhance, strength, num_inference_steps, width, height, num_frames,
                              enable_video_enhance, stop_generation):
    global stop_signal
    stop_signal = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model_path = os.path.join("inputs", "image", "zeroscope2", "zeroscope_v2_576w")
    if not os.path.exists(base_model_path):
        print("Downloading ZeroScope 2 base model...")
        os.makedirs(base_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/cerspense/zeroscope_v2_576w", base_model_path)
        print("ZeroScope 2 base model downloaded")

    enhance_model_path = os.path.join("inputs", "image", "zeroscope2", "zeroscope_v2_XL")
    if not os.path.exists(enhance_model_path):
        print("Downloading ZeroScope 2 enhance model...")
        os.makedirs(enhance_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/cerspense/zeroscope_v2_XL", enhance_model_path)
        print("ZeroScope 2 enhance model downloaded")

    today = datetime.now().date()
    video_dir = os.path.join('outputs', f"ZeroScope2_{today.strftime('%Y%m%d')}")
    os.makedirs(video_dir, exist_ok=True)

    if enable_video_enhance:
        if not video_to_enhance:
            return None, "Please upload a video to enhance."

        try:
            enhance_pipe = DiffusionPipeline.from_pretrained(enhance_model_path, torch_dtype=torch.float16)
            enhance_pipe.to(device)
            enhance_pipe.scheduler = DPMSolverMultistepScheduler.from_config(enhance_pipe.scheduler.config)
            enhance_pipe.enable_model_cpu_offload()
            enhance_pipe.enable_vae_slicing()

            video = imageio.get_reader(video_to_enhance)
            frames = []
            for frame in video:
                frames.append(Image.fromarray(frame).resize((1024, 576)))

            video_frames = enhance_pipe(prompt, video=frames, strength=strength).frames

            if stop_signal:
                return None, "Generation stopped"

            video_filename = f"zeroscope2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            video_path = os.path.join(video_dir, video_filename)
            export_to_video(video_frames, video_path)

            return video_path, None

        finally:
            try:
                del enhance_pipe
            except UnboundLocalError:
                pass
            torch.cuda.empty_cache()

    else:
        try:
            base_pipe = DiffusionPipeline.from_pretrained(base_model_path, torch_dtype=torch.float16)
            base_pipe.scheduler = DPMSolverMultistepScheduler.from_config(base_pipe.scheduler.config)
            base_pipe.to(device)
            base_pipe.enable_model_cpu_offload()
            base_pipe.enable_vae_slicing()
            base_pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)

            video_frames = base_pipe(prompt, num_inference_steps=num_inference_steps, width=width, height=height, num_frames=num_frames).frames[0]

            if stop_signal:
                return None, "Generation stopped"

            video_filename = f"zeroscope2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            video_path = os.path.join(video_dir, video_filename)
            export_to_video(video_frames, video_path)

            return video_path, None

        finally:
            try:
                del base_pipe
            except UnboundLocalError:
                pass
            torch.cuda.empty_cache()


def generate_3d_triposr(image, mc_resolution, foreground_ratio=0.85, output_format="obj", stop_generation=None):
    global stop_signal
    stop_signal = False

    model_path = os.path.join("inputs", "image", "triposr")

    if not os.path.exists(model_path):
        print("Downloading TripoSR model...")
        os.makedirs(model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/stabilityai/TripoSR", model_path)
        print("TripoSR model downloaded")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TSR.from_pretrained(
        model_path,
        config_name="config.yaml",
        weight_name="model.ckpt",
    )

    model.renderer.set_chunk_size(8192)
    model.to(device)

    try:
        def fill_background(image):
            image = np.array(image).astype(np.float32) / 255.0
            image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
            image = Image.fromarray((image * 255.0).astype(np.uint8))
            return image

        image_without_background = remove(image)
        image_without_background = resize_foreground(image_without_background, foreground_ratio)
        image_without_background = fill_background(image_without_background)

        processed_image = model.image_processor(image_without_background, model.cfg.cond_image_size)[0].to(device)

        scene_codes = model(processed_image, device=device)
        mesh = model.extract_mesh(scene_codes, resolution=mc_resolution)[0]
        mesh = to_gradio_3d_orientation(mesh)

        today = datetime.now().date()
        output_dir = os.path.join('outputs', f"TripoSR_{today.strftime('%Y%m%d')}")
        os.makedirs(output_dir, exist_ok=True)

        if output_format == "obj":
            output_filename = f"3d_object_{datetime.now().strftime('%Y%m%d_%H%M%S')}.obj"
            output_path = os.path.join(output_dir, output_filename)
            mesh.export(output_path)
        else:
            output_filename = f"3d_object_{datetime.now().strftime('%Y%m%d_%H%M%S')}.glb"
            output_path = os.path.join(output_dir, output_filename)
            mesh.export(output_path)

        return output_path, None

    finally:
        del model
        torch.cuda.empty_cache()


def generate_3d(prompt, init_image, num_inference_steps, guidance_scale, frame_size, stop_generation):
    global stop_signal
    stop_signal = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if init_image:
        model_name = "openai/shap-e-img2img"
        model_path = os.path.join("inputs", "image", "shap-e", "img2img")
        if not os.path.exists(model_path):
            print("Downloading Shap-E img2img model...")
            os.makedirs(model_path, exist_ok=True)
            Repo.clone_from(f"https://huggingface.co/{model_name}", model_path)
            print("Shap-E img2img model downloaded")

        pipe = ShapEImg2ImgPipeline.from_pretrained(model_path, torch_dtype=torch.float16, variant="fp16").to(device)
        image = Image.open(init_image).resize((256, 256))
        images = pipe(
            image,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            frame_size=frame_size,
            output_type="mesh",
        ).images
    else:
        model_name = "openai/shap-e"
        model_path = os.path.join("inputs", "image", "shap-e", "text2img")
        if not os.path.exists(model_path):
            print("Downloading Shap-E text2img model...")
            os.makedirs(model_path, exist_ok=True)
            Repo.clone_from(f"https://huggingface.co/{model_name}", model_path)
            print("Shap-E text2img model downloaded")

        pipe = ShapEPipeline.from_pretrained(model_path, torch_dtype=torch.float16, variant="fp16").to(device)
        images = pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            frame_size=frame_size,
            output_type="mesh",
        ).images

    if stop_signal:
        return None, "Generation stopped"

    today = datetime.now().date()
    output_dir = os.path.join('outputs', f"Shap-E_{today.strftime('%Y%m%d')}")
    os.makedirs(output_dir, exist_ok=True)

    ply_filename = f"3d_object_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ply"
    ply_path = os.path.join(output_dir, ply_filename)
    export_to_ply(images[0], ply_path)

    mesh = trimesh.load(ply_path)
    rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    mesh = mesh.apply_transform(rot)
    glb_filename = f"3d_object_{datetime.now().strftime('%Y%m%d_%H%M%S')}.glb"
    glb_path = os.path.join(output_dir, glb_filename)
    mesh.export(glb_path, file_type="glb")

    return glb_path, None


def generate_audio_audiocraft(prompt, input_audio=None, model_name=None, audiocraft_settings_html=None, model_type="musicgen",
                              duration=10, top_k=250, top_p=0.0,
                              temperature=1.0, cfg_coef=3.0, enable_multiband=False, output_format="mp3", stop_generation=None):
    global audiocraft_model_path, stop_signal
    stop_signal = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not model_name:
        return None, "Please, select an AudioCraft model!"

    if enable_multiband and model_type in ["audiogen", "magnet"]:
        return None, "Multiband Diffusion is not supported with 'audiogen' or 'magnet' model types. Please select 'musicgen' or disable Multiband Diffusion"

    if model_type == "magnet":
        return None, "The 'magnet' model type is currently not supported, but it will be available in a future update. Please select another model type for now"

    if not audiocraft_model_path:
        audiocraft_model_path = load_audiocraft_model(model_name)

    today = datetime.now().date()
    audio_dir = os.path.join('outputs', f"AudioCraft_{today.strftime('%Y%m%d')}")
    os.makedirs(audio_dir, exist_ok=True)

    try:
        if model_type == "musicgen":
            model = MusicGen.get_pretrained(audiocraft_model_path).to(device)
            model.set_generation_params(duration=duration)
        elif model_type == "audiogen":
            model = AudioGen.get_pretrained(audiocraft_model_path).to(device)
            model.set_generation_params(duration=duration)
        #        elif model_type == "magnet":
        #            model = MAGNeT.get_pretrained(audiocraft_model_path).to(device)
        #            model.set_generation_params()
        else:
            return None, "Invalid model type!"
    except (ValueError, AssertionError):
        return None, "The selected model is not compatible with the chosen model type"

    mbd = None

    if enable_multiband:
        mbd = MultiBandDiffusion.get_mbd_musicgen().to(device)

    try:
        progress_bar = tqdm(total=duration, desc="Generating audio")
        if input_audio and model_type == "musicgen":
            audio_path = input_audio
            melody, sr = torchaudio.load(audio_path)
            model.set_generation_params(duration=duration, top_k=top_k, top_p=top_p, temperature=temperature,
                                        cfg_coef=cfg_coef)
            wav, tokens = model.generate_with_chroma([prompt], melody[None].expand(1, -1, -1), sr, return_tokens=True)
            progress_bar.update(duration)
            if wav.ndim > 2:
                wav = wav.squeeze()
            if stop_signal:
                return None, "Generation stopped"
        else:
            descriptions = [prompt]
            model.set_generation_params(duration=duration, top_k=top_k, top_p=top_p, temperature=temperature,
                                        cfg_coef=cfg_coef)
            if model_type == "musicgen":
                wav, tokens = model.generate(descriptions, return_tokens=True)
            elif model_type == "audiogen":
                wav = model.generate(descriptions)
            progress_bar.update(duration)
            if wav.ndim > 2:
                wav = wav.squeeze()
            if stop_signal:
                return None, "Generation stopped"
        progress_bar.close()

        if mbd:
            if stop_signal:
                return None, "Generation stopped"
            tokens = rearrange(tokens, "b n d -> n b d")
            wav_diffusion = mbd.tokens_to_wav(tokens)
            wav_diffusion = wav_diffusion.squeeze()
            if wav_diffusion.ndim == 1:
                wav_diffusion = wav_diffusion.unsqueeze(0)
            max_val = wav_diffusion.abs().max()
            if max_val > 1:
                wav_diffusion = wav_diffusion / max_val
            wav_diffusion = wav_diffusion * 0.99
            audio_filename_diffusion = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}_diffusion.wav"
            audio_path_diffusion = os.path.join(audio_dir, audio_filename_diffusion)
            torchaudio.save(audio_path_diffusion, wav_diffusion.cpu().detach(), model.sample_rate)

        audio_filename = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        audio_path = os.path.join(audio_dir, audio_filename)
        if output_format == "mp3":
            audio_write(audio_path, wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True,
                        format='mp3')
        elif output_format == "ogg":
            audio_write(audio_path, wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True,
                        format='ogg')
        else:
            audio_write(audio_path, wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

        if output_format == "mp3":
            return audio_path + ".mp3", None
        elif output_format == "ogg":
            return audio_path + ".ogg", None
        else:
            return audio_path + ".wav", None

    finally:
        del model
        if mbd:
            del mbd
        torch.cuda.empty_cache()


def generate_audio_audioldm2(prompt, negative_prompt, model_name, num_inference_steps, audio_length_in_s,
                             num_waveforms_per_prompt, output_format, stop_generation):
    global stop_signal
    stop_signal = False

    if not model_name:
        return None, "Please, select an AudioLDM 2 model!"

    model_path = os.path.join("inputs", "audio", "audioldm2", model_name)

    if not os.path.exists(model_path):
        print(f"Downloading AudioLDM 2 model: {model_name}...")
        os.makedirs(model_path, exist_ok=True)
        Repo.clone_from(f"https://huggingface.co/{model_name}", model_path)
        print(f"AudioLDM 2 model {model_name} downloaded")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = AudioLDM2Pipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    generator = torch.Generator(device).manual_seed(0)

    try:
        audio = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            audio_length_in_s=audio_length_in_s,
            num_waveforms_per_prompt=num_waveforms_per_prompt,
            generator=generator,
        ).audios

        if stop_signal:
            return None, "Generation stopped"

        today = datetime.now().date()
        audio_dir = os.path.join('outputs', f"AudioLDM2_{today.strftime('%Y%m%d')}")
        os.makedirs(audio_dir, exist_ok=True)
        audio_filename = f"audioldm2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        audio_path = os.path.join(audio_dir, audio_filename)

        if output_format == "mp3":
            scipy.io.wavfile.write(audio_path, rate=16000, data=audio[0])
        elif output_format == "ogg":
            scipy.io.wavfile.write(audio_path, rate=16000, data=audio[0])
        else:
            scipy.io.wavfile.write(audio_path, rate=16000, data=audio[0])

        return audio_path, None

    finally:
        del pipe
        torch.cuda.empty_cache()


def demucs_separate(audio_file, output_format="wav"):
    global stop_signal
    if stop_signal:
        return None, None, "Generation stopped"

    if not audio_file:
        return None, None, "Please upload an audio file!"

    today = datetime.now().date()
    demucs_dir = os.path.join("outputs", f"Demucs_{today.strftime('%Y%m%d')}")
    os.makedirs(demucs_dir, exist_ok=True)

    now = datetime.now()
    separate_dir = os.path.join(demucs_dir, f"separate_{now.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(separate_dir, exist_ok=True)

    try:
        command = f"demucs --two-stems=vocals {audio_file} -o {separate_dir}"
        subprocess.run(command, shell=True, check=True)

        if stop_signal:
            return None, None, "Generation stopped"

        temp_vocal_file = os.path.join(separate_dir, "htdemucs", os.path.splitext(os.path.basename(audio_file))[0], "vocals.wav")
        temp_instrumental_file = os.path.join(separate_dir, "htdemucs", os.path.splitext(os.path.basename(audio_file))[0], "no_vocals.wav")

        vocal_file = os.path.join(separate_dir, "vocals.wav")
        instrumental_file = os.path.join(separate_dir, "instrumental.wav")

        os.rename(temp_vocal_file, vocal_file)
        os.rename(temp_instrumental_file, instrumental_file)

        if output_format == "mp3":
            vocal_output = os.path.join(separate_dir, "vocal.mp3")
            instrumental_output = os.path.join(separate_dir, "instrumental.mp3")
            subprocess.run(f"ffmpeg -i {vocal_file} -b:a 192k {vocal_output}", shell=True, check=True)
            subprocess.run(f"ffmpeg -i {instrumental_file} -b:a 192k {instrumental_output}", shell=True, check=True)
        elif output_format == "ogg":
            vocal_output = os.path.join(separate_dir, "vocal.ogg")
            instrumental_output = os.path.join(separate_dir, "instrumental.ogg")
            subprocess.run(f"ffmpeg -i {vocal_file} -c:a libvorbis -qscale:a 5 {vocal_output}", shell=True, check=True)
            subprocess.run(f"ffmpeg -i {instrumental_file} -c:a libvorbis -qscale:a 5 {instrumental_output}", shell=True, check=True)
        else:
            vocal_output = vocal_file
            instrumental_output = instrumental_file

        return vocal_output, instrumental_output, None

    except Exception as e:
        return None, None, str(e)


def download_model(model_name_llm, model_name_sd):
    if not model_name_llm and not model_name_sd:
        return "Please select a model to download"

    if model_name_llm and model_name_sd:
        return "Please select one model type for downloading"

    if model_name_llm:
        model_url = ""
        if model_name_llm == "Phi3(Transformers3B)":
            model_url = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct"
        elif model_name_llm == "OpenChat(Llama7B.Q4)":
            model_url = "https://huggingface.co/TheBloke/openchat-3.5-0106-GGUF/resolve/main/openchat-3.5-0106.Q4_K_M.gguf"
        model_path = os.path.join("inputs", "text", "llm_models", model_name_llm)

        if model_url:
            if model_name_llm == "Phi3(Transformers3B)":
                Repo.clone_from(model_url, model_path)
            else:
                response = requests.get(model_url, allow_redirects=True)
                with open(model_path, "wb") as file:
                    file.write(response.content)
            return f"LLM model {model_name_llm} downloaded successfully!"
        else:
            return "Invalid LLM model name"

    if model_name_sd:
        model_url = ""
        if model_name_sd == "Dreamshaper8(SD1.5)":
            model_url = "https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors"
        elif model_name_sd == "RealisticVisionV4.0(SDXL)":
            model_url = "https://huggingface.co/SG161222/RealVisXL_V4.0/resolve/main/RealVisXL_V4.0.safetensors"
        model_path = os.path.join("inputs", "image", "sd_models", f"{model_name_sd}.safetensors")

        if model_url:
            response = requests.get(model_url, allow_redirects=True)
            with open(model_path, "wb") as file:
                file.write(response.content)
            return f"StableDiffusion model {model_name_sd} downloaded successfully!"
        else:
            return "Invalid StableDiffusion model name"


def settings_interface(share_value):
    global share_mode
    share_mode = share_value == "True"
    message = f"Settings updated successfully!"

    stop_all_processes()

    app.launch(share=share_mode, server_name="localhost")

    return message


share_mode = False


def get_system_info():
    gpu = GPUtil.getGPUs()[0]
    gpu_total_memory = f"{gpu.memoryTotal} MB"
    gpu_used_memory = f"{gpu.memoryUsed} MB"
    gpu_free_memory = f"{gpu.memoryFree} MB"

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    gpu_temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)

    cpu_info = get_cpu_info()
    cpu_temp = cpu_info.get("cpu_temp", None)

    ram = psutil.virtual_memory()
    ram_total = f"{ram.total // (1024 ** 3)} GB"
    ram_used = f"{ram.used // (1024 ** 3)} GB"
    ram_free = f"{ram.available // (1024 ** 3)} GB"

    return gpu_total_memory, gpu_used_memory, gpu_free_memory, gpu_temp, cpu_temp, ram_total, ram_used, ram_free


def stop_all_processes():
    global stop_signal
    stop_signal = True


def close_terminal():
    os._exit(1)


def open_outputs_folder():
    outputs_folder = "outputs"
    if os.path.exists(outputs_folder):
        if os.name == "nt":
            os.startfile(outputs_folder)
        else:
            os.system(f'open "{outputs_folder}"' if os.name == "darwin" else f'xdg-open "{outputs_folder}"')


llm_models_list = [None, "moondream2"] + [model for model in os.listdir("inputs/text/llm_models") if not model.endswith(".txt") and model != "vikhyatk"]
speaker_wavs_list = [None] + [wav for wav in os.listdir("inputs/audio/voices") if not wav.endswith(".txt")]
stable_diffusion_models_list = [None] + [model.replace(".safetensors", "") for model in
                                         os.listdir("inputs/image/sd_models")
                                         if (model.endswith(".safetensors") or not model.endswith(".txt") and not os.path.isdir(os.path.join("inputs/image/sd_models")))]
audiocraft_models_list = [None] + ["musicgen-stereo-medium", "audiogen-medium", "musicgen-stereo-melody", "musicgen-medium", "musicgen-melody", "musicgen-large",
                                   "hybrid-magnet-medium", "magnet-medium-30sec", "magnet-medium-10sec", "audio-magnet-medium"]
vae_models_list = [None] + [model.replace(".safetensors", "") for model in os.listdir("inputs/image/sd_models/vae") if
                            model.endswith(".safetensors") or not model.endswith(".txt")]
lora_models_list = [None] + [model for model in os.listdir("inputs/image/sd_models/lora") if
                             model.endswith(".safetensors")]
textual_inversion_models_list = [None] + [model for model in os.listdir("inputs/image/sd_models/embedding") if model.endswith(".pt")]
inpaint_models_list = [None] + [model.replace(".safetensors", "") for model in
                                os.listdir("inputs/image/sd_models/inpaint")
                                if model.endswith(".safetensors") or not model.endswith(".txt")]

chat_interface = gr.Interface(
    fn=generate_text_and_speech,
    inputs=[
        gr.Textbox(label="Enter your request"),
        gr.Audio(type="filepath", label="Record your request (optional)"),
        gr.Image(label="Upload your image (optional)", type="filepath"),
        gr.Dropdown(choices=llm_models_list, label="Select LLM model", value=None),
        gr.HTML("<h3>LLM Settings</h3>"),
        gr.Radio(choices=["transformers", "llama"], label="Select model type", value="transformers"),
        gr.Slider(minimum=1, maximum=4096, value=512, step=1, label="Max length (for transformers type models)"),
        gr.Slider(minimum=1, maximum=4096, value=512, step=1, label="Max tokens (for llama type models)"),
        gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.1, label="Top P"),
        gr.Slider(minimum=0, maximum=100, value=20, step=1, label="Top K"),
        gr.Radio(choices=["txt", "json"], label="Select chat history format", value="txt", interactive=True),
        gr.Checkbox(label="Enable WebSearch", value=False),
        gr.Checkbox(label="Enable LibreTranslate", value=False),
        gr.Dropdown(choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hi"], label="Select target language", value="ru", interactive=True),
        gr.Checkbox(label="Enable Multimodal", value=False),
        gr.Checkbox(label="Enable TTS", value=False),
        gr.HTML("<h3>TTS Settings</h3>"),
        gr.Dropdown(choices=speaker_wavs_list, label="Select voice", interactive=True),
        gr.Dropdown(choices=["en", "ru"], label="Select language", interactive=True),
        gr.Slider(minimum=0.0, maximum=1.9, value=1.0, step=0.1, label="TTS Temperature", interactive=True),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.1, label="TTS Top P", interactive=True),
        gr.Slider(minimum=0, maximum=100, value=20, step=1, label="TTS Top K", interactive=True),
        gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="TTS Speed", interactive=True),
        gr.Radio(choices=["wav", "mp3", "ogg"], label="Select output format", value="wav", interactive=True),
        gr.Button(value="Stop generation", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Chatbot(label="LLM text response", value=[]),
        gr.Audio(label="LLM audio response", type="filepath"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - LLM",
    description="This user interface allows you to enter any text or audio and receive generated response. You can select the LLM model, "
                "avatar, voice and language for tts from the drop-down lists. You can also customize the model settings from the sliders. "
                "Try it and see what happens!",
    allow_flagging="never",
)

tts_stt_interface = gr.Interface(
    fn=generate_tts_stt,
    inputs=[
        gr.Textbox(label="Enter text for TTS"),
        gr.Audio(label="Record audio for STT", type="filepath"),
        gr.HTML("<h3>TTS Settings</h3>"),
        gr.Dropdown(choices=speaker_wavs_list, label="Select voice", interactive=True),
        gr.Dropdown(choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"], label="Select language", interactive=True),
        gr.Slider(minimum=0.0, maximum=1.9, value=1.0, step=0.1, label="TTS Temperature", interactive=True),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.1, label="TTS Top P", interactive=True),
        gr.Slider(minimum=0, maximum=100, value=20, step=1, label="TTS Top K", interactive=True),
        gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="TTS Speed", interactive=True),
        gr.Radio(choices=["wav", "mp3", "ogg"], label="Select TTS output format", value="wav", interactive=True),
        gr.Dropdown(choices=["txt", "json"], label="Select STT output format", value="txt", interactive=True),
    ],
    outputs=[
        gr.Audio(label="TTS Audio", type="filepath"),
        gr.Textbox(label="STT Text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - TTS-STT",
    description="This user interface allows you to enter text for Text-to-Speech(CoquiTTS) and record audio for Speech-to-Text(OpenAIWhisper). "
                "For TTS, you can select the voice and language, and customize the generation settings from the sliders. "
                "For STT, simply record your audio and the spoken text will be displayed. "
                "Try it and see what happens!",
    allow_flagging="never",
)

bark_interface = gr.Interface(
    fn=generate_bark_audio,
    inputs=[
        gr.Textbox(label="Enter text for the request"),
        gr.Dropdown(choices=[None, "v2/en_speaker_1", "v2/ru_speaker_1"], label="Select voice preset", value=None),
        gr.Slider(minimum=1, maximum=1000, value=100, step=1, label="Max length"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.4, step=0.1, label="Fine temperature"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Coarse temperature"),
        gr.Radio(choices=["wav", "mp3", "ogg"], label="Select output format", value="wav", interactive=True),
        gr.Button(value="Stop generation", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Audio(label="Generated audio", type="filepath"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - SunoBark",
    description="This user interface allows you to enter text and generate audio using SunoBark. "
                "You can select the voice preset and customize the max length. "
                "Try it and see what happens!",
    allow_flagging="never",
)

translate_interface = gr.Interface(
    fn=translate_text,
    inputs=[
        gr.Textbox(label="Enter text to translate"),
        gr.Dropdown(choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hi"], label="Select source language", value="en"),
        gr.Dropdown(choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hi"], label="Select target language", value="ru"),
        gr.Checkbox(label="Enable translate history save", value=False),
        gr.Radio(choices=["txt", "json"], label="Select translate history format", value="txt", interactive=True),
        gr.File(label="Upload text file (optional)", file_count="single", interactive=True),
    ],
    outputs=[
        gr.Textbox(label="Translated text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - LibreTranslate",
    description="This user interface allows you to enter text and translate it using LibreTranslate. "
                "Select the source and target languages and click Submit to get the translation. "
                "Try it and see what happens!",
    allow_flagging="never",
)

wav2lip_interface = gr.Interface(
    fn=generate_wav2lip,
    inputs=[
        gr.Image(label="Input image", type="filepath"),
        gr.Audio(label="Input audio", type="filepath"),
        gr.Slider(minimum=1, maximum=60, value=30, step=1, label="FPS"),
        gr.Textbox(label="Pads", value="0 10 0 0"),
        gr.Slider(minimum=1, maximum=64, value=16, step=1, label="Face Detection Batch Size"),
        gr.Slider(minimum=1, maximum=512, value=128, step=1, label="Wav2Lip Batch Size"),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Resize Factor"),
        gr.Textbox(label="Crop", value="0 -1 0 -1"),
    ],
    outputs=[
        gr.Video(label="Generated lip-sync"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - Wav2Lip",
    description="This user interface allows you to generate talking head videos by combining an image and an audio file using Wav2Lip. "
                "Upload an image and an audio file, and click Generate to create the talking head video. "
                "Try it and see what happens!",
    allow_flagging="never",
)

txt2img_interface = gr.Interface(
    fn=generate_image_txt2img,
    inputs=[
        gr.Textbox(label="Enter your prompt"),
        gr.Textbox(label="Enter your negative prompt", value=""),
        gr.Dropdown(choices=stable_diffusion_models_list, label="Select StableDiffusion model", value=None),
        gr.Dropdown(choices=vae_models_list, label="Select VAE model (optional)", value=None),
        gr.Dropdown(choices=lora_models_list, label="Select LORA models (optional)", value=None, multiselect=True),
        gr.Dropdown(choices=textual_inversion_models_list, label="Select Embedding models (optional)", value=None, multiselect=True),
        gr.HTML("<h3>StableDiffusion Settings</h3>"),
        gr.Radio(choices=["SD", "SD2", "SDXL"], label="Select model type", value="SD"),
        gr.Dropdown(choices=["euler_ancestral", "euler", "lms", "heun", "dpm", "dpm_solver", "dpm_solver++"],
                    label="Select sampler", value="euler_ancestral"),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Steps"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label="CFG"),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="Width"),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="Height"),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Clip skip"),
        gr.Checkbox(label="Enable upscale", value=False),
        gr.Radio(choices=["x2", "x4"], label="Upscale size", value="x2"),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Upscale steps"),
        gr.Slider(minimum=1.0, maximum=30.0, value=6, step=0.1, label="Upscale CFG"),
        gr.Radio(choices=["png", "jpeg"], label="Select output format", value="png", interactive=True),
        gr.Button(value="Stop generation", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="Generated image"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion (txt2img)",
    description="This user interface allows you to enter any text and generate images using StableDiffusion. "
                "You can select the model and customize the generation settings from the sliders. "
                "Try it and see what happens!",
    allow_flagging="never",
)

img2img_interface = gr.Interface(
    fn=generate_image_img2img,
    inputs=[
        gr.Textbox(label="Enter your prompt"),
        gr.Textbox(label="Enter your negative prompt", value=""),
        gr.Image(label="Initial image", type="filepath"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="Strength"),
        gr.Dropdown(choices=stable_diffusion_models_list, label="Select StableDiffusion model", value=None),
        gr.Dropdown(choices=vae_models_list, label="Select VAE model (optional)", value=None),
        gr.HTML("<h3>StableDiffusion Settings</h3>"),
        gr.Radio(choices=["SD", "SD2", "SDXL"], label="Select model type", value="SD"),
        gr.Dropdown(choices=["euler_ancestral", "euler", "lms", "heun", "dpm", "dpm_solver", "dpm_solver++"],
                    label="Select sampler", value="euler_ancestral"),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Steps"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label="CFG"),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Clip skip"),
        gr.Radio(choices=["png", "jpeg"], label="Select output format", value="png", interactive=True),
        gr.Button(value="Stop generation", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="Generated image"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion (img2img)",
    description="This user interface allows you to enter any text and image to generate new images using StableDiffusion. "
                "You can select the model and customize the generation settings from the sliders. "
                "Try it and see what happens!",
    allow_flagging="never",
)

depth2img_interface = gr.Interface(
    fn=generate_image_depth2img,
    inputs=[
        gr.Textbox(label="Enter your prompt"),
        gr.Textbox(label="Enter your negative prompt", value=""),
        gr.Image(label="Initial image", type="filepath"),
        gr.HTML("<h3>StableDiffusion Settings</h3>"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.01, label="Strength"),
        gr.Radio(choices=["png", "jpeg"], label="Select output format", value="png", interactive=True),
        gr.Button(value="Stop generation", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="Generated image"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion (depth2img)",
    description="This user interface allows you to enter a prompt, an initial image to generate depth-aware images using StableDiffusion. "
                "Try it and see what happens!",
    allow_flagging="never",
)

upscale_interface = gr.Interface(
    fn=generate_image_upscale,
    inputs=[
        gr.Image(label="Image to upscale", type="filepath"),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Steps"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label="CFG"),
        gr.Radio(choices=["png", "jpeg"], label="Select output format", value="png", interactive=True),
        gr.Button(value="Stop generation", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="Upscaled image"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion (upscale)",
    description="This user interface allows you to upload an image and upscale it",
    allow_flagging="never",
)

inpaint_interface = gr.Interface(
    fn=generate_image_inpaint,
    inputs=[
        gr.Textbox(label="Enter your prompt"),
        gr.Textbox(label="Enter your negative prompt", value=""),
        gr.Image(label="Initial image", type="filepath"),
        gr.ImageEditor(label="Mask image", type="filepath"),
        gr.Dropdown(choices=inpaint_models_list, label="Select Inpaint model", value=None),
        gr.Dropdown(choices=vae_models_list, label="Select VAE model (optional)", value=None),
        gr.HTML("<h3>StableDiffusion Settings</h3>"),
        gr.Radio(choices=["SD", "SD2", "SDXL"], label="Select model type", value="SD"),
        gr.Dropdown(choices=["euler_ancestral", "euler", "lms", "heun", "dpm", "dpm_solver", "dpm_solver++"],
                    label="Select sampler", value="euler_ancestral"),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Steps"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label="CFG"),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="Width"),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="Height"),
        gr.Radio(choices=["png", "jpeg"], label="Select output format", value="png", interactive=True),
        gr.Button(value="Stop generation", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="Inpainted image"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion (inpaint)",
    description="This user interface allows you to enter a prompt, an initial image, and a mask image to inpaint using StableDiffusion. "
                "You can select the model and customize the generation settings from the sliders. "
                "Try it and see what happens!",
    allow_flagging="never",
)

gligen_interface = gr.Interface(
    fn=generate_image_gligen,
    inputs=[
        gr.Textbox(label="Enter your prompt"),
        gr.Textbox(label="Enter GLIGEN phrases", value=""),
        gr.Textbox(label="Enter GLIGEN boxes", value=""),
        gr.Dropdown(choices=stable_diffusion_models_list, label="Select StableDiffusion model", value=None),
        gr.HTML("<h3>StableDiffusion Settings</h3>"),
        gr.Radio(choices=["SD", "SD2", "SDXL"], label="Select model type", value="SD"),
        gr.Dropdown(choices=["euler_ancestral", "euler", "lms", "heun", "dpm", "dpm_solver", "dpm_solver++"],
                    label="Select sampler", value="euler_ancestral"),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Steps"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label="CFG"),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="Width"),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="Height"),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Clip skip"),
        gr.Radio(choices=["png", "jpeg"], label="Select output format", value="png", interactive=True),
        gr.Button(value="Stop generation", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="Generated image"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion (gligen)",
    description="This user interface allows you to generate images using Stable Diffusion and insert objects using GLIGEN. "
                "Select the Stable Diffusion model, customize the generation settings, enter a prompt, GLIGEN phrases, and bounding boxes. "
                "Try it and see what happens!",
    allow_flagging="never",
)

animatediff_interface = gr.Interface(
    fn=generate_animation_animatediff,
    inputs=[
        gr.Textbox(label="Enter your prompt"),
        gr.Textbox(label="Enter your negative prompt", value=""),
        gr.Dropdown(choices=stable_diffusion_models_list, label="Select StableDiffusion model (only SD1.5)", value=None),
        gr.Slider(minimum=1, maximum=200, value=20, step=1, label="Frames"),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Steps"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label="Guidance Scale"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height"),
        gr.Button(value="Stop generation", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(label="Generated GIF", type="filepath"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion (animatediff)",
    description="This user interface allows you to enter a prompt and generate animated GIFs using AnimateDiff. "
                "You can select the model and customize the generation settings from the sliders. "
                "Try it and see what happens!",
    allow_flagging="never",
)

video_interface = gr.Interface(
    fn=generate_video,
    inputs=[
        gr.Image(label="Initial image", type="filepath"),
        gr.Radio(choices=["mp4", "gif"], label="Select output format", value="mp4", interactive=True),
        gr.HTML("<h3>SVD Settings (mp4)</h3>"),
        gr.Slider(minimum=0, maximum=360, value=180, step=1, label="Motion Bucket ID"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.01, label="Noise Augmentation Strength"),
        gr.Slider(minimum=1, maximum=60, value=10, step=1, label="FPS"),
        gr.Slider(minimum=2, maximum=120, value=25, step=1, label="Frames"),
        gr.Slider(minimum=1, maximum=32, value=8, step=1, label="Decode Chunk Size"),
        gr.HTML("<h3>I2VGen-xl Settings (gif)</h3>"),
        gr.Textbox(label="Prompt", value=""),
        gr.Textbox(label="Negative Prompt", value=""),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Steps"),
        gr.Slider(minimum=1.0, maximum=30.0, value=9.0, step=0.1, label="CFG"),
        gr.Button(value="Stop generation", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Video(label="Generated video"),
        gr.Image(label="Generated GIF", type="filepath"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion (video)",
    description="This user interface allows you to enter an initial image and generate a video using StableVideoDiffusion(mp4) and I2VGen-xl(gif). "
                "You can customize the generation settings from the sliders. "
                "Try it and see what happens!",
    allow_flagging="never",
)

cascade_interface = gr.Interface(
    fn=generate_image_cascade,
    inputs=[
        gr.Textbox(label="Enter your prompt"),
        gr.Textbox(label="Enter your negative prompt", value=""),
        gr.HTML("<h3>Stable Cascade Settings</h3>"),
        gr.Slider(minimum=256, maximum=4096, value=1024, step=64, label="Width"),
        gr.Slider(minimum=256, maximum=4096, value=1024, step=64, label="Height"),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Prior Steps"),
        gr.Slider(minimum=1.0, maximum=30.0, value=4.0, step=0.1, label="Prior Guidance Scale"),
        gr.Slider(minimum=1, maximum=100, value=20, step=1, label="Decoder Steps"),
        gr.Slider(minimum=0.0, maximum=30.0, value=8.0, step=0.1, label="Decoder Guidance Scale"),
        gr.Radio(choices=["png", "jpeg"], label="Select output format", value="png", interactive=True),
        gr.Button(value="Stop generation", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="Generated image"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion (cascade)",
    description="This user interface allows you to enter a prompt and generate images using Stable Cascade. "
                "You can customize the generation settings from the sliders. "
                "Try it and see what happens!",
    allow_flagging="never",
)

extras_interface = gr.Interface(
    fn=generate_image_extras,
    inputs=[
        gr.Image(label="Image to modify", type="filepath"),
        gr.Radio(choices=["png", "jpeg"], label="Select output format", value="png", interactive=True),
        gr.Checkbox(label="Remove Background", value=False),
        gr.Button(value="Stop generation", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(label="Modified image", type="filepath"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion (extras)",
    description="This user interface allows you to modify the image",
    allow_flagging="never",
)

zeroscope2_interface = gr.Interface(
    fn=generate_video_zeroscope2,
    inputs=[
        gr.Textbox(label="Enter your prompt"),
        gr.Video(label="Video to enhance (optional)", interactive=True),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.1, label="Strength"),
        gr.Slider(minimum=1, maximum=100, value=40, step=1, label="Steps"),
        gr.Slider(minimum=256, maximum=1280, value=576, step=64, label="Width"),
        gr.Slider(minimum=256, maximum=1280, value=320, step=64, label="Height"),
        gr.Slider(minimum=1, maximum=100, value=36, step=1, label="Frames"),
        gr.Checkbox(label="Enable Video Enhancement", value=False),
        gr.Button(value="Stop generation", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Video(label="Generated video"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - ZeroScope 2",
    description="This user interface allows you to generate and enhance videos using ZeroScope 2 models. "
                "You can enter a text prompt, upload an optional video for enhancement, and customize the generation settings. "
                "Try it and see what happens!",
    allow_flagging="never",
)

triposr_interface = gr.Interface(
    fn=generate_3d_triposr,
    inputs=[
        gr.Image(label="Input image", type="pil"),
        gr.Slider(minimum=32, maximum=320, value=256, step=32, label="Marching Cubes Resolution"),
        gr.Slider(minimum=0.5, maximum=1.0, value=0.85, step=0.05, label="Foreground Ratio"),
        gr.Radio(choices=["obj", "glb"], label="Select output format", value="obj", interactive=True),
        gr.Button(value="Stop generation", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Model3D(label="Generated 3D object"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - TripoSR",
    description="This user interface allows you to generate 3D objects using TripoSR. "
                "Upload an image and customize the generation settings. "
                "Try it and see what happens!",
    allow_flagging="never",
)

shap_e_interface = gr.Interface(
    fn=generate_3d,
    inputs=[
        gr.Textbox(label="Enter your prompt"),
        gr.Image(label="Initial image (optional)", type="filepath", interactive=True),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Steps"),
        gr.Slider(minimum=1.0, maximum=30.0, value=10.0, step=0.1, label="CFG"),
        gr.Slider(minimum=64, maximum=512, value=256, step=64, label="Frame size"),
        gr.Button(value="Stop generation", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Model3D(label="Generated 3D object"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - Shap-E",
    description="This user interface allows you to generate 3D objects using Shap-E. "
                "You can enter a text prompt or upload an initial image, and customize the generation settings. "
                "Try it and see what happens!",
    allow_flagging="never",
)

audiocraft_interface = gr.Interface(
    fn=generate_audio_audiocraft,
    inputs=[
        gr.Textbox(label="Enter your prompt"),
        gr.Audio(type="filepath", label="Melody audio (optional)", interactive=True),
        gr.Dropdown(choices=audiocraft_models_list, label="Select AudioCraft model", value=None),
        gr.HTML("<h3>AudioCraft Settings</h3>"),
        gr.Radio(choices=["musicgen", "audiogen", "magnet"], label="Select model type", value="musicgen"),
        gr.Slider(minimum=1, maximum=120, value=10, step=1, label="Duration (seconds)"),
        gr.Slider(minimum=1, maximum=1000, value=250, step=1, label="Top K"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="Top P"),
        gr.Slider(minimum=0.0, maximum=1.9, value=1.0, step=0.1, label="Temperature"),
        gr.Slider(minimum=1.0, maximum=10.0, value=3.0, step=0.1, label="CFG"),
        gr.Checkbox(label="Enable Multiband Diffusion", value=False),
        gr.Radio(choices=["wav", "mp3", "ogg"], label="Select output format (Works only without Multiband Diffusion)", value="wav", interactive=True),
        gr.Button(value="Stop generation", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Audio(label="Generated audio", type="filepath"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - AudioCraft",
    description="This user interface allows you to enter any text and generate audio using AudioCraft. "
                "You can select the model and customize the generation settings from the sliders. "
                "Try it and see what happens!",
    allow_flagging="never",
)

audioldm2_interface = gr.Interface(
    fn=generate_audio_audioldm2,
    inputs=[
        gr.Textbox(label="Enter your prompt"),
        gr.Textbox(label="Enter your negative prompt", value=""),
        gr.Dropdown(choices=["cvssp/audioldm2", "cvssp/audioldm2-music"], label="Select AudioLDM 2 model", value="cvssp/audioldm2"),
        gr.Slider(minimum=1, maximum=1000, value=200, step=1, label="Steps"),
        gr.Slider(minimum=1, maximum=60, value=10, step=1, label="Length (seconds)"),
        gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Waveforms number"),
        gr.Radio(choices=["wav", "mp3", "ogg"], label="Select output format", value="wav", interactive=True),
        gr.Button(value="Stop generation", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Audio(label="Generated audio", type="filepath"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - AudioLDM 2",
    description="This user interface allows you to enter any text and generate audio using AudioLDM 2. "
                "You can select the model and customize the generation settings from the sliders. "
                "Try it and see what happens!",
    allow_flagging="never",
)

demucs_interface = gr.Interface(
    fn=demucs_separate,
    inputs=[
        gr.Audio(type="filepath", label="Audio file to separate"),
        gr.Radio(choices=["wav", "mp3", "ogg"], label="Select output format", value="wav", interactive=True),
    ],
    outputs=[
        gr.Audio(label="Vocal", type="filepath"),
        gr.Audio(label="Instrumental", type="filepath"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - Demucs",
    description="This user interface allows you to upload an audio file and separate it into vocal and instrumental using Demucs. "
                "The separated audio files will be saved in the outputs folder. "
                "Try it and see what happens!",
    allow_flagging="never",
)

model_downloader_interface = gr.Interface(
    fn=download_model,
    inputs=[
        gr.Dropdown(choices=[None, "Phi3(Transformers3B)", "OpenChat(Llama7B.Q4)"], label="Download LLM model", value=None),
        gr.Dropdown(choices=[None, "Dreamshaper8(SD1.5)", "RealisticVisionV4.0(SDXL)"], label="Download StableDiffusion model", value=None),
    ],
    outputs=[
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - ModelDownloader",
    description="This user interface allows you to download LLM and StableDiffusion models",
    allow_flagging="never",
)

settings_interface = gr.Interface(
    fn=settings_interface,
    inputs=[
        gr.Radio(choices=["True", "False"], label="Share Mode", value="False")
    ],
    outputs=[
        gr.Textbox(label="Message", type="text")
    ],
    title="NeuroSandboxWebUI (ALPHA) - Settings",
    description="This user interface allows you to change settings of application",
    allow_flagging="never",
)

system_interface = gr.Interface(
    fn=get_system_info,
    inputs=[],
    outputs=[
        gr.Textbox(label="GPU Total Memory"),
        gr.Textbox(label="GPU Used Memory"),
        gr.Textbox(label="GPU Free Memory"),
        gr.Textbox(label="GPU Temperature"),
        gr.Textbox(label="CPU Temperature"),
        gr.Textbox(label="RAM Total"),
        gr.Textbox(label="RAM Used"),
        gr.Textbox(label="RAM Free"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - System",
    description="This interface displays system information",
    allow_flagging="never",
)

with gr.TabbedInterface(
        [chat_interface, tts_stt_interface, bark_interface, translate_interface, wav2lip_interface, gr.TabbedInterface([txt2img_interface, img2img_interface, depth2img_interface, upscale_interface, inpaint_interface, gligen_interface, animatediff_interface, video_interface, cascade_interface, extras_interface],
        tab_names=["txt2img", "img2img", "depth2img", "upscale", "inpaint", "gligen", "animatediff", "video", "cascade", "extras"]),
                    zeroscope2_interface, triposr_interface, shap_e_interface, audiocraft_interface, audioldm2_interface, demucs_interface, model_downloader_interface, settings_interface, system_interface],
        tab_names=["LLM", "TTS-STT", "SunoBark", "LibreTranslate", "Wav2Lip", "StableDiffusion", "ZeroScope 2", "TripoSR", "Shap-E", "AudioCraft", "AudioLDM 2", "Demucs", "ModelDownloader", "Settings", "System"]
) as app:
    chat_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    bark_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    txt2img_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    img2img_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    depth2img_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    upscale_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    inpaint_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    gligen_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    animatediff_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    video_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    cascade_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    extras_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    zeroscope2_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    triposr_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    shap_e_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    audiocraft_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    audioldm2_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)

    close_button = gr.Button("Close terminal")
    close_button.click(close_terminal, [], [], queue=False)

    folder_button = gr.Button("Folder")
    folder_button.click(open_outputs_folder, [], [], queue=False)

    github_link = gr.HTML(
        '<div style="text-align: center; margin-top: 20px;">'
        '<a href="https://github.com/Dartvauder/NeuroSandboxWebUI" target="_blank" style="color: blue; text-decoration: none; font-size: 16px;">'
        'GitHub'
        '</a>'
        '</div>'
    )

    app.launch(share=share_mode, server_name="localhost")
