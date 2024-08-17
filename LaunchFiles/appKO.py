import gradio as gr
import langdetect
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, BarkModel, pipeline, T5EncoderModel, BitsAndBytesConfig, DPTForDepthEstimation, DPTFeatureExtractor
from peft import PeftModel
from libretranslatepy import LibreTranslateAPI
import urllib.error
import soundfile as sf
from pydub import AudioSegment
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import cv2
import subprocess
import json
import torch
from einops import rearrange
from TTS.api import TTS
import whisper
from datetime import datetime
from huggingface_hub import snapshot_download
from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline, StableDiffusionXLPipeline, StableDiffusionImg2ImgPipeline, StableDiffusion3Img2ImgPipeline, SD3ControlNetModel, StableDiffusion3ControlNetPipeline, StableDiffusion3InpaintPipeline, StableDiffusionDepth2ImgPipeline, ControlNetModel, StableDiffusionXLControlNetPipeline, StableDiffusionControlNetPipeline, AutoencoderKL, StableDiffusionLatentUpscalePipeline, StableDiffusionUpscalePipeline, StableDiffusionInpaintPipeline, StableDiffusionGLIGENPipeline, AnimateDiffPipeline, AnimateDiffSDXLPipeline, AnimateDiffVideoToVideoPipeline, MotionAdapter, StableVideoDiffusionPipeline, I2VGenXLPipeline, StableCascadePriorPipeline, StableCascadeDecoderPipeline, DiffusionPipeline, ShapEPipeline, ShapEImg2ImgPipeline, StableAudioPipeline, AudioLDM2Pipeline, StableDiffusionInstructPix2PixPipeline, StableDiffusionLDM3DPipeline, FluxPipeline, KandinskyPipeline, KandinskyPriorPipeline, KandinskyV22Pipeline, KandinskyV22PriorPipeline, AutoPipelineForText2Image, KandinskyImg2ImgPipeline, AutoPipelineForImage2Image, AutoPipelineForInpainting, HunyuanDiTPipeline, LuminaText2ImgPipeline, IFPipeline, IFSuperResolutionPipeline, IFImg2ImgPipeline, IFInpaintingPipeline, IFImg2ImgSuperResolutionPipeline, IFInpaintingSuperResolutionPipeline, PixArtAlphaPipeline, PixArtSigmaPipeline, CogVideoXPipeline, LattePipeline, KolorsPipeline, AuraFlowPipeline, WuerstchenDecoderPipeline, WuerstchenPriorPipeline
from diffusers.utils import load_image, export_to_video, export_to_gif, export_to_ply, pt_to_pil
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS
from instantid.pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps
from photomaker import PhotoMakerStableDiffusionXLPipeline
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
from aura_sr import AuraSR
from controlnet_aux import OpenposeDetector, LineartDetector, HEDdetector
from insightface.app import FaceAnalysis
from compel import Compel, ReturnedEmbeddingsType
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
from llama_cpp.llama_chat_format import MoondreamChatHandler
import requests
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from fake_useragent import UserAgent
from googlesearch import search
import html2text
from rembg import remove
import torchaudio
from audiocraft.models import MusicGen, AudioGen, MultiBandDiffusion, MAGNeT
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

chat_dir = None
tts_model = None
whisper_model = None
audiocraft_model_path = None

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def authenticate(username, password):
    try:
        with open("GradioAuth.txt", "r") as file:
            stored_credentials = file.read().strip().split(":")
            if len(stored_credentials) == 2:
                stored_username, stored_password = stored_credentials
                return username == stored_username and password == stored_password
    except FileNotFoundError:
        pass
    return False


def get_hf_token():
    token_file = "HF-Token.txt"
    if os.path.exists(token_file):
        with open(token_file, "r") as f:
            return f.read().strip()
    return None


def perform_web_search(query, num_results=5, max_length=500):
    ua = UserAgent()
    user_agent = ua.random

    options = webdriver.ChromeOptions()
    options.add_argument(f'user-agent={user_agent}')
    options.add_argument('--headless')
    options.add_argument('--disable-images')
    options.add_argument('--disable-extensions')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(30)

    search_results = []
    for url in search(query, num_results=num_results):
        try:
            driver.get(url)
            page_source = driver.page_source
            h = html2text.HTML2Text()
            h.ignore_links = True
            h.ignore_images = True
            h.ignore_emphasis = True
            page_text = h.handle(page_source)
            page_text = re.sub(r'\s+', ' ', page_text).strip()
            search_results.append(page_text)
        except (TimeoutException, NoSuchElementException):
            continue

    driver.quit()

    search_text = " ".join(search_results)
    if len(search_text) > max_length:
        search_text = search_text[:max_length]

    return search_text


def remove_bg(src_img_path, out_img_path):
    model_path = "inputs/image/sd_models/rembg"
    os.makedirs(model_path, exist_ok=True)

    os.environ["U2NET_HOME"] = model_path

    with open(src_img_path, "rb") as input_file:
        input_data = input_file.read()

    output_data = remove(input_data)

    with open(out_img_path, "wb") as output_file:
        output_file.write(output_data)


def generate_mel_spectrogram(audio_path):
    audio_format = audio_path.split('.')[-1].lower()

    if audio_format == 'wav':
        sr, y = wavfile.read(audio_path)
        y = y.astype(np.float32) / np.iinfo(y.dtype).max
    else:
        audio = AudioSegment.from_file(audio_path, format=audio_format)
        y = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        sr = audio.frame_rate

    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(img, format='%+2.0f dB')
    plt.title('Mel-spectrogram')
    plt.tight_layout()

    spectrogram_path = audio_path.rsplit('.', 1)[0] + '_melspec.png'
    plt.savefig(spectrogram_path)
    plt.close()

    return spectrogram_path


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


def load_lora_model(base_model_name, lora_model_name, model_type):
    global stop_signal
    if stop_signal:
        return None, None, "Generation stopped"

    if model_type == "llama":
        return None, None, "LORA model with 'llama' model type is not supported yet!"

    base_model_path = f"inputs/text/llm_models/{base_model_name}"
    lora_model_path = f"inputs/text/llm_models/lora/{lora_model_name}"

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path).to(device)
        model = PeftModel.from_pretrained(base_model, lora_model_path).to(device)
        merged_model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        return tokenizer, merged_model, None
    finally:
        del tokenizer
        del merged_model
        torch.cuda.empty_cache()


def load_moondream2_model(model_id, revision):
    global stop_signal
    if stop_signal:
        return "Generation stopped"
    moondream2_model_path = os.path.join("inputs", "text", "llm_models", model_id)
    if not os.path.exists(moondream2_model_path):
        print(f"Downloading MoonDream2 model...")
        os.makedirs(moondream2_model_path, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, revision=revision
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model.save_pretrained(moondream2_model_path)
        tokenizer.save_pretrained(moondream2_model_path)
        print("MoonDream2 model downloaded")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(moondream2_model_path, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(moondream2_model_path)
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
    tts_model_path = "inputs/audio/XTTS-v2"
    if not os.path.exists(tts_model_path):
        print("Downloading TTS...")
        os.makedirs(tts_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/coqui/XTTS-v2", tts_model_path)
        print("TTS model downloaded")
    return TTS(model_path=tts_model_path, config_path=f"{tts_model_path}/config.json")


def load_whisper_model():
    global stop_signal
    if stop_signal:
        return "Generation stopped"
    whisper_model_path = "inputs/text/whisper-medium"
    if not os.path.exists(whisper_model_path):
        print("Downloading Whisper...")
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
    audiocraft_model_path = os.path.join("inputs", "audio", "audiocraft", model_name)
    if not os.path.exists(audiocraft_model_path):
        print(f"Downloading AudioCraft model: {model_name}...")
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
    multiband_diffusion_path = os.path.join("inputs", "audio", "audiocraft", "multiband-diffusion")
    if not os.path.exists(multiband_diffusion_path):
        print(f"Downloading Multiband Diffusion model")
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

    if not os.path.exists(upscale_model_path):
        print(f"Downloading Upscale model: {upscale_model_name}")
        os.makedirs(upscale_model_path, exist_ok=True)
        Repo.clone_from(f"https://huggingface.co/{upscale_model_name}", upscale_model_path)
        print(f"Upscale model {upscale_model_name} downloaded")

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

    upscaler.upscale_factor = upscale_factor
    return upscaler


stop_signal = False

chat_history = []


def generate_text_and_speech(input_text, input_audio, input_image, llm_model_name, llm_lora_model_name, llm_settings_html, llm_model_type, max_length, max_tokens,
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
            moondream2_path = os.path.join("inputs", "text", "llm_models", "moondream2")

            if not os.path.exists(moondream2_path):
                print("Downloading Moondream2 model...")
                os.makedirs(moondream2_path, exist_ok=True)
                Repo.clone_from("https://huggingface.co/vikhyatk/moondream2", moondream2_path)
                print("Moondream2 model downloaded")

            chat_handler = MoondreamChatHandler.from_pretrained(
                moondream2_path,
                filename="*mmproj*",
            )

            llm = Llama.from_pretrained(
                moondream2_path,
                filename="*text-model*",
                chat_handler=chat_handler,
                n_ctx=2048
            )

            if input_image:
                image_path = input_image
            else:
                return chat_history, None, None, "Please upload an image for multimodal input."

            context = ""
            for human_text, ai_text in chat_history[-5:]:
                if human_text:
                    context += f"Human: {human_text}\n"
                if ai_text:
                    context += f"AI: {ai_text}\n"

            response = llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant capable of analyzing images and text.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{context}Human: {prompt}"},
                            {"type": "image_url", "image_url": {"url": f"file://{image_path}"}}
                        ]
                    }
                ]
            )

            text = response["choices"][0]["message"]["content"]
        model_id = "vikhyatk/moondream2"
        revision = "2024-07-23"
        model, tokenizer = load_moondream2_model(model_id, revision)

        try:
            image = Image.open(input_image)
            enc_image = model.encode_image(image)

            detect_lang = langdetect.detect(prompt)
            if detect_lang == "ru":
                bot_instruction = "Вы дружелюбный чат-бот, который всегда дает полезные и содержательные ответы на основе данного изображения и текстового ввода."
            else:
                bot_instruction = "You are a helpful AI assistant capable of analyzing images and text."

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
        if llm_lora_model_name:
            tokenizer, llm_model, error_message = load_lora_model(llm_model_name, llm_lora_model_name, llm_model_type)
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
            if llm_model:
                if llm_model_type == "transformers":
                    detect_lang = langdetect.detect(prompt)
                    if detect_lang == "ru":
                        bot_instruction = "Вы дружелюбный чат-бот, который всегда дает полезные и содержательные ответы на любом языке"
                    else:
                        bot_instruction = "I am a chatbot created to help with any questions. I use my knowledge and abilities to provide useful and meaningful answers in any language"

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

                    if enable_web_search:
                        search_results = perform_web_search(prompt)
                        prompt_with_search = f"{prompt}\nWeb search results:\n{search_results}"
                        messages[1]["content"] = context + prompt_with_search
                        model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, padding=True,
                                                                     return_tensors="pt").to(device)
                        input_length = model_inputs.shape[1]

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
                    if detect_lang == "ru":
                        instruction = "Вы дружелюбный чат-бот, который всегда дает полезные и содержательные ответы на любом языке\n\n"
                    else:
                        instruction = "I am a chatbot created to help with any questions. I use my knowledge and abilities to provide useful and meaningful answers in any language\n\n"

                    context = ""
                    for human_text, ai_text in chat_history[-10:]:
                        if human_text:
                            context += f"Human: {human_text}\n"
                        if ai_text:
                            context += f"Assistant: {ai_text}\n"

                    prompt_with_context = instruction + context + "Human: " + prompt + "\nAssistant: "

                    progress_bar = tqdm(total=max_tokens, desc="Generating text")
                    progress_tokens = 0

                    if enable_web_search:
                        search_results = perform_web_search(prompt)
                        prompt_with_context = f"{prompt_with_context}\nWeb search results:\n{search_results}\nAssistant: "

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

        try:
            wav = tts_model.tts(text=text, speaker_wav=f"inputs/audio/voices/{speaker_wav}", language=language,
                                temperature=tts_temperature, top_p=tts_top_p, top_k=tts_top_k, speed=tts_speed)
        finally:
            del tts_model
            torch.cuda.empty_cache()

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

        try:
            stt_output = transcribe_audio(audio)
        finally:
            del whisper_model
            torch.cuda.empty_cache()

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
        return None, None, "Please enter text for the request!"

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
            return None, None, "Generation stopped"

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

        spectrogram_path = generate_mel_spectrogram(audio_path)

        return audio_path, spectrogram_path, None

    except Exception as e:
        return None, None, str(e)

    finally:
        del model
        del processor
        torch.cuda.empty_cache()


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
            print("Downloading Wav2Lip GAN model...")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            url = "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth"
            response = requests.get(url, allow_redirects=True)
            with open(checkpoint_path, "wb") as file:
                file.write(response.content)
            print("Wav2Lip GAN model downloaded")

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
                           stable_diffusion_clip_skip, enable_freeu=False, enable_tiled_vae=False, enable_upscale=False, upscale_factor="x2", upscale_steps=50, upscale_cfg=6, output_format="png", stop_generation=None):
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

    if enable_freeu:
        stable_diffusion_model.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)

    if enable_tiled_vae:
        stable_diffusion_model.enable_vae_tiling()

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
        if stable_diffusion_model_type == "SDXL":
            compel = Compel(
                tokenizer=[stable_diffusion_model.tokenizer, stable_diffusion_model.tokenizer_2],
                text_encoder=[stable_diffusion_model.text_encoder, stable_diffusion_model.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True]
            )
            prompt_embeds, pooled_prompt_embeds = compel(prompt)
            images = stable_diffusion_model(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
                                            negative_prompt=negative_prompt,
                                            num_inference_steps=stable_diffusion_steps,
                                            guidance_scale=stable_diffusion_cfg, height=stable_diffusion_height,
                                            width=stable_diffusion_width, clip_skip=stable_diffusion_clip_skip,
                                            sampler=stable_diffusion_sampler)
        else:
            compel_proc = Compel(tokenizer=stable_diffusion_model.tokenizer,
                                 text_encoder=stable_diffusion_model.text_encoder)
            prompt_embeds = compel_proc(prompt)
            negative_prompt_embeds = compel_proc(negative_prompt)

            images = stable_diffusion_model(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
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

        if stable_diffusion_model_type == "SDXL":
            compel = Compel(
                tokenizer=[stable_diffusion_model.tokenizer, stable_diffusion_model.tokenizer_2],
                text_encoder=[stable_diffusion_model.text_encoder, stable_diffusion_model.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True]
            )
            prompt_embeds, pooled_prompt_embeds = compel(prompt)

            images = stable_diffusion_model(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_prompt=negative_prompt,
                                            num_inference_steps=stable_diffusion_steps,
                                            guidance_scale=stable_diffusion_cfg, clip_skip=stable_diffusion_clip_skip,
                                            sampler=stable_diffusion_sampler, image=init_image, strength=strength)
        else:
            compel_proc = Compel(tokenizer=stable_diffusion_model.tokenizer,
                                 text_encoder=stable_diffusion_model.text_encoder)
            prompt_embeds = compel_proc(prompt)
            negative_prompt_embeds = compel_proc(negative_prompt)

            images = stable_diffusion_model(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
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

        compel_proc = Compel(tokenizer=stable_diffusion_model.tokenizer,
                             text_encoder=stable_diffusion_model.text_encoder)
        prompt_embeds = compel_proc(prompt)
        negative_prompt_embeds = compel_proc(negative_prompt)

        image = stable_diffusion_model(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, image=init_image, strength=strength).images[0]

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


def generate_image_pix2pix(prompt, negative_prompt, init_image, num_inference_steps, guidance_scale,
                           output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    if not init_image:
        return None, "Please, upload an initial image!"

    pix2pix_model_path = os.path.join("inputs", "image", "sd_models", "pix2pix")

    if not os.path.exists(pix2pix_model_path):
        print("Downloading Pix2Pix model...")
        os.makedirs(pix2pix_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/timbrooks/instruct-pix2pix", pix2pix_model_path)
        print("Pix2Pix model downloaded")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(pix2pix_model_path, torch_dtype=torch.float16,
                                                                      safety_checker=None)
        pipe.to(device)

        image = Image.open(init_image).convert("RGB")

        compel_proc = Compel(tokenizer=pipe.tokenizer,
                             text_encoder=pipe.text_encoder)
        prompt_embeds = compel_proc(prompt)
        negative_prompt_embeds = compel_proc(negative_prompt)

        image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                     image=image, num_inference_steps=num_inference_steps, image_guidance_scale=guidance_scale).images[
            0]

        if stop_signal:
            return None, "Generation stopped"

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"pix2pix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        return image_path, None

    finally:
        del pipe
        torch.cuda.empty_cache()


def generate_image_controlnet(prompt, negative_prompt, init_image, sd_version, stable_diffusion_model_name, controlnet_model_name,
                              num_inference_steps, guidance_scale, width, height, controlnet_conditioning_scale, output_format="png",
                              stop_generation=None):
    global stop_signal
    stop_signal = False

    if not init_image:
        return None, None, "Please, upload an initial image!"

    if not stable_diffusion_model_name:
        return None, None, "Please, select a StableDiffusion model!"

    if not controlnet_model_name:
        return None, None, "Please, select a ControlNet model!"

    stable_diffusion_model_path = os.path.join("inputs", "image", "sd_models",
                                               f"{stable_diffusion_model_name}.safetensors")

    if not os.path.exists(stable_diffusion_model_path):
        return None, None, f"StableDiffusion model not found: {stable_diffusion_model_path}"

    controlnet_model_path = os.path.join("inputs", "image", "sd_models", "controlnet", controlnet_model_name)
    if not os.path.exists(controlnet_model_path):
        print(f"Downloading ControlNet {controlnet_model_name} model...")
        os.makedirs(controlnet_model_path, exist_ok=True)
        if controlnet_model_name == "openpose":
            Repo.clone_from("https://huggingface.co/lllyasviel/control_v11p_sd15_openpose", controlnet_model_path)
        elif controlnet_model_name == "depth":
            if sd_version == "SD":
                Repo.clone_from("https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth", controlnet_model_path)
            else:
                Repo.clone_from("https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0", controlnet_model_path)
        elif controlnet_model_name == "canny":
            if sd_version == "SD":
                Repo.clone_from("https://huggingface.co/lllyasviel/control_v11p_sd15_canny", controlnet_model_path)
            else:
                Repo.clone_from("https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0", controlnet_model_path)
        elif controlnet_model_name == "lineart":
            Repo.clone_from("https://huggingface.co/lllyasviel/control_v11p_sd15_lineart", controlnet_model_path)
        elif controlnet_model_name == "scribble":
            Repo.clone_from("https://huggingface.co/lllyasviel/control_v11p_sd15_scribble", controlnet_model_path)
        print(f"ControlNet {controlnet_model_name} model downloaded")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if sd_version == "SD":
            controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)
            pipe = StableDiffusionControlNetPipeline.from_single_file(
                stable_diffusion_model_path,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                device_map="auto",
                use_safetensors=True,
            )
            pipe.enable_model_cpu_offload()
            pipe.to(device)

            image = Image.open(init_image).convert("RGB")

            if controlnet_model_name == "openpose":
                processor = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
                control_image = processor(image, hand_and_face=True)
            elif controlnet_model_name == "depth":
                depth_estimator = pipeline('depth-estimation')
                control_image = depth_estimator(image)['depth']
                control_image = np.array(control_image)
                control_image = control_image[:, :, None]
                control_image = np.concatenate([control_image, control_image, control_image], axis=2)
                control_image = Image.fromarray(control_image)
            elif controlnet_model_name == "canny":
                image_array = np.array(image)
                low_threshold = 100
                high_threshold = 200
                control_image = cv2.Canny(image_array, low_threshold, high_threshold)
                control_image = control_image[:, :, None]
                control_image = np.concatenate([control_image, control_image, control_image], axis=2)
                control_image = Image.fromarray(control_image)
            elif controlnet_model_name == "lineart":
                processor = LineartDetector.from_pretrained("lllyasviel/Annotators")
                control_image = processor(image)
            elif controlnet_model_name == "scribble":
                processor = HEDdetector.from_pretrained("lllyasviel/Annotators")
                control_image = processor(image, scribble=True)

            generator = torch.manual_seed(0)

            compel_proc = Compel(tokenizer=pipe.tokenizer,
                                 text_encoder=pipe.text_encoder)
            prompt_embeds = compel_proc(prompt)
            negative_prompt_embeds = compel_proc(negative_prompt)

            image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                         num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, width=width,
                         height=height, generator=generator, image=control_image).images[0]

        else:  # SDXL
            controlnet = ControlNetModel.from_pretrained(
                controlnet_model_path,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                stable_diffusion_model_path,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
            pipe.enable_model_cpu_offload()

            image = Image.open(init_image).convert("RGB")

            if controlnet_model_name == "depth":
                depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device)
                feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

                def get_depth_map(image):
                    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
                    with torch.no_grad(), torch.autocast(device):
                        depth_map = depth_estimator(image).predicted_depth

                    depth_map = torch.nn.functional.interpolate(
                        depth_map.unsqueeze(1),
                        size=(1024, 1024),
                        mode="bicubic",
                        align_corners=False,
                    )
                    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
                    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
                    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
                    image = torch.cat([depth_map] * 3, dim=1)

                    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
                    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
                    return image

                control_image = get_depth_map(image)
            elif controlnet_model_name == "canny":
                image = np.array(image)
                control_image = cv2.Canny(image, 100, 200)
                control_image = control_image[:, :, None]
                control_image = np.concatenate([control_image, control_image, control_image], axis=2)
                control_image = Image.fromarray(control_image)
            else:
                return None, None, f"ControlNet model {controlnet_model_name} is not supported for SDXL"

            image = pipe(
                prompt, negative_prompt=negative_prompt, image=control_image,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                width=width, height=height
            ).images[0]

        if stop_signal:
            return None, None, "Generation stopped"

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)

        image_filename = f"controlnet_{controlnet_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        control_image_filename = f"controlnet_{controlnet_model_name}_control_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        control_image_path = os.path.join(image_dir, control_image_filename)
        control_image.save(control_image_path, format=output_format.upper())

        return image_path, control_image_path, None

    except Exception as e:
        return None, None, str(e)

    finally:
        try:
            del controlnet
            del pipe
            if 'depth_estimator' in locals():
                del depth_estimator
            if 'feature_extractor' in locals():
                del feature_extractor
        except UnboundLocalError:
            pass
        torch.cuda.empty_cache()


def generate_image_upscale_latent(image_path, num_inference_steps, guidance_scale, output_format="png", stop_generation=None):
    global stop_signal
    if stop_signal:
        return None, "Generation stopped"

    if not image_path:
        return None, "Please, upload an initial image!"

    upscale_factor = 2
    upscaler = load_upscale_model(upscale_factor)

    if upscaler:
        try:
            image = Image.open(image_path).convert("RGB")
            upscaled_image = upscaler(prompt="", image=image, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]

            today = datetime.now().date()
            image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
            os.makedirs(image_dir, exist_ok=True)
            image_filename = f"upscaled_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
            image_path = os.path.join(image_dir, image_filename)
            upscaled_image.save(image_path, format=output_format.upper())

            return image_path, None

        finally:
            del upscaler
            torch.cuda.empty_cache()
    else:
        return None, "Failed to load upscale model"


def generate_image_upscale_realesrgan(image_path, outscale, output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    if not image_path:
        return None, "Please upload an image file!"

    realesrgan_path = os.path.join("inputs", "image", "Real-ESRGAN")
    if not os.path.exists(realesrgan_path):
        print("Downloading Real-ESRGAN...")
        os.makedirs(realesrgan_path, exist_ok=True)
        Repo.clone_from("https://github.com/xinntao/Real-ESRGAN.git", realesrgan_path)
        print("Real-ESRGAN downloaded")

    today = datetime.now().date()
    output_dir = os.path.join('outputs', f"RealESRGAN_{today.strftime('%Y%m%d')}")
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"upscaled_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
    output_path = os.path.join(output_dir, output_filename)

    try:
        command = f"python {os.path.join(realesrgan_path, 'inference_realesrgan.py')} --model_name RealESRGAN_x4plus --input {image_path} --output {output_dir} --outscale {outscale} --fp16"
        subprocess.run(command, shell=True, check=True)

        if stop_signal:
            return None, "Generation stopped"

        return output_path, None

    except subprocess.CalledProcessError as e:
        return None, f"Error occurred: {str(e)}"

    except Exception as e:
        return None, str(e)


def generate_image_inpaint(prompt, negative_prompt, init_image, mask_image, blur_factor, stable_diffusion_model_name, vae_model_name,
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

        if blur_factor > 0:
            blurred_mask = stable_diffusion_model.mask_processor.blur(mask_array, blur_factor=blur_factor)
        else:
            blurred_mask = mask_array

        if stable_diffusion_model_type == "SDXL":
            compel = Compel(
                tokenizer=[stable_diffusion_model.tokenizer, stable_diffusion_model.tokenizer_2],
                text_encoder=[stable_diffusion_model.text_encoder, stable_diffusion_model.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True]
            )
            prompt_embeds, pooled_prompt_embeds = compel(prompt)

            images = stable_diffusion_model(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_prompt=negative_prompt,
                                            image=init_image,
                                            mask_image=blurred_mask, width=width, height=height,
                                            num_inference_steps=stable_diffusion_steps,
                                            guidance_scale=stable_diffusion_cfg, sampler=stable_diffusion_sampler)
        else:
            compel_proc = Compel(tokenizer=stable_diffusion_model.tokenizer,
                                 text_encoder=stable_diffusion_model.text_encoder)
            prompt_embeds = compel_proc(prompt)
            negative_prompt_embeds = compel_proc(negative_prompt)

            images = stable_diffusion_model(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                                            image=init_image,
                                            mask_image=blurred_mask, width=width, height=height,
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


def generate_image_gligen(prompt, negative_prompt, gligen_phrases, gligen_boxes, stable_diffusion_model_name, stable_diffusion_settings_html,
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
        if stable_diffusion_model_type == "SDXL":
            compel = Compel(
                tokenizer=[stable_diffusion_model.tokenizer, stable_diffusion_model.tokenizer_2],
                text_encoder=[stable_diffusion_model.text_encoder, stable_diffusion_model.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True]
            )
            prompt_embeds, pooled_prompt_embeds = compel(prompt)

            image = stable_diffusion_model(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_prompt=negative_prompt,
                                           num_inference_steps=stable_diffusion_steps,
                                           guidance_scale=stable_diffusion_cfg, height=stable_diffusion_height,
                                           width=stable_diffusion_width, clip_skip=stable_diffusion_clip_skip,
                                           sampler=stable_diffusion_sampler)["images"][0]
        else:
            compel_proc = Compel(tokenizer=stable_diffusion_model.tokenizer,
                                 text_encoder=stable_diffusion_model.text_encoder)
            prompt_embeds = compel_proc(prompt)
            negative_prompt_embeds = compel_proc(negative_prompt)

            image = stable_diffusion_model(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                                           num_inference_steps=stable_diffusion_steps,
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

        compel_proc = Compel(tokenizer=pipe.tokenizer,
                             text_encoder=pipe.text_encoder)
        prompt_embeds = compel_proc(prompt)
        negative_prompt_embeds = compel_proc(negative_prompt)

        images = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
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


def generate_image_animatediff(prompt, negative_prompt, input_video, strength, model_type, stable_diffusion_model_name, motion_lora_name, num_frames, num_inference_steps,
                               guidance_scale, width, height, stop_generation):
    global stop_signal
    stop_signal = False

    if not stable_diffusion_model_name:
        return None, "Please, select a StableDiffusion model!"

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
        if model_type == "sd":
            if input_video:
                adapter = MotionAdapter.from_pretrained(motion_adapter_path, torch_dtype=torch.float16)
                original_config_file = "configs/sd/v1-inference.yaml"
                stable_diffusion_model = StableDiffusionPipeline.from_single_file(
                    stable_diffusion_model_path,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    original_config_file=original_config_file,
                    device_map="auto",
                )

                pipe = AnimateDiffVideoToVideoPipeline(
                    unet=stable_diffusion_model.unet,
                    text_encoder=stable_diffusion_model.text_encoder,
                    vae=stable_diffusion_model.vae,
                    motion_adapter=adapter,
                    tokenizer=stable_diffusion_model.tokenizer,
                    feature_extractor=stable_diffusion_model.feature_extractor,
                    scheduler=stable_diffusion_model.scheduler,
                )

                pipe.enable_vae_slicing()
                pipe.enable_model_cpu_offload()

                compel_proc = Compel(tokenizer=stable_diffusion_model.tokenizer,
                                     text_encoder=stable_diffusion_model.text_encoder)
                prompt_embeds = compel_proc(prompt)
                negative_prompt_embeds = compel_proc(negative_prompt)

                output = pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    video=input_video,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=torch.manual_seed(-1),
                )

            else:
                adapter = MotionAdapter.from_pretrained(motion_adapter_path, torch_dtype=torch.float16)
                original_config_file = "configs/sd/v1-inference.yaml"
                stable_diffusion_model = StableDiffusionPipeline.from_single_file(
                    stable_diffusion_model_path,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    original_config_file=original_config_file,
                    device_map="auto",
                )

                pipe = AnimateDiffPipeline(
                    unet=stable_diffusion_model.unet,
                    text_encoder=stable_diffusion_model.text_encoder,
                    vae=stable_diffusion_model.vae,
                    motion_adapter=adapter,
                    tokenizer=stable_diffusion_model.tokenizer,
                    feature_extractor=stable_diffusion_model.feature_extractor,
                    scheduler=stable_diffusion_model.scheduler,
                )

                if motion_lora_name:
                    motion_lora_path = os.path.join("inputs", "image", "sd_models", "motion_lora", motion_lora_name)
                    if not os.path.exists(motion_lora_path):
                        print(f"Downloading {motion_lora_name} motion lora...")
                        os.makedirs(motion_lora_path, exist_ok=True)
                        if motion_lora_name == "zoom-in":
                            Repo.clone_from("https://huggingface.co/guoyww/animatediff-motion-lora-zoom-in",
                                            motion_lora_path)
                        elif motion_lora_name == "zoom-out":
                            Repo.clone_from("https://huggingface.co/guoyww/animatediff-motion-lora-zoom-out",
                                            motion_lora_path)
                        elif motion_lora_name == "tilt-up":
                            Repo.clone_from("https://huggingface.co/guoyww/animatediff-motion-lora-tilt-up",
                                            motion_lora_path)
                        elif motion_lora_name == "tilt-down":
                            Repo.clone_from("https://huggingface.co/guoyww/animatediff-motion-lora-tilt-down",
                                            motion_lora_path)
                        elif motion_lora_name == "pan-right":
                            Repo.clone_from("https://huggingface.co/guoyww/animatediff-motion-lora-pan-right",
                                            motion_lora_path)
                        elif motion_lora_name == "pan-left":
                            Repo.clone_from("https://huggingface.co/guoyww/animatediff-motion-lora-pan-left",
                                            motion_lora_path)
                        print(f"{motion_lora_name} motion lora downloaded")
                    pipe.load_lora_weights(motion_lora_path, adapter_name=motion_lora_name)

                pipe.enable_vae_slicing()
                pipe.enable_model_cpu_offload()

                compel_proc = Compel(tokenizer=stable_diffusion_model.tokenizer,
                                     text_encoder=stable_diffusion_model.text_encoder)
                prompt_embeds = compel_proc(prompt)
                negative_prompt_embeds = compel_proc(negative_prompt)

                output = pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=torch.manual_seed(-1),
                    width=width,
                    height=height,
                )

        elif model_type == "sdxl":
            sdxl_adapter_path = os.path.join("inputs", "image", "sd_models", "motion_adapter_sdxl")
            if not os.path.exists(sdxl_adapter_path):
                print("Downloading SDXL motion adapter...")
                os.makedirs(sdxl_adapter_path, exist_ok=True)
                Repo.clone_from("https://huggingface.co/guoyww/animatediff-motion-adapter-sdxl-beta", sdxl_adapter_path)
                print("SDXL motion adapter downloaded")

            adapter = MotionAdapter.from_pretrained(sdxl_adapter_path, torch_dtype=torch.float16)

            pipe = AnimateDiffSDXLPipeline.from_pretrained(
                stable_diffusion_model_path,
                motion_adapter=adapter,
                torch_dtype=torch.float16,
                variant="fp16",
            ).to("cuda")

            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()

            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                num_frames=num_frames,
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

        if not os.path.exists(video_model_path):
            print(f"Downloading StableVideoDiffusion model")
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

            generator = torch.manual_seed(0)
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

        if not os.path.exists(video_model_path):
            print(f"Downloading i2vgen-xl model")
            os.makedirs(video_model_path, exist_ok=True)
            Repo.clone_from(f"https://huggingface.co/{video_model_name}", video_model_path)
            print(f"i2vgen-xl model downloaded")

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = I2VGenXLPipeline.from_pretrained(video_model_path, torch_dtype=torch.float16, variant="fp16")
            pipe.to(device)
            pipe.enable_model_cpu_offload()

            image = load_image(init_image).convert("RGB")

            generator = torch.manual_seed(0)

            frames = pipe(
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


def generate_image_ldm3d(prompt, negative_prompt, width, height, num_inference_steps, guidance_scale, output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    if not prompt:
        return None, None, "Please enter a prompt!"

    ldm3d_model_path = os.path.join("inputs", "image", "sd_models", "ldm3d")

    if not os.path.exists(ldm3d_model_path):
        print("Downloading LDM3D model...")
        os.makedirs(ldm3d_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/Intel/ldm3d-4c", ldm3d_model_path)
        print("LDM3D model downloaded")

    try:
        pipe = StableDiffusionLDM3DPipeline.from_pretrained(ldm3d_model_path, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        if stop_signal:
            return None, None, "Generation stopped"

        rgb_image, depth_image = output.rgb[0], output.depth[0]

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)

        rgb_filename = f"ldm3d_rgb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        depth_filename = f"ldm3d_depth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"

        rgb_path = os.path.join(image_dir, rgb_filename)
        depth_path = os.path.join(image_dir, depth_filename)

        rgb_image.save(rgb_path)
        depth_image.save(depth_path)

        return rgb_path, depth_path, None

    except Exception as e:
        return None, None, str(e)

    finally:
        del pipe
        torch.cuda.empty_cache()


def generate_image_sd3_txt2img(prompt, negative_prompt, num_inference_steps, guidance_scale, width, height, max_sequence_length, output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    sd3_model_path = os.path.join("inputs", "image", "sd_models", "sd3")

    if not os.path.exists(sd3_model_path):
        print("Downloading Stable Diffusion 3 model...")
        os.makedirs(sd3_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/v2ray/stable-diffusion-3-medium-diffusers", sd3_model_path)
        print("Stable Diffusion 3 model downloaded")

    try:

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        text_encoder = T5EncoderModel.from_pretrained(
            sd3_model_path,
            subfolder="text_encoder_3",
            quantization_config=quantization_config,
        )
        pipe = StableDiffusion3Pipeline.from_pretrained(sd3_model_path, device_map="balanced", text_encoder_3=text_encoder, torch_dtype=torch.float16)

        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            max_sequence_length=max_sequence_length,
        ).images[0]

        if stop_signal:
            return None, "Generation stopped"

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"sd3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        return image_path, None

    finally:
        del pipe
        torch.cuda.empty_cache()


def generate_image_sd3_img2img(prompt, negative_prompt, init_image, strength, num_inference_steps, guidance_scale, width, height, max_sequence_length, output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    sd3_model_path = os.path.join("inputs", "image", "sd_models", "sd3")

    if not os.path.exists(sd3_model_path):
        print("Downloading Stable Diffusion 3 model...")
        os.makedirs(sd3_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/v2ray/stable-diffusion-3-medium-diffusers", sd3_model_path)
        print("Stable Diffusion 3 model downloaded")

    try:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        text_encoder = T5EncoderModel.from_pretrained(
            sd3_model_path,
            subfolder="text_encoder_3",
            quantization_config=quantization_config,
        )
        pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(sd3_model_path, device_map="balanced", text_encoder_3=text_encoder, torch_dtype=torch.float16)

        init_image = Image.open(init_image).convert("RGB")
        init_image = init_image.resize((width, height))

        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
        ).images[0]

        if stop_signal:
            return None, "Generation stopped"

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"sd3_img2img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        return image_path, None

    finally:
        del pipe
        torch.cuda.empty_cache()


def generate_image_sd3_controlnet(prompt, negative_prompt, init_image, controlnet_model, num_inference_steps, guidance_scale, controlnet_conditioning_scale, width, height, max_sequence_length, output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    if not init_image:
        return None, None, "Please upload an initial image!"

    sd3_model_path = os.path.join("inputs", "image", "sd_models", "sd3")
    controlnet_path = os.path.join("inputs", "image", "sd_models", "controlnet", f"sd3_{controlnet_model}")

    if not os.path.exists(sd3_model_path):
        print("Downloading Stable Diffusion 3 model...")
        os.makedirs(sd3_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/v2ray/stable-diffusion-3-medium-diffusers", sd3_model_path)
        print("Stable Diffusion 3 model downloaded")

    if not os.path.exists(controlnet_path):
        print(f"Downloading SD3 ControlNet {controlnet_model} model...")
        os.makedirs(controlnet_path, exist_ok=True)
        Repo.clone_from(f"https://huggingface.co/InstantX/SD3-Controlnet-{controlnet_model}", controlnet_path)
        print(f"SD3 ControlNet {controlnet_model} model downloaded")

    try:
        controlnet = SD3ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            sd3_model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16
        )
        pipe.to("cuda")

        init_image = Image.open(init_image).convert("RGB")
        init_image = init_image.resize((width, height))

        if controlnet_model.lower() == "canny":
            control_image = init_image
        elif controlnet_model.lower() == "pose":
            control_image = init_image
        else:
            return None, None, f"Unsupported ControlNet model: {controlnet_model}"

        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            control_image=control_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            max_sequence_length=max_sequence_length,
        ).images[0]

        if stop_signal:
            return None, None, "Generation stopped"

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)

        image_filename = f"sd3_controlnet_{controlnet_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        control_image_filename = f"sd3_controlnet_{controlnet_model}_control_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        control_image_path = os.path.join(image_dir, control_image_filename)
        control_image.save(control_image_path, format=output_format.upper())

        return image_path, control_image_path, None

    finally:
        del pipe
        torch.cuda.empty_cache()


def generate_image_sd3_inpaint(prompt, negative_prompt, init_image, mask_image, num_inference_steps, guidance_scale, width, height, max_sequence_length, output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    sd3_model_path = os.path.join("inputs", "image", "sd_models", "sd3")

    if not os.path.exists(sd3_model_path):
        print("Downloading Stable Diffusion 3 model...")
        os.makedirs(sd3_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/v2ray/stable-diffusion-3-medium-diffusers", sd3_model_path)
        print("Stable Diffusion 3 model downloaded")

    try:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        text_encoder = T5EncoderModel.from_pretrained(
            sd3_model_path,
            subfolder="text_encoder_3",
            quantization_config=quantization_config,
        )
        pipe = StableDiffusion3InpaintPipeline.from_pretrained(sd3_model_path, device_map="balanced", text_encoder_3=text_encoder, torch_dtype=torch.float16)

        init_image = Image.open(init_image).convert("RGB")
        init_image = init_image.resize((width, height))

        mask_image = Image.open(mask_image).convert("L")
        mask_image = mask_image.resize((width, height))

        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
        ).images[0]

        if stop_signal:
            return None, "Generation stopped"

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"sd3_inpaint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        return image_path, None

    finally:
        del pipe
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


def generate_image_instantid(prompt, negative_prompt, face_image, stable_diffusion_model_name, controlnet_conditioning_scale, ip_adapter_scale, num_inference_steps, guidance_scale, width, height, output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    if not face_image:
        return None, "Please upload a face image!"

    if not stable_diffusion_model_name:
        return None, "Please select a StableDiffusion model!"

    instantid_model_path = os.path.join("inputs", "image", "sd_models", "instantid")
    if not os.path.exists(instantid_model_path):
        print("Downloading InstantID model...")
        os.makedirs(instantid_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/InstantX/InstantID", instantid_model_path)
        print("InstantID model downloaded")

    try:
        app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))

        face_adapter = os.path.join(instantid_model_path, "ip-adapter.bin")
        controlnet_path = os.path.join(instantid_model_path, "ControlNetModel")

        controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            stable_diffusion_model_name, controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe.cuda()
        pipe.load_ip_adapter_instantid(face_adapter)

        face_image = Image.open(face_image).convert("RGB")
        face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]
        face_emb = face_info['embedding']
        face_kps = draw_kps(face_image, face_info['kps'])

        pipe.set_ip_adapter_scale(ip_adapter_scale)

        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            image=face_kps,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height
        ).images[0]

        if stop_signal:
            return None, "Generation stopped"

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"InstantID_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"instantid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        return image_path, None

    except Exception as e:
        return None, str(e)

    finally:
        del pipe
        torch.cuda.empty_cache()


def generate_image_photomaker(prompt, negative_prompt, input_images, base_model_path, trigger_word, num_inference_steps,
                              guidance_scale, width, height, output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    if not input_images:
        return None, "Please upload at least one input image!"

    if not base_model_path:
        return None, "Please select a base model!"

    photomaker_path = os.path.join("inputs", "image", "sd_models", "photomaker")
    if not os.path.exists(photomaker_path):
        print("Downloading PhotoMaker model...")
        os.makedirs(photomaker_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/TencentARC/PhotoMaker-V2", photomaker_path)
        print("PhotoMaker model downloaded")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to(device)

        pipe.load_photomaker_adapter(
            photomaker_path,
            subfolder="",
            weight_name="photomaker-v2.bin",
            trigger_word=trigger_word
        )

        pipe.fuse_lora()

        input_id_images = [load_image(img) for img in input_images]

        generator = torch.Generator(device=device).manual_seed(42)
        image = pipe(
            prompt=prompt,
            input_id_images=input_id_images,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        ).images[0]

        if stop_signal:
            return None, "Generation stopped"

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"PhotoMaker_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"photomaker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        return image_path, None

    except Exception as e:
        return None, str(e)

    finally:
        del pipe
        torch.cuda.empty_cache()


def generate_image_ip_adapter_faceid(prompt, negative_prompt, face_image, base_model_path, ip_adapter_version,
                                     num_inference_steps, guidance_scale, width, height, output_format="png",
                                     stop_generation=None):
    global stop_signal
    stop_signal = False

    if not face_image:
        return None, "Please upload a face image!"

    if not base_model_path:
        return None, "Please select a base model!"

    ip_adapter_path = os.path.join("inputs", "image", "sd_models", "ip-adapter-faceid")
    if not os.path.exists(ip_adapter_path):
        print("Downloading IP-Adapter-FaceID model...")
        os.makedirs(ip_adapter_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/h94/IP-Adapter-FaceID", ip_adapter_path)
        print("IP-Adapter-FaceID model downloaded")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        image = cv2.imread(face_image)
        faces = app.get(image)
        faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0).to(device)

        pipe = StableDiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
        ).to(device)

        ip_ckpt = os.path.join(ip_adapter_path, f"{ip_adapter_version}.bin")
        ip_model = IPAdapterFaceID(pipe, ip_ckpt, device)

        images = ip_model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            faceid_embeds=faceid_embeds,
            num_samples=1,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        if stop_signal:
            return None, "Generation stopped"

        image = images[0]

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"IPAdapterFaceID_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"ip_adapter_faceid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        return image_path, None

    except Exception as e:
        return None, str(e)

    finally:
        del pipe
        del ip_model
        torch.cuda.empty_cache()


def generate_image_extras(input_image, source_image, remove_background, enable_faceswap, enable_facerestore, image_output_format, stop_generation):
    if not input_image:
        return None, "Please upload an image file!"

    if not remove_background and not enable_faceswap and not enable_facerestore:
        return None, "Please choose an option to modify the image"

    today = datetime.now().date()
    output_dir = os.path.join('outputs', f"Extras_{today.strftime('%Y%m%d')}")
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"background_removed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{image_output_format}"
    output_path = os.path.join(output_dir, output_filename)

    try:
        if remove_background:
            remove_bg(input_image, output_path)

        if enable_faceswap:
            if not source_image:
                return None, "Please upload a source image for faceswap!"

            roop_model_path = os.path.join("inputs", "image", "roop")

            if not os.path.exists(roop_model_path):
                print("Downloading roop model...")
                os.makedirs(roop_model_path, exist_ok=True)
                Repo.clone_from("https://github.com/s0md3v/roop", roop_model_path)
                print("roop model downloaded")

            faceswap_output_path = os.path.join(output_dir, f"faceswapped_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{image_output_format}")

            command = f"python {os.path.join(roop_model_path, 'run.py')} --target {input_image} --source {source_image} --output {faceswap_output_path}"
            subprocess.run(command, shell=True, check=True)

            output_path = faceswap_output_path

        if enable_facerestore:
            codeformer_path = os.path.join("inputs", "image", "CodeFormer")

            facerestore_output_path = os.path.join(output_dir, f"facerestored_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{image_output_format}")

            command = f"python {os.path.join(codeformer_path, 'inference_codeformer.py')} -w 0.7 --bg_upsampler realesrgan --face_upsample --input_path {input_image} --output_path {facerestore_output_path}"
            subprocess.run(command, shell=True, check=True)

            output_path = facerestore_output_path

        return output_path, None

    except Exception as e:
        return None, str(e)


def generate_image_kandinsky_txt2img(prompt, negative_prompt, version, num_inference_steps, guidance_scale, height, width, output_format="png",
                             stop_generation=None):
    global stop_signal
    stop_signal = False

    if not prompt:
        return None, "Please enter a prompt!"

    kandinsky_model_path = os.path.join("inputs", "image", "sd_models", "kandinsky")

    if not os.path.exists(kandinsky_model_path):
        print(f"Downloading Kandinsky {version} model...")
        os.makedirs(kandinsky_model_path, exist_ok=True)
        if version == "2.1":
            Repo.clone_from("https://huggingface.co/kandinsky-community/kandinsky-2-1-prior",
                            os.path.join(kandinsky_model_path, "2-1-prior"))
            Repo.clone_from("https://huggingface.co/kandinsky-community/kandinsky-2-1",
                            os.path.join(kandinsky_model_path, "2-1"))
        elif version == "2.2":
            Repo.clone_from("https://huggingface.co/kandinsky-community/kandinsky-2-2-prior",
                            os.path.join(kandinsky_model_path, "2-2-prior"))
            Repo.clone_from("https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder",
                            os.path.join(kandinsky_model_path, "2-2-decoder"))
        elif version == "3":
            Repo.clone_from("https://huggingface.co/kandinsky-community/kandinsky-3",
                            os.path.join(kandinsky_model_path, "3"))
        print(f"Kandinsky {version} model downloaded")

    try:
        if version == "2.1":

            pipe_prior = KandinskyPriorPipeline.from_pretrained(os.path.join(kandinsky_model_path, "2-1-prior"))
            pipe_prior.to("cuda")

            out = pipe_prior(prompt, negative_prompt=negative_prompt)
            image_emb = out.image_embeds
            negative_image_emb = out.negative_image_embeds

            pipe = KandinskyPipeline.from_pretrained(os.path.join(kandinsky_model_path, "2-1"))
            pipe.to("cuda")

            image = pipe(
                prompt,
                negative_prompt=negative_prompt,
                image_embeds=image_emb,
                negative_image_embeds=negative_image_emb,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]

        elif version == "2.2":

            pipe_prior = KandinskyV22PriorPipeline.from_pretrained(os.path.join(kandinsky_model_path, "2-2-prior"))
            pipe_prior.to("cuda")

            image_emb, negative_image_emb = pipe_prior(prompt, negative_prompt=negative_prompt).to_tuple()

            pipe = KandinskyV22Pipeline.from_pretrained(os.path.join(kandinsky_model_path, "2-2-decoder"))
            pipe.to("cuda")

            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image_embeds=image_emb,
                negative_image_embeds=negative_image_emb,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]

        elif version == "3":

            pipe = AutoPipelineForText2Image.from_pretrained(
                os.path.join(kandinsky_model_path, "3"), variant="fp16", torch_dtype=torch.float16
            )
            pipe.enable_model_cpu_offload()

            generator = torch.Generator(device="cpu").manual_seed(0)
            image = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                generator=generator,
                guidance_scale=guidance_scale,
            ).images[0]

        if stop_signal:
            return None, "Generation stopped"

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"Kandinsky_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"kandinsky_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        return image_path, None

    except Exception as e:
        return None, str(e)

    finally:
        try:
            del pipe_prior
            del pipe
        except:
            pass
        torch.cuda.empty_cache()


def generate_image_kandinsky_img2img(prompt, negative_prompt, init_image, version, num_inference_steps, guidance_scale, strength, height, width, output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    if not prompt or not init_image:
        return None, "Please enter a prompt and upload an initial image!"

    kandinsky_model_path = os.path.join("inputs", "image", "sd_models", "kandinsky")

    if not os.path.exists(kandinsky_model_path):
        print(f"Downloading Kandinsky {version} model...")
        os.makedirs(kandinsky_model_path, exist_ok=True)
        if version == "2.1":
            Repo.clone_from("https://huggingface.co/kandinsky-community/kandinsky-2-1-prior",
                            os.path.join(kandinsky_model_path, "2-1-prior"))
            Repo.clone_from("https://huggingface.co/kandinsky-community/kandinsky-2-1",
                            os.path.join(kandinsky_model_path, "2-1"))
        elif version == "2.2":
            Repo.clone_from("https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder",
                            os.path.join(kandinsky_model_path, "2-2-decoder"))
        elif version == "3":
            Repo.clone_from("https://huggingface.co/kandinsky-community/kandinsky-3",
                            os.path.join(kandinsky_model_path, "3"))
        print(f"Kandinsky {version} model downloaded")

    try:
        if version == "2.1":
            pipe_prior = KandinskyPriorPipeline.from_pretrained(
                os.path.join(kandinsky_model_path, "2-1-prior"), torch_dtype=torch.float16
            )
            pipe_prior.to("cuda")

            image_emb, zero_image_emb = pipe_prior(prompt, negative_prompt=negative_prompt, return_dict=False)

            pipe = KandinskyImg2ImgPipeline.from_pretrained(
                os.path.join(kandinsky_model_path, "2-1"), torch_dtype=torch.float16
            )
            pipe.to("cuda")

            init_image = Image.open(init_image).convert("RGB")
            init_image = init_image.resize((width, height))

            image = pipe(
                prompt,
                image=init_image,
                image_embeds=image_emb,
                negative_image_embeds=zero_image_emb,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
            ).images[0]

        elif version == "2.2":
            pipe = AutoPipelineForImage2Image.from_pretrained(
                os.path.join(kandinsky_model_path, "2-2-decoder"), torch_dtype=torch.float16
            )
            pipe.enable_model_cpu_offload()

            init_image = Image.open(init_image).convert("RGB")
            init_image = init_image.resize((width, height))

            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
            ).images[0]

        elif version == "3":
            pipe = AutoPipelineForImage2Image.from_pretrained(
                os.path.join(kandinsky_model_path, "3"), variant="fp16", torch_dtype=torch.float16
            )
            pipe.enable_model_cpu_offload()

            init_image = Image.open(init_image).convert("RGB")
            init_image = init_image.resize((width, height))

            generator = torch.Generator(device="cpu").manual_seed(0)
            image = pipe(
                prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=generator,
            ).images[0]

        if stop_signal:
            return None, "Generation stopped"

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"Kandinsky_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"kandinsky_{version}_img2img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        return image_path, None

    except Exception as e:
        return None, str(e)

    finally:
        try:
            del pipe_prior
            del pipe
        except:
            pass
        torch.cuda.empty_cache()


def generate_image_kandinsky_inpaint(prompt, negative_prompt, init_image, mask_image, version, num_inference_steps, guidance_scale, strength, height, width, output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    if not prompt or not init_image or not mask_image:
        return None, "Please enter a prompt, upload an initial image, and provide a mask image!"

    kandinsky_model_path = os.path.join("inputs", "image", "sd_models", "kandinsky")

    if not os.path.exists(kandinsky_model_path):
        print(f"Downloading Kandinsky {version} model...")
        os.makedirs(kandinsky_model_path, exist_ok=True)
        if version == "2.1":
            Repo.clone_from("https://huggingface.co/kandinsky-community/kandinsky-2-1-inpainter",
                            os.path.join(kandinsky_model_path, "2-1-inpainter"))
        elif version == "2.2":
            Repo.clone_from("https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint",
                            os.path.join(kandinsky_model_path, "2-2-decoder-inpaint"))
        print(f"Kandinsky {version} inpainting model downloaded")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if version == "2.1":
            pipe = AutoPipelineForInpainting.from_pretrained(
                os.path.join(kandinsky_model_path, "2-1-inpainter"),
                torch_dtype=torch.float16,
                variant="fp16"
            )
        else:  # version 2.2
            pipe = AutoPipelineForInpainting.from_pretrained(
                os.path.join(kandinsky_model_path, "2-2-decoder-inpaint"),
                torch_dtype=torch.float16
            )

        pipe.to(device)
        pipe.enable_model_cpu_offload()
        if version == "2.1":
            pipe.enable_xformers_memory_efficient_attention()

        init_image = Image.open(init_image).convert("RGB")
        init_image = init_image.resize((width, height))

        mask_image = Image.open(mask_image).convert("L")
        mask_image = mask_image.resize((width, height))

        generator = torch.Generator(device).manual_seed(0)

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=generator,
        ).images[0]

        if stop_signal:
            return None, "Generation stopped"

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"Kandinsky_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"kandinsky_{version}_inpaint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        return image_path, None

    except Exception as e:
        return None, str(e)

    finally:
        del pipe
        torch.cuda.empty_cache()


def generate_image_flux(prompt, model_name, guidance_scale, height, width, num_inference_steps, max_sequence_length, output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    if not model_name:
        return None, "Please select a Flux model!"

    flux_model_path = os.path.join("inputs", "image", "sd_models", "flux", model_name)

    if not os.path.exists(flux_model_path):
        print(f"Downloading Flux {model_name} model...")
        os.makedirs(flux_model_path, exist_ok=True)
        Repo.clone_from(f"https://huggingface.co/black-forest-labs/{model_name}", flux_model_path)
        print(f"Flux {model_name} model downloaded")

    try:
        pipe = FluxPipeline.from_pretrained(flux_model_path, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        pipe.to(torch.float16)

        if model_name == "FLUX.1-schnell":
            out = pipe(
                prompt=prompt,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                max_sequence_length=max_sequence_length,
            ).images[0]
        else:  # FLUX.1-dev
            out = pipe(
                prompt=prompt,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                max_sequence_length=max_sequence_length,
            ).images[0]

        if stop_signal:
            return None, "Generation stopped"

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"Flux_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"flux_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        out.save(image_path, format=output_format.upper())

        return image_path, None

    except Exception as e:
        return None, str(e)

    finally:
        del pipe
        torch.cuda.empty_cache()


def generate_image_hunyuandit(prompt, negative_prompt, num_inference_steps, guidance_scale, height, width, output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    if not prompt:
        return None, "Please enter a prompt!"

    hunyuandit_model_path = os.path.join("inputs", "image", "sd_models", "hunyuandit")

    if not os.path.exists(hunyuandit_model_path):
        print("Downloading HunyuanDiT model...")
        os.makedirs(hunyuandit_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-Diffusers", hunyuandit_model_path)
        print("HunyuanDiT model downloaded")

    try:
        pipe = HunyuanDiTPipeline.from_pretrained(hunyuandit_model_path, torch_dtype=torch.float16)
        pipe.to("cuda")

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width
        ).images[0]

        if stop_signal:
            return None, "Generation stopped"

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"HunyuanDiT_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"hunyuandit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        return image_path, None

    except Exception as e:
        return None, str(e)

    finally:
        del pipe
        torch.cuda.empty_cache()


def generate_image_lumina(prompt, negative_prompt, num_inference_steps, guidance_scale, height, width, max_sequence_length, output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    if not prompt:
        return None, "Please enter a prompt!"

    lumina_model_path = os.path.join("inputs", "image", "sd_models", "lumina")

    if not os.path.exists(lumina_model_path):
        print("Downloading Lumina-T2X model...")
        os.makedirs(lumina_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/Alpha-VLLM/Lumina-Next-SFT-diffusers", lumina_model_path)
        print("Lumina-T2X model downloaded")

    try:
        pipe = LuminaText2ImgPipeline.from_pretrained(
            lumina_model_path, torch_dtype=torch.bfloat16
        ).cuda()
        pipe.enable_model_cpu_offload()

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            max_sequence_length=max_sequence_length
        ).images[0]

        if stop_signal:
            return None, "Generation stopped"

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"Lumina_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"lumina_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        return image_path, None

    except Exception as e:
        return None, str(e)

    finally:
        del pipe
        torch.cuda.empty_cache()


def generate_image_kolors(prompt, negative_prompt, guidance_scale, num_inference_steps, max_sequence_length, output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    if not prompt:
        return None, "Please enter a prompt!"

    kolors_model_path = os.path.join("inputs", "image", "sd_models", "kolors")

    if not os.path.exists(kolors_model_path):
        print("Downloading Kolors model...")
        os.makedirs(kolors_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/Kwai-Kolors/Kolors-diffusers", kolors_model_path)
        print("Kolors model downloaded")

    try:
        pipe = KolorsPipeline.from_pretrained(kolors_model_path, torch_dtype=torch.float16, variant="fp16")
        pipe.to("cuda")

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length
        ).images[0]

        if stop_signal:
            return None, "Generation stopped"

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"Kolors_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"kolors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        return image_path, None

    except Exception as e:
        return None, str(e)

    finally:
        del pipe
        torch.cuda.empty_cache()


def generate_image_auraflow(prompt, negative_prompt, num_inference_steps, guidance_scale, height, width, max_sequence_length, enable_aurasr, output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    if not prompt:
        return None, "Please enter a prompt!"

    auraflow_model_path = os.path.join("inputs", "image", "sd_models", "auraflow")
    aurasr_model_path = os.path.join("inputs", "image", "sd_models", "auraflow", "aurasr")

    if not os.path.exists(auraflow_model_path):
        print("Downloading AuraFlow model...")
        os.makedirs(auraflow_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/fal/AuraFlow-v0.3", auraflow_model_path)
        print("AuraFlow model downloaded")

    if not os.path.exists(aurasr_model_path):
        print("Downloading AuraSR model...")
        os.makedirs(aurasr_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/fal/AuraSR", aurasr_model_path)
        print("AuraSR model downloaded")

    try:
        pipe = AuraFlowPipeline.from_pretrained(auraflow_model_path, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            max_sequence_length=max_sequence_length
        ).images[0]

        if stop_signal:
            return None, "Generation stopped"

        if enable_aurasr:
            aura_sr = AuraSR.from_pretrained(aurasr_model_path)
            image = image.resize((256, 256))
            image = aura_sr.upscale_4x_overlapped(image)

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"AuraFlow_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"auraflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        return image_path, None

    except Exception as e:
        return None, str(e)

    finally:
        del pipe
        torch.cuda.empty_cache()


def generate_image_wurstchen(prompt, negative_prompt, width, height, prior_steps, prior_guidance_scale, decoder_steps, decoder_guidance_scale, output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    if not prompt:
        return None, "Please enter a prompt!"

    wurstchen_model_path = os.path.join("inputs", "image", "sd_models", "wurstchen")

    if not os.path.exists(wurstchen_model_path):
        print("Downloading Würstchen models...")
        os.makedirs(wurstchen_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/warp-ai/wuerstchen-prior", os.path.join(wurstchen_model_path, "prior"))
        Repo.clone_from("https://huggingface.co/warp-ai/wuerstchen", os.path.join(wurstchen_model_path, "decoder"))
        print("Würstchen models downloaded")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        prior_pipeline = WuerstchenPriorPipeline.from_pretrained(
            os.path.join(wurstchen_model_path, "prior"),
            torch_dtype=dtype
        ).to(device)

        decoder_pipeline = WuerstchenDecoderPipeline.from_pretrained(
            os.path.join(wurstchen_model_path, "decoder"),
            torch_dtype=dtype
        ).to(device)

        prior_output = prior_pipeline(
            prompt=prompt,
            height=height,
            width=width,
            timesteps=DEFAULT_STAGE_C_TIMESTEPS,
            negative_prompt=negative_prompt,
            guidance_scale=prior_guidance_scale,
            num_inference_steps=prior_steps,
        )

        if stop_signal:
            return None, "Generation stopped"

        decoder_output = decoder_pipeline(
            image_embeddings=prior_output.image_embeddings,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=decoder_guidance_scale,
            num_inference_steps=decoder_steps,
            output_type="pil",
        ).images[0]

        if stop_signal:
            return None, "Generation stopped"

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"Wurstchen_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"wurstchen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        decoder_output.save(image_path, format=output_format.upper())

        return image_path, None

    except Exception as e:
        return None, str(e)

    finally:
        del prior_pipeline
        del decoder_pipeline
        torch.cuda.empty_cache()


def generate_image_deepfloyd_txt2img(prompt, negative_prompt, num_inference_steps, guidance_scale, width, height, output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    if not prompt:
        return None, None, None, "Please enter a prompt!"

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Stage I
        pipe_i = IFPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
        pipe_i.to(device)
        pipe_i.enable_model_cpu_offload()

        prompt_embeds, negative_embeds = pipe_i.encode_prompt(prompt)
        image = pipe_i(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            output_type="pt"
        ).images

        if stop_signal:
            return None, None, None, "Generation stopped"

        # Stage II
        pipe_ii = IFSuperResolutionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
        )
        pipe_ii.to(device)
        pipe_ii.enable_model_cpu_offload()

        image = pipe_ii(
            image=image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="pt"
        ).images

        if stop_signal:
            return None, None, None, "Generation stopped"

        # Stage III
        safety_modules = {
            "feature_extractor": pipe_i.feature_extractor,
            "safety_checker": pipe_i.safety_checker,
            "watermarker": pipe_i.watermarker,
        }
        pipe_iii = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16
        )
        pipe_iii.to(device)
        pipe_iii.enable_model_cpu_offload()

        image = pipe_iii(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        if stop_signal:
            return None, None, None, "Generation stopped"

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"DeepFloydIF_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)

        stage_i_filename = f"deepfloyd_if_stage_I_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        stage_ii_filename = f"deepfloyd_if_stage_II_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        stage_iii_filename = f"deepfloyd_if_stage_III_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"

        stage_i_path = os.path.join(image_dir, stage_i_filename)
        stage_ii_path = os.path.join(image_dir, stage_ii_filename)
        stage_iii_path = os.path.join(image_dir, stage_iii_filename)

        pt_to_pil(image[0])[0].save(stage_i_path)
        pt_to_pil(image[0])[0].save(stage_ii_path)
        image.save(stage_iii_path)

        return stage_i_path, stage_ii_path, stage_iii_path, None

    except Exception as e:
        return None, None, None, str(e)

    finally:
        try:
            del pipe_i
            del pipe_ii
            del pipe_iii
        except:
            pass
        torch.cuda.empty_cache()


def generate_image_deepfloyd_img2img(prompt, negative_prompt, init_image, num_inference_steps, guidance_scale, width, height, output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    if not prompt or not init_image:
        return None, None, None, "Please enter a prompt and upload an initial image!"

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Stage I
        stage_1 = IFImg2ImgPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
        stage_1.to(device)
        stage_1.enable_model_cpu_offload()

        # Stage II
        stage_2 = IFImg2ImgSuperResolutionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
        )
        stage_2.to(device)
        stage_2.enable_model_cpu_offload()

        # Stage III
        safety_modules = {
            "feature_extractor": stage_1.feature_extractor,
            "safety_checker": stage_1.safety_checker,
            "watermarker": stage_1.watermarker,
        }
        stage_3 = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16
        )
        stage_3.to(device)
        stage_3.enable_model_cpu_offload()

        original_image = Image.open(init_image).convert("RGB")
        original_image = original_image.resize((width, height))

        generator = torch.manual_seed(0)

        prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)

        # Stage I
        stage_1_output = stage_1(
            image=original_image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="pt",
        ).images

        if stop_signal:
            return None, None, None, "Generation stopped"

        # Stage II
        stage_2_output = stage_2(
            image=stage_1_output,
            original_image=original_image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="pt",
        ).images

        if stop_signal:
            return None, None, None, "Generation stopped"

        # Stage III
        stage_3_output = stage_3(
            prompt=prompt,
            image=stage_2_output,
            generator=generator,
            noise_level=100,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        if stop_signal:
            return None, None, None, "Generation stopped"

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"DeepFloydIF_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)

        stage_1_filename = f"deepfloyd_if_img2img_stage_I_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        stage_2_filename = f"deepfloyd_if_img2img_stage_II_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        stage_3_filename = f"deepfloyd_if_img2img_stage_III_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"

        stage_1_path = os.path.join(image_dir, stage_1_filename)
        stage_2_path = os.path.join(image_dir, stage_2_filename)
        stage_3_path = os.path.join(image_dir, stage_3_filename)

        pt_to_pil(stage_1_output)[0].save(stage_1_path)
        pt_to_pil(stage_2_output)[0].save(stage_2_path)
        stage_3_output.save(stage_3_path)

        return stage_1_path, stage_2_path, stage_3_path, None

    except Exception as e:
        return None, None, None, str(e)

    finally:
        try:
            del stage_1
            del stage_2
            del stage_3
        except:
            pass
        torch.cuda.empty_cache()

def generate_image_deepfloyd_inpaint(prompt, negative_prompt, init_image, mask_image, num_inference_steps, guidance_scale, output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    if not prompt or not init_image or not mask_image:
        return None, None, None, "Please enter a prompt, upload an initial image, and provide a mask image!"

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Stage I
        stage_1 = IFInpaintingPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
        stage_1.to(device)
        stage_1.enable_model_cpu_offload()

        # Stage II
        stage_2 = IFInpaintingSuperResolutionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
        )
        stage_2.to(device)
        stage_2.enable_model_cpu_offload()

        # Stage III
        safety_modules = {
            "feature_extractor": stage_1.feature_extractor,
            "safety_checker": stage_1.safety_checker,
            "watermarker": stage_1.watermarker,
        }
        stage_3 = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16
        )
        stage_3.to(device)
        stage_3.enable_model_cpu_offload()

        original_image = Image.open(init_image).convert("RGB")
        mask_image = Image.open(mask_image).convert("RGB")

        generator = torch.manual_seed(0)

        prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)

        # Stage I
        stage_1_output = stage_1(
            image=original_image,
            mask_image=mask_image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="pt",
        ).images

        if stop_signal:
            return None, None, None, "Generation stopped"

        # Stage II
        stage_2_output = stage_2(
            image=stage_1_output,
            original_image=original_image,
            mask_image=mask_image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="pt",
        ).images

        if stop_signal:
            return None, None, None, "Generation stopped"

        # Stage III
        stage_3_output = stage_3(
            prompt=prompt,
            image=stage_2_output,
            generator=generator,
            noise_level=100,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        if stop_signal:
            return None, None, None, "Generation stopped"

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"DeepFloydIF_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)

        stage_1_filename = f"deepfloyd_if_inpaint_stage_I_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        stage_2_filename = f"deepfloyd_if_inpaint_stage_II_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        stage_3_filename = f"deepfloyd_if_inpaint_stage_III_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"

        stage_1_path = os.path.join(image_dir, stage_1_filename)
        stage_2_path = os.path.join(image_dir, stage_2_filename)
        stage_3_path = os.path.join(image_dir, stage_3_filename)

        pt_to_pil(stage_1_output)[0].save(stage_1_path)
        pt_to_pil(stage_2_output)[0].save(stage_2_path)
        stage_3_output.save(stage_3_path)

        return stage_1_path, stage_2_path, stage_3_path, None

    except Exception as e:
        return None, None, None, str(e)

    finally:
        try:
            del stage_1
            del stage_2
            del stage_3
        except:
            pass
        torch.cuda.empty_cache()


def generate_image_pixart(prompt, negative_prompt, version, num_inference_steps, guidance_scale, height, width,
                          max_sequence_length, output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    if not prompt:
        return None, "Please enter a prompt!"

    pixart_model_path = os.path.join("inputs", "image", "sd_models", "pixart")

    if not os.path.exists(pixart_model_path):
        print(f"Downloading PixArt {version} model...")
        os.makedirs(pixart_model_path, exist_ok=True)
        if version == "Alpha-512":
            Repo.clone_from("https://huggingface.co/PixArt-alpha/PixArt-XL-2-512-MS",
                            os.path.join(pixart_model_path, "Alpha-512"))
        elif version == "Alpha-1024":
            Repo.clone_from("https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS",
                            os.path.join(pixart_model_path, "Alpha-1024"))
        elif version == "Sigma-512":
            Repo.clone_from("https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
                            os.path.join(pixart_model_path, "Sigma-512"))
        elif version == "Sigma-1024":
            Repo.clone_from("https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
                            os.path.join(pixart_model_path, "Sigma-1024"))
        print(f"PixArt {version} model downloaded")

    try:
        if version.startswith("Alpha"):
            pipe = PixArtAlphaPipeline.from_pretrained(os.path.join(pixart_model_path, version),
                                                       torch_dtype=torch.float16)
        else:
            pipe = PixArtSigmaPipeline.from_pretrained(os.path.join(pixart_model_path, version),
                                                       torch_dtype=torch.float16)

        pipe.enable_model_cpu_offload()

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            max_sequence_length=max_sequence_length
        ).images[0]

        if stop_signal:
            return None, "Generation stopped"

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"PixArt_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"pixart_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        return image_path, None

    except Exception as e:
        return None, str(e)

    finally:
        del pipe
        torch.cuda.empty_cache()


def generate_video_modelscope(prompt, negative_prompt, num_inference_steps, guidance_scale, height, width, num_frames,
                              output_format, stop_generation):
    global stop_signal
    stop_signal = False

    if not prompt:
        return None, "Please enter a prompt!"

    modelscope_model_path = os.path.join("inputs", "video", "modelscope")

    if not os.path.exists(modelscope_model_path):
        print("Downloading ModelScope model...")
        os.makedirs(modelscope_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/damo-vilab/text-to-video-ms-1.7b", modelscope_model_path)
        print("ModelScope model downloaded")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = DiffusionPipeline.from_pretrained(modelscope_model_path, torch_dtype=torch.float16, variant="fp16")
        pipe = pipe.to(device)
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()

        video_frames = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_frames=num_frames
        ).frames[0]

        if stop_signal:
            return None, "Generation stopped"

        today = datetime.now().date()
        video_dir = os.path.join('outputs', f"ModelScope_{today.strftime('%Y%m%d')}")
        os.makedirs(video_dir, exist_ok=True)

        video_filename = f"modelscope_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        video_path = os.path.join(video_dir, video_filename)

        if output_format == "mp4":
            export_to_video(video_frames, video_path)
        else:
            export_to_gif(video_frames, video_path)

        return video_path, None

    except Exception as e:
        return None, str(e)

    finally:
        del pipe
        torch.cuda.empty_cache()


def generate_video_zeroscope2(prompt, video_to_enhance, strength, num_inference_steps, width, height, num_frames,
                              enable_video_enhance, stop_generation):
    global stop_signal
    stop_signal = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model_path = os.path.join("inputs", "video", "zeroscope2", "zeroscope_v2_576w")
    if not os.path.exists(base_model_path):
        print("Downloading ZeroScope 2 base model...")
        os.makedirs(base_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/cerspense/zeroscope_v2_576w", base_model_path)
        print("ZeroScope 2 base model downloaded")

    enhance_model_path = os.path.join("inputs", "video", "zeroscope2", "zeroscope_v2_XL")
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


def generate_video_cogvideox(prompt, negative_prompt, num_inference_steps, guidance_scale, height, width, num_frames, fps, stop_generation):
    global stop_signal
    stop_signal = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cogvideox_model_path = os.path.join("inputs", "video", "cogvideox")

    if not os.path.exists(cogvideox_model_path):
        print("Downloading CogVideoX model...")
        os.makedirs(cogvideox_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/THUDM/CogVideoX-2b", cogvideox_model_path)
        print("CogVideoX model downloaded")

    try:
        pipe = CogVideoXPipeline.from_pretrained(cogvideox_model_path, torch_dtype=torch.float16).to(device)
        pipe.enable_model_cpu_offload()

        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_frames=num_frames
        ).frames[0]

        if stop_signal:
            return None, "Generation stopped"

        today = datetime.now().date()
        video_dir = os.path.join('outputs', f"CogVideoX_{today.strftime('%Y%m%d')}")
        os.makedirs(video_dir, exist_ok=True)

        video_filename = f"cogvideox_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        video_path = os.path.join(video_dir, video_filename)
        export_to_video(video, video_path, fps=fps)

        return video_path, None

    except Exception as e:
        return None, str(e)

    finally:
        del pipe
        torch.cuda.empty_cache()

def generate_video_latte(prompt, negative_prompt, num_inference_steps, guidance_scale, height, width, video_length, stop_generation):
    global stop_signal
    stop_signal = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    latte_model_path = os.path.join("inputs", "video", "latte")

    if not os.path.exists(latte_model_path):
        print("Downloading Latte model...")
        os.makedirs(latte_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/maxin-cn/Latte-1", latte_model_path)
        print("Latte model downloaded")

    try:
        pipe = LattePipeline.from_pretrained(latte_model_path, torch_dtype=torch.float16).to(device)
        pipe.enable_model_cpu_offload()

        videos = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            video_length=video_length
        ).frames[0]

        if stop_signal:
            return None, "Generation stopped"

        today = datetime.now().date()
        gif_dir = os.path.join('outputs', f"Latte_{today.strftime('%Y%m%d')}")
        os.makedirs(gif_dir, exist_ok=True)

        gif_filename = f"latte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
        gif_path = os.path.join(gif_dir, gif_filename)
        export_to_gif(videos, gif_path)

        return gif_path, None

    except Exception as e:
        return None, str(e)

    finally:
        del pipe
        torch.cuda.empty_cache()


def generate_3d_triposr(image, mc_resolution, foreground_ratio=0.85, output_format="obj", stop_generation=None):
    global stop_signal
    stop_signal = False

    model_path = os.path.join("inputs", "3D", "triposr")

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


def generate_3d_stablefast3d(image, texture_resolution, foreground_ratio, remesh_option, output_format="obj",
                             stop_generation=None):
    global stop_signal
    stop_signal = False

    if not image:
        return None, "Please upload an image!"

    hf_token = get_hf_token()
    if hf_token is None:
        return None, "Hugging Face token not found. Please create a file named 'HF-Token.txt' in the root directory and paste your token there."

    try:
        today = datetime.now().date()
        output_dir = os.path.join('outputs', f"StableFast3D_{today.strftime('%Y%m%d')}")
        os.makedirs(output_dir, exist_ok=True)

        output_filename = f"3d_object_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        output_path = os.path.join(output_dir, output_filename)

        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

        command = f"python StableFast3D/run.py \"{image}\" --output-dir {output_dir} --texture-resolution {texture_resolution} --foreground-ratio {foreground_ratio} --remesh_option {remesh_option}"

        subprocess.run(command, shell=True, check=True)

        if stop_signal:
            return None, "Generation stopped"

        return output_path, None

    except Exception as e:
        return None, str(e)

    finally:
        if "HUGGING_FACE_HUB_TOKEN" in os.environ:
            del os.environ["HUGGING_FACE_HUB_TOKEN"]


def generate_3d_shap_e(prompt, init_image, num_inference_steps, guidance_scale, frame_size, stop_generation):
    global stop_signal
    stop_signal = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if init_image:
        model_name = "openai/shap-e-img2img"
        model_path = os.path.join("inputs", "3D", "shap-e", "img2img")
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
        model_path = os.path.join("inputs", "3D", "shap-e", "text2img")
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

    try:
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

    finally:
        del pipe
        torch.cuda.empty_cache()


def generate_sv34d(input_file, version, elevation_deg=None, stop_generation=None):
    global stop_signal
    stop_signal = False

    if not input_file:
        return None, "Please upload an input file!"

    model_files = {
        "3D-U": "https://huggingface.co/stabilityai/sv3d/resolve/main/sv3d_u.safetensors",
        "3D-P": "https://huggingface.co/stabilityai/sv3d/resolve/main/sv3d_p.safetensors",
        "4D": "https://huggingface.co/stabilityai/sv4d/resolve/main/sv4d.safetensors"
    }

    checkpoints_dir = "checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)
    model_path = os.path.join(checkpoints_dir, f"sv{version.lower()}.safetensors")

    if not os.path.exists(model_path):
        print(f"Downloading SV34D {version} model...")
        hf_token = get_hf_token()
        if hf_token is None:
            return None, "Hugging Face token not found. Please create a file named 'HF-Token.txt' in the root directory and paste your token there."

        try:
            response = requests.get(model_files[version], headers={"Authorization": f"Bearer {hf_token}"})
            response.raise_for_status()
            with open(model_path, 'wb') as f:
                f.write(response.content)
            print(f"SV34D {version} model downloaded")
        except Exception as e:
            return None, f"Error downloading model: {str(e)}"

    today = datetime.now().date()
    output_dir = os.path.join('outputs', '3D', f"SV34D_{today.strftime('%Y%m%d')}")
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"sv34d_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = os.path.join(output_dir, output_filename)

    if version in ["3D-U", "3D-P"]:
        if not input_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            return None, "Please upload an image file for 3D-U or 3D-P version!"

        if version == "3D-U":
            command = f"python generative-models/scripts/sampling/simple_video_sample.py --input_path {input_file} --version sv3d_u --output_folder {output_dir}"
        else:  # 3D-P
            if elevation_deg is None:
                return None, "Please provide elevation degree for 3D-P version!"
            command = f"python generative-models/scripts/sampling/simple_video_sample.py --input_path {input_file} --version sv3d_p --elevations_deg {elevation_deg} --output_folder {output_dir}"
    elif version == "4D":
        if not input_file.lower().endswith('.mp4'):
            return None, "Please upload an MP4 video file for 4D version!"
        command = f"python generative-models/scripts/sampling/simple_video_sample_4d.py --input_path {input_file} --output_folder {output_dir}"
    else:
        return None, "Invalid version selected!"

    try:
        subprocess.run(command, shell=True, check=True)

        if stop_signal:
            return None, "Generation stopped"

        for file in os.listdir(output_dir):
            if file.startswith(output_filename):
                return os.path.join(output_dir, file), None

        return None, "Output file not found"

    except subprocess.CalledProcessError as e:
        return None, f"Error occurred: {str(e)}"

    except Exception as e:
        return None, str(e)


def generate_3d_zero123plus(input_image, num_inference_steps, output_format="png", stop_generation=None):
    global stop_signal
    stop_signal = False

    if not input_image:
        return None, "Please upload an input image!"

    zero123plus_model_path = os.path.join("inputs", "3D", "zero123plus")

    if not os.path.exists(zero123plus_model_path):
        print("Downloading Zero123Plus model...")
        os.makedirs(zero123plus_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/sudo-ai/zero123plus-v1.2", zero123plus_model_path)
        print("Zero123Plus model downloaded")

    try:
        pipeline = DiffusionPipeline.from_pretrained(
            zero123plus_model_path,
            custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch.float16
        )
        pipeline.to('cuda:0')

        cond = Image.open(input_image)
        result = pipeline(cond, num_inference_steps=num_inference_steps).images[0]

        if stop_signal:
            return None, "Generation stopped"

        today = datetime.now().date()
        output_dir = os.path.join('outputs', f"Zero123Plus_{today.strftime('%Y%m%d')}")
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"zero123plus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        output_path = os.path.join(output_dir, output_filename)
        result.save(output_path)

        return output_path, None

    except Exception as e:
        return None, str(e)

    finally:
        del pipeline
        torch.cuda.empty_cache()


def generate_stableaudio(prompt, negative_prompt, num_inference_steps, guidance_scale, audio_length, audio_start, num_waveforms, output_format,
                         stop_generation):
    global stop_signal
    stop_signal = False

    sa_model_path = os.path.join("inputs", "audio", "stableaudio")

    if not os.path.exists(sa_model_path):
        print("Downloading Stable Audio Open model...")
        os.makedirs(sa_model_path, exist_ok=True)

        hf_token = get_hf_token()
        if hf_token is None:
            return None, None, "Hugging Face token not found. Please create a file named 'hftoken.txt' in the root directory and paste your token there."

        try:
            snapshot_download(repo_id="stabilityai/stable-audio-open-1.0",
                              local_dir=sa_model_path,
                              token=hf_token)
            print("Stable Audio Open model downloaded")
        except Exception as e:
            return None, None, f"Error downloading model: {str(e)}"

    pipe = StableAudioPipeline.from_pretrained(sa_model_path, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    generator = torch.Generator("cuda").manual_seed(0)

    try:
        audio = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            audio_end_in_s=audio_length,
            audio_start_in_s=audio_start,
            num_waveforms_per_prompt=num_waveforms,
            generator=generator,
        ).audios

        if stop_signal:
            return None, None, "Generation stopped"

        output = audio[0].T.float().cpu().numpy()

        today = datetime.now().date()
        audio_dir = os.path.join('outputs', f"StableAudio_{today.strftime('%Y%m%d')}")
        os.makedirs(audio_dir, exist_ok=True)
        audio_filename = f"stableaudio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        audio_path = os.path.join(audio_dir, audio_filename)

        if output_format == "mp3":
            sf.write(audio_path, output, pipe.vae.sampling_rate, format='mp3')
        elif output_format == "ogg":
            sf.write(audio_path, output, pipe.vae.sampling_rate, format='ogg')
        else:
            sf.write(audio_path, output, pipe.vae.sampling_rate)

        spectrogram_path = generate_mel_spectrogram(audio_path)

        return audio_path, spectrogram_path, None

    except Exception as e:
        return None, None, str(e)

    finally:
        del pipe
        torch.cuda.empty_cache()


def generate_audio_audiocraft(prompt, input_audio=None, model_name=None, audiocraft_settings_html=None, model_type="musicgen",
                              duration=10, top_k=250, top_p=0.0,
                              temperature=1.0, cfg_coef=3.0, enable_multiband=False, output_format="mp3", stop_generation=None):
    global audiocraft_model_path, stop_signal
    stop_signal = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not model_name:
        return None, None, "Please, select an AudioCraft model!"

    if enable_multiband and model_type in ["audiogen", "magnet"]:
        return None, None, "Multiband Diffusion is not supported with 'audiogen' or 'magnet' model types. Please select 'musicgen' or disable Multiband Diffusion"

    if not audiocraft_model_path:
        audiocraft_model_path = load_audiocraft_model(model_name)

    today = datetime.now().date()
    audio_dir = os.path.join('outputs', f"AudioCraft_{today.strftime('%Y%m%d')}")
    os.makedirs(audio_dir, exist_ok=True)

    try:
        if model_type == "musicgen":
            model = MusicGen.get_pretrained(audiocraft_model_path)
            model.set_generation_params(duration=duration)
        elif model_type == "audiogen":
            model = AudioGen.get_pretrained(audiocraft_model_path)
            model.set_generation_params(duration=duration)
        elif model_type == "magnet":
            model = MAGNeT.get_pretrained(audiocraft_model_path)
            model.set_generation_params()
        else:
            return None, None, "Invalid model type!"
    except (ValueError, AssertionError):
        return None, None, "The selected model is not compatible with the chosen model type"

    mbd = None

    if enable_multiband:
        mbd = MultiBandDiffusion.get_mbd_musicgen()

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
                return None, None, "Generation stopped"
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
                return None, None, "Generation stopped"
        progress_bar.close()

        if mbd:
            if stop_signal:
                return None, None, "Generation stopped"
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

        spectrogram_path = generate_mel_spectrogram(audio_path)

        if output_format == "mp3":
            return audio_path + ".mp3", spectrogram_path, None
        elif output_format == "ogg":
            return audio_path + ".ogg", spectrogram_path, None
        else:
            return audio_path + ".wav", spectrogram_path, None

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
        return None, None, "Please, select an AudioLDM 2 model!"

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
            return None, None, "Generation stopped"

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

        spectrogram_path = generate_mel_spectrogram(audio_path)

        return audio_path, spectrogram_path, None,

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


def get_output_files():
    output_dir = "outputs"
    text_files = []
    image_files = []
    video_files = []
    audio_files = []
    model3d_files = []

    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".txt") or file.endswith(".json"):
                text_files.append(os.path.join(root, file))
            elif file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".gif"):
                image_files.append(os.path.join(root, file))
            elif file.endswith(".mp4"):
                video_files.append(os.path.join(root, file))
            elif file.endswith(".wav") or file.endswith(".mp3") or file.endswith(".ogg"):
                audio_files.append(os.path.join(root, file))
            elif file.endswith(".obj") or file.endswith(".ply") or file.endswith(".glb"):
                model3d_files.append(os.path.join(root, file))

    def display_output_file(text_file, image_file, video_file, audio_file, model3d_file):
        if text_file:
            with open(text_file, "r") as f:
                text_content = f.read()
            return text_content, None, None, None, None
        elif image_file:
            return None, image_file, None, None, None
        elif video_file:
            return None, None, video_file, None, None
        elif audio_file:
            return None, None, None, audio_file, None
        elif model3d_file:
            return None, None, None, None, model3d_file
        else:
            return None, None, None, None, None

    return text_files, image_files, video_files, audio_files, model3d_files, display_output_file


def download_model(model_name_llm, model_name_sd):
    if not model_name_llm and not model_name_sd:
        return "Please select a model to download"

    if model_name_llm and model_name_sd:
        return "Please select one model type for downloading"

    if model_name_llm:
        model_url = ""
        if model_name_llm == "StarlingLM(Transformers7B)":
            model_url = "https://huggingface.co/Nexusflow/Starling-LM-7B-beta"
        elif model_name_llm == "OpenChat(Llama7B.Q4)":
            model_url = "https://huggingface.co/TheBloke/openchat-3.5-0106-GGUF/resolve/main/openchat-3.5-0106.Q4_K_M.gguf"
        model_path = os.path.join("inputs", "text", "llm_models", model_name_llm)

        if model_url:
            if model_name_llm == "StarlingLM(Transformers7B)":
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


llm_models_list = [None, "moondream2"] + [model for model in os.listdir("inputs/text/llm_models") if not model.endswith(".txt") and model != "vikhyatk" and model != "lora"]
llm_lora_models_list = [None] + [model for model in os.listdir("inputs/text/llm_models/lora") if not model.endswith(".txt")]
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
controlnet_models_list = [None, "openpose", "depth", "canny", "lineart", "scribble"]

chat_interface = gr.Interface(
    fn=generate_text_and_speech,
    inputs=[
        gr.Textbox(label="요청을 입력하세요"),
        gr.Audio(type="filepath", label="요청을 녹음하세요 (선택사항)"),
        gr.Image(label="이미지를 업로드하세요 (선택사항)", type="filepath"),
        gr.Dropdown(choices=llm_models_list, label="LLM 모델 선택", value=None),
        gr.Dropdown(choices=llm_lora_models_list, label="LoRA 모델 선택 (선택사항)", value=None),
        gr.HTML("<h3>LLM 설정</h3>"),
        gr.Radio(choices=["transformers", "llama"], label="모델 유형 선택", value="transformers"),
        gr.Slider(minimum=1, maximum=4096, value=512, step=1, label="최대 길이 (transformers 유형 모델용)"),
        gr.Slider(minimum=1, maximum=4096, value=512, step=1, label="최대 토큰 수 (llama 유형 모델용)"),
        gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="온도"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.1, label="Top P"),
        gr.Slider(minimum=0, maximum=100, value=20, step=1, label="Top K"),
        gr.Radio(choices=["txt", "json"], label="채팅 기록 형식 선택", value="txt", interactive=True),
        gr.Checkbox(label="웹 검색 활성화", value=False),
        gr.Checkbox(label="LibreTranslate 활성화", value=False),
        gr.Dropdown(choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hi"], label="대상 언어 선택", value="ru", interactive=True),
        gr.Checkbox(label="다중 모달 활성화", value=False),
        gr.Checkbox(label="TTS 활성화", value=False),
        gr.HTML("<h3>TTS 설정</h3>"),
        gr.Dropdown(choices=speaker_wavs_list, label="음성 선택", interactive=True),
        gr.Dropdown(choices=["en", "ru"], label="언어 선택", interactive=True),
        gr.Slider(minimum=0.0, maximum=1.9, value=1.0, step=0.1, label="TTS 온도", interactive=True),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.1, label="TTS Top P", interactive=True),
        gr.Slider(minimum=0, maximum=100, value=20, step=1, label="TTS Top K", interactive=True),
        gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="TTS 속도", interactive=True),
        gr.Radio(choices=["wav", "mp3", "ogg"], label="출력 형식 선택", value="wav", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Chatbot(label="LLM 텍스트 응답", value=[]),
        gr.Audio(label="LLM 오디오 응답", type="filepath"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - LLM",
    description="이 사용자 인터페이스를 사용하면 텍스트나 오디오를 입력하여 생성된 응답을 받을 수 있습니다. 드롭다운 목록에서 LLM 모델, 아바타, 음성 및 TTS 언어를 선택할 수 있습니다. 슬라이더를 사용하여 모델 설정을 사용자 지정할 수도 있습니다. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

tts_stt_interface = gr.Interface(
    fn=generate_tts_stt,
    inputs=[
        gr.Textbox(label="TTS용 텍스트 입력"),
        gr.Audio(label="STT용 오디오 녹음", type="filepath"),
        gr.HTML("<h3>TTS 설정</h3>"),
        gr.Dropdown(choices=speaker_wavs_list, label="음성 선택", interactive=True),
        gr.Dropdown(choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"], label="언어 선택", interactive=True),
        gr.Slider(minimum=0.0, maximum=1.9, value=1.0, step=0.1, label="TTS 온도", interactive=True),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.1, label="TTS Top P", interactive=True),
        gr.Slider(minimum=0, maximum=100, value=20, step=1, label="TTS Top K", interactive=True),
        gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="TTS 속도", interactive=True),
        gr.Radio(choices=["wav", "mp3", "ogg"], label="TTS 출력 형식 선택", value="wav", interactive=True),
        gr.Dropdown(choices=["txt", "json"], label="STT 출력 형식 선택", value="txt", interactive=True),
    ],
    outputs=[
        gr.Audio(label="TTS 오디오", type="filepath"),
        gr.Textbox(label="STT 텍스트"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - TTS-STT",
    description="이 사용자 인터페이스를 사용하면 텍스트를 음성으로 변환(CoquiTTS)하고 오디오를 텍스트로 변환(OpenAIWhisper)할 수 있습니다. TTS의 경우 음성과 언어를 선택하고 슬라이더를 사용하여 생성 설정을 사용자 지정할 수 있습니다. STT의 경우 오디오를 녹음하면 말한 텍스트가 표시됩니다. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

bark_interface = gr.Interface(
    fn=generate_bark_audio,
    inputs=[
        gr.Textbox(label="요청 텍스트 입력"),
        gr.Dropdown(choices=[None, "v2/en_speaker_1", "v2/ru_speaker_1", "v2/de_speaker_1", "v2/fr_speaker_1", "v2/es_speaker_1", "v2/hi_speaker_1", "v2/it_speaker_1", "v2/ja_speaker_1", "v2/ko_speaker_1", "v2/pt_speaker_1", "v2/zh_speaker_1", "v2/tr_speaker_1", "v2/pl_speaker_1"], label="음성 프리셋 선택", value=None),
        gr.Slider(minimum=1, maximum=1000, value=100, step=1, label="최대 길이"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.4, step=0.1, label="미세 온도"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="대략적 온도"),
        gr.Radio(choices=["wav", "mp3", "ogg"], label="출력 형식 선택", value="wav", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Audio(label="생성된 오디오", type="filepath"),
        gr.Image(label="멜 스펙트로그램", type="filepath"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - SunoBark",
    description="이 사용자 인터페이스를 사용하면 텍스트를 입력하고 SunoBark를 사용하여 오디오를 생성할 수 있습니다. 음성 프리셋을 선택하고 최대 길이를 사용자 지정할 수 있습니다. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

translate_interface = gr.Interface(
    fn=translate_text,
    inputs=[
        gr.Textbox(label="번역할 텍스트 입력"),
        gr.Dropdown(choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hi"], label="원본 언어 선택", value="en"),
        gr.Dropdown(choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hi"], label="대상 언어 선택", value="ru"),
        gr.Checkbox(label="번역 기록 저장 활성화", value=False),
        gr.Radio(choices=["txt", "json"], label="번역 기록 형식 선택", value="txt", interactive=True),
        gr.File(label="텍스트 파일 업로드 (선택사항)", file_count="single", interactive=True),
    ],
    outputs=[
        gr.Textbox(label="번역된 텍스트"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - LibreTranslate",
    description="이 사용자 인터페이스를 사용하면 텍스트를 입력하고 LibreTranslate를 사용하여 번역할 수 있습니다. 원본 언어와 대상 언어를 선택하고 제출을 클릭하여 번역을 얻으세요. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

wav2lip_interface = gr.Interface(
    fn=generate_wav2lip,
    inputs=[
        gr.Image(label="입력 이미지", type="filepath"),
        gr.Audio(label="입력 오디오", type="filepath"),
        gr.Slider(minimum=1, maximum=60, value=30, step=1, label="FPS"),
        gr.Textbox(label="패딩", value="0 10 0 0"),
        gr.Slider(minimum=1, maximum=64, value=16, step=1, label="얼굴 감지 배치 크기"),
        gr.Slider(minimum=1, maximum=512, value=128, step=1, label="Wav2Lip 배치 크기"),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label="크기 조정 요소"),
        gr.Textbox(label="자르기", value="0 -1 0 -1"),
    ],
    outputs=[
        gr.Video(label="생성된 립싱크 비디오"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - Wav2Lip",
    description="이 사용자 인터페이스를 사용하면 Wav2Lip을 사용하여 이미지와 오디오 파일을 결합하여 말하는 헤드 비디오를 생성할 수 있습니다. 이미지와 오디오 파일을 업로드하고 생성을 클릭하여 말하는 헤드 비디오를 만드세요. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

txt2img_interface = gr.Interface(
    fn=generate_image_txt2img,
    inputs=[
        gr.Textbox(label="프롬프트 입력"),
        gr.Textbox(label="부정적 프롬프트 입력", value=""),
        gr.Dropdown(choices=stable_diffusion_models_list, label="StableDiffusion 모델 선택", value=None),
        gr.Dropdown(choices=vae_models_list, label="VAE 모델 선택 (선택사항)", value=None),
        gr.Dropdown(choices=lora_models_list, label="LORA 모델 선택 (선택사항)", value=None, multiselect=True),
        gr.Dropdown(choices=textual_inversion_models_list, label="Embedding 모델 선택 (선택사항)", value=None, multiselect=True),
        gr.HTML("<h3>StableDiffusion 설정</h3>"),
        gr.Radio(choices=["SD", "SD2", "SDXL"], label="모델 유형 선택", value="SD"),
        gr.Dropdown(choices=["euler_ancestral", "euler", "lms", "heun", "dpm", "dpm_solver", "dpm_solver++"],
                    label="샘플러 선택", value="euler_ancestral"),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="단계"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label="CFG"),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="너비"),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="높이"),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Clip skip"),
        gr.Checkbox(label="FreeU 활성화", value=False),
        gr.Checkbox(label="Tiled VAE 활성화", value=False),
        gr.Checkbox(label="업스케일 활성화", value=False),
        gr.Radio(choices=["x2", "x4"], label="업스케일 크기", value="x2"),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="업스케일 단계"),
        gr.Slider(minimum=1.0, maximum=30.0, value=6, step=0.1, label="업스케일 CFG"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion (txt2img)",
    description="이 사용자 인터페이스를 사용하면 텍스트를 입력하고 StableDiffusion을 사용하여 이미지를 생성할 수 있습니다. 모델을 선택하고 슬라이더를 사용하여 생성 설정을 사용자 지정할 수 있습니다. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

img2img_interface = gr.Interface(
    fn=generate_image_img2img,
    inputs=[
        gr.Textbox(label="프롬프트 입력"),
        gr.Textbox(label="부정적 프롬프트 입력", value=""),
        gr.Image(label="초기 이미지", type="filepath"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="강도"),
        gr.Dropdown(choices=stable_diffusion_models_list, label="StableDiffusion 모델 선택", value=None),
        gr.Dropdown(choices=vae_models_list, label="VAE 모델 선택 (선택사항)", value=None),
        gr.HTML("<h3>StableDiffusion 설정</h3>"),
        gr.Radio(choices=["SD", "SD2", "SDXL"], label="모델 유형 선택", value="SD"),
        gr.Dropdown(choices=["euler_ancestral", "euler", "lms", "heun", "dpm", "dpm_solver", "dpm_solver++"],
                    label="샘플러 선택", value="euler_ancestral"),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="단계"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label="CFG"),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Clip skip"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion (img2img)",
    description="이 사용자 인터페이스를 사용하면 텍스트와 이미지를 입력하여 StableDiffusion으로 새 이미지를 생성할 수 있습니다. 모델을 선택하고 슬라이더를 사용하여 생성 설정을 사용자 지정할 수 있습니다. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

depth2img_interface = gr.Interface(
    fn=generate_image_depth2img,
    inputs=[
        gr.Textbox(label="프롬프트 입력"),
        gr.Textbox(label="부정적 프롬프트 입력", value=""),
        gr.Image(label="초기 이미지", type="filepath"),
        gr.HTML("<h3>StableDiffusion 설정</h3>"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.01, label="강도"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion (depth2img)",
    description="이 사용자 인터페이스를 사용하면 프롬프트와 초기 이미지를 입력하여 StableDiffusion으로 깊이 인식 이미지를 생성할 수 있습니다. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

pix2pix_interface = gr.Interface(
    fn=generate_image_pix2pix,
    inputs=[
        gr.Textbox(label="프롬프트 입력"),
        gr.Textbox(label="부정적 프롬프트 입력", value=""),
        gr.Image(label="초기 이미지", type="filepath"),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="단계"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label="CFG"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion (pix2pix)",
    description="이 사용자 인터페이스를 사용하면 프롬프트와 초기 이미지를 입력하여 Pix2Pix로 새 이미지를 생성할 수 있습니다. 슬라이더를 사용하여 생성 설정을 사용자 지정할 수 있습니다. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

controlnet_interface = gr.Interface(
    fn=generate_image_controlnet,
    inputs=[
        gr.Textbox(label="프롬프트 입력"),
        gr.Textbox(label="부정적 프롬프트 입력", value=""),
        gr.Image(label="초기 이미지", type="filepath"),
        gr.Radio(choices=["SD", "SDXL"], label="모델 유형 선택", value="SD"),
        gr.Dropdown(choices=stable_diffusion_models_list, label="StableDiffusion 모델 선택", value=None),
        gr.Dropdown(choices=controlnet_models_list, label="ControlNet 모델 선택", value=None),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="단계"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label="CFG"),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="너비"),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="높이"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.1, label="ControlNet 조절 스케일"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지"),
        gr.Image(type="filepath", label="ControlNet 제어 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion (controlnet)",
    description="이 사용자 인터페이스를 사용하면 ControlNet 모델로 이미지를 생성할 수 있습니다. 초기 이미지를 업로드하고, 프롬프트를 입력하고, Stable Diffusion 모델을 선택하고 생성 설정을 사용자 지정하세요. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

latent_upscale_interface = gr.Interface(
    fn=generate_image_upscale_latent,
    inputs=[
        gr.Image(label="업스케일할 이미지", type="filepath"),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="단계"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label="CFG"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="업스케일된 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion (upscale-latent)",
    description="이 사용자 인터페이스를 사용하면 이미지를 업로드하고 잠재 공간에서 업스케일할 수 있습니다",
    allow_flagging="never",
)

realesrgan_upscale_interface = gr.Interface(
    fn=generate_image_upscale_realesrgan,
    inputs=[
        gr.Image(label="업스케일할 이미지", type="filepath"),
        gr.Slider(minimum=0.1, maximum=8, value=4, step=0.1, label="업스케일 비율"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="업스케일된 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion (upscale-realesrgan)",
    description="이 사용자 인터페이스를 사용하면 이미지를 업로드하고 Real-ESRGAN을 사용하여 업스케일할 수 있습니다",
    allow_flagging="never",
)

inpaint_interface = gr.Interface(
    fn=generate_image_inpaint,
    inputs=[
        gr.Textbox(label="프롬프트 입력"),
        gr.Textbox(label="부정적 프롬프트 입력", value=""),
        gr.Image(label="초기 이미지", type="filepath"),
        gr.ImageEditor(label="마스크 이미지", type="filepath"),
        gr.Slider(minimum=0, maximum=100, value=0, step=1, label="마스크 블러 요소"),
        gr.Dropdown(choices=inpaint_models_list, label="인페인트 모델 선택", value=None),
        gr.Dropdown(choices=vae_models_list, label="VAE 모델 선택 (선택사항)", value=None),
        gr.HTML("<h3>StableDiffusion 설정</h3>"),
        gr.Radio(choices=["SD", "SD2", "SDXL"], label="모델 유형 선택", value="SD"),
        gr.Dropdown(choices=["euler_ancestral", "euler", "lms", "heun", "dpm", "dpm_solver", "dpm_solver++"],
                    label="샘플러 선택", value="euler_ancestral"),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="단계"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label="CFG"),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="너비"),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="높이"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="인페인트된 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion (inpaint)",
    description="이 사용자 인터페이스를 사용하면 프롬프트, 초기 이미지 및 마스크 이미지를 입력하여 StableDiffusion으로 이미지를 인페인트할 수 있습니다. 모델을 선택하고 슬라이더를 사용하여 생성 설정을 사용자 지정할 수 있습니다. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

gligen_interface = gr.Interface(
    fn=generate_image_gligen,
    inputs=[
        gr.Textbox(label="프롬프트 입력"),
        gr.Textbox(label="부정적 프롬프트 입력", value=""),
        gr.Textbox(label="GLIGEN 구문 입력", value=""),
        gr.Textbox(label="GLIGEN 박스 입력", value=""),
        gr.Dropdown(choices=stable_diffusion_models_list, label="StableDiffusion 모델 선택", value=None),
        gr.HTML("<h3>StableDiffusion 설정</h3>"),
        gr.Radio(choices=["SD", "SD2", "SDXL"], label="모델 유형 선택", value="SD"),
        gr.Dropdown(choices=["euler_ancestral", "euler", "lms", "heun", "dpm", "dpm_solver", "dpm_solver++"],
                    label="샘플러 선택", value="euler_ancestral"),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="단계"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label="CFG"),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="너비"),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="높이"),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Clip skip"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion (gligen)",
    description="이 사용자 인터페이스를 사용하면 Stable Diffusion으로 이미지를 생성하고 GLIGEN을 사용하여 객체를 삽입할 수 있습니다. Stable Diffusion 모델을 선택하고, 생성 설정을 사용자 지정하고, 프롬프트, GLIGEN 구문 및 경계 상자를 입력하세요. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

animatediff_interface = gr.Interface(
    fn=generate_image_animatediff,
    inputs=[
        gr.Textbox(label="프롬프트 입력"),
        gr.Textbox(label="부정적 프롬프트 입력", value=""),
        gr.Image(label="초기 GIF", type="filepath"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="강도"),
        gr.Radio(choices=["sd", "sdxl"], label="모델 유형 선택", value="sd"),
        gr.Dropdown(choices=stable_diffusion_models_list, label="StableDiffusion 모델 선택 (SD1.5만 해당)", value=None),
        gr.Dropdown(choices=[None, "zoom-in", "zoom-out", "tilt-up", "tilt-down", "pan-right", "pan-left"], label="모션 LORA 선택", value=None, multiselect=True),
        gr.Slider(minimum=1, maximum=200, value=20, step=1, label="프레임"),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="단계"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label="가이던스 스케일"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="너비"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="높이"),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(label="생성된 GIF", type="filepath"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion (animatediff)",
    description="이 사용자 인터페이스를 사용하면 프롬프트를 입력하고 AnimateDiff를 사용하여 애니메이션 GIF를 생성할 수 있습니다. 모델을 선택하고 슬라이더를 사용하여 생성 설정을 사용자 지정할 수 있습니다. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

video_interface = gr.Interface(
    fn=generate_video,
    inputs=[
        gr.Image(label="초기 이미지", type="filepath"),
        gr.Radio(choices=["mp4", "gif"], label="출력 형식 선택", value="mp4", interactive=True),
        gr.HTML("<h3>SVD 설정 (mp4)</h3>"),
        gr.Slider(minimum=0, maximum=360, value=180, step=1, label="모션 버킷 ID"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.01, label="노이즈 증강 강도"),
        gr.Slider(minimum=1, maximum=60, value=10, step=1, label="FPS"),
        gr.Slider(minimum=2, maximum=120, value=25, step=1, label="프레임"),
        gr.Slider(minimum=1, maximum=32, value=8, step=1, label="디코드 청크 크기"),
        gr.HTML("<h3>I2VGen-xl 설정 (gif)</h3>"),
        gr.Textbox(label="프롬프트", value=""),
        gr.Textbox(label="부정적 프롬프트", value=""),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="단계"),
        gr.Slider(minimum=1.0, maximum=30.0, value=9.0, step=0.1, label="CFG"),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Video(label="생성된 비디오"),
        gr.Image(label="생성된 GIF", type="filepath"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion (video)",
    description="이 사용자 인터페이스를 사용하면 초기 이미지를 입력하고 StableVideoDiffusion(mp4) 및 I2VGen-xl(gif)을 사용하여 비디오를 생성할 수 있습니다. 슬라이더를 사용하여 생성 설정을 사용자 지정할 수 있습니다. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

ldm3d_interface = gr.Interface(
    fn=generate_image_ldm3d,
    inputs=[
        gr.Textbox(label="프롬프트 입력"),
        gr.Textbox(label="부정적 프롬프트 입력", value=""),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="너비"),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="높이"),
        gr.Slider(minimum=1, maximum=100, value=40, step=1, label="단계"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label="가이던스 스케일"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 RGB 이미지"),
        gr.Image(type="filepath", label="생성된 깊이 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion (LDM3D)",
    description="이 사용자 인터페이스를 사용하면 프롬프트를 입력하고 LDM3D를 사용하여 RGB 및 깊이 이미지를 생성할 수 있습니다. 슬라이더를 사용하여 생성 설정을 사용자 지정할 수 있습니다. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

sd3_txt2img_interface = gr.Interface(
    fn=generate_image_sd3_txt2img,
    inputs=[
        gr.Textbox(label="프롬프트를 입력하세요"),
        gr.Textbox(label="네거티브 프롬프트를 입력하세요", value=""),
        gr.Slider(minimum=1, maximum=100, value=40, step=1, label="단계"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8.0, step=0.1, label="CFG"),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label="너비"),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label="높이"),
        gr.Slider(minimum=64, maximum=2048, value=256, label="최대 길이"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion 3 (txt2img)",
    description="이 사용자 인터페이스를 사용하면 텍스트를 입력하고 Stable Diffusion 3을 사용하여 이미지를 생성할 수 있습니다. "
                "슬라이더를 사용하여 생성 설정을 사용자 정의할 수 있습니다. "
                "시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

sd3_img2img_interface = gr.Interface(
    fn=generate_image_sd3_img2img,
    inputs=[
        gr.Textbox(label="프롬프트를 입력하세요"),
        gr.Textbox(label="네거티브 프롬프트를 입력하세요", value=""),
        gr.Image(label="초기 이미지", type="filepath"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.8, step=0.01, label="강도"),
        gr.Slider(minimum=1, maximum=100, value=40, step=1, label="단계"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8.0, step=0.1, label="CFG"),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label="너비"),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label="높이"),
        gr.Slider(minimum=64, maximum=2048, value=256, label="최대 길이"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion 3 (img2img)",
    description="이 사용자 인터페이스를 사용하면 텍스트와 초기 이미지를 입력하고 Stable Diffusion 3을 사용하여 새로운 이미지를 생성할 수 있습니다. "
                "슬라이더를 사용하여 생성 설정을 사용자 정의할 수 있습니다. "
                "시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

sd3_controlnet_interface = gr.Interface(
    fn=generate_image_sd3_controlnet,
    inputs=[
        gr.Textbox(label="프롬프트를 입력하세요"),
        gr.Textbox(label="네거티브 프롬프트를 입력하세요", value=""),
        gr.Image(label="초기 이미지", type="filepath"),
        gr.Dropdown(choices=["Pose", "Canny"], label="ControlNet 모델 선택", value="Pose"),
        gr.Slider(minimum=1, maximum=100, value=40, step=1, label="단계"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8.0, step=0.1, label="CFG"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.1, label="ControlNet 컨디셔닝 스케일"),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label="너비"),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label="높이"),
        gr.Slider(minimum=64, maximum=2048, value=256, label="최대 길이"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지"),
        gr.Image(type="filepath", label="ControlNet 제어 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion 3 (ControlNet)",
    description="이 사용자 인터페이스를 사용하면 Stable Diffusion 3과 함께 ControlNet 모델을 사용할 수 있습니다. "
                "슬라이더를 사용하여 생성 설정을 사용자 정의할 수 있습니다. "
                "시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

sd3_inpaint_interface = gr.Interface(
    fn=generate_image_sd3_inpaint,
    inputs=[
        gr.Textbox(label="프롬프트를 입력하세요"),
        gr.Textbox(label="네거티브 프롬프트를 입력하세요", value=""),
        gr.Image(label="초기 이미지", type="filepath"),
        gr.ImageEditor(label="마스크 이미지", type="filepath"),
        gr.Slider(minimum=1, maximum=100, value=40, step=1, label="단계"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8.0, step=0.1, label="CFG"),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label="너비"),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label="높이"),
        gr.Slider(minimum=64, maximum=2048, value=256, label="최대 길이"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion 3 (Inpaint)",
    description="이 사용자 인터페이스를 사용하면 Stable Diffusion 3을 사용하여 인페인팅을 수행할 수 있습니다. "
                "슬라이더를 사용하여 생성 설정을 사용자 정의할 수 있습니다. "
                "시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

cascade_interface = gr.Interface(
    fn=generate_image_cascade,
    inputs=[
        gr.Textbox(label="프롬프트 입력"),
        gr.Textbox(label="부정적 프롬프트 입력", value=""),
        gr.HTML("<h3>Stable Cascade 설정</h3>"),
        gr.Slider(minimum=256, maximum=4096, value=1024, step=64, label="너비"),
        gr.Slider(minimum=256, maximum=4096, value=1024, step=64, label="높이"),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="사전 단계"),
        gr.Slider(minimum=1.0, maximum=30.0, value=4.0, step=0.1, label="사전 가이던스 스케일"),
        gr.Slider(minimum=1, maximum=100, value=20, step=1, label="디코더 단계"),
        gr.Slider(minimum=0.0, maximum=30.0, value=8.0, step=0.1, label="디코더 가이던스 스케일"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion (cascade)",
    description="이 사용자 인터페이스를 사용하면 프롬프트를 입력하고 Stable Cascade를 사용하여 이미지를 생성할 수 있습니다. 슬라이더를 사용하여 생성 설정을 사용자 지정할 수 있습니다. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

instantid_interface = gr.Interface(
    fn=generate_image_instantid,
    inputs=[
        gr.Textbox(label="프롬프트를 입력하세요"),
        gr.Textbox(label="네거티브 프롬프트를 입력하세요", value=""),
        gr.Image(label="얼굴 이미지", type="filepath"),
        gr.Dropdown(choices=stable_diffusion_models_list, label="StableDiffusion XL 모델 선택", value=None),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.8, step=0.01, label="ControlNet 조절 스케일"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.8, step=0.01, label="IP-Adapter 스케일"),
        gr.Slider(minimum=1, maximum=150, value=30, step=1, label="스텝 수"),
        gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.1, label="가이던스 스케일 (CFG)"),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="너비"),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label="높이"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - InstantID",
    description="이 사용자 인터페이스를 사용하면 InstantID를 이용하여 이미지를 생성할 수 있습니다. "
                "얼굴 이미지를 업로드하고, 프롬프트를 입력하고, Stable Diffusion 모델을 선택한 다음 생성 설정을 조정하세요. "
                "시도해보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

photomaker_interface = gr.Interface(
    fn=generate_image_photomaker,
    inputs=[
        gr.Textbox(label="프롬프트를 입력하세요"),
        gr.Textbox(label="네거티브 프롬프트를 입력하세요", value=""),
        gr.File(label="입력 이미지 업로드", file_count="multiple", type="filepath"),
        gr.Textbox(label="트리거 단어 입력", value="img"),
        gr.Dropdown(choices=stable_diffusion_models_list, label="StableDiffusion XL 모델 선택", value=None),
        gr.Slider(minimum=1, maximum=150, value=30, step=1, label="스텝 수"),
        gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.1, label="가이던스 스케일 (CFG)"),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="너비"),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label="높이"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - PhotoMaker",
    description="이 사용자 인터페이스를 사용하면 PhotoMaker를 이용하여 이미지를 생성할 수 있습니다. "
                "입력 이미지를 업로드하고, 프롬프트를 입력하고, 기본 모델을 선택한 다음 생성 설정을 조정하세요. "
                "시도해보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

ip_adapter_faceid_interface = gr.Interface(
    fn=generate_image_ip_adapter_faceid,
    inputs=[
        gr.Textbox(label="프롬프트를 입력하세요"),
        gr.Textbox(label="네거티브 프롬프트를 입력하세요", value=""),
        gr.Image(label="얼굴 이미지 업로드", type="filepath"),
        gr.Dropdown(choices=stable_diffusion_models_list, label="StableDiffusion 모델 선택", value=None),
        gr.Dropdown(
            choices=[
                "ip-adapter-faceid-plusv2_sdxl",
                "ip-adapter-faceid-plusv2_sd15",
                "ip-adapter-faceid-portrait-v11_sd15",
                "ip-adapter-faceid-portrait_sdxl"
            ],
            label="IP-Adapter-FaceID 버전 선택",
            value="ip-adapter-faceid-plusv2_sd15"
        ),
        gr.Slider(minimum=1, maximum=150, value=30, step=1, label="스텝 수"),
        gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.1, label="가이던스 스케일 (CFG)"),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="너비"),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label="높이"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - IP-Adapter-FaceID",
    description="이 사용자 인터페이스를 사용하면 IP-Adapter-FaceID를 이용하여 이미지를 생성할 수 있습니다. "
                "얼굴 이미지를 업로드하고, 프롬프트를 입력하고, 기본 모델과 IP-Adapter 버전을 선택한 다음 생성 설정을 조정하세요. "
                "시도해보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

adapters_interface = gr.TabbedInterface(
    [instantid_interface, photomaker_interface, ip_adapter_faceid_interface],
    tab_names=["InstantID", "PhotoMaker", "IP-Adapter-FaceID"]
)

extras_interface = gr.Interface(
    fn=generate_image_extras,
    inputs=[
        gr.Image(label="수정할 이미지", type="filepath"),
        gr.Image(label="소스 이미지", type="filepath"),
        gr.Checkbox(label="배경 제거", value=False),
        gr.Checkbox(label="얼굴 교체 활성화", value=False),
        gr.Checkbox(label="얼굴 복원 활성화", value=False),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(label="수정된 이미지", type="filepath"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion (extras)",
    description="이 사용자 인터페이스를 사용하면 이미지를 수정할 수 있습니다",
    allow_flagging="never",
)

kandinsky_txt2img_interface = gr.Interface(
    fn=generate_image_kandinsky_txt2img,
    inputs=[
        gr.Textbox(label="프롬프트를 입력하세요"),
        gr.Textbox(label="네거티브 프롬프트를 입력하세요", value=""),
        gr.Radio(choices=["2.1", "2.2", "3"], label="Kandinsky 버전", value="2.2"),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="단계"),
        gr.Slider(minimum=0.1, maximum=20, value=4, step=0.1, label="CFG"),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label="높이"),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label="너비"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - Kandinsky (txt2img)",
    description="이 사용자 인터페이스를 사용하면 Kandinsky 모델을 사용하여 이미지를 생성할 수 있습니다. "
                "버전 2.1, 2.2 및 3 중에서 선택할 수 있으며 생성 설정을 사용자 정의할 수 있습니다. "
                "시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

kandinsky_img2img_interface = gr.Interface(
    fn=generate_image_kandinsky_img2img,
    inputs=[
        gr.Textbox(label="프롬프트를 입력하세요"),
        gr.Textbox(label="네거티브 프롬프트를 입력하세요", value=""),
        gr.Image(label="초기 이미지", type="filepath"),
        gr.Radio(choices=["2.1", "2.2", "3"], label="Kandinsky 버전", value="2.2"),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="단계"),
        gr.Slider(minimum=0.1, maximum=20, value=4, step=0.1, label="CFG"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.8, step=0.01, label="강도"),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label="높이"),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label="너비"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - Kandinsky (img2img)",
    description="이 사용자 인터페이스를 사용하면 Kandinsky 모델을 사용하여 이미지를 생성할 수 있습니다. "
                "버전 2.1, 2.2 및 3 중에서 선택할 수 있으며 생성 설정을 사용자 정의할 수 있습니다. "
                "시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

kandinsky_inpaint_interface = gr.Interface(
    fn=generate_image_kandinsky_inpaint,
    inputs=[
        gr.Textbox(label="프롬프트를 입력하세요"),
        gr.Textbox(label="부정적 프롬프트를 입력하세요", value=""),
        gr.Image(label="초기 이미지", type="filepath"),
        gr.ImageEditor(label="마스크 이미지", type="filepath"),
        gr.Radio(choices=["2.1", "2.2"], label="Kandinsky 버전", value="2.2"),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="스텝"),
        gr.Slider(minimum=0.1, maximum=20, value=4, step=0.1, label="CFG"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.8, step=0.01, label="강도"),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label="높이"),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label="너비"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (Alpha) - Kandinsky (inpaint)",
    description="이 사용자 인터페이스를 사용하면 Kandinsky 모델을 사용하여 인페인팅을 수행할 수 있습니다. "
                "버전 2.1과 2.2 중에서 선택할 수 있으며 생성 설정을 사용자 정의할 수 있습니다. "
                "시도해 보고 어떤 일이 일어나는지 확인해 보세요!",
    allow_flagging="never",
)

kandinsky_interface = gr.TabbedInterface(
    [kandinsky_txt2img_interface, kandinsky_img2img_interface, kandinsky_inpaint_interface],
    tab_names=["txt2img", "img2img", "inpaint"]
)

flux_interface = gr.Interface(
    fn=generate_image_flux,
    inputs=[
        gr.Textbox(label="프롬프트 입력"),
        gr.Dropdown(choices=["FLUX.1-schnell", "FLUX.1-dev"], label="Flux 모델 선택", value="FLUX.1-schnell"),
        gr.Slider(minimum=0.0, maximum=10.0, value=0.0, step=0.1, label="가이던스 스케일"),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label="높이"),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label="너비"),
        gr.Slider(minimum=1, maximum=100, value=10, step=1, label="단계"),
        gr.Slider(minimum=1, maximum=1024, value=256, step=1, label="최대 시퀀스 길이"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - Flux",
    description="이 사용자 인터페이스를 사용하면 Flux 모델을 사용하여 이미지를 생성할 수 있습니다. Schnell과 Dev 모델 중에서 선택하고 생성 설정을 사용자 지정할 수 있습니다. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

hunyuandit_interface = gr.Interface(
    fn=generate_image_hunyuandit,
    inputs=[
        gr.Textbox(label="프롬프트 입력"),
        gr.Textbox(label="부정적 프롬프트 입력", value=""),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="단계"),
        gr.Slider(minimum=0.1, maximum=30.0, value=7.5, step=0.1, label="가이던스 스케일"),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label="높이"),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label="너비"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - HunyuanDiT",
    description="이 사용자 인터페이스를 사용하면 HunyuanDiT 모델을 사용하여 이미지를 생성할 수 있습니다. 프롬프트(영어 또는 중국어)를 입력하고 생성 설정을 사용자 지정하세요. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

lumina_interface = gr.Interface(
    fn=generate_image_lumina,
    inputs=[
        gr.Textbox(label="프롬프트 입력"),
        gr.Textbox(label="부정적 프롬프트 입력", value=""),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="단계"),
        gr.Slider(minimum=0.1, maximum=30.0, value=4, step=0.1, label="가이던스 스케일"),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label="높이"),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label="너비"),
        gr.Slider(minimum=1, maximum=1024, value=256, step=1, label="최대 시퀀스 길이"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - Lumina-T2X",
    description="이 사용자 인터페이스를 사용하면 Lumina-T2X 모델을 사용하여 이미지를 생성할 수 있습니다. 프롬프트를 입력하고 생성 설정을 사용자 지정하세요. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

kolors_interface = gr.Interface(
    fn=generate_image_kolors,
    inputs=[
        gr.Textbox(label="프롬프트 입력"),
        gr.Textbox(label="부정적 프롬프트 입력", value=""),
        gr.Slider(minimum=1.0, maximum=20.0, value=6.5, step=0.1, label="가이던스 스케일"),
        gr.Slider(minimum=1, maximum=100, value=25, step=1, label="단계"),
        gr.Slider(minimum=1, maximum=1024, value=256, step=1, label="최대 시퀀스 길이"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - Kolors",
    description="이 사용자 인터페이스를 사용하면 Kolors 모델을 사용하여 이미지를 생성할 수 있습니다. 프롬프트를 입력하고 생성 설정을 사용자 지정하세요. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

auraflow_interface = gr.Interface(
    fn=generate_image_auraflow,
    inputs=[
        gr.Textbox(label="프롬프트를 입력하세요"),
        gr.Textbox(label="네거티브 프롬프트를 입력하세요", value=""),
        gr.Slider(minimum=1, maximum=100, value=25, step=1, label="스텝 수"),
        gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.1, label="가이던스 스케일"),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="높이"),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="너비"),
        gr.Slider(minimum=1, maximum=1024, value=256, step=1, label="최대 시퀀스 길이"),
        gr.Checkbox(label="AuraSR 활성화", value=False),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - AuraFlow",
    description="이 사용자 인터페이스를 사용하면 AuraFlow 모델을 이용하여 이미지를 생성할 수 있습니다. "
                "프롬프트를 입력하고 생성 설정을 조정하세요. "
                "또한 생성된 이미지를 4배 확대하는 AuraSR을 활성화할 수도 있습니다. "
                "시도해보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

wurstchen_interface = gr.Interface(
    fn=generate_image_wurstchen,
    inputs=[
        gr.Textbox(label="프롬프트 입력"),
        gr.Textbox(label="부정적 프롬프트 입력", value=""),
        gr.Slider(minimum=256, maximum=2048, value=1536, step=64, label="너비"),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label="높이"),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="사전 단계"),
        gr.Slider(minimum=0.1, maximum=30.0, value=4.0, step=0.1, label="사전 가이던스 스케일"),
        gr.Slider(minimum=1, maximum=100, value=20, step=1, label="디코더 단계"),
        gr.Slider(minimum=0.0, maximum=30.0, value=0.0, step=0.1, label="디코더 가이던스 스케일"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - Würstchen",
    description="이 사용자 인터페이스를 사용하면 Würstchen 모델을 사용하여 이미지를 생성할 수 있습니다. 프롬프트를 입력하고 생성 설정을 사용자 지정하세요. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

deepfloyd_if_txt2img_interface = gr.Interface(
    fn=generate_image_deepfloyd_txt2img,
    inputs=[
        gr.Textbox(label="프롬프트를 입력하세요"),
        gr.Textbox(label="네거티브 프롬프트를 입력하세요", value=""),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="단계"),
        gr.Slider(minimum=0.1, maximum=30.0, value=6, step=0.1, label="가이던스 스케일"),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="너비"),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="높이"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지 (단계 I)"),
        gr.Image(type="filepath", label="생성된 이미지 (단계 II)"),
        gr.Image(type="filepath", label="생성된 이미지 (단계 III)"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - DeepFloyd IF (txt2img)",
    description="이 사용자 인터페이스를 사용하면 DeepFloyd IF 모델을 사용하여 이미지를 생성할 수 있습니다. "
                "프롬프트를 입력하고 생성 설정을 사용자 정의하세요. "
                "이 프로세스는 세 단계의 생성을 포함하며, 각 단계는 점점 더 높은 품질의 이미지를 생성합니다. "
                "시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

deepfloyd_if_img2img_interface = gr.Interface(
    fn=generate_image_deepfloyd_img2img,
    inputs=[
        gr.Textbox(label="프롬프트를 입력하세요"),
        gr.Textbox(label="네거티브 프롬프트를 입력하세요", value=""),
        gr.Image(label="초기 이미지", type="filepath"),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="단계"),
        gr.Slider(minimum=0.1, maximum=30.0, value=6, step=0.1, label="가이던스 스케일"),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="너비"),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="높이"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지 (단계 I)"),
        gr.Image(type="filepath", label="생성된 이미지 (단계 II)"),
        gr.Image(type="filepath", label="생성된 이미지 (단계 III)"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - DeepFloyd IF (img2img)",
    description="이 인터페이스를 사용하면 DeepFloyd IF의 이미지 대 이미지 파이프라인을 사용하여 이미지를 생성할 수 있습니다. "
                "프롬프트를 입력하고, 초기 이미지를 업로드하고, 생성 설정을 사용자 정의하세요. "
                "이 프로세스는 세 단계의 생성을 포함하며, 각 단계는 점점 더 높은 품질의 이미지를 생성합니다. "
                "시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

deepfloyd_if_inpaint_interface = gr.Interface(
    fn=generate_image_deepfloyd_inpaint,
    inputs=[
        gr.Textbox(label="프롬프트를 입력하세요"),
        gr.Textbox(label="네거티브 프롬프트를 입력하세요", value=""),
        gr.Image(label="초기 이미지", type="filepath"),
        gr.ImageEditor(label="마스크 이미지", type="filepath"),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="단계"),
        gr.Slider(minimum=0.1, maximum=30.0, value=6, step=0.1, label="가이던스 스케일"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지 (단계 I)"),
        gr.Image(type="filepath", label="생성된 이미지 (단계 II)"),
        gr.Image(type="filepath", label="생성된 이미지 (단계 III)"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - DeepFloyd IF (inpaint)",
    description="이 인터페이스를 사용하면 DeepFloyd IF를 사용하여 인페인팅을 수행할 수 있습니다. "
                "프롬프트를 입력하고, 초기 이미지와 마스크 이미지를 업로드하고, 생성 설정을 사용자 정의하세요. "
                "이 프로세스는 세 단계의 생성을 포함하며, 각 단계는 점점 더 높은 품질의 이미지를 생성합니다. "
                "시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

deepfloyd_if_interface = gr.TabbedInterface(
    [deepfloyd_if_txt2img_interface, deepfloyd_if_img2img_interface, deepfloyd_if_inpaint_interface],
    tab_names=["txt2img", "img2img", "inpaint"]
)

pixart_interface = gr.Interface(
    fn=generate_image_pixart,
    inputs=[
        gr.Textbox(label="프롬프트 입력"),
        gr.Textbox(label="부정적 프롬프트 입력", value=""),
        gr.Radio(choices=["Alpha-512", "Alpha-1024", "Sigma-512", "Sigma-1024"], label="PixArt 버전", value="Alpha-512"),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="단계"),
        gr.Slider(minimum=0.1, maximum=30.0, value=7.5, step=0.1, label="가이던스 스케일"),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="높이"),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="너비"),
        gr.Slider(minimum=1, maximum=1024, value=256, step=1, label="최대 시퀀스 길이"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - PixArt",
    description="이 사용자 인터페이스를 사용하면 PixArt 모델을 사용하여 이미지를 생성할 수 있습니다. Alpha와 Sigma 버전 중에서 선택하고 생성 설정을 사용자 지정할 수 있습니다. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

modelscope_interface = gr.Interface(
    fn=generate_video_modelscope,
    inputs=[
        gr.Textbox(label="프롬프트 입력"),
        gr.Textbox(label="부정적 프롬프트 입력", value=""),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="단계"),
        gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.1, label="가이던스 스케일"),
        gr.Slider(minimum=256, maximum=1024, value=320, step=64, label="높이"),
        gr.Slider(minimum=256, maximum=1024, value=576, step=64, label="너비"),
        gr.Slider(minimum=16, maximum=128, value=64, step=1, label="프레임 수"),
        gr.Radio(choices=["mp4", "gif"], label="출력 형식 선택", value="mp4", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Video(label="생성된 비디오"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - ModelScope",
    description="이 사용자 인터페이스를 사용하면 ModelScope를 사용하여 비디오를 생성할 수 있습니다. 프롬프트를 입력하고 생성 설정을 사용자 지정하세요. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

zeroscope2_interface = gr.Interface(
    fn=generate_video_zeroscope2,
    inputs=[
        gr.Textbox(label="프롬프트 입력"),
        gr.Video(label="향상시킬 비디오 (선택사항)", interactive=True),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.1, label="강도"),
        gr.Slider(minimum=1, maximum=100, value=40, step=1, label="단계"),
        gr.Slider(minimum=256, maximum=1280, value=576, step=64, label="너비"),
        gr.Slider(minimum=256, maximum=1280, value=320, step=64, label="높이"),
        gr.Slider(minimum=1, maximum=100, value=36, step=1, label="프레임"),
        gr.Checkbox(label="비디오 향상 활성화", value=False),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Video(label="생성된 비디오"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - ZeroScope 2",
    description="이 사용자 인터페이스를 사용하면 ZeroScope 2 모델을 사용하여 비디오를 생성하고 향상시킬 수 있습니다. 텍스트 프롬프트를 입력하고, 선택적으로 향상시킬 비디오를 업로드하고, 생성 설정을 사용자 지정할 수 있습니다. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

cogvideox_interface = gr.Interface(
    fn=generate_video_cogvideox,
    inputs=[
        gr.Textbox(label="프롬프트 입력"),
        gr.Textbox(label="부정적 프롬프트 입력", value=""),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="단계"),
        gr.Slider(minimum=1.0, maximum=20.0, value=6.0, step=0.1, label="가이던스 스케일"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="높이"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="너비"),
        gr.Slider(minimum=1, maximum=100, value=16, step=1, label="프레임 수"),
        gr.Slider(minimum=1, maximum=60, value=8, step=1, label="FPS"),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Video(label="생성된 비디오"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - CogVideoX",
    description="이 사용자 인터페이스를 사용하면 CogVideoX를 사용하여 비디오를 생성할 수 있습니다. 프롬프트를 입력하고 생성 설정을 사용자 지정하세요. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

latte_interface = gr.Interface(
    fn=generate_video_latte,
    inputs=[
        gr.Textbox(label="프롬프트 입력"),
        gr.Textbox(label="부정적 프롬프트 입력", value=""),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="단계"),
        gr.Slider(minimum=1.0, maximum=20.0, value=6.0, step=0.1, label="가이던스 스케일"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="높이"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="너비"),
        gr.Slider(minimum=1, maximum=100, value=16, step=1, label="비디오 길이"),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 GIF"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - Latte",
    description="이 사용자 인터페이스를 사용하면 Latte를 사용하여 GIF를 생성할 수 있습니다. 프롬프트를 입력하고 생성 설정을 사용자 지정하세요. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

triposr_interface = gr.Interface(
    fn=generate_3d_triposr,
    inputs=[
        gr.Image(label="입력 이미지", type="pil"),
        gr.Slider(minimum=32, maximum=320, value=256, step=32, label="마칭 큐브 해상도"),
        gr.Slider(minimum=0.5, maximum=1.0, value=0.85, step=0.05, label="전경 비율"),
        gr.Radio(choices=["obj", "glb"], label="출력 형식 선택", value="obj", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Model3D(label="생성된 3D 객체"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - TripoSR",
    description="이 사용자 인터페이스를 사용하면 TripoSR을 사용하여 3D 객체를 생성할 수 있습니다. 이미지를 업로드하고 생성 설정을 사용자 지정하세요. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

stablefast3d_interface = gr.Interface(
    fn=generate_3d_stablefast3d,
    inputs=[
        gr.Image(label="입력 이미지", type="filepath"),
        gr.Slider(minimum=256, maximum=4096, value=1024, step=256, label="텍스처 해상도"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.85, step=0.05, label="전경 비율"),
        gr.Radio(choices=["none", "triangle", "quad"], label="리메시 옵션", value="none"),
        gr.Radio(choices=["obj", "glb"], label="출력 형식 선택", value="obj", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Model3D(label="생성된 3D 객체"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableFast3D",
    description="이 사용자 인터페이스를 사용하면 StableFast3D를 사용하여 이미지에서 3D 객체를 생성할 수 있습니다. 이미지를 업로드하고 생성 설정을 사용자 지정하세요. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

shap_e_interface = gr.Interface(
    fn=generate_3d_shap_e,
    inputs=[
        gr.Textbox(label="프롬프트 입력"),
        gr.Image(label="초기 이미지 (선택사항)", type="filepath", interactive=True),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="단계"),
        gr.Slider(minimum=1.0, maximum=30.0, value=10.0, step=0.1, label="CFG"),
        gr.Slider(minimum=64, maximum=512, value=256, step=64, label="프레임 크기"),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Model3D(label="생성된 3D 객체"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - Shap-E",
    description="이 사용자 인터페이스를 사용하면 Shap-E를 사용하여 3D 객체를 생성할 수 있습니다. 텍스트 프롬프트를 입력하거나 초기 이미지를 업로드하고 생성 설정을 사용자 지정할 수 있습니다. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

sv34d_interface = gr.Interface(
    fn=generate_sv34d,
    inputs=[
        gr.File(label="입력 파일 (3D-U와 3D-P는 이미지, 4D는 MP4 비디오)", type="filepath"),
        gr.Radio(choices=["3D-U", "3D-P", "4D"], label="버전", value="3D-U"),
        gr.Slider(minimum=0.0, maximum=90.0, value=10.0, step=0.1, label="고도 각도 (3D-P 전용)"),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Video(label="생성된 출력"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - SV34D",
    description="이 인터페이스를 사용하면 SV34D 모델을 사용하여 3D 및 4D 콘텐츠를 생성할 수 있습니다. 3D-U와 3D-P 버전의 경우 이미지(PNG, JPG, JPEG)를 업로드하고, 4D 버전의 경우 MP4 비디오를 업로드하세요. 버전을 선택하고 필요에 따라 설정을 사용자 지정하세요.",
    allow_flagging="never",
)

zero123plus_interface = gr.Interface(
    fn=generate_3d_zero123plus,
    inputs=[
        gr.Image(label="입력 이미지", type="filepath"),
        gr.Slider(minimum=1, maximum=100, value=75, step=1, label="추론 단계"),
        gr.Radio(choices=["png", "jpeg"], label="출력 형식 선택", value="png", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="생성된 이미지"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - Zero123Plus",
    description="이 사용자 인터페이스를 사용하면 Zero123Plus를 사용하여 3D와 유사한 이미지를 생성할 수 있습니다. 입력 이미지를 업로드하고 추론 단계 수를 사용자 지정하세요. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

stableaudio_interface = gr.Interface(
    fn=generate_stableaudio,
    inputs=[
        gr.Textbox(label="프롬프트 입력"),
        gr.Textbox(label="부정적 프롬프트 입력"),
        gr.Slider(minimum=1, maximum=1000, value=200, step=1, label="단계"),
        gr.Slider(minimum=0.1, maximum=12, value=4, step=0.1, label="CFG"),
        gr.Slider(minimum=1, maximum=60, value=10, step=1, label="오디오 길이 (초)"),
        gr.Slider(minimum=1, maximum=60, value=0, step=1, label="오디오 시작 (초)"),
        gr.Slider(minimum=1, maximum=10, value=3, step=1, label="파형 수"),
        gr.Radio(choices=["wav", "mp3", "ogg"], label="출력 형식 선택", value="wav", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Audio(label="생성된 오디오", type="filepath"),
        gr.Image(label="멜 스펙트로그램", type="filepath"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableAudio",
    description="이 사용자 인터페이스를 사용하면 텍스트를 입력하고 StableAudio를 사용하여 오디오를 생성할 수 있습니다. 슬라이더를 사용하여 생성 설정을 사용자 지정할 수 있습니다. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

audiocraft_interface = gr.Interface(
    fn=generate_audio_audiocraft,
    inputs=[
        gr.Textbox(label="프롬프트 입력"),
        gr.Audio(type="filepath", label="멜로디 오디오 (선택사항)", interactive=True),
        gr.Dropdown(choices=audiocraft_models_list, label="AudioCraft 모델 선택", value=None),
        gr.HTML("<h3>AudioCraft 설정</h3>"),
        gr.Radio(choices=["musicgen", "audiogen", "magnet"], label="모델 유형 선택", value="musicgen"),
        gr.Slider(minimum=1, maximum=120, value=10, step=1, label="지속 시간 (초)"),
        gr.Slider(minimum=1, maximum=1000, value=250, step=1, label="Top K"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="Top P"),
        gr.Slider(minimum=0.0, maximum=1.9, value=1.0, step=0.1, label="온도"),
        gr.Slider(minimum=1.0, maximum=10.0, value=3.0, step=0.1, label="CFG"),
        gr.Checkbox(label="다중 밴드 확산 활성화", value=False),
        gr.Radio(choices=["wav", "mp3", "ogg"], label="출력 형식 선택 (다중 밴드 확산 없이만 작동)", value="wav", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Audio(label="생성된 오디오", type="filepath"),
        gr.Image(label="멜 스펙트로그램", type="filepath"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - AudioCraft",
    description="이 사용자 인터페이스를 사용하면 텍스트를 입력하고 AudioCraft를 사용하여 오디오를 생성할 수 있습니다. 모델을 선택하고 슬라이더를 사용하여 생성 설정을 사용자 지정할 수 있습니다. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

audioldm2_interface = gr.Interface(
    fn=generate_audio_audioldm2,
    inputs=[
        gr.Textbox(label="프롬프트 입력"),
        gr.Textbox(label="부정적 프롬프트 입력", value=""),
        gr.Dropdown(choices=["cvssp/audioldm2", "cvssp/audioldm2-music"], label="AudioLDM 2 모델 선택", value="cvssp/audioldm2"),
        gr.Slider(minimum=1, maximum=1000, value=200, step=1, label="단계"),
        gr.Slider(minimum=1, maximum=60, value=10, step=1, label="길이 (초)"),
        gr.Slider(minimum=1, maximum=10, value=3, step=1, label="파형 수"),
        gr.Radio(choices=["wav", "mp3", "ogg"], label="출력 형식 선택", value="wav", interactive=True),
        gr.Button(value="생성 중지", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Audio(label="생성된 오디오", type="filepath"),
        gr.Image(label="멜 스펙트로그램", type="filepath"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - AudioLDM 2",
    description="이 사용자 인터페이스를 사용하면 텍스트를 입력하고 AudioLDM 2를 사용하여 오디오를 생성할 수 있습니다. 모델을 선택하고 슬라이더를 사용하여 생성 설정을 사용자 지정할 수 있습니다. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

demucs_interface = gr.Interface(
    fn=demucs_separate,
    inputs=[
        gr.Audio(type="filepath", label="분리할 오디오 파일"),
        gr.Radio(choices=["wav", "mp3", "ogg"], label="출력 형식 선택", value="wav", interactive=True),
    ],
    outputs=[
        gr.Audio(label="보컬", type="filepath"),
        gr.Audio(label="악기", type="filepath"),
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - Demucs",
    description="이 사용자 인터페이스를 사용하면 오디오 파일을 업로드하고 Demucs를 사용하여 보컬과 악기로 분리할 수 있습니다. 시도해 보고 어떤 일이 일어나는지 확인해보세요!",
    allow_flagging="never",
)

gallery_interface = gr.Interface(
    fn=lambda *args: get_output_files()[-1](*args),
    inputs=[
        gr.Dropdown(label="텍스트 파일", choices=get_output_files()[0], interactive=True),
        gr.Dropdown(label="이미지 파일", choices=get_output_files()[1], interactive=True),
        gr.Dropdown(label="비디오 파일", choices=get_output_files()[2], interactive=True),
        gr.Dropdown(label="오디오 파일", choices=get_output_files()[3], interactive=True),
        gr.Dropdown(label="3D 모델 파일", choices=get_output_files()[4], interactive=True),
    ],
    outputs=[
        gr.Textbox(label="텍스트"),
        gr.Image(label="이미지", type="filepath"),
        gr.Video(label="비디오"),
        gr.Audio(label="오디오", type="filepath"),
        gr.Model3D(label="3D 모델"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - 갤러리",
    description="이 인터페이스를 사용하면 outputs 디렉토리의 파일을 볼 수 있습니다",
    allow_flagging="never",
)

model_downloader_interface = gr.Interface(
    fn=download_model,
    inputs=[
        gr.Dropdown(choices=[None, "StarlingLM(Transformers7B)", "OpenChat(Llama7B.Q4)"], label="LLM 모델 다운로드", value=None),
        gr.Dropdown(choices=[None, "Dreamshaper8(SD1.5)", "RealisticVisionV4.0(SDXL)"], label="StableDiffusion 모델 다운로드", value=None),
    ],
    outputs=[
        gr.Textbox(label="메시지", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - 모델 다운로더",
    description="이 사용자 인터페이스를 사용하면 LLM 및 StableDiffusion 모델을 다운로드할 수 있습니다",
    allow_flagging="never",
)

settings_interface = gr.Interface(
    fn=settings_interface,
    inputs=[
        gr.Radio(choices=["True", "False"], label="공유 모드", value="False")
    ],
    outputs=[
        gr.Textbox(label="메시지", type="text")
    ],
    title="NeuroSandboxWebUI (ALPHA) - 설정",
    description="이 사용자 인터페이스를 사용하면 애플리케이션의 설정을 변경할 수 있습니다",
    allow_flagging="never",
)

system_interface = gr.Interface(
    fn=get_system_info,
    inputs=[],
    outputs=[
        gr.Textbox(label="GPU 총 메모리"),
        gr.Textbox(label="GPU 사용 메모리"),
        gr.Textbox(label="GPU 가용 메모리"),
        gr.Textbox(label="GPU 온도"),
        gr.Textbox(label="CPU 온도"),
        gr.Textbox(label="RAM 총량"),
        gr.Textbox(label="RAM 사용량"),
        gr.Textbox(label="RAM 가용량"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - 시스템",
    description="이 인터페이스는 시스템 정보를 표시합니다",
    allow_flagging="never",
)

with gr.TabbedInterface(
    [
        gr.TabbedInterface(
            [chat_interface, tts_stt_interface, translate_interface],
            tab_names=["LLM", "TTS-STT", "LibreTranslate"]
        ),
        gr.TabbedInterface(
            [
                gr.TabbedInterface(
                    [txt2img_interface, img2img_interface, depth2img_interface, pix2pix_interface, controlnet_interface, latent_upscale_interface, realesrgan_upscale_interface, inpaint_interface, gligen_interface, animatediff_interface, video_interface, ldm3d_interface,
                     gr.TabbedInterface([sd3_txt2img_interface, sd3_img2img_interface, sd3_controlnet_interface, sd3_inpaint_interface],
                                        tab_names=["txt2img", "img2img", "controlnet", "inpaint"]),
                     cascade_interface, adapters_interface, extras_interface],
                    tab_names=["txt2img", "img2img", "depth2img", "pix2pix", "controlnet", "upscale(latent)", "upscale(Real-ESRGAN)", "inpaint", "gligen", "animatediff", "video", "ldm3d", "sd3", "cascade", "adapters", "extras"]
                ),
                kandinsky_interface, flux_interface, hunyuandit_interface, lumina_interface, kolors_interface, auraflow_interface, wurstchen_interface, deepfloyd_if_interface, pixart_interface
            ],
            tab_names=["StableDiffusion", "Kandinsky", "Flux", "HunyuanDiT", "Lumina-T2X", "Kolors", "AuraFlow", "Würstchen", "DeepFloydIF", "PixArt"]
        ),
        gr.TabbedInterface(
            [wav2lip_interface, modelscope_interface, zeroscope2_interface, cogvideox_interface, latte_interface],
            tab_names=["Wav2Lip", "ModelScope", "ZeroScope 2", "CogVideoX", "Latte"]
        ),
        gr.TabbedInterface(
            [triposr_interface, stablefast3d_interface, shap_e_interface, sv34d_interface, zero123plus_interface],
            tab_names=["TripoSR", "StableFast3D", "Shap-E", "SV34D", "Zero123Plus"]
        ),
        gr.TabbedInterface(
            [stableaudio_interface, audiocraft_interface, audioldm2_interface, bark_interface, demucs_interface],
            tab_names=["StableAudioOpen", "AudioCraft", "AudioLDM 2", "SunoBark", "Demucs"]
        ),
        gr.TabbedInterface(
            [gallery_interface, model_downloader_interface, settings_interface, system_interface],
            tab_names=["갤러리", "모델 다운로더", "설정", "시스템"]
        )
    ],
    tab_names=["텍스트", "이미지", "비디오", "3D", "오디오", "인터페이스"]
) as app:
    chat_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    bark_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    txt2img_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    img2img_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    depth2img_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    pix2pix_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    controlnet_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    latent_upscale_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    realesrgan_upscale_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    inpaint_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    gligen_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    animatediff_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    video_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    ldm3d_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    sd3_txt2img_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    sd3_img2img_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    sd3_controlnet_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    sd3_inpaint_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    cascade_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    instantid_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    photomaker_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    ip_adapter_faceid_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    extras_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    kandinsky_txt2img_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    kandinsky_img2img_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    kandinsky_inpaint_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    flux_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    hunyuandit_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    lumina_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    kolors_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    auraflow_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    wurstchen_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    deepfloyd_if_txt2img_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    deepfloyd_if_img2img_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    deepfloyd_if_inpaint_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    pixart_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    modelscope_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    zeroscope2_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    cogvideox_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    latte_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    triposr_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    stablefast3d_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    shap_e_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    sv34d_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    zero123plus_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    stableaudio_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    audiocraft_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    audioldm2_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)

    close_button = gr.Button("Close terminal")
    close_button.click(close_terminal, [], [], queue=False)

    folder_button = gr.Button("Outputs")
    folder_button.click(open_outputs_folder, [], [], queue=False)

    github_link = gr.HTML(
        '<div style="text-align: center; margin-top: 20px;">'
        '<a href="https://github.com/Dartvauder/NeuroSandboxWebUI" target="_blank" style="color: blue; text-decoration: none; font-size: 16px; margin-right: 20px;">'
        'GitHub'
        '</a>'
        '<a href="https://huggingface.co/Dartvauder007" target="_blank" style="color: blue; text-decoration: none; font-size: 16px;">'
        'Hugging Face'
        '</a>'
        '</div>'
    )

    app.launch(share=share_mode, server_name="localhost", auth=authenticate)
