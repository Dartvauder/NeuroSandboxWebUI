import sys
import os
import warnings
import platform
import logging
import importlib
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)
cache_dir = os.path.join("TechnicalFiles/cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = cache_dir
temp_dir = os.path.join("TechnicalFiles/temp")
os.makedirs(temp_dir, exist_ok=True)
os.environ["TMPDIR"] = temp_dir
sys.modules['triton'] = None
from threading import Thread
import gradio as gr
import langdetect
from datasets import load_dataset, Audio
from libretranslatepy import LibreTranslateAPI
import urllib.error
import soundfile as sf
from pydub import AudioSegment
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
import librosa.display
import base64
import gc
import cv2
import av
import subprocess
import json
import torch
from einops import rearrange
import random
import tempfile
import shutil
from datetime import datetime
from diffusers.utils import load_image, load_video, export_to_video, export_to_gif, export_to_ply, pt_to_pil
from compel import Compel, ReturnedEmbeddingsType
from sd_embed.embedding_funcs import get_weighted_text_embeddings_sd3
import trimesh
from git import Repo
import numpy as np
import scipy
import imageio
from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo
import taglib
import requests
import markdown
import urllib.parse
import torchaudio
from audiocraft.data.audio import audio_write
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetTemperature, NVML_TEMPERATURE_GPU
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from DeepCache import DeepCacheSDHelper
from tgate import TgateSDLoader, TgateSDXLLoader
import tomesd
import openparse
import string
import hashlib
from ThirdPartyRepository.taesd import taesd
import pandas as pd

GPUtil = None
if torch.cuda.is_available():
    try:
        import GPUtil
    except ImportError:
        GPUtil = None

WinTmp = None
if sys.platform in ['win32', 'win64']:
    try:
        import WinTmp
    except ImportError:
        WinTmp = None


def lazy_import(module_name, fromlist):
    module = None
    def wrapper():
        nonlocal module
        if module is None:
            if fromlist:
                module = importlib.import_module(module_name, fromlist)
            else:
                module = importlib.import_module(module_name)
        return module
    return wrapper

# Transformers import
AutoModelForCausalLM = lazy_import('transformers', 'AutoModelForCausalLM')
AutoTokenizer = lazy_import('transformers', 'AutoTokenizer')
AutoProcessor = lazy_import('transformers', 'AutoProcessor')
TextIteratorStreamer = lazy_import('transformers', 'TextIteratorStreamer')
LlavaNextVideoProcessor = lazy_import('transformers', 'LlavaNextVideoProcessor')
LlavaNextVideoForConditionalGeneration = lazy_import('transformers', 'LlavaNextVideoForConditionalGeneration')
Qwen2AudioForConditionalGeneration = lazy_import('transformers', 'Qwen2AudioForConditionalGeneration')
BarkModel = lazy_import('transformers', 'BarkModel')
pipeline = lazy_import('transformers', 'pipeline')
T5EncoderModel = lazy_import('transformers', 'T5EncoderModel')
CLIPTextModel = lazy_import('transformers', 'CLIPTextModel')
BitsAndBytesConfig = lazy_import('transformers', 'BitsAndBytesConfig')
DPTForDepthEstimation = lazy_import('transformers', 'DPTForDepthEstimation')
DPTFeatureExtractor = lazy_import('transformers', 'DPTFeatureExtractor')
CLIPVisionModelWithProjection = lazy_import('transformers', 'CLIPVisionModelWithProjection')
SeamlessM4Tv2Model = lazy_import('transformers', 'SeamlessM4Tv2Model')
SeamlessM4Tv2ForSpeechToSpeech = lazy_import('transformers', 'SeamlessM4Tv2ForSpeechToSpeech')
SeamlessM4Tv2ForTextToText = lazy_import('transformers', 'SeamlessM4Tv2ForTextToText')
SeamlessM4Tv2ForSpeechToText = lazy_import('transformers', 'SeamlessM4Tv2ForSpeechToText')
SeamlessM4Tv2ForTextToSpeech = lazy_import('transformers', 'SeamlessM4Tv2ForTextToSpeech')
VitsTokenizer = lazy_import('transformers', 'VitsTokenizer')
VitsModel = lazy_import('transformers', 'VitsModel')
Wav2Vec2ForCTC = lazy_import('transformers', 'Wav2Vec2ForCTC')
GPT2Tokenizer = lazy_import('transformers', 'GPT2Tokenizer')
GPT2LMHeadModel = lazy_import('transformers', 'GPT2LMHeadModel')
GenerationConfig = lazy_import('transformers', 'GenerationConfig')

# Diffusers import
diffusers = lazy_import('diffusers', '')
BlipDiffusionPipeline = lazy_import('diffusers.pipelines', 'BlipDiffusionPipeline')
StableDiffusionPipeline = lazy_import('diffusers', 'StableDiffusionPipeline')
StableDiffusionPanoramaPipeline = lazy_import('diffusers', 'StableDiffusionPanoramaPipeline')
StableDiffusion3Pipeline = lazy_import('diffusers', 'StableDiffusion3Pipeline')
StableDiffusionXLPipeline = lazy_import('diffusers', 'StableDiffusionXLPipeline')
StableDiffusionImg2ImgPipeline = lazy_import('diffusers', 'StableDiffusionImg2ImgPipeline')
StableDiffusionXLImg2ImgPipeline = lazy_import('diffusers', 'StableDiffusionXLImg2ImgPipeline')
StableDiffusion3Img2ImgPipeline = lazy_import('diffusers', 'StableDiffusion3Img2ImgPipeline')
SD3ControlNetModel = lazy_import('diffusers', 'SD3ControlNetModel')
StableDiffusion3ControlNetPipeline = lazy_import('diffusers', 'StableDiffusion3ControlNetPipeline')
StableDiffusion3InpaintPipeline = lazy_import('diffusers', 'StableDiffusion3InpaintPipeline')
StableDiffusionXLInpaintPipeline = lazy_import('diffusers', 'StableDiffusionXLInpaintPipeline')
StableDiffusionDepth2ImgPipeline = lazy_import('diffusers', 'StableDiffusionDepth2ImgPipeline')
ControlNetModel = lazy_import('diffusers', 'ControlNetModel')
StableDiffusionXLControlNetPipeline = lazy_import('diffusers', 'StableDiffusionXLControlNetPipeline')
StableDiffusionControlNetPipeline = lazy_import('diffusers', 'StableDiffusionControlNetPipeline')
AutoencoderKL = lazy_import('diffusers', 'AutoencoderKL')
StableDiffusionLatentUpscalePipeline = lazy_import('diffusers', 'StableDiffusionLatentUpscalePipeline')
StableDiffusionUpscalePipeline = lazy_import('diffusers', 'StableDiffusionUpscalePipeline')
StableDiffusionInpaintPipeline = lazy_import('diffusers', 'StableDiffusionInpaintPipeline')
StableDiffusionGLIGENPipeline = lazy_import('diffusers', 'StableDiffusionGLIGENPipeline')
StableDiffusionDiffEditPipeline = lazy_import('diffusers', 'StableDiffusionDiffEditPipeline')
AnimateDiffPipeline = lazy_import('diffusers', 'AnimateDiffPipeline')
AnimateDiffSDXLPipeline = lazy_import('diffusers', 'AnimateDiffSDXLPipeline')
AnimateDiffVideoToVideoPipeline = lazy_import('diffusers', 'AnimateDiffVideoToVideoPipeline')
MotionAdapter = lazy_import('diffusers', 'MotionAdapter')
StableVideoDiffusionPipeline = lazy_import('diffusers', 'StableVideoDiffusionPipeline')
I2VGenXLPipeline = lazy_import('diffusers', 'I2VGenXLPipeline')
StableCascadePriorPipeline = lazy_import('diffusers', 'StableCascadePriorPipeline')
StableCascadeDecoderPipeline = lazy_import('diffusers', 'StableCascadeDecoderPipeline')
DiffusionPipeline = lazy_import('diffusers', 'DiffusionPipeline')
ShapEPipeline = lazy_import('diffusers', 'ShapEPipeline')
ShapEImg2ImgPipeline = lazy_import('diffusers', 'ShapEImg2ImgPipeline')
StableAudioPipeline = lazy_import('diffusers', 'StableAudioPipeline')
AudioLDM2Pipeline = lazy_import('diffusers', 'AudioLDM2Pipeline')
StableDiffusionInstructPix2PixPipeline = lazy_import('diffusers', 'StableDiffusionInstructPix2PixPipeline')
StableDiffusionLDM3DPipeline = lazy_import('diffusers', 'StableDiffusionLDM3DPipeline')
FluxPipeline = lazy_import('diffusers', 'FluxPipeline')
FluxTransformer2DModel = lazy_import('diffusers', 'FluxTransformer2DModel')
FluxImg2ImgPipeline = lazy_import('diffusers', 'FluxImg2ImgPipeline')
FluxInpaintPipeline = lazy_import('diffusers', 'FluxInpaintPipeline')
FluxControlNetPipeline = lazy_import('diffusers', 'FluxControlNetPipeline')
FluxControlNetModel = lazy_import('diffusers', 'FluxControlNetModel')
KandinskyPipeline = lazy_import('diffusers', 'KandinskyPipeline')
KandinskyPriorPipeline = lazy_import('diffusers', 'KandinskyPriorPipeline')
KandinskyV22Pipeline = lazy_import('diffusers', 'KandinskyV22Pipeline')
KandinskyV22PriorPipeline = lazy_import('diffusers', 'KandinskyV22PriorPipeline')
AutoPipelineForText2Image = lazy_import('diffusers', 'AutoPipelineForText2Image')
KandinskyImg2ImgPipeline = lazy_import('diffusers', 'KandinskyImg2ImgPipeline')
AutoPipelineForImage2Image = lazy_import('diffusers', 'AutoPipelineForImage2Image')
AutoPipelineForInpainting = lazy_import('diffusers', 'AutoPipelineForInpainting')
HunyuanDiTPipeline = lazy_import('diffusers', 'HunyuanDiTPipeline')
HunyuanDiTControlNetPipeline = lazy_import('diffusers', 'HunyuanDiTControlNetPipeline')
HunyuanDiT2DControlNetModel = lazy_import('diffusers', 'HunyuanDiT2DControlNetModel')
LuminaText2ImgPipeline = lazy_import('diffusers', 'LuminaText2ImgPipeline')
IFPipeline = lazy_import('diffusers', 'IFPipeline')
IFSuperResolutionPipeline = lazy_import('diffusers', 'IFSuperResolutionPipeline')
IFImg2ImgPipeline = lazy_import('diffusers', 'IFImg2ImgPipeline')
IFInpaintingPipeline = lazy_import('diffusers', 'IFInpaintingPipeline')
IFImg2ImgSuperResolutionPipeline = lazy_import('diffusers', 'IFImg2ImgSuperResolutionPipeline')
IFInpaintingSuperResolutionPipeline = lazy_import('diffusers', 'IFInpaintingSuperResolutionPipeline')
PixArtAlphaPipeline = lazy_import('diffusers', 'PixArtAlphaPipeline')
PixArtSigmaPipeline = lazy_import('diffusers', 'PixArtSigmaPipeline')
CogView3PlusPipeline = lazy_import('diffusers', 'CogView3PlusPipeline')
CogVideoXPipeline = lazy_import('diffusers', 'CogVideoXPipeline')
CogVideoXDPMScheduler = lazy_import('diffusers', 'CogVideoXDPMScheduler')
CogVideoXVideoToVideoPipeline = lazy_import('diffusers', 'CogVideoXVideoToVideoPipeline')
CogVideoXImageToVideoPipeline = lazy_import('diffusers', 'CogVideoXImageToVideoPipeline')
LattePipeline = lazy_import('diffusers', 'LattePipeline')
KolorsPipeline = lazy_import('diffusers', 'KolorsPipeline')
AuraFlowPipeline = lazy_import('diffusers', 'AuraFlowPipeline')
WuerstchenDecoderPipeline = lazy_import('diffusers', 'WuerstchenDecoderPipeline')
WuerstchenPriorPipeline = lazy_import('diffusers', 'WuerstchenPriorPipeline')
StableDiffusionSAGPipeline = lazy_import('diffusers', 'StableDiffusionSAGPipeline')
DPMSolverSinglestepScheduler = lazy_import('diffusers', 'DPMSolverSinglestepScheduler')
DPMSolverMultistepScheduler = lazy_import('diffusers', 'DPMSolverMultistepScheduler')
EDMDPMSolverMultistepScheduler = lazy_import('diffusers', 'EDMDPMSolverMultistepScheduler')
EDMEulerScheduler = lazy_import('diffusers', 'EDMEulerScheduler')
KDPM2DiscreteScheduler = lazy_import('diffusers', 'KDPM2DiscreteScheduler')
KDPM2AncestralDiscreteScheduler = lazy_import('diffusers', 'KDPM2AncestralDiscreteScheduler')
EulerDiscreteScheduler = lazy_import('diffusers', 'EulerDiscreteScheduler')
EulerAncestralDiscreteScheduler = lazy_import('diffusers', 'EulerAncestralDiscreteScheduler')
HeunDiscreteScheduler = lazy_import('diffusers', 'HeunDiscreteScheduler')
LMSDiscreteScheduler = lazy_import('diffusers', 'LMSDiscreteScheduler')
DEISMultistepScheduler = lazy_import('diffusers', 'DEISMultistepScheduler')
UniPCMultistepScheduler = lazy_import('diffusers', 'UniPCMultistepScheduler')
LCMScheduler = lazy_import('diffusers', 'LCMScheduler')
DPMSolverSDEScheduler = lazy_import('diffusers', 'DPMSolverSDEScheduler')
TCDScheduler = lazy_import('diffusers', 'TCDScheduler')
DDIMScheduler = lazy_import('diffusers', 'DDIMScheduler')
DDPMScheduler = lazy_import('diffusers', 'DDPMScheduler')
DDIMInverseScheduler = lazy_import('diffusers', 'DDIMInverseScheduler')
PNDMScheduler = lazy_import('diffusers', 'PNDMScheduler')
FlowMatchEulerDiscreteScheduler = lazy_import('diffusers', 'FlowMatchEulerDiscreteScheduler')
FlowMatchHeunDiscreteScheduler = lazy_import('diffusers', 'FlowMatchHeunDiscreteScheduler')
CosineDPMSolverMultistepScheduler = lazy_import('diffusers', 'CosineDPMSolverMultistepScheduler')
DEFAULT_STAGE_C_TIMESTEPS = lazy_import('diffusers.pipelines.wuerstchen', 'DEFAULT_STAGE_C_TIMESTEPS')
AutoencoderTiny = lazy_import('diffusers', 'AutoencoderTiny')
ConsistencyDecoderVAE = lazy_import('diffusers', 'ConsistencyDecoderVAE')

# Another imports
AutoGPTQForCausalLM = lazy_import('auto_gptq', 'AutoGPTQForCausalLM')
AutoAWQForCausalLM = lazy_import('awq', 'AutoAWQForCausalLM')
ExLlamaV2 = lazy_import('exllamav2', 'ExLlamaV2')
ExLlamaV2Config = lazy_import('exllamav2', 'ExLlamaV2Config')
ExLlamaV2Cache = lazy_import('exllamav2', 'ExLlamaV2Cache')
ExLlamaV2Tokenizer = lazy_import('exllamav2', 'ExLlamaV2Tokenizer')
ExLlamaV2StreamingGenerator = lazy_import('exllamav2.generator', 'ExLlamaV2StreamingGenerator')
ExLlamaV2Sampler = lazy_import('exllamav2.generator', 'ExLlamaV2Sampler')
ExLlamaV2Lora = lazy_import('exllamav2.lora', 'ExLlamaV2Lora')
TTS = lazy_import('TTS.api', 'TTS')
whisper = lazy_import('whisper', "")
AuraSR = lazy_import('aura_sr', 'AuraSR')
OpenposeDetector = lazy_import('controlnet_aux', 'OpenposeDetector')
LineartDetector = lazy_import('controlnet_aux', 'LineartDetector')
HEDdetector = lazy_import('controlnet_aux', 'HEDdetector')
Llama = lazy_import('llama_cpp', 'Llama')
MoondreamChatHandler = lazy_import('llama_cpp.llama_chat_format', 'MoondreamChatHandler')
StableDiffusion = lazy_import('stable_diffusion_cpp', 'StableDiffusion')
MusicGen = lazy_import('audiocraft.models', 'MusicGen')
AudioGen = lazy_import('audiocraft.models', 'AudioGen')
MultiBandDiffusion = lazy_import('audiocraft.models', 'MultiBandDiffusion')
MAGNeT = lazy_import('audiocraft.models', 'MAGNeT')
IPAdapterFaceIDPlus = lazy_import('ip_adapter.ip_adapter_faceid', 'IPAdapterFaceIDPlus')
Separator = lazy_import('audio_separator.separator', 'Separator')
pixelize = lazy_import('pixeloe.pixelize', 'pixelize')
RVCInference = lazy_import('rvc_python.infer', 'RVCInference')
remove = lazy_import('rembg', 'remove')
PeftModel = lazy_import('peft', 'PeftModel')


XFORMERS_AVAILABLE = False
try:
    torch.cuda.is_available()
    import xformers
    import xformers.ops

    XFORMERS_AVAILABLE = True
except ImportError:
    print("Xformers is not installed. Proceeding without it")
    pass

chat_history = None
chat_dir = None
tts_model = None
whisper_model = None
audiocraft_model_path = None
multiband_diffusion_path = None
stop_signal = False


def print_system_info():
    print(f"NeuroSandboxWebUI")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {sys.platform}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"CPU cores: {os.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")
    print(f"Disk space: {psutil.disk_usage('/').total / (1024 ** 3):.2f} GB")

    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
    else:
        print("CUDA available: No")

    if XFORMERS_AVAILABLE:
        print(f"Xformers version: {xformers.__version__}")
    elif not XFORMERS_AVAILABLE:
        print("Xformers available: No")

    print(f"PyTorch version: {torch.__version__}")


try:
    print_system_info()
except Exception as e:
    print(f"Unable to access system information: {e}")
    pass


def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def stop_generation():
    global stop_signal
    stop_signal = True


def generate_key(length=16):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def string_to_seed(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**32)


def encrypt_image(image, key):
    img_array = np.array(image)
    seed = string_to_seed(key)
    np.random.seed(seed)
    noise = np.random.randint(0, 256, img_array.shape, dtype=np.uint8)
    encrypted_img = img_array ^ noise
    return Image.fromarray(encrypted_img), key


def decrypt_image(encrypted_image, key):
    encrypted_array = np.array(encrypted_image)
    seed = string_to_seed(key)
    np.random.seed(seed)
    noise = np.random.randint(0, 256, encrypted_array.shape, dtype=np.uint8)
    decrypted_img = encrypted_array ^ noise
    return Image.fromarray(decrypted_img)


def add_metadata_to_file(file_path, metadata):
    try:
        metadata_str = json.dumps(metadata)
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension in ['.png', '.jpeg', '.gif']:
            img = Image.open(file_path)
            metadata = PngInfo()
            metadata.add_text("COMMENT", metadata_str)
            img.save(file_path, pnginfo=metadata)
        elif file_extension in ['.mp4', '.avi', '.mov']:
            with taglib.File(file_path, save_on_exit=True) as video:
                video.tags["COMMENT"] = [metadata_str]
        elif file_extension in ['.wav', '.mp3', '.ogg']:
            with taglib.File(file_path, save_on_exit=True) as audio:
                audio.tags["COMMENT"] = [metadata_str]
        else:
            print(f"Unsupported file type: {file_extension}")

        gr.Info(f"Metadata successfully added to {file_path}")
    except Exception as e:
        gr.Warning(f"Error adding metadata to {file_path}: {str(e)}", duration=5)


def load_translation(lang):
    try:
        with open(f"TechnicalFiles/translations/{lang}.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

translations = {
    "EN": {},
    "RU": load_translation("ru"),
    "ZH": load_translation("zh")
}


def _(text, lang="EN"):
    return translations[lang].get(text, text)


def load_settings():
    if not os.path.exists('Settings.json'):
        default_settings = {
            "language": "EN",
            "share_mode": False,
            "debug_mode": False,
            "monitoring_mode": False,
            "auto_launch": False,
            "show_api": False,
            "api_open": False,
            "queue_max_size": 10,
            "status_update_rate": "auto",
            "max_file_size": 1000,
            "auth": {"username": "admin", "password": "admin"},
            "server_name": "localhost",
            "server_port": 7860,
            "hf_token": "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "share_server_address": None,
            "theme": "Default",
            "custom_theme": {
                "enabled": False,
                "primary_hue": "red",
                "secondary_hue": "pink",
                "neutral_hue": "stone",
                "spacing_size": "spacing_md",
                "radius_size": "radius_md",
                "text_size": "text_md",
                "font": "Arial",
                "font_mono": "Courier New"
            }
        }
        with open('Settings.json', 'w') as f:
            json.dump(default_settings, f, indent=4)
        return default_settings

    with open('Settings.json', 'r') as f:
        settings = json.load(f)

    if settings['share_server_address'] == "":
        settings['share_server_address'] = None

    return settings


def save_settings(settings):
    if settings['share_server_address'] is None:
        settings['share_server_address'] = ""

    with open('Settings.json', 'w') as f:
        json.dump(settings, f, indent=4)


def authenticate(username, password):
    settings = load_settings()
    auth = settings.get('auth', {})
    return username == auth.get('username') and password == auth.get('password')


def perform_web_search(query):
    try:
        query_lower = query.lower()

        if "What is current time in" in query_lower:
            location = query_lower.replace("current time in", "").strip()
            url = f"http://worldtimeapi.org/api/timezone/{urllib.parse.quote(location)}"
            response = requests.get(url)
            data = response.json()
            result = data.get('datetime', 'Could not retrieve time information.')

        elif "What is today day in" in query_lower:
            location = query_lower.replace("today day in", "").strip()
            url = f"http://worldtimeapi.org/api/timezone/{urllib.parse.quote(location)}"
            response = requests.get(url)
            data = response.json()
            datetime_str = data.get('datetime', 'Could not retrieve date information.')
            date_str = datetime_str.split("T")[0] if 'datetime' in data else 'Date not available'
            result = date_str

        elif "What is today weather in" in query_lower:
            location = query_lower.replace("today weather in", "").strip()
            url = f"https://wttr.in/{urllib.parse.quote(location)}?format=%C+%t+%w"
            response = requests.get(url)
            result = response.text.strip()

        else:
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1"
            response = requests.get(url)
            data = response.json()

            if data.get('AbstractText'):
                result = data['AbstractText']
            elif data.get('Answer'):
                result = data['Answer']
            elif data.get('RelatedTopics'):
                result = data['RelatedTopics']
            else:
                result = "No relevant information found."

        gr.Info(f"Web search results for '{query}': {result}")

        return result

    finally:
        flush()


def parse_pdf(pdf_path):
    parser = openparse.DocumentParser()
    parsed_doc = parser.parse(pdf_path)
    parsed_content = ""
    for node in parsed_doc.nodes:
        parsed_content += str(node) + "\n"
    return parsed_content


def remove_bg(src_img_path, out_img_path):
    model_path = "inputs/image/sd_models/rembg"
    os.makedirs(model_path, exist_ok=True)

    os.environ["U2NET_HOME"] = model_path

    with open(src_img_path, "rb") as input_file:
        input_data = input_file.read()

    output_data = remove().remove(input_data)

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


def downscale_image(image_path, scale_factor):
    with Image.open(image_path) as img:
        original_width, original_height = img.size
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        output_format = os.path.splitext(image_path)[1][1:]
        today = datetime.now().date()
        output_dir = os.path.join('outputs', f"Extras_{today.strftime('%Y%m%d')}")
        os.makedirs(output_dir, exist_ok=True)

        output_filename = f"downscaled_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        output_path = os.path.join(output_dir, output_filename)
        resized_img.save(output_path)

    return output_path


def downscale_video(video_path, scale_factor):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    today = datetime.now().date()
    output_dir = os.path.join('outputs', f"Extras_{today.strftime('%Y%m%d')}")
    os.makedirs(output_dir, exist_ok=True)

    output_format = os.path.splitext(video_path)[1][1:]
    output_filename = f"downscaled_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
    output_path = os.path.join(output_dir, output_filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        out.write(resized_frame)

    cap.release()
    out.release()

    return output_path


def downscale_audio(input_path, output_path, downscale_factor):
    audio = AudioSegment.from_file(input_path)
    original_sample_rate = audio.frame_rate
    target_sample_rate = int(original_sample_rate * downscale_factor)
    downscaled_audio = audio.set_frame_rate(target_sample_rate)
    downscaled_audio.export(output_path, format="wav")
    return original_sample_rate, target_sample_rate


def change_image_format(input_image, new_format, enable_format_changer):
    if not input_image or not enable_format_changer:
        gr.Info("Please upload an image and enable format changer!")
        return None, None

    try:
        input_format = os.path.splitext(input_image)[1][1:]
        if input_format == new_format:
            return input_image, "Input and output formats are the same. No change needed."

        output_filename = f"format_changed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{new_format}"
        output_path = os.path.join(os.path.dirname(input_image), output_filename)

        img = Image.open(input_image)
        img.save(output_path, format=new_format.upper())

        return output_path, f"Image format changed from {input_format} to {new_format}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None


def change_video_format(input_video, new_format, enable_format_changer):
    if not input_video or not enable_format_changer:
        gr.Info("Please upload a video and enable format changer!")
        return None, None

    try:
        input_format = os.path.splitext(input_video)[1][1:]
        if input_format == new_format:
            return input_video, "Input and output formats are the same. No change needed."

        output_filename = f"format_changed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{new_format}"
        output_path = os.path.join(os.path.dirname(input_video), output_filename)

        command = f"ffmpeg -i {input_video} -c copy {output_path}"
        subprocess.run(command, shell=True, check=True)

        return output_path, f"Video format changed from {input_format} to {new_format}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None


def change_audio_format(input_audio, new_format, enable_format_changer):
    if not input_audio or not enable_format_changer:
        gr.Info("Please upload an audio file and enable format changer!")
        return None, None

    try:
        input_format = os.path.splitext(input_audio)[1][1:]
        if input_format == new_format:
            return input_audio, "Input and output formats are the same. No change needed."

        output_filename = f"format_changed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{new_format}"
        output_path = os.path.join(os.path.dirname(input_audio), output_filename)

        command = f"ffmpeg -i {input_audio} {output_path}"
        subprocess.run(command, shell=True, check=True)

        return output_path, f"Audio format changed from {input_format} to {new_format}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None


def load_magicprompt_model():
    model_path = os.path.join("inputs", "image", "sd_models", "MagicPrompt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    if not os.path.exists(model_path):
        gr.Info("Downloading MagicPrompt model...")
        os.makedirs(model_path, exist_ok=True)
        tokenizer = GPT2Tokenizer().GPT2Tokenizer.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
        model = GPT2LMHeadModel().GPT2LMHeadModel.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion", torch_dtype=torch_dtype)
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
        gr.Info("MagicPrompt model downloaded")
    else:
        tokenizer = GPT2Tokenizer().GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel().GPT2LMHeadModel.from_pretrained(model_path, torch_dtype=torch_dtype)
    return tokenizer, model


def generate_magicprompt(prompt, max_new_tokens):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_magicprompt_model()
    model.to(device)
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    token_count = inputs["input_ids"].shape[1]
    max_new_tokens = min(max_new_tokens, 256 - token_count)
    generation_config = GenerationConfig().GenerationConfig(
        penalty_alpha=0.7,
        top_k=50,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.eos_token_id,
        pad_token=model.config.pad_token_id,
        do_sample=True,
    )
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            generation_config=generation_config,
        )
    enhanced_prompt = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return enhanced_prompt


def load_model(model_name, model_type, n_ctx, n_batch, n_ubatch, freq_base, freq_scale):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    if model_name:
        model_path = f"inputs/text/llm_models/{model_name}"
        if model_type == "Transformers":
            try:
                tokenizer = AutoTokenizer().AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM().AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True
                )
                return tokenizer, model, None
            except Exception as e:
                gr.Error(f"An error occurred: {str(e)}")
                return None, None, None
        elif model_type == "Llama":
            try:
                model = Llama().Llama(model_path, n_gpu_layers=-1 if device == "cuda" else 0, n_ctx=n_ctx, n_batch=n_batch, n_ubatch=n_ubatch, freq_base=freq_base, freq_scale=freq_scale)
                tokenizer = None
                return tokenizer, model, None
            except Exception as e:
                gr.Error(f"An error occurred: {str(e)}")
                return None, None, None
        elif model_type == "GPTQ":
            try:
                tokenizer = AutoTokenizer().AutoTokenizer.from_pretrained(model_path)
                model = AutoGPTQForCausalLM().AutoGPTQForCausalLM.from_quantized(
                    model_path,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    trust_remote_code=True
                )
                return tokenizer, model, None
            except Exception as e:
                gr.Error(f"An error occurred: {str(e)}")
                return None, None, None
        elif model_type == "AWQ":
            try:
                tokenizer = AutoTokenizer().AutoTokenizer.from_pretrained(model_path)
                model = AutoAWQForCausalLM().AutoAWQForCausalLM.from_quantized(
                    model_path,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    trust_remote_code=True
                )
                return tokenizer, model, None
            except Exception as e:
                gr.Error(f"An error occurred: {str(e)}")
                return None, None, None
        elif model_type == "BNB":
            try:
                tokenizer = AutoTokenizer().AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM().AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True
                )
                return tokenizer, model, None
            except Exception as e:
                gr.Error(f"An error occurred: {str(e)}")
                return None, None, None
        elif model_type == "ExLlamaV2":
            try:
                config = ExLlamaV2Config().ExLlamaV2Config()
                config.model_dir = model_path
                config.prepare()
                model = ExLlamaV2().ExLlamaV2(config)
                model.load()
                cache = ExLlamaV2Cache().ExLlamaV2Cache(model)
                model.load_autosplit(cache)
                tokenizer = ExLlamaV2Tokenizer().ExLlamaV2Tokenizer(config)
                return cache, tokenizer, model
            except Exception as e:
                gr.Error(f"An error occurred: {str(e)}")
                return None, None, None


def load_lora_model(base_model_name, lora_model_name, model_type):
    base_model_path = f"inputs/text/llm_models/{base_model_name}"
    lora_model_path = f"inputs/text/llm_models/lora/{lora_model_name}"

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        if model_type == "Llama":
            model = Llama().Llama(base_model_path, n_gpu_layers=-1 if device == "cuda" else 0,
                                  lora_path=lora_model_path)
            tokenizer = None
            return tokenizer, model, None
        elif model_type in ["Transformers", "GPTQ", "AWQ", "BNB"]:
            if model_type == "Transformers":
                base_model = AutoModelForCausalLM().AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    device_map=device,
                    load_in_8bit=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True
                )
            elif model_type == "GPTQ":
                base_model = AutoGPTQForCausalLM().AutoGPTQForCausalLM.from_quantized(
                    base_model_path,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    trust_remote_code=True
                )
            elif model_type == "AWQ":
                base_model = AutoAWQForCausalLM().AutoAWQForCausalLM.from_quantized(
                    base_model_path,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    trust_remote_code=True
                )
            elif model_type == "BNB":
                base_model = AutoModelForCausalLM().AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    device_map=device,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True
                )
            model = PeftModel().PeftModel.from_pretrained(base_model, lora_model_path).to(device)
            merged_model = model.merge_and_unload()
            tokenizer = AutoTokenizer().AutoTokenizer.from_pretrained(base_model_path)
            return tokenizer, merged_model, None
        elif model_type == "ExLlamaV2":
            config = ExLlamaV2Config().ExLlamaV2Config()
            config.model_dir = base_model_path
            config.prepare()
            model = ExLlamaV2().ExLlamaV2(config)
            model.load()
            cache = ExLlamaV2Cache().ExLlamaV2Cache(model)
            model.load_autosplit(cache)
            tokenizer = ExLlamaV2Tokenizer().ExLlamaV2Tokenizer(config)
            ExLlamaV2Lora().ExLlamaV2Lora.from_directory(base_model_path, lora_model_path)
            return cache, tokenizer, model
    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None, None
    finally:
        if 'tokenizer' in locals():
            del tokenizer
        if 'merged_model' in locals():
            del merged_model
        flush()


def load_moondream2_model(model_id, revision):
    moondream2_model_path = os.path.join("inputs", "text", model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if not os.path.exists(moondream2_model_path):
            gr.Info(f"Downloading MoonDream2 model...")
            os.makedirs(moondream2_model_path, exist_ok=True)
            model = AutoModelForCausalLM().AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, revision=revision
            ).to(device)
            tokenizer = AutoTokenizer().AutoTokenizer.from_pretrained(model_id, revision=revision)
            model.save_pretrained(moondream2_model_path)
            tokenizer.save_pretrained(moondream2_model_path)
            gr.Info("MoonDream2 model downloaded")
        else:
            model = AutoModelForCausalLM().AutoModelForCausalLM.from_pretrained(moondream2_model_path, trust_remote_code=True).to(device)
            tokenizer = AutoTokenizer().AutoTokenizer.from_pretrained(moondream2_model_path)
        return model, tokenizer, None

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None, None
    finally:
        if 'tokenizer' in locals():
            del tokenizer
        if 'model' in locals():
            del model
        flush()


def load_llava_next_video_model():
    model_path = os.path.join("inputs", "text", "LLaVA-NeXT-Video")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        if not os.path.exists(model_path):
            gr.Info("Downloading LLaVA-NeXT-Video model...")
            os.makedirs(model_path, exist_ok=True)
            model = LlavaNextVideoForConditionalGeneration().LlavaNextVideoForConditionalGeneration.from_pretrained(
                "llava-hf/LLaVA-NeXT-Video-7B-hf",
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(device)
            processor = LlavaNextVideoProcessor().LlavaNextVideoProcessor.from_pretrained(
                "llava-hf/LLaVA-NeXT-Video-7B-hf")
            model.save_pretrained(model_path)
            processor.save_pretrained(model_path)
            gr.Info("LLaVA-NeXT-Video model downloaded")
        else:
            model = LlavaNextVideoForConditionalGeneration().LlavaNextVideoForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, trust_remote_code=True).to(device)
            processor = LlavaNextVideoProcessor().LlavaNextVideoProcessor.from_pretrained(model_path)
        return model, processor, None

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None, None
    finally:
        if 'processor' in locals():
            del processor
        if 'model' in locals():
            del model
        flush()


def load_qwen2_audio_model():
    qwen2_audio_path = os.path.join("inputs", "text", "Qwen2-Audio")
    if not os.path.exists(qwen2_audio_path):
        gr.Info("Downloading Qwen2-Audio model...")
        os.makedirs(qwen2_audio_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct", qwen2_audio_path)
        gr.Info("Qwen2-Audio model downloaded")

    processor = AutoProcessor().AutoProcessor.from_pretrained(qwen2_audio_path)
    model = Qwen2AudioForConditionalGeneration().Qwen2AudioForConditionalGeneration.from_pretrained(qwen2_audio_path,
                                                                                                    device_map="auto")
    return processor, model


def process_qwen2_audio(processor, model, input_audio_mm, prompt):
    conversation = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {"role": "user", "content": [
            {"type": "audio", "audio_url": input_audio_mm},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audio, _ = librosa.load(input_audio_mm, sr=processor.feature_extractor.sampling_rate)
    inputs = processor(text=text, audios=[audio], return_tensors="pt", padding=True)
    inputs.input_ids = inputs.input_ids.to("cuda")
    generate_ids = model.generate(**inputs, max_length=256)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response


def get_existing_chats():
    chat_files = []
    for root, dirs, files in os.walk('outputs'):
        for file in files:
            if file.endswith('.txt') and file.startswith('chat_history'):
                chat_files.append(os.path.join(root, file))
    return [None] + chat_files


def transcribe_audio(audio_file_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model_path = "inputs/text/whisper-medium"
    if not os.path.exists(whisper_model_path):
        gr.Info("Downloading Whisper...")
        os.makedirs(whisper_model_path, exist_ok=True)
        url = ("https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt")
        r = requests.get(url, allow_redirects=True)
        open(os.path.join(whisper_model_path, "large-v3-turbo.pt"), "wb").write(r.content)
        gr.Info("Whisper model downloaded")
    model_file = os.path.join(whisper_model_path, "large-v3-turbo.pt")
    model = whisper().whisper.load_model(model_file, device=device)
    result = model.transcribe(audio_file_path)
    return result["text"]


def load_tts_model():
    tts_model_path = "inputs/audio/XTTS-v2"
    if not os.path.exists(tts_model_path):
        gr.Info("Downloading TTS...")
        os.makedirs(tts_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/coqui/XTTS-v2", tts_model_path)
        gr.Info("TTS model downloaded")
    return TTS().TTS(model_path=tts_model_path, config_path=f"{tts_model_path}/config.json")


def load_freevc_model():
    model_path = "voice_conversion_models/multilingual/vctk/freevc24"
    if not os.path.exists(model_path):
        gr.Info("Downloading FreeVC...")
        os.makedirs(model_path, exist_ok=True)
        url = "https://huggingface.co/spaces/OlaWod/FreeVC/resolve/main/checkpoints/freevc.pth"
        r = requests.get(url, allow_redirects=True)
        with open(os.path.join(model_path, "freevc.pth"), "wb") as f:
            f.write(r.content)
        gr.Info("FreeVC model downloaded")
    return model_path


def load_whisper_model():
    whisper_model_path = "inputs/text/whisper-medium"
    if not os.path.exists(whisper_model_path):
        gr.Info("Downloading Whisper...")
        os.makedirs(whisper_model_path, exist_ok=True)
        url = (
            "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt")
        r = requests.get(url, allow_redirects=True)
        open(os.path.join(whisper_model_path, "large-v3-turbo.pt"), "wb").write(r.content)
        gr.Info("Whisper model downloaded")
    model_file = os.path.join(whisper_model_path, "large-v3-turbo.pt")
    return whisper().whisper.load_model(model_file)


def load_audiocraft_model(model_name):
    global audiocraft_model_path
    audiocraft_model_path = os.path.join("inputs", "audio", "audiocraft", model_name)
    if not os.path.exists(audiocraft_model_path):
        gr.Info(f"Downloading AudioCraft model: {model_name}...")
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
        gr.Info(f"AudioCraft model {model_name} downloaded")
    return audiocraft_model_path


def load_multiband_diffusion_model():
    multiband_diffusion_path = os.path.join("inputs", "audio", "audiocraft", "multiband-diffusion")
    if not os.path.exists(multiband_diffusion_path):
        gr.Info(f"Downloading Multiband Diffusion model")
        os.makedirs(multiband_diffusion_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/facebook/multiband-diffusion", multiband_diffusion_path)
        gr.Info("Multiband Diffusion model downloaded")
    return multiband_diffusion_path


def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def get_languages():
    return {
        "Arabic": "ara", "Chinese": "cmn", "English": "eng", "French": "fra",
        "German": "deu", "Hindi": "hin", "Italian": "ita", "Japanese": "jpn",
        "Korean": "kor", "Polish": "pol", "Portuguese": "por", "Russian": "rus",
        "Spanish": "spa", "Turkish": "tur",
    }


styles_filepath = "inputs/image/styles.csv"


def load_styles_from_csv(filepath):
    data = pd.read_csv(filepath)
    styles_dict = {}
    for _, row in data.iterrows():
        name = row['name']
        prompt = row['prompt']
        negative_prompt = row['negative_prompt']
        styles_dict[name] = {
            'prompt': prompt,
            'negative_prompt': negative_prompt
        }
    return styles_dict


styles = load_styles_from_csv(styles_filepath)


def generate_text_and_speech(input_text, system_prompt, input_audio, llm_model_type, llm_model_name, llm_lora_model_name, selected_chat_file, enable_web_search, enable_libretranslate, target_lang, enable_openparse, pdf_file, enable_multimodal, input_image, input_video, input_audio_mm, enable_tts,
                             llm_settings_html, max_new_tokens, max_length, min_length, n_ctx, n_batch, n_ubatch, freq_base, freq_scale, temperature, top_p, min_p, typical_p, top_k,
                             do_sample, early_stopping, stopping, repetition_penalty, frequency_penalty, presence_penalty, length_penalty, no_repeat_ngram_size, num_beams, num_return_sequences, chat_history_format, tts_settings_html, speaker_wav, language, tts_temperature, tts_top_p, tts_top_k, tts_speed, tts_repetition_penalty, tts_length_penalty, output_format, chat_history_state):
    global chat_dir, tts_model, whisper_model

    chat_history = chat_history_state

    if selected_chat_file:
        chat_dir = os.path.dirname(selected_chat_file)
        file_extension = os.path.splitext(selected_chat_file)[1].lower()

        if file_extension == '.txt':
            with open(selected_chat_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                chat_history = []
                for line in lines:
                    if line.startswith("Human:"):
                        chat_history.append([line.split(":", 1)[1].strip(), None])
                    elif line.startswith("AI:"):
                        if chat_history and chat_history[-1][1] is None:
                            chat_history[-1][1] = line.split(":", 1)[1].strip()
                        else:
                            chat_history.append([None, line.split(":", 1)[1].strip()])
        elif file_extension == '.json':
            with open(selected_chat_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            chat_history = []
            for entry in json_data:
                if isinstance(entry, list) and len(entry) == 2:
                    human_message, ai_message = entry
                    if human_message.startswith("Human:"):
                        human_message = human_message[6:].strip()
                    if ai_message.startswith("AI:"):
                        ai_message = ai_message[3:].strip()
                    chat_history.append([human_message, ai_message])
                else:
                    gr.Info(f"Skipping invalid entry in JSON: {entry}")
        else:
            gr.Info(f"Unsupported file format: {file_extension}")
            return None, None

        yield chat_history_state, None
    elif 'chat_history' not in globals() or chat_history is None:
        chat_history = []

    if not input_text and not input_audio:
        gr.Info("Please, enter your request!")
        return None, None
    prompt = transcribe_audio(input_audio) if input_audio else input_text

    if not llm_model_name:
        gr.Info("Please, select a LLM model!")
        return None, None

    if not system_prompt:
        system_prompt = "You are a helpful assistant."

    if enable_web_search:
        search_results = perform_web_search(prompt)
        web_context = f"Web search results: {search_results}\n\n"
    else:
        web_context = ""

    if enable_openparse and pdf_file:
        parsed_content = parse_pdf(pdf_file.name)
        openparse_context = f"Parsed PDF content: {parsed_content}\n\n"
    else:
        openparse_context = ""

    if enable_multimodal and llm_model_name == "Moondream2-Image":
        if llm_model_type == "Llama":
            moondream2_path = os.path.join("inputs", "text", "moondream2-cpp")

            if not os.path.exists(moondream2_path):
                gr.Info("Downloading Moondream2 model...")
                os.makedirs(moondream2_path, exist_ok=True)
                Repo.clone_from("https://huggingface.co/vikhyatk/moondream2", moondream2_path)
                gr.Info("Moondream2 model downloaded")

            chat_handler = MoondreamChatHandler().MoondreamChatHandler.from_pretrained(
                repo_id="vikhyatk/moondream2",
                local_dir=moondream2_path,
                filename="*mmproj*",
            )

            llm = Llama().Llama.from_pretrained(
                repo_id="vikhyatk/moondream2",
                local_dir=moondream2_path,
                filename="*text-model*",
                chat_handler=chat_handler,
                n_ctx=n_ctx,
            )
            try:
                if input_image:
                    image_path = input_image

                    def image_to_base64_data_uri(image_path):
                        with open(image_path, "rb") as img_file:
                            base64_data = base64.b64encode(img_file.read()).decode('utf-8')
                            return f"data:image/png;base64,{base64_data}"

                    data_uri = image_to_base64_data_uri(image_path)
                else:
                    gr.Info("Please upload an image for multimodal input.")
                    return None, None

                context = ""
                for human_text, ai_text in chat_history[-5:]:
                    if human_text:
                        context += f"Human: {human_text}\n"
                    if ai_text:
                        context += f"AI: {ai_text}\n"

                if not chat_history or chat_history[-1][1] is not None:
                    chat_history.append([prompt, ""])

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"{context}Human: {prompt}"},
                        {"type": "image_url", "image_url": {"url": data_uri}}
                    ]}
                ]

                for token in llm.create_chat_completion(messages=messages):
                    chat_history[-1][1] += token
                    yield chat_history_state, None

            except Exception as e:
                gr.Error(f"An error occurred: {str(e)}")
                return None, None

            finally:
                if 'llm' in locals():
                    del llm
                if 'chat_handler' in locals():
                    del chat_handler
                flush()

        else:
            model_id = "vikhyatk/moondream2"
            revision = "2024-08-26"
            model, tokenizer = load_moondream2_model(model_id, revision)

            try:
                if input_image:
                    image = Image.open(input_image)
                    enc_image = model.encode_image(image)
                else:
                    gr.Info("Please upload an image for multimodal input.")
                    return None, None

                context = ""
                for human_text, ai_text in chat_history[-5:]:
                    if human_text:
                        context += f"Human: {human_text}\n"
                    if ai_text:
                        context += f"AI: {ai_text}\n"

                prompt_with_context = f"{system_prompt}\n\n{context}Human: {prompt}\nAI:"

                if not chat_history or chat_history[-1][1] is not None:
                    chat_history.append([prompt, ""])

                for token in model.answer_question(enc_image, prompt_with_context, tokenizer):
                    chat_history[-1][1] += token
                    yield chat_history_state, None

            except Exception as e:
                gr.Error(f"An error occurred: {str(e)}")
                return None, None

            finally:
                if 'tokenizer' in locals():
                    del tokenizer
                if 'model' in locals():
                    del model
                flush()

    elif enable_multimodal and llm_model_name == "LLaVA-NeXT-Video":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if llm_model_type == "Llama":
            gr.Info("LLaVA-NeXT-Video is not supported with llama model type.")
            return None, None
        else:

            model, processor = load_llava_next_video_model()

            try:
                if input_video:
                    container = av.open(input_video)
                    total_frames = container.streams.video[0].frames
                    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
                    clip = read_video_pyav(container, indices)
                else:
                    gr.Info("Please upload a video for LLaVA-NeXT-Video input.")
                    return None, None

                context = ""
                for human_text, ai_text in chat_history[-5:]:
                    if human_text:
                        context += f"Human: {human_text}\n"
                    if ai_text:
                        context += f"AI: {ai_text}\n"

                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{context}Human: {prompt}"},
                            {"type": "video"},
                        ],
                    },
                ]

                if not chat_history or chat_history[-1][1] is not None:
                    chat_history.append([prompt, ""])

                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs_video = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(device)

                output = model.generate(**inputs_video, max_new_tokens=max_new_tokens, do_sample=do_sample,
                                        temperature=temperature, top_p=top_p, top_k=top_k)

                response = processor.decode(output[0][2:], skip_special_tokens=True)
                chat_history[-1][1] = response
                yield chat_history_state, None

            except Exception as e:
                gr.Error(f"An error occurred: {str(e)}")
                return None, None

            finally:
                if 'processor' in locals():
                    del processor
                if 'model' in locals():
                    del model
                flush()

    elif enable_multimodal and llm_model_name == "Qwen2-Audio":
        if not input_audio_mm:
            gr.Info("Please upload a audio for Qwen2-Audio input.")
            return None, None
        processor, model = load_qwen2_audio_model()
        if llm_model_type == "Llama":
            gr.Info("Qwen2-Audio is not supported with llama model type.")
            return None, None
        else:
            try:
                response = process_qwen2_audio(processor, model, input_audio_mm, prompt)
                if not chat_history or chat_history[-1][1] is not None:
                    chat_history.append([prompt, ""])
                chat_history[-1][1] = response
                yield chat_history_state, None
            except Exception as e:
                gr.Error(f"An error occurred: {str(e)}")
                return None, None
            finally:
                if 'processor' in locals():
                    del processor
                if 'model' in locals():
                    del model
                flush()

    else:
        if llm_model_type == "Llama":
            tokenizer, llm_model, * _ = load_model(llm_model_name, llm_model_type, n_ctx, n_batch, n_ubatch, freq_base, freq_scale)
        elif llm_model_type == "ExLlamaV2":
            cache, tokenizer, llm_model, * _ = load_model(llm_model_name, llm_model_type, n_ctx=None, n_batch=None, n_ubatch=None, freq_base=None, freq_scale=None)
        else:
            tokenizer, llm_model, * _ = load_model(llm_model_name, llm_model_type, n_ctx=None, n_batch=None, n_ubatch=None, freq_base=None, freq_scale=None)

        if llm_lora_model_name:
            tokenizer, llm_model = load_lora_model(llm_model_name, llm_lora_model_name, llm_model_type)
        tts_model = None
        whisper_model = None
        text = None
        audio_path = None

        try:
            if enable_tts:
                if not tts_model:
                    tts_model = load_tts_model()
                if not speaker_wav or not language:
                    gr.Info("Please, select a voice and language for TTS!")
                    return None, None
                device = "cuda" if torch.cuda.is_available() else "cpu"
                tts_model = tts_model.to(device)
            if input_audio:
                if not whisper_model:
                    whisper_model = load_whisper_model()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                whisper_model = whisper_model.to(device)
            if llm_model:
                context = ""
                for human_text, ai_text in chat_history[-10:]:
                    if human_text:
                        context += f"Human: {human_text}\n"
                    if ai_text:
                        context += f"AI: {ai_text}\n"

                if llm_model_type in ["Transformers", "GPTQ", "AWQ", "BNB"]:
                    device = "cuda" if torch.cuda.is_available() else "cpu"

                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.padding_side = "left"
                    stop_words = stopping.split(',') if stopping.strip() else None
                    stop_ids = [tokenizer.encode(word.strip(), add_special_tokens=False)[0] for word in
                                stop_words] if stop_words else None

                    full_prompt = f"<|im_start|>system\n{openparse_context}{web_context}{context}{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

                    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True).to(device)

                    streamer = TextIteratorStreamer().TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

                    generation_kwargs = dict(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        length_penalty=length_penalty,
                        num_beams=num_beams,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        num_return_sequences=num_return_sequences,
                        early_stopping=early_stopping,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=stop_ids if stop_ids else tokenizer.eos_token_id,
                        streamer=streamer
                    )

                    thread = Thread(target=llm_model.generate, kwargs=generation_kwargs)
                    thread.start()

                    text = ""
                    if not chat_history or chat_history[-1][1] is not None:
                        chat_history.append([prompt, ""])

                    for new_text in streamer:
                        text += new_text
                        chat_history[-1][1] = text
                        yield chat_history_state, None

                    thread.join()

                elif llm_model_type == "Llama":
                    text = ""
                    if not chat_history or chat_history[-1][1] is not None:
                        chat_history.append([prompt, ""])

                    stop_sequences = [seq.strip() for seq in stopping.split(',')] if stopping.strip() else None

                    full_prompt = f"<|im_start|>system\n{openparse_context}{web_context}{context}{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

                    for token in llm_model(
                            full_prompt,
                            max_tokens=max_new_tokens,
                            top_p=top_p,
                            min_p=min_p,
                            typical_p=typical_p,
                            top_k=top_k,
                            temperature=temperature,
                            repeat_penalty=repetition_penalty,
                            frequency_penalty=frequency_penalty,
                            presence_penalty=presence_penalty,
                            stop=stop_sequences,
                            stream=True,
                            echo=False
                    ):

                        text += token['choices'][0]['text']

                        chat_history[-1][1] = text

                        yield chat_history_state, None

                elif llm_model_type == "ExLlamaV2":
                    text = ""
                    if not chat_history or chat_history[-1][1] is not None:
                        chat_history.append([prompt, ""])

                    generator = ExLlamaV2StreamingGenerator().ExLlamaV2StreamingGenerator(cache, llm_model, tokenizer)

                    settings = ExLlamaV2Sampler().ExLlamaV2Sampler.Settings(
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        typical=typical_p,
                        token_repetition_penalty=repetition_penalty,
                        token_frequency_penalty=frequency_penalty,
                        token_presence_penalty=presence_penalty
                    )
                    generator.set_stop_conditions(tokenizer.encode(stopping) if stopping else None)

                    full_prompt = f"<|im_start|>system\n{openparse_context}{web_context}{context}{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

                    for token in generator.generate_simple(full_prompt, settings, max_new_tokens):
                        text += token['choices'][0]['text']
                        chat_history[-1][1] = text
                        yield chat_history_state, None

                if enable_libretranslate:
                    try:
                        translator = LibreTranslateAPI("http://127.0.0.1:5000")
                        detect_lang = langdetect.detect(text)
                        translation = translator.translate(text, detect_lang, target_lang)
                        text = translation
                    except urllib.error.URLError:
                        gr.Info("LibreTranslate is not running. Please start the LibreTranslate server.")
                        return None, None

                if not chat_dir:
                    now = datetime.now()
                    chat_dir = os.path.join('outputs', f"LLM_{now.strftime('%Y%m%d_%H%M%S')}")
                    os.makedirs(chat_dir)
                    os.makedirs(os.path.join(chat_dir, 'text'))
                    os.makedirs(os.path.join(chat_dir, 'audio'))

                chat_history_path = selected_chat_file if selected_chat_file else os.path.join(chat_dir, 'text', f'chat_history.{chat_history_format}')
                if chat_history_format == "txt":
                    with open(chat_history_path, "a", encoding="utf-8") as f:
                        f.write(f"Human: {prompt}\n")
                        if text:
                            f.write(f"AI: {text}\n\n")
                elif chat_history_format == "json":
                    chat_history_json = []
                    if os.path.exists(chat_history_path):
                        with open(chat_history_path, "r", encoding="utf-8") as f:
                            chat_history_json = json.load(f)
                    chat_history_json.append([f"Human: {prompt}", f"AI: {text}" if text else ""])
                    with open(chat_history_path, "w", encoding="utf-8") as f:
                        json.dump(chat_history_json, f, ensure_ascii=False, indent=4)
                if enable_tts and text:
                    wav = tts_model.tts(text=text, speaker_wav=f"inputs/audio/voices/{speaker_wav}", language=language,
                                        temperature=tts_temperature, top_p=tts_top_p, top_k=tts_top_k, speed=tts_speed,
                                        repetition_penalty=tts_repetition_penalty, length_penalty=tts_length_penalty)
                    now = datetime.now()
                    audio_filename = f"TTS_{now.strftime('%Y%m%d_%H%M%S')}.{output_format}"
                    audio_path = os.path.join(chat_dir, 'audio', audio_filename)
                    if output_format == "mp3":
                        sf.write(audio_path, wav, 22050, format='mp3')
                    elif output_format == "ogg":
                        sf.write(audio_path, wav, 22050, format='ogg')
                    else:
                        sf.write(audio_path, wav, 22050)

        except Exception as e:
            gr.Error(f"An error occurred: {str(e)}")
            return None, None

        finally:
            if 'tokenizer' in locals():
                del tokenizer
            if 'llm_model' in locals():
                del llm_model
            if 'tts_model' in locals():
                del tts_model
            if 'whisper_model' in locals():
                del whisper_model
            flush()

        if chat_history:
            chat_history[-1][1] = text
        else:
            chat_history.append([prompt, text])

        yield chat_history_state, audio_path


def generate_tts_stt(text, audio, tts_settings_html, speaker_wav, language, tts_temperature, tts_top_p, tts_top_k, tts_speed, tts_repetition_penalty, tts_length_penalty, enable_conversion, source_wav, target_wav, tts_output_format, stt_output_format, progress=gr.Progress()):
    global tts_model, whisper_model

    tts_output = None
    stt_output = None

    if not enable_conversion and not audio and not text:
        gr.Info("Please enter text for TTS or record audio for STT!")
        return None, None

    try:
        if enable_conversion:
            if not source_wav and target_wav:
                gr.Info("Please upload source and target files for voice conversion!")
                return None, None

            device = "cuda" if torch.cuda.is_available() else "cpu"
            progress(0.5, desc="Performing voice conversion")
            freevc_model_path = load_freevc_model()
            tts = TTS().TTS(model_name=freevc_model_path, progress_bar=False).to(device)

            today = datetime.now().date()
            conversion_dir = os.path.join('outputs', f"VoiceConversion_{today.strftime('%Y%m%d')}")
            os.makedirs(conversion_dir, exist_ok=True)
            conversion_filename = f"converted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            conversion_output = os.path.join(conversion_dir, conversion_filename)

            tts.voice_conversion_to_file(source_wav=source_wav, target_wav=target_wav, file_path=conversion_output)

            progress(1.0, desc="Voice conversion complete")

        if text:
            progress(0, desc="Processing TTS")
            if not tts_model:
                progress(0.1, desc="Loading TTS model")
                tts_model = load_tts_model()
            if not speaker_wav or not language:
                gr.Info("Please select a voice and language for TTS!")
                return None, None
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tts_model = tts_model.to(device)

            progress(0.3, desc="Generating speech")
            wav = tts_model.tts(text=text, speaker_wav=f"inputs/audio/voices/{speaker_wav}", language=language,
                                temperature=tts_temperature, top_p=tts_top_p, top_k=tts_top_k, speed=tts_speed,
                                repetition_penalty=tts_repetition_penalty, length_penalty=tts_length_penalty)

            progress(0.7, desc="Saving audio file")
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

            progress(1.0, desc="TTS processing complete")

        if audio:
            progress(0, desc="Processing STT")
            if not whisper_model:
                progress(0.2, desc="Loading STT model")
                whisper_model = load_whisper_model()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            whisper_model = whisper_model.to(device)

            progress(0.5, desc="Transcribing audio")
            stt_output = transcribe_audio(audio)

            if stt_output:
                progress(0.8, desc="Saving transcription")
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

            progress(1.0, desc="STT processing complete")

        if text or audio:
            return tts_output, stt_output
        elif enable_conversion:
            return conversion_output, None

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'tts_model' in locals():
            del tts_model
        if 'whisper_model' in locals():
            del whisper_model
        flush()


def generate_mms_tts(text, language, output_format, progress=gr.Progress()):
    model_names = {
        "English": "facebook/mms-tts-eng",
        "Russian": "facebook/mms-tts-rus",
        "Korean": "facebook/mms-tts-kor",
        "Hindu": "facebook/mms-tts-hin",
        "Turkish": "facebook/mms-tts-tur",
        "French": "facebook/mms-tts-fra",
        "Spanish": "facebook/mms-tts-spa",
        "German": "facebook/mms-tts-deu",
        "Arabic": "facebook/mms-tts-ara",
        "Polish": "facebook/mms-tts-pol"
    }

    model_path = os.path.join("inputs", "text", "mms", "text2speech", language)

    if not os.path.exists(model_path):
        progress(0.1, desc="Downloading MMS TTS model...")
        gr.Info(f"Downloading MMS TTS model for {language}...")
        os.makedirs(model_path, exist_ok=True)
        Repo.clone_from(f"https://huggingface.co/{model_names[language]}", model_path)
        gr.Info(f"MMS TTS model for {language} downloaded")
        progress(0.3, desc="Model downloaded")

    if not text:
        gr.Info("Please enter your request!")
        return None

    if not language:
        gr.Info("Please select a language!")
        return None

    try:
        progress(0.4, desc="Loading model")
        tokenizer = VitsTokenizer().VitsTokenizer.from_pretrained(model_path)
        model = VitsModel().VitsModel.from_pretrained(model_path)

        progress(0.6, desc="Generating speech")
        inputs = tokenizer(text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        waveform = outputs.waveform[0]

        progress(0.8, desc="Saving audio file")
        output_dir = os.path.join("outputs", "MMS_TTS")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"synthesized_speech_{language}{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}")

        if output_format == "wav":
            scipy.io.wavfile.write(output_file, rate=model.config.sampling_rate, data=waveform.numpy())
        elif output_format == "mp3":
            sf.write(output_file, waveform.numpy(), model.config.sampling_rate, format='mp3')
        elif output_format == "ogg":
            sf.write(output_file, waveform.numpy(), model.config.sampling_rate, format='ogg')
        else:
            gr.Info(f"Unsupported output format: {output_format}")
            return None

        progress(1.0, desc="Speech generation complete")
        return output_file

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None

    finally:
        if 'tokenizer' in locals():
            del tokenizer
        if 'model' in locals():
            del model
        flush()


def transcribe_mms_stt(audio_file, language, output_format, progress=gr.Progress()):
    model_path = os.path.join("inputs", "text", "mms", "speech2text")

    if not os.path.exists(model_path):
        progress(0.1, desc="Downloading MMS STT model...")
        gr.Info("Downloading MMS STT model...")
        os.makedirs(model_path, exist_ok=True)
        repo = Repo.clone_from("https://huggingface.co/facebook/mms-1b-all", model_path, no_checkout=True)
        repo.git.checkout("HEAD", "--", ".")
        gr.Info("MMS STT model downloaded")
        progress(0.3, desc="Model downloaded")

    if not audio_file:
        gr.Info("Please record your request!")
        return None

    if not language:
        gr.Info("Please select a language!")
        return None

    try:
        progress(0.4, desc="Loading model")
        processor = AutoProcessor().AutoProcessor.from_pretrained(model_path)
        model = Wav2Vec2ForCTC().Wav2Vec2ForCTC.from_pretrained(model_path)

        progress(0.5, desc="Preparing audio data")
        stream_data = load_dataset("mozilla-foundation/common_voice_17_0", language.lower(), split="test", streaming=True, trust_remote_code=True)
        stream_data = stream_data.cast_column("audio", Audio(sampling_rate=16000))

        progress(0.6, desc="Processing audio")
        audio, sr = librosa.load(audio_file, sr=16000)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

        progress(0.7, desc="Transcribing audio")
        with torch.no_grad():
            outputs = model(**inputs).logits

        ids = torch.argmax(outputs, dim=-1)[0]
        transcription = processor.decode(ids)

        progress(0.9, desc="Saving transcription")
        output_dir = os.path.join("outputs", "MMS_STT")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"transcription_{language}{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}")

        if output_format == "txt":
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(transcription)
        elif output_format == "json":
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({"transcription": transcription}, f, ensure_ascii=False, indent=4)
        else:
            gr.Info(f"Unsupported output format: {output_format}")
            return None

        progress(1.0, desc="Transcription complete")
        return transcription

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None

    finally:
        if 'processor' in locals():
            del processor
        if 'model' in locals():
            del model
        flush()


def seamless_m4tv2_process(input_type, input_text, input_audio, src_lang, tgt_lang, dataset_lang,
                           enable_speech_generation, speaker_id, text_num_beams,
                           enable_text_do_sample, enable_speech_do_sample,
                           speech_temperature, text_temperature, task_type,
                           text_output_format, audio_output_format, progress=gr.Progress()):

    MODEL_PATH = os.path.join("inputs", "text", "seamless-m4t-v2")

    if not os.path.exists(MODEL_PATH):
        progress(0.1, desc="Downloading SeamlessM4Tv2 model...")
        gr.Info("Downloading SeamlessM4Tv2 model...")
        os.makedirs(MODEL_PATH, exist_ok=True)
        Repo.clone_from("https://huggingface.co/facebook/seamless-m4t-v2-large", MODEL_PATH)
        gr.Info("SeamlessM4Tv2 model downloaded")
        progress(0.2, desc="Model downloaded")

    if not input_text and not input_audio:
        gr.Info("Please enter your request!")
        return None, None

    if not src_lang:
        gr.Info("Please select your source language!")
        return None, None

    if not tgt_lang:
        gr.Info("Please select your target language!")
        return None, None

    if not dataset_lang:
        gr.Info("Please select your dataset language!")
        return None, None

    try:
        progress(0.3, desc="Loading model")
        processor = AutoProcessor().AutoProcessor.from_pretrained(MODEL_PATH)

        progress(0.4, desc="Processing input")
        if input_type == "Text":
            inputs = processor(text=input_text, src_lang=get_languages()[src_lang], return_tensors="pt")
        elif input_type == "Audio" and input_audio:
            dataset = load_dataset("mozilla-foundation/common_voice_17_0", dataset_lang, split="test", streaming=True, trust_remote_code=True)
            audio_sample = next(iter(dataset))["audio"]
            inputs = processor(audios=audio_sample["array"], return_tensors="pt")

        generate_kwargs = {
            "tgt_lang": get_languages()[tgt_lang],
            "generate_speech": enable_speech_generation,
            "speaker_id": speaker_id
        }

        if enable_text_do_sample:
            generate_kwargs["text_do_sample"] = True
            generate_kwargs["text_temperature"] = text_temperature
        if enable_speech_do_sample:
            generate_kwargs["speech_do_sample"] = True
            generate_kwargs["speech_temperature"] = speech_temperature

        generate_kwargs["text_num_beams"] = text_num_beams

        progress(0.5, desc="Loading task-specific model")
        if task_type == "Speech to Speech":
            model = SeamlessM4Tv2ForSpeechToSpeech().SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(MODEL_PATH)
        elif task_type == "Text to Text":
            model = SeamlessM4Tv2ForTextToText().SeamlessM4Tv2ForTextToText.from_pretrained(MODEL_PATH)
        elif task_type == "Speech to Text":
            model = SeamlessM4Tv2ForSpeechToText().SeamlessM4Tv2ForSpeechToText.from_pretrained(MODEL_PATH)
        elif task_type == "Text to Speech":
            model = SeamlessM4Tv2ForTextToSpeech().SeamlessM4Tv2ForTextToSpeech.from_pretrained(MODEL_PATH)
        else:
            model = SeamlessM4Tv2Model().SeamlessM4Tv2Model.from_pretrained(MODEL_PATH)

        progress(0.7, desc="Generating output")
        outputs = model.generate(**inputs, **generate_kwargs)

        progress(0.8, desc="Processing output")
        if enable_speech_generation:
            audio_output = outputs[0].cpu().numpy().squeeze()
            audio_path = os.path.join('outputs', f"SeamlessM4T_{datetime.now().strftime('%Y%m%d')}",
                                      f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{audio_output_format}")
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)

            if audio_output_format == "wav":
                sf.write(audio_path, audio_output, 16000)
            elif audio_output_format == "mp3":
                sf.write(audio_path, audio_output, 16000, format='mp3')
            elif audio_output_format == "ogg":
                sf.write(audio_path, audio_output, 16000, format='ogg')
            else:
                gr.Info(f"Unsupported audio format: {audio_output_format}")
                return None, None
        else:
            audio_path = None

        if not enable_speech_generation:
            text_output = processor.decode(outputs[0].tolist()[0], skip_special_tokens=True)
            text_path = os.path.join('outputs', f"SeamlessM4T_{datetime.now().strftime('%Y%m%d')}",
                                     f"text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{text_output_format}")
            os.makedirs(os.path.dirname(text_path), exist_ok=True)

            if text_output_format == "txt":
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(text_output)
            elif text_output_format == "json":
                with open(text_path, "w", encoding="utf-8") as f:
                    json.dump({"text": text_output}, f, ensure_ascii=False, indent=4)
            else:
                gr.Info(f"Unsupported text format: {text_output_format}")
                return None, None
        else:
            text_output = None

        progress(1.0, desc="Process complete")
        return text_output, audio_path

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'processor' in locals():
            del processor
        if 'model' in locals():
            del model
        flush()


def translate_text(text, source_lang, target_lang, enable_translate_history, translate_history_format, file=None,
                   progress=gr.Progress()):
    if not text:
        gr.Info("Please enter your request!")
        return None

    if not source_lang:
        gr.Info("Please select your source language!")
        return None

    if not target_lang:
        gr.Info("Please select your target language!")
        return None

    try:
        progress(0.1, desc="Initializing translation")
        translator = LibreTranslateAPI("http://127.0.0.1:5000")
        if file:
            progress(0.2, desc="Reading file")
            with open(file.name, "r", encoding="utf-8") as f:
                text = f.read()

        progress(0.4, desc="Translating text")
        translation = translator.translate(text, source_lang, target_lang)

        if enable_translate_history:
            progress(0.6, desc="Saving translation history")
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

        progress(1.0, desc="Translation complete")
        return translation

    except urllib.error.URLError:
        gr.Info("LibreTranslate is not running. Please start the LibreTranslate server.")
        return None

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None

    finally:
        flush()


def generate_image_txt2img(prompt, negative_prompt, style_name, stable_diffusion_model_name, enable_quantize, vae_model_name, lora_model_names, lora_scales, textual_inversion_model_names, stable_diffusion_settings_html,
                           stable_diffusion_model_type, stable_diffusion_scheduler, stable_diffusion_steps,
                           stable_diffusion_cfg, stable_diffusion_width, stable_diffusion_height,
                           stable_diffusion_clip_skip, num_images_per_prompt, seed, stop_button,
                           enable_freeu, freeu_s1, freeu_s2, freeu_b1, freeu_b2,
                           enable_sag, sag_scale, enable_pag, pag_scale, enable_token_merging, ratio,
                           enable_deepcache, cache_interval, cache_branch_id, enable_tgate, gate_step,
                           enable_magicprompt, magicprompt_max_new_tokens, enable_cdvae, enable_taesd, enable_multidiffusion, circular_padding, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    if not stable_diffusion_model_name:
        gr.Info("Please, select a StableDiffusion model!")
        return None, None, None

    if style_name and style_name in styles:
        style_prompt = styles[style_name]['prompt']
        style_negative_prompt = styles[style_name]['negative_prompt']
        prompt = f"{prompt}, {style_prompt}" if prompt else style_prompt
        negative_prompt = f"{negative_prompt}, {style_negative_prompt}" if negative_prompt else style_negative_prompt

    if enable_quantize:

        if seed == "" or seed is None:
            seed = random.randint(0, 2 ** 32 - 1)
        else:
            seed = int(seed)

        try:
            params = {
                'model_name': stable_diffusion_model_name,
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'cfg_scale': stable_diffusion_cfg,
                'height': stable_diffusion_height,
                'width': stable_diffusion_width,
                'sample_steps': stable_diffusion_steps,
                'seed': seed
            }

            env = os.environ.copy()
            env['PYTHONPATH'] = os.pathsep.join(sys.path)

            result = subprocess.run(
                [sys.executable, 'inputs/image/sd_models/sd-txt2img-quantize.py', json.dumps(params)],
                capture_output=True, text=True, env=env
            )

            if result.returncode != 0:
                gr.Info(f"Error in sd-txt2img-quantize.py: {result.stderr}")
                return None, None, None

            image_paths = []
            output = result.stdout.strip()
            for part in output.split("IMAGE_PATH:"):
                if part:
                    image_path = part.strip().split('\n')[0]
                    if os.path.exists(image_path):
                        image_paths.append(image_path)

            return image_paths, None, f"Images generated successfully using quantized model. Seed used: {seed}"

        except Exception as e:
            gr.Error(f"An error occurred: {str(e)}")
            return None, None, None

    else:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            variant = "fp16" if torch_dtype == torch.float16 else "fp32"
            stable_diffusion_model_path = os.path.join("inputs", "image", "sd_models",
                                                       f"{stable_diffusion_model_name}")

            if not os.path.exists(stable_diffusion_model_path):
                gr.Info(f"StableDiffusion model not found: {stable_diffusion_model_path}")
                return None, None, None

            if enable_sag:
                stable_diffusion_model = StableDiffusionSAGPipeline().StableDiffusionSAGPipeline.from_single_file(
                    stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                    torch_dtype=torch_dtype, variant=variant
                )
            elif enable_pag:
                stable_diffusion_model = AutoPipelineForText2Image().AutoPipelineForText2Image.from_single_file(
                    stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                    torch_dtype=torch_dtype, variant=variant, enable_pag=True
                )
            elif enable_cdvae:
                vae = ConsistencyDecoderVAE().ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder", torch_dtype=torch_dtype)
                stable_diffusion_model = StableDiffusionPipeline().StableDiffusionPipeline.from_single_file(
                    stable_diffusion_model_path, vae=vae, use_safetensors=True, device_map="auto",
                    torch_dtype=torch_dtype, variant=variant
                )
            elif enable_taesd and stable_diffusion_model_type in ["SD", "SD2"]:
                vae_sd = AutoencoderTiny().AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch_dtype)
                stable_diffusion_model = StableDiffusionPipeline().StableDiffusionPipeline.from_single_file(
                    stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                    torch_dtype=torch_dtype, variant=variant, vae=vae_sd)
            elif enable_taesd and stable_diffusion_model_type == "SDXL":
                vae_xl = AutoencoderTiny().AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch_dtype)
                stable_diffusion_model = StableDiffusionXLPipeline().StableDiffusionXLPipeline.from_single_file(
                    stable_diffusion_model_path, use_safetensors=True, device_map="auto", attention_slice=1,
                    torch_dtype=torch_dtype, variant=variant, vae=vae_xl)
            elif enable_multidiffusion:
                stable_diffusion_model = StableDiffusionPanoramaPipeline().StableDiffusionPanoramaPipeline.from_single_file(
                    stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                    torch_dtype=torch_dtype, variant=variant)
            else:
                if stable_diffusion_model_type == "SD":
                    stable_diffusion_model = StableDiffusionPipeline().StableDiffusionPipeline.from_single_file(
                        stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                        torch_dtype=torch_dtype, variant=variant)
                elif stable_diffusion_model_type == "SD2":
                    stable_diffusion_model = StableDiffusionPipeline().StableDiffusionPipeline.from_single_file(
                        stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                        torch_dtype=torch_dtype, variant=variant)
                elif stable_diffusion_model_type == "SDXL":
                    stable_diffusion_model = StableDiffusionXLPipeline().StableDiffusionXLPipeline.from_single_file(
                        stable_diffusion_model_path, use_safetensors=True, device_map="auto", attention_slice=1,
                        torch_dtype=torch_dtype, variant=variant)
                else:
                    gr.Info("Invalid StableDiffusion model type!")
                    return None, None, None
        except (ValueError, KeyError):
            gr.Error("The selected model is not compatible with the chosen model type")
            return None, None, None

        if XFORMERS_AVAILABLE:
            stable_diffusion_model.enable_xformers_memory_efficient_attention(attention_op=None)
            stable_diffusion_model.vae.enable_xformers_memory_efficient_attention(attention_op=None)
            stable_diffusion_model.unet.enable_xformers_memory_efficient_attention(attention_op=None)

        stable_diffusion_model.to(device)
        stable_diffusion_model.text_encoder.to(device)
        stable_diffusion_model.vae.to(device)
        stable_diffusion_model.unet.to(device)

        try:
            if stable_diffusion_scheduler == "EulerDiscreteScheduler":
                stable_diffusion_model.scheduler = EulerDiscreteScheduler().EulerDiscreteScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "DPMSolverSinglestepScheduler":
                stable_diffusion_model.scheduler = DPMSolverSinglestepScheduler().DPMSolverSinglestepScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "DPMSolverMultistepScheduler":
                stable_diffusion_model.scheduler = DPMSolverMultistepScheduler().DPMSolverMultistepScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "EDMDPMSolverMultistepScheduler":
                stable_diffusion_model.scheduler = EDMDPMSolverMultistepScheduler().EDMDPMSolverMultistepScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "EDMEulerScheduler":
                stable_diffusion_model.scheduler = EDMEulerScheduler().EDMEulerScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "KDPM2DiscreteScheduler":
                stable_diffusion_model.scheduler = KDPM2DiscreteScheduler().KDPM2DiscreteScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "KDPM2AncestralDiscreteScheduler":
                stable_diffusion_model.scheduler = KDPM2AncestralDiscreteScheduler().KDPM2AncestralDiscreteScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "EulerAncestralDiscreteScheduler":
                stable_diffusion_model.scheduler = EulerAncestralDiscreteScheduler().EulerAncestralDiscreteScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "HeunDiscreteScheduler":
                stable_diffusion_model.scheduler = HeunDiscreteScheduler().HeunDiscreteScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "LMSDiscreteScheduler":
                stable_diffusion_model.scheduler = LMSDiscreteScheduler().LMSDiscreteScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "DEISMultistepScheduler":
                stable_diffusion_model.scheduler = DEISMultistepScheduler().DEISMultistepScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "UniPCMultistepScheduler":
                stable_diffusion_model.scheduler = UniPCMultistepScheduler().UniPCMultistepScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "LCMScheduler":
                stable_diffusion_model.scheduler = LCMScheduler().LCMScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "DPMSolverSDEScheduler":
                stable_diffusion_model.scheduler = DPMSolverSDEScheduler().DPMSolverSDEScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "TCDScheduler":
                stable_diffusion_model.scheduler = TCDScheduler().TCDScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "DDIMScheduler":
                stable_diffusion_model.scheduler = DDIMScheduler().DDIMScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "PNDMScheduler":
                stable_diffusion_model.scheduler = PNDMScheduler().PNDMScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "DDPMScheduler":
                stable_diffusion_model.scheduler = DDPMScheduler().DDPMScheduler.from_config(
                    stable_diffusion_model.scheduler.config)

            gr.Info(f"Scheduler successfully set to {stable_diffusion_scheduler}")
        except Exception as e:
            gr.Error(f"Error initializing scheduler: {e}")
            gr.Warning("Using default scheduler", duration=5)

        stable_diffusion_model.safety_checker = None

        stable_diffusion_model.enable_vae_slicing()
        stable_diffusion_model.enable_vae_tiling()
        stable_diffusion_model.enable_model_cpu_offload()

        if enable_freeu:
            stable_diffusion_model.enable_freeu(s1=freeu_s1, s2=freeu_s2, b1=freeu_b1, b2=freeu_b2)

        if enable_magicprompt:
            prompt = generate_magicprompt(prompt, magicprompt_max_new_tokens)

        if vae_model_name is not None:
            vae_model_path = os.path.join("inputs", "image", "sd_models", "vae", f"{vae_model_name}")
            if os.path.exists(vae_model_path):
                vae = AutoencoderKL().AutoencoderKL.from_single_file(vae_model_path, device_map="auto",
                                                                     torch_dtype=torch_dtype,
                                                                     variant=variant)
                stable_diffusion_model.vae = vae.to(device)

        if isinstance(lora_scales, str):
            lora_scales = [float(scale.strip()) for scale in lora_scales.split(',') if scale.strip()]
        elif isinstance(lora_scales, (int, float)):
            lora_scales = [float(lora_scales)]

        lora_loaded = False
        if lora_model_names and lora_scales:
            if len(lora_model_names) != len(lora_scales):
                gr.Warning(
                    f"Number of LoRA models ({len(lora_model_names)}) does not match number of scales ({len(lora_scales)}). Using available scales.", duration=5)

            for i, lora_model_name in enumerate(lora_model_names):
                if i < len(lora_scales):
                    lora_scale = lora_scales[i]
                else:
                    lora_scale = 1.0

                lora_model_path = os.path.join("inputs", "image", "sd_models", "lora", lora_model_name)
                if os.path.exists(lora_model_path):
                    adapter_name = os.path.splitext(os.path.basename(lora_model_name))[0]
                    try:
                        stable_diffusion_model.load_lora_weights(lora_model_path, adapter_name=adapter_name)
                        stable_diffusion_model.fuse_lora(lora_scale=lora_scale)
                        lora_loaded = True
                        gr.Info(f"Loaded LoRA {lora_model_name} with scale {lora_scale}")
                    except Exception as e:
                        gr.Warning(f"Error loading LoRA {lora_model_name}: {str(e)}", duration=5)

        ti_loaded = False
        if textual_inversion_model_names:
            for textual_inversion_model_name in textual_inversion_model_names:
                textual_inversion_model_path = os.path.join("inputs", "image", "sd_models", "embedding",
                                                            textual_inversion_model_name)
                if os.path.exists(textual_inversion_model_path):
                    try:
                        token = f"<{os.path.splitext(textual_inversion_model_name)[0]}>"
                        stable_diffusion_model.load_textual_inversion(textual_inversion_model_path, token=token)
                        ti_loaded = True
                        gr.Info(f"Loaded textual inversion: {token}")
                    except Exception as e:
                        gr.Warning(f"Error loading Textual Inversion {textual_inversion_model_name}: {str(e)}", duration=5)

        def process_prompt_with_ti(input_prompt, textual_inversion_model_names):
            if not textual_inversion_model_names:
                return input_prompt

            processed_prompt = input_prompt
            for ti_name in textual_inversion_model_names:
                base_name = os.path.splitext(ti_name)[0]
                token = f"<{base_name}>"
                if base_name in processed_prompt or token.lower() in processed_prompt or token.upper() in processed_prompt:
                    processed_prompt = processed_prompt.replace(base_name, token)
                    processed_prompt = processed_prompt.replace(token.lower(), token)
                    processed_prompt = processed_prompt.replace(token.upper(), token)
                    gr.Info(f"Applied Textual Inversion token: {token}")

            if processed_prompt != input_prompt:
                gr.Info(f"Prompt changed from '{input_prompt}' to '{processed_prompt}'")
            else:
                gr.Info("No Textual Inversion tokens applied to this prompt")

            return processed_prompt

        try:
            if seed == "" or seed is None:
                seed = random.randint(0, 2 ** 32 - 1)
            else:
                seed = int(seed)
            generator = torch.Generator(device).manual_seed(seed)

            processed_prompt = process_prompt_with_ti(prompt, textual_inversion_model_names)
            processed_negative_prompt = process_prompt_with_ti(negative_prompt, textual_inversion_model_names)

            if enable_token_merging:
                tomesd.apply_patch(stable_diffusion_model, ratio=ratio)

            if enable_deepcache:
                helper = DeepCacheSDHelper(pipe=stable_diffusion_model)
                helper.set_params(cache_interval=cache_interval, cache_branch_id=cache_branch_id)
                helper.enable()

            taesd_dec = taesd.Decoder().to(device).requires_grad_(False)

            if stable_diffusion_model_type in ["SD", "SD2"]:
                taesd_weights_path = "ThirdPartyRepository/taesd/taesd_decoder.pth"
            elif stable_diffusion_model_type == "SDXL":
                taesd_weights_path = "ThirdPartyRepository/taesd/taesdxl_decoder.pth"
            else:
                gr.Error(f"Unsupported model type: {stable_diffusion_model_type}")
                return None, None, None

            taesd_dec.load_state_dict(torch.load(taesd_weights_path, map_location=device))

            def get_pred_original_sample(sched, model_output, timestep, sample):
                device = model_output.device
                timestep = timestep.to(device)
                alpha_prod_t = sched.alphas_cumprod.to(device)[timestep.long()]
                return (sample - (1 - alpha_prod_t) ** 0.5 * model_output) / alpha_prod_t ** 0.5

            preview_images = []

            def combined_callback(stable_diffusion_model, i, t, callback_kwargs):
                nonlocal stop_idx
                if stop_signal and stop_idx is None:
                    stop_idx = i
                if i == stop_idx:
                    stable_diffusion_model._interrupt = True

                device = callback_kwargs["latents"].device
                t = torch.tensor(t).to(device)
                latents = get_pred_original_sample(stable_diffusion_model.scheduler, callback_kwargs["latents"], t,
                                                   callback_kwargs["latents"])
                decoded = \
                stable_diffusion_model.image_processor.postprocess(taesd_dec(latents.float()).mul_(2).sub_(1))[0]
                preview_images.append(decoded)

                progress((i + 1) / stable_diffusion_steps, f"Step {i + 1}/{stable_diffusion_steps}")

                return callback_kwargs

            if stable_diffusion_model_type == "SDXL":
                compel = Compel(
                    tokenizer=[stable_diffusion_model.tokenizer, stable_diffusion_model.tokenizer_2],
                    text_encoder=[stable_diffusion_model.text_encoder, stable_diffusion_model.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True]
                )
                prompt_embeds, pooled_prompt_embeds = compel(processed_prompt)
                negative_prompt = processed_negative_prompt
                images = stable_diffusion_model(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
                                                negative_prompt=negative_prompt,
                                                num_inference_steps=stable_diffusion_steps,
                                                guidance_scale=stable_diffusion_cfg, height=stable_diffusion_height,
                                                width=stable_diffusion_width, clip_skip=stable_diffusion_clip_skip,
                                                num_images_per_prompt=num_images_per_prompt,
                                                generator=generator, callback_on_step_end=combined_callback,
                                                callback_on_step_end_tensor_inputs=["latents"]).images
            elif enable_sag:
                images = stable_diffusion_model(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=stable_diffusion_steps,
                    guidance_scale=stable_diffusion_cfg,
                    height=stable_diffusion_height,
                    width=stable_diffusion_width,
                    clip_skip=stable_diffusion_clip_skip,
                    sag_scale=sag_scale,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                    callback_on_step_end=combined_callback,
                    callback_on_step_end_tensor_inputs=["latents"]
                ).images
            elif enable_pag:
                images = stable_diffusion_model(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=stable_diffusion_steps,
                    guidance_scale=stable_diffusion_cfg,
                    height=stable_diffusion_height,
                    width=stable_diffusion_width,
                    clip_skip=stable_diffusion_clip_skip,
                    pag_scale=pag_scale,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                    callback_on_step_end=combined_callback,
                    callback_on_step_end_tensor_inputs=["latents"]
                ).images
            elif enable_tgate and stable_diffusion_model_type == "SDXL":
                stable_diffusion_model = TgateSDXLLoader(
                    stable_diffusion_model,
                    gate_step=gate_step,
                    num_inference_steps=stable_diffusion_steps,
                ).to(device)
                images = stable_diffusion_model.tgate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=stable_diffusion_steps,
                    gate_step=gate_step,
                    guidance_scale=stable_diffusion_cfg,
                    height=stable_diffusion_height,
                    width=stable_diffusion_width,
                    clip_skip=stable_diffusion_clip_skip,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                    callback_on_step_end=combined_callback,
                    callback_on_step_end_tensor_inputs=["latents"]
                ).images
            elif enable_tgate and stable_diffusion_model_type == "SD":
                stable_diffusion_model = TgateSDLoader(
                    stable_diffusion_model,
                    gate_step=gate_step,
                    num_inference_steps=stable_diffusion_steps,
                ).to(device)
                images = stable_diffusion_model.tgate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=stable_diffusion_steps,
                    gate_step=gate_step,
                    guidance_scale=stable_diffusion_cfg,
                    height=stable_diffusion_height,
                    width=stable_diffusion_width,
                    clip_skip=stable_diffusion_clip_skip,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                    callback_on_step_end=combined_callback,
                    callback_on_step_end_tensor_inputs=["latents"]
                ).images
            elif enable_taesd or enable_cdvae and stable_diffusion_model_type in ["SD", "SD2"]:
                compel_proc = Compel(tokenizer=stable_diffusion_model.tokenizer,
                                     text_encoder=stable_diffusion_model.text_encoder)
                prompt_embeds = compel_proc(processed_prompt)
                negative_prompt_embeds = compel_proc(processed_negative_prompt)

                images = stable_diffusion_model(prompt_embeds=prompt_embeds,
                                                negative_prompt_embeds=negative_prompt_embeds,
                                                num_inference_steps=stable_diffusion_steps,
                                                guidance_scale=stable_diffusion_cfg, height=stable_diffusion_height,
                                                width=stable_diffusion_width, clip_skip=stable_diffusion_clip_skip,
                                                num_images_per_prompt=num_images_per_prompt,
                                                generator=generator, callback_on_step_end=combined_callback,
                                                callback_on_step_end_tensor_inputs=["latents"]).images
            elif enable_taesd and stable_diffusion_model_type == "SDXL":
                compel = Compel(
                    tokenizer=[stable_diffusion_model.tokenizer, stable_diffusion_model.tokenizer_2],
                    text_encoder=[stable_diffusion_model.text_encoder, stable_diffusion_model.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True]
                )
                prompt_embeds, pooled_prompt_embeds = compel(processed_prompt)
                negative_prompt = processed_negative_prompt
                images = stable_diffusion_model(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
                                                negative_prompt=negative_prompt,
                                                num_inference_steps=stable_diffusion_steps,
                                                guidance_scale=stable_diffusion_cfg, height=stable_diffusion_height,
                                                width=stable_diffusion_width, clip_skip=stable_diffusion_clip_skip,
                                                num_images_per_prompt=num_images_per_prompt,
                                                generator=generator, callback_on_step_end=combined_callback,
                                                callback_on_step_end_tensor_inputs=["latents"]).images
            elif enable_multidiffusion:
                images = stable_diffusion_model(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=stable_diffusion_steps,
                    guidance_scale=stable_diffusion_cfg,
                    height=stable_diffusion_height,
                    width=stable_diffusion_width,
                    clip_skip=stable_diffusion_clip_skip,
                    circular_padding=circular_padding,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                    callback_on_step_end=combined_callback,
                    callback_on_step_end_tensor_inputs=["latents"]
                ).images
            else:
                compel_proc = Compel(tokenizer=stable_diffusion_model.tokenizer,
                                     text_encoder=stable_diffusion_model.text_encoder)
                prompt_embeds = compel_proc(processed_prompt)
                negative_prompt_embeds = compel_proc(processed_negative_prompt)

                images = stable_diffusion_model(prompt_embeds=prompt_embeds,
                                                negative_prompt_embeds=negative_prompt_embeds,
                                                num_inference_steps=stable_diffusion_steps,
                                                guidance_scale=stable_diffusion_cfg, height=stable_diffusion_height,
                                                width=stable_diffusion_width, clip_skip=stable_diffusion_clip_skip,
                                                num_images_per_prompt=num_images_per_prompt,
                                                generator=generator, callback_on_step_end=combined_callback,
                                                callback_on_step_end_tensor_inputs=["latents"]).images

            image_paths = []
            for i, image in enumerate(images):
                today = datetime.now().date()
                image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
                os.makedirs(image_dir, exist_ok=True)
                image_filename = f"txt2img_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.{output_format}"
                image_path = os.path.join(image_dir, image_filename)

                metadata = {
                    "tab": "txt2img",
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": stable_diffusion_steps,
                    "guidance_scale": stable_diffusion_cfg,
                    "height": stable_diffusion_height,
                    "width": stable_diffusion_width,
                    "clip_skip": stable_diffusion_clip_skip,
                    "scheduler": stable_diffusion_scheduler,
                    "seed": seed,
                    "Stable_diffusion_model": stable_diffusion_model_name,
                    "Stable_diffusion_model_type": stable_diffusion_model_type,
                    "vae": vae_model_name if vae_model_name else None,
                    "lora_models": lora_model_names if lora_model_names else None,
                    "lora_scales": lora_scales if lora_model_names else None,
                    "textual_inversion": textual_inversion_model_names if textual_inversion_model_names else None
                }

                image.save(image_path, format=output_format.upper())
                add_metadata_to_file(image_path, metadata)
                image_paths.append(image_path)

            gif_filename = f"txt2img_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
            gif_path = os.path.join(image_dir, gif_filename)
            preview_images[0].save(gif_path, save_all=True, append_images=preview_images[1:], duration=100, loop=0)

            return image_paths, [gif_path] if gif_path else [], f"Images generated successfully. Seed used: {seed}"

        except Exception as e:
            gr.Error(f"An error occurred: {str(e)}")
            return None, None, None

        finally:
            if 'stable_diffusion_model' in locals():
                del stable_diffusion_model
            flush()


def generate_image_img2img(prompt, negative_prompt, init_image, strength, stable_diffusion_model_type,
                           stable_diffusion_model_name, enable_quantize, vae_model_name, lora_model_names, lora_scales, textual_inversion_model_names, seed, stop_button,
                           stable_diffusion_scheduler, stable_diffusion_steps, stable_diffusion_cfg,
                           stable_diffusion_clip_skip, num_images_per_prompt, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    if not stable_diffusion_model_name:
        gr.Info("Please, select a StableDiffusion model!")
        return None, None

    if not init_image:
        gr.Info("Please, upload an initial image!")
        return None, None

    if enable_quantize:
        try:
            if seed == "" or seed is None:
                seed = random.randint(0, 2 ** 32 - 1)
            else:
                seed = int(seed)

            params = {
                'model_name': stable_diffusion_model_name,
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'cfg_scale': stable_diffusion_cfg,
                'sample_steps': stable_diffusion_steps,
                'image': init_image,
                'strength': strength,
                'seed': seed
            }

            env = os.environ.copy()
            env['PYTHONPATH'] = os.pathsep.join(sys.path)

            result = subprocess.run(
                [sys.executable, 'inputs/image/sd_models/sd-img2img-quantize.py', json.dumps(params)],
                capture_output=True, text=True, env=env
            )

            if result.returncode != 0:
                gr.Info(f"Error in sd-img2img-quantize.py: {result.stderr}")
                return None, None

            image_paths = []
            output = result.stdout.strip()
            for part in output.split("IMAGE_PATH:"):
                if part:
                    image_path = part.strip().split('\n')[0]
                    if os.path.exists(image_path):
                        image_paths.append(image_path)

            return image_paths, f"Images generated successfully using quantized model. Seed used: {seed}"

        except Exception as e:
            gr.Error(f"An error occurred: {str(e)}")
            return None, None

    else:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            variant = "fp16" if torch_dtype == torch.float16 else "fp32"
            stable_diffusion_model_path = os.path.join("inputs", "image", "sd_models",
                                                       f"{stable_diffusion_model_name}")

            if not os.path.exists(stable_diffusion_model_path):
                gr.Info(f"StableDiffusion model not found: {stable_diffusion_model_path}")
                return None, None

            if stable_diffusion_model_type == "SD":
                stable_diffusion_model = StableDiffusionImg2ImgPipeline().StableDiffusionImg2ImgPipeline.from_single_file(
                    stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                    torch_dtype=torch_dtype, variant=variant)
            elif stable_diffusion_model_type == "SD2":
                stable_diffusion_model = StableDiffusionImg2ImgPipeline().StableDiffusionImg2ImgPipeline.from_single_file(
                    stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                    torch_dtype=torch_dtype, variant=variant)
            elif stable_diffusion_model_type == "SDXL":
                stable_diffusion_model = StableDiffusionXLImg2ImgPipeline().StableDiffusionXLImg2ImgPipeline.from_single_file(
                    stable_diffusion_model_path, use_safetensors=True, device_map="auto", attention_slice=1,
                    torch_dtype=torch_dtype, variant=variant)
            else:
                return None, "Invalid StableDiffusion model type!"
        except (ValueError, KeyError):
            gr.Error("The selected model is not compatible with the chosen model type")
            return None, None

        if XFORMERS_AVAILABLE:
            stable_diffusion_model.enable_xformers_memory_efficient_attention(attention_op=None)
            stable_diffusion_model.vae.enable_xformers_memory_efficient_attention(attention_op=None)
            stable_diffusion_model.unet.enable_xformers_memory_efficient_attention(attention_op=None)

        stable_diffusion_model.to(device)
        stable_diffusion_model.text_encoder.to(device)
        stable_diffusion_model.vae.to(device)
        stable_diffusion_model.unet.to(device)

        try:
            if stable_diffusion_scheduler == "EulerDiscreteScheduler":
                stable_diffusion_model.scheduler = EulerDiscreteScheduler().EulerDiscreteScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "DPMSolverSinglestepScheduler":
                stable_diffusion_model.scheduler = DPMSolverSinglestepScheduler().DPMSolverSinglestepScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "DPMSolverMultistepScheduler":
                stable_diffusion_model.scheduler = DPMSolverMultistepScheduler().DPMSolverMultistepScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "EDMDPMSolverMultistepScheduler":
                stable_diffusion_model.scheduler = EDMDPMSolverMultistepScheduler().EDMDPMSolverMultistepScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "EDMEulerScheduler":
                stable_diffusion_model.scheduler = EDMEulerScheduler().EDMEulerScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "KDPM2DiscreteScheduler":
                stable_diffusion_model.scheduler = KDPM2DiscreteScheduler().KDPM2DiscreteScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "KDPM2AncestralDiscreteScheduler":
                stable_diffusion_model.scheduler = KDPM2AncestralDiscreteScheduler().KDPM2AncestralDiscreteScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "EulerAncestralDiscreteScheduler":
                stable_diffusion_model.scheduler = EulerAncestralDiscreteScheduler().EulerAncestralDiscreteScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "HeunDiscreteScheduler":
                stable_diffusion_model.scheduler = HeunDiscreteScheduler().HeunDiscreteScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "LMSDiscreteScheduler":
                stable_diffusion_model.scheduler = LMSDiscreteScheduler().LMSDiscreteScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "DEISMultistepScheduler":
                stable_diffusion_model.scheduler = DEISMultistepScheduler().DEISMultistepScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "UniPCMultistepScheduler":
                stable_diffusion_model.scheduler = UniPCMultistepScheduler().UniPCMultistepScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "LCMScheduler":
                stable_diffusion_model.scheduler = LCMScheduler().LCMScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "DPMSolverSDEScheduler":
                stable_diffusion_model.scheduler = DPMSolverSDEScheduler().DPMSolverSDEScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "TCDScheduler":
                stable_diffusion_model.scheduler = TCDScheduler().TCDScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "DDIMScheduler":
                stable_diffusion_model.scheduler = DDIMScheduler().DDIMScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "PNDMScheduler":
                stable_diffusion_model.scheduler = PNDMScheduler().PNDMScheduler.from_config(
                    stable_diffusion_model.scheduler.config)
            elif stable_diffusion_scheduler == "DDPMScheduler":
                stable_diffusion_model.scheduler = DDPMScheduler().DDPMScheduler.from_config(
                    stable_diffusion_model.scheduler.config)

            gr.Info(f"Scheduler successfully set to {stable_diffusion_scheduler}")
        except Exception as e:
            gr.Error(f"Error initializing scheduler: {e}")
            gr.Warning("Using default scheduler", duration=5)

        stable_diffusion_model.safety_checker = None

        stable_diffusion_model.enable_vae_slicing()
        stable_diffusion_model.enable_vae_tiling()
        stable_diffusion_model.enable_model_cpu_offload()

        if vae_model_name is not None:
            vae_model_path = os.path.join("inputs", "image", "sd_models", "vae", f"{vae_model_name}")
            if os.path.exists(vae_model_path):
                vae = AutoencoderKL().AutoencoderKL.from_single_file(vae_model_path, device_map=device,
                                                                     torch_dtype=torch_dtype,
                                                                     variant=variant)
                stable_diffusion_model.vae = vae.to(device)

        if isinstance(lora_scales, str):
            lora_scales = [float(scale.strip()) for scale in lora_scales.split(',') if scale.strip()]
        elif isinstance(lora_scales, (int, float)):
            lora_scales = [float(lora_scales)]

        lora_loaded = False
        if lora_model_names and lora_scales:
            if len(lora_model_names) != len(lora_scales):
                gr.Warning(
                    f"Number of LoRA models ({len(lora_model_names)}) does not match number of scales ({len(lora_scales)}). Using available scales.", duration=5)

            for i, lora_model_name in enumerate(lora_model_names):
                if i < len(lora_scales):
                    lora_scale = lora_scales[i]
                else:
                    lora_scale = 1.0

                lora_model_path = os.path.join("inputs", "image", "sd_models", "lora", lora_model_name)
                if os.path.exists(lora_model_path):
                    adapter_name = os.path.splitext(os.path.basename(lora_model_name))[0]
                    try:
                        stable_diffusion_model.load_lora_weights(lora_model_path, adapter_name=adapter_name)
                        stable_diffusion_model.fuse_lora(lora_scale=lora_scale)
                        lora_loaded = True
                        gr.Info(f"Loaded LoRA {lora_model_name} with scale {lora_scale}")
                    except Exception as e:
                        gr.Warning(f"Error loading LoRA {lora_model_name}: {str(e)}", duration=5)

        ti_loaded = False
        if textual_inversion_model_names:
            for textual_inversion_model_name in textual_inversion_model_names:
                textual_inversion_model_path = os.path.join("inputs", "image", "sd_models", "embedding",
                                                            textual_inversion_model_name)
                if os.path.exists(textual_inversion_model_path):
                    try:
                        token = f"<{os.path.splitext(textual_inversion_model_name)[0]}>"
                        stable_diffusion_model.load_textual_inversion(textual_inversion_model_path, token=token)
                        ti_loaded = True
                        gr.Info(f"Loaded textual inversion: {token}")
                    except Exception as e:
                        gr.Warning(f"Error loading Textual Inversion {textual_inversion_model_name}: {str(e)}", duration=5)

        def process_prompt_with_ti(input_prompt, textual_inversion_model_names):
            if not textual_inversion_model_names:
                return input_prompt

            processed_prompt = input_prompt
            for ti_name in textual_inversion_model_names:
                base_name = os.path.splitext(ti_name)[0]
                token = f"<{base_name}>"
                if base_name in processed_prompt or token.lower() in processed_prompt or token.upper() in processed_prompt:
                    processed_prompt = processed_prompt.replace(base_name, token)
                    processed_prompt = processed_prompt.replace(token.lower(), token)
                    processed_prompt = processed_prompt.replace(token.upper(), token)
                    gr.Info(f"Applied Textual Inversion token: {token}")

            if processed_prompt != input_prompt:
                gr.Info(f"Prompt changed from '{input_prompt}' to '{processed_prompt}'")
            else:
                gr.Info("No Textual Inversion tokens applied to this prompt")

            return processed_prompt

        try:
            if seed == "" or seed is None:
                seed = random.randint(0, 2 ** 32 - 1)
            else:
                seed = int(seed)
            generator = torch.Generator(device).manual_seed(seed)

            init_image = Image.open(init_image).convert("RGB")
            init_image = stable_diffusion_model.image_processor.preprocess(init_image)

            processed_prompt = process_prompt_with_ti(prompt, textual_inversion_model_names)
            processed_negative_prompt = process_prompt_with_ti(negative_prompt, textual_inversion_model_names)

            def combined_callback(stable_diffusion_model, i, t, callback_kwargs):
                nonlocal stop_idx
                if stop_signal and stop_idx is None:
                    stop_idx = i
                if i == stop_idx:
                    stable_diffusion_model._interrupt = True

                progress((i + 1) / stable_diffusion_steps, f"Step {i + 1}/{stable_diffusion_steps}")

                return callback_kwargs

            if stable_diffusion_model_type == "SDXL":
                compel = Compel(
                    tokenizer=[stable_diffusion_model.tokenizer, stable_diffusion_model.tokenizer_2],
                    text_encoder=[stable_diffusion_model.text_encoder, stable_diffusion_model.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True]
                )
                prompt_embeds, pooled_prompt_embeds = compel(processed_prompt)
                negative_prompt = processed_negative_prompt

                images = stable_diffusion_model(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
                                                negative_prompt=negative_prompt,
                                                num_inference_steps=stable_diffusion_steps, generator=generator,
                                                guidance_scale=stable_diffusion_cfg,
                                                clip_skip=stable_diffusion_clip_skip,
                                                image=init_image, strength=strength,
                                                num_images_per_prompt=num_images_per_prompt,
                                                callback_on_step_end=combined_callback,
                                                callback_on_step_end_tensor_inputs=["latents"]).images
            else:
                compel_proc = Compel(tokenizer=stable_diffusion_model.tokenizer,
                                     text_encoder=stable_diffusion_model.text_encoder)
                prompt_embeds = compel_proc(processed_prompt)
                negative_prompt_embeds = compel_proc(processed_negative_prompt)

                images = stable_diffusion_model(prompt_embeds=prompt_embeds,
                                                negative_prompt_embeds=negative_prompt_embeds,
                                                num_inference_steps=stable_diffusion_steps, generator=generator,
                                                guidance_scale=stable_diffusion_cfg,
                                                clip_skip=stable_diffusion_clip_skip,
                                                image=init_image, strength=strength,
                                                num_images_per_prompt=num_images_per_prompt,
                                                callback_on_step_end=combined_callback,
                                                callback_on_step_end_tensor_inputs=["latents"]).images

            image_paths = []
            for i, image in enumerate(images):
                today = datetime.now().date()
                image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
                os.makedirs(image_dir, exist_ok=True)
                image_filename = f"img2img_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.{output_format}"
                image_path = os.path.join(image_dir, image_filename)
                metadata = {
                    "tab": "img2img",
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": stable_diffusion_steps,
                    "guidance_scale": stable_diffusion_cfg,
                    "clip_skip": stable_diffusion_clip_skip,
                    "scheduler": stable_diffusion_scheduler,
                    "strength": strength,
                    "seed": seed,
                    "Stable_diffusion_model": stable_diffusion_model_name,
                    "Stable_diffusion_model_type": stable_diffusion_model_type,
                    "vae": vae_model_name if vae_model_name else None,
                    "lora_models": lora_model_names if lora_model_names else None,
                    "lora_scales": lora_scales if lora_model_names else None,
                    "textual_inversion": textual_inversion_model_names if textual_inversion_model_names else None
                }

                image.save(image_path, format=output_format.upper())
                add_metadata_to_file(image_path, metadata)
                image_paths.append(image_path)

            return image_paths, f"Images generated successfully. Seed used: {seed}"

        except Exception as e:
            gr.Error(f"An error occurred: {str(e)}")
            return None, None

        finally:
            if 'stable_diffusion_model' in locals():
                del stable_diffusion_model
            flush()


def generate_image_depth2img(prompt, negative_prompt, init_image, seed, strength, clip_skip, num_images_per_prompt,
                             output_format):

    if not init_image:
        gr.Info("Please, upload an initial image!")
        return None, None

    stable_diffusion_model_path = os.path.join("inputs", "image", "sd_models", "depth")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    variant = "fp16" if torch_dtype == torch.float16 else "fp32"

    if not os.path.exists(stable_diffusion_model_path):
        gr.Info("Downloading depth2img model...")
        os.makedirs(stable_diffusion_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/stabilityai/stable-diffusion-2-depth", stable_diffusion_model_path)
        gr.Info("Depth2img model downloaded")

    try:
        stable_diffusion_model = StableDiffusionDepth2ImgPipeline().StableDiffusionDepth2ImgPipeline.from_pretrained(
            stable_diffusion_model_path, use_safetensors=True,
            torch_dtype=torch_dtype, variant=variant,
        )
    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    if XFORMERS_AVAILABLE:
        stable_diffusion_model.enable_xformers_memory_efficient_attention(attention_op=None)
        stable_diffusion_model.vae.enable_xformers_memory_efficient_attention(attention_op=None)
        stable_diffusion_model.unet.enable_xformers_memory_efficient_attention(attention_op=None)

    stable_diffusion_model.to(device)
    stable_diffusion_model.text_encoder.to(device)
    stable_diffusion_model.vae.to(device)
    stable_diffusion_model.unet.to(device)

    stable_diffusion_model.enable_vae_slicing()
    stable_diffusion_model.enable_vae_tiling()
    stable_diffusion_model.enable_model_cpu_offload()

    stable_diffusion_model.safety_checker = None

    try:
        init_image = Image.open(init_image).convert("RGB")

        if seed == "" or seed is None:
            seed = random.randint(0, 2 ** 32 - 1)
        else:
            seed = int(seed)
        generator = torch.Generator(device).manual_seed(seed)

        compel_proc = Compel(tokenizer=stable_diffusion_model.tokenizer,
                             text_encoder=stable_diffusion_model.text_encoder)
        prompt_embeds = compel_proc(prompt)
        negative_prompt_embeds = compel_proc(negative_prompt)

        images = stable_diffusion_model(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, image=init_image, strength=strength, num_images_per_prompt=num_images_per_prompt, clip_skip=clip_skip, generator=generator).images

        image_paths = []
        for i, image in enumerate(images):
            today = datetime.now().date()
            image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
            os.makedirs(image_dir, exist_ok=True)
            image_filename = f"depth2img_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.{output_format}"
            image_path = os.path.join(image_dir, image_filename)
            image.save(image_path, format=output_format.upper())
            image_paths.append(image_path)

        return image_paths, f"Images generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'stable_diffusion_model' in locals():
            del stable_diffusion_model
        flush()


def generate_image_marigold(input_image, num_inference_steps, ensemble_size):
    if not input_image:
        gr.Info("Please upload an image!")
        return None, None, None

    depth_model_path = os.path.join("inputs", "image", "marigold", "depth")
    normals_model_path = os.path.join("inputs", "image", "marigold", "normals")

    if not os.path.exists(depth_model_path):
        gr.Info("Downloading Marigold Depth model...")
        os.makedirs(depth_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/prs-eth/marigold-depth-v1-0", depth_model_path)
        gr.Info("Marigold Depth model downloaded")

    if not os.path.exists(normals_model_path):
        gr.Info("Downloading Marigold Normals model...")
        os.makedirs(normals_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/prs-eth/marigold-normals-v0-1", normals_model_path)
        gr.Info("Marigold Normals model downloaded")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        variant = "fp16" if torch_dtype == torch.float16 else "fp32"

        depth_pipe = diffusers().diffusers.MarigoldDepthPipeline.from_pretrained(
            depth_model_path, variant=variant, torch_dtype=torch_dtype
        ).to(device)

        normals_pipe = diffusers().diffusers.MarigoldNormalsPipeline.from_pretrained(
            normals_model_path, variant=variant, torch_dtype=torch_dtype
        ).to(device)

        image = load_image(input_image)

        depth = depth_pipe(image, num_inference_steps=num_inference_steps, ensemble_size=ensemble_size)
        depth_vis = depth_pipe.image_processor.visualize_depth(depth.prediction)

        normals = normals_pipe(image, num_inference_steps=num_inference_steps, ensemble_size=ensemble_size)
        normals_vis = normals_pipe.image_processor.visualize_normals(normals.prediction)

        today = datetime.now().date()
        output_dir = os.path.join('outputs', f"Marigold_{today.strftime('%Y%m%d')}")
        os.makedirs(output_dir, exist_ok=True)

        depth_filename = f"marigold_depth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        normals_filename = f"marigold_normals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        depth_path = os.path.join(output_dir, depth_filename)
        normals_path = os.path.join(output_dir, normals_filename)

        depth_vis[0].save(depth_path)
        normals_vis[0].save(normals_path)

        return depth_path, normals_path, "Images generated successfully"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None, None

    finally:
        if 'depth_pipe' in locals():
            del depth_pipe
        if 'normals_pipe' in locals():
            del normals_pipe
        flush()


def generate_image_pix2pix(prompt, negative_prompt, init_image, seed, num_inference_steps, guidance_scale,
                           clip_skip, num_images_per_prompt, output_format):

    if not init_image:
        gr.Info("Please, upload an initial image!")
        return None, None

    pix2pix_model_path = os.path.join("inputs", "image", "sd_models", "pix2pix")

    if not os.path.exists(pix2pix_model_path):
        gr.Info("Downloading Pix2Pix model...")
        os.makedirs(pix2pix_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/timbrooks/instruct-pix2pix", pix2pix_model_path)
        gr.Info("Pix2Pix model downloaded")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = StableDiffusionInstructPix2PixPipeline().StableDiffusionInstructPix2PixPipeline.from_pretrained(pix2pix_model_path, torch_dtype=torch_dtype, safety_checker=None)
        pipe.to(device)

        image = Image.open(init_image).convert("RGB")

        if XFORMERS_AVAILABLE:
            pipe.enable_xformers_memory_efficient_attention(attention_op=None)
            pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
            pipe.unet.enable_xformers_memory_efficient_attention(attention_op=None)

        pipe.to(device)
        pipe.text_encoder.to(device)
        pipe.vae.to(device)
        pipe.unet.to(device)

        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        pipe.enable_model_cpu_offload()

        pipe.safety_checker = None

        if seed == "" or seed is None:
            seed = random.randint(0, 2 ** 32 - 1)
        else:
            seed = int(seed)
        generator = torch.Generator(device).manual_seed(seed)

        compel_proc = Compel(tokenizer=pipe.tokenizer,
                             text_encoder=pipe.text_encoder)
        prompt_embeds = compel_proc(prompt)
        negative_prompt_embeds = compel_proc(negative_prompt)

        images = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                      image=image, clip_skip=clip_skip, num_inference_steps=num_inference_steps, image_guidance_scale=guidance_scale, num_images_per_prompt=num_images_per_prompt, generator=generator).images

        image_paths = []
        for i, image in enumerate(images):
            today = datetime.now().date()
            image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
            os.makedirs(image_dir, exist_ok=True)
            image_filename = f"pix2pix_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.{output_format}"
            image_path = os.path.join(image_dir, image_filename)
            image.save(image_path, format=output_format.upper())
            image_paths.append(image_path)

        return image_paths, f"Images generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        flush()


def generate_image_controlnet(prompt, negative_prompt, init_image, sd_version, stable_diffusion_scheduler, stable_diffusion_model_name, controlnet_model_name, seed, stop_button,
                              num_inference_steps, guidance_scale, width, height, controlnet_conditioning_scale, clip_skip, num_images_per_prompt, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    if not init_image:
        gr.Info("Please, upload an initial image!")
        return None, None, None

    if not stable_diffusion_model_name:
        gr.Info("Please, select a StableDiffusion model!")
        return None, None, None

    if not controlnet_model_name:
        gr.Info("Please, select a ControlNet model!")
        return None, None, None

    stable_diffusion_model_path = os.path.join("inputs", "image", "sd_models",
                                               f"{stable_diffusion_model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    if not os.path.exists(stable_diffusion_model_path):
        gr.Info(f"StableDiffusion model not found: {stable_diffusion_model_path}")
        return None, None, None

    controlnet_model_path = os.path.join("inputs", "image", "sd_models", "controlnet", controlnet_model_name)
    if not os.path.exists(controlnet_model_path):
        gr.Info(f"Downloading ControlNet {controlnet_model_name} model...")
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
        gr.Info(f"ControlNet {controlnet_model_name} model downloaded")

    try:

        if seed == "" or seed is None:
            seed = random.randint(0, 2 ** 32 - 1)
        else:
            seed = int(seed)
        generator = torch.Generator(device).manual_seed(seed)

        if sd_version == "SD":
            controlnet = ControlNetModel().ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch_dtype)
            pipe = StableDiffusionControlNetPipeline().StableDiffusionControlNetPipeline.from_single_file(
                stable_diffusion_model_path,
                controlnet=controlnet,
                torch_dtype=torch_dtype,
                device_map="auto",
                use_safetensors=True,
            )

            if XFORMERS_AVAILABLE:
                pipe.enable_xformers_memory_efficient_attention(attention_op=None)
                pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
                pipe.unet.enable_xformers_memory_efficient_attention(attention_op=None)

            pipe.to(device)
            pipe.text_encoder.to(device)
            pipe.vae.to(device)
            pipe.unet.to(device)

            try:
                if stable_diffusion_scheduler == "EulerDiscreteScheduler":
                    pipe.scheduler = EulerDiscreteScheduler().EulerDiscreteScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "DPMSolverSinglestepScheduler":
                    pipe.scheduler = DPMSolverSinglestepScheduler().DPMSolverSinglestepScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "DPMSolverMultistepScheduler":
                    pipe.scheduler = DPMSolverMultistepScheduler().DPMSolverMultistepScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "EDMDPMSolverMultistepScheduler":
                    pipe.scheduler = EDMDPMSolverMultistepScheduler().EDMDPMSolverMultistepScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "EDMEulerScheduler":
                    pipe.scheduler = EDMEulerScheduler().EDMEulerScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "KDPM2DiscreteScheduler":
                    pipe.scheduler = KDPM2DiscreteScheduler().KDPM2DiscreteScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "KDPM2AncestralDiscreteScheduler":
                    pipe.scheduler = KDPM2AncestralDiscreteScheduler().KDPM2AncestralDiscreteScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "EulerAncestralDiscreteScheduler":
                    pipe.scheduler = EulerAncestralDiscreteScheduler().EulerAncestralDiscreteScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "HeunDiscreteScheduler":
                    pipe.scheduler = HeunDiscreteScheduler().HeunDiscreteScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "LMSDiscreteScheduler":
                    pipe.scheduler = LMSDiscreteScheduler().LMSDiscreteScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "DEISMultistepScheduler":
                    pipe.scheduler = DEISMultistepScheduler().DEISMultistepScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "UniPCMultistepScheduler":
                    pipe.scheduler = UniPCMultistepScheduler().UniPCMultistepScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "LCMScheduler":
                    pipe.scheduler = LCMScheduler().LCMScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "DPMSolverSDEScheduler":
                    pipe.scheduler = DPMSolverSDEScheduler().DPMSolverSDEScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "TCDScheduler":
                    pipe.scheduler = TCDScheduler().TCDScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "DDIMScheduler":
                    pipe.scheduler = DDIMScheduler().DDIMScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "PNDMScheduler":
                    pipe.scheduler = PNDMScheduler().PNDMScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "DDPMScheduler":
                    pipe.scheduler = DDPMScheduler().DDPMScheduler.from_config(
                        pipe.scheduler.config)

                gr.Info(f"Scheduler successfully set to {stable_diffusion_scheduler}")
            except Exception as e:
                gr.Error(f"Error initializing scheduler: {e}")
                gr.Warning("Using default scheduler", duration=5)

            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
            pipe.enable_model_cpu_offload()

            pipe.safety_checker = None

            image = Image.open(init_image).convert("RGB")

            if controlnet_model_name == "openpose":
                processor = OpenposeDetector().OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
                control_image = processor(image, hand_and_face=True)
            elif controlnet_model_name == "depth":
                depth_estimator = pipeline().pipeline('depth-estimation')
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
                processor = LineartDetector().LineartDetector.from_pretrained("lllyasviel/Annotators")
                control_image = processor(image)
            elif controlnet_model_name == "scribble":
                processor = HEDdetector().HEDdetector.from_pretrained("lllyasviel/Annotators")
                control_image = processor(image, scribble=True)

            generator = torch.manual_seed(0)

            compel_proc = Compel(tokenizer=pipe.tokenizer,
                                 text_encoder=pipe.text_encoder)
            prompt_embeds = compel_proc(prompt)
            negative_prompt_embeds = compel_proc(negative_prompt)

            def combined_callback_sd(pipe, i, t, callback_kwargs):
                nonlocal stop_idx
                if stop_signal and stop_idx is None:
                    stop_idx = i
                if i == stop_idx:
                    pipe._interrupt = True
                progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")
                return callback_kwargs

            images = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                          num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, width=width,
                          height=height, clip_skip=clip_skip, generator=generator, image=control_image, num_images_per_prompt=num_images_per_prompt, callback_on_step_end=combined_callback_sd, callback_on_step_end_tensor_inputs=["latents"]).images

        else:
            controlnet = ControlNetModel().ControlNetModel.from_pretrained(
                controlnet_model_path,
                torch_dtype=torch_dtype,
                use_safetensors=True
            )
            pipe = StableDiffusionXLControlNetPipeline().StableDiffusionXLControlNetPipeline.from_single_file(
                stable_diffusion_model_path,
                controlnet=controlnet,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )

            if XFORMERS_AVAILABLE:
                pipe.enable_xformers_memory_efficient_attention(attention_op=None)
                pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
                pipe.unet.enable_xformers_memory_efficient_attention(attention_op=None)

            pipe.to(device)
            pipe.text_encoder.to(device)
            pipe.vae.to(device)
            pipe.unet.to(device)

            try:
                if stable_diffusion_scheduler == "EulerDiscreteScheduler":
                    pipe.scheduler = EulerDiscreteScheduler().EulerDiscreteScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "DPMSolverSinglestepScheduler":
                    pipe.scheduler = DPMSolverSinglestepScheduler().DPMSolverSinglestepScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "DPMSolverMultistepScheduler":
                    pipe.scheduler = DPMSolverMultistepScheduler().DPMSolverMultistepScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "EDMDPMSolverMultistepScheduler":
                    pipe.scheduler = EDMDPMSolverMultistepScheduler().EDMDPMSolverMultistepScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "EDMEulerScheduler":
                    pipe.scheduler = EDMEulerScheduler().EDMEulerScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "KDPM2DiscreteScheduler":
                    pipe.scheduler = KDPM2DiscreteScheduler().KDPM2DiscreteScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "KDPM2AncestralDiscreteScheduler":
                    pipe.scheduler = KDPM2AncestralDiscreteScheduler().KDPM2AncestralDiscreteScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "EulerAncestralDiscreteScheduler":
                    pipe.scheduler = EulerAncestralDiscreteScheduler().EulerAncestralDiscreteScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "HeunDiscreteScheduler":
                    pipe.scheduler = HeunDiscreteScheduler().HeunDiscreteScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "LMSDiscreteScheduler":
                    pipe.scheduler = LMSDiscreteScheduler().LMSDiscreteScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "DEISMultistepScheduler":
                    pipe.scheduler = DEISMultistepScheduler().DEISMultistepScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "UniPCMultistepScheduler":
                    pipe.scheduler = UniPCMultistepScheduler().UniPCMultistepScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "LCMScheduler":
                    pipe.scheduler = LCMScheduler().LCMScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "DPMSolverSDEScheduler":
                    pipe.scheduler = DPMSolverSDEScheduler().DPMSolverSDEScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "TCDScheduler":
                    pipe.scheduler = TCDScheduler().TCDScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "DDIMScheduler":
                    pipe.scheduler = DDIMScheduler().DDIMScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "PNDMScheduler":
                    pipe.scheduler = PNDMScheduler().PNDMScheduler.from_config(
                        pipe.scheduler.config)
                elif stable_diffusion_scheduler == "DDPMScheduler":
                    pipe.scheduler = DDPMScheduler().DDPMScheduler.from_config(
                        pipe.scheduler.config)

                gr.Info(f"Scheduler successfully set to {stable_diffusion_scheduler}")
            except Exception as e:
                gr.Error(f"Error initializing scheduler: {e}")
                gr.Warning("Using default scheduler", duration=5)

            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
            pipe.enable_model_cpu_offload()

            pipe.safety_checker = None

            image = Image.open(init_image).convert("RGB")

            if controlnet_model_name == "depth":
                depth_estimator = DPTForDepthEstimation().DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device)
                feature_extractor = DPTFeatureExtractor().DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

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

            compel = Compel(
                tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True]
            )
            prompt_embeds, pooled_prompt_embeds = compel(prompt)

            def combined_callback_sdxl(pipe, i, t, callback_kwargs):
                nonlocal stop_idx
                if stop_signal and stop_idx is None:
                    stop_idx = i
                if i == stop_idx:
                    pipe._interrupt = True
                progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")
                return callback_kwargs

            images = pipe(
                prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_prompt=negative_prompt, image=control_image,
                controlnet_conditioning_scale=controlnet_conditioning_scale, generator=generator,
                num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, clip_skip=clip_skip,
                width=width, height=height, num_images_per_prompt=num_images_per_prompt, callback_on_step_end=combined_callback_sdxl, callback_on_step_end_tensor_inputs=["latents"]).images

        image_paths = []
        control_image_paths = []
        for i, (image, control_image) in enumerate(zip(images, [control_image] * len(images))):
            today = datetime.now().date()
            image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
            os.makedirs(image_dir, exist_ok=True)

            image_filename = f"controlnet_{controlnet_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.{output_format}"
            image_path = os.path.join(image_dir, image_filename)
            image.save(image_path, format=output_format.upper())
            image_paths.append(image_path)

            control_image_filename = f"controlnet_{controlnet_model_name}_control_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.{output_format}"
            control_image_path = os.path.join(image_dir, control_image_filename)
            control_image.save(control_image_path, format=output_format.upper())
            control_image_paths.append(control_image_path)

        return image_paths, control_image_paths, f"Images generated successfully. Seed used: {seed}"

    except Exception as e:
        if UnboundLocalError:
            pass
        else:
            gr.Error(f"An error occurred: {str(e)}")
        return None, None, None

    finally:
        if 'controlnet' in locals():
            del controlnet
        if 'pipe' in locals():
            del pipe
        if 'depth_estimator' in locals():
            del depth_estimator
        if 'feature_extractor' in locals():
            del feature_extractor
        flush()


def generate_image_upscale_latent(prompt, image_path, upscale_factor, seed, num_inference_steps, guidance_scale, output_format):

    if not image_path:
        gr.Info("Please, upload an initial image!")
        return None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    try:
        if upscale_factor == "x2":
            model_id = "stabilityai/sd-x2-latent-upscaler"
            model_path = os.path.join("inputs", "image", "sd_models", "upscale", "x2-upscaler")
            if not os.path.exists(model_path):
                gr.Info(f"Downloading Upscale model: {model_id}")
                os.makedirs(model_path, exist_ok=True)
                Repo.clone_from(f"https://huggingface.co/{model_id}", model_path)
                gr.Info(f"Upscale model {model_id} downloaded")

            upscaler = StableDiffusionLatentUpscalePipeline().StableDiffusionLatentUpscalePipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype
            )
            if XFORMERS_AVAILABLE:
                upscaler.enable_xformers_memory_efficient_attention(attention_op=None)
                upscaler.vae.enable_xformers_memory_efficient_attention(attention_op=None)
                upscaler.unet.enable_xformers_memory_efficient_attention(attention_op=None)

            upscaler.to(device)
            upscaler.text_encoder.to(device)
            upscaler.vae.to(device)
            upscaler.unet.to(device)

            upscaler.enable_vae_slicing()
            upscaler.enable_vae_tiling()
            upscaler.enable_model_cpu_offload()

            upscaler.safety_checker = None

            init_image = Image.open(image_path).convert("RGB")
            init_image = init_image.resize((512, 512))

            low_res_latents = upscaler(prompt=prompt, image=init_image, output_type="latent",
                                       generator=generator).images

            upscaled_image = upscaler(
                prompt=prompt,
                image=low_res_latents,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]

        else:
            model_id = "stabilityai/stable-diffusion-x4-upscaler"
            model_path = os.path.join("inputs", "image", "sd_models", "upscale", "x4-upscaler")
            if not os.path.exists(model_path):
                gr.Info(f"Downloading Upscale model: {model_id}")
                os.makedirs(model_path, exist_ok=True)
                Repo.clone_from(f"https://huggingface.co/{model_id}", model_path)
                gr.Info(f"Upscale model {model_id} downloaded")

            upscaler = StableDiffusionUpscalePipeline().StableDiffusionUpscalePipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype
            )
            if XFORMERS_AVAILABLE:
                upscaler.enable_xformers_memory_efficient_attention(attention_op=None)
                upscaler.vae.enable_xformers_memory_efficient_attention(attention_op=None)
                upscaler.unet.enable_xformers_memory_efficient_attention(attention_op=None)

            upscaler.to(device)
            upscaler.text_encoder.to(device)
            upscaler.vae.to(device)
            upscaler.unet.to(device)

            upscaler.safety_checker = None

            low_res_img = Image.open(image_path).convert("RGB")
            low_res_img = low_res_img.resize((128, 128))

            upscaled_image = upscaler(
                prompt=prompt,
                image=low_res_img,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"upscaled_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        upscaled_image.save(image_path, format=output_format.upper())

        return image_path, None

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'upscaler' in locals():
            del upscaler
        flush()


def generate_image_sdxl_refiner(prompt, init_image, output_format):

    if not init_image:
        gr.Info("Please upload an initial image!")
        return None, None

    sdxl_refiner_path = os.path.join("inputs", "image", "sd_models", "sdxl-refiner-1.0")

    if not os.path.exists(sdxl_refiner_path):
        gr.Info("Downloading SDXL Refiner model...")
        os.makedirs(sdxl_refiner_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0", sdxl_refiner_path)
        gr.Info("SDXL Refiner model downloaded")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        variant = "fp16" if torch_dtype == torch.float16 else "fp32"

        pipe = StableDiffusionXLImg2ImgPipeline().StableDiffusionXLImg2ImgPipeline.from_pretrained(
            sdxl_refiner_path, torch_dtype=torch_dtype, variant=variant, use_safetensors=True
        )
        pipe = pipe.to(device)

        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        pipe.enable_model_cpu_offload()

        init_image = Image.open(init_image).convert("RGB")
        image = pipe(prompt, image=init_image).images[0]

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"sdxl_refiner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        return image_path, None

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        flush()


def generate_image_inpaint(prompt, negative_prompt, init_image, mask_image, blur_factor, stable_diffusion_model_type, stable_diffusion_scheduler, stable_diffusion_model_name, vae_model_name, seed, stop_button,
                           stable_diffusion_steps, stable_diffusion_cfg, width, height, clip_skip, num_images_per_prompt, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    if not stable_diffusion_model_name:
        gr.Info("Please, select a StableDiffusion model!")
        return None, None

    if not init_image or not mask_image:
        gr.Info("Please, upload an initial image and a mask image!")
        return None, None

    stable_diffusion_model_path = os.path.join("inputs", "image", "sd_models", "inpaint",
                                               f"{stable_diffusion_model_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    variant = "fp16" if torch_dtype == torch.float16 else "fp32"

    if not os.path.exists(stable_diffusion_model_path):
        gr.Info(f"StableDiffusion model not found: {stable_diffusion_model_path}")
        return None, None

    try:
        if stable_diffusion_model_type == "SD":
            stable_diffusion_model = StableDiffusionInpaintPipeline().StableDiffusionInpaintPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                torch_dtype=torch_dtype, variant=variant
            )
        elif stable_diffusion_model_type == "SD2":
            stable_diffusion_model = StableDiffusionInpaintPipeline().StableDiffusionInpaintPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                torch_dtype=torch_dtype, variant=variant
            )
        elif stable_diffusion_model_type == "SDXL":
            stable_diffusion_model = StableDiffusionXLInpaintPipeline().StableDiffusionXLInpaintPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto", attention_slice=1,
                torch_dtype=torch_dtype, variant=variant
            )
        else:
            gr.Info("Invalid StableDiffusion model type!")
            return None, None
    except (ValueError, KeyError):
        gr.Error("The selected model is not compatible with the chosen model type")
        return None, None

    if XFORMERS_AVAILABLE:
        stable_diffusion_model.enable_xformers_memory_efficient_attention(attention_op=None)
        stable_diffusion_model.vae.enable_xformers_memory_efficient_attention(attention_op=None)
        stable_diffusion_model.unet.enable_xformers_memory_efficient_attention(attention_op=None)

    stable_diffusion_model.to(device)
    stable_diffusion_model.text_encoder.to(device)
    stable_diffusion_model.vae.to(device)
    stable_diffusion_model.unet.to(device)

    try:
        if stable_diffusion_scheduler == "EulerDiscreteScheduler":
            stable_diffusion_model.scheduler = EulerDiscreteScheduler().EulerDiscreteScheduler.from_config(
                stable_diffusion_model.scheduler.config)
        elif stable_diffusion_scheduler == "DPMSolverSinglestepScheduler":
            stable_diffusion_model.scheduler = DPMSolverSinglestepScheduler().DPMSolverSinglestepScheduler.from_config(
                stable_diffusion_model.scheduler.config)
        elif stable_diffusion_scheduler == "DPMSolverMultistepScheduler":
            stable_diffusion_model.scheduler = DPMSolverMultistepScheduler().DPMSolverMultistepScheduler.from_config(
                stable_diffusion_model.scheduler.config)
        elif stable_diffusion_scheduler == "EDMDPMSolverMultistepScheduler":
            stable_diffusion_model.scheduler = EDMDPMSolverMultistepScheduler().EDMDPMSolverMultistepScheduler.from_config(
                stable_diffusion_model.scheduler.config)
        elif stable_diffusion_scheduler == "EDMEulerScheduler":
            stable_diffusion_model.scheduler = EDMEulerScheduler().EDMEulerScheduler.from_config(
                stable_diffusion_model.scheduler.config)
        elif stable_diffusion_scheduler == "KDPM2DiscreteScheduler":
            stable_diffusion_model.scheduler = KDPM2DiscreteScheduler().KDPM2DiscreteScheduler.from_config(
                stable_diffusion_model.scheduler.config)
        elif stable_diffusion_scheduler == "KDPM2AncestralDiscreteScheduler":
            stable_diffusion_model.scheduler = KDPM2AncestralDiscreteScheduler().KDPM2AncestralDiscreteScheduler.from_config(
                stable_diffusion_model.scheduler.config)
        elif stable_diffusion_scheduler == "EulerAncestralDiscreteScheduler":
            stable_diffusion_model.scheduler = EulerAncestralDiscreteScheduler().EulerAncestralDiscreteScheduler.from_config(
                stable_diffusion_model.scheduler.config)
        elif stable_diffusion_scheduler == "HeunDiscreteScheduler":
            stable_diffusion_model.scheduler = HeunDiscreteScheduler().HeunDiscreteScheduler.from_config(
                stable_diffusion_model.scheduler.config)
        elif stable_diffusion_scheduler == "LMSDiscreteScheduler":
            stable_diffusion_model.scheduler = LMSDiscreteScheduler().LMSDiscreteScheduler.from_config(
                stable_diffusion_model.scheduler.config)
        elif stable_diffusion_scheduler == "DEISMultistepScheduler":
            stable_diffusion_model.scheduler = DEISMultistepScheduler().DEISMultistepScheduler.from_config(
                stable_diffusion_model.scheduler.config)
        elif stable_diffusion_scheduler == "UniPCMultistepScheduler":
            stable_diffusion_model.scheduler = UniPCMultistepScheduler().UniPCMultistepScheduler.from_config(
                stable_diffusion_model.scheduler.config)
        elif stable_diffusion_scheduler == "LCMScheduler":
            stable_diffusion_model.scheduler = LCMScheduler().LCMScheduler.from_config(
                stable_diffusion_model.scheduler.config)
        elif stable_diffusion_scheduler == "DPMSolverSDEScheduler":
            stable_diffusion_model.scheduler = DPMSolverSDEScheduler().DPMSolverSDEScheduler.from_config(
                stable_diffusion_model.scheduler.config)
        elif stable_diffusion_scheduler == "TCDScheduler":
            stable_diffusion_model.scheduler = TCDScheduler().TCDScheduler.from_config(
                stable_diffusion_model.scheduler.config)
        elif stable_diffusion_scheduler == "DDIMScheduler":
            stable_diffusion_model.scheduler = DDIMScheduler().DDIMScheduler.from_config(
                stable_diffusion_model.scheduler.config)
        elif stable_diffusion_scheduler == "PNDMScheduler":
            stable_diffusion_model.scheduler = PNDMScheduler().PNDMScheduler.from_config(
                stable_diffusion_model.scheduler.config)
        elif stable_diffusion_scheduler == "DDPMScheduler":
            stable_diffusion_model.scheduler = DDPMScheduler().DDPMScheduler.from_config(
                stable_diffusion_model.scheduler.config)

        gr.Info(f"Scheduler successfully set to {stable_diffusion_scheduler}")
    except Exception as e:
        gr.Error(f"Error initializing scheduler: {e}")
        gr.Warning("Using default scheduler", duration=5)

    stable_diffusion_model.enable_vae_slicing()
    stable_diffusion_model.enable_vae_tiling()
    stable_diffusion_model.enable_model_cpu_offload()

    stable_diffusion_model.safety_checker = None

    if vae_model_name is not None:
        vae_model_path = os.path.join("inputs", "image", "sd_models", "vae", f"{vae_model_name}")
        if os.path.exists(vae_model_path):
            vae = AutoencoderKL().AutoencoderKL.from_single_file(vae_model_path, device_map="auto",
                                                 torch_dtype=torch_dtype,
                                                 variant=variant)
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

        if seed == "" or seed is None:
            seed = random.randint(0, 2 ** 32 - 1)
        else:
            seed = int(seed)
        generator = torch.Generator(device).manual_seed(seed)

        def combined_callback(stable_diffusion_model, i, t, callback_kwargs):
            nonlocal stop_idx
            if stop_signal and stop_idx is None:
                stop_idx = i
            if i == stop_idx:
                stable_diffusion_model._interrupt = True
            progress((i + 1) / stable_diffusion_steps, f"Step {i + 1}/{stable_diffusion_steps}")
            return callback_kwargs

        if stable_diffusion_model_type == "SDXL":
            compel = Compel(
                tokenizer=[stable_diffusion_model.tokenizer, stable_diffusion_model.tokenizer_2],
                text_encoder=[stable_diffusion_model.text_encoder, stable_diffusion_model.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True]
            )
            prompt_embeds, pooled_prompt_embeds = compel(prompt)

            images = stable_diffusion_model(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_prompt=negative_prompt,
                                            image=init_image, generator=generator, clip_skip=clip_skip,
                                            mask_image=blurred_mask, width=width, height=height,
                                            num_inference_steps=stable_diffusion_steps,
                                            guidance_scale=stable_diffusion_cfg, num_images_per_prompt=num_images_per_prompt, callback_on_step_end=combined_callback, callback_on_step_end_tensor_inputs=["latents"]).images
        else:
            compel_proc = Compel(tokenizer=stable_diffusion_model.tokenizer,
                                 text_encoder=stable_diffusion_model.text_encoder)
            prompt_embeds = compel_proc(prompt)
            negative_prompt_embeds = compel_proc(negative_prompt)

            images = stable_diffusion_model(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                                            image=init_image, generator=generator, clip_skip=clip_skip,
                                            mask_image=blurred_mask, width=width, height=height,
                                            num_inference_steps=stable_diffusion_steps,
                                            guidance_scale=stable_diffusion_cfg, num_images_per_prompt=num_images_per_prompt, callback_on_step_end=combined_callback, callback_on_step_end_tensor_inputs=["latents"]).images

        image_paths = []
        for i, image in enumerate(images):
            today = datetime.now().date()
            image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
            os.makedirs(image_dir, exist_ok=True)
            image_filename = f"inpaint_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.{output_format}"
            image_path = os.path.join(image_dir, image_filename)
            image.save(image_path, format=output_format.upper())
            image_paths.append(image_path)

        return image_paths, f"Images generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'stable_diffusion_model' in locals():
            del stable_diffusion_model
        flush()


def generate_image_outpaint(prompt, negative_prompt, init_image, stable_diffusion_model_type, stable_diffusion_model_name, seed,
                            stable_diffusion_steps, stable_diffusion_cfg, outpaint_direction, outpaint_expansion, clip_skip, num_images_per_prompt, output_format):

    if not init_image:
        gr.Info("Please upload an initial image!")
        return None, None

    if not stable_diffusion_model_name:
        gr.Info("Please select a StableDiffusion model!")
        return None, None

    stable_diffusion_model_path = os.path.join("inputs", "image", "sd_models", "inpaint",
                                               f"{stable_diffusion_model_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    variant = "fp16" if torch_dtype == torch.float16 else "fp32"

    if not os.path.exists(stable_diffusion_model_path):
        gr.Info(f"StableDiffusion model not found: {stable_diffusion_model_path}")
        return None, None

    try:
        if stable_diffusion_model_type == "SD":
            pipe = StableDiffusionInpaintPipeline().StableDiffusionInpaintPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                torch_dtype=torch_dtype, variant=variant
            )
        elif stable_diffusion_model_type == "SD2":
            pipe = StableDiffusionInpaintPipeline().StableDiffusionInpaintPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                torch_dtype=torch_dtype, variant=variant
            )
        elif stable_diffusion_model_type == "SDXL":
            pipe = StableDiffusionXLInpaintPipeline().StableDiffusionXLInpaintPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto", attention_slice=1,
                torch_dtype=torch_dtype, variant=variant
            )
        else:
            gr.Info("Invalid StableDiffusion model type!")
            return None, None

        if XFORMERS_AVAILABLE:
            pipe.enable_xformers_memory_efficient_attention(attention_op=None)
            pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
            pipe.unet.enable_xformers_memory_efficient_attention(attention_op=None)

        pipe.to(device)
        pipe.text_encoder.to(device)
        pipe.vae.to(device)
        pipe.unet.to(device)

        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        pipe.enable_model_cpu_offload()

        pipe.safety_checker = None

        if seed == "" or seed is None:
            seed = random.randint(0, 2 ** 32 - 1)
        else:
            seed = int(seed)
        generator = torch.Generator(device).manual_seed(seed)

        init_image = Image.open(init_image).convert("RGB")
        init_width, init_height = init_image.size

        if outpaint_direction in ['left', 'right']:
            new_width = int(init_width * (1 + outpaint_expansion / 100))
            new_height = init_height
        else:
            new_width = init_width
            new_height = int(init_height * (1 + outpaint_expansion / 100))

        new_image = Image.new('RGB', (new_width, new_height), (0, 0, 0))

        if outpaint_direction == 'left':
            paste_position = (new_width - init_width, 0)
        elif outpaint_direction == 'right':
            paste_position = (0, 0)
        elif outpaint_direction == 'up':
            paste_position = (0, new_height - init_height)
        else:
            paste_position = (0, 0)

        new_image.paste(init_image, paste_position)

        mask = Image.new('L', (new_width, new_height), 0)
        mask_draw = ImageDraw.Draw(mask)
        if outpaint_direction == 'left':
            mask_draw.rectangle([0, 0, new_width - init_width, new_height], fill=255)
        elif outpaint_direction == 'right':
            mask_draw.rectangle([init_width, 0, new_width, new_height], fill=255)
        elif outpaint_direction == 'up':
            mask_draw.rectangle([0, 0, new_width, new_height - init_height], fill=255)
        else:
            mask_draw.rectangle([0, init_height, new_width, new_height], fill=255)

        if stable_diffusion_model_type == "SDXL":
            compel = Compel(
                tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True]
            )
            prompt_embeds, pooled_prompt_embeds = compel(prompt)

            results = pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt=negative_prompt,
                image=new_image,
                mask_image=mask,
                num_inference_steps=stable_diffusion_steps,
                guidance_scale=stable_diffusion_cfg,
                width=new_width,
                height=new_height,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                generator=generator,
            ).images
        else:
            compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
            prompt_embeds = compel_proc(prompt)
            negative_prompt_embeds = compel_proc(negative_prompt)

            results = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                image=new_image,
                mask_image=mask,
                num_inference_steps=stable_diffusion_steps,
                guidance_scale=stable_diffusion_cfg,
                width=new_width,
                height=new_height,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                generator=generator,
            ).images

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)

        image_paths = []
        for i, result in enumerate(results):
            image_filename = f"outpaint_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.{output_format}"
            image_path = os.path.join(image_dir, image_filename)
            result.save(image_path, format=output_format.upper())
            image_paths.append(image_path)

        return image_paths, f"Images generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        flush()


def generate_image_gligen(prompt, negative_prompt, gligen_phrases, gligen_boxes, stable_diffusion_model_type, stable_diffusion_model_name, seed, stable_diffusion_steps,
                          stable_diffusion_cfg, stable_diffusion_width, stable_diffusion_height,
                          stable_diffusion_clip_skip, num_images_per_prompt, output_format):

    if not stable_diffusion_model_name:
        gr.Info("Please, select a StableDiffusion model!")
        return None, None

    stable_diffusion_model_path = os.path.join("inputs", "image", "sd_models",
                                               f"{stable_diffusion_model_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    variant = "fp16" if torch_dtype == torch.float16 else "fp32"

    if not os.path.exists(stable_diffusion_model_path):
        gr.Info(f"StableDiffusion model not found: {stable_diffusion_model_path}")
        return None, None

    try:
        if stable_diffusion_model_type == "SD":
            stable_diffusion_model = StableDiffusionPipeline().StableDiffusionPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                torch_dtype=torch_dtype, variant=variant
            )
        elif stable_diffusion_model_type == "SD2":
            stable_diffusion_model = StableDiffusionPipeline().StableDiffusionPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                torch_dtype=torch_dtype, variant=variant
            )
        elif stable_diffusion_model_type == "SDXL":
            stable_diffusion_model = StableDiffusionXLPipeline().StableDiffusionXLPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto", attention_slice=1,
                torch_dtype=torch_dtype, variant=variant
            )
        else:
            gr.Info("Invalid StableDiffusion model type!")
            return None, None
    except (ValueError, KeyError):
        gr.Error("The selected model is not compatible with the chosen model type")
        return None, None

    if XFORMERS_AVAILABLE:
        stable_diffusion_model.enable_xformers_memory_efficient_attention(attention_op=None)
        stable_diffusion_model.vae.enable_xformers_memory_efficient_attention(attention_op=None)
        stable_diffusion_model.unet.enable_xformers_memory_efficient_attention(attention_op=None)

    stable_diffusion_model.to(device)
    stable_diffusion_model.text_encoder.to(device)
    stable_diffusion_model.vae.to(device)
    stable_diffusion_model.unet.to(device)

    stable_diffusion_model.enable_vae_slicing()
    stable_diffusion_model.enable_vae_tiling()
    stable_diffusion_model.enable_model_cpu_offload()

    stable_diffusion_model.safety_checker = None

    try:
        if seed == "" or seed is None:
            seed = random.randint(0, 2 ** 32 - 1)
        else:
            seed = int(seed)
        generator = torch.Generator(device).manual_seed(seed)

        if stable_diffusion_model_type == "SDXL":
            compel = Compel(
                tokenizer=[stable_diffusion_model.tokenizer, stable_diffusion_model.tokenizer_2],
                text_encoder=[stable_diffusion_model.text_encoder, stable_diffusion_model.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True]
            )
            prompt_embeds, pooled_prompt_embeds = compel(prompt)

            images = stable_diffusion_model(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_prompt=negative_prompt,
                                           num_inference_steps=stable_diffusion_steps, generator=generator,
                                           guidance_scale=stable_diffusion_cfg, height=stable_diffusion_height,
                                           width=stable_diffusion_width, clip_skip=stable_diffusion_clip_skip,
                                           num_images_per_prompt=num_images_per_prompt).images
        else:
            compel_proc = Compel(tokenizer=stable_diffusion_model.tokenizer,
                                 text_encoder=stable_diffusion_model.text_encoder)
            prompt_embeds = compel_proc(prompt)
            negative_prompt_embeds = compel_proc(negative_prompt)

            images = stable_diffusion_model(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                                           num_inference_steps=stable_diffusion_steps, generator=generator,
                                           guidance_scale=stable_diffusion_cfg, height=stable_diffusion_height,
                                           width=stable_diffusion_width, clip_skip=stable_diffusion_clip_skip,
                                           num_images_per_prompt=num_images_per_prompt).images

        gligen_model_path = os.path.join("inputs", "image", "sd_models", "gligen")

        if not os.path.exists(gligen_model_path):
            gr.Info("Downloading GLIGEN model...")
            os.makedirs(gligen_model_path, exist_ok=True)
            Repo.clone_from("https://huggingface.co/masterful/gligen-1-4-inpainting-text-box", os.path.join(gligen_model_path, "inpainting"))
            gr.Info("GLIGEN model downloaded")

        gligen_boxes = json.loads(gligen_boxes)

        pipe = StableDiffusionGLIGENPipeline().StableDiffusionGLIGENPipeline.from_pretrained(
            os.path.join(gligen_model_path, "inpainting"), variant=variant, torch_dtype=torch_dtype
        )

        pipe.to(device)
        pipe.text_encoder.to(device)
        pipe.vae.to(device)
        pipe.unet.to(device)

        pipe.safety_checker = None

        compel_proc = Compel(tokenizer=pipe.tokenizer,
                             text_encoder=pipe.text_encoder)
        prompt_embeds = compel_proc(prompt)
        negative_prompt_embeds = compel_proc(negative_prompt)

        images = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            gligen_phrases=gligen_phrases,
            gligen_inpaint_image=images,
            gligen_boxes=[gligen_boxes],
            gligen_scheduled_sampling_beta=1,
            output_type="pil",
            num_inference_steps=stable_diffusion_steps,
        ).images

        image_paths = []
        for i, image in enumerate(images):
            today = datetime.now().date()
            image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
            os.makedirs(image_dir, exist_ok=True)
            image_filename = f"gligen_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.{output_format}"
            image_path = os.path.join(image_dir, image_filename)
            image.save(image_path, format=output_format.upper())
            image_paths.append(image_path)

        return image_paths, f"Images generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'stable_diffusion_model' in locals():
            del stable_diffusion_model
        if 'pipe' in locals():
            del pipe
        flush()


def generate_image_diffedit(source_prompt, source_negative_prompt, target_prompt, target_negative_prompt, init_image, seed, num_inference_steps, guidance_scale, output_format):
    if not init_image:
        gr.Info("Please upload an initial image!")
        return None, None

    sd2_1_model_path = os.path.join("inputs", "image", "sd_models", "sd2-1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    if not os.path.exists(sd2_1_model_path):
        gr.Info("Downloading Stable Diffusion 2.1 model...")
        os.makedirs(sd2_1_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/stabilityai/stable-diffusion-2-1", sd2_1_model_path)
        gr.Info("Stable Diffusion 2.1 model downloaded")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        pipe = StableDiffusionDiffEditPipeline().StableDiffusionDiffEditPipeline.from_pretrained(
            sd2_1_model_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
            use_safetensors=True,
        ).to(device)
        pipe.scheduler = DDIMScheduler().DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.inverse_scheduler = DDIMInverseScheduler().DDIMInverseScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()

        if seed == "" or seed is None:
            seed = random.randint(0, 2 ** 32 - 1)
        else:
            seed = int(seed)
        generator = torch.Generator(device).manual_seed(seed)

        raw_image = Image.open(init_image).convert("RGB").resize((768, 768))

        compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
        source_prompt_embeds = compel_proc(source_prompt)
        source_negative_prompt_embeds = compel_proc(source_negative_prompt)
        target_prompt_embeds = compel_proc(target_prompt)
        target_negative_prompt_embeds = compel_proc(target_negative_prompt)

        mask_image = pipe.generate_mask(
            image=raw_image,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
        )

        inv_latents = pipe.invert(
            prompt_embeds=source_prompt_embeds,
            negative_prompt_embeds=source_negative_prompt_embeds,
            image=raw_image
        ).latents

        output_image = pipe(
            prompt_embeds=target_prompt_embeds,
            negative_prompt_embeds=target_negative_prompt_embeds,
            mask_image=mask_image,
            image_latents=inv_latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        mask_image_pil = Image.fromarray((mask_image.squeeze()*255).astype("uint8"), "L").resize((768, 768))

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)

        output_filename = f"diffedit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        output_path = os.path.join(image_dir, output_filename)
        output_image.save(output_path, format=output_format.upper())

        mask_filename = f"diffedit_mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        mask_path = os.path.join(image_dir, mask_filename)
        mask_image_pil.save(mask_path, format=output_format.upper())

        return output_path, mask_path, f"Images generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        flush()


def generate_image_blip_diffusion(text_prompt_input, negative_prompt, cond_image, cond_subject, tgt_subject,
                                  num_inference_steps, guidance_scale, height, width, output_format):
    blip_diffusion_path = os.path.join("inputs", "image", "sd_models", "blip-diff")

    if not cond_image:
        gr.Info("Please upload an conditional image!")
        return None, None

    if not os.path.exists(blip_diffusion_path):
        gr.Info("Downloading BlipDiffusion model...")
        os.makedirs(blip_diffusion_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/Salesforce/blipdiffusion", blip_diffusion_path)
        gr.Info("BlipDiffusion model downloaded")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        blip_diffusion_pipe = BlipDiffusionPipeline().BlipDiffusionPipeline.from_pretrained(
            blip_diffusion_path, torch_dtype=torch_dtype
        ).to(device)

        cond_image = Image.open(cond_image).convert("RGB")

        output = blip_diffusion_pipe(
            text_prompt_input,
            cond_image,
            cond_subject,
            tgt_subject,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
        ).images

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"BlipDiffusion_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"blip_diffusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)

        output[0].save(image_path, format=output_format.upper())

        return image_path, "Image generated successfully."

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'blip_diffusion_pipe' in locals():
            del blip_diffusion_pipe
        flush()


def generate_image_animatediff(prompt, negative_prompt, input_video, strength, model_type, stable_diffusion_model_name, seed, motion_lora_name, num_frames, num_inference_steps,
                               guidance_scale, width, height, clip_skip):

    if not stable_diffusion_model_name:
        gr.Info("Please, select a StableDiffusion model!")
        return None, None

    stable_diffusion_model_path = os.path.join("inputs", "image", "sd_models", f"{stable_diffusion_model_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    variant = "fp16" if torch_dtype == torch.float16 else "fp32"

    if not os.path.exists(stable_diffusion_model_path):
        gr.Info(f"StableDiffusion model not found: {stable_diffusion_model_path}")
        return None, None

    motion_adapter_path = os.path.join("inputs", "image", "sd_models", "motion_adapter")
    if not os.path.exists(motion_adapter_path):
        gr.Info("Downloading motion adapter...")
        os.makedirs(motion_adapter_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-2", motion_adapter_path)
        gr.Info("Motion adapter downloaded")

    try:

        if seed == "" or seed is None:
            seed = random.randint(0, 2 ** 32 - 1)
        else:
            seed = int(seed)
        generator = torch.Generator(device).manual_seed(seed)

        if model_type == "sd":
            if input_video:
                def load_video(input_video: str):
                    images = []
                    vid = imageio.get_reader(input_video)

                    for frame in vid:
                        pil_image = Image.fromarray(frame)
                        images.append(pil_image)

                    return images

                video = load_video(input_video)

                adapter = MotionAdapter().MotionAdapter.from_pretrained(motion_adapter_path, torch_dtype=torch_dtype)
                stable_diffusion_model = StableDiffusionPipeline().StableDiffusionPipeline.from_single_file(
                    stable_diffusion_model_path,
                    torch_dtype=torch_dtype,
                    variant=variant,
                ).to(device)

                stable_diffusion_model.enable_vae_slicing()
                stable_diffusion_model.enable_vae_tiling()
                stable_diffusion_model.enable_model_cpu_offload()

                pipe = AnimateDiffVideoToVideoPipeline().AnimateDiffVideoToVideoPipeline(
                    unet=stable_diffusion_model.unet,
                    text_encoder=stable_diffusion_model.text_encoder,
                    vae=stable_diffusion_model.vae,
                    motion_adapter=adapter,
                    tokenizer=stable_diffusion_model.tokenizer,
                    feature_extractor=stable_diffusion_model.feature_extractor,
                    scheduler=stable_diffusion_model.scheduler,
                ).to(device)

                pipe.safety_checker = None

                compel_proc = Compel(tokenizer=pipe.tokenizer,
                                     text_encoder=pipe.text_encoder)
                prompt_embeds = compel_proc(prompt)
                negative_prompt_embeds = compel_proc(negative_prompt)

                output = pipe(
                    video=video,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    clip_skip=clip_skip,
                )

            else:
                adapter = MotionAdapter().MotionAdapter.from_pretrained(motion_adapter_path, torch_dtype=torch_dtype)
                stable_diffusion_model = StableDiffusionPipeline().StableDiffusionPipeline.from_single_file(
                    stable_diffusion_model_path,
                    torch_dtype=torch_dtype,
                    variant=variant,
                ).to(device)

                stable_diffusion_model.enable_vae_slicing()
                stable_diffusion_model.enable_vae_tiling()
                stable_diffusion_model.enable_model_cpu_offload()

                pipe = AnimateDiffPipeline().AnimateDiffPipeline(
                    unet=stable_diffusion_model.unet,
                    text_encoder=stable_diffusion_model.text_encoder,
                    vae=stable_diffusion_model.vae,
                    motion_adapter=adapter,
                    tokenizer=stable_diffusion_model.tokenizer,
                    feature_extractor=stable_diffusion_model.feature_extractor,
                    scheduler=stable_diffusion_model.scheduler,
                ).to(device)

                if motion_lora_name:
                    motion_lora_path = os.path.join("inputs", "image", "sd_models", "motion_lora", motion_lora_name)
                    if not os.path.exists(motion_lora_path):
                        gr.Info(f"Downloading {motion_lora_name} motion lora...")
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
                        gr.Info(f"{motion_lora_name} motion lora downloaded")
                    pipe.load_lora_weights(motion_lora_path, adapter_name=motion_lora_name)

                pipe.enable_vae_slicing()
                pipe.enable_model_cpu_offload()
                pipe.enable_free_init(method="butterworth", use_fast_sampling=False)

                pipe.safety_checker = None

                compel_proc = Compel(tokenizer=pipe.tokenizer,
                                     text_encoder=pipe.text_encoder)
                prompt_embeds = compel_proc(prompt)
                negative_prompt_embeds = compel_proc(negative_prompt)

                output = pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    width=width,
                    height=height,
                    generator=generator,
                    clip_skip=clip_skip,
                )

        elif model_type == "sdxl":
            sdxl_adapter_path = os.path.join("inputs", "image", "sd_models", "motion_adapter_sdxl")
            if not os.path.exists(sdxl_adapter_path):
                gr.Info("Downloading SDXL motion adapter...")
                os.makedirs(sdxl_adapter_path, exist_ok=True)
                Repo.clone_from("https://huggingface.co/guoyww/animatediff-motion-adapter-sdxl-beta", sdxl_adapter_path)
                gr.Info("SDXL motion adapter downloaded")

            adapter = MotionAdapter().MotionAdapter.from_pretrained(sdxl_adapter_path, torch_dtype=torch_dtype)
            stable_diffusion_model = StableDiffusionXLPipeline().StableDiffusionXLPipeline.from_single_file(
                stable_diffusion_model_path,
                torch_dtype=torch_dtype,
                variant=variant,
            ).to(device)

            stable_diffusion_model.enable_vae_slicing()
            stable_diffusion_model.enable_vae_tiling()
            stable_diffusion_model.enable_model_cpu_offload()

            pipe = AnimateDiffSDXLPipeline().AnimateDiffSDXLPipeline(
                unet=stable_diffusion_model.unet,
                text_encoder=stable_diffusion_model.text_encoder,
                text_encoder_2=stable_diffusion_model.text_encoder_2,
                vae=stable_diffusion_model.vae,
                motion_adapter=adapter,
                tokenizer=stable_diffusion_model.tokenizer,
                tokenizer_2=stable_diffusion_model.tokenizer_2,
                feature_extractor=stable_diffusion_model.feature_extractor,
                scheduler=stable_diffusion_model.scheduler,
            ).to(device)

            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
            pipe.enable_free_init(method="butterworth", use_fast_sampling=False)

            pipe.safety_checker = None

            compel = Compel(
                tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True]
            )
            prompt_embeds, pooled_prompt_embeds = compel(prompt)

            output = pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                num_frames=num_frames,
                generator=generator,
                clip_skip=clip_skip,
            )

        frames = output.frames[0]

        today = datetime.now().date()
        output_dir = os.path.join('outputs', f"AnimateDiff_{today.strftime('%Y%m%d')}")
        os.makedirs(output_dir, exist_ok=True)

        gif_filename = f"animatediff_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
        gif_path = os.path.join(output_dir, gif_filename)

        export_to_gif(frames, gif_path)

        return gif_path, f"GIF generated successfully. Seed used: {seed}"

    except Exception as e:
        if UnboundLocalError:
            pass
        else:
            gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        if 'stable_diffusion_model' in locals():
            del stable_diffusion_model
        if 'adapter' in locals():
            del adapter
        flush()


def generate_hotshotxl(prompt, negative_prompt, steps, width, height, video_length, video_duration):

    hotshotxl_model_path = os.path.join("inputs", "image", "sd_models", "hotshot_xl")
    hotshotxl_base_model_path = os.path.join("inputs", "image", "sd_models", "hotshot_xl_base")

    if not os.path.exists(hotshotxl_model_path):
        gr.Info("Downloading HotShot-XL...")
        os.makedirs(hotshotxl_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/hotshotco/Hotshot-XL", hotshotxl_model_path)
        gr.Info("HotShot-XL downloaded")

    if not os.path.exists(hotshotxl_base_model_path):
        gr.Info("Downloading HotShot-XL base model...")
        os.makedirs(hotshotxl_base_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/hotshotco/SDXL-512", hotshotxl_base_model_path)
        gr.Info("HotShot-XL base model downloaded")

    try:
        output_filename = f"hotshotxl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
        output_path = os.path.join('outputs', f"HotshotXL_{datetime.now().strftime('%Y%m%d')}", output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        command = [
            "python", "ThirdPartyRepository/Hotshot-XL/inference.py",
            f"--pretrained_path={hotshotxl_model_path}",
            f"--spatial_unet_base={hotshotxl_base_model_path}",
            f"--prompt={prompt}",
            f"--negative_prompt={negative_prompt}",
            f"--output={output_path}",
            f"--steps={steps}",
            f"--width={width}",
            f"--height={height}",
            f"--video_length={video_length}",
            f"--video_duration={video_duration}"
        ]

        subprocess.run(command, shell=True, check=True)

        return output_path, "GIF generated successfully!"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        flush()


def generate_video(init_image, output_format, seed, video_settings_html, motion_bucket_id, noise_aug_strength, fps, num_frames, decode_chunk_size,
                   iv2gen_xl_settings_html, prompt, negative_prompt, num_inference_steps, guidance_scale):

    if not init_image:
        gr.Info("Please upload an initial image!")
        return None, None, None

    today = datetime.now().date()
    video_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
    os.makedirs(video_dir, exist_ok=True)

    if output_format == "mp4":
        video_model_name = "vdo/stable-video-diffusion-img2vid-xt-1-1"
        video_model_path = os.path.join("inputs", "image", "sd_models", "video", "SVD")

        if not os.path.exists(video_model_path):
            gr.Info(f"Downloading StableVideoDiffusion model")
            os.makedirs(video_model_path, exist_ok=True)
            Repo.clone_from(f"https://huggingface.co/{video_model_name}", video_model_path)
            gr.Info(f"StableVideoDiffusion model downloaded")

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            variant = "fp16" if torch_dtype == torch.float16 else "fp32"
            pipe = StableVideoDiffusionPipeline().StableVideoDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path=video_model_path,
                torch_dtype=torch_dtype,
                variant=variant
            ).to(device)
            pipe.enable_model_cpu_offload()
            pipe.unet.enable_forward_chunking()

            image = load_image(init_image)
            image = image.resize((1024, 576))

            if seed == "" or seed is None:
                seed = random.randint(0, 2 ** 32 - 1)
            else:
                seed = int(seed)
            generator = torch.Generator(device).manual_seed(seed)
            frames = pipe(image, decode_chunk_size=decode_chunk_size, generator=generator,
                          motion_bucket_id=motion_bucket_id, noise_aug_strength=noise_aug_strength, num_frames=num_frames).frames[0]

            video_filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            video_path = os.path.join(video_dir, video_filename)
            export_to_video(frames, video_path, fps=fps)

            return video_path, None, f"MP4 generated successfully. Seed used: {seed}"

        except Exception as e:
            if UnboundLocalError:
                pass
            else:
                gr.Error(f"An error occurred: {str(e)}")
            return None, None, None

        finally:
            if 'pipe' in locals():
                del pipe
            flush()

    elif output_format == "gif":
        video_model_name = "ali-vilab/i2vgen-xl"
        video_model_path = os.path.join("inputs", "image", "sd_models", "video", "i2vgenxl")

        if not os.path.exists(video_model_path):
            gr.Info(f"Downloading i2vgen-xl model")
            os.makedirs(video_model_path, exist_ok=True)
            Repo.clone_from(f"https://huggingface.co/{video_model_name}", video_model_path)
            gr.Info(f"i2vgen-xl model downloaded")

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            variant = "fp16" if torch_dtype == torch.float16 else "fp32"
            pipe = I2VGenXLPipeline().I2VGenXLPipeline.from_pretrained(video_model_path, torch_dtype=torch_dtype, variant=variant).to(device)
            pipe.enable_model_cpu_offload()

            image = load_image(init_image).convert("RGB")

            if seed == "" or seed is None:
                seed = random.randint(0, 2 ** 32 - 1)
            else:
                seed = int(seed)
            generator = torch.Generator(device).manual_seed(seed)

            frames = pipe(
                prompt=prompt,
                image=image,
                num_inference_steps=num_inference_steps,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                generator=generator
            ).frames[0]

            video_filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
            video_path = os.path.join(video_dir, video_filename)
            export_to_gif(frames, video_path)

            return None, video_path, f"GIF generated successfully. Seed used: {seed}"

        except Exception as e:
            if UnboundLocalError:
                pass
            else:
                gr.Error(f"An error occurred: {str(e)}")
            return None, None, None

        finally:
            if 'pipe' in locals():
                del pipe
            flush()


def generate_image_ldm3d(prompt, negative_prompt, seed, width, height, num_inference_steps, guidance_scale, num_images_per_prompt, output_format):

    ldm3d_model_path = os.path.join("inputs", "image", "sd_models", "ldm3d")

    if not os.path.exists(ldm3d_model_path):
        gr.Info("Downloading LDM3D model...")
        os.makedirs(ldm3d_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/Intel/ldm3d-4c", ldm3d_model_path)
        gr.Info("LDM3D model downloaded")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = StableDiffusionLDM3DPipeline().StableDiffusionLDM3DPipeline.from_pretrained(ldm3d_model_path, torch_dtype=torch_dtype).to(device)

        if seed == "" or seed is None:
            seed = random.randint(0, 2 ** 32 - 1)
        else:
            seed = int(seed)
        generator = torch.Generator(device).manual_seed(seed)

        compel_proc = Compel(tokenizer=pipe.tokenizer,
                             text_encoder=pipe.text_encoder)
        prompt_embeds = compel_proc(prompt)
        negative_prompt_embeds = compel_proc(negative_prompt)

        output = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
        )

        rgb_image_paths = []
        depth_image_paths = []

        for i, (rgb_image, depth_image) in enumerate(zip(output.rgb, output.depth)):
            today = datetime.now().date()
            image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
            os.makedirs(image_dir, exist_ok=True)

            rgb_filename = f"ldm3d_rgb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.{output_format}"
            depth_filename = f"ldm3d_depth_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.{output_format}"

            rgb_path = os.path.join(image_dir, rgb_filename)
            depth_path = os.path.join(image_dir, depth_filename)

            rgb_image.save(rgb_path)
            depth_image.save(depth_path)

            rgb_image_paths.append(rgb_path)
            depth_image_paths.append(depth_path)

        return rgb_image_paths, depth_image_paths, f"Images generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None, None

    finally:
        if 'pipe' in locals():
            del pipe
        flush()


def generate_image_sd3_txt2img(prompt, negative_prompt, model_type, diffusers_version, quantize_sd3_model_name, enable_quantize, quantize_version, seed, stop_button, vae_model_name, lora_model_names, lora_scales, scheduler, num_inference_steps, guidance_scale, width, height, max_sequence_length, clip_skip, num_images_per_prompt, enable_taesd, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.bfloat32

    if enable_quantize:
        try:
            if not quantize_sd3_model_name:
                gr.Info("Please select a GGUF model!")
                return None, None

            if seed == "" or seed is None:
                seed = random.randint(0, 2 ** 32 - 1)
            else:
                seed = int(seed)

            params = {
                'model_name': quantize_sd3_model_name,
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'cfg_scale': guidance_scale,
                'height': height,
                'width': width,
                'sample_steps': num_inference_steps,
                'seed': seed
            }

            env = os.environ.copy()
            env['PYTHONPATH'] = os.pathsep.join(sys.path)

            if quantize_version == "SD3":
                result = subprocess.run(
                    [sys.executable, 'inputs/image/sd_models/sd-txt2img-quantize.py', json.dumps(params)],
                    capture_output=True, text=True, env=env
                )
            else:
                result = subprocess.run(
                    [sys.executable, 'inputs/image/sd_models/sd3.5-txt2img-quantize.py', json.dumps(params)],
                    capture_output=True, text=True, env=env
                )

            if result.returncode != 0:
                gr.Info(f"Error in sd-txt2img-quantize.py: {result.stderr}")
                return None, None

            image_paths = []
            output = result.stdout.strip()
            for part in output.split("IMAGE_PATH:"):
                if part:
                    image_path = part.strip().split('\n')[0]
                    if os.path.exists(image_path):
                        image_paths.append(image_path)

            return image_paths, f"Images generated successfully using quantized model. Seed used: {seed}"

        except Exception as e:
            gr.Error(f"An error occurred: {str(e)}")
            return None, None

    else:
        try:
            if not quantize_sd3_model_name:
                gr.Info("Please select a model!")
                return None, None

            stable_diffusion_model_path = os.path.join("inputs", "image", "sd_models", quantize_sd3_model_name)

            if model_type == "Diffusers":
                if diffusers_version == "3-Medium":
                    sd3_model_path = os.path.join("inputs", "image", "sd_models", "sd3")
                    if not os.path.exists(sd3_model_path):
                        gr.Info("Downloading Stable Diffusion 3 medium model...")
                        os.makedirs(sd3_model_path, exist_ok=True)
                        Repo.clone_from("https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers",
                                        sd3_model_path)
                        gr.Info("Stable Diffusion 3 medium model downloaded")
                    quantization_config = BitsAndBytesConfig().BitsAndBytesConfig(load_in_8bit=True)

                    text_encoder = T5EncoderModel().T5EncoderModel.from_pretrained(
                        sd3_model_path,
                        subfolder="text_encoder_3",
                        quantization_config=quantization_config,
                    )
                    pipe = StableDiffusion3Pipeline().StableDiffusion3Pipeline.from_pretrained(sd3_model_path,
                                                                                               device_map="balanced",
                                                                                               text_encoder_3=text_encoder,
                                                                                               torch_dtype=torch_dtype)
                elif diffusers_version == "3.5-Large":
                    sd3_5_model_path = os.path.join("inputs", "image", "sd_models", "sd3-5")
                    if not os.path.exists(sd3_5_model_path):
                        gr.Info("Downloading Stable Diffusion 3.5 large model...")
                        os.makedirs(sd3_5_model_path, exist_ok=True)
                        Repo.clone_from("https://huggingface.co/stabilityai/stable-diffusion-3.5-large",
                                        sd3_5_model_path)
                        gr.Info("Stable Diffusion 3.5 large model downloaded")
                    quantization_config = BitsAndBytesConfig().BitsAndBytesConfig(load_in_8bit=True)

                    text_encoder = T5EncoderModel().T5EncoderModel.from_pretrained(
                        sd3_5_model_path,
                        subfolder="text_encoder_3",
                        quantization_config=quantization_config,
                    )
                    pipe = StableDiffusion3Pipeline().StableDiffusion3Pipeline.from_pretrained(sd3_5_model_path,
                                                                                               device_map="balanced",
                                                                                               text_encoder_3=text_encoder,
                                                                                               torch_dtype=torch_dtype)
                elif diffusers_version == "3.5-Large-Turbo":
                    sd3_5_turbo_model_path = os.path.join("inputs", "image", "sd_models", "sd3-5-turbo")
                    if not os.path.exists(sd3_5_turbo_model_path):
                        gr.Info("Downloading Stable Diffusion 3.5 large turbo model...")
                        os.makedirs(sd3_5_turbo_model_path, exist_ok=True)
                        Repo.clone_from("https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo",
                                        sd3_5_turbo_model_path)
                        gr.Info("Stable Diffusion 3.5 large turbo model downloaded")
                    quantization_config = BitsAndBytesConfig().BitsAndBytesConfig(load_in_8bit=True)

                    text_encoder = T5EncoderModel().T5EncoderModel.from_pretrained(
                        sd3_5_turbo_model_path,
                        subfolder="text_encoder_3",
                        quantization_config=quantization_config,
                    )
                    pipe = StableDiffusion3Pipeline().StableDiffusion3Pipeline.from_pretrained(sd3_5_turbo_model_path,
                                                                                               device_map="balanced",
                                                                                               text_encoder_3=text_encoder,
                                                                                               torch_dtype=torch_dtype)
            else:
                pipe = StableDiffusion3Pipeline().StableDiffusion3Pipeline.from_single_file(stable_diffusion_model_path,
                                                                                            device_map=device,
                                                                                            torch_dtype=torch_dtype)

            if scheduler == "FlowMatchEulerDiscreteScheduler":
                pipe.scheduler = FlowMatchEulerDiscreteScheduler().FlowMatchEulerDiscreteScheduler.from_config(
                    pipe.scheduler.config)
            elif scheduler == "FlowMatchHeunDiscreteScheduler":
                pipe.scheduler = FlowMatchHeunDiscreteScheduler().FlowMatchHeunDiscreteScheduler.from_config(
                    pipe.scheduler.config)

            if vae_model_name is not None:
                vae_model_path = os.path.join("inputs", "image", "sd_models", "vae", f"{vae_model_name}")
                if os.path.exists(vae_model_path):
                    vae = AutoencoderKL().AutoencoderKL.from_single_file(vae_model_path, torch_dtype=torch_dtype)
                    pipe.vae = vae.to(device)

            if enable_taesd:
                pipe.vae = AutoencoderTiny().AutoencoderTiny.from_pretrained("madebyollin/taesd3",
                                                                             torch_dtype=torch_dtype)
                pipe.vae.config.shift_factor = 0.0

            pipe.enable_model_cpu_offload()

            if seed == "" or seed is None:
                seed = random.randint(0, 2 ** 32 - 1)
            else:
                seed = int(seed)
            generator = torch.Generator(device).manual_seed(seed)

            if isinstance(lora_scales, str):
                lora_scales = [float(scale.strip()) for scale in lora_scales.split(',') if scale.strip()]
            elif isinstance(lora_scales, (int, float)):
                lora_scales = [float(lora_scales)]

            lora_loaded = False
            if lora_model_names and lora_scales:
                if len(lora_model_names) != len(lora_scales):
                    gr.Warning(
                        f"Number of LoRA models ({len(lora_model_names)}) does not match number of scales ({len(lora_scales)}). Using available scales.", duration=5)

                for i, lora_model_name in enumerate(lora_model_names):
                    if i < len(lora_scales):
                        lora_scale = lora_scales[i]
                    else:
                        lora_scale = 1.0

                    lora_model_path = os.path.join("inputs", "image", "sd_models", "lora", lora_model_name)
                    if os.path.exists(lora_model_path):
                        adapter_name = os.path.splitext(os.path.basename(lora_model_name))[0]
                        try:
                            pipe.load_lora_weights(lora_model_path, adapter_name=adapter_name)
                            pipe.fuse_lora(lora_scale=lora_scale)
                            lora_loaded = True
                            gr.Info(f"Loaded LoRA {lora_model_name} with scale {lora_scale}")
                        except Exception as e:
                            gr.Warning(f"Error loading LoRA {lora_model_name}: {str(e)}", duration=5)

            def combined_callback(pipe, i, t, callback_kwargs):
                nonlocal stop_idx
                if stop_signal and stop_idx is None:
                    stop_idx = i
                if i == stop_idx:
                    pipe._interrupt = True

                progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")

                return callback_kwargs

            prompt_embeds, prompt_neg_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = get_weighted_text_embeddings_sd3(pipe=pipe, prompt=prompt, neg_prompt=negative_prompt)

            images = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=prompt_neg_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                max_sequence_length=max_sequence_length,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                generator=generator,
                callback_on_step_end=combined_callback,
                callback_on_step_end_tensor_inputs=["latents"]
            ).images

            image_paths = []
            for i, image in enumerate(images):
                today = datetime.now().date()
                image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
                os.makedirs(image_dir, exist_ok=True)
                image_filename = f"sd3_txt2img_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.{output_format}"
                image_path = os.path.join(image_dir, image_filename)
                metadata = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "model": "Stable Diffusion 3",
                    "seed": seed,
                    "lora_models": lora_model_names if lora_model_names else None,
                    "lora_scales": lora_scales if lora_model_names else None,
                    "steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "width": width,
                    "height": height,
                    "max_sequence_length": max_sequence_length,
                    "clip_skip": clip_skip,
                }

                image.save(image_path, format=output_format.upper())
                add_metadata_to_file(image_path, metadata)
                image_paths.append(image_path)

            return image_paths, f"Images generated successfully. Seed used: {seed}"

        except Exception as e:
            gr.Error(f"An error occurred: {str(e)}")
            return None, None, None

        finally:
            if 'pipe' in locals():
                del pipe
            if 'text_encoder' in locals():
                del text_encoder
            flush()


def generate_image_sd3_img2img(prompt, negative_prompt, init_image, strength, diffusers_version, quantize_sd3_model_name, enable_quantize, quantize_version, seed, stop_button, scheduler, num_inference_steps, guidance_scale, width, height, max_sequence_length, clip_skip, num_images_per_prompt, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    if not init_image:
        gr.Info("Please upload an initial image!")
        return None, None

    if enable_quantize:
        try:
            if not quantize_sd3_model_name:
                gr.Info("Please select a GGUF model!")
                return None, None

            if seed == "" or seed is None:
                seed = random.randint(0, 2 ** 32 - 1)
            else:
                seed = int(seed)

            params = {
                'model_name': quantize_sd3_model_name,
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'cfg_scale': guidance_scale,
                'height': height,
                'width': width,
                'sample_steps': num_inference_steps,
                'image': init_image,
                'strength': strength,
                'seed': seed
            }

            env = os.environ.copy()
            env['PYTHONPATH'] = os.pathsep.join(sys.path)

            if quantize_version == "SD3":
                result = subprocess.run(
                    [sys.executable, 'inputs/image/sd_models/sd-img2img-quantize.py', json.dumps(params)],
                    capture_output=True, text=True, env=env
                )
            else:
                result = subprocess.run(
                    [sys.executable, 'inputs/image/sd_models/sd3.5-img2img-quantize.py', json.dumps(params)],
                    capture_output=True, text=True, env=env
                )

            if result.returncode != 0:
                gr.Info(f"Error in sd-img2img-quantize.py: {result.stderr}")
                return None, None

            image_paths = []
            output = result.stdout.strip()
            for part in output.split("IMAGE_PATH:"):
                if part:
                    image_path = part.strip().split('\n')[0]
                    if os.path.exists(image_path):
                        image_paths.append(image_path)

            return image_paths, f"Images generated successfully using quantized model. Seed used: {seed}"

        except Exception as e:
            gr.Error(f"An error occurred: {str(e)}")
            return None, None

    else:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if device == "cuda" else torch.float32

            if diffusers_version == "3-Medium":
                sd3_model_path = os.path.join("inputs", "image", "sd_models", "sd3")
                if not os.path.exists(sd3_model_path):
                    gr.Info("Downloading Stable Diffusion 3 medium model...")
                    os.makedirs(sd3_model_path, exist_ok=True)
                    Repo.clone_from("https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers",
                                    sd3_model_path)
                    gr.Info("Stable Diffusion 3 medium model downloaded")
                quantization_config = BitsAndBytesConfig().BitsAndBytesConfig(load_in_8bit=True)

                text_encoder = T5EncoderModel().T5EncoderModel.from_pretrained(
                    sd3_model_path,
                    subfolder="text_encoder_3",
                    quantization_config=quantization_config,
                )
                pipe = StableDiffusion3Img2ImgPipeline().StableDiffusion3Img2ImgPipeline.from_pretrained(sd3_model_path,
                                                                                           device_map="balanced",
                                                                                           text_encoder_3=text_encoder,
                                                                                           torch_dtype=torch_dtype)
            elif diffusers_version == "3.5-Large":
                sd3_5_model_path = os.path.join("inputs", "image", "sd_models", "sd3-5")
                if not os.path.exists(sd3_5_model_path):
                    gr.Info("Downloading Stable Diffusion 3.5 large model...")
                    os.makedirs(sd3_5_model_path, exist_ok=True)
                    Repo.clone_from("https://huggingface.co/stabilityai/stable-diffusion-3.5-large",
                                    sd3_5_model_path)
                    gr.Info("Stable Diffusion 3.5 large model downloaded")
                quantization_config = BitsAndBytesConfig().BitsAndBytesConfig(load_in_8bit=True)

                text_encoder = T5EncoderModel().T5EncoderModel.from_pretrained(
                    sd3_5_model_path,
                    subfolder="text_encoder_3",
                    quantization_config=quantization_config,
                )
                pipe = StableDiffusion3Img2ImgPipeline().StableDiffusion3Img2ImgPipeline.from_pretrained(sd3_5_model_path,
                                                                                           device_map="balanced",
                                                                                           text_encoder_3=text_encoder,
                                                                                           torch_dtype=torch_dtype)
            elif diffusers_version == "3.5-Large-Turbo":
                sd3_5_turbo_model_path = os.path.join("inputs", "image", "sd_models", "sd3-5-turbo")
                if not os.path.exists(sd3_5_turbo_model_path):
                    gr.Info("Downloading Stable Diffusion 3.5 large turbo model...")
                    os.makedirs(sd3_5_turbo_model_path, exist_ok=True)
                    Repo.clone_from("https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo",
                                    sd3_5_turbo_model_path)
                    gr.Info("Stable Diffusion 3.5 large turbo model downloaded")
                quantization_config = BitsAndBytesConfig().BitsAndBytesConfig(load_in_8bit=True)

                text_encoder = T5EncoderModel().T5EncoderModel.from_pretrained(
                    sd3_5_turbo_model_path,
                    subfolder="text_encoder_3",
                    quantization_config=quantization_config,
                )
                pipe = StableDiffusion3Img2ImgPipeline().StableDiffusion3Img2ImgPipeline.from_pretrained(sd3_5_turbo_model_path,
                                                                                           device_map="balanced",
                                                                                           text_encoder_3=text_encoder,
                                                                                           torch_dtype=torch_dtype)

            if scheduler == "FlowMatchEulerDiscreteScheduler":
                pipe.scheduler = FlowMatchEulerDiscreteScheduler().FlowMatchEulerDiscreteScheduler.from_config(
                    pipe.scheduler.config)
            elif scheduler == "FlowMatchHeunDiscreteScheduler":
                pipe.scheduler = FlowMatchHeunDiscreteScheduler().FlowMatchHeunDiscreteScheduler.from_config(
                    pipe.scheduler.config)

            pipe.enable_model_cpu_offload()

            init_image = Image.open(init_image).convert("RGB")
            init_image = init_image.resize((width, height))

            if seed == "" or seed is None:
                seed = random.randint(0, 2 ** 32 - 1)
            else:
                seed = int(seed)
            generator = torch.Generator(device).manual_seed(seed)

            def combined_callback(pipe, i, t, callback_kwargs):
                nonlocal stop_idx
                if stop_signal and stop_idx is None:
                    stop_idx = i
                if i == stop_idx:
                    pipe._interrupt = True
                progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")
                return callback_kwargs

            images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                generator=generator,
                callback_on_step_end=combined_callback,
                callback_on_step_end_tensor_inputs=["latents"]
            ).images

            image_paths = []
            for i, image in enumerate(images):
                today = datetime.now().date()
                image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
                os.makedirs(image_dir, exist_ok=True)
                image_filename = f"sd3_img2img_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.{output_format}"
                image_path = os.path.join(image_dir, image_filename)
                image.save(image_path, format=output_format.upper())
                image_paths.append(image_path)

            return image_paths, f"Images generated successfully. Seed used: {seed}"

        except Exception as e:
            gr.Error(f"An error occurred: {str(e)}")
            return None, None

        finally:
            if 'pipe' in locals():
                del pipe
            if 'text_encoder' in locals():
                del text_encoder
            flush()


def generate_image_sd3_controlnet(prompt, negative_prompt, init_image, diffusers_version, controlnet_model, seed, stop_button, scheduler, num_inference_steps, guidance_scale, controlnet_conditioning_scale, width, height, max_sequence_length, clip_skip, num_images_per_prompt, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    if not init_image:
        gr.Info("Please upload an initial image!")
        return None, None, None

    if not controlnet_model:
        gr.Info("Please select a controlnet model!")
        return None, None, None

    controlnet_path = os.path.join("inputs", "image", "sd_models", "sd3", "controlnet", f"sd3_{controlnet_model}")

    if not os.path.exists(controlnet_path):
        gr.Info(f"Downloading SD3 ControlNet {controlnet_model} model...")
        os.makedirs(controlnet_path, exist_ok=True)
        Repo.clone_from(f"https://huggingface.co/InstantX/SD3-Controlnet-{controlnet_model}", controlnet_path)
        gr.Info(f"SD3 ControlNet {controlnet_model} model downloaded")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        controlnet = SD3ControlNetModel().SD3ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch_dtype)

        if diffusers_version == "3-Medium":
            sd3_model_path = os.path.join("inputs", "image", "sd_models", "sd3")
            if not os.path.exists(sd3_model_path):
                gr.Info("Downloading Stable Diffusion 3 medium model...")
                os.makedirs(sd3_model_path, exist_ok=True)
                Repo.clone_from("https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers",
                                sd3_model_path)
                gr.Info("Stable Diffusion 3 medium model downloaded")
            quantization_config = BitsAndBytesConfig().BitsAndBytesConfig(load_in_8bit=True)

            text_encoder = T5EncoderModel().T5EncoderModel.from_pretrained(
                sd3_model_path,
                subfolder="text_encoder_3",
                quantization_config=quantization_config,
            )
            pipe = StableDiffusion3ControlNetPipeline().StableDiffusion3ControlNetPipeline.from_pretrained(sd3_model_path,
                                                                                                     device_map="balanced",
                                                                                                     text_encoder_3=text_encoder,
                                                                                                     torch_dtype=torch_dtype,
                                                                                                     controlnet=controlnet)
        elif diffusers_version == "3.5-Large":
            sd3_5_model_path = os.path.join("inputs", "image", "sd_models", "sd3-5")
            if not os.path.exists(sd3_5_model_path):
                gr.Info("Downloading Stable Diffusion 3.5 large model...")
                os.makedirs(sd3_5_model_path, exist_ok=True)
                Repo.clone_from("https://huggingface.co/stabilityai/stable-diffusion-3.5-large",
                                sd3_5_model_path)
                gr.Info("Stable Diffusion 3.5 large model downloaded")
            quantization_config = BitsAndBytesConfig().BitsAndBytesConfig(load_in_8bit=True)

            text_encoder = T5EncoderModel().T5EncoderModel.from_pretrained(
                sd3_5_model_path,
                subfolder="text_encoder_3",
                quantization_config=quantization_config,
            )
            pipe = StableDiffusion3ControlNetPipeline().StableDiffusion3ControlNetPipeline.from_pretrained(sd3_5_model_path,
                                                                                                     device_map="balanced",
                                                                                                     text_encoder_3=text_encoder,
                                                                                                     torch_dtype=torch_dtype,
                                                                                                     controlnet=controlnet)
        elif diffusers_version == "3.5-Large-Turbo":
            sd3_5_turbo_model_path = os.path.join("inputs", "image", "sd_models", "sd3-5-turbo")
            if not os.path.exists(sd3_5_turbo_model_path):
                gr.Info("Downloading Stable Diffusion 3.5 large turbo model...")
                os.makedirs(sd3_5_turbo_model_path, exist_ok=True)
                Repo.clone_from("https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo",
                                sd3_5_turbo_model_path)
                gr.Info("Stable Diffusion 3.5 large turbo model downloaded")
            quantization_config = BitsAndBytesConfig().BitsAndBytesConfig(load_in_8bit=True)

            text_encoder = T5EncoderModel().T5EncoderModel.from_pretrained(
                sd3_5_turbo_model_path,
                subfolder="text_encoder_3",
                quantization_config=quantization_config,
            )
            pipe = StableDiffusion3ControlNetPipeline().StableDiffusion3ControlNetPipeline.from_pretrained(
                sd3_5_turbo_model_path,
                device_map="balanced",
                text_encoder_3=text_encoder,
                torch_dtype=torch_dtype,
                controlnet=controlnet)

        if scheduler == "FlowMatchEulerDiscreteScheduler":
            pipe.scheduler = FlowMatchEulerDiscreteScheduler().FlowMatchEulerDiscreteScheduler.from_config(
                pipe.scheduler.config)
        elif scheduler == "FlowMatchHeunDiscreteScheduler":
            pipe.scheduler = FlowMatchHeunDiscreteScheduler().FlowMatchHeunDiscreteScheduler.from_config(
                pipe.scheduler.config)

        pipe.enable_model_cpu_offload()

        init_image = Image.open(init_image).convert("RGB")
        init_image = init_image.resize((width, height))

        if controlnet_model.lower() == "canny":
            control_image = init_image
        elif controlnet_model.lower() == "pose":
            control_image = init_image

        if seed == "" or seed is None:
            seed = random.randint(0, 2 ** 32 - 1)
        else:
            seed = int(seed)
        generator = torch.Generator(device).manual_seed(seed)

        def combined_callback(pipe, i, t, callback_kwargs):
            nonlocal stop_idx
            if stop_signal and stop_idx is None:
                stop_idx = i
            if i == stop_idx:
                pipe._interrupt = True
            progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")
            return callback_kwargs

        images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            control_image=control_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=clip_skip,
            generator=generator,
            callback_on_step_end=combined_callback,
            callback_on_step_end_tensor_inputs=["latents"]
        ).images

        image_paths = []
        control_image_paths = []
        for i, (image, control_image) in enumerate(zip(images, [control_image] * len(images))):
            today = datetime.now().date()
            image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
            os.makedirs(image_dir, exist_ok=True)

            image_filename = f"sd3_controlnet_{controlnet_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.{output_format}"
            image_path = os.path.join(image_dir, image_filename)
            image.save(image_path, format=output_format.upper())
            image_paths.append(image_path)

            control_image_filename = f"sd3_controlnet_{controlnet_model}_control_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.{output_format}"
            control_image_path = os.path.join(image_dir, control_image_filename)
            control_image.save(control_image_path, format=output_format.upper())
            control_image_paths.append(control_image_path)

        return image_paths, control_image_paths, f"Images generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None, None

    finally:
        if 'pipe' in locals():
            del pipe
        if 'controlnet' in locals():
            del controlnet
        if 'text_encoder' in locals():
            del text_encoder
        flush()


def generate_image_sd3_inpaint(prompt, negative_prompt, diffusers_version, init_image, mask_image, seed, stop_button, scheduler, num_inference_steps, guidance_scale, width, height, max_sequence_length, clip_skip, num_images_per_prompt, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    if not init_image or not mask_image:
        gr.Info("Please, upload an initial image and a mask image!")
        return None, None

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        if diffusers_version == "3-Medium":
            sd3_model_path = os.path.join("inputs", "image", "sd_models", "sd3")
            if not os.path.exists(sd3_model_path):
                gr.Info("Downloading Stable Diffusion 3 medium model...")
                os.makedirs(sd3_model_path, exist_ok=True)
                Repo.clone_from("https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers",
                                sd3_model_path)
                gr.Info("Stable Diffusion 3 medium model downloaded")
            quantization_config = BitsAndBytesConfig().BitsAndBytesConfig(load_in_8bit=True)

            text_encoder = T5EncoderModel().T5EncoderModel.from_pretrained(
                sd3_model_path,
                subfolder="text_encoder_3",
                quantization_config=quantization_config,
            )
            pipe = StableDiffusion3InpaintPipeline().StableDiffusion3InpaintPipeline.from_pretrained(sd3_model_path,
                                                                                                     device_map="balanced",
                                                                                                     text_encoder_3=text_encoder,
                                                                                                     torch_dtype=torch_dtype)
        elif diffusers_version == "3.5-Large":
            sd3_5_model_path = os.path.join("inputs", "image", "sd_models", "sd3-5")
            if not os.path.exists(sd3_5_model_path):
                gr.Info("Downloading Stable Diffusion 3.5 large model...")
                os.makedirs(sd3_5_model_path, exist_ok=True)
                Repo.clone_from("https://huggingface.co/stabilityai/stable-diffusion-3.5-large",
                                sd3_5_model_path)
                gr.Info("Stable Diffusion 3.5 large model downloaded")
            quantization_config = BitsAndBytesConfig().BitsAndBytesConfig(load_in_8bit=True)

            text_encoder = T5EncoderModel().T5EncoderModel.from_pretrained(
                sd3_5_model_path,
                subfolder="text_encoder_3",
                quantization_config=quantization_config,
            )
            pipe = StableDiffusion3InpaintPipeline().StableDiffusion3InpaintPipeline.from_pretrained(sd3_5_model_path,
                                                                                                     device_map="balanced",
                                                                                                     text_encoder_3=text_encoder,
                                                                                                     torch_dtype=torch_dtype)
        elif diffusers_version == "3.5-Large-Turbo":
            sd3_5_turbo_model_path = os.path.join("inputs", "image", "sd_models", "sd3-5-turbo")
            if not os.path.exists(sd3_5_turbo_model_path):
                gr.Info("Downloading Stable Diffusion 3.5 large turbo model...")
                os.makedirs(sd3_5_turbo_model_path, exist_ok=True)
                Repo.clone_from("https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo",
                                sd3_5_turbo_model_path)
                gr.Info("Stable Diffusion 3.5 large turbo model downloaded")
            quantization_config = BitsAndBytesConfig().BitsAndBytesConfig(load_in_8bit=True)

            text_encoder = T5EncoderModel().T5EncoderModel.from_pretrained(
                sd3_5_turbo_model_path,
                subfolder="text_encoder_3",
                quantization_config=quantization_config,
            )
            pipe = StableDiffusion3InpaintPipeline().StableDiffusion3InpaintPipeline.from_pretrained(
                sd3_5_turbo_model_path,
                device_map="balanced",
                text_encoder_3=text_encoder,
                torch_dtype=torch_dtype)

        if scheduler == "FlowMatchEulerDiscreteScheduler":
            pipe.scheduler = FlowMatchEulerDiscreteScheduler().FlowMatchEulerDiscreteScheduler.from_config(
                pipe.scheduler.config)
        elif scheduler == "FlowMatchHeunDiscreteScheduler":
            pipe.scheduler = FlowMatchHeunDiscreteScheduler().FlowMatchHeunDiscreteScheduler.from_config(
                pipe.scheduler.config)

        pipe.enable_model_cpu_offload()

        init_image = Image.open(init_image).convert("RGB")
        init_image = init_image.resize((width, height))

        mask_image = Image.open(mask_image).convert("L")
        mask_image = mask_image.resize((width, height))

        if seed == "" or seed is None:
            seed = random.randint(0, 2 ** 32 - 1)
        else:
            seed = int(seed)
        generator = torch.Generator(device).manual_seed(seed)

        def combined_callback(pipe, i, t, callback_kwargs):
            nonlocal stop_idx
            if stop_signal and stop_idx is None:
                stop_idx = i
            if i == stop_idx:
                pipe._interrupt = True
            progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")
            return callback_kwargs

        images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=clip_skip,
            generator=generator,
            callback_on_step_end=combined_callback,
            callback_on_step_end_tensor_inputs=["latents"]
        ).images

        image_paths = []
        for i, image in enumerate(images):
            today = datetime.now().date()
            image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
            os.makedirs(image_dir, exist_ok=True)
            image_filename = f"sd3_inpaint_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.{output_format}"
            image_path = os.path.join(image_dir, image_filename)
            image.save(image_path, format=output_format.upper())
            image_paths.append(image_path)

        return image_paths, f"Images generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        if 'text_encoder' in locals():
            del text_encoder
        flush()


def generate_image_cascade(prompt, negative_prompt, seed, stop_button, width, height, prior_steps, prior_guidance_scale,
                           decoder_steps, decoder_guidance_scale, num_images_per_prompt, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    stable_cascade_model_path = os.path.join("inputs", "image", "sd_models", "cascade")

    if not os.path.exists(stable_cascade_model_path):
        gr.Info("Downloading Stable Cascade models...")
        os.makedirs(stable_cascade_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/stabilityai/stable-cascade-prior",
                        os.path.join(stable_cascade_model_path, "prior"))
        Repo.clone_from("https://huggingface.co/stabilityai/stable-cascade",
                        os.path.join(stable_cascade_model_path, "decoder"))
        gr.Info("Stable Cascade models downloaded")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.bfloat32
        variant = "bf16" if torch_dtype == torch.float16 else "bf32"
        prior = StableCascadePriorPipeline().StableCascadePriorPipeline.from_pretrained(os.path.join(stable_cascade_model_path, "prior"),
                                                           variant=variant, torch_dtype=torch_dtype).to(device)
        decoder = StableCascadeDecoderPipeline().StableCascadeDecoderPipeline.from_pretrained(os.path.join(stable_cascade_model_path, "decoder"),
                                                               variant=variant, torch_dtype=torch_dtype).to(device)
    except (ValueError, OSError):
        gr.Error("Failed to load the Stable Cascade models")
        return None, None

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    def combined_callback_prior(prior, i, t, callback_kwargs):
        nonlocal stop_idx
        if stop_signal and stop_idx is None:
            stop_idx = i
        if i == stop_idx:
            prior._interrupt = True
        progress((i + 1) / prior_steps, f"Step {i + 1}/{prior_steps}")
        return callback_kwargs

    def combined_callback_decoder(decoder, i, t, callback_kwargs):
        nonlocal stop_idx
        if stop_signal and stop_idx is None:
            stop_idx = i
        if i == stop_idx:
            decoder._interrupt = True
        progress((i + 1) / decoder_steps, f"Step {i + 1}/{decoder_steps}")
        return callback_kwargs

    try:
        prior.enable_model_cpu_offload()

        prior_output = prior(
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            guidance_scale=prior_guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=prior_steps,
            generator=generator,
            callback_on_step_end=combined_callback_prior,
            callback_on_step_end_tensor_inputs=["latents"]
        )

        decoder.enable_model_cpu_offload()

        decoder_output = decoder(
            image_embeddings=prior_output.image_embeddings.to(torch_dtype),
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=decoder_guidance_scale,
            output_type="pil",
            num_inference_steps=decoder_steps,
            generator=generator,
            callback_on_step_end=combined_callback_decoder,
            callback_on_step_end_tensor_inputs=["latents"]
        ).images

        image_paths = []
        for i, image in enumerate(decoder_output):
            today = datetime.now().date()
            image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
            os.makedirs(image_dir, exist_ok=True)
            image_filename = f"cascade_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.{output_format}"
            image_path = os.path.join(image_dir, image_filename)

            metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "model": "StableCascade",
                "width": width,
                "height": height,
                "prior_steps": prior_steps,
                "prior_guidance_scale": prior_guidance_scale,
                "decoder_steps": decoder_steps,
                "decoder_guidance_scale": decoder_guidance_scale,
                "seed": seed
            }

            image.save(image_path, format=output_format.upper())
            add_metadata_to_file(image_path, metadata)

            image_paths.append(image_path)

        return image_paths, f"Images generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'prior' in locals():
            del prior
        if 'decoder' in locals():
            del decoder
        flush()


def generate_image_t2i_ip_adapter(prompt, negative_prompt, ip_adapter_image, stable_diffusion_model_type, stable_diffusion_model_name, seed, num_inference_steps, guidance_scale, width, height, output_format):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    if not ip_adapter_image:
        gr.Info("Please upload an image!")
        return None, None

    if not stable_diffusion_model_name:
        gr.Info("Please select a StableDiffusion model!")
        return None, None

    stable_diffusion_model_path = os.path.join("inputs", "image", "sd_models", f"{stable_diffusion_model_name}")

    if not os.path.exists(stable_diffusion_model_path):
        gr.Info(f"StableDiffusion model not found: {stable_diffusion_model_path}")
        return None, None

    t2i_ip_adapter_path = os.path.join("inputs", "image", "sd_models", "t2i-ip-adapter")
    if not os.path.exists(t2i_ip_adapter_path):
        gr.Info("Downloading T2I IP-Adapter model...")
        os.makedirs(t2i_ip_adapter_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/h94/IP-Adapter", t2i_ip_adapter_path)
        gr.Info("T2I IP-Adapter model downloaded")

    try:

        if stable_diffusion_model_type == "SD":
            image_encoder = CLIPVisionModelWithProjection().CLIPVisionModelWithProjection.from_pretrained(
                t2i_ip_adapter_path,
                subfolder="models/image_encoder",
                torch_dtype=torch_dtype
            ).to(device)

            pipe = StableDiffusionPipeline().StableDiffusionPipeline.from_single_file(
                stable_diffusion_model_path,
                image_encoder=image_encoder,
                torch_dtype=torch_dtype
            ).to(device)

            pipe.load_ip_adapter(t2i_ip_adapter_path, subfolder="models",
                                 weight_name="ip-adapter-plus_sd15.safetensors")
        else:
            image_encoder = CLIPVisionModelWithProjection().CLIPVisionModelWithProjection.from_pretrained(
                t2i_ip_adapter_path,
                subfolder="sdxl_models/image_encoder",
                torch_dtype=torch_dtype
            ).to(device)

            pipe = StableDiffusionXLPipeline().StableDiffusionXLPipeline.from_single_file(
                stable_diffusion_model_path,
                image_encoder=image_encoder,
                torch_dtype=torch_dtype
            ).to(device)

            pipe.load_ip_adapter(t2i_ip_adapter_path, subfolder="sdxl_models",
                                 weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")

        image = Image.open(ip_adapter_image).convert("RGB")

        images = pipe(
            prompt=prompt,
            ip_adapter_image=image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        ).images

        image_paths = []
        for i, image in enumerate(images):
            today = datetime.now().date()
            image_dir = os.path.join('outputs', f"T2I_IP_Adapter_{today.strftime('%Y%m%d')}")
            os.makedirs(image_dir, exist_ok=True)
            image_filename = f"t2i_ip_adapter_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.{output_format}"
            image_path = os.path.join(image_dir, image_filename)
            image.save(image_path, format=output_format.upper())
            image_paths.append(image_path)

        return image_paths, f"Images generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        if 'image_encoder' in locals():
            del image_encoder
        flush()


def generate_image_ip_adapter_faceid(prompt, negative_prompt, face_image, s_scale, stable_diffusion_model_type, stable_diffusion_model_name, num_inference_steps, guidance_scale, width, height, output_format):

    if not face_image:
        gr.Info("Please upload a face image!")
        return None, None

    if not stable_diffusion_model_name:
        gr.Info("Please select a StableDiffusion model!")
        return None, None

    stable_diffusion_model_path = os.path.join("inputs", "image", "sd_models", f"{stable_diffusion_model_name}")

    if not os.path.exists(stable_diffusion_model_path):
        gr.Info(f"StableDiffusion model not found: {stable_diffusion_model_path}")
        return None, None

    image_encoder_path = os.path.join("inputs", "image", "sd_models", "image_encoder")
    if not os.path.exists(image_encoder_path):
        gr.Info("Downloading image encoder...")
        os.makedirs(image_encoder_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K", image_encoder_path)
        gr.Info("Image encoder downloaded")

    ip_ckpt_path = os.path.join("inputs", "image", "sd_models", "ip_adapter_faceid")
    if not os.path.exists(ip_ckpt_path):
        gr.Info("Downloading IP-Adapter FaceID checkpoint...")
        os.makedirs(ip_ckpt_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/h94/IP-Adapter-FaceID", ip_ckpt_path)
        gr.Info("IP-Adapter FaceID checkpoint downloaded")

    if stable_diffusion_model_type == "SD":
        ip_ckpt = os.path.join(ip_ckpt_path, "ip-adapter-faceid-plusv2_sd15.bin")
    else:
        ip_ckpt = os.path.join(ip_ckpt_path, "ip-adapter-faceid-plusv2_sdxl.bin")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        variant = "fp16" if torch_dtype == torch.float16 else "fp32"

        app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))

        image = cv2.imread(face_image)
        faces = app.get(image)

        if not faces:
            gr.Info("No face detected in the image.")
            return None, None

        faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224)

        noise_scheduler = DDIMScheduler().DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        if stable_diffusion_model_type == "SD":
            pipe = StableDiffusionPipeline().StableDiffusionPipeline.from_single_file(
                stable_diffusion_model_path, scheduler=noise_scheduler, use_safetensors=True, device_map="auto",
                torch_dtype=torch_dtype, variant=variant)
        elif stable_diffusion_model_type == "SDXL":
            pipe = StableDiffusionXLPipeline().StableDiffusionXLPipeline.from_single_file(
                stable_diffusion_model_path, scheduler=noise_scheduler, use_safetensors=True, device_map="auto", attention_slice=1,
                torch_dtype=torch_dtype, variant=variant)
        else:
            gr.Info("Invalid StableDiffusion model type!")
            return None, None

        ip_model = IPAdapterFaceIDPlus().IPAdapterFaceIDPlus(pipe, image_encoder_path, ip_ckpt, device)

        images = ip_model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            face_image=face_image,
            faceid_embeds=faceid_embeds,
            shortcut=True,
            s_scale=s_scale,
            num_samples=1,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"IPAdapterFaceID_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)

        image_paths = []
        for i, image in enumerate(images):
            image_filename = f"ip_adapter_faceid_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.{output_format}"
            image_path = os.path.join(image_dir, image_filename)
            image.save(image_path, format=output_format.upper())
            image_paths.append(image_path)

        return image_paths, None

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        if 'ip_model' in locals():
            del ip_model
        flush()


def generate_riffusion_text2image(prompt, negative_prompt, seed, stop_button, num_inference_steps, guidance_scale, height, width, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    riffusion_model_path = os.path.join("inputs", "image", "sd_models", "riffusion")

    if not os.path.exists(riffusion_model_path):
        gr.Info("Downloading Riffusion model...")
        os.makedirs(riffusion_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/riffusion/riffusion-model-v1", riffusion_model_path)
        gr.Info("Riffusion model downloaded")

    try:
        pipe = StableDiffusionPipeline().StableDiffusionPipeline.from_pretrained(riffusion_model_path, torch_dtype=torch_dtype)
        pipe = pipe.to(device)

        def combined_callback(pipe, i, t, callback_kwargs):
            nonlocal stop_idx
            if stop_signal and stop_idx is None:
                stop_idx = i
            if i == stop_idx:
                pipe._interrupt = True
            progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")
            return callback_kwargs

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            callback_on_step_end=combined_callback,
            callback_on_step_end_tensor_inputs=["latents"]
        ).images[0]

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"Riffusion_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"riffusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        return image_path, f"Image generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        flush()


def generate_riffusion_image2audio(image_path, output_format, progress=gr.Progress()):
    progress(0.1, desc="Initializing riffusion")
    if not image_path:
        gr.Info("Please upload an image file!")
        return None, None

    try:
        today = datetime.now().date()
        audio_dir = os.path.join('outputs', f"Riffusion_{today.strftime('%Y%m%d')}")
        os.makedirs(audio_dir, exist_ok=True)
        audio_filename = f"riffusion_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        audio_path = os.path.join(audio_dir, audio_filename)

        progress(0.4, desc="Preparing riffusion command")
        command = f"python -m ThirdPartyRepository/riffusion.cli image-to-audio {image_path} {audio_path}"

        progress(0.7, desc="Running riffusion image2audio")
        subprocess.run(command, shell=True, check=True)

        progress(1.0, desc="Riffusion image2audio complete")

        return audio_path, None

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        flush()


def generate_riffusion_audio2image(audio_path, output_format, progress=gr.Progress()):
    progress(0.1, desc="Initializing riffusion")
    if not audio_path:
        gr.Info("Please upload an audio file!")
        return None, None

    try:
        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"Riffusion_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"riffusion_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)

        progress(0.4, desc="Preparing riffusion command")
        command = f"python -m ThirdPartyRepository/riffusion.cli audio-to-image {audio_path} {image_path}"

        progress(0.7, desc="Running riffusion audio2image")
        subprocess.run(command, shell=True, check=True)

        progress(1.0, desc="Riffusion audio2image complete")

        return image_path, None

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        flush()


def generate_image_kandinsky_txt2img(prompt, negative_prompt, version, seed, stop_button, num_inference_steps, guidance_scale, height, width, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    variant = "fp16" if torch_dtype == torch.float16 else "fp32"

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    kandinsky_model_path = os.path.join("inputs", "image", "kandinsky")

    if not os.path.exists(kandinsky_model_path):
        gr.Info(f"Downloading Kandinsky {version} model...")
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
        gr.Info(f"Kandinsky {version} model downloaded")

    try:
        if version == "2.1":

            pipe_prior = KandinskyPriorPipeline().KandinskyPriorPipeline.from_pretrained(os.path.join(kandinsky_model_path, "2-1-prior"))
            pipe_prior.to(device)

            out = pipe_prior(prompt, negative_prompt=negative_prompt)
            image_emb = out.image_embeds
            negative_image_emb = out.negative_image_embeds

            pipe = KandinskyPipeline().KandinskyPipeline.from_pretrained(os.path.join(kandinsky_model_path, "2-1"))
            pipe.to(device)

            def combined_callback_2_1(pipe, i, t):
                nonlocal stop_idx
                if stop_signal and stop_idx is None:
                    stop_idx = i
                if i == stop_idx:
                    pipe._interrupt = True
                progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")

            image = pipe(
                prompt,
                negative_prompt=negative_prompt,
                image_embeds=image_emb,
                negative_image_embeds=negative_image_emb,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                callback=combined_callback_2_1,
                callback_steps=1
            ).images[0]

        elif version == "2.2":

            pipe_prior = KandinskyV22PriorPipeline().KandinskyV22PriorPipeline.from_pretrained(os.path.join(kandinsky_model_path, "2-2-prior"))
            pipe_prior.to(device)

            image_emb, negative_image_emb = pipe_prior(prompt, negative_prompt=negative_prompt).to_tuple()

            pipe = KandinskyV22Pipeline().KandinskyV22Pipeline.from_pretrained(os.path.join(kandinsky_model_path, "2-2-decoder"))
            pipe.to(device)

            def combined_callback_2_2(pipe, i, t, callback_kwargs):
                nonlocal stop_idx
                if stop_signal and stop_idx is None:
                    stop_idx = i
                if i == stop_idx:
                    pipe._interrupt = True
                progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")
                return callback_kwargs

            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image_embeds=image_emb,
                negative_image_embeds=negative_image_emb,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                callback_on_step_end=combined_callback_2_2,
                callback_on_step_end_tensor_inputs=["latents"]
            ).images[0]

        elif version == "3":

            pipe = AutoPipelineForText2Image().AutoPipelineForText2Image.from_pretrained(
                os.path.join(kandinsky_model_path, "3"), variant=variant, torch_dtype=torch_dtype
            )
            pipe.to(device)
            pipe.enable_model_cpu_offload()

            def combined_callback_3_0(pipe, i, t):
                nonlocal stop_idx
                if stop_signal and stop_idx is None:
                    stop_idx = i
                if i == stop_idx:
                    pipe._interrupt = True
                progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")

            image = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                generator=generator,
                callback=combined_callback_3_0,
                callback_steps=1
            ).images[0]

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"Kandinsky_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"kandinsky_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "seed": seed,
            "Kandinsky_version": version
        }

        image.save(image_path, format=output_format.upper())
        add_metadata_to_file(image_path, metadata)

        return image_path, f"Image generated successfully. Seed used: {seed}"

    except Exception as e:
        if UnboundLocalError:
            pass
        else:
            gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe_prior' in locals():
            del pipe_prior
        if 'pipe' in locals():
            del pipe
        flush()


def generate_image_kandinsky_img2img(prompt, negative_prompt, init_image, version, seed, stop_button, num_inference_steps, guidance_scale, strength, height, width, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    variant = "fp16" if torch_dtype == torch.float16 else "fp32"

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    if not init_image:
        gr.Info("Please upload an initial image!")
        return None, None

    kandinsky_model_path = os.path.join("inputs", "image", "kandinsky")

    if not os.path.exists(kandinsky_model_path):
        gr.Info(f"Downloading Kandinsky {version} model...")
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
        gr.Info(f"Kandinsky {version} model downloaded")

    try:
        if version == "2.1":
            pipe_prior = KandinskyPriorPipeline().KandinskyPriorPipeline.from_pretrained(
                os.path.join(kandinsky_model_path, "2-1-prior"), torch_dtype=torch_dtype
            )
            pipe_prior.to(device)

            image_emb, zero_image_emb = pipe_prior(prompt, negative_prompt=negative_prompt, return_dict=False)

            pipe = KandinskyImg2ImgPipeline().KandinskyImg2ImgPipeline.from_pretrained(
                os.path.join(kandinsky_model_path, "2-1"), torch_dtype=torch_dtype
            )
            pipe.to(device)

            init_image = Image.open(init_image).convert("RGB")
            init_image = init_image.resize((width, height))

            def combined_callback_2_1(pipe, i, t):
                nonlocal stop_idx
                if stop_signal and stop_idx is None:
                    stop_idx = i
                if i == stop_idx:
                    pipe._interrupt = True
                progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")

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
                generator=generator,
                callback=combined_callback_2_1,
                callback_steps=1
            ).images[0]

        elif version == "2.2":
            pipe = AutoPipelineForImage2Image().AutoPipelineForImage2Image.from_pretrained(
                os.path.join(kandinsky_model_path, "2-2-decoder"), torch_dtype=torch_dtype
            )
            pipe.to(device)
            pipe.enable_model_cpu_offload()

            init_image = Image.open(init_image).convert("RGB")
            init_image = init_image.resize((width, height))

            def combined_callback_2_2(pipe, i, t):
                nonlocal stop_idx
                if stop_signal and stop_idx is None:
                    stop_idx = i
                if i == stop_idx:
                    pipe._interrupt = True
                progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")

            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=generator,
                callback=combined_callback_2_2,
                callback_steps=1
            ).images[0]

        elif version == "3":
            pipe = AutoPipelineForImage2Image().AutoPipelineForImage2Image.from_pretrained(
                os.path.join(kandinsky_model_path, "3"), variant=variant, torch_dtype=torch_dtype
            )
            pipe.to(device)
            pipe.enable_model_cpu_offload()

            init_image = Image.open(init_image).convert("RGB")
            init_image = init_image.resize((width, height))

            def combined_callback_3_0(pipe, i, t):
                nonlocal stop_idx
                if stop_signal and stop_idx is None:
                    stop_idx = i
                if i == stop_idx:
                    pipe._interrupt = True
                progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")

            image = pipe(
                prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=generator,
                callback=combined_callback_3_0,
                callback_steps=1
            ).images[0]

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"Kandinsky_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"kandinsky_{version}_img2img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        return image_path, f"Image generated successfully. Seed used: {seed}"

    except Exception as e:
        if UnboundLocalError:
            pass
        else:
            gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe_prior' in locals():
            del pipe_prior
        if 'pipe' in locals():
            del pipe
        flush()


def generate_image_kandinsky_inpaint(prompt, negative_prompt, init_image, mask_image, version, stop_button, num_inference_steps, guidance_scale, strength, height, width, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    variant = "fp16" if torch_dtype == torch.float16 else "fp32"

    if not init_image or not mask_image:
        gr.Info("Please upload an initial image and provide a mask image!")
        return None, None

    kandinsky_model_path = os.path.join("inputs", "image", "kandinsky")

    if not os.path.exists(kandinsky_model_path):
        gr.Info(f"Downloading Kandinsky {version} model...")
        os.makedirs(kandinsky_model_path, exist_ok=True)
        if version == "2.1":
            Repo.clone_from("https://huggingface.co/kandinsky-community/kandinsky-2-1-inpainter",
                            os.path.join(kandinsky_model_path, "2-1-inpainter"))
        elif version == "2.2":
            Repo.clone_from("https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint",
                            os.path.join(kandinsky_model_path, "2-2-decoder-inpaint"))
        gr.Info(f"Kandinsky {version} inpainting model downloaded")

    try:

        if version == "2.1":
            pipe = AutoPipelineForInpainting().AutoPipelineForInpainting.from_pretrained(
                os.path.join(kandinsky_model_path, "2-1-inpainter"),
                torch_dtype=torch_dtype,
                variant=variant
            )
        else:
            pipe = AutoPipelineForInpainting().AutoPipelineForInpainting.from_pretrained(
                os.path.join(kandinsky_model_path, "2-2-decoder-inpaint"),
                torch_dtype=torch_dtype
            )

        pipe.to(device)
        pipe.enable_model_cpu_offload()

        init_image = Image.open(init_image).convert("RGB")
        init_image = init_image.resize((width, height))

        mask_image = Image.open(mask_image).convert("L")
        mask_image = mask_image.resize((width, height))

        def combined_callback_2_1_and_2_2(pipe, i, t):
            nonlocal stop_idx
            if stop_signal and stop_idx is None:
                stop_idx = i
            if i == stop_idx:
                pipe._interrupt = True
            progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            callback=combined_callback_2_1_and_2_2,
            callback_steps=1
        ).images[0]

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"Kandinsky_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"kandinsky_{version}_inpaint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        return image_path, None

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        flush()


def generate_image_flux_txt2img(prompt, model_name, quantize_model_name, enable_quantize, seed, stop_button, vae_model_name, lora_model_names, lora_scales, guidance_scale, height, width, num_inference_steps, max_sequence_length, enable_taesd, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.bfloat32

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    flux_model_path = os.path.join("inputs", "image", "flux", model_name)

    if not quantize_model_name:
        gr.Info("Please select a Flux safetensors/GGUF model!")
        return None, None

    try:
        if enable_quantize:
            params = {
                'prompt': prompt,
                'guidance_scale': guidance_scale,
                'height': height,
                'width': width,
                'num_inference_steps': num_inference_steps,
                'seed': seed,
                'quantize_flux_model_path': f"{quantize_model_name}"
            }

            env = os.environ.copy()
            env['PYTHONPATH'] = os.pathsep.join(sys.path)

            result = subprocess.run([sys.executable, 'inputs/image/flux/quantize-flux/flux-txt2img-quantize.py', json.dumps(params)],
                                    capture_output=True, text=True)

            if result.returncode != 0:
                gr.Info(f"Error in flux-txt2img-quantize.py: {result.stderr}")
                return None, None

            image_path = None
            output = result.stdout.strip()
            if "IMAGE_PATH:" in output:
                image_path = output.split("IMAGE_PATH:")[-1].strip()

            if not image_path:
                gr.Info("Image path not found in the output")
                return None, None

            if not os.path.exists(image_path):
                gr.Info(f"Generated image not found at {image_path}")
                return None, None

            return image_path, f"Image generated successfully. Seed used: {seed}"
        else:
            if not os.path.exists(flux_model_path):
                gr.Info(f"Downloading Flux {model_name} model...")
                os.makedirs(flux_model_path, exist_ok=True)
                Repo.clone_from(f"https://huggingface.co/black-forest-labs/{model_name}", flux_model_path)
                gr.Info(f"Flux {model_name} model downloaded")

            quantize_model_path = os.path.join("inputs", "image", "flux", "quantize-flux", quantize_model_name)

            transformer = FluxTransformer2DModel().FluxTransformer2DModel.from_single_file(quantize_model_path, torch_dtype=torch_dtype)

            text_encoder_2 = T5EncoderModel().T5EncoderModel.from_pretrained(flux_model_path, subfolder="text_encoder_2", torch_dtype=torch_dtype)

            pipe = FluxPipeline().FluxPipeline.from_pretrained(flux_model_path, transformer=None, text_encoder_2=None, torch_dtype=torch_dtype).to(device)
            pipe.transformer = transformer
            pipe.text_encoder_2 = text_encoder_2
            if enable_taesd:
                pipe.vae = AutoencoderTiny().AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=torch_dtype)
            if vae_model_name is not None:
                vae_model_path = os.path.join("inputs", "image", "flux", "flux-vae", f"{vae_model_name}")
                if os.path.exists(vae_model_path):
                    vae = AutoencoderKL().AutoencoderKL.from_single_file(vae_model_path, torch_dtype=torch_dtype)
                    pipe.vae = vae.to(device)

            pipe.enable_model_cpu_offload()
            pipe.enable_sequential_cpu_offload()
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
            pipe.to(torch_dtype)

            if isinstance(lora_scales, str):
                lora_scales = [float(scale.strip()) for scale in lora_scales.split(',') if scale.strip()]
            elif isinstance(lora_scales, (int, float)):
                lora_scales = [float(lora_scales)]

            lora_loaded = False
            if lora_model_names and lora_scales:
                if len(lora_model_names) != len(lora_scales):
                    gr.Warning(
                        f"Number of LoRA models ({len(lora_model_names)}) does not match number of scales ({len(lora_scales)}). Using available scales.", duration=5)

                for i, lora_model_name in enumerate(lora_model_names):
                    if i < len(lora_scales):
                        lora_scale = lora_scales[i]
                    else:
                        lora_scale = 1.0

                    lora_model_path = os.path.join("inputs", "image", "flux", "flux-lora", lora_model_name)
                    if os.path.exists(lora_model_path):
                        adapter_name = os.path.splitext(os.path.basename(lora_model_name))[0]
                        try:
                            pipe.load_lora_weights(lora_model_path, adapter_name=adapter_name)
                            pipe.fuse_lora(lora_scale=lora_scale)
                            lora_loaded = True
                            gr.Info(f"Loaded LoRA {lora_model_name} with scale {lora_scale}")
                        except Exception as e:
                            gr.Warning(f"Error loading LoRA {lora_model_name}: {str(e)}", duration=5)

            def combined_callback(pipe, i, t, callback_kwargs):
                nonlocal stop_idx
                if stop_signal and stop_idx is None:
                    stop_idx = i
                if i == stop_idx:
                    pipe._interrupt = True

                progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")

                return callback_kwargs

            output = pipe(
                prompt=prompt,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                max_sequence_length=max_sequence_length,
                generator=generator,
                callback_on_step_end=combined_callback,
                callback_on_step_end_tensor_inputs=["latents"]
            ).images[0]

            today = datetime.now().date()
            image_dir = os.path.join('outputs', f"Flux_{today.strftime('%Y%m%d')}")
            os.makedirs(image_dir, exist_ok=True)
            image_filename = f"flux_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
            image_path = os.path.join(image_dir, image_filename)
            metadata = {
                "prompt": prompt,
                "flux_model": model_name,
                "quantized_model": quantize_model_name if enable_quantize else None,
                "lora_models": lora_model_names if lora_model_names else None,
                "lora_scales": lora_scales if lora_model_names else None,
                "guidance_scale": guidance_scale,
                "height": height,
                "width": width,
                "steps": num_inference_steps,
                "max_sequence_length": max_sequence_length,
                "seed": seed
            }

            output.save(image_path, format=output_format.upper())
            add_metadata_to_file(image_path, metadata)

            return image_path, f"Image generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if enable_quantize:
            flush()
        else:
            if 'pipe' in locals():
                del pipe
        flush()


def generate_image_flux_img2img(prompt, init_image, model_name, quantize_model_name, enable_quantize, seed, stop_button, vae_model_name, lora_model_names, lora_scales, num_inference_steps, strength, guidance_scale, width, height, max_sequence_length, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    if not init_image:
        gr.Info("Please upload an initial image!")
        return None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.bfloat32

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    flux_model_path = os.path.join("inputs", "image", "flux", model_name)
    quantize_model_path = os.path.join("inputs", "image", "flux", "quantize-flux", quantize_model_name)

    if not quantize_model_name:
        gr.Info("Please select a Flux safetensors/GGUF model!")
        return None, None

    if enable_quantize:
        params = {
            'prompt': prompt,
            'guidance_scale': guidance_scale,
            'height': height,
            'width': width,
            'num_inference_steps': num_inference_steps,
            'seed': seed,
            'quantize_flux_model_path': quantize_model_path,
            'image': init_image,
            'strength': strength
        }

        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join(sys.path)

        result = subprocess.run([sys.executable, 'inputs/image/flux/quantize-flux/flux-img2img-quantize.py', json.dumps(params)],
                                capture_output=True, text=True)

        if result.returncode != 0:
            gr.Info(f"Error in flux-img2img-quantize.py: {result.stderr}")
            return None, None

        image_path = None
        output = result.stdout.strip()
        if "IMAGE_PATH:" in output:
            image_path = output.split("IMAGE_PATH:")[-1].strip()

        if not image_path:
            gr.Info("Image path not found in the output")
            return None, None

        if not os.path.exists(image_path):
            gr.Info(f"Generated image not found at {image_path}")
            return None, None

        return image_path, f"Image generated successfully. Seed used: {seed}"
    else:
        if not os.path.exists(flux_model_path):
            gr.Info(f"Downloading Flux {model_name} model...")
            os.makedirs(flux_model_path, exist_ok=True)
            Repo.clone_from(f"https://huggingface.co/black-forest-labs/{model_name}", flux_model_path)
            gr.Info(f"Flux {model_name} model downloaded")

        try:
            pipe = FluxImg2ImgPipeline().FluxImg2ImgPipeline.from_pretrained(flux_model_path, torch_dtype=torch_dtype).to(device)
            pipe.enable_model_cpu_offload()
            pipe.enable_sequential_cpu_offload()
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
            pipe.to(torch_dtype)

            if vae_model_name is not None:
                vae_model_path = os.path.join("inputs", "image", "flux", "flux-vae", f"{vae_model_name}")
                if os.path.exists(vae_model_path):
                    vae = AutoencoderKL().AutoencoderKL.from_single_file(vae_model_path, torch_dtype=torch_dtype)
                    pipe.vae = vae.to(device)

            init_image = Image.open(init_image).convert("RGB")
            init_image = init_image.resize((width, height))

            if isinstance(lora_scales, str):
                lora_scales = [float(scale.strip()) for scale in lora_scales.split(',') if scale.strip()]
            elif isinstance(lora_scales, (int, float)):
                lora_scales = [float(lora_scales)]

            lora_loaded = False
            if lora_model_names and lora_scales:
                if len(lora_model_names) != len(lora_scales):
                    gr.Warning(
                        f"Number of LoRA models ({len(lora_model_names)}) does not match number of scales ({len(lora_scales)}). Using available scales.", duration=5)

                for i, lora_model_name in enumerate(lora_model_names):
                    if i < len(lora_scales):
                        lora_scale = lora_scales[i]
                    else:
                        lora_scale = 1.0

                    lora_model_path = os.path.join("inputs", "image", "flux", "flux-lora", lora_model_name)
                    if os.path.exists(lora_model_path):
                        adapter_name = os.path.splitext(os.path.basename(lora_model_name))[0]
                        try:
                            pipe.load_lora_weights(lora_model_path, adapter_name=adapter_name)
                            pipe.fuse_lora(lora_scale=lora_scale)
                            lora_loaded = True
                            gr.Info(f"Loaded LoRA {lora_model_name} with scale {lora_scale}")
                        except Exception as e:
                            gr.Warning(f"Error loading LoRA {lora_model_name}: {str(e)}", duration=5)

            def combined_callback(pipe, i, t, callback_kwargs):
                nonlocal stop_idx
                if stop_signal and stop_idx is None:
                    stop_idx = i
                if i == stop_idx:
                    pipe._interrupt = True
                progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")
                return callback_kwargs

            image = pipe(
                prompt=prompt,
                image=init_image,
                num_inference_steps=num_inference_steps,
                strength=strength,
                guidance_scale=guidance_scale,
                generator=generator,
                max_sequence_length=max_sequence_length,
                callback_on_step_end=combined_callback,
                callback_on_step_end_tensor_inputs=["latents"]
            ).images[0]

            today = datetime.now().date()
            image_dir = os.path.join('outputs', f"Flux_{today.strftime('%Y%m%d')}")
            os.makedirs(image_dir, exist_ok=True)
            image_filename = f"flux_img2img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
            image_path = os.path.join(image_dir, image_filename)

            metadata = {
                "prompt": prompt,
                "model_name": model_name,
                "num_inference_steps": num_inference_steps,
                "strength": strength,
                "guidance_scale": guidance_scale,
                "max_sequence_length": max_sequence_length,
                "seed": seed,
                "model": "Flux"
            }

            image.save(image_path, format=output_format.upper())
            add_metadata_to_file(image_path, metadata)

            return image_path, f"Image generated successfully. Seed used: {seed}"

        except Exception as e:
            gr.Error(f"An error occurred: {str(e)}")
            return None, None

        finally:
            if 'pipe' in locals():
                del pipe
            flush()


def generate_image_flux_inpaint(prompt, init_image, mask_image, model_name, seed, stop_button, num_inference_steps, strength, guidance_scale,
                                max_sequence_length, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.bfloat32

    if not init_image or not mask_image:
        gr.Info("Please upload an initial image and provide a mask image!")
        return None, None

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    flux_model_path = os.path.join("inputs", "image", "flux", model_name)

    if not os.path.exists(flux_model_path):
        gr.Info(f"Downloading Flux {model_name} model...")
        os.makedirs(flux_model_path, exist_ok=True)
        Repo.clone_from(f"https://huggingface.co/black-forest-labs/{model_name}", flux_model_path)
        gr.Info(f"Flux {model_name} model downloaded")

    try:
        pipe = FluxInpaintPipeline().FluxInpaintPipeline.from_pretrained(flux_model_path, torch_dtype=torch_dtype).to(device)
        pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        pipe.to(torch_dtype)

        init_image = Image.open(init_image).convert("RGB")
        mask_image = Image.open(mask_image).convert("L")

        def combined_callback(pipe, i, t, callback_kwargs):
            nonlocal stop_idx
            if stop_signal and stop_idx is None:
                stop_idx = i
            if i == stop_idx:
                pipe._interrupt = True
            progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")
            return callback_kwargs

        image = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            generator=generator,
            max_sequence_length=max_sequence_length,
            callback_on_step_end=combined_callback,
            callback_on_step_end_tensor_inputs=["latents"]
        ).images[0]

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"Flux_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"flux_inpaint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)

        metadata = {
            "prompt": prompt,
            "model_name": model_name,
            "num_inference_steps": num_inference_steps,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "max_sequence_length": max_sequence_length,
            "seed": seed,
            "model": "Flux"
        }

        image.save(image_path, format=output_format.upper())
        add_metadata_to_file(image_path, metadata)

        return image_path, f"Image generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        flush()


def generate_image_flux_controlnet(prompt, init_image, base_model_name, seed, stop_button, control_mode, controlnet_conditioning_scale, num_inference_steps,
                                   guidance_scale, width, height, max_sequence_length, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.bfloat32

    if not init_image:
        gr.Info("Please upload an initial image!")
        return None, None

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    base_model_path = os.path.join("inputs", "image", "flux", base_model_name)
    controlnet_model_path = os.path.join("inputs", "image", "flux-controlnet")

    if not os.path.exists(base_model_path):
        gr.Info(f"Downloading Flux {base_model_name} model...")
        os.makedirs(base_model_path, exist_ok=True)
        Repo.clone_from(f"https://huggingface.co/black-forest-labs/{base_model_name}", base_model_path)
        gr.Info(f"Flux {base_model_name} model downloaded")

    if not os.path.exists(controlnet_model_path):
        gr.Info("Downloading Flux ControlNet model...")
        os.makedirs(controlnet_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union", controlnet_model_path)
        gr.Info("Flux ControlNet model downloaded")

    try:
        controlnet = FluxControlNetModel().FluxControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch_dtype)
        pipe = FluxControlNetPipeline().FluxControlNetPipeline.from_pretrained(base_model_path, controlnet=controlnet, torch_dtype=torch_dtype).to(device)
        pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        pipe.to(torch_dtype)

        init_image = Image.open(init_image).convert("RGB")

        def combined_callback(pipe, i, t, callback_kwargs):
            nonlocal stop_idx
            if stop_signal and stop_idx is None:
                stop_idx = i
            if i == stop_idx:
                pipe._interrupt = True
            progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")
            return callback_kwargs

        image = pipe(
            prompt=prompt,
            control_image=init_image,
            control_mode=control_mode,
            width=width,
            height=height,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            max_sequence_length=max_sequence_length,
            callback_on_step_end=combined_callback,
            callback_on_step_end_tensor_inputs=["latents"]
        ).images[0]

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"Flux_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"flux_controlnet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)

        metadata = {
            "prompt": prompt,
            "base_model_name": base_model_name,
            "control_mode": control_mode,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "max_sequence_length": max_sequence_length,
            "seed": seed,
            "model": "Flux ControlNet"
        }

        image.save(image_path, format=output_format.upper())
        add_metadata_to_file(image_path, metadata)

        return image_path, f"Image generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        if 'controlnet' in locals():
            del controlnet
        flush()


def generate_image_hunyuandit_txt2img(prompt, negative_prompt, seed, stop_button, num_inference_steps, guidance_scale, height, width, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    hunyuandit_model_path = os.path.join("inputs", "image", "hunyuandit")

    if not os.path.exists(hunyuandit_model_path):
        gr.Info("Downloading HunyuanDiT model...")
        os.makedirs(hunyuandit_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers", hunyuandit_model_path)
        gr.Info("HunyuanDiT model downloaded")

    try:
        pipe = HunyuanDiTPipeline().HunyuanDiTPipeline.from_pretrained(hunyuandit_model_path, torch_dtype=torch_dtype)
        pipe.to(device)
        pipe.transformer.enable_forward_chunking(chunk_size=1, dim=1)

        def combined_callback(pipe, i, t, callback_kwargs):
            nonlocal stop_idx
            if stop_signal and stop_idx is None:
                stop_idx = i
            if i == stop_idx:
                pipe._interrupt = True
            progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")
            return callback_kwargs

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            callback_on_step_end=combined_callback,
            callback_on_step_end_tensor_inputs=["latents"]
        ).images[0]

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"HunyuanDiT_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"hunyuandit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model": "HunyuanDiT",
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "seed": seed
        }

        image.save(image_path, format=output_format.upper())
        add_metadata_to_file(image_path, metadata)

        return image_path, f"Image generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        flush()


def generate_image_hunyuandit_controlnet(prompt, negative_prompt, init_image, controlnet_model, seed, stop_button, num_inference_steps, guidance_scale, height, width, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    if not init_image:
        gr.Info("Please upload an initial image!")
        return None, None

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    hunyuandit_model_path = os.path.join("inputs", "image", "hunyuandit")
    controlnet_model_path = os.path.join("inputs", "image", "hunyuandit", "controlnet", controlnet_model)

    if not os.path.exists(hunyuandit_model_path):
        gr.Info("Downloading HunyuanDiT model...")
        os.makedirs(hunyuandit_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers", hunyuandit_model_path)
        gr.Info("HunyuanDiT model downloaded")

    if not os.path.exists(controlnet_model_path):
        gr.Info(f"Downloading HunyuanDiT ControlNet {controlnet_model} model...")
        os.makedirs(controlnet_model_path, exist_ok=True)
        Repo.clone_from(f"https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-ControlNet-Diffusers-{controlnet_model}", controlnet_model_path)
        gr.Info(f"HunyuanDiT ControlNet {controlnet_model} model downloaded")

    try:
        controlnet = HunyuanDiT2DControlNetModel().HunyuanDiT2DControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch_dtype)
        pipe = HunyuanDiTControlNetPipeline().HunyuanDiTControlNetPipeline.from_pretrained(hunyuandit_model_path, controlnet=controlnet, torch_dtype=torch_dtype)
        pipe.to(device)

        init_image = Image.open(init_image).convert("RGB")
        init_image = init_image.resize((width, height))

        def combined_callback(pipe, i, t, callback_kwargs):
            nonlocal stop_idx
            if stop_signal and stop_idx is None:
                stop_idx = i
            if i == stop_idx:
                pipe._interrupt = True
            progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")
            return callback_kwargs

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            control_image=init_image,
            num_inference_steps=num_inference_steps,
            generator=generator,
            callback_on_step_end=combined_callback,
            callback_on_step_end_tensor_inputs=["latents"]
        ).images[0]

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"HunyuanDiT_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"hunyuandit_controlnet_{controlnet_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        return image_path, f"Image generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        if 'controlnet' in locals():
            del controlnet
        flush()


def generate_image_lumina(prompt, negative_prompt, seed, num_inference_steps, guidance_scale, height, width, max_sequence_length, output_format):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.bfloat32

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    lumina_model_path = os.path.join("inputs", "image", "lumina")

    if not os.path.exists(lumina_model_path):
        gr.Info("Downloading Lumina-T2X model...")
        os.makedirs(lumina_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/Alpha-VLLM/Lumina-Next-SFT-diffusers", lumina_model_path)
        gr.Info("Lumina-T2X model downloaded")

    try:
        pipe = LuminaText2ImgPipeline().LuminaText2ImgPipeline.from_pretrained(
            lumina_model_path, torch_dtype=torch_dtype
        ).to(device)
        pipe.enable_model_cpu_offload()

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            max_sequence_length=max_sequence_length,
            generator=generator
        ).images[0]

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"Lumina_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"lumina_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model": "Lumina-T2X",
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "max_sequence_length": max_sequence_length,
            "seed": seed
        }

        image.save(image_path, format=output_format.upper())
        add_metadata_to_file(image_path, metadata)

        return image_path, f"Image generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        flush()


def generate_image_kolors_txt2img(prompt, negative_prompt, seed, lora_model_names, lora_scales, guidance_scale, num_inference_steps, max_sequence_length, output_format):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    variant = "fp16" if torch_dtype == torch.float16 else "fp32"

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    kolors_model_path = os.path.join("inputs", "image", "kolors")

    if not os.path.exists(kolors_model_path):
        gr.Info("Downloading Kolors model...")
        os.makedirs(kolors_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/Kwai-Kolors/Kolors-diffusers", kolors_model_path)
        gr.Info("Kolors model downloaded")

    try:
        pipe = KolorsPipeline().KolorsPipeline.from_pretrained(kolors_model_path, torch_dtype=torch_dtype, variant=variant)
        pipe.to(device)
        pipe.enable_model_cpu_offload()

        if isinstance(lora_scales, str):
            lora_scales = [float(scale.strip()) for scale in lora_scales.split(',') if scale.strip()]
        elif isinstance(lora_scales, (int, float)):
            lora_scales = [float(lora_scales)]

        lora_loaded = False
        if lora_model_names and lora_scales:
            if len(lora_model_names) != len(lora_scales):
                gr.Warning(
                    f"Number of LoRA models ({len(lora_model_names)}) does not match number of scales ({len(lora_scales)}). Using available scales.", duration=5)

            for i, lora_model_name in enumerate(lora_model_names):
                if i < len(lora_scales):
                    lora_scale = lora_scales[i]
                else:
                    lora_scale = 1.0

                lora_model_path = os.path.join("inputs", "image", "kolors-lora", lora_model_name)
                if os.path.exists(lora_model_path):
                    adapter_name = os.path.splitext(os.path.basename(lora_model_name))[0]
                    try:
                        pipe.load_lora_weights(lora_model_path, adapter_name=adapter_name)
                        pipe.fuse_lora(lora_scale=lora_scale)
                        lora_loaded = True
                        gr.Info(f"Loaded LoRA {lora_model_name} with scale {lora_scale}")
                    except Exception as e:
                        gr.Warning(f"Error loading LoRA {lora_model_name}: {str(e)}", duration=5)

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=generator
        ).images[0]

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"Kolors_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"kolors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model": "Kolors",
            "lora_models": lora_model_names if lora_model_names else None,
            "lora_scales": lora_scales if lora_model_names else None,
            "guidance_scale": guidance_scale,
            "steps": num_inference_steps,
            "max_sequence_length": max_sequence_length,
            "seed": seed
        }

        image.save(image_path, format=output_format.upper())
        add_metadata_to_file(image_path, metadata)

        return image_path, f"Image generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        flush()


def generate_image_kolors_img2img(prompt, negative_prompt, init_image, seed, guidance_scale, num_inference_steps, max_sequence_length, output_format):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    variant = "fp16" if torch_dtype == torch.float16 else "fp32"

    if not init_image:
        gr.Info("Please upload an initial image!")
        return None, None

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    kolors_model_path = os.path.join("inputs", "image", "kolors")

    if not os.path.exists(kolors_model_path):
        gr.Info("Downloading Kolors model...")
        os.makedirs(kolors_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/Kwai-Kolors/Kolors-diffusers", kolors_model_path)
        gr.Info("Kolors model downloaded")

    try:
        pipe = KolorsPipeline().KolorsPipeline.from_pretrained(kolors_model_path, torch_dtype=torch_dtype, variant=variant)
        pipe.to(device)
        pipe.enable_model_cpu_offload()

        init_image = Image.open(init_image).convert("RGB")
        init_image = pipe.image_processor.preprocess(init_image)

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=generator
        ).images[0]

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"Kolors_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"kolors_img2img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        return image_path, f"Image generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        flush()


def generate_image_kolors_ip_adapter_plus(prompt, negative_prompt, ip_adapter_image, seed, guidance_scale, num_inference_steps, output_format):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    variant = "fp16" if torch_dtype == torch.float16 else "fp32"

    if not ip_adapter_image:
        gr.Info("Please upload an initial image!")
        return None, None

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    kolors_model_path = os.path.join("inputs", "image", "kolors")
    ip_adapter_path = os.path.join("inputs", "image", "kolors", "ip_adapter_plus")

    if not os.path.exists(kolors_model_path):
        gr.Info("Downloading Kolors model...")
        os.makedirs(kolors_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/Kwai-Kolors/Kolors-diffusers", kolors_model_path)
        gr.Info("Kolors model downloaded")

    if not os.path.exists(ip_adapter_path):
        gr.Info("Downloading Kolors IP-Adapter-Plus...")
        os.makedirs(ip_adapter_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-Plus", ip_adapter_path)
        gr.Info("Kolors IP-Adapter-Plus downloaded")

    try:
        image_encoder = CLIPVisionModelWithProjection().CLIPVisionModelWithProjection.from_pretrained(
            ip_adapter_path,
            subfolder="image_encoder",
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
        )
        pipe = KolorsPipeline().KolorsPipeline.from_pretrained(
            kolors_model_path, image_encoder=image_encoder, torch_dtype=torch_dtype, variant=variant
        )
        pipe.scheduler = DPMSolverMultistepScheduler().DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        pipe.load_ip_adapter(
            ip_adapter_path,
            subfolder="",
            weight_name="ip_adapter_plus_general.safetensors",
            image_encoder_folder=None,
        )
        pipe.enable_model_cpu_offload()

        ipa_image = Image.open(ip_adapter_image).convert("RGB")

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            ip_adapter_image=ipa_image,
            generator=generator
        ).images[0]

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"Kolors_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"kolors_ip_adapter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format=output_format.upper())

        return image_path, f"Image generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        if 'image_encoder' in locals():
            del image_encoder
        flush()


def generate_image_auraflow(prompt, negative_prompt, seed, lora_model_names, lora_scales, num_inference_steps, guidance_scale, height, width, max_sequence_length, enable_aurasr, output_format):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    auraflow_model_path = os.path.join("inputs", "image", "auraflow")
    aurasr_model_path = os.path.join("inputs", "image", "auraflow", "aurasr")

    if not os.path.exists(auraflow_model_path):
        gr.Info("Downloading AuraFlow model...")
        os.makedirs(auraflow_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/fal/AuraFlow-v0.3", auraflow_model_path)
        gr.Info("AuraFlow model downloaded")

    if not os.path.exists(aurasr_model_path):
        gr.Info("Downloading AuraSR model...")
        os.makedirs(aurasr_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/fal/AuraSR-v2", aurasr_model_path)
        gr.Info("AuraSR model downloaded")

    try:
        pipe = AuraFlowPipeline().AuraFlowPipeline.from_pretrained(auraflow_model_path, torch_dtype=torch_dtype)
        pipe = pipe.to(device)

        if isinstance(lora_scales, str):
            lora_scales = [float(scale.strip()) for scale in lora_scales.split(',') if scale.strip()]
        elif isinstance(lora_scales, (int, float)):
            lora_scales = [float(lora_scales)]

        lora_loaded = False
        if lora_model_names and lora_scales:
            if len(lora_model_names) != len(lora_scales):
                gr.Warning(
                    f"Number of LoRA models ({len(lora_model_names)}) does not match number of scales ({len(lora_scales)}). Using available scales.", duration=5)

            for i, lora_model_name in enumerate(lora_model_names):
                if i < len(lora_scales):
                    lora_scale = lora_scales[i]
                else:
                    lora_scale = 1.0

                lora_model_path = os.path.join("inputs", "image", "auraflow-lora", lora_model_name)
                if os.path.exists(lora_model_path):
                    adapter_name = os.path.splitext(os.path.basename(lora_model_name))[0]
                    try:
                        pipe.load_lora_weights(lora_model_path, adapter_name=adapter_name)
                        pipe.fuse_lora(lora_scale=lora_scale)
                        lora_loaded = True
                        gr.Info(f"Loaded LoRA {lora_model_name} with scale {lora_scale}")
                    except Exception as e:
                        gr.Warning(f"Error loading LoRA {lora_model_name}: {str(e)}", duration=5)

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            max_sequence_length=max_sequence_length,
            generator=generator
        ).images[0]

        if enable_aurasr:
            aura_sr = AuraSR().AuraSR.from_pretrained(aurasr_model_path)
            image = image.resize((256, 256))
            image = aura_sr.upscale_4x_overlapped(image)

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"AuraFlow_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"auraflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model": "AuraFlow",
            "lora_models": lora_model_names if lora_model_names else None,
            "lora_scales": lora_scales if lora_model_names else None,
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "max_sequence_length": max_sequence_length,
            "enable_aurasr": enable_aurasr if enable_aurasr else None,
            "seed": seed
        }

        image.save(image_path, format=output_format.upper())
        add_metadata_to_file(image_path, metadata)

        return image_path, f"Image generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        if 'aura_sr' in locals():
            del aura_sr
        flush()


def generate_image_wurstchen(prompt, negative_prompt, seed, stop_button, width, height, prior_steps, prior_guidance_scale, decoder_steps, decoder_guidance_scale, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    wurstchen_model_path = os.path.join("inputs", "image", "wurstchen")

    if not os.path.exists(wurstchen_model_path):
        gr.Info("Downloading Würstchen models...")
        os.makedirs(wurstchen_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/warp-ai/wuerstchen-prior", os.path.join(wurstchen_model_path, "prior"))
        Repo.clone_from("https://huggingface.co/warp-ai/wuerstchen", os.path.join(wurstchen_model_path, "decoder"))
        gr.Info("Würstchen models downloaded")

    try:

        prior_pipeline = WuerstchenPriorPipeline().WuerstchenPriorPipeline.from_pretrained(
            os.path.join(wurstchen_model_path, "prior"),
            torch_dtype=torch_dtype
        ).to(device)

        decoder_pipeline = WuerstchenDecoderPipeline().WuerstchenDecoderPipeline.from_pretrained(
            os.path.join(wurstchen_model_path, "decoder"),
            torch_dtype=torch_dtype
        ).to(device)

        prior_pipeline.prior = torch.compile(prior_pipeline.prior, mode="reduce-overhead", fullgraph=True)
        decoder_pipeline.decoder = torch.compile(decoder_pipeline.decoder, mode="reduce-overhead", fullgraph=True)

        def combined_callback_prior(prior_pipeline, i, t, callback_kwargs):
            nonlocal stop_idx
            if stop_signal and stop_idx is None:
                stop_idx = i
            if i == stop_idx:
                prior_pipeline._interrupt = True
            progress((i + 1) / prior_steps, f"Step {i + 1}/{prior_steps}")
            return callback_kwargs

        def combined_callback_decoder(decoder_pipeline, i, t, callback_kwargs):
            nonlocal stop_idx
            if stop_signal and stop_idx is None:
                stop_idx = i
            if i == stop_idx:
                decoder_pipeline._interrupt = True
            progress((i + 1) / decoder_steps, f"Step {i + 1}/{decoder_steps}")
            return callback_kwargs

        prior_output = prior_pipeline(
            prompt=prompt,
            height=height,
            width=width,
            timesteps=DEFAULT_STAGE_C_TIMESTEPS().DEFAULT_STAGE_C_TIMESTEPS,
            negative_prompt=negative_prompt,
            guidance_scale=prior_guidance_scale,
            num_inference_steps=prior_steps,
            generator=generator,
            callback_on_step_end=combined_callback_prior,
            callback_on_step_end_tensor_inputs=["latents"]
        )

        decoder_output = decoder_pipeline(
            image_embeddings=prior_output.image_embeddings,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=decoder_guidance_scale,
            num_inference_steps=decoder_steps,
            output_type="pil",
            generator=generator,
            callback_on_step_end=combined_callback_decoder,
            callback_on_step_end_tensor_inputs=["latents"]
        ).images[0]

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"Wurstchen_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"wurstchen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model": "Würstchen",
            "width": width,
            "height": height,
            "prior_steps": prior_steps,
            "prior_guidance_scale": prior_guidance_scale,
            "decoder_steps": decoder_steps,
            "decoder_guidance_scale": decoder_guidance_scale,
            "seed": seed
        }

        decoder_output.save(image_path, format=output_format.upper())
        add_metadata_to_file(image_path, metadata)

        return image_path, f"Image generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'prior_pipeline' in locals():
            del prior_pipeline
        if 'decoder_pipeline' in locals():
            del decoder_pipeline
        flush()


def generate_image_deepfloyd_txt2img(prompt, negative_prompt, seed, stop_button, num_inference_steps, guidance_scale, width, height, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    variant = "fp16" if torch_dtype == torch.float16 else "fp32"

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    deepfloydI_model_path = os.path.join("inputs", "image", "deepfloydI")
    deepfloydII_model_path = os.path.join("inputs", "image", "deepfloydII")
    upscale_model_path = os.path.join("inputs", "image", "sd_models", "upscale", "x4-upscaler")

    if not os.path.exists(deepfloydI_model_path):
        gr.Info("Downloading DeepfloydIF-I-XL-v1.0 model")
        os.makedirs(deepfloydI_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/DeepFloyd/IF-I-XL-v1.0", os.path.join(deepfloydI_model_path))
        gr.Info("DeepfloydIF-I-XL-v1.0 model downloaded")

    if not os.path.exists(deepfloydII_model_path):
        gr.Info("Downloading DeepFloydIF-II-L-v1.0 model")
        os.makedirs(deepfloydII_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/DeepFloyd/IF-II-L-v1.0", os.path.join(deepfloydII_model_path))
        gr.Info("DeepFloydIF-II-L-v1.0 model downloaded")

    if not os.path.exists(upscale_model_path):
        gr.Info("Downloading 4x-Upscale models...")
        os.makedirs(upscale_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler", os.path.join(upscale_model_path))
        gr.Info("Deepfloyd models downloaded")

    try:

        pipe_i = IFPipeline().IFPipeline.from_pretrained(deepfloydI_model_path, variant=variant, torch_dtype=torch_dtype)
        text_encoder = T5EncoderModel().T5EncoderModel.from_pretrained(
            deepfloydI_model_path, subfolder="text_encoder", device_map="auto", load_in_8bit=True, variant="8bit"
        )
        pipe_i.to(device)
        pipe_i.enable_model_cpu_offload()
        pipe_i.enable_sequential_cpu_offload()
        pipe_i.text_encoder = torch.compile(pipe_i.text_encoder, mode="reduce-overhead", fullgraph=True)
        pipe_i.unet = torch.compile(pipe_i.unet, mode="reduce-overhead", fullgraph=True)

        def combined_callback_pipe_i(pipe_i, i, t):
            nonlocal stop_idx
            if stop_signal and stop_idx is None:
                stop_idx = i
            if i == stop_idx:
                pipe_i._interrupt = True
            progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")

        image = pipe_i(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            output_type="pt",
            text_encoder=text_encoder,
            generator=generator,
            callback=combined_callback_pipe_i,
            callback_steps=1
        ).images

        pipe_ii = IFSuperResolutionPipeline().IFSuperResolutionPipeline.from_pretrained(
            deepfloydII_model_path, text_encoder=None, variant=variant, torch_dtype=torch_dtype
        )
        pipe_ii.to(device)
        pipe_ii.enable_model_cpu_offload()
        pipe_ii.enable_sequential_cpu_offload()

        def combined_callback_pipe_ii(pipe_ii, i, t):
            nonlocal stop_idx
            if stop_signal and stop_idx is None:
                stop_idx = i
            if i == stop_idx:
                pipe_ii._interrupt = True
            progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")

        image = pipe_ii(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="pt",
            generator=generator,
            callback=combined_callback_pipe_ii,
            callback_steps=1
        ).images

        safety_modules = {
            "feature_extractor": pipe_i.feature_extractor,
            "safety_checker": pipe_i.safety_checker,
            "watermarker": pipe_i.watermarker,
        }
        pipe_iii = DiffusionPipeline().DiffusionPipeline.from_pretrained(
            upscale_model_path, **safety_modules, torch_dtype=torch_dtype
        )
        pipe_iii.to(device)
        pipe_iii.enable_model_cpu_offload()
        pipe_iii.enable_sequential_cpu_offload()

        image = pipe_iii(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

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
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model": "DeepFloydIF",
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": seed
        }

        image.save(stage_iii_path, format=output_format.upper())
        add_metadata_to_file(stage_iii_path, metadata)

        return stage_i_path, stage_ii_path, stage_iii_path, f"Image generated successfully. Seed used: {seed}"

    except Exception as e:
        if UnboundLocalError:
            pass
        else:
            gr.Error(f"An error occurred: {str(e)}")
        return None, None, None, None

    finally:
        if 'pipe_i' in locals():
            del pipe_i
        if 'pipe_ii' in locals():
            del pipe_ii
        if 'pipe_iii' in locals():
            del pipe_iii
        if 'text_encoder' in locals():
            del text_encoder
        flush()


def generate_image_deepfloyd_img2img(prompt, negative_prompt, init_image, seed, stop_button, num_inference_steps, guidance_scale, width, height, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    variant = "fp16" if torch_dtype == torch.float16 else "fp32"

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    if not init_image:
        gr.Info("Please upload an initial image!")
        return None, None, None, None

    deepfloydI_model_path = os.path.join("inputs", "image", "deepfloydI")
    deepfloydII_model_path = os.path.join("inputs", "image", "deepfloydII")
    upscale_model_path = os.path.join("inputs", "image", "sd_models", "upscale", "x4-upscaler")

    if not os.path.exists(deepfloydI_model_path):
        gr.Info("Downloading DeepfloydIF-I-XL-v1.0 model")
        os.makedirs(deepfloydI_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/DeepFloyd/IF-I-XL-v1.0", os.path.join(deepfloydI_model_path))
        gr.Info("DeepfloydIF-I-XL-v1.0 model downloaded")

    if not os.path.exists(deepfloydII_model_path):
        gr.Info("Downloading DeepFloydIF-II-L-v1.0 model")
        os.makedirs(deepfloydII_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/DeepFloyd/IF-II-L-v1.0", os.path.join(deepfloydII_model_path))
        gr.Info("DeepFloydIF-II-L-v1.0 model downloaded")

    if not os.path.exists(upscale_model_path):
        gr.Info("Downloading 4x-Upscale models...")
        os.makedirs(upscale_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler",
                        os.path.join(upscale_model_path))
        gr.Info("Deepfloyd models downloaded")

    try:

        stage_1 = IFImg2ImgPipeline().IFImg2ImgPipeline.from_pretrained(deepfloydI_model_path, variant=variant, torch_dtype=torch_dtype)
        text_encoder = T5EncoderModel().T5EncoderModel.from_pretrained(
            deepfloydI_model_path, subfolder="text_encoder", device_map="auto", load_in_8bit=True, variant="8bit"
        )
        stage_1.to(device)
        stage_1.enable_model_cpu_offload()
        stage_1.enable_sequential_cpu_offload()
        stage_1.text_encoder = torch.compile(stage_1.text_encoder, mode="reduce-overhead", fullgraph=True)
        stage_1.unet = torch.compile(stage_1.unet, mode="reduce-overhead", fullgraph=True)

        stage_2 = IFImg2ImgSuperResolutionPipeline().IFImg2ImgSuperResolutionPipeline.from_pretrained(
            deepfloydII_model_path, text_encoder=None, variant=variant, torch_dtype=torch_dtype
        )
        stage_2.to(device)
        stage_2.enable_model_cpu_offload()
        stage_2.enable_sequential_cpu_offload()

        safety_modules = {
            "feature_extractor": stage_1.feature_extractor,
            "safety_checker": stage_1.safety_checker,
            "watermarker": stage_1.watermarker,
        }
        stage_3 = DiffusionPipeline().DiffusionPipeline.from_pretrained(
            upscale_model_path, **safety_modules, torch_dtype=torch_dtype
        )
        stage_3.to(device)
        stage_3.enable_model_cpu_offload()
        stage_3.enable_sequential_cpu_offload()

        original_image = Image.open(init_image).convert("RGB")
        original_image = original_image.resize((width, height))

        def combined_callback_stage_1(stage_1, i, t):
            nonlocal stop_idx
            if stop_signal and stop_idx is None:
                stop_idx = i
            if i == stop_idx:
                stage_1._interrupt = True
            progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")

        def combined_callback_stage_2(stage_2, i, t):
            nonlocal stop_idx
            if stop_signal and stop_idx is None:
                stop_idx = i
            if i == stop_idx:
                stage_2._interrupt = True
            progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")

        stage_1_output = stage_1(
            image=original_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="pt",
            text_encoder=text_encoder,
            generator=generator,
            callback=combined_callback_stage_1,
            callback_steps=1
        ).images

        stage_2_output = stage_2(
            image=stage_1_output,
            original_image=original_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="pt",
            generator=generator,
            callback=combined_callback_stage_2,
            callback_steps=1
        ).images

        stage_3_output = stage_3(
            prompt=prompt,
            image=stage_2_output,
            noise_level=100,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

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

        return stage_1_path, stage_2_path, stage_3_path, f"Image generated successfully. Seed used: {seed}"

    except Exception as e:
        if UnboundLocalError:
            pass
        else:
            gr.Error(f"An error occurred: {str(e)}")
        return None, None, None, None

    finally:
        if 'stage_1' in locals():
            del stage_1
        if 'stage_2' in locals():
            del stage_2
        if 'stage_3' in locals():
            del stage_3
        if 'text_encoder' in locals():
            del text_encoder
        flush()


def generate_image_deepfloyd_inpaint(prompt, negative_prompt, init_image, mask_image, seed, stop_button, num_inference_steps, guidance_scale, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    variant = "fp16" if torch_dtype == torch.float16 else "fp32"

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    if not init_image or not mask_image:
        gr.Info("Please upload an initial image and provide a mask image!")
        return None, None, None, None

    deepfloydI_model_path = os.path.join("inputs", "image", "deepfloydI")
    deepfloydII_model_path = os.path.join("inputs", "image", "deepfloydII")
    upscale_model_path = os.path.join("inputs", "image", "sd_models", "upscale", "x4-upscaler")

    if not os.path.exists(deepfloydI_model_path):
        gr.Info("Downloading DeepfloydIF-I-XL-v1.0 model")
        os.makedirs(deepfloydI_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/DeepFloyd/IF-I-XL-v1.0", os.path.join(deepfloydI_model_path))
        gr.Info("DeepfloydIF-I-XL-v1.0 model downloaded")

    if not os.path.exists(deepfloydII_model_path):
        gr.Info("Downloading DeepFloydIF-II-L-v1.0 model")
        os.makedirs(deepfloydII_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/DeepFloyd/IF-II-L-v1.0", os.path.join(deepfloydII_model_path))
        gr.Info("DeepFloydIF-II-L-v1.0 model downloaded")

    if not os.path.exists(upscale_model_path):
        gr.Info("Downloading 4x-Upscale models...")
        os.makedirs(upscale_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler",
                        os.path.join(upscale_model_path))
        gr.Info("Deepfloyd models downloaded")

    try:

        stage_1 = IFInpaintingPipeline().IFInpaintingPipeline.from_pretrained(deepfloydI_model_path, variant=variant, torch_dtype=torch_dtype)
        text_encoder = T5EncoderModel().T5EncoderModel.from_pretrained(
            deepfloydI_model_path, subfolder="text_encoder", device_map="auto", load_in_8bit=True, variant="8bit"
        )
        stage_1.to(device)
        stage_1.enable_model_cpu_offload()
        stage_1.enable_sequential_cpu_offload()
        stage_1.text_encoder = torch.compile(stage_1.text_encoder, mode="reduce-overhead", fullgraph=True)
        stage_1.unet = torch.compile(stage_1.unet, mode="reduce-overhead", fullgraph=True)

        stage_2 = IFInpaintingSuperResolutionPipeline().IFInpaintingSuperResolutionPipeline.from_pretrained(
            deepfloydII_model_path, text_encoder=None, variant=variant, torch_dtype=torch_dtype
        )
        stage_2.to(device)
        stage_2.enable_model_cpu_offload()
        stage_2.enable_sequential_cpu_offload()

        safety_modules = {
            "feature_extractor": stage_1.feature_extractor,
            "safety_checker": stage_1.safety_checker,
            "watermarker": stage_1.watermarker,
        }
        stage_3 = DiffusionPipeline().DiffusionPipeline.from_pretrained(
            upscale_model_path, **safety_modules, torch_dtype=torch_dtype
        )
        stage_3.to(device)
        stage_3.enable_model_cpu_offload()
        stage_3.enable_sequential_cpu_offload()

        original_image = Image.open(init_image).convert("RGB")
        mask_image = Image.open(mask_image).convert("RGB")

        def combined_callback_stage_1(stage_1, i, t):
            nonlocal stop_idx
            if stop_signal and stop_idx is None:
                stop_idx = i
            if i == stop_idx:
                stage_1._interrupt = True
            progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")

        def combined_callback_stage_2(stage_2, i, t):
            nonlocal stop_idx
            if stop_signal and stop_idx is None:
                stop_idx = i
            if i == stop_idx:
                stage_2._interrupt = True
            progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")

        stage_1_output = stage_1(
            image=original_image,
            mask_image=mask_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="pt",
            text_encoder=text_encoder,
            generator=generator,
            callback=combined_callback_stage_1,
            callback_steps=1
        ).images

        stage_2_output = stage_2(
            image=stage_1_output,
            original_image=original_image,
            mask_image=mask_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="pt",
            generator=generator,
            callback=combined_callback_stage_2,
            callback_steps=1
        ).images

        stage_3_output = stage_3(
            prompt=prompt,
            image=stage_2_output,
            noise_level=100,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

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

        return stage_1_path, stage_2_path, stage_3_path, f"Image generated successfully. Seed used: {seed}"

    except Exception as e:
        if UnboundLocalError:
            pass
        else:
            gr.Error(f"An error occurred: {str(e)}")
        return None, None, None, None

    finally:
        if 'stage_1' in locals():
            del stage_1
        if 'stage_2' in locals():
            del stage_2
        if 'stage_3' in locals():
            del stage_3
        if 'text_encoder' in locals():
            del text_encoder
        flush()


def generate_image_pixart(prompt, negative_prompt, version, seed, stop_button, num_inference_steps, guidance_scale, height, width,
                          max_sequence_length, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    pixart_model_path = os.path.join("inputs", "image", "pixart")

    if not os.path.exists(pixart_model_path):
        gr.Info(f"Downloading PixArt {version} model...")
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
        gr.Info(f"PixArt {version} model downloaded")

    try:
        if version.startswith("Alpha"):
            text_encoder = T5EncoderModel().T5EncoderModel.from_pretrained(
                pixart_model_path,
                subfolder="text_encoder",
                load_in_8bit=True,
                device_map="auto",

            )
            pipe = PixArtAlphaPipeline().PixArtAlphaPipeline.from_pretrained(os.path.join(pixart_model_path, version),
                                                       torch_dtype=torch_dtype, text_encoder=text_encoder).to(device)
        else:
            text_encoder = T5EncoderModel().T5EncoderModel.from_pretrained(
                pixart_model_path,
                subfolder="text_encoder",
                load_in_8bit=True,
                device_map="auto",

            )
            pipe = PixArtSigmaPipeline().PixArtSigmaPipeline.from_pretrained(os.path.join(pixart_model_path, version),
                                                       torch_dtype=torch_dtype, text_encoder=text_encoder).to(device)

        pipe.enable_model_cpu_offload()

        def combined_callback(pipe, i, t):
            nonlocal stop_idx
            if stop_signal and stop_idx is None:
                stop_idx = i
            if i == stop_idx:
                pipe._interrupt = True
            progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            max_sequence_length=max_sequence_length,
            generator=generator,
            callback=combined_callback,
            callback_steps=1
        ).images[0]

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"PixArt_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"pixart_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "pixart_version": version,
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "max_sequence_length": max_sequence_length,
            "seed": seed
        }

        image.save(image_path, format=output_format.upper())
        add_metadata_to_file(image_path, metadata)

        return image_path, f"Image generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        if 'text_encoder' in locals():
            del text_encoder
        flush()


def generate_image_cogview3plus(prompt, negative_prompt, seed, stop_button, height, width, num_inference_steps, guidance_scale, max_sequence_length, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.bfloat32

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    cogview3plus_model_path = os.path.join("inputs", "image", "cogview3plus")

    if not os.path.exists(cogview3plus_model_path):
        gr.Info("Downloading CogView3-Plus model...")
        os.makedirs(cogview3plus_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/THUDM/CogView3-Plus-3B", cogview3plus_model_path)
        gr.Info("CogView3-Plus model downloaded")

    try:
        pipe = CogView3PlusPipeline().CogView3PlusPipeline.from_pretrained(
            cogview3plus_model_path,
            torch_dtype=torch_dtype
        ).to(device)
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        def combined_callback(pipe, i, t, callback_kwargs):
            nonlocal stop_idx
            if stop_signal and stop_idx is None:
                stop_idx = i
            if i == stop_idx:
                pipe._interrupt = True

            progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")

            return callback_kwargs

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
            generator=generator,
            callback_on_step_end=combined_callback,
            callback_on_step_end_tensor_inputs=["latents"]
        ).images[0]

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"Cogview3plus_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"Cogview3plus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model": "Cogview3plus",
            "height": height,
            "width": width,
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed
        }

        image.save(image_path, format=output_format.upper())
        add_metadata_to_file(image_path, metadata)

        return image_path, f"Image generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        flush()


def generate_image_playgroundv2(prompt, negative_prompt, seed, height, width, num_inference_steps, guidance_scale, output_format):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    variant = "fp16" if torch_dtype == torch.float16 else "fp32"

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    playgroundv2_model_path = os.path.join("inputs", "image", "playgroundv2")

    if not os.path.exists(playgroundv2_model_path):
        gr.Info("Downloading PlaygroundV2.5 model...")
        os.makedirs(playgroundv2_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic", playgroundv2_model_path)
        gr.Info("PlaygroundV2.5 model downloaded")

    try:
        pipe = DiffusionPipeline().DiffusionPipeline.from_pretrained(
            playgroundv2_model_path,
            torch_dtype=torch_dtype,
            variant=variant
        ).to(device)

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"PlaygroundV2_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"playgroundv2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        image_path = os.path.join(image_dir, image_filename)
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model": "PlaygroundV2.5",
            "height": height,
            "width": width,
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed
        }

        image.save(image_path, format=output_format.upper())
        add_metadata_to_file(image_path, metadata)

        return image_path, f"Image generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        flush()


def generate_wav2lip(image_path, audio_path, fps, pads, face_det_batch_size, wav2lip_batch_size, resize_factor, crop, enable_no_smooth, progress=gr.Progress()):

    if not image_path or not audio_path:
        gr.Info("Please upload an image and an audio file!")
        return None, None

    try:
        progress(0.1, desc="Initializing Wav2Lip")
        wav2lip_path = os.path.join("inputs", "image", "Wav2Lip")

        checkpoint_path = os.path.join(wav2lip_path, "checkpoints", "wav2lip_gan.pth")

        if not os.path.exists(checkpoint_path):
            progress(0.2, desc="Downloading Wav2Lip GAN model...")
            gr.Info("Downloading Wav2Lip GAN model...")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            url = "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth"
            response = requests.get(url, allow_redirects=True)
            with open(checkpoint_path, "wb") as file:
                file.write(response.content)
            gr.Info("Wav2Lip GAN model downloaded")

        progress(0.3, desc="Preparing output directory")
        today = datetime.now().date()
        output_dir = os.path.join("outputs", f"FaceAnimation_{today.strftime('%Y%m%d')}")
        os.makedirs(output_dir, exist_ok=True)

        output_filename = f"face_animation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_path = os.path.join(output_dir, output_filename)

        progress(0.4, desc="Constructing command")
        command = f"py {os.path.join(wav2lip_path, 'inference.py')} --checkpoint_path {checkpoint_path} --face {image_path} --audio {audio_path} --outfile {output_path} --fps {fps} --pads {pads} --face_det_batch_size {face_det_batch_size} --wav2lip_batch_size {wav2lip_batch_size} --resize_factor {resize_factor} --crop {crop} --box {-1}"
        if enable_no_smooth:
            command += " --no-smooth"

        progress(0.5, desc="Running Wav2Lip")
        subprocess.run(command, shell=True, check=True)

        progress(1.0, desc="Wav2Lip process complete")
        return output_path, None

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        flush()


def generate_liveportrait(source_image, driving_video, output_format, progress=gr.Progress()):
    if not source_image or not driving_video:
        gr.Info("Please upload both a source image and a driving video!")
        return None, None

    liveportrait_model_path = os.path.join("ThirdPartyRepository", "LivePortrait", "pretrained_weights")

    progress(0.1, desc="Checking LivePortrait model")
    if not os.path.exists(liveportrait_model_path):
        progress(0.2, desc="Downloading LivePortrait model...")
        gr.Info("Downloading LivePortrait model...")
        os.makedirs(liveportrait_model_path, exist_ok=True)
        os.system(
            f"huggingface-cli download KwaiVGI/LivePortrait --local-dir {liveportrait_model_path} --exclude '*.git*' 'README.md' 'docs'")
        gr.Info("LivePortrait model downloaded")

    try:
        progress(0.3, desc="Preparing output directory")
        today = datetime.now().date()
        output_dir = os.path.join('outputs', f"LivePortrait_{today.strftime('%Y%m%d')}")
        os.makedirs(output_dir, exist_ok=True)

        output_filename = f"liveportrait_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_path = os.path.join(output_dir, output_filename)

        progress(0.4, desc="Constructing command")
        command = f"python ThirdPartyRepository/LivePortrait/inference.py -s {source_image} -d {driving_video} -o {output_path}"

        progress(0.5, desc="Running LivePortrait")
        subprocess.run(command, shell=True, check=True)

        progress(0.8, desc="Processing output")
        result_folder = [f for f in os.listdir(output_dir) if
                         f.startswith(output_filename) and os.path.isdir(os.path.join(output_dir, f))]
        if not result_folder:
            gr.Info("Output folder not found")
            return None, None

        result_folder = os.path.join(output_dir, result_folder[0])

        video_files = [f for f in os.listdir(result_folder) if f.endswith(f'.{output_format}') and 'concat' not in f]
        if not video_files:
            gr.Info("Output video not found")
            return None, None

        output_video = os.path.join(result_folder, video_files[0])

        progress(1.0, desc="LivePortrait process complete")
        return output_video, None

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        flush()


def generate_video_modelscope(prompt, negative_prompt, seed, stop_button, num_inference_steps, guidance_scale, height, width, num_frames,
                              output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    variant = "fp16" if torch_dtype == torch.float16 else "fp32"

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    modelscope_model_path = os.path.join("inputs", "video", "modelscope")

    if not os.path.exists(modelscope_model_path):
        gr.Info("Downloading ModelScope model...")
        os.makedirs(modelscope_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/damo-vilab/text-to-video-ms-1.7b", modelscope_model_path)
        gr.Info("ModelScope model downloaded")

    try:
        pipe = DiffusionPipeline().DiffusionPipeline.from_pretrained(modelscope_model_path, torch_dtype=torch_dtype, variant=variant)
        pipe = pipe.to(device)
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()

        def combined_callback(pipe, i, t):
            nonlocal stop_idx
            if stop_signal and stop_idx is None:
                stop_idx = i
            if i == stop_idx:
                pipe._interrupt = True
            progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")

        video_frames = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_frames=num_frames,
            generator=generator,
            callback=combined_callback,
            callback_steps=1
        ).frames[0]

        today = datetime.now().date()
        video_dir = os.path.join('outputs', f"ModelScope_{today.strftime('%Y%m%d')}")
        os.makedirs(video_dir, exist_ok=True)

        video_filename = f"modelscope_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        video_path = os.path.join(video_dir, video_filename)

        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "model": "ModelScope"
        }

        export_to_video(video_frames, video_path)
        add_metadata_to_file(video_path, metadata)

        return video_path, f"Video generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        flush()


def generate_video_zeroscope2(prompt, video_to_enhance, seed, stop_button, strength, num_inference_steps, width, height, num_frames,
                              enable_video_enhance, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    base_model_path = os.path.join("inputs", "video", "zeroscope2", "zeroscope_v2_576w")
    if not os.path.exists(base_model_path):
        gr.Info("Downloading ZeroScope 2 base model...")
        os.makedirs(base_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/cerspense/zeroscope_v2_576w", base_model_path)
        gr.Info("ZeroScope 2 base model downloaded")

    enhance_model_path = os.path.join("inputs", "video", "zeroscope2", "zeroscope_v2_XL")
    if not os.path.exists(enhance_model_path):
        gr.Info("Downloading ZeroScope 2 enhance model...")
        os.makedirs(enhance_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/cerspense/zeroscope_v2_XL", enhance_model_path)
        gr.Info("ZeroScope 2 enhance model downloaded")

    today = datetime.now().date()
    video_dir = os.path.join('outputs', f"ZeroScope2_{today.strftime('%Y%m%d')}")
    os.makedirs(video_dir, exist_ok=True)

    if enable_video_enhance:
        if not video_to_enhance:
            gr.Info("Please upload a video to enhance.")
            return None, None

        try:
            enhance_pipe = DiffusionPipeline().DiffusionPipeline.from_pretrained(enhance_model_path, torch_dtype=torch_dtype)
            enhance_pipe.to(device)
            enhance_pipe.enable_model_cpu_offload()
            enhance_pipe.enable_vae_slicing()

            cap = cv2.VideoCapture(video_to_enhance)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame).resize((width, height))
                frames.append(frame)
            cap.release()

            def combined_callback_enhance(enhance_pipe, i, t):
                nonlocal stop_idx
                if stop_signal and stop_idx is None:
                    stop_idx = i
                if i == stop_idx:
                    enhance_pipe._interrupt = True
                progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")

            video_frames = enhance_pipe(prompt, video=frames, strength=strength, generator=generator, callback=combined_callback_enhance, callback_steps=1).frames

            processed_frames = []
            for frame in video_frames:
                if isinstance(frame, Image.Image):
                    frame = np.array(frame)

                if frame.ndim == 2:
                    frame = np.stack((frame,) * 3, axis=-1)
                elif frame.shape[-1] == 4:
                    frame = frame[:, :, :3]
                elif frame.shape[-1] != 3:
                    raise ValueError(f"Unexpected number of channels: {frame.shape[-1]}")

                frame = (frame * 255).astype(np.uint8)

                processed_frames.append(frame)

            video_filename = f"zeroscope2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
            video_path = os.path.join(video_dir, video_filename)
            metadata = {
                "prompt": prompt,
                "seed": seed,
                "strength": strength,
                "steps": num_inference_steps,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "enable_video_enhance": enable_video_enhance,
                "model": "ZeroScope 2"
            }

            export_to_video(video_frames, video_path)
            add_metadata_to_file(video_path, metadata)

            return video_path, f"Video generated successfully. Seed used: {seed}"

        except Exception as e:
            if UnboundLocalError:
                pass
            else:
                gr.Error(f"An error occurred: {str(e)}")
            return None, None

        finally:
            if 'enhance_pipe' in locals():
                del enhance_pipe
            flush()

    else:
        try:
            base_pipe = DiffusionPipeline().DiffusionPipeline.from_pretrained(base_model_path, torch_dtype=torch_dtype)
            base_pipe.to(device)
            base_pipe.enable_model_cpu_offload()
            base_pipe.enable_vae_slicing()
            base_pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)

            def combined_callback_base(base_pipe, i, t):
                nonlocal stop_idx
                if stop_signal and stop_idx is None:
                    stop_idx = i
                if i == stop_idx:
                    base_pipe._interrupt = True
                progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")

            video_frames = base_pipe(prompt, num_inference_steps=num_inference_steps, width=width, height=height, num_frames=num_frames, generator=generator, callback=combined_callback_base, callback_steps=1).frames[0]

            video_filename = f"zeroscope2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
            video_path = os.path.join(video_dir, video_filename)
            metadata = {
                "prompt": prompt,
                "seed": seed,
                "steps": num_inference_steps,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "model": "ZeroScope 2"
            }

            export_to_video(video_frames, video_path)
            add_metadata_to_file(video_path, metadata)

            return video_path, f"Video generated successfully. Seed used: {seed}"

        except Exception as e:
            if UnboundLocalError:
                pass
            else:
                gr.Error(f"An error occurred: {str(e)}")
            return None, None

        finally:
            if 'base_pipe' in locals():
                del base_pipe
            flush()


def generate_video_cogvideox_text2video(prompt, negative_prompt, cogvideox_version, seed, stop_button, num_inference_steps, guidance_scale, height, width, num_frames, fps, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    cogvideox_model_path = os.path.join("inputs", "video", "cogvideox", cogvideox_version)

    if not os.path.exists(cogvideox_model_path):
        gr.Info(f"Downloading {cogvideox_version} model...")
        os.makedirs(cogvideox_model_path, exist_ok=True)
        Repo.clone_from(f"https://huggingface.co/THUDM/{cogvideox_version}", cogvideox_model_path)
        gr.Info(f"{cogvideox_version} model downloaded")

    try:
        pipe = CogVideoXPipeline().CogVideoXPipeline.from_pretrained(cogvideox_model_path, torch_dtype=torch_dtype).to(device)
        pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        def combined_callback(pipe, i, t, callback_kwargs):
            nonlocal stop_idx
            if stop_signal and stop_idx is None:
                stop_idx = i
            if i == stop_idx:
                pipe._interrupt = True
            progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")
            return callback_kwargs

        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_frames=num_frames,
            generator=generator,
            callback_on_step_end=combined_callback,
            callback_on_step_end_tensor_inputs=["latents"]
        ).frames[0]

        today = datetime.now().date()
        video_dir = os.path.join('outputs', f"CogVideoX_{today.strftime('%Y%m%d')}")
        os.makedirs(video_dir, exist_ok=True)

        video_filename = f"cogvideox_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        video_path = os.path.join(video_dir, video_filename)
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "cogvideox_version": cogvideox_version,
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "fps": fps
        }

        export_to_video(video, video_path, fps=fps)
        add_metadata_to_file(video_path, metadata)

        return video_path, f"Video generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        flush()


def generate_video_cogvideox_image2video(prompt, negative_prompt, init_image, seed, stop_button, guidance_scale, num_inference_steps, fps, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.bfloat32

    if not init_image:
        gr.Info("Please, upload an initial image!")
        return None, None

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    cogvideox_i2v_model_path = os.path.join("inputs", "video", "cogvideox-i2v")

    if not os.path.exists(cogvideox_i2v_model_path):
        gr.Info(f"Downloading CogVideoX I2V model...")
        os.makedirs(cogvideox_i2v_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/THUDM/CogVideoX-5b-I2V", cogvideox_i2v_model_path)
        gr.Info(f"CogVideoX I2V model downloaded")

    try:
        pipe = CogVideoXImageToVideoPipeline().CogVideoXImageToVideoPipeline.from_pretrained(cogvideox_i2v_model_path, torch_dtype=torch_dtype)
        pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        image = load_image(init_image)

        def combined_callback(pipe, i, t, callback_kwargs):
            nonlocal stop_idx
            if stop_signal and stop_idx is None:
                stop_idx = i
            if i == stop_idx:
                pipe._interrupt = True
            progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")
            return callback_kwargs

        video = pipe(image=image, prompt=prompt, negative_prompt=negative_prompt, generator=generator,
                     guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, callback_on_step_end=combined_callback, callback_on_step_end_tensor_inputs=["latents"], use_dynamic_cfg=True)

        today = datetime.now().date()
        video_dir = os.path.join('outputs', f"CogVideoX_{today.strftime('%Y%m%d')}")
        os.makedirs(video_dir, exist_ok=True)

        video_filename = f"cogvideox_i2v_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        video_path = os.path.join(video_dir, video_filename)
        export_to_video(video.frames[0], video_path, fps=fps)

        return video_path, f"Video generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        flush()


def generate_video_cogvideox_video2video(prompt, negative_prompt, init_video, cogvideox_version, seed, stop_button, strength, guidance_scale, num_inference_steps, fps, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.bfloat32

    if not init_video:
        gr.Info("Please, upload an initial video!")
        return None, None

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    cogvideox_model_path = os.path.join("inputs", "video", "cogvideox", cogvideox_version)

    if not os.path.exists(cogvideox_model_path):
        gr.Info(f"Downloading {cogvideox_version} model...")
        os.makedirs(cogvideox_model_path, exist_ok=True)
        Repo.clone_from(f"https://huggingface.co/THUDM/{cogvideox_version}", cogvideox_model_path)
        gr.Info(f"{cogvideox_version} model downloaded")

    try:
        pipe = CogVideoXVideoToVideoPipeline().CogVideoXVideoToVideoPipeline.from_pretrained(cogvideox_model_path, torch_dtype=torch_dtype)
        pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        pipe.scheduler = CogVideoXDPMScheduler().CogVideoXDPMScheduler.from_config(pipe.scheduler.config)

        input_video = load_video(init_video)

        def combined_callback(pipe, i, t, callback_kwargs):
            nonlocal stop_idx
            if stop_signal and stop_idx is None:
                stop_idx = i
            if i == stop_idx:
                pipe._interrupt = True
            progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")
            return callback_kwargs

        video = pipe(
            video=input_video,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            callback_on_step_end=combined_callback,
            callback_on_step_end_tensor_inputs=["latents"]
        ).frames[0]

        today = datetime.now().date()
        video_dir = os.path.join('outputs', f"CogVideoX_{today.strftime('%Y%m%d')}")
        os.makedirs(video_dir, exist_ok=True)

        video_filename = f"cogvideox_v2v_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        video_path = os.path.join(video_dir, video_filename)
        export_to_video(video, video_path, fps=fps)

        return video_path, f"Video generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        flush()


def generate_video_latte(prompt, negative_prompt, seed, stop_button, num_inference_steps, guidance_scale, height, width, video_length, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    latte_model_path = os.path.join("inputs", "video", "latte")

    if not os.path.exists(latte_model_path):
        gr.Info("Downloading Latte model...")
        os.makedirs(latte_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/maxin-cn/Latte-1", latte_model_path)
        gr.Info("Latte model downloaded")

    try:
        pipe = LattePipeline().LattePipeline.from_pretrained(latte_model_path, torch_dtype=torch_dtype).to(device)
        pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        def combined_callback(pipe, i, t, callback_kwargs):
            nonlocal stop_idx
            if stop_signal and stop_idx is None:
                stop_idx = i
            if i == stop_idx:
                pipe._interrupt = True
            progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")
            return callback_kwargs

        videos = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            video_length=video_length,
            generator=generator,
            callback_on_step_end=combined_callback,
            callback_on_step_end_tensor_inputs=["latents"]
        ).frames[0]

        today = datetime.now().date()
        gif_dir = os.path.join('outputs', f"Latte_{today.strftime('%Y%m%d')}")
        os.makedirs(gif_dir, exist_ok=True)

        gif_filename = f"latte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
        gif_path = os.path.join(gif_dir, gif_filename)
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "video_length": video_length,
            "model": "Latte"
        }

        export_to_gif(videos, gif_path)
        add_metadata_to_file(gif_path, metadata)

        return gif_path, f"Video generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        flush()


def generate_3d_stablefast3d(image, texture_resolution, foreground_ratio, remesh_option, progress=gr.Progress()):

    if not image:
        gr.Info("Please upload an image!")
        return None, None

    try:
        progress(0.1, desc="Preparing output directory")
        today = datetime.now().date()
        output_dir = os.path.join('outputs', f"StableFast3D_{today.strftime('%Y%m%d')}")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "0", "mesh.glb")

        progress(0.5, desc="Constructing command")
        command = f"python ThirdPartyRepository/StableFast3D/run.py \"{image}\" --output-dir {output_dir} --texture-resolution {texture_resolution} --foreground-ratio {foreground_ratio} --remesh_option {remesh_option}"

        progress(0.7, desc="Running StableFast3D")
        subprocess.run(command, shell=True, check=True)

        progress(1.0, desc="StableFast3D process complete")
        return output_path, None

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        flush()


def generate_3d_shap_e(prompt, init_image, seed, num_inference_steps, guidance_scale, frame_size, progress=gr.Progress()):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    variant = "fp16" if torch_dtype == torch.float16 else "fp32"

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    progress(0.1, desc="Initializing Shap-E")
    if init_image:
        model_name = "openai/shap-e-img2img"
        model_path = os.path.join("inputs", "3D", "shap-e", "img2img")
        if not os.path.exists(model_path):
            progress(0.2, desc="Downloading Shap-E img2img model...")
            gr.Info("Downloading Shap-E img2img model...")
            os.makedirs(model_path, exist_ok=True)
            Repo.clone_from(f"https://huggingface.co/{model_name}", model_path)
            gr.Info("Shap-E img2img model downloaded")

        progress(0.3, desc="Loading img2img pipeline")
        pipe = ShapEImg2ImgPipeline().ShapEImg2ImgPipeline.from_pretrained(model_path, torch_dtype=torch_dtype, variant=variant).to(device)
        image = Image.open(init_image).resize((256, 256))
    else:
        model_name = "openai/shap-e"
        model_path = os.path.join("inputs", "3D", "shap-e", "text2img")
        if not os.path.exists(model_path):
            progress(0.2, desc="Downloading Shap-E text2img model...")
            gr.Info("Downloading Shap-E text2img model...")
            os.makedirs(model_path, exist_ok=True)
            Repo.clone_from(f"https://huggingface.co/{model_name}", model_path)
            gr.Info("Shap-E text2img model downloaded")

        progress(0.3, desc="Loading text2img pipeline")
        pipe = ShapEPipeline().ShapEPipeline.from_pretrained(model_path, torch_dtype=torch_dtype, variant=variant).to(device)

    progress(0.4, desc="Generating 3D object")
    if init_image:
        images = pipe(
            image,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            frame_size=frame_size,
            output_type="mesh",
            generator=generator,
        ).images
    else:
        images = pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            frame_size=frame_size,
            output_type="mesh",
            generator=generator,
        ).images

    progress(0.7, desc="Processing output")
    today = datetime.now().date()
    output_dir = os.path.join('outputs', f"Shap-E_{today.strftime('%Y%m%d')}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        progress(0.8, desc="Saving 3D object")
        ply_filename = f"3d_object_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ply"
        ply_path = os.path.join(output_dir, ply_filename)
        export_to_ply(images[0], ply_path)

        progress(0.9, desc="Converting to GLB format")
        mesh = trimesh.load(ply_path)
        rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        mesh = mesh.apply_transform(rot)
        glb_filename = f"3d_object_{datetime.now().strftime('%Y%m%d_%H%M%S')}.glb"
        glb_path = os.path.join(output_dir, glb_filename)
        mesh.export(glb_path, file_type="glb")

        progress(1.0, desc="Shap-E process complete")
        return glb_path, f"3D generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        flush()


def generate_3d_zero123plus(input_image, num_inference_steps, output_format, progress=gr.Progress()):
    if not input_image:
        gr.Info("Please upload an input image!")
        return None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    progress(0.1, desc="Initializing Zero123Plus")
    zero123plus_model_path = os.path.join("inputs", "3D", "zero123plus")

    if not os.path.exists(zero123plus_model_path):
        progress(0.2, desc="Downloading Zero123Plus model...")
        gr.Info("Downloading Zero123Plus model...")
        os.makedirs(zero123plus_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/sudo-ai/zero123plus-v1.2", zero123plus_model_path)
        gr.Info("Zero123Plus model downloaded")

    try:
        progress(0.3, desc="Loading Zero123Plus pipeline")
        pipe = DiffusionPipeline().DiffusionPipeline.from_pretrained(
            zero123plus_model_path,
            custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch_dtype
        )
        pipe.to(device)

        progress(0.4, desc="Processing input image")
        cond = Image.open(input_image)

        progress(0.5, desc="Generating 3D view")
        result = pipe(cond, num_inference_steps=num_inference_steps).images[0]

        progress(0.8, desc="Saving output")
        today = datetime.now().date()
        output_dir = os.path.join('outputs', f"Zero123Plus_{today.strftime('%Y%m%d')}")
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"zero123plus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        output_path = os.path.join(output_dir, output_filename)
        result.save(output_path)

        progress(1.0, desc="Zero123Plus process complete")
        return output_path, None

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'pipe' in locals():
            del pipe
        flush()


def generate_stableaudio(prompt, negative_prompt, seed, stop_button, num_inference_steps, guidance_scale, audio_length, audio_start, num_waveforms, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    sa_model_path = os.path.join("inputs", "audio", "stableaudio")

    if not os.path.exists(sa_model_path):
        gr.Info("Downloading Stable Audio Open model...")
        os.makedirs(sa_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/stabilityai/stable-audio-open-1.0", sa_model_path)
        gr.Info("Stable Audio Open model downloaded")

    pipe = StableAudioPipeline().StableAudioPipeline.from_pretrained(sa_model_path, torch_dtype=torch_dtype)
    pipe.scheduler = CosineDPMSolverMultistepScheduler().CosineDPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    def combined_callback(pipe, i, t):
        nonlocal stop_idx
        if stop_signal and stop_idx is None:
            stop_idx = i
        if i == stop_idx:
            pipe._interrupt = True
        progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")

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
            callback=combined_callback,
            callback_steps=1
        ).audios

        output = audio[0].T.float().cpu().numpy()

        today = datetime.now().date()
        audio_dir = os.path.join('outputs', f"StableAudio_{today.strftime('%Y%m%d')}")
        os.makedirs(audio_dir, exist_ok=True)
        audio_filename = f"stableaudio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        audio_path = os.path.join(audio_dir, audio_filename)

        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "audio_length": audio_length,
            "audio_start": audio_start,
            "num_waveforms": num_waveforms,
            "model": "StableAudio"
        }

        if output_format == "mp3":
            sf.write(audio_path, output, pipe.vae.sampling_rate, format='mp3')
        elif output_format == "ogg":
            sf.write(audio_path, output, pipe.vae.sampling_rate, format='ogg')
        else:
            sf.write(audio_path, output, pipe.vae.sampling_rate)

        add_metadata_to_file(audio_path, metadata)

        spectrogram_path = generate_mel_spectrogram(audio_path)

        return audio_path, spectrogram_path, f"Audio generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None, None

    finally:
        if 'pipe' in locals():
            del pipe
        flush()


def generate_audio_audiocraft(prompt, input_audio=None, model_name=None, model_type="musicgen",
                              duration=10, top_k=250, top_p=0.0,
                              temperature=1.0, cfg_coef=3.0, min_cfg_coef=1.0, max_cfg_coef=3.0, two_step_cfg=False, use_sampling=True, enable_multiband=False, output_format="mp3", progress=gr.Progress()):
    global audiocraft_model_path, multiband_diffusion_path

    progress(0.1, desc="Initializing AudioCraft")
    if not model_name:
        gr.Info("Please, select an AudioCraft model!")
        return None, None, None

    if enable_multiband and model_type in ["audiogen", "magnet"]:
        gr.Info("Multiband Diffusion is not supported with 'audiogen' or 'magnet' model types. Please select 'musicgen' or disable Multiband Diffusion")
        return None, None, None

    if not audiocraft_model_path:
        progress(0.2, desc="Loading AudioCraft model")
        audiocraft_model_path = load_audiocraft_model(model_name)

    if not multiband_diffusion_path:
        progress(0.3, desc="Loading Multiband Diffusion model")
        multiband_diffusion_path = load_multiband_diffusion_model()

    today = datetime.now().date()
    audio_dir = os.path.join('outputs', f"AudioCraft_{today.strftime('%Y%m%d')}")
    os.makedirs(audio_dir, exist_ok=True)

    try:
        progress(0.4, desc="Initializing model")
        if model_type == "musicgen":
            model = MusicGen().MusicGen.get_pretrained(audiocraft_model_path)
        elif model_type == "audiogen":
            model = AudioGen().AudioGen.get_pretrained(audiocraft_model_path)
        elif model_type == "magnet":
            model = MAGNeT().MAGNeT.get_pretrained(audiocraft_model_path)
        else:
            gr.Info("Invalid model type!")
            return None, None, None

        mbd = None

        if enable_multiband:
            progress(0.5, desc="Loading Multiband Diffusion")
            mbd = MultiBandDiffusion().MultiBandDiffusion.get_mbd_musicgen()

        progress(0.6, desc="Generating audio")
        if model_type == "musicgen" and input_audio:
            audio_path = input_audio
            melody, sr = torchaudio.load(audio_path)
            descriptions = [prompt]
            model.set_generation_params(duration=duration, top_k=top_k, top_p=top_p, temperature=temperature,
                                        cfg_coef=cfg_coef, two_step_cfg=two_step_cfg, use_sampling=use_sampling)
            wav, tokens = model.generate_with_chroma(descriptions, melody[None].expand(1, -1, -1), sr, return_tokens=True)
        elif model_type == "magnet":
            descriptions = [prompt]
            model.set_generation_params(top_k=top_k, top_p=top_p, temperature=temperature,
                                        min_cfg_coef=min_cfg_coef, max_cfg_coef=max_cfg_coef)
            wav = model.generate(descriptions)
        elif model_type == "musicgen":
            descriptions = [prompt]
            model.set_generation_params(duration=duration, top_k=top_k, top_p=top_p, temperature=temperature,
                                        cfg_coef=cfg_coef, two_step_cfg=two_step_cfg, use_sampling=use_sampling)
            wav, tokens = model.generate(descriptions, return_tokens=True)
        elif model_type == "audiogen":
            descriptions = [prompt]
            model.set_generation_params(duration=duration, top_k=top_k, top_p=top_p, temperature=temperature,
                                        cfg_coef=cfg_coef, two_step_cfg=two_step_cfg, use_sampling=use_sampling)
            wav = model.generate(descriptions)
        else:
            gr.Info(f"Unsupported model type: {model_type}")
            return None, None, None

        progress(0.7, desc="Processing generated audio")
        if wav.ndim > 2:
            wav = wav.squeeze()

        if mbd:
            progress(0.8, desc="Applying Multiband Diffusion")
            tokens = rearrange(tokens, "b n d -> n b d")
            wav_diffusion = mbd.tokens_to_wav(tokens)
            wav_diffusion = wav_diffusion.squeeze()
            if wav_diffusion.ndim == 1:
                wav_diffusion = wav_diffusion.unsqueeze(0)
            max_val = wav_diffusion.abs().max()
            if max_val > 1:
                wav_diffusion = wav_diffusion / max_val
            audio_filename_diffusion = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}_diffusion.wav"
            audio_path_diffusion = os.path.join(audio_dir, audio_filename_diffusion)
            torchaudio.save(audio_path_diffusion, wav_diffusion.cpu().detach(), model.sample_rate)

        progress(0.9, desc="Saving generated audio")
        audio_filename = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        audio_path = os.path.join(audio_dir, audio_filename)

        metadata = {
            "prompt": prompt,
            "model_name": model_name,
            "model_type": model_type,
            "duration": duration,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "cfg_coef": cfg_coef,
            "min_cfg_coef": min_cfg_coef,
            "max_cfg_coef": max_cfg_coef,
            "enable_multiband": enable_multiband if enable_multiband else None,
            "model": "AudioCraft"
        }
        if output_format == "mp3":
            audio_write(audio_path, wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True,
                        format='mp3')
        elif output_format == "ogg":
            audio_write(audio_path, wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True,
                        format='ogg')
        else:
            audio_write(audio_path, wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

        add_metadata_to_file(audio_path, metadata)

        progress(1.0, desc="Audio generation complete")
        if output_format == "mp3":
            spectrogram_path = generate_mel_spectrogram(audio_path + ".mp3")
            return audio_path + ".mp3", spectrogram_path, None
        elif output_format == "ogg":
            spectrogram_path = generate_mel_spectrogram(audio_path + ".ogg")
            return audio_path + ".ogg", spectrogram_path, None
        else:
            spectrogram_path = generate_mel_spectrogram(audio_path + ".wav")
            return audio_path + ".wav", spectrogram_path, None

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None, None

    finally:
        if 'model' in locals():
            del model
        if 'mbd' in locals():
            del mbd
        flush()


def generate_audio_audioldm2(prompt, negative_prompt, model_name, seed, stop_button, num_inference_steps, audio_length_in_s,
                             num_waveforms_per_prompt, output_format, progress=gr.Progress()):
    global stop_signal
    stop_signal = False
    stop_idx = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    if seed == "" or seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        seed = int(seed)
    generator = torch.Generator(device).manual_seed(seed)

    if not model_name:
        gr.Info("Please, select an AudioLDM 2 model!")
        return None, None, None

    model_path = os.path.join("inputs", "audio", "audioldm2", model_name)

    if not os.path.exists(model_path):
        gr.Info(f"Downloading AudioLDM 2 model: {model_name}...")
        os.makedirs(model_path, exist_ok=True)
        Repo.clone_from(f"https://huggingface.co/{model_name}", model_path)
        gr.Info(f"AudioLDM 2 model {model_name} downloaded")

    pipe = AudioLDM2Pipeline().AudioLDM2Pipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
    pipe = pipe.to(device)

    def combined_callback(pipe, i, t):
        nonlocal stop_idx
        if stop_signal and stop_idx is None:
            stop_idx = i
        if i == stop_idx:
            pipe._interrupt = True
        progress((i + 1) / num_inference_steps, f"Step {i + 1}/{num_inference_steps}")

    try:
        audio = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            audio_length_in_s=audio_length_in_s,
            num_waveforms_per_prompt=num_waveforms_per_prompt,
            generator=generator,
            callback=combined_callback,
            callback_steps=1
        ).audios

        today = datetime.now().date()
        audio_dir = os.path.join('outputs', f"AudioLDM2_{today.strftime('%Y%m%d')}")
        os.makedirs(audio_dir, exist_ok=True)
        audio_filename = f"audioldm2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        audio_path = os.path.join(audio_dir, audio_filename)

        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model_name": model_name,
            "seed": seed,
            "steps": num_inference_steps,
            "audio_length_in_s": audio_length_in_s,
            "num_waveforms_per_prompt": num_waveforms_per_prompt,
            "model": "AudioLDM 2"
        }

        if output_format == "mp3":
            scipy.io.wavfile.write(audio_path, rate=16000, data=audio[0])
        elif output_format == "ogg":
            scipy.io.wavfile.write(audio_path, rate=16000, data=audio[0])
        else:
            scipy.io.wavfile.write(audio_path, rate=16000, data=audio[0])

        add_metadata_to_file(audio_path, metadata)

        spectrogram_path = generate_mel_spectrogram(audio_path)

        return audio_path, spectrogram_path, f"Audio generated successfully. Seed used: {seed}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None, None

    finally:
        if 'pipe' in locals():
            del pipe
        flush()


def generate_bark_audio(text, voice_preset, max_length, fine_temperature, coarse_temperature, output_format, progress=gr.Progress()):

    progress(0.1, desc="Initializing Bark")
    if not text:
        gr.Info("Please enter text for the request!")
        return None, None, None

    bark_model_path = os.path.join("inputs", "audio", "bark")

    if not os.path.exists(bark_model_path):
        progress(0.2, desc="Downloading Bark model")
        gr.Info("Downloading Bark model...")
        os.makedirs(bark_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/suno/bark", bark_model_path)
        gr.Info("Bark model downloaded")

    try:
        progress(0.3, desc="Setting up device")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        progress(0.4, desc="Loading Bark model")
        processor = AutoProcessor().AutoProcessor.from_pretrained(bark_model_path)
        model = BarkModel().BarkModel.from_pretrained(bark_model_path, torch_dtype=torch_dtype)
        model.enable_cpu_offload()
        model = model.to_bettertransformer()

        progress(0.5, desc="Processing input")
        if voice_preset:
            inputs = processor(text, voice_preset=voice_preset, return_tensors="pt")
        else:
            inputs = processor(text, return_tensors="pt")

        progress(0.6, desc="Generating audio")
        audio_array = model.generate(**inputs, max_length=max_length, do_sample=True, fine_temperature=fine_temperature, coarse_temperature=coarse_temperature)

        progress(0.7, desc="Preparing output")
        today = datetime.now().date()
        audio_dir = os.path.join('outputs', f"Bark_{today.strftime('%Y%m%d')}")
        os.makedirs(audio_dir, exist_ok=True)

        audio_array = audio_array.cpu().numpy().squeeze()

        audio_filename = f"bark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        audio_path = os.path.join(audio_dir, audio_filename)

        metadata = {
            "text": text,
            "voice_preset": voice_preset,
            "max_length": max_length,
            "fine_temperature": fine_temperature,
            "coarse_temperature": coarse_temperature,
            "model": "SunoBark"
        }

        progress(0.8, desc="Saving audio file")
        if output_format == "mp3":
            sf.write(audio_path, audio_array, 24000)
        elif output_format == "ogg":
            sf.write(audio_path, audio_array, 24000)
        else:
            sf.write(audio_path, audio_array, 24000)

        add_metadata_to_file(audio_path, metadata)

        progress(0.9, desc="Generating spectrogram")
        spectrogram_path = generate_mel_spectrogram(audio_path)

        progress(1.0, desc="Audio generation complete")
        return audio_path, spectrogram_path, None

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None, None

    finally:
        if 'model' in locals():
            del model
        if 'processor' in locals():
            del processor
        flush()


def process_rvc(input_audio, model_folder, f0method, f0up_key, index_rate, filter_radius, resample_sr, rms_mix_rate, protect, output_format, progress=gr.Progress()):
    progress(0.1, desc="Initializing RVC")
    if not input_audio:
        gr.Info("Please upload an audio file!")
        return None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = os.path.join("inputs", "audio", "rvc_models", model_folder)

    try:
        progress(0.2, desc="Loading RVC model")
        pth_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
        if not pth_files:
            gr.Info(f"No .pth file found in the selected model folder: {model_folder}")
            return None, None

        model_file = os.path.join(model_path, pth_files[0])

        rvc = RVCInference().RVCInference(device=device)
        rvc.load_model(model_file)
        rvc.set_params(f0method=f0method, f0up_key=f0up_key, index_rate=index_rate, filter_radius=filter_radius, resample_sr=resample_sr, rms_mix_rate=rms_mix_rate, protect=protect)

        progress(0.3, desc="Preparing output directory")
        today = datetime.now().date()
        output_dir = os.path.join('outputs', f"RVC_{today.strftime('%Y%m%d')}")
        os.makedirs(output_dir, exist_ok=True)

        output_filename = f"rvc_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        output_path = os.path.join(output_dir, output_filename)

        progress(0.4, desc="Processing audio")
        rvc.infer_file(input_audio, output_path)

        progress(0.9, desc="Unloading model")
        rvc.unload_model()

        progress(1.0, desc="RVC processing complete")
        return output_path, None

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        if 'rvc' in locals():
            del rvc
        flush()


def separate_audio_uvr(audio_file, output_format, normalization_threshold, sample_rate, progress=gr.Progress()):

    progress(0.1, desc="Initializing UVR")
    if not audio_file:
        gr.Info("Please upload an audio file!")
        return None, None, None

    try:
        progress(0.2, desc="Preparing output directory")
        today = datetime.now().date()
        output_dir = os.path.join('outputs', f"UVR_{today.strftime('%Y%m%d')}")
        os.makedirs(output_dir, exist_ok=True)

        progress(0.3, desc="Loading UVR model")
        separator = Separator().Separator(output_format=output_format, normalization_threshold=normalization_threshold,
                              sample_rate=sample_rate, output_dir=output_dir)
        separator.load_model(model_filename='UVR-MDX-NET-Inst_HQ_3.onnx')

        progress(0.4, desc="Separating audio")
        output_files = separator.separate(audio_file)

        if len(output_files) != 2:
            gr.Info(f"Unexpected number of output files: {len(output_files)}")
            return None, None, None

        progress(1.0, desc="UVR separation complete")
        return output_files[0], output_files[1], f"Separation complete! Output files: {' '.join(output_files)}"

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None, None

    finally:
        if 'separator' in locals():
            del separator
        flush()


def demucs_separate(audio_file, output_format, progress=gr.Progress()):
    progress(0.1, desc="Initializing Demucs")
    if not audio_file:
        gr.Info("Please upload an audio file!")
        return None, None, None

    today = datetime.now().date()
    demucs_dir = os.path.join("outputs", f"Demucs_{today.strftime('%Y%m%d')}")
    os.makedirs(demucs_dir, exist_ok=True)

    now = datetime.now()
    separate_dir = os.path.join(demucs_dir, f"separate_{now.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(separate_dir, exist_ok=True)

    try:
        progress(0.2, desc="Preparing separation command")
        command = f"demucs --two-stems=vocals {audio_file} -o {separate_dir}"

        progress(0.3, desc="Running Demucs separation")
        subprocess.run(command, shell=True, check=True)

        progress(0.6, desc="Processing separated files")
        temp_vocal_file = os.path.join(separate_dir, "htdemucs", os.path.splitext(os.path.basename(audio_file))[0],
                                       "vocals.wav")
        temp_instrumental_file = os.path.join(separate_dir, "htdemucs",
                                              os.path.splitext(os.path.basename(audio_file))[0], "no_vocals.wav")

        vocal_file = os.path.join(separate_dir, "vocals.wav")
        instrumental_file = os.path.join(separate_dir, "instrumental.wav")

        os.rename(temp_vocal_file, vocal_file)
        os.rename(temp_instrumental_file, instrumental_file)

        progress(0.8, desc="Converting to desired format")
        if output_format == "mp3":
            vocal_output = os.path.join(separate_dir, "vocal.mp3")
            instrumental_output = os.path.join(separate_dir, "instrumental.mp3")
            subprocess.run(f"ffmpeg -i {vocal_file} -b:a 192k {vocal_output}", shell=True, check=True)
            subprocess.run(f"ffmpeg -i {instrumental_file} -b:a 192k {instrumental_output}", shell=True, check=True)
        elif output_format == "ogg":
            vocal_output = os.path.join(separate_dir, "vocal.ogg")
            instrumental_output = os.path.join(separate_dir, "instrumental.ogg")
            subprocess.run(f"ffmpeg -i {vocal_file} -c:a libvorbis -qscale:a 5 {vocal_output}", shell=True, check=True)
            subprocess.run(f"ffmpeg -i {instrumental_file} -c:a libvorbis -qscale:a 5 {instrumental_output}",
                           shell=True, check=True)
        else:
            vocal_output = vocal_file
            instrumental_output = instrumental_file

        progress(1.0, desc="Demucs separation complete")
        return vocal_output, instrumental_output, None

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None, None

    finally:
        flush()


def generate_image_extras(input_image, remove_background, enable_facerestore, fidelity_weight, restore_upscale,
                          enable_pixeloe, mode, target_size, patch_size, thickness, contrast, saturation,
                          enable_ddcolor, ddcolor_input_size, enable_downscale, downscale_factor,
                          enable_format_changer, new_format, enable_encryption, enable_decryption, decryption_key,
                          progress=gr.Progress()):
    global key
    progress(0.1, desc="Initializing image processing")
    if not input_image:
        gr.Info("Please upload an image!")
        return None, None

    if not remove_background and not enable_facerestore and not enable_pixeloe and not enable_ddcolor and not enable_downscale and not enable_format_changer and not enable_encryption and not enable_decryption:
        gr.Info("Please choose an option to modify the image")
        return None, None

    try:
        output_path = input_image
        output_format = os.path.splitext(input_image)[1][1:]

        if remove_background:
            progress(0.2, desc="Removing background")
            output_filename = f"background_removed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
            output_path = os.path.join(os.path.dirname(input_image), output_filename)
            remove_bg(input_image, output_path)

        if enable_facerestore:
            progress(0.3, desc="Restoring face")
            codeformer_path = os.path.join("inputs", "image", "CodeFormer")
            facerestore_output_path = os.path.join(os.path.dirname(output_path), f"facerestored_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}")
            command = f"python {os.path.join(codeformer_path, 'inference_codeformer.py')} -w {fidelity_weight} --upscale {restore_upscale} --bg_upsampler realesrgan --face_upsample --input_path {output_path} --output_path {facerestore_output_path}"
            subprocess.run(command, shell=True, check=True)
            output_path = facerestore_output_path

        if enable_pixeloe:
            progress(0.4, desc="Applying PixelOE")
            img = cv2.imread(output_path)
            img = pixelize().pixelize(img, mode=mode, target_size=target_size, patch_size=patch_size, thickness=thickness, contrast=contrast, saturation=saturation)
            pixeloe_output_path = os.path.join(os.path.dirname(output_path), f"pixeloe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}")
            cv2.imwrite(pixeloe_output_path, img)
            output_path = pixeloe_output_path

        if enable_ddcolor:
            progress(0.5, desc="Applying DDColor")
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_input_path = os.path.join(temp_dir, os.path.basename(output_path))
                shutil.copy(output_path, temp_input_path)

                temp_output_dir = os.path.join(temp_dir, "output")
                os.makedirs(temp_output_dir, exist_ok=True)

                ddcolor_output_path = os.path.join(os.path.dirname(output_path), f"ddcolor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}")
                command = f"python ThirdPartyRepository/DDColor/colorization_pipeline_hf.py --model_name ddcolor_modelscope --input {temp_input_path} --output {temp_output_dir} --input_size {ddcolor_input_size}"
                subprocess.run(command, shell=True, check=True)

                generated_files = os.listdir(temp_output_dir)
                if generated_files:
                    generated_file = generated_files[0]
                    shutil.copy(os.path.join(temp_output_dir, generated_file), ddcolor_output_path)
                    output_path = ddcolor_output_path
                else:
                    gr.Info("No DDcolor images")

        if enable_downscale:
            progress(0.6, desc="Downscaling image")
            output_path = downscale_image(output_path, downscale_factor)

        if enable_format_changer:
            progress(0.7, desc="Changing image format")
            output_path, message = change_image_format(output_path, new_format, enable_format_changer)
            if message.startswith("Error"):
                return None, message

        if enable_encryption:
            progress(0.8, desc="Encrypting image")
            img = Image.open(output_path)
            key = generate_key()
            encrypted_img, _ = encrypt_image(img, key)
            encrypted_output_path = os.path.join(os.path.dirname(output_path), f"encrypted_{os.path.basename(output_path)}")
            encrypted_img.save(encrypted_output_path)
            output_path = encrypted_output_path

        if enable_decryption:
            progress(0.9, desc="Decrypting image")
            if not decryption_key:
                gr.Info("Please provide a decryption key!")
                return None, None
            img = Image.open(output_path)
            try:
                decrypted_img = decrypt_image(img, decryption_key)
                decrypted_output_path = os.path.join(os.path.dirname(output_path), f"decrypted_{os.path.basename(output_path)}")
                decrypted_img.save(decrypted_output_path)
                output_path = decrypted_output_path
            except Exception as e:
                gr.Error(f"Decryption failed. Please check your key. Error: {str(e)}")
                return None, None

        progress(1.0, desc="Image processing complete")
        if enable_encryption:
            return output_path, f"Image encrypted successfully. Decryption key: {key}"
        else:
            return output_path, "Image processing completed successfully."

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        flush()


def generate_video_extras(input_video, enable_downscale, downscale_factor, enable_format_changer, new_format, progress=gr.Progress()):
    progress(0.1, desc="Initializing video processing")
    if not input_video:
        gr.Info("Please upload a video!")
        return None, None

    if not enable_downscale and not enable_format_changer:
        gr.Info("Please choose an option to modify the video")
        return None, None

    try:
        output_path = input_video

        if enable_downscale:
            progress(0.3, desc="Downscaling video")
            output_path = downscale_video(output_path, downscale_factor)

        if enable_format_changer:
            progress(0.6, desc="Changing video format")
            output_path, message = change_video_format(output_path, new_format, enable_format_changer)
            if message.startswith("Error"):
                return None, message

        progress(1.0, desc="Video processing complete")
        return output_path, "Video processing completed successfully."

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        flush()


def generate_audio_extras(input_audio, enable_format_changer, new_format, enable_audiosr, steps, guidance_scale, enable_downscale, downscale_factor, progress=gr.Progress()):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    progress(0.1, desc="Initializing audio processing")
    if not input_audio:
        gr.Info("Please upload an audio file!")
        return None, None

    if not enable_format_changer and not enable_audiosr and not enable_downscale:
        gr.Info("Please choose an option to modify the audio")
        return None, None

    try:
        output_path = input_audio

        if enable_format_changer:
            progress(0.3, desc="Changing audio format")
            output_path, message = change_audio_format(output_path, new_format, enable_format_changer)
            if message.startswith("Error"):
                return None, message

        if enable_audiosr:
            progress(0.5, desc="Enhancing audio with AudioSR")
            today = datetime.now().date()
            audio_dir = os.path.join('outputs', f"AudioSR_{today.strftime('%Y%m%d')}")
            os.makedirs(audio_dir, exist_ok=True)

            command = f"audiosr -i {output_path} -s {audio_dir} --ddim_steps {steps} -gs {guidance_scale} -d {device}"
            subprocess.run(command, shell=True, check=True)

            subfolders = [f for f in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, f))]
            if subfolders:
                latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(audio_dir, x)))
                subfolder_path = os.path.join(audio_dir, latest_subfolder)

                wav_files = [f for f in os.listdir(subfolder_path) if f.endswith('.wav')]
                if wav_files:
                    output_path = os.path.join(subfolder_path, wav_files[0])
                else:
                    gr.Info("AudioSR processed the file, but no output WAV file was found.")
                    return None, None
            else:
                gr.Info("AudioSR did not create any output folders.")
                return None, None

        if enable_downscale:
            progress(0.8, desc="Downscaling audio")
            today = datetime.now().date()
            audio_dir = os.path.join('outputs', f"AudioDownscale_{today.strftime('%Y%m%d')}")
            os.makedirs(audio_dir, exist_ok=True)
            output_filename = f"downscaled_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            downscale_output_path = os.path.join(audio_dir, output_filename)

            original_rate, target_rate = downscale_audio(output_path, downscale_output_path, downscale_factor)

            output_path = downscale_output_path

        progress(1.0, desc="Audio processing complete")
        return output_path, "Audio processing completed successfully."

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None

    finally:
        flush()


def generate_upscale_realesrgan(input_image, input_video, model_name, outscale, face_enhance, tile, tile_pad, pre_pad, denoise_strength, output_format, progress=gr.Progress()):
    progress(0.1, desc="Initializing Real-ESRGAN")
    realesrgan_path = os.path.join("inputs", "image", "Real-ESRGAN")

    if not input_image and not input_video:
        gr.Info("Please, upload an initial image or video!")
        return None, None, None

    today = datetime.now().date()
    output_dir = os.path.join('outputs', f"RealESRGAN_{today.strftime('%Y%m%d')}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        if input_image:
            input_file = input_image
            is_video = False
        else:
            input_file = input_video
            is_video = True

        input_filename = os.path.basename(input_file)
        input_name, input_ext = os.path.splitext(input_filename)

        progress(0.2, desc="Preparing input")
        if is_video:
            cap = cv2.VideoCapture(input_file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            output_filename = f"{input_name}_out.mp4"
            output_path = os.path.join(output_dir, output_filename)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width * outscale, height * outscale))

            progress(0.3, desc="Processing video frames")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            processed_frames = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_path = os.path.join(output_dir, "temp_frame.png")
                cv2.imwrite(frame_path, frame)

                command = f"python {os.path.join(realesrgan_path, 'inference_realesrgan.py')} -i {frame_path} -o {output_dir} -n {model_name} -s {outscale} --tile {tile} --tile_pad {tile_pad} --pre_pad {pre_pad} --denoise_strength {denoise_strength}"
                if face_enhance:
                    command += " --face_enhance"

                subprocess.run(command, shell=True, check=True)

                upscaled_frame = cv2.imread(os.path.join(output_dir, "temp_frame_out.png"))
                out.write(upscaled_frame)

                os.remove(frame_path)
                os.remove(os.path.join(output_dir, "temp_frame_out.png"))

                processed_frames += 1
                progress(0.3 + (0.6 * processed_frames / total_frames), desc=f"Processing frame {processed_frames}/{total_frames}")

            cap.release()
            out.release()
        else:
            progress(0.4, desc="Upscaling image")
            command = f"python {os.path.join(realesrgan_path, 'inference_realesrgan.py')} -i {input_file} -o {output_dir} -n {model_name} -s {outscale} --tile {tile} --tile_pad {tile_pad} --pre_pad {pre_pad} --denoise_strength {denoise_strength}"
            if face_enhance:
                command += " --face_enhance"

            subprocess.run(command, shell=True, check=True)

        progress(0.9, desc="Finalizing output")
        expected_output_filename = f"{input_name}_out{input_ext}"
        output_path = os.path.join(output_dir, expected_output_filename)

        if os.path.exists(output_path):
            if not is_video and output_format.lower() != input_ext[1:].lower():
                new_output_filename = f"{input_name}_out.{output_format}"
                new_output_path = os.path.join(output_dir, new_output_filename)
                Image.open(output_path).save(new_output_path)
                output_path = new_output_path

            progress(1.0, desc="Upscaling complete")
            if is_video:
                return None, output_path, None
            else:
                return output_path, None, None
        else:
            gr.Info("Output file not found")
            return None, None, None

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None, None

    finally:
        flush()


def generate_faceswap(source_image, target_image, target_video, enable_many_faces, reference_face,
                      reference_frame, enable_facerestore, fidelity_weight, restore_upscale, progress=gr.Progress()):
    progress(0.1, desc="Initializing face swap")
    if not source_image or (not target_image and not target_video):
        gr.Info("Please upload source image and either target image or target video!")
        return None, None, None

    try:
        roop_path = os.path.join("inputs", "image", "roop")

        today = datetime.now().date()
        output_dir = os.path.join('outputs', f"FaceSwap_{today.strftime('%Y%m%d')}")
        os.makedirs(output_dir, exist_ok=True)

        is_video = bool(target_video)

        progress(0.2, desc="Preparing output files")
        if is_video:
            faceswap_output_filename = f"faceswapped_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        else:
            faceswap_output_filename = f"faceswapped_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        faceswap_output_path = os.path.join(output_dir, faceswap_output_filename)

        progress(0.3, desc="Constructing face swap command")
        command = f"python {os.path.join(roop_path, 'run.py')} --source {source_image} --output {faceswap_output_path}"

        if is_video:
            command += f" --target {target_video}"
        else:
            command += f" --target {target_image}"

        if enable_many_faces:
            command += f" --many-faces"
            command += f" --reference-face-position {reference_face}"
            command += f" --reference-frame-number {reference_frame}"

        progress(0.4, desc="Running face swap")
        subprocess.run(command, shell=True, check=True)

        if enable_facerestore:
            progress(0.6, desc="Applying face restoration")
            codeformer_path = os.path.join("inputs", "image", "CodeFormer")

            if is_video:
                facerestore_output_filename = f"facerestored_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            else:
                facerestore_output_filename = f"facerestored_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

            facerestore_output_path = os.path.join(output_dir, facerestore_output_filename)

            facerestore_input = faceswap_output_path

            command = f"python {os.path.join(codeformer_path, 'inference_codeformer.py')} -w {fidelity_weight} --upscale {restore_upscale} --bg_upsampler realesrgan --face_upsample --input_path {facerestore_input} --output_path {facerestore_output_path}"
            subprocess.run(command, shell=True, check=True)

            output_path = facerestore_output_path
        else:
            output_path = faceswap_output_path

        progress(1.0, desc="Face swap complete")
        if is_video:
            return None, output_path, None
        else:
            return output_path, None, None

    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None, None, None

    finally:
        flush()


def display_metadata(file, progress=gr.Progress()):
    progress(0.1, desc="Initializing metadata display")
    if file is None:
        gr.Info("Please upload a file.")
        return None

    try:
        progress(0.3, desc="Reading file")
        file_extension = os.path.splitext(file.name)[1].lower()
        metadata = None

        progress(0.5, desc="Extracting metadata")
        if file_extension in ['.png', '.jpeg', '.gif']:
            img = Image.open(file.name)
            metadata_str = img.info.get("COMMENT")
            if metadata_str:
                metadata = json.loads(metadata_str)
        elif file_extension in ['.mp4', '.avi', '.mov']:
            with taglib.File(file.name) as video:
                metadata_str = video.tags.get("COMMENT")
                if metadata_str:
                    metadata = json.loads(metadata_str[0])
        elif file_extension in ['.wav', '.mp3', '.ogg']:
            with taglib.File(file.name) as audio:
                metadata_str = audio.tags.get("COMMENT")
                if metadata_str:
                    metadata = json.loads(metadata_str[0])

        progress(1.0, desc="Metadata extraction complete")
        if metadata:
            return f"COMMENT:\n{json.dumps(metadata, indent=2)}"
        else:
            return "No COMMENT metadata found in the file."
    except Exception as e:
        gr.Error(f"An error occurred: {str(e)}")
        return None


def get_wiki_content(url, local_file="TechnicalFiles/Wikies/WikiEN.md", progress=gr.Progress()):
    progress(0.1, desc="Initializing wiki content retrieval")
    try:
        progress(0.3, desc="Attempting to fetch online content")
        response = requests.get(url)
        if response.status_code == 200:
            progress(1.0, desc="Online content retrieved successfully")
            return response.text
    except:
        pass

    try:
        progress(0.6, desc="Fetching local content")
        with open(local_file, 'r', encoding='utf-8') as file:
            content = file.read()
            progress(0.9, desc="Converting Markdown to HTML")
            html_content = markdown.markdown(content)
            progress(1.0, desc="Local content processed successfully")
            return html_content
    except:
        progress(1.0, desc="Content retrieval failed")
        return "<p>Wiki content is not available.</p>"


def get_output_files(progress=gr.Progress()):
    progress(0.1, desc="Initializing file search")
    output_dir = "outputs"
    files = []

    for root, dirs, filenames in os.walk(output_dir):
        for filename in filenames:
            files.append(os.path.join(root, filename))

    progress(1.0, desc="File search complete")
    return files


def display_output_file(file_path):
    if file_path is None:
        return None, None, None, None, None

    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension in ['.txt', '.json']:
        with open(file_path, "r") as f:
            content = f.read()
        return content, None, None, None, None
    elif file_extension in ['.png', '.jpeg', '.jpg', '.gif']:
        return None, file_path, None, None, None
    elif file_extension == '.mp4':
        return None, None, file_path, None, None
    elif file_extension in ['.wav', '.mp3', '.ogg']:
        return None, None, None, file_path, None
    elif file_extension in ['.obj', '.ply', '.glb']:
        return None, None, None, None, file_path
    else:
        return None, None, None, None, None


DOWNLOADER_MODEL_TYPES = [
    "LLM-Model", "LLM-LoRA", "SD-Model", "SD-Inpaint", "SD-VAE", "SD-LoRA",
    "SD-Embedding", "FLUX-Model", "FLUX-VAE", "FLUX-LoRA", "AuraFlow-LoRA", "Kolors-LoRA", "RVC-Model"
]

DOWNLOADER_MODEL_PATHS = {
    "LLM-Model": "inputs/text/llm_models",
    "LLM-LoRA": "inputs/text/llm_models/lora",
    "SD-Model": "inputs/image/sd_models",
    "SD-Inpaint": "inputs/image/sd_models/inpaint",
    "SD-VAE": "inputs/image/sd_models/vae",
    "SD-LoRA": "inputs/image/sd_models/lora",
    "SD-Embedding": "inputs/image/sd_models/embedding",
    "FLUX-Model": "inputs/image/flux/quantize-flux",
    "FLUX-VAE": "inputs/image/flux/flux-vae",
    "FLUX-LoRA": "inputs/image/flux/flux-lora",
    "AuraFlow-LoRA": "inputs/image/auraflow-lora",
    "Kolors-LoRA": "inputs/image/kolors-lora",
    "RVC-Model": "inputs/audio/rvc_models"
}


def download_model(model_url, model_type, progress=gr.Progress()):
    progress(0.1, desc="Initializing model download")
    if not model_url:
        gr.Info("Please enter a model URL to download")
        return None

    try:
        model_path = DOWNLOADER_MODEL_PATHS.get(model_type)
        if not model_path:
            gr.Error(f"Invalid model type: {model_type}")
            return None

        os.makedirs(model_path, exist_ok=True)

        if "/" in model_url and "blob/main" not in model_url and not model_url.startswith("http"):
            progress(0.3, desc="Cloning repository")
            repo_name = model_url.split("/")[-1]
            repo_path = os.path.join(model_path, repo_name)
            Repo.clone_from(f"https://huggingface.co/{model_url}", repo_path)
            progress(0.9, desc="Repository cloned successfully")
            return f"{model_type} repository {repo_name} downloaded successfully!"
        else:
            progress(0.3, desc="Downloading file")
            if "blob/main" in model_url:
                model_url = model_url.replace("blob/main", "resolve/main")
            file_name = model_url.split("/")[-1]
            file_path = os.path.join(model_path, file_name)
            response = requests.get(model_url, allow_redirects=True, stream=True)
            with open(file_path, "wb") as file:
                file.write(response.content)
            progress(0.9, desc="File downloaded successfully")
            return f"{model_type} file {file_name} downloaded successfully!"

    except Exception as e:
        gr.Error(f"Error downloading model: {str(e)}")
        return None
    finally:
        progress(1.0, desc="Download process complete")


def settings_interface(language, share_value, debug_value, monitoring_value, auto_launch, api_status, open_api, queue_max_size, status_update_rate, max_file_size, gradio_auth, server_name, server_port, hf_token, share_server_address, theme,
                       enable_custom_theme, primary_hue, secondary_hue, neutral_hue,
                       spacing_size, radius_size, text_size, font, font_mono, progress=gr.Progress()):
    progress(0.1, desc="Initializing settings update")
    settings = load_settings()

    progress(0.2, desc="Updating general settings")
    settings['language'] = language
    settings['share_mode'] = share_value == "True"
    settings['debug_mode'] = debug_value == "True"
    settings['monitoring_mode'] = monitoring_value == "True"
    settings['auto_launch'] = auto_launch == "True"
    settings['show_api'] = api_status == "True"
    settings['api_open'] = open_api == "True"
    settings['queue_max_size'] = int(queue_max_size) if queue_max_size else 10
    settings['status_update_rate'] = status_update_rate
    settings['max_file_size'] = int(max_file_size) if max_file_size else 1000

    progress(0.4, desc="Updating authentication settings")
    if gradio_auth:
        username, password = gradio_auth.split(':')
        settings['auth'] = {"username": username, "password": password}

    progress(0.6, desc="Updating server settings")
    settings['server_name'] = server_name
    settings['server_port'] = int(server_port) if server_port else 7860
    settings['hf_token'] = hf_token
    settings['share_server_address'] = share_server_address

    progress(0.8, desc="Updating theme settings")
    settings['theme'] = theme
    settings['custom_theme']['enabled'] = enable_custom_theme
    settings['custom_theme']['primary_hue'] = primary_hue
    settings['custom_theme']['secondary_hue'] = secondary_hue
    settings['custom_theme']['neutral_hue'] = neutral_hue
    settings['custom_theme']['spacing_size'] = spacing_size
    settings['custom_theme']['radius_size'] = radius_size
    settings['custom_theme']['text_size'] = text_size
    settings['custom_theme']['font'] = font
    settings['custom_theme']['font_mono'] = font_mono

    progress(0.9, desc="Saving settings")
    save_settings(settings)

    progress(1.0, desc="Settings update complete")

    message = "Settings updated successfully!"
    message += f"\nLanguage set to {settings['language']}"
    message += f"\nShare mode is {settings['share_mode']}"
    message += f"\nDebug mode is {settings['debug_mode']}"
    message += f"\nMonitoring mode is {settings['monitoring_mode']}"
    message += f"\nAutoLaunch mode is {settings['auto_launch']}"
    message += f"\nShow API mode is {settings['show_api']}"
    message += f"\nOpen API mode is {settings['api_open']}"
    message += f"\nQueue max size is {settings['queue_max_size']}"
    message += f"\nStatus update rate is {settings['status_update_rate']}"
    message += f"\nNew Gradio Auth is {settings['auth']}"
    message += f"\nServer will run on {settings['server_name']}:{settings['server_port']}"
    message += f"\nNew HF-Token is {settings['hf_token']}"
    message += f"\nNew Share server address is {settings['share_server_address']}"
    message += f"\nNew Max file size is {settings['max_file_size']}"
    message += f"\nTheme set to {theme and settings['custom_theme'] if enable_custom_theme else theme}"
    message += f"\nPlease restart the application for changes to take effect!"

    return message


def get_system_info(progress=gr.Progress()):
    progress(0.1, desc="Initializing system info retrieval")

    progress(0.2, desc="Getting GPU information")
    if torch.cuda.is_available():
        gpu = GPUtil.getGPUs()[0]
        gpu_total_memory = f"{gpu.memoryTotal} MB"
        gpu_used_memory = f"{gpu.memoryUsed} MB"
        gpu_free_memory = f"{gpu.memoryFree} MB"
    else:
        pass

    progress(0.3, desc="Getting GPU temperature")
    if torch.cuda.is_available():
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        gpu_temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
    else:
        pass

    progress(0.4, desc="Getting CPU temperature")
    if sys.platform == 'win32' or 'win64':
        cpu_temp = (WinTmp.CPU_Temp())
    else:
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                cpu_temp = temps['coretemp'][0].current
            else:
                result = subprocess.run(["sensors"], capture_output=True, text=True)
                lines = result.stdout.splitlines()
                for line in lines:
                    if "Core" in line:
                        cpu_temp = float(line.split()[1].strip('+°C'))
                        break
        else:
            pass

    progress(0.5, desc="Getting RAM information")
    ram = psutil.virtual_memory()
    ram_total = f"{ram.total // (1024 ** 3)} GB"
    ram_used = f"{ram.used // (1024 ** 3)} GB"
    ram_free = f"{ram.available // (1024 ** 3)} GB"

    progress(0.6, desc="Getting disk information")
    disk = psutil.disk_usage('/')
    disk_total = f"{disk.total // (1024 ** 3)} GB"
    disk_free = f"{disk.free // (1024 ** 3)} GB"

    progress(0.8, desc="Calculating application folder size")
    app_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    app_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                   for dirpath, dirnames, filenames in os.walk(app_folder)
                   for filename in filenames)
    app_size = f"{app_size // (1024 ** 3):.2f} GB"

    progress(1.0, desc="System info retrieval complete")
    return (gpu_total_memory, gpu_used_memory, gpu_free_memory, gpu_temp, cpu_temp,
            ram_total, ram_used, ram_free, disk_total, disk_free, app_size)


def close_terminal():
    os._exit(1)


def open_outputs_folder():
    outputs_folder = "outputs"
    if os.path.exists(outputs_folder):
        if os.name == "nt":
            os.startfile(outputs_folder)
        else:
            os.system(f'open "{outputs_folder}"' if os.name == "darwin" else f'xdg-open "{outputs_folder}"')


llm_models_list = [None, "Moondream2-Image", "LLaVA-NeXT-Video", "Qwen2-Audio"] + [model for model in os.listdir("inputs/text/llm_models") if not model.endswith(".txt") and model != "vikhyatk" and model != "lora"]
llm_lora_models_list = [None] + [model for model in os.listdir("inputs/text/llm_models/lora") if not model.endswith(".txt")]
speaker_wavs_list = [None] + [wav for wav in os.listdir("inputs/audio/voices") if not wav.endswith(".txt")]
stable_diffusion_models_list = [None] + [model for model in os.listdir("inputs/image/sd_models")
                                         if (model.endswith(".safetensors") or model.endswith(".ckpt") or model.endswith(".gguf") or not model.endswith(".txt") and not model.endswith(".py") and not os.path.isdir(os.path.join("inputs/image/sd_models")))]
audiocraft_models_list = [None] + ["musicgen-stereo-medium", "audiogen-medium", "musicgen-stereo-melody", "musicgen-medium", "musicgen-melody", "musicgen-large",
                                   "hybrid-magnet-medium", "magnet-medium-30sec", "magnet-medium-10sec", "audio-magnet-medium"]
vae_models_list = [None] + [model for model in os.listdir("inputs/image/sd_models/vae") if
                            model.endswith(".safetensors") or not model.endswith(".txt")]
flux_vae_models_list = [None] + [model for model in os.listdir("inputs/image/flux/flux-vae") if
                                model.endswith(".safetensors") or not model.endswith(".txt")]
lora_models_list = [None] + [model for model in os.listdir("inputs/image/sd_models/lora") if
                             model.endswith(".safetensors") or model.endswith(".pt")]
quantized_flux_models_list = [None] + [model for model in os.listdir("inputs/image/flux/quantize-flux") if
                            model.endswith(".gguf") or model.endswith(".safetensors") or not model.endswith(".txt") and not model.endswith(".safetensors") and not model.endswith(".py")]
flux_lora_models_list = [None] + [model for model in os.listdir("inputs/image/flux/flux-lora") if
                             model.endswith(".safetensors")]
auraflow_lora_models_list = [None] + [model for model in os.listdir("inputs/image/auraflow-lora") if
                             model.endswith(".safetensors")]
kolors_lora_models_list = [None] + [model for model in os.listdir("inputs/image/kolors-lora") if
                             model.endswith(".safetensors")]
textual_inversion_models_list = [None] + [model for model in os.listdir("inputs/image/sd_models/embedding") if model.endswith(".pt") or model.endswith(".safetensors")]
inpaint_models_list = [None] + [model for model in
                                os.listdir("inputs/image/sd_models/inpaint")
                                if (model.endswith(".safetensors") or model.endswith(".ckpt") or not model.endswith(".txt"))]
controlnet_models_list = [None, "openpose", "depth", "canny", "lineart", "scribble"]
rvc_models_list = [model_folder for model_folder in os.listdir("inputs/audio/rvc_models")
                   if os.path.isdir(os.path.join("inputs/audio/rvc_models", model_folder))
                   and any(file.endswith('.pth') for file in os.listdir(os.path.join("inputs/audio/rvc_models", model_folder)))]


def reload_model_lists():
    global llm_models_list, llm_lora_models_list, speaker_wavs_list, stable_diffusion_models_list, vae_models_list, lora_models_list, quantized_flux_models_list, flux_lora_models_list, auraflow_lora_models_list, kolors_lora_models_list, textual_inversion_models_list, inpaint_models_list, rvc_models_list

    llm_models_list = [None, "Moondream2-Image", "LLaVA-NeXT-Video", "Qwen2-Audio"] + [model for model in os.listdir("inputs/text/llm_models") if
                                              not model.endswith(".txt") and model != "vikhyatk" and model != "lora"]
    llm_lora_models_list = [None] + [model for model in os.listdir("inputs/text/llm_models/lora") if
                                     not model.endswith(".txt")]
    speaker_wavs_list = [None] + [wav for wav in os.listdir("inputs/audio/voices") if not wav.endswith(".txt")]
    stable_diffusion_models_list = [None] + [model for model in os.listdir("inputs/image/sd_models")
                                             if (model.endswith(".safetensors") or model.endswith(
            ".ckpt") or model.endswith(".gguf") or not model.endswith(".txt") and not model.endswith(
            ".py") and not os.path.isdir(os.path.join("inputs/image/sd_models")))]
    vae_models_list = [None] + [model for model in os.listdir("inputs/image/sd_models/vae") if
                                model.endswith(".safetensors") or not model.endswith(".txt")]
    flux_vae_models_list = [None] + [model for model in os.listdir("inputs/image/flux/flux-vae") if
                                model.endswith(".safetensors") or not model.endswith(".txt")]
    lora_models_list = [None] + [model for model in os.listdir("inputs/image/sd_models/lora") if
                                 model.endswith(".safetensors") or model.endswith(".pt")]
    quantized_flux_models_list = [None] + [model for model in os.listdir("inputs/image/flux/quantize-flux") if
                                           model.endswith(".gguf") or model.endswith(".safetensors") or not model.endswith(".txt") and not model.endswith(
                                               ".safetensors") and not model.endswith(".py")]
    flux_lora_models_list = [None] + [model for model in os.listdir("inputs/image/flux/flux-lora") if
                                      model.endswith(".safetensors")]
    auraflow_lora_models_list = [None] + [model for model in os.listdir("inputs/image/auraflow-lora") if
                                          model.endswith(".safetensors")]
    kolors_lora_models_list = [None] + [model for model in os.listdir("inputs/image/kolors-lora") if
                                        model.endswith(".safetensors")]
    textual_inversion_models_list = [None] + [model for model in os.listdir("inputs/image/sd_models/embedding") if
                                              model.endswith(".pt") or model.endswith(".safetensors")]
    inpaint_models_list = [None] + [model for model in
                                    os.listdir("inputs/image/sd_models/inpaint")
                                    if (model.endswith(".safetensors") or model.endswith(".ckpt") or not model.endswith(
            ".txt"))]
    rvc_models_list = [model_folder for model_folder in os.listdir("inputs/audio/rvc_models")
                       if os.path.isdir(os.path.join("inputs/audio/rvc_models", model_folder))
                       and any(
            file.endswith('.pth') for file in os.listdir(os.path.join("inputs/audio/rvc_models", model_folder)))]

    chat_files = get_existing_chats()

    gallery_files = get_output_files()

    return [llm_models_list, llm_lora_models_list, speaker_wavs_list, stable_diffusion_models_list, vae_models_list, flux_vae_models_list, lora_models_list, quantized_flux_models_list, flux_lora_models_list, auraflow_lora_models_list, kolors_lora_models_list, textual_inversion_models_list, inpaint_models_list, rvc_models_list, chat_files, gallery_files]


def reload_interface():
    updated_lists = reload_model_lists()[:16]
    return [gr.Dropdown(choices=list) for list in updated_lists]


def create_footer():
    footer_html = """
    <div style="text-align: center; background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-top: 20px;">
        <span style="margin-right: 15px;">🔥 diffusers: 0.31.0</span>
        <span style="margin-right: 15px;">📄 transformers: 4.46.1</span>
        <span style="margin-right: 15px;">🦙 llama-cpp-python: 0.3.1</span>
        <span style="margin-right: 15px;">🖼️ stable-diffusion-cpp-python: 0.2.0</span>
        <span>ℹ️ gradio: 5.4.0</span>
    </div>
    """
    return gr.Markdown(footer_html)


avatar_user = "inputs/text/llm_models/avatars/user.png"
avatar_ai = "inputs/text/llm_models/avatars/ai.png"
settings = load_settings()
lang = settings['language']

chat_interface = gr.Interface(
    fn=generate_text_and_speech,
    inputs=[
        gr.Textbox(label=_("Enter your request", lang)),
        gr.Textbox(label=_("Enter your system prompt", lang)),
        gr.Audio(type="filepath", label=_("Record your request (optional)", lang)),
        gr.Radio(choices=["Transformers", "GPTQ", "AWQ", "BNB", "Llama", "ExLlamaV2"], label=_("Select model type", lang), value="Transformers"),
        gr.Dropdown(choices=llm_models_list, label=_("Select LLM model", lang), value=None),
        gr.Dropdown(choices=llm_lora_models_list, label=_("Select LoRA model (optional)", lang), value=None),
        gr.Dropdown(choices=get_existing_chats(), label=_("Select existing chat (optional)", lang), value=None)
    ],
    additional_inputs=[
        gr.Checkbox(label=_("Enable WebSearch", lang), value=False),
        gr.Checkbox(label=_("Enable LibreTranslate", lang), value=False),
        gr.Dropdown(choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hi"],
                    label=_("Select target language", lang), value="ru", interactive=True),
        gr.Checkbox(label=_("Enable OpenParse", lang), value=False),
        gr.File(label=_("Upload PDF file (for OpenParse)", lang), type="filepath"),
        gr.Checkbox(label=_("Enable Multimodal", lang), value=False),
        gr.Image(label=_("Upload your image (for Multimodal)", lang), type="filepath"),
        gr.Video(label=_("Upload your video (for Multimodal)", lang)),
        gr.Audio(label=_("Upload your audio (for Multimodal)", lang), type="filepath"),
        gr.Checkbox(label=_("Enable TTS", lang), value=False),
        gr.HTML(_("<h3>LLM Settings</h3>", lang)),
        gr.Slider(minimum=256, maximum=32768, value=512, step=1, label=_("Max tokens", lang)),
        gr.Slider(minimum=256, maximum=32768, value=512, step=1, label=_("Max length", lang)),
        gr.Slider(minimum=256, maximum=32768, value=512, step=1, label=_("Min length", lang)),
        gr.Slider(minimum=0, maximum=32768, value=0, step=1, label=_("Context size (N_CTX) for llama type models", lang)),
        gr.Slider(minimum=0, maximum=32768, value=0, step=1, label=_("Context batch (N_BATCH) for llama type models", lang)),
        gr.Slider(minimum=0, maximum=32768, value=0, step=1, label=_("Physical batch size (N_UBATCH) for llama type models", lang)),
        gr.Slider(minimum=0, maximum=1000000, value=0, step=1, label=_("RoPE base frequency (FREQ_BASE) for llama type models", lang)),
        gr.Slider(minimum=0, maximum=10000, value=0, step=1, label=_("RoPE frequency scaling factor (FREQ_SCALE) for llama type models", lang)),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label=_("Temperature", lang)),
        gr.Slider(minimum=0.01, maximum=1.0, value=0.9, step=0.01, label=_("Top P", lang)),
        gr.Slider(minimum=0.01, maximum=1.0, value=0.05, step=0.01, label=_("Min P", lang)),
        gr.Slider(minimum=0.01, maximum=1.0, value=1.0, step=0.01, label=_("Typical P", lang)),
        gr.Slider(minimum=1, maximum=200, value=40, step=1, label=_("Top K", lang)),
        gr.Checkbox(label=_("Enable Do Sample", lang), value=False),
        gr.Checkbox(label=_("Enable Early Stopping", lang), value=False),
        gr.Textbox(label=_("Stop sequences (optional)", lang), value=""),
        gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label=_("Repetition penalty", lang)),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.0, step=0.1, label=_("Frequency penalty", lang)),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.0, step=0.1, label=_("Presence penalty", lang)),
        gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label=_("Length penalty", lang)),
        gr.Slider(minimum=1, maximum=10, value=2, step=1, label=_("No repeat ngram size", lang)),
        gr.Slider(minimum=1, maximum=10, value=1, step=1, label=_("Num beams", lang)),
        gr.Slider(minimum=1, maximum=10, value=1, step=1, label=_("Num return sequences", lang)),
        gr.Radio(choices=["txt", "json"], label=_("Select chat history format", lang), value="txt", interactive=True),
        gr.HTML(_("<h3>TTS Settings</h3>", lang)),
        gr.Dropdown(choices=speaker_wavs_list, label=_("Select voice", lang), interactive=True),
        gr.Dropdown(choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"], label=_("Select language", lang), interactive=True),
        gr.Slider(minimum=0.1, maximum=1.9, value=1.0, step=0.1, label=_("TTS Temperature", lang), interactive=True),
        gr.Slider(minimum=0.01, maximum=1.0, value=0.9, step=0.01, label=_("TTS Top P", lang), interactive=True),
        gr.Slider(minimum=1, maximum=100, value=20, step=1, label=_("TTS Top K", lang), interactive=True),
        gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label=_("TTS Speed", lang), interactive=True),
        gr.Slider(minimum=0.1, maximum=1.9, value=1.1, step=0.1, label=_("TTS Repetition penalty", lang), interactive=True),
        gr.Slider(minimum=0.1, maximum=1.9, value=1.1, step=0.1, label=_("TTS Length penalty", lang), interactive=True),
        gr.Radio(choices=["wav", "mp3", "ogg"], label=_("Select output format", lang), value="wav", interactive=True),
        gr.State([])
    ],
    additional_inputs_accordion=gr.Accordion(label=_("LLM and TTS Settings", lang), open=False),
    outputs=[
        gr.Chatbot(label=_("LLM text response", lang), avatar_images=[avatar_user, avatar_ai], show_copy_button=True),
        gr.Audio(label=_("LLM audio response", lang), type="filepath")
    ],
    title=_("NeuroSandboxWebUI - LLM", lang),
    description=_("This user interface allows you to enter any text or audio and receive generated response. You can select the LLM model, avatar, voice and language for tts from the drop-down lists. You can also customize the model settings from the sliders. Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

tts_stt_interface = gr.Interface(
    fn=generate_tts_stt,
    inputs=[
        gr.Textbox(label=_("Enter text for TTS", lang)),
        gr.Audio(label=_("Record audio for STT", lang), type="filepath")
    ],
    additional_inputs=[
        gr.HTML(_("<h3>TTS Settings</h3>", lang)),
        gr.Dropdown(choices=speaker_wavs_list, label=_("Select voice", lang), interactive=True),
        gr.Dropdown(choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"], label=_("Select language", lang), interactive=True),
        gr.Slider(minimum=0.1, maximum=1.9, value=1.0, step=0.1, label=_("TTS Temperature", lang), interactive=True),
        gr.Slider(minimum=0.01, maximum=1.0, value=0.9, step=0.01, label=_("TTS Top P", lang), interactive=True),
        gr.Slider(minimum=1, maximum=100, value=20, step=1, label=_("TTS Top K", lang), interactive=True),
        gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label=_("TTS Speed", lang), interactive=True),
        gr.Slider(minimum=0.1, maximum=1.9, value=1.1, step=0.1, label=_("TTS Repetition penalty", lang), interactive=True),
        gr.Slider(minimum=0.1, maximum=1.9, value=1.1, step=0.1, label=_("TTS Length penalty", lang), interactive=True),
        gr.Checkbox(label=_("Enable Voice Conversion", lang), value=False),
        gr.Audio(label=_("Source wav", lang), type="filepath"),
        gr.Audio(label=_("Target wav", lang), type="filepath"),
        gr.Radio(choices=["wav", "mp3", "ogg"], label=_("Select TTS output format", lang), value="wav", interactive=True),
        gr.Dropdown(choices=["txt", "json"], label=_("Select STT output format", lang), value="txt", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("TTS and STT Settings", lang), open=False),
    outputs=[
        gr.Audio(label=_("TTS Audio", lang), type="filepath"),
        gr.Textbox(label=_("STT Text", lang))
    ],
    title=_("NeuroSandboxWebUI - TTS-STT", lang),
    description=_("This user interface allows you to enter text for Text-to-Speech(CoquiTTS) and record audio for Speech-to-Text(OpenAIWhisper). For TTS, you can select the voice and language, and customize the generation settings from the sliders. For STT, simply record your audio and the spoken text will be displayed. Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

mms_tts_interface = gr.Interface(
    fn=generate_mms_tts,
    inputs=[
        gr.Textbox(label=_("Enter text to synthesize", lang)),
        gr.Dropdown(choices=["English", "Russian", "Korean", "Hindu", "Turkish", "French", "Spanish", "German", "Arabic", "Polish"], label=_("Select language", lang), value="English"),
        gr.Radio(choices=["wav", "mp3", "ogg"], label=_("Select output format", lang), value="wav")
    ],
    outputs=[
        gr.Audio(label=_("Synthesized speech", lang), type="filepath")
    ],
    title=_("NeuroSandboxWebUI - MMS Text-to-Speech", lang),
    description=_("Generate speech from text using MMS TTS models.", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

mms_stt_interface = gr.Interface(
    fn=transcribe_mms_stt,
    inputs=[
        gr.Audio(label=_("Upload or record audio", lang), type="filepath"),
        gr.Dropdown(choices=["en", "ru", "ko", "hi", "tr", "fr", "sp", "de", "ar", "pl"], label=_("Select language", lang), value="En"),
        gr.Radio(choices=["txt", "json"], label=_("Select output format", lang), value="txt")
    ],
    outputs=[
        gr.Textbox(label=_("Transcription", lang))
    ],
    title=_("NeuroSandboxWebUI - MMS Speech-to-Text", lang),
    description=_("Transcribe speech to text using MMS STT model.", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

mms_interface = gr.TabbedInterface(
    [mms_tts_interface, mms_stt_interface],
    tab_names=[_("Text-to-Speech", lang), _("Speech-to-Text", lang)]
)

seamless_m4tv2_interface = gr.Interface(
    fn=seamless_m4tv2_process,
    inputs=[
        gr.Radio(choices=["Text", "Audio"], label=_("Input Type", lang), value="Text", interactive=True),
        gr.Textbox(label=_("Input Text", lang)),
        gr.Audio(label=_("Input Audio", lang), type="filepath"),
        gr.Dropdown(choices=get_languages(), label=_("Source Language", lang), value=None, interactive=True),
        gr.Dropdown(choices=get_languages(), label=_("Target Language", lang), value=None, interactive=True),
        gr.Dropdown(choices=["en", "ru", "ko", "hi", "tr", "fr", "sp", "de", "ar", "pl"], label=_("Dataset Language", lang), value="En", interactive=True)
    ],
    additional_inputs=[
        gr.Checkbox(label=_("Enable Speech Generation", lang), value=False),
        gr.Number(label=_("Speaker ID", lang), value=0),
        gr.Slider(minimum=1, maximum=10, value=4, step=1, label=_("Text Num Beams", lang)),
        gr.Checkbox(label=_("Enable Text Sampling", lang)),
        gr.Checkbox(label=_("Enable Speech Sampling", lang)),
        gr.Slider(minimum=0.1, maximum=2, value=0.6, step=0.1, label=_("Speech Temperature", lang)),
        gr.Slider(minimum=0.1, maximum=2, value=0.6, step=0.1, label=_("Text Temperature", lang)),
        gr.Radio(choices=["General", "Speech to Speech", "Text to Text", "Speech to Text", "Text to Speech"], label=_("Task Type", lang), value="General", interactive=True),
        gr.Radio(choices=["txt", "json"], label=_("Text Output Format", lang), value="txt", interactive=True),
        gr.Radio(choices=["wav", "mp3", "ogg"], label=_("Audio Output Format", lang), value="wav", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("SeamlessM4T Settings", lang), open=False),
    outputs=[
        gr.Textbox(label=_("Generated Text", lang)),
        gr.Audio(label=_("Generated Audio", lang), type="filepath")
    ],
    title=_("NeuroSandboxWebUI - SeamlessM4Tv2", lang),
    description=_("This interface allows you to use the SeamlessM4Tv2 model for various translation and speech tasks.", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Translate", lang)
)

translate_interface = gr.Interface(
    fn=translate_text,
    inputs=[
        gr.Textbox(label=_("Enter text to translate", lang)),
        gr.Dropdown(choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hi"], label=_("Select source language", lang), value="en"),
        gr.Dropdown(choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hi"], label=_("Select target language", lang), value="ru")
    ],
    additional_inputs=[
        gr.Checkbox(label=_("Enable translate history save", lang), value=False),
        gr.Radio(choices=["txt", "json"], label=_("Select translate history format", lang), value="txt", interactive=True),
        gr.File(label=_("Upload text file (optional)", lang), file_count="single", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Additional LibreTranslate Settings", lang), open=False),
    outputs=[
        gr.Textbox(label=_("Translated text", lang))
    ],
    title=_("NeuroSandboxWebUI - LibreTranslate", lang),
    description=_("This user interface allows you to enter text and translate it using LibreTranslate. Select the source and target languages and click Submit to get the translation. Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Translate", lang)
)

txt2img_interface = gr.Interface(
    fn=generate_image_txt2img,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang), placeholder=_("+ and - for Weighting; ('p', 'p').blend(0.x, 0.x) for Blending; ['p', 'p', 'p'].and() for Conjunction", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Dropdown(choices=list(styles.keys()), label=_("Select Style (optional)", lang), value=None),
        gr.Dropdown(choices=stable_diffusion_models_list, label=_("Select StableDiffusion model", lang), value=None),
        gr.Checkbox(label=_("Enable Quantize", lang), value=False),
        gr.Dropdown(choices=vae_models_list, label=_("Select VAE model (optional)", lang), value=None),
        gr.Dropdown(choices=lora_models_list, label=_("Select LORA models (optional)", lang), value=None, multiselect=True),
        gr.Textbox(label=_("LoRA Scales", lang)),
        gr.Dropdown(choices=textual_inversion_models_list, label=_("Select Embedding models (optional)", lang), value=None, multiselect=True),
        gr.HTML(_("<h3>StableDiffusion Settings</h3>", lang)),
        gr.Radio(choices=["SD", "SD2", "SDXL"], label=_("Select model type", lang), value="SD"),
        gr.Dropdown(choices=[
            "EulerDiscreteScheduler", "DPMSolverSinglestepScheduler", "DPMSolverMultistepScheduler",
            "EDMDPMSolverMultistepScheduler", "EDMEulerScheduler", "KDPM2DiscreteScheduler",
            "KDPM2AncestralDiscreteScheduler", "EulerAncestralDiscreteScheduler",
            "HeunDiscreteScheduler", "LMSDiscreteScheduler", "DEISMultistepScheduler",
            "UniPCMultistepScheduler", "LCMScheduler", "DPMSolverSDEScheduler",
            "TCDScheduler", "DDIMScheduler", "PNDMScheduler", "DDPMScheduler"
        ], label=_("Select scheduler", lang), value="EulerDiscreteScheduler"),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label=_("CFG", lang)),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label=_("Width", lang)),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label=_("Height", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Clip skip", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Number of images to generate", lang)),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Checkbox(label=_("Enable FreeU", lang), value=False),
        gr.Slider(minimum=0.1, maximum=4, value=0.9, step=0.1, label=_("FreeU-S1", lang)),
        gr.Slider(minimum=0.1, maximum=4, value=0.2, step=0.1, label=_("FreeU-S2", lang)),
        gr.Slider(minimum=0.1, maximum=4, value=1.2, step=0.1, label=_("FreeU-B1", lang)),
        gr.Slider(minimum=0.1, maximum=4, value=1.4, step=0.1, label=_("FreeU-B2", lang)),
        gr.Checkbox(label=_("Enable SAG", lang), value=False),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.75, step=0.01, label=_("SAG Scale", lang)),
        gr.Checkbox(label=_("Enable PAG", lang), value=False),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.01, label=_("PAG Scale", lang)),
        gr.Checkbox(label=_("Enable Token Merging", lang), value=False),
        gr.Slider(minimum=0.01, maximum=1.0, value=0.5, step=0.01, label=_("Token Merging Ratio", lang)),
        gr.Checkbox(label=_("Enable DeepCache", lang), value=False),
        gr.Slider(minimum=1, maximum=5, value=3, step=1, label=_("DeepCache Interval", lang)),
        gr.Slider(minimum=0, maximum=1, value=0, step=1, label=_("DeepCache BranchID", lang)),
        gr.Checkbox(label=_("Enable T-GATE", lang), value=False),
        gr.Slider(minimum=1, maximum=50, value=10, step=1, label=_("T-GATE steps", lang)),
        gr.Checkbox(label=_("Enable MagicPrompt", lang), value=False),
        gr.Slider(minimum=32, maximum=256, value=50, step=1, label=_("MagicPrompt Max New Tokens", lang)),
        gr.Checkbox(label=_("Enable CDVAE", lang), value=False),
        gr.Checkbox(label=_("Enable TAESD", lang), value=False),
        gr.Checkbox(label=_("Enable MultiDiffusion", lang), value=False),
        gr.Checkbox(label=_("Enable Circular padding (for MultiDiffusion)", lang), value=False),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Additional StableDiffusion Settings", lang), open=False),
    outputs=[
        gr.Gallery(label=_("Generated images", lang), elem_id="gallery", columns=[2], rows=[2], object_fit="contain", height="auto"),
        gr.Gallery(label=_("Generation process", lang), elem_id="process_gallery", columns=[2], rows=[2], object_fit="contain", height="auto"),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - StableDiffusion (txt2img)", lang),
    description=_("This user interface allows you to enter any text and generate images using StableDiffusion. You can select the model and customize the generation settings from the sliders. Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

img2img_interface = gr.Interface(
    fn=generate_image_img2img,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang), placeholder=_("+ and - for Weighting; ('p', 'p').blend(0.x, 0.x) for Blending; ['p', 'p', 'p'].and() for Conjunction", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Image(label=_("Initial image", lang), type="filepath"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label=_("Strength (Initial image)", lang)),
        gr.Radio(choices=["SD", "SD2", "SDXL"], label=_("Select model type", lang), value="SD"),
        gr.Dropdown(choices=stable_diffusion_models_list, label=_("Select StableDiffusion model", lang), value=None),
        gr.Checkbox(label=_("Enable Quantize", lang), value=False),
        gr.Dropdown(choices=vae_models_list, label=_("Select VAE model (optional)", lang), value=None),
        gr.Dropdown(choices=lora_models_list, label=_("Select LORA models (optional)", lang), value=None, multiselect=True),
        gr.Textbox(label=_("LoRA Scales", lang)),
        gr.Dropdown(choices=textual_inversion_models_list, label=_("Select Embedding models (optional)", lang), value=None, multiselect=True),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Dropdown(choices=[
            "EulerDiscreteScheduler", "DPMSolverSinglestepScheduler", "DPMSolverMultistepScheduler",
            "EDMDPMSolverMultistepScheduler", "EDMEulerScheduler", "KDPM2DiscreteScheduler",
            "KDPM2AncestralDiscreteScheduler", "EulerAncestralDiscreteScheduler",
            "HeunDiscreteScheduler", "LMSDiscreteScheduler", "DEISMultistepScheduler",
            "UniPCMultistepScheduler", "LCMScheduler", "DPMSolverSDEScheduler",
            "TCDScheduler", "DDIMScheduler", "PNDMScheduler", "DDPMScheduler"
        ], label=_("Select scheduler", lang), value="EulerDiscreteScheduler"),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label=_("CFG", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Clip skip", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Number of images to generate", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("StableDiffusion Settings", lang), open=False),
    outputs=[
        gr.Gallery(label=_("Generated images", lang), elem_id="gallery", columns=[2], rows=[2], object_fit="contain", height="auto"),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - StableDiffusion (img2img)", lang),
    description=_("This user interface allows you to enter any text and image to generate new images using StableDiffusion. You can select the model and customize the generation settings from the sliders. Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

depth2img_interface = gr.Interface(
    fn=generate_image_depth2img,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Image(label=_("Initial image", lang), type="filepath"),
        gr.Textbox(label=_("Seed (optional)", lang), value="")
    ],
    additional_inputs=[
        gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.01, label=_("Strength", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Clip skip", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Number of images to generate", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("StableDiffusion Settings", lang), open=False),
    outputs=[
        gr.Gallery(label=_("Generated images", lang), elem_id="gallery", columns=[2], rows=[2], object_fit="contain", height="auto"),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - StableDiffusion (depth2img)", lang),
    description=_("This user interface allows you to enter a prompt, an initial image to generate depth-aware images using StableDiffusion. Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

marigold_interface = gr.Interface(
    fn=generate_image_marigold,
    inputs=[
        gr.Image(label=_("Input image", lang), type="filepath"),
        gr.Slider(minimum=1, maximum=30, value=15, step=1, label=_("Num inference steps", lang)),
        gr.Slider(minimum=1, maximum=10, value=5, step=1, label=_("Ensemble size", lang))
    ],
    outputs=[
        gr.Image(label=_("Depth image", lang)),
        gr.Image(label=_("Normals image", lang)),
        gr.Textbox(label=_("Message", lang))
    ],
    title=_("NeuroSandboxWebUI - Marigold", lang),
    description=_("This interface allows you to generate depth and normal maps using Marigold models. Upload an image and adjust the inference steps and ensemble size. The model will generate both depth and normal maps.", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

pix2pix_interface = gr.Interface(
    fn=generate_image_pix2pix,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Image(label=_("Initial image", lang), type="filepath"),
        gr.Textbox(label=_("Seed (optional)", lang), value="")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label=_("CFG", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Clip skip", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Number of images to generate", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Pix2Pix Settings", lang), open=False),
    outputs=[
        gr.Gallery(label=_("Generated images", lang), elem_id="gallery", columns=[2], rows=[2], object_fit="contain", height="auto"),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - StableDiffusion (pix2pix)", lang),
    description=_("This user interface allows you to enter a prompt and an initial image to generate new images using Pix2Pix. You can customize the generation settings from the sliders. Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

controlnet_interface = gr.Interface(
    fn=generate_image_controlnet,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Image(label=_("Initial image", lang), type="filepath"),
        gr.Radio(choices=["SD", "SDXL"], label=_("Select model type", lang), value="SD"),
        gr.Dropdown(choices=[
            "EulerDiscreteScheduler", "DPMSolverSinglestepScheduler", "DPMSolverMultistepScheduler",
            "EDMDPMSolverMultistepScheduler", "EDMEulerScheduler", "KDPM2DiscreteScheduler",
            "KDPM2AncestralDiscreteScheduler", "EulerAncestralDiscreteScheduler",
            "HeunDiscreteScheduler", "LMSDiscreteScheduler", "DEISMultistepScheduler",
            "UniPCMultistepScheduler", "LCMScheduler", "DPMSolverSDEScheduler",
            "TCDScheduler", "DDIMScheduler", "PNDMScheduler", "DDPMScheduler"
        ], label=_("Select scheduler", lang), value="EulerDiscreteScheduler"),
        gr.Dropdown(choices=stable_diffusion_models_list, label=_("Select StableDiffusion model", lang), value=None),
        gr.Dropdown(choices=controlnet_models_list, label=_("Select ControlNet model", lang), value=None),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label=_("CFG", lang)),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label=_("Width", lang)),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label=_("Height", lang)),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.1, label=_("ControlNet conditioning scale", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Clip skip", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Number of images to generate", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("StableDiffusion Settings", lang), open=False),
    outputs=[
        gr.Gallery(label=_("Generated images", lang), elem_id="gallery", columns=[2], rows=[2], object_fit="contain", height="auto"),
        gr.Gallery(label=_("ControlNet control images", lang), elem_id="gallery", columns=[2], rows=[2], object_fit="contain", height="auto"),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - StableDiffusion (controlnet)", lang),
    description=_("This user interface allows you to generate images using ControlNet models. Upload an initial image, enter a prompt, select a Stable Diffusion model, and customize the generation settings. Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

latent_upscale_interface = gr.Interface(
    fn=generate_image_upscale_latent,
    inputs=[
        gr.Textbox(label=_("Prompt (optional)", lang), value=""),
        gr.Image(label=_("Image to upscale", lang), type="filepath"),
        gr.Radio(choices=["x2", "x4"], label=_("Upscale size", lang), value="x2"),
        gr.Textbox(label=_("Seed (optional)", lang), value="")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=0.0, maximum=30.0, value=4, step=0.1, label=_("CFG", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Upscale-latent Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Upscaled image", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - StableDiffusion (upscale-latent)", lang),
    description=_("This user interface allows you to upload an image and latent-upscale it using x2 or x4 upscale factor", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Upscale", lang)
)

sdxl_refiner_interface = gr.Interface(
    fn=generate_image_sdxl_refiner,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Image(label=_("Initial image", lang), type="filepath"),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    outputs=[
        gr.Image(type="filepath", label=_("Refined image", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - SDXL Refiner", lang),
    description=_("This interface allows you to refine images using the SDXL Refiner model. Enter a prompt, upload an initial image, and see the refined result.", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Refine", lang)
)

inpaint_interface = gr.Interface(
    fn=generate_image_inpaint,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Image(label=_("Initial image", lang), type="filepath"),
        gr.ImageEditor(label=_("Mask image", lang), type="filepath"),
        gr.Slider(minimum=0, maximum=100, value=0, step=1, label=_("Mask Blur Factor", lang)),
        gr.Radio(choices=["SD", "SD2", "SDXL"], label=_("Select model type", lang), value="SD"),
        gr.Dropdown(choices=[
            "EulerDiscreteScheduler", "DPMSolverSinglestepScheduler", "DPMSolverMultistepScheduler",
            "EDMDPMSolverMultistepScheduler", "EDMEulerScheduler", "KDPM2DiscreteScheduler",
            "KDPM2AncestralDiscreteScheduler", "EulerAncestralDiscreteScheduler",
            "HeunDiscreteScheduler", "LMSDiscreteScheduler", "DEISMultistepScheduler",
            "UniPCMultistepScheduler", "LCMScheduler", "DPMSolverSDEScheduler",
            "TCDScheduler", "DDIMScheduler", "PNDMScheduler", "DDPMScheduler"
        ], label=_("Select scheduler", lang), value="EulerDiscreteScheduler"),
        gr.Dropdown(choices=inpaint_models_list, label=_("Select Inpaint model", lang), value=None),
        gr.Dropdown(choices=vae_models_list, label=_("Select VAE model (optional)", lang), value=None),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label=_("CFG", lang)),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label=_("Width", lang)),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label=_("Height", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Clip skip", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Number of images to generate", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Inpaint Settings", lang), open=False),
    outputs=[
        gr.Gallery(label=_("Generated images", lang), elem_id="gallery", columns=[2], rows=[2], object_fit="contain", height="auto"),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - StableDiffusion (inpaint)", lang),
    description=_("This user interface allows you to enter a prompt, an initial image, and a mask image to inpaint using StableDiffusion. You can select the model and customize the generation settings from the sliders. Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

outpaint_interface = gr.Interface(
    fn=generate_image_outpaint,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Image(label=_("Initial image", lang), type="filepath"),
        gr.Radio(choices=["SD", "SD2", "SDXL"], label=_("Select model type", lang), value="SD"),
        gr.Dropdown(choices=inpaint_models_list, label=_("Select StableDiffusion model", lang), value=None),
        gr.Textbox(label=_("Seed (optional)", lang), value="")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label=_("CFG", lang)),
        gr.Radio(choices=["left", "right", "up", "down"], label=_("Outpaint direction", lang), value="right"),
        gr.Slider(minimum=10, maximum=200, value=50, step=1, label=_("Expansion percentage", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Clip skip", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Number of images to generate", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Outpaint Settings", lang), open=False),
    outputs=[
        gr.Gallery(label=_("Generated images", lang), elem_id="gallery", columns=[2], rows=[2], object_fit="contain", height="auto"),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - StableDiffusion (outpaint)", lang),
    description=_("This user interface allows you to expand an existing image using outpainting with StableDiffusion. Upload an image, enter a prompt, select a model type and direction to expand, and customize the generation settings. The image will be expanded according to the chosen percentage. Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

gligen_interface = gr.Interface(
    fn=generate_image_gligen,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Textbox(label=_("Enter GLIGEN phrases", lang), value=""),
        gr.Textbox(label=_("Enter GLIGEN boxes", lang), value=""),
        gr.Radio(choices=["SD", "SD2", "SDXL"], label=_("Select model type", lang), value="SD"),
        gr.Dropdown(choices=stable_diffusion_models_list, label=_("Select StableDiffusion model", lang), value=None),
        gr.Textbox(label=_("Seed (optional)", lang), value="")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label=_("CFG", lang)),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label=_("Width", lang)),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label=_("Height", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Clip skip", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Number of images to generate", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("StableDiffusion Settings", lang), open=False),
    outputs=[
        gr.Gallery(label=_("Generated images", lang), elem_id="gallery", columns=[2], rows=[2], object_fit="contain", height="auto"),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - StableDiffusion (gligen)", lang),
    description=_("This user interface allows you to generate images using Stable Diffusion and insert objects using GLIGEN. "
                "Select the Stable Diffusion model, customize the generation settings, enter a prompt, GLIGEN phrases, and bounding boxes. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

diffedit_interface = gr.Interface(
    fn=generate_image_diffedit,
    inputs=[
        gr.Textbox(label=_("Source Prompt", lang)),
        gr.Textbox(label=_("Source Negative Prompt", lang), value=""),
        gr.Textbox(label=_("Target Prompt", lang)),
        gr.Textbox(label=_("Target Negative Prompt", lang), value=""),
        gr.Image(label=_("Initial image", lang), type="filepath"),
        gr.Textbox(label=_("Seed (optional)", lang), value="")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label=_("Guidance Scale", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("DiffEdit Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Generated image", lang)),
        gr.Image(type="filepath", label=_("Mask image", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - StableDiffusion (DiffEdit)", lang),
    description=_("This user interface allows you to edit images using DiffEdit. "
                "Upload an image, provide source and target prompts, and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

blip_diffusion_interface = gr.Interface(
    fn=generate_image_blip_diffusion,
    inputs=[
        gr.Textbox(label=_("Prompt", lang)),
        gr.Textbox(label=_("Negative Prompt", lang), value=""),
        gr.Image(label=_("Conditioning Image", lang), type="filepath"),
        gr.Textbox(label=_("Conditioning Subject", lang)),
        gr.Textbox(label=_("Target Subject", lang)),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label=_("Inference Steps", lang)),
        gr.Slider(minimum=0.1, maximum=30.0, value=8, step=0.1, label=_("Guidance Scale", lang)),
        gr.Slider(minimum=64, maximum=2048, value=512, step=64, label=_("Height", lang)),
        gr.Slider(minimum=64, maximum=2048, value=512, step=64, label=_("Width", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Output Format", lang), value="png")
    ],
    outputs=[
        gr.Image(type="filepath", label=_("Generated Image", lang)),
        gr.Textbox(label=_("Message", lang))
    ],
    title=_("NeuroSandboxWebUI - BlipDiffusion", lang),
    description=_("This interface allows you to generate images using BlipDiffusion. Upload a conditioning image, provide text prompts and subjects, and customize generation parameters.", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

animatediff_interface = gr.Interface(
    fn=generate_image_animatediff,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Image(label=_("Initial GIF", lang), type="filepath"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label=_("Strength (Initial GIF)", lang)),
        gr.Radio(choices=["sd", "sdxl"], label=_("Select model type", lang), value="sd"),
        gr.Dropdown(choices=stable_diffusion_models_list, label=_("Select StableDiffusion model", lang), value=None),
        gr.Textbox(label=_("Seed (optional)", lang), value="")
    ],
    additional_inputs=[
        gr.Dropdown(choices=[None, "zoom-in", "zoom-out", "tilt-up", "tilt-down", "pan-right", "pan-left"],
                    label=_("Select Motion LORA (Optional)", lang), value=None),
        gr.Slider(minimum=2, maximum=25, value=16, step=1, label=_("Frames", lang)),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label=_("Guidance Scale", lang)),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label=_("Width", lang)),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label=_("Height", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Clip skip", lang))
    ],
    additional_inputs_accordion=gr.Accordion(label=_("AnimateDiff Settings", lang), open=False),
    outputs=[
        gr.Image(label=_("Generated GIF", lang), type="filepath"),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - StableDiffusion (animatediff)", lang),
    description=_("This user interface allows you to enter a prompt and generate animated GIFs using AnimateDiff. "
                "You can select the model and customize the generation settings from the sliders. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

hotshotxl_interface = gr.Interface(
    fn=generate_hotshotxl,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value="")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label=_("Width", lang)),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label=_("Height", lang)),
        gr.Slider(minimum=2, maximum=80, value=8, step=1, label=_("Video Length (frames)", lang)),
        gr.Slider(minimum=100, maximum=10000, value=1000, step=1, label=_("Video Duration (seconds)", lang))
    ],
    additional_inputs_accordion=gr.Accordion(label=_("HotShot-XL Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Generated GIF", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - Hotshot-XL", lang),
    description=_("This user interface allows you to generate animated GIFs using Hotshot-XL. "
                "Enter a prompt and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

video_interface = gr.Interface(
    fn=generate_video,
    inputs=[
        gr.Image(label=_("Initial image", lang), type="filepath"),
        gr.Radio(choices=["mp4", "gif"], label=_("Select output format", lang), value="mp4", interactive=True),
        gr.Textbox(label=_("Seed (optional)", lang), value="")
    ],
    additional_inputs=[
        gr.HTML(_("<h3>SVD Settings (mp4)</h3>", lang)),
        gr.Slider(minimum=0, maximum=360, value=180, step=1, label=_("Motion Bucket ID", lang)),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.01, label=_("Noise Augmentation Strength", lang)),
        gr.Slider(minimum=1, maximum=60, value=10, step=1, label=_("FPS", lang)),
        gr.Slider(minimum=2, maximum=120, value=25, step=1, label=_("Frames", lang)),
        gr.Slider(minimum=1, maximum=32, value=8, step=1, label=_("Decode Chunk Size", lang)),
        gr.HTML(_("<h3>I2VGen-xl Settings (gif)</h3>", lang)),
        gr.Textbox(label=_("Prompt", lang), value=""),
        gr.Textbox(label=_("Negative Prompt", lang), value=""),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1.0, maximum=30.0, value=9.0, step=0.1, label=_("CFG", lang))
    ],
    additional_inputs_accordion=gr.Accordion(label=_("SD-Video Settings", lang), open=False),
    outputs=[
        gr.Video(label=_("Generated video", lang)),
        gr.Image(label=_("Generated GIF", lang), type="filepath"),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - StableDiffusion (video)", lang),
    description=_("This user interface allows you to enter an initial image and generate a video using StableVideoDiffusion(mp4) and I2VGen-xl(gif). "
                "You can customize the generation settings from the sliders. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

ldm3d_interface = gr.Interface(
    fn=generate_image_ldm3d,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Textbox(label=_("Seed (optional)", lang), value="")
    ],
    additional_inputs=[
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label=_("Width", lang)),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label=_("Height", lang)),
        gr.Slider(minimum=1, maximum=100, value=40, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label=_("Guidance Scale", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Number of images to generate", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("LDM3D Settings", lang), open=False),
    outputs=[
        gr.Gallery(label=_("Generated RGBs", lang), elem_id="gallery", columns=[2], rows=[2], object_fit="contain", height="auto"),
        gr.Gallery(label=_("Generated Depth images", lang), elem_id="gallery", columns=[2], rows=[2], object_fit="contain", height="auto"),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - StableDiffusion (LDM3D)", lang),
    description=_("This user interface allows you to enter a prompt and generate RGB and Depth images using LDM3D. "
                "You can customize the generation settings from the sliders. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

sd3_txt2img_interface = gr.Interface(
    fn=generate_image_sd3_txt2img,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang), placeholder=_("(prompt:x.x) for Weighting", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), placeholder=_("(prompt:x.x) for Weighting", lang), value=""),
        gr.Radio(choices=["Diffusers", "Safetensors"], label=_("Select model type", lang), value="Diffusers"),
        gr.Radio(choices=["3-Medium", "3.5-Large", "3.5-Large-Turbo"], label=_("Select Diffusers model", lang), value="3-Medium"),
        gr.Dropdown(choices=stable_diffusion_models_list, label=_("Select StableDiffusion model", lang), value=None),
        gr.Checkbox(label=_("Enable Quantize", lang), value=False),
        gr.Radio(choices=["SD3", "SD3.5"], label=_("Select quantize model type", lang), value="SD3"),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Dropdown(choices=vae_models_list, label=_("Select VAE model (optional)", lang), value=None),
        gr.Dropdown(choices=lora_models_list, label=_("Select LORA models (optional)", lang), value=None, multiselect=True),
        gr.Textbox(label=_("LoRA Scales", lang)),
        gr.Dropdown(choices=["FlowMatchEulerDiscreteScheduler", "FlowMatchHeunDiscreteScheduler"], label=_("Select scheduler", lang), value="FlowMatchEulerDiscreteScheduler"),
        gr.Slider(minimum=1, maximum=100, value=40, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1.0, maximum=30.0, value=8.0, step=0.1, label=_("CFG", lang)),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label=_("Width", lang)),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label=_("Height", lang)),
        gr.Slider(minimum=64, maximum=2048, value=256, label=_("Max Length", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Clip skip", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Number of images to generate", lang)),
        gr.Checkbox(label=_("Enable TAESD", lang), value=False),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("StableDiffusion3 Settings", lang), open=False),
    outputs=[
        gr.Gallery(label=_("Generated images", lang), elem_id="gallery", columns=[2], rows=[2], object_fit="contain", height="auto"),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - StableDiffusion 3 (txt2img)", lang),
    description=_("This user interface allows you to enter any text and generate images using Stable Diffusion 3. "
                "You can customize the generation settings from the sliders. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

sd3_img2img_interface = gr.Interface(
    fn=generate_image_sd3_img2img,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Image(label=_("Initial image", lang), type="filepath"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.8, step=0.01, label=_("Strength (Initial image)", lang)),
        gr.Radio(choices=["3-Medium", "3.5-Large", "3.5-Large-Turbo"], label=_("Select Diffusers model", lang), value="3-Medium"),
        gr.Dropdown(choices=stable_diffusion_models_list, label=_("Select StableDiffusion model", lang), value=None),
        gr.Checkbox(label=_("Enable Quantize", lang), value=False),
        gr.Radio(choices=["SD3", "SD3.5"], label=_("Select quantize model type", lang), value="SD3"),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Dropdown(choices=["FlowMatchEulerDiscreteScheduler", "FlowMatchHeunDiscreteScheduler"],
                    label=_("Select scheduler", lang), value="FlowMatchEulerDiscreteScheduler"),
        gr.Slider(minimum=1, maximum=100, value=40, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1.0, maximum=30.0, value=8.0, step=0.1, label=_("CFG", lang)),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label=_("Width", lang)),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label=_("Height", lang)),
        gr.Slider(minimum=64, maximum=2048, value=256, label=_("Max Length", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Clip skip", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Number of images to generate", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("StableDiffusion3 Settings", lang), open=False),
    outputs=[
        gr.Gallery(label=_("Generated images", lang), elem_id="gallery", columns=[2], rows=[2], object_fit="contain", height="auto"),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - StableDiffusion 3 (img2img)", lang),
    description=_("This user interface allows you to enter any text and initial image to generate new images using Stable Diffusion 3. "
                "You can customize the generation settings from the sliders. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

sd3_controlnet_interface = gr.Interface(
    fn=generate_image_sd3_controlnet,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Image(label=_("Initial image", lang), type="filepath"),
        gr.Radio(choices=["3-Medium", "3.5-Large", "3.5-Large-Turbo"], label=_("Select Diffusers model", lang), value="3-Medium"),
        gr.Dropdown(choices=["Pose", "Canny"], label=_("Select ControlNet model", lang), value="Pose"),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Dropdown(choices=["FlowMatchEulerDiscreteScheduler", "FlowMatchHeunDiscreteScheduler"],
                    label=_("Select scheduler", lang), value="FlowMatchEulerDiscreteScheduler"),
        gr.Slider(minimum=1, maximum=100, value=40, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1.0, maximum=30.0, value=8.0, step=0.1, label=_("CFG", lang)),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.1, label=_("ControlNet conditioning scale", lang)),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label=_("Width", lang)),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label=_("Height", lang)),
        gr.Slider(minimum=64, maximum=2048, value=256, label=_("Max Length", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Clip skip", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Number of images to generate", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("StableDiffusion3 Settings", lang), open=False),
    outputs=[
        gr.Gallery(label=_("Generated images", lang), elem_id="gallery", columns=[2], rows=[2], object_fit="contain", height="auto"),
        gr.Gallery(label=_("ControlNet control images", lang), elem_id="gallery", columns=[2], rows=[2], object_fit="contain", height="auto"),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - StableDiffusion 3 (ControlNet)", lang),
    description=_("This user interface allows you to use ControlNet models with Stable Diffusion 3. "
                "You can customize the generation settings from the sliders. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

sd3_inpaint_interface = gr.Interface(
    fn=generate_image_sd3_inpaint,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Radio(choices=["3-Medium", "3.5-Large", "3.5-Large-Turbo"], label=_("Select Diffusers model", lang), value="3-Medium"),
        gr.Image(label=_("Initial image", lang), type="filepath"),
        gr.ImageEditor(label=_("Mask image", lang), type="filepath"),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Dropdown(choices=["FlowMatchEulerDiscreteScheduler", "FlowMatchHeunDiscreteScheduler"],
                    label=_("Select scheduler", lang), value="FlowMatchEulerDiscreteScheduler"),
        gr.Slider(minimum=1, maximum=100, value=40, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1.0, maximum=30.0, value=8.0, step=0.1, label=_("CFG", lang)),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label=_("Width", lang)),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label=_("Height", lang)),
        gr.Slider(minimum=64, maximum=2048, value=256, label=_("Max Length", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Clip skip", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Number of images to generate", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("StableDiffusion3 Settings", lang), open=False),
    outputs=[
        gr.Gallery(label=_("Generated images", lang), elem_id="gallery", columns=[2], rows=[2], object_fit="contain", height="auto"),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - StableDiffusion 3 (Inpaint)", lang),
    description=_("This user interface allows you to perform inpainting using Stable Diffusion 3. "
                "You can customize the generation settings from the sliders. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

cascade_interface = gr.Interface(
    fn=generate_image_cascade,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Slider(minimum=256, maximum=4096, value=1024, step=64, label=_("Width", lang)),
        gr.Slider(minimum=256, maximum=4096, value=1024, step=64, label=_("Height", lang)),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label=_("Prior Steps", lang)),
        gr.Slider(minimum=1.0, maximum=30.0, value=4.0, step=0.1, label=_("Prior Guidance Scale", lang)),
        gr.Slider(minimum=1, maximum=100, value=20, step=1, label=_("Decoder Steps", lang)),
        gr.Slider(minimum=0.0, maximum=30.0, value=8.0, step=0.1, label=_("Decoder Guidance Scale", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Number of images to generate", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("StableCascade Settings", lang), open=False),
    outputs=[
        gr.Gallery(label=_("Generated images", lang), elem_id="gallery", columns=[2], rows=[2], object_fit="contain", height="auto"),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - StableDiffusion (cascade)", lang),
    description=_("This user interface allows you to enter a prompt and generate images using Stable Cascade. "
                "You can customize the generation settings from the sliders. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

t2i_ip_adapter_interface = gr.Interface(
    fn=generate_image_t2i_ip_adapter,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Image(label=_("IP-Adapter Image", lang), type="filepath"),
        gr.Radio(choices=["SD", "SDXL"], label=_("Select model type", lang), value="SD"),
        gr.Dropdown(choices=stable_diffusion_models_list, label=_("Select StableDiffusion model", lang), value=None),
        gr.Textbox(label=_("Seed (optional)", lang), value="")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.1, label=_("Guidance Scale", lang)),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label=_("Width", lang)),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label=_("Height", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("T2I IP-Adapter Settings", lang), open=False),
    outputs=[
        gr.Gallery(label=_("Generated images", lang), elem_id="gallery", columns=[2], rows=[2], object_fit="contain", height="auto"),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - StableDiffusion (T2I IP-Adapter)", lang),
    description=_("This user interface allows you to generate images using T2I IP-Adapter. "
                "Upload an image, enter a prompt, select a Stable Diffusion model, and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

ip_adapter_faceid_interface = gr.Interface(
    fn=generate_image_ip_adapter_faceid,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Image(label=_("Face image", lang), type="filepath"),
        gr.Slider(minimum=0.1, maximum=2, value=1, step=0.1, label=_("Scale (Face image)", lang)),
        gr.Radio(choices=["SD", "SDXL"], label=_("Select model type", lang), value="SD"),
        gr.Dropdown(choices=stable_diffusion_models_list, label=_("Select StableDiffusion model", lang), value=None),
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1.0, maximum=30.0, value=6, step=0.1, label=_("Guidance Scale", lang)),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label=_("Width", lang)),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label=_("Height", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("IP-Adapter FaceID Settings", lang), open=False),
    outputs=[
        gr.Gallery(label=_("Generated images", lang), elem_id="gallery", columns=[2], rows=[2], object_fit="contain", height="auto"),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - StableDiffusion (IP-Adapter FaceID)", lang),
    description=_("This user interface allows you to generate images using IP-Adapter FaceID. "
                "Upload a face image, enter a prompt, select a Stable Diffusion model, and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

riffusion_text2image_interface = gr.Interface(
    fn=generate_riffusion_text2image,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.1, label=_("Guidance Scale", lang)),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label=_("Height", lang)),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label=_("Width", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Riffusion Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Generated image", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - Riffusion (Text-to-Image)", lang),
    description=_("Generate a spectrogram image from text using Riffusion.", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

riffusion_image2audio_interface = gr.Interface(
    fn=generate_riffusion_image2audio,
    inputs=[
        gr.Image(label=_("Input spectrogram image", lang), type="filepath"),
        gr.Radio(choices=["wav", "mp3", "ogg"], label=_("Select output format", lang), value="wav", interactive=True)
    ],
    outputs=[
        gr.Audio(label=_("Generated audio", lang), type="filepath"),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - Riffusion (Image-to-Audio)", lang),
    description=_("Convert a spectrogram image to audio using Riffusion.", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Convert", lang)
)

riffusion_audio2image_interface = gr.Interface(
    fn=generate_riffusion_audio2image,
    inputs=[
        gr.Audio(label=_("Input audio", lang), type="filepath"),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    outputs=[
        gr.Image(type="filepath", label=_("Generated spectrogram image", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - Riffusion (Audio-to-Image)", lang),
    description=_("Convert audio to a spectrogram image using Riffusion.", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Convert", lang)
)

riffusion_interface = gr.TabbedInterface(
    [riffusion_text2image_interface, riffusion_image2audio_interface, riffusion_audio2image_interface],
    tab_names=[_("Text-to-Image", lang), _("Image-to-Audio", lang), _("Audio-to-Image", lang)]
)

kandinsky_txt2img_interface = gr.Interface(
    fn=generate_image_kandinsky_txt2img,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Radio(choices=["2.1", "2.2", "3"], label=_("Kandinsky Version", lang), value="2.2"),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=0.1, maximum=20, value=4, step=0.1, label=_("CFG", lang)),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label=_("Height", lang)),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label=_("Width", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Kandinsky Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Generated image", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - Kandinsky (txt2img)", lang),
    description=_("This user interface allows you to generate images using Kandinsky models. "
                "You can select between versions 2.1, 2.2, and 3, and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

kandinsky_img2img_interface = gr.Interface(
    fn=generate_image_kandinsky_img2img,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Image(label=_("Initial image", lang), type="filepath"),
        gr.Radio(choices=["2.1", "2.2", "3"], label=_("Kandinsky Version", lang), value="2.2"),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=0.1, maximum=20, value=4, step=0.1, label=_("CFG", lang)),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.8, step=0.01, label=_("Strength (Initial image)", lang)),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label=_("Height", lang)),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label=_("Width", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Kandinsky Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Generated image", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - Kandinsky (img2img)", lang),
    description=_("This user interface allows you to generate images using Kandinsky models. "
                "You can select between versions 2.1, 2.2, and 3, and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

kandinsky_inpaint_interface = gr.Interface(
    fn=generate_image_kandinsky_inpaint,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Image(label=_("Initial image", lang), type="filepath"),
        gr.ImageEditor(label=_("Mask image", lang), type="filepath"),
        gr.Radio(choices=["2.1", "2.2"], label=_("Kandinsky Version", lang), value="2.2"),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=0.1, maximum=20, value=4, step=0.1, label=_("CFG", lang)),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.8, step=0.01, label=_("Strength", lang)),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label=_("Height", lang)),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label=_("Width", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Kandinsky Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Generated image", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - Kandinsky (inpaint)", lang),
    description=_("This user interface allows you to perform inpainting using Kandinsky models. "
                "You can select between versions 2.1 and 2.2, and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

kandinsky_interface = gr.TabbedInterface(
    [kandinsky_txt2img_interface, kandinsky_img2img_interface, kandinsky_inpaint_interface],
    tab_names=[_("txt2img", lang), _("img2img", lang), _("inpaint", lang)]
)

flux_txt2img_interface = gr.Interface(
    fn=generate_image_flux_txt2img,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Radio(choices=["FLUX.1-schnell", "FLUX.1-dev"], label=_("Select model type", lang), value="FLUX.1-schnell"),
        gr.Dropdown(choices=quantized_flux_models_list, label=_("Select safetensors Flux model (GGUF if enabled quantize)", lang), value=None),
        gr.Checkbox(label=_("Enable Quantize", lang), value=False),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Dropdown(choices=flux_vae_models_list, label=_("Select VAE model (optional)", lang), value=None),
        gr.Dropdown(choices=flux_lora_models_list, label=_("Select LORA models (optional)", lang), value=None, multiselect=True),
        gr.Textbox(label=_("LoRA Scales", lang)),
        gr.Slider(minimum=0.0, maximum=10.0, value=0.0, step=0.1, label=_("Guidance Scale", lang)),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label=_("Height", lang)),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label=_("Width", lang)),
        gr.Slider(minimum=1, maximum=100, value=10, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1, maximum=1024, value=256, step=1, label=_("Max Sequence Length", lang)),
        gr.Checkbox(label=_("Enable TAESD", lang), value=False),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Flux txt2img Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Generated image", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - Flux (txt2img)", lang),
    description=_("This user interface allows you to generate images using Flux models. "
                "You can select between Schnell and Dev models, and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

flux_img2img_interface = gr.Interface(
    fn=generate_image_flux_img2img,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Image(label=_("Initial image", lang), type="filepath"),
        gr.Dropdown(choices=["FLUX.1-schnell", "FLUX.1-dev"], label=_("Select Flux model", lang),
                    value="FLUX.1-schnell"),
        gr.Dropdown(choices=quantized_flux_models_list,
                    label=_("Select quantized Flux model (optional if enabled quantize)", lang), value=None),
        gr.Checkbox(label=_("Enable Quantize", lang), value=False),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Dropdown(choices=vae_models_list, label=_("Select VAE model (optional)", lang), value=None),
        gr.Dropdown(choices=flux_lora_models_list, label=_("Select LORA models (optional)", lang), value=None,
                    multiselect=True),
        gr.Textbox(label=_("LoRA Scales", lang)),
        gr.Slider(minimum=1, maximum=100, value=4, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.95, step=0.01, label=_("Strength", lang)),
        gr.Slider(minimum=0.0, maximum=10.0, value=0.0, step=0.1, label=_("Guidance Scale", lang)),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label=_("Height", lang)),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label=_("Width", lang)),
        gr.Slider(minimum=1, maximum=1024, value=256, step=1, label=_("Max Sequence Length", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Flux img2img Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Generated image", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - Flux (img2img)", lang),
    description=_("This user interface allows you to generate images from existing images using Flux models. "
                "Upload an initial image, enter a prompt, and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

flux_inpaint_interface = gr.Interface(
    fn=generate_image_flux_inpaint,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Image(label=_("Initial image", lang), type="filepath"),
        gr.ImageEditor(label=_("Mask image", lang), type="filepath"),
        gr.Dropdown(choices=["FLUX.1-schnell", "FLUX.1-dev"], label=_("Select Flux model", lang),
                    value="FLUX.1-schnell"),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=100, value=4, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.95, step=0.01, label=_("Strength", lang)),
        gr.Slider(minimum=0.0, maximum=10.0, value=0.0, step=0.1, label=_("Guidance Scale", lang)),
        gr.Slider(minimum=1, maximum=1024, value=256, step=1, label=_("Max Sequence Length", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Flux inpaint Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Generated image", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - Flux (inpaint)", lang),
    description=_("This user interface allows you to perform inpainting using Flux models. "
                "Upload an initial image, draw a mask, enter a prompt, and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

flux_controlnet_interface = gr.Interface(
    fn=generate_image_flux_controlnet,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Image(label=_("Control image", lang), type="filepath"),
        gr.Dropdown(choices=["FLUX.1-schnell", "FLUX.1-dev"], label=_("Select Flux base model", lang),
                    value="FLUX.1-dev"),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Slider(minimum=0, maximum=6, value=0, step=1, label=_("Control Mode", lang)),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label=_("ControlNet Conditioning Scale", lang)),
        gr.Slider(minimum=1, maximum=100, value=24, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=0.0, maximum=10.0, value=3.5, step=0.1, label=_("Guidance Scale", lang)),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label=_("Height", lang)),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label=_("Width", lang)),
        gr.Slider(minimum=1, maximum=1024, value=256, step=1, label=_("Max Sequence Length", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Flux ControlNet Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Generated image", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - Flux (ControlNet)", lang),
    description=_("This user interface allows you to generate images using Flux ControlNet. "
                "Upload a control image, enter a prompt, and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

flux_interface = gr.TabbedInterface(
    [flux_txt2img_interface, flux_img2img_interface, flux_inpaint_interface, flux_controlnet_interface],
    tab_names=[_("txt2img", lang), _("img2img", lang), _("inpaint", lang), _("controlnet", lang)]
)

hunyuandit_txt2img_interface = gr.Interface(
    fn=generate_image_hunyuandit_txt2img,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=0.1, maximum=30.0, value=7.5, step=0.1, label=_("Guidance Scale", lang)),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label=_("Height", lang)),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label=_("Width", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("HunyuanDiT Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Generated image", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - HunyuanDiT (txt2img)", lang),
    description=_("This user interface allows you to generate images using HunyuanDiT model. "
                "Enter a prompt (in English or Chinese) and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

hunyuandit_controlnet_interface = gr.Interface(
    fn=generate_image_hunyuandit_controlnet,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Image(label=_("Input image", lang), type="filepath"),
        gr.Dropdown(choices=["Depth", "Canny", "Pose"], label=_("Select ControlNet model", lang), value="Depth"),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1.0, maximum=20.0, value=6.0, step=0.1, label=_("Guidance Scale", lang)),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label=_("Height", lang)),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label=_("Width", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("HunyuanDiT Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Generated image", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - HunyuanDiT (ControlNet)", lang),
    description=_("This user interface allows you to generate images using HunyuanDiT ControlNet models. "
                "Enter a prompt, upload an input image, select a ControlNet model, and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

hunyuandit_interface = gr.TabbedInterface(
    [hunyuandit_txt2img_interface, hunyuandit_controlnet_interface],
    tab_names=[_("txt2img", lang), _("controlnet", lang)]
)

lumina_interface = gr.Interface(
    fn=generate_image_lumina,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Textbox(label=_("Seed (optional)", lang), value="")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=0.1, maximum=30.0, value=4, step=0.1, label=_("Guidance Scale", lang)),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label=_("Height", lang)),
        gr.Slider(minimum=256, maximum=2048, value=768, step=64, label=_("Width", lang)),
        gr.Slider(minimum=1, maximum=1024, value=256, step=1, label=_("Max Sequence Length", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Lumina-T2X Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Generated image", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - Lumina-T2X", lang),
    description=_("This user interface allows you to generate images using the Lumina-T2X model. "
                "Enter a prompt and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

kolors_txt2img_interface = gr.Interface(
    fn=generate_image_kolors_txt2img,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Textbox(label=_("Seed (optional)", lang), value="")
    ],
    additional_inputs=[
        gr.Dropdown(choices=kolors_lora_models_list, label=_("Select LORA models (optional)", lang), value=None,
                    multiselect=True),
        gr.Textbox(label=_("LoRA Scales", lang)),
        gr.Slider(minimum=1.0, maximum=20.0, value=6.5, step=0.1, label=_("Guidance Scale", lang)),
        gr.Slider(minimum=1, maximum=100, value=25, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1, maximum=1024, value=256, step=1, label=_("Max Sequence Length", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Kolors Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Generated image", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - Kolors (txt2img)", lang),
    description=_("This user interface allows you to generate images using the Kolors model. "
                "Enter a prompt and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

kolors_img2img_interface = gr.Interface(
    fn=generate_image_kolors_img2img,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Image(label=_("Initial image", lang), type="filepath"),
        gr.Textbox(label=_("Seed (optional)", lang), value="")
    ],
    additional_inputs=[
        gr.Slider(minimum=1.0, maximum=20.0, value=6.5, step=0.1, label=_("Guidance Scale", lang)),
        gr.Slider(minimum=1, maximum=100, value=25, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1, maximum=1024, value=256, step=1, label=_("Max Sequence Length", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Kolors Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Generated image", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - Kolors (img2img)", lang),
    description=_("This user interface allows you to generate images using the Kolors model. "
                "Enter a prompt and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

kolors_ip_adapter_interface = gr.Interface(
    fn=generate_image_kolors_ip_adapter_plus,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Image(label=_("IP-Adapter Image", lang), type="filepath"),
        gr.Textbox(label=_("Seed (optional)", lang), value="")
    ],
    additional_inputs=[
        gr.Slider(minimum=1.0, maximum=20.0, value=6.5, step=0.1, label=_("Guidance Scale", lang)),
        gr.Slider(minimum=1, maximum=100, value=25, step=1, label=_("Steps", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Kolors Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Generated image", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - Kolors (ip-adapter-plus)", lang),
    description=_("This user interface allows you to generate images using the Kolors model. "
                "Enter a prompt and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

kolors_interface = gr.TabbedInterface(
    [kolors_txt2img_interface, kolors_img2img_interface, kolors_ip_adapter_interface],
    tab_names=[_("txt2img", lang), _("img2img", lang), _("ip-adapter-plus", lang)]
)

auraflow_interface = gr.Interface(
    fn=generate_image_auraflow,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Textbox(label=_("Seed (optional)", lang), value="")
    ],
    additional_inputs=[
        gr.Dropdown(choices=auraflow_lora_models_list, label=_("Select LORA models (optional)", lang), value=None,
                    multiselect=True),
        gr.Textbox(label=_("LoRA Scales", lang)),
        gr.Slider(minimum=1, maximum=100, value=25, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.1, label=_("Guidance Scale", lang)),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label=_("Height", lang)),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label=_("Width", lang)),
        gr.Slider(minimum=1, maximum=1024, value=256, step=1, label=_("Max Sequence Length", lang)),
        gr.Checkbox(label=_("Enable AuraSR", lang), value=False),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("AuraFlow Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Generated image", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - AuraFlow", lang),
    description=_("This user interface allows you to generate images using the AuraFlow model. "
                "Enter a prompt and customize the generation settings. "
                "You can also enable AuraSR for 4x upscaling of the generated image. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

wurstchen_interface = gr.Interface(
    fn=generate_image_wurstchen,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Slider(minimum=256, maximum=2048, value=1536, step=64, label=_("Width", lang)),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label=_("Height", lang)),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label=_("Prior Steps", lang)),
        gr.Slider(minimum=0.1, maximum=30.0, value=4.0, step=0.1, label=_("Prior Guidance Scale", lang)),
        gr.Slider(minimum=1, maximum=100, value=20, step=1, label=_("Decoder Steps", lang)),
        gr.Slider(minimum=0.0, maximum=30.0, value=0.0, step=0.1, label=_("Decoder Guidance Scale", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Würstchen Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Generated image", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - Würstchen", lang),
    description=_("This user interface allows you to generate images using the Würstchen model. "
                "Enter a prompt and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

deepfloyd_if_txt2img_interface = gr.Interface(
    fn=generate_image_deepfloyd_txt2img,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=0.1, maximum=30.0, value=6, step=0.1, label=_("Guidance Scale", lang)),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label=_("Width", lang)),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label=_("Height", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("DeepFloyd IF Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Generated image (Stage I)", lang)),
        gr.Image(type="filepath", label=_("Generated image (Stage II)", lang)),
        gr.Image(type="filepath", label=_("Generated image (Stage III)", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - DeepFloyd IF (txt2img)", lang),
    description=_("This user interface allows you to generate images using the DeepFloyd IF model. "
                "Enter a prompt and customize the generation settings. "
                "The process includes three stages of generation, each producing an image of increasing quality. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

deepfloyd_if_img2img_interface = gr.Interface(
    fn=generate_image_deepfloyd_img2img,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Image(label=_("Initial image", lang), type="filepath"),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=0.1, maximum=30.0, value=6, step=0.1, label=_("Guidance Scale", lang)),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label=_("Width", lang)),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label=_("Height", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("DeepFloyd IF Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Generated image (Stage I)", lang)),
        gr.Image(type="filepath", label=_("Generated image (Stage II)", lang)),
        gr.Image(type="filepath", label=_("Generated image (Stage III)", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - DeepFloyd IF (img2img)", lang),
    description=_("This interface allows you to generate images using DeepFloyd IF's image-to-image pipeline. "
                "Enter a prompt, upload an initial image, and customize the generation settings. "
                "The process includes three stages of generation, each producing an image of increasing quality. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

deepfloyd_if_inpaint_interface = gr.Interface(
    fn=generate_image_deepfloyd_inpaint,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Image(label=_("Initial image", lang), type="filepath"),
        gr.ImageEditor(label=_("Mask image", lang), type="filepath"),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=0.1, maximum=30.0, value=6, step=0.1, label=_("Guidance Scale", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("DeepFloyd IF Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Generated image (Stage I)", lang)),
        gr.Image(type="filepath", label=_("Generated image (Stage II)", lang)),
        gr.Image(type="filepath", label=_("Generated image (Stage III)", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - DeepFloyd IF (inpaint)", lang),
    description=_("This interface allows you to perform inpainting using DeepFloyd IF. "
                "Enter a prompt, upload an initial image and a mask image, and customize the generation settings. "
                "The process includes three stages of generation, each producing an image of increasing quality. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

deepfloyd_if_interface = gr.TabbedInterface(
    [deepfloyd_if_txt2img_interface, deepfloyd_if_img2img_interface, deepfloyd_if_inpaint_interface],
    tab_names=[_("txt2img", lang), _("img2img", lang), _("inpaint", lang)]
)

pixart_interface = gr.Interface(
    fn=generate_image_pixart,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Radio(choices=["Alpha-512", "Alpha-1024", "Sigma-512", "Sigma-1024"], label=_("PixArt Version", lang), value="Alpha-512"),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=0.1, maximum=30.0, value=7.5, step=0.1, label=_("Guidance Scale", lang)),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label=_("Height", lang)),
        gr.Slider(minimum=256, maximum=2048, value=512, step=64, label=_("Width", lang)),
        gr.Slider(minimum=1, maximum=1024, value=256, step=1, label=_("Max Sequence Length", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("PixArt Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Generated image", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - PixArt", lang),
    description=_("This user interface allows you to generate images using PixArt models. "
                "You can select between Alpha and Sigma versions, and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

cogview3plus_interface = gr.Interface(
    fn=generate_image_cogview3plus,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label=_("Height", lang)),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label=_("Width", lang)),
        gr.Slider(minimum=1, maximum=100, value=40, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1.0, maximum=30.0, value=8.0, step=0.1, label=_("CFG", lang)),
        gr.Slider(minimum=64, maximum=2048, value=256, label=_("Max Length", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Cogview3-plus Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Generated image", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - Cogview3-plus", lang),
    description=_("This user interface allows you to enter any text and generate images using Cogview3-plus. "
                "You can customize the generation settings from the sliders. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

playgroundv2_interface = gr.Interface(
    fn=generate_image_playgroundv2,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Textbox(label=_("Seed (optional)", lang), value="")
    ],
    additional_inputs=[
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label=_("Height", lang)),
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label=_("Width", lang)),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=0.1, maximum=30.0, value=3.0, step=0.1, label=_("Guidance Scale", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("PlaygroundV2.5 Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Generated image", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - PlaygroundV2.5", lang),
    description=_("This user interface allows you to generate images using PlaygroundV2.5. "
                "Enter a prompt and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

wav2lip_interface = gr.Interface(
    fn=generate_wav2lip,
    inputs=[
        gr.Image(label=_("Input image", lang), type="filepath"),
        gr.Audio(label=_("Input audio", lang), type="filepath")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=60, value=30, step=1, label=_("FPS", lang)),
        gr.Textbox(label=_("Pads", lang), value="0 10 0 0"),
        gr.Slider(minimum=1, maximum=64, value=16, step=1, label=_("Face Detection Batch Size", lang)),
        gr.Slider(minimum=1, maximum=512, value=128, step=1, label=_("Wav2Lip Batch Size", lang)),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label=_("Resize Factor", lang)),
        gr.Textbox(label=_("Crop", lang), value="0 -1 0 -1"),
        gr.Checkbox(label=_("Enable no smooth", lang), value=False)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Wav2Lip Settings", lang), open=False),
    outputs=[
        gr.Video(label=_("Generated lip-sync", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - Wav2Lip", lang),
    description=_("This user interface allows you to generate talking head videos by combining an image and an audio file using Wav2Lip. "
                "Upload an image and an audio file, and click Generate to create the talking head video. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Animate", lang)
)

liveportrait_interface = gr.Interface(
    fn=generate_liveportrait,
    inputs=[
        gr.Image(label=_("Source image", lang), type="filepath"),
        gr.Video(label=_("Driving video", lang)),
        gr.Radio(choices=["mp4", "mov", "avi"], label=_("Select output format", lang), value="mp4", interactive=True)
    ],
    outputs=[
        gr.Video(label=_("Generated video", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - LivePortrait", lang),
    description=_("This user interface allows you to animate a source image based on the movements in a driving video using LivePortrait. "
                "Upload a source image and a driving video, then click Generate to create the animated video. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Animate", lang)
)

modelscope_interface = gr.Interface(
    fn=generate_video_modelscope,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.1, label=_("Guidance Scale", lang)),
        gr.Slider(minimum=256, maximum=1024, value=320, step=64, label=_("Height", lang)),
        gr.Slider(minimum=256, maximum=1024, value=576, step=64, label=_("Width", lang)),
        gr.Slider(minimum=16, maximum=128, value=64, step=1, label=_("Number of Frames", lang)),
        gr.Radio(choices=["mp4", "mov", "avi"], label=_("Select output format", lang), value="mp4", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("ModelScope Settings", lang), open=False),
    outputs=[
        gr.Video(label=_("Generated video", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - ModelScope", lang),
    description=_("This user interface allows you to generate videos using ModelScope. "
                "Enter a prompt and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

zeroscope2_interface = gr.Interface(
    fn=generate_video_zeroscope2,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Video(label=_("Video to enhance (optional)", lang), interactive=True),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.1, label=_("Strength (Video to enhance)", lang)),
        gr.Slider(minimum=1, maximum=100, value=40, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=256, maximum=1280, value=576, step=64, label=_("Width", lang)),
        gr.Slider(minimum=256, maximum=1280, value=320, step=64, label=_("Height", lang)),
        gr.Slider(minimum=1, maximum=100, value=36, step=1, label=_("Frames", lang)),
        gr.Checkbox(label=_("Enable Video Enhancement", lang), value=False),
        gr.Radio(choices=["mp4", "mov", "avi"], label=_("Select output format", lang), value="mp4", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("ZeroScope 2 Settings", lang), open=False),
    outputs=[
        gr.Video(label=_("Generated video", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - ZeroScope 2", lang),
    description=_("This user interface allows you to generate and enhance videos using ZeroScope 2 models. "
                "You can enter a text prompt, upload an optional video for enhancement, and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

cogvideox_text2video_interface = gr.Interface(
    fn=generate_video_cogvideox_text2video,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Radio(choices=["CogVideoX-2B", "CogVideoX-5B"], label=_("Select CogVideoX model version", lang), value="CogVideoX-2B"),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1.0, maximum=20.0, value=6.0, step=0.1, label=_("Guidance Scale", lang)),
        gr.Slider(minimum=256, maximum=1024, value=480, step=64, label=_("Height", lang)),
        gr.Slider(minimum=256, maximum=1024, value=720, step=64, label=_("Width", lang)),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label=_("Number of Frames", lang)),
        gr.Slider(minimum=1, maximum=60, value=10, step=1, label=_("FPS", lang)),
        gr.Radio(choices=["mp4", "mov", "avi"], label=_("Select output format", lang), value="mp4", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("CogVideoX T2V Settings", lang), open=False),
    outputs=[
        gr.Video(label=_("Generated video", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - CogVideoX (Text2Video)", lang),
    description=_("This user interface allows you to generate videos from text using CogVideoX (Text2Video). "
                "Enter a prompt and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

cogvideox_image2video_interface = gr.Interface(
    fn=generate_video_cogvideox_image2video,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Image(label=_("Initial image", lang), type="filepath"),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Slider(minimum=1.0, maximum=20.0, value=6.0, step=0.1, label=_("Guidance Scale", lang)),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1, maximum=60, value=8, step=1, label=_("FPS", lang)),
        gr.Radio(choices=["mp4", "mov", "avi"], label=_("Select output format", lang), value="mp4", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("CogVideoX I2V Settings", lang), open=False),
    outputs=[
        gr.Video(label=_("Generated video", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - CogVideoX (Image2Video)", lang),
    description=_("This user interface allows you to generate videos from images using CogVideoX (Image2Video). "
                "Upload an image, enter a prompt, and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

cogvideox_video2video_interface = gr.Interface(
    fn=generate_video_cogvideox_video2video,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Video(label=_("Input video", lang)),
        gr.Radio(choices=["CogVideoX-2B", "CogVideoX-5B"], label=_("Select CogVideoX model version", lang), value="CogVideoX-5B"),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Slider(minimum=0.0, maximum=1.0, value=0.8, step=0.1, label=_("Strength", lang)),
        gr.Slider(minimum=1.0, maximum=20.0, value=6.0, step=0.1, label=_("Guidance Scale", lang)),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1, maximum=60, value=8, step=1, label=_("FPS", lang)),
        gr.Radio(choices=["mp4", "mov", "avi"], label=_("Select output format", lang), value="mp4", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("CogVideoX V2V Settings", lang), open=False),
    outputs=[
        gr.Video(label=_("Generated video", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - CogVideoX (Video2Video)", lang),
    description=_("This user interface allows you to generate videos from videos using CogVideoX (Video2Video). "
                "Upload a video, enter a prompt, and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

cogvideox_interface = gr.TabbedInterface(
    [cogvideox_text2video_interface, cogvideox_image2video_interface, cogvideox_video2video_interface],
    tab_names=[_("Text2Video", lang), _("Image2Video", lang), _("Video2Video", lang)]
)

latte_interface = gr.Interface(
    fn=generate_video_latte,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1.0, maximum=20.0, value=6.0, step=0.1, label=_("Guidance Scale", lang)),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label=_("Height", lang)),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label=_("Width", lang)),
        gr.Slider(minimum=1, maximum=100, value=16, step=1, label=_("Video Length", lang))
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Latte Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Generated GIF", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - Latte", lang),
    description=_("This user interface allows you to generate GIFs using Latte. "
                "Enter a prompt and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

stablefast3d_interface = gr.Interface(
    fn=generate_3d_stablefast3d,
    inputs=[
        gr.Image(label=_("Input image", lang), type="filepath")
    ],
    additional_inputs=[
        gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label=_("Texture Resolution", lang)),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.85, step=0.05, label=_("Foreground Ratio", lang)),
        gr.Radio(choices=["none", "triangle", "quad"], label=_("Remesh Option", lang), value="none")
    ],
    additional_inputs_accordion=gr.Accordion(label=_("StableFast3D Settings", lang), open=False),
    outputs=[
        gr.Model3D(label=_("Generated 3D object", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - StableFast3D", lang),
    description=_("This user interface allows you to generate 3D objects from images using StableFast3D. "
                "Upload an image and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

shap_e_interface = gr.Interface(
    fn=generate_3d_shap_e,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Image(label=_("Initial image (optional)", lang), type="filepath", interactive=True),
        gr.Textbox(label=_("Seed (optional)", lang), value="")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1.0, maximum=30.0, value=10.0, step=0.1, label=_("CFG", lang)),
        gr.Slider(minimum=64, maximum=512, value=256, step=64, label=_("Frame size", lang))
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Shap-E Settings", lang), open=False),
    outputs=[
        gr.Model3D(label=_("Generated 3D object", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - Shap-E", lang),
    description=_("This user interface allows you to generate 3D objects using Shap-E. "
                "You can enter a text prompt or upload an initial image, and customize the generation settings. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

zero123plus_interface = gr.Interface(
    fn=generate_3d_zero123plus,
    inputs=[
        gr.Image(label=_("Input image", lang), type="filepath"),
        gr.Slider(minimum=1, maximum=100, value=75, step=1, label=_("Inference steps", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True),
    ],
    outputs=[
        gr.Image(type="filepath", label=_("Generated image", lang)),
        gr.Textbox(label=_("Message", lang), type="text"),
    ],
    title=_("NeuroSandboxWebUI - Zero123Plus", lang),
    description=_("This user interface allows you to generate 3D-like images using Zero123Plus. "
                "Upload an input image and customize the number of inference steps. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

stableaudio_interface = gr.Interface(
    fn=generate_stableaudio,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang)),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=1000, value=200, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=0.1, maximum=12, value=4, step=0.1, label=_("CFG", lang)),
        gr.Slider(minimum=1, maximum=60, value=10, step=1, label=_("Audio Length (seconds)", lang)),
        gr.Slider(minimum=1, maximum=60, value=0, step=1, label=_("Audio Start (seconds)", lang)),
        gr.Slider(minimum=1, maximum=10, value=3, step=1, label=_("Number of Waveforms", lang)),
        gr.Radio(choices=["wav", "mp3", "ogg"], label=_("Select output format", lang), value="wav", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("StableAudio Settings", lang), open=False),
    outputs=[
        gr.Audio(label=_("Generated audio", lang), type="filepath"),
        gr.Image(label=_("Mel-Spectrogram", lang), type="filepath"),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - StableAudio", lang),
    description=_("This user interface allows you to enter any text and generate audio using StableAudio. "
                "You can customize the generation settings from the sliders. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

audiocraft_interface = gr.Interface(
    fn=generate_audio_audiocraft,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Audio(type="filepath", label=_("Melody audio (optional)", lang), interactive=True),
        gr.Dropdown(choices=audiocraft_models_list, label=_("Select AudioCraft model", lang), value=None),
        gr.Radio(choices=["musicgen", "audiogen", "magnet"], label=_("Select model type", lang), value="musicgen"),
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=120, value=10, step=1, label=_("Duration (seconds)", lang)),
        gr.Slider(minimum=1, maximum=1000, value=250, step=1, label=_("Top K", lang)),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label=_("Top P", lang)),
        gr.Slider(minimum=0.0, maximum=1.9, value=1.0, step=0.1, label=_("Temperature", lang)),
        gr.Slider(minimum=1.0, maximum=10.0, value=3.0, step=0.1, label=_("CFG", lang)),
        gr.Slider(minimum=1.0, maximum=10.0, value=3.0, step=0.1, label=_("Min CFG coef (Magnet model only)", lang)),
        gr.Slider(minimum=1.0, maximum=10.0, value=1.0, step=0.1, label=_("Max CFG coef (Magnet model only)", lang)),
        gr.Checkbox(label=_("Enable Two-step CFG (Musicgen and Audiogen models only)", lang), value=False),
        gr.Checkbox(label=_("Enable Sampling (Musicgen and Audiogen models only)", lang), value=True),
        gr.Checkbox(label=_("Enable Multiband Diffusion (Musicgen model only)", lang), value=False),
        gr.Radio(choices=["wav", "mp3", "ogg"], label=_("Select output format (Works only without Multiband Diffusion)", lang),
                 value="wav", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("AudioCraft Settings", lang), open=False),
    outputs=[
        gr.Audio(label=_("Generated audio", lang), type="filepath"),
        gr.Image(label=_("Mel-Spectrogram", lang), type="filepath"),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - AudioCraft", lang),
    description=_("This user interface allows you to enter any text and generate audio using AudioCraft. "
                "You can select the model and customize the generation settings from the sliders. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

audioldm2_interface = gr.Interface(
    fn=generate_audio_audioldm2,
    inputs=[
        gr.Textbox(label=_("Enter your prompt", lang)),
        gr.Textbox(label=_("Enter your negative prompt", lang), value=""),
        gr.Dropdown(choices=["cvssp/audioldm2", "cvssp/audioldm2-music"], label=_("Select AudioLDM 2 model", lang), value="cvssp/audioldm2"),
        gr.Textbox(label=_("Seed (optional)", lang), value=""),
        gr.Button(value=_("Stop generation", lang), interactive=True, variant="stop")
    ],
    additional_inputs=[
        gr.Slider(minimum=1, maximum=1000, value=200, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=1, maximum=60, value=10, step=1, label=_("Length (seconds)", lang)),
        gr.Slider(minimum=1, maximum=10, value=3, step=1, label=_("Waveforms number", lang)),
        gr.Radio(choices=["wav", "mp3", "ogg"], label=_("Select output format", lang), value="wav", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("AudioLDM 2 Settings", lang), open=False),
    outputs=[
        gr.Audio(label=_("Generated audio", lang), type="filepath"),
        gr.Image(label=_("Mel-Spectrogram", lang), type="filepath"),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - AudioLDM 2", lang),
    description=_("This user interface allows you to enter any text and generate audio using AudioLDM 2. "
                "You can select the model and customize the generation settings from the sliders. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

bark_interface = gr.Interface(
    fn=generate_bark_audio,
    inputs=[
        gr.Textbox(label=_("Enter text for the request", lang)),
        gr.Dropdown(choices=[None, "v2/en_speaker_1", "v2/ru_speaker_1", "v2/de_speaker_1", "v2/fr_speaker_1", "v2/es_speaker_1", "v2/hi_speaker_1", "v2/it_speaker_1", "v2/ja_speaker_1", "v2/ko_speaker_1", "v2/pt_speaker_1", "v2/zh_speaker_1", "v2/tr_speaker_1", "v2/pl_speaker_1"], label=_("Select voice preset", lang), value=None)
    ],
    additional_inputs=[
        gr.Slider(minimum=100, maximum=1000, value=200, step=1, label=_("Max length", lang)),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.4, step=0.1, label=_("Fine temperature", lang)),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label=_("Coarse temperature", lang)),
        gr.Radio(choices=["wav", "mp3", "ogg"], label=_("Select output format", lang), value="wav", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Bark Settings", lang), open=False),
    outputs=[
        gr.Audio(label=_("Generated audio", lang), type="filepath"),
        gr.Image(label=_("Mel-Spectrogram", lang), type="filepath"),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - SunoBark", lang),
    description=_("This user interface allows you to enter text and generate audio using SunoBark. "
                "You can select the voice preset and customize the max length. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Generate", lang)
)

rvc_interface = gr.Interface(
    fn=process_rvc,
    inputs=[
        gr.Audio(label=_("Input audio", lang), type="filepath"),
        gr.Dropdown(choices=rvc_models_list, label=_("Select RVC model", lang), value=None)
    ],
    additional_inputs=[
        gr.Radio(choices=['harvest', "crepe", "rmvpe", 'pm'], label=_("RVC Method", lang), value="harvest", interactive=True),
        gr.Number(label=_("Up-key", lang), value=0),
        gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label=_("Index rate", lang)),
        gr.Slider(minimum=0, maximum=12, value=3, step=1, label=_("Filter radius", lang)),
        gr.Slider(minimum=0, maximum=1, value=0, step=0.01, label=_("Resample-Sr", lang)),
        gr.Slider(minimum=0, maximum=1, value=1, step=0.01, label=_("RMS Mixrate", lang)),
        gr.Slider(minimum=0, maximum=1, value=0.33, step=0.01, label=_("Protection", lang)),
        gr.Radio(choices=["wav", "mp3", "ogg"], label=_("Select output format", lang), value="wav", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("RVC Settings", lang), open=False),
    outputs=[
        gr.Audio(label=_("Processed audio", lang), type="filepath"),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - RVC", lang),
    description=_("This user interface allows you to process audio using RVC (Retrieval-based Voice Conversion). "
                "Upload an audio file, select an RVC model, and choose the output format. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Conversion", lang)
)

uvr_interface = gr.Interface(
    fn=separate_audio_uvr,
    inputs=[
        gr.Audio(type="filepath", label=_("Audio file to separate", lang)),
        gr.Radio(choices=["wav", "mp3", "ogg"], label=_("Select output format", lang), value="wav", interactive=True),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.1, label=_("Normalization Threshold", lang)),
        gr.Slider(minimum=16000, maximum=44100, value=44100, step=100, label=_("Sample Rate", lang)),
    ],
    outputs=[
        gr.Audio(label=_("Vocals", lang), type="filepath"),
        gr.Audio(label=_("Instrumental", lang), type="filepath"),
        gr.Textbox(label=_("Message", lang), type="text"),
    ],
    title=_("NeuroSandboxWebUI - UVR", lang),
    description=_("This user interface allows you to upload an audio file and separate it into vocals and instrumental using Ultimate Vocal Remover (UVR). "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Separate", lang)
)

demucs_interface = gr.Interface(
    fn=demucs_separate,
    inputs=[
        gr.Audio(type="filepath", label=_("Audio file to separate", lang)),
        gr.Radio(choices=["wav", "mp3", "ogg"], label=_("Select output format", lang), value="wav", interactive=True),
    ],
    outputs=[
        gr.Audio(label=_("Vocal", lang), type="filepath"),
        gr.Audio(label=_("Instrumental", lang), type="filepath"),
        gr.Textbox(label=_("Message", lang), type="text"),
    ],
    title=_("NeuroSandboxWebUI - Demucs", lang),
    description=_("This user interface allows you to upload an audio file and separate it into vocal and instrumental using Demucs. "
                "Try it and see what happens!", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Separate", lang)
)

image_extras_interface = gr.Interface(
    fn=generate_image_extras,
    inputs=[
        gr.Image(label=_("Image to modify", lang), type="filepath"),
        gr.Checkbox(label=_("Remove BackGround", lang), value=False),
        gr.Checkbox(label=_("Enable FaceRestore", lang), value=False),
        gr.Slider(minimum=0.01, maximum=1, value=0.5, step=0.01, label=_("Fidelity weight (For FaceRestore)", lang)),
        gr.Slider(minimum=0.1, maximum=2, value=2, step=0.1, label=_("Upscale (For FaceRestore)", lang)),
        gr.Checkbox(label=_("Enable PixelOE", lang), value=False),
        gr.Radio(choices=["center", "contrast", "k-centroid", "bicubic", "nearest"], label=_("PixelOE Mode", lang), value="contrast"),
        gr.Slider(minimum=32, maximum=1024, value=256, step=32, label=_("Target Size (For PixelOE)", lang)),
        gr.Slider(minimum=1, maximum=48, value=8, step=1, label=_("Patch Size (For PixelOE)", lang)),
        gr.Slider(minimum=1, maximum=10, value=1, step=1, label=_("Thickness (For PixelOE)", lang)),
        gr.Slider(minimum=1, maximum=10, value=1, step=1, label=_("contrast (For PixelOE)", lang)),
        gr.Slider(minimum=1, maximum=10, value=1, step=1, label=_("saturation (For PixelOE)", lang)),
        gr.Checkbox(label=_("Enable DDColor", lang), value=False),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label=_("Input Size (For DDColor)", lang)),
        gr.Checkbox(label=_("Enable DownScale", lang), value=False),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.1, label=_("DownScale Factor", lang)),
        gr.Checkbox(label=_("Enable Format Changer", lang), value=False),
        gr.Radio(choices=["png", "jpeg"], label=_("New Image Format", lang), value="png"),
        gr.Checkbox(label=_("Enable Encryption", lang), value=False),
        gr.Checkbox(label=_("Enable Decryption", lang), value=False),
        gr.Textbox(label=_("Decryption Key", lang), value="")
    ],
    outputs=[
        gr.Image(label=_("Modified image", lang), type="filepath"),
        gr.Textbox(label=_("Message", lang), type="text"),
    ],
    title=_("NeuroSandboxWebUI - Extras (Image)", lang),
    description=_("This interface allows you to modify images", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Modify", lang)
)

video_extras_interface = gr.Interface(
    fn=generate_video_extras,
    inputs=[
        gr.Video(label=_("Video to modify", lang)),
        gr.Checkbox(label=_("Enable DownScale", lang), value=False),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.1, label=_("DownScale Factor", lang)),
        gr.Checkbox(label=_("Enable Format Changer", lang), value=False),
        gr.Radio(choices=["mp4", "mov", "avi"], label=_("New Video Format", lang), value="mp4")
    ],
    outputs=[
        gr.Video(label=_("Modified video", lang)),
        gr.Textbox(label=_("Message", lang), type="text"),
    ],
    title=_("NeuroSandboxWebUI - Extras (Video)", lang),
    description=_("This interface allows you to modify videos", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Modify", lang)
)

audio_extras_interface = gr.Interface(
    fn=generate_audio_extras,
    inputs=[
        gr.Audio(label=_("Audio to modify", lang), type="filepath"),
        gr.Checkbox(label=_("Enable Format Changer", lang), value=False),
        gr.Radio(choices=["wav", "mp3", "ogg"], label=_("New Audio Format", lang), value="wav"),
        gr.Checkbox(label=_("Enable AudioSR", lang), value=False),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label=_("Steps", lang)),
        gr.Slider(minimum=0.1, maximum=30.0, value=8, step=0.1, label=_("Guidance Scale", lang)),
        gr.Checkbox(label=_("Enable Downscale", lang), value=False),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.1, label=_("Downscale Factor", lang))
    ],
    outputs=[
        gr.Audio(label=_("Modified audio", lang), type="filepath"),
        gr.Textbox(label=_("Message", lang), type="text"),
    ],
    title=_("NeuroSandboxWebUI - Extras (Audio)", lang),
    description=_("This interface allows you to modify audio files", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Modify", lang)
)

realesrgan_upscale_interface = gr.Interface(
    fn=generate_upscale_realesrgan,
    inputs=[
        gr.Image(label=_("Image to upscale", lang), type="filepath"),
        gr.Video(label=_("Input video", lang)),
        gr.Radio(choices=["RealESRGAN_x2plus", "RealESRNet_x4plus", "RealESRGAN_x4plus", "realesr-general-x4v3", "RealESRGAN_x4plus_anime_6B"], label=_("Select model", lang), value="RealESRGAN_x4plus"),
        gr.Slider(minimum=0.1, maximum=4, value=2, step=0.1, label=_("Upscale factor", lang))
    ],
    additional_inputs=[
        gr.Checkbox(label=_("Enable Face Enhance", lang), value=False),
        gr.Slider(minimum=0, maximum=10, value=0, step=1, label=_("Tile", lang)),
        gr.Slider(minimum=0, maximum=100, value=10, step=1, label=_("Tile pad", lang)),
        gr.Slider(minimum=0, maximum=50, value=0, step=1, label=_("Pre pad", lang)),
        gr.Slider(minimum=0.01, maximum=1, value=0.5, step=0.01, label=_("Denoise strength", lang)),
        gr.Radio(choices=["png", "jpeg"], label=_("Select output format", lang), value="png", interactive=True)
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Face Enhance Settings", lang), open=False),
    outputs=[
        gr.Image(type="filepath", label=_("Upscaled image", lang)),
        gr.Video(label=_("Upscaled video", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - Upscale (Real-ESRGAN)", lang),
    description=_("This user interface allows you to upload an image and upscale it using Real-ESRGAN models", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Upscale", lang)
)

faceswap_interface = gr.Interface(
    fn=generate_faceswap,
    inputs=[
        gr.Image(label=_("Source Image", lang), type="filepath"),
        gr.Image(label=_("Target Image", lang), type="filepath"),
        gr.Video(label=_("Target Video", lang))
    ],
    additional_inputs=[
        gr.Checkbox(label=_("Enable many faces", lang), value=False),
        gr.Number(label=_("Reference face position", lang)),
        gr.Number(label=_("Reference frame number", lang)),
        gr.Checkbox(label=_("Enable FaceRestore", lang), value=False),
        gr.Slider(minimum=0.01, maximum=1, value=0.5, step=0.01, label=_("Fidelity weight", lang)),
        gr.Slider(minimum=0.1, maximum=2, value=2, step=0.1, label=_("Upscale", lang))
    ],
    additional_inputs_accordion=gr.Accordion(label=_("FaceSwap (Roop) Settings", lang), open=False),
    outputs=[
        gr.Image(label=_("Processed image", lang), type="filepath"),
        gr.Video(label=_("Processed video", lang)),
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - FaceSwap (Roop)", lang),
    description=_("This user interface allows you to perform face swapping on images or videos and optional face restoration.", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Swap", lang)
)

metadata_interface = gr.Interface(
    fn=display_metadata,
    inputs=[
        gr.File(label=_("Upload file", lang))
    ],
    outputs=[
        gr.Textbox(label=_("Metadata", lang), lines=10)
    ],
    title=_("NeuroSandboxWebUI - Metadata-Info", lang),
    description=_("This interface allows you to view generation metadata for image, video, and audio files.", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("View", lang)
)

extras_interface = gr.TabbedInterface(
    [image_extras_interface, video_extras_interface, audio_extras_interface, realesrgan_upscale_interface, faceswap_interface, metadata_interface],
    tab_names=[_("Image", lang), _("Video", lang), _("Audio", lang), _("Upscale (Real-ESRGAN)", lang), _("FaceSwap", lang), _("Metadata-Info", lang)]
)

wiki_interface = gr.Interface(
    fn=get_wiki_content,
    inputs=[
        gr.Textbox(label=_("Online Wiki", lang), value=(
            "https://github.com/Dartvauder/NeuroSandboxWebUI/wiki/EN‐Wiki" if lang == "EN" else
            "https://github.com/Dartvauder/NeuroSandboxWebUI/wiki/ZH‐Wiki" if lang == "ZH" else
            "https://github.com/Dartvauder/NeuroSandboxWebUI/wiki/RU‐Wiki"
        ), interactive=False),
        gr.Textbox(label=_("Local Wiki", lang), value=(
            "TechnicalFiles/Wikies/WikiEN.md" if lang == "EN" else
            "TechnicalFiles/Wikies/WikiZH.md" if lang == "ZH" else
            "TechnicalFiles/Wikies/WikiRU.md"
        ), interactive=False)
    ],
    outputs=gr.HTML(label=_("Wiki Content", lang)),
    title=_("NeuroSandboxWebUI - Wiki", lang),
    description=_("This interface displays the Wiki content from the specified URL or local file.", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Learn", lang)
)

gallery_interface = gr.Interface(
    fn=display_output_file,
    inputs=[
        gr.FileExplorer(label=_("Output Files", lang), root_dir="outputs", file_count="single", value=get_output_files())
    ],
    outputs=[
        gr.Textbox(label=_("Text", lang)),
        gr.Image(label=_("Image", lang), type="filepath"),
        gr.Video(label=_("Video", lang)),
        gr.Audio(label=_("Audio", lang), type="filepath"),
        gr.Model3D(label=_("3D Model", lang)),
    ],
    title=_("NeuroSandboxWebUI - Gallery", lang),
    description=_("This interface allows you to view files from the outputs directory", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("View", lang)
)

model_downloader_interface = gr.Interface(
    fn=download_model,
    inputs=[
        gr.Textbox(label=_("Model URL", lang), placeholder="repo-author/repo-name or https://huggingface.co/.../model.file"),
        gr.Radio(choices=DOWNLOADER_MODEL_TYPES, label=_("Model Type", lang))
    ],
    outputs=[
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - ModelDownloader", lang),
    description=_("This interface allows you to download various types of models", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Download", lang)
)

settings_interface = gr.Interface(
    fn=settings_interface,
    inputs=[
        gr.Radio(choices=["EN", "RU", "ZH"], label=_("Language", lang), value=settings['language']),
        gr.Radio(choices=["True", "False"], label=_("Share Mode", lang), value="False"),
        gr.Radio(choices=["True", "False"], label=_("Debug Mode", lang), value="False"),
        gr.Radio(choices=["True", "False"], label=_("Monitoring Mode", lang), value="False"),
        gr.Radio(choices=["True", "False"], label=_("Enable AutoLaunch", lang), value="False"),
        gr.Radio(choices=["True", "False"], label=_("Show API", lang), value="False"),
        gr.Radio(choices=["True", "False"], label=_("Open API", lang), value="False"),
        gr.Number(label=_("Queue max size", lang), value=settings['queue_max_size']),
        gr.Textbox(label=_("Queue status update rate", lang), value=settings['status_update_rate']),
        gr.Number(label=_("Max file size", lang), value=settings['max_file_size']),
        gr.Textbox(label=_("Gradio Auth", lang), value=settings['auth']),
        gr.Textbox(label=_("Server Name", lang), value=settings['server_name']),
        gr.Number(label=_("Server Port", lang), value=settings['server_port']),
        gr.Textbox(label=_("Hugging Face Token", lang), value=settings['hf_token']),
        gr.Textbox(label=_("Share server address", lang), value=settings['share_server_address'])
    ],
    additional_inputs=[
        gr.Radio(choices=["Base", "Default", "Glass", "Monochrome", "Soft"], label=_("Theme", lang), value=settings['theme']),
        gr.Checkbox(label=_("Enable Custom Theme", lang), value=settings['custom_theme']['enabled']),
        gr.Textbox(label=_("Primary Hue", lang), value=settings['custom_theme']['primary_hue']),
        gr.Textbox(label=_("Secondary Hue", lang), value=settings['custom_theme']['secondary_hue']),
        gr.Textbox(label=_("Neutral Hue", lang), value=settings['custom_theme']['neutral_hue']),
        gr.Radio(choices=["spacing_sm", "spacing_md", "spacing_lg"], label=_("Spacing Size", lang), value=settings['custom_theme'].get('spacing_size', 'spacing_md')),
        gr.Radio(choices=["radius_none", "radius_sm", "radius_md", "radius_lg"], label=_("Radius Size", lang), value=settings['custom_theme'].get('radius_size', 'radius_md')),
        gr.Radio(choices=["text_sm", "text_md", "text_lg"], label=_("Text Size", lang), value=settings['custom_theme'].get('text_size', 'text_md')),
        gr.Textbox(label=_("Font", lang), value=settings['custom_theme'].get('font', 'Arial')),
        gr.Textbox(label=_("Monospaced Font", lang), value=settings['custom_theme'].get('font_mono', 'Courier New'))
    ],
    additional_inputs_accordion=gr.Accordion(label=_("Theme builder", lang), open=False),
    outputs=[
        gr.Textbox(label=_("Message", lang), type="text")
    ],
    title=_("NeuroSandboxWebUI - Settings", lang),
    description=_("This user interface allows you to change settings of the application", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Update", lang)
)

system_interface = gr.Interface(
    fn=get_system_info,
    inputs=[],
    outputs=[
        gr.Textbox(label=_("GPU Total Memory", lang)),
        gr.Textbox(label=_("GPU Used Memory", lang)),
        gr.Textbox(label=_("GPU Free Memory", lang)),
        gr.Textbox(label=_("GPU Temperature", lang)),
        gr.Textbox(label=_("CPU Temperature", lang)),
        gr.Textbox(label=_("RAM Total", lang)),
        gr.Textbox(label=_("RAM Used", lang)),
        gr.Textbox(label=_("RAM Free", lang)),
        gr.Textbox(label=_("Disk Total Space", lang)),
        gr.Textbox(label=_("Disk Free Space", lang)),
        gr.Textbox(label=_("Application Folder Size", lang)),
    ],
    title=_("NeuroSandboxWebUI - System", lang),
    description=_("This interface displays system information", lang),
    allow_flagging="never",
    clear_btn=None,
    stop_btn=_("Stop", lang),
    submit_btn=_("Display", lang)
)

if settings['custom_theme']['enabled']:
    theme = getattr(gr.themes, settings['theme'])(
        primary_hue=settings['custom_theme']['primary_hue'],
        secondary_hue=settings['custom_theme']['secondary_hue'],
        neutral_hue=settings['custom_theme']['neutral_hue'],
        spacing_size=getattr(gr.themes.sizes, settings['custom_theme']['spacing_size']),
        radius_size=getattr(gr.themes.sizes, settings['custom_theme']['radius_size']),
        text_size=getattr(gr.themes.sizes, settings['custom_theme']['text_size']),
        font=settings['custom_theme']['font'],
        font_mono=settings['custom_theme']['font_mono']
    )
else:
    theme = getattr(gr.themes, settings['theme'])()

with gr.TabbedInterface(
    [
        gr.TabbedInterface(
            [chat_interface, tts_stt_interface, mms_interface, seamless_m4tv2_interface, translate_interface],
            tab_names=[_("LLM", lang), _("TTS-STT", lang), _("MMS", lang), _("SeamlessM4Tv2", lang), _("LibreTranslate", lang)]
        ),
        gr.TabbedInterface(
            [
                gr.TabbedInterface(
                    [txt2img_interface, img2img_interface, depth2img_interface, marigold_interface, pix2pix_interface, controlnet_interface, latent_upscale_interface, sdxl_refiner_interface, inpaint_interface, outpaint_interface, gligen_interface, diffedit_interface, blip_diffusion_interface, animatediff_interface, hotshotxl_interface, video_interface, ldm3d_interface,
                     gr.TabbedInterface([sd3_txt2img_interface, sd3_img2img_interface, sd3_controlnet_interface, sd3_inpaint_interface],
                                        tab_names=[_("txt2img", lang), _("img2img", lang), _("controlnet", lang), _("inpaint", lang)]),
                     cascade_interface, t2i_ip_adapter_interface, ip_adapter_faceid_interface, riffusion_interface],
                    tab_names=[_("txt2img", lang), _("img2img", lang), _("depth2img", lang), _("marigold", lang), _("pix2pix", lang), _("controlnet", lang), _("upscale(latent)", lang), _("refiner", lang), _("inpaint", lang), _("outpaint", lang), _("gligen", lang), _("diffedit", lang), _("blip-diffusion", lang), _("animatediff", lang), _("hotshotxl", lang), _("video", lang), _("ldm3d", lang), _("sd3", lang), _("cascade", lang), _("t2i-ip-adapter", lang), _("ip-adapter-faceid", lang), _("riffusion", lang)]
                ),
                kandinsky_interface, flux_interface, hunyuandit_interface, lumina_interface, kolors_interface, auraflow_interface, wurstchen_interface, deepfloyd_if_interface, pixart_interface, cogview3plus_interface, playgroundv2_interface
            ],
            tab_names=[_("StableDiffusion", lang), _("Kandinsky", lang), _("Flux", lang), _("HunyuanDiT", lang), _("Lumina-T2X", lang), _("Kolors", lang), _("AuraFlow", lang), _("Würstchen", lang), _("DeepFloydIF", lang), _("PixArt", lang), _("CogView3-Plus", lang), _("PlaygroundV2.5", lang)]
        ),
        gr.TabbedInterface(
            [wav2lip_interface, liveportrait_interface, modelscope_interface, zeroscope2_interface, cogvideox_interface, latte_interface],
            tab_names=[_("Wav2Lip", lang), _("LivePortrait", lang), _("ModelScope", lang), _("ZeroScope2", lang), _("CogVideoX", lang), _("Latte", lang)]
        ),
        gr.TabbedInterface(
            [stablefast3d_interface, shap_e_interface, zero123plus_interface],
            tab_names=[_("StableFast3D", lang), _("Shap-E", lang), _("Zero123Plus", lang)]
        ),
        gr.TabbedInterface(
            [stableaudio_interface, audiocraft_interface, audioldm2_interface, bark_interface, rvc_interface, uvr_interface, demucs_interface],
            tab_names=[_("StableAudio", lang), _("AudioCraft", lang), _("AudioLDM2", lang), _("SunoBark", lang), _("RVC", lang), _("UVR", lang), _("Demucs", lang)]
        ),
        extras_interface,
        gr.TabbedInterface(
            [wiki_interface, gallery_interface, model_downloader_interface, settings_interface, system_interface],
            tab_names=[_("Wiki", lang), _("Gallery", lang), _("ModelDownloader", lang), _("Settings", lang), _("System", lang)]
        )
    ],
    tab_names=[_("Text", lang), _("Image", lang), _("Video", lang), _("3D", lang), _("Audio", lang), _("Extras", lang), _("Interface", lang)],
    theme=theme
) as app:
    txt2img_interface.input_components[19].click(stop_generation, [], [], queue=False)
    img2img_interface.input_components[12].click(stop_generation, [], [], queue=False)
    controlnet_interface.input_components[8].click(stop_generation, [], [], queue=False)
    inpaint_interface.input_components[10].click(stop_generation, [], [], queue=False)
    cascade_interface.input_components[3].click(stop_generation, [], [], queue=False)
    riffusion_text2image_interface.input_components[3].click(stop_generation, [], [], queue=False)
    sd3_txt2img_interface.input_components[8].click(stop_generation, [], [], queue=False)
    sd3_img2img_interface.input_components[9].click(stop_generation, [], [], queue=False)
    sd3_controlnet_interface.input_components[6].click(stop_generation, [], [], queue=False)
    sd3_inpaint_interface.input_components[6].click(stop_generation, [], [], queue=False)
    kandinsky_txt2img_interface.input_components[4].click(stop_generation, [], [], queue=False)
    kandinsky_img2img_interface.input_components[5].click(stop_generation, [], [], queue=False)
    kandinsky_inpaint_interface.input_components[5].click(stop_generation, [], [], queue=False)
    flux_txt2img_interface.input_components[5].click(stop_generation, [], [], queue=False)
    flux_img2img_interface.input_components[6].click(stop_generation, [], [], queue=False)
    flux_inpaint_interface.input_components[5].click(stop_generation, [], [], queue=False)
    flux_controlnet_interface.input_components[4].click(stop_generation, [], [], queue=False)
    hunyuandit_txt2img_interface.input_components[3].click(stop_generation, [], [], queue=False)
    hunyuandit_controlnet_interface.input_components[5].click(stop_generation, [], [], queue=False)
    wurstchen_interface.input_components[3].click(stop_generation, [], [], queue=False)
    deepfloyd_if_txt2img_interface.input_components[3].click(stop_generation, [], [], queue=False)
    deepfloyd_if_img2img_interface.input_components[4].click(stop_generation, [], [], queue=False)
    deepfloyd_if_inpaint_interface.input_components[5].click(stop_generation, [], [], queue=False)
    pixart_interface.input_components[4].click(stop_generation, [], [], queue=False)
    cogview3plus_interface.input_components[3].click(stop_generation, [], [], queue=False)
    modelscope_interface.input_components[3].click(stop_generation, [], [], queue=False)
    zeroscope2_interface.input_components[3].click(stop_generation, [], [], queue=False)
    cogvideox_text2video_interface.input_components[4].click(stop_generation, [], [], queue=False)
    cogvideox_image2video_interface.input_components[4].click(stop_generation, [], [], queue=False)
    cogvideox_video2video_interface.input_components[5].click(stop_generation, [], [], queue=False)
    latte_interface.input_components[3].click(stop_generation, [], [], queue=False)
    stableaudio_interface.input_components[3].click(stop_generation, [], [], queue=False)
    audioldm2_interface.input_components[4].click(stop_generation, [], [], queue=False)

    reload_button = gr.Button(_("Reload interface", lang))
    close_button = gr.Button(_("Close terminal", lang))
    folder_button = gr.Button(_("Outputs", lang))

    dropdowns_to_update = [
        chat_interface.input_components[4],
        chat_interface.input_components[5],
        chat_interface.input_components[6],
        chat_interface.input_components[44],
        tts_stt_interface.input_components[3],
        txt2img_interface.input_components[3],
        txt2img_interface.input_components[5],
        txt2img_interface.input_components[6],
        txt2img_interface.input_components[8],
        img2img_interface.input_components[5],
        img2img_interface.input_components[7],
        img2img_interface.input_components[8],
        img2img_interface.input_components[10],
        controlnet_interface.input_components[4],
        inpaint_interface.input_components[6],
        inpaint_interface.input_components[7],
        outpaint_interface.input_components[4],
        gligen_interface.input_components[5],
        animatediff_interface.input_components[5],
        sd3_txt2img_interface.input_components[4],
        sd3_txt2img_interface.input_components[9],
        sd3_txt2img_interface.input_components[10],
        sd3_img2img_interface.input_components[6],
        t2i_ip_adapter_interface.input_components[4],
        ip_adapter_faceid_interface.input_components[5],
        flux_txt2img_interface.input_components[2],
        flux_txt2img_interface.input_components[6],
        flux_txt2img_interface.input_components[7],
        flux_img2img_interface.input_components[3],
        flux_img2img_interface.input_components[7],
        flux_img2img_interface.input_components[8],
        auraflow_interface.input_components[3],
        kolors_txt2img_interface.input_components[3],
        rvc_interface.input_components[1],
        gallery_interface.input_components[0]
    ]

    reload_button.click(reload_interface, outputs=dropdowns_to_update[:16])
    close_button.click(close_terminal, [], [], queue=False)
    folder_button.click(open_outputs_folder, [], [], queue=False)

    github_link = gr.HTML(
        '<div style="text-align: center; margin-top: 20px;">'
        '<a href="https://github.com/Dartvauder/NeuroSandboxWebUI" target="_blank" style="color: blue; text-decoration: none; font-size: 16px; margin-right: 20px;">'
        'GitHub'
        '</a>'
        '<a href="https://huggingface.co/Dartvauder007" target="_blank" style="color: blue; text-decoration: none; font-size: 16px; margin-right: 20px;">'
        'Hugging Face'
        '</a>'
        '<a href="https://civitai.com/user/Dartvauder057" target="_blank" style="color: blue; text-decoration: none; font-size: 16px;">'
        'CivitAI'
        '</a>'
        '</div>'
    )
    create_footer()

    project_root = os.getcwd()

    app.queue(api_open=settings['api_open'], max_size=settings['queue_max_size'], status_update_rate=settings['status_update_rate'])
    app.launch(
        share=settings['share_mode'],
        debug=settings['debug_mode'],
        enable_monitoring=settings['monitoring_mode'],
        inbrowser=settings['auto_launch'],
        show_api=settings['show_api'],
        auth=authenticate if settings['auth'] else None,
        server_name=settings['server_name'],
        server_port=settings['server_port'],
        max_file_size=settings['max_file_size'] * gr.FileSize.MB,
        share_server_address=settings['share_server_address'] if settings['share_server_address'] else None,
        favicon_path="project-image.png",
        allowed_paths=[project_root],
        auth_message=_("Welcome to NeuroSandboxWebUI!", lang)
    )