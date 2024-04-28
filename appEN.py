import gradio as gr
import langdetect
from transformers import AutoModelForCausalLM, AutoTokenizer
from libretranslatepy import LibreTranslateAPI
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
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionDepth2ImgPipeline, AutoencoderKL, StableDiffusionLatentUpscalePipeline, StableDiffusionUpscalePipeline, StableDiffusionInpaintPipeline, StableVideoDiffusionPipeline, I2VGenXLPipeline
from diffusers.utils import load_image, export_to_video, export_to_gif
from git import Repo
import numpy as np
from PIL import Image
from tqdm import tqdm
from llama_cpp import Llama
import requests
import torchaudio
from audiocraft.models import MusicGen, AudioGen, MultiBandDiffusion  # MAGNeT
from audiocraft.data.audio import audio_write

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
                    torch_dtype=torch.float16,
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


def generate_text_and_speech(input_text, input_audio, llm_model_name, llm_settings_html, llm_model_type, max_length, max_tokens,
                             temperature, top_p, top_k, chat_history_format, avatar_name, enable_tts, tts_settings_html,
                             speaker_wav, language, tts_temperature, tts_top_p, tts_top_k, tts_speed, output_format, stop_generation):
    global chat_history, chat_dir, tts_model, whisper_model, stop_signal
    stop_signal = False
    if not input_text and not input_audio:
        chat_history.append(["Please, enter your request!", None])
        return chat_history, None, None, None, None
    prompt = transcribe_audio(input_audio) if input_audio else input_text
    if not llm_model_name:
        chat_history.append([None, "Please, select a LLM model!"])
        return chat_history, None, None, None, None
    tokenizer, llm_model, error_message = load_model(llm_model_name, llm_model_type)
    if error_message:
        chat_history.append([None, error_message])
        return chat_history, None, None, None, None
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
                chat_history.append([None, "Please, select a voice and language for TTS!"])
                return chat_history, None, None, None, None
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
                if detect_lang == "en":
                    bot_instruction = "I am a chatbot created to help with any questions. I use my knowledge and abilities to provide useful and meaningful answers in any language"
                else:
                    bot_instruction = "Я чат-бот, созданный для помощи по любым вопросам. Я использую свои знания и способности, чтобы давать полезные и содержательные ответы на любом языке"
                inputs = tokenizer.encode(bot_instruction + prompt, return_tensors="pt", truncation=True)
                device = llm_model.device
                inputs = inputs.to(device)

                progress_bar = tqdm(total=max_length, desc="Generating text")
                progress_tokens = 0

                outputs = llm_model.generate(
                    inputs,
                    max_new_tokens=None,
                    max_length=max_length,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    repetition_penalty=1.15,
                    early_stopping=True,
                    num_beams=5,
                    no_repeat_ngram_size=2,
                    do_sample=True,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    num_return_sequences=1,
                )

                for i in range(len(outputs.sequences)):
                    generated_sequence = outputs.sequences[i][inputs.shape[-1]:]
                    progress_tokens += len(generated_sequence)
                    progress_bar.update(progress_tokens - progress_bar.n)

                if stop_signal:
                    chat_history.append([prompt, "Generation stopped"])
                    return chat_history, None, None, None

                progress_bar.close()

                generated_sequence = outputs.sequences[0][inputs.shape[-1]:]
                text = tokenizer.decode(generated_sequence, skip_special_tokens=True)

            elif llm_model_type == "llama":
                detect_lang = langdetect.detect(prompt)
                if detect_lang == "en":
                    instruction = "I am a chatbot created to help with any questions. I use my knowledge and abilities to provide useful and meaningful answers in any language\n\nHuman: "
                else:
                    instruction = "Я чат-бот, созданный для помощи по любым вопросам. Я использую свои знания и способности, чтобы давать полезные и содержательные ответы на любом языке\n\nЧеловек: "

                prompt_with_instruction = instruction + prompt + "\nAssistant: "

                progress_bar = tqdm(total=max_tokens, desc="Generating text")
                progress_tokens = 0

                output = llm_model(
                    prompt_with_instruction,
                    max_tokens=max_tokens,
                    stop=["Q:", "\n"],
                    echo=False,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=1.15,
                )

                progress_tokens = max_tokens
                progress_bar.update(progress_tokens - progress_bar.n)

                if stop_signal:
                    chat_history.append([prompt, "Generation stopped"])
                    return chat_history, None, None, None

                progress_bar.close()

                text = output['choices'][0]['text']

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
        avatar_path = f"inputs/image/avatars/{avatar_name}" if avatar_name else None
        if enable_tts and text:
            if stop_signal:
                chat_history.append([prompt, text])
                return chat_history, None, avatar_path, chat_dir, "Generation stopped"
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
    return chat_history, audio_path, avatar_path, chat_dir, None


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


def translate_text(text, source_lang, target_lang):
    translator = LibreTranslateAPI("http://127.0.0.1:5000")
    translation = translator.translate(text, source_lang, target_lang)
    return translation


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
        images = stable_diffusion_model(prompt, negative_prompt=negative_prompt,
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

        images = stable_diffusion_model(prompt, negative_prompt=negative_prompt,
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
        os.makedirs(stable_diffusion_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/stabilityai/stable-diffusion-2-depth", stable_diffusion_model_path)

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


def generate_video(init_image, output_format, video_settings_html, motion_bucket_id, noise_aug_strength, fps, num_frames, decode_chunk_size,
                   iv2gen_xl_settings_html, prompt, negative_prompt, num_inference_steps, guidance_scale, stop_generation):
    global stop_signal
    stop_signal = False

    if not init_image:
        return None, "Please upload an initial image!"

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
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path=video_model_path,
                torch_dtype=torch.float16,
                variant="fp16"
            )
            pipe.enable_model_cpu_offload()
            pipe.unet.enable_forward_chunking()

            image = load_image(init_image)
            image = image.resize((1024, 576))

            generator = torch.manual_seed(42)
            frames = pipe(image, decode_chunk_size=decode_chunk_size, generator=generator,
                          motion_bucket_id=motion_bucket_id, noise_aug_strength=noise_aug_strength, num_frames=num_frames).frames[0]

            if stop_signal:
                return None, "Generation stopped"

            video_filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            video_path = os.path.join(video_dir, video_filename)
            export_to_video(frames, video_path, fps=fps)

            return video_path, None

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
            pipeline = I2VGenXLPipeline.from_pretrained(video_model_path, torch_dtype=torch.float16, variant="fp16")
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
                return None, "Generation stopped"

            video_filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
            video_path = os.path.join(video_dir, video_filename)
            export_to_gif(frames, video_path)

            return video_path, None

        finally:
            try:
                del pipeline
            except UnboundLocalError:
                pass
            torch.cuda.empty_cache()


def generate_audio(prompt, input_audio=None, model_name=None, audiocraft_settings_html=None, model_type="musicgen",
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
            model = MusicGen.get_pretrained(audiocraft_model_path)
            model.set_generation_params(duration=duration)
        elif model_type == "audiogen":
            model = AudioGen.get_pretrained(audiocraft_model_path)
            model.set_generation_params(duration=duration)
        #        elif model_type == "magnet":
        #            model = MAGNeT.get_pretrained(audiocraft_model_path)
        #            model.set_generation_params()
        else:
            return None, "Invalid model type!"
    except (ValueError, AssertionError):
        return None, "The selected model is not compatible with the chosen model type"

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


def demucs_separate(audio_file, output_format="wav"):
    global stop_signal
    if stop_signal:
        return None, "Generation stopped"

    if not audio_file:
        return None, "Please upload an audio file!"

    today = datetime.now().date()
    demucs_dir = os.path.join("outputs", f"Demucs_{today.strftime('%Y%m%d')}")
    os.makedirs(demucs_dir, exist_ok=True)

    separate_dir = os.path.join(demucs_dir, "separate")
    os.makedirs(separate_dir, exist_ok=True)

    try:
        command = f"demucs --two-stems=vocals {audio_file} -o {separate_dir}"
        subprocess.run(command, shell=True, check=True)

        if stop_signal:
            return None, "Generation stopped"

        vocal_file = os.path.join(separate_dir, "mdx_extra", "vocals.wav")
        instrumental_file = os.path.join(separate_dir, "mdx_extra", "no_vocals.wav")

        if output_format == "mp3":
            vocal_output = os.path.join(separate_dir, "mdx_extra", "vocal.mp3")
            instrumental_output = os.path.join(separate_dir, "mdx_extra", "instrumental.mp3")
            subprocess.run(f"ffmpeg -i {vocal_file} -b:a 192k {vocal_output}", shell=True, check=True)
            subprocess.run(f"ffmpeg -i {instrumental_file} -b:a 192k {instrumental_output}", shell=True, check=True)
        elif output_format == "ogg":
            vocal_output = os.path.join(separate_dir, "mdx_extra", "vocal.ogg")
            instrumental_output = os.path.join(separate_dir, "mdx_extra", "instrumental.ogg")
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


llm_models_list = [None] + [model for model in os.listdir("inputs/text/llm_models") if not model.endswith(".txt")]
avatars_list = [None] + [avatar for avatar in os.listdir("inputs/image/avatars") if not avatar.endswith(".txt")]
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
        gr.Dropdown(choices=llm_models_list, label="Select LLM model", value=None),
        gr.HTML("<h3>LLM Settings</h3>"),
        gr.Radio(choices=["transformers", "llama"], label="Select model type", value="transformers"),
        gr.Slider(minimum=1, maximum=4096, value=512, step=1, label="Max length (for transformers type models)"),
        gr.Slider(minimum=1, maximum=4096, value=512, step=1, label="Max tokens (for llama type models)"),
        gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.1, label="Top P"),
        gr.Slider(minimum=0, maximum=100, value=20, step=1, label="Top K"),
        gr.Radio(choices=["txt", "json"], label="Select chat history format", value="txt", interactive=True),
        gr.Dropdown(choices=avatars_list, label="Select avatar", value=None),
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
        gr.Image(type="filepath", label="Avatar"),
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

translate_interface = gr.Interface(
    fn=translate_text,
    inputs=[
        gr.Textbox(label="Enter text to translate"),
        gr.Dropdown(choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hi"], label="Select source language", value="en"),
        gr.Dropdown(choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hi"], label="Select target language", value="ru"),
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
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroSandboxWebUI (ALPHA) - StableDiffusion (video)",
    description="This user interface allows you to enter an initial image and generate a video using StableVideoDiffusion(mp4) and I2VGen-xl(gif). "
                "You can customize the generation settings from the sliders. "
                "Try it and see what happens!",
    allow_flagging="never",
)

audiocraft_interface = gr.Interface(
    fn=generate_audio,
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

with gr.TabbedInterface(
        [chat_interface, tts_stt_interface, translate_interface, gr.TabbedInterface([txt2img_interface, img2img_interface, depth2img_interface, upscale_interface, inpaint_interface, video_interface],
        tab_names=["txt2img", "img2img", "depth2img", "upscale", "inpaint", "video"]),
         audiocraft_interface, demucs_interface, model_downloader_interface, settings_interface],
        tab_names=["LLM", "TTS-STT", "LibreTranslate", "StableDiffusion", "AudioCraft", "Demucs", "ModelDownloader", "Settings"]
) as app:
    chat_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    txt2img_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    img2img_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    depth2img_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    upscale_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    inpaint_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    video_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    audiocraft_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)

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
