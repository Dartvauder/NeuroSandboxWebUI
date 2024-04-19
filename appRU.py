import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import soundfile as sf
import os
import torch
from einops import rearrange
from TTS.api import TTS
import whisper
from datetime import datetime
import warnings
import logging
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionImg2ImgPipeline, AutoencoderKL, StableDiffusionLatentUpscalePipeline, StableDiffusionUpscalePipeline
from git import Repo
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
            except (ValueError, RuntimeError):
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
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
    else:
        upscaler = StableDiffusionUpscalePipeline.from_pretrained(
            upscale_model_path,
            original_config_file=original_config_file,
            revision="fp16",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )

    upscaler.to(device)
    upscaler.enable_attention_slicing()
    if XFORMERS_AVAILABLE:
        upscaler.enable_xformers_memory_efficient_attention()

    print(f"Upscale model {upscale_model_name} downloaded")

    upscaler.upscale_factor = upscale_factor
    return upscaler


stop_signal = False


def generate_text_and_speech(input_text, input_audio, llm_model_name, llm_settings_html, llm_model_type, chat_template, max_tokens, max_length,
                             n_ctx, temperature, top_p, top_k, avatar_html, avatar_name, enable_tts, tts_settings_html,
                             speaker_wav, language, tts_temperature, tts_top_p, tts_top_k, tts_speed, stop_generation):
    global chat_dir, tts_model, whisper_model, stop_signal
    stop_signal = False
    if not input_text and not input_audio:
        return "Please, enter your request!", None, None, None, None
    prompt = transcribe_audio(input_audio) if input_audio else input_text
    if not llm_model_name:
        return "Please, select a LLM model!", None, None, None, None
    tokenizer, llm_model, error_message = load_model(llm_model_name, llm_model_type,
                                                     n_ctx=n_ctx if llm_model_type == "llama" else None)
    if error_message:
        return error_message, None, None, None, None
    if chat_template:
        chat_template_path = os.path.join("configs", "LLM", chat_template)
        with open(chat_template_path, "r") as f:
            prompt = f.read().replace("{input}", prompt)
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
                return "Please, select a voice and language for TTS!", None, None, None, None
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tts_model = tts_model.to(device)
        if input_audio:
            if not whisper_model:
                whisper_model = load_whisper_model()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            whisper_model = whisper_model.to(device)
        if llm_model:
            if llm_model_type == "transformers":
                inputs = tokenizer.encode(prompt, return_tensors="pt")
                device = llm_model.device
                inputs = inputs.to(device)

                progress_bar = tqdm(total=max_tokens, desc="Generating text")
                progress_tokens = 0

                outputs = llm_model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    max_length=max_length,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    pad_token_id=tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    num_return_sequences=1,
                )

                for i in range(len(outputs.sequences)):
                    generated_sequence = outputs.sequences[i][inputs.shape[-1]:]
                    progress_tokens += len(generated_sequence)
                    progress_bar.update(progress_tokens - progress_bar.n)

                if stop_signal:
                    return "Generation stopped", None, None, None

                progress_bar.close()

                generated_sequence = outputs.sequences[0][inputs.shape[-1]:]
                text = tokenizer.decode(generated_sequence, skip_special_tokens=True)

            elif llm_model_type == "llama":
                llm_model.n_ctx = n_ctx

                progress_bar = tqdm(total=max_tokens, desc="Generating text")
                progress_tokens = 0

                output = llm_model(
                    prompt,
                    max_tokens=max_tokens,
                    stop=None,
                    echo=False,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=1.1
                )

                progress_tokens = max_tokens
                progress_bar.update(progress_tokens - progress_bar.n)

                if stop_signal:
                    return "Generation stopped", None, None, None

                progress_bar.close()

                text = output['choices'][0]['text']

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
            if stop_signal:
                return text, None, avatar_path, chat_dir, "Generation stopped"
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
                    audio_filename = f"output_{now.strftime('%Y%m%d_%H%M%S')}.wav"
                    audio_path = os.path.join(chat_dir, 'audio', audio_filename)
                    sf.write(audio_path, wav, 22050)
            else:
                wav = tts_model.tts(text=text, speaker_wav=f"inputs/audio/voices/{speaker_wav}", language=language,
                                    temperature=tts_temperature, top_p=tts_top_p, top_k=tts_top_k, speed=tts_speed,
                                    repetition_penalty=repetition_penalty, length_penalty=length_penalty)
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
    return text, audio_path, avatar_path


def generate_image_txt2img(prompt, negative_prompt, stable_diffusion_model_name, vae_model_name, lora_model_names,
                           stable_diffusion_settings_html,
                           stable_diffusion_model_type, stable_diffusion_sampler, stable_diffusion_steps,
                           stable_diffusion_cfg, stable_diffusion_width, stable_diffusion_height,
                           stable_diffusion_clip_skip, enable_upscale=False, upscale_factor="x2", stop_generation=None):
    global stop_signal
    stop_signal = False

    if not stable_diffusion_model_name:
        return None, "Please, select a Stable Diffusion model!"

    stable_diffusion_model_path = os.path.join("inputs", "image", "sd_models",
                                               f"{stable_diffusion_model_name}.safetensors")

    if not os.path.exists(stable_diffusion_model_path):
        return None, f"Stable Diffusion model not found: {stable_diffusion_model_path}"

    try:
        if stable_diffusion_model_type == "SD":
            original_config_file = "configs/sd/v1-inference.yaml"
            vae_config_file = "configs/sd/v1-inference.yaml"
            stable_diffusion_model = StableDiffusionPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                original_config_file=original_config_file, torch_dtype=torch.float16
            )
        elif stable_diffusion_model_type == "SD2":
            original_config_file = "configs/sd/v2-inference.yaml"
            vae_config_file = "configs/sd/v2-inference.yaml"
            stable_diffusion_model = StableDiffusionPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                original_config_file=original_config_file, torch_dtype=torch.float16
            )
        elif stable_diffusion_model_type == "SDXL":
            original_config_file = "configs/sd/sd_xl_base.yaml"
            vae_config_file = "configs/sd/sd_xl_base.yaml"
            stable_diffusion_model = StableDiffusionXLPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto", attention_slice=1,
                original_config_file=original_config_file, torch_dtype=torch.float16
            )
        else:
            return None, "Invalid Stable Diffusion model type!"
    except (ValueError, KeyError):
        return None, "The selected model is not compatible with the chosen model type"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if XFORMERS_AVAILABLE:
        stable_diffusion_model.enable_xformers_memory_efficient_attention()
        stable_diffusion_model.vae.enable_xformers_memory_efficient_attention()
        stable_diffusion_model.unet.enable_xformers_memory_efficient_attention()

    stable_diffusion_model.to(device)
    stable_diffusion_model.text_encoder.to(device)
    stable_diffusion_model.vae.to(device)
    stable_diffusion_model.unet.to(device)

    stable_diffusion_model.safety_checker = None

    if vae_model_name is not None:
        vae_model_path = os.path.join("inputs", "image", "sd_models", "vae", f"{vae_model_name}.safetensors")
        if os.path.exists(vae_model_path):
            vae = AutoencoderKL.from_single_file(vae_model_path, device_map="auto",
                                                 original_config_file=vae_config_file)
            stable_diffusion_model.vae = vae.to(device)
        else:
            print(f"VAE model not found: {vae_model_path}")

    if lora_model_names is not None:
        for lora_model_name in lora_model_names:
            lora_model_path = os.path.join("inputs", "image", "sd_models", "lora", lora_model_name)
            stable_diffusion_model.load_lora_weights(lora_model_path)

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
                    upscaled_image = upscaler(prompt=prompt, image=image, num_inference_steps=30, guidance_scale=0).images[0]
                else:
                    upscaled_image = upscaler(prompt=prompt, image=image)["images"][0]
                image = upscaled_image

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


def generate_image_img2img(prompt, negative_prompt, init_image,
                           strength, stable_diffusion_model_name, vae_model_name, stable_diffusion_settings_html,
                           stable_diffusion_model_type,
                           stable_diffusion_sampler, stable_diffusion_steps, stable_diffusion_cfg,
                           stable_diffusion_clip_skip, stop_generation):
    global stop_signal
    stop_signal = False

    if not stable_diffusion_model_name:
        return None, "Please, select a Stable Diffusion model!"

    if not init_image:
        return None, "Please, upload an initial image!"

    stable_diffusion_model_path = os.path.join("inputs", "image", "sd_models",
                                               f"{stable_diffusion_model_name}.safetensors")

    if not os.path.exists(stable_diffusion_model_path):
        return None, f"Stable Diffusion model not found: {stable_diffusion_model_path}"

    try:
        if stable_diffusion_model_type == "SD":
            original_config_file = "configs/sd/v1-inference.yaml"
            vae_config_file = "configs/sd/v1-inference.yaml"
            stable_diffusion_model = StableDiffusionImg2ImgPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                original_config_file=original_config_file, torch_dtype=torch.float16
            )
        elif stable_diffusion_model_type == "SD2":
            original_config_file = "configs/sd/v2-inference.yaml"
            vae_config_file = "configs/sd/v2-inference.yaml"
            stable_diffusion_model = StableDiffusionPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                original_config_file=original_config_file, torch_dtype=torch.float16
            )
        elif stable_diffusion_model_type == "SDXL":
            original_config_file = "configs/sd/sd_xl_base.yaml"
            vae_config_file = "configs/sd/sd_xl_base.yaml"
            stable_diffusion_model = StableDiffusionImg2ImgPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto", attention_slice=1,
                original_config_file=original_config_file, torch_dtype=torch.float16
            )
        else:
            return None, "Invalid Stable Diffusion model type!"
    except (ValueError, KeyError):
        return None, "The selected model is not compatible with the chosen model type"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if XFORMERS_AVAILABLE:
        stable_diffusion_model.enable_xformers_memory_efficient_attention()
        stable_diffusion_model.vae.enable_xformers_memory_efficient_attention()
        stable_diffusion_model.unet.enable_xformers_memory_efficient_attention()

    stable_diffusion_model.to(device)
    stable_diffusion_model.text_encoder.to(device)
    stable_diffusion_model.vae.to(device)
    stable_diffusion_model.unet.to(device)

    stable_diffusion_model.safety_checker = None

    if vae_model_name is not None:
        vae_model_path = os.path.join("inputs", "image", "sd_models", "vae", f"{vae_model_name}.safetensors")
        if os.path.exists(vae_model_path):
            vae = AutoencoderKL.from_single_file(vae_model_path, device_map=device,
                                                 original_config_file=vae_config_file)
            stable_diffusion_model.vae = vae.to(device)
        else:
            print(f"VAE model not found: {vae_model_path}")

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
        image_dir = os.path.join('outputs', f"images_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, format="PNG")

        return image_path, None

    except Exception as e:
        return None, str(e)

    finally:
        del stable_diffusion_model
        torch.cuda.empty_cache()


def upscale_image(image_path, enable_upscale, stop_generation):
    global stop_signal
    if stop_signal:
        return None, "Generation stopped"

    if not enable_upscale:
        return None, "Please enable upscale to generate an image!"

    if not image_path:
        return None, "Please, upload an initial image!"

    upscale_factor = 2
    upscaler = load_upscale_model(upscale_factor)
    if upscaler:
        image = Image.open(image_path).convert("RGB")
        upscaled_image = upscaler(prompt="", image=image, num_inference_steps=30, guidance_scale=0).images[0]

        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"images_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"upscaled_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        image_path = os.path.join(image_dir, image_filename)
        upscaled_image.save(image_path, format="PNG")

        return image_path, None
    else:
        return None, "Failed to load upscale model"


def generate_audio(prompt, input_audio=None, model_name=None, audiocraft_settings_html=None, model_type="musicgen",
                   duration=10, top_k=250, top_p=0.0,
                   temperature=1.0, cfg_coef=4.0, enable_multiband=False, stop_generation=None):
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
    audio_dir = os.path.join('outputs', f"audio_{today.strftime('%Y%m%d')}")
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
            wav, tokens = model.generate(descriptions, return_tokens=True)
            progress_bar.update(duration)
            if wav.ndim > 2:
                wav = wav.squeeze()
            if stop_signal:
                return None, "Generation stopped"
        progress_bar.close()

        if mbd:
            if stop_signal:
                return None, "Generation stopped"
            print(f"Tokens shape: {tokens.shape}")
            tokens = rearrange(tokens, "b n d -> n b d")  # Изменение формы тензора
            wav_diffusion = mbd.tokens_to_wav(tokens)
            wav_diffusion = wav_diffusion.squeeze()
            if wav_diffusion.ndim == 1:
                wav_diffusion = wav_diffusion.unsqueeze(0)
            max_val = wav_diffusion.abs().max()
            if max_val > 1:
                wav_diffusion = wav_diffusion / max_val
            wav_diffusion = wav_diffusion * 0.99
            audio_filename_diffusion = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}_diffusion.wav"
            audio_path_diffusion = os.path.join(audio_dir, audio_filename_diffusion)
            torchaudio.save(audio_path_diffusion, wav_diffusion.cpu().detach(), model.sample_rate)

        audio_filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        audio_path = os.path.join(audio_dir, audio_filename)
        audio_write(audio_path, wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

        return audio_path + ".wav", None

    finally:
        del model
        if mbd:
            del mbd
        torch.cuda.empty_cache()


def stop_all_processes():
    global stop_signal
    stop_signal = True


def close_terminal():
    os._exit(1)


def get_chat_templates():
    chat_templates_dir = "configs/llm"
    return [None] + [template for template in os.listdir(chat_templates_dir) if template.endswith(".yaml")]


llm_models_list = [None] + [model for model in os.listdir("inputs/text/llm_models") if not model.endswith(".txt")]
avatars_list = [None] + [avatar for avatar in os.listdir("inputs/image/avatars") if not avatar.endswith(".txt")]
speaker_wavs_list = [None] + [wav for wav in os.listdir("inputs/audio/voices") if not wav.endswith(".txt")]
stable_diffusion_models_list = [None] + [model.replace(".safetensors", "") for model in
                                         os.listdir("inputs/image/sd_models")
                                         if (model.endswith(".safetensors") or not model.endswith(".txt") and not os.path.isdir(os.path.join("inputs/image/sd_models")))]
audiocraft_models_list = [None] + ["musicgen-stereo-medium", "audiogen-medium", "musicgen-stereo-melody",
                                   "magnet-medium-30sec", "magnet-medium-10sec", "audio-magnet-medium"]
vae_models_list = [None] + [model.replace(".safetensors", "") for model in os.listdir("inputs/image/sd_models/vae") if
                            model.endswith(".safetensors") or not model.endswith(".txt")]
lora_models_list = [None] + [model for model in os.listdir("inputs/image/sd_models/lora") if
                             model.endswith(".safetensors")]


chat_interface = gr.Interface(
    fn=generate_text_and_speech,
    inputs=[
        gr.Textbox(label="Enter your request"),
        gr.Audio(type="filepath", label="Record your request (optional)"),
        gr.Dropdown(choices=llm_models_list, label="Select LLM model", value=None),
        gr.HTML("<h3>LLM Settings</h3>"),
        gr.Radio(choices=["transformers", "llama"], label="Select model type", value="transformers"),
        gr.Dropdown(choices=get_chat_templates(), label="Select chat template", value=None),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max tokens"),
        gr.Slider(minimum=1, maximum=2048, value=1024, step=1, label="Max length"),
        gr.Slider(minimum=0, maximum=4096, value=2048, step=1, label="n_ctx (for llama models only)", interactive=True),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.1, label="Top P"),
        gr.Slider(minimum=0, maximum=100, value=30, step=1, label="Top K"),
        gr.HTML("<br>"),
        gr.Dropdown(choices=avatars_list, label="Select avatar", value=None),
        gr.Checkbox(label="Enable TTS", value=False),
        gr.HTML("<h3>TTS Settings</h3>"),
        gr.Dropdown(choices=speaker_wavs_list, label="Select voice", interactive=True),
        gr.Dropdown(choices=["en", "ru"], label="Select language", interactive=True),
        gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.1, label="TTS Temperature", interactive=True),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.1, label="TTS Top P", interactive=True),
        gr.Slider(minimum=0, maximum=100, value=30, step=1, label="TTS Top K", interactive=True),
        gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="TTS Speed", interactive=True),
        gr.Button(value="Stop generation", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Textbox(label="LLM text response", type="text"),
        gr.Audio(label="LLM audio response", type="filepath"),
        gr.Image(type="filepath", label="Avatar"),
    ],
    title="NeuroChatWebUI (ALPHA) - LLM",
    description="This user interface allows you to enter any text or audio and receive "
                "generated response. You can select the LLM model, "
                "avatar, voice and language from the drop-down lists. You can also customize the model settings from "
                "using sliders. Try it and see what happens!",
    allow_flagging="never",
)

txt2img_interface = gr.Interface(
    fn=generate_image_txt2img,
    inputs=[
        gr.Textbox(label="Enter your prompt"),
        gr.Textbox(label="Enter your negative prompt", value=""),
        gr.Dropdown(choices=stable_diffusion_models_list, label="Select Stable Diffusion model", value=None),
        gr.Dropdown(choices=vae_models_list, label="Select VAE model (optional)", value=None),
        gr.Dropdown(choices=lora_models_list, label="Select LORA models", value=None, multiselect=True),
        gr.HTML("<h3>Stable Diffusion Settings</h3>"),
        gr.Radio(choices=["SD", "SD2", "SDXL"], label="Select model type", value="SD"),
        gr.Dropdown(choices=["euler_ancestral", "euler", "lms", "heun", "dpm", "dpm_solver", "dpm_solver++"],
                    label="Select sampler", value="euler_ancestral"),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Steps"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label="CFG"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height"),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Clip skip"),
        gr.Checkbox(label="Enable upscale", value=False),
        gr.Radio(choices=["x2", "x4"], label="Upscale size", value="x2"),
        gr.Button(value="Stop generation", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="Generated image"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroChatWebUI (ALPHA) - Stable Diffusion (txt2img)",
    description="This user interface allows you to enter any text and generate images using Stable Diffusion. "
                "You can select the Stable Diffusion model and customize the generation settings from the sliders. "
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
        gr.Dropdown(choices=stable_diffusion_models_list, label="Select Stable Diffusion model", value=None),
        gr.Dropdown(choices=vae_models_list, label="Select VAE model (optional)", value=None),
        gr.HTML("<h3>Stable Diffusion Settings</h3>"),
        gr.Radio(choices=["SD", "SD2", "SDXL"], label="Select model type", value="SD"),
        gr.Dropdown(choices=["euler_ancestral", "euler", "lms", "heun", "dpm", "dpm_solver", "dpm_solver++"],
                    label="Select sampler", value="euler_ancestral"),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Steps"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label="CFG"),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Clip skip"),
        gr.Button(value="Stop generation", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="Generated image"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroChatWebUI (ALPHA) - Stable Diffusion (img2img)",
    description="This user interface allows you to enter any text and image to generate new images using Stable Diffusion. "
                "You can select the Stable Diffusion model and customize the generation settings from the sliders. "
                "Try it and see what happens!",
    allow_flagging="never",
)

extras_interface = gr.Interface(
    fn=upscale_image,
    inputs=[
        gr.Image(label="Image to upscale", type="filepath"),
        gr.Checkbox(label="Enable upscale", value=False),
        gr.Button(value="Stop generation", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Image(type="filepath", label="Modified Image"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroChatWebUI (ALPHA) - Stable Diffusion (Extras)",
    description="This user interface allows you to upload an image and transform it using different options",
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
        gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Temperature"),
        gr.Slider(minimum=1.0, maximum=10.0, value=4.0, step=0.1, label="CFG"),
        gr.Checkbox(label="Enable Multiband Diffusion", value=False),
        gr.Button(value="Stop generation", interactive=True, variant="stop"),
    ],
    outputs=[
        gr.Audio(label="Generated audio", type="filepath"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroChatWebUI (ALPHA) - AudioCraft",
    description="This user interface allows you to enter any text and generate audio using AudioCraft. "
                "You can select the AudioCraft model and customize the generation settings from the sliders. "
                "Try it and see what happens!",
    allow_flagging="never",
)

with gr.TabbedInterface(
        [chat_interface, gr.TabbedInterface([txt2img_interface, img2img_interface, extras_interface],
                                            tab_names=["txt2img", "img2img", "Extras"]),
         audiocraft_interface],
        tab_names=["LLM", "Stable Diffusion", "AudioCraft"]
) as app:
    chat_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    txt2img_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    img2img_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    extras_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)
    audiocraft_interface.input_components[-1].click(stop_all_processes, [], [], queue=False)

    close_button = gr.Button("Close terminal")
    close_button.click(close_terminal, [], [], queue=False)

    github_link = gr.HTML(
        '<div style="text-align: center; margin-top: 20px;">'
        '<a href="https://github.com/Dartvauder/NeuroChatWebUI" target="_blank" style="color: blue; text-decoration: none; font-size: 16px;">'
        'GitHub'
        '</a>'
        '</div>'
    )

    app.launch()
    
