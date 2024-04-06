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
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionImg2ImgPipeline, AutoencoderKL, StableDiffusionLatentUpscalePipeline, StableDiffusionUpscalePipeline
from git import Repo
from PIL import Image
from llama_cpp import Llama
import requests
import torchaudio
from audiocraft.models import MusicGen, AudioGen, MultiBandDiffusion
from audiocraft.data.audio import audio_write

XFORMERS_AVAILABLE = False
torch.cuda.is_available()
try:
    import xformers
    import xformers.ops
    XFORMERS_AVAILABLE = True
except ImportError:
    print("Xformers не установлен. Генерация будет без него")

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
    if model_name:
        model_path = f"inputs/text/llm_models/{model_name}"
        if model_type == "transformers":
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = AutoModelForCausalLM.from_pretrained(model_path)
            except (ValueError, OSError):
                return None, None, "Выбранная модель несовместима с типом модели 'transformers'."
        elif model_type == "llama":
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = Llama(model_path, n_gpu_layers=-1 if device == "cuda" else 0)
                model.n_ctx = n_ctx
                tokenizer = None
            except (ValueError, RuntimeError):
                return None, None, "Выбранная модель несовместима с типом модели 'llama'."
        else:
            return None, None, "Выбран неверный тип модели"

        if XFORMERS_AVAILABLE:
            try:
                model = model.with_xformers()
            except (AttributeError, ImportError):
                try:
                    model.decoder.enable_xformers_memory_efficient_attention()
                    model.encoder.enable_xformers_memory_efficient_attention()
                except AttributeError:
                    pass

        if model_type == "transformers":
            return tokenizer, model.to(device), None
        else:
            return None, model, None

    return None, None, None


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
    print("Загрузка TTS...")
    tts_model_path = "inputs/audio/XTTS-v2"
    if not os.path.exists(tts_model_path):
        os.makedirs(tts_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/coqui/XTTS-v2", tts_model_path)
    print("TTS загружен")
    return TTS(model_path=tts_model_path, config_path=f"{tts_model_path}/config.json")


def load_whisper_model():
    print("Загрузка Whisper...")
    whisper_model_path = "inputs/text/whisper-medium"
    if not os.path.exists(whisper_model_path):
        os.makedirs(whisper_model_path, exist_ok=True)
        url = ("https://openaipublic.azureedge.net/main/whisper/models"
               "/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt")
        r = requests.get(url, allow_redirects=True)
        open(os.path.join(whisper_model_path, "medium.pt"), "wb").write(r.content)
    print("Whisper загружен")
    model_file = os.path.join(whisper_model_path, "medium.pt")
    return whisper.load_model(model_file)


def load_audiocraft_model(model_name):
    global audiocraft_model_path
    print(f"Загрузка модели AudioCraft: {model_name}...")
    audiocraft_model_path = os.path.join("inputs", "audio", "audiocraft", model_name)
    if not os.path.exists(audiocraft_model_path):
        os.makedirs(audiocraft_model_path, exist_ok=True)
        if model_name == "musicgen-stereo-medium":
            Repo.clone_from("https://huggingface.co/facebook/musicgen-stereo-medium", audiocraft_model_path)
        elif model_name == "audiogen-medium":
            Repo.clone_from("https://huggingface.co/facebook/audiogen-medium", audiocraft_model_path)
        elif model_name == "musicgen-stereo-melody":
            Repo.clone_from("https://huggingface.co/facebook/musicgen-stereo-melody", audiocraft_model_path)
    print(f"Модель AudioCraft {model_name} загружена")
    return audiocraft_model_path


def load_multiband_diffusion_model():
    print(f"Загрузка Multiband Diffusion")
    multiband_diffusion_path = os.path.join("inputs", "audio", "audiocraft", "multiband-diffusion")
    if not os.path.exists(multiband_diffusion_path):
        os.makedirs(multiband_diffusion_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/facebook/multiband-diffusion", multiband_diffusion_path)
        print("Multiband Diffusion загружен")
    return multiband_diffusion_path


def load_upscale_model(upscale_factor):
    original_config_file = None
    
    if upscale_factor == 2:
        upscale_model_name = "stabilityai/sd-x2-latent-upscaler"
        upscale_model_path = os.path.join("inputs", "image", "sd_models", "upscale", "x2-upscaler")
    else:
        upscale_model_name = "stabilityai/stable-diffusion-x4-upscaler"
        upscale_model_path = os.path.join("inputs", "image", "sd_models", "upscale", "x4-upscaler")
        original_config_file = "configs/sd/x4-upscaling.yaml"

    print(f"Загрузка модели Upscale: {upscale_model_name}")

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

    print(f"Модель Upscale {upscale_model_name} загружена")

    upscaler.upscale_factor = upscale_factor
    return upscaler


stop_signal = False


def generate_text_and_speech(input_text, input_audio, llm_model_name, llm_model_type, max_tokens, n_ctx, temperature,
                             top_p, top_k, avatar_name, enable_tts, speaker_wav, language):
    global chat_dir, tts_model, whisper_model, stop_signal
    stop_signal = False

    if not input_text and not input_audio:
        return "Пожалуйста, введите ваш запрос!", None, None, None, None

    prompt = transcribe_audio(input_audio) if input_audio else input_text

    if not llm_model_name:
        return "Пожалуйста, выберите LLM модель!", None, None, None, None

    tokenizer, llm_model, error_message = load_model(llm_model_name, llm_model_type,
                                                     n_ctx=n_ctx if llm_model_type == "llama" else None)
    if error_message:
        return error_message, None, None, None, None

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
                return "Пожалуйста, выберите голос и язык для TTS!", None, None, None, None
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
                outputs = llm_model.generate(inputs, max_new_tokens=max_tokens, top_p=top_p, top_k=top_k,
                                             temperature=temperature, pad_token_id=tokenizer.eos_token_id)
                if stop_signal:
                    return "Генерация остановлена", None, None, None
                generated_sequence = outputs[0][inputs.shape[-1]:]
                text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
            elif llm_model_type == "llama":
                llm_model.n_ctx = n_ctx
                output = llm_model(prompt, max_tokens=max_tokens, stop=None, echo=False,
                                   temperature=temperature, top_p=top_p, top_k=top_k,
                                   repeat_penalty=1.1)
                if stop_signal:
                    return "Генерация остановлена", None, None, None
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


def generate_image_txt2img(prompt, negative_prompt, stable_diffusion_model_name, vae_model_name,
                           stable_diffusion_model_type, stable_diffusion_sampler, stable_diffusion_steps,
                           stable_diffusion_cfg, stable_diffusion_width, stable_diffusion_height,
                           stable_diffusion_clip_skip, enable_upscale=False, upscale_factor="x2"):
    global stop_signal
    stop_signal = False

    if not stable_diffusion_model_name:
        return None, "Пожалуйста, выберите модель Stable Diffusion!"

    stable_diffusion_model_path = os.path.join("inputs", "image", "sd_models",
                                               f"{stable_diffusion_model_name}.safetensors")

    if not os.path.exists(stable_diffusion_model_path):
        return None, f"Модель Stable Diffusion не найдена: {stable_diffusion_model_path}"

    try:
        if stable_diffusion_model_type == "SD":
            original_config_file = "configs/sd/v1-inference.yaml"
            vae_config_file = "configs/sd/v1-inference.yaml"
            stable_diffusion_model = StableDiffusionPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                original_config_file=original_config_file
            )
        elif stable_diffusion_model_type == "SDXL":
            original_config_file = "configs/sd/sd_xl_base.yaml"
            vae_config_file = "configs/sd/sd_xl_base.yaml"
            stable_diffusion_model = StableDiffusionXLPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto", attention_slice=1,
                original_config_file=original_config_file
            )
        else:
            return None, "Неверный тип модели Stable Diffusion!"
    except (ValueError, KeyError):
        return None, "Выбранная модель несовместима с выбранным типом модели"

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
            print(f"Модель VAE не найдена: {vae_model_path}")

    try:
        images = stable_diffusion_model(prompt, negative_prompt=negative_prompt,
                                        num_inference_steps=stable_diffusion_steps,
                                        guidance_scale=stable_diffusion_cfg, height=stable_diffusion_height,
                                        width=stable_diffusion_width, clip_skip=stable_diffusion_clip_skip,
                                        sampler=stable_diffusion_sampler)
        if stop_signal:
            return None, "Генерация остановлена"
        image = images["images"][0]

        if enable_upscale:
            upscale_factor_value = 2 if upscale_factor == "x2" else 4
            upscaler = load_upscale_model(upscale_factor_value)
            if upscaler:
                if upscale_factor == "x2":
                    upscaled_image = \
                        upscaler(prompt=prompt, image=image, num_inference_steps=30, guidance_scale=0).images[0]
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
                           strength, stable_diffusion_model_name, vae_model_name, stable_diffusion_model_type,
                           stable_diffusion_sampler, stable_diffusion_steps, stable_diffusion_cfg,
                           stable_diffusion_clip_skip):
    global stop_signal
    stop_signal = False

    if not stable_diffusion_model_name:
        return None, "Пожалуйста, выберите модель Stable Diffusion!"

    if not init_image:
        return None, "Пожалуйста, загрузите исходное изображение!"

    stable_diffusion_model_path = os.path.join("inputs", "image", "sd_models",
                                               f"{stable_diffusion_model_name}.safetensors")

    if not os.path.exists(stable_diffusion_model_path):
        return None, f"Модель Stable Diffusion не найдена: {stable_diffusion_model_path}"

    try:
        if stable_diffusion_model_type == "SD":
            original_config_file = "configs/sd/v1-inference.yaml"
            vae_config_file = "configs/sd/v1-inference.yaml"
            stable_diffusion_model = StableDiffusionImg2ImgPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto",
                original_config_file=original_config_file
            )
        elif stable_diffusion_model_type == "SDXL":
            original_config_file = "configs/sd/sd_xl_base.yaml"
            vae_config_file = "configs/sd/sd_xl_base.yaml"
            stable_diffusion_model = StableDiffusionImg2ImgPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto", attention_slice=1,
                original_config_file=original_config_file
            )
        else:
            return None, "Неверный тип модели Stable Diffusion!"
    except (ValueError, KeyError):
        return None, "Выбранная модель несовместима с выбранным типом модели"

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
            print(f"Модель VAE не найдена: {vae_model_path}")

    try:
        init_image = Image.open(init_image).convert("RGB")
        init_image = stable_diffusion_model.image_processor.preprocess(init_image)

        images = stable_diffusion_model(prompt, negative_prompt=negative_prompt,
                                        num_inference_steps=stable_diffusion_steps,
                                        guidance_scale=stable_diffusion_cfg, clip_skip=stable_diffusion_clip_skip,
                                        sampler=stable_diffusion_sampler, image=init_image, strength=strength)
        if stop_signal:
            return None, "Генерация остановлена"
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


def generate_audio(prompt, input_audio=None, model_name=None, model_type="musicgen", duration=10, top_k=250, top_p=0.0,
                   temperature=1.0, cfg_coef=4.0, enable_multiband=False):
    global audiocraft_model_path, stop_signal
    stop_signal = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not model_name:
        return None, "Пожалуйста, выберите модель AudioCraft!"

    if not audiocraft_model_path:
        audiocraft_model_path = load_audiocraft_model(model_name)

    try:
        if model_type == "musicgen":
            model = MusicGen.get_pretrained(audiocraft_model_path)
            model.set_generation_params(duration=duration)
        elif model_type == "audiogen":
            model = AudioGen.get_pretrained(audiocraft_model_path)
            model.set_generation_params(duration=duration)
        else:
            return None, "Неверный тип модели!"
    except (ValueError, AssertionError):
        return None, "Выбранная модель несовместима с выбранным типом модели"

    multiband_diffusion_model = None

    if enable_multiband:
        multiband_diffusion_path = load_multiband_diffusion_model()
        if model_type == "musicgen":
            multiband_diffusion_model = MultiBandDiffusion.get_mbd_musicgen(multiband_diffusion_path)
            multiband_diffusion_model.to(device)

    try:
        if input_audio and model_type == "musicgen":
            audio_path = input_audio
            melody, sr = torchaudio.load(audio_path)
            model.set_generation_params(duration=duration, top_k=top_k, top_p=top_p, temperature=temperature,
                                        cfg_coef=cfg_coef)
            wav = model.generate_with_chroma([prompt], melody[None].expand(1, -1, -1), sr)
            if wav.ndim > 2:
                wav = wav.squeeze()
            if stop_signal:
                return None, "Генерация остановлена"
        else:
            descriptions = [prompt]
            model.set_generation_params(duration=duration, top_k=top_k, top_p=top_p, temperature=temperature,
                                        cfg_coef=cfg_coef)
            wav = model.generate(descriptions)
            if wav.ndim > 2:
                wav = wav.squeeze()
            if stop_signal:
                return None, "Генерация остановлена"

        if multiband_diffusion_model:
            wav = wav.unsqueeze(0)
            wav = wav.to(device)
            wav = multiband_diffusion_model.compress_and_decompress(wav)
            wav = wav.squeeze(0)

        today = datetime.now().date()
        audio_dir = os.path.join('outputs', f"audio_{today.strftime('%Y%m%d')}")
        os.makedirs(audio_dir, exist_ok=True)
        audio_filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        audio_path = os.path.join(audio_dir, audio_filename)
        audio_write(audio_path, wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

        return audio_path + ".wav", None

    finally:
        del model
        if multiband_diffusion_model:
            del multiband_diffusion_model
        torch.cuda.empty_cache()

def upscale_image(image_path):
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
        return None, "Не удалось загрузить модель Upscale"


def stop_all_processes():
    global stop_signal
    stop_signal = True


llm_models_list = [None] + [model for model in os.listdir("inputs/text/llm_models") if not model.endswith(".txt")]
avatars_list = [None] + [avatar for avatar in os.listdir("inputs/image/avatars") if not avatar.endswith(".txt")]
speaker_wavs_list = [None] + [wav for wav in os.listdir("inputs/audio/voices") if not wav.endswith(".txt")]
stable_diffusion_models_list = [None] + [model.replace(".safetensors", "") for model in
                                         os.listdir("inputs/image/sd_models")
                                         if (model.endswith(".safetensors") or not model.endswith(".txt") and not os.path.isdir(os.path.join("inputs/image/sd_models")))]
audiocraft_models_list = [None] + ["musicgen-stereo-medium", "audiogen-medium", "musicgen-stereo-melody"]
vae_models_list = [None] + [model.replace(".safetensors", "") for model in os.listdir("inputs/image/sd_models/vae") if
                            model.endswith(".safetensors") or not model.endswith(".txt")]

chat_interface = gr.Interface(
    fn=generate_text_and_speech,
    inputs=[
        gr.Textbox(label="Ввести ваш запрос"),
        gr.Audio(type="filepath", label="Записать ваш запрос (опционально)"),
        gr.Dropdown(choices=llm_models_list, label="Выбрать LLM модель", value=None),
        gr.Radio(choices=["transformers", "llama"], label="Выбрать тип модели", value="transformers"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Максимум токенов"),
        gr.Slider(minimum=0, maximum=4096, value=2048, step=1, label="n_ctx (только для моделей llama)", interactive=True),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, label="Температура"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.1, label="Top P"),
        gr.Slider(minimum=0, maximum=100, value=30, step=1, label="Top K"),
        gr.Dropdown(choices=avatars_list, label="Выбрать аватар", value=None),
        gr.Checkbox(label="Включить TTS", value=False),
        gr.Dropdown(choices=speaker_wavs_list, label="Выбрать голос", interactive=True),
        gr.Dropdown(choices=["en", "ru"], label="Выбрать язык", interactive=True),
    ],
    outputs=[
        gr.Textbox(label="Текстовый ответ от LLM", type="text"),
        gr.Audio(label="Аудио ответ от LLM", type="filepath"),
        gr.Image(type="filepath", label="Аватар"),
    ],
    title="НейроЧатWebUI (АЛЬФА) - LLM",
    description="Этот пользовательский интерфейс позволяет вам вводить любой текст или аудио и получать "
                "сгенерированный ответ. Вы можете выбрать модель LLM, "
                "аватар, голос и язык из раскрывающихся списков. Вы также можете настроить параметры модели "
                "используя ползунки. Попробуйте и посмотрите, что получится!",
    allow_flagging="never",
)

txt2img_interface = gr.Interface(
    fn=generate_image_txt2img,
    inputs=[
        gr.Textbox(label="Ввести ваш запрос"),
        gr.Textbox(label="Ввести ваш негативный запрос", value=""),
        gr.Dropdown(choices=stable_diffusion_models_list, label="Выбрать модель Stable Diffusion", value=None),
        gr.Dropdown(choices=vae_models_list, label="Выбрать модель VAE (опционально)", value=None),
        gr.Radio(choices=["SD", "SDXL"], label="Выбрать тип модели", value="SD"),
        gr.Dropdown(choices=["euler_ancestral", "euler", "lms", "heun", "dpm", "dpm_solver", "dpm_solver++"],
                    label="Выбрать Sampler", value="euler_ancestral"),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Шаги"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label="CFG"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Ширина"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Высота"),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Пропуск клипа"),
        gr.Checkbox(label="Включить Upscale", value=False),
        gr.Radio(choices=["x2", "x4"], label="Размер Upscale", value="x2"),
    ],
    outputs=[
        gr.Image(type="filepath", label="Сгенерированное изображение"),
        gr.Textbox(label="Сообщение", type="text"),
    ],
    title="НейроЧатWebUI (АЛЬФА) - Stable Diffusion (txt2img)",
    description="Этот пользовательский интерфейс позволяет вводить любой текст и генерировать изображения с помощью Stable Diffusion. "
                "Вы можете выбрать модель Stable Diffusion и настроить параметры генерации с помощью ползунков. "
                "Попробуйте и посмотрите, что получится!",
    allow_flagging="never",
)

img2img_interface = gr.Interface(
    fn=generate_image_img2img,
    inputs=[
        gr.Textbox(label="Ввести ваш запрос"),
        gr.Textbox(label="Ввести ваш негативный запрос", value=""),
        gr.Image(label="Исходное изображение", type="filepath"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="Сила"),
        gr.Dropdown(choices=stable_diffusion_models_list, label="Выбрать модель Stable Diffusion", value=None),
        gr.Dropdown(choices=vae_models_list, label="Выбрать модель VAE (опционально)", value=None),
        gr.Radio(choices=["SD", "SDXL"], label="Выбрать тип модели", value="SD"),
        gr.Dropdown(choices=["euler_ancestral", "euler", "lms", "heun", "dpm", "dpm_solver", "dpm_solver++"],
                    label="Выбрать Sampler", value="euler_ancestral"),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Шаги"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label="CFG"),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Пропуск клипа"),
    ],
    outputs=[
        gr.Image(type="filepath", label="Сгенерированное изображение"),
        gr.Textbox(label="Сообщение", type="text"),
    ],
    title="НейроЧатWebUI (АЛЬФА) - Stable Diffusion (img2img)",
    description="Этот пользовательский интерфейс позволяет вам вводить любой текст и изображение для создания новых изображений с помощью Stable Diffusion. "
                "Вы можете выбрать модель Stable Diffusion и настроить параметры генерации с помощью ползунков. "
                "Попробуйте и посмотрите, что получится!",
    allow_flagging="never",
)

extras_interface = gr.Interface(
    fn=upscale_image,
    inputs=[
        gr.Image(label="Изображение для upscale", type="filepath"),
    ],
    outputs=[
        gr.Image(type="filepath", label="Увеличенное изображение"),
        gr.Textbox(label="Сообщение", type="text"),
    ],
    title="НейроЧатWebUI (АЛЬФА) - Stable Diffusion (Дополнительно)",
    description="Этот пользовательский интерфейс позволяет загружать изображение и выполнять масштабирование в 2 раза.",
    allow_flagging="never",
)

audiocraft_interface = gr.Interface(
    fn=generate_audio,
    inputs=[
        gr.Textbox(label="Введите ваш запрос"),
        gr.Audio(type="filepath", label="Аудио мелодии (опционально)", interactive=True),
        gr.Dropdown(choices=audiocraft_models_list, label="Выбрать модель AudioCraft", value=None),
        gr.Radio(choices=["musicgen", "audiogen"], label="Выбрать тип модели", value="musicgen"),
        gr.Slider(minimum=1, maximum=120, value=10, step=1, label="Длительность (секунды)"),
        gr.Slider(minimum=1, maximum=1000, value=250, step=1, label="Top K"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="Top P"),
        gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Температура"),
        gr.Slider(minimum=1.0, maximum=10.0, value=4.0, step=0.1, label="CFG"),
        gr.Checkbox(label="Включить Multiband Diffusion", value=False),
    ],
    outputs=[
        gr.Audio(label="Сгенерированное аудио", type="filepath"),
        gr.Textbox(label="Сообщение", type="text"),
    ],
    title="НейроЧатWebUI (АЛЬФА) - AudioCraft",
    description="Этот пользовательский интерфейс позволяет вам вводить любой текст и генерировать аудио с помощью AudioCraft. "
                "Вы можете выбрать модель AudioCraft и настроить параметры генерации с помощью ползунков. "
                "Попробуйте и посмотрите, что получится!",
    allow_flagging="never",
)

with gr.TabbedInterface(
        [chat_interface, gr.TabbedInterface([txt2img_interface, img2img_interface, extras_interface],
                                            tab_names=["txt2img", "img2img", "Дополнительно"]),
         audiocraft_interface],
        tab_names=["LLM", "Stable Diffusion", "AudioCraft"]
) as app:
    stop_button = gr.Button(value="Stop generation", interactive=True)
    stop_button.click(stop_all_processes, [], [], queue=False)

    app.launch()
