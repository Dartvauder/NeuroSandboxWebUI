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
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionImg2ImgPipeline, AutoencoderKL
from git import Repo
from PIL import Image
from llama_cpp import Llama
import requests
import torchaudio
from audiocraft.models import MusicGen, AudioGen, MultiBandDiffusion
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
logging.getLogger('xformers').setLevel(logging.ERROR)

chat_dir = None
tts_model = None
whisper_model = None
audiocraft_model_path = None


def load_model(model_name, model_type, n_ctx=None):
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
            model = Llama(model_path, n_ctx=n_ctx)
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
    print("Downloading TTS...")
    tts_model_path = "inputs/audio/XTTS-v2"
    if not os.path.exists(tts_model_path):
        os.makedirs(tts_model_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/coqui/XTTS-v2", tts_model_path)
    print("TTS model downloaded")
    return TTS(model_path=tts_model_path, config_path=f"{tts_model_path}/config.json")


def load_whisper_model():
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
    print(f"AudioCraft model {model_name} downloaded")
    return audiocraft_model_path


def load_multiband_diffusion_model():
    print(f"Downloading Multiband Diffusion model")
    multiband_diffusion_path = os.path.join("inputs", "audio", "audiocraft", "multiband-diffusion")
    if not os.path.exists(multiband_diffusion_path):
        os.makedirs(multiband_diffusion_path, exist_ok=True)
        Repo.clone_from("https://huggingface.co/facebook/multiband-diffusion", multiband_diffusion_path)
        print("Multiband Diffusion model downloaded")
    return multiband_diffusion_path

def load_upscale_model(upscale_factor):
    upscale_model_name = "stabilityai/stable-diffusion-x4-upscaler"
    upscale_model_path = os.path.join("inputs", "image", "sd_models", "upscale", f"{upscale_model_name}.safetensors")
    if not os.path.exists(upscale_model_path):
        os.makedirs(os.path.dirname(upscale_model_path), exist_ok=True)
        Repo.clone_from(f"https://huggingface.co/{upscale_model_name}", os.path.dirname(upscale_model_path))
    print(f"Upscale model {upscale_model_name} downloaded")
    upscaler = StableDiffusionUpscalePipeline.from_single_file(
        upscale_model_path,
        revision="upscale",
        use_safetensors=True,
        device_map="auto",
    )
    upscaler.upscale_factor = upscale_factor
    return upscaler


def generate_text_and_speech(input_text, input_audio, llm_model_name, llm_model_type, max_tokens, n_ctx, temperature,
                             top_p,
                             top_k, avatar_name, enable_tts, speaker_wav, language):
    global chat_dir, tts_model, whisper_model
    if not input_text and not input_audio:
        return "Please, enter your request!", None, None, None, None
    prompt = transcribe_audio(input_audio) if input_audio else input_text
    if not llm_model_name:
        return "Please, select a LLM model!", None, None, None, None
    tokenizer, llm_model = load_model(llm_model_name, llm_model_type,
                                      n_ctx=n_ctx if llm_model_type == "llama" else None)
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


def generate_image_txt2img(prompt, negative_prompt, stable_diffusion_model_name, vae_model_name, stable_diffusion_model_type, stable_diffusion_sampler, stable_diffusion_steps, stable_diffusion_cfg, stable_diffusion_width, stable_diffusion_height, stable_diffusion_clip_skip, enable_upscale=False, upscale_factor=4):
    if not stable_diffusion_model_name:
        return None, "Please, select a Stable Diffusion model!"
    stable_diffusion_model_path = os.path.join("inputs", "image", "sd_models", f"{stable_diffusion_model_name}.safetensors")
    if stable_diffusion_model_type == "SD":
        if os.path.exists(stable_diffusion_model_path):
            stable_diffusion_model = StableDiffusionPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto"
            )
        else:
            print(f"Stable Diffusion model not found: {stable_diffusion_model_path}")
            stable_diffusion_model = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", use_safetensors=True, device_map="auto"
            )
    elif stable_diffusion_model_type == "SDXL":
        if os.path.exists(stable_diffusion_model_path):
            stable_diffusion_model = StableDiffusionXLPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto", attention_slice=1
            )
        else:
            print(f"Stable Diffusion XL model not found: {stable_diffusion_model_path}")
            stable_diffusion_model = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True, device_map="auto", attention_slice=1
            )
    else:
        return None, "Invalid Stable Diffusion model type!"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if XFORMERS_AVAILABLE:
        stable_diffusion_model.unet.enable_xformers_memory_efficient_attention()
    stable_diffusion_model.to(device)
    stable_diffusion_model.text_encoder.to(device)
    stable_diffusion_model.vae.to(device)
    stable_diffusion_model.unet.to(device)
    stable_diffusion_model.safety_checker = None
    if vae_model_name is not None:
        vae_model_path = os.path.join("inputs", "image", "sd_models", "vae", f"{vae_model_name}.safetensors")
        if os.path.exists(vae_model_path):
            vae = AutoencoderKL.from_single_file(vae_model_path, device_map="auto")
            stable_diffusion_model.vae = vae.to(device)
        else:
            print(f"VAE model not found: {vae_model_path}")
    try:
        images = stable_diffusion_model(prompt, negative_prompt=negative_prompt,
                                        num_inference_steps=stable_diffusion_steps,
                                        guidance_scale=stable_diffusion_cfg, height=stable_diffusion_height,
                                        width=stable_diffusion_width, clip_skip=stable_diffusion_clip_skip,
                                        sampler=stable_diffusion_sampler)
        image = images["images"][0]
        if enable_upscale:
            upscaler = load_upscale_model(upscale_factor)
            if upscaler:
                upscaled_image = upscaler(images=image)["images"][0]
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
                           stable_diffusion_width, stable_diffusion_height, stable_diffusion_clip_skip):
    if not stable_diffusion_model_name:
        return None, "Please, select a Stable Diffusion model!"
    if not init_image:
        return None, "Please, upload an initial image!"
    stable_diffusion_model_path = os.path.join("inputs", "image", "sd_models",
                                               f"{stable_diffusion_model_name}.safetensors")
    if stable_diffusion_model_type == "SD":
        if os.path.exists(stable_diffusion_model_path):
            stable_diffusion_model = StableDiffusionImg2ImgPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto"
            )
        else:
            print(f"Stable Diffusion model not found: {stable_diffusion_model_path}")
            stable_diffusion_model = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", use_safetensors=True, device_map="auto"
            )
    elif stable_diffusion_model_type == "SDXL":
        if os.path.exists(stable_diffusion_model_path):
            stable_diffusion_model = StableDiffusionImg2ImgPipeline.from_single_file(
                stable_diffusion_model_path, use_safetensors=True, device_map="auto", attention_slice=1
            )
        else:
            print(f"Stable Diffusion XL model not found: {stable_diffusion_model_path}")
            stable_diffusion_model = StableDiffusionImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True, device_map="auto", attention_slice=1
            )
    else:
        return None, "Invalid Stable Diffusion model type!"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if XFORMERS_AVAILABLE:
        stable_diffusion_model.unet.enable_xformers_memory_efficient_attention()
    stable_diffusion_model.to(device)
    stable_diffusion_model.text_encoder.to(device)
    stable_diffusion_model.vae.to(device)
    stable_diffusion_model.unet.to(device)
    stable_diffusion_model.safety_checker = None
    if vae_model_name is not None:
        vae_model_path = os.path.join("inputs", "image", "sd_models", "vae", f"{vae_model_name}.safetensors")
        if os.path.exists(vae_model_path):
            vae = AutoencoderKL.from_single_file(vae_model_path, device_map=device)
            stable_diffusion_model.vae = vae.to(device)
        else:
            print(f"VAE model not found: {vae_model_path}")
    try:
        init_image = Image.open(init_image).convert("RGB")
        images = stable_diffusion_model(prompt, negative_prompt=negative_prompt,
                                        num_inference_steps=stable_diffusion_steps,
                                        guidance_scale=stable_diffusion_cfg, height=stable_diffusion_height,
                                        width=stable_diffusion_width, clip_skip=stable_diffusion_clip_skip,
                                        sampler=stable_diffusion_sampler, init_image=init_image, strength=strength)
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


def generate_audio(prompt, input_audio=None, model_name=None, model_type="musicgen", duration=10, top_k=250, top_p=0.0,
                   temperature=1.0, cfg_coef=4.0, enable_multiband=False):
    global audiocraft_model_path

    if not model_name:
        return None, "Please, select an AudioCraft model!"

    if not audiocraft_model_path:
        audiocraft_model_path = load_audiocraft_model(model_name)

    if model_type == "musicgen":
        model = MusicGen.get_pretrained(audiocraft_model_path)
        model.set_generation_params(duration=duration)
    elif model_type == "audiogen":
        model = AudioGen.get_pretrained(audiocraft_model_path)
        model.set_generation_params(duration=duration)
    else:
        return None, "Invalid model type!"

    multiband_diffusion_model = None
    if enable_multiband:
        multiband_diffusion_path = load_multiband_diffusion_model()
        multiband_diffusion_model = MultiBandDiffusion.get_pretrained(multiband_diffusion_path)

    audio_paths = []

    try:
        if input_audio and model_type == "musicgen":
            audio_path = input_audio
            melody, sr = torchaudio.load(audio_path)
            model.set_generation_params(duration=duration, top_k=top_k, top_p=top_p, temperature=temperature,
                                        cfg_coef=cfg_coef)
            wav = model.generate_with_chroma([prompt], melody[None].expand(1, -1, -1), sr)
        else:
            descriptions = [prompt]
            model.set_generation_params(duration=duration, top_k=top_k, top_p=top_p, temperature=temperature,
                                        cfg_coef=cfg_coef)
            wav = model.generate(descriptions)

        if multiband_diffusion_model:
            wav = multiband_diffusion_model.enhance(wav)

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
            return None, "Failed to generate audio"

    finally:
        del model
        if multiband_diffusion_model:
            del multiband_diffusion_model
        torch.cuda.empty_cache()


llm_models_list = [None] + [model for model in os.listdir("inputs/text/llm_models") if not model.endswith(".txt")]
avatars_list = [None] + [avatar for avatar in os.listdir("inputs/image/avatars") if not avatar.endswith(".txt")]
speaker_wavs_list = [None] + [wav for wav in os.listdir("inputs/audio/voices") if not wav.endswith(".txt")]
stable_diffusion_models_list = [None] + [model.replace(".safetensors", "") for model in
                                         os.listdir("inputs/image/sd_models")
                                         if (model.endswith(".safetensors") or not model.endswith(".txt"))]
audiocraft_models_list = [None] + ["musicgen-stereo-medium", "audiogen-medium", "musicgen-stereo-melody"]
vae_models_list = [None] + [model.replace(".safetensors", "") for model in os.listdir("inputs/image/sd_models/vae") if
                            model.endswith(".safetensors")]

chat_interface = gr.Interface(
    fn=generate_text_and_speech,
    inputs=[
        gr.Textbox(label="Enter your request"),
        gr.Audio(type="filepath", label="Record your request"),
        gr.Dropdown(choices=llm_models_list, label="Select LLM Model", value=None),
        gr.Dropdown(choices=["transformers", "llama"], label="Select Model Type", value="transformers"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max Tokens"),
        gr.Slider(minimum=0, maximum=4096, value=2048, step=1, label="n_ctx (for llama models only)", interactive=True),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.1, label="Top P"),
        gr.Slider(minimum=0, maximum=100, value=30, step=1, label="Top K"),
        gr.Dropdown(choices=avatars_list, label="Select Avatar", value=None),
        gr.Checkbox(label="Enable TTS", value=False),
        gr.Dropdown(choices=speaker_wavs_list, label="Select Voice", interactive=True),
        gr.Dropdown(choices=["en", "ru"], label="Select Language", interactive=True),
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
    allow_flagging="never"
)

txt2img_interface = gr.Interface(
    fn=generate_image_txt2img,
    inputs=[
        gr.Textbox(label="Enter your prompt"),
        gr.Textbox(label="Enter your negative prompt", value=""),
        gr.Dropdown(choices=stable_diffusion_models_list, label="Select Stable Diffusion Model", value=None),
        gr.Dropdown(choices=vae_models_list, label="Select VAE Model", value=None),
        gr.Dropdown(choices=["SD", "SDXL"], label="Select Model Type", value="SD"),
        gr.Dropdown(choices=["euler_ancestral", "euler", "lms", "heun", "dpm", "dpm_solver", "dpm_solver++"],
                    label="Select Sampler", value="euler_ancestral"),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Steps"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label="CFG"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height"),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Clip Skip"),
    ],
    outputs=[
        gr.Image(type="filepath", label="Generated Image"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroChatWebUI (ALPHA) - Stable Diffusion (txt2img)",
    description="This user interface allows you to enter any text and generate images using Stable Diffusion. "
                "You can select the Stable Diffusion model and customize the generation settings from the sliders. "
                "Try it and see what happens!",
    allow_flagging="never"
)

img2img_interface = gr.Interface(
    fn=generate_image_img2img,
    inputs=[
        gr.Textbox(label="Enter your prompt"),
        gr.Textbox(label="Enter your negative prompt", value=""),
        gr.Image(label="Initial Image", type="filepath"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="Strength"),
        gr.Dropdown(choices=stable_diffusion_models_list, label="Select Stable Diffusion Model", value=None),
        gr.Dropdown(choices=vae_models_list, label="Select VAE Model", value=None),
        gr.Dropdown(choices=["SD", "SDXL"], label="Select Model Type", value="SD"),
        gr.Dropdown(choices=["euler_ancestral", "euler", "lms", "heun", "dpm", "dpm_solver", "dpm_solver++"],
                    label="Select Sampler", value="euler_ancestral"),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Steps"),
        gr.Slider(minimum=1.0, maximum=30.0, value=8, step=0.1, label="CFG"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width"),
        gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height"),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Clip Skip"),
    ],
    outputs=[
        gr.Image(type="filepath", label="Generated Image"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroChatWebUI (ALPHA) - Stable Diffusion (img2img)",
    description="This user interface allows you to enter any text and image to generate new images using Stable Diffusion. "
                "You can select the Stable Diffusion model and customize the generation settings from the sliders. "
                "Try it and see what happens!",
    allow_flagging="never"
)

audiocraft_interface = gr.Interface(
    fn=generate_audio,
    inputs=[
        gr.Textbox(label="Enter your prompt"),
        gr.Audio(type="filepath", label="Melody audio (optional)", interactive=True),
        gr.Dropdown(choices=audiocraft_models_list, label="Select AudioCraft Model", value=None),
        gr.Dropdown(choices=["musicgen", "audiogen"], label="Select Model Type", value="musicgen"),
        gr.Slider(minimum=1, maximum=120, value=10, step=1, label="Duration (seconds)"),
        gr.Slider(minimum=1, maximum=1000, value=250, step=1, label="Top K"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="Top P"),
        gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Temperature"),
        gr.Slider(minimum=1.0, maximum=10.0, value=4.0, step=0.1, label="CFG"),
        gr.Checkbox(label="Enable Multiband Diffusion", value=False),
    ],
    outputs=[
        gr.Audio(label="Generated Audio", type="filepath"),
        gr.Textbox(label="Message", type="text"),
    ],
    title="NeuroChatWebUI (ALPHA) - AudioCraft",
    description="This user interface allows you to enter any text and generate audio using AudioCraft. "
                "You can select the AudioCraft model and customize the generation settings from the sliders. "
                "Try it and see what happens!",
    allow_flagging="never"
)

with gr.TabbedInterface(
        [chat_interface, gr.TabbedInterface([txt2img_interface, img2img_interface], tab_names=["txt2img", "img2img"]),
         audiocraft_interface],
        tab_names=["LLM", "Stable Diffusion", "AudioCraft"]) as app:
    app.launch()
