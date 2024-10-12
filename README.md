## [Features](/#Features) | [Dependencies](/#Required-Dependencies) | [SystemRequirements](/#Minimum-System-Requirements) | [Install](/#How-to-install) | [Wiki](/#Wiki) | [Acknowledgment](/#Acknowledgment-to-developers) | [Licenses](/#Third-Party-Licenses)

# ![main](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4ea0d891-8979-45ad-b052-626c41ae991a)
* Work in progress but stable!
* English | [Русский](/Readmes/README_RU.md) | [漢語](/Readmes/README_ZH.md) 

## Description:

A simple and convenient interface for using various neural network models. You can communicate with LLM using text, voice and image input; use StableDiffusion, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF, PixArt and PlaygroundV2.5, to generate images; ModelScope, ZeroScope 2, CogVideoX and Latte to generate videos; StableFast3D, Shap-E, SV34D and Zero123Plus to generate 3D objects; StableAudioOpen, AudioCraft and AudioLDM 2 to generate music and audio; CoquiTTS, MMS and SunoBark for text-to-speech; OpenAI-Whisper and MMS for speech-to-text; Wav2Lip for lip-sync; LivePortrait for animate an image; Roop to faceswap; Rembg to remove background; CodeFormer for face restore; PixelOE for image pixelization; DDColor for image colorization; LibreTranslate and SeamlessM4Tv2 for text translation; Demucs and UVR for audio file separation; RVC for voice conversion. You can also view files from the outputs directory in gallery, download the LLM and StableDiffusion models, change the application settings inside the interface and check system sensors

The goal of the project - to create the easiest possible application to use neural network models

### Text: <img width="1118" alt="1" src="https://github.com/user-attachments/assets/4eb61c69-4368-40a4-bf91-a37f8126043c">

### Image: <img width="1109" alt="2" src="https://github.com/user-attachments/assets/309a7e7b-a54e-4b5c-837b-1d295d3960db">

### Video: <img width="1117" alt="3" src="https://github.com/user-attachments/assets/5745c785-fede-4e11-aa46-73fd6696d032">

### 3D: <img width="1114" alt="4" src="https://github.com/user-attachments/assets/db10b012-e9b8-40b9-95fa-150200ce0f11">

### Audio: <img width="1109" alt="5" src="https://github.com/user-attachments/assets/fac227ec-eb97-498f-8c67-e5304d1e0496">

### Extras: <img width="1111" alt="6" src="https://github.com/user-attachments/assets/ca3d451f-ca52-47c9-9ee4-e9d1aed0e4cb">

### Interface: <img width="1116" alt="7" src="https://github.com/user-attachments/assets/2adf0118-560f-4040-821f-417fae40645b">

## Features:

* Easy installation via install.bat (Windows) or install.sh (Linux)
* You can use the application via your mobile device in localhost (Via IPv4) or anywhere online (Via Share)
* Flexible and optimized interface (By Gradio)
* Debug logging to logs from `Install` and `Update` files
* Available in three languages
* Support for Transformers, BNB, GPTQ, AWQ, ExLlamaV2 and llama.cpp models (LLM)
* Support for diffusers and safetensors models (StableDiffusion) - txt2img, img2img, depth2img, marigold, pix2pix, controlnet, upscale (latent), upscale (SUPIR), refiner, inpaint, outpaint, gligen, diffedit, blip-diffusion, animatediff, hotshot-xl, video, ldm3d, sd3, cascade, t2i-ip-adapter, ip-adapter-faceid and riffusion tabs
* Support for stable-diffusion-cpp models for FLUX and Stable Diffusion
* Support of additional models for image generation: Kandinsky (txt2img, img2img, inpaint), Flux (txt2img with cpp quantize and LoRA support, img2img, inpaint, controlnet), HunyuanDiT (txt2img, controlnet), Lumina-T2X, Kolors (txt2img with LoRA support, img2img, ip-adapter-plus), AuraFlow (with LoRA and AuraSR support), Würstchen, DeepFloydIF (txt2img, img2img, inpaint), PixArt and PlaygroundV2.5
* Support Extras with Rembg, CodeFormer, PixelOE, DDColor, DownScale, Format changer, FaceSwap (Roop) and Upscale (Real-ESRGAN) models for image, video and audio
* Support StableAudio
* Support AudioCraft (Models: musicgen, audiogen and magnet)
* Support AudioLDM 2 (Models: audio and music)
* Supports TTS and Whisper models (For LLM and TTS-STT)
* Support MMS for text-to-speech and speech-to-text
* Supports Lora, Textual inversion (embedding), Vae, MagicPrompt, Img2img, Depth, Marigold, Pix2Pix, Controlnet, Upscalers (latent and SUPIR), Refiner, Inpaint, Outpaint, GLIGEN, DiffEdit, BLIP-Diffusion, AnimateDiff, HotShot-XL, Videos, LDM3D, SD3, Cascade, T2I-IP-ADAPTER, IP-Adapter-FaceID and Riffusion models (For StableDiffusion)
* Support Multiband Diffusion model (For AudioCraft)
* Support LibreTranslate (Local API) and SeamlessM4Tv2 for language translations
* Support ModelScope, ZeroScope 2, CogVideoX and Latte for video generation
* Support SunoBark
* Support Demucs and UVR for audio file separation
* Support RVC for voice conversion
* Support StableFast3D, Shap-E, SV34D and Zero123Plus for 3D generation
* Support Wav2Lip
* Support LivePortrait for animate an image
* Support Multimodal (Moondream 2, LLaVA-NeXT-Video, Qwen2-Audio), PDF-Parsing (OpenParse), TTS (CoquiTTS), STT (Whisper), LORA and WebSearch (with DuckDuckGo) for LLM
* MetaData-Info viewer for generating image, video and audio
* Model settings inside the interface
* Online and offline Wiki
* Gallery
* ModelDownloader (For LLM and StableDiffusion)
* Application settings
* Ability to see system sensors

## Required Dependencies:

* [Python](https://www.python.org/downloads/) (3.10.11)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (12.4) and [cuDNN](https://developer.nvidia.com/cudnn-downloads) (9.1)
* [FFMPEG](https://ffmpeg.org/download.html)
- C+ compiler
  - Windows: [VisualStudio](https://visualstudio.microsoft.com/ru/), [VisualStudioCode](https://code.visualstudio.com) and [Cmake](https://cmake.org)
  - Linux: [GCC](https://gcc.gnu.org/), [VisualStudioCode](https://code.visualstudio.com) and [Cmake](https://cmake.org)

## Minimum System Requirements:

* System: Windows or Linux
* GPU: 6GB+ or CPU: 8 core 3.6GHZ
* RAM: 16GB+
* Disk space: 20GB+
* Internet for downloading models and installing

## How to install:

### Windows

1) First install all [RequiredDependencies](/#Required-Dependencies)
2) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` to any location
3) Run the `Install.bat` and wait for installation
4) After installation, run `Start.bat`
5) Wait for the application to launch
6) Now you can start generating!

To get update, run `Update.bat`
To work with the virtual environment through the terminal, run `Venv.bat`

### Linux

1) First install all [RequiredDependencies](/#Required-Dependencies)
2) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` to any location
3) In the terminal, run the `./Install.sh` and wait for installation of all dependencies
4) After installation, run `./Start.sh`
5) Wait for the application to launch
6) Now you can start generating!

To get update, run `./Update.sh`
To work with the virtual environment through the terminal, run `./Venv.sh`

## Wiki

* https://github.com/Dartvauder/NeuroSandboxWebUI/wiki/EN‐Wiki

## Acknowledgment to developers

#### Many thanks to these projects because thanks to their applications/libraries, i was able to create my application:

First of all, I want to thank the developers of [PyCharm](https://www.jetbrains.com/pycharm/) and [GitHub](https://desktop.github.com). With the help of their applications, i was able to create and share my code

* `gradio` - https://github.com/gradio-app/gradio
* `transformers` - https://github.com/huggingface/transformers
* `auto-gptq` - https://github.com/AutoGPTQ/AutoGPTQ
* `autoawq` - https://github.com/casper-hansen/AutoAWQ
* `exllamav2` - https://github.com/turboderp/exllamav2
* `tts` - https://github.com/coqui-ai/TTS
* `openai-whisper` - https://github.com/openai/whisper
* `torch` - https://github.com/pytorch/pytorch
* `soundfile` - https://github.com/bastibe/python-soundfile
* `cuda-python` - https://github.com/NVIDIA/cuda-python
* `gitpython` - https://github.com/gitpython-developers/GitPython
* `diffusers` - https://github.com/huggingface/diffusers
* `llama.cpp-python` - https://github.com/abetlen/llama-cpp-python
* `stable-diffusion-cpp-python` - https://github.com/william-murray1204/stable-diffusion-cpp-python
* `audiocraft` - https://github.com/facebookresearch/audiocraft
* `AudioLDM2` - https://github.com/haoheliu/AudioLDM2
* `xformers` - https://github.com/facebookresearch/xformers
* `demucs` - https://github.com/facebookresearch/demucs
* `libretranslate` - https://github.com/LibreTranslate/LibreTranslate
* `libretranslatepy` - https://github.com/argosopentech/LibreTranslate-py
* `rembg` - https://github.com/danielgatis/rembg
* `trimesh` - https://github.com/mikedh/trimesh
* `suno-bark` - https://github.com/suno-ai/bark
* `IP-Adapter` - https://github.com/tencent-ailab/IP-Adapter
* `PyNanoInstantMeshes` - https://github.com/vork/PyNanoInstantMeshes
* `CLIP` - https://github.com/openai/CLIP
* `rvc-python` - https://github.com/daswer123/rvc-python
* `audio-separator` - https://github.com/nomadkaraoke/python-audio-separator
* `pixeloe` - https://github.com/KohakuBlueleaf/PixelOE
* `k-diffusion` - https://github.com/crowsonkb/k-diffusion
* `open-parse` - https://github.com/Filimoa/open-parse
* `AudioSR` - https://github.com/haoheliu/versatile_audio_super_resolution

## Third Party Licenses:

#### Many models have their own license for use. Before using it, I advise you to familiarize yourself with them:

* [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE)
* [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ/blob/main/LICENSE)
* [AutoAWQ](https://github.com/casper-hansen/AutoAWQ/blob/main/LICENSE)
* [exllamav2](https://github.com/turboderp/exllamav2/blob/master/LICENSE)
* [llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/LICENSE)
* [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp/blob/master/LICENSE)
* [CoquiTTS](https://coqui.ai/cpml)
* [OpenAI-Whisper](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
* [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate/blob/main/LICENSE)
* [Diffusers](https://github.com/huggingface/diffusers/blob/main/LICENSE)
* [StableDiffusion1.5](https://huggingface.co/spaces/CompVis/stable-diffusion-license)
* [StableDiffusion2](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)
* [StableDiffusion3](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/blob/main/LICENSE)
* [StableDiffusionXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)
* [StableCascade](https://huggingface.co/stabilityai/stable-cascade/blob/main/LICENSE)
* [LatentDiffusionModel3D](https://huggingface.co/spaces/CompVis/stable-diffusion-license)
* [StableVideoDiffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/blob/main/LICENSE)
* [I2VGen-XL](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)
* [Rembg](https://github.com/danielgatis/rembg/blob/main/LICENSE.txt)
* [Shap-E](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)
* [StableAudioOpen](https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/LICENSE)
* [AudioCraft](https://spdx.org/licenses/CC-BY-NC-4.0)
* [AudioLDM2](https://spdx.org/licenses/CC-BY-NC-SA-4.0)
* [Demucs](https://github.com/facebookresearch/demucs/blob/main/LICENSE)
* [SunoBark](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)
* [Moondream2](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
* [LLaVA-NeXT-Video](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/LICENSE.txt)
* [Qwen2-Audio](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
* [ZeroScope2](https://spdx.org/licenses/CC-BY-NC-4.0)
* [GLIGEN](https://huggingface.co/spaces/CompVis/stable-diffusion-license)
* [Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
* [Roop](https://github.com/s0md3v/roop/blob/main/LICENSE)
* [CodeFormer](https://github.com/sczhou/CodeFormer/blob/master/LICENSE)
* [ControlNet](https://github.com/lllyasviel/ControlNet/blob/main/LICENSE)
* [AnimateDiff](https://github.com/guoyww/AnimateDiff/blob/main/LICENSE.txt)
* [Pix2Pix](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)
* [Kandinsky 2.1; 2.2; 3](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
* [Flux-schnell](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
* [Flux-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)
* [HunyuanDiT](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/blob/main/LICENSE.txt)
* [Lumina-T2X](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
* [DeepFloydIF](https://huggingface.co/spaces/DeepFloyd/deepfloyd-if-license)
* [PixArt](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)
* [CogVideoX](https://huggingface.co/THUDM/CogVideoX-2b/blob/main/LICENSE)
* [Latte](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
* [Kolors](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
* [AuraFlow](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
* [Würstchen](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)
* [ModelScope](https://spdx.org/licenses/CC-BY-NC-4.0)
* [StableFast3D](https://github.com/Stability-AI/stable-fast-3d/blob/main/LICENSE.md)
* [SV34D](https://huggingface.co/stabilityai/sv4d/blob/main/LICENSE.md)
* [Zero123Plus](https://huggingface.co/blog/open_rail)
* [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE)
* [Refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)
* [PlaygroundV2.5](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic/blob/main/LICENSE.md)
* [AuraSR](https://huggingface.co/fal/AuraSR/blob/main/LICENSE.md)
* [IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID)
* [T2I-IP-Adapter](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
* [MMS](https://spdx.org/licenses/CC-BY-NC-4.0)
* [SeamlessM4Tv2](https://spdx.org/licenses/CC-BY-NC-4.0)
* [HotShot-XL](https://github.com/hotshotco/Hotshot-XL/blob/main/LICENSE)
* [Riffusion](https://huggingface.co/spaces/CompVis/stable-diffusion-license)
* [MozillaCommonVoice17](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/cc0-1.0.md)
* [UVR-MDX](https://github.com/kuielab/mdx-net/blob/main/LICENSE)
* [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
* [DDColor](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
* [PixelOE](https://github.com/KohakuBlueleaf/PixelOE/blob/main/LICENSE)
* [LivePortrait](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)
* [SUPIR](https://github.com/Fanghua-Yu/SUPIR/blob/master/LICENSE)
* [MagicPrompt](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)
* [Marigold](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
* [BLIP-Diffusion](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)

#### These third-party repository codes are also used in my project:

* [Generative-Models for SV34D](https://github.com/Stability-AI/generative-models)
* [CodeFormer for extras](https://github.com/sczhou/CodeFormer)
* [Real-ESRGAN for upscale](https://github.com/xinntao/Real-ESRGAN)
* [HotShot-XL for StableDiffusion](https://github.com/hotshotco/Hotshot-XL)
* [Roop for extras](https://github.com/s0md3v/roop)
* [StableFast3D for 3D](https://github.com/Stability-AI/stable-fast-3d)
* [Riffusion for StableDiffusion](https://github.com/riffusion/riffusion-hobby)
* [DDColor for extras](https://github.com/piddnad/DDColor)
* [LivePortrait for video](https://github.com/KwaiVGI/LivePortrait)
* [SUPIR for StableDiffusion](https://github.com/Fanghua-Yu/SUPIR)

## Donation

### *If you liked my project and want to donate, here is options to donate. Thank you very much in advance!*

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Dartvauder/NeuroSandboxWebUI&type=Date)](https://star-history.com/#Dartvauder/NeuroSandboxWebUI&Date)
