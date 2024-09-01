## [Features](/#Features) | [Dependencies](/#Required-Dependencies) | [SystemRequirements](/#Minimum-System-Requirements) | [Install](/#How-to-install) | [Usage](/#How-to-use) | [Models](/#Where-can-I-get-models-voices-and-avatars) | [Wiki](/#Wiki) | [Acknowledgment](/#Acknowledgment-to-developers) | [Licenses](/#Third-Party-Licenses)

# ![main](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4ea0d891-8979-45ad-b052-626c41ae991a)
* Work in progress! (ALPHA)
* English | [عربي](/Readmes/README_AR.md) | [Deutsche](/Readmes/README_DE.md) | [Español](/Readmes/README_ES.md) | [Français](/Readmes/README_FR.md) | [日本語](/Readmes/README_JP.md) | [Русский](/Readmes/README_RU.md) | [漢語](/Readmes/README_ZH.md) | [Português](/Readmes/README_PT.md) | [हिंदी](/Readmes/README_HI.md) | [Italiano](/Readmes/README_IT.md) | [韓國語](/Readmes/README_KO.md) | [Polski](/Readmes/README_PL.md) | [Türkçe](/Readmes/README_TR.md)

## Description:

A simple and convenient interface for using various neural network models. You can communicate with LLM and Moondream2 using text, voice and image input; use StableDiffusion, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF, PixArt and PlaygroundV2.5, to generate images; ModelScope, ZeroScope 2, CogVideoX and Latte to generate videos; StableFast3D, Shap-E, SV34D and Zero123Plus to generate 3D objects; StableAudioOpen, AudioCraft and AudioLDM 2 to generate music and audio; CoquiTTS, MMS and SunoBark for text-to-speech; OpenAI-Whisper and MMS for speech-to-text; Wav2Lip for lip-sync; Roop to faceswap; Rembg to remove background; CodeFormer for face restore; LibreTranslate and SeamlessM4Tv2 for text translation; Demucs for audio file separation. You can also view files from the outputs directory in gallery, download the LLM and StableDiffusion models, change the application settings inside the interface and check system sensors

The goal of the project - to create the easiest possible application to use neural network models

### Text: <img width="1118" alt="1" src="https://github.com/user-attachments/assets/e22b37bb-0d93-4308-8f6a-1b28745cc602">

### Image: <img width="1117" alt="2" src="https://github.com/user-attachments/assets/f2704fca-cee3-4967-ad4b-d104ccd4f708">

### Video: <img width="1120" alt="3" src="https://github.com/user-attachments/assets/8a5ad71d-8193-4415-9e74-6daa4ddf1d8a">

### 3D: <img width="1115" alt="4" src="https://github.com/user-attachments/assets/90cb1a5a-7714-444f-915d-9a619f0f8657">

### Audio: <img width="1115" alt="5" src="https://github.com/user-attachments/assets/c84c82ff-316b-4f52-8d2e-d407273dfe0c">

### Interface: <img width="1115" alt="6" src="https://github.com/user-attachments/assets/741f617f-8f28-4ff6-8781-2391e1251e4a">

## Features:

* Easy installation via install.bat (Windows) or install.sh (Linux)
* You can use the application via your mobile device in localhost (Via IPv4) or anywhere online (Via Share)
* Flexible and optimized interface (By Gradio)
* Authentication via admin:admin (You can enter your login details in the GradioAuth.txt file)
* You can add your own HuggingFace-Token to download a specific models (You can enter your token in the HF-Token.txt file)
* Debug logging to logs from `Install` and `Update` files
* Support for Transformers and llama.cpp models (LLM)
* Support for diffusers and safetensors models (StableDiffusion) - txt2img, img2img, depth2img, pix2pix, controlnet, upscale (latent), upscale (Real-ESRGAN), refiner, inpaint, outpaint, gligen, animatediff, video, ldm3d, sd3, cascade, t2i-ip-adapter, ip-adapter-faceid and extras tabs
* Support of additional models for image generation: Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF, PixArt and PlaygroundV2.5
* StableAudioOpen support
* AudioCraft support (Models: musicgen, audiogen and magnet)
* AudioLDM 2 support (Models: audio and music)
* Supports TTS and Whisper models (For LLM and TTS-STT)
* Support MMS for text-to-speech and speech-to-text
* Supports Lora, Textual inversion (embedding), Vae, Img2img, Depth, Pix2Pix, Controlnet, Upscalers (latent and Real-ESRGAN), Refiner, Inpaint, Outpaint, GLIGEN, AnimateDiff, Videos, LDM3D, SD3, Cascade, T2I-IP-ADAPTER, IP-Adapter-FaceID, Rembg, CodeFormer and Roop models (For StableDiffusion)
* Support Multiband Diffusion model (For AudioCraft)
* Support LibreTranslate (Local API) and SeamlessM4Tv2 for language translations
* Support ModelScope, ZeroScope 2, CogVideoX and Latte for video generation
* Support SunoBark
* Support Demucs
* Support StableFast3D, Shap-E, SV34D and Zero123Plus for 3D generation
* Support Wav2Lip
* Support Multimodal (Moondream 2), LORA and WebSearch (with DuckDuckGo) for LLM
* Model settings inside the interface
* Gallery
* ModelDownloader (For LLM and StableDiffusion)
* Application settings
* Ability to see system sensors

## Required Dependencies:

* [Python](https://www.python.org/downloads/) (3.11)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (12.4) and [cuDNN](https://developer.nvidia.com/cudnn-downloads) (9.1)
* [FFMPEG](https://ffmpeg.org/download.html)
- C+ compiler
  - Windows: [VisualStudio](https://visualstudio.microsoft.com/ru/)
  - Linux: [GCC](https://gcc.gnu.org/)

## Minimum System Requirements:

* System: Windows or Linux
* GPU: 6GB+ or CPU: 8 core 3.2GHZ
* RAM: 16GB+
* Disk space: 20GB+
* Internet for downloading models and installing

## How to install:

### Windows

1) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` to any location
2) Run the `Install.bat` and wait for installation
3) After installation, run `Start.bat`
4) Select the file version and wait for the application to launch
5) Now you can start generating!

To get update, run `Update.bat`
To work with the virtual environment through the terminal, run `Venv.bat`

### Linux

1) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` to any location
2) In the terminal, run the `./Install.sh` and wait for installation of all dependencies
3) After installation, run `./Start.sh`
4) Wait for the application to launch
5) Now you can start generating!

To get update, run `./Update.sh`
To work with the virtual environment through the terminal, run `./Venv.sh`

## How to use:

#### Interface has thirty four tabs in six main tabs (Text, Image, Video, 3D, Audio and Interface): LLM, TTS-STT, MMS, SeamlessM4Tv2, SunoBark, LibreTranslate, Wav2Lip, StableDiffusion, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF, PixArt, PlaygroundV2.5, ModelScope, ZeroScope 2, CogVideoX, Latte, StableFast3D, Shap-E, SV34D, Zero123Plus, StableAudio, AudioCraft, AudioLDM 2, Demucs, Gallery, ModelDownloader, Settings and System. Select the one you need and follow the instructions below 

### LLM:

1) First upload your models to the folder: *inputs/text/llm_models*
2) Select your model from the drop-down list
3) Select model type (`transformers` or `llama`)
4) Set up the model according to the parameters you need
5) Type (or speak) your request
6) Click the `Submit` button to receive the generated text and audio response
#### Optional: you can enable `TTS` mode, select the `voice` and `language` needed to receive an audio response. You can enable `multimodal` and upload an image to get its description. You can enable `websearch` for Internet access. You can enable `libretranslate` to get the translate. Also you can choose `LORA` model to improve generation
#### Voice samples = *inputs/audio/voices*
#### LORA = *inputs/text/llm_models/lora*
#### The voice must be pre-processed (22050 kHz, mono, WAV)
#### Avatars of LLM, you change in *avatars* folder

### TTS-STT:

1) Type text for text to speech
2) Input audio for speech to text
3) Click the `Submit` button to receive the generated text and audio response
#### Voice samples = *inputs/audio/voices*
#### The voice must be pre-processed (22050 kHz, mono, WAV)

### MMS (text-to-speech and speech-to-text):

1) Type text for text to speech
2) Input audio for speech to text
3) Click the `Submit` button to receive the generated text or audio response

### SeamlessM4Tv2:

1) Type (or speak) your request
2) Select source and target languages
3) Set up the model according to the parameters you need
4) Click the `Submit` button to get the translate

### SunoBark:

1) Type your request
2) Set up the model according to the parameters you need
3) Click the `Submit` button to receive the generated audio response

### LibreTranslate:

* First you need to install and run [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate)
1) Select source and target languages
2) Click the `Submit` button to get the translate
#### Optional: you can save the translation history by turning on the corresponding button

### Wav2Lip:

1) Upload the initial image of face
2) Upload the initial audio of voice
3) Set up the model according to the parameters you need
4) Click the `Submit` button to receive the lip-sync

### StableDiffusion - has twenty sub-tabs:

#### txt2img:

1) First upload your models to the folder: *inputs/image/sd_models*
2) Select your model from the drop-down list
3) Select model type (`SD`, `SD2` or `SDXL`)
4) Set up the model according to the parameters you need
5) Enter your request (+ and - for prompt weighting)
6) Click the `Submit` button to get the generated image
#### Optional: You can select your `vae`, `embedding` and `lora` models to improve the generation method
#### vae = *inputs/image/sd_models/vae*
#### lora = *inputs/image/sd_models/lora*
#### embedding = *inputs/image/sd_models/embedding*

#### img2img:

1) First upload your models to the folder: *inputs/image/sd_models*
2) Select your model from the drop-down list
3) Select model type (`SD`, `SD2` or `SDXL`)
4) Set up the model according to the parameters you need
5) Upload the initial image with which the generation will take place
6) Enter your request (+ and - for prompt weighting)
7) Click the `Submit` button to get the generated image
#### Optional: You can select your `vae` model to improve the generation method
#### vae = *inputs/image/sd_models/vae*

#### depth2img:

1) Upload the initial image
2) Set up the model according to the parameters you need
3) Enter your request (+ and - for prompt weighting)
4) Click the `Submit` button to get the generated image

#### pix2pix:

1) Upload the initial image
2) Set up the model according to the parameters you need
3) Enter your request (+ and - for prompt weighting)
4) Click the `Submit` button to get the generated image

#### controlnet:

1) First upload your stable diffusion models to the folder: *inputs/image/sd_models*
2) Upload the initial image
3) Select your stable diffusion and controlnet models from the drop-down lists
4) Set up the models according to the parameters you need
5) Enter your request (+ and - for prompt weighting)
6) Click the `Submit` button to get the generated image

#### upscale (latent):

1) Upload the initial image
2) Set up the model according to the parameters you need
3) Click the `Submit` button to get the upscaled image

#### upscale (Real-ESRGAN):

1) Upload the initial image
2) Set up the model according to the parameters you need
3) Click the `Submit` button to get the upscaled image

#### refiner (SDXL):

1) Upload the initial image
2) Click the `Submit` button to get the refined image

#### inpaint:

1) First upload your models to the folder: *inputs/image/sd_models/inpaint*
2) Select your model from the drop-down list
3) Select model type (`SD`, `SD2` or `SDXL`)
4) Set up the model according to the parameters you need
5) Upload the image with which the generation will take place to `initial image` and `mask image`
6) In `mask image`, select the brush, then the palette and change the color to `#FFFFFF`
7) Draw a place for generation and enter your request (+ and - for prompt weighting)
8) Click the `Submit` button to get the inpainted image
#### Optional: You can select your `vae` model to improve the generation method
#### vae = *inputs/image/sd_models/vae*

#### outpaint:

1) First upload your models to the folder: *inputs/image/sd_models/inpaint*
2) Select your model from the drop-down list
3) Select model type (`SD`, `SD2` or `SDXL`)
4) Set up the model according to the parameters you need
5) Upload the image with which the generation will take place to `initial image`
6) Enter your request (+ and - for prompt weighting)
7) Click the `Submit` button to get the outpainted image

#### gligen:

1) First upload your models to the folder: *inputs/image/sd_models*
2) Select your model from the drop-down list
3) Select model type (`SD`, `SD2` or `SDXL`)
4) Set up the model according to the parameters you need
5) Enter your request for prompt (+ and - for prompt weighting) and GLIGEN phrases (in "" for box)
6) Enter GLIGEN boxes (Like a [0.1387, 0.2051, 0.4277, 0.7090] for box)
7) Click the `Submit` button to get the generated image

#### animatediff:

1) First upload your models to the folder: *inputs/image/sd_models*
2) Select your model from the drop-down list
3) Set up the model according to the parameters you need
4) Enter your request (+ and - for prompt weighting)
5) Click the `Submit` button to get the generated image animation
#### Optional: you can select a motion LORA to control your generation

### hotshot-xl

1) Enter your request
2) Set up the model according to the parameters you need
3) Click the `Submit` button to get the generated GIF-image

#### video:

1) Upload the initial image
2) Select your model
3) Enter your request (for IV2Gen-XL)
4) Set up the model according to the parameters you need
5) Click the `Submit` button to get the video from image

#### ldm3d:

1) Enter your request
2) Set up the model according to the parameters you need
3) Click the `Submit` button to get the generated images

#### sd3 (txt2img, img2img, controlnet, inpaint):

1) Enter your request
2) Set up the model according to the parameters you need
3) Click the `Submit` button to get the generated image

#### cascade:

1) Enter your request
2) Set up the model according to the parameters you need
3) Click the `Submit` button to get the generated image

#### t2i-ip-adapter:

1) Upload the initial image
2) Select the options you need
3) Click the `Submit` button to get the modified image

#### ip-adapter-faceid:

1) Upload the initial image
2) Select the options you need
3) Click the `Submit` button to get the modified image

#### extras:

1) Upload the initial image
2) Select the options you need
3) Click the `Submit` button to get the modified image

### Kandinsky (txt2img, img2img, inpaint):

1) Enter your prompt
2) Select a model from the drop-down list
3) Set up the model according to the parameters you need
4) Click `Submit` to get the generated image

### Flux:

1) Enter your prompt
2) Select your model
3) Set up the model according to the parameters you need
4) Click `Submit` to get the generated image

### HunyuanDiT:

1) Enter your prompt
2) Set up the model according to the parameters you need
3) Click `Submit` to get the generated image

### Lumina-T2X:

1) Enter your prompt
2) Set up the model according to the parameters you need
3) Click `Submit` to get the generated image

### Kolors:

1) Enter your prompt
2) Set up the model according to the parameters you need
3) Click `Submit` to get the generated image

### AuraFlow:

1) Enter your prompt
2) Set up the model according to the parameters you need
3) Click `Submit` to get the generated image

### Würstchen:

1) Enter your prompt
2) Set up the model according to the parameters you need
3) Click `Submit` to get the generated image

### DeepFloydIF (txt2img, img2img, inpaint):

1) Enter your prompt
2) Set up the model according to the parameters you need
3) Click `Submit` to get the generated image

### PixArt:

1) Enter your prompt
2) Select your model
3) Set up the model according to the parameters you need
4) Click `Submit` to get the generated image

### PlaygroundV2.5:

1) Enter your prompt
2) Set up the model according to the parameters you need
3) Click `Submit` to get the generated image

### ModelScope:

1) Enter your prompt
2) Set up the model according to the parameters you need
3) Click `Submit` to get the generated video

### ZeroScope 2:

1) Enter your prompt
2) Set up the model according to the parameters you need
3) Click `Submit` to get the generated video

### CogVideoX:

1) Enter your prompt
2) Set up the model according to the parameters you need
3) Click `Submit` to get the generated video

### Latte:

1) Enter your prompt
2) Set up the model according to the parameters you need
3) Click `Submit` to get the generated video

### StableFast3D:

1) Upload the initial image
2) Set up the model according to the parameters you need
3) Click the `Submit` button to get the generated 3D object

### Shap-E:

1) Enter your request or upload the initial image
2) Set up the model according to the parameters you need
3) Click the `Submit` button to get the generated 3D object

### SV34D:

1) Upload the initial image (for 3D) or video (for 4D)
2) Set up the model according to the parameters you need
3) Click the `Submit` button to get the generated 3D video

### Zero123Plus:

1) Upload the initial image
2) Set up the model according to the parameters you need
3) Click the `Submit` button to get the generated 3D rotation of image

### StableAudio:

1) Set up the model according to the parameters you need
2) Enter your request
3) Click the `Submit` button to get the generated audio

### AudioCraft:

1) Select a model from the drop-down list
2) Select model type (`musicgen`, `audiogen` or `magnet`)
3) Set up the model according to the parameters you need
4) Enter your request
5) (Optional) upload the initial audio if you are using `melody` model 
6) Click the `Submit` button to get the generated audio
#### Optional: You can enable `multiband diffusion` to improve the generated audio

### AudioLDM 2:

1) Select a model from the drop-down list
2) Set up the model according to the parameters you need
3) Enter your request
4) Click the `Submit` button to get the generated audio

### Demucs:

1) Upload the initial audio to separate
2) Click the `Submit` button to get the separated audio

### Gallery:

* Here you can view files from the outputs directory

### ModelDownloader:

* Here you can download `LLM` and `StableDiffusion` models. Just choose the model from the drop-down list and click the `Submit` button
#### `LLM` models are downloaded here: *inputs/text/llm_models*
#### `StableDiffusion` models are downloaded here: *inputs/image/sd_models*

### Settings: 

* Here you can change the application settings. For now you can only change `Share` mode to `True` or `False`

### System: 

* Here you can see the indicators of your computer's sensors by clicking on the `Submit` button

### Additional Information:

1) All generations are saved in the *outputs* folder
2) You can press the `Clear` button to reset your selection
3) To stop the generation process, press the `Stop generation` button
4) You can turn off the application using the `Close terminal` button
5) You can open the *outputs* folder using the `Outputs` button
6) You can reload your interface using the `Reload models` button

## Where can i get models and voices?

* LLM models can be taken from [HuggingFace](https://huggingface.co/models) or from ModelDownloader inside interface 
* StableDiffusion, vae, inpaint, embedding and lora models can be taken from [CivitAI](https://civitai.com/models) or from ModelDownloader inside interface
* StableAudioOpen, AudioCraft, AudioLDM 2, TTS, Whisper, MMS, SeamlessM4Tv2, Wav2Lip, SunoBark, MoonDream2, Upscalers (Latent and Real-ESRGAN), Refiner, GLIGEN, Depth, Pix2Pix, Controlnet, AnimateDiff, HotShot-XL, Videos, LDM3D, SD3, Cascade, T2I-IP-ADAPTER, IP-Adapter-FaceID, Rembg, Roop, CodeFormer, Real-ESRGAN, StableFast3D, Shap-E, SV34D, Zero123Plus, Demucs, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, AuraSR, Würstchen, DeepFloydIF, PixArt, PlaygroundV2.5, ModelScope, ZeroScope 2, CogVideoX, Latte and Multiband diffusion models are downloads automatically in *inputs* folder when are they used 
* You can take voices anywhere. Record yours or take a recording from the Internet. Or just use those that are already in the project. The main thing is that it is pre-processed!

## Wiki

* https://github.com/Dartvauder/NeuroSandboxWebUI/wiki

## Acknowledgment to developers

#### Many thanks to these projects because thanks to their applications/libraries, i was able to create my application:

First of all, I want to thank the developers of [PyCharm](https://www.jetbrains.com/pycharm/) and [GitHub](https://desktop.github.com). With the help of their applications, i was able to create and share my code

* `gradio` - https://github.com/gradio-app/gradio
* `transformers` - https://github.com/huggingface/transformers
* `tts` - https://github.com/coqui-ai/TTS
* `openai-whisper` - https://github.com/openai/whisper
* `torch` - https://github.com/pytorch/pytorch
* `soundfile` - https://github.com/bastibe/python-soundfile
* `cuda-python` - https://github.com/NVIDIA/cuda-python
* `gitpython` - https://github.com/gitpython-developers/GitPython
* `diffusers` - https://github.com/huggingface/diffusers
* `llama.cpp-python` - https://github.com/abetlen/llama-cpp-python
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

## Third Party Licenses:

#### Many models have their own license for use. Before using it, I advise you to familiarize yourself with them:

* [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE)
* [llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/LICENSE)
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

## Donation

### *If you liked my project and want to donate, here is options to donate. Thank you very much in advance!*

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Dartvauder/NeuroSandboxWebUI&type=Date)](https://star-history.com/#Dartvauder/NeuroSandboxWebUI&Date)
