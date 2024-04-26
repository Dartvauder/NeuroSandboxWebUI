## [Features](/#Features) | [Dependencies](/#Required-Dependencies) | [SystemRequirements](/#Minimum-System-Requirements) | [Install](/#How-to-install) | [Usage](/#How-to-use) | [Models](/#Where-can-I-get-models-voices-and-avatars) | [Roadmap](/#Roadmap) | [Acknowledgment](/#Acknowledgment-to-developers)

# ![main](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4ea0d891-8979-45ad-b052-626c41ae991a)
* Work in progress! (ALPHA)
* English | [Русский](/README_RU.md)

## Description:

Simple and easy interface for use of different neural network models. You can chat with LLM using text or voice input, use StableDiffusion for generating images and videos, and also AudioCraft for generating audio

The goal of the project - to create the easiest possible application to use neural network models

### LLM: ![1](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4063f1fb-e3fe-4580-9d26-cd4ea34dcc41)

### StableDiffusion: 
 #### txt2img: ![2](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/c91d2fa4-d8dc-4fc7-b881-8e369346e15f)
 #### img2img: ![3](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/61dc653f-1ccd-4d50-897c-c9fcc49b59f5)
 #### inpaint: ![4](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/b331ade5-d0ba-46dc-9d8e-352ea9a33631)
 #### video: ![5](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/07bba0b0-4f40-44d5-8446-5cf856921cd4)
 #### extras: ![6](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/e5103afb-aa22-4468-a2b4-fbd2f8fed280)

### AudioCraft: ![7](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/2be4caa2-ebb8-45c8-9b88-a4c7d5d1bdfe)

### Settings: ![8](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/52b0b01e-e6cf-49b0-8680-836ed30cf780)

## Features:

* Easy installation via install.bat (Windows only)
* You can use the application via your mobile device in localhost (Via IPv4)
* Flexible and optimized interface (By Gradio)
* Support for Transformers and llama.cpp models (LLM)
* Support for diffusers (safetensors) models (StableDiffusion) - txt2img, img2img, inpaint, video and extras tabs
* AudioCraft support (Models: musicgen, audiogen and magnet)
* Supports TTS and Whisper models (For LLM)
* Support for Lora, Vae, Inpaint, Upscale and Video models (For Stable Diffusion)
* Support Multiband Diffusion model (For AudioCraft)
* Ability to select an avatar (For LLM)
* Model settings inside the interface
* Application settings

## Required Dependencies:

* [Python](https://www.python.org/downloads/) (3.10 minimum)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (12.1 only)
* [FFMPEG](https://ffmpeg.org/download.html)

## Minimum System Requirements:

* System: Windows or Linux
* GPU: 6GB+ or CPU: 8 core 3.2GHZ
* RAM: 16GB+
* Disk space: 20GB+
* Internet for downloading models and installing

## How to install:

### Windows

1) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` to any location
2) Run the `install.bat` and wait for installation
3) After installation, run `start.bat`
4) Select the file version and wait for the application to launch
5) Now you can start generating!

To get update, run `update.bat`

### Linux

1) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` to any location
2) In the terminal, run the `pip install --no-deps -r requirements.txt`and wait for installation of all dependencies
3) After installation, run `py appEN.py` or `py appRU.py`
4) Wait for the application to launch
5) Now you can start generating!

To get update, run `git pull`

## How to use:

#### Interface has four tabs: LLM, StableDiffusion, AudioCraft and Settings. Select the one you need and follow the instructions below 

### LLM:

1) First upload your models to the folder: *inputs/text/llm_models*
2) Select your model from the drop-down list
3) Select model type (`transformers` or `llama`)
4) Set up the model according to the parameters you need
5) Type (or speak) your request
6) Click the `Submit` button to receive the generated text and audio response
#### Optional: you can enable `TTS` mode, select the `voice` and `language` needed to receive an audio response. You can also select `avatar`
#### Avatars = *inputs/image/avatars*
#### Voice samples = *inputs/audio/voices*
#### The voice must be pre-processed (22050 kHz, mono, WAV), the avatar should preferably be `PNG` or `JPG`

### StableDiffusion - has five sub-tabs:

#### txt2img:

1) First upload your models to the folder: *inputs/image/sd_models*
2) Select your model from the drop-down list
3) Select model type (`SD`, `SD2` or `SDXL`)
4) Set up the model according to the parameters you need
5) Enter your request
6) Click the `Submit` button to get the generated image
#### Optional: You can select your `vae` and `lora` models to improve the generation method, also you can enable `upscale` to increase the size of the generated image 
#### vae = *inputs/image/sd_models/vae*
#### lora = *inputs/image/sd_models/lora*

#### img2img:

1) First upload your models to the folder: *inputs/image/sd_models*
2) Select your model from the drop-down list
3) Select model type (`SD`, `SD2` or `SDXL`)
4) Set up the model according to the parameters you need
5) Upload the initial image with which the generation will take place
6) Enter your request
7) Click the `Submit` button to get the generated image
#### Optional: You can select your `vae` model
#### vae = *inputs/image/sd_models/vae*

#### inpaint:

1) First upload your models to the folder: *inputs/image/sd_models/inpaint*
2) Select your model from the drop-down list
3) Select model type (`SD`, `SD2` or `SDXL`)
4) Set up the model according to the parameters you need
5) Upload the image with which the generation will take place to `initial image` and `mask image`
6) In `mask image`, select the brush, then the palette and change the color to `#FFFFFF`
7) Draw a place for generation and enter your request
8) Click the `Submit` button to get the inpainted image
#### Optional: You can select your `vae` model
#### vae = *inputs/image/sd_models/vae*

#### video:

1) Upload the initial image
2) Set up the model according to the parameters you need
3) Click the `Submit` button to get the video from image

#### extras:

1) Upload the initial image
2) Select the options you need
3) Click the `Submit` button to get the modified image

### AudioCraft:

1) Select a model from the drop-down list
2) Select model type (`musicgen` or `audiogen`)
3) Set up the model according to the parameters you need
4) Enter your request
5) (Optional) upload the initial audio if you are using `melody` model 
6) Click the `Submit` button to get the generated audio
#### Optional: You can enable `multiband diffusion` to improve the generated audio

### Settings: 

* Here you can change the application settings. For now you can only change `Share` mode to `True` or `False`

### Additional Information:

1) Chat history, generated images and generated audio, are saved in the *outputs* folder
2) You can press the `Clear` button to reset your selection
3) To stop the generation process, click the `Stop generation` button
4) You can turn off the application using the `Close terminal` button
5) You can open the *outputs* folder by clicking on the `Folder` button


## Where can I get models, voices and avatars?

* LLM models can be taken from [HuggingFace](https://huggingface.co/models)
* StableDiffusion, vae, inpaint and lora models can be taken from [CivitAI](https://civitai.com/models)
* AudioCraft models are downloads automatically in *inputs* folder, when you select a model and press the submit button
* TTS, Whisper, Upscale and Multiband diffusion models are downloads automatically in *inputs* folder when are they used 
* You can take voices anywhere. Record yours or take a recording from the Internet. The main thing is that it is pre-processed!
* It’s the same with avatars as with voices. You can download them on the Internet, generate them using neural networks, or take a photo of yourself. The main thing is to comply with the required file format

## Roadmap

* https://github.com/Dartvauder/NeuroSandboxWebUI/wiki/RoadmapEN

## Acknowledgment to developers

Thank you very much to these projects for allowing me to create my application:

* `Gradio` - https://github.com/gradio-app/gradio
* `Transformers` - https://github.com/huggingface/transformers
* `TTS` - https://github.com/coqui-ai/TTS
* `openai-Whisper` - https://github.com/openai/whisper
* `torch` - https://github.com/pytorch/pytorch
* `Soundfile` - https://github.com/bastibe/python-soundfile
* `accelerate` - https://github.com/huggingface/accelerate
* `cuda-python` - https://github.com/NVIDIA/cuda-python
* `GitPython` - https://github.com/gitpython-developers/GitPython
* `diffusers` - https://github.com/huggingface/diffusers
* `llama.cpp-python` - https://github.com/abetlen/llama-cpp-python
* `AudioCraft` - https://github.com/facebookresearch/audiocraft
* `xformers` - https://github.com/facebookresearch/xformers

## Donation

### *If you liked my project and want to donate, here is options to donate. Thank you very much in advance!*

* CryptoWallet(BEP-20) - 0x3d86bdb5f50b92d0d7Eb44F1a833acC5e91aAEcA

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)

