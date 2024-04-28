## [Features](/#Features) | [Dependencies](/#Required-Dependencies) | [SystemRequirements](/#Minimum-System-Requirements) | [Install](/#How-to-install) | [Usage](/#How-to-use) | [Models](/#Where-can-I-get-models-voices-and-avatars) | [Roadmap](/#Roadmap) | [Acknowledgment](/#Acknowledgment-to-developers)

# ![main](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4ea0d891-8979-45ad-b052-626c41ae991a)
* Work in progress! (ALPHA)
* English | [Русский](/README_RU.md)

## Description:

A simple and convenient interface for using various neural network models. You can communicate with LLM using text or voice input, use StableDiffusion to generate images and videos, AudioCraft to generate music and audio, CoquiTTS for text-to-speech, OpenAI-Whisper for speech-to-text, LibreTranslate for text translation and Demucs for audio file separation. You can also download the LLM and StableDiffusion models, change the application settings inside the interface and check system sensors

The goal of the project - to create the easiest possible application to use neural network models

### LLM: ![1](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/e162f89e-b767-42a5-9522-fcaa0f652258)

### TTS-STT: ![2](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/8137b1d1-5f96-41ac-979a-5add6a78fe60)

### LibreTranslate: ![3](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/6de47646-b058-42c2-a2fa-58de06616863)

### StableDiffusion:
 #### txt2img: ![4](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/c4387329-44e8-4372-aba3-44ec2c140c2e)
 #### img2img: ![5](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/df15b0c2-e0e3-4f51-94bb-e1a469db4c76)
 #### depth2img: ![6](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/b275712f-94e5-4147-8f02-ca5e2c2a88d4)
 #### upscale: ![7](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/a8a394ee-c1ab-4248-9ca5-e7f4564ff0a8)
 #### inpaint: ![8](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/71ee7944-f510-4922-a489-20f3ef576179)
 #### video: ![9](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/95586145-f41e-443a-af61-1eb7db79d8bd)
 #### extras: ![10](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/b249593d-42a6-4990-8790-1e00ee4333af)
 
### AudioCraft: ![11](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/48de57c4-e0a9-4a7c-b4b5-d67224b54833)

### Demucs: ![12](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/895f0f1e-f944-4ae2-9fc8-6ebc18db94e8)

### ModelDownloader: ![13](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/8326511d-8671-4a17-8ab8-78f9e14c1d28)

### Settings: ![14](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/7276e470-508d-4d43-bd21-9ebe24b5c0ab)

### System: ![15](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/6f65bbf8-741f-4e22-868c-83b51b0a6ed8)

## Features:

* Easy installation via install.bat (Windows only)
* You can use the application via your mobile device in localhost (Via IPv4)
* Flexible and optimized interface (By Gradio)
* Support for Transformers and llama.cpp models (LLM)
* Support for diffusers and safetensors models (StableDiffusion) - txt2img, img2img, depth2img, upscale, inpaint, video and extras tabs
* AudioCraft support (Models: musicgen, audiogen and magnet)
* Supports TTS and Whisper models (For LLM and TTS-STT)
* Supports Lora, Textual inversion (embedding), Vae, Inpaint, Upscale and Video models (For StableDiffusion)
* Support Multiband Diffusion model (For AudioCraft)
* Support LibreTranslate (Local API)
* Support Demucs
* Support Rembg
* Ability to select an avatar (For LLM)
* Model settings inside the interface
* ModelDownloader (For LLM and StableDiffusion)
* Application settings
* Ability to see system sensors

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

#### Interface has nine tabs: LLM, TTS-STT, LibreTranslate, StableDiffusion, AudioCraft, Demucs, ModelDownloader, Settings and System. Select the one you need and follow the instructions below 

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
#### The voice must be pre-processed (22050 kHz, mono, WAV), the avatar should preferably be `PNG` or `JPEG`

### TTS-STT

1) Type text for text to speech
2) Input audio for speech to text
3) Click the `Submit` button to receive the generated text and audio response
#### Voice samples = *inputs/audio/voices*
#### The voice must be pre-processed (22050 kHz, mono, WAV)

### LibreTranslate:

* First you need to install and run [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate)
1) Select source and target language
2) (Optional) You can save the translation history by turning on the corresponding button
3) Click the `Submit` button to get the translate

### StableDiffusion - has seven sub-tabs:

#### txt2img:

1) First upload your models to the folder: *inputs/image/sd_models*
2) Select your model from the drop-down list
3) Select model type (`SD`, `SD2` or `SDXL`)
4) Set up the model according to the parameters you need
5) Enter your request
6) Click the `Submit` button to get the generated image
#### Optional: You can select your `vae`, `embedding` and `lora` models to improve the generation method, also you can enable `upscale` to increase the size of the generated image 
#### vae = *inputs/image/sd_models/vae*
#### lora = *inputs/image/sd_models/lora*
#### embedding = *inputs/image/sd_models/embedding*

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

#### depth2img:

1) Upload the initial image
2) Set up the model according to the parameters you need
3) Enter your request
4) Click the `Submit` button to get the generated image

#### upscale:

1) Upload the initial image
2) Select the options you need
3) Click the `Submit` button to get the upscaled image

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

### Demucs:

1) Upload the initial audio to separate
2) Click the `Submit` button to get the separated audio

### ModelDownloader:

* Here you can download `LLM` and `StableDiffusion` models. Just choose the model from the drop-down list and click the `Submit` button
#### `LLM` models are downloaded here: *inputs/text/llm_models*
#### `StableDiffusion` models are downloaded here: *inputs/image/sd_models*

### Settings: 

* Here you can change the application settings. For now you can only change `Share` mode to `True` or `False`

### System: 

* Here you can see your computer's sensor readings

### Additional Information:

1) Chat history, generated images and generated audio, are saved in the *outputs* folder
2) You can press the `Clear` button to reset your selection
3) To stop the generation process, click the `Stop generation` button
4) You can turn off the application using the `Close terminal` button
5) You can open the *outputs* folder by clicking on the `Folder` button

## Where can I get models, voices and avatars?

* LLM models can be taken from [HuggingFace](https://huggingface.co/models) or from ModelDownloader inside interface
* StableDiffusion, vae, inpaint, embedding and lora models can be taken from [CivitAI](https://civitai.com/models) or from ModelDownloader inside interface
* AudioCraft models are downloads automatically in *inputs* folder, when you select a model and press the submit button
* TTS, Whisper, Upscale and Multiband diffusion models are downloads automatically in *inputs* folder when are they used 
* You can take voices anywhere. Record yours or take a recording from the Internet. Or just use those that are already in the project. The main thing is that it is pre-processed!
* It’s the same with avatars as with voices. You can download them on the Internet, generate them using neural networks, or take a photo of yourself. The main thing is to comply with the required file format

## Roadmap

* https://github.com/Dartvauder/NeuroSandboxWebUI/wiki/RoadmapEN

## Acknowledgment to developers

Thank you very much to these projects for allowing me to create my application:

* `gradio` - https://github.com/gradio-app/gradio
* `transformers` - https://github.com/huggingface/transformers
* `tts` - https://github.com/coqui-ai/TTS
* `openai-whisper` - https://github.com/openai/whisper
* `torch` - https://github.com/pytorch/pytorch
* `soundfile` - https://github.com/bastibe/python-soundfile
* `accelerate` - https://github.com/huggingface/accelerate
* `cuda-python` - https://github.com/NVIDIA/cuda-python
* `gitpython` - https://github.com/gitpython-developers/GitPython
* `diffusers` - https://github.com/huggingface/diffusers
* `llama.cpp-python` - https://github.com/abetlen/llama-cpp-python
* `audiocraft` - https://github.com/facebookresearch/audiocraft
* `xformers` - https://github.com/facebookresearch/xformers
* `demucs` - https://github.com/facebookresearch/demucs
* `libretranslatepy` - https://github.com/argosopentech/LibreTranslate-py

## Donation

### *If you liked my project and want to donate, here is options to donate. Thank you very much in advance!*

* CryptoWallet(BEP-20) - 0x3d86bdb5f50b92d0d7Eb44F1a833acC5e91aAEcA

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)
