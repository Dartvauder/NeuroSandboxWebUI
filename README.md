## [Features](/#Features) | [Dependencies](/#Required-Dependencies) | [SystemRequirements](/#Minimum-System-Requirements) | [Install](/#How-to-install) | [Usage](/#How-to-use) | [Models](/#Where-can-I-get-models-voices-and-avatars) | [Roadmap](/#Roadmap) | [Acknowledgment](/#Acknowledgment-to-developers)

# ![icon (1)](https://github.com/Dartvauder/NeuroChatWebUI/assets/140557322/e3c1d95a-828f-4a65-bea6-64c336dbe6fa) NeuroChatWebUI (ALPHA)
* Work in progress!
* English | [Русский](/README_RU.md)
## Description:

Simple and easy interface for chatting with LLM using text or voice input and also Stable Diffusion for generating images. TTS and Whisper functions are available here for voice input or output with a choice of language and voice sample. The goal of the project - to create the easiest possible application for beginners in the topic of neural networks.

|![Image1](https://github.com/Dartvauder/NeuroChatWebUI/assets/140557322/6991dc9e-ba26-43a9-8a61-e09b72ed2235) | ![Image2](https://github.com/Dartvauder/NeuroChatWebUI/assets/140557322/9217199a-07a6-481a-a652-913da995030b) |
|:---:|:---:|

## Features:

* Easy install (Windows only)
* Flexible and optimized interface
* TTS and STT models (For LLM)
* Avatar choose (For LLM)
* Transformers and llama.cpp (For LLM)
* Diffusers and safetensors (For Stable Diffusion)
* Models settings in interface
* And much more...

## Required Dependencies:

* [Python](https://www.python.org/downloads/) (3.10 minimum)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (12.1 only)
* [FFMPEG](https://ffmpeg.org/download.html)

## Minimum System Requirements:

* System: Windows or Linux
* GPU: 4GB+ or CPU: 8 core 3.2GHZ
* RAM: 16GB+
* Disk space: 30GB+
* Internet for downloading models and installing

## How to install:

### Windows

1) `Git clone https://github.com/Dartvauder/NeuroChatWebUI.git` to any location
2) Run the `install.bat` and choose a version for installation
3) After installation, run `start.bat`
4) Select the file version and wait for the application to launch
5) Now you can start generating!

To get update, run `update.bat`

### Linux

1) `Git clone https://github.com/Dartvauder/NeuroChatWebUI.git` to any location
2) In the terminal, run the `pip install -r requirementsGPU.txt` or `pip install -r requirementsCPU.txt` and wait for installation of all dependencies
3) After installation, run `py appEN.py` or `py appRU.py`
4) Wait for the application to launch
5) Now you can start generating!

To get update, run `git pull`

## How to use:

### LLM:

1) First upload your models to the folder: *inputs/text/llm_models*
2) Select your model from the `LLM` drop-down list
3) Select model type (`transformers` or `llama`)
4) Set up the model according to the parameters you need
5) Type (or speak) your request
6) Click the `Generate` button to receive text and audio response
#### Optional: you can enable `TTS` mode, select the `voice` and `language` needed to receive an audio response. You can also select `avatar`
#### Voice samples = *inputs/audio/voices*;
#### Avatars = *inputs/image/avatars*
#### The voice must be pre-processed (22050 kHz, mono, WAV), the avatar should preferably be `PNG` or `JPG`. When you enable `TTS`, you need to select a language and voice, otherwise there will be errors

### Stable Diffusion:

#### Before generating the image, disable the `LLM` and `TTS` models, and also clear the audio input

1) First upload your models to the folder: *inputs/image/sd_models*
2) Enable `Stable Diffusion`
3) Select a model from the drop-down list
4) Set up the model according to the parameters you need
5) Enter your prompt
6) Click the `generate` button to get the image

### General:

#### Chat history and generated images are saved in the *outputs* folder
#### You can also press the `Clear` button to clear the selection
#### To close the application, close the terminal

## Where can I get models, voices and avatars?

* LLM models can be taken from [HuggingFace](https://huggingface.co/models)
* Stable Diffusion models can be taken from [CivitAI](https://civitai.com/models)
* You can take voices anywhere. Record yours or take a recording from the Internet. The main thing is that it is pre-processed!
* It’s the same with avatars as with voices. You can download them on the Internet, generate them using neural networks, or take a photo of yourself. The main thing is to comply with the required file format
* #### `TTS` and `STT` models are downloaded automatically in *inputs* folder. The base `diffusers` model of `Stable Diffusion` is also downloaded there 

## Roadmap

* https://github.com/Dartvauder/NeuroChatWebUI/wiki/RoadmapEN

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

## Donation

### *If you liked my project and want to donate, here is options to donate. Thank you very much in advance!*

* CryptoWallet(BEP-20) - 0x3d86bdb5f50b92d0d7Eb44F1a833acC5e91aAEcA

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)

