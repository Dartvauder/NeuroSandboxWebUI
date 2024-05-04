## [Features](/#Features) | [Dependencies](/#Required-Dependencies) | [SystemRequirements](/#Minimum-System-Requirements) | [Install](/#How-to-install) | [Usage](/#How-to-use) | [Models](/#Where-can-I-get-models-voices-and-avatars) | [Roadmap](/#Roadmap) | [Acknowledgment](/#Acknowledgment-to-developers) | [Licenses](/#Third-Party-Licenses)

# ![main](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4ea0d891-8979-45ad-b052-626c41ae991a)
* Work in progress! (ALPHA)
* English | [Русский](/README_RU.md)

## Description:

A simple and convenient interface for using various neural network models. You can communicate with LLM and Moondream2 using text, voice and image input, use StableDiffusion to generate images, ZeroScope 2 to generate videos, Shap-E to generate 3D objects, AudioCraft and AudioLDM 2 to generate music and audio, CoquiTTS and SunoBark for text-to-speech, OpenAI-Whisper for speech-to-text, LibreTranslate for text translation and Demucs for audio file separation. You can also download the LLM and StableDiffusion models, change the application settings inside the interface and check system sensors

The goal of the project - to create the easiest possible application to use neural network models

### LLM: ![1](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/2e7c1e23-80a9-4937-8076-a62f256e8d12)

### TTS-STT: ![2](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/0b94bb40-4d38-4ebd-9f1e-86931614ee32)

### SunoBark: ![3](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/67d0018f-c6c8-47fa-b513-779bcc85d9ca)

### LibreTranslate: ![4](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/c8c0e72c-44dc-443a-bd20-56c7872fd8d4)

### StableDiffusion: ![5](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/b4c82e1a-94e8-42c7-a348-8beab068b227)

### ZeroScope 2: ![6](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/98ba05ab-70e3-4738-8263-26e0d86fefc7)

### Shap-E: ![7](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/2daa2c45-b2e0-492b-a3ff-cf2f793ce7b8)
 
### AudioCraft: ![8](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/13090c30-2ac8-4fa5-bcdf-37cbc9ffbdb1)

### AudioLDM 2: ![9](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/d0a2fe77-6933-4ac7-9434-dcce5220b4d0)

### Demucs: ![10](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/d6e3d040-8d03-4107-a473-4994be468a9e)

### ModelDownloader: ![11](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/7a940624-cc5f-43ab-9229-e07e49cdf294)

### Settings: ![12](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/439481be-3b61-4bf8-825b-0ae2d0244c1e)

### System: ![13](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/a2e56450-31e6-44c6-9be0-388515800bbb)

## Features:

* Easy installation via install.bat (Windows only)
* You can use the application via your mobile device in localhost (Via IPv4)
* Flexible and optimized interface (By Gradio)
* Support for Transformers and llama.cpp models (LLM)
* Support for diffusers and safetensors models (StableDiffusion) - txt2img, img2img, depth2img, upscale, inpaint, video, cascade and extras tabs
* AudioCraft support (Models: musicgen, audiogen and magnet)
* AudioLDM 2 support
* Supports TTS and Whisper models (For LLM and TTS-STT)
* Supports Lora, Textual inversion (embedding), Vae, Inpaint, Depth, Upscale and Video models (For StableDiffusion)
* Support Multiband Diffusion model (For AudioCraft)
* Support LibreTranslate (Local API)
* Support ZeroScope 2
* Support SunoBark
* Support Demucs
* Support Rembg (For StableDiffusion)
* Support Shap-E
* Support multimodal (Moondream 2 for LLM)
* Support WebSearch (For LLM with GoogleSearch)
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
2) In the terminal, run the `pip install --no-deps -r requirements.txt`, `pip install --no-deps -r requirements-cuda.txt`, `pip install --no-deps -r requirements-llama-cpp.txt` and wait for installation of all dependencies
3) After installation, run `py appEN.py` or `py appRU.py`
4) Wait for the application to launch
5) Now you can start generating!

To get update, run `git pull`

## How to use:

#### Interface has thirteen tabs: LLM, TTS-STT, SunoBark, LibreTranslate, StableDiffusion, ZeroScope 2, Shap-E, AudioCraft, AudioLDM 2, Demucs, ModelDownloader, Settings and System. Select the one you need and follow the instructions below 

### LLM:

1) First upload your models to the folder: *inputs/text/llm_models*
2) Select your model from the drop-down list
3) Select model type (`transformers` or `llama`)
4) Set up the model according to the parameters you need
5) Type (or speak) your request
6) Click the `Submit` button to receive the generated text and audio response
#### Optional: you can enable `TTS` mode, select the `voice` and `language` needed to receive an audio response. You can also enable `multimodal` and upload an image to get its description
#### Voice samples = *inputs/audio/voices*
#### The voice must be pre-processed (22050 kHz, mono, WAV)

### TTS-STT:

1) Type text for text to speech
2) Input audio for speech to text
3) Click the `Submit` button to receive the generated text and audio response
#### Voice samples = *inputs/audio/voices*
#### The voice must be pre-processed (22050 kHz, mono, WAV)

### SunoBark:

1) Type your request
2) Set up the model according to the parameters you need
3) Click the `Submit` button to receive the generated audio response

### LibreTranslate:

* First you need to install and run [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate)
1) Select source and target language
2) (Optional) You can save the translation history by turning on the corresponding button
3) Click the `Submit` button to get the translate

### StableDiffusion - has eight sub-tabs:

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
2) Set up the model according to the parameters you need
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

#### cascade:

1) Enter your request
2) Set up the model according to the parameters you need
3) Click the `Submit` button to get the generated image

#### extras:

1) Upload the initial image
2) Select the options you need
3) Click the `Submit` button to get the modified image

### ZeroScope 2:

1) Enter your request
2) Set up the model according to the parameters you need
3) Click the `Submit` button to get the generated video

### Shap-E:

1) Enter your request or upload the initial image
2) Set up the model according to the parameters you need
3) Click the `Submit` button to get the generated 3D object

### AudioCraft:

1) Select a model from the drop-down list
2) Select model type (`musicgen` or `audiogen`)
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
3) To stop the generation process, click the `Stop generation` button
4) You can turn off the application using the `Close terminal` button
5) You can open the *outputs* folder by clicking on the `Folder` button

## Where can i get models and voices?

* LLM models can be taken from [HuggingFace](https://huggingface.co/models) or from ModelDownloader inside interface 
* StableDiffusion, vae, inpaint, embedding and lora models can be taken from [CivitAI](https://civitai.com/models) or from ModelDownloader inside interface
* AudioCraft and AudioLDM 2 models are downloads automatically in *inputs* folder, when you select a model and press the submit button
* TTS, Whisper, SunoBark, MoonDream2, Upscale, Depth, Videos, Cascade, Rembg, Shap-E, Demucs, ZeroScope and Multiband diffusion models are downloads automatically in *inputs* folder when are they used 
* You can take voices anywhere. Record yours or take a recording from the Internet. Or just use those that are already in the project. The main thing is that it is pre-processed!

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
* `AudioLDM2` - https://github.com/haoheliu/AudioLDM2
* `xformers` - https://github.com/facebookresearch/xformers
* `demucs` - https://github.com/facebookresearch/demucs
* `libretranslatepy` - https://github.com/argosopentech/LibreTranslate-py
* `rembg` - https://github.com/danielgatis/rembg
* `trimesh` - https://github.com/mikedh/trimesh
* `googlesearch-python` - https://github.com/Nv7-GitHub/googlesearch

## Third Party Licenses:

#### Many models have their own license for use. Before using it, I advise you to familiarize yourself with them:

* [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE)
* [llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/LICENSE)
* [CoquiTTS](https://github.com/coqui-ai/TTS/blob/dev/LICENSE.txt)
* [OpenAI-Whisper](https://github.com/openai/whisper/blob/main/LICENSE)
* [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate/blob/main/LICENSE)
* [Diffusers](https://github.com/huggingface/diffusers/blob/main/LICENSE)
* [StableDiffusion1.5](https://github.com/runwayml/stable-diffusion/blob/main/LICENSE)
* [StableDiffusion2](https://github.com/Stability-AI/stablediffusion/blob/main/LICENSE)
* [StableDiffusionXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)
* [StableCascade](https://huggingface.co/stabilityai/stable-cascade/blob/main/LICENSE)
* [StableVideoDiffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/blob/main/LICENSE)
* [Rembg](https://github.com/danielgatis/rembg/blob/main/LICENSE.txt)
* [Shap-E](https://github.com/openai/shap-e/blob/main/LICENSE)
* [AudioCraft](https://github.com/facebookresearch/audiocraft/blob/main/LICENSE)
* [AudioLDM2](https://github.com/haoheliu/AudioLDM2/blob/main/LICENSE)
* [Demucs](https://github.com/facebookresearch/demucs/blob/main/LICENSE)
* [SunoBark](https://github.com/suno-ai/bark/blob/main/LICENSE)
* [Moondream2](https://github.com/vikhyat/moondream/blob/main/LICENSE)
* [ZeroScope2](https://spdx.org/licenses/CC-BY-NC-4.0)

## Donation

### *If you liked my project and want to donate, here is options to donate. Thank you very much in advance!*

* CryptoWallet(BEP-20) - 0x3d86bdb5f50b92d0d7Eb44F1a833acC5e91aAEcA

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)
