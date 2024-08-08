## [Features](/#Features) | [Dependencies](/#Required-Dependencies) | [SystemRequirements](/#Minimum-System-Requirements) | [Install](/#How-to-install) | [Usage](/#How-to-use) | [Models](/#Where-can-I-get-models-voices-and-avatars) | [Wiki](/#Wiki) | [Acknowledgment](/#Acknowledgment-to-developers) | [Licenses](/#Third-Party-Licenses)

# ![main](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4ea0d891-8979-45ad-b052-626c41ae991a)
* Work in progress! (ALPHA)
* English | [Русский](/README_RU.md)

## Description:

A simple and convenient interface for using various neural network models. You can communicate with LLM and Moondream2 using text, voice and image input, use StableDiffusion to generate images, ZeroScope 2 to generate videos, TripoSR and Shap-E to generate 3D objects, StableAudioOpen, AudioCraft and AudioLDM 2 to generate music and audio, CoquiTTS and SunoBark for text-to-speech, OpenAI-Whisper for speech-to-text, Wav2Lip for lip-sync, Roop to faceswap, Rembg to remove background, CodeFormer for face restore, LibreTranslate for text translation and Demucs for audio file separation. You can also view files from the outputs directory in gallery, download the LLM and StableDiffusion models, change the application settings inside the interface and check system sensors

The goal of the project - to create the easiest possible application to use neural network models

### LLM: <img width="1111" alt="1" src="https://github.com/user-attachments/assets/6caf443a-03d0-465a-a96d-3485e6373be1">

### TTS-STT: <img width="1109" alt="2" src="https://github.com/user-attachments/assets/5290db5e-d55c-4ca5-8500-dfd726c846ae">

### SunoBark: <img width="1110" alt="3" src="https://github.com/user-attachments/assets/73a7d006-37d6-4e96-9a73-285bc7960686">

### LibreTranslate: <img width="1115" alt="4" src="https://github.com/user-attachments/assets/1747c931-6052-4972-b012-e74034f78bc0">

### Wav2Lip: <img width="1113" alt="5" src="https://github.com/user-attachments/assets/3135d71a-dcac-4080-b8a8-8d8f41a545bb">

### StableDiffusion: <img width="1109" alt="6" src="https://github.com/user-attachments/assets/2146897a-737d-4a54-8d68-3efb6abc894b">

### ZeroScope 2: <img width="1115" alt="7" src="https://github.com/user-attachments/assets/ae3d6372-a216-4826-9bb7-a5a1ffdddece">

### TripoSR: <img width="1112" alt="8" src="https://github.com/user-attachments/assets/7aade6d0-0c98-4f26-acf8-d9635a3645c0">

### Shap-E: <img width="1117" alt="9" src="https://github.com/user-attachments/assets/2da9ffee-28f7-4c75-aa64-2e74e521569c">

### StableAudio: <img width="1113" alt="10" src="https://github.com/user-attachments/assets/e2e53880-7a52-4f25-aa95-f1479a921e5f">

### AudioCraft: <img width="1118" alt="11" src="https://github.com/user-attachments/assets/42cceb62-04a2-46a2-a0c4-71fea6254d24">

### AudioLDM 2: <img width="1121" alt="12" src="https://github.com/user-attachments/assets/65fd1c9e-19e6-40c4-af89-8671184e3ea0">

### Demucs: <img width="1125" alt="13" src="https://github.com/user-attachments/assets/39fb383c-9d60-433a-8d72-5660df59b10d">

### Gallery: <img width="1129" alt="14" src="https://github.com/user-attachments/assets/bb1d05c6-faaa-4a25-8eba-430e2d814563">

### ModelDownloader: <img width="1130" alt="15" src="https://github.com/user-attachments/assets/a5313b00-2add-4111-be16-deeb27a8d0ff">

### Settings: <img width="1121" alt="16" src="https://github.com/user-attachments/assets/67f2f4c3-59a2-4cd1-9247-2f16d79cdb86">

### System: <img width="1114" alt="17" src="https://github.com/user-attachments/assets/fa7aa876-0510-437e-8fdd-1313e871aac5">

## Features:

* Easy installation via install.bat(Windows) or install.sh(Linux)
* You can use the application via your mobile device in localhost(Via IPv4) or anywhere online(Via Share)
* Flexible and optimized interface (By Gradio)
* Authentication via admin:admin (You can enter your login details in the GradioAuth.txt file)
* Support for Transformers and llama.cpp models (LLM)
* Support for diffusers and safetensors models (StableDiffusion) - txt2img, img2img, depth2img, pix2pix, controlnet, upscale, inpaint, gligen, animatediff, video, ldm3d, sd3, cascade and extras tabs
* StableAudioOpen support
* AudioCraft support (Models: musicgen, audiogen and magnet)
* AudioLDM 2 support (Models: audio and music)
* Supports TTS and Whisper models (For LLM and TTS-STT)
* Supports Lora, Textual inversion (embedding), Vae, Img2img, Depth, Pix2Pix, Controlnet, Upscale, Inpaint, GLIGEN, AnimateDiff, Videos, LDM3D, SD3, Cascade, Rembg, CodeFormer and Roop models (For StableDiffusion)
* Support Multiband Diffusion model (For AudioCraft)
* Support LibreTranslate (Local API)
* Support ZeroScope 2
* Support SunoBark
* Support Demucs
* Support Shap-E
* Support TripoSR
* Support Wav2Lip
* Support Multimodal (Moondream 2), LORA (transformers) and WebSearch (with GoogleSearch) for LLM
* Model settings inside the interface
* Gallery
* ModelDownloader (For LLM and StableDiffusion)
* Application settings
* Ability to see system sensors

## Required Dependencies:

* [Python](https://www.python.org/downloads/) (3.10+)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (12.X) and [cuDNN](https://developer.nvidia.com/cudnn-downloads) (9.X)
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

#### Interface has seventeen tabs: LLM, TTS-STT, SunoBark, LibreTranslate, Wav2Lip, StableDiffusion, ZeroScope 2, TripoSR, Shap-E, StableAudio, AudioCraft, AudioLDM 2, Demucs, Gallery, ModelDownloader, Settings and System. Select the one you need and follow the instructions below 

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
1) Select source and target languages
2) Click the `Submit` button to get the translate
#### Optional: you can save the translation history by turning on the corresponding button

### Wav2Lip:

1) Upload the initial image of face
2) Upload the initial audio of voice
3) Set up the model according to the parameters you need
4) Click the `Submit` button to receive the lip-sync

### StableDiffusion - has fourteen sub-tabs:

#### txt2img:

1) First upload your models to the folder: *inputs/image/sd_models*
2) Select your model from the drop-down list
3) Select model type (`SD`, `SD2` or `SDXL`)
4) Set up the model according to the parameters you need
5) Enter your request (+ and - for prompt weighting)
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
6) Enter your request (+ and - for prompt weighting)
7) Click the `Submit` button to get the generated image
#### Optional: You can select your `vae` model
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
7) Draw a place for generation and enter your request (+ and - for prompt weighting)
8) Click the `Submit` button to get the inpainted image
#### Optional: You can select your `vae` model
#### vae = *inputs/image/sd_models/vae*

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

#### video:

1) Upload the initial image
2) Enter your request (for IV2Gen-XL)
3) Set up the model according to the parameters you need
4) Click the `Submit` button to get the video from image

#### ldm3d:

1) Enter your request
2) Set up the model according to the parameters you need
3) Click the `Submit` button to get the generated images

#### sd3:

1) Enter your request
2) Set up the model according to the parameters you need
3) Click the `Submit` button to get the generated image

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

### TripoSR:

1) Upload the initial image
2) Set up the model according to the parameters you need
3) Click the `Submit` button to get the generated 3D object

### Shap-E:

1) Enter your request or upload the initial image
2) Set up the model according to the parameters you need
3) Click the `Submit` button to get the generated 3D object

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
3) To stop the generation process, click the `Stop generation` button
4) You can turn off the application using the `Close terminal` button
5) You can open the *outputs* folder by clicking on the `Outputs` button

## Where can i get models and voices?

* LLM models can be taken from [HuggingFace](https://huggingface.co/models) or from ModelDownloader inside interface 
* StableDiffusion, vae, inpaint, embedding and lora models can be taken from [CivitAI](https://civitai.com/models) or from ModelDownloader inside interface
* StableAudioOpen, AudioCraft, AudioLDM 2, TTS, Whisper, Wav2Lip, SunoBark, MoonDream2, Upscale, GLIGEN, Depth, Pix2Pix, Controlnet, AnimateDiff, Videos, LDM3D, SD3, Cascade, Rembg, Roop, CodeFormer, TripoSR, Shap-E, Demucs, ZeroScope and Multiband diffusion models are downloads automatically in *inputs* folder when are they used 
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
* `googlesearch-python` - https://github.com/Nv7-GitHub/googlesearch
* `torchmcubes` - https://github.com/tatsy/torchmcubes
* `suno-bark` - https://github.com/suno-ai/bark

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
* [TripoSR](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)
* [GLIGEN](https://huggingface.co/spaces/CompVis/stable-diffusion-license)
* [Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
* [Roop](https://github.com/s0md3v/roop/blob/main/LICENSE)
* [CodeFormer](https://github.com/sczhou/CodeFormer/blob/master/LICENSE)
* [ControlNet](https://github.com/lllyasviel/ControlNet/blob/main/LICENSE)
* [AnimateDiff](https://github.com/guoyww/AnimateDiff/blob/main/LICENSE.txt)
* [Pix2Pix](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)

## Donation

### *If you liked my project and want to donate, here is options to donate. Thank you very much in advance!*

* CryptoWallet(BEP-20) - 0x3d86bdb5f50b92d0d7Eb44F1a833acC5e91aAEcA

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)
