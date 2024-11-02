## [功能](/#功能) | [依赖](/#必需依赖) | [系统要求](/#最低系统要求) | [安装](/#如何安装) | [Wiki](/#Wiki) | [致谢](/#致开发者的感谢) | [许可证](/#第三方许可证)

# ![主图](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4ea0d891-8979-45ad-b052-626c41ae991a)
* 仍在进行中但已稳定！
* [English](/README.md) | [Русский](/Readmes/README_RU.md) | 漢語

## 描述：

一个简单方便的界面，用于使用各种神经网络模型。您可以通过文本、语音和图像输入与LLM进行通信；使用StableDiffusion、Kandinsky、Flux、HunyuanDiT、Lumina-T2X、Kolors、AuraFlow、Würstchen、DeepFloydIF、PixArt, CogView3-Plus和PlaygroundV2.5生成图像；使用ModelScope、ZeroScope 2、CogVideoX和Latte生成视频；使用StableFast3D、Shap-E和Zero123Plus生成3D对象；使用StableAudioOpen、AudioCraft和AudioLDM 2生成音乐和音频；使用CoquiTTS、MMS和SunoBark进行文本到语音转换；使用OpenAI-Whisper和MMS进行语音到文本转换；使用Wav2Lip进行唇形同步；使用LivePortrait为图像添加动画；使用Roop进行换脸；使用Rembg移除背景；使用CodeFormer修复面部；使用PixelOE进行图像像素化；使用DDColor为图像上色；使用LibreTranslate和SeamlessM4Tv2进行文本翻译；使用Demucs和UVR进行音频文件分离；使用RVC进行语音转换。您还可以在图库中查看输出目录中的文件，下载LLM和StableDiffusion模型，在界面内更改应用程序设置并检查系统传感器。

项目目标 - 创建一个尽可能简单易用的神经网络模型应用程序

### 文本：<img width="1115" alt="1zh" src="https://github.com/user-attachments/assets/28c31659-2ce4-46dc-8cee-406cf6cb620e">

### 图像：<img width="1112" alt="2zh" src="https://github.com/user-attachments/assets/cd86ad42-abee-4dec-8619-c50958ea4fe3">

### 视频：<img width="1113" alt="3zh" src="https://github.com/user-attachments/assets/71405dba-e1e2-4a5b-bef7-e7e54eb84a86">

### 3D：<img width="1111" alt="4zh" src="https://github.com/user-attachments/assets/5c0ec1d7-5ed9-4eeb-b054-d67884d4f940">

### 音频：<img width="1114" alt="5zh" src="https://github.com/user-attachments/assets/c66c6131-61ca-4141-8c2a-3f9404e82102">

### 额外功能：<img width="1109" alt="6zh" src="https://github.com/user-attachments/assets/0f7eec1e-c60e-4c86-b273-28a872407df0">

### 界面：<img width="1115" alt="7zh" src="https://github.com/user-attachments/assets/587592fe-c773-4729-825a-0b18adcf8c06">

## 功能：

* 通过install.bat（Windows）或install.sh（Linux）轻松安装
* 您可以通过移动设备在本地主机（通过IPv4）或在线任何地方（通过Share）使用应用程序
* 灵活且优化的界面（由Gradio提供）
* 从`Install`和`Update`文件进行调试日志记录
* 提供三种语言版本
* 支持Transformers, BNB, GPTQ, AWQ, ExLlamaV2和llama.cpp模型（LLM）
* 支持diffusers和safetensors模型（StableDiffusion）- txt2img、img2img、depth2img、marigold、pix2pix、controlnet、upscale（latent)、refiner、inpaint、outpaint、gligen、diffedit、blip-diffusion、animatediff、hotshot-xl、video、ldm3d、sd3、cascade、t2i-ip-adapter、ip-adapter-faceid和riffusion标签
* 支持stable-diffusion-cpp模型用于FLUX和StableDiffusion
* 支持额外的图像生成模型：Kandinsky（txt2img、img2img、inpaint）、Flux (txt2img 支持 cpp quantize 和 LoRA, img2img, inpaint, controlnet) 、HunyuanDiT（txt2img、controlnet）、Lumina-T2X、Kolors（支持LoRA的txt2img、img2img、ip-adapter-plus）、AuraFlow（支持LoRA和AuraSR）、Würstchen、DeepFloydIF（txt2img、img2img、inpaint）、PixArt, CogView3-Plus和PlaygroundV2.5
* 支持使用Rembg、CodeFormer、PixelOE、DDColor、DownScale、格式转换器、换脸（Roop）和放大（Real-ESRGAN）模型进行图像、视频和音频的额外处理
* 支持StableAudio
* 支持AudioCraft（模型：musicgen、audiogen和magnet）
* 支持AudioLDM 2（模型：audio和music）
* 支持TTS和Whisper模型（用于LLM和TTS-STT）
* 支持MMS进行文本到语音和语音到文本转换
* 支持Lora、Textual inversion（embedding）、Vae、MagicPrompt、Img2img、Depth、Marigold、Pix2Pix、Controlnet、Upscale（latent）、Refiner、Inpaint、Outpaint、GLIGEN、DiffEdit、BLIP-Diffusion、AnimateDiff、HotShot-XL、Videos、LDM3D、SD3、Cascade、T2I-IP-ADAPTER、IP-Adapter-FaceID和Riffusion模型（用于StableDiffusion）
* 支持Multiband Diffusion模型（用于AudioCraft）
* 支持LibreTranslate（本地API）和SeamlessM4Tv2进行语言翻译
* 支持ModelScope、ZeroScope 2、CogVideoX和Latte进行视频生成
* 支持SunoBark
* 支持Demucs和UVR进行音频文件分离
* 支持RVC进行语音转换
* 支持StableFast3D、Shap-E和Zero123Plus进行3D生成
* 支持Wav2Lip
* 支持LivePortrait为图像添加动画
* 支持LLM的多模态（Moondream 2, LLaVA-NeXT-Video, Qwen2-Audio）、PDF解析（OpenParse）、TTS（CoquiTTS）、STT（Whisper）、LORA和网络搜索（使用DuckDuckGo）
* 用于生成图像、视频和音频的元数据信息查看器
* 界面内的模型设置
* 在线和离线Wiki
* 图库
* 模型下载器（用于LLM和StableDiffusion）
* 应用程序设置
* 能够查看系统传感器

## 必需依赖：

* [Python](https://www.python.org/downloads/)（3.10.11）
* [Git](https://git-scm.com/downloads)
* 仅适用于GPU版本: [CUDA](https://developer.nvidia.com/cuda-downloads)（12.4）和[cuDNN](https://developer.nvidia.com/cudnn-downloads)（9.1）
* [FFMPEG](https://ffmpeg.org/download.html)
- C++编译器
  - Windows：[VisualStudio](https://visualstudio.microsoft.com/ru/)、[VisualStudioCode](https://code.visualstudio.com)和[Cmake](https://cmake.org)
  - Linux：[GCC](https://gcc.gnu.org/)、[VisualStudioCode](https://code.visualstudio.com)和[Cmake](https://cmake.org)

## 最低系统要求：

* 系统：Windows或Linux
* GPU：6GB+或CPU：8核3.6GHZ
* RAM：16GB+
* 磁盘空间：20GB+
* 需要互联网连接以下载模型和进行安装

## 如何安装：

### Windows

1) 首先安装所有[必需依赖](/#必需依赖)
2) 在任意位置执行`Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git`
3) 运行`Install.bat`并等待安装完成
4) 安装完成后，运行`Start.bat`
5) 等待应用程序启动
6) 现在您可以开始生成了！

要获取更新，请运行`Update.bat`
要通过终端使用虚拟环境，请运行`Venv.bat`

### Linux

1) 首先安装所有[必需依赖](/#必需依赖)
2) 在任意位置执行`Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git`
3) 在终端中运行`./Install.sh`并等待所有依赖项安装完成
4) 安装完成后，运行`./Start.sh`
5) 等待应用程序启动
6) 现在您可以开始生成了！

要获取更新，请运行`./Update.sh`
要通过终端使用虚拟环境，请运行`./Venv.sh`

## Wiki

* https://github.com/Dartvauder/NeuroSandboxWebUI/wiki/ZH‐Wiki

## 致开发者的感谢

#### 非常感谢这些项目，因为正是通过他们的应用程序/库，我才能够创建我的应用程序：

首先，我要感谢[PyCharm](https://www.jetbrains.com/pycharm/)和[GitHub](https://desktop.github.com)的开发者。借助他们的应用程序，我能够创建并分享我的代码

* `gradio` - https://github.com/gradio-app/gradio
* `transformers` - https://github.com/huggingface/transformers
* `auto-gptq` - https://github.com/AutoGPTQ/AutoGPTQ
* `autoawq` - https://github.com/casper-hansen/AutoAWQ
* `exllamav2` - https://github.com/turboderp/exllamav2
* `coqui-tts` - https://github.com/idiap/coqui-ai-TTS
* `openai-whisper` - https://github.com/openai/whisper
* `torch` - https://github.com/pytorch/pytorch
* `cuda-python` - https://github.com/NVIDIA/cuda-python
* `gitpython` - https://github.com/gitpython-developers/GitPython
* `diffusers` - https://github.com/huggingface/diffusers
* `llama.cpp-python` - https://github.com/abetlen/llama-cpp-python
* `stable-diffusion-cpp-python` - https://github.com/william-murray1204/stable-diffusion-cpp-python
* `audiocraft` - https://github.com/facebookresearch/audiocraft
* `xformers` - https://github.com/facebookresearch/xformers
* `demucs` - https://github.com/facebookresearch/demucs
* `libretranslatepy` - https://github.com/argosopentech/LibreTranslate-py
* `rembg` - https://github.com/danielgatis/rembg
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
* `sd_embed` - https://github.com/xhinker/sd_embed
* `triton` - https://github.com/triton-lang/triton/

## 第三方许可证：

#### 许多模型都有自己的使用许可证。在使用之前，我建议您熟悉它们：

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
* [StableDiffusion3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/LICENSE.md)
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
* [MagicPrompt](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)
* [Marigold](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
* [BLIP-Diffusion](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
* [Consistency-Decoder](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)
* [Tiny-AutoEncoder](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)
* [CogView3-Plus](https://huggingface.co/THUDM/CogView3-Plus-3B/blob/main/LICENSE.md)

#### 这些第三方仓库代码也在我的项目中使用：

* [CodeFormer for extras](https://github.com/sczhou/CodeFormer)
* [Real-ESRGAN for upscale](https://github.com/xinntao/Real-ESRGAN)
* [HotShot-XL for StableDiffusion](https://github.com/hotshotco/Hotshot-XL)
* [Roop for extras](https://github.com/s0md3v/roop)
* [StableFast3D for 3D](https://github.com/Stability-AI/stable-fast-3d)
* [Riffusion for StableDiffusion](https://github.com/riffusion/riffusion-hobby)
* [DDColor for extras](https://github.com/piddnad/DDColor)
* [LivePortrait for video](https://github.com/KwaiVGI/LivePortrait)
* [TAESD for StableDiffusion and Flux](https://github.com/madebyollin/taesd)

## 捐赠

### *如果您喜欢我的项目并想要捐赠，这里有捐赠选项。非常感谢您的支持！*

* [!["给我买杯咖啡"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)

## 星星的历史

[![Star History Chart](https://api.star-history.com/svg?repos=Dartvauder/NeuroSandboxWebUI&type=Date)](https://star-history.com/#Dartvauder/NeuroSandboxWebUI&Date)
