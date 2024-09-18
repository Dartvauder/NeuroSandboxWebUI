## [功能](/#功能) | [依赖](/#必需依赖) | [系统要求](/#最低系统要求) | [安装](/#如何安装) | [Wiki](/#Wiki) | [致谢](/#致开发者的感谢) | [许可证](/#第三方许可证)

# ![主图](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4ea0d891-8979-45ad-b052-626c41ae991a)
* 仍在进行中但已稳定！
* [English](/README.md) | [Русский](/Readmes/README_RU.md) | 漢語

## 描述：

一个简单方便的界面，用于使用各种神经网络模型。您可以通过文本、语音和图像输入与LLM和Moondream2进行通信；使用StableDiffusion、Kandinsky、Flux、HunyuanDiT、Lumina-T2X、Kolors、AuraFlow、Würstchen、DeepFloydIF、PixArt和PlaygroundV2.5生成图像；使用ModelScope、ZeroScope 2、CogVideoX和Latte生成视频；使用StableFast3D、Shap-E、SV34D和Zero123Plus生成3D对象；使用StableAudioOpen、AudioCraft和AudioLDM 2生成音乐和音频；使用CoquiTTS、MMS和SunoBark进行文本到语音转换；使用OpenAI-Whisper和MMS进行语音到文本转换；使用Wav2Lip进行唇形同步；使用LivePortrait为图像添加动画；使用Roop进行换脸；使用Rembg移除背景；使用CodeFormer修复面部；使用PixelOE进行图像像素化；使用DDColor为图像上色；使用LibreTranslate和SeamlessM4Tv2进行文本翻译；使用Demucs和UVR进行音频文件分离；使用RVC进行语音转换。您还可以在图库中查看输出目录中的文件，下载LLM和StableDiffusion模型，在界面内更改应用程序设置并检查系统传感器。

项目目标 - 创建一个尽可能简单易用的神经网络模型应用程序

### 文本：<img width="1119" alt="1zh" src="https://github.com/user-attachments/assets/1cbda009-8230-4dc2-beb7-a77505e96d81">

### 图像：<img width="1127" alt="2zh" src="https://github.com/user-attachments/assets/151001bc-27b2-4561-84f1-7ec521ad972e">

### 视频：<img width="1117" alt="3zh" src="https://github.com/user-attachments/assets/70ebf95e-f82e-467a-a027-64ee917527cc">

### 3D：<img width="1121" alt="4zh" src="https://github.com/user-attachments/assets/6c78d5a9-3794-43af-b167-fb63a4102d83">

### 音频：<img width="1117" alt="5zh" src="https://github.com/user-attachments/assets/f87efe4d-095f-4e99-abfb-759801ff4f29">

### 额外功能：<img width="1116" alt="6zh" src="https://github.com/user-attachments/assets/f92da750-9ce6-4982-80e1-324c0ee749c3">

### 界面：<img width="1120" alt="7zh" src="https://github.com/user-attachments/assets/f444b0bb-cc7d-46fc-8eb6-a944e9269838">

## 功能：

* 通过install.bat（Windows）或install.sh（Linux）轻松安装
* 您可以通过移动设备在本地主机（通过IPv4）或在线任何地方（通过Share）使用应用程序
* 灵活且优化的界面（由Gradio提供）
* 从`Install`和`Update`文件进行调试日志记录
* 提供三种语言版本
* 支持Transformers和llama.cpp模型（LLM）
* 支持diffusers和safetensors模型（StableDiffusion）- txt2img、img2img、depth2img、marigold、pix2pix、controlnet、upscale（latent）、upscale（SUPIR）、refiner、inpaint、outpaint、gligen、diffedit、blip-diffusion、animatediff、hotshot-xl、video、ldm3d、sd3、cascade、t2i-ip-adapter、ip-adapter-faceid和riffusion标签
* 支持stable-diffusion-cpp模型用于FLUX
* 支持额外的图像生成模型：Kandinsky（txt2img、img2img、inpaint）、Flux（支持LoRA）、HunyuanDiT（txt2img、controlnet）、Lumina-T2X、Kolors（支持LoRA的txt2img、img2img、ip-adapter-plus）、AuraFlow（支持LoRA和AuraSR）、Würstchen、DeepFloydIF（txt2img、img2img、inpaint）、PixArt和PlaygroundV2.5
* 支持使用Rembg、CodeFormer、PixelOE、DDColor、DownScale、格式转换器、换脸（Roop）和放大（Real-ESRGAN）模型进行图像、视频和音频的额外处理
* 支持StableAudio
* 支持AudioCraft（模型：musicgen、audiogen和magnet）
* 支持AudioLDM 2（模型：audio和music）
* 支持TTS和Whisper模型（用于LLM和TTS-STT）
* 支持MMS进行文本到语音和语音到文本转换
* 支持Lora、Textual inversion（embedding）、Vae、MagicPrompt、Img2img、Depth、Marigold、Pix2Pix、Controlnet、Upscalers（latent和SUPIR）、Refiner、Inpaint、Outpaint、GLIGEN、DiffEdit、BLIP-Diffusion、AnimateDiff、HotShot-XL、Videos、LDM3D、SD3、Cascade、T2I-IP-ADAPTER、IP-Adapter-FaceID和Riffusion模型（用于StableDiffusion）
* 支持Multiband Diffusion模型（用于AudioCraft）
* 支持LibreTranslate（本地API）和SeamlessM4Tv2进行语言翻译
* 支持ModelScope、ZeroScope 2、CogVideoX和Latte进行视频生成
* 支持SunoBark
* 支持Demucs和UVR进行音频文件分离
* 支持RVC进行语音转换
* 支持StableFast3D、Shap-E、SV34D和Zero123Plus进行3D生成
* 支持Wav2Lip
* 支持LivePortrait为图像添加动画
* 支持LLM的多模态（Moondream 2）、PDF解析（OpenParse）、TTS（CoquiTTS）、STT（Whisper）、LORA和网络搜索（使用DuckDuckGo）
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
* [CUDA](https://developer.nvidia.com/cuda-downloads)（12.4）和[cuDNN](https://developer.nvidia.com/cudnn-downloads)（9.1）
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
5) 选择文件版本并等待应用程序启动
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

* https://github.com/Dartvauder/NeuroSandboxWebUI/wiki

## 致开发者的感谢

#### 非常感谢这些项目，因为正是通过他们的应用程序/库，我才能够创建我的应用程序：

首先，我要感谢[PyCharm](https://www.jetbrains.com/pycharm/)和[GitHub](https://desktop.github.com)的开发者。借助他们的应用程序，我能够创建并分享我的代码

[列出了所有使用的库和项目]

## 第三方许可证：

#### 许多模型都有自己的使用许可证。在使用之前，我建议您熟悉它们：

[列出了所有使用的模型及其许可证链接]

#### 这些第三方仓库代码也在我的项目中使用：

[列出了所有使用的第三方仓库代码]

## 捐赠

### *如果您喜欢我的项目并想要捐赠，这里有捐赠选项。非常感谢您的支持！*

* [!["给我买杯咖啡"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)

## Star历史

[![Star历史图表](https://api.star-history.com/svg?repos=Dartvauder/NeuroSandboxWebUI&type=Date)](https://star-history.com/#Dartvauder/NeuroSandboxWebUI&Date)
