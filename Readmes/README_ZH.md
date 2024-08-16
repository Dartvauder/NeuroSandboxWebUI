## [功能](/#功能) | [依赖](/#必需依赖) | [系统要求](/#最低系统要求) | [安装](/#如何安装) | [使用](/#如何使用) | [模型](/#我在哪里可以获得模型语音和头像) | [Wiki](/#Wiki) | [致谢](/#致开发者的感谢) | [许可证](/#第三方许可证)

# ![main](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4ea0d891-8979-45ad-b052-626c41ae991a)
* 正在进行中！（Alpha版）
* [English](/README.md)  | [عربي](/Readmes/README_AR.md) | [Deutsche](/Readmes/README_DE.md) | [Español](/Readmes/README_ES.md) | [Français](/Readmes/README_FR.md) | [日本語](/Readmes/README_JP.md) | [Русский](/Readmes/README_RU.md) | 漢語 | [Português](/Readmes/README_PT.md) | [हिंदी](/Readmes/README_HI.md) | [Italiano](/Readmes/README_IT.md) | [韓國語](/Readmes/README_KO.md)

## 描述：

一个简单方便的界面，用于使用各种神经网络模型。您可以通过文本、语音和图像输入与LLM和Moondream2进行通信；使用StableDiffusion、Kandinsky、Flux、HunyuanDiT、Lumina-T2X、Kolors、AuraFlow、Würstchen、DeepFloydIF和PixArt生成图像；使用ModelScope、ZeroScope 2、CogVideoX和Latte生成视频；使用TripoSR、StableFast3D、Shap-E、SV34D和Zero123Plus生成3D对象；使用StableAudioOpen、AudioCraft和AudioLDM 2生成音乐和音频；使用CoquiTTS和SunoBark进行文本转语音；使用OpenAI-Whisper进行语音转文本；使用Wav2Lip进行唇形同步；使用Roop进行换脸；使用Rembg移除背景；使用CodeFormer恢复面部；使用LibreTranslate进行文本翻译；使用Demucs进行音频文件分离。您还可以在画廊中查看outputs目录中的文件，下载LLM和StableDiffusion模型，在界面内更改应用程序设置并检查系统传感器

该项目的目标是创建一个尽可能简单易用的神经网络模型应用程序

### 文本：<img width="1127" alt="1zh" src="https://github.com/user-attachments/assets/c8f71af3-49ac-48e6-be22-45b933fc7b2c">

### 图像：<img width="1109" alt="2zh" src="https://github.com/user-attachments/assets/1a7923d1-d112-453b-be96-a85313e18b2b">

### 视频：<img width="1126" alt="3zh" src="https://github.com/user-attachments/assets/a26222d4-cfcd-4869-aa0e-18b984f49d00">

### 3D：<img width="1127" alt="4zh" src="https://github.com/user-attachments/assets/84edbd5f-0437-490f-bfb2-2930f8a5d05e">

### 音频：<img width="1118" alt="5zh" src="https://github.com/user-attachments/assets/b7ff6de8-2aaa-4674-ae95-d9084128ac23">

### 界面：<img width="1118" alt="6zh" src="https://github.com/user-attachments/assets/2df0ba1a-f0ea-4a9e-9e71-2c9f7d625051">

## 功能：

* 通过install.bat（Windows）或install.sh（Linux）轻松安装
* 您可以通过移动设备在本地主机（通过IPv4）或在线任何地方（通过Share）使用应用程序
* 灵活且优化的界面（由Gradio提供）
* 通过admin:admin进行身份验证（您可以在GradioAuth.txt文件中输入您的登录详细信息）
* 您可以添加自己的HuggingFace-Token以下载特定模型（您可以在HF-Token.txt文件中输入您的令牌）
* 支持Transformers和llama.cpp模型（LLM）
* 支持diffusers和safetensors模型（StableDiffusion）- txt2img、img2img、depth2img、pix2pix、controlnet、upscale、inpaint、gligen、animatediff、video、ldm3d、sd3、cascade, adapters和extras选项卡
* 支持额外的图像生成模型：Kandinsky、Flux、HunyuanDiT、Lumina-T2X、Kolors、AuraFlow、Würstchen、DeepFloydIF和PixArt
* 支持StableAudioOpen
* 支持AudioCraft（模型：musicgen、audiogen和magnet）
* 支持AudioLDM 2（模型：audio和music）
* 支持TTS和Whisper模型（用于LLM和TTS-STT）
* 支持Lora、Textual inversion（嵌入）、Vae、Img2img、Depth、Pix2Pix、Controlnet、Upscale（latent）、Upscale（Real-ESRGAN）、Inpaint、GLIGEN、AnimateDiff、Videos、LDM3D、SD3、Cascade, Adapters (InstantID, PhotoMaker, IP-Adapter-FaceID), Rembg、CodeFormer和Roop模型（用于StableDiffusion）
* 支持Multiband Diffusion模型（用于AudioCraft）
* 支持LibreTranslate（本地API）
* 支持ModelScope、ZeroScope 2、CogVideoX和Latte进行视频生成
* 支持SunoBark
* 支持Demucs
* 支持TripoSR、StableFast3D、Shap-E、SV34D和Zero123Plus进行3D生成
* 支持Wav2Lip
* 支持多模态（Moondream 2）、LORA（transformers）和WebSearch（使用GoogleSearch）用于LLM
* 界面内的模型设置
* 画廊
* ModelDownloader（用于LLM和StableDiffusion）
* 应用程序设置
* 能够查看系统传感器

## 必需依赖：

* [Python](https://www.python.org/downloads/)（3.11）
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads)（12.4）和[cuDNN](https://developer.nvidia.com/cudnn-downloads)（9.1）
* [FFMPEG](https://ffmpeg.org/download.html)
- C++编译器
  - Windows：[VisualStudio](https://visualstudio.microsoft.com/ru/)
  - Linux：[GCC](https://gcc.gnu.org/)

## 最低系统要求：

* 系统：Windows或Linux
* GPU：6GB+或CPU：8核3.2GHZ
* RAM：16GB+
* 磁盘空间：20GB+
* 互联网用于下载模型和安装

## 如何安装：

### Windows

1) 在任意位置执行`Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git`
2) 运行`Install.bat`并等待安装完成
3) 安装完成后，运行`Start.bat`
4) 选择文件版本并等待应用程序启动
5) 现在您可以开始生成了！

要获取更新，请运行`Update.bat`
要通过终端使用虚拟环境，请运行`Venv.bat`

### Linux

1) 在任意位置执行`Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git`
2) 在终端中运行`./Install.sh`并等待所有依赖项安装完成
3) 安装完成后，运行`./Start.sh`
4) 等待应用程序启动
5) 现在您可以开始生成了！

要获取更新，请运行`./Update.sh`
要通过终端使用虚拟环境，请运行`./Venv.sh`

## 如何使用：

#### 界面有六个主选项卡中的三十二个子选项卡 (文本, 图像, 视频, 3D, 音频 和 界面)：LLM、TTS-STT、SunoBark、LibreTranslate、Wav2Lip、StableDiffusion、Kandinsky、Flux、HunyuanDiT、Lumina-T2X、Kolors、AuraFlow、Würstchen、DeepFloydIF、PixArt、ModelScope、ZeroScope 2、CogVideoX、Latte、TripoSR、StableFast3D、Shap-E、SV34D、Zero123Plus、StableAudio、AudioCraft、AudioLDM 2、Demucs、Gallery、ModelDownloader、Settings和System。选择您需要的选项卡并按照以下说明操作

### LLM：

1) 首先将您的模型上传到文件夹：*inputs/text/llm_models*
2) 从下拉列表中选择您的模型
3) 选择模型类型（`transformers`或`llama`）
4) 根据您需要的参数设置模型
5) 键入（或说出）您的请求
6) 点击`Submit`按钮以接收生成的文本和音频响应
#### 可选：您可以启用`TTS`模式，选择所需的`voice`和`language`以接收音频响应。您可以启用`multimodal`并上传图像以获取其描述。您可以启用`websearch`以访问互联网。您可以启用`libretranslate`以获得翻译。您还可以选择`LORA`模型来改善生成
#### 语音样本 = *inputs/audio/voices*
#### LORA = *inputs/text/llm_models/lora*
#### 语音必须预处理（22050 kHz，单声道，WAV）

### TTS-STT：

1) 输入文本以进行文本转语音
2) 输入音频以进行语音转文本
3) 点击`Submit`按钮以接收生成的文本和音频响应
#### 语音样本 = *inputs/audio/voices*
#### 语音必须预处理（22050 kHz，单声道，WAV）

### SunoBark：

1) 输入您的请求
2) 根据您需要的参数设置模型
3) 点击`Submit`按钮以接收生成的音频响应

### LibreTranslate：

* 首先，您需要安装并运行[LibreTranslate](https://github.com/LibreTranslate/LibreTranslate)
1) 选择源语言和目标语言
2) 点击`Submit`按钮以获得翻译
#### 可选：您可以通过打开相应的按钮来保存翻译历史记录

### Wav2Lip：

1) 上传初始人脸图像
2) 上传初始语音音频
3) 根据您需要的参数设置模型
4) 点击`Submit`按钮以接收唇形同步结果

### StableDiffusion - 有十五个子选项卡：

#### txt2img：

1) 首先将您的模型上传到文件夹：*inputs/image/sd_models*
2) 从下拉列表中选择您的模型
3) 选择模型类型（`SD`、`SD2`或`SDXL`）
4) 根据您需要的参数设置模型
5) 输入您的请求（使用+和-进行提示权重）
6) 点击`Submit`按钮以获取生成的图像
#### 可选：您可以选择自己的`vae`、`embedding`和`lora`模型来改善生成方法，还可以启用`upscale`以增加生成图像的大小
#### vae = *inputs/image/sd_models/vae*
#### lora = *inputs/image/sd_models/lora*
#### embedding = *inputs/image/sd_models/embedding*

#### img2img：

1) 首先将您的模型上传到文件夹：*inputs/image/sd_models*
2) 从下拉列表中选择您的模型
3) 选择模型类型（`SD`、`SD2`或`SDXL`）
4) 根据您需要的参数设置模型
5) 上传将用于生成的初始图像
6) 输入您的请求（使用+和-进行提示权重）
7) 点击`Submit`按钮以获取生成的图像
#### 可选：您可以选择自己的`vae`模型
#### vae = *inputs/image/sd_models/vae*

#### depth2img：

1) 上传初始图像
2) 根据您需要的参数设置模型
3) 输入您的请求（使用+和-进行提示权重）
4) 点击`Submit`按钮以获取生成的图像

#### pix2pix：

1) 上传初始图像
2) 根据您需要的参数设置模型
3) 输入您的请求（使用+和-进行提示权重）
4) 点击`Submit`按钮以获取生成的图像

#### controlnet：

1) 首先将您的stable diffusion模型上传到文件夹：*inputs/image/sd_models*
2) 上传初始图像
3) 从下拉列表中选择您的stable diffusion和controlnet模型
4) 根据您需要的参数设置模型
5) 输入您的请求（使用+和-进行提示权重）
6) 点击`Submit`按钮以获取生成的图像

#### upscale(latent)：

1) 上传初始图像
2) 根据您需要的参数设置模型
3) 点击`Submit`按钮以获取放大的图像

#### upscale(Real-ESRGAN)：

1) 上传初始图像
2) 根据您需要的参数设置模型
3) 点击`Submit`按钮以获取放大的图像

#### inpaint：

1) 首先将您的模型上传到文件夹：*inputs/image/sd_models/inpaint*
2) 从下拉列表中选择您的模型
3) 选择模型类型（`SD`、`SD2`或`SDXL`）
4) 根据您需要的参数设置模型
5) 将用于生成的图像上传到`initial image`和`mask image`
6) 在`mask image`中，选择画笔，然后选择调色板并将颜色更改为`#FFFFFF`
7) 绘制生成区域并输入您的请求（使用+和-进行提示权重）
8) 点击`Submit`按钮以获取修复后的图像
#### 可选：您可以选择自己的`vae`模型
#### vae = *inputs/image/sd_models/vae*

#### gligen：

1) 首先将您的模型上传到文件夹：*inputs/image/sd_models*
2) 从下拉列表中选择您的模型
3) 选择模型类型（`SD`、`SD2`或`SDXL`）
4) 根据您需要的参数设置模型
5) 输入您的提示请求（使用+和-进行提示权重）和GLIGEN短语（在""中表示框）
6) 输入GLIGEN框（例如[0.1387, 0.2051, 0.4277, 0.7090]表示框）
7) 点击`Submit`按钮以获取生成的图像

#### animatediff：

1) 首先将您的模型上传到文件夹：*inputs/image/sd_models*
2) 从下拉列表中选择您的模型
3) 根据您需要的参数设置模型
4) 输入您的请求（使用+和-进行提示权重）
5) 点击`Submit`按钮以获取生成的图像动画

#### video：

1) 上传初始图像
2) 输入您的请求（适用于IV2Gen-XL）
3) 根据您需要的参数设置模型
4) 点击`Submit`按钮以从图像获取视频

#### ldm3d：

1) 输入您的请求
2) 根据您需要的参数设置模型
3) 点击`Submit`按钮以获取生成的图像

#### sd3 (txt2img, img2img, controlnet, inpaint)：

1) 输入您的请求
2) 根据您需要的参数设置模型
3) 点击`Submit`按钮以获取生成的图像

#### cascade：

1) 输入您的请求
2) 根据您需要的参数设置模型
3) 点击`Submit`按钮以获取生成的图像

#### extras：

1) 上传初始图像
2) 选择您需要的选项
3) 点击`Submit`按钮以获取修改后的图像

### Kandinsky (txt2img, img2img, inpaint)：

1) 输入您的提示
2) 从下拉列表中选择模型
3) 根据您需要的参数设置模型
4) 点击`Submit`以获取生成的图像

### Flux：

1) 输入您的提示
2) 从下拉列表中选择模型
3) 根据您需要的参数设置模型
4) 点击`Submit`以获取生成的图像

### HunyuanDiT：

1) 输入您的提示
2) 根据您需要的参数设置模型
3) 点击`Submit`以获取生成的图像

### Lumina-T2X：

1) 输入您的提示
2) 根据您需要的参数设置模型
3) 点击`Submit`以获取生成的图像

### Kolors：

1) 输入您的提示
2) 根据您需要的参数设置模型
3) 点击`Submit`以获取生成的图像

### AuraFlow：

1) 输入您的提示
2) 根据您需要的参数设置模型
3) 点击`Submit`以获取生成的图像

### Würstchen：

1) 输入您的提示
2) 根据您需要的参数设置模型
3) 点击`Submit`以获取生成的图像

### DeepFloydIF (txt2img, img2img, inpaint)：

1) 输入您的提示
2) 根据您需要的参数设置模型
3) 点击`Submit`以获取生成的图像

### PixArt：

1) 输入您的提示
2) 从下拉列表中选择模型
3) 根据您需要的参数设置模型
4) 点击`Submit`以获取生成的图像

### ModelScope：

1) 输入您的提示
2) 根据您需要的参数设置模型
3) 点击`Submit`以获取生成的视频

### ZeroScope 2：

1) 输入您的提示
2) 根据您需要的参数设置模型
3) 点击`Submit`以获取生成的视频

### CogVideoX：

1) 输入您的提示
2) 根据您需要的参数设置模型
3) 点击`Submit`以获取生成的视频

### Latte：

1) 输入您的提示
2) 根据您需要的参数设置模型
3) 点击`Submit`以获取生成的视频

### TripoSR：

1) 上传初始图像
2) 根据您需要的参数设置模型
3) 点击`Submit`按钮以获取生成的3D对象

### StableFast3D：

1) 上传初始图像
2) 根据您需要的参数设置模型
3) 点击`Submit`按钮以获取生成的3D对象

### Shap-E：

1) 输入您的请求或上传初始图像
2) 根据您需要的参数设置模型
3) 点击`Submit`按钮以获取生成的3D对象

### SV34D：

1) 上传初始图像（用于3D）或视频（用于4D）
2) 根据您需要的参数设置模型
3) 点击`Submit`按钮以获取生成的3D视频

### Zero123Plus：

1) 上传初始图像
2) 根据您需要的参数设置模型
3) 点击`Submit`按钮以获取生成的3D图像旋转

### StableAudio：

1) 根据您需要的参数设置模型
2) 输入您的请求
3) 点击`Submit`按钮以获取生成的音频

### AudioCraft：

1) 从下拉列表中选择模型
2) 选择模型类型（`musicgen`、`audiogen`或`magnet`）
3) 根据您需要的参数设置模型
4) 输入您的请求
5) （可选）如果您使用的是`melody`模型，请上传初始音频
6) 点击`Submit`按钮以获取生成的音频
#### 可选：您可以启用`multiband diffusion`来改善生成的音频

### AudioLDM 2：

1) 从下拉列表中选择模型
2) 根据您需要的参数设置模型
3) 输入您的请求
4) 点击`Submit`按钮以获取生成的音频

### Demucs：

1) 上传要分离的初始音频
2) 点击`Submit`按钮以获取分离后的音频

### Gallery：

* 您可以在这里查看outputs目录中的文件

### ModelDownloader：

* 您可以在这里下载`LLM`和`StableDiffusion`模型。只需从下拉列表中选择模型，然后点击`Submit`按钮
#### `LLM`模型下载到这里：*inputs/text/llm_models*
#### `StableDiffusion`模型下载到这里：*inputs/image/sd_models*

### Settings：

* 您可以在这里更改应用程序设置。目前，您只能将`Share`模式更改为`True`或`False`

### System：

* 您可以通过点击`Submit`按钮在这里查看计算机传感器的指标

### 附加信息：

1) 所有生成的内容都保存在*outputs*文件夹中
2) 您可以按`Clear`按钮重置您的选择
3) 要停止生成过程，请点击`Stop generation`按钮
4) 您可以使用`Close terminal`按钮关闭应用程序
5) 您可以通过点击`Outputs`按钮打开*outputs*文件夹

## 我在哪里可以获得模型和语音？

* LLM模型可以从[HuggingFace](https://huggingface.co/models)获取，或者通过界面内的ModelDownloader获取
* StableDiffusion、vae、inpaint、embedding和lora模型可以从[CivitAI](https://civitai.com/models)获取，或者通过界面内的ModelDownloader获取
* StableAudioOpen, AudioCraft, AudioLDM 2, TTS, Whisper, Wav2Lip, SunoBark, MoonDream2, Upscale, GLIGEN, Depth, Pix2Pix, Controlnet, AnimateDiff, Videos, LDM3D, SD3, Cascade, InstantID, PhotoMaker, IP-Adapter-FaceID, Rembg, Roop, CodeFormer, Real-ESRGAN, TripoSR, StableFast3D, Shap-E, SV34D, Zero123Plus, Demucs, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, AuraSR, Würstchen, DeepFloydIF, PixArt, ModelScope, ZeroScope 2, CogVideoX, Latte和Multiband diffusion模型在使用时会自动下载到*inputs*文件夹中
* 您可以从任何地方获取语音。录制您自己的声音或从互联网上获取录音。或者直接使用项目中已有的语音。主要是要经过预处理！

## Wiki

* https://github.com/Dartvauder/NeuroSandboxWebUI/wiki

## 致开发者的感谢

#### 非常感谢这些项目，因为正是通过他们的应用程序/库，我才能够创建我的应用程序：

首先，我要感谢[PyCharm](https://www.jetbrains.com/pycharm/)和[GitHub](https://desktop.github.com)的开发者。借助他们的应用程序，我能够创建并分享我的代码

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
* `PhotoMaker` - https://github.com/TencentARC/PhotoMaker
* `IP-Adapter` - https://github.com/tencent-ailab/IP-Adapter
* `PyNanoInstantMeshes` - https://github.com/vork/PyNanoInstantMeshes
* `CLIP` - https://github.com/openai/CLIP

## 第三方许可证：

#### 许多模型都有自己的使用许可证。在使用之前，我建议您熟悉它们：

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
* [InstantID](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
* [PhotoMaker-V2](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
* [IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID)
* [AuraSR](https://huggingface.co/fal/AuraSR/blob/main/LICENSE.md)

## 捐赠

### *如果您喜欢我的项目并想要捐赠，这里有一些捐赠选项。非常感谢您的支持！*

* 加密钱包(BEP-20) - 0x3d86bdb5f50b92d0d7Eb44F1a833acC5e91aAEcA

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Dartvauder/NeuroSandboxWebUI&type=Date)](https://star-history.com/#Dartvauder/NeuroSandboxWebUI&Date)
