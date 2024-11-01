# 使用方法：

#### 界面有七个主选项卡（文本、图像、视频、3D、音频、附加功能和界面），共四十一个子选项卡（部分带有自己的子选项卡）：LLM、TTS-STT、MMS、SeamlessM4Tv2、LibreTranslate、StableDiffusion、Kandinsky、Flux、HunyuanDiT、Lumina-T2X、Kolors、AuraFlow、Würstchen、DeepFloydIF、PixArt, CogView3-Plus, PlaygroundV2.5、Wav2Lip、LivePortrait、ModelScope、ZeroScope 2、CogVideoX、Latte、StableFast3D、Shap-E、Zero123Plus、StableAudio、AudioCraft、AudioLDM 2、SunoBark、RVC、UVR、Demucs、Upscale (Real-ESRGAN)、FaceSwap、MetaData-Info、Wiki、Gallery、ModelDownloader、Settings和System。选择您需要的选项卡并按照以下说明操作

# 文本：

### LLM：

1) 首先将您的模型上传到文件夹：*inputs/text/llm_models*
2) 从下拉列表中选择您的模型
3) 选择模型类型
4) 根据您需要的参数设置模型
5) 输入（或说出）您的请求
6) 点击`Submit`按钮接收生成的文本和音频响应
#### 可选：您可以启用 TTS 模式，选择所需的 声音 和 语言 以接收音频回复。您可以启用 多模态 并上传图像、视频和音频文件以获取其描述。您可以启用 网络搜索 以访问互联网。您可以启用 libretranslate 以获得翻译。您可以启用 OpenParse 以处理 PDF 文件。此外，您还可以选择 LORA 模型来改善生成。
#### 语音样本 = *inputs/audio/voices*
#### LORA = *inputs/text/llm_models/lora*
#### 语音必须预处理（22050 kHz，单声道，WAV）
#### LLM的头像，您可以在*avatars*文件夹中更改

### TTS-STT：

1) 输入文本进行文本到语音转换
2) 输入音频进行语音到文本转换
3) 点击`Submit`按钮接收生成的文本和音频响应
#### 语音样本 = *inputs/audio/voices*
#### 语音必须预处理（22050 kHz，单声道，WAV）

### MMS（文本到语音和语音到文本）：

1) 输入文本进行文本到语音转换
2) 输入音频进行语音到文本转换
3) 点击`Submit`按钮接收生成的文本或音频响应

### SeamlessM4Tv2：

1) 输入（或说出）您的请求
2) 选择源语言、目标语言和数据集语言
3) 根据您需要的参数设置模型
4) 点击`Submit`按钮获取翻译

### LibreTranslate：

* 首先您需要安装并运行[LibreTranslate](https://github.com/LibreTranslate/LibreTranslate)
1) 选择源语言和目标语言
2) 点击`Submit`按钮获取翻译
#### 可选：您可以通过打开相应的按钮来保存翻译历史记录

# 图像：

### StableDiffusion - 有二十四个子选项卡：

#### txt2img：

1) 首先将您的模型上传到文件夹：*inputs/image/sd_models*
2) 从下拉列表中选择您的模型
3) 选择模型类型（`SD`、`SD2`或`SDXL`）
4) 根据您需要的参数设置模型
5) 输入您的请求（+和-用于提示权重）
6) 点击`Submit`按钮获取生成的图像
#### 可选：您可以选择您的`vae`、`embedding`和`lora`模型，还可以启用`MagicPrompt`来改进生成方法
#### vae = *inputs/image/sd_models/vae*
#### lora = *inputs/image/sd_models/lora*
#### embedding = *inputs/image/sd_models/embedding*

#### img2img：

1) 首先将您的模型上传到文件夹：*inputs/image/sd_models*
2) 从下拉列表中选择您的模型
3) 选择模型类型（`SD`、`SD2`或`SDXL`）
4) 根据您需要的参数设置模型
5) 上传将进行生成的初始图像
6) 输入您的请求（+和-用于提示权重）
7) 点击`Submit`按钮获取生成的图像
#### 可选：您可以选择您的`vae`、`embedding`和`lora`模型，还可以启用`MagicPrompt`来改进生成方法
#### vae = *inputs/image/sd_models/vae*
#### lora = *inputs/image/sd_models/lora*
#### embedding = *inputs/image/sd_models/embedding*

#### depth2img：

1) 上传初始图像
2) 根据您需要的参数设置模型
3) 输入您的请求（+和-用于提示权重）
4) 点击`Submit`按钮获取生成的图像

#### marigold：

1) 上传初始图像
2) 根据您需要的参数设置模型
3) 点击`Submit`按钮获取生成的深度图像

#### pix2pix：

1) 上传初始图像
2) 根据您需要的参数设置模型
3) 输入您的请求（+和-用于提示权重）
4) 点击`Submit`按钮获取生成的图像

#### controlnet：

1) 首先将您的stable diffusion模型上传到文件夹：*inputs/image/sd_models*
2) 上传初始图像
3) 从下拉列表中选择您的stable diffusion和controlnet模型
4) 根据您需要的参数设置模型
5) 输入您的请求（+和-用于提示权重）
6) 点击`Submit`按钮获取生成的图像

#### upscale（潜在）：

1) 上传初始图像
2) 选择您的模型
3) 根据您需要的参数设置模型
4) 点击`Submit`按钮获取放大的图像

#### refiner（SDXL）：

1) 上传初始图像
2) 点击`Submit`按钮获取精修后的图像

#### inpaint：

1) 首先将您的模型上传到文件夹：*inputs/image/sd_models/inpaint*
2) 从下拉列表中选择您的模型
3) 选择模型类型（`SD`、`SD2`或`SDXL`）
4) 根据您需要的参数设置模型
5) 将要进行生成的图像上传到`initial image`和`mask image`
6) 在`mask image`中，选择画笔，然后选择调色板并将颜色更改为`#FFFFFF`
7) 绘制生成区域并输入您的请求（+和-用于提示权重）
8) 点击`Submit`按钮获取修复后的图像
#### 可选：您可以选择您的`vae`模型来改进生成方法
#### vae = *inputs/image/sd_models/vae*

#### outpaint：

1) 首先将您的模型上传到文件夹：*inputs/image/sd_models/inpaint*
2) 从下拉列表中选择您的模型
3) 选择模型类型（`SD`、`SD2`或`SDXL`）
4) 根据您需要的参数设置模型
5) 将要进行生成的图像上传到`initial image`
6) 输入您的请求（+和-用于提示权重）
7) 点击`Submit`按钮获取扩展后的图像

#### gligen：

1) 首先将您的模型上传到文件夹：*inputs/image/sd_models*
2) 从下拉列表中选择您的模型
3) 选择模型类型（`SD`、`SD2`或`SDXL`）
4) 根据您需要的参数设置模型
5) 输入您的提示请求（+和-用于提示权重）和GLIGEN短语（在""中表示框）
6) 输入GLIGEN框（例如[0.1387, 0.2051, 0.4277, 0.7090]表示一个框）
7) 点击`Submit`按钮获取生成的图像

#### diffedit：

1) 输入您的源提示和源负面提示以进行图像遮罩
2) 输入您的目标提示和目标负面提示以进行图像差异编辑
3) 上传初始图像
4) 根据您需要的参数设置模型
5) 点击`Submit`按钮获取生成的图像

#### blip-diffusion：

1) 输入您的提示
2) 上传初始图像
3) 输入您的条件和目标主题
4) 根据您需要的参数设置模型
5) 点击`Submit`按钮获取生成的图像

#### animatediff：

1) 首先将您的模型上传到文件夹：*inputs/image/sd_models*
2) 从下拉列表中选择您的模型
3) 根据您需要的参数设置模型
4) 输入您的请求（+和-用于提示权重）
5) 点击`Submit`按钮获取生成的图像动画
#### 可选：您可以选择运动LORA来控制生成

#### hotshot-xl

1) 输入您的请求
2) 根据您需要的参数设置模型
3) 点击`Submit`按钮获取生成的GIF图像

#### video：

1) 上传初始图像
2) 选择您的模型
3) 输入您的请求（适用于IV2Gen-XL）
4) 根据您需要的参数设置模型
5) 点击`Submit`按钮获取从图像生成的视频

#### ldm3d：

1) 输入您的请求
2) 根据您需要的参数设置模型
3) 点击`Submit`按钮获取生成的图像

#### sd3（txt2img、img2img、controlnet、inpaint）：

1) 输入您的请求
2) 根据您需要的参数设置模型
3) 点击`Submit`按钮获取生成的图像
#### 可选：您可以选择您的 `lora` 模型以改进生成方法。如果您的显存较低，可以通过单击 `Enable quantize` 按钮来使用量化模型，但您需要自己下载模型: [CLIP-L](https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/clip_l.safetensors), [CLIP-G](https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/clip_g.safetensors)和[T5XXL](https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/t5xxl_fp16.safetensors)
#### lora = *inputs/image/sd_models/lora*
#### 量化模型 = *inputs/image/sd_models*

#### cascade：

1) 输入您的请求
2) 根据您需要的参数设置模型
3) 点击`Submit`按钮获取生成的图像

#### t2i-ip-adapter：

1) 上传初始图像
2) 选择您需要的选项
3) 点击`Submit`按钮获取修改后的图像

#### ip-adapter-faceid：

1) 上传初始图像
2) 选择您需要的选项
3) 点击`Submit`按钮获取修改后的图像

#### riffusion（文本到图像、图像到音频、音频到图像）：

- 文本到图像：
  - 1) 输入您的请求
    2) 根据您需要的参数设置模型
    3) 点击`Submit`按钮获取生成的图像
- 图像到音频：
  - 1) 上传初始图像
    2) 选择您需要的选项
    3) 点击`Submit`按钮获取从图像生成的音频
- 音频到图像：
  - 1) 上传初始音频
    2) 选择您需要的选项
    3) 点击`Submit`按钮获取从音频生成的图像
   
### Kandinsky（txt2img、img2img、inpaint）：

1) 输入您的提示
2) 从下拉列表中选择一个模型
3) 根据您需要的参数设置模型
4) 点击`Submit`获取生成的图像

### Flux (txt2img, img2img, inpaint, controlnet):

1) 输入您的提示
2) 选择您的模型
3) 根据您需要的参数设置模型
4) 点击`Submit`获取生成的图像
#### 可选：您可以选择您的`lora`模型来改进生成方法。如果您的VRAM较低，还可以通过点击`Enable quantize`按钮使用量化模型，但您需要自行下载模型：[FLUX.1-dev](https://huggingface.co/city96/FLUX.1-dev-gguf/tree/main)或[FLUX.1-schnell](https://huggingface.co/city96/FLUX.1-schnell-gguf/tree/main)，以及[VAE](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors)、[CLIP](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors)和[T5XXL](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors)
#### lora = *inputs/image/flux-lora*
#### 量化模型 = *inputs/image/quantize-flux*

### HunyuanDiT（txt2img、controlnet）：

1) 输入您的提示
2) 根据您需要的参数设置模型
3) 点击`Submit`获取生成的图像

### Lumina-T2X：

1) 输入您的提示
2) 根据您需要的参数设置模型
3) 点击`Submit`获取生成的图像

### Kolors（txt2img、img2img、ip-adapter-plus）：

1) 输入您的提示
2) 根据您需要的参数设置模型
3) 点击`Submit`获取生成的图像
#### 可选：您可以选择您的`lora`模型来改进生成方法
#### lora = *inputs/image/kolors-lora*

### AuraFlow：

1) 输入您的提示
2) 根据您需要的参数设置模型
3) 点击`Submit`获取生成的图像
#### 可选：您可以选择您的`lora`模型并启用`AuraSR`来改进生成方法
#### lora = *inputs/image/auraflow-lora*

### Würstchen：

1) 输入您的提示
2) 根据您需要的参数设置模型
3) 点击`Submit`获取生成的图像

### DeepFloydIF（txt2img、img2img、inpaint）：

1) 输入您的提示
2) 根据您需要的参数设置模型
3) 点击`Submit`获取生成的图像

### PixArt：

1) 输入您的提示
2) 选择您的模型
3) 根据您需要的参数设置模型
4) 点击`Submit`获取生成的图像

### CogView3-Plus:

1) 输入您的提示
2) 根据您需要的参数设置模型
3) 点击`Submit`获取生成的图像

### PlaygroundV2.5：

1) 输入您的提示
2) 根据您需要的参数设置模型
3) 点击`Submit`获取生成的图像

# 视频：

### Wav2Lip：

1) 上传初始面部图像
2) 上传初始语音音频
3) 根据您需要的参数设置模型
4) 点击`Submit`按钮接收唇形同步结果

### LivePortrait：

1) 上传初始面部图像
2) 上传初始面部移动视频
3) 点击`Submit`按钮接收动画面部图像

### ModelScope：

1) 输入您的提示
2) 根据您需要的参数设置模型
3) 点击`Submit`获取生成的视频

### ZeroScope 2：

1) 输入您的提示
2) 根据您需要的参数设置模型
3) 点击`Submit`获取生成的视频

### CogVideoX (text2video, image2video, video2video):

1) 输入您的提示
2) 根据您需要的参数设置模型
3) 点击`Submit`获取生成的视频

### Latte：

1) 输入您的提示
2) 根据您需要的参数设置模型
3) 点击`Submit`获取生成的视频

# 3D：

### StableFast3D：

1) 上传初始图像
2) 根据您需要的参数设置模型
3) 点击`Submit`按钮获取生成的3D对象

### Shap-E：

1) 输入您的请求或上传初始图像
2) 根据您需要的参数设置模型
3) 点击`Submit`按钮获取生成的3D对象

### Zero123Plus：

1) 上传初始图像
2) 根据您需要的参数设置模型
3) 点击`Submit`按钮获取生成的图像3D旋转

# 音频：

### StableAudio：

1) 根据您需要的参数设置模型
2) 输入您的请求
3) 点击`Submit`按钮获取生成的音频

### AudioCraft：

1) 从下拉列表中选择一个模型
2) 选择模型类型（`musicgen`、`audiogen`或`magnet`）
3) 根据您需要的参数设置模型
4) 输入您的请求
5) （可选）如果您使用`melody`模型，请上传初始音频
6) 点击`Submit`按钮获取生成的音频
#### 可选：您可以启用`multiband diffusion`来改进生成的音频

### AudioLDM 2：

1) 从下拉列表中选择一个模型
2) 根据您需要的参数设置模型
3) 输入您的请求
4) 点击`Submit`按钮获取生成的音频

### SunoBark：

1) 输入您的请求
2) 根据您需要的参数设置模型
3) 点击`Submit`按钮接收生成的音频响应

### RVC：

1) 首先将您的模型上传到文件夹：*inputs/audio/rvc_models*
2) 上传初始音频
3) 从下拉列表中选择您的模型
4) 根据您需要的参数设置模型
5) 点击`Submit`按钮接收生成的语音克隆

### UVR：

1) 上传要分离的初始音频
2) 点击`Submit`按钮获取分离后的音频

### Demucs：

1) 上传要分离的初始音频
2) 点击`Submit`按钮获取分离后的音频

# 附加功能（图像、视频、音频）：

1) 上传初始文件
2) 选择您需要的选项
3) 点击`Submit`按钮获取修改后的文件

### Upscale（Real-ESRGAN）：

1) 上传初始图像
2) 选择您的模型
3) 根据您需要的参数设置模型
4) 点击`Submit`按钮获取放大后的图像

### FaceSwap：

1) 上传源面部图像
2) 上传目标面部图像或视频
3) 选择您需要的选项
4) 点击`Submit`按钮获取换脸后的图像
#### 可选：您可以启用FaceRestore来放大和恢复您的面部图像/视频

### MetaData-Info：

1) 上传生成的文件
2) 点击`Submit`按钮获取文件的元数据信息

# 界面：

### Wiki：

* 在这里您可以查看项目的在线或离线wiki

### Gallery：

* 在这里您可以查看outputs目录中的文件

### ModelDownloader：

* 在这里您可以下载`LLM`和`StableDiffusion`模型

### Settings：

* 在这里您可以更改应用程序设置

### System：

* 在这里您可以查看计算机传感器的指标

### 附加信息：

1) 所有生成的内容都保存在*outputs*文件夹中。您可以使用`Outputs`按钮打开*outputs*文件夹
2) 您可以使用`Close terminal`按钮关闭应用程序

## 我在哪里可以获取模型和语音？

* LLM模型可以从[HuggingFace](https://huggingface.co/models)获取，或者从界面内的ModelDownloader获取
* StableDiffusion、vae、inpaint、embedding和lora模型可以从[CivitAI](https://civitai.com/models)获取，或者从界面内的ModelDownloader获取
* RVC模型可以从[VoiceModels](https://voice-models.com)获取
* StableAudio、AudioCraft、AudioLDM 2、TTS、Whisper、MMS、SeamlessM4Tv2、Wav2Lip、LivePortrait、SunoBark、MoonDream2、Upscalers（Latent和Real-ESRGAN）、Refiner、GLIGEN、DiffEdit、BLIP-Diffusion、Depth、Marigold、Pix2Pix、Controlnet、AnimateDiff、HotShot-XL、Videos、LDM3D、SD3、Cascade、T2I-IP-ADAPTER、IP-Adapter-FaceID、Riffusion、Rembg、Roop、CodeFormer、DDColor、PixelOE、Real-ESRGAN、StableFast3D、Shap-E、Zero123Plus、UVR、Demucs、Kandinsky、Flux、HunyuanDiT、Lumina-T2X、Kolors、AuraFlow、AuraSR、Würstchen、DeepFloydIF、PixArt、CogView3-Plus, PlaygroundV2.5、ModelScope、ZeroScope 2、CogVideoX、MagicPrompt、Latte和Multiband diffusion模型在使用时会自动下载到*inputs*文件夹中
* 您可以从任何地方获取语音。录制您自己的声音或从互联网上获取录音。或者直接使用项目中已有的语音。主要是要经过预处理！

## 路线图和错误追踪器：

[DiscussionLink](https://github.com/Dartvauder/NeuroSandboxWebUI/discussions/248)
