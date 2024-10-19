## [Функции](/#Features) | [Зависимости](/#Required-Dependencies) | [Системные требования](/#Minimum-System-Requirements) | [Установка](/#How-to-install) | [Вики](/#Wiki) | [Благодарность разработчикам](/#Acknowledgment-to-developers) | [Лицензии](/#Third-Party-Licenses)

# ![main](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4ea0d891-8979-45ad-b052-626c41ae991a)
* В процессе разработки, но стабильно!
* [English](/README.md) | Русский | [漢語](/Readmes/README_ZH.md)

## Описание:

Простой и удобный интерфейс для использования различных моделей нейронных сетей. Вы можете общаться с LLM, используя текстовый, голосовой и визуальный ввод; использовать StableDiffusion, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF, PixArt и PlaygroundV2.5 для генерации изображений; ModelScope, ZeroScope 2, CogVideoX и Latte для генерации видео; StableFast3D, Shap-E, SV34D и Zero123Plus для генерации 3D-объектов; StableAudioOpen, AudioCraft и AudioLDM 2 для генерации музыки и аудио; CoquiTTS, MMS и SunoBark для преобразования текста в речь; OpenAI-Whisper и MMS для преобразования речи в текст; Wav2Lip для синхронизации губ; LivePortrait для анимации изображений; Roop для замены лиц; Rembg для удаления фона; CodeFormer для восстановления лиц; PixelOE для пикселизации изображений; DDColor для раскрашивания изображений; LibreTranslate и SeamlessM4Tv2 для перевода текста; Demucs и UVR для разделения аудиофайлов; RVC для преобразования голоса. Вы также можете просматривать файлы из выходной директории в галерее, загружать модели LLM и StableDiffusion, изменять настройки приложения внутри интерфейса и проверять системные датчики.

Цель проекта - создать максимально простое в использовании приложение для работы с моделями нейронных сетей.

### Текст: <img width="1114" alt="1ru" src="https://github.com/user-attachments/assets/fa2425e2-2528-44ae-90e0-d22bc9e4960e">

### Изображение: <img width="1115" alt="2ru" src="https://github.com/user-attachments/assets/739ee1e9-a5e6-46d5-b57e-1494bc52433b">

### Видео: <img width="1115" alt="3ru" src="https://github.com/user-attachments/assets/3557c766-1dec-47dd-b56d-2d8c24dd6aa2">

### 3D: <img width="1112" alt="4ru" src="https://github.com/user-attachments/assets/09f4ecc8-5098-45ce-8dd4-35ecf410ca35">

### Аудио: <img width="1111" alt="5ru" src="https://github.com/user-attachments/assets/73144d26-65fe-4a81-8d37-fdcf40757bac">

### Дополнительно: <img width="1112" alt="6ru" src="https://github.com/user-attachments/assets/5250b26e-67b4-4d6a-a2e9-ce6920459e4f">

### Интерфейс: <img width="1115" alt="7ru" src="https://github.com/user-attachments/assets/48d4a4a7-6ce6-494d-a17d-368ff4bcf368">

## Функции:

* Легкая установка через install.bat (Windows) или install.sh (Linux)
* Возможность использования приложения через мобильное устройство в локальной сети (через IPv4) или в интернете (через Share)
* Гибкий и оптимизированный интерфейс (на основе Gradio)
* Ведение журнала отладки в логах из файлов `Install` и `Update`
* Доступно на трех языках
* Поддержка моделей Transformers, BNB, GPTQ, AWQ, ExLlamaV2 и llama.cpp (LLM)
* Поддержка моделей diffusers и safetensors (StableDiffusion) - txt2img, img2img, depth2img, marigold, pix2pix, controlnet, upscale (latent), upscale (SUPIR), refiner, inpaint, outpaint, gligen, diffedit, blip-diffusion, animatediff, hotshot-xl, video, ldm3d, sd3, cascade, t2i-ip-adapter, ip-adapter-faceid и riffusion вкладки
* Поддержка моделей stable-diffusion-cpp для FLUX и Stable Diffusion
* Поддержка дополнительных моделей для генерации изображений: Kandinsky (txt2img, img2img, inpaint), Flux (txt2img с поддержкой cpp quantize и LoRA, img2img, inpaint, controlnet), HunyuanDiT (txt2img, controlnet), Lumina-T2X, Kolors (txt2img с поддержкой LoRA, img2img, ip-adapter-plus), AuraFlow (с поддержкой LoRA и AuraSR), Würstchen, DeepFloydIF (txt2img, img2img, inpaint), PixArt и PlaygroundV2.5
* Поддержка Extras с моделями Rembg, CodeFormer, PixelOE, DDColor, DownScale, Format changer, FaceSwap (Roop) и Upscale (Real-ESRGAN) для изображений, видео и аудио
* Поддержка StableAudio
* Поддержка AudioCraft (Модели: musicgen, audiogen и magnet)
* Поддержка AudioLDM 2 (Модели: audio и music)
* Поддержка моделей TTS и Whisper (Для LLM и TTS-STT)
* Поддержка MMS для преобразования текста в речь и речи в текст
* Поддержка моделей Lora, Textual inversion (embedding), Vae, MagicPrompt, Img2img, Depth, Marigold, Pix2Pix, Controlnet, Upscalers (latent и SUPIR), Refiner, Inpaint, Outpaint, GLIGEN, DiffEdit, BLIP-Diffusion, AnimateDiff, HotShot-XL, Videos, LDM3D, SD3, Cascade, T2I-IP-ADAPTER, IP-Adapter-FaceID и Riffusion (Для StableDiffusion)
* Поддержка модели Multiband Diffusion (Для AudioCraft)
* Поддержка LibreTranslate (Локальный API) и SeamlessM4Tv2 для переводов языков
* Поддержка ModelScope, ZeroScope 2, CogVideoX и Latte для генерации видео
* Поддержка SunoBark
* Поддержка Demucs и UVR для разделения аудиофайлов
* Поддержка RVC для преобразования голоса
* Поддержка StableFast3D, Shap-E, SV34D и Zero123Plus для 3D генерации
* Поддержка Wav2Lip
* Поддержка LivePortrait для анимации изображений
* Поддержка Multimodal (Moondream 2, LLaVA-NeXT-Video, Qwen2-Audio), PDF-Parsing (OpenParse), TTS (CoquiTTS), STT (Whisper), LORA и WebSearch (с DuckDuckGo) для LLM
* Просмотр MetaData-Info для сгенерированных изображений, видео и аудио
* Настройки моделей внутри интерфейса
* Онлайн и оффлайн Wiki
* Галерея
* ModelDownloader (Для LLM и StableDiffusion)
* Настройки приложения
* Возможность просмотра системных датчиков

## Необходимые зависимости:

* [Python](https://www.python.org/downloads/) (3.10.11)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (12.4) и [cuDNN](https://developer.nvidia.com/cudnn-downloads) (9.1)
* [FFMPEG](https://ffmpeg.org/download.html)
- C+ компилятор
  - Windows: [VisualStudio](https://visualstudio.microsoft.com/ru/), [VisualStudioCode](https://code.visualstudio.com) и [Cmake](https://cmake.org)
  - Linux: [GCC](https://gcc.gnu.org/), [VisualStudioCode](https://code.visualstudio.com) и [Cmake](https://cmake.org)

## Минимальные системные требования:

* Система: Windows или Linux
* GPU: 6GB+ или CPU: 8 ядер 3.6GHZ
* ОЗУ: 16GB+
* Место на диске: 20GB+
* Интернет для загрузки моделей и установки

## Как установить:

### Windows

1) Сначала установите все [Необходимые зависимости](/#Required-Dependencies)
2) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` в любое место
3) Запустите `Install.bat` и дождитесь завершения установки
4) После установки запустите `Start.bat`
5) Дождитесь запуска приложения
6) Теперь вы можете начать генерацию!

Для получения обновлений запустите `Update.bat`
Для работы с виртуальной средой через терминал запустите `Venv.bat`

### Linux

1) Сначала установите все [Необходимые зависимости](/#Required-Dependencies)
2) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` в любое место
3) В терминале запустите `./Install.sh` и дождитесь установки всех зависимостей
4) После установки запустите `./Start.sh`
5) Дождитесь запуска приложения
6) Теперь вы можете начать генерацию!

Для получения обновлений запустите `./Update.sh`
Для работы с виртуальной средой через терминал запустите `./Venv.sh`

## Вики

* https://github.com/Dartvauder/NeuroSandboxWebUI/wiki/RU‐Wiki

## Благодарность разработчикам

#### Большое спасибо этим проектам, потому что благодаря их приложениям/библиотекам я смог создать свое приложение:

Прежде всего, я хочу поблагодарить разработчиков [PyCharm](https://www.jetbrains.com/pycharm/) и [GitHub](https://desktop.github.com). С помощью их приложений я смог создать и поделиться своим кодом

* `gradio` - https://github.com/gradio-app/gradio
* `transformers` - https://github.com/huggingface/transformers
* `auto-gptq` - https://github.com/AutoGPTQ/AutoGPTQ
* `autoawq` - https://github.com/casper-hansen/AutoAWQ
* `exllamav2` - https://github.com/turboderp/exllamav2
* `tts` - https://github.com/coqui-ai/TTS
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

## Лицензии третьих сторон:

#### Многие модели имеют свои собственные лицензии на использование. Перед использованием я советую вам ознакомиться с ними:

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
* [Riffusion](https://huggingface.co/spaces/CompVis/stable-diffusion-license)
* [MozillaCommonVoice17](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/cc0-1.0.md)
* [UVR-MDX](https://github.com/kuielab/mdx-net/blob/main/LICENSE)
* [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
* [DDColor](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
* [PixelOE](https://github.com/KohakuBlueleaf/PixelOE/blob/main/LICENSE)
* [LivePortrait](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)
* [SUPIR](https://github.com/Fanghua-Yu/SUPIR/blob/master/LICENSE)
* [MagicPrompt](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)
* [Marigold](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
* [BLIP-Diffusion](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
* [Consistency-Decoder](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)
* [Tiny-AutoEncoder](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)

#### Эти сторонние репозитории кода также используются в моем проекте:

* [Generative-Models для SV34D](https://github.com/Stability-AI/generative-models)
* [CodeFormer для extras](https://github.com/sczhou/CodeFormer)
* [Real-ESRGAN для upscale](https://github.com/xinntao/Real-ESRGAN)
* [HotShot-XL для StableDiffusion](https://github.com/hotshotco/Hotshot-XL)
* [Roop для extras](https://github.com/s0md3v/roop)
* [StableFast3D для 3D](https://github.com/Stability-AI/stable-fast-3d)
* [Riffusion для StableDiffusion](https://github.com/riffusion/riffusion-hobby)
* [DDColor для extras](https://github.com/piddnad/DDColor)
* [LivePortrait для видео](https://github.com/KwaiVGI/LivePortrait)
* [SUPIR для StableDiffusion](https://github.com/Fanghua-Yu/SUPIR)
* [TAESD для StableDiffusion и Flux](https://github.com/madebyollin/taesd)

## Пожертвование

### *Если вам понравился мой проект и вы хотите сделать пожертвование, вот варианты для этого. Заранее большое спасибо!*

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)

## История звезд

[![Star History Chart](https://api.star-history.com/svg?repos=Dartvauder/NeuroSandboxWebUI&type=Date)](https://star-history.com/#Dartvauder/NeuroSandboxWebUI&Date)
