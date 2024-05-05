## [Функции](/#Функции) | [Зависимости](/#Необходимые-зависимости) | [СистемныеТребования](/#Минимальные-системные-требования) | [Установка](/#Как-установить) | [Использование](/#Как-использовать) | [Модели](/#Где-я-могу-взять-модели-голоса-и-аватары) | [Дорожная карта](/#Дорожная-карта) | [Благодарность](/#Благодарность-разработчикам) [Лицензии](/#Сторонние-лицензии)

# ![main](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4ea0d891-8979-45ad-b052-626c41ae991a)
* В процессе разработки! (АЛЬФА)
* [Английский](/README.md) | Russian

## Описание:

Простой и удобный интерфейс для использования различных моделей нейронных сетей. Вы можете общаться с LLM и Moondream2, используя текстовый или голосовой ввод и загрузив изображение, использовать StableDiffusion для генерации изображений, ZeroScope 2 для генерации видео, Shap-E для генерации 3Д обьектов, AudioCraft и AudioLDM 2 для генерации музыки и аудио, CoquiTTS и SunoBark для преобразования текста в речь, OpenAI-Whisper для преобразования речи в текст, LibreTranslate для перевода текста и Demucs для сепарации аудио файлов. Также вы можете скачать модели LLM и StableDiffusion, изменить настройки приложения внутри интерфейса и проверить датчики системы

Цель проекта — создать максимально простое приложение для использования нейросетевых моделей

### LLM: ![1](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/2e7c1e23-80a9-4937-8076-a62f256e8d12)

### TTS-STT: ![2](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/0b94bb40-4d38-4ebd-9f1e-86931614ee32)

### SunoBark: ![3](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/01a8b3cb-0d62-49ad-88b0-fe3fe68f98ce)

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

## Функции:

* Простая установка через install.bat(Windows) или install.sh(Linux)
* Вы можете использовать приложение через мобильное устройство в localhost (Через IPv4)
* Гибкий и оптимизированный интерфейс (От Gradio)
* Поддержка Transformers и llama.cpp моделей (LLM)
* Поддержка diffusers и safetensors моделей (StableDiffusion) - Вкладки txt2img, img2img, depth2img, upscale, inpaint, animatediff, video, cascade и extras
* Поддержка AudioCraft (Модели: musicgen, audiogen и magnet)
* Поддержка AudioLDM 2
* Поддержка TTS и Whisper моделей (Для LLM и TTS-STT)
* Поддержка Lora, Textual inversion (embedding), Vae, Img2img, Depth, Upscale, Inpaint, AnimateDiff, Videos и Cascade моделей (Для StableDiffusion)
* Поддержка Multiband Diffusion модели (Для AudioCraft)
* Поддержка LibreTranslate (Локальный API)
* Поддержка ZeroScope 2
* Поддержка SunoBark
* Поддержка Demucs
* Поддержка Rembg (Для StableDiffusion)
* Поддержка Shap-E
* Поддержка multimodal (Moondream 2 для LLM)
* Поддержка WebSearch (Для LLM через googleSearch)
* Настройки моделей внутри интерфейса
* ModelDownloader (Для LLM и StableDiffusion)
* Настройки приложения
* Возможность видеть датчики системы

## Необходимые зависимости:

* [Python](https://www.python.org/downloads/) (3.10+)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (12.1)
* [FFMPEG](https://ffmpeg.org/download.html)
- C+ компилятор
  - Windows: [VisualStudio](https://visualstudio.microsoft.com/ru/)
  - Linux: [GCC](https://gcc.gnu.org/)

## Минимальные системные требования:

* Система: Windows или Linux
* Графический процессор: 6ГБ+ или центральный процессор: 8 ядер, 3,2ГГц
* Оперативная память: 16ГБ+
* Пространство на жестком диске: 20ГБ+
* Интернет для скачивания моделей и установки

## Как установить:

### Windows

1) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` в любое место (Без кириллицы)
2) Запускаем `install.bat` и ждем когда все установиться
3) После установки запустите `start.bat`
4) Выберите версию файла и дождитесь запуска приложения
5) Теперь вы можете приступать к генерациям!

Чтобы получить обновление, запустите `update.bat`

### Linux

1) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` в любое место (Без кириллицы)
2) В терминале запустите `./install.sh` и дождитесь установки всех зависимостей
3) После установки, в терминале запустите `./start.sh`
4) Дождитесь запуска приложения
5) Теперь вы можете приступать к генерациям!

Чтобы получить обновление, запустите `./update.sh`

## Как использовать:

#### Интерфейс имеет Тринадцать вкладок: LLM, TTS-STT, SunoBark, LibreTranslate, StableDiffusion, ZeroScope 2, Shap-E, AudioCraft, AudioLDM 2, Demucs, ModelDownloader, Settings и System. Выберите ту которая вам нужна и следуйте инструкциям ниже

### LLM:

1) Сначала загрузите ваши модели в папку: *inputs/text/llm_models*
2) Выберите вашу модель из выпадающего списка
3) Выберите тип модели (`transformers` или `llama`)
4) Настройте модель по нужным вам параметрам
5) Введите (или произнесите) ваш запрос
6) Нажмите кнопку `Submit` чтобы получить сгенерированный текстовый и аудио ответ
#### Дополнительно: вы можете включить режим `TTS`, выбрать `голос` и `язык` необходимые для получения аудио ответа. Так же вы можете включить `multimodal` и загрузить изображение чтобы получить его описание
#### Образцы голоса = *inputs/audio/voices*
#### Голос должен быть предварительно обработан (22050 кГц, монозвук, WAV)

### TTS-STT:

1) Введите текст для преобразования текста в речь
2) Введите звук для преобразования речи в текст
3) Нажмите кнопку `Submit`, чтобы получить сгенерированный текстовый и аудио ответ
#### Образцы голоса = *inputs/audio/voices*
#### Голос должен быть предварительно обработан (22050 кГц, монозвук, WAV)

### SunoBark:

1) Введите ваш запрос
2) Настройте модель по нужным вам параметрам
3) Нажмите кнопку `Submit` чтобы получить сгенерированный аудио ответ

### LibreTranslate:

* Сначала вам нужно установить и запустить [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate)
1) Выберите исходный и целевой язык
2) (Дополнительно) Вы можете сохранить историю перевода включив соответствующую кнопку
3) Нажмите кнопку `Submit`, чтобы получить перевод

### StableDiffusion - имеет девять под-вкладок:

#### txt2img:

1) Сначала загрузите ваши модели в папку: *inputs/image/sd_models*
2) Выберите вашу модель из выпадающего списка
3) Выберите тип модели (`SD`, `SD2` или `SDXL`)
4) Настройте модель по нужным вам параметрам
5) Введите ваш запрос
6) Нажмите кнопку `Submit`, чтобы получить сгенерированное изображение
#### Дополнительно: вы можете выбрать свои модели `vae`, `embedding` и `lora` чтобы улучшить метод генерации, а так же включить `upscale` чтобы увеличить размер сгенерированного изображения
#### vae = *inputs/image/sd_models/vae*
#### lora = *inputs/image/sd_models/lora*
#### embedding = *inputs/image/sd_models/embedding*

#### img2img:

1) Сначала загрузите ваши модели в папку: *inputs/image/sd_models*
2) Выберите вашу модель из выпадающего списка
3) Выберите тип модели (`SD`, `SD2` или `SDXL`)
4) Настройте модель по нужным вам параметрам
5) Загрузите исходное изображение, с которым будет происходить генерация
6) Введите ваш запрос
7) Нажмите кнопку `Submit`, чтобы получить сгенерированное изображение
#### Дополнительно: вы можете выбрать свою модель `vae`
#### vae = *inputs/image/sd_models/vae*

#### depth2img:

1) Загрузите исходное изображение
2) Настройте модель по нужным вам параметрам
3) Введите ваш запрос
4) Нажмите кнопку `Submit`, чтобы получить сгенерированное изображение

#### upscale:

1) Загрузите исходное изображение
2) Настройте модель по нужным вам параметрам
3) Нажмите кнопку `Submit`, чтобы получить увеличенное изображение

#### inpaint:

1) Сначала загрузите ваши модели в папку: *inputs/image/sd_models/inpaint*
2) Выберите вашу модель из выпадающего списка
3) Выберите тип модели (`SD`, `SD2` или `SDXL`)
4) Настройте модель по нужным вам параметрам
5) Загрузите изображение, с которым будет происходить генерация, в `initial image` и `mask image`
6) В `mask image` выберите кисть, затем палитру и измените цвет на `#FFFFFF`
7) Нарисуйте место для генерации и введите ваш запрос
8) Нажмите кнопку `Submit`, чтобы получить измененное изображение
#### Дополнительно: вы можете выбрать свою модель `vae`
#### vae = *inputs/image/sd_models/vae*

#### animatediff:

1) Сначала загрузите ваши модели в папку: *inputs/image/sd_models*
2) Выберите вашу модель из выпадающего списка
3) Настройте модель по нужным вам параметрам
8) Нажмите кнопку `Submit`, чтобы получить анимированное изображение

#### video:

1) Загрузите исходное изображение
2) Настройте модель по нужным вам параметрам
3) Нажмите кнопку `Submit`, чтобы получить видео из изображения

#### cascade:

1) Введите ваш запрос
2) Настройте модель по нужным вам параметрам
3) Нажмите кнопку `Submit`, чтобы получить сгенерированное изображение 

#### extras:

1) Загрузите исходное изображение
2) Выберите нужные вам опции
3) Нажмите кнопку `Submit`, чтобы получить измененное изображение

### ZeroScope 2:

1) Введите ваш запрос
2) Настройте модель по нужным вам параметрам
3) Нажмите кнопку `Submit`, чтобы получить сгенерированное видео

### Shap-E:

1) Введите запрос или загрузите исходное изображение
2) Настройте модель по нужным вам параметрам
3) Нажмите кнопку `Submit`, чтобы получить сгенерированный 3D-объект.

### AudioCraft:

1) Выберите модель из выпадающего списка
2) Выберите тип модели (`musicgen` или `audiogen`)
3) Настройте модель по нужным вам параметрам
4) Введите ваш запрос
5) (Дополнительно) загрузите исходный звук, если вы используете модель `melody`
6) Нажмите кнопку `Submit`, чтобы получить сгенерированное аудио
#### Дополнительно: вы можете включить `multiband diffusion`, чтобы улучшить генерируемый звук

### AudioLDM 2:

1) Выберите модель из выпадающего списка
2) Настройте модель по нужным вам параметрам
3) Введите ваш запрос
4) Нажмите кнопку `Submit`, чтобы получить сгенерированное аудио

### Demucs:

1) Загрузите исходный звук для сепарации
2) Нажмите кнопку `Submit`, чтобы получить сепарированное аудио

### ModelDownloader:

* Здесь вы можете скачать модели `LLM` и `StableDiffusion`. Просто выберите модель из выпадающего списка и нажмите кнопку `Submit`
#### Модели `LLM` скачиваються сюда: *inputs/text/llm_models*
#### Модели `StableDiffusion` скачиваються сюда: *inputs/image/sd_models*

### Settings:

* Здесь вы можете изменить настройки приложения. На данный момент вы можете изменить только режим `Share` на `True` или `False`

### System: 

* Здесь вы можете увидеть показатели датчиков вашего компьютера нажав на кнопку `Submit`

### Общее:

1) Все генерации сохраняются в папке *outputs*
2) Вы также можете нажать кнопку `Clear`, чтобы сбросить ваш выбор
3) Чтобы остановить процесс генерации, нажмите кнопку `Stop generation`
4) Вы также можете выключить приложение с помощью кнопки `Close terminal`
5) Вы можете открыть папку *outputs* нажав на кнопку `Folder`

## Где я могу взять модели, голоса и аватары?

* LLM модели можно взять с сайта [HuggingFace](https://huggingface.co/models) или из внутреннего интерфейса ModelDownloader
* Модели StableDiffusion, vae, inpaint, embedding и lora можно взять с сайта [CivitAI](https://civitai.com/models) или из внутреннего интерфейса ModelDownloader
* Модели AudioCraft и AudioLDM 2 загружаются автоматически в папку *inputs*, когда вы выбираете модель и нажимаете кнопку `submit`
* TTS, Whisper, SunoBark, MoonDream2, Upscale, Depth, AnimateDiff, Videos, Cascade, Rembg, Shap-E, Demucs, ZeroScope и Multiband diffusion модели скачиваються автоматически в папку *inputs* при их использовании
* Вы можете использовать голоса откуда угодно. Запишите свой или возьмите запись из интернета. Или просто используйте те, которые уже есть в проекте. Главное, чтобы оно было предварительно обработано!

## Дорожная карта

* https://github.com/Dartvauder/NeuroSandboxWebUI/wiki/RoadmapRU

## Благодарность разработчикам

Большое спасибо этим проектам за то, что благодаря им я смог создать свое приложение:

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

## Сторонние лицензии:

#### У многих моделей есть свои собственные лицензии на использование. Перед тем как ее использовать, советую ознакомиться с ними:

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

## Пожертвование

### *Если вам понравился мой проект и вы хотите сделать пожертвование, вот варианты для пожертвований. Заранее большое спасибо!*

* КриптоКошелек(BEP-20) - 0x3d86bdb5f50b92d0d7Eb44F1a833acC5e91aAEcA

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)
