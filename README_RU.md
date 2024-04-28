## [Функции](/#Функции) | [Зависимости](/#Необходимые-зависимости) | [СистемныеТребования](/#Минимальные-системные-требования) | [Установка](/#Как-установить) | [Использование](/#Как-использовать) | [Модели](/#Где-я-могу-взять-модели-голоса-и-аватары) | [Дорожная карта](/#Дорожная-карта) | [Благодарность](/#Благодарность-разработчикам)

# ![main](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4ea0d891-8979-45ad-b052-626c41ae991a)
* В процессе разработки! (АЛЬФА)
* [Английский](/README.md) | Russian

## Описание:

Простой и удобный интерфейс для использования различных моделей нейронных сетей. Вы можете общаться с LLM, используя текстовый или голосовой ввод, использовать StableDiffusion для генерации изображений и видео, AudioCraft для генерации музыки и аудио, CoquiTTS для преобразования текста в речь, OpenAI-Whisper для преобразования речи в текст, LibreTranslate для перевода текста и Demucs для сепарации аудио файлов. Также вы можете скачать модели LLM и StableDiffusion, изменить настройки приложения внутри интерфейса и проверить датчики системы

Цель проекта — создать максимально простое приложение для использования нейросетевых моделей

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

## Функции:

* Простая установка через install.bat (Только для Windows)
* Вы можете использовать приложение через мобильное устройство в localhost (Через IPv4)
* Гибкий и оптимизированный интерфейс (От Gradio)
* Поддержка Transformers и llama.cpp моделей (LLM)
* Поддержка diffusers и safetensors моделей (StableDiffusion) - Вкладки txt2img, img2img, depth2img, upscale, inpaint, video и extras
* Поддержка AudioCraft (Модели: musicgen, audiogen и magnet)
* Поддержка TTS и Whisper моделей (Для LLM и TTS-STT)
* Поддержка Lora, Textual inversion (embedding), Vae, Inpaint, Upscale и Video моделей (Для StableDiffusion)
* Поддержка Multiband Diffusion модели (Для AudioCraft)
* Поддержка LibreTranslate (Локальный API)
* Поддержка Demucs
* Поддержка Rembg
* Поддержка multimodal LLaVA 1.6 (Для LLM)
* Настройки моделей внутри интерфейса
* ModelDownloader (Для LLM и StableDiffusion)
* Настройки приложения
* Возможность видеть датчики системы

## Необходимые зависимости:

* [Python](https://www.python.org/downloads/) (3.10 минимум)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (Только 12.1)
* [FFMPEG](https://ffmpeg.org/download.html)

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
2) В терминале запустите `pip install --no-deps -r requirements.txt` и дождитесь установки всех зависимостей
3) После установки, в терминале запустите `py appEN.py` или `py appRU.py`
4) Дождитесь запуска приложения
5) Теперь вы можете приступать к генерациям!

Чтобы получить обновление, запустите `git pull`

## Как использовать:

#### Интерфейс имеет девять вкладок: LLM, TTS-STT, LibreTranslate, StableDiffusion, AudioCraft, Demucs, ModelDownloader, Settings и System. Выберите ту которая вам нужна и следуйте инструкциям ниже

### LLM:

1) Сначала загрузите ваши модели в папку: *inputs/text/llm_models*
2) Выберите вашу модель из выпадающего списка
3) Выберите тип модели (`transformers` или `llama`)
4) Настройте модель по нужным вам параметрам
5) Введите (или произнесите) ваш запрос
6) Нажмите кнопку `Submit` чтобы получить сгенерированный текстовый и аудио ответ
#### Дополнительно: вы можете включить режим `TTS`, выбрать `голос` и `язык` необходимые для получения аудио ответа. Так же вы можете выбрать `аватар`
#### Аватары = *inputs/image/avatars*
#### Образцы голоса = *inputs/audio/voices*
#### Голос должен быть предварительно обработан (22050 кГц, монозвук, WAV), аватар предпочтительно должен быть `PNG` или `JPEG`

### TTS-STT:

1) Введите текст для преобразования текста в речь
2) Введите звук для преобразования речи в текст
3) Нажмите кнопку `Submit`, чтобы получить сгенерированный текстовый и аудио ответ
#### Образцы голоса = *inputs/audio/voices*
#### Голос должен быть предварительно обработан (22050 кГц, монозвук, WAV)

### LibreTranslate:

* Сначала вам нужно установить и запустить [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate)
1) Выберите исходный и целевой язык
2) (Дополнительно) Вы можете сохранить историю перевода включив соответствующую кнопку
3) Нажмите кнопку `Submit`, чтобы получить перевод

### StableDiffusion - имеет семь под-вкладок:

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

#### video:

1) Загрузите исходное изображение
2) Настройте модель по нужным вам параметрам
3) Нажмите кнопку `Submit`, чтобы получить видео из изображения

#### extras:

1) Загрузите исходное изображение
2) Выберите нужные вам опции
3) Нажмите кнопку `Submit`, чтобы получить измененное изображение

### AudioCraft:

1) Выберите модель из выпадающего списка
2) Выберите тип модели (`musicgen` или `audiogen`)
3) Настройте модель по нужным вам параметрам
4) Введите ваш запрос
5) (Дополнительно) загрузите исходный звук, если вы используете модель `melody`
6) Нажмите кнопку `Submit`, чтобы получить сгенерированный звук
#### Дополнительно: вы можете включить `multiband diffusion`, чтобы улучшить генерируемый звук

### Demucs:

1) Загрузите исходный звук для сепарации
2) Нажмите кнопку `Submit`, чтобы получить сепарированный звук

### ModelDownloader:

* Здесь вы можете скачать модели `LLM` и `StableDiffusion`. Просто выберите модель из выпадающего списка и нажмите кнопку `Submit`
#### Модели `LLM` скачиваються сюда: *inputs/text/llm_models*
#### Модели `StableDiffusion` скачиваються сюда: *inputs/image/sd_models*

### Settings:

* Здесь вы можете изменить настройки приложения. На данный момент вы можете изменить только режим `Share` на `True` или `False`

### System: 

* Здесь вы можете увидеть показатели датчиков вашего компьютера

### Общее:

1) История чата, сгенерированные изображения и сгенерированные аудио сохраняются в папке *outputs*
2) Вы также можете нажать кнопку `Clear`, чтобы сбросить ваш выбор
3) Чтобы остановить процесс генерации, нажмите кнопку `Stop generation`
4) Вы также можете выключить приложение с помощью кнопки `Close terminal`
5) Вы можете открыть папку *outputs* нажав на кнопку `Folder`

## Где я могу взять модели, голоса и аватары?

* LLM модели можно взять с сайта [HuggingFace](https://huggingface.co/models) или из внутреннего интерфейса ModelDownloader
* Модели StableDiffusion, vae, inpaint, embedding и lora можно взять с сайта [CivitAI](https://civitai.com/models) или из внутреннего интерфейса ModelDownloader
* Модели AudioCraft загружаются автоматически в папку *inputs*, когда вы выбираете модель и нажимаете кнопку `submit`
* TTS, Whisper, Upscale и Multiband diffusion модели скачиваються автоматически в папку *inputs* при их использовании
* Вы можете использовать голоса откуда угодно. Запишите свой или возьмите запись из интернета. Главное, чтобы оно было предварительно обработано!
* С аватарами то же самое, что и с голосами. Вы можете скачать их в интернете, сгенерировать с помощью нейросетей или сфотографировать себя. Главное – соблюсти необходимый формат файла

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
* `xformers` - https://github.com/facebookresearch/xformers
* `demucs` - https://github.com/facebookresearch/demucs
* `libretranslatepy` - https://github.com/argosopentech/LibreTranslate-py
* `rembg` - https://github.com/danielgatis/rembg

## Пожертвование

### *Если вам понравился мой проект и вы хотите сделать пожертвование, вот варианты для пожертвований. Заранее большое спасибо!*

* КриптоКошелек(BEP-20) - 0x3d86bdb5f50b92d0d7Eb44F1a833acC5e91aAEcA

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)
