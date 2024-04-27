## [Функции](/#Функции) | [Зависимости](/#Необходимые-зависимости) | [СистемныеТребования](/#Минимальные-системные-требования) | [Установка](/#Как-установить) | [Использование](/#Как-использовать) | [Модели](/#Где-я-могу-взять-модели-голоса-и-аватары) | [Дорожная карта](/#Дорожная-карта) | [Благодарность](/#Благодарность-разработчикам)

# ![main](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4ea0d891-8979-45ad-b052-626c41ae991a)
* В процессе разработки! (АЛЬФА)
* [Английский](/README.md) | Russian

## Описание:

Простой и удобный интерфейс для использования различных моделей нейронных сетей. Вы можете общаться с LLM, используя текстовый или голосовой ввод, использовать StableDiffusion для генерации изображений и видео, AudioCraft для генерации аудио, CoquiTTS для преобразования текста в речь и OpenAI-Whisper для преобразования речи в текст. Также вы можете скачать модели LLM и StableDiffusion, и изменить настройки приложения внутри интерфейса

Цель проекта — создать максимально простое приложение для использования нейросетевых моделей

### LLM: ![1](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/9f6c7c86-03fe-400d-9f28-4824a93100f0)

### TTS-STT: ![2](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/b327c698-66a8-4649-a754-98830c2cbf27)

### StableDiffusion:
 #### txt2img: ![3](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/e18f7992-7b79-4250-ab6c-27fac5de0776)
 #### img2img: ![4](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/d51168e6-29ab-4a58-ac0e-b83a8f7c40e0)
 #### inpaint: ![5](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/953ab08c-30ff-4b6e-9590-1037b4ffa8f4)
 #### video: ![6](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/0e5738de-f8c0-4ace-badc-af1e64fae196)
 #### extras: ![7](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/55f37964-2a56-49e2-9995-df9122f76172)

### AudioCraft: ![8](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/382ca556-1a14-4099-af36-936c212048f3)

### ModelDownloader: ![9](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/ff98c1f1-33df-4018-981c-32671700659f)

### Settings: ![10](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/40b38b60-0e32-4a04-a278-fdb0aa15c224)

## Функции:

* Простая установка через install.bat (Только для Windows)
* Вы можете использовать приложение через мобильное устройство в localhost (Через IPv4)
* Гибкий и оптимизированный интерфейс (От Gradio)
* Поддержка Transformers и llama.cpp моделей (LLM)
* Поддержка diffusers (safetensors) моделей (StableDiffusion) - Вкладки txt2img, img2img, inpaint, video и extras
* Поддержка AudioCraft (Модели: musicgen, audiogen и magnet)
* Поддержка TTS и Whisper моделей (Для LLM и TTS-STT)
* Поддержка Lora, Textual inversion (embedding), Vae, Inpaint, Upscale и Video моделей (Для StableDiffusion)
* Поддержка Multiband Diffusion модели (Для AudioCraft)
* Возможность выбора аватара (Для LLM)
* Настройки моделей внутри интерфейса
* ModelDownloader (Для LLM и StableDiffusion)
* Настройки приложения

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

#### Интерфейс имеет шесть вкладок: LLM, TTS-STT, StableDiffusion, AudioCraft, ModelDownloader и Settings. Выберите ту которая вам нужна и следуйте инструкциям ниже

### LLM:

1) Сначала загрузите ваши модели в папку: *inputs/text/llm_models*
2) Выберите вашу модель из выпадающего списка
3) Выберите тип модели (`transformers` или `llama`)
4) Настройте модель по нужным вам параметрам
5) Введите (или произнесите) ваш запрос
6) Нажмите кнопку `Submit` чтобы получить сгенерированный текстовый и аудио ответ
#### Необязательно: вы можете включить режим `TTS`, выбрать `голос` и `язык` необходимые для получения аудио ответа. Так же вы можете выбрать `аватар`
#### Аватары = *inputs/image/avatars*
#### Образцы голоса = *inputs/audio/voices*
#### Голос должен быть предварительно обработан (22050 кГц, монозвук, WAV), аватар предпочтительно должен быть `PNG` или `JPEG`

### TTS-STT:

1) Введите текст для преобразования текста в речь
2) Введите звук для преобразования речи в текст
3) Нажмите кнопку `Submit`, чтобы получить сгенерированный текстовый и аудио ответ
#### Образцы голоса = *inputs/audio/voices*
#### Голос должен быть предварительно обработан (22050 кГц, монозвук, WAV)

### StableDiffusion - имеет пять под-вкладок:

#### txt2img:

1) Сначала загрузите ваши модели в папку: *inputs/image/sd_models*
2) Выберите вашу модель из выпадающего списка
3) Выберите тип модели (`SD`, `SD2` или `SDXL`)
4) Настройте модель по нужным вам параметрам
5) Введите ваш запрос
6) Нажмите кнопку `Submit`, чтобы получить сгенерированное изображение
#### Необязательно: вы можете выбрать свои модели `vae`, `embedding` и `lora` чтобы улучшить метод генерации, а так же включить `upscale` чтобы увеличить размер сгенерированного изображения
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
#### Необязательно: вы можете выбрать свою модель `vae`
#### vae = *inputs/image/sd_models/vae*

#### inpaint:

1) Сначала загрузите ваши модели в папку: *inputs/image/sd_models/inpaint*
2) Выберите вашу модель из выпадающего списка
3) Выберите тип модели (`SD`, `SD2` или `SDXL`)
4) Настройте модель по нужным вам параметрам
5) Загрузите изображение, с которым будет происходить генерация, в `initial image` и `mask image`
6) В `mask image` выберите кисть, затем палитру и измените цвет на `#FFFFFF`
7) Нарисуйте место для генерации и введите ваш запрос
8) Нажмите кнопку `Submit`, чтобы получить измененное изображение
#### Необязательно: вы можете выбрать свою модель `vae`
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
5) (Необязательно) загрузите исходный звук, если вы используете модель `melody`
6) Нажмите кнопку `Submit`, чтобы получить сгенерированный звук
#### Необязательно: вы можете включить `multiband diffusion`, чтобы улучшить генерируемый звук

### ModelDownloader:

* Здесь вы можете скачать модели `LLM` и `StableDiffusion`. Просто выберите модель из выпадающего списка и нажмите кнопку `Submit`
#### Модели `LLM` скачиваються сюда: *inputs/text/llm_models*
#### Модели `StableDiffusion` скачиваються сюда: *inputs/image/sd_models*

### Settings:

* Здесь вы можете изменить настройки приложения. На данный момент вы можете изменить только режим `Share` на `True` или `False`

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
* `AudioCraft` - https://github.com/facebookresearch/audiocraft
* `xformers` - https://github.com/facebookresearch/xformers

## Пожертвование

### *Если вам понравился мой проект и вы хотите сделать пожертвование, вот варианты для пожертвований. Заранее большое спасибо!*

* КриптоКошелек(BEP-20) - 0x3d86bdb5f50b92d0d7Eb44F1a833acC5e91aAEcA

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)
