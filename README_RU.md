## [Функции](/#Функции) | [Зависимости](/#Необходимые-зависимости) | [СистемныеТребования](/#Минимальные-системные-требования) | [Установка](/#Как-установить) | [Использование](/#Как-использовать) | [Модели](/#Где-я-могу-взять-модели-голоса-и-аватары) | [Дорожная карта](/#Дорожная-карта) | [Благодарность](/#Благодарность-разработчикам)

# ![1](https://github.com/Dartvauder/NeuroChatWebUI/assets/140557322/ae17831a-a0d0-4245-9bb3-996910f3094b)
* В процессе разработки! (АЛЬФА)
* [Английский](/README.md) | Russian

## Описание:

Простой и удобный интерфейс для использования различных моделей нейросетей. Вы можете общаться с LLM используя текстовый или голосовой ввод, Stable Diffusion для генерации изображений и AudioCraft для генерации аудио

Цель проекта — создать максимально простое приложение для использования нейросетевых моделей

### LLM: ![1](https://github.com/Dartvauder/NeuroChatWebUI/assets/140557322/165b8b01-03f6-4be1-8424-544c3a000bf3)

### Stable Diffusion: 
 #### txt2img: ![2](https://github.com/Dartvauder/NeuroChatWebUI/assets/140557322/1c147103-daf4-458d-b956-1843ee6ef989)
 #### img2img: ![3](https://github.com/Dartvauder/NeuroChatWebUI/assets/140557322/03c9edf7-9742-47c4-a2cd-da097fc79abf)
 #### Extras: ![4](https://github.com/Dartvauder/NeuroChatWebUI/assets/140557322/4931f00a-b025-4d4a-8705-418fb0c5a257)

### AudioCraft: ![5](https://github.com/Dartvauder/NeuroChatWebUI/assets/140557322/33be43dd-c3e3-45e0-8769-51f5e9b9f24d)

## Функции:

* Простая установка через install.bat (Только для Windows)
* Гибкий и оптимизированный интерфейс (От Gradio)
* Поддержка Transformers и llama.cpp моделей (LLM)
* Поддержка diffusers (safetensors) моделей (Stable Diffusion) - Вкладки txt2img, img2img и Extras
* Поддержка AudioCraft (Модели: musicgen, audiogen и magnet)
* Поддержка TTS и Whisper моделей (Для LLM)
* Поддержка Lora, Vae и Upscale моделей (Для Stable Diffusion)
* Поддержка Multiband Diffusion модели (Для AudioCraft)
* Возможность выбора аватара (Для LLM)
* Настройки моделей внутри интерфейса

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

1) `Git clone https://github.com/Dartvauder/NeuroChatWebUI.git` в любое место (Без кириллицы)
2) Запускаем `install.bat` и ждем когда все установиться
3) После установки запустите `start.bat`
4) Выберите версию файла и дождитесь запуска приложения
5) Теперь вы можете приступать к генерациям!

Чтобы получить обновление, запустите `update.bat`

### Linux

1) `Git clone https://github.com/Dartvauder/NeuroChatWebUI.git` в любое место (Без кириллицы)
2) В терминале запустите `pip install --no-deps -r requirements.txt` и дождитесь установки всех зависимостей
3) После установки, в терминале запустите `py appEN.py` или `py appRU.py`
4) Дождитесь запуска приложения
5) Теперь вы можете приступать к генерациям!

Чтобы получить обновление, запустите `git pull`

## Как использовать:

#### Интерфейс имеет три вкладки: LLM, Stable Diffusion и AudioCraft. Выберите ту, которая вам нужна и следуйте инструкциям ниже

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
#### Голос должен быть предварительно обработан (22050 кГц, монозвук, WAV), аватар предпочтительно должен быть `PNG` или `JPG`

### Stable Diffusion - имеет три под-вкладки:

#### txt2img:

1) Сначала загрузите ваши модели в папку: *inputs/image/sd_models*
2) Выберите вашу модель из выпадающего списка
3) Выберите тип модели (`SD`, `SD2` или `SDXL`)
4) Настройте модель по нужным вам параметрам
5) Введите ваш запрос
6) Нажмите кнопку `Submit`, чтобы получить сгенерированное изображение
#### Необязательно: вы можете выбрать свои модели `vae` и `lora` чтобы улучшить метод генерации, а так же включить `upscale` чтобы увеличить размер сгенерированного изображения
#### vae = *inputs/image/sd_models/vae*
#### lora = *inputs/image/sd_models/lora*

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

#### Extras:

1) Выберите нужные вам опции
2) Загрузите исходное изображение
3) Нажмите кнопку `Submit`, чтобы получить измененное изображение

### AudioCraft:

1) Выберите модель из выпадающего списка
2) Выберите тип модели (`musicgen` или `audiogen`)
3) Настройте модель по нужным вам параметрам
4) Введите ваш запрос
5) (Необязательно) загрузите исходный звук, если вы используете модель `melody`
6) Нажмите кнопку `Submit`, чтобы получить сгенерированный звук
#### Необязательно: вы можете включить `multiband diffusion`, чтобы улучшить генерируемый звук

### Общее:

1) История чата, сгенерированные изображения и сгенерированные аудио сохраняются в папке *outputs*
2) Вы также можете нажать кнопку `Clear`, чтобы сбросить ваш выбор
3) Чтобы остановить процесс генерации, нажмите кнопку `Stop generation`
4) Вы также можете выключить приложение с помощью кнопки `Close terminal`

## Где я могу взять модели, голоса и аватары?

* LLM модели можно взять с сайта [HuggingFace](https://huggingface.co/models)
* Модели Stable Diffusion, vae и lora можно взять с сайта [CivitAI](https://civitai.com/models)
* Модели AudioCraft загружаются автоматически в папку *inputs*, когда вы выбираете модель и нажимаете кнопку `submit`
* TTS, Whisper, Upscale и Multiband diffusion модели скачиваються автоматически в папку *inputs* при их использовании
* Вы можете использовать голоса откуда угодно. Запишите свой или возьмите запись из интернета. Главное, чтобы оно было предварительно обработано!
* С аватарами то же самое, что и с голосами. Вы можете скачать их в интернете, сгенерировать с помощью нейросетей или сфотографировать себя. Главное – соблюсти необходимый формат файла

## Дорожная карта

* https://github.com/Dartvauder/NeuroChatWebUI/wiki/RoadmapRU

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
