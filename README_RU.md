## [Функции](/#Функции) | [Зависимости](/#Необходимые-зависимости) | [Установка](/Как-установить) | [Использование](/#Как-использовать) | [Модели](/#Где-я-могу-взять-модели-голоса-и-аватары) | [Дорожная карта](/#Дорожная-карта) | [Благодарность](/#Благодарность-разработчикам)

# ![icon (1)](https://github.com/Dartvauder/NeuroChatWebUI/assets/140557322/e3c1d95a-828f-4a65-bea6-64c336dbe6fa)  НейроЧатWebUI (АЛЬФА)
* В процессе разработки!
* [Английский](/README.md) | Russian

## Описание:

Простой и удобный интерфейс для общения с LLM с использованием текстового или голосового ввода, а также для создания изображений с помощью Stable Diffusion. Здесь доступны функции TTS и Whisper для голосового ввода или вывода с выбором языка и образца голоса. Цель проекта — создать максимально простое приложение для новичков в теме нейронных сетей.

|![Image1](https://github.com/Dartvauder/NeuroChatWebUI/assets/140557322/098c9b93-253d-44e7-9d34-dd4fe3317b41) |
|:---:|

## Функции:

* Простая установка (Только для Windows)
* Гибкий и оптимизированный интерфейс
* TTS и STT модели (Для LLM)
* Выбор аватара (Для LLM)
* Transformers и llama.cpp (Для LLM)
* Diffusers и safetensors (Для Stable Diffusion)
* Настройка моделей в интерфейсе
* И многое другое...

## Необходимые зависимости:

* [Python](https://www.python.org/downloads/) (3.10 минимум)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (Только 12.1)
* [FFMPEG](https://ffmpeg.org/download.html)

## Как установить:

### Windows

1) `Git clone` или [скачайте](https://github.com/Dartvauder/NeuroChatWebUI/archive/refs/tags/Alpha.zip) репозиторий.
2) Разархивируйте файл архива в любое место.
3) Запускаем `install.bat` и выбираем версию для установки.
4) После установки запустите `start.bat`.
5) Выберите версию файла и дождитесь запуска приложения.
6) Веселитесь!

Чтобы получить обновление, запустите `update.bat`

### Linux

1) `Git clone` или [скачайте](https://github.com/Dartvauder/NeuroChatWebUI/archive/refs/tags/Alpha.zip) репозиторий.
2) Разархивируйте файл архива в любое место.
3) В терминале запустите `pip install -r requirementsGPU.txt` или `pip install -r requirementsCPU.txt` и дождитесь установки всех зависимостей.
4) После установки запустите `py appEN.py` или `py appRU.py`.
5) Дождитесь запуска приложения.
6) Веселитесь!

Чтобы получить обновление, запустите `git pull`

## Как использовать:

### LLM:

* Сначала загрузите модели по папкам. Пути к моделям: LLM = *inputs/text/llm_models*
#### Необязательно: образцы голоса = *inputs/audio/voices*; Аватары = *inputs/image/avatars*
* Настройте модель по нужным вам настройкам
* Для начала выберите свою модель в раскрывающемся списке `LLM`, введите (или произнесите) подсказку, нажмите кнопку `Generate` чтобы получить текстовый и аудио ответ. Если хотите, можете выбрать `аватар`
#### Необязательно: вы можете включить режим `TTS`, выбрать `голос` и `язык` необходимые для получения звукового ответа.

### Stable Diffusion:

#### Перед генерацией изображения, отключите LLM и TTS модели, а так же очистите аудио ввод

* Сначала впишите текстовый запрос
* Настройте модель по нужным вам настройкам
* Потом включите `Stable Diffusion` и нажмите кнопку `generate`, чтобы получить изображение.

П.С. Голос должен быть предварительно обработан (22050 кГц, монозвук, WAV), аватар предпочтительно должен быть PNG или JPG, модель LLM должна быть Transformers. При включении TTS требуется выбор языка и голоса, иначе будут ошибки.

#### История чата и сгенерированные изображения сохраняется в папке *outputs*.
#### Вы также можете нажать кнопку `Clear`, чтобы сбросить выбор.
#### Чтобы закрыть приложение, закройте терминал.

## Где я могу взять модели, голоса и аватары?

* Языковые модели можно взять с сайта [HuggingFace](https://huggingface.co/models)
* Вы можете использовать голоса откуда угодно. Запишите свой или возьмите запись из интернета. Главное, чтобы оно было предварительно обработано!
* С аватарами то же самое, что и с голосами. Вы можете скачать их в интернете, сгенерировать с помощью нейро сетей или сфотографировать себя. Главное – соблюсти необходимый формат файла.
* #### TTS, STT и Stable Diffusion модели скачиваються автоматически в папку *inputs*

## Дорожная карта

* https://github.com/Dartvauder/NeuroChatWebUI/wiki/RoadmapRU

## Благодарность разработчикам

Большое спасибо этим проектам за то, что позволили мне создать свое приложение:

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

## Пожертвование

### Если вам понравился мой проект и вы хотите сделать пожертвование, вот варианты для пожертвований. Заранее большое спасибо!

* КриптоКошелек(BEP-20) - 0x3d86bdb5f50b92d0d7Eb44F1a833acC5e91aAEcA

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)
