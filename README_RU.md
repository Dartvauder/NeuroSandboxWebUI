## [Функции](/#Функции) | [Зависимости](/#Необходимые-зависимости) | [СистемныеТребования](/#Минимальные-системные-требования) | [Установка](/#Как-установить) | [Использование](/#Как-использовать) | [Модели](/#Где-я-могу-взять-модели-голоса-и-аватары) | [Дорожная карта](/#Дорожная-карта) | [Благодарность](/#Благодарность-разработчикам)

# ![icon (1)](https://github.com/Dartvauder/NeuroChatWebUI/assets/140557322/e3c1d95a-828f-4a65-bea6-64c336dbe6fa)  НейроЧатWebUI (АЛЬФА)
* В процессе разработки!
* [Английский](/README.md) | Russian

## Описание:

Простой и удобный интерфейс для использования различных моделей нейросетей. Вы можете общаться с LLM используя текстовый или голосовой ввод, Stable Diffusion для генерации изображений и AudioCraft для генерации аудио. Здесь доступны функции TTS и Whisper для голосового ввода и вывода с выбором языка и образца голоса

Цель проекта — создать максимально простое приложение для использования нейросетевых моделей.

 

|![Image1](https://github.com/Dartvauder/NeuroChatWebUI/assets/140557322/654c39bd-c952-47c8-b2a0-957368fc36be) | ![Image2](https://github.com/Dartvauder/NeuroChatWebUI/assets/140557322/76fa5cd8-e2b0-4025-8d06-480d61a47e11)
|:---:|:---:|

## Функции:

* Простая установка (Только для Windows)
* Гибкий и оптимизированный интерфейс
* Transformers и llama.cpp (LLM)
* Diffusers и safetensors (Stable Diffusion)
* AudioCraft
* TTS и STT модели (Для LLM)
* Выбор аватара (Для LLM)
* Настройка моделей в интерфейсе

## Необходимые зависимости:

* [Python](https://www.python.org/downloads/) (3.10 минимум)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (Только 12.1)
* [FFMPEG](https://ffmpeg.org/download.html)

## Минимальные системные требования:

* Система: Windows или Linux
* Графический процессор: 4 ГБ+ или центральный процессор: 8 ядер, 3,2 ГГц
* Оперативная память: 16 ГБ+
* Пространство на жестком диске: 20 ГБ+
* Интернет для скачивания моделей и установки

## Как установить:

### Windows

1) `Git clone https://github.com/Dartvauder/NeuroChatWebUI.git` в любое место (Без кириллицы)
2) Запускаем `install.bat` и ждем когда все установиться
3) После установки запустите `start.bat`
4) Выберите версию файла и дождитесь запуска приложения
5) Теперь можете приступать к генерациям!

Чтобы получить обновление, запустите `update.bat`

### Linux

1) `Git clone https://github.com/Dartvauder/NeuroChatWebUI.git` в любое место (Без кириллицы)
2) В терминале запустите `pip install --no-deps -r requirementsGPU.txt` и дождитесь установки всех зависимостей
3) После установки, в терминале запустите `py appEN.py` или `py appRU.py`
4) Дождитесь запуска приложения
5) Теперь можете приступать к генерациям!

Чтобы получить обновление, запустите `git pull`

## Как использовать:

#### Интерфейс имеет три вкладки: LLM, Stable Diffusion и AudioCraft. Выберите ту, которая вам нужна и следуйте инструкциям ниже

### LLM:

1) Сначала загрузите ваши модели в папку: *inputs/text/llm_models*
2) Выберите свою модель в раскрывающемся списке `LLM`
3) Выберите тип модели (`transformers` или `llama`)
4) Настройте модель по нужным вам параметрам
5) Введите (или произнесите) ваш запрос
6) Нажмите кнопку `Submit` чтобы получить текстовый и аудио ответ
#### Необязательно: вы можете включить режим `TTS`, выбрать `голос` и `язык` необходимые для получения аудио ответа. Так же вы можете выбрать `аватар`
#### Образцы голоса = *inputs/audio/voices*
#### Аватары = *inputs/image/avatars*
#### Голос должен быть предварительно обработан (22050 кГц, монозвук, WAV), аватар предпочтительно должен быть `PNG` или `JPG`

### Stable Diffusion:

1) Сначала загрузите ваши модели в папку: *inputs/image/sd_models*
2) Выберите модель из выпадающего списка 
3) Настройте модель по нужным вам параметрам
4) Введите ваш промпт
5) Нажмите кнопку `Submit`, чтобы получить изображение

### AudioCraft:

1) Выберите модель из выпадающего списка.
2) Настройте модель по нужным вам параметрам
3) Введите ваш промпт
4) Нажмите кнопку `Submit`, чтобы получить аудио.

### Общее:

1) История чата и сгенерированные изображения сохраняется в папке *outputs*
2) Вы также можете нажать кнопку `Clear`, чтобы сбросить выбор
#### Чтобы закрыть приложение, закройте терминал

## Где я могу взять модели, голоса и аватары?

* LLM модели можно взять с сайта [HuggingFace](https://huggingface.co/models)
* Модели Stable Diffusion можно взять с сайта [CivitAI](https://civitai.com/models)
* Модели AudioCraft загружаются автоматически, когда вы выбираете модель и нажимаете кнопку `submit`.
* Вы можете использовать голоса откуда угодно. Запишите свой или возьмите запись из интернета. Главное, чтобы оно было предварительно обработано!
* С аватарами то же самое, что и с голосами. Вы можете скачать их в интернете, сгенерировать с помощью нейро сетей или сфотографировать себя. Главное – соблюсти необходимый формат файла
* #### `TTS` и `STT` модели скачиваються автоматически в папку *inputs*. Так же туда скачиваеться базовая `diffusers` модель `Stable Diffusion`

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
* `llama.cpp-python` - https://github.com/abetlen/llama-cpp-python
* `AudioCraft` - https://github.com/facebookresearch/audiocraft
* `xformers` - https://github.com/facebookresearch/xformers

## Пожертвование

### *Если вам понравился мой проект и вы хотите сделать пожертвование, вот варианты для пожертвований. Заранее большое спасибо!*

* КриптоКошелек(BEP-20) - 0x3d86bdb5f50b92d0d7Eb44F1a833acC5e91aAEcA

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)
