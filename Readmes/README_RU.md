## [Функции](/#Функции) | [Зависимости](/#Необходимые-зависимости) | [СистемныеТребования](/#Минимальные-системные-требования) | [Установка](/#Как-установить) | [Использование](/#Как-использовать) | [Модели](/#Где-я-могу-взять-модели-голоса-и-аватары) | [Вики](/#Вики) | [Благодарность](/#Благодарность-разработчикам) | [Лицензии](/#Сторонние-лицензии)

# ![main](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4ea0d891-8979-45ad-b052-626c41ae991a)
* В процессе разработки! (АЛЬФА)
* [English](/README.md)  | [عربي](/Readmes/README_AR.md) | [Deutsche](/Readmes/README_DE.md) | [Español](/Readmes/README_ES.md) | [Français](/Readmes/README_FR.md) | [日本語](/Readmes/README_JP.md) | Русский | [漢語](/Readmes/README_ZH.md) | [Português](/Readmes/README_PT.md) | [हिंदी](/Readmes/README_HI.md) | [Italiano](/Readmes/README_IT.md) | [韓國語](/Readmes/README_KO.md)

## Описание:

Простой и удобный интерфейс для использования различных моделей нейронных сетей. Вы можете общаться с LLM и Moondream2 используя текстовый или голосовой ввод и загрузив изображение; использовать StableDiffusion, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF и PixArt для генерации изображений; ModelScope, ZeroScope 2, CogVideoX и Latte для генерации видео; TripoSR, StableFast3D, Shap-E, SV34D и Zero123Plus для генерации 3Д обьектов; StableAudioOpen, AudioCraft и AudioLDM 2 для генерации музыки и аудио; CoquiTTS и SunoBark для преобразования текста в речь; OpenAI-Whisper для преобразования речи в текст; Wav2Lip для лицевой синхронизации губ; Roop для замены лица; Rembg для удаления фона; CodeFormer для восстановления лиц; LibreTranslate для перевода текста; Demucs для сепарации аудио файлов. Также вы можете просматривать файлы из папок outputs в галерее, скачать модели LLM и StableDiffusion, изменить настройки приложения внутри интерфейса и проверить датчики системы

Цель проекта — создать максимально простое приложение для использования нейросетевых моделей

### Текст: <img width="1127" alt="1ru" src="https://github.com/user-attachments/assets/dbcfdd8d-f893-48ff-90b5-55d03d47ab3e">

### Изображение: <img width="1118" alt="2ru" src="https://github.com/user-attachments/assets/9cd3231e-3821-414f-aa43-e2b5e6ad6f9b">

### Видео: <img width="1118" alt="3ru" src="https://github.com/user-attachments/assets/efe4f62f-d870-46ee-b803-48e00af9b17c">

### 3D: <img width="1121" alt="4ru" src="https://github.com/user-attachments/assets/df783a3c-f2bf-439c-85f3-c992a9a8e910">

### Аудио: <img width="1115" alt="5ru" src="https://github.com/user-attachments/assets/5881226a-df57-4623-bef1-0df71b557ca2">

### Интерфейс: <img width="1122" alt="6ru" src="https://github.com/user-attachments/assets/3912f4d0-7d0f-4f51-9863-a16b46f88585">

## Функции:

* Простая установка через install.bat(Windows) или install.sh(Linux)
* Вы можете использовать приложение через мобильное устройство в localhost(Через IPv4) или через любое онлайн(Через Share)
* Гибкий и оптимизированный интерфейс (От Gradio)
* Аутентификация через admin:admin (Вы можете вписать свои данные для входа в файле GradioAuth.txt)
* Вы можете добавить свой собственный HuggingFace-Token для загрузки определенных моделей (Вы можете ввести свой токен в файле HF-Token.txt)
* Поддержка Transformers и llama.cpp моделей (LLM)
* Поддержка diffusers и safetensors моделей (StableDiffusion) - Вкладки txt2img, img2img, depth2img, pix2pix, controlnet, upscale, inpaint, gligen, animatediff, video, ldm3d, sd3, cascade, adapters и extras
* Поддержка дополнительных моделей для генерации изображений: Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF и PixArt
* Поддержка StableAudioOpen
* Поддержка AudioCraft (Модели: musicgen, audiogen и magnet)
* Поддержка AudioLDM 2 (Модели: audio и music)
* Поддержка TTS и Whisper моделей (Для LLM и TTS-STT)
* Поддержка Lora, Textual inversion (embedding), Vae, Img2img, Depth, Pix2Pix, Controlnet, Upscale(latent), Upscale(Real-ESRGAN), Inpaint, GLIGEN, AnimateDiff, Videos, LDM3D, SD3, Cascade, Adapters (InstantID, PhotoMaker, IP-Adapter-FaceID), Rembg, CodeFormer и Roop моделей (Для StableDiffusion)
* Поддержка Multiband Diffusion модели (Для AudioCraft)
* Поддержка LibreTranslate (Локальный API)
* Поддержка ModelScope, ZeroScope 2, CogVideoX и Latte для генерации видео
* Поддержка SunoBark
* Поддержка Demucs
* Поддержка TripoSR, StableFast3D, Shap-E, SV34D и Zero123Plus для 3Д генерации
* Поддержка Wav2Lip
* Поддержка Multimodal (Moondream 2), LORA (transformers) и WebSearch (with GoogleSearch) для LLM
* Настройки моделей внутри интерфейса
* Галерея
* Загрузчик моделей (Для LLM и StableDiffusion)
* Настройки приложения
* Возможность видеть датчики системы

## Необходимые зависимости:

* [Python](https://www.python.org/downloads/) (3.11)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (12.4) и [cuDNN](https://developer.nvidia.com/cudnn-downloads) (9.1)
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
2) Запускаем `Install.bat` и ждем когда все установиться
3) После установки запустите `Start.bat`
4) Выберите версию файла и дождитесь запуска приложения
5) Теперь вы можете приступать к генерациям!

Чтобы получить обновление, запустите `Update.bat`
Для работы с виртуальной средой через терминал, запустите `Venv.bat`

### Linux

1) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` в любое место (Без кириллицы)
2) В терминале запустите `./Install.sh` и дождитесь установки всех зависимостей
3) После установки, в терминале запустите `./Start.sh`
4) Дождитесь запуска приложения
5) Теперь вы можете приступать к генерациям!

Чтобы получить обновление, запустите `./Update.sh`
Для работы с виртуальной средой через терминал, запустите `./Venv.sh`

## Как использовать:

#### Интерфейс имеет тридцать две вкладки в шести главных вкладках (Текст, Изображение, Видео, 3D, Аудио и Интерфейс): LLM, TTS-STT, SunoBark, LibreTranslate, Wav2Lip, StableDiffusion, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF, PixArt, ModelScope, ZeroScope 2, CogVideoX, Latte, TripoSR, StableFast3D, Shap-E, SV34D, Zero123Plus, StableAudio, AudioCraft, AudioLDM 2, Demucs, Галерея, Загрузчик моделей, Настройки и Система. Выберите ту которая вам нужна и следуйте инструкциям ниже

### LLM:

1) Сначала загрузите ваши модели в папку: *inputs/text/llm_models*
2) Выберите вашу модель из выпадающего списка
3) Выберите тип модели (`transformers` или `llama`)
4) Настройте модель по нужным вам параметрам
5) Введите (или произнесите) ваш запрос
6) Нажмите кнопку `Submit` чтобы получить сгенерированный текстовый и аудио ответ
#### Дополнительно: вы можете включить режим TTS, выбрать `voice` и `language`, необходимые для получения аудио ответа. Вы можете включить `multimodal` и загрузить изображение, чтобы получить его описание. Вы можете включить `websearch` для доступа в Интернет. Вы можете включить `libretranslate`, чтобы получить перевод. Также вы можете выбрать модель `LORA` для улучшения генерации
#### Образцы голоса = *inputs/audio/voices*
#### LORA = *inputs/text/llm_models/lora*
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
1) Выберите исходный и целевой языки
2) Нажмите кнопку `Submit`, чтобы получить перевод
#### Дополнительно: вы можете сохранить историю перевода включив соответствующую кнопку

### Wav2Lip:

1) Загрузите исходное изображение лица
2) Загрузите исходный звук голоса
3) Настройте модель по нужным вам параметрам
4) Нажмите кнопку `Submit`, чтобы получить лицевую синхронизацию губ

### StableDiffusion - имеет шестнадцать под-вкладок:

#### txt2img:

1) Сначала загрузите ваши модели в папку: *inputs/image/sd_models*
2) Выберите вашу модель из выпадающего списка
3) Выберите тип модели (`SD`, `SD2` или `SDXL`)
4) Настройте модель по нужным вам параметрам
5) Введите ваш запрос (+ и - для prompt weighting)
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
6) Введите ваш запрос (+ и - для prompt weighting)
7) Нажмите кнопку `Submit`, чтобы получить сгенерированное изображение
#### Дополнительно: вы можете выбрать свою модель `vae`
#### vae = *inputs/image/sd_models/vae*

#### depth2img:

1) Загрузите исходное изображение
2) Настройте модель по нужным вам параметрам
3) Введите ваш запрос (+ и - для prompt weighting)
4) Нажмите кнопку `Submit`, чтобы получить сгенерированное изображение

#### pix2pix:

1) Загрузите исходное изображение
2) Настройте модель по нужным вам параметрам
3) Введите ваш запрос (+ и - для prompt weighting)
4) Нажмите кнопку `Submit`, чтобы получить сгенерированное изображение

#### controlnet:

1) Сначала загрузите ваши модели stable diffusion в папку: *inputs/image/sd_models*
2) Загрузите исходное изображение
3) Выберите модели stable diffusion и controlnet из раскрывающихся списков
4) Настройте модели по нужным вам параметрам
5) Введите ваш запрос (+ и - для prompt weighting)
6) Нажмите кнопку `Submit`, чтобы получить сгенерированное изображение

#### upscale(latent):

1) Загрузите исходное изображение
2) Настройте модель по нужным вам параметрам
3) Нажмите кнопку `Submit`, чтобы получить увеличенное изображение

#### upscale(Real-ESRGAN):

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
7) Нарисуйте место для генерации и введите ваш запрос (+ и - для prompt weighting)
8) Нажмите кнопку `Submit`, чтобы получить измененное изображение
#### Дополнительно: вы можете выбрать свою модель `vae`
#### vae = *inputs/image/sd_models/vae*

#### gligen:

1) Сначала загрузите ваши модели в папку: *inputs/image/sd_models*
2) Выберите вашу модель из выпадающего списка
3) Выберите тип модели (`SD`, `SD2` или `SDXL`)
4) Настройте модель по нужным вам параметрам
5) Введите ваш запрос для prompt (+ и - для prompt weighting) и GLIGEN phrases (в "" для box)
6) Введите GLIGEN boxes (Как [0.1387, 0.2051, 0.4277, 0.7090] для box)
7) Нажмите кнопку `Submit`, чтобы получить сгенерированное изображение

#### animatediff:

1) Сначала загрузите ваши модели в папку: *inputs/image/sd_models*
2) Выберите вашу модель из выпадающего списка
3) Настройте модель по нужным вам параметрам
4) Введите ваш запрос (+ и - для prompt weighting)
5) Нажмите кнопку `Submit`, чтобы получить анимированное изображение

#### video:

1) Загрузите исходное изображение
2) Введите ваш запрос (для IV2Gen-XL)
3) Настройте модель по нужным вам параметрам
4) Нажмите кнопку `Submit`, чтобы получить видео из изображения

#### ldm3d:

1) Введите ваш запрос
2) Настройте модель по нужным вам параметрам
3) Нажмите кнопку `Submit`, чтобы получить сгенерированное изображения

#### sd3 (txt2img, img2img, controlnet, inpaint):

1) Введите ваш запрос
2) Настройте модель по нужным вам параметрам
3) Нажмите кнопку `Submit`, чтобы получить сгенерированное изображение

#### cascade:

1) Введите ваш запрос
2) Настройте модель по нужным вам параметрам
3) Нажмите кнопку `Submit`, чтобы получить сгенерированное изображение

#### adapters (InstantID, PhotoMaker и IP-Adapter-FaceID):

1) Сначала загрузите свои модели в папку: *inputs/image/sd_models*
2) Загрузите исходное изображение
3) Выберите свою модель из выпадающего списка
4) Настройте модель в соответствии с необходимыми параметрами
5) Выберите нужную вам вкладку
6) Нажмите кнопку `Submit`, чтобы получить измененное изображение

#### extras:

1) Загрузите исходное изображение
2) Выберите нужные вам опции
3) Нажмите кнопку `Submit`, чтобы получить измененное изображение

### Kandinsky (txt2img, img2img, inpaint):

1) Введите ваш запрос
2) Выберите модель из выпадающего списка
3) Настройте модель по нужным вам параметрам
4) Нажмите кнопку `Submit`, чтобы получить сгенерированное изображение

### Flux:

1) Введите ваш запрос
2) Выберите модель из выпадающего списка
3) Настройте модель по нужным вам параметрам
4) Нажмите кнопку `Submit`, чтобы получить сгенерированное изображение

### HunyuanDiT:

1) Введите ваш запрос
2) Настройте модель по нужным вам параметрам
3) Нажмите кнопку `Submit`, чтобы получить сгенерированное изображение

### Lumina-T2X:

1) Введите ваш запрос
2) Настройте модель по нужным вам параметрам
3) Нажмите кнопку `Submit`, чтобы получить сгенерированное изображение

### Kolors:

1) Введите ваш запрос
2) Настройте модель по нужным вам параметрам
3) Нажмите кнопку `Submit`, чтобы получить сгенерированное изображение

### AuraFlow:

1) Введите ваш запрос
2) Настройте модель по нужным вам параметрам
3) Нажмите кнопку `Submit`, чтобы получить сгенерированное изображение

### Würstchen:

1) Введите ваш запрос
2) Настройте модель по нужным вам параметрам
3) Нажмите кнопку `Submit`, чтобы получить сгенерированное изображение

### DeepFloydIF (txt2img, img2img, inpaint):

1) Введите ваш запрос
2) Настройте модель по нужным вам параметрам
3) Нажмите кнопку `Submit`, чтобы получить сгенерированное изображение

### PixArt:

1) Введите ваш запрос
2) Выберите модель из выпадающего списка
3) Настройте модель по нужным вам параметрам
4) Нажмите кнопку `Submit`, чтобы получить сгенерированное изображение

### ModelScope:

1) Введите ваш запрос
2) Настройте модель по нужным вам параметрам
3) Нажмите кнопку `Submit`, чтобы получить сгенерированное видео

### ZeroScope 2:

1) Введите ваш запрос
2) Настройте модель по нужным вам параметрам
3) Нажмите кнопку `Submit`, чтобы получить сгенерированное видео

### CogVideoX:

1) Введите ваш запрос
2) Настройте модель по нужным вам параметрам
3) Нажмите кнопку `Submit`, чтобы получить сгенерированное видео

### Latte:

1) Введите ваш запрос
2) Настройте модель по нужным вам параметрам
3) Нажмите кнопку `Submit`, чтобы получить сгенерированное видео

### TripoSR:

1) Загрузите исходное изображение
2) Настройте модель по нужным вам параметрам
3) Нажмите кнопку `Submit`, чтобы получить сгенерированный 3D-объект

### StableFast3D:

1) Загрузите исходное изображение
2) Настройте модель по нужным вам параметрам
3) Нажмите кнопку `Submit`, чтобы получить сгенерированный 3D-объект

### Shap-E:

1) Введите запрос или загрузите исходное изображение
2) Настройте модель по нужным вам параметрам
3) Нажмите кнопку `Submit`, чтобы получить сгенерированный 3D-объект

### SV34D:

1) Загрузите исходное изображение (для 3D) или видео (для 4D)
2) Настройте модель по нужным вам параметрам
3) Нажмите кнопку `Submit`, чтобы получить сгенерированное 3D видео

### Zero123Plus:

1) Загрузите исходное изображение
2) Настройте модель по нужным вам параметрам
3) Нажмите кнопку `Submit`, чтобы получить сгенерированное 3D вращение изображения

### StableAudio:

1) Настройте модель по нужным вам параметрам
2) Введите ваш запрос
3) Нажмите кнопку `Submit`, чтобы получить сгенерированное аудио

### AudioCraft:

1) Выберите модель из выпадающего списка
2) Выберите тип модели (`musicgen`, `audiogen` или `magnet`)
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

### Галерея:

* Здесь вы можете просмотреть файлы из папок outputs

### Загрузчик моделей:

* Здесь вы можете скачать модели `LLM` и `StableDiffusion`. Просто выберите модель из выпадающего списка и нажмите кнопку `Submit`
#### Модели `LLM` скачиваються сюда: *inputs/text/llm_models*
#### Модели `StableDiffusion` скачиваються сюда: *inputs/image/sd_models*

### Настройки:

* Здесь вы можете изменить настройки приложения. На данный момент вы можете изменить только режим `Share` на `True` или `False`

### Система: 

* Здесь вы можете увидеть показатели датчиков вашего компьютера нажав на кнопку `Submit`

### Общее:

1) Все генерации сохраняются в папке *outputs*
2) Вы также можете нажать кнопку `Clear`, чтобы сбросить ваш выбор
3) Чтобы остановить процесс генерации, нажмите кнопку `Stop generation`
4) Вы также можете выключить приложение с помощью кнопки `Close terminal`
5) Вы можете открыть папку *outputs* нажав на кнопку `Outputs`

## Где я могу взять модели, голоса и аватары?

* LLM модели можно взять с сайта [HuggingFace](https://huggingface.co/models) или из внутреннего интерфейса ModelDownloader
* Модели StableDiffusion, vae, inpaint, embedding и lora можно взять с сайта [CivitAI](https://civitai.com/models) или из внутреннего интерфейса ModelDownloader
* StableAudioOpen, AudioCraft, AudioLDM 2, TTS, Whisper, Wav2Lip, SunoBark, MoonDream2, Upscale, GLIGEN, Depth, Pix2Pix, Controlnet, AnimateDiff, Videos, LDM3D, SD3, Cascade, InstantID, PhotoMaker, IP-Adapter-FaceID, Rembg, Roop, CodeFormer, Real-ESRGAN, TripoSR, StableFast3D, Shap-E, SV34D, Zero123Plus, Demucs, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, AuraSR, Würstchen, DeepFloydIF, PixArt, ModelScope, ZeroScope 2, CogVideoX, Latte и Multiband diffusion модели скачиваються автоматически в папку *inputs* при их использовании
* Вы можете использовать голоса откуда угодно. Запишите свой или возьмите запись из интернета. Или просто используйте те, которые уже есть в проекте. Главное, чтобы оно было предварительно обработано!

## Вики

* https://github.com/Dartvauder/NeuroSandboxWebUI/wiki

## Благодарность разработчикам

#### Большое спасибо этим проектам за то, что благодаря их приложениям/библиотекам, я смог создать свое приложение:

В первую очередь я хочу поблагодарить разработчиков [PyCharm](https://www.jetbrains.com/pycharm/) и [GitHub](https://desktop.github.com). С помощью их приложений, я смог создать и выложить свой код

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

## Сторонние лицензии:

#### У многих моделей есть свои собственные лицензии на использование. Перед тем как ее использовать, советую ознакомиться с ними:

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

## Пожертвование

### *Если вам понравился мой проект и вы хотите сделать пожертвование, вот варианты для пожертвований. Заранее большое спасибо!*

* КриптоКошелек(BEP-20) - 0x3d86bdb5f50b92d0d7Eb44F1a833acC5e91aAEcA

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Dartvauder/NeuroSandboxWebUI&type=Date)](https://star-history.com/#Dartvauder/NeuroSandboxWebUI&Date)
