## [Funkcje](/#Features) | [Zależności](/#Required-Dependencies) | [WymaganiaSystemowe](/#Minimum-System-Requirements) | [Instalacja](/#How-to-install) | [Użytkowanie](/#How-to-use) | [Modele](/#Where-can-I-get-models-voices-and-avatars) | [Wiki](/#Wiki) | [Podziękowania](/#Acknowledgment-to-developers) | [Licencje](/#Third-Party-Licenses)

# ![main](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4ea0d891-8979-45ad-b052-626c41ae991a)
* Praca w toku! (ALPHA)
* [English](/README.md) | [عربي](/Readmes/README_AR.md) | [Deutsche](/Readmes/README_DE.md) | [Español](/Readmes/README_ES.md) | [Français](/Readmes/README_FR.md) | [日本語](/Readmes/README_JP.md) | [Русский](/Readmes/README_RU.md) | [漢語](/Readmes/README_ZH.md) | [Português](/Readmes/README_PT.md) | [हिंदी](/Readmes/README_HI.md) | [Italiano](/Readmes/README_IT.md) | [韓國語](/Readmes/README_KO.md) | Polski | [Türkçe](/Readmes/README_TR.md)

## Opis:

Prosty i wygodny interfejs do korzystania z różnych modeli sieci neuronowych. Możesz komunikować się z LLM i Moondream2 za pomocą tekstu, głosu i wprowadzania obrazu; używać StableDiffusion, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF i PixArt do generowania obrazów; ModelScope, ZeroScope 2, CogVideoX i Latte do generowania wideo; TripoSR, StableFast3D, Shap-E, SV34D i Zero123Plus do generowania obiektów 3D; StableAudioOpen, AudioCraft i AudioLDM 2 do generowania muzyki i audio; CoquiTTS i SunoBark do konwersji tekstu na mowę; OpenAI-Whisper do konwersji mowy na tekst; Wav2Lip do synchronizacji ruchu warg; Roop do zamiany twarzy; Rembg do usuwania tła; CodeFormer do odnawiania twarzy; LibreTranslate do tłumaczenia tekstu; Demucs do separacji plików audio. Możesz również przeglądać pliki z katalogu wyjściowego w galerii, pobierać modele LLM i StableDiffusion, zmieniać ustawienia aplikacji wewnątrz interfejsu i sprawdzać czujniki systemowe.

Celem projektu jest stworzenie jak najprostszej w użyciu aplikacji do korzystania z modeli sieci neuronowych.

### Tekst: <img width="1115" alt="1pl" src="https://github.com/user-attachments/assets/d1f520e6-900b-42f7-9aa5-90d9fb5090b0">

### Obraz: <img width="1118" alt="2pl" src="https://github.com/user-attachments/assets/c712b411-3317-4e55-93c9-cd684271f5f5">

### Wideo: <img width="1118" alt="3pl" src="https://github.com/user-attachments/assets/ea17ffab-7cf2-4ff3-b48a-0b8aec9faaa0">

### 3D: <img width="1121" alt="4pl" src="https://github.com/user-attachments/assets/f971b18a-3e0d-4d42-8a79-0abb0ae2662e">

### Audio: <img width="1120" alt="5pl" src="https://github.com/user-attachments/assets/011dad91-e92c-4221-82c7-c058def17239">

### Interfejs: <img width="1118" alt="6pl" src="https://github.com/user-attachments/assets/7f4b676c-011e-431c-a2a0-a7e488466211">

## Funkcje:

* Łatwa instalacja za pomocą install.bat (Windows) lub install.sh (Linux)
* Możliwość korzystania z aplikacji za pomocą urządzenia mobilnego w sieci lokalnej (przez IPv4) lub w dowolnym miejscu online (przez Share)
* Elastyczny i zoptymalizowany interfejs (dzięki Gradio)
* Uwierzytelnianie przez admin:admin (Możesz wprowadzić swoje dane logowania w pliku GradioAuth.txt)
* Możliwość dodania własnego tokena HuggingFace do pobierania określonych modeli (Możesz wprowadzić swój token w pliku HF-Token.txt)
* Wsparcie dla modeli Transformers i llama.cpp (LLM)
* Wsparcie dla modeli diffusers i safetensors (StableDiffusion) - zakładki txt2img, img2img, depth2img, pix2pix, controlnet, upscale, inpaint, gligen, animatediff, video, ldm3d, sd3, cascade, adapters i extras
* Wsparcie dla dodatkowych modeli do generowania obrazów: Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF i PixArt
* Wsparcie dla StableAudioOpen
* Wsparcie dla AudioCraft (Modele: musicgen, audiogen i magnet)
* Wsparcie dla AudioLDM 2 (Modele: audio i music)
* Wsparcie dla modeli TTS i Whisper (dla LLM i TTS-STT)
* Wsparcie dla modeli Lora, Textual inversion (embedding), Vae, Img2img, Depth, Pix2Pix, Controlnet, Upscale (latent), Upscale (Real-ESRGAN), Inpaint, GLIGEN, AnimateDiff, Videos, LDM3D, SD3, Cascade, Adapters (InstantID, PhotoMaker, IP-Adapter-FaceID), Rembg, CodeFormer i Roop (dla StableDiffusion)
* Wsparcie dla modelu Multiband Diffusion (dla AudioCraft)
* Wsparcie dla LibreTranslate (lokalny API)
* Wsparcie dla ModelScope, ZeroScope 2, CogVideoX i Latte do generowania wideo
* Wsparcie dla SunoBark
* Wsparcie dla Demucs
* Wsparcie dla TripoSR, StableFast3D, Shap-E, SV34D i Zero123Plus do generowania obiektów 3D
* Wsparcie dla Wav2Lip
* Wsparcie dla Multimodal (Moondream 2), LORA (transformers) i WebSearch (z GoogleSearch) dla LLM
* Ustawienia modelu wewnątrz interfejsu
* Galeria
* ModelDownloader (dla LLM i StableDiffusion)
* Ustawienia aplikacji
* Możliwość przeglądania czujników systemowych

## Wymagane zależności:

* [Python](https://www.python.org/downloads/) (3.11)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (12.4) i [cuDNN](https://developer.nvidia.com/cudnn-downloads) (9.1)
* [FFMPEG](https://ffmpeg.org/download.html)
- Kompilator C+
  - Windows: [VisualStudio](https://visualstudio.microsoft.com/ru/)
  - Linux: [GCC](https://gcc.gnu.org/)

## Minimalne wymagania systemowe:

* System: Windows lub Linux
* GPU: 6GB+ lub CPU: 8 rdzeni 3.2GHZ
* RAM: 16GB+
* Miejsce na dysku: 20GB+
* Internet do pobierania modeli i instalacji

## Jak zainstalować:

### Windows

1) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` do dowolnej lokalizacji
2) Uruchom `Install.bat` i poczekaj na zakończenie instalacji
3) Po instalacji uruchom `Start.bat`
4) Wybierz wersję pliku i poczekaj na uruchomienie aplikacji
5) Teraz możesz zacząć generować!

Aby uzyskać aktualizację, uruchom `Update.bat`
Aby pracować z wirtualnym środowiskiem przez terminal, uruchom `Venv.bat`

### Linux

1) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` do dowolnej lokalizacji
2) W terminalu uruchom `./Install.sh` i poczekaj na zakończenie instalacji wszystkich zależności
3) Po instalacji uruchom `./Start.sh`
4) Poczekaj na uruchomienie aplikacji
5) Teraz możesz zacząć generować!

Aby uzyskać aktualizację, uruchom `./Update.sh`
Aby pracować z wirtualnym środowiskiem przez terminal, uruchom `./Venv.sh`

## Jak używać:

#### Interfejs ma trzydzieści dwie zakładki w sześciu głównych zakładkach (Tekst, Obraz, Wideo, 3D, Audio i Interfejs): LLM, TTS-STT, SunoBark, LibreTranslate, Wav2Lip, StableDiffusion, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF, PixArt, ModelScope, ZeroScope 2, CogVideoX, Latte, TripoSR, StableFast3D, Shap-E, SV34D, Zero123Plus, StableAudio, AudioCraft, AudioLDM 2, Demucs, Galeria, ModelDownloader, Ustawienia i System. Wybierz tę, której potrzebujesz i postępuj zgodnie z poniższymi instrukcjami

### LLM:

1) Najpierw załaduj swoje modele do folderu: *inputs/text/llm_models*
2) Wybierz swój model z listy rozwijanej
3) Wybierz typ modelu (`transformers` lub `llama`)
4) Skonfiguruj model zgodnie z potrzebnymi parametrami
5) Wpisz (lub powiedz) swoje zapytanie
6) Kliknij przycisk `Submit`, aby otrzymać wygenerowaną odpowiedź tekstową i dźwiękową
#### Opcjonalnie: możesz włączyć tryb `TTS`, wybrać potrzebny `głos` i `język`, aby otrzymać odpowiedź dźwiękową. Możesz włączyć `multimodal` i przesłać obraz, aby uzyskać jego opis. Możesz włączyć `websearch` dla dostępu do Internetu. Możesz włączyć `libretranslate`, aby uzyskać tłumaczenie. Możesz również wybrać model `LORA`, aby poprawić generowanie
#### Próbki głosowe = *inputs/audio/voices*
#### LORA = *inputs/text/llm_models/lora*
#### Głos musi być wstępnie przetworzony (22050 kHz, mono, WAV)

### TTS-STT:

1) Wpisz tekst do zamiany tekstu na mowę
2) Wprowadź audio do zamiany mowy na tekst
3) Kliknij przycisk `Submit`, aby otrzymać wygenerowaną odpowiedź tekstową i dźwiękową
#### Próbki głosowe = *inputs/audio/voices*
#### Głos musi być wstępnie przetworzony (22050 kHz, mono, WAV)

### SunoBark:

1) Wpisz swoje zapytanie
2) Skonfiguruj model zgodnie z potrzebnymi parametrami
3) Kliknij przycisk `Submit`, aby otrzymać wygenerowaną odpowiedź dźwiękową

### LibreTranslate:

* Najpierw musisz zainstalować i uruchomić [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate)
1) Wybierz język źródłowy i docelowy
2) Kliknij przycisk `Submit`, aby uzyskać tłumaczenie
#### Opcjonalnie: możesz zapisać historię tłumaczeń, włączając odpowiedni przycisk

### Wav2Lip:

1) Prześlij początkowy obraz twarzy
2) Prześlij początkowe audio głosu
3) Skonfiguruj model zgodnie z potrzebnymi parametrami
4) Kliknij przycisk `Submit`, aby otrzymać synchronizację ruchu warg

### StableDiffusion - ma szesnaście podzakładek:

#### txt2img:

1) Najpierw załaduj swoje modele do folderu: *inputs/image/sd_models*
2) Wybierz swój model z listy rozwijanej
3) Wybierz typ modelu (`SD`, `SD2` lub `SDXL`)
4) Skonfiguruj model zgodnie z potrzebnymi parametrami
5) Wprowadź swoje zapytanie (+ i - dla ważenia prompt)
6) Kliknij przycisk `Submit`, aby uzyskać wygenerowany obraz
#### Opcjonalnie: Możesz wybrać swoje modele `vae`, `embedding` i `lora`, aby poprawić metodę generowania, możesz też włączyć `upscale`, aby zwiększyć rozmiar wygenerowanego obrazu
#### vae = *inputs/image/sd_models/vae*
#### lora = *inputs/image/sd_models/lora*
#### embedding = *inputs/image/sd_models/embedding*

#### img2img:

1) Najpierw załaduj swoje modele do folderu: *inputs/image/sd_models*
2) Wybierz swój model z listy rozwijanej
3) Wybierz typ modelu (`SD`, `SD2` lub `SDXL`)
4) Skonfiguruj model zgodnie z potrzebnymi parametrami
5) Prześlij początkowy obraz, na podstawie którego odbędzie się generowanie
6) Wprowadź swoje zapytanie (+ i - dla ważenia prompt)
7) Kliknij przycisk `Submit`, aby uzyskać wygenerowany obraz
#### Opcjonalnie: Możesz wybrać swój model `vae`
#### vae = *inputs/image/sd_models/vae*

#### depth2img:

1) Prześlij początkowy obraz
2) Skonfiguruj model zgodnie z potrzebnymi parametrami
3) Wprowadź swoje zapytanie (+ i - dla ważenia prompt)
4) Kliknij przycisk `Submit`, aby uzyskać wygenerowany obraz

#### pix2pix:

1) Prześlij początkowy obraz
2) Skonfiguruj model zgodnie z potrzebnymi parametrami
3) Wprowadź swoje zapytanie (+ i - dla ważenia prompt)
4) Kliknij przycisk `Submit`, aby uzyskać wygenerowany obraz

#### controlnet:

1) Najpierw załaduj swoje modele StableDiffusion do folderu: *inputs/image/sd_models*
2) Prześlij początkowy obraz
3) Wybierz swoje modele StableDiffusion i ControlNet z list rozwijanych
4) Skonfiguruj modele zgodnie z potrzebnymi parametrami
5) Wprowadź swoje zapytanie (+ i - dla ważenia prompt)
6) Kliknij przycisk `Submit`, aby uzyskać wygenerowany obraz

#### upscale(latent):

1) Prześlij początkowy obraz
2) Skonfiguruj model zgodnie z potrzebnymi parametrami
3) Kliknij przycisk `Submit`, aby uzyskać powiększony obraz

#### upscale(Real-ESRGAN):

1) Prześlij początkowy obraz
2) Skonfiguruj model zgodnie z potrzebnymi parametrami
3) Kliknij przycisk `Submit`, aby uzyskać powiększony obraz

#### inpaint:

1) Najpierw załaduj swoje modele do folderu: *inputs/image/sd_models/inpaint*
2) Wybierz swój model z listy rozwijanej
3) Wybierz typ modelu (`SD`, `SD2` lub `SDXL`)
4) Skonfiguruj model zgodnie z potrzebnymi parametrami
5) Prześlij obraz, na którym będzie przeprowadzane generowanie, do `initial image` i `mask image`
6) W `mask image` wybierz pędzel, następnie paletę i zmień kolor na `#FFFFFF`
7) Narysuj miejsce do generowania i wprowadź swoje zapytanie (+ i - dla ważenia prompt)
8) Kliknij przycisk `Submit`, aby uzyskać uzupełniony obraz
#### Opcjonalnie: Możesz wybrać swój model `vae`
#### vae = *inputs/image/sd_models/vae*

#### gligen:

1) Najpierw załaduj swoje modele do folderu: *inputs/image/sd_models*
2) Wybierz swój model z listy rozwijanej
3) Wybierz typ modelu (`SD`, `SD2` lub `SDXL`)
4) Skonfiguruj model zgodnie z potrzebnymi parametrami
5) Wprowadź swoje zapytanie dla prompt (+ i - dla ważenia prompt) i frazy GLIGEN (w "" dla box)
6) Wprowadź pola GLIGEN (Na przykład [0.1387, 0.2051, 0.4277, 0.7090] dla box)
7) Kliknij przycisk `Submit`, aby uzyskać wygenerowany obraz

#### animatediff:

1) Najpierw załaduj swoje modele do folderu: *inputs/image/sd_models*
2) Wybierz swój model z listy rozwijanej
3) Skonfiguruj model zgodnie z potrzebnymi parametrami
4) Wprowadź swoje zapytanie (+ i - dla ważenia prompt)
5) Kliknij przycisk `Submit`, aby uzyskać wygenerowaną animację obrazu

#### video:

1) Prześlij początkowy obraz
2) Wprowadź swoje zapytanie (dla IV2Gen-XL)
3) Skonfiguruj model zgodnie z potrzebnymi parametrami
4) Kliknij przycisk `Submit`, aby uzyskać wideo z obrazu

#### ldm3d:

1) Wprowadź swoje zapytanie
2) Skonfiguruj model zgodnie z potrzebnymi parametrami
3) Kliknij przycisk `Submit`, aby uzyskać wygenerowane obrazy

#### sd3 (txt2img, img2img, controlnet, inpaint):

1) Wprowadź swoje zapytanie
2) Skonfiguruj model zgodnie z potrzebnymi parametrami
3) Kliknij przycisk `Submit`, aby uzyskać wygenerowany obraz

#### cascade:

1) Wprowadź swoje zapytanie
2) Skonfiguruj model zgodnie z potrzebnymi parametrami
3) Kliknij przycisk `Submit`, aby uzyskać wygenerowany obraz

#### adapters (InstantID, PhotoMaker i IP-Adapter-FaceID):

1) Najpierw załaduj swoje modele do folderu: *inputs/image/sd_models*
2) Prześlij początkowy obraz
3) Wybierz swój model z listy rozwijanej
4) Skonfiguruj model zgodnie z potrzebnymi parametrami
5) Wybierz podzakładkę, której potrzebujesz
6) Kliknij przycisk `Submit`, aby uzyskać zmodyfikowany obraz

#### extras:

1) Prześlij początkowy obraz
2) Wybierz potrzebne opcje
3) Kliknij przycisk `Submit`, aby uzyskać zmodyfikowany obraz

### Kandinsky (txt2img, img2img, inpaint):

1) Wprowadź swój prompt
2) Wybierz model z listy rozwijanej
3) Skonfiguruj model zgodnie z potrzebnymi parametrami
4) Kliknij `Submit`, aby uzyskać wygenerowany obraz

### Flux:

1) Wprowadź swój prompt
2) Wybierz model z listy rozwijanej
3) Skonfiguruj model zgodnie z potrzebnymi parametrami
4) Kliknij `Submit`, aby uzyskać wygenerowany obraz

### HunyuanDiT:

1) Wprowadź swój prompt
2) Skonfiguruj model zgodnie z potrzebnymi parametrami
3) Kliknij `Submit`, aby uzyskać wygenerowany obraz

### Lumina-T2X:

1) Wprowadź swój prompt
2) Skonfiguruj model zgodnie z potrzebnymi parametrami
3) Kliknij `Submit`, aby uzyskać wygenerowany obraz

### Kolors:

1) Wprowadź swój prompt
2) Skonfiguruj model zgodnie z potrzebnymi parametrami
3) Kliknij `Submit`, aby uzyskać wygenerowany obraz

### AuraFlow:

1) Wprowadź swój prompt
2) Skonfiguruj model zgodnie z potrzebnymi parametrami
3) Kliknij `Submit`, aby uzyskać wygenerowany obraz

### Würstchen:

1) Wprowadź swój prompt
2) Skonfiguruj model zgodnie z potrzebnymi parametrami
3) Kliknij `Submit`, aby uzyskać wygenerowany obraz

### DeepFloydIF (txt2img, img2img, inpaint):

1) Wprowadź swój prompt
2) Skonfiguruj model zgodnie z potrzebnymi parametrami
3) Kliknij `Submit`, aby uzyskać wygenerowany obraz

### PixArt:

1) Wprowadź swój prompt
2) Wybierz model z listy rozwijanej
3) Skonfiguruj model zgodnie z potrzebnymi parametrami
4) Kliknij `Submit`, aby uzyskać wygenerowany obraz

### ModelScope:

1) Wprowadź swój prompt
2) Skonfiguruj model zgodnie z potrzebnymi parametrami
3) Kliknij `Submit`, aby uzyskać wygenerowane wideo

### ZeroScope 2:

1) Wprowadź swój prompt
2) Skonfiguruj model zgodnie z potrzebnymi parametrami
3) Kliknij `Submit`, aby uzyskać wygenerowane wideo

### CogVideoX:

1) Wprowadź swój prompt
2) Skonfiguruj model zgodnie z potrzebnymi parametrami
3) Kliknij `Submit`, aby uzyskać wygenerowane wideo

### Latte:

1) Wprowadź swój prompt
2) Skonfiguruj model zgodnie z potrzebnymi parametrami
3) Kliknij `Submit`, aby uzyskać wygenerowane wideo

### TripoSR:

1) Prześlij początkowy obraz
2) Skonfiguruj model zgodnie z potrzebnymi parametrami
3) Kliknij przycisk `Submit`, aby uzyskać wygenerowany obiekt 3D

### StableFast3D:

1) Prześlij początkowy obraz
2) Skonfiguruj model zgodnie z potrzebnymi parametrami
3) Kliknij przycisk `Submit`, aby uzyskać wygenerowany obiekt 3D

### Shap-E:

1) Wprowadź swoje zapytanie lub prześlij początkowy obraz
2) Skonfiguruj model zgodnie z potrzebnymi parametrami
3) Kliknij przycisk `Submit`, aby uzyskać wygenerowany obiekt 3D

### SV34D:

1) Prześlij początkowy obraz (dla 3D) lub wideo (dla 4D)
2) Skonfiguruj model zgodnie z potrzebnymi parametrami
3) Kliknij przycisk `Submit`, aby uzyskać wygenerowane wideo 3D

### Zero123Plus:

1) Prześlij początkowy obraz
2) Skonfiguruj model zgodnie z potrzebnymi parametrami
3) Kliknij przycisk `Submit`, aby uzyskać wygenerowaną rotację 3D obrazu

### StableAudio:

1) Skonfiguruj model zgodnie z potrzebnymi parametrami
2) Wprowadź swoje zapytanie
3) Kliknij przycisk `Submit`, aby uzyskać wygenerowane audio

### AudioCraft:

1) Wybierz model z listy rozwijanej
2) Wybierz typ modelu (`musicgen`, `audiogen` lub `magnet`)
3) Skonfiguruj model zgodnie z potrzebnymi parametrami
4) Wprowadź swoje zapytanie
5) (Opcjonalnie) prześlij początkowe audio, jeśli używasz modelu `melody`
6) Kliknij przycisk `Submit`, aby uzyskać wygenerowane audio
#### Opcjonalnie: Możesz włączyć `multiband diffusion`, aby poprawić wygenerowane audio

### AudioLDM 2:

1) Wybierz model z listy rozwijanej
2) Skonfiguruj model zgodnie z potrzebnymi parametrami
3) Wprowadź swoje zapytanie
4) Kliknij przycisk `Submit`, aby uzyskać wygenerowane audio

### Demucs:

1) Prześlij początkowe audio do separacji
2) Kliknij przycisk `Submit`, aby uzyskać rozdzielone audio

### Galeria:

* Tutaj możesz przeglądać pliki z katalogu outputs

### ModelDownloader:

* Tutaj możesz pobrać modele `LLM` i `StableDiffusion`. Po prostu wybierz model z listy rozwijanej i kliknij przycisk `Submit`
#### Modele `LLM` są pobierane tutaj: *inputs/text/llm_models*
#### Modele `StableDiffusion` są pobierane tutaj: *inputs/image/sd_models*

### Ustawienia: 

* Tutaj możesz zmienić ustawienia aplikacji. Na razie możesz tylko zmienić tryb `Share` na `True` lub `False`

### System: 

* Tutaj możesz zobaczyć wskaźniki czujników twojego komputera, klikając przycisk `Submit`

### Dodatkowe informacje:

1) Wszystkie generacje są zapisywane w folderze *outputs*
2) Możesz nacisnąć przycisk `Clear`, aby zresetować swój wybór
3) Aby zatrzymać proces generowania, kliknij przycisk `Stop generation`
4) Możesz wyłączyć aplikację za pomocą przycisku `Close terminal`
5) Możesz otworzyć folder *outputs*, klikając przycisk `Outputs`

## Gdzie mogę znaleźć modele i głosy?

* Modele LLM można znaleźć na [HuggingFace](https://huggingface.co/models) lub w ModelDownloader wewnątrz interfejsu 
* Modele StableDiffusion, vae, inpaint, embedding i lora można znaleźć na [CivitAI](https://civitai.com/models) lub w ModelDownloader wewnątrz interfejsu
* Modele StableAudioOpen, AudioCraft, AudioLDM 2, TTS, Whisper, Wav2Lip, SunoBark, MoonDream2, Upscale, GLIGEN, Depth, Pix2Pix, Controlnet, AnimateDiff, Videos, LDM3D, SD3, Cascade, InstantID, PhotoMaker, IP-Adapter-FaceID, Rembg, Roop, CodeFormer, Real-ESRGAN, TripoSR, StableFast3D, Shap-E, SV34D, Zero123Plus, Demucs, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, AuraSR, Würstchen, DeepFloydIF, PixArt, ModelScope, ZeroScope 2, CogVideoX, Latte i Multiband diffusion są pobierane automatycznie do folderu *inputs*, gdy są używane 
* Głosy możesz znaleźć wszędzie. Nagraj swoje własne lub weź nagranie z Internetu. Albo po prostu użyj tych, które są już w projekcie. Najważniejsze, żeby były wstępnie przetworzone!

## Wiki

* https://github.com/Dartvauder/NeuroSandboxWebUI/wiki

## Podziękowania dla programistów

#### Wielkie podziękowania dla tych projektów, ponieważ dzięki ich aplikacjom/bibliotekom mogłem stworzyć moją aplikację:

Przede wszystkim chcę podziękować programistom [PyCharm](https://www.jetbrains.com/pycharm/) i [GitHub](https://desktop.github.com). Dzięki ich aplikacjom mogłem stworzyć i udostępnić mój kod

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

## Licencje stron trzecich:

#### Wiele modeli ma własne licencje użytkowania. Przed użyciem zalecam zapoznanie się z nimi:

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

## Darowizna

### *Jeśli spodobał Ci się mój projekt i chcesz przekazać darowiznę, oto opcje. Z góry bardzo dziękuję!*

* Portfel kryptowalut (BEP-20) - 0x3d86bdb5f50b92d0d7Eb44F1a833acC5e91aAEcA

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Dartvauder/NeuroSandboxWebUI&type=Date)](https://star-history.com/#Dartvauder/NeuroSandboxWebUI&Date)
