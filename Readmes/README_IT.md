## [Caratteristiche](/#Caratteristiche) | [Dipendenze](/#Dipendenze-Richieste) | [RequisitiDiSistema](/#Requisiti-Minimi-di-Sistema) | [Installazione](/#Come-installare) | [Utilizzo](/#Come-utilizzare) | [Modelli](/#Dove-posso-ottenere-modelli-voci-e-avatar) | [Wiki](/#Wiki) | [Ringraziamenti](/#Ringraziamenti-agli-sviluppatori) | [Licenze](/#Licenze-di-Terze-Parti)

# ![main](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4ea0d891-8979-45ad-b052-626c41ae991a)
* Lavori in corso! (ALPHA)
* [English](/README.md) | [عربي](/Readmes/README_AR.md) | [Deutsche](/Readmes/README_DE.md) | [Español](/Readmes/README_ES.md) | [Français](/Readmes/README_FR.md) | [日本語](/Readmes/README_JP.md) | [Русский](/Readmes/README_RU.md) | [漢語](/Readmes/README_ZH.md) | [Português](/Readmes/README_PT.md) | [हिंदी](/Readmes/README_HI.md) | Italiano | [韓國語](/Readmes/README_KO.md)

## Descrizione:

Un'interfaccia semplice e comoda per l'utilizzo di vari modelli di reti neurali. Puoi comunicare con LLM e Moondream2 utilizzando input di testo, voce e immagine; utilizzare StableDiffusion, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF e PixArt per generare immagini; ModelScope, ZeroScope 2, CogVideoX e Latte per generare video; TripoSR, StableFast3D, Shap-E, SV34D e Zero123Plus per generare oggetti 3D; StableAudioOpen, AudioCraft e AudioLDM 2 per generare musica e audio; CoquiTTS e SunoBark per la sintesi vocale; OpenAI-Whisper per il riconoscimento vocale; Wav2Lip per la sincronizzazione labiale; Roop per lo scambio di volti; Rembg per rimuovere lo sfondo; CodeFormer per il restauro del viso; LibreTranslate per la traduzione del testo; Demucs per la separazione dei file audio. Puoi anche visualizzare i file dalla directory degli output nella galleria, scaricare i modelli LLM e StableDiffusion, modificare le impostazioni dell'applicazione all'interno dell'interfaccia e controllare i sensori di sistema

L'obiettivo del progetto è creare l'applicazione più semplice possibile per utilizzare modelli di reti neurali

### Testo: <img width="1124" alt="1it" src="https://github.com/user-attachments/assets/e9940ff8-c84a-41d2-b798-ffcfba921096">

### Immagine: <img width="1120" alt="2it" src="https://github.com/user-attachments/assets/43eb0f85-0509-4780-99d2-7c930b70015a">

### Video: <img width="1118" alt="3it" src="https://github.com/user-attachments/assets/fc50d218-46cb-4f76-8315-807d6164547f">

### 3D: <img width="1121" alt="4it" src="https://github.com/user-attachments/assets/c607c8a4-1926-457f-8e1e-0cb40b247db2">

### Audio: <img width="1120" alt="5it" src="https://github.com/user-attachments/assets/ed2f77e4-f523-49f9-a3a0-03ab2e5445ac">

### Interfaccia: <img width="1119" alt="6it" src="https://github.com/user-attachments/assets/f14650c0-1b56-42a7-aa56-68955c0e3e89">

## Caratteristiche:

* Facile installazione tramite install.bat (Windows) o install.sh (Linux)
* Puoi utilizzare l'applicazione tramite il tuo dispositivo mobile in localhost (tramite IPv4) o ovunque online (tramite Share)
* Interfaccia flessibile e ottimizzata (tramite Gradio)
* Autenticazione tramite admin:admin (Puoi inserire i tuoi dati di accesso nel file GradioAuth.txt)
* Puoi aggiungere il tuo HuggingFace-Token per scaricare modelli specifici (Puoi inserire il tuo token nel file HF-Token.txt)
* Supporto per modelli Transformers e llama.cpp (LLM)
* Supporto per modelli diffusers e safetensors (StableDiffusion) - schede txt2img, img2img, depth2img, pix2pix, controlnet, upscale, inpaint, gligen, animatediff, video, ldm3d, sd3, cascade ed extras
* Supporto di modelli aggiuntivi per la generazione di immagini: Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF e PixArt
* Supporto StableAudioOpen
* Supporto AudioCraft (Modelli: musicgen, audiogen e magnet)
* Supporto AudioLDM 2 (Modelli: audio e music)
* Supporta modelli TTS e Whisper (Per LLM e TTS-STT)
* Supporta modelli Lora, Textual inversion (embedding), Vae, Img2img, Depth, Pix2Pix, Controlnet, Upscale(latent), Upscale(Real-ESRGAN), Inpaint, GLIGEN, AnimateDiff, Videos, LDM3D, SD3, Cascade, Rembg, CodeFormer e Roop (Per StableDiffusion)
* Supporto modello Multiband Diffusion (Per AudioCraft)
* Supporto LibreTranslate (API locale)
* Supporto ModelScope, ZeroScope 2, CogVideoX e Latte per la generazione di video
* Supporto SunoBark
* Supporto Demucs
* Supporto TripoSR, StableFast3D, Shap-E, SV34D e Zero123Plus per la generazione 3D
* Supporto Wav2Lip
* Supporto Multimodale (Moondream 2), LORA (transformers) e WebSearch (con GoogleSearch) per LLM
* Impostazioni del modello all'interno dell'interfaccia
* Galleria
* ModelDownloader (Per LLM e StableDiffusion)
* Impostazioni dell'applicazione
* Possibilità di vedere i sensori di sistema

## Dipendenze Richieste:

* [Python](https://www.python.org/downloads/) (3.11)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (12.4) e [cuDNN](https://developer.nvidia.com/cudnn-downloads) (9.1)
* [FFMPEG](https://ffmpeg.org/download.html)
- Compilatore C+
  - Windows: [VisualStudio](https://visualstudio.microsoft.com/ru/)
  - Linux: [GCC](https://gcc.gnu.org/)

## Requisiti Minimi di Sistema:

* Sistema: Windows o Linux
* GPU: 6GB+ o CPU: 8 core 3.2GHZ
* RAM: 16GB+
* Spazio su disco: 20GB+
* Internet per scaricare modelli e installare

## Come installare:

### Windows

1) Eseguire `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` in qualsiasi posizione
2) Eseguire `Install.bat` e attendere l'installazione
3) Dopo l'installazione, eseguire `Start.bat`
4) Selezionare la versione del file e attendere l'avvio dell'applicazione
5) Ora puoi iniziare a generare!

Per ottenere l'aggiornamento, eseguire `Update.bat`
Per lavorare con l'ambiente virtuale tramite il terminale, eseguire `Venv.bat`

### Linux

1) Eseguire `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` in qualsiasi posizione
2) Nel terminale, eseguire `./Install.sh` e attendere l'installazione di tutte le dipendenze
3) Dopo l'installazione, eseguire `./Start.sh`
4) Attendere l'avvio dell'applicazione
5) Ora puoi iniziare a generare!

Per ottenere l'aggiornamento, eseguire `./Update.sh`
Per lavorare con l'ambiente virtuale tramite il terminale, eseguire `./Venv.sh`

## Come utilizzare:

#### L'interfaccia ha trentadue schede in sei schede principali: LLM, TTS-STT, SunoBark, LibreTranslate, Wav2Lip, StableDiffusion, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF, PixArt, ModelScope, ZeroScope 2, CogVideoX, Latte, TripoSR, StableFast3D, Shap-E, SV34D, Zero123Plus, StableAudio, AudioCraft, AudioLDM 2, Demucs, Gallery, ModelDownloader, Settings e System. Seleziona quella di cui hai bisogno e segui le istruzioni di seguito

### LLM:

1) Per prima cosa carica i tuoi modelli nella cartella: *inputs/text/llm_models*
2) Seleziona il tuo modello dall'elenco a discesa
3) Seleziona il tipo di modello (`transformers` o `llama`)
4) Configura il modello secondo i parametri di cui hai bisogno
5) Digita (o parla) la tua richiesta
6) Clicca sul pulsante `Submit` per ricevere la risposta testuale e audio generata
#### Opzionale: puoi abilitare la modalità `TTS`, selezionare la `voce` e la `lingua` necessarie per ricevere una risposta audio. Puoi abilitare `multimodal` e caricare un'immagine per ottenere la sua descrizione. Puoi abilitare `websearch` per l'accesso a Internet. Puoi abilitare `libretranslate` per ottenere la traduzione. Inoltre puoi scegliere il modello `LORA` per migliorare la generazione
#### Campioni vocali = *inputs/audio/voices*
#### LORA = *inputs/text/llm_models/lora*
#### La voce deve essere pre-elaborata (22050 kHz, mono, WAV)

### TTS-STT:

1) Digita il testo per la sintesi vocale
2) Inserisci l'audio per il riconoscimento vocale
3) Clicca sul pulsante `Submit` per ricevere la risposta testuale e audio generata
#### Campioni vocali = *inputs/audio/voices*
#### La voce deve essere pre-elaborata (22050 kHz, mono, WAV)

### SunoBark:

1) Digita la tua richiesta
2) Configura il modello secondo i parametri di cui hai bisogno
3) Clicca sul pulsante `Submit` per ricevere la risposta audio generata

### LibreTranslate:

* Prima devi installare ed eseguire [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate)
1) Seleziona le lingue di origine e di destinazione
2) Clicca sul pulsante `Submit` per ottenere la traduzione
#### Opzionale: puoi salvare la cronologia delle traduzioni attivando il pulsante corrispondente

### Wav2Lip:

1) Carica l'immagine iniziale del volto
2) Carica l'audio iniziale della voce
3) Configura il modello secondo i parametri di cui hai bisogno
4) Clicca sul pulsante `Submit` per ricevere la sincronizzazione labiale

### StableDiffusion - ha quindici sotto-schede:

#### txt2img:

1) Per prima cosa carica i tuoi modelli nella cartella: *inputs/image/sd_models*
2) Seleziona il tuo modello dall'elenco a discesa
3) Seleziona il tipo di modello (`SD`, `SD2` o `SDXL`)
4) Configura il modello secondo i parametri di cui hai bisogno
5) Inserisci la tua richiesta (+ e - per la ponderazione del prompt)
6) Clicca sul pulsante `Submit` per ottenere l'immagine generata
#### Opzionale: Puoi selezionare i tuoi modelli `vae`, `embedding` e `lora` per migliorare il metodo di generazione, inoltre puoi abilitare `upscale` per aumentare le dimensioni dell'immagine generata
#### vae = *inputs/image/sd_models/vae*
#### lora = *inputs/image/sd_models/lora*
#### embedding = *inputs/image/sd_models/embedding*

#### img2img:

1) Per prima cosa carica i tuoi modelli nella cartella: *inputs/image/sd_models*
2) Seleziona il tuo modello dall'elenco a discesa
3) Seleziona il tipo di modello (`SD`, `SD2` o `SDXL`)
4) Configura il modello secondo i parametri di cui hai bisogno
5) Carica l'immagine iniziale con cui avverrà la generazione
6) Inserisci la tua richiesta (+ e - per la ponderazione del prompt)
7) Clicca sul pulsante `Submit` per ottenere l'immagine generata
#### Opzionale: Puoi selezionare il tuo modello `vae`
#### vae = *inputs/image/sd_models/vae*

#### depth2img:

1) Carica l'immagine iniziale
2) Configura il modello secondo i parametri di cui hai bisogno
3) Inserisci la tua richiesta (+ e - per la ponderazione del prompt)
4) Clicca sul pulsante `Submit` per ottenere l'immagine generata

#### pix2pix:

1) Carica l'immagine iniziale
2) Configura il modello secondo i parametri di cui hai bisogno
3) Inserisci la tua richiesta (+ e - per la ponderazione del prompt)
4) Clicca sul pulsante `Submit` per ottenere l'immagine generata

#### controlnet:

1) Per prima cosa carica i tuoi modelli stable diffusion nella cartella: *inputs/image/sd_models*
2) Carica l'immagine iniziale
3) Seleziona i tuoi modelli stable diffusion e controlnet dagli elenchi a discesa
4) Configura i modelli secondo i parametri di cui hai bisogno
5) Inserisci la tua richiesta (+ e - per la ponderazione del prompt)
6) Clicca sul pulsante `Submit` per ottenere l'immagine generata

#### upscale(latent):

1) Carica l'immagine iniziale
2) Configura il modello secondo i parametri di cui hai bisogno
3) Clicca sul pulsante `Submit` per ottenere l'immagine ingrandita

#### upscale(Real-ESRGAN):

1) Carica l'immagine iniziale
2) Configura il modello secondo i parametri di cui hai bisogno
3) Clicca sul pulsante `Submit` per ottenere l'immagine ingrandita

#### inpaint:

1) Per prima cosa carica i tuoi modelli nella cartella: *inputs/image/sd_models/inpaint*
2) Seleziona il tuo modello dall'elenco a discesa
3) Seleziona il tipo di modello (`SD`, `SD2` o `SDXL`)
4) Configura il modello secondo i parametri di cui hai bisogno
5) Carica l'immagine con cui avverrà la generazione in `initial image` e `mask image`
6) In `mask image`, seleziona il pennello, poi la tavolozza e cambia il colore in `#FFFFFF`
7) Disegna un'area per la generazione e inserisci la tua richiesta (+ e - per la ponderazione del prompt)
8) Clicca sul pulsante `Submit` per ottenere l'immagine con inpainting
#### Opzionale: Puoi selezionare il tuo modello `vae`
#### vae = *inputs/image/sd_models/vae*

#### gligen:

1) Per prima cosa carica i tuoi modelli nella cartella: *inputs/image/sd_models*
2) Seleziona il tuo modello dall'elenco a discesa
3) Seleziona il tipo di modello (`SD`, `SD2` o `SDXL`)
4) Configura il modello secondo i parametri di cui hai bisogno
5) Inserisci la tua richiesta per il prompt (+ e - per la ponderazione del prompt) e le frasi GLIGEN (in "" per il box)
6) Inserisci i box GLIGEN (Come [0.1387, 0.2051, 0.4277, 0.7090] per il box)
7) Clicca sul pulsante `Submit` per ottenere l'immagine generata

#### animatediff:

1) Per prima cosa carica i tuoi modelli nella cartella: *inputs/image/sd_models*
2) Seleziona il tuo modello dall'elenco a discesa
3) Configura il modello secondo i parametri di cui hai bisogno
4) Inserisci la tua richiesta (+ e - per la ponderazione del prompt)
5) Clicca sul pulsante `Submit` per ottenere l'animazione dell'immagine generata

#### video:

1) Carica l'immagine iniziale
2) Inserisci la tua richiesta (per IV2Gen-XL)
3) Configura il modello secondo i parametri di cui hai bisogno
4) Clicca sul pulsante `Submit` per ottenere il video dall'immagine

#### ldm3d:

1) Inserisci la tua richiesta
2) Configura il modello secondo i parametri di cui hai bisogno
3) Clicca sul pulsante `Submit` per ottenere le immagini generate

#### sd3:

1) Inserisci la tua richiesta
2) Configura il modello secondo i parametri di cui hai bisogno
3) Clicca sul pulsante `Submit` per ottenere l'immagine generata

#### cascade:

1) Inserisci la tua richiesta
2) Configura il modello secondo i parametri di cui hai bisogno
3) Clicca sul pulsante `Submit` per ottenere l'immagine generata

#### extras:

1) Carica l'immagine iniziale
2) Seleziona le opzioni di cui hai bisogno
3) Clicca sul pulsante `Submit` per ottenere l'immagine modificata

### Kandinsky:

1) Inserisci il tuo prompt
2) Seleziona un modello dall'elenco a discesa
3) Configura il modello secondo i parametri di cui hai bisogno
4) Clicca su `Submit` per ottenere l'immagine generata

### Flux:

1) Inserisci il tuo prompt
2) Seleziona un modello dall'elenco a discesa
3) Configura il modello secondo i parametri di cui hai bisogno
4) Clicca su `Submit` per ottenere l'immagine generata

### HunyuanDiT:

1) Inserisci il tuo prompt
2) Configura il modello secondo i parametri di cui hai bisogno
3) Clicca su `Submit` per ottenere l'immagine generata

### Lumina-T2X:

1) Inserisci il tuo prompt
2) Configura il modello secondo i parametri di cui hai bisogno
3) Clicca su `Submit` per ottenere l'immagine generata

### Kolors:

1) Inserisci il tuo prompt
2) Configura il modello secondo i parametri di cui hai bisogno
3) Clicca su `Submit` per ottenere l'immagine generata

### AuraFlow:

1) Inserisci il tuo prompt
2) Configura il modello secondo i parametri di cui hai bisogno
3) Clicca su `Submit` per ottenere l'immagine generata

### Würstchen:

1) Inserisci il tuo prompt
2) Configura il modello secondo i parametri di cui hai bisogno
3) Clicca su `Submit` per ottenere l'immagine generata

### DeepFloydIF:

1) Inserisci il tuo prompt
2) Configura il modello secondo i parametri di cui hai bisogno
3) Clicca su `Submit` per ottenere l'immagine generata

### PixArt:

1) Inserisci il tuo prompt
2) Seleziona il modello dall'elenco a discesa
3) Configura il modello secondo i parametri di cui hai bisogno
4) Clicca su `Submit` per ottenere l'immagine generata

### ModelScope:

1) Inserisci il tuo prompt
2) Configura il modello secondo i parametri di cui hai bisogno
3) Clicca su `Submit` per ottenere il video generato

### ZeroScope 2:

1) Inserisci il tuo prompt
2) Configura il modello secondo i parametri di cui hai bisogno
3) Clicca su `Submit` per ottenere il video generato

### CogVideoX:

1) Inserisci il tuo prompt
2) Configura il modello secondo i parametri di cui hai bisogno
3) Clicca su `Submit` per ottenere il video generato

### Latte:

1) Inserisci il tuo prompt
2) Configura il modello secondo i parametri di cui hai bisogno
3) Clicca su `Submit` per ottenere il video generato

### TripoSR:

1) Carica l'immagine iniziale
2) Configura il modello secondo i parametri di cui hai bisogno
3) Clicca sul pulsante `Submit` per ottenere l'oggetto 3D generato

### StableFast3D:

1) Carica l'immagine iniziale
2) Configura il modello secondo i parametri di cui hai bisogno
3) Clicca sul pulsante `Submit` per ottenere l'oggetto 3D generato

### Shap-E:

1) Inserisci la tua richiesta o carica l'immagine iniziale
2) Configura il modello secondo i parametri di cui hai bisogno
3) Clicca sul pulsante `Submit` per ottenere l'oggetto 3D generato

### SV34D:

1) Carica l'immagine iniziale (per 3D) o il video (per 4D)
2) Configura il modello secondo i parametri di cui hai bisogno
3) Clicca sul pulsante `Submit` per ottenere il video 3D generato

### Zero123Plus:

1) Carica l'immagine iniziale
2) Configura il modello secondo i parametri di cui hai bisogno
3) Clicca sul pulsante `Submit` per ottenere la rotazione 3D generata dell'immagine

### StableAudio:

1) Configura il modello secondo i parametri di cui hai bisogno
2) Inserisci la tua richiesta
3) Clicca sul pulsante `Submit` per ottenere l'audio generato

### AudioCraft:

1) Seleziona un modello dall'elenco a discesa
2) Seleziona il tipo di modello (`musicgen`, `audiogen` o `magnet`)
3) Configura il modello secondo i parametri di cui hai bisogno
4) Inserisci la tua richiesta
5) (Opzionale) carica l'audio iniziale se stai utilizzando il modello `melody`
6) Clicca sul pulsante `Submit` per ottenere l'audio generato
#### Opzionale: Puoi abilitare `multiband diffusion` per migliorare l'audio generato

### AudioLDM 2:

1) Seleziona un modello dall'elenco a discesa
2) Configura il modello secondo i parametri di cui hai bisogno
3) Inserisci la tua richiesta
4) Clicca sul pulsante `Submit` per ottenere l'audio generato

### Demucs:

1) Carica l'audio iniziale da separare
2) Clicca sul pulsante `Submit` per ottenere l'audio separato

### Galleria:

* Qui puoi visualizzare i file dalla directory degli output

### ModelDownloader:

* Qui puoi scaricare modelli `LLM` e `StableDiffusion`. Basta scegliere il modello dall'elenco a discesa e cliccare sul pulsante `Submit`
#### I modelli `LLM` vengono scaricati qui: *inputs/text/llm_models*
#### I modelli `StableDiffusion` vengono scaricati qui: *inputs/image/sd_models*

### Impostazioni:

* Qui puoi modificare le impostazioni dell'applicazione. Per ora puoi solo cambiare la modalità `Share` in `True` o `False`

### Sistema:

* Qui puoi vedere gli indicatori dei sensori del tuo computer cliccando sul pulsante `Submit`

### Informazioni aggiuntive:

1) Tutte le generazioni vengono salvate nella cartella *outputs*
2) Puoi premere il pulsante `Clear` per ripristinare la tua selezione
3) Per interrompere il processo di generazione, clicca sul pulsante `Stop generation`
4) Puoi spegnere l'applicazione utilizzando il pulsante `Close terminal`
5) Puoi aprire la cartella *outputs* cliccando sul pulsante `Outputs`

## Dove posso ottenere modelli e voci?

* I modelli LLM possono essere presi da [HuggingFace](https://huggingface.co/models) o dal ModelDownloader all'interno dell'interfaccia
* I modelli StableDiffusion, vae, inpaint, embedding e lora possono essere presi da [CivitAI](https://civitai.com/models) o dal ModelDownloader all'interno dell'interfaccia
* I modelli StableAudioOpen, AudioCraft, AudioLDM 2, TTS, Whisper, Wav2Lip, SunoBark, MoonDream2, Upscale, GLIGEN, Depth, Pix2Pix, Controlnet, AnimateDiff, Videos, LDM3D, SD3, Cascade, Rembg, Roop, CodeFormer, Real-ESRGAN, TripoSR, StableFast3D, Shap-E, SV34D, Zero123Plus, Demucs, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF, PixArt, ModelScope, ZeroScope 2, CogVideoX, Latte e Multiband diffusion vengono scaricati automaticamente nella cartella *inputs* quando vengono utilizzati
* Puoi prendere le voci ovunque. Registra le tue o prendi una registrazione da Internet. Oppure usa semplicemente quelle che sono già nel progetto. L'importante è che siano pre-elaborate!

## Wiki

* https://github.com/Dartvauder/NeuroSandboxWebUI/wiki

## Ringraziamenti agli sviluppatori

#### Molte grazie a questi progetti perché grazie alle loro applicazioni/librerie, sono stato in grado di creare la mia applicazione:

Prima di tutto, voglio ringraziare gli sviluppatori di [PyCharm](https://www.jetbrains.com/pycharm/) e [GitHub](https://desktop.github.com). Con l'aiuto delle loro applicazioni, sono stato in grado di creare e condividere il mio codice

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

## Licenze di Terze Parti:

#### Molti modelli hanno la propria licenza d'uso. Prima di utilizzarli, ti consiglio di familiarizzare con esse:

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

## Donazione

### *Se ti è piaciuto il mio progetto e vuoi fare una donazione, ecco le opzioni per donare. Grazie mille in anticipo!*

* CryptoWallet(BEP-20) - 0x3d86bdb5f50b92d0d7Eb44F1a833acC5e91aAEcA

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Dartvauder/NeuroSandboxWebUI&type=Date)](https://star-history.com/#Dartvauder/NeuroSandboxWebUI&Date)