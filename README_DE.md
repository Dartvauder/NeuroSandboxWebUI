## [Funktionen](/#Features) | [Abhängigkeiten](/#Required-Dependencies) | [Systemanforderungen](/#Minimum-System-Requirements) | [Installation](/#How-to-install) | [Verwendung](/#How-to-use) | [Modelle](/#Where-can-I-get-models-voices-and-avatars) | [Wiki](/#Wiki) | [Anerkennung](/#Acknowledgment-to-developers) | [Lizenzen](/#Third-Party-Licenses)

# ![main](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4ea0d891-8979-45ad-b052-626c41ae991a)
* Arbeit in Arbeit! (ALPHA)
* [English](/README.md)  | [عربي](/README_AR.md) | Deutsche | [Español](/README_ES.md) | [Français](/README_FR.md) | [Русский](/README_RU.md) | [漢語](/README_ZH.md) | [Português](/README_PT.md)

## Beschreibung:

Eine einfache und bequeme Oberfläche zur Nutzung verschiedener neuronaler Netzwerkmodelle. Sie können mit LLM und Moondream2 über Text-, Sprach- und Bildeingabe kommunizieren; StableDiffusion, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF und PixArt zur Bilderzeugung verwenden; ModelScope, ZeroScope 2, CogVideoX und Latte zur Videoerzeugung; TripoSR, StableFast3D, Shap-E, SV34D und Zero123Plus zur Erzeugung von 3D-Objekten; StableAudioOpen, AudioCraft und AudioLDM 2 zur Erzeugung von Musik und Audio; CoquiTTS und SunoBark für Text-zu-Sprache; OpenAI-Whisper für Sprache-zu-Text; Wav2Lip für Lippensynchronisation; Roop zum Gesichtertausch; Rembg zum Entfernen des Hintergrunds; CodeFormer zur Gesichtswiederherstellung; LibreTranslate zur Textübersetzung; Demucs zur Trennung von Audiodateien. Sie können auch Dateien aus dem Ausgabeverzeichnis in der Galerie anzeigen, LLM- und StableDiffusion-Modelle herunterladen, die Anwendungseinstellungen innerhalb der Oberfläche ändern und Systemsensoren überprüfen

Das Ziel des Projekts ist es, die einfachstmögliche Anwendung zur Nutzung von neuronalen Netzwerkmodellen zu erstellen

### Text: <img width="1119" alt="1" src="https://github.com/user-attachments/assets/e1ac4e8e-feb2-484b-a399-61ddc8a098c1">

### Bild: <img width="1121" alt="2" src="https://github.com/user-attachments/assets/a5f2cbde-5812-45db-a58a-dbadda5a01ac">

### Video: <img width="1118" alt="3" src="https://github.com/user-attachments/assets/a568c3ed-3b00-4e21-b802-a3e63f6cf97c">

### 3D: <img width="1118" alt="4" src="https://github.com/user-attachments/assets/0ba23ac4-aecc-44e6-b252-1fc0f478c75e">

### Audio: <img width="1127" alt="5" src="https://github.com/user-attachments/assets/ea7f1bd0-ff85-4873-b9dd-cabd1cc89cee">

### Oberfläche: <img width="1120" alt="6" src="https://github.com/user-attachments/assets/81c4e40c-cf01-488d-adc8-7330f1edd610">

## Funktionen:

* Einfache Installation über install.bat (Windows) oder install.sh (Linux)
* Sie können die Anwendung über Ihr mobiles Gerät im localhost (über IPv4) oder überall online (über Share) nutzen
* Flexible und optimierte Oberfläche (durch Gradio)
* Authentifizierung über admin:admin (Sie können Ihre Anmeldedaten in der Datei GradioAuth.txt eingeben)
* Sie können Ihren eigenen HuggingFace-Token hinzufügen, um bestimmte Modelle herunterzuladen (Sie können Ihren Token in der Datei HF-Token.txt eingeben)
* Unterstützung für Transformers und llama.cpp Modelle (LLM)
* Unterstützung für diffusers und safetensors Modelle (StableDiffusion) - txt2img, img2img, depth2img, pix2pix, controlnet, upscale, inpaint, gligen, animatediff, video, ldm3d, sd3, cascade und extras Tabs
* Unterstützung zusätzlicher Modelle zur Bilderzeugung: Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF und PixArt
* StableAudioOpen Unterstützung
* AudioCraft Unterstützung (Modelle: musicgen, audiogen und magnet)
* AudioLDM 2 Unterstützung (Modelle: audio und music)
* Unterstützt TTS und Whisper Modelle (Für LLM und TTS-STT)
* Unterstützt Lora, Textual inversion (embedding), Vae, Img2img, Depth, Pix2Pix, Controlnet, Upscale(latent), Upscale(Real-ESRGAN), Inpaint, GLIGEN, AnimateDiff, Videos, LDM3D, SD3, Cascade, Rembg, CodeFormer und Roop Modelle (Für StableDiffusion)
* Unterstützung des Multiband Diffusion Modells (Für AudioCraft)
* Unterstützung von LibreTranslate (Lokale API)
* Unterstützung von ModelScope, ZeroScope 2, CogVideoX und Latte zur Videoerzeugung
* Unterstützung von SunoBark
* Unterstützung von Demucs
* Unterstützung von TripoSR, StableFast3D, Shap-E, SV34D und Zero123Plus für 3D-Erzeugung
* Unterstützung von Wav2Lip
* Unterstützung von Multimodal (Moondream 2), LORA (transformers) und WebSearch (mit GoogleSearch) für LLM
* Modelleinstellungen innerhalb der Oberfläche
* Galerie
* ModelDownloader (Für LLM und StableDiffusion)
* Anwendungseinstellungen
* Möglichkeit, Systemsensoren zu sehen

## Erforderliche Abhängigkeiten:

* [Python](https://www.python.org/downloads/) (3.11)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (12.4) und [cuDNN](https://developer.nvidia.com/cudnn-downloads) (9.1)
* [FFMPEG](https://ffmpeg.org/download.html)
- C+ Compiler
  - Windows: [VisualStudio](https://visualstudio.microsoft.com/ru/)
  - Linux: [GCC](https://gcc.gnu.org/)

## Minimale Systemanforderungen:

* System: Windows oder Linux
* GPU: 6GB+ oder CPU: 8 Kerne 3.2GHZ
* RAM: 16GB+
* Speicherplatz: 20GB+
* Internet zum Herunterladen von Modellen und zur Installation

## Installationsanleitung:

### Windows

1) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` an einen beliebigen Ort
2) Führen Sie `Install.bat` aus und warten Sie auf die Installation
3) Führen Sie nach der Installation `Start.bat` aus
4) Wählen Sie die Dateiversionen aus und warten Sie, bis die Anwendung gestartet ist
5) Jetzt können Sie mit der Generierung beginnen!

Um ein Update zu erhalten, führen Sie `Update.bat` aus
Um mit der virtuellen Umgebung über das Terminal zu arbeiten, führen Sie `Venv.bat` aus

### Linux

1) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` an einen beliebigen Ort
2) Führen Sie im Terminal `./Install.sh` aus und warten Sie auf die Installation aller Abhängigkeiten
3) Führen Sie nach der Installation `./Start.sh` aus
4) Warten Sie, bis die Anwendung gestartet ist
5) Jetzt können Sie mit der Generierung beginnen!

Um ein Update zu erhalten, führen Sie `./Update.sh` aus
Um mit der virtuellen Umgebung über das Terminal zu arbeiten, führen Sie `./Venv.sh` aus

## Verwendung:

#### Die Oberfläche hat zweiunddreißig Tabs in sechs Haupttabs: LLM, TTS-STT, SunoBark, LibreTranslate, Wav2Lip, StableDiffusion, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF, PixArt, ModelScope, ZeroScope 2, CogVideoX, Latte, TripoSR, StableFast3D, Shap-E, SV34D, Zero123Plus, StableAudio, AudioCraft, AudioLDM 2, Demucs, Galerie, ModelDownloader, Einstellungen und System. Wählen Sie den gewünschten aus und folgen Sie den Anweisungen unten

### LLM:

1) Laden Sie zuerst Ihre Modelle in den Ordner: *inputs/text/llm_models*
2) Wählen Sie Ihr Modell aus der Dropdown-Liste
3) Wählen Sie den Modelltyp (`transformers` oder `llama`)
4) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
5) Geben Sie (oder sprechen Sie) Ihre Anfrage ein
6) Klicken Sie auf die Schaltfläche `Submit`, um den generierten Text und die Audioantwort zu erhalten
#### Optional: Sie können den `TTS`-Modus aktivieren, die benötigte `Stimme` und `Sprache` auswählen, um eine Audioantwort zu erhalten. Sie können `multimodal` aktivieren und ein Bild hochladen, um dessen Beschreibung zu erhalten. Sie können `websearch` für Internetzugriff aktivieren. Sie können `libretranslate` aktivieren, um die Übersetzung zu erhalten. Außerdem können Sie das `LORA`-Modell wählen, um die Generierung zu verbessern
#### Stimmenbeispiele = *inputs/audio/voices*
#### LORA = *inputs/text/llm_models/lora*
#### Die Stimme muss vorverarbeitet sein (22050 kHz, mono, WAV)

### TTS-STT:

1) Geben Sie Text für Text-zu-Sprache ein
2) Geben Sie Audio für Sprache-zu-Text ein
3) Klicken Sie auf die Schaltfläche `Submit`, um den generierten Text und die Audioantwort zu erhalten
#### Stimmenbeispiele = *inputs/audio/voices*
#### Die Stimme muss vorverarbeitet sein (22050 kHz, mono, WAV)

### SunoBark:

1) Geben Sie Ihre Anfrage ein
2) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
3) Klicken Sie auf die Schaltfläche `Submit`, um die generierte Audioantwort zu erhalten

### LibreTranslate:

* Zuerst müssen Sie [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate) installieren und ausführen
1) Wählen Sie Quell- und Zielsprachen
2) Klicken Sie auf die Schaltfläche `Submit`, um die Übersetzung zu erhalten
#### Optional: Sie können den Übersetzungsverlauf speichern, indem Sie die entsprechende Schaltfläche aktivieren

### Wav2Lip:

1) Laden Sie das Ausgangsbild des Gesichts hoch
2) Laden Sie das Ausgangsaudio der Stimme hoch
3) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
4) Klicken Sie auf die Schaltfläche `Submit`, um die Lippensynchronisation zu erhalten

### StableDiffusion - hat fünfzehn Unter-Tabs:

#### txt2img:

1) Laden Sie zuerst Ihre Modelle in den Ordner: *inputs/image/sd_models*
2) Wählen Sie Ihr Modell aus der Dropdown-Liste
3) Wählen Sie den Modelltyp (`SD`, `SD2` oder `SDXL`)
4) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
5) Geben Sie Ihre Anfrage ein (+ und - für Prompt-Gewichtung)
6) Klicken Sie auf die Schaltfläche `Submit`, um das generierte Bild zu erhalten
#### Optional: Sie können Ihre `vae`, `embedding` und `lora` Modelle auswählen, um die Generierungsmethode zu verbessern, außerdem können Sie `upscale` aktivieren, um die Größe des generierten Bildes zu erhöhen
#### vae = *inputs/image/sd_models/vae*
#### lora = *inputs/image/sd_models/lora*
#### embedding = *inputs/image/sd_models/embedding*

#### img2img:

1) Laden Sie zuerst Ihre Modelle in den Ordner: *inputs/image/sd_models*
2) Wählen Sie Ihr Modell aus der Dropdown-Liste
3) Wählen Sie den Modelltyp (`SD`, `SD2` oder `SDXL`)
4) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
5) Laden Sie das Ausgangsbild hoch, mit dem die Generierung stattfinden soll
6) Geben Sie Ihre Anfrage ein (+ und - für Prompt-Gewichtung)
7) Klicken Sie auf die Schaltfläche `Submit`, um das generierte Bild zu erhalten
#### Optional: Sie können Ihr `vae` Modell auswählen
#### vae = *inputs/image/sd_models/vae*

#### depth2img:

1) Laden Sie das Ausgangsbild hoch
2) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
3) Geben Sie Ihre Anfrage ein (+ und - für Prompt-Gewichtung)
4) Klicken Sie auf die Schaltfläche `Submit`, um das generierte Bild zu erhalten

#### pix2pix:

1) Laden Sie das Ausgangsbild hoch
2) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
3) Geben Sie Ihre Anfrage ein (+ und - für Prompt-Gewichtung)
4) Klicken Sie auf die Schaltfläche `Submit`, um das generierte Bild zu erhalten

#### controlnet:

1) Laden Sie zuerst Ihre stable diffusion Modelle in den Ordner: *inputs/image/sd_models*
2) Laden Sie das Ausgangsbild hoch
3) Wählen Sie Ihre stable diffusion und controlnet Modelle aus den Dropdown-Listen
4) Richten Sie die Modelle gemäß den von Ihnen benötigten Parametern ein
5) Geben Sie Ihre Anfrage ein (+ und - für Prompt-Gewichtung)
6) Klicken Sie auf die Schaltfläche `Submit`, um das generierte Bild zu erhalten

#### upscale(latent):

1) Laden Sie das Ausgangsbild hoch
2) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
3) Klicken Sie auf die Schaltfläche `Submit`, um das hochskalierte Bild zu erhalten

#### upscale(Real-ESRGAN):

1) Laden Sie das Ausgangsbild hoch
2) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
3) Klicken Sie auf die Schaltfläche `Submit`, um das hochskalierte Bild zu erhalten

#### inpaint:

1) Laden Sie zuerst Ihre Modelle in den Ordner: *inputs/image/sd_models/inpaint*
2) Wählen Sie Ihr Modell aus der Dropdown-Liste
3) Wählen Sie den Modelltyp (`SD`, `SD2` oder `SDXL`)
4) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
5) Laden Sie das Bild, mit dem die Generierung stattfinden soll, in `initial image` und `mask image` hoch
6) Wählen Sie in `mask image` den Pinsel, dann die Palette und ändern Sie die Farbe auf `#FFFFFF`
7) Zeichnen Sie einen Bereich für die Generierung und geben Sie Ihre Anfrage ein (+ und - für Prompt-Gewichtung)
8) Klicken Sie auf die Schaltfläche `Submit`, um das inpainted Bild zu erhalten
#### Optional: Sie können Ihr `vae` Modell auswählen
#### vae = *inputs/image/sd_models/vae*

#### gligen:

1) Laden Sie zuerst Ihre Modelle in den Ordner: *inputs/image/sd_models*
2) Wählen Sie Ihr Modell aus der Dropdown-Liste
3) Wählen Sie den Modelltyp (`SD`, `SD2` oder `SDXL`)
4) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
5) Geben Sie Ihre Anfrage für den Prompt (+ und - für Prompt-Gewichtung) und GLIGEN-Phrasen (in "" für Box) ein
6) Geben Sie GLIGEN-Boxen ein (Wie z.B. [0.1387, 0.2051, 0.4277, 0.7090] für eine Box)
7) Klicken Sie auf die Schaltfläche `Submit`, um das generierte Bild zu erhalten

#### animatediff:

1) Laden Sie zuerst Ihre Modelle in den Ordner: *inputs/image/sd_models*
2) Wählen Sie Ihr Modell aus der Dropdown-Liste
3) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
4) Geben Sie Ihre Anfrage ein (+ und - für Prompt-Gewichtung)
5) Klicken Sie auf die Schaltfläche `Submit`, um die generierte Bildanimation zu erhalten

#### video:

1) Laden Sie das Ausgangsbild hoch
2) Geben Sie Ihre Anfrage ein (für IV2Gen-XL)
3) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
4) Klicken Sie auf die Schaltfläche `Submit`, um das Video aus dem Bild zu erhalten

#### ldm3d:

1) Geben Sie Ihre Anfrage ein
2) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
3) Klicken Sie auf die Schaltfläche `Submit`, um die generierten Bilder zu erhalten

#### sd3:

1) Geben Sie Ihre Anfrage ein
2) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
3) Klicken Sie auf die Schaltfläche `Submit`, um das generierte Bild zu erhalten

#### cascade:

1) Geben Sie Ihre Anfrage ein
2) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
3) Klicken Sie auf die Schaltfläche `Submit`, um das generierte Bild zu erhalten

#### extras:

1) Laden Sie das Ausgangsbild hoch
2) Wählen Sie die gewünschten Optionen aus
3) Klicken Sie auf die Schaltfläche `Submit`, um das modifizierte Bild zu erhalten

### Kandinsky:

1) Geben Sie Ihren Prompt ein
2) Wählen Sie ein Modell aus der Dropdown-Liste
3) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
4) Klicken Sie auf `Submit`, um das generierte Bild zu erhalten

### Flux:

1) Geben Sie Ihren Prompt ein
2) Wählen Sie ein Modell aus der Dropdown-Liste
3) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
4) Klicken Sie auf `Submit`, um das generierte Bild zu erhalten

### HunyuanDiT:

1) Geben Sie Ihren Prompt ein
2) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
3) Klicken Sie auf `Submit`, um das generierte Bild zu erhalten

### Lumina-T2X:

1) Geben Sie Ihren Prompt ein
2) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
3) Klicken Sie auf `Submit`, um das generierte Bild zu erhalten

### Kolors:

1) Geben Sie Ihren Prompt ein
2) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
3) Klicken Sie auf `Submit`, um das generierte Bild zu erhalten

### AuraFlow:

1) Geben Sie Ihren Prompt ein
2) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
3) Klicken Sie auf `Submit`, um das generierte Bild zu erhalten

### Würstchen:

1) Geben Sie Ihren Prompt ein
2) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
3) Klicken Sie auf `Submit`, um das generierte Bild zu erhalten

### DeepFloydIF:

1) Geben Sie Ihren Prompt ein
2) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
3) Klicken Sie auf `Submit`, um das generierte Bild zu erhalten

### PixArt:

1) Geben Sie Ihren Prompt ein
2) Wählen Sie das Modell aus der Dropdown-Liste
3) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
4) Klicken Sie auf `Submit`, um das generierte Bild zu erhalten

### ModelScope:

1) Geben Sie Ihren Prompt ein
2) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
3) Klicken Sie auf `Submit`, um das generierte Video zu erhalten

### ZeroScope 2:

1) Geben Sie Ihren Prompt ein
2) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
3) Klicken Sie auf `Submit`, um das generierte Video zu erhalten

### CogVideoX:

1) Geben Sie Ihren Prompt ein
2) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
3) Klicken Sie auf `Submit`, um das generierte Video zu erhalten

### Latte:

1) Geben Sie Ihren Prompt ein
2) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
3) Klicken Sie auf `Submit`, um das generierte Video zu erhalten

### TripoSR:

1) Laden Sie das Ausgangsbild hoch
2) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
3) Klicken Sie auf die Schaltfläche `Submit`, um das generierte 3D-Objekt zu erhalten

### StableFast3D:

1) Laden Sie das Ausgangsbild hoch
2) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
3) Klicken Sie auf die Schaltfläche `Submit`, um das generierte 3D-Objekt zu erhalten

### Shap-E:

1) Geben Sie Ihre Anfrage ein oder laden Sie das Ausgangsbild hoch
2) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
3) Klicken Sie auf die Schaltfläche `Submit`, um das generierte 3D-Objekt zu erhalten

### SV34D:

1) Laden Sie das Ausgangsbild (für 3D) oder Video (für 4D) hoch
2) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
3) Klicken Sie auf die Schaltfläche `Submit`, um das generierte 3D-Video zu erhalten

### Zero123Plus:

1) Laden Sie das Ausgangsbild hoch
2) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
3) Klicken Sie auf die Schaltfläche `Submit`, um die generierte 3D-Rotation des Bildes zu erhalten

### StableAudio:

1) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
2) Geben Sie Ihre Anfrage ein
3) Klicken Sie auf die Schaltfläche `Submit`, um das generierte Audio zu erhalten

### AudioCraft:

1) Wählen Sie ein Modell aus der Dropdown-Liste
2) Wählen Sie den Modelltyp (`musicgen`, `audiogen` oder `magnet`)
3) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
4) Geben Sie Ihre Anfrage ein
5) (Optional) Laden Sie das Ausgangsaudio hoch, wenn Sie das `melody`-Modell verwenden
6) Klicken Sie auf die Schaltfläche `Submit`, um das generierte Audio zu erhalten
#### Optional: Sie können `multiband diffusion` aktivieren, um das generierte Audio zu verbessern

### AudioLDM 2:

1) Wählen Sie ein Modell aus der Dropdown-Liste
2) Richten Sie das Modell gemäß den von Ihnen benötigten Parametern ein
3) Geben Sie Ihre Anfrage ein
4) Klicken Sie auf die Schaltfläche `Submit`, um das generierte Audio zu erhalten

### Demucs:

1) Laden Sie das zu trennende Ausgangsaudio hoch
2) Klicken Sie auf die Schaltfläche `Submit`, um das getrennte Audio zu erhalten

### Galerie:

* Hier können Sie Dateien aus dem Ausgabeverzeichnis anzeigen

### ModelDownloader:

* Hier können Sie `LLM` und `StableDiffusion` Modelle herunterladen. Wählen Sie einfach das Modell aus der Dropdown-Liste und klicken Sie auf die Schaltfläche `Submit`
#### `LLM` Modelle werden hier heruntergeladen: *inputs/text/llm_models*
#### `StableDiffusion` Modelle werden hier heruntergeladen: *inputs/image/sd_models*

### Einstellungen:

* Hier können Sie die Anwendungseinstellungen ändern. Momentan können Sie nur den `Share`-Modus auf `True` oder `False` setzen

### System:

* Hier können Sie die Anzeigen der Sensoren Ihres Computers sehen, indem Sie auf die Schaltfläche `Submit` klicken

### Zusätzliche Informationen:

1) Alle Generierungen werden im Ordner *outputs* gespeichert
2) Sie können die Schaltfläche `Clear` drücken, um Ihre Auswahl zurückzusetzen
3) Um den Generierungsprozess zu stoppen, klicken Sie auf die Schaltfläche `Stop generation`
4) Sie können die Anwendung mit der Schaltfläche `Close terminal` ausschalten
5) Sie können den Ordner *outputs* öffnen, indem Sie auf die Schaltfläche `Outputs` klicken

## Wo kann ich Modelle und Stimmen bekommen?

* LLM Modelle können von [HuggingFace](https://huggingface.co/models) oder vom ModelDownloader innerhalb der Oberfläche bezogen werden
* StableDiffusion, vae, inpaint, embedding und lora Modelle können von [CivitAI](https://civitai.com/models) oder vom ModelDownloader innerhalb der Oberfläche bezogen werden
* StableAudioOpen, AudioCraft, AudioLDM 2, TTS, Whisper, Wav2Lip, SunoBark, MoonDream2, Upscale, GLIGEN, Depth, Pix2Pix, Controlnet, AnimateDiff, Videos, LDM3D, SD3, Cascade, Rembg, Roop, CodeFormer, Real-ESRGAN, TripoSR, StableFast3D, Shap-E, SV34D, Zero123Plus, Demucs, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF, PixArt, ModelScope, ZeroScope 2, CogVideoX, Latte und Multiband diffusion Modelle werden automatisch im Ordner *inputs* heruntergeladen, wenn sie verwendet werden
* Sie können Stimmen überall hernehmen. Nehmen Sie Ihre eigene auf oder verwenden Sie eine Aufnahme aus dem Internet. Oder verwenden Sie einfach diejenigen, die bereits im Projekt enthalten sind. Die Hauptsache ist, dass sie vorverarbeitet sind!

## Wiki

* https://github.com/Dartvauder/NeuroSandboxWebUI/wiki

## Dank an die Entwickler

#### Vielen Dank an diese Projekte, denn dank ihrer Anwendungen/Bibliotheken konnte ich meine Anwendung erstellen:

Zuallererst möchte ich den Entwicklern von [PyCharm](https://www.jetbrains.com/pycharm/) und [GitHub](https://desktop.github.com) danken. Mit Hilfe ihrer Anwendungen konnte ich meinen Code erstellen und teilen

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

## Lizenzen von Drittanbietern:

#### Viele Modelle haben ihre eigene Nutzungslizenz. Bevor Sie sie verwenden, empfehle ich Ihnen, sich mit diesen vertraut zu machen:

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

## Spende

### *Wenn Ihnen mein Projekt gefallen hat und Sie spenden möchten, hier sind Möglichkeiten zu spenden. Vielen Dank im Voraus!*

* CryptoWallet(BEP-20) - 0x3d86bdb5f50b92d0d7Eb44F1a833acC5e91aAEcA

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)
