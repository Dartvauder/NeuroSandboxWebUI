## [Fonctionnalités](/#Fonctionnalités) | [Dépendances](/#Dépendances-requises) | [Configuration système](/#Configuration-système-minimale) | [Installation](/#Comment-installer) | [Utilisation](/#Comment-utiliser) | [Modèles](/#Où-puis-je-obtenir-des-modèles-voix-et-avatars) | [Wiki](/#Wiki) | [Remerciements](/#Remerciements-aux-développeurs) | [Licences](/#Licences-tierces)

# ![main](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4ea0d891-8979-45ad-b052-626c41ae991a)
* Travail en cours ! (ALPHA)
* [English](/README.md)  | [عربي](/Readmes/README_AR.md) | [Deutsche](/Readmes/README_DE.md) | [Español](/Readmes/README_ES.md) | Français | [日本語](/Readmes/README_JP.md) | [Русский](/Readmes/README_RU.md) | [漢語](/Readmes/README_ZH.md) | [Português](/Readmes/README_PT.md) | [हिंदी](/Readmes/README_HI.md) | [Italiano](/Readmes/README_IT.md)

## Description :

Une interface simple et pratique pour utiliser divers modèles de réseaux neuronaux. Vous pouvez communiquer avec LLM et Moondream2 en utilisant du texte, de la voix et des images en entrée ; utiliser StableDiffusion, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF et PixArt pour générer des images ; ModelScope, ZeroScope 2, CogVideoX et Latte pour générer des vidéos ; TripoSR, StableFast3D, Shap-E, SV34D et Zero123Plus pour générer des objets 3D ; StableAudioOpen, AudioCraft et AudioLDM 2 pour générer de la musique et de l'audio ; CoquiTTS et SunoBark pour la synthèse vocale ; OpenAI-Whisper pour la reconnaissance vocale ; Wav2Lip pour la synchronisation labiale ; Roop pour l'échange de visages ; Rembg pour supprimer l'arrière-plan ; CodeFormer pour la restauration de visages ; LibreTranslate pour la traduction de texte ; Demucs pour la séparation de fichiers audio. Vous pouvez également visualiser les fichiers du répertoire outputs dans la galerie, télécharger les modèles LLM et StableDiffusion, modifier les paramètres de l'application dans l'interface et vérifier les capteurs système.

L'objectif du projet est de créer l'application la plus simple possible pour utiliser les modèles de réseaux neuronaux.

### Texte : <img width="1121" alt="1fr" src="https://github.com/user-attachments/assets/aee80929-9ec4-4ba5-8dad-fdca802feead">

### Image : <img width="1118" alt="2fr" src="https://github.com/user-attachments/assets/a9fe4d4d-c8a4-4dec-8a8b-e59cb3b1ff02">

### Vidéo : <img width="1116" alt="3fr" src="https://github.com/user-attachments/assets/00d846ec-e706-4c4b-b250-fbdb7e9a2df2">

### 3D : <img width="1120" alt="4fr" src="https://github.com/user-attachments/assets/41cf6d89-6dd7-4f7b-b6c2-b72d40a032de">

### Audio : <img width="1117" alt="5fr" src="https://github.com/user-attachments/assets/b331e58a-fe4a-4c30-b960-bbd7cc9aac72">

### Interface : <img width="1118" alt="6fr" src="https://github.com/user-attachments/assets/5bfe164e-d697-430f-8571-746203b37ee9">

## Fonctionnalités :

* Installation facile via install.bat (Windows) ou install.sh (Linux)
* Vous pouvez utiliser l'application via votre appareil mobile en localhost (via IPv4) ou n'importe où en ligne (via Share)
* Interface flexible et optimisée (par Gradio)
* Authentification via admin:admin (Vous pouvez saisir vos identifiants dans le fichier GradioAuth.txt)
* Vous pouvez ajouter votre propre HuggingFace-Token pour télécharger des modèles spécifiques (Vous pouvez saisir votre token dans le fichier HF-Token.txt)
* Prise en charge des modèles Transformers et llama.cpp (LLM)
* Prise en charge des modèles diffusers et safetensors (StableDiffusion) - onglets txt2img, img2img, depth2img, pix2pix, controlnet, upscale, inpaint, gligen, animatediff, video, ldm3d, sd3, cascade et extras
* Prise en charge de modèles supplémentaires pour la génération d'images : Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF et PixArt
* Prise en charge de StableAudioOpen
* Prise en charge d'AudioCraft (Modèles : musicgen, audiogen et magnet)
* Prise en charge d'AudioLDM 2 (Modèles : audio et musique)
* Prise en charge des modèles TTS et Whisper (pour LLM et TTS-STT)
* Prise en charge des modèles Lora, Textual inversion (embedding), Vae, Img2img, Depth, Pix2Pix, Controlnet, Upscale(latent), Upscale(Real-ESRGAN), Inpaint, GLIGEN, AnimateDiff, Videos, LDM3D, SD3, Cascade, Rembg, CodeFormer et Roop (pour StableDiffusion)
* Prise en charge du modèle Multiband Diffusion (pour AudioCraft)
* Prise en charge de LibreTranslate (API locale)
* Prise en charge de ModelScope, ZeroScope 2, CogVideoX et Latte pour la génération de vidéos
* Prise en charge de SunoBark
* Prise en charge de Demucs
* Prise en charge de TripoSR, StableFast3D, Shap-E, SV34D et Zero123Plus pour la génération 3D
* Prise en charge de Wav2Lip
* Prise en charge du multimodal (Moondream 2), LORA (transformers) et WebSearch (avec GoogleSearch) pour LLM
* Paramètres des modèles dans l'interface
* Galerie
* ModelDownloader (pour LLM et StableDiffusion)
* Paramètres de l'application
* Possibilité de voir les capteurs système

## Dépendances requises :

* [Python](https://www.python.org/downloads/) (3.11)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (12.4) et [cuDNN](https://developer.nvidia.com/cudnn-downloads) (9.1)
* [FFMPEG](https://ffmpeg.org/download.html)
- Compilateur C++
  - Windows : [VisualStudio](https://visualstudio.microsoft.com/fr/)
  - Linux : [GCC](https://gcc.gnu.org/)

## Configuration système minimale :

* Système : Windows ou Linux
* GPU : 6 Go+ ou CPU : 8 cœurs 3,2 GHz
* RAM : 16 Go+
* Espace disque : 20 Go+
* Internet pour télécharger les modèles et installer

## Comment installer :

### Windows

1) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` à l'emplacement de votre choix
2) Exécutez `Install.bat` et attendez la fin de l'installation
3) Après l'installation, exécutez `Start.bat`
4) Sélectionnez la version du fichier et attendez le lancement de l'application
5) Vous pouvez maintenant commencer à générer !

Pour obtenir une mise à jour, exécutez `Update.bat`
Pour travailler avec l'environnement virtuel via le terminal, exécutez `Venv.bat`

### Linux

1) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` à l'emplacement de votre choix
2) Dans le terminal, exécutez `./Install.sh` et attendez l'installation de toutes les dépendances
3) Après l'installation, exécutez `./Start.sh`
4) Attendez le lancement de l'application
5) Vous pouvez maintenant commencer à générer !

Pour obtenir une mise à jour, exécutez `./Update.sh`
Pour travailler avec l'environnement virtuel via le terminal, exécutez `./Venv.sh`

## Comment utiliser :

#### L'interface comporte trente-deux onglets dans six onglets principaux : LLM, TTS-STT, SunoBark, LibreTranslate, Wav2Lip, StableDiffusion, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF, PixArt, ModelScope, ZeroScope 2, CogVideoX, Latte, TripoSR, StableFast3D, Shap-E, SV34D, Zero123Plus, StableAudio, AudioCraft, AudioLDM 2, Demucs, Gallery, ModelDownloader, Settings et System. Sélectionnez celui dont vous avez besoin et suivez les instructions ci-dessous 

### LLM :

1) Téléchargez d'abord vos modèles dans le dossier : *inputs/text/llm_models*
2) Sélectionnez votre modèle dans la liste déroulante
3) Sélectionnez le type de modèle (`transformers` ou `llama`)
4) Configurez le modèle selon les paramètres dont vous avez besoin
5) Tapez (ou parlez) votre requête
6) Cliquez sur le bouton `Submit` pour recevoir le texte généré et la réponse audio
#### Optionnel : vous pouvez activer le mode `TTS`, sélectionner la `voice` et la `language` nécessaires pour recevoir une réponse audio. Vous pouvez activer `multimodal` et télécharger une image pour obtenir sa description. Vous pouvez activer `websearch` pour l'accès à Internet. Vous pouvez activer `libretranslate` pour obtenir la traduction. Vous pouvez également choisir le modèle `LORA` pour améliorer la génération
#### Échantillons de voix = *inputs/audio/voices*
#### LORA = *inputs/text/llm_models/lora*
#### La voix doit être prétraitée (22050 kHz, mono, WAV)

### TTS-STT :

1) Tapez le texte pour la synthèse vocale
2) Entrez l'audio pour la reconnaissance vocale
3) Cliquez sur le bouton `Submit` pour recevoir le texte généré et la réponse audio
#### Échantillons de voix = *inputs/audio/voices*
#### La voix doit être prétraitée (22050 kHz, mono, WAV)

### SunoBark :

1) Tapez votre requête
2) Configurez le modèle selon les paramètres dont vous avez besoin
3) Cliquez sur le bouton `Submit` pour recevoir la réponse audio générée

### LibreTranslate :

* Vous devez d'abord installer et exécuter [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate)
1) Sélectionnez les langues source et cible
2) Cliquez sur le bouton `Submit` pour obtenir la traduction
#### Optionnel : vous pouvez enregistrer l'historique des traductions en activant le bouton correspondant

### Wav2Lip :

1) Téléchargez l'image initiale du visage
2) Téléchargez l'audio initial de la voix
3) Configurez le modèle selon les paramètres dont vous avez besoin
4) Cliquez sur le bouton `Submit` pour recevoir la synchronisation labiale

### StableDiffusion - a quinze sous-onglets :

#### txt2img :

1) Téléchargez d'abord vos modèles dans le dossier : *inputs/image/sd_models*
2) Sélectionnez votre modèle dans la liste déroulante
3) Sélectionnez le type de modèle (`SD`, `SD2` ou `SDXL`)
4) Configurez le modèle selon les paramètres dont vous avez besoin
5) Entrez votre requête (+ et - pour la pondération des prompts)
6) Cliquez sur le bouton `Submit` pour obtenir l'image générée
#### Optionnel : Vous pouvez sélectionner vos modèles `vae`, `embedding` et `lora` pour améliorer la méthode de génération, vous pouvez également activer `upscale` pour augmenter la taille de l'image générée 
#### vae = *inputs/image/sd_models/vae*
#### lora = *inputs/image/sd_models/lora*
#### embedding = *inputs/image/sd_models/embedding*

#### img2img :

1) Téléchargez d'abord vos modèles dans le dossier : *inputs/image/sd_models*
2) Sélectionnez votre modèle dans la liste déroulante
3) Sélectionnez le type de modèle (`SD`, `SD2` ou `SDXL`)
4) Configurez le modèle selon les paramètres dont vous avez besoin
5) Téléchargez l'image initiale avec laquelle la génération aura lieu
6) Entrez votre requête (+ et - pour la pondération des prompts)
7) Cliquez sur le bouton `Submit` pour obtenir l'image générée
#### Optionnel : Vous pouvez sélectionner votre modèle `vae`
#### vae = *inputs/image/sd_models/vae*

#### depth2img :

1) Téléchargez l'image initiale
2) Configurez le modèle selon les paramètres dont vous avez besoin
3) Entrez votre requête (+ et - pour la pondération des prompts)
4) Cliquez sur le bouton `Submit` pour obtenir l'image générée

#### pix2pix :

1) Téléchargez l'image initiale
2) Configurez le modèle selon les paramètres dont vous avez besoin
3) Entrez votre requête (+ et - pour la pondération des prompts)
4) Cliquez sur le bouton `Submit` pour obtenir l'image générée

#### controlnet :

1) Téléchargez d'abord vos modèles stable diffusion dans le dossier : *inputs/image/sd_models*
2) Téléchargez l'image initiale
3) Sélectionnez vos modèles stable diffusion et controlnet dans les listes déroulantes
4) Configurez les modèles selon les paramètres dont vous avez besoin
5) Entrez votre requête (+ et - pour la pondération des prompts)
6) Cliquez sur le bouton `Submit` pour obtenir l'image générée

#### upscale(latent) :

1) Téléchargez l'image initiale
2) Configurez le modèle selon les paramètres dont vous avez besoin
3) Cliquez sur le bouton `Submit` pour obtenir l'image agrandie

#### upscale(Real-ESRGAN) :

1) Téléchargez l'image initiale
2) Configurez le modèle selon les paramètres dont vous avez besoin
3) Cliquez sur le bouton `Submit` pour obtenir l'image agrandie

#### inpaint :

1) Téléchargez d'abord vos modèles dans le dossier : *inputs/image/sd_models/inpaint*
2) Sélectionnez votre modèle dans la liste déroulante
3) Sélectionnez le type de modèle (`SD`, `SD2` ou `SDXL`)
4) Configurez le modèle selon les paramètres dont vous avez besoin
5) Téléchargez l'image avec laquelle la génération aura lieu dans `initial image` et `mask image`
6) Dans `mask image`, sélectionnez le pinceau, puis la palette et changez la couleur en `#FFFFFF`
7) Dessinez un endroit pour la génération et entrez votre requête (+ et - pour la pondération des prompts)
8) Cliquez sur le bouton `Submit` pour obtenir l'image inpaintée
#### Optionnel : Vous pouvez sélectionner votre modèle `vae`
#### vae = *inputs/image/sd_models/vae*

#### gligen :

1) Téléchargez d'abord vos modèles dans le dossier : *inputs/image/sd_models*
2) Sélectionnez votre modèle dans la liste déroulante
3) Sélectionnez le type de modèle (`SD`, `SD2` ou `SDXL`)
4) Configurez le modèle selon les paramètres dont vous avez besoin
5) Entrez votre requête pour le prompt (+ et - pour la pondération des prompts) et les phrases GLIGEN (entre "" pour les boîtes)
6) Entrez les boîtes GLIGEN (Comme [0.1387, 0.2051, 0.4277, 0.7090] pour une boîte)
7) Cliquez sur le bouton `Submit` pour obtenir l'image générée

#### animatediff :

1) Téléchargez d'abord vos modèles dans le dossier : *inputs/image/sd_models*
2) Sélectionnez votre modèle dans la liste déroulante
3) Configurez le modèle selon les paramètres dont vous avez besoin
4) Entrez votre requête (+ et - pour la pondération des prompts)
5) Cliquez sur le bouton `Submit` pour obtenir l'animation d'image générée

#### video :

1) Téléchargez l'image initiale
2) Entrez votre requête (pour IV2Gen-XL)
3) Configurez le modèle selon les paramètres dont vous avez besoin
4) Cliquez sur le bouton `Submit` pour obtenir la vidéo à partir de l'image

#### ldm3d :

1) Entrez votre requête
2) Configurez le modèle selon les paramètres dont vous avez besoin
3) Cliquez sur le bouton `Submit` pour obtenir les images générées

#### sd3 :

1) Entrez votre requête
2) Configurez le modèle selon les paramètres dont vous avez besoin
3) Cliquez sur le bouton `Submit` pour obtenir l'image générée

#### cascade :

1) Entrez votre requête
2) Configurez le modèle selon les paramètres dont vous avez besoin
3) Cliquez sur le bouton `Submit` pour obtenir l'image générée

#### extras :

1) Téléchargez l'image initiale
2) Sélectionnez les options dont vous avez besoin
3) Cliquez sur le bouton `Submit` pour obtenir l'image modifiée

### Kandinsky :

1) Entrez votre prompt
2) Sélectionnez un modèle dans la liste déroulante
3) Configurez le modèle selon les paramètres dont vous avez besoin
4) Cliquez sur `Submit` pour obtenir l'image générée

### Flux :

1) Entrez votre prompt
2) Sélectionnez un modèle dans la liste déroulante
3) Configurez le modèle selon les paramètres dont vous avez besoin
4) Cliquez sur `Submit` pour obtenir l'image générée

### HunyuanDiT :

1) Entrez votre prompt
2) Configurez le modèle selon les paramètres dont vous avez besoin
3) Cliquez sur `Submit` pour obtenir l'image générée

### Lumina-T2X :

1) Entrez votre prompt
2) Configurez le modèle selon les paramètres dont vous avez besoin
3) Cliquez sur `Submit` pour obtenir l'image générée

### Kolors :

1) Entrez votre prompt
2) Configurez le modèle selon les paramètres dont vous avez besoin
3) Cliquez sur `Submit` pour obtenir l'image générée

### AuraFlow :

1) Entrez votre prompt
2) Configurez le modèle selon les paramètres dont vous avez besoin
3) Cliquez sur `Submit` pour obtenir l'image générée

### Würstchen :

1) Entrez votre prompt
2) Configurez le modèle selon les paramètres dont vous avez besoin
3) Cliquez sur `Submit` pour obtenir l'image générée

### DeepFloydIF :

1) Entrez votre prompt
2) Configurez le modèle selon les paramètres dont vous avez besoin
3) Cliquez sur `Submit` pour obtenir l'image générée

### PixArt :

1) Entrez votre prompt
2) Sélectionnez le modèle dans la liste déroulante
3) Configurez le modèle selon les paramètres dont vous avez besoin
4) Cliquez sur `Submit` pour obtenir l'image générée

### ModelScope :

1) Entrez votre prompt
2) Configurez le modèle selon les paramètres dont vous avez besoin
3) Cliquez sur `Submit` pour obtenir la vidéo générée

### ZeroScope 2 :

1) Entrez votre prompt
2) Configurez le modèle selon les paramètres dont vous avez besoin
3) Cliquez sur `Submit` pour obtenir la vidéo générée

### CogVideoX :

1) Entrez votre prompt
2) Configurez le modèle selon les paramètres dont vous avez besoin
3) Cliquez sur `Submit` pour obtenir la vidéo générée

### Latte :

1) Entrez votre prompt
2) Configurez le modèle selon les paramètres dont vous avez besoin
3) Cliquez sur `Submit` pour obtenir la vidéo générée

### TripoSR :

1) Téléchargez l'image initiale
2) Configurez le modèle selon les paramètres dont vous avez besoin
3) Cliquez sur le bouton `Submit` pour obtenir l'objet 3D généré

### StableFast3D :

1) Téléchargez l'image initiale
2) Configurez le modèle selon les paramètres dont vous avez besoin
3) Cliquez sur le bouton `Submit` pour obtenir l'objet 3D généré

### Shap-E :

1) Entrez votre requête ou téléchargez l'image initiale
2) Configurez le modèle selon les paramètres dont vous avez besoin
3) Cliquez sur le bouton `Submit` pour obtenir l'objet 3D généré

### SV34D :

1) Téléchargez l'image initiale (pour 3D) ou la vidéo (pour 4D)
2) Configurez le modèle selon les paramètres dont vous avez besoin
3) Cliquez sur le bouton `Submit` pour obtenir la vidéo 3D générée

### Zero123Plus :

1) Téléchargez l'image initiale
2) Configurez le modèle selon les paramètres dont vous avez besoin
3) Cliquez sur le bouton `Submit` pour obtenir la rotation 3D générée de l'image

### StableAudio :

1) Configurez le modèle selon les paramètres dont vous avez besoin
2) Entrez votre requête
3) Cliquez sur le bouton `Submit` pour obtenir l'audio généré

### AudioCraft :

1) Sélectionnez un modèle dans la liste déroulante
2) Sélectionnez le type de modèle (`musicgen`, `audiogen` ou `magnet`)
3) Configurez le modèle selon les paramètres dont vous avez besoin
4) Entrez votre requête
5) (Optionnel) téléchargez l'audio initial si vous utilisez le modèle `melody` 
6) Cliquez sur le bouton `Submit` pour obtenir l'audio généré
#### Optionnel : Vous pouvez activer `multiband diffusion` pour améliorer l'audio généré

### AudioLDM 2 :

1) Sélectionnez un modèle dans la liste déroulante
2) Configurez le modèle selon les paramètres dont vous avez besoin
3) Entrez votre requête
4) Cliquez sur le bouton `Submit` pour obtenir l'audio généré

### Demucs :

1) Téléchargez l'audio initial à séparer
2) Cliquez sur le bouton `Submit` pour obtenir l'audio séparé

### Gallery :

* Ici, vous pouvez visualiser les fichiers du répertoire outputs

### ModelDownloader :

* Ici, vous pouvez télécharger des modèles `LLM` et `StableDiffusion`. Choisissez simplement le modèle dans la liste déroulante et cliquez sur le bouton `Submit`
#### Les modèles `LLM` sont téléchargés ici : *inputs/text/llm_models*
#### Les modèles `StableDiffusion` sont téléchargés ici : *inputs/image/sd_models*

### Settings : 

* Ici, vous pouvez modifier les paramètres de l'application. Pour l'instant, vous ne pouvez que changer le mode `Share` en `True` ou `False`

### System : 

* Ici, vous pouvez voir les indicateurs des capteurs de votre ordinateur en cliquant sur le bouton `Submit`

### Informations supplémentaires :

1) Toutes les générations sont sauvegardées dans le dossier *outputs*
2) Vous pouvez appuyer sur le bouton `Clear` pour réinitialiser votre sélection
3) Pour arrêter le processus de génération, cliquez sur le bouton `Stop generation`
4) Vous pouvez éteindre l'application en utilisant le bouton `Close terminal`
5) Vous pouvez ouvrir le dossier *outputs* en cliquant sur le bouton `Outputs`

## Où puis-je obtenir des modèles et des voix ?

* Les modèles LLM peuvent être obtenus sur [HuggingFace](https://huggingface.co/models) ou à partir du ModelDownloader dans l'interface 
* Les modèles StableDiffusion, vae, inpaint, embedding et lora peuvent être obtenus sur [CivitAI](https://civitai.com/models) ou à partir du ModelDownloader dans l'interface
* Les modèles StableAudioOpen, AudioCraft, AudioLDM 2, TTS, Whisper, Wav2Lip, SunoBark, MoonDream2, Upscale, GLIGEN, Depth, Pix2Pix, Controlnet, AnimateDiff, Videos, LDM3D, SD3, Cascade, Rembg, Roop, CodeFormer, Real-ESRGAN, TripoSR, StableFast3D, Shap-E, SV34D, Zero123Plus, Demucs, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF, PixArt, ModelScope, ZeroScope 2, CogVideoX, Latte et Multiband diffusion sont téléchargés automatiquement dans le dossier *inputs* lorsqu'ils sont utilisés 
* Vous pouvez prendre des voix n'importe où. Enregistrez les vôtres ou prenez un enregistrement sur Internet. Ou utilisez simplement celles qui sont déjà dans le projet. L'essentiel est qu'elles soient prétraitées !

## Wiki

* https://github.com/Dartvauder/NeuroSandboxWebUI/wiki

## Remerciements aux développeurs

#### Un grand merci à ces projets car grâce à leurs applications/bibliothèques, j'ai pu créer mon application :

Tout d'abord, je tiens à remercier les développeurs de [PyCharm](https://www.jetbrains.com/pycharm/) et [GitHub](https://desktop.github.com). Grâce à leurs applications, j'ai pu créer et partager mon code

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

## Licences tierces :

#### De nombreux modèles ont leur propre licence d'utilisation. Avant de les utiliser, je vous conseille de vous familiariser avec celles-ci :

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

## Don

### *Si vous avez aimé mon projet et souhaitez faire un don, voici les options pour le faire. Merci beaucoup d'avance !*

* Portefeuille crypto (BEP-20) - 0x3d86bdb5f50b92d0d7Eb44F1a833acC5e91aAEcA

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)
