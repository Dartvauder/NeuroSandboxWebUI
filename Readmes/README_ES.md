## [Características](/#Características) | [Dependencias](/#Dependencias-Requeridas) | [RequisitosDelSistema](/#Requisitos-Mínimos-del-Sistema) | [Instalación](/#Cómo-instalar) | [Uso](/#Cómo-usar) | [Modelos](/#Dónde-puedo-obtener-modelos-y-voces) | [Wiki](/#Wiki) | [Agradecimiento](/#Agradecimiento-a-los-desarrolladores) | [Licencias](/#Licencias-de-Terceros)

# ![main](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4ea0d891-8979-45ad-b052-626c41ae991a)
* ¡Trabajo en progreso! (ALFA)
* [English](/README.md)  | [عربي](/Readmes/README_AR.md) | [Deutsche](/Readmes/README_DE.md) | Español | [Français](/Readmes/README_FR.md) | [日本語](/Readmes/README_JP.md) | [Русский](/Readmes/README_RU.md) | [漢語](/Readmes/README_ZH.md) | [Português](/Readmes/README_PT.md) | [हिंदी](/Readmes/README_HI.md) | [Italiano](/Readmes/README_IT.md) | [韓國語](/Readmes/README_KO.md) | [Polski](/Readmes/README_PL.md) | [Türkçe](/Readmes/README_TR.md)

## Descripción:

Una interfaz simple y conveniente para usar varios modelos de redes neuronales. Puedes comunicarte con LLM y Moondream2 usando entrada de texto, voz e imagen; usar StableDiffusion, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF y PixArt para generar imágenes; ModelScope, ZeroScope 2, CogVideoX y Latte para generar videos; TripoSR, StableFast3D, Shap-E, SV34D y Zero123Plus para generar objetos 3D; StableAudioOpen, AudioCraft y AudioLDM 2 para generar música y audio; CoquiTTS y SunoBark para texto a voz; OpenAI-Whisper para voz a texto; Wav2Lip para sincronización labial; Roop para intercambio de caras; Rembg para eliminar el fondo; CodeFormer para restauración facial; LibreTranslate para traducción de texto; Demucs para separación de archivos de audio. También puedes ver archivos del directorio de salidas en la galería, descargar los modelos LLM y StableDiffusion, cambiar la configuración de la aplicación dentro de la interfaz y verificar los sensores del sistema.

El objetivo del proyecto es crear la aplicación más fácil posible para usar modelos de redes neuronales.

### Texto: <img width="1121" alt="1es" src="https://github.com/user-attachments/assets/c0cd46d8-7f7c-48f1-ae8d-3f7df32db2c8">

### Imagen: <img width="1122" alt="2es" src="https://github.com/user-attachments/assets/3aa54076-390e-429d-ab8f-b44b7b2e1006">

### Video: <img width="1120" alt="3es" src="https://github.com/user-attachments/assets/0bcdcc45-66a0-4f14-8772-92e7e549a65e">

### 3D: <img width="1120" alt="4es" src="https://github.com/user-attachments/assets/ba3457f1-df60-4a84-9047-f78dfe7ec8a8">

### Audio: <img width="1117" alt="5es" src="https://github.com/user-attachments/assets/4fe46de5-521a-4109-ac51-6fb56e8a958d">

### Interfaz: <img width="1118" alt="6es" src="https://github.com/user-attachments/assets/83b292a3-87ef-421c-b574-e47b95dcd26f">

## Características:

* Fácil instalación a través de install.bat (Windows) o install.sh (Linux)
* Puedes usar la aplicación a través de tu dispositivo móvil en localhost (a través de IPv4) o en cualquier lugar en línea (a través de Share)
* Interfaz flexible y optimizada (por Gradio)
* Autenticación a través de admin:admin (Puedes ingresar tus detalles de inicio de sesión en el archivo GradioAuth.txt)
* Puedes agregar tu propio HuggingFace-Token para descargar modelos específicos (Puedes ingresar tu token en el archivo HF-Token.txt)
* Soporte para modelos Transformers y llama.cpp (LLM)
* Soporte para modelos diffusers y safetensors (StableDiffusion) - pestañas txt2img, img2img, depth2img, pix2pix, controlnet, upscale, inpaint, gligen, animatediff, video, ldm3d, sd3, cascade, adapters y extras
* Soporte de modelos adicionales para generación de imágenes: Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF y PixArt
* Soporte StableAudioOpen
* Soporte AudioCraft (Modelos: musicgen, audiogen y magnet)
* Soporte AudioLDM 2 (Modelos: audio y música)
* Soporta modelos TTS y Whisper (Para LLM y TTS-STT)
* Soporta modelos Lora, Textual inversion (embedding), Vae, Img2img, Depth, Pix2Pix, Controlnet, Upscale(latent), Upscale(Real-ESRGAN), Inpaint, GLIGEN, AnimateDiff, Videos, LDM3D, SD3, Cascade, Adapters (InstantID, PhotoMaker, IP-Adapter-FaceID), Rembg, CodeFormer y Roop (Para StableDiffusion)
* Soporte para el modelo Multiband Diffusion (Para AudioCraft)
* Soporte LibreTranslate (API Local)
* Soporte ModelScope, ZeroScope 2, CogVideoX y Latte para generación de video
* Soporte SunoBark
* Soporte Demucs
* Soporte TripoSR, StableFast3D, Shap-E, SV34D y Zero123Plus para generación 3D
* Soporte Wav2Lip
* Soporte Multimodal (Moondream 2), LORA (transformers) y WebSearch (con GoogleSearch) para LLM
* Configuración del modelo dentro de la interfaz
* Galería
* ModelDownloader (Para LLM y StableDiffusion)
* Configuración de la aplicación
* Capacidad de ver sensores del sistema

## Dependencias Requeridas:

* [Python](https://www.python.org/downloads/) (3.11)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (12.4) y [cuDNN](https://developer.nvidia.com/cudnn-downloads) (9.1)
* [FFMPEG](https://ffmpeg.org/download.html)
- Compilador C+
  - Windows: [VisualStudio](https://visualstudio.microsoft.com/ru/)
  - Linux: [GCC](https://gcc.gnu.org/)

## Requisitos Mínimos del Sistema:

* Sistema: Windows o Linux
* GPU: 6GB+ o CPU: 8 núcleos 3.2GHZ
* RAM: 16GB+
* Espacio en disco: 20GB+
* Internet para descargar modelos e instalar

## Cómo instalar:

### Windows

1) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` en cualquier ubicación
2) Ejecuta `Install.bat` y espera a que se complete la instalación
3) Después de la instalación, ejecuta `Start.bat`
4) Selecciona la versión del archivo y espera a que se inicie la aplicación
5) ¡Ahora puedes empezar a generar!

Para obtener actualizaciones, ejecuta `Update.bat`
Para trabajar con el entorno virtual a través de la terminal, ejecuta `Venv.bat`

### Linux

1) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` en cualquier ubicación
2) En la terminal, ejecuta `./Install.sh` y espera a que se instalen todas las dependencias
3) Después de la instalación, ejecuta `./Start.sh`
4) Espera a que se inicie la aplicación
5) ¡Ahora puedes empezar a generar!

Para obtener actualizaciones, ejecuta `./Update.sh`
Para trabajar con el entorno virtual a través de la terminal, ejecuta `./Venv.sh`

## Cómo usar:

#### La interfaz tiene treinta y dos pestañas en seis pestañas principales (Texto, Imagen, Video, 3D, Audio y Interfaz): LLM, TTS-STT, SunoBark, LibreTranslate, Wav2Lip, StableDiffusion, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF, PixArt, ModelScope, ZeroScope 2, CogVideoX, Latte, TripoSR, StableFast3D, Shap-E, SV34D, Zero123Plus, StableAudio, AudioCraft, AudioLDM 2, Demucs, Galería, ModelDownloader, Configuración y Sistema. Selecciona la que necesites y sigue las instrucciones a continuación

### LLM:

1) Primero sube tus modelos a la carpeta: *inputs/text/llm_models*
2) Selecciona tu modelo de la lista desplegable
3) Selecciona el tipo de modelo (`transformers` o `llama`)
4) Configura el modelo según los parámetros que necesites
5) Escribe (o habla) tu solicitud
6) Haz clic en el botón `Submit` para recibir la respuesta generada en texto y audio
#### Opcional: puedes habilitar el modo `TTS`, seleccionar la `voz` y el `idioma` necesarios para recibir una respuesta de audio. Puedes habilitar `multimodal` y subir una imagen para obtener su descripción. Puedes habilitar `websearch` para acceso a Internet. Puedes habilitar `libretranslate` para obtener la traducción. También puedes elegir el modelo `LORA` para mejorar la generación
#### Muestras de voz = *inputs/audio/voices*
#### LORA = *inputs/text/llm_models/lora*
#### La voz debe ser preprocesada (22050 kHz, mono, WAV)

### TTS-STT:

1) Escribe texto para texto a voz
2) Ingresa audio para voz a texto
3) Haz clic en el botón `Submit` para recibir la respuesta generada en texto y audio
#### Muestras de voz = *inputs/audio/voices*
#### La voz debe ser preprocesada (22050 kHz, mono, WAV)

### SunoBark:

1) Escribe tu solicitud
2) Configura el modelo según los parámetros que necesites
3) Haz clic en el botón `Submit` para recibir la respuesta de audio generada

### LibreTranslate:

* Primero necesitas instalar y ejecutar [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate)
1) Selecciona los idiomas de origen y destino
2) Haz clic en el botón `Submit` para obtener la traducción
#### Opcional: puedes guardar el historial de traducción activando el botón correspondiente

### Wav2Lip:

1) Sube la imagen inicial del rostro
2) Sube el audio inicial de la voz
3) Configura el modelo según los parámetros que necesites
4) Haz clic en el botón `Submit` para recibir la sincronización labial

### StableDiffusion - tiene dieciséis sub-pestañas:

#### txt2img:

1) Primero sube tus modelos a la carpeta: *inputs/image/sd_models*
2) Selecciona tu modelo de la lista desplegable
3) Selecciona el tipo de modelo (`SD`, `SD2` o `SDXL`)
4) Configura el modelo según los parámetros que necesites
5) Ingresa tu solicitud (+ y - para la ponderación del prompt)
6) Haz clic en el botón `Submit` para obtener la imagen generada
#### Opcional: Puedes seleccionar tus modelos `vae`, `embedding` y `lora` para mejorar el método de generación, también puedes habilitar `upscale` para aumentar el tamaño de la imagen generada
#### vae = *inputs/image/sd_models/vae*
#### lora = *inputs/image/sd_models/lora*
#### embedding = *inputs/image/sd_models/embedding*

#### img2img:

1) Primero sube tus modelos a la carpeta: *inputs/image/sd_models*
2) Selecciona tu modelo de la lista desplegable
3) Selecciona el tipo de modelo (`SD`, `SD2` o `SDXL`)
4) Configura el modelo según los parámetros que necesites
5) Sube la imagen inicial con la que se realizará la generación
6) Ingresa tu solicitud (+ y - para la ponderación del prompt)
7) Haz clic en el botón `Submit` para obtener la imagen generada
#### Opcional: Puedes seleccionar tu modelo `vae`
#### vae = *inputs/image/sd_models/vae*

#### depth2img:

1) Sube la imagen inicial
2) Configura el modelo según los parámetros que necesites
3) Ingresa tu solicitud (+ y - para la ponderación del prompt)
4) Haz clic en el botón `Submit` para obtener la imagen generada

#### pix2pix:

1) Sube la imagen inicial
2) Configura el modelo según los parámetros que necesites
3) Ingresa tu solicitud (+ y - para la ponderación del prompt)
4) Haz clic en el botón `Submit` para obtener la imagen generada

#### controlnet:

1) Primero sube tus modelos de stable diffusion a la carpeta: *inputs/image/sd_models*
2) Sube la imagen inicial
3) Selecciona tus modelos de stable diffusion y controlnet de las listas desplegables
4) Configura los modelos según los parámetros que necesites
5) Ingresa tu solicitud (+ y - para la ponderación del prompt)
6) Haz clic en el botón `Submit` para obtener la imagen generada

#### upscale(latent):

1) Sube la imagen inicial
2) Configura el modelo según los parámetros que necesites
3) Haz clic en el botón `Submit` para obtener la imagen ampliada

#### upscale(Real-ESRGAN):

1) Sube la imagen inicial
2) Configura el modelo según los parámetros que necesites
3) Haz clic en el botón `Submit` para obtener la imagen ampliada

#### inpaint:

1) Primero sube tus modelos a la carpeta: *inputs/image/sd_models/inpaint*
2) Selecciona tu modelo de la lista desplegable
3) Selecciona el tipo de modelo (`SD`, `SD2` o `SDXL`)
4) Configura el modelo según los parámetros que necesites
5) Sube la imagen con la que se realizará la generación a `initial image` y `mask image`
6) En `mask image`, selecciona el pincel, luego la paleta y cambia el color a `#FFFFFF`
7) Dibuja un lugar para la generación e ingresa tu solicitud (+ y - para la ponderación del prompt)
8) Haz clic en el botón `Submit` para obtener la imagen con inpainting
#### Opcional: Puedes seleccionar tu modelo `vae`
#### vae = *inputs/image/sd_models/vae*

#### gligen:

1) Primero sube tus modelos a la carpeta: *inputs/image/sd_models*
2) Selecciona tu modelo de la lista desplegable
3) Selecciona el tipo de modelo (`SD`, `SD2` o `SDXL`)
4) Configura el modelo según los parámetros que necesites
5) Ingresa tu solicitud para el prompt (+ y - para la ponderación del prompt) y frases GLIGEN (entre "" para el cuadro)
6) Ingresa los cuadros GLIGEN (Como [0.1387, 0.2051, 0.4277, 0.7090] para el cuadro)
7) Haz clic en el botón `Submit` para obtener la imagen generada

#### animatediff:

1) Primero sube tus modelos a la carpeta: *inputs/image/sd_models*
2) Selecciona tu modelo de la lista desplegable
3) Configura el modelo según los parámetros que necesites
4) Ingresa tu solicitud (+ y - para la ponderación del prompt)
5) Haz clic en el botón `Submit` para obtener la animación de imagen generada

#### video:

1) Sube la imagen inicial
2) Ingresa tu solicitud (para IV2Gen-XL)
3) Configura el modelo según los parámetros que necesites
4) Haz clic en el botón `Submit` para obtener el video a partir de la imagen

#### ldm3d:

1) Ingresa tu solicitud
2) Configura el modelo según los parámetros que necesites
3) Haz clic en el botón `Submit` para obtener las imágenes generadas

#### sd3 (txt2img, img2img, controlnet, inpaint):

1) Ingresa tu solicitud
2) Configura el modelo según los parámetros que necesites
3) Haz clic en el botón `Submit` para obtener la imagen generada

#### cascade:

1) Ingresa tu solicitud
2) Configura el modelo según los parámetros que necesites
3) Haz clic en el botón `Submit` para obtener la imagen generada

#### adapters (InstantID, PhotoMaker e IP-Adapter-FaceID):

1) Primero suba sus modelos a la carpeta: *inputs/image/sd_models*
2) Suba la imagen inicial
3) Seleccione su modelo de la lista desplegable
4) Configure el modelo según los parámetros que necesite
5) Seleccione la subpestaña que necesite
6) Haga clic en el botón `Submit` para obtener la imagen modificada

#### extras:

1) Sube la imagen inicial
2) Selecciona las opciones que necesites
3) Haz clic en el botón `Submit` para obtener la imagen modificada

### Kandinsky (txt2img, img2img, inpaint):

1) Ingresa tu prompt
2) Selecciona un modelo de la lista desplegable
3) Configura el modelo según los parámetros que necesites
4) Haz clic en `Submit` para obtener la imagen generada

### Flux:

1) Ingresa tu prompt
2) Selecciona un modelo de la lista desplegable
3) Configura el modelo según los parámetros que necesites
4) Haz clic en `Submit` para obtener la imagen generada

### HunyuanDiT:

1) Ingresa tu prompt
2) Configura el modelo según los parámetros que necesites
3) Haz clic en `Submit` para obtener la imagen generada

### Lumina-T2X:

1) Ingresa tu prompt
2) Configura el modelo según los parámetros que necesites
3) Haz clic en `Submit` para obtener la imagen generada

### Kolors:

1) Ingresa tu prompt
2) Configura el modelo según los parámetros que necesites
3) Haz clic en `Submit` para obtener la imagen generada

### AuraFlow:

1) Ingresa tu prompt
2) Configura el modelo según los parámetros que necesites
3) Haz clic en `Submit` para obtener la imagen generada

### Würstchen:

1) Ingresa tu prompt
2) Configura el modelo según los parámetros que necesites
3) Haz clic en `Submit` para obtener la imagen generada

### DeepFloydIF (txt2img, img2img, inpaint):

1) Ingresa tu prompt
2) Configura el modelo según los parámetros que necesites
3) Haz clic en `Submit` para obtener la imagen generada

### PixArt:

1) Ingresa tu prompt
2) Selecciona el modelo de la lista desplegable
3) Configura el modelo según los parámetros que necesites
4) Haz clic en `Submit` para obtener la imagen generada

### ModelScope:

1) Ingresa tu prompt
2) Configura el modelo según los parámetros que necesites
3) Haz clic en `Submit` para obtener el video generado

### ZeroScope 2:

1) Ingresa tu prompt
2) Configura el modelo según los parámetros que necesites
3) Haz clic en `Submit` para obtener el video generado

### CogVideoX:

1) Ingresa tu prompt
2) Configura el modelo según los parámetros que necesites
3) Haz clic en `Submit` para obtener el video generado

### Latte:

1) Ingresa tu prompt
2) Configura el modelo según los parámetros que necesites
3) Haz clic en `Submit` para obtener el video generado

### TripoSR:

1) Sube la imagen inicial
2) Configura el modelo según los parámetros que necesites
3) Haz clic en el botón `Submit` para obtener el objeto 3D generado

### StableFast3D:

1) Sube la imagen inicial
2) Configura el modelo según los parámetros que necesites
3) Haz clic en el botón `Submit` para obtener el objeto 3D generado

### Shap-E:

1) Ingresa tu solicitud o sube la imagen inicial
2) Configura el modelo según los parámetros que necesites
3) Haz clic en el botón `Submit` para obtener el objeto 3D generado

### SV34D:

1) Sube la imagen inicial (para 3D) o video (para 4D)
2) Configura el modelo según los parámetros que necesites
3) Haz clic en el botón `Submit` para obtener el video 3D generado

### Zero123Plus:

1) Sube la imagen inicial
2) Configura el modelo según los parámetros que necesites
3) Haz clic en el botón `Submit` para obtener la rotación 3D generada de la imagen

### StableAudio:

1) Configura el modelo según los parámetros que necesites
2) Ingresa tu solicitud
3) Haz clic en el botón `Submit` para obtener el audio generado

### AudioCraft:

1) Selecciona un modelo de la lista desplegable
2) Selecciona el tipo de modelo (`musicgen`, `audiogen` o `magnet`)
3) Configura el modelo según los parámetros que necesites
4) Ingresa tu solicitud
5) (Opcional) sube el audio inicial si estás usando el modelo `melody`
6) Haz clic en el botón `Submit` para obtener el audio generado
#### Opcional: Puedes habilitar `multiband diffusion` para mejorar el audio generado

### AudioLDM 2:

1) Selecciona un modelo de la lista desplegable
2) Configura el modelo según los parámetros que necesites
3) Ingresa tu solicitud
4) Haz clic en el botón `Submit` para obtener el audio generado

### Demucs:

1) Sube el audio inicial para separar
2) Haz clic en el botón `Submit` para obtener el audio separado

### Galería:

* Aquí puedes ver archivos del directorio de salidas

### ModelDownloader:

* Aquí puedes descargar modelos `LLM` y `StableDiffusion`. Solo elige el modelo de la lista desplegable y haz clic en el botón `Submit`
#### Los modelos `LLM` se descargan aquí: *inputs/text/llm_models*
#### Los modelos `StableDiffusion` se descargan aquí: *inputs/image/sd_models*

### Configuración: 

* Aquí puedes cambiar la configuración de la aplicación. Por ahora solo puedes cambiar el modo `Share` a `True` o `False`

### Sistema: 

* Aquí puedes ver los indicadores de los sensores de tu computadora haciendo clic en el botón `Submit`

### Información adicional:

1) Todas las generaciones se guardan en la carpeta *outputs*
2) Puedes presionar el botón `Clear` para restablecer tu selección
3) Para detener el proceso de generación, haz clic en el botón `Stop generation`
4) Puedes apagar la aplicación usando el botón `Close terminal`
5) Puedes abrir la carpeta *outputs* haciendo clic en el botón `Outputs`

## ¿Dónde puedo obtener modelos y voces?

* Los modelos LLM se pueden obtener de [HuggingFace](https://huggingface.co/models) o desde ModelDownloader dentro de la interfaz
* Los modelos StableDiffusion, vae, inpaint, embedding y lora se pueden obtener de [CivitAI](https://civitai.com/models) o desde ModelDownloader dentro de la interfaz
* Los modelos StableAudioOpen, AudioCraft, AudioLDM 2, TTS, Whisper, Wav2Lip, SunoBark, MoonDream2, Upscale, GLIGEN, Depth, Pix2Pix, Controlnet, AnimateDiff, Videos, LDM3D, SD3, Cascade, InstantID, PhotoMaker, IP-Adapter-FaceID, Rembg, Roop, CodeFormer, Real-ESRGAN, TripoSR, StableFast3D, Shap-E, SV34D, Zero123Plus, Demucs, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, AuraSR, Würstchen, DeepFloydIF, PixArt, ModelScope, ZeroScope 2, CogVideoX, Latte y Multiband diffusion se descargan automáticamente en la carpeta *inputs* cuando se utilizan
* Puedes obtener voces en cualquier lugar. Graba las tuyas o toma una grabación de Internet. O simplemente usa las que ya están en el proyecto. ¡Lo principal es que esté preprocesada!

## Wiki

* https://github.com/Dartvauder/NeuroSandboxWebUI/wiki

## Agradecimiento a los desarrolladores

#### Muchas gracias a estos proyectos porque gracias a sus aplicaciones/bibliotecas, pude crear mi aplicación:

En primer lugar, quiero agradecer a los desarrolladores de [PyCharm](https://www.jetbrains.com/pycharm/) y [GitHub](https://desktop.github.com). Con la ayuda de sus aplicaciones, pude crear y compartir mi código

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

## Licencias de Terceros:

#### Muchos modelos tienen su propia licencia de uso. Antes de usarlos, te aconsejo que te familiarices con ellas:

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

## Donación

### *Si te gustó mi proyecto y quieres hacer una donación, aquí tienes opciones para donar. ¡Muchas gracias de antemano!*

* CryptoWallet(BEP-20) - 0x3d86bdb5f50b92d0d7Eb44F1a833acC5e91aAEcA

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Dartvauder/NeuroSandboxWebUI&type=Date)](https://star-history.com/#Dartvauder/NeuroSandboxWebUI&Date)
