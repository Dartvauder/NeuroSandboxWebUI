## [Recursos](/#Recursos) | [Dependências](/#Dependências-Necessárias) | [RequisitosDoSistema](/#Requisitos-Mínimos-do-Sistema) | [Instalação](/#Como-instalar) | [Uso](/#Como-usar) | [Modelos](/#Onde-posso-obter-modelos-vozes-e-avatares) | [Wiki](/#Wiki) | [Agradecimento](/#Agradecimento-aos-desenvolvedores) | [Licenças](/#Licenças-de-Terceiros)

# ![main](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4ea0d891-8979-45ad-b052-626c41ae991a)
* Trabalho em andamento! (ALFA)
* [English](/README.md) | [عربي](/Readmes/README_AR.md) | [Deutsche](/Readmes/README_DE.md) | [Español](/Readmes/README_ES.md) | [Français](/Readmes/README_FR.md) | [日本語](/Readmes/README_JP.md) | [Русский](/Readmes/README_RU.md) | [漢語](/Readmes/README_ZH.md) | Português | [हिंदी](/Readmes/README_HI.md) | [Italiano](/Readmes/README_IT.md) | [韓國語](/Readmes/README_KO.md)

## Descrição:

Uma interface simples e conveniente para usar vários modelos de redes neurais. Você pode se comunicar com LLM e Moondream2 usando entrada de texto, voz e imagem; usar StableDiffusion, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF e PixArt para gerar imagens; ModelScope, ZeroScope 2, CogVideoX e Latte para gerar vídeos; TripoSR, StableFast3D, Shap-E, SV34D e Zero123Plus para gerar objetos 3D; StableAudioOpen, AudioCraft e AudioLDM 2 para gerar música e áudio; CoquiTTS e SunoBark para texto para fala; OpenAI-Whisper para fala para texto; Wav2Lip para sincronização labial; Roop para troca de rostos; Rembg para remover o fundo; CodeFormer para restauração facial; LibreTranslate para tradução de texto; Demucs para separação de arquivos de áudio. Você também pode visualizar arquivos do diretório de saídas na galeria, baixar os modelos LLM e StableDiffusion, alterar as configurações do aplicativo dentro da interface e verificar os sensores do sistema

O objetivo do projeto - criar o aplicativo mais fácil possível de usar modelos de redes neurais

### Texto: <img width="1127" alt="1pt" src="https://github.com/user-attachments/assets/298aa70b-655e-44c5-bc73-f14e4c7c34f4">

### Imagem: <img width="1123" alt="2pt" src="https://github.com/user-attachments/assets/462fff2a-8350-4aed-8bee-8b107a0befe8">

### Vídeo: <img width="1124" alt="3pt" src="https://github.com/user-attachments/assets/875748bf-efa0-44f9-92e7-204d46ceb914">

### 3D: <img width="1120" alt="4pt" src="https://github.com/user-attachments/assets/6a2a9842-8955-4d46-a253-798ee27bf96f">

### Áudio: <img width="1124" alt="5pt" src="https://github.com/user-attachments/assets/fd530f6a-22c1-4797-8b6e-d9c0ccc79200">

### Interface: <img width="1124" alt="6pt" src="https://github.com/user-attachments/assets/f168fba8-9f18-4f62-bfc5-59cc291acb04">

## Recursos:

* Instalação fácil via install.bat(Windows) ou install.sh(Linux)
* Você pode usar o aplicativo através do seu dispositivo móvel em localhost(Via IPv4) ou em qualquer lugar online(Via Share)
* Interface flexível e otimizada (Por Gradio)
* Autenticação via admin:admin (Você pode inserir seus detalhes de login no arquivo GradioAuth.txt)
* Você pode adicionar seu próprio HuggingFace-Token para baixar modelos específicos (Você pode inserir seu token no arquivo HF-Token.txt)
* Suporte para modelos Transformers e llama.cpp (LLM)
* Suporte para modelos diffusers e safetensors (StableDiffusion) - abas txt2img, img2img, depth2img, pix2pix, controlnet, upscale, inpaint, gligen, animatediff, video, ldm3d, sd3, cascade, adapters e extras
* Suporte de modelos adicionais para geração de imagens: Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF e PixArt
* Suporte StableAudioOpen
* Suporte AudioCraft (Modelos: musicgen, audiogen e magnet)
* Suporte AudioLDM 2 (Modelos: audio e music)
* Suporta modelos TTS e Whisper (Para LLM e TTS-STT)
* Suporta modelos Lora, Textual inversion (embedding), Vae, Img2img, Depth, Pix2Pix, Controlnet, Upscale(latent), Upscale(Real-ESRGAN), Inpaint, GLIGEN, AnimateDiff, Videos, LDM3D, SD3, Cascade, Adapters (InstantID, PhotoMaker, IP-Adapter-FaceID), Rembg, CodeFormer e Roop (Para StableDiffusion)
* Suporte ao modelo Multiband Diffusion (Para AudioCraft)
* Suporte LibreTranslate (API Local)
* Suporte ModelScope, ZeroScope 2, CogVideoX e Latte para geração de vídeo
* Suporte SunoBark
* Suporte Demucs
* Suporte TripoSR, StableFast3D, Shap-E, SV34D e Zero123Plus para geração 3D
* Suporte Wav2Lip
* Suporte Multimodal (Moondream 2), LORA (transformers) e WebSearch (com GoogleSearch) para LLM
* Configurações do modelo dentro da interface
* Galeria
* ModelDownloader (Para LLM e StableDiffusion)
* Configurações do aplicativo
* Capacidade de ver sensores do sistema

## Dependências Necessárias:

* [Python](https://www.python.org/downloads/) (3.11)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (12.4) e [cuDNN](https://developer.nvidia.com/cudnn-downloads) (9.1)
* [FFMPEG](https://ffmpeg.org/download.html)
- Compilador C+
  - Windows: [VisualStudio](https://visualstudio.microsoft.com/ru/)
  - Linux: [GCC](https://gcc.gnu.org/)

## Requisitos Mínimos do Sistema:

* Sistema: Windows ou Linux
* GPU: 6GB+ ou CPU: 8 núcleos 3.2GHZ
* RAM: 16GB+
* Espaço em disco: 20GB+
* Internet para baixar modelos e instalação

## Como instalar:

### Windows

1) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` em qualquer local
2) Execute o `Install.bat` e aguarde a instalação
3) Após a instalação, execute `Start.bat`
4) Selecione a versão do arquivo e aguarde o lançamento do aplicativo
5) Agora você pode começar a gerar!

Para obter atualização, execute `Update.bat`
Para trabalhar com o ambiente virtual através do terminal, execute `Venv.bat`

### Linux

1) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` em qualquer local
2) No terminal, execute o `./Install.sh` e aguarde a instalação de todas as dependências
3) Após a instalação, execute `./Start.sh`
4) Aguarde o lançamento do aplicativo
5) Agora você pode começar a gerar!

Para obter atualização, execute `./Update.sh`
Para trabalhar com o ambiente virtual através do terminal, execute `./Venv.sh`

## Como usar:

#### A interface tem trinta e duas abas em seis abas principais (Texto, Imagem, Vídeo, 3D, Áudio e Interface): LLM, TTS-STT, SunoBark, LibreTranslate, Wav2Lip, StableDiffusion, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF, PixArt, ModelScope, ZeroScope 2, CogVideoX, Latte, TripoSR, StableFast3D, Shap-E, SV34D, Zero123Plus, StableAudio, AudioCraft, AudioLDM 2, Demucs, Galeria, ModelDownloader, Configurações e Sistema. Selecione a que você precisa e siga as instruções abaixo

### LLM:

1) Primeiro, carregue seus modelos na pasta: *inputs/text/llm_models*
2) Selecione seu modelo na lista suspensa
3) Selecione o tipo de modelo (`transformers` ou `llama`)
4) Configure o modelo de acordo com os parâmetros que você precisa
5) Digite (ou fale) sua solicitação
6) Clique no botão `Submit` para receber o texto gerado e a resposta de áudio
#### Opcional: você pode ativar o modo `TTS`, selecionar a `voz` e o `idioma` necessários para receber uma resposta de áudio. Você pode ativar `multimodal` e carregar uma imagem para obter sua descrição. Você pode ativar `websearch` para acesso à Internet. Você pode ativar `libretranslate` para obter a tradução. Também pode escolher o modelo `LORA` para melhorar a geração
#### Amostras de voz = *inputs/audio/voices*
#### LORA = *inputs/text/llm_models/lora*
#### A voz deve ser pré-processada (22050 kHz, mono, WAV)

### TTS-STT:

1) Digite o texto para texto para fala
2) Insira o áudio para fala para texto
3) Clique no botão `Submit` para receber o texto gerado e a resposta de áudio
#### Amostras de voz = *inputs/audio/voices*
#### A voz deve ser pré-processada (22050 kHz, mono, WAV)

### SunoBark:

1) Digite sua solicitação
2) Configure o modelo de acordo com os parâmetros que você precisa
3) Clique no botão `Submit` para receber a resposta de áudio gerada

### LibreTranslate:

* Primeiro, você precisa instalar e executar [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate)
1) Selecione os idiomas de origem e destino
2) Clique no botão `Submit` para obter a tradução
#### Opcional: você pode salvar o histórico de tradução ativando o botão correspondente

### Wav2Lip:

1) Carregue a imagem inicial do rosto
2) Carregue o áudio inicial da voz
3) Configure o modelo de acordo com os parâmetros que você precisa
4) Clique no botão `Submit` para receber a sincronização labial

### StableDiffusion - tem quinze sub-abas:

#### txt2img:

1) Primeiro, carregue seus modelos na pasta: *inputs/image/sd_models*
2) Selecione seu modelo na lista suspensa
3) Selecione o tipo de modelo (`SD`, `SD2` ou `SDXL`)
4) Configure o modelo de acordo com os parâmetros que você precisa
5) Digite sua solicitação (+ e - para ponderação de prompt)
6) Clique no botão `Submit` para obter a imagem gerada
#### Opcional: Você pode selecionar seus modelos `vae`, `embedding` e `lora` para melhorar o método de geração, também pode ativar `upscale` para aumentar o tamanho da imagem gerada
#### vae = *inputs/image/sd_models/vae*
#### lora = *inputs/image/sd_models/lora*
#### embedding = *inputs/image/sd_models/embedding*

#### img2img:

1) Primeiro, carregue seus modelos na pasta: *inputs/image/sd_models*
2) Selecione seu modelo na lista suspensa
3) Selecione o tipo de modelo (`SD`, `SD2` ou `SDXL`)
4) Configure o modelo de acordo com os parâmetros que você precisa
5) Carregue a imagem inicial com a qual a geração ocorrerá
6) Digite sua solicitação (+ e - para ponderação de prompt)
7) Clique no botão `Submit` para obter a imagem gerada
#### Opcional: Você pode selecionar seu modelo `vae`
#### vae = *inputs/image/sd_models/vae*

#### depth2img:

1) Carregue a imagem inicial
2) Configure o modelo de acordo com os parâmetros que você precisa
3) Digite sua solicitação (+ e - para ponderação de prompt)
4) Clique no botão `Submit` para obter a imagem gerada

#### pix2pix:

1) Carregue a imagem inicial
2) Configure o modelo de acordo com os parâmetros que você precisa
3) Digite sua solicitação (+ e - para ponderação de prompt)
4) Clique no botão `Submit` para obter a imagem gerada

#### controlnet:

1) Primeiro, carregue seus modelos stable diffusion na pasta: *inputs/image/sd_models*
2) Carregue a imagem inicial
3) Selecione seus modelos stable diffusion e controlnet nas listas suspensas
4) Configure os modelos de acordo com os parâmetros que você precisa
5) Digite sua solicitação (+ e - para ponderação de prompt)
6) Clique no botão `Submit` para obter a imagem gerada

#### upscale(latent):

1) Carregue a imagem inicial
2) Configure o modelo de acordo com os parâmetros que você precisa
3) Clique no botão `Submit` para obter a imagem ampliada

#### upscale(Real-ESRGAN):

1) Carregue a imagem inicial
2) Configure o modelo de acordo com os parâmetros que você precisa
3) Clique no botão `Submit` para obter a imagem ampliada

#### inpaint:

1) Primeiro, carregue seus modelos na pasta: *inputs/image/sd_models/inpaint*
2) Selecione seu modelo na lista suspensa
3) Selecione o tipo de modelo (`SD`, `SD2` ou `SDXL`)
4) Configure o modelo de acordo com os parâmetros que você precisa
5) Carregue a imagem com a qual a geração ocorrerá em `initial image` e `mask image`
6) Em `mask image`, selecione o pincel, depois a paleta e mude a cor para `#FFFFFF`
7) Desenhe um local para geração e digite sua solicitação (+ e - para ponderação de prompt)
8) Clique no botão `Submit` para obter a imagem com inpainting
#### Opcional: Você pode selecionar seu modelo `vae`
#### vae = *inputs/image/sd_models/vae*

#### gligen:

1) Primeiro, carregue seus modelos na pasta: *inputs/image/sd_models*
2) Selecione seu modelo na lista suspensa
3) Selecione o tipo de modelo (`SD`, `SD2` ou `SDXL`)
4) Configure o modelo de acordo com os parâmetros que você precisa
5) Digite sua solicitação para o prompt (+ e - para ponderação de prompt) e frases GLIGEN (entre "" para caixa)
6) Digite as caixas GLIGEN (Como [0.1387, 0.2051, 0.4277, 0.7090] para caixa)
7) Clique no botão `Submit` para obter a imagem gerada

#### animatediff:

1) Primeiro, carregue seus modelos na pasta: *inputs/image/sd_models*
2) Selecione seu modelo na lista suspensa
3) Configure o modelo de acordo com os parâmetros que você precisa
4) Digite sua solicitação (+ e - para ponderação de prompt)
5) Clique no botão `Submit` para obter a animação de imagem gerada

#### video:

1) Carregue a imagem inicial
2) Digite sua solicitação (para IV2Gen-XL)
3) Configure o modelo de acordo com os parâmetros que você precisa
4) Clique no botão `Submit` para obter o vídeo a partir da imagem

#### ldm3d:

1) Digite sua solicitação
2) Configure o modelo de acordo com os parâmetros que você precisa
3) Clique no botão `Submit` para obter as imagens geradas

#### sd3 (txt2img, img2img, controlnet, inpaint):

1) Digite sua solicitação
2) Configure o modelo de acordo com os parâmetros que você precisa
3) Clique no botão `Submit` para obter a imagem gerada

#### cascade:

1) Digite sua solicitação
2) Configure o modelo de acordo com os parâmetros que você precisa
3) Clique no botão `Submit` para obter a imagem gerada

#### extras:

1) Carregue a imagem inicial
2) Selecione as opções que você precisa
3) Clique no botão `Submit` para obter a imagem modificada

### Kandinsky (txt2img, img2img, inpaint):

1) Digite seu prompt
2) Selecione um modelo na lista suspensa
3) Configure o modelo de acordo com os parâmetros que você precisa
4) Clique em `Submit` para obter a imagem gerada

### Flux:

1) Digite seu prompt
2) Selecione um modelo na lista suspensa
3) Configure o modelo de acordo com os parâmetros que você precisa
4) Clique em `Submit` para obter a imagem gerada

### HunyuanDiT:

1) Digite seu prompt
2) Configure o modelo de acordo com os parâmetros que você precisa
3) Clique em `Submit` para obter a imagem gerada

### Lumina-T2X:

1) Digite seu prompt
2) Configure o modelo de acordo com os parâmetros que você precisa
3) Clique em `Submit` para obter a imagem gerada

### Kolors:

1) Digite seu prompt
2) Configure o modelo de acordo com os parâmetros que você precisa
3) Clique em `Submit` para obter a imagem gerada

### AuraFlow:

1) Digite seu prompt
2) Configure o modelo de acordo com os parâmetros que você precisa
3) Clique em `Submit` para obter a imagem gerada

### Würstchen:

1) Digite seu prompt
2) Configure o modelo de acordo com os parâmetros que você precisa
3) Clique em `Submit` para obter a imagem gerada

### DeepFloydIF (txt2img, img2img, inpaint):

1) Digite seu prompt
2) Configure o modelo de acordo com os parâmetros que você precisa
3) Clique em `Submit` para obter a imagem gerada

### PixArt:

1) Digite seu prompt
2) Selecione o modelo na lista suspensa
3) Configure o modelo de acordo com os parâmetros que você precisa
4) Clique em `Submit` para obter a imagem gerada

### ModelScope:

1) Digite seu prompt
2) Configure o modelo de acordo com os parâmetros que você precisa
3) Clique em `Submit` para obter o vídeo gerado

### ZeroScope 2:

1) Digite seu prompt
2) Configure o modelo de acordo com os parâmetros que você precisa
3) Clique em `Submit` para obter o vídeo gerado

### CogVideoX:

1) Digite seu prompt
2) Configure o modelo de acordo com os parâmetros que você precisa
3) Clique em `Submit` para obter o vídeo gerado

### Latte:

1) Digite seu prompt
2) Configure o modelo de acordo com os parâmetros que você precisa
3) Clique em `Submit` para obter o vídeo gerado

### TripoSR:

1) Carregue a imagem inicial
2) Configure o modelo de acordo com os parâmetros que você precisa
3) Clique no botão `Submit` para obter o objeto 3D gerado

### StableFast3D:

1) Carregue a imagem inicial
2) Configure o modelo de acordo com os parâmetros que você precisa
3) Clique no botão `Submit` para obter o objeto 3D gerado

### Shap-E:

1) Digite sua solicitação ou carregue a imagem inicial
2) Configure o modelo de acordo com os parâmetros que você precisa
3) Clique no botão `Submit` para obter o objeto 3D gerado

### SV34D:

1) Carregue a imagem inicial (para 3D) ou vídeo (para 4D)
2) Configure o modelo de acordo com os parâmetros que você precisa
3) Clique no botão `Submit` para obter o vídeo 3D gerado

### Zero123Plus:

1) Carregue a imagem inicial
2) Configure o modelo de acordo com os parâmetros que você precisa
3) Clique no botão `Submit` para obter a rotação 3D gerada da imagem

### StableAudio:

1) Configure o modelo de acordo com os parâmetros que você precisa
2) Digite sua solicitação
3) Clique no botão `Submit` para obter o áudio gerado

### AudioCraft:

1) Selecione um modelo na lista suspensa
2) Selecione o tipo de modelo (`musicgen`, `audiogen` ou `magnet`)
3) Configure o modelo de acordo com os parâmetros que você precisa
4) Digite sua solicitação
5) (Opcional) carregue o áudio inicial se estiver usando o modelo `melody`
6) Clique no botão `Submit` para obter o áudio gerado
#### Opcional: Você pode ativar `multiband diffusion` para melhorar o áudio gerado

### AudioLDM 2:

1) Selecione um modelo na lista suspensa
2) Configure o modelo de acordo com os parâmetros que você precisa
3) Digite sua solicitação
4) Clique no botão `Submit` para obter o áudio gerado

### Demucs:

1) Carregue o áudio inicial para separar
2) Clique no botão `Submit` para obter o áudio separado

### Galeria:

* Aqui você pode visualizar arquivos do diretório de saídas

### ModelDownloader:

* Aqui você pode baixar modelos `LLM` e `StableDiffusion`. Basta escolher o modelo na lista suspensa e clicar no botão `Submit`
#### Modelos `LLM` são baixados aqui: *inputs/text/llm_models*
#### Modelos `StableDiffusion` são baixados aqui: *inputs/image/sd_models*

### Configurações:

* Aqui você pode alterar as configurações do aplicativo. Por enquanto, você só pode alterar o modo `Share` para `True` ou `False`

### Sistema:

* Aqui você pode ver os indicadores dos sensores do seu computador clicando no botão `Submit`

### Informações Adicionais:

1) Todas as gerações são salvas na pasta *outputs*
2) Você pode pressionar o botão `Clear` para redefinir sua seleção
3) Para interromper o processo de geração, clique no botão `Stop generation`
4) Você pode desligar o aplicativo usando o botão `Close terminal`
5) Você pode abrir a pasta *outputs* clicando no botão `Outputs`

## Onde posso obter modelos e vozes?

* Modelos LLM podem ser obtidos no [HuggingFace](https://huggingface.co/models) ou no ModelDownloader dentro da interface
* Modelos StableDiffusion, vae, inpaint, embedding e lora podem ser obtidos no [CivitAI](https://civitai.com/models) ou no ModelDownloader dentro da interface
* Modelos StableAudioOpen, AudioCraft, AudioLDM 2, TTS, Whisper, Wav2Lip, SunoBark, MoonDream2, Upscale, GLIGEN, Depth, Pix2Pix, Controlnet, AnimateDiff, Videos, LDM3D, SD3, Cascade, InstantID, PhotoMaker, IP-Adapter-FaceID, Rembg, Roop, CodeFormer, Real-ESRGAN, TripoSR, StableFast3D, Shap-E, SV34D, Zero123Plus, Demucs, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, AuraSR, Würstchen, DeepFloydIF, PixArt, ModelScope, ZeroScope 2, CogVideoX, Latte e Multiband diffusion são baixados automaticamente na pasta *inputs* quando são usados
* Você pode obter vozes em qualquer lugar. Grave as suas ou pegue uma gravação da Internet. Ou simplesmente use aquelas que já estão no projeto. O importante é que seja pré-processado!

## Wiki

* https://github.com/Dartvauder/NeuroSandboxWebUI/wiki

## Agradecimento aos desenvolvedores

#### Muito obrigado a estes projetos porque, graças às suas aplicações/bibliotecas, pude criar minha aplicação:

Antes de tudo, quero agradecer aos desenvolvedores do [PyCharm](https://www.jetbrains.com/pycharm/) e [GitHub](https://desktop.github.com). Com a ajuda de suas aplicações, pude criar e compartilhar meu código

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

## Licenças de Terceiros:

#### Muitos modelos têm sua própria licença de uso. Antes de usá-lo, aconselho que você se familiarize com elas:

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

## Doação

### *Se você gostou do meu projeto e deseja fazer uma doação, aqui estão as opções para doar. Muito obrigado antecipadamente!*

* CryptoWallet(BEP-20) - 0x3d86bdb5f50b92d0d7Eb44F1a833acC5e91aAEcA

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Dartvauder/NeuroSandboxWebUI&type=Date)](https://star-history.com/#Dartvauder/NeuroSandboxWebUI&Date)
