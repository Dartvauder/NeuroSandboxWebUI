## [기능](/#Features) | [의존성](/#Required-Dependencies) | [시스템요구사항](/#Minimum-System-Requirements) | [설치](/#How-to-install) | [사용법](/#How-to-use) | [모델](/#Where-can-I-get-models-voices-and-avatars) | [위키](/#Wiki) | [개발자 감사](/#Acknowledgment-to-developers) | [라이선스](/#Third-Party-Licenses)

# ![main](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4ea0d891-8979-45ad-b052-626c41ae991a)
* 진행 중! (알파)
* [English](/README.md) | [عربي](/Readmes/README_AR.md) | [Deutsche](/Readmes/README_DE.md) | [Español](/Readmes/README_ES.md) | [Français](/Readmes/README_FR.md) | [日本語](/Readmes/README_JP.md) | [Русский](/Readmes/README_RU.md) | [漢語](/Readmes/README_ZH.md) | [Português](/Readmes/README_PT.md) | [हिंदी](/Readmes/README_HI.md) | [Italiano](/Readmes/README_IT.md) | 韓國語

## 설명:

다양한 신경망 모델을 사용하기 위한 간단하고 편리한 인터페이스입니다. 텍스트, 음성 및 이미지 입력을 사용하여 LLM 및 Moondream2와 통신할 수 있습니다; StableDiffusion, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF 및 PixArt를 사용하여 이미지를 생성할 수 있습니다; ModelScope, ZeroScope 2, CogVideoX 및 Latte를 사용하여 비디오를 생성할 수 있습니다; TripoSR, StableFast3D, Shap-E, SV34D 및 Zero123Plus를 사용하여 3D 객체를 생성할 수 있습니다; StableAudioOpen, AudioCraft 및 AudioLDM 2를 사용하여 음악과 오디오를 생성할 수 있습니다; CoquiTTS 및 SunoBark를 사용하여 텍스트를 음성으로 변환할 수 있습니다; OpenAI-Whisper를 사용하여 음성을 텍스트로 변환할 수 있습니다; Wav2Lip을 사용하여 립싱크를 할 수 있습니다; Roop을 사용하여 얼굴 교체를 할 수 있습니다; Rembg를 사용하여 배경을 제거할 수 있습니다; CodeFormer를 사용하여 얼굴을 복원할 수 있습니다; LibreTranslate를 사용하여 텍스트를 번역할 수 있습니다; Demucs를 사용하여 오디오 파일을 분리할 수 있습니다. 또한 갤러리에서 outputs 디렉토리의 파일을 볼 수 있고, LLM 및 StableDiffusion 모델을 다운로드할 수 있으며, 인터페이스 내에서 애플리케이션 설정을 변경하고 시스템 센서를 확인할 수 있습니다.

이 프로젝트의 목표는 신경망 모델을 사용하는 가장 쉬운 애플리케이션을 만드는 것입니다.

### 텍스트: 

### 이미지: 

### 비디오: 

### 3D: 

### 오디오: 

### 인터페이스: 

## 기능:

* install.bat(Windows) 또는 install.sh(Linux)를 통한 쉬운 설치
* 로컬호스트(IPv4 경유) 또는 온라인 어디서나(Share 경유) 모바일 장치를 통해 애플리케이션을 사용할 수 있습니다
* 유연하고 최적화된 인터페이스 (Gradio 사용)
* admin:admin을 통한 인증 (GradioAuth.txt 파일에 로그인 세부 정보를 입력할 수 있습니다)
* 특정 모델을 다운로드하기 위해 자신의 HuggingFace-Token을 추가할 수 있습니다 (HF-Token.txt 파일에 토큰을 입력할 수 있습니다)
* Transformers 및 llama.cpp 모델 지원 (LLM)
* diffusers 및 safetensors 모델 지원 (StableDiffusion) - txt2img, img2img, depth2img, pix2pix, controlnet, upscale, inpaint, gligen, animatediff, video, ldm3d, sd3, cascade 및 extras 탭
* 이미지 생성을 위한 추가 모델 지원: Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF 및 PixArt
* StableAudioOpen 지원
* AudioCraft 지원 (모델: musicgen, audiogen 및 magnet)
* AudioLDM 2 지원 (모델: audio 및 music)
* TTS 및 Whisper 모델 지원 (LLM 및 TTS-STT용)
* Lora, Textual inversion (embedding), Vae, Img2img, Depth, Pix2Pix, Controlnet, Upscale(latent), Upscale(Real-ESRGAN), Inpaint, GLIGEN, AnimateDiff, Videos, LDM3D, SD3, Cascade, Rembg, CodeFormer 및 Roop 모델 지원 (StableDiffusion용)
* Multiband Diffusion 모델 지원 (AudioCraft용)
* LibreTranslate 지원 (로컬 API)
* 비디오 생성을 위한 ModelScope, ZeroScope 2, CogVideoX 및 Latte 지원
* SunoBark 지원
* Demucs 지원
* 3D 생성을 위한 TripoSR, StableFast3D, Shap-E, SV34D 및 Zero123Plus 지원
* Wav2Lip 지원
* LLM을 위한 Multimodal (Moondream 2), LORA (transformers) 및 WebSearch (GoogleSearch 사용) 지원
* 인터페이스 내 모델 설정
* 갤러리
* ModelDownloader (LLM 및 StableDiffusion용)
* 애플리케이션 설정
* 시스템 센서 확인 기능

## 필수 의존성:

* [Python](https://www.python.org/downloads/) (3.11)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (12.4) 및 [cuDNN](https://developer.nvidia.com/cudnn-downloads) (9.1)
* [FFMPEG](https://ffmpeg.org/download.html)
- C+ 컴파일러
  - Windows: [VisualStudio](https://visualstudio.microsoft.com/ru/)
  - Linux: [GCC](https://gcc.gnu.org/)

## 최소 시스템 요구사항:

* 시스템: Windows 또는 Linux
* GPU: 6GB+ 또는 CPU: 8 코어 3.2GHZ
* RAM: 16GB+
* 디스크 공간: 20GB+
* 모델 다운로드 및 설치를 위한 인터넷 연결

## 설치 방법:

### Windows

1) 원하는 위치에 `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git`
2) `Install.bat`을 실행하고 설치가 완료될 때까지 기다립니다
3) 설치 후 `Start.bat`을 실행합니다
4) 파일 버전을 선택하고 애플리케이션이 실행될 때까지 기다립니다
5) 이제 생성을 시작할 수 있습니다!

업데이트를 받으려면 `Update.bat`을 실행하세요
터미널을 통해 가상 환경으로 작업하려면 `Venv.bat`을 실행하세요

### Linux

1) 원하는 위치에 `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git`
2) 터미널에서 `./Install.sh`을 실행하고 모든 의존성 설치가 완료될 때까지 기다립니다
3) 설치 후 `./Start.sh`을 실행합니다
4) 애플리케이션이 실행될 때까지 기다립니다
5) 이제 생성을 시작할 수 있습니다!

업데이트를 받으려면 `./Update.sh`을 실행하세요
터미널을 통해 가상 환경으로 작업하려면 `./Venv.sh`을 실행하세요

## 사용 방법:

#### 인터페이스에는 여섯 개의 메인 탭에 32개의 탭이 있습니다: LLM, TTS-STT, SunoBark, LibreTranslate, Wav2Lip, StableDiffusion, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF, PixArt, ModelScope, ZeroScope 2, CogVideoX, Latte, TripoSR, StableFast3D, Shap-E, SV34D, Zero123Plus, StableAudio, AudioCraft, AudioLDM 2, Demucs, Gallery, ModelDownloader, Settings 및 System. 필요한 것을 선택하고 아래 지침을 따르세요 

### LLM:

1) 먼저 모델을 *inputs/text/llm_models* 폴더에 업로드합니다
2) 드롭다운 목록에서 모델을 선택합니다
3) 모델 유형을 선택합니다 (`transformers` 또는 `llama`)
4) 필요한 매개변수에 따라 모델을 설정합니다
5) 요청을 입력하거나 말합니다
6) `Submit` 버튼을 클릭하여 생성된 텍스트와 오디오 응답을 받습니다
#### 선택 사항: `TTS` 모드를 활성화하고 오디오 응답을 받기 위해 필요한 `voice`와 `language`를 선택할 수 있습니다. `multimodal`을 활성화하고 이미지를 업로드하여 설명을 얻을 수 있습니다. 인터넷 접속을 위해 `websearch`를 활성화할 수 있습니다. 번역을 얻기 위해 `libretranslate`를 활성화할 수 있습니다. 또한 생성을 개선하기 위해 `LORA` 모델을 선택할 수 있습니다
#### 음성 샘플 = *inputs/audio/voices*
#### LORA = *inputs/text/llm_models/lora*
#### 음성은 사전 처리되어야 합니다 (22050 kHz, 모노, WAV)

### TTS-STT:

1) 텍스트를 음성으로 변환할 텍스트를 입력합니다
2) 음성을 텍스트로 변환할 오디오를 입력합니다
3) `Submit` 버튼을 클릭하여 생성된 텍스트와 오디오 응답을 받습니다
#### 음성 샘플 = *inputs/audio/voices*
#### 음성은 사전 처리되어야 합니다 (22050 kHz, 모노, WAV)

### SunoBark:

1) 요청을 입력합니다
2) 필요한 매개변수에 따라 모델을 설정합니다
3) `Submit` 버튼을 클릭하여 생성된 오디오 응답을 받습니다

### LibreTranslate:

* 먼저 [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate)를 설치하고 실행해야 합니다
1) 원본 언어와 대상 언어를 선택합니다
2) `Submit` 버튼을 클릭하여 번역을 받습니다
#### 선택 사항: 해당 버튼을 켜서 번역 기록을 저장할 수 있습니다

### Wav2Lip:

1) 얼굴의 초기 이미지를 업로드합니다
2) 음성의 초기 오디오를 업로드합니다
3) 필요한 매개변수에 따라 모델을 설정합니다
4) `Submit` 버튼을 클릭하여 립싱크를 받습니다

### StableDiffusion - 15개의 하위 탭이 있습니다:

#### txt2img:

1) 먼저 모델을 *inputs/image/sd_models* 폴더에 업로드합니다
2) 드롭다운 목록에서 모델을 선택합니다
3) 모델 유형을 선택합니다 (`SD`, `SD2` 또는 `SDXL`)
4) 필요한 매개변수에 따라 모델을 설정합니다
5) 요청을 입력합니다 (프롬프트 가중치를 위해 + 및 - 사용)
6) `Submit` 버튼을 클릭하여 생성된 이미지를 받습니다
#### 선택 사항: 생성 방법을 개선하기 위해 `vae`, `embedding` 및 `lora` 모델을 선택할 수 있으며, 생성된 이미지의 크기를 늘리기 위해 `upscale`을 활성화할 수 있습니다 
#### vae = *inputs/image/sd_models/vae*
#### lora = *inputs/image/sd_models/lora*
#### embedding = *inputs/image/sd_models/embedding*

#### img2img:

1) 먼저 모델을 *inputs/image/sd_models* 폴더에 업로드합니다
2) 드롭다운 목록에서 모델을 선택합니다
3) 모델 유형을 선택합니다 (`SD`, `SD2` 또는 `SDXL`)
4) 필요한 매개변수에 따라 모델을 설정합니다
5) 생성에 사용될 초기 이미지를 업로드합니다
6) 요청을 입력합니다 (프롬프트 가중치를 위해 + 및 - 사용)
7) `Submit` 버튼을 클릭하여 생성된 이미지를 받습니다
#### 선택 사항: `vae` 모델을 선택할 수 있습니다
#### vae = *inputs/image/sd_models/vae*

#### depth2img:

1) 초기 이미지를 업로드합니다
2) 필요한 매개변수에 따라 모델을 설정합니다
3) 요청을 입력합니다 (프롬프트 가중치를 위해 + 및 - 사용)
4) `Submit` 버튼을 클릭하여 생성된 이미지를 받습니다

#### pix2pix:

1) 초기 이미지를 업로드합니다
2) 필요한 매개변수에 따라 모델을 설정합니다
3) 요청을 입력합니다 (프롬프트 가중치를 위해 + 및 - 사용)
4) `Submit` 버튼을 클릭하여 생성된 이미지를 받습니다

#### controlnet:

1) 먼저 stable diffusion 모델을 *inputs/image/sd_models* 폴더에 업로드합니다
2) 초기 이미지를 업로드합니다
3) 드롭다운 목록에서 stable diffusion 및 controlnet 모델을 선택합니다
4) 필요한 매개변수에 따라 모델을 설정합니다
5) 요청을 입력합니다 (프롬프트 가중치를 위해 + 및 - 사용)
6) `Submit` 버튼을 클릭하여 생성된 이미지를 받습니다

#### upscale(latent):

1) 초기 이미지를 업로드합니다
2) 필요한 매개변수에 따라 모델을 설정합니다
3) `Submit` 버튼을 클릭하여 업스케일된 이미지를 받습니다

#### upscale(Real-ESRGAN):

1) 초기 이미지를 업로드합니다
2) 필요한 매개변수에 따라 모델을 설정합니다
3) `Submit` 버튼을 클릭하여 업스케일된 이미지를 받습니다

#### inpaint:

1) 먼저 모델을 *inputs/image/sd_models/inpaint* 폴더에 업로드합니다
2) 드롭다운 목록에서 모델을 선택합니다
3) 모델 유형을 선택합니다 (`SD`, `SD2` 또는 `SDXL`)
4) 필요한 매개변수에 따라 모델을 설정합니다
5) 생성에 사용될 이미지를 `initial image`와 `mask image`에 업로드합니다
6) `mask image`에서 브러시를 선택한 다음 팔레트를 선택하고 색상을 `#FFFFFF`로 변경합니다
7) 생성할 위치를 그리고 요청을 입력합니다 (프롬프트 가중치를 위해 + 및 - 사용)
8) `Submit` 버튼을 클릭하여 인페인팅된 이미지를 받습니다
#### 선택 사항: `vae` 모델을 선택할 수 있습니다
#### vae = *inputs/image/sd_models/vae*

#### gligen:

1) 먼저 모델을 *inputs/image/sd_models* 폴더에 업로드합니다
2) 드롭다운 목록에서 모델을 선택합니다
3) 모델 유형을 선택합니다 (`SD`, `SD2` 또는 `SDXL`)
4) 필요한 매개변수에 따라 모델을 설정합니다
5) 프롬프트에 대한 요청 (프롬프트 가중치를 위해 + 및 - 사용)과 GLIGEN 구문 (박스의 경우 ""로 표시)을 입력합니다
6) GLIGEN 박스를 입력합니다 (박스의 경우 [0.1387, 0.2051, 0.4277, 0.7090]와 같이)
7) `Submit` 버튼을 클릭하여 생성된 이미지를 받습니다

#### animatediff:

1) 먼저 모델을 *inputs/image/sd_models* 폴더에 업로드합니다
2) 드롭다운 목록에서 모델을 선택합니다
3) 필요한 매개변수에 따라 모델을 설정합니다
4) 요청을 입력합니다 (프롬프트 가중치를 위해 + 및 - 사용)
5) `Submit` 버튼을 클릭하여 생성된 이미지 애니메이션을 받습니다

#### video:

1) 초기 이미지를 업로드합니다
2) 요청을 입력합니다 (IV2Gen-XL용)
3) 필요한 매개변수에 따라 모델을 설정합니다
4) `Submit` 버튼을 클릭하여 이미지에서 비디오를 받습니다

#### ldm3d:

1) 요청을 입력합니다
2) 필요한 매개변수에 따라 모델을 설정합니다
3) `Submit` 버튼을 클릭하여 생성된 이미지를 받습니다

#### sd3:

1) 요청을 입력합니다
2) 필요한 매개변수에 따라 모델을 설정합니다
3) `Submit` 버튼을 클릭하여 생성된 이미지를 받습니다

#### cascade:

1) 요청을 입력합니다
2) 필요한 매개변수에 따라 모델을 설정합니다
3) `Submit` 버튼을 클릭하여 생성된 이미지를 받습니다

#### extras:

1) 초기 이미지를 업로드합니다
2) 필요한 옵션을 선택합니다
3) `Submit` 버튼을 클릭하여 수정된 이미지를 받습니다

### Kandinsky:

1) 프롬프트를 입력합니다
2) 드롭다운 목록에서 모델을 선택합니다
3) 필요한 매개변수에 따라 모델을 설정합니다
4) `Submit`을 클릭하여 생성된 이미지를 받습니다

### Flux:

1) 프롬프트를 입력합니다
2) 드롭다운 목록에서 모델을 선택합니다
3) 필요한 매개변수에 따라 모델을 설정합니다
4) `Submit`을 클릭하여 생성된 이미지를 받습니다

### HunyuanDiT:

1) 프롬프트를 입력합니다
2) 필요한 매개변수에 따라 모델을 설정합니다
3) `Submit`을 클릭하여 생성된 이미지를 받습니다

### Lumina-T2X:

1) 프롬프트를 입력합니다
2) 필요한 매개변수에 따라 모델을 설정합니다
3) `Submit`을 클릭하여 생성된 이미지를 받습니다

### Kolors:

1) 프롬프트를 입력합니다
2) 필요한 매개변수에 따라 모델을 설정합니다
3) `Submit`을 클릭하여 생성된 이미지를 받습니다

### AuraFlow:

1) 프롬프트를 입력합니다
2) 필요한 매개변수에 따라 모델을 설정합니다
3) `Submit`을 클릭하여 생성된 이미지를 받습니다

### Würstchen:

1) 프롬프트를 입력합니다
2) 필요한 매개변수에 따라 모델을 설정합니다
3) `Submit`을 클릭하여 생성된 이미지를 받습니다

### DeepFloydIF:

1) 프롬프트를 입력합니다
2) 필요한 매개변수에 따라 모델을 설정합니다
3) `Submit`을 클릭하여 생성된 이미지를 받습니다

### PixArt:

1) 프롬프트를 입력합니다
2) 드롭다운 목록에서 모델을 선택합니다
3) 필요한 매개변수에 따라 모델을 설정합니다
4) `Submit`을 클릭하여 생성된 이미지를 받습니다

### ModelScope:

1) 프롬프트를 입력합니다
2) 필요한 매개변수에 따라 모델을 설정합니다
3) `Submit`을 클릭하여 생성된 비디오를 받습니다

### ZeroScope 2:

1) 프롬프트를 입력합니다
2) 필요한 매개변수에 따라 모델을 설정합니다
3) `Submit`을 클릭하여 생성된 비디오를 받습니다

### CogVideoX:

1) 프롬프트를 입력합니다
2) 필요한 매개변수에 따라 모델을 설정합니다
3) `Submit`을 클릭하여 생성된 비디오를 받습니다

### Latte:

1) 프롬프트를 입력합니다
2) 필요한 매개변수에 따라 모델을 설정합니다
3) `Submit`을 클릭하여 생성된 비디오를 받습니다

### TripoSR:

1) 초기 이미지를 업로드합니다
2) 필요한 매개변수에 따라 모델을 설정합니다
3) `Submit` 버튼을 클릭하여 생성된 3D 객체를 받습니다

### StableFast3D:

1) 초기 이미지를 업로드합니다
2) 필요한 매개변수에 따라 모델을 설정합니다
3) `Submit` 버튼을 클릭하여 생성된 3D 객체를 받습니다

### Shap-E:

1) 요청을 입력하거나 초기 이미지를 업로드합니다
2) 필요한 매개변수에 따라 모델을 설정합니다
3) `Submit` 버튼을 클릭하여 생성된 3D 객체를 받습니다

### SV34D:

1) 3D의 경우 초기 이미지를 업로드하거나 4D의 경우 비디오를 업로드합니다
2) 필요한 매개변수에 따라 모델을 설정합니다
3) `Submit` 버튼을 클릭하여 생성된 3D 비디오를 받습니다

### Zero123Plus:

1) 초기 이미지를 업로드합니다
2) 필요한 매개변수에 따라 모델을 설정합니다
3) `Submit` 버튼을 클릭하여 이미지의 생성된 3D 회전을 받습니다

### StableAudio:

1) 필요한 매개변수에 따라 모델을 설정합니다
2) 요청을 입력합니다
3) `Submit` 버튼을 클릭하여 생성된 오디오를 받습니다

### AudioCraft:

1) 드롭다운 목록에서 모델을 선택합니다
2) 모델 유형을 선택합니다 (`musicgen`, `audiogen` 또는 `magnet`)
3) 필요한 매개변수에 따라 모델을 설정합니다
4) 요청을 입력합니다
5) (선택 사항) `melody` 모델을 사용하는 경우 초기 오디오를 업로드합니다 
6) `Submit` 버튼을 클릭하여 생성된 오디오를 받습니다
#### 선택 사항: 생성된 오디오를 개선하기 위해 `multiband diffusion`을 활성화할 수 있습니다

### AudioLDM 2:

1) 드롭다운 목록에서 모델을 선택합니다
2) 필요한 매개변수에 따라 모델을 설정합니다
3) 요청을 입력합니다
4) `Submit` 버튼을 클릭하여 생성된 오디오를 받습니다

### Demucs:

1) 분리할 초기 오디오를 업로드합니다
2) `Submit` 버튼을 클릭하여 분리된 오디오를 받습니다

### 갤러리:

* 여기에서 outputs 디렉토리의 파일을 볼 수 있습니다

### ModelDownloader:

* 여기에서 `LLM` 및 `StableDiffusion` 모델을 다운로드할 수 있습니다. 드롭다운 목록에서 모델을 선택하고 `Submit` 버튼을 클릭하기만 하면 됩니다
#### `LLM` 모델은 여기에 다운로드됩니다: *inputs/text/llm_models*
#### `StableDiffusion` 모델은 여기에 다운로드됩니다: *inputs/image/sd_models*

### 설정: 

* 여기에서 애플리케이션 설정을 변경할 수 있습니다. 현재는 `Share` 모드를 `True` 또는 `False`로 변경할 수 있습니다

### 시스템: 

* 여기에서 `Submit` 버튼을 클릭하여 컴퓨터 센서의 지표를 볼 수 있습니다

### 추가 정보:

1) 모든 생성물은 *outputs* 폴더에 저장됩니다
2) `Clear` 버튼을 눌러 선택을 초기화할 수 있습니다
3) 생성 프로세스를 중지하려면 `Stop generation` 버튼을 클릭하세요
4) `Close terminal` 버튼을 사용하여 애플리케이션을 끌 수 있습니다
5) `Outputs` 버튼을 클릭하여 *outputs* 폴더를 열 수 있습니다

## 모델과 음성은 어디서 구할 수 있나요?

* LLM 모델은 [HuggingFace](https://huggingface.co/models)에서 가져오거나 인터페이스 내의 ModelDownloader에서 가져올 수 있습니다 
* StableDiffusion, vae, inpaint, embedding 및 lora 모델은 [CivitAI](https://civitai.com/models)에서 가져오거나 인터페이스 내의 ModelDownloader에서 가져올 수 있습니다
* StableAudioOpen, AudioCraft, AudioLDM 2, TTS, Whisper, Wav2Lip, SunoBark, MoonDream2, Upscale, GLIGEN, Depth, Pix2Pix, Controlnet, AnimateDiff, Videos, LDM3D, SD3, Cascade, Rembg, Roop, CodeFormer, Real-ESRGAN, TripoSR, StableFast3D, Shap-E, SV34D, Zero123Plus, Demucs, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF, PixArt, ModelScope, ZeroScope 2, CogVideoX, Latte 및 Multiband diffusion 모델은 사용될 때 *inputs* 폴더에 자동으로 다운로드됩니다 
* 음성은 어디에서나 가져올 수 있습니다. 자신의 목소리를 녹음하거나 인터넷에서 녹음을 가져오세요. 또는 프로젝트에 이미 있는 것을 사용하세요. 중요한 것은 사전 처리되어 있어야 한다는 것입니다!

## 위키

* https://github.com/Dartvauder/NeuroSandboxWebUI/wiki

## 개발자 감사

#### 이 프로젝트들 덕분에 제 애플리케이션을 만들 수 있었습니다. 그들의 애플리케이션/라이브러리에 많은 감사를 드립니다:

먼저, [PyCharm](https://www.jetbrains.com/pycharm/)과 [GitHub](https://desktop.github.com) 개발자들에게 감사드립니다. 그들의 애플리케이션 덕분에 제 코드를 만들고 공유할 수 있었습니다

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

## 제3자 라이선스:

#### 많은 모델들이 자체 사용 라이선스를 가지고 있습니다. 사용하기 전에 이를 숙지하는 것이 좋습니다:

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

## 후원

### *제 프로젝트가 마음에 들어 후원하고 싶으시다면 여기 옵션이 있습니다. 미리 대단히 감사드립니다!*

* 암호화폐 지갑(BEP-20) - 0x3d86bdb5f50b92d0d7Eb44F1a833acC5e91aAEcA

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)
