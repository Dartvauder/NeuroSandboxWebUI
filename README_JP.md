## [機能](/#機能) | [依存関係](/#必要な依存関係) | [システム要件](/#最小システム要件) | [インストール](/#インストール方法) | [使用方法](/#使用方法) | [モデル](/#モデル、音声、アバターの入手方法) | [Wiki](/#Wiki) | [開発者への感謝](/#開発者への感謝) | [ライセンス](/#サードパーティライセンス)

# ![main](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4ea0d891-8979-45ad-b052-626c41ae991a)
* 開発中！ (ALPHA)
* 英語 | [ロシア語](/README_RU.md)

## 説明:

さまざまなニューラルネットワークモデルを使用するためのシンプルで便利なインターフェース。テキスト、音声、画像入力を使用してLLMおよびMoondream2と通信できます。StableDiffusion、Kandinsky、Flux、HunyuanDiT、Lumina-T2X、Kolors、AuraFlow、Würstchen、DeepFloydIF、PixArtを使用して画像を生成します。ModelScope、ZeroScope 2、CogVideoX、Latteを使用してビデオを生成します。TripoSR、StableFast3D、Shap-E、SV34D、Zero123Plusを使用して3Dオブジェクトを生成します。StableAudioOpen、AudioCraft、AudioLDM 2を使用して音楽やオーディオを生成します。CoquiTTSおよびSunoBarkを使用してテキストを音声に変換します。OpenAI-Whisperを使用して音声をテキストに変換します。Wav2Lipを使用してリップシンクを行います。Roopを使用して顔を交換します。Rembgを使用して背景を削除します。CodeFormerを使用して顔を復元します。LibreTranslateを使用してテキストを翻訳します。Demucsを使用してオーディオファイルを分離します。ギャラリーでoutputsディレクトリのファイルを表示したり、LLMおよびStableDiffusionモデルをダウンロードしたり、インターフェース内でアプリケーション設定を変更したり、システムセンサーを確認したりできます。

プロジェクトの目標 - ニューラルネットワークモデルを使用するための最も簡単なアプリケーションを作成すること

### テキスト: <img width="1119" alt="1" src="https://github.com/user-attachments/assets/e1ac4e8e-feb2-484b-a399-61ddc8a098c1">

### 画像: <img width="1123" alt="2" src="https://github.com/user-attachments/assets/6ee09668-b084-40fe-9e4f-ae776740f2d5">

### ビデオ: <img width="1118" alt="3" src="https://github.com/user-attachments/assets/a568c3ed-3b00-4e21-b802-a3e63f6cf97c">

### 3D: <img width="1118" alt="4" src="https://github.com/user-attachments/assets/0ba23ac4-aecc-44e6-b252-1fc0f478c75e">

### オーディオ: <img width="1127" alt="5" src="https://github.com/user-attachments/assets/ea7f1bd0-ff85-4873-b9dd-cabd1cc89cee">

### インターフェース: <img width="1120" alt="6" src="https://github.com/user-attachments/assets/81c4e40c-cf01-488d-adc8-7330f1edd610">

## 機能:

* install.bat(Windows)またはinstall.sh(Linux)を使用した簡単なインストール
* モバイルデバイスを使用してlocalhost(Via IPv4)またはオンライン(Via Share)でアプリケーションを使用できます
* 柔軟で最適化されたインターフェース (Gradioによる)
* admin:adminを使用した認証 (GradioAuth.txtファイルにログイン情報を入力できます)
* 特定のモデルをダウンロードするために独自のHuggingFace-Tokenを追加できます (HF-Token.txtファイルにトークンを入力できます)
* Transformersおよびllama.cppモデルのサポート (LLM)
* diffusersおよびsafetensorsモデルのサポート (StableDiffusion) - txt2img、img2img、depth2img、pix2pix、controlnet、upscale、inpaint、gligen、animatediff、video、ldm3d、sd3、cascade、extrasタブ
* 画像生成のための追加モデルのサポート: Kandinsky、Flux、HunyuanDiT、Lumina-T2X、Kolors、AuraFlow、Würstchen、DeepFloydIF、PixArt
* StableAudioOpenのサポート
* AudioCraftのサポート (モデル: musicgen、audiogen、magnet)
* AudioLDM 2のサポート (モデル: audio、music)
* TTSおよびWhisperモデルのサポート (LLMおよびTTS-STT用)
* Lora、Textual inversion (embedding)、Vae、Img2img、Depth、Pix2Pix、Controlnet、Upscale、Inpaint、GLIGEN、AnimateDiff、Videos、LDM3D、SD3、Cascade、Rembg、CodeFormer、Roopモデルのサポート (StableDiffusion用)
* Multiband Diffusionモデルのサポート (AudioCraft用)
* LibreTranslateのサポート (ローカルAPI)
* ModelScope、ZeroScope 2、CogVideoX、Latteのサポート (ビデオ生成用)
* SunoBarkのサポート
* Demucsのサポート
* TripoSR、StableFast3D、Shap-E、SV34D、Zero123Plusのサポート (3D生成用)
* Wav2Lipのサポート
* Multimodal (Moondream 2)、LORA (transformers)、WebSearch (GoogleSearch付き)のサポート (LLM用)
* インターフェース内のモデル設定
* ギャラリー
* ModelDownloader (LLMおよびStableDiffusion用)
* アプリケーション設定
* システムセンサーの表示機能

## 必要な依存関係:

* [Python](https://www.python.org/downloads/) (3.11)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (12.4) および [cuDNN](https://developer.nvidia.com/cudnn-downloads) (9.1)
* [FFMPEG](https://ffmpeg.org/download.html)
- C+コンパイラ
  - Windows: [VisualStudio](https://visualstudio.microsoft.com/ru/)
  - Linux: [GCC](https://gcc.gnu.org/)

## 最小システム要件:

* システム: WindowsまたはLinux
* GPU: 6GB+ または CPU: 8コア 3.2GHZ
* RAM: 16GB+
* ディスクスペース: 20GB+
* モデルのダウンロードとインストールのためのインターネット

## インストール方法:

### Windows

1) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` を任意の場所にクローン
2) `Install.bat`を実行し、インストールが完了するまで待ちます
3) インストール後、`Start.bat`を実行します
4) ファイルバージョンを選択し、アプリケーションの起動を待ちます
5) これで生成を開始できます！

更新を取得するには、`Update.bat`を実行します
ターミナルを通じて仮想環境で作業するには、`Venv.bat`を実行します

### Linux

1) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` を任意の場所にクローン
2) ターミナルで`./Install.sh`を実行し、すべての依存関係のインストールが完了するまで待ちます
3) インストール後、ターミナルで`./Start.sh`を実行します
4) アプリケーションの起動を待ちます
5) これで生成を開始できます！

更新を取得するには、`./Update.sh`を実行します
ターミナルを通じて仮想環境で作業するには、`./Venv.sh`を実行します

## 使用方法:

#### インターフェースには、LLM、TTS-STT、SunoBark、LibreTranslate、Wav2Lip、StableDiffusion、Kandinsky、Flux、HunyuanDiT、Lumina-T2X、Kolors、AuraFlow、Würstchen、DeepFloydIF、PixArt、ModelScope、ZeroScope 2、CogVideoX、Latte、TripoSR、StableFast3D、Shap-E、SV34D、Zero123Plus、StableAudio、AudioCraft、AudioLDM 2、Demucs、ギャラリー、ModelDownloader、設定、システムの6つのメインタブに32のサブタブがあります。必要なタブを選択し、以下の手順に従ってください。

### LLM:

1) まず、モデルをフォルダにアップロードします: *inputs/text/llm_models*
2) ドロップダウンリストからモデルを選択します
3) モデルタイプを選択します (`transformers` または `llama`)
4) 必要なパラメータに従ってモデルを設定します
5) リクエストを入力（または話す）します
6) `Submit`ボタンをクリックして、生成されたテキストと音声の応答を受け取ります
#### オプション: `TTS`モードを有効にし、音声応答を受け取るために必要な`voice`と`language`を選択できます。`multimodal`を有効にして画像をアップロードし、その説明を取得できます。`websearch`を有効にしてインターネットにアクセスできます。`libretranslate`を有効にして翻訳を取得できます。また、生成を改善するために`LORA`モデルを選択できます。
#### 音声サンプル = *inputs/audio/voices*
#### LORA = *inputs/text/llm_models/lora*
#### 音声は事前に処理されている必要があります (22050 kHz、モノラル、WAV)

### TTS-STT:

1) テキストを入力してテキストを音声に変換します
2) 音声を入力して音声をテキストに変換します
3) `Submit`ボタンをクリックして、生成されたテキストと音声の応答を受け取ります
#### 音声サンプル = *inputs/audio/voices*
#### 音声は事前に処理されている必要があります (22050 kHz、モノラル、WAV)

### SunoBark:

1) リクエストを入力します
2) 必要なパラメータに従ってモデルを設定します
3) `Submit`ボタンをクリックして、生成された音声の応答を受け取ります

### LibreTranslate:

* まず、[LibreTranslate](https://github.com/LibreTranslate/LibreTranslate)をインストールして実行する必要があります
1) ソース言語とターゲット言語を選択します
2) `Submit`ボタンをクリックして翻訳を取得します
#### オプション: 対応するボタンをオンにして翻訳履歴を保存できます

### Wav2Lip:

1) 顔の初期画像をアップロードします
2) 音声の初期音声をアップロードします
3) 必要なパラメータに従ってモデルを設定します
4) `Submit`ボタンをクリックしてリップシンクを受け取ります

### StableDiffusion - 14のサブタブがあります:

#### txt2img:

1) まず、モデルをフォルダにアップロードします: *inputs/image/sd_models*
2) ドロップダウンリストからモデルを選択します
3) モデルタイプを選択します (`SD`、`SD2`、`SDXL`)
4) 必要なパラメータに従ってモデルを設定します
5) リクエストを入力します (+ と - でプロンプトの重み付け)
6) `Submit`ボタンをクリックして生成された画像を取得します
#### オプション: `vae`、`embedding`、`lora`モデルを選択して生成方法を改善できます。また、`upscale`を有効にして生成された画像のサイズを大きくできます。
#### vae = *inputs/image/sd_models/vae*
#### lora = *inputs/image/sd_models/lora*
#### embedding = *inputs/image/sd_models/embedding*

#### img2img:

1) まず、モデルをフォルダにアップロードします: *inputs/image/sd_models*
2) ドロップダウンリストからモデルを選択します
3) モデルタイプを選択します (`SD`、`SD2`、`SDXL`)
4) 必要なパラメータに従ってモデルを設定します
5) 生成に使用する初期画像をアップロードします
6) リクエストを入力します (+ と - でプロンプトの重み付け)
7) `Submit`ボタンをクリックして生成された画像を取得します
#### オプション: `vae`モデルを選択できます
#### vae = *inputs/image/sd_models/vae*

#### depth2img:

1) 初期画像をアップロードします
2) 必要なパラメータに従ってモデルを設定します
3) リクエストを入力します (+ と - でプロンプトの重み付け)
4) `Submit`ボタンをクリックして生成された画像を取得します

#### pix2pix:

1) 初期画像をアップロードします
2) 必要なパラメータに従ってモデルを設定します
3) リクエストを入力します (+ と - でプロンプトの重み付け)
4) `Submit`ボタンをクリックして生成された画像を取得します

#### controlnet:

1) まず、Stable Diffusionモデルをフォルダにアップロードします: *inputs/image/sd_models*
2) 初期画像をアップロードします
3) ドロップダウンリストからStable DiffusionおよびControlNetモデルを選択します
4) 必要なパラメータに従ってモデルを設定します
5) リクエストを入力します (+ と - でプロンプトの重み付け)
6) `Submit`ボタンをクリックして生成された画像を取得します

#### upscale:

1) 初期画像をアップロードします
2) 必要なパラメータに従ってモデルを設定します
3) `Submit`ボタンをクリックしてアップスケールされた画像を取得します

#### inpaint:

1) まず、モデルをフォルダにアップロードします: *inputs/image/sd_models/inpaint*
2) ドロップダウンリストからモデルを選択します
3) モデルタイプを選択します (`SD`、`SD2`、`SDXL`)
4) 必要なパラメータに従ってモデルを設定します
5) 生成に使用する画像を`initial image`および`mask image`にアップロードします
6) `mask image`でブラシを選択し、パレットを選択して色を`#FFFFFF`に変更します
7) 生成する場所を描き、リクエストを入力します (+ と - でプロンプトの重み付け)
8) `Submit`ボタンをクリックしてインペイントされた画像を取得します
#### オプション: `vae`モデルを選択できます
#### vae = *inputs/image/sd_models/vae*

#### gligen:

1) まず、モデルをフォルダにアップロードします: *inputs/image/sd_models*
2) ドロップダウンリストからモデルを選択します
3) モデルタイプを選択します (`SD`、`SD2`、`SDXL`)
4) 必要なパラメータに従ってモデルを設定します
5) プロンプトのリクエストを入力します (+ と - でプロンプトの重み付け) およびGLIGENフレーズ (ボックス用に "" で囲む)
6) GLIGENボックスを入力します (ボックス用に [0.1387, 0.2051, 0.4277, 0.7090] のように)
7) `Submit`ボタンをクリックして生成された画像を取得します

#### animatediff:

1) まず、モデルをフォルダにアップロードします: *inputs/image/sd_models*
2) ドロップダウンリストからモデルを選択します
3) 必要なパラメータに従ってモデルを設定します
4) リクエストを入力します (+ と - でプロンプトの重み付け)
5) `Submit`ボタンをクリックして生成された画像アニメーションを取得します

#### video:

1) 初期画像をアップロードします
2) リクエストを入力します (IV2Gen-XL用)
3) 必要なパラメータに従ってモデルを設定します
4) `Submit`ボタンをクリックして画像からビデオを取得します

#### ldm3d:

1) リクエストを入力します
2) 必要なパラメータに従ってモデルを設定します
3) `Submit`ボタンをクリックして生成された画像を取得します

#### sd3:

1) リクエストを入力します
2) 必要なパラメータに従ってモデルを設定します
3) `Submit`ボタンをクリックして生成された画像を取得します

#### cascade:

1) リクエストを入力します
2) 必要なパラメータに従ってモデルを設定します
3) `Submit`ボタンをクリックして生成された画像を取得します

#### extras:

1) 初期画像をアップロードします
2) 必要なオプションを選択します
3) `Submit`ボタンをクリックして変更された画像を取得します

### Kandinsky:

1) プロンプトを入力します
2) ドロップダウンリストからモデルを選択します
3) 必要なパラメータに従ってモデルを設定します
4) `Submit`をクリックして生成された画像を取得します

### Flux:

1) プロンプトを入力します
2) ドロップダウンリストからモデルを選択します
3) 必要なパラメータに従ってモデルを設定します
4) `Submit`をクリックして生成された画像を取得します

### HunyuanDiT:

1) プロンプトを入力します
2) 必要なパラメータに従ってモデルを設定します
3) `Submit`をクリックして生成された画像を取得します

### Lumina-T2X:

1) プロンプトを入力します
2) 必要なパラメータに従ってモデルを設定します
3) `Submit`をクリックして生成された画像を取得します

### Kolors:

1) プロンプトを入力します
2) 必要なパラメータに従ってモデルを設定します
3) `Submit`をクリックして生成された画像を取得します

### AuraFlow:

1) プロンプトを入力します
2) 必要なパラメータに従ってモデルを設定します
3) `Submit`をクリックして生成された画像を取得します

### Würstchen:

1) プロンプトを入力します
2) 必要なパラメータに従ってモデルを設定します
3) `Submit`をクリックして生成された画像を取得します

### DeepFloydIF:

1) プロンプトを入力します
2) 必要なパラメータに従ってモデルを設定します
3) `Submit`をクリックして生成された画像を取得します

### PixArt:

1) プロンプトを入力します
2) ドロップダウンリストからモデルを選択します
3) 必要なパラメータに従ってモデルを設定します
4) `Submit`をクリックして生成された画像を取得します

### ModelScope:

1) プロンプトを入力します
2) 必要なパラメータに従ってモデルを設定します
3) `Submit`をクリックして生成されたビデオを取得します

### ZeroScope 2:

1) プロンプトを入力します
2) 必要なパラメータに従ってモデルを設定します
3) `Submit`をクリックして生成されたビデオを取得します

### CogVideoX:

1) プロンプトを入力します
2) 必要なパラメータに従ってモデルを設定します
3) `Submit`をクリックして生成されたビデオを取得します

### Latte:

1) プロンプトを入力します
2) 必要なパラメータに従ってモデルを設定します
3) `Submit`をクリックして生成されたビデオを取得します

### TripoSR:

1) 初期画像をアップロードします
2) 必要なパラメータに従ってモデルを設定します
3) `Submit`ボタンをクリックして生成された3Dオブジェクトを取得します

### StableFast3D:

1) 初期画像をアップロードします
2) 必要なパラメータに従ってモデルを設定します
3) `Submit`ボタンをクリックして生成された3Dオブジェクトを取得します

### Shap-E:

1) リクエストを入力するか、初期画像をアップロードします
2) 必要なパラメータに従ってモデルを設定します
3) `Submit`ボタンをクリックして生成された3Dオブジェクトを取得します

### SV34D:

1) 初期画像（3D用）またはビデオ（4D用）をアップロードします
2) 必要なパラメータに従ってモデルを設定します
3) `Submit`ボタンをクリックして生成された3Dビデオを取得します

### Zero123Plus:

1) 初期画像をアップロードします
2) 必要なパラメータに従ってモデルを設定します
3) `Submit`ボタンをクリックして生成された3D回転画像を取得します

### StableAudio:

1) 必要なパラメータに従ってモデルを設定します
2) リクエストを入力します
3) `Submit`ボタンをクリックして生成されたオーディオを取得します

### AudioCraft:

1) ドロップダウンリストからモデルを選択します
2) モデルタイプを選択します (`musicgen`、`audiogen`、`magnet`)
3) 必要なパラメータに従ってモデルを設定します
4) リクエストを入力します
5) （オプション）`melody`モデルを使用している場合は初期音声をアップロードします
6) `Submit`ボタンをクリックして生成されたオーディオを取得します
#### オプション: `multiband diffusion`を有効にして生成されたオーディオを改善できます

### AudioLDM 2:

1) ドロップダウンリストからモデルを選択します
2) 必要なパラメータに従ってモデルを設定します
3) リクエストを入力します
4) `Submit`ボタンをクリックして生成されたオーディオを取得します

### Demucs:

1) 分離する初期音声をアップロードします
2) `Submit`ボタンをクリックして分離されたオーディオを取得します

### ギャラリー:

* ここでoutputsディレクトリのファイルを表示できます

### ModelDownloader:

* ここで`LLM`および`StableDiffusion`モデルをダウンロードできます。ドロップダウンリストからモデルを選択し、`Submit`ボタンをクリックするだけです
#### `LLM`モデルはここにダウンロードされます: *inputs/text/llm_models*
#### `StableDiffusion`モデルはここにダウンロードされます: *inputs/image/sd_models*

### 設定:

* ここでアプリケーション設定を変更できます。現在のところ、`Share`モードを`True`または`False`に変更することしかできません

### システム:

* ここで`Submit`ボタンをクリックしてコンピュータのセンサーの指標を確認できます

### 追加情報:

1) すべての生成は*outputs*フォルダに保存されます
2) 選択をリセットするには、`Clear`ボタンをクリックできます
3) 生成プロセスを停止するには、`Stop generation`ボタンをクリックします
4) `Close terminal`ボタンを使用してアプリケーションを終了できます
5) `Outputs`ボタンをクリックして*outputs*フォルダを開くことができます

## モデル、音声、アバターの入手方法

* LLMモデルは[HuggingFace](https://huggingface.co/models)から取得するか、インターフェース内のModelDownloaderから取得できます
* StableDiffusion、vae、inpaint、embedding、loraモデルは[CivitAI](https://civitai.com/models)から取得するか、インターフェース内のModelDownloaderから取得できます
* StableAudioOpen、AudioCraft、AudioLDM 2、TTS、Whisper、Wav2Lip、SunoBark、MoonDream2、Upscale、GLIGEN、Depth、Pix2Pix、Controlnet、AnimateDiff、Videos、LDM3D、SD3、Cascade、Rembg、Roop、CodeFormer、TripoSR、StableFast3D、Shap-E、SV34D、Zero123Plus、Demucs、Kandinsky、Flux、HunyuanDiT、Lumina-T2X、Kolors、AuraFlow、Würstchen、DeepFloydIF、PixArt、ModelScope、ZeroScope 2、CogVideoX、Latte、Multiband diffusionモデルは使用時に*inputs*フォルダに自動的にダウンロードされます
* 音声はどこからでも取得できます。自分の音声を録音するか、インターネットから録音を取得します。または、プロジェクトに既に含まれているものを使用します。主なことは、事前に処理されていることです！

## Wiki

* https://github.com/Dartvauder/NeuroSandboxWebUI/wiki

## 開発者への感謝

#### これらのプロジェクトに感謝します。彼らのアプリケーション/ライブラリのおかげで、私は自分のアプリケーションを作成することができました:

まず、[PyCharm](https://www.jetbrains.com/pycharm/)および[GitHub](https://desktop.github.com)の開発者に感謝します。彼らのアプリケーションの助けを借りて、私はコードを作成し、共有することができました。

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

## サードパーティライセンス:

#### 多くのモデルには使用するための独自のライセンスがあります。使用する前に、それらを確認することをお勧めします:

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

## 寄付

### *私のプロジェクトが気に入った場合、寄付をしたい場合は、以下のオプションがあります。事前にありがとうございます！*

* CryptoWallet(BEP-20) - 0x3d86bdb5f50b92d0d7Eb44F1a833acC5e91aAEcA

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)
