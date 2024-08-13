## [特徴](/#Features) | [依存関係](/#Required-Dependencies) | [システム要件](/#Minimum-System-Requirements) | [インストール](/#How-to-install) | [使用方法](/#How-to-use) | [モデル](/#Where-can-I-get-models-voices-and-avatars) | [Wiki](/#Wiki) | [開発者への謝辞](/#Acknowledgment-to-developers) | [ライセンス](/#Third-Party-Licenses)

# ![main](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4ea0d891-8979-45ad-b052-626c41ae991a)
* 開発中！（アルファ版）
* [English](/README.md) | [عربي](/Readmes/README_AR.md) | [Deutsche](/Readmes/README_DE.md) | [Español](/Readmes/README_ES.md) | [Français](/Readmes/README_FR.md) | 日本語 | [Русский](/Readmes/README_RU.md) | [漢語](/Readmes/README_ZH.md) | [Português](/Readmes/README_PT.md) | [हिंदी](/Readmes/README_HI.md) | [Italiano](/Readmes/README_IT.md) | [韓國語](/Readmes/README_KO.md)

## 説明：

様々なニューラルネットワークモデルを使用するためのシンプルで便利なインターフェース。テキスト、音声、画像入力を使用してLLMやMoondream2とコミュニケーションを取ることができます。StableDiffusion、Kandinsky、Flux、HunyuanDiT、Lumina-T2X、Kolors、AuraFlow、Würstchen、DeepFloydIF、PixArtを使用して画像を生成し、ModelScope、ZeroScope 2、CogVideoX、Latteを使用して動画を生成し、TripoSR、StableFast3D、Shap-E、SV34D、Zero123Plusを使用して3Dオブジェクトを生成し、StableAudioOpen、AudioCraft、AudioLDM 2を使用して音楽とオーディオを生成し、CoquiTTSとSunoBarkを使用してテキスト読み上げを行い、OpenAI-Whisperを使用して音声をテキストに変換し、Wav2Lipを使用してリップシンクを行い、Roopを使用して顔のスワップを行い、Rembgを使用して背景を削除し、CodeFormerを使用して顔の修復を行い、LibreTranslateを使用してテキスト翻訳を行い、Demucsを使用してオーディオファイルの分離を行うことができます。また、outputsディレクトリのファイルをギャラリーで表示したり、LLMやStableDiffusionモデルをダウンロードしたり、インターフェース内でアプリケーション設定を変更したり、システムセンサーをチェックしたりすることもできます。

プロジェクトの目標は、ニューラルネットワークモデルを使用するための最も簡単なアプリケーションを作成することです。

### テキスト： <img width="1127" alt="1jp" src="https://github.com/user-attachments/assets/37e91e1b-0e1d-4085-8ae0-b3feaecb72e5">

### 画像： <img width="1120" alt="2jp" src="https://github.com/user-attachments/assets/d09e422b-f4e2-46cc-ba94-082e0c6ecebf">

### 動画： <img width="1121" alt="3jp" src="https://github.com/user-attachments/assets/ddfe00d4-eede-4bb3-9b6b-03eb0519a491">

### 3D： <img width="1120" alt="4jp" src="https://github.com/user-attachments/assets/f2f8522e-1e66-4bd1-afb1-2cabc11b64ef">

### オーディオ： <img width="1121" alt="5jp" src="https://github.com/user-attachments/assets/a77724c6-daa1-4677-9d45-71c9060f7266">

### インターフェース： <img width="1121" alt="6jp" src="https://github.com/user-attachments/assets/b01c9acd-a740-473a-8e21-5e7fd79298b2">

## 特徴：

* install.bat（Windows）またはinstall.sh（Linux）を介した簡単なインストール
* モバイルデバイスを介してローカルホスト（IPv4経由）またはオンライン上のどこでも（Share経由）アプリケーションを使用可能
* 柔軟で最適化されたインターフェース（Gradioによる）
* admin:adminによる認証（GradioAuth.txtファイルにログイン詳細を入力可能）
* 特定のモデルをダウンロードするために自分のHuggingFace-Tokenを追加可能（HF-Token.txtファイルにトークンを入力可能）
* TransformersとLlama.cppモデル（LLM）のサポート
* DiffusersとSafetensorsモデル（StableDiffusion）のサポート - txt2img、img2img、depth2img、pix2pix、controlnet、upscale、inpaint、gligen、animatediff、video、ldm3d、sd3、cascadeおよびextrasタブ
* 画像生成のための追加モデルのサポート：Kandinsky、Flux、HunyuanDiT、Lumina-T2X、Kolors、AuraFlow、Würstchen、DeepFloydIF、PixArt
* StableAudioOpenのサポート
* AudioCraftのサポート（モデル：musicgen、audiogen、magnet）
* AudioLDM 2のサポート（モデル：audio、music）
* TTSとWhisperモデルのサポート（LLMとTTS-STT用）
* Lora、Textual inversion（embedding）、Vae、Img2img、Depth、Pix2Pix、Controlnet、Upscale（latent）、Upscale（Real-ESRGAN）、Inpaint、GLIGEN、AnimateDiff、Videos、LDM3D、SD3、Cascade、Rembg、CodeFormer、Roopモデルのサポート（StableDiffusion用）
* Multiband Diffusionモデルのサポート（AudioCraft用）
* LibreTranslateのサポート（ローカルAPI）
* ModelScope、ZeroScope 2、CogVideoX、Latteを使用した動画生成のサポート
* SunoBarkのサポート
* Demucsのサポート
* TripoSR、StableFast3D、Shap-E、SV34D、Zero123Plusを使用した3D生成のサポート
* Wav2Lipのサポート
* LLM用のMultimodal（Moondream 2）、LORA（transformers）、WebSearch（GoogleSearch使用）のサポート
* インターフェース内のモデル設定
* ギャラリー
* ModelDownloader（LLMとStableDiffusion用）
* アプリケーション設定
* システムセンサーの表示機能

## 必要な依存関係：

* [Python](https://www.python.org/downloads/)（3.11）
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads)（12.4）と[cuDNN](https://developer.nvidia.com/cudnn-downloads)（9.1）
* [FFMPEG](https://ffmpeg.org/download.html)
- C++コンパイラ
  - Windows：[VisualStudio](https://visualstudio.microsoft.com/ru/)
  - Linux：[GCC](https://gcc.gnu.org/)

## 最小システム要件：

* システム：WindowsまたはLinux
* GPU：6GB以上またはCPU：8コア3.2GHz
* RAM：16GB以上
* ディスク容量：20GB以上
* モデルのダウンロードとインストールのためのインターネット接続

## インストール方法：

### Windows

1) 任意の場所に `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` を実行
2) `Install.bat` を実行し、インストールが完了するまで待機
3) インストール後、`Start.bat` を実行
4) ファイルバージョンを選択し、アプリケーションの起動を待機
5) これで生成を開始できます！

アップデートを取得するには、`Update.bat` を実行します。
ターミナルを通じて仮想環境で作業するには、`Venv.bat` を実行します。

### Linux

1) 任意の場所に `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` を実行
2) ターミナルで `./Install.sh` を実行し、すべての依存関係のインストールが完了するまで待機
3) インストール後、`./Start.sh` を実行
4) アプリケーションの起動を待機
5) これで生成を開始できます！

アップデートを取得するには、`./Update.sh` を実行します。
ターミナルを通じて仮想環境で作業するには、`./Venv.sh` を実行します。

## 使用方法：

#### インターフェースには6つのメインタブに30のタブがあります：LLM、TTS-STT、SunoBark、LibreTranslate、Wav2Lip、StableDiffusion、Kandinsky、Flux、HunyuanDiT、Lumina-T2X、Kolors、AuraFlow、Würstchen、DeepFloydIF、PixArt、ModelScope、ZeroScope 2、CogVideoX、Latte、TripoSR、StableFast3D、Shap-E、SV34D、Zero123Plus、StableAudio、AudioCraft、AudioLDM 2、Demucs、Gallery、ModelDownloader、Settings、System。必要なものを選択し、以下の指示に従ってください。

### LLM：

1) まず、モデルをフォルダ：*inputs/text/llm_models* にアップロードします
2) ドロップダウンリストからモデルを選択します
3) モデルタイプ（`transformers` または `llama`）を選択します
4) 必要なパラメータに応じてモデルを設定します
5) リクエストを入力（または話す）します
6) `Submit` ボタンをクリックして、生成されたテキストと音声応答を受け取ります
#### オプション：`TTS` モードを有効にし、音声応答を受け取るために必要な `voice` と `language` を選択できます。`multimodal` を有効にして画像をアップロードし、その説明を取得できます。インターネットアクセスのために `websearch` を有効にできます。翻訳を取得するために `libretranslate` を有効にできます。また、生成を改善するために `LORA` モデルを選択することもできます。
#### 音声サンプル = *inputs/audio/voices*
#### LORA = *inputs/text/llm_models/lora*
#### 音声は事前に処理されている必要があります（22050 kHz、モノラル、WAV）

### TTS-STT：

1) テキスト読み上げのためのテキストを入力します
2) 音声からテキストへの変換のための音声を入力します
3) `Submit` ボタンをクリックして、生成されたテキストと音声応答を受け取ります
#### 音声サンプル = *inputs/audio/voices*
#### 音声は事前に処理されている必要があります（22050 kHz、モノラル、WAV）

### SunoBark：

1) リクエストを入力します
2) 必要なパラメータに応じてモデルを設定します
3) `Submit` ボタンをクリックして、生成された音声応答を受け取ります

### LibreTranslate：

* まず、[LibreTranslate](https://github.com/LibreTranslate/LibreTranslate) をインストールして実行する必要があります
1) ソース言語とターゲット言語を選択します
2) `Submit` ボタンをクリックして翻訳を取得します
#### オプション：対応するボタンをオンにすることで、翻訳履歴を保存できます

### Wav2Lip：

1) 顔の初期画像をアップロードします
2) 音声の初期音声をアップロードします
3) 必要なパラメータに応じてモデルを設定します
4) `Submit` ボタンをクリックして、リップシンクを受け取ります

### StableDiffusion - 15のサブタブがあります：

#### txt2img：

1) まず、モデルをフォルダ：*inputs/image/sd_models* にアップロードします
2) ドロップダウンリストからモデルを選択します
3) モデルタイプ（`SD`、`SD2`、または `SDXL`）を選択します
4) 必要なパラメータに応じてモデルを設定します
5) リクエストを入力します（+と-でプロンプトの重みづけ）
6) `Submit` ボタンをクリックして、生成された画像を取得します
#### オプション：生成方法を改善するために `vae`、`embedding`、`lora` モデルを選択できます。また、生成された画像のサイズを拡大するために `upscale` を有効にすることもできます。
#### vae = *inputs/image/sd_models/vae*
#### lora = *inputs/image/sd_models/lora*
#### embedding = *inputs/image/sd_models/embedding*

#### img2img：

1) まず、モデルをフォルダ：*inputs/image/sd_models* にアップロードします
2) ドロップダウンリストからモデルを選択します
3) モデルタイプ（`SD`、`SD2`、または `SDXL`）を選択します
4) 必要なパラメータに応じてモデルを設定します
5) 生成の基となる初期画像をアップロードします
6) リクエストを入力します（+と-でプロンプトの重みづけ）
7) `Submit` ボタンをクリックして、生成された画像を取得します
#### オプション：`vae` モデルを選択できます
#### vae = *inputs/image/sd_models/vae*

#### depth2img：

1) 初期画像をアップロードします
2) 必要なパラメータに応じてモデルを設定します
3) リクエストを入力します（+と-でプロンプトの重みづけ）
4) `Submit` ボタンをクリックして、生成された画像を取得します

#### pix2pix：

1) 初期画像をアップロードします
2) 必要なパラメータに応じてモデルを設定します
3) リクエストを入力します（+と-でプロンプトの重みづけ）
4) `Submit` ボタンをクリックして、生成された画像を取得します

#### controlnet：

1) まず、stable diffusionモデルをフォルダ：*inputs/image/sd_models* にアップロードします
2) 初期画像をアップロードします
3) ドロップダウンリストからstable diffusionモデルとcontrolnetモデルを選択します
4) 必要なパラメータに応じてモデルを設定します
5) リクエストを入力します（+と-でプロンプトの重みづけ）
6) `Submit` ボタンをクリックして、生成された画像を取得します

#### upscale(latent)：

1) 初期画像をアップロードします
2) 必要なパラメータに応じてモデルを設定します
3) `Submit` ボタンをクリックして、アップスケールされた画像を取得します

#### upscale(Real-ESRGAN)：

1) 初期画像をアップロードします
2) 必要なパラメータに応じてモデルを設定します
3) `Submit` ボタンをクリックして、アップスケールされた画像を取得します

#### inpaint：

1) まず、モデルをフォルダ：*inputs/image/sd_models/inpaint* にアップロードします
2) ドロップダウンリストからモデルを選択します
3) モデルタイプ（`SD`、`SD2`、または `SDXL`）を選択します
4) 必要なパラメータに応じてモデルを設定します
5) 生成の基となる画像を `initial image` と `mask image` にアップロードします
6) `mask image` で、ブラシを選択し、次にパレットを選択して色を `#FFFFFF` に変更します
7) 生成する場所を描画し、リクエストを入力します（+と-でプロンプトの重みづけ）
8) `Submit` ボタンをクリックして、インペイントされた画像を取得します
#### オプション：`vae` モデルを選択できます
#### vae = *inputs/image/sd_models/vae*

#### gligen：

1) まず、モデルをフォルダ：*inputs/image/sd_models* にアップロードします
2) ドロップダウンリストからモデルを選択します
3) モデルタイプ（`SD`、`SD2`、または `SDXL`）を選択します
4) 必要なパラメータに応じてモデルを設定します
5) プロンプト（+と-でプロンプトの重みづけ）とGLIGENフレーズ（ボックスの場合は ""で囲む）のリクエストを入力します
6) GLIGENボックスを入力します（ボックスの場合は [0.1387, 0.2051, 0.4277, 0.7090] のように）
7) `Submit` ボタンをクリックして、生成された画像を取得します

#### animatediff：

1) まず、モデルをフォルダ：*inputs/image/sd_models* にアップロードします
2) ドロップダウンリストからモデルを選択します
3) 必要なパラメータに応じてモデルを設定します
4) リクエストを入力します（+と-でプロンプトの重みづけ）
5) `Submit` ボタンをクリックして、生成された画像アニメーションを取得します

#### video：

1) 初期画像をアップロードします
2) リクエストを入力します（IV2Gen-XL用）
3) 必要なパラメータに応じてモデルを設定します
4) `Submit` ボタンをクリックして、画像から生成された動画を取得します

#### ldm3d：

1) リクエストを入力します
2) 必要なパラメータに応じてモデルを設定します
3) `Submit` ボタンをクリックして、生成された画像を取得します

#### sd3：

1) リクエストを入力します
2) 必要なパラメータに応じてモデルを設定します
3) `Submit` ボタンをクリックして、生成された画像を取得します

#### cascade：

1) リクエストを入力します
2) 必要なパラメータに応じてモデルを設定します
3) `Submit` ボタンをクリックして、生成された画像を取得します

#### extras：

1) 初期画像をアップロードします
2) 必要なオプションを選択します
3) `Submit` ボタンをクリックして、修正された画像を取得します

### Kandinsky：

1) プロンプトを入力します
2) ドロップダウンリストからモデルを選択します
3) 必要なパラメータに応じてモデルを設定します
4) `Submit` をクリックして、生成された画像を取得します

### Flux：

1) プロンプトを入力します
2) ドロップダウンリストからモデルを選択します
3) 必要なパラメータに応じてモデルを設定します
4) `Submit` をクリックして、生成された画像を取得します

### HunyuanDiT：

1) プロンプトを入力します
2) 必要なパラメータに応じてモデルを設定します
3) `Submit` をクリックして、生成された画像を取得します

### Lumina-T2X：

1) プロンプトを入力します
2) 必要なパラメータに応じてモデルを設定します
3) `Submit` をクリックして、生成された画像を取得します

### Kolors：

1) プロンプトを入力します
2) 必要なパラメータに応じてモデルを設定します
3) `Submit` をクリックして、生成された画像を取得します

### AuraFlow：

1) プロンプトを入力します
2) 必要なパラメータに応じてモデルを設定します
3) `Submit` をクリックして、生成された画像を取得します

### Würstchen：

1) プロンプトを入力します
2) 必要なパラメータに応じてモデルを設定します
3) `Submit` をクリックして、生成された画像を取得します

### DeepFloydIF：

1) プロンプトを入力します
2) 必要なパラメータに応じてモデルを設定します
3) `Submit` をクリックして、生成された画像を取得します

### PixArt：

1) プロンプトを入力します
2) ドロップダウンリストからモデルを選択します
3) 必要なパラメータに応じてモデルを設定します
4) `Submit` をクリックして、生成された画像を取得します

### ModelScope：

1) プロンプトを入力します
2) 必要なパラメータに応じてモデルを設定します
3) `Submit` をクリックして、生成された動画を取得します

### ZeroScope 2：

1) プロンプトを入力します
2) 必要なパラメータに応じてモデルを設定します
3) `Submit` をクリックして、生成された動画を取得します

### CogVideoX：

1) プロンプトを入力します
2) 必要なパラメータに応じてモデルを設定します
3) `Submit` をクリックして、生成された動画を取得します

### Latte：

1) プロンプトを入力します
2) 必要なパラメータに応じてモデルを設定します
3) `Submit` をクリックして、生成された動画を取得します

### TripoSR：

1) 初期画像をアップロードします
2) 必要なパラメータに応じてモデルを設定します
3) `Submit` ボタンをクリックして、生成された3Dオブジェクトを取得します

### StableFast3D：

1) 初期画像をアップロードします
2) 必要なパラメータに応じてモデルを設定します
3) `Submit` ボタンをクリックして、生成された3Dオブジェクトを取得します

### Shap-E：

1) リクエストを入力するか、初期画像をアップロードします
2) 必要なパラメータに応じてモデルを設定します
3) `Submit` ボタンをクリックして、生成された3Dオブジェクトを取得します

### SV34D：

1) 初期画像（3D用）または動画（4D用）をアップロードします
2) 必要なパラメータに応じてモデルを設定します
3) `Submit` ボタンをクリックして、生成された3D動画を取得します

### Zero123Plus：

1) 初期画像をアップロードします
2) 必要なパラメータに応じてモデルを設定します
3) `Submit` ボタンをクリックして、生成された画像の3D回転を取得します

### StableAudio：

1) 必要なパラメータに応じてモデルを設定します
2) リクエストを入力します
3) `Submit` ボタンをクリックして、生成されたオーディオを取得します

### AudioCraft：

1) ドロップダウンリストからモデルを選択します
2) モデルタイプ（`musicgen`、`audiogen`、または `magnet`）を選択します
3) 必要なパラメータに応じてモデルを設定します
4) リクエストを入力します
5) （オプション）`melody` モデルを使用している場合は、初期オーディオをアップロードします
6) `Submit` ボタンをクリックして、生成されたオーディオを取得します
#### オプション：生成されたオーディオを改善するために `multiband diffusion` を有効にできます

### AudioLDM 2：

1) ドロップダウンリストからモデルを選択します
2) 必要なパラメータに応じてモデルを設定します
3) リクエストを入力します
4) `Submit` ボタンをクリックして、生成されたオーディオを取得します

### Demucs：

1) 分離する初期オーディオをアップロードします
2) `Submit` ボタンをクリックして、分離されたオーディオを取得します

### Gallery：

* ここでoutputsディレクトリのファイルを表示できます

### ModelDownloader：

* ここで `LLM` と `StableDiffusion` モデルをダウンロードできます。ドロップダウンリストからモデルを選択し、`Submit` ボタンをクリックするだけです
#### `LLM` モデルはここにダウンロードされます：*inputs/text/llm_models*
#### `StableDiffusion` モデルはここにダウンロードされます：*inputs/image/sd_models*

### Settings：

* ここでアプリケーション設定を変更できます。現在は `Share` モードを `True` または `False` に変更することしかできません

### System：

* ここで `Submit` ボタンをクリックすることで、コンピューターのセンサーの指標を確認できます

### 追加情報：

1) すべての生成物は *outputs* フォルダに保存されます
2) `Clear` ボタンを押して選択をリセットできます
3) 生成プロセスを停止するには、`Stop generation` ボタンをクリックします
4) `Close terminal` ボタンを使用してアプリケーションをオフにできます
5) `Outputs` ボタンをクリックして *outputs* フォルダを開くことができます

## モデルと音声はどこで入手できますか？

* LLMモデルは [HuggingFace](https://huggingface.co/models) またはインターフェース内のModelDownloaderから入手できます
* StableDiffusion、vae、inpaint、embedding、loraモデルは [CivitAI](https://civitai.com/models) またはインターフェース内のModelDownloaderから入手できます
* StableAudioOpen、AudioCraft、AudioLDM 2、TTS、Whisper、Wav2Lip、SunoBark、MoonDream2、Upscale、GLIGEN、Depth、Pix2Pix、Controlnet、AnimateDiff、Videos、LDM3D、SD3、Cascade、Rembg、Roop、CodeFormer、Real-ESRGAN、TripoSR、StableFast3D、Shap-E、SV34D、Zero123Plus、Demucs、Kandinsky、Flux、HunyuanDiT、Lumina-T2X、Kolors、AuraFlow、Würstchen、DeepFloydIF、PixArt、ModelScope、ZeroScope 2、CogVideoX、Latteおよびmultiband diffusionモデルは、使用時に *inputs* フォルダに自動的にダウンロードされます
* 音声はどこからでも入手できます。自分で録音するか、インターネットから録音を入手してください。または、プロジェクトにすでにあるものを使用してください。重要なのは、事前に処理されていることです！

## Wiki

* https://github.com/Dartvauder/NeuroSandboxWebUI/wiki

## 開発者への謝辞

#### これらのプロジェクトに多大な感謝を捧げます。彼らのアプリケーション/ライブラリのおかげで、私はこのアプリケーションを作成することができました：

まず、[PyCharm](https://www.jetbrains.com/pycharm/) と [GitHub](https://desktop.github.com) の開発者に感謝します。彼らのアプリケーションのおかげで、私はコードを作成し共有することができました。

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

## サードパーティライセンス：

#### 多くのモデルには独自の使用ライセンスがあります。使用する前に、以下のライセンスをよくお読みください：

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

## 寄付

### *私のプロジェクトが気に入り、寄付したい場合は、以下のオプションがあります。事前に大変感謝いたします！*

* CryptoWallet(BEP-20) - 0x3d86bdb5f50b92d0d7Eb44F1a833acC5e91aAEcA

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Dartvauder/NeuroSandboxWebUI&type=Date)](https://star-history.com/#Dartvauder/NeuroSandboxWebUI&Date)