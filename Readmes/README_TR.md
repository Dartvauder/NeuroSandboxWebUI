## [Özellikler](/#Özellikler) | [Bağımlılıklar](/#Gerekli-Bağımlılıklar) | [SistemGereksinimleri](/#Minimum-Sistem-Gereksinimleri) | [Kurulum](/#Nasıl-kurulur) | [Kullanım](/#Nasıl-kullanılır) | [Modeller](/#Modelleri-sesleri-ve-avatarları-nereden-alabilirim) | [Wiki](/#Wiki) | [Teşekkür](/#Geliştiricilere-teşekkür) | [Lisanslar](/#Üçüncü-Taraf-Lisansları)

# ![main](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4ea0d891-8979-45ad-b052-626c41ae991a)
* Üzerinde çalışılıyor! (ALFA)
* [English](/README.md) | [عربي](/Readmes/README_AR.md) | [Deutsche](/Readmes/README_DE.md) | [Español](/Readmes/README_ES.md) | [Français](/Readmes/README_FR.md) | [日本語](/Readmes/README_JP.md) | [Русский](/Readmes/README_RU.md) | [漢語](/Readmes/README_ZH.md) | [Português](/Readmes/README_PT.md) | [हिंदी](/Readmes/README_HI.md) | [Italiano](/Readmes/README_IT.md) | [韓國語](/Readmes/README_KO.md) | [Polski](/Readmes/README_PL.md) | Türkçe

## Açıklama:

Çeşitli sinir ağı modellerini kullanmak için basit ve kullanışlı bir arayüz. LLM ve Moondream2 ile metin, ses ve görüntü girişi kullanarak iletişim kurabilirsiniz; StableDiffusion, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF ve PixArt'ı görüntü oluşturmak için kullanabilirsiniz; ModelScope, ZeroScope 2, CogVideoX ve Latte'yi video oluşturmak için kullanabilirsiniz; TripoSR, StableFast3D, Shap-E, SV34D ve Zero123Plus'ı 3D nesneler oluşturmak için kullanabilirsiniz; StableAudioOpen, AudioCraft ve AudioLDM 2'yi müzik ve ses oluşturmak için kullanabilirsiniz; metin-konuşma için CoquiTTS ve SunoBark; konuşma-metin için OpenAI-Whisper; dudak senkronizasyonu için Wav2Lip; yüz değiştirme için Roop; arka planı kaldırmak için Rembg; yüz düzeltme için CodeFormer; metin çevirisi için LibreTranslate; ses dosyası ayırma için Demucs. Ayrıca çıktılar dizinindeki dosyaları galeride görüntüleyebilir, LLM ve StableDiffusion modellerini indirebilir, uygulama ayarlarını arayüz içinde değiştirebilir ve sistem sensörlerini kontrol edebilirsiniz.

Projenin amacı - sinir ağı modellerini kullanmak için mümkün olan en kolay uygulamayı oluşturmak

### Metin: <img width="1118" alt="1tr" src="https://github.com/user-attachments/assets/5d991beb-cac2-4996-acb5-9343547a0a90">

### Görüntü: <img width="1123" alt="2tr" src="https://github.com/user-attachments/assets/35b30df1-d7aa-46d7-af12-bc101bcdea73">

### Video: <img width="1120" alt="3tr" src="https://github.com/user-attachments/assets/ba3ca3e3-0b96-400d-820b-8c71142749e3">

### 3D: <img width="1114" alt="4tr" src="https://github.com/user-attachments/assets/73c8dbef-0c53-417a-b26b-4794bfb42da3">

### Ses: <img width="1115" alt="5tr" src="https://github.com/user-attachments/assets/f28b51e4-0f6b-48f3-8ebf-f53b90e6ef4f">

### Arayüz: <img width="1120" alt="6tr" src="https://github.com/user-attachments/assets/9d64c9d6-5446-4525-95f5-540282390247">

## Özellikler:

* install.bat(Windows) veya install.sh(Linux) aracılığıyla kolay kurulum
* Uygulamayı localhost'ta (IPv4 üzerinden) veya çevrimiçi herhangi bir yerde (Share üzerinden) mobil cihazınız üzerinden kullanabilirsiniz
* Esnek ve optimize edilmiş arayüz (Gradio tarafından)
* admin:admin aracılığıyla kimlik doğrulama (Giriş bilgilerinizi GradioAuth.txt dosyasına girebilirsiniz)
* Belirli modelleri indirmek için kendi HuggingFace-Token'ınızı ekleyebilirsiniz (Token'ınızı HF-Token.txt dosyasına girebilirsiniz)
* Transformers ve llama.cpp modellerini destekler (LLM)
* Diffusers ve safetensors modellerini destekler (StableDiffusion) - txt2img, img2img, depth2img, pix2pix, controlnet, upscale, inpaint, gligen, animatediff, video, ldm3d, sd3, cascade, adapters ve extras sekmeleri
* Görüntü oluşturma için ek modelleri destekler: Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF ve PixArt
* StableAudioOpen desteği
* AudioCraft desteği (Modeller: musicgen, audiogen ve magnet)
* AudioLDM 2 desteği (Modeller: audio ve music)
* TTS ve Whisper modellerini destekler (LLM ve TTS-STT için)
* Lora, Textual inversion (embedding), Vae, Img2img, Depth, Pix2Pix, Controlnet, Upscale(latent), Upscale(Real-ESRGAN), Inpaint, GLIGEN, AnimateDiff, Videos, LDM3D, SD3, Cascade, Adapters (InstantID, PhotoMaker, IP-Adapter-FaceID), Rembg, CodeFormer ve Roop modellerini destekler (StableDiffusion için)
* Multiband Diffusion modelini destekler (AudioCraft için)
* LibreTranslate desteği (Yerel API)
* Video oluşturma için ModelScope, ZeroScope 2, CogVideoX ve Latte desteği
* SunoBark desteği
* Demucs desteği
* 3D oluşturma için TripoSR, StableFast3D, Shap-E, SV34D ve Zero123Plus desteği
* Wav2Lip desteği
* LLM için Multimodal (Moondream 2), LORA (transformers) ve WebSearch (GoogleSearch ile) desteği
* Arayüz içinde model ayarları
* Galeri
* ModelDownloader (LLM ve StableDiffusion için)
* Uygulama ayarları
* Sistem sensörlerini görme yeteneği

## Gerekli Bağımlılıklar:

* [Python](https://www.python.org/downloads/) (3.11)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (12.4) ve [cuDNN](https://developer.nvidia.com/cudnn-downloads) (9.1)
* [FFMPEG](https://ffmpeg.org/download.html)
- C+ derleyici
  - Windows: [VisualStudio](https://visualstudio.microsoft.com/ru/)
  - Linux: [GCC](https://gcc.gnu.org/)

## Minimum Sistem Gereksinimleri:

* Sistem: Windows veya Linux
* GPU: 6GB+ veya CPU: 8 çekirdek 3.2GHZ
* RAM: 16GB+
* Disk alanı: 20GB+
* Modelleri indirmek ve kurmak için internet

## Nasıl kurulur:

### Windows

1) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` komutunu herhangi bir konuma çalıştırın
2) `Install.bat` dosyasını çalıştırın ve kurulumun tamamlanmasını bekleyin
3) Kurulumdan sonra `Start.bat` dosyasını çalıştırın
4) Dosya sürümünü seçin ve uygulamanın başlamasını bekleyin
5) Artık oluşturmaya başlayabilirsiniz!

Güncelleme almak için `Update.bat` dosyasını çalıştırın
Terminal üzerinden sanal ortamla çalışmak için `Venv.bat` dosyasını çalıştırın

### Linux

1) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` komutunu herhangi bir konuma çalıştırın
2) Terminalde `./Install.sh` dosyasını çalıştırın ve tüm bağımlılıkların kurulumunu bekleyin
3) Kurulumdan sonra `./Start.sh` dosyasını çalıştırın
4) Uygulamanın başlamasını bekleyin
5) Artık oluşturmaya başlayabilirsiniz!

Güncelleme almak için `./Update.sh` dosyasını çalıştırın
Terminal üzerinden sanal ortamla çalışmak için `./Venv.sh` dosyasını çalıştırın

## Nasıl kullanılır:

#### Arayüzde altı ana sekmede otuz iki sekme bulunmaktadır (Metin, Görüntü, Video, 3D, Ses ve Arayüz): LLM, TTS-STT, SunoBark, LibreTranslate, Wav2Lip, StableDiffusion, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF, PixArt, ModelScope, ZeroScope 2, CogVideoX, Latte, TripoSR, StableFast3D, Shap-E, SV34D, Zero123Plus, StableAudio, AudioCraft, AudioLDM 2, Demucs, Galeri, ModelDownloader, Ayarlar ve Sistem. İhtiyacınız olanı seçin ve aşağıdaki talimatları izleyin 

### LLM:

1) İlk olarak modellerinizi şu klasöre yükleyin: *inputs/text/llm_models*
2) Açılır listeden modelinizi seçin
3) Model türünü seçin (`transformers` veya `llama`)
4) Modeli ihtiyacınız olan parametrelere göre ayarlayın
5) İsteğinizi yazın (veya söyleyin)
6) Oluşturulan metin ve ses yanıtını almak için `Submit` düğmesine tıklayın
#### İsteğe bağlı: ses yanıtı almak için `TTS` modunu etkinleştirebilir, gerekli `ses` ve `dil`i seçebilirsiniz. Açıklamasını almak için `multimodal`ı etkinleştirebilir ve bir görüntü yükleyebilirsiniz. İnternet erişimi için `websearch`ı etkinleştirebilirsiniz. Çeviri almak için `libretranslate`ı etkinleştirebilirsiniz. Ayrıca üretimi iyileştirmek için `LORA` modelini seçebilirsiniz
#### Ses örnekleri = *inputs/audio/voices*
#### LORA = *inputs/text/llm_models/lora*
#### Ses önceden işlenmiş olmalıdır (22050 kHz, mono, WAV)

### TTS-STT:

1) Metinden konuşmaya için metin girin
2) Konuşmadan metne için ses girin
3) Oluşturulan metin ve ses yanıtını almak için `Submit` düğmesine tıklayın
#### Ses örnekleri = *inputs/audio/voices*
#### Ses önceden işlenmiş olmalıdır (22050 kHz, mono, WAV)

### SunoBark:

1) İsteğinizi yazın
2) Modeli ihtiyacınız olan parametrelere göre ayarlayın
3) Oluşturulan ses yanıtını almak için `Submit` düğmesine tıklayın

### LibreTranslate:

* Öncelikle [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate)'ı kurmanız ve çalıştırmanız gerekiyor
1) Kaynak ve hedef dilleri seçin
2) Çeviriyi almak için `Submit` düğmesine tıklayın
#### İsteğe bağlı: ilgili düğmeyi açarak çeviri geçmişini kaydedebilirsiniz

### Wav2Lip:

1) Yüzün başlangıç görüntüsünü yükleyin
2) Sesin başlangıç ses dosyasını yükleyin
3) Modeli ihtiyacınız olan parametrelere göre ayarlayın
4) Dudak senkronizasyonunu almak için `Submit` düğmesine tıklayın

### StableDiffusion - on altı alt sekmesi vardır:

#### txt2img:

1) İlk olarak modellerinizi şu klasöre yükleyin: *inputs/image/sd_models*
2) Açılır listeden modelinizi seçin
3) Model türünü seçin (`SD`, `SD2` veya `SDXL`)
4) Modeli ihtiyacınız olan parametrelere göre ayarlayın
5) İsteğinizi girin (prompt ağırlığı için + ve -)
6) Oluşturulan görüntüyü almak için `Submit` düğmesine tıklayın
#### İsteğe bağlı: Oluşturma yöntemini geliştirmek için `vae`, `embedding` ve `lora` modellerinizi seçebilirsiniz, ayrıca oluşturulan görüntünün boyutunu artırmak için `upscale`ı etkinleştirebilirsiniz 
#### vae = *inputs/image/sd_models/vae*
#### lora = *inputs/image/sd_models/lora*
#### embedding = *inputs/image/sd_models/embedding*

#### img2img:

1) İlk olarak modellerinizi şu klasöre yükleyin: *inputs/image/sd_models*
2) Açılır listeden modelinizi seçin
3) Model türünü seçin (`SD`, `SD2` veya `SDXL`)
4) Modeli ihtiyacınız olan parametrelere göre ayarlayın
5) Oluşturmanın gerçekleşeceği başlangıç görüntüsünü yükleyin
6) İsteğinizi girin (prompt ağırlığı için + ve -)
7) Oluşturulan görüntüyü almak için `Submit` düğmesine tıklayın
#### İsteğe bağlı: `vae` modelinizi seçebilirsiniz
#### vae = *inputs/image/sd_models/vae*

#### depth2img:

1) Başlangıç görüntüsünü yükleyin
2) Modeli ihtiyacınız olan parametrelere göre ayarlayın
3) İsteğinizi girin (prompt ağırlığı için + ve -)
4) Oluşturulan görüntüyü almak için `Submit` düğmesine tıklayın

#### pix2pix:

1) Başlangıç görüntüsünü yükleyin
2) Modeli ihtiyacınız olan parametrelere göre ayarlayın
3) İsteğinizi girin (prompt ağırlığı için + ve -)
4) Oluşturulan görüntüyü almak için `Submit` düğmesine tıklayın

#### controlnet:

1) İlk olarak stable diffusion modellerinizi şu klasöre yükleyin: *inputs/image/sd_models*
2) Başlangıç görüntüsünü yükleyin
3) Açılır listelerden stable diffusion ve controlnet modellerinizi seçin
4) Modelleri ihtiyacınız olan parametrelere göre ayarlayın
5) İsteğinizi girin (prompt ağırlığı için + ve -)
6) Oluşturulan görüntüyü almak için `Submit` düğmesine tıklayın

#### upscale(latent):

1) Başlangıç görüntüsünü yükleyin
2) Modeli ihtiyacınız olan parametrelere göre ayarlayın
3) Ölçeği büyütülmüş görüntüyü almak için `Submit` düğmesine tıklayın

#### upscale(Real-ESRGAN):

1) Başlangıç görüntüsünü yükleyin
2) Modeli ihtiyacınız olan parametrelere göre ayarlayın
3) Ölçeği büyütülmüş görüntüyü almak için `Submit` düğmesine tıklayın

#### inpaint:

1) İlk olarak modellerinizi şu klasöre yükleyin: *inputs/image/sd_models/inpaint*
2) Açılır listeden modelinizi seçin
3) Model türünü seçin (`SD`, `SD2` veya `SDXL`)
4) Modeli ihtiyacınız olan parametrelere göre ayarlayın
5) Oluşturmanın gerçekleşeceği görüntüyü `initial image` ve `mask image` alanlarına yükleyin
6) `mask image` alanında, fırçayı seçin, ardından paleti seçin ve rengi `#FFFFFF` olarak değiştirin
7) Oluşturma için bir yer çizin ve isteğinizi girin (prompt ağırlığı için + ve -)
8) İnpaint edilmiş görüntüyü almak için `Submit` düğmesine tıklayın
#### İsteğe bağlı: `vae` modelinizi seçebilirsiniz
#### vae = *inputs/image/sd_models/vae*

#### gligen:

1) İlk olarak modellerinizi şu klasöre yükleyin: *inputs/image/sd_models*
2) Açılır listeden modelinizi seçin
3) Model türünü seçin (`SD`, `SD2` veya `SDXL`)
4) Modeli ihtiyacınız olan parametrelere göre ayarlayın
5) Prompt için isteğinizi (prompt ağırlığı için + ve -) ve GLIGEN cümlelerini girin (kutu için "" içinde)
6) GLIGEN kutularını girin (Kutu için [0.1387, 0.2051, 0.4277, 0.7090] gibi)
7) Oluşturulan görüntüyü almak için `Submit` düğmesine tıklayın

#### animatediff:

1) İlk olarak modellerinizi şu klasöre yükleyin: *inputs/image/sd_models*
2) Açılır listeden modelinizi seçin
3) Modeli ihtiyacınız olan parametrelere göre ayarlayın
4) İsteğinizi girin (prompt ağırlığı için + ve -)
5) Oluşturulan görüntü animasyonunu almak için `Submit` düğmesine tıklayın

#### video:

1) Başlangıç görüntüsünü yükleyin
2) İsteğinizi girin (IV2Gen-XL için)
3) Modeli ihtiyacınız olan parametrelere göre ayarlayın
4) Görüntüden video elde etmek için `Submit` düğmesine tıklayın

#### ldm3d:

1) İsteğinizi girin
2) Modeli ihtiyacınız olan parametrelere göre ayarlayın
3) Oluşturulan görüntüleri almak için `Submit` düğmesine tıklayın

#### sd3 (txt2img, img2img, controlnet, inpaint):

1) İsteğinizi girin
2) Modeli ihtiyacınız olan parametrelere göre ayarlayın
3) Oluşturulan görüntüyü almak için `Submit` düğmesine tıklayın

#### cascade:

1) İsteğinizi girin
2) Modeli ihtiyacınız olan parametrelere göre ayarlayın
3) Oluşturulan görüntüyü almak için `Submit` düğmesine tıklayın

#### adapters (InstantID, PhotoMaker ve IP-Adapter-FaceID):

1) İlk olarak modellerinizi şu klasöre yükleyin: *inputs/image/sd_models*
2) Başlangıç görüntüsünü yükleyin
3) Açılır listeden modelinizi seçin
4) Modeli ihtiyacınız olan parametrelere göre ayarlayın
5) İhtiyacınız olan alt sekmeyi seçin
6) Değiştirilmiş görüntüyü almak için `Submit` düğmesine tıklayın

#### extras:

1) Başlangıç görüntüsünü yükleyin
2) İhtiyacınız olan seçenekleri belirleyin
3) Değiştirilmiş görüntüyü almak için `Submit` düğmesine tıklayın

### Kandinsky (txt2img, img2img, inpaint):

1) İstemcinizi girin
2) Açılır listeden bir model seçin
3) Modeli ihtiyacınız olan parametrelere göre ayarlayın
4) Oluşturulan görüntüyü almak için `Submit`e tıklayın

### Flux:

1) İstemcinizi girin
2) Açılır listeden bir model seçin
3) Modeli ihtiyacınız olan parametrelere göre ayarlayın
4) Oluşturulan görüntüyü almak için `Submit`e tıklayın

### HunyuanDiT:

1) İstemcinizi girin
2) Modeli ihtiyacınız olan parametrelere göre ayarlayın
3) Oluşturulan görüntüyü almak için `Submit`e tıklayın

### Lumina-T2X:

1) İstemcinizi girin
2) Modeli ihtiyacınız olan parametrelere göre ayarlayın
3) Oluşturulan görüntüyü almak için `Submit`e tıklayın

### Kolors:

1) İstemcinizi girin
2) Modeli ihtiyacınız olan parametrelere göre ayarlayın
3) Oluşturulan görüntüyü almak için `Submit`e tıklayın

### AuraFlow:

1) İstemcinizi girin
2) Modeli ihtiyacınız olan parametrelere göre ayarlayın
3) Oluşturulan görüntüyü almak için `Submit`e tıklayın

### Würstchen:

1) İstemcinizi girin
2) Modeli ihtiyacınız olan parametrelere göre ayarlayın
3) Oluşturulan görüntüyü almak için `Submit`e tıklayın

### DeepFloydIF (txt2img, img2img, inpaint):

1) İstemcinizi girin
2) Modeli ihtiyacınız olan parametrelere göre ayarlayın
3) Oluşturulan görüntüyü almak için `Submit`e tıklayın

### PixArt:

1) İstemcinizi girin
2) Açılır listeden modeli seçin
3) Modeli ihtiyacınız olan parametrelere göre ayarlayın
4) Oluşturulan görüntüyü almak için `Submit`e tıklayın

### ModelScope:

1) İstemcinizi girin
2) Modeli ihtiyacınız olan parametrelere göre ayarlayın
3) Oluşturulan videoyu almak için `Submit`e tıklayın

### ZeroScope 2:

1) İstemcinizi girin
2) Modeli ihtiyacınız olan parametrelere göre ayarlayın
3) Oluşturulan videoyu almak için `Submit`e tıklayın

### CogVideoX:

1) İstemcinizi girin
2) Modeli ihtiyacınız olan parametrelere göre ayarlayın
3) Oluşturulan videoyu almak için `Submit`e tıklayın

### Latte:

1) İstemcinizi girin
2) Modeli ihtiyacınız olan parametrelere göre ayarlayın
3) Oluşturulan videoyu almak için `Submit`e tıklayın

### TripoSR:

1) Başlangıç görüntüsünü yükleyin
2) Modeli ihtiyacınız olan parametrelere göre ayarlayın
3) Oluşturulan 3D nesneyi almak için `Submit` düğmesine tıklayın

### StableFast3D:

1) Başlangıç görüntüsünü yükleyin
2) Modeli ihtiyacınız olan parametrelere göre ayarlayın
3) Oluşturulan 3D nesneyi almak için `Submit` düğmesine tıklayın

### Shap-E:

1) İsteğinizi girin veya başlangıç görüntüsünü yükleyin
2) Modeli ihtiyacınız olan parametrelere göre ayarlayın
3) Oluşturulan 3D nesneyi almak için `Submit` düğmesine tıklayın

### SV34D:

1) Başlangıç görüntüsünü (3D için) veya videoyu (4D için) yükleyin
2) Modeli ihtiyacınız olan parametrelere göre ayarlayın
3) Oluşturulan 3D videoyu almak için `Submit` düğmesine tıklayın

### Zero123Plus:

1) Başlangıç görüntüsünü yükleyin
2) Modeli ihtiyacınız olan parametrelere göre ayarlayın
3) Görüntünün oluşturulan 3D rotasyonunu almak için `Submit` düğmesine tıklayın

### StableAudio:

1) Modeli ihtiyacınız olan parametrelere göre ayarlayın
2) İsteğinizi girin
3) Oluşturulan sesi almak için `Submit` düğmesine tıklayın

### AudioCraft:

1) Açılır listeden bir model seçin
2) Model türünü seçin (`musicgen`, `audiogen` veya `magnet`)
3) Modeli ihtiyacınız olan parametrelere göre ayarlayın
4) İsteğinizi girin
5) (İsteğe bağlı) `melody` modelini kullanıyorsanız başlangıç sesini yükleyin 
6) Oluşturulan sesi almak için `Submit` düğmesine tıklayın
#### İsteğe bağlı: Oluşturulan sesi iyileştirmek için `multiband diffusion`ı etkinleştirebilirsiniz

### AudioLDM 2:

1) Açılır listeden bir model seçin
2) Modeli ihtiyacınız olan parametrelere göre ayarlayın
3) İsteğinizi girin
4) Oluşturulan sesi almak için `Submit` düğmesine tıklayın

### Demucs:

1) Ayırmak için başlangıç sesini yükleyin
2) Ayrılmış sesi almak için `Submit` düğmesine tıklayın

### Galeri:

* Burada çıktılar dizinindeki dosyaları görüntüleyebilirsiniz

### ModelDownloader:

* Burada `LLM` ve `StableDiffusion` modellerini indirebilirsiniz. Açılır listeden modeli seçin ve `Submit` düğmesine tıklayın
#### `LLM` modelleri buraya indirilir: *inputs/text/llm_models*
#### `StableDiffusion` modelleri buraya indirilir: *inputs/image/sd_models*

### Ayarlar: 

* Burada uygulama ayarlarını değiştirebilirsiniz. Şu anda sadece `Share` modunu `True` veya `False` olarak değiştirebilirsiniz

### Sistem: 

* Burada `Submit` düğmesine tıklayarak bilgisayarınızın sensörlerinin göstergelerini görebilirsiniz

### Ek Bilgiler:

1) Tüm oluşturmalar *outputs* klasörüne kaydedilir
2) Seçiminizi sıfırlamak için `Clear` düğmesine basabilirsiniz
3) Oluşturma işlemini durdurmak için `Stop generation` düğmesine tıklayın
4) `Close terminal` düğmesini kullanarak uygulamayı kapatabilirsiniz
5) `Outputs` düğmesine tıklayarak *outputs* klasörünü açabilirsiniz

## Modelleri ve sesleri nereden alabilirim?

* LLM modelleri [HuggingFace](https://huggingface.co/models)'den veya arayüz içindeki ModelDownloader'dan alınabilir 
* StableDiffusion, vae, inpaint, embedding ve lora modelleri [CivitAI](https://civitai.com/models)'den veya arayüz içindeki ModelDownloader'dan alınabilir
* StableAudioOpen, AudioCraft, AudioLDM 2, TTS, Whisper, Wav2Lip, SunoBark, MoonDream2, Upscale, GLIGEN, Depth, Pix2Pix, Controlnet, AnimateDiff, Videos, LDM3D, SD3, Cascade, InstantID, PhotoMaker, IP-Adapter-FaceID, Rembg, Roop, CodeFormer, Real-ESRGAN, TripoSR, StableFast3D, Shap-E, SV34D, Zero123Plus, Demucs, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, AuraSR, Würstchen, DeepFloydIF, PixArt, ModelScope, ZeroScope 2, CogVideoX, Latte ve Multiband diffusion modelleri kullanıldıklarında *inputs* klasörüne otomatik olarak indirilir 
* Sesleri herhangi bir yerden alabilirsiniz. Kendinizinkini kaydedin veya internetten bir kayıt alın. Ya da projede zaten var olanları kullanın. Önemli olan önceden işlenmiş olmasıdır!

## Wiki

* https://github.com/Dartvauder/NeuroSandboxWebUI/wiki

## Geliştiricilere teşekkür

#### Bu projelere çok teşekkür ederim çünkü onların uygulamaları/kütüphaneleri sayesinde kendi uygulamımı oluşturabildim:

Her şeyden önce, [PyCharm](https://www.jetbrains.com/pycharm/) ve [GitHub](https://desktop.github.com) geliştiricilerine teşekkür etmek istiyorum. Onların uygulamaları sayesinde kodumu oluşturup paylaşabildim

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

## Üçüncü Taraf Lisansları:

#### Birçok modelin kendi kullanım lisansı vardır. Kullanmadan önce bunlarla tanışmanızı tavsiye ederim:

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

## Bağış

### *Eğer projemi beğendiyseniz ve bağış yapmak istiyorsanız, işte bağış seçenekleri. Şimdiden çok teşekkür ederim!*

* CryptoWallet(BEP-20) - 0x3d86bdb5f50b92d0d7Eb44F1a833acC5e91aAEcA

* [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Dartvauder/NeuroSandboxWebUI&type=Date)](https://star-history.com/#Dartvauder/NeuroSandboxWebUI&Date)
