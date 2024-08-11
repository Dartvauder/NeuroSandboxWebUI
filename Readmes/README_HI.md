## [विशेषताएं](/#विशेषताएं) | [निर्भरताएं](/#आवश्यक-निर्भरताएं) | [सिस्टम आवश्यकताएं](/#न्यूनतम-सिस्टम-आवश्यकताएं) | [स्थापना](/#कैसे-स्थापित-करें) | [उपयोग](/#कैसे-उपयोग-करें) | [मॉडल](/#मैं-मॉडल-आवाजें-और-अवतार-कहां-से-प्राप्त-कर-सकता-हूं) | [विकी](/#विकी) | [स्वीकृति](/#डेवलपर्स-को-स्वीकृति) | [लाइसेंस](/#तृतीय-पक्ष-लाइसेंस)

# ![main](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4ea0d891-8979-45ad-b052-626c41ae991a)
* कार्य प्रगति पर है! (अल्फा)
* [English](/README.md) | [عربي](/Readmes/README_AR.md) | [Deutsche](/Readmes/README_DE.md) | [Español](/Readmes/README_ES.md) | [Français](/Readmes/README_FR.md) | [日本語](/Readmes/README_JP.md) | [Русский](/Readmes/README_RU.md) | [漢語](/Readmes/README_ZH.md) | [Português](/Readmes/README_PT.md) | हिंदी | [Italiano](/Readmes/README_IT.md)

## विवरण:

विभिन्न न्यूरल नेटवर्क मॉडल का उपयोग करने के लिए एक सरल और सुविधाजनक इंटरफ़ेस। आप टेक्स्ट, आवाज और छवि इनपुट का उपयोग करके LLM और Moondream2 के साथ संवाद कर सकते हैं; छवियां उत्पन्न करने के लिए StableDiffusion, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF और PixArt का उपयोग कर सकते हैं; वीडियो उत्पन्न करने के लिए ModelScope, ZeroScope 2, CogVideoX और Latte का उपयोग कर सकते हैं; 3D वस्तुएं उत्पन्न करने के लिए TripoSR, StableFast3D, Shap-E, SV34D और Zero123Plus का उपयोग कर सकते हैं; संगीत और ऑडियो उत्पन्न करने के लिए StableAudioOpen, AudioCraft और AudioLDM 2 का उपयोग कर सकते हैं; टेक्स्ट-टू-स्पीच के लिए CoquiTTS और SunoBark का उपयोग कर सकते हैं; स्पीच-टू-टेक्स्ट के लिए OpenAI-Whisper का उपयोग कर सकते हैं; लिप-सिंक के लिए Wav2Lip का उपयोग कर सकते हैं; फेसस्वैप के लिए Roop का उपयोग कर सकते हैं; बैकग्राउंड हटाने के लिए Rembg का उपयोग कर सकते हैं; चेहरे की बहाली के लिए CodeFormer का उपयोग कर सकते हैं; टेक्स्ट अनुवाद के लिए LibreTranslate का उपयोग कर सकते हैं; ऑडियो फ़ाइल अलगाव के लिए Demucs का उपयोग कर सकते हैं। आप गैलरी में आउटपुट डायरेक्टरी से फ़ाइलें भी देख सकते हैं, LLM और StableDiffusion मॉडल डाउनलोड कर सकते हैं, इंटरफ़ेस के अंदर एप्लिकेशन सेटिंग्स बदल सकते हैं और सिस्टम सेंसर जांच सकते हैं

परियोजना का लक्ष्य - न्यूरल नेटवर्क मॉडल का उपयोग करने के लिए संभव सबसे आसान एप्लिकेशन बनाना

### टेक्स्ट: <img width="1130" alt="1hi" src="https://github.com/user-attachments/assets/da4aa91c-7771-4688-b592-48f67ff350b2">

### छवि: <img width="1123" alt="2hi" src="https://github.com/user-attachments/assets/de35be10-579c-4955-bd91-842824367edd">

### वीडियो: <img width="1121" alt="3hi" src="https://github.com/user-attachments/assets/76ad9a35-593c-4dfc-99b6-85428f242007">

### 3D: <img width="1119" alt="4hi" src="https://github.com/user-attachments/assets/0ec4c9d1-68a0-42ae-8016-801b09e8e210">

### ऑडियो: <img width="1121" alt="5hi" src="https://github.com/user-attachments/assets/5fc6fb75-e3ec-4c13-a033-da5aed8d0d1c">

### इंटरफ़ेस: <img width="1119" alt="6hi" src="https://github.com/user-attachments/assets/0f45d6d5-1119-48e3-84f5-8e1481ecebfe">

## विशेषताएं:

* install.bat(Windows) या install.sh(Linux) के माध्यम से आसान स्थापना
* आप अपने मोबाइल डिवाइस पर localhost(IPv4 के माध्यम से) या कहीं भी ऑनलाइन(शेयर के माध्यम से) एप्लिकेशन का उपयोग कर सकते हैं
* लचीला और अनुकूलित इंटरफ़ेस (Gradio द्वारा)
* admin:admin के माध्यम से प्रमाणीकरण (आप GradioAuth.txt फ़ाइल में अपना लॉगिन विवरण दर्ज कर सकते हैं)
* आप एक विशिष्ट मॉडल डाउनलोड करने के लिए अपना स्वयं का HuggingFace-Token जोड़ सकते हैं (आप HF-Token.txt फ़ाइल में अपना टोकन दर्ज कर सकते हैं)
* Transformers और llama.cpp मॉडल के लिए समर्थन (LLM)
* diffusers और safetensors मॉडल के लिए समर्थन (StableDiffusion) - txt2img, img2img, depth2img, pix2pix, controlnet, upscale, inpaint, gligen, animatediff, video, ldm3d, sd3, cascade और extras टैब
* छवि उत्पादन के लिए अतिरिक्त मॉडल का समर्थन: Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF और PixArt
* StableAudioOpen समर्थन
* AudioCraft समर्थन (मॉडल: musicgen, audiogen और magnet)
* AudioLDM 2 समर्थन (मॉडल: audio और music)
* TTS और Whisper मॉडल के लिए समर्थन (LLM और TTS-STT के लिए)
* Lora, Textual inversion (embedding), Vae, Img2img, Depth, Pix2Pix, Controlnet, Upscale(latent), Upscale(Real-ESRGAN), Inpaint, GLIGEN, AnimateDiff, Videos, LDM3D, SD3, Cascade, Rembg, CodeFormer और Roop मॉडल के लिए समर्थन (StableDiffusion के लिए)
* Multiband Diffusion मॉडल के लिए समर्थन (AudioCraft के लिए)
* LibreTranslate के लिए समर्थन (स्थानीय API)
* वीडियो उत्पादन के लिए ModelScope, ZeroScope 2, CogVideoX और Latte के लिए समर्थन
* SunoBark के लिए समर्थन
* Demucs के लिए समर्थन
* 3D उत्पादन के लिए TripoSR, StableFast3D, Shap-E, SV34D और Zero123Plus के लिए समर्थन
* Wav2Lip के लिए समर्थन
* LLM के लिए मल्टीमोडल (Moondream 2), LORA (transformers) और WebSearch (GoogleSearch के साथ) के लिए समर्थन
* इंटरफ़ेस के अंदर मॉडल सेटिंग्स
* गैलरी
* ModelDownloader (LLM और StableDiffusion के लिए)
* एप्लिकेशन सेटिंग्स
* सिस्टम सेंसर देखने की क्षमता

## आवश्यक निर्भरताएं:

* [Python](https://www.python.org/downloads/) (3.11)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (12.4) और [cuDNN](https://developer.nvidia.com/cudnn-downloads) (9.1)
* [FFMPEG](https://ffmpeg.org/download.html)
- C+ कंपाइलर
  - Windows: [VisualStudio](https://visualstudio.microsoft.com/ru/)
  - Linux: [GCC](https://gcc.gnu.org/)

## न्यूनतम सिस्टम आवश्यकताएं:

* सिस्टम: Windows या Linux
* GPU: 6GB+ या CPU: 8 कोर 3.2GHZ
* RAM: 16GB+
* डिस्क स्पेस: 20GB+
* मॉडल डाउनलोड करने और स्थापित करने के लिए इंटरनेट

## कैसे स्थापित करें:

### Windows

1) किसी भी स्थान पर `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` करें
2) `Install.bat` चलाएं और स्थापना की प्रतीक्षा करें
3) स्थापना के बाद, `Start.bat` चलाएं
4) फ़ाइल संस्करण का चयन करें और एप्लिकेशन के लॉन्च होने की प्रतीक्षा करें
5) अब आप जनरेट करना शुरू कर सकते हैं!

अपडेट प्राप्त करने के लिए, `Update.bat` चलाएं
टर्मिनल के माध्यम से वर्चुअल वातावरण के साथ काम करने के लिए, `Venv.bat` चलाएं

### Linux

1) किसी भी स्थान पर `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` करें
2) टर्मिनल में, `./Install.sh` चलाएं और सभी निर्भरताओं की स्थापना की प्रतीक्षा करें
3) स्थापना के बाद, `./Start.sh` चलाएं
4) एप्लिकेशन के लॉन्च होने की प्रतीक्षा करें
5) अब आप जनरेट करना शुरू कर सकते हैं!

अपडेट प्राप्त करने के लिए, `./Update.sh` चलाएं
टर्मिनल के माध्यम से वर्चुअल वातावरण के साथ काम करने के लिए, `./Venv.sh` चलाएं

## कैसे उपयोग करें:

#### इंटरफ़ेस में छह मुख्य टैब में बत्तीस टैब हैं: LLM, TTS-STT, SunoBark, LibreTranslate, Wav2Lip, StableDiffusion, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF, PixArt, ModelScope, ZeroScope 2, CogVideoX, Latte, TripoSR, StableFast3D, Shap-E, SV34D, Zero123Plus, StableAudio, AudioCraft, AudioLDM 2, Demucs, Gallery, ModelDownloader, Settings और System। आपको जिसकी आवश्यकता है उसका चयन करें और नीचे दिए गए निर्देशों का पालन करें

### LLM:

1) सबसे पहले अपने मॉडल को इस फ़ोल्डर में अपलोड करें: *inputs/text/llm_models*
2) ड्रॉप-डाउन सूची से अपना मॉडल चुनें
3) मॉडल प्रकार चुनें (`transformers` या `llama`)
4) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
5) अपना अनुरोध टाइप करें (या बोलें)
6) जनरेट किए गए टेक्स्ट और ऑडियो प्रतिक्रिया प्राप्त करने के लिए `Submit` बटन पर क्लिक करें
#### वैकल्पिक: आप ऑडियो प्रतिक्रिया प्राप्त करने के लिए `TTS` मोड सक्षम कर सकते हैं, आवश्यक `voice` और `language` का चयन कर सकते हैं। आप छवि का विवरण प्राप्त करने के लिए `multimodal` सक्षम कर सकते हैं और एक छवि अपलोड कर सकते हैं। आप इंटरनेट एक्सेस के लिए `websearch` सक्षम कर सकते हैं। अनुवाद प्राप्त करने के लिए आप `libretranslate` सक्षम कर सकते हैं। जनरेशन में सुधार के लिए आप `LORA` मॉडल भी चुन सकते हैं
#### आवाज के नमूने = *inputs/audio/voices*
#### LORA = *inputs/text/llm_models/lora*
#### आवाज पहले से संसाधित होनी चाहिए (22050 kHz, मोनो, WAV)

### TTS-STT:

1) टेक्स्ट टू स्पीच के लिए टेक्स्ट टाइप करें
2) स्पीच टू टेक्स्ट के लिए ऑडियो इनपुट करें
3) जनरेट किए गए टेक्स्ट और ऑडियो प्रतिक्रिया प्राप्त करने के लिए `Submit` बटन पर क्लिक करें
#### आवाज के नमूने = *inputs/audio/voices*
#### आवाज पहले से संसाधित होनी चाहिए (22050 kHz, मोनो, WAV)

### SunoBark:

1) अपना अनुरोध टाइप करें
2) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
3) जनरेट की गई ऑडियो प्रतिक्रिया प्राप्त करने के लिए `Submit` बटन पर क्लिक करें

### LibreTranslate:

* सबसे पहले आपको [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate) स्थापित और चलाना होगा
1) स्रोत और लक्ष्य भाषाओं का चयन करें
2) अनुवाद प्राप्त करने के लिए `Submit` बटन पर क्लिक करें
#### वैकल्पिक: आप संबंधित बटन को चालू करके अनुवाद इतिहास सहेज सकते हैं

### Wav2Lip:

1) चेहरे की प्रारंभिक छवि अपलोड करें
2) आवाज की प्रारंभिक ऑडियो अपलोड करें
3) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
4) लिप-सिंक प्राप्त करने के लिए `Submit` बटन पर क्लिक करें

### StableDiffusion - में पंद्रह उप-टैब हैं:

#### txt2img:

1) सबसे पहले अपने मॉडल को इस फ़ोल्डर में अपलोड करें: *inputs/image/sd_models*
2) ड्रॉप-डाउन सूची से अपना मॉडल चुनें
3) मॉडल प्रकार चुनें (`SD`, `SD2` या `SDXL`)
4) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
5) अपना अनुरोध दर्ज करें (प्रॉम्प्ट भारांक के लिए + और -)
6) जनरेट की गई छवि प्राप्त करने के लिए `Submit` बटन पर क्लिक करें
#### वैकल्पिक: आप जनरेशन विधि में सुधार करने के लिए अपने `vae`, `embedding` और `lora` मॉडल का चयन कर सकते हैं, साथ ही आप जनरेट की गई छवि के आकार को बढ़ाने के लिए `upscale` सक्षम कर सकते हैं 
#### vae = *inputs/image/sd_models/vae*
#### lora = *inputs/image/sd_models/lora*
#### embedding = *inputs/image/sd_models/embedding*

#### img2img:

1) सबसे पहले अपने मॉडल को इस फ़ोल्डर में अपलोड करें: *inputs/image/sd_models*
2) ड्रॉप-डाउन सूची से अपना मॉडल चुनें
3) मॉडल प्रकार चुनें (`SD`, `SD2` या `SDXL`)
4) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
5) प्रारंभिक छवि अपलोड करें जिसके साथ जनरेशन होगा
6) अपना अनुरोध दर्ज करें (प्रॉम्प्ट भारांक के लिए + और -)
7) जनरेट की गई छवि प्राप्त करने के लिए `Submit` बटन पर क्लिक करें
#### वैकल्पिक: आप अपने `vae` मॉडल का चयन कर सकते हैं
#### vae = *inputs/image/sd_models/vae*

#### depth2img:

1) प्रारंभिक छवि अपलोड करें
2) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
3) अपना अनुरोध दर्ज करें (प्रॉम्प्ट भारांक के लिए + और -)
4) जनरेट की गई छवि प्राप्त करने के लिए `Submit` बटन पर क्लिक करें

#### pix2pix:

1) प्रारंभिक छवि अपलोड करें
2) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
3) अपना अनुरोध दर्ज करें (प्रॉम्प्ट भारांक के लिए + और -)
4) जनरेट की गई छवि प्राप्त करने के लिए `Submit` बटन पर क्लिक करें

#### controlnet:

1) सबसे पहले अपने स्टेबल डिफ्यूजन मॉडल को इस फ़ोल्डर में अपलोड करें: *inputs/image/sd_models*
2) प्रारंभिक छवि अपलोड करें
3) ड्रॉप-डाउन सूचियों से अपने स्टेबल डिफ्यूजन और कंट्रोलनेट मॉडल का चयन करें
4) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
5) अपना अनुरोध दर्ज करें (प्रॉम्प्ट भारांक के लिए + और -)
6) जनरेट की गई छवि प्राप्त करने के लिए `Submit` बटन पर क्लिक करें

#### upscale(latent):

1) प्रारंभिक छवि अपलोड करें
2) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
3) अपस्केल की गई छवि प्राप्त करने के लिए `Submit` बटन पर क्लिक करें

#### upscale(Real-ESRGAN):

1) प्रारंभिक छवि अपलोड करें
2) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
3) अपस्केल की गई छवि प्राप्त करने के लिए `Submit` बटन पर क्लिक करें

#### inpaint:

1) सबसे पहले अपने मॉडल को इस फ़ोल्डर में अपलोड करें: *inputs/image/sd_models/inpaint*
2) ड्रॉप-डाउन सूची से अपना मॉडल चुनें
3) मॉडल प्रकार चुनें (`SD`, `SD2` या `SDXL`)
4) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
5) जिस छवि के साथ जनरेशन होगा उसे `initial image` और `mask image` में अपलोड करें
6) `mask image` में, ब्रश का चयन करें, फिर पैलेट और रंग को `#FFFFFF` में बदलें
7) जनरेशन के लिए एक स्थान बनाएं और अपना अनुरोध दर्ज करें (प्रॉम्प्ट भारांक के लिए + और -)
8) इनपेंट की गई छवि प्राप्त करने के लिए `Submit` बटन पर क्लिक करें
#### वैकल्पिक: आप अपने `vae` मॉडल का चयन कर सकते हैं
#### vae = *inputs/image/sd_models/vae*

#### gligen:

1) सबसे पहले अपने मॉडल को इस फ़ोल्डर में अपलोड करें: *inputs/image/sd_models*
2) ड्रॉप-डाउन सूची से अपना मॉडल चुनें
3) मॉडल प्रकार चुनें (`SD`, `SD2` या `SDXL`)
4) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
5) प्रॉम्प्ट के लिए अपना अनुरोध दर्ज करें (प्रॉम्प्ट भारांक के लिए + और -) और GLIGEN वाक्यांश (बॉक्स के लिए "" में)
6) GLIGEN बॉक्स दर्ज करें (बॉक्स के लिए [0.1387, 0.2051, 0.4277, 0.7090] जैसे)
7) जनरेट की गई छवि प्राप्त करने के लिए `Submit` बटन पर क्लिक करें

#### animatediff:

1) सबसे पहले अपने मॉडल को इस फ़ोल्डर में अपलोड करें: *inputs/image/sd_models*
2) ड्रॉप-डाउन सूची से अपना मॉडल चुनें
3) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
4) अपना अनुरोध दर्ज करें (प्रॉम्प्ट भारांक के लिए + और -)
5) जनरेट की गई छवि एनिमेशन प्राप्त करने के लिए `Submit` बटन पर क्लिक करें

#### video:

1) प्रारंभिक छवि अपलोड करें
2) अपना अनुरोध दर्ज करें (IV2Gen-XL के लिए)
3) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
4) छवि से वीडियो प्राप्त करने के लिए `Submit` बटन पर क्लिक करें

#### ldm3d:

1) अपना अनुरोध दर्ज करें
2) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
3) जनरेट की गई छवियाँ प्राप्त करने के लिए `Submit` बटन पर क्लिक करें

#### sd3:

1) अपना अनुरोध दर्ज करें
2) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
3) जनरेट की गई छवि प्राप्त करने के लिए `Submit` बटन पर क्लिक करें

#### cascade:

1) अपना अनुरोध दर्ज करें
2) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
3) जनरेट की गई छवि प्राप्त करने के लिए `Submit` बटन पर क्लिक करें

#### extras:

1) प्रारंभिक छवि अपलोड करें
2) आपको आवश्यक विकल्पों का चयन करें
3) संशोधित छवि प्राप्त करने के लिए `Submit` बटन पर क्लिक करें

### Kandinsky:

1) अपना प्रॉम्प्ट दर्ज करें
2) ड्रॉप-डाउन सूची से एक मॉडल का चयन करें
3) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
4) जनरेट की गई छवि प्राप्त करने के लिए `Submit` पर क्लिक करें

### Flux:

1) अपना प्रॉम्प्ट दर्ज करें
2) ड्रॉप-डाउन सूची से एक मॉडल का चयन करें
3) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
4) जनरेट की गई छवि प्राप्त करने के लिए `Submit` पर क्लिक करें

### HunyuanDiT:

1) अपना प्रॉम्प्ट दर्ज करें
2) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
3) जनरेट की गई छवि प्राप्त करने के लिए `Submit` पर क्लिक करें

### Lumina-T2X:

1) अपना प्रॉम्प्ट दर्ज करें
2) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
3) जनरेट की गई छवि प्राप्त करने के लिए `Submit` पर क्लिक करें

### Kolors:

1) अपना प्रॉम्प्ट दर्ज करें
2) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
3) जनरेट की गई छवि प्राप्त करने के लिए `Submit` पर क्लिक करें

### AuraFlow:

1) अपना प्रॉम्प्ट दर्ज करें
2) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
3) जनरेट की गई छवि प्राप्त करने के लिए `Submit` पर क्लिक करें

### Würstchen:

1) अपना प्रॉम्प्ट दर्ज करें
2) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
3) जनरेट की गई छवि प्राप्त करने के लिए `Submit` पर क्लिक करें

### DeepFloydIF:

1) अपना प्रॉम्प्ट दर्ज करें
2) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
3) जनरेट की गई छवि प्राप्त करने के लिए `Submit` पर क्लिक करें

### PixArt:

1) अपना प्रॉम्प्ट दर्ज करें
2) ड्रॉप-डाउन सूची से मॉडल का चयन करें
3) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
4) जनरेट की गई छवि प्राप्त करने के लिए `Submit` पर क्लिक करें

### ModelScope:

1) अपना प्रॉम्प्ट दर्ज करें
2) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
3) जनरेट किया गया वीडियो प्राप्त करने के लिए `Submit` पर क्लिक करें

### ZeroScope 2:

1) अपना प्रॉम्प्ट दर्ज करें
2) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
3) जनरेट किया गया वीडियो प्राप्त करने के लिए `Submit` पर क्लिक करें

### CogVideoX:

1) अपना प्रॉम्प्ट दर्ज करें
2) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
3) जनरेट किया गया वीडियो प्राप्त करने के लिए `Submit` पर क्लिक करें

### Latte:

1) अपना प्रॉम्प्ट दर्ज करें
2) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
3) जनरेट किया गया वीडियो प्राप्त करने के लिए `Submit` पर क्लिक करें

### TripoSR:

1) प्रारंभिक छवि अपलोड करें
2) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
3) जनरेट की गई 3D वस्तु प्राप्त करने के लिए `Submit` बटन पर क्लिक करें

### StableFast3D:

1) प्रारंभिक छवि अपलोड करें
2) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
3) जनरेट की गई 3D वस्तु प्राप्त करने के लिए `Submit` बटन पर क्लिक करें

### Shap-E:

1) अपना अनुरोध दर्ज करें या प्रारंभिक छवि अपलोड करें
2) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
3) जनरेट की गई 3D वस्तु प्राप्त करने के लिए `Submit` बटन पर क्लिक करें

### SV34D:

1) प्रारंभिक छवि (3D के लिए) या वीडियो (4D के लिए) अपलोड करें
2) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
3) जनरेट किया गया 3D वीडियो प्राप्त करने के लिए `Submit` बटन पर क्लिक करें

### Zero123Plus:

1) प्रारंभिक छवि अपलोड करें
2) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
3) छवि की जनरेट की गई 3D रोटेशन प्राप्त करने के लिए `Submit` बटन पर क्लिक करें

### StableAudio:

1) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
2) अपना अनुरोध दर्ज करें
3) जनरेट किया गया ऑडियो प्राप्त करने के लिए `Submit` बटन पर क्लिक करें

### AudioCraft:

1) ड्रॉप-डाउन सूची से एक मॉडल का चयन करें
2) मॉडल प्रकार चुनें (`musicgen`, `audiogen` या `magnet`)
3) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
4) अपना अनुरोध दर्ज करें
5) (वैकल्पिक) यदि आप `melody` मॉडल का उपयोग कर रहे हैं तो प्रारंभिक ऑडियो अपलोड करें 
6) जनरेट किया गया ऑडियो प्राप्त करने के लिए `Submit` बटन पर क्लिक करें
#### वैकल्पिक: आप जनरेट किए गए ऑडियो में सुधार करने के लिए `multiband diffusion` सक्षम कर सकते हैं

### AudioLDM 2:

1) ड्रॉप-डाउन सूची से एक मॉडल का चयन करें
2) आपको आवश्यक पैरामीटर के अनुसार मॉडल सेट करें
3) अपना अनुरोध दर्ज करें
4) जनरेट किया गया ऑडियो प्राप्त करने के लिए `Submit` बटन पर क्लिक करें

### Demucs:

1) अलग करने के लिए प्रारंभिक ऑडियो अपलोड करें
2) अलग किए गए ऑडियो प्राप्त करने के लिए `Submit` बटन पर क्लिक करें

### Gallery:

* यहां आप आउटपुट डायरेक्टरी से फ़ाइलें देख सकते हैं

### ModelDownloader:

* यहां आप `LLM` और `StableDiffusion` मॉडल डाउनलोड कर सकते हैं। बस ड्रॉप-डाउन सूची से मॉडल चुनें और `Submit` बटन पर क्लिक करें
#### `LLM` मॉडल यहां डाउनलोड किए जाते हैं: *inputs/text/llm_models*
#### `StableDiffusion` मॉडल यहां डाउनलोड किए जाते हैं: *inputs/image/sd_models*

### Settings: 

* यहां आप एप्लिकेशन सेटिंग्स बदल सकते हैं। अभी के लिए आप केवल `Share` मोड को `True` या `False` में बदल सकते हैं

### System: 

* यहां आप `Submit` बटन पर क्लिक करके अपने कंप्यूटर के सेंसर के संकेतक देख सकते हैं

### अतिरिक्त जानकारी:

1) सभी जनरेशन *outputs* फ़ोल्डर में सहेजे जाते हैं
2) आप अपने चयन को रीसेट करने के लिए `Clear` बटन दबा सकते हैं
3) जनरेशन प्रक्रिया को रोकने के लिए, `Stop generation` बटन पर क्लिक करें
4) आप `Close terminal` बटन का उपयोग करके एप्लिकेशन बंद कर सकते हैं
5) आप `Outputs` बटन पर क्लिक करके *outputs* फ़ोल्डर खोल सकते हैं

## मैं मॉडल और आवाजें कहां से प्राप्त कर सकता हूं?

* LLM मॉडल [HuggingFace](https://huggingface.co/models) से या इंटरफेस के अंदर ModelDownloader से लिए जा सकते हैं 
* StableDiffusion, vae, inpaint, embedding और lora मॉडल [CivitAI](https://civitai.com/models) से या इंटरफेस के अंदर ModelDownloader से लिए जा सकते हैं
* StableAudioOpen, AudioCraft, AudioLDM 2, TTS, Whisper, Wav2Lip, SunoBark, MoonDream2, Upscale, GLIGEN, Depth, Pix2Pix, Controlnet, AnimateDiff, Videos, LDM3D, SD3, Cascade, Rembg, Roop, CodeFormer, Real-ESRGAN, TripoSR, StableFast3D, Shap-E, SV34D, Zero123Plus, Demucs, Kandinsky, Flux, HunyuanDiT, Lumina-T2X, Kolors, AuraFlow, Würstchen, DeepFloydIF, PixArt, ModelScope, ZeroScope 2, CogVideoX, Latte और Multiband diffusion मॉडल उपयोग किए जाने पर *inputs* फ़ोल्डर में स्वचालित रूप से डाउनलोड होते हैं 
* आप कहीं से भी आवाजें ले सकते हैं। अपनी खुद की रिकॉर्ड करें या इंटरनेट से रिकॉर्डिंग लें। या बस उन्हें इस्तेमाल करें जो पहले से ही प्रोजेक्ट में हैं। मुख्य बात यह है कि यह पहले से संसाधित हो!

## विकी

* https://github.com/Dartvauder/NeuroSandboxWebUI/wiki

## डेवलपर्स को स्वीकृति

#### इन परियोजनाओं को बहुत धन्यवाद क्योंकि उनके एप्लिकेशन/लाइब्रेरी की मदद से, मैं अपना एप्लिकेशन बना सका:

सबसे पहले, मैं [PyCharm](https://www.jetbrains.com/pycharm/) और [GitHub](https://desktop.github.com) के डेवलपर्स को धन्यवाद देना चाहता हूं। उनके एप्लिकेशन की मदद से, मैं अपना कोड बना और साझा कर सका

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

## तृतीय-पक्ष लाइसेंस:

#### कई मॉडलों के उपयोग के लिए उनके अपने लाइसेंस हैं। उपयोग करने से पहले, मैं आपको उनसे परिचित होने की सलाह देता हूं:

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

## दान

### *यदि आपको मेरा प्रोजेक्ट पसंद आया और आप दान करना चाहते हैं, तो यहां दान करने के विकल्प हैं। अग्रिम रूप से बहुत-बहुत धन्यवाद!*

* क्रिप्टोवॉलेट(BEP-20) - 0x3d86bdb5f50b92d0d7Eb44F1a833acC5e91aAEcA

* [!["मुझे एक कॉफी खरीदें"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)
