## [الميزات](/#Features) | [الاعتمادات](/#Required-Dependencies) | [متطلبات النظام](/#Minimum-System-Requirements) | [التثبيت](/#How-to-install) | [الاستخدام](/#How-to-use) | [النماذج](/#Where-can-I-get-models-voices-and-avatars) | [الويكي](/#Wiki) | [شكر للمطورين](/#Acknowledgment-to-developers) | [تراخيص الطرف الثالث](/#Third-Party-Licenses)

# ![main](https://github.com/Dartvauder/NeuroSandboxWebUI/assets/140557322/4ea0d891-8979-45ad-b052-626c41ae991a)
* العمل قيد التقدم! (ألفا)
* [English](/README.md) | عربي | [Deutsche](/Readmes/README_DE.md) | [Español](/Readmes/README_ES.md) | [Français](/Readmes/README_FR.md) | [日本語](/Readmes/README_JP.md) | [Русский](/Readmes/README_RU.md) | [漢語](/Readmes/README_ZH.md) | [Português](/Readmes/README_PT.md) | [हिंदी](/Readmes/README_HI.md) | [Italiano](/Readmes/README_IT.md) | [韓國語](/Readmes/README_KO.md)

## الوصف:

واجهة بسيطة ومريحة لاستخدام نماذج الشبكات العصبية المختلفة. يمكنك التواصل مع LLM و Moondream2 باستخدام إدخال النص والصوت والصورة؛ استخدام StableDiffusion و Kandinsky و Flux و HunyuanDiT و Lumina-T2X و Kolors و AuraFlow و Würstchen و DeepFloydIF و PixArt لإنشاء الصور؛ ModelScope و ZeroScope 2 و CogVideoX و Latte لإنشاء مقاطع الفيديو؛ TripoSR و StableFast3D و Shap-E و SV34D و Zero123Plus لإنشاء الأجسام ثلاثية الأبعاد؛ StableAudioOpen و AudioCraft و AudioLDM 2 لإنشاء الموسيقى والصوت؛ CoquiTTS و SunoBark لتحويل النص إلى كلام؛ OpenAI-Whisper لتحويل الكلام إلى نص؛ Wav2Lip لمزامنة الشفاه؛ Roop لتبديل الوجوه؛ Rembg لإزالة الخلفية؛ CodeFormer لاستعادة الوجه؛ LibreTranslate لترجمة النص؛ Demucs لفصل ملفات الصوت. يمكنك أيضًا عرض الملفات من دليل المخرجات في المعرض، وتنزيل نماذج LLM و StableDiffusion، وتغيير إعدادات التطبيق داخل الواجهة والتحقق من أجهزة استشعار النظام

الهدف من المشروع - إنشاء أسهل تطبيق ممكن لاستخدام نماذج الشبكات العصبية

### النص: <img width="1119" alt="1ar" src="https://github.com/user-attachments/assets/e115f987-5988-479a-b37d-97712fce66cc">

### الصورة: <img width="1121" alt="2ar" src="https://github.com/user-attachments/assets/ebe181af-bb7c-424c-bab0-11ac6b94c9eb">

### الفيديو: <img width="1121" alt="3ar" src="https://github.com/user-attachments/assets/500ebe14-1741-4392-a11c-781c87cc223c">

### ثلاثي الأبعاد: <img width="1120" alt="4ar" src="https://github.com/user-attachments/assets/64e3367d-dd83-43cd-ad15-47b24a444c58">

### الصوت: <img width="1123" alt="5ar" src="https://github.com/user-attachments/assets/88999b5b-cbe6-43eb-bacc-19c843f159d9">

### الواجهة: <img width="1118" alt="6ar" src="https://github.com/user-attachments/assets/30db9899-fba7-49f3-8ba7-d310f82af045">

## الميزات:

* التثبيت السهل عبر install.bat (ويندوز) أو install.sh (لينكس)
* يمكنك استخدام التطبيق عبر جهازك المحمول في localhost (عبر IPv4) أو في أي مكان على الإنترنت (عبر المشاركة)
* واجهة مرنة ومحسنة (بواسطة Gradio)
* المصادقة عبر admin:admin (يمكنك إدخال تفاصيل تسجيل الدخول الخاصة بك في ملف GradioAuth.txt)
* يمكنك إضافة HuggingFace-Token الخاص بك لتنزيل نماذج معينة (يمكنك إدخال الرمز المميز الخاص بك في ملف HF-Token.txt)
* دعم نماذج Transformers و llama.cpp (LLM)
* دعم نماذج diffusers و safetensors (StableDiffusion) - علامات التبويب txt2img و img2img و depth2img و pix2pix و controlnet و upscale و inpaint و gligen و animatediff و video و ldm3d و sd3 و cascade و adapters و الإضافات
* دعم نماذج إضافية لإنشاء الصور: Kandinsky و Flux و HunyuanDiT و Lumina-T2X و Kolors و AuraFlow و Würstchen و DeepFloydIF و PixArt
* دعم StableAudioOpen
* دعم AudioCraft (النماذج: musicgen و audiogen و magnet)
* دعم AudioLDM 2 (النماذج: audio و music)
* دعم نماذج TTS و Whisper (لـ LLM و TTS-STT)
* دعم نماذج Lora و Textual inversion (embedding) و Vae و Img2img و Depth و Pix2Pix و Controlnet و Upscale(latent) و Upscale(Real-ESRGAN) و Inpaint و GLIGEN و AnimateDiff و Videos و LDM3D و SD3 و Cascade و Adapters (InstantID و PhotoMaker و IP-Adapter-FaceID) و Rembg و CodeFormer و Roop (لـ StableDiffusion)
* دعم نموذج Multiband Diffusion (لـ AudioCraft)
* دعم LibreTranslate (واجهة برمجة التطبيقات المحلية)
* دعم ModelScope و ZeroScope 2 و CogVideoX و Latte لإنشاء الفيديو
* دعم SunoBark
* دعم Demucs
* دعم TripoSR و StableFast3D و Shap-E و SV34D و Zero123Plus لإنشاء ثلاثي الأبعاد
* دعم Wav2Lip
* دعم Multimodal (Moondream 2) و LORA (transformers) و WebSearch (مع GoogleSearch) لـ LLM
* إعدادات النموذج داخل الواجهة
* معرض
* ModelDownloader (لـ LLM و StableDiffusion)
* إعدادات التطبيق
* القدرة على رؤية أجهزة استشعار النظام

## الاعتمادات المطلوبة:

* [Python](https://www.python.org/downloads/) (3.11)
* [Git](https://git-scm.com/downloads)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (12.4) و [cuDNN](https://developer.nvidia.com/cudnn-downloads) (9.1)
* [FFMPEG](https://ffmpeg.org/download.html)
- مترجم ++C
  - ويندوز: [VisualStudio](https://visualstudio.microsoft.com/ru/)
  - لينكس: [GCC](https://gcc.gnu.org/)

## الحد الأدنى من متطلبات النظام:

* النظام: ويندوز أو لينكس
* وحدة معالجة الرسومات: 6 جيجابايت+ أو وحدة المعالجة المركزية: 8 نواة 3.2 جيجاهرتز
* ذاكرة الوصول العشوائي: 16 جيجابايت+
* مساحة القرص: 20 جيجابايت+
* الإنترنت لتنزيل النماذج والتثبيت

## كيفية التثبيت:

### ويندوز

1) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` إلى أي موقع
2) قم بتشغيل `Install.bat` وانتظر حتى يتم التثبيت
3) بعد التثبيت، قم بتشغيل `Start.bat`
4) حدد إصدار الملف وانتظر حتى يتم تشغيل التطبيق
5) الآن يمكنك البدء في الإنشاء!

للحصول على التحديث، قم بتشغيل `Update.bat`
للعمل مع البيئة الافتراضية من خلال الطرفية، قم بتشغيل `Venv.bat`

### لينكس

1) `Git clone https://github.com/Dartvauder/NeuroSandboxWebUI.git` إلى أي موقع
2) في الطرفية، قم بتشغيل `./Install.sh` وانتظر حتى يتم تثبيت جميع الاعتمادات
3) بعد التثبيت، قم بتشغيل `./Start.sh`
4) انتظر حتى يتم تشغيل التطبيق
5) الآن يمكنك البدء في الإنشاء!

للحصول على التحديث، قم بتشغيل `./Update.sh`
للعمل مع البيئة الافتراضية من خلال الطرفية، قم بتشغيل `./Venv.sh`

## كيفية الاستخدام:

#### تحتوي الواجهة على اثنين وثلاثين علامة تبويب في ستة علامات تبويب رئيسية (النص و الصورة و الفيديو و ثلاثي الأبعاد و الصوت و الواجهة): LLM و TTS-STT و SunoBark و LibreTranslate و Wav2Lip و StableDiffusion و Kandinsky و Flux و HunyuanDiT و Lumina-T2X و Kolors و AuraFlow و Würstchen و DeepFloydIF و PixArt و ModelScope و ZeroScope 2 و CogVideoX و Latte و TripoSR و StableFast3D و Shap-E و SV34D و Zero123Plus و StableAudio و AudioCraft و AudioLDM 2 و Demucs و Gallery و ModelDownloader و Settings و System. حدد ما تحتاجه واتبع التعليمات أدناه

### LLM:

1) أولاً قم بتحميل النماذج الخاصة بك إلى المجلد: *inputs/text/llm_models*
2) حدد النموذج الخاص بك من القائمة المنسدلة
3) حدد نوع النموذج (`transformers` أو `llama`)
4) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
5) اكتب (أو تحدث) طلبك
6) انقر على زر `Submit` لتلقي النص المولد والاستجابة الصوتية
#### اختياري: يمكنك تمكين وضع `TTS`، وتحديد `الصوت` و `اللغة` اللازمة لتلقي استجابة صوتية. يمكنك تمكين `multimodal` وتحميل صورة للحصول على وصفها. يمكنك تمكين `websearch` للوصول إلى الإنترنت. يمكنك تمكين `libretranslate` للحصول على الترجمة. يمكنك أيضًا اختيار نموذج `LORA` لتحسين التوليد
#### عينات الصوت = *inputs/audio/voices*
#### LORA = *inputs/text/llm_models/lora*
#### يجب معالجة الصوت مسبقًا (22050 كيلوهرتز، أحادي، WAV)

### TTS-STT:

1) اكتب النص لتحويل النص إلى كلام
2) أدخل الصوت لتحويل الكلام إلى نص
3) انقر على زر `Submit` لتلقي النص المولد والاستجابة الصوتية
#### عينات الصوت = *inputs/audio/voices*
#### يجب معالجة الصوت مسبقًا (22050 كيلوهرتز، أحادي، WAV)

### SunoBark:

1) اكتب طلبك
2) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
3) انقر على زر `Submit` لتلقي الاستجابة الصوتية المولدة

### LibreTranslate:

* أولاً تحتاج إلى تثبيت وتشغيل [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate)
1) حدد اللغات المصدر والهدف
2) انقر على زر `Submit` للحصول على الترجمة
#### اختياري: يمكنك حفظ سجل الترجمة عن طريق تشغيل الزر المقابل

### Wav2Lip:

1) قم بتحميل الصورة الأولية للوجه
2) قم بتحميل الصوت الأولي للصوت
3) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
4) انقر على زر `Submit` لتلقي مزامنة الشفاه

### StableDiffusion - يحتوي على خمسة عشر علامة تبويب فرعية:

#### txt2img:

1) أولاً قم بتحميل النماذج الخاصة بك إلى المجلد: *inputs/image/sd_models*
2) حدد النموذج الخاص بك من القائمة المنسدلة
3) حدد نوع النموذج (`SD` أو `SD2` أو `SDXL`)
4) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
5) أدخل طلبك (+ و - لترجيح الموجه)
6) انقر على زر `Submit` للحصول على الصورة المولدة
#### اختياري: يمكنك تحديد نماذج `vae` و `embedding` و `lora` الخاصة بك لتحسين طريقة التوليد، يمكنك أيضًا تمكين `upscale` لزيادة حجم الصورة المولدة 
#### vae = *inputs/image/sd_models/vae*
#### lora = *inputs/image/sd_models/lora*
#### embedding = *inputs/image/sd_models/embedding*

#### img2img:

1) أولاً قم بتحميل النماذج الخاصة بك إلى المجلد: *inputs/image/sd_models*
2) حدد النموذج الخاص بك من القائمة المنسدلة
3) حدد نوع النموذج (`SD` أو `SD2` أو `SDXL`)
4) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
5) قم بتحميل الصورة الأولية التي سيتم إجراء التوليد بها
6) أدخل طلبك (+ و - لترجيح الموجه)
7) انقر على زر `Submit` للحصول على الصورة المولدة
#### اختياري: يمكنك تحديد نموذج `vae` الخاص بك
#### vae = *inputs/image/sd_models/vae*

#### depth2img:

1) قم بتحميل الصورة الأولية
2) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
3) أدخل طلبك (+ و - لترجيح الموجه)
4) انقر على زر `Submit` للحصول على الصورة المولدة

#### pix2pix:

1) قم بتحميل الصورة الأولية
2) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
3) أدخل طلبك (+ و - لترجيح الموجه)
4) انقر على زر `Submit` للحصول على الصورة المولدة

#### controlnet:

1) أولاً قم بتحميل نماذج stable diffusion الخاصة بك إلى المجلد: *inputs/image/sd_models*
2) قم بتحميل الصورة الأولية
3) حدد نماذج stable diffusion و controlnet الخاصة بك من القوائم المنسدلة
4) قم بإعداد النماذج وفقًا للمعلمات التي تحتاجها
5) أدخل طلبك (+ و - لترجيح الموجه)
6) انقر على زر `Submit` للحصول على الصورة المولدة

#### upscale(latent):

1) قم بتحميل الصورة الأولية
2) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
3) انقر على زر `Submit` للحصول على الصورة المكبرة

#### upscale(Real-ESRGAN):

1) قم بتحميل الصورة الأولية
2) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
3) انقر على زر `Submit` للحصول على الصورة المكبرة

#### inpaint:

1) أولاً قم بتحميل النماذج الخاصة بك إلى المجلد: *inputs/image/sd_models/inpaint*
2) حدد النموذج الخاص بك من القائمة المنسدلة
3) حدد نوع النموذج (`SD` أو `SD2` أو `SDXL`)
4) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
5) قم بتحميل الصورة التي سيتم إجراء التوليد بها إلى `initial image` و `mask image`
6) في `mask image`، حدد الفرشاة، ثم اللوحة وغير اللون إلى `#FFFFFF`
7) ارسم مكانًا للتوليد وأدخل طلبك (+ و - لترجيح الموجه)
8) انقر على زر `Submit` للحصول على الصورة المرممة
#### اختياري: يمكنك تحديد نموذج `vae` الخاص بك
#### vae = *inputs/image/sd_models/vae*

#### gligen:

1) أولاً قم بتحميل النماذج الخاصة بك إلى المجلد: *inputs/image/sd_models*
2) حدد النموذج الخاص بك من القائمة المنسدلة
3) حدد نوع النموذج (`SD` أو `SD2` أو `SDXL`)
4) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
5) أدخل طلبك للموجه (+ و - لترجيح الموجه) وعبارات GLIGEN (في "" للمربع)
6) أدخل مربعات GLIGEN (مثل [0.1387, 0.2051, 0.4277, 0.7090] للمربع)
7) انقر على زر `Submit` للحصول على الصورة المولدة

#### animatediff:

1) أولاً قم بتحميل النماذج الخاصة بك إلى المجلد: *inputs/image/sd_models*
2) حدد النموذج الخاص بك من القائمة المنسدلة
3) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
4) أدخل طلبك (+ و - لترجيح الموجه)
5) انقر على زر `Submit` للحصول على الرسوم المتحركة للصورة المولدة

#### video:

1) قم بتحميل الصورة الأولية
2) أدخل طلبك (لـ IV2Gen-XL)
3) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
4) انقر على زر `Submit` للحصول على الفيديو من الصورة

#### ldm3d:

1) أدخل طلبك
2) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
3) انقر على زر `Submit` للحصول على الصور المولدة

#### sd3 (txt2img, img2img, controlnet, inpaint):

1) أدخل طلبك
2) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
3) انقر على زر `Submit` للحصول على الصورة المولدة

#### cascade:

1) أدخل طلبك
2) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
3) انقر على زر `Submit` للحصول على الصورة المولدة

#### extras:

1) قم بتحميل الصورة الأولية
2) حدد الخيارات التي تحتاجها
3) انقر على زر `Submit` للحصول على الصورة المعدلة

### Kandinsky (txt2img, img2img, inpaint):

1) أدخل الموجه الخاص بك
2) حدد نموذجًا من القائمة المنسدلة
3) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
4) انقر على `Submit` للحصول على الصورة المولدة

### Flux:

1) أدخل الموجه الخاص بك
2) حدد نموذجًا من القائمة المنسدلة
3) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
4) انقر على `Submit` للحصول على الصورة المولدة

### HunyuanDiT:

1) أدخل الموجه الخاص بك
2) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
3) انقر على `Submit` للحصول على الصورة المولدة

### Lumina-T2X:

1) أدخل الموجه الخاص بك
2) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
3) انقر على `Submit` للحصول على الصورة المولدة

### Kolors:

1) أدخل الموجه الخاص بك
2) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
3) انقر على `Submit` للحصول على الصورة المولدة

### AuraFlow:

1) أدخل الموجه الخاص بك
2) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
3) انقر على `Submit` للحصول على الصورة المولدة

### Würstchen:

1) أدخل الموجه الخاص بك
2) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
3) انقر على `Submit` للحصول على الصورة المولدة

### DeepFloydIF (txt2img, img2img, inpaint):

1) أدخل الموجه الخاص بك
2) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
3) انقر على `Submit` للحصول على الصورة المولدة

### PixArt:

1) أدخل الموجه الخاص بك
2) حدد النموذج من القائمة المنسدلة
3) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
4) انقر على `Submit` للحصول على الصورة المولدة

### ModelScope:

1) أدخل الموجه الخاص بك
2) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
3) انقر على `Submit` للحصول على الفيديو المولد

### ZeroScope 2:

1) أدخل الموجه الخاص بك
2) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
3) انقر على `Submit` للحصول على الفيديو المولد

### CogVideoX:

1) أدخل الموجه الخاص بك
2) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
3) انقر على `Submit` للحصول على الفيديو المولد

### Latte:

1) أدخل الموجه الخاص بك
2) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
3) انقر على `Submit` للحصول على الفيديو المولد

### TripoSR:

1) قم بتحميل الصورة الأولية
2) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
3) انقر على زر `Submit` للحصول على الكائن ثلاثي الأبعاد المولد

### StableFast3D:

1) قم بتحميل الصورة الأولية
2) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
3) انقر على زر `Submit` للحصول على الكائن ثلاثي الأبعاد المولد

### Shap-E:

1) أدخل طلبك أو قم بتحميل الصورة الأولية
2) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
3) انقر على زر `Submit` للحصول على الكائن ثلاثي الأبعاد المولد

### SV34D:

1) قم بتحميل الصورة الأولية (للثلاثي الأبعاد) أو الفيديو (للرباعي الأبعاد)
2) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
3) انقر على زر `Submit` للحصول على الفيديو ثلاثي الأبعاد المولد

### Zero123Plus:

1) قم بتحميل الصورة الأولية
2) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
3) انقر على زر `Submit` للحصول على الدوران ثلاثي الأبعاد المولد للصورة

### StableAudio:

1) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
2) أدخل طلبك
3) انقر على زر `Submit` للحصول على الصوت المولد

### AudioCraft:

1) حدد نموذجًا من القائمة المنسدلة
2) حدد نوع النموذج (`musicgen` أو `audiogen` أو `magnet`)
3) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
4) أدخل طلبك
5) (اختياري) قم بتحميل الصوت الأولي إذا كنت تستخدم نموذج `melody` 
6) انقر على زر `Submit` للحصول على الصوت المولد
#### اختياري: يمكنك تمكين `multiband diffusion` لتحسين الصوت المولد

### AudioLDM 2:

1) حدد نموذجًا من القائمة المنسدلة
2) قم بإعداد النموذج وفقًا للمعلمات التي تحتاجها
3) أدخل طلبك
4) انقر على زر `Submit` للحصول على الصوت المولد

### Demucs:

1) قم بتحميل الصوت الأولي للفصل
2) انقر على زر `Submit` للحصول على الصوت المفصول

### Gallery:

* هنا يمكنك عرض الملفات من دليل المخرجات

### ModelDownloader:

* هنا يمكنك تنزيل نماذج `LLM` و `StableDiffusion`. ما عليك سوى اختيار النموذج من القائمة المنسدلة والنقر على زر `Submit`
#### يتم تنزيل نماذج `LLM` هنا: *inputs/text/llm_models*
#### يتم تنزيل نماذج `StableDiffusion` هنا: *inputs/image/sd_models*

### Settings: 

* هنا يمكنك تغيير إعدادات التطبيق. في الوقت الحالي يمكنك فقط تغيير وضع `Share` إلى `True` أو `False`

### System: 

* هنا يمكنك رؤية مؤشرات أجهزة استشعار جهاز الكمبيوتر الخاص بك عن طريق النقر على زر `Submit`

### معلومات إضافية:

1) يتم حفظ جميع الإنشاءات في مجلد *outputs*
2) يمكنك الضغط على زر `Clear` لإعادة تعيين اختيارك
3) لإيقاف عملية التوليد، انقر على زر `Stop generation`
4) يمكنك إيقاف تشغيل التطبيق باستخدام زر `Close terminal`
5) يمكنك فتح مجلد *outputs* بالنقر على زر `Outputs`

## أين يمكنني الحصول على النماذج والأصوات؟

* يمكن الحصول على نماذج LLM من [HuggingFace](https://huggingface.co/models) أو من ModelDownloader داخل الواجهة 
* يمكن الحصول على نماذج StableDiffusion و vae و inpaint و embedding و lora من [CivitAI](https://civitai.com/models) أو من ModelDownloader داخل الواجهة
* يتم تنزيل نماذج StableAudioOpen و AudioCraft و AudioLDM 2 و TTS و Whisper و Wav2Lip و SunoBark و MoonDream2 و Upscale و GLIGEN و Depth و Pix2Pix و Controlnet و AnimateDiff و Videos و LDM3D و SD3 و InstantID و PhotoMaker و IP-Adapter-FaceID و Cascade و Rembg و Roop و CodeFormer و Real-ESRGAN و TripoSR و StableFast3D و Shap-E و SV34D و Zero123Plus و Demucs و Kandinsky و Flux و HunyuanDiT و Lumina-T2X و Kolors و AuraFlow و Würstchen و DeepFloydIF و PixArt و ModelScope و ZeroScope 2 و CogVideoX و Latte و Multiband diffusion تلقائيًا في مجلد *inputs* عند استخدامها 
* يمكنك الحصول على الأصوات من أي مكان. سجل صوتك أو خذ تسجيلًا من الإنترنت. أو فقط استخدم تلك الموجودة بالفعل في المشروع. الشيء الرئيسي هو أن يتم معالجتها مسبقًا!

## الويكي

* https://github.com/Dartvauder/NeuroSandboxWebUI/wiki

## شكر للمطورين

#### شكرًا جزيلاً لهذه المشاريع لأنه بفضل تطبيقاتهم/مكتباتهم، تمكنت من إنشاء تطبيقي:

أولاً وقبل كل شيء، أود أن أشكر مطوري [PyCharm](https://www.jetbrains.com/pycharm/) و [GitHub](https://desktop.github.com). بمساعدة تطبيقاتهم، تمكنت من إنشاء ومشاركة الكود الخاص بي

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
## تراخيص الطرف الثالث:

#### العديد من النماذج لها تراخيص خاصة بها للاستخدام. قبل استخدامها، أنصحك بالتعرف عليها:

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

## التبرع

### *إذا أعجبك مشروعي وتريد التبرع، فهنا خيارات للتبرع. شكرًا جزيلاً مقدمًا!*

* محفظة العملات المشفرة (BEP-20) - 0x3d86bdb5f50b92d0d7Eb44F1a833acC5e91aAEcA

* [!["اشترِ لي قهوة"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Dartvauder)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Dartvauder/NeuroSandboxWebUI&type=Date)](https://star-history.com/#Dartvauder/NeuroSandboxWebUI&Date)
