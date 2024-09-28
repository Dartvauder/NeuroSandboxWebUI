import sys
import json
from stable_diffusion_cpp import StableDiffusion
import os
from datetime import datetime


def generate_image(params):
    model_dir = os.path.dirname(os.path.abspath(__file__))

    diffusion_model_path = os.path.join(model_dir, params.get('quantize_flux_model_path'))
    clip_l_path = os.path.join(model_dir, "clip_l.safetensors")
    t5xxl_path = os.path.join(model_dir, "t5xxl_fp16.safetensors")
    vae_path = os.path.join(model_dir, "ae.safetensors")

    stable_diffusion = StableDiffusion(
        diffusion_model_path=diffusion_model_path,
        clip_l_path=clip_l_path,
        t5xxl_path=t5xxl_path,
        vae_path=vae_path,
        wtype="default"
    )

    output = stable_diffusion.img_to_img(
        prompt=params['prompt'],
        cfg_scale=params['guidance_scale'],
        height=params['height'],
        width=params['width'],
        sample_steps=params['num_inference_steps'],
        seed=params['seed'],
        image=params['init_image'],
        strength=params['strength'],
        sample_method="euler"
    )

    output_dir = os.path.join('outputs', f"Flux_{datetime.now().strftime('%Y%m%d')}")
    os.makedirs(output_dir, exist_ok=True)
    image_filename = f"flux-img2img-quantize_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    image_path = os.path.join(output_dir, image_filename)

    output[0].save(image_path, format="PNG")
    return image_path


if __name__ == "__main__":
    try:
        params = json.loads(sys.argv[1])
        image_path = generate_image(params)
        print(f"IMAGE_PATH:{image_path}")
    except Exception as e:
        print(f"Error occurred: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
