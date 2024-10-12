import sys
import json
import os
from datetime import datetime
from stable_diffusion_cpp import StableDiffusion


def generate_images(params):
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, params['model_name'])

    stable_diffusion = StableDiffusion(
        model_path=model_path,
        wtype="default"
    )

    output = stable_diffusion.txt_to_img(
        prompt=params['prompt'],
        negative_prompt=params['negative_prompt'],
        cfg_scale=params['cfg_scale'],
        height=params['height'],
        width=params['width'],
        sample_steps=params['sample_steps'],
        seed=params['seed'],
        sample_method="euler"
    )

    image_paths = []
    for i, image in enumerate(output):
        today = datetime.now().date()
        image_dir = os.path.join('outputs', f"StableDiffusion_{today.strftime('%Y%m%d')}")
        os.makedirs(image_dir, exist_ok=True)
        image_filename = f"sd-txt2img-quantize_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.png"
        image_path = os.path.join(image_dir, image_filename)

        image.save(image_path, format="png")
        image_paths.append(image_path)

    return image_paths


if __name__ == "__main__":
    try:
        params = json.loads(sys.argv[1])
        image_paths = generate_images(params)
        for path in image_paths:
            print(f"IMAGE_PATH:{path}")
    except Exception as e:
        print(f"Error occurred: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
