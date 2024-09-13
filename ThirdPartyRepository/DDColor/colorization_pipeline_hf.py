import argparse
import cv2
import os
from tqdm import tqdm
import torch
from basicsr.archs.ddcolor_arch import DDColor

from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from inference.colorization_pipeline import ImageColorizationPipeline


class DDColorHF(DDColor, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__(**config)


class ImageColorizationPipelineHF(ImageColorizationPipeline):
    def __init__(self, model, input_size):
        self.input_size = input_size
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device)
        self.model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ddcolor_modelscope")
    parser.add_argument(
        "--input",
        type=str,
        default="figure/",
        help="input test image folder or single image path",
    )
    parser.add_argument(
        "--output", type=str, default="results", help="output folder"
    )
    parser.add_argument(
        "--input_size", type=int, default=512, help="input size for model"
    )

    args = parser.parse_args()

    ddcolor_model = DDColorHF.from_pretrained(f"piddnad/{args.model_name}")

    print(f"Output path: {args.output}")
    os.makedirs(args.output, exist_ok=True)

    if os.path.isdir(args.input):
        img_list = [f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        img_list = [os.path.join(args.input, f) for f in img_list]
    elif os.path.isfile(args.input):
        img_list = [args.input]
    else:
        raise ValueError(f"Input path {args.input} is neither a file nor a directory")

    assert len(img_list) > 0, "No valid image files found"

    colorizer = ImageColorizationPipelineHF(
        model=ddcolor_model, input_size=args.input_size
    )

    for img_path in tqdm(img_list):
        img = cv2.imread(img_path)
        image_out = colorizer.process(img)
        output_filename = os.path.basename(img_path)
        cv2.imwrite(os.path.join(args.output, output_filename), image_out)


if __name__ == "__main__":
    main()
