import os
import fileinput


def modify_degradations_file():
    file_path = os.path.join('venv', 'Lib', 'site-packages', 'basicsr', 'data', 'degradations.py')

    if os.path.exists(file_path):
        with fileinput.FileInput(file_path, inplace=True) as file:
            for line in file:
                print(line.replace(
                    'from torchvision.transforms.functional_tensor import rgb_to_grayscale',
                    'from torchvision.transforms.functional import rgb_to_grayscale'
                ), end='')
        print(f"File {file_path} has been modified successfully.")
    else:
        print(f"File {file_path} not found.")


if __name__ == "__main__":
    modify_degradations_file()