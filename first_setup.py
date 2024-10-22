import json
import os
import sys
from typing import Dict, Optional


def load_settings() -> Dict:
    try:
        with open('Settings.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Ошибка: Файл Settings.json не найден!")
        print("Error: Settings.json file not found!")
        print("错误：未找到Settings.json文件！")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Ошибка: Неверный формат файла Settings.json!")
        print("Error: Invalid Settings.json file format!")
        print("错误：Settings.json文件格式无效！")
        sys.exit(1)


def save_settings(settings: Dict) -> None:
    with open('Settings.json', 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)


def select_language() -> str:
    while True:
        print("\nВыберите язык / Select language / 选择语言:")
        print("1. English (EN)")
        print("2. Русский (RU)")
        print("3. 中文 (ZH)")

        choice = input("Введите номер / Enter number / 输入数字 (1-3): ").strip()

        language_map = {"1": "EN", "2": "RU", "3": "ZH"}
        if choice in language_map:
            return language_map[choice]
        print("\nНеверный выбор! / Invalid choice! / 选择无效！")


def input_hf_token() -> Optional[str]:
    print("\nВведите ваш Hugging Face токен (или нажмите Enter для пропуска):")
    print("Enter your Hugging Face token (or press Enter to skip):")
    print("输入您的 Hugging Face 令牌（或按 Enter 跳过）：")

    token = input().strip()
    if token and not token.startswith('hf_'):
        print("\nПредупреждение: Токен должен начинаться с 'hf_'")
        print("Warning: Token should start with 'hf_'")
        print("警告：令牌应该以 'hf_' 开头")
        return None
    return token if token else None


def check_first_run() -> bool:
    flag_file = ".setup_complete"
    if os.path.exists(flag_file):
        return False
    with open(flag_file, 'w') as f:
        f.write('1')
    return True


def main():
    if not check_first_run():
        return

    settings = load_settings()

    language = select_language()
    settings['language'] = language

    token = input_hf_token()
    if token:
        settings['hf_token'] = token

    save_settings(settings)

    print("\nНастройка завершена! / Setup complete! / 设置完成！")


if __name__ == "__main__":
    main()
