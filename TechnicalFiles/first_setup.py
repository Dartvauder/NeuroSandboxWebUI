import json
import os
import sys
from typing import Dict, Optional, Tuple


def load_settings() -> Dict:
    try:
        with open('TechnicalFiles/Settings.json', 'r', encoding='utf-8') as f:
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
    with open('TechnicalFiles/Settings.json', 'w', encoding='utf-8') as f:
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


def select_auto_launch() -> bool:
    while True:
        print("\nВключить автозапуск? / Enable auto-launch? / 启用自动启动？")
        print("1. Да / Yes / 是")
        print("2. Нет / No / 否")

        choice = input("Введите номер / Enter number / 输入数字 (1-2): ").strip()

        if choice == "1":
            return True
        elif choice == "2":
            return False
        print("\nНеверный выбор! / Invalid choice! / 选择无效！")


def input_auth_credentials() -> Tuple[str, str]:
    print("\nВведите логин и пароль (формат login:password) или нажмите Enter для пропуска:")
    print("Enter login and password (format login:password) or press Enter to skip:")
    print("输入登录名和密码（格式 login:password）或按 Enter 跳过：")

    credentials = input().strip()

    if not credentials:
        return "admin", "admin"

    try:
        username, password = credentials.split(':')
        if username and password:
            return username.strip(), password.strip()
        raise ValueError
    except ValueError:
        print("\nНеверный формат! Используется значение по умолчанию (admin:admin)")
        print("Invalid format! Using default value (admin:admin)")
        print("格式无效！使用默认值 (admin:admin)")
        return "admin", "admin"


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

    auto_launch = select_auto_launch()
    settings['auto_launch'] = auto_launch

    username, password = input_auth_credentials()
    settings['auth']['username'] = username
    settings['auth']['password'] = password

    token = input_hf_token()
    if token:
        settings['hf_token'] = token

    save_settings(settings)

    print("\nНастройка завершена! / Setup complete! / 设置完成！")


if __name__ == "__main__":
    main()
