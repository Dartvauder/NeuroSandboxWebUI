#!/bin/bash

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${CURRENT_DIR}/venv/bin/activate"

echo "Virtual environment activated"
echo

while true; do
    echo "Available options:"
    echo "1. Install package"
    echo "2. Uninstall package"
    echo "3. Upgrade package"
    echo "4. List installed packages"
    echo "5. Show package details"
    echo "6. Check dependencies"
    echo "7. Debug information"
    echo "8. Exit"
    echo

    read -p "Enter your choice (1-8): " choice

    case $choice in
        1)
            read -p "Enter package name to install: " package
            pip install $package
            ;;
        2)
            read -p "Enter package name to uninstall: " package
            pip uninstall $package
            ;;
        3)
            read -p "Enter package name to upgrade: " package
            pip install --upgrade $package
            ;;
        4)
            pip list
            echo
            ;;
        5)
            read -p "Enter package name to show details: " package
            pip show $package
            echo
            ;;
        6)
            pip check
            echo
            ;;
        7)
            pip debug --verbose
            echo
            ;;
        8)
            deactivate
            exit 0
            ;;
        *)
            echo "Invalid choice. Please try again."
            ;;
    esac
    echo
done