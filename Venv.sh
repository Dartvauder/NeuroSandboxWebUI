#!/bin/bash

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${CURRENT_DIR}/venv/bin/activate"

echo "Virtual environment activated"
echo

while true; do
    echo "Available options:"
    echo "1. Install package"
    echo "2. Uninstall package"
    echo "3. Exit"
    echo

    read -p "Enter your choice (1-4): " choice

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
            deactivate
            exit 0
            ;;
        *)
            echo "Invalid choice. Please try again."
            ;;
    esac
    echo
done
