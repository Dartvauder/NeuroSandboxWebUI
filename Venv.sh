#!/bin/bash

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${CURRENT_DIR}/venv/bin/activate"

echo "Virtual environment activated"
echo

while true; do
    echo "Available options:"
    echo "1. Install package"
    echo "2. Uninstall package"
    echo "3. Delete application"
    echo "4. Exit"
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
            read -p "Are you sure you want to delete the application? (y/n): " confirm
            if [[ $confirm == [Yy]* ]]; then
                echo "Deleting application..."
                cd .. || exit 1
                rm -rf "${CURRENT_DIR}"
                echo "Application deleted successfully."
                exit 0
            fi
            ;;
        4)
            deactivate
            exit 0
            ;;
        *)
            echo "Invalid choice. Please try again."
            ;;
    esac
    echo
done
