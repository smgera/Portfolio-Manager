#!/usr/bin/env python3
"""
Installation script for Portfolio Manager.

This script installs the package in editable mode, which is recommended
for development as changes are immediately available.
"""

import subprocess
import sys
import os

def main():
    """Install the package in editable mode."""
    print("Portfolio Manager - Installation Script")
    print("=" * 50)
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("setup.py"):
        print("Error: setup.py not found. Please run from the project root.")
        sys.exit(1)
    
    print("Installing Portfolio Manager in editable mode...")
    print("This allows changes to be immediately available without reinstallation.")
    print()
    
    # Run pip install -e .
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            check=True,
            capture_output=True,
            text=True
        )
        print("✓ Installation successful!")
        print()
        print("To run the application:")
        print("  python Portmanv2.py")
        print()
        print("Or use the console command:")
        print("  portfolio-manager")
        
    except subprocess.CalledProcessError as e:
        print("✗ Installation failed!")
        print()
        print("Error output:")
        print(e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
