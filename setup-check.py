#!/usr/bin/env python3
"""
Setup verification script for SGLang Mistral client.

This script checks if the environment is properly configured and dependencies are available.
"""

import os
import sys
from pathlib import Path

def check_env_file():
    """Check if .env file exists and contains required variables."""
    env_file = Path(".env")
    env_example = Path(".env.example")

    print("🔍 Checking environment configuration...")

    if not env_example.exists():
        print("❌ .env.example file not found!")
        return False

    if not env_file.exists():
        # Check if HF_TOKEN is available as global environment variable
        global_hf_token = os.getenv("HF_TOKEN", "")
        if global_hf_token and global_hf_token != "your_huggingface_token_here":
            print("✅ .env file not found, but HF_TOKEN is available globally")
            return True
        else:
            print("⚠️  .env file not found. Please run: cp .env.example .env")
            print("   Then edit .env with your actual HuggingFace token.")
            print("   Alternatively, set HF_TOKEN as a global environment variable.")
            return False

    # Check if .env has the required HF_TOKEN
    try:
        with open(env_file) as f:
            content = f.read()
            if "HF_TOKEN=your_huggingface_token_here" in content:
                # Check if HF_TOKEN is available as global environment variable
                global_hf_token = os.getenv("HF_TOKEN", "")
                if global_hf_token and global_hf_token != "your_huggingface_token_here":
                    print("✅ .env has placeholder value, but HF_TOKEN is available globally")
                    return True
                else:
                    print("⚠️  HF_TOKEN still has placeholder value.")
                    print("   Please edit .env and set your actual HuggingFace token.")
                    print("   Alternatively, set HF_TOKEN as a global environment variable.")
                    return False
            elif "HF_TOKEN=" not in content:
                # Check if HF_TOKEN is available as global environment variable
                global_hf_token = os.getenv("HF_TOKEN", "")
                if global_hf_token and global_hf_token != "your_huggingface_token_here":
                    print("✅ .env missing HF_TOKEN, but it's available globally")
                    return True
                else:
                    print("⚠️  HF_TOKEN not found in .env file.")
                    print("   Alternatively, set HF_TOKEN as a global environment variable.")
                    return False
    except Exception as e:
        print(f"❌ Error reading .env file: {e}")
        return False

    print("✅ Environment file configured")
    return True

def check_dependencies():
    """Check if required dependencies are available."""
    print("\n🔍 Checking dependencies...")

    try:
        import requests
        del requests  # Just checking if it can be imported
        print("✅ requests library available")
    except ImportError:
        print("❌ requests library not found. Run: uv sync")
        return False

    try:
        from dotenv import load_dotenv
        del load_dotenv  # Just checking if it can be imported
        print("✅ python-dotenv library available")
    except ImportError:
        print("⚠️  python-dotenv not found (optional)")

    return True

def check_script():
    """Check if the main script works."""
    print("\n🔍 Checking main script...")

    try:
        # Try to import the package modules
        from sglang_mistral import client, message, cli
        print("✅ sglang_mistral package imports successfully")

        # Check if main function exists
        if hasattr(cli, 'main'):
            print("✅ main function found")
        else:
            print("⚠️  main function not found")

        # Check if key classes/functions exist
        if hasattr(client, 'SGLangMistralClient'):
            print("✅ SGLangMistralClient class found")
        if hasattr(message, 'create_message'):
            print("✅ create_message function found")

        return True
    except Exception as e:
        print(f"❌ Error importing sglang_mistral package: {e}")
        return False

def check_environment_variables():
    """Check environment variable loading."""
    print("\n🔍 Checking environment variables...")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    host = os.getenv("SGLANG_HOST", "localhost")
    port = os.getenv("SGLANG_PORT", "30000")
    hf_token = os.getenv("HF_TOKEN", "")

    print(f"📍 SGLANG_HOST: {host}")
    print(f"📍 SGLANG_PORT: {port}")

    if hf_token and hf_token != "your_huggingface_token_here":
        print("✅ HF_TOKEN is configured")
    else:
        print("⚠️  HF_TOKEN not configured or using placeholder")
        return False

    return True

def main():
    """Run all checks."""
    print("🚀 SGLang Mistral Setup Verification")
    print("=" * 40)

    checks = [
        check_env_file,
        check_dependencies,
        check_script,
        check_environment_variables
    ]

    all_passed = True
    for check in checks:
        if not check():
            all_passed = False

    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All checks passed! Your setup is ready.")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
