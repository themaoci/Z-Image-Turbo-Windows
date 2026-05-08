# Z-Image Turbo - One-Click Windows Installer (Low VRAM)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Platform: Windows](https://img.shields.io/badge/Platform-Windows-0078D6?logo=windows)](https://www.microsoft.com/windows)
[![Built with Gradio](https://img.shields.io/badge/Built%20with-Gradio-FFD21F?logo=gradio)](https://gradio.app)
[![Low VRAM Support](https://img.shields.io/badge/VRAM-4GB+-green)](https://github.com/leejet/stable-diffusion.cpp)

A beginner-friendly Windows package to run **Z-Image Turbo (GGUF)** locally with a simple **Gradio Web UI**.

Target users:

- Low-VRAM NVIDIA GPUs (including 4GB)
- Anyone who wants free local image generation without complex tools

## Table of Contents
- [Features](#features)
- [Quickstart](#quickstart)
- [Requirements](#requirements)
- [Running the Setup](#running-the-setup)
- [Why the Backend Files Are Manual](#why-the-backend-files-are-manual)
- [Where to Get the Executable (Windows)](#where-to-get-the-executable-windows)
- [NVIDIA GPU / CUDA Notes](#nvidia-gpu--cuda-notes)
- [LoRA Support](#lora-support)
- [Downloads](#what-the-installer-downloads-and-what-is-manual)
- [Manual Download Sources](#manual-download-sources)
- [Troubleshooting](#troubleshooting)
- [Credits](#credits--upstream)

## Features

- One-click installer: `start_zimage.bat`
- Creates an isolated Python `venv` automatically
- Downloads required model weights (GGUF) automatically
- Gradio Web UI with prompt, resolution, seed, CFG, timer, stop button, and LoRA controls
- LoRA support through `models\loras\`
- Safety-first: does **not** auto-download executables (`.exe`)

## Quickstart

1. Download / clone this repo.
2. Put the stable-diffusion.cpp Windows files in `sd_bin\` (see instructions below).
3. Double-click `start_zimage.bat`.
4. Open the UI:
   - http://127.0.0.1:9000

## Requirements

- Windows 10/11 (64-bit)
- Python 3.10+
- Microsoft Visual C++ Redistributable 2015-2022 (x64)
- NVIDIA GPU users (optional)
  - Latest NVIDIA driver recommended

## Running the setup

Double-click:

- `start_zimage.bat`

The installer will:

- Create a Python virtual environment (`venv\`)
- Ask you to choose a VRAM tier (4GB / 6-8GB / 10GB+)
- Download the required weights
- Launch the Gradio UI at http://127.0.0.1:9000

Keep the terminal window open while it downloads models.

## Why the backend files are manual

This project **will never download executable (.exe) files automatically**.

The app uses **stable-diffusion.cpp** as the inference backend. Older releases used one main executable named `sd.exe`. Newer releases split this into files such as `sd-cli.exe`, `sd-server.exe`, and `stable-diffusion.dll`.

If you followed an older tutorial for this project that says to copy `sd.exe`, that was correct for the old stable-diffusion.cpp release. For current releases, use `sd-cli.exe` instead. The installer and UI will automatically detect either `sd-cli.exe` or legacy `sd.exe`.

## Where to get the executable (Windows)

Download a Windows build from the **stable-diffusion.cpp Releases** page.

Recommended assets (names include a commit/hash):

- NVIDIA executable package: `sd-...-bin-win-cuda12-x64.zip`
- NVIDIA CUDA runtime/DLL package: `cudart-sd-bin-win-cu12-x64.zip`
- CPU only: `sd-...-bin-win-x64.zip`

For NVIDIA GPUs, use the CUDA 12 Windows x64 build. Recent builds may look like:

- `sd-master-90e87bc-bin-win-cuda12-x64.zip`
- `cudart-sd-bin-win-cu12-x64.zip`

The `sd-...-bin-win-cuda12-x64.zip` file usually contains the stable-diffusion.cpp program files, such as:

- `sd-cli.exe`
- `sd-server.exe`
- `stable-diffusion.dll`

The larger `cudart-sd-bin-win-cu12-x64.zip` file contains the CUDA runtime DLLs required by the CUDA build. It commonly includes files such as CUDA/cuBLAS DLLs. Copy those DLL files into `sd_bin\` too.

Install steps:

1. Download the latest `sd-...-bin-win-cuda12-x64.zip`.
2. Extract it and copy these files into `sd_bin\`:
   - `sd-cli.exe`
   - `sd-server.exe`
   - `stable-diffusion.dll`
3. Download the matching `cudart-sd-bin-win-cu12-x64.zip`.
4. Extract it and copy its `*.dll` files into `sd_bin\`.
5. If you are using an older stable-diffusion.cpp build that only has `sd.exe`, copy it to `sd_bin\sd.exe`.

Important:

- New users should copy both the executable package files and the CUDA runtime DLL package files.
- Existing users who already have the older CUDA DLLs may only need to replace `sd-cli.exe`, `sd-server.exe`, and `stable-diffusion.dll` from the newer executable package.
- If CUDA fails, crashes, or your dedicated GPU is not used, refresh the CUDA runtime DLLs from the matching `cudart-sd-bin-win-cu12-x64.zip`.
- Do not mix `stable-diffusion.dll` from one release with `sd-cli.exe` from another release.

## NVIDIA GPU / CUDA notes

If generation works but your dedicated NVIDIA GPU is not being used, your stable-diffusion.cpp files are probably too old or you copied a CPU-only build.

Use the latest Windows CUDA 12 x64 build from stable-diffusion.cpp. For example, recent working assets are:

- `sd-master-90e87bc-bin-win-cuda12-x64.zip`
- `cudart-sd-bin-win-cu12-x64.zip`

After downloading it:

1. Stop the Gradio app.
2. Replace `sd-cli.exe`, `sd-server.exe`, and `stable-diffusion.dll` in `sd_bin\` from the `sd-...-bin-win-cuda12-x64.zip` file.
3. If you are setting up fresh, or if CUDA still does not work, also copy the DLLs from `cudart-sd-bin-win-cu12-x64.zip` into `sd_bin\`.
4. Start the app again with `start_zimage.bat`.

The important migration point for old users is this: the old tutorial used `sd.exe` because stable-diffusion.cpp used to ship that way. New stable-diffusion.cpp releases use `sd-cli.exe` plus `stable-diffusion.dll`, so updating those files is what fixes many NVIDIA GPU detection/utilization problems.

## LoRA support

The setup now automatically creates this folder:

- `models\loras\`

To use a LoRA:

1. Download a Z-Image-compatible LoRA from Civitai or use your own trained LoRA.
2. Put the `.safetensors` file in:
   - `models\loras\`
3. Start or refresh the UI.
4. In the LoRA section, click **Refresh** if needed.
5. Check the LoRA you want to use and set the LoRA strength.
6. Generate as usual.

The UI passes selected LoRAs to stable-diffusion.cpp using prompt tags and the configured LoRA model directory.

## What the installer downloads (and what is manual)

Automatic (safe, non-executable downloads):

- Z-Image Turbo GGUF (diffusion model)
- Qwen GGUF (LLM/text encoder)

Manual:

- stable-diffusion.cpp backend files:
  - Current NVIDIA build: `sd-cli.exe`, `sd-server.exe`, `stable-diffusion.dll`
  - CUDA runtime DLLs from `cudart-sd-bin-win-cu12-x64.zip`
  - Legacy builds: `sd.exe`
- VAE: `models\vae\ae.safetensors`
  - This file may require a Hugging Face login, so the installer asks you to download it manually.

Manual download sources:

- Z-Image Turbo GGUF:
  - https://huggingface.co/leejet/Z-Image-Turbo-GGUF/tree/main
- VAE (`ae.safetensors`):
  - https://huggingface.co/black-forest-labs/FLUX.1-schnell/tree/main
- Qwen GGUF:
  - https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/tree/main

## Troubleshooting

If generation fails or the executable crashes:

- For current CUDA builds, make sure `sd-cli.exe`, `sd-server.exe`, and `stable-diffusion.dll` came from the same `sd-...-bin-win-cuda12-x64.zip` package.
- For fresh installs, also copy the CUDA runtime DLLs from `cudart-sd-bin-win-cu12-x64.zip`.
- If you upgraded from an older working setup and only GPU usage was broken, replacing `sd-cli.exe`, `sd-server.exe`, and `stable-diffusion.dll` may be enough.
- If CUDA still fails, crashes, or your NVIDIA GPU is not used, refresh the CUDA runtime DLLs from `cudart-sd-bin-win-cu12-x64.zip`.
- Install Microsoft Visual C++ Redistributable 2015-2022 (x64).
- If the CUDA build fails, try the CPU build to confirm everything else works.
- Common crash code:
  - `3221225781` (`0xC0000135`) typically means a missing DLL/runtime dependency.

If model downloads fail in the installer but the same URL works in your browser:

- Your network (proxy/firewall/antivirus) may block programmatic downloads from Hugging Face/CDN.
- The installer prefers `curl.exe` (resume + retries + progress bar). If that is unavailable, it falls back to `Invoke-WebRequest`.
- If it still fails, use the manual download links above and place the files into the indicated `models\...` folders.

## Credits / Upstream

This project is a Windows-friendly wrapper around the excellent **stable-diffusion.cpp** backend:

- https://github.com/leejet/stable-diffusion.cpp

Z-Image weights and related resources are hosted on Hugging Face by their respective authors.

