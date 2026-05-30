# setup_and_run.ps1 - One click setup for Z-Image Turbo (GGUF) with minimal UI
# Place this file in ZImage-Windows and double-click start_zimage.bat to run.

Write-Host '=== Z-Image Turbo: One-Click (4/6/10GB) - Low VRAM UI ==='
Write-Host ''

# 0. Basic checks
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python not found. Install Python 3.10+ from https://python.org and re-run."
    exit 1
}

function Download-FileWithProgress {
    param(
        [Parameter(Mandatory=$true)][string]$Url,
        [Parameter(Mandatory=$true)][string]$Destination,
        [Parameter(Mandatory=$true)][string]$Label
    )

    try {
        # Hugging Face downloads can fail on older TLS defaults
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    } catch {
        # ignore
    }

    $dir = Split-Path -Parent $Destination
    if (!(Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }

    try {
        $curlCmd = Get-Command curl.exe -ErrorAction SilentlyContinue
        if ($curlCmd -and $curlCmd.Source) {
            $curl = $curlCmd.Source
            Write-Host ("{0} - downloading via curl (with resume/retry)..." -f $Label)

            $args = @(
                '--location',
                '--fail',
                '--retry', '10',
                '--retry-delay', '5',
                '--retry-all-errors',
                '--connect-timeout', '30',
                '--speed-time', '30',
                '--speed-limit', '10240',
                '--progress-bar',
                '--header', 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                '--header', 'Accept: */*'
            )

            if (Test-Path $Destination) {
                $args += @('--continue-at', '-')
            }

            $args += @('--output', $Destination, $Url)

            & $curl @args
            if ($LASTEXITCODE -ne 0) {
                throw "curl failed with exit code $LASTEXITCODE"
            }
            return
        }
    } catch {
        Write-Host "`nPrimary download method (curl) failed. Falling back..."
        Write-Host ("Reason: {0}" -f $_.Exception.Message)
    }

    try {
        Write-Host ("{0} - downloading via Invoke-WebRequest (fallback)..." -f $Label)
        $headers = @{
            'User-Agent' = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            'Accept' = '*/*'
        }
        Invoke-WebRequest -Uri $Url -OutFile $Destination -Headers $headers -MaximumRedirection 10 -ProxyUseDefaultCredentials
        return
    } catch {
        $msg = $_.Exception.Message
        try {
            if ($_.Exception.Response -and $_.Exception.Response.StatusCode) {
                $msg = "HTTP {0} - {1}" -f [int]$_.Exception.Response.StatusCode, $_.Exception.Response.StatusDescription
            }
        } catch {
            # ignore
        }
        throw $msg
    }
}

# 1. Create folders
$root = $PSScriptRoot
$sdBin = Join-Path $root "sd_bin"
$modelsDir = Join-Path $root "models"
$zimageDir = Join-Path $modelsDir "zimage"
$vaeDir = Join-Path $modelsDir "vae"
$llmDir = Join-Path $modelsDir "llm"
$loraDir = Join-Path $modelsDir "loras"
if (!(Test-Path $sdBin)) { New-Item -ItemType Directory -Path $sdBin | Out-Null }
if (!(Test-Path $modelsDir)) { New-Item -ItemType Directory -Path $modelsDir | Out-Null }
if (!(Test-Path $zimageDir)) { New-Item -ItemType Directory -Path $zimageDir | Out-Null }
if (!(Test-Path $vaeDir)) { New-Item -ItemType Directory -Path $vaeDir | Out-Null }
if (!(Test-Path $llmDir)) { New-Item -ItemType Directory -Path $llmDir | Out-Null }
if (!(Test-Path $loraDir)) { New-Item -ItemType Directory -Path $loraDir | Out-Null }

Write-Host 'Folders prepared:'
Write-Host (" - sd_bin  : {0}" -f $sdBin)
Write-Host (" - models/zimage  : {0}" -f $zimageDir)
Write-Host (" - models/vae  : {0}" -f $vaeDir)
Write-Host (" - models/llm  : {0}" -f $llmDir)
Write-Host ''

# 2. Ask user about VRAM tier
Write-Host 'Choose your GPU VRAM tier (pick the number):'
Write-Host ' 1) 4 GB  (Fastest, smallest model, recommended for RTX 3050 4GB)'
Write-Host ' 2) 6-8 GB  (Better quality)'
Write-Host ' 3) 10+ GB  (Highest quality - not recommended for 4GB)'
$choice = Read-Host 'Enter 1, 2 or 3'

switch ($choice) {
    "1" {
        $moshort = "4GB"
        $model_name = "z_image_turbo_Q4_0.gguf"
        # Example public URL placeholder - replace if you prefer another source.
        $model_url = "https://huggingface.co/leejet/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q4_0.gguf"
    }
    "2" {
        $moshort = "6-8GB"
        $model_name = "z_image_turbo_Q6_K.gguf"
        $model_url = "https://huggingface.co/leejet/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q6_K.gguf"
    }
    "3" {
        $moshort = "10+GB"
        $model_name = "z_image_turbo_Q8_0.gguf"
        $model_url = "https://huggingface.co/leejet/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q8_0.gguf"
    }
    default {
        Write-Host "Invalid choice. Exiting."
        exit 1
    }
}

Write-Host ''
Write-Host ("You picked: {0}" -f $moshort)
Write-Host ("Model will be saved as: {0}" -f $model_name)
Write-Host ''

# 3. Create venv (if missing)
$venv = Join-Path $root "venv"
if (!(Test-Path $venv)) {
    Write-Host "Creating Python virtual environment..."
    python -m venv venv
} else {
    Write-Host "Virtual environment already exists (venv/)."
}

# 4. Use venv python directly (avoids PowerShell execution policy issues with Activate.ps1)
$venvPython = Join-Path $venv "Scripts\python.exe"
if (!(Test-Path $venvPython)) {
    Write-Host "ERROR: venv python not found at: $venvPython"
    exit 1
}

# 5. Upgrade pip safely
Write-Host "Upgrading pip..."
& $venvPython -m pip install --upgrade pip

# 6. Install Python deps for minimal UI
Write-Host 'Installing Python requirements (gradio, requests)...'
& $venvPython -m pip install gradio requests tqdm

# 7. Check for sd binary (sd-cli.exe or sd.exe)
$sdCliExe = Join-Path $sdBin "sd-cli.exe"
$sdOldExe = Join-Path $sdBin "sd.exe"

if ((Test-Path $sdCliExe)) {
    Write-Host "Found sd-cli.exe (recommended)"
    $sdexe = $sdCliExe
} elseif ((Test-Path $sdOldExe)) {
    Write-Host "Found sd.exe (legacy)"
    $sdexe = $sdOldExe
} else {
    Write-Host ""
    Write-Host "IMPORTANT: A stable-diffusion.cpp Windows binary is REQUIRED to run the model."
    Write-Host "Please download from the official stable-diffusion.cpp releases:"
    Write-Host "    https://github.com/leejet/stable-diffusion.cpp/releases"
    Write-Host ""
    Write-Host "Recommended: Extract 'sd-cli.exe' (or 'sd.exe' from older releases) to:"
    Write-Host "    $sdBin"
    Write-Host ""
    Write-Host "Note: Recent releases use 'sd-cli.exe'. Older releases use 'sd.exe'. Either will work."
    Write-Host ""
    Write-Host "Press Enter after you have placed the executable, or Ctrl+C to exit."
    Read-Host
}

if (!(Test-Path $sdexe)) {
    Write-Host "Executable still not found in $sdBin. Exiting."
    exit 1
}

# 7b. Sanity-check executable (common crash is missing DLL / wrong build)
Write-Host "`nChecking executable..."
try {
    & $sdexe --help | Out-Null
} catch {
    # swallow - we will check exit code below
}
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Executable failed to start (exit code: $LASTEXITCODE)."
    Write-Host "This usually means a missing dependency or wrong build." 
    Write-Host "" 
    Write-Host "Please check:" 
    Write-Host " 1) You extracted the release ZIP and copied the executable AND any .dll files into:"
    Write-Host "    $sdBin"
    Write-Host " 2) Microsoft Visual C++ Redistributable 2015-2022 (x64) is installed"
    Write-Host " 3) If you downloaded a CUDA build, your NVIDIA driver supports that CUDA version"
    Write-Host " 4) Try the CPU-only ZIP (sd-...-bin-win-x64.zip) to confirm it works on your PC"
    Write-Host ""
    Write-Host "Press Enter to exit."
    Read-Host
    exit 1
}

# 8. Download the chosen quantized GGUF model if it does not exist
$dest = Join-Path $zimageDir $model_name
if (Test-Path $dest) {
    Write-Host "Model already exists: $dest"
} else {
    Write-Host "`nDownloading quantized model (this can be several GB)."
    Write-Host "Source URL (if it fails, open link in browser and download manually):"
    Write-Host "  $model_url`n"
    try {
        Download-FileWithProgress -Url $model_url -Destination $dest -Label ("Downloading Z-Image model: {0}" -f $model_name)
        Write-Host "Downloaded model to: $dest"
    } catch {
        Write-Host "Automatic download failed. Please download the file manually and place it into:"
        Write-Host "   $dest"
        Write-Host "Then press Enter to continue."
        Read-Host
        if (!(Test-Path $dest)) {
            Write-Host "Model not found. Exiting."
            exit 1
        }
    }
}

# 9. Download VAE + LLM (required by Z-Image pipeline)
$vaeName = "ae.safetensors"
$vaeUrl = "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors"
$vaePath = Join-Path $vaeDir $vaeName

$llmName = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$llmUrl = "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$llmPath = Join-Path $llmDir $llmName

if (Test-Path $vaePath) {
    Write-Host "VAE already exists: $vaePath"
} else {
    Write-Host "`nVAE is required but may be restricted for non-logged-in downloads on Hugging Face."
    Write-Host "Please download it manually (login may be required):"
    Write-Host "  $vaeUrl"
    Write-Host "Save it to:"
    Write-Host "  $vaePath"
    Write-Host "`nPress Enter after you have placed ae.safetensors, or Ctrl+C to exit."
    Read-Host
    if (!(Test-Path $vaePath)) {
        Write-Host "VAE not found. Exiting."
        exit 1
    }
}

if (Test-Path $llmPath) {
    Write-Host "LLM already exists: $llmPath"
} else {
    Write-Host "`nDownloading LLM (Qwen): $llmName"
    Write-Host "Source URL (if it fails, open link in browser and download manually):"
    Write-Host "  $llmUrl`n"
    try {
        Download-FileWithProgress -Url $llmUrl -Destination $llmPath -Label ("Downloading Qwen LLM: {0}" -f $llmName)
        Write-Host "Downloaded LLM to: $llmPath"
    } catch {
        Write-Host "Automatic download failed. Please download the file manually and place it into:"
        Write-Host "   $llmPath"
        Write-Host "Then press Enter to continue."
        Read-Host
        if (!(Test-Path $llmPath)) {
            Write-Host "LLM not found. Exiting."
            exit 1
        }
    }
}

# 10. Use the checked-in Gradio UI script
$uiScript = Join-Path $root "run_gradio_ui.py"
if (!(Test-Path $uiScript)) {
    Write-Host "run_gradio_ui.py not found. Please make sure this file exists in the project folder."
    exit 1
}
$env:ZIMAGE_MODEL_NAME = $model_name
$selectedModelPath = Join-Path $zimageDir "selected_model.txt"
$model_name | Out-File -Encoding ascii $selectedModelPath
Write-Host "Using run_gradio_ui.py with model: $model_name"
# 11. Run the UI
Write-Host "`nStarting the minimal UI (Gradio) at http://127.0.0.1:9000"
Write-Host "Press Ctrl+C in this window to stop."
& $venvPython (Join-Path $root "run_gradio_ui.py")

