# PowerShell helper to download accessible assets for the ML contest
param()

$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $root

New-Item -ItemType Directory -Force -Path downloads | Out-Null
New-Item -ItemType Directory -Force -Path wheels | Out-Null

Write-Host "Downloading FER-2013 CSV (public mirror) into downloads/fer2013.csv"
$ferUrl = 'https://raw.githubusercontent.com/atulapra/Emotion-detection/master/fer2013.csv'
try {
    Invoke-WebRequest -Uri $ferUrl -OutFile "downloads\fer2013.csv" -UseBasicParsing -ErrorAction Stop
    Write-Host "Downloaded fer2013.csv"
} catch {
    Write-Warning "Failed to download fer2013.csv from $ferUrl. Please download manually and place in downloads/."
}

Write-Host "Attempting to download a small VOSK Russian model into downloads/"
$voskUrl = 'https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip'
try {
    Invoke-WebRequest -Uri $voskUrl -OutFile "downloads\vosk-model-small-ru-0.22.zip" -UseBasicParsing -ErrorAction Stop
    Write-Host "Downloaded vosk model"
} catch {
    Write-Warning "Failed to download VOSK model from $voskUrl. Please download manually from https://alphacephei.com/vosk/models and place into downloads/."
}

Write-Host "Using pip to download wheels for packages in requirements.txt into wheels/"
try {
    python -m pip download -r ..\requirements.txt -d wheels --only-binary=:all: -q
    Write-Host "pip download finished (check wheels/). Note: torch/torchvision may not be available via pip for your platform — download from pytorch.org and add to wheels/."
} catch {
    Write-Warning "pip download failed or partial. Try running: python -m pip download -r ..\requirements.txt -d wheels"
}

Write-Host "Done. Check downloads/ and wheels/ folders."
