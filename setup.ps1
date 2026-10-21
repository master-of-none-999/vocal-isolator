param(
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectDir

function Find-Python311 {
    $Launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($Launcher) {
        $Found = & $Launcher.Source -3.11 -c "import sys; print(sys.executable)"
        if ($LASTEXITCODE -eq 0 -and $Found) {
            return $Found
        }
    }
    $Fallback = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"
    if (Test-Path -LiteralPath $Fallback) {
        return $Fallback
    }
    throw "Python 3.11 was not found. Install Python 3.11, then run setup.ps1 again."
}

$Python = Find-Python311
$VenvDir = Join-Path $ProjectDir ".venv"
$VenvPython = Join-Path (Join-Path $VenvDir "Scripts") "python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    & $Python -m venv $VenvDir
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $ProjectDir "requirements.txt")

$TorchIndex = "https://download.pytorch.org/whl/cu128"
if ($Device -eq "cpu") {
    & $VenvPython -m pip install torch torchaudio
}
else {
    & $VenvPython -m pip install torch torchaudio --index-url $TorchIndex
}

& $VenvPython -m pip install -r (Join-Path $ProjectDir "requirements-ai.txt")
if ($Device -ne "cpu") {
    & $VenvPython -m pip install --force-reinstall torch torchaudio --index-url $TorchIndex
}

& $VenvPython -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
Write-Host "Setup complete. Run .\start.ps1 to open the app." -ForegroundColor Green
