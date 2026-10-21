$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectDir

$Python = Join-Path (Join-Path $ProjectDir ".venv") "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "Not installed yet. Run .\setup.ps1 first." -ForegroundColor Yellow
    exit 1
}

$App = Join-Path $ProjectDir "streamlit_app.py"
& $Python -m streamlit run $App --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
