# Image-3D Windows PowerShell startup script
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if (-not $env:VENV_DIR) {
    $env:VENV_DIR = ".venv"
}

$VenvDir = $env:VENV_DIR
$Uvicorn = Join-Path $VenvDir "Scripts\uvicorn.exe"
$Pip = Join-Path $VenvDir "Scripts\pip.exe"

if (-not (Test-Path $VenvDir)) {
    Write-Host "venvが見つかりません。先に以下を実行してください:"
    Write-Host "  py -3.12 -m venv $VenvDir"
    Write-Host "  $Pip install -r requirements.txt"
    exit 1
}

if (-not (Test-Path $Uvicorn)) {
    Write-Host "uvicornが見つかりません。依存をインストールしてください:"
    Write-Host "  $Pip install -r requirements.txt"
    exit 1
}

# auto: GPU + hy3dgen があれば hunyuan3d、なければ mock(テスト用形状)に自動解決
if (-not $env:IMAGE3D_GENERATOR) {
    $env:IMAGE3D_GENERATOR = "auto"
}
if (-not $env:IMAGE3D_HOST) {
    $env:IMAGE3D_HOST = "127.0.0.1"
}
if (-not $env:IMAGE3D_PORT) {
    $env:IMAGE3D_PORT = "8100"
}

& $Uvicorn server.main:app --host $env:IMAGE3D_HOST --port $env:IMAGE3D_PORT @args
