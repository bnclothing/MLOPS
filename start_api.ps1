param(
    [int]$Port = 8000,
    [string]$HostAddress = "127.0.0.1",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual environment not found. Expected: $Python"
}

$env:MODEL_DEVICE = $Device
Set-Location -LiteralPath $ProjectRoot
& $Python -m uvicorn app.main:app --host $HostAddress --port $Port
