# Create .venv and install the project (Windows):  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
# Then: .venv\Scripts\Activate.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$py = $null
foreach ($v in "3.11", "3.12", "3.10") {
    & py "-$v" -c "import sys" 2>$null
    if ($LASTEXITCODE -eq 0) { $py = @("py", "-$v"); break }
}
if ($null -eq $py) { Write-Error "Need Python >= 3.10 (py launcher): https://www.python.org/downloads/" }

if (-not (Test-Path .venv)) {
    Write-Host "Creating .venv with Python $($py[1])"
    & $py[0] $py[1] -m venv .venv
}
$venvPy = ".venv\Scripts\python.exe"
& $venvPy -m pip install -q --upgrade pip
& $venvPy -m pip install -q -r requirements-dev.txt -e .
Write-Host "Installed into .venv ($(& $venvPy --version))"

& $venvPy -c "import sys; from lifemate.fetch_data import has_data; from lifemate.data import default_data_root; sys.exit(not has_data(default_data_root()))"
if ($LASTEXITCODE -eq 0) { Write-Host "Data found in data\. Check: pytest" }
else { Write-Host "No data yet. Run: python -m lifemate.fetch_data --url `"<zip link>`"  (or --zip <downloaded zip>)" }
Write-Host "Activate with: .venv\Scripts\Activate.ps1"
