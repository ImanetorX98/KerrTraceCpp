# Download LibTorch CPU (Windows x64) and extract to .\libtorch\
# Run once before cmake.
#
# Usage:
#   .\scripts\download_libtorch.ps1
#   .\scripts\download_libtorch.ps1 -TorchVersion 2.3.1

param(
    [string]$TorchVersion = "2.3.1"
)

$Root   = Split-Path $PSScriptRoot -Parent
$Target = Join-Path $Root "libtorch"

if (Test-Path $Target) {
    Write-Host "LibTorch already present at $Target"
    exit 0
}

$Url = "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-$TorchVersion%2Bcpu.zip"
$Zip = Join-Path $Root "libtorch.zip"

Write-Host "Downloading LibTorch $TorchVersion from:"
Write-Host "  $Url"

curl.exe -fL -o $Zip $Url
Expand-Archive -Path $Zip -DestinationPath $Root -Force
Remove-Item $Zip

Write-Host "Done. LibTorch installed at: $Target"
