# 一键打包（需先安装 requirements-dev.txt）
param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path "models\best.pt")) {
    throw "缺少 models\best.pt，请先把模型权重放入 models 目录"
}

& $Python -m PyInstaller --noconfirm --clean build.spec

$version = (Get-Content "pyproject.toml" | Select-String -Pattern '^version = "(.+)"').Matches.Groups[1].Value
$output = "dist\AgriPestDetection"
if (-not (Test-Path $output)) {
    throw "打包失败：未找到 $output"
}

$zip = "dist\AgriPestDetection-$version-win64.zip"
Compress-Archive -Path "$output\*" -DestinationPath $zip -Force
Write-Host "打包完成: $zip"
