# 启动 HTTP 服务
param(
    [string]$HostAddr = "127.0.0.1",
    [int]$Port = 8000,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

& $Python -m uvicorn app.api.server:app --host $HostAddr --port $Port
