# Start Kotaemon with Docker Compose (persistent ./ktem_app_data)
# Usage:
#   .\scripts\docker-up.ps1
#   .\scripts\docker-up.ps1 -Ollama
#   .\scripts\docker-up.ps1 -Build

param(
    [switch]$Ollama,
    [switch]$Reranker,
    [switch]$Build
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path ".env")) {
    Write-Host "Creating .env from .env.example ..."
    Copy-Item ".env.example" ".env"
    Write-Host "Edit .env (API keys, LOCAL_MODEL) then re-run this script."
}

$args = @("compose")
if ($Ollama) { $args += "--profile", "ollama" }
if ($Reranker) { $args += "--profile", "reranker" }
$args += "up", "-d"
if ($Build) { $args += "--build" }

Write-Host "docker $($args -join ' ')"
& docker @args

Write-Host ""
Write-Host "App:     http://localhost:7860"
Write-Host "Data:    $(Resolve-Path -ErrorAction SilentlyContinue ./ktem_app_data) (bind mount)"
Write-Host "Logs:    docker compose logs -f kotaemon"
Write-Host "Stop:    docker compose down   (keeps data; use -v only to delete volumes)"
