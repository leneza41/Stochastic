<#
.SYNOPSIS
    Script maestro para Windows 11 que ejecuta el banco de pruebas experimental (Pasos 3 y 4)
    y genera la carpeta completa de resultados de investigación en la ruta destino indicada.

.PARAMETER OutputDir
    Ruta de la carpeta destino donde se guardarán los datos crudos, tablas estadísticas,
    figuras vectoriales (.svg) y reportes académicos (.md y .html).

.PARAMETER RunsPerMap
    Número de corridas experimentales independientes por mapa (por defecto: 8; usar 30 para muestra piloto completa).

.PARAMETER FailuresPerRun
    Número de fallos estocásticos secuenciales por corrida (por defecto: 8).

.EXAMPLE
    .\run_experiments.ps1 "C:\Resultados_Investigacion"
    .\run_experiments.ps1 -OutputDir "C:\Resultados_Investigacion" -RunsPerMap 30 -FailuresPerRun 10
#>
param(
    [Parameter(Position = 0, Mandatory = $false)]
    [string]$OutputDir = "results\research_output",

    [Parameter(Mandatory = $false)]
    [int]$RunsPerMap = 8,

    [Parameter(Mandatory = $false)]
    [int]$FailuresPerRun = 8
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $RepoRoot

try {
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "  EJECUTANDO PIPELINE EXPERIMENTAL DE BÚSQUEDA INCREMENTAL (WINDOWS 11)" -ForegroundColor Cyan
    Write-Host "================================================================================" -ForegroundColor Cyan

    python "python\run_research_pipeline.py" "$OutputDir" --runs-per-map $RunsPerMap --failures-per-run $FailuresPerRun
    if ($LASTEXITCODE -ne 0) {
        throw "El pipeline de investigación terminó con código de error $LASTEXITCODE"
    }

    $DashboardPath = Join-Path (Resolve-Path $OutputDir).Path "04_research_report\RESEARCH_DASHBOARD.html"
    Write-Host ""
    Write-Host "Para visualizar el tablero interactivo en tu navegador de Windows 11, ejecuta:" -ForegroundColor Green
    Write-Host "  Start-Process `"$DashboardPath`"" -ForegroundColor Yellow
}
finally {
    Pop-Location
}
