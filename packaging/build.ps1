param([string]$Python = "python")
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    & $Python -m PyInstaller packaging/PoreSAM.spec --workpath build/cpu --noconfirm
    if ($LASTEXITCODE -ne 0) { throw "PoreSAM build failed." }
    Write-Output "Built: $projectRoot\dist\PoreSAM\PoreSAM.exe"
} finally { Pop-Location }
