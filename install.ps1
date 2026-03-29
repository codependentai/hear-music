param(
    [switch]$Visualize,
    [switch]$Editable
)

$ErrorActionPreference = "Stop"

function Get-PythonCommand {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @("python")
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @("py", "-3")
    }
    throw "Python was not found on PATH. Install Python 3.11+ and try again."
}

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$PythonCommand,

        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    $pythonExe = $PythonCommand[0]
    $pythonArgs = @()
    if ($PythonCommand.Length -gt 1) {
        $pythonArgs = $PythonCommand[1..($PythonCommand.Length - 1)]
    }

    & $pythonExe @pythonArgs @Arguments
}

function Add-PathIfMissing {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TargetPath
    )

    $current = [Environment]::GetEnvironmentVariable("Path", "User")
    $parts = @()
    if ($current) {
        $parts = $current.Split(";", [System.StringSplitOptions]::RemoveEmptyEntries)
    }

    if ($parts -contains $TargetPath) {
        Write-Host "PATH already contains: $TargetPath"
        return
    }

    $newPath = (($parts + $TargetPath) -join ";")
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "Added to user PATH: $TargetPath"
}

function Get-ScriptsDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$PythonCommand,

        [Parameter(Mandatory = $true)]
        [bool]$UserInstall
    )

    if ($UserInstall) {
        return (Invoke-Python -PythonCommand $PythonCommand -c "import os, sysconfig; scheme = 'nt_user' if os.name == 'nt' else 'posix_user'; print(sysconfig.get_path('scripts', scheme=scheme))").Trim()
    }

    return (Invoke-Python -PythonCommand $PythonCommand -c "import sysconfig; print(sysconfig.get_path('scripts'))").Trim()
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Get-PythonCommand

$packageSpec = "."
if ($Visualize) {
    $packageSpec = ".[visualize]"
}

if ($Editable) {
    $pipArgs = @("-m", "pip", "install", "-e", $packageSpec)
} else {
    $pipArgs = @("-m", "pip", "install", "--user", $packageSpec)
}

Push-Location $repoRoot
try {
    Write-Host "Installing hear-music from $repoRoot"
    Invoke-Python -PythonCommand $python @pipArgs

    $scriptsDir = Get-ScriptsDirectory -PythonCommand $python -UserInstall (-not $Editable)
    if (-not $scriptsDir) {
        throw "Could not determine Python scripts directory."
    }

    Add-PathIfMissing -TargetPath $scriptsDir

    $cmdCandidates = @(
        (Join-Path $scriptsDir "hear-music.exe"),
        (Join-Path $scriptsDir "hear-music-script.py"),
        (Join-Path $scriptsDir "hear-music.cmd")
    )

    $installedCommand = $cmdCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($installedCommand) {
        Write-Host "Installed command: $installedCommand"
    } else {
        Write-Warning "Install finished, but the hear-music launcher was not found in $scriptsDir."
    }

    Write-Host ""
    Write-Host "Install complete."
    Write-Host "Open a new terminal, then run:"
    Write-Host "  hear-music --help"
} finally {
    Pop-Location
}
