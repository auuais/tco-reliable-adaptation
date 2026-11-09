param(
    [ValidateSet('Probe', 'Validate', 'Run')][string]$Action = 'Run',
    [string]$RunId = '',
    [int]$Size = 224,
    [int[]]$Frames = @(0, 200),
    [int]$Steps = 0,
    [string]$Scene = 'heads/seq-01',
    [string]$DataRoot = 'E:\ResearchData\TCO'
)
$ErrorActionPreference = 'Stop'
$researchRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$resolvedDataRoot = (Resolve-Path -LiteralPath $DataRoot).Path
$dockerArguments = @('run', '--rm', '--gpus', 'all', '--shm-size', '8g',
    '--mount', "type=bind,source=$researchRoot,target=/workspace",
    '--mount', "type=bind,source=$resolvedDataRoot,target=/data,readonly",
    '--mount', 'type=volume,source=tco-torch-extensions,target=/root/.cache/torch_extensions',
    'tco-research:20261108', 'python', '-u')
if ($Action -eq 'Probe') {
    $dockerArguments += 'scripts/probe_renderer.py'
    $logFile = Join-Path $researchRoot 'environment/renderer-probe.log'
} elseif ($Action -eq 'Validate') {
    $dockerArguments += 'scripts/validate_data.py'
    $logFile = Join-Path $researchRoot 'environment/data-validation.log'
} else {
    if ($RunId -notmatch '^[A-Za-z0-9_-]+$') { throw 'Supply a unique alphanumeric RunId.' }
    $dockerArguments += @('scripts/run_baseline.py', '--run-id', $RunId,
        '--size', "$Size", '--steps', "$Steps", '--scene', $Scene, '--frames')
    $dockerArguments += @($Frames | ForEach-Object { "$_" })
    $logRoot = Join-Path $researchRoot 'runs/logs'
    New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
    $logFile = Join-Path $logRoot "$RunId.log"
    if (Test-Path -LiteralPath $logFile) { throw 'Run log already exists; select a new RunId.' }
}
& docker @dockerArguments 2>&1 | Tee-Object -FilePath $logFile
$dockerExit = $LASTEXITCODE
exit $dockerExit
