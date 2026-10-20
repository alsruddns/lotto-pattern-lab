param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,
    [int]$Port = 8501
)

$root = [System.IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\')
try {
    $listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
        Select-Object -First 1
} catch {
    Write-Output "Could not inspect TCP port ${Port}: $($_.Exception.Message)"
    exit 1
}

if (-not $listener) {
    Write-Output "Port $Port is available."
    exit 2
}

$process = Get-CimInstance Win32_Process -Filter "ProcessId = $($listener.OwningProcess)" -ErrorAction SilentlyContinue
if (-not $process) {
    Write-Output "Port $Port is occupied by PID $($listener.OwningProcess); its command line could not be read."
    exit 1
}

$rootPrefix = $root + [System.IO.Path]::DirectorySeparatorChar
$sameProject = ($process.ExecutablePath -and $process.ExecutablePath.StartsWith(
    $rootPrefix,
    [System.StringComparison]::OrdinalIgnoreCase
)) -or ($process.CommandLine -and $process.CommandLine.IndexOf(
    $root,
    [System.StringComparison]::OrdinalIgnoreCase
) -ge 0)

if ($sameProject) {
    Write-Output "Port $Port is already serving this project (PID $($process.ProcessId))."
    exit 0
}

Write-Output "Port $Port is occupied by a different process (PID $($process.ProcessId))."
Write-Output "Executable: $($process.ExecutablePath)"
Write-Output "Command: $($process.CommandLine)"
exit 1
