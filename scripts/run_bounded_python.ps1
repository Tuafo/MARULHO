param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 86400)]
    [int]$TimeoutSeconds,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PythonArguments
)

$ErrorActionPreference = "Stop"
$pythonExecutable = (& python -c "import sys; print(sys.executable)").Trim()
if (-not $pythonExecutable) {
    throw "Could not resolve the active Python executable."
}

$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = $pythonExecutable
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
foreach ($argument in $PythonArguments) {
    [void]$startInfo.ArgumentList.Add($argument)
}

$process = [System.Diagnostics.Process]::new()
$process.StartInfo = $startInfo
if (-not $process.Start()) {
    throw "Failed to start bounded Python process."
}

$deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
while (-not $process.HasExited -and [DateTime]::UtcNow -lt $deadline) {
    Start-Sleep -Milliseconds 200
    $process.Refresh()
}

if (-not $process.HasExited) {
    try {
        $process.Kill($true)
    }
    catch {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    }
    $process.WaitForExit()
    [Console]::Error.WriteLine(
        "MARULHO bounded process tree exceeded ${TimeoutSeconds}s and was terminated."
    )
    exit 124
}

exit $process.ExitCode
