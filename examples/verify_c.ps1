param(
    [Parameter(Mandatory = $true)]
    [string]$VsDevCmdPath
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$source = Join-Path $PSScriptRoot 'c\preallocated_pipeline.c'
$outputDirectory = Join-Path $repoRoot 'build\examples'
$executable = Join-Path $outputDirectory 'preallocated_pipeline.exe'
$objectFile = Join-Path $outputDirectory 'preallocated_pipeline.obj'
$releaseDirectory = Join-Path $repoRoot 'build\x64\Release'
$includeDirectory = Join-Path $repoRoot 'include'
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null

$compileCommand = 'call "{0}" -no_logo -arch=x64 -host_arch=x64 && cl.exe /nologo /TC /W4 /I"{1}" "{2}" /Fo:"{3}" /Fe:"{4}" /link /LIBPATH:"{5}" cnumpy_ahk.lib' -f `
    $VsDevCmdPath, $includeDirectory, $source, $objectFile, $executable, `
    $releaseDirectory

$previousPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$compileOutput = & $env:ComSpec /d /s /c $compileCommand 2>&1 | Out-String
$compileExitCode = $LASTEXITCODE
$ErrorActionPreference = $previousPreference
Write-Output $compileOutput.TrimEnd()
if ($compileExitCode -ne 0) {
    throw "C example compilation failed with exit code $compileExitCode"
}

$previousPath = $env:PATH
$env:PATH = $releaseDirectory + [IO.Path]::PathSeparator + $env:PATH
try {
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $captured = & $executable 2>&1 | Out-String
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
} finally {
    $env:PATH = $previousPath
}

if ($exitCode -ne 0) {
    throw "C example failed with exit code $exitCode`n$captured"
}
$expectedLines = @(
    'iterations=10000',
    'amplitudes=[2, 3, 4, 5]',
    'cumulative=[2, 5, 9, 14]',
    'amplitude_sum=14',
    'hot_loop_retained_bytes=0',
    'retained_bytes=0'
)
foreach ($line in $expectedLines) {
    if (-not $captured.Contains($line)) {
        throw "Expected '$line' in C example output`n$captured"
    }
}

Write-Output $captured.TrimEnd()
Write-Output 'c_example_contracts=passed'
