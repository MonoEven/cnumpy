param(
    [Parameter(Mandatory = $true)]
    [string]$AhkPath
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$outputRoot = Join-Path $repoRoot 'build\examples'

function Invoke-CheckedExample {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Script,
        [Parameter(Mandatory = $true)]
        [string[]]$ExpectedLines
    )

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $captured = & $AhkPath /ErrorStdOut $Script 2>&1 | Out-String
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    if ($exitCode -ne 0) {
        throw "Example failed with exit code $exitCode`n$captured"
    }
    foreach ($line in $ExpectedLines) {
        if (-not $captured.Contains($line)) {
            throw "Expected '$line' in output from $Script`n$captured"
        }
    }
    Write-Output $captured.TrimEnd()
}

Invoke-CheckedExample `
    -Script (Join-Path $PSScriptRoot 'ahk\sales_report.ahk') `
    -ExpectedLines @(
        'orders=6',
        'total_revenue=1695.00',
        'total_profit=607.00',
        'profit_margin=35.81%',
        'best_order_id=105',
        'retained_bytes=0'
    )

Invoke-CheckedExample `
    -Script (Join-Path $PSScriptRoot 'ahk\linear_regression.ahk') `
    -ExpectedLines @(
        'observations=6',
        'intercept=2.557143',
        'slope=1.737143',
        'rank=2',
        'r_squared=0.999416',
        'retained_bytes=0'
    )

Invoke-CheckedExample `
    -Script (Join-Path $PSScriptRoot 'ahk\signal_smoothing.ahk') `
    -ExpectedLines @(
        'samples=9',
        'smoothed_samples=7',
        'spike_index=4',
        'spike_observed=45.00',
        'spike_moving_average=22.67',
        'spike_residual=22.33',
        'retained_bytes=0'
    )

$missingInput = Join-Path $outputRoot 'intentionally-missing.csv'
if (Test-Path -LiteralPath $missingInput) {
    throw "Failure-test precondition violated: $missingInput exists"
}
$failureOutput = Join-Path $outputRoot 'should-not-be-created.csv'
$previousPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$failure = & $AhkPath /ErrorStdOut `
    (Join-Path $PSScriptRoot 'ahk\sales_report.ahk') `
    $missingInput $failureOutput 2>&1 | Out-String
$failureExitCode = $LASTEXITCODE
$ErrorActionPreference = $previousPreference
if ($failureExitCode -eq 0) {
    throw "Missing input unexpectedly succeeded`n$failure"
}
if (-not $failure.Contains('Numpy.Loadtxt')) {
    throw "Missing-input failure did not expose Numpy.Loadtxt`n$failure"
}
Write-Output "expected_missing_input_exit=$failureExitCode"
Write-Output $failure.TrimEnd()

$expectedFiles = @(
    @{ Path = Join-Path $outputRoot 'sales_report.csv'; First = '101.00,240.00,90.00' },
    @{ Path = Join-Path $outputRoot 'linear_regression_predictions.csv'; First = '0.000000,2.600000,2.557143' },
    @{ Path = Join-Path $outputRoot 'signal_smoothing.csv'; First = '1.000000,11.000000,10.333333,0.666667' }
)
foreach ($expected in $expectedFiles) {
    if (-not (Test-Path -LiteralPath $expected.Path)) {
        throw "Expected output file was not created: $($expected.Path)"
    }
    $firstLine = Get-Content -LiteralPath $expected.Path -TotalCount 1
    if ($firstLine -ne $expected.First) {
        throw "Unexpected first row in $($expected.Path): $firstLine"
    }
}

Write-Output 'ahk_example_contracts=passed'
