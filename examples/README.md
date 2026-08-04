# Practical examples

These programs are complete workflows rather than isolated API calls. Each one
reads or owns real input data, performs the calculation through cnumpy, writes a
machine-readable result, reports a deterministic summary, and verifies that no
native allocation remains live.

Run commands from the repository root. The included DLL is qualified for
Windows x64, so use a 64-bit AutoHotkey v2 executable and an x64 C toolchain.

## Run the complete application

Double-click the root [main.ahk](../main.ahk) for an interactive sales report.
Success and failure are displayed in a Windows dialog, so the program does not
depend on stdout or stderr handles supplied by a parent terminal.

Automated callers must opt into the deterministic headless contract:

```powershell
$Ahk = 'C:\Program Files\AutoHotkey\v2\AutoHotkey64.exe'
& $Ahk /ErrorStdOut=UTF-8 .\main.ahk --headless
& $Ahk /ErrorStdOut=UTF-8 .\main.ahk --headless `
  'D:\data\orders.csv' 'D:\reports\order-profit.csv'
```

The default data is [sales.csv](data/sales.csv), and the default result is
`build/examples/forum_sales_report.csv`. A failure returns exit code 1, exposes
the original cnumpy exception, and does not create the requested output file.

## Run the AutoHotkey suite

The verification script starts every AHK program as a separate process, checks
its numerical summary and generated CSV, then confirms that a missing input file
returns a nonzero exit with the original native error visible:

```powershell
$Ahk = 'C:\Program Files\AutoHotkey\v2\AutoHotkey64.exe'
.\examples\verify_ahk.ps1 -AhkPath $Ahk
```

Replace the executable path with your AutoHotkey v2 x64 installation. A
successful run ends with:

```text
expected_root_missing_input_exit=1
ERROR: Numpy.Loadtxt (cnp_loadtxt) failed with status -11: Cannot open file: ...
expected_missing_input_exit=1
ERROR: Numpy.Loadtxt (cnp_loadtxt) failed with status -11: Cannot open file: ...
ahk_example_contracts=passed
```

Generated files go to `build/examples/`, which is intentionally ignored by Git.

## Sales CSV report

[sales_report.ahk](ahk/sales_report.ahk) turns order rows into a compact financial
report. Its input is numeric CSV without a header:

| Column | Meaning |
|---:|---|
| 1 | Order ID |
| 2 | Quantity |
| 3 | Unit price |
| 4 | Unit cost |

Run it with the bundled [sales.csv](data/sales.csv):

```powershell
& $Ahk /ErrorStdOut .\examples\ahk\sales_report.ahk
```

Expected summary:

```text
orders=6
total_revenue=1695.00
total_profit=607.00
profit_margin=35.81%
best_order_id=105
output=...\build\examples\sales_report.csv
retained_bytes=0
```

The output columns are order ID, revenue, and profit. To process another file,
pass input and output paths explicitly:

```powershell
& $Ahk /ErrorStdOut .\examples\ahk\sales_report.ahk `
  'D:\data\orders.csv' 'D:\reports\order-profit.csv'
```

The calculation is vectorized: `quantity * unit_price` produces revenue, while
`quantity * (unit_price - unit_cost)` produces profit. Malformed shapes and
unreadable files fail with their real exception and process exit code 1.

## Ordinary least-squares regression

[linear_regression.ahk](ahk/linear_regression.ahk) fits
`observed = intercept + slope * x` using `Numpy.Lstsq`, evaluates predictions
with `Numpy.Matmul`, and calculates R-squared from array operations.

The [regression fixture](data/regression_points.csv) has two columns: `x` and the
observed value. Run:

```powershell
& $Ahk /ErrorStdOut .\examples\ahk\linear_regression.ahk
```

Expected summary:

```text
observations=6
intercept=2.557143
slope=1.737143
rank=2
residual_sum=0.030857
r_squared=0.999416
output=...\build\examples\linear_regression_predictions.csv
retained_bytes=0
```

The generated CSV columns are `x`, observed value, and predicted value. Custom
paths use the same two positional arguments as the sales example:

```powershell
& $Ahk /ErrorStdOut .\examples\ahk\linear_regression.ahk `
  'D:\data\calibration.csv' 'D:\reports\calibration-fit.csv'
```

The example retains all four `lstsq` outputs long enough to demonstrate their
ownership: coefficients, residual sums, rank, and singular values.

## Moving average and spike localization

[signal_smoothing.ahk](ahk/signal_smoothing.ahk) reads one numeric sensor sample
per row, applies a three-sample moving average with `Numpy.Convolve(...,
"valid")`, aligns the original signal to the valid window, and locates the
largest absolute residual.

Run it with [sensor_readings.csv](data/sensor_readings.csv):

```powershell
& $Ahk /ErrorStdOut .\examples\ahk\signal_smoothing.ahk
```

Expected summary:

```text
samples=9
smoothed_samples=7
spike_index=4
spike_observed=45.00
spike_moving_average=22.67
spike_residual=22.33
output=...\build\examples\signal_smoothing.csv
retained_bytes=0
```

The zero-based `spike_index` refers to the original input. Output columns are
sample index, observed value, moving average, and absolute residual. Supply a
different input and output in the same way:

```powershell
& $Ahk /ErrorStdOut .\examples\ahk\signal_smoothing.ahk `
  'D:\data\temperature.csv' 'D:\reports\temperature-smoothed.csv'
```

## Preallocated C pipeline

[preallocated_pipeline.c](c/preallocated_pipeline.c) shows how to keep owned
arrays outside a high-frequency loop. It allocates two inputs and three output
buffers once, then runs these public ABI calls 10,000 times:

```text
samples + calibration_offsets -> calibrated_power
sqrt(calibrated_power)         -> amplitudes
cumsum(amplitudes)             -> cumulative
sum(amplitudes)                -> C double
```

From a standard PowerShell window, compile and verify it through the Visual
Studio developer environment:

```powershell
.\examples\verify_c.ps1 `
  -VsDevCmdPath 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat'
```

Expected result:

```text
iterations=10000
amplitudes=[2, 3, 4, 5]
cumulative=[2, 5, 9, 14]
amplitude_sum=14
hot_loop_retained_bytes=0
retained_bytes=0
c_example_contracts=passed
```

If you are already in an x64 Developer PowerShell for Visual Studio 2022, the
equivalent direct compile command is:

```powershell
New-Item -ItemType Directory -Force build\examples | Out-Null
cl.exe /nologo /TC /W4 /Iinclude `
  examples\c\preallocated_pipeline.c `
  /Fo:build\examples\preallocated_pipeline.obj `
  /Fe:build\examples\preallocated_pipeline.exe `
  /link /LIBPATH:build\x64\Release cnumpy_ahk.lib
```

Add `build/x64/Release` to `PATH` before running the executable if it is not next
to `cnumpy_ahk.dll`. All status failures print the stored `CnpErrorState`, the
operation name, and the loop iteration; no replacement result is generated.

## Ownership pattern

The AHK examples deliberately initialize every `NdArray` variable to `0` and
drop references in reverse dependency order inside `finally`. They call
`Numpy.AllocatedMemory()` before `Numpy.Cleanup()` and reject any retained byte.
The C example follows the same rule with `cnp_array_free`.

This explicit pattern matters in long-running automation: cleanup shuts down
library-global state, while dropping an array releases that array's owned native
storage. Cleanup is not a substitute for releasing live owners.

For smaller API introductions, see [quickstart.ahk](ahk/quickstart.ahk). For
bulk callback integration, see [callbacks.ahk](ahk/callbacks.ahk) and the
[bulk callback guide](../docs/guides/bulk-callbacks.md).
