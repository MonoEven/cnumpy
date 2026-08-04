# cnumpy for AutoHotkey v2

cnumpy is a native x64 numerical array library with a public C ABI and an
AutoHotkey v2 facade. Its represented API is qualified against NumPy 1.25.0,
including numerical results, error behavior, ownership, release ordering, and
public-boundary performance.

This is an API projection, not a Python runtime or a claim that every NumPy
argument and return category can cross the C/AutoHotkey boundary. The exact
scope and intentional differences are documented in the
[NumPy 1.25 compatibility statement](docs/compatibility/numpy-1.25.md).

## Qualified release

| Item | Value |
|---|---|
| Platform | Windows x64 |
| cnumpy version | `1.21.0-cnumpy` |
| NumPy oracle | `1.25.0` |
| AutoHotkey | v2, qualified with 2.1-alpha.30 x64 |
| Release DLL | `build/x64/Release/cnumpy_ahk.dll` |
| SHA-256 | `9b15dc7fc2a19fbb63ae919e6f3429cbd33443a04f588223c62773d6bd67e21b` |
| Manifest | 752/752 declarations owned; zero known gaps |
| Complete suites | Python 1,677; AutoHotkey 206; manifest/catalog 181 |

The hash identifies the final qualified binary. MSVC link-time code generation
is not byte-reproducible for this project, so a source-identical local rebuild
can have a different hash and must be requalified before being described as
the same artifact.

## Quick start

Set the DLL path before the first library call, release every `NdArray`, and
call `Numpy.Cleanup()` last:

```ahk
#Requires AutoHotkey v2.0
#Include ahk\numpy.ahk

Numpy.DllPath := A_ScriptDir "\build\x64\Release\cnumpy_ahk.dll"
Numpy.Init()
baseline := Numpy.AllocatedMemory()

source := 0
offsets := 0
shifted := 0
rowSums := 0
try {
    source := Numpy.Array([1, 2, 3, 4, 5, 6], [2, 3])
    offsets := Numpy.Array([10, 20, 30], [1, 3])
    shifted := Numpy.Add(source, offsets)
    rowSums := Numpy.Sum(shifted, 1)

    MsgBox shifted.ToString() "`nrow sums: " rowSums.ToString()
} finally {
    rowSums := 0
    shifted := 0
    offsets := 0
    source := 0
    retained := Numpy.AllocatedMemory()
    Numpy.Cleanup()
}

if retained != baseline
    throw Error("retained native memory: " (retained - baseline) " bytes")
```

The complete [quickstart example](examples/ahk/quickstart.ahk) also demonstrates
a native shape error and prints deterministic output. The
[callback example](examples/ahk/callbacks.ahk) covers the high-level callback
facade and exception propagation. For end-to-end applications, use the
[practical examples](examples/README.md): CSV sales analysis, ordinary
least-squares regression, signal smoothing and spike localization, and a
preallocated C pipeline.

## Build

The supported build in this repository is the MSVC v143 x64 Release project.
From PowerShell:

```powershell
$MSBuild = 'C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\amd64\MSBuild.exe'
& $MSBuild src\cnumpy_ahk.vcxproj /m /t:Rebuild '/p:Configuration=Release;Platform=x64' /nologo /v:normal
```

The build writes the DLL and import library to `build/x64/Release/`. Compiler
warnings remain visible and a nonzero build exit is a failure.

## Run the examples and tests

Set `$Ahk` to a 64-bit AutoHotkey v2 executable:

```powershell
$Ahk = 'C:\Program Files\AutoHotkey\v2\AutoHotkey64.exe'
& $Ahk /ErrorStdOut examples\ahk\quickstart.ahk
& $Ahk /ErrorStdOut examples\ahk\callbacks.ahk
.\examples\verify_ahk.ps1 -AhkPath $Ahk

python -B -W error::ResourceWarning -m unittest discover -s benchmark\tests -v
& $Ahk /ErrorStdOut=UTF-8 ahk\numpy.test.ahk
& $Ahk /ErrorStdOut=UTF-8 benchmark\benchmark_smoke.test.ahk
python benchmark\benchmark.py --profile focus --size-scale smoke --warmups 1 --samples 3 --target-sample-ms 1
```

Use the full benchmark protocol before making performance claims; a smoke run
checks the pipeline, not regression stability. See the
[benchmark guide](benchmark/README.md) for profiles, timing boundaries,
qualification metadata, and report interpretation.

## Documentation

- [Getting started with AutoHotkey v2](docs/getting-started.md)
- [Practical, complete examples](examples/README.md)
- [Bulk callbacks from AutoHotkey and C](docs/guides/bulk-callbacks.md)
- [NumPy 1.25 compatibility and migration](docs/compatibility/numpy-1.25.md)
- [Final performance and behavior evidence](docs/compatibility/2026-08-04-performance-behavior-convergence-baseline.md)
- [Public C API](include/cnumpy/cnumpy.h)
- [AutoHotkey/bulk callback ABI](include/cnumpy/cnumpy_ahk.h)

## Operating rules

- Treat returned arrays as owned unless an API explicitly documents a borrowed
  value. Release C owners with `cnp_array_free`/`cnp_array_decref`; release AHK
  owners by dropping every `NdArray` reference.
- Do not call `Numpy.Cleanup()` while arrays or callback results are live.
- Native failures are exceptions in the AHK facade and error states/statuses in
  C. They are not converted into empty arrays or substitute results.
- Prefer v2 exports where a legacy sentinel or scalar return loses NumPy
  meaning. The compatibility guide lists each important migration boundary.
