# Getting started with cnumpy in AutoHotkey v2

This guide takes an AutoHotkey v2 script from loading the qualified x64 DLL to
creating arrays, performing NumPy-style operations, handling native errors,
and proving that native ownership returns to its starting point.

## Prerequisites

Use all components at the same architecture:

- Windows x64
- 64-bit AutoHotkey v2
- `ahk/numpy.ahk`
- `build/x64/Release/cnumpy_ahk.dll`

The release was qualified with AutoHotkey 2.1-alpha.30 x64. Other v2 releases
are not part of the frozen qualification record. If you rebuild the DLL, keep
the wrapper and DLL from the same source tree; the wrapper deliberately raises
an error when a required export is missing.

## Load the wrapper and DLL

The example directory is two levels below the repository root, so its explicit
path setup is:

```ahk
#Requires AutoHotkey v2.0
#Include ..\..\ahk\numpy.ahk

Numpy.DllPath := A_ScriptDir "\..\..\build\x64\Release\cnumpy_ahk.dll"
Numpy.Init()
```

Set `Numpy.DllPath` before any array factory or operation. Those methods call
`Numpy.Init()` on demand, and changing the path after the DLL is loaded does
not replace the loaded module.

For a script in the repository root, use:

```ahk
#Include ahk\numpy.ahk
Numpy.DllPath := A_ScriptDir "\build\x64\Release\cnumpy_ahk.dll"
```

## Create arrays

`Numpy.Array` accepts flat AHK numeric data and a row-major shape. Supply
exactly the product of the shape dimensions:

```ahk
matrix := Numpy.Array([1, 2, 3, 4, 5, 6], [2, 3])
vector := Numpy.Array([10, 20, 30])
integers := Numpy.IntArray([1, 2, 3], [3])
zeros := Numpy.Zeros([2, 3], Numpy.DT_FLOAT64)
```

The common dtype constants are:

| AHK constant | Native dtype |
|---|---|
| `Numpy.DT_BOOL` | bool |
| `Numpy.DT_INT16` | signed 16-bit integer |
| `Numpy.DT_INT32` | signed 32-bit integer |
| `Numpy.DT_LONGLONG` | signed 64-bit integer |
| `Numpy.DT_FLOAT32` | 32-bit float |
| `Numpy.DT_FLOAT64` | 64-bit float; the default |
| `Numpy.DT_COMPLEX128` | two 64-bit floating components |

AHK scalar conversion and `NdArray.ToArray()` expose real values as AHK
numbers. APIs that project a complex value to a single `double` reject the
complex dtype; use array-preserving APIs for complex results.

## Broadcast and reduce

Array operations follow the qualified NumPy 1.25 broadcasting and axis rules:

```ahk
source := Numpy.Array([1, 2, 3, 4, 5, 6], [2, 3])
offsets := Numpy.Array([10, 20, 30], [1, 3])
shifted := Numpy.Add(source, offsets)
rowSums := Numpy.Sum(shifted, 1)

; shifted: [11, 22, 33, 14, 25, 36], shape [2, 3]
; rowSums: [66, 75], shape [2]
```

Omitting `axis` means `axis=None`; passing `-1` means the last axis through the
facade's v2 reduction route:

```ahk
allElements := Numpy.Sum(source)
lastAxis := Numpy.Sum(source, -1)
```

Most operations allocate a new owner. APIs with an explicit destination avoid
that allocation and return the destination:

```ahk
output := Numpy.Empty([2, 3])
Numpy.Add(source, offsets, output)
```

Destination shape, dtype, and layout are validated. A mismatch is a native
error; the facade does not allocate a replacement destination.

## Inspect an array

`NdArray` metadata is cached after the first metadata call and returned in AHK
types:

```ahk
shape := shifted.Shape       ; cloned AHK Array: [2, 3]
dtype := shifted.Dtype       ; Numpy.DT_FLOAT64 (13)
size := shifted.Size         ; 6
bytes := shifted.Nbytes      ; 48
isC := shifted.CContiguous   ; true
values := shifted.ToArray()  ; flat AHK Array of doubles
text := shifted.ToString()
```

Relevant properties are `Ndim`, `Shape`, `Size`, `Dtype`, `ItemSize`, `Nbytes`,
`Strides`, `CContiguous`, and `FContiguous`. `Shape` returns a clone, so changing
that AHK array does not reshape the native owner.

## Handle errors without losing their cause

The facade clears the native error state before operations that need it and
raises an AHK exception containing the native status and message. Catch only
errors you can handle:

```ahk
try invalid := source.Reshape([4, 2])
catch Error as err {
    ; Real result:
    ; NdArray.Reshape failed with status -4:
    ; Cannot reshape array of size 6 into shape (8 elements)
    FileAppend(err.Message "`n", "**")
}
```

A missing DLL, wrong architecture, missing export, invalid shape, invalid axis,
and callback exception remain distinct failures. There is no empty-array or
scalar fallback.

## Own and release resources

An `NdArray` owns its opaque native handle. Its destructor decrements the
native reference when the last AHK reference disappears. Make the release
order explicit around long-running scripts and tests:

```ahk
Numpy.Init()
baseline := Numpy.AllocatedMemory()

source := 0
result := 0
try {
    source := Numpy.Arange(0, 1024)
    result := Numpy.Sqrt(source)
} finally {
    result := 0
    source := 0
    retained := Numpy.AllocatedMemory()
    Numpy.Cleanup()
}

if retained != baseline
    throw Error("retained native bytes: " (retained - baseline))
```

Release derived arrays before their sources. Views retain their owners
internally, so source-first release is safe when required, but derived-first
release makes the intended lifecycle clear. Never call `Numpy.Cleanup()` while
an `NdArray` or native callback result remains live.

`AllocatedMemory()` is the library's tracked native allocation total. Compare
it with a baseline taken after initialization; it is a lifecycle assertion,
not the Windows process working set.

## Configure native threads

The setting controls cnumpy's initialized GEMM thread pool:

```ahk
Numpy.SetNumThreads(4)
configured := Numpy.GetNumThreads() ; 4
Numpy.SetNumThreads(0)              ; restore automatic count
```

A negative value is rejected. The setting does not turn scalar callbacks into
parallel or vectorized NumPy callables, and it should not be changed during a
timed comparison unless both configuration and timing artifacts record it.

## Run the complete example

From the repository root:

```powershell
$Ahk = 'C:\Program Files\AutoHotkey\v2\AutoHotkey64.exe'
& $Ahk /ErrorStdOut examples\ahk\quickstart.ahk
```

The validated source is [examples/ahk/quickstart.ahk](../examples/ahk/quickstart.ahk).
Its final line must be:

```text
retained_bytes=0
```

## Troubleshooting

| Symptom | Meaning and action |
|---|---|
| `failed to load DLL ... (GetLastError 193)` | The process and DLL architectures do not match. Run 64-bit AutoHotkey with the x64 DLL. |
| `failed to load DLL ... (GetLastError 126)` | The path or a native dependency is missing. Verify the absolute `Numpy.DllPath` and Release output. |
| `missing native export ...` | `ahk/numpy.ahk` and the DLL are from different builds. Rebuild or deploy them together. |
| `failed with status -4` | A shape contract failed. Read the complete native message; do not discard it and retry with another shape automatically. |
| `failed with status -6` | The axis is invalid for the array rank or projected API. Check whether the API is legacy or v2. |
| Callback exception text is rethrown | This is intentional. Fix the callback; the facade removes its registered context before rethrowing. |
| Retained bytes are nonzero | At least one owner, view, result, string, mapping, or callback context remains live. Release the concrete owner and rerun the assertion. |

For callback-specific contracts, continue with
[Bulk callbacks from AutoHotkey and C](guides/bulk-callbacks.md). For semantic
boundaries and legacy migration, see the
[NumPy 1.25 compatibility statement](compatibility/numpy-1.25.md).
