# Bulk callbacks from AutoHotkey and C

cnumpy's v2 callback exports group logical values into batches of at most 256.
This reduces native-to-AutoHotkey crossings while preserving explicit status,
production count, output shape, and ownership contracts. A nonzero callback
status aborts the operation; no adapter retries through the scalar legacy ABI.

## Choose the interface

Use the AutoHotkey facade for ordinary scalar numeric callbacks. Use the public
C v2 ABI when you need an array-valued `apply_along_axis` result, unknown-length
iteration, explicit `userdata`, or direct control of `produced_count`.

| Requirement | AHK facade | C v2 ABI |
|---|---:|---:|
| Real `double` callback values | Yes | Yes |
| Scalar `ApplyAlongAxis` result | Yes | Yes |
| Explicit array-valued result shape | No | Yes |
| Fixed-length `FromIter` | Yes | Yes |
| Unknown length with `count == -1` | No | Yes |
| Explicit callback status and `userdata` | Managed by facade | Yes |
| Object dtype or Python object result | No | No |

## AutoHotkey facade

The high-level signatures are:

```ahk
Numpy.ApplyAlongAxis(callback, axis, source)
Numpy.ApplyOverAxes(callback, source, axes)
Numpy.FromFunction(callback, shape)
Numpy.FromIter(callback, count, dtype := Numpy.DT_FLOAT64)
Numpy.FromPyFunc(callback, source)
Numpy.Vectorize(callback, source)
Numpy.Piecewise(source, conditions, callback)
Numpy.Select(conditions, choices, defaultValue := 0.0)
Numpy.PutAlongAxis(destination, indices, values, axis)
```

### Callback forms

```ahk
SumLine(values) {
    total := 0.0
    for value in values
        total += value
    return total
}

source := Numpy.Array([1, 2, 3, 4, 5, 6], [2, 3])
applied := Numpy.ApplyAlongAxis(SumLine, 0, source)

generated := Numpy.FromFunction(
    (row, column) => row * 10 + column + 0.5, [2, 3])

iterator := SequenceIterator([1.9, -2.2, 7.0, 8.0])
iterated := Numpy.FromIter(iterator, 4, Numpy.DT_INT16)

vectorized := Numpy.Vectorize(value => value * 2 + 1, source)
```

- `ApplyAlongAxis` receives one AHK array per logical line and requires one
  numeric result. The facade therefore produces a scalar callback result per
  line even though the C v2 export can insert an explicit result shape.
- `ApplyOverAxes` receives real-valued lines and retains the reduced axes as
  length-one dimensions, matching the represented facade contract.
- `FromFunction` passes each coordinate as a separate callback argument.
- `FromIter` calls a zero-argument function exactly `count` times and converts
  its real results to `dtype`. AHK requires a non-negative count.
- `FromPyFunc` and `Vectorize` each call a unary real-number function and
  return a float64 array in the represented surface.
- `Piecewise` uses one shared callback for every matching condition, in
  condition order. It is not NumPy's general list of functions and constants.
- `Select` is not a callback API. It broadcasts conditions and choices and
  selects the first true condition, then the default.
- `PutAlongAxis` mutates its destination only after validating all indices; an
  invalid index leaves the destination unchanged.

### Exception propagation

The facade records the first AHK callback exception, returns a nonzero status
to native code, unregisters the callback context, and rethrows the original
exception object. It does not make a scalar retry:

```ahk
class FailingCallback {
    Call(arguments*) {
        throw Error("callback example failure")
    }
}

try temporary := Numpy.Vectorize(FailingCallback(), source)
catch Error as err {
    if err.Message != "callback example failure"
        throw
}
```

The complete, validated program is
[examples/ahk/callbacks.ahk](../../examples/ahk/callbacks.ahk). It releases all
callback results, requires `Numpy.CallbackContexts.Count == 0` after failure,
and finishes with `retained_bytes=0`.

## C v2 callback ABI

Include [cnumpy_ahk.h](../../include/cnumpy/cnumpy_ahk.h):

```c
#include "cnumpy/cnumpy_ahk.h"
```

The public callback types are:

```c
typedef CNP_STATUS (CNP_CALL *CnpAhkLineBatchCallback)(
    const double *lines, int64_t line_count, int64_t line_length,
    double *results, int64_t result_capacity,
    int64_t *produced_count, void *userdata);

typedef CNP_STATUS (CNP_CALL *CnpAhkCoordinateBatchCallback)(
    const int64_t *coordinates, int64_t point_count, int ndim,
    double *results, int64_t result_capacity,
    int64_t *produced_count, void *userdata);

typedef CNP_STATUS (CNP_CALL *CnpAhkIteratorBatchCallback)(
    double *results, int64_t result_capacity,
    int64_t *produced_count, void *userdata);

typedef CNP_STATUS (CNP_CALL *CnpAhkUnaryBatchCallback)(
    const double *values, int64_t value_count,
    double *results, int64_t result_capacity,
    int64_t *produced_count, void *userdata);
```

`CNP_AHK_CALLBACK_BATCH_SIZE` is 256 logical items. The actual count/capacity
can be smaller for a final or small batch.

### Production rules

| Export family | Success requirement |
|---|---|
| `apply_along_axis_v2` | Produce exactly `line_count * product(result_shape)` values. |
| `apply_over_axes_v2` | Produce exactly the scalar result capacity for each line batch. |
| `fromfunction_v2` | Produce exactly one value per coordinate (`point_count`). |
| `frompyfunc_v2`, `vectorize_v2`, `piecewise_v2` | Produce exactly one value per offered unary input. |
| `fromiter_v2`, fixed count | Produce exactly the offered capacity on every call. |
| `fromiter_v2`, `count == -1` | Produce zero through capacity; a short batch signals exhaustion. |

On success, every callback must set `*produced_count`. A value below zero or
above capacity is a `CNP_ERR_VALUE`. A short result is also an error for every
fixed-size contract. On callback failure, return a nonzero `CNP_STATUS`; the
export stops immediately, frees its partial result, and exposes that same
status in the native error state.

### Array-valued line result

This callback returns two values for each input line: its sum and the
difference between its first and last values.

```c
static CNP_STATUS CNP_CALL summarize_lines(
    const double *lines, int64_t line_count, int64_t line_length,
    double *results, int64_t result_capacity,
    int64_t *produced_count, void *userdata) {
    int64_t line_index;
    (void)userdata;

    if (result_capacity != line_count * 2) {
        *produced_count = 0;
        return CNP_ERR_VALUE;
    }
    for (line_index = 0; line_index < line_count; ++line_index) {
        const double *line = lines + line_index * line_length;
        double sum = 0.0;
        int64_t item;
        for (item = 0; item < line_length; ++item) {
            sum += line[item];
        }
        results[line_index * 2] = sum;
        results[line_index * 2 + 1] = line[0] - line[line_length - 1];
    }
    *produced_count = result_capacity;
    return CNP_OK;
}
```

Pass the result shape explicitly:

```c
const int64_t callback_shape[] = {2};
CnpArray *result = (CnpArray *)cnp_ahk_apply_along_axis_v2(
    summarize_lines, NULL, 1, source, 1, callback_shape);
```

For a source shape `[2, 3]`, this produces shape `[2, 2]`. The result-shape
dimensions replace the applied axis at that position, matching NumPy's output
layout for the represented real-valued callback.

### Unknown-length iterator

With `count == -1`, a partial batch terminates the iterator:

```c
static CNP_STATUS CNP_CALL next_values(
    double *results, int64_t result_capacity,
    int64_t *produced_count, void *userdata) {
    IteratorState *state = (IteratorState *)userdata;
    int64_t remaining = state->count - state->cursor;
    int64_t count = remaining < result_capacity ? remaining : result_capacity;
    int64_t index;

    for (index = 0; index < count; ++index) {
        results[index] = state->values[state->cursor + index];
    }
    state->cursor += count;
    *produced_count = count;
    return CNP_OK;
}

CnpArray *result = (CnpArray *)cnp_ahk_fromiter_v2(
    next_values, &iterator, -1, CNP_DOUBLE);
```

A fixed count does not treat a short batch as exhaustion; it is an explicit
error because the promised number of elements was not produced.

## Ownership and error state

Input arrays and `userdata` are borrowed for the duration of the synchronous
call. Every non-null array returned by a v2 export is a new owner. Release it
with `cnp_array_free` or the matching reference-count decrement.

```c
cnp_clear_error();
CnpArray *result = (CnpArray *)cnp_ahk_vectorize_v2(
    callback, userdata, source);
if (result == NULL) {
    CnpErrorState error = {0};
    CNP_STATUS status = cnp_get_error(&error);
    fprintf(stderr, "%d %s: %s\n",
        (int)status, error.func, error.message);
}
```

Do not inspect only the null pointer and discard the error state. Do not retry
the same logical operation through a legacy scalar export: that changes both
the callback count and its failure/atomicity contract.

## Build and run the C example

From an x64 Visual Studio developer PowerShell at the repository root:

```powershell
New-Item -ItemType Directory -Force build\examples | Out-Null
cl.exe /nologo /std:c11 /W4 /TC /Iinclude examples\c\bulk_callbacks.c `
    /Fobuild\examples\bulk_callbacks.obj `
    /Febuild\examples\bulk_callbacks.exe `
    /link build\x64\Release\cnumpy_ahk.lib
$env:Path = (Resolve-Path build\x64\Release).Path + ';' + $env:Path
& build\examples\bulk_callbacks.exe
```

The complete source is
[examples/c/bulk_callbacks.c](../../examples/c/bulk_callbacks.c). It exercises
array-valued line results, unknown-length iteration, a real nonzero callback
status, and retained-memory equality. Its successful stdout ends with
`retained_bytes=0`; the expected callback failure is printed to stderr with
status `-13` and function `cnp_ahk_vectorize_v2`.

## Performance expectations

Batching reduces crossings but does not turn scalar user code into a NumPy
vectorized kernel. In the final qualification, v2 reduced coordinate,
iterator, and unary calls from thousands to batches of 256 and improved the
direct AHK legacy-to-v2 timings. `fromfunction`, `piecewise`, and
`apply_over_axes` still remain much slower than vectorized NumPy because their
represented callback semantics execute scalar work inside each batch.

Use native array operations when the computation already exists in cnumpy.
Use callbacks when custom scalar behavior is required. A future array-valued
coordinate/mask or opaque-reduction-state ABI would be needed to close the
remaining semantic performance gap; changing the batch-size constant alone is
not equivalent.
