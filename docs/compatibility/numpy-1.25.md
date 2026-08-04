# NumPy 1.25 compatibility

Qualification date: 2026-08-04 (Asia/Shanghai)

cnumpy's represented public C ABI and AutoHotkey v2 facade are qualified
against NumPy 1.25.0. All 752 declarations have an explicit test owner
and the manifest has zero known gaps. This statement applies to the behavior
that each ABI can express; it does not claim Python object semantics,
unrepresented arguments, or result categories that cannot cross the boundary.

## Qualified artifact

| Component | Qualified value |
|---|---|
| Operating system/architecture | Windows x64 |
| Python | 3.10.11, 64-bit |
| NumPy | 1.25.0 |
| SciPy used by owned special-function tests | 1.12.0 |
| AutoHotkey | 2.1-alpha.30, 64-bit |
| C compiler | MSVC 19.44.35228.0, v143 toolset 14.44.35207 |
| Build | x64 Release, `/O2 /GL /arch:SSE2 /fp:fast`, isolated AVX2 units |
| Runtime SIMD/thread setting | AVX2; library automatic thread count |
| cnumpy version | `1.21.0-cnumpy` |
| DLL | `build/x64/Release/cnumpy_ahk.dll` |
| DLL SHA-256 | `9b15dc7fc2a19fbb63ae919e6f3429cbd33443a04f588223c62773d6bd67e21b` |
| DLL size | 1,183,232 bytes |
| Named exports | 1,027 unique names |

MSVC/LTCG did not produce a byte-identical DLL from an unchanged final source
tree. The final hash above was therefore followed by a complete semantic,
focus, standard, and callback rerun, and no later rebuild was performed. A
source-identical rebuild with another hash is not automatically the qualified
artifact.

## Completion gates

| Gate | Result |
|---|---|
| x64 Release rebuild | PASS: zero errors; two retained C4756 warnings |
| Full Python discovery with `ResourceWarning` as error | PASS: 1,677 tests |
| AutoHotkey captured-stdout suite | PASS: 206/206; no failures, errors, or skips |
| AutoHotkey GUI/no-stdout branch | PASS: exit 0 with the expected native stdout error |
| AutoHotkey benchmark smoke | PASS: exit 0 |
| Manifest/catalog suite | PASS: 181 tests |
| Manifest declaration ownership | PASS: 752/752 |
| Effective manifest status | 655 differential, 88 characterized, 9 native, 0 known gaps |
| Final callback qualification | PASS: 9/9; all final retained-byte deltas zero |
| Final focus benchmark | PASS: 152/152 per runtime; all cnumpy retained bytes zero |
| Final standard benchmark | PASS: 455/455 per runtime; all cnumpy retained bytes zero |

Primary evidence:

- [Final convergence record](2026-08-04-performance-behavior-convergence-baseline.md)
- [Compatibility manifest](../../compat/manifest.json)
- [Compatibility-manifest tests](../../benchmark/tests/test_compat_manifest.py)
- [Functional callback differential tests](../../benchmark/tests/test_functional_callback_surface_semantics.py)
- [AutoHotkey facade and lifecycle tests](../../ahk/numpy.test.ahk)
- [Benchmark protocol](../../benchmark/README.md)

## How to read the manifest status

- **Differential** means the represented inputs, outputs, mutations, errors,
  layout, and lifecycle are compared directly with NumPy 1.25.
- **Characterized** means the public behavior is intentional and tested, but
  its ABI is a narrower projection or a project-specific contract. It is not
  described as identical to an unrepresentable NumPy call.
- **Native** means the symbol exists for ownership, interop, diagnostics, or a
  project extension rather than as a NumPy-call result analogue.
- **Known gap** would mean a public declaration lacks the required owner or has
  an unresolved claimed behavior. The final count is zero.

Zero known gaps is therefore a release-ownership result, not a claim that
cnumpy embeds Python or supports every NumPy dtype/object protocol.

## Task 10–14 coverage

| Task | Qualified public behavior |
|---|---|
| 10: random choice and permutations | Weighted/unweighted choice, replacement rules, probability validation, deterministic cnumpy generator state, permutation, in-place shuffle, strided row exchange, atomic errors, release order |
| 11: binary I/O and memory maps | NPY v1/v2, endian/layout handling, ZIP NPZ members, CRC/name/method errors, memmap modes, mapping-view ownership, Windows handle closure, error atomicity |
| 12: parsing, polynomial, special functions | Split behavior, byte-regex capture, safe numeric literals, polynomial families, represented special functions, malformed-input errors |
| 13: dtype and casting | Every represented dtype pair, five casting modes, scalar/array promotion, layout, public facade ownership, repeated release |
| 14: remaining surface and callbacks | Remaining array/linalg/runtime families; bulk callback v2; callback exception identity; arbitrary result shape; unknown iteration; select/piecewise/put-along-axis semantics |

The manifest names the direct domain-by-domain test owners. The final
performance work, retained RED/GREEN history, and rerun conditions are indexed
in the
[convergence record](2026-08-04-performance-behavior-convergence-baseline.md).

## Observable differences from NumPy 1.25

The following differences are public, intentional or legacy-constrained, and
tested. They are not hidden by automatic retries or substitute results.

### Callback projection

- Callback values are real `double` values. Object dtype, Python objects,
  general array-valued Python callables, and multi-output ufunc objects are not
  represented.
- AHK `ApplyAlongAxis` returns one scalar per callback line. The C
  `cnp_ahk_apply_along_axis_v2` export can insert an explicit result shape.
- AHK `FromIter` requires a non-negative count. C v2 accepts `count == -1` for
  unknown length and terminates on a partial produced batch.
- AHK `Piecewise` exposes one shared pointwise callback in condition order. It
  is not NumPy's general list of functions and constants.
- Callback exceptions/statuses abort atomically. There is no scalar retry.

### Legacy axis and result projections

- Several legacy indexing and reduction exports use `axis == -1` for
  `axis=None`, so they cannot also express negative last-axis indexing. v2
  exports carry a separate `axis_none` flag, and the AHK facade routes through
  them where that distinction matters.
- Legacy scalar reductions cannot preserve array-valued, complex, or
  dtype-specific results.
- Legacy partition accepts one scalar `kth`; legacy unique returns values only;
  legacy broadcast/meshgrid returns one projected result; legacy
  `where_indices` returns flat indices. Their v2 forms expose the represented
  multi-input or multi-result behavior.

### Linalg and dtype projections

- Legacy `slogdet` is packed or projected to two real scalars, and legacy
  `lstsq` returns `x` only. v2 exports return owned result arrays through
  caller-provided result slots.
- Condition-number rank deficiency is characterized across finite/infinite
  LAPACK roundoff. `matrix_rank` does not expose NumPy's complete
  tolerance/hermitian/batched argument surface.
- Signed `iinfo` accessors cannot return the full unsigned 64-bit maximum.

### Bit, text, datetime, and I/O projections

- Legacy pack/unpack exports retain historical bit-order/count sentinels.
- Legacy character split cannot return a nested Python list. Text formatting is
  expressed through caller buffers, CSV, one format, or stdout rather than the
  full NumPy formatting surface. A GUI process without stdout receives a real
  native error.
- Datetime string formatting is a single-element/day-oriented projection.

### Representation projections

- `newbyteorder` materializes byte-swapped native values because `CnpArray`
  has no byte-order descriptor.
- `recfromtxt` returns a homogeneous numeric array rather than a general
  structured Python record array.
- `item` projects to `double` and rejects complex values; `clip` takes scalar
  double bounds; `sqrt_into` is the explicit float64 C-contiguous destination
  fast path.

### Project extensions and sequences

- `bitwise_count` postdates NumPy 1.25, and the NEP 32 financial functions are
  project extensions.
- Random/matlib random uses the characterized xoshiro256**/SplitMix64 sequence,
  not NumPy's bit-generator sequence. Distribution semantics are qualified;
  seeded element-for-element sequence identity with NumPy is not claimed.
- Ownership administration (`incref`, `decref`, owned strings/buffers/masked
  arrays) and ctypes accessors have no NumPy-call return analogue. They are
  tested directly for correct release and zero retained bytes.

## Legacy-to-v2 migration

### Preserve `axis=None` versus the last axis

Legacy code overloads `-1`:

```c
/* Legacy: -1 means axis=None, not the last axis. */
double total = cnp_ahk_sum(array, -1);
```

Use a v2 array result when the distinction matters:

```c
/* Last axis: axis=-1, axis_none=0. */
CnpArray *last_axis = (CnpArray *)cnp_ahk_sum_v2(array, -1, 0);

/* All axes: axis value is ignored when axis_none=1. */
CnpArray *all_axes = (CnpArray *)cnp_ahk_sum_v2(array, 0, 1);
```

Both returned pointers are owned and must be released. The AutoHotkey facade
already makes this distinction:

```ahk
allAxes := Numpy.Sum(source)
lastAxis := Numpy.Sum(source, -1)
```

### Preserve callback result shape and status

Legacy `cnp_ahk_apply_along_axis` returns one real scalar per line and its
callback crosses the boundary once per line. The v2 form accepts a batch
callback and an explicit result shape:

```c
const int64_t result_shape[] = {2};
CnpArray *result = (CnpArray *)cnp_ahk_apply_along_axis_v2(
    summarize_lines, userdata, axis, source, 1, result_shape);
```

Set `*produced_count` on every successful callback and return a real
`CNP_STATUS`. A nonzero status terminates the operation and is preserved in the
native error state. See the executable
[C bulk callback example](../../examples/c/bulk_callbacks.c) and the
[bulk callback guide](../guides/bulk-callbacks.md).

### Use unknown-length iteration only through C v2

```c
CnpArray *result = (CnpArray *)cnp_ahk_fromiter_v2(
    next_values, userdata, -1, CNP_DOUBLE);
```

For unknown length, producing fewer values than `result_capacity` signals
exhaustion. For a fixed count, a short production is an error. The AHK facade
rejects negative count before it registers or invokes a callback.

## Performance status and remaining headroom

The final standard comparison times the public NumPy call against AHK
`DllCall` plus native work and result lifecycle. Ratios are cnumpy/NumPy; lower
is better for cnumpy. No bridge-overhead subtraction is applied.

The highest-value remaining work is ranked by absolute time:

1. Million-element heapsort/argsort and set construction remain roughly
   63–191 ms behind NumPy. Typed radix/timsort families, shared sorted-key
   reuse, and fewer copies have the largest expected return.
2. Callback batching reduces boundary calls but not scalar callback work.
   Final ratios include `fromfunction` 174.179x, `piecewise` 49.248x, and
   `apply_over_axes` 35.002x. An array-valued coordinate/mask or opaque
   reduction-state contract is needed; a larger batch alone is insufficient.
3. Non-contiguous axis kernels remain costly: axis-0 `trapz` is about 4.92x
   NumPy with about 7.86 ms excess, and axis-0 softmax/log-softmax are about
   2.06–2.24x. Blocked or stride-specialized kernels are the next route.
4. Native math/linalg gaps include about 9.49 ms excess for one-million-element
   `sin`, about 9.30 ms for 128x128 `eig`, and about 2.3–2.4 ms for 512x512
   dot/matmul. Vector math and tuned BLAS/LAPACK have higher leverage than AHK
   wrapper micro-optimization.
5. Metadata/view calls show large ratios but only about 13–20 microseconds
   absolute time. Cached function addresses and more composed batch operations
   can reduce the fixed AHK boundary, but this is lower priority than the
   native kernels above.

The [final convergence record](2026-08-04-performance-behavior-convergence-baseline.md)
retains four unresolved allocation/cache/link-layout-sensitive movements from
the final regression diagnostic:
`linspace/100000`, `invert/100000`, `ones/100000`, and `floor/1000000`. Their
implementations were not changed in the convergence wave, adjacent sizes did
not show the same pattern, and all diagnostic records retained zero bytes.
They remain reported rather than being relabeled as algorithmic improvements
or hidden through repeated full runs.

## Retained warnings and failure history

The final Release rebuild has zero errors and two pre-existing warnings:

- `src/masked.c:287`: MSVC C4756, overflow in constant arithmetic.
- `src/masked.c:307`: MSVC C4756, overflow in constant arithmetic.

The oracle/test runs also retain their real warnings:

- SciPy emitted a `RuntimeWarning` from `poch` for an invalid subtraction.
- NumPy 1.25 emitted its `np.msort` deprecation warning in focus/standard
  workers.

Important failures found during Task 14 were fixed at their source and remain
recorded in the convergence evidence:

- The first arbitrary-result-shape bulk differential exposed a real output
  layout bug before the final `(2, 2, 4)` line order matched NumPy.
- AHK `FromIter(-1)` originally reached and failed inside the callback. The
  facade now rejects the negative count before callback registration; C v2
  intentionally retains `-1` for unknown length.
- An intermediate complete AHK run retained the GUI-only stdout assertion as a
  real failure even though the process exited zero. The final qualification
  separately owns captured-stdout and GUI/no-stdout behavior.
- Early line-batch experiments regressed because legacy and v2 measured
  different user-function layers. Both failure artifacts were retained; the
  final equal-work comparison reduced 64 crossings to one and was rerun after
  the final rebuild.

No failure was converted to a skip, mock result, fallback success, silent cap,
or scalar retry.

## Reproduce the public gates

From the repository root, with Python 3.10/NumPy 1.25 and a 64-bit AutoHotkey
v2 executable available:

```powershell
python -B -W error::ResourceWarning -m unittest discover -s benchmark\tests -v
& $Ahk /ErrorStdOut=UTF-8 ahk\numpy.test.ahk
& $Ahk /ErrorStdOut=UTF-8 benchmark\benchmark_smoke.test.ahk
python benchmark\benchmark.py --profile focus --warmups 5 --samples 15 --target-sample-ms 20
python benchmark\benchmark.py --profile standard --warmups 5 --samples 15 --target-sample-ms 20
```

Fresh benchmark runs are new artifacts and may reflect current host load. Use
their embedded environment, DLL hash, runtime order, sample count, validation
status, coefficients of variation, and retained-byte fields when comparing
results.
