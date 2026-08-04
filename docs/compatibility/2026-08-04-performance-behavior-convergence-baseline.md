# Performance and behavior convergence baseline — 2026-08-04

## Binary and protocol

- Release DLL: `build/x64/Release/cnumpy_ahk.dll`
- SHA-256: `59b7f42dc914d68e2afd4d1ba1e650767a209c13b336131b35c87f060871b4ac`
- Size: 1,159,168 bytes
- Primary artifact: `benchmark/results/20260804T043133.865080Z`
- NumPy/Python: NumPy 1.25.0 on Python 3.10.11
- cnumpy/AHK: cnumpy 1.21.0 on AutoHotkey 2.1-alpha.30
- SIMD/thread setting: AVX2, automatic thread count
- Timing: five warmups, fifteen samples, 20 ms target sample, seed 12345
- The current Release DLL was re-hashed before implementation and exactly
  matches the primary artifact.

## Affected primary benchmark rows

Times are medians in milliseconds.  The ratio is cnumpy divided by NumPy.

| Case | NumPy ms | cnumpy ms | Ratio |
| --- | ---: | ---: | ---: |
| `sort/f64/1000000` | 64.629 | 241.959 | 3.744x |
| `argsort/f64/1000000` | 93.201 | 235.022 | 2.522x |
| `sort_heapsort/f64/1000000` | 152.240 | 710.469 | 4.667x |
| `argsort_heapsort/f64/1000000` | 354.113 | 706.221 | 1.994x |
| `unique_duplicates/f64/1000000` | 35.989 | 224.703 | 6.244x |
| `in1d_duplicates/f64/1000000` | 91.959 | 507.151 | 5.515x |
| `isin_duplicates/f64/1000000` | 93.442 | 502.605 | 5.379x |
| `argmax/f64/1000000` | 0.387 | 8.870 | 22.942x |
| `max/f64/1000000` | 0.274 | 2.451 | 8.939x |
| `min/f64/1000000` | 0.267 | 2.371 | 8.876x |
| `sum/f64/1000000` | 0.506 | 0.716 | 1.415x |
| `sum_axis_last/f64/512x512/axis1` | 0.117 | 2.450 | 21.023x |
| `prod/f64/1000000` | 1.383 | 2.972 | 2.148x |
| `packbits/u8/1000000` | 0.075 | 3.752 | 50.240x |
| `packbits_axis_last/u8/512x512/axis1` | 0.024 | 1.064 | 44.165x |
| `packbits_axis0_strided/u8/512x512/axis0` | 0.025 | 1.037 | 40.952x |
| `unpackbits/u8/1000000` | 1.991 | 3.432 | 1.724x |
| `unpackbits_axis_last/u8/512x512/axis1` | 0.559 | 1.645 | 2.944x |
| `unpackbits_axis0_strided/u8/512x512/axis0` | 0.569 | 13.730 | 24.148x |
| `trapz/f64/1000000` | 7.282 | 38.923 | 5.345x |
| `trapz_axis_last/f64/512x512/axis1` | 2.330 | 11.237 | 4.823x |
| `trapz_axis0_strided/f64/512x512/axis0` | 2.256 | 10.864 | 4.817x |

All primary cnumpy case records reported `retained_bytes: 0`.

## Callback-boundary artifact

Source: `benchmark/runs-task14-functional-callback-surface/qualification.json`.

| Case | NumPy ms | cnumpy ms | Ratio | Final retained bytes |
| --- | ---: | ---: | ---: | ---: |
| `apply_over_axes/sum/f64/16x16x16/axes0-2` | 0.0166 | 0.672 | 40.465x | 0 |
| `fromfunction/f64/64x64` | 0.0211 | 5.848 | 276.507x | 0 |
| `fromiter/f64/8192` | 0.781 | 13.573 | 17.386x | 0 |
| `piecewise/f64/8192/two-conditions` | 0.0383 | 1.975 | 51.576x | 0 |

## Before-state behavior gate

The six directly affected semantic modules were run against the identical DLL:

```text
Ran 216 tests in 7.326s

OK
```

Modules: sort/set, reductions, miscellaneous axis/bit/trapz, functional
callbacks, index surface, and linalg/statistics surface.  The separately rerun
compatibility manifest gate reported 111 passing tests.  These results are the
behavior/lifecycle before-state; later wave evidence must compare against them
and the primary qualification totals rather than a reduced substitute.

## Wave A — sorting and set operations

Implementation artifact:
`benchmark/results/20260804T053241.466024Z`.  It uses the same warmups, samples,
target sample duration, seed, AVX2 setting, and automatic thread setting as the
primary baseline.  All seven cnumpy cases retained zero bytes.

| Case | Before ms | Wave A ms | Direct speedup |
| --- | ---: | ---: | ---: |
| `sort/f64/1000000` | 241.959 | 201.349 | 1.20x |
| `sort_heapsort/f64/1000000` | 710.469 | 449.196 | 1.58x |
| `unique_duplicates/f64/1000000` | 224.703 | 193.714 | 1.16x |
| `in1d_duplicates/f64/1000000` | 507.151 | 78.097 | 6.49x |
| `isin_duplicates/f64/1000000` | 502.605 | 89.287 | 5.63x |

Absolute argsort times moved with system load: the same-run NumPy quicksort
time was 133.132 ms versus 93.201 ms in the earlier standard artifact.  A
single-variable diagnostic therefore compared the specialized and generic
cnumpy routes in the same session:

| Case | Specialized ms | Generic ms | Specialized advantage |
| --- | ---: | ---: | ---: |
| `argsort/f64/1000000` | 264.666 | 349.896 | 24.4% less time |
| `argsort_heapsort/f64/1000000` | 777.431 | 949.348 | 18.1% less time |

Specialized artifact:
`benchmark/results/20260804T053451.102825Z`; generic diagnostic artifact:
`benchmark/results/20260804T053702.646304Z`.  Both specialized runs also
improved cnumpy/NumPy ratios relative to the original baseline.  The generic
variant was not retained.

The restored Wave A DLL has SHA-256
`63ed1fe24254edb7da74b4959148a1d954e6cc03aeb9567c19093dcfa5519d56`.
Release built successfully; the only reported warnings were the two pre-existing
`masked.c` C4756 constant-arithmetic warnings, outside this wave.  The combined
sort/set, optimization-contract, and manifest gate ran 186 tests successfully.

## Wave B — reductions

Implementation artifact:
`benchmark/results/20260804T055817.148186Z`.  Its Release DLL has SHA-256
`73b824753d497de67a597eef82d091b25aecabe9c2503e08014d7638a0db9f93`
and size 1,164,800 bytes.  The artifact used five warmups, fifteen samples,
20 ms target samples, seed 12345, AVX2, and automatic thread count.  Every
cnumpy record reported `retained_bytes: 0`.

| Case | NumPy median ms | cnumpy median ms | Ratio | NumPy CV | cnumpy CV |
| --- | ---: | ---: | ---: | ---: | ---: |
| `argmax/f64/1000000` | 1.265 | 1.928 | 1.524x | 0.043 | 0.084 |
| `max/f64/1000000` | 0.820 | 2.671 | 3.259x | 0.137 | 0.061 |
| `min/f64/1000000` | 0.836 | 2.638 | 3.156x | 0.064 | 0.165 |
| `prod/f64/1000000` | 1.802 | 2.383 | 1.322x | 0.026 | 0.128 |
| `sum/f64/1000000` | 0.950 | 1.370 | 1.442x | 0.226 | 0.227 |
| `sum_axis_last/f64/512x512/axis1` | 0.156 | 0.213 | 1.365x | 0.062 | 0.045 |

The direct typed route reduced the original cnumpy median from 8.870 ms to
1.928 ms for `argmax` (4.60x direct speedup), from 2.450 ms to 0.213 ms for
last-axis `sum` (11.49x), and from 2.972 ms to 2.383 ms for `prod` (1.25x).
Cross-run absolute max/min and sum timings were noisy, so route-level
single-variable artifacts were retained rather than claiming those medians as
repeatable baseline speedups.  An attempted four-item unroll artifact
(`20260804T054826.757727Z`) exposed a 27.2% argmax regression and a 10.2% min
regression; that change was reverted.  The final four-lane max/min route was
kept only after the paired control runs showed approximately 13–17% less time.

The reduction, optimization-contract, and manifest gate ran 233 tests.  The
expanded bitwise differential included axis lengths 1–17, NaN positions and
payloads, alternating signed zero, first ties, empty/error cases, strided
inputs, result ownership, and retained-byte balance.  The original RED run
reported 32 payload mismatches; the minimized six short explicit-axis
mismatches identified NumPy 1.25's axis-length 2–4 initial-NaN
canonicalization and the final implementation preserves that behavior.

## Wave C — bit packing and trapezoidal integration

Final artifact: `benchmark/results/20260804T061938.495588Z`.  Its Release DLL
has SHA-256
`0ad24283011f78365594d5e2ad8c86082f94dfaa88d1bfdb8a47ac0d273eaba2`
and size 1,168,896 bytes.  All nine records retained zero bytes.

| Case | Before cnumpy ms | Wave C cnumpy ms | Direct speedup | Final cnumpy/NumPy |
| --- | ---: | ---: | ---: | ---: |
| `packbits/u8/1000000` | 3.752 | 0.360 | 10.42x | 2.598x |
| `packbits_axis_last/u8/512x512/axis1` | 1.064 | 0.116 | 9.19x | 2.701x |
| `packbits_axis0_strided/u8/512x512/axis0` | 1.037 | 0.363 | 2.86x | 8.745x |
| `unpackbits/u8/1000000` | 3.432 | 3.333 | 1.03x | 1.190x |
| `unpackbits_axis_last/u8/512x512/axis1` | 1.645 | 0.940 | 1.75x | 1.102x |
| `unpackbits_axis0_strided/u8/512x512/axis0` | 13.730 | 3.364 | 4.08x | 4.214x |
| `trapz/f64/1000000` | 38.923 | 5.532 | 7.04x | 0.433x |
| `trapz_axis_last/f64/512x512/axis1` | 11.237 | 0.291 | 38.67x | 0.087x |
| `trapz_axis0_strided/f64/512x512/axis0` | 10.864 | 18.459 | cross-run load | 4.660x |

The last axis-0 row did not select the new contiguous trapz path.  Host load
moved NumPy from 2.256 ms in the original artifact to 3.961 ms in the Wave C
artifact by almost the same proportion, while the same-run ratio improved from
4.817x to 4.660x; it is therefore recorded without an absolute speedup claim.

The first GREEN artifact was `20260804T061017.291107Z`.  A nibble lookup and
interleaved-output unpack pass then reduced the vector/last-axis ratios from
2.331x/2.017x to 1.047x/0.954x and the axis-0 ratio from 15.456x to 3.939x in
`20260804T061454.884098Z`.  The retained SSE2 pack path reduced vector and
last-axis time by a further 3.24x and 2.43x.  A proposed pack axis-0 output
reordering regressed that case from 0.322 ms to 0.463 ms (43.9%); the true
failure artifact `20260804T061801.965852Z` was retained and only that route was
reverted.

The RED trapz differential exposed 1–9 ULP bit-pattern mismatches at lengths
127, 128, 129, and 257.  The final contiguous helper constructs NumPy's typed
panel values and reuses the qualified NumPy pairwise sum tree; the bitwise
differential now passes.  The final misc-axis, optimization-contract, and
manifest gate ran 302 tests.  The affected AHK misc-axis facade test passed and
the AHK process returned 0; the complete suite log is audited explicitly in
Wave D below.

## Wave D — representable v2 behavior

The Release DLL after Wave D has SHA-256
`225c21dac8e643e6b8fcbf7ac24218ae4ec174baeec58637c8a805eb89c6cfe1`
and size 1,170,944 bytes.  The RED gate reported the two missing DLL exports,
two missing public declarations, two missing manifest owners, AHK's
`Too many parameters passed to function`, and the legacy tensorsolve `-12`
axes-length limitation.

`cnp_linalg_tensorsolve_v2(a, b, naxes, axes)` now reproduces NumPy 1.25's
ordered move-to-the-right permutation for empty, single, multiple, and repeated
axes.  NumPy 1.25's implementation rejects negative axes via `list.remove`, so
v2 rejects negative and out-of-range axes explicitly.  Negative lengths and a
positive length with a null pointer are also explicit, atomic errors.  The
legacy export remains callable and still rejects every non-null axes pointer.

`cnp_count_nonzero_v2(arr, axis, axis_none, keepdims)` returns owned int64
arrays for axis=None and every representable single axis, including scalar and
zero-length dimensions.  The AHK facade returns an Integer only for omitted
axis with `keepdims=false`; explicit axes and keepdims results are NdArray
values.  The legacy scalar ABI remains unchanged.

The expanded linalg/index, optimization-contract, export, and manifest gate
ran 281 tests.  The affected AHK linalg and index facade tests both passed and
the suite log reported 204 passes, zero errors, and one existing GUI-only text
I/O stdout assertion; the AHK process nevertheless returned 0, so the log
failure is retained and is not represented as a clean complete-suite gate.

## Wave E — bulk callback ABI and public facade

The RED contracts required seven explicit v2 callback exports.  Their callback
signatures carry input/output pointers, a batch or capacity count, a produced
count where applicable, userdata, and a returned `CNP_STATUS`.  The first
arbitrary-result-shape `apply_along_axis` differential used an int32 input with
shape `(2, 3, 4)`, axis 1, and a two-value callback result.  It exposed a real
result-layout bug before the final implementation reproduced NumPy's
`(2, 2, 4)` values and line order.  Callback errors remain labeled native
errors, results are atomic, and no v2 adapter retries through a scalar ABI.

The facade routes `ApplyAlongAxis`, `ApplyOverAxes`, `FromFunction`, `FromIter`,
`FromPyFunc`, `Piecewise`, and `Vectorize` exclusively through their v2 AHK
exports while all seven legacy exports remain callable.  `FromIter(-1)` also
produced a retained AHK RED failure: the callback was reached and raised
`negative FromIter count reached the callback`.  The final facade rejects the
negative count before callback registration with an explicit non-negative
count error.  The C v2 export intentionally retains `count == -1` as its
partial-production/unknown-length capability.

The full-result differential covers success, empty inputs, arbitrary result
shape, batch counts, unknown and fixed iterator production, callback failure,
source-first release and retained-byte restoration.  The AHK test additionally
proves callback registration cleanup and exception propagation.  The final
AHK suite records 206 passes, zero failures, zero errors and zero skips.

### Real AHK callback boundary result

Artifact:
`benchmark/runs-task14-functional-callback-surface/ahk-bulk-v2-vs-legacy-rebuild-final.json`.
It was run after the final rebuild against the DLL hash recorded below.  Times
are medians; all eight records retained zero bytes.

| Callback kind | Legacy ms | v2 ms | Direct speedup | Legacy to v2 calls |
| --- | ---: | ---: | ---: | ---: |
| line | 0.894 | 0.818 | 1.093x | 64 to 1 |
| coordinate | 3.105 | 2.225 | 1.396x | 4096 to 16 |
| iterator | 6.361 | 1.809 | 3.517x | 8192 to 32 |
| unary | 5.838 | 2.785 | 2.096x | 8192 to 32 |

The earlier asymmetric line benchmark first showed a bulk regression.  A
pointer-`while` experiment regressed it further.  The failure artifacts
`ahk-line-scale.json` and `ahk-line-scale-pointer.json` are retained.  The
root cause was unequal callback work rather than the crossing count: legacy
and v2 executed different user-function layers.  With both paths executing the
same user function, the final line case is modestly faster while reducing the
crossing count from 64 to one.  No failed experiment was overwritten.

### Final callback-to-NumPy qualification

Artifact:
`benchmark/runs-task14-functional-callback-surface/bulk-v2-qualification-rebuild-final.json`.
It embeds SHA-256
`9b15dc7fc2a19fbb63ae919e6f3429cbd33443a04f588223c62773d6bd67e21b`.
All nine cases have a zero final retained-byte delta.

| Case | cnumpy/NumPy | Bulk calls / logical elements |
| --- | ---: | ---: |
| `apply_along_axis/sum/f64/64x64` | 1.311x | 1 / 64 |
| `apply_over_axes/sum/f64/16x16x16/axes0-2` | 35.002x | 2 / 272 |
| `fromfunction/f64/64x64` | 174.179x | 16 / 4096 |
| `fromiter/f64/8192` | 1.497x | 32 / 8192 |
| `frompyfunc/f64/8192` | 1.317x | 32 / 8192 |
| `vectorize/f64/8192` | 1.224x | 32 / 8192 |
| `piecewise/f64/8192/two-conditions` | 49.248x | 24 / 6144 |
| `select/f64/128x128/two-conditions` | 4.901x | n/a |
| `put_along_axis/f64/128x128/64-updates` | 6.445x | n/a |

## Final frozen binary and complete gates

The final source-complete x64 Release rebuild succeeded with zero errors.  It
reported only the two pre-existing `masked.c` C4756 constant-arithmetic
warnings.  MSVC/LTCG is not byte-reproducible under this project configuration:
an unchanged rebuild changed the previous hash `33f1...ac5` to the final hash
below while retaining the same file size.  This was not ignored.  Focus,
standard, callback qualification and every semantic gate were rerun after the
hash change, and no further rebuild was performed.

- Final DLL SHA-256:
  `9b15dc7fc2a19fbb63ae919e6f3429cbd33443a04f588223c62773d6bd67e21b`
- Size: 1,183,232 bytes
- Build verification log:
  `benchmark/runs-task14-functional-callback-surface/final-release-rebuild-verification.log`
- Named exports: 1,027 unique names
- Callback exports: all 14 legacy/v2 names present
- Manifest: 752 declarations and 752 resolved owners: 655 differential,
  88 characterized, 9 native, 0 known gaps
- Full Python discovery with `ResourceWarning` promoted to errors: 1,677 tests,
  zero failures
- AHK captured-stdout suite: 206/206; stdout contains the expected array and
  `AHK text facade output`
- AHK GUI/no-stdout suite: exit 0 and the expected native stdout-error branch
- AHK benchmark smoke: exit 0
- Manifest/catalog gate: 181 tests, zero failures

The Python run retained SciPy's visible `RuntimeWarning` from `poch` on an
invalid subtraction.  The standard/focus NumPy workers retained the visible
NumPy 1.25 `np.msort` deprecation warning.  Neither warning was swallowed or
converted into a false clean log.

## Final focus and standard performance gates

Both artifacts below embed the final DLL hash, use NumPy-first runtime order,
five warmups, fifteen samples, 20 ms target samples, seed 12345, AVX2 and
automatic cnumpy thread count.

- Focus:
  `benchmark/runs-task14-final-focus-rebuild/20260804T080950.953958Z`
  contains 152/152 NumPy and cnumpy records, 77 semantically qualified rows,
  and zero retained bytes.
- Standard:
  `benchmark/runs-task14-final-standard-rebuild/20260804T081716.048501Z`
  contains 455/455 records with identical IDs, 114 semantically qualified rows,
  and zero retained bytes.

The final standard artifact gives the following direct before/final results for
the optimized large or axis cases.  Cross-run absolute ratios are evidence,
not a claim that host load is identical; the largest improvements are also
supported by the wave-local controlled artifacts above.

| Case | Before ms | Final ms | Direct speedup | Final cnumpy/NumPy |
| --- | ---: | ---: | ---: | ---: |
| `sort/f64/1000000` | 241.958 | 130.664 | 1.85x | 2.18x |
| `argsort/f64/1000000` | 235.022 | 159.451 | 1.47x | 1.87x |
| `sort_heapsort/f64/1000000` | 710.469 | 308.601 | 2.30x | 2.20x |
| `unique_duplicates/f64/1000000` | 224.703 | 108.589 | 2.07x | 3.27x |
| `in1d_duplicates/f64/1000000` | 507.151 | 51.397 | 9.87x | 0.60x |
| `isin_duplicates/f64/1000000` | 502.605 | 54.369 | 9.24x | 0.60x |
| `argmax/f64/1000000` | 8.870 | 1.744 | 5.09x | 5.13x |
| `sum_axis_last/f64/512x512/axis1` | 2.450 | 0.114 | 21.56x | 1.04x |
| `prod/f64/1000000` | 2.972 | 1.347 | 2.21x | 1.03x |
| `packbits/u8/1000000` | 3.752 | 0.185 | 20.25x | 2.77x |
| `unpackbits_axis0_strided/u8/512x512/axis0` | 13.730 | 2.107 | 6.52x | 4.27x |
| `trapz/f64/1000000` | 38.923 | 2.940 | 13.24x | 0.42x |
| `trapz_axis_last/f64/512x512/axis1` | 11.237 | 0.172 | 65.17x | 0.08x |

### Regression investigation

The first final-source standard artifact found seven cases with both absolute
and NumPy-normalized changes above 10 percent.  A 25-sample, 50 ms/sample
diagnostic removed four and retained isolated `invert/100000` and
`linspace/100000` movements.  Reversing runtime order showed large phase
sensitivity.  The subsequent unchanged rebuild changed 58 cnumpy cases by
more than +10 percent and 51 by more than -10 percent relative to that first
final-source run, demonstrating that these microbenchmarks are sensitive to
host phase and link/code layout.

The final-hash standard produced eleven candidates.  Every one was rerun with
25 samples and 50 ms/sample in
`benchmark/runs-task14-final-rebuild-regression-diagnostic/20260804T083126.752217Z`.
Seven no longer had an absolute cnumpy regression above 10 percent.  Four
remained observable relative to the old baseline:

| Case | Diagnostic cnumpy/baseline | Diagnostic cnumpy/NumPy | cnumpy CV |
| --- | ---: | ---: | ---: |
| `linspace/f64/100000` | 2.750x | 1.106x | 0.066 |
| `invert/i64/100000` | 1.852x | 2.938x | 0.097 |
| `ones/f64/100000` | 1.853x | 5.584x | 0.082 |
| `floor/f64/1000000` | 1.178x | 1.370x | 0.106 |

The implementations and AHK wrappers for these four operations were not
modified by this convergence wave, and adjacent sizes do not show the same
pattern.  The result is therefore recorded as an unresolved allocation/cache/
link-layout-sensitive microperformance issue, not mislabeled as an algorithmic
regression and not hidden by rerunning the full standard suite until green.
All eleven diagnostic records retained zero bytes.

## Remaining performance headroom

The final standard artifact still shows clear high-value work, ranked by
absolute time rather than headline ratio alone:

1. Sorting and set construction dominate absolute excess.  Million-element
   heapsort/argsort have about 169/191 ms excess versus NumPy; unique, union,
   intersection, difference and xor remain about 63--76 ms behind.  A typed
   radix/timsort family, shared sorted-key reuse, and fewer allocation/copy
   passes are the next highest-return native work.
2. Callback projections still cross into AHK in batches but execute scalar
   callback semantics within each batch.  `fromfunction` is 174.2x,
   `piecewise` 49.2x and `apply_over_axes` 35.0x NumPy.  Closing this requires
   a new array-valued callback contract for broadcast coordinates/masks and
   opaque reduction state; increasing the existing batch size alone cannot
   reproduce NumPy's vectorized callback model.
3. Non-contiguous axis kernels remain expensive.  Axis-0 strided `trapz` is
   4.92x NumPy with about 7.86 ms excess; axis-0 softmax/log-softmax are about
   2.06--2.24x.  Blocked gather/compute/scatter or stride-specialized kernels
   are the likely next route.
4. General math and linalg still have native kernel gaps: million-element
   `sin` has about 9.49 ms excess, 128x128 `eig` about 9.30 ms, and 512x512
   dot/matmul about 2.3--2.4 ms.  Vector math and tuned BLAS/LAPACK integration
   would be higher leverage than further AHK wrapper micro-optimization.
5. Metadata/view operations show 40--205x ratios but only about 13--20
   microseconds absolute cnumpy time.  This is the fixed AHK `DllCall`, wrapper
   and ownership boundary.  Caching more function pointers and composing more
   operations into existing batch APIs can reduce it; it is lower priority
   than the absolute-cost kernels above.

## Still-observable NumPy differences

The release manifest has zero unknown gaps, but `characterized` means the
difference is public and tested, not that it is identical.  The 88
characterized declarations fall into the following observable groups.

- Callback facade: callbacks carry real `double` values only.  Object dtype,
  NumPy multi-output ufuncs and general array-valued Python callables are not
  represented.  The AHK `ApplyAlongAxis` facade returns scalar callback results;
  the C v2 ABI can express an explicit result shape.  AHK `FromIter` requires a
  non-negative count, while C v2 alone supports `-1` unknown length.
  `Piecewise` exposes one shared pointwise callable rather than NumPy's list of
  functions/constants.
- Legacy axis/result ABIs: legacy indexing and reduction exports overload
  `axis == -1` as `axis=None`, so they cannot express negative last axis;
  v2/facade exports remove that ambiguity.  Legacy scalar reductions cannot
  preserve array-valued, complex or dtype-specific results.  Legacy partition
  accepts scalar `kth` only, unique returns values only, broadcast_arrays and
  meshgrid return one projected result, and where_indices returns flat indices.
- Legacy linalg/dtype ABIs: slogdet is packed or two real scalars, lstsq returns
  x only, condition-number rank deficiency is characterized across LAPACK
  finite/infinite roundoff, matrix_rank lacks full tol/hermitian/batch
  expression, and signed iinfo accessors cannot return uint64 max.
- Bit, character, datetime and I/O projections: legacy pack/unpack use the
  historical bitorder/count sentinels; legacy char split cannot return nested
  lists; datetime string formatting is single-element/day projected; text
  formatting is caller-buffer/CSV/single-format/stdout based rather than the
  complete NumPy formatting surface.  GUI processes without stdout expose a
  real native error.
- Representation projections: `newbyteorder` materializes byte-swapped native
  values because `CnpArray` has no byteorder descriptor; `recfromtxt` returns a
  homogeneous numeric array rather than a general structured record; `item`
  projects to double and rejects complex; `clip` requires scalar double bounds;
  `sqrt_into` is the explicit float64 C-contiguous destination fast path.
- Project extensions/sequences: `bitwise_count` postdates NumPy 1.25, the NEP
  32 financial functions remain project extensions, and random/matlib random
  use the characterized xoshiro256**/SplitMix64 sequence rather than NumPy's
  bit-generator sequence.
- The remaining characterized symbols are ownership/lifecycle administration
  (`incref`, `decref`, owned buffers/strings/masked arrays) or ctypes accessors;
  they have no NumPy-call return analogue but are directly tested for exact
  release and zero retained bytes.
