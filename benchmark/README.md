# cnumpy vs NumPy benchmark suite

This directory contains the reproducible performance gate for cnumpy. It compares the two public user paths:

- Python calling public NumPy APIs.
- AutoHotkey v2 calling exported cnumpy APIs through `DllCall`, including marshalling, C execution, returned-array allocation, and release.

The headline numbers do not subtract AHK/DLL bridge overhead. Six bridge cases make that cost visible: `property_call` is a raw allocation-free DllCall; `property_cached`, `nbytes_cached`, `c_contiguous_cached`, and `f_contiguous_cached` read the lazy AHK metadata cache; and `static_add_call` measures the NumPy-style `np.add(a, b, out)` facade with a preallocated destination. The comparison family includes allocation-free `allclose` scalar-result cases at vector scale.

## Prerequisites

- Windows x64.
- Python 3.10 with NumPy 1.25.0 and SciPy 1.12.0 installed. Task 8 uses
  `scipy.special.softmax` and `scipy.special.log_softmax` as its pinned
  reference implementations.
- AutoHotkey v2 x64.
- A built x64 `cnumpy_ahk.dll`, or Visual Studio 2022/v143 when using `--build`.

Repository defaults discover:

- `build/x64/Release/cnumpy_ahk.dll`.
- AutoHotkey from an explicit `--ahk` path, `HKLM\SOFTWARE\AutoHotkey\InstallDir`, then the documented project installation.
- MSBuild only when `--build` is requested or `--msbuild` is explicitly supplied.

An explicitly supplied invalid path is always fatal; it is never replaced with a discovered path.

## Quick start

Run the optimization-focused profile and rebuild the real Release DLL:

```powershell
F:\Python\Python310\python.exe benchmark\benchmark.py --profile focus --build
```

Run every normal workload without rebuilding:

```powershell
python benchmark\benchmark.py --profile standard
```

Select serial or automatic GEMM scheduling explicitly. The requested value is
recorded as `metadata.num_threads` in `cnumpy.json`; invalid values fail the run.

```powershell
$env:CNP_NUM_THREADS = '1'  # serial GEMM
python benchmark\benchmark.py --profile standard --case matmul/f64/512x512

$env:CNP_NUM_THREADS = '0'  # automatic private thread pool
python benchmark\benchmark.py --profile standard --case matmul/f64/512x512
```

Run all cases, including the four heavy 512×512 cubic cases: determinant,
inverse, solve, and Cholesky:

```powershell
python benchmark\benchmark.py --profile full --samples 15 --target-sample-ms 20
```

Run a fast integration smoke using canonical small dimensions:

```powershell
python benchmark\benchmark.py --profile standard --size-scale smoke --warmups 1 --samples 3 --target-sample-ms 1
```

`smoke` collapses repeated sizes to one 8-element vector or 2×2 matrix per operation. It verifies dispatch, schema, correctness, cleanup, and reporting; it is not a production performance result.

The 2026-08-04
[authoritative bilingual report](../docs/performance/2026-08-04-authoritative-performance-report.md)
publishes the complete 459-case primary run, reversed-order control, 25-sample
stability diagnostic, exact CSV evidence, behavior/lifecycle gates, and a
host-scoped interpretation. Read its limitations before using any ratio as a
deployment claim.

## Profiles

| Profile | Cases | Intended use |
|---|---:|---|
| `focus` | 152 | Bridge plus zeros, 1M weighted choice, angle, real/imag/real_if_close, allclose, reductions (including weighted average), 10K/1M quick/merge/heap/stable sorting, lexsort, msort, and sort_complex, duplicate-heavy and NaN-heavy partition/argpartition, left/right searchsorted, increasing/decreasing digitize, duplicate-heavy and NaN-heavy unique/intersection/union/difference/xor/membership set workloads, copy/reshape/flatten/atleast_1d/2d/3d, vector and axis-0 block/strided take/compress, Task 8 vector and contiguous/strided matrix softmax, log-softmax, trapezoidal integration, packbits, and unpackbits, preallocated/pipeline paths, and 512×512 concatenate bottlenecks. |
| `standard` | 455 | All operations at normal vector, matrix, FFT, and bridge sizes, excluding the four heavy 512×512 cubic cases. |
| `full` | 459 | Complete catalog; adds only 512×512 determinant, inverse, solve, and Cholesky to `standard`. |

The default profile is `focus`. Exact case and category filters may be repeated:

```powershell
python benchmark\benchmark.py --profile standard `
  --case argmax/f64/1000000 `
  --case cumsum/f64/1000000

python benchmark\benchmark.py --profile standard --category reduction
```

When both filter types are present, a case must match both. Unknown or empty selections fail explicitly.

Useful options:

```text
--build
--warmups N
--samples ODD_N
--target-sample-ms FLOAT
--seed 0..2147483647
--runtime-order numpy,cnumpy | cnumpy,numpy
--python PATH
--ahk PATH
--dll PATH
--msbuild PATH
--project-file PATH
--output-root DIRECTORY
--baseline comparison.csv
```

## Timing protocol

Both workers perform untimed semantic validation, configured warmups, calibrated batching, and an odd number of measured samples. Calibration doubles `inner_loops` until a batch reaches the requested target duration. Operations that already exceed the target use one iteration.

- Python uses `time.perf_counter_ns`.
- AHK uses `QueryPerformanceCounter` and `QueryPerformanceFrequency`.
- Setup inputs are outside the timed region.
- Newly returned arrays are created and destroyed inside each timed invocation; explicit `out` cases reuse their preallocated destination.
- Samples are stored as raw nanoseconds per public operation.
- DllCall overhead remains inside cnumpy measurements.

Shared Python reporting calculates median, mean, min/max, population standard deviation, coefficient of variation, p05/p25/p75/p95 with linear interpolation, and median absolute deviation. The primary ratio is:

```text
cnumpy_over_numpy = cnumpy_median_ns / numpy_median_ns
```

Category aggregates are geometric means of per-case ratios. Timings for unrelated operations are never summed into an “overall speed” value.

Default production settings are five warmups, fifteen measured samples, a 20 ms
target batch, and deterministic seed `12345`. Performance investigations should
keep the machine on AC power, close competing CPU-intensive tasks, and rerun
noisy cases marked with CV above 5%.

## Deterministic workloads

Both runtimes reproduce the same float64 input formulas using zero-based index `i`:

```text
general(i) = ((i * 37 + 11) mod 1009) / 1009 + 0.01
binary(i)  = ((i * 53 + 19) mod 1013) / 1013 + 0.02
sorting(i) = ((i * 48271 + 17) mod 65521) mod 4096
searchsorted_source(i) = 2 * i
searchsorted_values(i) = (i * 48271 + 17) mod (2 * size + 1)
product(i) = 1 + ((((i * 37 + 11) mod 1009) - 504) * 1e-9)
```

The duplicate-rich sorting sequence exercises tie handling, nondecreasing key order, and index reconstruction. Equal keys may use different valid index orders in NumPy and cnumpy; each worker still validates a complete permutation. Partition workloads use `kth = size // 2`; each worker validates the value/permutation and kth partition invariants before sorting the untimed validation result into a deterministic cross-runtime signature. For `sort_stable_nan` and `unique_nan`, the strict `numeric_nan` signature first validates the raw NaN count and trailing placement plus every deterministic finite member, records `finite_members_exact=true`, and only then normalizes NaNs to `8192` for finite samples and sums. `prod` processes the complete requested size using near-one values; it is never truncated. Matrix inputs are `general(i) * 0.001` in C order with `2.0` added to the diagonal. Random creation uses the recorded seed; validation intentionally checks its shape because NumPy and cnumpy use different RNG algorithms.

The `choice_weighted/f64/1000000` case draws one million replacement samples
from the fixed float64 population `0..256` with a normalized one-hot probability
at population index 193. Python times the public
`numpy.random.RandomState(seed).choice(...)` call. AutoHotkey times the public
`DllCall` bridge to `cnp_ahk_random_choice_v2`, native sampling, and result
release. Population/probability construction, seed reset, and deterministic
validation are outside timing. Both workers validate every returned member, not
only signature samples; the native row additionally requires
`retained_bytes=0`. NumPy uses MT19937 through `RandomState`, while cnumpy uses
its documented SplitMix64-seeded xoshiro256** generator, so general random
sequences are not required to be byte-identical across runtimes.

Searchsorted workloads use the monotonic even-number source and a full-size deterministic query vector shown above. Separate cases exercise `side="left"` and `side="right"`, including queries equal to, between, and above source values; both workers validate the complete int64 result before timing.

Digitize workloads reuse `searchsorted_values` as `x` and `searchsorted_source` as bins. The increasing case uses bins as written; the decreasing case reverses them logically. Both time `right=false`, validate the full int64 result, and include bins monotonicity validation in every public invocation.

Lexsort workloads use the duplicate-rich sorting sequence as the secondary key and `binary` as the primary key. Both workers time the stable default last-axis ordering, validate the complete int64 permutation, and release every result handle inside the measured invocation.

Msort workloads use the duplicate-rich sorting sequence and time the public first-axis operation (which is equivalent to axis 0 for the 1-D performance vectors). Sort-complex workloads use the same real float64 sequence, time creation of the required complex128 result, verify its dtype and zero imaginary components outside the timed region, and compare the sorted real component through the standard validation signature.

Semantic parity includes:

- `average`: `general` is the data vector and `binary` is the same-length,
  strictly positive weight vector; both workers time `axis=None`, and the AHK
  worker reads and releases the native scalar array inside every invocation.
- `divmod`: the timed operation creates both outputs; validation uses the logical
  stacked shape `(2, size)` and releases both native handles on every call.
- `reshape`: view semantics with output shape `(size, 1)`.
- `flatten`: copy semantics.
- `transpose_copy`: C-order copy of the transpose.
- `concatenate`: two equal matrices joined on axis 0.
- `fft`: logical float64 real/imaginary pairs with shape `(n, 2)` for cross-runtime validation.
- `iscomplexobj` / `isrealobj` / `isscalar`: scalar bool object-kind queries
  over a retained float64 array; the timed operation reads metadata without
  traversing elements. `isscalar` is false because the input is an ndarray.
- `nbytes_cached` / `c_contiguous_cached` / `f_contiguous_cached`: scalar
  queries over a retained float64 array after one metadata snapshot has been
  loaded; timed calls do not traverse elements or cross the DLL boundary.

Task 8 measures four formal cases per family: vectors of 10,000 and 1,000,000
elements, a contiguous 512×512 last-axis matrix, and an axis-0 transposed native
view. Softmax, log-softmax, and trapz use float64; packbits and unpackbits use
canonical uint8 with big bit order. Trapz fixes `dx=0.25` and does not supply
`x`; unpackbits fixes `count=None`. Both workers validate every element of every
raw result buffer outside the timed region, including unsampled interior values.
The AHK worker verifies that the axis-0 source is a real transposed native view
and asserts allocator equality after setup, validation, and cleanup; every
published Task 8 cnumpy row must record `retained_bytes=0`.

NumPy 1.25.0 leaves the extra bytes uninitialized when `unpackbits` receives an
empty input and a positive `count`. That combination is therefore deliberately
excluded from differential qualification instead of treating nondeterministic
memory as an oracle. The native contract remains deterministic and explicitly
zero-extends those positive-count empty results.

## Output artifacts

Every successful run creates `benchmark/results/<UTC timestamp>/` unless `--output-root` is supplied:

| Artifact | Contents |
|---|---|
| `environment.json` | Run ID, host/CPU, selection, protocol, order, absolute paths, runtime/timer metadata, exact commands, DLL SHA-256/size/timestamp, and Release compiler settings. |
| `numpy.json` | Canonical case metadata, validation signature, calibrated loops, raw samples, and shared summary statistics. |
| `cnumpy.json` | Same fields plus `retained_bytes`, which must be exactly zero. |
| `comparison.csv` | Stable machine-readable per-case medians, ratios, winners, CV, and p95. |
| `comparison.md` | Environment context, category tables, CV warnings, bottleneck ranking, and geometric means. |
| `jobs.tsv` | Exact canonical jobs consumed by both workers. |
| `*.stdout.log`, `*.stderr.log` | Complete streamed child-process output. |

JSON publication and text reports use same-directory temporary files followed by atomic replacement. Worker JSON rejects duplicate keys, non-finite samples, malformed metadata, noncanonical case fields, invalid validation signatures, and operation/output-contract drift.

## Baseline comparison

Capture an unoptimized native-size focus run:

```powershell
python benchmark\benchmark.py --profile focus --warmups 5 --samples 15 --target-sample-ms 20
```

Then compare a fresh run without changing its current measurements:

```powershell
python benchmark\benchmark.py --profile focus `
  --baseline benchmark\baselines\pre-optimization.csv
```

The baseline must have exactly the current case-ID set. Omitting `--baseline`
disables baseline attachment completely and leaves all baseline columns out of the
report.

Every comparison CSV persists a row-level `semantic_qualification`.
`environment.json` schema version 2 records the authoritative plural
`semantic_qualifications` registry. The Task6 reduction scope remains
`numpy-1.25/task6-reduction-v1`; Task7's exact sort/argsort and set-operation
rows use `numpy-1.25/task7-sort-set-v1`. Task 8 uses
`numpy-1.25/task8-misc-axis-v1`, pinned to `NumPy 1.25.0; scipy.special 1.12.0`,
for exactly these `misc_axis` operations: `softmax`, `softmax_axis_last`,
`softmax_axis0_strided`, `log_softmax`, `log_softmax_axis_last`,
`log_softmax_axis0_strided`, `trapz`, `trapz_axis_last`,
`trapz_axis0_strided`, `packbits`, `packbits_axis_last`,
`packbits_axis0_strided`, `unpackbits`, `unpackbits_axis_last`, and
`unpackbits_axis0_strided`.

Its exact direct semantic owners are:

```text
benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests.test_softmax_and_log_softmax_match_stable_axis_reference
benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests.test_softmax_rank_zero_through_four_all_axes_and_real_dtypes
benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests.test_softmax_nan_and_infinity_behavior_matches_reference
benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests.test_softmax_huge_empty_reduction_axis_errors_before_data_access
benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests.test_trapz_matches_numpy_for_axes_dx_and_noncontiguous_inputs
benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests.test_trapz_matches_numpy_dtype_promotion_and_typed_arithmetic
benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests.test_trapz_matches_numpy_x_dtype_promotion_and_product_overflow
benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests.test_trapz_preserves_wide_integer_panels_until_float_conversion
benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests.test_trapz_rank_zero_through_four_all_axes
benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests.test_packbits_matches_numpy_for_axes_bitorders_and_views
benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests.test_unpackbits_matches_numpy_counts_padding_axes_and_bitorders
benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests.test_bit_packing_rank_zero_through_four_all_axes_and_dtypes
benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests.test_bit_packing_empty_dimensions_match_numpy
benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests.test_bit_packing_huge_zero_shapes_are_bounded_and_exact
benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests.test_unpackbits_rejects_huge_multidimensional_result_before_allocation
benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests.test_legacy_bit_packing_rank_zero_through_four_and_errors
benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests.test_task8_core_exports_report_exact_invalid_axes_all_ranks
benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests.test_task8_core_exports_report_exact_dtype_and_null_errors
benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests.test_task8_results_survive_source_release_and_retain_zero_bytes
benchmark.tests.test_orchestrator.ComparisonTests.test_task8_qualification_requires_exact_scipy_reference_version
ahk.numpy.test.NumpyFoundationTest.TestMiscAxisFacadeV2
```

Task 9 uses `numpy-1.25/task9-linalg-v1`, pinned to `NumPy 1.25.0`,
for exactly these `linalg` operations: `einsum`, `eig`, `svd`, `solve`, and
`lstsq`. The standard profile has 455 cases and the full profile has 459.
Task 9 contributes `einsum` at 32, 128, and 512 square matrices, plus `eig`,
`svd`, and `lstsq` at 32 and 128; `solve` retains its existing 32/128 standard
rows and 512 full row. `cond` is deliberately outside this exact qualification:
exactly rank-deficient LAPACK condition values are characterized in
`compat/manifest.json` because equivalent inputs can yield either infinity or
finite values near reciprocal machine precision.

Both workers validate complete deterministic results outside timing. The native
worker checks every einsum and solve output member; the complete eigenvalue and
eigenvector decomposition; all U, singular-value, and Vh buffers plus SVD
reconstruction and orthogonality; and all four lstsq results, including the
empty residual shape, rank, and singular values. It also checks source
immutability, frees every 1/2/3/4-result invocation, and requires
`retained_bytes=0`. Smoke tests corrupt unsampled members in every Task 9 result
family and require those counterfeit outputs to be rejected.

Task 9 has 45 registered direct owners:

```text
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_einsum_explicit_and_implicit_outputs_match_numpy
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_einsum_repeated_labels_diagonals_and_reductions
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_einsum_ellipsis_scalar_and_broadcasting_match_numpy
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_einsum_preserves_views_and_numpy_dtype_promotion
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_einsum_float16_forms_promotion_rounding_and_lifetime
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_einsum_fast_patterns_respect_named_label_broadcasting
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_einsum_invalid_subscripts_shapes_and_nulls_are_explicit
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_general_eig_returns_complex_eigenpairs_for_real_matrix
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_general_eig_preserves_real_dtype_for_real_spectrum
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_general_eig_supports_batched_matrices_and_owned_results
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_general_eig_dtype_promotion_matches_numpy
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_general_eig_dense_nonsymmetric_noncontiguous_view
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_general_eig_seeded_dense_differential
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_general_eig_repeated_and_defective_spectra
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_general_eig_validation_is_explicit_and_clears_outputs
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_ahk_eig_bridge_clears_every_provided_result_slot
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_eigvals_wrapper_inherits_general_semantics_and_owns_errors
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_general_eig_zero_sized_matrix_and_batch_shapes
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_general_eig_tiny_complex_pairs_are_scale_relative
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_general_eig_tiny_nonnormal_eigenvectors_are_scale_relative
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_general_eig_exact_zero_degenerate_eigenspaces
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_svd_workspace_products_are_checked_before_allocation
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_svd_legacy_default_returns_complete_rectangular_factors
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_svd_v2_reduced_tall_wide_and_batched_shapes
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_svd_v2_dtype_promotion_complex_and_unitarity
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_svd_v2_reads_noncontiguous_complex_view
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_svd_v2_compute_uv_false_and_zero_sized_shapes
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_svd_v2_complete_wide_and_hermitian_factors
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_svd_v2_seeded_dense_and_rank_deficient_differential
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_svd_v2_validation_is_explicit_and_atomic
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_solve_square_batched_rhs_dtypes_and_lifetimes_match_numpy_125
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_solve_zero_batch_broadcasts_without_reading_empty_sources
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_solve_singular_failure_is_explicit_atomic_and_nonretaining
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_lstsq_v2_rectangular_outputs_rcond_and_lifetimes_match_numpy_125
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_lstsq_v2_numpy_125_rcond_boundary_values
benchmark.tests.test_linalg_semantics.LinalgSemanticsTests.test_lstsq_v2_and_cond_v2_validation_is_explicit_atomic_and_retained0
benchmark.tests.test_numpy_worker.PreparedOperationTests.test_task9_linalg_workers_validate_complete_deterministic_results
benchmark.tests.test_numpy_worker.PreparedOperationTests.test_task9_linalg_full_validators_reject_unsampled_counterfeits
benchmark.tests.test_orchestrator.ComparisonTests.test_every_exact_task9_operation_receives_only_task9_qualification
benchmark.tests.test_orchestrator.ComparisonTests.test_task9_qualification_requires_exact_numpy_reference_version
ahk.numpy.test.NumpyFoundationTest.TestEinsumFacadeV2
ahk.numpy.test.NumpyFoundationTest.TestGeneralEigFacadeV2
ahk.numpy.test.NumpyFoundationTest.TestLinalgSpectralDelegatesV2
ahk.numpy.test.NumpyFoundationTest.TestSvdFacadeV2
ahk.numpy.test.NumpyFoundationTest.TestTask9SolveLstsqAndCondFacadeV2
```

Task 10 uses `numpy-1.25/task10-random-choice-v1`, pinned to
`NumPy 1.25.0`, for exactly the `random/choice_weighted` benchmark operation.
Its scope does not qualify the other random-distribution symbols. The exact
direct owners are:

```text
benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests.test_seeded_unweighted_choice_preserves_shape_dtype_and_sequence
benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests.test_numpy_125_differential_shape_dtype_replace_and_probability_dtypes
benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests.test_numpy_125_differential_noncontiguous_and_negative_stride_inputs
benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests.test_numpy_125_differential_rejections_are_atomic
benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests.test_choice_v2_raw_size_contract_rejections_are_atomic
benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests.test_probability_sum_tolerance_matches_numpy_125_by_dtype
benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests.test_weighted_replacement_matches_requested_distribution
benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests.test_weighted_without_replacement_non_degenerate_distribution_matches_numpy_125
benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests.test_scalar_and_empty_shapes_follow_numpy_size_semantics
benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests.test_validation_failures_are_explicit_and_allocation_atomic
benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests.test_legacy_choice_uses_weighted_semantics
benchmark.tests.test_numpy_worker.PreparedOperationTests.test_weighted_choice_uses_full_deterministic_validation
benchmark.tests.test_orchestrator.ComparisonTests.test_task10_random_choice_has_exact_scope_and_version_gate
benchmark.tests.test_catalog.SharedCatalogTests.test_weighted_choice_is_a_canonical_one_million_sample_case
ahk.numpy.test.NumpyFoundationTest.TestWeightedRandomChoiceFacadeV2
ahk.numpy.test.NumpyFoundationTest.TestRandomChoiceSeedScalarAndErrorFacadeV2
ahk.numpy.test.NumpyFoundationTest.TestRandomChoiceProbabilityErrorsAndTemporaryLifetimeV2
```

The exact NumPy version gate is evaluated before comparison rows or reports are
published. A missing, non-string, empty, or non-`1.25.0` `numpy_version`
therefore fails the run explicitly.

Each declaration records its exact catalog category/operations, reference,
direct test-method owners, and compatibility manifest. Neighboring sorting rows
such as partition, search, lexsort, msort, and sort_complex, and neighboring
linalg rows such as matmul, dot, det, inv, norm, and cholesky remain explicitly
`N/A` for these task-specific qualifications. An identifier must be manually
bumped whenever its observable semantic contracts change.
The qualification is pinned to the exact NumPy worker version `1.25.0`. A
different, missing, empty, or non-string `numpy_version` is an explicit error:
the benchmark does not publish comparison rows or an environment qualification
for a worker whose actual reference version is not exactly the pinned version.

Baseline ratios and `baseline_ratio / current_ratio` are calculated only when
the current and baseline rows carry the same nonempty, non-`N/A` qualification.
Missing, empty, `N/A`, or different identifiers produce literal `N/A` numeric
delta fields and a specific compatibility reason in CSV and Markdown. DLL SHA-256
is retained as artifact identity in `environment.json`, but it is not the
semantic compatibility gate: different optimized DLL builds remain comparable
when they share the same semantic qualification. Old CSV files without the
qualification remain loadable only so the new report can state why their deltas
are `N/A`; the old files are never rewritten. A native baseline still cannot be
attached to a `smoke` run because their canonical case IDs differ.
In particular, any old or otherwise unqualified Task 8, Task 9, or Task 10 baseline is
explicitly `N/A`; it is never reported as a semantic performance comparison.

## Failure behavior

A run is invalid and exits nonzero on the first build, child-process, path, catalog, TSV, JSON, schema, case-set, metadata, validation, non-finite timing, or retained-memory error. Failed run directories and logs are intentionally retained for diagnosis, but no comparison report is published as successful.

There are no skipped operations, mock results, silent path substitutions, bridge-overhead subtraction, or partial-success reports.

## Verification

```powershell
$Python = 'F:\Python\Python310\python.exe' # Python 3.10.11 with NumPy 1.25.0
& $Python -B -m unittest discover -s benchmark\tests -v
D:\Tech\Projects\Autohotkey\AutoHotkey64.exe /ErrorStdOut=UTF-8 benchmark\benchmark_smoke.test.ahk
D:\Tech\Projects\Autohotkey\AutoHotkey64.exe /ErrorStdOut=UTF-8 ahk\numpy.test.ahk
& $Python benchmark\benchmark.py --profile focus --size-scale smoke --warmups 1 --samples 3 --target-sample-ms 1
```
