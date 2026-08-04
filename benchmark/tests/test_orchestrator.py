from __future__ import annotations

import csv
import importlib
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from benchmark.benchmark import (
    atomic_write_text,
    collect_environment,
    create_run_directory,
    parse_runtime_order,
    python_worker_prefix,
    read_release_compiler_settings,
    resolve_explicit_or_candidates,
    resolve_msbuild_executable,
    run_logged_command,
    scale_cases_for_smoke,
    select_cases,
    sha256_file,
)
from benchmark.report import (
    attach_baseline,
    compare_results,
    load_worker_json,
    load_baseline_csv,
    normalize_worker_result,
    render_comparison_csv,
    render_markdown,
    semantic_qualification_declarations,
)


TASK6_SEMANTIC_QUALIFICATION_ID = "numpy-1.25/task6-reduction-v1"
TASK7_SEMANTIC_QUALIFICATION_ID = "numpy-1.25/task7-sort-set-v1"
TASK8_SEMANTIC_QUALIFICATION_ID = "numpy-1.25/task8-misc-axis-v1"
TASK9_SEMANTIC_QUALIFICATION_ID = "numpy-1.25/task9-linalg-v1"
TASK10_SEMANTIC_QUALIFICATION_ID = "numpy-1.25/task10-random-choice-v1"
TASK10_SEMANTIC_QUALIFICATION_OWNERS = (
    "benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests."
    "test_seeded_unweighted_choice_preserves_shape_dtype_and_sequence",
    "benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests."
    "test_numpy_125_differential_shape_dtype_replace_and_probability_dtypes",
    "benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests."
    "test_numpy_125_differential_noncontiguous_and_negative_stride_inputs",
    "benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests."
    "test_numpy_125_differential_rejections_are_atomic",
    "benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests."
    "test_choice_v2_raw_size_contract_rejections_are_atomic",
    "benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests."
    "test_probability_sum_tolerance_matches_numpy_125_by_dtype",
    "benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests."
    "test_weighted_replacement_matches_requested_distribution",
    "benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests."
    "test_weighted_without_replacement_non_degenerate_distribution_matches_numpy_125",
    "benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests."
    "test_scalar_and_empty_shapes_follow_numpy_size_semantics",
    "benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests."
    "test_validation_failures_are_explicit_and_allocation_atomic",
    "benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests."
    "test_legacy_choice_uses_weighted_semantics",
    "benchmark.tests.test_numpy_worker.PreparedOperationTests."
    "test_weighted_choice_uses_full_deterministic_validation",
    "benchmark.tests.test_orchestrator.ComparisonTests."
    "test_task10_random_choice_has_exact_scope_and_version_gate",
    "benchmark.tests.test_catalog.SharedCatalogTests."
    "test_weighted_choice_is_a_canonical_one_million_sample_case",
    "ahk.numpy.test.NumpyFoundationTest."
    "TestWeightedRandomChoiceFacadeV2",
    "ahk.numpy.test.NumpyFoundationTest."
    "TestRandomChoiceSeedScalarAndErrorFacadeV2",
    "ahk.numpy.test.NumpyFoundationTest."
    "TestRandomChoiceProbabilityErrorsAndTemporaryLifetimeV2",
)
TASK9_SEMANTIC_QUALIFICATION_OPERATIONS = (
    "einsum",
    "eig",
    "svd",
    "solve",
    "lstsq",
)
TASK9_LINALG_OWNER_PREFIX = (
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
)
TASK9_SEMANTIC_QUALIFICATION_OWNERS = tuple(
    TASK9_LINALG_OWNER_PREFIX + method
    for method in (
        "test_einsum_explicit_and_implicit_outputs_match_numpy",
        "test_einsum_repeated_labels_diagonals_and_reductions",
        "test_einsum_ellipsis_scalar_and_broadcasting_match_numpy",
        "test_einsum_preserves_views_and_numpy_dtype_promotion",
        "test_einsum_float16_forms_promotion_rounding_and_lifetime",
        "test_einsum_fast_patterns_respect_named_label_broadcasting",
        "test_einsum_invalid_subscripts_shapes_and_nulls_are_explicit",
        "test_general_eig_returns_complex_eigenpairs_for_real_matrix",
        "test_general_eig_preserves_real_dtype_for_real_spectrum",
        "test_general_eig_supports_batched_matrices_and_owned_results",
        "test_general_eig_dtype_promotion_matches_numpy",
        "test_general_eig_dense_nonsymmetric_noncontiguous_view",
        "test_general_eig_seeded_dense_differential",
        "test_general_eig_repeated_and_defective_spectra",
        "test_general_eig_validation_is_explicit_and_clears_outputs",
        "test_ahk_eig_bridge_clears_every_provided_result_slot",
        "test_eigvals_wrapper_inherits_general_semantics_and_owns_errors",
        "test_general_eig_zero_sized_matrix_and_batch_shapes",
        "test_general_eig_tiny_complex_pairs_are_scale_relative",
        "test_general_eig_tiny_nonnormal_eigenvectors_are_scale_relative",
        "test_general_eig_exact_zero_degenerate_eigenspaces",
        "test_svd_workspace_products_are_checked_before_allocation",
        "test_svd_legacy_default_returns_complete_rectangular_factors",
        "test_svd_v2_reduced_tall_wide_and_batched_shapes",
        "test_svd_v2_dtype_promotion_complex_and_unitarity",
        "test_svd_v2_reads_noncontiguous_complex_view",
        "test_svd_v2_compute_uv_false_and_zero_sized_shapes",
        "test_svd_v2_complete_wide_and_hermitian_factors",
        "test_svd_v2_seeded_dense_and_rank_deficient_differential",
        "test_svd_v2_validation_is_explicit_and_atomic",
        "test_solve_square_batched_rhs_dtypes_and_lifetimes_match_numpy_125",
        "test_solve_zero_batch_broadcasts_without_reading_empty_sources",
        "test_solve_singular_failure_is_explicit_atomic_and_nonretaining",
        "test_lstsq_v2_rectangular_outputs_rcond_and_lifetimes_match_numpy_125",
        "test_lstsq_v2_numpy_125_rcond_boundary_values",
        "test_lstsq_v2_and_cond_v2_validation_is_explicit_atomic_and_retained0",
    )
) + (
    "benchmark.tests.test_numpy_worker.PreparedOperationTests."
    "test_task9_linalg_workers_validate_complete_deterministic_results",
    "benchmark.tests.test_numpy_worker.PreparedOperationTests."
    "test_task9_linalg_full_validators_reject_unsampled_counterfeits",
    "benchmark.tests.test_orchestrator.ComparisonTests."
    "test_every_exact_task9_operation_receives_only_task9_qualification",
    "benchmark.tests.test_orchestrator.ComparisonTests."
    "test_task9_qualification_requires_exact_numpy_reference_version",
    "ahk.numpy.test.NumpyFoundationTest.TestEinsumFacadeV2",
    "ahk.numpy.test.NumpyFoundationTest.TestGeneralEigFacadeV2",
    "ahk.numpy.test.NumpyFoundationTest.TestLinalgSpectralDelegatesV2",
    "ahk.numpy.test.NumpyFoundationTest.TestSvdFacadeV2",
    "ahk.numpy.test.NumpyFoundationTest."
    "TestTask9SolveLstsqAndCondFacadeV2",
)
TASK8_SEMANTIC_QUALIFICATION_OPERATIONS = (
    "softmax",
    "softmax_axis_last",
    "softmax_axis0_strided",
    "log_softmax",
    "log_softmax_axis_last",
    "log_softmax_axis0_strided",
    "trapz",
    "trapz_axis_last",
    "trapz_axis0_strided",
    "packbits",
    "packbits_axis_last",
    "packbits_axis0_strided",
    "unpackbits",
    "unpackbits_axis_last",
    "unpackbits_axis0_strided",
)
TASK8_SEMANTIC_QUALIFICATION_OWNERS = (
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_softmax_and_log_softmax_match_stable_axis_reference",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_softmax_rank_zero_through_four_all_axes_and_real_dtypes",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_softmax_nan_and_infinity_behavior_matches_reference",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_softmax_huge_empty_reduction_axis_errors_before_data_access",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_trapz_matches_numpy_for_axes_dx_and_noncontiguous_inputs",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_trapz_matches_numpy_dtype_promotion_and_typed_arithmetic",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_trapz_matches_numpy_x_dtype_promotion_and_product_overflow",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_trapz_preserves_wide_integer_panels_until_float_conversion",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_trapz_rank_zero_through_four_all_axes",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_packbits_matches_numpy_for_axes_bitorders_and_views",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_unpackbits_matches_numpy_counts_padding_axes_and_bitorders",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_bit_packing_rank_zero_through_four_all_axes_and_dtypes",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_bit_packing_empty_dimensions_match_numpy",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_bit_packing_huge_zero_shapes_are_bounded_and_exact",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_unpackbits_rejects_huge_multidimensional_result_before_allocation",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_legacy_bit_packing_rank_zero_through_four_and_errors",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_task8_core_exports_report_exact_invalid_axes_all_ranks",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_task8_core_exports_report_exact_dtype_and_null_errors",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_task8_results_survive_source_release_and_retain_zero_bytes",
    "benchmark.tests.test_orchestrator.ComparisonTests."
    "test_task8_qualification_requires_exact_scipy_reference_version",
    "ahk.numpy.test.NumpyFoundationTest.TestMiscAxisFacadeV2",
)
TASK7_SEMANTIC_QUALIFICATION_SORTING_OPERATIONS = (
    "argsort",
    "argsort_heapsort",
    "argsort_mergesort",
    "argsort_stable",
    "argsort_stable_nan",
    "sort",
    "sort_heapsort",
    "sort_mergesort",
    "sort_stable",
    "sort_stable_nan",
)
TASK7_SEMANTIC_QUALIFICATION_SET_OPERATIONS = (
    "in1d_duplicates",
    "intersect1d_duplicates",
    "isin_duplicates",
    "setdiff1d_duplicates",
    "setxor1d_duplicates",
    "union1d_duplicates",
    "unique_duplicates",
    "unique_nan",
)
TASK6_SEMANTIC_QUALIFICATION_OWNERS = (
    "benchmark.tests.test_reduction_semantics.ReductionSemanticsTests."
    "test_every_legacy_reduction_export_covers_rank_zero_through_four_and_expressible_axes",
    "benchmark.tests.test_reduction_semantics.ReductionSemanticsTests."
    "test_every_legacy_reduction_export_reports_exact_axis_errors",
    "benchmark.tests.test_reduction_semantics.ReductionSemanticsTests."
    "test_legacy_reduction_axis_minus_one_characterizes_none_sentinel",
    "benchmark.tests.test_reduction_semantics.ReductionSemanticsTests."
    "test_every_reduction_family_covers_rank_zero_through_four_and_all_axes",
    "benchmark.tests.test_reduction_semantics.ReductionSemanticsTests."
    "test_average_distinguishes_none_from_the_negative_last_axis",
    "benchmark.tests.test_reduction_semantics.ReductionSemanticsTests."
    "test_average_v2_covers_rank_zero_through_four_axes_weights_errors_and_lifetimes",
    "benchmark.tests.test_reduction_semantics.ReductionSemanticsTests."
    "test_legacy_nan_scalar_exports_cover_rank_zero_through_four_axes_values_errors_and_lifetimes",
    "benchmark.tests.test_reduction_semantics.ReductionSemanticsTests."
    "test_scalar_convenience_exports_cover_rank_zero_through_four_layouts_values_errors_and_lifetimes",
)


def worker_document(
    runtime: str,
    case_ids: tuple[str, ...] = ("sum/f64/8",),
    *,
    median: float = 100.0,
    retained_bytes: int = 0,
) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for case_id in case_ids:
        operation = case_id.split("/")[0]
        if case_id == "bridge/property_call":
            category = "bridge"
            size, rows, cols = 1, 0, 0
            validation = {
                "mode": "numeric",
                "shape": [],
                "size": 1,
                "logical_dtype": "i64",
                "sample_indices": [0],
                "values": [1.0],
                "sum": 1.0,
            }
        else:
            size, rows, cols = int(case_id.split("/")[2]), 0, 0
            if operation.startswith("argsort"):
                category = "sorting"
                validation = {
                    "mode": "numeric",
                    "shape": [size],
                    "size": size,
                    "logical_dtype": "i64",
                    "sample_indices": [0, size // 2, size - 1],
                    "values": [678003.0, 76974.0, 673370.0],
                    "sum": size * (size - 1) / 2,
                }
            else:
                category = "reduction"
                logical_dtype = "i64" if operation == "argmax" else "f64"
                scalar_value = 4.0 if logical_dtype == "i64" else 4.25
                validation = {
                    "mode": "numeric",
                    "shape": [],
                    "size": 1,
                    "logical_dtype": logical_dtype,
                    "sample_indices": [0],
                    "values": [scalar_value],
                    "sum": scalar_value,
                }
        case: dict[str, object] = {
            "id": case_id,
            "category": category,
            "operation": operation,
            "dtype": "f64",
            "size": size,
            "rows": rows,
            "cols": cols,
            "axis": -1,
            "inner_loops": 4,
            "samples_ns": [median * 0.9, median, median * 1.1],
            "validation": validation,
        }
        if runtime == "cnumpy":
            case["retained_bytes"] = retained_bytes
        cases.append(case)
    if runtime == "numpy":
        metadata = {
            "python_version": "3.10.11",
            "numpy_version": "1.25.0",
            "scipy_version": "1.12.0",
            "timer": "perf_counter_ns",
            "timer_resolution_ns": 100.0,
            "warmups": 1,
            "sample_count": 3,
            "target_sample_ns": 1_000_000.0,
            "seed": 12345,
        }
    else:
        metadata = {
            "ahk_version": "2.1-alpha.30",
            "dll_version": "0.1.0",
            "dll_path": r"C:\cnumpy\cnumpy_ahk.dll",
            "timer": "QueryPerformanceCounter",
            "timer_frequency": 10_000_000,
            "warmups": 1,
            "sample_count": 3,
            "target_sample_ns": 1_000_000.0,
            "seed": 12345,
            "simd_level": 2,
            "simd_name": "avx2",
        }
    return {
        "schema_version": 1,
        "runtime": runtime,
        "metadata": metadata,
        "cases": cases,
    }


def task8_worker_document(runtime: str, operation: str) -> dict[str, object]:
    is_matrix = "axis_" in operation or "axis0_" in operation
    axis = 0 if "axis0_" in operation else 1 if is_matrix else -1
    rows = cols = 8 if is_matrix else 0
    input_size = 64 if is_matrix else 8
    dtype = "u8" if "packbits" in operation else "f64"
    case_id = (
        f"{operation}/{dtype}/8x8/axis{axis}"
        if is_matrix
        else f"{operation}/{dtype}/8"
    )
    document = worker_document(runtime)
    case = document["cases"][0]  # type: ignore[index]
    case.update(
        {
            "id": case_id,
            "category": "misc_axis",
            "operation": operation,
            "dtype": dtype,
            "size": input_size,
            "rows": rows,
            "cols": cols,
            "axis": axis,
        }
    )
    if operation in {"trapz", "trapz_axis_last", "trapz_axis0_strided"}:
        shape = [] if operation == "trapz" else [8]
    elif operation == "packbits":
        shape = [1]
    elif operation == "packbits_axis_last":
        shape = [8, 1]
    elif operation == "packbits_axis0_strided":
        shape = [1, 8]
    elif operation == "unpackbits":
        shape = [64]
    elif operation == "unpackbits_axis_last":
        shape = [8, 64]
    elif operation == "unpackbits_axis0_strided":
        shape = [64, 8]
    else:
        shape = [8, 8] if is_matrix else [8]
    output_size = math.prod(shape) if shape else 1
    sample_indices = list(
        dict.fromkeys((0, output_size // 2, output_size - 1))
    )
    case["validation"] = {
        "mode": "numeric",
        "shape": shape,
        "size": output_size,
        "logical_dtype": "u8" if dtype == "u8" else "f64",
        "sample_indices": sample_indices,
        "values": [0.0] * len(sample_indices),
        "sum": 0.0,
    }
    return document


def task9_worker_document(runtime: str, operation: str) -> dict[str, object]:
    document = worker_document(runtime)
    case = document["cases"][0]  # type: ignore[index]
    output_shape = [4, 4] if operation == "einsum" else [4]
    output_size = math.prod(output_shape)
    sample_indices = list(
        dict.fromkeys((0, output_size // 2, output_size - 1))
    )
    case.update(
        {
            "id": f"{operation}/f64/4x4",
            "category": "linalg",
            "operation": operation,
            "size": 16,
            "rows": 4,
            "cols": 4,
            "validation": {
                "mode": "numeric",
                "shape": output_shape,
                "size": output_size,
                "logical_dtype": "f64",
                "sample_indices": sample_indices,
                "values": [0.0] * len(sample_indices),
                "sum": 0.0,
            },
        }
    )
    return document


def task10_worker_document(runtime: str) -> dict[str, object]:
    document = worker_document(runtime)
    case = document["cases"][0]
    case.update(
        {
            "id": "choice_weighted/f64/8",
            "category": "random",
            "operation": "choice_weighted",
            "size": 8,
            "validation": {
                "mode": "numeric",
                "shape": [8],
                "size": 8,
                "logical_dtype": "f64",
                "sample_indices": [0, 4, 7],
                "values": [193.0, 193.0, 193.0],
                "sum": 1544.0,
            },
        }
    )
    return document


def collect_test_environment(
    numpy_metadata: dict[str, object],
    *,
    case_ids: tuple[str, ...] = ("sum/f64/8",),
    categories: tuple[str, ...] = (),
) -> dict[str, object]:
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        python = root / "python.exe"
        ahk = root / "AutoHotkey64.exe"
        dll = root / "cnumpy_ahk.dll"
        for path, contents in (
            (python, b"python"),
            (ahk, b"ahk"),
            (dll, b"dll"),
        ):
            path.write_bytes(contents)

        return collect_environment(
            run_id="20260720T120000.000000Z",
            started_at_utc="2026-07-20T12:00:00+00:00",
            profile="focus",
            size_scale="native",
            case_ids=case_ids,
            categories=categories,
            warmups=5,
            samples=15,
            target_sample_ms=20.0,
            seed=12345,
            runtime_order=("numpy", "cnumpy"),
            build_requested=False,
            project_root=project_root,
            project_file=project_root / "src" / "cnumpy_ahk.vcxproj",
            python_executable=python,
            ahk_executable=ahk,
            msbuild_executable=None,
            dll_path=dll,
            numpy_metadata=numpy_metadata,
            cnumpy_metadata={
                "ahk_version": "2.1-alpha.30",
                "dll_version": "0.1.0",
                "timer": "QueryPerformanceCounter",
                "timer_frequency": 10_000_000,
                "simd_level": 2,
                "simd_name": "avx2",
            },
            commands={"numpy": ["python", "worker"], "cnumpy": ["ahk", "worker"]},
        )


class ComparisonTests(unittest.TestCase):
    def test_task10_random_choice_has_exact_scope_and_version_gate(self) -> None:
        declarations = {
            declaration["id"]: declaration
            for declaration in semantic_qualification_declarations()
        }
        task10 = declarations[TASK10_SEMANTIC_QUALIFICATION_ID]
        self.assertEqual(
            {
                "cases": [
                    {
                        "category": "random",
                        "operations": ["choice_weighted"],
                    }
                ]
            },
            task10["scope"],
        )
        self.assertEqual("NumPy 1.25.0", task10["reference"])
        self.assertEqual(
            list(TASK10_SEMANTIC_QUALIFICATION_OWNERS),
            task10["owners"],
        )
        ahk_tests = (
            Path(__file__).resolve().parents[2] / "ahk" / "numpy.test.ahk"
        ).read_text(encoding="utf-8")
        readme = (
            Path(__file__).resolve().parents[1] / "README.md"
        ).read_text(encoding="utf-8")
        for owner in TASK10_SEMANTIC_QUALIFICATION_OWNERS:
            with self.subTest(owner=owner):
                self.assertIn(owner, readme)
                if owner.startswith("ahk."):
                    self.assertIn(
                        f"static {owner.rsplit('.', 1)[1]}(", ahk_tests
                    )
                else:
                    module_name, class_name, method_name = owner.rsplit(".", 2)
                    owner_class = getattr(
                        importlib.import_module(module_name), class_name
                    )
                    self.assertTrue(callable(getattr(owner_class, method_name)))

        rows = compare_results(
            task10_worker_document("numpy"),
            task10_worker_document("cnumpy"),
        )
        self.assertEqual(
            TASK10_SEMANTIC_QUALIFICATION_ID,
            rows[0]["semantic_qualification"],
        )

        for label, actual in (
            ("mismatch", "1.26.0"),
            ("missing", None),
            ("empty", ""),
            ("non-string", 1250),
        ):
            with self.subTest(version=label):
                wrong_version = task10_worker_document("numpy")
                metadata = wrong_version["metadata"]
                if label == "missing":
                    del metadata["numpy_version"]
                else:
                    metadata["numpy_version"] = actual
                with self.assertRaises(ValueError) as caught:
                    compare_results(
                        wrong_version,
                        task10_worker_document("cnumpy"),
                    )
                message = str(caught.exception)
                self.assertIn("expected='1.25.0'", message)
                self.assertIn(f"actual={actual!r}", message)

    def test_task9_qualification_has_exact_scope_and_resolvable_direct_owners(
        self,
    ) -> None:
        declarations = {
            declaration["id"]: declaration
            for declaration in semantic_qualification_declarations()
        }
        task9 = declarations[TASK9_SEMANTIC_QUALIFICATION_ID]
        self.assertEqual(
            {
                "cases": [
                    {
                        "category": "linalg",
                        "operations": list(
                            TASK9_SEMANTIC_QUALIFICATION_OPERATIONS
                        ),
                    }
                ]
            },
            task9["scope"],
        )
        self.assertEqual("NumPy 1.25.0", task9["reference"])
        owners = task9["owners"]
        self.assertIsInstance(owners, list)
        self.assertEqual(
            list(TASK9_SEMANTIC_QUALIFICATION_OWNERS),
            owners,
        )
        self.assertEqual(45, len(owners))
        self.assertEqual(len(owners), len(set(owners)))
        ahk_owners = {
            owner for owner in owners if owner.startswith("ahk.")
        }
        self.assertEqual(
            {
                "ahk.numpy.test.NumpyFoundationTest.TestEinsumFacadeV2",
                "ahk.numpy.test.NumpyFoundationTest.TestGeneralEigFacadeV2",
                "ahk.numpy.test.NumpyFoundationTest."
                "TestLinalgSpectralDelegatesV2",
                "ahk.numpy.test.NumpyFoundationTest.TestSvdFacadeV2",
                "ahk.numpy.test.NumpyFoundationTest."
                "TestTask9SolveLstsqAndCondFacadeV2",
            },
            ahk_owners,
        )
        for owner in owners:
            if owner.startswith("ahk."):
                method_name = owner.rsplit(".", 1)[1]
                ahk_tests = (
                    Path(__file__).resolve().parents[2]
                    / "ahk"
                    / "numpy.test.ahk"
                ).read_text(encoding="utf-8")
                with self.subTest(owner=owner):
                    self.assertIn(f"static {method_name}(", ahk_tests)
                continue
            self.assertNotEqual(
                "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests",
                owner,
            )
            module_name, class_name, method_name = owner.rsplit(".", 2)
            with self.subTest(owner=owner):
                module = importlib.import_module(module_name)
                owner_class = getattr(module, class_name)
                self.assertTrue(callable(getattr(owner_class, method_name)))
        readme = (
            Path(__file__).resolve().parents[1] / "README.md"
        ).read_text(encoding="utf-8")
        self.assertIn("Task 9 has 45 registered direct owners:", readme)
        for owner in owners:
            with self.subTest(documented_owner=owner):
                self.assertIn(owner, readme)

    def test_every_exact_task9_operation_receives_only_task9_qualification(
        self,
    ) -> None:
        for operation in TASK9_SEMANTIC_QUALIFICATION_OPERATIONS:
            with self.subTest(operation=operation):
                rows = compare_results(
                    task9_worker_document("numpy", operation),
                    task9_worker_document("cnumpy", operation),
                )
                self.assertEqual(
                    TASK9_SEMANTIC_QUALIFICATION_ID,
                    rows[0]["semantic_qualification"],
                )

    def test_neighboring_linalg_operations_remain_outside_task9_qualification(
        self,
    ) -> None:
        for operation in ("matmul", "dot", "det", "inv", "norm", "cholesky"):
            with self.subTest(operation=operation):
                documents = (
                    worker_document("numpy"),
                    worker_document("cnumpy"),
                )
                for document in documents:
                    case = document["cases"][0]  # type: ignore[index]
                    output_shape = (
                        [] if operation in {"det", "norm"} else [4, 4]
                    )
                    output_size = math.prod(output_shape) if output_shape else 1
                    sample_indices = list(
                        dict.fromkeys((0, output_size // 2, output_size - 1))
                    )
                    case.update(
                        {
                            "id": f"{operation}/f64/4x4",
                            "category": "linalg",
                            "operation": operation,
                            "size": 16,
                            "rows": 4,
                            "cols": 4,
                            "validation": {
                                "mode": "numeric",
                                "shape": output_shape,
                                "size": output_size,
                                "logical_dtype": "f64",
                                "sample_indices": sample_indices,
                                "values": [0.0] * len(sample_indices),
                                "sum": 0.0,
                            },
                        }
                    )
                rows = compare_results(*documents)
                self.assertEqual("N/A", rows[0]["semantic_qualification"])

    def test_every_task8_python_owner_resolves_to_a_real_test_method(
        self,
    ) -> None:
        for owner in TASK8_SEMANTIC_QUALIFICATION_OWNERS:
            if not owner.startswith("benchmark."):
                continue
            module_name, class_name, method_name = owner.rsplit(".", 2)
            with self.subTest(owner=owner):
                try:
                    module = importlib.import_module(module_name)
                    owner_class = getattr(module, class_name)
                    owner_method = getattr(owner_class, method_name)
                except (ImportError, AttributeError) as error:
                    self.fail(f"Task 8 owner does not resolve: {owner!r}: {error}")
                self.assertTrue(callable(owner_method))

    def test_reduction_qualification_requires_exact_numpy_reference_version(
        self,
    ) -> None:
        versions = (
            ("2.x", "2.1.0"),
            ("missing", None),
            ("empty", ""),
            ("non-string", 125),
        )
        for label, actual in versions:
            with self.subTest(label=label):
                numpy = worker_document("numpy")
                metadata = numpy["metadata"]
                assert isinstance(metadata, dict)
                if label == "missing":
                    del metadata["numpy_version"]
                else:
                    metadata["numpy_version"] = actual

                with self.assertRaises(ValueError) as caught:
                    compare_results(numpy, worker_document("cnumpy"))

                message = str(caught.exception)
                self.assertIn("expected='1.25.0'", message)
                self.assertIn(f"actual={actual!r}", message)

    def test_reduction_qualification_accepts_exact_numpy_reference_version(
        self,
    ) -> None:
        rows = compare_results(
            worker_document("numpy"),
            worker_document("cnumpy"),
        )

        self.assertEqual(
            rows[0]["semantic_qualification"],
            TASK6_SEMANTIC_QUALIFICATION_ID,
        )

    def test_task8_qualification_requires_exact_scipy_reference_version(
        self,
    ) -> None:
        versions = (
            ("mismatch", "1.11.4"),
            ("missing", None),
            ("empty", ""),
            ("non-string", 1120),
        )
        for label, actual in versions:
            with self.subTest(label=label):
                numpy = task8_worker_document("numpy", "softmax")
                metadata = numpy["metadata"]
                assert isinstance(metadata, dict)
                if label == "missing":
                    del metadata["scipy_version"]
                else:
                    metadata["scipy_version"] = actual

                with self.assertRaises(ValueError) as caught:
                    compare_results(
                        numpy,
                        task8_worker_document("cnumpy", "softmax"),
                    )

                message = str(caught.exception)
                self.assertIn("expected='1.12.0'", message)
                self.assertIn(f"actual={actual!r}", message)

    def test_task9_qualification_requires_exact_numpy_reference_version(
        self,
    ) -> None:
        versions = (
            ("mismatch", "1.26.0"),
            ("missing", None),
            ("empty", ""),
            ("non-string", 1250),
        )
        for label, actual in versions:
            with self.subTest(label=label):
                numpy = task9_worker_document("numpy", "einsum")
                metadata = numpy["metadata"]
                assert isinstance(metadata, dict)
                if label == "missing":
                    del metadata["numpy_version"]
                else:
                    metadata["numpy_version"] = actual
                with self.assertRaises(ValueError) as caught:
                    compare_results(
                        numpy,
                        task9_worker_document("cnumpy", "einsum"),
                    )
                message = str(caught.exception)
                self.assertIn("expected='1.25.0'", message)
                self.assertIn(f"actual={actual!r}", message)

    def test_non_task8_scopes_do_not_require_scipy_reference_version(
        self,
    ) -> None:
        cases = (
            ("sum/f64/8", TASK6_SEMANTIC_QUALIFICATION_ID),
            (
                "argsort_stable/f64/1000000",
                TASK7_SEMANTIC_QUALIFICATION_ID,
            ),
        )
        for case_id, qualification in cases:
            for scipy_version in (None, "0.0.0"):
                with self.subTest(
                    case_id=case_id,
                    scipy_version=scipy_version,
                ):
                    numpy = worker_document("numpy", (case_id,))
                    metadata = numpy["metadata"]
                    assert isinstance(metadata, dict)
                    if scipy_version is None:
                        del metadata["scipy_version"]
                    else:
                        metadata["scipy_version"] = scipy_version
                    rows = compare_results(
                        numpy, worker_document("cnumpy", (case_id,))
                    )
                    self.assertEqual(
                        qualification,
                        rows[0]["semantic_qualification"],
                    )

    def test_sort_set_rows_receive_their_own_semantic_qualification(
        self,
    ) -> None:
        rows = compare_results(
            worker_document("numpy", ("argsort_stable/f64/1000000",)),
            worker_document("cnumpy", ("argsort_stable/f64/1000000",)),
        )

        self.assertEqual(
            rows[0]["semantic_qualification"],
            TASK7_SEMANTIC_QUALIFICATION_ID,
        )
        self.assertNotEqual(
            rows[0]["semantic_qualification"],
            TASK6_SEMANTIC_QUALIFICATION_ID,
        )

    def test_misc_axis_rows_receive_their_own_semantic_qualification(
        self,
    ) -> None:
        documents = (
            worker_document("numpy", ("softmax/f64/10000",)),
            worker_document("cnumpy", ("softmax/f64/10000",)),
        )
        for document in documents:
            case = document["cases"][0]  # type: ignore[index]
            case["category"] = "misc_axis"
            case["validation"] = {
                "mode": "numeric",
                "shape": [10_000],
                "size": 10_000,
                "logical_dtype": "f64",
                "sample_indices": [0, 5_000, 9_999],
                "values": [0.0001, 0.0001, 0.0001],
                "sum": 1.0,
            }

        rows = compare_results(*documents)

        self.assertEqual(
            TASK8_SEMANTIC_QUALIFICATION_ID,
            rows[0]["semantic_qualification"],
        )
        self.assertNotIn(
            rows[0]["semantic_qualification"],
            {
                TASK6_SEMANTIC_QUALIFICATION_ID,
                TASK7_SEMANTIC_QUALIFICATION_ID,
            },
        )

    def test_every_exact_task8_operation_receives_only_task8_qualification(
        self,
    ) -> None:
        for operation in TASK8_SEMANTIC_QUALIFICATION_OPERATIONS:
            with self.subTest(operation=operation):
                rows = compare_results(
                    task8_worker_document("numpy", operation),
                    task8_worker_document("cnumpy", operation),
                )
                self.assertEqual(
                    TASK8_SEMANTIC_QUALIFICATION_ID,
                    rows[0]["semantic_qualification"],
                )

    def test_other_sorting_families_remain_unqualified(self) -> None:
        for operation in (
            "partition",
            "searchsorted",
            "lexsort",
            "msort",
            "sort_complex",
        ):
            with self.subTest(operation=operation):
                numpy = worker_document("numpy", ("argsort_stable/f64/1000000",))
                cnumpy = worker_document("cnumpy", ("argsort_stable/f64/1000000",))
                for document in (numpy, cnumpy):
                    case = document["cases"][0]  # type: ignore[index]
                    case["id"] = f"{operation}/f64/1000000"
                    case["category"] = "sorting"
                    case["operation"] = operation
                    if operation in {"partition", "msort", "sort_complex"}:
                        case["validation"]["logical_dtype"] = "f64"
                rows = compare_results(numpy, cnumpy)
                self.assertEqual("N/A", rows[0]["semantic_qualification"])

    def test_rejects_case_set_drift(self) -> None:
        numpy = worker_document("numpy", ("sum/f64/8",))
        cnumpy = worker_document("cnumpy", ("sum/f64/8", "argmax/f64/8"))

        with self.assertRaisesRegex(ValueError, "case set"):
            compare_results(numpy, cnumpy)

    def test_rejects_case_metadata_drift(self) -> None:
        numpy = worker_document("numpy")
        cnumpy = worker_document("cnumpy")
        cnumpy["cases"][0]["axis"] = 0  # type: ignore[index]

        with self.assertRaisesRegex(ValueError, "metadata"):
            compare_results(numpy, cnumpy)

    def test_rejects_validation_signature_drift(self) -> None:
        numpy = worker_document("numpy")
        cnumpy = worker_document("cnumpy")
        cnumpy["cases"][0]["validation"]["values"] = [5.0]  # type: ignore[index]

        with self.assertRaisesRegex(ValueError, "validation"):
            compare_results(numpy, cnumpy)

    def test_rejects_finite_8192_counterfeits_for_raw_nan_operations(self) -> None:
        cases = (
            (
                "sort_stable_nan",
                "sorting",
                8,
                [8],
                [0.0, 3.0, 8192.0],
                32780.0,
            ),
            (
                "unique_nan",
                "set",
                8,
                [5],
                [1.0, 5.0, 8192.0],
                8208.0,
            ),
        )
        for operation, category, input_size, shape, values, aggregate in cases:
            with self.subTest(operation=operation):
                documents = (
                    worker_document("numpy", ("sum/f64/8",)),
                    worker_document("cnumpy", ("sum/f64/8",)),
                )
                for document in documents:
                    case = document["cases"][0]  # type: ignore[index]
                    case.update(
                        {
                            "id": f"{operation}/f64/{input_size}",
                            "category": category,
                            "operation": operation,
                        }
                    )
                    case["validation"] = {
                        "mode": "numeric",
                        "shape": shape,
                        "size": math.prod(shape),
                        "logical_dtype": "f64",
                        "sample_indices": list(
                            dict.fromkeys((0, math.prod(shape) // 2, math.prod(shape) - 1))
                        ),
                        "values": values,
                        "sum": aggregate,
                    }

                with self.assertRaisesRegex(ValueError, "numeric_nan|contract"):
                    compare_results(*documents)

    def test_rejects_contradictory_raw_nan_facts(self) -> None:
        documents = (
            worker_document("numpy", ("sum/f64/8",)),
            worker_document("cnumpy", ("sum/f64/8",)),
        )
        for document in documents:
            case = document["cases"][0]  # type: ignore[index]
            case.update(
                {
                    "id": "unique_nan/f64/8",
                    "category": "set",
                    "operation": "unique_nan",
                }
            )
            case["validation"] = {
                "mode": "numeric_nan",
                "shape": [5],
                "size": 5,
                "logical_dtype": "f64",
                "sample_indices": [0, 2, 4],
                "values": [1.0, 5.0, 8192.0],
                "sum": 8208.0,
                "nan_count": 0,
                "first_nan_index": -1,
                "trailing_nan": False,
                "finite_members_exact": True,
            }

        with self.assertRaisesRegex(ValueError, "nan_count|raw NaN"):
            compare_results(*documents)

    def test_rejects_unverified_finite_members_with_valid_nan_facts(self) -> None:
        documents = (
            worker_document("numpy", ("sum/f64/8",)),
            worker_document("cnumpy", ("sum/f64/8",)),
        )
        for document in documents:
            case = document["cases"][0]  # type: ignore[index]
            case.update(
                {
                    "id": "unique_nan/f64/8",
                    "category": "set",
                    "operation": "unique_nan",
                }
            )
            case["validation"] = {
                "mode": "numeric_nan",
                "shape": [5],
                "size": 5,
                "logical_dtype": "f64",
                "sample_indices": [0, 2, 4],
                "values": [1.0, 5.0, 8192.0],
                "sum": 8208.0,
                "nan_count": 1,
                "first_nan_index": 4,
                "trailing_nan": True,
                "finite_members_exact": False,
            }

        with self.assertRaisesRegex(ValueError, "finite_members_exact"):
            compare_results(*documents)

    def test_accepts_small_float_rounding_differences(self) -> None:
        numpy = worker_document("numpy")
        cnumpy = worker_document("cnumpy")
        cnumpy["cases"][0]["validation"]["sum"] = 4.25000000001  # type: ignore[index]

        rows = compare_results(numpy, cnumpy)

        self.assertEqual(len(rows), 1)

    def test_argsort_compares_tie_equivalent_keys_not_arbitrary_tie_indices(self) -> None:
        numpy = worker_document("numpy", ("argsort/f64/1000000",))
        cnumpy = worker_document("cnumpy", ("argsort/f64/1000000",))
        cnumpy["cases"][0]["validation"]["values"] = [  # type: ignore[index]
            940087,
            718770,
            706978,
        ]

        rows = compare_results(numpy, cnumpy)

        self.assertEqual(len(rows), 1)

        cnumpy["cases"][0]["validation"]["values"][0] = 0  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "argsort|validation"):
            compare_results(numpy, cnumpy)

    def test_nan_argsort_compares_nan_class_instead_of_raw_tie_indices(self) -> None:
        numpy = worker_document("numpy", ("argsort_stable_nan/f64/1000000",))
        cnumpy = worker_document("cnumpy", ("argsort_stable_nan/f64/1000000",))
        cnumpy["cases"][0]["validation"]["values"] = [  # type: ignore[index]
            940087,
            718770,
            706978,
        ]

        rows = compare_results(numpy, cnumpy)

        self.assertEqual(len(rows), 1)
        cnumpy["cases"][0]["validation"]["values"][0] = 0  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "argsort|validation"):
            compare_results(numpy, cnumpy)

    def test_rejects_nonzero_retained_bytes(self) -> None:
        with self.assertRaisesRegex(ValueError, "retained_bytes"):
            compare_results(
                worker_document("numpy"),
                worker_document("cnumpy", retained_bytes=8),
            )

    def test_computes_ratio_winner_shape_and_shared_statistics(self) -> None:
        rows = compare_results(
            worker_document("numpy", median=100.0),
            worker_document("cnumpy", median=250.0),
        )

        self.assertEqual(rows[0]["shape"], "8")
        self.assertEqual(rows[0]["numpy_median_ns"], 100.0)
        self.assertEqual(rows[0]["cnumpy_median_ns"], 250.0)
        self.assertEqual(rows[0]["cnumpy_over_numpy"], 2.5)
        self.assertEqual(rows[0]["winner"], "NumPy")
        self.assertGreater(rows[0]["numpy_p95_ns"], 100.0)

    def test_normalized_worker_document_keeps_raw_samples_and_adds_summary(self) -> None:
        document = worker_document("numpy", median=100.0)

        normalized = normalize_worker_result(document, expected_runtime="numpy")

        self.assertIsNot(normalized, document)
        case = normalized["cases"][0]
        self.assertEqual(case["samples_ns"], [90.0, 100.0, 110.00000000000001])
        self.assertEqual(case["summary"]["median_ns"], 100.0)
        self.assertEqual(case["summary"]["total_timed_operations"], 12)
        self.assertNotIn("summary", document["cases"][0])


class RenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = compare_results(
            worker_document("numpy", median=100.0),
            worker_document("cnumpy", median=250.0),
        )

    def test_csv_has_stable_machine_readable_columns(self) -> None:
        text = render_comparison_csv(self.rows)
        parsed = list(csv.DictReader(io.StringIO(text)))

        self.assertEqual(
            list(parsed[0]),
            [
                "id",
                "category",
                "operation",
                "shape",
                "semantic_qualification",
                "numpy_median_ns",
                "cnumpy_median_ns",
                "cnumpy_over_numpy",
                "winner",
                "numpy_cv",
                "cnumpy_cv",
                "numpy_p95_ns",
                "cnumpy_p95_ns",
            ],
        )
        self.assertEqual(parsed[0]["cnumpy_over_numpy"], "2.5")
        self.assertEqual(
            parsed[0]["semantic_qualification"],
            TASK6_SEMANTIC_QUALIFICATION_ID,
        )
        self.assertNotIn("baseline_compatibility", parsed[0])

    def test_markdown_contains_environment_geomean_ratio_and_bottleneck_ranking(self) -> None:
        markdown = render_markdown(
            self.rows,
            environment={"run_id": "20260720T120000Z", "profile": "focus"},
        )

        self.assertIn("20260720T120000Z", markdown)
        self.assertIn("2.500x", markdown)
        self.assertIn("Geometric mean", markdown)
        self.assertIn("Bottleneck ranking", markdown)
        self.assertIn("sum/f64/8", markdown)

    def test_markdown_flags_coefficients_of_variation_above_five_percent(self) -> None:
        markdown = render_markdown(self.rows, environment={"run_id": "test"})

        self.assertIn("⚠", markdown)

    def test_baseline_adds_before_after_columns_and_improvement_factor(self) -> None:
        baseline_text = render_comparison_csv(self.rows)
        baseline = load_baseline_csv(io.StringIO(baseline_text))
        current = compare_results(
            worker_document("numpy", median=100.0),
            worker_document("cnumpy", median=125.0),
        )

        with_baseline = attach_baseline(current, baseline)
        text = render_comparison_csv(with_baseline)

        self.assertEqual(with_baseline[0]["baseline_cnumpy_over_numpy"], 2.5)
        self.assertEqual(with_baseline[0]["ratio_improvement"], 2.0)
        self.assertEqual(with_baseline[0]["baseline_compatibility"], "compatible")
        self.assertIn("baseline_cnumpy_over_numpy", text.splitlines()[0])
        self.assertIn("ratio_improvement", text.splitlines()[0])

    def test_baseline_without_semantic_qualification_reports_na(self) -> None:
        baseline = load_baseline_csv(
            io.StringIO("id,cnumpy_over_numpy\nsum/f64/8,2.5\n")
        )

        attached = attach_baseline(self.rows, baseline)

        row = attached[0]
        self.assertEqual(row["baseline_semantic_qualification"], "N/A")
        self.assertEqual(
            row["baseline_compatibility"],
            "N/A: baseline semantic qualification is missing",
        )
        self.assertEqual(row["baseline_cnumpy_over_numpy"], "N/A")
        self.assertEqual(row["ratio_improvement"], "N/A")
        self.assertNotIsInstance(row["baseline_cnumpy_over_numpy"], float)
        self.assertNotIsInstance(row["ratio_improvement"], float)
        markdown = render_markdown(
            attached,
            environment={"run_id": "new-qualified-run"},
        )
        self.assertIn("N/A: baseline semantic qualification is missing", markdown)
        self.assertIn("| `sum/f64/8` | N/A |", markdown)

    def test_baseline_with_different_semantic_qualification_reports_na(
        self,
    ) -> None:
        baseline = load_baseline_csv(
            io.StringIO(
                "id,semantic_qualification,cnumpy_over_numpy\n"
                "sum/f64/8,numpy-1.25/pre-task6-reduction,2.5\n"
            )
        )

        attached = attach_baseline(self.rows, baseline)

        row = attached[0]
        self.assertEqual(
            row["baseline_semantic_qualification"],
            "numpy-1.25/pre-task6-reduction",
        )
        self.assertIn("N/A: semantic qualification mismatch", row["baseline_compatibility"])
        self.assertEqual(row["baseline_cnumpy_over_numpy"], "N/A")
        self.assertEqual(row["ratio_improvement"], "N/A")

    def test_matching_semantic_qualification_compares_different_dll_hashes(
        self,
    ) -> None:
        current = [dict(self.rows[0], dll_sha256="new-build-hash")]
        baseline = load_baseline_csv(
            io.StringIO(
                "id,semantic_qualification,dll_sha256,cnumpy_over_numpy\n"
                f"sum/f64/8,{TASK6_SEMANTIC_QUALIFICATION_ID},old-build-hash,2.5\n"
            )
        )

        attached = attach_baseline(current, baseline)

        self.assertEqual(attached[0]["baseline_compatibility"], "compatible")
        self.assertEqual(attached[0]["baseline_cnumpy_over_numpy"], 2.5)
        self.assertEqual(attached[0]["ratio_improvement"], 1.0)
        self.assertEqual(attached[0]["dll_sha256"], "new-build-hash")

    def test_na_or_empty_semantic_qualification_never_compares(
        self,
    ) -> None:
        cases = (
            ("N/A", "N/A"),
            ("", ""),
            (TASK6_SEMANTIC_QUALIFICATION_ID, "N/A"),
        )
        for current_qualification, baseline_qualification in cases:
            with self.subTest(
                current=current_qualification,
                baseline=baseline_qualification,
            ):
                current = [
                    dict(
                        self.rows[0],
                        semantic_qualification=current_qualification,
                    )
                ]
                baseline = {
                    "sum/f64/8": {
                        "semantic_qualification": baseline_qualification,
                        "cnumpy_over_numpy": "2.5",
                    }
                }

                attached = attach_baseline(current, baseline)

                self.assertTrue(
                    str(attached[0]["baseline_compatibility"]).startswith(
                        "N/A:"
                    )
                )
                self.assertEqual(
                    attached[0]["baseline_cnumpy_over_numpy"], "N/A"
                )
                self.assertEqual(attached[0]["ratio_improvement"], "N/A")

    def test_baseline_rejects_duplicate_and_different_case_sets(self) -> None:
        duplicate = io.StringIO(
            "id,cnumpy_over_numpy\na,2\na,3\n"
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_baseline_csv(duplicate)

        with self.assertRaisesRegex(ValueError, "case set"):
            attach_baseline(self.rows, {"different": {"cnumpy_over_numpy": "2"}})

    def test_baseline_rejects_nonfinite_or_nonpositive_ratios(self) -> None:
        for value in ("0", "-1", "nan", "inf", "text"):
            with self.subTest(value=value):
                source = io.StringIO(f"id,cnumpy_over_numpy\na,{value}\n")
                with self.assertRaisesRegex(ValueError, "positive"):
                    load_baseline_csv(source)


class OrchestratorSelectionTests(unittest.TestCase):
    def test_runtime_order_requires_each_runtime_exactly_once(self) -> None:
        self.assertEqual(parse_runtime_order("numpy,cnumpy"), ("numpy", "cnumpy"))
        self.assertEqual(parse_runtime_order("cnumpy,numpy"), ("cnumpy", "numpy"))

        for invalid in ("numpy", "numpy,numpy", "numpy,other", "cnumpy, numpy"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "runtime order"):
                    parse_runtime_order(invalid)

    def test_explicit_bad_path_is_fatal_and_never_uses_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.exe"
            candidate.write_bytes(b"candidate")

            with self.assertRaisesRegex(FileNotFoundError, "explicit"):
                resolve_explicit_or_candidates(
                    root / "missing.exe",
                    [candidate],
                    "AHK executable",
                )

    def test_candidate_discovery_returns_an_absolute_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "tool.exe"
            candidate.write_bytes(b"tool")

            resolved = resolve_explicit_or_candidates(None, [candidate], "tool")

            self.assertTrue(resolved.is_absolute())
            self.assertEqual(resolved, candidate.resolve())

    def test_selection_applies_exact_case_and_category_filters(self) -> None:
        cases = [
            {"id": "sum/f64/8", "category": "reduction"},
            {"id": "zeros/f64/8", "category": "creation"},
        ]

        self.assertEqual(
            [case["id"] for case in select_cases(cases, {"sum/f64/8"}, set())],
            ["sum/f64/8"],
        )
        self.assertEqual(
            [case["id"] for case in select_cases(cases, set(), {"creation"})],
            ["zeros/f64/8"],
        )
        with self.assertRaisesRegex(ValueError, "no benchmark cases"):
            select_cases(cases, {"missing"}, set())

    def test_smoke_scaling_is_canonical_small_and_deduplicated(self) -> None:
        cases = [
            {
                "id": "sum/f64/1000",
                "category": "reduction",
                "operation": "sum",
                "dtype": "f64",
                "size": 1000,
                "rows": 0,
                "cols": 0,
                "axis": -1,
            },
            {
                "id": "sum/f64/1000000",
                "category": "reduction",
                "operation": "sum",
                "dtype": "f64",
                "size": 1000000,
                "rows": 0,
                "cols": 0,
                "axis": -1,
            },
            {
                "id": "concatenate/f64/512x512/axis0",
                "category": "shape",
                "operation": "concatenate",
                "dtype": "f64",
                "size": 262144,
                "rows": 512,
                "cols": 512,
                "axis": 0,
            },
        ]

        scaled = scale_cases_for_smoke(cases)

        self.assertEqual(
            [case["id"] for case in scaled],
            ["concatenate/f64/2x2/axis0", "sum/f64/8"],
        )
        self.assertEqual(scaled[0]["size"], 4)
        self.assertEqual(scaled[0]["rows"], 2)
        self.assertEqual(scaled[0]["cols"], 2)


class OrchestratorInfrastructureTests(unittest.TestCase):
    def test_environment_qualification_requires_exact_numpy_reference_version(
        self,
    ) -> None:
        versions = (
            ("2.x", "2.1.0"),
            ("missing", None),
            ("empty", ""),
            ("non-string", 125),
        )
        for label, actual in versions:
            with self.subTest(label=label):
                metadata: dict[str, object] = {
                    "python_version": "3.10.11",
                    "timer": "perf_counter_ns",
                }
                if label != "missing":
                    metadata["numpy_version"] = actual

                with self.assertRaises(ValueError) as caught:
                    collect_test_environment(metadata)

                message = str(caught.exception)
                self.assertIn("expected='1.25.0'", message)
                self.assertIn(f"actual={actual!r}", message)

    def test_non_reduction_selection_cannot_publish_qualification_for_wrong_numpy_version(
        self,
    ) -> None:
        with self.assertRaises(ValueError) as caught:
            collect_test_environment(
                {
                    "python_version": "3.10.11",
                    "numpy_version": "2.1.0",
                    "timer": "perf_counter_ns",
                },
                case_ids=("bridge/property_call",),
                categories=("bridge",),
            )

        message = str(caught.exception)
        self.assertIn("expected='1.25.0'", message)
        self.assertIn("actual='2.1.0'", message)

    def test_direct_script_help_and_invalid_cli_are_explicit(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = project_root / "benchmark" / "benchmark.py"

        help_result = subprocess.run(
            [sys.executable, "-B", str(script), "--help"],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--profile", help_result.stdout)
        self.assertIn("--baseline", help_result.stdout)

        invalid_result = subprocess.run(
            [sys.executable, "-B", str(script), "--samples", "2"],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(invalid_result.returncode, 0)
        self.assertIn("positive odd", invalid_result.stderr)

    def test_worker_json_loader_rejects_duplicate_keys_at_any_depth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.json"
            path.write_text(
                '{"schema_version":1,"runtime":"numpy","metadata":{"x":1,"x":2},"cases":[]}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate key"):
                load_worker_json(path)

    def test_run_directory_creation_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            created = create_run_directory(root, "20260720T120000.000000Z")

            self.assertEqual(created, root / "20260720T120000.000000Z")
            self.assertTrue(created.is_dir())
            with self.assertRaises(FileExistsError):
                create_run_directory(root, "20260720T120000.000000Z")

    def test_atomic_text_write_replaces_complete_utf8_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "report.md"
            destination.write_text("old", encoding="utf-8")

            atomic_write_text(destination, "专业 benchmark\n")

            self.assertEqual(destination.read_text(encoding="utf-8"), "专业 benchmark\n")
            self.assertEqual(list(destination.parent.glob(f".{destination.name}.*.tmp")), [])

    def test_logged_command_preserves_stdout_stderr_and_fails_visibly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stdout_log = root / "stdout.log"
            stderr_log = root / "stderr.log"
            command = [
                sys.executable,
                "-c",
                "import sys; print('out'); print('err', file=sys.stderr)",
            ]

            terminal_stdout = io.StringIO()
            terminal_stderr = io.StringIO()
            with redirect_stdout(terminal_stdout), redirect_stderr(terminal_stderr):
                result = run_logged_command(
                    command,
                    root,
                    stdout_log,
                    stderr_log,
                    encoding="utf-8",
                )

            self.assertEqual(result.returncode, 0)
            self.assertEqual(terminal_stdout.getvalue(), "out\n")
            self.assertEqual(terminal_stderr.getvalue(), "err\n")
            self.assertEqual(stdout_log.read_text(encoding="utf-8"), "out\n")
            self.assertEqual(stderr_log.read_text(encoding="utf-8"), "err\n")

            failing = [sys.executable, "-c", "import sys; print('bad'); sys.exit(7)"]
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    run_logged_command(
                        failing,
                        root,
                        stdout_log,
                        stderr_log,
                        encoding="utf-8",
                    )
            self.assertEqual(caught.exception.returncode, 7)
            self.assertEqual(stdout_log.read_text(encoding="utf-8"), "bad\n")

    def test_non_build_run_does_not_require_msbuild_discovery(self) -> None:
        self.assertIsNone(
            resolve_msbuild_executable(None, build_requested=False)
        )

        with tempfile.TemporaryDirectory() as directory:
            explicit = Path(directory) / "MSBuild.exe"
            explicit.write_bytes(b"msbuild")
            self.assertEqual(
                resolve_msbuild_executable(explicit, build_requested=False),
                explicit.resolve(),
            )

    def test_python_worker_does_not_change_the_interpreter_encoding_mode(self) -> None:
        executable = Path(sys.executable).resolve()

        self.assertEqual(python_worker_prefix(executable), [str(executable), "-B"])

    def test_sha256_and_release_settings_are_captured_from_real_files(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        project = project_root / "src" / "cnumpy_ahk.vcxproj"
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact.dll"
            artifact.write_bytes(b"cnumpy")

            digest = sha256_file(artifact)
            settings = read_release_compiler_settings(project)

        self.assertEqual(
            digest,
            "da152bad22a3ced2de81afcfef38362ddc6c03b7ea99ae1d2bbaa74282d885de",
        )
        self.assertEqual(settings["Optimization"], "MaxSpeed")
        self.assertEqual(settings["FloatingPointModel"], "Fast")
        self.assertEqual(
            settings["EnableEnhancedInstructionSet"],
            "StreamingSIMDExtensions2",
        )

    def test_environment_records_reproducibility_inputs_and_artifact_identity(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "python.exe"
            ahk = root / "AutoHotkey64.exe"
            msbuild = root / "MSBuild.exe"
            dll = root / "cnumpy_ahk.dll"
            for path, contents in (
                (python, b"python"),
                (ahk, b"ahk"),
                (msbuild, b"msbuild"),
                (dll, b"dll"),
            ):
                path.write_bytes(contents)

            environment = collect_environment(
                run_id="20260720T120000.000000Z",
                started_at_utc="2026-07-20T12:00:00+00:00",
                profile="focus",
                size_scale="native",
                case_ids=["sum/f64/8"],
                categories=[],
                warmups=5,
                samples=15,
                target_sample_ms=20.0,
                seed=12345,
                runtime_order=("numpy", "cnumpy"),
                build_requested=False,
                project_root=project_root,
                project_file=project_root / "src" / "cnumpy_ahk.vcxproj",
                python_executable=python,
                ahk_executable=ahk,
                msbuild_executable=None,
                dll_path=dll,
                numpy_metadata={"python_version": "3.10.11", "numpy_version": "1.25.0", "scipy_version": "1.12.0", "timer": "perf_counter_ns"},
                cnumpy_metadata={"ahk_version": "2.1-alpha.30", "dll_version": "0.1.0", "timer": "QueryPerformanceCounter", "timer_frequency": 10_000_000, "simd_level": 2, "simd_name": "avx2"},
                commands={"numpy": ["python", "worker"], "cnumpy": ["ahk", "worker"]},
            )

            dll_digest = sha256_file(dll)

        self.assertEqual(environment["schema_version"], 2)
        self.assertEqual(environment["run_id"], "20260720T120000.000000Z")
        self.assertEqual(environment["selection"]["case_ids"], ["sum/f64/8"])
        self.assertEqual(environment["protocol"]["sample_count"], 15)
        self.assertEqual(environment["runtime_order"], ["numpy", "cnumpy"])
        self.assertEqual(environment["runtimes"]["numpy"]["numpy_version"], "1.25.0")
        self.assertEqual(environment["runtimes"]["numpy"]["scipy_version"], "1.12.0")
        self.assertEqual(environment["runtimes"]["cnumpy"]["ahk_version"], "2.1-alpha.30")
        self.assertEqual(environment["runtimes"]["cnumpy"]["simd_level"], 2)
        self.assertEqual(environment["runtimes"]["cnumpy"]["simd_name"], "avx2")
        self.assertEqual(environment["artifacts"]["dll"]["sha256"], dll_digest)
        self.assertNotIn("semantic_qualification", environment)
        qualifications = environment["semantic_qualifications"]
        self.assertEqual(5, len(qualifications))
        qualification_by_id = {
            qualification["id"]: qualification
            for qualification in qualifications
        }
        qualification = qualification_by_id[TASK6_SEMANTIC_QUALIFICATION_ID]
        self.assertEqual(
            qualification["id"], TASK6_SEMANTIC_QUALIFICATION_ID
        )
        self.assertEqual(
            qualification["scope"],
            {"cases": [{"category": "reduction", "operations": "*"}]},
        )
        self.assertEqual(qualification["reference"], "NumPy 1.25.0")
        self.assertEqual(
            list(TASK6_SEMANTIC_QUALIFICATION_OWNERS),
            qualification["owners"],
        )
        task7 = qualification_by_id[TASK7_SEMANTIC_QUALIFICATION_ID]
        self.assertEqual("NumPy 1.25.0", task7["reference"])
        self.assertEqual(
            task7["scope"],
            {
                "cases": [
                    {
                        "category": "sorting",
                        "operations": list(
                            TASK7_SEMANTIC_QUALIFICATION_SORTING_OPERATIONS
                        ),
                    },
                    {
                        "category": "set",
                        "operations": list(
                            TASK7_SEMANTIC_QUALIFICATION_SET_OPERATIONS
                        ),
                    },
                ]
            },
        )
        task8 = qualification_by_id[TASK8_SEMANTIC_QUALIFICATION_ID]
        self.assertEqual(
            "NumPy 1.25.0; scipy.special 1.12.0",
            task8["reference"],
        )
        self.assertEqual(
            {
                "cases": [
                    {
                        "category": "misc_axis",
                        "operations": list(
                            TASK8_SEMANTIC_QUALIFICATION_OPERATIONS
                        ),
                    }
                ]
            },
            task8["scope"],
        )
        task9 = qualification_by_id[TASK9_SEMANTIC_QUALIFICATION_ID]
        self.assertEqual("NumPy 1.25.0", task9["reference"])
        self.assertEqual(
            {
                "cases": [
                    {
                        "category": "linalg",
                        "operations": list(
                            TASK9_SEMANTIC_QUALIFICATION_OPERATIONS
                        ),
                    }
                ]
            },
            task9["scope"],
        )
        self.assertEqual(
            list(TASK8_SEMANTIC_QUALIFICATION_OWNERS),
            task8["owners"],
        )
        self.assertEqual(environment["build"]["configuration"], "Release")
        self.assertEqual(environment["build"]["platform"], "x64")
        self.assertFalse(environment["build"]["requested"])
        self.assertEqual(environment["build"]["compiler_settings"]["FloatingPointModel"], "Fast")
        self.assertIsNone(environment["paths"]["msbuild"])
        for key in ("python", "ahk", "dll", "project"):
            self.assertTrue(Path(environment["paths"][key]).is_absolute())


if __name__ == "__main__":
    unittest.main()
