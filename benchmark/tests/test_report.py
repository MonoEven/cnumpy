from __future__ import annotations

import math
import unittest
from copy import deepcopy

from benchmark.report import geometric_mean, percentile, summarize_samples, validate_worker_result


def valid_document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "runtime": "numpy",
        "metadata": {
            "python_version": "3.10.11",
            "numpy_version": "1.25.0",
            "scipy_version": "1.12.0",
            "timer": "perf_counter_ns",
            "timer_resolution_ns": 100.0,
            "warmups": 1,
            "sample_count": 3,
            "target_sample_ns": 1_000_000.0,
            "seed": 12345,
        },
        "cases": [
            {
                "id": "sum/f64/8",
                "category": "reduction",
                "operation": "sum",
                "dtype": "f64",
                "size": 8,
                "rows": 0,
                "cols": 0,
                "axis": -1,
                "inner_loops": 2,
                "samples_ns": [10, 20, 30],
                "validation": {
                    "mode": "numeric",
                    "shape": [],
                    "size": 1,
                    "logical_dtype": "f64",
                    "sample_indices": [0],
                    "values": [4.25],
                    "sum": 4.25,
                },
            }
        ],
    }


def valid_cnumpy_document() -> dict[str, object]:
    document = valid_document()
    document["runtime"] = "cnumpy"
    document["metadata"] = {
        "ahk_version": "2.1-alpha.30",
        "dll_version": "1.21.0-cnumpy",
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
    document["cases"][0]["retained_bytes"] = 0  # type: ignore[index]
    return document


class PercentileTests(unittest.TestCase):
    def test_uses_linear_interpolation(self) -> None:
        samples = [0, 10, 20]

        self.assertEqual(percentile(samples, 0.5), 10)
        self.assertEqual(percentile(samples, 0.25), 5)

    def test_rejects_empty_samples_and_out_of_range_quantiles(self) -> None:
        with self.assertRaisesRegex(ValueError, "samples"):
            percentile([], 0.5)

        for quantile in (-0.01, 1.01):
            with self.subTest(quantile=quantile):
                with self.assertRaisesRegex(ValueError, "quantile"):
                    percentile([1], quantile)

    def test_handles_endpoints_single_samples_and_non_finite_quantiles(self) -> None:
        self.assertEqual(percentile([30, 10, 20], 0), 10)
        self.assertEqual(percentile([30, 10, 20], 1), 30)
        self.assertEqual(percentile([7], 0.37), 7)

        for quantile in (math.nan, math.inf, -math.inf):
            with self.subTest(quantile=quantile):
                with self.assertRaisesRegex(ValueError, "quantile"):
                    percentile([1], quantile)

    def test_accepts_negative_finite_samples(self) -> None:
        self.assertEqual(percentile([-10, -30, -20], 0.25), -25)

    def test_rejects_non_real_and_non_finite_samples(self) -> None:
        invalid_samples = ([True, 1], ["1", 2], [math.nan], [math.inf], [-math.inf])

        for samples in invalid_samples:
            with self.subTest(samples=samples):
                with self.assertRaisesRegex(ValueError, "samples"):
                    percentile(samples, 0.5)  # type: ignore[arg-type]


class SummaryTests(unittest.TestCase):
    def test_summarizes_distribution(self) -> None:
        summary = summarize_samples([10, 20, 30])

        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["median_ns"], 20)
        self.assertEqual(summary["mad_ns"], 10)
        self.assertEqual(summary["mean_ns"], 20)
        self.assertEqual(summary["min_ns"], 10)
        self.assertEqual(summary["max_ns"], 30)
        self.assertAlmostEqual(summary["stdev_ns"], math.sqrt(200 / 3))
        self.assertAlmostEqual(summary["cv"], math.sqrt(200 / 3) / 20)
        self.assertEqual(summary["p05_ns"], 11)
        self.assertEqual(summary["p25_ns"], 15)
        self.assertEqual(summary["p75_ns"], 25)
        self.assertEqual(summary["p95_ns"], 29)

    def test_rejects_invalid_samples(self) -> None:
        invalid_samples = ([], [10, 0, 20], [10, -1, 20], [10, math.inf, 20], [10, math.nan, 20])

        for samples in invalid_samples:
            with self.subTest(samples=samples):
                with self.assertRaisesRegex(ValueError, "samples"):
                    summarize_samples(samples)

    def test_handles_one_sample_and_rejects_non_numbers(self) -> None:
        summary = summarize_samples([17])

        self.assertEqual(summary["stdev_ns"], 0)
        self.assertEqual(summary["cv"], 0)
        self.assertEqual(summary["mad_ns"], 0)

        for samples in ([True], ["17"]):
            with self.subTest(samples=samples):
                with self.assertRaisesRegex(ValueError, "samples"):
                    summarize_samples(samples)


class GeometricMeanTests(unittest.TestCase):
    def test_returns_geometric_mean(self) -> None:
        self.assertEqual(geometric_mean([1, 4]), 2)

    def test_rejects_invalid_values(self) -> None:
        invalid_values = ([], [1, 0], [1, -1], [1, math.inf], [1, math.nan])

        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "values"):
                    geometric_mean(values)

    def test_uses_logarithms_and_rejects_non_numbers(self) -> None:
        self.assertAlmostEqual(geometric_mean([1e-300, 1e300]), 1)
        self.assertAlmostEqual(geometric_mean([1e308, 1e308]) / 1e308, 1)

        for values in ([True], ["4"]):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "values"):
                    geometric_mean(values)


class WorkerResultValidationTests(unittest.TestCase):
    def test_accepts_valid_document_and_preserves_unknown_fields(self) -> None:
        document = valid_document()
        document["future_top_level_field"] = {"kept": True}
        cases = document["cases"]
        assert isinstance(cases, list)
        cases[0]["future_case_field"] = "kept"
        original_document = deepcopy(document)

        result = validate_worker_result(
            document,
            expected_runtime="numpy",
            expected_case_ids=["sum/f64/8"],
        )

        self.assertIs(result, document)
        self.assertEqual(document, original_document)
        self.assertIn("future_top_level_field", result)
        self.assertIn("future_case_field", cases[0])

    def test_rejects_invalid_top_level_fields(self) -> None:
        invalid_documents = (
            ({}, "schema_version"),
            ({**valid_document(), "schema_version": 2}, "schema_version"),
            ({**valid_document(), "runtime": ""}, "runtime"),
            ({**valid_document(), "metadata": []}, "metadata"),
            ({**valid_document(), "cases": []}, "cases"),
        )

        for document, field in invalid_documents:
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    validate_worker_result(document)

    def test_rejects_invalid_case_fields(self) -> None:
        mutations = (
            ("id", ""),
            ("inner_loops", 0),
            ("inner_loops", True),
            ("samples_ns", [1, 2]),
            ("samples_ns", [1, 0, 3]),
            ("samples_ns", [1, math.inf, 3]),
            ("validation", []),
        )

        for field, value in mutations:
            with self.subTest(field=field, value=value):
                document = valid_document()
                cases = document["cases"]
                assert isinstance(cases, list)
                cases[0][field] = value
                with self.assertRaisesRegex(ValueError, field):
                    validate_worker_result(document)

        duplicate = valid_document()
        cases = duplicate["cases"]
        assert isinstance(cases, list)
        cases.append(deepcopy(cases[0]))
        with self.assertRaisesRegex(ValueError, "id"):
            validate_worker_result(duplicate)

    def test_enforces_expected_runtime_and_case_ids(self) -> None:
        document = valid_document()

        with self.assertRaisesRegex(ValueError, "runtime"):
            validate_worker_result(document, expected_runtime="cnumpy")
        with self.assertRaisesRegex(ValueError, "case"):
            validate_worker_result(document, expected_case_ids=["different_case"])

    def test_rejects_wrong_container_and_scalar_types(self) -> None:
        with self.assertRaisesRegex(ValueError, "document"):
            validate_worker_result([])  # type: ignore[arg-type]

        invalid_documents = (
            ({**valid_document(), "schema_version": True}, "schema_version"),
            ({**valid_document(), "runtime": 1}, "runtime"),
            ({**valid_document(), "cases": tuple()}, "cases"),
        )
        for document, field in invalid_documents:
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    validate_worker_result(document)

        invalid_case_values = (
            ("id", 1),
            ("inner_loops", 1.0),
            ("samples_ns", (1, 2, 3)),
            ("samples_ns", []),
            ("samples_ns", [1, True, 3]),
            ("samples_ns", [1, "2", 3]),
            ("samples_ns", [1, math.nan, 3]),
        )
        for field, value in invalid_case_values:
            with self.subTest(field=field, value=value):
                document = valid_document()
                cases = document["cases"]
                assert isinstance(cases, list)
                cases[0][field] = value
                with self.assertRaisesRegex(ValueError, field):
                    validate_worker_result(document)

    def test_expected_case_ids_match_as_a_set(self) -> None:
        document = valid_document()
        cases = document["cases"]
        assert isinstance(cases, list)
        second_case = deepcopy(cases[0])
        second_case["id"] = "max/f64/8"
        second_case["operation"] = "max"
        cases.append(second_case)

        self.assertIs(
            validate_worker_result(
                document,
                expected_case_ids=["sum/f64/8", "max/f64/8"],
            ),
            document,
        )
        self.assertIs(
            validate_worker_result(
                document,
                expected_case_ids=["max/f64/8", "sum/f64/8"],
            ),
            document,
        )

    def test_rejects_duplicate_expected_case_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected_case_ids.*duplicate"):
            validate_worker_result(
                valid_document(),
                expected_case_ids=["sum/f64/8", "sum/f64/8"],
            )

    def test_rejects_missing_or_extra_expected_case_ids(self) -> None:
        document = valid_document()
        cases = document["cases"]
        assert isinstance(cases, list)
        second_case = deepcopy(cases[0])
        second_case["id"] = "max/f64/8"
        second_case["operation"] = "max"
        cases.append(second_case)

        invalid_expected_ids = (
            ["sum/f64/8"],
            ["sum/f64/8", "max/f64/8", "extra_case"],
        )
        for expected_ids in invalid_expected_ids:
            with self.subTest(expected_ids=expected_ids):
                with self.assertRaisesRegex(ValueError, "case ids.*do not match"):
                    validate_worker_result(document, expected_case_ids=expected_ids)

    def test_expected_case_ids_require_non_empty_strings(self) -> None:
        for invalid_id in ("", 1):
            with self.subTest(invalid_id=invalid_id):
                with self.assertRaisesRegex(
                    ValueError,
                    r"expected_case_ids\[0\].*non-empty string",
                ):
                    validate_worker_result(
                        valid_document(),
                        expected_case_ids=[invalid_id],  # type: ignore[list-item]
                    )

    def test_rejects_missing_or_noncanonical_case_metadata(self) -> None:
        missing = valid_document()
        del missing["cases"][0]["category"]  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "category"):
            validate_worker_result(missing)

        noncanonical = valid_document()
        noncanonical["cases"][0]["id"] = "wrong"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "canonical|id"):
            validate_worker_result(noncanonical)

    def test_rejects_malformed_validation_signature(self) -> None:
        document = valid_document()
        del document["cases"][0]["validation"]["sum"]  # type: ignore[index]

        with self.assertRaisesRegex(ValueError, "validation.*sum|missing"):
            validate_worker_result(document)

    def test_validation_signature_must_match_the_operation_contract(self) -> None:
        wrong_shape = valid_document()
        validation = wrong_shape["cases"][0]["validation"]  # type: ignore[index]
        validation["shape"] = [1]
        validation["size"] = 1
        with self.assertRaisesRegex(ValueError, "validation.*contract|operation"):
            validate_worker_result(wrong_shape)

        wrong_dtype = valid_document()
        validation = wrong_dtype["cases"][0]["validation"]  # type: ignore[index]
        validation["logical_dtype"] = "i64"
        validation["values"] = [4.0]
        validation["sum"] = 4.0
        with self.assertRaisesRegex(ValueError, "validation.*contract|operation"):
            validate_worker_result(wrong_dtype)

        wrong_mode = valid_document()
        wrong_mode["cases"][0]["validation"] = {  # type: ignore[index]
            "mode": "shape",
            "shape": [],
            "size": 1,
            "logical_dtype": "f64",
        }
        with self.assertRaisesRegex(ValueError, "validation.*contract|operation"):
            validate_worker_result(wrong_mode)

    def test_searchsorted_worker_contract_requires_i64_validation(self) -> None:
        for operation in ("searchsorted", "searchsorted_right"):
            document = valid_document()
            case = document["cases"][0]  # type: ignore[index]
            case.update(
                {
                    "id": f"{operation}/f64/8",
                    "category": "sorting",
                    "operation": operation,
                    "validation": {
                        "mode": "numeric",
                        "shape": [8],
                        "size": 8,
                        "logical_dtype": "i64",
                        "sample_indices": [0, 4, 7],
                        "values": [1.0, 3.0, 8.0],
                        "sum": 30.0,
                    },
                }
            )
            with self.subTest(operation=operation):
                self.assertIs(validate_worker_result(document), document)

    def test_digitize_worker_contract_requires_i64_validation(self) -> None:
        for operation in ("digitize", "digitize_decreasing"):
            document = valid_document()
            case = document["cases"][0]  # type: ignore[index]
            case.update(
                {
                    "id": f"{operation}/f64/8",
                    "category": "sorting",
                    "operation": operation,
                    "validation": {
                        "mode": "numeric",
                        "shape": [8],
                        "size": 8,
                        "logical_dtype": "i64",
                        "sample_indices": [0, 4, 7],
                        "values": [1.0, 3.0, 8.0],
                        "sum": 30.0,
                    },
                }
            )
            with self.subTest(operation=operation):
                self.assertIs(validate_worker_result(document), document)

    def test_lexsort_worker_contract_requires_i64_validation(self) -> None:
        document = valid_document()
        case = document["cases"][0]  # type: ignore[index]
        case.update(
            {
                "id": "lexsort/f64/8",
                "category": "sorting",
                "operation": "lexsort",
                "validation": {
                    "mode": "numeric",
                    "shape": [8],
                    "size": 8,
                    "logical_dtype": "i64",
                    "sample_indices": [0, 4, 7],
                    "values": [1.0, 3.0, 7.0],
                    "sum": 28.0,
                },
            }
        )
        self.assertIs(validate_worker_result(document), document)

    def test_msort_and_sort_complex_worker_contracts_are_f64(self) -> None:
        for operation in ("msort", "sort_complex"):
            document = valid_document()
            case = document["cases"][0]  # type: ignore[index]
            case.update(
                {
                    "id": f"{operation}/f64/8",
                    "category": "sorting",
                    "operation": operation,
                    "validation": {
                        "mode": "numeric",
                        "shape": [8],
                        "size": 8,
                        "logical_dtype": "f64",
                        "sample_indices": [0, 4, 7],
                        "values": [1.0, 3.0, 7.0],
                        "sum": 28.0,
                    },
                }
            )
            with self.subTest(operation=operation):
                self.assertIs(validate_worker_result(document), document)

    def test_equal_worker_contract_requires_boolean_validation_values(self) -> None:
        document = valid_document()
        case = document["cases"][0]  # type: ignore[index]
        case.update(
            {
                "id": "equal/f64/8",
                "category": "comparison",
                "operation": "equal",
                "validation": {
                    "mode": "numeric",
                    "shape": [8],
                    "size": 8,
                    "logical_dtype": "bool",
                    "sample_indices": [0, 4, 7],
                    "values": [1.0, 0.0, 1.0],
                    "sum": 5.0,
                },
            }
        )
        self.assertIs(validate_worker_result(document), document)

        invalid = deepcopy(document)
        invalid["cases"][0]["validation"]["values"][1] = 2.0  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "bool|boolean"):
            validate_worker_result(invalid)

    def test_logical_worker_contracts_require_boolean_validation_values(
        self,
    ) -> None:
        for operation in (
            "logical_and", "logical_or", "logical_xor", "logical_not",
            "isnan", "isinf", "isfinite", "signbit",
        ):
            document = valid_document()
            case = document["cases"][0]  # type: ignore[index]
            case.update(
                {
                    "id": f"{operation}/f64/8",
                    "category": "logical",
                    "operation": operation,
                    "validation": {
                        "mode": "numeric",
                        "shape": [8],
                        "size": 8,
                        "logical_dtype": "bool",
                        "sample_indices": [0, 4, 7],
                        "values": [1.0, 0.0, 1.0],
                        "sum": 5.0,
                    },
                }
            )
            with self.subTest(operation=operation):
                self.assertIs(validate_worker_result(document), document)

            invalid = deepcopy(document)
            invalid["cases"][0]["validation"]["values"][1] = 2.0  # type: ignore[index]
            with self.subTest(operation=operation, invalid="bool value"):
                with self.assertRaisesRegex(ValueError, "bool|boolean"):
                    validate_worker_result(invalid)

    def test_bitwise_worker_contracts_require_i64_jobs_and_validation(self) -> None:
        for operation in (
            "bitwise_and", "bitwise_or", "bitwise_xor", "invert",
            "left_shift", "right_shift",
        ):
            document = valid_document()
            case = document["cases"][0]  # type: ignore[index]
            case.update(
                {
                    "id": f"{operation}/i64/8",
                    "category": "bitwise",
                    "operation": operation,
                    "dtype": "i64",
                    "validation": {
                        "mode": "numeric",
                        "shape": [8],
                        "size": 8,
                        "logical_dtype": "i64",
                        "sample_indices": [0, 4, 7],
                        "values": [-2040.0, -1024.0, 256.0],
                        "sum": -4096.0,
                    },
                }
            )
            with self.subTest(operation=operation):
                self.assertIs(validate_worker_result(document), document)

            wrong_job_dtype = deepcopy(document)
            wrong_job_dtype["cases"][0]["id"] = f"{operation}/f64/8"  # type: ignore[index]
            wrong_job_dtype["cases"][0]["dtype"] = "f64"  # type: ignore[index]
            with self.subTest(operation=operation, invalid="job dtype"):
                with self.assertRaisesRegex(ValueError, "dtype"):
                    validate_worker_result(wrong_job_dtype)

            wrong_validation_dtype = deepcopy(document)
            wrong_validation_dtype["cases"][0]["validation"]["logical_dtype"] = "f64"  # type: ignore[index]
            with self.subTest(operation=operation, invalid="validation dtype"):
                with self.assertRaisesRegex(ValueError, "validation.*contract|operation"):
                    validate_worker_result(wrong_validation_dtype)

    def test_rejects_invalid_runtime_metadata(self) -> None:
        document = valid_document()
        document["metadata"]["sample_count"] = 2  # type: ignore[index]

        with self.assertRaisesRegex(ValueError, "metadata.*sample_count"):
            validate_worker_result(document)

    def test_cnumpy_metadata_requires_a_consistent_simd_level_and_name(self) -> None:
        valid = valid_cnumpy_document()
        self.assertIs(validate_worker_result(valid), valid)

        invalid_pairs = (
            (0, "sse2"),
            (3, "avx2"),
            (1, "avx2"),
            (2, "sse2"),
            (True, "sse2"),
        )
        for level, name in invalid_pairs:
            with self.subTest(level=level, name=name):
                document = valid_cnumpy_document()
                document["metadata"]["simd_level"] = level  # type: ignore[index]
                document["metadata"]["simd_name"] = name  # type: ignore[index]
                with self.assertRaisesRegex(ValueError, "metadata.*simd"):
                    validate_worker_result(document)

        for missing_field in ("simd_level", "simd_name"):
            with self.subTest(missing_field=missing_field):
                document = valid_cnumpy_document()
                del document["metadata"][missing_field]  # type: ignore[index]
                with self.assertRaisesRegex(ValueError, "metadata.*simd"):
                    validate_worker_result(document)


if __name__ == "__main__":
    unittest.main()
