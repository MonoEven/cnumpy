from __future__ import annotations

import io
import json
import math
import re
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np
from scipy import special as scipy_special
import benchmark.bench_numpy as bench_numpy

from benchmark.bench_numpy import (
    binary_vector,
    general_vector,
    predicate_vector,
    prepare_case,
    read_jobs_tsv,
    sorting_vector,
    time_operation,
    validate_argsort_indices,
    validation_signature,
)
from benchmark.report import expand_cases, validate_worker_result, write_jobs_tsv


HEADER = "id\tcategory\toperation\tdtype\tsize\trows\tcols\taxis\n"
CATALOG_PATH = Path(__file__).resolve().parents[1] / "cases.json"
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "focus-small.tsv"


def job_row(
    case_id: str = "zeros/f64/8",
    category: str = "creation",
    operation: str = "zeros",
    size: str = "8",
    rows: str = "0",
    cols: str = "0",
    axis: str = "-1",
) -> str:
    return "\t".join(
        (case_id, category, operation, "f64", size, rows, cols, axis)
    ) + "\n"


def benchmark_case(
    operation: str,
    category: str,
    *,
    dtype: str = "f64",
    size: int = 8,
    rows: int = 0,
    cols: int = 0,
    axis: int = -1,
) -> dict[str, object]:
    if category == "bridge":
        case_id = f"bridge/{operation}"
    elif rows:
        case_id = f"{operation}/{dtype}/{rows}x{cols}"
        if axis != -1:
            case_id += f"/axis{axis}"
    else:
        case_id = f"{operation}/{dtype}/{size}"
    return {
        "id": case_id,
        "category": category,
        "operation": operation,
        "dtype": dtype,
        "size": size,
        "rows": rows,
        "cols": cols,
        "axis": axis,
    }


class NumPyWorkerPublicApiTests(unittest.TestCase):
    def test_public_worker_api_is_importable(self) -> None:
        from benchmark.bench_numpy import (
            PreparedCase,
            binary_vector,
            general_vector,
            predicate_vector,
            prepare_case,
            read_jobs_tsv,
            sorting_vector,
            time_operation,
            validation_signature,
        )

        for public_api in (
            PreparedCase,
            binary_vector,
            general_vector,
            prepare_case,
            read_jobs_tsv,
            sorting_vector,
            time_operation,
            validation_signature,
        ):
            self.assertIsNotNone(public_api)


class DeterministicInputTests(unittest.TestCase):
    def test_general_and_binary_vectors_follow_wire_formulas_exactly(self) -> None:
        size = 8
        expected_general = np.array(
            [((index * 37 + 11) % 1009) / 1009.0 + 0.01 for index in range(size)],
            dtype=np.float64,
        )
        expected_binary = np.array(
            [((index * 53 + 19) % 1013) / 1013.0 + 0.02 for index in range(size)],
            dtype=np.float64,
        )

        np.testing.assert_array_equal(general_vector(size), expected_general)
        np.testing.assert_array_equal(binary_vector(size), expected_binary)
        np.testing.assert_array_equal(general_vector(size), general_vector(size))
        np.testing.assert_array_equal(binary_vector(size), binary_vector(size))

    def test_sorting_vector_is_exact_repeatable_and_contains_duplicates(self) -> None:
        size = 5_000
        expected = np.array(
            [float(((index * 48271 + 17) % 65521) % 4096) for index in range(size)],
            dtype=np.float64,
        )

        actual = sorting_vector(size)

        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual, sorting_vector(size))
        self.assertLess(np.unique(actual).size, size)

    def test_nan_sorting_vector_is_repeatable_and_half_nan(self) -> None:
        size = 10_000

        actual = sorting_vector(size, "nan")

        np.testing.assert_array_equal(actual, sorting_vector(size, "nan"))
        self.assertEqual(size // 2, int(np.count_nonzero(np.isnan(actual))))
        self.assertTrue(np.all(np.isfinite(actual[1::2])))

    def test_predicate_vector_repeats_all_required_floating_edges(self) -> None:
        expected_bits = np.array(
            [
                0x7FF8000000000000,
                0x7FF0000000000000,
                0xFFF0000000000000,
                0x0000000000000000,
                0x8000000000000000,
                0x3FF4000000000000,
                0xC004000000000000,
                0x4008000000000000,
            ],
            dtype=np.uint64,
        )

        actual = predicate_vector(16)

        np.testing.assert_array_equal(actual[:8].view(np.uint64), expected_bits)
        np.testing.assert_array_equal(actual[8:].view(np.uint64), expected_bits)
        np.testing.assert_array_equal(actual, predicate_vector(16))


class JobReaderTests(unittest.TestCase):
    def test_reads_exact_tsv_header_and_converts_numeric_fields(self) -> None:
        jobs = read_jobs_tsv(io.StringIO(HEADER + job_row()))

        self.assertEqual(
            jobs,
            [
                {
                    "id": "zeros/f64/8",
                    "category": "creation",
                    "operation": "zeros",
                    "dtype": "f64",
                    "size": 8,
                    "rows": 0,
                    "cols": 0,
                    "axis": -1,
                }
            ],
        )

    def test_rejects_empty_bad_header_bad_integer_and_empty_rows(self) -> None:
        bad_inputs = (
            "",
            HEADER.replace("axis", "wrong") + job_row(),
            HEADER + job_row(size="eight"),
            HEADER,
            HEADER + "\n",
        )
        for contents in bad_inputs:
            with self.subTest(contents=contents):
                with self.assertRaises(ValueError):
                    read_jobs_tsv(io.StringIO(contents))

    def test_reuses_job_validation_for_duplicates_and_field_contracts(self) -> None:
        duplicate = HEADER + job_row() + job_row()
        wrong_dtype = HEADER + job_row().replace("\tf64\t", "\tf32\t")

        for contents in (duplicate, wrong_dtype):
            with self.subTest(contents=contents):
                with self.assertRaises(ValueError):
                    read_jobs_tsv(io.StringIO(contents))

    def test_rejects_jobs_that_violate_operation_semantics(self) -> None:
        invalid_jobs = (
            (job_row(category="unary"), "zeros/f64/8", "category"),
            (job_row(case_id="wrong/f64/8"), "wrong/f64/8", "id"),
            (job_row(axis="0"), "zeros/f64/8", "axis"),
            (
                job_row(
                    case_id="mystery/f64/8",
                    category="creation",
                    operation="mystery",
                ),
                "mystery/f64/8",
                "operation",
            ),
            (job_row(rows="2", cols="4"), "zeros/f64/8", "rows"),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            jobs_path = Path(temporary_directory) / "jobs.tsv"
            for row, case_id, field in invalid_jobs:
                with self.subTest(case_id=case_id, field=field):
                    jobs_path.write_text(HEADER + row, encoding="utf-8")
                    with self.assertRaisesRegex(
                        ValueError,
                        rf"{re.escape(case_id)}.*{field}",
                    ):
                        read_jobs_tsv(jobs_path)


class ValidationSignatureTests(unittest.TestCase):
    def test_numeric_signature_has_logical_shape_samples_and_sum(self) -> None:
        signature = validation_signature(np.array([1.5, 2.5, 3.5]), (3,))

        self.assertEqual(
            signature,
            {
                "mode": "numeric",
                "shape": [3],
                "size": 3,
                "sample_indices": [0, 1, 2],
                "values": [1.5, 2.5, 3.5],
                "sum": 7.5,
                "logical_dtype": "f64",
            },
        )

    def test_scalar_and_short_results_deduplicate_sample_indices(self) -> None:
        scalar = validation_signature(np.float64(4.25), ())
        pair = validation_signature(np.array([10, 20], dtype=np.int64), (2,), logical_dtype="i64")

        self.assertEqual(scalar["shape"], [])
        self.assertEqual(scalar["sample_indices"], [0])
        self.assertEqual(scalar["values"], [4.25])
        self.assertEqual(pair["sample_indices"], [0, 1])
        self.assertEqual(pair["values"], [10.0, 20.0])
        self.assertEqual(pair["logical_dtype"], "i64")

    def test_shape_mode_omits_numeric_values(self) -> None:
        signature = validation_signature(np.zeros((2, 3)), (2, 3), mode="shape")

        self.assertEqual(
            signature,
            {"mode": "shape", "shape": [2, 3], "size": 6, "logical_dtype": "f64"},
        )

    def test_numeric_nan_signature_records_raw_nan_facts_before_normalizing(self) -> None:
        signature = validation_signature(
            np.array([1.0, 2.0, np.nan, np.nan]),
            (4,),
            mode="numeric_nan",
            expected_finite_members=np.array([1.0, 2.0]),
        )

        self.assertEqual(
            signature,
            {
                "mode": "numeric_nan",
                "shape": [4],
                "size": 4,
                "logical_dtype": "f64",
                "sample_indices": [0, 2, 3],
                "values": [1.0, 8192.0, 8192.0],
                "sum": 16387.0,
                "nan_count": 2,
                "first_nan_index": 2,
                "trailing_nan": True,
                "finite_members_exact": True,
            },
        )

    def test_numeric_nan_signature_rejects_a_finite_8192_counterfeit(self) -> None:
        with self.assertRaisesRegex(ValueError, "raw.*NaN"):
            validation_signature(
                np.array([1.0, 2.0, 8192.0, 8192.0]),
                (4,),
                mode="numeric_nan",
                expected_finite_members=np.array([1.0, 2.0]),
            )

    def test_numeric_nan_signature_rejects_a_wrong_finite_member(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite members"):
            validation_signature(
                np.array([1.0, 3.0, np.nan, np.nan]),
                (4,),
                mode="numeric_nan",
                expected_finite_members=np.array([1.0, 2.0]),
            )

    def test_bool_signature_preserves_boolean_result_contract(self) -> None:
        signature = validation_signature(
            np.array([True, False, True], dtype=np.bool_),
            (3,),
            logical_dtype="bool",
        )

        self.assertEqual(signature["logical_dtype"], "bool")
        self.assertEqual(signature["values"], [1.0, 0.0, 1.0])
        self.assertEqual(signature["sum"], 2.0)
        with self.assertRaisesRegex(ValueError, "logical dtype"):
            validation_signature(
                np.array([1, 0, 1], dtype=np.int64),
                (3,),
                logical_dtype="bool",
            )

    def test_rejects_shape_mismatch_nonfinite_values_and_unknown_mode(self) -> None:
        invalid_calls = (
            lambda: validation_signature(np.zeros(2), (3,)),
            lambda: validation_signature(np.array([np.inf]), (1,)),
            lambda: validation_signature(np.zeros(1), (1,), mode="unknown"),
        )
        for invalid_call in invalid_calls:
            with self.subTest(invalid_call=invalid_call):
                with self.assertRaises(ValueError):
                    invalid_call()

    def test_rejects_a_nonfinite_numeric_aggregate(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum.*finite"):
            validation_signature(np.array([1e308, 1e308]), (2,))

    def test_rejects_values_that_do_not_match_the_logical_dtype(self) -> None:
        invalid_calls = (
            lambda: validation_signature(np.array([1.0]), (1,), logical_dtype="i64"),
            lambda: validation_signature(np.array([1]), (1,), logical_dtype="f64"),
            lambda: validation_signature(np.array([True]), (1,), logical_dtype="i64"),
        )
        for invalid_call in invalid_calls:
            with self.subTest(invalid_call=invalid_call):
                with self.assertRaisesRegex(ValueError, "logical dtype"):
                    invalid_call()


class PreparedOperationTests(unittest.TestCase):
    VECTOR_OPERATIONS = {
        "creation": ("zeros", "ones", "arange", "random", "linspace"),
        "random": ("choice_weighted",),
        "unary": (
            "sin", "cos", "exp", "expm1", "sqrt", "log", "log2", "log10", "log1p", "absolute",
            "floor", "tanh", "angle", "real", "imag", "real_if_close",
        ),
        "binary": (
            "add", "subtract", "multiply", "divide", "divmod", "power", "float_power", "heaviside", "maximum",
            "minimum", "fmax", "fmin",
            "logaddexp", "logaddexp2",
        ),
        "logical": (
            "logical_and", "logical_or", "logical_xor", "logical_not",
            "isnan", "isinf", "isfinite", "signbit",
            "iscomplexobj", "isrealobj", "isscalar",
        ),
        "bitwise": (
            "bitwise_and", "bitwise_or", "bitwise_xor", "invert",
            "left_shift", "right_shift",
        ),
        "integer": ("gcd", "lcm"),
        "signal": ("convolve", "correlate"),
        "comparison": ("allclose", "equal"),
        "reduction": (
            "sum", "mean", "average", "std", "max", "min",
            "argmax", "cumsum", "prod",
        ),
        "sorting": (
            "sort",
            "argsort",
            "sort_mergesort",
            "argsort_mergesort",
            "sort_heapsort",
            "argsort_heapsort",
            "sort_stable",
            "argsort_stable",
            "sort_stable_nan",
            "argsort_stable_nan",
            "partition",
            "argpartition",
            "partition_nan",
            "argpartition_nan",
            "searchsorted",
            "searchsorted_right",
            "digitize",
            "digitize_decreasing",
            "lexsort",
            "msort",
            "sort_complex",
        ),
        "set": (
            "unique_duplicates", "unique_nan",
            "intersect1d_duplicates", "union1d_duplicates",
            "setdiff1d_duplicates", "setxor1d_duplicates",
            "in1d_duplicates", "isin_duplicates",
        ),
        "shape": (
            "copy", "reshape", "flatten",
            "atleast_1d", "atleast_2d", "atleast_3d",
        ),
        "indexing": ("take", "compress"),
        "preallocated": ("add_into", "sqrt_into", "cumsum_into"),
        "pipeline": ("pipeline_separate", "pipeline_batch"),
    }

    def test_expm1_invokes_numpy_125_ufunc(self) -> None:
        prepared = prepare_case(benchmark_case("expm1", "unary"))
        expected = np.expm1(general_vector(8))

        np.testing.assert_array_equal(prepared.invoke(), expected)
        np.testing.assert_array_equal(prepared.validation(), expected)
        self.assertEqual((8,), prepared.expected_shape)
        self.assertEqual("f64", prepared.logical_dtype)
        self.assertEqual("numeric", prepared.validation_mode)

    def test_axis0_block_and_strided_indexing_payloads_are_equivalent(
        self,
    ) -> None:
        block_operand = bench_numpy.symmetric_indexing_matrix(4, 4)
        strided_operand = bench_numpy.symmetric_indexing_matrix(
            4, 4, strided=True
        )
        self.assertTrue(block_operand.flags.c_contiguous)
        self.assertFalse(strided_operand.flags.c_contiguous)
        self.assertTrue(strided_operand.flags.f_contiguous)
        np.testing.assert_array_equal(block_operand, strided_operand)

        prepared = {}
        for operation in (
            "take_axis0_block",
            "take_axis0_strided",
            "compress_axis0_block",
            "compress_axis0_strided",
        ):
            prepared[operation] = prepare_case(
                benchmark_case(
                    operation, "indexing", size=16, rows=4, cols=4, axis=0
                )
            )
            self.assertEqual((2, 4), prepared[operation].expected_shape)
            self.assertEqual("f64", prepared[operation].logical_dtype)

        take_block = prepared["take_axis0_block"].invoke()
        take_strided = prepared["take_axis0_strided"].invoke()
        compress_block = prepared["compress_axis0_block"].invoke()
        compress_strided = prepared["compress_axis0_strided"].invoke()
        np.testing.assert_array_equal(take_block, take_strided)
        np.testing.assert_array_equal(compress_block, compress_strided)
        np.testing.assert_array_equal(take_block, compress_block)
        for operation, worker_case in prepared.items():
            with self.subTest(operation=operation):
                np.testing.assert_array_equal(
                    worker_case.invoke(), worker_case.validation()
                )

    def test_average_invokes_numpy_125_with_deterministic_weights(
        self,
    ) -> None:
        prepared = prepare_case(benchmark_case("average", "reduction"))
        expected = np.average(
            general_vector(8), weights=binary_vector(8)
        )

        np.testing.assert_array_equal(prepared.invoke(), expected)
        np.testing.assert_array_equal(prepared.validation(), expected)
        self.assertEqual((), prepared.expected_shape)
        self.assertEqual("f64", prepared.logical_dtype)
        self.assertEqual("numeric", prepared.validation_mode)

    def test_log2_invokes_numpy_125_ufunc(self) -> None:
        prepared = prepare_case(benchmark_case("log2", "unary"))
        expected = np.log2(general_vector(8))

        np.testing.assert_array_equal(prepared.invoke(), expected)
        np.testing.assert_array_equal(prepared.validation(), expected)
        self.assertEqual((8,), prepared.expected_shape)
        self.assertEqual("f64", prepared.logical_dtype)
        self.assertEqual("numeric", prepared.validation_mode)

    def test_log10_invokes_numpy_125_ufunc(self) -> None:
        prepared = prepare_case(benchmark_case("log10", "unary"))
        expected = np.log10(general_vector(8))

        np.testing.assert_array_equal(prepared.invoke(), expected)
        np.testing.assert_array_equal(prepared.validation(), expected)
        self.assertEqual((8,), prepared.expected_shape)
        self.assertEqual("f64", prepared.logical_dtype)
        self.assertEqual("numeric", prepared.validation_mode)

    def test_log1p_invokes_numpy_125_ufunc(self) -> None:
        prepared = prepare_case(benchmark_case("log1p", "unary"))
        expected = np.log1p(general_vector(8))

        np.testing.assert_array_equal(prepared.invoke(), expected)
        np.testing.assert_array_equal(prepared.validation(), expected)
        self.assertEqual((8,), prepared.expected_shape)
        self.assertEqual("f64", prepared.logical_dtype)
        self.assertEqual("numeric", prepared.validation_mode)

    def test_logaddexp_invokes_numpy_125_ufunc(self) -> None:
        prepared = prepare_case(benchmark_case("logaddexp", "binary"))
        expected = np.logaddexp(general_vector(8), binary_vector(8))

        np.testing.assert_array_equal(prepared.invoke(), expected)
        np.testing.assert_array_equal(prepared.validation(), expected)
        self.assertEqual((8,), prepared.expected_shape)
        self.assertEqual("f64", prepared.logical_dtype)
        self.assertEqual("numeric", prepared.validation_mode)

    def test_float_power_invokes_numpy_125_ufunc(self) -> None:
        prepared = prepare_case(benchmark_case("float_power", "binary"))
        expected = np.float_power(general_vector(8), binary_vector(8))

        np.testing.assert_array_equal(prepared.invoke(), expected)
        np.testing.assert_array_equal(prepared.validation(), expected)
        self.assertEqual((8,), prepared.expected_shape)
        self.assertEqual("f64", prepared.logical_dtype)
        self.assertEqual("numeric", prepared.validation_mode)

    def test_heaviside_invokes_numpy_125_ufunc(self) -> None:
        prepared = prepare_case(benchmark_case("heaviside", "binary"))
        expected = np.heaviside(general_vector(8), binary_vector(8))

        np.testing.assert_array_equal(prepared.invoke(), expected)
        np.testing.assert_array_equal(prepared.validation(), expected)
        self.assertEqual((8,), prepared.expected_shape)
        self.assertEqual("f64", prepared.logical_dtype)
        self.assertEqual("numeric", prepared.validation_mode)

    def test_gcd_lcm_invoke_numpy_125_integer_ufuncs(self) -> None:
        for operation, oracle in (("gcd", np.gcd), ("lcm", np.lcm)):
            with self.subTest(operation=operation):
                prepared = prepare_case(
                    benchmark_case(operation, "integer", dtype="i64")
                )
                expected = oracle(
                    np.asarray(
                        [-2037, -2000, -1963, -1926,
                         -1889, -1852, -1815, -1778],
                        dtype=np.int64,
                    ),
                    np.asarray(
                        [-2029, -1976, -1923, -1870,
                         -1817, -1764, -1711, -1658],
                        dtype=np.int64,
                    ),
                )
                np.testing.assert_array_equal(prepared.invoke(), expected)
                np.testing.assert_array_equal(prepared.validation(), expected)
                self.assertEqual((8,), prepared.expected_shape)
                self.assertEqual("i64", prepared.logical_dtype)
                self.assertEqual("numeric", prepared.validation_mode)

    def test_divmod_times_both_numpy_outputs_and_validates_the_stacked_pair(
        self,
    ) -> None:
        prepared = prepare_case(benchmark_case("divmod", "binary"))
        expected = np.divmod(general_vector(8), binary_vector(8))

        actual = prepared.invoke()
        self.assertIsInstance(actual, tuple)
        self.assertEqual(2, len(actual))
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(
            prepared.validation(), np.stack(expected, axis=0)
        )
        self.assertEqual((2, 8), prepared.expected_shape)
        self.assertEqual("f64", prepared.logical_dtype)
        self.assertEqual("numeric", prepared.validation_mode)

    def test_logaddexp2_invokes_numpy_125_ufunc(self) -> None:
        prepared = prepare_case(benchmark_case("logaddexp2", "binary"))
        expected = np.logaddexp2(general_vector(8), binary_vector(8))

        np.testing.assert_array_equal(prepared.invoke(), expected)
        np.testing.assert_array_equal(prepared.validation(), expected)
        self.assertEqual((8,), prepared.expected_shape)
        self.assertEqual("f64", prepared.logical_dtype)
        self.assertEqual("numeric", prepared.validation_mode)

    def test_equal_invokes_numpy_125_ufunc_and_preserves_bool_dtype(self) -> None:
        prepared = prepare_case(benchmark_case("equal", "comparison"))
        expected = np.equal(general_vector(8), binary_vector(8))

        np.testing.assert_array_equal(prepared.invoke(), expected)
        np.testing.assert_array_equal(prepared.validation(), expected)
        self.assertEqual((8,), prepared.expected_shape)
        self.assertEqual("bool", prepared.logical_dtype)
        self.assertEqual("numeric", prepared.validation_mode)

    def test_extrema_invoke_numpy_125_ufuncs(self) -> None:
        for operation in ("maximum", "minimum", "fmax", "fmin"):
            prepared = prepare_case(benchmark_case(operation, "binary"))
            expected = getattr(np, operation)(
                general_vector(8), binary_vector(8)
            )

            with self.subTest(operation=operation):
                np.testing.assert_array_equal(prepared.invoke(), expected)
                np.testing.assert_array_equal(
                    prepared.validation(), expected
                )
                self.assertEqual((8,), prepared.expected_shape)
                self.assertEqual("f64", prepared.logical_dtype)
                self.assertEqual("numeric", prepared.validation_mode)

    def test_logical_operations_invoke_numpy_125_ufuncs_and_return_bool(
        self,
    ) -> None:
        left = general_vector(8)
        right = binary_vector(8)
        for operation in (
            "logical_and", "logical_or", "logical_xor", "logical_not",
        ):
            prepared = prepare_case(benchmark_case(operation, "logical"))
            numpy_operation = getattr(np, operation)
            expected = (
                numpy_operation(left)
                if operation == "logical_not"
                else numpy_operation(left, right)
            )

            with self.subTest(operation=operation):
                np.testing.assert_array_equal(prepared.invoke(), expected)
                np.testing.assert_array_equal(
                    prepared.validation(), expected
                )
                self.assertEqual((8,), prepared.expected_shape)
                self.assertEqual("bool", prepared.logical_dtype)
                self.assertEqual("numeric", prepared.validation_mode)

    def test_array_predicates_use_floating_edges_and_return_bool(self) -> None:
        operand = predicate_vector(8)
        for operation in ("isnan", "isinf", "isfinite", "signbit"):
            prepared = prepare_case(benchmark_case(operation, "logical"))
            expected = getattr(np, operation)(operand)

            with self.subTest(operation=operation):
                np.testing.assert_array_equal(prepared.invoke(), expected)
                np.testing.assert_array_equal(prepared.validation(), expected)
                self.assertTrue(expected.any())
                self.assertTrue((~expected).any())
                self.assertEqual((8,), prepared.expected_shape)
                self.assertEqual("bool", prepared.logical_dtype)
                self.assertEqual("numeric", prepared.validation_mode)

    def test_object_kind_predicates_return_scalar_bool(self) -> None:
        operand = general_vector(8)
        expected_by_operation = {
            "iscomplexobj": np.iscomplexobj(operand),
            "isrealobj": np.isrealobj(operand),
            "isscalar": np.isscalar(operand),
        }
        for operation, expected in expected_by_operation.items():
            prepared = prepare_case(benchmark_case(operation, "logical"))

            with self.subTest(operation=operation):
                self.assertEqual(expected, prepared.invoke())
                self.assertEqual(expected, prepared.validation())
                self.assertEqual((), prepared.expected_shape)
                self.assertEqual("bool", prepared.logical_dtype)
                self.assertEqual("numeric", prepared.validation_mode)

    def test_array_metadata_bridge_properties_return_scalar_values(
        self,
    ) -> None:
        expected_by_operation = {
            "nbytes_cached": (8, "i64"),
            "c_contiguous_cached": (True, "bool"),
            "f_contiguous_cached": (True, "bool"),
        }
        for operation, (expected, logical_dtype) in (
            expected_by_operation.items()
        ):
            prepared = prepare_case(
                benchmark_case(operation, "bridge", size=1)
            )

            with self.subTest(operation=operation):
                self.assertEqual(expected, prepared.invoke())
                self.assertEqual(expected, prepared.validation())
                self.assertEqual((), prepared.expected_shape)
                self.assertEqual(logical_dtype, prepared.logical_dtype)
                self.assertEqual("numeric", prepared.validation_mode)

    def test_bitwise_operations_invoke_numpy_125_ufuncs_on_int64_inputs(
        self,
    ) -> None:
        indices = np.arange(8, dtype=np.int64)
        left = ((indices * 37 + 11) % 4096) - 2048
        right = ((indices * 53 + 19) % 4096) - 2048
        shifts = (indices * 5 + 3) % 8
        expected_by_operation = {
            "bitwise_and": np.bitwise_and(left, right),
            "bitwise_or": np.bitwise_or(left, right),
            "bitwise_xor": np.bitwise_xor(left, right),
            "invert": np.invert(left),
            "left_shift": np.left_shift(left, shifts),
            "right_shift": np.right_shift(left, shifts),
        }

        for operation, expected in expected_by_operation.items():
            prepared = prepare_case(
                benchmark_case(operation, "bitwise", dtype="i64")
            )
            with self.subTest(operation=operation):
                np.testing.assert_array_equal(prepared.invoke(), expected)
                np.testing.assert_array_equal(prepared.validation(), expected)
                self.assertEqual(np.dtype(np.int64), expected.dtype)
                self.assertEqual((8,), prepared.expected_shape)
                self.assertEqual("i64", prepared.logical_dtype)
                self.assertEqual("numeric", prepared.validation_mode)

    def test_every_catalog_operation_invokes_and_validates_at_small_size(self) -> None:
        cases: list[tuple[dict[str, object], tuple[int, ...], str, str]] = []
        scalar_operations = {
            "sum",
            "mean",
            "average",
            "std",
            "max",
            "min",
            "argmax",
            "prod",
            "allclose",
            "pipeline_separate",
            "pipeline_batch",
            "iscomplexobj",
            "isrealobj",
            "isscalar",
        }
        for category, operations in self.VECTOR_OPERATIONS.items():
            for operation in operations:
                expected_shape = () if operation in scalar_operations else (8,)
                if operation == "reshape":
                    expected_shape = (8, 1)
                elif operation == "atleast_2d":
                    expected_shape = (1, 8)
                elif operation == "atleast_3d":
                    expected_shape = (1, 8, 1)
                elif operation == "divmod":
                    expected_shape = (2, 8)
                elif operation in {"take", "compress"}:
                    expected_shape = (4,)
                elif operation == "unique_duplicates":
                    expected_shape = (8,)
                elif operation == "unique_nan":
                    expected_shape = (5,)
                elif operation == "intersect1d_duplicates":
                    expected_shape = (4,)
                elif operation == "union1d_duplicates":
                    expected_shape = (132,)
                elif operation == "setdiff1d_duplicates":
                    expected_shape = (4,)
                elif operation == "setxor1d_duplicates":
                    expected_shape = (128,)
                logical_dtype = (
                    "bool"
                    if operation == "equal" or category == "logical"
                    or operation in {"in1d_duplicates", "isin_duplicates"}
                    else "i64"
                    if category in {"bitwise", "integer"}
                    or operation in {"argmax", "allclose"}
                    or operation.startswith("argsort")
                    or operation.startswith("searchsorted")
                    or operation.startswith("digitize")
                    or operation == "lexsort"
                    else "f64"
                )
                mode = (
                    "shape"
                    if operation == "random"
                    else "numeric_nan"
                    if operation in {"sort_stable_nan", "unique_nan"}
                    else "numeric"
                )
                cases.append(
                    (
                        benchmark_case(
                            operation,
                            category,
                            dtype=(
                                "i64"
                                if category in {"bitwise", "integer"}
                                else "f64"
                            ),
                        ),
                        expected_shape,
                        logical_dtype,
                        mode,
                    )
                )

        for operation in (
            "matmul", "dot", "det", "inv", "norm", "solve", "cholesky",
            "einsum", "eig", "svd", "lstsq",
        ):
            expected_shape = {
                "det": (),
                "norm": (),
                "solve": (4,),
                "eig": (4,),
                "svd": (4,),
                "lstsq": (4,),
            }.get(operation, (4, 4))
            cases.append(
                (
                    benchmark_case(operation, "linalg", size=16, rows=4, cols=4),
                    expected_shape,
                    "f64",
                    "numeric",
                )
            )

        for operation in (
            "take_axis0_block", "take_axis0_strided",
            "compress_axis0_block", "compress_axis0_strided",
        ):
            cases.append(
                (
                    benchmark_case(
                        operation, "indexing",
                        size=16, rows=4, cols=4, axis=0,
                    ),
                    (2, 4),
                    "f64",
                    "numeric",
                )
            )

        for operation in ("softmax", "log_softmax"):
            cases.extend(
                (
                    (
                        benchmark_case(operation, "misc_axis"),
                        (8,),
                        "f64",
                        "numeric",
                    ),
                    (
                        benchmark_case(
                            f"{operation}_axis_last", "misc_axis",
                            size=16, rows=4, cols=4, axis=1,
                        ),
                        (4, 4),
                        "f64",
                        "numeric",
                    ),
                    (
                        benchmark_case(
                            f"{operation}_axis0_strided", "misc_axis",
                            size=16, rows=4, cols=4, axis=0,
                        ),
                        (4, 4),
                        "f64",
                        "numeric",
                    ),
                )
            )

        cases.extend(
            (
                (
                    benchmark_case("trapz", "misc_axis"),
                    (),
                    "f64",
                    "numeric",
                ),
                (
                    benchmark_case(
                        "trapz_axis_last", "misc_axis",
                        size=16, rows=4, cols=4, axis=1,
                    ),
                    (4,),
                    "f64",
                    "numeric",
                ),
                (
                    benchmark_case(
                        "trapz_axis0_strided", "misc_axis",
                        size=16, rows=4, cols=4, axis=0,
                    ),
                    (4,),
                    "f64",
                    "numeric",
                ),
                (
                    benchmark_case("packbits", "misc_axis", dtype="u8"),
                    (1,),
                    "u8",
                    "numeric",
                ),
                (
                    benchmark_case(
                        "packbits_axis_last", "misc_axis", dtype="u8",
                        size=16, rows=4, cols=4, axis=1,
                    ),
                    (4, 1),
                    "u8",
                    "numeric",
                ),
                (
                    benchmark_case(
                        "packbits_axis0_strided", "misc_axis", dtype="u8",
                        size=16, rows=4, cols=4, axis=0,
                    ),
                    (1, 4),
                    "u8",
                    "numeric",
                ),
                (
                    benchmark_case("unpackbits", "misc_axis", dtype="u8"),
                    (64,),
                    "u8",
                    "numeric",
                ),
                (
                    benchmark_case(
                        "unpackbits_axis_last", "misc_axis", dtype="u8",
                        size=16, rows=4, cols=4, axis=1,
                    ),
                    (4, 32),
                    "u8",
                    "numeric",
                ),
                (
                    benchmark_case(
                        "unpackbits_axis0_strided", "misc_axis", dtype="u8",
                        size=16, rows=4, cols=4, axis=0,
                    ),
                    (32, 4),
                    "u8",
                    "numeric",
                ),
            )
        )

        cases.extend(
            (
                (
                    benchmark_case(
                        "sum_axis_last", "reduction",
                        size=16, rows=4, cols=4, axis=1,
                    ),
                    (4,),
                    "f64",
                    "numeric",
                ),
                (
                    benchmark_case(
                        "cumsum_axis_last", "reduction",
                        size=16, rows=4, cols=4, axis=1,
                    ),
                    (4, 4),
                    "f64",
                    "numeric",
                ),
                (
                    benchmark_case("transpose_copy", "shape", size=16, rows=4, cols=4),
                    (4, 4),
                    "f64",
                    "numeric",
                ),
                (
                    benchmark_case(
                        "concatenate", "shape", size=16, rows=4, cols=4, axis=0
                    ),
                    (8, 4),
                    "f64",
                    "numeric",
                ),
                (benchmark_case("fft", "fft"), (8, 2), "f64", "numeric"),
                (
                    benchmark_case("property_call", "bridge", size=1),
                    (),
                    "i64",
                    "numeric",
                ),
                (
                    benchmark_case("property_cached", "bridge", size=1),
                    (),
                    "i64",
                    "numeric",
                ),
                (
                    benchmark_case("nbytes_cached", "bridge", size=1),
                    (),
                    "i64",
                    "numeric",
                ),
                (
                    benchmark_case(
                        "c_contiguous_cached", "bridge", size=1
                    ),
                    (),
                    "bool",
                    "numeric",
                ),
                (
                    benchmark_case(
                        "f_contiguous_cached", "bridge", size=1
                    ),
                    (),
                    "bool",
                    "numeric",
                ),
                (
                    benchmark_case("static_add_call", "bridge", size=1),
                    (),
                    "i64",
                    "numeric",
                ),
            )
        )

        with CATALOG_PATH.open("r", encoding="utf-8") as source:
            small_catalog = json.load(source)
        small_catalog["size_sets"] = {"vector": [8], "matrix": [4], "fft": [8]}
        for template in small_catalog["templates"]:
            profile_size = [1] if template["shape_kind"] == "bridge" else template["shape_kind"]
            template["profiles"] = {"focus": profile_size}
        expanded = expand_cases(small_catalog, "focus")
        expected_by_operation = {
            case["operation"]: (expected_shape, logical_dtype, mode)
            for case, expected_shape, logical_dtype, mode in cases
        }
        self.assertEqual(
            {case["operation"] for case in expanded},
            set(expected_by_operation),
        )

        wire = io.StringIO()
        write_jobs_tsv(expanded, wire)
        wire.seek(0)
        wire_cases = read_jobs_tsv(wire)

        for case in wire_cases:
            with self.subTest(operation=case["operation"]):
                expected_shape, logical_dtype, mode = expected_by_operation[case["operation"]]
                prepared = prepare_case(case)
                result = prepared.invoke()
                del result
                validation_value = prepared.validation()
                signature = validation_signature(
                    validation_value,
                    prepared.expected_shape,
                    logical_dtype=prepared.logical_dtype,
                    mode=prepared.validation_mode,
                    expected_finite_members=prepared.expected_finite_members,
                )
                self.assertEqual(tuple(signature["shape"]), expected_shape)
                self.assertEqual(signature["logical_dtype"], logical_dtype)
                self.assertEqual(signature["mode"], mode)
                if mode in {"numeric", "numeric_nan"}:
                    self.assertTrue(np.isfinite(signature["sum"]))
                    self.assertTrue(np.all(np.isfinite(signature["values"])))
                if mode == "numeric_nan":
                    self.assertGreater(signature["nan_count"], 0)
                    self.assertTrue(signature["trailing_nan"])
                    self.assertIs(signature["finite_members_exact"], True)
                del validation_value
                del prepared

    def test_task9_linalg_workers_validate_complete_deterministic_results(self) -> None:
        for operation in ("einsum", "eig", "svd", "solve", "lstsq"):
            with self.subTest(operation=operation):
                prepared = prepare_case(
                    benchmark_case(
                        operation,
                        "linalg",
                        size=16,
                        rows=4,
                        cols=4,
                    )
                )
                result = prepared.invoke()
                validation = prepared.validation()
                self.assertIsNotNone(result)
                self.assertIsNotNone(validation)
                self.assertEqual("numeric", prepared.validation_mode)

    def test_task9_linalg_full_validators_reject_unsampled_counterfeits(
        self,
    ) -> None:
        size = 4

        einsum_source = bench_numpy.task9_einsum_matrix(size)
        einsum_result = einsum_source @ einsum_source
        einsum_result[1, 2] += 1.0
        with self.assertRaisesRegex(ValueError, "complete result"):
            bench_numpy.validate_task9_einsum_output(
                einsum_result, einsum_source, einsum_source.copy()
            )

        matrix = bench_numpy._matrix(size, size)
        expected_solution = binary_vector(size)
        rhs = matrix @ expected_solution
        counterfeit_solution = expected_solution.copy()
        counterfeit_solution[1] += 1.0
        with self.assertRaisesRegex(ValueError, "complete result"):
            bench_numpy.validate_task9_solve_output(
                counterfeit_solution,
                matrix,
                rhs,
                matrix.copy(),
                rhs.copy(),
                expected_solution,
            )

        eig_source = bench_numpy.task9_eig_matrix(size)
        eig_values, eig_vectors = np.linalg.eig(eig_source)
        eig_vectors[1, 2] += 1.0
        with self.assertRaisesRegex(ValueError, "eig decomposition"):
            bench_numpy.validate_task9_eig_output(
                (eig_values, eig_vectors), eig_source, eig_source.copy()
            )

        svd_source = matrix.copy()
        left, singular, right = np.linalg.svd(
            svd_source, full_matrices=False
        )
        left[1, 2] += 1.0
        with self.assertRaisesRegex(ValueError, "SVD reconstruction"):
            bench_numpy.validate_task9_svd_output(
                (left, singular, right), svd_source, svd_source.copy()
            )

        lstsq_matrix = bench_numpy.task9_lstsq_matrix(size)
        lstsq_rhs = lstsq_matrix @ expected_solution
        lstsq_result = list(
            np.linalg.lstsq(lstsq_matrix, lstsq_rhs, rcond=None)
        )
        lstsq_result[3] = np.asarray(lstsq_result[3]).copy()
        lstsq_result[3][1] += 1.0
        with self.assertRaisesRegex(ValueError, "singular values"):
            bench_numpy.validate_task9_lstsq_output(
                tuple(lstsq_result),
                lstsq_matrix,
                lstsq_rhs,
                lstsq_matrix.copy(),
                lstsq_rhs.copy(),
                expected_solution,
            )

    def test_cholesky_invokes_numpy_125_with_deterministic_spd_input(
        self,
    ) -> None:
        rows = cols = 4
        matrix = (general_vector(rows * cols) * 0.001).reshape(
            (rows, cols)
        )
        matrix[np.diag_indices(rows)] += 2.0
        expected = np.linalg.cholesky(matrix)
        prepared = prepare_case(
            benchmark_case(
                "cholesky", "linalg",
                size=rows * cols, rows=rows, cols=cols,
            )
        )

        actual = prepared.invoke()
        validated = prepared.validation()

        self.assertEqual((rows, cols), prepared.expected_shape)
        self.assertEqual("f64", prepared.logical_dtype)
        np.testing.assert_array_equal(expected, actual)
        np.testing.assert_array_equal(expected, validated)

    def test_fft_validation_normalizes_complex_output_without_changing_timed_invoke(self) -> None:
        prepared = prepare_case(benchmark_case("fft", "fft"))

        timed_result = prepared.invoke()
        normalized = prepared.validation()

        self.assertEqual(timed_result.shape, (8,))
        self.assertTrue(np.iscomplexobj(timed_result))
        self.assertEqual(normalized.shape, (8, 2))
        np.testing.assert_allclose(normalized[:, 0], timed_result.real)
        np.testing.assert_allclose(normalized[:, 1], timed_result.imag)

    def test_rejects_unknown_operation(self) -> None:
        with self.assertRaisesRegex(ValueError, "operation.*unknown"):
            prepare_case(benchmark_case("not-real", "unknown"))

    def test_prepare_and_run_worker_share_semantic_job_validation(self) -> None:
        from benchmark.bench_numpy import run_worker

        valid = benchmark_case("zeros", "creation")
        invalid_cases = (
            ({**valid, "category": "unary"}, "category"),
            ({**valid, "id": "wrong/f64/8"}, "id"),
            ({**valid, "axis": 0}, "axis"),
        )
        boundaries = (
            prepare_case,
            lambda case: run_worker(
                [case],
                warmups=0,
                sample_count=1,
                target_sample_ns=1,
            ),
        )
        for boundary in boundaries:
            for case, field in invalid_cases:
                with self.subTest(boundary=boundary, field=field):
                    with self.assertRaisesRegex(
                        ValueError,
                        rf"{re.escape(case['id'])}.*{field}",
                    ):
                        boundary(case)

    def test_prod_uses_the_complete_safe_input_without_a_size_cap(self) -> None:
        size = 1_001
        indices = np.arange(size, dtype=np.int64)
        safe_input = 1.0 + ((((indices * 37 + 11) % 1009) - 504) * 1e-9)
        expected = np.prod(safe_input)

        prepared = prepare_case(benchmark_case("prod", "reduction", size=size))

        self.assertEqual(prepared.invoke(), expected)

    def test_last_axis_matrix_reductions_preserve_result_shapes(self) -> None:
        matrix = (general_vector(16) * 0.001).reshape((4, 4))
        matrix[np.diag_indices(4)] += 2.0
        for operation, expected in (
            ("sum_axis_last", np.sum(matrix, axis=1)),
            ("cumsum_axis_last", np.cumsum(matrix, axis=1)),
        ):
            with self.subTest(operation=operation):
                prepared = prepare_case(benchmark_case(
                    operation, "reduction", size=16,
                    rows=4, cols=4, axis=1,
                ))
                np.testing.assert_array_equal(prepared.invoke(), expected)
                np.testing.assert_array_equal(prepared.validation(), expected)
                self.assertEqual(
                    "numeric_nan" if operation == "sort_stable_nan" else "numeric",
                    prepared.validation_mode,
                )

    def test_softmax_workloads_validate_every_output_value_and_strided_source(
        self,
    ) -> None:
        matrix = (general_vector(16) * 0.001).reshape((4, 4))
        matrix[np.diag_indices(4)] += 2.0
        cases = (
            (
                benchmark_case("softmax", "misc_axis"),
                general_vector(8),
                -1,
            ),
            (
                benchmark_case(
                    "softmax_axis_last", "misc_axis",
                    size=16, rows=4, cols=4, axis=1,
                ),
                matrix,
                1,
            ),
            (
                benchmark_case(
                    "softmax_axis0_strided", "misc_axis",
                    size=16, rows=4, cols=4, axis=0,
                ),
                matrix.T,
                0,
            ),
        )
        for case, source, axis in cases:
            with self.subTest(operation=case["operation"]):
                prepared = prepare_case(case)
                expected = scipy_special.softmax(source, axis=axis)

                actual = prepared.invoke()
                validated = prepared.validation()

                np.testing.assert_array_equal(expected, actual)
                np.testing.assert_array_equal(expected, validated)
                self.assertEqual(source.shape, prepared.expected_shape)
                self.assertIs(
                    actual,
                    bench_numpy.validate_softmax_output(actual, source, axis),
                )

                corrupted = expected.copy()
                corrupted.reshape(-1)[1] += 1e-6
                with self.assertRaisesRegex(ValueError, "softmax full-output"):
                    bench_numpy.validate_softmax_output(
                        corrupted, source, axis
                    )

        strided = bench_numpy.misc_axis_matrix(4, 4, strided=True)
        self.assertFalse(strided.flags.c_contiguous)
        self.assertTrue(strided.flags.f_contiguous)

    def test_log_softmax_workloads_validate_every_output_value(
        self,
    ) -> None:
        matrix = (general_vector(16) * 0.001).reshape((4, 4))
        matrix[np.diag_indices(4)] += 2.0
        cases = (
            (
                benchmark_case("log_softmax", "misc_axis"),
                general_vector(8),
                -1,
            ),
            (
                benchmark_case(
                    "log_softmax_axis_last", "misc_axis",
                    size=16, rows=4, cols=4, axis=1,
                ),
                matrix,
                1,
            ),
            (
                benchmark_case(
                    "log_softmax_axis0_strided", "misc_axis",
                    size=16, rows=4, cols=4, axis=0,
                ),
                matrix.T,
                0,
            ),
        )
        for case, source, axis in cases:
            with self.subTest(operation=case["operation"]):
                prepared = prepare_case(case)
                expected = scipy_special.log_softmax(source, axis=axis)

                actual = prepared.invoke()
                validated = prepared.validation()

                np.testing.assert_array_equal(expected, actual)
                np.testing.assert_array_equal(expected, validated)
                self.assertIs(
                    actual,
                    bench_numpy.validate_log_softmax_output(
                        actual, source, axis
                    ),
                )
                corrupted = expected.copy()
                corrupted.reshape(-1)[1] += 1e-6
                with self.assertRaisesRegex(
                    ValueError, "log_softmax full-output"
                ):
                    bench_numpy.validate_log_softmax_output(
                        corrupted, source, axis
                    )

    def test_trapz_workloads_validate_every_output_value(self) -> None:
        matrix = (general_vector(16) * 0.001).reshape((4, 4))
        matrix[np.diag_indices(4)] += 2.0
        cases = (
            (
                benchmark_case("trapz", "misc_axis"),
                general_vector(8),
                -1,
            ),
            (
                benchmark_case(
                    "trapz_axis_last", "misc_axis",
                    size=16, rows=4, cols=4, axis=1,
                ),
                matrix,
                1,
            ),
            (
                benchmark_case(
                    "trapz_axis0_strided", "misc_axis",
                    size=16, rows=4, cols=4, axis=0,
                ),
                matrix.T,
                0,
            ),
        )
        for case, source, axis in cases:
            with self.subTest(operation=case["operation"]):
                prepared = prepare_case(case)
                expected = np.trapz(source, dx=0.25, axis=axis)

                actual = prepared.invoke()
                validated = prepared.validation()

                np.testing.assert_array_equal(expected, actual)
                np.testing.assert_array_equal(expected, validated)
                self.assertIs(
                    actual,
                    bench_numpy.validate_trapz_output(
                        actual, source, axis, 0.25
                    ),
                )
                if np.asarray(expected).size > 1:
                    corrupted = np.asarray(expected).copy()
                    corrupted.reshape(-1)[1] += 1e-6
                    with self.assertRaisesRegex(
                        ValueError, "trapz full-output"
                    ):
                        bench_numpy.validate_trapz_output(
                            corrupted, source, axis, 0.25
                        )

    def test_packbits_workloads_validate_every_output_byte(self) -> None:
        vector = (((np.arange(8) * 5 + 3) % 2) != 0).astype(np.uint8)
        matrix = (((np.arange(16) * 5 + 3) % 2) != 0).astype(
            np.uint8
        ).reshape((4, 4))
        cases = (
            (
                benchmark_case(
                    "packbits", "misc_axis", dtype="u8"
                ),
                vector,
                -1,
            ),
            (
                benchmark_case(
                    "packbits_axis_last", "misc_axis", dtype="u8",
                    size=16, rows=4, cols=4, axis=1,
                ),
                matrix,
                1,
            ),
            (
                benchmark_case(
                    "packbits_axis0_strided", "misc_axis", dtype="u8",
                    size=16, rows=4, cols=4, axis=0,
                ),
                matrix.T,
                0,
            ),
        )
        for case, source, axis in cases:
            with self.subTest(operation=case["operation"]):
                prepared = prepare_case(case)
                expected = np.packbits(source, axis=axis, bitorder="big")

                actual = prepared.invoke()
                validated = prepared.validation()

                np.testing.assert_array_equal(expected, actual)
                np.testing.assert_array_equal(expected, validated)
                self.assertEqual("u8", prepared.logical_dtype)
                self.assertIs(
                    actual,
                    bench_numpy.validate_packbits_output(
                        actual, source, axis
                    ),
                )
                if expected.size > 1:
                    corrupted = expected.copy()
                    corrupted.reshape(-1)[1] ^= np.uint8(1)
                    with self.assertRaisesRegex(
                        ValueError, "packbits full-output"
                    ):
                        bench_numpy.validate_packbits_output(
                            corrupted, source, axis
                        )

    def test_unpackbits_workloads_validate_every_output_byte(self) -> None:
        vector = ((np.arange(8) * 73 + 19) % 256).astype(np.uint8)
        matrix = ((np.arange(16) * 73 + 19) % 256).astype(
            np.uint8
        ).reshape((4, 4))
        cases = (
            (
                benchmark_case(
                    "unpackbits", "misc_axis", dtype="u8"
                ),
                vector,
                -1,
            ),
            (
                benchmark_case(
                    "unpackbits_axis_last", "misc_axis", dtype="u8",
                    size=16, rows=4, cols=4, axis=1,
                ),
                matrix,
                1,
            ),
            (
                benchmark_case(
                    "unpackbits_axis0_strided", "misc_axis", dtype="u8",
                    size=16, rows=4, cols=4, axis=0,
                ),
                matrix.T,
                0,
            ),
        )
        for case, source, axis in cases:
            with self.subTest(operation=case["operation"]):
                prepared = prepare_case(case)
                expected = np.unpackbits(
                    source, axis=axis, count=None, bitorder="big"
                )

                actual = prepared.invoke()
                validated = prepared.validation()

                np.testing.assert_array_equal(expected, actual)
                np.testing.assert_array_equal(expected, validated)
                self.assertEqual("u8", prepared.logical_dtype)
                self.assertIs(
                    actual,
                    bench_numpy.validate_unpackbits_output(
                        actual, source, axis
                    ),
                )
                corrupted = expected.copy()
                corrupted.reshape(-1)[1] ^= np.uint8(1)
                with self.assertRaisesRegex(
                    ValueError, "unpackbits full-output"
                ):
                    bench_numpy.validate_unpackbits_output(
                        corrupted, source, axis
                    )

    def test_transpose_copy_returns_transposed_values_in_c_order(self) -> None:
        matrix = (general_vector(16) * 0.001).reshape((4, 4))
        matrix[np.diag_indices(4)] += 2.0
        prepared = prepare_case(
            benchmark_case("transpose_copy", "shape", size=16, rows=4, cols=4)
        )

        result = prepared.invoke()

        np.testing.assert_array_equal(result, matrix.T)
        self.assertTrue(result.flags.c_contiguous)

    def test_random_validation_does_not_advance_the_timed_rng(self) -> None:
        first = prepare_case(benchmark_case("random", "creation"))
        second = prepare_case(benchmark_case("random", "creation"))

        validation_result = first.validation()
        del validation_result
        first_timed_result = first.invoke()
        second_timed_result = second.invoke()

        np.testing.assert_array_equal(first_timed_result, second_timed_result)

    def test_random_operation_honors_the_requested_seed(self) -> None:
        first = prepare_case(benchmark_case("random", "creation"), seed=7)
        second = prepare_case(benchmark_case("random", "creation"), seed=8)

        first_result = first.invoke()
        second_result = second.invoke()

        self.assertFalse(np.array_equal(first_result, second_result))

    def test_weighted_choice_uses_full_deterministic_validation(self) -> None:
        case = benchmark_case("choice_weighted", "random")
        first = prepare_case(case, seed=20260803)
        second = prepare_case(case, seed=20260803)

        validation_result = first.validation()
        np.testing.assert_array_equal(
            validation_result,
            np.full(8, 193.0, dtype=np.float64),
        )
        np.testing.assert_array_equal(first.invoke(), second.invoke())

        corrupted = validation_result.copy()
        corrupted[1] = 192.0
        with self.assertRaisesRegex(ValueError, "weighted choice full-output"):
            bench_numpy.validate_weighted_choice_output(corrupted, 8)

    def test_integer_result_operations_declare_i64(self) -> None:
        cases = (
            benchmark_case("property_call", "bridge", size=1),
            benchmark_case("property_cached", "bridge", size=1),
            benchmark_case("nbytes_cached", "bridge", size=1),
            benchmark_case("static_add_call", "bridge", size=1),
            benchmark_case("argmax", "reduction"),
            benchmark_case("argsort", "sorting"),
            benchmark_case("argsort_stable_nan", "sorting"),
        )
        for case in cases:
            with self.subTest(operation=case["operation"]):
                prepared = prepare_case(case)
                self.assertEqual(prepared.logical_dtype, "i64")
                validation_signature(
                    prepared.validation(),
                    prepared.expected_shape,
                    logical_dtype=prepared.logical_dtype,
                )

    def test_argsort_semantic_validation_rejects_duplicates_and_wrong_order(self) -> None:
        operand = sorting_vector(8)
        valid = np.argsort(operand)

        self.assertIs(validate_argsort_indices(valid, operand), valid)

        duplicate = valid.copy()
        duplicate[-1] = duplicate[-2]
        with self.assertRaisesRegex(ValueError, "permutation"):
            validate_argsort_indices(duplicate, operand)

        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            validate_argsort_indices(np.arange(8, dtype=np.int64), operand)

        nan_operand = sorting_vector(8, "nan")
        valid_nan = np.argsort(nan_operand, kind="stable")
        self.assertIs(validate_argsort_indices(valid_nan, nan_operand), valid_nan)
        with self.assertRaisesRegex(ValueError, "NaN|nondecreasing"):
            validate_argsort_indices(
                np.array([0, 3, 7, 1, 5, 2, 4, 6], dtype=np.int64),
                nan_operand,
            )

    def test_sorting_cases_use_requested_kind_and_nan_pattern(self) -> None:
        size = 5_000
        cases = (
            ("sort", "sort", "quicksort", "duplicates"),
            ("argsort", "argsort", "quicksort", "duplicates"),
            ("sort_mergesort", "sort", "mergesort", "duplicates"),
            ("argsort_mergesort", "argsort", "mergesort", "duplicates"),
            ("sort_heapsort", "sort", "heapsort", "duplicates"),
            ("argsort_heapsort", "argsort", "heapsort", "duplicates"),
            ("sort_stable", "sort", "stable", "duplicates"),
            ("argsort_stable", "argsort", "stable", "duplicates"),
            ("sort_stable_nan", "sort", "stable", "nan"),
            ("argsort_stable_nan", "argsort", "stable", "nan"),
        )

        for operation, family, kind, pattern in cases:
            with self.subTest(operation=operation):
                operand = sorting_vector(size, pattern)
                prepared = prepare_case(
                    benchmark_case(operation, "sorting", size=size)
                )
                expected = (
                    np.sort(operand, kind=kind)
                    if family == "sort"
                    else np.argsort(operand, kind=kind)
                )
                np.testing.assert_array_equal(prepared.invoke(), expected)

    def test_task7_set_cases_use_duplicate_and_nan_heavy_inputs(self) -> None:
        if not hasattr(bench_numpy, "set_vector"):
            self.fail("set_vector public API is missing")
        set_vector = bench_numpy.set_vector
        size = 10_000
        left = set_vector(size, "duplicates")
        nan_left = set_vector(size, "nan")
        right = np.concatenate(
            (
                np.arange(0, 128, 2, dtype=np.float64),
                np.arange(256, 320, dtype=np.float64),
            )
        )
        expected_by_operation = {
            "unique_duplicates": np.unique(left),
            "unique_nan": np.unique(nan_left, equal_nan=True),
            "intersect1d_duplicates": np.intersect1d(left, right),
            "union1d_duplicates": np.union1d(left, right),
            "setdiff1d_duplicates": np.setdiff1d(left, right),
            "setxor1d_duplicates": np.setxor1d(left, right),
            "in1d_duplicates": np.in1d(left, right),
            "isin_duplicates": np.isin(left, right),
        }
        for operation, expected in expected_by_operation.items():
            with self.subTest(operation=operation):
                prepared = prepare_case(
                    benchmark_case(operation, "set", size=size)
                )
                np.testing.assert_array_equal(
                    prepared.invoke(), expected, strict=True
                )
                np.testing.assert_array_equal(
                    prepared.validation(),
                    expected,
                    strict=True,
                )
                self.assertEqual(
                    "numeric_nan" if operation == "unique_nan" else "numeric",
                    prepared.validation_mode,
                )

    def test_partition_validators_reject_invalid_values_and_indices(
        self,
    ) -> None:
        if not hasattr(bench_numpy, "validate_partition_values"):
            self.fail("validate_partition_values public API is missing")
        if not hasattr(bench_numpy, "validate_argpartition_indices"):
            self.fail("validate_argpartition_indices public API is missing")
        validate_partition_values = bench_numpy.validate_partition_values
        validate_argpartition_indices = (
            bench_numpy.validate_argpartition_indices
        )
        operand = sorting_vector(8)
        kth = operand.size // 2
        valid_values = np.partition(operand, kth)
        normalized = validate_partition_values(
            valid_values, operand, kth
        )
        np.testing.assert_array_equal(normalized, np.sort(operand))

        invalid_values = valid_values.copy()
        invalid_values[0], invalid_values[-1] = (
            invalid_values[-1],
            invalid_values[0],
        )
        with self.assertRaisesRegex(ValueError, "partition"):
            validate_partition_values(invalid_values, operand, kth)

        valid_indices = np.argpartition(operand, kth)
        normalized_indices = validate_argpartition_indices(
            valid_indices, operand, kth
        )
        np.testing.assert_array_equal(normalized_indices, np.sort(operand))

        duplicate = valid_indices.copy()
        duplicate[-1] = duplicate[-2]
        with self.assertRaisesRegex(ValueError, "permutation"):
            validate_argpartition_indices(duplicate, operand, kth)
        with self.assertRaisesRegex(ValueError, "partition"):
            validate_argpartition_indices(
                np.arange(operand.size, dtype=np.int64), operand, kth
            )

    def test_partition_cases_use_midpoint_kth_and_normalized_validation(
        self,
    ) -> None:
        if not hasattr(bench_numpy, "validate_partition_values"):
            self.fail("validate_partition_values public API is missing")
        cases = (
            ("partition", "partition", "duplicates"),
            ("argpartition", "argpartition", "duplicates"),
            ("partition_nan", "partition", "nan"),
            ("argpartition_nan", "argpartition", "nan"),
        )
        size = 5_000
        kth = size // 2
        for operation, family, pattern in cases:
            with self.subTest(operation=operation):
                operand = sorting_vector(size, pattern)
                prepared = prepare_case(
                    benchmark_case(operation, "sorting", size=size)
                )
                expected = (
                    np.partition(operand, kth)
                    if family == "partition"
                    else np.argpartition(operand, kth)
                )
                np.testing.assert_array_equal(prepared.invoke(), expected)
                normalized = prepared.validation()
                np.testing.assert_array_equal(
                    normalized,
                    np.nan_to_num(np.sort(operand), nan=8192.0),
                )
                self.assertEqual("f64", prepared.logical_dtype)


class TimingProtocolTests(unittest.TestCase):
    def test_calibrates_and_collects_three_positive_samples(self) -> None:
        calls = 0

        def callback() -> np.ndarray:
            nonlocal calls
            calls += 1
            return np.arange(8, dtype=np.float64)

        inner_loops, samples = time_operation(
            callback,
            warmups=2,
            sample_count=3,
            target_sample_ns=100_000,
        )

        self.assertGreater(inner_loops, 0)
        self.assertEqual(inner_loops & (inner_loops - 1), 0)
        self.assertEqual(len(samples), 3)
        self.assertTrue(all(math.isfinite(sample) and sample > 0 for sample in samples))
        self.assertGreater(calls, 2 + inner_loops * 3)

    def test_rejects_invalid_timing_arguments(self) -> None:
        invalid_arguments = (
            (-1, 3, 1),
            (True, 3, 1),
            (1.0, 3, 1),
            (0, 0, 1),
            (0, 2, 1),
            (0, True, 1),
            (0, 3, 0),
            (0, 3, -1),
            (0, 3, math.inf),
            (0, 3, math.nan),
            (0, 3, True),
        )
        for warmups, samples, target in invalid_arguments:
            with self.subTest(warmups=warmups, samples=samples, target=target):
                with self.assertRaises(ValueError):
                    time_operation(lambda: 1, warmups, samples, target)


class WorkerCliTests(unittest.TestCase):
    def test_cholesky_worker_document_matches_report_shape_contract(
        self,
    ) -> None:
        from benchmark.bench_numpy import main

        case = benchmark_case(
            "cholesky", "linalg", size=16, rows=4, cols=4
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            jobs_path = Path(temporary_directory) / "jobs.tsv"
            output_path = Path(temporary_directory) / "numpy.json"
            with jobs_path.open("w", encoding="utf-8", newline="") as jobs:
                write_jobs_tsv([case], jobs)

            exit_code = main(
                [
                    "--jobs", str(jobs_path),
                    "--output", str(output_path),
                    "--warmups", "0",
                    "--samples", "3",
                    "--target-sample-ms", "0.001",
                    "--seed", "7",
                ]
            )
            self.assertEqual(0, exit_code)
            with output_path.open("r", encoding="utf-8") as source:
                document = json.load(source)

        validate_worker_result(
            document,
            expected_runtime="numpy",
            expected_case_ids=[case["id"]],
        )
        self.assertEqual([4, 4], document["cases"][0]["validation"]["shape"])

    def test_tiny_fixture_produces_a_valid_numpy_worker_document(self) -> None:
        from benchmark.bench_numpy import main

        expected_jobs = read_jobs_tsv(FIXTURE_PATH)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "numpy.json"

            exit_code = main(
                [
                    "--jobs",
                    str(FIXTURE_PATH),
                    "--output",
                    str(output_path),
                    "--warmups",
                    "0",
                    "--samples",
                    "3",
                    "--target-sample-ms",
                    "0.001",
                    "--seed",
                    "7",
                ]
            )

            self.assertEqual(exit_code, 0)
            with output_path.open("r", encoding="utf-8") as source:
                document = json.load(source)

        validated = validate_worker_result(
            document,
            expected_runtime="numpy",
            expected_case_ids=[job["id"] for job in expected_jobs],
        )
        self.assertIs(validated, document)
        self.assertEqual(
            set(document["metadata"]),
            {
                "python_version",
                "numpy_version",
                "scipy_version",
                "timer",
                "timer_resolution_ns",
                "warmups",
                "sample_count",
                "target_sample_ns",
                "seed",
            },
        )
        self.assertEqual(document["metadata"]["timer"], "perf_counter_ns")
        self.assertEqual(document["metadata"]["scipy_version"], "1.12.0")
        self.assertEqual(document["metadata"]["warmups"], 0)
        self.assertEqual(document["metadata"]["sample_count"], 3)
        self.assertEqual(document["metadata"]["seed"], 7)
        for expected_job, result_case in zip(expected_jobs, document["cases"], strict=True):
            self.assertEqual(
                {field: result_case[field] for field in expected_job},
                expected_job,
            )

    def test_cli_requires_paths_and_rejects_invalid_timing_values(self) -> None:
        from benchmark.bench_numpy import main

        invalid_argument_lists = (
            ([], "--jobs", "required"),
            (
                ["--jobs", str(FIXTURE_PATH), "--output", "unused.json", "--warmups", "-1"],
                "--warmups",
                "non-negative integer",
            ),
            (
                ["--jobs", str(FIXTURE_PATH), "--output", "unused.json", "--samples", "2"],
                "--samples",
                "positive odd integer",
            ),
            (
                [
                    "--jobs",
                    str(FIXTURE_PATH),
                    "--output",
                    "unused.json",
                    "--target-sample-ms",
                    "inf",
                ],
                "--target-sample-ms",
                "finite positive number",
            ),
            (
                ["--jobs", str(FIXTURE_PATH), "--output", "unused.json", "--seed", "2147483648"],
                "--seed",
                "[0, 2147483647]",
            ),
        )
        for arguments, expected_flag, expected_reason in invalid_argument_lists:
            with self.subTest(arguments=arguments):
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    with self.assertRaises(SystemExit):
                        main(arguments)
                message = stderr.getvalue()
                self.assertIn(expected_flag, message)
                self.assertIn(expected_reason, message)


class AtomicJsonWriterTests(unittest.TestCase):
    def test_failed_serialization_preserves_destination_and_removes_temp_file(self) -> None:
        from benchmark.bench_numpy import write_json_atomic

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            destination = directory / "result.json"
            destination.write_bytes(b"original")

            with self.assertRaises(TypeError):
                write_json_atomic(destination, {"bad": object()})

            self.assertEqual(destination.read_bytes(), b"original")
            self.assertEqual(list(directory.iterdir()), [destination])

    def test_successful_write_produces_loadable_json(self) -> None:
        from benchmark.bench_numpy import write_json_atomic

        document = {"schema_version": 1, "cases": ["ok"]}
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "result.json"

            write_json_atomic(destination, document)

            with destination.open("r", encoding="utf-8") as source:
                self.assertEqual(json.load(source), document)

if __name__ == "__main__":
    unittest.main()
