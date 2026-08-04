from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from benchmark.report import (
    expand_cases,
    load_catalog,
    semantic_qualification_declarations,
    write_jobs_tsv,
)


CATALOG_PATH = Path(__file__).resolve().parents[1] / "cases.json"
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
CASE_FIELDS = {
    "id",
    "category",
    "operation",
    "dtype",
    "size",
    "rows",
    "cols",
    "axis",
}
OPERATION_FAMILIES = {
    "creation": {"zeros", "ones", "arange", "random", "linspace"},
    "unary": {
        "sin", "cos", "exp", "expm1", "sqrt", "log", "log2", "log10", "log1p", "absolute",
        "floor", "tanh", "angle", "real", "imag", "real_if_close",
    },
    "binary": {
        "add", "subtract", "multiply", "divide", "divmod", "power", "float_power", "heaviside", "maximum",
        "minimum", "fmax", "fmin",
        "logaddexp", "logaddexp2",
    },
    "logical": {
        "logical_and", "logical_or", "logical_xor", "logical_not",
        "isnan", "isinf", "isfinite", "signbit",
        "iscomplexobj", "isrealobj", "isscalar",
    },
    "bitwise": {
        "bitwise_and", "bitwise_or", "bitwise_xor", "invert",
        "left_shift", "right_shift",
    },
    "integer": {"gcd", "lcm"},
    "signal": {"convolve", "correlate"},
    "comparison": {"allclose", "equal"},
    "random": {"choice_weighted"},
    "reduction": {
        "sum", "mean", "average", "std", "max", "min", "argmax", "cumsum", "prod",
        "sum_axis_last", "cumsum_axis_last",
    },
    "misc_axis": {
        "softmax", "softmax_axis_last", "softmax_axis0_strided",
        "log_softmax", "log_softmax_axis_last", "log_softmax_axis0_strided",
        "trapz", "trapz_axis_last", "trapz_axis0_strided",
        "packbits", "packbits_axis_last", "packbits_axis0_strided",
        "unpackbits", "unpackbits_axis_last", "unpackbits_axis0_strided",
    },
    "linalg": {
        "matmul", "dot", "det", "inv", "norm", "solve", "cholesky",
        "einsum", "eig", "svd", "lstsq",
    },
    "sorting": {
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
    },
    "set": {
        "unique_duplicates", "unique_nan",
        "intersect1d_duplicates", "union1d_duplicates",
        "setdiff1d_duplicates", "setxor1d_duplicates",
        "in1d_duplicates", "isin_duplicates",
    },
    "shape": {
        "copy", "reshape", "flatten",
        "atleast_1d", "atleast_2d", "atleast_3d",
        "transpose_copy", "concatenate",
    },
    "indexing": {
        "take", "compress",
        "take_axis0_block", "take_axis0_strided",
        "compress_axis0_block", "compress_axis0_strided",
    },
    "preallocated": {"add_into", "sqrt_into", "cumsum_into"},
    "pipeline": {"pipeline_separate", "pipeline_batch"},
    "fft": {"fft"},
    "bridge": {
        "property_call",
        "property_cached",
        "nbytes_cached",
        "c_contiguous_cached",
        "f_contiguous_cached",
        "static_add_call",
    },
}
OPERATION_CONTRACTS = {
    operation: (category, "vector", -1)
    for category in (
        "creation",
        "unary",
        "binary",
        "logical",
        "signal",
        "comparison",
        "random",
        "reduction",
        "sorting",
        "set",
        "indexing",
        "preallocated",
        "pipeline",
    )
    for operation in OPERATION_FAMILIES[category]
}
OPERATION_CONTRACTS.update(
    {
        operation: ("indexing", "matrix", 0)
        for operation in (
            "take_axis0_block",
            "take_axis0_strided",
            "compress_axis0_block",
            "compress_axis0_strided",
        )
    }
)
OPERATION_CONTRACTS.update(
    {
        operation: ("bitwise", "vector", -1)
        for operation in OPERATION_FAMILIES["bitwise"]
    }
)
OPERATION_CONTRACTS.update(
    {
        operation: ("integer", "vector", -1)
        for operation in OPERATION_FAMILIES["integer"]
    }
)
OPERATION_CONTRACTS.update(
    {
        operation: ("shape", "vector", -1)
        for operation in (
            "copy", "reshape", "flatten",
            "atleast_1d", "atleast_2d", "atleast_3d",
        )
    }
)
OPERATION_CONTRACTS.update(
    {
        operation: ("linalg", "matrix", -1)
        for operation in OPERATION_FAMILIES["linalg"]
    }
)
OPERATION_CONTRACTS.update(
    {
        "sum_axis_last": ("reduction", "matrix", 1),
        "cumsum_axis_last": ("reduction", "matrix", 1),
        "softmax": ("misc_axis", "vector", -1),
        "softmax_axis_last": ("misc_axis", "matrix", 1),
        "softmax_axis0_strided": ("misc_axis", "matrix", 0),
        "log_softmax": ("misc_axis", "vector", -1),
        "log_softmax_axis_last": ("misc_axis", "matrix", 1),
        "log_softmax_axis0_strided": ("misc_axis", "matrix", 0),
        "trapz": ("misc_axis", "vector", -1),
        "trapz_axis_last": ("misc_axis", "matrix", 1),
        "trapz_axis0_strided": ("misc_axis", "matrix", 0),
        "packbits": ("misc_axis", "vector", -1),
        "packbits_axis_last": ("misc_axis", "matrix", 1),
        "packbits_axis0_strided": ("misc_axis", "matrix", 0),
        "unpackbits": ("misc_axis", "vector", -1),
        "unpackbits_axis_last": ("misc_axis", "matrix", 1),
        "unpackbits_axis0_strided": ("misc_axis", "matrix", 0),
        "transpose_copy": ("shape", "matrix", -1),
        "concatenate": ("shape", "matrix", 0),
        "fft": ("fft", "fft", -1),
        **{
            operation: ("bridge", "bridge", -1)
            for operation in OPERATION_FAMILIES["bridge"]
        },
    }
)


def minimal_catalog() -> dict[str, object]:
    return {
        "schema_version": 1,
        "size_sets": {
            "vector": [10],
            "matrix": [4],
            "fft": [8],
        },
        "templates": [
            {
                "category": "creation",
                "operation": "zeros",
                "dtype": "f64",
                "shape_kind": "vector",
                "axis": -1,
                "profiles": {"focus": "vector"},
            }
        ],
    }


def catalog_for_operation(operation: str) -> dict[str, object]:
    document = minimal_catalog()
    category, shape_kind, axis = OPERATION_CONTRACTS[operation]
    template = document["templates"][0]
    template.update(
        {
            "category": category,
            "operation": operation,
            "dtype": (
                "u8" if operation.startswith(("packbits", "unpackbits"))
                else "i64" if category in {"bitwise", "integer"}
                else "f64"
            ),
            "shape_kind": shape_kind,
            "axis": axis,
            "profiles": {"focus": [1] if shape_kind == "bridge" else shape_kind},
        }
    )
    return document


class TemporaryCatalogMixin:
    def load_document(self, document: object) -> dict[str, object]:
        return self.load_text(json.dumps(document))

    def load_text(self, text: str) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "cases.json"
            path.write_text(text, encoding="utf-8")
            return load_catalog(path)


class SharedCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog(CATALOG_PATH)

    def test_has_exact_schema_size_sets_and_operation_families(self) -> None:
        self.assertEqual({"schema_version", "size_sets", "templates"}, set(self.catalog))
        self.assertEqual(1, self.catalog["schema_version"])
        self.assertEqual(
            {
                "vector": [1_000, 10_000, 100_000, 1_000_000],
                "matrix": [32, 128, 512],
                "fft": [1_024, 4_096, 16_384, 65_536],
            },
            self.catalog["size_sets"],
        )

        actual_families: dict[str, set[str]] = {}
        for template in self.catalog["templates"]:
            actual_families.setdefault(template["category"], set()).add(template["operation"])
        self.assertEqual(OPERATION_FAMILIES, actual_families)

    def test_shared_case_id_formatter_matches_expansion_for_every_operation(self) -> None:
        from benchmark.report import format_case_id

        for operation in OPERATION_CONTRACTS:
            with self.subTest(operation=operation):
                case = expand_cases(catalog_for_operation(operation), "focus")[0]
                self.assertEqual(
                    format_case_id(
                        operation,
                        case["dtype"],
                        size=case["size"],
                        rows=case["rows"],
                        cols=case["cols"],
                        axis=case["axis"],
                    ),
                    case["id"],
                )

    def test_weighted_choice_is_a_canonical_one_million_sample_case(self) -> None:
        for profile in ("focus", "standard", "full"):
            cases = [
                case
                for case in expand_cases(self.catalog, profile)
                if case["operation"] == "choice_weighted"
            ]
            with self.subTest(profile=profile):
                self.assertEqual(1, len(cases))
                self.assertEqual("choice_weighted/f64/1000000", cases[0]["id"])
                self.assertEqual("random", cases[0]["category"])
                self.assertEqual(1_000_000, cases[0]["size"])

    def test_task10_choice_qualification_has_exact_direct_owners(self) -> None:
        declarations = {
            declaration["id"]: declaration
            for declaration in semantic_qualification_declarations()
        }
        qualification = declarations[TASK10_SEMANTIC_QUALIFICATION_ID]
        self.assertEqual(
            list(TASK10_SEMANTIC_QUALIFICATION_OWNERS),
            qualification["owners"],
        )

        manifest = json.loads(
            (CATALOG_PATH.parents[1] / "compat" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        choice = next(
            entry
            for entry in manifest["groups"]
            if entry["family"] == "random_choice"
        )
        self.assertEqual(
            list(TASK10_SEMANTIC_QUALIFICATION_OWNERS),
            choice["tests"],
        )
        self.assertEqual(
            ["cnp_random_choice", "cnp_random_choice_v2"],
            choice["symbols"],
        )

        readme = (CATALOG_PATH.parent / "README.md").read_text(
            encoding="utf-8"
        )
        for owner in TASK10_SEMANTIC_QUALIFICATION_OWNERS:
            with self.subTest(owner=owner):
                self.assertIn(owner, readme)

    def test_focus_contains_the_intended_fast_regression_surface(self) -> None:
        focus_ids = {case["id"] for case in expand_cases(self.catalog, "focus")}
        expected_ids = {
            "bridge/property_call",
            "bridge/property_cached",
            "bridge/nbytes_cached",
            "bridge/c_contiguous_cached",
            "bridge/f_contiguous_cached",
            "bridge/static_add_call",
            "concatenate/f64/512x512/axis0",
            "choice_weighted/f64/1000000",
        }
        expected_ids.update({f"zeros/f64/{size}" for size in (1_000, 1_000_000)})
        expected_ids.update(
            f"allclose/f64/{size}" for size in (1_000, 1_000_000)
        )
        for operation in ("angle", "real", "imag", "real_if_close"):
            expected_ids.update(
                f"{operation}/f64/{size}" for size in (1_000, 1_000_000)
            )
        for operation in OPERATION_FAMILIES["signal"]:
            expected_ids.update(
                f"{operation}/f64/{size}" for size in (1_000, 1_000_000)
            )
        for operation in OPERATION_FAMILIES["reduction"] - {
            "sum_axis_last", "cumsum_axis_last"
        }:
            expected_ids.update(f"{operation}/f64/{size}" for size in (1_000, 1_000_000))
        expected_ids.update({
            "sum_axis_last/f64/512x512/axis1",
            "cumsum_axis_last/f64/512x512/axis1",
        })
        expected_ids.update({
            "softmax/f64/10000",
            "softmax/f64/1000000",
            "softmax_axis_last/f64/512x512/axis1",
            "softmax_axis0_strided/f64/512x512/axis0",
            "log_softmax/f64/10000",
            "log_softmax/f64/1000000",
            "log_softmax_axis_last/f64/512x512/axis1",
            "log_softmax_axis0_strided/f64/512x512/axis0",
            "trapz/f64/10000",
            "trapz/f64/1000000",
            "trapz_axis_last/f64/512x512/axis1",
            "trapz_axis0_strided/f64/512x512/axis0",
            "packbits/u8/10000",
            "packbits/u8/1000000",
            "packbits_axis_last/u8/512x512/axis1",
            "packbits_axis0_strided/u8/512x512/axis0",
            "unpackbits/u8/10000",
            "unpackbits/u8/1000000",
            "unpackbits_axis_last/u8/512x512/axis1",
            "unpackbits_axis0_strided/u8/512x512/axis0",
        })
        for operation in OPERATION_FAMILIES["sorting"]:
            expected_ids.update(f"{operation}/f64/{size}" for size in (10_000, 1_000_000))
        for operation in OPERATION_FAMILIES["set"]:
            expected_ids.update(f"{operation}/f64/{size}" for size in (10_000, 1_000_000))
        for operation in {
            "copy", "reshape", "flatten",
            "atleast_1d", "atleast_2d", "atleast_3d",
        }:
            expected_ids.update(f"{operation}/f64/{size}" for size in (1_000, 1_000_000))
        for operation in {"take", "compress"}:
            expected_ids.update(
                f"{operation}/f64/{size}" for size in (1_000, 1_000_000)
            )
        expected_ids.update(
            f"{operation}/f64/512x512/axis0"
            for operation in (
                "take_axis0_block", "take_axis0_strided",
                "compress_axis0_block", "compress_axis0_strided",
            )
        )
        for category in ("preallocated", "pipeline"):
            for operation in OPERATION_FAMILIES[category]:
                expected_ids.update(
                    f"{operation}/f64/{size}" for size in (1_000, 1_000_000)
                )

        self.assertEqual(expected_ids, focus_ids)
        self.assertEqual(152, len(focus_ids))

    def test_focus_contains_large_take_and_compress_cases(self) -> None:
        focus_ids = {case["id"] for case in expand_cases(self.catalog, "focus")}
        self.assertEqual(
            {
                f"{operation}/f64/{size}"
                for operation in ("take", "compress")
                for size in (1_000, 1_000_000)
            },
            {
                case_id
                for case_id in focus_ids
                if case_id.startswith(("take/", "compress/"))
            },
        )

    def test_focus_contains_exact_task8_softmax_vector_and_axis_cases(
        self,
    ) -> None:
        cases = {
            case["id"]: case
            for case in expand_cases(self.catalog, "focus")
            if case["operation"].startswith("softmax")
        }

        self.assertEqual(
            {
                "softmax/f64/10000",
                "softmax/f64/1000000",
                "softmax_axis_last/f64/512x512/axis1",
                "softmax_axis0_strided/f64/512x512/axis0",
            },
            set(cases),
        )
        self.assertEqual("misc_axis", cases["softmax/f64/10000"]["category"])
        self.assertEqual(
            (512 * 512, 512, 512, 1),
            self.shape_tuple(cases["softmax_axis_last/f64/512x512/axis1"]),
        )
        self.assertEqual(
            (512 * 512, 512, 512, 0),
            self.shape_tuple(
                cases["softmax_axis0_strided/f64/512x512/axis0"]
            ),
        )

    def test_focus_contains_exact_task8_log_softmax_vector_and_axis_cases(
        self,
    ) -> None:
        cases = {
            case["id"]: case
            for case in expand_cases(self.catalog, "focus")
            if case["operation"].startswith("log_softmax")
        }

        self.assertEqual(
            {
                "log_softmax/f64/10000",
                "log_softmax/f64/1000000",
                "log_softmax_axis_last/f64/512x512/axis1",
                "log_softmax_axis0_strided/f64/512x512/axis0",
            },
            set(cases),
        )

    def test_focus_contains_exact_task8_trapz_vector_and_axis_cases(
        self,
    ) -> None:
        cases = {
            case["id"]: case
            for case in expand_cases(self.catalog, "focus")
            if case["operation"].startswith("trapz")
        }

        self.assertEqual(
            {
                "trapz/f64/10000",
                "trapz/f64/1000000",
                "trapz_axis_last/f64/512x512/axis1",
                "trapz_axis0_strided/f64/512x512/axis0",
            },
            set(cases),
        )

    def test_focus_contains_exact_task8_packbits_vector_and_axis_cases(
        self,
    ) -> None:
        cases = {
            case["id"]: case
            for case in expand_cases(self.catalog, "focus")
            if case["operation"].startswith("packbits")
        }

        self.assertEqual(
            {
                "packbits/u8/10000",
                "packbits/u8/1000000",
                "packbits_axis_last/u8/512x512/axis1",
                "packbits_axis0_strided/u8/512x512/axis0",
            },
            set(cases),
        )

    def test_focus_contains_exact_task8_unpackbits_vector_and_axis_cases(
        self,
    ) -> None:
        cases = {
            case["id"]: case
            for case in expand_cases(self.catalog, "focus")
            if case["operation"].startswith("unpackbits")
        }

        self.assertEqual(
            {
                "unpackbits/u8/10000",
                "unpackbits/u8/1000000",
                "unpackbits_axis_last/u8/512x512/axis1",
                "unpackbits_axis0_strided/u8/512x512/axis0",
            },
            set(cases),
        )

    def test_focus_contains_axis0_block_and_strided_indexing_matrix_cases(
        self,
    ) -> None:
        indexing_cases = {
            case["id"]: case
            for case in expand_cases(self.catalog, "focus")
            if case["operation"] in {
                "take_axis0_block",
                "take_axis0_strided",
                "compress_axis0_block",
                "compress_axis0_strided",
            }
        }
        expected_ids = {
            f"{operation}/f64/512x512/axis0"
            for operation in (
                "take_axis0_block",
                "take_axis0_strided",
                "compress_axis0_block",
                "compress_axis0_strided",
            )
        }

        self.assertEqual(expected_ids, set(indexing_cases))
        for case in indexing_cases.values():
            self.assertEqual("indexing", case["category"])
            self.assertEqual(512 * 512, case["size"])
            self.assertEqual(512, case["rows"])
            self.assertEqual(512, case["cols"])
            self.assertEqual(0, case["axis"])

    def test_standard_contains_expm1_native_vector_cases(self) -> None:
        standard_ids = {
            case["id"] for case in expand_cases(self.catalog, "standard")
        }
        self.assertEqual(
            {
                f"expm1/f64/{size}"
                for size in (1_000, 10_000, 100_000, 1_000_000)
            },
            {
                case_id
                for case_id in standard_ids
                if case_id.startswith("expm1/")
            },
        )

    def test_standard_contains_log2_native_vector_cases(self) -> None:
        standard_ids = {
            case["id"] for case in expand_cases(self.catalog, "standard")
        }
        self.assertEqual(
            {
                f"log2/f64/{size}"
                for size in (1_000, 10_000, 100_000, 1_000_000)
            },
            {
                case_id
                for case_id in standard_ids
                if case_id.startswith("log2/")
            },
        )

    def test_standard_contains_log10_native_vector_cases(self) -> None:
        standard_ids = {
            case["id"] for case in expand_cases(self.catalog, "standard")
        }
        self.assertEqual(
            {
                f"log10/f64/{size}"
                for size in (1_000, 10_000, 100_000, 1_000_000)
            },
            {
                case_id
                for case_id in standard_ids
                if case_id.startswith("log10/")
            },
        )

    def test_standard_contains_log1p_native_vector_cases(self) -> None:
        standard_ids = {
            case["id"] for case in expand_cases(self.catalog, "standard")
        }
        self.assertEqual(
            {
                f"log1p/f64/{size}"
                for size in (1_000, 10_000, 100_000, 1_000_000)
            },
            {
                case_id
                for case_id in standard_ids
                if case_id.startswith("log1p/")
            },
        )

    def test_standard_contains_exact_cholesky_cases(self) -> None:
        standard_ids = {
            case["id"] for case in expand_cases(self.catalog, "standard")
        }
        self.assertEqual(
            {"cholesky/f64/32x32", "cholesky/f64/128x128"},
            {
                case_id
                for case_id in standard_ids
                if case_id.startswith("cholesky/")
            },
        )

    def test_standard_contains_logaddexp_native_vector_cases(self) -> None:
        standard_ids = {
            case["id"] for case in expand_cases(self.catalog, "standard")
        }
        self.assertEqual(
            {
                f"logaddexp/f64/{size}"
                for size in (1_000, 10_000, 100_000, 1_000_000)
            },
            {
                case_id
                for case_id in standard_ids
                if case_id.startswith("logaddexp/")
            },
        )

    def test_standard_contains_power_family_native_vector_cases(self) -> None:
        standard_ids = {
            case["id"] for case in expand_cases(self.catalog, "standard")
        }
        for operation in ("power", "float_power"):
            with self.subTest(operation=operation):
                self.assertEqual(
                    {
                        f"{operation}/f64/{size}"
                        for size in (1_000, 10_000, 100_000, 1_000_000)
                    },
                    {
                        case_id
                        for case_id in standard_ids
                        if case_id.startswith(f"{operation}/")
                    },
                )

    def test_standard_contains_heaviside_native_vector_cases(self) -> None:
        standard_ids = {
            case["id"] for case in expand_cases(self.catalog, "standard")
        }
        self.assertEqual(
            {
                f"heaviside/f64/{size}"
                for size in (1_000, 10_000, 100_000, 1_000_000)
            },
            {
                case_id
                for case_id in standard_ids
                if case_id.startswith("heaviside/")
            },
        )

    def test_standard_contains_divmod_native_vector_cases(self) -> None:
        standard_ids = {
            case["id"] for case in expand_cases(self.catalog, "standard")
        }
        self.assertEqual(
            {
                f"divmod/f64/{size}"
                for size in (1_000, 10_000, 100_000, 1_000_000)
            },
            {
                case_id
                for case_id in standard_ids
                if case_id.startswith("divmod/")
            },
        )

    def test_standard_contains_logaddexp2_native_vector_cases(self) -> None:
        standard_ids = {
            case["id"] for case in expand_cases(self.catalog, "standard")
        }
        self.assertEqual(
            {
                f"logaddexp2/f64/{size}"
                for size in (1_000, 10_000, 100_000, 1_000_000)
            },
            {
                case_id
                for case_id in standard_ids
                if case_id.startswith("logaddexp2/")
            },
        )

    def test_standard_contains_equal_native_vector_cases(self) -> None:
        standard_ids = {
            case["id"] for case in expand_cases(self.catalog, "standard")
        }
        self.assertEqual(
            {
                f"equal/f64/{size}"
                for size in (1_000, 10_000, 100_000, 1_000_000)
            },
            {
                case_id
                for case_id in standard_ids
                if case_id.startswith("equal/")
            },
        )

    def test_standard_contains_all_extrema_native_vector_cases(self) -> None:
        standard_ids = {
            case["id"] for case in expand_cases(self.catalog, "standard")
        }
        for operation in ("maximum", "minimum", "fmax", "fmin"):
            with self.subTest(operation=operation):
                self.assertEqual(
                    {
                        f"{operation}/f64/{size}"
                        for size in (1_000, 10_000, 100_000, 1_000_000)
                    },
                    {
                        case_id
                        for case_id in standard_ids
                        if case_id.startswith(f"{operation}/")
                    },
                )

    def test_standard_contains_all_logical_native_vector_cases(self) -> None:
        standard_ids = {
            case["id"] for case in expand_cases(self.catalog, "standard")
        }
        for operation in (
            "logical_and", "logical_or", "logical_xor", "logical_not"
        ):
            with self.subTest(operation=operation):
                self.assertEqual(
                    {
                        f"{operation}/f64/{size}"
                        for size in (1_000, 10_000, 100_000, 1_000_000)
                    },
                    {
                        case_id
                        for case_id in standard_ids
                        if case_id.startswith(f"{operation}/")
                    },
                )

    def test_standard_contains_all_array_predicate_native_vector_cases(
        self,
    ) -> None:
        standard_ids = {
            case["id"] for case in expand_cases(self.catalog, "standard")
        }
        for operation in ("isnan", "isinf", "isfinite", "signbit"):
            with self.subTest(operation=operation):
                self.assertEqual(
                    {
                        f"{operation}/f64/{size}"
                        for size in (1_000, 10_000, 100_000, 1_000_000)
                    },
                    {
                        case_id
                        for case_id in standard_ids
                        if case_id.startswith(f"{operation}/")
                    },
                )

    def test_standard_contains_all_object_kind_predicate_cases(self) -> None:
        standard_ids = {
            case["id"] for case in expand_cases(self.catalog, "standard")
        }
        for operation in ("iscomplexobj", "isrealobj", "isscalar"):
            with self.subTest(operation=operation):
                self.assertEqual(
                    {
                        f"{operation}/f64/{size}"
                        for size in (1_000, 10_000, 100_000, 1_000_000)
                    },
                    {
                        case_id
                        for case_id in standard_ids
                        if case_id.startswith(f"{operation}/")
                    },
                )

    def test_standard_contains_all_bitwise_native_integer_vector_cases(
        self,
    ) -> None:
        standard_ids = {
            case["id"] for case in expand_cases(self.catalog, "standard")
        }
        for operation in OPERATION_FAMILIES["bitwise"]:
            with self.subTest(operation=operation):
                self.assertEqual(
                    {
                        f"{operation}/i64/{size}"
                        for size in (1_000, 10_000, 100_000, 1_000_000)
                    },
                    {
                        case_id
                        for case_id in standard_ids
                        if case_id.startswith(f"{operation}/")
                    },
                )

    def test_standard_contains_gcd_lcm_native_integer_vector_cases(
        self,
    ) -> None:
        standard_ids = {
            case["id"] for case in expand_cases(self.catalog, "standard")
        }
        for operation in OPERATION_FAMILIES["integer"]:
            with self.subTest(operation=operation):
                self.assertEqual(
                    {
                        f"{operation}/i64/{size}"
                        for size in (1_000, 10_000, 100_000, 1_000_000)
                    },
                    {
                        case_id
                        for case_id in standard_ids
                        if case_id.startswith(f"{operation}/")
                    },
                )

    def test_standard_covers_every_operation_at_normal_sizes(self) -> None:
        standard = expand_cases(self.catalog, "standard")
        self.assertEqual(
            set().union(*OPERATION_FAMILIES.values()),
            {case["operation"] for case in standard},
        )

        standard_ids = {case["id"] for case in standard}
        self.assertTrue(
            {f"zeros/f64/{size}" for size in (1_000, 10_000, 100_000, 1_000_000)}
            <= standard_ids
        )
        self.assertTrue(
            {f"matmul/f64/{size}x{size}" for size in (32, 128, 512)} <= standard_ids
        )
        self.assertTrue(
            {f"fft/f64/{size}" for size in (1_024, 4_096, 16_384, 65_536)} <= standard_ids
        )
        for operation in ("det", "inv", "solve"):
            self.assertEqual(
                {f"{operation}/f64/32x32", f"{operation}/f64/128x128"},
                {case_id for case_id in standard_ids if case_id.startswith(f"{operation}/")},
            )

        self.assertEqual(
            {f"einsum/f64/{size}x{size}" for size in (32, 128, 512)},
            {case_id for case_id in standard_ids if case_id.startswith("einsum/")},
        )
        for operation in ("eig", "svd", "lstsq"):
            self.assertEqual(
                {f"{operation}/f64/32x32", f"{operation}/f64/128x128"},
                {
                    case_id
                    for case_id in standard_ids
                    if case_id.startswith(f"{operation}/")
                },
            )

    def test_full_is_standard_plus_only_the_heavy_linalg_cases(self) -> None:
        standard_ids = {case["id"] for case in expand_cases(self.catalog, "standard")}
        full_ids = {case["id"] for case in expand_cases(self.catalog, "full")}

        self.assertEqual(455, len(standard_ids))
        self.assertEqual(459, len(full_ids))
        self.assertLess(standard_ids, full_ids)
        self.assertEqual(
            {
                "det/f64/512x512",
                "inv/f64/512x512",
                "solve/f64/512x512",
                "cholesky/f64/512x512",
            },
            full_ids - standard_ids,
        )

    def test_expanded_cases_are_sorted_unique_and_have_exact_fields(self) -> None:
        for profile in ("focus", "standard", "full"):
            with self.subTest(profile=profile):
                cases = expand_cases(self.catalog, profile)
                ids = [case["id"] for case in cases]
                self.assertEqual(sorted(ids), ids)
                self.assertEqual(len(ids), len(set(ids)))
                self.assertTrue(cases)
                self.assertTrue(all(set(case) == CASE_FIELDS for case in cases))

    def test_expansion_populates_bridge_vector_fft_and_matrix_shapes(self) -> None:
        cases = {case["id"]: case for case in expand_cases(self.catalog, "standard")}

        self.assertEqual(
            {
                "id": "bridge/property_call",
                "category": "bridge",
                "operation": "property_call",
                "dtype": "f64",
                "size": 1,
                "rows": 0,
                "cols": 0,
                "axis": -1,
            },
            cases["bridge/property_call"],
        )
        for operation in (
            "property_cached",
            "nbytes_cached",
            "c_contiguous_cached",
            "f_contiguous_cached",
            "static_add_call",
        ):
            with self.subTest(operation=operation):
                self.assertEqual(
                    (1, 0, 0, -1),
                    self.shape_tuple(cases[f"bridge/{operation}"]),
                )
        self.assertEqual((1_000, 0, 0, -1), self.shape_tuple(cases["zeros/f64/1000"]))
        self.assertEqual((1_024, 0, 0, -1), self.shape_tuple(cases["fft/f64/1024"]))
        self.assertEqual(
            (512 * 512, 512, 512, 0),
            self.shape_tuple(cases["concatenate/f64/512x512/axis0"]),
        )
        for operation in ("sum_axis_last", "cumsum_axis_last"):
            with self.subTest(operation=operation):
                self.assertEqual(
                    (512 * 512, 512, 512, 1),
                    self.shape_tuple(
                        cases[f"{operation}/f64/512x512/axis1"]
                    ),
                )
        self.assertTrue(
            {"dot/f64/32x32", "norm/f64/32x32"} <= set(cases),
            "all linalg operations must expand as matrices",
        )

    @staticmethod
    def shape_tuple(case: dict[str, object]) -> tuple[object, object, object, object]:
        return case["size"], case["rows"], case["cols"], case["axis"]

    def test_tsv_round_trip_preserves_every_case_value(self) -> None:
        cases = expand_cases(self.catalog, "focus")
        destination = io.StringIO()

        write_jobs_tsv(cases, destination)

        destination.seek(0)
        rows = list(csv.DictReader(destination, delimiter="\t"))
        expected_rows = [{field: str(case[field]) for field in CASE_FIELDS} for case in cases]
        self.assertEqual(
            ["id", "category", "operation", "dtype", "size", "rows", "cols", "axis"],
            list(rows[0]),
        )
        self.assertEqual(expected_rows, rows)


class CatalogValidationTests(TemporaryCatalogMixin, unittest.TestCase):
    def assert_invalid(self, document: object, pattern: str) -> None:
        with self.assertRaisesRegex(ValueError, pattern):
            self.load_document(document)

    def test_cholesky_catalog_contract_is_strict(self) -> None:
        document = catalog_for_operation("cholesky")
        loaded = self.load_document(document)
        self.assertEqual("cholesky", loaded["templates"][0]["operation"])

        invalid_fields = (
            ("category", "shape"),
            ("shape_kind", "vector"),
            ("axis", 0),
            ("dtype", "i64"),
        )
        for field, value in invalid_fields:
            with self.subTest(field=field):
                invalid = deepcopy(document)
                invalid["templates"][0][field] = value
                self.assert_invalid(invalid, r"templates\[0\]")

    def test_accepts_size_set_references_and_explicit_positive_size_lists(self) -> None:
        document = minimal_catalog()
        document["size_sets"]["vector"] = [3, 7, 10]
        template = document["templates"][0]
        template["profiles"] = {"focus": [3, 7], "standard": "vector"}

        catalog = self.load_document(document)

        self.assertEqual(
            ["zeros/f64/3", "zeros/f64/7"],
            [case["id"] for case in expand_cases(catalog, "focus")],
        )
        self.assertEqual(
            ["zeros/f64/10", "zeros/f64/3", "zeros/f64/7"],
            [case["id"] for case in expand_cases(catalog, "standard")],
        )

    def test_rejects_wrong_top_level_container_missing_keys_and_unknown_keys(self) -> None:
        self.assert_invalid([], "catalog")

        missing = minimal_catalog()
        del missing["templates"]
        self.assert_invalid(missing, "top-level.*missing.*templates")

        unknown = minimal_catalog()
        unknown["unexpected"] = True
        self.assert_invalid(unknown, "top-level.*unknown.*unexpected")

    def test_rejects_wrong_schema_versions(self) -> None:
        for schema_version in (True, 0, 2, "1"):
            with self.subTest(schema_version=schema_version):
                document = minimal_catalog()
                document["schema_version"] = schema_version
                self.assert_invalid(document, "schema_version")

    def test_rejects_duplicate_keys_at_every_json_object_depth(self) -> None:
        serialized = json.dumps(minimal_catalog())
        duplicate_documents = (
            (
                serialized.replace(
                    '"schema_version": 1',
                    '"schema_version": 1, "schema_version": 1',
                    1,
                ),
                "schema_version",
            ),
            (
                serialized.replace('"axis": -1', '"axis": -1, "axis": -1', 1),
                "axis",
            ),
        )

        for text, duplicate_key in duplicate_documents:
            with self.subTest(duplicate_key=duplicate_key):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"duplicate key.*{duplicate_key}",
                ):
                    self.load_text(text)

    def test_rejects_wrong_size_set_container_missing_keys_and_unknown_keys(self) -> None:
        wrong_container = minimal_catalog()
        wrong_container["size_sets"] = []
        self.assert_invalid(wrong_container, "size_sets")

        missing = minimal_catalog()
        del missing["size_sets"]["fft"]
        self.assert_invalid(missing, "size_sets.*missing.*fft")

        unknown = minimal_catalog()
        unknown["size_sets"]["extra"] = [1]
        self.assert_invalid(unknown, "size_sets.*unknown.*extra")

    def test_rejects_empty_duplicate_non_integer_and_non_positive_sizes(self) -> None:
        invalid_values = ([], [1, 1], [0], [-1], [True], [1.5], ["1"])
        for value in invalid_values:
            with self.subTest(value=value):
                document = minimal_catalog()
                document["size_sets"]["vector"] = value
                self.assert_invalid(document, r"size_sets\.vector")

    def test_rejects_wrong_or_empty_template_collection(self) -> None:
        for templates in ({}, [], ["not-a-template"]):
            with self.subTest(templates=templates):
                document = minimal_catalog()
                document["templates"] = templates
                self.assert_invalid(document, "templates")

    def test_rejects_template_missing_and_unknown_keys(self) -> None:
        missing = minimal_catalog()
        del missing["templates"][0]["axis"]
        self.assert_invalid(missing, r"templates\[0\].*missing.*axis")

        unknown = minimal_catalog()
        unknown["templates"][0]["unexpected"] = True
        self.assert_invalid(unknown, r"templates\[0\].*unknown.*unexpected")

    def test_rejects_empty_or_non_string_category_and_operation(self) -> None:
        for field in ("category", "operation"):
            for value in ("", 1):
                with self.subTest(field=field, value=value):
                    document = minimal_catalog()
                    document["templates"][0][field] = value
                    self.assert_invalid(document, rf"templates\[0\]\.{field}")

    def test_rejects_ahk_unsafe_catalog_identity_characters(self) -> None:
        for field in ("category", "operation", "dtype"):
            for forbidden_character in ("\t", "\r", "\n", '"'):
                with self.subTest(field=field, forbidden_character=forbidden_character):
                    document = minimal_catalog()
                    original = document["templates"][0][field]
                    document["templates"][0][field] = (
                        f"{original}{forbidden_character}suffix"
                    )
                    self.assert_invalid(
                        document,
                        rf"templates\[0\]\.{field}.*forbidden",
                    )

    def test_rejects_duplicate_operations_even_across_categories(self) -> None:
        document = minimal_catalog()
        duplicate = deepcopy(document["templates"][0])
        duplicate["category"] = "different-category"
        document["templates"].append(duplicate)

        self.assert_invalid(document, r"templates\[1\]\.operation.*duplicate.*zeros")

    def test_enforces_category_shape_and_axis_contract_for_every_operation(self) -> None:
        shape_kinds = ("bridge", "vector", "matrix", "fft")
        for operation, (category, shape_kind, axis) in OPERATION_CONTRACTS.items():
            with self.subTest(operation=operation, field="valid"):
                self.load_document(catalog_for_operation(operation))

            invalid_values = (
                ("category", "wrong-category"),
                (
                    "shape_kind",
                    next(candidate for candidate in shape_kinds if candidate != shape_kind),
                ),
                ("axis", 0 if axis == -1 else -1),
            )
            for field, value in invalid_values:
                with self.subTest(operation=operation, field=field):
                    document = catalog_for_operation(operation)
                    document["templates"][0][field] = value
                    self.assert_invalid(document, rf"templates\[0\].*{operation}")

    def test_accepts_all_distinct_supported_bridge_templates(self) -> None:
        document = minimal_catalog()
        document["templates"] = [
            catalog_for_operation(operation)["templates"][0]
            for operation in OPERATION_FAMILIES["bridge"]
        ]

        loaded = self.load_document(document)
        self.assertEqual(6, len(loaded["templates"]))

    def test_profile_sizes_must_match_the_template_shape(self) -> None:
        invalid_profiles = (
            ("zeros", "matrix"),
            ("zeros", [4]),
            ("matmul", "vector"),
            ("matmul", [10]),
            ("fft", "vector"),
            ("fft", [10]),
            ("property_call", "vector"),
            ("property_call", [1, 2]),
        )
        for operation, profile_sizes in invalid_profiles:
            with self.subTest(operation=operation, profile_sizes=profile_sizes):
                document = catalog_for_operation(operation)
                document["templates"][0]["profiles"] = {"focus": profile_sizes}
                self.assert_invalid(document, r"templates\[0\]\.profiles\.focus")

    def test_rejects_boolean_bridge_dimensions(self) -> None:
        document = catalog_for_operation("property_call")
        document["templates"][0]["profiles"] = {"focus": [True]}

        self.assert_invalid(document, r"templates\[0\]\.profiles\.focus")

    def test_rejects_invalid_dtype_shape_kind_and_axis(self) -> None:
        invalid_fields = (
            ("dtype", "f32"),
            ("dtype", 64),
            ("shape_kind", "tensor"),
            ("shape_kind", 1),
            ("axis", True),
            ("axis", 0.0),
        )
        for field, value in invalid_fields:
            with self.subTest(field=field, value=value):
                document = minimal_catalog()
                document["templates"][0][field] = value
                self.assert_invalid(document, rf"templates\[0\]\.{field}")

    def test_rejects_empty_wrong_or_unknown_profile_mappings(self) -> None:
        invalid_profiles = (
            {},
            [],
            {"quick": "vector"},
            {"focus": "unknown-size-set"},
            {"focus": 10},
        )
        for profiles in invalid_profiles:
            with self.subTest(profiles=profiles):
                document = minimal_catalog()
                document["templates"][0]["profiles"] = profiles
                self.assert_invalid(document, r"templates\[0\]\.profiles")

    def test_rejects_empty_duplicate_non_integer_and_non_positive_explicit_sizes(self) -> None:
        invalid_values = ([], [1, 1], [0], [-1], [True], [1.5], ["1"])
        for value in invalid_values:
            with self.subTest(value=value):
                document = minimal_catalog()
                document["templates"][0]["profiles"] = {"focus": value}
                self.assert_invalid(document, r"templates\[0\]\.profiles\.focus")


class CaseExpansionValidationTests(TemporaryCatalogMixin, unittest.TestCase):
    def test_rejects_unknown_profile(self) -> None:
        catalog = self.load_document(minimal_catalog())

        with self.assertRaisesRegex(ValueError, "profile"):
            expand_cases(catalog, "quick")

    def test_rejects_duplicate_expanded_ids(self) -> None:
        catalog = self.load_document(minimal_catalog())
        catalog["templates"].append(deepcopy(catalog["templates"][0]))

        with self.assertRaisesRegex(ValueError, "duplicate.*zeros"):
            expand_cases(catalog, "focus")

    def test_revalidates_catalog_dict_at_the_expansion_boundary(self) -> None:
        try:
            expand_cases({}, "focus")
        except Exception as error:
            self.assertIsInstance(error, ValueError)
        else:
            self.fail("malformed catalog must raise ValueError")

        invalid_catalog = minimal_catalog()
        invalid_catalog["templates"][0]["axis"] = 0
        with self.assertRaisesRegex(ValueError, "zeros"):
            expand_cases(invalid_catalog, "focus")


class JobTsvValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = {
            "id": "zeros/f64/1000",
            "category": "creation",
            "operation": "zeros",
            "dtype": "f64",
            "size": 1_000,
            "rows": 0,
            "cols": 0,
            "axis": -1,
        }

    def test_rejects_empty_jobs(self) -> None:
        with self.assertRaisesRegex(ValueError, "cases.*empty"):
            write_jobs_tsv([], io.StringIO())

    def test_rejects_missing_and_extra_case_fields(self) -> None:
        missing = dict(self.case)
        del missing["axis"]
        extra = {**self.case, "unexpected": True}

        for case, pattern in ((missing, "missing.*axis"), (extra, "extra.*unexpected")):
            with self.subTest(case=case):
                with self.assertRaisesRegex(ValueError, pattern):
                    write_jobs_tsv([case], io.StringIO())

    def test_rejects_duplicate_case_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate.*zeros/f64/1000"):
            write_jobs_tsv([self.case, dict(self.case)], io.StringIO())

    def test_rejects_a_structurally_valid_job_with_wrong_operation_semantics(self) -> None:
        invalid = {**self.case, "category": "unary"}

        with self.assertRaisesRegex(ValueError, r"zeros/f64/1000.*category"):
            write_jobs_tsv([invalid], io.StringIO())

    def test_rejects_empty_and_non_string_identity_fields(self) -> None:
        for field in ("id", "category", "operation"):
            for value in ("", 1):
                with self.subTest(field=field, value=value):
                    case = {**self.case, field: value}
                    with self.assertRaisesRegex(ValueError, rf"cases\[0\]\.{field}"):
                        write_jobs_tsv([case], io.StringIO())

    def test_rejects_non_f64_and_non_string_dtypes(self) -> None:
        for dtype in ("f32", 64):
            with self.subTest(dtype=dtype):
                case = {**self.case, "dtype": dtype}
                with self.assertRaisesRegex(ValueError, r"cases\[0\]\.dtype"):
                    write_jobs_tsv([case], io.StringIO())

    def test_rejects_ahk_unsafe_string_field_characters(self) -> None:
        for field in ("id", "category", "operation", "dtype"):
            for forbidden_character in ("\t", "\r", "\n", '"'):
                with self.subTest(field=field, forbidden_character=forbidden_character):
                    case = {
                        **self.case,
                        field: f"{self.case[field]}{forbidden_character}suffix",
                    }
                    with self.assertRaisesRegex(
                        ValueError,
                        rf"cases\[0\]\.{field}.*forbidden",
                    ):
                        write_jobs_tsv([case], io.StringIO())

    def test_rejects_non_integer_boolean_and_non_positive_sizes(self) -> None:
        for size in (1.5, "1000", True, 0, -1):
            with self.subTest(size=size):
                case = {**self.case, "size": size}
                with self.assertRaisesRegex(ValueError, r"cases\[0\]\.size"):
                    write_jobs_tsv([case], io.StringIO())

    def test_rejects_non_integer_boolean_and_negative_dimensions(self) -> None:
        for field in ("rows", "cols"):
            for value in (1.5, "1", True, -1):
                with self.subTest(field=field, value=value):
                    case = {**self.case, field: value}
                    with self.assertRaisesRegex(ValueError, rf"cases\[0\]\.{field}"):
                        write_jobs_tsv([case], io.StringIO())

    def test_rejects_non_integer_and_boolean_axes(self) -> None:
        for axis in (0.0, "0", True):
            with self.subTest(axis=axis):
                case = {**self.case, "axis": axis}
                with self.assertRaisesRegex(ValueError, r"cases\[0\]\.axis"):
                    write_jobs_tsv([case], io.StringIO())

    def test_rejects_one_zero_matrix_dimension(self) -> None:
        for rows, cols in ((0, 4), (4, 0)):
            with self.subTest(rows=rows, cols=cols):
                case = {**self.case, "size": 16, "rows": rows, "cols": cols}
                with self.assertRaisesRegex(ValueError, r"cases\[0\].*rows.*cols"):
                    write_jobs_tsv([case], io.StringIO())

    def test_rejects_matrix_size_that_does_not_match_dimensions(self) -> None:
        case = {**self.case, "size": 15, "rows": 4, "cols": 4}

        with self.assertRaisesRegex(ValueError, r"cases\[0\]\.size"):
            write_jobs_tsv([case], io.StringIO())

    def test_accepts_bridge_vector_and_square_matrix_jobs(self) -> None:
        bridge = {
            "id": "bridge/property_call",
            "category": "bridge",
            "operation": "property_call",
            "dtype": "f64",
            "size": 1,
            "rows": 0,
            "cols": 0,
            "axis": -1,
        }
        matrix = {
            "id": "matmul/f64/4x4",
            "category": "linalg",
            "operation": "matmul",
            "dtype": "f64",
            "size": 16,
            "rows": 4,
            "cols": 4,
            "axis": -1,
        }
        destination = io.StringIO()

        write_jobs_tsv([bridge, self.case, matrix], destination)

        self.assertNotIn('"', destination.getvalue())
        destination.seek(0)
        self.assertEqual(3, len(list(csv.DictReader(destination, delimiter="\t"))))

    def test_cholesky_job_contract_accepts_only_canonical_square_f64_jobs(
        self,
    ) -> None:
        case = {
            "id": "cholesky/f64/4x4",
            "category": "linalg",
            "operation": "cholesky",
            "dtype": "f64",
            "size": 16,
            "rows": 4,
            "cols": 4,
            "axis": -1,
        }
        write_jobs_tsv([case], io.StringIO())

        invalid_cases = (
            ({**case, "category": "shape"}, "category"),
            (
                {**case, "cols": 2, "size": 8,
                 "id": "cholesky/f64/4x2"},
                "cols",
            ),
            (
                {**case, "dtype": "i64",
                 "id": "cholesky/i64/4x4"},
                "dtype",
            ),
            ({**case, "axis": 0}, "axis"),
            ({**case, "id": "cholesky/f64/16"}, "id"),
        )
        for invalid, field in invalid_cases:
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    write_jobs_tsv([invalid], io.StringIO())


if __name__ == "__main__":
    unittest.main()
