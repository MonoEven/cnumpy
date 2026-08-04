from __future__ import annotations

from pathlib import Path
import importlib
import unittest

from compat.manifest import load_manifest, parse_public_declarations


ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "include" / "cnumpy" / "cnumpy.h"
MANIFEST = ROOT / "compat" / "manifest.json"

REDUCTION_V2_MATRIX_OWNER = (
    "benchmark.tests.test_reduction_semantics.ReductionSemanticsTests."
    "test_every_reduction_family_covers_rank_zero_through_four_and_all_axes"
)
LEGACY_REDUCTION_MATRIX_OWNER = (
    "benchmark.tests.test_reduction_semantics.ReductionSemanticsTests."
    "test_every_legacy_reduction_export_covers_rank_zero_through_four_and_expressible_axes"
)
LEGACY_REDUCTION_ERROR_OWNER = (
    "benchmark.tests.test_reduction_semantics.ReductionSemanticsTests."
    "test_every_legacy_reduction_export_reports_exact_axis_errors"
)
LEGACY_REDUCTION_SENTINEL_OWNER = (
    "benchmark.tests.test_reduction_semantics.ReductionSemanticsTests."
    "test_legacy_reduction_axis_minus_one_characterizes_none_sentinel"
)
AVERAGE_V2_MATRIX_OWNER = (
    "benchmark.tests.test_reduction_semantics.ReductionSemanticsTests."
    "test_average_v2_covers_rank_zero_through_four_axes_weights_errors_and_lifetimes"
)
LEGACY_NAN_SCALAR_MATRIX_OWNER = (
    "benchmark.tests.test_reduction_semantics.ReductionSemanticsTests."
    "test_legacy_nan_scalar_exports_cover_rank_zero_through_four_axes_values_errors_and_lifetimes"
)
SCALAR_CONVENIENCE_MATRIX_OWNER = (
    "benchmark.tests.test_reduction_semantics.ReductionSemanticsTests."
    "test_scalar_convenience_exports_cover_rank_zero_through_four_layouts_values_errors_and_lifetimes"
)
AHK_REDUCTION_OWNER = (
    "ahk.numpy.test.NumpyFoundationTest."
    "TestHighLevelScalarAndAxisReductions"
)
REDUCTION_LEGACY_SYMBOLS = (
    "cnp_sum",
    "cnp_prod",
    "cnp_mean",
    "cnp_var",
    "cnp_std",
    "cnp_max",
    "cnp_min",
    "cnp_amax",
    "cnp_amin",
    "cnp_argmax",
    "cnp_argmin",
    "cnp_any",
    "cnp_all",
    "cnp_ptp",
    "cnp_cumsum",
    "cnp_cumprod",
    "cnp_nansum",
    "cnp_nanprod",
    "cnp_nanmean",
    "cnp_nanvar",
    "cnp_nanstd",
    "cnp_nanmax",
    "cnp_nanmin",
    "cnp_nancumsum",
    "cnp_nancumprod",
    "cnp_median",
    "cnp_percentile",
    "cnp_quantile",
)
REDUCTION_V2_SYMBOLS = (
    "cnp_sum_v2",
    "cnp_prod_v2",
    "cnp_mean_v2",
    "cnp_var_v2",
    "cnp_std_v2",
    "cnp_max_v2",
    "cnp_min_v2",
    "cnp_argmax_v2",
    "cnp_argmin_v2",
    "cnp_any_v2",
    "cnp_all_v2",
    "cnp_ptp_v2",
    "cnp_cumsum_v2",
    "cnp_cumprod_v2",
    "cnp_nansum_v2",
    "cnp_nanprod_v2",
    "cnp_nanmean_v2",
    "cnp_nanvar_v2",
    "cnp_nanstd_v2",
    "cnp_nanmax_v2",
    "cnp_nanmin_v2",
    "cnp_nanargmax_v2",
    "cnp_nanargmin_v2",
    "cnp_median_v2",
    "cnp_nanmedian_v2",
    "cnp_percentile_v2",
    "cnp_nanpercentile_v2",
    "cnp_quantile_v2",
    "cnp_nanquantile_v2",
    "cnp_nancumsum_v2",
    "cnp_nancumprod_v2",
)
REDUCTION_OWNER_GROUPS = (
    (
        REDUCTION_LEGACY_SYMBOLS,
        (
            LEGACY_REDUCTION_MATRIX_OWNER,
            LEGACY_REDUCTION_ERROR_OWNER,
            LEGACY_REDUCTION_SENTINEL_OWNER,
            AHK_REDUCTION_OWNER,
        ),
    ),
    (
        REDUCTION_V2_SYMBOLS,
        (REDUCTION_V2_MATRIX_OWNER, AHK_REDUCTION_OWNER),
    ),
)
SET_VALUE_MATRIX_OWNER = (
    "benchmark.tests.test_sort_set_semantics.SortSetSemanticsTests."
    "test_set_value_operations_cover_rank_zero_through_four_numeric_dtypes"
)
SET_MEMBERSHIP_MATRIX_OWNER = (
    "benchmark.tests.test_sort_set_semantics.SortSetSemanticsTests."
    "test_membership_operations_cover_rank_zero_through_four_numeric_dtypes"
)
SET_MIXED_DTYPE_OWNER = (
    "benchmark.tests.test_sort_set_semantics.SortSetSemanticsTests."
    "test_set_operations_all_ordered_represented_dtype_pairs"
)
SET_EDGE_OWNER = (
    "benchmark.tests.test_sort_set_semantics.SortSetSemanticsTests."
    "test_set_operations_match_float_nan_zero_empty_and_mixed_dtype"
)
SET_PRECISION_OWNER = (
    "benchmark.tests.test_sort_set_semantics.SortSetSemanticsTests."
    "test_set_operations_preserve_int64_and_membership_shape"
)
SET_LIFETIME_OWNER = (
    "benchmark.tests.test_sort_set_semantics.SortSetSemanticsTests."
    "test_sort_unique_and_set_results_outlive_released_sources"
)
SET_ERROR_OWNER = (
    "benchmark.tests.test_sort_set_semantics.SortSetSemanticsTests."
    "test_set_operation_errors_are_exact_and_atomic"
)
SET_AHK_OWNER = (
    "ahk.numpy.test.NumpyFoundationTest.TestSortUniqueAndSetFacadeV2"
)
MISC_SOFTMAX_MATRIX_OWNER = (
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_softmax_rank_zero_through_four_all_axes_and_real_dtypes"
)
MISC_SOFTMAX_EDGE_OWNER = (
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_softmax_extremes_signed_zero_and_output_layout"
)
MISC_SOFTMAX_HUGE_EMPTY_OWNER = (
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_softmax_huge_empty_reduction_axis_errors_before_data_access"
)
MISC_TRAPZ_MATRIX_OWNER = (
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_trapz_rank_zero_through_four_all_axes"
)
MISC_TRAPZ_DTYPE_OWNER = (
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_trapz_matches_numpy_dtype_promotion_and_typed_arithmetic"
)
MISC_TRAPZ_X_DTYPE_OWNER = (
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_trapz_matches_numpy_x_dtype_promotion_and_product_overflow"
)
MISC_TRAPZ_WIDE_INTEGER_OWNER = (
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_trapz_preserves_wide_integer_panels_until_float_conversion"
)
MISC_BIT_V2_MATRIX_OWNER = (
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_bit_packing_rank_zero_through_four_all_axes_and_dtypes"
)
MISC_BIT_LEGACY_OWNER = (
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_legacy_bit_packing_rank_zero_through_four_and_errors"
)
MISC_BIT_EMPTY_OWNER = (
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_bit_packing_empty_dimensions_match_numpy"
)
MISC_BIT_HUGE_ZERO_OWNER = (
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_bit_packing_huge_zero_shapes_are_bounded_and_exact"
)
MISC_UNPACK_SIZE_OVERFLOW_OWNER = (
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_unpackbits_rejects_huge_multidimensional_result_before_allocation"
)
MISC_UNPACK_PADDING_EXTENSION_OWNER = (
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_unpackbits_empty_positive_count_zero_initializes_padding"
)
MISC_ERROR_OWNER = (
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_task8_core_exports_report_exact_invalid_axes_all_ranks"
)
MISC_LIFETIME_OWNER = (
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_task8_results_survive_source_release_and_retain_zero_bytes"
)
MISC_AHK_OWNER = (
    "ahk.numpy.test.NumpyFoundationTest.TestMiscAxisFacadeV2"
)


class CompatibilityManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.declarations = parse_public_declarations(
            HEADER.read_text(encoding="utf-8")
        )
        cls.manifest = load_manifest(MANIFEST)

    def test_public_declarations_are_unique(self) -> None:
        symbols = [declaration.symbol for declaration in self.declarations]
        duplicates = sorted(
            symbol for symbol in set(symbols) if symbols.count(symbol) > 1
        )
        self.assertEqual([], duplicates)

    def test_manifest_has_exact_public_symbol_coverage(self) -> None:
        declared = {item.symbol for item in self.declarations}
        inventoried = {item.symbol for item in self.manifest.exports}
        self.assertEqual(
            declared,
            inventoried,
            {
                "missing": sorted(declared - inventoried),
                "stale": sorted(inventoried - declared),
            },
        )

    def test_set_exports_have_exact_direct_semantic_owners(self) -> None:
        expected = {
            "sorting_set_values": (
                {
                    "cnp_intersect1d",
                    "cnp_union1d",
                    "cnp_setdiff1d",
                    "cnp_setxor1d",
                },
                (
                    SET_VALUE_MATRIX_OWNER,
                    SET_MIXED_DTYPE_OWNER,
                    SET_EDGE_OWNER,
                    SET_PRECISION_OWNER,
                    SET_LIFETIME_OWNER,
                    SET_ERROR_OWNER,
                    SET_AHK_OWNER,
                ),
            ),
            "sorting_membership": (
                {"cnp_in1d", "cnp_isin"},
                (
                    SET_MEMBERSHIP_MATRIX_OWNER,
                    SET_MIXED_DTYPE_OWNER,
                    SET_EDGE_OWNER,
                    SET_PRECISION_OWNER,
                    SET_LIFETIME_OWNER,
                    SET_ERROR_OWNER,
                    SET_AHK_OWNER,
                ),
            ),
        }
        for family, (symbols, owners) in expected.items():
            actual = {
                item.symbol: item
                for item in self.manifest.exports
                if item.family == family
            }
            with self.subTest(family=family):
                self.assertEqual(symbols, set(actual))
            for symbol, item in actual.items():
                with self.subTest(family=family, symbol=symbol):
                    self.assertEqual("differential", item.status)
                    self.assertEqual("array", item.result)
                    self.assertEqual(owners, item.tests)

    def test_task8_exports_have_exact_direct_semantic_owners(self) -> None:
        expected = {
            "misc_softmax": {
                "cnp_softmax": (
                    MISC_SOFTMAX_MATRIX_OWNER,
                    MISC_SOFTMAX_EDGE_OWNER,
                    MISC_SOFTMAX_HUGE_EMPTY_OWNER,
                    MISC_ERROR_OWNER,
                    MISC_LIFETIME_OWNER,
                    MISC_AHK_OWNER,
                ),
                "cnp_log_softmax": (
                    MISC_SOFTMAX_MATRIX_OWNER,
                    MISC_SOFTMAX_EDGE_OWNER,
                    MISC_SOFTMAX_HUGE_EMPTY_OWNER,
                    MISC_ERROR_OWNER,
                    MISC_LIFETIME_OWNER,
                    MISC_AHK_OWNER,
                ),
            },
            "misc_trapz": {
                "cnp_trapz": (
                    MISC_TRAPZ_MATRIX_OWNER,
                    MISC_TRAPZ_DTYPE_OWNER,
                    MISC_TRAPZ_X_DTYPE_OWNER,
                    MISC_TRAPZ_WIDE_INTEGER_OWNER,
                    MISC_ERROR_OWNER,
                    MISC_LIFETIME_OWNER,
                    MISC_AHK_OWNER,
                ),
            },
            "bit_pack_legacy": {
                "cnp_packbits": (
                    MISC_BIT_LEGACY_OWNER,
                    MISC_LIFETIME_OWNER,
                ),
                "cnp_unpackbits": (
                    MISC_BIT_LEGACY_OWNER,
                    MISC_LIFETIME_OWNER,
                ),
            },
            "bit_pack_v2": {
                "cnp_packbits_v2": (
                    MISC_BIT_V2_MATRIX_OWNER,
                    MISC_BIT_EMPTY_OWNER,
                    MISC_BIT_HUGE_ZERO_OWNER,
                    MISC_ERROR_OWNER,
                    MISC_LIFETIME_OWNER,
                    MISC_AHK_OWNER,
                ),
                "cnp_unpackbits_v2": (
                    MISC_BIT_V2_MATRIX_OWNER,
                    MISC_BIT_EMPTY_OWNER,
                    MISC_BIT_HUGE_ZERO_OWNER,
                    MISC_UNPACK_PADDING_EXTENSION_OWNER,
                    MISC_UNPACK_SIZE_OVERFLOW_OWNER,
                    MISC_ERROR_OWNER,
                    MISC_LIFETIME_OWNER,
                    MISC_AHK_OWNER,
                ),
            },
        }
        for family, owner_mapping in expected.items():
            actual = {
                item.symbol: item
                for item in self.manifest.exports
                if item.family == family
            }
            with self.subTest(family=family):
                self.assertEqual(set(owner_mapping), set(actual))
            for symbol, owners in owner_mapping.items():
                item = actual[symbol]
                with self.subTest(family=family, symbol=symbol):
                    expected_status = (
                        "characterized"
                        if family == "bit_pack_legacy"
                        else "differential"
                    )
                    self.assertEqual(expected_status, item.status)
                    self.assertEqual("array", item.result)
                    self.assertEqual(owners, item.tests)
                    self.assertIn(
                        "1.25.0"
                        if family != "misc_softmax"
                        else "1.12.0",
                        item.reference,
                    )
        unpack = next(
            item
            for item in self.manifest.exports
            if item.symbol == "cnp_unpackbits_v2"
        )
        self.assertIn("uninitialized", unpack.reference)
        self.assertIn("deterministic zero", unpack.reference)

    def test_recorded_declaration_count_matches_parser(self) -> None:
        self.assertEqual(
            len(self.declarations),
            self.manifest.declaration_count,
        )

    def test_every_export_has_a_semantic_owner(self) -> None:
        allowed_statuses = {
            "native",
            "characterized",
            "differential",
            "known_gap",
        }
        allowed_results = {"array", "multi_array", "scalar", "status", "void"}
        for item in self.manifest.exports:
            with self.subTest(symbol=item.symbol):
                self.assertTrue(item.family)
                self.assertIn(item.status, allowed_statuses)
                self.assertIn(item.result, allowed_results)
                self.assertTrue(item.tests)
                if item.status != "native":
                    self.assertIsNotNone(item.reference)
                    self.assertTrue(item.reference)

    def test_every_differential_export_has_a_direct_ahk_facade_owner(
        self,
    ) -> None:
        missing_by_family: dict[str, list[str]] = {}
        for item in self.manifest.exports:
            if item.status != "differential":
                continue
            if any(owner.startswith("ahk.numpy.test.") for owner in item.tests):
                continue
            missing_by_family.setdefault(item.family, []).append(item.symbol)

        self.assertEqual(
            {},
            missing_by_family,
            "differential exports are public NumPy surface and must name a "
            "direct AutoHotkey facade owner; internal/admin exports must be "
            "classified outside the differential surface",
        )

    def test_release_manifest_has_zero_known_gaps(self) -> None:
        gaps_by_family: dict[str, list[str]] = {}
        for item in self.manifest.exports:
            if item.status == "known_gap":
                gaps_by_family.setdefault(item.family, []).append(item.symbol)

        if gaps_by_family:
            lines = ["release manifest still contains known gaps:"]
            for family, symbols in sorted(gaps_by_family.items()):
                lines.append(f"{family} ({len(symbols)}):")
                lines.extend(f"  {symbol}" for symbol in sorted(symbols))
            self.fail("\n".join(lines))

    def test_array_surface_has_exact_direct_semantic_owners(self) -> None:
        python_owner = (
            "benchmark.tests.test_remaining_surface_semantics."
            "ArraySurfaceSemanticsTests"
        )
        ahk_owner = (
            "ahk.numpy.test.NumpyFoundationTest."
            "TestArraySurfaceFacadeSemantics"
        )
        differential_symbols = {
            "cnp_arange",
            "cnp_array_copy",
            "cnp_array_empty",
            "cnp_array_flat_get",
            "cnp_array_flat_set",
            "cnp_array_from_double_array",
            "cnp_array_from_float_array",
            "cnp_array_from_int_array",
            "cnp_array_from_scalar",
            "cnp_array_full",
            "cnp_array_get_double",
            "cnp_array_get_int",
            "cnp_array_new",
            "cnp_array_ones",
            "cnp_array_set_double",
            "cnp_array_set_int",
            "cnp_array_view",
            "cnp_array_zeros",
            "cnp_astype",
            "cnp_copy",
            "cnp_diag",
            "cnp_empty_like",
            "cnp_eye",
            "cnp_full_like",
            "cnp_geomspace",
            "cnp_identity",
            "cnp_linspace",
            "cnp_logspace",
            "cnp_ones_like",
            "cnp_tri",
            "cnp_zeros_like",
        }
        lifecycle_symbols = {"cnp_array_free", "cnp_array_incref"}
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family in {"array_surface_numpy", "array_lifecycle_admin"}
        }
        self.assertEqual(differential_symbols | lifecycle_symbols, set(actual))
        for symbol in differential_symbols:
            with self.subTest(symbol=symbol):
                item = actual[symbol]
                self.assertEqual("array_surface_numpy", item.family)
                self.assertEqual("differential", item.status)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertEqual((python_owner, ahk_owner), item.tests)
        for symbol in lifecycle_symbols:
            with self.subTest(symbol=symbol):
                item = actual[symbol]
                self.assertEqual("array_lifecycle_admin", item.family)
                self.assertEqual("characterized", item.status)
                self.assertIn("ownership", item.reference)
                self.assertEqual((python_owner, ahk_owner), item.tests)

    def test_dtype_descriptor_surface_has_exact_semantic_owners(self) -> None:
        python_owner = (
            "benchmark.tests.test_remaining_surface_semantics."
            "DtypeDescriptorSurfaceTests"
        )
        ahk_owner = (
            "ahk.numpy.test.NumpyFoundationTest.TestDtypeMetadataFacade"
        )
        differential_symbols = {
            "cnp_dtype_new",
            "cnp_dtype_from_char",
            "cnp_dtype_from_string",
        }
        lifecycle_symbols = {"cnp_dtype_incref", "cnp_dtype_decref"}
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family in {
                "dtype_descriptor_surface",
                "dtype_descriptor_lifecycle",
            }
        }
        self.assertEqual(differential_symbols | lifecycle_symbols, set(actual))
        for symbol in differential_symbols:
            with self.subTest(symbol=symbol):
                item = actual[symbol]
                self.assertEqual("differential", item.status)
                self.assertEqual((python_owner, ahk_owner), item.tests)
        for symbol in lifecycle_symbols:
            with self.subTest(symbol=symbol):
                item = actual[symbol]
                self.assertEqual("characterized", item.status)
                self.assertEqual((python_owner,), item.tests)

    def test_shape_surface_has_exact_semantic_owners(self) -> None:
        python_owner = (
            "benchmark.tests.test_shape_surface_semantics."
            "ShapeSurfaceSemanticsTests"
        )
        ahk_owner = (
            "ahk.numpy.test.NumpyFoundationTest.TestShapeSurfaceFacadeSemantics"
        )
        differential_symbols = {
            "cnp_append",
            "cnp_array_slice",
            "cnp_broadcast_to",
            "cnp_column_stack",
            "cnp_concatenate",
            "cnp_dstack",
            "cnp_expand_dims",
            "cnp_flatten",
            "cnp_flip",
            "cnp_hstack",
            "cnp_moveaxis",
            "cnp_pad",
            "cnp_ravel",
            "cnp_repeat",
            "cnp_reshape",
            "cnp_row_stack",
            "cnp_roll",
            "cnp_rot90",
            "cnp_squeeze",
            "cnp_stack",
            "cnp_swapaxes",
            "cnp_tile",
            "cnp_vstack",
        }
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family in {"shape_surface", "broadcast_arrays_legacy"}
        }
        self.assertEqual(differential_symbols | {"cnp_broadcast_arrays"}, set(actual))
        for symbol in differential_symbols:
            with self.subTest(symbol=symbol):
                item = actual[symbol]
                self.assertEqual("shape_surface", item.family)
                self.assertEqual("differential", item.status)
                self.assertEqual((python_owner, ahk_owner), item.tests)
        legacy = actual["cnp_broadcast_arrays"]
        self.assertEqual("broadcast_arrays_legacy", legacy.family)
        self.assertEqual("characterized", legacy.status)
        self.assertEqual((python_owner,), legacy.tests)

    def test_masked_surface_has_exact_semantic_owners(self) -> None:
        python_owner = (
            "benchmark.tests.test_masked_surface_semantics."
            "MaskedSurfaceSemanticsTests"
        )
        ahk_owner = (
            "ahk.numpy.test.NumpyFoundationTest.TestMaskedArrayFacadeSemantics"
        )
        array_symbols = {
            "cnp_masked_array_compressed",
            "cnp_masked_array_create",
            "cnp_masked_array_filled",
            "cnp_masked_array_get_data",
            "cnp_masked_array_get_mask",
            "cnp_masked_equal",
            "cnp_masked_greater",
            "cnp_masked_inside",
            "cnp_masked_invalid",
            "cnp_masked_less",
            "cnp_masked_not_equal",
            "cnp_masked_outside",
            "cnp_masked_where",
        }
        scalar_symbols = {
            "cnp_masked_array_count",
            "cnp_masked_array_max",
            "cnp_masked_array_mean",
            "cnp_masked_array_min",
            "cnp_masked_array_std",
            "cnp_masked_array_sum",
        }
        status_symbols = {"cnp_masked_array_set_mask"}
        lifecycle_symbols = {"cnp_masked_array_free"}
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family in {"masked_array_surface", "masked_array_lifecycle"}
        }
        expected_symbols = (
            array_symbols | scalar_symbols | status_symbols | lifecycle_symbols
        )
        self.assertEqual(expected_symbols, set(actual))
        for symbol in array_symbols | scalar_symbols | status_symbols:
            with self.subTest(symbol=symbol):
                item = actual[symbol]
                self.assertEqual("masked_array_surface", item.family)
                self.assertEqual("differential", item.status)
                self.assertEqual((python_owner, ahk_owner), item.tests)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("explicit errors", item.reference)
                self.assertIn("lifetime", item.reference)
        for symbol in array_symbols:
            self.assertEqual("array", actual[symbol].result)
        for symbol in scalar_symbols:
            self.assertEqual("scalar", actual[symbol].result)
        for symbol in status_symbols:
            self.assertEqual("status", actual[symbol].result)
        lifecycle = actual["cnp_masked_array_free"]
        self.assertEqual("masked_array_lifecycle", lifecycle.family)
        self.assertEqual("characterized", lifecycle.status)
        self.assertEqual("void", lifecycle.result)
        self.assertEqual((python_owner, ahk_owner), lifecycle.tests)
        self.assertIn("ownership", lifecycle.reference)

    def test_window_surface_has_exact_semantic_owners(self) -> None:
        symbols = {
            "cnp_bartlett",
            "cnp_blackman",
            "cnp_hamming",
            "cnp_hanning",
            "cnp_kaiser",
        }
        owners = (
            "benchmark.tests.test_window_financial_semantics."
            "WindowSurfaceSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestWindowFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "window_surface"
        }
        self.assertEqual(symbols, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("lifetime", item.reference)

    def test_window_mutation_surface_has_exact_semantic_owners(self) -> None:
        expected_results = {
            "cnp_choose": "array",
            "cnp_place": "status",
            "cnp_put": "status",
            "cnp_putmask": "status",
        }
        owners = (
            "benchmark.tests.test_window_financial_semantics."
            "WindowMutationSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestWindowMutationFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "window_mutation_surface"
        }
        self.assertEqual(set(expected_results), set(actual))
        for symbol, result in expected_results.items():
            with self.subTest(symbol=symbol):
                item = actual[symbol]
                self.assertEqual("differential", item.status)
                self.assertEqual(result, item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("atomic", item.reference)
                self.assertIn("lifetime", item.reference)

    def test_matrix_surface_has_exact_semantic_owners(self) -> None:
        symbols = {
            "cnp_mat",
            "cnp_matlib_eye",
            "cnp_matlib_ones",
            "cnp_matlib_repmat",
            "cnp_matlib_zeros",
        }
        owners = (
            "benchmark.tests.test_matrix_surface_semantics."
            "MatrixSurfaceSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestMatrixCompatibilityFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "matrix_surface"
        }
        self.assertEqual(symbols, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("explicit errors", item.reference)
                self.assertIn("lifetime", item.reference)

    def test_matlib_random_aliases_have_exact_characterized_owners(self) -> None:
        symbols = {"cnp_matlib_rand", "cnp_matlib_randn"}
        owners = (
            "benchmark.tests.test_matrix_surface_semantics."
            "MatlibRandomSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestMatrixCompatibilityFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "matlib_random_alias"
        }
        self.assertEqual(symbols, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("characterized", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(owners, item.tests)

    def test_array_relation_surface_has_exact_semantic_owners(self) -> None:
        expected_results = {
            "cnp_array_equal": "scalar",
            "cnp_array_equiv": "scalar",
            "cnp_byte_bounds": "status",
            "cnp_fliplr": "array",
            "cnp_flipud": "array",
            "cnp_isclose": "array",
            "cnp_may_share_memory": "scalar",
            "cnp_shares_memory": "scalar",
        }
        owners = (
            "benchmark.tests.test_array_relation_surface_semantics."
            "ArrayRelationSurfaceSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestArrayRelationSurfaceFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "array_relation_surface"
        }
        self.assertEqual(set(expected_results), set(actual))
        for symbol, result in expected_results.items():
            with self.subTest(symbol=symbol):
                item = actual[symbol]
                self.assertEqual("differential", item.status)
                self.assertEqual(result, item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("explicit errors", item.reference)
                self.assertIn("lifetime", item.reference)

    def test_array_layout_surface_has_exact_semantic_owners(self) -> None:
        symbols = {
            "cnp_asarray_chkfinite",
            "cnp_ascontiguousarray",
            "cnp_asfortranarray",
            "cnp_require",
            "cnp_resize",
        }
        owners = (
            "benchmark.tests.test_array_layout_surface_semantics."
            "ArrayLayoutSurfaceSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestArrayLayoutSurfaceFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "array_layout_surface"
        }
        self.assertEqual(symbols, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("explicit errors", item.reference)
                self.assertIn("lifetime", item.reference)

    def test_index_surface_has_exact_semantic_owners(self) -> None:
        expected_results = {
            "cnp_argwhere": "array",
            "cnp_array_boolean_index": "array",
            "cnp_array_fancy_index": "array",
            "cnp_array_getitem": "array",
            "cnp_array_nonzero": "array",
            "cnp_array_where": "array",
            "cnp_count_nonzero": "scalar",
            "cnp_count_nonzero_v2": "array",
            "cnp_flatnonzero": "array",
        }
        owners = (
            "benchmark.tests.test_index_surface_semantics."
            "IndexSurfaceSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestIndexSurfaceFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "index_surface"
        }
        self.assertEqual(set(expected_results), set(actual))
        for symbol, result in expected_results.items():
            with self.subTest(symbol=symbol):
                item = actual[symbol]
                self.assertEqual("differential", item.status)
                self.assertEqual(result, item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("explicit errors", item.reference)
                self.assertIn("lifetime", item.reference)

    def test_discrete_utility_surface_has_exact_semantic_owners(self) -> None:
        symbols = {
            "cnp_bincount",
            "cnp_ediff1d",
            "cnp_ravel_multi_index",
            "cnp_sinc",
            "cnp_tril_indices",
            "cnp_triu_indices",
            "cnp_unravel_index",
        }
        owners = (
            "benchmark.tests.test_discrete_utility_surface_semantics."
            "DiscreteUtilitySurfaceSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestDiscreteUtilitySurfaceFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "discrete_utility_surface"
        }
        self.assertEqual(symbols, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("explicit errors", item.reference)
                self.assertIn("lifetime", item.reference)

    def test_structural_utility_surface_has_exact_semantic_owners(self) -> None:
        expected_results = {
            "cnp_diagflat": "array",
            "cnp_diagonal": "array",
            "cnp_extract": "array",
            "cnp_fill_diagonal": "status",
            "cnp_nonzero": "array",
            "cnp_tril": "array",
            "cnp_trim_zeros": "array",
            "cnp_triu": "array",
        }
        owners = (
            "benchmark.tests.test_structural_utility_surface_semantics."
            "StructuralUtilitySurfaceSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestStructuralUtilitySurfaceFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "structural_utility_surface"
        }
        self.assertEqual(set(expected_results), set(actual))
        for symbol, result in expected_results.items():
            with self.subTest(symbol=symbol):
                item = actual[symbol]
                self.assertEqual("differential", item.status)
                self.assertEqual(result, item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("explicit errors", item.reference)
                self.assertIn("lifetime", item.reference)

    def test_stride_view_surface_has_exact_semantic_owners(self) -> None:
        symbols = {
            "cnp_as_strided",
            "cnp_rollaxis",
            "cnp_sliding_window_view",
        }
        owners = (
            "benchmark.tests.test_stride_view_surface_semantics."
            "StrideViewSurfaceSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestStrideViewSurfaceFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "stride_view_surface"
        }
        self.assertEqual(symbols, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("explicit errors", item.reference)
                self.assertIn("lifetime", item.reference)

    def test_assertion_surface_has_exact_semantic_owners(self) -> None:
        symbols = {
            "cnp_assert_allclose",
            "cnp_assert_array_almost_equal",
            "cnp_assert_array_equal",
        }
        owners = (
            "benchmark.tests.test_assertion_surface_semantics."
            "AssertionSurfaceSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestAssertionSurfaceFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "assertion_surface"
        }
        self.assertEqual(symbols, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual("scalar", item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("explicit errors", item.reference)
                self.assertIn("lifetime", item.reference)

    def test_runtime_utility_surface_has_exact_semantic_owners(self) -> None:
        expected_results = {
            "cnp_common_type": "scalar",
            "cnp_format_float": "status",
            "cnp_get_printoptions": "void",
            "cnp_getbufsize": "scalar",
            "cnp_geterr": "void",
            "cnp_min_scalar_type": "scalar",
            "cnp_set_printoptions": "void",
            "cnp_setbufsize": "status",
            "cnp_seterr": "void",
        }
        owners = (
            "benchmark.tests.test_runtime_utility_surface_semantics."
            "RuntimeUtilitySurfaceSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestRuntimeUtilitySurfaceFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "runtime_utility_surface"
        }
        self.assertEqual(set(expected_results), set(actual))
        for symbol, result in expected_results.items():
            with self.subTest(symbol=symbol):
                item = actual[symbol]
                self.assertEqual("differential", item.status)
                self.assertEqual(result, item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("explicit errors", item.reference)
                self.assertIn("lifetime", item.reference)

    def test_emath_surface_has_exact_semantic_owners(self) -> None:
        symbols = {
            "cnp_emath_arccos",
            "cnp_emath_arcsin",
            "cnp_emath_arctanh",
            "cnp_emath_log",
            "cnp_emath_log10",
            "cnp_emath_log2",
            "cnp_emath_power",
            "cnp_emath_sqrt",
        }
        owners = (
            "benchmark.tests.test_emath_surface_semantics."
            "EmathSurfaceSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestEmathSurfaceFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "emath_surface"
        }
        self.assertEqual(symbols, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("explicit errors", item.reference)
                self.assertIn("lifetime", item.reference)

    def test_einsum_alias_surface_has_exact_semantic_owners(self) -> None:
        expected_results = {
            "cnp_einsum_diag": "array",
            "cnp_einsum_dot": "scalar",
            "cnp_einsum_matmul": "array",
            "cnp_einsum_matvec": "array",
            "cnp_einsum_outer": "array",
            "cnp_einsum_sum": "scalar",
            "cnp_einsum_trace": "scalar",
            "cnp_einsum_transpose": "array",
        }
        owners = (
            "benchmark.tests.test_einsum_alias_surface_semantics."
            "EinsumAliasSurfaceSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestEinsumAliasSurfaceFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "einsum_alias_surface"
        }
        self.assertEqual(set(expected_results), set(actual))
        for symbol, result in expected_results.items():
            with self.subTest(symbol=symbol):
                item = actual[symbol]
                self.assertEqual("differential", item.status)
                self.assertEqual(result, item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("explicit errors", item.reference)
                self.assertIn("lifetime", item.reference)

    def test_product_linalg_surface_has_exact_differential_owners(self) -> None:
        expected_results = {
            "cnp_cross": "array",
            "cnp_dot": "array",
            "cnp_dot_1d": "array",
            "cnp_dot_general": "array",
            "cnp_inner": "array",
            "cnp_kron": "array",
            "cnp_matmul": "array",
            "cnp_multi_dot": "array",
            "cnp_outer": "array",
            "cnp_tensordot": "array",
            "cnp_tensordot_default": "array",
            "cnp_trace": "array",
            "cnp_trace_ext": "scalar",
            "cnp_vdot": "scalar",
        }
        owners = (
            "benchmark.tests.test_product_linalg_surface_semantics."
            "ProductLinalgSurfaceTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestProductLinalgSurfaceFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "product_linalg_surface"
        }
        self.assertEqual(set(expected_results), set(actual))
        for symbol, result in expected_results.items():
            with self.subTest(symbol=symbol):
                item = actual[symbol]
                self.assertEqual("differential", item.status)
                self.assertEqual(result, item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("explicit errors", item.reference)
                self.assertIn("lifetime", item.reference)

    def test_linalg_statistics_surface_has_exact_differential_owners(
        self,
    ) -> None:
        expected_results = {
            "cnp_linalg_inv": "array",
            "cnp_linalg_norm": "array",
            "cnp_linalg_norm_ext": "scalar",
            "cnp_linalg_pinv": "array",
            "cnp_linalg_qr": "multi_array",
            "cnp_linalg_tensorinv": "array",
            "cnp_linalg_tensorsolve": "array",
            "cnp_linalg_tensorsolve_v2": "array",
            "cnp_pinv": "array",
            "cnp_corrcoef": "array",
            "cnp_cov": "array",
        }
        owners = (
            "benchmark.tests.test_linalg_statistics_surface_semantics."
            "LinalgStatisticsSurfaceTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestLinalgStatisticsSurfaceFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "linalg_statistics_surface"
        }
        self.assertEqual(set(expected_results), set(actual))
        for symbol, result in expected_results.items():
            with self.subTest(symbol=symbol):
                item = actual[symbol]
                self.assertEqual("differential", item.status)
                self.assertEqual(result, item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("explicit errors", item.reference)
                self.assertIn("lifetime", item.reference)

    def test_random_core_surface_has_exact_semantic_owners(self) -> None:
        symbols = {
            "cnp_random_beta",
            "cnp_random_binomial",
            "cnp_random_exponential",
            "cnp_random_gamma",
            "cnp_random_integers",
            "cnp_random_normal",
            "cnp_random_poisson",
            "cnp_random_randint",
            "cnp_random_random",
            "cnp_random_standard_normal",
            "cnp_random_uniform",
        }
        owners = (
            "benchmark.tests.test_random_core_surface_semantics."
            "RandomCoreSurfaceSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestRandomCoreSurfaceFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "random_core_surface"
        }
        self.assertEqual(symbols, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("explicit errors", item.reference)
                self.assertIn("lifetime", item.reference)
                self.assertIn("deterministic replay", item.reference)

    def test_random_extended_surface_has_exact_semantic_owners(self) -> None:
        symbols = {
            "cnp_random_dirichlet",
            "cnp_random_f",
            "cnp_random_gumbel",
            "cnp_random_laplace",
            "cnp_random_logistic",
            "cnp_random_logseries",
            "cnp_random_multinomial",
            "cnp_random_negative_binomial",
            "cnp_random_noncentral_chisquare",
            "cnp_random_noncentral_f",
            "cnp_random_pareto",
            "cnp_random_power",
            "cnp_random_rayleigh",
            "cnp_random_standard_cauchy",
            "cnp_random_standard_t",
            "cnp_random_triangular",
            "cnp_random_vonmises",
            "cnp_random_weibull",
        }
        owners = (
            "benchmark.tests.test_random_extended_surface_semantics."
            "RandomExtendedSurfaceSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestRandomExtendedSurfaceFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "random_extended_surface"
        }
        self.assertEqual(symbols, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("explicit errors", item.reference)
                self.assertIn("lifetime", item.reference)
                self.assertIn("deterministic replay", item.reference)

    def test_random_special_surface_has_exact_semantic_owners(self) -> None:
        expected_results = {
            "cnp_random_bytes": "scalar",
            "cnp_random_bytes_free": "void",
            "cnp_random_chisquare": "array",
            "cnp_random_geometric": "array",
            "cnp_random_hypergeometric": "array",
            "cnp_random_multivariate_normal": "array",
            "cnp_random_wald": "array",
            "cnp_random_zipf": "array",
        }
        owners = (
            "benchmark.tests.test_random_special_surface_semantics."
            "RandomSpecialSurfaceSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestRandomSpecialSurfaceFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "random_special_surface"
        }
        self.assertEqual(set(expected_results), set(actual))
        for symbol, result in expected_results.items():
            with self.subTest(symbol=symbol):
                item = actual[symbol]
                self.assertEqual("differential", item.status)
                self.assertEqual(result, item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("explicit errors", item.reference)
                self.assertIn("lifetime", item.reference)
                self.assertIn("deterministic replay", item.reference)

    def test_legacy_financial_surface_has_exact_semantic_owners(self) -> None:
        symbols = {
            "cnp_fv",
            "cnp_pv",
            "cnp_pmt",
            "cnp_nper",
            "cnp_rate",
            "cnp_npv",
            "cnp_irr",
        }
        owners = (
            "benchmark.tests.test_window_financial_semantics."
            "LegacyFinancialSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestLegacyFinancialFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "legacy_financial_extension"
        }
        self.assertEqual(symbols, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("characterized", item.status)
                self.assertEqual("scalar", item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("removed", item.reference)
                self.assertIn("explicit errors", item.reference)

    def test_manifest_does_not_duplicate_symbols(self) -> None:
        symbols = [item.symbol for item in self.manifest.exports]
        duplicates = sorted(
            symbol for symbol in set(symbols) if symbols.count(symbol) > 1
        )
        self.assertEqual([], duplicates)

    def test_cholesky_has_an_exact_differential_semantic_owner(self) -> None:
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "linalg_cholesky"
        }
        self.assertEqual({"cnp_linalg_cholesky"}, set(actual))
        item = actual["cnp_linalg_cholesky"]
        self.assertEqual("differential", item.status)
        self.assertEqual("status", item.result)
        for phrase in (
            "NumPy 1.25.0",
            "dtype",
            "batch",
            "lower",
            "Hermitian",
            "stride",
            "nonfinite",
            "atomic",
            "lifetime",
            "AutoHotkey v2",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, item.reference)
        self.assertEqual(
            (
                "benchmark.tests.test_linalg_semantics."
                "LinalgSemanticsTests",
                "ahk.numpy.test.NumpyFoundationTest."
                "TestCholeskyFacadeV2",
            ),
            item.tests,
        )

    def test_task9_exports_have_exact_direct_owners_and_characterized_exclusions(
        self,
    ) -> None:
        prefix = (
            "benchmark.tests.test_linalg_semantics."
            "LinalgSemanticsTests."
        )
        einsum_tests = tuple(
            prefix + method
            for method in (
                "test_einsum_explicit_and_implicit_outputs_match_numpy",
                "test_einsum_repeated_labels_diagonals_and_reductions",
                "test_einsum_ellipsis_scalar_and_broadcasting_match_numpy",
                "test_einsum_preserves_views_and_numpy_dtype_promotion",
                "test_einsum_float16_forms_promotion_rounding_and_lifetime",
                "test_einsum_fast_patterns_respect_named_label_broadcasting",
                "test_einsum_invalid_subscripts_shapes_and_nulls_are_explicit",
            )
        ) + ("ahk.numpy.test.NumpyFoundationTest.TestEinsumFacadeV2",)
        eig_tests = tuple(
            prefix + method
            for method in (
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
            )
        ) + (
            "ahk.numpy.test.NumpyFoundationTest.TestGeneralEigFacadeV2",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestLinalgSpectralDelegatesV2",
        )
        eigvals_tests = tuple(
            prefix + method
            for method in (
                "test_general_eig_returns_complex_eigenpairs_for_real_matrix",
                "test_general_eig_preserves_real_dtype_for_real_spectrum",
                "test_general_eig_supports_batched_matrices_and_owned_results",
                "test_general_eig_dtype_promotion_matches_numpy",
                "test_general_eig_dense_nonsymmetric_noncontiguous_view",
                "test_general_eig_seeded_dense_differential",
                "test_general_eig_repeated_and_defective_spectra",
                "test_eigvals_wrapper_inherits_general_semantics_and_owns_errors",
                "test_general_eig_zero_sized_matrix_and_batch_shapes",
            )
        ) + (
            "ahk.numpy.test.NumpyFoundationTest.TestEigvalsFacadeV2",
        )
        svd_tests = tuple(
            prefix + method
            for method in (
                "test_svd_workspace_products_are_checked_before_allocation",
                "test_svd_legacy_default_returns_complete_rectangular_factors",
                "test_svd_v2_reduced_tall_wide_and_batched_shapes",
                "test_svd_v2_dtype_promotion_complex_and_unitarity",
                "test_svd_v2_reads_noncontiguous_complex_view",
                "test_svd_v2_compute_uv_false_and_zero_sized_shapes",
                "test_svd_v2_complete_wide_and_hermitian_factors",
                "test_svd_v2_seeded_dense_and_rank_deficient_differential",
                "test_svd_v2_validation_is_explicit_and_atomic",
            )
        ) + (
            "ahk.numpy.test.NumpyFoundationTest.TestSvdFacadeV2",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestLinalgSpectralDelegatesV2",
        )
        solve_tests = tuple(
            prefix + method
            for method in (
                "test_solve_square_batched_rhs_dtypes_and_lifetimes_match_numpy_125",
                "test_solve_zero_batch_broadcasts_without_reading_empty_sources",
                "test_solve_singular_failure_is_explicit_atomic_and_nonretaining",
            )
        ) + (
            "ahk.numpy.test.NumpyFoundationTest."
            "TestTask9SolveLstsqAndCondFacadeV2",
        )
        lstsq_tests = tuple(
            prefix + method
            for method in (
                "test_lstsq_v2_rectangular_outputs_rcond_and_lifetimes_match_numpy_125",
                "test_lstsq_v2_numpy_125_rcond_boundary_values",
                "test_lstsq_v2_and_cond_v2_validation_is_explicit_atomic_and_retained0",
            )
        ) + (
            "ahk.numpy.test.NumpyFoundationTest."
            "TestTask9SolveLstsqAndCondFacadeV2",
        )
        exact = {
            "cnp_einsum": ("array", einsum_tests),
            "cnp_linalg_eig": ("multi_array", eig_tests),
            "cnp_eigvals": ("array", eigvals_tests),
            "cnp_linalg_svd": ("multi_array", svd_tests),
            "cnp_linalg_svd_v2": ("multi_array", svd_tests),
            "cnp_linalg_solve": ("status", solve_tests),
            "cnp_linalg_lstsq_v2": ("multi_array", lstsq_tests),
        }
        by_symbol = {item.symbol: item for item in self.manifest.exports}
        for symbol, (result, tests) in exact.items():
            item = by_symbol[symbol]
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual(result, item.result)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertNotIn(
                    "benchmark.tests.test_linalg_semantics."
                    "LinalgSemanticsTests",
                    item.tests,
                )
                self.assertEqual(tests, item.tests)
                self.assertEqual(len(item.tests), len(set(item.tests)))
                for owner in item.tests:
                    if not owner.startswith("benchmark."):
                        continue
                    module_name, class_name, method_name = owner.rsplit(".", 2)
                    module = importlib.import_module(module_name)
                    owner_class = getattr(module, class_name)
                    self.assertTrue(callable(getattr(owner_class, method_name)))

        characterized = {
            "cnp_lstsq",
            "cnp_linalg_cond_v2",
            "cnp_linalg_cond",
        }
        for symbol in characterized:
            with self.subTest(characterized=symbol):
                self.assertEqual("characterized", by_symbol[symbol].status)

    def test_atleast_nd_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_shape_semantics."
            "AtleastNdSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestAtleastNdFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "shape_atleast_nd"
        }
        self.assertEqual(
            {"cnp_atleast_1d", "cnp_atleast_2d", "cnp_atleast_3d"},
            set(actual),
        )
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("single-array", item.reference)
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(expected_tests, item.tests)

    def test_legacy_indexing_abis_are_explicitly_characterized(self) -> None:
        owner = (
            "benchmark.tests.test_index_semantics."
            "IndexMutationSemanticsTests."
            "test_legacy_indexing_exports_preserve_documented_axis_minus_one_abi"
        )
        expected_symbols = {
            "cnp_array_take",
            "cnp_take",
            "cnp_take_along_axis",
            "cnp_compress",
            "cnp_delete",
            "cnp_insert",
        }
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "indexing_legacy"
        }

        self.assertEqual(expected_symbols, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("characterized", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual((owner,), item.tests)

    def test_partition_abis_have_exact_semantic_owners(self) -> None:
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family in {
                "sorting_partition",
                "sorting_partition_legacy",
            }
        }
        self.assertEqual(
            {
                "cnp_partition",
                "cnp_argpartition",
                "cnp_partition_v2",
                "cnp_argpartition_v2",
            },
            set(actual),
        )
        legacy_owner = (
            "benchmark.tests.test_sort_set_semantics."
            "PartitionSemanticsTests."
            "test_legacy_partition_abis_preserve_axis_minus_one_flatten_sentinel"
        )
        for symbol in ("cnp_partition", "cnp_argpartition"):
            with self.subTest(symbol=symbol):
                item = actual[symbol]
                self.assertEqual("sorting_partition_legacy", item.family)
                self.assertEqual("characterized", item.status)
                self.assertEqual((legacy_owner,), item.tests)

        differential_owners = (
            "benchmark.tests.test_sort_set_semantics."
            "PartitionSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestPartitionFacadeV2",
        )
        for symbol in ("cnp_partition_v2", "cnp_argpartition_v2"):
            with self.subTest(symbol=symbol):
                item = actual[symbol]
                self.assertEqual("sorting_partition", item.family)
                self.assertEqual("differential", item.status)
                self.assertEqual(differential_owners, item.tests)

    def test_searchsorted_abis_have_differential_semantic_owners(
        self,
    ) -> None:
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "sorting_searchsorted"
        }
        self.assertEqual(
            {"cnp_searchsorted", "cnp_searchsorted_v2"}, set(actual)
        )
        owners = (
            "benchmark.tests.test_sort_set_semantics."
            "SearchsortedSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestSearchsortedFacadeV2",
        )
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(owners, item.tests)

    def test_digitize_has_differential_semantic_owners(self) -> None:
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "sorting_digitize"
        }
        self.assertEqual({"cnp_digitize"}, set(actual))
        item = actual["cnp_digitize"]
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(
            (
                "benchmark.tests.test_sort_set_semantics."
                "DigitizeSemanticsTests",
                "ahk.numpy.test.NumpyFoundationTest."
                "TestDigitizeFacadeSemantics",
            ),
            item.tests,
        )

    def test_lexsort_abis_have_differential_semantic_owners(self) -> None:
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "sorting_lexsort"
        }
        self.assertEqual({"cnp_lexsort", "cnp_lexsort_v2"}, set(actual))
        owners = (
            "benchmark.tests.test_sort_set_semantics."
            "LexsortSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestLexsortFacadeSemantics",
        )
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(owners, item.tests)

    def test_msort_and_sort_complex_have_differential_semantic_owners(
        self,
    ) -> None:
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "sorting_msort_complex"
        }
        self.assertEqual({"cnp_msort", "cnp_sort_complex"}, set(actual))
        owners = (
            "benchmark.tests.test_sort_set_semantics."
            "MsortAndSortComplexSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestMsortAndSortComplexFacadeSemantics",
        )
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(owners, item.tests)

    def test_effective_reduction_mapping_tracks_all_differential_array_abis(
        self,
    ) -> None:
        expected_symbols = {
            symbol
            for symbols, _ in REDUCTION_OWNER_GROUPS
            for symbol in symbols
        }
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "reduction"
        }
        self.assertEqual(expected_symbols, set(actual))
        for symbols, expected_owners in REDUCTION_OWNER_GROUPS:
            for symbol in symbols:
                with self.subTest(symbol=symbol):
                    item = actual[symbol]
                    self.assertEqual("differential", item.status)
                    self.assertEqual("array", item.result)
                    self.assertEqual(expected_owners, item.tests)

    def test_average_has_numpy_and_ahk_differential_owners(self) -> None:
        expected_tests = {
            "cnp_average": (
                LEGACY_REDUCTION_MATRIX_OWNER,
                LEGACY_REDUCTION_ERROR_OWNER,
                LEGACY_REDUCTION_SENTINEL_OWNER,
                "ahk.numpy.test.NumpyFoundationTest."
                "TestAverageFacadeSemantics",
            ),
            "cnp_average_v2": (
                AVERAGE_V2_MATRIX_OWNER,
                "ahk.numpy.test.NumpyFoundationTest."
                "TestAverageFacadeSemantics",
            ),
        }
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "reduction_average"
        }
        self.assertEqual(set(expected_tests), set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("numpy.average", item.reference)
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(expected_tests[symbol], item.tests)

    def test_reduction_convenience_abis_are_explicitly_characterized(
        self,
    ) -> None:
        expected = {
            "reduction_into_legacy": (
                {
                    "cnp_cumsum_into": (
                        "benchmark.tests.test_reduction_semantics."
                        "ReductionSemanticsTests."
                        "test_float64_cumsum_preserves_numpy_left_to_right_rounding",
                        "ahk.numpy.test.NumpyFoundationTest."
                        "TestPreallocatedIntoAndBatchApis",
                        "ahk.numpy.test.NumpyFoundationTest."
                        "TestPreallocatedIntoRejectsInvalidInputsAndStopsBatch",
                    ),
                    "cnp_sum_into_scalar": (
                        "benchmark.tests.test_reduction_semantics."
                        "ReductionSemanticsTests."
                        "test_sum_into_scalar_matches_numpy_pairwise_tree_and_errors_atomically",
                        "ahk.numpy.test.NumpyFoundationTest."
                        "TestPreallocatedIntoAndBatchApis",
                    ),
                },
                "float64",
            ),
            "reduction_scalar_legacy": (
                {
                    "cnp_sum_scalar": (SCALAR_CONVENIENCE_MATRIX_OWNER,),
                    "cnp_prod_scalar": (SCALAR_CONVENIENCE_MATRIX_OWNER,),
                    "cnp_mean_scalar": (SCALAR_CONVENIENCE_MATRIX_OWNER,),
                },
                "double-return",
            ),
        }
        for family, (owner_mapping, reference_text) in expected.items():
            actual = {
                item.symbol: item
                for item in self.manifest.exports
                if item.family == family
            }
            with self.subTest(family=family):
                self.assertEqual(set(owner_mapping), set(actual))
            for symbol, item in actual.items():
                with self.subTest(family=family, symbol=symbol):
                    self.assertEqual("characterized", item.status)
                    self.assertIn(reference_text, item.reference)
                    self.assertEqual(owner_mapping[symbol], item.tests)

    def test_legacy_nan_scalar_abis_are_explicitly_characterized(self) -> None:
        statistic_owners = (
            LEGACY_NAN_SCALAR_MATRIX_OWNER,
            "benchmark.tests.test_reduction_semantics."
            "ReductionSemanticsTests."
            "test_legacy_nan_statistics_reject_unrepresentable_and_invalid_calls",
            "benchmark.tests.test_reduction_semantics."
            "ReductionSemanticsTests."
            "test_legacy_nan_statistics_return_scalar_numpy_values",
        )
        statistics = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "reduction_legacy_scalar"
        }
        self.assertEqual(
            {"cnp_nanmedian", "cnp_nanpercentile", "cnp_nanquantile"},
            set(statistics),
        )
        for symbol, item in statistics.items():
            with self.subTest(family="statistics", symbol=symbol):
                self.assertEqual("characterized", item.status)
                self.assertEqual("scalar", item.result)
                self.assertEqual(statistic_owners, item.tests)

        nanarg_owners = (
            LEGACY_NAN_SCALAR_MATRIX_OWNER,
            "benchmark.tests.test_reduction_semantics."
            "ReductionSemanticsTests."
            "test_legacy_nanarg_extrema_return_only_representable_scalar_results",
        )
        nanargs = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "reduction_legacy_nanarg"
        }
        self.assertEqual({"cnp_nanargmax", "cnp_nanargmin"}, set(nanargs))
        for symbol, item in nanargs.items():
            with self.subTest(family="nanarg", symbol=symbol):
                self.assertEqual("characterized", item.status)
                self.assertEqual("scalar", item.result)
                self.assertIn("axis == -1", item.reference)
                self.assertEqual(nanarg_owners, item.tests)

    def test_signal_products_have_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics."
            "ConvolveCorrelateSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestConvolveCorrelateFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_convolve_correlate"
        }
        self.assertEqual({"cnp_convolve", "cnp_correlate"}, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(expected_tests, item.tests)

    def test_subtract_and_multiply_have_differential_semantic_owners(
        self,
    ) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics."
            "SubtractMultiplySemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestSubtractMultiplyFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_subtract_multiply"
        }
        self.assertEqual({"cnp_subtract", "cnp_multiply"}, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(expected_tests, item.tests)

    def test_divide_aliases_have_differential_semantic_owners(
        self,
    ) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics."
            "DivideTrueDivideSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestDivideTrueDivideFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_divide_true_divide"
        }
        self.assertEqual({"cnp_divide", "cnp_true_divide"}, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(expected_tests, item.tests)

    def test_floor_divide_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics."
            "FloorDivideSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestFloorDivideFacadeSemantics",
        )
        item = next(
            export
            for export in self.manifest.exports
            if export.symbol == "cnp_floor_divide"
        )
        self.assertEqual("dtype_floor_divide", item.family)
        self.assertEqual("numpy.floor_divide", item.reference)
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_mod_and_remainder_have_differential_semantic_owners(
        self,
    ) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics."
            "ModRemainderSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestModRemainderFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_mod_remainder"
        }
        self.assertEqual({"cnp_mod", "cnp_remainder"}, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(expected_tests, item.tests)

    def test_divmod_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.DivmodSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestDivmodFacadeSemantics",
        )
        item = next(
            export
            for export in self.manifest.exports
            if export.symbol == "cnp_divmod"
        )
        self.assertEqual("dtype_divmod", item.family)
        self.assertIn("NumPy 1.25.0 divmod ufunc loops", item.reference)
        self.assertEqual("differential", item.status)
        self.assertEqual("multi_array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_array_predicates_have_differential_semantic_owners(
        self,
    ) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics."
            "ArrayPredicateSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestArrayPredicateFacadeSemantics",
        )
        finiteness = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_finiteness_predicates"
        }
        self.assertEqual(
            {
                "cnp_isnan",
                "cnp_isnan_arr",
                "cnp_isinf",
                "cnp_isinf_arr",
                "cnp_isfinite",
                "cnp_isfinite_arr",
            },
            set(finiteness),
        )
        for symbol, item in finiteness.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(expected_tests, item.tests)

        signbit = next(
            item
            for item in self.manifest.exports
            if item.symbol == "cnp_signbit"
        )
        self.assertEqual("dtype_signbit", signbit.family)
        self.assertEqual("differential", signbit.status)
        self.assertEqual("array", signbit.result)
        self.assertEqual(expected_tests, signbit.tests)

    def test_object_kind_predicates_have_differential_semantic_owners(
        self,
    ) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics."
            "ObjectKindPredicateSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestObjectKindPredicateFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_object_kind_predicates"
        }
        self.assertEqual(
            {"cnp_iscomplexobj", "cnp_isrealobj"}, set(actual)
        )
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertEqual("differential", item.status)
                self.assertEqual("scalar", item.result)
                self.assertEqual(expected_tests, item.tests)

    def test_isscalar_has_a_differential_semantic_owner(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.IsscalarSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestIsScalarFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "scalar_object_predicate"
        }
        self.assertEqual({"cnp_isscalar"}, set(actual))
        item = actual["cnp_isscalar"]
        self.assertIn("NumPy 1.25.0", item.reference)
        self.assertEqual("differential", item.status)
        self.assertEqual("scalar", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_array_metadata_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics."
            "ArrayMetadataSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestArrayMetadataFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "array_metadata"
        }
        self.assertEqual(
            {
                "cnp_array_nbytes",
                "cnp_nbytes",
                "cnp_array_is_c_contiguous",
                "cnp_array_is_f_contiguous",
            },
            set(actual),
        )
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertEqual("differential", item.status)
                self.assertEqual("scalar", item.result)
                self.assertEqual(expected_tests, item.tests)

    def test_fmod_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.FmodSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestFmodFacadeSemantics",
        )
        item = next(
            export
            for export in self.manifest.exports
            if export.symbol == "cnp_fmod"
        )
        self.assertEqual("dtype_fmod", item.family)
        self.assertEqual("numpy.fmod", item.reference)
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_negative_and_positive_have_differential_semantic_owners(
        self,
    ) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics."
            "NegativePositiveSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestNegativePositiveFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_negative_positive"
        }
        self.assertEqual({"cnp_negative", "cnp_positive"}, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(expected_tests, item.tests)

    def test_sign_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.SignSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestSignFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_sign"
        }
        self.assertEqual({"cnp_sign"}, set(actual))
        item = actual["cnp_sign"]
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_reciprocal_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics."
            "ReciprocalSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestReciprocalFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_reciprocal"
        }
        self.assertEqual({"cnp_reciprocal"}, set(actual))
        item = actual["cnp_reciprocal"]
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_square_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.SquareSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestSquareFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_square"
        }
        self.assertEqual({"cnp_square"}, set(actual))
        item = actual["cnp_square"]
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_sqrt_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.SqrtSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestSqrtFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_sqrt"
        }
        self.assertEqual({"cnp_sqrt"}, set(actual))
        item = actual["cnp_sqrt"]
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_cbrt_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.CbrtSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestCbrtFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_cbrt"
        }
        self.assertEqual({"cnp_cbrt"}, set(actual))
        item = actual["cnp_cbrt"]
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_conjugate_aliases_have_differential_semantic_owners(
        self,
    ) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.ConjugateSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestConjugateFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_conjugate"
        }
        self.assertEqual({"cnp_conj", "cnp_conjugate"}, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertIn(
                    "NumPy 1.25.0 conjugate/conj ufunc loops",
                    item.reference,
                )
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(expected_tests, item.tests)

    def test_cos_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.CosSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestCosFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_cos"
        }
        self.assertEqual({"cnp_cos"}, set(actual))
        item = actual["cnp_cos"]
        self.assertIn("NumPy 1.25.0 cos ufunc loops", item.reference)
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_sin_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.SinSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestSinFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_sin"
        }
        self.assertEqual({"cnp_sin"}, set(actual))
        item = actual["cnp_sin"]
        self.assertIn("NumPy 1.25.0 sin ufunc loops", item.reference)
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_tan_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.TanSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestTanFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_tan"
        }
        self.assertEqual({"cnp_tan"}, set(actual))
        item = actual["cnp_tan"]
        self.assertIn("NumPy 1.25.0 tan ufunc loops", item.reference)
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_arcsin_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.ArcsinSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestArcsinFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_arcsin"
        }
        self.assertEqual({"cnp_arcsin"}, set(actual))
        item = actual["cnp_arcsin"]
        self.assertIn("NumPy 1.25.0 arcsin ufunc loops", item.reference)
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_arccos_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.ArccosSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestArccosFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_arccos"
        }
        self.assertEqual({"cnp_arccos"}, set(actual))
        item = actual["cnp_arccos"]
        self.assertIn("NumPy 1.25.0 arccos ufunc loops", item.reference)
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_arctan_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.ArctanSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestArctanFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_arctan"
        }
        self.assertEqual({"cnp_arctan"}, set(actual))
        item = actual["cnp_arctan"]
        self.assertIn("NumPy 1.25.0 arctan ufunc loops", item.reference)
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_arctan2_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.Arctan2SemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestArctan2FacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_arctan2"
        }
        self.assertEqual({"cnp_arctan2"}, set(actual))
        item = actual["cnp_arctan2"]
        self.assertIn("NumPy 1.25.0 arctan2 ufunc loops", item.reference)
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_hypot_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.HypotSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestHypotFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_hypot"
        }
        self.assertEqual({"cnp_hypot"}, set(actual))
        item = actual["cnp_hypot"]
        self.assertIn("NumPy 1.25.0 hypot ufunc loops", item.reference)
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_angle_conversion_has_differential_semantic_owners(self) -> None:
        expected_symbols = {
            "cnp_degrees",
            "cnp_radians",
            "cnp_deg2rad",
            "cnp_rad2deg",
        }
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.AngleConversionSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestAngleConversionFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_angle_conversion"
        }
        self.assertEqual(expected_symbols, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertIn(
                    "NumPy 1.25.0 degrees/radians ufunc loops",
                    item.reference,
                )
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(expected_tests, item.tests)

    def test_sinh_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.SinhSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestSinhFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_sinh"
        }
        self.assertEqual({"cnp_sinh"}, set(actual))
        item = actual["cnp_sinh"]
        self.assertIn("NumPy 1.25.0 sinh ufunc loops", item.reference)
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_cosh_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.CoshSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestCoshFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_cosh"
        }
        self.assertEqual({"cnp_cosh"}, set(actual))
        item = actual["cnp_cosh"]
        self.assertIn("NumPy 1.25.0 cosh ufunc loops", item.reference)
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_tanh_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.TanhSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestTanhFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_tanh"
        }
        self.assertEqual({"cnp_tanh"}, set(actual))
        item = actual["cnp_tanh"]
        self.assertIn("NumPy 1.25.0 tanh ufunc loops", item.reference)
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_arcsinh_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.ArcsinhSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestArcsinhFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_arcsinh"
        }
        self.assertEqual({"cnp_arcsinh"}, set(actual))
        item = actual["cnp_arcsinh"]
        self.assertIn("NumPy 1.25.0 arcsinh ufunc loops", item.reference)
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_arccosh_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.ArccoshSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestArccoshFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_arccosh"
        }
        self.assertEqual({"cnp_arccosh"}, set(actual))
        item = actual["cnp_arccosh"]
        self.assertIn("NumPy 1.25.0 arccosh ufunc loops", item.reference)
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_arctanh_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.ArctanhSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestArctanhFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_arctanh"
        }
        self.assertEqual({"cnp_arctanh"}, set(actual))
        item = actual["cnp_arctanh"]
        self.assertIn("NumPy 1.25.0 arctanh ufunc loops", item.reference)
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_exp_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.ExpSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestExpFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_exp"
        }
        self.assertEqual({"cnp_exp"}, set(actual))
        item = actual["cnp_exp"]
        self.assertIn("NumPy 1.25.0 exp ufunc loops", item.reference)
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_exp2_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.Exp2SemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestExp2FacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_exp2"
        }
        self.assertEqual({"cnp_exp2"}, set(actual))
        item = actual["cnp_exp2"]
        self.assertIn("NumPy 1.25.0 exp2 ufunc loops", item.reference)
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_expm1_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.Expm1SemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestExpm1FacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_expm1"
        }
        self.assertEqual({"cnp_expm1"}, set(actual))
        item = actual["cnp_expm1"]
        self.assertIn("NumPy 1.25.0 expm1 ufunc loops", item.reference)
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_log_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.LogSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestLogFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_log"
        }
        self.assertEqual({"cnp_log"}, set(actual))
        item = actual["cnp_log"]
        self.assertIn("NumPy 1.25.0 log ufunc loops", item.reference)
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_log2_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.Log2SemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestLog2FacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_log2"
        }
        self.assertEqual({"cnp_log2"}, set(actual))
        item = actual["cnp_log2"]
        self.assertIn("NumPy 1.25.0 log2 ufunc loops", item.reference)
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_log10_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.Log10SemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestLog10FacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_log10"
        }
        self.assertEqual({"cnp_log10"}, set(actual))
        item = actual["cnp_log10"]
        self.assertIn("NumPy 1.25.0 log10 ufunc loops", item.reference)
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_log1p_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.Log1pSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestLog1pFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_log1p"
        }
        self.assertEqual({"cnp_log1p"}, set(actual))
        item = actual["cnp_log1p"]
        self.assertIn("NumPy 1.25.0 log1p ufunc loops", item.reference)
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_logaddexp_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.LogaddexpSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestLogaddexpFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_logaddexp"
        }
        self.assertEqual({"cnp_logaddexp"}, set(actual))
        item = actual["cnp_logaddexp"]
        self.assertIn(
            "NumPy 1.25.0 logaddexp ufunc loops", item.reference
        )
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_logaddexp2_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.Logaddexp2SemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestLogaddexp2FacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_logaddexp2"
        }
        self.assertEqual({"cnp_logaddexp2"}, set(actual))
        item = actual["cnp_logaddexp2"]
        self.assertIn(
            "NumPy 1.25.0 logaddexp2 ufunc loops", item.reference
        )
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_comparisons_have_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.ComparisonSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestComparisonFacadeSemantics",
        )
        expected_symbols = {
            "cnp_equal",
            "cnp_not_equal",
            "cnp_less",
            "cnp_less_equal",
            "cnp_greater",
            "cnp_greater_equal",
        }
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_comparison"
        }
        self.assertEqual(expected_symbols, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(expected_tests, item.tests)

    def test_extrema_have_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics."
            "MaximumMinimumSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestExtremaFacadeSemantics",
        )
        expected_symbols = {
            "cnp_maximum",
            "cnp_minimum",
            "cnp_fmax",
            "cnp_fmin",
        }
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_extrema"
        }
        self.assertEqual(expected_symbols, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(expected_tests, item.tests)

    def test_logical_operations_have_differential_semantic_owners(
        self,
    ) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.LogicalSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestLogicalFacadeSemantics",
        )
        expected_symbols = {
            "cnp_logical_and",
            "cnp_logical_or",
            "cnp_logical_xor",
            "cnp_logical_not",
        }
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_logical"
        }
        self.assertEqual(expected_symbols, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(expected_tests, item.tests)

    def test_bitwise_operations_have_differential_semantic_owners(
        self,
    ) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.BitwiseSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestBitwiseFacadeSemantics",
        )
        expected_symbols = {
            "cnp_bitwise_and",
            "cnp_bitwise_or",
            "cnp_bitwise_xor",
            "cnp_invert",
            "cnp_bitwise_not",
            "cnp_left_shift",
            "cnp_right_shift",
        }
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_bitwise"
        }
        self.assertEqual(expected_symbols, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(expected_tests, item.tests)

    def test_power_operations_have_differential_semantic_owners(
        self,
    ) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.PowerSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestPowerFacadeSemantics",
        )
        expected_symbols = {"cnp_power", "cnp_float_power"}
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_power"
        }
        self.assertEqual(expected_symbols, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(expected_tests, item.tests)

    def test_heaviside_has_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.HeavisideSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestHeavisideFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_heaviside"
        }
        self.assertEqual({"cnp_heaviside"}, set(actual))
        item = actual["cnp_heaviside"]
        self.assertIn("NumPy 1.25.0", item.reference)
        self.assertEqual("differential", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual(expected_tests, item.tests)

    def test_gcd_lcm_have_differential_semantic_owners(self) -> None:
        expected_tests = (
            "benchmark.tests.test_dtype_semantics.GcdLcmSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestGcdLcmFacadeSemantics",
        )
        expected_symbols = {"cnp_gcd", "cnp_lcm"}
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "dtype_gcd_lcm"
        }
        self.assertEqual(expected_symbols, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(expected_tests, item.tests)

    def test_eigh_and_eigvalsh_have_differential_semantic_owners(
        self,
    ) -> None:
        prefix = (
            "benchmark.tests.test_linalg_semantics."
            "LinalgSemanticsTests."
        )
        eigh_tests = (
            prefix + "test_eigh_v2_matches_numpy_dtypes_batches_and_uplo",
            prefix + "test_eigh_v2_empty_shapes_and_validation_are_atomic",
            prefix + "test_eigh_v2_nonfinite_triangle_selection_matches_numpy",
            prefix + "test_eigh_v2_tiny_scale_and_degenerate_spectra",
            prefix
            + "test_eigh_v2_finite_extreme_scale_avoids_intermediate_overflow",
            prefix + "test_eigh_legacy_abis_delegate_to_lower_triangle",
            "ahk.numpy.test.NumpyFoundationTest.TestSymmetricEigFacadeV2",
            "ahk.numpy.test.NumpyFoundationTest.TestLinalgSpectralDelegatesV2",
        )
        eigvalsh_tests = (
            prefix + "test_eigvalsh_v2_matches_numpy_views_and_result_lifetime",
            prefix + "test_eigh_legacy_abis_delegate_to_lower_triangle",
            "ahk.numpy.test.NumpyFoundationTest.TestSymmetricEigFacadeV2",
            "ahk.numpy.test.NumpyFoundationTest.TestLinalgSpectralDelegatesV2",
        )
        expected_symbols = {
            "cnp_linalg_eigh",
            "cnp_linalg_eigh_v2",
            "cnp_eigvalsh",
            "cnp_eigvalsh_v2",
        }
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "linalg_eigh"
        }
        self.assertEqual(expected_symbols, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertEqual("differential", item.status)
                self.assertIn(item.result, {"array", "status"})
                expected = (
                    eigvalsh_tests
                    if symbol.startswith("cnp_eigvalsh")
                    else eigh_tests
                )
                self.assertEqual(expected, item.tests)
                for owner in item.tests:
                    if not owner.startswith("benchmark."):
                        continue
                    module_name, class_name, method_name = owner.rsplit(".", 2)
                    module = importlib.import_module(module_name)
                    owner_class = getattr(module, class_name)
                    self.assertTrue(callable(getattr(owner_class, method_name)))

    def test_det_and_slogdet_have_explicit_semantic_owners(self) -> None:
        differential_tests = (
            "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestDetAndSlogdetFacadeV2",
        )
        differential = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "linalg_det_slogdet"
        }
        self.assertEqual(
            {"cnp_linalg_det", "cnp_linalg_slogdet_v2"},
            set(differential),
        )
        for symbol, item in differential.items():
            with self.subTest(symbol=symbol):
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertEqual("differential", item.status)
                self.assertIn(item.result, {"array", "status"})
                self.assertEqual(differential_tests, item.tests)

        legacy = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "linalg_slogdet_legacy"
        }
        self.assertEqual(
            {"cnp_linalg_slogdet", "cnp_slogdet"}, set(legacy)
        )
        for symbol, item in legacy.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("characterized", item.status)
                self.assertEqual(
                    (
                        "benchmark.tests.test_linalg_semantics."
                        "LinalgSemanticsTests",
                    ),
                    item.tests,
                )

    def test_character_surface_has_exact_semantic_and_lifecycle_owners(
        self,
    ) -> None:
        python_owner = (
            "benchmark.tests.test_character_surface_semantics."
            "CharacterSurfaceSemanticsTests"
        )
        ahk_owner = (
            "ahk.numpy.test.NumpyFoundationTest."
            "TestCharacterSurfaceFacadeSemantics"
        )
        expected_differential = {
            "cnp_char_add",
            "cnp_char_center",
            "cnp_char_count",
            "cnp_char_find",
            "cnp_char_join_v2",
            "cnp_char_ljust",
            "cnp_char_lower",
            "cnp_char_lstrip",
            "cnp_char_multiply",
            "cnp_char_replace",
            "cnp_char_rjust",
            "cnp_char_rstrip",
            "cnp_char_strip",
            "cnp_char_strlen",
            "cnp_char_upper",
            "cnp_char_zfill",
        }
        differential = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "character_surface"
        }
        self.assertEqual(expected_differential, set(differential))
        for symbol, item in differential.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual((python_owner, ahk_owner), item.tests)

        legacy = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "character_legacy"
        }
        self.assertEqual({"cnp_char_join", "cnp_char_split"}, set(legacy))
        for symbol, item in legacy.items():
            with self.subTest(legacy=symbol):
                self.assertEqual("characterized", item.status)
                self.assertEqual((python_owner,), item.tests)

        lifecycle = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "character_lifecycle"
        }
        self.assertEqual(
            {"cnp_char_free_result", "cnp_char_free_string"},
            set(lifecycle),
        )
        for symbol, item in lifecycle.items():
            with self.subTest(lifecycle=symbol):
                self.assertEqual("characterized", item.status)
                self.assertEqual("void", item.result)
                self.assertEqual((python_owner,), item.tests)

    def test_datetime_surface_has_exact_semantic_and_legacy_owners(self) -> None:
        python_owner = (
            "benchmark.tests.test_datetime_surface_semantics."
            "DatetimeSurfaceSemanticsTests"
        )
        ahk_owner = (
            "ahk.numpy.test.NumpyFoundationTest."
            "TestDatetimeSurfaceFacadeSemantics"
        )
        expected = {
            "cnp_arange_datetime",
            "cnp_busday_count",
            "cnp_busday_offset",
            "cnp_datetime64_add",
            "cnp_datetime64_array_create",
            "cnp_datetime64_compare",
            "cnp_datetime64_from_date",
            "cnp_datetime64_from_string",
            "cnp_datetime64_from_time",
            "cnp_datetime64_now",
            "cnp_datetime64_subtract",
            "cnp_datetime64_to_date",
            "cnp_datetime64_to_string",
            "cnp_datetime64_to_time",
            "cnp_datetime_as_string_v2",
            "cnp_datetime_unit_name",
            "cnp_is_busday",
            "cnp_timedelta64_create",
        }
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "datetime_surface"
        }
        self.assertEqual(expected, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual((python_owner, ahk_owner), item.tests)

        legacy = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "datetime_legacy"
        }
        self.assertEqual({"cnp_datetime_as_string"}, set(legacy))
        item = legacy["cnp_datetime_as_string"]
        self.assertEqual("characterized", item.status)
        self.assertEqual("status", item.result)
        self.assertEqual((python_owner,), item.tests)

    def test_iterator_surface_has_exact_semantic_and_facade_owners(self) -> None:
        python_owner = (
            "benchmark.tests.test_iterator_surface_semantics."
            "IteratorSurfaceSemanticsTests"
        )
        ahk_owner = (
            "ahk.numpy.test.NumpyFoundationTest."
            "TestIteratorSurfaceFacadeSemantics"
        )
        expected = {
            "cnp_iter_data",
            "cnp_iter_free",
            "cnp_iter_new",
            "cnp_iter_next",
            "cnp_iter_reset",
            "cnp_multi_iter_data",
            "cnp_multi_iter_free",
            "cnp_multi_iter_new",
            "cnp_multi_iter_next",
            "cnp_ndenumerate_next",
            "cnp_ndindex_next",
        }
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "iterator_surface"
        }
        self.assertEqual(expected, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual((python_owner, ahk_owner), item.tests)

    def test_buffer_representation_has_exact_surface_and_lifecycle_owners(
        self,
    ) -> None:
        python_owner = (
            "benchmark.tests.test_buffer_representation_semantics."
            "BufferRepresentationSemanticsTests"
        )
        ahk_owner = (
            "ahk.numpy.test.NumpyFoundationTest."
            "TestBufferRepresentationFacadeSemantics"
        )
        expected = {
            "cnp_base_repr",
            "cnp_binary_repr",
            "cnp_frombuffer",
            "cnp_tobytes",
            "cnp_tolist",
        }
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "buffer_representation_surface"
        }
        self.assertEqual(expected, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual((python_owner, ahk_owner), item.tests)

        lifecycle = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "buffer_representation_lifecycle"
        }
        self.assertEqual({"cnp_buffer_free"}, set(lifecycle))
        item = lifecycle["cnp_buffer_free"]
        self.assertEqual("characterized", item.status)
        self.assertEqual("void", item.result)
        self.assertEqual((python_owner, ahk_owner), item.tests)

    def test_grid_assembly_has_exact_surface_legacy_and_lifecycle_owners(
        self,
    ) -> None:
        python_owner = (
            "benchmark.tests.test_grid_assembly_semantics."
            "GridAssemblySemanticsTests"
        )
        ahk_owner = (
            "ahk.numpy.test.NumpyFoundationTest."
            "TestGridAssemblyFacadeSemantics"
        )
        expected_results = {
            "cnp_block": "array",
            "cnp_bmat": "array",
            "cnp_broadcast_shapes": "status",
            "cnp_can_broadcast": "scalar",
            "cnp_indices": "array",
            "cnp_mgrid": "status",
            "cnp_ogrid": "status",
        }
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "grid_assembly_surface"
        }
        self.assertEqual(set(expected_results), set(actual))
        for symbol, result in expected_results.items():
            with self.subTest(symbol=symbol):
                item = actual[symbol]
                self.assertEqual("differential", item.status)
                self.assertEqual(result, item.result)
                self.assertEqual((python_owner, ahk_owner), item.tests)

        legacy = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "grid_assembly_legacy"
        }
        self.assertEqual({"cnp_meshgrid"}, set(legacy))
        item = legacy["cnp_meshgrid"]
        self.assertEqual("characterized", item.status)
        self.assertEqual("array", item.result)
        self.assertEqual((python_owner, ahk_owner), item.tests)

        lifecycle = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "grid_assembly_lifecycle"
        }
        self.assertEqual({"cnp_broadcast_shape_free"}, set(lifecycle))
        item = lifecycle["cnp_broadcast_shape_free"]
        self.assertEqual("characterized", item.status)
        self.assertEqual("void", item.result)
        self.assertEqual((python_owner, ahk_owner), item.tests)

    def test_legacy_numeric_utilities_have_exact_characterized_owners(
        self,
    ) -> None:
        python_owner = (
            "benchmark.tests.test_legacy_numeric_utility_semantics."
            "LegacyNumericUtilitySemanticsTests"
        )
        facade_owner = (
            "ahk.numpy.test.NumpyFoundationTest."
            "TestLegacyNumericUtilityFacadeSemantics"
        )
        expected = {
            "cnp_bitwise_count": ("array", (python_owner, facade_owner)),
            "cnp_clip": (
                "array",
                (
                    python_owner,
                    "ahk.numpy.test.NumpyFoundationTest.TestClip",
                ),
            ),
            "cnp_item": ("scalar", (python_owner, facade_owner)),
            "cnp_sqrt_into": ("status", (python_owner, facade_owner)),
        }
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "legacy_numeric_utility"
        }
        self.assertEqual(set(expected), set(actual))
        for symbol, (result, owners) in expected.items():
            with self.subTest(symbol=symbol):
                item = actual[symbol]
                self.assertEqual("characterized", item.status)
                self.assertEqual(result, item.result)
                self.assertEqual(owners, item.tests)

    def test_calculus_surface_has_exact_differential_owners(self) -> None:
        symbols = {"cnp_diff", "cnp_gradient", "cnp_unwrap"}
        owners = (
            "benchmark.tests.test_calculus_surface_semantics."
            "CalculusSurfaceSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestCalculusSurfaceFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "calculus_surface"
        }
        self.assertEqual(symbols, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(owners, item.tests)

    def test_sampling_utility_surface_has_exact_differential_owners(
        self,
    ) -> None:
        symbols = {
            "cnp_histogram",
            "cnp_histogram2d",
            "cnp_interp",
            "cnp_interp_nd",
            "cnp_nan_to_num",
            "cnp_vander",
        }
        owners = (
            "benchmark.tests.test_sampling_utility_surface_semantics."
            "SamplingUtilitySurfaceSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest."
            "TestSamplingUtilitySurfaceFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "sampling_utility_surface"
        }
        self.assertEqual(symbols, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("explicit errors", item.reference)
                self.assertIn("lifetime", item.reference)

    def test_fft_surface_has_exact_differential_owners(self) -> None:
        symbols = {
            "cnp_fft",
            "cnp_fft2",
            "cnp_fftfreq",
            "cnp_fftn",
            "cnp_fftshift",
            "cnp_hfft",
            "cnp_ifft",
            "cnp_ifft2",
            "cnp_ifftn",
            "cnp_ifftshift",
            "cnp_ihfft",
            "cnp_irfft",
            "cnp_irfftn",
            "cnp_rfft",
            "cnp_rfftfreq",
            "cnp_rfftn",
        }
        owners = (
            "benchmark.tests.test_fft_surface_semantics."
            "FftSurfaceSemanticsTests",
            "ahk.numpy.test.NumpyFoundationTest.TestFftSurfaceFacadeSemantics",
        )
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "fft_surface"
        }
        self.assertEqual(symbols, set(actual))
        for symbol, item in actual.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("explicit errors", item.reference)
                self.assertIn("lifetime", item.reference)

    def test_structured_dtype_surface_has_exact_semantic_owners(self) -> None:
        python_owner = (
            "benchmark.tests.test_structured_dtype_surface_semantics."
            "StructuredDtypeSurfaceTests"
        )
        ahk_owner = (
            "ahk.numpy.test.NumpyFoundationTest."
            "TestStructuredDtypeSurfaceFacadeSemantics"
        )
        owners = (python_owner, ahk_owner)
        expected_results = {
            "cnp_getfield": "array",
            "cnp_recarray_get_field": "array",
            "cnp_recarray_get_record": "array",
            "cnp_recarray_names": "scalar",
            "cnp_recarray_new": "array",
            "cnp_recarray_set_field": "status",
            "cnp_recarray_set_record": "status",
            "cnp_setfield": "status",
            "cnp_struct_dtype_create": "scalar",
            "cnp_struct_dtype_field_name": "scalar",
            "cnp_struct_dtype_field_offset": "scalar",
            "cnp_struct_dtype_find_field": "scalar",
            "cnp_struct_dtype_itemsize": "scalar",
            "cnp_struct_dtype_nfields": "scalar",
            "cnp_view": "array",
        }
        actual = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "structured_dtype_surface"
        }
        self.assertEqual(set(expected_results), set(actual))
        for symbol, result in expected_results.items():
            with self.subTest(symbol=symbol):
                item = actual[symbol]
                self.assertEqual("differential", item.status)
                self.assertEqual(result, item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("explicit errors", item.reference)
                self.assertIn("lifetime", item.reference)

        projections = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "structured_dtype_projection"
        }
        self.assertEqual(
            {"cnp_newbyteorder", "cnp_recfromtxt"}, set(projections)
        )
        for symbol, item in projections.items():
            with self.subTest(projection=symbol):
                self.assertEqual("characterized", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("explicit projection", item.reference)
        self.assertIn(
            "native dtype",
            projections["cnp_newbyteorder"].reference,
        )
        self.assertIn(
            "homogeneous numeric",
            projections["cnp_recfromtxt"].reference,
        )

    def test_text_io_surface_has_exact_semantic_owners(self) -> None:
        python_owner = (
            "benchmark.tests.test_text_io_surface_semantics."
            "TextIoSurfaceSemanticsTests"
        )
        ahk_owner = (
            "ahk.numpy.test.NumpyFoundationTest."
            "TestTextIoSurfaceFacadeSemantics"
        )
        owners = (python_owner, ahk_owner)
        differential = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "text_io_surface"
        }
        self.assertEqual({"cnp_genfromtxt", "cnp_loadtxt"}, set(differential))
        for symbol, item in differential.items():
            with self.subTest(symbol=symbol):
                self.assertEqual("differential", item.status)
                self.assertEqual("array", item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("explicit errors", item.reference)
                self.assertIn("lifetime", item.reference)

        expected_projection_results = {
            "cnp_array2string": "scalar",
            "cnp_array_print": "void",
            "cnp_array_to_csv": "status",
            "cnp_array_to_string": "scalar",
            "cnp_disp": "status",
            "cnp_savetxt": "status",
        }
        projections = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "text_io_projection"
        }
        self.assertEqual(set(expected_projection_results), set(projections))
        for symbol, result in expected_projection_results.items():
            with self.subTest(projection=symbol):
                item = projections[symbol]
                self.assertEqual("characterized", item.status)
                self.assertEqual(result, item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("explicit projection", item.reference)
                self.assertIn("atomic", item.reference)

    def test_functional_callback_surface_has_exact_semantic_owners(self) -> None:
        python_owner = (
            "benchmark.tests.test_functional_callback_surface_semantics."
            "FunctionalCallbackSurfaceSemanticsTests"
        )
        ahk_owner = (
            "ahk.numpy.test.NumpyFoundationTest."
            "TestFunctionalCallbackSurfaceFacadeSemantics"
        )
        owners = (python_owner, ahk_owner)
        expected_projections = {
            "cnp_apply_along_axis": "array",
            "cnp_apply_over_axes": "array",
            "cnp_fromfunction": "array",
            "cnp_fromiter": "array",
            "cnp_frompyfunc": "array",
            "cnp_piecewise": "array",
            "cnp_vectorize": "array",
        }
        projections = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "functional_callback_projection"
        }
        self.assertEqual(set(expected_projections), set(projections))
        for symbol, result in expected_projections.items():
            with self.subTest(projection=symbol):
                item = projections[symbol]
                self.assertEqual("characterized", item.status)
                self.assertEqual(result, item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("explicit projection", item.reference)
                self.assertIn("callback order", item.reference)
                self.assertIn("explicit errors", item.reference)
                self.assertIn("lifetime", item.reference)

        expected_differential = {
            "cnp_put_along_axis": "status",
            "cnp_select": "array",
        }
        differential = {
            item.symbol: item
            for item in self.manifest.exports
            if item.family == "functional_surface"
        }
        self.assertEqual(set(expected_differential), set(differential))
        for symbol, result in expected_differential.items():
            with self.subTest(symbol=symbol):
                item = differential[symbol]
                self.assertEqual("differential", item.status)
                self.assertEqual(result, item.result)
                self.assertEqual(owners, item.tests)
                self.assertIn("NumPy 1.25.0", item.reference)
                self.assertIn("broadcast", item.reference)
                self.assertIn("atomic", item.reference)
                self.assertIn("explicit errors", item.reference)
                self.assertIn("lifetime", item.reference)


class PublicDeclarationParserTests(unittest.TestCase):
    def test_merges_identical_redeclarations(self) -> None:
        declaration = "CNP_API int CNP_CALL cnp_example(const int value);"
        parsed = parse_public_declarations(declaration + declaration)
        self.assertEqual(1, len(parsed))
        self.assertEqual("cnp_example", parsed[0].symbol)

    def test_rejects_conflicting_redeclarations(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicting public declarations"):
            parse_public_declarations(
                "CNP_API int CNP_CALL cnp_example(int value);"
                "CNP_API double CNP_CALL cnp_example(int value);"
            )


if __name__ == "__main__":
    unittest.main()
