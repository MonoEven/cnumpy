from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


def source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def function_body(text: str, function_name: str) -> str:
    signature = re.search(
        rf"\b{re.escape(function_name)}\s*\([^;]*?\)\s*\{{",
        text,
    )
    if signature is None:
        raise AssertionError(f"function not found: {function_name}")

    opening = text.index("{", signature.start())
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index]
    raise AssertionError(f"unbalanced function body: {function_name}")


class PublicInterfaceContracts(unittest.TestCase):
    def test_cross_has_one_public_signature_with_axis(self) -> None:
        header = source("include/cnumpy/cnumpy.h")
        declarations = re.findall(
            r"CNP_API\s+CnpArray\s*\*\s*CNP_CALL\s+cnp_cross\s*"
            r"\(([^;]+)\)\s*;",
            header,
        )
        self.assertEqual(1, len(declarations), declarations)
        self.assertRegex(declarations[0], r"\bint\s+axis\b")

    def test_behavior_v2_abis_expose_explicit_axis_lengths_and_none(self) -> None:
        header = source("include/cnumpy/cnumpy.h")
        self.assertRegex(
            header,
            r"cnp_linalg_tensorsolve_v2\s*\(\s*"
            r"const CnpArray \*a,\s*const CnpArray \*b,\s*"
            r"int naxes,\s*const int \*axes\s*\)",
        )
        self.assertRegex(
            header,
            r"cnp_count_nonzero_v2\s*\(\s*"
            r"const CnpArray \*arr,\s*int axis,\s*"
            r"bool axis_none,\s*bool keepdims\s*\)",
        )

    def test_bulk_ahk_callback_v2_abi_is_explicit_and_counted(self) -> None:
        header = source("include/cnumpy/cnumpy_ahk.h")
        self.assertIn("#define CNP_AHK_CALLBACK_BATCH_SIZE 256", header)

        callback_contracts = {
            "CnpAhkLineBatchCallback": (
                "const double *lines",
                "int64_t line_count",
                "int64_t line_length",
                "double *results",
                "int64_t result_capacity",
                "int64_t *produced_count",
                "void *userdata",
            ),
            "CnpAhkCoordinateBatchCallback": (
                "const int64_t *coordinates",
                "int64_t point_count",
                "int ndim",
                "double *results",
                "int64_t result_capacity",
                "int64_t *produced_count",
                "void *userdata",
            ),
            "CnpAhkIteratorBatchCallback": (
                "double *results",
                "int64_t result_capacity",
                "int64_t *produced_count",
                "void *userdata",
            ),
            "CnpAhkUnaryBatchCallback": (
                "const double *values",
                "int64_t value_count",
                "double *results",
                "int64_t result_capacity",
                "int64_t *produced_count",
                "void *userdata",
            ),
        }
        for callback, arguments in callback_contracts.items():
            with self.subTest(callback=callback):
                declaration = re.search(
                    rf"typedef\s+CNP_STATUS\s*\(CNP_CALL\s*\*{callback}\)"
                    rf"\s*\((?P<arguments>[^;]+)\)\s*;",
                    header,
                )
                self.assertIsNotNone(declaration)
                assert declaration is not None
                normalized = re.sub(r"\s+", " ", declaration.group("arguments"))
                for argument in arguments:
                    self.assertIn(argument, normalized)

        exports = {
            "cnp_ahk_apply_along_axis_v2": (
                "CnpAhkLineBatchCallback callback",
                "void *userdata",
                "int axis",
                "void *source",
                "int result_ndim",
                "const int64_t *result_shape",
            ),
            "cnp_ahk_apply_over_axes_v2": (
                "CnpAhkLineBatchCallback callback",
                "void *userdata",
                "int naxes",
                "const int *axes",
                "void *source",
            ),
            "cnp_ahk_fromfunction_v2": (
                "CnpAhkCoordinateBatchCallback callback",
                "void *userdata",
                "int ndim",
                "const int64_t *shape",
            ),
            "cnp_ahk_fromiter_v2": (
                "CnpAhkIteratorBatchCallback callback",
                "void *userdata",
                "int64_t count",
                "int dtype",
            ),
            "cnp_ahk_frompyfunc_v2": (
                "CnpAhkUnaryBatchCallback callback",
                "void *userdata",
                "void *source",
            ),
            "cnp_ahk_vectorize_v2": (
                "CnpAhkUnaryBatchCallback callback",
                "void *userdata",
                "void *source",
            ),
            "cnp_ahk_piecewise_v2": (
                "void *source",
                "int nconditions",
                "void *const *conditions",
                "CnpAhkUnaryBatchCallback callback",
                "void *userdata",
            ),
        }
        for symbol, arguments in exports.items():
            with self.subTest(export=symbol):
                declaration = re.search(
                    rf"CNP_API\s+void\s*\*\s*CNP_CALL\s+{symbol}\s*"
                    rf"\((?P<arguments>[^;]+)\)\s*;",
                    header,
                )
                self.assertIsNotNone(declaration)
                assert declaration is not None
                normalized = re.sub(r"\s+", " ", declaration.group("arguments"))
                for argument in arguments:
                    self.assertIn(argument, normalized)

    def test_bulk_callback_facade_routes_only_to_v2_exports(self) -> None:
        wrapper = source("ahk/numpy.ahk")
        expected = {
            "ApplyAlongAxis": "cnp_ahk_apply_along_axis_v2",
            "ApplyOverAxes": "cnp_ahk_apply_over_axes_v2",
            "FromFunction": "cnp_ahk_fromfunction_v2",
            "FromIter": "cnp_ahk_fromiter_v2",
            "FromPyFunc": "cnp_ahk_frompyfunc_v2",
            "Vectorize": "cnp_ahk_vectorize_v2",
            "Piecewise": "cnp_ahk_piecewise_v2",
        }
        for method, symbol in expected.items():
            with self.subTest(method=method):
                self.assertIn(symbol, function_body(wrapper, method))

        self.assertIn("_InvokeLineBatch", wrapper)
        self.assertIn("_InvokeCoordinateBatch", wrapper)
        self.assertIn("_InvokeIteratorBatch", wrapper)
        self.assertIn("_InvokeUnaryBatch", wrapper)
        self.assertNotIn("scalar retry", wrapper.lower())


class AhkFunctionPointerContracts(unittest.TestCase):
    def test_wrapper_caches_resolved_exports_and_uses_pointer_dllcalls(self) -> None:
        wrapper = source("ahk/numpy.ahk")
        self.assertIn("ProcCache", wrapper)
        self.assertIn("GetProcAddress", wrapper)
        self.assertRegex(wrapper, r"static\s+Proc\s*\(name\)")
        self.assertNotRegex(
            wrapper,
            r'DllCall\(Numpy\.RequireDllPath\(\)\s*"\\cnp_',
        )

    def test_static_add_uses_method_local_native_pointers(self) -> None:
        wrapper = source("ahk/numpy.ahk")
        body = function_body(wrapper, "Add")
        self.assertIn('Numpy.Proc("cnp_ahk_add")', body)
        self.assertIn('Numpy.Proc("cnp_ahk_add_into")', body)
        self.assertRegex(body, r"\bstatic\b")
        self.assertIn("cachedHandle", body)
        self.assertRegex(body, r"cachedHandle\s*!=\s*Numpy\.DllHandle")
        self.assertIn("DllCall", body)
        self.assertNotIn("left.Add", body)

    def test_pointer_cache_is_associated_with_the_loaded_dll_handle(self) -> None:
        wrapper = source("ahk/numpy.ahk")
        proc = function_body(wrapper, "Proc")
        self.assertIn("ProcCacheHandle", wrapper)
        self.assertRegex(
            proc,
            r"ProcCacheHandle\s*!=\s*Numpy\.DllHandle",
        )
        self.assertIn("Numpy.ProcCache := Map()", proc)

        worker = source("benchmark/bench_cnumpy.ahk")
        main = function_body(worker, "Main")
        cache_clear = main.index("Numpy.ProcCache := Map()")
        free_library = main.index('DllCall("kernel32\\FreeLibrary"')
        self.assertLess(cache_clear, free_library)

    def test_object_kind_facade_uses_one_error_transparent_dllcall(
        self,
    ) -> None:
        wrapper = source("ahk/numpy.ahk")
        declaration = wrapper.index("static _ObjectKindPredicate")
        wrapper_body = function_body(
            wrapper[declaration:], "_ObjectKindPredicate"
        )
        native_body = function_body(
            source("src/cnumpy_ahk.c"),
            "cnp_ahk_object_kind_predicate",
        )

        self.assertEqual(1, wrapper_body.count("DllCall"))
        self.assertIn("result < 0", wrapper_body)
        self.assertIn("Numpy.CheckStatus", wrapper_body)
        self.assertNotIn("cnp_clear_error", wrapper_body)
        self.assertNotIn("cnp_get_error", wrapper_body)
        self.assertIn("cnp_clear_error()", native_body)
        self.assertIn("cnp_get_error(NULL)", native_body)

    def test_isscalar_facade_has_explicit_ahk_and_native_domains(
        self,
    ) -> None:
        wrapper = source("ahk/numpy.ahk")
        declaration = wrapper.index("static IsScalar(source)")
        body = function_body(wrapper[declaration:], "IsScalar")
        predicate_body = function_body(
            wrapper[wrapper.index("static _ObjectKindPredicate"):],
            "_ObjectKindPredicate",
        )

        self.assertIn("source is Numpy.NdArray", body)
        self.assertIn("cnp_ahk_isscalar", body)
        for source_type in ("Integer", "Float", "String", "Buffer"):
            with self.subTest(source_type=source_type):
                self.assertIn(f'"{source_type}"', body)
        self.assertNotIn("DllCall", body)
        self.assertEqual(1, predicate_body.count("DllCall"))
        self.assertIn("result < 0", predicate_body)
        self.assertIn("Numpy.CheckStatus", predicate_body)

    def test_array_metadata_facade_uses_the_lazy_metadata_snapshot(
        self,
    ) -> None:
        wrapper = source("ahk/numpy.ahk")
        expected_bodies = {
            "Nbytes": "this._size * this._itemSize",
            "CContiguous": (
                "(this._flags & Numpy.ARRAY_C_CONTIGUOUS) != 0"
            ),
            "FContiguous": (
                "(this._flags & Numpy.ARRAY_F_CONTIGUOUS) != 0"
            ),
        }
        for property_name, expected in expected_bodies.items():
            with self.subTest(property=property_name):
                match = re.search(
                    rf"\n        {property_name} \{{\s+get \{{"
                    rf"(?P<body>.*?)\n            \}}\s+\}}",
                    wrapper,
                    re.DOTALL,
                )
                self.assertIsNotNone(match)
                body = match.group("body")
                self.assertIn("this._LoadMetadata()", body)
                self.assertIn(expected, body)
                self.assertNotIn("DllCall", body)

        self.assertIn("static ARRAY_C_CONTIGUOUS := 0x0001", wrapper)
        self.assertIn("static ARRAY_F_CONTIGUOUS := 0x0002", wrapper)

    def test_timed_worker_uses_cached_pointers_except_raw_bridge_case(self) -> None:
        worker = source("benchmark/bench_cnumpy.ahk")
        invoke = re.search(
            r"\n\s*InvokeOperation\(\)\s*\{([\s\S]*?)"
            r"\n\s*InvokeTransposeCopy\(\)",
            worker,
        )
        self.assertIsNotNone(invoke)
        assert invoke is not None
        body = invoke.group(1)
        raw_bridge = re.search(
            r'case\s+"property_call":\s*\n\s*return\s+'
            r'DllCall\(dllPath\s+"\\cnp_ahk_size"[^\n]+',
            body,
        )
        self.assertIsNotNone(raw_bridge)
        assert raw_bridge is not None
        production_paths = body.replace(raw_bridge.group(0), "")
        self.assertIn('Numpy.Proc("cnp_ahk_" operation)', production_paths)
        self.assertNotRegex(production_paths, r"DllCall\(dllPath\s+")
        transpose = re.search(
            r"\n\s*InvokeTransposeCopy\(\)\s*\{([\s\S]*?)\n\s*\}",
            worker,
        )
        self.assertIsNotNone(transpose)
        assert transpose is not None
        self.assertIn('Numpy.Proc("cnp_ahk_transpose_copy")', transpose.group(1))


class PhaseAContracts(unittest.TestCase):
    def test_dead_reduction_simd_symbols_are_absent(self) -> None:
        production_sources = {
            path: source(path)
            for path in (
                "include/cnumpy/cnumpy_internal.h",
                "src/simd_ops.c",
                "src/simd_avx2.c",
                "src/simd_dispatch.c",
            )
        }
        dead_symbols = tuple(
            f"cnp_{level}_{operation}"
            for level in ("sse2", "avx2", "simd")
            for operation in (
                "argmax",
                "prod",
                "max",
                "min",
                "sum_squared_deviation",
            )
        )
        for path, implementation in production_sources.items():
            for symbol in dead_symbols:
                with self.subTest(path=path, symbol=symbol):
                    self.assertNotRegex(
                        implementation, rf"\b{symbol}\b"
                    )

        reduction = source("src/reduce.c")
        self.assertIn(
            "reduction_arg_extrema",
            function_body(reduction, "cnp_argmax_v2"),
        )
        self.assertIn(
            "reduction_sumprod_real",
            function_body(reduction, "cnp_prod_v2"),
        )

    def test_argmax_routes_legacy_calls_through_shared_semantics(self) -> None:
        implementation = source("src/reduce.c")
        legacy = function_body(implementation, "cnp_argmax")
        modern = function_body(implementation, "cnp_argmax_v2")
        shared = function_body(implementation, "reduction_arg_extrema")
        self.assertIn("cnp_argmax_v2", legacy)
        self.assertNotIn("cnp_simd_argmax", legacy)
        self.assertIn("reduction_arg_extrema", modern)
        self.assertIn("isnan", shared)
        self.assertIn("reduction_compare_offsets", shared)
        self.assertNotIn("cnp_simd_argmax", shared)

    def test_cumsum_routes_legacy_calls_to_ordered_shared_engine(self) -> None:
        implementation = source("src/reduce.c")
        legacy = function_body(implementation, "cnp_cumsum")
        modern = function_body(implementation, "cnp_cumsum_v2")
        shared = function_body(implementation, "reduction_cumulative_real")
        self.assertIn("cnp_cumsum_v2", legacy)
        self.assertNotIn("cnp_flatten", legacy)
        self.assertIn("reduction_cumulative_integer", modern)
        self.assertIn("reduction_cumulative_real", modern)
        for token in ("CNP_DOUBLE", "arr->offset", "accumulator + value"):
            with self.subTest(token=token):
                self.assertIn(token, shared)
        self.assertNotIn("cnp_flatten", shared)
        for path in (
            "include/cnumpy/cnumpy_internal.h",
            "src/reduce.c",
            "src/into_ops.c",
            "src/simd_ops.c",
        ):
            with self.subTest(path=path):
                self.assertNotRegex(
                    source(path), r"\bcnp_simd_cumsum\b"
                )

    def test_contiguous_reduce_all_uses_linear_source_offsets(self) -> None:
        body = function_body(
            source("src/reduce.c"), "reduction_source_offset"
        )

        self.assertIn("traversal->axis_none", body)
        self.assertIn("CNP_ARRAY_C_CONTIGUOUS", body)
        self.assertIn("arr->offset", body)
        self.assertIn("item * arr->dtype->elsize", body)
        self.assertIn("reduction_flat_offset", body)
        self.assertLess(
            body.index("CNP_ARRAY_C_CONTIGUOUS"),
            body.index("reduction_flat_offset"),
        )

    def test_float64_extrema_fast_path_is_strictly_contiguous_slices(
        self,
    ) -> None:
        implementation = source("src/reduce.c")
        public_path = function_body(implementation, "reduction_extrema")

        for token in (
            "traversal.axis_none",
            "traversal.axis == arr->ndim - 1",
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_DOUBLE",
            "reduction_extrema_contiguous_double",
        ):
            with self.subTest(token=token):
                self.assertIn(token, public_path)

        fast_path = function_body(
            implementation, "reduction_extrema_contiguous_double"
        )
        for token in (
            "traversal->axis_length",
            "result->size",
            "isnan(selected)",
            "isnan(source)",
            "source >= selected",
            "source <= selected",
            "selected_index",
            "memcpy",
        ):
            with self.subTest(token=token):
                self.assertIn(token, fast_path)
        self.assertNotRegex(fast_path, r"\bcnp_(?:simd|sse2|avx2)_")

    def test_float64_extrema_and_arg_fast_paths_cover_contiguous_last_axis(
        self,
    ) -> None:
        implementation = source("src/reduce.c")
        extrema_path = function_body(implementation, "reduction_extrema")
        arg_path = function_body(implementation, "reduction_arg_extrema")
        value_fast_path = function_body(
            implementation, "reduction_extrema_contiguous_double"
        )
        arg_fast_path = function_body(
            implementation, "reduction_arg_extrema_contiguous_double"
        )

        for public_path, helper in (
            (extrema_path, "reduction_extrema_contiguous_double"),
            (arg_path, "reduction_arg_extrema_contiguous_double"),
        ):
            self.assertIn("traversal.axis_none", public_path)
            self.assertIn("traversal.axis == arr->ndim - 1", public_path)
            self.assertIn("CNP_ARRAY_C_CONTIGUOUS", public_path)
            self.assertIn("CNP_DOUBLE", public_path)
            self.assertIn(helper, public_path)

        for fast_path in (value_fast_path, arg_fast_path):
            self.assertIn("traversal->axis_length", fast_path)
            self.assertIn("result->size", fast_path)
            self.assertNotIn("reduction_source_offset", fast_path)
            self.assertNotIn("cnp_get_element_double", fast_path)
        self.assertIn("source >= selected", value_fast_path)
        self.assertIn("source <= selected", value_fast_path)
        self.assertIn("source > selected", arg_fast_path)
        self.assertIn("source < selected", arg_fast_path)

    def test_float64_sum_and_product_fast_paths_cover_contiguous_slices(
        self,
    ) -> None:
        implementation = source("src/reduce.c")
        public_path = function_body(implementation, "reduction_sumprod_real")
        product_path = function_body(
            implementation, "reduction_product_contiguous_double"
        )

        self.assertIn("contiguous_double_slices", public_path)
        self.assertIn("traversal.axis == arr->ndim - 1", public_path)
        self.assertIn("output_index * traversal.axis_length", public_path)
        self.assertIn(
            "reduction_pairwise_sum_contiguous_double", public_path
        )
        self.assertIn("reduction_product_contiguous_double", public_path)
        self.assertIn("accumulator *= values[item]", product_path)
        self.assertNotIn("reduction_source_double", product_path)

    def test_float64_pairwise_fast_path_copies_the_existing_tree_strictly(
        self,
    ) -> None:
        implementation = source("src/reduce.c")
        fast_path = function_body(
            implementation, "reduction_pairwise_sum_contiguous_double"
        )
        self.assertRegex(
            implementation,
            r"reduction_pairwise_sum_contiguous_double\s*\(\s*"
            r"const double \*values",
        )
        for token in (
            "length < 8",
            "length <= CNP_REDUCTION_PAIRWISE_BLOCK",
            "double partial[8]",
            "length - (length % 8)",
            "((partial[0] + partial[1]) +",
            "left_length = length / 2",
            "left_length -= left_length % 8",
        ):
            with self.subTest(token=token):
                self.assertIn(token, fast_path)
        self.assertGreaterEqual(
            fast_path.count("reduction_pairwise_sum_contiguous_double("),
            2,
        )
        self.assertNotIn("reduction_source_double", fast_path)
        self.assertNotRegex(fast_path, r"\b(?:fmax|cnp_(?:simd|sse2|avx2)_)")

        leaf = function_body(
            implementation, "reduction_contiguous_double_value"
        )
        self.assertRegex(
            implementation,
            r"reduction_contiguous_double_value\s*\(\s*"
            r"const double \*values",
        )
        self.assertIn("CNP_REDUCTION_SQUARED_DEVIATION", leaf)
        self.assertIn("reduction_squared_deviation_double", leaf)
        self.assertNotIn("CNP_REDUCTION_NAN", leaf)

        sum_path = function_body(implementation, "reduction_sumprod_real")
        for token in (
            "contiguous_double_slices",
            "traversal.axis_none",
            "traversal.axis == arr->ndim - 1",
            "CNP_ARRAY_C_CONTIGUOUS",
            "arr->dtype->type_num == CNP_DOUBLE",
            "out_dtype == CNP_DOUBLE",
            "reduction_pairwise_sum_contiguous_double",
        ):
            with self.subTest(path="sum", token=token):
                self.assertIn(token, sum_path)

        variance_path = function_body(
            implementation, "reduction_variance"
        )
        for token in (
            "contiguous_double_variance",
            "traversal.axis_none",
            "CNP_ARRAY_C_CONTIGUOUS",
            "arr->dtype->type_num == CNP_DOUBLE",
            "out_dtype == CNP_DOUBLE",
        ):
            with self.subTest(path="variance", token=token):
                self.assertIn(token, variance_path)
        self.assertGreaterEqual(
            variance_path.count(
                "reduction_pairwise_sum_contiguous_double("
            ),
            2,
        )

    def test_typed_bitpacking_uses_direct_unit_stride_slice_helpers(
        self,
    ) -> None:
        implementation = source("src/io_extra.c")
        pack_public = function_body(implementation, "cnp_packbits_v2")
        unpack_public = function_body(implementation, "cnp_unpackbits_v2")
        pack_fast = function_body(
            implementation, "packbits_contiguous_byte_slice"
        )
        unpack_fast = function_body(
            implementation, "unpackbits_contiguous_byte_slice"
        )
        unpack_interleaved = function_body(
            implementation,
            "unpackbits_contiguous_interleaved_slices",
        )

        for token in (
            "CNP_BOOL",
            "CNP_UBYTE",
            "traversal.axis_stride == 1",
            "packbits_contiguous_byte_slice",
        ):
            with self.subTest(path="pack", token=token):
                self.assertIn(token, pack_public)
        for token in (
            "source[bit] != 0",
            "full_bytes",
            "remaining_bits",
            "CNP_BITORDER_BIG",
            "_mm_movemask_epi8",
            "packbits_reverse_byte",
        ):
            with self.subTest(helper="pack", token=token):
                self.assertIn(token, pack_fast)
        self.assertNotIn("cnp_get_element_double", pack_fast)
        for token in (
            "traversal.axis_stride == 1",
            "unpackbits_contiguous_byte_slice",
        ):
            with self.subTest(path="unpack", token=token):
                self.assertIn(token, unpack_public)
        for token in (
            "copied_bits",
            "output_stride",
            "CNP_BITORDER_BIG",
            "unpackbits_nibble_big",
            "unpackbits_nibble_little",
            "memcpy",
        ):
            with self.subTest(helper="unpack", token=token):
                self.assertIn(token, unpack_fast)
        self.assertNotIn("cnp_reduction_source_offset", unpack_fast)
        for token in (
            "slice_offsets",
            "gathered_bytes",
            "destination[inner]",
            "cnp_malloc",
            "cnp_free",
        ):
            with self.subTest(helper="interleaved", token=token):
                self.assertIn(token, unpack_interleaved)
        self.assertIn(
            "unpackbits_contiguous_interleaved_slices",
            unpack_public,
        )

    def test_float64_trapz_uses_contiguous_panel_and_pairwise_sum(
        self,
    ) -> None:
        implementation = source("src/extra.c")
        public = function_body(implementation, "cnp_trapz")
        fast = function_body(
            implementation, "trapz_contiguous_double_slices"
        )
        for token in (
            "plan.use_slice_path",
            "x == NULL",
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_DOUBLE",
            "trapz_contiguous_double_slices",
        ):
            with self.subTest(path="trapz", token=token):
                self.assertIn(token, public)
        for token in (
            "panel_values[item]",
            "cnp_reduction_sum_contiguous_double",
            "cnp_malloc",
            "cnp_free",
        ):
            with self.subTest(helper="trapz", token=token):
                self.assertIn(token, fast)
        self.assertNotIn("cnp_reduction_source_offset", fast)
        self.assertNotIn("trapz_pair_value", fast)

    def test_prod_routes_to_order_preserving_v2_reduction(self) -> None:
        body = function_body(source("src/reduce.c"), "cnp_prod")
        self.assertIn("cnp_prod_v2", body)
        self.assertNotIn("cnp_simd_prod", body)
        self.assertNotIn("cnp_reduce_op", body)

    def test_percentile_family_rejects_nan_before_range_comparisons(self) -> None:
        implementation = source("src/typeconv.c")
        functions = (
            "cnp_percentile_family_v2",
            "cnp_quantile_v2",
            "cnp_nanquantile_v2",
        )
        for function_name in functions:
            with self.subTest(function=function_name):
                body = function_body(implementation, function_name)
                self.assertRegex(
                    body,
                    r"if\s*\(\s*isnan\(q\)\s*\|\|\s*q\s*<",
                )


class RangeGenerationContracts(unittest.TestCase):
    def test_contiguous_float64_ranges_use_runtime_dispatched_simd(self) -> None:
        internal = source("include/cnumpy/cnumpy_internal.h")
        baseline = source("src/simd_ops.c")
        avx2 = source("src/simd_avx2.c")
        dispatch = source("src/simd_dispatch.c")
        array_source = source("src/array.c")

        for symbol, text in (
            ("cnp_sse2_arange", internal),
            ("cnp_avx2_arange", internal),
            ("cnp_simd_arange", internal),
            ("cnp_sse2_arange", baseline),
            ("cnp_avx2_arange", avx2),
            ("cnp_simd_arange", dispatch),
        ):
            with self.subTest(symbol=symbol):
                self.assertRegex(text, rf"\b{symbol}\s*\(")
        self.assertIn("g_arange_kernel", dispatch)

        for function_name in ("cnp_arange", "cnp_linspace"):
            with self.subTest(function=function_name):
                body = function_body(array_source, function_name)
                self.assertIn("CNP_DOUBLE", body)
                self.assertIn("cnp_simd_arange", body)
                self.assertLess(body.index("cnp_simd_arange"), body.index("for ("))


class PhaseBContracts(unittest.TestCase):
    def test_sorting_benchmark_exercises_flattened_radix_eligible_path(self) -> None:
        body = function_body(source("benchmark/bench_cnumpy.ahk"), "InvokeOperation")
        sorting_start = body.index('case "sort", "argsort"')
        sorting_end = body.index('case "copy"', sorting_start)
        sorting_case = body[sorting_start:sorting_end]
        self.assertRegex(
            sorting_case,
            r'"Int",\s*0,\s*"Int",\s*1,\s*"Int",\s*this\.SortingContract\[2\]',
        )

    def test_radix_argsort_scatter_carries_keys_and_indices(self) -> None:
        body = function_body(source("src/sort.c"), "radix_argsort_doubles")
        self.assertIn("double_to_sortable", body)
        self.assertIn("result_indices", body)
        self.assertIn("indices_tmp", body)
        self.assertRegex(body, r"key_out\s*\[\s*destination\s*\]")
        self.assertRegex(body, r"index_out\s*\[\s*destination\s*\]")
        self.assertIn("index_in != result_indices", body)

    def test_radix_argsort_allocation_failure_is_explicit(self) -> None:
        body = function_body(source("src/sort.c"), "radix_argsort_doubles")
        self.assertIn("CNP_ERR_MEMORY", body)
        self.assertIn("cnp_set_error", body)
        self.assertNotIn("qsort", body)

    def test_large_contiguous_float64_argsort_avoids_flatten(self) -> None:
        body = function_body(source("src/sort.c"), "cnp_argsort_v2")
        self.assertIn("CNP_ARRAY_C_CONTIGUOUS", body)
        self.assertIn("CNP_DOUBLE", body)
        self.assertRegex(body, r"n\s*>=\s*CNP_RADIX_SORT_THRESHOLD")
        self.assertIn("radix_argsort_doubles", body)
        self.assertLess(body.index("radix_argsort_doubles"), body.index("cnp_flatten"))

    def test_radix_threshold_covers_medium_contiguous_arrays(self) -> None:
        sort_source = source("src/sort.c")
        self.assertRegex(
            sort_source,
            r"#define\s+CNP_RADIX_SORT_THRESHOLD\s+512\b",
        )
        for function_name in ("cnp_sort_v2", "cnp_argsort_v2"):
            with self.subTest(function=function_name):
                body = function_body(sort_source, function_name)
                self.assertRegex(
                    body,
                    r"n\s*>=\s*CNP_RADIX_SORT_THRESHOLD",
                )

    def test_searchsorted_has_an_offset_correct_contiguous_float64_path(
        self,
    ) -> None:
        implementation = source("src/sort.c")
        public_body = function_body(implementation, "cnp_searchsorted_v2")
        fast_body = function_body(
            implementation, "searchsorted_contiguous_float64"
        )

        self.assertIn("searchsorted_contiguous_float64", public_body)
        self.assertLess(
            public_body.index("searchsorted_contiguous_float64"),
            public_body.index("sort_flat_offset(values, i)"),
        )
        for token in (
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_DOUBLE",
            "arr->offset",
            "values->offset",
            "compare_numpy_doubles",
        ):
            with self.subTest(token=token):
                self.assertIn(token, fast_body)
        self.assertNotIn("sort_flat_offset", fast_body)
        self.assertNotIn("cnp_cast_scalar_value", fast_body)

    def test_digitize_has_a_direct_decreasing_contiguous_float64_path(
        self,
    ) -> None:
        implementation = source("src/extra.c")
        public_body = function_body(implementation, "cnp_digitize")
        fast_body = function_body(
            implementation, "digitize_decreasing_contiguous_float64"
        )

        self.assertIn("digitize_decreasing_contiguous_float64", public_body)
        self.assertLess(
            public_body.index("digitize_decreasing_contiguous_float64"),
            public_body.index("cnp_array_slice"),
        )
        for token in (
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_DOUBLE",
            "x->offset",
            "bins->offset",
            "cnp_compare_numpy_doubles",
            "bin_count - 1 - middle",
            "bin_count - low",
        ):
            with self.subTest(token=token):
                self.assertIn(token, fast_body)
        self.assertNotIn("digitize_flat_offset", fast_body)
        self.assertNotIn("cnp_cast_scalar_value", fast_body)

    def test_lexsort_has_a_stable_contiguous_float64_radix_path(
        self,
    ) -> None:
        implementation = source("src/array_ops.c")
        public_body = function_body(implementation, "lexsort_impl")
        fast_body = function_body(
            implementation, "lexsort_contiguous_float64"
        )

        self.assertIn("lexsort_contiguous_float64", public_body)
        self.assertLess(
            public_body.index("lexsort_contiguous_float64"),
            public_body.index("lexsort_merge_indices"),
        )
        for token in (
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_DOUBLE",
            "cnp_double_to_sortable",
            "key_index = 0",
            "CNP_LEXSORT_RADIX_SIZE",
        ):
            with self.subTest(token=token):
                self.assertIn(token, fast_body)
        self.assertNotIn("cnp_compare_numeric_elements", fast_body)

    def test_msort_and_sort_complex_use_the_shared_typed_axis_sort(
        self,
    ) -> None:
        array_ops = source("src/array_ops.c")
        sorting = source("src/sort.c") + source("src/numpy_ext.c")
        msort_body = function_body(array_ops, "cnp_msort")
        sort_complex_body = function_body(sorting, "cnp_sort_complex")
        dtype_body = function_body(source("src/sort.c"), "sort_dtype_is_supported")

        self.assertIn("cnp_sort_v2", msort_body)
        self.assertIn("cnp_sort_v2", sort_complex_body)
        self.assertIn("cnp_cast_scalar_value", sort_complex_body)
        self.assertNotIn("cnp_array_flat_get", sort_complex_body)
        self.assertNotIn("cnp_array_flat_set", sort_complex_body)
        self.assertIn("cnp_type_is_complex", dtype_body)
        self.assertIn("CNP_DATETIME", dtype_body)
        self.assertIn("CNP_TIMEDELTA", dtype_body)

    def test_float64_quicksort_routes_to_direct_values_before_indices(
        self,
    ) -> None:
        sorting = source("src/sort.c")
        axis_body = function_body(sorting, "sort_axis_result")
        dispatch_body = function_body(sorting, "sort_double_values")
        quicksort_body = function_body(sorting, "quicksort_double_values")
        less_body = function_body(sorting, "double_value_less")

        direct_route = axis_body.index("sort_double_values")
        index_allocation = axis_body.index("cnp_malloc(index_bytes)")
        self.assertLess(direct_route, index_allocation)
        self.assertIn("quicksort_double_values", dispatch_body)
        self.assertIn("double_value_less", quicksort_body)
        self.assertIn("cnp_compare_numpy_doubles", less_body)
        self.assertNotIn("cnp_malloc", quicksort_body)

    def test_axis_none_float64_quicksort_and_heapsort_sort_values_directly(
        self,
    ) -> None:
        sorting = source("src/sort.c")
        public_body = function_body(sorting, "cnp_sort_v2")
        dispatch_body = function_body(sorting, "sort_double_values")

        self.assertIn("sort_double_values", public_body)
        self.assertLess(
            public_body.index("sort_double_values"),
            public_body.index("cnp_malloc(index_bytes)"),
        )
        self.assertIn("CNP_SORT_QUICKSORT", dispatch_body)
        self.assertIn("CNP_SORT_HEAPSORT", dispatch_body)
        self.assertIn("quicksort_double_values", dispatch_body)
        self.assertIn("heapsort_double_values", dispatch_body)
        self.assertNotIn("indices", dispatch_body)

    def test_float64_argsort_specializes_index_comparisons_once_per_call(
        self,
    ) -> None:
        sorting = source("src/sort.c")
        public_body = function_body(sorting, "argsort_raw")
        quicksort_body = function_body(
            sorting, "quicksort_double_indices"
        )
        heapsort_body = function_body(
            sorting, "heapsort_double_indices"
        )

        self.assertIn("CNP_DOUBLE", public_body)
        self.assertIn("quicksort_double_indices", public_body)
        self.assertIn("heapsort_double_indices", public_body)
        for body in (quicksort_body, heapsort_body):
            self.assertIn("double_value_less", body)
            self.assertNotIn("compare_raw_values", body)

    def test_unique_values_only_route_precedes_optional_index_argsort(
        self,
    ) -> None:
        setops = source("src/setops.c")
        public_body = function_body(setops, "cnp_unique_v2")
        values_body = function_body(setops, "unique_values_only")

        self.assertIn("unique_values_only", public_body)
        self.assertLess(
            public_body.index("unique_values_only"),
            public_body.index("cnp_argsort_v2"),
        )
        self.assertIn("cnp_sort_v2", values_body)
        self.assertNotIn("cnp_argsort_v2", values_body)
        self.assertNotIn("sorted_indices", values_body)

    def test_membership_uses_sorted_search_instead_of_nested_scan(self) -> None:
        setops = source("src/setops.c")
        public_body = function_body(setops, "cnp_in1d")
        search_body = function_body(setops, "membership_sorted_contains")

        self.assertIn("cnp_sort_v2", public_body)
        self.assertIn("membership_sorted_contains", public_body)
        self.assertEqual(1, search_body.count("while ("))
        self.assertIn("set_scalar_equal", search_body)
        self.assertNotIn("right_index", public_body)


class PhaseCContracts(unittest.TestCase):
    def test_contiguous_flatten_uses_one_offset_adjusted_memcpy(self) -> None:
        body = function_body(source("src/shape.c"), "cnp_flatten")
        self.assertIn("CNP_ARRAY_C_CONTIGUOUS", body)
        self.assertIn("arr->offset", body)
        self.assertRegex(body, r"memcpy\s*\(\s*result->data")
        self.assertLess(body.index("CNP_ARRAY_C_CONTIGUOUS"), body.index("coords"))

    def test_contiguous_concatenate_uses_axis_blocks(self) -> None:
        body = function_body(source("src/shape.c"), "cnp_concatenate")
        for name in ("outer", "inner_bytes", "src_block_bytes"):
            with self.subTest(name=name):
                self.assertIn(name, body)
        self.assertIn("CNP_ARRAY_C_CONTIGUOUS", body)
        self.assertIn("dtype->type_num", body)
        self.assertIn("src->offset", body)
        self.assertRegex(body, r"memcpy\s*\(\s*dst")
        self.assertLess(body.index("src_block_bytes"), body.index("dst_coords"))


class PhaseDContracts(unittest.TestCase):
    def test_virtual_owner_flag_and_exact_threshold_are_declared(self) -> None:
        public_header = source("include/cnumpy/cnumpy.h")
        array_source = source("src/array.c")
        self.assertRegex(
            public_header,
            r"#define\s+CNP_ARRAY_VIRTUAL_ALLOC\s+0x0800",
        )
        self.assertRegex(
            array_source,
            r"#define\s+CNP_VIRTUAL_ZERO_THRESHOLD\s+\(64u\s*\*\s*1024u\)",
        )

    def test_virtual_alloc_and_free_use_required_win32_contract(self) -> None:
        core_source = source("src/core.c")
        allocate = function_body(core_source, "cnp_virtual_alloc")
        release = function_body(core_source, "cnp_virtual_free")
        self.assertIn("MEM_RESERVE | MEM_COMMIT", allocate)
        self.assertIn("PAGE_READWRITE", allocate)
        self.assertIn("g_cnp_allocated_memory += size", allocate)
        self.assertIn("VirtualFree(ptr, 0, MEM_RELEASE)", release)
        self.assertIn("cnp_set_error", release)
        self.assertIn("g_cnp_allocated_memory -= size", release)

    def test_zeros_and_free_dispatch_by_allocator_ownership(self) -> None:
        array_source = source("src/array.c")
        zeros = function_body(array_source, "cnp_array_zeros")
        release = function_body(array_source, "cnp_array_free")
        self.assertIn("cnp_virtual_alloc", zeros)
        self.assertIn("CNP_ARRAY_VIRTUAL_ALLOC", zeros)
        virtual_success = zeros.index(
            "return arr;", zeros.index("CNP_ARRAY_VIRTUAL_ALLOC")
        )
        self.assertLess(virtual_success, zeros.index("cnp_simd_zeros"))
        self.assertIn("cnp_virtual_free", release)
        self.assertIn("CNP_ARRAY_VIRTUAL_ALLOC", release)

    def test_ahk_release_is_refcounted_and_reports_errors(self) -> None:
        bridge = source("src/cnumpy_ahk.c")
        release = function_body(bridge, "cnp_ahk_free")
        self.assertIn("cnp_clear_error", release)
        self.assertIn("cnp_array_decref", release)
        self.assertIn("cnp_get_error", release)
        self.assertRegex(bridge, r"\bcnp_ahk_flags\s*\(")


class PhaseEContracts(unittest.TestCase):
    def test_atleast_nd_uses_reference_or_single_metadata_view_only(self) -> None:
        implementation = source("src/array_ops.c")
        same_reference = function_body(
            implementation, "atleast_same_array_reference"
        )
        self.assertIn("cnp_array_incref", same_reference)
        self.assertRegex(same_reference, r"return\s+array\s*;")

        forbidden = (
            "cnp_array_copy",
            "cnp_array_new",
            "cnp_reshape",
            "cnp_array_flat_get",
            "cnp_set_element_double",
        )
        for minimum_ndim in (1, 2, 3):
            function_name = f"cnp_atleast_{minimum_ndim}d"
            with self.subTest(function=function_name):
                body = function_body(implementation, function_name)
                self.assertIn("atleast_validate_array", body)
                self.assertIn("atleast_same_array_reference", body)
                self.assertIn("cnp_array_view_from_metadata", body)
                for token in forbidden:
                    self.assertNotIn(token, body)

    def test_atleast_nd_ahk_bridge_and_facade_are_error_transparent(self) -> None:
        header = source("include/cnumpy/cnumpy_ahk.h")
        bridge = source("src/cnumpy_ahk.c")
        wrapper = source("ahk/numpy.ahk")
        for minimum_ndim in (1, 2, 3):
            native_name = f"cnp_atleast_{minimum_ndim}d"
            bridge_name = f"cnp_ahk_atleast_{minimum_ndim}d"
            with self.subTest(function=bridge_name):
                self.assertRegex(header, rf"\b{bridge_name}\s*\(")
                body = function_body(bridge, bridge_name)
                self.assertIn(native_name, body)
                self.assertIn("cnp_relabel_error", body)
                self.assertIn(bridge_name, wrapper)
                self.assertIn(f"Atleast{minimum_ndim}d", wrapper)
                self.assertIn(f"Atleast_{minimum_ndim}d", wrapper)

    def test_direct_reshape_view_delegates_to_merged_metadata_constructor(self) -> None:
        internal = source("include/cnumpy/cnumpy_internal.h")
        array_source = source("src/array.c")
        self.assertRegex(internal, r"\bcnp_array_reshape_view\s*\(")
        body = function_body(array_source, "cnp_array_reshape_view")
        self.assertIn("cnp_compute_strides", body)
        self.assertIn("cnp_array_view_from_metadata", body)
        self.assertNotIn("cnp_malloc", body)
        self.assertNotIn("cnp_array_incref", body)

    def test_contiguous_reshape_calls_direct_constructor_only(self) -> None:
        body = function_body(source("src/shape.c"), "cnp_reshape")
        self.assertIn("cnp_array_reshape_view", body)
        self.assertNotIn("cnp_array_view", body)
        self.assertNotIn("cnp_free(result->shape", body)
        self.assertNotIn("cnp_free(result->strides", body)

    def test_ahk_exposes_read_only_shared_storage_diagnostic(self) -> None:
        body = function_body(source("src/cnumpy_ahk.c"), "cnp_ahk_shares_data")
        self.assertIn("left->data == right->data", body)
        self.assertIn("left->offset == right->offset", body)
        self.assertNotIn("return left->data;", body)


class ViewLifetimeContracts(unittest.TestCase):
    def test_ahk_suite_executes_and_propagates_exit_code(self) -> None:
        suite = source("ahk/numpy.test.ahk")
        self.assertRegex(suite, r"AhkTest\.SetOutputFile\s*\(")
        self.assertRegex(suite, r"suiteResult\s*:=\s*AhkTest\.Run\s*\(")
        self.assertLess(suite.index("AhkTest.SetOutputFile"), suite.index("AhkTest.Run"))
        self.assertRegex(suite, r"ExitApp\s+suiteResult\.ExitCode")
        self.assertNotIn("ExitApp(", suite)

        smoke = source("benchmark/benchmark_smoke.test.ahk")
        self.assertIn("ExitApp 0", smoke)
        self.assertIn("ExitApp 1", smoke)
        self.assertNotIn("ExitApp(", smoke)

        worker = source("benchmark/bench_cnumpy.ahk")
        self.assertRegex(worker, r"exitCode\s*:=\s*Main\s*\(\s*A_Args\s*\)")
        self.assertRegex(worker, r"ExitApp\s+exitCode")
        self.assertIn("ExitApp 1", worker)
        self.assertNotIn("ExitApp(", worker)

    def test_ahk_typename_errors_use_standalone_exception(self) -> None:
        wrapper = source("ahk/numpy.ahk")
        self.assertIn("class NumpyKeyError extends Error", wrapper)
        self.assertIn("throw NumpyKeyError(message)", wrapper)
        self.assertNotRegex(wrapper, r"\bthrow\s+KeyError\s*\(")

    def test_shared_constructor_uses_merged_metadata_and_late_incref(self) -> None:
        internal = source("include/cnumpy/cnumpy_internal.h")
        array_source = source("src/array.c")
        self.assertRegex(internal, r"\bcnp_array_view_from_metadata\s*\(")
        body = function_body(array_source, "cnp_array_view_from_metadata")
        self.assertRegex(
            body,
            r"cnp_malloc\s*\(\s*2\s*\*[^;]*ndim[^;]*sizeof\s*\(\s*int64_t\s*\)",
        )
        self.assertRegex(
            body,
            r"view->strides\s*=\s*metadata\s*\?\s*metadata\s*\+\s*ndim\s*:\s*NULL",
        )
        self.assertIn("CNP_ARRAY_OWNDATA", body)
        self.assertIn("CNP_ARRAY_VIRTUAL_ALLOC", body)
        self.assertIn("layout_flags", body)
        self.assertLess(body.index("cnp_malloc"), body.index("cnp_array_incref"))

    def test_array_view_and_reshape_adapter_delegate(self) -> None:
        array_source = source("src/array.c")
        for function_name in ("cnp_array_view", "cnp_array_reshape_view"):
            with self.subTest(function_name=function_name):
                body = function_body(array_source, function_name)
                self.assertIn("cnp_array_view_from_metadata", body)
                self.assertNotIn("alloc_array_struct", body)
        reshape = function_body(array_source, "cnp_array_reshape_view")
        self.assertIn("cnp_compute_strides", reshape)

    def test_shape_and_slice_paths_construct_metadata_once(self) -> None:
        cases = (
            ("src/shape.c", "cnp_transpose"),
            ("src/shape.c", "cnp_broadcast_to"),
            ("src/indexing.c", "cnp_array_slice"),
        )
        for path, function_name in cases:
            with self.subTest(function_name=function_name):
                body = function_body(source(path), function_name)
                self.assertIn("cnp_array_view_from_metadata", body)
                self.assertNotRegex(body, r"\bcnp_array_view\s*\(")
                self.assertNotRegex(body, r"cnp_free\s*\(\s*result->(shape|strides)")
                self.assertNotRegex(
                    body,
                    r"result->(shape|strides)\s*=\s*\(int64_t\*\)cnp_malloc",
                )

    def test_as_strided_delegates_with_inherited_offset(self) -> None:
        body = function_body(source("src/stride_tricks.c"), "cnp_as_strided")
        self.assertIn("cnp_array_view_from_metadata", body)
        self.assertIn("arr->offset", body)
        self.assertNotIn("result->offset = 0", body)
        self.assertNotIn("result->base", body)

    def test_transpose_rejects_duplicate_axes_before_construction(self) -> None:
        body = function_body(source("src/shape.c"), "cnp_transpose")
        self.assertIn("seen_axes", body)
        self.assertIn("Duplicate axis", body)
        self.assertLess(
            body.index("Duplicate axis"),
            body.index("cnp_array_view_from_metadata"),
        )


class IndexingAxisTraversalContracts(unittest.TestCase):
    def test_axis0_smoke_asserts_distinct_block_and_strided_layouts(self) -> None:
        body = function_body(
            source("benchmark/benchmark_smoke.test.ahk"),
            "TestAxis0IndexingBlockWorkers",
        )

        self.assertIn('"Ptr", prepared.IndexingSourceHandle', body)
        self.assertIn('"\\cnp_ahk_flags"', body)
        self.assertIn("Numpy.ARRAY_C_CONTIGUOUS", body)
        self.assertIn("Numpy.ARRAY_F_CONTIGUOUS", body)
        self.assertIn("AssertTrue(isCContiguous", body)
        self.assertIn("AssertTrue(!isCContiguous", body)
        self.assertIn("AssertTrue(isFContiguous", body)

    def test_take_along_axis_uses_shared_checked_axis_traversal(self) -> None:
        indexing = source("src/indexing.c")
        body = function_body(indexing, "cnp_take_along_axis_v2")

        self.assertIn("CnpAxisTraversal traversal", body)
        self.assertIn("indexing_axis_traversal", body)
        self.assertIn("indexing_copy_axis_element", body)
        self.assertNotRegex(
            body,
            r"memcpy\s*\([^;]*arr->data",
        )


class IntoBatchContracts(unittest.TestCase):
    def test_into_api_is_public_and_compiled_in_dedicated_source(self) -> None:
        public = source("include/cnumpy/cnumpy.h")
        project = source("src/cnumpy_ahk.vcxproj")
        for symbol in (
            "cnp_add_into",
            "cnp_sqrt_into",
            "cnp_cumsum_into",
            "cnp_sum_into_scalar",
        ):
            with self.subTest(symbol=symbol):
                self.assertRegex(public, rf"\b{symbol}\s*\(")
        self.assertIn('<ClCompile Include="into_ops.c" />', project)
        self.assertIn('<ClCompile Include="cnumpy_ahk_batch.c" />', project)
        self.assertIn(
            '<ClInclude Include="..\\include\\cnumpy\\cnumpy_ahk.h" />',
            project,
        )

    def test_into_functions_use_direct_kernels_without_allocating_fallbacks(
        self,
    ) -> None:
        path = ROOT / "src/into_ops.c"
        self.assertTrue(path.is_file(), "src/into_ops.c must exist")
        implementation = path.read_text(encoding="utf-8")
        required_tokens = {
            "cnp_add_into": ("cnp_simd_add",),
            "cnp_sqrt_into": ("cnp_simd_sqrt",),
            "cnp_cumsum_into": (
                "accumulator += source_data[index]",
                "out_data[index] = accumulator",
            ),
            "cnp_sum_into_scalar": (
                "cnp_reduction_sum_contiguous_double",
            ),
        }
        forbidden = (
            "cnp_array_new",
            "cnp_malloc",
            "cnp_add(",
            "cnp_sqrt(",
            "cnp_cumsum(",
            "cnp_sum(",
            "cnp_simd_sum",
        )
        for function_name, tokens in required_tokens.items():
            with self.subTest(function_name=function_name):
                body = function_body(implementation, function_name)
                for token in tokens:
                    self.assertIn(token, body)
                self.assertIn("validate_f64_contiguous", body)
                for token in forbidden:
                    self.assertNotIn(token, body)

    def test_sum_into_scalar_reuses_the_order_preserving_pairwise_tree(
        self,
    ) -> None:
        implementation = source("src/reduce.c")
        body = function_body(
            implementation, "cnp_reduction_sum_contiguous_double"
        )
        self.assertIn("reduction_pairwise_sum_contiguous_double", body)
        self.assertIn("reduction_apply_sum_identity_double", body)
        self.assertNotRegex(body, r"\bcnp_(?:simd|array_new|malloc)")

    def test_fixed_batch_layout_and_opcodes_are_stable(self) -> None:
        header_path = ROOT / "include/cnumpy/cnumpy_ahk.h"
        batch_path = ROOT / "src/cnumpy_ahk_batch.c"
        self.assertTrue(header_path.is_file(), "cnumpy_ahk.h must exist")
        self.assertTrue(batch_path.is_file(), "cnumpy_ahk_batch.c must exist")
        header = header_path.read_text(encoding="utf-8")
        batch = batch_path.read_text(encoding="utf-8")
        for name, value in (
            ("CNP_AHK_BATCH_ADD_INTO", 1),
            ("CNP_AHK_BATCH_SQRT_INTO", 2),
            ("CNP_AHK_BATCH_CUMSUM_INTO", 3),
            ("CNP_AHK_BATCH_SUM_SCALAR", 4),
        ):
            self.assertRegex(header, rf"#define\s+{name}\s+{value}u")
        for field in ("opcode", "reserved", "input0", "input1", "output", "axis"):
            self.assertIn(field, header)
        self.assertIn("sizeof(CnpAhkBatchCommand) == 40", batch)

    def test_batch_reports_first_failure_without_dynamic_dispatch(self) -> None:
        path = ROOT / "src/cnumpy_ahk_batch.c"
        self.assertTrue(path.is_file(), "cnumpy_ahk_batch.c must exist")
        body = function_body(
            path.read_text(encoding="utf-8"),
            "cnp_ahk_execute_batch",
        )
        self.assertIn("*failed_index = -1", body)
        self.assertIn("*failed_index = i", body)
        self.assertIn("Unknown batch opcode", body)
        self.assertIn("command->reserved", body)
        self.assertNotIn("cnp_malloc", body)
        self.assertNotRegex(body, r"\(\s*\*\s*\w+\s*\)\s*\(")

    def test_ahk_wrapper_exposes_destination_and_batch_methods(self) -> None:
        wrapper = source("ahk/numpy.ahk")
        for method in (
            "AddInto(",
            "SqrtInto(",
            "CumsumInto(",
            "SumIntoScalar(",
            "AddSqrtSumBatch(",
        ):
            with self.subTest(method=method):
                self.assertIn(method, wrapper)


class AhkReductionContracts(unittest.TestCase):
    OPERATIONS = ("sum", "prod", "mean", "var", "std", "max", "min")
    AXIS_V2 = (
        "max", "min", "argmax", "argmin", "any", "all", "ptp",
        "nanmax", "nanmin", "nanargmax", "nanargmin", "median",
        "nanmedian",
    )
    DTYPE_V2 = (
        "sum", "prod", "mean", "cumsum", "cumprod", "nansum",
        "nanprod", "nanmean", "nancumsum", "nancumprod",
    )
    DEVIATION_V2 = ("var", "std", "nanvar", "nanstd")
    PERCENTILE_V2 = (
        "percentile", "nanpercentile", "quantile", "nanquantile",
    )

    def test_axis_array_exports_are_declared_and_defined(self) -> None:
        header = source("include/cnumpy/cnumpy_ahk.h")
        bridge = source("src/cnumpy_ahk.c")
        for operation in self.OPERATIONS:
            symbol = f"cnp_ahk_{operation}_array"
            with self.subTest(symbol=symbol):
                self.assertRegex(header, rf"\b{symbol}\s*\(")
                self.assertRegex(bridge, rf"\b{symbol}\s*\(")

    def test_order_sensitive_scalar_exports_do_not_use_reassociating_simd(self) -> None:
        bridge = source("src/cnumpy_ahk.c")
        for operation in ("sum", "prod", "mean"):
            body = function_body(bridge, f"cnp_ahk_{operation}")
            with self.subTest(operation=operation):
                self.assertNotIn("cnp_simd_sum", body)
                self.assertNotIn("cnp_simd_prod", body)
                self.assertIn(f"cnp_{operation}_v2", body)

    def test_other_scalar_exports_route_to_shared_v2_semantics(self) -> None:
        bridge = source("src/cnumpy_ahk.c")
        for operation in ("var", "std", "max", "min"):
            body = function_body(bridge, f"cnp_ahk_{operation}")
            with self.subTest(operation=operation):
                self.assertIn(f"cnp_{operation}_v2", body)
                self.assertNotRegex(body, r"\bcnp_simd_")

    def test_real_sum_and_prod_v2_use_the_shared_traversal(self) -> None:
        implementation = source("src/reduce.c")
        for operation in ("sum", "prod"):
            body = function_body(implementation, f"cnp_{operation}_v2")
            with self.subTest(operation=operation):
                self.assertIn("reduction_sumprod_real", body)
                self.assertNotIn(f"return cnp_{operation}(", body)
                self.assertNotIn("cnp_simd_", body)

    def test_scalar_sum_and_mean_route_through_owned_v2_results(self) -> None:
        implementation = source("src/reduce.c")
        for operation in ("sum", "mean"):
            body = function_body(
                implementation, f"cnp_{operation}_scalar"
            )
            with self.subTest(operation=operation):
                self.assertIn(f"cnp_{operation}_v2", body)
                self.assertIn("cnp_get_element_double", body)
                self.assertIn("cnp_array_free", body)
                self.assertIn("cnp_relabel_error", body)
                self.assertNotIn("coords", body)

    def test_every_reduction_v2_bridge_has_an_exact_public_declaration(
        self,
    ) -> None:
        header = source("include/cnumpy/cnumpy_ahk.h")
        bridge = source("src/cnumpy_ahk.c")
        self.assertEqual(
            31,
            len(self.AXIS_V2)
            + len(self.DTYPE_V2)
            + len(self.DEVIATION_V2)
            + len(self.PERCENTILE_V2),
        )
        families = (
            (
                "AXIS",
                self.AXIS_V2,
                r"void \*handle,\s*int axis,\s*int axis_none",
            ),
            (
                "DTYPE",
                self.DTYPE_V2,
                r"void \*handle,\s*int axis,\s*int axis_none",
            ),
            (
                "DEVIATION",
                self.DEVIATION_V2,
                r"void \*handle,\s*int axis,\s*int axis_none,\s*int ddof",
            ),
            (
                "PERCENTILE",
                self.PERCENTILE_V2,
                r"void \*handle,\s*double q,\s*int axis,\s*int axis_none",
            ),
        )
        for family, operations, arguments in families:
            for operation in operations:
                symbol = f"cnp_ahk_{operation}_v2"
                with self.subTest(family=family, symbol=symbol):
                    self.assertRegex(
                        header,
                        rf"CNP_API\s+void\s*\*\s*CNP_CALL\s+{symbol}\s*"
                        rf"\(\s*{arguments}\s*\)\s*;",
                    )
                    self.assertIn(
                        f"AHK_{family}_REDUCTION_V2("
                        f"{operation}, cnp_{operation}_v2)",
                        bridge,
                    )

    def test_reduction_v2_macros_raise_and_relabel_bridge_errors(self) -> None:
        bridge = source("src/cnumpy_ahk.c")
        macro_names = (
            "AHK_AXIS_REDUCTION_V2",
            "AHK_DTYPE_REDUCTION_V2",
            "AHK_DEVIATION_REDUCTION_V2",
            "AHK_PERCENTILE_REDUCTION_V2",
        )
        for index, macro_name in enumerate(macro_names):
            start = bridge.index(f"#define {macro_name}")
            end = (
                bridge.index(f"#define {macro_names[index + 1]}", start)
                if index + 1 < len(macro_names)
                else bridge.index("AHK_DTYPE_REDUCTION_V2(sum", start)
            )
            body = bridge[start:end]
            for token in (
                'const char *function_name = "cnp_ahk_" #name "_v2"',
                "if (!handle)",
                "cnp_set_error",
                '"source array must not be null"',
                "CnpArray *result",
                "if (!result) cnp_relabel_error(function_name)",
            ):
                with self.subTest(macro=macro_name, token=token):
                    self.assertIn(token, body)

    def test_wrapper_distinguishes_omitted_axis_from_explicit_negative_one(self) -> None:
        wrapper = source("ahk/numpy.ahk")
        self.assertIn("_ReductionV2(", wrapper)
        self.assertIn("axisNone := !IsSet(axis)", wrapper)
        for operation in self.OPERATIONS:
            method = operation.capitalize()
            with self.subTest(method=method):
                if operation in {"var", "std"}:
                    self.assertRegex(
                        wrapper,
                        rf"\b{method}\s*\(axis\s*:=\s*unset,\s*ddof\s*:=\s*0\)",
                    )
                else:
                    self.assertRegex(
                        wrapper,
                        rf"\b{method}\s*\(axis\s*:=\s*unset\)",
                    )
                self.assertIn(f'"cnp_ahk_{operation}_v2"', wrapper)


class BlockedTransposeCopyContracts(unittest.TestCase):
    def test_array_copy_has_a_blocked_2d_float64_path(self) -> None:
        body = function_body(source("src/array.c"), "cnp_array_copy")
        for token in (
            "src->ndim == 2",
            "CNP_DOUBLE",
            "CNP_COPY_TILE_SIZE",
            "src->strides[0]",
            "src->strides[1]",
            "row_block",
            "column_block",
        ):
            with self.subTest(token=token):
                self.assertIn(token, body)
        self.assertRegex(source("src/array.c"), r"#define\s+CNP_COPY_TILE_SIZE\s+32")
        if "row_block" in body:
            self.assertLess(body.index("row_block"), body.index("coords"))

    def test_fused_export_and_wrapper_methods_are_exposed(self) -> None:
        header = source("include/cnumpy/cnumpy_ahk.h")
        bridge = source("src/cnumpy_ahk.c")
        wrapper = source("ahk/numpy.ahk")
        self.assertRegex(header, r"\bcnp_ahk_transpose_copy\s*\(")
        body = function_body(bridge, "cnp_ahk_transpose_copy")
        for token in ("cnp_transpose", "cnp_array_copy", "cnp_array_free"):
            with self.subTest(token=token):
                self.assertIn(token, body)
        self.assertIn("TransposeCopy(", wrapper)
        self.assertIn('"cnp_ahk_transpose_copy"', wrapper)
        self.assertIn("AsContiguousArray(", wrapper)

    def test_benchmark_uses_one_fused_dllcall(self) -> None:
        worker = source("benchmark/bench_cnumpy.ahk")
        body = function_body(worker, "InvokeTransposeCopy")
        self.assertIn("cnp_ahk_transpose_copy", body)
        self.assertNotIn("cnp_ahk_transpose\"", body)
        self.assertNotIn("cnp_ahk_copy", body)
        self.assertNotIn("cnp_ahk_free", body)


class ElementwiseDispatchContracts(unittest.TestCase):
    BINARY_OPERATIONS = ("add", "subtract", "multiply", "divide")
    EXTREMA_OPERATIONS = ("maximum", "minimum", "fmax", "fmin")
    LOGICAL_OPERATIONS = (
        "logical_and", "logical_or", "logical_xor", "logical_not",
    )
    UNARY_OPERATIONS = ("negative", "absolute", "sqrt", "floor")

    def test_sse2_avx2_and_stable_kernels_are_declared_and_defined(self) -> None:
        internal = source("include/cnumpy/cnumpy_internal.h")
        baseline = source("src/simd_ops.c")
        avx2 = source("src/simd_avx2.c")
        dispatch = source("src/simd_dispatch.c")
        for operation in (
            self.BINARY_OPERATIONS + self.EXTREMA_OPERATIONS +
            self.UNARY_OPERATIONS
        ):
            with self.subTest(operation=operation):
                self.assertRegex(internal, rf"\bcnp_sse2_{operation}\s*\(")
                self.assertRegex(internal, rf"\bcnp_avx2_{operation}\s*\(")
                self.assertRegex(internal, rf"\bcnp_simd_{operation}\s*\(")
                self.assertRegex(baseline, rf"\bcnp_sse2_{operation}\s*\(")
                self.assertRegex(avx2, rf"\bcnp_avx2_{operation}\s*\(")
                self.assertRegex(dispatch, rf"\bcnp_simd_{operation}\s*\(")

    def test_avx2_kernels_encode_required_ieee_operations(self) -> None:
        avx2 = source("src/simd_avx2.c")
        for token in (
            "_mm256_div_pd",
            "_mm256_floor_pd",
            "_mm256_cmp_pd",
            "_CMP_UNORD_Q",
            "_mm256_blendv_pd",
        ):
            with self.subTest(token=token):
                self.assertIn(token, avx2)

    def test_allocating_public_operations_route_before_elementwise_engines(self) -> None:
        math_ops = source("src/math_ops.c")
        elementwise_engines = {
            **{operation: "cnp_binary_op" for operation in self.BINARY_OPERATIONS},
            **{operation: "cnp_unary_op" for operation in self.UNARY_OPERATIONS},
            "add": "arithmetic_promoted_arrays",
            "subtract": "arithmetic_promoted_arrays",
            "multiply": "arithmetic_promoted_arrays",
            "divide": "arithmetic_promoted_arrays",
        }
        for operation, elementwise_engine in elementwise_engines.items():
            body = function_body(math_ops, f"cnp_{operation}")
            kernel = f"cnp_simd_{operation}"
            with self.subTest(operation=operation):
                if operation == "negative":
                    self.assertIn("unary_sign_arrays", body)
                    body = function_body(math_ops, "unary_sign_arrays")
                    elementwise_engine = "unary_sign_element"
                if operation == "sqrt":
                    self.assertIn("unary_sqrt_arrays", body)
                    body = function_body(math_ops, "unary_sqrt_arrays")
                    elementwise_engine = "unary_sqrt_element"
                self.assertIn("CNP_ARRAY_C_CONTIGUOUS", body)
                self.assertIn("CNP_DOUBLE", body)
                self.assertIn(kernel, body)
                self.assertIn(elementwise_engine, body)
                self.assertLess(body.index(kernel), body.index(elementwise_engine))

    def test_extrema_share_typed_engine_and_runtime_dispatched_kernels(
        self,
    ) -> None:
        math_ops = source("src/math_ops.c")
        engine = function_body(math_ops, "extrema_arrays")
        contiguous = function_body(
            math_ops, "extrema_contiguous_double"
        )
        self.assertIn("extrema_select_left", engine)
        self.assertIn("extrema_contiguous_double", engine)
        for operation in self.EXTREMA_OPERATIONS:
            with self.subTest(operation=operation):
                public_body = function_body(
                    math_ops, f"cnp_{operation}"
                )
                self.assertIn("extrema_arrays", public_body)
                self.assertIn(f"cnp_simd_{operation}", contiguous)

    def test_logical_float64_fast_paths_are_runtime_dispatched_and_strict(
        self,
    ) -> None:
        internal = source("include/cnumpy/cnumpy_internal.h")
        baseline = source("src/simd_ops.c")
        avx2 = source("src/simd_avx2.c")
        dispatch = source("src/simd_dispatch.c")
        broadcast = source("src/broadcast.c")
        math_ops = source("src/math_ops.c")

        for operation in self.LOGICAL_OPERATIONS:
            with self.subTest(operation=operation):
                self.assertRegex(
                    internal, rf"\bcnp_sse2_{operation}\s*\("
                )
                self.assertRegex(
                    internal, rf"\bcnp_avx2_{operation}\s*\("
                )
                self.assertRegex(
                    internal, rf"\bcnp_simd_{operation}\s*\("
                )
                self.assertRegex(
                    baseline, rf"\bcnp_sse2_{operation}\s*\("
                )
                self.assertRegex(
                    avx2, rf"\bcnp_avx2_{operation}\s*\("
                )
                self.assertRegex(
                    dispatch, rf"\bcnp_simd_{operation}\s*\("
                )

        binary_engine = function_body(broadcast, "cnp_logical_op")
        for token in (
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_DOUBLE",
            "simd_kernel",
            "cnp_binary_logical_truth",
        ):
            with self.subTest(binary_engine=token):
                self.assertIn(token, binary_engine)
        for operation in ("logical_and", "logical_or", "logical_xor"):
            public_body = function_body(math_ops, f"cnp_{operation}")
            self.assertIn(f"cnp_simd_{operation}", public_body)

        unary_body = function_body(math_ops, "cnp_logical_not")
        for token in (
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_DOUBLE",
            "cnp_simd_logical_not",
            "cnp_scalar_truth",
        ):
            with self.subTest(unary_engine=token):
                self.assertIn(token, unary_body)
        self.assertLess(
            unary_body.index("cnp_simd_logical_not"),
            unary_body.index("cnp_scalar_truth"),
        )

        sse2_logical = (
            function_body(baseline, "cnp_sse2_truth_mask_f64x2")
            + function_body(baseline, "cnp_sse2_logical_binary2")
            + function_body(baseline, "cnp_sse2_logical_not")
        )
        for token in (
            "_mm_cmpeq_epi32", "_mm_shuffle_epi32", "_mm_movemask_pd",
        ):
            with self.subTest(sse2=token):
                self.assertIn(token, sse2_logical)
        self.assertNotIn("_mm_cmpneq_pd", sse2_logical)

        avx2_logical = (
            function_body(avx2, "cnp_avx2_truth_mask_f64x4")
            + function_body(avx2, "cnp_avx2_logical_binary4")
            + function_body(avx2, "cnp_avx2_logical_not")
        )
        for token in (
            "_mm256_and_si256",
            "_mm256_cmpeq_epi64",
            "_mm256_movemask_pd",
        ):
            with self.subTest(avx2=token):
                self.assertIn(token, avx2_logical)
        self.assertNotIn("_CMP_NEQ_UQ", avx2_logical)

    def test_core_arithmetic_operations_share_the_typed_engine(
        self,
    ) -> None:
        math_ops = source("src/math_ops.c")
        engine = function_body(math_ops, "arithmetic_promoted_element")
        for token in (
            "CNP_ARITHMETIC_ADD",
            "CNP_ARITHMETIC_SUBTRACT",
            "CNP_ARITHMETIC_MULTIPLY",
            "CNP_ARITHMETIC_DIVIDE",
            "CNP_ARITHMETIC_FLOOR_DIVIDE",
            "CNP_ARITHMETIC_REMAINDER",
            "CNP_ARITHMETIC_FMOD",
            "CNP_BOOL",
            "CNP_HALF",
            "CNP_CLONGDOUBLE",
        ):
            with self.subTest(token=token):
                self.assertIn(token, engine)
        for operation in (
            "add",
            "subtract",
            "multiply",
            "divide",
            "floor_divide",
            "mod",
            "remainder",
            "fmod",
        ):
            with self.subTest(operation=operation):
                body = function_body(math_ops, f"cnp_{operation}")
                self.assertIn("arithmetic_promoted_arrays", body)

    def test_bitwise_shifts_use_a_direct_contiguous_integer_path(self) -> None:
        math_ops = source("src/math_ops.c")
        engine = function_body(math_ops, "bitwise_binary_arrays")
        contiguous = function_body(math_ops, "bitwise_shift_contiguous")
        self.assertIn("bitwise_shift_contiguous", engine)
        self.assertLess(
            engine.index("bitwise_shift_contiguous"),
            engine.index("coordinates"),
        )
        for token in (
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_ARRAY_F_CONTIGUOUS",
            "same_shape",
            "CNP_BOOL",
            "CNP_BYTE",
            "CNP_LONGLONG",
            "bitwise_apply_shift",
        ):
            with self.subTest(token=token):
                self.assertIn(token, contiguous)

    def test_power_uses_typed_broadcast_and_direct_contiguous_paths(
        self,
    ) -> None:
        math_ops = source("src/math_ops.c")
        for operation, float_power in (
            ("power", "false"),
            ("float_power", "true"),
        ):
            with self.subTest(operation=operation):
                body = function_body(math_ops, f"cnp_{operation}")
                self.assertIn("power_arrays", body)
                self.assertIn(float_power, body)
                self.assertNotIn("cnp_binary_op", body)

        combined = "".join(
            function_body(math_ops, name)
            for name in (
                "power_result_dtype",
                "power_validate_inputs",
                "power_integer_bits",
                "power_cfloat",
                "power_cdouble",
                "power_clongdouble",
                "power_element",
                "power_contiguous_typed",
                "power_arrays",
            )
        )
        for token in (
            "CNP_BOOL",
            "CNP_HALF",
            "CNP_FLOAT",
            "CNP_DOUBLE",
            "CNP_LONGDOUBLE",
            "CNP_CFLOAT",
            "CNP_CDOUBLE",
            "CNP_CLONGDOUBLE",
            "powf",
            "pow(",
            "powl",
            "cpowf",
            "cpow(",
            "cpowl",
            "Integers to negative integer powers are not allowed",
            "CNP_ERR_TYPE",
            "CNP_ERR_BROADCAST",
            "arithmetic_prepare_result",
            "arithmetic_broadcast_offset",
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_ARRAY_F_CONTIGUOUS",
            "same_shape",
            "index + 3 < result->size",
        ):
            with self.subTest(token=token):
                self.assertIn(token, combined)
        self.assertIn(
            "arithmetic_result_order",
            function_body(math_ops, "arithmetic_prepare_result"),
        )
        contiguous = function_body(math_ops, "power_contiguous_typed")
        self.assertLess(
            contiguous.index("result->size == 0"),
            contiguous.index("(const char*)left->data"),
        )

    def test_negative_and_positive_share_the_typed_unary_sign_engine(
        self,
    ) -> None:
        math_ops = source("src/math_ops.c")
        for operation, mode in (
            ("negative", "CNP_UNARY_SIGN_NEGATIVE"),
            ("positive", "CNP_UNARY_SIGN_POSITIVE"),
        ):
            with self.subTest(operation=operation):
                body = function_body(math_ops, f"cnp_{operation}")
                self.assertIn("unary_sign_arrays", body)
                self.assertIn(mode, body)
        engine = function_body(math_ops, "unary_sign_arrays")
        element = function_body(math_ops, "unary_sign_element")
        for token in (
            "CNP_HALF",
            "CNP_CFLOAT",
            "CNP_CDOUBLE",
            "CNP_CLONGDOUBLE",
            "CNP_TIMEDELTA",
            "cnp_multi_to_offset",
        ):
            with self.subTest(token=token):
                self.assertIn(token, engine + element)

    def test_negative_and_positive_fast_paths_accept_both_contiguous_orders(
        self,
    ) -> None:
        engine = function_body(source("src/math_ops.c"), "unary_sign_arrays")
        contiguous_orders = (
            r"\(\s*CNP_ARRAY_C_CONTIGUOUS\s*\|\s*"
            r"CNP_ARRAY_F_CONTIGUOUS\s*\)"
        )
        self.assertRegex(
            engine,
            r"operation\s*==\s*CNP_UNARY_SIGN_NEGATIVE\s*&&\s*"
            r"\(source->flags\s*&\s*" + contiguous_orders + r"\)\s*&&\s*"
            r"dtype\s*==\s*CNP_DOUBLE",
        )
        self.assertIn("cnp_simd_negative", engine)
        self.assertRegex(
            engine,
            r"operation\s*==\s*CNP_UNARY_SIGN_POSITIVE\s*&&\s*"
            r"\(source->flags\s*&\s*" + contiguous_orders + r"\)",
        )
        self.assertIn("memcpy", engine)

    def test_sign_uses_the_typed_keep_order_engine(self) -> None:
        math_ops = source("src/math_ops.c")
        public_body = function_body(math_ops, "cnp_sign")
        self.assertIn("unary_signum_arrays", public_body)
        engine = function_body(math_ops, "unary_signum_arrays")
        element = function_body(math_ops, "unary_signum_element")
        for token in (
            "CNP_HALF",
            "CNP_CFLOAT",
            "CNP_CDOUBLE",
            "CNP_CLONGDOUBLE",
            "CNP_TIMEDELTA",
            "unary_set_keep_order_layout",
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_ARRAY_F_CONTIGUOUS",
            "cnp_multi_to_offset",
        ):
            with self.subTest(token=token):
                self.assertIn(token, engine + element)

    def test_reciprocal_uses_the_typed_keep_order_engine(self) -> None:
        math_ops = source("src/math_ops.c")
        public_body = function_body(math_ops, "cnp_reciprocal")
        self.assertIn("unary_reciprocal_arrays", public_body)
        engine = function_body(math_ops, "unary_reciprocal_arrays")
        element = function_body(math_ops, "unary_reciprocal_element")
        for token in (
            "CNP_BOOL",
            "CNP_HALF",
            "CNP_CFLOAT",
            "CNP_CDOUBLE",
            "CNP_CLONGDOUBLE",
            "unary_set_keep_order_layout",
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_ARRAY_F_CONTIGUOUS",
            "cnp_multi_to_offset",
        ):
            with self.subTest(token=token):
                self.assertIn(token, engine + element)

    def test_square_uses_the_typed_keep_order_engine(self) -> None:
        math_ops = source("src/math_ops.c")
        public_body = function_body(math_ops, "cnp_square")
        self.assertIn("unary_square_arrays", public_body)
        engine = function_body(math_ops, "unary_square_arrays")
        element = function_body(math_ops, "unary_square_element")
        for token in (
            "CNP_BOOL",
            "CNP_HALF",
            "CNP_CFLOAT",
            "CNP_CDOUBLE",
            "CNP_CLONGDOUBLE",
            "unary_set_keep_order_layout",
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_ARRAY_F_CONTIGUOUS",
            "cnp_simd_multiply",
            "cnp_multi_to_offset",
        ):
            with self.subTest(token=token):
                self.assertIn(token, engine + element)

    def test_sqrt_uses_the_typed_keep_order_and_simd_engine(self) -> None:
        math_ops = source("src/math_ops.c")
        public_body = function_body(math_ops, "cnp_sqrt")
        self.assertIn("unary_sqrt_arrays", public_body)
        result_types = function_body(math_ops, "unary_sqrt_result_dtype")
        engine = function_body(math_ops, "unary_sqrt_arrays")
        element = function_body(math_ops, "unary_sqrt_element")
        complex_loop = function_body(math_ops, "unary_sqrt_cfloat")
        for token in (
            "CNP_BOOL",
            "CNP_HALF",
            "CNP_CFLOAT",
            "CNP_CDOUBLE",
            "CNP_CLONGDOUBLE",
            "unary_set_keep_order_layout",
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_ARRAY_F_CONTIGUOUS",
            "cnp_simd_sqrt",
            "cnp_multi_to_offset",
        ):
            with self.subTest(token=token):
                self.assertIn(token, result_types + engine + element)
        self.assertIn("unary_sqrt_hypotf", complex_loop)
        self.assertIn("threshold", complex_loop)

    def test_cbrt_uses_the_typed_keep_order_engine(self) -> None:
        math_ops = source("src/math_ops.c")
        public_body = function_body(math_ops, "cnp_cbrt")
        self.assertIn("unary_cbrt_arrays", public_body)
        result_types = function_body(math_ops, "unary_cbrt_result_dtype")
        engine = function_body(math_ops, "unary_cbrt_arrays")
        element = function_body(math_ops, "unary_cbrt_element")
        for token in (
            "CNP_BOOL",
            "CNP_HALF",
            "CNP_FLOAT",
            "CNP_DOUBLE",
            "CNP_LONGDOUBLE",
            "unary_set_keep_order_layout",
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_ARRAY_F_CONTIGUOUS",
            "cnp_multi_to_offset",
            "cbrtf",
            "cbrt",
            "cbrtl",
        ):
            with self.subTest(token=token):
                self.assertIn(token, result_types + engine + element)
        for token in ("CNP_CFLOAT", "CNP_CDOUBLE", "CNP_CLONGDOUBLE"):
            with self.subTest(unsupported=token):
                self.assertNotIn(token, result_types)

    def test_conjugate_aliases_use_one_typed_keep_order_engine(self) -> None:
        math_ops = source("src/math_ops.c")
        for symbol in ("cnp_conj", "cnp_conjugate"):
            with self.subTest(symbol=symbol):
                self.assertIn(
                    "unary_conjugate_arrays",
                    function_body(math_ops, symbol),
                )
        engine = function_body(math_ops, "unary_conjugate_arrays")
        element = function_body(math_ops, "unary_conjugate_element")
        for token in (
            "CNP_BOOL",
            "CNP_CFLOAT",
            "CNP_CDOUBLE",
            "CNP_CLONGDOUBLE",
            "unary_set_keep_order_layout",
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_ARRAY_F_CONTIGUOUS",
            "cnp_multi_to_offset",
            "memcpy",
        ):
            with self.subTest(token=token):
                self.assertIn(token, engine + element)

    def test_cos_uses_the_typed_keep_order_and_native_complex_engine(
        self,
    ) -> None:
        math_ops = source("src/math_ops.c")
        public_body = function_body(math_ops, "cnp_cos")
        self.assertIn("unary_cos_arrays", public_body)
        result_types = function_body(
            math_ops, "unary_trigonometric_result_dtype"
        )
        engine = function_body(math_ops, "unary_cos_arrays")
        element = function_body(math_ops, "unary_cos_element")
        for token in (
            "CNP_BOOL",
            "CNP_HALF",
            "CNP_CFLOAT",
            "CNP_CDOUBLE",
            "CNP_CLONGDOUBLE",
            "unary_set_keep_order_layout",
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_ARRAY_F_CONTIGUOUS",
            "cnp_multi_to_offset",
        ):
            with self.subTest(token=token):
                self.assertIn(token, result_types + engine + element)
        complex_loops = (
            function_body(math_ops, "unary_cos_cfloat"),
            function_body(math_ops, "unary_cos_cdouble"),
            function_body(math_ops, "unary_cos_clongdouble"),
        )
        for body, native_type, native_function in zip(
            complex_loops,
            ("_Fcomplex", "_Dcomplex", "_Lcomplex"),
            ("ccosf", "ccos", "ccosl"),
            strict=True,
        ):
            with self.subTest(native_function=native_function):
                self.assertIn(native_type, body)
                self.assertIn(native_function, body)
                self.assertIn("memcpy", body)

    def test_sin_uses_typed_keep_order_and_numpy_f32_dispatch(self) -> None:
        math_ops = source("src/math_ops.c")
        public_body = function_body(math_ops, "cnp_sin")
        self.assertIn("unary_sin_arrays", public_body)
        result_types = function_body(
            math_ops, "unary_trigonometric_result_dtype"
        )
        engine = function_body(math_ops, "unary_sin_arrays")
        element = function_body(math_ops, "unary_sin_element")
        for token in (
            "CNP_BOOL",
            "CNP_HALF",
            "CNP_CFLOAT",
            "CNP_CDOUBLE",
            "CNP_CLONGDOUBLE",
            "unary_set_keep_order_layout",
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_ARRAY_F_CONTIGUOUS",
            "cnp_multi_to_offset",
        ):
            with self.subTest(token=token):
                self.assertIn(token, result_types + engine + element)
        self.assertIn("cnp_simd_sin_f32", engine)

        complex_loops = (
            function_body(math_ops, "unary_sin_cfloat"),
            function_body(math_ops, "unary_sin_cdouble"),
            function_body(math_ops, "unary_sin_clongdouble"),
        )
        for body, native_type, native_function in zip(
            complex_loops,
            ("_Fcomplex", "_Dcomplex", "_Lcomplex"),
            ("csinf", "csin", "csinl"),
            strict=True,
        ):
            with self.subTest(native_function=native_function):
                self.assertIn(native_type, body)
                self.assertIn(native_function, body)
                self.assertIn("memcpy", body)

        dispatch = source("src/simd_dispatch.c")
        self.assertIn("(1 << 12)", dispatch)
        self.assertIn("g_sin_f32_kernel = cnp_avx2_sin_f32", dispatch)
        avx2 = source("src/simd_avx2.c")
        block = function_body(avx2, "cnp_avx2_sin_f32_block")
        for token in (
            "117435.992f",
            "0x1.45f306p-1f",
            "_mm256_fmadd_ps",
            "sinf",
        ):
            with self.subTest(avx2_token=token):
                self.assertIn(token, block)

    def test_tan_uses_the_typed_keep_order_and_native_complex_engine(
        self,
    ) -> None:
        math_ops = source("src/math_ops.c")
        public_body = function_body(math_ops, "cnp_tan")
        self.assertIn("unary_tan_arrays", public_body)
        self.assertNotIn("cnp_unary_op", public_body)
        result_types = function_body(
            math_ops, "unary_trigonometric_result_dtype"
        )
        engine = function_body(math_ops, "unary_tan_arrays")
        element = function_body(math_ops, "unary_tan_element")
        for token in (
            "CNP_BOOL",
            "CNP_HALF",
            "CNP_CFLOAT",
            "CNP_CDOUBLE",
            "CNP_CLONGDOUBLE",
            "unary_set_keep_order_layout",
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_ARRAY_F_CONTIGUOUS",
            "cnp_multi_to_offset",
            "tanf",
            "tan",
            "tanl",
        ):
            with self.subTest(token=token):
                self.assertIn(token, result_types + engine + element)

        complex_loops = (
            function_body(math_ops, "unary_tan_cfloat"),
            function_body(math_ops, "unary_tan_cdouble"),
            function_body(math_ops, "unary_tan_clongdouble"),
        )
        for body, native_type, native_function in zip(
            complex_loops,
            ("_Fcomplex", "_Dcomplex", "_Lcomplex"),
            ("ctanf", "ctan", "ctanl"),
            strict=True,
        ):
            with self.subTest(native_function=native_function):
                self.assertIn(native_type, body)
                self.assertIn(native_function, body)
                self.assertIn("memcpy", body)

    def test_arcsin_uses_typed_keep_order_and_numpy_complex_engine(
        self,
    ) -> None:
        math_ops = source("src/math_ops.c")
        public_body = function_body(math_ops, "cnp_arcsin")
        self.assertIn("unary_arcsin_arrays", public_body)
        self.assertNotIn("cnp_unary_op", public_body)
        result_types = function_body(
            math_ops, "unary_trigonometric_result_dtype"
        )
        engine = function_body(math_ops, "unary_arcsin_arrays")
        element = function_body(math_ops, "unary_arcsin_element")
        for token in (
            "CNP_BOOL",
            "CNP_HALF",
            "CNP_CFLOAT",
            "CNP_CDOUBLE",
            "CNP_CLONGDOUBLE",
            "unary_set_keep_order_layout",
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_ARRAY_F_CONTIGUOUS",
            "cnp_multi_to_offset",
            "asinf",
            "asin",
            "asinl",
        ):
            with self.subTest(token=token):
                self.assertIn(token, result_types + engine + element)

        cfloat = function_body(math_ops, "unary_arcsin_cfloat")
        cdouble = function_body(math_ops, "unary_arcsin_cdouble")
        self.assertIn("unary_arcsin_numpy_asinh_float", cfloat)
        self.assertIn("unary_arcsin_numpy_asinh_double", cdouble)
        numpy_float = function_body(
            math_ops, "unary_arcsin_numpy_asinh_float"
        )
        numpy_double = function_body(
            math_ops, "unary_arcsin_numpy_asinh_double"
        )
        for token in (
            "reciprocal_epsilon",
            "unary_arcsin_numpy_clog_large_float",
            "unary_arcsin_numpy_hard_work_float",
            "copysignf",
        ):
            with self.subTest(float_token=token):
                self.assertIn(token, numpy_float)
        for token in (
            "reciprocal_epsilon",
            "unary_arcsin_numpy_clog_large_double",
            "unary_arcsin_numpy_hard_work_double",
            "copysign",
        ):
            with self.subTest(double_token=token):
                self.assertIn(token, numpy_double)

    def test_arccos_uses_typed_keep_order_and_numpy_complex_engine(
        self,
    ) -> None:
        math_ops = source("src/math_ops.c")
        public_body = function_body(math_ops, "cnp_arccos")
        self.assertIn("unary_arccos_arrays", public_body)
        self.assertNotIn("cnp_unary_op", public_body)
        result_types = function_body(
            math_ops, "unary_trigonometric_result_dtype"
        )
        engine = function_body(math_ops, "unary_arccos_arrays")
        element = function_body(math_ops, "unary_arccos_element")
        for token in (
            "CNP_BOOL",
            "CNP_HALF",
            "CNP_CFLOAT",
            "CNP_CDOUBLE",
            "CNP_CLONGDOUBLE",
            "unary_set_keep_order_layout",
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_ARRAY_F_CONTIGUOUS",
            "cnp_multi_to_offset",
            "acosf",
            "acos",
            "acosl",
        ):
            with self.subTest(token=token):
                self.assertIn(token, result_types + engine + element)

        cfloat = function_body(math_ops, "unary_arccos_cfloat")
        cdouble = function_body(math_ops, "unary_arccos_cdouble")
        self.assertIn("unary_arccos_numpy_float", cfloat)
        self.assertIn("unary_arccos_numpy_double", cdouble)
        numpy_float = function_body(math_ops, "unary_arccos_numpy_float")
        numpy_double = function_body(math_ops, "unary_arccos_numpy_double")
        for token in (
            "pi_over_2_low",
            "unary_arcsin_numpy_clog_large_float",
            "unary_arcsin_numpy_hard_work_float",
            "acosf",
            "atan2f",
        ):
            with self.subTest(float_token=token):
                self.assertIn(token, numpy_float)
        for token in (
            "pi_over_2_low",
            "unary_arcsin_numpy_clog_large_double",
            "unary_arcsin_numpy_hard_work_double",
            "acos",
            "atan2",
        ):
            with self.subTest(double_token=token):
                self.assertIn(token, numpy_double)

    def test_arctan_uses_typed_keep_order_and_numpy_complex_engine(
        self,
    ) -> None:
        math_ops = source("src/math_ops.c")
        public_body = function_body(math_ops, "cnp_arctan")
        self.assertIn("unary_arctan_arrays", public_body)
        self.assertNotIn("cnp_unary_op", public_body)
        result_types = function_body(
            math_ops, "unary_trigonometric_result_dtype"
        )
        engine = function_body(math_ops, "unary_arctan_arrays")
        element = function_body(math_ops, "unary_arctan_element")
        for token in (
            "CNP_BOOL", "CNP_HALF", "CNP_CFLOAT", "CNP_CDOUBLE",
            "CNP_CLONGDOUBLE", "unary_set_keep_order_layout",
            "CNP_ARRAY_C_CONTIGUOUS", "CNP_ARRAY_F_CONTIGUOUS",
            "cnp_multi_to_offset", "atanf", "atan", "atanl",
        ):
            with self.subTest(token=token):
                self.assertIn(token, result_types + engine + element)

        cfloat = function_body(math_ops, "unary_arctan_cfloat")
        cdouble = function_body(math_ops, "unary_arctan_cdouble")
        self.assertIn("unary_arctan_numpy_atanh_float", cfloat)
        self.assertIn("unary_arctan_numpy_atanh_double", cdouble)
        numpy_float = function_body(
            math_ops, "unary_arctan_numpy_atanh_float"
        )
        numpy_double = function_body(
            math_ops, "unary_arctan_numpy_atanh_double"
        )
        for token in (
            "reciprocal_epsilon",
            "unary_arctan_numpy_real_reciprocal_float",
            "unary_arctan_numpy_sum_squares_float",
            "log1pf", "atan2f", "copysignf",
        ):
            with self.subTest(float_token=token):
                self.assertIn(token, numpy_float)
        for token in (
            "reciprocal_epsilon",
            "unary_arctan_numpy_real_reciprocal_double",
            "unary_arctan_numpy_sum_squares_double",
            "log1p", "atan2", "copysign",
        ):
            with self.subTest(double_token=token):
                self.assertIn(token, numpy_double)

    def test_arctan2_uses_typed_broadcast_and_contiguous_engines(
        self,
    ) -> None:
        math_ops = source("src/math_ops.c")
        public_body = function_body(math_ops, "cnp_arctan2")
        self.assertIn("arctan2_arrays", public_body)
        self.assertNotIn("cnp_binary_op", public_body)

        result_types = function_body(math_ops, "real_binary_result_dtype")
        validation = function_body(math_ops, "arctan2_validate_inputs")
        prepare = function_body(math_ops, "arctan2_prepare_result")
        engine = function_body(math_ops, "arctan2_arrays")
        contiguous = function_body(
            math_ops, "arctan2_contiguous_typed"
        )
        element = function_body(math_ops, "arctan2_element")
        for token in (
            "CNP_HALF",
            "CNP_FLOAT",
            "CNP_DOUBLE",
            "CNP_LONGDOUBLE",
            "CNP_ERR_TYPE",
            "CNP_ERR_BROADCAST",
            "arithmetic_result_order",
            "arithmetic_broadcast_offset",
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_ARRAY_F_CONTIGUOUS",
            "cnp_multi_to_offset",
            "atan2f",
            "atan2",
            "atan2l",
        ):
            with self.subTest(token=token):
                self.assertIn(
                    token,
                    result_types + validation + prepare + engine +
                    contiguous + element,
                )

    def test_hypot_uses_typed_broadcast_and_contiguous_engines(
        self,
    ) -> None:
        math_ops = source("src/math_ops.c")
        public_body = function_body(math_ops, "cnp_hypot")
        self.assertIn("hypot_arrays", public_body)
        self.assertNotIn("cnp_binary_op", public_body)

        result_types = function_body(math_ops, "real_binary_result_dtype")
        validation = function_body(math_ops, "hypot_validate_inputs")
        prepare = function_body(math_ops, "hypot_prepare_result")
        engine = function_body(math_ops, "hypot_arrays")
        contiguous = function_body(math_ops, "hypot_contiguous_typed")
        element = function_body(math_ops, "hypot_element")
        numpy_float = function_body(math_ops, "hypot_numpy_float")
        numpy_double = function_body(math_ops, "hypot_numpy_double")
        numpy_longdouble = function_body(
            math_ops, "hypot_numpy_longdouble"
        )
        combined = (
            result_types + validation + prepare + engine + contiguous +
            element + numpy_float + numpy_double + numpy_longdouble
        )
        for token in (
            "CNP_HALF",
            "CNP_FLOAT",
            "CNP_DOUBLE",
            "CNP_LONGDOUBLE",
            "CNP_ERR_TYPE",
            "CNP_ERR_BROADCAST",
            "arithmetic_result_order",
            "arithmetic_broadcast_offset",
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_ARRAY_F_CONTIGUOUS",
            "cnp_multi_to_offset",
            "hypot_numpy_float",
            "hypot_numpy_double",
            "hypot_numpy_longdouble",
            "DBL_MIN",
        ):
            with self.subTest(token=token):
                self.assertIn(token, combined)

    def test_angle_conversion_aliases_use_typed_keep_order_engine(
        self,
    ) -> None:
        math_ops = source("src/math_ops.c")
        public_modes = {
            "cnp_degrees": "CNP_ANGLE_TO_DEGREES",
            "cnp_radians": "CNP_ANGLE_TO_RADIANS",
            "cnp_deg2rad": "CNP_ANGLE_TO_RADIANS",
            "cnp_rad2deg": "CNP_ANGLE_TO_DEGREES",
        }
        for symbol, mode in public_modes.items():
            with self.subTest(symbol=symbol):
                public_body = function_body(math_ops, symbol)
                self.assertIn("angle_conversion_arrays", public_body)
                self.assertIn(mode, public_body)
                self.assertNotIn("cnp_unary_op", public_body)

        result_types = function_body(math_ops, "real_binary_result_dtype")
        engine = function_body(math_ops, "angle_conversion_arrays")
        element = function_body(math_ops, "angle_conversion_element")
        float_loop = function_body(math_ops, "angle_conversion_float")
        double_loop = function_body(math_ops, "angle_conversion_double")
        longdouble_loop = function_body(
            math_ops, "angle_conversion_longdouble"
        )
        combined = (
            result_types + engine + element + float_loop + double_loop +
            longdouble_loop
        )
        for token in (
            "CNP_HALF",
            "CNP_FLOAT",
            "CNP_DOUBLE",
            "CNP_LONGDOUBLE",
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_ARRAY_F_CONTIGUOUS",
            "unary_set_keep_order_layout",
            "cnp_multi_to_offset",
            "angle_conversion_float",
            "angle_conversion_double",
            "angle_conversion_longdouble",
        ):
            with self.subTest(token=token):
                self.assertIn(token, combined)
        self.assertIn("57.2957763671875f", float_loop)
        self.assertIn("0.01745329238474369049f", float_loop)
        self.assertIn("180.0 / 3.14159265358979323846", double_loop)
        self.assertIn("3.14159265358979323846 / 180.0", double_loop)

    def test_transcendental_functions_use_shared_typed_keep_order_engine(
        self,
    ) -> None:
        math_ops = source("src/math_ops.c")
        for function_name, operation in (
            ("cnp_sinh", "CNP_HYPERBOLIC_SINH"),
            ("cnp_cosh", "CNP_HYPERBOLIC_COSH"),
            ("cnp_tanh", "CNP_HYPERBOLIC_TANH"),
            ("cnp_arcsinh", "CNP_HYPERBOLIC_ARCSINH"),
            ("cnp_arccosh", "CNP_HYPERBOLIC_ARCCOSH"),
            ("cnp_arctanh", "CNP_HYPERBOLIC_ARCTANH"),
            ("cnp_exp", "CNP_TRANSCENDENTAL_EXP"),
            ("cnp_exp2", "CNP_TRANSCENDENTAL_EXP2"),
            ("cnp_expm1", "CNP_TRANSCENDENTAL_EXPM1"),
            ("cnp_log", "CNP_TRANSCENDENTAL_LOG"),
            ("cnp_log2", "CNP_TRANSCENDENTAL_LOG2"),
            ("cnp_log10", "CNP_TRANSCENDENTAL_LOG10"),
            ("cnp_log1p", "CNP_TRANSCENDENTAL_LOG1P"),
        ):
            with self.subTest(function_name=function_name):
                public_body = function_body(math_ops, function_name)
                self.assertIn("unary_transcendental_arrays", public_body)
                self.assertIn(operation, public_body)
                self.assertNotIn("cnp_unary_op", public_body)

        result_types = function_body(
            math_ops, "unary_transcendental_result_dtype"
        )
        engine = function_body(math_ops, "unary_transcendental_arrays")
        element = function_body(math_ops, "unary_transcendental_element")
        complex_loops = (
            function_body(math_ops, "unary_transcendental_cfloat"),
            function_body(math_ops, "unary_transcendental_cdouble"),
            function_body(math_ops, "unary_transcendental_clongdouble"),
        )
        log_loops = (
            function_body(math_ops, "unary_log_numpy_float"),
            function_body(math_ops, "unary_log_numpy_cfloat"),
            function_body(math_ops, "unary_log_numpy_cdouble"),
            function_body(math_ops, "unary_log_numpy_clongdouble"),
            function_body(math_ops, "unary_log1p_numpy_cfloat"),
            function_body(math_ops, "unary_log1p_numpy_cdouble"),
            function_body(math_ops, "unary_log1p_numpy_clongdouble"),
        )
        combined = (
            result_types + engine + element + "".join(complex_loops) +
            "".join(log_loops)
        )
        for token in (
            "CNP_HALF",
            "CNP_FLOAT",
            "CNP_DOUBLE",
            "CNP_LONGDOUBLE",
            "CNP_CFLOAT",
            "CNP_CDOUBLE",
            "CNP_CLONGDOUBLE",
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_ARRAY_F_CONTIGUOUS",
            "unary_set_keep_order_layout",
            "cnp_multi_to_offset",
            "sinhf",
            "sinh",
            "sinhl",
            "csinhf",
            "csinh",
            "csinhl",
            "coshf",
            "cosh",
            "coshl",
            "ccoshf",
            "ccosh",
            "ccoshl",
            "tanhf",
            "tanh",
            "tanhl",
            "ctanhf",
            "ctanh",
            "ctanhl",
            "asinhf",
            "asinh",
            "asinhl",
            "acoshf",
            "acosh",
            "acoshl",
            "atanhf",
            "atanh",
            "atanhl",
            "cnp_simd_tanh_f32",
            "cnp_simd_tanh_f64",
            "unary_exp_numpy_float",
            "cexpf",
            "cexp",
            "cexpl",
            "exp2f",
            "exp2",
            "exp2l",
            "expm1f",
            "expm1",
            "expm1l",
            "unary_log_numpy_float",
            "unary_log_numpy_cfloat",
            "unary_log_numpy_cdouble",
            "unary_log_numpy_clongdouble",
            "unary_log1p_numpy_cfloat",
            "unary_log1p_numpy_cdouble",
            "unary_log1p_numpy_clongdouble",
            "log1pf",
            "log1p",
            "log1pl",
            "log2f",
            "log2",
            "log2l",
            "1.442695040888963407359924681001892137",
            "log10f",
            "log10",
            "log10l",
            "0.434294481903251827651128918916605082",
            "FLT_MANT_DIG",
            "DBL_MANT_DIG",
            "LDBL_MANT_DIG",
            "hypot_numpy_float",
            "hypot_numpy_double",
            "hypot_numpy_longdouble",
            "source.imag / 2.0f",
            "expm1f(source.real) * cosf(source.imag)",
            "2.0f * sine_half * sine_half",
            "expf(source.real) * sinf(source.imag)",
            "0.693147180559945309417232121458176568f",
            "0.693147180559945309417232121458176568L",
        ):
            with self.subTest(token=token):
                self.assertIn(token, combined)

        bridge_source = source("src/cnumpy_ahk.c")
        wrapper = source("ahk/numpy.ahk")
        for public_name, c_function in (
            ("Sinh", "cnp_sinh"),
            ("Cosh", "cnp_cosh"),
            ("Tanh", "cnp_tanh"),
            ("Arcsinh", "cnp_arcsinh"),
            ("Arccosh", "cnp_arccosh"),
            ("Arctanh", "cnp_arctanh"),
            ("Exp", "cnp_exp"),
            ("Exp2", "cnp_exp2"),
            ("Expm1", "cnp_expm1"),
            ("Log", "cnp_log"),
            ("Log2", "cnp_log2"),
            ("Log10", "cnp_log10"),
            ("Log1p", "cnp_log1p"),
        ):
            bridge_name = f"cnp_ahk_{c_function.removeprefix('cnp_')}"
            bridge = function_body(bridge_source, bridge_name)
            for token in ("cnp_set_error", c_function, "cnp_relabel_error"):
                with self.subTest(bridge=bridge_name, token=token):
                    self.assertIn(token, bridge)
            static_body = function_body(wrapper, public_name)
            self.assertIn(f'Numpy.Proc("{bridge_name}")', static_body)
            self.assertIn("Numpy.WrapHandle", static_body)
            self.assertIn(
                f"{public_name}() => Numpy.{public_name}(this)", wrapper
            )

    def test_exp_double_contiguous_path_is_direct_and_unrolled(self) -> None:
        engine = function_body(
            source("src/math_ops.c"), "unary_transcendental_arrays"
        )
        for token in (
            "operation == CNP_TRANSCENDENTAL_EXP",
            "source_dtype == CNP_DOUBLE",
            "result_dtype == CNP_DOUBLE",
            "index + 3 < source->size",
            "exp(source_values[index])",
            "exp(source_values[index + 3])",
        ):
            with self.subTest(token=token):
                self.assertIn(token, engine)

    def test_exp2_double_contiguous_path_is_direct_and_unrolled(self) -> None:
        engine = function_body(
            source("src/math_ops.c"), "unary_transcendental_arrays"
        )
        for token in (
            "operation == CNP_TRANSCENDENTAL_EXP2",
            "source_dtype == CNP_DOUBLE",
            "result_dtype == CNP_DOUBLE",
            "index + 3 < source->size",
            "exp2(source_values[index])",
            "exp2(source_values[index + 3])",
        ):
            with self.subTest(token=token):
                self.assertIn(token, engine)

    def test_expm1_double_contiguous_path_is_direct_and_unrolled(self) -> None:
        engine = function_body(
            source("src/math_ops.c"), "unary_transcendental_arrays"
        )
        for token in (
            "operation == CNP_TRANSCENDENTAL_EXPM1",
            "source_dtype == CNP_DOUBLE",
            "result_dtype == CNP_DOUBLE",
            "index + 3 < source->size",
            "expm1(source_values[index])",
            "expm1(source_values[index + 3])",
        ):
            with self.subTest(token=token):
                self.assertIn(token, engine)

    def test_log_double_contiguous_path_is_direct_and_unrolled(self) -> None:
        engine = function_body(
            source("src/math_ops.c"), "unary_transcendental_arrays"
        )
        for token in (
            "operation == CNP_TRANSCENDENTAL_LOG",
            "source_dtype == CNP_DOUBLE",
            "result_dtype == CNP_DOUBLE",
            "index + 3 < source->size",
            "log(source_values[index])",
            "log(source_values[index + 3])",
        ):
            with self.subTest(token=token):
                self.assertIn(token, engine)

    def test_log2_double_contiguous_path_is_direct_and_unrolled(self) -> None:
        engine = function_body(
            source("src/math_ops.c"), "unary_transcendental_arrays"
        )
        for token in (
            "operation == CNP_TRANSCENDENTAL_LOG2",
            "source_dtype == CNP_DOUBLE",
            "result_dtype == CNP_DOUBLE",
            "index + 3 < source->size",
            "log2(source_values[index])",
            "log2(source_values[index + 3])",
        ):
            with self.subTest(token=token):
                self.assertIn(token, engine)

    def test_log10_double_contiguous_path_is_direct_and_unrolled(self) -> None:
        engine = function_body(
            source("src/math_ops.c"), "unary_transcendental_arrays"
        )
        for token in (
            "operation == CNP_TRANSCENDENTAL_LOG10",
            "source_dtype == CNP_DOUBLE",
            "result_dtype == CNP_DOUBLE",
            "index + 3 < source->size",
            "log10(source_values[index])",
            "log10(source_values[index + 3])",
        ):
            with self.subTest(token=token):
                self.assertIn(token, engine)

    def test_log1p_double_contiguous_path_is_direct_and_unrolled(self) -> None:
        engine = function_body(
            source("src/math_ops.c"), "unary_transcendental_arrays"
        )
        for token in (
            "operation == CNP_TRANSCENDENTAL_LOG1P",
            "source_dtype == CNP_DOUBLE",
            "result_dtype == CNP_DOUBLE",
            "index + 3 < source->size",
            "log1p(source_values[index])",
            "log1p(source_values[index + 3])",
        ):
            with self.subTest(token=token):
                self.assertIn(token, engine)

    def test_logaddexp_uses_typed_broadcast_and_unrolled_contiguous_engine(
        self,
    ) -> None:
        math_ops = source("src/math_ops.c")
        public_body = function_body(math_ops, "cnp_logaddexp")
        self.assertIn("logaddexp_arrays", public_body)
        self.assertNotIn("cnp_binary_op", public_body)

        result_types = function_body(math_ops, "real_binary_result_dtype")
        validation = function_body(math_ops, "logaddexp_validate_inputs")
        prepare = function_body(math_ops, "logaddexp_prepare_result")
        engine = function_body(math_ops, "logaddexp_arrays")
        contiguous = function_body(
            math_ops, "logaddexp_contiguous_typed"
        )
        element = function_body(math_ops, "logaddexp_element")
        combined = (
            result_types + validation + prepare + engine + contiguous +
            element
        )
        for token in (
            "CNP_HALF",
            "CNP_FLOAT",
            "CNP_DOUBLE",
            "CNP_LONGDOUBLE",
            "CNP_ERR_TYPE",
            "CNP_ERR_BROADCAST",
            "arithmetic_result_order",
            "arithmetic_broadcast_offset",
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_ARRAY_F_CONTIGUOUS",
            "same_shape",
            "index + 3 < result->size",
            "result_values[index] = logaddexp_numpy_double",
            "result_values[index + 3] = logaddexp_numpy_double",
            "logaddexp_contiguous_typed",
        ):
            with self.subTest(token=token):
                self.assertIn(token, combined)

    def test_logaddexp2_uses_typed_broadcast_and_unrolled_contiguous_engine(
        self,
    ) -> None:
        math_ops = source("src/math_ops.c")
        public_body = function_body(math_ops, "cnp_logaddexp2")
        self.assertIn("logaddexp_arrays", public_body)
        self.assertIn("CNP_LOGADDEXP_BASE2", public_body)
        self.assertNotIn("cnp_binary_op", public_body)

        result_types = function_body(math_ops, "real_binary_result_dtype")
        validation = function_body(math_ops, "logaddexp_validate_inputs")
        prepare = function_body(math_ops, "logaddexp_prepare_result")
        engine = function_body(math_ops, "logaddexp_arrays")
        contiguous = function_body(
            math_ops, "logaddexp_contiguous_typed"
        )
        element = function_body(math_ops, "logaddexp_element")
        combined = (
            result_types + validation + prepare + engine + contiguous +
            element
        )
        for token in (
            "CNP_HALF",
            "CNP_FLOAT",
            "CNP_DOUBLE",
            "CNP_LONGDOUBLE",
            "CNP_ERR_TYPE",
            "CNP_ERR_BROADCAST",
            "arithmetic_result_order",
            "arithmetic_broadcast_offset",
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_ARRAY_F_CONTIGUOUS",
            "same_shape",
            "index + 3 < result->size",
            "logaddexp_contiguous_typed",
        ):
            with self.subTest(token=token):
                self.assertIn(token, combined)
        for token in (
            "operation == CNP_LOGADDEXP_BASE2",
            "result_values[index] = logaddexp2_numpy_double",
            "result_values[index + 3] = logaddexp2_numpy_double",
        ):
            with self.subTest(contiguous_token=token):
                self.assertIn(token, contiguous)

    def test_comparison_has_direct_unrolled_contiguous_float64_path(self) -> None:
        math_ops = source("src/math_ops.c")
        engine = function_body(math_ops, "comparison_arrays")
        contiguous = function_body(
            math_ops, "comparison_contiguous_double"
        )
        self.assertIn("comparison_contiguous_double", engine)
        self.assertLess(
            engine.index("comparison_contiguous_double"),
            engine.index("int64_t coordinates"),
        )
        for token in (
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_ARRAY_F_CONTIGUOUS",
            "same_shape",
            "CNP_DOUBLE",
            "switch (operation)",
            "index + 3 < result->size",
            "left_values[index] == right_values[index]",
            "left_values[index + 3] == right_values[index + 3]",
        ):
            with self.subTest(token=token):
                self.assertIn(token, contiguous)
        self.assertLess(
            contiguous.index("switch (operation)"),
            contiguous.index("for ("),
        )

    def test_transcendental_element_rejects_unknown_operations(self) -> None:
        element = function_body(
            source("src/math_ops.c"), "unary_transcendental_element"
        )
        self.assertEqual(
            4,
            element.count(
                "else if (operation == CNP_TRANSCENDENTAL_EXP2)"
            ),
        )
        self.assertEqual(
            4,
            element.count(
                "else if (operation == CNP_TRANSCENDENTAL_EXPM1)"
            ),
        )
        self.assertEqual(
            4,
            element.count(
                "else if (operation == CNP_TRANSCENDENTAL_LOG)"
            ),
        )
        self.assertEqual(
            4,
            element.count(
                "else if (operation == CNP_TRANSCENDENTAL_LOG2)"
            ),
        )
        self.assertEqual(
            4,
            element.count(
                "else if (operation == CNP_TRANSCENDENTAL_LOG10)"
            ),
        )
        self.assertEqual(
            4,
            element.count(
                "else if (operation == CNP_TRANSCENDENTAL_LOG1P)"
            ),
        )
        self.assertEqual(
            4,
            element.count('"invalid transcendental operation %d"'),
        )
        self.assertGreaterEqual(element.count("return CNP_ERR_GENERIC;"), 4)

    def test_arcsinh_uses_numpy_complex_engine_and_preserves_nan_bits(
        self,
    ) -> None:
        math_ops = source("src/math_ops.c")
        self.assertIn(
            "unary_arcsin_numpy_asinh_float",
            function_body(math_ops, "unary_transcendental_cfloat"),
        )
        for function_name in (
            "unary_transcendental_cdouble",
            "unary_transcendental_clongdouble",
        ):
            with self.subTest(function_name=function_name):
                self.assertIn(
                    "unary_arcsin_numpy_asinh_double",
                    function_body(math_ops, function_name),
                )

        element = function_body(math_ops, "unary_transcendental_element")
        for token in (
            "source_dtype == CNP_FLOAT",
            "source_bits & 0x7f800000u",
            "source_bits & 0x007fffffu",
            "source_bits & 0x7c00u",
            "source_bits & 0x03ffu",
            "source_bits & 0x7ff0000000000000ull",
            "source_bits & 0x000fffffffffffffull",
        ):
            with self.subTest(token=token):
                self.assertIn(token, element)

    def test_arccosh_uses_numpy_complex_engine_and_preserves_nan_bits(
        self,
    ) -> None:
        math_ops = source("src/math_ops.c")
        cfloat = function_body(math_ops, "unary_transcendental_cfloat")
        cdouble = function_body(math_ops, "unary_transcendental_cdouble")
        clongdouble = function_body(
            math_ops, "unary_transcendental_clongdouble"
        )
        for body, tokens in (
            (
                cfloat,
                (
                    "CNP_HYPERBOLIC_ARCCOSH",
                    "unary_arccos_numpy_float",
                    "fabsf",
                    "copysignf",
                ),
            ),
            (
                cdouble,
                (
                    "CNP_HYPERBOLIC_ARCCOSH",
                    "unary_arccos_numpy_double",
                    "fabs",
                    "copysign",
                ),
            ),
            (
                clongdouble,
                (
                    "CNP_HYPERBOLIC_ARCCOSH",
                    "unary_transcendental_cdouble",
                    "unary_arccos_clongdouble",
                    "fabsl",
                    "copysignl",
                ),
            ),
        ):
            for token in tokens:
                with self.subTest(token=token):
                    self.assertIn(token, body)

        element = function_body(math_ops, "unary_transcendental_element")
        for token in (
            "operation == CNP_HYPERBOLIC_ARCCOSH",
            "source_bits & 0x7f800000u",
            "source_bits & 0x007fffffu",
            "source_bits & 0x7c00u",
            "source_bits & 0x03ffu",
            "source_bits & 0x7ff0000000000000ull",
            "source_bits & 0x000fffffffffffffull",
        ):
            with self.subTest(token=token):
                self.assertIn(token, element)

    def test_arctanh_uses_numpy_complex_engine_and_preserves_nan_bits(
        self,
    ) -> None:
        math_ops = source("src/math_ops.c")
        for function_name, engine_name in (
            ("unary_transcendental_cfloat", "unary_arctan_numpy_atanh_float"),
            ("unary_transcendental_cdouble", "unary_arctan_numpy_atanh_double"),
            (
                "unary_transcendental_clongdouble",
                "unary_arctan_numpy_atanh_double",
            ),
        ):
            body = function_body(math_ops, function_name)
            with self.subTest(function_name=function_name):
                self.assertIn("CNP_HYPERBOLIC_ARCTANH", body)
                self.assertIn(engine_name, body)

        element = function_body(math_ops, "unary_transcendental_element")
        for token in (
            "operation == CNP_HYPERBOLIC_ARCTANH",
            "source_bits & 0x7f800000u",
            "source_bits & 0x007fffffu",
            "source_bits & 0x7c00u",
            "source_bits & 0x03ffu",
            "source_bits & 0x7ff0000000000000ull",
            "source_bits & 0x000fffffffffffffull",
        ):
            with self.subTest(token=token):
                self.assertIn(token, element)

    def test_tanh_fma3_dispatch_uses_numpy_125_lut_kernels(self) -> None:
        internal = source("include/cnumpy/cnumpy_internal.h")
        dispatch = source("src/simd_dispatch.c")
        avx2 = source("src/simd_avx2.c")
        for suffix in ("f32", "f64"):
            with self.subTest(suffix=suffix):
                self.assertIn(f"cnp_sse2_tanh_{suffix}", internal)
                self.assertIn(f"cnp_avx2_tanh_{suffix}", internal)
                self.assertIn(f"cnp_simd_tanh_{suffix}", internal)
                self.assertIn(
                    f"g_tanh_{suffix}_kernel = cnp_avx2_tanh_{suffix}",
                    function_body(dispatch, "initialize_dispatch"),
                )
                self.assertIn(
                    f"g_tanh_{suffix}_kernel",
                    function_body(dispatch, f"cnp_simd_tanh_{suffix}"),
                )

        self.assertIn("cnp_numpy_125_tanh_f32_lut[32][8]", avx2)
        self.assertIn("cnp_numpy_125_tanh_f64_lut[16][18]", avx2)
        float_block = function_body(avx2, "cnp_avx2_tanh_f32_block")
        double_block = function_body(avx2, "cnp_avx2_tanh_f64_block")
        for token in (
            "_mm256_i32gather_ps",
            "_mm256_fmadd_ps",
            "0x7fc00000",
        ):
            with self.subTest(float_token=token):
                self.assertIn(token, avx2 if "gather" in token else float_block)
        for token in (
            "_mm256_i64gather_pd",
            "_mm256_fmadd_pd",
            "0x7ff8000000000000ull",
        ):
            with self.subTest(double_token=token):
                self.assertIn(token, avx2 if "gather" in token else double_block)

    def test_destination_operations_use_the_same_stable_names(self) -> None:
        into_ops = source("src/into_ops.c")
        self.assertIn("cnp_simd_add", function_body(into_ops, "cnp_add_into"))
        self.assertIn("cnp_simd_sqrt", function_body(into_ops, "cnp_sqrt_into"))


class GemmDispatchContracts(unittest.TestCase):
    def test_gemm_contract_and_kernel_files_exist(self) -> None:
        internal = source("include/cnumpy/cnumpy_internal.h")
        for relative_path in ("src/gemm.c", "src/gemm_sse2.c", "src/gemm_avx2.c"):
            with self.subTest(path=relative_path):
                self.assertTrue((ROOT / relative_path).is_file())
        self.assertRegex(internal, r"typedef\s+void\s*\(\s*\*CnpGemmTileKernel\s*\)")
        self.assertRegex(internal, r"\bcnp_gemm_f64\s*\(")
        self.assertRegex(internal, r"\bcnp_sse2_gemm_tile\s*\(")
        self.assertRegex(internal, r"\bcnp_avx2_gemm_tile\s*\(")
        self.assertRegex(internal, r"\bcnp_simd_gemm_tile\s*\(")

    def test_kernels_vectorize_contiguous_b_rows_without_transpose(self) -> None:
        baseline = source("src/gemm_sse2.c")
        avx2 = source("src/gemm_avx2.c")
        for text, prefix in ((baseline, "_mm_"), (avx2, "_mm256_")):
            with self.subTest(prefix=prefix):
                self.assertIn("i_block", text)
                self.assertIn("k_block", text)
                self.assertIn("j_block", text)
                self.assertIn("b + inner * n", text)
                self.assertIn(prefix + "set1_pd", text)
                self.assertIn(prefix + "mul_pd", text)
                self.assertIn(prefix + "add_pd", text)
                self.assertNotIn("transpose", text.casefold())
                self.assertNotIn("fmadd", text.casefold())

    def test_dispatch_and_linalg_delegate_to_gemm(self) -> None:
        dispatch = source("src/simd_dispatch.c")
        self.assertIn("cnp_sse2_gemm_tile", dispatch)
        self.assertIn("cnp_avx2_gemm_tile", dispatch)
        self.assertIn("cnp_simd_gemm_tile", dispatch)
        body = function_body(source("src/linalg.c"), "cnp_matmul")
        self.assertIn("cnp_gemm_f64", body)
        self.assertIn("CNP_OK", body)
        self.assertIn("cnp_array_free(result)", body)
        self.assertNotIn("Bt", body)

    def test_two_dimensional_dot_delegates_to_verified_matmul_before_generic(
        self,
    ) -> None:
        body = function_body(source("src/linalg.c"), "cnp_dot")
        for token in (
            "a->ndim == 2",
            "b->ndim == 2",
            "cnp_matmul(a, b)",
            "cnp_relabel_error(function_name)",
            "dot_generic(a, b, function_name)",
        ):
            with self.subTest(token=token):
                self.assertIn(token, body)
        if "cnp_matmul(a, b)" in body:
            self.assertLess(
                body.index("cnp_matmul(a, b)"),
                body.index("dot_generic(a, b, function_name)"),
            )

    def test_einsum_fast_dispatch_uses_parsed_axes_not_literal_labels(
        self,
    ) -> None:
        einsum = source("src/einsum.c")
        body = function_body(einsum, "einsum_execute_fast")
        self.assertNotIn("strcmp", body)
        for token in (
            "axis_labels",
            "output_labels",
            "contraction_labels",
            "cnp_matmul",
            "cnp_dot",
            "cnp_outer",
            "cnp_trace_ext",
            "cnp_transpose",
        ):
            with self.subTest(token=token):
                self.assertIn(token, einsum)

    def test_einsum_rejects_subscripts_that_exceed_its_parser_index_type(
        self,
    ) -> None:
        implementation = source("src/einsum.c")
        body = function_body(
            implementation, "einsum_execute_expression"
        )
        for entrypoint in ("cnp_einsum", "cnp_einsum_generic"):
            with self.subTest(entrypoint=entrypoint):
                entrypoint_body = function_body(implementation, entrypoint)
                self.assertIn(
                    "einsum_execute_expression", entrypoint_body
                )
        guard = "if (source_length >= (size_t)INT_MAX)"
        allocation = "cnp_malloc(source_length + 1)"
        self.assertIn(guard, body)
        self.assertIn(allocation, body)
        self.assertLess(body.index(guard), body.index(allocation))
        self.assertIn("CNP_ERR_VALUE", body)
        self.assertIn("subscripts are too long to parse", body)


class ParallelGemmContracts(unittest.TestCase):
    def test_public_and_ahk_thread_control_contracts_exist(self) -> None:
        public = source("include/cnumpy/cnumpy.h")
        ahk_header = source("include/cnumpy/cnumpy_ahk.h")
        ahk_source = source("src/cnumpy_ahk.c")
        wrapper = source("ahk/numpy.ahk")
        for declaration in (
            r"\bcnp_set_num_threads\s*\(\s*int\s+count\s*\)",
            r"\bcnp_get_num_threads\s*\(\s*void\s*\)",
        ):
            with self.subTest(declaration=declaration):
                self.assertRegex(public, declaration)
        for name in ("cnp_ahk_set_num_threads", "cnp_ahk_get_num_threads"):
            with self.subTest(name=name):
                self.assertIn(name, ahk_header)
                self.assertIn(name, ahk_source)
        self.assertRegex(wrapper, r"static\s+SetNumThreads\s*\(")
        self.assertRegex(wrapper, r"static\s+GetNumThreads\s*\(")
        self.assertIn("Numpy.CheckStatus", function_body(wrapper, "SetNumThreads"))

    def test_private_pool_has_explicit_lifecycle_and_threshold(self) -> None:
        internal = source("include/cnumpy/cnumpy_internal.h")
        core = source("src/core.c")
        gemm = source("src/gemm.c")
        self.assertIn(
            "#define CNP_GEMM_PARALLEL_MIN_WORK INT64_C(16777216)", gemm
        )
        for name in ("cnp_gemm_thread_pool_init", "cnp_gemm_thread_pool_cleanup"):
            with self.subTest(name=name):
                self.assertIn(name, internal)
                self.assertIn(name, gemm)
        self.assertIn("cnp_gemm_thread_pool_init", function_body(core, "cnp_init"))
        self.assertIn("cnp_gemm_thread_pool_cleanup", function_body(core, "cnp_cleanup"))
        for win32_name in (
            "CreateThreadpool",
            "CreateThreadpoolCleanupGroup",
            "InitializeThreadpoolEnvironment",
            "SetThreadpoolCallbackPool",
            "SetThreadpoolCallbackCleanupGroup",
            "CloseThreadpoolCleanupGroup",
            "CloseThreadpool",
        ):
            with self.subTest(win32_name=win32_name):
                self.assertIn(win32_name, gemm)

    def test_parallel_scheduler_submits_disjoint_row_work_and_waits(self) -> None:
        gemm = source("src/gemm.c")
        body = function_body(gemm, "cnp_gemm_f64")
        self.assertIn("cnp_gemm_reaches_parallel_threshold", body)
        self.assertIn(
            "CNP_GEMM_PARALLEL_MIN_WORK",
            function_body(gemm, "cnp_gemm_reaches_parallel_threshold"),
        )
        self.assertIn("CNP_GEMM_PARALLEL_ROW_BLOCK", body)
        self.assertIn("CreateThreadpoolWork", body)
        self.assertIn("SubmitThreadpoolWork", body)
        self.assertIn("CloseThreadpoolCleanupGroupMembers", body)
        self.assertIn("row_begin", body)
        self.assertIn("row_end", body)
        self.assertIn("return status", body)

    def test_benchmark_worker_configures_and_records_requested_threads(self) -> None:
        worker = source("benchmark/bench_cnumpy.ahk")
        self.assertIn('EnvGet("CNP_NUM_THREADS")', worker)
        self.assertIn("cnp_ahk_set_num_threads", worker)
        self.assertIn("cnp_ahk_get_num_threads", worker)
        self.assertIn('Chr(34) "num_threads" Chr(34)', worker)


class NumpyFacadeContracts(unittest.TestCase):
    def test_fixed_metadata_abi_is_declared_and_exported(self) -> None:
        header = source("include/cnumpy/cnumpy_ahk.h")
        ahk_source = source("src/cnumpy_ahk.c")
        self.assertIn("#define CNP_AHK_METADATA_ABI_VERSION 1u", header)
        self.assertIn("#define CNP_AHK_METADATA_SIZE 544u", header)
        self.assertRegex(header, r"typedef\s+struct\s*\{[^}]+\}\s*CnpAhkMetadata\s*;")
        for field in (
            "abi_version",
            "struct_size",
            "ndim",
            "dtype",
            "itemsize",
            "flags",
            "size",
            "shape[CNP_MAXDIMS]",
        ):
            with self.subTest(field=field):
                self.assertIn(field, header)
        self.assertIn("cnp_ahk_get_metadata", header)
        body = function_body(ahk_source, "cnp_ahk_get_metadata")
        self.assertIn("CNP_AHK_METADATA_SIZE", body)
        self.assertIn("CNP_AHK_METADATA_ABI_VERSION", body)
        self.assertIn("metadata.shape", body)
        self.assertIn("memcpy(out_metadata, &metadata", body)

    def test_wrapper_loads_and_caches_metadata_once(self) -> None:
        wrapper = source("ahk/numpy.ahk")
        loader = function_body(wrapper, "_LoadMetadata")
        self.assertEqual(1, loader.count("cnp_ahk_get_metadata"))
        self.assertIn("_metadataLoaded", loader)
        self.assertIn("_shape", loader)
        self.assertIn("_flags", loader)
        self.assertIn("this._shape.Clone()", wrapper)
        for legacy_call in (
            "\\cnp_ahk_ndim",
            "\\cnp_ahk_size",
            "\\cnp_ahk_dtype",
            "\\cnp_ahk_itemsize",
            "\\cnp_ahk_shape",
        ):
            with self.subTest(legacy_call=legacy_call):
                self.assertNotIn(legacy_call, wrapper)

    def test_numpy_style_static_surface_and_many_concatenate_exist(self) -> None:
        header = source("include/cnumpy/cnumpy_ahk.h")
        ahk_source = source("src/cnumpy_ahk.c")
        wrapper = source("ahk/numpy.ahk")
        methods = (
            "Add",
            "Subtract",
            "Multiply",
            "Divide",
            "Sqrt",
            "Absolute",
            "Sum",
            "Mean",
            "Var",
            "Std",
            "Max",
            "Min",
            "Prod",
            "Argmax",
            "Cumsum",
            "Dot",
            "Matmul",
            "Reshape",
            "Transpose",
            "Flatten",
            "Concatenate",
            "AsContiguousArray",
        )
        for method in methods:
            with self.subTest(method=method):
                self.assertRegex(wrapper, rf"static\s+{method}\s*\(")
        for text in (header, ahk_source):
            self.assertIn("cnp_ahk_concatenate_many", text)
        self.assertIn("cnp_ahk_concatenate_many", function_body(wrapper, "Concatenate"))
        add_body = function_body(wrapper, "Add")
        self.assertIn("IsSet(out)", add_body)
        self.assertIn("cnp_ahk_add_into", add_body)
        self.assertIn("DllCall", add_body)
        for method, into_name in (
            ("Sqrt", "SqrtInto"),
            ("Cumsum", "CumsumInto"),
        ):
            body = function_body(wrapper, method)
            self.assertIn("IsSet(out)", body)
            self.assertIn(into_name, body)

    def test_static_add_benchmark_uses_preallocated_destination(self) -> None:
        ahk_worker = source("benchmark/bench_cnumpy.ahk")
        numpy_worker = source("benchmark/bench_numpy.py")
        invocation = function_body(ahk_worker, "InvokeOperation")
        static_case = re.search(
            r'case\s+"static_add_call"\s*:\s*(.*?)(?=\n\s*case\s+|\n\s*default\s*:)',
            invocation,
            re.DOTALL,
        )
        self.assertIsNotNone(static_case)
        assert static_case is not None
        self.assertIn("Numpy.Add", static_case.group(1))
        self.assertIn("WrapperInputs[3]", static_case.group(1))
        self.assertRegex(
            numpy_worker,
            r"static_add_call[\s\S]+np\.add\([^\n]+out=destination",
        )

    def test_allclose_benchmark_executes_the_real_scalar_apis(self) -> None:
        ahk_worker = source("benchmark/bench_cnumpy.ahk")
        numpy_worker = source("benchmark/bench_numpy.py")
        invocation = function_body(ahk_worker, "InvokeOperation")
        allclose_case = re.search(
            r'case\s+"allclose"\s*:\s*(.*?)(?=\n\s*case\s+|\n\s*default\s*:)',
            invocation,
            re.DOTALL,
        )
        self.assertIsNotNone(allclose_case)
        assert allclose_case is not None
        self.assertIn("cnp_ahk_allclose", allclose_case.group(1))
        self.assertRegex(
            numpy_worker,
            r'operation\s*==\s*"allclose"[\s\S]+np\.allclose\(',
        )

    def test_angle_benchmark_executes_the_real_array_apis(self) -> None:
        ahk_worker = source("benchmark/bench_cnumpy.ahk")
        numpy_worker = source("benchmark/bench_numpy.py")
        invocation = function_body(ahk_worker, "InvokeOperation")
        angle_case = re.search(
            r'case\s+"angle"\s*:\s*(.*?)(?=\n\s*case\s+|\n\s*default\s*:)',
            invocation,
            re.DOTALL,
        )
        self.assertIsNotNone(angle_case)
        assert angle_case is not None
        self.assertIn("cnp_ahk_angle", angle_case.group(1))
        self.assertRegex(
            numpy_worker,
            r'"angle"\s*:\s*np\.angle[\s\S]+operation\s*==\s*"angle"',
        )

    def test_log2_benchmark_executes_the_real_array_apis(self) -> None:
        ahk_worker = source("benchmark/bench_cnumpy.ahk")
        numpy_worker = source("benchmark/bench_numpy.py")
        operation_contract = function_body(ahk_worker, "OperationContract")
        setup = function_body(ahk_worker, "Setup")
        invocation = function_body(ahk_worker, "InvokeOperation")
        for body, context in (
            (operation_contract, "operation contract"),
            (setup, "setup"),
            (invocation, "invocation"),
        ):
            with self.subTest(context=context):
                self.assertIn('"log2"', body)
        self.assertIn('Numpy.Proc("cnp_ahk_" operation)', invocation)
        self.assertIn('"log2": np.log2', numpy_worker)

    def test_log10_benchmark_executes_the_real_array_apis(self) -> None:
        ahk_worker = source("benchmark/bench_cnumpy.ahk")
        numpy_worker = source("benchmark/bench_numpy.py")
        operation_contract = function_body(ahk_worker, "OperationContract")
        setup = function_body(ahk_worker, "Setup")
        invocation = function_body(ahk_worker, "InvokeOperation")
        for body, context in (
            (operation_contract, "operation contract"),
            (setup, "setup"),
            (invocation, "invocation"),
        ):
            with self.subTest(context=context):
                self.assertIn('"log10"', body)
        self.assertIn('Numpy.Proc("cnp_ahk_" operation)', invocation)
        self.assertIn('"log10": np.log10', numpy_worker)

    def test_log1p_benchmark_executes_the_real_array_apis(self) -> None:
        ahk_worker = source("benchmark/bench_cnumpy.ahk")
        numpy_worker = source("benchmark/bench_numpy.py")
        operation_contract = function_body(ahk_worker, "OperationContract")
        setup = function_body(ahk_worker, "Setup")
        invocation = function_body(ahk_worker, "InvokeOperation")
        for body, context in (
            (operation_contract, "operation contract"),
            (setup, "setup"),
            (invocation, "invocation"),
        ):
            with self.subTest(context=context):
                self.assertIn('"log1p"', body)
        self.assertIn('Numpy.Proc("cnp_ahk_" operation)', invocation)
        self.assertIn('"log1p": np.log1p', numpy_worker)

    def test_logaddexp_benchmark_executes_the_real_binary_array_api(
        self,
    ) -> None:
        ahk_worker = source("benchmark/bench_cnumpy.ahk")
        numpy_worker = source("benchmark/bench_numpy.py")
        operation_contract = function_body(ahk_worker, "OperationContract")
        setup = function_body(ahk_worker, "Setup")
        invocation = function_body(ahk_worker, "InvokeOperation")
        for body, context in (
            (operation_contract, "operation contract"),
            (setup, "setup"),
            (invocation, "invocation"),
        ):
            with self.subTest(context=context):
                self.assertIn('"logaddexp"', body)
        self.assertIn('Numpy.Proc("cnp_ahk_" operation)', invocation)
        self.assertIn('"logaddexp": np.logaddexp', numpy_worker)

    def test_logaddexp2_benchmark_executes_the_real_binary_array_api(
        self,
    ) -> None:
        ahk_worker = source("benchmark/bench_cnumpy.ahk")
        numpy_worker = source("benchmark/bench_numpy.py")
        operation_contract = function_body(ahk_worker, "OperationContract")
        setup = function_body(ahk_worker, "Setup")
        invocation = function_body(ahk_worker, "InvokeOperation")
        for body, context in (
            (operation_contract, "operation contract"),
            (setup, "setup"),
            (invocation, "invocation"),
        ):
            with self.subTest(context=context):
                self.assertIn('"logaddexp2"', body)
        self.assertIn('Numpy.Proc("cnp_ahk_" operation)', invocation)
        self.assertIn('"logaddexp2": np.logaddexp2', numpy_worker)

    def test_equal_benchmark_executes_the_real_boolean_array_api(self) -> None:
        ahk_worker = source("benchmark/bench_cnumpy.ahk")
        numpy_worker = source("benchmark/bench_numpy.py")
        operation_contract = function_body(ahk_worker, "OperationContract")
        setup = function_body(ahk_worker, "Setup")
        invocation = function_body(ahk_worker, "InvokeOperation")
        for body, context in (
            (operation_contract, "operation contract"),
            (setup, "setup"),
            (invocation, "invocation"),
        ):
            with self.subTest(context=context):
                self.assertIn('"equal"', body)
        self.assertIn('Numpy.Proc("cnp_ahk_" operation)', invocation)
        self.assertIn('"equal": np.equal', numpy_worker)

    def test_logical_benchmarks_execute_the_real_boolean_array_apis(
        self,
    ) -> None:
        ahk_worker = source("benchmark/bench_cnumpy.ahk")
        numpy_worker = source("benchmark/bench_numpy.py")
        operation_contract = function_body(ahk_worker, "OperationContract")
        setup = function_body(ahk_worker, "Setup")
        invocation = function_body(ahk_worker, "InvokeOperation")
        for operation in (
            "logical_and", "logical_or", "logical_xor", "logical_not",
        ):
            with self.subTest(operation=operation):
                self.assertIn(f'"{operation}"', operation_contract)
                self.assertIn(f'"{operation}"', setup)
                self.assertIn(f'"{operation}"', invocation)
                self.assertIn(f'"{operation}": np.{operation}', numpy_worker)
        self.assertIn('Numpy.Proc("cnp_ahk_" operation)', invocation)

    def test_array_predicates_use_direct_contiguous_float64_path(self) -> None:
        math_ops = source("src/math_ops.c")
        arrays_body = function_body(math_ops, "predicate_arrays")
        fast_body = function_body(
            math_ops, "predicate_contiguous_double"
        )
        bits_body = function_body(math_ops, "predicate_double_bits")

        self.assertIn("predicate_contiguous_double", arrays_body)
        self.assertLess(
            arrays_body.index("predicate_contiguous_double"),
            arrays_body.index("int64_t coords"),
        )
        for token in (
            "CNP_DOUBLE",
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_ARRAY_F_CONTIGUOUS",
            "index + 3",
            "predicate_double_bits",
        ):
            with self.subTest(token=token):
                self.assertIn(token, fast_body)
        self.assertIn("memcpy", bits_body)
        self.assertIn("UINT64_C(0x7ff0000000000000)", fast_body)
        self.assertIn("UINT64_C(0x000fffffffffffff)", fast_body)

    def test_bitwise_benchmarks_execute_the_real_int64_array_apis(self) -> None:
        ahk_worker = source("benchmark/bench_cnumpy.ahk")
        numpy_worker = source("benchmark/bench_numpy.py")
        operation_contract = function_body(ahk_worker, "OperationContract")
        setup = function_body(ahk_worker, "Setup")
        invocation = function_body(ahk_worker, "InvokeOperation")
        integer_input = function_body(ahk_worker, "CreateIntVectorHandle")
        for operation in (
            "bitwise_and", "bitwise_or", "bitwise_xor", "invert",
            "left_shift", "right_shift",
        ):
            with self.subTest(operation=operation):
                self.assertIn(f'"{operation}"', operation_contract)
                self.assertIn(f'"{operation}"', setup)
                self.assertIn(f'"{operation}"', invocation)
                self.assertIn(f'"{operation}": np.{operation}', numpy_worker)
        self.assertIn("cnp_ahk_from_ints", integer_input)
        self.assertIn('Numpy.Proc("cnp_ahk_" operation)', invocation)

    def test_power_facade_and_benchmarks_use_real_array_apis(self) -> None:
        header = source("include/cnumpy/cnumpy_ahk.h")
        ahk_source = source("src/cnumpy_ahk.c")
        wrapper = source("ahk/numpy.ahk")
        ahk_worker = source("benchmark/bench_cnumpy.ahk")
        numpy_worker = source("benchmark/bench_numpy.py")

        self.assertIn("cnp_ahk_float_power", header)
        self.assertIn("cnp_ahk_float_power", ahk_source)
        for method, native in (
            ("Power", "cnp_ahk_power"),
            ("FloatPower", "cnp_ahk_float_power"),
        ):
            with self.subTest(method=method):
                static_body = function_body(wrapper, method)
                self.assertIn(native, static_body)
                self.assertIn("_RequireSetPair", static_body)
                self.assertIn(f"{method}(other) => Numpy.{method}", wrapper)

        operation_contract = function_body(ahk_worker, "OperationContract")
        setup = function_body(ahk_worker, "Setup")
        invocation = function_body(ahk_worker, "InvokeOperation")
        for operation in ("power", "float_power"):
            with self.subTest(operation=operation):
                self.assertIn(f'"{operation}"', operation_contract)
                self.assertIn(f'"{operation}"', setup)
                self.assertIn(f'"{operation}"', invocation)
                self.assertIn(
                    f'"{operation}": np.{operation}', numpy_worker
                )
        self.assertIn('Numpy.Proc("cnp_ahk_" operation)', invocation)

    def test_heaviside_facade_and_benchmarks_use_real_array_api(self) -> None:
        header = source("include/cnumpy/cnumpy_ahk.h")
        ahk_source = source("src/cnumpy_ahk.c")
        wrapper = source("ahk/numpy.ahk")
        ahk_worker = source("benchmark/bench_cnumpy.ahk")
        numpy_worker = source("benchmark/bench_numpy.py")

        self.assertIn("cnp_ahk_heaviside", header)
        self.assertIn("cnp_ahk_heaviside", ahk_source)
        static_body = function_body(wrapper, "Heaviside")
        self.assertIn("cnp_ahk_heaviside", static_body)
        self.assertIn("_RequireSetPair", static_body)
        self.assertIn("Heaviside(other) => Numpy.Heaviside", wrapper)

        operation_contract = function_body(ahk_worker, "OperationContract")
        setup = function_body(ahk_worker, "Setup")
        invocation = function_body(ahk_worker, "InvokeOperation")
        self.assertIn('"heaviside"', operation_contract)
        self.assertIn('"heaviside"', setup)
        self.assertIn('"heaviside"', invocation)
        self.assertIn('"heaviside": np.heaviside', numpy_worker)
        self.assertIn('Numpy.Proc("cnp_ahk_" operation)', invocation)

    def test_heaviside_uses_stride_aware_typed_math_engine(self) -> None:
        math_ops = source("src/math_ops.c")
        math_extra = source("src/math_extra.c")

        self.assertNotIn("cnp_heaviside(", math_extra)
        public_body = function_body(math_ops, "cnp_heaviside")
        arrays_body = function_body(math_ops, "heaviside_arrays")
        contiguous_body = function_body(math_ops, "heaviside_contiguous_typed")
        self.assertIn("heaviside_arrays", public_body)
        self.assertIn("arithmetic_prepare_result", arrays_body)
        self.assertIn("arithmetic_broadcast_offset", arrays_body)
        self.assertNotIn("%", arrays_body)
        self.assertLess(
            contiguous_body.index("result->size == 0"),
            contiguous_body.index("left->data"),
        )
        self.assertIn("index + 3", contiguous_body)
        self.assertIn("isnan", contiguous_body)

    def test_gcd_lcm_facade_and_benchmarks_use_integer_array_apis(self) -> None:
        header = source("include/cnumpy/cnumpy_ahk.h")
        ahk_source = source("src/cnumpy_ahk.c")
        wrapper = source("ahk/numpy.ahk")
        ahk_worker = source("benchmark/bench_cnumpy.ahk")
        numpy_worker = source("benchmark/bench_numpy.py")

        for method, operation in (("Gcd", "gcd"), ("Lcm", "lcm")):
            native = f"cnp_ahk_{operation}"
            with self.subTest(operation=operation):
                self.assertIn(native, header)
                self.assertIn(native, ahk_source)
                static_body = function_body(wrapper, method)
                self.assertIn(native, static_body)
                self.assertIn("_RequireSetPair", static_body)
                self.assertIn(f"{method}(other) => Numpy.{method}", wrapper)

        operation_contract = function_body(ahk_worker, "OperationContract")
        setup = function_body(ahk_worker, "Setup")
        invocation = function_body(ahk_worker, "InvokeOperation")
        for operation in ("gcd", "lcm"):
            with self.subTest(benchmark_operation=operation):
                self.assertIn(f'"{operation}"', operation_contract)
                self.assertIn(f'"{operation}"', setup)
                self.assertIn(f'"{operation}"', invocation)
                self.assertIn(f'"{operation}": np.{operation}', numpy_worker)
        self.assertIn("CreateIntVectorHandle", setup)
        self.assertIn('Numpy.Proc("cnp_ahk_" operation)', invocation)

    def test_gcd_lcm_use_unsigned_stride_aware_integer_engine(self) -> None:
        math_ops = source("src/math_ops.c")
        math_extra = source("src/math_extra.c")

        self.assertNotIn("cnp_gcd(", math_extra)
        self.assertNotIn("cnp_lcm(", math_extra)
        for operation in ("gcd", "lcm"):
            public_body = function_body(math_ops, f"cnp_{operation}")
            self.assertIn("integer_gcd_lcm_arrays", public_body)
        arrays_body = function_body(math_ops, "integer_gcd_lcm_arrays")
        contiguous_body = function_body(
            math_ops, "integer_gcd_lcm_contiguous"
        )
        apply_body = function_body(math_ops, "integer_gcd_lcm_bits")
        magnitude_body = function_body(math_ops, "integer_magnitude_bits")
        self.assertIn("bitwise_element_pointer", arrays_body)
        self.assertIn("bitwise_write_bits", arrays_body)
        self.assertNotIn("cnp_array_flat_get", arrays_body)
        self.assertLess(
            contiguous_body.index("result->size == 0"),
            contiguous_body.index("left->data"),
        )
        self.assertIn("uint64_t", apply_body)
        self.assertNotRegex(apply_body, r"\bint64_t\b")
        self.assertNotIn("bitwise_mask", apply_body)
        self.assertNotIn("cnp_dtype_itemsize", magnitude_body)
        self.assertIn("bitwise_mask", contiguous_body)
        self.assertIn("bitwise_mask", arrays_body)

    def test_divmod_facade_and_benchmarks_preserve_both_native_results(
        self,
    ) -> None:
        header = source("include/cnumpy/cnumpy_ahk.h")
        ahk_source = source("src/cnumpy_ahk.c")
        wrapper = source("ahk/numpy.ahk")
        ahk_worker = source("benchmark/bench_cnumpy.ahk")
        numpy_worker = source("benchmark/bench_numpy.py")

        self.assertIn("cnp_ahk_divmod", header)
        bridge_body = function_body(ahk_source, "cnp_ahk_divmod")
        self.assertIn("cnp_divmod", bridge_body)
        self.assertIn("result_capacity", bridge_body)
        static_body = function_body(wrapper, "Divmod")
        self.assertIn("cnp_ahk_divmod", static_body)
        self.assertIn("WrapHandleBuffer", static_body)
        self.assertIn("Divmod(other) => Numpy.Divmod", wrapper)

        operation_contract = function_body(ahk_worker, "OperationContract")
        setup = function_body(ahk_worker, "Setup")
        invocation = function_body(ahk_worker, "InvokeOperation")
        timed = function_body(ahk_worker, "TimedInvoke")
        validation = function_body(ahk_worker, "ValidationSignature")
        self.assertIn('"divmod"', operation_contract)
        self.assertIn('"divmod"', setup)
        self.assertIn("cnp_ahk_divmod", invocation)
        self.assertIn("ReturnsHandlePair", timed)
        self.assertIn("HandlePairValidationSignature", validation)
        self.assertIn("np.divmod", numpy_worker)
        self.assertIn("np.stack", numpy_worker)

        math_ops = source("src/math_ops.c")
        numpy_ext = source("src/numpy_ext.c")
        self.assertNotIn("cnp_divmod(", numpy_ext)
        public_body = function_body(math_ops, "cnp_divmod")
        numeric_body = function_body(math_ops, "divmod_numeric_arrays")
        contiguous_body = function_body(
            math_ops, "divmod_contiguous_typed"
        )
        element_body = function_body(
            math_ops, "arithmetic_promoted_divmod_element"
        )
        self.assertIn("divmod_numeric_arrays", public_body)
        self.assertNotIn("cnp_floor_divide", public_body)
        self.assertNotIn("cnp_remainder", public_body)
        self.assertIn("arithmetic_prepare_result", numeric_body)
        self.assertIn("arithmetic_broadcast_offset", numeric_body)
        self.assertIn("divmod_contiguous_typed", numeric_body)
        self.assertIn("same_shape", contiguous_body)
        self.assertIn("arithmetic_divmod_double", contiguous_body)
        self.assertEqual(1, element_body.count("arithmetic_divmod_float("))
        self.assertEqual(1, element_body.count("arithmetic_divmod_double("))
        self.assertEqual(
            1, element_body.count("arithmetic_divmod_longdouble(")
        )

    def test_fft_benchmark_projects_validation_without_replacing_timed_fft(
        self,
    ) -> None:
        ahk_worker = source("benchmark/bench_cnumpy.ahk")
        invocation = function_body(ahk_worker, "InvokeOperation")
        timed = function_body(ahk_worker, "TimedInvoke")
        validation = function_body(ahk_worker, "ValidationSignature")
        projection_start = ahk_worker.index("\nFftValidationSignature(")
        projection = function_body(
            ahk_worker[projection_start:], "FftValidationSignature"
        )

        self.assertIn('case "fft"', invocation)
        self.assertIn("cnp_ahk_fft", invocation)
        self.assertIn("this.RawInvoke.Call()", timed)
        self.assertNotIn("cnp_real", timed)
        self.assertNotIn("cnp_imag", timed)
        self.assertIn("FftValidationSignature", validation)
        self.assertIn("cnp_real", projection)
        self.assertIn("cnp_imag", projection)
        self.assertIn("HandleValidationSignature", projection)

    def test_real_imag_benchmarks_execute_the_component_array_apis(self) -> None:
        ahk_worker = source("benchmark/bench_cnumpy.ahk")
        numpy_worker = source("benchmark/bench_numpy.py")
        invocation = function_body(ahk_worker, "InvokeOperation")
        component_case = re.search(
            r'case\s+"real",\s*"imag"\s*:\s*'
            r'(.*?)(?=\n\s*case\s+|\n\s*default\s*:)',
            invocation,
            re.DOTALL,
        )
        self.assertIsNotNone(component_case)
        assert component_case is not None
        self.assertIn('"cnp_ahk_" operation', component_case.group(1))
        self.assertIn('"real": np.real', numpy_worker)
        self.assertIn('"imag": np.imag', numpy_worker)

    def test_real_if_close_benchmark_scans_close_complex128_inputs(self) -> None:
        ahk_worker = source("benchmark/bench_cnumpy.ahk")
        numpy_worker = source("benchmark/bench_numpy.py")
        setup = function_body(ahk_worker, "Setup")
        invocation = function_body(ahk_worker, "InvokeOperation")
        self.assertRegex(
            setup,
            r'case\s+"real_if_close"[\s\S]+cnp_ahk_create[\s\S]+'
            r'\b16\b',
        )
        self.assertRegex(
            invocation,
            r'case\s+"real_if_close"[\s\S]+cnp_ahk_real_if_close',
        )
        self.assertRegex(
            numpy_worker,
            r'operation\s*==\s*"real_if_close"[\s\S]+'
            r'np\.zeros\([^\n]+dtype=np\.complex128\)[\s\S]+'
            r'np\.real_if_close',
        )

    def test_signal_benchmarks_execute_real_same_mode_array_apis(self) -> None:
        ahk_worker = source("benchmark/bench_cnumpy.ahk")
        numpy_worker = source("benchmark/bench_numpy.py")
        setup = function_body(ahk_worker, "Setup")
        invocation = function_body(ahk_worker, "InvokeOperation")
        self.assertRegex(
            setup,
            r'case\s+"convolve",\s*"correlate"[\s\S]+'
            r'CreateVectorHandle\([\s\S]+Min\(size, 8\)',
        )
        signal_case = re.search(
            r'case\s+"convolve",\s*"correlate"\s*:\s*'
            r'(.*?)(?=\n\s*case\s+|\n\s*default\s*:)',
            invocation,
            re.DOTALL,
        )
        self.assertIsNotNone(signal_case)
        assert signal_case is not None
        self.assertIn('"cnp_ahk_" operation', signal_case.group(1))
        self.assertRegex(signal_case.group(1), r'"Int",\s*1')
        self.assertRegex(
            numpy_worker,
            r'operation\s+in\s+\{"convolve",\s*"correlate"\}'
            r'[\s\S]+np\.convolve[\s\S]+np\.correlate'
            r'[\s\S]+mode="same"',
        )

    def test_signal_products_have_a_direct_contiguous_float64_dot_path(
        self,
    ) -> None:
        implementation = source("src/extra.c")
        public_body = function_body(implementation, "signal_dot")
        fast_body = function_body(
            implementation, "signal_dot_contiguous_float64"
        )
        self.assertIn("signal_dot_contiguous_float64", public_body)
        self.assertLess(
            public_body.index("signal_dot_contiguous_float64"),
            public_body.index("SIGNAL_READ_PAIR"),
        )
        for token in (
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_DOUBLE",
            "const double *",
            "reverse_kernel",
            "count == 8",
        ):
            with self.subTest(token=token):
                self.assertIn(token, fast_body)

    def test_signal_products_reuse_eight_float64_kernel_values_across_outputs(
        self,
    ) -> None:
        implementation = source("src/extra.c")
        product_body = function_body(implementation, "signal_product")
        central_body = function_body(
            implementation, "signal_central_contiguous_float64"
        )
        self.assertIn("signal_central_contiguous_float64", product_body)
        for token in (
            "kernel_length != 8",
            "kernel_values[0 * kernel_step]",
            "kernel_values[7 * kernel_step]",
            "for (int64_t index = 0; index < output_count; ++index)",
            "reverse_output",
        ):
            with self.subTest(token=token):
                self.assertIn(token, central_body)

    def test_angle_has_a_direct_real_float64_path(self) -> None:
        implementation = source("src/extra.c")
        public_body = function_body(implementation, "cnp_angle")
        fast_body = function_body(
            implementation, "angle_contiguous_real_float64"
        )
        self.assertIn("angle_contiguous_real_float64", public_body)
        self.assertLess(
            public_body.index("angle_contiguous_real_float64"),
            public_body.index("coordinates"),
        )
        for token in (
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_DOUBLE",
            "uint64_t",
            "0x0008000000000000",
        ):
            self.assertIn(token, fast_body)

    def test_allclose_has_a_direct_contiguous_float64_path(self) -> None:
        implementation = source("src/array_ops.c")
        public_body = function_body(implementation, "cnp_allclose_v2")
        fast_body = function_body(
            implementation, "allclose_contiguous_f64"
        )
        self.assertIn("allclose_contiguous_f64", public_body)
        self.assertLess(
            public_body.index("allclose_contiguous_f64"),
            public_body.index("coordinates"),
        )
        for token in (
            "CNP_ARRAY_C_CONTIGUOUS",
            "CNP_DOUBLE",
            "const double *",
        ):
            with self.subTest(token=token):
                self.assertIn(token, fast_body)
        self.assertNotIn("cnp_cast_scalar_value", fast_body)


class Avx2DispatchContracts(unittest.TestCase):
    def test_dot_has_dispatched_kernels(self) -> None:
        internal = source("include/cnumpy/cnumpy_internal.h")
        baseline = source("src/simd_ops.c")
        avx2 = source("src/simd_avx2.c")
        dispatch = source("src/simd_dispatch.c")
        self.assertRegex(internal, r"\bcnp_sse2_dot\s*\(")
        self.assertRegex(internal, r"\bcnp_avx2_dot\s*\(")
        self.assertRegex(internal, r"\bcnp_simd_dot\s*\(")
        self.assertRegex(baseline, r"\bcnp_sse2_dot\s*\(")
        self.assertRegex(avx2, r"\bcnp_avx2_dot\s*\(")
        self.assertRegex(dispatch, r"\bcnp_simd_dot\s*\(")

    def test_norm_dispatch_and_variance_routes_to_shared_semantics(self) -> None:
        implementation = source("src/reduce.c")
        norm = function_body(source("src/linalg.c"), "cnp_linalg_norm")
        legacy = function_body(implementation, "cnp_var")
        modern = function_body(implementation, "cnp_var_v2")
        shared = function_body(implementation, "reduction_variance")
        for token in ("CNP_ARRAY_C_CONTIGUOUS", "CNP_DOUBLE", "cnp_simd_dot"):
            with self.subTest(function="cnp_linalg_norm", token=token):
                self.assertIn(token, norm)
        if "cnp_simd_dot" in norm:
            self.assertLess(norm.index("cnp_simd_dot"), norm.index("coords"))
        self.assertIn("cnp_var_v2", legacy)
        self.assertNotIn("cnp_simd_sum_squared_deviation", legacy)
        self.assertIn("reduction_variance", modern)
        self.assertIn("reduction_pairwise_sum_float", shared)
        self.assertIn("reduction_pairwise_sum_double", shared)
        self.assertNotIn("cnp_simd_sum_squared_deviation", shared)

    def test_baseline_dispatch_checks_cpu_os_and_avx2_once(self) -> None:
        dispatch_path = ROOT / "src/simd_dispatch.c"
        self.assertTrue(dispatch_path.is_file(), "src/simd_dispatch.c must exist")
        dispatch = dispatch_path.read_text(encoding="utf-8")
        for token in (
            "InitOnceExecuteOnce",
            "__cpuid",
            "__cpuidex",
            "_xgetbv",
            "CNP_SIMD_LEVEL_SSE2",
            "CNP_SIMD_LEVEL_AVX2",
        ):
            with self.subTest(token=token):
                self.assertIn(token, dispatch)
        self.assertIn("CNP_ERR_GENERIC", dispatch)
        self.assertIn("cnp_set_error", dispatch)
        self.assertNotIn("cnp_malloc", dispatch)

    def test_avx2_kernels_are_isolated_and_vectorized(self) -> None:
        avx2_path = ROOT / "src/simd_avx2.c"
        self.assertTrue(avx2_path.is_file(), "src/simd_avx2.c must exist")
        avx2 = avx2_path.read_text(encoding="utf-8")
        for symbol in ("cnp_avx2_maximum", "cnp_avx2_dot"):
            with self.subTest(symbol=symbol):
                self.assertRegex(avx2, rf"\b{symbol}\s*\(")
        for intrinsic in (
            "_mm256_loadu_pd",
            "_mm256_max_pd",
            "_mm256_mul_pd",
            "_mm256_cmp_pd",
        ):
            with self.subTest(intrinsic=intrinsic):
                self.assertIn(intrinsic, avx2)
        self.assertIn("__declspec(noinline)", avx2)
        self.assertNotIn("cnp_malloc", avx2)

    def test_runtime_level_is_exposed_to_ahk(self) -> None:
        internal = source("include/cnumpy/cnumpy_internal.h")
        ahk_header = source("include/cnumpy/cnumpy_ahk.h")
        bridge = source("src/cnumpy_ahk.c")
        self.assertRegex(internal, r"#define\s+CNP_SIMD_LEVEL_SSE2\s+1")
        self.assertRegex(internal, r"#define\s+CNP_SIMD_LEVEL_AVX2\s+2")
        self.assertRegex(ahk_header, r"\bcnp_ahk_simd_level\s*\(")
        self.assertRegex(bridge, r"\bcnp_ahk_simd_level\s*\(")


if __name__ == "__main__":
    unittest.main()
