from __future__ import annotations

from contextlib import ExitStack
import ctypes
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from compat.cnumpy_ctypes import CnumpyError, CnumpyRuntime
from compat.oracle import assert_array_equivalent


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"
SPLIT_V2_EXPORTS = (
    "cnp_split_sections_v2",
    "cnp_split_indices_v2",
    "cnp_array_split_sections_v2",
    "cnp_array_split_indices_v2",
)


def close_all(arrays) -> None:
    for array in reversed(tuple(arrays)):
        array.close()


class AtleastNdSemanticsTests(unittest.TestCase):
    def test_scalar_empty_and_ranked_arrays_match_numpy_metadata(self) -> None:
        values = (
            np.array(7.5, dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.arange(4, dtype=np.int32),
            np.empty((2, 0), dtype=np.float64),
            np.arange(6, dtype=np.float64).reshape(2, 3),
            (
                np.arange(24, dtype=np.float64).reshape(2, 3, 4)
                + 1j
            ).astype(np.complex128),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for value in values:
                for minimum_ndim in (1, 2, 3):
                    with self.subTest(
                        shape=value.shape,
                        dtype=str(value.dtype),
                        minimum_ndim=minimum_ndim,
                    ), ExitStack() as stack:
                        source = stack.enter_context(runtime.from_numpy(value))
                        result = stack.enter_context(
                            runtime.atleast(source, minimum_ndim)
                        )
                        expected = getattr(
                            np, f"atleast_{minimum_ndim}d"
                        )(value)
                        assert_array_equivalent(
                            self, result, expected, compare_strides=True
                        )
                        if value.ndim >= minimum_ndim:
                            self.assertEqual(
                                source.pointer.value, result.pointer.value
                            )
                            self.assertTrue(result.owns_data)
                        else:
                            self.assertNotEqual(
                                source.pointer.value, result.pointer.value
                            )
                            self.assertFalse(result.owns_data)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_inserted_axes_preserve_fortran_view_strides(self) -> None:
        base_value = np.arange(6, dtype=np.float64).reshape(2, 3)
        source_value = base_value.T
        with CnumpyRuntime(DLL) as runtime:
            base = runtime.from_numpy(base_value)
            source = runtime.transpose(base, (1, 0))
            result = runtime.atleast(source, 3)
            try:
                assert_array_equivalent(
                    self,
                    result,
                    np.atleast_3d(source_value),
                    compare_strides=True,
                )
                self.assertFalse(result.c_contiguous)
                self.assertTrue(result.f_contiguous)
                self.assertFalse(result.owns_data)
            finally:
                result.close()
                source.close()
                base.close()

    def test_expanded_views_outlive_released_sources(self) -> None:
        cases = (
            (np.array(2.5, dtype=np.float64), 3),
            (np.arange(4, dtype=np.int64), 2),
            (np.arange(6, dtype=np.float32).reshape(2, 3), 3),
        )
        with CnumpyRuntime(DLL) as runtime:
            for value, minimum_ndim in cases:
                with self.subTest(
                    shape=value.shape, minimum_ndim=minimum_ndim
                ):
                    source = runtime.from_numpy(value)
                    result = runtime.atleast(source, minimum_ndim)
                    try:
                        source.close()
                        assert_array_equivalent(
                            self,
                            result,
                            getattr(
                                np, f"atleast_{minimum_ndim}d"
                            )(value),
                            compare_strides=True,
                        )
                    finally:
                        result.close()

    def test_repeated_results_release_their_reference_or_view(self) -> None:
        with CnumpyRuntime(DLL) as runtime, runtime.from_numpy(
            np.arange(8, dtype=np.float64)
        ) as source:
            active = runtime.retained_bytes
            for minimum_ndim in (1, 2, 3):
                for _ in range(64):
                    runtime.atleast(source, minimum_ndim).close()
                self.assertEqual(active, runtime.retained_bytes)

    def test_null_inputs_surface_function_specific_native_errors(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for minimum_ndim in (1, 2, 3):
                function_name = f"cnp_atleast_{minimum_ndim}d"
                function = getattr(runtime.dll, function_name)
                runtime.dll.cnp_clear_error()
                self.assertFalse(function(None))
                error = runtime.error_state()
                self.assertEqual(-1, error.status)
                self.assertEqual(function_name, error.function)
                self.assertEqual("source array is required", error.message)
                self.assertEqual(baseline, runtime.retained_bytes)

    def test_malformed_metadata_fails_explicitly_in_isolated_process(
        self,
    ) -> None:
        code = f"""
import ctypes

class ErrorState(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int),
        ("message", ctypes.c_char * 256),
        ("function", ctypes.c_char * 64),
    ]

class CnpArray(ctypes.Structure):
    _fields_ = [
        ("ndim", ctypes.c_int),
        ("shape", ctypes.c_void_p),
        ("strides", ctypes.c_void_p),
        ("size", ctypes.c_int64),
        ("data", ctypes.c_void_p),
        ("dtype", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("refcount", ctypes.c_int),
        ("base", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("owner", ctypes.c_void_p),
        ("owner_release", ctypes.c_void_p),
    ]

def error_tuple(dll):
    state = ErrorState()
    dll.cnp_get_error(ctypes.byref(state))
    return (
        state.status,
        bytes(state.function).split(b"\\0", 1)[0].decode("ascii"),
        bytes(state.message).split(b"\\0", 1)[0].decode("utf-8"),
    )

dll = ctypes.CDLL({str(DLL)!r})
dll.cnp_init.argtypes = []
dll.cnp_init.restype = ctypes.c_int
dll.cnp_cleanup.argtypes = []
dll.cnp_cleanup.restype = None
dll.cnp_clear_error.argtypes = []
dll.cnp_clear_error.restype = None
dll.cnp_get_error.argtypes = [ctypes.POINTER(ErrorState)]
dll.cnp_get_error.restype = ctypes.c_int
dll.cnp_get_allocated_memory.argtypes = []
dll.cnp_get_allocated_memory.restype = ctypes.c_size_t
dll.cnp_array_from_data.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64), ctypes.c_int, ctypes.c_int,
]
dll.cnp_array_from_data.restype = ctypes.c_void_p
dll.cnp_array_decref.argtypes = [ctypes.c_void_p]
dll.cnp_array_decref.restype = None

if dll.cnp_init() != 0:
    raise SystemExit(10)
baseline = dll.cnp_get_allocated_memory()
shape = (ctypes.c_int64 * 1)(4)
data = (ctypes.c_double * 4)(1.0, 2.0, 3.0, 4.0)
pointer = dll.cnp_array_from_data(data, 1, shape, 13, 0)
if not pointer:
    raise SystemExit(11)
array = ctypes.cast(pointer, ctypes.POINTER(CnpArray)).contents
try:
    active = dll.cnp_get_allocated_memory()
    cases = (
        ("dtype", None, -3, "source array must have a dtype"),
        ("data", None, -1, "source array must have a data buffer"),
        ("shape", None, -4, "source array requires shape and strides"),
        ("strides", None, -4, "source array requires shape and strides"),
        ("ndim", -1, -4, "source array ndim must be between 0 and 64"),
    )
    for minimum_ndim in (1, 2, 3):
        name = "cnp_atleast_" + str(minimum_ndim) + "d"
        function = getattr(dll, name)
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_void_p
        for field, invalid, status, message in cases:
            original = getattr(array, field)
            setattr(array, field, invalid)
            try:
                dll.cnp_clear_error()
                result = function(pointer)
            finally:
                setattr(array, field, original)
            if result:
                dll.cnp_array_decref(result)
                raise AssertionError((name, field, "returned a result"))
            expected = (status, name, message)
            actual = error_tuple(dll)
            if actual != expected:
                raise AssertionError((name, field, actual, expected))
            if dll.cnp_get_allocated_memory() != active:
                raise AssertionError((name, field, "retained bytes changed"))
finally:
    dll.cnp_array_decref(pointer)
    if dll.cnp_get_allocated_memory() != baseline:
        raise AssertionError("source array leaked")
    dll.cnp_cleanup()
"""
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            0,
            completed.returncode,
            msg=(completed.stdout + completed.stderr),
        )


class BroadcastArraysSemanticsTests(unittest.TestCase):
    def test_insufficient_capacity_clears_every_available_result_slot(self) -> None:
        inputs = (
            np.ones((2, 1), dtype=np.float64),
            np.ones((1, 3), dtype=np.float64),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            sources = [
                stack.enter_context(runtime.from_numpy(value)) for value in inputs
            ]
            handles = (ctypes.c_void_p * 2)(
                *(source.pointer.value for source in sources)
            )
            results = (ctypes.c_void_p * 1)(sources[0].pointer.value)
            runtime.dll.cnp_clear_error()
            status = runtime.dll.cnp_broadcast_arrays_v2(
                2, handles, results, 1
            )
            self.assertNotEqual(0, status)
            self.assertFalse(results[0])

    def test_broadcasts_mixed_dtypes_as_stride_views(self) -> None:
        inputs = (
            np.array([[1.5], [2.5]], dtype=np.float64),
            np.array([10, 20, 30], dtype=np.int64),
        )
        expected = np.broadcast_arrays(*inputs)
        with CnumpyRuntime(DLL) as runtime:
            sources = [runtime.from_numpy(value) for value in inputs]
            outputs = runtime.broadcast_arrays(sources)
            try:
                for actual, oracle in zip(outputs, expected):
                    assert_array_equivalent(
                        self, actual, oracle, compare_strides=True
                    )
                    self.assertFalse(actual.owns_data)

                close_all(sources)
                sources = []
                for actual, oracle in zip(outputs, expected):
                    assert_array_equivalent(
                        self, actual, oracle, compare_strides=True
                    )
            finally:
                close_all(outputs)
                close_all(sources)

    def test_handles_scalar_three_inputs_and_zero_dimensions(self) -> None:
        cases = (
            (
                np.array(7.0, dtype=np.float64),
                np.empty((2, 0, 3), dtype=np.float64),
                np.ones((1, 1, 3), dtype=np.float64),
            ),
            (
                np.ones((4, 1, 1), dtype=np.int32),
                np.ones((1, 5, 1), dtype=np.int32),
                np.ones((1, 1, 6), dtype=np.int32),
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            for inputs in cases:
                with self.subTest(shapes=[value.shape for value in inputs]):
                    with ExitStack() as stack:
                        sources = [
                            stack.enter_context(runtime.from_numpy(value))
                            for value in inputs
                        ]
                        outputs = runtime.broadcast_arrays(sources)
                        for output in outputs:
                            stack.enter_context(output)
                        for actual, oracle in zip(
                            outputs, np.broadcast_arrays(*inputs)
                        ):
                            assert_array_equivalent(
                                self, actual, oracle, compare_strides=True
                            )

    def test_zero_arguments_returns_an_empty_result(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            self.assertEqual((), runtime.broadcast_arrays([]))

    def test_incompatible_shapes_raise_without_partial_results(self) -> None:
        inputs = (
            np.ones((2, 3), dtype=np.float64),
            np.ones((4,), dtype=np.float64),
        )
        with CnumpyRuntime(DLL) as runtime:
            with ExitStack() as stack:
                sources = [
                    stack.enter_context(runtime.from_numpy(value))
                    for value in inputs
                ]
                before = runtime.retained_bytes
                with self.assertRaises(CnumpyError) as raised:
                    runtime.broadcast_arrays(sources)
                self.assertEqual(before, runtime.retained_bytes)
                self.assertEqual(-7, raised.exception.status)
                self.assertEqual(
                    "cnp_broadcast_arrays_v2", raised.exception.function
                )
                self.assertIn("axis", raised.exception.message)


class MeshgridSemanticsTests(unittest.TestCase):
    def test_insufficient_capacity_clears_every_available_result_slot(self) -> None:
        inputs = (
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0, 5.0]),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            sources = [
                stack.enter_context(runtime.from_numpy(value)) for value in inputs
            ]
            handles = (ctypes.c_void_p * 2)(
                *(source.pointer.value for source in sources)
            )
            results = (ctypes.c_void_p * 1)(sources[0].pointer.value)
            runtime.dll.cnp_clear_error()
            status = runtime.dll.cnp_meshgrid_v2(
                2, handles, False, False, True, results, 1
            )
            self.assertNotEqual(0, status)
            self.assertFalse(results[0])

    def test_dense_xy_copy_preserves_each_dtype(self) -> None:
        inputs = (
            np.array([1.5, 2.5], dtype=np.float64),
            np.array([10, 20, 30], dtype=np.int64),
        )
        expected = np.meshgrid(*inputs, indexing="xy", sparse=False, copy=True)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            sources = [
                stack.enter_context(runtime.from_numpy(value)) for value in inputs
            ]
            outputs = runtime.meshgrid(
                sources, indexing="xy", sparse=False, copy=True
            )
            for output in outputs:
                stack.enter_context(output)
            for actual, oracle in zip(outputs, expected):
                assert_array_equivalent(
                    self, actual, oracle, compare_strides=True
                )
                self.assertTrue(actual.owns_data)

    def test_empty_inputs_preserve_numpy_shapes_strides_and_ownership(self) -> None:
        inputs = (
            np.array([], dtype=np.float64),
            np.array([1, 2], dtype=np.int32),
        )
        with CnumpyRuntime(DLL) as runtime:
            for sparse in (False, True):
                for copy in (False, True):
                    with self.subTest(sparse=sparse, copy=copy), ExitStack() as stack:
                        sources = [
                            stack.enter_context(runtime.from_numpy(value))
                            for value in inputs
                        ]
                        outputs = runtime.meshgrid(
                            sources,
                            indexing="ij",
                            sparse=sparse,
                            copy=copy,
                        )
                        for output in outputs:
                            stack.enter_context(output)
                        expected = np.meshgrid(
                            *inputs,
                            indexing="ij",
                            sparse=sparse,
                            copy=copy,
                        )
                        for actual, oracle in zip(outputs, expected):
                            assert_array_equivalent(
                                self,
                                actual,
                                oracle,
                                compare_strides=True,
                            )
                            self.assertEqual(copy, actual.owns_data)

    def test_sparse_and_dense_three_input_views_match_numpy(self) -> None:
        inputs = (
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([3, 4, 5], dtype=np.int32),
            np.array([False, True], dtype=np.bool_),
        )
        with CnumpyRuntime(DLL) as runtime:
            for indexing in ("xy", "ij"):
                for sparse in (False, True):
                    with self.subTest(indexing=indexing, sparse=sparse):
                        sources = [runtime.from_numpy(value) for value in inputs]
                        outputs = runtime.meshgrid(
                            sources,
                            indexing=indexing,
                            sparse=sparse,
                            copy=False,
                        )
                        try:
                            expected = np.meshgrid(
                                *inputs,
                                indexing=indexing,
                                sparse=sparse,
                                copy=False,
                            )
                            for actual, oracle in zip(outputs, expected):
                                assert_array_equivalent(
                                    self,
                                    actual,
                                    oracle,
                                    compare_strides=True,
                                )
                                self.assertFalse(actual.owns_data)
                            close_all(sources)
                            sources = []
                            for actual, oracle in zip(outputs, expected):
                                assert_array_equivalent(
                                    self,
                                    actual,
                                    oracle,
                                    compare_strides=True,
                                )
                        finally:
                            close_all(outputs)
                            close_all(sources)

    def test_flattens_non_1d_inputs_like_numpy(self) -> None:
        inputs = (
            np.array([[1, 2], [3, 4]], dtype=np.int32),
            np.array([5, 6], dtype=np.int32),
        )
        expected = np.meshgrid(*inputs, indexing="xy", copy=False)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            sources = [
                stack.enter_context(runtime.from_numpy(value)) for value in inputs
            ]
            outputs = runtime.meshgrid(sources, indexing="xy", copy=False)
            for output in outputs:
                stack.enter_context(output)
            for actual, oracle in zip(outputs, expected):
                assert_array_equivalent(
                    self, actual, oracle, compare_strides=True
                )

    def test_zero_arguments_returns_an_empty_result(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            self.assertEqual((), runtime.meshgrid([]))

    def test_rejects_invalid_indexing_before_native_execution(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            with self.assertRaisesRegex(ValueError, "indexing"):
                runtime.meshgrid([], indexing="invalid")


class SplitFamilySemanticsTests(unittest.TestCase):
    def require_v2_exports(self, runtime: CnumpyRuntime) -> None:
        missing = [
            symbol for symbol in SPLIT_V2_EXPORTS
            if not hasattr(runtime.dll, symbol)
        ]
        self.assertEqual([], missing, f"missing split v2 exports: {missing}")

    def assert_split_equivalent(self, actual, expected) -> None:
        self.assertEqual(len(expected), len(actual))
        for output, oracle in zip(actual, expected):
            assert_array_equivalent(
                self, output, oracle, compare_strides=True
            )
            self.assertFalse(output.owns_data)

    def test_equal_sections_support_positive_and_negative_axes(self) -> None:
        source_value = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            self.require_v2_exports(runtime)
            source = stack.enter_context(runtime.from_numpy(source_value))
            for axis, sections in ((0, 2), (1, 3), (-1, 2)):
                with self.subTest(axis=axis, sections=sections):
                    outputs = runtime.split(source, sections, axis)
                    for output in outputs:
                        stack.enter_context(output)
                    self.assert_split_equivalent(
                        outputs, np.split(source_value, sections, axis=axis)
                    )

    def test_array_split_distributes_remainder_to_leading_pieces(self) -> None:
        source_value = np.arange(30, dtype=np.float64).reshape(2, 3, 5)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            self.require_v2_exports(runtime)
            source = stack.enter_context(runtime.from_numpy(source_value))
            for axis, sections in ((0, 3), (1, 2), (-1, 3)):
                with self.subTest(axis=axis, sections=sections):
                    outputs = runtime.array_split(source, sections, axis)
                    for output in outputs:
                        stack.enter_context(output)
                    self.assert_split_equivalent(
                        outputs,
                        np.array_split(source_value, sections, axis=axis),
                    )

    def test_explicit_indices_preserve_order_clamping_and_empty_pieces(self) -> None:
        source_value = np.arange(12, dtype=np.int32).reshape(2, 6)
        index_cases = ([4, 2], [2, 2], [-99, 99], [], [-2, 3])
        with CnumpyRuntime(DLL) as runtime:
            self.require_v2_exports(runtime)
            for operation_name in ("split", "array_split"):
                operation = getattr(runtime, operation_name)
                oracle = getattr(np, operation_name)
                for indices in index_cases:
                    with self.subTest(
                        operation=operation_name, indices=indices
                    ), ExitStack() as stack:
                        source = stack.enter_context(
                            runtime.from_numpy(source_value)
                        )
                        outputs = operation(source, indices, axis=1)
                        for output in outputs:
                            stack.enter_context(output)
                        self.assert_split_equivalent(
                            outputs,
                            oracle(source_value, indices, axis=1),
                        )

    def test_noncontiguous_views_outlive_their_released_source(self) -> None:
        base_value = np.arange(24, dtype=np.int64).reshape(4, 6)
        source_value = base_value.T
        expected = np.split(source_value, [1, 4], axis=0)
        with CnumpyRuntime(DLL) as runtime:
            self.require_v2_exports(runtime)
            base = runtime.from_numpy(base_value)
            source = runtime.transpose(base, (1, 0))
            outputs = runtime.split(source, [1, 4], axis=0)
            try:
                self.assert_split_equivalent(outputs, expected)
                source.close()
                base.close()
                source = None
                base = None
                self.assert_split_equivalent(outputs, expected)
            finally:
                close_all(outputs)
                if source is not None:
                    source.close()
                if base is not None:
                    base.close()

    def test_split_rejects_uneven_sections_and_invalid_axes_without_leaks(self) -> None:
        source_value = np.arange(10, dtype=np.float64)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            self.require_v2_exports(runtime)
            source = stack.enter_context(runtime.from_numpy(source_value))
            before = runtime.retained_bytes
            for sections, axis, status, message in (
                (3, 0, -4, "equal division"),
                (2, 1, -5, "axis"),
            ):
                with self.subTest(sections=sections, axis=axis):
                    with self.assertRaises(CnumpyError) as raised:
                        runtime.split(source, sections, axis)
                    self.assertEqual(status, raised.exception.status)
                    self.assertEqual(
                        "cnp_split_sections_v2",
                        raised.exception.function,
                    )
                    self.assertIn(message, raised.exception.message.lower())
                    self.assertEqual(before, runtime.retained_bytes)

    def test_insufficient_capacity_clears_every_available_result_slot(self) -> None:
        source_value = np.arange(8, dtype=np.float64)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            self.require_v2_exports(runtime)
            source = stack.enter_context(runtime.from_numpy(source_value))
            results = (ctypes.c_void_p * 1)(source.pointer.value)
            function = runtime._split_function("cnp_split_sections_v2")
            runtime.dll.cnp_clear_error()
            status = function(source.pointer, 2, 0, results, 1)
            self.assertNotEqual(0, status)
            self.assertFalse(results[0])

    def test_empty_axes_and_scalar_axis_errors_match_numpy_categories(self) -> None:
        empty_value = np.empty((2, 0, 3), dtype=np.float64)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            self.require_v2_exports(runtime)
            empty = stack.enter_context(runtime.from_numpy(empty_value))
            for operation_name in ("split", "array_split"):
                operation = getattr(runtime, operation_name)
                oracle = getattr(np, operation_name)
                outputs = operation(empty, 3, axis=1)
                for output in outputs:
                    stack.enter_context(output)
                self.assert_split_equivalent(
                    outputs, oracle(empty_value, 3, axis=1)
                )

            scalar = stack.enter_context(
                runtime.from_numpy(np.array(5, dtype=np.int64))
            )
            with self.assertRaises(CnumpyError) as raised:
                runtime.split(scalar, 1, axis=0)
            self.assertEqual(-5, raised.exception.status)
            self.assertIn("dimension 0", raised.exception.message)

    def test_legacy_split_exports_delegate_to_the_common_view_engine(self) -> None:
        matrix_value = np.arange(24, dtype=np.int64).reshape(4, 6)
        cube_value = np.arange(48, dtype=np.float64).reshape(2, 4, 6)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            matrix = stack.enter_context(runtime.from_numpy(matrix_value))
            cube = stack.enter_context(runtime.from_numpy(cube_value))
            cases = (
                ("cnp_split", matrix, 3, 1,
                 np.split(matrix_value, 3, axis=1)),
                ("cnp_split", matrix, [4, 2], 1,
                 np.split(matrix_value, [4, 2], axis=1)),
                ("cnp_array_split", matrix, 4, 1,
                 np.array_split(matrix_value, 4, axis=1)),
                ("cnp_hsplit", matrix, [4, 2], 0,
                 np.hsplit(matrix_value, [4, 2])),
                ("cnp_vsplit", matrix, 2, 0,
                 np.vsplit(matrix_value, 2)),
                ("cnp_dsplit", cube, [4, 2], 0,
                 np.dsplit(cube_value, [4, 2])),
            )
            for function_name, source, split_spec, axis, expected in cases:
                with self.subTest(function=function_name):
                    outputs = runtime.legacy_split(
                        function_name, source, split_spec, axis
                    )
                    for output in outputs:
                        stack.enter_context(output)
                    self.assert_split_equivalent(outputs, expected)

    def test_legacy_hvdsplit_enforce_numpy_rank_constraints(self) -> None:
        values = (
            ("cnp_hsplit", np.array(1, dtype=np.int64), -4,
             "1 or more dimensions"),
            ("cnp_vsplit", np.arange(4, dtype=np.int64), -4,
             "2 or more dimensions"),
            ("cnp_dsplit", np.arange(8, dtype=np.int64).reshape(2, 4), -4,
             "3 or more dimensions"),
        )
        with CnumpyRuntime(DLL) as runtime:
            for function_name, value, status, message in values:
                with self.subTest(function=function_name), ExitStack() as stack:
                    source = stack.enter_context(runtime.from_numpy(value))
                    before = runtime.retained_bytes
                    with self.assertRaises(CnumpyError) as raised:
                        runtime.legacy_split(function_name, source, 1)
                    self.assertEqual(status, raised.exception.status)
                    self.assertEqual(function_name, raised.exception.function)
                    self.assertIn(message, raised.exception.message)
                    self.assertEqual(before, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
