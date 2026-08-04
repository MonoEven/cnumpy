from __future__ import annotations

from contextlib import ExitStack
import ctypes
from pathlib import Path
import unittest

import numpy as np

from compat.cnumpy_ctypes import CnumpyError, CnumpyRuntime
from compat.oracle import assert_array_equivalent


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"
CNP_AXIS_NONE = -1
CNP_SORT_QUICKSORT = 0
CNP_SORT_MERGESORT = 1
CNP_SORT_HEAPSORT = 2
CNP_SORT_STABLE = 3


class _CnpSlice(ctypes.Structure):
    _fields_ = [
        ("start", ctypes.c_int64),
        ("stop", ctypes.c_int64),
        ("step", ctypes.c_int64),
        ("has_start", ctypes.c_bool),
        ("has_stop", ctypes.c_bool),
        ("has_step", ctypes.c_bool),
    ]


class SortSetSemanticsTests(unittest.TestCase):
    _SET_DTYPES = (
        (np.dtype(np.bool_), 1),
        (np.dtype(np.int8), 2),
        (np.dtype(np.uint8), 3),
        (np.dtype(np.int16), 4),
        (np.dtype(np.uint16), 5),
        (np.dtype(np.int32), 6),
        (np.dtype(np.uint32), 7),
        (np.dtype(np.int64), 10),
        (np.dtype(np.uint64), 11),
        (np.dtype(np.float16), 24),
        (np.dtype(np.float32), 12),
        (np.dtype(np.float64), 13),
    )

    def sort_result(
        self,
        runtime: CnumpyRuntime,
        source,
        function_name: str,
        axis: int | None,
        kind: int,
    ):
        v2_name = f"{function_name}_v2"
        function = getattr(runtime.dll, v2_name)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_bool,
            ctypes.c_int,
        ]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        pointer = function(
            source.pointer,
            0 if axis is None else axis,
            axis is None,
            kind,
        )
        return runtime._owned_result(pointer, v2_name)

    def legacy_sort_result(
        self,
        runtime: CnumpyRuntime,
        source,
        function_name: str,
        axis: int,
        kind: int,
    ):
        function = getattr(runtime.dll, function_name)
        function.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        pointer = function(source.pointer, axis, kind)
        return runtime._owned_result(pointer, function_name)

    def unique_results(
        self,
        runtime: CnumpyRuntime,
        source,
        return_index: bool,
        return_inverse: bool,
        return_counts: bool,
    ):
        result_count = 1 + sum(
            (return_index, return_inverse, return_counts)
        )
        outputs = (ctypes.c_void_p * result_count)()
        function = runtime.dll.cnp_unique_v2
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_bool,
            ctypes.c_bool,
            ctypes.c_bool,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
        ]
        function.restype = ctypes.c_int
        runtime.dll.cnp_clear_error()
        status = function(
            source.pointer,
            return_index,
            return_inverse,
            return_counts,
            outputs,
            result_count,
        )
        if status != 0:
            raise runtime.native_error("cnp_unique_v2", status)
        return tuple(
            runtime._owned_result(outputs[index], "cnp_unique_v2")
            for index in range(result_count)
        )

    def setop_result(
        self,
        runtime: CnumpyRuntime,
        function_name: str,
        left,
        right,
        *boolean_arguments: bool,
    ):
        function = getattr(runtime.dll, function_name)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            *([ctypes.c_bool] * len(boolean_arguments)),
        ]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        pointer = function(
            left.pointer, right.pointer, *boolean_arguments
        )
        return runtime._owned_result(pointer, function_name)

    def astype_result(
        self,
        runtime: CnumpyRuntime,
        source,
        destination_type: int,
    ):
        function = runtime.dll.cnp_astype
        function.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        pointer = function(source.pointer, destination_type, 4)
        return runtime._owned_result(pointer, "cnp_astype")

    def test_stable_float64_sort_and_argsort_match_numpy_total_order(
        self,
    ) -> None:
        source_value = np.array(
            [
                np.nan,
                -0.0,
                0.0,
                np.nan,
                -np.inf,
                np.inf,
                -1.0,
                1.0,
                -0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            sorted_result = stack.enter_context(
                self.sort_result(
                    runtime,
                    source,
                    "cnp_sort",
                    None,
                    CNP_SORT_STABLE,
                )
            )
            argsorted_result = stack.enter_context(
                self.sort_result(
                    runtime,
                    source,
                    "cnp_argsort",
                    None,
                    CNP_SORT_STABLE,
                )
            )
            assert_array_equivalent(
                self,
                sorted_result,
                np.sort(source_value, axis=None, kind="stable"),
            )
            assert_array_equivalent(
                self,
                argsorted_result,
                np.argsort(source_value, axis=None, kind="stable"),
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_large_stable_float64_radix_path_groups_nan_and_equal_zeros(
        self,
    ) -> None:
        negative_nan = np.array(
            [0xFFF8000000000001], dtype=np.uint64
        ).view(np.float64)[0]
        positive_nan = np.array(
            [0x7FF8000000000002], dtype=np.uint64
        ).view(np.float64)[0]
        source_value = np.resize(
            np.array(
                [
                    negative_nan,
                    -0.0,
                    3.0,
                    0.0,
                    positive_nan,
                    -2.0,
                    -0.0,
                    0.0,
                    np.inf,
                    -np.inf,
                ],
                dtype=np.float64,
            ),
            1024,
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            sorted_result = stack.enter_context(
                self.sort_result(
                    runtime,
                    source,
                    "cnp_sort",
                    None,
                    CNP_SORT_STABLE,
                )
            )
            argsorted_result = stack.enter_context(
                self.sort_result(
                    runtime,
                    source,
                    "cnp_argsort",
                    None,
                    CNP_SORT_STABLE,
                )
            )
            assert_array_equivalent(
                self,
                sorted_result,
                np.sort(source_value, axis=None, kind="stable"),
            )
            assert_array_equivalent(
                self,
                argsorted_result,
                np.argsort(source_value, axis=None, kind="stable"),
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_stable_int64_sort_preserves_values_beyond_float_precision(
        self,
    ) -> None:
        source_value = np.array(
            [
                2**53 + 1,
                np.iinfo(np.int64).min,
                2**53,
                0,
                np.iinfo(np.int64).max,
                2**53 + 1,
                -7,
            ],
            dtype=np.int64,
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            sorted_result = stack.enter_context(
                self.sort_result(
                    runtime,
                    source,
                    "cnp_sort",
                    None,
                    CNP_SORT_STABLE,
                )
            )
            argsorted_result = stack.enter_context(
                self.sort_result(
                    runtime,
                    source,
                    "cnp_argsort",
                    None,
                    CNP_SORT_STABLE,
                )
            )
            assert_array_equivalent(
                self,
                sorted_result,
                np.sort(source_value, axis=None, kind="stable"),
            )
            assert_array_equivalent(
                self,
                argsorted_result,
                np.argsort(source_value, axis=None, kind="stable"),
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_stable_axis_sort_handles_noncontiguous_float64_slices(self) -> None:
        storage_value = np.array(
            [
                [np.nan, -0.0, 4.0, 1.0],
                [3.0, 0.0, np.nan, -1.0],
                [2.0, -0.0, 0.0, -np.inf],
            ],
            dtype=np.float64,
        )
        logical_value = storage_value.transpose(1, 0)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            storage = stack.enter_context(runtime.from_numpy(storage_value))
            source = stack.enter_context(runtime.transpose(storage, (1, 0)))
            for axis in (0, 1, -1):
                with self.subTest(axis=axis):
                    sorted_result = stack.enter_context(
                        self.sort_result(
                            runtime,
                            source,
                            "cnp_sort",
                            axis,
                            CNP_SORT_STABLE,
                        )
                    )
                    argsorted_result = stack.enter_context(
                        self.sort_result(
                            runtime,
                            source,
                            "cnp_argsort",
                            axis,
                            CNP_SORT_STABLE,
                        )
                    )
                    assert_array_equivalent(
                        self,
                        sorted_result,
                        np.sort(logical_value, axis=axis, kind="stable"),
                    )
                    assert_array_equivalent(
                        self,
                        argsorted_result,
                        np.argsort(logical_value, axis=axis, kind="stable"),
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_all_kinds_preserve_int64_values_on_every_axis(self) -> None:
        storage_value = np.array(
            [
                [2**53 + 7, -11, np.iinfo(np.int64).max, 4],
                [2**53 + 1, 19, np.iinfo(np.int64).min, -6],
                [2**53 + 5, 3, 0, -2],
            ],
            dtype=np.int64,
        )
        logical_value = storage_value.transpose(1, 0)
        kinds = {
            CNP_SORT_QUICKSORT: "quicksort",
            CNP_SORT_MERGESORT: "mergesort",
            CNP_SORT_HEAPSORT: "heapsort",
            CNP_SORT_STABLE: "stable",
        }
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            storage = stack.enter_context(runtime.from_numpy(storage_value))
            source = stack.enter_context(runtime.transpose(storage, (1, 0)))
            for kind, kind_name in kinds.items():
                for axis in (None, 0, 1, -1):
                    with self.subTest(kind=kind_name, axis=axis):
                        sorted_result = stack.enter_context(
                            self.sort_result(
                                runtime, source, "cnp_sort", axis, kind
                            )
                        )
                        argsorted_result = stack.enter_context(
                            self.sort_result(
                                runtime, source, "cnp_argsort", axis, kind
                            )
                        )
                        assert_array_equivalent(
                            self,
                            sorted_result,
                            np.sort(logical_value, axis=axis, kind=kind_name),
                        )
                        assert_array_equivalent(
                            self,
                            argsorted_result,
                            np.argsort(
                                logical_value, axis=axis, kind=kind_name
                            ),
                        )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_all_kinds_match_float32_duplicate_and_nan_order(self) -> None:
        pattern = np.array(
            [
                2.0,
                -0.0,
                np.nan,
                1.0,
                0.0,
                -3.0,
                2.0,
                np.nan,
                -0.0,
                1.0,
            ],
            dtype=np.float32,
        )
        storage_value = np.stack(
            [np.resize(pattern, 20), np.resize(np.roll(pattern, 3), 20)],
            axis=1,
        )
        logical_value = storage_value.transpose(1, 0)
        kinds = {
            CNP_SORT_QUICKSORT: "quicksort",
            CNP_SORT_MERGESORT: "mergesort",
            CNP_SORT_HEAPSORT: "heapsort",
            CNP_SORT_STABLE: "stable",
        }
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            storage = stack.enter_context(runtime.from_numpy(storage_value))
            source = stack.enter_context(runtime.transpose(storage, (1, 0)))
            for kind, kind_name in kinds.items():
                for axis in (None, 1, -1):
                    with self.subTest(kind=kind_name, axis=axis):
                        sorted_result = stack.enter_context(
                            self.sort_result(
                                runtime, source, "cnp_sort", axis, kind
                            )
                        )
                        argsorted_result = stack.enter_context(
                            self.sort_result(
                                runtime, source, "cnp_argsort", axis, kind
                            )
                        )
                        assert_array_equivalent(
                            self,
                            sorted_result,
                            np.sort(logical_value, axis=axis, kind=kind_name),
                        )
                        assert_array_equivalent(
                            self,
                            argsorted_result,
                            np.argsort(
                                logical_value, axis=axis, kind=kind_name
                            ),
                        )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_sort_v2_rank_zero_through_four_all_axes_and_numeric_dtypes(
        self,
    ) -> None:
        shapes = ((), (5,), (2, 3), (2, 2, 3), (2, 2, 2, 3))
        dtypes = (
            np.bool_,
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
            np.float16,
            np.float32,
            np.float64,
        )
        kinds = {
            CNP_SORT_QUICKSORT: "quicksort",
            CNP_SORT_MERGESORT: "mergesort",
            CNP_SORT_HEAPSORT: "heapsort",
            CNP_SORT_STABLE: "stable",
        }
        pattern = np.array(
            [3.0, -0.0, np.nan, 1.0, 0.0, -2.0, 3.0],
            dtype=np.float64,
        )

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for dtype in dtypes:
                for shape in shapes:
                    size = int(np.prod(shape, dtype=np.int64)) if shape else 1
                    values = np.resize(pattern, size)
                    if np.issubdtype(dtype, np.integer) or dtype is np.bool_:
                        values = np.nan_to_num(values, nan=2.0)
                        if np.issubdtype(dtype, np.unsignedinteger):
                            values = np.abs(values)
                    source_value = np.asarray(values, dtype=dtype).reshape(shape)
                    axes = (None,) if not shape else (
                        None,
                        *range(len(shape)),
                        *range(-len(shape), 0),
                    )
                    with runtime.from_numpy(source_value) as source:
                        for kind, kind_name in kinds.items():
                            for axis in axes:
                                for function_name, numpy_function in (
                                    ("cnp_sort", np.sort),
                                    ("cnp_argsort", np.argsort),
                                ):
                                    with self.subTest(
                                        dtype=np.dtype(dtype).name,
                                        shape=shape,
                                        kind=kind_name,
                                        axis=axis,
                                        function=function_name,
                                    ):
                                        with self.sort_result(
                                            runtime,
                                            source,
                                            function_name,
                                            axis,
                                            kind,
                                        ) as result:
                                            assert_array_equivalent(
                                                self,
                                                result,
                                                numpy_function(
                                                    source_value,
                                                    axis=axis,
                                                    kind=kind_name,
                                                ),
                                            )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_empty_dimensions_and_rank_zero_match_numpy(self) -> None:
        values_and_axes = (
            (np.empty((0,), dtype=np.float64), (None, 0, -1)),
            (np.empty((2, 0, 3), dtype=np.int64), (None, 0, 1, 2, -1)),
            (np.array(7, dtype=np.int64), (None,)),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for source_value, axes in values_and_axes:
                with runtime.from_numpy(source_value) as source:
                    for axis in axes:
                        for function_name, numpy_function in (
                            ("cnp_sort", np.sort),
                            ("cnp_argsort", np.argsort),
                        ):
                            with self.subTest(
                                shape=source_value.shape,
                                axis=axis,
                                function=function_name,
                            ):
                                with self.sort_result(
                                    runtime,
                                    source,
                                    function_name,
                                    axis,
                                    CNP_SORT_STABLE,
                                ) as result:
                                    assert_array_equivalent(
                                        self,
                                        result,
                                        numpy_function(
                                            source_value,
                                            axis=axis,
                                            kind="stable",
                                        ),
                                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_rank_zero_axis_and_invalid_kind_errors_are_explicit(self) -> None:
        scalar_value = np.array(7, dtype=np.int64)
        with CnumpyRuntime(DLL) as runtime, runtime.from_numpy(
            scalar_value
        ) as source:
            baseline = runtime.retained_bytes
            for function_name in ("cnp_sort", "cnp_argsort"):
                for axis in (0, -1):
                    with self.subTest(function=function_name, axis=axis):
                        with self.assertRaises(CnumpyError) as raised:
                            self.sort_result(
                                runtime,
                                source,
                                function_name,
                                axis,
                                CNP_SORT_STABLE,
                            )
                        self.assertEqual(-5, raised.exception.status)
                        self.assertEqual(
                            f"{function_name}_v2", raised.exception.function
                        )
                with self.subTest(function=function_name, kind=99):
                    with self.assertRaises(CnumpyError) as raised:
                        self.sort_result(
                            runtime,
                            source,
                            function_name,
                            None,
                            99,
                        )
                    self.assertEqual(-3, raised.exception.status)
                    self.assertEqual(
                        f"{function_name}_v2", raised.exception.function
                    )
                    self.assertIn("Invalid sort kind: 99", raised.exception.message)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_v2_null_sort_inputs_set_precise_native_errors(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name in ("cnp_sort_v2", "cnp_argsort_v2"):
                with self.subTest(function=function_name):
                    function = getattr(runtime.dll, function_name)
                    function.argtypes = [
                        ctypes.c_void_p,
                        ctypes.c_int,
                        ctypes.c_bool,
                        ctypes.c_int,
                    ]
                    function.restype = ctypes.c_void_p
                    runtime.dll.cnp_clear_error()

                    pointer = function(None, 0, True, CNP_SORT_STABLE)

                    self.assertFalse(pointer)
                    error = runtime.error_state()
                    self.assertEqual(-1, error.status)
                    self.assertEqual(function_name, error.function)
                    self.assertIn("arr must not be NULL", error.message)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_axis_none_and_v2_last_axis_are_unambiguous(self) -> None:
        source_value = np.array([[3, 1, 2], [6, 4, 5]], dtype=np.int64)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            for function_name, numpy_function in (
                ("cnp_sort", np.sort),
                ("cnp_argsort", np.argsort),
            ):
                legacy = stack.enter_context(
                    self.legacy_sort_result(
                        runtime,
                        source,
                        function_name,
                        CNP_AXIS_NONE,
                        CNP_SORT_STABLE,
                    )
                )
                last_axis = stack.enter_context(
                    self.sort_result(
                        runtime,
                        source,
                        function_name,
                        -1,
                        CNP_SORT_STABLE,
                    )
                )
                assert_array_equivalent(
                    self,
                    legacy,
                    numpy_function(source_value, axis=None, kind="stable"),
                )
                assert_array_equivalent(
                    self,
                    last_axis,
                    numpy_function(source_value, axis=-1, kind="stable"),
                )
                for axis, kind in ((3, CNP_SORT_STABLE), (0, 99)):
                    with self.subTest(
                        function=function_name, axis=axis, kind=kind
                    ):
                        with self.assertRaises(CnumpyError) as raised:
                            self.legacy_sort_result(
                                runtime, source, function_name, axis, kind
                            )
                        self.assertEqual(
                            function_name, raised.exception.function
                        )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_sort_exports_report_original_invalid_axis_exactly(self) -> None:
        shapes = ((), (5,), (2, 3), (2, 2, 3), (2, 2, 2, 3))
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for shape in shapes:
                size = int(np.prod(shape, dtype=np.int64)) if shape else 1
                source_value = np.arange(size, dtype=np.float64).reshape(shape)
                invalid_axes = (
                    (0, -1) if not shape else (len(shape), -len(shape) - 1)
                )
                with runtime.from_numpy(source_value) as source:
                    for function_name in ("cnp_sort", "cnp_argsort"):
                        for axis in invalid_axes:
                            expected_message = (
                                f"axis {axis} is out of bounds for array of "
                                f"dimension {len(shape)}"
                            )
                            with self.subTest(
                                function=function_name,
                                abi="v2",
                                shape=shape,
                                axis=axis,
                            ):
                                with self.assertRaises(CnumpyError) as raised:
                                    self.sort_result(
                                        runtime,
                                        source,
                                        function_name,
                                        axis,
                                        CNP_SORT_STABLE,
                                    )
                                self.assertEqual(-5, raised.exception.status)
                                self.assertEqual(
                                    f"{function_name}_v2",
                                    raised.exception.function,
                                )
                                self.assertEqual(
                                    expected_message, raised.exception.message
                                )
                            if not shape and axis == -1:
                                continue
                            with self.subTest(
                                function=function_name,
                                abi="legacy",
                                shape=shape,
                                axis=axis,
                            ):
                                with self.assertRaises(CnumpyError) as raised:
                                    self.legacy_sort_result(
                                        runtime,
                                        source,
                                        function_name,
                                        axis,
                                        CNP_SORT_STABLE,
                                    )
                                self.assertEqual(-5, raised.exception.status)
                                self.assertEqual(
                                    function_name, raised.exception.function
                                )
                                self.assertEqual(
                                    expected_message, raised.exception.message
                                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_direct_float64_sort_and_membership_candidates_match_numpy_edges(
        self,
    ) -> None:
        negative_nan = np.array(
            [0xFFF8000000000001], dtype=np.uint64
        ).view(np.float64)[0]
        positive_nan = np.array(
            [0x7FF8000000000002], dtype=np.uint64
        ).view(np.float64)[0]
        sort_value = np.array(
            [
                positive_nan,
                4.0,
                -0.0,
                np.inf,
                0.0,
                -3.0,
                negative_nan,
                -np.inf,
                4.0,
                0.0,
                -0.0,
            ],
            dtype=np.float64,
        )
        left_value = np.resize(sort_value, 4096).reshape(64, 64).T
        right_value = np.resize(
            np.array(
                [negative_nan, 0.0, 4.0, -7.0, np.inf],
                dtype=np.float64,
            ),
            128,
        )

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(sort_value) as sort_source:
                for kind, numpy_kind in (
                    (CNP_SORT_QUICKSORT, "quicksort"),
                    (CNP_SORT_HEAPSORT, "heapsort"),
                ):
                    with self.subTest(kind=numpy_kind), self.sort_result(
                        runtime,
                        sort_source,
                        "cnp_sort",
                        None,
                        kind,
                    ) as actual:
                        expected = np.sort(
                            sort_value, axis=None, kind=numpy_kind
                        )
                        assert_array_equivalent(self, actual, expected)
                        np.testing.assert_array_equal(
                            actual.to_numpy().view(np.uint64),
                            expected.view(np.uint64),
                            strict=True,
                        )

            left_storage_value = np.ascontiguousarray(left_value.T)
            with ExitStack() as stack:
                storage = stack.enter_context(
                    runtime.from_numpy(left_storage_value)
                )
                left = stack.enter_context(runtime.transpose(storage, (1, 0)))
                right = stack.enter_context(runtime.from_numpy(right_value))
                for invert in (False, True):
                    for function_name, numpy_function in (
                        ("cnp_in1d", np.in1d),
                        ("cnp_isin", np.isin),
                    ):
                        with self.subTest(
                            function=function_name, invert=invert
                        ), self.setop_result(
                            runtime,
                            function_name,
                            left,
                            right,
                            False,
                            invert,
                        ) as actual:
                            assert_array_equivalent(
                                self,
                                actual,
                                numpy_function(
                                    left_value,
                                    right_value,
                                    assume_unique=False,
                                    invert=invert,
                                ),
                            )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_unique_v2_returns_all_optional_int64_results(self) -> None:
        storage_value = np.array(
            [
                [2**53 + 1, -7, 2**53 + 1, 4],
                [np.iinfo(np.int64).min, 4, -7, 9],
                [9, np.iinfo(np.int64).max, 4, 2**53 + 3],
            ],
            dtype=np.int64,
        )
        logical_value = storage_value.transpose(1, 0)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            storage = stack.enter_context(runtime.from_numpy(storage_value))
            source = stack.enter_context(runtime.transpose(storage, (1, 0)))
            for return_index in (False, True):
                for return_inverse in (False, True):
                    for return_counts in (False, True):
                        with self.subTest(
                            return_index=return_index,
                            return_inverse=return_inverse,
                            return_counts=return_counts,
                        ):
                            actual = self.unique_results(
                                runtime,
                                source,
                                return_index,
                                return_inverse,
                                return_counts,
                            )
                            for result in actual:
                                stack.enter_context(result)
                            expected = np.unique(
                                logical_value,
                                return_index=return_index,
                                return_inverse=return_inverse,
                                return_counts=return_counts,
                                equal_nan=True,
                            )
                            expected_results = (
                                expected if isinstance(expected, tuple)
                                else (expected,)
                            )
                            self.assertEqual(
                                len(expected_results), len(actual)
                            )
                            for actual_result, expected_result in zip(
                                actual, expected_results
                            ):
                                assert_array_equivalent(
                                    self, actual_result, expected_result
                                )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_unique_v2_float_edges_empty_and_capacity_errors(self) -> None:
        negative_nan = np.array(
            [0xFFF8000000000001], dtype=np.uint64
        ).view(np.float64)[0]
        positive_nan = np.array(
            [0x7FF8000000000002], dtype=np.uint64
        ).view(np.float64)[0]
        source_value = np.array(
            [
                positive_nan,
                -0.0,
                0.0,
                negative_nan,
                -np.inf,
                np.inf,
                0.0,
                -0.0,
            ],
            dtype=np.float64,
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            empty = stack.enter_context(
                runtime.from_numpy(np.empty((2, 0), dtype=np.float32))
            )
            for return_index in (False, True):
                actual = self.unique_results(
                    runtime, source, return_index, True, True
                )
                for result in actual:
                    stack.enter_context(result)
                expected = np.unique(
                    source_value,
                    return_index=return_index,
                    return_inverse=True,
                    return_counts=True,
                    equal_nan=True,
                )
                for actual_result, expected_result in zip(actual, expected):
                    assert_array_equivalent(
                        self, actual_result, expected_result
                    )
                np.testing.assert_array_equal(
                    actual[0].to_numpy().view(np.uint64),
                    expected[0].view(np.uint64),
                    strict=True,
                )

            empty_actual = self.unique_results(
                runtime, empty, True, True, True
            )
            for result in empty_actual:
                stack.enter_context(result)
            empty_expected = np.unique(
                np.empty((2, 0), dtype=np.float32),
                return_index=True,
                return_inverse=True,
                return_counts=True,
                equal_nan=True,
            )
            for actual_result, expected_result in zip(
                empty_actual, empty_expected
            ):
                assert_array_equivalent(self, actual_result, expected_result)

            function = runtime.dll.cnp_unique_v2
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_bool,
                ctypes.c_bool,
                ctypes.c_bool,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_int,
            ]
            function.restype = ctypes.c_int
            too_small = (ctypes.c_void_p * 3)(1, 2, 3)
            runtime.dll.cnp_clear_error()
            status = function(
                source.pointer, True, True, True, too_small, len(too_small)
            )
            self.assertEqual(-4, status)
            self.assertEqual([None, None, None], list(too_small))
            error = runtime.error_state()
            self.assertEqual("cnp_unique_v2", error.function)
            self.assertIn("smaller than required 4", error.message)

            complex_source = stack.enter_context(
                runtime.from_numpy(
                    np.array([1 + 2j, 1 + 2j], dtype=np.complex64)
                )
            )
            unsupported = (ctypes.c_void_p * 1)(1)
            runtime.dll.cnp_clear_error()
            status = function(
                complex_source.pointer,
                False,
                False,
                False,
                unsupported,
                len(unsupported),
            )
            self.assertEqual(-3, status)
            self.assertEqual([None], list(unsupported))
            error = runtime.error_state()
            self.assertEqual("cnp_unique_v2", error.function)
            self.assertIn("not supported", error.message)
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_unique_v2_covers_all_represented_integer_and_float_dtypes(
        self,
    ) -> None:
        dtypes = (
            np.bool_,
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
            np.float16,
            np.float32,
            np.float64,
        )
        pattern = np.array(
            [3.0, -0.0, np.nan, 1.0, 0.0, -2.0, 3.0, np.inf],
            dtype=np.float64,
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for dtype in dtypes:
                values = pattern
                if np.issubdtype(dtype, np.integer) or dtype is np.bool_:
                    values = np.nan_to_num(values, nan=2.0, posinf=4.0)
                    if np.issubdtype(dtype, np.unsignedinteger):
                        values = np.abs(values)
                source_value = np.asarray(values, dtype=dtype).reshape(2, 4)
                with runtime.from_numpy(source_value) as source:
                    for return_index in (False, True):
                        for return_inverse in (False, True):
                            for return_counts in (False, True):
                                with self.subTest(
                                    dtype=np.dtype(dtype).name,
                                    return_index=return_index,
                                    return_inverse=return_inverse,
                                    return_counts=return_counts,
                                ):
                                    actual = self.unique_results(
                                        runtime,
                                        source,
                                        return_index,
                                        return_inverse,
                                        return_counts,
                                    )
                                    expected = np.unique(
                                        source_value,
                                        return_index=return_index,
                                        return_inverse=return_inverse,
                                        return_counts=return_counts,
                                        equal_nan=True,
                                    )
                                    expected_results = (
                                        expected
                                        if isinstance(expected, tuple)
                                        else (expected,)
                                    )
                                    try:
                                        self.assertEqual(
                                            len(expected_results), len(actual)
                                        )
                                        for actual_result, expected_result in zip(
                                            actual, expected_results
                                        ):
                                            assert_array_equivalent(
                                                self,
                                                actual_result,
                                                expected_result,
                                            )
                                    finally:
                                        for result in actual:
                                            result.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_unique_returns_values_only_for_every_optional_flag(
        self,
    ) -> None:
        source_value = np.array([3, 1, 3, 2, 1], dtype=np.int64)
        with CnumpyRuntime(DLL) as runtime, runtime.from_numpy(
            source_value
        ) as source:
            baseline = runtime.retained_bytes
            function = runtime.dll.cnp_unique
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_bool,
                ctypes.c_bool,
                ctypes.c_bool,
            ]
            function.restype = ctypes.c_void_p
            for return_index in (False, True):
                for return_inverse in (False, True):
                    for return_counts in (False, True):
                        with self.subTest(
                            return_index=return_index,
                            return_inverse=return_inverse,
                            return_counts=return_counts,
                        ):
                            runtime.dll.cnp_clear_error()
                            pointer = function(
                                source.pointer,
                                return_index,
                                return_inverse,
                                return_counts,
                            )
                            with runtime._owned_result(
                                pointer, "cnp_unique"
                            ) as result:
                                assert_array_equivalent(
                                    self,
                                    result,
                                    np.unique(source_value),
                                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_sort_unique_and_set_results_outlive_released_sources(self) -> None:
        left_value = np.array(
            [np.nan, -0.0, 3.0, 1.0, 0.0, 3.0], dtype=np.float64
        )
        right_value = np.array([0.0, 2.0, 3.0], dtype=np.float64)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            left = runtime.from_numpy(left_value)
            right = runtime.from_numpy(right_value)
            sorted_result = self.sort_result(
                runtime, left, "cnp_sort", None, CNP_SORT_STABLE
            )
            argsorted_result = self.sort_result(
                runtime, left, "cnp_argsort", None, CNP_SORT_STABLE
            )
            unique_results = self.unique_results(
                runtime, left, True, True, True
            )
            set_results = {
                "cnp_intersect1d": self.setop_result(
                    runtime, "cnp_intersect1d", left, right, False
                ),
                "cnp_union1d": self.setop_result(
                    runtime, "cnp_union1d", left, right
                ),
                "cnp_setdiff1d": self.setop_result(
                    runtime, "cnp_setdiff1d", left, right, False
                ),
                "cnp_setxor1d": self.setop_result(
                    runtime, "cnp_setxor1d", left, right, False
                ),
                "cnp_in1d": self.setop_result(
                    runtime, "cnp_in1d", left, right, False, False
                ),
                "cnp_isin": self.setop_result(
                    runtime, "cnp_isin", left, right, False, False
                ),
            }
            left.close()
            right.close()
            try:
                assert_array_equivalent(
                    self,
                    sorted_result,
                    np.sort(left_value, axis=None, kind="stable"),
                )
                assert_array_equivalent(
                    self,
                    argsorted_result,
                    np.argsort(left_value, axis=None, kind="stable"),
                )
                expected_unique = np.unique(
                    left_value,
                    return_index=True,
                    return_inverse=True,
                    return_counts=True,
                    equal_nan=True,
                )
                for actual, expected in zip(unique_results, expected_unique):
                    assert_array_equivalent(self, actual, expected)
                expected_set_results = {
                    "cnp_intersect1d": np.intersect1d(
                        left_value, right_value
                    ),
                    "cnp_union1d": np.union1d(left_value, right_value),
                    "cnp_setdiff1d": np.setdiff1d(
                        left_value, right_value
                    ),
                    "cnp_setxor1d": np.setxor1d(
                        left_value, right_value
                    ),
                    "cnp_in1d": np.in1d(left_value, right_value),
                    "cnp_isin": np.isin(left_value, right_value),
                }
                for function_name, result in set_results.items():
                    assert_array_equivalent(
                        self, result, expected_set_results[function_name]
                    )
            finally:
                for result in set_results.values():
                    result.close()
                for result in unique_results:
                    result.close()
                argsorted_result.close()
                sorted_result.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_mixed_float16_float32_set_operations_minimal_repro(self) -> None:
        left_value = np.asarray([1, 3], dtype=np.float16)
        right_value = np.asarray([2, 3], dtype=np.float32)
        operations = (
            ("cnp_intersect1d", np.intersect1d, (False,)),
            ("cnp_union1d", np.union1d, ()),
            ("cnp_setdiff1d", np.setdiff1d, (False,)),
            ("cnp_setxor1d", np.setxor1d, (False,)),
            ("cnp_in1d", np.in1d, (False, False)),
            ("cnp_isin", np.isin, (False, False)),
        )
        with CnumpyRuntime(DLL) as runtime, runtime.from_numpy(
            left_value
        ) as left, runtime.from_numpy(right_value) as right:
            baseline = runtime.retained_bytes
            for function_name, numpy_function, arguments in operations:
                with self.subTest(function=function_name):
                    with self.setop_result(
                        runtime,
                        function_name,
                        left,
                        right,
                        *arguments,
                    ) as actual:
                        assert_array_equivalent(
                            self,
                            actual,
                            numpy_function(
                                left_value, right_value, *arguments
                            ),
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_astype_half_and_nonhalf_share_correct_scalar_conversion(self) -> None:
        values = np.asarray([1.0, 2.0, 3.5], dtype=np.float64)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for source_dtype, _ in self._SET_DTYPES:
                for destination_dtype, destination_type in self._SET_DTYPES:
                    if (
                        source_dtype != np.dtype(np.float16)
                        and destination_dtype != np.dtype(np.float16)
                    ):
                        continue
                    source_value = values.astype(source_dtype)
                    with self.subTest(
                        source=source_dtype.name,
                        destination=destination_dtype.name,
                    ), runtime.from_numpy(source_value) as source, self.astype_result(
                        runtime, source, destination_type
                    ) as actual:
                        assert_array_equivalent(
                            self,
                            actual,
                            source_value.astype(destination_dtype),
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_set_operations_all_ordered_represented_dtype_pairs(self) -> None:
        value_operations = (
            ("cnp_intersect1d", np.intersect1d, (False,)),
            ("cnp_union1d", np.union1d, ()),
            ("cnp_setdiff1d", np.setdiff1d, (False,)),
            ("cnp_setxor1d", np.setxor1d, (False,)),
        )
        membership_operations = (
            ("cnp_in1d", np.in1d),
            ("cnp_isin", np.isin),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            call_count = 0
            for left_dtype, _ in self._SET_DTYPES:
                left_value = np.asarray([1, 3], dtype=left_dtype)
                for right_dtype, _ in self._SET_DTYPES:
                    right_value = np.asarray([2, 3], dtype=right_dtype)
                    with runtime.from_numpy(
                        left_value
                    ) as left, runtime.from_numpy(right_value) as right:
                        for function_name, numpy_function, arguments in value_operations:
                            with self.subTest(
                                function=function_name,
                                left=left_dtype.name,
                                right=right_dtype.name,
                            ), self.setop_result(
                                runtime,
                                function_name,
                                left,
                                right,
                                *arguments,
                            ) as actual:
                                assert_array_equivalent(
                                    self,
                                    actual,
                                    numpy_function(
                                        left_value, right_value, *arguments
                                    ),
                                )
                            call_count += 1
                        for function_name, numpy_function in membership_operations:
                            for invert in (False, True):
                                arguments = (False, invert)
                                with self.subTest(
                                    function=function_name,
                                    left=left_dtype.name,
                                    right=right_dtype.name,
                                    invert=invert,
                                ), self.setop_result(
                                    runtime,
                                    function_name,
                                    left,
                                    right,
                                    *arguments,
                                ) as actual:
                                    assert_array_equivalent(
                                        self,
                                        actual,
                                        numpy_function(
                                            left_value,
                                            right_value,
                                            assume_unique=False,
                                            invert=invert,
                                        ),
                                    )
                                call_count += 1
            self.assertEqual(1152, call_count)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_set_operation_errors_are_exact_and_atomic(self) -> None:
        null_contracts = (
            (
                "cnp_intersect1d",
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool],
                (False,),
                "input array must not be NULL",
            ),
            (
                "cnp_union1d",
                [ctypes.c_void_p, ctypes.c_void_p],
                (),
                "input arrays must not be NULL",
            ),
            (
                "cnp_setdiff1d",
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool],
                (False,),
                "input array must not be NULL",
            ),
            (
                "cnp_setxor1d",
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool],
                (False,),
                "input array must not be NULL",
            ),
            (
                "cnp_in1d",
                [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_bool,
                    ctypes.c_bool,
                ],
                (False, False),
                "input arrays must not be NULL",
            ),
            (
                "cnp_isin",
                [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_bool,
                    ctypes.c_bool,
                ],
                (False, False),
                "input arrays must not be NULL",
            ),
        )
        with CnumpyRuntime(DLL) as runtime, runtime.from_numpy(
            np.asarray([1.0], dtype=np.float64)
        ) as valid, runtime.from_numpy(
            np.asarray([1.0j], dtype=np.complex128)
        ) as unsupported:
            baseline = runtime.retained_bytes
            for function_name, argument_types, flags, message in null_contracts:
                function = getattr(runtime.dll, function_name)
                function.argtypes = argument_types
                function.restype = ctypes.c_void_p
                for left, right in (
                    (None, valid.pointer),
                    (valid.pointer, None),
                ):
                    with self.subTest(
                        function=function_name,
                        null="left" if left is None else "right",
                    ):
                        runtime.dll.cnp_clear_error()
                        self.assertFalse(function(left, right, *flags))
                        error = runtime.error_state()
                        self.assertEqual(-1, error.status)
                        self.assertEqual(function_name, error.function)
                        self.assertEqual(message, error.message)

            unsupported_messages = {
                "cnp_intersect1d": "Sorting dtype 16 is not supported",
                "cnp_union1d": "Set operation dtype 16 is not supported",
                "cnp_setdiff1d": "Sorting dtype 16 is not supported",
                "cnp_setxor1d": "Sorting dtype 16 is not supported",
                "cnp_in1d": "Set operation dtype 16 is not supported",
                "cnp_isin": "Set operation dtype 16 is not supported",
            }
            for function_name, _, flags, _ in null_contracts:
                with self.subTest(
                    function=function_name, error="unsupported_dtype"
                ):
                    with self.assertRaises(CnumpyError) as raised:
                        self.setop_result(
                            runtime,
                            function_name,
                            unsupported,
                            valid,
                            *flags,
                        )
                    self.assertEqual(-3, raised.exception.status)
                    self.assertEqual(function_name, raised.exception.function)
                    self.assertEqual(
                        unsupported_messages[function_name],
                        raised.exception.message,
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_set_operations_preserve_int64_and_membership_shape(self) -> None:
        left_storage_value = np.array(
            [
                [2**53 + 1, -7, 4],
                [9, 2**53 + 3, -7],
                [np.iinfo(np.int64).min, 4, 11],
            ],
            dtype=np.int64,
        )
        right_storage_value = np.array(
            [
                [4, 2**53 + 3, 12],
                [np.iinfo(np.int64).max, -7, 4],
            ],
            dtype=np.int64,
        )
        left_value = left_storage_value.transpose(1, 0)
        right_value = right_storage_value.transpose(1, 0)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            left_storage = stack.enter_context(
                runtime.from_numpy(left_storage_value)
            )
            right_storage = stack.enter_context(
                runtime.from_numpy(right_storage_value)
            )
            left = stack.enter_context(runtime.transpose(left_storage, (1, 0)))
            right = stack.enter_context(
                runtime.transpose(right_storage, (1, 0))
            )

            value_operations = (
                ("cnp_intersect1d", np.intersect1d, (False,)),
                ("cnp_union1d", np.union1d, ()),
                ("cnp_setdiff1d", np.setdiff1d, (False,)),
                ("cnp_setxor1d", np.setxor1d, (False,)),
            )
            for function_name, numpy_function, arguments in value_operations:
                with self.subTest(function=function_name):
                    actual = stack.enter_context(
                        self.setop_result(
                            runtime,
                            function_name,
                            left,
                            right,
                            *arguments,
                        )
                    )
                    expected = numpy_function(
                        left_value,
                        right_value,
                        *arguments,
                    )
                    assert_array_equivalent(self, actual, expected)

            for invert in (False, True):
                in1d = stack.enter_context(
                    self.setop_result(
                        runtime, "cnp_in1d", left, right, False, invert
                    )
                )
                isin = stack.enter_context(
                    self.setop_result(
                        runtime, "cnp_isin", left, right, False, invert
                    )
                )
                assert_array_equivalent(
                    self,
                    in1d,
                    np.in1d(left_value, right_value, invert=invert),
                )
                assert_array_equivalent(
                    self,
                    isin,
                    np.isin(left_value, right_value, invert=invert),
                )

            unique_left_value = np.unique(left_value)
            unique_right_value = np.unique(right_value)
            unique_left = stack.enter_context(
                runtime.from_numpy(unique_left_value)
            )
            unique_right = stack.enter_context(
                runtime.from_numpy(unique_right_value)
            )
            for function_name, numpy_function in (
                ("cnp_intersect1d", np.intersect1d),
                ("cnp_setdiff1d", np.setdiff1d),
                ("cnp_setxor1d", np.setxor1d),
            ):
                actual = stack.enter_context(
                    self.setop_result(
                        runtime,
                        function_name,
                        unique_left,
                        unique_right,
                        True,
                    )
                )
                expected = numpy_function(
                    unique_left_value,
                    unique_right_value,
                    assume_unique=True,
                )
                assert_array_equivalent(self, actual, expected)
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_set_operations_match_float_nan_zero_empty_and_mixed_dtype(self) -> None:
        negative_nan = np.array(
            [0xFFF8000000000001], dtype=np.uint64
        ).view(np.float64)[0]
        positive_nan = np.array(
            [0x7FF8000000000002], dtype=np.uint64
        ).view(np.float64)[0]
        float_pairs = (
            (
                np.array(
                    [positive_nan, -0.0, 2.0, np.inf, -0.0],
                    dtype=np.float64,
                ),
                np.array(
                    [negative_nan, 0.0, 3.0, np.inf], dtype=np.float64
                ),
            ),
            (
                np.empty((0,), dtype=np.float64),
                np.array([negative_nan, -0.0], dtype=np.float64),
            ),
            (
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            ),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for left_value, right_value in float_pairs:
                left = stack.enter_context(runtime.from_numpy(left_value))
                right = stack.enter_context(runtime.from_numpy(right_value))
                for function_name, numpy_function, arguments in (
                    ("cnp_intersect1d", np.intersect1d, (False,)),
                    ("cnp_union1d", np.union1d, ()),
                    ("cnp_setdiff1d", np.setdiff1d, (False,)),
                    ("cnp_setxor1d", np.setxor1d, (False,)),
                ):
                    with self.subTest(
                        function=function_name,
                        left_size=left_value.size,
                        right_size=right_value.size,
                    ):
                        actual = stack.enter_context(
                            self.setop_result(
                                runtime,
                                function_name,
                                left,
                                right,
                                *arguments,
                            )
                        )
                        expected = numpy_function(
                            left_value, right_value, *arguments
                        )
                        assert_array_equivalent(self, actual, expected)
                        np.testing.assert_array_equal(
                            actual.to_numpy().view(np.uint64),
                            expected.view(np.uint64),
                            strict=True,
                        )
                for invert in (False, True):
                    for function_name, numpy_function in (
                        ("cnp_in1d", np.in1d),
                        ("cnp_isin", np.isin),
                    ):
                        actual = stack.enter_context(
                            self.setop_result(
                                runtime,
                                function_name,
                                left,
                                right,
                                False,
                                invert,
                            )
                        )
                        expected = numpy_function(
                            left_value, right_value, invert=invert
                        )
                        assert_array_equivalent(self, actual, expected)

            mixed_left_value = np.array([1, 2, 3], dtype=np.int32)
            mixed_right_value = np.array(
                [2.0, 4.5, np.nan], dtype=np.float64
            )
            mixed_left = stack.enter_context(
                runtime.from_numpy(mixed_left_value)
            )
            mixed_right = stack.enter_context(
                runtime.from_numpy(mixed_right_value)
            )
            for function_name, numpy_function, arguments in (
                ("cnp_intersect1d", np.intersect1d, (False,)),
                ("cnp_union1d", np.union1d, ()),
                ("cnp_setxor1d", np.setxor1d, (False,)),
            ):
                actual = stack.enter_context(
                    self.setop_result(
                        runtime,
                        function_name,
                        mixed_left,
                        mixed_right,
                        *arguments,
                    )
                )
                expected = numpy_function(
                    mixed_left_value, mixed_right_value, *arguments
                )
                assert_array_equivalent(self, actual, expected)
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_membership_operations_cover_rank_zero_through_four_numeric_dtypes(
        self,
    ) -> None:
        shapes = ((), (5,), (2, 3), (2, 2, 3), (2, 2, 2, 3), (2, 0, 3))
        left_pattern = np.asarray(
            [3.0, -0.0, np.nan, 1.0, 0.0, -2.0], dtype=np.float64
        )
        right_pattern = np.asarray(
            [np.nan, 0.0, 2.0, 3.0], dtype=np.float64
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            call_count = 0
            for dtype, _ in self._SET_DTYPES:
                prepared_left = left_pattern
                prepared_right = right_pattern
                if dtype.kind in "biu":
                    prepared_left = np.nan_to_num(prepared_left, nan=2.0)
                    prepared_right = np.nan_to_num(prepared_right, nan=2.0)
                    if dtype.kind == "u":
                        prepared_left = np.abs(prepared_left)
                        prepared_right = np.abs(prepared_right)
                prepared_left = np.asarray(prepared_left, dtype=dtype)
                right_value = np.asarray(prepared_right, dtype=dtype)
                with runtime.from_numpy(right_value) as right:
                    for shape in shapes:
                        size = (
                            int(np.prod(shape, dtype=np.int64))
                            if shape
                            else 1
                        )
                        axes = tuple(reversed(range(len(shape))))
                        storage_shape = tuple(reversed(shape))
                        storage_value = np.resize(
                            prepared_left, size
                        ).reshape(storage_shape)
                        left_value = (
                            storage_value.transpose(axes)
                            if len(shape) >= 2
                            else storage_value
                        )
                        with ExitStack() as stack:
                            storage = stack.enter_context(
                                runtime.from_numpy(storage_value)
                            )
                            left = (
                                stack.enter_context(
                                    runtime.transpose(storage, axes)
                                )
                                if len(shape) >= 2
                                else storage
                            )
                            for invert in (False, True):
                                for function_name, numpy_function in (
                                    ("cnp_in1d", np.in1d),
                                    ("cnp_isin", np.isin),
                                ):
                                    with self.subTest(
                                        dtype=dtype.name,
                                        shape=shape,
                                        function=function_name,
                                        invert=invert,
                                        assume_unique=False,
                                    ), self.setop_result(
                                        runtime,
                                        function_name,
                                        left,
                                        right,
                                        False,
                                        invert,
                                    ) as actual:
                                        assert_array_equivalent(
                                            self,
                                            actual,
                                            numpy_function(
                                                left_value,
                                                right_value,
                                                assume_unique=False,
                                                invert=invert,
                                            ),
                                        )
                                    call_count += 1

                unique_left_value = np.unique(prepared_left)
                unique_right_value = np.unique(right_value)
                with runtime.from_numpy(
                    unique_left_value
                ) as unique_left, runtime.from_numpy(
                    unique_right_value
                ) as unique_right:
                    for assume_unique in (False, True):
                        for invert in (False, True):
                            for function_name, numpy_function in (
                                ("cnp_in1d", np.in1d),
                                ("cnp_isin", np.isin),
                            ):
                                with self.subTest(
                                    dtype=dtype.name,
                                    function=function_name,
                                    invert=invert,
                                    assume_unique=assume_unique,
                                ), self.setop_result(
                                    runtime,
                                    function_name,
                                    unique_left,
                                    unique_right,
                                    assume_unique,
                                    invert,
                                ) as actual:
                                    assert_array_equivalent(
                                        self,
                                        actual,
                                        numpy_function(
                                            unique_left_value,
                                            unique_right_value,
                                            assume_unique=assume_unique,
                                            invert=invert,
                                        ),
                                    )
                                call_count += 1
            self.assertEqual(384, call_count)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_set_value_operations_cover_rank_zero_through_four_numeric_dtypes(
        self,
    ) -> None:
        dtypes = (
            np.bool_,
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
            np.float16,
            np.float32,
            np.float64,
        )
        shapes = ((), (5,), (2, 3), (2, 2, 3), (2, 2, 2, 3))
        left_pattern = np.array(
            [3.0, -0.0, np.nan, 1.0, 0.0, -2.0], dtype=np.float64
        )
        right_pattern = np.array(
            [np.nan, 0.0, 2.0, 3.0], dtype=np.float64
        )
        operations = (
            ("cnp_intersect1d", np.intersect1d, (False,)),
            ("cnp_union1d", np.union1d, ()),
            ("cnp_setdiff1d", np.setdiff1d, (False,)),
            ("cnp_setxor1d", np.setxor1d, (False,)),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for dtype in dtypes:
                prepared_left = left_pattern
                prepared_right = right_pattern
                if np.issubdtype(dtype, np.integer) or dtype is np.bool_:
                    prepared_left = np.nan_to_num(prepared_left, nan=2.0)
                    prepared_right = np.nan_to_num(prepared_right, nan=2.0)
                    if np.issubdtype(dtype, np.unsignedinteger):
                        prepared_left = np.abs(prepared_left)
                        prepared_right = np.abs(prepared_right)
                right_value = np.asarray(prepared_right, dtype=dtype)
                with runtime.from_numpy(right_value) as right:
                    for shape in shapes:
                        size = (
                            int(np.prod(shape, dtype=np.int64))
                            if shape
                            else 1
                        )
                        left_value = np.resize(
                            np.asarray(prepared_left, dtype=dtype), size
                        ).reshape(shape)
                        with runtime.from_numpy(left_value) as left:
                            for function_name, numpy_function, arguments in operations:
                                with self.subTest(
                                    dtype=np.dtype(dtype).name,
                                    shape=shape,
                                    function=function_name,
                                ):
                                    with self.setop_result(
                                        runtime,
                                        function_name,
                                        left,
                                        right,
                                        *arguments,
                                    ) as actual:
                                        assert_array_equivalent(
                                            self,
                                            actual,
                                            numpy_function(
                                                left_value,
                                                right_value,
                                                *arguments,
                                            ),
                                        )
            self.assertEqual(baseline, runtime.retained_bytes)


class SearchsortedSemanticsTests(unittest.TestCase):
    def slice_view(
        self,
        runtime: CnumpyRuntime,
        source,
        slices: tuple[_CnpSlice, ...],
    ):
        function = runtime.dll.cnp_array_slice
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(_CnpSlice),
        ]
        function.restype = ctypes.c_void_p
        storage = (_CnpSlice * len(slices))(*slices)
        runtime.dll.cnp_clear_error()
        pointer = function(source.pointer, len(slices), storage)
        return runtime._owned_result(pointer, "cnp_array_slice")

    def searchsorted_result(
        self,
        runtime: CnumpyRuntime,
        source,
        values,
        side: str,
    ):
        function = runtime.dll.cnp_searchsorted
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        pointer = function(
            source.pointer,
            values.pointer,
            side.encode("ascii"),
        )
        return runtime._owned_result(pointer, "cnp_searchsorted")

    def searchsorted_v2_result(
        self,
        runtime: CnumpyRuntime,
        source,
        values,
        side: str,
        sorter=None,
    ):
        function = runtime.dll.cnp_searchsorted_v2
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
        ]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        pointer = function(
            source.pointer,
            values.pointer,
            side.encode("ascii"),
            None if sorter is None else sorter.pointer,
        )
        return runtime._owned_result(pointer, "cnp_searchsorted_v2")

    def test_legacy_searchsorted_uses_input_order_without_sorting(self) -> None:
        source_values = np.array([3, 1, 2], dtype=np.int64)
        query_values = np.array([0, 1, 2, 3, 4], dtype=np.int64)

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with ExitStack() as stack:
                source = stack.enter_context(runtime.from_numpy(source_values))
                queries = stack.enter_context(runtime.from_numpy(query_values))
                actual = stack.enter_context(
                    self.searchsorted_result(
                        runtime, source, queries, "left"
                    )
                )
                assert_array_equivalent(
                    self,
                    actual,
                    np.searchsorted(
                        source_values, query_values, side="left"
                    ),
                )
                self.assertEqual(tuple(source_values), source.values())
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_searchsorted_preserves_scalar_and_matrix_query_shape(self) -> None:
        source_values = np.array([1, 3, 5], dtype=np.int32)
        query_cases = (
            np.array(4, dtype=np.int32),
            np.array([[0, 1], [4, 6]], dtype=np.int32),
        )

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with ExitStack() as stack:
                source = stack.enter_context(runtime.from_numpy(source_values))
                for query_values in query_cases:
                    with self.subTest(shape=query_values.shape):
                        queries = stack.enter_context(
                            runtime.from_numpy(query_values)
                        )
                        actual = stack.enter_context(
                            self.searchsorted_result(
                                runtime, source, queries, "right"
                            )
                        )
                        assert_array_equivalent(
                            self,
                            actual,
                            np.searchsorted(
                                source_values,
                                query_values,
                                side="right",
                            ),
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_searchsorted_preserves_int64_and_uint64_precision(self) -> None:
        cases = (
            (
                np.array(
                    [2**53, 2**53 + 1, 2**63 - 1], dtype=np.int64
                ),
                np.array([2**53 + 1, 2**63 - 1], dtype=np.int64),
            ),
            (
                np.array(
                    [2**63, 2**63 + 1, 2**64 - 1], dtype=np.uint64
                ),
                np.array([2**63 + 1, 2**64 - 1], dtype=np.uint64),
            ),
        )

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with ExitStack() as stack:
                for source_values, query_values in cases:
                    source = stack.enter_context(
                        runtime.from_numpy(source_values)
                    )
                    queries = stack.enter_context(
                        runtime.from_numpy(query_values)
                    )
                    for side in ("left", "right"):
                        with self.subTest(
                            dtype=source_values.dtype, side=side
                        ):
                            actual = stack.enter_context(
                                self.searchsorted_result(
                                    runtime, source, queries, side
                                )
                            )
                            assert_array_equivalent(
                                self,
                                actual,
                                np.searchsorted(
                                    source_values,
                                    query_values,
                                    side=side,
                                ),
                            )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_searchsorted_v2_applies_sorter_without_mutation(self) -> None:
        source_values = np.array([40, 10, 30, 20], dtype=np.int64)
        query_values = np.array([[10, 25], [40, 50]], dtype=np.int64)
        sorter_values = np.argsort(source_values, kind="stable")

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with ExitStack() as stack:
                source = stack.enter_context(runtime.from_numpy(source_values))
                queries = stack.enter_context(runtime.from_numpy(query_values))
                sorter = stack.enter_context(runtime.from_numpy(sorter_values))
                for side in ("left", "right"):
                    with self.subTest(side=side):
                        actual = stack.enter_context(
                            self.searchsorted_v2_result(
                                runtime,
                                source,
                                queries,
                                side,
                                sorter,
                            )
                        )
                        assert_array_equivalent(
                            self,
                            actual,
                            np.searchsorted(
                                source_values,
                                query_values,
                                side=side,
                                sorter=sorter_values,
                            ),
                        )
                self.assertEqual(tuple(source_values), source.values())
                self.assertEqual(tuple(sorter_values), sorter.values())
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_ctypes_adapter_exposes_searchsorted_v2_contract(self) -> None:
        source_values = np.array([8, 2, 5, 1], dtype=np.int64)
        query_values = np.array([1, 4, 8], dtype=np.int64)
        sorter_values = np.argsort(source_values, kind="stable")

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_values))
            queries = stack.enter_context(runtime.from_numpy(query_values))
            sorter = stack.enter_context(runtime.from_numpy(sorter_values))
            actual = stack.enter_context(
                runtime.searchsorted(
                    source, queries, side="right", sorter=sorter
                )
            )
            assert_array_equivalent(
                self,
                actual,
                np.searchsorted(
                    source_values,
                    query_values,
                    side="right",
                    sorter=sorter_values,
                ),
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_searchsorted_v2_invalid_inputs_surface_precise_errors(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            source = stack.enter_context(
                runtime.from_numpy(np.array([10, 20, 30, 40], dtype=np.int64))
            )
            source_2d = stack.enter_context(
                runtime.from_numpy(
                    np.array([[10, 20], [30, 40]], dtype=np.int64)
                )
            )
            queries = stack.enter_context(
                runtime.from_numpy(np.array([50], dtype=np.int64))
            )
            sorter_short = stack.enter_context(
                runtime.from_numpy(np.array([0, 1], dtype=np.int64))
            )
            sorter_2d = stack.enter_context(
                runtime.from_numpy(
                    np.array([[0, 1], [2, 3]], dtype=np.int64)
                )
            )
            sorter_float = stack.enter_context(
                runtime.from_numpy(
                    np.array([0, 1, 2, 3], dtype=np.float64)
                )
            )
            sorter_oob = stack.enter_context(
                runtime.from_numpy(np.array([0, 1, 2, 4], dtype=np.int64))
            )

            invalid_cases = (
                (source_2d, "left", None, -4, "one-dimensional"),
                (source, "middle", None, -13, "left.*right"),
                (source, "left", sorter_short, -4, "size"),
                (source, "left", sorter_2d, -3, "one-dimensional"),
                (source, "left", sorter_float, -3, "integer"),
                (source, "left", sorter_oob, -6, "out of range"),
            )
            for (
                invalid_source,
                side,
                sorter,
                expected_status,
                message_pattern,
            ) in invalid_cases:
                with self.subTest(
                    side=side,
                    sorter=None if sorter is None else sorter.numpy_dtype,
                    status=expected_status,
                ):
                    error = None
                    try:
                        unexpected = self.searchsorted_v2_result(
                            runtime,
                            invalid_source,
                            queries,
                            side,
                            sorter,
                        )
                    except CnumpyError as raised:
                        error = raised
                    else:
                        unexpected.close()
                        self.fail("invalid searchsorted call returned an array")
                    assert error is not None
                    self.assertRegex(error.message, message_pattern)
                    self.assertEqual(expected_status, error.status)
                    self.assertEqual(
                        "cnp_searchsorted_v2", error.function
                    )

            function = runtime.dll.cnp_searchsorted_v2
            runtime.dll.cnp_clear_error()
            pointer = function(None, queries.pointer, b"left", None)
            self.assertFalse(pointer)
            error = runtime.error_state()
            self.assertEqual(-1, error.status)
            self.assertEqual("cnp_searchsorted_v2", error.function)
            self.assertRegex(error.message, "source.*NULL")

    def test_searchsorted_float16_uses_numpy_total_order(self) -> None:
        source_values = np.array(
            [-np.inf, -0.0, 0.0, 1.0, np.inf, np.nan, np.nan],
            dtype=np.float16,
        )
        query_values = np.array(
            [-0.0, 0.0, 0.5, np.inf, np.nan], dtype=np.float16
        )

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with ExitStack() as stack:
                source = stack.enter_context(runtime.from_numpy(source_values))
                queries = stack.enter_context(runtime.from_numpy(query_values))
                for side in ("left", "right"):
                    with self.subTest(side=side):
                        actual = stack.enter_context(
                            self.searchsorted_v2_result(
                                runtime, source, queries, side
                            )
                        )
                        assert_array_equivalent(
                            self,
                            actual,
                            np.searchsorted(
                                source_values, query_values, side=side
                            ),
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_searchsorted_complex_uses_lexicographic_numpy_order(self) -> None:
        raw_values = [
            complex(-np.inf, 0.0),
            complex(-1.0, 2.0),
            complex(-1.0, 3.0),
            complex(-0.0, -0.0),
            complex(0.0, 0.0),
            complex(1.0, -np.inf),
            complex(1.0, 2.0),
            complex(np.inf, 0.0),
            complex(np.nan, -np.inf),
            complex(np.nan, 1.0),
            complex(np.nan, np.nan),
        ]
        raw_queries = [
            complex(-1.0, 2.0),
            complex(0.0, 0.0),
            complex(1.0, 2.0),
            complex(np.nan, 1.0),
            complex(np.nan, np.nan),
        ]

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with ExitStack() as stack:
                for dtype in (np.complex64, np.complex128):
                    source_values = np.array(raw_values, dtype=dtype)
                    query_values = np.array(raw_queries, dtype=dtype)
                    source = stack.enter_context(
                        runtime.from_numpy(source_values)
                    )
                    queries = stack.enter_context(
                        runtime.from_numpy(query_values)
                    )
                    for side in ("left", "right"):
                        with self.subTest(dtype=dtype, side=side):
                            actual = stack.enter_context(
                                self.searchsorted_v2_result(
                                    runtime, source, queries, side
                                )
                            )
                            assert_array_equivalent(
                                self,
                                actual,
                                np.searchsorted(
                                    source_values,
                                    query_values,
                                    side=side,
                                ),
                            )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_searchsorted_promotes_mixed_real_and_complex_dtypes(self) -> None:
        cases = (
            (
                np.array([1 + 1j, 1 + 2j, 2 + 0j], dtype=np.complex128),
                np.array([1.0, 1.5, 2.0], dtype=np.float64),
            ),
            (
                np.array([1.0, 2.0, 3.0], dtype=np.float64),
                np.array([2 + 0j, 2 + 1j], dtype=np.complex128),
            ),
            (
                np.array([1 + 1j, 2 + 0j, 3 + 2j], dtype=np.complex64),
                np.array([1.0, 2.5], dtype=np.float32),
            ),
        )

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with ExitStack() as stack:
                for source_values, query_values in cases:
                    source = stack.enter_context(
                        runtime.from_numpy(source_values)
                    )
                    queries = stack.enter_context(
                        runtime.from_numpy(query_values)
                    )
                    for side in ("left", "right"):
                        with self.subTest(
                            source=source_values.dtype,
                            queries=query_values.dtype,
                            side=side,
                        ):
                            actual = stack.enter_context(
                                self.searchsorted_v2_result(
                                    runtime, source, queries, side
                                )
                            )
                            assert_array_equivalent(
                                self,
                                actual,
                                np.searchsorted(
                                    source_values,
                                    query_values,
                                    side=side,
                                ),
                            )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_searchsorted_pairwise_numeric_dtype_promotion_matches_numpy(
        self,
    ) -> None:
        dtypes = (
            np.bool_,
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
            np.float16,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        )

        def source_for(dtype):
            if dtype is np.bool_:
                values = [False, False, True]
            elif np.issubdtype(dtype, np.unsignedinteger):
                values = [0, 1, 3, 5]
            elif np.issubdtype(dtype, np.complexfloating):
                values = [-3 + 0j, -1 + 1j, 0 + 0j, 2 - 1j, 5 + 0j]
            else:
                values = [-3, -1, 0, 2, 5]
            return np.sort(np.asarray(values, dtype=dtype), kind="stable")

        def queries_for(dtype):
            if dtype is np.bool_:
                values = [False, True]
            elif np.issubdtype(dtype, np.unsignedinteger):
                values = [0, 2, 6]
            elif np.issubdtype(dtype, np.complexfloating):
                values = [-2 + 1j, 0 + 0j, 3 - 1j]
            else:
                values = [-2, 0, 3]
            return np.asarray(values, dtype=dtype)

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for source_dtype in dtypes:
                source_values = source_for(source_dtype)
                with runtime.from_numpy(source_values) as source:
                    for query_dtype in dtypes:
                        query_values = queries_for(query_dtype)
                        with self.subTest(
                            source=np.dtype(source_dtype),
                            queries=np.dtype(query_dtype),
                        ), runtime.from_numpy(query_values) as queries:
                            for side in ("left", "right"):
                                with self.searchsorted_v2_result(
                                    runtime, source, queries, side
                                ) as actual:
                                    assert_array_equivalent(
                                        self,
                                        actual,
                                        np.searchsorted(
                                            source_values,
                                            query_values,
                                            side=side,
                                        ),
                                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_searchsorted_handles_negative_stride_source_and_queries(
        self,
    ) -> None:
        source_storage_values = np.array(
            [5.0, 91.0, 3.0, 92.0, 1.0], dtype=np.float64
        )
        source_values = source_storage_values[::-2]
        query_storage_values = np.array(
            [[6.0, 4.0, 2.0], [5.0, 3.0, 1.0]], dtype=np.float64
        )
        query_values = query_storage_values[:, ::-1].T

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with ExitStack() as stack:
                source_storage = stack.enter_context(
                    runtime.from_numpy(source_storage_values)
                )
                source = stack.enter_context(
                    self.slice_view(
                        runtime,
                        source_storage,
                        (_CnpSlice(0, 0, -2, False, False, True),),
                    )
                )
                query_storage = stack.enter_context(
                    runtime.from_numpy(query_storage_values)
                )
                reversed_queries = stack.enter_context(
                    self.slice_view(
                        runtime,
                        query_storage,
                        (
                            _CnpSlice(0, 0, 0, False, False, False),
                            _CnpSlice(0, 0, -1, False, False, True),
                        ),
                    )
                )
                queries = stack.enter_context(
                    runtime.transpose(reversed_queries, (1, 0))
                )
                self.assertFalse(source.c_contiguous)
                self.assertFalse(queries.c_contiguous)
                for side in ("left", "right"):
                    with self.subTest(side=side):
                        actual = stack.enter_context(
                            self.searchsorted_v2_result(
                                runtime, source, queries, side
                            )
                        )
                        assert_array_equivalent(
                            self,
                            actual,
                            np.searchsorted(
                                source_values, query_values, side=side
                            ),
                        )
            self.assertEqual(baseline, runtime.retained_bytes)


class DigitizeSemanticsTests(unittest.TestCase):
    def slice_view(
        self,
        runtime: CnumpyRuntime,
        source,
        slices: tuple[_CnpSlice, ...],
    ):
        function = runtime.dll.cnp_array_slice
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(_CnpSlice),
        ]
        function.restype = ctypes.c_void_p
        storage = (_CnpSlice * len(slices))(*slices)
        runtime.dll.cnp_clear_error()
        pointer = function(source.pointer, len(slices), storage)
        return runtime._owned_result(pointer, "cnp_array_slice")

    def test_digitize_preserves_scalar_and_matrix_shape_for_both_orders(
        self,
    ) -> None:
        query_cases = (
            np.array(2.0, dtype=np.float64),
            np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float64),
        )
        bin_cases = (
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([3.0, 2.0, 1.0], dtype=np.float64),
        )

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with ExitStack() as stack:
                for query_values in query_cases:
                    queries = stack.enter_context(
                        runtime.from_numpy(query_values)
                    )
                    for bin_values in bin_cases:
                        bins = stack.enter_context(
                            runtime.from_numpy(bin_values)
                        )
                        for right in (False, True):
                            with self.subTest(
                                shape=query_values.shape,
                                decreasing=bin_values[0] > bin_values[-1],
                                right=right,
                            ):
                                actual = stack.enter_context(
                                    runtime.digitize(
                                        queries, bins, right=right
                                    )
                                )
                                assert_array_equivalent(
                                    self,
                                    actual,
                                    np.digitize(
                                        query_values,
                                        bin_values,
                                        right=right,
                                    ),
                                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_digitize_handles_empty_repeated_nan_and_infinite_bins(
        self,
    ) -> None:
        cases = (
            (
                np.array([0.0, 1.0, 2.0], dtype=np.float64),
                np.array([1.0, 1.0, 1.0], dtype=np.float64),
            ),
            (
                np.array(
                    [-np.inf, -0.0, 0.0, np.inf, np.nan],
                    dtype=np.float64,
                ),
                np.array(
                    [-np.inf, -0.0, 0.0, np.inf, np.nan],
                    dtype=np.float64,
                ),
            ),
            (
                np.array(
                    [-np.inf, -0.0, 0.0, np.inf, np.nan],
                    dtype=np.float64,
                ),
                np.array(
                    [np.nan, np.inf, 0.0, -0.0, -np.inf],
                    dtype=np.float64,
                ),
            ),
            (
                np.array([1.0, np.nan], dtype=np.float64),
                np.array([], dtype=np.float64),
            ),
        )

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with ExitStack() as stack:
                for query_values, bin_values in cases:
                    queries = stack.enter_context(
                        runtime.from_numpy(query_values)
                    )
                    bins = stack.enter_context(runtime.from_numpy(bin_values))
                    for right in (False, True):
                        with self.subTest(
                            bins=bin_values.tolist(), right=right
                        ):
                            actual = stack.enter_context(
                                runtime.digitize(
                                    queries, bins, right=right
                                )
                            )
                            assert_array_equivalent(
                                self,
                                actual,
                                np.digitize(
                                    query_values, bin_values, right=right
                                ),
                            )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_digitize_preserves_int64_uint64_precision_and_promotion(
        self,
    ) -> None:
        cases = (
            (
                np.array(
                    [2**53, 2**53 + 1, 2**63 - 1], dtype=np.int64
                ),
                np.array(
                    [2**53, 2**53 + 1, 2**63 - 1], dtype=np.int64
                ),
            ),
            (
                np.array([2**63 - 1], dtype=np.int64),
                np.array([0, 2**63, 2**64 - 1], dtype=np.uint64),
            ),
            (
                np.array([2**63, 2**64 - 1], dtype=np.uint64),
                np.array([-1, 0, 2**63 - 1], dtype=np.int64),
            ),
        )

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with ExitStack() as stack:
                for query_values, bin_values in cases:
                    queries = stack.enter_context(
                        runtime.from_numpy(query_values)
                    )
                    bins = stack.enter_context(runtime.from_numpy(bin_values))
                    for right in (False, True):
                        with self.subTest(
                            queries=query_values.dtype,
                            bins=bin_values.dtype,
                            right=right,
                        ):
                            actual = stack.enter_context(
                                runtime.digitize(
                                    queries, bins, right=right
                                )
                            )
                            assert_array_equivalent(
                                self,
                                actual,
                                np.digitize(
                                    query_values, bin_values, right=right
                                ),
                            )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_digitize_pairwise_real_dtype_promotion_matches_numpy(
        self,
    ) -> None:
        dtypes = (
            np.bool_,
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
            np.float16,
            np.float32,
            np.float64,
        )

        def bins_for(dtype):
            if dtype is np.bool_:
                values = [False, True]
            elif np.issubdtype(dtype, np.unsignedinteger):
                values = [0, 1, 3, 5]
            else:
                values = [-3, -1, 0, 2, 5]
            return np.asarray(values, dtype=dtype)

        def queries_for(dtype):
            if dtype is np.bool_:
                values = [False, True]
            elif np.issubdtype(dtype, np.unsignedinteger):
                values = [0, 2, 6]
            else:
                values = [-2, 0, 3]
            return np.asarray(values, dtype=dtype)

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for query_dtype in dtypes:
                query_values = queries_for(query_dtype)
                with runtime.from_numpy(query_values) as queries:
                    for bin_dtype in dtypes:
                        bin_values = bins_for(bin_dtype)
                        with self.subTest(
                            queries=np.dtype(query_dtype),
                            bins=np.dtype(bin_dtype),
                        ), runtime.from_numpy(bin_values) as bins:
                            for right in (False, True):
                                with runtime.digitize(
                                    queries, bins, right=right
                                ) as actual:
                                    assert_array_equivalent(
                                        self,
                                        actual,
                                        np.digitize(
                                            query_values,
                                            bin_values,
                                            right=right,
                                        ),
                                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_digitize_handles_noncontiguous_queries_and_decreasing_bins(
        self,
    ) -> None:
        query_storage_values = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, np.nan]],
            dtype=np.float64,
        )
        query_values = query_storage_values.T
        bin_storage_values = np.array(
            [0.0, 91.0, 2.0, 92.0, 4.0], dtype=np.float64
        )
        bin_values = bin_storage_values[::-2]

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with ExitStack() as stack:
                query_storage = stack.enter_context(
                    runtime.from_numpy(query_storage_values)
                )
                queries = stack.enter_context(
                    runtime.transpose(query_storage, (1, 0))
                )
                bin_storage = stack.enter_context(
                    runtime.from_numpy(bin_storage_values)
                )
                bins = stack.enter_context(
                    self.slice_view(
                        runtime,
                        bin_storage,
                        (_CnpSlice(0, 0, -2, False, False, True),),
                    )
                )
                self.assertFalse(queries.c_contiguous)
                self.assertFalse(bins.c_contiguous)
                for right in (False, True):
                    with self.subTest(right=right):
                        actual = stack.enter_context(
                            runtime.digitize(queries, bins, right=right)
                        )
                        assert_array_equivalent(
                            self,
                            actual,
                            np.digitize(
                                query_values, bin_values, right=right
                            ),
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_digitize_invalid_inputs_surface_precise_errors(self) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            queries = stack.enter_context(
                runtime.from_numpy(np.array([1.0, 2.0], dtype=np.float64))
            )
            complex_queries = stack.enter_context(
                runtime.from_numpy(
                    np.array([1.0 + 0.0j], dtype=np.complex128)
                )
            )
            bins = stack.enter_context(
                runtime.from_numpy(np.array([1.0, 2.0], dtype=np.float64))
            )
            scalar_bins = stack.enter_context(
                runtime.from_numpy(np.array(1.0, dtype=np.float64))
            )
            matrix_bins = stack.enter_context(
                runtime.from_numpy(
                    np.array([[1.0, 2.0]], dtype=np.float64)
                )
            )
            nonmonotonic_bins = stack.enter_context(
                runtime.from_numpy(
                    np.array([1.0, 3.0, 2.0], dtype=np.float64)
                )
            )
            complex_bins = stack.enter_context(
                runtime.from_numpy(
                    np.array([1.0 + 0.0j, 2.0 + 0.0j])
                )
            )

            invalid_cases = (
                (queries, scalar_bins, -4, "one-dimensional"),
                (queries, matrix_bins, -4, "one-dimensional"),
                (queries, nonmonotonic_bins, -13, "monotonically"),
                (complex_queries, bins, -3, "x may not be complex"),
                (queries, complex_bins, -3, "complex"),
            )
            for invalid_queries, invalid_bins, status, pattern in invalid_cases:
                with self.subTest(status=status, pattern=pattern):
                    with self.assertRaises(CnumpyError) as raised:
                        runtime.digitize(invalid_queries, invalid_bins)
                    self.assertEqual(status, raised.exception.status)
                    self.assertEqual(
                        "cnp_digitize", raised.exception.function
                    )
                    self.assertRegex(raised.exception.message, pattern)

            function = runtime.dll.cnp_digitize
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_bool,
            ]
            function.restype = ctypes.c_void_p
            runtime.dll.cnp_clear_error()
            pointer = function(None, bins.pointer, False)
            self.assertFalse(pointer)
            error = runtime.error_state()
            self.assertEqual(-1, error.status)
            self.assertEqual("cnp_digitize", error.function)
            self.assertRegex(error.message, "x.*NULL")


class LexsortSemanticsTests(unittest.TestCase):
    def slice_view(
        self,
        runtime: CnumpyRuntime,
        source,
        slices: tuple[_CnpSlice, ...],
    ):
        function = runtime.dll.cnp_array_slice
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(_CnpSlice),
        ]
        function.restype = ctypes.c_void_p
        storage = (_CnpSlice * len(slices))(*slices)
        runtime.dll.cnp_clear_error()
        pointer = function(source.pointer, len(slices), storage)
        return runtime._owned_result(pointer, "cnp_array_slice")

    def legacy_lexsort_result(self, runtime: CnumpyRuntime, keys):
        storage = (ctypes.c_void_p * len(keys))(
            *(key.pointer for key in keys)
        )
        function = runtime.dll.cnp_lexsort
        function.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        pointer = function(len(keys), storage)
        return runtime._owned_result(pointer, "cnp_lexsort")

    def test_ctypes_adapter_exposes_basic_stable_lexsort_contract(self) -> None:
        primary_value = np.array([1, 1, 0, 1, 0], dtype=np.int64)
        secondary_value = np.array([2, 1, 2, 1, 2], dtype=np.int32)
        expected = np.lexsort((secondary_value, primary_value))

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            primary = stack.enter_context(runtime.from_numpy(primary_value))
            secondary = stack.enter_context(runtime.from_numpy(secondary_value))
            if not hasattr(runtime, "lexsort"):
                self.fail("ctypes adapter method is missing: lexsort")
            result = stack.enter_context(
                runtime.lexsort((secondary, primary), axis=-1)
            )
            np.testing.assert_array_equal(
                result.to_numpy(), expected, strict=True
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_lexsort_matches_numpy_for_axes_scalars_and_empty_shapes(
        self,
    ) -> None:
        cases = (
            (
                (
                    np.array([[2, 1, 2], [0, 2, 1]], dtype=np.int16),
                    np.array([[1, 1, 0], [1, 0, 0]], dtype=np.int64),
                ),
                -1,
            ),
            (
                (
                    np.array([[2, 1, 2], [0, 2, 1]], dtype=np.uint32),
                    np.array([[1, 1, 0], [1, 0, 0]], dtype=np.float32),
                ),
                0,
            ),
            ((np.array(3, dtype=np.int8),), -1),
            ((np.empty((2, 0), dtype=np.float64),), -1),
            ((np.empty((0, 2), dtype=np.uint64),), 0),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for values, axis in cases:
                with self.subTest(
                    shapes=tuple(value.shape for value in values), axis=axis
                ):
                    keys = tuple(
                        stack.enter_context(runtime.from_numpy(value))
                        for value in values
                    )
                    before = tuple(key.to_numpy().copy() for key in keys)
                    result = stack.enter_context(runtime.lexsort(keys, axis))
                    np.testing.assert_array_equal(
                        result.to_numpy(), np.lexsort(values, axis=axis),
                        strict=True,
                    )
                    for key, original in zip(keys, before, strict=True):
                        np.testing.assert_array_equal(
                            key.to_numpy(), original, strict=True
                        )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_lexsort_preserves_integer_precision_and_numpy_total_order(
        self,
    ) -> None:
        cases = (
            (
                np.array(
                    [2**53 + 3, 2**53 + 1, 2**53 + 2, 2**53 + 1],
                    dtype=np.int64,
                ),
                np.array([1, 1, 1, 1], dtype=np.int8),
            ),
            (
                np.array(
                    [
                        np.iinfo(np.uint64).max,
                        2**63 + 1,
                        2**63 + 3,
                        2**63 + 2,
                    ],
                    dtype=np.uint64,
                ),
                np.array([0, 0, 0, 0], dtype=np.uint8),
            ),
            (
                np.array(
                    [np.nan, -0.0, 0.0, -np.inf, np.inf, np.nan],
                    dtype=np.float64,
                ),
                np.array([1, 1, 1, 1, 1, 1], dtype=np.float16),
            ),
            (
                np.array(
                    [
                        complex(np.nan, 1.0),
                        complex(1.0, np.nan),
                        1.0 + 2.0j,
                        1.0 - 2.0j,
                        -1.0 + 9.0j,
                    ],
                    dtype=np.complex128,
                ),
                np.array([0, 0, 0, 0, 0], dtype=np.int32),
            ),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for secondary_value, primary_value in cases:
                with self.subTest(dtype=str(secondary_value.dtype)):
                    secondary = stack.enter_context(
                        runtime.from_numpy(secondary_value)
                    )
                    primary = stack.enter_context(
                        runtime.from_numpy(primary_value)
                    )
                    result = stack.enter_context(
                        runtime.lexsort((secondary, primary))
                    )
                    np.testing.assert_array_equal(
                        result.to_numpy(),
                        np.lexsort((secondary_value, primary_value)),
                        strict=True,
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_lexsort_handles_transposed_and_negative_stride_keys(self) -> None:
        first_storage_value = np.array(
            [[9, 2, 7, 1], [4, 8, 3, 6], [5, 0, 5, 0]],
            dtype=np.int64,
        )
        second_storage_value = np.array(
            [[2.0, 1.0, 2.0, 1.0], [0.0, 0.0, 1.0, 1.0], [3, 2, 3, 2]],
            dtype=np.float64,
        )
        expected_keys = (
            first_storage_value[:, ::-1].T,
            second_storage_value[:, ::-1].T,
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            keys = []
            for storage_value in (
                first_storage_value, second_storage_value
            ):
                storage = stack.enter_context(
                    runtime.from_numpy(storage_value)
                )
                reversed_key = stack.enter_context(
                    self.slice_view(
                        runtime,
                        storage,
                        (
                            _CnpSlice(0, 0, 0, False, False, False),
                            _CnpSlice(0, 0, -1, False, False, True),
                        ),
                    )
                )
                keys.append(
                    stack.enter_context(runtime.transpose(reversed_key, (1, 0)))
                )
            for axis in (0, -1):
                with self.subTest(axis=axis):
                    result = stack.enter_context(runtime.lexsort(keys, axis))
                    np.testing.assert_array_equal(
                        result.to_numpy(),
                        np.lexsort(expected_keys, axis=axis),
                        strict=True,
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_lexsort_result_remains_owned_after_keys_are_released(self) -> None:
        values = (
            np.array([3, 1, 2, 1], dtype=np.int64),
            np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            keys = tuple(
                runtime.from_numpy(value) for value in values
            )
            result = None
            try:
                result = runtime.lexsort(keys)
                for key in keys:
                    key.close()
                np.testing.assert_array_equal(
                    result.to_numpy(), np.lexsort(values), strict=True
                )
            finally:
                if result is not None and not result._closed:
                    result.close()
                for key in keys:
                    if not key._closed:
                        key.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_lexsort_invalid_inputs_surface_precise_native_errors(self) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            first = stack.enter_context(
                runtime.from_numpy(np.array([2, 1], dtype=np.int64))
            )
            mismatched = stack.enter_context(
                runtime.from_numpy(np.array([2], dtype=np.int64))
            )
            matrix = stack.enter_context(
                runtime.from_numpy(
                    np.array([[2, 1], [0, 3]], dtype=np.int64)
                )
            )
            invalid_cases = (
                ((), -1, -13, "at least one"),
                ((first, mismatched), -1, -4, "same shape"),
                ((matrix,), 2, -5, "axis"),
            )
            for keys, axis, status, pattern in invalid_cases:
                with self.subTest(axis=axis, status=status):
                    with self.assertRaises(CnumpyError) as caught:
                        runtime.lexsort(keys, axis)
                    self.assertEqual(status, caught.exception.status)
                    self.assertEqual(
                        "cnp_lexsort_v2", caught.exception.function
                    )
                    self.assertRegex(
                        caught.exception.message.lower(), pattern
                    )

            unexpected = None
            try:
                with self.assertRaises(TypeError):
                    unexpected = runtime.lexsort((first,), 0.5)
            finally:
                if unexpected is not None:
                    unexpected.close()
            unexpected = None
            try:
                with self.assertRaises(OverflowError):
                    unexpected = runtime.lexsort((first,), 2**40)
            finally:
                if unexpected is not None:
                    unexpected.close()

            function = runtime.dll.cnp_lexsort_v2
            function.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_int,
            ]
            function.restype = ctypes.c_void_p
            for nkeys, key_storage in (
                (1, None),
                (-1, None),
                (1, (ctypes.c_void_p * 1)(None)),
            ):
                runtime.dll.cnp_clear_error()
                pointer = function(nkeys, key_storage, -1)
                self.assertFalse(pointer)
                error = runtime.error_state()
                self.assertEqual("cnp_lexsort_v2", error.function)
                self.assertTrue(error.message)
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_lexsort_preserves_default_last_axis_contract(self) -> None:
        values = (
            np.array([[2, 1, 2], [0, 2, 1]], dtype=np.int64),
            np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            keys = tuple(
                stack.enter_context(runtime.from_numpy(value))
                for value in values
            )
            result = stack.enter_context(
                self.legacy_lexsort_result(runtime, keys)
            )
            np.testing.assert_array_equal(
                result.to_numpy(), np.lexsort(values), strict=True
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)


class MsortAndSortComplexSemanticsTests(unittest.TestCase):
    def required_result(
        self,
        runtime: CnumpyRuntime,
        method_name: str,
        source,
    ):
        try:
            return getattr(runtime, method_name)(source)
        except CnumpyError as error:
            self.fail(
                f"{method_name} raised an unexpected native error: {error}"
            )

    def native_array(
        self,
        runtime: CnumpyRuntime,
        values,
        shape: tuple[int, ...],
        dtype_number: int,
        ctype,
    ):
        storage = (ctype * max(1, len(values)))(
            *(values if values else (0,))
        )
        shape_storage = (ctypes.c_int64 * max(1, len(shape)))(
            *(shape if shape else (0,))
        )
        runtime.dll.cnp_clear_error()
        pointer = runtime.dll.cnp_array_from_data(
            ctypes.cast(storage, ctypes.c_void_p),
            len(shape),
            shape_storage if shape else None,
            dtype_number,
            0,
        )
        return runtime._owned_result(pointer, "cnp_array_from_data")

    def slice_view(
        self,
        runtime: CnumpyRuntime,
        source,
        slices: tuple[_CnpSlice, ...],
    ):
        function = runtime.dll.cnp_array_slice
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(_CnpSlice),
        ]
        function.restype = ctypes.c_void_p
        storage = (_CnpSlice * len(slices))(*slices)
        runtime.dll.cnp_clear_error()
        pointer = function(source.pointer, len(slices), storage)
        return runtime._owned_result(pointer, "cnp_array_slice")

    def test_ctypes_adapter_exposes_both_operations(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            for method_name in ("msort", "sort_complex"):
                with self.subTest(method=method_name):
                    if not hasattr(runtime, method_name):
                        self.fail(
                            f"ctypes adapter method is missing: {method_name}"
                        )

    def test_large_contiguous_float64_quicksort_matches_numpy_edges(
        self,
    ) -> None:
        negative_nan = np.array(
            [0xFFF8000000000001], dtype=np.uint64
        ).view(np.float64)[0]
        positive_nan = np.array(
            [0x7FF8000000000002], dtype=np.uint64
        ).view(np.float64)[0]
        pattern = np.array(
            [
                negative_nan,
                -0.0,
                3.0,
                0.0,
                positive_nan,
                -2.0,
                -0.0,
                0.0,
                np.inf,
                -np.inf,
                3.0,
                1.0,
            ],
            dtype=np.float64,
        )
        values = np.resize(pattern, 10007)
        expected_msort = np.sort(values, axis=0, kind="quicksort")
        expected_complex = np.sort_complex(values)

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(values))
            msort_result = stack.enter_context(runtime.msort(source))
            complex_result = stack.enter_context(
                runtime.sort_complex(source)
            )

            assert_array_equivalent(self, msort_result, expected_msort)
            np.testing.assert_array_equal(
                complex_result.to_numpy(), expected_complex, strict=True
            )
            np.testing.assert_array_equal(
                source.to_numpy(), values, strict=True
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_msort_matches_axis_zero_for_dtypes_shapes_and_layouts(
        self,
    ) -> None:
        cases = (
            np.array([[3, 1, 2], [0, 4, -1]], dtype=np.int64),
            np.array(
                [[np.nan, -0.0, np.inf], [-np.inf, 0.0, np.nan]],
                dtype=np.float64,
            ),
            np.array(
                [[3 + 1j, 1 + np.nan * 1j], [1 - 2j, 1 + 2j]],
                dtype=np.complex128,
            ),
            np.empty((0, 2), dtype=np.float32),
            np.empty((2, 0), dtype=np.uint16),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for value in cases:
                with self.subTest(dtype=str(value.dtype), shape=value.shape):
                    source = stack.enter_context(runtime.from_numpy(value))
                    before = source.to_numpy().copy()
                    result = stack.enter_context(
                        self.required_result(runtime, "msort", source)
                    )
                    assert_array_equivalent(
                        self, result, np.sort(value, axis=0)
                    )
                    np.testing.assert_array_equal(
                        source.to_numpy(), before, strict=True
                    )

            storage_value = np.array(
                [[3.0, 0.0, 1.0], [1.0, 2.0, -1.0]],
                dtype=np.float64,
            )
            storage = stack.enter_context(
                runtime.from_numpy(storage_value)
            )
            fortran_source = stack.enter_context(
                runtime.transpose(storage, (1, 0))
            )
            fortran_result = stack.enter_context(
                self.required_result(runtime, "msort", fortran_source)
            )
            assert_array_equivalent(
                self,
                fortran_result,
                np.sort(storage_value.T, axis=0),
            )
            self.assertTrue(fortran_result.f_contiguous)
            self.assertFalse(fortran_result.c_contiguous)

            datetime_source = stack.enter_context(
                self.native_array(
                    runtime, (3, 1, 2, -4), (2, 2), 22,
                    ctypes.c_int64,
                )
            )
            datetime_result = stack.enter_context(
                self.required_result(runtime, "msort", datetime_source)
            )
            self.assertEqual(22, datetime_result.dtype_number)
            np.testing.assert_array_equal(
                datetime_result.to_numpy(),
                np.sort(
                    np.array([[3, 1], [2, -4]], dtype=np.int64),
                    axis=0,
                ),
                strict=True,
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_sort_complex_matches_numpy_dtype_promotion_and_last_axis(
        self,
    ) -> None:
        cases = (
            np.array([True, False], dtype=np.bool_),
            np.array([3, 1, 2], dtype=np.int8),
            np.array([3, 1, 2], dtype=np.uint8),
            np.array([3, 1, 2], dtype=np.int16),
            np.array([3, 1, 2], dtype=np.uint16),
            np.array([3, 1, 2], dtype=np.int32),
            np.array([3, 1, 2], dtype=np.uint32),
            np.array(
                [2**53 + 3, 2**53 + 1, 2**53 + 2],
                dtype=np.int64,
            ),
            np.array(
                [np.iinfo(np.uint64).max, 2**63 + 1, 2**63 + 3],
                dtype=np.uint64,
            ),
            np.array([3, 1, 2], dtype=np.float16),
            np.array([3, 1, 2], dtype=np.float32),
            np.array([3, 1, 2], dtype=np.float64),
            np.array(
                [[3 + 1j, 1 + 9j], [2 + 0j, 1 - 9j]],
                dtype=np.complex64,
            ),
            np.array(
                [[3 + 1j, 1 + 9j], [2 + 0j, 1 - 9j]],
                dtype=np.complex128,
            ),
            np.empty((2, 0), dtype=np.float32),
            np.empty((0, 2), dtype=np.complex128),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for value in cases:
                with self.subTest(dtype=str(value.dtype), shape=value.shape):
                    source = stack.enter_context(runtime.from_numpy(value))
                    before = source.to_numpy().copy()
                    result = stack.enter_context(
                        self.required_result(
                            runtime, "sort_complex", source
                        )
                    )
                    np.testing.assert_array_equal(
                        result.to_numpy(), np.sort_complex(value), strict=True
                    )
                    np.testing.assert_array_equal(
                        source.to_numpy(), before, strict=True
                    )

            for dtype_number in (22, 23):
                source = stack.enter_context(
                    self.native_array(
                        runtime, (3, 1, 2), (3,), dtype_number,
                        ctypes.c_int64,
                    )
                )
                result = stack.enter_context(
                    self.required_result(runtime, "sort_complex", source)
                )
                self.assertEqual(16, result.dtype_number)
                np.testing.assert_array_equal(
                    result.to_numpy(),
                    np.array([1, 2, 3], dtype=np.complex128),
                    strict=True,
                )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_sort_complex_matches_numpy_total_order_and_native_views(
        self,
    ) -> None:
        edge_values = np.array(
            [
                complex(np.nan, 1.0),
                complex(1.0, np.nan),
                complex(np.nan, np.nan),
                complex(-0.0, 0.0),
                complex(0.0, -0.0),
                complex(-np.inf, 2.0),
                complex(np.inf, -3.0),
                1.0 - 2.0j,
                1.0 + 2.0j,
            ],
            dtype=np.complex128,
        )
        storage_value = np.array(
            [[3.0, 1.0, 2.0], [0.0, 4.0, -1.0]],
            dtype=np.float32,
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            edge_source = stack.enter_context(runtime.from_numpy(edge_values))
            edge_result = stack.enter_context(
                self.required_result(runtime, "sort_complex", edge_source)
            )
            assert_array_equivalent(
                self, edge_result, np.sort_complex(edge_values)
            )

            storage = stack.enter_context(runtime.from_numpy(storage_value))
            reversed_source = stack.enter_context(
                self.slice_view(
                    runtime,
                    storage,
                    (
                        _CnpSlice(0, 0, 0, False, False, False),
                        _CnpSlice(0, 0, -1, False, False, True),
                    ),
                )
            )
            reversed_result = stack.enter_context(
                self.required_result(
                    runtime, "sort_complex", reversed_source
                )
            )
            np.testing.assert_array_equal(
                reversed_result.to_numpy(),
                np.sort_complex(storage_value[:, ::-1]),
                strict=True,
            )

            fortran_source = stack.enter_context(
                runtime.transpose(storage, (1, 0))
            )
            fortran_result = stack.enter_context(
                self.required_result(
                    runtime, "sort_complex", fortran_source
                )
            )
            np.testing.assert_array_equal(
                fortran_result.to_numpy(),
                np.sort_complex(storage_value.T),
                strict=True,
            )
            self.assertTrue(fortran_result.f_contiguous)
            self.assertFalse(fortran_result.c_contiguous)
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_results_outlive_sources_and_errors_are_explicit(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            values = np.array([[3, 1], [2, 0]], dtype=np.int64)
            source = runtime.from_numpy(values)
            msort_result = runtime.msort(source)
            complex_result = runtime.sort_complex(source)
            source.close()
            try:
                np.testing.assert_array_equal(
                    msort_result.to_numpy(), np.sort(values, axis=0),
                    strict=True,
                )
                np.testing.assert_array_equal(
                    complex_result.to_numpy(), np.sort_complex(values),
                    strict=True,
                )
            finally:
                msort_result.close()
                complex_result.close()

            scalar = runtime.from_numpy(np.array(3, dtype=np.int64))
            try:
                for method_name, function_name in (
                    ("msort", "cnp_msort"),
                    ("sort_complex", "cnp_sort_complex"),
                ):
                    with self.subTest(method=method_name):
                        with self.assertRaises(CnumpyError) as caught:
                            getattr(runtime, method_name)(scalar)
                        self.assertEqual(-5, caught.exception.status)
                        self.assertEqual(
                            function_name, caught.exception.function
                        )
                        self.assertRegex(
                            caught.exception.message.lower(), "axis"
                        )
            finally:
                scalar.close()

            for function_name in ("cnp_msort", "cnp_sort_complex"):
                function = getattr(runtime.dll, function_name)
                function.argtypes = [ctypes.c_void_p]
                function.restype = ctypes.c_void_p
                runtime.dll.cnp_clear_error()
                self.assertFalse(function(None))
                error = runtime.error_state()
                self.assertEqual(function_name, error.function)
                self.assertRegex(error.message.lower(), "must not be null")
            self.assertEqual(baseline, runtime.retained_bytes)


class PartitionSemanticsTests(unittest.TestCase):
    def legacy_partition_result(
        self,
        runtime: CnumpyRuntime,
        source,
        function_name: str,
        kth: int,
        axis: int,
    ):
        function = getattr(runtime.dll, function_name)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int,
        ]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        pointer = function(source.pointer, kth, axis)
        return runtime._owned_result(pointer, function_name)

    def partition_result(
        self,
        runtime: CnumpyRuntime,
        source,
        function_name: str,
        kth: tuple[int, ...],
        axis: int | None,
    ):
        method_name = function_name.removeprefix("cnp_")
        if not hasattr(runtime, method_name):
            self.fail(f"ctypes adapter method is missing: {method_name}")
        return getattr(runtime, method_name)(source, kth, axis)

    def assert_partition_invariants(
        self,
        actual: np.ndarray,
        source: np.ndarray,
        kth: tuple[int, ...],
        axis: int | None,
    ) -> None:
        prepared = (
            np.ravel(source, order="C") if axis is None else source
        )
        resolved_axis = 0 if axis is None else axis % prepared.ndim
        self.assertEqual(prepared.shape, actual.shape)
        self.assertEqual(prepared.dtype, actual.dtype)
        np.testing.assert_array_equal(
            np.sort(actual, axis=resolved_axis, kind="stable"),
            np.sort(prepared, axis=resolved_axis, kind="stable"),
            strict=True,
        )
        if prepared.size == 0:
            return
        axis_size = prepared.shape[resolved_axis]
        expected_sorted = np.sort(
            prepared, axis=resolved_axis, kind="stable"
        )
        for raw_kth in kth:
            normalized_kth = raw_kth % axis_size
            np.testing.assert_array_equal(
                np.take(actual, normalized_kth, axis=resolved_axis),
                np.take(
                    expected_sorted, normalized_kth, axis=resolved_axis
                ),
                strict=True,
            )

    def assert_argpartition_invariants(
        self,
        indices: np.ndarray,
        source: np.ndarray,
        kth: tuple[int, ...],
        axis: int | None,
    ) -> None:
        prepared = (
            np.ravel(source, order="C") if axis is None else source
        )
        resolved_axis = 0 if axis is None else axis % prepared.ndim
        self.assertEqual(prepared.shape, indices.shape)
        self.assertEqual(np.dtype(np.int64), indices.dtype)
        axis_size = prepared.shape[resolved_axis]
        moved_indices = np.moveaxis(indices, resolved_axis, -1)
        for permutation in moved_indices.reshape(-1, axis_size):
            np.testing.assert_array_equal(
                np.sort(permutation),
                np.arange(axis_size, dtype=np.int64),
                strict=True,
            )
        partitioned = np.take_along_axis(
            prepared, indices, axis=resolved_axis
        )
        self.assert_partition_invariants(
            partitioned, prepared, kth, resolved_axis
        )

    def test_partition_and_argpartition_cover_axes_kth_and_exact_dtypes(
        self,
    ) -> None:
        cases = (
            (
                np.array([9, 1, 7, 2, 8, 4], dtype=np.int64),
                None,
                (1, -2),
                None,
            ),
            (
                np.array([[9, 1, 7], [2, 8, 4]], dtype=np.int32),
                None,
                (1,),
                0,
            ),
            (
                np.array([[9, 1, 7], [2, 8, 4]], dtype=np.int32),
                None,
                (-1, 0),
                -1,
            ),
            (
                np.array(
                    [
                        2**53 + 3,
                        2**53 + 1,
                        np.iinfo(np.int64).max,
                        np.iinfo(np.int64).min,
                        2**53 + 2,
                    ],
                    dtype=np.int64,
                ),
                None,
                (2,),
                -1,
            ),
            (
                np.array(
                    [
                        np.iinfo(np.uint64).max,
                        2**63 + 3,
                        2**63 + 1,
                        0,
                        2**63 + 2,
                    ],
                    dtype=np.uint64,
                ),
                None,
                (1, 3),
                -1,
            ),
            (
                np.array(
                    [
                        [np.nan, -0.0, np.inf, -3.0],
                        [0.0, -np.inf, np.nan, -0.0],
                        [4.0, 4.0, 1.0, np.nan],
                    ],
                    dtype=np.float64,
                ),
                (1, 0),
                (0, -1),
                -1,
            ),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for base_value, transpose_axes, kth, axis in cases:
                with self.subTest(
                    dtype=str(base_value.dtype),
                    shape=base_value.shape,
                    transpose=transpose_axes,
                    kth=kth,
                    axis=axis,
                ):
                    base = stack.enter_context(
                        runtime.from_numpy(base_value)
                    )
                    source = base
                    logical_value = base_value
                    if transpose_axes is not None:
                        source = stack.enter_context(
                            runtime.transpose(base, transpose_axes)
                        )
                        logical_value = base_value.transpose(transpose_axes)
                    before = source.to_numpy().copy()
                    partitioned = stack.enter_context(
                        self.partition_result(
                            runtime, source, "cnp_partition", kth, axis
                        )
                    )
                    indices = stack.enter_context(
                        self.partition_result(
                            runtime,
                            source,
                            "cnp_argpartition",
                            kth,
                            axis,
                        )
                    )
                    self.assert_partition_invariants(
                        partitioned.to_numpy(), logical_value, kth, axis
                    )
                    self.assert_argpartition_invariants(
                        indices.to_numpy(), logical_value, kth, axis
                    )
                    np.testing.assert_array_equal(
                        source.to_numpy(), before, strict=True
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_empty_typed_kth_and_zero_sized_inputs_match_numpy(self) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source_value = np.array([[9, 1, 7], [2, 8, 4]], dtype=np.int64)
            source = stack.enter_context(runtime.from_numpy(source_value))
            unchanged = stack.enter_context(
                self.partition_result(
                    runtime, source, "cnp_partition", (), -1
                )
            )
            identity = stack.enter_context(
                self.partition_result(
                    runtime, source, "cnp_argpartition", (), -1
                )
            )
            np.testing.assert_array_equal(
                unchanged.to_numpy(),
                np.partition(
                    source_value,
                    np.array([], dtype=np.int64),
                    axis=-1,
                ),
                strict=True,
            )
            np.testing.assert_array_equal(
                identity.to_numpy(),
                np.argpartition(
                    source_value,
                    np.array([], dtype=np.int64),
                    axis=-1,
                ),
                strict=True,
            )

            for empty_value, axis in (
                (np.empty((0,), dtype=np.float64), -1),
                (np.empty((2, 0), dtype=np.int32), -1),
                (np.empty((0, 2), dtype=np.uint64), 0),
                (np.empty((0, 2), dtype=np.int64), None),
            ):
                empty = stack.enter_context(runtime.from_numpy(empty_value))
                for function_name, numpy_function in (
                    ("cnp_partition", np.partition),
                    ("cnp_argpartition", np.argpartition),
                ):
                    actual = stack.enter_context(
                        self.partition_result(
                            runtime,
                            empty,
                            function_name,
                            (999, -999),
                            axis,
                        )
                    )
                    expected = numpy_function(
                        empty_value, [999, -999], axis=axis
                    )
                    np.testing.assert_array_equal(
                        actual.to_numpy(), expected, strict=True
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_rank_zero_partition_and_argpartition_match_numpy_asymmetry(
        self,
    ) -> None:
        scalar_value = np.array(3, dtype=np.int64)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            scalar = stack.enter_context(runtime.from_numpy(scalar_value))
            with self.assertRaises(CnumpyError) as caught:
                self.partition_result(
                    runtime, scalar, "cnp_partition", (0,), -1
                )
            self.assertEqual("cnp_partition_v2", caught.exception.function)
            self.assertIn("axis", caught.exception.message.lower())

            for axis in (-1, 0, None):
                indices = stack.enter_context(
                    self.partition_result(
                        runtime,
                        scalar,
                        "cnp_argpartition",
                        (0,),
                        axis,
                    )
                )
                np.testing.assert_array_equal(
                    indices.to_numpy(),
                    np.argpartition(scalar_value, 0, axis=axis),
                    strict=True,
                )
            flattened = stack.enter_context(
                self.partition_result(
                    runtime, scalar, "cnp_partition", (-1,), None
                )
            )
            np.testing.assert_array_equal(
                flattened.to_numpy(),
                np.partition(scalar_value, -1, axis=None),
                strict=True,
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_partition_v2_invalid_inputs_surface_precise_errors(self) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(
                runtime.from_numpy(np.array([3, 1, 2], dtype=np.int64))
            )
            for function_name in ("cnp_partition", "cnp_argpartition"):
                for kth, axis in (((3,), -1), ((-4,), -1), ((0,), 2)):
                    with self.subTest(
                        function=function_name, kth=kth, axis=axis
                    ):
                        with self.assertRaises(CnumpyError) as caught:
                            self.partition_result(
                                runtime, source, function_name, kth, axis
                            )
                        self.assertEqual(
                            f"{function_name}_v2", caught.exception.function
                        )

                v2_name = f"{function_name}_v2"
                if not hasattr(runtime.dll, v2_name):
                    self.fail(
                        f"required NumPy-compatible export is missing: {v2_name}"
                    )
                function = getattr(runtime.dll, v2_name)
                function.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_bool,
                ]
                function.restype = ctypes.c_void_p
                one_kth = (ctypes.c_int64 * 1)(0)
                for arguments in (
                    (None, one_kth, 1, -1, False),
                    (source.pointer, None, 1, -1, False),
                    (source.pointer, None, -1, -1, False),
                ):
                    runtime.dll.cnp_clear_error()
                    pointer = function(*arguments)
                    self.assertFalse(pointer)
                    error = runtime.native_error(v2_name)
                    self.assertEqual(v2_name, error.function)
                    self.assertTrue(error.message)
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_partition_abis_preserve_axis_minus_one_flatten_sentinel(
        self,
    ) -> None:
        source_value = np.array([[9, 1, 7], [2, 8, 4]], dtype=np.int64)
        kth = 2
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            partitioned = stack.enter_context(
                self.legacy_partition_result(
                    runtime, source, "cnp_partition", kth, CNP_AXIS_NONE
                )
            )
            indices = stack.enter_context(
                self.legacy_partition_result(
                    runtime,
                    source,
                    "cnp_argpartition",
                    kth,
                    CNP_AXIS_NONE,
                )
            )
            self.assert_partition_invariants(
                partitioned.to_numpy(), source_value, (kth,), None
            )
            self.assert_argpartition_invariants(
                indices.to_numpy(), source_value, (kth,), None
            )

            axis_partitioned = stack.enter_context(
                self.legacy_partition_result(
                    runtime, source, "cnp_partition", 0, 0
                )
            )
            axis_indices = stack.enter_context(
                self.legacy_partition_result(
                    runtime, source, "cnp_argpartition", 0, 0
                )
            )
            self.assert_partition_invariants(
                axis_partitioned.to_numpy(), source_value, (0,), 0
            )
            self.assert_argpartition_invariants(
                axis_indices.to_numpy(), source_value, (0,), 0
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
