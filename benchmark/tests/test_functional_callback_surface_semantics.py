from __future__ import annotations

import ctypes
from contextlib import ExitStack
from pathlib import Path
import unittest

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime
from compat.oracle import assert_array_equivalent


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"

CNP_BOOL = 1
CNP_SHORT = 4
CNP_DOUBLE = 13
CNP_ERR_GENERIC = -1
CNP_ERR_TYPE = -3
CNP_ERR_SHAPE = -4
CNP_ERR_AXIS = -5
CNP_ERR_INDEX = -6
CNP_ERR_BROADCAST = -7
CNP_ERR_VALUE = -13

LINE_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int64,
    ctypes.c_void_p,
)
COORD_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_int,
    ctypes.c_void_p,
)
ITER_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_void_p)
UNARY_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_double, ctypes.c_double, ctypes.c_void_p
)
BULK_LINE_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_void_p,
)
BULK_COORD_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_int64,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_void_p,
)
BULK_ITER_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_void_p,
)
BULK_UNARY_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_void_p,
)
ARRAY_POINTER = ctypes.POINTER(ctypes.c_void_p)


class FunctionalCallbackSurfaceSemanticsTests(unittest.TestCase):
    @staticmethod
    def _bind(runtime: CnumpyRuntime) -> None:
        dll = runtime.dll
        dll.cnp_apply_along_axis.argtypes = [
            LINE_CALLBACK,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        dll.cnp_apply_along_axis.restype = ctypes.c_void_p
        dll.cnp_apply_over_axes.argtypes = [
            LINE_CALLBACK,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        dll.cnp_apply_over_axes.restype = ctypes.c_void_p
        dll.cnp_fromfunction.argtypes = [
            COORD_CALLBACK,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_void_p,
        ]
        dll.cnp_fromfunction.restype = ctypes.c_void_p
        dll.cnp_fromiter.argtypes = [
            ITER_CALLBACK,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int,
        ]
        dll.cnp_fromiter.restype = ctypes.c_void_p
        for symbol in ("cnp_frompyfunc", "cnp_vectorize"):
            function = getattr(dll, symbol)
            function.argtypes = [
                UNARY_CALLBACK,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            function.restype = ctypes.c_void_p
        dll.cnp_select.argtypes = [
            ctypes.c_int,
            ARRAY_POINTER,
            ARRAY_POINTER,
            ctypes.c_double,
        ]
        dll.cnp_select.restype = ctypes.c_void_p
        dll.cnp_piecewise.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ARRAY_POINTER,
            UNARY_CALLBACK,
            ctypes.c_void_p,
        ]
        dll.cnp_piecewise.restype = ctypes.c_void_p
        dll.cnp_put_along_axis.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        dll.cnp_put_along_axis.restype = ctypes.c_int

    @staticmethod
    def _bind_bulk(runtime: CnumpyRuntime) -> None:
        dll = runtime.dll
        dll.cnp_ahk_apply_along_axis_v2.argtypes = [
            BULK_LINE_CALLBACK,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
        ]
        dll.cnp_ahk_apply_along_axis_v2.restype = ctypes.c_void_p
        dll.cnp_ahk_apply_over_axes_v2.argtypes = [
            BULK_LINE_CALLBACK,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_void_p,
        ]
        dll.cnp_ahk_apply_over_axes_v2.restype = ctypes.c_void_p
        dll.cnp_ahk_fromfunction_v2.argtypes = [
            BULK_COORD_CALLBACK,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
        ]
        dll.cnp_ahk_fromfunction_v2.restype = ctypes.c_void_p
        dll.cnp_ahk_fromiter_v2.argtypes = [
            BULK_ITER_CALLBACK,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int,
        ]
        dll.cnp_ahk_fromiter_v2.restype = ctypes.c_void_p
        for symbol in ("cnp_ahk_frompyfunc_v2", "cnp_ahk_vectorize_v2"):
            function = getattr(dll, symbol)
            function.argtypes = [
                BULK_UNARY_CALLBACK,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            function.restype = ctypes.c_void_p
        dll.cnp_ahk_piecewise_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ARRAY_POINTER,
            BULK_UNARY_CALLBACK,
            ctypes.c_void_p,
        ]
        dll.cnp_ahk_piecewise_v2.restype = ctypes.c_void_p

    def _assert_null_error(
        self,
        runtime: CnumpyRuntime,
        pointer: int | None,
        function_name: str,
        statuses: tuple[int, ...],
    ) -> None:
        if pointer:
            runtime._owned_result(pointer, function_name).close()
        self.assertFalse(pointer)
        error = runtime.error_state()
        self.assertIn(error.status, statuses)
        self.assertEqual(function_name, error.function)
        self.assertTrue(error.message)

    def test_bulk_apply_v2_preserves_numpy_result_shape_and_line_order(
        self,
    ) -> None:
        values = np.arange(2 * 3 * 4, dtype=np.int32).reshape(2, 3, 4)
        expected = np.apply_along_axis(
            lambda line: np.asarray([line.sum(), line[0] - line[-1]]),
            1,
            values,
        ).astype(np.float64)
        observed_lines: list[tuple[float, ...]] = []
        batch_sizes: list[int] = []
        capacities: list[int] = []

        @BULK_LINE_CALLBACK
        def callback(
            lines: ctypes.POINTER(ctypes.c_double),
            line_count: int,
            line_length: int,
            results: ctypes.POINTER(ctypes.c_double),
            result_capacity: int,
            produced: ctypes.POINTER(ctypes.c_int64),
            _state: int,
        ) -> int:
            batch_sizes.append(line_count)
            capacities.append(result_capacity)
            for line_index in range(line_count):
                line = tuple(
                    lines[line_index * line_length + item]
                    for item in range(line_length)
                )
                observed_lines.append(line)
                results[line_index * 2] = sum(line)
                results[line_index * 2 + 1] = line[0] - line[-1]
            produced[0] = result_capacity
            return 0

        with CnumpyRuntime(DLL) as runtime:
            self._bind_bulk(runtime)
            baseline = runtime.retained_bytes
            with runtime.from_numpy(values) as source:
                result_shape = (ctypes.c_int64 * 1)(2)
                pointer = runtime.dll.cnp_ahk_apply_along_axis_v2(
                    callback,
                    None,
                    1,
                    source.pointer,
                    1,
                    result_shape,
                )
                self.assertTrue(pointer, runtime.error_state())
                with runtime._owned_result(
                    pointer, "cnp_ahk_apply_along_axis_v2"
                ) as actual:
                    assert_array_equivalent(self, actual, expected)
            self.assertEqual(baseline, runtime.retained_bytes)

        expected_lines = [
            tuple(values[first, :, last].astype(np.float64))
            for first in range(values.shape[0])
            for last in range(values.shape[2])
        ]
        self.assertEqual(expected_lines, observed_lines)
        self.assertEqual([8], batch_sizes)
        self.assertEqual([16], capacities)

    def test_bulk_coordinate_and_unary_v2_batch_counts_are_not_element_counts(
        self,
    ) -> None:
        coordinate_batches: list[int] = []
        coordinate_capacities: list[int] = []
        coordinate_elements = 0

        @BULK_COORD_CALLBACK
        def coordinates_callback(
            coordinates: ctypes.POINTER(ctypes.c_int64),
            point_count: int,
            ndim: int,
            results: ctypes.POINTER(ctypes.c_double),
            result_capacity: int,
            produced: ctypes.POINTER(ctypes.c_int64),
            _state: int,
        ) -> int:
            nonlocal coordinate_elements
            coordinate_batches.append(point_count)
            coordinate_capacities.append(result_capacity)
            coordinate_elements += point_count
            for point in range(point_count):
                row = coordinates[point * ndim]
                column = coordinates[point * ndim + 1]
                results[point] = row * 10.0 + column + 0.5
            produced[0] = point_count
            return 0

        unary_batches: list[int] = []
        unary_capacities: list[int] = []
        unary_elements = 0

        @BULK_UNARY_CALLBACK
        def unary_callback(
            inputs: ctypes.POINTER(ctypes.c_double),
            value_count: int,
            results: ctypes.POINTER(ctypes.c_double),
            result_capacity: int,
            produced: ctypes.POINTER(ctypes.c_int64),
            _state: int,
        ) -> int:
            nonlocal unary_elements
            unary_batches.append(value_count)
            unary_capacities.append(result_capacity)
            unary_elements += value_count
            for index in range(value_count):
                results[index] = inputs[index] * 1.5 + 0.25
            produced[0] = value_count
            return 0

        with CnumpyRuntime(DLL) as runtime:
            self._bind_bulk(runtime)
            baseline = runtime.retained_bytes
            shape_tuple = (20, 30)
            shape = (ctypes.c_int64 * 2)(*shape_tuple)
            pointer = runtime.dll.cnp_ahk_fromfunction_v2(
                coordinates_callback, None, 2, shape
            )
            with runtime._owned_result(
                pointer, "cnp_ahk_fromfunction_v2"
            ) as generated:
                expected = np.fromfunction(
                    lambda row, column: row * 10 + column + 0.5,
                    shape_tuple,
                    dtype=np.int64,
                ).astype(np.float64)
                assert_array_equivalent(self, generated, expected)

            values = np.arange(600, dtype=np.float64).reshape(20, 30).T
            base = runtime.from_numpy(values.T)
            source = runtime.transpose(base, (1, 0))
            base.close()
            pointer = runtime.dll.cnp_ahk_vectorize_v2(
                unary_callback, None, source.pointer
            )
            result = runtime._owned_result(pointer, "cnp_ahk_vectorize_v2")
            source.close()
            with result:
                assert_array_equivalent(
                    self, result, values * 1.5 + 0.25
                )

            empty_calls = len(unary_batches)
            with runtime.from_numpy(np.empty((0, 3), dtype=np.float64)) as empty:
                pointer = runtime.dll.cnp_ahk_frompyfunc_v2(
                    unary_callback, None, empty.pointer
                )
                with runtime._owned_result(
                    pointer, "cnp_ahk_frompyfunc_v2"
                ) as empty_result:
                    assert_array_equivalent(self, empty_result, np.empty((0, 3)))
            self.assertEqual(empty_calls, len(unary_batches))
            self.assertEqual(baseline, runtime.retained_bytes)

        self.assertEqual(600, coordinate_elements)
        self.assertEqual([256, 256, 88], coordinate_batches)
        self.assertEqual(coordinate_batches, coordinate_capacities)
        self.assertEqual(600, unary_elements)
        self.assertEqual([256, 256, 88], unary_batches)
        self.assertEqual(unary_batches, unary_capacities)

    def test_bulk_fromiter_v2_partial_production_signals_exhaustion_only_for_unknown_count(
        self,
    ) -> None:
        values = np.arange(513, dtype=np.float64) * 0.25 - 3.0
        cursor = 0
        capacities: list[int] = []

        @BULK_ITER_CALLBACK
        def iterator_callback(
            results: ctypes.POINTER(ctypes.c_double),
            result_capacity: int,
            produced: ctypes.POINTER(ctypes.c_int64),
            _state: int,
        ) -> int:
            nonlocal cursor
            capacities.append(result_capacity)
            count = min(result_capacity, len(values) - cursor)
            for index in range(count):
                results[index] = values[cursor + index]
            cursor += count
            produced[0] = count
            return 0

        @BULK_ITER_CALLBACK
        def short_fixed_callback(
            results: ctypes.POINTER(ctypes.c_double),
            result_capacity: int,
            produced: ctypes.POINTER(ctypes.c_int64),
            _state: int,
        ) -> int:
            count = max(0, result_capacity - 1)
            for index in range(count):
                results[index] = float(index)
            produced[0] = count
            return 0

        with CnumpyRuntime(DLL) as runtime:
            self._bind_bulk(runtime)
            baseline = runtime.retained_bytes
            pointer = runtime.dll.cnp_ahk_fromiter_v2(
                iterator_callback, None, -1, CNP_DOUBLE
            )
            with runtime._owned_result(
                pointer, "cnp_ahk_fromiter_v2"
            ) as actual:
                assert_array_equivalent(self, actual, values)
            self.assertEqual([256, 256, 256], capacities)

            runtime.dll.cnp_clear_error()
            pointer = runtime.dll.cnp_ahk_fromiter_v2(
                short_fixed_callback, None, 8, CNP_DOUBLE
            )
            self._assert_null_error(
                runtime,
                pointer,
                "cnp_ahk_fromiter_v2",
                (CNP_ERR_VALUE,),
            )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_bulk_callback_failure_is_atomic_and_never_scalar_retried(self) -> None:
        callback_calls = 0

        @BULK_UNARY_CALLBACK
        def failing_callback(
            _inputs: ctypes.POINTER(ctypes.c_double),
            _value_count: int,
            _results: ctypes.POINTER(ctypes.c_double),
            _result_capacity: int,
            produced: ctypes.POINTER(ctypes.c_int64),
            _state: int,
        ) -> int:
            nonlocal callback_calls
            callback_calls += 1
            produced[0] = 0
            return CNP_ERR_VALUE

        with CnumpyRuntime(DLL) as runtime:
            self._bind_bulk(runtime)
            baseline = runtime.retained_bytes
            with runtime.from_numpy(np.arange(600, dtype=np.float64)) as source:
                runtime.dll.cnp_clear_error()
                pointer = runtime.dll.cnp_ahk_vectorize_v2(
                    failing_callback, None, source.pointer
                )
                self._assert_null_error(
                    runtime,
                    pointer,
                    "cnp_ahk_vectorize_v2",
                    (CNP_ERR_VALUE,),
                )
            self.assertEqual(baseline, runtime.retained_bytes)
        self.assertEqual(1, callback_calls)

    def test_apply_axis_scalar_projection_matches_numpy_shape_order_and_lifetime(
        self,
    ) -> None:
        values = np.arange(24, dtype=np.int32).reshape(2, 3, 4).transpose(1, 0, 2)
        expected_calls: list[list[float]] = []

        def oracle(line: np.ndarray) -> float:
            expected_calls.append(np.asarray(line, dtype=np.float64).tolist())
            return float(np.sum(line) + 0.5)

        expected = np.apply_along_axis(oracle, -2, values).astype(np.float64)
        actual_calls: list[list[float]] = []

        @LINE_CALLBACK
        def callback(line: ctypes.POINTER(ctypes.c_double), length: int, _state: int) -> float:
            actual = [line[index] for index in range(length)]
            actual_calls.append(actual)
            return float(sum(actual) + 0.5)

        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            base = runtime.from_numpy(np.arange(24, dtype=np.int32).reshape(2, 3, 4))
            source = runtime.transpose(base, (1, 0, 2))
            base.close()
            pointer = runtime.dll.cnp_apply_along_axis(
                callback, -2, source.pointer, None
            )
            result = runtime._owned_result(pointer, "cnp_apply_along_axis")
            source.close()
            with result:
                assert_array_equivalent(self, result, expected)
                self.assertEqual(expected_calls, actual_calls)

            with runtime.from_numpy(np.asarray([1.0, 2.0, 3.0])) as one_dimensional:
                pointer = runtime.dll.cnp_apply_along_axis(
                    callback, 0, one_dimensional.pointer, None
                )
                with runtime._owned_result(
                    pointer, "cnp_apply_along_axis"
                ) as scalar:
                    assert_array_equivalent(
                        self,
                        scalar,
                        np.asarray(
                            np.apply_along_axis(
                                lambda line: np.sum(line) + 0.5,
                                0,
                                np.asarray([1.0, 2.0, 3.0]),
                            ),
                            dtype=np.float64,
                        ),
                    )

            with runtime.from_numpy(np.empty((0, 3), dtype=np.float64)) as empty:
                runtime.dll.cnp_clear_error()
                pointer = runtime.dll.cnp_apply_along_axis(
                    callback, 1, empty.pointer, None
                )
                self._assert_null_error(
                    runtime,
                    pointer,
                    "cnp_apply_along_axis",
                    (CNP_ERR_VALUE,),
                )

            runtime.dll.cnp_clear_error()
            pointer = runtime.dll.cnp_apply_along_axis(
                callback, 7, None, None
            )
            self._assert_null_error(
                runtime,
                pointer,
                "cnp_apply_along_axis",
                (CNP_ERR_GENERIC, CNP_ERR_AXIS),
            )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_apply_over_axes_preserves_reduced_dimensions_and_axis_sequence(
        self,
    ) -> None:
        values = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        calls: list[tuple[float, ...]] = []

        @LINE_CALLBACK
        def callback(line: ctypes.POINTER(ctypes.c_double), length: int, _state: int) -> float:
            actual = tuple(line[index] for index in range(length))
            calls.append(actual)
            return float(sum(actual))

        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            with runtime.from_numpy(values) as source:
                axes = (ctypes.c_int * 2)(0, 2)
                pointer = runtime.dll.cnp_apply_over_axes(
                    callback, 2, axes, source.pointer, None
                )
                self.assertTrue(pointer, runtime.error_state())
                with runtime._owned_result(
                    pointer, "cnp_apply_over_axes"
                ) as actual:
                    assert_array_equivalent(
                        self,
                        actual,
                        np.apply_over_axes(np.sum, values, [0, 2]).astype(
                            np.float64
                        ),
                    )
                self.assertEqual(15, len(calls))

                calls.clear()
                no_axes = (ctypes.c_int * 1)(0)
                pointer = runtime.dll.cnp_apply_over_axes(
                    callback, 0, no_axes, source.pointer, None
                )
                with runtime._owned_result(
                    pointer, "cnp_apply_over_axes"
                ) as unchanged:
                    assert_array_equivalent(self, unchanged, values)
                self.assertEqual([], calls)

                bad_axes = (ctypes.c_int * 1)(3)
                runtime.dll.cnp_clear_error()
                pointer = runtime.dll.cnp_apply_over_axes(
                    callback, 1, bad_axes, source.pointer, None
                )
                self._assert_null_error(
                    runtime,
                    pointer,
                    "cnp_apply_over_axes",
                    (CNP_ERR_AXIS,),
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_callback_constructors_match_bounded_numpy_projections_and_errors(
        self,
    ) -> None:
        coordinate_calls: list[tuple[int, ...]] = []

        @COORD_CALLBACK
        def coordinate_callback(
            coordinates: ctypes.POINTER(ctypes.c_int64),
            ndim: int,
            _state: int,
        ) -> float:
            point = tuple(coordinates[index] for index in range(ndim))
            coordinate_calls.append(point)
            return float(sum((index + 1) * value for index, value in enumerate(point)) + 0.25)

        iterator_values = [1.9, -2.2, 7.0, 32767.0]
        iterator_index = ctypes.c_int64(0)

        @ITER_CALLBACK
        def iterator_callback(state: int) -> float:
            index_pointer = ctypes.cast(state, ctypes.POINTER(ctypes.c_int64))
            index = index_pointer.contents.value
            index_pointer.contents.value += 1
            return iterator_values[index]

        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            shape = (ctypes.c_int64 * 2)(2, 3)
            pointer = runtime.dll.cnp_fromfunction(
                coordinate_callback, 2, shape, None
            )
            with runtime._owned_result(pointer, "cnp_fromfunction") as actual:
                expected = np.fromfunction(
                    lambda first, second: first + 2 * second + 0.25,
                    (2, 3),
                    dtype=np.int64,
                ).astype(np.float64)
                assert_array_equivalent(self, actual, expected)
            self.assertEqual(
                list(np.ndindex((2, 3))), coordinate_calls
            )

            coordinate_calls.clear()
            pointer = runtime.dll.cnp_fromfunction(
                coordinate_callback, 0, None, None
            )
            with runtime._owned_result(pointer, "cnp_fromfunction") as scalar:
                assert_array_equivalent(
                    self, scalar, np.asarray(0.25, dtype=np.float64)
                )
            self.assertEqual([()], coordinate_calls)

            pointer = runtime.dll.cnp_fromiter(
                iterator_callback,
                ctypes.byref(iterator_index),
                len(iterator_values),
                CNP_SHORT,
            )
            with runtime._owned_result(pointer, "cnp_fromiter") as actual:
                expected = np.fromiter(
                    iterator_values,
                    dtype=np.int16,
                    count=len(iterator_values),
                )
                assert_array_equivalent(self, actual, expected)
            self.assertEqual(len(iterator_values), iterator_index.value)

            runtime.dll.cnp_clear_error()
            pointer = runtime.dll.cnp_fromfunction(
                COORD_CALLBACK(), 1, shape, None
            )
            self._assert_null_error(
                runtime,
                pointer,
                "cnp_fromfunction",
                (CNP_ERR_GENERIC,),
            )
            runtime.dll.cnp_clear_error()
            pointer = runtime.dll.cnp_fromiter(
                iterator_callback,
                ctypes.byref(iterator_index),
                -1,
                CNP_DOUBLE,
            )
            self._assert_null_error(
                runtime,
                pointer,
                "cnp_fromiter",
                (CNP_ERR_VALUE,),
            )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_unary_callback_projections_match_explicit_float64_numpy_calls(
        self,
    ) -> None:
        values = np.asarray(
            [[-2.0, -0.0, 1.5], [np.inf, np.nan, 4.0]], dtype=np.float32
        ).T

        for symbol in ("cnp_frompyfunc", "cnp_vectorize"):
            with self.subTest(symbol=symbol), CnumpyRuntime(DLL) as runtime:
                self._bind(runtime)
                baseline = runtime.retained_bytes
                calls: list[float] = []

                @UNARY_CALLBACK
                def callback(value: float, _state: int) -> float:
                    calls.append(value)
                    return value * 2.0 + 1.0

                base = runtime.from_numpy(values.T)
                source = runtime.transpose(base, (1, 0))
                base.close()
                function = getattr(runtime.dll, symbol)
                pointer = function(callback, source.pointer, None)
                result = runtime._owned_result(pointer, symbol)
                source.close()
                with result:
                    expected = np.vectorize(
                        lambda value: value * 2.0 + 1.0,
                        otypes=[np.float64],
                    )(values)
                    assert_array_equivalent(self, result, expected)
                np.testing.assert_array_equal(
                    np.asarray(calls), values.ravel(order="C").astype(np.float64)
                )

                runtime.dll.cnp_clear_error()
                pointer = function(UNARY_CALLBACK(), None, None)
                self._assert_null_error(
                    runtime,
                    pointer,
                    symbol,
                    (CNP_ERR_GENERIC,),
                )
                self.assertEqual(baseline, runtime.retained_bytes)

    def test_select_and_piecewise_match_broadcast_dtype_priority_and_errors(
        self,
    ) -> None:
        condition_values = (
            np.asarray([[True], [False]]),
            np.asarray([[False, True, True]]),
        )
        choice_values = (
            np.asarray([[1, 2, 3]], dtype=np.int16),
            np.asarray([[10.5], [20.5]], dtype=np.float32),
        )
        expected_select = np.select(
            condition_values, choice_values, default=-1.5
        )

        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            with ExitStack() as stack:
                conditions = [
                    stack.enter_context(runtime.from_numpy(value))
                    for value in condition_values
                ]
                choices = [
                    stack.enter_context(runtime.from_numpy(value))
                    for value in choice_values
                ]
                condition_pointers = (ctypes.c_void_p * 2)(
                    *(item.pointer for item in conditions)
                )
                choice_pointers = (ctypes.c_void_p * 2)(
                    *(item.pointer for item in choices)
                )
                pointer = runtime.dll.cnp_select(
                    2, condition_pointers, choice_pointers, -1.5
                )
                self.assertTrue(pointer, runtime.error_state())
                with runtime._owned_result(pointer, "cnp_select") as actual:
                    assert_array_equivalent(self, actual, expected_select)

            x_values = np.asarray([0, 1, 2], dtype=np.int16)
            piece_conditions = (
                np.asarray([True, True, False]),
                np.asarray([False, True, True]),
            )
            callback_calls: list[float] = []

            @UNARY_CALLBACK
            def stateful_callback(value: float, _state: int) -> float:
                callback_calls.append(value)
                return 10.0 * len(callback_calls) + value

            with ExitStack() as stack:
                x = stack.enter_context(runtime.from_numpy(x_values))
                conditions = [
                    stack.enter_context(runtime.from_numpy(value))
                    for value in piece_conditions
                ]
                condition_pointers = (ctypes.c_void_p * 2)(
                    *(item.pointer for item in conditions)
                )
                pointer = runtime.dll.cnp_piecewise(
                    x.pointer,
                    2,
                    condition_pointers,
                    stateful_callback,
                    None,
                )
                with runtime._owned_result(pointer, "cnp_piecewise") as actual:
                    assert_array_equivalent(
                        self,
                        actual,
                        np.asarray([10, 31, 42], dtype=np.int16),
                    )
            self.assertEqual([0.0, 1.0, 1.0, 2.0], callback_calls)

            with runtime.from_numpy(np.arange(6).reshape(2, 3)) as x, runtime.from_numpy(
                np.asarray([[True], [False]])
            ) as bad_condition:
                bad_pointers = (ctypes.c_void_p * 1)(bad_condition.pointer)
                runtime.dll.cnp_clear_error()
                pointer = runtime.dll.cnp_piecewise(
                    x.pointer,
                    1,
                    bad_pointers,
                    stateful_callback,
                    None,
                )
                self._assert_null_error(
                    runtime,
                    pointer,
                    "cnp_piecewise",
                    (CNP_ERR_SHAPE,),
                )

            runtime.dll.cnp_clear_error()
            pointer = runtime.dll.cnp_select(1, None, None, 0.0)
            self._assert_null_error(
                runtime,
                pointer,
                "cnp_select",
                (CNP_ERR_GENERIC,),
            )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_put_along_axis_matches_numpy_broadcasting_strides_and_atomic_errors(
        self,
    ) -> None:
        destination_values = np.arange(12, dtype=np.int16).reshape(3, 4).T
        index_values = np.asarray(
            [[0, -1], [1, 0], [2, 1], [0, 2]], dtype=np.int64
        )
        value_values = np.asarray([100.9, 200.1], dtype=np.float64)
        expected = destination_values.copy()
        np.put_along_axis(expected, index_values, value_values, axis=1)

        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            base = runtime.from_numpy(destination_values.T)
            destination = runtime.transpose(base, (1, 0))
            base.close()
            try:
                with runtime.from_numpy(index_values) as indices, runtime.from_numpy(
                    value_values
                ) as values:
                    runtime.dll.cnp_clear_error()
                    status = runtime.dll.cnp_put_along_axis(
                        destination.pointer, indices.pointer, values.pointer, -1
                    )
                    self.assertEqual(0, status, runtime.error_state())
                    np.testing.assert_array_equal(
                        destination.to_numpy(), expected, strict=True
                    )

                before = destination.to_numpy().copy()
                bad_indices_values = index_values.copy()
                bad_indices_values[0, 1] = destination.shape[1]
                with runtime.from_numpy(bad_indices_values) as bad_indices, runtime.from_numpy(
                    np.full(index_values.shape, 7, dtype=np.int16)
                ) as values:
                    runtime.dll.cnp_clear_error()
                    status = runtime.dll.cnp_put_along_axis(
                        destination.pointer,
                        bad_indices.pointer,
                        values.pointer,
                        1,
                    )
                    self.assertEqual(CNP_ERR_INDEX, status)
                    error = runtime.error_state()
                    self.assertEqual("cnp_put_along_axis", error.function)
                    self.assertTrue(error.message)
                    np.testing.assert_array_equal(
                        destination.to_numpy(), before, strict=True
                    )

                with runtime.from_numpy(index_values.astype(np.float64)) as float_indices, runtime.from_numpy(
                    value_values
                ) as values:
                    runtime.dll.cnp_clear_error()
                    status = runtime.dll.cnp_put_along_axis(
                        destination.pointer,
                        float_indices.pointer,
                        values.pointer,
                        1,
                    )
                    self.assertEqual(CNP_ERR_TYPE, status)
                    self.assertEqual(
                        "cnp_put_along_axis", runtime.error_state().function
                    )
                    np.testing.assert_array_equal(
                        destination.to_numpy(), before, strict=True
                    )
            finally:
                destination.close()
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
