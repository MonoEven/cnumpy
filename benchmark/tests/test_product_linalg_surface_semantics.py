from __future__ import annotations

import ctypes
import unittest
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime
from compat.oracle import assert_array_equivalent


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"


class ProductLinalgSurfaceTests(unittest.TestCase):
    def _array_function(
        self, runtime: CnumpyRuntime, name: str, argtypes: list[object]
    ) -> ctypes._CFuncPtr:
        function = getattr(runtime.dll, name)
        function.argtypes = argtypes
        function.restype = ctypes.c_void_p
        return function

    def _binary_result(
        self,
        runtime: CnumpyRuntime,
        name: str,
        left,
        right,
    ):
        function = self._array_function(
            runtime, name, [ctypes.c_void_p, ctypes.c_void_p]
        )
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(
            function(left.pointer, right.pointer), name
        )

    def test_dot_and_matmul_match_numpy_rank_dtype_complex_and_batch_rules(
        self,
    ) -> None:
        dot_cases = (
            (
                np.asarray(2, dtype=np.int16),
                np.arange(6, dtype=np.int16).reshape(2, 3),
            ),
            (
                np.asarray([1 + 2j, 3 - 4j], dtype=np.complex64),
                np.asarray([2 - 1j, -5 + 2j], dtype=np.complex64),
            ),
            (
                np.arange(24, dtype=np.float32).reshape(2, 3, 4),
                np.arange(40, dtype=np.float32).reshape(5, 4, 2),
            ),
            (
                np.asarray([120, 120], dtype=np.int8),
                np.asarray([2, 3], dtype=np.int8),
            ),
        )
        matmul_cases = (
            (
                np.arange(24, dtype=np.float64).reshape(2, 1, 3, 4),
                np.arange(40, dtype=np.float64).reshape(1, 5, 4, 2),
            ),
            (
                np.asarray([1 + 2j, 3 - 4j], dtype=np.complex128),
                np.asarray(
                    [[2 - 1j, 4 + 3j], [-5 + 2j, 1 - 6j]],
                    dtype=np.complex128,
                ),
            ),
            (
                np.asarray([[1, 2], [3, 4]], dtype=np.int16),
                np.asarray([[5, 6], [7, 8]], dtype=np.int16),
            ),
            (
                np.arange(12, dtype=np.float32).reshape(3, 4),
                np.arange(4, dtype=np.float32),
            ),
        )

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for name, cases, oracle in (
                ("cnp_dot", dot_cases, np.dot),
                ("cnp_dot_general", dot_cases, np.dot),
                ("cnp_matmul", matmul_cases, np.matmul),
            ):
                for left_value, right_value in cases:
                    with self.subTest(
                        operation=name,
                        left_shape=left_value.shape,
                        right_shape=right_value.shape,
                        dtype=str(left_value.dtype),
                    ), ExitStack() as stack:
                        left = stack.enter_context(
                            runtime.from_numpy(left_value)
                        )
                        right = stack.enter_context(
                            runtime.from_numpy(right_value)
                        )
                        actual = stack.enter_context(
                            self._binary_result(
                                runtime, name, left, right
                            )
                        )
                        expected = oracle(left_value, right_value)
                        assert_array_equivalent(
                            self,
                            actual,
                            expected,
                            rtol=2e-12,
                            atol=2e-12,
                        )
                    self.assertEqual(baseline, runtime.retained_bytes)

    def test_dot_and_matmul_failures_are_labeled_atomic_and_nonretaining(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            cases = (
                (
                    "cnp_dot",
                    np.ones((2, 3), dtype=np.float64),
                    np.ones((4,), dtype=np.float64),
                ),
                (
                    "cnp_dot_general",
                    np.ones((2, 3), dtype=np.float64),
                    np.ones((4,), dtype=np.float64),
                ),
                (
                    "cnp_matmul",
                    np.asarray(2.0),
                    np.ones((2, 2), dtype=np.float64),
                ),
                (
                    "cnp_matmul",
                    np.ones((2, 3), dtype=np.float64),
                    np.ones((4, 2), dtype=np.float64),
                ),
            )
            for name, left_value, right_value in cases:
                with self.subTest(operation=name), ExitStack() as stack:
                    left = stack.enter_context(runtime.from_numpy(left_value))
                    right = stack.enter_context(runtime.from_numpy(right_value))
                    function = self._array_function(
                        runtime,
                        name,
                        [ctypes.c_void_p, ctypes.c_void_p],
                    )
                    runtime.dll.cnp_clear_error()
                    self.assertFalse(function(left.pointer, right.pointer))
                    error = runtime.error_state()
                    self.assertNotEqual(0, error.status)
                    self.assertEqual(name, error.function)
                self.assertEqual(baseline, runtime.retained_bytes)

            for name in ("cnp_dot", "cnp_dot_general", "cnp_matmul"):
                function = self._array_function(
                    runtime,
                    name,
                    [ctypes.c_void_p, ctypes.c_void_p],
                )
                runtime.dll.cnp_clear_error()
                self.assertFalse(function(None, None))
                error = runtime.error_state()
                self.assertNotEqual(0, error.status)
                self.assertEqual(name, error.function)
                self.assertEqual(baseline, runtime.retained_bytes)

    def test_inner_outer_cross_kron_and_tensordot_match_numpy(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes

            binary_cases = {
                "cnp_inner": (
                    (
                        np.arange(24, dtype=np.float32).reshape(2, 3, 4),
                        np.arange(20, dtype=np.float32).reshape(5, 4),
                    ),
                    (
                        np.asarray(2 + 3j, dtype=np.complex64),
                        np.arange(6, dtype=np.float32).reshape(2, 3),
                    ),
                ),
                "cnp_outer": (
                    (
                        np.arange(6, dtype=np.int16).reshape(2, 3),
                        np.asarray([3, -2], dtype=np.int16),
                    ),
                    (
                        np.asarray([1 + 2j, 3 - 4j], dtype=np.complex64),
                        np.asarray([[2 - 1j], [-5 + 2j]], dtype=np.complex64),
                    ),
                ),
                "cnp_kron": (
                    (
                        np.arange(4, dtype=np.int16).reshape(2, 1, 2),
                        np.arange(6, dtype=np.int16).reshape(3, 2),
                    ),
                    (
                        np.asarray([[1 + 2j], [3 - 4j]], dtype=np.complex128),
                        np.asarray([2 - 1j, -5 + 2j], dtype=np.complex128),
                    ),
                ),
            }
            oracles = {
                "cnp_inner": np.inner,
                "cnp_outer": np.outer,
                "cnp_kron": np.kron,
            }
            for name, cases in binary_cases.items():
                for left_value, right_value in cases:
                    with self.subTest(
                        operation=name,
                        left_shape=left_value.shape,
                        right_shape=right_value.shape,
                    ), ExitStack() as stack:
                        left = stack.enter_context(runtime.from_numpy(left_value))
                        right = stack.enter_context(runtime.from_numpy(right_value))
                        actual = stack.enter_context(
                            self._binary_result(runtime, name, left, right)
                        )
                        assert_array_equivalent(
                            self,
                            actual,
                            oracles[name](left_value, right_value),
                            rtol=2e-12,
                            atol=2e-12,
                        )
                    self.assertEqual(baseline, runtime.retained_bytes)

            cross = self._array_function(
                runtime,
                "cnp_cross",
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int],
            )
            cross_cases = (
                (
                    np.arange(6, dtype=np.int32).reshape(2, 1, 3),
                    np.arange(12, dtype=np.int32).reshape(1, 4, 3),
                    -1,
                ),
                (
                    np.arange(6, dtype=np.float64).reshape(3, 2, 1),
                    np.arange(12, dtype=np.float64).reshape(3, 1, 4),
                    0,
                ),
                (
                    np.asarray([[1, 2], [3, 4]], dtype=np.int16),
                    np.asarray([[5, 6], [7, 8]], dtype=np.int16),
                    -1,
                ),
                (
                    np.asarray([[1 + 2j, 3 - 4j, 2 + 1j]], dtype=np.complex128),
                    np.asarray([[2 - 1j, -5 + 2j, 1 - 6j]], dtype=np.complex128),
                    -1,
                ),
            )
            for left_value, right_value, axis in cross_cases:
                with self.subTest(
                    operation="cnp_cross", axis=axis
                ), ExitStack() as stack:
                    left = stack.enter_context(runtime.from_numpy(left_value))
                    right = stack.enter_context(runtime.from_numpy(right_value))
                    runtime.dll.cnp_clear_error()
                    actual = stack.enter_context(
                        runtime._owned_result(
                            cross(left.pointer, right.pointer, axis),
                            "cnp_cross",
                        )
                    )
                    expected = np.cross(
                        left_value,
                        right_value,
                        axisa=axis,
                        axisb=axis,
                        axisc=axis,
                    )
                    assert_array_equivalent(
                        self, actual, expected, rtol=2e-12, atol=2e-12
                    )
                self.assertEqual(baseline, runtime.retained_bytes)

            tensordot = self._array_function(
                runtime,
                "cnp_tensordot",
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
            )
            tensordot_default = self._array_function(
                runtime,
                "cnp_tensordot_default",
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int],
            )
            for axes in (0, 1, 2):
                left_value = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
                right_value = (
                    np.arange(24, dtype=np.float64).reshape(3, 4, 2)
                    if axes == 2
                    else (
                        np.arange(20, dtype=np.float64).reshape(4, 5)
                        if axes == 1
                        else np.arange(6, dtype=np.float64).reshape(2, 3)
                    )
                )
                with self.subTest(
                    operation="tensordot", axes=axes
                ), ExitStack() as stack:
                    left = stack.enter_context(runtime.from_numpy(left_value))
                    right = stack.enter_context(runtime.from_numpy(right_value))
                    expected = np.tensordot(left_value, right_value, axes=axes)
                    for name, call in (
                        (
                            "cnp_tensordot",
                            lambda: tensordot(
                                left.pointer, right.pointer, axes, axes
                            ),
                        ),
                        (
                            "cnp_tensordot_default",
                            lambda: tensordot_default(
                                left.pointer, right.pointer, axes
                            ),
                        ),
                    ):
                        actual = stack.enter_context(
                            runtime._owned_result(call(), name)
                        )
                        assert_array_equivalent(
                            self,
                            actual,
                            expected,
                            rtol=2e-12,
                            atol=2e-12,
                        )
                self.assertEqual(baseline, runtime.retained_bytes)

    def test_product_alias_failures_are_explicit_atomic_and_labeled(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            left = runtime.from_numpy(np.ones((2, 3), dtype=np.float64))
            right = runtime.from_numpy(np.ones((4, 2), dtype=np.float64))
            try:
                failures = (
                    (
                        "cnp_inner",
                        [ctypes.c_void_p, ctypes.c_void_p],
                        (left.pointer, right.pointer),
                    ),
                    (
                        "cnp_outer",
                        [ctypes.c_void_p, ctypes.c_void_p],
                        (None, right.pointer),
                    ),
                    (
                        "cnp_kron",
                        [ctypes.c_void_p, ctypes.c_void_p],
                        (left.pointer, None),
                    ),
                    (
                        "cnp_cross",
                        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int],
                        (left.pointer, right.pointer, 2),
                    ),
                    (
                        "cnp_tensordot",
                        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
                        (left.pointer, right.pointer, 1, 2),
                    ),
                    (
                        "cnp_tensordot_default",
                        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int],
                        (left.pointer, right.pointer, -1),
                    ),
                )
                for name, argtypes, arguments in failures:
                    with self.subTest(operation=name):
                        function = self._array_function(runtime, name, argtypes)
                        runtime.dll.cnp_clear_error()
                        self.assertFalse(function(*arguments))
                        error = runtime.error_state()
                        self.assertNotEqual(0, error.status)
                        self.assertEqual(name, error.function)
            finally:
                left.close()
                right.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_dot1d_trace_vdot_and_multi_dot_match_numpy_projections(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            dot_1d = self._array_function(
                runtime,
                "cnp_dot_1d",
                [ctypes.c_void_p, ctypes.c_void_p],
            )
            for left_value, right_value in (
                (
                    np.asarray([1 + 2j, 3 - 4j], dtype=np.complex64),
                    np.asarray([2 - 1j, -5 + 2j], dtype=np.complex64),
                ),
                (
                    np.asarray([120, 120], dtype=np.int8),
                    np.asarray([2, 3], dtype=np.int8),
                ),
            ):
                with self.subTest(operation="cnp_dot_1d"), ExitStack() as stack:
                    left = stack.enter_context(runtime.from_numpy(left_value))
                    right = stack.enter_context(runtime.from_numpy(right_value))
                    actual = stack.enter_context(
                        runtime._owned_result(
                            dot_1d(left.pointer, right.pointer), "cnp_dot_1d"
                        )
                    )
                    assert_array_equivalent(
                        self,
                        actual,
                        np.dot(left_value, right_value),
                        rtol=2e-12,
                        atol=2e-12,
                    )
                self.assertEqual(baseline, runtime.retained_bytes)

            trace = self._array_function(
                runtime,
                "cnp_trace",
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int],
            )
            trace_cases = (
                (
                    np.arange(60, dtype=np.int16).reshape(3, 4, 5),
                    1,
                    0,
                    2,
                ),
                (
                    (
                        np.arange(24, dtype=np.float32).reshape(2, 3, 4)
                        + 1j * np.arange(24, dtype=np.float32).reshape(2, 3, 4)
                    ).astype(np.complex64),
                    -1,
                    -2,
                    -1,
                ),
                (np.empty((2, 0, 3), dtype=np.float64), 0, 1, 2),
            )
            for value, offset, axis1, axis2 in trace_cases:
                with self.subTest(
                    operation="cnp_trace",
                    shape=value.shape,
                    offset=offset,
                ), ExitStack() as stack:
                    source = stack.enter_context(runtime.from_numpy(value))
                    actual = stack.enter_context(
                        runtime._owned_result(
                            trace(source.pointer, offset, axis1, axis2),
                            "cnp_trace",
                        )
                    )
                    assert_array_equivalent(
                        self,
                        actual,
                        np.trace(value, offset=offset, axis1=axis1, axis2=axis2),
                        rtol=2e-12,
                        atol=2e-12,
                    )
                self.assertEqual(baseline, runtime.retained_bytes)

            trace_ext = getattr(runtime.dll, "cnp_trace_ext")
            trace_ext.argtypes = [ctypes.c_void_p, ctypes.c_int]
            trace_ext.restype = ctypes.c_double
            vdot = getattr(runtime.dll, "cnp_vdot")
            vdot.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            vdot.restype = ctypes.c_double
            left_value = np.arange(12, dtype=np.float64).reshape(3, 4).T
            right_value = np.linspace(-2.0, 3.0, 12).reshape(4, 3)
            with ExitStack() as stack:
                left_owner = stack.enter_context(
                    runtime.from_numpy(left_value.T.copy())
                )
                left = stack.enter_context(runtime.transpose(left_owner))
                right = stack.enter_context(runtime.from_numpy(right_value))
                self.assertEqual(np.trace(left_value, offset=-1), trace_ext(left.pointer, -1))
                self.assertAlmostEqual(
                    float(np.vdot(left_value, right_value)),
                    vdot(left.pointer, right.pointer),
                    places=12,
                )
            self.assertEqual(baseline, runtime.retained_bytes)

            multi_dot = self._array_function(
                runtime,
                "cnp_multi_dot",
                [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)],
            )
            multi_cases = (
                (
                    np.arange(6, dtype=np.float64).reshape(2, 3),
                    np.arange(12, dtype=np.float64).reshape(3, 4),
                    np.arange(8, dtype=np.float64).reshape(4, 2),
                ),
                (
                    np.arange(3, dtype=np.float32),
                    np.arange(12, dtype=np.float32).reshape(3, 4),
                    np.arange(4, dtype=np.float32),
                ),
                (
                    np.asarray([[1 + 2j, 3 - 4j]], dtype=np.complex128),
                    np.asarray([[2 - 1j], [-5 + 2j]], dtype=np.complex128),
                    np.asarray([[1 - 6j, 4 + 3j]], dtype=np.complex128),
                ),
            )
            for values in multi_cases:
                with self.subTest(
                    operation="cnp_multi_dot",
                    shapes=tuple(value.shape for value in values),
                ), ExitStack() as stack:
                    arrays = [
                        stack.enter_context(runtime.from_numpy(value))
                        for value in values
                    ]
                    pointers = (ctypes.c_void_p * len(arrays))(
                        *(array.pointer.value for array in arrays)
                    )
                    actual = stack.enter_context(
                        runtime._owned_result(
                            multi_dot(len(arrays), pointers), "cnp_multi_dot"
                        )
                    )
                    assert_array_equivalent(
                        self,
                        actual,
                        np.linalg.multi_dot(values),
                        rtol=2e-11,
                        atol=2e-11,
                    )
                self.assertEqual(baseline, runtime.retained_bytes)

    def test_dot1d_trace_vdot_and_multi_dot_errors_keep_real_failures(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            left = runtime.from_numpy(np.ones((2,), dtype=np.float64))
            right = runtime.from_numpy(np.ones((3,), dtype=np.float64))
            complex_value = runtime.from_numpy(
                np.asarray([1 + 2j, 3 - 4j], dtype=np.complex128)
            )
            matrix = runtime.from_numpy(np.ones((2, 2), dtype=np.float64))
            try:
                array_failures = (
                    (
                        "cnp_dot_1d",
                        [ctypes.c_void_p, ctypes.c_void_p],
                        (left.pointer, right.pointer),
                    ),
                    (
                        "cnp_trace",
                        [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int],
                        (matrix.pointer, 0, 0, 0),
                    ),
                )
                for name, argtypes, arguments in array_failures:
                    function = self._array_function(runtime, name, argtypes)
                    runtime.dll.cnp_clear_error()
                    pointer = function(*arguments)
                    if pointer:
                        runtime.dll.cnp_array_decref(pointer)
                    self.assertFalse(pointer)
                    self.assertEqual(name, runtime.error_state().function)

                for name, argtypes, arguments in (
                    (
                        "cnp_vdot",
                        [ctypes.c_void_p, ctypes.c_void_p],
                        (left.pointer, right.pointer),
                    ),
                    (
                        "cnp_vdot",
                        [ctypes.c_void_p, ctypes.c_void_p],
                        (complex_value.pointer, complex_value.pointer),
                    ),
                    (
                        "cnp_trace_ext",
                        [ctypes.c_void_p, ctypes.c_int],
                        (complex_value.pointer, 0),
                    ),
                ):
                    function = getattr(runtime.dll, name)
                    function.argtypes = argtypes
                    function.restype = ctypes.c_double
                    runtime.dll.cnp_clear_error()
                    self.assertTrue(np.isnan(function(*arguments)))
                    self.assertNotEqual(0, runtime.error_state().status)
                    self.assertEqual(name, runtime.error_state().function)

                multi_dot = self._array_function(
                    runtime,
                    "cnp_multi_dot",
                    [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)],
                )
                for count, pointers in ((1, None), (2, None)):
                    runtime.dll.cnp_clear_error()
                    self.assertFalse(multi_dot(count, pointers))
                    self.assertEqual(
                        "cnp_multi_dot", runtime.error_state().function
                    )
            finally:
                matrix.close()
                complex_value.close()
                right.close()
                left.close()
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
