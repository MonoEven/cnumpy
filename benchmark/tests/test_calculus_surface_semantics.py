from __future__ import annotations

import ctypes
import unittest
from pathlib import Path

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"
CNP_ERR_AXIS = -5
CNP_ERR_SHAPE = -4
CNP_ERR_TYPE = -3
CNP_ERR_VALUE = -13


class CalculusSurfaceSemanticsTests(unittest.TestCase):
    @staticmethod
    def _bind(runtime: CnumpyRuntime) -> None:
        runtime.dll.cnp_diff.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        runtime.dll.cnp_diff.restype = ctypes.c_void_p
        runtime.dll.cnp_gradient.argtypes = [ctypes.c_void_p, ctypes.c_int]
        runtime.dll.cnp_gradient.restype = ctypes.c_void_p
        runtime.dll.cnp_unwrap.argtypes = [ctypes.c_void_p, ctypes.c_double]
        runtime.dll.cnp_unwrap.restype = ctypes.c_void_p

    def test_diff_matches_numpy_axes_dtypes_empty_results_and_errors(self) -> None:
        cases = (
            (np.arange(24, dtype=np.float64).reshape(2, 3, 4), 1, -1),
            (np.arange(24, dtype=np.int64).reshape(2, 3, 4), 2, 1),
            (
                np.asarray([[True, False, True], [False, False, True]]),
                1,
                0,
            ),
            (np.empty((2, 0, 3), dtype=np.uint64), 3, 1),
        )
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for values, order, axis in cases:
                with self.subTest(dtype=values.dtype, n=order, axis=axis):
                    with runtime.from_numpy(values) as source:
                        pointer = runtime.dll.cnp_diff(source.pointer, order, axis)
                        with runtime._owned_result(pointer, "cnp_diff") as actual:
                            np.testing.assert_array_equal(
                                np.diff(values, n=order, axis=axis),
                                actual.to_numpy(),
                                strict=True,
                            )

            with runtime.from_numpy(np.arange(6.0).reshape(2, 3)) as source:
                for order, axis, status in (
                    (-1, 1, CNP_ERR_VALUE),
                    (1, 2, CNP_ERR_AXIS),
                ):
                    runtime.dll.cnp_clear_error()
                    pointer = runtime.dll.cnp_diff(source.pointer, order, axis)
                    if pointer:
                        runtime._owned_result(pointer, "cnp_diff").close()
                    self.assertFalse(pointer)
                    error = runtime.error_state()
                    self.assertEqual(status, error.status)
                    self.assertEqual("cnp_diff", error.function)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_gradient_matches_numpy_default_spacing_axes_dtypes_and_errors(
        self,
    ) -> None:
        cases = (
            (np.arange(24, dtype=np.float64).reshape(2, 3, 4) ** 2, -1),
            (np.arange(12, dtype=np.int64).reshape(3, 4) ** 3, 0),
            (np.linspace(-2, 3, 20, dtype=np.float32).reshape(4, 5) ** 2, 1),
            (
                np.asarray(
                    [[1 + 2j, 3 - 1j, 7 + 4j], [2 + 0j, 5 + 2j, 9 - 3j]],
                    dtype=np.complex64,
                ),
                -1,
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for values, axis in cases:
                with self.subTest(dtype=values.dtype, axis=axis):
                    with runtime.from_numpy(values) as source:
                        pointer = runtime.dll.cnp_gradient(source.pointer, axis)
                        with runtime._owned_result(pointer, "cnp_gradient") as actual:
                            expected = np.gradient(values, axis=axis)
                            self.assertEqual(expected.dtype, actual.numpy_dtype)
                            np.testing.assert_allclose(
                                expected,
                                actual.to_numpy(),
                                rtol=2e-6,
                                atol=2e-6,
                                equal_nan=True,
                            )

            for values, axis, status in (
                (np.asarray([True, False]), 0, CNP_ERR_TYPE),
                (np.asarray([[1.0], [2.0]]), 1, CNP_ERR_SHAPE),
                (np.arange(6.0).reshape(2, 3), 3, CNP_ERR_AXIS),
            ):
                with runtime.from_numpy(values) as source:
                    runtime.dll.cnp_clear_error()
                    pointer = runtime.dll.cnp_gradient(source.pointer, axis)
                    if pointer:
                        runtime._owned_result(pointer, "cnp_gradient").close()
                    self.assertFalse(pointer)
                    error = runtime.error_state()
                    self.assertEqual(status, error.status)
                    self.assertEqual("cnp_gradient", error.function)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_unwrap_matches_numpy_last_axis_dtype_thresholds_and_errors(
        self,
    ) -> None:
        cases = (
            (
                np.asarray([[0.0, 3.5, 7.0], [1.0, -3.0, -7.0]]),
                np.pi,
            ),
            (
                np.asarray([[0.0, 4.0, 8.0], [2.0, 6.0, 10.0]], dtype=np.float32),
                4.5,
            ),
            (np.asarray([0, 4, 8, 12], dtype=np.int64), np.pi),
            (np.empty((3, 0), dtype=np.float64), np.pi),
        )
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for values, discont in cases:
                with self.subTest(dtype=values.dtype, shape=values.shape):
                    with runtime.from_numpy(values) as source:
                        pointer = runtime.dll.cnp_unwrap(source.pointer, discont)
                        with runtime._owned_result(pointer, "cnp_unwrap") as actual:
                            expected = np.unwrap(values, discont=discont, axis=-1)
                            self.assertEqual(expected.dtype, actual.numpy_dtype)
                            np.testing.assert_allclose(
                                expected,
                                actual.to_numpy(),
                                rtol=2e-6,
                                atol=2e-6,
                                equal_nan=True,
                            )

            with runtime.from_numpy(
                np.asarray([1 + 2j, 3 + 4j], dtype=np.complex128)
            ) as source:
                runtime.dll.cnp_clear_error()
                pointer = runtime.dll.cnp_unwrap(source.pointer, np.pi)
                if pointer:
                    runtime._owned_result(pointer, "cnp_unwrap").close()
                self.assertFalse(pointer)
                error = runtime.error_state()
                self.assertEqual(CNP_ERR_TYPE, error.status)
                self.assertEqual("cnp_unwrap", error.function)
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
