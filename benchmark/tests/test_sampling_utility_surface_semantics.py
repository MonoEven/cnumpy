from __future__ import annotations

import ctypes
import unittest
from pathlib import Path

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"
CNP_ERR_GENERIC = -1
CNP_ERR_SHAPE = -4
CNP_ERR_TYPE = -3
CNP_ERR_VALUE = -13


class SamplingUtilitySurfaceSemanticsTests(unittest.TestCase):
    @staticmethod
    def _bind(runtime: CnumpyRuntime) -> None:
        dll = runtime.dll
        dll.cnp_histogram.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_double,
            ctypes.c_double,
        ]
        dll.cnp_histogram.restype = ctypes.c_void_p
        dll.cnp_histogram2d.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
        ]
        dll.cnp_histogram2d.restype = ctypes.c_void_p
        dll.cnp_interp.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        dll.cnp_interp.restype = ctypes.c_void_p
        dll.cnp_interp_nd.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_double,
        ]
        dll.cnp_interp_nd.restype = ctypes.c_void_p
        dll.cnp_nan_to_num.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ]
        dll.cnp_nan_to_num.restype = ctypes.c_void_p
        dll.cnp_vander.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_bool,
        ]
        dll.cnp_vander.restype = ctypes.c_void_p

    def test_histogram_count_projection_matches_numpy_ranges_and_errors(self) -> None:
        cases = (
            (np.asarray([-3.0, -1.0, 0.0, 1.0, 4.0]), 4, 0.0, 0.0, None),
            (np.asarray([5.0, 5.0, 5.0]), 5, 0.0, 0.0, None),
            (np.asarray([], dtype=np.float64), 3, 0.0, 0.0, None),
            (
                np.asarray([-10.0, 0.0, 0.5, 1.0, 10.0]),
                2,
                0.0,
                1.0,
                (0.0, 1.0),
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for values, bins, lower, upper, range_value in cases:
                with runtime.from_numpy(values) as source:
                    pointer = runtime.dll.cnp_histogram(
                        source.pointer, bins, lower, upper
                    )
                    with runtime._owned_result(pointer, "cnp_histogram") as actual:
                        expected = np.histogram(
                            values, bins=bins, range=range_value
                        )
                        np.testing.assert_array_equal(
                            expected[0], actual.to_numpy(), strict=True
                        )

            with runtime.from_numpy(np.asarray([1.0, 2.0])) as source:
                for bins, lower, upper, status in (
                    (0, 0.0, 0.0, CNP_ERR_VALUE),
                    (3, 2.0, 1.0, CNP_ERR_VALUE),
                ):
                    runtime.dll.cnp_clear_error()
                    pointer = runtime.dll.cnp_histogram(
                        source.pointer, bins, lower, upper
                    )
                    if pointer:
                        runtime._owned_result(pointer, "cnp_histogram").close()
                    self.assertFalse(pointer)
                    error = runtime.error_state()
                    self.assertEqual(status, error.status)
                    self.assertEqual("cnp_histogram", error.function)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_histogram2d_count_projection_matches_numpy_and_errors(self) -> None:
        cases = (
            (
                np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0]),
                np.asarray([3.0, 1.0, 0.0, 1.0, 3.0]),
                4,
            ),
            (np.asarray([5.0, 5.0]), np.asarray([-2.0, -2.0]), 3),
            (np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), 2),
        )
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for x_values, y_values, bins in cases:
                with runtime.from_numpy(x_values) as x, runtime.from_numpy(
                    y_values
                ) as y:
                    pointer = runtime.dll.cnp_histogram2d(
                        x.pointer, y.pointer, bins
                    )
                    with runtime._owned_result(pointer, "cnp_histogram2d") as actual:
                        expected = np.histogram2d(x_values, y_values, bins=bins)[0]
                        np.testing.assert_array_equal(
                            expected, actual.to_numpy(), strict=True
                        )

            with runtime.from_numpy(np.asarray([1.0])) as x, runtime.from_numpy(
                np.asarray([1.0, 2.0])
            ) as y:
                runtime.dll.cnp_clear_error()
                self.assertFalse(runtime.dll.cnp_histogram2d(x.pointer, y.pointer, 2))
                error = runtime.error_state()
                self.assertEqual(CNP_ERR_SHAPE, error.status)
                self.assertEqual("cnp_histogram2d", error.function)

            for x_values, y_values, bins, status in (
                (np.ones((1, 2)), np.ones((1, 2)), 2, CNP_ERR_SHAPE),
                (np.asarray([1.0]), np.asarray([1.0]), 0, CNP_ERR_VALUE),
                (np.asarray([np.nan]), np.asarray([1.0]), 2, CNP_ERR_VALUE),
            ):
                with runtime.from_numpy(x_values) as x, runtime.from_numpy(
                    y_values
                ) as y:
                    runtime.dll.cnp_clear_error()
                    pointer = runtime.dll.cnp_histogram2d(
                        x.pointer, y.pointer, bins
                    )
                    if pointer:
                        runtime._owned_result(pointer, "cnp_histogram2d").close()
                    self.assertFalse(pointer)
                    error = runtime.error_state()
                    self.assertEqual(status, error.status)
                    self.assertEqual("cnp_histogram2d", error.function)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_interp_default_and_explicit_boundaries_match_numpy_shape_and_dtype(
        self,
    ) -> None:
        x_values = np.asarray([[-2.0, 0.5, 1.5], [2.5, 4.0, np.nan]])
        xp_values = np.asarray([0.0, 1.0, 2.0, 3.0])
        fp_cases = (
            np.asarray([0.0, 1.0, 4.0, 9.0], dtype=np.float32),
            np.asarray([0 + 1j, 1 + 2j, 4 - 1j, 9 + 3j], dtype=np.complex64),
        )
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            with runtime.from_numpy(x_values) as x, runtime.from_numpy(
                xp_values
            ) as xp:
                for fp_values in fp_cases:
                    with runtime.from_numpy(fp_values) as fp:
                        pointer = runtime.dll.cnp_interp(
                            x.pointer, xp.pointer, fp.pointer
                        )
                        with runtime._owned_result(pointer, "cnp_interp") as actual:
                            expected = np.interp(x_values, xp_values, fp_values)
                            np.testing.assert_allclose(
                                expected,
                                actual.to_numpy(),
                                rtol=1e-12,
                                atol=1e-12,
                                equal_nan=True,
                            )
                            self.assertEqual(expected.dtype, actual.numpy_dtype)

                with runtime.from_numpy(fp_cases[0].astype(np.float64)) as fp:
                    pointer = runtime.dll.cnp_interp_nd(
                        x.pointer, xp.pointer, fp.pointer, 0.0, 0.0
                    )
                    with runtime._owned_result(pointer, "cnp_interp_nd") as actual:
                        expected = np.interp(
                            x_values, xp_values, fp_cases[0], left=0.0, right=0.0
                        )
                        np.testing.assert_allclose(
                            expected, actual.to_numpy(), equal_nan=True
                        )

            with runtime.from_numpy(np.asarray([], dtype=np.float64)) as x, runtime.from_numpy(
                np.asarray([], dtype=np.float64)
            ) as xp, runtime.from_numpy(np.asarray([], dtype=np.float64)) as fp:
                runtime.dll.cnp_clear_error()
                self.assertFalse(runtime.dll.cnp_interp(x.pointer, xp.pointer, fp.pointer))
                error = runtime.error_state()
                self.assertEqual(CNP_ERR_VALUE, error.status)
                self.assertEqual("cnp_interp", error.function)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_nan_to_num_explicit_replacements_preserve_numpy_dtype_and_complex_parts(
        self,
    ) -> None:
        cases = (
            np.asarray([np.nan, np.inf, -np.inf, -0.0, 3.0], dtype=np.float32),
            np.asarray(
                [complex(np.nan, np.inf), complex(-np.inf, np.nan), 2 - 3j],
                dtype=np.complex128,
            ),
            np.asarray([1, -2, 3], dtype=np.int64),
        )
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for values in cases:
                with self.subTest(dtype=values.dtype), runtime.from_numpy(values) as source:
                    pointer = runtime.dll.cnp_nan_to_num(
                        source.pointer, 7.0, 11.0, -13.0
                    )
                    with runtime._owned_result(pointer, "cnp_nan_to_num") as actual:
                        expected = np.nan_to_num(
                            values, nan=7.0, posinf=11.0, neginf=-13.0
                        )
                        np.testing.assert_array_equal(
                            expected, actual.to_numpy(), strict=True
                        )
                        np.testing.assert_array_equal(values, source.to_numpy())

            runtime.dll.cnp_clear_error()
            self.assertFalse(runtime.dll.cnp_nan_to_num(None, 7.0, 11.0, -13.0))
            error = runtime.error_state()
            self.assertEqual(CNP_ERR_GENERIC, error.status)
            self.assertEqual("cnp_nan_to_num", error.function)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_vander_matches_numpy_dtype_order_empty_columns_and_errors(self) -> None:
        cases = (
            (np.asarray([1, 2, -3], dtype=np.int8), -1, False, None),
            (np.asarray([1.5, -2.0], dtype=np.float32), 4, True, 4),
            (np.asarray([1 + 2j, -1j], dtype=np.complex64), 3, False, 3),
            (np.asarray([1.0, 2.0]), 0, False, 0),
        )
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for values, columns, increasing, oracle_columns in cases:
                with self.subTest(dtype=values.dtype, columns=columns), runtime.from_numpy(
                    values
                ) as source:
                    pointer = runtime.dll.cnp_vander(
                        source.pointer, columns, increasing
                    )
                    with runtime._owned_result(pointer, "cnp_vander") as actual:
                        expected = np.vander(
                            values, N=oracle_columns, increasing=increasing
                        )
                        np.testing.assert_array_equal(
                            expected, actual.to_numpy(), strict=True
                        )

            with runtime.from_numpy(np.ones((2, 2))) as source:
                runtime.dll.cnp_clear_error()
                pointer = runtime.dll.cnp_vander(source.pointer, 3, False)
                if pointer:
                    runtime._owned_result(pointer, "cnp_vander").close()
                self.assertFalse(pointer)
                error = runtime.error_state()
                self.assertEqual(CNP_ERR_SHAPE, error.status)
                self.assertEqual("cnp_vander", error.function)
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
