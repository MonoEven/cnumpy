from __future__ import annotations

import ctypes
import math
import unittest
from pathlib import Path

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"
CNP_ERR_INDEX = -6
CNP_ERR_TYPE = -3
CNP_UBYTE = 3


class LegacyNumericUtilitySemanticsTests(unittest.TestCase):
    def _bind(self, runtime: CnumpyRuntime) -> None:
        dll = runtime.dll
        dll.cnp_clip.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
        dll.cnp_clip.restype = ctypes.c_void_p
        dll.cnp_sqrt_into.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        dll.cnp_sqrt_into.restype = ctypes.c_int
        dll.cnp_item.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        dll.cnp_item.restype = ctypes.c_double
        dll.cnp_bitwise_count.argtypes = [ctypes.c_void_p]
        dll.cnp_bitwise_count.restype = ctypes.c_void_p

    def test_clip_double_bounds_matches_numpy_promotion_errors_and_lifetime(
        self,
    ) -> None:
        cases = (
            (np.asarray([-4, 1, 9], dtype=np.int64), -0.5, 4.5),
            (np.asarray([-np.inf, -0.0, np.nan, np.inf]), -2.0, 3.0),
            (np.asarray([[1.0], [7.0]], dtype=np.float32), 5.0, 2.0),
        )
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for values, minimum, maximum in cases:
                with self.subTest(dtype=values.dtype, shape=values.shape):
                    with runtime.from_numpy(values) as source:
                        pointer = runtime.dll.cnp_clip(
                            source.pointer, minimum, maximum
                        )
                        with runtime._owned_result(pointer, "cnp_clip") as actual:
                            expected = np.clip(values, minimum, maximum)
                            np.testing.assert_array_equal(
                                expected, actual.to_numpy(), strict=True
                            )

            runtime.dll.cnp_clear_error()
            self.assertFalse(runtime.dll.cnp_clip(None, -1.0, 1.0))
            error = runtime.error_state()
            self.assertEqual("cnp_clip", error.function)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_sqrt_into_matches_numpy_in_place_and_rejects_invalid_output(
        self,
    ) -> None:
        values = np.asarray([[0.0, 1.0, 4.0], [9.0, 16.0, np.inf]])
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            with runtime.from_numpy(values) as source, runtime.from_numpy(
                np.zeros_like(values)
            ) as output:
                status = runtime.dll.cnp_sqrt_into(source.pointer, output.pointer)
                self.assertEqual(0, status, runtime.error_state())
                np.testing.assert_array_equal(np.sqrt(values), output.to_numpy())

            with runtime.from_numpy(values) as in_place:
                status = runtime.dll.cnp_sqrt_into(in_place.pointer, in_place.pointer)
                self.assertEqual(0, status, runtime.error_state())
                np.testing.assert_array_equal(np.sqrt(values), in_place.to_numpy())

            with runtime.from_numpy(values) as source, runtime.from_numpy(
                np.zeros(values.shape, dtype=np.int64)
            ) as output:
                before = output.to_numpy().copy()
                runtime.dll.cnp_clear_error()
                status = runtime.dll.cnp_sqrt_into(source.pointer, output.pointer)
                self.assertEqual(CNP_ERR_TYPE, status)
                self.assertEqual("cnp_sqrt_into", runtime.error_state().function)
                np.testing.assert_array_equal(before, output.to_numpy())
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_item_double_projection_supports_negative_indices_and_errors(
        self,
    ) -> None:
        values = np.asarray([[1.5, -2.25], [3.75, 8.5]])
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            with runtime.from_numpy(values) as source:
                for index in (0, 2, -1, -4):
                    with self.subTest(index=index):
                        self.assertEqual(
                            float(values.item(index)),
                            runtime.dll.cnp_item(source.pointer, index),
                        )

                runtime.dll.cnp_clear_error()
                self.assertTrue(math.isnan(runtime.dll.cnp_item(source.pointer, 4)))
                error = runtime.error_state()
                self.assertEqual(CNP_ERR_INDEX, error.status)
                self.assertEqual("cnp_item", error.function)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_bitwise_count_uses_uint8_and_exact_integer_magnitudes(self) -> None:
        cases = (
            np.asarray([0, 1, 3, 255, 1023], dtype=np.int64),
            np.asarray([0, 1, 2**32, 2**64 - 1], dtype=np.uint64),
            np.asarray([False, True, True], dtype=np.bool_),
            np.asarray([-1, -3, np.iinfo(np.int64).min], dtype=np.int64),
        )
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for values in cases:
                with self.subTest(dtype=values.dtype):
                    expected = np.asarray(
                        [abs(int(value)).bit_count() for value in values.flat],
                        dtype=np.uint8,
                    ).reshape(values.shape)
                    with runtime.from_numpy(values) as source:
                        pointer = runtime.dll.cnp_bitwise_count(source.pointer)
                        with runtime._owned_result(
                            pointer, "cnp_bitwise_count"
                        ) as actual:
                            self.assertEqual(CNP_UBYTE, actual.dtype_number)
                            np.testing.assert_array_equal(
                                expected, actual.to_numpy(), strict=True
                            )

            with runtime.from_numpy(np.asarray([1.0])) as source:
                runtime.dll.cnp_clear_error()
                self.assertFalse(runtime.dll.cnp_bitwise_count(source.pointer))
                error = runtime.error_state()
                self.assertEqual(CNP_ERR_TYPE, error.status)
                self.assertEqual("cnp_bitwise_count", error.function)
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
