from __future__ import annotations

import ctypes
from contextlib import ExitStack
from pathlib import Path
import unittest

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"


class _CnpSlice(ctypes.Structure):
    _fields_ = (
        ("start", ctypes.c_int64),
        ("stop", ctypes.c_int64),
        ("step", ctypes.c_int64),
        ("has_start", ctypes.c_bool),
        ("has_stop", ctypes.c_bool),
        ("has_step", ctypes.c_bool),
    )


class ArrayRelationSurfaceSemanticsTests(unittest.TestCase):
    SYMBOLS = frozenset(
        {
            "cnp_array_equal",
            "cnp_array_equiv",
            "cnp_byte_bounds",
            "cnp_fliplr",
            "cnp_flipud",
            "cnp_isclose",
            "cnp_may_share_memory",
            "cnp_shares_memory",
        }
    )

    @staticmethod
    def _function(runtime: CnumpyRuntime, name: str, argtypes, restype):
        function = getattr(runtime.dll, name)
        function.argtypes = argtypes
        function.restype = restype
        return function

    def test_flip_and_isclose_match_numpy_broadcast_nan_dtype_and_lifetime(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as results:
            baseline = runtime.retained_bytes
            fliplr = self._function(
                runtime, "cnp_fliplr", [ctypes.c_void_p], ctypes.c_void_p
            )
            flipud = self._function(
                runtime, "cnp_flipud", [ctypes.c_void_p], ctypes.c_void_p
            )
            isclose = self._function(
                runtime,
                "cnp_isclose",
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double, ctypes.c_double],
                ctypes.c_void_p,
            )

            values = np.arange(12, dtype=np.int64).reshape(3, 4)
            with runtime.from_numpy(values) as source:
                transposed = runtime.transpose(source, (1, 0))
                left_right = results.enter_context(
                    runtime._owned_result(fliplr(transposed.pointer), "cnp_fliplr")
                )
                up_down = results.enter_context(
                    runtime._owned_result(flipud(transposed.pointer), "cnp_flipud")
                )
                transposed.close()
            np.testing.assert_array_equal(
                np.fliplr(values.T), left_right.to_numpy(), strict=True
            )
            np.testing.assert_array_equal(
                np.flipud(values.T), up_down.to_numpy(), strict=True
            )

            left_values = np.asarray([[1.0], [np.nan]], dtype=np.float64)
            right_values = np.asarray([1.0, 1.0 + 1e-7, np.nan], dtype=np.float64)
            with runtime.from_numpy(left_values) as left, runtime.from_numpy(
                right_values
            ) as right:
                close = results.enter_context(
                    runtime._owned_result(
                        isclose(left.pointer, right.pointer, 1e-5, 1e-8),
                        "cnp_isclose",
                    )
                )
            np.testing.assert_array_equal(
                np.isclose(left_values, right_values, rtol=1e-5, atol=1e-8),
                close.to_numpy(),
                strict=True,
            )

            results.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_array_equal_and_equiv_match_numpy_wide_integer_and_broadcast(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            equal = self._function(
                runtime,
                "cnp_array_equal",
                [ctypes.c_void_p, ctypes.c_void_p],
                ctypes.c_bool,
            )
            equiv = self._function(
                runtime,
                "cnp_array_equiv",
                [ctypes.c_void_p, ctypes.c_void_p],
                ctypes.c_bool,
            )
            wide = np.asarray([2**60 + 1, 2**60 + 3], dtype=np.int64)
            cases = (
                (wide, wide.copy()),
                (wide, np.asarray([2**60 + 1, 2**60 + 5], dtype=np.int64)),
                (np.asarray([[1, 2, 3], [1, 2, 3]]), np.asarray([1, 2, 3])),
                (np.asarray([1, 2]), np.asarray([1, 2, 3])),
            )
            for left_values, right_values in cases:
                with self.subTest(
                    left_shape=left_values.shape, right_shape=right_values.shape
                ), runtime.from_numpy(left_values) as left, runtime.from_numpy(
                    right_values
                ) as right:
                    runtime.dll.cnp_clear_error()
                    self.assertEqual(
                        bool(np.array_equal(left_values, right_values)),
                        bool(equal(left.pointer, right.pointer)),
                    )
                    self.assertEqual(0, runtime.error_state().status)
                    runtime.dll.cnp_clear_error()
                    self.assertEqual(
                        bool(np.array_equiv(left_values, right_values)),
                        bool(equiv(left.pointer, right.pointer)),
                    )
                    self.assertEqual(0, runtime.error_state().status)

            for name, function in (("cnp_array_equal", equal), ("cnp_array_equiv", equiv)):
                with self.subTest(symbol=name):
                    runtime.dll.cnp_clear_error()
                    self.assertFalse(function(None, None))
                    self.assertEqual(name, runtime.error_state().function)
                    self.assertEqual(baseline, runtime.retained_bytes)

    def test_byte_bounds_and_memory_overlap_match_numpy_strided_views(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            array_slice = self._function(
                runtime,
                "cnp_array_slice",
                [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(_CnpSlice)],
                ctypes.c_void_p,
            )
            array_at = self._function(
                runtime,
                "cnp_array_at",
                [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64)],
                ctypes.c_void_p,
            )
            bounds = self._function(
                runtime,
                "cnp_byte_bounds",
                [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.POINTER(ctypes.c_void_p),
                ],
                ctypes.c_int,
            )
            shares = self._function(
                runtime,
                "cnp_shares_memory",
                [ctypes.c_void_p, ctypes.c_void_p],
                ctypes.c_bool,
            )
            may_share = self._function(
                runtime,
                "cnp_may_share_memory",
                [ctypes.c_void_p, ctypes.c_void_p],
                ctypes.c_bool,
            )

            source = runtime.from_numpy(np.arange(8, dtype=np.int64))
            independent = stack.enter_context(
                runtime.from_numpy(np.arange(8, dtype=np.int64))
            )
            even_spec = _CnpSlice(0, 8, 2, True, True, True)
            odd_spec = _CnpSlice(1, 8, 2, True, True, True)
            reverse_spec = _CnpSlice(0, 0, -1, False, False, True)
            even = stack.enter_context(
                runtime._owned_result(
                    array_slice(source.pointer, 1, ctypes.byref(even_spec)),
                    "cnp_array_slice",
                )
            )
            odd = stack.enter_context(
                runtime._owned_result(
                    array_slice(source.pointer, 1, ctypes.byref(odd_spec)),
                    "cnp_array_slice",
                )
            )
            reverse = stack.enter_context(
                runtime._owned_result(
                    array_slice(source.pointer, 1, ctypes.byref(reverse_spec)),
                    "cnp_array_slice",
                )
            )
            source.close()

            low = ctypes.c_void_p()
            high = ctypes.c_void_p()
            self.assertEqual(0, bounds(reverse.pointer, ctypes.byref(low), ctypes.byref(high)))
            addresses = []
            for index in range(reverse.size):
                coordinate = (ctypes.c_int64 * 1)(index)
                addresses.append(int(array_at(reverse.pointer, coordinate)))
            self.assertEqual(min(addresses), low.value)
            self.assertEqual(max(addresses) + reverse.itemsize, high.value)

            runtime.dll.cnp_clear_error()
            numpy_base = np.arange(8)
            self.assertFalse(
                np.shares_memory(numpy_base[::2], numpy_base[1::2])
            )
            self.assertFalse(shares(even.pointer, odd.pointer))
            self.assertEqual(0, runtime.error_state().status)
            self.assertTrue(may_share(even.pointer, odd.pointer))
            self.assertTrue(shares(reverse.pointer, even.pointer))
            self.assertFalse(shares(reverse.pointer, independent.pointer))
            self.assertFalse(may_share(reverse.pointer, independent.pointer))

            for name, function in (
                ("cnp_shares_memory", shares),
                ("cnp_may_share_memory", may_share),
            ):
                with self.subTest(symbol=name):
                    runtime.dll.cnp_clear_error()
                    self.assertFalse(function(None, even.pointer))
                    self.assertEqual(name, runtime.error_state().function)

            runtime.dll.cnp_clear_error()
            null_low = ctypes.c_void_p(123)
            null_high = ctypes.c_void_p(456)
            self.assertNotEqual(
                0, bounds(None, ctypes.byref(null_low), ctypes.byref(null_high))
            )
            self.assertFalse(null_low.value)
            self.assertFalse(null_high.value)
            self.assertEqual("cnp_byte_bounds", runtime.error_state().function)

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
