from __future__ import annotations

import ctypes
from pathlib import Path
import unittest

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"


class AssertionSurfaceSemanticsTests(unittest.TestCase):
    SYMBOLS = frozenset(
        {
            "cnp_assert_allclose",
            "cnp_assert_array_almost_equal",
            "cnp_assert_array_equal",
        }
    )

    @staticmethod
    def _function(runtime: CnumpyRuntime, name: str, argtypes):
        function = getattr(runtime.dll, name)
        function.argtypes = argtypes
        function.restype = ctypes.c_bool
        return function

    @staticmethod
    def _numpy_assertion_succeeds(function, actual, desired, **kwargs) -> bool:
        try:
            function(actual, desired, **kwargs)
        except AssertionError:
            return False
        return True

    def test_assertion_predicates_match_numpy_125_values_shapes_and_nan(self) -> None:
        exact_cases = (
            (
                np.asarray([2**60 + 1, 2**60 + 3], dtype=np.int64),
                np.asarray([2**60 + 1, 2**60 + 3], dtype=np.int64),
            ),
            (
                np.asarray([2**60 + 1], dtype=np.int64),
                np.asarray([2**60 + 2], dtype=np.int64),
            ),
            (np.asarray([np.nan, np.inf]), np.asarray([np.nan, np.inf])),
            (np.asarray([1.0]), np.asarray([[1.0]])),
            (np.asarray(1.0), np.asarray([1.0, 1.0])),
        )
        almost_cases = (
            (np.asarray([0.0]), np.asarray([0.149]), 1),
            (np.asarray([0.0]), np.asarray([0.151]), 1),
            (np.asarray([np.nan]), np.asarray([np.nan]), 7),
            (np.asarray([1.0]), np.asarray([[1.0]]), 7),
        )
        close_cases = (
            (np.asarray([1.0, np.nan]), np.asarray([1.0 + 1e-7, np.nan])),
            (np.asarray([1.0]), np.asarray([[1.0]])),
            (np.asarray(1.0), np.asarray([1.0, 1.0])),
            (np.asarray([1.0]), np.asarray([1.01])),
        )

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            exact = self._function(
                runtime,
                "cnp_assert_array_equal",
                [ctypes.c_void_p, ctypes.c_void_p],
            )
            almost = self._function(
                runtime,
                "cnp_assert_array_almost_equal",
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int],
            )
            close = self._function(
                runtime,
                "cnp_assert_allclose",
                [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_double,
                    ctypes.c_double,
                ],
            )

            for actual, desired in exact_cases:
                expected = self._numpy_assertion_succeeds(
                    np.testing.assert_array_equal, actual, desired
                )
                with self.subTest(kind="equal", actual=actual, desired=desired), runtime.from_numpy(
                    actual
                ) as left, runtime.from_numpy(desired) as right:
                    runtime.dll.cnp_clear_error()
                    self.assertEqual(expected, bool(exact(left.pointer, right.pointer)))
                    self.assertEqual(0, runtime.error_state().status)

            for actual, desired, decimal in almost_cases:
                expected = self._numpy_assertion_succeeds(
                    np.testing.assert_array_almost_equal,
                    actual,
                    desired,
                    decimal=decimal,
                )
                with self.subTest(kind="almost", decimal=decimal), runtime.from_numpy(
                    actual
                ) as left, runtime.from_numpy(desired) as right:
                    runtime.dll.cnp_clear_error()
                    self.assertEqual(
                        expected, bool(almost(left.pointer, right.pointer, decimal))
                    )
                    self.assertEqual(0, runtime.error_state().status)

            for actual, desired in close_cases:
                expected = self._numpy_assertion_succeeds(
                    np.testing.assert_allclose,
                    actual,
                    desired,
                    rtol=1e-5,
                    atol=1e-8,
                )
                with self.subTest(kind="allclose", actual=actual, desired=desired), runtime.from_numpy(
                    actual
                ) as left, runtime.from_numpy(desired) as right:
                    runtime.dll.cnp_clear_error()
                    self.assertEqual(
                        expected,
                        bool(close(left.pointer, right.pointer, 1e-5, 1e-8)),
                    )
                    self.assertEqual(0, runtime.error_state().status)

            self.assertEqual(baseline, runtime.retained_bytes)

    def test_assertion_invalid_inputs_are_explicit_and_lifetime_neutral(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            cases = (
                (
                    "cnp_assert_array_equal",
                    [ctypes.c_void_p, ctypes.c_void_p],
                    (None, None),
                ),
                (
                    "cnp_assert_array_almost_equal",
                    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int],
                    (None, None, 7),
                ),
                (
                    "cnp_assert_allclose",
                    [
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                        ctypes.c_double,
                        ctypes.c_double,
                    ],
                    (None, None, 1e-5, 1e-8),
                ),
            )
            for name, argtypes, arguments in cases:
                with self.subTest(symbol=name):
                    function = self._function(runtime, name, argtypes)
                    runtime.dll.cnp_clear_error()
                    self.assertFalse(function(*arguments))
                    error = runtime.error_state()
                    self.assertNotEqual(0, error.status)
                    self.assertEqual(name, error.function)
                    self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
