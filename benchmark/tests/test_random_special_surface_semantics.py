from __future__ import annotations

import ctypes
from pathlib import Path
import unittest

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"


class RandomSpecialSurfaceSemanticsTests(unittest.TestCase):
    SYMBOLS = frozenset(
        {
            "cnp_random_bytes",
            "cnp_random_bytes_free",
            "cnp_random_chisquare",
            "cnp_random_geometric",
            "cnp_random_hypergeometric",
            "cnp_random_multivariate_normal",
            "cnp_random_wald",
            "cnp_random_zipf",
        }
    )

    @staticmethod
    def _configure(runtime: CnumpyRuntime) -> dict[str, ctypes._CFuncPtr]:
        shape_pointer = ctypes.POINTER(ctypes.c_int64)
        signatures = {
            "cnp_random_chisquare": [ctypes.c_double, ctypes.c_int, shape_pointer],
            "cnp_random_geometric": [ctypes.c_double, ctypes.c_int, shape_pointer],
            "cnp_random_zipf": [ctypes.c_double, ctypes.c_int, shape_pointer],
            "cnp_random_wald": [
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
                shape_pointer,
            ],
            "cnp_random_hypergeometric": [
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int,
                shape_pointer,
            ],
            "cnp_random_multivariate_normal": [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int64,
            ],
        }
        functions: dict[str, ctypes._CFuncPtr] = {}
        for name, argtypes in signatures.items():
            function = getattr(runtime.dll, name)
            function.argtypes = argtypes
            function.restype = ctypes.c_void_p
            functions[name] = function
        runtime.dll.cnp_random_bytes.argtypes = [ctypes.c_int64]
        runtime.dll.cnp_random_bytes.restype = ctypes.c_void_p
        runtime.dll.cnp_random_bytes_free.argtypes = [ctypes.c_void_p]
        runtime.dll.cnp_random_bytes_free.restype = None
        runtime.dll.cnp_random_seed.argtypes = [ctypes.c_uint64]
        runtime.dll.cnp_random_seed.restype = None
        return functions

    def test_array_results_match_numpy_shape_dtype_support_and_replay(self) -> None:
        shape = (ctypes.c_int64 * 2)(2, 3)
        with CnumpyRuntime(DLL) as runtime:
            functions = self._configure(runtime)
            baseline = runtime.retained_bytes
            cases = {
                "cnp_random_chisquare": (
                    (3.0, 2, shape), np.random.chisquare(3.0, (2, 3))
                ),
                "cnp_random_geometric": (
                    (0.4, 2, shape), np.random.geometric(0.4, (2, 3))
                ),
                "cnp_random_zipf": (
                    (2.5, 2, shape), np.random.zipf(2.5, (2, 3))
                ),
                "cnp_random_wald": (
                    (2.0, 3.0, 2, shape), np.random.wald(2.0, 3.0, (2, 3))
                ),
                "cnp_random_hypergeometric": (
                    (7, 5, 4, 2, shape),
                    np.random.hypergeometric(7, 5, 4, (2, 3)),
                ),
            }
            for name, (args, expected) in cases.items():
                with self.subTest(symbol=name):
                    runtime.dll.cnp_random_seed(1977)
                    with runtime._owned_result(
                        functions[name](*args), name
                    ) as first:
                        values = first.to_numpy()
                        self.assertEqual(expected.shape, values.shape)
                        self.assertEqual(expected.dtype, values.dtype)
                    runtime.dll.cnp_random_seed(1977)
                    with runtime._owned_result(
                        functions[name](*args), name
                    ) as second:
                        np.testing.assert_array_equal(values, second.to_numpy())
                    self.assertEqual(baseline, runtime.retained_bytes)

            mean_values = np.asarray([1.0, -1.0], dtype=np.float32)
            covariance_values = np.asarray(
                [[2.0, 0.5], [0.5, 1.0]], dtype=np.float32
            )
            with runtime.from_numpy(mean_values) as mean, runtime.from_numpy(
                covariance_values
            ) as covariance:
                function = functions["cnp_random_multivariate_normal"]
                runtime.dll.cnp_random_seed(991)
                with runtime._owned_result(
                    function(mean.pointer, covariance.pointer, 4),
                    "cnp_random_multivariate_normal",
                ) as first:
                    values = first.to_numpy()
                    expected = np.random.multivariate_normal(
                        mean_values, covariance_values, 4
                    )
                    self.assertEqual(expected.shape, values.shape)
                    self.assertEqual(expected.dtype, values.dtype)
                runtime.dll.cnp_random_seed(991)
                with runtime._owned_result(
                    function(mean.pointer, covariance.pointer, 4),
                    "cnp_random_multivariate_normal",
                ) as second:
                    np.testing.assert_array_equal(values, second.to_numpy())
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_random_bytes_are_seeded_owned_and_support_empty_results(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            self._configure(runtime)
            baseline = runtime.retained_bytes
            runtime.dll.cnp_random_seed(77)
            first = runtime.dll.cnp_random_bytes(16)
            self.assertTrue(first)
            first_values = ctypes.string_at(first, 16)
            runtime.dll.cnp_random_bytes_free(first)
            self.assertEqual(baseline, runtime.retained_bytes)

            runtime.dll.cnp_random_seed(77)
            second = runtime.dll.cnp_random_bytes(16)
            self.assertEqual(first_values, ctypes.string_at(second, 16))
            runtime.dll.cnp_random_bytes_free(second)

            empty = runtime.dll.cnp_random_bytes(0)
            self.assertTrue(empty)
            self.assertEqual(np.random.bytes(0), ctypes.string_at(empty, 0))
            runtime.dll.cnp_random_bytes_free(empty)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_errors_are_explicit_atomic_and_use_public_labels(self) -> None:
        shape = (ctypes.c_int64 * 1)(4)
        with CnumpyRuntime(DLL) as runtime:
            functions = self._configure(runtime)
            baseline = runtime.retained_bytes
            invalid = {
                "cnp_random_chisquare": ((0.0, 1, shape), lambda: np.random.chisquare(0.0, 4)),
                "cnp_random_geometric": ((0.0, 1, shape), lambda: np.random.geometric(0.0, 4)),
                "cnp_random_zipf": ((1.0, 1, shape), lambda: np.random.zipf(1.0, 4)),
                "cnp_random_wald": ((0.0, 1.0, 1, shape), lambda: np.random.wald(0.0, 1.0, 4)),
                "cnp_random_hypergeometric": (
                    (2, 3, 6, 1, shape),
                    lambda: np.random.hypergeometric(2, 3, 6, 4),
                ),
            }
            for name, (args, numpy_call) in invalid.items():
                with self.subTest(symbol=name, case="parameter"):
                    with self.assertRaises(ValueError):
                        numpy_call()
                    runtime.dll.cnp_clear_error()
                    pointer = functions[name](*args)
                    if pointer:
                        runtime.dll.cnp_array_decref(pointer)
                    self.assertFalse(pointer)
                    self.assertEqual(name, runtime.error_state().function)
                    self.assertEqual(baseline, runtime.retained_bytes)

            for name, args in {
                "cnp_random_chisquare": (2.0, 1, None),
                "cnp_random_geometric": (0.5, 1, None),
                "cnp_random_zipf": (2.0, 1, None),
                "cnp_random_wald": (1.0, 1.0, 1, None),
                "cnp_random_hypergeometric": (2, 3, 2, 1, None),
            }.items():
                with self.subTest(symbol=name, case="shape"):
                    runtime.dll.cnp_clear_error()
                    try:
                        pointer = functions[name](*args)
                    except OSError:
                        pointer = None
                        self.fail(f"{name} accessed a null shape")
                    if pointer:
                        runtime.dll.cnp_array_decref(pointer)
                    self.assertFalse(pointer)
                    self.assertEqual(name, runtime.error_state().function)
                    self.assertEqual(baseline, runtime.retained_bytes)

            function = functions["cnp_random_multivariate_normal"]
            runtime.dll.cnp_clear_error()
            self.assertFalse(function(None, None, 1))
            self.assertEqual(
                "cnp_random_multivariate_normal",
                runtime.error_state().function,
            )
            runtime.dll.cnp_clear_error()
            self.assertFalse(runtime.dll.cnp_random_bytes(-1))
            self.assertEqual("cnp_random_bytes", runtime.error_state().function)
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
