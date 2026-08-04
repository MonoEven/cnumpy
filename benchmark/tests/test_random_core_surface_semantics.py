from __future__ import annotations

import ctypes
from contextlib import ExitStack
from pathlib import Path
import unittest
import warnings

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"


class RandomCoreSurfaceSemanticsTests(unittest.TestCase):
    SYMBOLS = frozenset(
        {
            "cnp_random_random",
            "cnp_random_uniform",
            "cnp_random_normal",
            "cnp_random_standard_normal",
            "cnp_random_integers",
            "cnp_random_randint",
            "cnp_random_binomial",
            "cnp_random_poisson",
            "cnp_random_exponential",
            "cnp_random_gamma",
            "cnp_random_beta",
        }
    )

    @staticmethod
    def _configure(runtime: CnumpyRuntime) -> dict[str, ctypes._CFuncPtr]:
        pointer = ctypes.POINTER(ctypes.c_int64)
        signatures = {
            "cnp_random_random": [ctypes.c_int, pointer],
            "cnp_random_uniform": [
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
                pointer,
            ],
            "cnp_random_normal": [
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
                pointer,
            ],
            "cnp_random_standard_normal": [ctypes.c_int, pointer],
            "cnp_random_integers": [
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int,
                pointer,
            ],
            "cnp_random_randint": [
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int,
                pointer,
            ],
            "cnp_random_binomial": [
                ctypes.c_int64,
                ctypes.c_double,
                ctypes.c_int,
                pointer,
            ],
            "cnp_random_poisson": [ctypes.c_double, ctypes.c_int, pointer],
            "cnp_random_exponential": [
                ctypes.c_double,
                ctypes.c_int,
                pointer,
            ],
            "cnp_random_gamma": [
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
                pointer,
            ],
            "cnp_random_beta": [
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
                pointer,
            ],
        }
        functions: dict[str, ctypes._CFuncPtr] = {}
        for name, argtypes in signatures.items():
            function = getattr(runtime.dll, name)
            function.argtypes = argtypes
            function.restype = ctypes.c_void_p
            functions[name] = function
        runtime.dll.cnp_random_seed.argtypes = [ctypes.c_uint64]
        runtime.dll.cnp_random_seed.restype = None
        return functions

    @staticmethod
    def _valid_calls(
        functions: dict[str, ctypes._CFuncPtr], shape: ctypes.Array,
    ) -> dict[str, tuple[ctypes._CFuncPtr, tuple[object, ...], np.ndarray]]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            references = {
                "cnp_random_random": np.random.random_sample((3, 4)),
                "cnp_random_uniform": np.random.uniform(-2.0, 3.0, (3, 4)),
                "cnp_random_normal": np.random.normal(1.0, 2.0, (3, 4)),
                "cnp_random_standard_normal": np.random.standard_normal((3, 4)),
                "cnp_random_integers": np.random.random_integers(2, 5, (3, 4)),
                "cnp_random_randint": np.random.randint(2, 5, (3, 4)),
                "cnp_random_binomial": np.random.binomial(7, 0.3, (3, 4)),
                "cnp_random_poisson": np.random.poisson(3.0, (3, 4)),
                "cnp_random_exponential": np.random.exponential(2.0, (3, 4)),
                "cnp_random_gamma": np.random.gamma(2.5, 1.5, (3, 4)),
                "cnp_random_beta": np.random.beta(2.0, 5.0, (3, 4)),
            }
        args = {
            "cnp_random_random": (2, shape),
            "cnp_random_uniform": (-2.0, 3.0, 2, shape),
            "cnp_random_normal": (1.0, 2.0, 2, shape),
            "cnp_random_standard_normal": (2, shape),
            "cnp_random_integers": (2, 5, 2, shape),
            "cnp_random_randint": (2, 5, 2, shape),
            "cnp_random_binomial": (7, 0.3, 2, shape),
            "cnp_random_poisson": (3.0, 2, shape),
            "cnp_random_exponential": (2.0, 2, shape),
            "cnp_random_gamma": (2.5, 1.5, 2, shape),
            "cnp_random_beta": (2.0, 5.0, 2, shape),
        }
        return {
            name: (functions[name], args[name], references[name])
            for name in functions
        }

    def test_valid_results_match_numpy_shape_dtype_support_and_replay(
        self,
    ) -> None:
        shape = (ctypes.c_int64 * 2)(3, 4)
        with CnumpyRuntime(DLL) as runtime:
            functions = self._configure(runtime)
            calls = self._valid_calls(functions, shape)
            baseline = runtime.retained_bytes
            for name, (function, args, expected) in calls.items():
                with self.subTest(symbol=name):
                    runtime.dll.cnp_random_seed(0x123456789ABCDEF0)
                    with runtime._owned_result(function(*args), name) as first:
                        first_values = first.to_numpy()
                        self.assertEqual(expected.shape, first_values.shape)
                        self.assertEqual(expected.dtype, first_values.dtype)
                    runtime.dll.cnp_random_seed(0x123456789ABCDEF0)
                    with runtime._owned_result(function(*args), name) as second:
                        np.testing.assert_array_equal(
                            first_values, second.to_numpy()
                        )
                    self.assertEqual(baseline, runtime.retained_bytes)

            domain_checks = {
                "cnp_random_random": lambda x: np.all((0 <= x) & (x < 1)),
                "cnp_random_uniform": lambda x: np.all((-2 <= x) & (x < 3)),
                "cnp_random_integers": lambda x: np.all((2 <= x) & (x <= 5)),
                "cnp_random_randint": lambda x: np.all((2 <= x) & (x < 5)),
                "cnp_random_binomial": lambda x: np.all((0 <= x) & (x <= 7)),
                "cnp_random_poisson": lambda x: np.all(x >= 0),
                "cnp_random_exponential": lambda x: np.all(x >= 0),
                "cnp_random_gamma": lambda x: np.all(x >= 0),
                "cnp_random_beta": lambda x: np.all((0 <= x) & (x <= 1)),
            }
            with ExitStack() as results:
                runtime.dll.cnp_random_seed(17)
                for name, check in domain_checks.items():
                    function, args, _ = calls[name]
                    result = results.enter_context(
                        runtime._owned_result(function(*args), name)
                    )
                    self.assertTrue(check(result.to_numpy()), name)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_invalid_parameters_and_shapes_surface_public_errors_atomically(
        self,
    ) -> None:
        shape = (ctypes.c_int64 * 1)(4)
        with CnumpyRuntime(DLL) as runtime:
            functions = self._configure(runtime)
            baseline = runtime.retained_bytes
            null_shape_args = {
                "cnp_random_random": (1, None),
                "cnp_random_uniform": (0.0, 1.0, 1, None),
                "cnp_random_normal": (0.0, 1.0, 1, None),
                "cnp_random_standard_normal": (1, None),
                "cnp_random_integers": (0, 1, 1, None),
                "cnp_random_randint": (0, 1, 1, None),
                "cnp_random_binomial": (1, 0.5, 1, None),
                "cnp_random_poisson": (1.0, 1, None),
                "cnp_random_exponential": (1.0, 1, None),
                "cnp_random_gamma": (1.0, 1.0, 1, None),
                "cnp_random_beta": (1.0, 1.0, 1, None),
            }
            for name, args in null_shape_args.items():
                with self.subTest(symbol=name, case="null_shape"):
                    runtime.dll.cnp_clear_error()
                    self.assertFalse(functions[name](*args))
                    error = runtime.error_state()
                    self.assertNotEqual(0, error.status)
                    self.assertEqual(name, error.function)
                    self.assertEqual(baseline, runtime.retained_bytes)

            invalid_parameters = {
                "cnp_random_uniform": (0.0, np.inf, 1, shape),
                "cnp_random_normal": (0.0, -1.0, 1, shape),
                "cnp_random_integers": (5, 4, 1, shape),
                "cnp_random_randint": (5, 5, 1, shape),
                "cnp_random_binomial": (3, 1.5, 1, shape),
                "cnp_random_poisson": (-1.0, 1, shape),
                "cnp_random_exponential": (-1.0, 1, shape),
                "cnp_random_gamma": (2.0, -1.0, 1, shape),
            }
            numpy_invalid = {
                "cnp_random_uniform": lambda: np.random.uniform(0.0, np.inf, 4),
                "cnp_random_normal": lambda: np.random.normal(0.0, -1.0, 4),
                "cnp_random_integers": lambda: np.random.random_integers(5, 4, 4),
                "cnp_random_randint": lambda: np.random.randint(5, 5, 4),
                "cnp_random_binomial": lambda: np.random.binomial(3, 1.5, 4),
                "cnp_random_poisson": lambda: np.random.poisson(-1.0, 4),
                "cnp_random_exponential": lambda: np.random.exponential(-1.0, 4),
                "cnp_random_gamma": lambda: np.random.gamma(2.0, -1.0, 4),
            }
            for name, args in invalid_parameters.items():
                with self.subTest(symbol=name, case="parameter"):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        with self.assertRaises((ValueError, OverflowError)):
                            numpy_invalid[name]()
                    runtime.dll.cnp_clear_error()
                    self.assertFalse(functions[name](*args))
                    error = runtime.error_state()
                    self.assertNotEqual(0, error.status)
                    self.assertEqual(name, error.function)
                    self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
