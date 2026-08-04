from __future__ import annotations

import ctypes
from pathlib import Path
import unittest

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"


class RandomExtendedSurfaceSemanticsTests(unittest.TestCase):
    SYMBOLS = frozenset(
        {
            "cnp_random_logseries",
            "cnp_random_negative_binomial",
            "cnp_random_pareto",
            "cnp_random_power",
            "cnp_random_rayleigh",
            "cnp_random_standard_cauchy",
            "cnp_random_standard_t",
            "cnp_random_triangular",
            "cnp_random_vonmises",
            "cnp_random_noncentral_chisquare",
            "cnp_random_noncentral_f",
            "cnp_random_f",
            "cnp_random_laplace",
            "cnp_random_logistic",
            "cnp_random_gumbel",
            "cnp_random_dirichlet",
            "cnp_random_multinomial",
            "cnp_random_weibull",
        }
    )

    @staticmethod
    def _configure(runtime: CnumpyRuntime) -> dict[str, ctypes._CFuncPtr]:
        shape_pointer = ctypes.POINTER(ctypes.c_int64)
        double_pointer = ctypes.POINTER(ctypes.c_double)
        signatures = {
            "cnp_random_logseries": [ctypes.c_double, ctypes.c_int, shape_pointer],
            "cnp_random_negative_binomial": [
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
                shape_pointer,
            ],
            "cnp_random_pareto": [ctypes.c_double, ctypes.c_int, shape_pointer],
            "cnp_random_power": [ctypes.c_double, ctypes.c_int, shape_pointer],
            "cnp_random_rayleigh": [ctypes.c_double, ctypes.c_int, shape_pointer],
            "cnp_random_standard_cauchy": [ctypes.c_int, shape_pointer],
            "cnp_random_standard_t": [ctypes.c_double, ctypes.c_int, shape_pointer],
            "cnp_random_triangular": [
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
                shape_pointer,
            ],
            "cnp_random_vonmises": [
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
                shape_pointer,
            ],
            "cnp_random_noncentral_chisquare": [
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
                shape_pointer,
            ],
            "cnp_random_noncentral_f": [
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
                shape_pointer,
            ],
            "cnp_random_f": [
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
                shape_pointer,
            ],
            "cnp_random_laplace": [
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
                shape_pointer,
            ],
            "cnp_random_logistic": [
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
                shape_pointer,
            ],
            "cnp_random_gumbel": [
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
                shape_pointer,
            ],
            "cnp_random_dirichlet": [
                double_pointer,
                ctypes.c_int,
                ctypes.c_int64,
            ],
            "cnp_random_multinomial": [
                ctypes.c_int64,
                double_pointer,
                ctypes.c_int,
                ctypes.c_int64,
            ],
            "cnp_random_weibull": [ctypes.c_double, ctypes.c_int, shape_pointer],
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
    def _calls(
        functions: dict[str, ctypes._CFuncPtr],
        shape: ctypes.Array,
        alpha: ctypes.Array,
        probabilities: ctypes.Array,
    ) -> dict[str, tuple[ctypes._CFuncPtr, tuple[object, ...], np.ndarray]]:
        references = {
            "cnp_random_logseries": np.random.logseries(0.4, (2, 3)),
            "cnp_random_negative_binomial": np.random.negative_binomial(2.5, 0.4, (2, 3)),
            "cnp_random_pareto": np.random.pareto(3.0, (2, 3)),
            "cnp_random_power": np.random.power(2.0, (2, 3)),
            "cnp_random_rayleigh": np.random.rayleigh(1.5, (2, 3)),
            "cnp_random_standard_cauchy": np.random.standard_cauchy((2, 3)),
            "cnp_random_standard_t": np.random.standard_t(5.0, (2, 3)),
            "cnp_random_triangular": np.random.triangular(-2.0, 0.5, 4.0, (2, 3)),
            "cnp_random_vonmises": np.random.vonmises(0.5, 2.0, (2, 3)),
            "cnp_random_noncentral_chisquare": np.random.noncentral_chisquare(3.0, 1.5, (2, 3)),
            "cnp_random_noncentral_f": np.random.noncentral_f(4.0, 6.0, 1.5, (2, 3)),
            "cnp_random_f": np.random.f(4.0, 6.0, (2, 3)),
            "cnp_random_laplace": np.random.laplace(0.5, 1.5, (2, 3)),
            "cnp_random_logistic": np.random.logistic(0.5, 1.5, (2, 3)),
            "cnp_random_gumbel": np.random.gumbel(0.5, 1.5, (2, 3)),
            "cnp_random_dirichlet": np.random.dirichlet([1.0, 2.0, 3.0], 4),
            "cnp_random_multinomial": np.random.multinomial(7, [0.2, 0.3, 0.5], 4),
            "cnp_random_weibull": np.random.weibull(2.0, (2, 3)),
        }
        args = {
            "cnp_random_logseries": (0.4, 2, shape),
            "cnp_random_negative_binomial": (2.5, 0.4, 2, shape),
            "cnp_random_pareto": (3.0, 2, shape),
            "cnp_random_power": (2.0, 2, shape),
            "cnp_random_rayleigh": (1.5, 2, shape),
            "cnp_random_standard_cauchy": (2, shape),
            "cnp_random_standard_t": (5.0, 2, shape),
            "cnp_random_triangular": (-2.0, 0.5, 4.0, 2, shape),
            "cnp_random_vonmises": (0.5, 2.0, 2, shape),
            "cnp_random_noncentral_chisquare": (3.0, 1.5, 2, shape),
            "cnp_random_noncentral_f": (4.0, 6.0, 1.5, 2, shape),
            "cnp_random_f": (4.0, 6.0, 2, shape),
            "cnp_random_laplace": (0.5, 1.5, 2, shape),
            "cnp_random_logistic": (0.5, 1.5, 2, shape),
            "cnp_random_gumbel": (0.5, 1.5, 2, shape),
            "cnp_random_dirichlet": (alpha, 3, 4),
            "cnp_random_multinomial": (7, probabilities, 3, 4),
            "cnp_random_weibull": (2.0, 2, shape),
        }
        return {
            name: (functions[name], args[name], references[name])
            for name in functions
        }

    def test_valid_results_match_numpy_shape_dtype_support_and_replay(self) -> None:
        shape = (ctypes.c_int64 * 2)(2, 3)
        alpha = (ctypes.c_double * 3)(1.0, 2.0, 3.0)
        probabilities = (ctypes.c_double * 3)(0.2, 0.3, 0.5)
        with CnumpyRuntime(DLL) as runtime:
            functions = self._configure(runtime)
            calls = self._calls(functions, shape, alpha, probabilities)
            baseline = runtime.retained_bytes
            for name, (function, args, expected) in calls.items():
                with self.subTest(symbol=name):
                    runtime.dll.cnp_random_seed(0xA55A1234)
                    with runtime._owned_result(function(*args), name) as first:
                        values = first.to_numpy()
                        self.assertEqual(expected.shape, values.shape)
                        self.assertEqual(expected.dtype, values.dtype)
                    runtime.dll.cnp_random_seed(0xA55A1234)
                    with runtime._owned_result(function(*args), name) as second:
                        np.testing.assert_array_equal(values, second.to_numpy())
                    self.assertEqual(baseline, runtime.retained_bytes)

            runtime.dll.cnp_random_seed(41)
            with runtime._owned_result(
                functions["cnp_random_dirichlet"](alpha, 3, 4),
                "cnp_random_dirichlet",
            ) as result:
                np.testing.assert_allclose(result.to_numpy().sum(axis=1), 1.0)
            with runtime._owned_result(
                functions["cnp_random_multinomial"](7, probabilities, 3, 4),
                "cnp_random_multinomial",
            ) as result:
                np.testing.assert_array_equal(result.to_numpy().sum(axis=1), 7)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_invalid_parameters_and_shapes_are_explicit_and_nonretaining(self) -> None:
        shape = (ctypes.c_int64 * 1)(4)
        alpha_bad = (ctypes.c_double * 3)(1.0, 0.0, 2.0)
        probabilities_bad = (ctypes.c_double * 3)(0.8, 0.5, 0.1)
        with CnumpyRuntime(DLL) as runtime:
            functions = self._configure(runtime)
            baseline = runtime.retained_bytes

            null_shape_args = {
                "cnp_random_logseries": (0.4, 1, None),
                "cnp_random_negative_binomial": (2.0, 0.4, 1, None),
                "cnp_random_pareto": (2.0, 1, None),
                "cnp_random_power": (2.0, 1, None),
                "cnp_random_rayleigh": (1.0, 1, None),
                "cnp_random_standard_cauchy": (1, None),
                "cnp_random_standard_t": (2.0, 1, None),
                "cnp_random_triangular": (0.0, 0.5, 1.0, 1, None),
                "cnp_random_vonmises": (0.0, 1.0, 1, None),
                "cnp_random_noncentral_chisquare": (2.0, 1.0, 1, None),
                "cnp_random_noncentral_f": (2.0, 3.0, 1.0, 1, None),
                "cnp_random_f": (2.0, 3.0, 1, None),
                "cnp_random_laplace": (0.0, 1.0, 1, None),
                "cnp_random_logistic": (0.0, 1.0, 1, None),
                "cnp_random_gumbel": (0.0, 1.0, 1, None),
                "cnp_random_weibull": (2.0, 1, None),
            }
            for name, args in null_shape_args.items():
                with self.subTest(symbol=name, case="shape"):
                    runtime.dll.cnp_clear_error()
                    pointer = functions[name](*args)
                    if pointer:
                        runtime.dll.cnp_array_decref(pointer)
                    self.assertFalse(pointer)
                    self.assertEqual(name, runtime.error_state().function)
                    self.assertEqual(baseline, runtime.retained_bytes)

            invalid_calls = {
                "cnp_random_logseries": (1.0, 1, shape),
                "cnp_random_negative_binomial": (0.0, 0.4, 1, shape),
                "cnp_random_pareto": (0.0, 1, shape),
                "cnp_random_power": (0.0, 1, shape),
                "cnp_random_rayleigh": (-1.0, 1, shape),
                "cnp_random_standard_t": (0.0, 1, shape),
                "cnp_random_triangular": (1.0, 0.5, 0.0, 1, shape),
                "cnp_random_vonmises": (0.0, -1.0, 1, shape),
                "cnp_random_noncentral_chisquare": (0.0, 1.0, 1, shape),
                "cnp_random_noncentral_f": (2.0, 0.0, 1.0, 1, shape),
                "cnp_random_f": (0.0, 2.0, 1, shape),
                "cnp_random_laplace": (0.0, -1.0, 1, shape),
                "cnp_random_logistic": (0.0, -1.0, 1, shape),
                "cnp_random_gumbel": (0.0, -1.0, 1, shape),
                "cnp_random_dirichlet": (alpha_bad, 3, 4),
                "cnp_random_multinomial": (7, probabilities_bad, 3, 4),
                "cnp_random_weibull": (-1.0, 1, shape),
            }
            numpy_invalid = {
                "cnp_random_logseries": lambda: np.random.logseries(1.0, 4),
                "cnp_random_negative_binomial": lambda: np.random.negative_binomial(0.0, 0.4, 4),
                "cnp_random_pareto": lambda: np.random.pareto(0.0, 4),
                "cnp_random_power": lambda: np.random.power(0.0, 4),
                "cnp_random_rayleigh": lambda: np.random.rayleigh(-1.0, 4),
                "cnp_random_standard_t": lambda: np.random.standard_t(0.0, 4),
                "cnp_random_triangular": lambda: np.random.triangular(1.0, 0.5, 0.0, 4),
                "cnp_random_vonmises": lambda: np.random.vonmises(0.0, -1.0, 4),
                "cnp_random_noncentral_chisquare": lambda: np.random.noncentral_chisquare(0.0, 1.0, 4),
                "cnp_random_noncentral_f": lambda: np.random.noncentral_f(2.0, 0.0, 1.0, 4),
                "cnp_random_f": lambda: np.random.f(0.0, 2.0, 4),
                "cnp_random_laplace": lambda: np.random.laplace(0.0, -1.0, 4),
                "cnp_random_logistic": lambda: np.random.logistic(0.0, -1.0, 4),
                "cnp_random_gumbel": lambda: np.random.gumbel(0.0, -1.0, 4),
                "cnp_random_dirichlet": lambda: np.random.dirichlet([1.0, 0.0, 2.0], 4),
                "cnp_random_multinomial": lambda: np.random.multinomial(7, [0.8, 0.5, 0.1], 4),
                "cnp_random_weibull": lambda: np.random.weibull(-1.0, 4),
            }
            for name, args in invalid_calls.items():
                with self.subTest(symbol=name, case="parameter"):
                    with self.assertRaises(ValueError):
                        numpy_invalid[name]()
                    runtime.dll.cnp_clear_error()
                    pointer = functions[name](*args)
                    if pointer:
                        runtime.dll.cnp_array_decref(pointer)
                    self.assertFalse(pointer)
                    error = runtime.error_state()
                    self.assertNotEqual(0, error.status)
                    self.assertEqual(name, error.function)
                    self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
