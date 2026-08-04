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


class EinsumAliasSurfaceSemanticsTests(unittest.TestCase):
    SYMBOLS = frozenset(
        {
            "cnp_einsum_diag",
            "cnp_einsum_dot",
            "cnp_einsum_matmul",
            "cnp_einsum_matvec",
            "cnp_einsum_outer",
            "cnp_einsum_sum",
            "cnp_einsum_trace",
            "cnp_einsum_transpose",
        }
    )

    @staticmethod
    def _array_function(
        runtime: CnumpyRuntime, name: str, arity: int
    ) -> ctypes._CFuncPtr:
        function = getattr(runtime.dll, name)
        function.argtypes = [ctypes.c_void_p] * arity
        function.restype = ctypes.c_void_p
        return function

    @staticmethod
    def _scalar_function(
        runtime: CnumpyRuntime, name: str, arity: int
    ) -> ctypes._CFuncPtr:
        function = getattr(runtime.dll, name)
        function.argtypes = [ctypes.c_void_p] * arity
        function.restype = ctypes.c_double
        return function

    def test_array_aliases_match_numpy_einsum_dtype_values_and_lifetime(
        self,
    ) -> None:
        unary_cases = (
            (
                "cnp_einsum_diag",
                "ii->i",
                np.arange(9, dtype=np.int64).reshape(3, 3),
            ),
            (
                "cnp_einsum_transpose",
                "ij->ji",
                np.arange(6, dtype=np.float32).reshape(2, 3),
            ),
        )
        binary_cases = (
            (
                "cnp_einsum_outer",
                "i,j->ij",
                np.asarray([1 + 2j, 3 - 1j], dtype=np.complex64),
                np.asarray([2 - 1j, 0.5 + 3j], dtype=np.complex64),
            ),
            (
                "cnp_einsum_matmul",
                "ij,jk->ik",
                np.arange(6, dtype=np.float64).reshape(2, 3),
                np.arange(12, dtype=np.float64).reshape(3, 4),
            ),
            (
                "cnp_einsum_matvec",
                "ij,j->i",
                np.arange(6, dtype=np.float32).reshape(2, 3),
                np.asarray([1.0, -2.0, 0.5], dtype=np.float32),
            ),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as results:
            baseline = runtime.retained_bytes
            for name, subscripts, values in unary_cases:
                with self.subTest(symbol=name):
                    source = runtime.from_numpy(values)
                    try:
                        function = self._array_function(runtime, name, 1)
                        result = results.enter_context(
                            runtime._owned_result(function(source.pointer), name)
                        )
                    finally:
                        source.close()
                    assert_array_equivalent(
                        self,
                        result,
                        np.einsum(subscripts, values),
                        compare_contiguity=False,
                    )

            for name, subscripts, left_values, right_values in binary_cases:
                with self.subTest(symbol=name):
                    left = runtime.from_numpy(left_values)
                    right = runtime.from_numpy(right_values)
                    try:
                        function = self._array_function(runtime, name, 2)
                        result = results.enter_context(
                            runtime._owned_result(
                                function(left.pointer, right.pointer), name
                            )
                        )
                    finally:
                        right.close()
                        left.close()
                    assert_array_equivalent(
                        self,
                        result,
                        np.einsum(subscripts, left_values, right_values),
                        compare_contiguity=False,
                        rtol=3e-6,
                        atol=3e-6,
                    )

            results.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_scalar_aliases_match_real_numpy_einsum_projection(self) -> None:
        binary = (
            "cnp_einsum_dot",
            "i,i->",
            np.asarray([1.5, -2.0, 4.0]),
            np.asarray([2.0, 3.0, -0.5]),
        )
        unary_cases = (
            (
                "cnp_einsum_trace",
                "ii->",
                np.arange(9, dtype=np.float64).reshape(3, 3),
            ),
            (
                "cnp_einsum_sum",
                "ij->",
                np.arange(6, dtype=np.float32).reshape(2, 3),
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            name, subscripts, left_values, right_values = binary
            with runtime.from_numpy(left_values) as left, runtime.from_numpy(
                right_values
            ) as right:
                function = self._scalar_function(runtime, name, 2)
                self.assertAlmostEqual(
                    float(np.einsum(subscripts, left_values, right_values)),
                    function(left.pointer, right.pointer),
                    places=13,
                )

            for name, subscripts, values in unary_cases:
                with self.subTest(symbol=name), runtime.from_numpy(values) as source:
                    function = self._scalar_function(runtime, name, 1)
                    self.assertAlmostEqual(
                        float(np.einsum(subscripts, values)),
                        function(source.pointer),
                        places=6 if values.dtype == np.float32 else 13,
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_alias_errors_are_explicit_atomic_and_nonretaining(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            array_arities = {
                "cnp_einsum_diag": 1,
                "cnp_einsum_matmul": 2,
                "cnp_einsum_matvec": 2,
                "cnp_einsum_outer": 2,
                "cnp_einsum_transpose": 1,
            }
            scalar_arities = {
                "cnp_einsum_dot": 2,
                "cnp_einsum_sum": 1,
                "cnp_einsum_trace": 1,
            }
            for name, arity in array_arities.items():
                function = self._array_function(runtime, name, arity)
                runtime.dll.cnp_clear_error()
                self.assertFalse(function(*([None] * arity)))
                self.assertEqual(name, runtime.error_state().function)
                self.assertEqual(baseline, runtime.retained_bytes)
            for name, arity in scalar_arities.items():
                function = self._scalar_function(runtime, name, arity)
                runtime.dll.cnp_clear_error()
                self.assertEqual(0.0, function(*([None] * arity)))
                self.assertEqual(name, runtime.error_state().function)
                self.assertEqual(baseline, runtime.retained_bytes)

            with runtime.from_numpy(
                np.asarray([1 + 2j], dtype=np.complex128)
            ) as complex_value:
                dot = self._scalar_function(runtime, "cnp_einsum_dot", 2)
                runtime.dll.cnp_clear_error()
                self.assertEqual(
                    0.0, dot(complex_value.pointer, complex_value.pointer)
                )
                self.assertEqual(
                    "cnp_einsum_dot", runtime.error_state().function
                )

            with runtime.from_numpy(np.ones((2, 3))) as left, runtime.from_numpy(
                np.ones((2, 4))
            ) as right:
                active = runtime.retained_bytes
                matmul = self._array_function(
                    runtime, "cnp_einsum_matmul", 2
                )
                runtime.dll.cnp_clear_error()
                self.assertFalse(matmul(left.pointer, right.pointer))
                self.assertEqual(
                    "cnp_einsum_matmul", runtime.error_state().function
                )
                self.assertEqual(active, runtime.retained_bytes)
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
