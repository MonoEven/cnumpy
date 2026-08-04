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


class EmathSurfaceSemanticsTests(unittest.TestCase):
    SYMBOLS = frozenset(
        {
            "cnp_emath_arccos",
            "cnp_emath_arcsin",
            "cnp_emath_arctanh",
            "cnp_emath_log",
            "cnp_emath_log10",
            "cnp_emath_log2",
            "cnp_emath_power",
            "cnp_emath_sqrt",
        }
    )

    @staticmethod
    def _unary(runtime: CnumpyRuntime, name: str):
        function = getattr(runtime.dll, name)
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_void_p
        return function

    def test_unary_emath_matches_numpy_125_domain_dtype_values_and_lifetime(
        self,
    ) -> None:
        cases = {
            "cnp_emath_sqrt": (
                np.asarray([-4.0, 0.0, 9.0], dtype=np.float32),
                np.asarray([0.0, 4.0, 9.0], dtype=np.float64),
                np.asarray([1.0 + 2.0j, -3.0 + 0.5j], dtype=np.complex128),
            ),
            "cnp_emath_log": (
                np.asarray([-4.0, 0.0, 9.0], dtype=np.float32),
                np.asarray([1.0, 4.0, 9.0], dtype=np.float64),
                np.asarray([1.0 + 2.0j, -3.0 + 0.5j], dtype=np.complex128),
            ),
            "cnp_emath_log10": (
                np.asarray([-4.0, 0.0, 100.0], dtype=np.float32),
                np.asarray([1.0, 10.0, 100.0], dtype=np.float64),
                np.asarray([1.0 + 2.0j, -3.0 + 0.5j], dtype=np.complex128),
            ),
            "cnp_emath_log2": (
                np.asarray([-4.0, 0.0, 8.0], dtype=np.float32),
                np.asarray([1.0, 4.0, 8.0], dtype=np.float64),
                np.asarray([1.0 + 2.0j, -3.0 + 0.5j], dtype=np.complex128),
            ),
            "cnp_emath_arcsin": (
                np.asarray([-2.0, 0.0, 2.0], dtype=np.float32),
                np.asarray([-0.5, 0.0, 0.5], dtype=np.float64),
                np.asarray([1.0 + 2.0j, -0.25 + 0.5j], dtype=np.complex128),
            ),
            "cnp_emath_arccos": (
                np.asarray([-2.0, 0.0, 2.0], dtype=np.float32),
                np.asarray([-0.5, 0.0, 0.5], dtype=np.float64),
                np.asarray([1.0 + 2.0j, -0.25 + 0.5j], dtype=np.complex128),
            ),
            "cnp_emath_arctanh": (
                np.asarray([-2.0, 0.0, 2.0], dtype=np.float32),
                np.asarray([-0.5, 0.0, 0.5], dtype=np.float64),
                np.asarray([1.0 + 2.0j, -0.25 + 0.5j], dtype=np.complex128),
            ),
        }
        with CnumpyRuntime(DLL) as runtime, ExitStack() as results:
            baseline = runtime.retained_bytes
            for name, inputs in cases.items():
                oracle = getattr(np.emath, name.removeprefix("cnp_emath_"))
                function = self._unary(runtime, name)
                for values in inputs:
                    with self.subTest(symbol=name, dtype=values.dtype, values=values):
                        with np.errstate(all="ignore"):
                            expected = np.asarray(oracle(values))
                        source = runtime.from_numpy(values)
                        try:
                            pointer = function(source.pointer)
                            result = results.enter_context(
                                runtime._owned_result(pointer, name)
                            )
                        finally:
                            source.close()
                        assert_array_equivalent(
                            self,
                            result,
                            expected,
                            compare_contiguity=False,
                            rtol=4e-6 if expected.dtype in (np.float32, np.complex64) else 3e-13,
                            atol=4e-6 if expected.dtype in (np.float32, np.complex64) else 3e-13,
                        )

            results.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_emath_power_matches_numpy_broadcast_dtype_and_source_release(
        self,
    ) -> None:
        cases = (
            (
                np.asarray([[-1.0], [4.0]], dtype=np.float32),
                np.asarray([[0.5, 2.0, -1.0]], dtype=np.float32),
            ),
            (
                np.asarray([[1.0], [4.0]], dtype=np.float32),
                np.asarray([[0.5, 2.0, -1.0]], dtype=np.float32),
            ),
            (
                np.asarray([-1, 4], dtype=np.int64),
                np.asarray([2, 3], dtype=np.int64),
            ),
            (
                np.asarray([1.0 + 2.0j], dtype=np.complex64),
                np.asarray([2], dtype=np.int64),
            ),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as results:
            baseline = runtime.retained_bytes
            power = runtime.dll.cnp_emath_power
            power.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            power.restype = ctypes.c_void_p
            for base_values, exponent_values in cases:
                with self.subTest(
                    base_dtype=base_values.dtype,
                    exponent_dtype=exponent_values.dtype,
                ):
                    with np.errstate(all="ignore"):
                        expected = np.emath.power(base_values, exponent_values)
                    base = runtime.from_numpy(base_values)
                    exponent = runtime.from_numpy(exponent_values)
                    try:
                        pointer = power(base.pointer, exponent.pointer)
                        result = results.enter_context(
                            runtime._owned_result(pointer, "cnp_emath_power")
                        )
                    finally:
                        exponent.close()
                        base.close()
                    assert_array_equivalent(
                        self,
                        result,
                        expected,
                        compare_contiguity=False,
                        rtol=5e-6 if expected.dtype in (np.float32, np.complex64) else 3e-13,
                        atol=5e-6 if expected.dtype in (np.float32, np.complex64) else 3e-13,
                    )

            results.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_emath_errors_are_explicit_atomic_and_nonretaining(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for name in sorted(self.SYMBOLS - {"cnp_emath_power"}):
                function = self._unary(runtime, name)
                runtime.dll.cnp_clear_error()
                self.assertFalse(function(None))
                self.assertEqual(name, runtime.error_state().function)
                self.assertEqual(baseline, runtime.retained_bytes)

            power = runtime.dll.cnp_emath_power
            power.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            power.restype = ctypes.c_void_p
            runtime.dll.cnp_clear_error()
            self.assertFalse(power(None, None))
            self.assertEqual("cnp_emath_power", runtime.error_state().function)

            with runtime.from_numpy(np.ones((2, 3))) as base, runtime.from_numpy(
                np.ones((4,))
            ) as exponent:
                active = runtime.retained_bytes
                runtime.dll.cnp_clear_error()
                self.assertFalse(power(base.pointer, exponent.pointer))
                self.assertEqual("cnp_emath_power", runtime.error_state().function)
                self.assertEqual(active, runtime.retained_bytes)
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
