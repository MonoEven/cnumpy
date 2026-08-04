from __future__ import annotations

import ctypes
from contextlib import ExitStack
from pathlib import Path
import unittest
import warnings

import numpy as np
import numpy.matlib as npmat

from compat.cnumpy_ctypes import CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"


class MatrixSurfaceSemanticsTests(unittest.TestCase):
    SYMBOLS = frozenset(
        {
            "cnp_mat",
            "cnp_matlib_eye",
            "cnp_matlib_ones",
            "cnp_matlib_repmat",
            "cnp_matlib_zeros",
        }
    )

    @staticmethod
    def _function(runtime: CnumpyRuntime, name: str, argtypes, restype):
        function = getattr(runtime.dll, name)
        function.argtypes = argtypes
        function.restype = restype
        return function

    def test_mat_and_repmat_match_numpy_shapes_exact_values_and_lifetimes(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            mat = self._function(
                runtime, "cnp_mat", [ctypes.c_void_p], ctypes.c_void_p
            )
            repmat = self._function(
                runtime,
                "cnp_matlib_repmat",
                [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64],
                ctypes.c_void_p,
            )

            inputs = (
                np.asarray(2**60 + 3, dtype=np.int64),
                np.asarray([2**60 + 1, 2**60 + 3], dtype=np.int64),
                np.asarray(
                    [[2**60 + 1, 2**60 + 2],
                     [2**60 + 3, 2**60 + 4]],
                    dtype=np.int64,
                ).T,
            )
            for values in inputs:
                with self.subTest(shape=values.shape), runtime.from_numpy(
                    values
                ) as source:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", PendingDeprecationWarning)
                        expected = np.asarray(np.mat(values))
                    result = stack.enter_context(
                        runtime._owned_result(mat(source.pointer), "cnp_mat")
                    )
                    np.testing.assert_array_equal(
                        expected, result.to_numpy(), strict=True
                    )

            source_values = np.asarray(
                [2**60 + 1, -(2**60) + 7, 19], dtype=np.int64
            )
            with runtime.from_numpy(source_values) as source:
                result = stack.enter_context(
                    runtime._owned_result(
                        repmat(source.pointer, 2, 3), "cnp_matlib_repmat"
                    )
                )
            expected = np.asarray(npmat.repmat(source_values, 2, 3))
            np.testing.assert_array_equal(expected, result.to_numpy(), strict=True)

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)


    def test_matlib_constructors_match_numpy_values_dtypes_and_release(self) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            eye = self._function(
                runtime,
                "cnp_matlib_eye",
                [ctypes.c_int64, ctypes.c_int64, ctypes.c_int],
                ctypes.c_void_p,
            )
            matrix_eye = stack.enter_context(
                runtime._owned_result(eye(3, 5, -1), "cnp_matlib_eye")
            )
            np.testing.assert_array_equal(
                np.asarray(npmat.eye(3, 5, -1)),
                matrix_eye.to_numpy(),
                strict=True,
            )

            for name, oracle in (
                ("cnp_matlib_ones", npmat.ones),
                ("cnp_matlib_zeros", npmat.zeros),
            ):
                function = self._function(
                    runtime,
                    name,
                    [ctypes.c_int64, ctypes.c_int64],
                    ctypes.c_void_p,
                )
                result = stack.enter_context(
                    runtime._owned_result(function(2, 4), name)
                )
                np.testing.assert_array_equal(
                    np.asarray(oracle((2, 4))), result.to_numpy(), strict=True
                )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)


    def test_invalid_matrix_requests_are_explicit_atomic_and_nonretaining(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            calls = (
                (
                    "cnp_mat",
                    [ctypes.c_void_p],
                    (None,),
                ),
                (
                    "cnp_matlib_repmat",
                    [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64],
                    (None, 2, 3),
                ),
                (
                    "cnp_matlib_eye",
                    [ctypes.c_int64, ctypes.c_int64, ctypes.c_int],
                    (-1, 3, 0),
                ),
                (
                    "cnp_matlib_ones",
                    [ctypes.c_int64, ctypes.c_int64],
                    (-1, 3),
                ),
                (
                    "cnp_matlib_zeros",
                    [ctypes.c_int64, ctypes.c_int64],
                    (3, -1),
                ),
            )
            for name, argtypes, arguments in calls:
                with self.subTest(symbol=name):
                    function = self._function(
                        runtime, name, argtypes, ctypes.c_void_p
                    )
                    runtime.dll.cnp_clear_error()
                    pointer = function(*arguments)
                    if pointer:
                        runtime._owned_result(pointer, name).close()
                    self.assertFalse(pointer)
                    self.assertEqual(name, runtime.error_state().function)
                    self.assertEqual(baseline, runtime.retained_bytes)

            with runtime.from_numpy(
                np.zeros((1, 1, 1), dtype=np.float64)
            ) as source:
                repmat = self._function(
                    runtime,
                    "cnp_matlib_repmat",
                    [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64],
                    ctypes.c_void_p,
                )
                runtime.dll.cnp_clear_error()
                self.assertFalse(repmat(source.pointer, 2, 3))
                self.assertEqual(
                    "cnp_matlib_repmat", runtime.error_state().function
                )
            self.assertEqual(baseline, runtime.retained_bytes)


class MatlibRandomSemanticsTests(unittest.TestCase):
    @staticmethod
    def _bind(runtime: CnumpyRuntime) -> None:
        runtime.dll.cnp_random_seed.argtypes = [ctypes.c_uint64]
        runtime.dll.cnp_random_seed.restype = None
        for name in ("cnp_matlib_rand", "cnp_matlib_randn"):
            function = getattr(runtime.dll, name)
            function.argtypes = [ctypes.c_int64, ctypes.c_int64]
            function.restype = ctypes.c_void_p

    def _sample(
        self,
        runtime: CnumpyRuntime,
        name: str,
        rows: int,
        columns: int,
    ) -> np.ndarray:
        pointer = getattr(runtime.dll, name)(rows, columns)
        with runtime._owned_result(pointer, name) as result:
            self.assertEqual((rows, columns), result.shape)
            self.assertEqual(np.dtype(np.float64), result.numpy_dtype)
            return result.to_numpy()

    def test_random_matrix_aliases_are_seeded_statistical_and_nonretaining(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for name in ("cnp_matlib_rand", "cnp_matlib_randn"):
                with self.subTest(symbol=name):
                    runtime.dll.cnp_random_seed(0x1234_5678_9ABC_DEF0)
                    first = self._sample(runtime, name, 64, 32)
                    runtime.dll.cnp_random_seed(0x1234_5678_9ABC_DEF0)
                    second = self._sample(runtime, name, 64, 32)
                    np.testing.assert_array_equal(first, second)

            runtime.dll.cnp_random_seed(20260804)
            uniform = self._sample(runtime, "cnp_matlib_rand", 512, 512)
            self.assertTrue(np.all((uniform >= 0.0) & (uniform < 1.0)))
            self.assertLess(abs(float(uniform.mean()) - 0.5), 0.005)

            runtime.dll.cnp_random_seed(20260804)
            normal = self._sample(runtime, "cnp_matlib_randn", 512, 512)
            self.assertLess(abs(float(normal.mean())), 0.01)
            self.assertLess(abs(float(normal.std()) - 1.0), 0.01)

            for name in ("cnp_matlib_rand", "cnp_matlib_randn"):
                empty = self._sample(runtime, name, 0, 7)
                self.assertEqual(0, empty.size)

                runtime.dll.cnp_clear_error()
                self.assertFalse(getattr(runtime.dll, name)(-1, 7))
                self.assertEqual(name, runtime.error_state().function)
                self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
