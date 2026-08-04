from __future__ import annotations

import ctypes
from contextlib import ExitStack
from pathlib import Path
import unittest

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"
CNP_NOTYPE = 0
CNP_DOUBLE = 13


class ArrayLayoutSurfaceSemanticsTests(unittest.TestCase):
    SYMBOLS = frozenset(
        {
            "cnp_asarray_chkfinite",
            "cnp_ascontiguousarray",
            "cnp_asfortranarray",
            "cnp_require",
            "cnp_resize",
        }
    )

    @staticmethod
    def _function(runtime: CnumpyRuntime, name: str, argtypes, restype):
        function = getattr(runtime.dll, name)
        function.argtypes = argtypes
        function.restype = restype
        return function

    def test_contiguous_fortran_and_require_match_numpy_layout_values_lifetime(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as results:
            baseline = runtime.retained_bytes
            contiguous = self._function(
                runtime,
                "cnp_ascontiguousarray",
                [ctypes.c_void_p],
                ctypes.c_void_p,
            )
            fortran = self._function(
                runtime,
                "cnp_asfortranarray",
                [ctypes.c_void_p],
                ctypes.c_void_p,
            )
            require = self._function(
                runtime,
                "cnp_require",
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_bool],
                ctypes.c_void_p,
            )

            original = np.asarray(
                [[2**60 + 1, 2**60 + 2, 2**60 + 3],
                 [2**60 + 4, 2**60 + 5, 2**60 + 6]],
                dtype=np.int64,
            )
            with runtime.from_numpy(original) as owner:
                source = runtime.transpose(owner, (1, 0))
                c_result = results.enter_context(
                    runtime._owned_result(
                        contiguous(source.pointer), "cnp_ascontiguousarray"
                    )
                )
                f_result = results.enter_context(
                    runtime._owned_result(
                        fortran(source.pointer), "cnp_asfortranarray"
                    )
                )
                required = results.enter_context(
                    runtime._owned_result(
                        require(source.pointer, CNP_DOUBLE, True), "cnp_require"
                    )
                )
                source.close()

            expected = original.T
            np.testing.assert_array_equal(
                np.ascontiguousarray(expected), c_result.to_numpy(), strict=True
            )
            self.assertTrue(c_result.c_contiguous)
            np.testing.assert_array_equal(
                np.asfortranarray(expected), f_result.to_numpy(), strict=True
            )
            self.assertTrue(f_result.f_contiguous)
            np.testing.assert_array_equal(
                np.require(expected, dtype=np.float64, requirements=["C"]),
                required.to_numpy(),
                strict=True,
            )
            self.assertTrue(required.c_contiguous)

            scalar_value = np.asarray(2**60 + 3, dtype=np.int64)
            with runtime.from_numpy(scalar_value) as scalar:
                scalar_c = results.enter_context(
                    runtime._owned_result(
                        contiguous(scalar.pointer), "cnp_ascontiguousarray"
                    )
                )
                scalar_f = results.enter_context(
                    runtime._owned_result(
                        fortran(scalar.pointer), "cnp_asfortranarray"
                    )
                )
            np.testing.assert_array_equal(
                np.ascontiguousarray(scalar_value),
                scalar_c.to_numpy(),
                strict=True,
            )
            np.testing.assert_array_equal(
                np.asfortranarray(scalar_value),
                scalar_f.to_numpy(),
                strict=True,
            )

            results.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_resize_matches_numpy_scalar_empty_wide_integer_and_errors(self) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as results:
            baseline = runtime.retained_bytes
            resize = self._function(
                runtime,
                "cnp_resize",
                [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)],
                ctypes.c_void_p,
            )
            cases = (
                (np.asarray(2**60 + 3, dtype=np.int64), (2, 3)),
                (
                    np.asarray([2**60 + 1, 2**60 + 3], dtype=np.int64),
                    (3, 3),
                ),
                (np.asarray([], dtype=np.int64), (2, 2)),
            )
            for values, shape in cases:
                with self.subTest(shape=shape, input_shape=values.shape), runtime.from_numpy(
                    values
                ) as source:
                    shape_storage = (ctypes.c_int64 * len(shape))(*shape)
                    result = results.enter_context(
                        runtime._owned_result(
                            resize(source.pointer, len(shape), shape_storage),
                            "cnp_resize",
                        )
                    )
                    np.testing.assert_array_equal(
                        np.resize(values, shape), result.to_numpy(), strict=True
                    )

            invalid_shape = (ctypes.c_int64 * 1)(-1)
            runtime.dll.cnp_clear_error()
            self.assertFalse(resize(None, 1, invalid_shape))
            self.assertEqual("cnp_resize", runtime.error_state().function)
            with runtime.from_numpy(np.asarray([1, 2], dtype=np.int64)) as source:
                runtime.dll.cnp_clear_error()
                self.assertFalse(resize(source.pointer, 1, invalid_shape))
                self.assertEqual("cnp_resize", runtime.error_state().function)

            results.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_asarray_chkfinite_matches_numpy_dtype_complex_errors_and_release(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as results:
            baseline = runtime.retained_bytes
            chkfinite = self._function(
                runtime,
                "cnp_asarray_chkfinite",
                [ctypes.c_void_p, ctypes.c_int],
                ctypes.c_void_p,
            )
            values = np.asarray([1, 2, 3], dtype=np.int64)
            with runtime.from_numpy(values) as source:
                converted = results.enter_context(
                    runtime._owned_result(
                        chkfinite(source.pointer, CNP_DOUBLE),
                        "cnp_asarray_chkfinite",
                    )
                )
            np.testing.assert_array_equal(
                np.asarray_chkfinite(values, dtype=np.float64),
                converted.to_numpy(),
                strict=True,
            )

            for invalid in (
                np.asarray([1.0, np.inf]),
                np.asarray([1.0 + 0.0j, 2.0 + np.inf * 1j]),
            ):
                with self.subTest(dtype=invalid.dtype), runtime.from_numpy(
                    invalid
                ) as source:
                    active = runtime.retained_bytes
                    runtime.dll.cnp_clear_error()
                    pointer = chkfinite(source.pointer, CNP_NOTYPE)
                    if pointer:
                        runtime._owned_result(
                            pointer, "cnp_asarray_chkfinite"
                        ).close()
                    self.assertFalse(pointer)
                    self.assertEqual(
                        "cnp_asarray_chkfinite", runtime.error_state().function
                    )
                    self.assertEqual(active, runtime.retained_bytes)

            runtime.dll.cnp_clear_error()
            self.assertFalse(chkfinite(None, CNP_NOTYPE))
            self.assertEqual("cnp_asarray_chkfinite", runtime.error_state().function)
            results.close()
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
