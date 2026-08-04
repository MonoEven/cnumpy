from __future__ import annotations

import ctypes
from contextlib import ExitStack
from pathlib import Path
import unittest

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"


class DiscreteUtilitySurfaceSemanticsTests(unittest.TestCase):
    SYMBOLS = frozenset(
        {
            "cnp_bincount",
            "cnp_ediff1d",
            "cnp_ravel_multi_index",
            "cnp_sinc",
            "cnp_tril_indices",
            "cnp_triu_indices",
            "cnp_unravel_index",
        }
    )

    @staticmethod
    def _function(runtime: CnumpyRuntime, name: str, argtypes, restype):
        function = getattr(runtime.dll, name)
        function.argtypes = argtypes
        function.restype = restype
        return function

    def test_bincount_and_ediff1d_match_numpy_dtypes_exact_values_lifetime(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as results:
            baseline = runtime.retained_bytes
            bincount = self._function(
                runtime,
                "cnp_bincount",
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64],
                ctypes.c_void_p,
            )
            ediff1d = self._function(
                runtime,
                "cnp_ediff1d",
                [
                    ctypes.c_void_p,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_bool,
                    ctypes.c_bool,
                ],
                ctypes.c_void_p,
            )

            x_values = np.asarray([0, 2, 1, 2, 2], dtype=np.int64)
            weight_values = np.asarray([0.5, 1.0, 1.5, -0.25, 2.0])
            with runtime.from_numpy(x_values) as x, runtime.from_numpy(
                weight_values
            ) as weights:
                counts = results.enter_context(
                    runtime._owned_result(
                        bincount(x.pointer, None, 5), "cnp_bincount"
                    )
                )
                weighted = results.enter_context(
                    runtime._owned_result(
                        bincount(x.pointer, weights.pointer, 1), "cnp_bincount"
                    )
                )
            np.testing.assert_array_equal(
                np.bincount(x_values, minlength=5),
                counts.to_numpy(),
                strict=True,
            )
            np.testing.assert_array_equal(
                np.bincount(x_values, weights=weight_values, minlength=1),
                weighted.to_numpy(),
                strict=True,
            )

            with runtime.from_numpy(np.asarray([], dtype=np.int64)) as empty:
                empty_counts = results.enter_context(
                    runtime._owned_result(
                        bincount(empty.pointer, None, 0), "cnp_bincount"
                    )
                )
            np.testing.assert_array_equal(
                np.bincount(np.asarray([], dtype=np.int64)),
                empty_counts.to_numpy(),
                strict=True,
            )

            wide_values = np.asarray(
                [2**60 + 1, 2**60 + 3, 2**60 + 8], dtype=np.int64
            )
            with runtime.from_numpy(wide_values) as wide:
                differences = results.enter_context(
                    runtime._owned_result(
                        ediff1d(wide.pointer, 0.0, 0.0, False, False),
                        "cnp_ediff1d",
                    )
                )
            np.testing.assert_array_equal(
                np.ediff1d(wide_values), differences.to_numpy(), strict=True
            )

            float_values = np.asarray([1.5, 2.25, 5.0], dtype=np.float64)
            with runtime.from_numpy(float_values) as source:
                bounded = results.enter_context(
                    runtime._owned_result(
                        ediff1d(source.pointer, -2.0, 9.0, True, True),
                        "cnp_ediff1d",
                    )
                )
            np.testing.assert_array_equal(
                np.ediff1d(float_values, to_begin=-2.0, to_end=9.0),
                bounded.to_numpy(),
                strict=True,
            )

            results.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_ravel_unravel_and_triangle_indices_match_numpy_projection(self) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as results:
            baseline = runtime.retained_bytes
            unravel = self._function(
                runtime,
                "cnp_unravel_index",
                [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)],
                ctypes.c_void_p,
            )
            ravel = self._function(
                runtime,
                "cnp_ravel_multi_index",
                [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)],
                ctypes.c_void_p,
            )
            triu = self._function(
                runtime,
                "cnp_triu_indices",
                [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64],
                ctypes.c_void_p,
            )
            tril = self._function(
                runtime,
                "cnp_tril_indices",
                [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64],
                ctypes.c_void_p,
            )

            shape = (2, 3, 4)
            shape_storage = (ctypes.c_int64 * 3)(*shape)
            flat_values = np.asarray([0, 5, 23], dtype=np.int64)
            coordinates = np.vstack(np.unravel_index(flat_values, shape))
            with runtime.from_numpy(flat_values) as flat_source, runtime.from_numpy(
                coordinates
            ) as coordinate_source:
                unraveled = results.enter_context(
                    runtime._owned_result(
                        unravel(flat_source.pointer, 3, shape_storage),
                        "cnp_unravel_index",
                    )
                )
                raveled = results.enter_context(
                    runtime._owned_result(
                        ravel(coordinate_source.pointer, 3, shape_storage),
                        "cnp_ravel_multi_index",
                    )
                )
            np.testing.assert_array_equal(
                coordinates, unraveled.to_numpy(), strict=True
            )
            np.testing.assert_array_equal(
                np.ravel_multi_index(tuple(coordinates), shape),
                raveled.to_numpy(),
                strict=True,
            )

            upper = results.enter_context(
                runtime._owned_result(triu(4, -1, 6), "cnp_triu_indices")
            )
            lower = results.enter_context(
                runtime._owned_result(tril(4, 1, 6), "cnp_tril_indices")
            )
            np.testing.assert_array_equal(
                np.vstack(np.triu_indices(4, -1, 6)),
                upper.to_numpy(),
                strict=True,
            )
            np.testing.assert_array_equal(
                np.vstack(np.tril_indices(4, 1, 6)),
                lower.to_numpy(),
                strict=True,
            )

            results.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_sinc_matches_numpy_real_complex_dtype_and_release(self) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as results:
            baseline = runtime.retained_bytes
            sinc = self._function(
                runtime, "cnp_sinc", [ctypes.c_void_p], ctypes.c_void_p
            )
            for values in (
                np.asarray([-1.0, 0.0, 0.5], dtype=np.float32),
                np.asarray([0.0 + 0.0j, 0.5 + 0.25j], dtype=np.complex128),
            ):
                with self.subTest(dtype=values.dtype), runtime.from_numpy(
                    values
                ) as source:
                    result = results.enter_context(
                        runtime._owned_result(sinc(source.pointer), "cnp_sinc")
                    )
                    np.testing.assert_allclose(
                        np.sinc(values), result.to_numpy(),
                        rtol=2e-6, atol=2e-7,
                    )
                    self.assertEqual(np.sinc(values).dtype, result.to_numpy().dtype)

            results.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_discrete_utility_errors_are_explicit_and_nonretaining(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            bincount = self._function(
                runtime,
                "cnp_bincount",
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64],
                ctypes.c_void_p,
            )
            ediff1d = self._function(
                runtime,
                "cnp_ediff1d",
                [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_bool, ctypes.c_bool],
                ctypes.c_void_p,
            )
            unravel = self._function(
                runtime,
                "cnp_unravel_index",
                [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)],
                ctypes.c_void_p,
            )
            ravel = self._function(
                runtime,
                "cnp_ravel_multi_index",
                [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)],
                ctypes.c_void_p,
            )
            shape_storage = (ctypes.c_int64 * 2)(2, 3)
            with runtime.from_numpy(np.asarray([-1, 2], dtype=np.int64)) as negative, runtime.from_numpy(
                np.asarray([6], dtype=np.int64)
            ) as out_of_bounds, runtime.from_numpy(
                np.asarray([[0], [3]], dtype=np.int64)
            ) as invalid_coords, runtime.from_numpy(
                np.asarray([[0.0], [1.0]], dtype=np.float64)
            ) as floating_coords, runtime.from_numpy(
                np.asarray([1, 2], dtype=np.int64)
            ) as integers:
                cases = (
                    ("cnp_bincount", lambda: bincount(negative.pointer, None, 0)),
                    (
                        "cnp_ediff1d",
                        lambda: ediff1d(integers.pointer, 0.5, 0.0, True, False),
                    ),
                    (
                        "cnp_unravel_index",
                        lambda: unravel(out_of_bounds.pointer, 2, shape_storage),
                    ),
                    (
                        "cnp_unravel_index",
                        lambda: unravel(negative.pointer, 2, shape_storage),
                    ),
                    (
                        "cnp_ravel_multi_index",
                        lambda: ravel(invalid_coords.pointer, 2, shape_storage),
                    ),
                    (
                        "cnp_ravel_multi_index",
                        lambda: ravel(floating_coords.pointer, 2, shape_storage),
                    ),
                )
                active = runtime.retained_bytes
                for name, call in cases:
                    with self.subTest(symbol=name):
                        runtime.dll.cnp_clear_error()
                        pointer = call()
                        if pointer:
                            runtime._owned_result(pointer, name).close()
                        self.assertFalse(pointer)
                        self.assertEqual(name, runtime.error_state().function)
                        self.assertEqual(active, runtime.retained_bytes)

            for name, argtypes, arguments in (
                ("cnp_sinc", [ctypes.c_void_p], (None,)),
                (
                    "cnp_triu_indices",
                    [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64],
                    (-1, 0, 3),
                ),
                (
                    "cnp_tril_indices",
                    [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64],
                    (-1, 0, 3),
                ),
            ):
                function = self._function(runtime, name, argtypes, ctypes.c_void_p)
                runtime.dll.cnp_clear_error()
                self.assertFalse(function(*arguments))
                self.assertEqual(name, runtime.error_state().function)
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
