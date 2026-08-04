from __future__ import annotations

import ctypes
from contextlib import ExitStack
from pathlib import Path
import unittest

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"


class StrideViewSurfaceSemanticsTests(unittest.TestCase):
    SYMBOLS = frozenset(
        {"cnp_as_strided", "cnp_rollaxis", "cnp_sliding_window_view"}
    )

    @staticmethod
    def _function(runtime: CnumpyRuntime, name: str, argtypes, restype):
        function = getattr(runtime.dll, name)
        function.argtypes = argtypes
        function.restype = restype
        return function

    def test_views_match_numpy_shape_strides_values_and_source_lifetime(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as results:
            baseline = runtime.retained_bytes
            as_strided = self._function(
                runtime,
                "cnp_as_strided",
                [
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.POINTER(ctypes.c_int64),
                ],
                ctypes.c_void_p,
            )
            sliding = self._function(
                runtime,
                "cnp_sliding_window_view",
                [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int],
                ctypes.c_void_p,
            )
            rollaxis = self._function(
                runtime,
                "cnp_rollaxis",
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
                ctypes.c_void_p,
            )

            values = np.arange(12, dtype=np.int64).reshape(3, 4)
            source = runtime.from_numpy(values)
            shape = (ctypes.c_int64 * 2)(2, 2)
            strides = (ctypes.c_int64 * 2)(values.strides[0], 2 * values.strides[1])
            strided = results.enter_context(
                runtime._owned_result(
                    as_strided(source.pointer, 2, shape, strides),
                    "cnp_as_strided",
                )
            )
            window_values = np.arange(5, dtype=np.int64)
            window_source = runtime.from_numpy(window_values)
            windows = results.enter_context(
                runtime._owned_result(
                    sliding(window_source.pointer, 3, 0),
                    "cnp_sliding_window_view",
                )
            )

            cube_values = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
            cube = runtime.from_numpy(cube_values)
            rolled = results.enter_context(
                runtime._owned_result(
                    rollaxis(cube.pointer, 2, 1), "cnp_rollaxis"
                )
            )

            source.close()
            window_source.close()
            cube.close()
            expected_strided = np.lib.stride_tricks.as_strided(
                values, shape=(2, 2), strides=strides
            )
            expected_windows = np.lib.stride_tricks.sliding_window_view(
                window_values, 3, axis=0
            )
            expected_rolled = np.rollaxis(cube_values, 2, 1)
            for expected, actual in (
                (expected_strided, strided),
                (expected_windows, windows),
                (expected_rolled, rolled),
            ):
                self.assertEqual(expected.shape, actual.shape)
                self.assertEqual(expected.strides, actual.strides)
                self.assertEqual(expected.dtype, actual.numpy_dtype)
                np.testing.assert_array_equal(
                    expected, actual.to_numpy(), strict=True
                )

            fill_diagonal = self._function(
                runtime,
                "cnp_fill_diagonal",
                [ctypes.c_void_p, ctypes.c_double],
                ctypes.c_int,
            )
            before = windows.to_numpy().copy()
            runtime.dll.cnp_clear_error()
            self.assertNotEqual(0, fill_diagonal(windows.pointer, 9.0))
            np.testing.assert_array_equal(before, windows.to_numpy())

            results.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_invalid_view_requests_are_explicit_and_nonretaining(self) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(
                runtime.from_numpy(np.arange(6, dtype=np.int64).reshape(2, 3))
            )
            as_strided = self._function(
                runtime,
                "cnp_as_strided",
                [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64)],
                ctypes.c_void_p,
            )
            sliding = self._function(
                runtime,
                "cnp_sliding_window_view",
                [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int],
                ctypes.c_void_p,
            )
            rollaxis = self._function(
                runtime,
                "cnp_rollaxis",
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
                ctypes.c_void_p,
            )
            negative_shape = (ctypes.c_int64 * 1)(-1)
            cases = (
                ("cnp_as_strided", lambda: as_strided(source.pointer, 1, negative_shape, None)),
                ("cnp_sliding_window_view", lambda: sliding(source.pointer, 4, 1)),
                ("cnp_sliding_window_view", lambda: sliding(source.pointer, 1, 2)),
                ("cnp_rollaxis", lambda: rollaxis(source.pointer, 2, 0)),
                ("cnp_rollaxis", lambda: rollaxis(source.pointer, 1, 3)),
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

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
