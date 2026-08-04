from __future__ import annotations

import ctypes
import unittest
from pathlib import Path

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"
CNP_ERR_BROADCAST = -7
CNP_ERR_VALUE = -13


class IteratorSurfaceSemanticsTests(unittest.TestCase):
    def _bind(self, runtime: CnumpyRuntime) -> None:
        dll = runtime.dll
        dll.cnp_iter_new.argtypes = [ctypes.c_void_p]
        dll.cnp_iter_new.restype = ctypes.c_void_p
        dll.cnp_iter_next.argtypes = [ctypes.c_void_p]
        dll.cnp_iter_next.restype = ctypes.c_bool
        dll.cnp_iter_data.argtypes = [ctypes.c_void_p]
        dll.cnp_iter_data.restype = ctypes.c_void_p
        dll.cnp_iter_free.argtypes = [ctypes.c_void_p]
        dll.cnp_iter_free.restype = None
        dll.cnp_iter_reset.argtypes = [ctypes.c_void_p]
        dll.cnp_iter_reset.restype = None

        dll.cnp_multi_iter_new.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        dll.cnp_multi_iter_new.restype = ctypes.c_void_p
        dll.cnp_multi_iter_next.argtypes = [ctypes.c_void_p]
        dll.cnp_multi_iter_next.restype = ctypes.c_bool
        dll.cnp_multi_iter_data.argtypes = [ctypes.c_void_p]
        dll.cnp_multi_iter_data.restype = ctypes.POINTER(ctypes.c_void_p)
        dll.cnp_multi_iter_free.argtypes = [ctypes.c_void_p]
        dll.cnp_multi_iter_free.restype = None

        dll.cnp_ndenumerate_next.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
        ]
        dll.cnp_ndenumerate_next.restype = ctypes.c_bool
        dll.cnp_ndindex_next.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
        ]
        dll.cnp_ndindex_next.restype = ctypes.c_bool

    def _iter_values(self, runtime: CnumpyRuntime, iterator: int) -> list[float]:
        result: list[float] = []
        pointer = runtime.dll.cnp_iter_data(iterator)
        if pointer:
            result.append(ctypes.c_double.from_address(pointer).value)
        while runtime.dll.cnp_iter_next(iterator):
            pointer = runtime.dll.cnp_iter_data(iterator)
            self.assertTrue(pointer)
            result.append(ctypes.c_double.from_address(pointer).value)
        self.assertFalse(runtime.dll.cnp_iter_data(iterator))
        return result

    def test_single_iterator_matches_numpy_c_order_and_retains_strided_source(
        self,
    ) -> None:
        expected = np.arange(12.0).reshape(3, 4).T
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            base = runtime.from_numpy(np.arange(12.0).reshape(3, 4))
            view = runtime.transpose(base)
            iterator = runtime.dll.cnp_iter_new(view.pointer)
            self.assertTrue(iterator, runtime.error_state())
            view.close()
            base.close()
            try:
                self.assertEqual(
                    expected.ravel(order="C").tolist(),
                    self._iter_values(runtime, iterator),
                )
                runtime.dll.cnp_iter_reset(iterator)
                self.assertEqual(
                    expected.ravel(order="C").tolist(),
                    self._iter_values(runtime, iterator),
                )
            finally:
                runtime.dll.cnp_iter_free(iterator)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_single_iterator_empty_and_invalid_states_are_explicit(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            with runtime.from_numpy(np.empty((0, 3), dtype=np.float64)) as source:
                iterator = runtime.dll.cnp_iter_new(source.pointer)
                self.assertTrue(iterator, runtime.error_state())
                try:
                    self.assertFalse(runtime.dll.cnp_iter_data(iterator))
                    self.assertFalse(runtime.dll.cnp_iter_next(iterator))
                finally:
                    runtime.dll.cnp_iter_free(iterator)

            runtime.dll.cnp_clear_error()
            self.assertFalse(runtime.dll.cnp_iter_new(None))
            error = runtime.error_state()
            self.assertEqual(CNP_ERR_VALUE, error.status)
            self.assertEqual("cnp_iter_new", error.function)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_multi_iterator_matches_numpy_broadcast_and_retains_sources(self) -> None:
        left_np = np.asarray([[1.0], [2.0]])
        right_np = np.asarray([[10.0, 20.0, 30.0]])
        expected_left, expected_right = np.broadcast_arrays(left_np, right_np)

        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            left = runtime.from_numpy(left_np)
            right = runtime.from_numpy(right_np)
            pointers = (ctypes.c_void_p * 2)(left.pointer, right.pointer)
            iterator = runtime.dll.cnp_multi_iter_new(2, pointers)
            self.assertTrue(iterator, runtime.error_state())
            left.close()
            right.close()
            actual: list[tuple[float, float]] = []
            try:
                while True:
                    data = runtime.dll.cnp_multi_iter_data(iterator)
                    if not data:
                        break
                    actual.append(
                        (
                            ctypes.c_double.from_address(data[0]).value,
                            ctypes.c_double.from_address(data[1]).value,
                        )
                    )
                    if not runtime.dll.cnp_multi_iter_next(iterator):
                        break
                self.assertFalse(runtime.dll.cnp_multi_iter_data(iterator))
            finally:
                runtime.dll.cnp_multi_iter_free(iterator)
            expected = list(
                zip(
                    expected_left.ravel(order="C").tolist(),
                    expected_right.ravel(order="C").tolist(),
                )
            )
            self.assertEqual(expected, actual)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_multi_iterator_rejects_incompatible_and_null_arrays_atomically(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            with runtime.from_numpy(np.ones((2, 2))) as left, runtime.from_numpy(
                np.ones((3, 2))
            ) as right:
                pointers = (ctypes.c_void_p * 2)(left.pointer, right.pointer)
                runtime.dll.cnp_clear_error()
                self.assertFalse(runtime.dll.cnp_multi_iter_new(2, pointers))
                error = runtime.error_state()
                self.assertEqual(CNP_ERR_BROADCAST, error.status)
                self.assertEqual("cnp_multi_iter_new", error.function)

                pointers[1] = None
                runtime.dll.cnp_clear_error()
                self.assertFalse(runtime.dll.cnp_multi_iter_new(2, pointers))
                error = runtime.error_state()
                self.assertEqual(CNP_ERR_VALUE, error.status)
                self.assertEqual("cnp_multi_iter_new", error.function)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_ndenumerate_next_matches_numpy_for_strided_arrays_and_errors(
        self,
    ) -> None:
        expected = np.arange(6.0).reshape(2, 3).T
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            with runtime.from_numpy(np.arange(6.0).reshape(2, 3)) as base:
                with runtime.transpose(base) as view:
                    state = ctypes.c_int64(0)
                    coordinates = (ctypes.c_int64 * view.ndim)()
                    value = ctypes.c_double()
                    actual: list[tuple[tuple[int, ...], float]] = []
                    while runtime.dll.cnp_ndenumerate_next(
                        view.pointer,
                        ctypes.byref(state),
                        coordinates,
                        ctypes.byref(value),
                    ):
                        actual.append(
                            (tuple(coordinates), value.value)
                        )
                    self.assertEqual(list(np.ndenumerate(expected)), actual)

                    state.value = -1
                    runtime.dll.cnp_clear_error()
                    self.assertFalse(
                        runtime.dll.cnp_ndenumerate_next(
                            view.pointer,
                            ctypes.byref(state),
                            coordinates,
                            ctypes.byref(value),
                        )
                    )
                    error = runtime.error_state()
                    self.assertEqual(CNP_ERR_VALUE, error.status)
                    self.assertEqual("cnp_ndenumerate_next", error.function)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_ndindex_successor_reconstructs_numpy_order_and_rejects_shapes(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for shape_tuple in ((2, 3), (1,), (2, 0, 3)):
                with self.subTest(shape=shape_tuple):
                    shape = (ctypes.c_int64 * len(shape_tuple))(*shape_tuple)
                    coordinates = (ctypes.c_int64 * len(shape_tuple))()
                    coordinates[-1] = -1
                    actual: list[tuple[int, ...]] = []
                    while runtime.dll.cnp_ndindex_next(
                        len(shape_tuple), shape, coordinates
                    ):
                        actual.append(tuple(coordinates))
                    self.assertEqual(list(np.ndindex(shape_tuple)), actual)

            invalid_shape = (ctypes.c_int64 * 2)(2, -1)
            coordinates = (ctypes.c_int64 * 2)(0, -1)
            runtime.dll.cnp_clear_error()
            self.assertFalse(
                runtime.dll.cnp_ndindex_next(2, invalid_shape, coordinates)
            )
            error = runtime.error_state()
            self.assertEqual(CNP_ERR_VALUE, error.status)
            self.assertEqual("cnp_ndindex_next", error.function)
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
