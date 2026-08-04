from __future__ import annotations

import ctypes
import unittest
from pathlib import Path

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"
CNP_ERR_BROADCAST = -7
CNP_ERR_SHAPE = -4
CNP_ERR_VALUE = -13
CNP_LONGLONG = 10


class GridAssemblySemanticsTests(unittest.TestCase):
    def _bind(self, runtime: CnumpyRuntime) -> None:
        dll = runtime.dll
        int64_pointer = ctypes.POINTER(ctypes.c_int64)
        dll.cnp_broadcast_shapes.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(int64_pointer),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(int64_pointer),
        ]
        dll.cnp_broadcast_shapes.restype = ctypes.c_int
        dll.cnp_broadcast_shape_free.argtypes = [int64_pointer, ctypes.c_int]
        dll.cnp_broadcast_shape_free.restype = None
        dll.cnp_can_broadcast.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        dll.cnp_can_broadcast.restype = ctypes.c_bool
        dll.cnp_indices.argtypes = [ctypes.c_int, int64_pointer]
        dll.cnp_indices.restype = ctypes.c_void_p
        dll.cnp_mgrid.argtypes = [
            ctypes.c_int,
            int64_pointer,
            int64_pointer,
            int64_pointer,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        dll.cnp_mgrid.restype = ctypes.c_int
        dll.cnp_ogrid.argtypes = dll.cnp_mgrid.argtypes
        dll.cnp_ogrid.restype = ctypes.c_int
        dll.cnp_meshgrid.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_bool,
            ctypes.c_bool,
        ]
        dll.cnp_meshgrid.restype = ctypes.c_void_p
        dll.cnp_block.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        dll.cnp_block.restype = ctypes.c_void_p
        dll.cnp_bmat.argtypes = dll.cnp_block.argtypes
        dll.cnp_bmat.restype = ctypes.c_void_p

    def _broadcast_shape(
        self, runtime: CnumpyRuntime, shapes: tuple[tuple[int, ...], ...]
    ) -> tuple[int, ...]:
        storages = [
            (ctypes.c_int64 * max(1, len(shape)))(*(shape or (0,)))
            for shape in shapes
        ]
        pointers = (ctypes.POINTER(ctypes.c_int64) * len(shapes))(
            *(ctypes.cast(item, ctypes.POINTER(ctypes.c_int64)) for item in storages)
        )
        ndims = (ctypes.c_int * len(shapes))(*(len(shape) for shape in shapes))
        out_ndim = ctypes.c_int(-1)
        out_shape = ctypes.POINTER(ctypes.c_int64)()
        runtime.dll.cnp_clear_error()
        status = runtime.dll.cnp_broadcast_shapes(
            len(shapes), pointers, ndims, ctypes.byref(out_ndim), ctypes.byref(out_shape)
        )
        self.assertEqual(0, status, runtime.error_state())
        try:
            return tuple(out_shape[index] for index in range(out_ndim.value))
        finally:
            runtime.dll.cnp_broadcast_shape_free(out_shape, out_ndim.value)

    def test_broadcast_shapes_and_can_broadcast_match_numpy(self) -> None:
        cases = (
            (((), ()), ()),
            (((3, 1), (1, 4)), (3, 4)),
            (((5, 1, 1), (3, 1), (1,)), (5, 3, 1)),
            (((2, 0, 3), (1, 3)), (2, 0, 3)),
        )
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for shapes, expected in cases:
                with self.subTest(shapes=shapes):
                    self.assertEqual(np.broadcast_shapes(*shapes), expected)
                    self.assertEqual(expected, self._broadcast_shape(runtime, shapes))

            with runtime.from_numpy(np.empty((3, 1))) as left, runtime.from_numpy(
                np.empty((1, 4))
            ) as right:
                self.assertTrue(runtime.dll.cnp_can_broadcast(left.pointer, right.pointer))
            with runtime.from_numpy(np.empty((2, 3))) as left, runtime.from_numpy(
                np.empty((4, 3))
            ) as right:
                runtime.dll.cnp_clear_error()
                self.assertFalse(runtime.dll.cnp_can_broadcast(left.pointer, right.pointer))
                self.assertEqual(0, runtime.error_state().status)

            bad_shapes = ((2, 3), (4, 3))
            storages = [(ctypes.c_int64 * 2)(*shape) for shape in bad_shapes]
            pointers = (ctypes.POINTER(ctypes.c_int64) * 2)(*storages)
            ndims = (ctypes.c_int * 2)(2, 2)
            out_ndim = ctypes.c_int(99)
            out_shape = ctypes.POINTER(ctypes.c_int64)(ctypes.c_int64(123))
            runtime.dll.cnp_clear_error()
            status = runtime.dll.cnp_broadcast_shapes(
                2, pointers, ndims, ctypes.byref(out_ndim), ctypes.byref(out_shape)
            )
            self.assertEqual(CNP_ERR_BROADCAST, status)
            self.assertEqual(0, out_ndim.value)
            self.assertFalse(out_shape)
            self.assertEqual("cnp_broadcast_shapes", runtime.error_state().function)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_indices_matches_numpy_shape_dtype_values_and_errors(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for dimensions_tuple in ((2, 3), (2, 0, 3), (1,)):
                dimensions = (ctypes.c_int64 * len(dimensions_tuple))(*dimensions_tuple)
                pointer = runtime.dll.cnp_indices(len(dimensions_tuple), dimensions)
                with runtime._owned_result(pointer, "cnp_indices") as result:
                    expected = np.indices(dimensions_tuple, dtype=np.int64)
                    self.assertEqual(CNP_LONGLONG, result.dtype_number)
                    self.assertEqual(expected.shape, result.shape)
                    np.testing.assert_array_equal(expected, result.to_numpy())

            invalid = (ctypes.c_int64 * 2)(2, -1)
            runtime.dll.cnp_clear_error()
            self.assertFalse(runtime.dll.cnp_indices(2, invalid))
            error = runtime.error_state()
            self.assertEqual(CNP_ERR_SHAPE, error.status)
            self.assertEqual("cnp_indices", error.function)
            self.assertEqual(baseline, runtime.retained_bytes)

    def _grid(
        self,
        runtime: CnumpyRuntime,
        symbol: str,
        starts: tuple[int, ...],
        stops: tuple[int, ...],
        steps: tuple[int, ...],
    ) -> tuple[np.ndarray, ...]:
        ndim = len(starts)
        start = (ctypes.c_int64 * ndim)(*starts)
        stop = (ctypes.c_int64 * ndim)(*stops)
        step = (ctypes.c_int64 * ndim)(*steps)
        outputs = (ctypes.c_void_p * ndim)()
        status = getattr(runtime.dll, symbol)(ndim, start, stop, step, outputs)
        self.assertEqual(0, status, runtime.error_state())
        arrays = [runtime._owned_result(outputs[index], symbol) for index in range(ndim)]
        try:
            return tuple(array.to_numpy() for array in arrays)
        finally:
            for array in arrays:
                array.close()

    def test_mgrid_and_ogrid_match_integer_numpy_slices_and_are_atomic(self) -> None:
        cases = (
            ((0, 1), (3, 6), (1, 2)),
            ((3, 5), (0, -1), (-1, -2)),
            ((0, 1), (0, 5), (1, 2)),
        )
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for starts, stops, steps in cases:
                slices = tuple(
                    slice(start, stop, step)
                    for start, stop, step in zip(starts, stops, steps)
                )
                dense_expected = tuple(np.mgrid[slices])
                sparse_expected = tuple(np.ogrid[slices])
                dense_actual = self._grid(runtime, "cnp_mgrid", starts, stops, steps)
                sparse_actual = self._grid(runtime, "cnp_ogrid", starts, stops, steps)
                for expected, actual in zip(dense_expected, dense_actual):
                    np.testing.assert_array_equal(expected, actual)
                for expected, actual in zip(sparse_expected, sparse_actual):
                    np.testing.assert_array_equal(expected, actual)

            starts = (ctypes.c_int64 * 2)(0, 0)
            stops = (ctypes.c_int64 * 2)(3, 3)
            steps = (ctypes.c_int64 * 2)(1, 0)
            outputs = (ctypes.c_void_p * 2)(1, 2)
            runtime.dll.cnp_clear_error()
            status = runtime.dll.cnp_mgrid(2, starts, stops, steps, outputs)
            self.assertEqual(CNP_ERR_VALUE, status)
            self.assertEqual([None, None], list(outputs))
            self.assertEqual("cnp_mgrid", runtime.error_state().function)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_meshgrid_returns_owned_first_numpy_result(self) -> None:
        left_np = np.asarray([1.0, 2.0])
        right_np = np.asarray([10.0, 20.0, 30.0])
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            with runtime.from_numpy(left_np) as left, runtime.from_numpy(right_np) as right:
                pointers = (ctypes.c_void_p * 2)(left.pointer, right.pointer)
                pointer = runtime.dll.cnp_meshgrid(2, pointers, False, False)
                with runtime._owned_result(pointer, "cnp_meshgrid") as result:
                    np.testing.assert_array_equal(
                        np.meshgrid(left_np, right_np, indexing="xy")[0],
                        result.to_numpy(),
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_block_and_bmat_match_numpy_and_reject_inconsistent_rows(self) -> None:
        block_values = (
            np.asarray([[1.0], [2.0]]),
            np.asarray([[3.0, 4.0], [5.0, 6.0]]),
            np.asarray([[7.0]]),
            np.asarray([[8.0, 9.0]]),
        )
        expected = np.block(
            [[block_values[0], block_values[1]], [block_values[2], block_values[3]]]
        )
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            arrays = [runtime.from_numpy(value) for value in block_values]
            pointers = (ctypes.c_void_p * 4)(*(array.pointer for array in arrays))
            try:
                for symbol in ("cnp_block", "cnp_bmat"):
                    pointer = getattr(runtime.dll, symbol)(2, 2, pointers)
                    with runtime._owned_result(pointer, symbol) as result:
                        np.testing.assert_array_equal(expected, result.to_numpy())
            finally:
                for array in arrays:
                    array.close()

            mismatch_values = (
                np.ones((2, 1)),
                np.ones((2, 2)),
                np.ones((1, 1)),
                np.ones((1, 1)),
            )
            arrays = [runtime.from_numpy(value) for value in mismatch_values]
            pointers = (ctypes.c_void_p * 4)(*(array.pointer for array in arrays))
            try:
                runtime.dll.cnp_clear_error()
                self.assertFalse(runtime.dll.cnp_block(2, 2, pointers))
                error = runtime.error_state()
                self.assertEqual(CNP_ERR_SHAPE, error.status)
                self.assertEqual("cnp_block", error.function)
            finally:
                for array in arrays:
                    array.close()
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
