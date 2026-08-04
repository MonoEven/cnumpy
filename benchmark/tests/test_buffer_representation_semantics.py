from __future__ import annotations

import ctypes
import unittest
from pathlib import Path

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"
CNP_ERR_TYPE = -3
CNP_ERR_VALUE = -13
CNP_INT16 = 4
CNP_FLOAT64 = 13


class BufferRepresentationSemanticsTests(unittest.TestCase):
    def _bind(self, runtime: CnumpyRuntime) -> None:
        dll = runtime.dll
        dll.cnp_tobytes.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int64),
        ]
        dll.cnp_tobytes.restype = ctypes.c_void_p
        dll.cnp_tolist.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int64),
        ]
        dll.cnp_tolist.restype = ctypes.POINTER(ctypes.c_double)
        dll.cnp_buffer_free.argtypes = [ctypes.c_void_p]
        dll.cnp_buffer_free.restype = None
        dll.cnp_frombuffer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int,
        ]
        dll.cnp_frombuffer.restype = ctypes.c_void_p
        dll.cnp_base_repr.argtypes = [ctypes.c_int64, ctypes.c_int, ctypes.c_int]
        dll.cnp_base_repr.restype = ctypes.c_void_p
        dll.cnp_binary_repr.argtypes = [ctypes.c_int64, ctypes.c_int]
        dll.cnp_binary_repr.restype = ctypes.c_void_p
        dll.cnp_char_free_string.argtypes = [ctypes.c_void_p]
        dll.cnp_char_free_string.restype = None

    def _owned_text(self, runtime: CnumpyRuntime, pointer: int, origin: str) -> str:
        self.assertTrue(pointer, runtime.error_state())
        try:
            return ctypes.string_at(pointer).decode("ascii")
        finally:
            runtime.dll.cnp_char_free_string(pointer)

    def test_tobytes_matches_numpy_for_dtypes_layouts_and_empty_arrays(self) -> None:
        arrays = (
            np.asarray([[1, -2], [3, -4]], dtype=np.int64),
            np.asarray([0, 2**63, 2**64 - 1], dtype=np.uint64),
            np.asarray([1 + 2j, -3 + 4j], dtype=np.complex128),
            np.asarray([True, False, True], dtype=np.bool_),
            np.empty((0, 3), dtype=np.float32),
        )
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for expected in arrays:
                with self.subTest(dtype=expected.dtype, shape=expected.shape):
                    with runtime.from_numpy(expected) as source:
                        size = ctypes.c_int64(-1)
                        pointer = runtime.dll.cnp_tobytes(
                            source.pointer, ctypes.byref(size)
                        )
                        self.assertTrue(pointer, runtime.error_state())
                        try:
                            self.assertEqual(expected.nbytes, size.value)
                            self.assertEqual(
                                expected.tobytes(order="C"),
                                ctypes.string_at(pointer, size.value),
                            )
                        finally:
                            runtime.dll.cnp_buffer_free(pointer)

            base_np = np.arange(12, dtype=np.int64).reshape(3, 4)
            with runtime.from_numpy(base_np) as base:
                with runtime.transpose(base) as source:
                    size = ctypes.c_int64(-1)
                    pointer = runtime.dll.cnp_tobytes(
                        source.pointer, ctypes.byref(size)
                    )
                    self.assertTrue(pointer, runtime.error_state())
                    try:
                        self.assertEqual(
                            base_np.T.tobytes(order="C"),
                            ctypes.string_at(pointer, size.value),
                        )
                    finally:
                        runtime.dll.cnp_buffer_free(pointer)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_tolist_flat_projection_matches_numpy_ravel_and_releases(self) -> None:
        expected = np.arange(12, dtype=np.float64).reshape(3, 4).T
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            with runtime.from_numpy(expected.T.copy()) as base:
                with runtime.transpose(base) as source:
                    size = ctypes.c_int64(-1)
                    pointer = runtime.dll.cnp_tolist(
                        source.pointer, ctypes.byref(size)
                    )
                    self.assertTrue(pointer, runtime.error_state())
                    try:
                        actual = [pointer[index] for index in range(size.value)]
                        self.assertEqual(
                            expected.ravel(order="C").tolist(), actual
                        )
                    finally:
                        runtime.dll.cnp_buffer_free(pointer)

            with runtime.from_numpy(np.empty((0, 2))) as empty:
                size = ctypes.c_int64(-1)
                pointer = runtime.dll.cnp_tolist(empty.pointer, ctypes.byref(size))
                self.assertTrue(pointer, runtime.error_state())
                runtime.dll.cnp_buffer_free(pointer)
                self.assertEqual(0, size.value)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_frombuffer_matches_numpy_values_and_returns_independent_owner(self) -> None:
        values = np.asarray([1, -2, 300, -400], dtype=np.int16)
        raw = bytearray(values.tobytes())
        storage = (ctypes.c_ubyte * len(raw)).from_buffer(raw)
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            pointer = runtime.dll.cnp_frombuffer(storage, len(raw), CNP_INT16)
            with runtime._owned_result(pointer, "cnp_frombuffer") as result:
                np.testing.assert_array_equal(np.frombuffer(raw, dtype=np.int16), result.to_numpy())
                raw[:] = b"\0" * len(raw)
                np.testing.assert_array_equal(values, result.to_numpy())

            empty_storage = ctypes.create_string_buffer(1)
            pointer = runtime.dll.cnp_frombuffer(
                empty_storage, 0, CNP_FLOAT64
            )
            with runtime._owned_result(pointer, "cnp_frombuffer") as result:
                self.assertEqual((0,), result.shape)
                self.assertEqual((), result.values())

            runtime.dll.cnp_clear_error()
            self.assertFalse(runtime.dll.cnp_frombuffer(empty_storage, 3, CNP_INT16))
            error = runtime.error_state()
            self.assertEqual(CNP_ERR_VALUE, error.status)
            self.assertEqual("cnp_frombuffer", error.function)

            runtime.dll.cnp_clear_error()
            self.assertFalse(runtime.dll.cnp_frombuffer(empty_storage, 8, 99))
            error = runtime.error_state()
            self.assertEqual(CNP_ERR_TYPE, error.status)
            self.assertEqual("cnp_frombuffer", error.function)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_integer_representation_matches_numpy_and_errors_are_labeled(self) -> None:
        base_cases = (
            (5, 2, 0),
            (5, 2, 5),
            (-5, 2, 0),
            (-5, 2, 5),
            (-(2**63), 16, 0),
            (123456789, 36, 3),
        )
        binary_cases = (
            (5, -1),
            (5, 8),
            (-5, -1),
            (-5, 8),
            (-(2**63), 64),
        )
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for number, base, padding in base_cases:
                self.assertEqual(
                    np.base_repr(number, base=base, padding=padding),
                    self._owned_text(
                        runtime,
                        runtime.dll.cnp_base_repr(number, base, padding),
                        "cnp_base_repr",
                    ),
                )
            for number, width in binary_cases:
                expected = (
                    np.binary_repr(number)
                    if width < 0
                    else np.binary_repr(number, width=width)
                )
                self.assertEqual(
                    expected,
                    self._owned_text(
                        runtime,
                        runtime.dll.cnp_binary_repr(number, width),
                        "cnp_binary_repr",
                    ),
                )

            for base in (1, 37):
                runtime.dll.cnp_clear_error()
                self.assertFalse(runtime.dll.cnp_base_repr(5, base, 0))
                error = runtime.error_state()
                self.assertEqual(CNP_ERR_VALUE, error.status)
                self.assertEqual("cnp_base_repr", error.function)

            for _ in range(256):
                pointer = runtime.dll.cnp_binary_repr(-123456789, 64)
                runtime.dll.cnp_char_free_string(pointer)
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
