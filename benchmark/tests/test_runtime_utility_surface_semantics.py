from __future__ import annotations

import ctypes
from contextlib import ExitStack
from pathlib import Path
import unittest

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"

DTYPE_NUMBERS = {
    np.dtype(np.float16): 24,
    np.dtype(np.float32): 12,
    np.dtype(np.float64): 13,
    np.dtype(np.complex64): 15,
    np.dtype(np.complex128): 16,
}


class RuntimeUtilitySurfaceSemanticsTests(unittest.TestCase):
    SYMBOLS = frozenset(
        {
            "cnp_common_type",
            "cnp_format_float",
            "cnp_get_printoptions",
            "cnp_getbufsize",
            "cnp_geterr",
            "cnp_min_scalar_type",
            "cnp_set_printoptions",
            "cnp_setbufsize",
            "cnp_seterr",
        }
    )

    @staticmethod
    def _bind(runtime: CnumpyRuntime) -> None:
        runtime.dll.cnp_min_scalar_type.argtypes = [ctypes.c_double]
        runtime.dll.cnp_min_scalar_type.restype = ctypes.c_int
        runtime.dll.cnp_common_type.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        runtime.dll.cnp_common_type.restype = ctypes.c_int
        runtime.dll.cnp_seterr.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        runtime.dll.cnp_seterr.restype = None
        runtime.dll.cnp_geterr.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        runtime.dll.cnp_geterr.restype = None
        runtime.dll.cnp_set_printoptions.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        runtime.dll.cnp_set_printoptions.restype = None
        runtime.dll.cnp_get_printoptions.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        runtime.dll.cnp_get_printoptions.restype = None
        runtime.dll.cnp_getbufsize.argtypes = []
        runtime.dll.cnp_getbufsize.restype = ctypes.c_int64
        runtime.dll.cnp_setbufsize.argtypes = [ctypes.c_int64]
        runtime.dll.cnp_setbufsize.restype = ctypes.c_int
        runtime.dll.cnp_format_float.argtypes = [
            ctypes.c_double,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_bool,
        ]
        runtime.dll.cnp_format_float.restype = ctypes.c_int

    @staticmethod
    def _get_error_modes(runtime: CnumpyRuntime) -> tuple[int, ...]:
        values = [ctypes.c_int() for _ in range(4)]
        runtime.dll.cnp_geterr(*(ctypes.byref(value) for value in values))
        return tuple(value.value for value in values)

    @staticmethod
    def _get_print_options(runtime: CnumpyRuntime) -> tuple[int, ...]:
        values = [ctypes.c_int() for _ in range(5)]
        runtime.dll.cnp_get_printoptions(
            *(ctypes.byref(value) for value in values)
        )
        return tuple(value.value for value in values)

    def test_scalar_and_common_types_match_numpy_125_projection(self) -> None:
        values = (0.0, 1.0, -1.0, 65000.0, 1e-7, 1e10, 1e40, np.inf, np.nan)
        dtype_cases = (
            (np.int8, np.float32),
            (np.float16, np.float32),
            (np.float32, np.complex64),
            (np.int64, np.float32),
            (np.float64, np.complex64),
            (np.int8, np.uint8),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for value in values:
                with self.subTest(value=value):
                    expected = DTYPE_NUMBERS[np.dtype(np.min_scalar_type(value))]
                    runtime.dll.cnp_clear_error()
                    self.assertEqual(expected, runtime.dll.cnp_min_scalar_type(value))
                    self.assertEqual(0, runtime.error_state().status)

            for dtype_pair in dtype_cases:
                arrays = tuple(
                    stack.enter_context(
                        runtime.from_numpy(np.asarray([1], dtype=dtype))
                    )
                    for dtype in dtype_pair
                )
                pointers = (ctypes.c_void_p * len(arrays))(
                    *(array.pointer.value for array in arrays)
                )
                expected = DTYPE_NUMBERS[
                    np.dtype(
                        np.common_type(
                            *(np.asarray([1], dtype=dtype) for dtype in dtype_pair)
                        )
                    )
                ]
                with self.subTest(dtypes=dtype_pair):
                    runtime.dll.cnp_clear_error()
                    self.assertEqual(
                        expected,
                        runtime.dll.cnp_common_type(len(arrays), pointers),
                    )
                    self.assertEqual(0, runtime.error_state().status)

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_error_print_and_buffer_state_match_numpy_and_reject_invalid_updates(
        self,
    ) -> None:
        mode_names = ("raise", "ignore", "warn", "call")
        mode_codes = {"warn": 0, "raise": 1, "ignore": 2, "call": 3}
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            original_errors = self._get_error_modes(runtime)
            original_print = self._get_print_options(runtime)
            original_buffer = runtime.dll.cnp_getbufsize()
            numpy_errors = np.geterr()
            numpy_print = np.get_printoptions()
            numpy_buffer = np.getbufsize()
            try:
                np.seterr(
                    divide=mode_names[0],
                    over=mode_names[1],
                    under=mode_names[2],
                    invalid=mode_names[3],
                )
                expected_modes = tuple(
                    mode_codes[np.geterr()[name]]
                    for name in ("divide", "over", "under", "invalid")
                )
                runtime.dll.cnp_clear_error()
                runtime.dll.cnp_seterr(*expected_modes)
                self.assertEqual(expected_modes, self._get_error_modes(runtime))
                self.assertEqual(0, runtime.error_state().status)

                before = self._get_error_modes(runtime)
                runtime.dll.cnp_clear_error()
                runtime.dll.cnp_seterr(4, 0, 0, 0)
                self.assertEqual(before, self._get_error_modes(runtime))
                self.assertEqual("cnp_seterr", runtime.error_state().function)

                requested_print = (5, 12, 2, 91, 1)
                np.set_printoptions(
                    precision=requested_print[0],
                    threshold=requested_print[1],
                    edgeitems=requested_print[2],
                    linewidth=requested_print[3],
                    suppress=bool(requested_print[4]),
                )
                expected_print = np.get_printoptions()
                runtime.dll.cnp_clear_error()
                runtime.dll.cnp_set_printoptions(*requested_print)
                self.assertEqual(
                    (
                        expected_print["precision"],
                        expected_print["threshold"],
                        expected_print["edgeitems"],
                        expected_print["linewidth"],
                        int(expected_print["suppress"]),
                    ),
                    self._get_print_options(runtime),
                )
                self.assertEqual(0, runtime.error_state().status)

                before = self._get_print_options(runtime)
                runtime.dll.cnp_clear_error()
                runtime.dll.cnp_set_printoptions(-2, -1, -1, -1, 0)
                self.assertEqual(before, self._get_print_options(runtime))
                self.assertEqual(
                    "cnp_set_printoptions", runtime.error_state().function
                )

                np.setbufsize(16)
                self.assertEqual(16, np.getbufsize())
                runtime.dll.cnp_clear_error()
                self.assertEqual(0, runtime.dll.cnp_setbufsize(16))
                self.assertEqual(16, runtime.dll.cnp_getbufsize())
                self.assertEqual(0, runtime.error_state().status)

                for invalid in (1, 17, 10_000_016):
                    with self.subTest(invalid_buffer_size=invalid):
                        before = runtime.dll.cnp_getbufsize()
                        runtime.dll.cnp_clear_error()
                        self.assertNotEqual(0, runtime.dll.cnp_setbufsize(invalid))
                        self.assertEqual(before, runtime.dll.cnp_getbufsize())
                        self.assertEqual(
                            "cnp_setbufsize", runtime.error_state().function
                        )
                self.assertEqual(baseline, runtime.retained_bytes)
            finally:
                runtime.dll.cnp_seterr(*original_errors)
                runtime.dll.cnp_set_printoptions(*original_print)
                runtime.dll.cnp_setbufsize(original_buffer)
                np.seterr(**numpy_errors)
                np.set_printoptions(**numpy_print)
                np.setbufsize(numpy_buffer)

    def test_format_float_matches_numpy_and_buffer_failure_is_atomic(self) -> None:
        cases = (
            (1.25, 2),
            (1.0, 3),
            (-0.0, 2),
            (12345.0, 1),
            (np.inf, 4),
            (np.nan, 4),
        )
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for scientific in (False, True):
                formatter = (
                    np.format_float_scientific
                    if scientific
                    else np.format_float_positional
                )
                for value, precision in cases:
                    kwargs = {
                        "precision": precision,
                        "unique": False,
                        "trim": "k",
                    }
                    if not scientific:
                        kwargs["fractional"] = True
                    expected = formatter(value, **kwargs)
                    buffer = ctypes.create_string_buffer(128)
                    runtime.dll.cnp_clear_error()
                    status = runtime.dll.cnp_format_float(
                        value, buffer, len(buffer), precision, scientific
                    )
                    with self.subTest(
                        value=value,
                        precision=precision,
                        scientific=scientific,
                    ):
                        self.assertEqual(0, status, runtime.error_state())
                        self.assertEqual(expected, buffer.value.decode("ascii"))

            buffer = ctypes.create_string_buffer(b"KEEP", 5)
            before = bytes(buffer)
            runtime.dll.cnp_clear_error()
            status = runtime.dll.cnp_format_float(12345.0, buffer, 5, 4, True)
            self.assertNotEqual(0, status)
            self.assertEqual(before, bytes(buffer))
            self.assertEqual("cnp_format_float", runtime.error_state().function)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_invalid_common_type_is_explicit_and_nonretaining(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            runtime.dll.cnp_clear_error()
            self.assertEqual(24, runtime.dll.cnp_common_type(0, None))
            self.assertEqual(0, runtime.error_state().status)
            runtime.dll.cnp_clear_error()
            self.assertEqual(0, runtime.dll.cnp_common_type(1, None))
            self.assertEqual("cnp_common_type", runtime.error_state().function)
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
