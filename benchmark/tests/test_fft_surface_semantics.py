from __future__ import annotations

import ctypes
import unittest
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

from compat.cnumpy_ctypes import CnumpyArray, CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"
CNP_ERR_TYPE = -3
CNP_ERR_SHAPE = -4
CNP_ERR_AXIS = -5
CNP_ERR_VALUE = -13


class FftSurfaceSemanticsTests(unittest.TestCase):
    @staticmethod
    def _bind(runtime: CnumpyRuntime) -> None:
        dll = runtime.dll
        for symbol in ("cnp_fft", "cnp_ifft", "cnp_rfft", "cnp_irfft", "cnp_hfft"):
            function = getattr(dll, symbol)
            function.argtypes = [ctypes.c_void_p, ctypes.c_int64]
            function.restype = ctypes.c_void_p
        for symbol in (
            "cnp_fft2",
            "cnp_ifft2",
            "cnp_fftshift",
            "cnp_ifftshift",
            "cnp_ihfft",
        ):
            function = getattr(dll, symbol)
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_void_p
        for symbol in ("cnp_fftfreq", "cnp_rfftfreq"):
            function = getattr(dll, symbol)
            function.argtypes = [ctypes.c_int64, ctypes.c_double]
            function.restype = ctypes.c_void_p
        for symbol in ("cnp_fftn", "cnp_ifftn", "cnp_rfftn"):
            function = getattr(dll, symbol)
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
            ]
            function.restype = ctypes.c_void_p
        dll.cnp_irfftn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
        ]
        dll.cnp_irfftn.restype = ctypes.c_void_p

    @staticmethod
    @contextmanager
    def _result(
        runtime: CnumpyRuntime, symbol: str, *args: object
    ) -> Iterator[CnumpyArray]:
        pointer = getattr(runtime.dll, symbol)(*args)
        with runtime._owned_result(pointer, symbol) as result:
            yield result

    def _assert_result(
        self,
        actual: CnumpyArray,
        expected: np.ndarray,
        *,
        rtol: float = 1e-10,
        atol: float = 1e-10,
    ) -> None:
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected.dtype, actual.numpy_dtype)
        np.testing.assert_allclose(
            expected,
            actual.to_numpy(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )

    def _assert_error(
        self,
        runtime: CnumpyRuntime,
        symbol: str,
        status: int,
        *args: object,
    ) -> None:
        runtime.dll.cnp_clear_error()
        pointer = getattr(runtime.dll, symbol)(*args)
        if pointer:
            runtime._owned_result(pointer, symbol).close()
        self.assertFalse(pointer)
        error = runtime.error_state()
        self.assertEqual(status, error.status)
        self.assertEqual(symbol, error.function)

    def test_one_dimensional_transforms_match_numpy_last_axis_shape_dtype_and_n(
        self,
    ) -> None:
        real = np.arange(24, dtype=np.float32).reshape(2, 3, 4)[:, :, ::2]
        complex_values = np.asarray(
            [[1 + 2j, 3 - 1j, -2 + 0.5j], [4j, -1 - 2j, 3 + 0j]],
            dtype=np.complex64,
        )
        rfft_input = np.linspace(-2.0, 3.0, 10).reshape(2, 5)
        irfft_input = np.fft.rfft(rfft_input, n=7, axis=-1)
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            with runtime.from_numpy(real) as source:
                with self._result(runtime, "cnp_fft", source.pointer, -1) as actual:
                    self._assert_result(actual, np.fft.fft(real, axis=-1))
            with runtime.from_numpy(complex_values) as source:
                with self._result(runtime, "cnp_fft", source.pointer, 5) as actual:
                    self._assert_result(
                        actual, np.fft.fft(complex_values, n=5, axis=-1)
                    )
                with self._result(runtime, "cnp_ifft", source.pointer, 2) as actual:
                    self._assert_result(
                        actual, np.fft.ifft(complex_values, n=2, axis=-1)
                    )
            with runtime.from_numpy(rfft_input) as source:
                with self._result(runtime, "cnp_rfft", source.pointer, 7) as actual:
                    self._assert_result(
                        actual, np.fft.rfft(rfft_input, n=7, axis=-1)
                    )
            with runtime.from_numpy(irfft_input) as source:
                with self._result(runtime, "cnp_irfft", source.pointer, 7) as actual:
                    self._assert_result(
                        actual, np.fft.irfft(irfft_input, n=7, axis=-1)
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_fft2_and_nd_axis_transforms_match_numpy_and_round_trip(self) -> None:
        values = np.arange(24, dtype=np.float64).reshape(2, 3, 4) - 7.0
        axes_values = (-3, -1)
        axes = (ctypes.c_int * len(axes_values))(*axes_values)
        shape_values = (2, 4)
        shape = (ctypes.c_int64 * len(shape_values))(*shape_values)
        expected_fft2 = np.fft.fft2(values)
        expected_fftn = np.fft.fftn(values, axes=axes_values)
        expected_rfftn = np.fft.rfftn(values, axes=axes_values)
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            with runtime.from_numpy(values) as source:
                with self._result(runtime, "cnp_fft2", source.pointer) as actual:
                    self._assert_result(actual, expected_fft2)
                with self._result(
                    runtime, "cnp_fftn", source.pointer, len(axes_values), axes
                ) as actual:
                    self._assert_result(actual, expected_fftn)
                with self._result(
                    runtime, "cnp_rfftn", source.pointer, len(axes_values), axes
                ) as actual:
                    self._assert_result(actual, expected_rfftn)
            with runtime.from_numpy(expected_fft2) as source:
                with self._result(runtime, "cnp_ifft2", source.pointer) as actual:
                    self._assert_result(actual, np.fft.ifft2(expected_fft2))
            with runtime.from_numpy(expected_fftn) as source:
                with self._result(
                    runtime, "cnp_ifftn", source.pointer, len(axes_values), axes
                ) as actual:
                    self._assert_result(
                        actual, np.fft.ifftn(expected_fftn, axes=axes_values)
                    )
            with runtime.from_numpy(expected_rfftn) as source:
                with self._result(
                    runtime,
                    "cnp_irfftn",
                    source.pointer,
                    len(axes_values),
                    axes,
                    len(shape_values),
                    shape,
                ) as actual:
                    self._assert_result(
                        actual,
                        np.fft.irfftn(
                            expected_rfftn, s=shape_values, axes=axes_values
                        ),
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_hermitian_transforms_match_numpy_default_and_explicit_lengths(
        self,
    ) -> None:
        hermitian = np.asarray([1 + 0j, 2 + 3j, -4 + 1j, 5 + 0j])
        real = np.asarray([0.5, -1.0, 2.0, 4.0, -3.0, 1.0, 0.25])
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            with runtime.from_numpy(hermitian) as source:
                for length in (-1, 8):
                    with self.subTest(symbol="cnp_hfft", length=length):
                        with self._result(
                            runtime, "cnp_hfft", source.pointer, length
                        ) as actual:
                            expected = np.fft.hfft(
                                hermitian, n=None if length < 0 else length
                            )
                            self._assert_result(actual, expected)
            with runtime.from_numpy(real) as source:
                with self._result(runtime, "cnp_ihfft", source.pointer) as actual:
                    self._assert_result(actual, np.fft.ihfft(real))
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_frequency_helpers_and_all_axis_shifts_match_numpy(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for symbol, oracle, length, spacing in (
                ("cnp_fftfreq", np.fft.fftfreq, 5, -0.5),
                ("cnp_fftfreq", np.fft.fftfreq, 6, 2.0),
                ("cnp_rfftfreq", np.fft.rfftfreq, 5, -0.5),
                ("cnp_rfftfreq", np.fft.rfftfreq, 6, 2.0),
            ):
                with self.subTest(symbol=symbol, length=length, spacing=spacing):
                    with self._result(
                        runtime, symbol, length, spacing
                    ) as actual:
                        self._assert_result(actual, oracle(length, d=spacing))

            shift_cases = (
                (
                    "cnp_fftshift",
                    np.arange(24, dtype=np.int16).reshape(2, 3, 4),
                    np.fft.fftshift,
                ),
                (
                    "cnp_ifftshift",
                    np.asarray(
                        [[1 + 2j, 3 - 4j, 5j, -2], [4, -1j, 2 + 3j, 7]],
                        dtype=np.complex128,
                    ),
                    np.fft.ifftshift,
                ),
            )
            for symbol, values, oracle in shift_cases:
                with runtime.from_numpy(values) as source:
                    with self._result(runtime, symbol, source.pointer) as actual:
                        expected = oracle(values)
                        self.assertEqual(expected.shape, actual.shape)
                        self.assertEqual(expected.dtype, actual.numpy_dtype)
                        np.testing.assert_array_equal(expected, actual.to_numpy())
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_errors_complex_rfft_projection_and_repeated_lifetime(self) -> None:
        complex_values = np.asarray([1 + 2j, 3 - 4j, -2 + 0.5j])
        real = np.arange(8, dtype=np.float64)
        empty = np.asarray([], dtype=np.float64)
        scalar = np.asarray(2.0)
        bad_axis = (ctypes.c_int * 1)(2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
            expected_complex_rfft = np.fft.rfft(complex_values)
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            with runtime.from_numpy(complex_values) as source:
                with self._result(runtime, "cnp_rfft", source.pointer, -1) as actual:
                    self._assert_result(actual, expected_complex_rfft)
            with runtime.from_numpy(scalar) as source:
                self._assert_error(
                    runtime, "cnp_fft", CNP_ERR_SHAPE, source.pointer, -1
                )
            with runtime.from_numpy(empty) as source:
                self._assert_error(
                    runtime, "cnp_fft", CNP_ERR_VALUE, source.pointer, -1
                )
            with runtime.from_numpy(real) as source:
                self._assert_error(
                    runtime, "cnp_fft", CNP_ERR_VALUE, source.pointer, 0
                )
                self._assert_error(
                    runtime, "cnp_fftn", CNP_ERR_AXIS, source.pointer, 1, bad_axis
                )
            self._assert_error(runtime, "cnp_fftfreq", CNP_ERR_VALUE, 0, 1.0)
            self._assert_error(runtime, "cnp_rfftfreq", CNP_ERR_VALUE, 8, 0.0)

            with runtime.from_numpy(real) as source:
                active = runtime.retained_bytes
                for _ in range(32):
                    for symbol, args in (
                        ("cnp_fft", (source.pointer, -1)),
                        ("cnp_ifft", (source.pointer, -1)),
                        ("cnp_rfft", (source.pointer, -1)),
                        ("cnp_fftshift", (source.pointer,)),
                        ("cnp_ifftshift", (source.pointer,)),
                        ("cnp_hfft", (source.pointer, -1)),
                        ("cnp_ihfft", (source.pointer,)),
                    ):
                        with self._result(runtime, symbol, *args):
                            pass
                self.assertEqual(active, runtime.retained_bytes)
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
