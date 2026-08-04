from __future__ import annotations

import ctypes
from contextlib import ExitStack
from pathlib import Path
import unittest

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"


class WindowSurfaceSemanticsTests(unittest.TestCase):
    SYMBOLS = frozenset(
        {
            "cnp_bartlett",
            "cnp_blackman",
            "cnp_hamming",
            "cnp_hanning",
            "cnp_kaiser",
        }
    )

    def test_standard_windows_match_numpy_125_values_lengths_and_lifetimes(
        self,
    ) -> None:
        cases = (
            ("cnp_bartlett", np.bartlett),
            ("cnp_blackman", np.blackman),
            ("cnp_hamming", np.hamming),
            ("cnp_hanning", np.hanning),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for symbol, oracle in cases:
                function = getattr(runtime.dll, symbol)
                function.argtypes = [ctypes.c_int64]
                function.restype = ctypes.c_void_p
                for length in (-4, 0, 1, 2, 7, 64):
                    with self.subTest(symbol=symbol, length=length):
                        runtime.dll.cnp_clear_error()
                        pointer = function(length)
                        self.assertTrue(pointer, runtime.error_state())
                        with runtime._owned_result(pointer, symbol) as actual:
                            actual_value = actual.to_numpy()
                            self.assertEqual(np.dtype(np.float64), actual_value.dtype)
                            np.testing.assert_allclose(
                                oracle(length),
                                actual_value,
                                rtol=2e-15,
                                atol=2e-15,
                            )
                        self.assertEqual(0, runtime.error_state().status)
                        self.assertEqual(baseline, runtime.retained_bytes)

    def test_kaiser_matches_numpy_125_beta_edges_and_releases_every_result(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            function = runtime.dll.cnp_kaiser
            function.argtypes = [ctypes.c_int64, ctypes.c_double]
            function.restype = ctypes.c_void_p
            for length in (-2, 0, 1, 2, 9, 64):
                for beta in (0.0, -5.0, 5.0, 14.0, np.nan, np.inf):
                    with self.subTest(length=length, beta=beta):
                        runtime.dll.cnp_clear_error()
                        pointer = function(length, beta)
                        self.assertTrue(pointer, runtime.error_state())
                        with runtime._owned_result(pointer, "cnp_kaiser") as actual:
                            actual_value = actual.to_numpy()
                            self.assertEqual(np.dtype(np.float64), actual_value.dtype)
                            with np.errstate(all="ignore"):
                                expected = np.kaiser(length, beta)
                            np.testing.assert_allclose(
                                expected,
                                actual_value,
                                rtol=2e-13,
                                atol=2e-15,
                                equal_nan=True,
                            )
                        self.assertEqual(0, runtime.error_state().status)
                        self.assertEqual(baseline, runtime.retained_bytes)


class LegacyFinancialSemanticsTests(unittest.TestCase):
    SYMBOLS = frozenset(
        {"cnp_fv", "cnp_pv", "cnp_pmt", "cnp_nper", "cnp_rate", "cnp_npv", "cnp_irr"}
    )

    @staticmethod
    def _scalar(runtime: CnumpyRuntime, name: str, argtypes, arguments) -> float:
        function = getattr(runtime.dll, name)
        function.argtypes = argtypes
        function.restype = ctypes.c_double
        runtime.dll.cnp_clear_error()
        return float(function(*arguments))

    def test_numpy_125_exposes_removed_financial_stubs(self) -> None:
        calls = {
            "fv": (0.05, 10, -100.0, 500.0),
            "pv": (0.05, 10, -100.0, 500.0),
            "pmt": (0.05, 10, 500.0),
            "nper": (0.05, -100.0, 500.0),
            "rate": (10, -100.0, 500.0),
            "npv": (0.05, [-100.0, 120.0]),
            "irr": ([-100.0, 120.0],),
        }
        for name, arguments in calls.items():
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                getattr(np, name)(*arguments)

    def test_financial_equations_inverse_roots_and_repeated_calls(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for when in (0, 1):
                for rate in (0.0, 0.025, -0.015):
                    with self.subTest(when=when, rate=rate):
                        nper = 24
                        payment = -125.0
                        present = 2500.0
                        future = self._scalar(
                            runtime,
                            "cnp_fv",
                            [ctypes.c_double, ctypes.c_int64, ctypes.c_double, ctypes.c_double, ctypes.c_int],
                            (rate, nper, payment, present, when),
                        )
                        recovered_present = self._scalar(
                            runtime,
                            "cnp_pv",
                            [ctypes.c_double, ctypes.c_int64, ctypes.c_double, ctypes.c_double, ctypes.c_int],
                            (rate, nper, payment, future, when),
                        )
                        recovered_payment = self._scalar(
                            runtime,
                            "cnp_pmt",
                            [ctypes.c_double, ctypes.c_int64, ctypes.c_double, ctypes.c_double, ctypes.c_int],
                            (rate, nper, present, future, when),
                        )
                        recovered_nper = self._scalar(
                            runtime,
                            "cnp_nper",
                            [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int],
                            (rate, payment, present, future, when),
                        )
                        recovered_rate = self._scalar(
                            runtime,
                            "cnp_rate",
                            [ctypes.c_int64, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int],
                            (nper, payment, present, future, when),
                        )
                        self.assertAlmostEqual(present, recovered_present, places=9)
                        self.assertAlmostEqual(payment, recovered_payment, places=9)
                        self.assertAlmostEqual(float(nper), recovered_nper, places=8)
                        self.assertAlmostEqual(rate, recovered_rate, places=9)
                        self.assertEqual(0, runtime.error_state().status)

            cashflows = np.asarray([-100.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
            buffer = cashflows.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            npv = self._scalar(
                runtime,
                "cnp_npv",
                [ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_int64],
                (0.08, buffer, cashflows.size),
            )
            expected_npv = sum(value / 1.08**index for index, value in enumerate(cashflows))
            self.assertAlmostEqual(expected_npv, npv, places=12)
            irr = self._scalar(
                runtime,
                "cnp_irr",
                [ctypes.POINTER(ctypes.c_double), ctypes.c_int64],
                (buffer, cashflows.size),
            )
            residual = sum(value / (1.0 + irr) ** index for index, value in enumerate(cashflows))
            self.assertAlmostEqual(0.0, residual, places=9)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_invalid_financial_requests_are_explicit_and_nonretaining(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            invalid_calls = (
                ("cnp_fv", [ctypes.c_double, ctypes.c_int64, ctypes.c_double, ctypes.c_double, ctypes.c_int], (0.05, 10, -1.0, 1.0, 2)),
                ("cnp_pv", [ctypes.c_double, ctypes.c_int64, ctypes.c_double, ctypes.c_double, ctypes.c_int], (0.05, 10, -1.0, 1.0, -1)),
                ("cnp_pmt", [ctypes.c_double, ctypes.c_int64, ctypes.c_double, ctypes.c_double, ctypes.c_int], (0.0, 0, 1.0, 1.0, 0)),
                ("cnp_nper", [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int], (0.0, 0.0, 1.0, 1.0, 0)),
                ("cnp_rate", [ctypes.c_int64, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int], (0, -1.0, 1.0, 1.0, 0)),
                ("cnp_npv", [ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_int64], (0.05, None, 1)),
                ("cnp_irr", [ctypes.POINTER(ctypes.c_double), ctypes.c_int64], (None, 2)),
            )
            for name, argtypes, arguments in invalid_calls:
                with self.subTest(name=name):
                    value = self._scalar(runtime, name, argtypes, arguments)
                    self.assertTrue(np.isnan(value))
                    error = runtime.error_state()
                    self.assertNotEqual(0, error.status)
                    self.assertEqual(name, error.function)
                    self.assertEqual(baseline, runtime.retained_bytes)

            same_sign = np.asarray([10.0, 20.0, 30.0], dtype=np.float64)
            value = self._scalar(
                runtime,
                "cnp_irr",
                [ctypes.POINTER(ctypes.c_double), ctypes.c_int64],
                (same_sign.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), same_sign.size),
            )
            self.assertTrue(np.isnan(value))
            self.assertEqual("cnp_irr", runtime.error_state().function)
            self.assertEqual(baseline, runtime.retained_bytes)


class WindowMutationSemanticsTests(unittest.TestCase):
    SYMBOLS = frozenset({"cnp_choose", "cnp_place", "cnp_put", "cnp_putmask"})

    def test_place_matches_numpy_flat_mask_repetition_precision_and_errors(self) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            function = runtime.dll.cnp_place
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            function.restype = ctypes.c_int
            original = np.asarray(
                [2**60 + 1, 2**60 + 2, 2**60 + 3, 2**60 + 4],
                dtype=np.int64,
            )
            mask_value = np.asarray([[True, False, True, True]], dtype=np.bool_)
            values_value = np.asarray([-(2**60) + 7, 19], dtype=np.int64)
            expected = original.copy()
            np.place(expected, mask_value, values_value)
            destination = stack.enter_context(runtime.from_numpy(original))
            mask = stack.enter_context(runtime.from_numpy(mask_value))
            values = stack.enter_context(runtime.from_numpy(values_value))
            self.assertEqual(0, function(destination.pointer, mask.pointer, values.pointer))
            np.testing.assert_array_equal(expected, destination.to_numpy(), strict=True)

            unchanged = destination.to_numpy().copy()
            wrong_mask = stack.enter_context(
                runtime.from_numpy(np.asarray([True, False], dtype=np.bool_))
            )
            empty_values = stack.enter_context(
                runtime.from_numpy(np.asarray([], dtype=np.int64))
            )
            unsafe_values = stack.enter_context(
                runtime.from_numpy(np.asarray([1.5], dtype=np.float64))
            )
            for arguments in (
                (destination.pointer, wrong_mask.pointer, values.pointer),
                (destination.pointer, mask.pointer, empty_values.pointer),
                (destination.pointer, mask.pointer, unsafe_values.pointer),
                (None, mask.pointer, values.pointer),
                (destination.pointer, None, values.pointer),
                (destination.pointer, mask.pointer, None),
            ):
                runtime.dll.cnp_clear_error()
                self.assertNotEqual(0, function(*arguments))
                error = runtime.error_state()
                self.assertEqual("cnp_place", error.function)
                np.testing.assert_array_equal(
                    unchanged, destination.to_numpy(), strict=True
                )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_put_matches_numpy_modes_order_precision_and_atomic_errors(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            function = runtime.dll.cnp_put
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_char_p,
            ]
            function.restype = ctypes.c_int
            original = np.asarray(
                [2**60 + index for index in range(5)], dtype=np.int64
            )
            indices_value = np.asarray([-5, -1, 4, 7], dtype=np.int64)
            values_value = np.asarray(
                [-(2**60) + 1, -(2**60) + 2, -(2**60) + 3, -(2**60) + 4],
                dtype=np.int64,
            )
            for mode in ("wrap", "clip"):
                with self.subTest(mode=mode), ExitStack() as stack:
                    expected = original.copy()
                    np.put(expected, indices_value, values_value, mode=mode)
                    destination = stack.enter_context(runtime.from_numpy(original))
                    indices = stack.enter_context(runtime.from_numpy(indices_value))
                    values = stack.enter_context(runtime.from_numpy(values_value))
                    self.assertEqual(
                        0,
                        function(
                            destination.pointer,
                            indices.pointer,
                            values.pointer,
                            mode.encode(),
                        ),
                    )
                    np.testing.assert_array_equal(
                        expected, destination.to_numpy(), strict=True
                    )
                self.assertEqual(baseline, runtime.retained_bytes)

            with ExitStack() as stack:
                destination = stack.enter_context(runtime.from_numpy(original))
                invalid_indices = stack.enter_context(
                    runtime.from_numpy(np.asarray([0, 8], dtype=np.int64))
                )
                float_indices_value = np.asarray([0.0, 1.0], dtype=np.float64)
                float_indices = stack.enter_context(
                    runtime.from_numpy(float_indices_value)
                )
                unsafe_values = stack.enter_context(
                    runtime.from_numpy(np.asarray([1.5], dtype=np.float64))
                )
                values = stack.enter_context(runtime.from_numpy(values_value))
                for mode in (b"raise", b"bad"):
                    with self.subTest(error_mode=mode):
                        runtime.dll.cnp_clear_error()
                        self.assertNotEqual(
                            0,
                            function(
                                destination.pointer,
                                invalid_indices.pointer,
                                values.pointer,
                                mode,
                            ),
                        )
                        self.assertEqual("cnp_put", runtime.error_state().function)
                        np.testing.assert_array_equal(
                            original, destination.to_numpy(), strict=True
                        )
                with self.assertRaises(TypeError):
                    numpy_destination = original.copy()
                    np.put(
                        numpy_destination,
                        float_indices_value,
                        values_value,
                        mode="wrap",
                    )
                runtime.dll.cnp_clear_error()
                self.assertNotEqual(
                    0,
                    function(
                        destination.pointer,
                        float_indices.pointer,
                        values.pointer,
                        b"wrap",
                    ),
                )
                self.assertEqual("cnp_put", runtime.error_state().function)
                np.testing.assert_array_equal(
                    original, destination.to_numpy(), strict=True
                )
                runtime.dll.cnp_clear_error()
                self.assertNotEqual(
                    0,
                    function(
                        destination.pointer,
                        invalid_indices.pointer,
                        unsafe_values.pointer,
                        b"clip",
                    ),
                )
                self.assertEqual("cnp_put", runtime.error_state().function)
                np.testing.assert_array_equal(
                    original, destination.to_numpy(), strict=True
                )
                for arguments in (
                    (None, invalid_indices.pointer, values.pointer, b"raise"),
                    (destination.pointer, None, values.pointer, b"raise"),
                    (destination.pointer, invalid_indices.pointer, None, b"raise"),
                ):
                    runtime.dll.cnp_clear_error()
                    self.assertNotEqual(0, function(*arguments))
                    self.assertEqual("cnp_put", runtime.error_state().function)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_putmask_matches_numpy_flat_value_alignment_precision_and_errors(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            function = runtime.dll.cnp_putmask
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            function.restype = ctypes.c_int
            original = np.asarray(
                [2**60 + 1, 2**60 + 2, 2**60 + 3, 2**60 + 4],
                dtype=np.int64,
            )
            mask_value = np.asarray([[True, False, True, True]], dtype=np.bool_)
            values_value = np.asarray([-(2**60) + 7, 19], dtype=np.int64)
            expected = original.copy()
            np.putmask(expected, mask_value, values_value)
            destination = stack.enter_context(runtime.from_numpy(original))
            mask = stack.enter_context(runtime.from_numpy(mask_value))
            values = stack.enter_context(runtime.from_numpy(values_value))
            self.assertEqual(0, function(destination.pointer, mask.pointer, values.pointer))
            np.testing.assert_array_equal(expected, destination.to_numpy(), strict=True)

            unchanged = destination.to_numpy().copy()
            wrong_mask = stack.enter_context(
                runtime.from_numpy(np.asarray([True, False], dtype=np.bool_))
            )
            unsafe_values = stack.enter_context(
                runtime.from_numpy(np.asarray([1.5], dtype=np.float64))
            )
            for arguments in (
                (destination.pointer, wrong_mask.pointer, values.pointer),
                (destination.pointer, mask.pointer, unsafe_values.pointer),
                (None, mask.pointer, values.pointer),
                (destination.pointer, None, values.pointer),
                (destination.pointer, mask.pointer, None),
            ):
                runtime.dll.cnp_clear_error()
                self.assertNotEqual(0, function(*arguments))
                self.assertEqual("cnp_putmask", runtime.error_state().function)
                np.testing.assert_array_equal(
                    unchanged, destination.to_numpy(), strict=True
                )

            empty_values = stack.enter_context(
                runtime.from_numpy(np.asarray([], dtype=np.int64))
            )
            self.assertEqual(
                0,
                function(destination.pointer, mask.pointer, empty_values.pointer),
            )
            np.testing.assert_array_equal(
                unchanged, destination.to_numpy(), strict=True
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_choose_matches_numpy_broadcast_promotion_lifetime_and_errors(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            function = runtime.dll.cnp_choose
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            function.restype = ctypes.c_void_p
            indices_value = np.asarray([[0], [1]], dtype=np.int64)
            choice_values = (
                np.asarray([2**60 + 3, 2**60 + 5, 2**60 + 7], dtype=np.int64),
                np.asarray([-(2**60) + 9], dtype=np.int64),
            )
            expected = np.choose(indices_value, choice_values)
            with ExitStack() as stack:
                indices = stack.enter_context(runtime.from_numpy(indices_value))
                choices = [stack.enter_context(runtime.from_numpy(x)) for x in choice_values]
                handles = (ctypes.c_void_p * len(choices))(
                    *(choice.pointer.value for choice in choices)
                )
                pointer = function(indices.pointer, len(choices), handles)
                result = runtime._owned_result(pointer, "cnp_choose")
            try:
                np.testing.assert_array_equal(expected, result.to_numpy(), strict=True)
            finally:
                result.close()
            self.assertEqual(baseline, runtime.retained_bytes)

            invalid_indices = (
                np.asarray([-1, 0], dtype=np.int64),
                np.asarray([0.0, 1.0], dtype=np.float64),
            )
            for invalid in invalid_indices:
                with self.subTest(dtype=invalid.dtype), ExitStack() as stack:
                    indices = stack.enter_context(runtime.from_numpy(invalid))
                    choices = [
                        stack.enter_context(runtime.from_numpy(x))
                        for x in choice_values
                    ]
                    handles = (ctypes.c_void_p * len(choices))(
                        *(choice.pointer.value for choice in choices)
                    )
                    runtime.dll.cnp_clear_error()
                    pointer = function(indices.pointer, len(choices), handles)
                    if pointer:
                        runtime._owned_result(pointer, "cnp_choose").close()
                    self.assertFalse(pointer)
                    self.assertEqual("cnp_choose", runtime.error_state().function)
                self.assertEqual(baseline, runtime.retained_bytes)

            runtime.dll.cnp_clear_error()
            self.assertFalse(function(None, 0, None))
            self.assertEqual("cnp_choose", runtime.error_state().function)
            self.assertEqual(baseline, runtime.retained_bytes)

if __name__ == "__main__":
    unittest.main()
