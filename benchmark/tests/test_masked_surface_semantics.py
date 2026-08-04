from __future__ import annotations

import ctypes
from contextlib import ExitStack
from pathlib import Path
import unittest

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"


def _function(runtime: CnumpyRuntime, name: str, argtypes, restype):
    function = getattr(runtime.dll, name)
    function.argtypes = argtypes
    function.restype = restype
    return function


class _MaskedOwner:
    def __init__(self, runtime: CnumpyRuntime, pointer, origin: str):
        if not pointer:
            runtime.raise_last_error(origin)
        self.runtime = runtime
        self.pointer = pointer

    def close(self) -> None:
        if self.pointer:
            self.runtime.dll.cnp_masked_array_free(self.pointer)
            self.pointer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _owned_array(runtime: CnumpyRuntime, pointer, origin: str):
    return runtime._owned_result(pointer, origin)


class MaskedSurfaceSemanticsTests(unittest.TestCase):
    SYMBOLS = frozenset(
        {
            "cnp_masked_array_compressed",
            "cnp_masked_array_count",
            "cnp_masked_array_create",
            "cnp_masked_array_filled",
            "cnp_masked_array_free",
            "cnp_masked_array_get_data",
            "cnp_masked_array_get_mask",
            "cnp_masked_array_max",
            "cnp_masked_array_mean",
            "cnp_masked_array_min",
            "cnp_masked_array_set_mask",
            "cnp_masked_array_std",
            "cnp_masked_array_sum",
            "cnp_masked_equal",
            "cnp_masked_greater",
            "cnp_masked_inside",
            "cnp_masked_invalid",
            "cnp_masked_less",
            "cnp_masked_not_equal",
            "cnp_masked_outside",
            "cnp_masked_where",
        }
    )

    def test_masked_array_values_reductions_updates_and_lifetimes_match_numpy_125(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            runtime.dll.cnp_masked_array_free.argtypes = [ctypes.c_void_p]
            runtime.dll.cnp_masked_array_free.restype = None
            values = np.asarray(
                [-(2**60) + 3, -2, 5, 2**60 + 7], dtype=np.int64
            )
            mask_value = np.asarray([False, True, False, True], dtype=np.bool_)
            expected = np.ma.array(values, mask=mask_value, fill_value=-9)
            data = runtime.from_numpy(values)
            mask = runtime.from_numpy(mask_value)
            create = _function(
                runtime,
                "cnp_masked_array_create",
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double],
                ctypes.c_void_p,
            )
            masked = stack.enter_context(
                _MaskedOwner(runtime, create(data.pointer, mask.pointer, -9), "cnp_masked_array_create")
            )
            data.close()
            mask.close()

            get_data = _function(
                runtime,
                "cnp_masked_array_get_data",
                [ctypes.c_void_p],
                ctypes.c_void_p,
            )
            get_mask = _function(
                runtime,
                "cnp_masked_array_get_mask",
                [ctypes.c_void_p],
                ctypes.c_void_p,
            )
            data_view = stack.enter_context(
                _owned_array(runtime, get_data(masked.pointer), "cnp_masked_array_get_data")
            )
            mask_view = stack.enter_context(
                _owned_array(runtime, get_mask(masked.pointer), "cnp_masked_array_get_mask")
            )
            np.testing.assert_array_equal(values, data_view.to_numpy(), strict=True)
            np.testing.assert_array_equal(mask_value, mask_view.to_numpy(), strict=True)

            filled = _function(
                runtime,
                "cnp_masked_array_filled",
                [ctypes.c_void_p, ctypes.c_double],
                ctypes.c_void_p,
            )
            compressed = _function(
                runtime,
                "cnp_masked_array_compressed",
                [ctypes.c_void_p],
                ctypes.c_void_p,
            )
            filled_result = stack.enter_context(
                _owned_array(runtime, filled(masked.pointer, -9), "cnp_masked_array_filled")
            )
            compressed_result = stack.enter_context(
                _owned_array(runtime, compressed(masked.pointer), "cnp_masked_array_compressed")
            )
            np.testing.assert_array_equal(
                expected.filled(-9), filled_result.to_numpy(), strict=True
            )
            np.testing.assert_array_equal(
                expected.compressed(), compressed_result.to_numpy(), strict=True
            )

            scalar_calls = {
                "cnp_masked_array_count": (ctypes.c_int64, expected.count()),
                "cnp_masked_array_sum": (ctypes.c_double, float(expected.sum())),
                "cnp_masked_array_mean": (ctypes.c_double, float(expected.mean())),
                "cnp_masked_array_std": (ctypes.c_double, float(expected.std())),
                "cnp_masked_array_min": (ctypes.c_double, float(expected.min())),
                "cnp_masked_array_max": (ctypes.c_double, float(expected.max())),
            }
            for name, (restype, expected_value) in scalar_calls.items():
                with self.subTest(operation=name):
                    function = _function(runtime, name, [ctypes.c_void_p], restype)
                    self.assertAlmostEqual(expected_value, function(masked.pointer))

            replacement_mask_value = np.asarray(
                [True, False, True, False], dtype=np.bool_
            )
            replacement_mask = runtime.from_numpy(replacement_mask_value)
            set_mask = _function(
                runtime,
                "cnp_masked_array_set_mask",
                [ctypes.c_void_p, ctypes.c_void_p],
                ctypes.c_int,
            )
            self.assertEqual(0, set_mask(masked.pointer, replacement_mask.pointer))
            replacement_mask.close()
            updated = np.ma.array(values, mask=replacement_mask_value)
            updated_result = stack.enter_context(
                _owned_array(runtime, compressed(masked.pointer), "cnp_masked_array_compressed")
            )
            np.testing.assert_array_equal(
                updated.compressed(), updated_result.to_numpy(), strict=True
            )

            masked.close()
            np.testing.assert_array_equal(values, data_view.to_numpy(), strict=True)
            np.testing.assert_array_equal(mask_value, mask_view.to_numpy(), strict=True)
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_mask_constructors_match_numpy_ma_masks_and_release_sources(self) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            runtime.dll.cnp_masked_array_free.argtypes = [ctypes.c_void_p]
            runtime.dll.cnp_masked_array_free.restype = None
            values = np.asarray([-np.inf, -2.0, 0.0, 3.0, np.nan])
            data = stack.enter_context(runtime.from_numpy(values))
            condition_value = np.asarray([0, 1, 0, 1, 0], dtype=np.bool_)
            condition = stack.enter_context(runtime.from_numpy(condition_value))
            cases = (
                (
                    "cnp_masked_where",
                    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double],
                    (condition.pointer, data.pointer, -7.0),
                    np.ma.masked_where(condition_value, values).mask,
                ),
                (
                    "cnp_masked_invalid",
                    [ctypes.c_void_p, ctypes.c_double],
                    (data.pointer, -7.0),
                    np.ma.masked_invalid(values).mask,
                ),
                (
                    "cnp_masked_greater",
                    [ctypes.c_void_p, ctypes.c_double, ctypes.c_double],
                    (data.pointer, 1.0, -7.0),
                    np.ma.masked_greater(values, 1.0).mask,
                ),
                (
                    "cnp_masked_less",
                    [ctypes.c_void_p, ctypes.c_double, ctypes.c_double],
                    (data.pointer, -1.0, -7.0),
                    np.ma.masked_less(values, -1.0).mask,
                ),
                (
                    "cnp_masked_equal",
                    [ctypes.c_void_p, ctypes.c_double, ctypes.c_double],
                    (data.pointer, 0.0, -7.0),
                    np.ma.masked_equal(values, 0.0).mask,
                ),
                (
                    "cnp_masked_not_equal",
                    [ctypes.c_void_p, ctypes.c_double, ctypes.c_double],
                    (data.pointer, 0.0, -7.0),
                    np.ma.masked_not_equal(values, 0.0).mask,
                ),
                (
                    "cnp_masked_inside",
                    [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double],
                    (data.pointer, -1.0, 2.0, -7.0),
                    np.ma.masked_inside(values, -1.0, 2.0).mask,
                ),
                (
                    "cnp_masked_outside",
                    [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double],
                    (data.pointer, -1.0, 2.0, -7.0),
                    np.ma.masked_outside(values, -1.0, 2.0).mask,
                ),
            )
            get_mask = _function(
                runtime,
                "cnp_masked_array_get_mask",
                [ctypes.c_void_p],
                ctypes.c_void_p,
            )
            for name, argtypes, arguments, expected_mask in cases:
                with self.subTest(operation=name):
                    function = _function(runtime, name, argtypes, ctypes.c_void_p)
                    owner = _MaskedOwner(runtime, function(*arguments), name)
                    mask_result = _owned_array(runtime, get_mask(owner.pointer), name)
                    owner.close()
                    try:
                        np.testing.assert_array_equal(
                            np.asarray(expected_mask, dtype=np.bool_),
                            mask_result.to_numpy(),
                            strict=True,
                        )
                    finally:
                        mask_result.close()
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_invalid_masked_requests_are_explicit_atomic_and_nonretaining(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            runtime.dll.cnp_masked_array_free.argtypes = [ctypes.c_void_p]
            runtime.dll.cnp_masked_array_free.restype = None
            pointer_calls = (
                ("cnp_masked_array_create", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double], (None, None, 0.0)),
                ("cnp_masked_array_get_data", [ctypes.c_void_p], (None,)),
                ("cnp_masked_array_get_mask", [ctypes.c_void_p], (None,)),
                ("cnp_masked_array_filled", [ctypes.c_void_p, ctypes.c_double], (None, 0.0)),
                ("cnp_masked_array_compressed", [ctypes.c_void_p], (None,)),
                ("cnp_masked_where", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double], (None, None, 0.0)),
                ("cnp_masked_invalid", [ctypes.c_void_p, ctypes.c_double], (None, 0.0)),
                ("cnp_masked_greater", [ctypes.c_void_p, ctypes.c_double, ctypes.c_double], (None, 0.0, 0.0)),
                ("cnp_masked_less", [ctypes.c_void_p, ctypes.c_double, ctypes.c_double], (None, 0.0, 0.0)),
                ("cnp_masked_equal", [ctypes.c_void_p, ctypes.c_double, ctypes.c_double], (None, 0.0, 0.0)),
                ("cnp_masked_not_equal", [ctypes.c_void_p, ctypes.c_double, ctypes.c_double], (None, 0.0, 0.0)),
                ("cnp_masked_inside", [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double], (None, 0.0, 1.0, 0.0)),
                ("cnp_masked_outside", [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double], (None, 0.0, 1.0, 0.0)),
            )
            for name, argtypes, arguments in pointer_calls:
                with self.subTest(operation=name):
                    function = _function(runtime, name, argtypes, ctypes.c_void_p)
                    runtime.dll.cnp_clear_error()
                    self.assertFalse(function(*arguments))
                    error = runtime.error_state()
                    self.assertNotEqual(0, error.status)
                    self.assertEqual(name, error.function)
                    self.assertEqual(baseline, runtime.retained_bytes)

    def test_invalid_mask_shapes_and_types_are_rejected_atomically(self) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            runtime.dll.cnp_masked_array_free.argtypes = [ctypes.c_void_p]
            runtime.dll.cnp_masked_array_free.restype = None
            data = stack.enter_context(
                runtime.from_numpy(np.asarray([1.0, 2.0, 3.0]))
            )
            valid_mask_value = np.asarray([False, True, False], dtype=np.bool_)
            valid_mask = stack.enter_context(runtime.from_numpy(valid_mask_value))
            wrong_type = stack.enter_context(
                runtime.from_numpy(np.asarray([0, 1, 0], dtype=np.int64))
            )
            wrong_shape = stack.enter_context(
                runtime.from_numpy(np.asarray([False, True], dtype=np.bool_))
            )
            create = _function(
                runtime,
                "cnp_masked_array_create",
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double],
                ctypes.c_void_p,
            )
            for invalid_mask in (wrong_type, wrong_shape):
                with self.subTest(operation="create", dtype=invalid_mask.to_numpy().dtype):
                    runtime.dll.cnp_clear_error()
                    unexpected = create(data.pointer, invalid_mask.pointer, 0.0)
                    if unexpected:
                        runtime.dll.cnp_masked_array_free(unexpected)
                    self.assertFalse(unexpected)
                    self.assertEqual(
                        "cnp_masked_array_create", runtime.error_state().function
                    )

            owner = stack.enter_context(
                _MaskedOwner(
                    runtime,
                    create(data.pointer, valid_mask.pointer, 0.0),
                    "cnp_masked_array_create",
                )
            )
            get_mask = _function(
                runtime,
                "cnp_masked_array_get_mask",
                [ctypes.c_void_p],
                ctypes.c_void_p,
            )
            set_mask = _function(
                runtime,
                "cnp_masked_array_set_mask",
                [ctypes.c_void_p, ctypes.c_void_p],
                ctypes.c_int,
            )
            for invalid_mask in (wrong_type, wrong_shape):
                with self.subTest(operation="set_mask", dtype=invalid_mask.to_numpy().dtype):
                    runtime.dll.cnp_clear_error()
                    self.assertNotEqual(
                        0, set_mask(owner.pointer, invalid_mask.pointer)
                    )
                    self.assertEqual(
                        "cnp_masked_array_set_mask", runtime.error_state().function
                    )
                    current = _owned_array(
                        runtime,
                        get_mask(owner.pointer),
                        "cnp_masked_array_get_mask",
                    )
                    try:
                        np.testing.assert_array_equal(
                            valid_mask_value, current.to_numpy(), strict=True
                        )
                    finally:
                        current.close()

            masked_where = _function(
                runtime,
                "cnp_masked_where",
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double],
                ctypes.c_void_p,
            )
            runtime.dll.cnp_clear_error()
            unexpected = masked_where(wrong_shape.pointer, data.pointer, 0.0)
            if unexpected:
                runtime.dll.cnp_masked_array_free(unexpected)
            self.assertFalse(unexpected)
            self.assertEqual("cnp_masked_where", runtime.error_state().function)

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

            set_mask = _function(
                runtime,
                "cnp_masked_array_set_mask",
                [ctypes.c_void_p, ctypes.c_void_p],
                ctypes.c_int,
            )
            runtime.dll.cnp_clear_error()
            self.assertNotEqual(0, set_mask(None, None))
            self.assertEqual(
                "cnp_masked_array_set_mask", runtime.error_state().function
            )

            scalar_calls = (
                ("cnp_masked_array_count", ctypes.c_int64),
                ("cnp_masked_array_sum", ctypes.c_double),
                ("cnp_masked_array_mean", ctypes.c_double),
                ("cnp_masked_array_std", ctypes.c_double),
                ("cnp_masked_array_min", ctypes.c_double),
                ("cnp_masked_array_max", ctypes.c_double),
            )
            for name, restype in scalar_calls:
                with self.subTest(operation=name):
                    function = _function(runtime, name, [ctypes.c_void_p], restype)
                    runtime.dll.cnp_clear_error()
                    value = function(None)
                    error = runtime.error_state()
                    self.assertNotEqual(0, error.status)
                    self.assertEqual(name, error.function)
                    if restype is ctypes.c_double:
                        self.assertTrue(np.isnan(value))
                    else:
                        self.assertEqual(-1, value)
                    self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
