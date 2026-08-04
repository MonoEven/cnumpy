from __future__ import annotations

from typing import Any

import numpy as np

from compat.cnumpy_ctypes import CnumpyArray


def assert_array_equivalent(
    test_case: Any,
    actual: CnumpyArray,
    expected: np.ndarray,
    *,
    compare_strides: bool = False,
    compare_contiguity: bool = True,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    expected_array = np.asarray(expected)
    test_case.assertEqual(expected_array.shape, actual.shape)
    test_case.assertEqual(expected_array.dtype, actual.numpy_dtype)
    if compare_strides:
        test_case.assertEqual(expected_array.strides, actual.strides)
    if compare_contiguity:
        test_case.assertEqual(
            bool(expected_array.flags.c_contiguous), actual.c_contiguous
        )
        test_case.assertEqual(
            bool(expected_array.flags.f_contiguous), actual.f_contiguous
        )

    actual_array = actual.to_numpy()
    if rtol is None and atol is None:
        np.testing.assert_array_equal(actual_array, expected_array, strict=True)
    else:
        np.testing.assert_allclose(
            actual_array,
            expected_array,
            rtol=0.0 if rtol is None else rtol,
            atol=0.0 if atol is None else atol,
            equal_nan=True,
        )

    if np.issubdtype(expected_array.dtype, np.floating):
        expected_zero = expected_array == 0
        actual_zero = actual_array == 0
        np.testing.assert_array_equal(actual_zero, expected_zero, strict=True)
        np.testing.assert_array_equal(
            np.signbit(actual_array[actual_zero]),
            np.signbit(expected_array[expected_zero]),
            strict=True,
        )
