from __future__ import annotations

from contextlib import ExitStack
import ctypes
from pathlib import Path
import unittest

import numpy as np

from compat.cnumpy_ctypes import CnumpyError, CnumpyRuntime
from compat.oracle import assert_array_equivalent


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"
CNP_BITORDER_BIG = 0
CNP_BITORDER_LITTLE = 1
CNP_ORDER_C = 0
CNP_BOOL = 1
CNP_UBYTE = 3
SCIPY_SPECIAL_REFERENCE = "scipy.special 1.12.0"


def reference_softmax(value: np.ndarray, axis: int) -> np.ndarray:
    calculation_dtype = (
        np.float32 if value.dtype == np.dtype(np.float32) else np.float64
    )
    value = value.astype(calculation_dtype, copy=False)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        maximum = np.amax(value, axis=axis, keepdims=True)
        shifted = value - maximum
        exponentials = np.exp(shifted)
        return exponentials / np.sum(
            exponentials, axis=axis, keepdims=True
        )


def reference_log_softmax(value: np.ndarray, axis: int) -> np.ndarray:
    calculation_dtype = (
        np.float32 if value.dtype == np.dtype(np.float32) else np.float64
    )
    value = value.astype(calculation_dtype, copy=False)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        maximum = np.amax(value, axis=axis, keepdims=True)
        maximum = np.where(
            np.isfinite(maximum),
            maximum,
            np.asarray(0, dtype=maximum.dtype),
        )
        shifted = value - maximum
        log_total = np.log(
            np.sum(np.exp(shifted), axis=axis, keepdims=True)
        )
        return shifted - log_total


class MiscAxisSemanticsTests(unittest.TestCase):
    def required_export(self, runtime: CnumpyRuntime, name: str):
        try:
            return getattr(runtime.dll, name)
        except AttributeError:
            self.fail(f"required native export is missing: {name}")

    def assert_native_error(
        self,
        call,
        status: int,
        function_name: str,
        message: str,
    ) -> None:
        try:
            unexpected_result = call()
        except CnumpyError as error:
            self.assertEqual(status, error.status)
            self.assertEqual(function_name, error.function)
            self.assertIn(message, error.message)
            return
        unexpected_result.close()
        self.fail(
            f"{function_name} unexpectedly succeeded; expected {message!r}"
        )

    def unary_axis_result(
        self,
        runtime: CnumpyRuntime,
        source,
        function_name: str,
        axis: int,
    ):
        function = self.required_export(runtime, function_name)
        function.argtypes = [ctypes.c_void_p, ctypes.c_int]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        pointer = function(source.pointer, int(axis))
        return runtime._owned_result(pointer, function_name)

    def trapz_result(
        self,
        runtime: CnumpyRuntime,
        y,
        x,
        dx: float,
        axis: int,
    ):
        function = self.required_export(runtime, "cnp_trapz")
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_int,
        ]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        pointer = function(
            y.pointer,
            None if x is None else x.pointer,
            float(dx),
            int(axis),
        )
        return runtime._owned_result(pointer, "cnp_trapz")

    def packbits_result(
        self,
        runtime: CnumpyRuntime,
        source,
        axis: int | None,
        bitorder: int,
    ):
        function = self.required_export(runtime, "cnp_packbits_v2")
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_bool,
            ctypes.c_int,
        ]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        pointer = function(
            source.pointer,
            0 if axis is None else int(axis),
            axis is None,
            int(bitorder),
        )
        return runtime._owned_result(pointer, "cnp_packbits_v2")

    def unpackbits_result(
        self,
        runtime: CnumpyRuntime,
        source,
        axis: int | None,
        count: int | None,
        bitorder: int,
    ):
        function = self.required_export(runtime, "cnp_unpackbits_v2")
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_bool,
            ctypes.c_int64,
            ctypes.c_bool,
            ctypes.c_int,
        ]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        pointer = function(
            source.pointer,
            0 if axis is None else int(axis),
            axis is None,
            0 if count is None else int(count),
            count is None,
            int(bitorder),
        )
        return runtime._owned_result(pointer, "cnp_unpackbits_v2")

    def native_array_with_shape(
        self,
        runtime: CnumpyRuntime,
        shape: tuple[int, ...],
        dtype: int,
    ):
        function = self.required_export(runtime, "cnp_array_new")
        function.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int,
            ctypes.c_int,
        ]
        function.restype = ctypes.c_void_p
        native_shape = (ctypes.c_int64 * len(shape))(*shape)
        runtime.dll.cnp_clear_error()
        pointer = function(
            len(shape), native_shape, int(dtype), CNP_ORDER_C
        )
        return runtime._owned_result(pointer, "cnp_array_new")

    def legacy_packbits_result(
        self, runtime: CnumpyRuntime, source, axis: int
    ):
        function = self.required_export(runtime, "cnp_packbits")
        function.argtypes = [ctypes.c_void_p, ctypes.c_int]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        pointer = function(source.pointer, int(axis))
        return runtime._owned_result(pointer, "cnp_packbits")

    def legacy_unpackbits_result(
        self, runtime: CnumpyRuntime, source, axis: int, count: int
    ):
        function = self.required_export(runtime, "cnp_unpackbits")
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int64,
        ]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        pointer = function(source.pointer, int(axis), int(count))
        return runtime._owned_result(pointer, "cnp_unpackbits")

    def test_softmax_and_log_softmax_match_stable_axis_reference(
        self,
    ) -> None:
        storage_value = np.array(
            [
                [
                    [1000.0, 1001.0, 999.0, 998.0],
                    [-1000.0, -999.0, -1001.0, -1002.0],
                    [3.0, 1.0, 4.0, 2.0],
                ],
                [
                    [5.0, 7.0, 6.0, 8.0],
                    [20.0, 10.0, 30.0, 0.0],
                    [-4.0, -2.0, -3.0, -1.0],
                ],
            ],
            dtype=np.float64,
        )
        transposed_value = storage_value.transpose(2, 0, 1)
        operations = (
            ("cnp_softmax", reference_softmax),
            ("cnp_log_softmax", reference_log_softmax),
        )

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            storage = stack.enter_context(runtime.from_numpy(storage_value))
            contiguous = storage
            transposed = stack.enter_context(
                runtime.transpose(storage, (2, 0, 1))
            )
            self.assertFalse(transposed.c_contiguous)

            for layout, source, value in (
                ("contiguous", contiguous, storage_value),
                ("transposed", transposed, transposed_value),
            ):
                for function_name, reference in operations:
                    for axis in range(-value.ndim, value.ndim):
                        with self.subTest(
                            layout=layout,
                            function=function_name,
                            axis=axis,
                        ):
                            result = stack.enter_context(
                                self.unary_axis_result(
                                    runtime,
                                    source,
                                    function_name,
                                    axis,
                                )
                            )
                            assert_array_equivalent(
                                self,
                                result,
                                reference(value, axis),
                                compare_contiguity=False,
                                rtol=2e-14,
                                atol=2e-15,
                            )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_softmax_preserves_float32_and_handles_scalar_and_empty_axes(
        self,
    ) -> None:
        float32_value = np.array(
            [[100.0, 101.0, 99.0], [3.0, 1.0, 2.0]],
            dtype=np.float32,
        )
        scalar_value = np.array(7.0, dtype=np.float64)
        empty_nonreduced_value = np.empty((0, 3), dtype=np.float64)
        empty_reduced_value = np.empty((0, 3), dtype=np.float64)

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            float32_source = stack.enter_context(
                runtime.from_numpy(float32_value)
            )
            scalar = stack.enter_context(runtime.from_numpy(scalar_value))
            empty_nonreduced = stack.enter_context(
                runtime.from_numpy(empty_nonreduced_value)
            )
            empty_reduced = stack.enter_context(
                runtime.from_numpy(empty_reduced_value)
            )

            for function_name, reference in (
                ("cnp_softmax", reference_softmax),
                ("cnp_log_softmax", reference_log_softmax),
            ):
                float32_result = stack.enter_context(
                    self.unary_axis_result(
                        runtime, float32_source, function_name, -1
                    )
                )
                assert_array_equivalent(
                    self,
                    float32_result,
                    reference(float32_value, -1),
                    compare_contiguity=False,
                    rtol=2e-6,
                    atol=2e-7,
                )

                for axis in (0, -1):
                    scalar_result = stack.enter_context(
                        self.unary_axis_result(
                            runtime, scalar, function_name, axis
                        )
                    )
                    assert_array_equivalent(
                        self,
                        scalar_result,
                        reference(scalar_value, axis),
                    )

                empty_result = stack.enter_context(
                    self.unary_axis_result(
                        runtime,
                        empty_nonreduced,
                        function_name,
                        1,
                    )
                )
                assert_array_equivalent(
                    self,
                    empty_result,
                    reference(empty_nonreduced_value, 1),
                )

                before_error = runtime.retained_bytes
                with self.assertRaises(CnumpyError) as captured:
                    self.unary_axis_result(
                        runtime, empty_reduced, function_name, 0
                    )
                self.assertEqual(-1, captured.exception.status)
                self.assertEqual(function_name, captured.exception.function)
                self.assertIn(
                    "zero-size reduction", captured.exception.message
                )
                self.assertEqual(before_error, runtime.retained_bytes)

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_softmax_nan_and_infinity_behavior_matches_reference(self) -> None:
        values = (
            np.array([np.inf, 1.0], dtype=np.float64),
            np.array([np.inf, np.inf], dtype=np.float64),
            np.array([-np.inf, -np.inf], dtype=np.float64),
            np.array([np.nan, 1.0], dtype=np.float64),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for value in values:
                source = stack.enter_context(runtime.from_numpy(value))
                for function_name, reference in (
                    ("cnp_softmax", reference_softmax),
                    ("cnp_log_softmax", reference_log_softmax),
                ):
                    with self.subTest(
                        function=function_name, value=repr(value)
                    ):
                        result = stack.enter_context(
                            self.unary_axis_result(
                                runtime, source, function_name, 0
                            )
                        )
                        assert_array_equivalent(
                            self,
                            result,
                            reference(value, 0),
                            compare_contiguity=False,
                        )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_softmax_extremes_signed_zero_and_output_layout(self) -> None:
        values = (
            np.asarray([-0.0, 0.0], dtype=np.float64),
            np.asarray([1.0e308, -1.0e308, 0.0], dtype=np.float64),
            np.asarray([1.0e30, -1.0e30, 0.0], dtype=np.float32),
            np.asarray([np.inf, 1.0, -np.inf], dtype=np.float64),
            np.asarray([-np.inf, -np.inf], dtype=np.float64),
            np.asarray([np.nan, -0.0, 0.0], dtype=np.float64),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for value in values:
                with runtime.from_numpy(value) as source:
                    for function_name, reference in (
                        ("cnp_softmax", reference_softmax),
                        ("cnp_log_softmax", reference_log_softmax),
                    ):
                        with self.subTest(
                            function=function_name,
                            dtype=value.dtype.name,
                            value=repr(value),
                        ):
                            with self.unary_axis_result(
                                runtime, source, function_name, 0
                            ) as result:
                                self.assertTrue(result.c_contiguous)
                                assert_array_equivalent(
                                    self,
                                    result,
                                    reference(value, 0),
                                    compare_contiguity=False,
                                    rtol=(
                                        2e-6
                                        if value.dtype == np.float32
                                        else 2e-14
                                    ),
                                    atol=(
                                        2e-7
                                        if value.dtype == np.float32
                                        else 2e-15
                                    ),
                                )
                                if function_name == "cnp_softmax" and np.array_equal(
                                    value,
                                    np.asarray(
                                        [-0.0, 0.0], dtype=np.float64
                                    ),
                                ):
                                    self.assertFalse(
                                        np.signbit(result.to_numpy()).any()
                                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_softmax_reports_axis_and_null_input_errors(self) -> None:
        source_value = np.arange(6, dtype=np.float64).reshape(2, 3)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(source_value) as source:
                for function_name in ("cnp_softmax", "cnp_log_softmax"):
                    before_error = runtime.retained_bytes
                    self.assert_native_error(
                        lambda function_name=function_name:
                        self.unary_axis_result(
                            runtime, source, function_name, 2
                        ),
                        -5,
                        function_name,
                        "axis 2 is out of bounds",
                    )
                    self.assertEqual(before_error, runtime.retained_bytes)

                    function = self.required_export(runtime, function_name)
                    runtime.dll.cnp_clear_error()
                    pointer = function(None, 0)
                    with self.assertRaises(CnumpyError) as null_error:
                        runtime._owned_result(pointer, function_name)
                    self.assertEqual(
                        function_name, null_error.exception.function
                    )
                    self.assertIn(
                        "must not be null", null_error.exception.message
                    )
                    self.assertEqual(before_error, runtime.retained_bytes)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_softmax_rank_zero_through_four_all_axes_and_real_dtypes(
        self,
    ) -> None:
        self.assertEqual("scipy.special 1.12.0", SCIPY_SPECIAL_REFERENCE)
        dtypes = (
            np.bool_,
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
            np.float16,
            np.float32,
            np.float64,
        )
        shapes = ((), (3,), (2, 3), (2, 2, 3), (2, 2, 2, 3))
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for dtype in dtypes:
                for shape in shapes:
                    value = np.arange(
                        int(np.prod(shape, dtype=np.int64)) or 1,
                        dtype=np.float64,
                    ).reshape(shape)
                    if dtype == np.bool_:
                        value = value % 2 == 0
                    else:
                        value = value.astype(dtype)
                    axes = (0, -1) if not shape else range(-len(shape), len(shape))
                    with runtime.from_numpy(value) as source:
                        for function_name, reference in (
                            ("cnp_softmax", reference_softmax),
                            ("cnp_log_softmax", reference_log_softmax),
                        ):
                            for axis in axes:
                                with self.subTest(
                                    dtype=np.dtype(dtype).name,
                                    rank=len(shape),
                                    function=function_name,
                                    axis=axis,
                                ):
                                    with self.unary_axis_result(
                                        runtime,
                                        source,
                                        function_name,
                                        axis,
                                    ) as result:
                                        assert_array_equivalent(
                                            self,
                                            result,
                                            reference(value, axis),
                                            compare_contiguity=False,
                                            rtol=(
                                                2e-6
                                                if dtype == np.float32
                                                else 2e-14
                                            ),
                                            atol=(
                                                2e-7
                                                if dtype == np.float32
                                                else 2e-15
                                            ),
                                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_task8_ahk_bridges_relabel_core_axis_errors(self) -> None:
        value = np.arange(6, dtype=np.uint8).reshape(2, 3)
        with CnumpyRuntime(DLL) as runtime, runtime.from_numpy(value) as source:
            baseline = runtime.retained_bytes
            calls = (
                (
                    "cnp_ahk_softmax",
                    [ctypes.c_void_p, ctypes.c_int],
                    (source.pointer, 2),
                ),
                (
                    "cnp_ahk_log_softmax",
                    [ctypes.c_void_p, ctypes.c_int],
                    (source.pointer, 2),
                ),
                (
                    "cnp_ahk_trapz",
                    [
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                        ctypes.c_double,
                        ctypes.c_int,
                    ],
                    (source.pointer, None, 1.0, 2),
                ),
                (
                    "cnp_ahk_packbits_v2",
                    [
                        ctypes.c_void_p,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                    ],
                    (source.pointer, 2, 0, CNP_BITORDER_BIG),
                ),
                (
                    "cnp_ahk_unpackbits_v2",
                    [
                        ctypes.c_void_p,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int64,
                        ctypes.c_int,
                        ctypes.c_int,
                    ],
                    (source.pointer, 2, 0, 0, 1, CNP_BITORDER_BIG),
                ),
            )
            for function_name, argtypes, arguments in calls:
                function = self.required_export(runtime, function_name)
                function.argtypes = argtypes
                function.restype = ctypes.c_void_p
                runtime.dll.cnp_clear_error()
                pointer = function(*arguments)
                with self.subTest(function=function_name):
                    with self.assertRaises(CnumpyError) as captured:
                        runtime._owned_result(pointer, function_name)
                    self.assertEqual(-5, captured.exception.status)
                    self.assertEqual(
                        function_name, captured.exception.function
                    )
                    self.assertIn(
                        "axis 2 is out of bounds",
                        captured.exception.message,
                    )
                    self.assertEqual(baseline, runtime.retained_bytes)

    def test_trapz_matches_numpy_for_axes_dx_and_noncontiguous_inputs(
        self,
    ) -> None:
        storage_value = (
            np.arange(24, dtype=np.float64).reshape(2, 3, 4) * 0.75
            - 5.0
        )
        transposed_value = storage_value.transpose(2, 0, 1)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            storage = stack.enter_context(runtime.from_numpy(storage_value))
            transposed = stack.enter_context(
                runtime.transpose(storage, (2, 0, 1))
            )
            self.assertFalse(transposed.c_contiguous)

            for layout, source, value in (
                ("contiguous", storage, storage_value),
                ("transposed", transposed, transposed_value),
            ):
                for axis in range(-value.ndim, value.ndim):
                    with self.subTest(layout=layout, axis=axis):
                        result = stack.enter_context(
                            self.trapz_result(
                                runtime, source, None, 0.375, axis
                            )
                        )
                        assert_array_equivalent(
                            self,
                            result,
                            np.trapz(value, dx=0.375, axis=axis),
                            compare_contiguity=False,
                            rtol=2e-14,
                            atol=2e-14,
                        )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_float64_contiguous_trapz_matches_numpy_pairwise_bits(
        self,
    ) -> None:
        rng = np.random.default_rng(20260804)
        lengths = (1, 2, 7, 8, 9, 127, 128, 129, 257)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for length in lengths:
                exponents = rng.integers(-40, 41, size=(3, length))
                values = rng.standard_normal((3, length)) * np.exp2(
                    exponents
                )
                if length:
                    values[0, 0] = -0.0
                    values[1, -1] = 0.0
                values = np.ascontiguousarray(values, dtype=np.float64)
                for label, candidate, axis in (
                    ("vector", values[0].copy(), 0),
                    ("last_axis", values, -1),
                ):
                    with self.subTest(
                        length=length, layout=label, axis=axis
                    ):
                        source = runtime.from_numpy(candidate)
                        result = self.trapz_result(
                            runtime, source, None, 0.375, axis
                        )
                        source.close()
                        try:
                            actual = np.ascontiguousarray(
                                result.to_numpy()
                            )
                            expected = np.ascontiguousarray(
                                np.trapz(
                                    candidate, dx=0.375, axis=axis
                                )
                            )
                            np.testing.assert_array_equal(
                                actual.view(np.uint64),
                                expected.view(np.uint64),
                                strict=True,
                            )
                        finally:
                            result.close()
                        self.assertEqual(
                            baseline, runtime.retained_bytes
                        )

    def test_trapz_supports_one_dimensional_and_broadcast_x(self) -> None:
        y_value = np.array(
            [
                [[1.0, 2.0], [4.0, 8.0], [9.0, 10.0]],
                [[3.0, 5.0], [7.0, 11.0], [13.0, 17.0]],
            ],
            dtype=np.float64,
        )
        x_1d_value = np.array([0.0, 0.5, 3.0], dtype=np.float64)
        x_broadcast_value = np.array(
            [[[0.0], [1.0], [4.0]]], dtype=np.float64
        )
        x_matching_value = np.broadcast_to(
            x_broadcast_value, y_value.shape
        ).copy()

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            y = stack.enter_context(runtime.from_numpy(y_value))
            for label, x_value in (
                ("one_dimensional", x_1d_value),
                ("broadcast", x_broadcast_value),
                ("matching", x_matching_value),
            ):
                with self.subTest(layout=label):
                    x = stack.enter_context(runtime.from_numpy(x_value))
                    result = stack.enter_context(
                        self.trapz_result(runtime, y, x, 1.0, 1)
                    )
                    assert_array_equivalent(
                        self,
                        result,
                        np.trapz(y_value, x=x_value, axis=1),
                        compare_contiguity=False,
                        rtol=2e-14,
                        atol=2e-14,
                    )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_trapz_broadcasts_lower_rank_x_like_numpy(self) -> None:
        y_value = np.arange(12, dtype=np.float64).reshape(2, 3, 2)
        x_value = np.array(
            [[0.0, 0.5, 2.0], [1.0, 2.5, 5.0]], dtype=np.float64
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            y = stack.enter_context(runtime.from_numpy(y_value))
            x = stack.enter_context(runtime.from_numpy(x_value))
            result = stack.enter_context(
                self.trapz_result(runtime, y, x, 1.0, 1)
            )
            assert_array_equivalent(
                self,
                result,
                np.trapz(y_value, x=x_value, axis=1),
                compare_contiguity=False,
                rtol=2e-14,
                atol=2e-14,
            )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_trapz_higher_rank_x_can_expand_result_shape(self) -> None:
        y_value = np.arange(6, dtype=np.float64).reshape(2, 3)
        x_value = (
            np.arange(24, dtype=np.float64).reshape(4, 2, 3) * 0.25
        )
        expected = np.trapz(y_value, x=x_value, axis=-1)
        self.assertEqual((4, 2), expected.shape)

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            y = stack.enter_context(runtime.from_numpy(y_value))
            x = stack.enter_context(runtime.from_numpy(x_value))
            result = stack.enter_context(
                self.trapz_result(runtime, y, x, 1.0, -1)
            )
            assert_array_equivalent(
                self,
                result,
                expected,
                compare_contiguity=False,
                rtol=2e-14,
                atol=2e-14,
            )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_trapz_preserves_float32_and_handles_empty_singleton_axes(
        self,
    ) -> None:
        values = (
            np.empty((0, 3), dtype=np.float32),
            np.empty((2, 0), dtype=np.float32),
            np.array([[4.0], [9.0]], dtype=np.float32),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for value in values:
                source = stack.enter_context(runtime.from_numpy(value))
                for axis in range(value.ndim):
                    with self.subTest(shape=value.shape, axis=axis):
                        result = stack.enter_context(
                            self.trapz_result(
                                runtime, source, None, 0.5, axis
                            )
                        )
                        assert_array_equivalent(
                            self,
                            result,
                            np.trapz(value, dx=0.5, axis=axis),
                            compare_contiguity=False,
                            rtol=2e-6,
                            atol=2e-7,
                        )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_trapz_matches_numpy_dtype_promotion_and_typed_arithmetic(
        self,
    ) -> None:
        dtype_values = (
            (np.bool_, (True, True, False)),
            (np.int8, (127, 127, -128)),
            (np.uint8, (255, 255, 0)),
            (np.int16, (1, 2, 4)),
            (np.uint16, (1, 2, 4)),
            (np.int32, (1, 2, 4)),
            (np.uint32, (1, 2, 4)),
            (np.int64, (1, 2, 4)),
            (np.uint64, (1, 2, 4)),
            (np.float16, (1, 2, 4)),
            (np.float32, (1, 2, 4)),
            (np.float64, (1, 2, 4)),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for dtype, values in dtype_values:
                y_value = np.asarray(values, dtype=dtype)
                y = stack.enter_context(runtime.from_numpy(y_value))
                result = stack.enter_context(
                    self.trapz_result(runtime, y, None, 0.5, 0)
                )
                with self.subTest(dtype=np.dtype(dtype).name):
                    assert_array_equivalent(
                        self,
                        result,
                        np.trapz(y_value, dx=0.5, axis=0),
                        compare_contiguity=False,
                    )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_trapz_matches_numpy_x_dtype_promotion_and_product_overflow(
        self,
    ) -> None:
        cases = (
            (np.bool_, (True, True), np.bool_, (True, False)),
            (np.int8, (50, 50), np.int8, (0, 2)),
            (np.uint8, (100, 100), np.uint8, (0, 2)),
            (np.int8, (50, 50), np.uint8, (0, 2)),
            (np.float16, (1, 2), np.float16, (0, 2)),
            (np.float16, (1, 2), np.float32, (0, 2)),
            (np.int8, (1, 2), np.float16, (0, 2)),
            (np.int16, (1, 2), np.float16, (0, 2)),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for y_dtype, y_values, x_dtype, x_values in cases:
                y_value = np.asarray(y_values, dtype=y_dtype)
                x_value = np.asarray(x_values, dtype=x_dtype)
                y = stack.enter_context(runtime.from_numpy(y_value))
                x = stack.enter_context(runtime.from_numpy(x_value))
                result = stack.enter_context(
                    self.trapz_result(runtime, y, x, 1.0, 0)
                )
                with self.subTest(
                    y_dtype=y_value.dtype.name,
                    x_dtype=x_value.dtype.name,
                ):
                    assert_array_equivalent(
                        self,
                        result,
                        np.trapz(y_value, x=x_value, axis=0),
                        compare_contiguity=False,
                    )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_trapz_preserves_wide_integer_panels_until_float_conversion(
        self,
    ) -> None:
        uint64_max = np.iinfo(np.uint64).max
        int64_max = np.iinfo(np.int64).max
        cases = (
            (
                "uint64_contiguous_axis_last",
                np.asarray(
                    [[2**53, 3], [0, 2]], dtype=np.uint64
                ),
                np.asarray(
                    [[0, 3], [0, 2**63 + 3]], dtype=np.uint64
                ),
                1,
            ),
            (
                "int64_transposed_axis0_signed",
                np.asarray(
                    [[2**53, 3], [-(2**53), -3]], dtype=np.int64
                ).T,
                np.asarray([[0, 3], [0, 3]], dtype=np.int64).T,
                0,
            ),
            (
                "uint64_pair_and_product_wrap",
                np.asarray(
                    [[uint64_max, 2], [0, 2]], dtype=np.uint64
                ),
                np.asarray(
                    [[0, 3], [0, 2**63 + 3]], dtype=np.uint64
                ),
                -1,
            ),
            (
                "int64_pair_and_product_wrap",
                np.asarray(
                    [[int64_max, 2], [-(2**53), -3]], dtype=np.int64
                ),
                np.asarray([[0, 3], [0, 3]], dtype=np.int64),
                -1,
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for label, y_value, x_value, axis in cases:
                self.assertEqual(
                    label.endswith("axis0_signed"),
                    not y_value.flags.c_contiguous,
                )
                with np.errstate(over="ignore", invalid="ignore"):
                    expected = np.trapz(y_value, x=x_value, axis=axis)
                self.assertEqual(np.dtype(np.float64), expected.dtype)
                y = runtime.from_numpy(y_value)
                x = runtime.from_numpy(x_value)
                result = self.trapz_result(
                    runtime, y, x, 1.0, axis
                )
                y.close()
                x.close()
                with self.subTest(label=label):
                    assert_array_equivalent(
                        self,
                        result,
                        expected,
                        compare_contiguity=False,
                    )
                result.close()
                self.assertEqual(baseline, runtime.retained_bytes)

    def test_trapz_rank_zero_through_four_all_axes(self) -> None:
        shapes = ((3,), (2, 3), (2, 2, 3), (2, 2, 2, 3))
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(
                np.asarray(4.0, dtype=np.float64)
            ) as scalar:
                for axis in (-1, 0):
                    with self.subTest(rank=0, axis=axis):
                        self.assert_native_error(
                            lambda axis=axis: self.trapz_result(
                                runtime, scalar, None, 0.25, axis
                            ),
                            -5,
                            "cnp_trapz",
                            f"axis {axis} is out of bounds",
                        )

            for shape in shapes:
                value = (
                    np.arange(np.prod(shape), dtype=np.float64)
                    .reshape(shape)
                    * 0.25
                )
                with runtime.from_numpy(value) as source:
                    for axis in range(-len(shape), len(shape)):
                        with self.subTest(rank=len(shape), axis=axis):
                            with self.trapz_result(
                                runtime, source, None, 0.25, axis
                            ) as result:
                                assert_array_equivalent(
                                    self,
                                    result,
                                    np.trapz(value, dx=0.25, axis=axis),
                                    compare_contiguity=False,
                                    rtol=2e-14,
                                    atol=2e-14,
                                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_task8_results_survive_source_release_and_retain_zero_bytes(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes

            for function_name, reference in (
                ("cnp_softmax", reference_softmax),
                ("cnp_log_softmax", reference_log_softmax),
            ):
                value = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)
                source = runtime.from_numpy(value)
                result = self.unary_axis_result(
                    runtime, source, function_name, 1
                )
                source.close()
                assert_array_equivalent(
                    self,
                    result,
                    reference(value, 1),
                    compare_contiguity=False,
                    rtol=2e-14,
                    atol=2e-15,
                )
                result.close()
                self.assertEqual(baseline, runtime.retained_bytes)

            y_value = np.asarray([1.0, 3.0, 5.0], dtype=np.float64)
            x_value = np.asarray([0.0, 0.5, 2.0], dtype=np.float64)
            y = runtime.from_numpy(y_value)
            x = runtime.from_numpy(x_value)
            result = self.trapz_result(runtime, y, x, 1.0, 0)
            y.close()
            x.close()
            assert_array_equivalent(
                self,
                result,
                np.trapz(y_value, x=x_value, axis=0),
                compare_contiguity=False,
            )
            result.close()
            self.assertEqual(baseline, runtime.retained_bytes)

            pack_value = np.asarray([1, 0, 2, 0], dtype=np.int16)
            pack_source = runtime.from_numpy(pack_value)
            packed = self.packbits_result(
                runtime, pack_source, None, CNP_BITORDER_LITTLE
            )
            pack_source.close()
            assert_array_equivalent(
                self,
                packed,
                np.packbits(pack_value, bitorder="little"),
                compare_contiguity=False,
            )
            packed.close()
            self.assertEqual(baseline, runtime.retained_bytes)

            byte_value = np.asarray([1, 128], dtype=np.uint8)
            byte_source = runtime.from_numpy(byte_value)
            unpacked = self.unpackbits_result(
                runtime,
                byte_source,
                None,
                12,
                CNP_BITORDER_BIG,
            )
            byte_source.close()
            assert_array_equivalent(
                self,
                unpacked,
                np.unpackbits(byte_value, count=12, bitorder="big"),
                compare_contiguity=False,
            )
            unpacked.close()
            self.assertEqual(baseline, runtime.retained_bytes)

            legacy_source = runtime.from_numpy(pack_value)
            legacy_packed = self.legacy_packbits_result(
                runtime, legacy_source, 0
            )
            legacy_source.close()
            assert_array_equivalent(
                self,
                legacy_packed,
                np.packbits(pack_value, axis=0, bitorder="big"),
                compare_contiguity=False,
            )
            legacy_packed.close()
            self.assertEqual(baseline, runtime.retained_bytes)

            legacy_bytes = runtime.from_numpy(byte_value)
            legacy_unpacked = self.legacy_unpackbits_result(
                runtime, legacy_bytes, 0, 0
            )
            legacy_bytes.close()
            assert_array_equivalent(
                self,
                legacy_unpacked,
                np.unpackbits(byte_value, axis=0, bitorder="big"),
                compare_contiguity=False,
            )
            legacy_unpacked.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_trapz_reports_invalid_axis_x_shape_and_null_y(self) -> None:
        y_value = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        invalid_x_value = np.arange(5, dtype=np.float64)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with (
                runtime.from_numpy(y_value) as y,
                runtime.from_numpy(invalid_x_value) as invalid_x,
            ):
                cases = (
                    (None, 3, -5, "axis 3 is out of bounds"),
                    (invalid_x, 1, -7, "cannot be broadcast"),
                )
                for x, axis, status, message in cases:
                    with self.subTest(axis=axis, x_is_none=x is None):
                        before_error = runtime.retained_bytes
                        self.assert_native_error(
                            lambda x=x, axis=axis: self.trapz_result(
                                runtime, y, x, 1.0, axis
                            ),
                            status,
                            "cnp_trapz",
                            message,
                        )
                        self.assertEqual(
                            before_error, runtime.retained_bytes
                        )

                function = self.required_export(runtime, "cnp_trapz")
                runtime.dll.cnp_clear_error()
                pointer = function(None, None, 1.0, -1)
                with self.assertRaises(CnumpyError) as captured:
                    runtime._owned_result(pointer, "cnp_trapz")
                self.assertEqual("cnp_trapz", captured.exception.function)
                self.assertIn("must not be null", captured.exception.message)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_packbits_matches_numpy_for_axes_bitorders_and_views(self) -> None:
        storage_value = (
            np.arange(60, dtype=np.int64).reshape(3, 4, 5) % 3 == 0
        ).astype(np.int64)
        transposed_value = storage_value.transpose(2, 0, 1)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            storage = stack.enter_context(runtime.from_numpy(storage_value))
            transposed = stack.enter_context(
                runtime.transpose(storage, (2, 0, 1))
            )
            self.assertFalse(transposed.c_contiguous)

            for layout, source, value in (
                ("contiguous", storage, storage_value),
                ("transposed", transposed, transposed_value),
            ):
                for axis in (None, *range(-value.ndim, value.ndim)):
                    for bitorder_name, bitorder in (
                        ("big", CNP_BITORDER_BIG),
                        ("little", CNP_BITORDER_LITTLE),
                    ):
                        with self.subTest(
                            layout=layout,
                            axis=axis,
                            bitorder=bitorder_name,
                        ):
                            result = stack.enter_context(
                                self.packbits_result(
                                    runtime, source, axis, bitorder
                                )
                            )
                            assert_array_equivalent(
                                self,
                                result,
                                np.packbits(
                                    value,
                                    axis=axis,
                                    bitorder=bitorder_name,
                                ),
                                compare_contiguity=False,
                            )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_unpackbits_matches_numpy_counts_padding_axes_and_bitorders(
        self,
    ) -> None:
        storage_value = np.array(
            [[0, 1, 2], [127, 128, 255]], dtype=np.uint8
        )
        transposed_value = storage_value.T
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            storage = stack.enter_context(runtime.from_numpy(storage_value))
            transposed = stack.enter_context(runtime.transpose(storage))
            self.assertFalse(transposed.c_contiguous)

            for layout, source, value in (
                ("contiguous", storage, storage_value),
                ("transposed", transposed, transposed_value),
            ):
                for axis in (None, *range(-value.ndim, value.ndim)):
                    axis_length = value.size if axis is None else value.shape[axis]
                    full_count = axis_length * 8
                    counts = (None, 0, 3, full_count, full_count + 3, -1)
                    for count in counts:
                        for bitorder_name, bitorder in (
                            ("big", CNP_BITORDER_BIG),
                            ("little", CNP_BITORDER_LITTLE),
                        ):
                            with self.subTest(
                                layout=layout,
                                axis=axis,
                                count=count,
                                bitorder=bitorder_name,
                            ):
                                result = stack.enter_context(
                                    self.unpackbits_result(
                                        runtime,
                                        source,
                                        axis,
                                        count,
                                        bitorder,
                                    )
                                )
                                assert_array_equivalent(
                                    self,
                                    result,
                                    np.unpackbits(
                                        value,
                                        axis=axis,
                                        count=count,
                                        bitorder=bitorder_name,
                                    ),
                                    compare_contiguity=False,
                                )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_typed_bitpacking_byte_boundaries_axes_and_lifetimes(
        self,
    ) -> None:
        lengths = (0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for dtype in (np.bool_, np.uint8):
                for length in lengths:
                    raw = (np.arange(length, dtype=np.uint8) * 37 + 5) % 11
                    value = raw.astype(dtype)
                    for bitorder_name, bitorder in (
                        ("big", CNP_BITORDER_BIG),
                        ("little", CNP_BITORDER_LITTLE),
                    ):
                        with self.subTest(
                            operation="pack",
                            dtype=np.dtype(dtype).name,
                            length=length,
                            bitorder=bitorder_name,
                        ):
                            source = runtime.from_numpy(value)
                            result = self.packbits_result(
                                runtime, source, None, bitorder
                            )
                            source.close()
                            try:
                                assert_array_equivalent(
                                    self,
                                    result,
                                    np.packbits(
                                        value, axis=None,
                                        bitorder=bitorder_name,
                                    ),
                                    compare_contiguity=False,
                                )
                            finally:
                                result.close()
                            self.assertEqual(
                                baseline, runtime.retained_bytes
                            )

            matrix = np.ascontiguousarray(
                (np.arange(51, dtype=np.uint8).reshape(3, 17) * 13)
                % 19
            )
            for bitorder_name, bitorder in (
                ("big", CNP_BITORDER_BIG),
                ("little", CNP_BITORDER_LITTLE),
            ):
                with runtime.from_numpy(matrix) as owner:
                    transposed = runtime.transpose(owner)
                    packed = self.packbits_result(
                        runtime, transposed, 0, bitorder
                    )
                    transposed.close()
                    try:
                        assert_array_equivalent(
                            self,
                            packed,
                            np.packbits(
                                matrix.T,
                                axis=0,
                                bitorder=bitorder_name,
                            ),
                            compare_contiguity=False,
                        )
                    finally:
                        packed.close()

            byte_lengths = (0, 1, 2, 7, 8, 9)
            for length in byte_lengths:
                value = (
                    np.arange(length, dtype=np.uint8) * 73 + 17
                ).astype(np.uint8)
                available = length * 8
                counts = {None, 0}
                if available:
                    counts.update(
                        {
                            1,
                            7,
                            8,
                            9,
                            available,
                            available + 3,
                            -1,
                            -7 if available >= 7 else -1,
                        }
                    )
                for count in counts:
                    for bitorder_name, bitorder in (
                        ("big", CNP_BITORDER_BIG),
                        ("little", CNP_BITORDER_LITTLE),
                    ):
                        with self.subTest(
                            operation="unpack",
                            length=length,
                            count=count,
                            bitorder=bitorder_name,
                        ):
                            source = runtime.from_numpy(value)
                            result = self.unpackbits_result(
                                runtime, source, None, count, bitorder
                            )
                            source.close()
                            try:
                                assert_array_equivalent(
                                    self,
                                    result,
                                    np.unpackbits(
                                        value,
                                        axis=None,
                                        count=count,
                                        bitorder=bitorder_name,
                                    ),
                                    compare_contiguity=False,
                                )
                            finally:
                                result.close()
                            self.assertEqual(
                                baseline, runtime.retained_bytes
                            )

            bytes_matrix = np.ascontiguousarray(
                np.arange(51, dtype=np.uint8).reshape(3, 17) * 29
            )
            for bitorder_name, bitorder in (
                ("big", CNP_BITORDER_BIG),
                ("little", CNP_BITORDER_LITTLE),
            ):
                with runtime.from_numpy(bytes_matrix) as owner:
                    transposed = runtime.transpose(owner)
                    unpacked = self.unpackbits_result(
                        runtime, transposed, 0, None, bitorder
                    )
                    transposed.close()
                    try:
                        assert_array_equivalent(
                            self,
                            unpacked,
                            np.unpackbits(
                                bytes_matrix.T,
                                axis=0,
                                bitorder=bitorder_name,
                            ),
                            compare_contiguity=False,
                        )
                    finally:
                        unpacked.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_bit_packing_rank_zero_through_four_all_axes_and_dtypes(
        self,
    ) -> None:
        pack_dtypes = (
            np.bool_,
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
        )
        shapes = ((), (3,), (2, 3), (2, 2, 3), (2, 2, 2, 3))
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for dtype in pack_dtypes:
                for shape in shapes:
                    size = int(np.prod(shape, dtype=np.int64)) or 1
                    value = (np.arange(size) % 3 - 1).reshape(shape)
                    value = value.astype(dtype)
                    axes = (
                        (None, 0, -1)
                        if not shape
                        else (None, *range(-len(shape), len(shape)))
                    )
                    with runtime.from_numpy(value) as source:
                        for axis in axes:
                            for bitorder_name, bitorder in (
                                ("big", CNP_BITORDER_BIG),
                                ("little", CNP_BITORDER_LITTLE),
                            ):
                                with self.subTest(
                                    operation="pack",
                                    dtype=np.dtype(dtype).name,
                                    rank=len(shape),
                                    axis=axis,
                                    bitorder=bitorder_name,
                                ):
                                    with self.packbits_result(
                                        runtime, source, axis, bitorder
                                    ) as result:
                                        assert_array_equivalent(
                                            self,
                                            result,
                                            np.packbits(
                                                value,
                                                axis=axis,
                                                bitorder=bitorder_name,
                                            ),
                                            compare_contiguity=False,
                                        )

            for shape in shapes:
                size = int(np.prod(shape, dtype=np.int64)) or 1
                value = (np.arange(size) * 37 + 1).astype(
                    np.uint8
                ).reshape(shape)
                axes = (
                    (None, 0, -1)
                    if not shape
                    else (None, *range(-len(shape), len(shape)))
                )
                with runtime.from_numpy(value) as source:
                    for axis in axes:
                        axis_length = (
                            value.size
                            if axis is None or not shape
                            else value.shape[axis]
                        )
                        available = axis_length * 8
                        for count in (
                            None,
                            0,
                            3,
                            available,
                            available + 3,
                            -1,
                        ):
                            for bitorder_name, bitorder in (
                                ("big", CNP_BITORDER_BIG),
                                ("little", CNP_BITORDER_LITTLE),
                            ):
                                with self.subTest(
                                    operation="unpack",
                                    rank=len(shape),
                                    axis=axis,
                                    count=count,
                                    bitorder=bitorder_name,
                                ):
                                    with self.unpackbits_result(
                                        runtime,
                                        source,
                                        axis,
                                        count,
                                        bitorder,
                                    ) as result:
                                        assert_array_equivalent(
                                            self,
                                            result,
                                            np.unpackbits(
                                                value,
                                                axis=axis,
                                                count=count,
                                                bitorder=bitorder_name,
                                            ),
                                            compare_contiguity=False,
                                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_bit_packing_empty_dimensions_match_numpy(self) -> None:
        shapes = ((0,), (0, 3), (2, 0, 3), (2, 0, 2, 3))
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for shape in shapes:
                pack_value = np.empty(shape, dtype=np.int16)
                byte_value = np.empty(shape, dtype=np.uint8)
                axes = (None, *range(-len(shape), len(shape)))
                with (
                    runtime.from_numpy(pack_value) as pack_source,
                    runtime.from_numpy(byte_value) as byte_source,
                ):
                    for axis in axes:
                        for bitorder_name, bitorder in (
                            ("big", CNP_BITORDER_BIG),
                            ("little", CNP_BITORDER_LITTLE),
                        ):
                            with self.subTest(
                                operation="pack",
                                shape=shape,
                                axis=axis,
                                bitorder=bitorder_name,
                            ):
                                with self.packbits_result(
                                    runtime,
                                    pack_source,
                                    axis,
                                    bitorder,
                                ) as result:
                                    assert_array_equivalent(
                                        self,
                                        result,
                                        np.packbits(
                                            pack_value,
                                            axis=axis,
                                            bitorder=bitorder_name,
                                        ),
                                        compare_contiguity=False,
                                    )
                            for count in (None, 0):
                                with self.subTest(
                                    operation="unpack",
                                    shape=shape,
                                    axis=axis,
                                    count=count,
                                    bitorder=bitorder_name,
                                ):
                                    with self.unpackbits_result(
                                        runtime,
                                        byte_source,
                                        axis,
                                        count,
                                        bitorder,
                                    ) as result:
                                        assert_array_equivalent(
                                            self,
                                            result,
                                            np.unpackbits(
                                                byte_value,
                                                axis=axis,
                                                count=count,
                                                bitorder=bitorder_name,
                                            ),
                                            compare_contiguity=False,
                                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_bit_packing_huge_zero_shapes_are_bounded_and_exact(
        self,
    ) -> None:
        int64_max = 2**63 - 1
        huge_outer = 2**62
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with self.native_array_with_shape(
                runtime, (0, int64_max), CNP_UBYTE
            ) as huge_axis:
                with self.packbits_result(
                    runtime, huge_axis, 1, CNP_BITORDER_BIG
                ) as result:
                    self.assertEqual(
                        (0, int64_max // 8 + int(int64_max % 8 != 0)),
                        result.shape,
                    )
                    self.assertEqual(0, result.size)
                    self.assertEqual(np.dtype(np.uint8), result.numpy_dtype)

            with self.native_array_with_shape(
                runtime, (huge_outer, 0, 4), CNP_UBYTE
            ) as huge_non_axis:
                with self.packbits_result(
                    runtime, huge_non_axis, 1, CNP_BITORDER_BIG
                ) as packed:
                    self.assertEqual((huge_outer, 0, 4), packed.shape)
                    self.assertEqual(0, packed.size)
                for count in (None, 0):
                    with self.subTest(operation="unpack", count=count):
                        with self.unpackbits_result(
                            runtime,
                            huge_non_axis,
                            1,
                            count,
                            CNP_BITORDER_BIG,
                        ) as unpacked:
                            self.assertEqual(
                                (huge_outer, 0, 4), unpacked.shape
                            )
                            self.assertEqual(0, unpacked.size)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_unpackbits_rejects_huge_multidimensional_result_before_allocation(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with self.native_array_with_shape(
                runtime, (2, 1), CNP_UBYTE
            ) as source:
                self.assert_native_error(
                    lambda: self.unpackbits_result(
                        runtime,
                        source,
                        1,
                        2**63 - 1,
                        CNP_BITORDER_BIG,
                    ),
                    -4,
                    "cnp_unpackbits_v2",
                    "result size exceeds int64 range",
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_softmax_huge_empty_reduction_axis_errors_before_data_access(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with self.native_array_with_shape(
                runtime, (2**62, 0, 4), CNP_BOOL
            ) as source:
                for function_name in (
                    "cnp_softmax",
                    "cnp_log_softmax",
                ):
                    with self.subTest(function=function_name):
                        self.assert_native_error(
                            lambda function_name=function_name: (
                                self.unary_axis_result(
                                    runtime, source, function_name, 1
                                )
                            ),
                            -1,
                            function_name,
                            "zero-size reduction has no identity",
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_unpackbits_empty_positive_count_zero_initializes_padding(
        self,
    ) -> None:
        value = np.empty((2, 0, 3), dtype=np.uint8)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(value) as source:
                for axis in (None, 1, -2):
                    expected_shape = (
                        (3,) if axis is None else (2, 3, 3)
                    )
                    for bitorder in (
                        CNP_BITORDER_BIG,
                        CNP_BITORDER_LITTLE,
                    ):
                        with self.subTest(axis=axis, bitorder=bitorder):
                            with self.unpackbits_result(
                                runtime,
                                source,
                                axis,
                                3,
                                bitorder,
                            ) as result:
                                self.assertEqual(expected_shape, result.shape)
                                self.assertEqual(
                                    np.dtype(np.uint8), result.numpy_dtype
                                )
                                np.testing.assert_array_equal(
                                    np.zeros(expected_shape, dtype=np.uint8),
                                    result.to_numpy(),
                                    strict=True,
                                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_bit_packing_preserves_big_order_and_none_sentinel(
        self,
    ) -> None:
        integer_value = np.array(
            [[1, 0, 1], [0, 1, 0]], dtype=np.int64
        )
        byte_value = np.array([[1, 128], [3, 255]], dtype=np.uint8)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            integer_source = stack.enter_context(
                runtime.from_numpy(integer_value)
            )
            byte_source = stack.enter_context(runtime.from_numpy(byte_value))

            for axis in (-1, 0, 1):
                numpy_axis = None if axis == -1 else axis
                packed = stack.enter_context(
                    self.legacy_packbits_result(
                        runtime, integer_source, axis
                    )
                )
                unpacked = stack.enter_context(
                    self.legacy_unpackbits_result(
                        runtime, byte_source, axis, 0
                    )
                )
                assert_array_equivalent(
                    self,
                    packed,
                    np.packbits(
                        integer_value, axis=numpy_axis, bitorder="big"
                    ),
                    compare_contiguity=False,
                )
                assert_array_equivalent(
                    self,
                    unpacked,
                    np.unpackbits(
                        byte_value, axis=numpy_axis, bitorder="big"
                    ),
                    compare_contiguity=False,
                )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_bit_packing_rank_zero_through_four_and_errors(
        self,
    ) -> None:
        shapes = ((), (3,), (2, 3), (2, 2, 3), (2, 2, 2, 3))
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for shape in shapes:
                size = int(np.prod(shape, dtype=np.int64)) or 1
                pack_value = (np.arange(size) % 2).astype(
                    np.int16
                ).reshape(shape)
                byte_value = (np.arange(size) * 31 + 1).astype(
                    np.uint8
                ).reshape(shape)
                if not shape:
                    axes = (-1, 0)
                else:
                    axes = (
                        -1,
                        *range(len(shape)),
                        *range(-len(shape), -1),
                    )
                with (
                    runtime.from_numpy(pack_value) as pack_source,
                    runtime.from_numpy(byte_value) as byte_source,
                ):
                    for axis in axes:
                        numpy_axis = None if axis == -1 else axis
                        with self.subTest(
                            operation="legacy_pack",
                            rank=len(shape),
                            axis=axis,
                        ):
                            with self.legacy_packbits_result(
                                runtime, pack_source, axis
                            ) as result:
                                assert_array_equivalent(
                                    self,
                                    result,
                                    np.packbits(
                                        pack_value,
                                        axis=numpy_axis,
                                        bitorder="big",
                                    ),
                                    compare_contiguity=False,
                                )
                        for count in (0, -1, 3):
                            numpy_count = None if count <= 0 else count
                            with self.subTest(
                                operation="legacy_unpack",
                                rank=len(shape),
                                axis=axis,
                                count=count,
                            ):
                                with self.legacy_unpackbits_result(
                                    runtime,
                                    byte_source,
                                    axis,
                                    count,
                                ) as result:
                                    assert_array_equivalent(
                                        self,
                                        result,
                                        np.unpackbits(
                                            byte_value,
                                            axis=numpy_axis,
                                            count=numpy_count,
                                            bitorder="big",
                                        ),
                                        compare_contiguity=False,
                                    )

                    invalid_axes = (
                        (1, -2) if not shape else (len(shape), -len(shape) - 1)
                    )
                    for function_name, call in (
                        (
                            "cnp_packbits",
                            lambda axis: self.legacy_packbits_result(
                                runtime, pack_source, axis
                            ),
                        ),
                        (
                            "cnp_unpackbits",
                            lambda axis: self.legacy_unpackbits_result(
                                runtime, byte_source, axis, 0
                            ),
                        ),
                    ):
                        for axis in invalid_axes:
                            with self.subTest(
                                function=function_name,
                                rank=len(shape),
                                axis=axis,
                            ):
                                self.assert_native_error(
                                    lambda call=call, axis=axis: call(axis),
                                    -5,
                                    function_name,
                                    f"axis {axis} is out of bounds",
                                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_bit_packing_reports_dtype_axis_count_and_bitorder_errors(
        self,
    ) -> None:
        integer_value = np.arange(6, dtype=np.int64).reshape(2, 3)
        float_value = integer_value.astype(np.float64)
        byte_value = integer_value.astype(np.uint8)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with (
                runtime.from_numpy(integer_value) as integer_source,
                runtime.from_numpy(float_value) as float_source,
                runtime.from_numpy(byte_value) as byte_source,
            ):
                cases = (
                    (
                        lambda: self.packbits_result(
                            runtime, float_source, None, CNP_BITORDER_BIG
                        ),
                        -3,
                        "cnp_packbits_v2",
                        "integer or boolean",
                    ),
                    (
                        lambda: self.packbits_result(
                            runtime, integer_source, 2, CNP_BITORDER_BIG
                        ),
                        -5,
                        "cnp_packbits_v2",
                        "axis 2 is out of bounds",
                    ),
                    (
                        lambda: self.packbits_result(
                            runtime, integer_source, None, 7
                        ),
                        -1,
                        "cnp_packbits_v2",
                        "bitorder",
                    ),
                    (
                        lambda: self.unpackbits_result(
                            runtime, integer_source, None, None,
                            CNP_BITORDER_BIG,
                        ),
                        -3,
                        "cnp_unpackbits_v2",
                        "unsigned byte",
                    ),
                    (
                        lambda: self.unpackbits_result(
                            runtime, byte_source, 2, None,
                            CNP_BITORDER_BIG,
                        ),
                        -5,
                        "cnp_unpackbits_v2",
                        "axis 2 is out of bounds",
                    ),
                    (
                        lambda: self.unpackbits_result(
                            runtime, byte_source, 1, -25,
                            CNP_BITORDER_BIG,
                        ),
                        -1,
                        "cnp_unpackbits_v2",
                        "larger than number of elements",
                    ),
                    (
                        lambda: self.unpackbits_result(
                            runtime, byte_source, None, None, 7
                        ),
                        -1,
                        "cnp_unpackbits_v2",
                        "bitorder",
                    ),
                )
                for call, status, function_name, message in cases:
                    with self.subTest(
                        function=function_name, message=message
                    ):
                        before_error = runtime.retained_bytes
                        with self.assertRaises(CnumpyError) as captured:
                            call()
                        self.assertEqual(status, captured.exception.status)
                        self.assertEqual(
                            function_name, captured.exception.function
                        )
                        self.assertIn(message, captured.exception.message)
                        self.assertEqual(
                            before_error, runtime.retained_bytes
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_task8_core_exports_report_exact_invalid_axes_all_ranks(
        self,
    ) -> None:
        shapes = ((), (3,), (2, 3), (2, 2, 3), (2, 2, 2, 3))
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for shape in shapes:
                size = int(np.prod(shape, dtype=np.int64)) or 1
                real_value = np.arange(size, dtype=np.float64).reshape(shape)
                byte_value = np.arange(size, dtype=np.uint8).reshape(shape)
                invalid_axes = (
                    (1, -2) if not shape else (len(shape), -len(shape) - 1)
                )
                with (
                    runtime.from_numpy(real_value) as real_source,
                    runtime.from_numpy(byte_value) as byte_source,
                ):
                    calls = (
                        (
                            "cnp_softmax",
                            lambda axis: self.unary_axis_result(
                                runtime,
                                real_source,
                                "cnp_softmax",
                                axis,
                            ),
                        ),
                        (
                            "cnp_log_softmax",
                            lambda axis: self.unary_axis_result(
                                runtime,
                                real_source,
                                "cnp_log_softmax",
                                axis,
                            ),
                        ),
                        (
                            "cnp_trapz",
                            lambda axis: self.trapz_result(
                                runtime, real_source, None, 1.0, axis
                            ),
                        ),
                        (
                            "cnp_packbits_v2",
                            lambda axis: self.packbits_result(
                                runtime,
                                byte_source,
                                axis,
                                CNP_BITORDER_BIG,
                            ),
                        ),
                        (
                            "cnp_unpackbits_v2",
                            lambda axis: self.unpackbits_result(
                                runtime,
                                byte_source,
                                axis,
                                None,
                                CNP_BITORDER_BIG,
                            ),
                        ),
                    )
                    for function_name, call in calls:
                        axes = (
                            (-1, 0)
                            if function_name == "cnp_trapz" and not shape
                            else invalid_axes
                        )
                        for axis in axes:
                            with self.subTest(
                                function=function_name,
                                rank=len(shape),
                                axis=axis,
                            ):
                                before_error = runtime.retained_bytes
                                self.assert_native_error(
                                    lambda call=call, axis=axis: call(axis),
                                    -5,
                                    function_name,
                                    f"axis {axis} is out of bounds",
                                )
                                self.assertEqual(
                                    before_error, runtime.retained_bytes
                                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_task8_core_exports_report_exact_dtype_and_null_errors(
        self,
    ) -> None:
        complex_value = np.asarray([1 + 2j, 3 + 4j], dtype=np.complex128)
        real_value = np.asarray([1.0, 2.0], dtype=np.float64)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with (
                runtime.from_numpy(complex_value) as complex_source,
                runtime.from_numpy(real_value) as real_source,
            ):
                cases = (
                    (
                        lambda: self.unary_axis_result(
                            runtime, complex_source, "cnp_softmax", 0
                        ),
                        "cnp_softmax",
                        "real numeric dtype",
                    ),
                    (
                        lambda: self.unary_axis_result(
                            runtime,
                            complex_source,
                            "cnp_log_softmax",
                            0,
                        ),
                        "cnp_log_softmax",
                        "real numeric dtype",
                    ),
                    (
                        lambda: self.trapz_result(
                            runtime, complex_source, None, 1.0, 0
                        ),
                        "cnp_trapz",
                        "y must have a real numeric dtype",
                    ),
                    (
                        lambda: self.trapz_result(
                            runtime, real_source, complex_source, 1.0, 0
                        ),
                        "cnp_trapz",
                        "x must have a real numeric dtype",
                    ),
                )
                for call, function_name, message in cases:
                    with self.subTest(function=function_name, message=message):
                        self.assert_native_error(
                            call, -3, function_name, message
                        )

            null_calls = (
                (
                    "cnp_packbits",
                    [ctypes.c_void_p, ctypes.c_int],
                    (None, 0),
                ),
                (
                    "cnp_packbits_v2",
                    [
                        ctypes.c_void_p,
                        ctypes.c_int,
                        ctypes.c_bool,
                        ctypes.c_int,
                    ],
                    (None, 0, False, CNP_BITORDER_BIG),
                ),
                (
                    "cnp_unpackbits",
                    [ctypes.c_void_p, ctypes.c_int, ctypes.c_int64],
                    (None, 0, 0),
                ),
                (
                    "cnp_unpackbits_v2",
                    [
                        ctypes.c_void_p,
                        ctypes.c_int,
                        ctypes.c_bool,
                        ctypes.c_int64,
                        ctypes.c_bool,
                        ctypes.c_int,
                    ],
                    (None, 0, False, 0, True, CNP_BITORDER_BIG),
                ),
            )
            for function_name, argtypes, arguments in null_calls:
                function = self.required_export(runtime, function_name)
                function.argtypes = argtypes
                function.restype = ctypes.c_void_p
                runtime.dll.cnp_clear_error()
                pointer = function(*arguments)
                with self.subTest(function=function_name, error="null"):
                    with self.assertRaises(CnumpyError) as captured:
                        runtime._owned_result(pointer, function_name)
                    self.assertEqual(-1, captured.exception.status)
                    self.assertEqual(
                        function_name, captured.exception.function
                    )
                    self.assertIn(
                        "must not be null", captured.exception.message
                    )
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
