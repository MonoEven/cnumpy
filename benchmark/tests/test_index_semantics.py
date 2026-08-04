from __future__ import annotations

from contextlib import ExitStack
import ctypes
from pathlib import Path
import unittest
import warnings

import numpy as np

from compat.cnumpy_ctypes import CnumpyError, CnumpyRuntime
from compat.oracle import assert_array_equivalent


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"
INDEX_V2_EXPORTS = (
    "cnp_take_v2",
    "cnp_take_along_axis_v2",
    "cnp_compress_v2",
    "cnp_delete_v2",
    "cnp_insert_v2",
)


class _CnpSlice(ctypes.Structure):
    _fields_ = (
        ("start", ctypes.c_int64),
        ("stop", ctypes.c_int64),
        ("step", ctypes.c_int64),
        ("has_start", ctypes.c_bool),
        ("has_stop", ctypes.c_bool),
        ("has_step", ctypes.c_bool),
    )


class IndexMutationSemanticsTests(unittest.TestCase):
    def require_v2_exports(self, runtime: CnumpyRuntime) -> None:
        missing = [
            symbol for symbol in INDEX_V2_EXPORTS
            if not hasattr(runtime.dll, symbol)
        ]
        self.assertEqual([], missing, f"missing indexing v2 exports: {missing}")

    def assert_index_error(
        self, runtime: CnumpyRuntime, status: int, message: str, operation
    ) -> None:
        before = runtime.retained_bytes
        try:
            unexpected = operation()
        except CnumpyError as raised:
            self.assertEqual(status, raised.status)
            self.assertIn(message.lower(), raised.message.lower())
        else:
            if hasattr(unexpected, "close"):
                unexpected.close()
            self.fail(f"CnumpyError({status}) not raised")
        self.assertEqual(before, runtime.retained_bytes)

    def assert_operand_unchanged(self, actual, expected: np.ndarray) -> None:
        self.assertEqual(tuple(expected.shape), actual.shape)
        self.assertEqual(expected.dtype, actual.numpy_dtype)
        np.testing.assert_array_equal(actual.to_numpy(), expected)

    def slice_every_other(self, runtime: CnumpyRuntime, source):
        function = runtime.dll.cnp_array_slice
        function.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(_CnpSlice),
        ]
        function.restype = ctypes.c_void_p
        slices = (_CnpSlice * 1)(
            _CnpSlice(0, 0, 2, False, False, True)
        )
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(
            function(source.pointer, 1, slices), "cnp_array_slice"
        )

    def assert_result_survives_operand_release(
        self, runtime: CnumpyRuntime, result, operands, expected: np.ndarray,
        baseline: int,
    ) -> None:
        for operand in operands:
            operand.close()
        try:
            assert_array_equivalent(
                self, result, expected, compare_contiguity=False
            )
        finally:
            result.close()
        self.assertEqual(baseline, runtime.retained_bytes)

    def legacy_array_result(
        self,
        runtime: CnumpyRuntime,
        function_name: str,
        source,
        argument,
        axis: int,
    ):
        function = getattr(runtime.dll, function_name)
        if function_name == "cnp_compress":
            function.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
            ]
            native_arguments = (argument.pointer, source.pointer, axis)
        else:
            function.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
            ]
            native_arguments = (source.pointer, argument.pointer, axis)
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        pointer = function(*native_arguments)
        return runtime._owned_result(pointer, function_name)

    def legacy_insert_result(
        self, runtime: CnumpyRuntime, source, obj: int, values, axis: int
    ):
        function = runtime.dll.cnp_insert
        function.argtypes = [
            ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int,
        ]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        pointer = function(source.pointer, obj, values.pointer, axis)
        return runtime._owned_result(pointer, "cnp_insert")

    def test_take_preserves_multidimensional_indices_and_negative_axis(self) -> None:
        source_value = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
        index_values = (
            (np.array([[2, 0], [1, 1]], dtype=np.int64), 1),
            (np.array([[3, -1], [0, 2]], dtype=np.int64), -1),
            (np.array([[5, -1], [0, 3]], dtype=np.int64), None),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            self.require_v2_exports(runtime)
            source = stack.enter_context(runtime.from_numpy(source_value))
            for indices_value, axis in index_values:
                with self.subTest(axis=axis):
                    indices = stack.enter_context(
                        runtime.from_numpy(indices_value)
                    )
                    result = stack.enter_context(
                        runtime.take(source, indices, axis)
                    )
                    assert_array_equivalent(
                        self, result,
                        np.take(source_value, indices_value, axis=axis),
                    )

    def test_rank_three_every_positive_and_negative_axis_matches_numpy(self) -> None:
        source_value = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            for axis in (0, 1, 2, -3, -2, -1):
                normalized = axis % source_value.ndim
                axis_size = source_value.shape[normalized]
                with self.subTest(axis=axis, operation="take"):
                    source = stack.enter_context(runtime.from_numpy(source_value))
                    indices_value = np.array(
                        [[-1, 0, -1]], dtype=np.int64
                    )
                    indices = stack.enter_context(
                        runtime.from_numpy(indices_value)
                    )
                    result = stack.enter_context(runtime.take(source, indices, axis))
                    assert_array_equivalent(
                        self, result,
                        np.take(source_value, indices_value, axis=axis),
                    )
                    self.assert_operand_unchanged(source, source_value)
                    self.assert_operand_unchanged(indices, indices_value)

                with self.subTest(axis=axis, operation="take_along_axis"):
                    source = stack.enter_context(runtime.from_numpy(source_value))
                    indices_shape = list(source_value.shape)
                    indices_shape[normalized] = 3
                    pattern = np.array([-1, 0, -1], dtype=np.int64)
                    reshape = [1, 1, 1]
                    reshape[normalized] = 3
                    indices_value = np.broadcast_to(
                        pattern.reshape(reshape), indices_shape
                    ).copy()
                    indices = stack.enter_context(
                        runtime.from_numpy(indices_value)
                    )
                    result = stack.enter_context(
                        runtime.take_along_axis(source, indices, axis)
                    )
                    assert_array_equivalent(
                        self, result,
                        np.take_along_axis(source_value, indices_value, axis),
                    )
                    self.assert_operand_unchanged(source, source_value)
                    self.assert_operand_unchanged(indices, indices_value)

                with self.subTest(axis=axis, operation="compress"):
                    source = stack.enter_context(runtime.from_numpy(source_value))
                    condition_value = (
                        np.arange(axis_size, dtype=np.int16) % 2 == 0
                    ).astype(np.int16)
                    condition = stack.enter_context(
                        runtime.from_numpy(condition_value)
                    )
                    result = stack.enter_context(
                        runtime.compress(condition, source, axis)
                    )
                    assert_array_equivalent(
                        self, result,
                        np.compress(condition_value, source_value, axis=axis),
                    )
                    self.assert_operand_unchanged(source, source_value)
                    self.assert_operand_unchanged(condition, condition_value)

                with self.subTest(axis=axis, operation="delete"):
                    source = stack.enter_context(runtime.from_numpy(source_value))
                    obj_value = np.array([-1, 0, -1], dtype=np.int64)
                    obj = stack.enter_context(runtime.from_numpy(obj_value))
                    result = stack.enter_context(runtime.delete(source, obj, axis))
                    assert_array_equivalent(
                        self, result,
                        np.delete(source_value, obj_value, axis=axis),
                        compare_contiguity=False,
                    )
                    self.assert_operand_unchanged(source, source_value)
                    self.assert_operand_unchanged(obj, obj_value)

                with self.subTest(axis=axis, operation="insert"):
                    source = stack.enter_context(runtime.from_numpy(source_value))
                    obj_value = np.array(1, dtype=np.int64)
                    values_value = np.array(99, dtype=np.int32)
                    obj = stack.enter_context(runtime.from_numpy(obj_value))
                    values = stack.enter_context(runtime.from_numpy(values_value))
                    result = stack.enter_context(
                        runtime.insert(source, obj, values, axis)
                    )
                    assert_array_equivalent(
                        self, result,
                        np.insert(source_value, obj_value, values_value, axis=axis),
                    )
                    self.assert_operand_unchanged(source, source_value)
                    self.assert_operand_unchanged(obj, obj_value)
                    self.assert_operand_unchanged(values, values_value)

    def test_take_along_axis_broadcasts_non_axis_dimensions(self) -> None:
        source_value = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
        cases = (
            (np.array([[[3, 0]], [[1, 2]]], dtype=np.int64), -1),
            (np.array([5, 0, -1], dtype=np.int64), None),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            self.require_v2_exports(runtime)
            source = stack.enter_context(runtime.from_numpy(source_value))
            for indices_value, axis in cases:
                with self.subTest(axis=axis):
                    indices = stack.enter_context(
                        runtime.from_numpy(indices_value)
                    )
                    result = stack.enter_context(
                        runtime.take_along_axis(source, indices, axis)
                    )
                    assert_array_equivalent(
                        self, result,
                        np.take_along_axis(source_value, indices_value, axis),
                    )

    def test_compress_honors_short_conditions_axis_and_flattening(self) -> None:
        source_value = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        cases = (
            (np.array([True, False]), 1),
            (np.array([True, False, True, False]), 1),
            (np.array([True, False, True, False, True]), None),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            self.require_v2_exports(runtime)
            source = stack.enter_context(runtime.from_numpy(source_value))
            for condition_value, axis in cases:
                with self.subTest(axis=axis):
                    condition = stack.enter_context(
                        runtime.from_numpy(condition_value)
                    )
                    result = stack.enter_context(
                        runtime.compress(condition, source, axis)
                    )
                    assert_array_equivalent(
                        self, result,
                        np.compress(condition_value, source_value, axis=axis),
                    )

    def test_delete_preserves_axis_and_deduplicates_repeated_indices(self) -> None:
        source_value = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
        cases = (
            (np.array([2, 0, 2], dtype=np.int64), -1),
            (np.array([5, 1, 5], dtype=np.int64), None),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            self.require_v2_exports(runtime)
            source = stack.enter_context(runtime.from_numpy(source_value))
            for obj_value, axis in cases:
                with self.subTest(axis=axis):
                    obj = stack.enter_context(runtime.from_numpy(obj_value))
                    result = stack.enter_context(runtime.delete(source, obj, axis))
                    assert_array_equivalent(
                        self, result,
                        np.delete(source_value, obj_value, axis=axis),
                        compare_contiguity=False,
                    )

    def test_insert_scalar_index_broadcasts_values_without_mutating_source(self) -> None:
        source_value = np.arange(6, dtype=np.int32).reshape(2, 3)
        cases = (
            (1, np.array([9, 8], dtype=np.int32), 1),
            (1, np.array([[9], [8]], dtype=np.int32), 1),
            (-1, np.array([8, 9], dtype=np.int32), None),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            self.require_v2_exports(runtime)
            source = stack.enter_context(runtime.from_numpy(source_value))
            for obj, values_value, axis in cases:
                with self.subTest(obj=obj, axis=axis, shape=values_value.shape):
                    values = stack.enter_context(runtime.from_numpy(values_value))
                    result = stack.enter_context(
                        runtime.insert(source, obj, values, axis)
                    )
                    assert_array_equivalent(
                        self, result,
                        np.insert(source_value, obj, values_value, axis=axis),
                    )
                    np.testing.assert_array_equal(source.to_numpy(), source_value)

    def test_noncontiguous_sources_and_index_errors_are_explicit(self) -> None:
        base_value = np.arange(12, dtype=np.int64).reshape(3, 4)
        source_value = base_value.T
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            self.require_v2_exports(runtime)
            base = stack.enter_context(runtime.from_numpy(base_value))
            source = stack.enter_context(runtime.transpose(base, (1, 0)))
            indices_value = np.array([2, 0, 2], dtype=np.int64)
            indices = stack.enter_context(runtime.from_numpy(indices_value))
            result = stack.enter_context(runtime.take(source, indices, axis=1))
            assert_array_equivalent(
                self, result,
                np.take(source_value, indices_value, axis=1),
            )

            invalid = stack.enter_context(
                runtime.from_numpy(np.array([3], dtype=np.int64))
            )
            before = runtime.retained_bytes
            with self.assertRaises(CnumpyError) as raised:
                runtime.take(source, invalid, axis=1)
            self.assertEqual(-6, raised.exception.status)
            self.assertIn("out of bounds", raised.exception.message.lower())
            self.assertEqual(before, runtime.retained_bytes)

    def test_noncontiguous_sources_and_operands_match_without_mutation(self) -> None:
        source_base_value = np.arange(24, dtype=np.int64).reshape(3, 2, 4)
        source_value = source_base_value.transpose(1, 0, 2)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes

            with ExitStack() as stack:
                source_base = stack.enter_context(
                    runtime.from_numpy(source_base_value)
                )
                source = stack.enter_context(
                    runtime.transpose(source_base, (1, 0, 2))
                )
                indices_base_value = np.resize(
                    np.array([-1, 0, 1], dtype=np.int64), (2, 2, 4)
                )
                indices_value = indices_base_value.transpose(1, 0, 2)
                indices_base = stack.enter_context(
                    runtime.from_numpy(indices_base_value)
                )
                indices = stack.enter_context(
                    runtime.transpose(indices_base, (1, 0, 2))
                )
                result = stack.enter_context(
                    runtime.take_along_axis(source, indices, axis=1)
                )
                assert_array_equivalent(
                    self, result,
                    np.take_along_axis(source_value, indices_value, axis=1),
                    compare_contiguity=False,
                )
                self.assert_operand_unchanged(source, source_value)
                self.assert_operand_unchanged(indices, indices_value)
                self.assert_operand_unchanged(source_base, source_base_value)
                self.assert_operand_unchanged(indices_base, indices_base_value)
            self.assertEqual(baseline, runtime.retained_bytes)

            with ExitStack() as stack:
                source_base = stack.enter_context(
                    runtime.from_numpy(source_base_value)
                )
                source = stack.enter_context(
                    runtime.transpose(source_base, (1, 0, 2))
                )
                condition_base_value = np.array(
                    [1, 91, 0, 92, 1, 93], dtype=np.int16
                )
                condition_value = condition_base_value[::2]
                condition_base = stack.enter_context(
                    runtime.from_numpy(condition_base_value)
                )
                condition = stack.enter_context(
                    self.slice_every_other(runtime, condition_base)
                )
                result = stack.enter_context(
                    runtime.compress(condition, source, axis=1)
                )
                assert_array_equivalent(
                    self, result,
                    np.compress(condition_value, source_value, axis=1),
                    compare_contiguity=False,
                )
                self.assert_operand_unchanged(source, source_value)
                self.assert_operand_unchanged(condition, condition_value)
                self.assert_operand_unchanged(source_base, source_base_value)
                self.assert_operand_unchanged(
                    condition_base, condition_base_value
                )
            self.assertEqual(baseline, runtime.retained_bytes)

            with ExitStack() as stack:
                source_base = stack.enter_context(
                    runtime.from_numpy(source_base_value)
                )
                source = stack.enter_context(
                    runtime.transpose(source_base, (1, 0, 2))
                )
                obj_base_value = np.array([2, 91, 0, 92], dtype=np.int64)
                obj_value = obj_base_value[::2]
                obj_base = stack.enter_context(runtime.from_numpy(obj_base_value))
                obj = stack.enter_context(
                    self.slice_every_other(runtime, obj_base)
                )
                values_base_value = (
                    np.arange(16, dtype=np.int32).reshape(2, 2, 4) + 100
                )
                values_value = values_base_value.transpose(1, 0, 2)
                values_base = stack.enter_context(
                    runtime.from_numpy(values_base_value)
                )
                values = stack.enter_context(
                    runtime.transpose(values_base, (1, 0, 2))
                )
                result = stack.enter_context(
                    runtime.insert(source, obj, values, axis=1)
                )
                assert_array_equivalent(
                    self, result,
                    np.insert(source_value, obj_value, values_value, axis=1),
                    compare_contiguity=False,
                )
                self.assert_operand_unchanged(source, source_value)
                self.assert_operand_unchanged(obj, obj_value)
                self.assert_operand_unchanged(values, values_value)
                self.assert_operand_unchanged(source_base, source_base_value)
                self.assert_operand_unchanged(obj_base, obj_base_value)
                self.assert_operand_unchanged(values_base, values_base_value)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_allocating_results_and_failures_have_independent_lifetimes(self) -> None:
        source_value = np.arange(6, dtype=np.int64).reshape(2, 3)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for repetition in range(16):
                with self.subTest(path="success", operation="take", repetition=repetition):
                    source = runtime.from_numpy(source_value)
                    indices_value = np.array([2, 0, 2], dtype=np.int64)
                    indices = runtime.from_numpy(indices_value)
                    result = runtime.take(source, indices, axis=1)
                    self.assert_result_survives_operand_release(
                        runtime, result, (indices, source),
                        np.take(source_value, indices_value, axis=1), baseline,
                    )

                with self.subTest(
                    path="success", operation="take_along_axis",
                    repetition=repetition,
                ):
                    source = runtime.from_numpy(source_value)
                    indices_value = np.array([[2, 0], [1, 1]], dtype=np.int64)
                    indices = runtime.from_numpy(indices_value)
                    result = runtime.take_along_axis(source, indices, axis=1)
                    self.assert_result_survives_operand_release(
                        runtime, result, (indices, source),
                        np.take_along_axis(source_value, indices_value, axis=1),
                        baseline,
                    )

                with self.subTest(
                    path="success", operation="compress", repetition=repetition,
                ):
                    source = runtime.from_numpy(source_value)
                    condition_value = np.array([1, 0, 1], dtype=np.int8)
                    condition = runtime.from_numpy(condition_value)
                    result = runtime.compress(condition, source, axis=1)
                    self.assert_result_survives_operand_release(
                        runtime, result, (condition, source),
                        np.compress(condition_value, source_value, axis=1),
                        baseline,
                    )

                with self.subTest(
                    path="success", operation="delete", repetition=repetition,
                ):
                    source = runtime.from_numpy(source_value)
                    obj_value = np.array([2, 0, 2], dtype=np.int64)
                    obj = runtime.from_numpy(obj_value)
                    result = runtime.delete(source, obj, axis=1)
                    self.assert_result_survives_operand_release(
                        runtime, result, (obj, source),
                        np.delete(source_value, obj_value, axis=1), baseline,
                    )

                with self.subTest(
                    path="success", operation="insert", repetition=repetition,
                ):
                    source = runtime.from_numpy(source_value)
                    obj_value = np.array([1], dtype=np.int64)
                    values_value = np.array([[9], [8]], dtype=np.int64)
                    obj = runtime.from_numpy(obj_value)
                    values = runtime.from_numpy(values_value)
                    result = runtime.insert(source, obj, values, axis=1)
                    self.assert_result_survives_operand_release(
                        runtime, result, (values, obj, source),
                        np.insert(source_value, obj_value, values_value, axis=1),
                        baseline,
                    )

            for repetition in range(16):
                failure_cases = (
                    (
                        "take", np.array([3], dtype=np.int64), None,
                        lambda source, argument, _values: runtime.take(
                            source, argument, axis=1
                        ),
                    ),
                    (
                        "take_along_axis",
                        np.array([[3], [3]], dtype=np.int64), None,
                        lambda source, argument, _values:
                            runtime.take_along_axis(source, argument, axis=1),
                    ),
                    (
                        "compress", np.array([0, 0, 0, 1], dtype=np.int8), None,
                        lambda source, argument, _values: runtime.compress(
                            argument, source, axis=1
                        ),
                    ),
                    (
                        "delete", np.array([3], dtype=np.int64), None,
                        lambda source, argument, _values: runtime.delete(
                            source, argument, axis=1
                        ),
                    ),
                    (
                        "insert", np.array([4], dtype=np.int64),
                        np.array([9], dtype=np.int64),
                        lambda source, argument, values: runtime.insert(
                            source, argument, values, axis=1
                        ),
                    ),
                )
                for operation, argument_value, values_value, invoke in failure_cases:
                    with self.subTest(
                        path="failure", operation=operation,
                        repetition=repetition,
                    ):
                        with ExitStack() as stack:
                            source = stack.enter_context(
                                runtime.from_numpy(source_value)
                            )
                            argument = stack.enter_context(
                                runtime.from_numpy(argument_value)
                            )
                            values = (
                                stack.enter_context(runtime.from_numpy(values_value))
                                if values_value is not None else None
                            )
                            self.assert_index_error(
                                runtime, -6, "out of bounds",
                                lambda: invoke(source, argument, values),
                            )
                        self.assertEqual(baseline, runtime.retained_bytes)

    def test_take_dtype_rules_match_numpy_125(self) -> None:
        source_value = np.arange(6, dtype=np.int32).reshape(2, 3)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            source = stack.enter_context(runtime.from_numpy(source_value))
            for dtype in (np.float32, np.float64, np.uint64):
                with self.subTest(dtype=np.dtype(dtype)):
                    indices = stack.enter_context(
                        runtime.from_numpy(np.array([0], dtype=dtype))
                    )
                    self.assert_index_error(
                        runtime, -3, "integer dtype",
                        lambda: runtime.take(source, indices, axis=1),
                    )

            boolean_indices_value = np.array([True, False], dtype=np.bool_)
            boolean_indices = stack.enter_context(
                runtime.from_numpy(boolean_indices_value)
            )
            result = stack.enter_context(
                runtime.take(source, boolean_indices, axis=1)
            )
            assert_array_equivalent(
                self, result,
                np.take(source_value, boolean_indices_value, axis=1),
            )

    def test_take_axis_scalar_empty_and_zero_length_results(self) -> None:
        cases = (
            (np.array(7, dtype=np.int64), np.array([0, -1]), 0),
            (np.arange(6, dtype=np.int64).reshape(2, 3),
             np.empty((2, 0), dtype=np.int32), 1),
            (np.empty((2, 0), dtype=np.float32),
             np.array([], dtype=np.int64), 1),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            for source_value, indices_value, axis in cases:
                with self.subTest(
                    source_shape=source_value.shape,
                    indices_shape=indices_value.shape,
                    axis=axis,
                ):
                    source = stack.enter_context(runtime.from_numpy(source_value))
                    indices = stack.enter_context(runtime.from_numpy(indices_value))
                    result = stack.enter_context(
                        runtime.take(source, indices, axis=axis)
                    )
                    assert_array_equivalent(
                        self, result,
                        np.take(source_value, indices_value, axis=axis),
                    )

            source = stack.enter_context(
                runtime.from_numpy(np.empty((2, 0), dtype=np.float64))
            )
            nonempty = stack.enter_context(
                runtime.from_numpy(np.array([0], dtype=np.int64))
            )
            self.assert_index_error(
                runtime, -6, "out of bounds",
                lambda: runtime.take(source, nonempty, axis=1),
            )

            ordinary = stack.enter_context(
                runtime.from_numpy(np.arange(6).reshape(2, 3))
            )
            empty = stack.enter_context(
                runtime.from_numpy(np.array([], dtype=np.int64))
            )
            self.assert_index_error(
                runtime, -5, "axis",
                lambda: runtime.take(ordinary, empty, axis=2),
            )

    def test_take_along_axis_dtype_rank_broadcast_and_axis_rules(self) -> None:
        source_value = np.arange(6, dtype=np.int64).reshape(2, 3)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            source = stack.enter_context(runtime.from_numpy(source_value))
            for dtype in (np.float64, np.bool_):
                with self.subTest(dtype=np.dtype(dtype)):
                    indices = stack.enter_context(runtime.from_numpy(
                        np.array([[0], [1]], dtype=dtype)
                    ))
                    self.assert_index_error(
                        runtime, -3, "integer dtype",
                        lambda: runtime.take_along_axis(source, indices, 1),
                    )

            uint_indices_value = np.array(
                [[2**64 - 1], [0]], dtype=np.uint64
            )
            uint_indices = stack.enter_context(
                runtime.from_numpy(uint_indices_value)
            )
            result = stack.enter_context(
                runtime.take_along_axis(source, uint_indices, 1)
            )
            assert_array_equivalent(
                self, result,
                np.take_along_axis(source_value, uint_indices_value, 1),
            )

            wrong_rank = stack.enter_context(
                runtime.from_numpy(np.array([0, 1], dtype=np.int64))
            )
            self.assert_index_error(
                runtime, -4, "same rank",
                lambda: runtime.take_along_axis(source, wrong_rank, 1),
            )
            mismatch = stack.enter_context(runtime.from_numpy(
                np.array([[0], [1], [2]], dtype=np.int64)
            ))
            self.assert_index_error(
                runtime, -7, "broadcast",
                lambda: runtime.take_along_axis(source, mismatch, 1),
            )
            self.assert_index_error(
                runtime, -5, "axis",
                lambda: runtime.take_along_axis(source, mismatch, 2),
            )

    def test_take_along_axis_zero_length_source_axis_is_explicit_and_leak_free(
        self,
    ) -> None:
        source_value = np.empty((2, 0, 3), dtype=np.float32)
        empty_indices_value = np.empty((2, 0, 3), dtype=np.int64)
        nonempty_indices_value = np.zeros((2, 1, 3), dtype=np.int64)

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for repetition in range(16):
                with self.subTest(path="success", repetition=repetition):
                    with ExitStack() as stack:
                        source = stack.enter_context(
                            runtime.from_numpy(source_value)
                        )
                        indices = stack.enter_context(
                            runtime.from_numpy(empty_indices_value)
                        )
                        result = stack.enter_context(
                            runtime.take_along_axis(source, indices, axis=1)
                        )
                        self.assertEqual((2, 0, 3), result.shape)
                        self.assertEqual(np.dtype(np.float32), result.numpy_dtype)
                        assert_array_equivalent(
                            self,
                            result,
                            np.take_along_axis(
                                source_value, empty_indices_value, axis=1
                            ),
                            compare_contiguity=False,
                        )
                    self.assertEqual(baseline, runtime.retained_bytes)

                with self.subTest(path="failure", repetition=repetition):
                    with ExitStack() as stack:
                        source = stack.enter_context(
                            runtime.from_numpy(source_value)
                        )
                        indices = stack.enter_context(
                            runtime.from_numpy(nonempty_indices_value)
                        )
                        self.assert_index_error(
                            runtime,
                            -6,
                            "out of bounds",
                            lambda: runtime.take_along_axis(
                                source, indices, axis=1
                            ),
                        )
                    self.assertEqual(baseline, runtime.retained_bytes)

    def test_compress_accepts_any_numeric_condition_dtype(self) -> None:
        source_value = np.arange(12, dtype=np.uint64).reshape(3, 4)
        conditions = (
            np.array([1, 0, -2], dtype=np.int64),
            np.array([0.5, 0.0, np.nan], dtype=np.float32),
            np.array([True, False, True], dtype=np.bool_),
            np.array([2**63, 0, 1], dtype=np.uint64),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            source = stack.enter_context(runtime.from_numpy(source_value))
            for condition_value in conditions:
                with self.subTest(dtype=condition_value.dtype):
                    condition = stack.enter_context(
                        runtime.from_numpy(condition_value)
                    )
                    result = stack.enter_context(
                        runtime.compress(condition, source, axis=0)
                    )
                    assert_array_equivalent(
                        self, result,
                        np.compress(condition_value, source_value, axis=0),
                    )

    def test_compress_validates_rank_and_overlong_true_entries_atomically(self) -> None:
        source_value = np.arange(6, dtype=np.int64).reshape(2, 3)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            source = stack.enter_context(runtime.from_numpy(source_value))
            false_tail_value = np.array([1, 0, 1, 0, 0], dtype=np.int16)
            false_tail = stack.enter_context(runtime.from_numpy(false_tail_value))
            result = stack.enter_context(
                runtime.compress(false_tail, source, axis=1)
            )
            assert_array_equivalent(
                self, result,
                np.compress(false_tail_value, source_value, axis=1),
            )

            true_tail = stack.enter_context(runtime.from_numpy(
                np.array([0, 0, 0, 1], dtype=np.int32)
            ))
            self.assert_index_error(
                runtime, -6, "out of bounds",
                lambda: runtime.compress(true_tail, source, axis=1),
            )
            rank_two = stack.enter_context(runtime.from_numpy(
                np.array([[1, 0, 1]], dtype=np.int8)
            ))
            self.assert_index_error(
                runtime, -4, "one-dimensional",
                lambda: runtime.compress(rank_two, source, axis=1),
            )

    def test_delete_integer_boolean_and_empty_objects_match_numpy(self) -> None:
        source_value = np.arange(12, dtype=np.int64).reshape(3, 4)
        objects = (
            np.array([3, -4, 3, 1], dtype=np.int64),
            np.array([True, False, True, False], dtype=np.bool_),
            np.array([], dtype=np.int32),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            source = stack.enter_context(runtime.from_numpy(source_value))
            for obj_value in objects:
                with self.subTest(dtype=obj_value.dtype, values=obj_value.tolist()):
                    obj = stack.enter_context(runtime.from_numpy(obj_value))
                    result = stack.enter_context(runtime.delete(source, obj, axis=-1))
                    assert_array_equivalent(
                        self, result,
                        np.delete(source_value, obj_value, axis=-1),
                        compare_contiguity=False,
                    )

    def test_delete_rejects_bad_masks_float_uint64_and_bounds_atomically(self) -> None:
        source = np.arange(6, dtype=np.int64).reshape(2, 3)
        cases = (
            (np.array([True, False]), -4, "match the axis length"),
            (np.array([[True, False, True]]), -4, "one-dimensional"),
            (np.array([1.0]), -3, "integer"),
            (np.array([2**64 - 1], dtype=np.uint64), -6, "out of bounds"),
            (np.array([3], dtype=np.int64), -6, "out of bounds"),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            native_source = stack.enter_context(runtime.from_numpy(source))
            for obj_value, status, message in cases:
                with self.subTest(dtype=obj_value.dtype, shape=obj_value.shape):
                    obj = stack.enter_context(runtime.from_numpy(obj_value))
                    self.assert_index_error(
                        runtime, status, message,
                        lambda obj=obj: runtime.delete(
                            native_source, obj, axis=1
                        ),
                    )

    def test_delete_axis_none_negative_axis_and_noncontiguous_source(self) -> None:
        base_value = np.arange(12, dtype=np.int32).reshape(3, 4)
        source_value = base_value.T
        cases = (
            (np.array([0, -1, 0], dtype=np.int64), None),
            (np.array([2, 0], dtype=np.int64), -1),
            (np.ones(3, dtype=np.bool_), 1),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            base = stack.enter_context(runtime.from_numpy(base_value))
            source = stack.enter_context(runtime.transpose(base, (1, 0)))
            for obj_value, axis in cases:
                with self.subTest(axis=axis, dtype=obj_value.dtype):
                    obj = stack.enter_context(runtime.from_numpy(obj_value))
                    result = stack.enter_context(runtime.delete(source, obj, axis))
                    assert_array_equivalent(
                        self, result,
                        np.delete(source_value, obj_value, axis=axis),
                        compare_contiguity=False,
                    )

    def test_insert_scalar_and_array_objects_have_distinct_broadcast_rules(self) -> None:
        source_value = np.arange(6, dtype=np.int32).reshape(2, 3)
        cases = (
            (np.array(1, dtype=np.int64), np.array([9, 8]), 1),
            (np.array([1], dtype=np.int64), np.array([9, 8]), 1),
            (np.array(1, dtype=np.int64), np.array([[9], [8]]), 1),
            (np.array([1], dtype=np.int64), np.array([[9], [8]]), 1),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            source = stack.enter_context(runtime.from_numpy(source_value))
            for obj_value, values_value, axis in cases:
                with self.subTest(
                    obj_shape=obj_value.shape, values_shape=values_value.shape
                ):
                    obj = stack.enter_context(runtime.from_numpy(obj_value))
                    values = stack.enter_context(runtime.from_numpy(values_value))
                    result = stack.enter_context(
                        runtime.insert(source, obj, values, axis)
                    )
                    assert_array_equivalent(
                        self, result,
                        np.insert(source_value, obj_value, values_value, axis),
                    )

    def test_insert_repeated_unsorted_negative_and_boolean_objects(self) -> None:
        source_value = np.arange(8, dtype=np.int64).reshape(2, 4)
        cases = (
            (np.array([2, 0, 2, -1]), np.array([[9, 8, 7, 6]])),
            (np.array([1, 1]), np.array([[9], [8]])),
            (np.array([True, False, True]), np.array([[9], [8]])),
            (np.array([], dtype=np.int64), np.empty((2, 0), dtype=np.int64)),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            source = stack.enter_context(runtime.from_numpy(source_value))
            for obj_value, values_value in cases:
                with self.subTest(
                    obj=obj_value.tolist(), values_shape=values_value.shape
                ):
                    obj = stack.enter_context(runtime.from_numpy(obj_value))
                    values = stack.enter_context(runtime.from_numpy(values_value))
                    result = stack.enter_context(
                        runtime.insert(source, obj, values, axis=1)
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        expected = np.insert(
                            source_value, obj_value, values_value, axis=1
                        )
                    assert_array_equivalent(self, result, expected)

    def test_insert_axis_none_empty_axis_and_noncontiguous_values(self) -> None:
        source_value = np.arange(6, dtype=np.int64).reshape(2, 3)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            source = stack.enter_context(runtime.from_numpy(source_value))
            flat_obj_value = np.array([1, 3], dtype=np.int64)
            flat_obj = stack.enter_context(runtime.from_numpy(flat_obj_value))
            flat_values_value = np.array([[9, 8]], dtype=np.int64)
            flat_values = stack.enter_context(
                runtime.from_numpy(flat_values_value)
            )
            flat_result = stack.enter_context(
                runtime.insert(source, flat_obj, flat_values, axis=None)
            )
            assert_array_equivalent(
                self, flat_result,
                np.insert(source_value, flat_obj_value, flat_values_value),
            )

            empty_source_value = np.empty((2, 0), dtype=np.int64)
            empty_source = stack.enter_context(
                runtime.from_numpy(empty_source_value)
            )
            scalar_obj = stack.enter_context(
                runtime.from_numpy(np.array(0, dtype=np.int64))
            )
            empty_values_value = np.array([9, 8], dtype=np.int64)
            empty_values = stack.enter_context(
                runtime.from_numpy(empty_values_value)
            )
            empty_result = stack.enter_context(
                runtime.insert(empty_source, scalar_obj, empty_values, axis=1)
            )
            assert_array_equivalent(
                self, empty_result,
                np.insert(empty_source_value, 0, empty_values_value, axis=1),
            )

            values_base_value = np.array([[9, 8], [7, 6]], dtype=np.int64)
            values_base = stack.enter_context(
                runtime.from_numpy(values_base_value)
            )
            values = stack.enter_context(runtime.transpose(values_base, (1, 0)))
            obj_value = np.array([1, 2], dtype=np.int64)
            obj = stack.enter_context(runtime.from_numpy(obj_value))
            result = stack.enter_context(runtime.insert(source, obj, values, 1))
            assert_array_equivalent(
                self, result,
                np.insert(source_value, obj_value, values_base_value.T, axis=1),
            )

    def test_insert_errors_are_explicit_and_do_not_return_partial_results(self) -> None:
        source_value = np.arange(6, dtype=np.int64).reshape(2, 3)
        cases = (
            (np.array([1.0]), np.array([9]), -3, "integer"),
            (np.array([2**64 - 1], dtype=np.uint64), np.array([9]),
             -6, "out of bounds"),
            (np.array([4], dtype=np.int64), np.array([9]),
             -6, "out of bounds"),
            (np.array([1, 2], dtype=np.int64), np.empty(0, dtype=np.int64),
             -7, "broadcast"),
            (np.array([[1]], dtype=np.int64), np.array([9]),
             -4, "one-dimensional"),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            source = stack.enter_context(runtime.from_numpy(source_value))
            for obj_value, values_value, status, message in cases:
                with self.subTest(
                    obj_dtype=obj_value.dtype,
                    obj_shape=obj_value.shape,
                    values_shape=values_value.shape,
                ):
                    obj = stack.enter_context(runtime.from_numpy(obj_value))
                    values = stack.enter_context(runtime.from_numpy(values_value))
                    self.assert_index_error(
                        runtime, status, message,
                        lambda obj=obj, values=values: runtime.insert(
                            source, obj, values, axis=1
                        ),
                    )

    def test_insert_casts_values_exactly_to_source_dtype(self) -> None:
        cases = (
            (np.array([2**53 + 1, 2**63 - 1], dtype=np.int64),
             np.array([2**63 + 5], dtype=np.uint64)),
            (np.array([2**63 + 1, 2**53 + 3], dtype=np.uint64),
             np.array([-1, 2**53 + 1], dtype=np.int64)),
            (np.array([1.25, -2.5], dtype=np.float32),
             np.array([2**53 + 1], dtype=np.int64)),
            (np.array([1.25, -2.5], dtype=np.float64),
             np.array([2**63 + 1], dtype=np.uint64)),
            (np.array([True, False], dtype=np.bool_),
             np.array([0.0, np.nan, -2.0], dtype=np.float64)),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            for source_value, values_value in cases:
                with self.subTest(
                    source_dtype=source_value.dtype,
                    values_dtype=values_value.dtype,
                ):
                    source = stack.enter_context(runtime.from_numpy(source_value))
                    obj_value = np.array([1], dtype=np.int64)
                    obj = stack.enter_context(runtime.from_numpy(obj_value))
                    values = stack.enter_context(runtime.from_numpy(values_value))
                    result = stack.enter_context(
                        runtime.insert(source, obj, values, axis=0)
                    )
                    actual = result.to_numpy()
                    expected = np.insert(
                        source_value, obj_value, values_value, axis=0
                    )
                    self.assertEqual(source_value.dtype, actual.dtype)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(expected.tobytes(), actual.tobytes())

    def test_insert_same_dtype_preserves_raw_bytes(self) -> None:
        values_by_dtype = (
            np.array([-(2**63) + 3, 2**53 + 1], dtype=np.int64),
            np.array([2**63 + 3, 2**64 - 2], dtype=np.uint64),
            np.array([np.nan, -0.0], dtype=np.float32),
            np.array([np.nan, -0.0], dtype=np.float64),
            np.array([True, False], dtype=np.bool_),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            for values_value in values_by_dtype:
                with self.subTest(dtype=values_value.dtype):
                    source_value = values_value[::-1].copy()
                    source = stack.enter_context(runtime.from_numpy(source_value))
                    obj_value = np.array([1, 1], dtype=np.int64)
                    obj = stack.enter_context(runtime.from_numpy(obj_value))
                    values = stack.enter_context(runtime.from_numpy(values_value))
                    result = stack.enter_context(runtime.insert(source, obj, values, 0))
                    actual = result.to_numpy()
                    expected = np.insert(source_value, obj_value, values_value)
                    self.assertEqual(expected.tobytes(), actual.tobytes())

    def test_legacy_indexing_exports_preserve_documented_axis_minus_one_abi(self) -> None:
        source_value = np.arange(6, dtype=np.int64).reshape(2, 3)
        indices_value = np.array([5, 0], dtype=np.int64)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            source = stack.enter_context(runtime.from_numpy(source_value))
            indices = stack.enter_context(runtime.from_numpy(indices_value))
            for function_name in ("cnp_take", "cnp_array_take"):
                with self.subTest(function=function_name):
                    result = stack.enter_context(self.legacy_array_result(
                        runtime, function_name, source, indices, -1
                    ))
                    assert_array_equivalent(
                        self, result,
                        np.take(source_value, indices_value, axis=None),
                    )

            along_indices_value = np.array([[2], [0]], dtype=np.int64)
            along_indices = stack.enter_context(
                runtime.from_numpy(along_indices_value)
            )
            along = stack.enter_context(self.legacy_array_result(
                runtime, "cnp_take_along_axis", source, along_indices, -1
            ))
            assert_array_equivalent(
                self, along,
                np.take_along_axis(source_value, along_indices_value, axis=-1),
            )

            condition_value = np.array([1, 0, 1, 0, 0, 0], dtype=np.int8)
            condition = stack.enter_context(runtime.from_numpy(condition_value))
            compressed = stack.enter_context(self.legacy_array_result(
                runtime, "cnp_compress", source, condition, -1
            ))
            assert_array_equivalent(
                self, compressed,
                np.compress(condition_value, source_value, axis=None),
            )

            deleted = stack.enter_context(self.legacy_array_result(
                runtime, "cnp_delete", source, indices, -1
            ))
            assert_array_equivalent(
                self, deleted,
                np.delete(source_value, indices_value, axis=None),
                compare_contiguity=False,
            )

            values_value = np.array([9], dtype=np.int64)
            values = stack.enter_context(runtime.from_numpy(values_value))
            inserted = stack.enter_context(self.legacy_insert_result(
                runtime, source, 1, values, -1
            ))
            assert_array_equivalent(
                self, inserted,
                np.insert(source_value, 1, values_value, axis=None),
            )

            bad = stack.enter_context(
                runtime.from_numpy(np.array([6], dtype=np.int64))
            )
            self.assert_index_error(
                runtime, -6, "out of bounds",
                lambda: self.legacy_array_result(
                    runtime, "cnp_take", source, bad, -1
                ),
            )


if __name__ == "__main__":
    unittest.main()
