from __future__ import annotations

import ctypes
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"


class CnpSlice(ctypes.Structure):
    _fields_ = [
        ("start", ctypes.c_int64),
        ("stop", ctypes.c_int64),
        ("step", ctypes.c_int64),
        ("has_start", ctypes.c_bool),
        ("has_stop", ctypes.c_bool),
        ("has_step", ctypes.c_bool),
    ]


def _function(runtime: CnumpyRuntime, name: str, argtypes, restype):
    function = getattr(runtime.dll, name)
    function.argtypes = argtypes
    function.restype = restype
    return function


def _owned(runtime: CnumpyRuntime, pointer, origin: str):
    return runtime._owned_result(pointer, origin)


def _assert_equal(case: unittest.TestCase, actual, expected: np.ndarray) -> None:
    case.assertEqual(expected.shape, actual.shape)
    case.assertEqual(expected.dtype, actual.numpy_dtype)
    np.testing.assert_array_equal(expected, actual.to_numpy(), strict=True)


def _invalid_axis_probe(name: str) -> None:
    with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
        baseline = runtime.retained_bytes
        source = stack.enter_context(
            runtime.from_numpy(np.arange(6, dtype=np.int64).reshape(2, 3))
        )
        other = stack.enter_context(
            runtime.from_numpy(np.arange(6, 12, dtype=np.int64).reshape(2, 3))
        )
        pointers = (ctypes.c_void_p * 2)(source.pointer, other.pointer)
        cases = {
            "cnp_swapaxes": (
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
                (source.pointer, 0, 3),
            ),
            "cnp_moveaxis": (
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
                (source.pointer, -3, 0),
            ),
            "cnp_squeeze": (
                [ctypes.c_void_p, ctypes.c_int],
                (source.pointer, 2),
            ),
            "cnp_expand_dims": (
                [ctypes.c_void_p, ctypes.c_int],
                (source.pointer, 4),
            ),
            "cnp_repeat": (
                [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int],
                (source.pointer, 2, 2),
            ),
            "cnp_flip": (
                [ctypes.c_void_p, ctypes.c_int],
                (source.pointer, 2),
            ),
            "cnp_rot90": (
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int],
                (source.pointer, 1, 0, 0),
            ),
            "cnp_roll": (
                [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int],
                (source.pointer, 1, 2),
            ),
            "cnp_concatenate": (
                [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
                (2, pointers, 2),
            ),
            "cnp_stack": (
                [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
                (2, pointers, 4),
            ),
            "cnp_append": (
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int],
                (source.pointer, other.pointer, 2),
            ),
        }
        argtypes, arguments = cases[name]
        function = _function(runtime, name, argtypes, ctypes.c_void_p)
        runtime.dll.cnp_clear_error()
        pointer = function(*arguments)
        if pointer:
            runtime.dll.cnp_array_free(pointer)
            raise AssertionError(f"{name} accepted an invalid axis")
        error = runtime.error_state()
        if error.status == 0 or error.function != name:
            raise AssertionError(
                f"{name} produced status={error.status} function={error.function!r}"
            )
        stack.close()
        if runtime.retained_bytes != baseline:
            raise AssertionError(
                f"{name} retained {runtime.retained_bytes - baseline} bytes"
            )


class ShapeSurfaceSemanticsTests(unittest.TestCase):
    SYMBOLS = frozenset(
        {
            "cnp_append",
            "cnp_array_slice",
            "cnp_broadcast_arrays",
            "cnp_broadcast_to",
            "cnp_column_stack",
            "cnp_concatenate",
            "cnp_dstack",
            "cnp_expand_dims",
            "cnp_flatten",
            "cnp_flip",
            "cnp_hstack",
            "cnp_moveaxis",
            "cnp_pad",
            "cnp_ravel",
            "cnp_repeat",
            "cnp_reshape",
            "cnp_row_stack",
            "cnp_roll",
            "cnp_rot90",
            "cnp_squeeze",
            "cnp_stack",
            "cnp_swapaxes",
            "cnp_tile",
            "cnp_vstack",
        }
    )

    def test_unary_shape_values_dtypes_strides_and_lifetimes_match_numpy_125(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            value = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
            source = stack.enter_context(runtime.from_numpy(value))

            shape_46 = (ctypes.c_int64 * 2)(4, 6)
            reshape = _function(
                runtime,
                "cnp_reshape",
                [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_int],
                ctypes.c_void_p,
            )
            unary_calls = (
                (
                    "cnp_reshape",
                    lambda: reshape(source.pointer, 2, shape_46, 0),
                    value.reshape(4, 6),
                ),
                (
                    "cnp_ravel",
                    lambda: _function(runtime, "cnp_ravel", [ctypes.c_void_p, ctypes.c_int], ctypes.c_void_p)(source.pointer, 0),
                    np.ravel(value, order="C"),
                ),
                (
                    "cnp_flatten",
                    lambda: _function(runtime, "cnp_flatten", [ctypes.c_void_p, ctypes.c_int], ctypes.c_void_p)(source.pointer, 0),
                    value.flatten(order="C"),
                ),
                (
                    "cnp_swapaxes",
                    lambda: _function(runtime, "cnp_swapaxes", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], ctypes.c_void_p)(source.pointer, 0, 2),
                    np.swapaxes(value, 0, 2),
                ),
                (
                    "cnp_moveaxis",
                    lambda: _function(runtime, "cnp_moveaxis", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], ctypes.c_void_p)(source.pointer, 0, 2),
                    np.moveaxis(value, 0, 2),
                ),
                (
                    "cnp_expand_dims",
                    lambda: _function(runtime, "cnp_expand_dims", [ctypes.c_void_p, ctypes.c_int], ctypes.c_void_p)(source.pointer, -1),
                    np.expand_dims(value, -1),
                ),
                (
                    "cnp_flip",
                    lambda: _function(runtime, "cnp_flip", [ctypes.c_void_p, ctypes.c_int], ctypes.c_void_p)(source.pointer, 1),
                    np.flip(value, 1),
                ),
                (
                    "cnp_roll",
                    lambda: _function(runtime, "cnp_roll", [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int], ctypes.c_void_p)(source.pointer, -2, 1),
                    np.roll(value, -2, axis=1),
                ),
            )
            for name, call, expected in unary_calls:
                with self.subTest(operation=name):
                    result = stack.enter_context(_owned(runtime, call(), name))
                    _assert_equal(self, result, expected)

            singleton_value = np.arange(6, dtype=np.int64).reshape(1, 2, 1, 3)
            singleton = stack.enter_context(runtime.from_numpy(singleton_value))
            squeeze = _function(
                runtime, "cnp_squeeze", [ctypes.c_void_p, ctypes.c_int], ctypes.c_void_p
            )
            squeezed = stack.enter_context(
                _owned(runtime, squeeze(singleton.pointer, 2), "cnp_squeeze")
            )
            _assert_equal(self, squeezed, np.squeeze(singleton_value, axis=2))

            small_value = np.arange(6, dtype=np.int64).reshape(2, 3)
            small = stack.enter_context(runtime.from_numpy(small_value))
            reps = (ctypes.c_int64 * 3)(2, 1, 2)
            tile = _function(
                runtime,
                "cnp_tile",
                [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)],
                ctypes.c_void_p,
            )
            tiled = stack.enter_context(
                _owned(runtime, tile(small.pointer, 3, reps), "cnp_tile")
            )
            _assert_equal(self, tiled, np.tile(small_value, (2, 1, 2)))

            repeat = _function(
                runtime,
                "cnp_repeat",
                [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int],
                ctypes.c_void_p,
            )
            repeated = stack.enter_context(
                _owned(runtime, repeat(small.pointer, 2, 0), "cnp_repeat")
            )
            _assert_equal(self, repeated, np.repeat(small_value, 2, axis=0))

            rot90 = _function(
                runtime,
                "cnp_rot90",
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int],
                ctypes.c_void_p,
            )
            for turns in (-1, 1, 2, 3):
                with self.subTest(operation="cnp_rot90", turns=turns):
                    rotated = stack.enter_context(
                        _owned(
                            runtime,
                            rot90(small.pointer, turns, 0, 1),
                            "cnp_rot90",
                        )
                    )
                    _assert_equal(
                        self, rotated, np.rot90(small_value, turns, (0, 1))
                    )

            pad = _function(
                runtime,
                "cnp_pad",
                [ctypes.c_void_p, ctypes.c_int64, ctypes.c_double],
                ctypes.c_void_p,
            )
            padded = stack.enter_context(
                _owned(runtime, pad(small.pointer, 1, -7.0), "cnp_pad")
            )
            _assert_equal(
                self,
                padded,
                np.pad(small_value, 1, mode="constant", constant_values=-7),
            )

            slices = (CnpSlice * 2)(
                CnpSlice(0, 0, 1, False, False, False),
                CnpSlice(0, 0, -1, False, False, True),
            )
            array_slice = _function(
                runtime,
                "cnp_array_slice",
                [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(CnpSlice)],
                ctypes.c_void_p,
            )
            sliced = stack.enter_context(
                _owned(runtime, array_slice(small.pointer, 2, slices), "cnp_array_slice")
            )
            _assert_equal(self, sliced, small_value[:, ::-1])

            vector_value = np.asarray([1, 2, 3], dtype=np.int64)
            vector = runtime.from_numpy(vector_value)
            target_shape = (ctypes.c_int64 * 2)(2, 3)
            broadcast_to = _function(
                runtime,
                "cnp_broadcast_to",
                [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)],
                ctypes.c_void_p,
            )
            broadcast = stack.enter_context(
                _owned(
                    runtime,
                    broadcast_to(vector.pointer, 2, target_shape),
                    "cnp_broadcast_to",
                )
            )
            vector.close()
            _assert_equal(self, broadcast, np.broadcast_to(vector_value, (2, 3)))

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_joining_and_legacy_projection_match_numpy_125(self) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            left_value = np.asarray([[1, 2], [3, 4]], dtype=np.int64)
            right_value = np.asarray([[5, 6], [7, 8]], dtype=np.int64)
            left = stack.enter_context(runtime.from_numpy(left_value))
            right = stack.enter_context(runtime.from_numpy(right_value))
            pointers = (ctypes.c_void_p * 2)(left.pointer, right.pointer)
            joins = (
                ("cnp_concatenate", [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int], (2, pointers, 0), np.concatenate((left_value, right_value), axis=0)),
                ("cnp_stack", [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int], (2, pointers, 1), np.stack((left_value, right_value), axis=1)),
                ("cnp_vstack", [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)], (2, pointers), np.vstack((left_value, right_value))),
                ("cnp_row_stack", [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)], (2, pointers), np.row_stack((left_value, right_value))),
                ("cnp_hstack", [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)], (2, pointers), np.hstack((left_value, right_value))),
                ("cnp_dstack", [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)], (2, pointers), np.dstack((left_value, right_value))),
                ("cnp_column_stack", [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)], (2, pointers), np.column_stack((left_value, right_value))),
            )
            for name, argtypes, arguments, expected in joins:
                with self.subTest(operation=name):
                    function = _function(runtime, name, argtypes, ctypes.c_void_p)
                    result = stack.enter_context(
                        _owned(runtime, function(*arguments), name)
                    )
                    _assert_equal(self, result, expected)

            append = _function(
                runtime,
                "cnp_append",
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int],
                ctypes.c_void_p,
            )
            appended = stack.enter_context(
                _owned(runtime, append(left.pointer, right.pointer, -1), "cnp_append")
            )
            _assert_equal(self, appended, np.append(left_value, right_value))

            mixed_left_value = np.asarray([[1, 2]], dtype=np.int16)
            mixed_right_value = np.asarray([[3.5, 4.5]], dtype=np.float32)
            mixed_left = stack.enter_context(runtime.from_numpy(mixed_left_value))
            mixed_right = stack.enter_context(runtime.from_numpy(mixed_right_value))
            mixed_pointers = (ctypes.c_void_p * 2)(
                mixed_left.pointer, mixed_right.pointer
            )
            concatenate = _function(
                runtime,
                "cnp_concatenate",
                [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
                ctypes.c_void_p,
            )
            mixed = stack.enter_context(
                _owned(runtime, concatenate(2, mixed_pointers, 0), "cnp_concatenate")
            )
            _assert_equal(
                self,
                mixed,
                np.concatenate((mixed_left_value, mixed_right_value), axis=0),
            )

            legacy_broadcast = _function(
                runtime,
                "cnp_broadcast_arrays",
                [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)],
                ctypes.c_void_p,
            )
            legacy_source = runtime.from_numpy(left_value)
            legacy_pointers = (ctypes.c_void_p * 2)(
                legacy_source.pointer, right.pointer
            )
            legacy = stack.enter_context(
                _owned(
                    runtime,
                    legacy_broadcast(2, legacy_pointers),
                    "cnp_broadcast_arrays",
                )
            )
            _assert_equal(self, legacy, left_value.copy())
            legacy_source.close()
            _assert_equal(self, legacy, left_value.copy())

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_null_shape_requests_are_explicit_and_allocation_atomic(self) -> None:
        shape = (ctypes.c_int64 * 1)(1)
        reps = (ctypes.c_int64 * 1)(1)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            calls = (
                ("cnp_reshape", [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_int], (None, 1, shape, 0)),
                ("cnp_ravel", [ctypes.c_void_p, ctypes.c_int], (None, 0)),
                ("cnp_flatten", [ctypes.c_void_p, ctypes.c_int], (None, 0)),
                ("cnp_swapaxes", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], (None, 0, 0)),
                ("cnp_moveaxis", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], (None, 0, 0)),
                ("cnp_squeeze", [ctypes.c_void_p, ctypes.c_int], (None, -1)),
                ("cnp_expand_dims", [ctypes.c_void_p, ctypes.c_int], (None, 0)),
                ("cnp_broadcast_to", [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)], (None, 1, shape)),
                ("cnp_broadcast_arrays", [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)], (0, None)),
                ("cnp_concatenate", [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int], (0, None, 0)),
                ("cnp_stack", [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int], (0, None, 0)),
                ("cnp_vstack", [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)], (0, None)),
                ("cnp_hstack", [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)], (0, None)),
                ("cnp_dstack", [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)], (0, None)),
                ("cnp_column_stack", [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)], (0, None)),
                ("cnp_tile", [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)], (None, 1, reps)),
                ("cnp_repeat", [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int], (None, 1, 0)),
                ("cnp_flip", [ctypes.c_void_p, ctypes.c_int], (None, 0)),
                ("cnp_rot90", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int], (None, 1, 0, 1)),
                ("cnp_roll", [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int], (None, 1, 0)),
                ("cnp_append", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int], (None, None, -1)),
                ("cnp_pad", [ctypes.c_void_p, ctypes.c_int64, ctypes.c_double], (None, 1, 0.0)),
                ("cnp_array_slice", [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(CnpSlice)], (None, 0, None)),
            )
            for name, argtypes, arguments in calls:
                with self.subTest(operation=name):
                    function = _function(runtime, name, argtypes, ctypes.c_void_p)
                    runtime.dll.cnp_clear_error()
                    self.assertFalse(function(*arguments))
                    error = runtime.error_state()
                    self.assertNotEqual(0, error.status)
                    self.assertEqual(name, error.function)
                    self.assertEqual(baseline, runtime.retained_bytes)

    def test_invalid_axes_fail_explicitly_without_process_corruption(self) -> None:
        for name in (
            "cnp_swapaxes",
            "cnp_moveaxis",
            "cnp_squeeze",
            "cnp_expand_dims",
            "cnp_repeat",
            "cnp_flip",
            "cnp_rot90",
            "cnp_roll",
            "cnp_concatenate",
            "cnp_stack",
            "cnp_append",
        ):
            with self.subTest(operation=name):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "benchmark.tests.test_shape_surface_semantics",
                        "--invalid-axis",
                        name,
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(
                    0,
                    completed.returncode,
                    completed.stdout + completed.stderr,
                )


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--invalid-axis":
        _invalid_axis_probe(sys.argv[2])
    else:
        unittest.main()
