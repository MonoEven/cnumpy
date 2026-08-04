from __future__ import annotations

import ctypes
from contextlib import ExitStack
from pathlib import Path
import unittest
import warnings

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"


class IndexSurfaceSemanticsTests(unittest.TestCase):
    SYMBOLS = frozenset(
        {
            "cnp_argwhere",
            "cnp_array_boolean_index",
            "cnp_array_fancy_index",
            "cnp_array_getitem",
            "cnp_array_nonzero",
            "cnp_array_where",
            "cnp_count_nonzero",
            "cnp_count_nonzero_v2",
            "cnp_flatnonzero",
        }
    )

    @staticmethod
    def _function(runtime: CnumpyRuntime, name: str, argtypes, restype):
        function = getattr(runtime.dll, name)
        function.argtypes = argtypes
        function.restype = restype
        return function

    def test_getitem_boolean_fancy_and_where_match_numpy_exact_values_lifetime(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as results:
            baseline = runtime.retained_bytes
            getitem = self._function(
                runtime,
                "cnp_array_getitem",
                [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64)],
                ctypes.c_void_p,
            )
            boolean_index = self._function(
                runtime,
                "cnp_array_boolean_index",
                [ctypes.c_void_p, ctypes.c_void_p],
                ctypes.c_void_p,
            )
            fancy_index = self._function(
                runtime,
                "cnp_array_fancy_index",
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int],
                ctypes.c_void_p,
            )
            where = self._function(
                runtime,
                "cnp_array_where",
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
                ctypes.c_void_p,
            )

            values = np.asarray(
                [[2**60 + 1, 2**60 + 2, 2**60 + 3],
                 [2**60 + 4, 2**60 + 5, 2**60 + 6]],
                dtype=np.int64,
            )
            mask_values = np.asarray(
                [[True, False, True], [False, True, False]], dtype=np.bool_
            )
            index_values = np.asarray([-1, 0], dtype=np.int64)
            condition_values = np.asarray([[True], [False]], dtype=np.bool_)
            alternative = np.asarray(
                [[-(2**60) + 1, -(2**60) + 2, -(2**60) + 3]],
                dtype=np.int64,
            )
            with runtime.from_numpy(values) as source, runtime.from_numpy(
                mask_values
            ) as mask, runtime.from_numpy(index_values) as indices, runtime.from_numpy(
                condition_values
            ) as condition, runtime.from_numpy(alternative) as other:
                coordinates = (ctypes.c_int64 * 2)(-1, 0)
                item = results.enter_context(
                    runtime._owned_result(
                        getitem(source.pointer, coordinates), "cnp_array_getitem"
                    )
                )
                selected = results.enter_context(
                    runtime._owned_result(
                        boolean_index(source.pointer, mask.pointer),
                        "cnp_array_boolean_index",
                    )
                )
                fancy = results.enter_context(
                    runtime._owned_result(
                        fancy_index(source.pointer, indices.pointer, 1),
                        "cnp_array_fancy_index",
                    )
                )
                chosen = results.enter_context(
                    runtime._owned_result(
                        where(condition.pointer, source.pointer, other.pointer),
                        "cnp_array_where",
                    )
                )

            np.testing.assert_array_equal(values[-1, 0], item.to_numpy(), strict=True)
            np.testing.assert_array_equal(
                values[mask_values], selected.to_numpy(), strict=True
            )
            np.testing.assert_array_equal(
                np.take(values, index_values, axis=1),
                fancy.to_numpy(),
                strict=True,
            )
            np.testing.assert_array_equal(
                np.where(condition_values, values, alternative),
                chosen.to_numpy(),
                strict=True,
            )

            results.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_nonzero_argwhere_flatnonzero_and_count_match_numpy_strides(self) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as results:
            baseline = runtime.retained_bytes
            nonzero = self._function(
                runtime, "cnp_array_nonzero", [ctypes.c_void_p], ctypes.c_void_p
            )
            argwhere = self._function(
                runtime, "cnp_argwhere", [ctypes.c_void_p], ctypes.c_void_p
            )
            flatnonzero = self._function(
                runtime, "cnp_flatnonzero", [ctypes.c_void_p], ctypes.c_void_p
            )
            count_nonzero = self._function(
                runtime,
                "cnp_count_nonzero",
                [ctypes.c_void_p, ctypes.c_int],
                ctypes.c_int64,
            )
            original = np.asarray(
                [[0.0 + 0.0j, 2.0 + 0.0j, 0.0 + 0.0j],
                 [np.nan + 0.0j, 0.0 + 0.0j, 0.0 + 3.0j]],
                dtype=np.complex128,
            )
            with runtime.from_numpy(original) as owner:
                source = runtime.transpose(owner, (1, 0))
                stacked = results.enter_context(
                    runtime._owned_result(
                        nonzero(source.pointer), "cnp_array_nonzero"
                    )
                )
                locations = results.enter_context(
                    runtime._owned_result(argwhere(source.pointer), "cnp_argwhere")
                )
                flat = results.enter_context(
                    runtime._owned_result(
                        flatnonzero(source.pointer), "cnp_flatnonzero"
                    )
                )
                runtime.dll.cnp_clear_error()
                count = int(count_nonzero(source.pointer, -1))
                self.assertEqual(0, runtime.error_state().status)
                source.close()

            values = original.T
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                expected_nonzero = np.vstack(np.nonzero(values))
            np.testing.assert_array_equal(
                expected_nonzero, stacked.to_numpy(), strict=True
            )
            np.testing.assert_array_equal(
                np.argwhere(values), locations.to_numpy(), strict=True
            )
            np.testing.assert_array_equal(
                np.flatnonzero(values), flat.to_numpy(), strict=True
            )
            self.assertEqual(int(np.count_nonzero(values)), count)

            results.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_count_nonzero_v2_axes_keepdims_empty_scalar_and_lifetime(
        self,
    ) -> None:
        values = (
            np.asarray(3, dtype=np.int64),
            np.asarray(
                [[0.0 + 0.0j, np.nan + 0.0j, 0.0 + 2.0j],
                 [0.0 + 0.0j, 4.0 + 0.0j, 0.0 + 0.0j]],
                dtype=np.complex128,
            ).T,
            np.empty((2, 0, 3), dtype=np.uint8),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            count_nonzero_v2 = self._function(
                runtime,
                "cnp_count_nonzero_v2",
                [
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_bool,
                    ctypes.c_bool,
                ],
                ctypes.c_void_p,
            )
            for value in values:
                axes = (None, 0, -1) if value.ndim == 0 else (
                    None,
                    *range(-value.ndim, value.ndim),
                )
                for axis in axes:
                    for keepdims in (False, True):
                        with self.subTest(
                            shape=value.shape,
                            axis=axis,
                            keepdims=keepdims,
                        ):
                            source = runtime.from_numpy(value)
                            runtime.dll.cnp_clear_error()
                            pointer = count_nonzero_v2(
                                source.pointer,
                                0 if axis is None else axis,
                                axis is None,
                                keepdims,
                            )
                            actual = runtime._owned_result(
                                pointer, "cnp_count_nonzero_v2"
                            )
                            source.close()
                            try:
                                expected = np.asarray(
                                    np.count_nonzero(
                                        value,
                                        axis=axis,
                                        keepdims=keepdims,
                                    ),
                                    dtype=np.int64,
                                )
                                np.testing.assert_array_equal(
                                    expected,
                                    actual.to_numpy(),
                                    strict=True,
                                )
                            finally:
                                actual.close()
                            self.assertEqual(
                                baseline, runtime.retained_bytes
                            )

            with runtime.from_numpy(
                np.arange(6, dtype=np.int64).reshape(2, 3)
            ) as source:
                for axis in (2, -3):
                    with self.subTest(error_axis=axis):
                        before_error = runtime.retained_bytes
                        runtime.dll.cnp_clear_error()
                        pointer = count_nonzero_v2(
                            source.pointer, axis, False, False
                        )
                        self.assertFalse(pointer)
                        error = runtime.error_state()
                        self.assertNotEqual(0, error.status)
                        self.assertEqual(
                            "cnp_count_nonzero_v2", error.function
                        )
                        self.assertIn(
                            f"axis {axis} is out of bounds", error.message
                        )
                        self.assertEqual(
                            before_error, runtime.retained_bytes
                        )
            runtime.dll.cnp_clear_error()
            self.assertFalse(
                count_nonzero_v2(None, 0, True, False)
            )
            self.assertEqual(
                "cnp_count_nonzero_v2", runtime.error_state().function
            )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_index_errors_are_explicit_atomic_and_nonretaining(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            getitem = self._function(
                runtime,
                "cnp_array_getitem",
                [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64)],
                ctypes.c_void_p,
            )
            boolean_index = self._function(
                runtime,
                "cnp_array_boolean_index",
                [ctypes.c_void_p, ctypes.c_void_p],
                ctypes.c_void_p,
            )
            fancy_index = self._function(
                runtime,
                "cnp_array_fancy_index",
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int],
                ctypes.c_void_p,
            )
            where = self._function(
                runtime,
                "cnp_array_where",
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
                ctypes.c_void_p,
            )
            count_nonzero = self._function(
                runtime,
                "cnp_count_nonzero",
                [ctypes.c_void_p, ctypes.c_int],
                ctypes.c_int64,
            )
            with runtime.from_numpy(np.arange(6, dtype=np.int64).reshape(2, 3)) as source, runtime.from_numpy(
                np.ones((2, 3), dtype=np.int64)
            ) as nonbool, runtime.from_numpy(np.asarray([0.0, 1.0])) as float_indices:
                cases = (
                    (
                        "cnp_array_getitem",
                        lambda: getitem(
                            source.pointer, (ctypes.c_int64 * 2)(2, 0)
                        ),
                    ),
                    (
                        "cnp_array_boolean_index",
                        lambda: boolean_index(source.pointer, nonbool.pointer),
                    ),
                    (
                        "cnp_array_fancy_index",
                        lambda: fancy_index(source.pointer, float_indices.pointer, 1),
                    ),
                    (
                        "cnp_array_where",
                        lambda: where(nonbool.pointer, source.pointer, None),
                    ),
                )
                active = runtime.retained_bytes
                for name, call in cases:
                    with self.subTest(symbol=name):
                        runtime.dll.cnp_clear_error()
                        pointer = call()
                        if pointer:
                            runtime._owned_result(pointer, name).close()
                        self.assertFalse(pointer)
                        self.assertEqual(name, runtime.error_state().function)
                        self.assertEqual(active, runtime.retained_bytes)

                runtime.dll.cnp_clear_error()
                self.assertEqual(0, count_nonzero(source.pointer, 0))
                self.assertEqual("cnp_count_nonzero", runtime.error_state().function)

            for name in (
                "cnp_array_nonzero",
                "cnp_argwhere",
                "cnp_flatnonzero",
            ):
                function = self._function(
                    runtime, name, [ctypes.c_void_p], ctypes.c_void_p
                )
                runtime.dll.cnp_clear_error()
                self.assertFalse(function(None))
                self.assertEqual(name, runtime.error_state().function)
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
