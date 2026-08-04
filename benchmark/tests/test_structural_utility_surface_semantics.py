from __future__ import annotations

import ctypes
from contextlib import ExitStack
from pathlib import Path
import unittest

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"


class StructuralUtilitySurfaceSemanticsTests(unittest.TestCase):
    SYMBOLS = frozenset(
        {
            "cnp_diagflat",
            "cnp_diagonal",
            "cnp_extract",
            "cnp_fill_diagonal",
            "cnp_nonzero",
            "cnp_tril",
            "cnp_trim_zeros",
            "cnp_triu",
        }
    )

    @staticmethod
    def _function(runtime: CnumpyRuntime, name: str, argtypes, restype):
        function = getattr(runtime.dll, name)
        function.argtypes = argtypes
        function.restype = restype
        return function

    def test_array_results_match_numpy_dtypes_strides_and_lifetimes(self) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as sources, ExitStack() as results:
            baseline = runtime.retained_bytes
            tril = self._function(
                runtime,
                "cnp_tril",
                [ctypes.c_void_p, ctypes.c_int],
                ctypes.c_void_p,
            )
            triu = self._function(
                runtime,
                "cnp_triu",
                [ctypes.c_void_p, ctypes.c_int],
                ctypes.c_void_p,
            )
            diagflat = self._function(
                runtime,
                "cnp_diagflat",
                [ctypes.c_void_p, ctypes.c_int],
                ctypes.c_void_p,
            )
            diagonal = self._function(
                runtime,
                "cnp_diagonal",
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int],
                ctypes.c_void_p,
            )
            extract = self._function(
                runtime,
                "cnp_extract",
                [ctypes.c_void_p, ctypes.c_void_p],
                ctypes.c_void_p,
            )
            nonzero = self._function(
                runtime,
                "cnp_nonzero",
                [ctypes.c_void_p],
                ctypes.c_void_p,
            )
            trim_zeros = self._function(
                runtime,
                "cnp_trim_zeros",
                [ctypes.c_void_p, ctypes.c_char_p],
                ctypes.c_void_p,
            )

            cube_values = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
            cube_owner = sources.enter_context(runtime.from_numpy(cube_values))
            cube = sources.enter_context(runtime.transpose(cube_owner))
            cube_expected = cube_values.T
            lower = results.enter_context(
                runtime._owned_result(tril(cube.pointer, -1), "cnp_tril")
            )
            upper = results.enter_context(
                runtime._owned_result(triu(cube.pointer, 1), "cnp_triu")
            )

            wide_values = np.asarray(
                [[2**60 + 1, 2**60 + 3]], dtype=np.int64
            )
            wide = sources.enter_context(runtime.from_numpy(wide_values))
            diagonal_matrix = results.enter_context(
                runtime._owned_result(
                    diagflat(wide.pointer, -2), "cnp_diagflat"
                )
            )

            diagonal_source_values = np.arange(24, dtype=np.int64).reshape(
                2, 3, 4
            )
            diagonal_source = sources.enter_context(
                runtime.from_numpy(diagonal_source_values)
            )
            diagonal_view = results.enter_context(
                runtime._owned_result(
                    diagonal(diagonal_source.pointer, 1, 0, 2),
                    "cnp_diagonal",
                )
            )

            selected_values = np.asarray(
                [2**60 + 1, 0, 2**60 + 3, 0], dtype=np.int64
            )
            condition_values = np.asarray([0 + 1j, 0, 2 + 0j])
            selected_source = sources.enter_context(
                runtime.from_numpy(selected_values)
            )
            condition = sources.enter_context(
                runtime.from_numpy(condition_values)
            )
            selected = results.enter_context(
                runtime._owned_result(
                    extract(condition.pointer, selected_source.pointer),
                    "cnp_extract",
                )
            )

            complex_values = np.asarray(
                [[0 + 0j, 0 + 2j], [3 + 0j, 0 + 0j]],
                dtype=np.complex128,
            )
            complex_source = sources.enter_context(
                runtime.from_numpy(complex_values)
            )
            locations = results.enter_context(
                runtime._owned_result(
                    nonzero(complex_source.pointer), "cnp_nonzero"
                )
            )

            trim_values = np.asarray(
                [0, 0, 2**60 + 1, 0], dtype=np.int64
            )
            trim_source = sources.enter_context(runtime.from_numpy(trim_values))
            trimmed = results.enter_context(
                runtime._owned_result(
                    trim_zeros(trim_source.pointer, b"fb"),
                    "cnp_trim_zeros",
                )
            )

            sources.close()

            np.testing.assert_array_equal(
                np.tril(cube_expected, -1), lower.to_numpy(), strict=True
            )
            np.testing.assert_array_equal(
                np.triu(cube_expected, 1), upper.to_numpy(), strict=True
            )
            np.testing.assert_array_equal(
                np.diagflat(wide_values, -2),
                diagonal_matrix.to_numpy(),
                strict=True,
            )
            np.testing.assert_array_equal(
                np.diagonal(diagonal_source_values, 1, 0, 2),
                diagonal_view.to_numpy(),
                strict=True,
            )
            np.testing.assert_array_equal(
                np.extract(condition_values, selected_values),
                selected.to_numpy(),
                strict=True,
            )
            np.testing.assert_array_equal(
                np.vstack(np.nonzero(complex_values)),
                locations.to_numpy(),
                strict=True,
            )
            np.testing.assert_array_equal(
                np.trim_zeros(trim_values, "fb"),
                trimmed.to_numpy(),
                strict=True,
            )

            results.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_fill_diagonal_matches_numpy_and_is_allocation_neutral(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            fill_diagonal = self._function(
                runtime,
                "cnp_fill_diagonal",
                [ctypes.c_void_p, ctypes.c_double],
                ctypes.c_int,
            )
            values = np.arange(12, dtype=np.int64).reshape(3, 4).T
            owner = runtime.from_numpy(values.T)
            source = runtime.transpose(owner)
            expected = values.copy()
            np.fill_diagonal(expected, 7.25)
            self.assertEqual(0, fill_diagonal(source.pointer, 7.25))
            np.testing.assert_array_equal(expected, source.to_numpy(), strict=True)
            source.close()
            owner.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_invalid_requests_are_explicit_atomic_and_nonretaining(self) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            tril = self._function(runtime, "cnp_tril", [ctypes.c_void_p, ctypes.c_int], ctypes.c_void_p)
            triu = self._function(runtime, "cnp_triu", [ctypes.c_void_p, ctypes.c_int], ctypes.c_void_p)
            diagflat = self._function(runtime, "cnp_diagflat", [ctypes.c_void_p, ctypes.c_int], ctypes.c_void_p)
            diagonal = self._function(runtime, "cnp_diagonal", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int], ctypes.c_void_p)
            fill_diagonal = self._function(runtime, "cnp_fill_diagonal", [ctypes.c_void_p, ctypes.c_double], ctypes.c_int)
            extract = self._function(runtime, "cnp_extract", [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_void_p)
            nonzero = self._function(runtime, "cnp_nonzero", [ctypes.c_void_p], ctypes.c_void_p)
            trim_zeros = self._function(runtime, "cnp_trim_zeros", [ctypes.c_void_p, ctypes.c_char_p], ctypes.c_void_p)

            vector = stack.enter_context(
                runtime.from_numpy(np.asarray([0, 1, 0], dtype=np.int64))
            )
            matrix = stack.enter_context(
                runtime.from_numpy(np.arange(6, dtype=np.int64).reshape(2, 3))
            )
            long_condition = stack.enter_context(
                runtime.from_numpy(np.asarray([1, 0, 1, 0, 1, 0, 1], dtype=np.int64))
            )
            diagonal_view = stack.enter_context(
                runtime._owned_result(
                    diagonal(matrix.pointer, 0, 0, 1), "cnp_diagonal"
                )
            )

            pointer_cases = (
                ("cnp_tril", lambda: tril(vector.pointer, 0)),
                ("cnp_triu", lambda: triu(vector.pointer, 0)),
                ("cnp_diagflat", lambda: diagflat(None, 0)),
                ("cnp_diagonal", lambda: diagonal(matrix.pointer, 0, 1, 1)),
                ("cnp_extract", lambda: extract(long_condition.pointer, matrix.pointer)),
                ("cnp_nonzero", lambda: nonzero(None)),
                ("cnp_trim_zeros", lambda: trim_zeros(matrix.pointer, b"fb")),
            )
            active = runtime.retained_bytes
            for name, call in pointer_cases:
                with self.subTest(symbol=name):
                    runtime.dll.cnp_clear_error()
                    pointer = call()
                    if pointer:
                        runtime._owned_result(pointer, name).close()
                    self.assertFalse(pointer)
                    self.assertEqual(name, runtime.error_state().function)
                    self.assertEqual(active, runtime.retained_bytes)

            before = diagonal_view.to_numpy().copy()
            runtime.dll.cnp_clear_error()
            self.assertNotEqual(0, fill_diagonal(diagonal_view.pointer, 9.0))
            self.assertEqual(
                "cnp_fill_diagonal", runtime.error_state().function
            )
            np.testing.assert_array_equal(before, diagonal_view.to_numpy())
            self.assertEqual(active, runtime.retained_bytes)

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
