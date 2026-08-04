from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest
import ctypes
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from compat.cnumpy_ctypes import CnumpyError, CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"


SHUFFLE_HEAP_PROBE = r"""
import ctypes
import sys

dll = ctypes.CDLL(sys.argv[1])
dll.cnp_init.argtypes = []
dll.cnp_init.restype = ctypes.c_int
dll.cnp_cleanup.argtypes = []
dll.cnp_cleanup.restype = None
dll.cnp_array_from_data.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_int,
    ctypes.c_int,
]
dll.cnp_array_from_data.restype = ctypes.c_void_p
dll.cnp_random_shuffle.argtypes = [ctypes.c_void_p]
dll.cnp_random_shuffle.restype = None
dll.cnp_array_decref.argtypes = [ctypes.c_void_p]
dll.cnp_array_decref.restype = None

if dll.cnp_init() != 0:
    raise RuntimeError("cnp_init failed")

rows = 64
columns = 256
shape = (ctypes.c_int64 * 2)(rows, columns)
values = (ctypes.c_double * (rows * columns))(
    *(row * columns + column for row in range(rows) for column in range(columns))
)
for _ in range(128):
    array = dll.cnp_array_from_data(values, 2, shape, 13, 0)
    if not array:
        raise RuntimeError("cnp_array_from_data failed")
    dll.cnp_random_shuffle(array)
    dll.cnp_array_decref(array)

dll.cnp_cleanup()
"""


class RandomShuffleMemorySafetyTests(unittest.TestCase):
    def test_multidimensional_row_shuffle_does_not_corrupt_the_heap(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(SHUFFLE_HEAP_PROBE), str(DLL)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(
            0,
            completed.returncode,
            {
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )


class RandomPermutationShuffleSemanticsTests(unittest.TestCase):
    @staticmethod
    def _transpose(runtime: CnumpyRuntime, source):
        function = runtime.dll.cnp_transpose
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(
            function(source.pointer, None), "cnp_transpose"
        )

    @staticmethod
    def _seed(runtime: CnumpyRuntime, seed: int) -> None:
        runtime.dll.cnp_random_seed.argtypes = [ctypes.c_uint64]
        runtime.dll.cnp_random_seed.restype = None
        runtime.dll.cnp_random_seed(seed)

    @staticmethod
    def _shuffle(runtime: CnumpyRuntime, source) -> None:
        function = runtime.dll.cnp_random_shuffle
        function.argtypes = [ctypes.c_void_p]
        function.restype = None
        runtime.dll.cnp_clear_error()
        function(source.pointer)
        error = runtime.error_state()
        if error.status != 0:
            raise error

    @staticmethod
    def _permutation(runtime: CnumpyRuntime, source):
        function = runtime.dll.cnp_random_permutation
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(
            function(source.pointer), "cnp_random_permutation"
        )

    @staticmethod
    def _sorted_rows(values: np.ndarray) -> list[tuple[int, ...]]:
        return sorted(tuple(int(value) for value in row) for row in values)

    def test_shuffle_matches_numpy_first_axis_contract_for_strided_view(
        self,
    ) -> None:
        self.assertEqual("1.25.0", np.__version__)
        logical = np.arange(20, dtype=np.int32).reshape(4, 5)
        numpy_owner = logical.T.copy()
        numpy_view = numpy_owner.T
        np.random.RandomState(20260804).shuffle(numpy_view)
        self.assertEqual(
            self._sorted_rows(logical), self._sorted_rows(numpy_view)
        )

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            first_owner = stack.enter_context(runtime.from_numpy(logical.T))
            first_view = stack.enter_context(
                self._transpose(runtime, first_owner)
            )
            second_owner = stack.enter_context(runtime.from_numpy(logical.T))
            second_view = stack.enter_context(
                self._transpose(runtime, second_owner)
            )
            self.assertFalse(first_view.c_contiguous)

            self._seed(runtime, 20260804)
            self._shuffle(runtime, first_view)
            self._seed(runtime, 20260804)
            self._shuffle(runtime, second_view)

            actual = first_view.to_numpy()
            self.assertEqual(logical.shape, actual.shape)
            self.assertEqual(logical.dtype, actual.dtype)
            self.assertEqual(
                self._sorted_rows(logical), self._sorted_rows(actual)
            )
            np.testing.assert_array_equal(actual, second_view.to_numpy())
            np.testing.assert_array_equal(actual, first_owner.to_numpy().T)

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_permutation_copies_strided_input_and_survives_source_release(
        self,
    ) -> None:
        logical = np.arange(24, dtype=np.int16).reshape(6, 4)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            owner = runtime.from_numpy(logical.T)
            source = self._transpose(runtime, owner)
            self._seed(runtime, 0x123456789ABCDEF0)
            result = self._permutation(runtime, source)
            self._seed(runtime, 0x123456789ABCDEF0)
            replay = self._permutation(runtime, source)

            np.testing.assert_array_equal(logical, source.to_numpy())
            self.assertEqual(logical.shape, result.shape)
            self.assertEqual(logical.dtype, result.numpy_dtype)
            self.assertEqual(
                self._sorted_rows(logical),
                self._sorted_rows(result.to_numpy()),
            )
            np.testing.assert_array_equal(result.to_numpy(), replay.to_numpy())

            owner.close()
            source.close()
            np.testing.assert_array_equal(result.to_numpy(), replay.to_numpy())
            result.close()
            replay.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_null_and_scalar_inputs_surface_numpy_compatible_errors(
        self,
    ) -> None:
        numpy_scalar = np.asarray(7, dtype=np.int64)
        with self.assertRaisesRegex(TypeError, "unsized"):
            np.random.RandomState(7).shuffle(numpy_scalar)
        with self.assertRaisesRegex(IndexError, "at least 1-dimensional"):
            np.random.RandomState(7).permutation(numpy_scalar)

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            scalar = stack.enter_context(runtime.from_numpy(numpy_scalar))
            shuffle = runtime.dll.cnp_random_shuffle
            shuffle.argtypes = [ctypes.c_void_p]
            shuffle.restype = None
            permutation = runtime.dll.cnp_random_permutation
            permutation.argtypes = [ctypes.c_void_p]
            permutation.restype = ctypes.c_void_p

            for source, label in ((None, "NULL"), (scalar.pointer, "0-D")):
                with self.subTest(operation="shuffle", source=label):
                    runtime.dll.cnp_clear_error()
                    before = runtime.retained_bytes
                    shuffle(source)
                    error = runtime.error_state()
                    self.assertNotEqual(0, error.status)
                    self.assertEqual("cnp_random_shuffle", error.function)
                    self.assertEqual(before, runtime.retained_bytes)

                with self.subTest(operation="permutation", source=label):
                    runtime.dll.cnp_clear_error()
                    before = runtime.retained_bytes
                    pointer = permutation(source)
                    if pointer:
                        runtime.dll.cnp_array_decref(pointer)
                    self.assertFalse(pointer)
                    error = runtime.error_state()
                    self.assertNotEqual(0, error.status)
                    self.assertEqual("cnp_random_permutation", error.function)
                    self.assertEqual(before, runtime.retained_bytes)

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
