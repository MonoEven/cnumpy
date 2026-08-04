from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np

from compat.cnumpy_ctypes import CnumpyError, CnumpyRuntime
from compat.oracle import assert_array_equivalent


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"


class CnumpyRuntimeProtocolTests(unittest.TestCase):
    def test_reference_versions_are_pinned(self) -> None:
        self.assertEqual("1.25.0", np.__version__)

    def test_round_trips_float64_edge_values(self) -> None:
        expected = np.array(
            [[0.0, -0.0, np.nan], [np.inf, -np.inf, 3.25]],
            dtype=np.float64,
        )
        with CnumpyRuntime(DLL) as runtime:
            with runtime.from_numpy(expected) as actual:
                assert_array_equivalent(self, actual, expected)
                values = actual.values()
                self.assertEqual(-1.0, math.copysign(1.0, values[1]))

    def test_round_trips_int64_without_float_conversion(self) -> None:
        expected = np.array(
            [-(2**63) + 1, -(2**53) - 1, 2**53 + 1, 2**63 - 1],
            dtype=np.int64,
        )
        with CnumpyRuntime(DLL) as runtime:
            with runtime.from_numpy(expected) as actual:
                assert_array_equivalent(self, actual, expected)

    def test_supports_scalar_and_empty_shapes(self) -> None:
        cases = (
            np.array(-0.0, dtype=np.float64),
            np.empty((2, 0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )
        with CnumpyRuntime(DLL) as runtime:
            for expected in cases:
                with self.subTest(shape=expected.shape, dtype=str(expected.dtype)):
                    with runtime.from_numpy(expected) as actual:
                        assert_array_equivalent(self, actual, expected)

    def test_reads_logical_values_and_strides_from_transposed_view(self) -> None:
        expected_source = np.arange(12, dtype=np.float64).reshape(3, 4)
        with CnumpyRuntime(DLL) as runtime:
            with runtime.from_numpy(expected_source) as source:
                with runtime.transpose(source) as actual:
                    expected = expected_source.T
                    assert_array_equivalent(
                        self,
                        actual,
                        expected,
                        compare_strides=True,
                    )
                    self.assertFalse(actual.c_contiguous)
                    self.assertTrue(actual.f_contiguous)

    def test_null_native_result_raises_complete_error(self) -> None:
        expected = np.arange(6, dtype=np.float64).reshape(2, 3)
        with CnumpyRuntime(DLL) as runtime:
            with runtime.from_numpy(expected) as source:
                with self.assertRaises(CnumpyError) as raised:
                    runtime.transpose(source, (0, 0))
        self.assertEqual(-5, raised.exception.status)
        self.assertEqual("cnp_transpose", raised.exception.function)
        self.assertIn("Duplicate axis", raised.exception.message)

    def test_each_runtime_context_returns_to_its_memory_baseline(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(np.arange(64, dtype=np.float64)):
                self.assertGreater(runtime.retained_bytes, baseline)
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
