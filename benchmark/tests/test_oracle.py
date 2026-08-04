from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime
from compat.oracle import assert_array_equivalent


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"


class ArrayEquivalentTests(unittest.TestCase):
    def test_tolerance_comparison_is_supported_by_numpy_1_25(self) -> None:
        source_value = np.array([1.0, 2.0], dtype=np.float64)
        expected = np.nextafter(source_value, np.inf)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(source_value) as actual:
                try:
                    assert_array_equivalent(
                        self,
                        actual,
                        expected,
                        rtol=2 * np.finfo(np.float64).eps,
                    )
                except TypeError as error:
                    self.fail(f"NumPy 1.25 rejected tolerance comparison: {error}")
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
