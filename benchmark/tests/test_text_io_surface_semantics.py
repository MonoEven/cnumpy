from __future__ import annotations

import ctypes
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime
from compat.oracle import assert_array_equivalent


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"
CNP_LONG = 8
CNP_DOUBLE = 13


class TextIoSurfaceSemanticsTests(unittest.TestCase):
    @staticmethod
    def _bind(runtime: CnumpyRuntime) -> None:
        runtime.dll.cnp_array_to_string.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        runtime.dll.cnp_array_to_string.restype = ctypes.c_void_p
        runtime.dll.cnp_array2string.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int64,
        ]
        runtime.dll.cnp_array2string.restype = ctypes.c_int
        runtime.dll.cnp_array_to_csv.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
        ]
        runtime.dll.cnp_array_to_csv.restype = ctypes.c_int
        runtime.dll.cnp_savetxt.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        runtime.dll.cnp_savetxt.restype = ctypes.c_int
        runtime.dll.cnp_loadtxt.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        runtime.dll.cnp_loadtxt.restype = ctypes.c_void_p
        runtime.dll.cnp_genfromtxt.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        runtime.dll.cnp_genfromtxt.restype = ctypes.c_void_p
        runtime.dll.cnp_disp.argtypes = [ctypes.c_char_p]
        runtime.dll.cnp_disp.restype = ctypes.c_int
        runtime.dll.cnp_char_free_string.argtypes = [ctypes.c_void_p]
        runtime.dll.cnp_char_free_string.restype = None

    @staticmethod
    def _owned_text(
        runtime: CnumpyRuntime,
        pointer: int | None,
        function_name: str,
    ) -> str:
        if not pointer:
            raise runtime.native_error(function_name)
        try:
            return ctypes.string_at(pointer).decode("utf-8")
        finally:
            runtime.dll.cnp_char_free_string(pointer)

    @staticmethod
    def _array2string(runtime: CnumpyRuntime, pointer: int) -> str:
        buffer = ctypes.create_string_buffer(4096)
        runtime.dll.cnp_clear_error()
        length = runtime.dll.cnp_array2string(
            pointer, buffer, len(buffer)
        )
        if length < 0:
            raise runtime.native_error("cnp_array2string")
        return buffer.value.decode("utf-8")

    def test_array_string_csv_and_lifetime_contracts(self) -> None:
        integer_values = np.asarray([[1, 2], [3, 4]], dtype=np.int64)
        float_values = np.asarray(
            [[1.25, -2.5], [3.0, 4.5]], dtype=np.float64
        )
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            with runtime.from_numpy(integer_values) as integers:
                self.assertEqual(
                    np.array2string(integer_values, separator=", "),
                    self._array2string(runtime, integers.pointer),
                )

            with runtime.from_numpy(float_values) as values:
                runtime.dll.cnp_clear_error()
                formatted = self._owned_text(
                    runtime,
                    runtime.dll.cnp_array_to_string(
                        values.pointer, b"%.2f"
                    ),
                    "cnp_array_to_string",
                )
                expected = np.array2string(
                    float_values,
                    separator=", ",
                    formatter={"float_kind": lambda value: f"{value:.2f}"},
                )
                self.assertEqual(expected, formatted)

                csv_buffer = ctypes.create_string_buffer(4096)
                runtime.dll.cnp_clear_error()
                self.assertEqual(
                    0,
                    runtime.dll.cnp_array_to_csv(
                        values.pointer,
                        csv_buffer,
                        len(csv_buffer),
                        b";",
                    ),
                    runtime.error_state(),
                )
                expected_csv = io.StringIO()
                np.savetxt(
                    expected_csv,
                    float_values,
                    delimiter=";",
                    fmt="%g",
                )
                self.assertEqual(
                    expected_csv.getvalue().rstrip("\n"),
                    csv_buffer.value.decode("utf-8"),
                )

                retained_with_source = runtime.retained_bytes
                for _ in range(128):
                    runtime.dll.cnp_clear_error()
                    text = self._owned_text(
                        runtime,
                        runtime.dll.cnp_array_to_string(
                            values.pointer, b"%g"
                        ),
                        "cnp_array_to_string",
                    )
                    self.assertTrue(text)
                self.assertEqual(
                    retained_with_source, runtime.retained_bytes
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_array2string_threshold_strides_and_buffer_errors(self) -> None:
        values = np.arange(8, dtype=np.int64)
        matrix = np.arange(12, dtype=np.int64).reshape(3, 4).T
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            runtime.dll.cnp_set_printoptions(8, 4, 1, 75, 0)
            with runtime.from_numpy(values) as source:
                self.assertEqual(
                    np.array2string(
                        values, separator=", ", threshold=4, edgeitems=1
                    ),
                    self._array2string(runtime, source.pointer),
                )
            runtime.dll.cnp_set_printoptions(8, 1000, 3, 75, 0)
            with runtime.from_numpy(matrix) as source:
                self.assertEqual(
                    np.array2string(
                        matrix,
                        separator=", ",
                        formatter={"int_kind": lambda value: str(value)},
                    ),
                    self._array2string(runtime, source.pointer),
                )
                buffer = ctypes.create_string_buffer(32)
                buffer.value = b"sentinel"
                runtime.dll.cnp_clear_error()
                self.assertLess(
                    runtime.dll.cnp_array2string(
                        source.pointer, buffer, len(buffer)
                    ),
                    0,
                )
                error = runtime.error_state()
                self.assertEqual("cnp_array2string", error.function)
                self.assertEqual(b"", buffer.value)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_savetxt_and_loadtxt_match_numpy_shapes_values_and_errors(
        self,
    ) -> None:
        values = np.asarray(
            [[1.25, -2.5, 3.75], [4.0, 5.125, -6.5]],
            dtype=np.float64,
        )[:, ::-1]
        with tempfile.TemporaryDirectory() as directory, CnumpyRuntime(
            DLL
        ) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            path = Path(directory) / "values.csv"
            with runtime.from_numpy(values) as source:
                runtime.dll.cnp_clear_error()
                self.assertEqual(
                    0,
                    runtime.dll.cnp_savetxt(
                        str(path).encode(),
                        source.pointer,
                        b",",
                        b"%.6g",
                    ),
                    runtime.error_state(),
                )
            expected_file = io.StringIO()
            np.savetxt(expected_file, values, delimiter=",", fmt="%.6g")
            self.assertEqual(expected_file.getvalue(), path.read_text())

            loaded_pointer = runtime.dll.cnp_loadtxt(
                str(path).encode(), b",", CNP_DOUBLE
            )
            with runtime._owned_result(
                loaded_pointer, "cnp_loadtxt"
            ) as loaded:
                assert_array_equivalent(
                    self,
                    loaded,
                    np.loadtxt(path, delimiter=",", dtype=np.float64),
                )

            shape_cases = {
                "single-row.csv": "1,2,3\n",
                "single-column.csv": "1\n2\n3\n",
                "empty.csv": "",
            }
            for filename, content in shape_cases.items():
                case_path = Path(directory) / filename
                case_path.write_text(content)
                with runtime._owned_result(
                    runtime.dll.cnp_loadtxt(
                        str(case_path).encode(), b",", CNP_DOUBLE
                    ),
                    "cnp_loadtxt",
                ) as actual:
                    with np.testing.suppress_warnings() as warnings:
                        warnings.filter(UserWarning)
                        expected = np.loadtxt(
                            case_path, delimiter=",", dtype=np.float64
                        )
                    assert_array_equivalent(self, actual, expected)

            integer_path = Path(directory) / "integers.csv"
            integer_path.write_text("1,2\n3,4\n")
            with runtime._owned_result(
                runtime.dll.cnp_loadtxt(
                    str(integer_path).encode(), b",", CNP_LONG
                ),
                "cnp_loadtxt",
            ) as integers:
                assert_array_equivalent(
                    self,
                    integers,
                    np.loadtxt(integer_path, delimiter=",", dtype=np.int64),
                )

            malformed_path = Path(directory) / "malformed.csv"
            malformed_path.write_text("1,2\n3\n")
            runtime.dll.cnp_clear_error()
            self.assertFalse(
                runtime.dll.cnp_loadtxt(
                    str(malformed_path).encode(), b",", CNP_DOUBLE
                )
            )
            self.assertEqual(
                "cnp_loadtxt", runtime.error_state().function
            )

            invalid_path = Path(directory) / "invalid.csv"
            invalid_path.write_text("1,bad\n")
            runtime.dll.cnp_clear_error()
            self.assertFalse(
                runtime.dll.cnp_loadtxt(
                    str(invalid_path).encode(), b",", CNP_DOUBLE
                )
            )
            self.assertEqual(
                "cnp_loadtxt", runtime.error_state().function
            )

            rank_path = Path(directory) / "rank-error.csv"
            rank_three = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
            with runtime.from_numpy(rank_three) as source:
                runtime.dll.cnp_clear_error()
                self.assertNotEqual(
                    0,
                    runtime.dll.cnp_savetxt(
                        str(rank_path).encode(),
                        source.pointer,
                        b",",
                        b"%g",
                    ),
                )
                self.assertEqual(
                    "cnp_savetxt", runtime.error_state().function
                )
            self.assertFalse(rank_path.exists())
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_genfromtxt_matches_missing_dtype_row_limit_and_errors(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory, CnumpyRuntime(
            DLL
        ) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            path = Path(directory) / "missing.csv"
            path.write_text(
                "first,second,third\n"
                "1,,3\n"
                "4,bad,6\n"
                "7,8,9\n"
            )
            with runtime._owned_result(
                runtime.dll.cnp_genfromtxt(
                    str(path).encode(), b",", 1, 2, CNP_DOUBLE
                ),
                "cnp_genfromtxt",
            ) as actual:
                expected = np.genfromtxt(
                    path,
                    delimiter=",",
                    skip_header=1,
                    max_rows=2,
                    dtype=np.float64,
                )
                assert_array_equivalent(self, actual, expected)

            integer_path = Path(directory) / "missing-integer.csv"
            integer_path.write_text("1,\n3,4\n")
            with runtime._owned_result(
                runtime.dll.cnp_genfromtxt(
                    str(integer_path).encode(), b",", 0, -1, CNP_LONG
                ),
                "cnp_genfromtxt",
            ) as actual:
                expected = np.genfromtxt(
                    integer_path, delimiter=",", dtype=np.int64
                )
                assert_array_equivalent(self, actual, expected)

            runtime.dll.cnp_clear_error()
            self.assertFalse(
                runtime.dll.cnp_genfromtxt(
                    str(path).encode(), b",", 1, 0, CNP_DOUBLE
                )
            )
            self.assertEqual(
                "cnp_genfromtxt", runtime.error_state().function
            )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_disp_and_array_print_have_real_stdout_contracts(self) -> None:
        script = textwrap.dedent(
            f"""
            import ctypes
            import numpy as np
            from compat.cnumpy_ctypes import CnumpyRuntime

            with CnumpyRuntime(r"{DLL}") as runtime:
                baseline = runtime.retained_bytes
                source = runtime.from_numpy(
                    np.asarray([[1, 2], [3, 4]], dtype=np.int64)
                )
                runtime.dll.cnp_array_print.argtypes = [ctypes.c_void_p]
                runtime.dll.cnp_array_print.restype = None
                runtime.dll.cnp_array_print(source.pointer)
                source.close()
                if runtime.retained_bytes != baseline:
                    raise RuntimeError("cnp_array_print retained native bytes")
            """
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        expected = np.array2string(
            np.asarray([[1, 2], [3, 4]], dtype=np.int64), separator=", "
        )
        self.assertEqual(expected + "\n", result.stdout)

        disp_script = textwrap.dedent(
            f"""
            import ctypes
            from compat.cnumpy_ctypes import CnumpyRuntime

            with CnumpyRuntime(r"{DLL}") as runtime:
                runtime.dll.cnp_disp.argtypes = [ctypes.c_char_p]
                runtime.dll.cnp_disp.restype = ctypes.c_int
                status = runtime.dll.cnp_disp(b"visible message")
                if status != 0:
                    raise runtime.native_error("cnp_disp")
            """
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", disp_script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual("visible message\n", result.stdout)

        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            runtime.dll.cnp_clear_error()
            self.assertNotEqual(0, runtime.dll.cnp_disp(None))
            self.assertEqual("cnp_disp", runtime.error_state().function)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_output_buffer_failures_are_labeled_atomic_and_nonretaining(
        self,
    ) -> None:
        values = np.arange(100, dtype=np.float64).reshape(10, 10)
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            with runtime.from_numpy(values) as source:
                buffer = ctypes.create_string_buffer(64)
                buffer.value = b"sentinel"
                runtime.dll.cnp_clear_error()
                self.assertNotEqual(
                    0,
                    runtime.dll.cnp_array_to_csv(
                        source.pointer, buffer, len(buffer), b","
                    ),
                )
                self.assertEqual(
                    "cnp_array_to_csv", runtime.error_state().function
                )
                self.assertEqual(b"", buffer.value)
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
