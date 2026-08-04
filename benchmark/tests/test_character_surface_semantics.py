from __future__ import annotations

import ctypes
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"

CNP_INT = 6
CNP_OBJECT = 18
CNP_ERR_BROADCAST = -7
CNP_ERR_NOT_IMPLEMENTED = -12
CNP_ERR_VALUE = -13


def _encoded_inputs(values: list[bytes]) -> ctypes.Array[ctypes.c_char_p]:
    return (ctypes.c_char_p * len(values))(*values)


class CharacterSurfaceSemanticsTests(unittest.TestCase):
    def _bind(self, runtime: CnumpyRuntime) -> None:
        dll = runtime.dll
        string_array = ctypes.POINTER(ctypes.c_char_p)
        repeat_array = ctypes.POINTER(ctypes.c_int64)

        dll.cnp_char_add.argtypes = [
            string_array,
            ctypes.c_int64,
            string_array,
            ctypes.c_int64,
        ]
        dll.cnp_char_add.restype = ctypes.c_void_p
        dll.cnp_char_multiply.argtypes = [
            string_array,
            ctypes.c_int64,
            repeat_array,
            ctypes.c_int64,
        ]
        dll.cnp_char_multiply.restype = ctypes.c_void_p
        for name in ("upper", "lower", "strlen"):
            function = getattr(dll, f"cnp_char_{name}")
            function.argtypes = [string_array, ctypes.c_int64]
            function.restype = ctypes.c_void_p
        for name in ("strip", "lstrip", "rstrip", "count", "find"):
            function = getattr(dll, f"cnp_char_{name}")
            function.argtypes = [
                string_array,
                ctypes.c_int64,
                ctypes.c_char_p,
            ]
            function.restype = ctypes.c_void_p
        for name in ("center", "ljust", "rjust"):
            function = getattr(dll, f"cnp_char_{name}")
            function.argtypes = [
                string_array,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_char,
            ]
            function.restype = ctypes.c_void_p
        dll.cnp_char_zfill.argtypes = [
            string_array,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        dll.cnp_char_zfill.restype = ctypes.c_void_p
        dll.cnp_char_replace.argtypes = [
            string_array,
            ctypes.c_int64,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int64,
        ]
        dll.cnp_char_replace.restype = ctypes.c_void_p
        dll.cnp_char_split.argtypes = [
            string_array,
            ctypes.c_int64,
            ctypes.c_char_p,
        ]
        dll.cnp_char_split.restype = ctypes.c_void_p
        dll.cnp_char_join.argtypes = [
            string_array,
            ctypes.c_int64,
            ctypes.c_char_p,
        ]
        dll.cnp_char_join.restype = ctypes.c_void_p
        dll.cnp_char_join_v2.argtypes = [
            string_array,
            ctypes.c_int64,
            ctypes.c_char_p,
        ]
        dll.cnp_char_join_v2.restype = ctypes.c_void_p
        dll.cnp_char_free_result.argtypes = [ctypes.c_void_p]
        dll.cnp_char_free_result.restype = None
        dll.cnp_char_free_string.argtypes = [ctypes.c_void_p]
        dll.cnp_char_free_string.restype = None

    def _require_result(
        self, runtime: CnumpyRuntime, pointer: int | None, function: str
    ) -> int:
        if not pointer:
            self.fail(str(runtime.native_error(function)))
        return int(pointer)

    def _read_strings(
        self, runtime: CnumpyRuntime, pointer: int
    ) -> list[bytes]:
        self.assertEqual(CNP_OBJECT, runtime.dll.cnp_array_dtype_num(pointer))
        size = int(runtime.dll.cnp_array_size(pointer))
        values: list[bytes] = []
        for index in range(size):
            coordinate = (ctypes.c_int64 * 1)(index)
            slot = runtime.dll.cnp_array_at(pointer, coordinate)
            self.assertTrue(slot, runtime.error_state())
            value_pointer = ctypes.cast(
                slot, ctypes.POINTER(ctypes.c_void_p)
            )[0]
            self.assertTrue(value_pointer, f"null string at index {index}")
            values.append(ctypes.string_at(value_pointer))
        return values

    def _read_ints(self, runtime: CnumpyRuntime, pointer: int) -> list[int]:
        self.assertEqual(CNP_INT, runtime.dll.cnp_array_dtype_num(pointer))
        size = int(runtime.dll.cnp_array_size(pointer))
        values: list[int] = []
        for index in range(size):
            coordinate = (ctypes.c_int64 * 1)(index)
            address = runtime.dll.cnp_array_at(pointer, coordinate)
            self.assertTrue(address, runtime.error_state())
            values.append(
                int(ctypes.cast(address, ctypes.POINTER(ctypes.c_int32))[0])
            )
        return values

    def _assert_native_error(
        self,
        runtime: CnumpyRuntime,
        pointer: int | None,
        function: str,
        status: int,
        message_fragment: str,
    ) -> None:
        self.assertFalse(pointer)
        error = runtime.error_state()
        self.assertEqual(status, error.status)
        self.assertEqual(function, error.function)
        self.assertIn(message_fragment, error.message.lower())

    def test_string_results_match_numpy_125_byte_string_values(self) -> None:
        values = [b" Ab ", b"xy", b"", b"-3"]
        source = np.asarray(values, dtype=np.bytes_)
        inputs = _encoded_inputs(values)
        right_values = [b"!", b"?"]
        right = _encoded_inputs(right_values)
        repeats_values = [2]
        repeats = (ctypes.c_int64 * 1)(*repeats_values)

        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            cases = [
                (
                    "cnp_char_add",
                    runtime.dll.cnp_char_add(inputs, len(values), right, 1),
                    np.char.add(source, np.asarray([b"!"], dtype=np.bytes_)),
                ),
                (
                    "cnp_char_multiply",
                    runtime.dll.cnp_char_multiply(
                        inputs, len(values), repeats, len(repeats_values)
                    ),
                    np.char.multiply(source, np.asarray(repeats_values)),
                ),
                ("cnp_char_upper", runtime.dll.cnp_char_upper(inputs, 4), np.char.upper(source)),
                ("cnp_char_lower", runtime.dll.cnp_char_lower(inputs, 4), np.char.lower(source)),
                ("cnp_char_strip", runtime.dll.cnp_char_strip(inputs, 4, None), np.char.strip(source)),
                ("cnp_char_lstrip", runtime.dll.cnp_char_lstrip(inputs, 4, b" A"), np.char.lstrip(source, b" A")),
                ("cnp_char_rstrip", runtime.dll.cnp_char_rstrip(inputs, 4, b" b"), np.char.rstrip(source, b" b")),
                ("cnp_char_center", runtime.dll.cnp_char_center(inputs, 4, 3, b"."), np.char.center(source, 3, b".")),
                ("cnp_char_center", runtime.dll.cnp_char_center(inputs, 4, 0, b"."), np.char.center(source, 0, b".")),
                ("cnp_char_ljust", runtime.dll.cnp_char_ljust(inputs, 4, 2, b"."), np.char.ljust(source, 2, b".")),
                ("cnp_char_ljust", runtime.dll.cnp_char_ljust(inputs, 4, 0, b"."), np.char.ljust(source, 0, b".")),
                ("cnp_char_rjust", runtime.dll.cnp_char_rjust(inputs, 4, 2, b"."), np.char.rjust(source, 2, b".")),
                ("cnp_char_rjust", runtime.dll.cnp_char_rjust(inputs, 4, 0, b"."), np.char.rjust(source, 0, b".")),
                ("cnp_char_zfill", runtime.dll.cnp_char_zfill(inputs, 4, 2), np.char.zfill(source, 2)),
                ("cnp_char_zfill", runtime.dll.cnp_char_zfill(inputs, 4, 0), np.char.zfill(source, 0)),
                (
                    "cnp_char_replace",
                    runtime.dll.cnp_char_replace(inputs, 4, b"", b"X", 2),
                    np.char.replace(source, b"", b"X", 2),
                ),
            ]
            try:
                for function, pointer, expected in cases:
                    with self.subTest(function=function):
                        result = self._require_result(runtime, pointer, function)
                        self.assertEqual(expected.tolist(), self._read_strings(runtime, result))
            finally:
                for _, pointer, _ in cases:
                    if pointer:
                        runtime.dll.cnp_array_decref(pointer)

            join_values = [b"ab", b"", b"XYZ"]
            join_inputs = _encoded_inputs(join_values)
            joined = self._require_result(
                runtime,
                runtime.dll.cnp_char_join_v2(join_inputs, 3, b"-"),
                "cnp_char_join_v2",
            )
            try:
                expected_join = [
                    value.encode("utf-8")
                    for value in np.char.join(
                        "-",
                        np.asarray(
                            [item.decode("ascii") for item in join_values],
                            dtype=np.str_,
                        ),
                    ).tolist()
                ]
                self.assertEqual(expected_join, self._read_strings(runtime, joined))
            finally:
                runtime.dll.cnp_array_decref(joined)

            empty = self._require_result(
                runtime,
                runtime.dll.cnp_char_upper(None, 0),
                "cnp_char_upper",
            )
            try:
                self.assertEqual(0, runtime.dll.cnp_array_size(empty))
                self.assertEqual([], self._read_strings(runtime, empty))
            finally:
                runtime.dll.cnp_array_decref(empty)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_integer_results_match_numpy_and_use_windows_int_dtype(self) -> None:
        values = [b"banana", b"", b"abcabc"]
        source = np.asarray(values, dtype=np.bytes_)
        inputs = _encoded_inputs(values)

        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            cases = [
                ("cnp_char_strlen", runtime.dll.cnp_char_strlen(inputs, 3), np.char.str_len(source)),
                ("cnp_char_count", runtime.dll.cnp_char_count(inputs, 3, b"an"), np.char.count(source, b"an")),
                ("cnp_char_find", runtime.dll.cnp_char_find(inputs, 3, b"na"), np.char.find(source, b"na")),
            ]
            try:
                for function, pointer, expected in cases:
                    with self.subTest(function=function):
                        result = self._require_result(runtime, pointer, function)
                        self.assertEqual(expected.tolist(), self._read_ints(runtime, result))
            finally:
                for _, pointer, _ in cases:
                    if pointer:
                        runtime.dll.cnp_array_decref(pointer)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_broadcast_and_parameter_failures_are_atomic_and_labeled(self) -> None:
        left = _encoded_inputs([b"a", b"b"])
        right = _encoded_inputs([b"1", b"2", b"3"])
        repeats = (ctypes.c_int64 * 3)(1, 2, 3)

        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            runtime.dll.cnp_clear_error()
            self._assert_native_error(
                runtime,
                runtime.dll.cnp_char_add(left, 2, right, 3),
                "cnp_char_add",
                CNP_ERR_BROADCAST,
                "broadcast",
            )
            runtime.dll.cnp_clear_error()
            self._assert_native_error(
                runtime,
                runtime.dll.cnp_char_multiply(left, 2, repeats, 3),
                "cnp_char_multiply",
                CNP_ERR_BROADCAST,
                "broadcast",
            )
            for symbol in ("center", "ljust", "rjust"):
                runtime.dll.cnp_clear_error()
                pointer = getattr(runtime.dll, f"cnp_char_{symbol}")(
                    left, 2, -1, b"."
                )
                self._assert_native_error(
                    runtime,
                    pointer,
                    f"cnp_char_{symbol}",
                    CNP_ERR_VALUE,
                    "width",
                )
            runtime.dll.cnp_clear_error()
            self._assert_native_error(
                runtime,
                runtime.dll.cnp_char_zfill(left, 2, -1),
                "cnp_char_zfill",
                CNP_ERR_VALUE,
                "width",
            )
            runtime.dll.cnp_clear_error()
            self._assert_native_error(
                runtime,
                runtime.dll.cnp_char_upper(None, -1),
                "cnp_char_upper",
                CNP_ERR_VALUE,
                "count",
            )
            runtime.dll.cnp_clear_error()
            self._assert_native_error(
                runtime,
                runtime.dll.cnp_char_split(left, 2, b","),
                "cnp_char_split",
                CNP_ERR_NOT_IMPLEMENTED,
                "split_v2",
            )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_empty_substrings_and_null_entries_fail_or_finish_explicitly(self) -> None:
        script = textwrap.dedent(
            f"""
            import ctypes
            dll = ctypes.CDLL({str(DLL)!r})
            class ErrorState(ctypes.Structure):
                _fields_ = [
                    ("status", ctypes.c_int),
                    ("message", ctypes.c_char * 256),
                    ("function", ctypes.c_char * 64),
                ]
            dll.cnp_init.restype = ctypes.c_int
            dll.cnp_get_error.argtypes = [ctypes.POINTER(ErrorState)]
            dll.cnp_get_error.restype = ctypes.c_int
            dll.cnp_get_allocated_memory.restype = ctypes.c_size_t
            dll.cnp_array_decref.argtypes = [ctypes.c_void_p]
            strings = ctypes.POINTER(ctypes.c_char_p)
            dll.cnp_char_count.argtypes = [strings, ctypes.c_int64, ctypes.c_char_p]
            dll.cnp_char_count.restype = ctypes.c_void_p
            dll.cnp_char_replace.argtypes = [strings, ctypes.c_int64, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int64]
            dll.cnp_char_replace.restype = ctypes.c_void_p
            dll.cnp_char_upper.argtypes = [strings, ctypes.c_int64]
            dll.cnp_char_upper.restype = ctypes.c_void_p
            assert dll.cnp_init() == 0
            baseline = dll.cnp_get_allocated_memory()
            inputs = (ctypes.c_char_p * 2)(b"ab", b"")
            count = dll.cnp_char_count(inputs, 2, b"")
            replace = dll.cnp_char_replace(inputs, 2, b"", b"X", -1)
            assert count and replace
            dll.cnp_array_decref(count)
            dll.cnp_array_decref(replace)
            invalid = (ctypes.c_char_p * 1)(None)
            dll.cnp_clear_error()
            assert not dll.cnp_char_upper(invalid, 1)
            state = ErrorState()
            assert dll.cnp_get_error(ctypes.byref(state)) == {CNP_ERR_VALUE}
            assert bytes(state.function).split(b"\\0", 1)[0] == b"cnp_char_upper"
            assert dll.cnp_get_allocated_memory() == baseline
            dll.cnp_cleanup()
            """
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        self.assertEqual(
            0,
            completed.returncode,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_legacy_join_and_both_release_contracts_restore_memory(self) -> None:
        values = [b"alpha", b"beta", b""]
        inputs = _encoded_inputs(values)

        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for release_symbol in ("cnp_array_decref", "cnp_char_free_result"):
                for _ in range(64):
                    pointer = self._require_result(
                        runtime,
                        runtime.dll.cnp_char_upper(inputs, len(values)),
                        "cnp_char_upper",
                    )
                    getattr(runtime.dll, release_symbol)(pointer)
                self.assertEqual(baseline, runtime.retained_bytes)

            for sequence in (values, []):
                encoded = _encoded_inputs(sequence) if sequence else None
                pointer = self._require_result(
                    runtime,
                    runtime.dll.cnp_char_join(
                        encoded, len(sequence), b"|"
                    ),
                    "cnp_char_join",
                )
                try:
                    self.assertEqual(b"|".join(sequence), ctypes.string_at(pointer))
                finally:
                    runtime.dll.cnp_char_free_string(pointer)
            runtime.dll.cnp_char_free_string(None)
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
