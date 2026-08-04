from __future__ import annotations

import ctypes
import io
import math
import unittest
from pathlib import Path

import numpy as np
from scipy import special as scipy_special

from compat.cnumpy_ctypes import CnumpyError, CnumpyRuntime
from compat.oracle import assert_array_equivalent


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"


class _CnpSlice(ctypes.Structure):
    _fields_ = [
        ("start", ctypes.c_int64),
        ("stop", ctypes.c_int64),
        ("step", ctypes.c_int64),
        ("has_start", ctypes.c_bool),
        ("has_stop", ctypes.c_bool),
        ("has_step", ctypes.c_bool),
    ]


class StringSplitSemanticsTests(unittest.TestCase):
    def test_split_v2_matches_whitespace_and_maxsplit_rules(self) -> None:
        values = ["  alpha\tbeta  gamma ", "   ", "one"]
        expected = np.char.split(
            np.asarray(values, dtype=np.str_), sep=None, maxsplit=1
        ).tolist()
        encoded = [value.encode("utf-8") for value in values]
        inputs = (ctypes.c_char_p * len(encoded))(*encoded)

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            split = runtime.dll.cnp_char_split_v2
            split.argtypes = [
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.c_int64,
                ctypes.c_char_p,
                ctypes.c_int64,
            ]
            split.restype = ctypes.c_void_p
            outer_count = runtime.dll.cnp_string_list_outer_count
            outer_count.argtypes = [ctypes.c_void_p]
            outer_count.restype = ctypes.c_int64
            token_count = runtime.dll.cnp_string_list_token_count
            token_count.argtypes = [ctypes.c_void_p, ctypes.c_int64]
            token_count.restype = ctypes.c_int64
            get_token = runtime.dll.cnp_string_list_get
            get_token.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
            ]
            get_token.restype = ctypes.c_char_p
            free_result = runtime.dll.cnp_string_list_free
            free_result.argtypes = [ctypes.c_void_p]
            free_result.restype = None

            runtime.dll.cnp_clear_error()
            result = split(inputs, len(values), None, 1)
            self.assertTrue(result, runtime.error_state())
            try:
                actual = [
                    [
                        get_token(result, row, token).decode("utf-8")
                        for token in range(token_count(result, row))
                    ]
                    for row in range(outer_count(result))
                ]
                self.assertEqual(expected, actual)
            finally:
                free_result(result)

            self.assertEqual(baseline, runtime.retained_bytes)

    def test_split_v2_returns_every_token_and_owns_nested_strings(
        self,
    ) -> None:
        values = ["alpha,beta,,gamma", "", ",leading,trailing,"]
        expected = np.char.split(
            np.asarray(values, dtype=np.str_), sep=","
        ).tolist()
        encoded = [value.encode("utf-8") for value in values]
        inputs = (ctypes.c_char_p * len(encoded))(*encoded)

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            split = runtime.dll.cnp_char_split_v2
            split.argtypes = [
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.c_int64,
                ctypes.c_char_p,
                ctypes.c_int64,
            ]
            split.restype = ctypes.c_void_p
            outer_count = runtime.dll.cnp_string_list_outer_count
            outer_count.argtypes = [ctypes.c_void_p]
            outer_count.restype = ctypes.c_int64
            token_count = runtime.dll.cnp_string_list_token_count
            token_count.argtypes = [ctypes.c_void_p, ctypes.c_int64]
            token_count.restype = ctypes.c_int64
            get_token = runtime.dll.cnp_string_list_get
            get_token.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
            ]
            get_token.restype = ctypes.c_char_p
            free_result = runtime.dll.cnp_string_list_free
            free_result.argtypes = [ctypes.c_void_p]
            free_result.restype = None

            runtime.dll.cnp_clear_error()
            result = split(inputs, len(values), b",", -1)
            self.assertTrue(result, runtime.error_state())
            try:
                self.assertEqual(len(expected), outer_count(result))
                actual: list[list[str]] = []
                for row in range(outer_count(result)):
                    count = token_count(result, row)
                    self.assertGreaterEqual(count, 0, runtime.error_state())
                    tokens = []
                    for column in range(count):
                        token = get_token(result, row, column)
                        self.assertIsNotNone(token, runtime.error_state())
                        tokens.append(token.decode("utf-8"))
                    actual.append(tokens)
                self.assertEqual(expected, actual)
            finally:
                free_result(result)

            self.assertEqual(baseline, runtime.retained_bytes)


class StringFunctionSemanticsTests(unittest.TestCase):
    CALLBACK = ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    )

    @staticmethod
    def _read_owned_string(runtime: CnumpyRuntime, pointer: int) -> str:
        if not pointer:
            raise AssertionError(runtime.error_state())
        try:
            return ctypes.string_at(pointer).decode("utf-8")
        finally:
            runtime.dll.cnp_char_free_string(pointer)

    def _bind(self, runtime: CnumpyRuntime) -> None:
        runtime.dll.cnp_set_string_function.argtypes = [ctypes.c_void_p]
        runtime.dll.cnp_set_string_function.restype = ctypes.c_int
        runtime.dll.cnp_set_string_function_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_bool,
        ]
        runtime.dll.cnp_set_string_function_v2.restype = ctypes.c_int
        runtime.dll.cnp_array_string_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_bool,
        ]
        runtime.dll.cnp_array_string_v2.restype = ctypes.c_void_p
        runtime.dll.cnp_char_free_string.argtypes = [ctypes.c_void_p]
        runtime.dll.cnp_char_free_string.restype = None

    def test_repr_and_str_callbacks_are_independent_and_resettable(self) -> None:
        expected = np.asarray([1.25, -2.5], dtype=np.float64)
        np.set_string_function(lambda array: "<custom repr>", repr=True)
        try:
            expected_repr = repr(expected)
            expected_default_str = str(expected)
        finally:
            np.set_string_function(None, repr=True)
        np.set_string_function(lambda array: "<custom str>", repr=False)
        try:
            expected_str = str(expected)
        finally:
            np.set_string_function(None, repr=False)
        self.assertNotEqual(expected_repr, expected_default_str)
        repr_storage = ctypes.create_string_buffer(b"<custom repr>")
        str_storage = ctypes.create_string_buffer(b"<custom str>")
        repr_userdata = ctypes.c_int64(17)
        str_userdata = ctypes.c_int64(29)
        calls: list[tuple[str, int, int | None]] = []

        @self.CALLBACK
        def repr_callback(array_pointer: int, userdata: int) -> int:
            calls.append(("repr", array_pointer, userdata))
            return ctypes.addressof(repr_storage)

        @self.CALLBACK
        def str_callback(array_pointer: int, userdata: int) -> int:
            calls.append(("str", array_pointer, userdata))
            return ctypes.addressof(str_storage)

        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            source = runtime.from_numpy(expected)

            status = runtime.dll.cnp_set_string_function_v2(
                repr_callback, ctypes.byref(repr_userdata), True
            )
            self.assertEqual(0, status, runtime.error_state())
            self.assertEqual(
                expected_repr,
                self._read_owned_string(
                    runtime,
                    runtime.dll.cnp_array_string_v2(source.pointer, True),
                ),
            )
            default_str = self._read_owned_string(
                runtime,
                runtime.dll.cnp_array_string_v2(source.pointer, False),
            )
            self.assertNotEqual(expected_repr, default_str)

            status = runtime.dll.cnp_set_string_function_v2(
                str_callback, ctypes.byref(str_userdata), False
            )
            self.assertEqual(0, status, runtime.error_state())
            self.assertEqual(
                expected_str,
                self._read_owned_string(
                    runtime,
                    runtime.dll.cnp_array_string_v2(source.pointer, False),
                ),
            )

            self.assertEqual(
                0,
                runtime.dll.cnp_set_string_function_v2(None, None, True),
                runtime.error_state(),
            )
            reset_repr = self._read_owned_string(
                runtime,
                runtime.dll.cnp_array_string_v2(source.pointer, True),
            )
            self.assertNotEqual("<custom repr>", reset_repr)
            self.assertEqual(
                expected_str,
                self._read_owned_string(
                    runtime,
                    runtime.dll.cnp_array_string_v2(source.pointer, False),
                ),
            )

            legacy_status = runtime.dll.cnp_set_string_function(
                ctypes.cast(repr_callback, ctypes.c_void_p)
            )
            self.assertEqual(0, legacy_status, runtime.error_state())
            self.assertEqual(
                expected_repr,
                self._read_owned_string(
                    runtime,
                    runtime.dll.cnp_array_string_v2(source.pointer, True),
                ),
            )
            self.assertEqual(
                0,
                runtime.dll.cnp_set_string_function(None),
                runtime.error_state(),
            )
            self.assertEqual(
                0,
                runtime.dll.cnp_set_string_function_v2(None, None, False),
                runtime.error_state(),
            )

            source_pointer = source.pointer.value
            self.assertEqual(
                [
                    ("repr", source_pointer, ctypes.addressof(repr_userdata)),
                    ("str", source_pointer, ctypes.addressof(str_userdata)),
                    ("str", source_pointer, ctypes.addressof(str_userdata)),
                    ("repr", source_pointer, None),
                ],
                calls,
            )
            source.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_cleanup_removes_external_callback_registration(self) -> None:
        storage = ctypes.create_string_buffer(b"<registered>")
        calls = 0

        @self.CALLBACK
        def callback(array_pointer: int, userdata: int) -> int:
            nonlocal calls
            calls += 1
            return ctypes.addressof(storage)

        with CnumpyRuntime(DLL) as runtime, runtime.from_numpy(
            np.asarray([4.0], dtype=np.float64)
        ) as source:
            self._bind(runtime)
            self.assertEqual(
                0,
                runtime.dll.cnp_set_string_function_v2(
                    callback, None, False
                ),
                runtime.error_state(),
            )
            self.assertEqual(
                "<registered>",
                self._read_owned_string(
                    runtime,
                    runtime.dll.cnp_array_string_v2(source.pointer, False),
                ),
            )

        with CnumpyRuntime(DLL) as runtime, runtime.from_numpy(
            np.asarray([4.0], dtype=np.float64)
        ) as source:
            self._bind(runtime)
            self.assertNotEqual(
                "<registered>",
                self._read_owned_string(
                    runtime,
                    runtime.dll.cnp_array_string_v2(source.pointer, False),
                ),
            )
        self.assertEqual(1, calls)

    def test_null_callback_result_is_an_explicit_native_error(self) -> None:
        @self.CALLBACK
        def failing_callback(array_pointer: int, userdata: int) -> int:
            return 0

        with CnumpyRuntime(DLL) as runtime, runtime.from_numpy(
            np.asarray([3.0], dtype=np.float64)
        ) as source:
            self._bind(runtime)
            self.assertEqual(
                0,
                runtime.dll.cnp_set_string_function_v2(
                    failing_callback, None, False
                ),
                runtime.error_state(),
            )
            runtime.dll.cnp_clear_error()
            self.assertFalse(
                runtime.dll.cnp_array_string_v2(source.pointer, False)
            )
            state = runtime.error_state()
            self.assertNotEqual(0, state.status)
            self.assertEqual("cnp_array_string_v2", state.function)
            self.assertIn("callback returned null", state.message)
            self.assertEqual(
                0,
                runtime.dll.cnp_set_string_function_v2(None, None, False),
                runtime.error_state(),
            )


class FromStringSemanticsTests(unittest.TestCase):
    @staticmethod
    def _bind(runtime: CnumpyRuntime) -> None:
        runtime.dll.cnp_fromstring.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int,
        ]
        runtime.dll.cnp_fromstring.restype = ctypes.c_void_p
        runtime.dll.cnp_fromstring_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_char_p,
        ]
        runtime.dll.cnp_fromstring_v2.restype = ctypes.c_void_p

    def test_binary_legacy_matches_numpy_and_rejects_partial_elements(
        self,
    ) -> None:
        values = np.asarray([1, -2, 300, -400], dtype=np.int16)
        payload = values.tobytes()
        storage = ctypes.create_string_buffer(payload, len(payload))
        expected = np.fromstring(payload, dtype=np.int16, sep="")

        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            runtime.dll.cnp_clear_error()
            pointer = runtime.dll.cnp_fromstring(storage, len(payload), 4)
            with runtime._owned_result(pointer, "cnp_fromstring") as actual:
                assert_array_equivalent(self, actual, expected)

            runtime.dll.cnp_clear_error()
            self.assertFalse(
                runtime.dll.cnp_fromstring(storage, len(payload) - 1, 4)
            )
            state = runtime.error_state()
            self.assertEqual(-13, state.status)
            self.assertEqual("cnp_fromstring", state.function)
            self.assertIn("multiple of the dtype itemsize", state.message)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_text_v2_matches_numpy_separator_count_dtype_and_empty_rules(
        self,
    ) -> None:
        cases = [
            ("1, 2, -3.5e1, 4", np.float64, 13, -1, ","),
            ("1  2\t-3\n4", np.int16, 4, -1, " "),
            ("255|0|17|9", np.uint8, 3, 3, "|"),
            ("", np.float64, 13, -1, ","),
            ("5;6;7", np.int32, 6, 0, ";"),
        ]
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for text, dtype, dtype_number, count, separator in cases:
                with self.subTest(
                    text=text, dtype=np.dtype(dtype), count=count, sep=separator
                ):
                    expected = np.fromstring(
                        text, dtype=dtype, count=count, sep=separator
                    )
                    encoded = text.encode("ascii")
                    storage = ctypes.create_string_buffer(encoded)
                    runtime.dll.cnp_clear_error()
                    pointer = runtime.dll.cnp_fromstring_v2(
                        storage,
                        len(encoded),
                        dtype_number,
                        count,
                        separator.encode("ascii"),
                    )
                    with runtime._owned_result(
                        pointer, "cnp_fromstring_v2"
                    ) as actual:
                        assert_array_equivalent(
                            self, actual, expected
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_text_v2_errors_are_explicit_atomic_and_leak_free(self) -> None:
        cases = [
            (None, 0, 13, -1, b",", "input string is required"),
            (b"1,2", 3, 0, -1, b",", "valid numeric dtype"),
            (b"1,2", 3, 13, -2, b",", "count must be -1 or nonnegative"),
            (b"1,2", 3, 13, -1, b"", "separator must not be empty"),
            (b"1,bad,3", 7, 13, -1, b",", "invalid numeric token"),
            (b"256", 3, 3, -1, b",", "out of range"),
        ]
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for text, length, dtype_number, count, separator, message in cases:
                with self.subTest(text=text, dtype=dtype_number, count=count):
                    storage = (
                        None
                        if text is None
                        else ctypes.create_string_buffer(text)
                    )
                    runtime.dll.cnp_clear_error()
                    self.assertFalse(
                        runtime.dll.cnp_fromstring_v2(
                            storage,
                            length,
                            dtype_number,
                            count,
                            separator,
                        )
                    )
                    state = runtime.error_state()
                    self.assertNotEqual(0, state.status)
                    self.assertEqual("cnp_fromstring_v2", state.function)
                    self.assertIn(message, state.message)
                    self.assertEqual(baseline, runtime.retained_bytes)


class FromRegexSemanticsTests(unittest.TestCase):
    @staticmethod
    def _bind(runtime: CnumpyRuntime) -> None:
        runtime.dll.cnp_fromregex.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int64,
        ]
        runtime.dll.cnp_fromregex.restype = ctypes.c_void_p
        runtime.dll.cnp_fromregex_v2.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int64,
        ]
        runtime.dll.cnp_fromregex_v2.restype = ctypes.c_void_p
        runtime.dll.cnp_regex_result_count.argtypes = [ctypes.c_void_p]
        runtime.dll.cnp_regex_result_count.restype = ctypes.c_int64
        runtime.dll.cnp_regex_result_nfields.argtypes = [ctypes.c_void_p]
        runtime.dll.cnp_regex_result_nfields.restype = ctypes.c_int
        runtime.dll.cnp_regex_result_field_name.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        runtime.dll.cnp_regex_result_field_name.restype = ctypes.c_char_p
        runtime.dll.cnp_regex_result_field.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        runtime.dll.cnp_regex_result_field.restype = ctypes.c_void_p
        runtime.dll.cnp_regex_result_free.argtypes = [ctypes.c_void_p]
        runtime.dll.cnp_regex_result_free.restype = None

    @staticmethod
    def _field_arrays(
        runtime: CnumpyRuntime, result: int
    ) -> list:
        fields = []
        count = runtime.dll.cnp_regex_result_nfields(result)
        for index in range(count):
            pointer = runtime.dll.cnp_regex_result_field(result, index)
            fields.append(
                runtime._owned_result(pointer, "cnp_regex_result_field")
            )
        return fields

    def test_v2_matches_numpy_structured_numeric_captures_and_limit(
        self,
    ) -> None:
        text = "A=1.5 B=-2\nnoise\nA=3e2 B=4.25\nA=-7 B=9\n"
        pattern = r"A=([+-]?[0-9.eE]+)\s+B=([+-]?[0-9.eE]+)"
        field_names = ["a", "b"]
        field_types = [np.float64, np.float32]
        expected = np.fromregex(
            io.StringIO(text),
            pattern,
            list(zip(field_names, field_types)),
        )[:2]
        encoded_names = [name.encode("ascii") for name in field_names]
        names = (ctypes.c_char_p * 2)(*encoded_names)
        dtypes = (ctypes.c_int * 2)(13, 12)

        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            runtime.dll.cnp_clear_error()
            result = runtime.dll.cnp_fromregex_v2(
                text.encode("ascii"),
                pattern.encode("ascii"),
                names,
                dtypes,
                2,
                2,
            )
            self.assertTrue(result, runtime.error_state())
            fields = []
            try:
                self.assertEqual(2, runtime.dll.cnp_regex_result_count(result))
                self.assertEqual(2, runtime.dll.cnp_regex_result_nfields(result))
                self.assertEqual(
                    field_names,
                    [
                        runtime.dll.cnp_regex_result_field_name(result, index)
                        .decode("ascii")
                        for index in range(2)
                    ],
                )
                fields = self._field_arrays(runtime, result)
            finally:
                runtime.dll.cnp_regex_result_free(result)

            try:
                for index, field in enumerate(fields):
                    assert_array_equivalent(
                        self,
                        field,
                        expected[field_names[index]],
                        compare_contiguity=False,
                    )
            finally:
                for field in fields:
                    field.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_v2_no_match_empty_fields_and_legacy_single_capture_projection(
        self,
    ) -> None:
        names = (ctypes.c_char_p * 1)(b"number")
        dtypes = (ctypes.c_int * 1)(6)
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            result = runtime.dll.cnp_fromregex_v2(
                b"alpha beta", rb"id=(\d+)", names, dtypes, 1, -1
            )
            self.assertTrue(result, runtime.error_state())
            fields = self._field_arrays(runtime, result)
            runtime.dll.cnp_regex_result_free(result)
            try:
                self.assertEqual((0,), fields[0].shape)
                assert_array_equivalent(
                    self,
                    fields[0],
                    np.asarray([], dtype=np.int32),
                )
            finally:
                fields[0].close()

            expected = np.asarray([12, -7, 99], dtype=np.int64)
            runtime.dll.cnp_clear_error()
            pointer = runtime.dll.cnp_fromregex(
                b"v=12 v=-7 v=99",
                rb"v=(-?\d+)",
                10,
                -1,
            )
            with runtime._owned_result(pointer, "cnp_fromregex") as actual:
                assert_array_equivalent(self, actual, expected)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_v2_errors_are_explicit_atomic_and_leak_free(self) -> None:
        valid_names = (ctypes.c_char_p * 1)(b"value")
        two_names = (ctypes.c_char_p * 2)(b"left", b"right")
        valid_types = (ctypes.c_int * 1)(13)
        two_types = (ctypes.c_int * 2)(13, 13)
        invalid_types = (ctypes.c_int * 1)(999)
        cases = [
            (None, rb"(\d+)", valid_names, valid_types, 1, -1, "input text"),
            (b"1", b"(", valid_names, valid_types, 1, -1, "pattern"),
            (b"1", rb"(\d+)", two_names, two_types, 2, -1, "capture group"),
            (b"1", rb"(\d+)", valid_names, invalid_types, 1, -1, "dtype"),
            (b"1", rb"(\d+)", valid_names, valid_types, 1, -2, "max_matches"),
        ]
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for args in cases:
                with self.subTest(pattern=args[1], message=args[-1]):
                    runtime.dll.cnp_clear_error()
                    self.assertFalse(runtime.dll.cnp_fromregex_v2(*args[:-1]))
                    state = runtime.error_state()
                    self.assertNotEqual(0, state.status)
                    self.assertEqual("cnp_fromregex_v2", state.function)
                    self.assertIn(args[-1], state.message)
                    self.assertEqual(baseline, runtime.retained_bytes)

            runtime.dll.cnp_clear_error()
            self.assertFalse(runtime.dll.cnp_fromregex(None, rb"(\d+)", 13, 1))
            state = runtime.error_state()
            self.assertNotEqual(0, state.status)
            self.assertEqual("cnp_fromregex", state.function)
            self.assertEqual(baseline, runtime.retained_bytes)


class SafeEvalSemanticsTests(unittest.TestCase):
    @staticmethod
    def _bind(runtime: CnumpyRuntime) -> None:
        runtime.dll.cnp_safe_eval.argtypes = [ctypes.c_char_p]
        runtime.dll.cnp_safe_eval.restype = ctypes.c_double
        runtime.dll.cnp_safe_eval_v2.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_double),
        ]
        runtime.dll.cnp_safe_eval_v2.restype = ctypes.c_int

    def test_numeric_literals_match_numpy_safe_eval(self) -> None:
        expressions = [
            "1",
            "-3.5",
            "+.5",
            "4e2",
            "-2.25E-3",
            "((1.25))",
            "-(2)",
            "1_000",
            "0xff",
            "0b101",
            "0o17",
        ]
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for expression in expressions:
                expected = float(np.safe_eval(expression))
                actual = ctypes.c_double()
                runtime.dll.cnp_clear_error()
                status = runtime.dll.cnp_safe_eval_v2(
                    expression.encode("ascii"), ctypes.byref(actual)
                )
                self.assertEqual(0, status, runtime.error_state())
                self.assertEqual(expected, actual.value)
                self.assertEqual(
                    expected,
                    runtime.dll.cnp_safe_eval(expression.encode("ascii")),
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_rejects_non_numeric_syntax_and_exposes_legacy_failure(self) -> None:
        expressions = [
            "",
            "1 + 2",
            "2 ** 3",
            "--3",
            "1 xyz",
            "name",
            "obj.attr",
            "func(1)",
            "value[0]",
            "'string'",
            "[1, 2, 3]",
            "1 / 0",
            "(1 + 2",
        ]
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for expression in expressions:
                with self.subTest(expression=expression):
                    actual = ctypes.c_double(123.0)
                    runtime.dll.cnp_clear_error()
                    status = runtime.dll.cnp_safe_eval_v2(
                        expression.encode("ascii"), ctypes.byref(actual)
                    )
                    self.assertNotEqual(0, status)
                    state = runtime.error_state()
                    self.assertEqual("cnp_safe_eval_v2", state.function)
                    self.assertTrue(state.message)
                    self.assertEqual(123.0, actual.value)

                    runtime.dll.cnp_clear_error()
                    legacy = runtime.dll.cnp_safe_eval(expression.encode("ascii"))
                    self.assertTrue(math.isnan(legacy))
                    state = runtime.error_state()
                    self.assertNotEqual(0, state.status)
                    self.assertEqual("cnp_safe_eval", state.function)
                    self.assertEqual(baseline, runtime.retained_bytes)

            actual = ctypes.c_double()
            runtime.dll.cnp_clear_error()
            self.assertNotEqual(
                0, runtime.dll.cnp_safe_eval_v2(None, ctypes.byref(actual))
            )
            self.assertEqual("cnp_safe_eval_v2", runtime.error_state().function)
            runtime.dll.cnp_clear_error()
            self.assertNotEqual(
                0, runtime.dll.cnp_safe_eval_v2(b"1", None)
            )
            self.assertEqual("cnp_safe_eval_v2", runtime.error_state().function)
            self.assertEqual(baseline, runtime.retained_bytes)


class PolynomialConversionSemanticsTests(unittest.TestCase):
    REFERENCES = {
        "cnp_poly2cheb": np.polynomial.chebyshev.poly2cheb,
        "cnp_cheb2poly": np.polynomial.chebyshev.cheb2poly,
        "cnp_poly2leg": np.polynomial.legendre.poly2leg,
        "cnp_leg2poly": np.polynomial.legendre.leg2poly,
    }

    @staticmethod
    def _call(runtime: CnumpyRuntime, source, function_name: str):
        function = getattr(runtime.dll, function_name)
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        pointer = function(source.pointer)
        return runtime._owned_result(pointer, function_name)

    @staticmethod
    def _slice(runtime: CnumpyRuntime, source, specification: _CnpSlice):
        function = runtime.dll.cnp_array_slice
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(_CnpSlice),
        ]
        function.restype = ctypes.c_void_p
        storage = (_CnpSlice * 1)(specification)
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(
            function(source.pointer, 1, storage), "cnp_array_slice"
        )

    def assert_error(
        self,
        runtime: CnumpyRuntime,
        source,
        function_name: str,
        status: int,
        message: str,
    ) -> None:
        with self.assertRaises(CnumpyError) as raised:
            self._call(runtime, source, function_name)
        self.assertEqual(status, raised.exception.status)
        self.assertEqual(function_name, raised.exception.function)
        self.assertIn(message, raised.exception.message)

    def test_real_and_complex_coefficients_match_numpy_125(self) -> None:
        cases = [
            np.asarray([3], dtype=np.int64),
            np.asarray([1, 2, 3], dtype=np.uint64),
            np.asarray([1.0, 2.0, 3.0], dtype=np.float16),
            np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            np.asarray([0.5, -1.25, 2.0, 0.75, -3.0, 4.0]),
            np.asarray(
                [1.0 + 2.0j, -3.0 + 0.5j, 0.25 - 4.0j],
                dtype=np.complex64,
            ),
            np.asarray(
                [0.5 - 1.0j, 2.0 + 3.0j, -4.0 + 0.25j, 1.5j],
                dtype=np.complex128,
            ),
            np.asarray([1.0, 2.0, 0.0, 0.0]),
            np.asarray([0.0, 0.0, 0.0]),
        ]
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, reference in self.REFERENCES.items():
                for source_value in cases:
                    with self.subTest(
                        function=function_name, dtype=source_value.dtype,
                        values=source_value,
                    ), runtime.from_numpy(source_value) as source, self._call(
                        runtime, source, function_name
                    ) as actual:
                        expected = reference(source_value)
                        tolerance = (
                            3e-6
                            if expected.dtype in (np.float16, np.float32, np.complex64)
                            else 2e-13
                        )
                        assert_array_equivalent(
                            self,
                            actual,
                            expected,
                            rtol=tolerance,
                            atol=tolerance,
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_scalar_noncontiguous_and_round_trip_semantics(self) -> None:
        coefficients = np.asarray(
            [0.75, -1.0, 2.5, 0.25, -3.0, 1.5, 0.125],
            dtype=np.float64,
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, reference in self.REFERENCES.items():
                with runtime.from_numpy(np.asarray(3.5)) as scalar, self._call(
                    runtime, scalar, function_name
                ) as actual:
                    assert_array_equivalent(
                        self, actual, reference(np.asarray(3.5))
                    )

                base_value = np.arange(13, dtype=np.float64) - 4.5
                base = runtime.from_numpy(base_value)
                view = self._slice(
                    runtime,
                    base,
                    _CnpSlice(0, 0, -2, False, False, True),
                )
                base.close()
                with view, self._call(runtime, view, function_name) as actual:
                    assert_array_equivalent(
                        self,
                        actual,
                        reference(base_value[::-2]),
                        rtol=2e-13,
                        atol=2e-13,
                    )

            with runtime.from_numpy(coefficients) as power:
                chebyshev = self._call(runtime, power, "cnp_poly2cheb")
                with chebyshev, self._call(
                    runtime, chebyshev, "cnp_cheb2poly"
                ) as restored:
                    np.testing.assert_allclose(
                        restored.to_numpy(), coefficients,
                        rtol=2e-13, atol=2e-13,
                    )
                legendre = self._call(runtime, power, "cnp_poly2leg")
                with legendre, self._call(
                    runtime, legendre, "cnp_leg2poly"
                ) as restored:
                    np.testing.assert_allclose(
                        restored.to_numpy(), coefficients,
                        rtol=2e-13, atol=2e-13,
                    )
                np.testing.assert_array_equal(
                    power.to_numpy(), coefficients, strict=True
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_invalid_coefficient_arrays_report_exact_errors(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name in self.REFERENCES:
                function = getattr(runtime.dll, function_name)
                function.argtypes = [ctypes.c_void_p]
                function.restype = ctypes.c_void_p
                runtime.dll.cnp_clear_error()
                self.assertFalse(function(None))
                state = runtime.error_state()
                self.assertEqual(-1, state.status)
                self.assertEqual(function_name, state.function)
                self.assertIn("coefficient array is required", state.message)
                with runtime.from_numpy(
                    np.asarray([], dtype=np.float64)
                ) as empty:
                    self.assert_error(
                        runtime, empty, function_name, -4,
                        "coefficient array is empty",
                    )
                with runtime.from_numpy(
                    np.asarray([[1.0, 2.0], [3.0, 4.0]])
                ) as rank_two:
                    self.assert_error(
                        runtime, rank_two, function_name, -4,
                        "coefficient array is not 1-d",
                    )
                with runtime.from_numpy(
                    np.asarray([True, False], dtype=np.bool_)
                ) as boolean:
                    self.assert_error(
                        runtime, boolean, function_name, -3,
                        "coefficient dtype is not numeric",
                    )
            self.assertEqual(baseline, runtime.retained_bytes)


class PolynomialCalculusSemanticsTests(unittest.TestCase):
    DERIVATIVES = {
        "cnp_polyder": np.polyder,
        "cnp_chebder": np.polynomial.chebyshev.chebder,
        "cnp_legder": np.polynomial.legendre.legder,
        "cnp_hermder": np.polynomial.hermite.hermder,
        "cnp_lagder": np.polynomial.laguerre.lagder,
    }

    @staticmethod
    def _derivative(runtime: CnumpyRuntime, source, function_name: str, m: int):
        function = getattr(runtime.dll, function_name)
        function.argtypes = [ctypes.c_void_p, ctypes.c_int]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(
            function(source.pointer, m), function_name
        )

    @staticmethod
    def _polyint(runtime: CnumpyRuntime, source, m: int, constants=None):
        function = runtime.dll.cnp_polyint
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        if constants is None:
            pointer = function(source.pointer, m, None)
        else:
            with runtime.from_numpy(np.asarray(constants)) as native_constants:
                pointer = function(source.pointer, m, native_constants.pointer)
                return runtime._owned_result(pointer, "cnp_polyint")
        return runtime._owned_result(pointer, "cnp_polyint")

    @staticmethod
    def _chebint(runtime: CnumpyRuntime, source, m: int, lbnd: float):
        function = runtime.dll.cnp_chebint
        function.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(
            function(source.pointer, m, lbnd), "cnp_chebint"
        )

    @staticmethod
    def _slice(runtime: CnumpyRuntime, source, specification: _CnpSlice):
        return PolynomialConversionSemanticsTests._slice(
            runtime, source, specification
        )

    @staticmethod
    def _tolerance(expected: np.ndarray) -> float:
        return (
            4e-3
            if expected.dtype == np.float16
            else 4e-6
            if expected.dtype in (np.float32, np.complex64)
            else 3e-13
        )

    def test_derivatives_match_numpy_125_for_dtypes_orders_and_views(self) -> None:
        coefficient_cases = [
            np.asarray([True, False, True, True], dtype=np.bool_),
            np.asarray([2, -3, 4, 5], dtype=np.int64),
            np.asarray([1.5, -2.0, 0.25, 3.0], dtype=np.float16),
            np.asarray([1.5, -2.0, 0.25, 3.0], dtype=np.float32),
            np.asarray(
                [1.0 + 2.0j, -3.0 + 0.5j, 2.0j, 4.0 - 1.0j],
                dtype=np.complex64,
            ),
            np.asarray(
                [0.5 - 1.0j, 2.0 + 3.0j, -4.0 + 0.25j, 1.5j],
                dtype=np.complex128,
            ),
        ]
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, reference in self.DERIVATIVES.items():
                for coefficients in coefficient_cases:
                    for order in (0, 1, 2, 5):
                        with self.subTest(
                            function=function_name,
                            dtype=coefficients.dtype,
                            order=order,
                        ), runtime.from_numpy(coefficients) as source, self._derivative(
                            runtime, source, function_name, order
                        ) as actual:
                            expected = np.asarray(reference(coefficients, order))
                            tolerance = self._tolerance(expected)
                            assert_array_equivalent(
                                self,
                                actual,
                                expected,
                                rtol=tolerance,
                                atol=tolerance,
                            )

                base_value = np.asarray(
                    [0.5, -1.0, 2.0, 3.5, -4.0, 0.25, 1.5]
                )
                base = runtime.from_numpy(base_value)
                view = self._slice(
                    runtime,
                    base,
                    _CnpSlice(0, 0, -2, False, False, True),
                )
                base.close()
                with view, self._derivative(
                    runtime, view, function_name, 2
                ) as actual:
                    expected = np.asarray(reference(base_value[::-2], 2))
                    assert_array_equivalent(
                        self, actual, expected, rtol=3e-13, atol=3e-13
                    )

            matrix = np.arange(12, dtype=np.float32).reshape(4, 3) - 2.5
            for function_name, reference in tuple(self.DERIVATIVES.items())[1:]:
                with self.subTest(
                    function=function_name, layout="rank-two-axis-zero"
                ), runtime.from_numpy(matrix) as source, self._derivative(
                    runtime, source, function_name, 2
                ) as actual:
                    expected = np.asarray(reference(matrix, 2, axis=0))
                    assert_array_equivalent(
                        self,
                        actual,
                        expected,
                        compare_contiguity=False,
                        rtol=4e-6,
                        atol=4e-6,
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_float16_calculus_uses_ieee_round_to_nearest_even(self) -> None:
        coefficients = np.asarray(
            [
                -0.646,
                -0.3303,
                -1.529,
                -0.903,
                1.541,
                -0.9023,
                0.07336,
                0.00984,
            ],
            dtype=np.float16,
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(coefficients) as source, self._derivative(
                runtime, source, "cnp_chebder", 1
            ) as actual:
                assert_array_equivalent(
                    self,
                    actual,
                    np.polynomial.chebyshev.chebder(coefficients),
                )

            conversion = runtime.dll.cnp_float_to_half
            conversion.argtypes = [ctypes.c_double]
            conversion.restype = ctypes.c_uint16
            boundary_values = np.asarray(
                [
                    -0.0,
                    2.0**-25,
                    np.nextafter(2.0**-25, np.inf),
                    2.0**-24,
                    1.00048828125,
                    1.00146484375,
                    65519.99999999999,
                    65520.0,
                    np.inf,
                    -np.inf,
                ],
                dtype=np.float64,
            )
            with np.errstate(over="ignore", invalid="ignore"):
                expected_bits = boundary_values.astype(np.float16).view(
                    np.uint16
                )
            actual_bits = np.fromiter(
                (conversion(float(value)) for value in boundary_values),
                dtype=np.uint16,
                count=boundary_values.size,
            )
            np.testing.assert_array_equal(
                actual_bits, expected_bits, strict=True
            )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_power_and_chebyshev_integrals_match_numpy_125(self) -> None:
        coefficient_cases = [
            np.asarray([True, False, True], dtype=np.bool_),
            np.asarray([2, -3, 4], dtype=np.int64),
            np.asarray([1.5, -2.0, 0.25], dtype=np.float16),
            np.asarray([1.5, -2.0, 0.25], dtype=np.float32),
            np.asarray(
                [1.0 + 2.0j, -3.0 + 0.5j, 4.0 - 1.0j],
                dtype=np.complex64,
            ),
            np.asarray(
                [0.5 - 1.0j, 2.0 + 3.0j, -4.0 + 0.25j],
                dtype=np.complex128,
            ),
        ]
        power_specs = (
            (0, None),
            (1, None),
            (1, np.asarray([2.5])),
            (3, np.asarray([1.25])),
            (3, np.asarray([1.0, -2.0, 3.0])),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for coefficients in coefficient_cases:
                for order, constants in power_specs:
                    with self.subTest(
                        function="cnp_polyint",
                        dtype=coefficients.dtype,
                        order=order,
                        constants=constants,
                    ), runtime.from_numpy(coefficients) as source, self._polyint(
                        runtime, source, order, constants
                    ) as actual:
                        expected = np.asarray(
                            np.polyint(coefficients, order, k=constants)
                        )
                        assert_array_equivalent(
                            self,
                            actual,
                            expected,
                            rtol=3e-13,
                            atol=3e-13,
                        )

                for order, lbnd in ((0, 0.0), (1, 0.0), (1, -0.75), (3, 0.25)):
                    with self.subTest(
                        function="cnp_chebint",
                        dtype=coefficients.dtype,
                        order=order,
                        lbnd=lbnd,
                    ), runtime.from_numpy(coefficients) as source, self._chebint(
                        runtime, source, order, lbnd
                    ) as actual:
                        expected = np.asarray(
                            np.polynomial.chebyshev.chebint(
                                coefficients, order, lbnd=lbnd
                            )
                        )
                        tolerance = self._tolerance(expected)
                        assert_array_equivalent(
                            self,
                            actual,
                            expected,
                            rtol=tolerance,
                            atol=tolerance,
                        )

            matrix = np.arange(12, dtype=np.float32).reshape(4, 3) - 2.5
            with runtime.from_numpy(matrix) as source, self._chebint(
                runtime, source, 2, -0.25
            ) as actual:
                expected = np.polynomial.chebyshev.chebint(
                    matrix, 2, lbnd=-0.25, axis=0
                )
                assert_array_equivalent(
                    self,
                    actual,
                    expected,
                    compare_contiguity=False,
                    rtol=4e-6,
                    atol=4e-6,
                )

            with runtime.from_numpy(
                np.asarray([0.0], dtype=np.float32)
            ) as source, self._chebint(runtime, source, 3, 1.5) as actual:
                assert_array_equivalent(
                    self,
                    actual,
                    np.polynomial.chebyshev.chebint(
                        np.asarray([0.0], dtype=np.float32), 3, lbnd=1.5
                    ),
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_calculus_errors_are_explicit_and_do_not_leak(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name in self.DERIVATIVES:
                function = getattr(runtime.dll, function_name)
                function.argtypes = [ctypes.c_void_p, ctypes.c_int]
                function.restype = ctypes.c_void_p
                runtime.dll.cnp_clear_error()
                self.assertFalse(function(None, 1))
                state = runtime.error_state()
                self.assertEqual(-1, state.status)
                self.assertEqual(function_name, state.function)
                self.assertIn("coefficient array is required", state.message)
                with runtime.from_numpy(np.asarray([1.0, 2.0])) as source:
                    with self.assertRaises(CnumpyError) as raised:
                        self._derivative(runtime, source, function_name, -1)
                    self.assertEqual(-4, raised.exception.status)
                    self.assertEqual(function_name, raised.exception.function)
                    self.assertIn("order must be non-negative", raised.exception.message)

            function = runtime.dll.cnp_polyint
            function.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
            function.restype = ctypes.c_void_p
            runtime.dll.cnp_clear_error()
            self.assertFalse(function(None, 1, None))
            state = runtime.error_state()
            self.assertEqual(-1, state.status)
            self.assertEqual("cnp_polyint", state.function)
            self.assertIn("coefficient array is required", state.message)
            with runtime.from_numpy(np.asarray([1.0, 2.0])) as source:
                with self.assertRaises(CnumpyError) as raised:
                    self._polyint(runtime, source, -1)
                self.assertEqual(-4, raised.exception.status)
                self.assertIn("order must be non-negative", raised.exception.message)
                with self.assertRaises(CnumpyError) as raised:
                    self._polyint(runtime, source, 3, np.asarray([1.0, 2.0]))
                self.assertEqual(-4, raised.exception.status)
                self.assertIn("integration constants", raised.exception.message)

            function = runtime.dll.cnp_chebint
            function.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
            function.restype = ctypes.c_void_p
            runtime.dll.cnp_clear_error()
            self.assertFalse(function(None, 1, 0.0))
            state = runtime.error_state()
            self.assertEqual(-1, state.status)
            self.assertEqual("cnp_chebint", state.function)
            self.assertIn("coefficient array is required", state.message)
            with runtime.from_numpy(np.asarray([1.0, 2.0])) as source:
                with self.assertRaises(CnumpyError) as raised:
                    self._chebint(runtime, source, -1, 0.0)
                self.assertEqual(-4, raised.exception.status)
                self.assertIn("order must be non-negative", raised.exception.message)
            self.assertEqual(baseline, runtime.retained_bytes)


class PolynomialMultiplicationSemanticsTests(unittest.TestCase):
    @staticmethod
    def _call(runtime: CnumpyRuntime, left, right, function_name: str):
        function = getattr(runtime.dll, function_name)
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(
            function(left.pointer, right.pointer), function_name
        )

    def test_basis_multiplication_matches_values_dtypes_and_trimming(
        self,
    ) -> None:
        functions = {
            "cnp_chebmul": np.polynomial.chebyshev.chebmul,
            "cnp_legmul": np.polynomial.legendre.legmul,
            "cnp_hermmul": np.polynomial.hermite.hermmul,
            "cnp_lagmul": np.polynomial.laguerre.lagmul,
        }
        cases = [
            (
                np.asarray([1, 2, 3], dtype=np.int16),
                np.asarray([3, 2, 1], dtype=np.int32),
            ),
            (
                np.asarray([1, -2, 0.5], dtype=np.float16),
                np.asarray([0.25, 3], dtype=np.float32),
            ),
            (
                np.asarray([1, 2, 3], dtype=np.float32),
                np.asarray([3, -2, 1], dtype=np.float32),
            ),
            (
                np.asarray([1 + 2j, 3 - 1j, -0.5j], dtype=np.complex64),
                np.asarray([2 - 1j, 0.25 + 3j], dtype=np.complex64),
            ),
            (
                np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
                np.asarray([2.0, 0.0], dtype=np.float64),
            ),
        ]
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, reference in functions.items():
                for left_values, right_values in cases:
                    expected = reference(left_values, right_values)
                    tolerance = (
                        3e-5
                        if expected.dtype in (np.float16, np.float32, np.complex64)
                        else 3e-13
                    )
                    with self.subTest(
                        function=function_name,
                        left=left_values.dtype,
                        right=right_values.dtype,
                    ), runtime.from_numpy(left_values) as left, runtime.from_numpy(
                        right_values
                    ) as right, self._call(
                        runtime, left, right, function_name
                    ) as actual:
                        assert_array_equivalent(
                            self,
                            actual,
                            expected,
                            rtol=tolerance,
                            atol=tolerance,
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_constant_multiplier_matches_numpy_125_scalar_promotion(self) -> None:
        functions = {
            "cnp_chebmul": np.polynomial.chebyshev.chebmul,
            "cnp_legmul": np.polynomial.legendre.legmul,
            "cnp_hermmul": np.polynomial.hermite.hermmul,
            "cnp_lagmul": np.polynomial.laguerre.lagmul,
        }
        cases = [
            (
                np.asarray([1.0, 2.0], dtype=np.float16),
                np.asarray([3.0], dtype=np.float16),
            ),
            (
                np.asarray([1.0, 2.0], dtype=np.float32),
                np.asarray([3.0, 0.0], dtype=np.float32),
            ),
            (
                np.asarray([1.0 + 2.0j, 3.0 - 1.0j], dtype=np.complex64),
                np.asarray([2.0 - 1.0j], dtype=np.complex64),
            ),
        ]
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, reference in functions.items():
                for left_values, right_values in cases:
                    expected = reference(left_values, right_values)
                    with self.subTest(
                        function=function_name, dtype=left_values.dtype
                    ), runtime.from_numpy(left_values) as left, runtime.from_numpy(
                        right_values
                    ) as right, self._call(
                        runtime, left, right, function_name
                    ) as actual:
                        assert_array_equivalent(
                            self,
                            actual,
                            expected,
                            rtol=3e-3,
                            atol=3e-3,
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legendre_and_laguerre_round_each_recurrence_step(self) -> None:
        cases = [
            (
                "cnp_legmul",
                np.polynomial.legendre.legmul,
                np.asarray(
                    [1.1318359375, -0.654296875, -0.810546875,
                     1.56640625, 2.314453125],
                    dtype=np.float16,
                ),
                np.asarray(
                    [-0.337158203125, 0.302978515625, 1.5703125,
                     0.175048828125, -0.40380859375, 1.025390625,
                     1.30078125],
                    dtype=np.float16,
                ),
            ),
            (
                "cnp_lagmul",
                np.polynomial.laguerre.lagmul,
                np.asarray(
                    [-0.7890625, 0.07769775390625,
                     0.27294921875, 0.350830078125],
                    dtype=np.float16,
                ),
                np.asarray(
                    [0.31591796875, 0.140869140625, 1.2861328125],
                    dtype=np.float16,
                ),
            ),
        ]
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, reference, left_values, right_values in cases:
                expected = reference(left_values, right_values)
                with self.subTest(function=function_name), runtime.from_numpy(
                    left_values
                ) as left, runtime.from_numpy(right_values) as right, self._call(
                    runtime, left, right, function_name
                ) as actual:
                    assert_array_equivalent(self, actual, expected)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_seeded_low_precision_recurrences_match_numpy_125(self) -> None:
        functions = {
            "cnp_legmul": np.polynomial.legendre.legmul,
            "cnp_hermmul": np.polynomial.hermite.hermmul,
            "cnp_lagmul": np.polynomial.laguerre.lagmul,
        }
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, reference in functions.items():
                for dtype in (np.float16, np.float32, np.complex64):
                    rng = np.random.default_rng(20260730)
                    left_values = rng.normal(0.0, 0.15, 6)
                    right_values = rng.normal(0.0, 0.15, 7)
                    if dtype == np.complex64:
                        left_values = left_values + 1j * rng.normal(
                            0.0, 0.15, 6
                        )
                        right_values = right_values + 1j * rng.normal(
                            0.0, 0.15, 7
                        )
                    left_values = np.asarray(left_values, dtype=dtype)
                    right_values = np.asarray(right_values, dtype=dtype)
                    expected = reference(left_values, right_values)
                    with self.subTest(
                        function=function_name, dtype=np.dtype(dtype).name
                    ), runtime.from_numpy(left_values) as left, runtime.from_numpy(
                        right_values
                    ) as right, self._call(
                        runtime, left, right, function_name
                    ) as actual:
                        if dtype == np.complex64:
                            # NumPy's complex64 array multiply is CPU-dispatched;
                            # SIMD/FMA paths can differ in the final ULP.
                            assert_array_equivalent(
                                self,
                                actual,
                                expected,
                                rtol=1e-6,
                                atol=1e-7,
                            )
                        else:
                            assert_array_equivalent(self, actual, expected)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_native_strided_views_match_numpy_125(self) -> None:
        functions = {
            "cnp_chebmul": np.polynomial.chebyshev.chebmul,
            "cnp_legmul": np.polynomial.legendre.legmul,
            "cnp_hermmul": np.polynomial.hermite.hermmul,
            "cnp_lagmul": np.polynomial.laguerre.lagmul,
        }
        left_base_values = np.asarray(
            [
                0.05 + 0.01j,
                -0.10 + 0.03j,
                0.15 - 0.02j,
                0.20 + 0.04j,
                -0.25 - 0.05j,
                0.30 + 0.02j,
                0.35 - 0.01j,
                -0.40 + 0.03j,
                0.45 + 0.05j,
            ],
            dtype=np.complex64,
        )
        right_base_values = np.asarray(
            [
                -0.08 + 0.02j,
                0.12 - 0.01j,
                0.16 + 0.03j,
                -0.20 - 0.04j,
                0.24 + 0.01j,
                0.28 - 0.02j,
                -0.32 + 0.05j,
                0.36 + 0.02j,
            ],
            dtype=np.complex64,
        )
        expected_left = left_base_values[::-2]
        expected_right = right_base_values[1::2]
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, reference in functions.items():
                left_base = runtime.from_numpy(left_base_values)
                left_view = PolynomialConversionSemanticsTests._slice(
                    runtime,
                    left_base,
                    _CnpSlice(0, 0, -2, False, False, True),
                )
                left_base.close()
                right_base = runtime.from_numpy(right_base_values)
                right_view = PolynomialConversionSemanticsTests._slice(
                    runtime,
                    right_base,
                    _CnpSlice(1, 0, 2, True, False, True),
                )
                right_base.close()
                with self.subTest(function=function_name), left_view, right_view, self._call(
                    runtime, left_view, right_view, function_name
                ) as actual:
                    assert_array_equivalent(
                        self,
                        actual,
                        reference(expected_left, expected_right),
                        rtol=2e-5,
                        atol=2e-6,
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_nan_infinity_and_seeded_degrees_match_numpy_125(self) -> None:
        functions = {
            "cnp_chebmul": np.polynomial.chebyshev.chebmul,
            "cnp_legmul": np.polynomial.legendre.legmul,
            "cnp_hermmul": np.polynomial.hermite.hermmul,
            "cnp_lagmul": np.polynomial.laguerre.lagmul,
        }
        degree_pairs = ((2, 3), (4, 6), (8, 5), (11, 9))
        dtypes = (
            np.float16,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, reference in functions.items():
                special_left = np.asarray(
                    [1.0, np.nan, -0.0, np.inf], dtype=np.float64
                )
                special_right = np.asarray(
                    [-np.inf, 2.0, -3.0], dtype=np.float64
                )
                with np.errstate(all="ignore"):
                    special_expected = reference(special_left, special_right)
                with self.subTest(
                    function=function_name, values="nan-infinity"
                ), runtime.from_numpy(special_left) as left, runtime.from_numpy(
                    special_right
                ) as right, self._call(
                    runtime, left, right, function_name
                ) as actual:
                    assert_array_equivalent(
                        self,
                        actual,
                        special_expected,
                        rtol=3e-13,
                        atol=3e-13,
                    )

                for dtype in dtypes:
                    for left_degree, right_degree in degree_pairs:
                        seed = (
                            20260730
                            + 101 * left_degree
                            + 17 * right_degree
                            + np.dtype(dtype).itemsize
                        )
                        rng = np.random.default_rng(seed)
                        scale = (
                            0.04
                            if function_name in ("cnp_hermmul", "cnp_lagmul")
                            else 0.30
                        )
                        left_values = rng.normal(
                            0.0, scale, left_degree + 1
                        )
                        right_values = rng.normal(
                            0.0, scale, right_degree + 1
                        )
                        if np.issubdtype(dtype, np.complexfloating):
                            left_values = left_values + 1j * rng.normal(
                                0.0, scale, left_degree + 1
                            )
                            right_values = right_values + 1j * rng.normal(
                                0.0, scale, right_degree + 1
                            )
                        left_values = np.asarray(left_values, dtype=dtype)
                        right_values = np.asarray(right_values, dtype=dtype)
                        with np.errstate(all="ignore"):
                            expected = reference(left_values, right_values)
                        tolerance = (
                            5e-3
                            if dtype == np.float16
                            else 2e-5
                            if dtype in (np.float32, np.complex64)
                            else 5e-12
                        )
                        with self.subTest(
                            function=function_name,
                            dtype=np.dtype(dtype).name,
                            degrees=(left_degree, right_degree),
                        ), runtime.from_numpy(
                            left_values
                        ) as left, runtime.from_numpy(
                            right_values
                        ) as right, self._call(
                            runtime, left, right, function_name
                        ) as actual:
                            assert_array_equivalent(
                                self,
                                actual,
                                expected,
                                rtol=tolerance,
                                atol=tolerance,
                            )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_invalid_inputs_are_explicit_and_do_not_leak(self) -> None:
        function_names = (
            "cnp_chebmul",
            "cnp_legmul",
            "cnp_hermmul",
            "cnp_lagmul",
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(np.asarray([1.0, 2.0])) as valid:
                for function_name in function_names:
                    function = getattr(runtime.dll, function_name)
                    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                    function.restype = ctypes.c_void_p
                    for left_pointer, right_pointer in (
                        (None, valid.pointer),
                        (valid.pointer, None),
                    ):
                        runtime.dll.cnp_clear_error()
                        self.assertFalse(function(left_pointer, right_pointer))
                        state = runtime.error_state()
                        self.assertEqual(-1, state.status)
                        self.assertEqual(function_name, state.function)
                        self.assertIn(
                            "input coefficient arrays are required",
                            state.message,
                        )

                    for invalid_values, status, message in (
                        (
                            np.asarray([], dtype=np.float64),
                            -4,
                            "coefficient array is empty",
                        ),
                        (
                            np.asarray([[1.0, 2.0], [3.0, 4.0]]),
                            -4,
                            "coefficient arrays must be one-dimensional",
                        ),
                        (
                            np.asarray([True, False], dtype=np.bool_),
                            -3,
                            "no common real or complex type",
                        ),
                    ):
                        with self.subTest(
                            function=function_name, invalid=message
                        ), runtime.from_numpy(invalid_values) as invalid:
                            with self.assertRaises(CnumpyError) as raised:
                                self._call(
                                    runtime, invalid, valid, function_name
                                )
                            self.assertEqual(status, raised.exception.status)
                            self.assertEqual(
                                function_name, raised.exception.function
                            )
                            self.assertIn(message, raised.exception.message)
            self.assertEqual(baseline, runtime.retained_bytes)


class ChebyshevAdditionSemanticsTests(unittest.TestCase):
    FUNCTIONS = {
        "cnp_chebadd": np.polynomial.chebyshev.chebadd,
        "cnp_chebsub": np.polynomial.chebyshev.chebsub,
    }

    @staticmethod
    def _call(runtime: CnumpyRuntime, left, right, function_name: str):
        function = getattr(runtime.dll, function_name)
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(
            function(left.pointer, right.pointer), function_name
        )

    def test_values_dtypes_scalars_trimming_and_special_values(self) -> None:
        cases = (
            (
                np.asarray([1, 2, 0], dtype=np.int16),
                np.asarray([3, -2], dtype=np.int32),
            ),
            (
                np.asarray([1.0, -2.0, 0.0], dtype=np.float16),
                np.asarray([-1.0, 2.0, 0.0], dtype=np.float16),
            ),
            (
                np.asarray([0.25, -3.0, 2.0], dtype=np.float32),
                np.asarray([1.5], dtype=np.float32),
            ),
            (
                np.asarray([1.0 + 2.0j, -3.0 + 0.5j], dtype=np.complex64),
                np.asarray([2.0 - 1.0j, 3.0 - 0.5j], dtype=np.complex64),
            ),
            (
                np.asarray(2.0, dtype=np.float32),
                np.asarray([1.0, 2.0], dtype=np.float32),
            ),
            (
                np.asarray([-0.0, np.nan, np.inf], dtype=np.float64),
                np.asarray([-0.0, 1.0, -np.inf], dtype=np.float64),
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, reference in self.FUNCTIONS.items():
                for left_values, right_values in cases:
                    with np.errstate(all="ignore"):
                        expected = reference(left_values, right_values)
                    with self.subTest(
                        function=function_name,
                        left_dtype=left_values.dtype,
                        right_dtype=right_values.dtype,
                    ), runtime.from_numpy(left_values) as left, runtime.from_numpy(
                        right_values
                    ) as right, self._call(
                        runtime, left, right, function_name
                    ) as actual:
                        assert_array_equivalent(self, actual, expected)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_native_strided_inputs_match_numpy_125(self) -> None:
        left_base_values = np.asarray(
            [0.5, -1.0, 2.0, 3.5, -4.0, 0.25, 1.5]
        )
        right_base_values = np.asarray(
            [-2.0, 0.75, 4.0, -1.25, 0.5, 3.0]
        )
        expected_left = left_base_values[::-2]
        expected_right = right_base_values[1::2]
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, reference in self.FUNCTIONS.items():
                left_base = runtime.from_numpy(left_base_values)
                left_view = PolynomialConversionSemanticsTests._slice(
                    runtime,
                    left_base,
                    _CnpSlice(0, 0, -2, False, False, True),
                )
                left_base.close()
                right_base = runtime.from_numpy(right_base_values)
                right_view = PolynomialConversionSemanticsTests._slice(
                    runtime,
                    right_base,
                    _CnpSlice(1, 0, 2, True, False, True),
                )
                right_base.close()
                with self.subTest(
                    function=function_name
                ), left_view, right_view, self._call(
                    runtime, left_view, right_view, function_name
                ) as actual:
                    assert_array_equivalent(
                        self,
                        actual,
                        reference(expected_left, expected_right),
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_invalid_inputs_are_explicit_and_do_not_leak(self) -> None:
        invalid_cases = (
            (
                np.asarray([], dtype=np.float64),
                -4,
                "coefficient array is empty",
            ),
            (
                np.asarray([[1.0, 2.0], [3.0, 4.0]]),
                -4,
                "coefficient arrays must be one-dimensional",
            ),
            (
                np.asarray([True, False], dtype=np.bool_),
                -3,
                "no common real or complex type",
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(np.asarray([1.0, 2.0])) as valid:
                for function_name in self.FUNCTIONS:
                    function = getattr(runtime.dll, function_name)
                    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                    function.restype = ctypes.c_void_p
                    for left_pointer, right_pointer in (
                        (None, valid.pointer),
                        (valid.pointer, None),
                    ):
                        with self.subTest(
                            function=function_name,
                            missing="left" if left_pointer is None else "right",
                        ):
                            runtime.dll.cnp_clear_error()
                            pointer = function(left_pointer, right_pointer)
                            if pointer:
                                runtime.dll.cnp_array_decref(pointer)
                            self.assertFalse(pointer)
                            state = runtime.error_state()
                            self.assertEqual(-1, state.status)
                            self.assertEqual(function_name, state.function)
                            self.assertIn(
                                "input coefficient arrays are required",
                                state.message,
                            )

                    for invalid_values, status, message in invalid_cases:
                        with self.subTest(
                            function=function_name, invalid=message
                        ), runtime.from_numpy(invalid_values) as invalid:
                            runtime.dll.cnp_clear_error()
                            pointer = function(invalid.pointer, valid.pointer)
                            if pointer:
                                runtime.dll.cnp_array_decref(pointer)
                            self.assertFalse(pointer)
                            state = runtime.error_state()
                            self.assertEqual(status, state.status)
                            self.assertEqual(function_name, state.function)
                            self.assertIn(message, state.message)
            self.assertEqual(baseline, runtime.retained_bytes)


class PowerPolynomialAdditionSemanticsTests(unittest.TestCase):
    FUNCTIONS = {
        "cnp_polyadd": np.polyadd,
        "cnp_polysub": np.polysub,
    }
    DTYPES = (
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
        np.complex64,
        np.complex128,
    )

    @staticmethod
    def _call(runtime: CnumpyRuntime, left, right, function_name: str):
        function = getattr(runtime.dll, function_name)
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(
            function(left.pointer, right.pointer), function_name
        )

    def test_values_dtypes_alignment_and_special_values(self) -> None:
        cases = (
            (
                np.asarray([1, -2, 3], dtype=np.int16),
                np.asarray([4, 5], dtype=np.int32),
            ),
            (
                np.asarray([120, -128], dtype=np.int8),
                np.asarray([120, -1], dtype=np.int8),
            ),
            (
                np.asarray([2**63 + 7, 2**64 - 1], dtype=np.uint64),
                np.asarray([2**63 + 9, 1], dtype=np.uint64),
            ),
            (
                np.asarray([1.0, -2.0, 0.0], dtype=np.float16),
                np.asarray([-1.0, 2.0, 0.0], dtype=np.float16),
            ),
            (
                np.asarray([0.25, -3.0, 2.0], dtype=np.float32),
                np.asarray([1.5], dtype=np.float32),
            ),
            (
                np.asarray([1.0 + 2.0j, -3.0 + 0.5j], dtype=np.complex64),
                np.asarray([2.0 - 1.0j, 3.0 - 0.5j], dtype=np.complex64),
            ),
            (
                np.asarray(2.0, dtype=np.float32),
                np.asarray([1.0, 2.0], dtype=np.float32),
            ),
            (
                np.asarray([-0.0, np.nan, np.inf], dtype=np.float64),
                np.asarray([0.0, 1.0, -np.inf], dtype=np.float64),
            ),
            (
                np.asarray([], dtype=np.float32),
                np.asarray([1, 2], dtype=np.int16),
            ),
            (
                np.asarray([], dtype=np.float32),
                np.asarray([], dtype=np.int16),
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, reference in self.FUNCTIONS.items():
                for left_values, right_values in cases:
                    with np.errstate(all="ignore"):
                        expected = reference(left_values, right_values)
                    with self.subTest(
                        function=function_name,
                        left_dtype=left_values.dtype,
                        right_dtype=right_values.dtype,
                        left_size=left_values.size,
                        right_size=right_values.size,
                    ), runtime.from_numpy(left_values) as left, runtime.from_numpy(
                        right_values
                    ) as right, self._call(
                        runtime, left, right, function_name
                    ) as actual:
                        assert_array_equivalent(self, actual, expected)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_pairwise_supported_dtype_promotion_matches_numpy_125(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, reference in self.FUNCTIONS.items():
                for left_dtype in self.DTYPES:
                    for right_dtype in self.DTYPES:
                        if (
                            function_name == "cnp_polysub"
                            and left_dtype is np.bool_
                            and right_dtype is np.bool_
                        ):
                            continue
                        left_values = np.asarray([1, 0], dtype=left_dtype)
                        right_values = np.asarray([1], dtype=right_dtype)
                        expected = reference(left_values, right_values)
                        with self.subTest(
                            function=function_name,
                            left_dtype=np.dtype(left_dtype),
                            right_dtype=np.dtype(right_dtype),
                        ), runtime.from_numpy(left_values) as left, runtime.from_numpy(
                            right_values
                        ) as right, self._call(
                            runtime, left, right, function_name
                        ) as actual:
                            assert_array_equivalent(self, actual, expected)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_boolean_addition_and_subtraction_match_numpy_125(self) -> None:
        left_values = np.asarray([True, True, False], dtype=np.bool_)
        right_values = np.asarray([True, False, False], dtype=np.bool_)
        with CnumpyRuntime(DLL) as runtime, runtime.from_numpy(
            left_values
        ) as left, runtime.from_numpy(right_values) as right:
            baseline = runtime.retained_bytes
            with self._call(runtime, left, right, "cnp_polyadd") as actual:
                assert_array_equivalent(
                    self, actual, np.polyadd(left_values, right_values)
                )
            runtime.dll.cnp_clear_error()
            function = runtime.dll.cnp_polysub
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            function.restype = ctypes.c_void_p
            pointer = function(left.pointer, right.pointer)
            if pointer:
                runtime.dll.cnp_array_decref(pointer)
            self.assertFalse(pointer)
            state = runtime.error_state()
            self.assertEqual(-3, state.status)
            self.assertEqual("cnp_polysub", state.function)
            self.assertIn("boolean subtract is not supported", state.message)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_native_strided_inputs_match_numpy_125(self) -> None:
        left_base_values = np.asarray(
            [0.5, -1.0, 2.0, 3.5, -4.0, 0.25, 1.5]
        )
        right_base_values = np.asarray(
            [-2.0, 0.75, 4.0, -1.25, 0.5, 3.0]
        )
        expected_left = left_base_values[::-2]
        expected_right = right_base_values[1::2]
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, reference in self.FUNCTIONS.items():
                left_base = runtime.from_numpy(left_base_values)
                left_view = PolynomialConversionSemanticsTests._slice(
                    runtime,
                    left_base,
                    _CnpSlice(0, 0, -2, False, False, True),
                )
                left_base.close()
                right_base = runtime.from_numpy(right_base_values)
                right_view = PolynomialConversionSemanticsTests._slice(
                    runtime,
                    right_base,
                    _CnpSlice(1, 0, 2, True, False, True),
                )
                right_base.close()
                with self.subTest(
                    function=function_name
                ), left_view, right_view, self._call(
                    runtime, left_view, right_view, function_name
                ) as actual:
                    assert_array_equivalent(
                        self,
                        actual,
                        reference(expected_left, expected_right),
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_invalid_inputs_are_explicit_and_do_not_leak(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(np.asarray([1.0, 2.0])) as valid:
                for function_name in self.FUNCTIONS:
                    function = getattr(runtime.dll, function_name)
                    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                    function.restype = ctypes.c_void_p
                    for left_pointer, right_pointer in (
                        (None, valid.pointer),
                        (valid.pointer, None),
                    ):
                        with self.subTest(
                            function=function_name,
                            missing="left" if left_pointer is None else "right",
                        ):
                            runtime.dll.cnp_clear_error()
                            pointer = function(left_pointer, right_pointer)
                            if pointer:
                                runtime.dll.cnp_array_decref(pointer)
                            self.assertFalse(pointer)
                            state = runtime.error_state()
                            self.assertEqual(-1, state.status)
                            self.assertEqual(function_name, state.function)
                            self.assertIn(
                                "polynomial inputs are required", state.message
                            )

                with runtime.from_numpy(
                    np.asarray([[1.0, 2.0], [3.0, 4.0]])
                ) as rank_two:
                    for function_name in self.FUNCTIONS:
                        with self.subTest(function=function_name):
                            runtime.dll.cnp_clear_error()
                            function = getattr(runtime.dll, function_name)
                            pointer = function(rank_two.pointer, valid.pointer)
                            if pointer:
                                runtime.dll.cnp_array_decref(pointer)
                            self.assertFalse(pointer)
                            state = runtime.error_state()
                            self.assertEqual(-4, state.status)
                            self.assertEqual(function_name, state.function)
                            self.assertIn(
                                "polynomial inputs must be one-dimensional",
                                state.message,
                            )
            self.assertEqual(baseline, runtime.retained_bytes)


class PowerPolynomialMultiplicationSemanticsTests(unittest.TestCase):
    DTYPES = PowerPolynomialAdditionSemanticsTests.DTYPES

    @staticmethod
    def _call(runtime: CnumpyRuntime, left, right):
        function = runtime.dll.cnp_polymul
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(
            function(left.pointer, right.pointer), "cnp_polymul"
        )

    def test_values_dtypes_trimming_rounding_and_special_values(self) -> None:
        cases = (
            (
                np.asarray([0, 1, -2, 3], dtype=np.int16),
                np.asarray([0, 4, 5], dtype=np.int32),
            ),
            (
                np.asarray([120, 120], dtype=np.int8),
                np.asarray([2, 2], dtype=np.int8),
            ),
            (
                np.asarray([2**63 + 7, 2**64 - 1], dtype=np.uint64),
                np.asarray([2, 3], dtype=np.uint64),
            ),
            (
                np.asarray([0.316162109375, -2.791015625], dtype=np.float16),
                np.asarray([-0.0877685546875, 2.0859375], dtype=np.float16),
            ),
            (
                np.asarray(
                    [11.434530258178711, -4.5261101722717285],
                    dtype=np.float32,
                ),
                np.asarray(
                    [4.30485725402832, 2.5093257427215576],
                    dtype=np.float32,
                ),
            ),
            (
                np.asarray(
                    [
                        -2.5212347507476807,
                        8.511785507202148,
                        4.1434712409973145,
                        1.593431830406189,
                    ],
                    dtype=np.float32,
                ),
                np.asarray(
                    [
                        6.035024642944336,
                        2.626380443572998,
                        18.79629898071289,
                        4.900320529937744,
                    ],
                    dtype=np.float32,
                ),
            ),
            (
                np.asarray([1.0 + 2.0j, -3.0 + 0.5j], dtype=np.complex64),
                np.asarray([2.0 - 1.0j, 3.0 - 0.5j], dtype=np.complex64),
            ),
            (
                np.asarray(2.0, dtype=np.float32),
                np.asarray([1.0, 2.0], dtype=np.float32),
            ),
            (
                np.asarray([0.0, 0.0], dtype=np.float32),
                np.asarray([1.0, 2.0], dtype=np.float32),
            ),
            (
                np.asarray([], dtype=np.float32),
                np.asarray([1, 2], dtype=np.int16),
            ),
            (
                np.asarray([], dtype=np.float32),
                np.asarray([], dtype=np.int16),
            ),
            (
                np.asarray([-0.0, np.nan, np.inf], dtype=np.float64),
                np.asarray([0.0, 1.0, -np.inf], dtype=np.float64),
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for left_values, right_values in cases:
                with np.errstate(all="ignore"):
                    expected = np.polymul(left_values, right_values)
                with self.subTest(
                    left_dtype=left_values.dtype,
                    right_dtype=right_values.dtype,
                    left_size=left_values.size,
                    right_size=right_values.size,
                ), runtime.from_numpy(left_values) as left, runtime.from_numpy(
                    right_values
                ) as right, self._call(runtime, left, right) as actual:
                    dispatch_sensitive = (
                        left_values.dtype == np.dtype(np.float32)
                        and right_values.dtype == np.dtype(np.float32)
                        and left_values.size == 4
                        and right_values.size == 4
                    )
                    assert_array_equivalent(
                        self,
                        actual,
                        expected,
                        rtol=1e-7 if dispatch_sensitive else None,
                        atol=0.0 if dispatch_sensitive else None,
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_pairwise_supported_dtype_promotion_matches_numpy_125(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for left_dtype in self.DTYPES:
                for right_dtype in self.DTYPES:
                    left_values = np.asarray([1, 0], dtype=left_dtype)
                    right_values = np.asarray([1, 1], dtype=right_dtype)
                    expected = np.polymul(left_values, right_values)
                    with self.subTest(
                        left_dtype=np.dtype(left_dtype),
                        right_dtype=np.dtype(right_dtype),
                    ), runtime.from_numpy(left_values) as left, runtime.from_numpy(
                        right_values
                    ) as right, self._call(runtime, left, right) as actual:
                        assert_array_equivalent(self, actual, expected)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_native_strided_inputs_match_numpy_125(self) -> None:
        left_base_values = np.asarray(
            [0.0, -1.0, 2.0, 3.5, -4.0, 0.25, 1.5]
        )
        right_base_values = np.asarray(
            [-2.0, 0.75, 4.0, -1.25, 0.5, 3.0]
        )
        expected_left = left_base_values[::-2]
        expected_right = right_base_values[1::2]
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            left_base = runtime.from_numpy(left_base_values)
            left_view = PolynomialConversionSemanticsTests._slice(
                runtime,
                left_base,
                _CnpSlice(0, 0, -2, False, False, True),
            )
            left_base.close()
            right_base = runtime.from_numpy(right_base_values)
            right_view = PolynomialConversionSemanticsTests._slice(
                runtime,
                right_base,
                _CnpSlice(1, 0, 2, True, False, True),
            )
            right_base.close()
            with left_view, right_view, self._call(
                runtime, left_view, right_view
            ) as actual:
                assert_array_equivalent(
                    self,
                    actual,
                    np.polymul(expected_left, expected_right),
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_invalid_inputs_are_explicit_and_do_not_leak(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(np.asarray([1.0, 2.0])) as valid:
                function = runtime.dll.cnp_polymul
                function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                function.restype = ctypes.c_void_p
                for left_pointer, right_pointer in (
                    (None, valid.pointer),
                    (valid.pointer, None),
                ):
                    with self.subTest(
                        missing="left" if left_pointer is None else "right"
                    ):
                        runtime.dll.cnp_clear_error()
                        pointer = function(left_pointer, right_pointer)
                        if pointer:
                            runtime.dll.cnp_array_decref(pointer)
                        self.assertFalse(pointer)
                        state = runtime.error_state()
                        self.assertEqual(-1, state.status)
                        self.assertEqual("cnp_polymul", state.function)
                        self.assertIn(
                            "polynomial inputs are required", state.message
                        )

                with runtime.from_numpy(
                    np.asarray([[1.0, 2.0], [3.0, 4.0]])
                ) as rank_two:
                    runtime.dll.cnp_clear_error()
                    pointer = function(rank_two.pointer, valid.pointer)
                    if pointer:
                        runtime.dll.cnp_array_decref(pointer)
                    self.assertFalse(pointer)
                    state = runtime.error_state()
                    self.assertEqual(-4, state.status)
                    self.assertEqual("cnp_polymul", state.function)
                    self.assertIn(
                        "polynomial inputs must be one-dimensional",
                        state.message,
                    )
            self.assertEqual(baseline, runtime.retained_bytes)


class PowerPolynomialDivisionSemanticsTests(unittest.TestCase):
    DTYPES = PowerPolynomialAdditionSemanticsTests.DTYPES

    @staticmethod
    def _function(runtime: CnumpyRuntime):
        function = runtime.dll.cnp_polydiv
        output_pointer = ctypes.POINTER(ctypes.c_void_p)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            output_pointer,
            output_pointer,
        ]
        function.restype = ctypes.c_int
        return function

    @classmethod
    def _call(cls, runtime: CnumpyRuntime, dividend, divisor):
        quotient_pointer = ctypes.c_void_p()
        remainder_pointer = ctypes.c_void_p()
        runtime.dll.cnp_clear_error()
        status = int(
            cls._function(runtime)(
                dividend.pointer,
                divisor.pointer,
                ctypes.byref(quotient_pointer),
                ctypes.byref(remainder_pointer),
            )
        )
        if status != 0:
            unexpected = (
                quotient_pointer.value,
                remainder_pointer.value,
            )
            for pointer in unexpected:
                if pointer:
                    runtime.dll.cnp_array_decref(pointer)
            if any(unexpected):
                raise RuntimeError(
                    "cnp_polydiv failed with live partial results: "
                    f"{unexpected}"
                )
            raise runtime.native_error("cnp_polydiv", status)
        if not quotient_pointer.value or not remainder_pointer.value:
            for pointer in (
                quotient_pointer.value,
                remainder_pointer.value,
            ):
                if pointer:
                    runtime.dll.cnp_array_decref(pointer)
            raise RuntimeError("cnp_polydiv succeeded with a null result")
        return (
            runtime._owned_result(quotient_pointer.value, "cnp_polydiv"),
            runtime._owned_result(remainder_pointer.value, "cnp_polydiv"),
        )

    def test_values_dtypes_trimming_and_special_values(self) -> None:
        cases = (
            (
                np.asarray([3.0, 5.0, 2.0], dtype=np.float64),
                np.asarray([2.0, 1.0], dtype=np.float64),
            ),
            (
                np.asarray([3.0, 5.0, 2.0], dtype=np.float16),
                np.asarray([2.0, 1.0], dtype=np.float16),
            ),
            (
                np.asarray([3.0, 5.0, 2.0], dtype=np.float32),
                np.asarray([2.0, 1.0], dtype=np.float32),
            ),
            (
                np.asarray(
                    [3.0 + 1.0j, 5.0 - 2.0j, 2.0 + 0.5j],
                    dtype=np.complex64,
                ),
                np.asarray([2.0 - 0.5j, 1.0 + 1.0j], dtype=np.complex64),
            ),
            (
                np.asarray([3, 5, 2], dtype=np.int16),
                np.asarray([2, 1], dtype=np.uint32),
            ),
            (
                np.asarray(2.0, dtype=np.float32),
                np.asarray([1.0, 2.0], dtype=np.float32),
            ),
            (
                np.asarray([1.0e-9, 2.0], dtype=np.float64),
                np.asarray([1.0, 0.0, 1.0], dtype=np.float64),
            ),
            (
                np.asarray([0.0, 3.0, 5.0, 2.0], dtype=np.float64),
                np.asarray([2.0, 1.0], dtype=np.float64),
            ),
            (
                np.asarray([-0.0, 1.0], dtype=np.float64),
                np.asarray([1.0], dtype=np.float64),
            ),
            (
                np.asarray([1.0], dtype=np.float64),
                np.asarray([0.0], dtype=np.float64),
            ),
            (
                np.asarray([1.0, 2.0], dtype=np.float64),
                np.asarray([0.0, 1.0], dtype=np.float64),
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for dividend_values, divisor_values in cases:
                with np.errstate(all="ignore"):
                    expected_quotient, expected_remainder = np.polydiv(
                        dividend_values, divisor_values
                    )
                with self.subTest(
                    dividend_dtype=dividend_values.dtype,
                    divisor_dtype=divisor_values.dtype,
                    dividend_size=dividend_values.size,
                    divisor_size=divisor_values.size,
                ), runtime.from_numpy(
                    dividend_values
                ) as dividend, runtime.from_numpy(
                    divisor_values
                ) as divisor:
                    quotient, remainder = self._call(
                        runtime, dividend, divisor
                    )
                    with quotient, remainder:
                        assert_array_equivalent(
                            self, quotient, expected_quotient
                        )
                        assert_array_equivalent(
                            self, remainder, expected_remainder
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_pairwise_supported_dtype_promotion_matches_numpy_125(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for dividend_dtype in self.DTYPES:
                for divisor_dtype in self.DTYPES:
                    dividend_values = np.asarray(
                        [3, 5, 2], dtype=dividend_dtype
                    )
                    divisor_values = np.asarray([2, 1], dtype=divisor_dtype)
                    with np.errstate(all="ignore"):
                        expected_quotient, expected_remainder = np.polydiv(
                            dividend_values, divisor_values
                        )
                    with self.subTest(
                        dividend_dtype=np.dtype(dividend_dtype),
                        divisor_dtype=np.dtype(divisor_dtype),
                    ), runtime.from_numpy(
                        dividend_values
                    ) as dividend, runtime.from_numpy(
                        divisor_values
                    ) as divisor:
                        quotient, remainder = self._call(
                            runtime, dividend, divisor
                        )
                        with quotient, remainder:
                            assert_array_equivalent(
                                self, quotient, expected_quotient
                            )
                            assert_array_equivalent(
                                self, remainder, expected_remainder
                            )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_native_strided_inputs_match_numpy_125(self) -> None:
        dividend_base_values = np.asarray(
            [3.0, -7.0, 5.0, 11.0, 2.0, -13.0]
        )
        divisor_base_values = np.asarray([2.0, 17.0, 1.0, -19.0])
        expected_dividend = dividend_base_values[::2]
        expected_divisor = divisor_base_values[::2]
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            dividend_base = runtime.from_numpy(dividend_base_values)
            dividend_view = PolynomialConversionSemanticsTests._slice(
                runtime,
                dividend_base,
                _CnpSlice(0, 0, 2, True, False, True),
            )
            dividend_base.close()
            divisor_base = runtime.from_numpy(divisor_base_values)
            divisor_view = PolynomialConversionSemanticsTests._slice(
                runtime,
                divisor_base,
                _CnpSlice(0, 0, 2, True, False, True),
            )
            divisor_base.close()
            with dividend_view, divisor_view:
                quotient, remainder = self._call(
                    runtime, dividend_view, divisor_view
                )
                with quotient, remainder:
                    expected = np.polydiv(
                        expected_dividend, expected_divisor
                    )
                    assert_array_equivalent(self, quotient, expected[0])
                    assert_array_equivalent(self, remainder, expected[1])
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_seeded_complex_recurrence_matches_numpy_125(self) -> None:
        cases = (
            (
                np.asarray(
                    [
                        3221818534,
                        3199674720,
                        1046344564,
                        1006127525,
                        3196278616,
                        3197577966,
                        3215742603,
                        3219263458,
                    ],
                    dtype=np.uint32,
                ).view(np.complex64),
                np.asarray(
                    [1058057378, 1058820369, 1024874970, 3208897633],
                    dtype=np.uint32,
                ).view(np.complex64),
            ),
            (
                np.asarray(
                    [
                        3209699534,
                        1040437346,
                        1074086921,
                        1058270714,
                        1059271053,
                        3197416922,
                        3187734976,
                        3190003810,
                        3165995712,
                        1061350671,
                    ],
                    dtype=np.uint32,
                ).view(np.complex64),
                np.asarray(
                    [
                        3221637445,
                        3220524261,
                        3213401843,
                        3206894853,
                        1040746148,
                        3198754507,
                    ],
                    dtype=np.uint32,
                ).view(np.complex64),
            ),
            (
                np.asarray(
                    [
                        4603452474224048134,
                        4609036723109882382,
                        4607843002272975607,
                        4608334071614129221,
                        4607433127244640759,
                        4602723625179180531,
                        13809148472794462486,
                        13828865998995044503,
                    ],
                    dtype=np.uint64,
                ).view(np.complex128),
                np.asarray(
                    [
                        13823468647528337781,
                        4611755795167345064,
                        4609341786175476554,
                        13830738559621956174,
                    ],
                    dtype=np.uint64,
                ).view(np.complex128),
            ),
            (
                np.asarray(
                    [13832938032711398850, 13829679288862934650],
                    dtype=np.uint64,
                ).view(np.complex128),
                np.asarray(
                    [4609620466970636267, 13829674208738847329],
                    dtype=np.uint64,
                ).view(np.complex128),
            ),
            (
                np.asarray(
                    [
                        4586718142042955819,
                        13826359658236737679,
                        13836276344122873205,
                        13823519960832329413,
                        4607949439486649636,
                        13819505056822360744,
                        4599792006436843531,
                        4598638055819542703,
                        4598208032044960840,
                        13830101957651397168,
                    ],
                    dtype=np.uint64,
                ).view(np.complex128),
                np.asarray(
                    [
                        13817927310715539204,
                        13823031952437530345,
                        13833596299679104715,
                        13832196561856443300,
                        13835256614226025371,
                        4601942293389751841,
                    ],
                    dtype=np.uint64,
                ).view(np.complex128),
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for dividend_values, divisor_values in cases:
                expected = np.polydiv(dividend_values, divisor_values)
                with self.subTest(
                    dtype=dividend_values.dtype,
                    dividend_size=dividend_values.size,
                    divisor_size=divisor_values.size,
                ), runtime.from_numpy(
                    dividend_values
                ) as dividend, runtime.from_numpy(
                    divisor_values
                ) as divisor:
                    quotient, remainder = self._call(
                        runtime, dividend, divisor
                    )
                    with quotient, remainder:
                        assert_array_equivalent(
                            self, quotient, expected[0]
                        )
                        assert_array_equivalent(
                            self, remainder, expected[1]
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_invalid_inputs_and_output_slots_are_transactional(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            function = self._function(runtime)
            with runtime.from_numpy(np.asarray([1.0, 2.0])) as valid:
                for dividend_pointer, divisor_pointer in (
                    (None, valid.pointer),
                    (valid.pointer, None),
                ):
                    quotient_pointer = ctypes.c_void_p(1)
                    remainder_pointer = ctypes.c_void_p(1)
                    runtime.dll.cnp_clear_error()
                    status = int(
                        function(
                            dividend_pointer,
                            divisor_pointer,
                            ctypes.byref(quotient_pointer),
                            ctypes.byref(remainder_pointer),
                        )
                    )
                    self.assertEqual(-1, status)
                    self.assertFalse(quotient_pointer.value)
                    self.assertFalse(remainder_pointer.value)
                    state = runtime.error_state()
                    self.assertEqual("cnp_polydiv", state.function)
                    self.assertIn(
                        "polynomial inputs are required", state.message
                    )

                for quotient_slot, remainder_slot in (
                    (None, ctypes.pointer(ctypes.c_void_p())),
                    (ctypes.pointer(ctypes.c_void_p()), None),
                ):
                    runtime.dll.cnp_clear_error()
                    status = int(
                        function(
                            valid.pointer,
                            valid.pointer,
                            quotient_slot,
                            remainder_slot,
                        )
                    )
                    self.assertEqual(-1, status)
                    state = runtime.error_state()
                    self.assertEqual("cnp_polydiv", state.function)
                    self.assertIn("output slots are required", state.message)

                for invalid_values, message in (
                    (
                        np.asarray([], dtype=np.float64),
                        "polynomial inputs must not be empty",
                    ),
                    (
                        np.asarray([[1.0, 2.0], [3.0, 4.0]]),
                        "polynomial inputs must be one-dimensional",
                    ),
                ):
                    with runtime.from_numpy(invalid_values) as invalid:
                        quotient_pointer = ctypes.c_void_p(1)
                        remainder_pointer = ctypes.c_void_p(1)
                        runtime.dll.cnp_clear_error()
                        status = int(
                            function(
                                invalid.pointer,
                                valid.pointer,
                                ctypes.byref(quotient_pointer),
                                ctypes.byref(remainder_pointer),
                            )
                        )
                        self.assertEqual(-4, status)
                        self.assertFalse(quotient_pointer.value)
                        self.assertFalse(remainder_pointer.value)
                        state = runtime.error_state()
                        self.assertEqual("cnp_polydiv", state.function)
                        self.assertIn(message, state.message)
            self.assertEqual(baseline, runtime.retained_bytes)


class PowerPolynomialEvaluationSemanticsTests(unittest.TestCase):
    DTYPES = PowerPolynomialAdditionSemanticsTests.DTYPES

    @staticmethod
    def _call(runtime: CnumpyRuntime, coefficients, points):
        function = runtime.dll.cnp_polyval
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(
            function(coefficients.pointer, points.pointer), "cnp_polyval"
        )

    def test_values_shapes_dtypes_and_empty_coefficients_match_numpy_125(
        self,
    ) -> None:
        cases = (
            (
                np.asarray([3, 0, 1], dtype=np.int16),
                np.asarray(5, dtype=np.int32),
            ),
            (
                np.asarray([120, 120], dtype=np.int8),
                np.asarray([2, -2], dtype=np.int8),
            ),
            (
                np.asarray([1, np.iinfo(np.int64).max], dtype=np.int64),
                np.asarray([1, 2], dtype=np.int64),
            ),
            (
                np.asarray([1, -1], dtype=np.int64),
                np.asarray([0, 1], dtype=np.uint64),
            ),
            (
                np.asarray([2**63 + 5, 1], dtype=np.uint64),
                np.asarray([-1, 2], dtype=np.int64),
            ),
            (
                np.asarray([0.5, -1.25, 2.0], dtype=np.float32),
                np.asarray(
                    [[-2.0, -0.0, 0.5], [1.0, 2.0, 4.0]],
                    dtype=np.float32,
                ),
            ),
            (
                np.asarray(
                    [1.0 + 2.0j, -3.0 + 0.5j, 2.0 - 1.0j],
                    dtype=np.complex64,
                ),
                np.asarray([-1.5, 0.0, 2.25], dtype=np.float32),
            ),
            (
                np.asarray([1.0, np.nan, -np.inf], dtype=np.float64),
                np.asarray([-0.0, 1.0, np.inf], dtype=np.float64),
            ),
            (
                np.asarray([], dtype=np.complex64),
                np.asarray([[1.0, -0.0], [np.inf, np.nan]], dtype=np.float32),
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for coefficient_values, point_values in cases:
                with np.errstate(all="ignore"):
                    expected = np.polyval(coefficient_values, point_values)
                with self.subTest(
                    coefficient_dtype=coefficient_values.dtype,
                    point_dtype=point_values.dtype,
                    point_shape=point_values.shape,
                    coefficient_count=coefficient_values.size,
                ), runtime.from_numpy(
                    coefficient_values
                ) as coefficients, runtime.from_numpy(
                    point_values
                ) as points, self._call(
                    runtime, coefficients, points
                ) as actual:
                    assert_array_equivalent(self, actual, expected)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_pairwise_supported_dtype_promotion_matches_numpy_125(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for coefficient_dtype in self.DTYPES:
                for point_dtype in self.DTYPES:
                    coefficient_values = np.asarray(
                        [2, -1, 3]
                    ).astype(coefficient_dtype)
                    point_values = np.asarray(
                        [2, -1]
                    ).astype(point_dtype)
                    with np.errstate(all="ignore"):
                        expected = np.polyval(
                            coefficient_values, point_values
                        )
                    with self.subTest(
                        coefficient_dtype=np.dtype(coefficient_dtype),
                        point_dtype=np.dtype(point_dtype),
                    ), runtime.from_numpy(
                        coefficient_values
                    ) as coefficients, runtime.from_numpy(
                        point_values
                    ) as points, self._call(
                        runtime, coefficients, points
                    ) as actual:
                        assert_array_equivalent(self, actual, expected)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_native_strided_coefficients_and_points_match_numpy_125(
        self,
    ) -> None:
        coefficient_base_values = np.asarray(
            [2.0, -11.0, -1.0, 13.0, 3.0, -17.0], dtype=np.float64
        )
        point_base_values = np.asarray(
            [-2.0, 19.0, -0.0, -23.0, 0.5, 29.0, 3.0],
            dtype=np.float64,
        )
        expected_coefficients = coefficient_base_values[::2]
        expected_points = point_base_values[::-2]
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            coefficient_base = runtime.from_numpy(coefficient_base_values)
            coefficients = PolynomialConversionSemanticsTests._slice(
                runtime,
                coefficient_base,
                _CnpSlice(0, 0, 2, True, False, True),
            )
            coefficient_base.close()
            point_base = runtime.from_numpy(point_base_values)
            points = PolynomialConversionSemanticsTests._slice(
                runtime,
                point_base,
                _CnpSlice(0, 0, -2, False, False, True),
            )
            point_base.close()
            with coefficients, points, self._call(
                runtime, coefficients, points
            ) as actual:
                assert_array_equivalent(
                    self,
                    actual,
                    np.polyval(expected_coefficients, expected_points),
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_fortran_contiguous_points_preserve_numpy_output_layout(
        self,
    ) -> None:
        coefficient_values = np.asarray(
            [0.5, -1.25, 2.0], dtype=np.float64
        )
        point_base_values = np.arange(12, dtype=np.float64).reshape(4, 3)
        expected_points = point_base_values.T
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            point_base = runtime.from_numpy(point_base_values)
            points = runtime.transpose(point_base, (1, 0))
            point_base.close()
            dirty = runtime.from_numpy(
                np.full(expected_points.shape, np.nan, dtype=np.float64)
            )
            dirty.close()
            with runtime.from_numpy(
                coefficient_values
            ) as coefficients, points, self._call(
                runtime, coefficients, points
            ) as actual:
                assert_array_equivalent(
                    self,
                    actual,
                    np.polyval(coefficient_values, expected_points),
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_invalid_inputs_report_exact_native_errors_without_leaks(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            function = runtime.dll.cnp_polyval
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            function.restype = ctypes.c_void_p
            with runtime.from_numpy(np.asarray([1.0, 2.0])) as valid:
                for coefficient_pointer, point_pointer in (
                    (None, valid.pointer),
                    (valid.pointer, None),
                ):
                    runtime.dll.cnp_clear_error()
                    pointer = function(coefficient_pointer, point_pointer)
                    if pointer:
                        runtime.dll.cnp_array_decref(pointer)
                    self.assertFalse(pointer)
                    state = runtime.error_state()
                    self.assertEqual(-1, state.status)
                    self.assertEqual("cnp_polyval", state.function)
                    self.assertIn(
                        "coefficient and point arrays are required",
                        state.message,
                    )

                for invalid_values in (
                    np.asarray(3.0),
                    np.asarray([[1.0, 2.0], [3.0, 4.0]]),
                ):
                    with self.subTest(
                        coefficient_shape=invalid_values.shape
                    ), runtime.from_numpy(invalid_values) as invalid:
                        runtime.dll.cnp_clear_error()
                        pointer = function(invalid.pointer, valid.pointer)
                        if pointer:
                            runtime.dll.cnp_array_decref(pointer)
                        self.assertFalse(pointer)
                        state = runtime.error_state()
                        self.assertEqual(-4, state.status)
                        self.assertEqual("cnp_polyval", state.function)
                        self.assertIn(
                            "coefficient array must be one-dimensional",
                            state.message,
                        )
            self.assertEqual(baseline, runtime.retained_bytes)


class PolynomialBasisEvaluationSemanticsTests(unittest.TestCase):
    FUNCTIONS = (
        ("cnp_chebval", np.polynomial.chebyshev.chebval),
        ("cnp_legval", np.polynomial.legendre.legval),
        ("cnp_hermval", np.polynomial.hermite.hermval),
        ("cnp_lagval", np.polynomial.laguerre.lagval),
    )

    @staticmethod
    def _call(runtime: CnumpyRuntime, function_name: str, points, coefficients):
        function = getattr(runtime.dll, function_name)
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(
            function(points.pointer, coefficients.pointer), function_name
        )

    @staticmethod
    def _tolerance(expected: np.ndarray) -> float:
        if expected.dtype == np.dtype(np.float16):
            return 3e-3
        if expected.dtype in (np.dtype(np.float32), np.dtype(np.complex64)):
            return 3e-5
        return 2e-12

    def _assert_case(
        self,
        runtime: CnumpyRuntime,
        function_name: str,
        oracle,
        point_values: np.ndarray,
        coefficient_values: np.ndarray,
    ) -> None:
        expected = np.asarray(oracle(point_values, coefficient_values))
        with runtime.from_numpy(point_values) as points, runtime.from_numpy(
            coefficient_values
        ) as coefficients, self._call(
            runtime, function_name, points, coefficients
        ) as actual:
            tolerance = self._tolerance(expected)
            assert_array_equivalent(
                self,
                actual,
                expected,
                compare_strides=expected.size > 0,
                rtol=tolerance,
                atol=tolerance,
            )

    def test_scalar_vector_and_matrix_points_match_numpy_125(self) -> None:
        point_cases = (
            np.asarray(2.0, dtype=np.float32),
            np.asarray([-1.0, 0.0, 2.0], dtype=np.float32),
            np.arange(6, dtype=np.float64).reshape(2, 3),
        )
        coefficient_values = np.asarray([1.0, 2.0, 3.0])
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, oracle in self.FUNCTIONS:
                for point_values in point_cases:
                    with self.subTest(
                        function=function_name, shape=point_values.shape
                    ):
                        self._assert_case(
                            runtime,
                            function_name,
                            oracle,
                            point_values,
                            coefficient_values,
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_tensor_coefficients_and_empty_dimensions_match_numpy_125(
        self,
    ) -> None:
        cases = (
            (
                np.asarray([-1.0, 2.0]),
                np.asarray([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
            ),
            (
                np.arange(4, dtype=np.float32).reshape(2, 2),
                np.arange(6, dtype=np.float32).reshape(3, 2, 1),
            ),
            (
                np.asarray([1.0, 2.0]),
                np.empty((3, 0), dtype=np.float32),
            ),
            (
                np.asarray([], dtype=np.float32),
                np.ones((3, 2), dtype=np.float32),
            ),
            (
                np.asarray([-1.0, 2.0]),
                np.asarray(3, dtype=np.int16),
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, oracle in self.FUNCTIONS:
                for point_values, coefficient_values in cases:
                    with self.subTest(
                        function=function_name,
                        x_shape=point_values.shape,
                        c_shape=coefficient_values.shape,
                    ):
                        self._assert_case(
                            runtime,
                            function_name,
                            oracle,
                            point_values,
                            coefficient_values,
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_dtype_and_complex_promotion_match_numpy_125(self) -> None:
        cases = (
            (
                np.asarray([False, True], dtype=np.bool_),
                np.asarray([True, False, True], dtype=np.bool_),
            ),
            (
                np.asarray([0.0, 1.0], dtype=np.float16),
                np.asarray([1.0, 2.0, 3.0], dtype=np.float16),
            ),
            (
                np.asarray([1.0 + 2.0j], dtype=np.complex64),
                np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            ),
            (
                np.asarray([1.0, 2.0], dtype=np.float32),
                np.asarray([1.0 + 1.0j, 2.0j], dtype=np.complex64),
            ),
            (
                np.asarray([np.nan, np.inf, -0.0], dtype=np.float64),
                np.asarray([1.0, -2.0, 0.5], dtype=np.float64),
            ),
        )
        with np.errstate(all="ignore"), CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, oracle in self.FUNCTIONS:
                for point_values, coefficient_values in cases:
                    with self.subTest(
                        function=function_name,
                        x_dtype=point_values.dtype,
                        c_dtype=coefficient_values.dtype,
                    ):
                        self._assert_case(
                            runtime,
                            function_name,
                            oracle,
                            point_values,
                            coefficient_values,
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_strided_sources_survive_source_first_release(self) -> None:
        point_base_values = np.asarray(
            [-1.0, 91.0, 0.0, 92.0, 2.0], dtype=np.float64
        )
        coefficient_base_values = np.asarray(
            [1.0, 81.0, 2.0, 82.0, 3.0], dtype=np.float64
        )
        expected_points = point_base_values[::2]
        expected_coefficients = coefficient_base_values[::2]
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, oracle in self.FUNCTIONS:
                point_base = runtime.from_numpy(point_base_values)
                coefficient_base = runtime.from_numpy(coefficient_base_values)
                points = PolynomialConversionSemanticsTests._slice(
                    runtime,
                    point_base,
                    _CnpSlice(0, 0, 2, False, False, True),
                )
                coefficients = PolynomialConversionSemanticsTests._slice(
                    runtime,
                    coefficient_base,
                    _CnpSlice(0, 0, 2, False, False, True),
                )
                point_base.close()
                coefficient_base.close()
                try:
                    actual = self._call(
                        runtime, function_name, points, coefficients
                    )
                finally:
                    points.close()
                    coefficients.close()
                with self.subTest(function=function_name), actual:
                    expected = np.asarray(
                        oracle(expected_points, expected_coefficients)
                    )
                    assert_array_equivalent(
                        self,
                        actual,
                        expected,
                        compare_strides=True,
                        rtol=2e-12,
                        atol=2e-12,
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_invalid_inputs_raise_explicit_errors_without_leaks(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            create = runtime.dll.cnp_array_new
            create.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_int,
                ctypes.c_int,
            ]
            create.restype = ctypes.c_void_p
            string_shape = (ctypes.c_int64 * 1)(2)

            for function_name, _ in self.FUNCTIONS:
                function = getattr(runtime.dll, function_name)
                function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                function.restype = ctypes.c_void_p

                def assert_native_error(
                    point_pointer, coefficient_pointer, status: int, message: str
                ) -> None:
                    runtime.dll.cnp_clear_error()
                    unexpected = function(point_pointer, coefficient_pointer)
                    if unexpected:
                        runtime._owned_result(
                            unexpected, function_name
                        ).close()
                    self.assertFalse(unexpected)
                    state = runtime.error_state()
                    self.assertEqual(status, state.status)
                    self.assertEqual(function_name, state.function)
                    self.assertIn(message, state.message)

                with runtime.from_numpy(
                    np.asarray([1.0, 2.0])
                ) as points, runtime.from_numpy(
                    np.asarray([1.0, 2.0])
                ) as coefficients:
                    assert_native_error(
                        None,
                        coefficients.pointer,
                        -1,
                        "point array is required",
                    )
                    assert_native_error(
                        points.pointer,
                        None,
                        -1,
                        "coefficient array is required",
                    )
                    with runtime.from_numpy(
                        np.asarray([], dtype=np.float64)
                    ) as empty:
                        assert_native_error(
                            points.pointer,
                            empty.pointer,
                            -4,
                            "coefficient axis is empty",
                        )

                    runtime.dll.cnp_clear_error()
                    with runtime._owned_result(
                        create(1, string_shape, 19, 0), "cnp_array_new"
                    ) as strings:
                        assert_native_error(
                            strings.pointer,
                            coefficients.pointer,
                            -3,
                            "point dtype is not numeric",
                        )
                        assert_native_error(
                            points.pointer,
                            strings.pointer,
                            -3,
                            "coefficient dtype is not numeric",
                        )
            self.assertEqual(baseline, runtime.retained_bytes)


class PolynomialBasisFitSemanticsTests(unittest.TestCase):
    FUNCTIONS = (
        ("cnp_chebfit", np.polynomial.chebyshev.chebfit),
        ("cnp_legfit", np.polynomial.legendre.legfit),
        ("cnp_hermfit", np.polynomial.hermite.hermfit),
        ("cnp_lagfit", np.polynomial.laguerre.lagfit),
    )

    @staticmethod
    def _call(
        runtime: CnumpyRuntime, function_name: str, x, y, degree: int
    ):
        function = getattr(runtime.dll, function_name)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(
            function(x.pointer, y.pointer, degree), function_name
        )

    @staticmethod
    def _reference(oracle, x, y, degree: int):
        with np.testing.suppress_warnings() as warning_context:
            warning_context.filter(np.polynomial.polyutils.RankWarning)
            return oracle(x, y, degree)

    @staticmethod
    def _tolerance(expected: np.ndarray) -> float:
        if expected.dtype in (np.dtype(np.float32), np.dtype(np.complex64)):
            return 4e-5
        return 2e-9

    def _assert_fit_equivalent(
        self,
        actual,
        expected: np.ndarray,
        *,
        compare_strides: bool,
    ) -> None:
        expected = np.asarray(expected)
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected.dtype, actual.numpy_dtype)
        self.assertEqual(
            bool(expected.flags.c_contiguous), actual.c_contiguous
        )
        self.assertEqual(
            bool(expected.flags.f_contiguous), actual.f_contiguous
        )
        if compare_strides:
            self.assertEqual(expected.strides, actual.strides)
        tolerance = self._tolerance(expected)
        np.testing.assert_allclose(
            actual.to_numpy(),
            expected,
            rtol=tolerance,
            atol=tolerance,
            equal_nan=True,
        )

    def _assert_case(
        self,
        runtime: CnumpyRuntime,
        function_name: str,
        oracle,
        x_values: np.ndarray,
        y_values: np.ndarray,
        degree: int,
    ) -> None:
        expected = np.asarray(self._reference(oracle, x_values, y_values, degree))
        with runtime.from_numpy(x_values) as x, runtime.from_numpy(
            y_values
        ) as y, self._call(runtime, function_name, x, y, degree) as actual:
            self._assert_fit_equivalent(
                actual,
                expected,
                compare_strides=expected.size > 0,
            )

    def test_real_values_shapes_and_result_dtypes_match_numpy_125(self) -> None:
        cases = (
            (
                np.asarray([-2, -1, 0, 1, 2], dtype=np.int16),
                np.asarray([5, 2, 1, 2, 5], dtype=np.int32),
                2,
            ),
            (
                np.linspace(-1.0, 1.0, 9, dtype=np.float32),
                np.asarray(
                    [1.25, 0.75, 0.5, 0.4, 0.5, 0.9, 1.75, 3.2, 5.5],
                    dtype=np.float32,
                ),
                3,
            ),
            (
                np.linspace(-2.0, 2.0, 13, dtype=np.float64),
                np.sin(np.linspace(-2.0, 2.0, 13, dtype=np.float64)),
                5,
            ),
            (
                np.asarray([False, True, True, False], dtype=np.bool_),
                np.asarray([False, True, False, True], dtype=np.bool_),
                2,
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, oracle in self.FUNCTIONS:
                for x_values, y_values, degree in cases:
                    with self.subTest(
                        function=function_name,
                        x_dtype=x_values.dtype,
                        y_dtype=y_values.dtype,
                        degree=degree,
                    ):
                        self._assert_case(
                            runtime,
                            function_name,
                            oracle,
                            x_values,
                            y_values,
                            degree,
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_complex_multiple_and_empty_rhs_values_match_numpy_125(self) -> None:
        x_values = np.asarray(
            [
                -1.5 + 0.25j,
                -0.75 - 0.5j,
                0.0 + 0.125j,
                0.75 + 0.4j,
                1.5 - 0.2j,
                2.0 + 0.6j,
            ],
            dtype=np.complex64,
        )
        first = (
            (1.0 - 0.5j) * x_values**2
            + (-0.25 + 0.75j) * x_values
            + (2.0 + 0.125j)
        )
        second = (
            (-0.5 + 0.25j) * x_values**2
            + (1.5 - 0.4j) * x_values
            - (0.75 + 1.0j)
        )
        cases = (
            (x_values, np.column_stack((first, second)).astype(np.complex64), 2),
            (
                np.linspace(-1.0, 1.0, 6, dtype=np.float64),
                np.empty((6, 0), dtype=np.float64),
                3,
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, oracle in self.FUNCTIONS:
                for x_case, y_case, degree in cases:
                    with self.subTest(
                        function=function_name,
                        x_dtype=x_case.dtype,
                        y_shape=y_case.shape,
                    ):
                        self._assert_case(
                            runtime,
                            function_name,
                            oracle,
                            x_case,
                            y_case,
                            degree,
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_strided_inputs_survive_source_first_release(self) -> None:
        x_base_values = np.asarray(
            [
                -2.0,
                91.0,
                -1.0,
                92.0,
                0.0,
                93.0,
                1.0,
                94.0,
                2.0,
                95.0,
                3.0,
            ],
            dtype=np.float64,
        )
        y_base_values = np.asarray(
            [
                9.0,
                -71.0,
                2.0,
                -72.0,
                1.0,
                -73.0,
                6.0,
                -74.0,
                17.0,
                -75.0,
                34.0,
            ],
            dtype=np.float64,
        )
        expected_x = x_base_values[::2]
        expected_y = y_base_values[::2]
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, oracle in self.FUNCTIONS:
                x_base = runtime.from_numpy(x_base_values)
                y_base = runtime.from_numpy(y_base_values)
                x = PolynomialConversionSemanticsTests._slice(
                    runtime,
                    x_base,
                    _CnpSlice(0, 0, 2, False, False, True),
                )
                y = PolynomialConversionSemanticsTests._slice(
                    runtime,
                    y_base,
                    _CnpSlice(0, 0, 2, False, False, True),
                )
                x_base.close()
                y_base.close()
                try:
                    actual = self._call(runtime, function_name, x, y, 3)
                finally:
                    x.close()
                    y.close()
                with self.subTest(function=function_name), actual:
                    expected = np.asarray(
                        self._reference(oracle, expected_x, expected_y, 3)
                    )
                    self._assert_fit_equivalent(
                        actual,
                        expected,
                        compare_strides=True,
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_underdetermined_rank_deficient_and_nonfinite_rhs_match_numpy_125(
        self,
    ) -> None:
        cases = (
            (
                np.asarray([0.0, 1.0, 2.0]),
                np.asarray([1.0, 2.0, 5.0]),
                5,
            ),
            (
                np.asarray([1.0, 1.0, 1.0, 2.0]),
                np.asarray([2.0, 2.25, 1.75, 5.0]),
                3,
            ),
            (
                np.asarray([0.0, 1.0, 2.0, 3.0]),
                np.asarray([1.0, np.nan, 5.0, 10.0]),
                2,
            ),
            (
                np.asarray([0.0, 1.0, 2.0, 3.0]),
                np.asarray([1.0, np.inf, 5.0, 10.0]),
                2,
            ),
        )
        with np.errstate(all="ignore"), CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, oracle in self.FUNCTIONS:
                for x_values, y_values, degree in cases:
                    with self.subTest(function=function_name, degree=degree):
                        self._assert_case(
                            runtime,
                            function_name,
                            oracle,
                            x_values,
                            y_values,
                            degree,
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_invalid_inputs_report_exact_native_errors_without_leaks(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, _ in self.FUNCTIONS:
                function = getattr(runtime.dll, function_name)
                function.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                ]
                function.restype = ctypes.c_void_p
                with runtime.from_numpy(
                    np.asarray([0.0, 1.0, 2.0])
                ) as valid:
                    pointer_cases = (
                        (None, valid.pointer),
                        (valid.pointer, None),
                    )
                    for x_pointer, y_pointer in pointer_cases:
                        runtime.dll.cnp_clear_error()
                        pointer = function(x_pointer, y_pointer, 1)
                        if pointer:
                            runtime.dll.cnp_array_decref(pointer)
                        self.assertFalse(pointer)
                        state = runtime.error_state()
                        self.assertEqual(-1, state.status)
                        self.assertEqual(function_name, state.function)
                        self.assertIn(
                            "x and y arrays are required", state.message
                        )

                    runtime.dll.cnp_clear_error()
                    pointer = function(valid.pointer, valid.pointer, -1)
                    if pointer:
                        runtime.dll.cnp_array_decref(pointer)
                    self.assertFalse(pointer)
                    state = runtime.error_state()
                    self.assertEqual(-4, state.status)
                    self.assertEqual(function_name, state.function)
                    self.assertIn("degree must be nonnegative", state.message)

                array_cases = (
                    (
                        np.asarray([[0.0, 1.0], [2.0, 3.0]]),
                        np.asarray([1.0, 2.0, 3.0, 4.0]),
                        -4,
                        "x must be one-dimensional",
                    ),
                    (
                        np.asarray([], dtype=np.float64),
                        np.asarray([], dtype=np.float64),
                        -4,
                        "x must not be empty",
                    ),
                    (
                        np.asarray([0.0, 1.0, 2.0]),
                        np.asarray(1.0),
                        -4,
                        "y must be one- or two-dimensional",
                    ),
                    (
                        np.asarray([0.0, 1.0, 2.0]),
                        np.ones((3, 1, 1)),
                        -4,
                        "y must be one- or two-dimensional",
                    ),
                    (
                        np.asarray([0.0, 1.0, 2.0]),
                        np.asarray([1.0, 2.0]),
                        -4,
                        "x and y must have the same length",
                    ),
                    (
                        np.asarray([0.0, 1.0, 2.0], dtype=np.float16),
                        np.asarray([1.0, 2.0, 3.0], dtype=np.float16),
                        -3,
                        "float16 is not supported by linear algebra",
                    ),
                    (
                        np.asarray([0.0, np.nan, 2.0]),
                        np.asarray([1.0, 2.0, 3.0]),
                        -10,
                        "SVD did not converge in Linear Least Squares",
                    ),
                    (
                        np.asarray([0.0, np.inf, 2.0]),
                        np.asarray([1.0, 2.0, 3.0]),
                        -10,
                        "SVD did not converge in Linear Least Squares",
                    ),
                )
                for x_values, y_values, status, message in array_cases:
                    with self.subTest(
                        function=function_name, message=message
                    ), runtime.from_numpy(x_values) as x, runtime.from_numpy(
                        y_values
                    ) as y:
                        runtime.dll.cnp_clear_error()
                        pointer = function(x.pointer, y.pointer, 1)
                        if pointer:
                            runtime.dll.cnp_array_decref(pointer)
                        self.assertFalse(pointer)
                        state = runtime.error_state()
                        self.assertEqual(status, state.status)
                        self.assertEqual(function_name, state.function)
                        self.assertIn(message, state.message)
            self.assertEqual(baseline, runtime.retained_bytes)


class PowerPolynomialFitSemanticsTests(unittest.TestCase):
    @staticmethod
    def _call(runtime: CnumpyRuntime, x, y, degree: int):
        function = runtime.dll.cnp_polyfit
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(
            function(x.pointer, y.pointer, degree), "cnp_polyfit"
        )

    @staticmethod
    def _reference(x, y, degree: int):
        with np.testing.suppress_warnings() as warning_context:
            warning_context.filter(np.RankWarning)
            return np.polyfit(x, y, degree)

    def test_real_values_and_result_dtypes_match_numpy_125(self) -> None:
        cases = (
            (
                np.asarray([-2, -1, 0, 1, 2], dtype=np.int16),
                np.asarray([17, 6, 1, 2, 9], dtype=np.int32),
                2,
            ),
            (
                np.linspace(-1.0, 1.0, 9, dtype=np.float32),
                np.asarray(
                    [1.25, 0.75, 0.5, 0.4, 0.5, 0.9, 1.75, 3.2, 5.5],
                    dtype=np.float32,
                ),
                3,
            ),
            (
                np.linspace(-2.0, 2.0, 13, dtype=np.float64),
                np.sin(np.linspace(-2.0, 2.0, 13, dtype=np.float64)),
                5,
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for x_values, y_values, degree in cases:
                expected = self._reference(x_values, y_values, degree)
                with self.subTest(
                    x_dtype=x_values.dtype,
                    y_dtype=y_values.dtype,
                    degree=degree,
                ), runtime.from_numpy(x_values) as x, runtime.from_numpy(
                    y_values
                ) as y, self._call(runtime, x, y, degree) as actual:
                    assert_array_equivalent(
                        self,
                        actual,
                        expected,
                        rtol=2e-10,
                        atol=2e-10,
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_complex_and_multiple_rhs_values_match_numpy_125(self) -> None:
        x_values = np.asarray(
            [-1.5 + 0.25j, -0.75 - 0.5j, 0.0 + 0.125j,
             0.75 + 0.4j, 1.5 - 0.2j, 2.0 + 0.6j],
            dtype=np.complex64,
        )
        first = (
            (1.0 - 0.5j) * x_values**2 +
            (-0.25 + 0.75j) * x_values + (2.0 + 0.125j)
        )
        second = (
            (-0.5 + 0.25j) * x_values**2 +
            (1.5 - 0.4j) * x_values - (0.75 + 1.0j)
        )
        y_values = np.column_stack((first, second)).astype(np.complex64)
        expected = self._reference(x_values, y_values, 2)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(x_values) as x, runtime.from_numpy(
                y_values
            ) as y, self._call(runtime, x, y, 2) as actual:
                assert_array_equivalent(
                    self,
                    actual,
                    expected,
                    rtol=2e-7,
                    atol=2e-7,
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_strided_inputs_and_source_first_release_match_numpy_125(
        self,
    ) -> None:
        x_base_values = np.asarray(
            [-2.0, 91.0, -1.0, 92.0, 0.0, 93.0,
             1.0, 94.0, 2.0, 95.0, 3.0],
            dtype=np.float64,
        )
        y_base_values = np.asarray(
            [9.0, -71.0, 2.0, -72.0, 1.0, -73.0,
             6.0, -74.0, 17.0, -75.0, 34.0],
            dtype=np.float64,
        )
        expected_x = x_base_values[::2]
        expected_y = y_base_values[::2]
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            x_base = runtime.from_numpy(x_base_values)
            x = PolynomialConversionSemanticsTests._slice(
                runtime,
                x_base,
                _CnpSlice(0, 0, 2, True, False, True),
            )
            x_base.close()
            y_base = runtime.from_numpy(y_base_values)
            y = PolynomialConversionSemanticsTests._slice(
                runtime,
                y_base,
                _CnpSlice(0, 0, 2, True, False, True),
            )
            y_base.close()
            with x, y, self._call(runtime, x, y, 2) as actual:
                assert_array_equivalent(
                    self,
                    actual,
                    self._reference(expected_x, expected_y, 2),
                    rtol=2e-10,
                    atol=2e-10,
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_underdetermined_and_rank_deficient_fits_match_numpy_125(
        self,
    ) -> None:
        cases = (
            (
                np.asarray([0.0, 1.0, 2.0]),
                np.asarray([1.0, 2.0, 5.0]),
                5,
            ),
            (
                np.asarray([1.0, 1.0, 1.0, 2.0]),
                np.asarray([2.0, 2.25, 1.75, 5.0]),
                3,
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for x_values, y_values, degree in cases:
                expected = self._reference(x_values, y_values, degree)
                with self.subTest(degree=degree), runtime.from_numpy(
                    x_values
                ) as x, runtime.from_numpy(
                    y_values
                ) as y, self._call(runtime, x, y, degree) as actual:
                    assert_array_equivalent(
                        self,
                        actual,
                        expected,
                        rtol=5e-8,
                        atol=5e-8,
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_invalid_inputs_report_exact_native_errors_without_leaks(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            function = runtime.dll.cnp_polyfit
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            function.restype = ctypes.c_void_p
            with runtime.from_numpy(
                np.asarray([0.0, 1.0, 2.0])
            ) as valid:
                for x_pointer, y_pointer in (
                    (None, valid.pointer),
                    (valid.pointer, None),
                ):
                    runtime.dll.cnp_clear_error()
                    pointer = function(x_pointer, y_pointer, 1)
                    if pointer:
                        runtime.dll.cnp_array_decref(pointer)
                    self.assertFalse(pointer)
                    state = runtime.error_state()
                    self.assertEqual(-1, state.status)
                    self.assertEqual("cnp_polyfit", state.function)
                    self.assertIn("x and y arrays are required", state.message)

                invalid_cases = (
                    (-1, valid, valid, -4, "degree must be nonnegative"),
                )
                for degree, x, y, status, message in invalid_cases:
                    runtime.dll.cnp_clear_error()
                    pointer = function(x.pointer, y.pointer, degree)
                    if pointer:
                        runtime.dll.cnp_array_decref(pointer)
                    self.assertFalse(pointer)
                    state = runtime.error_state()
                    self.assertEqual(status, state.status)
                    self.assertEqual("cnp_polyfit", state.function)
                    self.assertIn(message, state.message)

                array_cases = (
                    (
                        np.asarray([[0.0, 1.0], [2.0, 3.0]]),
                        np.asarray([1.0, 2.0, 3.0, 4.0]),
                        -4,
                        "x must be one-dimensional",
                    ),
                    (
                        np.asarray([], dtype=np.float64),
                        np.asarray([], dtype=np.float64),
                        -4,
                        "x must not be empty",
                    ),
                    (
                        np.asarray([0.0, 1.0, 2.0]),
                        np.asarray(1.0),
                        -4,
                        "y must be one- or two-dimensional",
                    ),
                    (
                        np.asarray([0.0, 1.0, 2.0]),
                        np.ones((3, 1, 1)),
                        -4,
                        "y must be one- or two-dimensional",
                    ),
                    (
                        np.asarray([0.0, 1.0, 2.0]),
                        np.asarray([1.0, 2.0]),
                        -4,
                        "x and y must have the same length",
                    ),
                    (
                        np.asarray([0.0, 1.0, 2.0], dtype=np.float16),
                        np.asarray([1.0, 2.0, 3.0], dtype=np.float16),
                        -3,
                        "float16 is not supported by linear algebra",
                    ),
                )
                for x_values, y_values, status, message in array_cases:
                    with self.subTest(message=message), runtime.from_numpy(
                        x_values
                    ) as x, runtime.from_numpy(y_values) as y:
                        runtime.dll.cnp_clear_error()
                        pointer = function(x.pointer, y.pointer, 1)
                        if pointer:
                            runtime.dll.cnp_array_decref(pointer)
                        self.assertFalse(pointer)
                        state = runtime.error_state()
                        self.assertEqual(status, state.status)
                        self.assertEqual("cnp_polyfit", state.function)
                        self.assertIn(message, state.message)
            self.assertEqual(baseline, runtime.retained_bytes)


class PowerPolynomialConstructionSemanticsTests(unittest.TestCase):
    @staticmethod
    def _call(runtime: CnumpyRuntime, roots):
        function = runtime.dll.cnp_poly
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(function(roots.pointer), "cnp_poly")

    def test_real_roots_are_returned_in_descending_power_order(
        self,
    ) -> None:
        root_values = np.asarray([1, 2, 3], dtype=np.int16)
        expected = np.poly(root_values)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(root_values) as roots, self._call(
                runtime, roots
            ) as actual:
                assert_array_equivalent(self, actual, expected)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_complex_roots_preserve_numpy_dtype_and_values(self) -> None:
        root_values = np.asarray(
            [1.0 + 2.0j, -0.5 + 0.25j, 3.0 - 1.0j],
            dtype=np.complex64,
        )
        expected = np.poly(root_values)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(root_values) as roots, self._call(
                runtime, roots
            ) as actual:
                assert_array_equivalent(
                    self, actual, expected, rtol=2e-6, atol=2e-6
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_mintypecode_and_conjugate_pairs_determine_result_dtype(
        self,
    ) -> None:
        cases = (
            np.asarray([1.5, -2.0, 0.25], dtype=np.float32),
            np.asarray([1.5, -2.0, 0.25], dtype=np.float16),
            np.asarray([1.0 + 2.0j, 1.0 - 2.0j], dtype=np.complex64),
            np.asarray([1.0 + 0.0j, -3.0 + 0.0j], dtype=np.complex128),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for root_values in cases:
                expected = np.poly(root_values)
                with self.subTest(dtype=root_values.dtype), runtime.from_numpy(
                    root_values
                ) as roots, self._call(runtime, roots) as actual:
                    assert_array_equivalent(
                        self, actual, expected, rtol=2e-6, atol=2e-6
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_nan_complex_roots_do_not_collapse_to_real_dtype(self) -> None:
        root_values = np.asarray(
            [complex(np.nan, 2.0), 1.0 - 2.0j], dtype=np.complex64
        )
        expected = np.poly(root_values)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(root_values) as roots, self._call(
                runtime, roots
            ) as actual:
                assert_array_equivalent(
                    self, actual, expected, rtol=2e-6, atol=2e-6
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_empty_roots_return_numpy_scalar_shape(self) -> None:
        root_values = np.asarray([], dtype=np.complex64)
        expected = np.poly(root_values)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(root_values) as roots, self._call(
                runtime, roots
            ) as actual:
                assert_array_equivalent(self, actual, expected)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_real_square_matrix_uses_its_eigenvalues_as_roots(self) -> None:
        matrix_values = np.asarray(
            [[1.0, 2.0], [3.0, 4.0]], dtype=np.float64
        )
        expected = np.poly(matrix_values)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(matrix_values) as matrix, self._call(
                runtime, matrix
            ) as actual:
                assert_array_equivalent(
                    self, actual, expected, rtol=4e-14, atol=4e-14
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_real_matrix_with_complex_eigenvalues_returns_real_dtype(
        self,
    ) -> None:
        matrix_values = np.asarray(
            [[0.0, -1.0], [1.0, 0.0]], dtype=np.float64
        )
        expected = np.poly(matrix_values)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(matrix_values) as matrix, self._call(
                runtime, matrix
            ) as actual:
                assert_array_equivalent(
                    self, actual, expected, rtol=4e-14, atol=4e-14
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_complex64_matrix_result_outlives_its_source(self) -> None:
        matrix_values = np.asarray(
            [[1.0 + 2.0j, 0.0], [0.0, -0.5 + 0.25j]],
            dtype=np.complex64,
        )
        expected = np.poly(matrix_values)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            matrix = runtime.from_numpy(matrix_values)
            try:
                actual = self._call(runtime, matrix)
            finally:
                matrix.close()
            with actual:
                assert_array_equivalent(
                    self, actual, expected, rtol=3e-5, atol=3e-5
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_strided_roots_and_source_first_release_match_numpy_125(
        self,
    ) -> None:
        base_values = np.asarray(
            [
                1.0 + 2.0j,
                91.0 - 17.0j,
                -0.5 + 0.25j,
                92.0 - 18.0j,
                3.0 - 1.0j,
            ],
            dtype=np.complex64,
        )
        expected = np.poly(base_values[::-2])
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            base = runtime.from_numpy(base_values)
            roots = PolynomialConversionSemanticsTests._slice(
                runtime,
                base,
                _CnpSlice(0, 0, -2, False, False, True),
            )
            base.close()
            try:
                actual = self._call(runtime, roots)
            finally:
                roots.close()
            with actual:
                assert_array_equivalent(
                    self, actual, expected, rtol=3e-5, atol=3e-5
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_invalid_inputs_raise_explicit_errors_without_leaks(self) -> None:
        invalid_shapes = (
            np.asarray(1.0, dtype=np.float64),
            np.ones((2, 3), dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
            np.ones((1, 2, 2), dtype=np.float64),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            function = runtime.dll.cnp_poly
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_void_p

            runtime.dll.cnp_clear_error()
            with self.assertRaises(CnumpyError) as raised:
                runtime._owned_result(function(None), "cnp_poly")
            self.assertEqual(-1, raised.exception.status)
            self.assertEqual("cnp_poly", raised.exception.function)
            self.assertIn("input array is required", raised.exception.message)

            for values in invalid_shapes:
                with self.subTest(shape=values.shape), runtime.from_numpy(
                    values
                ) as source, self.assertRaises(CnumpyError) as raised:
                    self._call(runtime, source)
                self.assertEqual(-4, raised.exception.status)
                self.assertEqual("cnp_poly", raised.exception.function)
                self.assertIn(
                    "input must be 1d or non-empty square 2d array",
                    raised.exception.message,
                )

            create = runtime.dll.cnp_array_new
            create.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_int,
                ctypes.c_int,
            ]
            create.restype = ctypes.c_void_p
            shape = (ctypes.c_int64 * 1)(2)
            runtime.dll.cnp_clear_error()
            with runtime._owned_result(
                create(1, shape, 19, 0), "cnp_array_new"
            ) as strings, self.assertRaises(CnumpyError) as raised:
                self._call(runtime, strings)
            self.assertEqual(-3, raised.exception.status)
            self.assertEqual("cnp_poly", raised.exception.function)
            self.assertIn("root dtype is not numeric", raised.exception.message)
            self.assertEqual(baseline, runtime.retained_bytes)


class PowerPolynomialFromRootsSemanticsTests(unittest.TestCase):
    @staticmethod
    def _call(runtime: CnumpyRuntime, roots):
        function = runtime.dll.cnp_polyfromroots
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(
            function(roots.pointer), "cnp_polyfromroots"
        )

    def test_real_roots_return_ascending_power_coefficients(self) -> None:
        root_values = np.asarray([1, 2, 3], dtype=np.int16)
        expected = np.polynomial.polynomial.polyfromroots(root_values)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(root_values) as roots, self._call(
                runtime, roots
            ) as actual:
                assert_array_equivalent(self, actual, expected)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_empty_roots_return_one_element_float64_identity(self) -> None:
        root_values = np.asarray([], dtype=np.complex64)
        expected = np.polynomial.polynomial.polyfromroots(root_values)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(root_values) as roots, self._call(
                runtime, roots
            ) as actual:
                assert_array_equivalent(self, actual, expected)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_result_dtype_uses_numpy_polynomial_common_type(self) -> None:
        cases = (
            np.asarray([1.5, -2.0, 0.25], dtype=np.float32),
            np.asarray([1.5, -2.0, 0.25], dtype=np.float16),
            np.asarray(
                [1.0 + 2.0j, -0.5 + 0.25j], dtype=np.complex64
            ),
            np.asarray(
                [1.0 + 2.0j, -0.5 + 0.25j], dtype=np.complex128
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for root_values in cases:
                expected = np.polynomial.polynomial.polyfromroots(root_values)
                with self.subTest(dtype=root_values.dtype), runtime.from_numpy(
                    root_values
                ) as roots, self._call(runtime, roots) as actual:
                    assert_array_equivalent(
                        self, actual, expected, rtol=4e-14, atol=4e-14
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_empty_first_dimension_precedes_rank_and_dtype_validation(
        self,
    ) -> None:
        empty_matrix = np.empty((0, 2), dtype=np.float32)
        expected_matrix = np.polynomial.polynomial.polyfromroots(empty_matrix)
        expected_strings = np.polynomial.polynomial.polyfromroots(
            np.asarray([], dtype="S1")
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(empty_matrix) as roots, self._call(
                runtime, roots
            ) as actual:
                assert_array_equivalent(self, actual, expected_matrix)

            create = runtime.dll.cnp_array_new
            create.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_int,
                ctypes.c_int,
            ]
            create.restype = ctypes.c_void_p
            shape = (ctypes.c_int64 * 1)(0)
            runtime.dll.cnp_clear_error()
            with runtime._owned_result(
                create(1, shape, 19, 0), "cnp_array_new"
            ) as string_roots, self._call(
                runtime, string_roots
            ) as actual:
                assert_array_equivalent(self, actual, expected_strings)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_nan_infinity_and_signed_zero_match_numpy_125(self) -> None:
        cases = (
            np.asarray([np.nan], dtype=np.float32),
            np.asarray([np.inf, -np.inf], dtype=np.float64),
            np.asarray([0.0, -0.0, 0.0], dtype=np.float64),
            np.asarray(
                [complex(np.nan, 1.0), 2.0j], dtype=np.complex64
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for root_values in cases:
                expected = np.polynomial.polynomial.polyfromroots(root_values)
                with self.subTest(values=root_values), runtime.from_numpy(
                    root_values
                ) as roots, self._call(runtime, roots) as actual:
                    assert_array_equivalent(
                        self, actual, expected, rtol=4e-14, atol=4e-14
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_sorted_pairwise_multiplication_matches_numpy_125_rounding(
        self,
    ) -> None:
        root_values = np.asarray(
            [
                1e-8,
                -1e8,
                3.25,
                -2.5,
                0.125,
                -0.75,
                11.0,
                -13.0,
            ],
            dtype=np.float64,
        )
        expected = np.polynomial.polynomial.polyfromroots(root_values)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(root_values) as roots, self._call(
                runtime, roots
            ) as actual:
                assert_array_equivalent(self, actual, expected)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_strided_roots_and_source_first_release_match_numpy_125(
        self,
    ) -> None:
        base_values = np.asarray(
            [1.5, 91.0, -2.0, 92.0, 0.25, 93.0, 3.0],
            dtype=np.float32,
        )
        expected = np.polynomial.polynomial.polyfromroots(base_values[::-2])
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            base = runtime.from_numpy(base_values)
            roots = PolynomialConversionSemanticsTests._slice(
                runtime,
                base,
                _CnpSlice(0, 0, -2, False, False, True),
            )
            base.close()
            try:
                actual = self._call(runtime, roots)
            finally:
                roots.close()
            with actual:
                assert_array_equivalent(self, actual, expected)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_invalid_inputs_raise_explicit_errors_without_leaks(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            function = runtime.dll.cnp_polyfromroots
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_void_p

            def assert_native_error(pointer, status: int, message: str) -> None:
                runtime.dll.cnp_clear_error()
                unexpected = function(pointer)
                if unexpected:
                    runtime._owned_result(
                        unexpected, "cnp_polyfromroots"
                    ).close()
                self.assertFalse(unexpected)
                state = runtime.error_state()
                self.assertEqual(status, state.status)
                self.assertEqual("cnp_polyfromroots", state.function)
                self.assertIn(message, state.message)

            assert_native_error(None, -1, "root array is required")
            with runtime.from_numpy(np.asarray(2.0)) as scalar:
                assert_native_error(
                    scalar.pointer, -4, "must be one-dimensional"
                )
            with runtime.from_numpy(np.ones((2, 1))) as rank_two:
                assert_native_error(
                    rank_two.pointer, -4, "must be one-dimensional"
                )
            with runtime.from_numpy(
                np.asarray([True, False], dtype=np.bool_)
            ) as boolean:
                assert_native_error(
                    boolean.pointer, -3,
                    "Coefficient arrays have no common type",
                )

            create = runtime.dll.cnp_array_new
            create.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_int,
                ctypes.c_int,
            ]
            create.restype = ctypes.c_void_p
            shape = (ctypes.c_int64 * 1)(2)
            runtime.dll.cnp_clear_error()
            with runtime._owned_result(
                create(1, shape, 19, 0), "cnp_array_new"
            ) as strings:
                assert_native_error(
                    strings.pointer, -3,
                    "Coefficient arrays have no common type",
                )
            self.assertEqual(baseline, runtime.retained_bytes)


class PowerPolynomialRootsSemanticsTests(unittest.TestCase):
    @staticmethod
    def _call(runtime: CnumpyRuntime, coefficients):
        function = runtime.dll.cnp_polyroots
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(
            function(coefficients.pointer), "cnp_polyroots"
        )

    def test_ascending_coefficients_return_sorted_real_roots(self) -> None:
        coefficient_values = np.asarray([2.0, -3.0, 1.0], dtype=np.float64)
        expected = np.polynomial.polynomial.polyroots(coefficient_values)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(coefficient_values) as coefficients, self._call(
                runtime, coefficients
            ) as actual:
                assert_array_equivalent(
                    self, actual, expected, rtol=2e-13, atol=2e-13
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_trims_high_degree_zeros_and_preserves_zero_roots(self) -> None:
        cases = (
            np.asarray([2, 4, 0, 0], dtype=np.int16),
            np.asarray([0, 0, 2, 1], dtype=np.int16),
            np.asarray([1.0, 2.0, -0.0], dtype=np.float64),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for coefficient_values in cases:
                expected = np.polynomial.polynomial.polyroots(
                    coefficient_values
                )
                with self.subTest(values=coefficient_values), runtime.from_numpy(
                    coefficient_values
                ) as coefficients, self._call(runtime, coefficients) as actual:
                    assert_array_equivalent(
                        self, actual, expected, rtol=2e-13, atol=2e-13
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_scalar_constant_and_zero_series_return_typed_empty_array(
        self,
    ) -> None:
        cases = (
            np.asarray(2, dtype=np.int16),
            np.asarray([3.0], dtype=np.float32),
            np.asarray([0.0, 0.0], dtype=np.float64),
            np.asarray([2.0 + 1.0j], dtype=np.complex64),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for coefficient_values in cases:
                expected = np.polynomial.polynomial.polyroots(
                    coefficient_values
                )
                with self.subTest(values=coefficient_values), runtime.from_numpy(
                    coefficient_values
                ) as coefficients, self._call(runtime, coefficients) as actual:
                    assert_array_equivalent(self, actual, expected)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_common_type_and_complex_roots_match_numpy_125(self) -> None:
        cases = (
            np.asarray([1, -3, 2], dtype=np.int16),
            np.asarray([1, 2], dtype=np.float16),
            np.asarray([1, -3, 2], dtype=np.float32),
            np.asarray([1, -3, 2], dtype=np.float64),
            np.asarray([1, 0, 1], dtype=np.float32),
            np.asarray([1 + 1j, -2j, 1], dtype=np.complex64),
            np.asarray([1 + 1j, -2j, 1], dtype=np.complex128),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for coefficient_values in cases:
                expected = np.polynomial.polynomial.polyroots(
                    coefficient_values
                )
                tolerance = (
                    4e-5
                    if expected.dtype in (np.dtype(np.float32), np.dtype(np.complex64))
                    else 2e-12
                )
                with self.subTest(dtype=coefficient_values.dtype), runtime.from_numpy(
                    coefficient_values
                ) as coefficients, self._call(runtime, coefficients) as actual:
                    assert_array_equivalent(
                        self,
                        actual,
                        expected,
                        rtol=tolerance,
                        atol=tolerance,
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_strided_coefficients_survive_source_first_release(self) -> None:
        base_values = np.asarray(
            [1.0, 91.0, -3.0, 92.0, 2.0], dtype=np.float64
        )
        expected = np.polynomial.polynomial.polyroots(base_values[::-2])
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            base = runtime.from_numpy(base_values)
            coefficients = PolynomialConversionSemanticsTests._slice(
                runtime,
                base,
                _CnpSlice(0, 0, -2, False, False, True),
            )
            base.close()
            try:
                actual = self._call(runtime, coefficients)
            finally:
                coefficients.close()
            with actual:
                assert_array_equivalent(
                    self, actual, expected, rtol=2e-13, atol=2e-13
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_nonfinite_companion_behavior_matches_numpy_125(self) -> None:
        finite_after_normalization = np.asarray(
            [1.0, 2.0, np.inf], dtype=np.float64
        )
        with np.errstate(all="ignore"):
            expected = np.polynomial.polynomial.polyroots(
                finite_after_normalization
            )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(
                finite_after_normalization
            ) as coefficients, self._call(runtime, coefficients) as actual:
                assert_array_equivalent(self, actual, expected)

            invalid = np.asarray([1.0, np.nan, 1.0], dtype=np.float64)
            with runtime.from_numpy(invalid) as coefficients:
                function = runtime.dll.cnp_polyroots
                function.argtypes = [ctypes.c_void_p]
                function.restype = ctypes.c_void_p
                runtime.dll.cnp_clear_error()
                unexpected = function(coefficients.pointer)
                if unexpected:
                    runtime._owned_result(unexpected, "cnp_polyroots").close()
                self.assertFalse(unexpected)
                state = runtime.error_state()
                self.assertEqual(-10, state.status)
                self.assertEqual("cnp_polyroots", state.function)
                self.assertIn("NaN or infinity", state.message)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_real_spectrum_classification_uses_polynomial_residuals(
        self,
    ) -> None:
        cases = (
            np.asarray(
                [
                    0.53771454,
                    -5.35028168,
                    2.29305045,
                    6.61123001,
                    -3.77182492,
                    -1.62585489,
                    1.0,
                ],
                dtype=np.float64,
            ),
            np.asarray([1e-24, 0.0, 1.0], dtype=np.float64),
            np.asarray([1e-300, 0.0, 1.0], dtype=np.float64),
            np.asarray([1e300, 0.0, 1.0], dtype=np.float64),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for coefficient_values in cases:
                expected = np.polynomial.polynomial.polyroots(
                    coefficient_values
                )
                absolute_tolerance = max(
                    float(np.max(np.abs(expected))) * 3e-8,
                    np.finfo(np.float64).tiny,
                )
                with self.subTest(values=coefficient_values), runtime.from_numpy(
                    coefficient_values
                ) as coefficients, self._call(runtime, coefficients) as actual:
                    assert_array_equivalent(
                        self,
                        actual,
                        expected,
                        rtol=3e-8,
                        atol=absolute_tolerance,
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_invalid_inputs_raise_explicit_errors_without_leaks(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            function = runtime.dll.cnp_polyroots
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_void_p

            def assert_native_error(pointer, status: int, message: str) -> None:
                runtime.dll.cnp_clear_error()
                unexpected = function(pointer)
                if unexpected:
                    runtime._owned_result(unexpected, "cnp_polyroots").close()
                self.assertFalse(unexpected)
                state = runtime.error_state()
                self.assertEqual(status, state.status)
                self.assertEqual("cnp_polyroots", state.function)
                self.assertIn(message, state.message)

            assert_native_error(None, -1, "coefficient array is required")
            with runtime.from_numpy(
                np.asarray([], dtype=np.float32)
            ) as empty:
                assert_native_error(
                    empty.pointer, -4, "Coefficient array is empty"
                )
            with runtime.from_numpy(np.ones((1, 2))) as rank_two:
                assert_native_error(
                    rank_two.pointer, -4, "Coefficient array is not 1-d"
                )
            with runtime.from_numpy(
                np.asarray([True, False], dtype=np.bool_)
            ) as boolean:
                assert_native_error(
                    boolean.pointer,
                    -3,
                    "Coefficient arrays have no common type",
                )
            with runtime.from_numpy(
                np.asarray([1, 2, 3], dtype=np.float16)
            ) as half_degree_two:
                assert_native_error(
                    half_degree_two.pointer,
                    -3,
                    "array type float16 is unsupported in linalg",
                )

            create = runtime.dll.cnp_array_new
            create.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_int,
                ctypes.c_int,
            ]
            create.restype = ctypes.c_void_p
            shape = (ctypes.c_int64 * 1)(2)
            runtime.dll.cnp_clear_error()
            with runtime._owned_result(
                create(1, shape, 19, 0), "cnp_array_new"
            ) as strings:
                assert_native_error(
                    strings.pointer,
                    -3,
                    "Coefficient arrays have no common type",
                )
            self.assertEqual(baseline, runtime.retained_bytes)


class LegacyPowerPolynomialRootsSemanticsTests(unittest.TestCase):
    @staticmethod
    def _call(runtime: CnumpyRuntime, coefficients):
        function = runtime.dll.cnp_roots
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(
            function(coefficients.pointer), "cnp_roots"
        )

    def test_descending_coefficients_trim_leading_zeros_and_append_zero_roots(
        self,
    ) -> None:
        cases = (
            np.asarray([1.0, -3.0, 2.0], dtype=np.float64),
            np.asarray([0, 0, 1, -3, 2], dtype=np.int16),
            np.asarray([1, -3, 2, 0, 0], dtype=np.int16),
            np.asarray([1.0, 2.0, -0.0], dtype=np.float64),
            np.asarray([0.0, 0.0, 2.0, 0.0, 0.0], dtype=np.float64),
            np.asarray([2.0, 0.0], dtype=np.float32),
            np.asarray([2.0 + 1.0j, 0.0, 0.0], dtype=np.complex64),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for coefficient_values in cases:
                expected = np.roots(coefficient_values)
                with self.subTest(values=coefficient_values), runtime.from_numpy(
                    coefficient_values
                ) as coefficients, self._call(runtime, coefficients) as actual:
                    assert_array_equivalent(
                        self,
                        actual,
                        expected,
                        compare_strides=True,
                        rtol=2e-13,
                        atol=2e-13,
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_empty_constant_and_all_zero_inputs_return_float64_empty_array(
        self,
    ) -> None:
        cases = (
            np.asarray([], dtype=np.int16),
            np.asarray([], dtype=np.float32),
            np.asarray([2], dtype=np.int16),
            np.asarray([2.0 + 1.0j], dtype=np.complex64),
            np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for coefficient_values in cases:
                expected = np.roots(coefficient_values)
                with self.subTest(values=coefficient_values), runtime.from_numpy(
                    coefficient_values
                ) as coefficients, self._call(runtime, coefficients) as actual:
                    assert_array_equivalent(
                        self, actual, expected, compare_strides=True
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_numeric_dtype_and_real_or_complex_result_match_numpy_125(
        self,
    ) -> None:
        cases = (
            np.asarray([True, False, True], dtype=np.bool_),
            np.asarray([1, -3, 2], dtype=np.int16),
            np.asarray([1, -3, 2], dtype=np.float32),
            np.asarray([1, 0, 1], dtype=np.float32),
            np.asarray([1, -3, 2], dtype=np.float64),
            np.asarray([1 + 1j, -2j, 1], dtype=np.complex64),
            np.asarray([1 + 1j, -2j, 1], dtype=np.complex128),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for coefficient_values in cases:
                expected = np.roots(coefficient_values)
                tolerance = (
                    4e-5
                    if expected.dtype in (
                        np.dtype(np.float32),
                        np.dtype(np.complex64),
                    )
                    else 2e-12
                )
                with self.subTest(dtype=coefficient_values.dtype), runtime.from_numpy(
                    coefficient_values
                ) as coefficients, self._call(runtime, coefficients) as actual:
                    assert_array_equivalent(
                        self,
                        actual,
                        expected,
                        compare_strides=True,
                        rtol=tolerance,
                        atol=tolerance,
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_strided_coefficients_survive_source_first_release(self) -> None:
        base_values = np.asarray(
            [1.0, 91.0, -3.0, 92.0, 2.0], dtype=np.float64
        )
        expected = np.roots(base_values[::2])
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            base = runtime.from_numpy(base_values)
            coefficients = PolynomialConversionSemanticsTests._slice(
                runtime,
                base,
                _CnpSlice(0, 0, 2, False, False, True),
            )
            base.close()
            try:
                actual = self._call(runtime, coefficients)
            finally:
                coefficients.close()
            with actual:
                assert_array_equivalent(
                    self,
                    actual,
                    expected,
                    compare_strides=True,
                    rtol=2e-13,
                    atol=2e-13,
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_nonfinite_companion_behavior_matches_numpy_125(self) -> None:
        finite_after_normalization_cases = (
            np.asarray([np.inf, 2.0, 1.0], dtype=np.float64),
            np.asarray([np.inf, -2.0, 1.0], dtype=np.float64),
            np.asarray([-np.inf, 2.0, 1.0], dtype=np.float64),
            np.asarray([np.inf, 2.0, 1.0], dtype=np.float32),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for coefficient_values in finite_after_normalization_cases:
                with np.errstate(all="ignore"):
                    expected = np.roots(coefficient_values)
                with self.subTest(values=coefficient_values), runtime.from_numpy(
                    coefficient_values
                ) as coefficients, self._call(runtime, coefficients) as actual:
                    assert_array_equivalent(
                        self,
                        actual,
                        expected,
                        compare_strides=True,
                    )

            function = runtime.dll.cnp_roots
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_void_p
            for invalid in (
                np.asarray([1.0, np.nan, 1.0], dtype=np.float64),
                np.asarray([1.0, 2.0, np.inf], dtype=np.float64),
                np.asarray([1.0, np.nan], dtype=np.float64),
                np.asarray([np.nan, 1.0], dtype=np.float64),
                np.asarray([np.inf, np.nan, 0.0], dtype=np.float64),
                np.asarray(
                    [1.0 + 0.0j, np.nan + 0.0j],
                    dtype=np.complex128,
                ),
            ):
                with self.subTest(values=invalid), runtime.from_numpy(
                    invalid
                ) as coefficients:
                    runtime.dll.cnp_clear_error()
                    unexpected = function(coefficients.pointer)
                    if unexpected:
                        runtime._owned_result(unexpected, "cnp_roots").close()
                    self.assertFalse(unexpected)
                    state = runtime.error_state()
                    self.assertEqual(-10, state.status)
                    self.assertEqual("cnp_roots", state.function)
                    self.assertIn("NaN or infinity", state.message)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_invalid_inputs_raise_explicit_errors_without_leaks(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            function = runtime.dll.cnp_roots
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_void_p

            def assert_native_error(pointer, status: int, message: str) -> None:
                runtime.dll.cnp_clear_error()
                unexpected = function(pointer)
                if unexpected:
                    runtime._owned_result(unexpected, "cnp_roots").close()
                self.assertFalse(unexpected)
                state = runtime.error_state()
                self.assertEqual(status, state.status)
                self.assertEqual("cnp_roots", state.function)
                self.assertIn(message, state.message)

            assert_native_error(None, -1, "coefficient array is required")
            with runtime.from_numpy(np.asarray(2.0)) as scalar:
                assert_native_error(
                    scalar.pointer, -4, "Input must be a rank-1 array"
                )
            with runtime.from_numpy(np.ones((1, 2))) as rank_two:
                assert_native_error(
                    rank_two.pointer, -4, "Input must be a rank-1 array"
                )
            with runtime.from_numpy(
                np.asarray([1.0, 2.0], dtype=np.float16)
            ) as half:
                assert_native_error(
                    half.pointer,
                    -3,
                    "array type float16 is unsupported in linalg",
                )
            self.assertEqual(baseline, runtime.retained_bytes)


class ChebyshevPointSemanticsTests(unittest.TestCase):
    @staticmethod
    def _call(runtime: CnumpyRuntime, function_name: str, count: int):
        function = getattr(runtime.dll, function_name)
        function.argtypes = [ctypes.c_int64]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(function(count), function_name)

    def test_points_match_numpy_125_values_order_and_endpoints(self) -> None:
        functions = {
            "cnp_chebpts1": np.polynomial.chebyshev.chebpts1,
            "cnp_chebpts2": np.polynomial.chebyshev.chebpts2,
        }
        counts = {
            "cnp_chebpts1": (1, 2, 3, 8, 17, 257),
            "cnp_chebpts2": (2, 3, 8, 17, 257),
        }
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, reference in functions.items():
                for count in counts[function_name]:
                    with self.subTest(
                        function=function_name, count=count
                    ), self._call(runtime, function_name, count) as actual:
                        assert_array_equivalent(
                            self,
                            actual,
                            reference(count),
                            rtol=3e-15,
                            atol=3e-15,
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_invalid_counts_are_explicit_and_do_not_leak(self) -> None:
        cases = (
            ("cnp_chebpts1", (0, -1), "npts must be >= 1"),
            ("cnp_chebpts2", (1, 0, -1), "npts must be >= 2"),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, invalid_counts, message in cases:
                function = getattr(runtime.dll, function_name)
                function.argtypes = [ctypes.c_int64]
                function.restype = ctypes.c_void_p
                for count in invalid_counts:
                    with self.subTest(function=function_name, count=count):
                        runtime.dll.cnp_clear_error()
                        pointer = function(count)
                        if pointer:
                            runtime.dll.cnp_array_decref(pointer)
                        self.assertFalse(pointer)
                        state = runtime.error_state()
                        self.assertEqual(-4, state.status)
                        self.assertEqual(function_name, state.function)
                        self.assertIn(message, state.message)
            self.assertEqual(baseline, runtime.retained_bytes)


class SpecialFunctionSemanticsTests(unittest.TestCase):
    FUNCTIONS = {
        "cnp_erf": scipy_special.erf,
        "cnp_erfc": scipy_special.erfc,
        "cnp_expit": scipy_special.expit,
        "cnp_erfinv": scipy_special.erfinv,
        "cnp_logit": scipy_special.logit,
    }

    @staticmethod
    def _call(runtime: CnumpyRuntime, source, function_name: str):
        function = getattr(runtime.dll, function_name)
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(function(source.pointer), function_name)

    def assert_special_equivalent(
        self, actual, expected: np.ndarray
    ) -> None:
        expected = np.asarray(expected)
        tolerance = 4e-7 if expected.dtype == np.float32 else 3e-15
        assert_array_equivalent(
            self,
            actual,
            expected,
            rtol=tolerance,
            atol=tolerance,
        )

    def test_erfinv_matches_interior_float64_values(self) -> None:
        values = np.asarray(
            [-0.9, -0.5, -0.125, 0.0, 0.125, 0.5, 0.9],
            dtype=np.float64,
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(values) as source, self._call(
                runtime, source, "cnp_erfinv"
            ) as actual:
                self.assert_special_equivalent(
                    actual, scipy_special.erfinv(values)
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_erfinv_matches_domain_boundaries_and_subnormals(self) -> None:
        smallest = np.nextafter(0.0, 1.0)
        values = np.asarray(
            [
                -np.inf,
                np.nextafter(-1.0, -np.inf),
                -1.0,
                np.nextafter(-1.0, 0.0),
                -smallest,
                -0.0,
                0.0,
                smallest,
                np.nextafter(1.0, 0.0),
                1.0,
                np.nextafter(1.0, np.inf),
                np.inf,
                np.nan,
            ],
            dtype=np.float64,
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(values) as source, self._call(
                runtime, source, "cnp_erfinv"
            ) as actual:
                self.assert_special_equivalent(
                    actual, scipy_special.erfinv(values)
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_erfinv_matches_small_argument_asymptotic(self) -> None:
        subnormals = np.arange(1, 33, dtype=np.uint64).view(np.float64)
        values = np.concatenate(
            [
                -subnormals[::-1],
                subnormals,
                np.asarray(
                    [
                        -1e-100,
                        -1e-20,
                        -1e-15,
                        -1e-14,
                        -1e-10,
                        -1e-8,
                        -0.9e-7,
                        -1.1e-7,
                        1e-100,
                        1e-20,
                        1e-15,
                        1e-14,
                        1e-10,
                        1e-8,
                        0.9e-7,
                        1.1e-7,
                    ],
                    dtype=np.float64,
                ),
            ]
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(values) as source, self._call(
                runtime, source, "cnp_erfinv"
            ) as actual:
                assert_array_equivalent(
                    self,
                    actual,
                    scipy_special.erfinv(values),
                    rtol=3e-15,
                    atol=0.0,
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_erfinv_matches_scipy_dtype_dispatch(self) -> None:
        cases = [
            np.asarray([False, True], dtype=np.bool_),
            np.asarray([-1, 0, 1], dtype=np.int8),
            np.asarray([0, 1], dtype=np.uint16),
            np.asarray([-1, 0, 1], dtype=np.int64),
            np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float16),
            np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32),
            np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64),
        ]
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for values in cases:
                with self.subTest(dtype=values.dtype), runtime.from_numpy(
                    values
                ) as source, self._call(
                    runtime, source, "cnp_erfinv"
                ) as actual:
                    self.assert_special_equivalent(
                        actual, scipy_special.erfinv(values)
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_erfinv_tail_stays_within_three_scipy_ulps(self) -> None:
        positive = np.concatenate(
            [
                np.linspace(0.500001, 0.999, 4096, dtype=np.float64),
                1.0
                - np.geomspace(
                    np.finfo(np.float64).eps,
                    1e-3,
                    4096,
                    dtype=np.float64,
                ),
                np.asarray(
                    [
                        0.5182537341325439,
                        0.7566485324635142,
                        0.7769069917621241,
                        0.8001490893611203,
                        0.829084191870004,
                        0.8361684091272112,
                        0.8601794833615783,
                        0.9839862665198569,
                    ],
                    dtype=np.float64,
                ),
            ]
        )
        values = np.concatenate([-positive[::-1], positive])
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(values) as source, self._call(
                runtime, source, "cnp_erfinv"
            ) as actual:
                np.testing.assert_array_max_ulp(
                    actual.to_numpy(), scipy_special.erfinv(values), maxulp=3
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_real_ufunc_values_dtypes_shapes_and_views(self) -> None:
        common_cases = [
            np.asarray([False, True], dtype=np.bool_),
            np.asarray([-10, -1, 0, 1, 10], dtype=np.int64),
            np.asarray(
                [-np.inf, -30.0, -1.0, -0.0, 0.0, 1.0, 30.0, np.inf, np.nan],
                dtype=np.float16,
            ),
            np.asarray(
                [
                    -np.inf,
                    -100.0,
                    -89.0,
                    -88.0,
                    -1.0,
                    -0.0,
                    0.0,
                    1.0,
                    88.0,
                    89.0,
                    100.0,
                    np.inf,
                    np.nan,
                ],
                dtype=np.float32,
            ),
            np.asarray(
                [
                    -np.inf,
                    -1000.0,
                    -710.0,
                    -709.0,
                    -30.0,
                    -1.0,
                    -0.0,
                    0.0,
                    1.0,
                    30.0,
                    709.0,
                    710.0,
                    1000.0,
                    np.inf,
                    np.nan,
                ],
                dtype=np.float64,
            ),
        ]
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, reference in self.FUNCTIONS.items():
                if function_name == "cnp_logit":
                    continue
                for values in common_cases:
                    with self.subTest(
                        function=function_name, dtype=values.dtype
                    ), runtime.from_numpy(values) as source, self._call(
                        runtime, source, function_name
                    ) as actual:
                        self.assert_special_equivalent(
                            actual, reference(values)
                        )

                matrix = np.asarray(
                    [[-3.0, -0.0, 0.25], [1.0, 4.0, np.inf]],
                    dtype=np.float32,
                )
                with runtime.from_numpy(matrix) as source, self._call(
                    runtime, source, function_name
                ) as actual:
                    self.assert_special_equivalent(actual, reference(matrix))

                base_values = np.linspace(-4.0, 4.0, 17)
                base = runtime.from_numpy(base_values)
                view = PolynomialConversionSemanticsTests._slice(
                    runtime,
                    base,
                    _CnpSlice(0, 0, -2, False, False, True),
                )
                base.close()
                with view, self._call(runtime, view, function_name) as actual:
                    self.assert_special_equivalent(
                        actual, reference(base_values[::-2])
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_logit_matches_endpoints_outside_domain_and_near_one(self) -> None:
        real_cases = [
            np.asarray([False, True], dtype=np.bool_),
            np.asarray([0, 1, 2, -1], dtype=np.int64),
            np.asarray(
                [-1.0, -0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0, np.nan],
                dtype=np.float16,
            ),
            np.asarray(
                [
                    -np.inf,
                    -1.0,
                    -0.0,
                    0.0,
                    np.nextafter(0.0, 1.0),
                    0.25,
                    0.5,
                    0.75,
                    np.nextafter(1.0, 0.0),
                    1.0,
                    2.0,
                    np.inf,
                    np.nan,
                ],
                dtype=np.float64,
            ),
        ]
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for values in real_cases:
                with self.subTest(dtype=values.dtype), runtime.from_numpy(
                    values
                ) as source, self._call(runtime, source, "cnp_logit") as actual:
                    self.assert_special_equivalent(
                        actual, scipy_special.logit(values)
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_real_ufunc_errors_are_explicit_and_do_not_leak(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name in self.FUNCTIONS:
                function = getattr(runtime.dll, function_name)
                function.argtypes = [ctypes.c_void_p]
                function.restype = ctypes.c_void_p
                runtime.dll.cnp_clear_error()
                self.assertFalse(function(None))
                state = runtime.error_state()
                self.assertEqual(-1, state.status)
                self.assertEqual(function_name, state.function)
                self.assertIn("source array is required", state.message)

                with runtime.from_numpy(
                    np.asarray([1.0 + 2.0j], dtype=np.complex128)
                ) as source:
                    with self.assertRaises(CnumpyError) as raised:
                        self._call(runtime, source, function_name)
                    self.assertEqual(-3, raised.exception.status)
                    self.assertEqual(function_name, raised.exception.function)
                    self.assertIn(
                        "real numeric dtype", raised.exception.message
                    )
            self.assertEqual(baseline, runtime.retained_bytes)


class CombinatorialFunctionSemanticsTests(unittest.TestCase):
    @staticmethod
    def _call_factorial(runtime: CnumpyRuntime, source, exact: bool):
        function = runtime.dll.cnp_factorial
        function.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(
            function(source.pointer, exact), "cnp_factorial"
        )

    @staticmethod
    def _call_binary(
        runtime: CnumpyRuntime,
        left,
        right,
        function_name: str,
        exact: bool,
    ):
        function = getattr(runtime.dll, function_name)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_bool,
        ]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(
            function(left.pointer, right.pointer, exact), function_name
        )

    def test_combinatorial_null_inputs_are_explicit_and_do_not_leak(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            factorial = runtime.dll.cnp_factorial
            factorial.argtypes = [ctypes.c_void_p, ctypes.c_bool]
            factorial.restype = ctypes.c_void_p
            runtime.dll.cnp_clear_error()
            self.assertFalse(factorial(None, False))
            state = runtime.error_state()
            self.assertEqual(-1, state.status)
            self.assertEqual("cnp_factorial", state.function)
            self.assertIn("source array is required", state.message)

            for function_name in ("cnp_comb", "cnp_perm"):
                function = getattr(runtime.dll, function_name)
                function.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_bool,
                ]
                function.restype = ctypes.c_void_p
                runtime.dll.cnp_clear_error()
                self.assertFalse(function(None, None, False))
                state = runtime.error_state()
                self.assertEqual(-1, state.status)
                self.assertEqual(function_name, state.function)
                self.assertIn("input arrays are required", state.message)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_combinatorial_complex_inputs_are_explicit_and_do_not_leak(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(
                np.asarray([1.0 + 2.0j], dtype=np.complex128)
            ) as complex_values, runtime.from_numpy(
                np.asarray([2.0], dtype=np.float64)
            ) as real_values:
                unexpected = None
                try:
                    with self.assertRaises(CnumpyError) as raised:
                        unexpected = self._call_factorial(
                            runtime, complex_values, False
                        )
                finally:
                    if unexpected is not None:
                        unexpected.close()
                self.assertEqual(-3, raised.exception.status)
                self.assertEqual(
                    "cnp_factorial", raised.exception.function
                )
                self.assertIn(
                    "integer or floating-point dtype",
                    raised.exception.message,
                )

                for function_name in ("cnp_comb", "cnp_perm"):
                    for left, right in (
                        (complex_values, real_values),
                        (real_values, complex_values),
                    ):
                        unexpected = None
                        try:
                            with self.subTest(
                                function=function_name,
                                complex_side=(
                                    "left"
                                    if left is complex_values
                                    else "right"
                                ),
                            ), self.assertRaises(CnumpyError) as raised:
                                unexpected = self._call_binary(
                                    runtime,
                                    left,
                                    right,
                                    function_name,
                                    False,
                                )
                        finally:
                            if unexpected is not None:
                                unexpected.close()
                        self.assertEqual(-3, raised.exception.status)
                        self.assertEqual(
                            function_name, raised.exception.function
                        )
                        self.assertIn(
                            "real numeric dtype", raised.exception.message
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_combinatorial_exact_rejects_floating_dtypes_without_leaks(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(
                np.asarray([5.0], dtype=np.float64)
            ) as floating_values, runtime.from_numpy(
                np.asarray([2], dtype=np.int64)
            ) as integer_values:
                unexpected = None
                try:
                    with self.assertRaises(CnumpyError) as raised:
                        unexpected = self._call_factorial(
                            runtime, floating_values, True
                        )
                finally:
                    if unexpected is not None:
                        unexpected.close()
                self.assertEqual(-3, raised.exception.status)
                self.assertEqual(
                    "cnp_factorial", raised.exception.function
                )
                self.assertIn(
                    "requires an integer dtype", raised.exception.message
                )

                for function_name, noun in (
                    ("cnp_comb", "combinations"),
                    ("cnp_perm", "permutations"),
                ):
                    for left, right in (
                        (floating_values, integer_values),
                        (integer_values, floating_values),
                    ):
                        unexpected = None
                        try:
                            with self.subTest(
                                function=function_name,
                                floating_side=(
                                    "left"
                                    if left is floating_values
                                    else "right"
                                ),
                            ), self.assertRaises(CnumpyError) as raised:
                                unexpected = self._call_binary(
                                    runtime,
                                    left,
                                    right,
                                    function_name,
                                    True,
                                )
                        finally:
                            if unexpected is not None:
                                unexpected.close()
                        self.assertEqual(-3, raised.exception.status)
                        self.assertEqual(
                            function_name, raised.exception.function
                        )
                        self.assertIn(
                            f"exact {noun} require integer dtypes",
                            raised.exception.message,
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_comb_and_perm_broadcast_errors_are_explicit_and_do_not_leak(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(
                np.asarray([1, 2], dtype=np.int64)
            ) as left, runtime.from_numpy(
                np.asarray([1, 2, 3], dtype=np.int64)
            ) as right:
                for function_name in ("cnp_comb", "cnp_perm"):
                    unexpected = None
                    try:
                        with self.subTest(
                            function=function_name
                        ), self.assertRaises(CnumpyError) as raised:
                            unexpected = self._call_binary(
                                runtime,
                                left,
                                right,
                                function_name,
                                False,
                            )
                    finally:
                        if unexpected is not None:
                            unexpected.close()
                    self.assertEqual(-7, raised.exception.status)
                    self.assertEqual(
                        function_name, raised.exception.function
                    )
                    self.assertIn(
                        "Cannot broadcast shapes", raised.exception.message
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_factorial_approximate_matches_real_domain_values(self) -> None:
        values = np.asarray(
            [
                -np.inf,
                -2.5,
                -1.0,
                -0.5,
                -0.0,
                0.0,
                0.5,
                1.5,
                5.5,
                np.inf,
                np.nan,
            ],
            dtype=np.float64,
        )
        expected = scipy_special.factorial(values, exact=False)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(values) as source, self._call_factorial(
                runtime, source, False
            ) as actual:
                assert_array_equivalent(
                    self, actual, expected, rtol=6e-14, atol=6e-14
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_factorial_matches_nonempty_dtypes_and_native_views(self) -> None:
        approximate_cases = [
            np.asarray([0, 1, 5], dtype=np.int8),
            np.asarray([0, 1, 5], dtype=np.uint16),
            np.asarray([0, 1, 5], dtype=np.int32),
            np.asarray([0, 1, 5], dtype=np.uint64),
            np.asarray([0.0, 0.5, 5.0], dtype=np.float16),
            np.asarray([0.0, 0.5, 5.0], dtype=np.float32),
            np.asarray([0.0, 0.5, 5.0], dtype=np.float64),
        ]
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for values in approximate_cases:
                with self.subTest(dtype=values.dtype), runtime.from_numpy(
                    values
                ) as source, self._call_factorial(
                    runtime, source, False
                ) as actual:
                    assert_array_equivalent(
                        self,
                        actual,
                        scipy_special.factorial(values, exact=False),
                        rtol=6e-6 if actual.numpy_dtype == np.float32 else 6e-14,
                        atol=6e-6 if actual.numpy_dtype == np.float32 else 6e-14,
                    )

            matrix = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
            with runtime.from_numpy(matrix) as source, runtime.transpose(
                source, (1, 0, 2)
            ) as view, self._call_factorial(
                runtime, view, False
            ) as actual:
                assert_array_equivalent(
                    self,
                    actual,
                    scipy_special.factorial(
                        matrix.transpose(1, 0, 2), exact=False
                    ),
                    rtol=6e-6,
                    atol=6e-6,
                )

            f_matrix = np.arange(6, dtype=np.int64).reshape(2, 3)
            with runtime.from_numpy(f_matrix) as source, runtime.transpose(
                source
            ) as view, self._call_factorial(
                runtime, view, True
            ) as actual:
                assert_array_equivalent(
                    self,
                    actual,
                    scipy_special.factorial(f_matrix.T, exact=True),
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_factorial_rejects_boolean_input_explicitly(self) -> None:
        values = np.asarray([False, True], dtype=np.bool_)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(values) as source:
                unexpected = None
                try:
                    with self.assertRaises(CnumpyError) as raised:
                        unexpected = self._call_factorial(
                            runtime, source, False
                        )
                finally:
                    if unexpected is not None:
                        unexpected.close()
                self.assertEqual(-3, raised.exception.status)
                self.assertEqual("cnp_factorial", raised.exception.function)
                self.assertIn(
                    "integer or floating-point dtype", raised.exception.message
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_factorial_exact_matches_representable_integer_arrays(self) -> None:
        values = np.asarray([-3, -1, 0, 1, 5, 12, 13, 20], dtype=np.int64)
        expected = scipy_special.factorial(values, exact=True)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(values) as source, self._call_factorial(
                runtime, source, True
            ) as actual:
                assert_array_equivalent(self, actual, expected)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_factorial_preserves_empty_arrays_before_dtype_dispatch(self) -> None:
        cases = [
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.complex128),
        ]
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for values in cases:
                for exact in (False, True):
                    expected = scipy_special.factorial(values, exact=exact)
                    with self.subTest(
                        dtype=values.dtype, exact=exact
                    ), runtime.from_numpy(values) as source, self._call_factorial(
                        runtime, source, exact
                    ) as actual:
                        assert_array_equivalent(self, actual, expected)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_comb_approximate_matches_broadcast_real_values(self) -> None:
        left_values = np.asarray([[5.0], [5.5], [-1.0]], dtype=np.float64)
        right_values = np.asarray([[2.0, 2.5, 6.0, -1.0]], dtype=np.float64)
        expected = scipy_special.comb(
            left_values, right_values, exact=False
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(left_values) as left, runtime.from_numpy(
                right_values
            ) as right, self._call_binary(
                runtime, left, right, "cnp_comb", False
            ) as actual:
                assert_array_equivalent(
                    self, actual, expected, rtol=8e-13, atol=8e-13
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_comb_approximate_matches_nan_and_infinity_rules(self) -> None:
        left_values = np.asarray(
            [-np.inf, -1.0, 0.0, 5.0, np.inf, np.inf, np.nan, 5.0]
        )
        right_values = np.asarray(
            [0.0, 0.0, 0.0, 2.0, 0.0, np.inf, 2.0, np.nan]
        )
        expected = scipy_special.comb(
            left_values, right_values, exact=False
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(left_values) as left, runtime.from_numpy(
                right_values
            ) as right, self._call_binary(
                runtime, left, right, "cnp_comb", False
            ) as actual:
                assert_array_equivalent(
                    self, actual, expected, rtol=8e-13, atol=8e-13
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_comb_and_perm_approximate_match_result_dtype_dispatch(self) -> None:
        cases = [
            (np.int8, np.uint16),
            (np.float16, np.float32),
            (np.float32, np.float32),
            (np.int32, np.float32),
            (np.float32, np.float64),
        ]
        functions = {
            "cnp_comb": scipy_special.comb,
            "cnp_perm": scipy_special.perm,
        }
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, reference in functions.items():
                for left_dtype, right_dtype in cases:
                    left_values = np.asarray([5], dtype=left_dtype)
                    right_values = np.asarray([2], dtype=right_dtype)
                    with self.subTest(
                        function=function_name,
                        left_dtype=np.dtype(left_dtype),
                        right_dtype=np.dtype(right_dtype),
                    ), runtime.from_numpy(left_values) as left, runtime.from_numpy(
                        right_values
                    ) as right, self._call_binary(
                        runtime, left, right, function_name, False
                    ) as actual:
                        assert_array_equivalent(
                            self,
                            actual,
                            reference(left_values, right_values, exact=False),
                            rtol=(
                                6e-6
                                if actual.numpy_dtype == np.float32
                                else 8e-13
                            ),
                            atol=(
                                6e-6
                                if actual.numpy_dtype == np.float32
                                else 8e-13
                            ),
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_perm_approximate_matches_broadcast_real_values(self) -> None:
        left_values = np.asarray([[5.0], [5.5], [-1.0]], dtype=np.float64)
        right_values = np.asarray([[2.0, 2.5, 6.0, -1.0]], dtype=np.float64)
        expected = scipy_special.perm(
            left_values, right_values, exact=False
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(left_values) as left, runtime.from_numpy(
                right_values
            ) as right, self._call_binary(
                runtime, left, right, "cnp_perm", False
            ) as actual:
                assert_array_equivalent(
                    self, actual, expected, rtol=8e-13, atol=8e-13
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_perm_approximate_matches_nan_and_infinity_rules(self) -> None:
        left_values = np.asarray(
            [-np.inf, -1.0, 0.0, 5.0, np.inf, np.inf, np.nan, 5.0]
        )
        right_values = np.asarray(
            [0.0, 0.0, 0.0, 2.0, 0.0, np.inf, 2.0, np.nan]
        )
        expected = scipy_special.perm(
            left_values, right_values, exact=False
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(left_values) as left, runtime.from_numpy(
                right_values
            ) as right, self._call_binary(
                runtime, left, right, "cnp_perm", False
            ) as actual:
                assert_array_equivalent(
                    self, actual, expected, rtol=8e-13, atol=8e-13
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_comb_exact_matches_representable_integer_broadcasts(self) -> None:
        left_values = np.asarray([[5], [20], [66]], dtype=np.int64)
        right_values = np.asarray([[0, 2, 3, 33]], dtype=np.int64)
        broadcast_left, broadcast_right = np.broadcast_arrays(
            left_values, right_values
        )
        expected = np.asarray(
            [
                scipy_special.comb(int(n), int(k), exact=True)
                for n, k in zip(
                    broadcast_left.ravel(), broadcast_right.ravel()
                )
            ],
            dtype=np.int64,
        ).reshape(broadcast_left.shape)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(left_values) as left, runtime.from_numpy(
                right_values
            ) as right, self._call_binary(
                runtime, left, right, "cnp_comb", True
            ) as actual:
                assert_array_equivalent(self, actual, expected)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_perm_exact_matches_representable_integer_broadcasts(self) -> None:
        left_values = np.asarray([[5], [10], [20]], dtype=np.int64)
        right_values = np.asarray([[0, 2, 5, 20, 21]], dtype=np.int64)
        broadcast_left, broadcast_right = np.broadcast_arrays(
            left_values, right_values
        )
        expected = np.asarray(
            [
                scipy_special.perm(int(n), int(k), exact=True)
                for n, k in zip(
                    broadcast_left.ravel(), broadcast_right.ravel()
                )
            ],
            dtype=np.int64,
        ).reshape(broadcast_left.shape)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(left_values) as left, runtime.from_numpy(
                right_values
            ) as right, self._call_binary(
                runtime, left, right, "cnp_perm", True
            ) as actual:
                assert_array_equivalent(self, actual, expected)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_combinatorial_exact_overflow_is_explicit_and_does_not_leak(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(
                np.asarray([21], dtype=np.int64)
            ) as factorial_values:
                unexpected = None
                try:
                    with self.assertRaises(CnumpyError) as raised:
                        unexpected = self._call_factorial(
                            runtime, factorial_values, True
                        )
                finally:
                    if unexpected is not None:
                        unexpected.close()
                self.assertEqual(-8, raised.exception.status)
                self.assertEqual(
                    "cnp_factorial", raised.exception.function
                )
                self.assertIn(
                    "exceeds int64 representation",
                    raised.exception.message,
                )

            overflow_cases = [
                ("cnp_comb", 67, 33),
                ("cnp_perm", 21, 21),
            ]
            for function_name, left_value, right_value in overflow_cases:
                with runtime.from_numpy(
                    np.asarray([left_value], dtype=np.int64)
                ) as left, runtime.from_numpy(
                    np.asarray([right_value], dtype=np.int64)
                ) as right:
                    unexpected = None
                    try:
                        with self.subTest(
                            function=function_name
                        ), self.assertRaises(CnumpyError) as raised:
                            unexpected = self._call_binary(
                                runtime,
                                left,
                                right,
                                function_name,
                                True,
                            )
                    finally:
                        if unexpected is not None:
                            unexpected.close()
                    self.assertEqual(-8, raised.exception.status)
                    self.assertEqual(
                        function_name, raised.exception.function
                    )
                    self.assertIn(
                        "exceeds int64 representation",
                        raised.exception.message,
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_combinatorial_exact_rejects_uint64_beyond_int64_without_leaks(self) -> None:
        values = np.asarray([np.iinfo(np.uint64).max], dtype=np.uint64)
        zero = np.asarray([0], dtype=np.uint64)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(values) as source:
                unexpected = None
                try:
                    with self.assertRaises(CnumpyError) as raised:
                        unexpected = self._call_factorial(
                            runtime, source, True
                        )
                finally:
                    if unexpected is not None:
                        unexpected.close()
                self.assertEqual(-8, raised.exception.status)
                self.assertEqual(
                    "cnp_factorial", raised.exception.function
                )
                self.assertIn(
                    "exceeds int64 representation",
                    raised.exception.message,
                )

            with runtime.from_numpy(values) as left, runtime.from_numpy(
                zero
            ) as right:
                for function_name in ("cnp_comb", "cnp_perm"):
                    unexpected = None
                    try:
                        with self.subTest(
                            function=function_name
                        ), self.assertRaises(CnumpyError) as raised:
                            unexpected = self._call_binary(
                                runtime,
                                left,
                                right,
                                function_name,
                                True,
                            )
                    finally:
                        if unexpected is not None:
                            unexpected.close()
                    self.assertEqual(-8, raised.exception.status)
                    self.assertEqual(
                        function_name, raised.exception.function
                    )
                    self.assertIn(
                        "inputs exceed int64 representation",
                        raised.exception.message,
                    )
            self.assertEqual(baseline, runtime.retained_bytes)


class BesselFunctionSemanticsTests(unittest.TestCase):
    @staticmethod
    def _call(runtime: CnumpyRuntime, source, function_name: str):
        function = getattr(runtime.dll, function_name)
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(
            function(source.pointer), function_name
        )

    def test_j0_and_j1_match_real_dtype_dispatch(self) -> None:
        cases = [
            np.asarray([False, True], dtype=np.bool_),
            np.asarray([0, 1], dtype=np.int8),
            np.asarray([0, 1], dtype=np.uint16),
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([0, 1], dtype=np.uint64),
            np.asarray([0.0, 1.0], dtype=np.float16),
            np.asarray([0.0, 1.0], dtype=np.float32),
            np.asarray([0.0, 1.0], dtype=np.float64),
        ]
        functions = {
            "cnp_j0": scipy_special.j0,
            "cnp_j1": scipy_special.j1,
        }
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, reference in functions.items():
                for values in cases:
                    expected = reference(values)
                    with self.subTest(
                        function=function_name, dtype=values.dtype
                    ), runtime.from_numpy(values) as source, self._call(
                        runtime, source, function_name
                    ) as actual:
                        self.assertEqual(expected.shape, actual.shape)
                        self.assertEqual(
                            expected.dtype, actual.numpy_dtype
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_j0_and_j1_match_real_domain_edges(self) -> None:
        values64 = np.asarray(
            [
                -np.inf,
                -1.0e12,
                -1.0e6,
                -100.0,
                -8.0,
                np.nextafter(-5.0, -np.inf),
                -5.0,
                np.nextafter(-5.0, 0.0),
                -2.4048255576957728,
                -1.0,
                -np.nextafter(0.0, 1.0),
                -0.0,
                0.0,
                np.nextafter(0.0, 1.0),
                1.0,
                2.4048255576957728,
                np.nextafter(5.0, 0.0),
                5.0,
                np.nextafter(5.0, np.inf),
                8.0,
                100.0,
                1.0e6,
                1.0e12,
                np.inf,
                np.nan,
            ],
            dtype=np.float64,
        )
        values32 = values64.astype(np.float32)
        functions = {
            "cnp_j0": scipy_special.j0,
            "cnp_j1": scipy_special.j1,
        }
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, reference in functions.items():
                for values in (values32, values64):
                    tolerance = (
                        6e-7 if values.dtype == np.float32 else 4e-14
                    )
                    with self.subTest(
                        function=function_name, dtype=values.dtype
                    ), runtime.from_numpy(values) as source, self._call(
                        runtime, source, function_name
                    ) as actual:
                        assert_array_equivalent(
                            self,
                            actual,
                            reference(values),
                            rtol=tolerance,
                            atol=tolerance,
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_j0_and_j1_preserve_native_view_order(self) -> None:
        functions = {
            "cnp_j0": scipy_special.j0,
            "cnp_j1": scipy_special.j1,
        }
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            matrix = np.asarray(
                [[-5.0, -1.0, 0.0], [1.0, 5.0, 10.0]],
                dtype=np.float32,
            )
            cube = np.arange(-12, 12, dtype=np.int16).reshape(2, 3, 4)
            for function_name, reference in functions.items():
                with runtime.from_numpy(matrix) as source, runtime.transpose(
                    source
                ) as view, self._call(
                    runtime, view, function_name
                ) as actual:
                    assert_array_equivalent(
                        self,
                        actual,
                        reference(matrix.T),
                        rtol=6e-7,
                        atol=6e-7,
                    )
                with runtime.from_numpy(cube) as source, runtime.transpose(
                    source, (1, 0, 2)
                ) as view, self._call(
                    runtime, view, function_name
                ) as actual:
                    assert_array_equivalent(
                        self,
                        actual,
                        reference(cube.transpose(1, 0, 2)),
                        rtol=4e-14,
                        atol=4e-14,
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_j0_and_j1_errors_are_explicit_and_do_not_leak(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name in ("cnp_j0", "cnp_j1"):
                function = getattr(runtime.dll, function_name)
                function.argtypes = [ctypes.c_void_p]
                function.restype = ctypes.c_void_p
                runtime.dll.cnp_clear_error()
                self.assertFalse(function(None))
                state = runtime.error_state()
                self.assertEqual(-1, state.status)
                self.assertEqual(function_name, state.function)
                self.assertIn("source array is required", state.message)

                with runtime.from_numpy(
                    np.asarray([1.0 + 2.0j], dtype=np.complex128)
                ) as source:
                    unexpected = None
                    try:
                        with self.assertRaises(CnumpyError) as raised:
                            unexpected = self._call(
                                runtime, source, function_name
                            )
                    finally:
                        if unexpected is not None:
                            unexpected.close()
                    self.assertEqual(-3, raised.exception.status)
                    self.assertEqual(
                        function_name, raised.exception.function
                    )
                    self.assertIn(
                        "real numeric dtype", raised.exception.message
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_i0_matches_numpy_125_dtype_dispatch(self) -> None:
        cases = [
            np.asarray([False, True], dtype=np.bool_),
            np.asarray([0, 1, 8], dtype=np.int8),
            np.asarray([0, 1, 8], dtype=np.uint16),
            np.asarray([0, 1, 8], dtype=np.int32),
            np.asarray([0, 1, 8], dtype=np.uint64),
            np.asarray([0.0, 1.0, 8.0], dtype=np.float16),
            np.asarray([0.0, 1.0, 8.0], dtype=np.float32),
            np.asarray([0.0, 1.0, 8.0], dtype=np.float64),
        ]
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for values in cases:
                expected = np.i0(values)
                tolerance = (
                    5e-7
                    if expected.dtype == np.dtype(np.float32)
                    else 0.0
                )
                with self.subTest(dtype=values.dtype), runtime.from_numpy(
                    values
                ) as source, self._call(
                    runtime, source, "cnp_i0"
                ) as actual:
                    assert_array_equivalent(
                        self,
                        actual,
                        expected,
                        rtol=tolerance,
                        atol=tolerance,
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_i0_matches_partition_nonfinite_and_overflow_edges(self) -> None:
        float64_values = np.asarray(
            [
                -np.inf,
                -709.7827128933841,
                -709.782712893384,
                -8.0,
                np.nextafter(-8.0, 0.0),
                -1.0,
                -np.nextafter(0.0, 1.0),
                -0.0,
                0.0,
                np.nextafter(0.0, 1.0),
                1.0,
                np.nextafter(8.0, 0.0),
                8.0,
                np.nextafter(8.0, np.inf),
                709.782712893384,
                709.7827128933841,
                np.inf,
                np.nan,
            ],
            dtype=np.float64,
        )
        float32_values = np.asarray(
            [
                -np.inf,
                -88.72284,
                -88.72283,
                -8.0,
                np.nextafter(np.float32(-8.0), np.float32(0.0)),
                -0.0,
                0.0,
                np.nextafter(np.float32(8.0), np.float32(0.0)),
                8.0,
                np.nextafter(np.float32(8.0), np.float32(np.inf)),
                88.72283,
                88.72284,
                np.inf,
                np.nan,
            ],
            dtype=np.float32,
        )
        float16_values = np.asarray(
            [
                -np.inf,
                -11.09,
                -11.086,
                -8.0,
                -0.0,
                0.0,
                8.0,
                11.086,
                11.09,
                np.inf,
                np.nan,
            ],
            dtype=np.float16,
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for values, tolerance in (
                (float16_values, 0.0),
                (float32_values, 5e-7),
                (float64_values, 0.0),
            ):
                with np.errstate(all="ignore"):
                    expected = np.i0(values)
                with self.subTest(dtype=values.dtype), runtime.from_numpy(
                    values
                ) as source, self._call(
                    runtime, source, "cnp_i0"
                ) as actual:
                    assert_array_equivalent(
                        self,
                        actual,
                        expected,
                        rtol=tolerance,
                        atol=tolerance,
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_i0_preserves_scalar_empty_and_native_view_order(self) -> None:
        scalar = np.asarray(1.25, dtype=np.float32)
        empty = np.empty((2, 0, 3), dtype=np.float64)
        matrix = np.arange(-6.0, 6.0, dtype=np.float32).reshape(3, 4)
        cube = np.arange(-12.0, 12.0, dtype=np.float64).reshape(2, 3, 4)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for values in (scalar, empty):
                with self.subTest(shape=values.shape), runtime.from_numpy(
                    values
                ) as source, self._call(
                    runtime, source, "cnp_i0"
                ) as actual:
                    assert_array_equivalent(
                        self,
                        actual,
                        np.i0(values),
                        compare_strides=True,
                        rtol=5e-7,
                        atol=5e-7,
                    )

            for values, axes, tolerance in (
                (matrix, None, 5e-7),
                (cube, (1, 0, 2), 0.0),
            ):
                expected_input = (
                    values.T if axes is None else values.transpose(axes)
                )
                with runtime.from_numpy(values) as source, runtime.transpose(
                    source, axes
                ) as view, self._call(runtime, view, "cnp_i0") as actual:
                    assert_array_equivalent(
                        self,
                        actual,
                        np.i0(expected_input),
                        compare_strides=True,
                        rtol=tolerance,
                        atol=tolerance,
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_i0_strided_source_survives_source_first_release(self) -> None:
        base_values = np.asarray(
            [-8.0, 91.0, -1.0, 92.0, -0.0, 93.0, 1.0, 94.0, 8.0],
            dtype=np.float32,
        )
        expected = np.i0(base_values[::2])
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            base = runtime.from_numpy(base_values)
            view = PolynomialConversionSemanticsTests._slice(
                runtime,
                base,
                _CnpSlice(0, 0, 2, False, False, True),
            )
            base.close()
            try:
                actual = self._call(runtime, view, "cnp_i0")
            finally:
                view.close()
            with actual:
                assert_array_equivalent(
                    self,
                    actual,
                    expected,
                    compare_strides=True,
                    rtol=5e-7,
                    atol=5e-7,
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_i0_errors_are_explicit_and_do_not_leak(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            function = runtime.dll.cnp_i0
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_void_p

            runtime.dll.cnp_clear_error()
            self.assertFalse(function(None))
            state = runtime.error_state()
            self.assertEqual(-1, state.status)
            self.assertEqual("cnp_i0", state.function)
            self.assertIn("source array is required", state.message)

            for dtype in (np.complex64, np.complex128):
                with self.subTest(dtype=np.dtype(dtype)), runtime.from_numpy(
                    np.asarray([1.0 + 2.0j], dtype=dtype)
                ) as source:
                    unexpected = None
                    try:
                        with self.assertRaises(CnumpyError) as raised:
                            unexpected = self._call(
                                runtime, source, "cnp_i0"
                            )
                    finally:
                        if unexpected is not None:
                            unexpected.close()
                    self.assertEqual(-3, raised.exception.status)
                    self.assertEqual("cnp_i0", raised.exception.function)
                    self.assertIn(
                        "i0 not supported for complex values",
                        raised.exception.message,
                    )
            self.assertEqual(baseline, runtime.retained_bytes)


class DigammaZetaSemanticsTests(unittest.TestCase):
    @staticmethod
    def _call(runtime: CnumpyRuntime, source, function_name: str):
        function = getattr(runtime.dll, function_name)
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(
            function(source.pointer), function_name
        )

    def test_real_dtype_dispatch_matches_scipy(self) -> None:
        cases = [
            np.asarray([False, True], dtype=np.bool_),
            np.asarray([1, 2], dtype=np.int8),
            np.asarray([1, 2], dtype=np.uint16),
            np.asarray([1, 2], dtype=np.int32),
            np.asarray([1, 2], dtype=np.uint64),
            np.asarray([1.0, 2.0], dtype=np.float16),
            np.asarray([1.0, 2.0], dtype=np.float32),
            np.asarray([1.0, 2.0], dtype=np.float64),
        ]
        functions = {
            "cnp_digamma": scipy_special.digamma,
            "cnp_zeta": scipy_special.zeta,
        }
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, reference in functions.items():
                for values in cases:
                    expected = reference(values)
                    with self.subTest(
                        function=function_name, dtype=values.dtype
                    ), runtime.from_numpy(values) as source, self._call(
                        runtime, source, function_name
                    ) as actual:
                        self.assertEqual(expected.shape, actual.shape)
                        self.assertEqual(
                            expected.dtype, actual.numpy_dtype
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_real_digamma_matches_domain_edges_and_roots(self) -> None:
        negative_root = -0.5040830082644554
        positive_root = 1.4616321449683623
        values = np.asarray(
            [
                -np.inf,
                -10.0,
                -4.5,
                -3.0,
                -2.5,
                -1.0,
                np.nextafter(negative_root, -np.inf),
                negative_root,
                np.nextafter(negative_root, np.inf),
                -0.1,
                -np.nextafter(0.0, 1.0),
                -0.0,
                0.0,
                np.nextafter(0.0, 1.0),
                0.1,
                1.0,
                np.nextafter(positive_root, 0.0),
                positive_root,
                np.nextafter(positive_root, np.inf),
                2.0,
                10.0,
                1.0e6,
                np.inf,
                np.nan,
            ],
            dtype=np.float64,
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(values) as source, self._call(
                runtime, source, "cnp_digamma"
            ) as actual:
                assert_array_equivalent(
                    self,
                    actual,
                    scipy_special.digamma(values),
                    rtol=6e-14,
                    atol=6e-14,
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_riemann_zeta_matches_analytic_continuation_edges(self) -> None:
        values = np.asarray(
            [
                -np.inf,
                -100.5,
                -100.0,
                -55.5,
                -20.0,
                -10.0,
                -8.0,
                -4.0,
                -2.0,
                -1.0,
                -0.5,
                -0.01,
                -np.nextafter(0.0, 1.0),
                -0.0,
                0.0,
                np.nextafter(0.0, 1.0),
                0.5,
                np.nextafter(1.0, 0.0),
                1.0,
                np.nextafter(1.0, np.inf),
                2.0,
                3.0,
                10.0,
                50.0,
                127.0,
                128.0,
                np.inf,
                np.nan,
            ],
            dtype=np.float64,
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(values) as source, self._call(
                runtime, source, "cnp_zeta"
            ) as actual:
                assert_array_equivalent(
                    self,
                    actual,
                    scipy_special.zeta(values),
                    rtol=2e-13,
                    atol=2e-14,
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_riemann_zeta_negative_reflection_matches_scaled_lanczos(self) -> None:
        values = np.asarray(
            [
                -76.76894613770091,
                -84.15089267677072,
                -85.39274093550884,
                -89.52870190715736,
                -93.76663733528116,
                -93.87550719344264,
                -93.99453656207399,
                -95.62821784390177,
                -95.83349933534224,
            ],
            dtype=np.float64,
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(values) as source, self._call(
                runtime, source, "cnp_zeta"
            ) as actual:
                assert_array_equivalent(
                    self,
                    actual,
                    scipy_special.zeta(values),
                    rtol=5e-14,
                    atol=0.0,
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_digamma_matches_complex_dtypes_poles_and_views(self) -> None:
        values = [
            -2.0 + 0.25j,
            -1.0 + 0.0j,
            -0.5040830082644554 + 0.0j,
            -0.1 + 0.5j,
            0.0 + 1.0j,
            0.1 - 0.5j,
            1.4616321449683623 + 0.0j,
            1.0 + 0.0j,
            5.0 + 2.0j,
            20.0 - 3.0j,
        ]
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for dtype in (np.complex64, np.complex128):
                matrix = np.asarray(values, dtype=dtype).reshape(2, 5)
                tolerance = 2e-5 if dtype == np.complex64 else 2e-13
                with self.subTest(dtype=np.dtype(dtype)), runtime.from_numpy(
                    matrix
                ) as source, runtime.transpose(source) as view, self._call(
                    runtime, view, "cnp_digamma"
                ) as actual:
                    assert_array_equivalent(
                        self,
                        actual,
                        scipy_special.digamma(matrix.T),
                        rtol=tolerance,
                        atol=tolerance,
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_real_functions_preserve_native_view_order(self) -> None:
        functions = {
            "cnp_digamma": scipy_special.digamma,
            "cnp_zeta": scipy_special.zeta,
        }
        matrix = np.asarray(
            [[-3.5, -0.5, 0.5], [1.0, 2.0, 10.0]],
            dtype=np.float32,
        )
        cube = np.arange(-12, 12, dtype=np.int16).reshape(2, 3, 4)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, reference in functions.items():
                with runtime.from_numpy(matrix) as source, runtime.transpose(
                    source
                ) as view, self._call(
                    runtime, view, function_name
                ) as actual:
                    assert_array_equivalent(
                        self,
                        actual,
                        reference(matrix.T),
                        rtol=2e-5,
                        atol=2e-5,
                    )
                with runtime.from_numpy(cube) as source, runtime.transpose(
                    source, (1, 0, 2)
                ) as view, self._call(
                    runtime, view, function_name
                ) as actual:
                    assert_array_equivalent(
                        self,
                        actual,
                        reference(cube.transpose(1, 0, 2)),
                        rtol=6e-14,
                        atol=6e-14,
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_errors_are_explicit_and_do_not_leak(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name in ("cnp_digamma", "cnp_zeta"):
                function = getattr(runtime.dll, function_name)
                function.argtypes = [ctypes.c_void_p]
                function.restype = ctypes.c_void_p
                runtime.dll.cnp_clear_error()
                self.assertFalse(function(None))
                state = runtime.error_state()
                self.assertEqual(-1, state.status)
                self.assertEqual(function_name, state.function)
                self.assertIn("source array is required", state.message)

            with runtime.from_numpy(
                np.asarray([1.0 + 2.0j], dtype=np.complex128)
            ) as source:
                unexpected = None
                try:
                    with self.assertRaises(CnumpyError) as raised:
                        unexpected = self._call(
                            runtime, source, "cnp_zeta"
                        )
                finally:
                    if unexpected is not None:
                        unexpected.close()
                self.assertEqual(-3, raised.exception.status)
                self.assertEqual("cnp_zeta", raised.exception.function)
                self.assertIn(
                    "real numeric dtype", raised.exception.message
                )
            self.assertEqual(baseline, runtime.retained_bytes)


class GammaFunctionSemanticsTests(unittest.TestCase):
    @staticmethod
    def _call_unary(runtime: CnumpyRuntime, source, function_name: str):
        function = getattr(runtime.dll, function_name)
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(function(source.pointer), function_name)

    @staticmethod
    def _call_beta(runtime: CnumpyRuntime, left, right):
        function = runtime.dll.cnp_beta
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(
            function(left.pointer, right.pointer), "cnp_beta"
        )

    def assert_special_equivalent(
        self, actual, expected: np.ndarray
    ) -> None:
        expected = np.asarray(expected)
        tolerance = (
            6e-6 if expected.dtype in (np.float32, np.complex64) else 6e-14
        )
        assert_array_equivalent(
            self,
            actual,
            expected,
            rtol=tolerance,
            atol=tolerance,
        )

    def test_gamma_and_gammaln_match_real_values_dtypes_and_views(self) -> None:
        cases = [
            np.asarray([False, True], dtype=np.bool_),
            np.asarray([-4, -1, 0, 1, 5], dtype=np.int8),
            np.asarray([0, 1, 5, 10], dtype=np.uint16),
            np.asarray([-4, -1, 0, 1, 5], dtype=np.int32),
            np.asarray([0, 1, 5, 10], dtype=np.uint64),
            np.asarray(
                [-np.inf, -4.5, -1.0, -0.0, 0.0, 0.5, 5.0, np.inf, np.nan],
                dtype=np.float16,
            ),
            np.asarray(
                [-np.inf, -35.5, -4.5, -1.0, -0.0, 0.0, 0.5, 35.0, np.inf, np.nan],
                dtype=np.float32,
            ),
            np.asarray(
                [
                    -np.inf,
                    -170.71431860367744,
                    -170.5,
                    -35.5,
                    -4.5,
                    -1.0,
                    -0.0,
                    0.0,
                    np.nextafter(0.0, 1.0),
                    0.5,
                    35.0,
                    171.0,
                    172.0,
                    np.inf,
                    np.nan,
                ],
                dtype=np.float64,
            ),
        ]
        functions = {
            "cnp_gamma": scipy_special.gamma,
            "cnp_gammaln": scipy_special.gammaln,
        }
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name, reference in functions.items():
                for values in cases:
                    with self.subTest(
                        function=function_name, dtype=values.dtype
                    ), runtime.from_numpy(values) as source, self._call_unary(
                        runtime, source, function_name
                    ) as actual:
                        self.assert_special_equivalent(actual, reference(values))

                matrix = np.asarray(
                    [[-4.5, 0.25, 2.5], [0.5, 5.5, 12.25]],
                    dtype=np.float64,
                )
                with runtime.from_numpy(matrix) as source, runtime.transpose(
                    source
                ) as view, self._call_unary(
                    runtime, view, function_name
                ) as actual:
                    self.assert_special_equivalent(actual, reference(matrix.T))
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_gamma_matches_complex_values_dtypes_and_views(self) -> None:
        values = [
            0.0 + 0.0j,
            -1.0 + 0.0j,
            0.0 + 1.0j,
            0.0 - 1.0j,
            -2.5 + 0.25j,
            0.5 + 0.5j,
            1.0 + 0.0j,
            5.0 + 2.0j,
            complex(np.inf, 0.0),
            complex(np.nan, 0.0),
            complex(1.0, np.inf),
        ]
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for dtype in (np.complex64, np.complex128):
                matrix = np.asarray(values, dtype=dtype).reshape(1, -1)
                with self.subTest(dtype=np.dtype(dtype)), runtime.from_numpy(
                    matrix
                ) as source, runtime.transpose(source) as view, self._call_unary(
                    runtime, view, "cnp_gamma"
                ) as actual:
                    self.assert_special_equivalent(
                        actual, scipy_special.gamma(matrix.T)
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_beta_matches_broadcast_dtype_edges_and_stable_large_values(self) -> None:
        broadcast_cases = [
            (
                np.asarray([[0.5], [2.0]], dtype=np.float32),
                np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
            ),
            (
                np.asarray([[1.0], [2.0]], dtype=np.float16),
                np.asarray([[1, 2, 3]], dtype=np.int16),
            ),
            (
                np.asarray([[0.5], [2.0]], dtype=np.float32),
                np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
            ),
            (
                np.asarray([[1], [2]], dtype=np.int32),
                np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
            ),
        ]
        edge_left = np.asarray(
            [
                -5.0,
                -4.0,
                -3.0,
                -0.5,
                -0.5,
                -2.5,
                -2.5,
                -0.0,
                0.0,
                0.5,
                100.0,
                172.0,
                1e6,
                1e12,
                1000.0,
                np.inf,
                -np.inf,
                np.nan,
            ],
            dtype=np.float64,
        )
        edge_right = np.asarray(
            [
                2.0,
                3.0,
                4.0,
                -0.5,
                -1.5,
                1.5,
                -1.5,
                2.0,
                0.0,
                0.5,
                100.0,
                2.0,
                0.5,
                2.5,
                1000.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float64,
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for left_values, right_values in broadcast_cases:
                with self.subTest(
                    left=left_values.dtype, right=right_values.dtype
                ), runtime.from_numpy(left_values) as left, runtime.from_numpy(
                    right_values
                ) as right, self._call_beta(runtime, left, right) as actual:
                    self.assert_special_equivalent(
                        actual, scipy_special.beta(left_values, right_values)
                    )

            with runtime.from_numpy(edge_left) as left, runtime.from_numpy(
                edge_right
            ) as right, self._call_beta(runtime, left, right) as actual:
                self.assert_special_equivalent(
                    actual, scipy_special.beta(edge_left, edge_right)
                )

            precision_left = np.asarray(
                [496.08985770929127, 385.2280261796104], dtype=np.float64
            )
            precision_right = np.asarray(
                [209.3814508792733, 458.86102755934775], dtype=np.float64
            )
            with runtime.from_numpy(precision_left) as left, runtime.from_numpy(
                precision_right
            ) as right, self._call_beta(runtime, left, right) as actual:
                # A single libm log-Gamma ULP is amplified by exponentiation
                # for these large parameters. Zero absolute tolerance keeps
                # the measured 6e-13 relative envelope explicit.
                np.testing.assert_allclose(
                    actual.to_numpy(),
                    scipy_special.beta(precision_left, precision_right),
                    rtol=6e-13,
                    atol=0.0,
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_gamma_family_errors_are_explicit_and_do_not_leak(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name in ("cnp_gamma", "cnp_gammaln"):
                function = getattr(runtime.dll, function_name)
                function.argtypes = [ctypes.c_void_p]
                function.restype = ctypes.c_void_p
                runtime.dll.cnp_clear_error()
                self.assertFalse(function(None))
                state = runtime.error_state()
                self.assertEqual(-1, state.status)
                self.assertEqual(function_name, state.function)
                self.assertIn("source array is required", state.message)

            beta = runtime.dll.cnp_beta
            beta.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            beta.restype = ctypes.c_void_p
            runtime.dll.cnp_clear_error()
            self.assertFalse(beta(None, None))
            state = runtime.error_state()
            self.assertEqual(-1, state.status)
            self.assertEqual("cnp_beta", state.function)
            self.assertIn("input arrays are required", state.message)

            with runtime.from_numpy(
                np.asarray([1.0 + 2.0j], dtype=np.complex128)
            ) as complex_source:
                with self.assertRaises(CnumpyError) as raised:
                    self._call_unary(runtime, complex_source, "cnp_gammaln")
                self.assertEqual(-3, raised.exception.status)
                self.assertEqual("cnp_gammaln", raised.exception.function)
                self.assertIn("real numeric dtype", raised.exception.message)

                with runtime.from_numpy(
                    np.asarray([1.0], dtype=np.float64)
                ) as real_source:
                    with self.assertRaises(CnumpyError) as raised:
                        self._call_beta(runtime, complex_source, real_source)
                    self.assertEqual(-3, raised.exception.status)
                    self.assertEqual("cnp_beta", raised.exception.function)
                    self.assertIn("real numeric dtype", raised.exception.message)

            with runtime.from_numpy(
                np.ones((2, 3), dtype=np.float64)
            ) as left, runtime.from_numpy(
                np.ones((2, 2), dtype=np.float64)
            ) as right:
                with self.assertRaises(CnumpyError) as raised:
                    self._call_beta(runtime, left, right)
                self.assertEqual(-7, raised.exception.status)
                self.assertEqual("cnp_beta", raised.exception.function)
                self.assertIn("broadcast", raised.exception.message.lower())
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
