from __future__ import annotations

import ctypes
import random
import time
import unittest
from pathlib import Path

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"

UNITS = ("Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as")
UNIT_INDEX = {name: index for index, name in enumerate(UNITS)}
NAT = np.iinfo(np.int64).min
CNP_DATETIME = 22
CNP_ERR_SHAPE = -4
CNP_ERR_VALUE = -13


class DatetimeSurfaceSemanticsTests(unittest.TestCase):
    def _bind(self, runtime: CnumpyRuntime) -> None:
        dll = runtime.dll
        dll.cnp_datetime64_from_date.argtypes = [
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        dll.cnp_datetime64_from_date.restype = ctypes.c_int64
        dll.cnp_datetime64_from_time.argtypes = [
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        dll.cnp_datetime64_from_time.restype = ctypes.c_int64
        dll.cnp_datetime64_from_string.argtypes = [ctypes.c_char_p, ctypes.c_int]
        dll.cnp_datetime64_from_string.restype = ctypes.c_int64
        dll.cnp_datetime64_now.argtypes = [ctypes.c_int]
        dll.cnp_datetime64_now.restype = ctypes.c_int64
        dll.cnp_datetime64_to_date.argtypes = [
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        dll.cnp_datetime64_to_date.restype = None
        dll.cnp_datetime64_to_time.argtypes = [
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        dll.cnp_datetime64_to_time.restype = None
        dll.cnp_datetime64_to_string.argtypes = [ctypes.c_int64, ctypes.c_int]
        dll.cnp_datetime64_to_string.restype = ctypes.c_void_p
        dll.cnp_timedelta64_create.argtypes = [ctypes.c_int64, ctypes.c_int]
        dll.cnp_timedelta64_create.restype = ctypes.c_int64
        dll.cnp_datetime64_add.argtypes = [
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int,
        ]
        dll.cnp_datetime64_add.restype = ctypes.c_int64
        dll.cnp_datetime64_subtract.argtypes = [
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int,
        ]
        dll.cnp_datetime64_subtract.restype = ctypes.c_int64
        dll.cnp_datetime64_compare.argtypes = [ctypes.c_int64, ctypes.c_int64]
        dll.cnp_datetime64_compare.restype = ctypes.c_int
        dll.cnp_is_busday.argtypes = [ctypes.c_int64]
        dll.cnp_is_busday.restype = ctypes.c_bool
        dll.cnp_busday_count.argtypes = [ctypes.c_int64, ctypes.c_int64]
        dll.cnp_busday_count.restype = ctypes.c_int64
        dll.cnp_busday_offset.argtypes = [ctypes.c_int64, ctypes.c_int64]
        dll.cnp_busday_offset.restype = ctypes.c_int64
        dll.cnp_datetime_unit_name.argtypes = [ctypes.c_int]
        dll.cnp_datetime_unit_name.restype = ctypes.c_char_p
        dll.cnp_datetime64_array_create.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int,
        ]
        dll.cnp_datetime64_array_create.restype = ctypes.c_void_p
        dll.cnp_arange_datetime.argtypes = [
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int,
        ]
        dll.cnp_arange_datetime.restype = ctypes.c_void_p
        dll.cnp_datetime_as_string.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
        ]
        dll.cnp_datetime_as_string.restype = ctypes.c_int
        dll.cnp_datetime_as_string_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int64,
        ]
        dll.cnp_datetime_as_string_v2.restype = ctypes.c_int
        dll.cnp_char_free_string.argtypes = [ctypes.c_void_p]
        dll.cnp_char_free_string.restype = None

    def _scalar(self, runtime: CnumpyRuntime, function, *arguments) -> int:
        runtime.dll.cnp_clear_error()
        value = int(function(*arguments))
        error = runtime.error_state()
        self.assertEqual(0, error.status, str(error))
        return value

    def _string(self, runtime: CnumpyRuntime, raw: int, unit: int) -> str:
        runtime.dll.cnp_clear_error()
        pointer = runtime.dll.cnp_datetime64_to_string(raw, unit)
        self.assertTrue(pointer, runtime.error_state())
        try:
            return ctypes.string_at(pointer).decode("ascii")
        finally:
            runtime.dll.cnp_char_free_string(pointer)

    def _date_components(self, value: np.datetime64) -> tuple[int, int, int]:
        text = np.datetime_as_string(value, unit="D")
        year, month, day = text.rsplit("-", 2)
        return int(year), int(month), int(day)

    def _time_components(self, value: np.datetime64) -> tuple[int, int, int]:
        text = np.datetime_as_string(value, unit="s")
        if "T" not in text:
            return 0, 0, 0
        hour, minute, second = text.split("T", 1)[1].split(":")
        return int(hour), int(minute), int(second)

    def test_scalar_construction_roundtrip_and_format_match_numpy_125(self) -> None:
        date_text = "2024-02-29"
        timestamp = "2024-02-29T12:34:56.123456789012345678"

        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for unit, unit_number in UNIT_INDEX.items():
                with self.subTest(unit=unit):
                    expected_date = np.datetime64(date_text, unit)
                    raw_date = self._scalar(
                        runtime,
                        runtime.dll.cnp_datetime64_from_date,
                        2024,
                        2,
                        29,
                        unit_number,
                    )
                    self.assertEqual(int(expected_date.astype(np.int64)), raw_date)

                    expected_time = np.datetime64(timestamp, unit)
                    raw_time = self._scalar(
                        runtime,
                        runtime.dll.cnp_datetime64_from_time,
                        2024,
                        2,
                        29,
                        12,
                        34,
                        56,
                        unit_number,
                    )
                    expected_whole_second = np.datetime64(
                        "2024-02-29T12:34:56", unit
                    )
                    self.assertEqual(
                        int(expected_whole_second.astype(np.int64)), raw_time
                    )

                    parsed = self._scalar(
                        runtime,
                        runtime.dll.cnp_datetime64_from_string,
                        timestamp.encode("ascii"),
                        unit_number,
                    )
                    self.assertEqual(int(expected_time.astype(np.int64)), parsed)
                    value = np.datetime64(parsed, unit)

                    year = ctypes.c_int64(777)
                    month = ctypes.c_int(777)
                    day = ctypes.c_int(777)
                    runtime.dll.cnp_clear_error()
                    runtime.dll.cnp_datetime64_to_date(
                        parsed,
                        unit_number,
                        ctypes.byref(year),
                        ctypes.byref(month),
                        ctypes.byref(day),
                    )
                    self.assertEqual(0, runtime.error_state().status)
                    self.assertEqual(
                        self._date_components(value),
                        (year.value, month.value, day.value),
                    )

                    hour = ctypes.c_int(777)
                    minute = ctypes.c_int(777)
                    second = ctypes.c_int(777)
                    runtime.dll.cnp_clear_error()
                    runtime.dll.cnp_datetime64_to_time(
                        parsed,
                        unit_number,
                        ctypes.byref(hour),
                        ctypes.byref(minute),
                        ctypes.byref(second),
                    )
                    self.assertEqual(0, runtime.error_state().status)
                    self.assertEqual(
                        self._time_components(value),
                        (hour.value, minute.value, second.value),
                    )
                    self.assertEqual(
                        np.datetime_as_string(value, unit=unit),
                        self._string(runtime, parsed, unit_number),
                    )
                    self.assertEqual(
                        unit,
                        runtime.dll.cnp_datetime_unit_name(unit_number).decode("ascii"),
                    )
                    self.assertEqual(
                        7,
                        self._scalar(
                            runtime,
                            runtime.dll.cnp_timedelta64_create,
                            7,
                            unit_number,
                        ),
                    )
                    self.assertEqual(
                        np.int64(np.datetime64(parsed, unit) + np.timedelta64(3, unit)),
                        self._scalar(
                            runtime,
                            runtime.dll.cnp_datetime64_add,
                            parsed,
                            3,
                            unit_number,
                        ),
                    )
                    self.assertEqual(
                        3,
                        self._scalar(
                            runtime,
                            runtime.dll.cnp_datetime64_subtract,
                            parsed,
                            self._scalar(
                                runtime,
                                runtime.dll.cnp_datetime64_add,
                                parsed,
                                -3,
                                unit_number,
                            ),
                            unit_number,
                        ),
                    )
            self.assertEqual("NaT", self._string(runtime, NAT, UNIT_INDEX["ns"]))
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_error_results_are_atomic_labeled_and_nat_is_explicit(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            scalar_failures = [
                (
                    runtime.dll.cnp_datetime64_from_date,
                    (2023, 2, 29, UNIT_INDEX["D"]),
                    "cnp_datetime64_from_date",
                ),
                (
                    runtime.dll.cnp_datetime64_from_time,
                    (2024, 1, 1, 24, 0, 0, UNIT_INDEX["s"]),
                    "cnp_datetime64_from_time",
                ),
                (
                    runtime.dll.cnp_datetime64_from_string,
                    (b"2024-01-01 trailing", UNIT_INDEX["D"]),
                    "cnp_datetime64_from_string",
                ),
                (
                    runtime.dll.cnp_datetime64_from_date,
                    (2024, 1, 1, 99),
                    "cnp_datetime64_from_date",
                ),
            ]
            for function, arguments, label in scalar_failures:
                with self.subTest(label=label):
                    runtime.dll.cnp_clear_error()
                    self.assertEqual(0, int(function(*arguments)))
                    error = runtime.error_state()
                    self.assertEqual(CNP_ERR_VALUE, error.status)
                    self.assertEqual(label, error.function)

            self.assertEqual(
                NAT,
                self._scalar(
                    runtime,
                    runtime.dll.cnp_datetime64_from_string,
                    b"NaT",
                    UNIT_INDEX["ns"],
                ),
            )

            year = ctypes.c_int64(11)
            month = ctypes.c_int(22)
            runtime.dll.cnp_clear_error()
            runtime.dll.cnp_datetime64_to_date(
                0, UNIT_INDEX["D"], ctypes.byref(year), ctypes.byref(month), None
            )
            error = runtime.error_state()
            self.assertEqual(CNP_ERR_VALUE, error.status)
            self.assertEqual("cnp_datetime64_to_date", error.function)
            self.assertEqual((11, 22), (year.value, month.value))

            runtime.dll.cnp_clear_error()
            self.assertEqual(0, runtime.dll.cnp_datetime64_compare(NAT, 0))
            error = runtime.error_state()
            self.assertEqual(CNP_ERR_VALUE, error.status)
            self.assertEqual("cnp_datetime64_compare", error.function)

            runtime.dll.cnp_clear_error()
            self.assertFalse(runtime.dll.cnp_datetime_unit_name(99))
            error = runtime.error_state()
            self.assertEqual(CNP_ERR_VALUE, error.status)
            self.assertEqual("cnp_datetime_unit_name", error.function)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_business_day_results_match_default_numpy_calendar(self) -> None:
        dates = np.arange(
            np.datetime64("1969-12-25", "D"),
            np.datetime64("1970-01-15", "D"),
        )
        raw_dates = dates.astype(np.int64).tolist()

        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            actual = [bool(runtime.dll.cnp_is_busday(value)) for value in raw_dates]
            self.assertEqual(np.is_busday(dates).tolist(), actual)

            ranges = [
                ("1969-12-25", "1970-01-15"),
                ("1970-01-15", "1969-12-25"),
                ("1970-01-04", "1970-01-01"),
                ("1900-01-01", "2100-01-01"),
                ("2024-02-29", "2024-02-29"),
            ]
            for start_text, end_text in ranges:
                start = int(np.datetime64(start_text, "D").astype(np.int64))
                end = int(np.datetime64(end_text, "D").astype(np.int64))
                runtime.dll.cnp_clear_error()
                actual_count = int(runtime.dll.cnp_busday_count(start, end))
                self.assertEqual(0, runtime.error_state().status)
                self.assertEqual(
                    int(np.busday_count(start_text, end_text)), actual_count
                )

            for date_text in ("1970-01-01", "1970-01-02", "1970-01-05"):
                raw = int(np.datetime64(date_text, "D").astype(np.int64))
                for offset in (-1000, -10, -1, 0, 1, 10, 1000):
                    runtime.dll.cnp_clear_error()
                    actual_offset = int(runtime.dll.cnp_busday_offset(raw, offset))
                    self.assertEqual(0, runtime.error_state().status)
                    expected = int(
                        np.busday_offset(date_text, offset).astype(np.int64)
                    )
                    self.assertEqual(expected, actual_offset)

            weekend = int(np.datetime64("1970-01-03", "D").astype(np.int64))
            runtime.dll.cnp_clear_error()
            self.assertEqual(0, runtime.dll.cnp_busday_offset(weekend, 0))
            error = runtime.error_state()
            self.assertEqual(CNP_ERR_VALUE, error.status)
            self.assertEqual("cnp_busday_offset", error.function)
            self.assertIn("business day", error.message.lower())
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_datetime_arrays_ranges_and_string_v2_are_owned_and_atomic(self) -> None:
        values = (0, 1, -1, NAT)
        shape = (ctypes.c_int64 * 2)(2, 2)
        data = (ctypes.c_int64 * len(values))(*values)
        unit = UNIT_INDEX["ms"]

        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            pointer = runtime.dll.cnp_datetime64_array_create(2, shape, data, unit)
            with runtime._owned_result(pointer, "cnp_datetime64_array_create") as array:
                self.assertEqual(CNP_DATETIME, array.dtype_number)
                self.assertEqual((2, 2), array.shape)
                self.assertEqual(values, array.values())

                outputs = (ctypes.c_void_p * len(values))()
                runtime.dll.cnp_clear_error()
                status = runtime.dll.cnp_datetime_as_string_v2(
                    array.pointer, unit, outputs, len(values)
                )
                self.assertEqual(0, status, runtime.error_state())
                try:
                    actual = [ctypes.string_at(item).decode("ascii") for item in outputs]
                    expected = np.datetime_as_string(
                        np.asarray(values, dtype=np.int64).view("datetime64[ms]"),
                        unit="ms",
                    ).reshape(-1).tolist()
                    self.assertEqual(expected, actual)
                finally:
                    for item in outputs:
                        runtime.dll.cnp_char_free_string(item)

                sentinels = (ctypes.c_void_p * 3)(1, 2, 3)
                runtime.dll.cnp_clear_error()
                status = runtime.dll.cnp_datetime_as_string_v2(
                    array.pointer, unit, sentinels, 3
                )
                self.assertEqual(CNP_ERR_SHAPE, status)
                self.assertEqual([None, None, None], list(sentinels))
                error = runtime.error_state()
                self.assertEqual("cnp_datetime_as_string_v2", error.function)

                legacy = ctypes.create_string_buffer(b"unchanged", 64)
                runtime.dll.cnp_clear_error()
                status = runtime.dll.cnp_datetime_as_string(
                    array.pointer, legacy, len(legacy)
                )
                self.assertNotEqual(0, status)
                self.assertEqual(b"unchanged", legacy.value)
                self.assertEqual(
                    "cnp_datetime_as_string", runtime.error_state().function
                )

            range_cases = ((0, 7, 2), (7, 0, -2), (3, 3, 1))
            for start, stop, step in range_cases:
                pointer = runtime.dll.cnp_arange_datetime(start, stop, step, unit)
                with runtime._owned_result(pointer, "cnp_arange_datetime") as result:
                    self.assertEqual(CNP_DATETIME, result.dtype_number)
                    expected = np.arange(
                        np.datetime64(start, "ms"),
                        np.datetime64(stop, "ms"),
                        np.timedelta64(step, "ms"),
                    ).astype(np.int64)
                    np.testing.assert_array_equal(expected, result.to_numpy())

            runtime.dll.cnp_clear_error()
            self.assertFalse(runtime.dll.cnp_arange_datetime(0, 10, 0, unit))
            error = runtime.error_state()
            self.assertEqual(CNP_ERR_VALUE, error.status)
            self.assertEqual("cnp_arange_datetime", error.function)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_fixed_seed_scalar_and_business_day_probe_matches_numpy_125(self) -> None:
        rng = random.Random(0xD47E)
        raw_values = (-1_000_000, -12_345, -1, 0, 1, 12_345, 1_000_000)

        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            for unit, unit_number in UNIT_INDEX.items():
                generated = raw_values + tuple(
                    rng.randrange(-10_000_000, 10_000_001) for _ in range(100)
                )
                for raw in generated:
                    with self.subTest(unit=unit, raw=raw):
                        value = np.datetime64(raw, unit)
                        expected = np.datetime_as_string(value, unit=unit)
                        self.assertEqual(
                            expected, self._string(runtime, raw, unit_number)
                        )
                        reparsed = self._scalar(
                            runtime,
                            runtime.dll.cnp_datetime64_from_string,
                            expected.encode("ascii"),
                            unit_number,
                        )
                        self.assertEqual(raw, reparsed)

            for _ in range(200):
                first = rng.randrange(-100_000, 100_001)
                second = rng.randrange(-100_000, 100_001)
                first_date = np.datetime64(first, "D")
                second_date = np.datetime64(second, "D")
                self.assertEqual(
                    bool(np.is_busday(first_date)),
                    bool(runtime.dll.cnp_is_busday(first)),
                )
                self.assertEqual(
                    int(np.busday_count(first_date, second_date)),
                    self._scalar(
                        runtime, runtime.dll.cnp_busday_count, first, second
                    ),
                    (first, second),
                )

            self.assertEqual(baseline, runtime.retained_bytes)

    def test_now_is_a_real_utc_timestamp_and_repeated_strings_release(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            self._bind(runtime)
            baseline = runtime.retained_bytes
            before = int(np.datetime64("now", "s").astype(np.int64))
            current = self._scalar(
                runtime,
                runtime.dll.cnp_datetime64_now,
                UNIT_INDEX["s"],
            )
            after = int(np.datetime64("now", "s").astype(np.int64))
            self.assertLessEqual(before, current)
            self.assertLessEqual(current, after)

            for index in range(256):
                raw = index - 128
                self.assertEqual(
                    np.datetime_as_string(np.datetime64(raw, "ns"), unit="ns"),
                    self._string(runtime, raw, UNIT_INDEX["ns"]),
                )
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
