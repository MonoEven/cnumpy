from __future__ import annotations

import argparse
import ctypes
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from compat.cnumpy_ctypes import CnumpyRuntime


DEFAULT_DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"
DEFAULT_OUTPUT = (
    ROOT
    / "benchmark"
    / "runs-task14-text-io-surface"
    / "qualification.json"
)
CNP_DOUBLE = 13


def coefficient_of_variation(samples: list[float]) -> float:
    return statistics.stdev(samples) / statistics.mean(samples)


def measure(
    operation: Callable[[], None],
    *,
    warmups: int,
    samples: int,
    target_sample_ns: int,
) -> dict[str, object]:
    started = time.perf_counter_ns()
    operation()
    elapsed = time.perf_counter_ns() - started
    if elapsed <= 0:
        raise RuntimeError("performance calibration clock did not advance")
    repeats = math.ceil(target_sample_ns / elapsed)

    def sample() -> float:
        started = time.perf_counter_ns()
        for _ in range(repeats):
            operation()
        return (time.perf_counter_ns() - started) / repeats

    for _ in range(warmups):
        sample()
    observations = [sample() for _ in range(samples)]
    return {
        "repeats_per_sample": repeats,
        "median_ns_per_operation": statistics.median(observations),
        "cv_percent": coefficient_of_variation(observations) * 100.0,
        "samples_ns_per_operation": observations,
    }


def timed_pair(
    native_operation: Callable[[], None],
    numpy_operation: Callable[[], None],
    arguments: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object]]:
    common = {
        "warmups": arguments.warmups,
        "samples": arguments.samples,
        "target_sample_ns": int(arguments.target_sample_ms * 1e6),
    }
    return measure(native_operation, **common), measure(numpy_operation, **common)


def assert_runtime_clear(runtime: CnumpyRuntime, function_name: str) -> None:
    error = runtime.error_state()
    if error.status != 0:
        raise RuntimeError(f"{function_name} left a native error: {error}")


def result(
    name: str,
    function_name: str,
    native: dict[str, object],
    numpy: dict[str, object],
    retained_before: int,
    retained_during: int,
    retained_after: int,
) -> dict[str, object]:
    native_median = float(native["median_ns_per_operation"])
    numpy_median = float(numpy["median_ns_per_operation"])
    return {
        "case": name,
        "function": function_name,
        "cnumpy": native,
        "numpy": numpy,
        "cnumpy_over_numpy": native_median / numpy_median,
        "retained_bytes": {
            "before": retained_before,
            "during": retained_during,
            "after": retained_after,
            "result_delta": 0,
            "final_delta": 0,
        },
    }


def bind(runtime: CnumpyRuntime) -> None:
    runtime.dll.cnp_array2string.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int64,
    ]
    runtime.dll.cnp_array2string.restype = ctypes.c_int
    runtime.dll.cnp_array_to_string.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    runtime.dll.cnp_array_to_string.restype = ctypes.c_void_p
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
    runtime.dll.cnp_char_free_string.argtypes = [ctypes.c_void_p]
    runtime.dll.cnp_char_free_string.restype = None


def run(arguments: argparse.Namespace) -> dict[str, object]:
    integer_values = np.arange(128 * 128, dtype=np.int64).reshape(128, 128)
    float_values = np.linspace(
        -100.0, 100.0, 128 * 128, dtype=np.float64
    ).reshape(128, 128)
    csv_values = float_values.reshape(256, 64)
    file_values = np.linspace(
        -10.0, 10.0, 8192 * 4, dtype=np.float64
    ).reshape(8192, 4)
    cases: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory() as directory, CnumpyRuntime(
        arguments.dll
    ) as runtime:
        bind(runtime)
        runtime_baseline = runtime.retained_bytes

        retained_before = runtime.retained_bytes
        with runtime.from_numpy(integer_values) as source:
            retained_during = runtime.retained_bytes
            expected = np.array2string(
                integer_values,
                separator=", ",
                threshold=integer_values.size + 1,
                max_line_width=10_000_000,
                formatter={"int_kind": lambda value: str(value)},
            )
            buffer = ctypes.create_string_buffer(len(expected.encode()) + 1)
            runtime.dll.cnp_set_printoptions(
                8, integer_values.size + 1, 3, 75, 0
            )
            length = runtime.dll.cnp_array2string(
                source.pointer, buffer, len(buffer)
            )
            if length < 0:
                raise runtime.native_error("cnp_array2string")
            if buffer.value.decode() != expected:
                raise AssertionError("cnp_array2string validation mismatch")

            def native_array2string() -> None:
                length = runtime.dll.cnp_array2string(
                    source.pointer, buffer, len(buffer)
                )
                if length < 0:
                    raise runtime.native_error("cnp_array2string")
                assert_runtime_clear(runtime, "cnp_array2string")

            native, numpy = timed_pair(
                native_array2string,
                lambda: np.array2string(
                    integer_values,
                    separator=", ",
                    threshold=integer_values.size + 1,
                    max_line_width=10_000_000,
                    formatter={"int_kind": lambda value: str(value)},
                ),
                arguments,
            )
            if runtime.retained_bytes != retained_during:
                raise RuntimeError("cnp_array2string retained native bytes")
        retained_after = runtime.retained_bytes
        cases.append(result(
            "array2string/i64/128x128",
            "cnp_array2string",
            native,
            numpy,
            retained_before,
            retained_during,
            retained_after,
        ))

        retained_before = runtime.retained_bytes
        with runtime.from_numpy(float_values) as source:
            retained_during = runtime.retained_bytes
            formatter = {"float_kind": lambda value: f"{value:.6g}"}
            expected = np.array2string(
                float_values,
                separator=", ",
                threshold=float_values.size + 1,
                max_line_width=10_000_000,
                formatter=formatter,
            )

            def call_owned_text() -> str:
                pointer = runtime.dll.cnp_array_to_string(
                    source.pointer, b"%.6g"
                )
                if not pointer:
                    raise runtime.native_error("cnp_array_to_string")
                try:
                    return ctypes.string_at(pointer).decode()
                finally:
                    runtime.dll.cnp_char_free_string(pointer)

            if call_owned_text() != expected:
                raise AssertionError("cnp_array_to_string validation mismatch")
            native, numpy = timed_pair(
                lambda: call_owned_text(),
                lambda: np.array2string(
                    float_values,
                    separator=", ",
                    threshold=float_values.size + 1,
                    max_line_width=10_000_000,
                    formatter=formatter,
                ),
                arguments,
            )
            if runtime.retained_bytes != retained_during:
                raise RuntimeError("cnp_array_to_string retained native bytes")
        retained_after = runtime.retained_bytes
        cases.append(result(
            "array_to_string/f64/128x128/fmt6g",
            "cnp_array_to_string",
            native,
            numpy,
            retained_before,
            retained_during,
            retained_after,
        ))

        expected_csv_file = io.StringIO()
        np.savetxt(expected_csv_file, csv_values, delimiter=",", fmt="%g")
        expected_csv = expected_csv_file.getvalue().rstrip("\n")
        csv_buffer = ctypes.create_string_buffer(len(expected_csv.encode()) + 1)
        retained_before = runtime.retained_bytes
        with runtime.from_numpy(csv_values) as source:
            retained_during = runtime.retained_bytes

            def native_csv() -> None:
                status = runtime.dll.cnp_array_to_csv(
                    source.pointer, csv_buffer, len(csv_buffer), b","
                )
                if status != 0:
                    raise runtime.native_error("cnp_array_to_csv")
                assert_runtime_clear(runtime, "cnp_array_to_csv")

            native_csv()
            if csv_buffer.value.decode() != expected_csv:
                raise AssertionError("cnp_array_to_csv validation mismatch")
            native, numpy = timed_pair(
                native_csv,
                lambda: np.savetxt(
                    io.StringIO(), csv_values, delimiter=",", fmt="%g"
                ),
                arguments,
            )
            if runtime.retained_bytes != retained_during:
                raise RuntimeError("cnp_array_to_csv retained native bytes")
        retained_after = runtime.retained_bytes
        cases.append(result(
            "array_to_csv/f64/256x64",
            "cnp_array_to_csv",
            native,
            numpy,
            retained_before,
            retained_during,
            retained_after,
        ))

        native_save_path = Path(directory) / "native-save.csv"
        numpy_save_path = Path(directory) / "numpy-save.csv"
        retained_before = runtime.retained_bytes
        with runtime.from_numpy(file_values) as source:
            retained_during = runtime.retained_bytes

            def native_save() -> None:
                status = runtime.dll.cnp_savetxt(
                    str(native_save_path).encode(),
                    source.pointer,
                    b",",
                    b"%.8g",
                )
                if status != 0:
                    raise runtime.native_error("cnp_savetxt")
                assert_runtime_clear(runtime, "cnp_savetxt")

            native_save()
            numpy_text = io.StringIO()
            np.savetxt(numpy_text, file_values, delimiter=",", fmt="%.8g")
            if native_save_path.read_text() != numpy_text.getvalue():
                raise AssertionError("cnp_savetxt validation mismatch")
            native, numpy = timed_pair(
                native_save,
                lambda: np.savetxt(
                    numpy_save_path,
                    file_values,
                    delimiter=",",
                    fmt="%.8g",
                ),
                arguments,
            )
            if runtime.retained_bytes != retained_during:
                raise RuntimeError("cnp_savetxt retained native bytes")
        retained_after = runtime.retained_bytes
        cases.append(result(
            "savetxt/f64/8192x4/fmt8g",
            "cnp_savetxt",
            native,
            numpy,
            retained_before,
            retained_during,
            retained_after,
        ))

        retained_before = runtime.retained_bytes
        retained_during = retained_before

        def native_load() -> None:
            pointer = runtime.dll.cnp_loadtxt(
                str(native_save_path).encode(), b",", CNP_DOUBLE
            )
            if not pointer:
                raise runtime.native_error("cnp_loadtxt")
            runtime.dll.cnp_array_decref(pointer)
            assert_runtime_clear(runtime, "cnp_loadtxt")

        with runtime._owned_result(
            runtime.dll.cnp_loadtxt(
                str(native_save_path).encode(), b",", CNP_DOUBLE
            ),
            "cnp_loadtxt",
        ) as loaded:
            np.testing.assert_allclose(
                loaded.to_numpy(),
                np.loadtxt(native_save_path, delimiter=",", dtype=np.float64),
            )
        native, numpy = timed_pair(
            native_load,
            lambda: np.loadtxt(
                native_save_path, delimiter=",", dtype=np.float64
            ),
            arguments,
        )
        retained_after = runtime.retained_bytes
        cases.append(result(
            "loadtxt/f64/8192x4",
            "cnp_loadtxt",
            native,
            numpy,
            retained_before,
            retained_during,
            retained_after,
        ))

        missing_path = Path(directory) / "missing.csv"
        missing_rows = ["first,second,third,fourth"]
        for row in range(4096):
            missing_rows.append(
                f"{row},,{row + 2},{'bad' if row % 7 == 0 else row + 3}"
            )
        missing_path.write_text("\n".join(missing_rows) + "\n")
        retained_before = runtime.retained_bytes
        retained_during = retained_before

        def native_genfromtxt() -> None:
            pointer = runtime.dll.cnp_genfromtxt(
                str(missing_path).encode(), b",", 1, -1, CNP_DOUBLE
            )
            if not pointer:
                raise runtime.native_error("cnp_genfromtxt")
            runtime.dll.cnp_array_decref(pointer)
            assert_runtime_clear(runtime, "cnp_genfromtxt")

        with runtime._owned_result(
            runtime.dll.cnp_genfromtxt(
                str(missing_path).encode(), b",", 1, -1, CNP_DOUBLE
            ),
            "cnp_genfromtxt",
        ) as loaded:
            expected = np.genfromtxt(
                missing_path,
                delimiter=",",
                skip_header=1,
                dtype=np.float64,
            )
            np.testing.assert_array_equal(loaded.to_numpy(), expected)
        native, numpy = timed_pair(
            native_genfromtxt,
            lambda: np.genfromtxt(
                missing_path,
                delimiter=",",
                skip_header=1,
                dtype=np.float64,
            ),
            arguments,
        )
        retained_after = runtime.retained_bytes
        cases.append(result(
            "genfromtxt/f64/4096x4/missing",
            "cnp_genfromtxt",
            native,
            numpy,
            retained_before,
            retained_during,
            retained_after,
        ))

        if runtime.retained_bytes != runtime_baseline:
            raise RuntimeError(
                "text I/O qualification did not restore runtime baseline"
            )

    return {
        "task": 14,
        "domain": "text_io_surface",
        "method": {
            "warmups": arguments.warmups,
            "samples": arguments.samples,
            "target_sample_ms": arguments.target_sample_ms,
            "clock": "time.perf_counter_ns",
            "order": "cnumpy then NumPy per case",
            "python": ".".join(map(str, sys.version_info[:3])),
            "numpy": np.__version__,
        },
        "dll": {
            "path": str(arguments.dll.resolve()),
            "sha256": hashlib.sha256(arguments.dll.read_bytes()).hexdigest(),
        },
        "cases": cases,
        "retained_bytes_after_release": 0,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dll", type=Path, default=DEFAULT_DLL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--target-sample-ms", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    qualification = run(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(qualification, indent=2) + "\n", encoding="utf-8"
    )
    print(arguments.output.resolve())
    for case in qualification["cases"]:
        print(
            f"{case['case']}: {case['cnumpy_over_numpy']:.6f}x; "
            f"cnumpy CV {case['cnumpy']['cv_percent']:.2f}%; "
            f"NumPy CV {case['numpy']['cv_percent']:.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
