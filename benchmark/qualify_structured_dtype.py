from __future__ import annotations

import argparse
import ctypes
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
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
    / "runs-task14-structured-dtype-surface"
    / "qualification.json"
)
CNP_INT = 6
CNP_LONG = 8
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


def array_function(
    runtime: CnumpyRuntime,
    name: str,
    argtypes: list[object],
) -> ctypes._CFuncPtr:
    function = getattr(runtime.dll, name)
    function.argtypes = argtypes
    function.restype = ctypes.c_void_p
    return function


def assert_runtime_clear(runtime: CnumpyRuntime, name: str) -> None:
    error = runtime.error_state()
    if error.status != 0:
        raise RuntimeError(f"{name} left a native error: {error}")


def timed_pair(
    native_operation: Callable[[], None],
    numpy_operation: Callable[[], None],
    *,
    warmups: int,
    samples: int,
    target_sample_ns: int,
) -> tuple[dict[str, object], dict[str, object]]:
    native = measure(
        native_operation,
        warmups=warmups,
        samples=samples,
        target_sample_ns=target_sample_ns,
    )
    numpy = measure(
        numpy_operation,
        warmups=warmups,
        samples=samples,
        target_sample_ns=target_sample_ns,
    )
    return native, numpy


def case_result(
    *,
    name: str,
    function_name: str,
    source_shape: tuple[int, ...],
    source_dtype: np.dtype[object],
    native: dict[str, object],
    numpy: dict[str, object],
    retained_before_sources: int,
    retained_with_sources: int,
    retained_after_sources: int,
) -> dict[str, object]:
    native_median = float(native["median_ns_per_operation"])
    numpy_median = float(numpy["median_ns_per_operation"])
    return {
        "case": name,
        "function": function_name,
        "source_shape": list(source_shape),
        "source_dtype": str(source_dtype),
        "cnumpy": native,
        "numpy": numpy,
        "cnumpy_over_numpy": native_median / numpy_median,
        "retained_bytes": {
            "before_sources": retained_before_sources,
            "with_sources": retained_with_sources,
            "after_sources": retained_after_sources,
            "result_delta": 0,
            "final_delta": 0,
        },
    }


def qualify_array_result(
    runtime: CnumpyRuntime,
    *,
    name: str,
    function_name: str,
    source_value: np.ndarray,
    argtypes: list[object],
    native_arguments: Callable[[int], tuple[object, ...]],
    numpy_operation: Callable[[], np.ndarray],
    warmups: int,
    samples: int,
    target_sample_ns: int,
) -> dict[str, object]:
    retained_before_sources = runtime.retained_bytes
    with runtime.from_numpy(source_value) as source:
        function = array_function(runtime, function_name, argtypes)
        arguments = native_arguments(source.pointer)
        retained_with_sources = runtime.retained_bytes
        runtime.dll.cnp_clear_error()
        with runtime._owned_result(
            function(*arguments), function_name
        ) as actual:
            actual_value = actual.to_numpy()
        expected = np.asarray(numpy_operation())
        np.testing.assert_array_equal(actual_value, expected)
        if actual_value.dtype != expected.dtype:
            raise AssertionError(
                f"{function_name} dtype {actual_value.dtype} != {expected.dtype}"
            )
        if runtime.retained_bytes != retained_with_sources:
            raise RuntimeError(
                f"{function_name} validation retained native bytes"
            )

        def native_operation() -> None:
            pointer = function(*arguments)
            if not pointer:
                raise runtime.native_error(function_name)
            runtime.dll.cnp_array_decref(pointer)
            assert_runtime_clear(runtime, function_name)

        native, numpy = timed_pair(
            native_operation,
            numpy_operation,
            warmups=warmups,
            samples=samples,
            target_sample_ns=target_sample_ns,
        )
        if runtime.retained_bytes != retained_with_sources:
            raise RuntimeError(
                f"{function_name} timed calls retained native bytes"
            )

    retained_after_sources = runtime.retained_bytes
    if retained_after_sources != retained_before_sources:
        raise RuntimeError(
            f"{function_name} source release retained native bytes"
        )
    return case_result(
        name=name,
        function_name=function_name,
        source_shape=source_value.shape,
        source_dtype=source_value.dtype,
        native=native,
        numpy=numpy,
        retained_before_sources=retained_before_sources,
        retained_with_sources=retained_with_sources,
        retained_after_sources=retained_after_sources,
    )


def qualify_setfield(
    runtime: CnumpyRuntime,
    source_value: np.ndarray,
    *,
    warmups: int,
    samples: int,
    target_sample_ns: int,
) -> dict[str, object]:
    function_name = "cnp_setfield"
    retained_before_sources = runtime.retained_bytes
    replacement_value = np.asarray(9, dtype=np.int32)
    expected = source_value.copy()
    expected.setfield(replacement_value, np.int32, 0)
    numpy_destination = source_value.copy()
    with ExitStack() as stack:
        destination = stack.enter_context(runtime.from_numpy(source_value))
        replacement = stack.enter_context(
            runtime.from_numpy(replacement_value)
        )
        function = runtime.dll.cnp_setfield
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int64,
        ]
        function.restype = ctypes.c_int
        retained_with_sources = runtime.retained_bytes

        runtime.dll.cnp_clear_error()
        status = function(
            destination.pointer, replacement.pointer, CNP_INT, 0
        )
        if status != 0:
            raise runtime.native_error(function_name)
        np.testing.assert_array_equal(destination.to_numpy(), expected)

        def native_operation() -> None:
            status = function(
                destination.pointer, replacement.pointer, CNP_INT, 0
            )
            if status != 0:
                raise runtime.native_error(function_name)
            assert_runtime_clear(runtime, function_name)

        def numpy_operation() -> None:
            numpy_destination.setfield(replacement_value, np.int32, 0)

        native, numpy = timed_pair(
            native_operation,
            numpy_operation,
            warmups=warmups,
            samples=samples,
            target_sample_ns=target_sample_ns,
        )
        if runtime.retained_bytes != retained_with_sources:
            raise RuntimeError(
                f"{function_name} timed calls retained native bytes"
            )

    retained_after_sources = runtime.retained_bytes
    if retained_after_sources != retained_before_sources:
        raise RuntimeError(
            f"{function_name} source release retained native bytes"
        )
    return case_result(
        name="setfield/i64-as-i32/512x512",
        function_name=function_name,
        source_shape=source_value.shape,
        source_dtype=source_value.dtype,
        native=native,
        numpy=numpy,
        retained_before_sources=retained_before_sources,
        retained_with_sources=retained_with_sources,
        retained_after_sources=retained_after_sources,
    )


def create_structured_dtype(runtime: CnumpyRuntime) -> int:
    names = (ctypes.c_char_p * 2)(b"identifier", b"score")
    dtypes = (ctypes.c_int * 2)(CNP_INT, CNP_DOUBLE)
    function = runtime.dll.cnp_struct_dtype_create
    function.argtypes = [
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    function.restype = ctypes.c_int
    runtime.dll.cnp_clear_error()
    dtype_id = function(names, dtypes, 2)
    if dtype_id < 0:
        raise runtime.native_error("cnp_struct_dtype_create")
    return dtype_id


def qualify_record_fields(
    runtime: CnumpyRuntime,
    *,
    warmups: int,
    samples: int,
    target_sample_ns: int,
) -> list[dict[str, object]]:
    dtype_id = create_structured_dtype(runtime)
    record_dtype = np.dtype(
        [("identifier", np.int32), ("score", np.float64)], align=False
    )
    shape = (512, 512)
    shape_buffer = (ctypes.c_int64 * 2)(*shape)
    new = array_function(
        runtime,
        "cnp_recarray_new",
        [ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_int],
    )
    get_field = array_function(
        runtime,
        "cnp_recarray_get_field",
        [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int],
    )
    set_field = runtime.dll.cnp_recarray_set_field
    set_field.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    set_field.restype = ctypes.c_int
    replacement_value = np.asarray(7, dtype=np.int32)
    numpy_records = np.zeros(shape, dtype=record_dtype)
    retained_before_sources = runtime.retained_bytes

    with runtime._owned_result(
        new(2, shape_buffer, dtype_id), "cnp_recarray_new"
    ) as records, runtime.from_numpy(replacement_value) as replacement:
        retained_with_sources = runtime.retained_bytes
        status = set_field(
            records.pointer, b"identifier", dtype_id, replacement.pointer
        )
        if status != 0:
            raise runtime.native_error("cnp_recarray_set_field")
        numpy_records["identifier"] = replacement_value

        with runtime._owned_result(
            get_field(records.pointer, b"identifier", dtype_id),
            "cnp_recarray_get_field",
        ) as field:
            np.testing.assert_array_equal(
                field.to_numpy(), numpy_records["identifier"]
            )
        if runtime.retained_bytes != retained_with_sources:
            raise RuntimeError(
                "record field validation retained native bytes"
            )

        def native_get() -> None:
            pointer = get_field(records.pointer, b"identifier", dtype_id)
            if not pointer:
                raise runtime.native_error("cnp_recarray_get_field")
            runtime.dll.cnp_array_decref(pointer)
            assert_runtime_clear(runtime, "cnp_recarray_get_field")

        native_get_result, numpy_get_result = timed_pair(
            native_get,
            lambda: numpy_records["identifier"],
            warmups=warmups,
            samples=samples,
            target_sample_ns=target_sample_ns,
        )

        def native_set() -> None:
            status = set_field(
                records.pointer,
                b"identifier",
                dtype_id,
                replacement.pointer,
            )
            if status != 0:
                raise runtime.native_error("cnp_recarray_set_field")
            assert_runtime_clear(runtime, "cnp_recarray_set_field")

        native_set_result, numpy_set_result = timed_pair(
            native_set,
            lambda: numpy_records.__setitem__(
                "identifier", replacement_value
            ),
            warmups=warmups,
            samples=samples,
            target_sample_ns=target_sample_ns,
        )
        if runtime.retained_bytes != retained_with_sources:
            raise RuntimeError(
                "record field timed calls retained native bytes"
            )

    retained_after_sources = runtime.retained_bytes
    if retained_after_sources != retained_before_sources:
        raise RuntimeError("record field sources retained native bytes")
    common = {
        "source_shape": shape,
        "source_dtype": record_dtype,
        "retained_before_sources": retained_before_sources,
        "retained_with_sources": retained_with_sources,
        "retained_after_sources": retained_after_sources,
    }
    return [
        case_result(
            name="recarray_get_field/packed/512x512",
            function_name="cnp_recarray_get_field",
            native=native_get_result,
            numpy=numpy_get_result,
            **common,
        ),
        case_result(
            name="recarray_set_field/packed/512x512/scalar",
            function_name="cnp_recarray_set_field",
            native=native_set_result,
            numpy=numpy_set_result,
            **common,
        ),
    ]


def run(arguments: argparse.Namespace) -> dict[str, object]:
    encoded = np.arange(512 * 512, dtype=np.int64).reshape(512, 512)
    common = {
        "warmups": arguments.warmups,
        "samples": arguments.samples,
        "target_sample_ns": int(arguments.target_sample_ms * 1e6),
    }
    cases: list[dict[str, object]] = []
    with CnumpyRuntime(arguments.dll) as runtime:
        runtime_baseline = runtime.retained_bytes
        cases.append(qualify_array_result(
            runtime,
            name="view/i64-as-i32/512x512",
            function_name="cnp_view",
            source_value=encoded,
            argtypes=[ctypes.c_void_p, ctypes.c_int],
            native_arguments=lambda pointer: (pointer, CNP_INT),
            numpy_operation=lambda: encoded.view(np.int32),
            **common,
        ))
        cases.append(qualify_array_result(
            runtime,
            name="getfield/i64-as-i32-offset4/512x512",
            function_name="cnp_getfield",
            source_value=encoded,
            argtypes=[ctypes.c_void_p, ctypes.c_int, ctypes.c_int64],
            native_arguments=lambda pointer: (pointer, CNP_INT, 4),
            numpy_operation=lambda: encoded.getfield(np.int32, 4),
            **common,
        ))
        cases.append(qualify_setfield(runtime, encoded, **common))
        cases.append(qualify_array_result(
            runtime,
            name="newbyteorder/i64/512x512/native-projection",
            function_name="cnp_newbyteorder",
            source_value=encoded,
            argtypes=[ctypes.c_void_p],
            native_arguments=lambda pointer: (pointer,),
            numpy_operation=lambda: encoded.newbyteorder().astype(np.int64),
            **common,
        ))
        cases.extend(qualify_record_fields(runtime, **common))
        if runtime.retained_bytes != runtime_baseline:
            raise RuntimeError(
                "structured dtype qualification did not restore runtime baseline"
            )

    return {
        "task": 14,
        "domain": "structured_dtype_surface",
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
    result = run(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(arguments.output.resolve())
    for case in result["cases"]:
        print(
            f"{case['case']}: {case['cnumpy_over_numpy']:.6f}x; "
            f"cnumpy CV {case['cnumpy']['cv_percent']:.2f}%; "
            f"NumPy CV {case['numpy']['cv_percent']:.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
