from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
from contextlib import ExitStack
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
DEFAULT_OUTPUT = ROOT / "benchmark" / "runs-task14-product-surface" / "qualification.json"


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


def native_operation(
    runtime: CnumpyRuntime,
    function: ctypes._CFuncPtr,
    name: str,
    arguments: tuple[object, ...],
) -> Callable[[], None]:
    def invoke() -> None:
        pointer = function(*arguments)
        if not pointer:
            raise runtime.native_error(name)
        runtime.dll.cnp_array_decref(pointer)
        error = runtime.error_state()
        if error.status != 0:
            raise error

    return invoke


def validate(
    runtime: CnumpyRuntime,
    function: ctypes._CFuncPtr,
    name: str,
    arguments: tuple[object, ...],
    expected: np.ndarray,
) -> None:
    runtime.dll.cnp_clear_error()
    with runtime._owned_result(function(*arguments), name) as actual:
        np.testing.assert_allclose(
            actual.to_numpy(), expected, rtol=2e-12, atol=2e-12
        )


def qualify_case(
    runtime: CnumpyRuntime,
    *,
    name: str,
    function_name: str,
    argtypes: list[object],
    source_values: tuple[np.ndarray, ...],
    native_arguments: Callable[[tuple[object, ...]], tuple[object, ...]],
    numpy_operation: Callable[[], np.ndarray],
    warmups: int,
    samples: int,
    target_sample_ns: int,
) -> dict[str, object]:
    retained_before_sources = runtime.retained_bytes
    with ExitStack() as stack:
        sources = tuple(
            stack.enter_context(runtime.from_numpy(value))
            for value in source_values
        )
        pointers = tuple(source.pointer for source in sources)
        arguments = native_arguments(pointers)
        function = array_function(runtime, function_name, argtypes)
        retained_with_sources = runtime.retained_bytes
        validate(
            runtime,
            function,
            function_name,
            arguments,
            np.asarray(numpy_operation()),
        )
        if runtime.retained_bytes != retained_with_sources:
            raise RuntimeError(
                f"{function_name} validation retained native bytes"
            )

        native = native_operation(
            runtime, function, function_name, arguments
        )
        cnumpy_result = measure(
            native,
            warmups=warmups,
            samples=samples,
            target_sample_ns=target_sample_ns,
        )
        if runtime.retained_bytes != retained_with_sources:
            raise RuntimeError(
                f"{function_name} timed calls retained native bytes"
            )
        numpy_result = measure(
            lambda: numpy_operation(),
            warmups=warmups,
            samples=samples,
            target_sample_ns=target_sample_ns,
        )

    retained_after_sources = runtime.retained_bytes
    if retained_after_sources != retained_before_sources:
        raise RuntimeError(
            f"{function_name} source release retained native bytes"
        )
    cnumpy_median = float(cnumpy_result["median_ns_per_operation"])
    numpy_median = float(numpy_result["median_ns_per_operation"])
    return {
        "case": name,
        "function": function_name,
        "source_shapes": [list(value.shape) for value in source_values],
        "cnumpy": cnumpy_result,
        "numpy": numpy_result,
        "cnumpy_over_numpy": cnumpy_median / numpy_median,
        "retained_bytes": {
            "before_sources": retained_before_sources,
            "with_sources": retained_with_sources,
            "after_sources": retained_after_sources,
            "result_delta": 0,
            "final_delta": 0,
        },
    }


def run(arguments: argparse.Namespace) -> dict[str, object]:
    matrix_left = np.arange(64 * 64, dtype=np.float64).reshape(64, 64) / 97.0
    matrix_right = (
        np.arange(64 * 64, dtype=np.float64).reshape(64, 64) / 89.0
    )
    kron_right = np.asarray(
        [[1.0, -2.0, 3.0, 4.0], [5.0, 6.0, -7.0, 8.0],
         [9.0, 10.0, 11.0, -12.0], [13.0, -14.0, 15.0, 16.0]],
        dtype=np.float64,
    )
    cross_left = np.linspace(
        -3.0, 4.0, 32768 * 3, dtype=np.float64
    ).reshape(32768, 3)
    cross_right = np.linspace(
        5.0, -2.0, 32768 * 3, dtype=np.float64
    ).reshape(32768, 3)
    trace_source = (
        np.arange(2048 * 2048, dtype=np.float64).reshape(2048, 2048)
        / 101.0
    )

    cases: list[dict[str, object]] = []
    with CnumpyRuntime(arguments.dll) as runtime:
        runtime_baseline = runtime.retained_bytes
        case_specs = (
            {
                "name": "inner/f64/64x64",
                "function_name": "cnp_inner",
                "argtypes": [ctypes.c_void_p, ctypes.c_void_p],
                "source_values": (matrix_left, matrix_right),
                "native_arguments": lambda pointers: pointers,
                "numpy_operation": lambda: np.inner(matrix_left, matrix_right),
            },
            {
                "name": "tensordot/f64/64x64/axes1",
                "function_name": "cnp_tensordot",
                "argtypes": [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                ],
                "source_values": (matrix_left, matrix_right),
                "native_arguments": lambda pointers: pointers + (1, 1),
                "numpy_operation": lambda: np.tensordot(
                    matrix_left, matrix_right, axes=1
                ),
            },
            {
                "name": "kron/f64/64x64-by-4x4",
                "function_name": "cnp_kron",
                "argtypes": [ctypes.c_void_p, ctypes.c_void_p],
                "source_values": (matrix_left, kron_right),
                "native_arguments": lambda pointers: pointers,
                "numpy_operation": lambda: np.kron(matrix_left, kron_right),
            },
            {
                "name": "cross/f64/32768x3",
                "function_name": "cnp_cross",
                "argtypes": [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                ],
                "source_values": (cross_left, cross_right),
                "native_arguments": lambda pointers: pointers + (-1,),
                "numpy_operation": lambda: np.cross(cross_left, cross_right),
            },
            {
                "name": "trace/f64/2048x2048",
                "function_name": "cnp_trace",
                "argtypes": [
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                ],
                "source_values": (trace_source,),
                "native_arguments": lambda pointers: pointers + (0, 0, 1),
                "numpy_operation": lambda: np.trace(trace_source),
            },
        )
        for spec in case_specs:
            cases.append(
                qualify_case(
                    runtime,
                    **spec,
                    warmups=arguments.warmups,
                    samples=arguments.samples,
                    target_sample_ns=int(arguments.target_sample_ms * 1e6),
                )
            )
        if runtime.retained_bytes != runtime_baseline:
            raise RuntimeError(
                "product qualification did not restore runtime baseline"
            )

    return {
        "task": 14,
        "domain": "product_linalg_surface",
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
