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
DEFAULT_OUTPUT = (
    ROOT
    / "benchmark"
    / "runs-task14-linalg-statistics-surface"
    / "qualification.json"
)


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


def qualify_array_case(
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
    rtol: float = 2e-10,
    atol: float = 2e-10,
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

        runtime.dll.cnp_clear_error()
        with runtime._owned_result(
            function(*arguments), function_name
        ) as actual:
            actual_value = actual.to_numpy()
        expected = np.asarray(numpy_operation())
        if actual_value.shape != expected.shape:
            raise AssertionError(
                f"{function_name} shape {actual_value.shape} != {expected.shape}"
            )
        if actual_value.dtype != expected.dtype:
            raise AssertionError(
                f"{function_name} dtype {actual_value.dtype} != {expected.dtype}"
            )
        np.testing.assert_allclose(
            actual_value, expected, rtol=rtol, atol=atol
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

        cnumpy_result = measure(
            native_operation,
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
    return case_result(
        name=name,
        function_name=function_name,
        source_values=source_values,
        cnumpy_result=cnumpy_result,
        numpy_result=numpy_result,
        retained_before_sources=retained_before_sources,
        retained_with_sources=retained_with_sources,
        retained_after_sources=retained_after_sources,
    )


def qualify_qr_case(
    runtime: CnumpyRuntime,
    *,
    name: str,
    source_value: np.ndarray,
    warmups: int,
    samples: int,
    target_sample_ns: int,
) -> dict[str, object]:
    function_name = "cnp_linalg_qr"
    retained_before_sources = runtime.retained_bytes
    with runtime.from_numpy(source_value) as source:
        function = getattr(runtime.dll, function_name)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        function.restype = ctypes.c_int
        retained_with_sources = runtime.retained_bytes

        def call() -> tuple[int, ctypes.c_void_p, ctypes.c_void_p]:
            q_pointer = ctypes.c_void_p()
            r_pointer = ctypes.c_void_p()
            status = function(
                source.pointer,
                ctypes.byref(q_pointer),
                ctypes.byref(r_pointer),
            )
            return status, q_pointer, r_pointer

        runtime.dll.cnp_clear_error()
        status, q_pointer, r_pointer = call()
        if status != 0:
            raise runtime.native_error(function_name)
        with runtime._owned_result(
            q_pointer.value, f"{function_name}.q"
        ) as q, runtime._owned_result(
            r_pointer.value, f"{function_name}.r"
        ) as r:
            q_value = q.to_numpy()
            r_value = r.to_numpy()
        np.testing.assert_allclose(
            q_value @ r_value, source_value, rtol=2e-10, atol=2e-10
        )
        identity = np.broadcast_to(
            np.eye(q_value.shape[-1], dtype=q_value.dtype),
            q_value.shape[:-2]
            + (q_value.shape[-1], q_value.shape[-1]),
        )
        np.testing.assert_allclose(
            np.swapaxes(q_value.conj(), -1, -2) @ q_value,
            identity,
            rtol=2e-10,
            atol=2e-10,
        )
        if runtime.retained_bytes != retained_with_sources:
            raise RuntimeError("cnp_linalg_qr validation retained native bytes")

        def native_operation() -> None:
            status, q_pointer, r_pointer = call()
            if status != 0:
                raise runtime.native_error(function_name)
            runtime.dll.cnp_array_decref(q_pointer)
            runtime.dll.cnp_array_decref(r_pointer)
            assert_runtime_clear(runtime, function_name)

        cnumpy_result = measure(
            native_operation,
            warmups=warmups,
            samples=samples,
            target_sample_ns=target_sample_ns,
        )
        if runtime.retained_bytes != retained_with_sources:
            raise RuntimeError("cnp_linalg_qr timed calls retained native bytes")
        numpy_result = measure(
            lambda: np.linalg.qr(source_value),
            warmups=warmups,
            samples=samples,
            target_sample_ns=target_sample_ns,
        )

    retained_after_sources = runtime.retained_bytes
    if retained_after_sources != retained_before_sources:
        raise RuntimeError("cnp_linalg_qr source release retained native bytes")
    return case_result(
        name=name,
        function_name=function_name,
        source_values=(source_value,),
        cnumpy_result=cnumpy_result,
        numpy_result=numpy_result,
        retained_before_sources=retained_before_sources,
        retained_with_sources=retained_with_sources,
        retained_after_sources=retained_after_sources,
    )


def qualify_scalar_case(
    runtime: CnumpyRuntime,
    *,
    name: str,
    function_name: str,
    source_value: np.ndarray,
    native_arguments: Callable[[int], tuple[object, ...]],
    numpy_operation: Callable[[], float],
    warmups: int,
    samples: int,
    target_sample_ns: int,
) -> dict[str, object]:
    retained_before_sources = runtime.retained_bytes
    with runtime.from_numpy(source_value) as source:
        function = getattr(runtime.dll, function_name)
        function.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int]
        function.restype = ctypes.c_double
        arguments = native_arguments(source.pointer)
        retained_with_sources = runtime.retained_bytes
        runtime.dll.cnp_clear_error()
        actual = function(*arguments)
        assert_runtime_clear(runtime, function_name)
        np.testing.assert_allclose(
            actual, numpy_operation(), rtol=2e-12, atol=2e-12
        )

        def native_operation() -> None:
            function(*arguments)
            assert_runtime_clear(runtime, function_name)

        cnumpy_result = measure(
            native_operation,
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
    return case_result(
        name=name,
        function_name=function_name,
        source_values=(source_value,),
        cnumpy_result=cnumpy_result,
        numpy_result=numpy_result,
        retained_before_sources=retained_before_sources,
        retained_with_sources=retained_with_sources,
        retained_after_sources=retained_after_sources,
    )


def case_result(
    *,
    name: str,
    function_name: str,
    source_values: tuple[np.ndarray, ...],
    cnumpy_result: dict[str, object],
    numpy_result: dict[str, object],
    retained_before_sources: int,
    retained_with_sources: int,
    retained_after_sources: int,
) -> dict[str, object]:
    cnumpy_median = float(cnumpy_result["median_ns_per_operation"])
    numpy_median = float(numpy_result["median_ns_per_operation"])
    return {
        "case": name,
        "function": function_name,
        "source_shapes": [list(value.shape) for value in source_values],
        "source_dtypes": [str(value.dtype) for value in source_values],
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
    rng = np.random.default_rng(20260804)
    raw_inverse = rng.normal(size=(32, 32))
    inverse_source = raw_inverse @ raw_inverse.T + np.eye(32)
    pinv_source = rng.normal(size=(40, 24))
    qr_source = rng.normal(size=(64, 32))
    norm_source = rng.normal(size=(256, 256))
    tensor_source = np.eye(16, dtype=np.float64).reshape(4, 4, 4, 4)
    tensor_rhs = rng.normal(size=(4, 4))
    observations = rng.normal(size=(16, 1024))

    cases: list[dict[str, object]] = []
    with CnumpyRuntime(arguments.dll) as runtime:
        runtime_baseline = runtime.retained_bytes
        common = {
            "warmups": arguments.warmups,
            "samples": arguments.samples,
            "target_sample_ns": int(arguments.target_sample_ms * 1e6),
        }
        cases.append(qualify_array_case(
            runtime,
            name="inv/f64/32x32",
            function_name="cnp_linalg_inv",
            argtypes=[ctypes.c_void_p],
            source_values=(inverse_source,),
            native_arguments=lambda pointers: pointers,
            numpy_operation=lambda: np.linalg.inv(inverse_source),
            rtol=2e-9,
            atol=2e-9,
            **common,
        ))
        cases.append(qualify_array_case(
            runtime,
            name="norm/f64/256x256/ord1-axis1",
            function_name="cnp_linalg_norm",
            argtypes=[ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int],
            source_values=(norm_source,),
            native_arguments=lambda pointers: pointers + (b"1", 1),
            numpy_operation=lambda: np.linalg.norm(
                norm_source, ord=1, axis=1
            ),
            **common,
        ))
        cases.append(qualify_scalar_case(
            runtime,
            name="norm_ext/f64/256x256/ord1",
            function_name="cnp_linalg_norm_ext",
            source_value=norm_source,
            native_arguments=lambda pointer: (pointer, 1.0, -1),
            numpy_operation=lambda: float(np.linalg.norm(norm_source, ord=1)),
            **common,
        ))
        for function_name in ("cnp_linalg_pinv", "cnp_pinv"):
            cases.append(qualify_array_case(
                runtime,
                name=f"{function_name}/f64/40x24",
                function_name=function_name,
                argtypes=[ctypes.c_void_p, ctypes.c_double],
                source_values=(pinv_source,),
                native_arguments=lambda pointers: pointers + (1e-12,),
                numpy_operation=lambda: np.linalg.pinv(
                    pinv_source, rcond=1e-12
                ),
                rtol=2e-8,
                atol=2e-8,
                **common,
            ))
        cases.append(qualify_qr_case(
            runtime,
            name="qr/f64/64x32/reduced",
            source_value=qr_source,
            **common,
        ))
        cases.append(qualify_array_case(
            runtime,
            name="tensorinv/f64/4x4x4x4/ind2",
            function_name="cnp_linalg_tensorinv",
            argtypes=[ctypes.c_void_p, ctypes.c_int],
            source_values=(tensor_source,),
            native_arguments=lambda pointers: pointers + (2,),
            numpy_operation=lambda: np.linalg.tensorinv(tensor_source, ind=2),
            **common,
        ))
        cases.append(qualify_array_case(
            runtime,
            name="tensorsolve/f64/4x4x4x4",
            function_name="cnp_linalg_tensorsolve",
            argtypes=[
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
            ],
            source_values=(tensor_source, tensor_rhs),
            native_arguments=lambda pointers: pointers + (None,),
            numpy_operation=lambda: np.linalg.tensorsolve(
                tensor_source, tensor_rhs
            ),
            **common,
        ))
        cases.append(qualify_array_case(
            runtime,
            name="corrcoef/f64/16x1024",
            function_name="cnp_corrcoef",
            argtypes=[ctypes.c_void_p, ctypes.c_void_p],
            source_values=(observations,),
            native_arguments=lambda pointers: pointers + (None,),
            numpy_operation=lambda: np.corrcoef(observations),
            **common,
        ))
        cases.append(qualify_array_case(
            runtime,
            name="cov/f64/16x1024/rowvar-ddof1",
            function_name="cnp_cov",
            argtypes=[
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
            ],
            source_values=(observations,),
            native_arguments=lambda pointers: pointers + (None, 1, 1),
            numpy_operation=lambda: np.cov(
                observations, rowvar=True, ddof=1
            ),
            **common,
        ))
        if runtime.retained_bytes != runtime_baseline:
            raise RuntimeError(
                "linalg/statistics qualification did not restore runtime baseline"
            )

    return {
        "task": 14,
        "domain": "linalg_statistics_surface",
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
