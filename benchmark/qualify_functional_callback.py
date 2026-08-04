from __future__ import annotations

import argparse
import ctypes
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
    / "runs-task14-functional-callback-surface"
    / "qualification.json"
)
CNP_DOUBLE = 13

LINE_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int64,
    ctypes.c_void_p,
)
COORD_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_int,
    ctypes.c_void_p,
)
ITER_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_void_p)
UNARY_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_double, ctypes.c_double, ctypes.c_void_p
)
BULK_LINE_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_void_p,
)
BULK_COORD_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_int64,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_void_p,
)
BULK_ITER_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_void_p,
)
BULK_UNARY_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_void_p,
)
ARRAY_POINTER = ctypes.POINTER(ctypes.c_void_p)


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


def case_result(
    case: str,
    symbols: list[str],
    native: dict[str, object],
    numpy: dict[str, object],
    retained_before: int,
    retained_during: int,
    retained_after: int,
    callback_invocations: int | None = None,
    callback_elements: int | None = None,
) -> dict[str, object]:
    if retained_after != retained_before:
        raise RuntimeError(
            f"{case} retained native bytes: before={retained_before}, "
            f"after={retained_after}"
        )
    native_median = float(native["median_ns_per_operation"])
    numpy_median = float(numpy["median_ns_per_operation"])
    result = {
        "case": case,
        "symbols": symbols,
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
    if callback_invocations is not None or callback_elements is not None:
        if callback_invocations is None or callback_elements is None:
            raise RuntimeError("callback counts must be recorded together")
        if callback_invocations <= 0 or callback_elements <= 0:
            raise RuntimeError("callback counts must be positive")
        if callback_invocations >= callback_elements:
            raise RuntimeError(
                f"{case} did not batch callback elements: "
                f"invocations={callback_invocations}, elements={callback_elements}"
            )
        result["callback_work"] = {
            "bulk_invocations": callback_invocations,
            "logical_elements": callback_elements,
            "elements_per_invocation": callback_elements / callback_invocations,
        }
    return result


def bind(runtime: CnumpyRuntime) -> None:
    dll = runtime.dll
    dll.cnp_ahk_apply_along_axis_v2.argtypes = [
        BULK_LINE_CALLBACK,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64),
    ]
    dll.cnp_ahk_apply_along_axis_v2.restype = ctypes.c_void_p
    dll.cnp_ahk_apply_over_axes_v2.argtypes = [
        BULK_LINE_CALLBACK,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
    ]
    dll.cnp_ahk_apply_over_axes_v2.restype = ctypes.c_void_p
    dll.cnp_ahk_fromfunction_v2.argtypes = [
        BULK_COORD_CALLBACK,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64),
    ]
    dll.cnp_ahk_fromfunction_v2.restype = ctypes.c_void_p
    dll.cnp_ahk_fromiter_v2.argtypes = [
        BULK_ITER_CALLBACK,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int,
    ]
    dll.cnp_ahk_fromiter_v2.restype = ctypes.c_void_p
    for symbol in ("cnp_ahk_frompyfunc_v2", "cnp_ahk_vectorize_v2"):
        function = getattr(dll, symbol)
        function.argtypes = [
            BULK_UNARY_CALLBACK,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        function.restype = ctypes.c_void_p
    dll.cnp_ahk_piecewise_v2.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ARRAY_POINTER,
        BULK_UNARY_CALLBACK,
        ctypes.c_void_p,
    ]
    dll.cnp_ahk_piecewise_v2.restype = ctypes.c_void_p

    dll.cnp_apply_along_axis.argtypes = [
        LINE_CALLBACK,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    dll.cnp_apply_along_axis.restype = ctypes.c_void_p
    dll.cnp_apply_over_axes.argtypes = [
        LINE_CALLBACK,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    dll.cnp_apply_over_axes.restype = ctypes.c_void_p
    dll.cnp_fromfunction.argtypes = [
        COORD_CALLBACK,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_void_p,
    ]
    dll.cnp_fromfunction.restype = ctypes.c_void_p
    dll.cnp_fromiter.argtypes = [
        ITER_CALLBACK,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int,
    ]
    dll.cnp_fromiter.restype = ctypes.c_void_p
    for symbol in ("cnp_frompyfunc", "cnp_vectorize"):
        function = getattr(dll, symbol)
        function.argtypes = [
            UNARY_CALLBACK,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        function.restype = ctypes.c_void_p
    dll.cnp_piecewise.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ARRAY_POINTER,
        UNARY_CALLBACK,
        ctypes.c_void_p,
    ]
    dll.cnp_piecewise.restype = ctypes.c_void_p
    dll.cnp_select.argtypes = [
        ctypes.c_int,
        ARRAY_POINTER,
        ARRAY_POINTER,
        ctypes.c_double,
    ]
    dll.cnp_select.restype = ctypes.c_void_p
    dll.cnp_put_along_axis.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    dll.cnp_put_along_axis.restype = ctypes.c_int


def release_result(runtime: CnumpyRuntime, pointer: int | None, symbol: str) -> None:
    if not pointer:
        raise runtime.native_error(symbol)
    runtime.dll.cnp_array_decref(pointer)
    error = runtime.error_state()
    if error.status != 0:
        raise RuntimeError(f"{symbol} left a native error: {error}")


def run(arguments: argparse.Namespace) -> dict[str, object]:
    cases: list[dict[str, object]] = []

    callback_work = {
        "line": [0, 0],
        "coordinate": [0, 0],
        "iterator": [0, 0],
        "unary": [0, 0],
    }

    def reset_callback_work(name: str) -> None:
        callback_work[name][:] = [0, 0]

    def callback_snapshot(name: str) -> tuple[int, int]:
        return callback_work[name][0], callback_work[name][1]

    @BULK_LINE_CALLBACK
    def line_sum(
        lines: ctypes.POINTER(ctypes.c_double),
        line_count: int,
        line_length: int,
        results: ctypes.POINTER(ctypes.c_double),
        result_capacity: int,
        produced: ctypes.POINTER(ctypes.c_int64),
        _state: int,
    ) -> int:
        if result_capacity != line_count:
            return -13
        callback_work["line"][0] += 1
        callback_work["line"][1] += line_count
        for line_index in range(line_count):
            start = line_index * line_length
            results[line_index] = float(
                sum(lines[start + index] for index in range(line_length))
            )
        produced[0] = line_count
        return 0

    @BULK_COORD_CALLBACK
    def coordinate_value(
        coordinates: ctypes.POINTER(ctypes.c_int64),
        point_count: int,
        ndim: int,
        results: ctypes.POINTER(ctypes.c_double),
        result_capacity: int,
        produced: ctypes.POINTER(ctypes.c_int64),
        _state: int,
    ) -> int:
        if result_capacity != point_count:
            return -13
        callback_work["coordinate"][0] += 1
        callback_work["coordinate"][1] += point_count
        for point in range(point_count):
            start = point * ndim
            results[point] = float(
                sum(
                    (index + 1) * coordinates[start + index]
                    for index in range(ndim)
                )
            )
        produced[0] = point_count
        return 0

    iterator_index = ctypes.c_int64(0)

    @BULK_ITER_CALLBACK
    def iterator_value(
        results: ctypes.POINTER(ctypes.c_double),
        result_capacity: int,
        produced: ctypes.POINTER(ctypes.c_int64),
        state: int,
    ) -> int:
        pointer = ctypes.cast(state, ctypes.POINTER(ctypes.c_int64))
        callback_work["iterator"][0] += 1
        callback_work["iterator"][1] += result_capacity
        first = pointer.contents.value
        for index in range(result_capacity):
            results[index] = float((first + index) * 0.25 - 3.0)
        pointer.contents.value = first + result_capacity
        produced[0] = result_capacity
        return 0

    @BULK_UNARY_CALLBACK
    def unary_value(
        values: ctypes.POINTER(ctypes.c_double),
        value_count: int,
        results: ctypes.POINTER(ctypes.c_double),
        result_capacity: int,
        produced: ctypes.POINTER(ctypes.c_int64),
        _state: int,
    ) -> int:
        if result_capacity != value_count:
            return -13
        callback_work["unary"][0] += 1
        callback_work["unary"][1] += value_count
        for index in range(value_count):
            results[index] = values[index] * 1.5 + 0.25
        produced[0] = value_count
        return 0

    with CnumpyRuntime(arguments.dll) as runtime:
        bind(runtime)
        runtime_baseline = runtime.retained_bytes

        values = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
        retained_before = runtime.retained_bytes
        with runtime.from_numpy(values) as source:
            retained_during = runtime.retained_bytes
            expected = np.apply_along_axis(np.sum, 1, values)
            with runtime._owned_result(
                runtime.dll.cnp_ahk_apply_along_axis_v2(
                    line_sum, None, 1, source.pointer, 0, None
                ),
                "cnp_ahk_apply_along_axis_v2",
            ) as actual:
                np.testing.assert_array_equal(actual.to_numpy(), expected)

            def native_apply_along() -> None:
                release_result(
                    runtime,
                    runtime.dll.cnp_ahk_apply_along_axis_v2(
                        line_sum, None, 1, source.pointer, 0, None
                    ),
                    "cnp_ahk_apply_along_axis_v2",
                )

            reset_callback_work("line")
            native_apply_along()
            apply_along_work = callback_snapshot("line")
            reset_callback_work("line")
            native, numpy = timed_pair(
                native_apply_along,
                lambda: np.apply_along_axis(np.sum, 1, values),
                arguments,
            )
            if runtime.retained_bytes != retained_during:
                raise RuntimeError("apply_along_axis retained result bytes")
        cases.append(case_result(
            "apply_along_axis/sum/f64/64x64",
            ["cnp_ahk_apply_along_axis_v2"], native, numpy,
            retained_before, retained_during, runtime.retained_bytes,
            *apply_along_work,
        ))

        values = np.arange(16 * 16 * 16, dtype=np.float64).reshape(16, 16, 16)
        axes = (ctypes.c_int * 2)(0, 2)
        retained_before = runtime.retained_bytes
        with runtime.from_numpy(values) as source:
            retained_during = runtime.retained_bytes
            expected = np.apply_over_axes(np.sum, values, [0, 2])
            with runtime._owned_result(
                runtime.dll.cnp_ahk_apply_over_axes_v2(
                    line_sum, None, 2, axes, source.pointer
                ),
                "cnp_ahk_apply_over_axes_v2",
            ) as actual:
                np.testing.assert_array_equal(actual.to_numpy(), expected)

            def native_apply_over() -> None:
                release_result(
                    runtime,
                    runtime.dll.cnp_ahk_apply_over_axes_v2(
                        line_sum, None, 2, axes, source.pointer
                    ),
                    "cnp_ahk_apply_over_axes_v2",
                )

            reset_callback_work("line")
            native_apply_over()
            apply_over_work = callback_snapshot("line")
            reset_callback_work("line")
            native, numpy = timed_pair(
                native_apply_over,
                lambda: np.apply_over_axes(np.sum, values, [0, 2]),
                arguments,
            )
            if runtime.retained_bytes != retained_during:
                raise RuntimeError("apply_over_axes retained result bytes")
        cases.append(case_result(
            "apply_over_axes/sum/f64/16x16x16/axes0-2",
            ["cnp_ahk_apply_over_axes_v2"], native, numpy,
            retained_before, retained_during, runtime.retained_bytes,
            *apply_over_work,
        ))

        shape_tuple = (64, 64)
        shape = (ctypes.c_int64 * 2)(*shape_tuple)
        retained_before = runtime.retained_bytes

        def native_fromfunction() -> None:
            release_result(
                runtime,
                runtime.dll.cnp_ahk_fromfunction_v2(
                    coordinate_value, None, 2, shape
                ),
                "cnp_ahk_fromfunction_v2",
            )

        with runtime._owned_result(
            runtime.dll.cnp_ahk_fromfunction_v2(
                coordinate_value, None, 2, shape
            ),
            "cnp_ahk_fromfunction_v2",
        ) as actual:
            expected = np.fromfunction(
                lambda row, column: row + 2 * column,
                shape_tuple,
                dtype=np.int64,
            ).astype(np.float64)
            np.testing.assert_array_equal(actual.to_numpy(), expected)
        reset_callback_work("coordinate")
        native_fromfunction()
        fromfunction_work = callback_snapshot("coordinate")
        reset_callback_work("coordinate")
        native, numpy = timed_pair(
            native_fromfunction,
            lambda: np.fromfunction(
                lambda row, column: row + 2 * column,
                shape_tuple,
                dtype=np.int64,
            ).astype(np.float64),
            arguments,
        )
        cases.append(case_result(
            "fromfunction/f64/64x64",
            ["cnp_ahk_fromfunction_v2"], native, numpy,
            retained_before, retained_before, runtime.retained_bytes,
            *fromfunction_work,
        ))

        iterator_count = 8192
        retained_before = runtime.retained_bytes

        def native_fromiter() -> None:
            iterator_index.value = 0
            release_result(
                runtime,
                runtime.dll.cnp_ahk_fromiter_v2(
                    iterator_value,
                    ctypes.byref(iterator_index),
                    iterator_count,
                    CNP_DOUBLE,
                ),
                "cnp_ahk_fromiter_v2",
            )

        iterator_index.value = 0
        with runtime._owned_result(
            runtime.dll.cnp_ahk_fromiter_v2(
                iterator_value,
                ctypes.byref(iterator_index),
                iterator_count,
                CNP_DOUBLE,
            ),
            "cnp_ahk_fromiter_v2",
        ) as actual:
            expected = np.fromiter(
                (index * 0.25 - 3.0 for index in range(iterator_count)),
                dtype=np.float64,
                count=iterator_count,
            )
            np.testing.assert_array_equal(actual.to_numpy(), expected)
        reset_callback_work("iterator")
        native_fromiter()
        fromiter_work = callback_snapshot("iterator")
        reset_callback_work("iterator")
        native, numpy = timed_pair(
            native_fromiter,
            lambda: np.fromiter(
                (index * 0.25 - 3.0 for index in range(iterator_count)),
                dtype=np.float64,
                count=iterator_count,
            ),
            arguments,
        )
        cases.append(case_result(
            "fromiter/f64/8192",
            ["cnp_ahk_fromiter_v2"], native, numpy,
            retained_before, retained_before, runtime.retained_bytes,
            *fromiter_work,
        ))

        values = np.linspace(-4.0, 4.0, 8192, dtype=np.float64)
        for symbol, case_name in (
            ("cnp_ahk_frompyfunc_v2", "frompyfunc"),
            ("cnp_ahk_vectorize_v2", "vectorize"),
        ):
            retained_before = runtime.retained_bytes
            with runtime.from_numpy(values) as source:
                retained_during = runtime.retained_bytes
                function = getattr(runtime.dll, symbol)
                with runtime._owned_result(
                    function(unary_value, None, source.pointer), symbol
                ) as actual:
                    expected = np.vectorize(
                        lambda value: value * 1.5 + 0.25,
                        otypes=[np.float64],
                    )(values)
                    np.testing.assert_array_equal(actual.to_numpy(), expected)

                def native_unary(function: object = function, symbol: str = symbol) -> None:
                    release_result(
                        runtime,
                        function(unary_value, None, source.pointer),
                        symbol,
                    )

                if symbol == "cnp_ahk_frompyfunc_v2":
                    ufunc = np.frompyfunc(
                        lambda value: value * 1.5 + 0.25, 1, 1
                    )
                    numpy_operation = lambda: ufunc(values).astype(np.float64)
                else:
                    vectorized = np.vectorize(
                        lambda value: value * 1.5 + 0.25,
                        otypes=[np.float64],
                    )
                    numpy_operation = lambda: vectorized(values)
                reset_callback_work("unary")
                native_unary()
                unary_work = callback_snapshot("unary")
                reset_callback_work("unary")
                native, numpy = timed_pair(
                    native_unary, numpy_operation, arguments
                )
                if runtime.retained_bytes != retained_during:
                    raise RuntimeError(f"{symbol} retained result bytes")
            cases.append(case_result(
                f"{case_name}/f64/8192",
                [symbol], native, numpy,
                retained_before, retained_during, runtime.retained_bytes,
                *unary_work,
            ))

        values = np.linspace(-4.0, 4.0, 8192, dtype=np.float64)
        condition_values = (values < -1.0, values >= 1.0)
        retained_before = runtime.retained_bytes
        with runtime.from_numpy(values) as source, runtime.from_numpy(
            condition_values[0]
        ) as condition_one, runtime.from_numpy(
            condition_values[1]
        ) as condition_two:
            retained_during = runtime.retained_bytes
            conditions = (ctypes.c_void_p * 2)(
                condition_one.pointer, condition_two.pointer
            )
            expected = np.piecewise(
                values,
                condition_values,
                [lambda value: value * 1.5 + 0.25] * 2,
            )
            with runtime._owned_result(
                runtime.dll.cnp_ahk_piecewise_v2(
                    source.pointer, 2, conditions, unary_value, None
                ),
                "cnp_ahk_piecewise_v2",
            ) as actual:
                np.testing.assert_array_equal(actual.to_numpy(), expected)

            def native_piecewise() -> None:
                release_result(
                    runtime,
                    runtime.dll.cnp_ahk_piecewise_v2(
                        source.pointer, 2, conditions, unary_value, None
                    ),
                    "cnp_ahk_piecewise_v2",
                )

            reset_callback_work("unary")
            native_piecewise()
            piecewise_work = callback_snapshot("unary")
            reset_callback_work("unary")
            native, numpy = timed_pair(
                native_piecewise,
                lambda: np.piecewise(
                    values,
                    condition_values,
                    [lambda value: value * 1.5 + 0.25] * 2,
                ),
                arguments,
            )
            if runtime.retained_bytes != retained_during:
                raise RuntimeError("cnp_piecewise retained result bytes")
        cases.append(case_result(
            "piecewise/f64/8192/two-conditions",
            ["cnp_ahk_piecewise_v2"], native, numpy,
            retained_before, retained_during, runtime.retained_bytes,
            *piecewise_work,
        ))

        condition_values = (
            np.arange(128, dtype=np.int64).reshape(128, 1) % 2 == 0,
            np.arange(128, dtype=np.int64).reshape(1, 128) % 3 == 0,
        )
        choice_values = (
            np.linspace(0.0, 1.0, 128).reshape(1, 128),
            np.linspace(2.0, 3.0, 128).reshape(128, 1),
        )
        retained_before = runtime.retained_bytes
        with runtime.from_numpy(condition_values[0]) as condition_one, runtime.from_numpy(
            condition_values[1]
        ) as condition_two, runtime.from_numpy(
            choice_values[0]
        ) as choice_one, runtime.from_numpy(choice_values[1]) as choice_two:
            retained_during = runtime.retained_bytes
            conditions = (ctypes.c_void_p * 2)(
                condition_one.pointer, condition_two.pointer
            )
            choices = (ctypes.c_void_p * 2)(
                choice_one.pointer, choice_two.pointer
            )
            expected = np.select(condition_values, choice_values, default=-1.5)
            with runtime._owned_result(
                runtime.dll.cnp_select(2, conditions, choices, -1.5),
                "cnp_select",
            ) as actual:
                np.testing.assert_array_equal(actual.to_numpy(), expected)

            def native_select() -> None:
                release_result(
                    runtime,
                    runtime.dll.cnp_select(2, conditions, choices, -1.5),
                    "cnp_select",
                )

            native, numpy = timed_pair(
                native_select,
                lambda: np.select(condition_values, choice_values, default=-1.5),
                arguments,
            )
            if runtime.retained_bytes != retained_during:
                raise RuntimeError("cnp_select retained result bytes")
        cases.append(case_result(
            "select/f64/128x128/two-conditions",
            ["cnp_select"], native, numpy,
            retained_before, retained_during, runtime.retained_bytes,
        ))

        destination_values = np.zeros((128, 128), dtype=np.float64)
        index_values = np.tile(np.arange(64, dtype=np.int64), (128, 1))
        value_values = np.linspace(-1.0, 1.0, 64).reshape(1, 64)
        numpy_destination = destination_values.copy()
        retained_before = runtime.retained_bytes
        with runtime.from_numpy(destination_values) as destination, runtime.from_numpy(
            index_values
        ) as indices, runtime.from_numpy(value_values) as values:
            retained_during = runtime.retained_bytes

            def native_put() -> None:
                runtime.dll.cnp_clear_error()
                status = runtime.dll.cnp_put_along_axis(
                    destination.pointer, indices.pointer, values.pointer, 1
                )
                if status != 0:
                    raise runtime.native_error("cnp_put_along_axis", status)

            def numpy_put() -> None:
                np.put_along_axis(
                    numpy_destination, index_values, value_values, axis=1
                )

            native_put()
            numpy_put()
            np.testing.assert_array_equal(
                destination.to_numpy(), numpy_destination
            )
            native, numpy = timed_pair(native_put, numpy_put, arguments)
            if runtime.retained_bytes != retained_during:
                raise RuntimeError("cnp_put_along_axis retained native bytes")
        cases.append(case_result(
            "put_along_axis/f64/128x128/64-updates",
            ["cnp_put_along_axis"], native, numpy,
            retained_before, retained_during, runtime.retained_bytes,
        ))

        if runtime.retained_bytes != runtime_baseline:
            raise RuntimeError(
                "functional/callback qualification did not restore runtime baseline"
            )

    return {
        "task": 14,
        "domain": "functional_callback_surface",
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
