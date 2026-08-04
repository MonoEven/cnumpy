"""Calibrated NumPy benchmark worker."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import platform
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import TextIO

import numpy as np
import scipy
from scipy import special as scipy_special

if __package__:
    from .report import validate_job_case, write_jobs_tsv
else:
    from report import validate_job_case, write_jobs_tsv


_JOB_FIELDS = ("id", "category", "operation", "dtype", "size", "rows", "cols", "axis")
_INTEGER_JOB_FIELDS = ("size", "rows", "cols", "axis")


@dataclass(slots=True)
class PreparedCase:
    """A benchmark operation whose setup is already complete."""

    invoke: Callable[[], object]
    validation: Callable[[], object]
    expected_shape: tuple[int, ...]
    logical_dtype: str = "f64"
    validation_mode: str = "numeric"
    expected_finite_members: np.ndarray | None = None


def _positive_length(size: int) -> int:
    if type(size) is not int or size <= 0:
        raise ValueError("size must be a positive integer")
    return size


def general_vector(size: int) -> np.ndarray:
    """Build the cross-runtime deterministic general-purpose vector."""
    indices = np.arange(_positive_length(size), dtype=np.int64)
    return ((indices * 37 + 11) % 1009) / 1009.0 + 0.01


def binary_vector(size: int) -> np.ndarray:
    """Build the deterministic second operand used by binary operations."""
    indices = np.arange(_positive_length(size), dtype=np.int64)
    return ((indices * 53 + 19) % 1013) / 1013.0 + 0.02


def predicate_vector(size: int) -> np.ndarray:
    """Build the repeated floating-edge input used by predicate benchmarks."""
    pattern = np.array(
        [np.nan, np.inf, -np.inf, 0.0, -0.0, 1.25, -2.5, 3.0],
        dtype=np.float64,
    )
    return np.resize(pattern, _positive_length(size))


def _bitwise_vector(size: int, kind: str) -> np.ndarray:
    """Build deterministic bounded int64 operands for bitwise benchmarks."""
    indices = np.arange(_positive_length(size), dtype=np.int64)
    if kind == "left":
        return ((indices * 37 + 11) % 4096) - 2048
    if kind == "right":
        return ((indices * 53 + 19) % 4096) - 2048
    if kind == "shift":
        return (indices * 5 + 3) % 8
    raise ValueError(f"unknown bitwise vector kind: {kind!r}")


def sorting_vector(size: int, pattern: str = "duplicates") -> np.ndarray:
    """Build deterministic duplicate-rich or NaN-heavy sorting input."""
    indices = np.arange(_positive_length(size), dtype=np.int64)
    result = (((indices * 48271 + 17) % 65521) % 4096).astype(np.float64)
    if pattern == "duplicates":
        return result
    if pattern == "nan":
        result[::2] = np.nan
        return result
    raise ValueError(f"unknown sorting input pattern: {pattern!r}")


def set_vector(size: int, pattern: str = "duplicates") -> np.ndarray:
    """Build Task7's deterministic duplicate-rich or NaN-heavy set input."""
    indices = np.arange(_positive_length(size), dtype=np.int64)
    result = (indices % 256).astype(np.float64)
    if pattern == "duplicates":
        return result
    if pattern == "nan":
        result[::2] = np.nan
        return result
    raise ValueError(f"unknown set input pattern: {pattern!r}")


def bitpack_vector(size: int) -> np.ndarray:
    """Build Task 8's deterministic byte-valued binary input."""
    indices = np.arange(_positive_length(size), dtype=np.int64)
    return (((indices * 5 + 3) % 2) != 0).astype(np.uint8)


def unpackbits_vector(size: int) -> np.ndarray:
    """Build Task 8's deterministic full-range byte input."""
    indices = np.arange(_positive_length(size), dtype=np.int64)
    return ((indices * 73 + 19) % 256).astype(np.uint8)


WEIGHTED_CHOICE_POPULATION_SIZE = 257
WEIGHTED_CHOICE_SELECTED_INDEX = 193


def weighted_choice_inputs() -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed float64 population and one-hot probability vector."""
    population = np.arange(WEIGHTED_CHOICE_POPULATION_SIZE, dtype=np.float64)
    probabilities = np.zeros(WEIGHTED_CHOICE_POPULATION_SIZE, dtype=np.float64)
    probabilities[WEIGHTED_CHOICE_SELECTED_INDEX] = 1.0
    return population, probabilities


def validate_weighted_choice_output(
    actual: object, size: int
) -> np.ndarray:
    """Validate every weighted-choice sample against the deterministic oracle."""
    expected_size = _positive_length(size)
    result = np.asarray(actual)
    if result.shape != (expected_size,) or result.dtype != np.dtype(np.float64):
        raise ValueError(
            "weighted choice full-output dtype and shape must be float64[size]"
        )
    if not bool(np.all(result == float(WEIGHTED_CHOICE_SELECTED_INDEX))):
        raise ValueError(
            "weighted choice full-output contains a value outside the one-hot population member"
        )
    return result


def symmetric_indexing_matrix(
    rows: int, cols: int, *, strided: bool = False
) -> np.ndarray:
    """Build equal C-contiguous or transposed-view axis-block operands."""
    rows = _positive_length(rows)
    cols = _positive_length(cols)
    if rows != cols:
        raise ValueError("symmetric indexing matrix must be square")
    row_indices = np.arange(rows, dtype=np.intp)[:, None]
    column_indices = np.arange(cols, dtype=np.intp)[None, :]
    symmetric_indices = (
        np.minimum(row_indices, column_indices) * cols
        + np.maximum(row_indices, column_indices)
    )
    matrix = general_vector(rows * cols)[symmetric_indices]
    return matrix.T if strided else matrix


def searchsorted_vectors(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a sorted source and deterministic full-size search values."""
    length = _positive_length(size)
    indices = np.arange(length, dtype=np.int64)
    source = (indices * 2).astype(np.float64)
    values = ((indices * 48271 + 17) % (2 * length + 1)).astype(
        np.float64
    )
    return source, values


def validate_argsort_indices(
    indices: np.ndarray,
    operand: np.ndarray,
) -> np.ndarray:
    """Validate a complete argsort permutation without imposing tie order."""
    if not isinstance(indices, np.ndarray) or indices.shape != operand.shape:
        raise ValueError("argsort indices must match the operand shape")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("argsort indices must have an integer dtype")
    size = operand.size
    if np.any(indices < 0) or np.any(indices >= size):
        raise ValueError("argsort indices must be within operand bounds")
    if not np.array_equal(np.sort(indices), np.arange(size, dtype=indices.dtype)):
        raise ValueError("argsort indices must be a complete permutation")
    ordered_values = operand[indices]
    nan_mask = np.isnan(ordered_values)
    if np.any(nan_mask[:-1] & ~nan_mask[1:]):
        raise ValueError("argsort values contain a finite key after a NaN key")
    finite_values = ordered_values[~nan_mask]
    if np.any(finite_values[1:] < finite_values[:-1]):
        raise ValueError("argsort values must be nondecreasing")
    return indices


def validate_partition_values(
    values: np.ndarray,
    operand: np.ndarray,
    kth: int,
) -> np.ndarray:
    """Validate a one-dimensional partition and return a stable signature."""
    if not isinstance(values, np.ndarray) or values.shape != operand.shape:
        raise ValueError("partition values must match the operand shape")
    if values.dtype != operand.dtype:
        raise ValueError("partition values must preserve the operand dtype")
    if values.ndim != 1:
        raise ValueError("partition benchmark validation requires one dimension")
    if type(kth) is not int or kth < 0 or kth >= operand.size:
        raise ValueError("partition kth is out of bounds")

    sorted_values = np.sort(values, kind="stable")
    sorted_operand = np.sort(operand, kind="stable")
    if not np.array_equal(sorted_values, sorted_operand, equal_nan=True):
        raise ValueError("partition values are not a permutation of the operand")

    pivot = values[kth]
    before = values[:kth]
    after = values[kth + 1 :]
    if np.isnan(pivot):
        if np.any(~np.isnan(after)):
            raise ValueError("partition contains a finite value after a NaN pivot")
    else:
        if np.any(np.isnan(before)) or np.any(before > pivot):
            raise ValueError("partition contains a value above the kth pivot before kth")
        finite_after = after[~np.isnan(after)]
        if np.any(finite_after < pivot):
            raise ValueError("partition contains a value below the kth pivot after kth")
    return np.nan_to_num(sorted_values, nan=8192.0)


def validate_argpartition_indices(
    indices: np.ndarray,
    operand: np.ndarray,
    kth: int,
) -> np.ndarray:
    """Validate an argpartition permutation and return normalized values."""
    if not isinstance(indices, np.ndarray) or indices.shape != operand.shape:
        raise ValueError("argpartition indices must match the operand shape")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("argpartition indices must have an integer dtype")
    size = operand.size
    if np.any(indices < 0) or np.any(indices >= size):
        raise ValueError("argpartition indices must be within operand bounds")
    if not np.array_equal(
        np.sort(indices), np.arange(size, dtype=indices.dtype)
    ):
        raise ValueError("argpartition indices must be a complete permutation")
    return validate_partition_values(operand[indices], operand, kth)


_SORTING_CASES = {
    "sort": ("sort", "quicksort", "duplicates"),
    "argsort": ("argsort", "quicksort", "duplicates"),
    "sort_mergesort": ("sort", "mergesort", "duplicates"),
    "argsort_mergesort": ("argsort", "mergesort", "duplicates"),
    "sort_heapsort": ("sort", "heapsort", "duplicates"),
    "argsort_heapsort": ("argsort", "heapsort", "duplicates"),
    "sort_stable": ("sort", "stable", "duplicates"),
    "argsort_stable": ("argsort", "stable", "duplicates"),
    "sort_stable_nan": ("sort", "stable", "nan"),
    "argsort_stable_nan": ("argsort", "stable", "nan"),
    "partition": ("partition", "introselect", "duplicates"),
    "argpartition": ("argpartition", "introselect", "duplicates"),
    "partition_nan": ("partition", "introselect", "nan"),
    "argpartition_nan": ("argpartition", "introselect", "nan"),
}


def make_product_vector(size: int) -> np.ndarray:
    """Build the stable, near-one input used by the full product workload."""
    indices = np.arange(_positive_length(size), dtype=np.int64)
    return 1.0 + ((((indices * 37 + 11) % 1009) - 504) * 1e-9)


def read_jobs_tsv(source: str | Path | TextIO) -> list[dict[str, object]]:
    """Read and strictly validate benchmark jobs from TSV."""
    if isinstance(source, (str, Path)):
        with Path(source).open("r", encoding="utf-8", newline="") as stream:
            return _read_jobs_stream(stream)
    return _read_jobs_stream(source)


def _read_jobs_stream(source: TextIO) -> list[dict[str, object]]:
    reader = csv.reader(source, delimiter="\t", strict=True)
    try:
        header = next(reader)
    except StopIteration as error:
        raise ValueError("jobs TSV is empty") from error
    if tuple(header) != _JOB_FIELDS:
        raise ValueError(f"jobs TSV header must be exactly {_JOB_FIELDS!r}")

    cases: list[dict[str, object]] = []
    for row_number, row in enumerate(reader, start=2):
        if len(row) != len(_JOB_FIELDS):
            raise ValueError(f"jobs TSV row {row_number} must have {len(_JOB_FIELDS)} fields")
        case: dict[str, object] = dict(zip(_JOB_FIELDS, row, strict=True))
        for field in _INTEGER_JOB_FIELDS:
            try:
                case[field] = int(case[field])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"jobs TSV row {row_number} field {field!r} must be an integer"
                ) from error
        cases.append(case)

    write_jobs_tsv(cases, io.StringIO())
    return cases


def prepare_case(case: dict[str, object], *, seed: int = 12345) -> PreparedCase:
    """Perform setup and strictly dispatch one catalog operation."""
    validate_job_case(case)
    seed = _validate_seed(seed)
    operation = case["operation"]
    size = case["size"]
    rows = case["rows"]
    cols = case["cols"]
    axis = case["axis"]

    if operation == "zeros":
        invoke = lambda: np.zeros(size, dtype=np.float64)
        return _prepared(invoke, (size,))
    if operation == "ones":
        invoke = lambda: np.ones(size, dtype=np.float64)
        return _prepared(invoke, (size,))
    if operation == "arange":
        invoke = lambda: np.arange(size, dtype=np.float64)
        return _prepared(invoke, (size,))
    if operation == "random":
        timed_rng = np.random.default_rng(seed)
        validation_rng = np.random.default_rng(seed)
        invoke = lambda: timed_rng.random(size)
        validate = lambda: validation_rng.random(size)
        return _prepared(
            invoke,
            (size,),
            validation=validate,
            validation_mode="shape",
        )
    if operation == "choice_weighted":
        population, probabilities = weighted_choice_inputs()
        timed_rng = np.random.RandomState(seed)
        validation_rng = np.random.RandomState(seed)
        invoke = lambda: timed_rng.choice(
            population, size=size, replace=True, p=probabilities
        )

        def validate() -> np.ndarray:
            result = validation_rng.choice(
                population, size=size, replace=True, p=probabilities
            )
            return validate_weighted_choice_output(result, size)

        return _prepared(invoke, (size,), validation=validate)
    if operation == "linspace":
        invoke = lambda: np.linspace(0.0, 1.0, size, dtype=np.float64)
        return _prepared(invoke, (size,))
    if operation == "real_if_close":
        operand = np.zeros(size, dtype=np.complex128)
        invoke = lambda: np.real_if_close(operand, tol=100.0)
        return _prepared(invoke, (size,))

    unary_operations: dict[str, Callable[[np.ndarray], object]] = {
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "expm1": np.expm1,
        "sqrt": np.sqrt,
        "log": np.log,
        "log2": np.log2,
        "log10": np.log10,
        "log1p": np.log1p,
        "absolute": np.absolute,
        "floor": np.floor,
        "tanh": np.tanh,
        "angle": np.angle,
        "real": np.real,
        "imag": np.imag,
    }
    unary = unary_operations.get(operation)
    if unary is not None:
        operand = general_vector(size)
        if operation == "absolute":
            operand = operand - 0.5
        elif operation == "angle":
            operand = operand - 0.5
        elif operation == "floor":
            operand = operand * 100.0
        invoke = lambda: unary(operand)
        return _prepared(invoke, (size,))

    predicate_operations: dict[str, Callable[[np.ndarray], object]] = {
        "isnan": np.isnan,
        "isinf": np.isinf,
        "isfinite": np.isfinite,
        "signbit": np.signbit,
    }
    predicate = predicate_operations.get(operation)
    if predicate is not None:
        operand = predicate_vector(size)
        invoke = lambda: predicate(operand)
        return _prepared(invoke, (size,), logical_dtype="bool")

    object_kind_predicates: dict[str, Callable[[object], bool]] = {
        "iscomplexobj": np.iscomplexobj,
        "isrealobj": np.isrealobj,
        "isscalar": np.isscalar,
    }
    object_kind_predicate = object_kind_predicates.get(operation)
    if object_kind_predicate is not None:
        operand = general_vector(size)
        invoke = lambda: object_kind_predicate(operand)
        return _prepared(invoke, (), logical_dtype="bool")

    if operation == "divmod":
        left = general_vector(size)
        right = binary_vector(size)
        invoke = lambda: np.divmod(left, right)
        validate = lambda: np.stack(np.divmod(left, right), axis=0)
        return _prepared(
            invoke,
            (2, size),
            validation=validate,
        )

    binary_operations: dict[str, Callable[[np.ndarray, np.ndarray], object]] = {
        "add": np.add,
        "subtract": np.subtract,
        "multiply": np.multiply,
        "divide": np.divide,
        "power": np.power,
        "float_power": np.float_power,
        "heaviside": np.heaviside,
        "maximum": np.maximum,
        "minimum": np.minimum,
        "fmax": np.fmax,
        "fmin": np.fmin,
        "logaddexp": np.logaddexp,
        "logaddexp2": np.logaddexp2,
        "equal": np.equal,
    }
    binary = binary_operations.get(operation)
    if binary is not None:
        left = general_vector(size)
        right = binary_vector(size)
        invoke = lambda: binary(left, right)
        return _prepared(
            invoke,
            (size,),
            logical_dtype="bool" if operation == "equal" else "f64",
        )

    logical_operations: dict[str, Callable[..., object]] = {
        "logical_and": np.logical_and,
        "logical_or": np.logical_or,
        "logical_xor": np.logical_xor,
        "logical_not": np.logical_not,
    }
    logical = logical_operations.get(operation)
    if logical is not None:
        left = general_vector(size)
        if operation == "logical_not":
            invoke = lambda: logical(left)
        else:
            right = binary_vector(size)
            invoke = lambda: logical(left, right)
        return _prepared(invoke, (size,), logical_dtype="bool")

    bitwise_operations: dict[str, Callable[..., object]] = {
        "bitwise_and": np.bitwise_and,
        "bitwise_or": np.bitwise_or,
        "bitwise_xor": np.bitwise_xor,
        "invert": np.invert,
        "left_shift": np.left_shift,
        "right_shift": np.right_shift,
    }
    bitwise = bitwise_operations.get(operation)
    if bitwise is not None:
        left = _bitwise_vector(size, "left")
        if operation == "invert":
            invoke = lambda: bitwise(left)
        else:
            right_kind = "shift" if operation in {"left_shift", "right_shift"} else "right"
            right = _bitwise_vector(size, right_kind)
            invoke = lambda: bitwise(left, right)
        return _prepared(invoke, (size,), logical_dtype="i64")

    integer_operations: dict[str, Callable[..., object]] = {
        "gcd": np.gcd,
        "lcm": np.lcm,
    }
    integer_operation = integer_operations.get(operation)
    if integer_operation is not None:
        left = _bitwise_vector(size, "left")
        right = _bitwise_vector(size, "right")
        invoke = lambda: integer_operation(left, right)
        return _prepared(invoke, (size,), logical_dtype="i64")

    if operation in {"convolve", "correlate"}:
        signal_function = (
            np.convolve if operation == "convolve" else np.correlate
        )
        data = general_vector(size)
        kernel = binary_vector(min(size, 8))
        invoke = lambda: signal_function(data, kernel, mode="same")
        return _prepared(invoke, (size,))

    if operation == "allclose":
        left = general_vector(size)
        right = left.copy()
        return _prepared(
            lambda: np.allclose(left, right),
            (),
            validation=lambda: np.int64(np.allclose(left, right)),
            logical_dtype="i64",
        )

    if operation == "add_into":
        left = general_vector(size)
        right = binary_vector(size)
        destination = np.empty_like(left)
        return _prepared(
            lambda: np.add(left, right, out=destination),
            (size,),
        )
    if operation == "sqrt_into":
        source = general_vector(size)
        destination = np.empty_like(source)
        return _prepared(
            lambda: np.sqrt(source, out=destination),
            (size,),
        )
    if operation == "cumsum_into":
        source = general_vector(size)
        destination = np.empty_like(source)
        return _prepared(
            lambda: np.cumsum(source, out=destination),
            (size,),
        )
    if operation in {"pipeline_separate", "pipeline_batch"}:
        left = general_vector(size)
        right = binary_vector(size)
        add_out = np.empty_like(left)
        sqrt_out = np.empty_like(left)

        def pipeline() -> np.float64:
            np.add(left, right, out=add_out)
            np.sqrt(add_out, out=sqrt_out)
            return np.sum(sqrt_out)

        return _prepared(pipeline, ())

    if operation == "average":
        operand = general_vector(size)
        weights = binary_vector(size)
        return _prepared(
            lambda: np.average(operand, weights=weights),
            (),
        )

    reduction_operations: dict[str, Callable[[np.ndarray], object]] = {
        "sum": np.sum,
        "mean": np.mean,
        "std": np.std,
        "max": np.max,
        "min": np.min,
        "argmax": np.argmax,
        "cumsum": np.cumsum,
        "prod": np.prod,
    }
    reduction = reduction_operations.get(operation)
    if reduction is not None:
        operand = make_product_vector(size) if operation == "prod" else general_vector(size)
        invoke = lambda: reduction(operand)
        expected_shape = (size,) if operation == "cumsum" else ()
        logical_dtype = "i64" if operation == "argmax" else "f64"
        return _prepared(invoke, expected_shape, logical_dtype=logical_dtype)

    if operation in {"sum_axis_last", "cumsum_axis_last"}:
        matrix = _matrix(rows, cols)
        if operation == "sum_axis_last":
            return _prepared(
                lambda: np.sum(matrix, axis=axis),
                (rows,),
            )
        return _prepared(
            lambda: np.cumsum(matrix, axis=axis),
            (rows, cols),
        )

    if operation in {
        "softmax",
        "softmax_axis_last",
        "softmax_axis0_strided",
        "log_softmax",
        "log_softmax_axis_last",
        "log_softmax_axis0_strided",
    }:
        source = (
            general_vector(size)
            if operation in {"softmax", "log_softmax"}
            else misc_axis_matrix(
                rows,
                cols,
                strided=operation.endswith("_strided"),
            )
        )
        is_log = operation.startswith("log_softmax")
        function = (
            scipy_special.log_softmax if is_log else scipy_special.softmax
        )
        invoke = lambda: function(source, axis=axis)
        validate = (
            (lambda: validate_log_softmax_output(invoke(), source, axis))
            if is_log
            else (lambda: validate_softmax_output(invoke(), source, axis))
        )
        return _prepared(
            invoke,
            source.shape,
            validation=validate,
        )

    if operation in {"trapz", "trapz_axis_last", "trapz_axis0_strided"}:
        source = (
            general_vector(size)
            if operation == "trapz"
            else misc_axis_matrix(
                rows,
                cols,
                strided=operation.endswith("_strided"),
            )
        )
        invoke = lambda: np.trapz(source, dx=0.25, axis=axis)
        expected_shape = tuple(
            dimension
            for index, dimension in enumerate(source.shape)
            if index != axis % source.ndim
        )
        return _prepared(
            invoke,
            expected_shape,
            validation=lambda: validate_trapz_output(
                invoke(), source, axis, 0.25
            ),
        )

    if operation in {
        "packbits",
        "packbits_axis_last",
        "packbits_axis0_strided",
    }:
        source = (
            bitpack_vector(size)
            if operation == "packbits"
            else bitpack_vector(size).reshape((rows, cols))
        )
        if operation.endswith("_strided"):
            source = source.T
        invoke = lambda: np.packbits(source, axis=axis, bitorder="big")
        expected_shape = list(source.shape)
        normalized_axis = axis % source.ndim
        expected_shape[normalized_axis] = (
            expected_shape[normalized_axis] + 7
        ) // 8
        return _prepared(
            invoke,
            tuple(expected_shape),
            validation=lambda: validate_packbits_output(
                invoke(), source, axis
            ),
            logical_dtype="u8",
        )

    if operation in {
        "unpackbits",
        "unpackbits_axis_last",
        "unpackbits_axis0_strided",
    }:
        source = (
            unpackbits_vector(size)
            if operation == "unpackbits"
            else unpackbits_vector(size).reshape((rows, cols))
        )
        if operation.endswith("_strided"):
            source = source.T
        invoke = lambda: np.unpackbits(
            source, axis=axis, count=None, bitorder="big"
        )
        expected_shape = list(source.shape)
        normalized_axis = axis % source.ndim
        expected_shape[normalized_axis] *= 8
        return _prepared(
            invoke,
            tuple(expected_shape),
            validation=lambda: validate_unpackbits_output(
                invoke(), source, axis
            ),
            logical_dtype="u8",
        )

    if operation in {
        "unique_duplicates",
        "unique_nan",
        "intersect1d_duplicates",
        "union1d_duplicates",
        "setdiff1d_duplicates",
        "setxor1d_duplicates",
        "in1d_duplicates",
        "isin_duplicates",
    }:
        left = set_vector(
            size, "nan" if operation == "unique_nan" else "duplicates"
        )
        right = np.concatenate(
            (
                np.arange(0, 128, 2, dtype=np.float64),
                np.arange(256, 320, dtype=np.float64),
            )
        )
        set_operations: dict[str, Callable[[], np.ndarray]] = {
            "unique_duplicates": lambda: np.unique(left),
            "unique_nan": lambda: np.unique(left, equal_nan=True),
            "intersect1d_duplicates": lambda: np.intersect1d(left, right),
            "union1d_duplicates": lambda: np.union1d(left, right),
            "setdiff1d_duplicates": lambda: np.setdiff1d(left, right),
            "setxor1d_duplicates": lambda: np.setxor1d(left, right),
            "in1d_duplicates": lambda: np.in1d(left, right),
            "isin_duplicates": lambda: np.isin(left, right),
        }
        invoke = set_operations[operation]
        expected = invoke()
        logical_dtype = (
            "bool"
            if operation in {"in1d_duplicates", "isin_duplicates"}
            else "f64"
        )
        validation_mode = "numeric"
        expected_finite_members = None
        if operation == "unique_nan":
            validation_mode = "numeric_nan"
            expected_finite_members = np.asarray(
                sorted({float(value) for value in left if not np.isnan(value)}),
                dtype=np.float64,
            )
        return _prepared(
            invoke,
            expected.shape,
            logical_dtype=logical_dtype,
            validation_mode=validation_mode,
            expected_finite_members=expected_finite_members,
        )

    sorting_case = _SORTING_CASES.get(operation)
    if sorting_case is not None:
        family, kind, pattern = sorting_case
        operand = sorting_vector(size, pattern)
        if family == "sort":
            invoke = lambda: np.sort(operand, kind=kind)
            if pattern == "nan":
                expected_finite_members = np.asarray(
                    sorted(float(value) for value in operand if not np.isnan(value)),
                    dtype=np.float64,
                )
                return _prepared(
                    invoke,
                    (size,),
                    validation_mode="numeric_nan",
                    expected_finite_members=expected_finite_members,
                )
            return _prepared(invoke, (size,))
        invoke = lambda: np.argsort(operand, kind=kind)
        if family == "argsort":
            validate = lambda: validate_argsort_indices(
                np.argsort(operand, kind=kind), operand
            )
            return _prepared(
                invoke,
                (size,),
                validation=validate,
                logical_dtype="i64",
            )
        kth = size // 2
        if family == "partition":
            invoke = lambda: np.partition(operand, kth, kind=kind)
            validate = lambda: validate_partition_values(
                np.partition(operand, kth, kind=kind), operand, kth
            )
        else:
            invoke = lambda: np.argpartition(operand, kth, kind=kind)
            validate = lambda: validate_argpartition_indices(
                np.argpartition(operand, kth, kind=kind), operand, kth
            )
        return _prepared(invoke, (size,), validation=validate)

    if operation in {"searchsorted", "searchsorted_right"}:
        source, values = searchsorted_vectors(size)
        side = "right" if operation == "searchsorted_right" else "left"
        invoke = lambda: np.searchsorted(source, values, side=side)
        return _prepared(invoke, (size,), logical_dtype="i64")

    if operation in {"digitize", "digitize_decreasing"}:
        bins, values = searchsorted_vectors(size)
        if operation == "digitize_decreasing":
            bins = bins[::-1]
        invoke = lambda: np.digitize(values, bins, right=False)
        return _prepared(invoke, (size,), logical_dtype="i64")

    if operation == "lexsort":
        secondary = sorting_vector(size)
        primary = binary_vector(size)
        invoke = lambda: np.lexsort((secondary, primary))
        return _prepared(invoke, (size,), logical_dtype="i64")

    if operation == "msort":
        operand = sorting_vector(size)
        invoke = lambda: np.msort(operand)
        return _prepared(invoke, (size,))

    if operation == "sort_complex":
        operand = sorting_vector(size)
        invoke = lambda: np.sort_complex(operand)

        def validate_sort_complex() -> np.ndarray:
            result = np.sort_complex(operand)
            if result.dtype != np.dtype(np.complex128):
                raise ValueError(
                    "sort_complex benchmark result must be complex128"
                )
            if np.any(result.imag != 0.0):
                raise ValueError(
                    "sort_complex real-input result must have zero imaginary parts"
                )
            return result.real

        return _prepared(
            invoke, (size,), validation=validate_sort_complex
        )

    if operation == "copy":
        operand = general_vector(size)
        invoke = lambda: operand.copy()
        return _prepared(invoke, (size,))
    if operation == "reshape":
        operand = general_vector(size)
        invoke = lambda: operand.reshape((size, 1))
        return _prepared(invoke, (size, 1))
    if operation == "flatten":
        operand = general_vector(size).reshape((1, size))
        invoke = lambda: operand.flatten()
        return _prepared(invoke, (size,))
    if operation in {"atleast_1d", "atleast_2d", "atleast_3d"}:
        operand = general_vector(size)
        minimum_ndim = int(operation[-2])
        function = getattr(np, operation)
        invoke = lambda: function(operand)
        expected_shape = (
            (size,)
            if minimum_ndim == 1
            else (1, size)
            if minimum_ndim == 2
            else (1, size, 1)
        )
        return _prepared(invoke, expected_shape)

    if operation in {"take", "compress"}:
        operand = general_vector(size)
        selected = np.arange(0, size, 2, dtype=np.intp)
        expected_shape = (selected.size,)
        if operation == "take":
            return _prepared(
                lambda: np.take(operand, selected),
                expected_shape,
            )
        condition = np.zeros(size, dtype=np.bool_)
        condition[selected] = True
        return _prepared(
            lambda: np.compress(condition, operand),
            expected_shape,
        )

    axis0_indexing_operations = {
        "take_axis0_block",
        "take_axis0_strided",
        "compress_axis0_block",
        "compress_axis0_strided",
    }
    if operation in axis0_indexing_operations:
        operand = symmetric_indexing_matrix(
            rows, cols, strided=operation.endswith("_strided")
        )
        selected = np.arange(0, rows, 2, dtype=np.intp)
        expected_shape = (selected.size, cols)
        if operation.startswith("take_"):
            return _prepared(
                lambda: np.take(operand, selected, axis=axis),
                expected_shape,
            )
        condition = np.zeros(rows, dtype=np.bool_)
        condition[selected] = True
        return _prepared(
            lambda: np.compress(condition, operand, axis=axis),
            expected_shape,
        )

    matrix_operations = {
        "matmul",
        "dot",
        "det",
        "inv",
        "norm",
        "solve",
        "cholesky",
        "einsum",
        "eig",
        "svd",
        "lstsq",
        "transpose_copy",
        "concatenate",
    }
    if operation in matrix_operations:
        matrix = _matrix(rows, cols)
        if operation == "matmul":
            invoke = lambda: np.matmul(matrix, matrix)
            return _prepared(invoke, (rows, cols))
        if operation == "dot":
            invoke = lambda: np.dot(matrix, matrix)
            return _prepared(invoke, (rows, cols))
        if operation == "det":
            invoke = lambda: np.linalg.det(matrix)
            return _prepared(invoke, ())
        if operation == "inv":
            invoke = lambda: np.linalg.inv(matrix)
            return _prepared(invoke, (rows, cols))
        if operation == "norm":
            invoke = lambda: np.linalg.norm(matrix)
            return _prepared(invoke, ())
        if operation == "solve":
            expected_solution = binary_vector(rows)
            rhs = matrix @ expected_solution
            matrix_before = matrix.copy()
            rhs_before = rhs.copy()
            invoke = lambda: np.linalg.solve(matrix, rhs)
            return _prepared(
                invoke,
                (rows,),
                validation=lambda: validate_task9_solve_output(
                    invoke(), matrix, rhs, matrix_before, rhs_before,
                    expected_solution,
                ),
            )
        if operation == "cholesky":
            invoke = lambda: np.linalg.cholesky(matrix)
            return _prepared(invoke, (rows, cols))
        if operation == "einsum":
            matrix = task9_einsum_matrix(rows)
            matrix_before = matrix.copy()
            invoke = lambda: np.einsum("ik,kj->ij", matrix, matrix)
            return _prepared(
                invoke,
                (rows, cols),
                validation=lambda: validate_task9_einsum_output(
                    invoke(), matrix, matrix_before,
                ),
            )
        if operation == "eig":
            matrix = task9_eig_matrix(rows)
            matrix_before = matrix.copy()
            invoke = lambda: np.linalg.eig(matrix)
            return _prepared(
                invoke,
                (rows,),
                validation=lambda: validate_task9_eig_output(
                    invoke(), matrix, matrix_before,
                ),
            )
        if operation == "svd":
            matrix_before = matrix.copy()
            invoke = lambda: np.linalg.svd(matrix, full_matrices=False)
            return _prepared(
                invoke,
                (rows,),
                validation=lambda: validate_task9_svd_output(
                    invoke(), matrix, matrix_before,
                ),
            )
        if operation == "lstsq":
            matrix = task9_lstsq_matrix(rows)
            expected_solution = binary_vector(rows)
            rhs = matrix @ expected_solution
            matrix_before = matrix.copy()
            rhs_before = rhs.copy()
            invoke = lambda: np.linalg.lstsq(matrix, rhs, rcond=None)
            return _prepared(
                invoke,
                (rows,),
                validation=lambda: validate_task9_lstsq_output(
                    invoke(), matrix, rhs, matrix_before, rhs_before,
                    expected_solution,
                ),
            )
        if operation == "transpose_copy":
            invoke = lambda: matrix.T.copy(order="C")
            return _prepared(invoke, (cols, rows))
        invoke = lambda: np.concatenate((matrix, matrix), axis=axis)
        return _prepared(invoke, (rows * 2, cols))

    if operation == "fft":
        operand = general_vector(size)

        def invoke_fft() -> object:
            return np.fft.fft(operand)

        def validate_fft() -> object:
            result = np.fft.fft(operand)
            return np.column_stack((result.real, result.imag))

        return _prepared(invoke_fft, (size, 2), validation=validate_fft)

    if operation in {
        "property_call",
        "property_cached",
        "nbytes_cached",
        "c_contiguous_cached",
        "f_contiguous_cached",
    }:
        owner = np.empty(1, dtype=np.float64)
        if operation in {"property_call", "property_cached"}:
            invoke = lambda: owner.size
            logical_dtype = "i64"
        elif operation == "nbytes_cached":
            invoke = lambda: owner.nbytes
            logical_dtype = "i64"
        elif operation == "c_contiguous_cached":
            invoke = lambda: owner.flags.c_contiguous
            logical_dtype = "bool"
        else:
            invoke = lambda: owner.flags.f_contiguous
            logical_dtype = "bool"
        return _prepared(invoke, (), logical_dtype=logical_dtype)
    if operation == "static_add_call":
        left = general_vector(1)
        right = binary_vector(1)
        destination = np.empty_like(left)
        invoke = lambda: (np.add(left, right, out=destination), destination.size)[1]
        return _prepared(invoke, (), logical_dtype="i64")

    raise ValueError(f"unknown benchmark operation: {operation!r}")


def _prepared(
    invoke: Callable[[], object],
    expected_shape: tuple[int, ...],
    *,
    validation: Callable[[], object] | None = None,
    logical_dtype: str = "f64",
    validation_mode: str = "numeric",
    expected_finite_members: np.ndarray | None = None,
) -> PreparedCase:
    return PreparedCase(
        invoke=invoke,
        validation=validation if validation is not None else invoke,
        expected_shape=expected_shape,
        logical_dtype=logical_dtype,
        validation_mode=validation_mode,
        expected_finite_members=expected_finite_members,
    )


def _matrix(rows: int, cols: int) -> np.ndarray:
    matrix = (general_vector(rows * cols) * 0.001).reshape((rows, cols))
    diagonal = np.diag_indices(min(rows, cols))
    matrix[diagonal] += 2.0
    return matrix


def task9_eig_matrix(size: int) -> np.ndarray:
    """Build a deterministic nonsymmetric triangular eig workload."""
    matrix = np.zeros((size, size), dtype=np.float64)
    diagonal = 2.0 + np.arange(size, dtype=np.float64) / max(size, 1)
    matrix[np.diag_indices(size)] = diagonal
    if size > 1:
        upper = np.triu_indices(size, 1)
        flat = np.arange(upper[0].size, dtype=np.int64)
        matrix[upper] = (((flat * 37 + 11) % 1009) + 1.0) * 1e-6
    return matrix


def task9_einsum_matrix(size: int) -> np.ndarray:
    """Build a dense-stored diagonal contraction with an exact full oracle."""
    matrix = np.zeros((size, size), dtype=np.float64)
    matrix[np.diag_indices(size)] = 1.0 + general_vector(size)
    return matrix


def task9_lstsq_matrix(size: int) -> np.ndarray:
    """Build a dense SPD workload with singular values exactly 3, 2, ..., 2."""
    return (
        2.0 * np.eye(size, dtype=np.float64)
        + np.full((size, size), 1.0 / size, dtype=np.float64)
    )


def _require_task9_unchanged(actual: np.ndarray, expected: np.ndarray, name: str) -> None:
    if not np.array_equal(actual, expected):
        raise ValueError(f"Task 9 {name} source changed during validation")


def _require_task9_allclose(
    actual: object,
    expected: object,
    name: str,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if actual_array.shape != expected_array.shape or not np.allclose(
        actual_array,
        expected_array,
        rtol=rtol,
        atol=atol,
    ):
        raise ValueError(f"Task 9 {name} complete result does not match reference")


def validate_task9_einsum_output(
    result: object,
    matrix: np.ndarray,
    matrix_before: np.ndarray,
) -> np.ndarray:
    actual = np.asarray(result)
    _require_task9_allclose(actual, matrix_before @ matrix_before, "einsum")
    _require_task9_unchanged(matrix, matrix_before, "einsum")
    return actual


def validate_task9_solve_output(
    result: object,
    matrix: np.ndarray,
    rhs: np.ndarray,
    matrix_before: np.ndarray,
    rhs_before: np.ndarray,
    expected_solution: np.ndarray,
) -> np.ndarray:
    actual = np.asarray(result)
    _require_task9_allclose(actual, expected_solution, "solve")
    _require_task9_allclose(matrix_before @ actual, rhs_before, "solve residual")
    _require_task9_unchanged(matrix, matrix_before, "solve matrix")
    _require_task9_unchanged(rhs, rhs_before, "solve rhs")
    return actual


def validate_task9_eig_output(
    result: object,
    matrix: np.ndarray,
    matrix_before: np.ndarray,
) -> np.ndarray:
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("Task 9 eig must return eigenvalues and eigenvectors")
    values = np.asarray(result[0])
    vectors = np.asarray(result[1])
    if values.shape != (matrix.shape[0],) or vectors.shape != matrix.shape:
        raise ValueError("Task 9 eig result shapes are incomplete")
    _require_task9_allclose(
        matrix_before @ vectors,
        vectors * values[np.newaxis, :],
        "eig decomposition",
        rtol=1e-8,
        atol=1e-10,
    )
    _require_task9_allclose(
        np.linalg.norm(vectors, axis=0),
        np.ones(matrix.shape[0]),
        "eig eigenvector norms",
        rtol=1e-9,
        atol=1e-11,
    )
    expected_values = np.sort(np.diag(matrix_before))
    if np.iscomplexobj(values) and np.any(np.abs(values.imag) > 1e-12):
        raise ValueError("Task 9 eig eigenvalues have unexpected imaginary parts")
    normalized_values = np.sort(np.real_if_close(values, tol=1000).real)
    _require_task9_allclose(
        normalized_values, expected_values, "eig eigenvalues",
        rtol=1e-10, atol=1e-12,
    )
    _require_task9_unchanged(matrix, matrix_before, "eig")
    return normalized_values


def validate_task9_svd_output(
    result: object,
    matrix: np.ndarray,
    matrix_before: np.ndarray,
) -> np.ndarray:
    if not isinstance(result, tuple) or len(result) != 3:
        raise ValueError("Task 9 SVD must return U, singular values, and Vh")
    left = np.asarray(result[0])
    singular = np.asarray(result[1])
    right = np.asarray(result[2])
    size = matrix.shape[0]
    if left.shape != matrix.shape or singular.shape != (size,) or right.shape != matrix.shape:
        raise ValueError("Task 9 SVD result shapes are incomplete")
    _require_task9_allclose(
        (left * singular[np.newaxis, :]) @ right,
        matrix_before,
        "SVD reconstruction",
        rtol=1e-9,
        atol=1e-11,
    )
    identity = np.eye(size, dtype=np.float64)
    _require_task9_allclose(left.T.conj() @ left, identity, "SVD left orthogonality")
    _require_task9_allclose(right @ right.T.conj(), identity, "SVD right orthogonality")
    if np.any(singular[:-1] < singular[1:]) or np.any(singular < 0.0):
        raise ValueError("Task 9 SVD singular values are not nonnegative descending")
    _require_task9_unchanged(matrix, matrix_before, "SVD")
    return singular


def validate_task9_lstsq_output(
    result: object,
    matrix: np.ndarray,
    rhs: np.ndarray,
    matrix_before: np.ndarray,
    rhs_before: np.ndarray,
    expected_solution: np.ndarray,
) -> np.ndarray:
    if not isinstance(result, tuple) or len(result) != 4:
        raise ValueError("Task 9 lstsq must return four results")
    solution = np.asarray(result[0])
    residuals = np.asarray(result[1])
    rank = result[2]
    singular = np.asarray(result[3])
    size = matrix.shape[0]
    if solution.shape != (size,) or residuals.shape != (0,) or singular.shape != (size,):
        raise ValueError("Task 9 lstsq result shapes are incomplete")
    if not isinstance(rank, (int, np.integer)) or int(rank) != size:
        raise ValueError("Task 9 lstsq rank result is not full rank")
    _require_task9_allclose(solution, expected_solution, "lstsq solution")
    _require_task9_allclose(matrix_before @ solution, rhs_before, "lstsq residual")
    if np.any(singular[:-1] < singular[1:]) or np.any(singular < 0.0):
        raise ValueError("Task 9 lstsq singular values are not nonnegative descending")
    expected_singular = np.concatenate(
        (np.array([3.0]), np.full(size - 1, 2.0))
    )
    _require_task9_allclose(
        singular, expected_singular, "lstsq singular values",
        rtol=1e-10, atol=1e-12,
    )
    _require_task9_unchanged(matrix, matrix_before, "lstsq matrix")
    _require_task9_unchanged(rhs, rhs_before, "lstsq rhs")
    return solution


def misc_axis_matrix(
    rows: int, cols: int, *, strided: bool = False
) -> np.ndarray:
    """Build Task 8's deterministic matrix or its transposed native view."""
    matrix = _matrix(rows, cols)
    return matrix.T if strided else matrix


def validate_softmax_output(
    result: object,
    source: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Validate every softmax output element against stable f64 arithmetic."""
    actual = np.asarray(result)
    if actual.dtype != np.dtype(np.float64) or actual.shape != source.shape:
        raise ValueError(
            "softmax full-output dtype and shape must match float64 source"
        )
    shifted = source - np.max(source, axis=axis, keepdims=True)
    expected = np.exp(shifted)
    expected /= np.sum(expected, axis=axis, keepdims=True)
    if not np.array_equal(actual, expected):
        raise ValueError(
            "softmax full-output values do not match stable reference"
        )
    return actual


def validate_log_softmax_output(
    result: object,
    source: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Validate every log-softmax element against stable f64 arithmetic."""
    actual = np.asarray(result)
    if actual.dtype != np.dtype(np.float64) or actual.shape != source.shape:
        raise ValueError(
            "log_softmax full-output dtype and shape must match float64 source"
        )
    shifted = source - np.max(source, axis=axis, keepdims=True)
    expected = shifted - np.log(
        np.sum(np.exp(shifted), axis=axis, keepdims=True)
    )
    if not np.array_equal(actual, expected):
        raise ValueError(
            "log_softmax full-output values do not match stable reference"
        )
    return actual


def validate_trapz_output(
    result: object,
    source: np.ndarray,
    axis: int,
    dx: float,
) -> object:
    """Validate every trapz result value against NumPy 1.25 arithmetic."""
    actual = np.asarray(result)
    normalized_axis = axis % source.ndim
    before = [slice(None)] * source.ndim
    after = before.copy()
    before[normalized_axis] = slice(None, -1)
    after[normalized_axis] = slice(1, None)
    expected = np.sum(
        (source[tuple(before)] + source[tuple(after)]) * 0.5 * dx,
        axis=normalized_axis,
    )
    if actual.dtype != np.dtype(np.float64) or actual.shape != np.shape(expected):
        raise ValueError(
            "trapz full-output dtype and shape must match float64 reference"
        )
    if not np.array_equal(actual, expected):
        raise ValueError(
            "trapz full-output values do not match NumPy 1.25 arithmetic"
        )
    return result


def validate_packbits_output(
    result: object,
    source: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Validate every packed byte with an independent big-order encoder."""
    actual = np.asarray(result)
    normalized_axis = axis % source.ndim
    moved = np.moveaxis(source, normalized_axis, -1)
    packed_length = (moved.shape[-1] + 7) // 8
    padding = packed_length * 8 - moved.shape[-1]
    padded = np.pad(
        moved != 0,
        [(0, 0)] * (moved.ndim - 1) + [(0, padding)],
        constant_values=False,
    )
    groups = padded.reshape((*moved.shape[:-1], packed_length, 8))
    weights = np.array(
        [128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint16
    )
    expected = np.sum(
        groups.astype(np.uint16) * weights,
        axis=-1,
        dtype=np.uint16,
    ).astype(np.uint8)
    expected = np.moveaxis(expected, -1, normalized_axis)
    if actual.dtype != np.dtype(np.uint8) or actual.shape != expected.shape:
        raise ValueError(
            "packbits full-output dtype and shape must match uint8 reference"
        )
    if not np.array_equal(actual, expected):
        raise ValueError(
            "packbits full-output bytes do not match big-order reference"
        )
    return actual


def validate_unpackbits_output(
    result: object,
    source: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Validate every unpacked bit with an independent big-order decoder."""
    actual = np.asarray(result)
    normalized_axis = axis % source.ndim
    moved = np.moveaxis(source, normalized_axis, -1)
    shifts = np.arange(7, -1, -1, dtype=np.uint8)
    expected = ((moved[..., :, None] >> shifts) & np.uint8(1)).reshape(
        (*moved.shape[:-1], moved.shape[-1] * 8)
    )
    expected = np.moveaxis(expected, -1, normalized_axis)
    if actual.dtype != np.dtype(np.uint8) or actual.shape != expected.shape:
        raise ValueError(
            "unpackbits full-output dtype and shape must match uint8 reference"
        )
    if not np.array_equal(actual, expected):
        raise ValueError(
            "unpackbits full-output bytes do not match big-order reference"
        )
    return actual


def validation_signature(
    value: object,
    expected_shape: Sequence[int],
    *,
    logical_dtype: str = "f64",
    mode: str = "numeric",
    expected_finite_members: np.ndarray | None = None,
) -> dict[str, object]:
    """Return a compact, cross-runtime validation signature."""
    if mode not in {"numeric", "numeric_nan", "shape"}:
        raise ValueError(f"unknown validation mode: {mode!r}")
    if logical_dtype not in {"f64", "i64", "u8", "bool"}:
        raise ValueError(f"unknown logical dtype: {logical_dtype!r}")

    logical_shape: list[int] = []
    for index, dimension in enumerate(expected_shape):
        if type(dimension) is not int or dimension < 0:
            raise ValueError(f"expected_shape[{index}] must be a non-negative integer")
        logical_shape.append(dimension)

    result = np.asarray(value)
    if list(result.shape) != logical_shape:
        raise ValueError(
            f"result shape {list(result.shape)!r} does not match expected shape {logical_shape!r}"
        )
    if logical_dtype == "bool":
        if result.dtype != np.bool_:
            raise ValueError("validation value does not match logical dtype 'bool'")
    elif logical_dtype == "i64":
        if result.dtype == np.bool_ or not np.issubdtype(result.dtype, np.integer):
            raise ValueError("validation value does not match logical dtype 'i64'")
    elif logical_dtype == "u8":
        if result.dtype != np.dtype(np.uint8):
            raise ValueError("validation value does not match logical dtype 'u8'")
    elif not np.issubdtype(result.dtype, np.floating):
        raise ValueError("validation value does not match logical dtype 'f64'")
    if np.iscomplexobj(result):
        raise ValueError("validation value must be real-valued")
    try:
        finite_mask = np.isfinite(result)
    except TypeError as error:
        raise ValueError("validation value must be numeric") from error
    if mode != "numeric_nan" and not bool(np.all(finite_mask)):
        raise ValueError("validation value must contain only finite numbers")

    signature: dict[str, object] = {
        "mode": mode,
        "shape": logical_shape,
        "size": int(result.size),
        "logical_dtype": logical_dtype,
    }
    if mode == "shape":
        return signature

    flat = result.reshape(-1)
    if mode == "numeric_nan":
        if logical_dtype != "f64":
            raise ValueError("numeric_nan validation requires logical dtype 'f64'")
        if expected_finite_members is None:
            raise ValueError("numeric_nan validation requires expected finite members")
        expected_members = np.asarray(expected_finite_members)
        if (
            expected_members.ndim != 1
            or not np.issubdtype(expected_members.dtype, np.floating)
            or not bool(np.all(np.isfinite(expected_members)))
        ):
            raise ValueError("expected finite members must be a finite floating vector")
        if bool(np.any(np.isinf(flat))):
            raise ValueError("numeric_nan raw output must not contain infinities")
        nan_mask = np.isnan(flat)
        nan_count = int(np.count_nonzero(nan_mask))
        expected_nan_count = int(flat.size - expected_members.size)
        if expected_nan_count <= 0 or nan_count != expected_nan_count:
            raise ValueError(
                "numeric_nan raw output does not contain the expected NaN count"
            )
        first_nan_index = int(np.flatnonzero(nan_mask)[0])
        trailing_nan = bool(
            first_nan_index == expected_members.size
            and np.all(nan_mask[first_nan_index:])
        )
        if not trailing_nan:
            raise ValueError("numeric_nan raw output NaNs must form the complete tail")
        actual_finite_members = flat[~nan_mask]
        if not np.array_equal(actual_finite_members, expected_members):
            raise ValueError(
                "numeric_nan raw output finite members do not match the deterministic expectation"
            )
        normalized_flat = np.where(nan_mask, 8192.0, flat)
        signature.update(
            {
                "nan_count": nan_count,
                "first_nan_index": first_nan_index,
                "trailing_nan": trailing_nan,
                "finite_members_exact": True,
            }
        )
    else:
        if expected_finite_members is not None:
            raise ValueError(
                "expected finite members are only valid for numeric_nan mode"
            )
        normalized_flat = flat
    sample_indices = list(dict.fromkeys((0, flat.size // 2, flat.size - 1)))
    values = [float(normalized_flat[index]) for index in sample_indices]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("validation sample values must be finite")
    with np.errstate(over="ignore"):
        aggregate = float(np.sum(normalized_flat, dtype=np.float64))
    if not math.isfinite(aggregate):
        raise ValueError("validation sum must be finite")
    signature.update(
        {
            "sample_indices": sample_indices,
            "values": values,
            "sum": aggregate,
        }
    )
    return signature


def time_operation(
    callback: Callable[[], object],
    warmups: int,
    sample_count: int,
    target_sample_ns: int | float,
) -> tuple[int, list[float]]:
    """Calibrate a callback and collect positive nanoseconds-per-call samples."""
    if type(warmups) is not int or warmups < 0:
        raise ValueError("warmups must be a non-negative integer")
    if type(sample_count) is not int or sample_count <= 0 or sample_count % 2 == 0:
        raise ValueError("sample_count must be a positive odd integer")
    if (
        isinstance(target_sample_ns, bool)
        or not isinstance(target_sample_ns, Real)
        or not math.isfinite(target_sample_ns)
        or target_sample_ns <= 0
    ):
        raise ValueError("target_sample_ns must be a finite positive number")

    for _ in range(warmups):
        result = callback()
        del result

    inner_loops = 1
    while True:
        elapsed = _time_batch(callback, inner_loops)
        if elapsed >= target_sample_ns:
            break
        inner_loops *= 2

    samples_ns: list[float] = []
    for _ in range(sample_count):
        elapsed = _time_batch(callback, inner_loops)
        sample = elapsed / inner_loops
        if not math.isfinite(sample) or sample <= 0:
            raise RuntimeError("benchmark sample must be a finite positive ns/op value")
        samples_ns.append(float(sample))
    return inner_loops, samples_ns


def _time_batch(callback: Callable[[], object], inner_loops: int) -> int:
    started = time.perf_counter_ns()
    for _ in range(inner_loops):
        result = callback()
        del result
    elapsed = time.perf_counter_ns() - started
    if elapsed <= 0:
        raise RuntimeError("perf_counter_ns returned a non-positive elapsed duration")
    return elapsed


def run_worker(
    cases: Sequence[dict[str, object]],
    *,
    warmups: int,
    sample_count: int,
    target_sample_ns: int | float,
    seed: int = 12345,
) -> dict[str, object]:
    """Execute all cases and return a version-one NumPy worker document."""
    write_jobs_tsv(cases, io.StringIO())
    result_cases: list[dict[str, object]] = []
    for case in cases:
        prepared = prepare_case(case, seed=seed)

        validation_value = prepared.validation()
        validation = validation_signature(
            validation_value,
            prepared.expected_shape,
            logical_dtype=prepared.logical_dtype,
            mode=prepared.validation_mode,
            expected_finite_members=prepared.expected_finite_members,
        )
        del validation_value

        inner_loops, samples_ns = time_operation(
            prepared.invoke,
            warmups,
            sample_count,
            target_sample_ns,
        )
        result_case = dict(case)
        result_case.update(
            {
                "inner_loops": inner_loops,
                "samples_ns": samples_ns,
                "validation": validation,
            }
        )
        result_cases.append(result_case)
        del prepared

    timer_resolution_ns = time.get_clock_info("perf_counter").resolution * 1_000_000_000
    return {
        "schema_version": 1,
        "runtime": "numpy",
        "metadata": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "timer": "perf_counter_ns",
            "timer_resolution_ns": timer_resolution_ns,
            "warmups": warmups,
            "sample_count": sample_count,
            "target_sample_ns": target_sample_ns,
            "seed": _validate_seed(seed),
        },
        "cases": result_cases,
    }


def _non_negative_integer(argument: str) -> int:
    try:
        value = int(argument)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from error
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def _positive_odd_integer(argument: str) -> int:
    try:
        value = int(argument)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive odd integer") from error
    if value <= 0 or value % 2 == 0:
        raise argparse.ArgumentTypeError("must be a positive odd integer")
    return value


def _positive_finite_number(argument: str) -> float:
    try:
        value = float(argument)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a finite positive number") from error
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def _validate_seed(value: object) -> int:
    if type(value) is not int or not 0 <= value <= 2_147_483_647:
        raise ValueError("seed must be an integer in [0, 2147483647]")
    return value


def _seed_integer(argument: str) -> int:
    try:
        value = int(argument)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be an integer in [0, 2147483647]"
        ) from error
    try:
        return _validate_seed(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error).removeprefix("seed ")) from error


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmups", type=_non_negative_integer, default=5)
    parser.add_argument("--samples", type=_positive_odd_integer, default=15)
    parser.add_argument("--target-sample-ms", type=_positive_finite_number, default=20.0)
    parser.add_argument("--seed", type=_seed_integer, default=12345)
    return parser


def write_json_atomic(path: str | Path, document: object) -> None:
    """Durably replace a JSON file without exposing partially written output."""
    destination = Path(path)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(document, temporary, allow_nan=False, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Run the NumPy benchmark command-line worker."""
    arguments = _argument_parser().parse_args(argv)
    cases = read_jobs_tsv(arguments.jobs)
    document = run_worker(
        cases,
        warmups=arguments.warmups,
        sample_count=arguments.samples,
        target_sample_ns=arguments.target_sample_ms * 1_000_000,
        seed=arguments.seed,
    )
    write_json_atomic(arguments.output, document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
