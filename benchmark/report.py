"""Deterministic statistics and validation for benchmark worker results."""

from __future__ import annotations

import csv
import copy
import json
import math
import statistics
from collections.abc import Sequence
from numbers import Real
from pathlib import Path
from typing import TextIO


_CATALOG_KEYS = {"schema_version", "size_sets", "templates"}
_SIZE_SET_NAMES = {"vector", "matrix", "fft"}
_TEMPLATE_KEYS = {"category", "operation", "dtype", "shape_kind", "axis", "profiles"}
_PROFILE_NAMES = {"focus", "standard", "full"}
_SHAPE_KINDS = {"bridge", "vector", "matrix", "fft"}
_CASE_FIELDS = ("id", "category", "operation", "dtype", "size", "rows", "cols", "axis")
_FORBIDDEN_WIRE_CHARACTERS = ("\t", "\r", "\n", '"')
_OPERATION_GROUPS = (
    ("creation", "vector", -1, ("zeros", "ones", "arange", "random", "linspace")),
    ("random", "vector", -1, ("choice_weighted",)),
    ("unary", "vector", -1, ("sin", "cos", "exp", "expm1", "sqrt", "log", "log2", "log10", "log1p", "absolute", "floor", "tanh", "angle", "real", "imag", "real_if_close")),
    ("binary", "vector", -1, ("add", "subtract", "multiply", "divide", "divmod", "power", "float_power", "heaviside", "maximum", "minimum", "fmax", "fmin", "logaddexp", "logaddexp2")),
    (
        "logical",
        "vector",
        -1,
        (
            "logical_and",
            "logical_or",
            "logical_xor",
            "logical_not",
            "isnan",
            "isinf",
            "isfinite",
            "signbit",
            "iscomplexobj",
            "isrealobj",
            "isscalar",
        ),
    ),
    ("bitwise", "vector", -1, ("bitwise_and", "bitwise_or", "bitwise_xor", "invert", "left_shift", "right_shift")),
    ("integer", "vector", -1, ("gcd", "lcm")),
    ("signal", "vector", -1, ("convolve", "correlate")),
    ("comparison", "vector", -1, ("allclose", "equal")),
    ("reduction", "vector", -1, ("sum", "mean", "average", "std", "max", "min", "argmax", "cumsum", "prod")),
    ("reduction", "matrix", 1, ("sum_axis_last", "cumsum_axis_last")),
    (
        "misc_axis",
        "vector",
        -1,
        ("softmax", "log_softmax", "trapz"),
    ),
    (
        "misc_axis",
        "matrix",
        1,
        ("softmax_axis_last", "log_softmax_axis_last", "trapz_axis_last"),
    ),
    (
        "misc_axis",
        "matrix",
        0,
        (
            "softmax_axis0_strided",
            "log_softmax_axis0_strided",
            "trapz_axis0_strided",
            "packbits_axis0_strided",
            "unpackbits_axis0_strided",
        ),
    ),
    ("misc_axis", "vector", -1, ("packbits", "unpackbits")),
    (
        "misc_axis",
        "matrix",
        1,
        ("packbits_axis_last", "unpackbits_axis_last"),
    ),
    (
        "sorting",
        "vector",
        -1,
        (
            "sort",
            "argsort",
            "sort_mergesort",
            "argsort_mergesort",
            "sort_heapsort",
            "argsort_heapsort",
            "sort_stable",
            "argsort_stable",
            "sort_stable_nan",
            "argsort_stable_nan",
            "partition",
            "argpartition",
            "partition_nan",
            "argpartition_nan",
            "searchsorted",
            "searchsorted_right",
            "digitize",
            "digitize_decreasing",
            "lexsort",
            "msort",
            "sort_complex",
        ),
    ),
    (
        "set",
        "vector",
        -1,
        (
            "unique_duplicates",
            "unique_nan",
            "intersect1d_duplicates",
            "union1d_duplicates",
            "setdiff1d_duplicates",
            "setxor1d_duplicates",
            "in1d_duplicates",
            "isin_duplicates",
        ),
    ),
    (
        "shape",
        "vector",
        -1,
        (
            "copy",
            "reshape",
            "flatten",
            "atleast_1d",
            "atleast_2d",
            "atleast_3d",
        ),
    ),
    ("indexing", "vector", -1, ("take", "compress")),
    (
        "indexing",
        "matrix",
        0,
        (
            "take_axis0_block",
            "take_axis0_strided",
            "compress_axis0_block",
            "compress_axis0_strided",
        ),
    ),
    ("preallocated", "vector", -1, ("add_into", "sqrt_into", "cumsum_into")),
    ("pipeline", "vector", -1, ("pipeline_separate", "pipeline_batch")),
    (
        "linalg",
        "matrix",
        -1,
        (
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
        ),
    ),
    ("shape", "matrix", -1, ("transpose_copy",)),
    ("shape", "matrix", 0, ("concatenate",)),
    ("fft", "fft", -1, ("fft",)),
    (
        "bridge",
        "bridge",
        -1,
        (
            "property_call",
            "property_cached",
            "nbytes_cached",
            "c_contiguous_cached",
            "f_contiguous_cached",
            "static_add_call",
        ),
    ),
)
_OPERATION_CONTRACTS = {
    operation: (category, shape_kind, axis)
    for category, shape_kind, axis, operations in _OPERATION_GROUPS
    for operation in operations
}


def _expected_job_dtype(operation: str) -> str:
    contract = _OPERATION_CONTRACTS.get(operation)
    if operation in {
        "packbits",
        "packbits_axis_last",
        "packbits_axis0_strided",
        "unpackbits",
        "unpackbits_axis_last",
        "unpackbits_axis0_strided",
    }:
        return "u8"
    return (
        "i64"
        if contract is not None and contract[0] in {"bitwise", "integer"}
        else "f64"
    )


def load_catalog(path: str | Path) -> dict[str, object]:
    """Load and strictly validate a version-one benchmark case catalog."""
    with Path(path).open("r", encoding="utf-8") as source:
        catalog = json.load(source, object_pairs_hook=_object_with_unique_keys)

    return _validate_catalog(catalog)


def load_worker_json(
    path: str | Path,
    *,
    expected_runtime: str | None = None,
    expected_case_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Load worker JSON with duplicate-key and result-schema validation."""
    with Path(path).open("r", encoding="utf-8") as source:
        document = json.load(source, object_pairs_hook=_object_with_unique_keys)
    validate_worker_result(document, expected_runtime, expected_case_ids)
    return document


def _validate_catalog(catalog: object) -> dict[str, object]:
    if not isinstance(catalog, dict):
        raise ValueError("catalog must be a dict")
    _validate_exact_keys(catalog, _CATALOG_KEYS, "top-level")

    schema_version = catalog["schema_version"]
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("schema_version must be the integer 1")

    size_sets = catalog["size_sets"]
    if not isinstance(size_sets, dict):
        raise ValueError("size_sets must be a dict")
    _validate_exact_keys(size_sets, _SIZE_SET_NAMES, "size_sets")
    for name, sizes in size_sets.items():
        _validate_sizes(sizes, f"size_sets.{name}")

    templates = catalog["templates"]
    if not isinstance(templates, list) or not templates:
        raise ValueError("templates must be a non-empty list")

    operations: set[str] = set()
    for index, template in enumerate(templates):
        location = f"templates[{index}]"
        if not isinstance(template, dict):
            raise ValueError(f"{location} must be a dict")
        _validate_exact_keys(template, _TEMPLATE_KEYS, location)

        for field in ("category", "operation"):
            value = template[field]
            _validate_wire_string(value, f"{location}.{field}")

        operation = template["operation"]
        dtype = template["dtype"]
        _validate_wire_string(dtype, f"{location}.dtype")
        expected_dtype = _expected_job_dtype(operation)
        if dtype != expected_dtype:
            raise ValueError(f"{location}.dtype must be {expected_dtype!r}")

        shape_kind = template["shape_kind"]
        if not isinstance(shape_kind, str) or shape_kind not in _SHAPE_KINDS:
            raise ValueError(
                f"{location}.shape_kind must be one of {sorted(_SHAPE_KINDS)!r}"
            )

        axis = template["axis"]
        if type(axis) is not int:
            raise ValueError(f"{location}.axis must be an integer")

        if operation in operations:
            raise ValueError(f"{location}.operation is duplicate: {operation!r}")
        operations.add(operation)

        contract = _OPERATION_CONTRACTS.get(operation)
        if contract is None:
            raise ValueError(f"{location}.operation is unsupported: {operation!r}")
        expected_category, expected_shape_kind, expected_axis = contract
        if (template["category"], shape_kind, axis) != contract:
            raise ValueError(
                f"{location} operation {operation!r} must use category "
                f"{expected_category!r}, shape_kind {expected_shape_kind!r}, "
                f"and axis {expected_axis}"
            )

        profiles = template["profiles"]
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError(f"{location}.profiles must be a non-empty dict")
        _validate_profile_names(profiles, location)
        for profile, size_specification in profiles.items():
            profile_location = f"{location}.profiles.{profile}"
            _validate_profile_sizes(
                size_specification,
                shape_kind,
                size_sets,
                profile_location,
            )

    return catalog


def _object_with_unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate key in JSON object: {key!r}")
        value[key] = item
    return value


def expand_cases(catalog: dict[str, object], profile: str) -> list[dict[str, object]]:
    """Expand one catalog profile into deterministic benchmark job dictionaries."""
    if not isinstance(profile, str) or profile not in _PROFILE_NAMES:
        raise ValueError(f"unknown profile: {profile!r}")

    catalog = _validate_catalog(catalog)
    size_sets = catalog["size_sets"]
    templates = catalog["templates"]
    cases: list[dict[str, object]] = []
    case_ids: set[str] = set()

    for template in templates:
        profiles = template["profiles"]
        if profile not in profiles:
            continue

        size_specification = profiles[profile]
        sizes = size_sets[size_specification] if isinstance(size_specification, str) else size_specification
        for size in sizes:
            case = _expand_case(template, size)
            case_id = case["id"]
            if case_id in case_ids:
                raise ValueError(f"duplicate expanded case id: {case_id!r}")
            case_ids.add(case_id)
            cases.append(case)

    return sorted(cases, key=lambda case: case["id"])


def write_jobs_tsv(cases: Sequence[dict[str, object]], destination: TextIO) -> None:
    """Write validated expanded jobs as a tab-separated table."""
    if not cases:
        raise ValueError("cases must not be empty")

    case_ids: set[object] = set()
    for index, case in enumerate(cases):
        validate_job_case(case, index)
        case_id = case["id"]
        if case_id in case_ids:
            raise ValueError(f"duplicate case id: {case_id!r}")
        case_ids.add(case_id)

    writer = csv.DictWriter(
        destination,
        fieldnames=_CASE_FIELDS,
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(cases)


def format_case_id(
    operation: str,
    dtype: str,
    *,
    size: int,
    rows: int,
    cols: int,
    axis: int,
) -> str:
    """Format the canonical wire ID for one supported benchmark operation."""
    contract = _OPERATION_CONTRACTS.get(operation)
    if contract is None:
        raise ValueError(f"unknown operation: {operation!r}")
    expected_dtype = _expected_job_dtype(operation)
    if dtype != expected_dtype:
        raise ValueError(f"dtype must be {expected_dtype!r} for operation {operation!r}")

    _, shape_kind, expected_axis = contract
    if axis != expected_axis:
        raise ValueError(
            f"operation {operation!r} axis must be {expected_axis}, not {axis}"
        )
    if shape_kind == "bridge":
        return f"bridge/{operation}"
    if shape_kind in {"vector", "fft"}:
        return f"{operation}/{dtype}/{size}"

    case_id = f"{operation}/{dtype}/{rows}x{cols}"
    if expected_axis != -1:
        case_id += f"/axis{expected_axis}"
    return case_id


def validate_job_case(case: object, index: int = 0) -> None:
    """Validate one job's structural and operation-specific wire contract."""
    location = f"cases[{index}]"
    if not isinstance(case, dict):
        raise ValueError(f"{location} must be a dict")
    _validate_exact_keys(case, set(_CASE_FIELDS), location, extra_label="extra")

    for field in ("id", "category", "operation"):
        value = case[field]
        _validate_wire_string(value, f"{location}.{field}")

    operation = case["operation"]
    dtype = case["dtype"]
    _validate_wire_string(dtype, f"{location}.dtype")
    expected_dtype = _expected_job_dtype(operation)
    if dtype != expected_dtype:
        raise ValueError(
            f"{location}.dtype must be {expected_dtype!r} for operation {operation!r}"
        )

    size = case["size"]
    if type(size) is not int or size <= 0:
        raise ValueError(f"{location}.size must be a positive integer")

    for field in ("rows", "cols"):
        value = case[field]
        if type(value) is not int or value < 0:
            raise ValueError(f"{location}.{field} must be a non-negative integer")

    axis = case["axis"]
    if type(axis) is not int:
        raise ValueError(f"{location}.axis must be an integer")

    rows = case["rows"]
    cols = case["cols"]
    if (rows == 0) != (cols == 0):
        raise ValueError(
            f"{location}.rows and {location}.cols must both be zero or both be positive"
        )
    if rows > 0 and size != rows * cols:
        raise ValueError(f"{location}.size must equal rows * cols for a matrix job")

    _validate_job_semantics(case, location)


def _validate_job_semantics(case: dict[str, object], location: str) -> None:
    case_id = case["id"]
    operation = case["operation"]
    contract = _OPERATION_CONTRACTS.get(operation)
    if contract is None:
        _job_semantic_error(location, case_id, "operation", "is unknown")

    expected_category, shape_kind, expected_axis = contract
    if case["category"] != expected_category:
        _job_semantic_error(
            location,
            case_id,
            "category",
            f"must be {expected_category!r} for operation {operation!r}",
        )
    if case["axis"] != expected_axis:
        _job_semantic_error(
            location,
            case_id,
            "axis",
            f"must be {expected_axis}",
        )

    if shape_kind == "bridge":
        if case["size"] != 1:
            _job_semantic_error(location, case_id, "size", "must be 1")
        _validate_zero_job_dimensions(case, location)
    elif shape_kind in {"vector", "fft"}:
        _validate_zero_job_dimensions(case, location)
    else:
        rows = case["rows"]
        cols = case["cols"]
        if rows <= 0:
            _job_semantic_error(location, case_id, "rows", "must be positive")
        if cols <= 0:
            _job_semantic_error(location, case_id, "cols", "must be positive")
        if rows != cols:
            _job_semantic_error(location, case_id, "cols", "must equal rows")

    expected_id = format_case_id(
        operation,
        case["dtype"],
        size=case["size"],
        rows=case["rows"],
        cols=case["cols"],
        axis=case["axis"],
    )
    if case_id != expected_id:
        _job_semantic_error(
            location,
            case_id,
            "id",
            f"must be {expected_id!r}",
        )


def _validate_zero_job_dimensions(case: dict[str, object], location: str) -> None:
    case_id = case["id"]
    if case["rows"] != 0:
        _job_semantic_error(location, case_id, "rows", "must be 0")
    if case["cols"] != 0:
        _job_semantic_error(location, case_id, "cols", "must be 0")


def _job_semantic_error(
    location: str,
    case_id: object,
    field: str,
    detail: str,
) -> None:
    raise ValueError(f"{location} case {case_id!r} field {field!r} {detail}")


def _expand_case(template: dict[str, object], dimension: int) -> dict[str, object]:
    operation = template["operation"]
    dtype = template["dtype"]
    shape_kind = template["shape_kind"]

    if shape_kind == "bridge":
        size = 1
        rows = 0
        cols = 0
        axis = -1
    elif shape_kind in {"vector", "fft"}:
        size = dimension
        rows = 0
        cols = 0
        axis = template["axis"]
    else:
        size = dimension * dimension
        rows = dimension
        cols = dimension
        axis = template["axis"]

    case_id = format_case_id(
        operation,
        dtype,
        size=size,
        rows=rows,
        cols=cols,
        axis=axis,
    )

    return {
        "id": case_id,
        "category": template["category"],
        "operation": operation,
        "dtype": dtype,
        "size": size,
        "rows": rows,
        "cols": cols,
        "axis": axis,
    }


def _validate_exact_keys(
    value: dict[object, object],
    expected: set[str],
    location: str,
    *,
    extra_label: str = "unknown",
) -> None:
    actual = set(value)
    missing = expected - actual
    if missing:
        raise ValueError(f"{location} has missing keys: {', '.join(sorted(missing))}")
    extra = actual - expected
    if extra:
        raise ValueError(
            f"{location} has {extra_label} keys: {', '.join(sorted(map(str, extra)))}"
        )


def _validate_wire_string(value: object, location: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} must be a non-empty string")
    for forbidden_character in _FORBIDDEN_WIRE_CHARACTERS:
        if forbidden_character in value:
            raise ValueError(
                f"{location} contains forbidden wire character "
                f"{forbidden_character!r}"
            )


def _validate_sizes(value: object, location: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} must be a non-empty list of positive integers")

    seen: set[int] = set()
    for index, size in enumerate(value):
        if type(size) is not int or size <= 0:
            raise ValueError(f"{location}[{index}] must be a positive integer")
        if size in seen:
            raise ValueError(f"{location}[{index}] is a duplicate size: {size}")
        seen.add(size)


def _validate_profile_sizes(
    value: object,
    shape_kind: str,
    size_sets: dict[object, object],
    location: str,
) -> None:
    if shape_kind == "bridge":
        _validate_sizes(value, location)
        if value != [1]:
            raise ValueError(f"{location} must be the explicit size list [1] for bridge")
        return

    if isinstance(value, str):
        if value not in size_sets:
            raise ValueError(f"{location} refers to unknown size set {value!r}")
        if value != shape_kind:
            raise ValueError(
                f"{location} must reference the {shape_kind!r} size set, not {value!r}"
            )
        return

    if isinstance(value, list):
        _validate_sizes(value, location)
        allowed_sizes = set(size_sets[shape_kind])
        unexpected_sizes = set(value) - allowed_sizes
        if unexpected_sizes:
            raise ValueError(
                f"{location} must be a subset of size_sets.{shape_kind}; "
                f"unexpected sizes: {sorted(unexpected_sizes)!r}"
            )
        return

    raise ValueError(f"{location} must be a size-set name or a list of sizes")


def _validate_profile_names(profiles: dict[object, object], template_location: str) -> None:
    unknown = set(profiles) - _PROFILE_NAMES
    if unknown:
        raise ValueError(
            f"{template_location}.profiles has unknown keys: "
            f"{', '.join(sorted(map(str, unknown)))}"
        )


def percentile(samples: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated quantile from *samples*."""
    if not samples:
        raise ValueError("samples must not be empty")
    if (
        isinstance(quantile, bool)
        or not isinstance(quantile, Real)
        or not math.isfinite(quantile)
        or not 0 <= quantile <= 1
    ):
        raise ValueError("quantile must be a finite number between 0 and 1")

    numeric_samples: list[float] = []
    for index, sample in enumerate(samples):
        if isinstance(sample, bool) or not isinstance(sample, Real):
            raise ValueError(f"samples[{index}] must be a finite real number")
        numeric_sample = float(sample)
        if not math.isfinite(numeric_sample):
            raise ValueError(f"samples[{index}] must be a finite real number")
        numeric_samples.append(numeric_sample)

    ordered = sorted(numeric_samples)
    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    fraction = position - lower_index
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    return lower_value + (upper_value - lower_value) * fraction


def summarize_samples(samples: Sequence[float]) -> dict[str, int | float]:
    """Return deterministic descriptive statistics for positive timings."""
    values = _positive_finite_values(samples, "samples")
    mean = statistics.fmean(values)
    median = statistics.median(values)
    stdev = statistics.pstdev(values)
    deviations = [abs(value - median) for value in values]

    return {
        "sample_count": len(values),
        "min_ns": min(values),
        "max_ns": max(values),
        "mean_ns": mean,
        "median_ns": median,
        "stdev_ns": stdev,
        "cv": stdev / mean,
        "mad_ns": statistics.median(deviations),
        "p05_ns": percentile(values, 0.05),
        "p25_ns": percentile(values, 0.25),
        "p75_ns": percentile(values, 0.75),
        "p95_ns": percentile(values, 0.95),
    }


def geometric_mean(values: Sequence[float]) -> float:
    """Return the geometric mean of finite positive values."""
    positive_values = _positive_finite_values(values, "values")
    mean_log = statistics.fmean(math.log(value) for value in positive_values)
    return math.exp(mean_log)


def validate_worker_result(
    document: dict[str, object],
    expected_runtime: str | None = None,
    expected_case_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Validate a version-one worker result without altering it."""
    if not isinstance(document, dict):
        raise ValueError("document must be a dict")

    schema_version = document.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("schema_version must be the integer 1")

    runtime = document.get("runtime")
    if not isinstance(runtime, str) or not runtime:
        raise ValueError("runtime must be a non-empty string")

    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dict")
    _validate_worker_metadata(runtime, metadata)

    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list")

    case_ids: list[str] = []
    for index, case in enumerate(cases):
        location = f"cases[{index}]"
        if not isinstance(case, dict):
            raise ValueError(f"{location} must be a dict")

        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{location}.id must be a non-empty string")
        if case_id in case_ids:
            raise ValueError(f"{location}.id is duplicated: {case_id!r}")
        case_ids.append(case_id)

        job = {field: case.get(field) for field in _CASE_FIELDS}
        try:
            validate_job_case(job, index)
        except ValueError as error:
            raise ValueError(f"{location} has invalid canonical job metadata: {error}") from error

        inner_loops = case.get("inner_loops")
        if type(inner_loops) is not int or inner_loops <= 0:
            raise ValueError(f"{location}.inner_loops must be a positive integer")

        samples_ns = case.get("samples_ns")
        if not isinstance(samples_ns, list) or not samples_ns or len(samples_ns) % 2 == 0:
            raise ValueError(f"{location}.samples_ns must be a non-empty list with an odd length")
        for sample_index, sample in enumerate(samples_ns):
            if (
                isinstance(sample, bool)
                or not isinstance(sample, Real)
                or not math.isfinite(sample)
                or sample <= 0
            ):
                raise ValueError(
                    f"{location}.samples_ns[{sample_index}] must be a finite positive number"
                )
        if runtime in {"numpy", "cnumpy"} and len(samples_ns) != metadata["sample_count"]:
            raise ValueError(
                f"{location}.samples_ns length does not match metadata.sample_count"
            )

        validation = case.get("validation")
        if not isinstance(validation, dict):
            raise ValueError(f"{location}.validation must be a dict")
        _validate_validation_signature(validation, job, location)

        if runtime == "cnumpy":
            retained_bytes = case.get("retained_bytes")
            if type(retained_bytes) is not int or retained_bytes != 0:
                raise ValueError(f"{location}.retained_bytes must be the integer zero")

    if expected_runtime is not None and runtime != expected_runtime:
        raise ValueError(f"runtime {runtime!r} does not match expected runtime {expected_runtime!r}")
    if expected_case_ids is not None:
        expected_ids = list(expected_case_ids)
        for index, expected_id in enumerate(expected_ids):
            if not isinstance(expected_id, str) or not expected_id:
                raise ValueError(
                    f"expected_case_ids[{index}] must be a non-empty string"
                )
        if len(expected_ids) != len(set(expected_ids)):
            raise ValueError("expected_case_ids must not contain duplicate ids")
        if set(case_ids) != set(expected_ids):
            raise ValueError(
                f"case ids {case_ids!r} do not match expected case ids {expected_ids!r}"
            )

    return document


def _validate_worker_metadata(runtime: str, metadata: dict[object, object]) -> None:
    if runtime not in {"numpy", "cnumpy"}:
        return
    common_fields = {
        "warmups",
        "sample_count",
        "target_sample_ns",
        "seed",
        "timer",
    }
    runtime_fields = (
        {
            "python_version",
            "numpy_version",
            "timer_resolution_ns",
        }
        if runtime == "numpy"
        else {
            "ahk_version",
            "dll_version",
            "dll_path",
            "timer_frequency",
            "simd_level",
            "simd_name",
        }
    )
    missing = (common_fields | runtime_fields) - set(metadata)
    if missing:
        raise ValueError(
            f"metadata has missing {runtime} fields: {', '.join(sorted(missing))}"
        )

    for field in (
        ("python_version", "numpy_version")
        if runtime == "numpy"
        else ("ahk_version", "dll_version", "dll_path")
    ):
        value = metadata[field]
        if not isinstance(value, str) or not value:
            raise ValueError(f"metadata.{field} must be a non-empty string")

    expected_timer = "perf_counter_ns" if runtime == "numpy" else "QueryPerformanceCounter"
    if metadata["timer"] != expected_timer:
        raise ValueError(f"metadata.timer must be {expected_timer!r} for {runtime}")

    warmups = metadata["warmups"]
    if type(warmups) is not int or warmups < 0:
        raise ValueError("metadata.warmups must be a non-negative integer")
    sample_count = metadata["sample_count"]
    if type(sample_count) is not int or sample_count <= 0 or sample_count % 2 == 0:
        raise ValueError("metadata.sample_count must be a positive odd integer")
    seed = metadata["seed"]
    if type(seed) is not int or not 0 <= seed <= 2_147_483_647:
        raise ValueError("metadata.seed must be an integer in [0, 2147483647]")
    _require_positive_finite_real(metadata["target_sample_ns"], "metadata.target_sample_ns")
    if runtime == "numpy":
        _require_positive_finite_real(
            metadata["timer_resolution_ns"],
            "metadata.timer_resolution_ns",
        )
    else:
        frequency = metadata["timer_frequency"]
        if type(frequency) is not int or frequency <= 0:
            raise ValueError("metadata.timer_frequency must be a positive integer")
        simd_level = metadata["simd_level"]
        simd_name = metadata["simd_name"]
        if (
            type(simd_level) is not int
            or not isinstance(simd_name, str)
            or (simd_level, simd_name) not in {(1, "sse2"), (2, "avx2")}
        ):
            raise ValueError(
                "metadata simd pair must be (1, 'sse2') or (2, 'avx2')"
            )


def _validate_validation_signature(
    validation: dict[object, object],
    job: dict[str, object],
    case_location: str,
) -> None:
    location = f"{case_location}.validation"
    mode = validation.get("mode")
    if mode not in {"numeric", "numeric_nan", "shape"}:
        raise ValueError(
            f"{location}.mode must be 'numeric', 'numeric_nan', or 'shape'"
        )
    base_fields = {"mode", "shape", "size", "logical_dtype"}
    if mode == "shape":
        expected_fields = base_fields
    elif mode == "numeric_nan":
        expected_fields = base_fields | {
            "sample_indices",
            "values",
            "sum",
            "nan_count",
            "first_nan_index",
            "trailing_nan",
            "finite_members_exact",
        }
    else:
        expected_fields = base_fields | {"sample_indices", "values", "sum"}
    _validate_exact_keys(validation, expected_fields, location)

    shape = validation["shape"]
    if not isinstance(shape, list):
        raise ValueError(f"{location}.shape must be a list")
    for index, dimension in enumerate(shape):
        if type(dimension) is not int or dimension <= 0:
            raise ValueError(f"{location}.shape[{index}] must be a positive integer")
    size = validation["size"]
    if type(size) is not int or size <= 0:
        raise ValueError(f"{location}.size must be a positive integer")
    expected_size = math.prod(shape) if shape else 1
    if size != expected_size:
        raise ValueError(
            f"{location}.size {size} does not match shape product {expected_size}"
        )
    logical_dtype = validation["logical_dtype"]
    if logical_dtype not in {"f64", "i64", "u8", "bool"}:
        raise ValueError(
            f"{location}.logical_dtype must be 'f64', 'i64', 'u8', or 'bool'"
        )
    expected_mode, expected_shape, expected_dtype = _expected_validation_contract(job)
    if (mode, shape, logical_dtype) != (
        expected_mode,
        expected_shape,
        expected_dtype,
    ):
        raise ValueError(
            f"{location} does not match operation {job['operation']!r} contract: "
            f"expected mode={expected_mode!r}, shape={expected_shape!r}, "
            f"logical_dtype={expected_dtype!r}"
        )
    if mode == "shape":
        return

    sample_indices = validation["sample_indices"]
    expected_indices = list(dict.fromkeys((0, size // 2, size - 1)))
    if sample_indices != expected_indices:
        raise ValueError(
            f"{location}.sample_indices must be {expected_indices!r}"
        )
    values = validation["values"]
    if not isinstance(values, list) or len(values) != len(expected_indices):
        raise ValueError(
            f"{location}.values must match the sample_indices length"
        )
    for index, value in enumerate(values):
        _validate_signature_number(
            value,
            logical_dtype,
            f"{location}.values[{index}]",
        )
        if logical_dtype == "bool" and value not in {0, 1}:
            raise ValueError(
                f"{location}.values[{index}] must be zero or one for logical dtype bool"
            )
        if logical_dtype == "u8" and not 0 <= value <= 255:
            raise ValueError(
                f"{location}.values[{index}] must be within uint8 bounds"
            )
    _validate_signature_number(validation["sum"], logical_dtype, f"{location}.sum")
    if logical_dtype == "bool" and not float(validation["sum"]).is_integer():
        raise ValueError(f"{location}.sum must be integer-valued for logical dtype bool")
    if mode == "numeric_nan":
        nan_count = validation["nan_count"]
        first_nan_index = validation["first_nan_index"]
        trailing_nan = validation["trailing_nan"]
        finite_members_exact = validation["finite_members_exact"]
        if type(nan_count) is not int or not 0 < nan_count <= size:
            raise ValueError(f"{location}.nan_count must be between one and size")
        if type(first_nan_index) is not int or not 0 <= first_nan_index < size:
            raise ValueError(
                f"{location}.first_nan_index must be a valid flat index"
            )
        if trailing_nan is not True:
            raise ValueError(f"{location}.trailing_nan must be true")
        if finite_members_exact is not True:
            raise ValueError(f"{location}.finite_members_exact must be true")
        operation = job["operation"]
        if operation == "sort_stable_nan":
            input_size = job["size"]
            assert type(input_size) is int
            expected_nan_count = (input_size + 1) // 2
        elif operation == "unique_nan":
            expected_nan_count = 1
        else:
            raise ValueError(
                f"{location}.numeric_nan mode is not valid for operation {operation!r}"
            )
        expected_first_nan_index = size - expected_nan_count
        if (nan_count, first_nan_index) != (
            expected_nan_count,
            expected_first_nan_index,
        ):
            raise ValueError(
                f"{location} raw NaN facts must be nan_count={expected_nan_count} "
                f"and first_nan_index={expected_first_nan_index}"
            )
    if _is_argsort_operation(job["operation"]):
        input_size = job["size"]
        assert type(input_size) is int
        for index, value in enumerate(values):
            if not 0 <= value < input_size:
                raise ValueError(
                    f"{location}.values[{index}] is outside argsort index bounds"
                )
        expected_sum = input_size * (input_size - 1) // 2
        if validation["sum"] != expected_sum:
            raise ValueError(
                f"{location}.sum must equal the complete argsort permutation sum "
                f"{expected_sum}"
            )


def _validate_signature_number(value: object, logical_dtype: object, location: str) -> None:
    _require_finite_real(value, location)
    if logical_dtype in {"i64", "u8"} and not float(value).is_integer():
        raise ValueError(
            f"{location} must be integer-valued for logical dtype {logical_dtype}"
        )


def _expected_validation_contract(
    job: dict[str, object],
) -> tuple[str, list[int], str]:
    operation = job["operation"]
    size = job["size"]
    rows = job["rows"]
    cols = job["cols"]
    assert isinstance(operation, str)
    assert type(size) is int
    assert type(rows) is int
    assert type(cols) is int

    mode = (
        "shape"
        if operation == "random"
        else "numeric_nan"
        if operation in {"sort_stable_nan", "unique_nan"}
        else "numeric"
    )
    logical_dtype = (
        "bool"
        if operation
        in {
            "equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "logical_not",
            "isnan",
            "isinf",
            "isfinite",
            "signbit",
            "iscomplexobj",
            "isrealobj",
            "isscalar",
            "c_contiguous_cached",
            "f_contiguous_cached",
            "in1d_duplicates",
            "isin_duplicates",
        }
        else "i64"
        if _expected_job_dtype(operation) == "i64"
        or _is_argsort_operation(operation)
        or operation
        in {
            "argmax",
            "property_call",
            "property_cached",
            "nbytes_cached",
            "static_add_call",
            "allclose",
            "searchsorted",
            "searchsorted_right",
            "digitize",
            "digitize_decreasing",
            "lexsort",
        }
        else "u8"
        if _expected_job_dtype(operation) == "u8"
        else "f64"
    )
    if operation in {
        "sum",
        "mean",
        "average",
        "std",
        "max",
        "min",
        "prod",
        "argmax",
        "det",
        "norm",
        "property_call",
        "property_cached",
        "nbytes_cached",
        "c_contiguous_cached",
        "f_contiguous_cached",
        "static_add_call",
        "allclose",
        "pipeline_separate",
        "pipeline_batch",
        "iscomplexobj",
        "isrealobj",
        "isscalar",
        "trapz",
    }:
        shape: list[int] = []
    elif operation in {
        "softmax_axis_last",
        "softmax_axis0_strided",
        "log_softmax_axis_last",
        "log_softmax_axis0_strided",
    }:
        shape = [rows, cols]
    elif operation == "trapz_axis_last":
        shape = [rows]
    elif operation == "trapz_axis0_strided":
        shape = [cols]
    elif operation == "packbits":
        shape = [(size + 7) // 8]
    elif operation == "packbits_axis_last":
        shape = [rows, (cols + 7) // 8]
    elif operation == "packbits_axis0_strided":
        shape = [(rows + 7) // 8, cols]
    elif operation == "unpackbits":
        shape = [size * 8]
    elif operation == "unpackbits_axis_last":
        shape = [rows, cols * 8]
    elif operation == "unpackbits_axis0_strided":
        shape = [rows * 8, cols]
    elif operation in {"matmul", "dot", "inv", "cholesky", "einsum"}:
        shape = [rows, cols]
    elif operation in {"solve", "eig", "svd", "lstsq"}:
        shape = [rows]
    elif operation == "reshape":
        shape = [size, 1]
    elif operation == "atleast_2d":
        shape = [1, size]
    elif operation == "atleast_3d":
        shape = [1, size, 1]
    elif operation == "transpose_copy":
        shape = [cols, rows]
    elif operation == "concatenate":
        shape = [rows * 2, cols]
    elif operation == "sum_axis_last":
        shape = [rows]
    elif operation == "cumsum_axis_last":
        shape = [rows, cols]
    elif operation == "fft":
        shape = [size, 2]
    elif operation == "divmod":
        shape = [2, size]
    elif operation in {"take", "compress"}:
        shape = [(size + 1) // 2]
    elif operation in {
        "take_axis0_block", "take_axis0_strided",
        "compress_axis0_block", "compress_axis0_strided",
    }:
        shape = [(rows + 1) // 2, cols]
    elif operation == "unique_duplicates":
        shape = [min(size, 256)]
    elif operation == "unique_nan":
        shape = [min(size // 2, 128) + 1]
    elif operation in {
        "intersect1d_duplicates",
        "union1d_duplicates",
        "setdiff1d_duplicates",
        "setxor1d_duplicates",
    }:
        unique_count = min(size, 256)
        intersection_count = (min(unique_count, 128) + 1) // 2
        if operation == "intersect1d_duplicates":
            shape = [intersection_count]
        elif operation == "union1d_duplicates":
            shape = [unique_count + 128 - intersection_count]
        elif operation == "setdiff1d_duplicates":
            shape = [unique_count - intersection_count]
        else:
            shape = [unique_count + 128 - 2 * intersection_count]
    else:
        shape = [size]
    return mode, shape, logical_dtype


def _require_finite_real(value: object, location: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{location} must be a finite real number")


def _require_positive_finite_real(value: object, location: str) -> None:
    _require_finite_real(value, location)
    if value <= 0:
        raise ValueError(f"{location} must be positive")


def _positive_finite_values(values: Sequence[float], field: str) -> list[float]:
    if not values:
        raise ValueError(f"{field} must not be empty")

    validated: list[float] = []
    for index, value in enumerate(values):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{field}[{index}] must be a finite positive number")
        validated.append(float(value))
    return validated


_CASE_METADATA_FIELDS = (
    "id",
    "category",
    "operation",
    "dtype",
    "size",
    "rows",
    "cols",
    "axis",
)
_COMPARISON_FIELDS = (
    "id",
    "category",
    "operation",
    "shape",
    "semantic_qualification",
    "numpy_median_ns",
    "cnumpy_median_ns",
    "cnumpy_over_numpy",
    "winner",
    "numpy_cv",
    "cnumpy_cv",
    "numpy_p95_ns",
    "cnumpy_p95_ns",
)
_BASELINE_FIELDS = (
    "baseline_semantic_qualification",
    "baseline_compatibility",
    "baseline_cnumpy_over_numpy",
    "ratio_improvement",
)
_VALIDATION_REL_TOLERANCE = 1e-9
_VALIDATION_ABS_TOLERANCE = 1e-10
SEMANTIC_QUALIFICATION_NUMPY_VERSION = "1.25.0"
SEMANTIC_QUALIFICATION_ID = (
    f"numpy-{SEMANTIC_QUALIFICATION_NUMPY_VERSION.rsplit('.', 1)[0]}"
    "/task6-reduction-v1"
)
SEMANTIC_QUALIFICATION_REFERENCE = (
    f"NumPy {SEMANTIC_QUALIFICATION_NUMPY_VERSION}"
)
SEMANTIC_QUALIFICATION_OWNERS = (
    "benchmark.tests.test_reduction_semantics.ReductionSemanticsTests."
    "test_every_legacy_reduction_export_covers_rank_zero_through_four_and_expressible_axes",
    "benchmark.tests.test_reduction_semantics.ReductionSemanticsTests."
    "test_every_legacy_reduction_export_reports_exact_axis_errors",
    "benchmark.tests.test_reduction_semantics.ReductionSemanticsTests."
    "test_legacy_reduction_axis_minus_one_characterizes_none_sentinel",
    "benchmark.tests.test_reduction_semantics.ReductionSemanticsTests."
    "test_every_reduction_family_covers_rank_zero_through_four_and_all_axes",
    "benchmark.tests.test_reduction_semantics.ReductionSemanticsTests."
    "test_average_distinguishes_none_from_the_negative_last_axis",
    "benchmark.tests.test_reduction_semantics.ReductionSemanticsTests."
    "test_average_v2_covers_rank_zero_through_four_axes_weights_errors_and_lifetimes",
    "benchmark.tests.test_reduction_semantics.ReductionSemanticsTests."
    "test_legacy_nan_scalar_exports_cover_rank_zero_through_four_axes_values_errors_and_lifetimes",
    "benchmark.tests.test_reduction_semantics.ReductionSemanticsTests."
    "test_scalar_convenience_exports_cover_rank_zero_through_four_layouts_values_errors_and_lifetimes",
)
TASK7_SEMANTIC_QUALIFICATION_ID = (
    f"numpy-{SEMANTIC_QUALIFICATION_NUMPY_VERSION.rsplit('.', 1)[0]}"
    "/task7-sort-set-v1"
)
TASK7_SEMANTIC_QUALIFICATION_SORTING_OPERATIONS = (
        "argsort",
        "argsort_heapsort",
        "argsort_mergesort",
        "argsort_stable",
        "argsort_stable_nan",
        "sort",
        "sort_heapsort",
        "sort_mergesort",
        "sort_stable",
        "sort_stable_nan",
)
TASK7_SEMANTIC_QUALIFICATION_SET_OPERATIONS = (
        "in1d_duplicates",
        "intersect1d_duplicates",
        "isin_duplicates",
        "setdiff1d_duplicates",
        "setxor1d_duplicates",
        "union1d_duplicates",
        "unique_duplicates",
        "unique_nan",
)
TASK7_SEMANTIC_QUALIFICATION_OPERATIONS = frozenset(
    {
        "argsort",
        *TASK7_SEMANTIC_QUALIFICATION_SORTING_OPERATIONS,
        *TASK7_SEMANTIC_QUALIFICATION_SET_OPERATIONS,
    }
)
TASK7_SEMANTIC_QUALIFICATION_OWNERS = (
    "benchmark.tests.test_sort_set_semantics.SortSetSemanticsTests."
    "test_sort_v2_rank_zero_through_four_all_axes_and_numeric_dtypes",
    "benchmark.tests.test_sort_set_semantics.SortSetSemanticsTests."
    "test_large_stable_float64_radix_path_groups_nan_and_equal_zeros",
    "benchmark.tests.test_sort_set_semantics.SortSetSemanticsTests."
    "test_legacy_axis_none_and_v2_last_axis_are_unambiguous",
    "benchmark.tests.test_sort_set_semantics.SortSetSemanticsTests."
    "test_sort_exports_report_original_invalid_axis_exactly",
    "benchmark.tests.test_sort_set_semantics.SortSetSemanticsTests."
    "test_unique_v2_covers_all_represented_integer_and_float_dtypes",
    "benchmark.tests.test_sort_set_semantics.SortSetSemanticsTests."
    "test_unique_v2_float_edges_empty_and_capacity_errors",
    "benchmark.tests.test_sort_set_semantics.SortSetSemanticsTests."
    "test_legacy_unique_returns_values_only_for_every_optional_flag",
    "benchmark.tests.test_sort_set_semantics.SortSetSemanticsTests."
    "test_set_value_operations_cover_rank_zero_through_four_numeric_dtypes",
    "benchmark.tests.test_sort_set_semantics.SortSetSemanticsTests."
    "test_membership_operations_cover_rank_zero_through_four_numeric_dtypes",
    "benchmark.tests.test_sort_set_semantics.SortSetSemanticsTests."
    "test_set_operations_all_ordered_represented_dtype_pairs",
    "benchmark.tests.test_sort_set_semantics.SortSetSemanticsTests."
    "test_set_operation_errors_are_exact_and_atomic",
    "benchmark.tests.test_sort_set_semantics.SortSetSemanticsTests."
    "test_set_operations_preserve_int64_and_membership_shape",
    "benchmark.tests.test_sort_set_semantics.SortSetSemanticsTests."
    "test_set_operations_match_float_nan_zero_empty_and_mixed_dtype",
    "benchmark.tests.test_sort_set_semantics.SortSetSemanticsTests."
    "test_sort_unique_and_set_results_outlive_released_sources",
    "ahk.numpy.test.NumpyFoundationTest.TestSortUniqueAndSetFacadeV2",
)
TASK8_SEMANTIC_QUALIFICATION_ID = (
    f"numpy-{SEMANTIC_QUALIFICATION_NUMPY_VERSION.rsplit('.', 1)[0]}"
    "/task8-misc-axis-v1"
)
TASK8_SEMANTIC_QUALIFICATION_SCIPY_VERSION = "1.12.0"
TASK8_SEMANTIC_QUALIFICATION_REFERENCE = (
    f"NumPy {SEMANTIC_QUALIFICATION_NUMPY_VERSION}; "
    f"scipy.special {TASK8_SEMANTIC_QUALIFICATION_SCIPY_VERSION}"
)
TASK8_SEMANTIC_QUALIFICATION_OPERATIONS = (
    "softmax",
    "softmax_axis_last",
    "softmax_axis0_strided",
    "log_softmax",
    "log_softmax_axis_last",
    "log_softmax_axis0_strided",
    "trapz",
    "trapz_axis_last",
    "trapz_axis0_strided",
    "packbits",
    "packbits_axis_last",
    "packbits_axis0_strided",
    "unpackbits",
    "unpackbits_axis_last",
    "unpackbits_axis0_strided",
)
TASK8_SEMANTIC_QUALIFICATION_OWNERS = (
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_softmax_and_log_softmax_match_stable_axis_reference",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_softmax_rank_zero_through_four_all_axes_and_real_dtypes",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_softmax_nan_and_infinity_behavior_matches_reference",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_softmax_huge_empty_reduction_axis_errors_before_data_access",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_trapz_matches_numpy_for_axes_dx_and_noncontiguous_inputs",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_trapz_matches_numpy_dtype_promotion_and_typed_arithmetic",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_trapz_matches_numpy_x_dtype_promotion_and_product_overflow",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_trapz_preserves_wide_integer_panels_until_float_conversion",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_trapz_rank_zero_through_four_all_axes",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_packbits_matches_numpy_for_axes_bitorders_and_views",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_unpackbits_matches_numpy_counts_padding_axes_and_bitorders",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_bit_packing_rank_zero_through_four_all_axes_and_dtypes",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_bit_packing_empty_dimensions_match_numpy",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_bit_packing_huge_zero_shapes_are_bounded_and_exact",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_unpackbits_rejects_huge_multidimensional_result_before_allocation",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_legacy_bit_packing_rank_zero_through_four_and_errors",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_task8_core_exports_report_exact_invalid_axes_all_ranks",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_task8_core_exports_report_exact_dtype_and_null_errors",
    "benchmark.tests.test_misc_axis_semantics.MiscAxisSemanticsTests."
    "test_task8_results_survive_source_release_and_retain_zero_bytes",
    "benchmark.tests.test_orchestrator.ComparisonTests."
    "test_task8_qualification_requires_exact_scipy_reference_version",
    "ahk.numpy.test.NumpyFoundationTest.TestMiscAxisFacadeV2",
)
TASK9_SEMANTIC_QUALIFICATION_ID = (
    f"numpy-{SEMANTIC_QUALIFICATION_NUMPY_VERSION.rsplit('.', 1)[0]}"
    "/task9-linalg-v1"
)
TASK9_SEMANTIC_QUALIFICATION_OPERATIONS = (
    "einsum",
    "eig",
    "svd",
    "solve",
    "lstsq",
)
TASK9_SEMANTIC_QUALIFICATION_OWNERS = (
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_einsum_explicit_and_implicit_outputs_match_numpy",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_einsum_repeated_labels_diagonals_and_reductions",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_einsum_ellipsis_scalar_and_broadcasting_match_numpy",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_einsum_preserves_views_and_numpy_dtype_promotion",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_einsum_float16_forms_promotion_rounding_and_lifetime",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_einsum_fast_patterns_respect_named_label_broadcasting",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_einsum_invalid_subscripts_shapes_and_nulls_are_explicit",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_general_eig_returns_complex_eigenpairs_for_real_matrix",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_general_eig_preserves_real_dtype_for_real_spectrum",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_general_eig_supports_batched_matrices_and_owned_results",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_general_eig_dtype_promotion_matches_numpy",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_general_eig_dense_nonsymmetric_noncontiguous_view",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_general_eig_seeded_dense_differential",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_general_eig_repeated_and_defective_spectra",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_general_eig_validation_is_explicit_and_clears_outputs",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_ahk_eig_bridge_clears_every_provided_result_slot",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_eigvals_wrapper_inherits_general_semantics_and_owns_errors",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_general_eig_zero_sized_matrix_and_batch_shapes",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_general_eig_tiny_complex_pairs_are_scale_relative",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_general_eig_tiny_nonnormal_eigenvectors_are_scale_relative",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_general_eig_exact_zero_degenerate_eigenspaces",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_svd_workspace_products_are_checked_before_allocation",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_svd_legacy_default_returns_complete_rectangular_factors",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_svd_v2_reduced_tall_wide_and_batched_shapes",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_svd_v2_dtype_promotion_complex_and_unitarity",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_svd_v2_reads_noncontiguous_complex_view",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_svd_v2_compute_uv_false_and_zero_sized_shapes",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_svd_v2_complete_wide_and_hermitian_factors",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_svd_v2_seeded_dense_and_rank_deficient_differential",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_svd_v2_validation_is_explicit_and_atomic",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_solve_square_batched_rhs_dtypes_and_lifetimes_match_numpy_125",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_solve_zero_batch_broadcasts_without_reading_empty_sources",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_solve_singular_failure_is_explicit_atomic_and_nonretaining",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_lstsq_v2_rectangular_outputs_rcond_and_lifetimes_match_numpy_125",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_lstsq_v2_numpy_125_rcond_boundary_values",
    "benchmark.tests.test_linalg_semantics.LinalgSemanticsTests."
    "test_lstsq_v2_and_cond_v2_validation_is_explicit_atomic_and_retained0",
    "benchmark.tests.test_numpy_worker.PreparedOperationTests."
    "test_task9_linalg_workers_validate_complete_deterministic_results",
    "benchmark.tests.test_numpy_worker.PreparedOperationTests."
    "test_task9_linalg_full_validators_reject_unsampled_counterfeits",
    "benchmark.tests.test_orchestrator.ComparisonTests."
    "test_every_exact_task9_operation_receives_only_task9_qualification",
    "benchmark.tests.test_orchestrator.ComparisonTests."
    "test_task9_qualification_requires_exact_numpy_reference_version",
    "ahk.numpy.test.NumpyFoundationTest.TestEinsumFacadeV2",
    "ahk.numpy.test.NumpyFoundationTest.TestGeneralEigFacadeV2",
    "ahk.numpy.test.NumpyFoundationTest.TestLinalgSpectralDelegatesV2",
    "ahk.numpy.test.NumpyFoundationTest.TestSvdFacadeV2",
    "ahk.numpy.test.NumpyFoundationTest.TestTask9SolveLstsqAndCondFacadeV2",
)
TASK10_SEMANTIC_QUALIFICATION_ID = (
    f"numpy-{SEMANTIC_QUALIFICATION_NUMPY_VERSION.rsplit('.', 1)[0]}"
    "/task10-random-choice-v1"
)
TASK10_SEMANTIC_QUALIFICATION_OWNERS = (
    "benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests."
    "test_seeded_unweighted_choice_preserves_shape_dtype_and_sequence",
    "benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests."
    "test_numpy_125_differential_shape_dtype_replace_and_probability_dtypes",
    "benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests."
    "test_numpy_125_differential_noncontiguous_and_negative_stride_inputs",
    "benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests."
    "test_numpy_125_differential_rejections_are_atomic",
    "benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests."
    "test_choice_v2_raw_size_contract_rejections_are_atomic",
    "benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests."
    "test_probability_sum_tolerance_matches_numpy_125_by_dtype",
    "benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests."
    "test_weighted_replacement_matches_requested_distribution",
    "benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests."
    "test_weighted_without_replacement_non_degenerate_distribution_matches_numpy_125",
    "benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests."
    "test_scalar_and_empty_shapes_follow_numpy_size_semantics",
    "benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests."
    "test_validation_failures_are_explicit_and_allocation_atomic",
    "benchmark.tests.test_random_semantics.RandomChoiceSemanticsTests."
    "test_legacy_choice_uses_weighted_semantics",
    "benchmark.tests.test_numpy_worker.PreparedOperationTests."
    "test_weighted_choice_uses_full_deterministic_validation",
    "benchmark.tests.test_orchestrator.ComparisonTests."
    "test_task10_random_choice_has_exact_scope_and_version_gate",
    "benchmark.tests.test_catalog.SharedCatalogTests."
    "test_weighted_choice_is_a_canonical_one_million_sample_case",
    "ahk.numpy.test.NumpyFoundationTest."
    "TestWeightedRandomChoiceFacadeV2",
    "ahk.numpy.test.NumpyFoundationTest."
    "TestRandomChoiceSeedScalarAndErrorFacadeV2",
    "ahk.numpy.test.NumpyFoundationTest."
    "TestRandomChoiceProbabilityErrorsAndTemporaryLifetimeV2",
)
SEMANTIC_QUALIFICATIONS = (
    {
        "id": SEMANTIC_QUALIFICATION_ID,
        "scope": {
            "cases": [{"category": "reduction", "operations": "*"}]
        },
        "reference": SEMANTIC_QUALIFICATION_REFERENCE,
        "owners": list(SEMANTIC_QUALIFICATION_OWNERS),
        "manifest": "compat/manifest.json",
    },
    {
        "id": TASK7_SEMANTIC_QUALIFICATION_ID,
        "scope": {
            "cases": [
                {
                    "category": "sorting",
                    "operations": list(
                        TASK7_SEMANTIC_QUALIFICATION_SORTING_OPERATIONS
                    ),
                },
                {
                    "category": "set",
                    "operations": list(
                        TASK7_SEMANTIC_QUALIFICATION_SET_OPERATIONS
                    ),
                },
            ]
        },
        "reference": SEMANTIC_QUALIFICATION_REFERENCE,
        "owners": list(TASK7_SEMANTIC_QUALIFICATION_OWNERS),
        "manifest": "compat/manifest.json",
    },
    {
        "id": TASK8_SEMANTIC_QUALIFICATION_ID,
        "scope": {
            "cases": [
                {
                    "category": "misc_axis",
                    "operations": list(
                        TASK8_SEMANTIC_QUALIFICATION_OPERATIONS
                    ),
                }
            ]
        },
        "reference": TASK8_SEMANTIC_QUALIFICATION_REFERENCE,
        "owners": list(TASK8_SEMANTIC_QUALIFICATION_OWNERS),
        "manifest": "compat/manifest.json",
    },
    {
        "id": TASK9_SEMANTIC_QUALIFICATION_ID,
        "scope": {
            "cases": [
                {
                    "category": "linalg",
                    "operations": list(
                        TASK9_SEMANTIC_QUALIFICATION_OPERATIONS
                    ),
                }
            ]
        },
        "reference": SEMANTIC_QUALIFICATION_REFERENCE,
        "owners": list(TASK9_SEMANTIC_QUALIFICATION_OWNERS),
        "manifest": "compat/manifest.json",
    },
    {
        "id": TASK10_SEMANTIC_QUALIFICATION_ID,
        "scope": {
            "cases": [
                {
                    "category": "random",
                    "operations": ["choice_weighted"],
                }
            ]
        },
        "reference": SEMANTIC_QUALIFICATION_REFERENCE,
        "owners": list(TASK10_SEMANTIC_QUALIFICATION_OWNERS),
        "manifest": "compat/manifest.json",
    },
)


def require_semantic_qualification_numpy_version(metadata: object) -> None:
    """Require the exact NumPy worker version covered by the qualification."""
    actual = metadata.get("numpy_version") if isinstance(metadata, dict) else None
    if not isinstance(actual, str) or actual != SEMANTIC_QUALIFICATION_NUMPY_VERSION:
        raise ValueError(
            "semantic qualification NumPy version mismatch: "
            f"expected={SEMANTIC_QUALIFICATION_NUMPY_VERSION!r}, actual={actual!r}"
        )


def require_task8_qualification_scipy_version(metadata: object) -> None:
    """Require the exact SciPy version used by Task 8 softmax references."""
    actual = metadata.get("scipy_version") if isinstance(metadata, dict) else None
    if (
        not isinstance(actual, str)
        or actual != TASK8_SEMANTIC_QUALIFICATION_SCIPY_VERSION
    ):
        raise ValueError(
            "Task 8 semantic qualification SciPy version mismatch: "
            f"expected={TASK8_SEMANTIC_QUALIFICATION_SCIPY_VERSION!r}, "
            f"actual={actual!r}"
        )


def _semantic_qualification_for_case(case: dict[str, object]) -> str:
    category = case.get("category")
    operation = case.get("operation")
    for qualification in SEMANTIC_QUALIFICATIONS:
        scope = qualification["scope"]
        assert isinstance(scope, dict)
        declarations = scope["cases"]
        assert isinstance(declarations, list)
        for declaration in declarations:
            declared_operations = declaration["operations"]
            if category != declaration["category"]:
                continue
            if declared_operations == "*" or operation in declared_operations:
                return str(qualification["id"])
    return "N/A"


def semantic_qualification_declarations() -> list[dict[str, object]]:
    """Return the single registry used by environment and row provenance."""
    return copy.deepcopy(list(SEMANTIC_QUALIFICATIONS))


def _numpy_worker_has_qualified_case(document: object) -> bool:
    if not isinstance(document, dict) or document.get("runtime") != "numpy":
        return False
    cases = document.get("cases")
    return isinstance(cases, list) and any(
        isinstance(case, dict) and _semantic_qualification_for_case(case) != "N/A"
        for case in cases
    )


def _numpy_worker_has_task8_case(document: object) -> bool:
    if not isinstance(document, dict) or document.get("runtime") != "numpy":
        return False
    cases = document.get("cases")
    return isinstance(cases, list) and any(
        isinstance(case, dict)
        and _semantic_qualification_for_case(case)
        == TASK8_SEMANTIC_QUALIFICATION_ID
        for case in cases
    )


def normalize_worker_result(
    document: dict[str, object],
    *,
    expected_runtime: str | None = None,
    expected_case_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Copy a raw worker result and attach shared sample summaries."""
    validate_worker_result(document, expected_runtime, expected_case_ids)
    normalized = copy.deepcopy(document)
    normalized_cases = normalized["cases"]
    assert isinstance(normalized_cases, list)
    for case in normalized_cases:
        assert isinstance(case, dict)
        samples = case["samples_ns"]
        assert isinstance(samples, list)
        summary = summarize_samples(samples)
        summary["total_timed_operations"] = case["inner_loops"] * len(samples)
        case["summary"] = summary
    return normalized


def compare_results(
    numpy_document: dict[str, object],
    cnumpy_document: dict[str, object],
) -> list[dict[str, object]]:
    """Validate two worker documents and calculate per-case comparisons."""
    if _numpy_worker_has_qualified_case(numpy_document):
        require_semantic_qualification_numpy_version(
            numpy_document.get("metadata")
        )
    if _numpy_worker_has_task8_case(numpy_document):
        require_task8_qualification_scipy_version(
            numpy_document.get("metadata")
        )
    validate_worker_result(numpy_document, expected_runtime="numpy")
    validate_worker_result(cnumpy_document, expected_runtime="cnumpy")
    numpy_cases = _cases_by_id(numpy_document)
    cnumpy_cases = _cases_by_id(cnumpy_document)
    if set(numpy_cases) != set(cnumpy_cases):
        raise ValueError(
            "worker case set mismatch: "
            f"NumPy={sorted(numpy_cases)!r}, cnumpy={sorted(cnumpy_cases)!r}"
        )

    rows: list[dict[str, object]] = []
    for case_id in sorted(numpy_cases):
        numpy_case = numpy_cases[case_id]
        cnumpy_case = cnumpy_cases[case_id]
        numpy_metadata = tuple(numpy_case.get(field) for field in _CASE_METADATA_FIELDS)
        cnumpy_metadata = tuple(cnumpy_case.get(field) for field in _CASE_METADATA_FIELDS)
        if numpy_metadata != cnumpy_metadata:
            raise ValueError(f"case metadata mismatch for {case_id!r}")
        _require_zero_retained_bytes(cnumpy_case, case_id)
        _compare_validation_signatures(
            numpy_case["validation"],
            cnumpy_case["validation"],
            numpy_case,
            case_id,
        )

        numpy_summary = summarize_samples(numpy_case["samples_ns"])
        cnumpy_summary = summarize_samples(cnumpy_case["samples_ns"])
        numpy_median = float(numpy_summary["median_ns"])
        cnumpy_median = float(cnumpy_summary["median_ns"])
        ratio = cnumpy_median / numpy_median
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError(f"invalid timing ratio for {case_id!r}")
        rows.append(
            {
                "id": case_id,
                "category": numpy_case["category"],
                "operation": numpy_case["operation"],
                "shape": _case_shape(numpy_case),
                "semantic_qualification": _semantic_qualification_for_case(
                    numpy_case
                ),
                "numpy_median_ns": numpy_median,
                "cnumpy_median_ns": cnumpy_median,
                "cnumpy_over_numpy": ratio,
                "winner": "cnumpy" if cnumpy_median < numpy_median else "NumPy",
                "numpy_cv": float(numpy_summary["cv"]),
                "cnumpy_cv": float(cnumpy_summary["cv"]),
                "numpy_p95_ns": float(numpy_summary["p95_ns"]),
                "cnumpy_p95_ns": float(cnumpy_summary["p95_ns"]),
            }
        )
    return rows


def load_baseline_csv(source: str | Path | TextIO) -> dict[str, dict[str, str]]:
    """Load and validate the ID-to-row mapping from a comparison CSV."""
    close_source = False
    if isinstance(source, (str, Path)):
        stream = Path(source).open("r", encoding="utf-8", newline="")
        close_source = True
    else:
        stream = source
    try:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"id", "cnumpy_over_numpy"} <= set(reader.fieldnames):
            raise ValueError("baseline CSV must contain id and cnumpy_over_numpy columns")
        rows: dict[str, dict[str, str]] = {}
        for index, row in enumerate(reader, start=2):
            case_id = row.get("id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(f"baseline row {index} id must be non-empty")
            if case_id in rows:
                raise ValueError(f"baseline contains duplicate id {case_id!r}")
            raw_ratio = row.get("cnumpy_over_numpy")
            try:
                ratio = float(raw_ratio) if raw_ratio is not None else math.nan
            except ValueError as error:
                raise ValueError(
                    f"baseline ratio for {case_id!r} must be finite and positive"
                ) from error
            if not math.isfinite(ratio) or ratio <= 0:
                raise ValueError(
                    f"baseline ratio for {case_id!r} must be finite and positive"
                )
            rows[case_id] = dict(row)
        if not rows:
            raise ValueError("baseline CSV must contain at least one data row")
        return rows
    finally:
        if close_source:
            stream.close()


def attach_baseline(
    rows: Sequence[dict[str, object]],
    baseline: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    """Attach baseline ratios without altering the current measurements."""
    current_ids = [row.get("id") for row in rows]
    if len(current_ids) != len(set(current_ids)):
        raise ValueError("current comparison contains duplicate ids")
    if set(current_ids) != set(baseline):
        raise ValueError(
            "baseline case set mismatch: "
            f"current={sorted(map(str, current_ids))!r}, baseline={sorted(baseline)!r}"
        )
    attached: list[dict[str, object]] = []
    for row in rows:
        case_id = row["id"]
        assert isinstance(case_id, str)
        result = dict(row)
        current_qualification = row.get("semantic_qualification")
        baseline_qualification = baseline[case_id].get(
            "semantic_qualification"
        )
        if not isinstance(current_qualification, str) or not current_qualification:
            compatibility = (
                "N/A: current semantic qualification is missing"
            )
        elif current_qualification == "N/A":
            compatibility = (
                "N/A: current case is outside the semantic qualification scope"
            )
        elif not baseline_qualification:
            compatibility = (
                "N/A: baseline semantic qualification is missing"
            )
        elif baseline_qualification != current_qualification:
            compatibility = (
                "N/A: semantic qualification mismatch "
                f"(current={current_qualification!r}, "
                f"baseline={baseline_qualification!r})"
            )
        else:
            compatibility = "compatible"

        result["baseline_semantic_qualification"] = (
            baseline_qualification or "N/A"
        )
        result["baseline_compatibility"] = compatibility
        if compatibility == "compatible":
            baseline_ratio = float(
                baseline[case_id]["cnumpy_over_numpy"]
            )
            current_ratio = _finite_positive_number(
                row.get("cnumpy_over_numpy"),
                f"current ratio for {case_id!r}",
            )
            result["baseline_cnumpy_over_numpy"] = baseline_ratio
            result["ratio_improvement"] = baseline_ratio / current_ratio
        else:
            result["baseline_cnumpy_over_numpy"] = "N/A"
            result["ratio_improvement"] = "N/A"
        attached.append(result)
    return attached


def render_comparison_csv(rows: Sequence[dict[str, object]]) -> str:
    """Render stable RFC-compatible comparison CSV text."""
    if not rows:
        raise ValueError("comparison rows must not be empty")
    include_baseline = any(field in row for row in rows for field in _BASELINE_FIELDS)
    fields = _COMPARISON_FIELDS + (_BASELINE_FIELDS if include_baseline else ())
    destination = __import__("io").StringIO(newline="")
    writer = csv.DictWriter(
        destination,
        fieldnames=fields,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return destination.getvalue()


def render_markdown(
    rows: Sequence[dict[str, object]],
    *,
    environment: dict[str, object],
) -> str:
    """Render a human-readable performance report with variability warnings."""
    if not rows:
        raise ValueError("comparison rows must not be empty")
    run_id = environment.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("environment.run_id must be a non-empty string")

    output = [
        "# cnumpy vs NumPy Performance Report",
        "",
        f"- Run ID: `{run_id}`",
    ]
    profile = environment.get("profile")
    if isinstance(profile, str) and profile:
        output.append(f"- Profile: `{profile}`")
    output.extend(
        [
            "- Ratio: `cnumpy median / NumPy median`; lower is better for cnumpy.",
            "- Timing boundary: public NumPy call versus AHK DllCall + C work + result lifecycle.",
            "- ⚠ marks a coefficient of variation above 5%.",
            "",
            "## Category summary",
            "",
            "| Category | Cases | Geometric mean | cnumpy wins | NumPy wins |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    categories = sorted({str(row["category"]) for row in rows})
    for category in categories:
        category_rows = [row for row in rows if row["category"] == category]
        ratios = [float(row["cnumpy_over_numpy"]) for row in category_rows]
        cnumpy_wins = sum(row["winner"] == "cnumpy" for row in category_rows)
        output.append(
            f"| {category} | {len(category_rows)} | {geometric_mean(ratios):.3f}x | "
            f"{cnumpy_wins} | {len(category_rows) - cnumpy_wins} |"
        )

    for category in categories:
        output.extend(
            [
                "",
                f"## {category}",
                "",
                "| Case | Shape | NumPy median | cnumpy median | Ratio | Winner | NumPy CV | cnumpy CV |",
                "|---|---:|---:|---:|---:|---|---:|---:|",
            ]
        )
        for row in (row for row in rows if row["category"] == category):
            numpy_cv = float(row["numpy_cv"])
            cnumpy_cv = float(row["cnumpy_cv"])
            numpy_warning = " ⚠" if numpy_cv > 0.05 else ""
            cnumpy_warning = " ⚠" if cnumpy_cv > 0.05 else ""
            output.append(
                f"| `{row['id']}` | {row['shape']} | {_format_duration(float(row['numpy_median_ns']))} | "
                f"{_format_duration(float(row['cnumpy_median_ns']))} | "
                f"{float(row['cnumpy_over_numpy']):.3f}x | {row['winner']} | "
                f"{numpy_cv:.2%}{numpy_warning} | {cnumpy_cv:.2%}{cnumpy_warning} |"
            )

    output.extend(
        [
            "",
            "## Bottleneck ranking",
            "",
            "| Rank | Case | cnumpy / NumPy |",
            "|---:|---|---:|",
        ]
    )
    ranked = sorted(rows, key=lambda row: float(row["cnumpy_over_numpy"]), reverse=True)
    for rank, row in enumerate(ranked, start=1):
        output.append(f"| {rank} | `{row['id']}` | {float(row['cnumpy_over_numpy']):.3f}x |")

    if any("baseline_compatibility" in row for row in rows):
        output.extend(
            [
                "",
                "## Baseline comparison",
                "",
                "| Case | Baseline ratio | Current ratio | Ratio improvement | Compatibility |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for row in rows:
            compatibility = str(row["baseline_compatibility"])
            if compatibility == "compatible":
                baseline_ratio = (
                    f"{float(row['baseline_cnumpy_over_numpy']):.3f}x"
                )
                improvement = f"{float(row['ratio_improvement']):.3f}x"
            else:
                baseline_ratio = "N/A"
                improvement = "N/A"
            output.append(
                f"| `{row['id']}` | {baseline_ratio} | "
                f"{float(row['cnumpy_over_numpy']):.3f}x | "
                f"{improvement} | {compatibility} |"
            )
    output.append("")
    return "\n".join(output)


def _cases_by_id(document: dict[str, object]) -> dict[str, dict[str, object]]:
    cases = document["cases"]
    assert isinstance(cases, list)
    result: dict[str, dict[str, object]] = {}
    for case in cases:
        assert isinstance(case, dict)
        case_id = case["id"]
        assert isinstance(case_id, str)
        result[case_id] = case
    return result


def _require_zero_retained_bytes(case: dict[str, object], case_id: str) -> None:
    retained = case.get("retained_bytes")
    if type(retained) is not int or retained != 0:
        raise ValueError(f"cnumpy retained_bytes for {case_id!r} must be the integer zero")


def _compare_validation_signatures(
    numpy_validation: object,
    cnumpy_validation: object,
    job: dict[str, object],
    case_id: str,
) -> None:
    if not isinstance(numpy_validation, dict) or not isinstance(cnumpy_validation, dict):
        raise ValueError(f"validation for {case_id!r} must be an object")
    structural_fields = ("mode", "shape", "size", "logical_dtype")
    for field in structural_fields:
        if numpy_validation.get(field) != cnumpy_validation.get(field):
            raise ValueError(f"validation {field} mismatch for {case_id!r}")
    mode = numpy_validation.get("mode")
    if mode == "shape":
        expected_keys = {"mode", "shape", "size", "logical_dtype"}
        if set(numpy_validation) != expected_keys or set(cnumpy_validation) != expected_keys:
            raise ValueError(f"validation schema mismatch for {case_id!r}")
        return
    if mode != "numeric":
        if mode != "numeric_nan":
            raise ValueError(f"validation mode mismatch for {case_id!r}")
        nan_fields = (
            "nan_count",
            "first_nan_index",
            "trailing_nan",
            "finite_members_exact",
        )
        for field in nan_fields:
            if numpy_validation.get(field) != cnumpy_validation.get(field):
                raise ValueError(f"validation {field} mismatch for {case_id!r}")
    expected_keys = {
        "mode",
        "shape",
        "size",
        "logical_dtype",
        "sample_indices",
        "values",
        "sum",
    }
    if mode == "numeric_nan":
        expected_keys.update(
            {
                "nan_count",
                "first_nan_index",
                "trailing_nan",
                "finite_members_exact",
            }
        )
    if set(numpy_validation) != expected_keys or set(cnumpy_validation) != expected_keys:
        raise ValueError(f"validation schema mismatch for {case_id!r}")
    numpy_values = numpy_validation["values"]
    cnumpy_values = cnumpy_validation["values"]
    if not isinstance(numpy_values, list) or not isinstance(cnumpy_values, list):
        raise ValueError(f"validation values for {case_id!r} must be arrays")
    if len(numpy_values) != len(cnumpy_values):
        raise ValueError(f"validation values length mismatch for {case_id!r}")
    logical_dtype = numpy_validation["logical_dtype"]
    for index, (numpy_value, cnumpy_value) in enumerate(zip(numpy_values, cnumpy_values)):
        location = f"validation value {index} for {case_id!r}"
        if _is_argsort_operation(job["operation"]):
            _compare_argsort_validation_index(
                numpy_value,
                cnumpy_value,
                location,
                job["operation"],
            )
        else:
            _compare_validation_number(
                numpy_value,
                cnumpy_value,
                logical_dtype,
                location,
            )
    _compare_validation_number(
        numpy_validation["sum"],
        cnumpy_validation["sum"],
        logical_dtype,
        f"validation sum for {case_id!r}",
    )


def _compare_validation_number(
    numpy_value: object,
    cnumpy_value: object,
    logical_dtype: object,
    location: str,
) -> None:
    if (
        isinstance(numpy_value, bool)
        or isinstance(cnumpy_value, bool)
        or not isinstance(numpy_value, Real)
        or not isinstance(cnumpy_value, Real)
        or not math.isfinite(numpy_value)
        or not math.isfinite(cnumpy_value)
    ):
        raise ValueError(f"{location} must contain finite numbers")
    if logical_dtype in {"i64", "u8", "bool"}:
        matches = numpy_value == cnumpy_value
    elif logical_dtype == "f64":
        matches = math.isclose(
            float(numpy_value),
            float(cnumpy_value),
            rel_tol=_VALIDATION_REL_TOLERANCE,
            abs_tol=_VALIDATION_ABS_TOLERANCE,
        )
    else:
        raise ValueError(f"{location} has unsupported logical dtype {logical_dtype!r}")
    if not matches:
        raise ValueError(f"{location} mismatch: NumPy={numpy_value!r}, cnumpy={cnumpy_value!r}")


def _compare_argsort_validation_index(
    numpy_value: object,
    cnumpy_value: object,
    location: str,
    operation: object,
) -> None:
    if not isinstance(operation, str) or not _is_argsort_operation(operation):
        raise ValueError(f"{location} has invalid argsort operation {operation!r}")
    numpy_index = int(numpy_value)
    cnumpy_index = int(cnumpy_value)
    numpy_key = _argsort_validation_key(operation, numpy_index)
    cnumpy_key = _argsort_validation_key(operation, cnumpy_index)
    if numpy_key != cnumpy_key:
        raise ValueError(
            f"argsort {location} key mismatch: "
            f"NumPy index={numpy_index}, key={numpy_key}; "
            f"cnumpy index={cnumpy_index}, key={cnumpy_key}"
        )


def _is_argsort_operation(operation: object) -> bool:
    return isinstance(operation, str) and (
        operation == "argsort" or operation.startswith("argsort_")
    )


def _argsort_validation_key(operation: str, index: int) -> tuple[int, int]:
    if operation.endswith("_nan") and index % 2 == 0:
        return (1, 0)
    return (0, ((index * 48271 + 17) % 65521) % 4096)


def _case_shape(case: dict[str, object]) -> str:
    rows = case.get("rows")
    cols = case.get("cols")
    size = case.get("size")
    if type(rows) is int and type(cols) is int and rows > 0 and cols > 0:
        return f"{rows}x{cols}"
    if isinstance(case.get("id"), str) and case["id"].startswith("bridge/"):
        return "scalar"
    return str(size)


def _finite_positive_number(value: object, location: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{location} must be finite and positive")
    return float(value)


def _format_duration(value_ns: float) -> str:
    if value_ns >= 1_000_000:
        return f"{value_ns / 1_000_000:.3f} ms"
    if value_ns >= 1_000:
        return f"{value_ns / 1_000:.3f} µs"
    return f"{value_ns:.3f} ns"
