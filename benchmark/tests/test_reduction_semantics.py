from __future__ import annotations

from contextlib import ExitStack
import ctypes
from pathlib import Path
import subprocess
import sys
import unittest
import warnings
from typing import NamedTuple

import numpy as np

from compat.cnumpy_ctypes import CnumpyError, CnumpyRuntime
from compat.oracle import assert_array_equivalent


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"
CNP_NOTYPE = 0
CNP_AXIS_NONE = -1


class _LegacyReductionCase(NamedTuple):
    symbol: str
    numpy_function: object
    abi: str
    q: float | None
    v2_method: str


LEGACY_REDUCTION_EXPORT_CASES = (
    _LegacyReductionCase("cnp_sum", np.sum, "axis_dtype", None, "sum"),
    _LegacyReductionCase("cnp_prod", np.prod, "axis_dtype", None, "prod"),
    _LegacyReductionCase("cnp_mean", np.mean, "axis_dtype", None, "mean"),
    _LegacyReductionCase("cnp_var", np.var, "axis_ddof_dtype", None, "variance"),
    _LegacyReductionCase("cnp_std", np.std, "axis_ddof_dtype", None, "std"),
    _LegacyReductionCase("cnp_max", np.max, "axis", None, "maximum"),
    _LegacyReductionCase("cnp_min", np.min, "axis", None, "minimum"),
    _LegacyReductionCase("cnp_amax", np.amax, "axis", None, "maximum"),
    _LegacyReductionCase("cnp_amin", np.amin, "axis", None, "minimum"),
    _LegacyReductionCase("cnp_argmax", np.argmax, "axis", None, "argmax"),
    _LegacyReductionCase("cnp_argmin", np.argmin, "axis", None, "argmin"),
    _LegacyReductionCase("cnp_any", np.any, "axis", None, "any"),
    _LegacyReductionCase("cnp_all", np.all, "axis", None, "all"),
    _LegacyReductionCase("cnp_ptp", np.ptp, "axis", None, "ptp"),
    _LegacyReductionCase("cnp_cumsum", np.cumsum, "axis_dtype", None, "cumsum"),
    _LegacyReductionCase("cnp_cumprod", np.cumprod, "axis_dtype", None, "cumprod"),
    _LegacyReductionCase("cnp_nansum", np.nansum, "axis_dtype", None, "nansum"),
    _LegacyReductionCase("cnp_nanprod", np.nanprod, "axis_dtype", None, "nanprod"),
    _LegacyReductionCase("cnp_nanmean", np.nanmean, "axis_dtype", None, "nanmean"),
    _LegacyReductionCase("cnp_nanvar", np.nanvar, "axis_ddof_dtype", None, "nanvar"),
    _LegacyReductionCase("cnp_nanstd", np.nanstd, "axis_ddof_dtype", None, "nanstd"),
    _LegacyReductionCase("cnp_nanmax", np.nanmax, "axis", None, "nanmax"),
    _LegacyReductionCase("cnp_nanmin", np.nanmin, "axis", None, "nanmin"),
    _LegacyReductionCase("cnp_nancumsum", np.nancumsum, "axis", None, "nancumsum"),
    _LegacyReductionCase("cnp_nancumprod", np.nancumprod, "axis", None, "nancumprod"),
    _LegacyReductionCase("cnp_median", np.median, "statistic", None, "median"),
    _LegacyReductionCase("cnp_percentile", np.percentile, "statistic", 37.5, "percentile"),
    _LegacyReductionCase("cnp_quantile", np.quantile, "statistic", 0.375, "quantile"),
    _LegacyReductionCase("cnp_average", np.average, "average", None, "average"),
)


class _CnpSlice(ctypes.Structure):
    _fields_ = [
        ("start", ctypes.c_int64),
        ("stop", ctypes.c_int64),
        ("step", ctypes.c_int64),
        ("has_start", ctypes.c_bool),
        ("has_stop", ctypes.c_bool),
        ("has_step", ctypes.c_bool),
    ]


class _CnpArrayHeader(ctypes.Structure):
    _fields_ = [
        ("ndim", ctypes.c_int),
        ("shape", ctypes.c_void_p),
        ("strides", ctypes.c_void_p),
        ("size", ctypes.c_int64),
        ("data", ctypes.c_void_p),
        ("dtype", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("refcount", ctypes.c_int),
        ("base", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("owner", ctypes.c_void_p),
        ("owner_release", ctypes.c_void_p),
    ]


class ReductionSemanticsTests(unittest.TestCase):
    def legacy_reduction(
        self,
        runtime: CnumpyRuntime,
        source,
        function_name: str,
        arguments: tuple[int, ...],
    ):
        function = getattr(runtime.dll, function_name)
        function.argtypes = [ctypes.c_void_p] + [
            ctypes.c_int
        ] * len(arguments)
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        pointer = function(source.pointer, *arguments)
        return runtime._owned_result(pointer, function_name)

    def slice_view(
        self, runtime: CnumpyRuntime, source, slices: tuple[_CnpSlice, ...]
    ):
        function = runtime.dll.cnp_array_slice
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(_CnpSlice),
        ]
        function.restype = ctypes.c_void_p
        slice_storage = (_CnpSlice * len(slices))(*slices)
        runtime.dll.cnp_clear_error()
        pointer = function(source.pointer, len(slices), slice_storage)
        return runtime._owned_result(pointer, "cnp_array_slice")

    def broadcast_view(
        self, runtime: CnumpyRuntime, source, shape: tuple[int, ...]
    ):
        function = runtime.dll.cnp_broadcast_to
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
        ]
        function.restype = ctypes.c_void_p
        shape_storage = (ctypes.c_int64 * len(shape))(*shape)
        runtime.dll.cnp_clear_error()
        pointer = function(source.pointer, len(shape), shape_storage)
        return runtime._owned_result(pointer, "cnp_broadcast_to")

    def legacy_statistic(
        self,
        runtime: CnumpyRuntime,
        source,
        function_name: str,
        q: float | None,
        axis: int,
    ):
        function = getattr(runtime.dll, function_name)
        if q is None:
            function.argtypes = [ctypes.c_void_p, ctypes.c_int]
            arguments = (source.pointer, int(axis))
        else:
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_double,
                ctypes.c_int,
            ]
            arguments = (source.pointer, float(q), int(axis))
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        pointer = function(*arguments)
        return runtime._owned_result(pointer, function_name)

    def legacy_export_pointer(
        self,
        runtime: CnumpyRuntime,
        source_pointer,
        case: _LegacyReductionCase,
        axis: int,
    ):
        function = getattr(runtime.dll, case.symbol)
        if case.abi == "axis_dtype":
            function.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ]
            arguments = (source_pointer, int(axis), CNP_NOTYPE)
        elif case.abi == "axis_ddof_dtype":
            function.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ]
            arguments = (source_pointer, int(axis), 0, CNP_NOTYPE)
        elif case.abi == "axis":
            function.argtypes = [ctypes.c_void_p, ctypes.c_int]
            arguments = (source_pointer, int(axis))
        elif case.abi == "statistic":
            if case.q is None:
                function.argtypes = [ctypes.c_void_p, ctypes.c_int]
                arguments = (source_pointer, int(axis))
            else:
                function.argtypes = [
                    ctypes.c_void_p, ctypes.c_double, ctypes.c_int,
                ]
                arguments = (source_pointer, float(case.q), int(axis))
        elif case.abi == "average":
            function.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
            ]
            arguments = (source_pointer, int(axis), None)
        else:
            raise AssertionError(f"unknown legacy ABI {case.abi!r}")
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        return function(*arguments)

    def legacy_export_result(
        self,
        runtime: CnumpyRuntime,
        source,
        case: _LegacyReductionCase,
        axis: int,
    ):
        pointer = self.legacy_export_pointer(
            runtime, source.pointer, case, axis
        )
        return runtime._owned_result(pointer, case.symbol)

    def numpy_legacy_export_result(
        self,
        case: _LegacyReductionCase,
        value: np.ndarray,
        axis: int | None,
    ) -> np.ndarray:
        function = case.numpy_function
        assert callable(function)
        if case.q is None:
            return np.asarray(function(value, axis=axis))
        return np.asarray(function(value, case.q, axis=axis))

    def v2_export_result(
        self,
        runtime: CnumpyRuntime,
        source,
        case: _LegacyReductionCase,
        axis: int | None,
    ):
        method = getattr(runtime, case.v2_method)
        if case.q is None:
            return method(source, axis)
        return method(source, case.q, axis)

    def test_legacy_statistics_preserve_axis_results_and_noncontiguous_views(
        self,
    ) -> None:
        two_dimensional = np.array(
            [
                [9.0, 1.0, 7.0, 3.0],
                [2.0, 8.0, 4.0, 6.0],
                [5.0, 0.0, 11.0, 10.0],
            ],
            dtype=np.float64,
        )
        three_dimensional_storage = (
            np.arange(24, dtype=np.float64).reshape(2, 3, 4) * 1.25
            - 7.0
        )
        three_dimensional = three_dimensional_storage.transpose(2, 0, 1)
        operations = (
            ("cnp_median", np.median, None),
            ("cnp_percentile", np.percentile, 37.5),
            ("cnp_quantile", np.quantile, 0.375),
        )

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            two_source = stack.enter_context(runtime.from_numpy(two_dimensional))
            three_storage = stack.enter_context(
                runtime.from_numpy(three_dimensional_storage)
            )
            three_source = stack.enter_context(
                runtime.transpose(three_storage, (2, 0, 1))
            )
            self.assertFalse(three_source.c_contiguous)

            for layout, source, source_value in (
                ("2d_contiguous", two_source, two_dimensional),
                ("3d_transposed", three_source, three_dimensional),
            ):
                for function_name, numpy_function, q in operations:
                    for axis in (CNP_AXIS_NONE, 0, 1, -2):
                        with self.subTest(
                            layout=layout,
                            function=function_name,
                            axis=axis,
                        ):
                            numpy_axis = (
                                None if axis == CNP_AXIS_NONE else axis
                            )
                            expected = (
                                numpy_function(source_value, axis=numpy_axis)
                                if q is None
                                else numpy_function(
                                    source_value, q, axis=numpy_axis
                                )
                            )
                            result = stack.enter_context(
                                self.legacy_statistic(
                                    runtime,
                                    source,
                                    function_name,
                                    q,
                                    axis,
                                )
                            )
                            assert_array_equivalent(
                                self,
                                result,
                                expected,
                                compare_contiguity=False,
                            )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_percentile_and_quantile_report_precise_q_errors(
        self,
    ) -> None:
        source_value = np.arange(12, dtype=np.float64).reshape(3, 4)
        cases = (
            ("cnp_percentile", -0.125, 0, "q must be in [0, 100]"),
            ("cnp_percentile", 100.125, -2, "q must be in [0, 100]"),
            ("cnp_quantile", -0.125, 1, "q must be in [0, 1]"),
            ("cnp_quantile", 1.125, -2, "q must be in [0, 1]"),
        )

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(source_value) as source:
                for function_name, q, axis, message in cases:
                    with self.subTest(
                        function=function_name, q=q, axis=axis
                    ):
                        before_error = runtime.retained_bytes
                        with self.assertRaises(CnumpyError) as captured:
                            self.legacy_statistic(
                                runtime,
                                source,
                                function_name,
                                q,
                                axis,
                            )
                        self.assertEqual(-1, captured.exception.status)
                        self.assertEqual(
                            function_name,
                            captured.exception.function,
                        )
                        self.assertEqual(message, captured.exception.message)
                        self.assertEqual(before_error, runtime.retained_bytes)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_nan_statistics_reject_unrepresentable_and_invalid_calls(
        self,
    ) -> None:
        source_value = np.asarray(
            [[np.nan, 1.0], [2.0, 3.0]], dtype=np.float64
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            nanmedian = runtime.dll.cnp_nanmedian
            nanmedian.argtypes = [ctypes.c_void_p, ctypes.c_int]
            nanmedian.restype = ctypes.c_double
            nanpercentile = runtime.dll.cnp_nanpercentile
            nanpercentile.argtypes = [
                ctypes.c_void_p,
                ctypes.c_double,
                ctypes.c_int,
            ]
            nanpercentile.restype = ctypes.c_double
            nanquantile = runtime.dll.cnp_nanquantile
            nanquantile.argtypes = [
                ctypes.c_void_p,
                ctypes.c_double,
                ctypes.c_int,
            ]
            nanquantile.restype = ctypes.c_double

            with runtime.from_numpy(source_value) as source:
                for function_name, invoke in (
                    (
                        "cnp_nanmedian",
                        lambda: nanmedian(source.pointer, 0),
                    ),
                    (
                        "cnp_nanpercentile",
                        lambda: nanpercentile(source.pointer, 50.0, 0),
                    ),
                    (
                        "cnp_nanquantile",
                        lambda: nanquantile(source.pointer, 0.5, 0),
                    ),
                ):
                    with self.subTest(function=function_name, case="axis"):
                        before_call = runtime.retained_bytes
                        runtime.dll.cnp_clear_error()
                        self.assertTrue(np.isnan(invoke()))
                        error = runtime.error_state()
                        self.assertEqual(-4, error.status)
                        self.assertEqual(function_name, error.function)
                        self.assertIn("cannot represent", error.message)
                        self.assertEqual(before_call, runtime.retained_bytes)

                for function_name, invoke in (
                    (
                        "cnp_nanpercentile",
                        lambda: nanpercentile(source.pointer, -0.5, -1),
                    ),
                    (
                        "cnp_nanquantile",
                        lambda: nanquantile(source.pointer, -0.005, -1),
                    ),
                ):
                    with self.subTest(function=function_name, case="q"):
                        runtime.dll.cnp_clear_error()
                        self.assertTrue(np.isnan(invoke()))
                        error = runtime.error_state()
                        self.assertEqual(-1, error.status)
                        self.assertEqual(function_name, error.function)
                        self.assertIn("q must be", error.message)

            for function_name, invoke in (
                ("cnp_nanmedian", lambda: nanmedian(None, -1)),
                (
                    "cnp_nanpercentile",
                    lambda: nanpercentile(None, 50.0, -1),
                ),
                (
                    "cnp_nanquantile",
                    lambda: nanquantile(None, 0.5, -1),
                ),
            ):
                with self.subTest(function=function_name, case="null"):
                    runtime.dll.cnp_clear_error()
                    self.assertTrue(np.isnan(invoke()))
                    error = runtime.error_state()
                    self.assertEqual(-1, error.status)
                    self.assertEqual(function_name, error.function)
                    self.assertIn("required", error.message)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_nan_statistics_return_scalar_numpy_values(self) -> None:
        flat_value = np.asarray(
            [[np.nan, -4.0, 8.0], [2.0, 10.0, 6.0]],
            dtype=np.float64,
        )
        strided_storage_value = np.asarray(
            [np.nan, 99.0, -3.0, 98.0, 7.0, 97.0, 11.0],
            dtype=np.float64,
        )
        strided_value = strided_storage_value[::2]
        scalar_cases = (
            ("flattened", flat_value, CNP_AXIS_NONE),
            ("one_dimensional", flat_value.reshape(-1), 0),
            ("empty", np.empty((0,), dtype=np.float64), 0),
            ("all_nan", np.full((4,), np.nan, dtype=np.float64), 0),
        )

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            nanmedian = runtime.dll.cnp_nanmedian
            nanmedian.argtypes = [ctypes.c_void_p, ctypes.c_int]
            nanmedian.restype = ctypes.c_double
            nanpercentile = runtime.dll.cnp_nanpercentile
            nanpercentile.argtypes = [
                ctypes.c_void_p,
                ctypes.c_double,
                ctypes.c_int,
            ]
            nanpercentile.restype = ctypes.c_double
            nanquantile = runtime.dll.cnp_nanquantile
            nanquantile.argtypes = [
                ctypes.c_void_p,
                ctypes.c_double,
                ctypes.c_int,
            ]
            nanquantile.restype = ctypes.c_double

            for layout, source_value, axis in scalar_cases:
                with runtime.from_numpy(source_value) as source:
                    numpy_axis = (
                        None if axis == CNP_AXIS_NONE else axis
                    )
                    for function_name, numpy_function, invoke in (
                        (
                            "nanmedian",
                            np.nanmedian,
                            lambda: nanmedian(source.pointer, axis),
                        ),
                        (
                            "nanpercentile",
                            lambda value, axis: np.nanpercentile(
                                value, 50.0, axis=axis
                            ),
                            lambda: nanpercentile(
                                source.pointer, 50.0, axis
                            ),
                        ),
                        (
                            "nanquantile",
                            lambda value, axis: np.nanquantile(
                                value, 0.5, axis=axis
                            ),
                            lambda: nanquantile(
                                source.pointer, 0.5, axis
                            ),
                        ),
                    ):
                        with self.subTest(
                            layout=layout, function=function_name
                        ):
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", RuntimeWarning)
                                expected = numpy_function(
                                    source_value, axis=numpy_axis
                                )
                            runtime.dll.cnp_clear_error()
                            actual = invoke()
                            np.testing.assert_allclose(
                                actual, expected, rtol=0.0, atol=0.0,
                                equal_nan=True,
                            )
                            self.assertEqual(0, runtime.error_state().status)

            with runtime.from_numpy(strided_storage_value) as storage:
                view = self.slice_view(
                    runtime,
                    storage,
                    (
                        _CnpSlice(
                            0, 0, 2,
                            False, False, True,
                        ),
                    ),
                )
                with view:
                    self.assertFalse(view.c_contiguous)
                    for q in (0.0, 25.0, 50.0, 75.0, 100.0):
                        with self.subTest(layout="strided", q=q):
                            expected = np.nanpercentile(strided_value, q)
                            before_call = runtime.retained_bytes
                            runtime.dll.cnp_clear_error()
                            actual = nanpercentile(view.pointer, q, 0)
                            np.testing.assert_allclose(
                                actual, expected, rtol=0.0, atol=0.0,
                                equal_nan=True,
                            )
                            self.assertEqual(0, runtime.error_state().status)
                            self.assertEqual(
                                before_call, runtime.retained_bytes
                            )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_nanarg_extrema_return_only_representable_scalar_results(
        self,
    ) -> None:
        source_value = np.array(
            [[np.nan, 4.0, 1.0], [2.0, -3.0, np.nan]],
            dtype=np.float64,
        )
        all_nan_value = np.full(4, np.nan, dtype=np.float64)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            all_nan = stack.enter_context(runtime.from_numpy(all_nan_value))
            for function_name, numpy_function in (
                ("cnp_nanargmax", np.nanargmax),
                ("cnp_nanargmin", np.nanargmin),
            ):
                function = getattr(runtime.dll, function_name)
                function.argtypes = [ctypes.c_void_p, ctypes.c_int]
                function.restype = ctypes.c_int64

                runtime.dll.cnp_clear_error()
                self.assertEqual(
                    int(numpy_function(source_value)),
                    function(source.pointer, CNP_AXIS_NONE),
                )
                self.assertEqual(0, runtime.error_state().status)

                runtime.dll.cnp_clear_error()
                self.assertEqual(-1, function(source.pointer, 0))
                error = runtime.error_state()
                self.assertEqual(-4, error.status)
                self.assertEqual(function_name, error.function)
                self.assertIn("cannot represent", error.message)

                for label, pointer in (
                    ("all_nan", all_nan.pointer),
                    ("null", None),
                ):
                    with self.subTest(function=function_name, case=label):
                        runtime.dll.cnp_clear_error()
                        self.assertEqual(
                            -1, function(pointer, CNP_AXIS_NONE)
                        )
                        error = runtime.error_state()
                        self.assertEqual(-1, error.status)
                        self.assertEqual(function_name, error.function)
                        self.assertTrue(error.message)
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_nan_scalar_exports_cover_rank_zero_through_four_axes_values_errors_and_lifetimes(
        self,
    ) -> None:
        shapes = ((), (5,), (2, 3), (2, 2, 3), (2, 2, 2, 3))
        cases = (
            ("cnp_nanmedian", np.nanmedian, None, ctypes.c_double),
            (
                "cnp_nanpercentile",
                np.nanpercentile,
                37.5,
                ctypes.c_double,
            ),
            (
                "cnp_nanquantile",
                np.nanquantile,
                0.375,
                ctypes.c_double,
            ),
            ("cnp_nanargmax", np.nanargmax, None, ctypes.c_int64),
            ("cnp_nanargmin", np.nanargmin, None, ctypes.c_int64),
        )

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            functions = {}
            for function_name, _, q, result_type in cases:
                function = getattr(runtime.dll, function_name)
                function.argtypes = (
                    [ctypes.c_void_p, ctypes.c_int]
                    if q is None else
                    [ctypes.c_void_p, ctypes.c_double, ctypes.c_int]
                )
                function.restype = result_type
                functions[function_name] = function

            for shape in shapes:
                rank = len(shape)
                if rank == 0:
                    source_value = np.array(5.0, dtype=np.float64)
                    axes = (CNP_AXIS_NONE, 0, 1, -2)
                else:
                    size = int(np.prod(shape))
                    flat_value = (
                        np.arange(size, dtype=np.float64) % 5.0
                    ) - 2.0
                    flat_value[0] = np.nan
                    flat_value[1] = 7.0
                    flat_value[2] = -6.0
                    flat_value[3] = 7.0
                    flat_value[4] = -6.0
                    source_value = flat_value.reshape(shape)
                    axes = (
                        (CNP_AXIS_NONE,)
                        + tuple(range(rank))
                        + tuple(range(-2, -rank - 1, -1))
                        + (rank, -rank - 1)
                    )

                with runtime.from_numpy(source_value) as source:
                    active = runtime.retained_bytes
                    for function_name, numpy_function, q, _ in cases:
                        function = functions[function_name]
                        scalar_result = function_name.startswith(
                            "cnp_nanarg"
                        )
                        for axis in axes:
                            with self.subTest(
                                rank=rank,
                                function=function_name,
                                axis=axis,
                            ):
                                runtime.dll.cnp_clear_error()
                                actual = (
                                    function(source.pointer, axis)
                                    if q is None else
                                    function(source.pointer, q, axis)
                                )
                                error = runtime.error_state()
                                invalid_axis = (
                                    axis != CNP_AXIS_NONE
                                    and (
                                        (
                                            rank == 0
                                            and not (
                                                scalar_result and axis == 0
                                            )
                                        )
                                        or (
                                            rank > 0
                                            and (
                                                axis < -rank
                                                or axis >= rank
                                            )
                                        )
                                    )
                                )
                                if invalid_axis:
                                    if scalar_result:
                                        self.assertEqual(-1, actual)
                                    else:
                                        self.assertTrue(np.isnan(actual))
                                    self.assertEqual(-5, error.status)
                                    self.assertEqual(
                                        function_name, error.function
                                    )
                                    self.assertEqual(
                                        f"axis {axis} is out of bounds for "
                                        f"array of dimension {rank}",
                                        error.message,
                                    )
                                else:
                                    numpy_axis = (
                                        None
                                        if axis == CNP_AXIS_NONE
                                        else axis
                                    )
                                    with warnings.catch_warnings():
                                        warnings.simplefilter(
                                            "ignore", RuntimeWarning
                                        )
                                        expected = (
                                            numpy_function(
                                                source_value,
                                                axis=numpy_axis,
                                            )
                                            if q is None else
                                            numpy_function(
                                                source_value,
                                                q,
                                                axis=numpy_axis,
                                            )
                                        )
                                    expected_array = np.asarray(expected)
                                    if expected_array.ndim != 0:
                                        if scalar_result:
                                            self.assertEqual(-1, actual)
                                        else:
                                            self.assertTrue(np.isnan(actual))
                                        self.assertEqual(-4, error.status)
                                        self.assertEqual(
                                            function_name, error.function
                                        )
                                        self.assertEqual(
                                            "legacy scalar return cannot "
                                            "represent an array result",
                                            error.message,
                                        )
                                    elif scalar_result:
                                        self.assertEqual(
                                            int(expected_array), actual
                                        )
                                        self.assertEqual(0, error.status)
                                    else:
                                        np.testing.assert_array_equal(
                                            np.asarray(
                                                actual, dtype=np.float64
                                            ),
                                            expected_array.astype(
                                                np.float64, copy=False
                                            ),
                                            strict=True,
                                        )
                                        self.assertEqual(0, error.status)
                                self.assertEqual(
                                    active, runtime.retained_bytes
                                )

            all_nan_value = np.full((4,), np.nan, dtype=np.float64)
            with runtime.from_numpy(all_nan_value) as all_nan:
                active = runtime.retained_bytes
                for function_name, numpy_function, q, _ in cases:
                    with self.subTest(
                        function=function_name, case="all_nan"
                    ):
                        function = functions[function_name]
                        runtime.dll.cnp_clear_error()
                        actual = (
                            function(all_nan.pointer, CNP_AXIS_NONE)
                            if q is None else
                            function(
                                all_nan.pointer, q, CNP_AXIS_NONE
                            )
                        )
                        error = runtime.error_state()
                        if function_name.startswith("cnp_nanarg"):
                            self.assertEqual(-1, actual)
                            self.assertEqual(-1, error.status)
                            self.assertEqual(
                                function_name, error.function
                            )
                            self.assertEqual(
                                "All-NaN slice encountered", error.message
                            )
                        else:
                            with warnings.catch_warnings():
                                warnings.simplefilter(
                                    "ignore", RuntimeWarning
                                )
                                expected = (
                                    numpy_function(all_nan_value)
                                    if q is None else
                                    numpy_function(all_nan_value, q)
                                )
                            self.assertTrue(np.isnan(expected))
                            self.assertTrue(np.isnan(actual))
                            self.assertEqual(0, error.status)
                        self.assertEqual(active, runtime.retained_bytes)

            for function_name, _, q, _ in cases:
                with self.subTest(function=function_name, case="null"):
                    function = functions[function_name]
                    runtime.dll.cnp_clear_error()
                    actual = (
                        function(None, CNP_AXIS_NONE)
                        if q is None else
                        function(None, q, CNP_AXIS_NONE)
                    )
                    error = runtime.error_state()
                    if function_name.startswith("cnp_nanarg"):
                        self.assertEqual(-1, actual)
                    else:
                        self.assertTrue(np.isnan(actual))
                    self.assertEqual(-1, error.status)
                    self.assertEqual(function_name, error.function)
                    self.assertEqual(
                        "source array is required", error.message
                    )
                    self.assertEqual(baseline, runtime.retained_bytes)

            q_source_value = np.arange(4, dtype=np.float64)
            with runtime.from_numpy(q_source_value) as q_source:
                active = runtime.retained_bytes
                for function_name, q, message in (
                    (
                        "cnp_nanpercentile",
                        -0.5,
                        "q must be in [0, 100]",
                    ),
                    (
                        "cnp_nanquantile",
                        1.5,
                        "q must be in [0, 1]",
                    ),
                ):
                    with self.subTest(function=function_name, case="q"):
                        runtime.dll.cnp_clear_error()
                        actual = functions[function_name](
                            q_source.pointer, q, CNP_AXIS_NONE
                        )
                        error = runtime.error_state()
                        self.assertTrue(np.isnan(actual))
                        self.assertEqual(-1, error.status)
                        self.assertEqual(function_name, error.function)
                        self.assertEqual(message, error.message)
                        self.assertEqual(active, runtime.retained_bytes)

            self.assertEqual(baseline, runtime.retained_bytes)

    def test_percentile_and_quantile_families_reject_nan_q(self) -> None:
        source_value = np.arange(12, dtype=np.float64).reshape(3, 4)
        cases = (
            ("cnp_percentile_v2", np.percentile, False),
            ("cnp_nanpercentile_v2", np.nanpercentile, False),
            ("cnp_quantile_v2", np.quantile, False),
            ("cnp_nanquantile_v2", np.nanquantile, False),
            ("cnp_percentile", np.percentile, True),
            ("cnp_quantile", np.quantile, True),
        )

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(source_value) as source:
                for function_name, numpy_function, legacy in cases:
                    with self.subTest(function=function_name):
                        with self.assertRaises(ValueError):
                            numpy_function(source_value, np.nan, axis=0)
                        before_error = runtime.retained_bytes
                        with self.assertRaises(CnumpyError) as captured:
                            if legacy:
                                self.legacy_statistic(
                                    runtime,
                                    source,
                                    function_name,
                                    np.nan,
                                    0,
                                )
                            else:
                                method_name = function_name.removeprefix(
                                    "cnp_"
                                ).removesuffix("_v2")
                                getattr(runtime, method_name)(
                                    source, np.nan, 0
                                )
                        self.assertEqual(-1, captured.exception.status)
                        self.assertIn(
                            "q must be in", captured.exception.message
                        )
                        self.assertEqual(
                            before_error, runtime.retained_bytes
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_reduction_errors_name_the_called_entry_point(self) -> None:
        source_value = np.arange(12, dtype=np.float64).reshape(3, 4)
        reduction_cases = (
            ("cnp_sum", (7, CNP_NOTYPE)),
            ("cnp_prod", (7, CNP_NOTYPE)),
            ("cnp_mean", (7, CNP_NOTYPE)),
            ("cnp_var", (7, 0, CNP_NOTYPE)),
            ("cnp_std", (7, 0, CNP_NOTYPE)),
            ("cnp_max", (7,)),
            ("cnp_min", (7,)),
            ("cnp_amax", (7,)),
            ("cnp_amin", (7,)),
            ("cnp_argmax", (7,)),
            ("cnp_argmin", (7,)),
            ("cnp_any", (7,)),
            ("cnp_all", (7,)),
            ("cnp_ptp", (7,)),
            ("cnp_cumsum", (7, CNP_NOTYPE)),
            ("cnp_cumprod", (7, CNP_NOTYPE)),
            ("cnp_nansum", (7, CNP_NOTYPE)),
            ("cnp_nanprod", (7, CNP_NOTYPE)),
            ("cnp_nanmean", (7, CNP_NOTYPE)),
            ("cnp_nanvar", (7, 0, CNP_NOTYPE)),
            ("cnp_nanstd", (7, 0, CNP_NOTYPE)),
            ("cnp_nanmax", (7,)),
            ("cnp_nanmin", (7,)),
            ("cnp_nancumsum", (7,)),
            ("cnp_nancumprod", (7,)),
        )
        statistic_cases = (
            ("cnp_median", None),
            ("cnp_percentile", 37.5),
            ("cnp_quantile", 0.375),
        )

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(source_value) as source:
                for function_name, arguments in reduction_cases:
                    with self.subTest(function=function_name):
                        before_error = runtime.retained_bytes
                        with self.assertRaises(CnumpyError) as captured:
                            self.legacy_reduction(
                                runtime,
                                source,
                                function_name,
                                arguments,
                            )
                        self.assertEqual(
                            function_name, captured.exception.function
                        )
                        self.assertEqual(
                            before_error, runtime.retained_bytes
                        )
                for function_name, q in statistic_cases:
                    with self.subTest(function=function_name):
                        before_error = runtime.retained_bytes
                        with self.assertRaises(CnumpyError) as captured:
                            self.legacy_statistic(
                                runtime,
                                source,
                                function_name,
                                q,
                                7,
                            )
                        self.assertEqual(
                            function_name, captured.exception.function
                        )
                        self.assertEqual(
                            before_error, runtime.retained_bytes
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_ptp_v2_errors_name_the_called_entry_point(self) -> None:
        source_value = np.arange(12, dtype=np.float64).reshape(3, 4)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(source_value) as source:
                before_error = runtime.retained_bytes
                with self.assertRaises(CnumpyError) as captured:
                    runtime.ptp(source, 7)
                self.assertEqual(
                    "cnp_ptp_v2", captured.exception.function
                )
                self.assertEqual(before_error, runtime.retained_bytes)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_all_legacy_reduction_abis_match_numpy_directly(self) -> None:
        axis = 1
        finite_value = np.array(
            [[2, 3, 4], [5, 6, 7]], dtype=np.int32
        )
        nan_value = np.array(
            [[np.nan, 2.0, 3.0], [4.0, np.nan, 6.0]],
            dtype=np.float64,
        )
        reduction_cases = (
            ("cnp_sum", "finite", (axis, CNP_NOTYPE), np.sum(finite_value, axis=axis)),
            ("cnp_prod", "finite", (axis, CNP_NOTYPE), np.prod(finite_value, axis=axis)),
            ("cnp_mean", "finite", (axis, CNP_NOTYPE), np.mean(finite_value, axis=axis)),
            ("cnp_var", "finite", (axis, 0, CNP_NOTYPE), np.var(finite_value, axis=axis)),
            ("cnp_std", "finite", (axis, 0, CNP_NOTYPE), np.std(finite_value, axis=axis)),
            ("cnp_max", "finite", (axis,), np.max(finite_value, axis=axis)),
            ("cnp_min", "finite", (axis,), np.min(finite_value, axis=axis)),
            ("cnp_amax", "finite", (axis,), np.amax(finite_value, axis=axis)),
            ("cnp_amin", "finite", (axis,), np.amin(finite_value, axis=axis)),
            ("cnp_argmax", "finite", (axis,), np.argmax(finite_value, axis=axis)),
            ("cnp_argmin", "finite", (axis,), np.argmin(finite_value, axis=axis)),
            ("cnp_any", "finite", (axis,), np.any(finite_value, axis=axis)),
            ("cnp_all", "finite", (axis,), np.all(finite_value, axis=axis)),
            ("cnp_ptp", "finite", (axis,), np.ptp(finite_value, axis=axis)),
            ("cnp_cumsum", "finite", (axis, CNP_NOTYPE), np.cumsum(finite_value, axis=axis)),
            ("cnp_cumprod", "finite", (axis, CNP_NOTYPE), np.cumprod(finite_value, axis=axis)),
            ("cnp_nansum", "nan", (axis, CNP_NOTYPE), np.nansum(nan_value, axis=axis)),
            ("cnp_nanprod", "nan", (axis, CNP_NOTYPE), np.nanprod(nan_value, axis=axis)),
            ("cnp_nanmean", "nan", (axis, CNP_NOTYPE), np.nanmean(nan_value, axis=axis)),
            ("cnp_nanvar", "nan", (axis, 0, CNP_NOTYPE), np.nanvar(nan_value, axis=axis)),
            ("cnp_nanstd", "nan", (axis, 0, CNP_NOTYPE), np.nanstd(nan_value, axis=axis)),
            ("cnp_nanmax", "nan", (axis,), np.nanmax(nan_value, axis=axis)),
            ("cnp_nanmin", "nan", (axis,), np.nanmin(nan_value, axis=axis)),
            ("cnp_nancumsum", "nan", (axis,), np.nancumsum(nan_value, axis=axis)),
            ("cnp_nancumprod", "nan", (axis,), np.nancumprod(nan_value, axis=axis)),
        )
        statistic_cases = (
            ("cnp_median", None, np.median(finite_value, axis=axis)),
            ("cnp_percentile", 37.5, np.percentile(finite_value, 37.5, axis=axis)),
            ("cnp_quantile", 0.375, np.quantile(finite_value, 0.375, axis=axis)),
        )

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            sources = {
                "finite": stack.enter_context(
                    runtime.from_numpy(finite_value)
                ),
                "nan": stack.enter_context(runtime.from_numpy(nan_value)),
            }
            for function_name, source_name, arguments, expected in reduction_cases:
                with self.subTest(function=function_name):
                    result = stack.enter_context(
                        self.legacy_reduction(
                            runtime,
                            sources[source_name],
                            function_name,
                            arguments,
                        )
                    )
                    assert_array_equivalent(
                        self,
                        result,
                        expected,
                        compare_contiguity=False,
                    )
            for function_name, q, expected in statistic_cases:
                with self.subTest(function=function_name):
                    result = stack.enter_context(
                        self.legacy_statistic(
                            runtime,
                            sources["finite"],
                            function_name,
                            q,
                            axis,
                        )
                    )
                    assert_array_equivalent(
                        self,
                        result,
                        expected,
                        compare_contiguity=False,
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_every_legacy_reduction_export_covers_rank_zero_through_four_and_expressible_axes(
        self,
    ) -> None:
        self.assertEqual(29, len(LEGACY_REDUCTION_EXPORT_CASES))
        shapes = ((), (2,), (2, 3), (2, 2, 3), (2, 2, 2, 3))
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for shape in shapes:
                size = int(np.prod(shape)) if shape else 1
                finite_value = (
                    np.array(3.0, dtype=np.float64)
                    if not shape
                    else (
                        np.arange(size, dtype=np.float64) % 5.0 + 1.0
                    ).reshape(shape)
                )
                nan_value = finite_value.copy()
                nan_value.flat[0] = np.nan
                axes = (
                    (CNP_AXIS_NONE, 0)
                    if not shape
                    else (
                        (CNP_AXIS_NONE,)
                        + tuple(range(len(shape)))
                        + tuple(range(-2, -len(shape) - 1, -1))
                    )
                )
                with (
                    runtime.from_numpy(finite_value) as finite_source,
                    runtime.from_numpy(nan_value) as nan_source,
                ):
                    for case in LEGACY_REDUCTION_EXPORT_CASES:
                        source_value = (
                            nan_value
                            if case.symbol.startswith("cnp_nan")
                            else finite_value
                        )
                        source = (
                            nan_source
                            if case.symbol.startswith("cnp_nan")
                            else finite_source
                        )
                        for axis in axes:
                            with self.subTest(
                                symbol=case.symbol,
                                rank=len(shape),
                                axis=axis,
                            ):
                                numpy_axis = (
                                    None if axis == CNP_AXIS_NONE else axis
                                )
                                with warnings.catch_warnings(), np.errstate(
                                    invalid="ignore", divide="ignore"
                                ):
                                    warnings.simplefilter("ignore")
                                    try:
                                        expected = self.numpy_legacy_export_result(
                                            case, source_value, numpy_axis
                                        )
                                    except (IndexError, ValueError):
                                        before_error = runtime.retained_bytes
                                        with self.assertRaises(
                                            CnumpyError
                                        ) as captured:
                                            self.legacy_export_result(
                                                runtime, source, case, axis
                                            )
                                        self.assertEqual(
                                            -5, captured.exception.status
                                        )
                                        self.assertEqual(
                                            case.symbol,
                                            captured.exception.function,
                                        )
                                        self.assertEqual(
                                            f"axis {axis} is out of bounds for array of dimension {len(shape)}",
                                            captured.exception.message,
                                        )
                                        self.assertEqual(
                                            before_error,
                                            runtime.retained_bytes,
                                        )
                                        continue
                                before_result = runtime.retained_bytes
                                with self.legacy_export_result(
                                    runtime, source, case, axis
                                ) as result:
                                    assert_array_equivalent(
                                        self,
                                        result,
                                        expected,
                                        compare_contiguity=False,
                                        rtol=16 * np.finfo(np.float64).eps,
                                    )
                                self.assertEqual(
                                    before_result, runtime.retained_bytes
                                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_every_legacy_reduction_export_reports_exact_axis_errors(
        self,
    ) -> None:
        self.assertEqual(29, len(LEGACY_REDUCTION_EXPORT_CASES))
        source_value = np.arange(6, dtype=np.float64).reshape(2, 3)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for case in LEGACY_REDUCTION_EXPORT_CASES:
                with self.subTest(symbol=case.symbol, error="null"):
                    pointer = self.legacy_export_pointer(
                        runtime, None, case, CNP_AXIS_NONE
                    )
                    if pointer:
                        runtime._owned_result(pointer, case.symbol).close()
                        self.fail(f"{case.symbol} accepted a null source")
                    error = runtime.error_state()
                    self.assertEqual(-1, error.status)
                    self.assertEqual(case.symbol, error.function)
                    self.assertEqual(
                        (
                            "source array is required"
                            if case.symbol == "cnp_average"
                            else "source array must not be null"
                        ),
                        error.message,
                    )
                    self.assertEqual(baseline, runtime.retained_bytes)

            with runtime.from_numpy(source_value) as source:
                active = runtime.retained_bytes
                for case in LEGACY_REDUCTION_EXPORT_CASES:
                    with self.subTest(symbol=case.symbol, error="axis"):
                        pointer = self.legacy_export_pointer(
                            runtime, source.pointer, case, 7
                        )
                        if pointer:
                            runtime._owned_result(
                                pointer, case.symbol
                            ).close()
                            self.fail(f"{case.symbol} accepted axis 7")
                        error = runtime.error_state()
                        self.assertEqual(-5, error.status)
                        self.assertEqual(case.symbol, error.function)
                        self.assertEqual(
                            "axis 7 is out of bounds for array of dimension 2",
                            error.message,
                        )
                        self.assertEqual(active, runtime.retained_bytes)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_reduction_axis_minus_one_characterizes_none_sentinel(
        self,
    ) -> None:
        self.assertEqual(29, len(LEGACY_REDUCTION_EXPORT_CASES))
        source_value = np.arange(1, 7, dtype=np.float64).reshape(2, 3)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(source_value) as source:
                active = runtime.retained_bytes
                for case in LEGACY_REDUCTION_EXPORT_CASES:
                    with self.subTest(symbol=case.symbol):
                        with warnings.catch_warnings(), np.errstate(
                            invalid="ignore", divide="ignore"
                        ):
                            warnings.simplefilter("ignore")
                            expected_none = self.numpy_legacy_export_result(
                                case, source_value, None
                            )
                            expected_last = self.numpy_legacy_export_result(
                                case, source_value, -1
                            )
                        with (
                            self.legacy_export_result(
                                runtime, source, case, CNP_AXIS_NONE
                            ) as legacy,
                            self.v2_export_result(
                                runtime, source, case, -1
                            ) as explicit_last,
                        ):
                            assert_array_equivalent(
                                self,
                                legacy,
                                expected_none,
                                compare_contiguity=False,
                                rtol=16 * np.finfo(np.float64).eps,
                            )
                            assert_array_equivalent(
                                self,
                                explicit_last,
                                expected_last,
                                compare_contiguity=False,
                                rtol=16 * np.finfo(np.float64).eps,
                            )
                            self.assertNotEqual(
                                legacy.shape, explicit_last.shape
                            )
                        self.assertEqual(active, runtime.retained_bytes)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_sum_distinguishes_none_from_every_negative_axis(self) -> None:
        source_value = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        logical_value = source_value.transpose(2, 0, 1)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            transposed = stack.enter_context(runtime.transpose(source, (2, 0, 1)))
            self.assertTrue(hasattr(runtime.dll, "cnp_sum_v2"))
            for axis in (None, 0, 1, 2, -1, -2, -3):
                with self.subTest(axis=axis):
                    result = stack.enter_context(runtime.sum(transposed, axis))
                    assert_array_equivalent(
                        self,
                        result,
                        np.sum(logical_value, axis=axis),
                        compare_contiguity=False,
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_prod_distinguishes_none_from_every_negative_axis(self) -> None:
        source_value = np.arange(1, 25, dtype=np.float64).reshape(2, 3, 4)
        logical_value = source_value.transpose(2, 0, 1)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            transposed = stack.enter_context(runtime.transpose(source, (2, 0, 1)))
            self.assertTrue(hasattr(runtime.dll, "cnp_prod_v2"))
            for axis in (None, 0, 1, 2, -1, -2, -3):
                with self.subTest(axis=axis):
                    result = stack.enter_context(runtime.prod(transposed, axis))
                    assert_array_equivalent(
                        self,
                        result,
                        np.prod(logical_value, axis=axis),
                        compare_contiguity=False,
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_sum_uses_numpy_integer_accumulator_promotion_exactly(self) -> None:
        source_values = (
            np.array([True, True, False, True], dtype=np.bool_),
            np.array([120, 7, -3], dtype=np.int8),
            np.array([250, 7, 3], dtype=np.uint8),
            np.array([2**53 + 1, 2**53 + 3, -5], dtype=np.int64),
            np.array([2**63 + 1, 2**63 + 3, 5], dtype=np.uint64),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for source_value in source_values:
                with self.subTest(dtype=str(source_value.dtype)):
                    source = stack.enter_context(runtime.from_numpy(source_value))
                    result = stack.enter_context(runtime.sum(source))
                    assert_array_equivalent(
                        self,
                        result,
                        np.sum(source_value),
                        compare_contiguity=False,
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_prod_uses_numpy_integer_accumulator_promotion_exactly(self) -> None:
        source_values = (
            np.array([True, True, False], dtype=np.bool_),
            np.array([3, 4, -2], dtype=np.int8),
            np.array([3, 4, 2], dtype=np.uint8),
            np.array([2**53 + 1, 1], dtype=np.int64),
            np.array([2**63 + 1, 1], dtype=np.uint64),
            np.array([np.iinfo(np.int64).max, 2], dtype=np.int64),
            np.array([np.iinfo(np.uint64).max, 2], dtype=np.uint64),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for case_index, source_value in enumerate(source_values):
                with self.subTest(
                    case=case_index, dtype=str(source_value.dtype)
                ):
                    source = stack.enter_context(runtime.from_numpy(source_value))
                    result = stack.enter_context(runtime.prod(source))
                    assert_array_equivalent(
                        self,
                        result,
                        np.prod(source_value),
                        compare_contiguity=False,
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_cumsum_distinguishes_none_from_every_negative_axis(self) -> None:
        source_value = np.arange(1, 25, dtype=np.float64).reshape(2, 3, 4)
        logical_value = source_value.transpose(2, 0, 1)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            transposed = stack.enter_context(runtime.transpose(source, (2, 0, 1)))
            self.assertTrue(hasattr(runtime.dll, "cnp_cumsum_v2"))
            for axis in (None, 0, 1, 2, -1, -2, -3):
                with self.subTest(axis=axis):
                    result = stack.enter_context(runtime.cumsum(transposed, axis))
                    assert_array_equivalent(
                        self,
                        result,
                        np.cumsum(logical_value, axis=axis),
                        compare_contiguity=False,
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_float64_cumsum_preserves_numpy_left_to_right_rounding(self) -> None:
        source_value = np.array(
            [1.0e16, 0.0, -1.0e16, 1.0], dtype=np.float64
        )
        expected = np.cumsum(source_value)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            output = stack.enter_context(
                runtime.from_numpy(np.empty_like(source_value))
            )
            results = (
                stack.enter_context(runtime.cumsum(source)),
                stack.enter_context(
                    self.legacy_reduction(
                        runtime,
                        source,
                        "cnp_cumsum",
                        (CNP_AXIS_NONE, CNP_NOTYPE),
                    )
                ),
            )

            cumsum_into = runtime.dll.cnp_cumsum_into
            cumsum_into.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            cumsum_into.restype = ctypes.c_int
            runtime.dll.cnp_clear_error()
            self.assertEqual(
                0,
                cumsum_into(
                    source.pointer, CNP_AXIS_NONE, output.pointer
                ),
            )

            for result in (*results, output):
                assert_array_equivalent(
                    self, result, expected, compare_contiguity=False
                )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_sum_into_scalar_matches_numpy_pairwise_tree_and_errors_atomically(
        self,
    ) -> None:
        sensitive = np.array(
            [1e16, 1.0, -1e16, 1.0], dtype=np.float64
        )
        source_values = (
            sensitive,
            np.pad(sensitive, (0, 4)),
            np.pad(sensitive, (0, 123)),
            np.pad(sensitive, (0, 124)),
            np.pad(sensitive, (0, 125)),
            np.array([-0.0, -0.0], dtype=np.float64),
            np.array([1.0, np.nan, 2.0], dtype=np.float64),
            np.array([np.inf, 1.0], dtype=np.float64),
            np.array([np.inf, -np.inf], dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            function = runtime.dll.cnp_sum_into_scalar
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_double),
            ]
            function.restype = ctypes.c_int
            sources = tuple(
                stack.enter_context(runtime.from_numpy(value))
                for value in source_values
            )
            active = runtime.retained_bytes
            with np.errstate(invalid="ignore"):
                for repeat in range(3):
                    for source_value, source in zip(source_values, sources):
                        with self.subTest(
                            repeat=repeat, length=source_value.size
                        ):
                            output = ctypes.c_double(123.5)
                            runtime.dll.cnp_clear_error()
                            self.assertEqual(
                                0, function(source.pointer, ctypes.byref(output))
                            )
                            expected = np.asarray(
                                np.sum(source_value), dtype=np.float64
                            )
                            actual = np.asarray(output.value, dtype=np.float64)
                            if np.isnan(expected):
                                self.assertTrue(np.isnan(actual))
                            else:
                                self.assertEqual(
                                    expected.tobytes(), actual.tobytes()
                                )
                            self.assertEqual(0, runtime.error_state().status)
                            self.assertEqual(active, runtime.retained_bytes)

                    null_source_output = ctypes.c_double(123.5)
                    runtime.dll.cnp_clear_error()
                    self.assertEqual(
                        -1, function(None, ctypes.byref(null_source_output))
                    )
                    error = runtime.error_state()
                    self.assertEqual(123.5, null_source_output.value)
                    self.assertEqual(-1, error.status)
                    self.assertEqual("cnp_sum_into_scalar", error.function)
                    self.assertEqual("source array is null", error.message)
                    self.assertEqual(active, runtime.retained_bytes)

                    runtime.dll.cnp_clear_error()
                    self.assertEqual(-1, function(sources[0].pointer, None))
                    error = runtime.error_state()
                    self.assertEqual(-1, error.status)
                    self.assertEqual("cnp_sum_into_scalar", error.function)
                    self.assertEqual(
                        "Output scalar pointer is null", error.message
                    )
                    self.assertEqual(active, runtime.retained_bytes)

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def _assert_scalar_pairwise_tree(
        self,
        function_name: str,
        numpy_function,
    ) -> None:
        sensitive = np.array(
            [1e16, 1.0, -1e16, 1.0], dtype=np.float64
        )
        source_values = (
            sensitive,
            np.pad(sensitive, (0, 4)),
            np.pad(sensitive, (0, 123)),
            np.pad(sensitive, (0, 124)),
            np.pad(sensitive, (0, 125)),
            np.array([-0.0, -0.0], dtype=np.float64),
            np.array([1.0, np.nan, 2.0], dtype=np.float64),
            np.array([np.inf, 1.0], dtype=np.float64),
            np.array([np.inf, -np.inf], dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            function = getattr(runtime.dll, function_name)
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_double
            sources = tuple(
                stack.enter_context(runtime.from_numpy(value))
                for value in source_values
            )
            active = runtime.retained_bytes
            with warnings.catch_warnings(), np.errstate(invalid="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                for repeat in range(3):
                    for source_value, source in zip(source_values, sources):
                        with self.subTest(
                            function=function_name,
                            repeat=repeat,
                            length=source_value.size,
                        ):
                            runtime.dll.cnp_clear_error()
                            actual = np.asarray(
                                function(source.pointer), dtype=np.float64
                            )
                            expected = np.asarray(
                                numpy_function(source_value), dtype=np.float64
                            )
                            if np.isnan(expected):
                                self.assertTrue(np.isnan(actual))
                            else:
                                self.assertEqual(
                                    expected.tobytes(), actual.tobytes()
                                )
                            self.assertEqual(0, runtime.error_state().status)
                            self.assertEqual(active, runtime.retained_bytes)

                    runtime.dll.cnp_clear_error()
                    self.assertTrue(np.isnan(function(None)))
                    error = runtime.error_state()
                    self.assertEqual(-1, error.status)
                    self.assertEqual(function_name, error.function)
                    self.assertEqual(
                        "source array must not be null", error.message
                    )
                    self.assertEqual(active, runtime.retained_bytes)

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_sum_scalar_matches_numpy_pairwise_tree_boundaries(self) -> None:
        self._assert_scalar_pairwise_tree("cnp_sum_scalar", np.sum)

    def test_mean_scalar_matches_numpy_pairwise_tree_boundaries(self) -> None:
        self._assert_scalar_pairwise_tree("cnp_mean_scalar", np.mean)

    def test_ahk_scalar_reductions_preserve_shared_numpy_semantics(self) -> None:
        nan_value = np.array([1.0, np.nan], dtype=np.float64)
        variance_value = np.array(
            [
                float.fromhex("0x1.ffffffffffff0p+49"),
                float.fromhex("0x1.ffffffffffff0p+49"),
                float.fromhex("0x1.ffffffffffff2p+49"),
                float.fromhex("0x1.ffffffffffff2p+49"),
            ],
            dtype=np.float64,
        )
        cases = (
            (nan_value, (("cnp_ahk_max", np.max), ("cnp_ahk_min", np.min))),
            (
                variance_value,
                (("cnp_ahk_var", np.var), ("cnp_ahk_std", np.std)),
            ),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for source_value, operations in cases:
                source = stack.enter_context(runtime.from_numpy(source_value))
                for function_name, numpy_function in operations:
                    with self.subTest(function=function_name):
                        function = getattr(runtime.dll, function_name)
                        function.argtypes = [ctypes.c_void_p, ctypes.c_int]
                        function.restype = ctypes.c_double
                        runtime.dll.cnp_clear_error()
                        actual = function(source.pointer, CNP_AXIS_NONE)
                        np.testing.assert_array_equal(
                            np.asarray(actual),
                            np.asarray(numpy_function(source_value)),
                            strict=True,
                        )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_ahk_reduction_v2_macro_families_report_exact_null_errors(
        self,
    ) -> None:
        cases = (
            (
                "cnp_ahk_max_v2",
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
                (None, 0, 1),
            ),
            (
                "cnp_ahk_sum_v2",
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
                (None, 0, 1),
            ),
            (
                "cnp_ahk_var_v2",
                [
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                ],
                (None, 0, 1, 0),
            ),
            (
                "cnp_ahk_percentile_v2",
                [
                    ctypes.c_void_p,
                    ctypes.c_double,
                    ctypes.c_int,
                    ctypes.c_int,
                ],
                (None, 50.0, 0, 1),
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for repeat in range(3):
                for function_name, argument_types, arguments in cases:
                    with self.subTest(
                        repeat=repeat, function=function_name
                    ):
                        function = getattr(runtime.dll, function_name)
                        function.argtypes = argument_types
                        function.restype = ctypes.c_void_p
                        runtime.dll.cnp_clear_error()
                        self.assertIsNone(function(*arguments))
                        error = runtime.error_state()
                        self.assertEqual(-1, error.status)
                        self.assertEqual(function_name, error.function)
                        self.assertEqual(
                            "source array must not be null", error.message
                        )
                        self.assertEqual(baseline, runtime.retained_bytes)

    def test_ahk_reduction_v2_relabels_core_failures(self) -> None:
        with CnumpyRuntime(DLL) as runtime, runtime.from_numpy(
            np.empty((0,), dtype=np.float64)
        ) as source:
            active = runtime.retained_bytes
            function = runtime.dll.cnp_ahk_max_v2
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
            ]
            function.restype = ctypes.c_void_p
            runtime.dll.cnp_clear_error()
            self.assertIsNone(function(source.pointer, 0, 1))
            error = runtime.error_state()
            self.assertEqual(-1, error.status)
            self.assertEqual("cnp_ahk_max_v2", error.function)
            self.assertEqual(
                "zero-size reduction has no identity", error.message
            )
            self.assertEqual(active, runtime.retained_bytes)

    def test_scalar_reduction_helpers_reject_null_sources_explicitly(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for function_name in (
                "cnp_sum_scalar",
                "cnp_prod_scalar",
                "cnp_mean_scalar",
            ):
                with self.subTest(function=function_name):
                    function = getattr(runtime.dll, function_name)
                    function.argtypes = [ctypes.c_void_p]
                    function.restype = ctypes.c_double
                    runtime.dll.cnp_clear_error()

                    actual = function(None)
                    error = runtime.error_state()

                    self.assertTrue(np.isnan(actual))
                    self.assertEqual(-1, error.status)
                    self.assertEqual(function_name, error.function)
                    self.assertEqual(
                        "source array must not be null", error.message
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_scalar_reduction_helpers_reject_malformed_metadata(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(
                runtime.from_numpy(np.empty(0, dtype=np.float64))
            )
            header = ctypes.cast(
                source.pointer, ctypes.POINTER(_CnpArrayHeader)
            ).contents
            cases = (
                ("dtype", None, -3, "numeric dtype"),
                ("shape", None, -4, "shape metadata"),
                ("strides", None, -4, "shape metadata"),
                ("ndim", 65, -4, "shape metadata"),
                ("size", -1, -4, "shape metadata"),
            )
            for function_name in (
                "cnp_sum_scalar",
                "cnp_prod_scalar",
                "cnp_mean_scalar",
            ):
                function = getattr(runtime.dll, function_name)
                function.argtypes = [ctypes.c_void_p]
                function.restype = ctypes.c_double
                for field, invalid, status, message in cases:
                    with self.subTest(function=function_name, field=field):
                        original = getattr(header, field)
                        setattr(header, field, invalid)
                        try:
                            runtime.dll.cnp_clear_error()
                            actual = function(source.pointer)
                        finally:
                            setattr(header, field, original)
                        error = runtime.error_state()
                        self.assertTrue(np.isnan(actual))
                        self.assertEqual(status, error.status)
                        self.assertEqual(function_name, error.function)
                        self.assertIn(message, error.message)
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_scalar_reduction_helpers_reject_unrepresentable_complex_results(
        self,
    ) -> None:
        source_value = np.array([1.0 + 2.0j, 3.0 - 4.0j], dtype=np.complex128)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            for function_name in (
                "cnp_sum_scalar",
                "cnp_prod_scalar",
                "cnp_mean_scalar",
            ):
                with self.subTest(function=function_name):
                    function = getattr(runtime.dll, function_name)
                    function.argtypes = [ctypes.c_void_p]
                    function.restype = ctypes.c_double
                    runtime.dll.cnp_clear_error()

                    actual = function(source.pointer)
                    error = runtime.error_state()

                    self.assertTrue(np.isnan(actual))
                    self.assertEqual(-3, error.status)
                    self.assertEqual(function_name, error.function)
                    self.assertIn("complex", error.message)
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_scalar_reduction_helpers_match_representable_numpy_values(
        self,
    ) -> None:
        source_value = np.array(
            [[1.5, -2.0, 4.0], [3.0, 0.5, -1.0]], dtype=np.float64
        )
        logical_value = source_value.T
        operations = (
            ("cnp_sum_scalar", np.sum),
            ("cnp_prod_scalar", np.prod),
            ("cnp_mean_scalar", np.mean),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            storage = stack.enter_context(runtime.from_numpy(source_value))
            source = stack.enter_context(runtime.transpose(storage, (1, 0)))
            empty = stack.enter_context(
                runtime.from_numpy(np.empty(0, dtype=np.float64))
            )
            for function_name, numpy_function in operations:
                function = getattr(runtime.dll, function_name)
                function.argtypes = [ctypes.c_void_p]
                function.restype = ctypes.c_double
                for label, operand, expected_source in (
                    ("transposed", source, logical_value),
                    ("empty", empty, np.empty(0, dtype=np.float64)),
                ):
                    with self.subTest(function=function_name, layout=label):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            expected = float(numpy_function(expected_source))
                        runtime.dll.cnp_clear_error()
                        actual = function(operand.pointer)
                        if np.isnan(expected):
                            self.assertTrue(np.isnan(actual))
                        else:
                            self.assertEqual(expected, actual)
                        self.assertEqual(0, runtime.error_state().status)
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_scalar_convenience_exports_cover_rank_zero_through_four_layouts_values_errors_and_lifetimes(
        self,
    ) -> None:
        shapes = ((), (5,), (2, 3), (2, 2, 3), (2, 2, 2, 3))
        operations = (
            ("cnp_sum_scalar", np.sum),
            ("cnp_prod_scalar", None),
            ("cnp_mean_scalar", np.mean),
        )

        def sequential_product(value: np.ndarray) -> np.float64:
            result = np.float64(1.0)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                for item in np.ravel(value, order="C"):
                    result = np.float64(result * np.float64(item))
            return result

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            functions = {}
            for function_name, _ in operations:
                function = getattr(runtime.dll, function_name)
                function.argtypes = [ctypes.c_void_p]
                function.restype = ctypes.c_double
                functions[function_name] = function

            def assert_scalar_results(source, value, label: str) -> None:
                active = runtime.retained_bytes
                with warnings.catch_warnings(), np.errstate(
                    over="ignore", under="ignore", invalid="ignore"
                ):
                    warnings.simplefilter("ignore", RuntimeWarning)
                    for function_name, numpy_function in operations:
                        with self.subTest(
                            layout=label,
                            rank=value.ndim,
                            function=function_name,
                        ):
                            expected = np.asarray(
                                sequential_product(value)
                                if numpy_function is None else
                                numpy_function(value),
                                dtype=np.float64,
                            )
                            runtime.dll.cnp_clear_error()
                            actual = np.asarray(
                                functions[function_name](source.pointer),
                                dtype=np.float64,
                            )
                            if np.isnan(expected):
                                self.assertTrue(np.isnan(actual))
                            else:
                                self.assertEqual(
                                    expected.tobytes(), actual.tobytes()
                                )
                            self.assertEqual(0, runtime.error_state().status)
                            self.assertEqual(
                                active, runtime.retained_bytes
                            )

            for shape in shapes:
                rank = len(shape)
                size = int(np.prod(shape)) if shape else 1
                source_value = (
                    np.array(-2.5, dtype=np.float64)
                    if rank == 0 else
                    (
                        (np.arange(size, dtype=np.float64) % 7.0)
                        - 3.0
                    ).reshape(shape) / 2.0
                )
                with runtime.from_numpy(source_value) as source:
                    self.assertTrue(source.c_contiguous)
                    assert_scalar_results(
                        source, source_value, "contiguous"
                    )

                if rank == 1:
                    strided_storage_value = np.empty(
                        source_value.size * 2, dtype=np.float64
                    )
                    strided_storage_value[::2] = source_value
                    strided_storage_value[1::2] = 99.0
                    with runtime.from_numpy(
                        strided_storage_value
                    ) as storage, self.slice_view(
                        runtime,
                        storage,
                        (_CnpSlice(0, 0, 2, False, False, True),),
                    ) as source:
                        self.assertFalse(source.c_contiguous)
                        assert_scalar_results(
                            source, source_value, "strided"
                        )
                elif rank > 1:
                    reversed_axes = tuple(range(rank - 1, -1, -1))
                    storage_value = np.transpose(
                        source_value, reversed_axes
                    ).copy()
                    with runtime.from_numpy(
                        storage_value
                    ) as storage, runtime.transpose(
                        storage, reversed_axes
                    ) as source:
                        self.assertFalse(source.c_contiguous)
                        assert_scalar_results(
                            source, source_value, "transposed"
                        )

            empty_shapes = (
                (0,),
                (2, 0),
                (2, 0, 3),
                (2, 2, 0, 3),
            )
            for shape in empty_shapes:
                value = np.empty(shape, dtype=np.float64)
                with runtime.from_numpy(value) as source:
                    assert_scalar_results(source, value, "empty")

            special_values = (
                np.array([-0.0, -0.0], dtype=np.float64),
                np.array([1.0, np.nan, 2.0], dtype=np.float64),
                np.array([np.inf, -np.inf], dtype=np.float64),
                np.array([np.inf, 0.0], dtype=np.float64),
                np.array(
                    [np.finfo(np.float64).tiny, 0.5, 2.0],
                    dtype=np.float64,
                ),
            )
            for case_index, value in enumerate(special_values):
                with runtime.from_numpy(value) as source:
                    assert_scalar_results(
                        source, value, f"special-{case_index}"
                    )

            sum_sensitive = np.array(
                [1e16, 1.0, -1e16, 1.0], dtype=np.float64
            )
            product_sensitive = (
                np.array(
                    [1e308, 1e308, 1e-308, 1e-308],
                    dtype=np.float64,
                ),
                np.array(
                    [1e308, 1e-308, 1e-308, 1e308],
                    dtype=np.float64,
                ),
            )
            order_sensitive_values = (
                tuple(
                    np.pad(sum_sensitive, (0, length - 4))
                    for length in (4, 8, 127, 128, 129)
                )
                + product_sensitive
            )
            for case_index, value in enumerate(order_sensitive_values):
                with runtime.from_numpy(value) as source:
                    assert_scalar_results(
                        source, value, f"order-{case_index}"
                    )

            complex_value = np.array(
                [1.0 + 2.0j, 3.0 - 4.0j], dtype=np.complex128
            )
            with runtime.from_numpy(complex_value) as complex_source:
                active = runtime.retained_bytes
                for function_name, _ in operations:
                    for label, pointer, status, message in (
                        (
                            "null",
                            None,
                            -1,
                            "source array must not be null",
                        ),
                        (
                            "complex",
                            complex_source.pointer,
                            -3,
                            "double-return scalar ABI cannot represent "
                            "complex results",
                        ),
                    ):
                        with self.subTest(
                            function=function_name, error=label
                        ):
                            runtime.dll.cnp_clear_error()
                            actual = functions[function_name](pointer)
                            error = runtime.error_state()
                            self.assertTrue(np.isnan(actual))
                            self.assertEqual(status, error.status)
                            self.assertEqual(
                                function_name, error.function
                            )
                            self.assertEqual(message, error.message)
                            self.assertEqual(
                                active, runtime.retained_bytes
                            )

            malformed_value = np.array([1.0], dtype=np.float64)
            with runtime.from_numpy(malformed_value) as malformed:
                active = runtime.retained_bytes
                header = ctypes.cast(
                    malformed.pointer, ctypes.POINTER(_CnpArrayHeader)
                ).contents
                for function_name, _ in operations:
                    function = functions[function_name]
                    for field, invalid, status, message in (
                        (
                            "size",
                            -1,
                            -4,
                            "source array has invalid shape metadata",
                        ),
                        (
                            "data",
                            None,
                            -1,
                            "source array requires a data buffer",
                        ),
                    ):
                        with self.subTest(
                            function=function_name, error=field
                        ):
                            original = getattr(header, field)
                            setattr(header, field, invalid)
                            try:
                                runtime.dll.cnp_clear_error()
                                actual = function(malformed.pointer)
                            finally:
                                setattr(header, field, original)
                            error = runtime.error_state()
                            self.assertTrue(np.isnan(actual))
                            self.assertEqual(status, error.status)
                            self.assertEqual(
                                function_name, error.function
                            )
                            self.assertEqual(message, error.message)
                            self.assertEqual(
                                active, runtime.retained_bytes
                            )

            self.assertEqual(baseline, runtime.retained_bytes)

    def test_cumsum_uses_numpy_integer_accumulators_for_every_element(self) -> None:
        source_values = (
            np.array([True, True, False, True], dtype=np.bool_),
            np.array([120, 7, -3], dtype=np.int8),
            np.array([250, 7, 3], dtype=np.uint8),
            np.array([2**53 + 1, 2, -3], dtype=np.int64),
            np.array([2**63 + 1, 2, 3], dtype=np.uint64),
            np.array([np.iinfo(np.int64).max, 1, 1], dtype=np.int64),
            np.array([np.iinfo(np.uint64).max, 1, 1], dtype=np.uint64),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for case_index, source_value in enumerate(source_values):
                with self.subTest(
                    case=case_index, dtype=str(source_value.dtype)
                ):
                    source = stack.enter_context(runtime.from_numpy(source_value))
                    result = stack.enter_context(runtime.cumsum(source))
                    assert_array_equivalent(
                        self,
                        result,
                        np.cumsum(source_value),
                        compare_contiguity=False,
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_cumprod_distinguishes_none_from_every_negative_axis(self) -> None:
        source_value = (
            np.arange(24, dtype=np.float64).reshape(2, 3, 4) / 24.0 + 1.0
        )
        logical_value = source_value.transpose(2, 0, 1)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            transposed = stack.enter_context(runtime.transpose(source, (2, 0, 1)))
            self.assertTrue(hasattr(runtime.dll, "cnp_cumprod_v2"))
            for axis in (None, 0, 1, 2, -1, -2, -3):
                with self.subTest(axis=axis):
                    result = stack.enter_context(runtime.cumprod(transposed, axis))
                    assert_array_equivalent(
                        self,
                        result,
                        np.cumprod(logical_value, axis=axis),
                        compare_contiguity=False,
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_float32_cumulative_operations_preserve_explicit_axis(self) -> None:
        source_value = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            for method_name, numpy_function in (
                ("cumsum", np.cumsum),
                ("cumprod", np.cumprod),
            ):
                with self.subTest(operation=method_name):
                    result = stack.enter_context(
                        getattr(runtime, method_name)(source, 0)
                    )
                    assert_array_equivalent(
                        self,
                        result,
                        numpy_function(source_value, axis=0),
                        compare_contiguity=False,
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_cumprod_uses_numpy_integer_accumulators_for_every_element(self) -> None:
        source_values = (
            np.array([True, True, False, True], dtype=np.bool_),
            np.array([3, 4, -2], dtype=np.int8),
            np.array([3, 4, 2], dtype=np.uint8),
            np.array([2**53 + 1, 1, -1], dtype=np.int64),
            np.array([2**63 + 1, 1, 3], dtype=np.uint64),
            np.array([np.iinfo(np.int64).max, 2, 2], dtype=np.int64),
            np.array([np.iinfo(np.uint64).max, 2, 2], dtype=np.uint64),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for case_index, source_value in enumerate(source_values):
                with self.subTest(
                    case=case_index, dtype=str(source_value.dtype)
                ):
                    source = stack.enter_context(runtime.from_numpy(source_value))
                    result = stack.enter_context(runtime.cumprod(source))
                    assert_array_equivalent(
                        self,
                        result,
                        np.cumprod(source_value),
                        compare_contiguity=False,
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_mean_distinguishes_none_from_every_negative_axis(self) -> None:
        source_value = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        logical_value = source_value.transpose(2, 0, 1)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            transposed = stack.enter_context(runtime.transpose(source, (2, 0, 1)))
            self.assertTrue(hasattr(runtime.dll, "cnp_mean_v2"))
            for axis in (None, 0, 1, 2, -1, -2, -3):
                with self.subTest(axis=axis):
                    result = stack.enter_context(runtime.mean(transposed, axis))
                    assert_array_equivalent(
                        self,
                        result,
                        np.mean(logical_value, axis=axis),
                        compare_contiguity=False,
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_mean_uses_numpy_default_dtype_for_real_inputs(self) -> None:
        source_values = (
            np.array([True, False, True], dtype=np.bool_),
            np.array([1, 2, 7], dtype=np.int32),
            np.array([1.25, 2.5, 7.0], dtype=np.float32),
            np.array([1.25, 2.5, 7.0], dtype=np.float64),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for source_value in source_values:
                with self.subTest(dtype=str(source_value.dtype)):
                    source = stack.enter_context(runtime.from_numpy(source_value))
                    result = stack.enter_context(runtime.mean(source))
                    expected = np.mean(source_value)
                    assert_array_equivalent(
                        self,
                        result,
                        expected,
                        compare_contiguity=False,
                        rtol=4 * np.finfo(expected.dtype).eps,
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_average_with_same_shape_weights_reduces_the_requested_axis(
        self,
    ) -> None:
        source_value = np.array(
            [[1.0, 4.0, 9.0], [2.0, 8.0, 18.0]], dtype=np.float64
        )
        weights_value = np.array(
            [[1.0, 2.0, 1.0], [3.0, 1.0, 2.0]], dtype=np.float64
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            weights = stack.enter_context(runtime.from_numpy(weights_value))
            result = stack.enter_context(runtime.average(source, 1, weights))
            assert_array_equivalent(
                self,
                result,
                np.average(source_value, axis=1, weights=weights_value),
                compare_contiguity=False,
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_average_accepts_one_dimensional_weights_for_the_axis(self) -> None:
        source_value = np.array(
            [[1.0, 4.0, 9.0], [2.0, 8.0, 18.0]], dtype=np.float64
        )
        weights_value = np.array([1.0, 2.0, 1.0], dtype=np.float64)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            weights = stack.enter_context(runtime.from_numpy(weights_value))
            result = stack.enter_context(runtime.average(source, 1, weights))
            assert_array_equivalent(
                self,
                result,
                np.average(source_value, axis=1, weights=weights_value),
                compare_contiguity=False,
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_average_distinguishes_none_from_the_negative_last_axis(self) -> None:
        source_value = np.arange(1, 25, dtype=np.float64).reshape(2, 3, 4)
        full_weights_value = np.arange(1, 25, dtype=np.float64).reshape(2, 3, 4)
        axis_weights_value = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            full_weights = stack.enter_context(
                runtime.from_numpy(full_weights_value)
            )
            axis_weights = stack.enter_context(
                runtime.from_numpy(axis_weights_value)
            )
            for axis, weights, expected_weights in (
                (None, full_weights, full_weights_value),
                (-1, axis_weights, axis_weights_value),
            ):
                with self.subTest(axis=axis):
                    result = stack.enter_context(
                        runtime.average(source, axis, weights)
                    )
                    assert_array_equivalent(
                        self,
                        result,
                        np.average(
                            source_value, axis=axis, weights=expected_weights
                        ),
                        compare_contiguity=False,
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_average_v2_covers_rank_zero_through_four_axes_weights_errors_and_lifetimes(
        self,
    ) -> None:
        shapes = ((), (3,), (2, 3), (2, 2, 3), (2, 2, 2, 3))
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            function = runtime.dll.cnp_average_v2
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_bool,
                ctypes.c_void_p,
            ]
            function.restype = ctypes.c_void_p

            for shape in shapes:
                rank = len(shape)
                size = int(np.prod(shape)) if shape else 1
                source_value = (
                    np.array(3.0, dtype=np.float64)
                    if rank == 0 else
                    (np.arange(size, dtype=np.float64) + 1.0).reshape(
                        shape
                    )
                )
                full_weights_value = (
                    np.array(2.0, dtype=np.float64)
                    if rank == 0 else
                    (np.arange(size, dtype=np.float64) + 2.0).reshape(
                        shape
                    )
                )
                axes = (
                    (None, 0, -1, 1, -2)
                    if rank == 0 else
                    (
                        (None,)
                        + tuple(range(rank))
                        + tuple(range(-1, -rank - 1, -1))
                        + (rank, -rank - 1)
                    )
                )

                with ExitStack() as stack:
                    source = stack.enter_context(
                        runtime.from_numpy(source_value)
                    )
                    full_weights = stack.enter_context(
                        runtime.from_numpy(full_weights_value)
                    )
                    axis_weight_values = tuple(
                        np.arange(extent, dtype=np.float64) + 1.0
                        for extent in shape
                    )
                    axis_weights = tuple(
                        stack.enter_context(runtime.from_numpy(value))
                        for value in axis_weight_values
                    )

                    for axis in axes:
                        variants = [
                            ("unweighted", None, None),
                            (
                                "full_shape",
                                full_weights,
                                full_weights_value,
                            ),
                        ]
                        valid_explicit_axis = (
                            axis is not None
                            and rank > 0
                            and -rank <= axis < rank
                        )
                        if valid_explicit_axis:
                            normalized_axis = (
                                axis if axis >= 0 else axis + rank
                            )
                            variants.append(
                                (
                                    "axis_weights",
                                    axis_weights[normalized_axis],
                                    axis_weight_values[normalized_axis],
                                )
                            )

                        for label, weights, expected_weights in variants:
                            with self.subTest(
                                rank=rank, axis=axis, weights=label
                            ):
                                active = runtime.retained_bytes
                                runtime.dll.cnp_clear_error()
                                pointer = function(
                                    source.pointer,
                                    0 if axis is None else axis,
                                    axis is None,
                                    None
                                    if weights is None else
                                    weights.pointer,
                                )
                                invalid_axis = (
                                    axis is not None
                                    and (
                                        (
                                            rank == 0
                                            and (
                                                weights is None
                                                or axis not in (0, -1)
                                            )
                                        )
                                        or (
                                            rank > 0
                                            and (
                                                axis < -rank
                                                or axis >= rank
                                            )
                                        )
                                    )
                                )
                                if invalid_axis:
                                    self.assertIsNone(pointer)
                                    error = runtime.error_state()
                                    self.assertEqual(-5, error.status)
                                    self.assertEqual(
                                        "cnp_average_v2", error.function
                                    )
                                    self.assertEqual(
                                        f"axis {axis} is out of bounds for "
                                        f"array of dimension {rank}",
                                        error.message,
                                    )
                                else:
                                    with runtime._owned_result(
                                        pointer, "cnp_average_v2"
                                    ) as result:
                                        expected = np.average(
                                            source_value,
                                            axis=axis,
                                            weights=expected_weights,
                                        )
                                        assert_array_equivalent(
                                            self,
                                            result,
                                            expected,
                                            compare_contiguity=False,
                                        )
                                        self.assertEqual(
                                            0,
                                            runtime.error_state().status,
                                        )
                                self.assertEqual(
                                    active, runtime.retained_bytes
                                )
                self.assertEqual(baseline, runtime.retained_bytes)

            source_value = np.arange(
                1, 7, dtype=np.float64
            ).reshape(2, 3)
            with ExitStack() as stack:
                source = stack.enter_context(
                    runtime.from_numpy(source_value)
                )
                short_weights = stack.enter_context(
                    runtime.from_numpy(
                        np.array([1.0, 2.0], dtype=np.float64)
                    )
                )
                zero_weights = stack.enter_context(
                    runtime.from_numpy(np.zeros(3, dtype=np.float64))
                )
                for label, axis, axis_none, weights, status, message in (
                    (
                        "none_shape",
                        0,
                        True,
                        short_weights,
                        -4,
                        "weights must have the same shape as the source "
                        "array",
                    ),
                    (
                        "axis_shape",
                        1,
                        False,
                        short_weights,
                        -4,
                        "weights must match the source shape or selected "
                        "axis",
                    ),
                    (
                        "zero_sum",
                        1,
                        False,
                        zero_weights,
                        -1,
                        "weights sum to zero, cannot normalize",
                    ),
                ):
                    with self.subTest(error=label):
                        active = runtime.retained_bytes
                        runtime.dll.cnp_clear_error()
                        self.assertIsNone(
                            function(
                                source.pointer,
                                axis,
                                axis_none,
                                weights.pointer,
                            )
                        )
                        error = runtime.error_state()
                        self.assertEqual(status, error.status)
                        self.assertEqual(
                            "cnp_average_v2", error.function
                        )
                        self.assertEqual(message, error.message)
                        self.assertEqual(active, runtime.retained_bytes)

            runtime.dll.cnp_clear_error()
            self.assertIsNone(function(None, 0, True, None))
            error = runtime.error_state()
            self.assertEqual(-1, error.status)
            self.assertEqual("cnp_average_v2", error.function)
            self.assertEqual("source array is required", error.message)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_average_preserves_float32_result_dtype(self) -> None:
        source_value = np.array(
            [[0.25, 1.5, 9.0], [2.0, 7.25, 18.5]], dtype=np.float32
        )
        weights_value = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            weights = stack.enter_context(runtime.from_numpy(weights_value))
            result = stack.enter_context(runtime.average(source, 1, weights))
            assert_array_equivalent(
                self,
                result,
                np.average(source_value, axis=1, weights=weights_value),
                compare_contiguity=False,
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_average_preserves_float16_values_and_dtype(self) -> None:
        source_value = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        weights_value = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            weights = stack.enter_context(runtime.from_numpy(weights_value))
            result = stack.enter_context(runtime.average(source, None, weights))
            assert_array_equivalent(
                self,
                result,
                np.average(source_value, weights=weights_value),
                compare_contiguity=False,
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_average_preserves_complex64_values_and_dtype(self) -> None:
        source_value = np.array(
            [[1.0 + 2.0j, 4.0 - 3.0j, 9.0 + 0.5j]],
            dtype=np.complex64,
        )
        weights_value = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            weights = stack.enter_context(runtime.from_numpy(weights_value))
            result = stack.enter_context(runtime.average(source, 1, weights))
            assert_array_equivalent(
                self,
                result,
                np.average(source_value, axis=1, weights=weights_value),
                compare_contiguity=False,
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_average_promotes_complex_inputs_to_complex128_when_required(
        self,
    ) -> None:
        source_value = np.array(
            [[1.0 + 2.0j, 4.0 - 3.0j, 9.0 + 0.5j]],
            dtype=np.complex128,
        )
        weights_value = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            weights = stack.enter_context(runtime.from_numpy(weights_value))
            result = stack.enter_context(runtime.average(source, 1, weights))
            assert_array_equivalent(
                self,
                result,
                np.average(source_value, axis=1, weights=weights_value),
                compare_contiguity=False,
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_average_uses_numpy_result_dtype_for_real_input_pairs(self) -> None:
        cases = (
            (
                np.arange(6, dtype=np.int32).reshape(2, 3),
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
            ),
            (
                np.arange(6, dtype=np.float32).reshape(2, 3),
                np.array([1.0, 2.0, 3.0], dtype=np.float64),
            ),
            (
                np.array([[True, False, True], [False, True, True]]),
                np.array([1, 2, 3], dtype=np.int8),
            ),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for source_value, weights_value in cases:
                with self.subTest(
                    source_dtype=str(source_value.dtype),
                    weights_dtype=str(weights_value.dtype),
                ):
                    source = stack.enter_context(
                        runtime.from_numpy(source_value)
                    )
                    weights = stack.enter_context(
                        runtime.from_numpy(weights_value)
                    )
                    result = stack.enter_context(
                        runtime.average(source, 1, weights)
                    )
                    assert_array_equivalent(
                        self,
                        result,
                        np.average(
                            source_value, axis=1, weights=weights_value
                        ),
                        compare_contiguity=False,
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_average_without_weights_matches_numpy_mean_projection(self) -> None:
        source_value = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            for axis in (None, 0, -1, -2):
                with self.subTest(axis=axis):
                    result = stack.enter_context(runtime.average(source, axis))
                    assert_array_equivalent(
                        self,
                        result,
                        np.average(source_value, axis=axis),
                        compare_contiguity=False,
                        rtol=4 * np.finfo(np.float32).eps,
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_average_matches_numpy_pairwise_weighted_sum_order(self) -> None:
        rng = np.random.default_rng(123)
        source_value = (
            rng.standard_normal(16) * np.logspace(-3, 3, 16)
        ).astype(np.float32)
        weights_value = (rng.random(16) + 0.01).astype(np.float32)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            weights = stack.enter_context(runtime.from_numpy(weights_value))
            result = stack.enter_context(
                runtime.average(source, None, weights)
            )
            assert_array_equivalent(
                self,
                result,
                np.average(source_value, weights=weights_value),
                compare_contiguity=False,
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_average_matches_numpy_large_pairwise_materialization_order(
        self,
    ) -> None:
        size = 100_000
        rng = np.random.default_rng(123 + size)
        source_value = (
            rng.standard_normal(size) * np.logspace(-3, 3, size)
        ).astype(np.float32)
        weights_value = (rng.random(size) + 0.01).astype(np.float32)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            weights = stack.enter_context(runtime.from_numpy(weights_value))
            result = stack.enter_context(
                runtime.average(source, None, weights)
            )
            expected = np.asarray(
                np.average(source_value, weights=weights_value)
            )
            self.assertEqual(expected.shape, result.shape)
            self.assertEqual(expected.dtype, result.numpy_dtype)
            np.testing.assert_array_max_ulp(
                result.to_numpy(), expected, maxulp=8
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_average_invalid_weights_and_axes_are_explicit_and_leak_free(
        self,
    ) -> None:
        source_value = np.arange(6, dtype=np.float64).reshape(2, 3)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            zero_weights = stack.enter_context(
                runtime.from_numpy(np.zeros(3, dtype=np.float64))
            )
            short_weights = stack.enter_context(
                runtime.from_numpy(np.ones(2, dtype=np.float64))
            )
            active = runtime.retained_bytes

            with self.assertRaises(CnumpyError) as zero_context:
                runtime.average(source, 1, zero_weights)
            self.assertEqual(-1, zero_context.exception.status)
            self.assertEqual(
                "cnp_average_v2", zero_context.exception.function
            )
            self.assertEqual(
                "weights sum to zero, cannot normalize",
                zero_context.exception.message,
            )

            for axis, weights in ((1, short_weights), (None, short_weights)):
                with self.subTest(axis=axis):
                    with self.assertRaises(CnumpyError) as shape_context:
                        runtime.average(source, axis, weights)
                    self.assertEqual(-4, shape_context.exception.status)
                    self.assertEqual(
                        "cnp_average_v2", shape_context.exception.function
                    )

            with self.assertRaises(CnumpyError) as axis_context:
                runtime.average(source, 2, short_weights)
            self.assertEqual(-5, axis_context.exception.status)
            self.assertEqual(
                "cnp_average_v2", axis_context.exception.function
            )
            self.assertEqual(active, runtime.retained_bytes)
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_average_legacy_abi_preserves_sentinel_and_error_labels(
        self,
    ) -> None:
        source_value = np.arange(1, 7, dtype=np.float64).reshape(2, 3)
        full_weights_value = np.arange(2, 8, dtype=np.float64).reshape(2, 3)
        axis_weights_value = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        short_weights_value = np.ones(2, dtype=np.float64)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            full_weights = stack.enter_context(
                runtime.from_numpy(full_weights_value)
            )
            axis_weights = stack.enter_context(
                runtime.from_numpy(axis_weights_value)
            )
            short_weights = stack.enter_context(
                runtime.from_numpy(short_weights_value)
            )
            function = runtime.dll.cnp_average
            function.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
            ]
            function.restype = ctypes.c_void_p

            for axis, weights, expected_weights in (
                (-1, full_weights, full_weights_value),
                (1, axis_weights, axis_weights_value),
            ):
                with self.subTest(axis=axis):
                    runtime.dll.cnp_clear_error()
                    pointer = function(
                        source.pointer, axis, weights.pointer
                    )
                    result = stack.enter_context(
                        runtime._owned_result(pointer, "cnp_average")
                    )
                    assert_array_equivalent(
                        self,
                        result,
                        np.average(
                            source_value,
                            axis=None if axis == -1 else axis,
                            weights=expected_weights,
                        ),
                        compare_contiguity=False,
                    )

            active = runtime.retained_bytes
            for axis, weights, status in (
                (2, short_weights, -5),
                (1, short_weights, -4),
            ):
                with self.subTest(invalid_axis=axis, status=status):
                    runtime.dll.cnp_clear_error()
                    self.assertFalse(
                        function(source.pointer, axis, weights.pointer)
                    )
                    error = runtime.error_state()
                    self.assertEqual(status, error.status)
                    self.assertEqual("cnp_average", error.function)
                    self.assertEqual(active, runtime.retained_bytes)
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_average_malformed_array_metadata_fails_without_crashing(
        self,
    ) -> None:
        code = f"""
import ctypes

class ErrorState(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int),
        ("message", ctypes.c_char * 256),
        ("function", ctypes.c_char * 64),
    ]

class CnpArray(ctypes.Structure):
    _fields_ = [
        ("ndim", ctypes.c_int),
        ("shape", ctypes.c_void_p),
        ("strides", ctypes.c_void_p),
        ("size", ctypes.c_int64),
        ("data", ctypes.c_void_p),
        ("dtype", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("refcount", ctypes.c_int),
        ("base", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("owner", ctypes.c_void_p),
        ("owner_release", ctypes.c_void_p),
    ]

def error_tuple(dll):
    state = ErrorState()
    dll.cnp_get_error(ctypes.byref(state))
    return (
        state.status,
        bytes(state.function).split(b"\\0", 1)[0].decode("ascii"),
        bytes(state.message).split(b"\\0", 1)[0].decode("utf-8"),
    )

dll = ctypes.CDLL({str(DLL)!r})
dll.cnp_init.argtypes = []
dll.cnp_init.restype = ctypes.c_int
dll.cnp_cleanup.argtypes = []
dll.cnp_cleanup.restype = None
dll.cnp_clear_error.argtypes = []
dll.cnp_clear_error.restype = None
dll.cnp_get_error.argtypes = [ctypes.POINTER(ErrorState)]
dll.cnp_get_error.restype = ctypes.c_int
dll.cnp_get_allocated_memory.argtypes = []
dll.cnp_get_allocated_memory.restype = ctypes.c_size_t
dll.cnp_array_from_data.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64), ctypes.c_int, ctypes.c_int,
]
dll.cnp_array_from_data.restype = ctypes.c_void_p
dll.cnp_array_decref.argtypes = [ctypes.c_void_p]
dll.cnp_array_decref.restype = None
dll.cnp_average_v2.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_bool, ctypes.c_void_p,
]
dll.cnp_average_v2.restype = ctypes.c_void_p

if dll.cnp_init() != 0:
    raise SystemExit(10)
baseline = dll.cnp_get_allocated_memory()
source_shape = (ctypes.c_int64 * 2)(2, 3)
source_data = (ctypes.c_double * 6)(1, 2, 3, 4, 5, 6)
weights_shape = (ctypes.c_int64 * 1)(3)
weights_data = (ctypes.c_double * 3)(1, 2, 3)
source_pointer = dll.cnp_array_from_data(
    source_data, 2, source_shape, 13, 0
)
weights_pointer = dll.cnp_array_from_data(
    weights_data, 1, weights_shape, 13, 0
)
if not source_pointer or not weights_pointer:
    raise SystemExit(11)
source = ctypes.cast(source_pointer, ctypes.POINTER(CnpArray)).contents
weights = ctypes.cast(weights_pointer, ctypes.POINTER(CnpArray)).contents
try:
    active = dll.cnp_get_allocated_memory()
    cases = (
        ("source", source, "dtype", None, -3,
         "source array must have a numeric dtype"),
        ("source", source, "shape", None, -4,
         "source array has invalid shape metadata"),
        ("source", source, "strides", None, -4,
         "source array has invalid shape metadata"),
        ("source", source, "ndim", 65, -4,
         "source array has invalid shape metadata"),
        ("source", source, "data", None, -1,
         "source array requires a data buffer"),
        ("weights", weights, "dtype", None, -3,
         "weights array must have a numeric dtype"),
        ("weights", weights, "shape", None, -4,
         "weights array has invalid shape metadata"),
        ("weights", weights, "strides", None, -4,
         "weights array has invalid shape metadata"),
        ("weights", weights, "ndim", 65, -4,
         "weights array has invalid shape metadata"),
        ("weights", weights, "data", None, -1,
         "weights array requires a data buffer"),
    )
    for role, array, field, invalid, status, message in cases:
        original = getattr(array, field)
        setattr(array, field, invalid)
        try:
            dll.cnp_clear_error()
            result = dll.cnp_average_v2(
                source_pointer, 1, False, weights_pointer
            )
        finally:
            setattr(array, field, original)
        if result:
            dll.cnp_array_decref(result)
            raise AssertionError((role, field, "returned a result"))
        expected = (status, "cnp_average_v2", message)
        actual = error_tuple(dll)
        if actual != expected:
            raise AssertionError((role, field, actual, expected))
        if dll.cnp_get_allocated_memory() != active:
            raise AssertionError((role, field, "retained bytes changed"))
finally:
    dll.cnp_array_decref(weights_pointer)
    dll.cnp_array_decref(source_pointer)
    if dll.cnp_get_allocated_memory() != baseline:
        raise AssertionError("input arrays leaked")
    dll.cnp_cleanup()
"""
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            0,
            completed.returncode,
            msg=(completed.stdout + completed.stderr),
        )

    def test_average_scalar_axis_rules_match_numpy(self) -> None:
        source_value = np.array(3.0, dtype=np.float64)
        scalar_weights_value = np.array(2.0, dtype=np.float64)
        vector_weights_value = np.array([2.0], dtype=np.float64)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            scalar_weights = stack.enter_context(
                runtime.from_numpy(scalar_weights_value)
            )
            vector_weights = stack.enter_context(
                runtime.from_numpy(vector_weights_value)
            )

            for axis, weights, expected_weights in (
                (None, None, None),
                (None, scalar_weights, scalar_weights_value),
                (0, scalar_weights, scalar_weights_value),
                (-1, scalar_weights, scalar_weights_value),
            ):
                with self.subTest(axis=axis, weighted=weights is not None):
                    result = stack.enter_context(
                        runtime.average(source, axis, weights)
                    )
                    assert_array_equivalent(
                        self,
                        result,
                        np.average(
                            source_value,
                            axis=axis,
                            weights=expected_weights,
                        ),
                        compare_contiguity=False,
                    )

            for axis in (0, -1):
                with self.subTest(axis=axis, weighted=False):
                    with self.assertRaises(CnumpyError) as axis_context:
                        runtime.average(source, axis)
                    self.assertEqual(-5, axis_context.exception.status)
                    self.assertEqual(
                        "cnp_average_v2", axis_context.exception.function
                    )
                    self.assertEqual(
                        f"axis {axis} is out of bounds for array of dimension 0",
                        axis_context.exception.message,
                    )

            with self.assertRaises(CnumpyError) as shape_context:
                runtime.average(source, None, vector_weights)
            self.assertEqual(-4, shape_context.exception.status)
            self.assertEqual(
                "cnp_average_v2", shape_context.exception.function
            )
            self.assertEqual(
                "weights must have the same shape as the source array",
                shape_context.exception.message,
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_average_unweighted_empty_result_surfaces_division_by_zero(
        self,
    ) -> None:
        values = (
            (np.empty((2, 0), dtype=np.float64), 0),
            (np.empty((0, 3), dtype=np.float64), 1),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for source_value, axis in values:
                with self.subTest(shape=source_value.shape, axis=axis):
                    source = stack.enter_context(
                        runtime.from_numpy(source_value)
                    )
                    active = runtime.retained_bytes
                    caught_error = None
                    try:
                        unexpected = runtime.average(source, axis)
                    except CnumpyError as error:
                        caught_error = error
                    else:
                        unexpected.close()
                        self.fail("empty-result average returned an array")
                    self.assertIsNotNone(caught_error)
                    self.assertEqual(-1, caught_error.status)
                    self.assertEqual(
                        "cnp_average_v2", caught_error.function
                    )
                    self.assertEqual(
                        "division by zero", caught_error.message
                    )
                    self.assertEqual(active, runtime.retained_bytes)
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_average_adapter_rejects_closed_and_foreign_arrays(self) -> None:
        with CnumpyRuntime(DLL) as runtime, CnumpyRuntime(DLL) as foreign:
            baseline = runtime.retained_bytes
            source = runtime.from_numpy(
                np.arange(6, dtype=np.float64).reshape(2, 3)
            )
            weights = runtime.from_numpy(
                np.ones(3, dtype=np.float64)
            )
            foreign_source = foreign.from_numpy(
                np.arange(6, dtype=np.float64).reshape(2, 3)
            )
            foreign_weights = foreign.from_numpy(
                np.ones(3, dtype=np.float64)
            )
            try:
                try:
                    unexpected = runtime.average(foreign_source, 1, weights)
                except ValueError as error:
                    self.assertEqual(
                        "source array must belong to this runtime",
                        str(error),
                    )
                else:
                    unexpected.close()
                    self.fail("foreign source array was accepted")
                try:
                    unexpected = runtime.average(source, 1, foreign_weights)
                except ValueError as error:
                    self.assertEqual(
                        "weights array must belong to this runtime",
                        str(error),
                    )
                else:
                    unexpected.close()
                    self.fail("foreign weights array was accepted")

                source.close()
                with self.assertRaisesRegex(
                    RuntimeError, "is already closed"
                ):
                    runtime.average(source, 1, weights)

                weights.close()
                replacement = runtime.from_numpy(
                    np.arange(6, dtype=np.float64).reshape(2, 3)
                )
                try:
                    with self.assertRaisesRegex(
                        RuntimeError, "is already closed"
                    ):
                        runtime.average(replacement, 1, weights)
                finally:
                    replacement.close()
            finally:
                for array in (
                    weights, source, foreign_weights, foreign_source
                ):
                    if not array._closed:
                        array.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_average_supports_native_longdouble_and_clongdouble_dtypes(
        self,
    ) -> None:
        class ComplexLongDouble(ctypes.Structure):
            _fields_ = [
                ("real", ctypes.c_longdouble),
                ("imag", ctypes.c_longdouble),
            ]

        def native_array(runtime, values, dtype_number):
            shape = (ctypes.c_int64 * 1)(len(values))
            runtime.dll.cnp_clear_error()
            pointer = runtime.dll.cnp_array_from_data(
                ctypes.cast(values, ctypes.c_void_p),
                1,
                shape,
                dtype_number,
                0,
            )
            return runtime._owned_result(pointer, "cnp_array_from_data")

        long_source_values = (ctypes.c_longdouble * 3)(1.0, 2.0, 4.0)
        long_weight_values = (ctypes.c_longdouble * 3)(1.0, 2.0, 1.0)
        complex_source_value = np.array(
            [1.0 + 2.0j, 4.0 - 1.0j, 2.0 + 3.0j],
            dtype=np.complex128,
        )
        complex_weight_value = np.array(
            [1.0 - 1.0j, 2.0 + 0.5j, 0.5 + 0.25j],
            dtype=np.complex128,
        )
        complex_source_values = (ComplexLongDouble * 3)(
            *(ComplexLongDouble(value.real, value.imag)
              for value in complex_source_value)
        )
        complex_weight_values = (ComplexLongDouble * 3)(
            *(ComplexLongDouble(value.real, value.imag)
              for value in complex_weight_value)
        )

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            long_source = stack.enter_context(
                native_array(runtime, long_source_values, 14)
            )
            long_weights = stack.enter_context(
                native_array(runtime, long_weight_values, 14)
            )
            long_result = stack.enter_context(
                runtime.average(long_source, None, long_weights)
            )
            self.assertEqual(14, long_result.dtype_number)
            np.testing.assert_array_equal(
                long_result.to_numpy(),
                np.asarray(
                    np.average(
                        np.array([1.0, 2.0, 4.0], dtype=np.longdouble),
                        weights=np.array(
                            [1.0, 2.0, 1.0], dtype=np.longdouble
                        ),
                    )
                ),
                strict=True,
            )

            complex_source = stack.enter_context(
                native_array(runtime, complex_source_values, 17)
            )
            complex_weights = stack.enter_context(
                native_array(runtime, complex_weight_values, 17)
            )
            complex_result = stack.enter_context(
                runtime.average(complex_source, None, complex_weights)
            )
            self.assertEqual(17, complex_result.dtype_number)
            np.testing.assert_allclose(
                complex_result.to_numpy(),
                np.asarray(
                    np.average(
                        complex_source_value,
                        weights=complex_weight_value,
                    )
                ),
                rtol=4 * np.finfo(np.float64).eps,
                atol=0.0,
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_average_empty_reductions_match_numpy_results_and_errors(
        self,
    ) -> None:
        success_cases = (
            (np.empty(0, dtype=np.float64), None, None),
            (np.empty((2, 0), dtype=np.float64), 1, None),
            (
                np.empty((2, 0), dtype=np.float64),
                0,
                np.empty((2, 0), dtype=np.float64),
            ),
            (
                np.empty((2, 0), dtype=np.float64),
                0,
                np.empty(2, dtype=np.float64),
            ),
        )
        zero_weight_cases = (
            (
                np.empty((2, 0), dtype=np.float64),
                1,
                np.empty(0, dtype=np.float64),
            ),
            (
                np.empty((0, 3), dtype=np.float64),
                0,
                np.empty((0, 3), dtype=np.float64),
            ),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for source_value, axis, weights_value in success_cases:
                with self.subTest(
                    shape=source_value.shape,
                    axis=axis,
                    weighted=weights_value is not None,
                ):
                    source = stack.enter_context(
                        runtime.from_numpy(source_value)
                    )
                    weights = (
                        None
                        if weights_value is None
                        else stack.enter_context(
                            runtime.from_numpy(weights_value)
                        )
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        expected = np.average(
                            source_value,
                            axis=axis,
                            weights=weights_value,
                        )
                    result = stack.enter_context(
                        runtime.average(source, axis, weights)
                    )
                    assert_array_equivalent(
                        self,
                        result,
                        expected,
                        compare_contiguity=False,
                    )

            for source_value, axis, weights_value in zero_weight_cases:
                with self.subTest(shape=source_value.shape, axis=axis):
                    source = stack.enter_context(
                        runtime.from_numpy(source_value)
                    )
                    weights = stack.enter_context(
                        runtime.from_numpy(weights_value)
                    )
                    active = runtime.retained_bytes
                    with self.assertRaises(CnumpyError) as context:
                        runtime.average(source, axis, weights)
                    self.assertEqual(-1, context.exception.status)
                    self.assertEqual(
                        "cnp_average_v2", context.exception.function
                    )
                    self.assertEqual(
                        "weights sum to zero, cannot normalize",
                        context.exception.message,
                    )
                    self.assertEqual(active, runtime.retained_bytes)
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_average_propagates_nan_inf_and_complex_weights_like_numpy(
        self,
    ) -> None:
        cases = (
            (
                np.array(
                    [[1.0, np.nan, 3.0], [np.inf, 2.0, -np.inf]],
                    dtype=np.float64,
                ),
                np.array([1.0, 2.0, 3.0], dtype=np.float64),
                1,
            ),
            (
                np.array([1.0, 2.0], dtype=np.float64),
                np.array([np.inf, 1.0], dtype=np.float64),
                None,
            ),
            (
                np.array(
                    [[1.0 + 2.0j, 4.0 - 1.0j, 2.0 + 3.0j]],
                    dtype=np.complex64,
                ),
                np.array(
                    [1.0 - 1.0j, 2.0 + 0.5j, 0.5 + 0.25j],
                    dtype=np.complex64,
                ),
                1,
            ),
            (
                np.array([1.0 + np.nan * 1j, 2.0 + 3.0j]),
                np.array([1.0 + 0.0j, np.inf + 0.0j]),
                None,
            ),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for source_value, weights_value, axis in cases:
                with self.subTest(
                    source_dtype=str(source_value.dtype), axis=axis
                ):
                    source = stack.enter_context(
                        runtime.from_numpy(source_value)
                    )
                    weights = stack.enter_context(
                        runtime.from_numpy(weights_value)
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        expected = np.average(
                            source_value,
                            axis=axis,
                            weights=weights_value,
                        )
                    result = stack.enter_context(
                        runtime.average(source, axis, weights)
                    )
                    tolerance = 4 * np.finfo(
                        np.float32
                        if source_value.dtype.itemsize <= 8
                        else np.float64
                    ).eps
                    assert_array_equivalent(
                        self,
                        result,
                        expected,
                        compare_contiguity=False,
                        rtol=tolerance,
                    )

            zero_source = stack.enter_context(
                runtime.from_numpy(
                    np.array([1.0 + 2.0j, 3.0 - 1.0j])
                )
            )
            zero_weights = stack.enter_context(
                runtime.from_numpy(
                    np.array([1.0 + 2.0j, -1.0 - 2.0j])
                )
            )
            active = runtime.retained_bytes
            with self.assertRaises(CnumpyError) as context:
                runtime.average(zero_source, None, zero_weights)
            self.assertEqual(-1, context.exception.status)
            self.assertEqual(
                "weights sum to zero, cannot normalize",
                context.exception.message,
            )
            self.assertEqual(active, runtime.retained_bytes)
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_average_complex_division_avoids_spurious_overflow(
        self,
    ) -> None:
        cases = (
            (np.complex64, 1e30),
            (np.complex128, 1e300),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for dtype, scale in cases:
                with self.subTest(dtype=np.dtype(dtype).name):
                    source_value = np.array([0.25 + 0.25j], dtype=dtype)
                    weights_value = np.array(
                        [scale + scale * 1j], dtype=dtype
                    )
                    source = stack.enter_context(
                        runtime.from_numpy(source_value)
                    )
                    weights = stack.enter_context(
                        runtime.from_numpy(weights_value)
                    )
                    result = stack.enter_context(
                        runtime.average(source, None, weights)
                    )
                    assert_array_equivalent(
                        self,
                        result,
                        np.average(source_value, weights=weights_value),
                        compare_contiguity=False,
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_average_respects_transposed_fortran_and_strided_views(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes

            source_storage_value = np.arange(
                1, 13, dtype=np.float64
            ).reshape(3, 4)
            weights_storage_value = np.arange(
                2, 14, dtype=np.float64
            ).reshape(3, 4)
            source_storage = stack.enter_context(
                runtime.from_numpy(source_storage_value)
            )
            weights_storage = stack.enter_context(
                runtime.from_numpy(weights_storage_value)
            )
            source_fortran = stack.enter_context(
                runtime.transpose(source_storage, (1, 0))
            )
            weights_fortran = stack.enter_context(
                runtime.transpose(weights_storage, (1, 0))
            )
            self.assertTrue(source_fortran.f_contiguous)
            for axis in (None, 0, -1):
                with self.subTest(layout="fortran", axis=axis):
                    result = stack.enter_context(
                        runtime.average(
                            source_fortran, axis, weights_fortran
                        )
                    )
                    assert_array_equivalent(
                        self,
                        result,
                        np.average(
                            source_storage_value.T,
                            axis=axis,
                            weights=weights_storage_value.T,
                        ),
                        compare_contiguity=False,
                    )

            strided_storage_value = np.arange(
                1, 25, dtype=np.float64
            ).reshape(4, 6)
            axis_weights_storage_value = np.array(
                [1.0, 2.0, 3.0], dtype=np.float64
            )
            strided_storage = stack.enter_context(
                runtime.from_numpy(strided_storage_value)
            )
            axis_weights_storage = stack.enter_context(
                runtime.from_numpy(axis_weights_storage_value)
            )
            strided_source = stack.enter_context(
                self.slice_view(
                    runtime,
                    strided_storage,
                    (
                        _CnpSlice(0, 0, -1, False, False, True),
                        _CnpSlice(0, 0, 2, False, False, True),
                    ),
                )
            )
            reversed_weights = stack.enter_context(
                self.slice_view(
                    runtime,
                    axis_weights_storage,
                    (_CnpSlice(0, 0, -1, False, False, True),),
                )
            )
            self.assertFalse(strided_source.c_contiguous)
            result = stack.enter_context(
                runtime.average(strided_source, 1, reversed_weights)
            )
            assert_array_equivalent(
                self,
                result,
                np.average(
                    strided_storage_value[::-1, ::2],
                    axis=1,
                    weights=axis_weights_storage_value[::-1],
                ),
                compare_contiguity=False,
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_average_repeated_success_and_failure_paths_release_memory(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(
                runtime.from_numpy(
                    np.arange(1, 25, dtype=np.float64).reshape(2, 3, 4)
                )
            )
            weights = stack.enter_context(
                runtime.from_numpy(
                    np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
                )
            )
            zero_weights = stack.enter_context(
                runtime.from_numpy(np.zeros(4, dtype=np.float64))
            )
            active = runtime.retained_bytes
            for _ in range(64):
                runtime.average(source, -1, weights).close()
                runtime.average(source, None).close()
                with self.assertRaises(CnumpyError):
                    runtime.average(source, -1, zero_weights)
                self.assertEqual(active, runtime.retained_bytes)
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_average_rejects_non_numeric_source_and_weights_dtypes(
        self,
    ) -> None:
        def object_array(runtime):
            shape = (ctypes.c_int64 * 1)(2)
            values = (ctypes.c_void_p * 2)(None, None)
            runtime.dll.cnp_clear_error()
            pointer = runtime.dll.cnp_array_from_data(
                ctypes.cast(values, ctypes.c_void_p),
                1,
                shape,
                18,
                0,
            )
            return runtime._owned_result(pointer, "cnp_array_from_data")

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            numeric = stack.enter_context(
                runtime.from_numpy(np.array([1.0, 2.0]))
            )
            object_source = stack.enter_context(object_array(runtime))
            object_weights = stack.enter_context(object_array(runtime))
            active = runtime.retained_bytes
            for source, weights, message in (
                (
                    object_source,
                    numeric,
                    "source array must have a numeric dtype",
                ),
                (
                    numeric,
                    object_weights,
                    "weights array must have a numeric dtype",
                ),
            ):
                with self.subTest(message=message):
                    with self.assertRaises(CnumpyError) as context:
                        runtime.average(source, None, weights)
                    self.assertEqual(-3, context.exception.status)
                    self.assertEqual(
                        "cnp_average_v2", context.exception.function
                    )
                    self.assertEqual(message, context.exception.message)
                    self.assertEqual(active, runtime.retained_bytes)
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_extrema_and_arg_reductions_cover_all_axes_ties_and_nan(self) -> None:
        source_value = np.array(
            [
                [[-0.0, 4.0, np.nan], [0.0, 4.0, 2.0]],
                [[0.0, -3.0, 8.0], [-0.0, -3.0, 8.0]],
            ],
            dtype=np.float64,
        )
        logical_value = source_value.transpose(2, 0, 1)
        operations = (
            ("maximum", np.max),
            ("minimum", np.min),
            ("argmax", np.argmax),
            ("argmin", np.argmin),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            transposed = stack.enter_context(runtime.transpose(source, (2, 0, 1)))
            for method_name, numpy_function in operations:
                method = getattr(runtime, method_name)
                for axis in (None, 0, 1, 2, -1, -2, -3):
                    with self.subTest(operation=method_name, axis=axis):
                        result = stack.enter_context(method(transposed, axis))
                        assert_array_equivalent(
                            self,
                            result,
                            numpy_function(logical_value, axis=axis),
                            compare_contiguity=False,
                        )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_var_and_std_cover_negative_axes_views_ddof_and_default_dtype(self) -> None:
        source_values = (
            np.arange(24, dtype=np.float32).reshape(2, 3, 4),
            np.arange(24, dtype=np.float64).reshape(2, 3, 4),
            np.arange(24, dtype=np.int16).reshape(2, 3, 4),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for source_value in source_values:
                source = stack.enter_context(runtime.from_numpy(source_value))
                transposed = stack.enter_context(runtime.transpose(source, (2, 0, 1)))
                logical_value = source_value.transpose(2, 0, 1)
                for method_name, numpy_function in (
                    ("variance", np.var),
                    ("std", np.std),
                ):
                    method = getattr(runtime, method_name)
                    for ddof in (0, 1, 3):
                        for axis in (None, 0, 1, 2, -1, -2, -3):
                            with self.subTest(
                                operation=method_name,
                                dtype=str(source_value.dtype),
                                ddof=ddof,
                                axis=axis,
                            ):
                                with warnings.catch_warnings(), np.errstate(
                                    invalid="ignore", divide="ignore"
                                ):
                                    warnings.simplefilter("ignore")
                                    expected = numpy_function(
                                        logical_value, axis=axis, ddof=ddof
                                    )
                                result = stack.enter_context(
                                    method(transposed, axis, ddof)
                                )
                                assert_array_equivalent(
                                    self,
                                    result,
                                    expected,
                                    compare_contiguity=False,
                                    rtol=16 * np.finfo(expected.dtype).eps,
                                )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_nan_reductions_cover_axes_all_nan_inf_signed_zero_and_ddof(self) -> None:
        source_value = np.array(
            [
                [[np.nan, -0.0, np.inf], [np.nan, 0.0, 3.0]],
                [[np.nan, 0.0, -np.inf], [np.nan, -0.0, 5.0]],
            ],
            dtype=np.float64,
        )
        logical_value = source_value.transpose(2, 0, 1)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            transposed = stack.enter_context(runtime.transpose(source, (2, 0, 1)))
            for method_name, numpy_function in (
                ("nansum", np.nansum),
                ("nanprod", np.nanprod),
                ("nanmean", np.nanmean),
                ("nanmax", np.nanmax),
                ("nanmin", np.nanmin),
            ):
                method = getattr(runtime, method_name)
                for axis in (None, 0, 1, 2, -1, -2, -3):
                    with self.subTest(operation=method_name, axis=axis):
                        with warnings.catch_warnings(), np.errstate(
                            invalid="ignore", divide="ignore"
                        ):
                            warnings.simplefilter("ignore")
                            expected = numpy_function(logical_value, axis=axis)
                        result = stack.enter_context(method(transposed, axis))
                        assert_array_equivalent(
                            self,
                            result,
                            expected,
                            compare_contiguity=False,
                            rtol=8 * np.finfo(np.float64).eps,
                        )
            for method_name, numpy_function in (
                ("nanvar", np.nanvar),
                ("nanstd", np.nanstd),
            ):
                method = getattr(runtime, method_name)
                for ddof in (0, 1, 3):
                    for axis in (None, 0, 1, 2, -1, -2, -3):
                        with self.subTest(
                            operation=method_name, ddof=ddof, axis=axis
                        ):
                            with warnings.catch_warnings(), np.errstate(
                                invalid="ignore", divide="ignore"
                            ):
                                warnings.simplefilter("ignore")
                                expected = numpy_function(
                                    logical_value, axis=axis, ddof=ddof
                                )
                            result = stack.enter_context(
                                method(transposed, axis, ddof)
                            )
                            assert_array_equivalent(
                                self,
                                result,
                                expected,
                                compare_contiguity=False,
                                rtol=16 * np.finfo(np.float64).eps,
                            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_nan_cumulative_operations_cover_none_axes_views_and_promotion(self) -> None:
        source_values = (
            np.array([True, False, True], dtype=np.bool_),
            np.array([120, 7, -3], dtype=np.int8),
            np.array(
                [[[1.0, np.nan], [2.0, 3.0]],
                 [[np.nan, 4.0], [5.0, np.inf]]],
                dtype=np.float64,
            ),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for source_value in source_values:
                source = stack.enter_context(runtime.from_numpy(source_value))
                if source_value.ndim == 3:
                    operand = stack.enter_context(runtime.transpose(source, (2, 0, 1)))
                    logical_value = source_value.transpose(2, 0, 1)
                    axes = (None, 0, 1, 2, -1, -2, -3)
                else:
                    operand = source
                    logical_value = source_value
                    axes = (None, 0, -1)
                for method_name, numpy_function in (
                    ("nancumsum", np.nancumsum),
                    ("nancumprod", np.nancumprod),
                ):
                    method = getattr(runtime, method_name)
                    for axis in axes:
                        with self.subTest(
                            operation=method_name,
                            dtype=str(source_value.dtype), axis=axis,
                        ):
                            result = stack.enter_context(method(operand, axis))
                            assert_array_equivalent(
                                self, result,
                                numpy_function(logical_value, axis=axis),
                                compare_contiguity=False,
                            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_float32_nan_cumulative_operations_round_each_step(self) -> None:
        cases = (
            (
                "nancumsum",
                np.nancumsum,
                np.array([1e8, 1.0, -1e8, np.nan], dtype=np.float32),
            ),
            (
                "nancumprod",
                np.nancumprod,
                np.array(
                    [1.1823518, 0.55382103, 0.72035986, np.nan],
                    dtype=np.float32,
                ),
            ),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for method_name, numpy_function, source_value in cases:
                with self.subTest(operation=method_name):
                    source = stack.enter_context(
                        runtime.from_numpy(source_value)
                    )
                    result = stack.enter_context(
                        getattr(runtime, method_name)(source, 0)
                    )
                    assert_array_equivalent(
                        self,
                        result,
                        numpy_function(source_value, axis=0),
                        compare_contiguity=False,
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_nan_cumulative_exports_route_axis_through_shared_engine(self) -> None:
        source_value = np.array(
            [[[1.0, np.nan], [3.0, 4.0]],
             [[np.nan, 6.0], [7.0, 8.0]]],
            dtype=np.float64,
        )
        logical_value = source_value.transpose(2, 0, 1)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            transposed = stack.enter_context(runtime.transpose(source, (2, 0, 1)))
            for function_name, numpy_function in (
                ("cnp_nancumsum", np.nancumsum),
                ("cnp_nancumprod", np.nancumprod),
            ):
                with self.subTest(function=function_name):
                    function = getattr(runtime.dll, function_name)
                    function.argtypes = [ctypes.c_void_p, ctypes.c_int]
                    function.restype = ctypes.c_void_p
                    runtime.dll.cnp_clear_error()
                    pointer = function(transposed.pointer, 1)
                    result = stack.enter_context(
                        runtime._owned_result(pointer, function_name)
                    )
                    assert_array_equivalent(
                        self, result,
                        numpy_function(logical_value, axis=1),
                        compare_contiguity=False,
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_nanmax_axis_none_preserves_numpy_signed_zero(self) -> None:
        source_value = np.array([0.0, -0.0], dtype=np.float64)
        expected = np.nanmax(source_value)
        self.assertFalse(np.signbit(expected))

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            legacy = stack.enter_context(
                self.legacy_reduction(
                    runtime, source, "cnp_nanmax", (-1,)
                )
            )
            assert_array_equivalent(
                self, legacy, expected, compare_contiguity=False
            )
            with runtime.nanmax(source, None) as modern:
                assert_array_equivalent(
                    self, modern, expected, compare_contiguity=False
                )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_nanmean_axis_uses_per_slice_valid_counts(self) -> None:
        source_value = np.array(
            [[1.0, np.nan], [3.0, 4.0]], dtype=np.float64
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            for axis in (0, -2):
                with self.subTest(axis=axis):
                    legacy = stack.enter_context(
                        self.legacy_reduction(
                            runtime, source, "cnp_nanmean", (axis, 0)
                        )
                    )
                    assert_array_equivalent(
                        self,
                        legacy,
                        np.nanmean(source_value, axis=axis),
                        compare_contiguity=False,
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_nanvar_and_nanstd_use_per_slice_valid_counts(self) -> None:
        source_value = np.array(
            [[1.0, np.nan], [3.0, 4.0]], dtype=np.float64
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            for function_name, numpy_function in (
                ("cnp_nanvar", np.nanvar),
                ("cnp_nanstd", np.nanstd),
            ):
                for axis in (0, -2):
                    with self.subTest(function=function_name, axis=axis):
                        legacy = stack.enter_context(
                            self.legacy_reduction(
                                runtime, source, function_name, (axis, 0, 0)
                            )
                        )
                        assert_array_equivalent(
                            self,
                            legacy,
                            numpy_function(source_value, axis=axis),
                            compare_contiguity=False,
                        )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_nansum_preserves_numpy_dtype_and_reduction_order(
        self,
    ) -> None:
        float32_value = np.array(
            [1e8, 1.0, -1e8], dtype=np.float32
        )
        int8_value = np.array([120, 7, -3], dtype=np.int8).reshape(1, 3)

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for source_value, axes in (
                (float32_value, (CNP_AXIS_NONE, 0)),
                (int8_value, (CNP_AXIS_NONE, 0, -2)),
            ):
                source = stack.enter_context(runtime.from_numpy(source_value))
                for axis in axes:
                    with self.subTest(
                        dtype=str(source_value.dtype), axis=axis
                    ):
                        legacy = stack.enter_context(
                            self.legacy_reduction(
                                runtime,
                                source,
                                "cnp_nansum",
                                (axis, CNP_NOTYPE),
                            )
                        )
                        assert_array_equivalent(
                            self,
                            legacy,
                            np.nansum(
                                source_value,
                                axis=None if axis == CNP_AXIS_NONE else axis,
                            ),
                            compare_contiguity=False,
                        )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_nanprod_preserves_numpy_dtype_signed_zero_and_nan(
        self,
    ) -> None:
        source_values = (
            np.array([3, 4, -2], dtype=np.int8),
            np.array([-0.0, np.nan, 2.0], dtype=np.float32),
        )

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for source_value in source_values:
                source = stack.enter_context(runtime.from_numpy(source_value))
                for axis in (CNP_AXIS_NONE, 0):
                    with self.subTest(
                        dtype=str(source_value.dtype), axis=axis
                    ):
                        legacy = stack.enter_context(
                            self.legacy_reduction(
                                runtime,
                                source,
                                "cnp_nanprod",
                                (axis, CNP_NOTYPE),
                            )
                        )
                        assert_array_equivalent(
                            self,
                            legacy,
                            np.nanprod(
                                source_value,
                                axis=None if axis == CNP_AXIS_NONE else axis,
                            ),
                            compare_contiguity=False,
                        )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_mean_var_and_std_preserve_float32_semantics(
        self,
    ) -> None:
        source_value = np.array(
            [[1e8, 1.0, -1e8], [1.0, 2.0, 4.0]],
            dtype=np.float32,
        )
        operations = (
            ("cnp_mean", np.mean, lambda axis: (axis, CNP_NOTYPE)),
            ("cnp_var", np.var, lambda axis: (axis, 1, CNP_NOTYPE)),
            ("cnp_std", np.std, lambda axis: (axis, 1, CNP_NOTYPE)),
        )

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            for function_name, numpy_function, arguments in operations:
                for axis in (CNP_AXIS_NONE, 0, 1, -2):
                    with self.subTest(function=function_name, axis=axis):
                        legacy = stack.enter_context(
                            self.legacy_reduction(
                                runtime,
                                source,
                                function_name,
                                arguments(axis),
                            )
                        )
                        numpy_axis = (
                            None if axis == CNP_AXIS_NONE else axis
                        )
                        keyword_arguments = (
                            {"ddof": 1}
                            if function_name in ("cnp_var", "cnp_std")
                            else {}
                        )
                        expected = numpy_function(
                            source_value,
                            axis=numpy_axis,
                            **keyword_arguments,
                        )
                        assert_array_equivalent(
                            self,
                            legacy,
                            expected,
                            compare_contiguity=False,
                            rtol=16 * np.finfo(expected.dtype).eps,
                        )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_cumulative_operations_preserve_promotion_and_axes(
        self,
    ) -> None:
        source_value = np.array(
            [[1, 2, 3], [4, 5, 6]], dtype=np.uint8
        )

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            for function_name, numpy_function in (
                ("cnp_cumsum", np.cumsum),
                ("cnp_cumprod", np.cumprod),
            ):
                for axis in (CNP_AXIS_NONE, 0, 1, -2):
                    with self.subTest(function=function_name, axis=axis):
                        legacy = stack.enter_context(
                            self.legacy_reduction(
                                runtime,
                                source,
                                function_name,
                                (axis, CNP_NOTYPE),
                            )
                        )
                        assert_array_equivalent(
                            self,
                            legacy,
                            numpy_function(
                                source_value,
                                axis=(
                                    None
                                    if axis == CNP_AXIS_NONE
                                    else axis
                                ),
                            ),
                            compare_contiguity=False,
                        )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_extrema_aliases_preserve_nan_and_signed_zero(
        self,
    ) -> None:
        source_value = np.array(
            [[0.0, -0.0, np.nan], [-0.0, 0.0, 2.0]],
            dtype=np.float64,
        )

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            for function_name, numpy_function in (
                ("cnp_max", np.max),
                ("cnp_amax", np.amax),
                ("cnp_min", np.min),
                ("cnp_amin", np.amin),
            ):
                for axis in (CNP_AXIS_NONE, 0, 1, -2):
                    with self.subTest(function=function_name, axis=axis):
                        legacy = stack.enter_context(
                            self.legacy_reduction(
                                runtime, source, function_name, (axis,)
                            )
                        )
                        assert_array_equivalent(
                            self,
                            legacy,
                            numpy_function(
                                source_value,
                                axis=(
                                    None
                                    if axis == CNP_AXIS_NONE
                                    else axis
                                ),
                            ),
                            compare_contiguity=False,
                        )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_arg_extrema_and_ptp_preserve_nan_semantics(
        self,
    ) -> None:
        source_value = np.array(
            [[1.0, np.nan, 3.0], [np.nan, -2.0, 5.0]],
            dtype=np.float64,
        )

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            for function_name, numpy_function in (
                ("cnp_argmax", np.argmax),
                ("cnp_argmin", np.argmin),
                ("cnp_ptp", np.ptp),
            ):
                for axis in (CNP_AXIS_NONE, 0, 1, -2):
                    with self.subTest(function=function_name, axis=axis):
                        legacy = stack.enter_context(
                            self.legacy_reduction(
                                runtime, source, function_name, (axis,)
                            )
                        )
                        assert_array_equivalent(
                            self,
                            legacy,
                            numpy_function(
                                source_value,
                                axis=(
                                    None
                                    if axis == CNP_AXIS_NONE
                                    else axis
                                ),
                            ),
                            compare_contiguity=False,
                        )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_boolean_reductions_use_views_and_reject_bad_axes(
        self,
    ) -> None:
        source_value = np.array(
            [[0, 1, 2], [3, 0, 4]], dtype=np.int8
        )
        logical_value = source_value.transpose(1, 0)

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            transposed = stack.enter_context(runtime.transpose(source, (1, 0)))
            for function_name, numpy_function in (
                ("cnp_any", np.any),
                ("cnp_all", np.all),
            ):
                for axis in (CNP_AXIS_NONE, 0, 1, -2):
                    with self.subTest(function=function_name, axis=axis):
                        legacy = stack.enter_context(
                            self.legacy_reduction(
                                runtime,
                                transposed,
                                function_name,
                                (axis,),
                            )
                        )
                        assert_array_equivalent(
                            self,
                            legacy,
                            numpy_function(
                                logical_value,
                                axis=(
                                    None
                                    if axis == CNP_AXIS_NONE
                                    else axis
                                ),
                            ),
                            compare_contiguity=False,
                        )

                with self.subTest(function=function_name, axis=7):
                    try:
                        unexpected = self.legacy_reduction(
                            runtime,
                            transposed,
                            function_name,
                            (7,),
                        )
                    except CnumpyError as error:
                        self.assertIn(
                            "axis 7 is out of bounds", str(error)
                        )
                    else:
                        unexpected.close()
                        self.fail(
                            f"{function_name} accepted out-of-bounds axis 7"
                        )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_nanarg_reductions_cover_axes_ties_views_and_all_nan_errors(self) -> None:
        source_value = np.array(
            [
                [[np.nan, 4.0, 4.0], [1.0, np.nan, 2.0]],
                [[3.0, 4.0, np.nan], [3.0, 0.0, 2.0]],
            ],
            dtype=np.float64,
        )
        logical_value = source_value.transpose(2, 0, 1)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            transposed = stack.enter_context(runtime.transpose(source, (2, 0, 1)))
            for method_name, numpy_function in (
                ("nanargmax", np.nanargmax),
                ("nanargmin", np.nanargmin),
            ):
                method = getattr(runtime, method_name)
                for axis in (None, 0, 1, 2, -1, -2, -3):
                    with self.subTest(operation=method_name, axis=axis):
                        result = stack.enter_context(method(transposed, axis))
                        assert_array_equivalent(
                            self, result,
                            numpy_function(logical_value, axis=axis),
                            compare_contiguity=False,
                        )
                all_nan_value = np.full((2, 3), np.nan)
                all_nan = stack.enter_context(runtime.from_numpy(all_nan_value))
                with self.assertRaises(CnumpyError):
                    method(all_nan, 1)
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_nanarg_partial_all_nan_view_reports_exact_leak_free_error(
        self,
    ) -> None:
        stored_value = np.array(
            [[np.nan, np.nan, np.nan], [4.0, -2.0, 7.0]],
            dtype=np.float64,
        )

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            stored = stack.enter_context(runtime.from_numpy(stored_value))
            view = stack.enter_context(runtime.transpose(stored, (1, 0)))
            self.assertFalse(view.c_contiguous)

            for method_name in ("nanargmax", "nanargmin"):
                method = getattr(runtime, method_name)
                for axis in (0, -2):
                    with self.subTest(operation=method_name, axis=axis):
                        before_error = runtime.retained_bytes
                        with self.assertRaises(CnumpyError) as captured:
                            method(view, axis)
                        self.assertEqual(-1, captured.exception.status)
                        self.assertEqual(
                            f"cnp_{method_name}_v2",
                            captured.exception.function,
                        )
                        self.assertEqual(
                            "All-NaN slice encountered",
                            captured.exception.message,
                        )
                        self.assertEqual(before_error, runtime.retained_bytes)

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_median_and_percentile_families_cover_axes_views_and_all_nan(self) -> None:
        source_value = np.array(
            [
                [[np.nan, 1.0, 9.0], [2.0, 8.0, 3.0]],
                [[4.0, 7.0, 5.0], [6.0, np.nan, 0.0]],
            ],
            dtype=np.float64,
        )
        logical_value = source_value.transpose(2, 0, 1)
        cases = (
            ("median", None, np.median),
            ("nanmedian", None, np.nanmedian),
            ("percentile", 37.5, np.percentile),
            ("nanpercentile", 37.5, np.nanpercentile),
            ("quantile", 0.375, np.quantile),
            ("nanquantile", 0.375, np.nanquantile),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            transposed = stack.enter_context(runtime.transpose(source, (2, 0, 1)))
            for method_name, q, numpy_function in cases:
                method = getattr(runtime, method_name)
                for axis in (None, 0, 1, 2, -1, -2, -3):
                    with self.subTest(operation=method_name, axis=axis):
                        with np.errstate(invalid="ignore", divide="ignore"):
                            expected = (
                                numpy_function(logical_value, axis=axis)
                                if q is None else
                                numpy_function(logical_value, q, axis=axis)
                            )
                        result = stack.enter_context(
                            method(transposed, axis)
                            if q is None else method(transposed, q, axis)
                        )
                        assert_array_equivalent(
                            self, result, expected,
                            compare_contiguity=False,
                            rtol=8 * np.finfo(np.float64).eps,
                        )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_boolean_and_peak_to_peak_reductions_cover_all_axes_and_views(self) -> None:
        source_value = np.array(
            [[[0, 2, -3], [4, 0, 6]], [[7, 8, 0], [1, 2, 3]]],
            dtype=np.int32,
        )
        logical_value = source_value.transpose(2, 0, 1)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            transposed = stack.enter_context(runtime.transpose(source, (2, 0, 1)))
            for method_name, numpy_function in (
                ("any", np.any), ("all", np.all), ("ptp", np.ptp),
            ):
                method = getattr(runtime, method_name)
                for axis in (None, 0, 1, 2, -1, -2, -3):
                    with self.subTest(operation=method_name, axis=axis):
                        result = stack.enter_context(method(transposed, axis))
                        assert_array_equivalent(
                            self, result,
                            numpy_function(logical_value, axis=axis),
                            compare_contiguity=False,
                        )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_float64_sum_and_prod_preserve_numpy_reduction_order(self) -> None:
        sum_value = np.array([1e16, 1.0, -1e16, 1.0], dtype=np.float64)
        product_value = np.array(
            [1e200] * 8 + [1e-200] * 8,
            dtype=np.float64,
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            sum_source = stack.enter_context(runtime.from_numpy(sum_value))
            with self.subTest(operation="sum", layout="reduce_all"):
                sum_all = stack.enter_context(runtime.sum(sum_source))
                assert_array_equivalent(
                    self, sum_all, np.sum(sum_value), compare_contiguity=False
                )

            matrix_value = sum_value.reshape(1, 4)
            matrix_source = stack.enter_context(
                runtime.from_numpy(matrix_value)
            )
            with self.subTest(operation="sum", layout="last_axis"):
                sum_last = stack.enter_context(runtime.sum(matrix_source, 1))
                assert_array_equivalent(
                    self,
                    sum_last,
                    np.sum(matrix_value, axis=1),
                    compare_contiguity=False,
                )

            product_source = stack.enter_context(
                runtime.from_numpy(product_value)
            )
            with np.errstate(over="ignore", invalid="ignore"):
                expected_product = np.prod(product_value)
            with self.subTest(operation="prod", layout="reduce_all"):
                product_all = stack.enter_context(runtime.prod(product_source))
                assert_array_equivalent(
                    self,
                    product_all,
                    expected_product,
                    compare_contiguity=False,
                )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_contiguous_float64_last_axis_candidates_keep_numpy_order_and_bits(
        self,
    ) -> None:
        sum_pattern = np.array(
            [1e16, 1.0, -1e16, 1.0, -0.0, 0.0, 3.0, -3.0],
            dtype=np.float64,
        )
        sum_value = np.resize(sum_pattern, (512, 512))
        product_pattern = np.array(
            [1e200] * 8 + [1e-200] * 8 + [1.0000000000000002],
            dtype=np.float64,
        )
        product_value = np.resize(product_pattern, (4, 257))
        nan_a = np.array(
            [0x7FF8000000000001], dtype=np.uint64
        ).view(np.float64)[0]
        nan_b = np.array(
            [0x7FF8000000000002], dtype=np.uint64
        ).view(np.float64)[0]
        extrema_value = np.array(
            [
                [0.0, -0.0, 0.0, -0.0],
                [4.0, 9.0, 9.0, -1.0],
                [nan_a, 7.0, nan_b, 8.0],
                [-5.0, -5.0, -7.0, -5.0],
            ],
            dtype=np.float64,
        )

        operations = (
            ("sum", sum_value, lambda runtime, source: runtime.sum(source, -1), np.sum),
            (
                "prod",
                product_value,
                lambda runtime, source: runtime.prod(source, -1),
                np.prod,
            ),
            (
                "maximum",
                extrema_value,
                lambda runtime, source: runtime.maximum(source, -1),
                np.max,
            ),
            (
                "minimum",
                extrema_value,
                lambda runtime, source: runtime.minimum(source, -1),
                np.min,
            ),
            (
                "argmax",
                extrema_value,
                lambda runtime, source: runtime.argmax(source, -1),
                np.argmax,
            ),
            (
                "argmin",
                extrema_value,
                lambda runtime, source: runtime.argmin(source, -1),
                np.argmin,
            ),
        )

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with np.errstate(over="ignore", invalid="ignore"):
                for operation, source_value, native_call, numpy_call in operations:
                    with self.subTest(operation=operation), runtime.from_numpy(
                        source_value
                    ) as source, native_call(runtime, source) as actual:
                        expected = np.asarray(numpy_call(source_value, axis=-1))
                        actual_value = actual.to_numpy()
                        if operation in ("maximum", "minimum"):
                            assert_array_equivalent(
                                self,
                                actual,
                                expected,
                                compare_contiguity=False,
                            )
                            finite = ~np.isnan(expected)
                            np.testing.assert_array_equal(
                                actual_value[finite].view(np.uint64),
                                expected[finite].view(np.uint64),
                                strict=True,
                            )
                        else:
                            self.assertEqual(
                                expected.tobytes(), actual_value.tobytes()
                            )

                flat = extrema_value.reshape(-1)
                with runtime.from_numpy(flat) as source:
                    for operation, native_call, numpy_call in (
                        ("argmax", runtime.argmax, np.argmax),
                        ("argmin", runtime.argmin, np.argmin),
                    ):
                        with self.subTest(
                            operation=operation, layout="reduce_all"
                        ), native_call(source) as actual:
                            expected = np.asarray(numpy_call(flat))
                            self.assertEqual(
                                expected.tobytes(), actual.to_numpy().tobytes()
                            )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_contiguous_float64_extrema_match_numpy_nan_payload_vector_tail(
        self,
    ) -> None:
        payloads = np.array(
            [0x7FF8000000000001, 0x7FF8000000000002],
            dtype=np.uint64,
        ).view(np.float64)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            mismatches: list[tuple[str, int, str, str, str]] = []
            for length in range(1, 18):
                rows: list[np.ndarray] = []
                cases = (
                    ("first", (0,)),
                    ("last", (length - 1,)),
                    ("first-last", (0, length - 1)),
                    ("second-last", (min(1, length - 1), length - 1)),
                )
                labels = [label for label, _ in cases]
                for _, selected_positions in cases:
                    row = np.arange(length, dtype=np.float64) - length
                    for payload_index, position in enumerate(
                        dict.fromkeys(selected_positions)
                    ):
                        row[position] = payloads[payload_index % 2]
                    rows.append(row)
                zero_bits = np.resize(
                    np.array(
                        [0x0000000000000000, 0x8000000000000000],
                        dtype=np.uint64,
                    ),
                    length,
                )
                rows.append(zero_bits.view(np.float64))
                labels.append("alternating-zero")
                source_value = np.stack(rows)
                with runtime.from_numpy(source_value) as source:
                    for method_name, numpy_function in (
                        ("maximum", np.max),
                        ("minimum", np.min),
                    ):
                        with self.subTest(
                            operation=method_name, length=length
                        ), getattr(runtime, method_name)(source, -1) as actual:
                            with np.errstate(invalid="ignore"):
                                expected = np.asarray(
                                    numpy_function(source_value, axis=-1)
                                )
                            actual_bits = actual.to_numpy().view(np.uint64)
                            expected_bits = expected.view(np.uint64)
                            for row, label in enumerate(labels):
                                if actual_bits[row] == expected_bits[row]:
                                    continue
                                for bits in (
                                    int(actual_bits[row]),
                                    int(expected_bits[row]),
                                ):
                                    self.assertEqual(
                                        0x7FF0000000000000,
                                        bits & 0x7FF0000000000000,
                                    )
                                    self.assertNotEqual(
                                        0, bits & 0x000FFFFFFFFFFFFF
                                    )
                                mismatches.append(
                                    (
                                        method_name,
                                        length,
                                        label,
                                        f"0x{int(actual_bits[row]):016x}",
                                        f"0x{int(expected_bits[row]):016x}",
                                    )
                                )
            self.assertEqual([], mismatches)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_contiguous_float64_reduction_fast_path_keeps_edge_contracts(
        self,
    ) -> None:
        value_cases = (
            (
                "sum",
                np.array([1e16, 1.0, -1e16, 1.0], dtype=np.float64),
                np.sum,
            ),
            (
                "mean",
                np.array(
                    [1.0519550354044216] + [0.0] * 8,
                    dtype=np.float64,
                ),
                np.mean,
            ),
            (
                "prod",
                np.array(
                    [1e200] * 8 + [1e-200] * 8,
                    dtype=np.float64,
                ),
                np.prod,
            ),
            (
                "std",
                np.array(
                    [1e16, 1.0, -1e16, 1.0], dtype=np.float64
                ),
                np.std,
            ),
            (
                "maximum",
                np.array([0.0, -0.0], dtype=np.float64),
                np.max,
            ),
            (
                "minimum",
                np.array([-0.0, 0.0], dtype=np.float64),
                np.min,
            ),
        )

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            with np.errstate(over="ignore", invalid="ignore"):
                for method_name, source_value, numpy_function in value_cases:
                    with self.subTest(operation=method_name, layout="flat"):
                        source = stack.enter_context(
                            runtime.from_numpy(source_value)
                        )
                        result = stack.enter_context(
                            getattr(runtime, method_name)(source)
                        )
                        expected = np.asarray(numpy_function(source_value))
                        self.assertEqual(
                            expected.tobytes(), result.to_numpy().tobytes()
                        )

                matrix_value = np.array(
                    [
                        [1e16, 1.0, -1e16, 1.0],
                        [0.0, -0.0, 0.0, -0.0],
                    ],
                    dtype=np.float64,
                )
                matrix = stack.enter_context(runtime.from_numpy(matrix_value))
                last_axis = stack.enter_context(runtime.sum(matrix, -1))
                expected_last_axis = np.sum(matrix_value, axis=-1)
                self.assertEqual(
                    expected_last_axis.tobytes(),
                    last_axis.to_numpy().tobytes(),
                )

                nan_source = stack.enter_context(
                    runtime.from_numpy(
                        np.array([1.0, np.nan, 2.0], dtype=np.float64)
                    )
                )
                nan_result = stack.enter_context(
                    runtime.maximum(nan_source)
                )
                self.assertTrue(np.isnan(nan_result.to_numpy()).all())

            empty = stack.enter_context(
                runtime.from_numpy(np.empty((0,), dtype=np.float64))
            )
            for method_name in ("maximum", "minimum"):
                with self.subTest(operation=method_name, layout="empty"):
                    before_error = runtime.retained_bytes
                    with self.assertRaises(CnumpyError) as captured:
                        getattr(runtime, method_name)(empty)
                    self.assertEqual(-1, captured.exception.status)
                    self.assertEqual(
                        f"cnp_{'max' if method_name == 'maximum' else 'min'}_v2",
                        captured.exception.function,
                    )
                    self.assertEqual(
                        "zero-size reduction has no identity",
                        captured.exception.message,
                    )
                    self.assertEqual(before_error, runtime.retained_bytes)

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_contiguous_float64_extrema_keep_selected_source_bits(
        self,
    ) -> None:
        payloads = np.array(
            [0x7FF8000000000001, 0x7FF8000000000002],
            dtype=np.uint64,
        ).view(np.float64)
        cases = (
            (
                "maximum",
                np.array(
                    [1.0, payloads[0], payloads[1], np.inf],
                    dtype=np.float64,
                ),
                0x7FF8000000000001,
            ),
            (
                "minimum",
                np.array(
                    [-1.0, payloads[1], payloads[0], -np.inf],
                    dtype=np.float64,
                ),
                0x7FF8000000000002,
            ),
            (
                "maximum",
                np.array([-np.inf, 1.0, np.inf], dtype=np.float64),
                0x7FF0000000000000,
            ),
            (
                "minimum",
                np.array([np.inf, 1.0, -np.inf], dtype=np.float64),
                0xFFF0000000000000,
            ),
            (
                "maximum",
                np.array([0.0, -0.0], dtype=np.float64),
                0x8000000000000000,
            ),
            (
                "minimum",
                np.array([-0.0, 0.0], dtype=np.float64),
                0x0000000000000000,
            ),
        )

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for method_name, source_value, expected_bits in cases:
                with self.subTest(
                    operation=method_name,
                    expected_bits=f"0x{expected_bits:016x}",
                ):
                    source = stack.enter_context(
                        runtime.from_numpy(source_value)
                    )
                    result = stack.enter_context(
                        getattr(runtime, method_name)(source)
                    )
                    actual_bits = int(
                        result.to_numpy().view(np.uint64)
                    )
                    self.assertEqual(expected_bits, actual_bits)

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_contiguous_float64_pairwise_tree_keeps_bits_and_edges(
        self,
    ) -> None:
        pattern = np.array(
            [1e16, 1.0, -1e16, 1.0, -0.0, 0.0, 3.0, -3.0],
            dtype=np.float64,
        )
        values = [
            np.resize(pattern, length)
            for length in (7, 8, 127, 128, 129, 257)
        ]
        values.extend(
            (
                np.array([1.0, np.nan, 2.0], dtype=np.float64),
                np.array([np.inf, -np.inf, 1.0], dtype=np.float64),
                np.array([-0.0, -0.0], dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            )
        )
        operations = (
            ("sum", lambda runtime, source: runtime.sum(source), np.sum),
            ("mean", lambda runtime, source: runtime.mean(source), np.mean),
            (
                "variance-ddof-0",
                lambda runtime, source: runtime.variance(source, None, 0),
                lambda value: np.var(value, ddof=0),
            ),
            (
                "variance-ddof-1",
                lambda runtime, source: runtime.variance(source, None, 1),
                lambda value: np.var(value, ddof=1),
            ),
            (
                "std-ddof-0",
                lambda runtime, source: runtime.std(source, None, 0),
                lambda value: np.std(value, ddof=0),
            ),
            (
                "std-ddof-1",
                lambda runtime, source: runtime.std(source, None, 1),
                lambda value: np.std(value, ddof=1),
            ),
        )

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            with warnings.catch_warnings(), np.errstate(
                invalid="ignore", divide="ignore"
            ):
                warnings.simplefilter("ignore")
                for source_value in values:
                    source = stack.enter_context(
                        runtime.from_numpy(source_value)
                    )
                    for operation, call, numpy_call in operations:
                        with self.subTest(
                            operation=operation,
                            length=source_value.size,
                            values=source_value.tolist(),
                        ):
                            result = stack.enter_context(call(runtime, source))
                            actual = result.to_numpy()
                            expected = np.asarray(numpy_call(source_value))
                            if np.isnan(expected).any():
                                np.testing.assert_equal(actual, expected)
                            else:
                                self.assertEqual(
                                    expected.tobytes(), actual.tobytes()
                                )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_sum_families_apply_numpy_positive_zero_identity_exactly(
        self,
    ) -> None:
        lengths = (1, 2, 7, 8, 9, 127, 128, 129)
        regular_operations = (
            ("sum", lambda runtime, source, axis: runtime.sum(source, axis), np.sum),
            (
                "mean",
                lambda runtime, source, axis: runtime.mean(source, axis),
                np.mean,
            ),
            (
                "variance",
                lambda runtime, source, axis: runtime.variance(source, axis, 0),
                np.var,
            ),
            (
                "std",
                lambda runtime, source, axis: runtime.std(source, axis, 0),
                np.std,
            ),
        )
        nan_operations = (
            (
                "nansum",
                lambda runtime, source, axis: runtime.nansum(source, axis),
                np.nansum,
            ),
            (
                "nanmean",
                lambda runtime, source, axis: runtime.nanmean(source, axis),
                np.nanmean,
            ),
            (
                "nanvar",
                lambda runtime, source, axis: runtime.nanvar(source, axis, 0),
                np.nanvar,
            ),
            (
                "nanstd",
                lambda runtime, source, axis: runtime.nanstd(source, axis, 0),
                np.nanstd,
            ),
        )

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with warnings.catch_warnings(), np.errstate(
                invalid="ignore", divide="ignore"
            ):
                warnings.simplefilter("ignore")
                for dtype in (np.float32, np.float64):
                    for length in lengths:
                        negative_zeros = np.full(length, -0.0, dtype=dtype)
                        negative_zero_rows = np.stack(
                            (negative_zeros, negative_zeros)
                        )
                        nan_and_negative_zero_rows = np.concatenate(
                            (
                                np.full((2, 1), np.nan, dtype=dtype),
                                negative_zero_rows,
                            ),
                            axis=1,
                        )
                        cases = (
                            (
                                "reduce_all",
                                negative_zeros,
                                None,
                                regular_operations,
                            ),
                            (
                                "last_axis",
                                negative_zero_rows,
                                -1,
                                regular_operations,
                            ),
                            (
                                "nan_reduce_all_without_nan",
                                negative_zeros,
                                None,
                                nan_operations,
                            ),
                            (
                                "nan_last_axis_without_nan",
                                negative_zero_rows,
                                -1,
                                nan_operations,
                            ),
                            (
                                "nan_reduce_all",
                                nan_and_negative_zero_rows.reshape(-1),
                                None,
                                nan_operations,
                            ),
                            (
                                "nan_last_axis",
                                nan_and_negative_zero_rows,
                                -1,
                                nan_operations,
                            ),
                        )
                        for layout, source_value, axis, operations in cases:
                            with runtime.from_numpy(source_value) as source:
                                for operation, call, numpy_call in operations:
                                    with self.subTest(
                                        operation=operation,
                                        layout=layout,
                                        dtype=np.dtype(dtype).name,
                                        length=length,
                                    ):
                                        with call(runtime, source, axis) as result:
                                            actual = result.to_numpy()
                                            expected = np.asarray(
                                                numpy_call(
                                                    source_value, axis=axis
                                                )
                                            )
                                            self.assertEqual(
                                                expected.tobytes(),
                                                actual.tobytes(),
                                            )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_float64_mean_uses_numpy_division_rounding(self) -> None:
        source_value = np.array(
            [1.0519550354044216] + [0.0] * 8,
            dtype=np.float64,
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(source_value) as source:
                with runtime.mean(source) as result:
                    assert_array_equivalent(
                        self,
                        result,
                        np.mean(source_value),
                        compare_contiguity=False,
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_ahk_scalar_reductions_preserve_numpy_order(self) -> None:
        sum_value = np.array([1e16, 1.0, -1e16, 1.0], dtype=np.float64)
        product_value = np.array(
            [1e200] * 8 + [1e-200] * 8,
            dtype=np.float64,
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            sum_source = stack.enter_context(runtime.from_numpy(sum_value))
            product_source = stack.enter_context(
                runtime.from_numpy(product_value)
            )
            with np.errstate(over="ignore", invalid="ignore"):
                expected_product = np.prod(product_value)
            for function_name, source, expected in (
                ("cnp_ahk_sum", sum_source, np.sum(sum_value)),
                ("cnp_ahk_mean", sum_source, np.mean(sum_value)),
                ("cnp_ahk_prod", product_source, expected_product),
            ):
                with self.subTest(function=function_name):
                    function = getattr(runtime.dll, function_name)
                    function.argtypes = [ctypes.c_void_p, ctypes.c_int]
                    function.restype = ctypes.c_double
                    with np.errstate(over="ignore", invalid="ignore"):
                        actual = function(source.pointer, -1)
                    np.testing.assert_equal(actual, expected)
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_float32_reductions_round_like_numpy_across_views(self) -> None:
        logical_value = np.array(
            [[1e8, 1.0, -1e8], [1e8, np.nan, -1e8]],
            dtype=np.float32,
        )
        stored_value = logical_value.T.copy()
        operations = (
            ("sum", np.sum),
            ("mean", np.mean),
            ("variance", np.var),
            ("std", np.std),
            ("nansum", np.nansum),
            ("nanmean", np.nanmean),
            ("nanvar", np.nanvar),
            ("nanstd", np.nanstd),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            flat_source = stack.enter_context(
                runtime.from_numpy(logical_value[0])
            )
            stored_source = stack.enter_context(
                runtime.from_numpy(stored_value)
            )
            view = stack.enter_context(runtime.transpose(stored_source, (1, 0)))
            for method_name, numpy_function in operations:
                method = getattr(runtime, method_name)
                with self.subTest(operation=method_name, layout="reduce_all"):
                    with warnings.catch_warnings(), np.errstate(
                        invalid="ignore", divide="ignore"
                    ):
                        warnings.simplefilter("ignore")
                        expected = numpy_function(logical_value[0])
                    result = stack.enter_context(method(flat_source))
                    assert_array_equivalent(
                        self, result, expected, compare_contiguity=False
                    )
                with self.subTest(operation=method_name, layout="view_axis1"):
                    with warnings.catch_warnings(), np.errstate(
                        invalid="ignore", divide="ignore"
                    ):
                        warnings.simplefilter("ignore")
                        expected = numpy_function(logical_value, axis=1)
                    result = stack.enter_context(method(view, 1))
                    assert_array_equivalent(
                        self, result, expected, compare_contiguity=False
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_nan_variance_uses_numpy_source_dtype_intermediates(self) -> None:
        source_value = np.array([1e8, -1e8, np.nan], dtype=np.float32)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            for method_name, numpy_function in (
                ("nanvar", np.nanvar),
                ("nanstd", np.nanstd),
            ):
                with self.subTest(operation=method_name):
                    with warnings.catch_warnings(), np.errstate(
                        invalid="ignore", divide="ignore"
                    ):
                        warnings.simplefilter("ignore")
                        expected = numpy_function(
                            source_value, dtype=np.float64
                        )
                    result = stack.enter_context(
                        getattr(runtime, method_name)(
                            source, dtype_number=13
                        )
                    )
                    assert_array_equivalent(
                        self, result, expected, compare_contiguity=False
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_nan_extrema_match_numpy_signed_zero_reduction_order(self) -> None:
        cases = (
            (
                "nanmin",
                np.nanmin,
                np.array([[0.0, -0.0], [0.0, 0.0]], dtype=np.float64),
                0,
            ),
            (
                "nanmax",
                np.nanmax,
                np.array([0.0, 0.0, 0.0, 0.0, -0.0], dtype=np.float64),
                None,
            ),
            (
                "nanmax",
                np.nanmax,
                np.array([0.0, -0.0], dtype=np.float64),
                None,
            ),
            (
                "nanmin",
                np.nanmin,
                np.array([-0.0, 0.0], dtype=np.float64),
                None,
            ),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for method_name, numpy_function, source_value, axis in cases:
                with self.subTest(
                    operation=method_name, shape=source_value.shape, axis=axis
                ):
                    source = stack.enter_context(
                        runtime.from_numpy(source_value)
                    )
                    result = stack.enter_context(
                        getattr(runtime, method_name)(source, axis)
                    )
                    assert_array_equivalent(
                        self,
                        result,
                        numpy_function(source_value, axis=axis),
                        compare_contiguity=False,
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_nanmin_reduce_all_noncontiguous_view_uses_buffered_run_order(
        self,
    ) -> None:
        stored_value = np.array(
            [[-0.0, -0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        numpy_view = stored_value.T
        expected = np.nanmin(numpy_view, axis=None)
        self.assertFalse(np.signbit(expected))

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            stored = stack.enter_context(runtime.from_numpy(stored_value))
            view = stack.enter_context(runtime.transpose(stored, (1, 0)))
            result = stack.enter_context(runtime.nanmin(view, None))
            assert_array_equivalent(
                self,
                result,
                expected,
                compare_contiguity=False,
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_nanmin_reduce_all_observes_numpy_buffer_boundaries(self) -> None:
        source_value = np.zeros(8193, dtype=np.float64)
        source_value[8189] = -0.0
        expected = np.nanmin(source_value, axis=None)
        self.assertTrue(np.signbit(expected))

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(source_value) as source:
                with runtime.nanmin(source, None) as result:
                    assert_array_equivalent(
                        self,
                        result,
                        expected,
                        compare_contiguity=False,
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_nan_extrema_explicit_axis_observes_numpy_buffer_boundaries(
        self,
    ) -> None:
        buffer_size = 8192
        lengths = (
            buffer_size - 1,
            buffer_size,
            buffer_size + 1,
            buffer_size + 4,
        )
        operations = (
            ("nanmin", np.nanmin, 0.0, -0.0),
            ("nanmax", np.nanmax, -0.0, 0.0),
        )

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for dtype in (np.float32, np.float64):
                lanes = 8 if dtype is np.float32 else 4
                for length in lengths:
                    positions = sorted(
                        {
                            0,
                            1,
                            lanes - 1,
                            lanes,
                            length // 2,
                            buffer_size - lanes - 1,
                            buffer_size - lanes,
                            buffer_size - 1,
                            buffer_size,
                            length - 1,
                        }
                    )
                    for position in positions:
                        if position < 0 or position >= length:
                            continue
                        for (
                            method_name,
                            numpy_function,
                            fill_value,
                            marker_value,
                        ) in operations:
                            with self.subTest(
                                dtype=np.dtype(dtype).name,
                                length=length,
                                position=position,
                                operation=method_name,
                            ):
                                source_value = np.full(
                                    length, fill_value, dtype=dtype
                                )
                                source_value[position] = marker_value
                                with runtime.from_numpy(source_value) as source:
                                    for axis in (None, 0):
                                        with self.subTest(axis=axis):
                                            expected = numpy_function(
                                                source_value, axis=axis
                                            )
                                            with getattr(
                                                runtime, method_name
                                            )(source, axis) as result:
                                                assert_array_equivalent(
                                                    self,
                                                    result,
                                                    expected,
                                                    compare_contiguity=False,
                                                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_nan_extrema_singleton_axis_has_a_valid_iterator_run(self) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for dtype in (np.float32, np.float64):
                source_value = np.array([3.0], dtype=dtype)
                source = stack.enter_context(runtime.from_numpy(source_value))
                for method_name, numpy_function in (
                    ("nanmin", np.nanmin),
                    ("nanmax", np.nanmax),
                ):
                    for axis in (0, -1):
                        with self.subTest(
                            dtype=np.dtype(dtype).name,
                            operation=method_name,
                            axis=axis,
                        ):
                            result = stack.enter_context(
                                getattr(runtime, method_name)(source, axis)
                            )
                            assert_array_equivalent(
                                self,
                                result,
                                numpy_function(source_value, axis=axis),
                                compare_contiguity=False,
                            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_singleton_axis_inner_stride_is_defined_before_dimension_access(
        self,
    ) -> None:
        implementation = (ROOT / "src" / "reduce.c").read_text(
            encoding="utf-8"
        )
        function_start = implementation.index(
            "static int64_t reduction_iterator_binary_inner_stride("
        )
        function_end = implementation.index("\n}\n", function_start)
        function_body = implementation[function_start:function_end]
        singleton_case = function_body.index("run->reduced_position <= 0")
        dimension_access = function_body.index("run->dimensions[0]")
        self.assertLess(singleton_case, dimension_access)

    def test_nanmax_reduce_all_negative_stride_uses_keeporder_direction(
        self,
    ) -> None:
        stored_value = np.array(
            [-0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        numpy_view = stored_value[::-1]
        expected = np.nanmax(numpy_view)
        self.assertFalse(np.signbit(expected))

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            stored = stack.enter_context(runtime.from_numpy(stored_value))
            view = stack.enter_context(
                self.slice_view(
                    runtime,
                    stored,
                    (_CnpSlice(0, 0, -1, False, False, True),),
                )
            )
            result = stack.enter_context(runtime.nanmax(view, None))
            assert_array_equivalent(
                self, result, expected, compare_contiguity=False
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_nanmax_explicit_axis_negative_stride_uses_output_run_order(
        self,
    ) -> None:
        stored_value = np.zeros((5, 2), dtype=np.float64)
        stored_value[0, 0] = -0.0
        numpy_view = stored_value[:, ::-1]
        expected = np.nanmax(numpy_view, axis=1)
        self.assertFalse(np.signbit(expected).any())

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            stored = stack.enter_context(runtime.from_numpy(stored_value))
            view = stack.enter_context(
                self.slice_view(
                    runtime,
                    stored,
                    (
                        _CnpSlice(0, 0, 0, False, False, False),
                        _CnpSlice(0, 0, -1, False, False, True),
                    ),
                )
            )
            result = stack.enter_context(runtime.nanmax(view, 1))
            assert_array_equivalent(
                self, result, expected, compare_contiguity=False
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_nanmin_zero_stride_broadcast_uses_keeporder_dimension_order(
        self,
    ) -> None:
        source_value = np.array([-0.0, 0.0], dtype=np.float64)
        numpy_view = np.broadcast_to(source_value, (3, 2))
        expected = np.nanmin(numpy_view)
        self.assertTrue(np.signbit(expected))

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            view = stack.enter_context(
                self.broadcast_view(runtime, source, (3, 2))
            )
            result = stack.enter_context(runtime.nanmin(view, None))
            assert_array_equivalent(
                self, result, expected, compare_contiguity=False
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_nan_extrema_zero_stride_output_run_uses_scalar_binary_order(
        self,
    ) -> None:
        cases = (
            ("nanmin", np.nanmin, np.array([[[-0.0], [0.0]]])),
            ("nanmax", np.nanmax, np.array([[[0.0], [-0.0]]])),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for shape in ((1, 2, 4), (4, 2, 1), (2, 2, 2)):
                for method_name, numpy_function, source_value in cases:
                    source_value = source_value.astype(np.float64)
                    numpy_view = np.broadcast_to(source_value, shape)
                    expected = numpy_function(numpy_view, axis=1)
                    with self.subTest(operation=method_name, shape=shape):
                        source = stack.enter_context(
                            runtime.from_numpy(source_value)
                        )
                        view = stack.enter_context(
                            self.broadcast_view(runtime, source, shape)
                        )
                        result = stack.enter_context(
                            getattr(runtime, method_name)(view, 1)
                        )
                        assert_array_equivalent(
                            self,
                            result,
                            expected,
                            compare_contiguity=False,
                        )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_nan_extrema_buffered_output_run_spans_pre_reduction_axes(
        self,
    ) -> None:
        operations = (
            ("nanmin", np.nanmin, 0.0, -0.0),
            ("nanmax", np.nanmax, -0.0, 0.0),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for dtype in (np.float32, np.float64):
                for method_name, numpy_function, fill_value, marker in operations:
                    source_value = np.full(
                        (2, 5, 5), fill_value, dtype=dtype
                    )
                    source_value[1, :, :] = marker
                    numpy_view = source_value.transpose(2, 0, 1)[
                        :, ::-1, ::-2
                    ]
                    with self.subTest(
                        dtype=np.dtype(dtype).name,
                        operation=method_name,
                    ):
                        source = stack.enter_context(
                            runtime.from_numpy(source_value)
                        )
                        transposed = stack.enter_context(
                            runtime.transpose(source, (2, 0, 1))
                        )
                        view = stack.enter_context(
                            self.slice_view(
                                runtime,
                                transposed,
                                (
                                    _CnpSlice(
                                        0, 0, 1, False, False, True
                                    ),
                                    _CnpSlice(
                                        0, 0, -1, False, False, True
                                    ),
                                    _CnpSlice(
                                        0, 0, -2, False, False, True
                                    ),
                                ),
                            )
                        )
                        result = stack.enter_context(
                            getattr(runtime, method_name)(view, 1)
                        )
                        assert_array_equivalent(
                            self,
                            result,
                            numpy_function(numpy_view, axis=1),
                            compare_contiguity=False,
                        )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_nan_extrema_vector_lane_reduction_matches_numpy_tree(
        self,
    ) -> None:
        sources = (
            np.array(
                [
                    np.nan,
                    0.0,
                    -0.0,
                    np.nan,
                    0.0,
                    np.nan,
                    0.0,
                    -0.0,
                    np.nan,
                    0.0,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    np.nan,
                    -0.0,
                    0.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    -0.0,
                    0.0,
                    np.nan,
                    np.nan,
                ],
                dtype=np.float64,
            ),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for source_value in sources:
                for method_name, numpy_function, value in (
                    ("nanmin", np.nanmin, source_value),
                    ("nanmax", np.nanmax, -source_value),
                ):
                    with self.subTest(
                        dtype=str(source_value.dtype),
                        operation=method_name,
                    ):
                        source = stack.enter_context(runtime.from_numpy(value))
                        result = stack.enter_context(
                            getattr(runtime, method_name)(source, None)
                        )
                        assert_array_equivalent(
                            self,
                            result,
                            numpy_function(value),
                            compare_contiguity=False,
                        )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_nan_extrema_negative_output_stride_matches_numpy_avx2_loop(
        self,
    ) -> None:
        operations = (
            ("nanmin", np.nanmin, 0.0, -0.0),
            ("nanmax", np.nanmax, -0.0, 0.0),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for dtype in (np.float32, np.float64):
                for method_name, numpy_function, fill_value, marker in operations:
                    source_value = np.full((6, 21), fill_value, dtype=dtype)
                    source_value[5, :] = marker
                    numpy_view = source_value.T[::-1, ::-2]
                    with self.subTest(
                        dtype=np.dtype(dtype).name,
                        operation=method_name,
                    ):
                        source = stack.enter_context(
                            runtime.from_numpy(source_value)
                        )
                        transposed = stack.enter_context(
                            runtime.transpose(source, (1, 0))
                        )
                        view = stack.enter_context(
                            self.slice_view(
                                runtime,
                                transposed,
                                (
                                    _CnpSlice(
                                        0, 0, -1, False, False, True
                                    ),
                                    _CnpSlice(
                                        0, 0, -2, False, False, True
                                    ),
                                ),
                            )
                        )
                        result = stack.enter_context(
                            getattr(runtime, method_name)(view, 1)
                        )
                        assert_array_equivalent(
                            self,
                            result,
                            numpy_function(numpy_view, axis=1),
                            compare_contiguity=False,
                        )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_nan_extrema_buffers_discontiguous_pre_reduction_run(
        self,
    ) -> None:
        operations = (
            ("nanmin", np.nanmin, (0.0, -0.0, 0.0)),
            ("nanmax", np.nanmax, (-0.0, 0.0, -0.0)),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for method_name, numpy_function, values in operations:
                source_value = np.repeat(
                    np.asarray(values, dtype=np.float32).reshape(3, 1, 1),
                    2,
                    axis=2,
                )
                numpy_view = np.broadcast_to(source_value, (3, 4, 2))[
                    :, :, ::-1
                ]
                with self.subTest(operation=method_name):
                    source = stack.enter_context(
                        runtime.from_numpy(source_value)
                    )
                    broadcast = stack.enter_context(
                        self.broadcast_view(runtime, source, (3, 4, 2))
                    )
                    view = stack.enter_context(
                        self.slice_view(
                            runtime,
                            broadcast,
                            (
                                _CnpSlice(0, 0, 1, False, False, True),
                                _CnpSlice(0, 0, 1, False, False, True),
                                _CnpSlice(0, 0, -1, False, False, True),
                            ),
                        )
                    )
                    result = stack.enter_context(
                        getattr(runtime, method_name)(view, 0)
                    )
                    assert_array_equivalent(
                        self,
                        result,
                        numpy_function(numpy_view, axis=0),
                        compare_contiguity=False,
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_nan_extrema_buffers_noncoalesced_horizontal_axis(
        self,
    ) -> None:
        operations = (
            ("nanmin", np.nanmin, 0.0, -0.0),
            ("nanmax", np.nanmax, -0.0, 0.0),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for method_name, numpy_function, fill_value, marker in operations:
                source_value = np.full(
                    (4, 4, 12), fill_value, dtype=np.float64
                )
                source_value[:, :, 11] = marker
                numpy_view = source_value.transpose(2, 1, 0)[
                    11:1:-2, ::2, ::2
                ]
                with self.subTest(operation=method_name):
                    source = stack.enter_context(
                        runtime.from_numpy(source_value)
                    )
                    transposed = stack.enter_context(
                            runtime.transpose(source, (2, 1, 0))
                    )
                    view = stack.enter_context(
                        self.slice_view(
                            runtime,
                            transposed,
                            (
                                _CnpSlice(11, 1, -2, True, True, True),
                                _CnpSlice(0, 0, 2, False, False, True),
                                _CnpSlice(0, 0, 2, False, False, True),
                            ),
                        )
                    )
                    result = stack.enter_context(
                        getattr(runtime, method_name)(view, 0)
                    )
                    assert_array_equivalent(
                        self,
                        result,
                        numpy_function(numpy_view, axis=0),
                        compare_contiguity=False,
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_nan_extrema_follow_numpy_coalesced_output_run_order(self) -> None:
        run_six_value = np.zeros((3, 2, 2), dtype=np.float64)
        run_six_value[:, 0, :] = -0.0
        run_three_value = np.zeros((3, 2, 2), dtype=np.float64)
        run_three_value[:, :, 0] = -0.0
        cases = (
            ("coalesced_run_6", run_six_value, 1),
            ("separate_runs_3", run_three_value, 2),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for layout, logical_value, axis in cases:
                stored_value = logical_value.transpose(1, 2, 0).copy()
                numpy_view = stored_value.transpose(2, 0, 1)
                stored = stack.enter_context(
                    runtime.from_numpy(stored_value)
                )
                view = stack.enter_context(runtime.transpose(stored, (2, 0, 1)))
                with self.subTest(layout=layout, axis=axis):
                    result = stack.enter_context(runtime.nanmin(view, axis))
                    assert_array_equivalent(
                        self,
                        result,
                        np.nanmin(numpy_view, axis=axis),
                        compare_contiguity=False,
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_ptp_preserves_int64_and_uint64_precision_across_views(self) -> None:
        cases = (
            np.array([2**53 + 1, 2**53 + 3], dtype=np.int64),
            np.array([2**63 + 1, 2**63 + 3], dtype=np.uint64),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for source_value in cases:
                with self.subTest(dtype=str(source_value.dtype), layout="flat"):
                    source = stack.enter_context(
                        runtime.from_numpy(source_value)
                    )
                    result = stack.enter_context(runtime.ptp(source))
                    assert_array_equivalent(
                        self,
                        result,
                        np.ptp(source_value),
                        compare_contiguity=False,
                    )

                logical_value = np.stack(
                    (source_value, source_value + np.array([4, 6], dtype=source_value.dtype))
                )
                stored_value = logical_value.T.copy()
                stored = stack.enter_context(runtime.from_numpy(stored_value))
                view = stack.enter_context(runtime.transpose(stored, (1, 0)))
                with self.subTest(dtype=str(source_value.dtype), layout="view"):
                    result = stack.enter_context(runtime.ptp(view, 1))
                    assert_array_equivalent(
                        self,
                        result,
                        np.ptp(logical_value, axis=1),
                        compare_contiguity=False,
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_percentile_families_match_numpy_infinity_interpolation(self) -> None:
        cases = (
            np.array([1.0, np.inf], dtype=np.float64),
            np.array([-np.inf, 1.0], dtype=np.float64),
            np.array([-np.inf, np.inf], dtype=np.float64),
            np.array([1.0, 3.0], dtype=np.float64),
        )
        operations = (
            ("percentile", np.percentile, 1.0),
            ("nanpercentile", np.nanpercentile, 1.0),
            ("quantile", np.quantile, 100.0),
            ("nanquantile", np.nanquantile, 100.0),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for source_value in cases:
                for method_name, numpy_function, scale in operations:
                    operand_value = (
                        np.append(source_value, np.nan)
                        if method_name.startswith("nan") else source_value
                    )
                    source = stack.enter_context(
                        runtime.from_numpy(operand_value)
                    )
                    for percentile in (0.0, 25.0, 50.0, 75.0, 100.0):
                        q = percentile / scale
                        with self.subTest(
                            operation=method_name,
                            source=source_value.tolist(),
                            q=q,
                        ):
                            with warnings.catch_warnings(), np.errstate(
                                invalid="ignore", divide="ignore"
                            ):
                                warnings.simplefilter("ignore")
                                expected = numpy_function(operand_value, q)
                            result = stack.enter_context(
                                getattr(runtime, method_name)(source, q)
                            )
                            assert_array_equivalent(
                                self,
                                result,
                                expected,
                                compare_contiguity=False,
                            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_every_reduction_family_covers_rank_zero_through_four_and_all_axes(self) -> None:
        shapes = ((), (2,), (2, 3), (2, 2, 3), (2, 2, 2, 3))
        operations = (
            ("sum", np.sum, None), ("prod", np.prod, None),
            ("mean", np.mean, None), ("variance", np.var, None),
            ("std", np.std, None), ("maximum", np.max, None),
            ("minimum", np.min, None), ("argmax", np.argmax, None),
            ("argmin", np.argmin, None), ("any", np.any, None),
            ("all", np.all, None), ("ptp", np.ptp, None),
            ("nansum", np.nansum, None),
            ("nanprod", np.nanprod, None),
            ("nanmean", np.nanmean, None),
            ("nanvar", np.nanvar, None),
            ("nanstd", np.nanstd, None),
            ("nanmax", np.nanmax, None),
            ("nanmin", np.nanmin, None),
            ("nanargmax", np.nanargmax, None),
            ("nanargmin", np.nanargmin, None),
            ("median", np.median, None),
            ("nanmedian", np.nanmedian, None),
            ("percentile", np.percentile, 37.5),
            ("nanpercentile", np.nanpercentile, 37.5),
            ("quantile", np.quantile, 0.375),
            ("nanquantile", np.nanquantile, 0.375),
            ("cumsum", np.cumsum, None),
            ("cumprod", np.cumprod, None),
            ("nancumsum", np.nancumsum, None),
            ("nancumprod", np.nancumprod, None),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for shape in shapes:
                size = int(np.prod(shape)) if shape else 1
                value = (
                    np.array(3.0, dtype=np.float64) if not shape else
                    (np.arange(size, dtype=np.float64) + 1.0).reshape(shape)
                )
                source = stack.enter_context(runtime.from_numpy(value))
                axes = (None, 0, -1) if not shape else (
                    (None,) + tuple(range(len(shape))) +
                    tuple(range(-1, -len(shape) - 1, -1))
                )
                for method_name, numpy_function, q in operations:
                    method = getattr(runtime, method_name)
                    for axis in axes:
                        with self.subTest(
                            rank=len(shape), operation=method_name, axis=axis
                        ):
                            try:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    expected = (
                                        numpy_function(value, axis=axis)
                                        if q is None else
                                        numpy_function(value, q, axis=axis)
                                    )
                            except (IndexError, ValueError):
                                before_error = runtime.retained_bytes
                                try:
                                    unexpected = (
                                        method(source, axis) if q is None else
                                        method(source, q, axis)
                                    )
                                except CnumpyError:
                                    pass
                                else:
                                    unexpected.close()
                                    self.fail("CnumpyError not raised")
                                self.assertEqual(
                                    before_error, runtime.retained_bytes
                                )
                            else:
                                result = stack.enter_context(
                                    method(source, axis) if q is None else
                                    method(source, q, axis)
                                )
                                assert_array_equivalent(
                                    self, result, expected,
                                    compare_contiguity=False,
                                    rtol=16 * np.finfo(np.float64).eps,
                                )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_empty_dimensions_follow_numpy_results_and_explicit_errors(self) -> None:
        shapes = (
            (0,), (2, 0), (0, 3), (2, 0, 3), (2, 3, 0),
            (1, 0, 2, 3), (1, 2, 0, 3), (1, 2, 3, 0),
        )
        operations = (
            ("sum", np.sum, None), ("prod", np.prod, None),
            ("mean", np.mean, None), ("variance", np.var, None),
            ("std", np.std, None), ("maximum", np.max, None),
            ("minimum", np.min, None), ("argmax", np.argmax, None),
            ("argmin", np.argmin, None), ("any", np.any, None),
            ("all", np.all, None), ("ptp", np.ptp, None),
            ("nansum", np.nansum, None), ("nanprod", np.nanprod, None),
            ("nanmean", np.nanmean, None), ("nanvar", np.nanvar, None),
            ("nanstd", np.nanstd, None), ("nanmax", np.nanmax, None),
            ("nanmin", np.nanmin, None),
            ("nanargmax", np.nanargmax, None),
            ("nanargmin", np.nanargmin, None),
            ("median", np.median, None), ("nanmedian", np.nanmedian, None),
            ("percentile", np.percentile, 37.5),
            ("nanpercentile", np.nanpercentile, 37.5),
            ("quantile", np.quantile, 0.375),
            ("nanquantile", np.nanquantile, 0.375),
            ("cumsum", np.cumsum, None), ("cumprod", np.cumprod, None),
            ("nancumsum", np.nancumsum, None),
            ("nancumprod", np.nancumprod, None),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for shape in shapes:
                value = np.empty(shape, dtype=np.float64)
                source = stack.enter_context(runtime.from_numpy(value))
                axes = ((None,) + tuple(range(len(shape))) +
                        tuple(range(-1, -len(shape) - 1, -1)))
                for method_name, numpy_function, q in operations:
                    method = getattr(runtime, method_name)
                    for axis in axes:
                        with self.subTest(
                            shape=shape, operation=method_name, axis=axis
                        ):
                            try:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    expected = (
                                        numpy_function(value, axis=axis)
                                        if q is None else
                                        numpy_function(value, q, axis=axis)
                                    )
                            except (IndexError, ValueError):
                                before_error = runtime.retained_bytes
                                with self.assertRaises(CnumpyError):
                                    if q is None:
                                        method(source, axis)
                                    else:
                                        method(source, q, axis)
                                self.assertEqual(before_error, runtime.retained_bytes)
                            else:
                                result = stack.enter_context(
                                    method(source, axis) if q is None else
                                    method(source, q, axis)
                                )
                                assert_array_equivalent(
                                    self, result, expected,
                                    compare_contiguity=False,
                                    rtol=16 * np.finfo(np.float64).eps,
                                )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
