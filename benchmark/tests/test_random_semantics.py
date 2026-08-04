from __future__ import annotations

import ctypes
import unittest
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from compat.cnumpy_ctypes import CNP_MAXDIMS, CnumpyError, CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"
UINT64_MASK = (1 << 64) - 1


class _CnpSlice(ctypes.Structure):
    _fields_ = (
        ("start", ctypes.c_int64),
        ("stop", ctypes.c_int64),
        ("step", ctypes.c_int64),
        ("has_start", ctypes.c_bool),
        ("has_stop", ctypes.c_bool),
        ("has_step", ctypes.c_bool),
    )


def xoshiro256ss_indices(seed: int, bound: int, count: int) -> np.ndarray:
    splitmix_state = seed & UINT64_MASK
    state: list[int] = []
    for _ in range(4):
        splitmix_state = (
            splitmix_state + 0x9E3779B97F4A7C15
        ) & UINT64_MASK
        value = splitmix_state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
        state.append((value ^ (value >> 31)) & UINT64_MASK)

    def rotate_left(value: int, shift: int) -> int:
        return (
            ((value << shift) & UINT64_MASK) | (value >> (64 - shift))
        )

    def next_uint64() -> int:
        result = (rotate_left((state[1] * 5) & UINT64_MASK, 7) * 9) & UINT64_MASK
        temporary = (state[1] << 17) & UINT64_MASK
        state[2] ^= state[0]
        state[3] ^= state[1]
        state[1] ^= state[2]
        state[0] ^= state[3]
        state[2] ^= temporary
        state[3] = rotate_left(state[3], 45)
        return result

    threshold = ((-bound) & UINT64_MASK) % bound
    result: list[int] = []
    while len(result) < count:
        value = next_uint64()
        if value >= threshold:
            result.append(value % bound)
    return np.asarray(result, dtype=np.int64)


class RandomChoiceSemanticsTests(unittest.TestCase):
    def slice_view(self, runtime: CnumpyRuntime, source, *, step: int):
        function = runtime.dll.cnp_array_slice
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(_CnpSlice),
        ]
        function.restype = ctypes.c_void_p
        slices = (_CnpSlice * 1)(
            _CnpSlice(0, 0, step, False, False, True)
        )
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(
            function(source.pointer, 1, slices), "cnp_array_slice"
        )

    def choice_v2(
        self,
        runtime: CnumpyRuntime,
        population,
        *,
        size: tuple[int, ...] | None,
        replace: bool,
        probabilities=None,
    ):
        function = runtime.dll.cnp_random_choice_v2
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_bool,
            ctypes.c_bool,
            ctypes.c_void_p,
        ]
        function.restype = ctypes.c_void_p
        shape = () if size is None else size
        shape_storage = (
            (ctypes.c_int64 * len(shape))(*shape) if shape else None
        )
        runtime.dll.cnp_clear_error()
        pointer = function(
            population.pointer,
            len(shape),
            shape_storage,
            size is None,
            replace,
            probabilities.pointer if probabilities is not None else None,
        )
        return runtime._owned_result(pointer, "cnp_random_choice_v2")

    def test_seeded_unweighted_choice_preserves_shape_dtype_and_sequence(
        self,
    ) -> None:
        seed = 0x7EDCBA9876543210
        values = np.array(
            [2**60 + 3, -(2**60) + 7, 17, 99, -5, 42, 8],
            dtype=np.int64,
        )
        expected_indices = xoshiro256ss_indices(seed, len(values), 12)
        expected = values[expected_indices].reshape(3, 4)

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            population = stack.enter_context(runtime.from_numpy(values))
            runtime.dll.cnp_random_seed.argtypes = [ctypes.c_uint64]
            runtime.dll.cnp_random_seed.restype = None
            runtime.dll.cnp_random_seed(seed)
            actual = stack.enter_context(
                self.choice_v2(
                    runtime,
                    population,
                    size=(3, 4),
                    replace=True,
                )
            )

            self.assertEqual((3, 4), actual.shape)
            self.assertEqual(np.dtype(np.int64), actual.numpy_dtype)
            np.testing.assert_array_equal(expected, actual.to_numpy())

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_weighted_choice_preserves_complex_values_and_shape(self) -> None:
        values = np.array(
            [1.0 + 2.0j, -3.5 + 0.25j, 7.0 - 9.0j],
            dtype=np.complex128,
        )
        weights = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            population = stack.enter_context(runtime.from_numpy(values))
            probabilities = stack.enter_context(runtime.from_numpy(weights))
            actual = stack.enter_context(
                self.choice_v2(
                    runtime,
                    population,
                    size=(2, 3),
                    replace=True,
                    probabilities=probabilities,
                )
            )

            self.assertEqual(np.dtype(np.complex128), actual.numpy_dtype)
            np.testing.assert_array_equal(
                np.full((2, 3), values[1], dtype=np.complex128),
                actual.to_numpy(),
            )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_weighted_replacement_matches_requested_distribution(self) -> None:
        values = np.array([10, 20, 30], dtype=np.int32)
        weights = np.array([0.05, 0.15, 0.80], dtype=np.float64)
        sample_count = 100_000
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            population = stack.enter_context(runtime.from_numpy(values))
            probabilities = stack.enter_context(runtime.from_numpy(weights))
            runtime.dll.cnp_random_seed.argtypes = [ctypes.c_uint64]
            runtime.dll.cnp_random_seed(20260730)
            actual = stack.enter_context(
                self.choice_v2(
                    runtime,
                    population,
                    size=(sample_count,),
                    replace=True,
                    probabilities=probabilities,
                )
            )

            observed = actual.to_numpy()
            frequencies = np.array(
                [np.mean(observed == value) for value in values]
            )
            np.testing.assert_allclose(frequencies, weights, rtol=0, atol=0.006)
            np.testing.assert_array_equal(weights, probabilities.to_numpy())

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_numpy_125_differential_shape_dtype_replace_and_probability_dtypes(
        self,
    ) -> None:
        self.assertEqual("1.25.0", np.__version__)
        cases = (
            (
                np.array([11, 22, 33], dtype=np.int64),
                None,
                True,
                np.array([False, True, False], dtype=np.bool_),
            ),
            (
                np.array([1.5, 2.5, 3.5], dtype=np.float16),
                (),
                True,
                np.array([0, 0, 1], dtype=np.int64),
            ),
            (
                np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
                (2, 3),
                True,
                np.array([1, 0, 0], dtype=np.float16),
            ),
            (
                np.array([7, 8, 9], dtype=np.int16),
                (1,),
                False,
                np.array([0, 1, 0], dtype=np.float32),
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for values, size, replace, weights in cases:
                with self.subTest(
                    population_dtype=values.dtype,
                    probability_dtype=weights.dtype,
                    size=size,
                    replace=replace,
                ), ExitStack() as stack:
                    numpy_result = np.random.RandomState(20260803).choice(
                        values, size=size, replace=replace, p=weights
                    )
                    population = stack.enter_context(
                        runtime.from_numpy(values)
                    )
                    probabilities = stack.enter_context(
                        runtime.from_numpy(weights)
                    )
                    actual = stack.enter_context(
                        self.choice_v2(
                            runtime,
                            population,
                            size=size,
                            replace=replace,
                            probabilities=probabilities,
                        )
                    )
                    self.assertEqual(np.asarray(numpy_result).shape, actual.shape)
                    self.assertEqual(np.asarray(numpy_result).dtype, actual.numpy_dtype)
                    np.testing.assert_array_equal(
                        np.asarray(numpy_result), actual.to_numpy()
                    )
                self.assertEqual(baseline, runtime.retained_bytes)

    def test_numpy_125_differential_noncontiguous_and_negative_stride_inputs(
        self,
    ) -> None:
        numpy_population_owner = np.array(
            [10, -1, 20, -1, 30, -1], dtype=np.int32
        )
        numpy_probability_owner = np.array(
            [1.0, 0.0, 0.0], dtype=np.float64
        )
        numpy_population = numpy_population_owner[::2]
        numpy_probabilities = numpy_probability_owner[::-1]
        expected = np.random.RandomState(7).choice(
            numpy_population,
            size=(2, 4),
            replace=True,
            p=numpy_probabilities,
        )

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            population_owner = stack.enter_context(
                runtime.from_numpy(numpy_population_owner)
            )
            probability_owner = stack.enter_context(
                runtime.from_numpy(numpy_probability_owner)
            )
            population = stack.enter_context(
                self.slice_view(runtime, population_owner, step=2)
            )
            probabilities = stack.enter_context(
                self.slice_view(runtime, probability_owner, step=-1)
            )
            self.assertFalse(population.c_contiguous)
            self.assertLess(probabilities.strides[0], 0)

            actual = stack.enter_context(
                self.choice_v2(
                    runtime,
                    population,
                    size=(2, 4),
                    replace=True,
                    probabilities=probabilities,
                )
            )
            self.assertEqual(expected.shape, actual.shape)
            self.assertEqual(expected.dtype, actual.numpy_dtype)
            np.testing.assert_array_equal(expected, actual.to_numpy())

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_numpy_125_differential_rejections_are_atomic(self) -> None:
        values = np.array([10, 20, 30], dtype=np.int64)
        probability_cases = {
            "length mismatch": np.array([0.5, 0.5], dtype=np.float64),
            "negative": np.array([0.5, -0.1, 0.6], dtype=np.float64),
            "nan": np.array([0.5, np.nan, 0.5], dtype=np.float64),
            "infinite": np.array([0.5, np.inf, 0.5], dtype=np.float64),
            "all zero": np.zeros(3, dtype=np.float64),
        }
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            population = stack.enter_context(runtime.from_numpy(values))
            for name, weights in probability_cases.items():
                with self.subTest(case=name):
                    with self.assertRaises(ValueError):
                        np.random.RandomState(7).choice(
                            values, size=3, replace=True, p=weights
                        )
                    probabilities = stack.enter_context(
                        runtime.from_numpy(weights)
                    )
                    before = runtime.retained_bytes
                    with self.assertRaises(CnumpyError):
                        self.choice_v2(
                            runtime,
                            population,
                            size=(3,),
                            replace=True,
                            probabilities=probabilities,
                        )
                    self.assertEqual(before, runtime.retained_bytes)

            impossible_cases = (
                (4, False, None),
                (3, False, np.array([0.5, 0.5, 0.0], dtype=np.float64)),
            )
            for count, replace, weights in impossible_cases:
                with self.subTest(
                    case="impossible sample", count=count, weighted=weights is not None
                ):
                    with self.assertRaises(ValueError):
                        np.random.RandomState(7).choice(
                            values, size=count, replace=replace, p=weights
                        )
                    probabilities = (
                        stack.enter_context(runtime.from_numpy(weights))
                        if weights is not None
                        else None
                    )
                    before = runtime.retained_bytes
                    with self.assertRaises(CnumpyError):
                        self.choice_v2(
                            runtime,
                            population,
                            size=(count,),
                            replace=replace,
                            probabilities=probabilities,
                        )
                    self.assertEqual(before, runtime.retained_bytes)

            with self.assertRaises(ValueError):
                np.random.RandomState(7).choice(values, size=-1)
            before = runtime.retained_bytes
            with self.assertRaises(CnumpyError):
                self.choice_v2(
                    runtime, population, size=(-1,), replace=True
                )
            self.assertEqual(before, runtime.retained_bytes)

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_choice_v2_raw_size_contract_rejections_are_atomic(self) -> None:
        values = np.array([10, 20, 30], dtype=np.int64)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            population = stack.enter_context(runtime.from_numpy(values))
            function = runtime.dll.cnp_random_choice_v2
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_bool,
                ctypes.c_bool,
                ctypes.c_void_p,
            ]
            function.restype = ctypes.c_void_p
            one = (ctypes.c_int64 * 1)(1)
            too_many_dimensions = (ctypes.c_int64 * (CNP_MAXDIMS + 1))(
                *([1] * (CNP_MAXDIMS + 1))
            )
            cases = (
                (1, one, True, "size=None"),
                (0, one, True, "size=None"),
                (1, None, False, "shape"),
                (
                    CNP_MAXDIMS + 1,
                    too_many_dimensions,
                    False,
                    "dimensions",
                ),
            )
            for ndim, shape, size_none, message in cases:
                with self.subTest(
                    ndim=ndim, shape_is_null=shape is None, size_none=size_none
                ):
                    runtime.dll.cnp_clear_error()
                    before = runtime.retained_bytes
                    pointer = function(
                        population.pointer,
                        ndim,
                        shape,
                        size_none,
                        True,
                        None,
                    )
                    self.assertFalse(pointer)
                    self.assertEqual(before, runtime.retained_bytes)
                    error = runtime.error_state()
                    self.assertEqual(-4, error.status)
                    self.assertIn(message.lower(), error.message.lower())

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_probability_sum_tolerance_matches_numpy_125_by_dtype(self) -> None:
        values = np.array([10, 20, 30], dtype=np.int64)
        cases = (
            (np.float64, 1e-9, True),
            (np.float64, 1e-7, False),
            (np.float32, 1e-4, True),
            (np.float32, 4e-4, False),
        )
        for dtype, delta, expected_acceptance in cases:
            with self.subTest(dtype=dtype, delta=delta):
                weights = np.array(
                    [0.2, 0.3, 0.5 + delta], dtype=dtype
                )
                try:
                    np.random.RandomState(7).choice(
                        values, size=8, replace=True, p=weights
                    )
                    numpy_accepted = True
                except ValueError:
                    numpy_accepted = False
                self.assertEqual(expected_acceptance, numpy_accepted)

                with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
                    baseline = runtime.retained_bytes
                    population = stack.enter_context(
                        runtime.from_numpy(values)
                    )
                    probabilities = stack.enter_context(
                        runtime.from_numpy(weights)
                    )
                    before_choice = runtime.retained_bytes
                    try:
                        result = self.choice_v2(
                            runtime,
                            population,
                            size=(8,),
                            replace=True,
                            probabilities=probabilities,
                        )
                    except CnumpyError as error:
                        cnumpy_accepted = False
                        self.assertEqual(
                            "cnp_random_choice_v2", error.function
                        )
                        self.assertIn("sum to 1", error.message)
                        self.assertEqual(before_choice, runtime.retained_bytes)
                    else:
                        cnumpy_accepted = True
                        stack.enter_context(result)

                    self.assertEqual(numpy_accepted, cnumpy_accepted)
                    stack.close()
                    self.assertEqual(baseline, runtime.retained_bytes)

    def test_weighted_without_replacement_is_unique_and_replayable(self) -> None:
        values = np.arange(5, dtype=np.int16)
        weights = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            population = stack.enter_context(runtime.from_numpy(values))
            probabilities = stack.enter_context(runtime.from_numpy(weights))
            runtime.dll.cnp_random_seed.argtypes = [ctypes.c_uint64]
            runtime.dll.cnp_random_seed(314159)
            first = stack.enter_context(
                self.choice_v2(
                    runtime,
                    population,
                    size=(4,),
                    replace=False,
                    probabilities=probabilities,
                )
            )
            runtime.dll.cnp_random_seed(314159)
            second = stack.enter_context(
                self.choice_v2(
                    runtime,
                    population,
                    size=(4,),
                    replace=False,
                    probabilities=probabilities,
                )
            )

            first_values = first.to_numpy()
            self.assertEqual(4, np.unique(first_values).size)
            self.assertNotIn(0, first_values)
            np.testing.assert_array_equal(first_values, second.to_numpy())

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_weighted_without_replacement_non_degenerate_distribution_matches_numpy_125(
        self,
    ) -> None:
        values = np.arange(5, dtype=np.int64)
        weights = np.array([0.02, 0.08, 0.15, 0.25, 0.50], dtype=np.float64)
        sample_size = 2
        trial_count = 4096
        numpy_first = np.zeros(values.size, dtype=np.int64)
        numpy_included = np.zeros(values.size, dtype=np.int64)
        cnumpy_first = np.zeros(values.size, dtype=np.int64)
        cnumpy_included = np.zeros(values.size, dtype=np.int64)

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            population = stack.enter_context(runtime.from_numpy(values))
            probabilities = stack.enter_context(runtime.from_numpy(weights))
            runtime.dll.cnp_random_seed.argtypes = [ctypes.c_uint64]
            runtime.dll.cnp_random_seed.restype = None

            for seed in range(trial_count):
                numpy_result = np.random.RandomState(seed).choice(
                    values,
                    size=sample_size,
                    replace=False,
                    p=weights,
                )
                numpy_first[numpy_result[0]] += 1
                numpy_included[numpy_result] += 1

                runtime.dll.cnp_random_seed(seed)
                with self.choice_v2(
                    runtime,
                    population,
                    size=(sample_size,),
                    replace=False,
                    probabilities=probabilities,
                ) as actual:
                    cnumpy_result = actual.to_numpy()
                cnumpy_first[cnumpy_result[0]] += 1
                cnumpy_included[cnumpy_result] += 1

            numpy_first_frequency = numpy_first / trial_count
            cnumpy_first_frequency = cnumpy_first / trial_count
            numpy_inclusion_frequency = numpy_included / trial_count
            cnumpy_inclusion_frequency = cnumpy_included / trial_count
            expected_inclusion = weights + weights * np.array(
                [
                    sum(
                        weights[other] / (1.0 - weights[other])
                        for other in range(weights.size)
                        if other != index
                    )
                    for index in range(weights.size)
                ],
                dtype=np.float64,
            )

            np.testing.assert_allclose(
                numpy_first_frequency, weights, rtol=0, atol=0.025
            )
            np.testing.assert_allclose(
                cnumpy_first_frequency, weights, rtol=0, atol=0.025
            )
            np.testing.assert_allclose(
                cnumpy_first_frequency,
                numpy_first_frequency,
                rtol=0,
                atol=0.035,
            )
            np.testing.assert_allclose(
                numpy_inclusion_frequency,
                expected_inclusion,
                rtol=0,
                atol=0.035,
            )
            np.testing.assert_allclose(
                cnumpy_inclusion_frequency,
                expected_inclusion,
                rtol=0,
                atol=0.035,
            )
            np.testing.assert_allclose(
                cnumpy_inclusion_frequency,
                numpy_inclusion_frequency,
                rtol=0,
                atol=0.045,
            )
            self.assertGreater(
                cnumpy_first_frequency[-1],
                cnumpy_first_frequency[-2] + 0.15,
                "weighted sampling must not degrade to uniform first draws",
            )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_scalar_and_empty_shapes_follow_numpy_size_semantics(self) -> None:
        values = np.array([4.0, 8.0], dtype=np.float32)
        empty = np.array([], dtype=np.float64)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            population = stack.enter_context(runtime.from_numpy(values))
            empty_population = stack.enter_context(runtime.from_numpy(empty))
            scalar = stack.enter_context(
                self.choice_v2(
                    runtime, population, size=None, replace=True
                )
            )
            zero_dimensional = stack.enter_context(
                self.choice_v2(runtime, population, size=(), replace=True)
            )
            empty_result = stack.enter_context(
                self.choice_v2(
                    runtime, empty_population, size=(0, 2), replace=False
                )
            )

            self.assertEqual((), scalar.shape)
            self.assertEqual((), zero_dimensional.shape)
            self.assertEqual((0, 2), empty_result.shape)
            self.assertIn(float(scalar.to_numpy()), values)
            self.assertIn(float(zero_dimensional.to_numpy()), values)

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_validation_failures_are_explicit_and_allocation_atomic(self) -> None:
        values = np.array([10, 20, 30], dtype=np.int64)
        matrix_values = values.reshape(1, 3)
        empty_values = np.array([], dtype=np.int64)
        probability_values = {
            "length": np.array([0.5, 0.5], dtype=np.float64),
            "matrix": np.array([[0.2, 0.3, 0.5]], dtype=np.float64),
            "negative": np.array([0.5, -0.1, 0.6], dtype=np.float64),
            "nan": np.array([0.5, np.nan, 0.5], dtype=np.float64),
            "infinite": np.array([0.5, np.inf, 0.5], dtype=np.float64),
            "zero": np.zeros(3, dtype=np.float64),
            "sum": np.ones(3, dtype=np.float64),
            "few": np.array([0.0, 0.4, 0.6], dtype=np.float64),
            "complex": np.array([0.2, 0.3, 0.5], dtype=np.complex128),
        }
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            population = stack.enter_context(runtime.from_numpy(values))
            matrix = stack.enter_context(runtime.from_numpy(matrix_values))
            empty = stack.enter_context(runtime.from_numpy(empty_values))
            probabilities = {
                name: stack.enter_context(runtime.from_numpy(value))
                for name, value in probability_values.items()
            }
            function = runtime.dll.cnp_random_choice_v2
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_bool,
                ctypes.c_bool,
                ctypes.c_void_p,
            ]
            function.restype = ctypes.c_void_p

            def invoke(source, count: int, replace: bool, probability=None):
                shape = (ctypes.c_int64 * 1)(count)
                runtime.dll.cnp_clear_error()
                before = runtime.retained_bytes
                pointer = function(
                    source,
                    1,
                    shape,
                    False,
                    replace,
                    probability.pointer if probability is not None else None,
                )
                self.assertFalse(pointer)
                self.assertEqual(before, runtime.retained_bytes)
                error = runtime.error_state()
                self.assertEqual("cnp_random_choice_v2", error.function)
                return error

            cases = (
                (None, 1, True, None, -1, "population"),
                (matrix.pointer, 1, True, None, -4, "one-dimensional"),
                (empty.pointer, 1, True, None, -4, "empty"),
                (population.pointer, 4, False, None, -4, "larger"),
                (population.pointer, 1, True, probabilities["length"], -4, "length"),
                (population.pointer, 1, True, probabilities["matrix"], -4, "one-dimensional"),
                (population.pointer, 1, True, probabilities["negative"], -1, "negative"),
                (population.pointer, 1, True, probabilities["nan"], -1, "finite"),
                (population.pointer, 1, True, probabilities["infinite"], -1, "finite"),
                (population.pointer, 1, True, probabilities["zero"], -1, "zero"),
                (population.pointer, 1, True, probabilities["sum"], -1, "sum"),
                (population.pointer, 3, False, probabilities["few"], -4, "non-zero"),
                (population.pointer, 1, True, probabilities["complex"], -3, "real"),
            )
            for source, count, replace, probability, status, message in cases:
                with self.subTest(message=message):
                    error = invoke(source, count, replace, probability)
                    self.assertEqual(status, error.status)
                    self.assertIn(message, error.message.lower())

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_choice_uses_weighted_semantics(self) -> None:
        values = np.array([11, 22, 33], dtype=np.int32)
        weights = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            population = stack.enter_context(runtime.from_numpy(values))
            probabilities = stack.enter_context(runtime.from_numpy(weights))
            function = runtime.dll.cnp_random_choice
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_bool,
                ctypes.c_void_p,
            ]
            function.restype = ctypes.c_void_p
            runtime.dll.cnp_clear_error()
            result = stack.enter_context(
                runtime._owned_result(
                    function(
                        population.pointer,
                        8,
                        True,
                        probabilities.pointer,
                    ),
                    "cnp_random_choice",
                )
            )
            np.testing.assert_array_equal(
                np.full(8, 33, dtype=np.int32), result.to_numpy()
            )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
