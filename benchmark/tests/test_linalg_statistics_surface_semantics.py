from __future__ import annotations

import ctypes
import unittest
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime
from compat.oracle import assert_array_equivalent


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"


class LinalgStatisticsSurfaceTests(unittest.TestCase):
    def _array_function(
        self, runtime: CnumpyRuntime, name: str, argtypes: list[object]
    ) -> ctypes._CFuncPtr:
        function = getattr(runtime.dll, name)
        function.argtypes = argtypes
        function.restype = ctypes.c_void_p
        return function

    def _array_result(
        self,
        runtime: CnumpyRuntime,
        function: ctypes._CFuncPtr,
        name: str,
        *arguments: object,
    ):
        runtime.dll.cnp_clear_error()
        return runtime._owned_result(function(*arguments), name)

    def test_inv_and_pinv_aliases_match_numpy_batches_dtypes_and_complex(
        self,
    ) -> None:
        inverse_cases = (
            np.asarray(
                [
                    [[2.0, 0.0], [0.0, 4.0]],
                    [[1.0, 2.0], [3.0, 5.0]],
                ],
                dtype=np.float32,
            ),
            np.asarray(
                [[2 + 1j, 1 - 2j], [3 + 0.5j, 5 - 1j]],
                dtype=np.complex128,
            ),
        )
        pinv_cases = (
            np.asarray(
                [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]],
                dtype=np.float64,
            ),
            np.asarray(
                [
                    [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]],
                    [[2.0, 0.0], [0.0, 4.0], [1.0, 1.0]],
                ],
                dtype=np.float32,
            ),
            np.asarray(
                [[1 + 2j, 3 - 1j], [2 - 4j, -1 + 0.5j], [4 + 0j, 2 + 3j]],
                dtype=np.complex128,
            ),
        )

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            inverse = self._array_function(
                runtime, "cnp_linalg_inv", [ctypes.c_void_p]
            )
            for value in inverse_cases:
                with self.subTest(operation="inv", dtype=str(value.dtype)):
                    with runtime.from_numpy(value) as source:
                        actual = self._array_result(
                            runtime,
                            inverse,
                            "cnp_linalg_inv",
                            source.pointer,
                        )
                    try:
                        assert_array_equivalent(
                            self,
                            actual,
                            np.linalg.inv(value),
                            rtol=3e-5,
                            atol=3e-5,
                        )
                    finally:
                        actual.close()
                self.assertEqual(baseline, runtime.retained_bytes)

            for name in ("cnp_linalg_pinv", "cnp_pinv"):
                pinv = self._array_function(
                    runtime,
                    name,
                    [ctypes.c_void_p, ctypes.c_double],
                )
                for value in pinv_cases:
                    with self.subTest(
                        operation=name, dtype=str(value.dtype)
                    ):
                        with runtime.from_numpy(value) as source:
                            actual = self._array_result(
                                runtime,
                                pinv,
                                name,
                                source.pointer,
                                1e-12,
                            )
                        try:
                            assert_array_equivalent(
                                self,
                                actual,
                                np.linalg.pinv(value, rcond=1e-12),
                                rtol=8e-5,
                                atol=8e-5,
                            )
                        finally:
                            actual.close()
                    self.assertEqual(baseline, runtime.retained_bytes)

    def test_reduced_qr_matches_numpy_shapes_dtypes_reconstruction_and_lifetime(
        self,
    ) -> None:
        cases = (
            np.asarray(
                [[1, 2], [3, 5], [7, 11]], dtype=np.float32
            ),
            np.asarray(
                [[1 + 2j, 3 - 1j, 2 + 0j], [2 - 4j, -1 + 0.5j, 5 + 2j]],
                dtype=np.complex128,
            ),
            np.asarray(
                [
                    [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]],
                    [[2.0, 1.0], [0.0, 4.0], [1.0, 3.0]],
                ],
                dtype=np.float64,
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            qr = getattr(runtime.dll, "cnp_linalg_qr")
            qr.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
            ]
            qr.restype = ctypes.c_int
            for value in cases:
                with self.subTest(shape=value.shape, dtype=str(value.dtype)):
                    with runtime.from_numpy(value) as source:
                        q_pointer = ctypes.c_void_p()
                        r_pointer = ctypes.c_void_p()
                        runtime.dll.cnp_clear_error()
                        status = qr(
                            source.pointer,
                            ctypes.byref(q_pointer),
                            ctypes.byref(r_pointer),
                        )
                        if status != 0:
                            if q_pointer.value:
                                runtime.dll.cnp_array_decref(q_pointer)
                            if r_pointer.value:
                                runtime.dll.cnp_array_decref(r_pointer)
                        self.assertEqual(0, status, runtime.error_state())
                        q = runtime._owned_result(
                            q_pointer.value, "cnp_linalg_qr.q"
                        )
                        r = runtime._owned_result(
                            r_pointer.value, "cnp_linalg_qr.r"
                        )
                    try:
                        q_value = q.to_numpy()
                        r_value = r.to_numpy()
                        expected_q, expected_r = np.linalg.qr(value)
                        self.assertEqual(expected_q.shape, q_value.shape)
                        self.assertEqual(expected_r.shape, r_value.shape)
                        self.assertEqual(expected_q.dtype, q_value.dtype)
                        self.assertEqual(expected_r.dtype, r_value.dtype)
                        np.testing.assert_allclose(
                            q_value @ r_value,
                            value,
                            rtol=4e-5,
                            atol=4e-5,
                        )
                        identity = np.broadcast_to(
                            np.eye(q_value.shape[-1], dtype=q_value.dtype),
                            q_value.shape[:-2]
                            + (q_value.shape[-1], q_value.shape[-1]),
                        )
                        np.testing.assert_allclose(
                            np.swapaxes(q_value.conj(), -1, -2) @ q_value,
                            identity,
                            rtol=4e-5,
                            atol=4e-5,
                        )
                    finally:
                        r.close()
                        q.close()
                self.assertEqual(baseline, runtime.retained_bytes)

    def test_norm_array_and_legacy_scalar_projection_match_numpy(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            norm = self._array_function(
                runtime,
                "cnp_linalg_norm",
                [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int],
            )
            cases = (
                (
                    np.asarray(
                        [[1 + 2j, 3 - 4j], [2 - 1j, -5 + 2j]],
                        dtype=np.complex128,
                    ),
                    None,
                    0,
                    np.linalg.norm,
                ),
                (
                    np.arange(24, dtype=np.float32).reshape(2, 3, 4),
                    b"1",
                    1,
                    lambda value: np.linalg.norm(value, ord=1, axis=1),
                ),
                (
                    np.asarray([[0, -2, 3], [4, 0, -1]], dtype=np.int16),
                    b"-inf",
                    -1,
                    lambda value: np.linalg.norm(
                        value, ord=-np.inf, axis=-1
                    ),
                ),
            )
            for value, order, axis, oracle in cases:
                with self.subTest(order=order, axis=axis):
                    with ExitStack() as stack:
                        source = stack.enter_context(runtime.from_numpy(value))
                        actual = stack.enter_context(
                            self._array_result(
                                runtime,
                                norm,
                                "cnp_linalg_norm",
                                source.pointer,
                                order,
                                axis,
                            )
                        )
                        assert_array_equivalent(
                            self,
                            actual,
                            oracle(value),
                            rtol=3e-6,
                            atol=3e-6,
                        )
                self.assertEqual(baseline, runtime.retained_bytes)

            norm_ext = getattr(runtime.dll, "cnp_linalg_norm_ext")
            norm_ext.argtypes = [
                ctypes.c_void_p,
                ctypes.c_double,
                ctypes.c_int,
            ]
            norm_ext.restype = ctypes.c_double
            vector_value = np.asarray([1.0, -2.0, 4.0], dtype=np.float64)
            matrix_value = np.asarray(
                [[1.0, -2.0], [3.0, 4.0]], dtype=np.float64
            )
            with ExitStack() as stack:
                vector = stack.enter_context(runtime.from_numpy(vector_value))
                matrix = stack.enter_context(runtime.from_numpy(matrix_value))
                self.assertAlmostEqual(
                    np.linalg.norm(vector_value, ord=3),
                    norm_ext(vector.pointer, 3.0, 0),
                    places=12,
                )
                self.assertAlmostEqual(
                    np.linalg.norm(matrix_value, ord=1),
                    norm_ext(matrix.pointer, 1.0, -1),
                    places=12,
                )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_tensorinv_and_default_tensorsolve_match_numpy_shape_and_values(
        self,
    ) -> None:
        tensor_value = (
            np.eye(24, dtype=np.float64).reshape(4, 6, 8, 3)
            * 2.0
        )
        solve_a = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)
        solve_a = solve_a * (2 + 1j)
        solve_b = np.asarray(
            [[1 + 2j, 3 - 1j], [2 - 4j, -1 + 0.5j]],
            dtype=np.complex128,
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            tensorinv = self._array_function(
                runtime,
                "cnp_linalg_tensorinv",
                [ctypes.c_void_p, ctypes.c_int],
            )
            source = runtime.from_numpy(tensor_value)
            inverse = self._array_result(
                runtime,
                tensorinv,
                "cnp_linalg_tensorinv",
                source.pointer,
                2,
            )
            source.close()
            try:
                assert_array_equivalent(
                    self,
                    inverse,
                    np.linalg.tensorinv(tensor_value, ind=2),
                    rtol=2e-12,
                    atol=2e-12,
                )
            finally:
                inverse.close()
            self.assertEqual(baseline, runtime.retained_bytes)

            tensorsolve = self._array_function(
                runtime,
                "cnp_linalg_tensorsolve",
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)],
            )
            left = runtime.from_numpy(solve_a)
            right = runtime.from_numpy(solve_b)
            solved = self._array_result(
                runtime,
                tensorsolve,
                "cnp_linalg_tensorsolve",
                left.pointer,
                right.pointer,
                None,
            )
            right.close()
            left.close()
            try:
                assert_array_equivalent(
                    self,
                    solved,
                    np.linalg.tensorsolve(solve_a, solve_b),
                    rtol=2e-12,
                    atol=2e-12,
                )
            finally:
                solved.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_tensorsolve_v2_axes_permutations_errors_and_lifetime(
        self,
    ) -> None:
        def moved_axes_permutation(
            ndim: int, axes: tuple[int, ...]
        ) -> tuple[int, ...]:
            permutation = list(range(ndim))
            for axis in axes:
                permutation.remove(axis)
                permutation.insert(ndim, axis)
            return tuple(permutation)

        matrix = np.diag(np.linspace(2.0, 7.0, 6))
        matrix += np.triu(np.ones((6, 6), dtype=np.float64), 1) * 0.05
        moved = matrix.reshape(2, 3, 3, 2)
        rhs_value = np.asarray(
            [[1.0, -2.0, 3.0], [4.0, 0.5, -1.0]],
            dtype=np.float64,
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            tensorsolve_v2 = self._array_function(
                runtime,
                "cnp_linalg_tensorsolve_v2",
                [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_int),
                ],
            )
            for axes in ((), (0,), (0, 2), (0, 0)):
                permutation = moved_axes_permutation(4, axes)
                inverse = tuple(np.argsort(permutation))
                coefficient = np.ascontiguousarray(
                    moved.transpose(inverse)
                )
                expected = np.linalg.tensorsolve(
                    coefficient, rhs_value, axes=axes
                )
                axes_pointer = (
                    (ctypes.c_int * len(axes))(*axes) if axes else None
                )
                with self.subTest(axes=axes):
                    left = runtime.from_numpy(coefficient)
                    right = runtime.from_numpy(rhs_value)
                    actual = self._array_result(
                        runtime,
                        tensorsolve_v2,
                        "cnp_linalg_tensorsolve_v2",
                        left.pointer,
                        right.pointer,
                        len(axes),
                        axes_pointer,
                    )
                    right.close()
                    left.close()
                    try:
                        assert_array_equivalent(
                            self,
                            actual,
                            expected,
                            rtol=2e-12,
                            atol=2e-12,
                        )
                    finally:
                        actual.close()
                    self.assertEqual(baseline, runtime.retained_bytes)

            with (
                runtime.from_numpy(moved) as left,
                runtime.from_numpy(rhs_value) as right,
            ):
                invalid_cases = (
                    ("negative_length", -1, None, "axes length"),
                    (
                        "missing_axes",
                        1,
                        None,
                        "axes must not be null",
                    ),
                    (
                        "negative_axis",
                        1,
                        (ctypes.c_int * 1)(-1),
                        "axis -1 is out of bounds",
                    ),
                    (
                        "large_axis",
                        1,
                        (ctypes.c_int * 1)(4),
                        "axis 4 is out of bounds",
                    ),
                )
                for label, naxes, axes_pointer, message in invalid_cases:
                    with self.subTest(error=label):
                        before_error = runtime.retained_bytes
                        runtime.dll.cnp_clear_error()
                        pointer = tensorsolve_v2(
                            left.pointer,
                            right.pointer,
                            naxes,
                            axes_pointer,
                        )
                        self.assertFalse(pointer)
                        error = runtime.error_state()
                        self.assertNotEqual(0, error.status)
                        self.assertEqual(
                            "cnp_linalg_tensorsolve_v2", error.function
                        )
                        self.assertIn(message, error.message)
                        self.assertEqual(
                            before_error, runtime.retained_bytes
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_corrcoef_and_cov_match_numpy_rowvar_y_complex_and_strides(
        self,
    ) -> None:
        x_value = np.asarray(
            [[1.0, 2.0, 4.0, 8.0], [2.0, -1.0, 3.0, 5.0]],
            dtype=np.float64,
        )
        y_value = np.asarray([3.0, 1.0, 0.0, 7.0], dtype=np.float64)
        complex_value = np.asarray(
            [[1 + 2j, 3 - 1j, 2 + 0j], [2 - 4j, -1 + 0.5j, 5 + 2j]],
            dtype=np.complex128,
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            corrcoef = self._array_function(
                runtime,
                "cnp_corrcoef",
                [ctypes.c_void_p, ctypes.c_void_p],
            )
            cov = self._array_function(
                runtime,
                "cnp_cov",
                [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                ],
            )
            cases = (
                (
                    corrcoef,
                    "cnp_corrcoef",
                    x_value,
                    y_value,
                    (1, 1),
                    lambda x, y: np.corrcoef(x, y),
                ),
                (
                    corrcoef,
                    "cnp_corrcoef",
                    complex_value,
                    None,
                    (1, 1),
                    lambda x, y: np.corrcoef(x),
                ),
                (
                    cov,
                    "cnp_cov",
                    x_value.T,
                    None,
                    (0, 1),
                    lambda x, y: np.cov(x, rowvar=False, ddof=1),
                ),
                (
                    cov,
                    "cnp_cov",
                    complex_value,
                    None,
                    (1, 1),
                    lambda x, y: np.cov(x, rowvar=True, ddof=1),
                ),
            )
            for function, name, left_value, right_value, options, oracle in cases:
                with self.subTest(operation=name, options=options):
                    with ExitStack() as stack:
                        left_owner = stack.enter_context(
                            runtime.from_numpy(left_value.T.copy())
                        )
                        left = stack.enter_context(runtime.transpose(left_owner))
                        right = (
                            stack.enter_context(runtime.from_numpy(right_value))
                            if right_value is not None
                            else None
                        )
                        arguments = (
                            (left.pointer, right.pointer if right else None)
                            if name == "cnp_corrcoef"
                            else (
                                left.pointer,
                                right.pointer if right else None,
                                options[0],
                                options[1],
                            )
                        )
                        actual = stack.enter_context(
                            self._array_result(
                                runtime, function, name, *arguments
                            )
                        )
                        assert_array_equivalent(
                            self,
                            actual,
                            oracle(left_value, right_value),
                            rtol=3e-12,
                            atol=3e-12,
                        )
                self.assertEqual(baseline, runtime.retained_bytes)

    def test_linalg_statistics_failures_are_labeled_atomic_and_nonretaining(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            vector = runtime.from_numpy(np.ones((3,), dtype=np.float64))
            nonsquare = runtime.from_numpy(np.ones((2, 3), dtype=np.float64))
            matrix = runtime.from_numpy(np.eye(2, dtype=np.float64))
            tensor = runtime.from_numpy(
                np.ones((2, 2, 3, 2), dtype=np.float64)
            )
            observations = runtime.from_numpy(
                np.ones((2, 4), dtype=np.float64)
            )
            short = runtime.from_numpy(np.ones((3,), dtype=np.float64))
            try:
                for name, argtypes, arguments in (
                    (
                        "cnp_linalg_inv",
                        [ctypes.c_void_p],
                        (nonsquare.pointer,),
                    ),
                    (
                        "cnp_linalg_pinv",
                        [ctypes.c_void_p, ctypes.c_double],
                        (vector.pointer, 1e-12),
                    ),
                    (
                        "cnp_pinv",
                        [ctypes.c_void_p, ctypes.c_double],
                        (vector.pointer, 1e-12),
                    ),
                    (
                        "cnp_linalg_norm",
                        [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int],
                        (matrix.pointer, b"invalid", 0),
                    ),
                    (
                        "cnp_linalg_tensorinv",
                        [ctypes.c_void_p, ctypes.c_int],
                        (tensor.pointer, 2),
                    ),
                    (
                        "cnp_corrcoef",
                        [ctypes.c_void_p, ctypes.c_void_p],
                        (observations.pointer, short.pointer),
                    ),
                    (
                        "cnp_cov",
                        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
                        (observations.pointer, None, 2, 1),
                    ),
                ):
                    function = self._array_function(runtime, name, argtypes)
                    runtime.dll.cnp_clear_error()
                    pointer = function(*arguments)
                    if pointer:
                        runtime.dll.cnp_array_decref(pointer)
                    self.assertFalse(pointer, name)
                    error = runtime.error_state()
                    self.assertNotEqual(0, error.status, name)
                    self.assertEqual(name, error.function)

                qr = getattr(runtime.dll, "cnp_linalg_qr")
                qr.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.POINTER(ctypes.c_void_p),
                ]
                qr.restype = ctypes.c_int
                q_pointer = ctypes.c_void_p(1)
                r_pointer = ctypes.c_void_p(1)
                runtime.dll.cnp_clear_error()
                status = qr(
                    vector.pointer,
                    ctypes.byref(q_pointer),
                    ctypes.byref(r_pointer),
                )
                self.assertNotEqual(0, status)
                self.assertFalse(q_pointer.value)
                self.assertFalse(r_pointer.value)
                self.assertEqual("cnp_linalg_qr", runtime.error_state().function)

                tensorsolve = self._array_function(
                    runtime,
                    "cnp_linalg_tensorsolve",
                    [
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                        ctypes.POINTER(ctypes.c_int),
                    ],
                )
                axes = (ctypes.c_int * 1)(0)
                runtime.dll.cnp_clear_error()
                self.assertFalse(
                    tensorsolve(matrix.pointer, vector.pointer, axes)
                )
                error = runtime.error_state()
                self.assertNotEqual(0, error.status)
                self.assertEqual("cnp_linalg_tensorsolve", error.function)

                norm_ext = getattr(runtime.dll, "cnp_linalg_norm_ext")
                norm_ext.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_double,
                    ctypes.c_int,
                ]
                norm_ext.restype = ctypes.c_double
                runtime.dll.cnp_clear_error()
                self.assertTrue(np.isnan(norm_ext(tensor.pointer, 2.0, 0)))
                error = runtime.error_state()
                self.assertNotEqual(0, error.status)
                self.assertEqual("cnp_linalg_norm_ext", error.function)
            finally:
                short.close()
                observations.close()
                tensor.close()
                matrix.close()
                nonsquare.close()
                vector.close()
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
