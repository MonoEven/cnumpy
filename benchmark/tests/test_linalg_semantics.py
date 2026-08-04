from __future__ import annotations

import ctypes
import unittest
from contextlib import ExitStack
from itertools import permutations
from pathlib import Path

import numpy as np

from compat.cnumpy_ctypes import CnumpyError, CnumpyRuntime
from compat.oracle import assert_array_equivalent


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"


class _CnpSlice(ctypes.Structure):
    _fields_ = [
        ("start", ctypes.c_int64),
        ("stop", ctypes.c_int64),
        ("step", ctypes.c_int64),
        ("has_start", ctypes.c_bool),
        ("has_stop", ctypes.c_bool),
        ("has_step", ctypes.c_bool),
    ]


class LinalgSemanticsTests(unittest.TestCase):
    def required_export(
        self, runtime: CnumpyRuntime, name: str
    ) -> ctypes._CFuncPtr:
        try:
            return getattr(runtime.dll, name)
        except AttributeError as error:
            self.fail(f"required export {name} is missing: {error}")

    def assert_eigenvalue_multiset(
        self,
        actual: np.ndarray,
        expected: np.ndarray,
        *,
        rtol: float,
        atol: float,
    ) -> None:
        actual_flat = np.asarray(actual).reshape(-1)
        expected_flat = np.asarray(expected).reshape(-1)
        self.assertEqual(expected_flat.shape, actual_flat.shape)
        best_order = min(
            permutations(range(expected_flat.size)),
            key=lambda order: float(
                np.max(np.abs(actual_flat - expected_flat[list(order)]))
            ),
        )
        np.testing.assert_allclose(
            actual_flat,
            expected_flat[list(best_order)],
            rtol=rtol,
            atol=atol,
        )

    def einsum_result(
        self,
        runtime: CnumpyRuntime,
        subscripts: str,
        operands: tuple,
    ):
        function = self.required_export(runtime, "cnp_einsum")
        function.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        function.restype = ctypes.c_void_p
        pointers = (ctypes.c_void_p * len(operands))(
            *(operand.pointer.value for operand in operands)
        )
        runtime.dll.cnp_clear_error()
        pointer = function(
            subscripts.encode("ascii"), len(operands), pointers
        )
        return runtime._owned_result(pointer, "cnp_einsum")

    def eig_result(
        self,
        runtime: CnumpyRuntime,
        operand,
    ):
        function = self.required_export(runtime, "cnp_linalg_eig")
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        function.restype = ctypes.c_int
        eigenvalues_pointer = ctypes.c_void_p()
        eigenvectors_pointer = ctypes.c_void_p()
        runtime.dll.cnp_clear_error()
        status = int(
            function(
                operand.pointer,
                ctypes.byref(eigenvalues_pointer),
                ctypes.byref(eigenvectors_pointer),
            )
        )
        if status != 0:
            partial = (
                eigenvalues_pointer.value,
                eigenvectors_pointer.value,
            )
            for pointer in partial:
                if pointer:
                    runtime.dll.cnp_array_decref(pointer)
            if any(partial):
                self.fail("cnp_linalg_eig failed with live partial results")
            raise runtime.native_error("cnp_linalg_eig", status)
        if not eigenvalues_pointer.value or not eigenvectors_pointer.value:
            for pointer in (
                eigenvalues_pointer.value,
                eigenvectors_pointer.value,
            ):
                if pointer:
                    runtime.dll.cnp_array_decref(pointer)
            self.fail("cnp_linalg_eig succeeded without both results")
        return (
            runtime._owned_result(
                eigenvalues_pointer.value, "cnp_linalg_eig:eigenvalues"
            ),
            runtime._owned_result(
                eigenvectors_pointer.value, "cnp_linalg_eig:eigenvectors"
            ),
        )

    def det_result(self, runtime: CnumpyRuntime, operand):
        function = self.required_export(runtime, "cnp_linalg_det")
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        pointer = function(operand.pointer)
        return runtime._owned_result(pointer, "cnp_linalg_det")

    def cholesky_result(self, runtime: CnumpyRuntime, operand):
        function = self.required_export(runtime, "cnp_linalg_cholesky")
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        function.restype = ctypes.c_int
        result_pointer = ctypes.c_void_p()
        runtime.dll.cnp_clear_error()
        status = int(function(operand.pointer, ctypes.byref(result_pointer)))
        if status != 0:
            if result_pointer.value:
                runtime.dll.cnp_array_decref(result_pointer.value)
                self.fail("cnp_linalg_cholesky failed with a partial result")
            raise runtime.native_error("cnp_linalg_cholesky", status)
        if not result_pointer.value:
            self.fail("cnp_linalg_cholesky succeeded without a result")
        return runtime._owned_result(
            result_pointer.value, "cnp_linalg_cholesky"
        )

    def slogdet_result(self, runtime: CnumpyRuntime, operand):
        function = self.required_export(runtime, "cnp_linalg_slogdet_v2")
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        function.restype = ctypes.c_int
        sign_pointer = ctypes.c_void_p()
        logabsdet_pointer = ctypes.c_void_p()
        runtime.dll.cnp_clear_error()
        status = int(
            function(
                operand.pointer,
                ctypes.byref(sign_pointer),
                ctypes.byref(logabsdet_pointer),
            )
        )
        if status != 0:
            partial = (sign_pointer.value, logabsdet_pointer.value)
            for pointer in partial:
                if pointer:
                    runtime.dll.cnp_array_decref(pointer)
            if any(partial):
                self.fail("cnp_linalg_slogdet_v2 failed with live results")
            raise runtime.native_error("cnp_linalg_slogdet_v2", status)
        if not sign_pointer.value or not logabsdet_pointer.value:
            for pointer in (sign_pointer.value, logabsdet_pointer.value):
                if pointer:
                    runtime.dll.cnp_array_decref(pointer)
            self.fail("cnp_linalg_slogdet_v2 omitted a result")
        return (
            runtime._owned_result(
                sign_pointer.value, "cnp_linalg_slogdet_v2:sign"
            ),
            runtime._owned_result(
                logabsdet_pointer.value,
                "cnp_linalg_slogdet_v2:logabsdet",
            ),
        )

    def _ieee_mismatch_mask(
        self, actual: np.ndarray, expected: np.ndarray
    ) -> np.ndarray:
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected.dtype, actual.dtype)
        if np.issubdtype(expected.dtype, np.complexfloating):
            return self._ieee_mismatch_mask(
                actual.real, expected.real
            ) | self._ieee_mismatch_mask(actual.imag, expected.imag)

        actual_nan = np.isnan(actual)
        expected_nan = np.isnan(expected)
        comparable = ~actual_nan & ~expected_nan
        both_zero = comparable & (actual == 0) & (expected == 0)
        return (
            (actual_nan != expected_nan)
            | (comparable & (actual != expected))
            | (
                both_zero
                & (np.signbit(actual) != np.signbit(expected))
            )
        )

    def assert_special_value_pattern(
        self,
        actual: np.ndarray,
        expected: np.ndarray,
        *,
        rtol: float,
    ) -> None:
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected.dtype, actual.dtype)
        if np.issubdtype(expected.dtype, np.complexfloating):
            self.assert_special_value_pattern(
                actual.real, expected.real, rtol=rtol
            )
            self.assert_special_value_pattern(
                actual.imag, expected.imag, rtol=rtol
            )
            return

        actual_nan = np.isnan(actual)
        expected_nan = np.isnan(expected)
        actual_inf = np.isinf(actual)
        expected_inf = np.isinf(expected)
        actual_zero = actual == 0
        expected_zero = expected == 0
        np.testing.assert_array_equal(actual_nan, expected_nan)
        np.testing.assert_array_equal(actual_inf, expected_inf)
        np.testing.assert_array_equal(actual_zero, expected_zero)
        np.testing.assert_array_equal(
            np.signbit(actual)[actual_inf | actual_zero],
            np.signbit(expected)[expected_inf | expected_zero],
        )
        finite_nonzero = (
            ~expected_nan & ~expected_inf & ~expected_zero
        )
        np.testing.assert_allclose(
            actual[finite_nonzero],
            expected[finite_nonzero],
            rtol=rtol,
            atol=0.0,
        )

    def svd_result(
        self,
        runtime: CnumpyRuntime,
        operand,
    ):
        function = self.required_export(runtime, "cnp_linalg_svd")
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        function.restype = ctypes.c_int
        pointers = [ctypes.c_void_p() for _ in range(3)]
        runtime.dll.cnp_clear_error()
        status = int(
            function(
                operand.pointer,
                *(ctypes.byref(pointer) for pointer in pointers),
            )
        )
        if status != 0:
            partial = tuple(pointer.value for pointer in pointers)
            for pointer in partial:
                if pointer:
                    runtime.dll.cnp_array_decref(pointer)
            if any(partial):
                self.fail("cnp_linalg_svd failed with live partial results")
            raise runtime.native_error("cnp_linalg_svd", status)
        if any(not pointer.value for pointer in pointers):
            for pointer in pointers:
                if pointer.value:
                    runtime.dll.cnp_array_decref(pointer.value)
            self.fail("cnp_linalg_svd succeeded without all three results")
        return tuple(
            runtime._owned_result(pointer.value, f"cnp_linalg_svd:{name}")
            for pointer, name in zip(pointers, ("u", "s", "vh"))
        )

    def svd_v2_result(
        self,
        runtime: CnumpyRuntime,
        operand,
        *,
        full_matrices: bool,
        compute_uv: bool = True,
        hermitian: bool = False,
    ):
        function = self.required_export(runtime, "cnp_linalg_svd_v2")
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_bool,
            ctypes.c_bool,
            ctypes.c_bool,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        function.restype = ctypes.c_int
        pointers = [ctypes.c_void_p() for _ in range(3)]
        runtime.dll.cnp_clear_error()
        status = int(
            function(
                operand.pointer,
                full_matrices,
                compute_uv,
                hermitian,
                *(ctypes.byref(pointer) for pointer in pointers),
            )
        )
        if status != 0:
            partial = tuple(pointer.value for pointer in pointers)
            for pointer in partial:
                if pointer:
                    runtime.dll.cnp_array_decref(pointer)
            if any(partial):
                self.fail("cnp_linalg_svd_v2 failed with live partial results")
            raise runtime.native_error("cnp_linalg_svd_v2", status)
        if compute_uv:
            if any(not pointer.value for pointer in pointers):
                for pointer in pointers:
                    if pointer.value:
                        runtime.dll.cnp_array_decref(pointer.value)
                self.fail("cnp_linalg_svd_v2 omitted a requested result")
            return tuple(
                runtime._owned_result(
                    pointer.value, f"cnp_linalg_svd_v2:{name}"
                )
                for pointer, name in zip(pointers, ("u", "s", "vh"))
            )
        if pointers[0].value or not pointers[1].value or pointers[2].value:
            for pointer in pointers:
                if pointer.value:
                    runtime.dll.cnp_array_decref(pointer.value)
            self.fail("cnp_linalg_svd_v2 compute_uv=False returned wrong slots")
        return runtime._owned_result(
            pointers[1].value, "cnp_linalg_svd_v2:s"
        )

    def solve_result(self, runtime: CnumpyRuntime, a, b):
        function = self.required_export(runtime, "cnp_linalg_solve")
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        function.restype = ctypes.c_int
        pointer = ctypes.c_void_p()
        runtime.dll.cnp_clear_error()
        status = int(function(a.pointer, b.pointer, ctypes.byref(pointer)))
        if status != 0:
            if pointer.value:
                runtime.dll.cnp_array_decref(pointer.value)
                self.fail("cnp_linalg_solve failed with a partial result")
            raise runtime.native_error("cnp_linalg_solve", status)
        if not pointer.value:
            self.fail("cnp_linalg_solve succeeded without a result")
        return runtime._owned_result(pointer.value, "cnp_linalg_solve")

    def lstsq_v2_result(
        self,
        runtime: CnumpyRuntime,
        a,
        b,
        *,
        rcond: float = 0.0,
        rcond_none: bool = True,
    ):
        function = self.required_export(runtime, "cnp_linalg_lstsq_v2")
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_bool,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        function.restype = ctypes.c_int
        pointers = [ctypes.c_void_p() for _ in range(4)]
        runtime.dll.cnp_clear_error()
        status = int(
            function(
                a.pointer,
                b.pointer,
                rcond,
                rcond_none,
                *(ctypes.byref(pointer) for pointer in pointers),
            )
        )
        if status != 0:
            partial = tuple(pointer.value for pointer in pointers)
            for pointer in partial:
                if pointer:
                    runtime.dll.cnp_array_decref(pointer)
            if any(partial):
                self.fail("cnp_linalg_lstsq_v2 failed with partial results")
            raise runtime.native_error("cnp_linalg_lstsq_v2", status)
        if any(not pointer.value for pointer in pointers):
            for pointer in pointers:
                if pointer.value:
                    runtime.dll.cnp_array_decref(pointer.value)
            self.fail("cnp_linalg_lstsq_v2 omitted a result")
        return tuple(
            runtime._owned_result(
                pointer.value, f"cnp_linalg_lstsq_v2:{name}"
            )
            for pointer, name in zip(
                pointers, ("x", "residuals", "rank", "singular_values")
            )
        )

    def cond_v2_result(self, runtime: CnumpyRuntime, operand):
        function = self.required_export(runtime, "cnp_linalg_cond_v2")
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_void_p
        runtime.dll.cnp_clear_error()
        pointer = function(operand.pointer)
        return runtime._owned_result(pointer, "cnp_linalg_cond_v2")

    def assert_einsum_matches(
        self,
        runtime: CnumpyRuntime,
        stack: ExitStack,
        subscripts: str,
        values: tuple[np.ndarray, ...],
        operands: tuple | None = None,
    ) -> None:
        if operands is None:
            operands = tuple(
                stack.enter_context(runtime.from_numpy(value))
                for value in values
            )
        result = stack.enter_context(
            self.einsum_result(runtime, subscripts, operands)
        )
        assert_array_equivalent(
            self,
            result,
            np.einsum(subscripts, *values),
            compare_contiguity=False,
            rtol=3e-6 if result.numpy_dtype == np.dtype(np.float32)
            else 3e-13,
            atol=3e-6 if result.numpy_dtype == np.dtype(np.float32)
            else 3e-13,
        )

    def test_einsum_explicit_and_implicit_outputs_match_numpy(self) -> None:
        left_value = np.arange(6, dtype=np.float64).reshape(2, 3) - 2.0
        right_value = (
            np.arange(12, dtype=np.float64).reshape(3, 4) * 0.25 + 1.0
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            left = stack.enter_context(runtime.from_numpy(left_value))
            right = stack.enter_context(runtime.from_numpy(right_value))
            operands = (left, right)
            for subscripts in ("ij,jk->ik", "ij,jk"):
                with self.subTest(subscripts=subscripts):
                    self.assert_einsum_matches(
                        runtime,
                        stack,
                        subscripts,
                        (left_value, right_value),
                        operands,
                    )

            self.assert_einsum_matches(
                runtime, stack, "ji", (left_value,), (left,)
            )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_einsum_repeated_labels_diagonals_and_reductions(self) -> None:
        matrix_value = (
            np.arange(16, dtype=np.float64).reshape(4, 4) * 0.5 - 3.0
        )
        tensor_value = np.arange(18, dtype=np.float64).reshape(3, 2, 3)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            matrix = stack.enter_context(runtime.from_numpy(matrix_value))
            tensor = stack.enter_context(runtime.from_numpy(tensor_value))
            for subscripts, values, operands in (
                ("ii->i", (matrix_value,), (matrix,)),
                ("ii", (matrix_value,), (matrix,)),
                ("iji->j", (tensor_value,), (tensor,)),
                ("ij->", (matrix_value,), (matrix,)),
            ):
                with self.subTest(subscripts=subscripts):
                    self.assert_einsum_matches(
                        runtime, stack, subscripts, values, operands
                    )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_einsum_ellipsis_scalar_and_broadcasting_match_numpy(
        self,
    ) -> None:
        left_value = (
            np.arange(12, dtype=np.float64).reshape(2, 1, 2, 3) + 1.0
        )
        right_value = (
            np.arange(24, dtype=np.float64).reshape(1, 4, 3, 2) * 0.1
        )
        scalar_value = np.array(2.5, dtype=np.float64)
        vector_value = np.arange(6, dtype=np.float64).reshape(2, 3)
        broadcast_value = np.array([[1.0], [2.0]], dtype=np.float64)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for subscripts, values in (
                ("...ij,...jk->...ik", (left_value, right_value)),
                (",...i->...i", (scalar_value, vector_value)),
                ("...i,...i->...", (vector_value, broadcast_value)),
            ):
                with self.subTest(subscripts=subscripts):
                    self.assert_einsum_matches(
                        runtime, stack, subscripts, values
                    )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_einsum_preserves_views_and_numpy_dtype_promotion(self) -> None:
        storage_value = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        view_value = storage_value.transpose(2, 0, 1)
        dtype_cases = (
            (
                np.array([1, 2, 3], dtype=np.int32),
                np.array([0.5, 1.5, 2.5], dtype=np.float32),
            ),
            (
                np.array([1, 2, 3], dtype=np.int64),
                np.array([0.5, 1.5, 2.5], dtype=np.float32),
            ),
            (
                np.array([True, True, False], dtype=np.bool_),
                np.array([True, False, True], dtype=np.bool_),
            ),
            (
                np.array([1 + 2j, 3 - 4j], dtype=np.complex64),
                np.array([2 - 1j, -1 + 0.5j], dtype=np.complex64),
            ),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            storage = stack.enter_context(runtime.from_numpy(storage_value))
            view = stack.enter_context(runtime.transpose(storage, (2, 0, 1)))
            self.assertFalse(view.c_contiguous)
            self.assert_einsum_matches(
                runtime,
                stack,
                "...i,...i->...",
                (view_value, view_value),
                (view, view),
            )

            for left_value, right_value in dtype_cases:
                with self.subTest(
                    left_dtype=str(left_value.dtype),
                    right_dtype=str(right_value.dtype),
                ):
                    self.assert_einsum_matches(
                        runtime,
                        stack,
                        "i,i->i",
                        (left_value, right_value),
                    )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_einsum_float16_forms_promotion_rounding_and_lifetime(self) -> None:
        half_sum = np.asarray(
            [1.0, *([0.0006] * 20)], dtype=np.float16
        )
        cases = (
            ("i->", (half_sum,)),
            (
                "i,i->",
                (
                    np.asarray([-1.091, 4.188, 0.2844], dtype=np.float16),
                    np.asarray([-153.8, -365.0, -576.5], dtype=np.float16),
                ),
            ),
            (
                "i,i",
                (
                    np.asarray([1.5, -2.0, 3.25], dtype=np.float16),
                    np.asarray([4.0, 0.5, -1.0], dtype=np.float16),
                ),
            ),
            (
                "...i,...i->...",
                (
                    np.arange(12, dtype=np.float16).reshape(2, 2, 3),
                    np.linspace(-1, 1, 12, dtype=np.float16).reshape(2, 2, 3),
                ),
            ),
            (
                "i,i->i",
                (
                    np.asarray([1, 2, 3], dtype=np.float16),
                    np.asarray([4, 5, 6], dtype=np.float32),
                ),
            ),
            (
                "i,i->i",
                (
                    np.asarray([1, 2, 3], dtype=np.float16),
                    np.asarray([4, 5, 6], dtype=np.int8),
                ),
            ),
            (
                "i,i->i",
                (
                    np.asarray([1, 2, 3], dtype=np.float16),
                    np.asarray([4, 5, 6], dtype=np.int16),
                ),
            ),
            (
                "i,i->i",
                (
                    np.asarray([1, 2, 3], dtype=np.float16),
                    np.asarray([4, 5, 6], dtype=np.int32),
                ),
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for subscripts, values in cases:
                with self.subTest(
                    subscripts=subscripts,
                    dtypes=tuple(str(value.dtype) for value in values),
                ), ExitStack() as stack:
                    operands = tuple(runtime.from_numpy(value) for value in values)
                    for operand in operands:
                        stack.callback(
                            lambda array=operand: None
                            if array._closed else array.close()
                        )
                    result = self.einsum_result(runtime, subscripts, operands)
                    for operand in operands:
                        operand.close()
                    stack.enter_context(result)
                    expected = np.einsum(subscripts, *values)
                    self.assertEqual(expected.dtype, result.numpy_dtype)
                    self.assertEqual(expected.shape, result.shape)
                    np.testing.assert_array_equal(result.to_numpy(), expected)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_einsum_fast_patterns_respect_named_label_broadcasting(
        self,
    ) -> None:
        cases = (
            (
                "ij,jk->ik",
                (
                    np.array([[2.0], [3.0]], dtype=np.float64),
                    np.arange(12, dtype=np.float64).reshape(3, 4),
                ),
            ),
            (
                "i,i->",
                (
                    np.array([2.0], dtype=np.float64),
                    np.array([1.0, 3.0, 5.0], dtype=np.float64),
                ),
            ),
            (
                "ij,j->i",
                (
                    np.array([[2.0], [4.0]], dtype=np.float64),
                    np.array([1.0, 3.0, 5.0], dtype=np.float64),
                ),
            ),
            (
                "ik,kj->ij",
                (
                    np.arange(6, dtype=np.float64).reshape(2, 3),
                    np.arange(12, dtype=np.float64).reshape(3, 4),
                ),
            ),
            (
                "ki,kj->ij",
                (
                    np.arange(6, dtype=np.float64).reshape(3, 2),
                    np.arange(12, dtype=np.float64).reshape(3, 4),
                ),
            ),
            (
                "ik,jk->ij",
                (
                    np.arange(6, dtype=np.float64).reshape(2, 3),
                    np.arange(12, dtype=np.float64).reshape(4, 3),
                ),
            ),
            (
                "ki,jk->ij",
                (
                    np.arange(6, dtype=np.float64).reshape(3, 2),
                    np.arange(12, dtype=np.float64).reshape(4, 3),
                ),
            ),
            (
                "k,ki->i",
                (
                    np.arange(3, dtype=np.float64),
                    np.arange(12, dtype=np.float64).reshape(3, 4),
                ),
            ),
            (
                "i,j->ji",
                (
                    np.arange(3, dtype=np.float64),
                    np.arange(4, dtype=np.float64),
                ),
            ),
            (
                "aa->",
                (np.arange(9, dtype=np.float64).reshape(3, 3),),
            ),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for subscripts, values in cases:
                with self.subTest(subscripts=subscripts):
                    self.assert_einsum_matches(
                        runtime, stack, subscripts, values
                    )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_einsum_invalid_subscripts_shapes_and_nulls_are_explicit(
        self,
    ) -> None:
        matrix_value = np.arange(6, dtype=np.float64).reshape(2, 3)
        right_value = np.arange(12, dtype=np.float64).reshape(3, 4)
        bad_right_value = np.arange(8, dtype=np.float64).reshape(4, 2)
        rectangular_value = np.arange(6, dtype=np.float64).reshape(2, 3)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with (
                runtime.from_numpy(matrix_value) as matrix,
                runtime.from_numpy(right_value) as right,
                runtime.from_numpy(bad_right_value) as bad_right,
                runtime.from_numpy(rectangular_value) as rectangular,
            ):
                cases = (
                    ("ij,jk->ii", (matrix, right), "output"),
                    ("ij->ik", (matrix,), "output"),
                    ("ij,jk->ik", (matrix,), "operand"),
                    ("ij,jk->ik", (matrix, bad_right), "broadcast"),
                    ("i$->i", (matrix,), "subscript"),
                    ("...i...->i", (matrix,), "ellipsis"),
                    ("...i->i", (matrix,), "ellipsis"),
                    ("ii->i", (rectangular,), "repeated"),
                )
                for subscripts, operands, message in cases:
                    with self.subTest(
                        subscripts=subscripts, message=message
                    ):
                        before_error = runtime.retained_bytes
                        try:
                            unexpected = self.einsum_result(
                                runtime, subscripts, operands
                            )
                        except CnumpyError as captured:
                            captured_error = captured
                        else:
                            unexpected.close()
                            self.fail(
                                f"{subscripts!r} unexpectedly succeeded"
                            )
                        self.assertEqual(
                            "cnp_einsum", captured_error.function
                        )
                        self.assertIn(
                            message, captured_error.message.lower()
                        )
                        self.assertEqual(
                            before_error, runtime.retained_bytes
                        )

                function = self.required_export(runtime, "cnp_einsum")
                function.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_void_p),
                ]
                function.restype = ctypes.c_void_p
                pointers = (ctypes.c_void_p * 1)(matrix.pointer.value)
                runtime.dll.cnp_clear_error()
                pointer = function(None, 1, pointers)
                with self.assertRaises(CnumpyError) as captured:
                    runtime._owned_result(pointer, "cnp_einsum")
                self.assertEqual("cnp_einsum", captured.exception.function)
                self.assertIn("subscripts", captured.exception.message)

            self.assertEqual(baseline, runtime.retained_bytes)

    def test_general_eig_returns_complex_eigenpairs_for_real_matrix(
        self,
    ) -> None:
        value = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
        expected_values, _ = np.linalg.eig(value)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            operand = stack.enter_context(runtime.from_numpy(value))
            eigenvalues, eigenvectors = self.eig_result(runtime, operand)
            stack.enter_context(eigenvalues)
            stack.enter_context(eigenvectors)

            self.assertEqual(np.dtype(np.complex128), eigenvalues.numpy_dtype)
            self.assertEqual(np.dtype(np.complex128), eigenvectors.numpy_dtype)
            self.assertEqual((2,), eigenvalues.shape)
            self.assertEqual((2, 2), eigenvectors.shape)

            actual_values = eigenvalues.to_numpy()
            actual_vectors = eigenvectors.to_numpy()
            self.assert_eigenvalue_multiset(
                actual_values,
                expected_values,
                rtol=2e-13,
                atol=2e-13,
            )
            np.testing.assert_allclose(
                value @ actual_vectors,
                actual_vectors * actual_values[np.newaxis, :],
                rtol=2e-13,
                atol=2e-13,
            )
            np.testing.assert_allclose(
                np.linalg.norm(actual_vectors, axis=0),
                np.ones(2),
                rtol=2e-13,
                atol=2e-13,
            )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_general_eig_preserves_real_dtype_for_real_spectrum(self) -> None:
        value = np.array(
            [[5.0, 4.0, 2.0], [0.0, 1.0, -1.0], [0.0, 0.0, 3.0]],
            dtype=np.float64,
        )
        expected_values, _ = np.linalg.eig(value)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            operand = stack.enter_context(runtime.from_numpy(value))
            eigenvalues, eigenvectors = self.eig_result(runtime, operand)
            stack.enter_context(eigenvalues)
            stack.enter_context(eigenvectors)

            self.assertEqual(expected_values.dtype, eigenvalues.numpy_dtype)
            self.assertEqual(np.dtype(np.float64), eigenvectors.numpy_dtype)
            self.assertEqual((3,), eigenvalues.shape)
            self.assertEqual((3, 3), eigenvectors.shape)

            actual_values = eigenvalues.to_numpy()
            actual_vectors = eigenvectors.to_numpy()
            self.assert_eigenvalue_multiset(
                actual_values,
                expected_values,
                rtol=3e-13,
                atol=3e-13,
            )
            np.testing.assert_allclose(
                value @ actual_vectors,
                actual_vectors * actual_values[np.newaxis, :],
                rtol=3e-13,
                atol=3e-13,
            )
            np.testing.assert_allclose(
                np.linalg.norm(actual_vectors, axis=0),
                np.ones(3),
                rtol=3e-13,
                atol=3e-13,
            )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_general_eig_supports_batched_matrices_and_owned_results(
        self,
    ) -> None:
        value = np.array(
            [
                [[2.0, 1.0], [0.0, 3.0]],
                [[0.0, -1.0], [1.0, 0.0]],
            ],
            dtype=np.float64,
        )
        expected_values, _ = np.linalg.eig(value)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(value) as operand:
                try:
                    eigenvalues, eigenvectors = self.eig_result(
                        runtime, operand
                    )
                except CnumpyError as error:
                    self.fail(f"batched general eig must succeed: {error}")
            stack.enter_context(eigenvalues)
            stack.enter_context(eigenvectors)

            self.assertEqual(expected_values.dtype, eigenvalues.numpy_dtype)
            self.assertEqual(np.dtype(np.complex128), eigenvectors.numpy_dtype)
            self.assertEqual((2, 2), eigenvalues.shape)
            self.assertEqual((2, 2, 2), eigenvectors.shape)

            actual_values = eigenvalues.to_numpy()
            actual_vectors = eigenvectors.to_numpy()
            for batch in range(value.shape[0]):
                with self.subTest(batch=batch):
                    self.assert_eigenvalue_multiset(
                        actual_values[batch],
                        expected_values[batch],
                        rtol=3e-13,
                        atol=3e-13,
                    )
                    np.testing.assert_allclose(
                        value[batch] @ actual_vectors[batch],
                        actual_vectors[batch]
                        * actual_values[batch][np.newaxis, :],
                        rtol=3e-13,
                        atol=3e-13,
                    )
                    np.testing.assert_allclose(
                        np.linalg.norm(actual_vectors[batch], axis=0),
                        np.ones(2),
                        rtol=3e-13,
                        atol=3e-13,
                    )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_general_eig_dtype_promotion_matches_numpy(self) -> None:
        cases = (
            np.array([[2.0, 1.0], [0.0, 3.0]], dtype=np.float32),
            np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32),
            np.array([[0, -1], [1, 0]], dtype=np.int32),
            np.array(
                [[1.0 + 1.0j, 2.0], [0.0, 3.0 - 2.0j]],
                dtype=np.complex64,
            ),
            np.array(
                [[0.0 + 1.0j, 2.0], [3.0, 4.0 - 0.5j]],
                dtype=np.complex128,
            ),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for value in cases:
                with self.subTest(dtype=str(value.dtype), value=value.tolist()):
                    expected_values, expected_vectors = np.linalg.eig(value)
                    operand = stack.enter_context(runtime.from_numpy(value))
                    eigenvalues, eigenvectors = self.eig_result(
                        runtime, operand
                    )
                    stack.enter_context(eigenvalues)
                    stack.enter_context(eigenvectors)
                    self.assertEqual(
                        expected_values.dtype, eigenvalues.numpy_dtype
                    )
                    self.assertEqual(
                        expected_vectors.dtype, eigenvectors.numpy_dtype
                    )
                    actual_values = eigenvalues.to_numpy()
                    actual_vectors = eigenvectors.to_numpy()
                    tolerance = (
                        3e-5 if expected_values.dtype.itemsize <= 8
                        else 3e-12
                    )
                    self.assert_eigenvalue_multiset(
                        actual_values,
                        expected_values,
                        rtol=tolerance,
                        atol=tolerance,
                    )
                    np.testing.assert_allclose(
                        value @ actual_vectors,
                        actual_vectors * actual_values[np.newaxis, :],
                        rtol=tolerance,
                        atol=tolerance,
                    )
                    np.testing.assert_allclose(
                        np.linalg.norm(actual_vectors, axis=0),
                        np.ones(value.shape[-1]),
                        rtol=tolerance,
                        atol=tolerance,
                    )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_general_eig_dense_nonsymmetric_noncontiguous_view(self) -> None:
        canonical = np.array(
            [
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 1.0],
                [0.0, 0.0, 0.0, 3.0],
            ],
            dtype=np.float64,
        )
        transform = np.array(
            [
                [1.0, 2.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, -1.0],
                [1.0, 0.0, 2.0, 1.0],
                [2.0, -1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        dense = transform @ canonical @ np.linalg.inv(transform)
        storage_value = dense.T.copy()
        expected_values, _ = np.linalg.eig(dense)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            storage = stack.enter_context(runtime.from_numpy(storage_value))
            view = stack.enter_context(runtime.transpose(storage, (1, 0)))
            self.assertFalse(view.c_contiguous)
            eigenvalues, eigenvectors = self.eig_result(runtime, view)
            stack.enter_context(eigenvalues)
            stack.enter_context(eigenvectors)

            actual_values = eigenvalues.to_numpy()
            actual_vectors = eigenvectors.to_numpy()
            self.assert_eigenvalue_multiset(
                actual_values,
                expected_values,
                rtol=2e-11,
                atol=2e-11,
            )
            np.testing.assert_allclose(
                dense @ actual_vectors,
                actual_vectors * actual_values[np.newaxis, :],
                rtol=2e-11,
                atol=2e-11,
            )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_general_eig_seeded_dense_differential(self) -> None:
        seed = 20260730
        generator = np.random.default_rng(seed)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for dtype in (np.float64, np.complex128):
                for size in range(1, 6):
                    for sample in range(3):
                        real = generator.standard_normal((size, size))
                        value = real.astype(dtype)
                        if dtype == np.complex128:
                            value += 1j * generator.standard_normal(
                                (size, size)
                            )
                        with self.subTest(
                            seed=seed,
                            dtype=str(np.dtype(dtype)),
                            size=size,
                            sample=sample,
                        ), ExitStack() as stack:
                            before = runtime.retained_bytes
                            expected_values, expected_vectors = np.linalg.eig(
                                value
                            )
                            operand = stack.enter_context(
                                runtime.from_numpy(value)
                            )
                            eigenvalues, eigenvectors = self.eig_result(
                                runtime, operand
                            )
                            stack.enter_context(eigenvalues)
                            stack.enter_context(eigenvectors)
                            self.assertEqual(
                                expected_values.dtype,
                                eigenvalues.numpy_dtype,
                            )
                            self.assertEqual(
                                expected_vectors.dtype,
                                eigenvectors.numpy_dtype,
                            )
                            actual_values = eigenvalues.to_numpy()
                            actual_vectors = eigenvectors.to_numpy()
                            self.assert_eigenvalue_multiset(
                                actual_values,
                                expected_values,
                                rtol=2e-10,
                                atol=2e-10,
                            )
                            np.testing.assert_allclose(
                                value @ actual_vectors,
                                actual_vectors
                                * actual_values[np.newaxis, :],
                                rtol=2e-10,
                                atol=2e-10,
                            )
                            stack.close()
                            self.assertEqual(before, runtime.retained_bytes)

            self.assertEqual(baseline, runtime.retained_bytes)

    def test_general_eig_repeated_and_defective_spectra(self) -> None:
        cases = (
            np.eye(3, dtype=np.float64),
            np.array(
                [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for value in cases:
                with self.subTest(value=value.tolist()):
                    expected_values, _ = np.linalg.eig(value)
                    operand = stack.enter_context(runtime.from_numpy(value))
                    eigenvalues, eigenvectors = self.eig_result(
                        runtime, operand
                    )
                    stack.enter_context(eigenvalues)
                    stack.enter_context(eigenvectors)
                    actual_values = eigenvalues.to_numpy()
                    actual_vectors = eigenvectors.to_numpy()
                    self.assertEqual(
                        expected_values.dtype, eigenvalues.numpy_dtype
                    )
                    self.assert_eigenvalue_multiset(
                        actual_values,
                        expected_values,
                        rtol=3e-12,
                        atol=3e-12,
                    )
                    np.testing.assert_allclose(
                        value @ actual_vectors,
                        actual_vectors * actual_values[np.newaxis, :],
                        rtol=3e-12,
                        atol=3e-12,
                    )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_general_eig_validation_is_explicit_and_clears_outputs(
        self,
    ) -> None:
        square_value = np.eye(2, dtype=np.float64)
        rectangular_value = np.arange(6, dtype=np.float64).reshape(2, 3)
        nonfinite_values = (
            np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=np.float64),
            np.array([[np.inf, 0.0], [0.0, 1.0]], dtype=np.float64),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            square = stack.enter_context(runtime.from_numpy(square_value))
            rectangular = stack.enter_context(
                runtime.from_numpy(rectangular_value)
            )
            nonfinite = tuple(
                stack.enter_context(runtime.from_numpy(value))
                for value in nonfinite_values
            )
            function = self.required_export(runtime, "cnp_linalg_eig")
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
            ]
            function.restype = ctypes.c_int

            cases = (
                (None, -1, "input"),
                (rectangular.pointer, -4, "square"),
                (nonfinite[0].pointer, -10, "nan"),
                (nonfinite[1].pointer, -10, "infinity"),
            )
            for pointer, expected_status, message in cases:
                with self.subTest(message=message):
                    values_slot = ctypes.c_void_p(1)
                    vectors_slot = ctypes.c_void_p(2)
                    before = runtime.retained_bytes
                    runtime.dll.cnp_clear_error()
                    status = int(
                        function(
                            pointer,
                            ctypes.byref(values_slot),
                            ctypes.byref(vectors_slot),
                        )
                    )
                    self.assertEqual(expected_status, status)
                    self.assertFalse(values_slot.value)
                    self.assertFalse(vectors_slot.value)
                    error = runtime.error_state()
                    self.assertEqual("cnp_linalg_eig", error.function)
                    self.assertIn(message, error.message.lower())
                    self.assertEqual(before, runtime.retained_bytes)

            vectors_slot = ctypes.c_void_p(2)
            runtime.dll.cnp_clear_error()
            status = int(
                function(square.pointer, None, ctypes.byref(vectors_slot))
            )
            self.assertEqual(-1, status)
            self.assertFalse(vectors_slot.value)
            self.assertEqual("cnp_linalg_eig", runtime.error_state().function)

            alias_slot = ctypes.c_void_p(3)
            runtime.dll.cnp_clear_error()
            status = int(
                function(
                    square.pointer,
                    ctypes.byref(alias_slot),
                    ctypes.byref(alias_slot),
                )
            )
            self.assertEqual(-1, status)
            self.assertFalse(alias_slot.value)
            self.assertEqual("cnp_linalg_eig", runtime.error_state().function)

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_ahk_eig_bridge_clears_every_provided_result_slot(self) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            square = stack.enter_context(
                runtime.from_numpy(np.asarray([[2.0, 1.0], [0.0, 3.0]]))
            )
            rectangular = stack.enter_context(
                runtime.from_numpy(np.ones((2, 3), dtype=np.float64))
            )
            function = self.required_export(runtime, "cnp_ahk_linalg_eig")
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_int,
            ]
            function.restype = ctypes.c_int

            failed_slots = (ctypes.c_void_p * 4)(1, 2, 3, 4)
            before_failure = runtime.retained_bytes
            runtime.dll.cnp_clear_error()
            status = int(function(rectangular.pointer, failed_slots, 4))
            self.assertEqual(-4, status)
            self.assertTrue(all(not pointer for pointer in failed_slots))
            self.assertEqual(before_failure, runtime.retained_bytes)

            success_slots = (ctypes.c_void_p * 4)(1, 2, 3, 4)
            runtime.dll.cnp_clear_error()
            status = int(function(square.pointer, success_slots, 4))
            self.assertEqual(0, status)
            self.assertTrue(success_slots[0])
            self.assertTrue(success_slots[1])
            self.assertFalse(success_slots[2])
            self.assertFalse(success_slots[3])
            eigenvalues = stack.enter_context(
                runtime._owned_result(
                    success_slots[0], "cnp_ahk_linalg_eig:eigenvalues"
                )
            )
            eigenvectors = stack.enter_context(
                runtime._owned_result(
                    success_slots[1], "cnp_ahk_linalg_eig:eigenvectors"
                )
            )
            np.testing.assert_allclose(
                np.sort(eigenvalues.to_numpy()), np.asarray([2.0, 3.0])
            )
            self.assertEqual((2, 2), eigenvectors.shape)

            one_slot = (ctypes.c_void_p * 1)(9)
            runtime.dll.cnp_clear_error()
            status = int(function(square.pointer, one_slot, 1))
            self.assertEqual(-4, status)
            self.assertFalse(one_slot[0])

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_eigvals_wrapper_inherits_general_semantics_and_owns_errors(
        self,
    ) -> None:
        value = np.array(
            [
                [[2.0, 1.0], [0.0, 3.0]],
                [[0.0, -1.0], [1.0, 0.0]],
            ],
            dtype=np.float32,
        )
        rectangular_value = np.arange(6, dtype=np.float64).reshape(2, 3)
        expected = np.linalg.eigvals(value)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            operand = stack.enter_context(runtime.from_numpy(value))
            rectangular = stack.enter_context(
                runtime.from_numpy(rectangular_value)
            )
            function = self.required_export(runtime, "cnp_eigvals")
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_void_p

            runtime.dll.cnp_clear_error()
            pointer = function(operand.pointer)
            self.assertTrue(
                pointer,
                f"cnp_eigvals batch call failed: {runtime.error_state()}",
            )
            result = stack.enter_context(
                runtime._owned_result(pointer, "cnp_eigvals")
            )
            self.assertEqual(expected.dtype, result.numpy_dtype)
            self.assertEqual((2, 2), result.shape)
            actual = result.to_numpy()
            for batch in range(value.shape[0]):
                self.assert_eigenvalue_multiset(
                    actual[batch],
                    expected[batch],
                    rtol=3e-5,
                    atol=3e-5,
                )

            before_error = runtime.retained_bytes
            runtime.dll.cnp_clear_error()
            pointer = function(rectangular.pointer)
            self.assertFalse(pointer)
            error = runtime.error_state()
            self.assertEqual(-4, error.status)
            self.assertEqual("cnp_eigvals", error.function)
            self.assertIn("square", error.message.lower())
            self.assertEqual(before_error, runtime.retained_bytes)

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_general_eig_zero_sized_matrix_and_batch_shapes(self) -> None:
        cases = (
            np.empty((0, 0), dtype=np.float64),
            np.empty((2, 0, 0), dtype=np.float32),
            np.empty((0, 2, 2), dtype=np.complex64),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for value in cases:
                with self.subTest(shape=value.shape, dtype=str(value.dtype)):
                    expected_values, expected_vectors = np.linalg.eig(value)
                    operand = stack.enter_context(runtime.from_numpy(value))
                    eigenvalues, eigenvectors = self.eig_result(
                        runtime, operand
                    )
                    stack.enter_context(eigenvalues)
                    stack.enter_context(eigenvectors)
                    assert_array_equivalent(
                        self,
                        eigenvalues,
                        expected_values,
                        compare_contiguity=False,
                    )
                    assert_array_equivalent(
                        self,
                        eigenvectors,
                        expected_vectors,
                        compare_contiguity=False,
                    )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_general_eig_tiny_complex_pairs_are_scale_relative(self) -> None:
        for dtype, scales in (
            (np.float32, (1e-14, 1e-30)),
            (np.float64, (1e-14, 1e-30)),
        ):
            values = np.asarray(
                [[[0.0, -scale], [scale, 0.0]] for scale in scales],
                dtype=dtype,
            )
            expected_values, _ = np.linalg.eig(values)
            with self.subTest(dtype=np.dtype(dtype).name), CnumpyRuntime(
                DLL
            ) as runtime, ExitStack() as stack:
                baseline = runtime.retained_bytes
                source = runtime.from_numpy(values)
                stack.callback(
                    lambda: None if source._closed else source.close()
                )
                eigenvalues, eigenvectors = self.eig_result(runtime, source)
                source.close()
                stack.enter_context(eigenvalues)
                stack.enter_context(eigenvectors)
                self.assertEqual(expected_values.dtype, eigenvalues.numpy_dtype)
                actual_values = eigenvalues.to_numpy()
                actual_vectors = eigenvectors.to_numpy()
                for batch, scale in enumerate(scales):
                    self.assert_eigenvalue_multiset(
                        actual_values[batch],
                        expected_values[batch],
                        rtol=4e-5 if dtype == np.float32 else 4e-12,
                        atol=abs(scale)
                        * (4e-5 if dtype == np.float32 else 4e-12),
                    )
                    residual = (
                        values[batch].astype(actual_vectors.dtype)
                        @ actual_vectors[batch]
                        - actual_vectors[batch] * actual_values[batch]
                    )
                    np.testing.assert_allclose(
                        residual,
                        0.0,
                        rtol=0.0,
                        atol=abs(scale)
                        * (8e-5 if dtype == np.float32 else 8e-12),
                    )
                stack.close()
                self.assertEqual(baseline, runtime.retained_bytes)

    def test_general_eig_tiny_nonnormal_eigenvectors_are_scale_relative(
        self,
    ) -> None:
        for dtype, scales in (
            (np.float32, (1e-14, 1e-30)),
            (np.float64, (1e-14, 1e-30, 1e-200)),
        ):
            values = np.asarray(
                [
                    scale * np.asarray(
                        [[1.0, 1.0], [0.0, 2.0]], dtype=dtype
                    )
                    for scale in scales
                ],
                dtype=dtype,
            )
            expected_values, expected_vectors = np.linalg.eig(values)
            with self.subTest(dtype=np.dtype(dtype).name), CnumpyRuntime(
                DLL
            ) as runtime, ExitStack() as stack:
                baseline = runtime.retained_bytes
                source = runtime.from_numpy(values)
                stack.callback(
                    lambda: None if source._closed else source.close()
                )
                eigenvalues, eigenvectors = self.eig_result(runtime, source)
                source.close()
                stack.enter_context(eigenvalues)
                stack.enter_context(eigenvectors)
                actual_values = eigenvalues.to_numpy()
                actual_vectors = eigenvectors.to_numpy()
                self.assertEqual(expected_values.dtype, eigenvalues.numpy_dtype)
                self.assertEqual(expected_vectors.dtype, eigenvectors.numpy_dtype)
                for batch, scale in enumerate(scales):
                    with self.subTest(
                        dtype=np.dtype(dtype).name, scale=scale
                    ):
                        tolerance = 6e-5 if dtype == np.float32 else 6e-12
                        for expected_index in range(2):
                            actual_index = int(
                                np.argmin(
                                    np.abs(
                                        actual_values[batch]
                                        - expected_values[
                                            batch, expected_index
                                        ]
                                    )
                                )
                            )
                            np.testing.assert_allclose(
                                actual_values[batch, actual_index],
                                expected_values[batch, expected_index],
                                rtol=tolerance,
                                atol=abs(scale) * tolerance,
                            )
                            similarity = abs(
                                np.vdot(
                                    expected_vectors[
                                        batch, :, expected_index
                                    ],
                                    actual_vectors[batch, :, actual_index],
                                )
                            )
                            np.testing.assert_allclose(
                                similarity,
                                1.0,
                                rtol=tolerance,
                                atol=tolerance,
                            )
                        residual = (
                            values[batch].astype(actual_vectors.dtype)
                            @ actual_vectors[batch]
                            - actual_vectors[batch] * actual_values[batch]
                        )
                        np.testing.assert_allclose(
                            residual,
                            0.0,
                            rtol=0.0,
                            atol=abs(scale) * tolerance,
                        )
                stack.close()
                self.assertEqual(baseline, runtime.retained_bytes)

    def test_general_eig_exact_zero_degenerate_eigenspaces(self) -> None:
        for dtype in (
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        ):
            values = np.zeros((2, 2, 2), dtype=dtype)
            expected_values, expected_vectors = np.linalg.eig(values)
            with self.subTest(dtype=np.dtype(dtype).name), CnumpyRuntime(
                DLL
            ) as runtime, ExitStack() as stack:
                baseline = runtime.retained_bytes
                source = runtime.from_numpy(values)
                stack.callback(
                    lambda: None if source._closed else source.close()
                )
                eigenvalues, eigenvectors = self.eig_result(runtime, source)
                source.close()
                stack.enter_context(eigenvalues)
                stack.enter_context(eigenvectors)
                actual_values = eigenvalues.to_numpy()
                actual_vectors = eigenvectors.to_numpy()
                self.assertEqual(expected_values.dtype, eigenvalues.numpy_dtype)
                self.assertEqual(expected_vectors.dtype, eigenvectors.numpy_dtype)
                self.assertEqual(expected_values.shape, eigenvalues.shape)
                self.assertEqual(expected_vectors.shape, eigenvectors.shape)
                np.testing.assert_array_equal(actual_values, expected_values)
                np.testing.assert_array_equal(actual_vectors, expected_vectors)
                np.testing.assert_array_equal(
                    values @ actual_vectors,
                    actual_vectors * actual_values[..., np.newaxis, :],
                )
                np.testing.assert_allclose(
                    np.linalg.norm(actual_vectors, axis=-2),
                    np.ones(actual_values.shape),
                    rtol=0.0,
                    atol=0.0,
                )
                stack.close()
                self.assertEqual(baseline, runtime.retained_bytes)

    def test_svd_workspace_products_are_checked_before_allocation(self) -> None:
        implementation = (ROOT / "src" / "svd.c").read_text(encoding="utf-8")

        checked_products = (
            ("rows", "rows", "left_count"),
            ("columns", "columns", "right_count"),
            ("tall_rows", "tall_columns", "matrix_count"),
        )
        for left, right, result in checked_products:
            self.assertRegex(
                implementation,
                rf"svd_checked_product\(\s*{left},\s*{right},\s*"
                rf"&{result}\s*\)",
            )
            self.assertNotRegex(
                implementation,
                rf"svd_allocate\(\s*{left}\s*\*\s*{right}\s*\)",
            )

    def test_svd_legacy_default_returns_complete_rectangular_factors(
        self,
    ) -> None:
        value = np.array(
            [[3.0, 1.0], [0.0, 2.0], [1.0, -1.0]],
            dtype=np.float64,
        )
        expected_u, expected_s, expected_vh = np.linalg.svd(value)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            operand = stack.enter_context(runtime.from_numpy(value))
            actual = self.svd_result(runtime, operand)
            actual_u, actual_s, actual_vh = (
                stack.enter_context(result) for result in actual
            )

            self.assertEqual(expected_u.shape, actual_u.shape)
            self.assertEqual(expected_s.shape, actual_s.shape)
            self.assertEqual(expected_vh.shape, actual_vh.shape)
            self.assertEqual(expected_u.dtype, actual_u.numpy_dtype)
            self.assertEqual(expected_s.dtype, actual_s.numpy_dtype)
            self.assertEqual(expected_vh.dtype, actual_vh.numpy_dtype)

            u = actual_u.to_numpy()
            singular_values = actual_s.to_numpy()
            vh = actual_vh.to_numpy()
            np.testing.assert_allclose(
                singular_values, expected_s, rtol=3e-13, atol=3e-13
            )
            np.testing.assert_allclose(
                u.T @ u, np.eye(3), rtol=3e-13, atol=3e-13
            )
            np.testing.assert_allclose(
                vh @ vh.T, np.eye(2), rtol=3e-13, atol=3e-13
            )
            np.testing.assert_allclose(
                u[:, :2] @ np.diag(singular_values) @ vh,
                value,
                rtol=3e-13,
                atol=3e-13,
            )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_svd_v2_reduced_tall_wide_and_batched_shapes(self) -> None:
        cases = (
            np.array(
                [[3.0, 1.0], [0.0, 2.0], [1.0, -1.0]],
                dtype=np.float64,
            ),
            np.array(
                [[1.0, 2.0, 0.0, -1.0], [0.0, 1.0, 3.0, 2.0]],
                dtype=np.float64,
            ),
            np.arange(24, dtype=np.float64).reshape(2, 4, 3) - 4.0,
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for value in cases:
                with self.subTest(shape=value.shape):
                    expected_u, expected_s, expected_vh = np.linalg.svd(
                        value, full_matrices=False
                    )
                    operand = stack.enter_context(runtime.from_numpy(value))
                    actual = self.svd_v2_result(
                        runtime, operand, full_matrices=False
                    )
                    actual_u, actual_s, actual_vh = (
                        stack.enter_context(result) for result in actual
                    )
                    self.assertEqual(expected_u.shape, actual_u.shape)
                    self.assertEqual(expected_s.shape, actual_s.shape)
                    self.assertEqual(expected_vh.shape, actual_vh.shape)
                    self.assertEqual(expected_u.dtype, actual_u.numpy_dtype)
                    self.assertEqual(expected_s.dtype, actual_s.numpy_dtype)
                    self.assertEqual(expected_vh.dtype, actual_vh.numpy_dtype)
                    u = actual_u.to_numpy()
                    singular_values = actual_s.to_numpy()
                    vh = actual_vh.to_numpy()
                    np.testing.assert_allclose(
                        singular_values,
                        expected_s,
                        rtol=4e-12,
                        atol=4e-12,
                    )
                    np.testing.assert_allclose(
                        (u * singular_values[..., np.newaxis, :]) @ vh,
                        value,
                        rtol=4e-12,
                        atol=4e-12,
                    )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_svd_v2_dtype_promotion_complex_and_unitarity(self) -> None:
        cases = (
            np.array([[3.0, 1.0], [0.0, 2.0], [1.0, -1.0]], np.float32),
            np.array([[1, 2, 0], [0, 1, 3]], np.int32),
            np.array(
                [[1.0 + 2.0j, 2.0], [0.0, 3.0 - 1.0j], [1.0j, -2.0]],
                np.complex64,
            ),
            np.array(
                [[1.0 + 1.0j, 2.0, -1.0j], [3.0, -2.0j, 0.5]],
                np.complex128,
            ),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for value in cases:
                with self.subTest(dtype=str(value.dtype), shape=value.shape):
                    expected_u, expected_s, expected_vh = np.linalg.svd(
                        value, full_matrices=False
                    )
                    operand = stack.enter_context(runtime.from_numpy(value))
                    actual = self.svd_v2_result(
                        runtime, operand, full_matrices=False
                    )
                    actual_u, actual_s, actual_vh = (
                        stack.enter_context(result) for result in actual
                    )
                    self.assertEqual(expected_u.dtype, actual_u.numpy_dtype)
                    self.assertEqual(expected_s.dtype, actual_s.numpy_dtype)
                    self.assertEqual(expected_vh.dtype, actual_vh.numpy_dtype)
                    u = actual_u.to_numpy()
                    singular_values = actual_s.to_numpy()
                    vh = actual_vh.to_numpy()
                    tolerance = (
                        5e-5 if expected_s.dtype == np.dtype(np.float32)
                        else 5e-12
                    )
                    np.testing.assert_allclose(
                        singular_values,
                        expected_s,
                        rtol=tolerance,
                        atol=tolerance,
                    )
                    np.testing.assert_allclose(
                        (u * singular_values[np.newaxis, :]) @ vh,
                        value,
                        rtol=tolerance,
                        atol=tolerance,
                    )
                    np.testing.assert_allclose(
                        u.conj().T @ u,
                        np.eye(u.shape[1]),
                        rtol=tolerance,
                        atol=tolerance,
                    )
                    np.testing.assert_allclose(
                        vh @ vh.conj().T,
                        np.eye(vh.shape[0]),
                        rtol=tolerance,
                        atol=tolerance,
                    )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_svd_v2_reads_noncontiguous_complex_view(self) -> None:
        value = np.array(
            [[1.0 + 2.0j, 2.0], [0.0, 3.0 - 1.0j], [1.0j, -2.0]],
            dtype=np.complex128,
        )
        storage_value = value.T.copy()
        expected_u, expected_s, expected_vh = np.linalg.svd(
            value, full_matrices=False
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            storage = stack.enter_context(runtime.from_numpy(storage_value))
            view = stack.enter_context(runtime.transpose(storage, (1, 0)))
            self.assertFalse(view.c_contiguous)
            actual = self.svd_v2_result(
                runtime, view, full_matrices=False
            )
            actual_u, actual_s, actual_vh = (
                stack.enter_context(result) for result in actual
            )
            np.testing.assert_allclose(
                actual_s.to_numpy(), expected_s, rtol=5e-12, atol=5e-12
            )
            np.testing.assert_allclose(
                (actual_u.to_numpy() * actual_s.to_numpy()[np.newaxis, :])
                @ actual_vh.to_numpy(),
                value,
                rtol=5e-12,
                atol=5e-12,
            )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_svd_v2_compute_uv_false_and_zero_sized_shapes(self) -> None:
        cases = (
            np.arange(24, dtype=np.float32).reshape(2, 4, 3) - 4.0,
            np.array(
                [[1.0 + 2.0j, 2.0], [0.0, 3.0 - 1.0j]],
                dtype=np.complex128,
            ),
            np.empty((3, 0), dtype=np.float64),
            np.empty((0, 3), dtype=np.complex64),
            np.empty((0, 2, 3), dtype=np.float32),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for value in cases:
                with self.subTest(shape=value.shape, dtype=str(value.dtype)):
                    expected = np.linalg.svd(value, compute_uv=False)
                    operand = stack.enter_context(runtime.from_numpy(value))
                    result = stack.enter_context(
                        self.svd_v2_result(
                            runtime,
                            operand,
                            full_matrices=True,
                            compute_uv=False,
                        )
                    )
                    assert_array_equivalent(
                        self,
                        result,
                        expected,
                        compare_contiguity=False,
                        rtol=5e-5 if expected.dtype == np.float32 else 5e-12,
                        atol=5e-5 if expected.dtype == np.float32 else 5e-12,
                    )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_svd_v2_complete_wide_and_hermitian_factors(self) -> None:
        cases = (
            (
                np.array(
                    [
                        [1.0 + 1.0j, 2.0, -1.0j],
                        [3.0, -2.0j, 0.5],
                    ],
                    dtype=np.complex128,
                ),
                False,
            ),
            (
                np.array(
                    [[2.0, 1.0j], [-1.0j, -3.0]],
                    dtype=np.complex128,
                ),
                True,
            ),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for value, hermitian in cases:
                with self.subTest(shape=value.shape, hermitian=hermitian):
                    expected_u, expected_s, expected_vh = np.linalg.svd(
                        value, full_matrices=True, hermitian=hermitian
                    )
                    operand = stack.enter_context(runtime.from_numpy(value))
                    actual = self.svd_v2_result(
                        runtime,
                        operand,
                        full_matrices=True,
                        hermitian=hermitian,
                    )
                    actual_u, actual_s, actual_vh = (
                        stack.enter_context(result) for result in actual
                    )
                    self.assertEqual(expected_u.shape, actual_u.shape)
                    self.assertEqual(expected_s.shape, actual_s.shape)
                    self.assertEqual(expected_vh.shape, actual_vh.shape)
                    u = actual_u.to_numpy()
                    singular_values = actual_s.to_numpy()
                    vh = actual_vh.to_numpy()
                    rank = min(value.shape[-2:])
                    np.testing.assert_allclose(
                        singular_values, expected_s, rtol=6e-12, atol=6e-12
                    )
                    np.testing.assert_allclose(
                        u[:, :rank] @ np.diag(singular_values) @ vh[:rank, :],
                        value,
                        rtol=6e-12,
                        atol=6e-12,
                    )
                    np.testing.assert_allclose(
                        u.conj().T @ u,
                        np.eye(u.shape[1]),
                        rtol=6e-12,
                        atol=6e-12,
                    )
                    np.testing.assert_allclose(
                        vh @ vh.conj().T,
                        np.eye(vh.shape[0]),
                        rtol=6e-12,
                        atol=6e-12,
                    )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_svd_v2_seeded_dense_and_rank_deficient_differential(self) -> None:
        seed = 20260730
        generator = np.random.default_rng(seed)
        shapes = ((1, 1), (4, 2), (2, 5), (4, 4), (5, 3), (3, 5))
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for dtype in (np.float64, np.complex128):
                for rows, columns in shapes:
                    for rank_deficient in (False, True):
                        value = generator.standard_normal((rows, columns)).astype(
                            dtype
                        )
                        if dtype == np.complex128:
                            value += 1j * generator.standard_normal(
                                (rows, columns)
                            )
                        if rank_deficient and columns > 1:
                            value[:, -1] = value[:, 0]
                        with self.subTest(
                            seed=seed,
                            dtype=str(np.dtype(dtype)),
                            shape=value.shape,
                            rank_deficient=rank_deficient,
                        ), ExitStack() as stack:
                            before = runtime.retained_bytes
                            expected = np.linalg.svd(value, compute_uv=False)
                            operand = stack.enter_context(
                                runtime.from_numpy(value)
                            )
                            u, s, vh = self.svd_v2_result(
                                runtime, operand, full_matrices=False
                            )
                            stack.enter_context(u)
                            stack.enter_context(s)
                            stack.enter_context(vh)
                            actual_u = u.to_numpy()
                            actual_s = s.to_numpy()
                            actual_vh = vh.to_numpy()
                            np.testing.assert_allclose(
                                actual_s,
                                expected,
                                rtol=2e-10,
                                atol=2e-10,
                            )
                            np.testing.assert_allclose(
                                (actual_u * actual_s[np.newaxis, :])
                                @ actual_vh,
                                value,
                                rtol=2e-10,
                                atol=2e-10,
                            )
                            stack.close()
                            self.assertEqual(before, runtime.retained_bytes)

            self.assertEqual(baseline, runtime.retained_bytes)

    def test_svd_v2_validation_is_explicit_and_atomic(self) -> None:
        square_value = np.eye(2, dtype=np.float64)
        vector_value = np.arange(3, dtype=np.float64)
        rectangular_value = np.arange(6, dtype=np.float64).reshape(2, 3)
        nonfinite_values = (
            np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=np.float64),
            np.array([[np.inf, 0.0], [0.0, 1.0]], dtype=np.float64),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            square = stack.enter_context(runtime.from_numpy(square_value))
            vector = stack.enter_context(runtime.from_numpy(vector_value))
            rectangular = stack.enter_context(
                runtime.from_numpy(rectangular_value)
            )
            nonfinite = tuple(
                stack.enter_context(runtime.from_numpy(value))
                for value in nonfinite_values
            )
            function = self.required_export(runtime, "cnp_linalg_svd_v2")
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_bool,
                ctypes.c_bool,
                ctypes.c_bool,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
            ]
            function.restype = ctypes.c_int
            cases = (
                (None, False, -1, "input"),
                (vector.pointer, False, -4, "dimensions"),
                (rectangular.pointer, True, -4, "square"),
                (nonfinite[0].pointer, False, -10, "nan"),
                (nonfinite[1].pointer, False, -10, "infinity"),
            )
            for source, hermitian, expected_status, message in cases:
                with self.subTest(message=message):
                    pointers = [ctypes.c_void_p(index + 1) for index in range(3)]
                    before = runtime.retained_bytes
                    runtime.dll.cnp_clear_error()
                    status = int(
                        function(
                            source,
                            False,
                            True,
                            hermitian,
                            *(ctypes.byref(pointer) for pointer in pointers),
                        )
                    )
                    self.assertEqual(expected_status, status)
                    self.assertTrue(all(not pointer.value for pointer in pointers))
                    error = runtime.error_state()
                    self.assertEqual("cnp_linalg_svd_v2", error.function)
                    self.assertIn(message, error.message.lower())
                    self.assertEqual(before, runtime.retained_bytes)

            singular_slot = ctypes.c_void_p(2)
            vh_slot = ctypes.c_void_p(3)
            runtime.dll.cnp_clear_error()
            status = int(
                function(
                    square.pointer,
                    False,
                    True,
                    False,
                    None,
                    ctypes.byref(singular_slot),
                    ctypes.byref(vh_slot),
                )
            )
            self.assertEqual(-1, status)
            self.assertFalse(singular_slot.value)
            self.assertFalse(vh_slot.value)

            alias_slot = ctypes.c_void_p(4)
            vh_slot = ctypes.c_void_p(5)
            runtime.dll.cnp_clear_error()
            status = int(
                function(
                    square.pointer,
                    False,
                    True,
                    False,
                    ctypes.byref(alias_slot),
                    ctypes.byref(alias_slot),
                    ctypes.byref(vh_slot),
                )
            )
            self.assertEqual(-1, status)
            self.assertFalse(alias_slot.value)
            self.assertFalse(vh_slot.value)

            singular_slot = ctypes.c_void_p()
            runtime.dll.cnp_clear_error()
            status = int(
                function(
                    square.pointer,
                    True,
                    False,
                    False,
                    None,
                    ctypes.byref(singular_slot),
                    None,
                )
            )
            self.assertEqual(0, status)
            result = stack.enter_context(
                runtime._owned_result(
                    singular_slot.value, "cnp_linalg_svd_v2:s"
                )
            )
            np.testing.assert_allclose(result.to_numpy(), np.ones(2))

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_solve_square_batched_rhs_dtypes_and_lifetimes_match_numpy_125(
        self,
    ) -> None:
        cases = (
            (
                np.asarray([[3, 1], [1, 2]], dtype=np.float32),
                np.asarray([9, 8], dtype=np.float32),
            ),
            (
                np.asarray([[3, 1], [1, 2]], dtype=np.float64),
                np.asarray([[9, 1], [8, 2]], dtype=np.float64),
            ),
            (
                np.asarray(
                    [[3 + 1j, 1 - 2j], [1 + 0.5j, 2 - 1j]],
                    dtype=np.complex128,
                ),
                np.asarray([9 - 1j, 8 + 2j], dtype=np.complex128),
            ),
            (
                np.asarray(
                    [
                        [[3, 1], [1, 2]],
                        [[4, -1], [2, 3]],
                        [[2, 0.5], [-1, 5]],
                    ],
                    dtype=np.float64,
                ),
                np.asarray(
                    [[9, 8], [7, 4], [3, -2]], dtype=np.float64
                ),
            ),
            (
                np.asarray([[[3, 1], [1, 2]]], dtype=np.float64),
                np.asarray(
                    [
                        [[9, 1], [8, 2]],
                        [[3, 2], [4, 5]],
                        [[-1, 6], [7, -2]],
                    ],
                    dtype=np.float64,
                ),
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for a_value, b_value in cases:
                expected = np.linalg.solve(a_value, b_value)
                with self.subTest(
                    a_shape=a_value.shape,
                    b_shape=b_value.shape,
                    dtype=str(expected.dtype),
                ), ExitStack() as stack:
                    a = runtime.from_numpy(a_value)
                    b = runtime.from_numpy(b_value)
                    stack.callback(
                        lambda array=a: None
                        if array._closed else array.close()
                    )
                    stack.callback(
                        lambda array=b: None
                        if array._closed else array.close()
                    )
                    result = self.solve_result(runtime, a, b)
                    a.close()
                    b.close()
                    stack.enter_context(result)
                    self.assertEqual(expected.shape, result.shape)
                    self.assertEqual(expected.dtype, result.numpy_dtype)
                    np.testing.assert_allclose(
                        result.to_numpy(),
                        expected,
                        rtol=4e-5 if expected.dtype == np.float32 else 4e-12,
                        atol=4e-5 if expected.dtype == np.float32 else 4e-12,
                    )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_solve_zero_batch_broadcasts_without_reading_empty_sources(
        self,
    ) -> None:
        cases = (
            ((0, 2, 2), (1, 2)),
            ((1, 2, 2), (0, 2)),
            ((2, 0, 2, 2), (1, 1, 2)),
            ((0, 2, 2), (1, 2, 3)),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for a_shape, b_shape in cases:
                a_value = np.empty(a_shape, dtype=np.float64)
                b_value = np.empty(b_shape, dtype=np.float64)
                expected = np.linalg.solve(a_value, b_value)
                with self.subTest(
                    a_shape=a_shape, b_shape=b_shape
                ), ExitStack() as stack:
                    a = runtime.from_numpy(a_value)
                    b = runtime.from_numpy(b_value)
                    stack.callback(
                        lambda: None if a._closed else a.close()
                    )
                    stack.callback(
                        lambda: None if b._closed else b.close()
                    )
                    result = self.solve_result(runtime, a, b)
                    a.close()
                    b.close()
                    stack.enter_context(result)
                    self.assertEqual(expected.shape, result.shape)
                    self.assertEqual(expected.dtype, result.numpy_dtype)
                    np.testing.assert_array_equal(result.to_numpy(), expected)
                self.assertEqual(baseline, runtime.retained_bytes)

    def test_lstsq_v2_rectangular_outputs_rcond_and_lifetimes_match_numpy_125(
        self,
    ) -> None:
        cases = (
            (
                np.asarray([[1, 0], [1, 1], [1, 2]], dtype=np.float64),
                np.asarray([1, 2, 2], dtype=np.float64),
                None,
            ),
            (
                np.asarray([[1, 0, 1], [0, 1, 1]], dtype=np.float32),
                np.asarray([[1, 2], [3, 4]], dtype=np.float32),
                -1.0,
            ),
            (
                np.asarray(
                    [[1 + 1j, 0], [1, 1 - 2j], [0.5j, 2]],
                    dtype=np.complex128,
                ),
                np.asarray([1 - 1j, 2 + 0.5j, -3j], dtype=np.complex128),
                1e-9,
            ),
            (
                np.asarray([[1, 2], [2, 4], [3, 6]], dtype=np.float64),
                np.asarray([1, 2, 3], dtype=np.float64),
                0.25,
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for a_value, b_value, rcond in cases:
                expected = np.linalg.lstsq(a_value, b_value, rcond=rcond)
                with self.subTest(
                    a_shape=a_value.shape,
                    b_shape=b_value.shape,
                    rcond=rcond,
                ), ExitStack() as stack:
                    a = runtime.from_numpy(a_value)
                    b = runtime.from_numpy(b_value)
                    stack.callback(
                        lambda array=a: None
                        if array._closed else array.close()
                    )
                    stack.callback(
                        lambda array=b: None
                        if array._closed else array.close()
                    )
                    results = self.lstsq_v2_result(
                        runtime,
                        a,
                        b,
                        rcond=0.0 if rcond is None else rcond,
                        rcond_none=rcond is None,
                    )
                    a.close()
                    b.close()
                    x, residuals, rank, singular_values = tuple(
                        stack.enter_context(result) for result in results
                    )
                    for actual, wanted in zip(
                        (x, residuals, rank, singular_values), expected
                    ):
                        self.assertEqual(wanted.shape, actual.shape)
                        self.assertEqual(wanted.dtype, actual.numpy_dtype)
                        np.testing.assert_allclose(
                            actual.to_numpy(),
                            wanted,
                            rtol=5e-5
                            if wanted.dtype in (np.float32, np.complex64)
                            else 5e-11,
                            atol=5e-5
                            if wanted.dtype in (np.float32, np.complex64)
                            else 5e-11,
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_lstsq_v2_numpy_125_rcond_boundary_values(self) -> None:
        a_value = np.diag(np.asarray([2.0, 1e-3], dtype=np.float64))
        b_value = np.asarray([2.0, 1e-3], dtype=np.float64)
        rconds = (
            None,
            -1.0,
            0.25,
            np.nextafter(1.0, 0.0),
            1.0,
            2.0,
            np.inf,
            np.nan,
            -np.inf,
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for rcond in rconds:
                expected = np.linalg.lstsq(a_value, b_value, rcond=rcond)
                with self.subTest(rcond=rcond), ExitStack() as stack:
                    a = runtime.from_numpy(a_value)
                    b = runtime.from_numpy(b_value)
                    stack.callback(
                        lambda: None if a._closed else a.close()
                    )
                    stack.callback(
                        lambda: None if b._closed else b.close()
                    )
                    results = self.lstsq_v2_result(
                        runtime,
                        a,
                        b,
                        rcond=0.0 if rcond is None else float(rcond),
                        rcond_none=rcond is None,
                    )
                    a.close()
                    b.close()
                    actual = tuple(
                        stack.enter_context(result) for result in results
                    )
                    for got, wanted in zip(actual, expected):
                        self.assertEqual(wanted.shape, got.shape)
                        self.assertEqual(wanted.dtype, got.numpy_dtype)
                        np.testing.assert_allclose(
                            got.to_numpy(), wanted, rtol=5e-11, atol=5e-11
                        )
                self.assertEqual(baseline, runtime.retained_bytes)

    def test_cond_v2_rectangular_batched_singular_and_lifetimes_match_numpy(
        self,
    ) -> None:
        values = (
            np.asarray([[3, 1], [1, 2]], dtype=np.float64),
            np.asarray([[1, 2], [2, 4]], dtype=np.float32),
            np.asarray(
                [
                    [[3, 1], [1, 2]],
                    [[1, 2], [2, 4]],
                    [[4, -1], [2, 3]],
                ],
                dtype=np.float64,
            ),
            np.asarray([[1, 0], [0, 2], [1, -1]], dtype=np.float64),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for value in values:
                expected = np.linalg.cond(value)
                with self.subTest(
                    shape=value.shape, dtype=str(value.dtype)
                ), ExitStack() as stack:
                    source = runtime.from_numpy(value)
                    stack.callback(
                        lambda array=source: None
                        if array._closed else array.close()
                    )
                    result = self.cond_v2_result(runtime, source)
                    source.close()
                    stack.enter_context(result)
                    self.assertEqual(np.shape(expected), result.shape)
                    self.assertEqual(
                        np.asarray(expected).dtype, result.numpy_dtype
                    )
                    actual = result.to_numpy()
                    rank = np.linalg.matrix_rank(value)
                    full_rank = min(value.shape[-2:])
                    rank_deficient = rank < full_rank
                    if np.any(rank_deficient):
                        precision = np.finfo(value.dtype).eps
                        deficient_values = np.asarray(actual)[rank_deficient]
                        self.assertTrue(
                            np.all(
                                np.isinf(deficient_values)
                                | (deficient_values >= 1.0 / precision)
                            )
                        )
                        if np.ndim(rank_deficient) == 0:
                            continue
                        regular = ~rank_deficient
                        np.testing.assert_allclose(
                            np.asarray(actual)[regular],
                            np.asarray(expected)[regular],
                            rtol=5e-5
                            if value.dtype == np.float32 else 5e-11,
                            atol=5e-5
                            if value.dtype == np.float32 else 5e-11,
                        )
                    else:
                        np.testing.assert_allclose(
                            actual,
                            expected,
                            rtol=5e-5
                            if value.dtype == np.float32 else 5e-11,
                            atol=5e-5
                            if value.dtype == np.float32 else 5e-11,
                        )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_lstsq_and_cond_legacy_projections_delegate_without_silent_errors(
        self,
    ) -> None:
        a_value = np.asarray(
            [[1 + 1j, 0], [1, 1 - 2j], [0.5j, 2]],
            dtype=np.complex128,
        )
        b_value = np.asarray(
            [1 - 1j, 2 + 0.5j, -3j], dtype=np.complex128
        )
        cond_value = np.asarray(
            [[3, 1], [1, 2]], dtype=np.float32
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            a = stack.enter_context(runtime.from_numpy(a_value))
            b = stack.enter_context(runtime.from_numpy(b_value))
            cond_source = stack.enter_context(runtime.from_numpy(cond_value))

            legacy_lstsq = self.required_export(runtime, "cnp_lstsq")
            legacy_lstsq.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_double,
            ]
            legacy_lstsq.restype = ctypes.c_void_p
            runtime.dll.cnp_clear_error()
            pointer = legacy_lstsq(a.pointer, b.pointer, 1e-9)
            result = stack.enter_context(
                runtime._owned_result(pointer, "cnp_lstsq")
            )
            expected = np.linalg.lstsq(a_value, b_value, rcond=1e-9)[0]
            self.assertEqual(expected.dtype, result.numpy_dtype)
            self.assertEqual(expected.shape, result.shape)
            np.testing.assert_allclose(
                result.to_numpy(), expected, rtol=5e-11, atol=5e-11
            )

            legacy_cond = self.required_export(runtime, "cnp_linalg_cond")
            legacy_cond.argtypes = [ctypes.c_void_p]
            legacy_cond.restype = ctypes.c_double
            runtime.dll.cnp_clear_error()
            actual_condition = legacy_cond(cond_source.pointer)
            self.assertEqual(0, runtime.error_state().status)
            self.assertAlmostEqual(
                float(np.linalg.cond(cond_value)),
                actual_condition,
                delta=5e-5,
            )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_lstsq_v2_and_cond_v2_validation_is_explicit_atomic_and_retained0(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            a = stack.enter_context(
                runtime.from_numpy(np.ones((3, 2), dtype=np.float64))
            )
            bad_rhs = stack.enter_context(
                runtime.from_numpy(np.ones(2, dtype=np.float64))
            )
            lstsq = self.required_export(runtime, "cnp_linalg_lstsq_v2")
            lstsq.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_double,
                ctypes.c_bool,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
            ]
            lstsq.restype = ctypes.c_int
            slots = [ctypes.c_void_p(index + 1) for index in range(4)]
            before = runtime.retained_bytes
            status = int(
                lstsq(
                    a.pointer,
                    bad_rhs.pointer,
                    0.0,
                    True,
                    *(ctypes.byref(slot) for slot in slots),
                )
            )
            self.assertEqual(-4, status)
            self.assertTrue(all(not slot.value for slot in slots))
            error = runtime.error_state()
            self.assertEqual("cnp_linalg_lstsq_v2", error.function)
            self.assertIn("row count", error.message.lower())
            self.assertEqual(before, runtime.retained_bytes)

            cond = self.required_export(runtime, "cnp_linalg_cond_v2")
            cond.argtypes = [ctypes.c_void_p]
            cond.restype = ctypes.c_void_p
            runtime.dll.cnp_clear_error()
            self.assertFalse(cond(None))
            error = runtime.error_state()
            self.assertEqual(-1, error.status)
            self.assertEqual("cnp_linalg_cond_v2", error.function)
            self.assertIn("null", error.message.lower())
            self.assertEqual(before, runtime.retained_bytes)

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    @staticmethod
    def selected_hermitian(
        value: np.ndarray, uplo: str
    ) -> np.ndarray:
        selected = np.asarray(value).copy()
        diagonal = np.arange(selected.shape[-1])
        selected[..., diagonal, diagonal] = np.real(
            selected[..., diagonal, diagonal]
        )
        if uplo == "L":
            lower = np.tril(selected)
            return lower + np.swapaxes(np.tril(selected, -1).conj(), -1, -2)
        upper = np.triu(selected)
        return upper + np.swapaxes(np.triu(selected, 1).conj(), -1, -2)

    def assert_eigh_decomposition(
        self,
        source_value: np.ndarray,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        uplo: str,
    ) -> None:
        expected_values, _ = np.linalg.eigh(source_value, UPLO=uplo)
        single_precision = expected_values.dtype == np.dtype(np.float32)
        rtol = 3e-5 if single_precision else 3e-11
        atol = 3e-5 if single_precision else 3e-11
        np.testing.assert_allclose(
            eigenvalues, expected_values, rtol=rtol, atol=atol
        )
        hermitian = self.selected_hermitian(source_value, uplo)
        np.testing.assert_allclose(
            hermitian @ eigenvectors,
            eigenvectors * eigenvalues[..., np.newaxis, :],
            rtol=rtol,
            atol=atol,
        )
        identity = np.broadcast_to(
            np.eye(source_value.shape[-1], dtype=eigenvectors.dtype),
            eigenvectors.shape,
        )
        np.testing.assert_allclose(
            np.swapaxes(eigenvectors.conj(), -1, -2) @ eigenvectors,
            identity,
            rtol=rtol,
            atol=atol,
        )

    def test_eigh_v2_matches_numpy_dtypes_batches_and_uplo(self) -> None:
        base = np.array(
            [
                [[4, 91, -23], [1, 3, 72], [2, -1, 5]],
                [[2, -17, 63], [-2, 6, 44], [1, 3, 7]],
            ],
            dtype=np.float64,
        )
        complex_base = base.astype(np.complex128)
        complex_base[..., 1, 0] += 2j
        complex_base[..., 2, 0] -= 3j
        complex_base[..., 2, 1] += 4j
        complex_base[..., 0, 1] += 7j
        cases = (
            base.astype(np.int64),
            base.astype(np.float32),
            base,
            complex_base.astype(np.complex64),
            complex_base,
        )

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for source_value in cases:
                source = stack.enter_context(runtime.from_numpy(source_value))
                source_before = source.to_numpy().copy()
                for uplo in ("L", "U"):
                    with self.subTest(dtype=str(source_value.dtype), uplo=uplo):
                        eigenvalues, eigenvectors = runtime.eigh(source, uplo)
                        eigenvalues = stack.enter_context(eigenvalues)
                        eigenvectors = stack.enter_context(eigenvectors)
                        expected_values, expected_vectors = np.linalg.eigh(
                            source_value, UPLO=uplo
                        )
                        self.assertEqual(
                            expected_values.dtype, eigenvalues.numpy_dtype
                        )
                        self.assertEqual(
                            expected_vectors.dtype, eigenvectors.numpy_dtype
                        )
                        self.assertEqual(
                            expected_values.shape, eigenvalues.shape
                        )
                        self.assertEqual(
                            expected_vectors.shape, eigenvectors.shape
                        )
                        self.assertTrue(eigenvalues.c_contiguous)
                        self.assertTrue(eigenvectors.c_contiguous)
                        self.assert_eigh_decomposition(
                            source_value,
                            eigenvalues.to_numpy(),
                            eigenvectors.to_numpy(),
                            uplo,
                        )
                np.testing.assert_array_equal(
                    source.to_numpy(), source_before, strict=True
                )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_eigvalsh_v2_matches_numpy_views_and_result_lifetime(self) -> None:
        storage_value = np.array(
            [
                [3 + 8j, 2 + 4j, -1 + 2j],
                [2 - 4j, 5 - 9j, 3 + 1j],
                [-1 - 2j, 3 - 1j, 7 + 6j],
            ],
            dtype=np.complex128,
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            storage = runtime.from_numpy(storage_value)
            source = runtime.transpose(storage, (1, 0))
            result = runtime.eigvalsh(source, "U")
            storage.close()
            source.close()
            try:
                expected = np.linalg.eigvalsh(storage_value.T, UPLO="U")
                np.testing.assert_allclose(
                    result.to_numpy(), expected, rtol=3e-11, atol=3e-11
                )
            finally:
                result.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_eigh_v2_empty_shapes_and_validation_are_atomic(self) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for source_value in (
                np.empty((0, 0), dtype=np.float64),
                np.empty((2, 0, 0), dtype=np.float32),
                np.empty((0, 2, 2), dtype=np.complex128),
            ):
                source = stack.enter_context(runtime.from_numpy(source_value))
                eigenvalues, eigenvectors = runtime.eigh(source)
                eigenvalues = stack.enter_context(eigenvalues)
                eigenvectors = stack.enter_context(eigenvectors)
                expected_values, expected_vectors = np.linalg.eigh(source_value)
                self.assertEqual(expected_values.shape, eigenvalues.shape)
                self.assertEqual(expected_vectors.shape, eigenvectors.shape)
                self.assertEqual(expected_values.dtype, eigenvalues.numpy_dtype)
                self.assertEqual(expected_vectors.dtype, eigenvectors.numpy_dtype)

            function = self.required_export(runtime, "cnp_linalg_eigh_v2")
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_bool,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
            ]
            function.restype = ctypes.c_int
            invalid_sources = (
                (None, -1, "null"),
                (
                    stack.enter_context(
                        runtime.from_numpy(np.array([1.0, 2.0]))
                    ).pointer,
                    -4,
                    "dimensions",
                ),
                (
                    stack.enter_context(
                        runtime.from_numpy(np.ones((2, 3)))
                    ).pointer,
                    -4,
                    "square",
                ),
                (
                    stack.enter_context(
                        runtime.from_numpy(np.ones((2, 2), dtype=np.float16))
                    ).pointer,
                    -3,
                    "dtype",
                ),
            )
            for pointer, expected_status, message in invalid_sources:
                with self.subTest(message=message):
                    values_slot = ctypes.c_void_p(1)
                    vectors_slot = ctypes.c_void_p(2)
                    before = runtime.retained_bytes
                    runtime.dll.cnp_clear_error()
                    status = int(
                        function(
                            pointer,
                            False,
                            ctypes.byref(values_slot),
                            ctypes.byref(vectors_slot),
                        )
                    )
                    self.assertEqual(expected_status, status)
                    self.assertFalse(values_slot.value)
                    self.assertFalse(vectors_slot.value)
                    error = runtime.error_state()
                    self.assertEqual("cnp_linalg_eigh_v2", error.function)
                    self.assertIn(message, error.message.lower())
                    self.assertEqual(before, runtime.retained_bytes)

            vectors_slot = ctypes.c_void_p(2)
            runtime.dll.cnp_clear_error()
            status = int(function(None, False, None, ctypes.byref(vectors_slot)))
            self.assertEqual(-1, status)
            self.assertFalse(vectors_slot.value)

            square = stack.enter_context(
                runtime.from_numpy(np.eye(2, dtype=np.float64))
            )
            alias_slot = ctypes.c_void_p(3)
            runtime.dll.cnp_clear_error()
            status = int(
                function(
                    square.pointer,
                    False,
                    ctypes.byref(alias_slot),
                    ctypes.byref(alias_slot),
                )
            )
            self.assertEqual(-1, status)
            self.assertFalse(alias_slot.value)
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_eigh_v2_nonfinite_triangle_selection_matches_numpy(self) -> None:
        base = np.array(
            [[2 + 5j, 1 + 2j], [1 - 2j, 3 - 7j]],
            dtype=np.complex128,
        )
        ignored_cases = []
        lower = base.copy()
        lower[0, 1] = complex(np.nan, np.inf)
        lower[0, 0] = complex(2.0, np.nan)
        ignored_cases.append((lower, "L"))
        upper = base.copy()
        upper[1, 0] = complex(-np.inf, np.nan)
        upper[1, 1] = complex(3.0, -np.inf)
        ignored_cases.append((upper, "U"))

        selected_cases = []
        lower = base.copy()
        lower[1, 0] = complex(np.nan, -2.0)
        selected_cases.append((lower, "L"))
        upper = base.copy()
        upper[0, 1] = complex(1.0, np.inf)
        selected_cases.append((upper, "U"))

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for source_value, uplo in ignored_cases:
                with self.subTest(kind="ignored", uplo=uplo):
                    source = stack.enter_context(
                        runtime.from_numpy(source_value)
                    )
                    values, vectors = runtime.eigh(source, uplo)
                    values = stack.enter_context(values)
                    vectors = stack.enter_context(vectors)
                    actual_values = values.to_numpy()
                    actual_vectors = vectors.to_numpy()
                    self.assertTrue(np.isfinite(actual_values).all())
                    self.assertTrue(np.isfinite(actual_vectors).all())
                    self.assert_eigh_decomposition(
                        source_value,
                        actual_values,
                        actual_vectors,
                        uplo,
                    )
                    value_only = stack.enter_context(
                        runtime.eigvalsh(source, uplo)
                    )
                    np.testing.assert_allclose(
                        value_only.to_numpy(),
                        np.linalg.eigvalsh(source_value, UPLO=uplo),
                        rtol=3e-11,
                        atol=3e-11,
                    )

            for source_value, uplo in selected_cases:
                with self.subTest(kind="selected", uplo=uplo):
                    source = stack.enter_context(
                        runtime.from_numpy(source_value)
                    )
                    expected_values, expected_vectors = np.linalg.eigh(
                        source_value, UPLO=uplo
                    )
                    values, vectors = runtime.eigh(source, uplo)
                    values = stack.enter_context(values)
                    vectors = stack.enter_context(vectors)
                    np.testing.assert_array_equal(
                        np.isnan(values.to_numpy()),
                        np.isnan(expected_values),
                    )
                    np.testing.assert_array_equal(
                        np.isnan(vectors.to_numpy()),
                        np.isnan(expected_vectors),
                    )
                    value_only = stack.enter_context(
                        runtime.eigvalsh(source, uplo)
                    )
                    np.testing.assert_array_equal(
                        np.isnan(value_only.to_numpy()),
                        np.isnan(
                            np.linalg.eigvalsh(source_value, UPLO=uplo)
                        ),
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_eigh_v2_tiny_scale_and_degenerate_spectra(self) -> None:
        tiny_real = np.array(
            [[3e-300, 9e-300], [2e-300, -2e-300]],
            dtype=np.float64,
        )
        tiny_complex = np.array(
            [
                [4e-300 + 8j, 7e-300 + 5e-300j],
                [1e-300 - 3e-300j, -1e-300 - 9j],
            ],
            dtype=np.complex128,
        )
        generator = np.random.default_rng(20260803)
        basis, _ = np.linalg.qr(
            generator.normal(size=(5, 5))
            + 1j * generator.normal(size=(5, 5))
        )
        repeated = basis @ np.diag([1.0, 1.0, 1.0, 4.0, 9.0]) @ basis.conj().T

        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for source_value, uplo in (
                (tiny_real, "L"),
                (tiny_complex, "L"),
            ):
                with self.subTest(kind="tiny", dtype=str(source_value.dtype)):
                    source = stack.enter_context(
                        runtime.from_numpy(source_value)
                    )
                    values, vectors = runtime.eigh(source, uplo)
                    values = stack.enter_context(values)
                    vectors = stack.enter_context(vectors)
                    expected_values = np.linalg.eigvalsh(
                        source_value, UPLO=uplo
                    )
                    np.testing.assert_allclose(
                        values.to_numpy(),
                        expected_values,
                        rtol=3e-12,
                        atol=0.0,
                    )
                    selected = self.selected_hermitian(source_value, uplo)
                    residual = (
                        selected @ vectors.to_numpy()
                        - vectors.to_numpy()
                        * values.to_numpy()[np.newaxis, :]
                    )
                    self.assertLessEqual(
                        np.linalg.norm(residual),
                        3e-12 * np.linalg.norm(selected),
                    )

            source = stack.enter_context(runtime.from_numpy(repeated))
            values, vectors = runtime.eigh(source)
            values = stack.enter_context(values)
            vectors = stack.enter_context(vectors)
            self.assert_eigh_decomposition(
                repeated,
                values.to_numpy(),
                vectors.to_numpy(),
                "L",
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_eigh_v2_finite_extreme_scale_avoids_intermediate_overflow(
        self,
    ) -> None:
        source_value = np.array(
            [[-1e308, 9e307], [5e307, 1e308]],
            dtype=np.float64,
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            for uplo in ("L", "U"):
                with self.subTest(uplo=uplo):
                    values, vectors = runtime.eigh(source, uplo)
                    values = stack.enter_context(values)
                    vectors = stack.enter_context(vectors)
                    expected_values = np.linalg.eigvalsh(
                        source_value, UPLO=uplo
                    )
                    self.assertTrue(np.isfinite(values.to_numpy()).all())
                    self.assertTrue(np.isfinite(vectors.to_numpy()).all())
                    np.testing.assert_allclose(
                        values.to_numpy(),
                        expected_values,
                        rtol=3e-12,
                        atol=0.0,
                    )
                    selected = self.selected_hermitian(source_value, uplo)
                    scaled_matrix = selected / 1e308
                    scaled_values = values.to_numpy() / 1e308
                    residual = (
                        scaled_matrix @ vectors.to_numpy()
                        - vectors.to_numpy()
                        * scaled_values[np.newaxis, :]
                    )
                    self.assertLessEqual(np.linalg.norm(residual), 3e-12)
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_eigh_legacy_abis_delegate_to_lower_triangle(self) -> None:
        source_value = np.array(
            [[1.0, 99.0], [2.0, 3.0]], dtype=np.float32
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))

            function = self.required_export(runtime, "cnp_linalg_eigh")
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
            ]
            function.restype = ctypes.c_int
            values_slot = ctypes.c_void_p()
            vectors_slot = ctypes.c_void_p()
            runtime.dll.cnp_clear_error()
            status = int(
                function(
                    source.pointer,
                    ctypes.byref(values_slot),
                    ctypes.byref(vectors_slot),
                )
            )
            self.assertEqual(0, status)
            values = stack.enter_context(
                runtime._owned_result(
                    values_slot.value, "cnp_linalg_eigh:values"
                )
            )
            vectors = stack.enter_context(
                runtime._owned_result(
                    vectors_slot.value, "cnp_linalg_eigh:vectors"
                )
            )
            expected_values = np.linalg.eigvalsh(source_value, UPLO="L")
            np.testing.assert_allclose(
                values.to_numpy(), expected_values, rtol=3e-5, atol=3e-5
            )
            self.assert_eigh_decomposition(
                source_value,
                values.to_numpy(),
                vectors.to_numpy(),
                "L",
            )

            value_function = self.required_export(runtime, "cnp_eigvalsh")
            value_function.argtypes = [ctypes.c_void_p]
            value_function.restype = ctypes.c_void_p
            runtime.dll.cnp_clear_error()
            value_only = stack.enter_context(
                runtime._owned_result(
                    value_function(source.pointer), "cnp_eigvalsh"
                )
            )
            np.testing.assert_allclose(
                value_only.to_numpy(),
                expected_values,
                rtol=3e-5,
                atol=3e-5,
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_det_matches_numpy_for_supported_two_dimensional_dtypes(
        self,
    ) -> None:
        values = np.array([[1, 2], [3, 5]])
        dtypes = (
            np.bool_,
            np.int8,
            np.int64,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for dtype in dtypes:
                source_value = values.astype(dtype)
                with self.subTest(dtype=str(source_value.dtype)):
                    source = stack.enter_context(
                        runtime.from_numpy(source_value)
                    )
                    result = stack.enter_context(
                        self.det_result(runtime, source)
                    )
                    expected = np.asarray(np.linalg.det(source_value))
                    self.assertEqual(expected.dtype, result.numpy_dtype)
                    self.assertEqual(expected.shape, result.shape)
                    np.testing.assert_allclose(
                        result.to_numpy(), expected, rtol=2e-6, atol=0.0
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_det_preserves_batch_shape_and_empty_matrix_identities(
        self,
    ) -> None:
        cases = (
            np.array(
                [
                    [[2, 1], [3, 4]],
                    [[-1, 2], [5, 3]],
                ],
                dtype=np.float32,
            ),
            np.empty((2, 0, 0), dtype=np.float64),
            np.empty((0, 2, 2), dtype=np.complex64),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for source_value in cases:
                with self.subTest(shape=source_value.shape):
                    source = stack.enter_context(
                        runtime.from_numpy(source_value)
                    )
                    result = stack.enter_context(
                        self.det_result(runtime, source)
                    )
                    expected = np.asarray(np.linalg.det(source_value))
                    self.assertEqual(expected.dtype, result.numpy_dtype)
                    self.assertEqual(expected.shape, result.shape)
                    np.testing.assert_allclose(
                        result.to_numpy(), expected, rtol=2e-6, atol=0.0
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_slogdet_v2_matches_numpy_sign_phase_and_result_dtypes(
        self,
    ) -> None:
        values = np.array([[1, 2], [3, 5]])
        dtypes = (
            np.bool_,
            np.int64,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for dtype in dtypes:
                source_value = values.astype(dtype)
                if np.issubdtype(dtype, np.complexfloating):
                    source_value[0, 0] += dtype(2j)
                    source_value[1, 1] -= dtype(4j)
                with self.subTest(dtype=str(source_value.dtype)):
                    source = stack.enter_context(
                        runtime.from_numpy(source_value)
                    )
                    sign, logabsdet = self.slogdet_result(runtime, source)
                    sign = stack.enter_context(sign)
                    logabsdet = stack.enter_context(logabsdet)
                    expected_sign, expected_logabsdet = np.linalg.slogdet(
                        source_value
                    )
                    expected_sign = np.asarray(expected_sign)
                    expected_logabsdet = np.asarray(expected_logabsdet)
                    self.assertEqual(expected_sign.dtype, sign.numpy_dtype)
                    self.assertEqual(
                        expected_logabsdet.dtype, logabsdet.numpy_dtype
                    )
                    self.assertEqual(expected_sign.shape, sign.shape)
                    self.assertEqual(expected_logabsdet.shape, logabsdet.shape)
                    np.testing.assert_allclose(
                        sign.to_numpy(), expected_sign,
                        rtol=2e-6, atol=2e-6
                    )
                    np.testing.assert_allclose(
                        logabsdet.to_numpy(), expected_logabsdet,
                        rtol=2e-6, atol=2e-6
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_det_and_slogdet_v2_propagate_nonfinite_lu_results(
        self,
    ) -> None:
        cases = (
            np.diag([np.nan, 1.0]),
            np.array([[1.0, np.inf], [0.0, 1.0]]),
            np.diag([np.inf, 1.0]),
            np.diag(np.array([np.inf + 0j, 1.0 + 0j])),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for source_value in cases:
                with self.subTest(source=repr(source_value)):
                    source = stack.enter_context(
                        runtime.from_numpy(source_value)
                    )
                    determinant = stack.enter_context(
                        self.det_result(runtime, source)
                    )
                    sign, logabsdet = self.slogdet_result(runtime, source)
                    sign = stack.enter_context(sign)
                    logabsdet = stack.enter_context(logabsdet)
                    with np.errstate(all="ignore"):
                        expected_determinant = np.asarray(
                            np.linalg.det(source_value)
                        )
                        expected_sign, expected_logabsdet = (
                            np.linalg.slogdet(source_value)
                        )
                    np.testing.assert_allclose(
                        determinant.to_numpy(), expected_determinant,
                        rtol=0.0, atol=0.0, equal_nan=True
                    )
                    np.testing.assert_allclose(
                        sign.to_numpy(), np.asarray(expected_sign),
                        rtol=0.0, atol=0.0, equal_nan=True
                    )
                    np.testing.assert_allclose(
                        logabsdet.to_numpy(),
                        np.asarray(expected_logabsdet),
                        rtol=0.0, atol=0.0, equal_nan=True
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_real_det_and_slogdet_match_numpy_for_every_nonfinite_2x2(
        self,
    ) -> None:
        values = np.array(
            [-np.inf, -1.0, -0.0, 0.0, 1.0, np.inf, np.nan],
            dtype=np.float64,
        )
        source_value = np.array(
            np.meshgrid(values, values, values, values, indexing="ij")
        ).reshape(4, -1).T.reshape(-1, 2, 2)
        with np.errstate(all="ignore"):
            expected_determinant = np.asarray(np.linalg.det(source_value))
            expected_sign, expected_logabsdet = np.linalg.slogdet(source_value)
            expected_sign = np.asarray(expected_sign)
            expected_logabsdet = np.asarray(expected_logabsdet)

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(source_value) as source:
                with self.det_result(runtime, source) as determinant:
                    sign, logabsdet = self.slogdet_result(runtime, source)
                    with sign, logabsdet:
                        actual_determinant = determinant.to_numpy().copy()
                        actual_sign = sign.to_numpy().copy()
                        actual_logabsdet = logabsdet.to_numpy().copy()
            self.assertEqual(baseline, runtime.retained_bytes)

        mismatches = {
            "det": int(
                np.count_nonzero(
                    self._ieee_mismatch_mask(
                        actual_determinant, expected_determinant
                    )
                )
            ),
            "sign": int(
                np.count_nonzero(
                    self._ieee_mismatch_mask(actual_sign, expected_sign)
                )
            ),
            "logabsdet": int(
                np.count_nonzero(
                    self._ieee_mismatch_mask(
                        actual_logabsdet, expected_logabsdet
                    )
                )
            ),
        }
        self.assertEqual(
            {"det": 0, "sign": 0, "logabsdet": 0}, mismatches
        )

    def test_complex_det_and_slogdet_match_numpy_getrf_special_values(
        self,
    ) -> None:
        for dtype in (np.complex64, np.complex128):
            real_dtype = np.empty((), dtype=dtype).real.dtype
            maximum = np.finfo(real_dtype).max
            cases = (
                np.array(
                    [[1 + 0j, np.inf + 0j], [1 + 0j, 1 + 0j]],
                    dtype=dtype,
                ),
                np.array(
                    [[1 + 0j, complex(0.0, np.inf)], [1 + 0j, 1 + 0j]],
                    dtype=dtype,
                ),
                np.array(
                    [
                        [1 + 0j, complex(np.inf, np.nan)],
                        [1 + 0j, 1 + 0j],
                    ],
                    dtype=dtype,
                ),
                np.array([[complex(maximum, maximum)]], dtype=dtype),
                np.array(
                    [[5 + 0j, 1 + 0j], [3 + 3j, 2 + 0j]],
                    dtype=dtype,
                ),
            )
            with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
                baseline = runtime.retained_bytes
                for source_value in cases:
                    with self.subTest(
                        dtype=str(source_value.dtype), source=repr(source_value)
                    ):
                        source = stack.enter_context(
                            runtime.from_numpy(source_value)
                        )
                        determinant = stack.enter_context(
                            self.det_result(runtime, source)
                        )
                        sign, logabsdet = self.slogdet_result(runtime, source)
                        sign = stack.enter_context(sign)
                        logabsdet = stack.enter_context(logabsdet)
                        with np.errstate(all="ignore"):
                            expected_determinant = np.asarray(
                                np.linalg.det(source_value)
                            )
                            expected_sign, expected_logabsdet = (
                                np.linalg.slogdet(source_value)
                            )
                        rtol = 2e-6 if dtype is np.complex64 else 2e-12
                        self.assert_special_value_pattern(
                            determinant.to_numpy(),
                            expected_determinant,
                            rtol=rtol,
                        )
                        self.assert_special_value_pattern(
                            sign.to_numpy(), np.asarray(expected_sign), rtol=rtol
                        )
                        self.assert_special_value_pattern(
                            logabsdet.to_numpy(),
                            np.asarray(expected_logabsdet),
                            rtol=rtol,
                        )
                stack.close()
                self.assertEqual(baseline, runtime.retained_bytes)

    def test_real_det_and_slogdet_match_numpy_pivot_scaling_boundaries(
        self,
    ) -> None:
        safe_minimum = np.finfo(np.float64).tiny
        true_minimum = np.nextafter(0.0, 1.0)
        source_value = np.array(
            [
                [[np.nan, -1.0], [-1.0, 1.0]],
                [[-np.inf, -1.0], [-np.inf, 0.0]],
                [[np.inf, -1.0], [np.inf, 0.0]],
                [[safe_minimum, 1.0], [safe_minimum, 2.0]],
                [[true_minimum, 1.0], [true_minimum, 2.0]],
                [[-true_minimum, 1.0], [-true_minimum, 2.0]],
            ],
            dtype=np.float64,
        )
        with np.errstate(all="ignore"):
            expected_determinant = np.asarray(np.linalg.det(source_value))
            expected_sign, expected_logabsdet = np.linalg.slogdet(source_value)
            expected_sign = np.asarray(expected_sign)
            expected_logabsdet = np.asarray(expected_logabsdet)

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(source_value) as source:
                with self.det_result(runtime, source) as determinant:
                    sign, logabsdet = self.slogdet_result(runtime, source)
                    with sign, logabsdet:
                        actual_determinant = determinant.to_numpy().copy()
                        actual_sign = sign.to_numpy().copy()
                        actual_logabsdet = logabsdet.to_numpy().copy()
            self.assertEqual(baseline, runtime.retained_bytes)

        mismatches = {
            "det": int(np.count_nonzero(self._ieee_mismatch_mask(
                actual_determinant, expected_determinant
            ))),
            "sign": int(np.count_nonzero(self._ieee_mismatch_mask(
                actual_sign, expected_sign
            ))),
            "logabsdet": int(np.count_nonzero(self._ieee_mismatch_mask(
                actual_logabsdet, expected_logabsdet
            ))),
        }
        self.assertEqual(
            {"det": 0, "sign": 0, "logabsdet": 0}, mismatches
        )

    def test_ahk_det_and_slogdet_v2_bridges_preserve_results_atomically(
        self,
    ) -> None:
        source_value = np.array(
            [
                [[1 + 2j, 2 - 1j], [3 + 4j, 5 + 1j]],
                [[2 - 3j, 1 + 1j], [-4 + 2j, 3 - 2j]],
            ],
            dtype=np.complex64,
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))

            det_function = self.required_export(
                runtime, "cnp_ahk_linalg_det_v2"
            )
            det_function.argtypes = [ctypes.c_void_p]
            det_function.restype = ctypes.c_void_p
            runtime.dll.cnp_clear_error()
            determinant = stack.enter_context(
                runtime._owned_result(
                    det_function(source.pointer),
                    "cnp_ahk_linalg_det_v2",
                )
            )
            expected_determinant = np.linalg.det(source_value)
            self.assertEqual(expected_determinant.dtype, determinant.numpy_dtype)
            np.testing.assert_allclose(
                determinant.to_numpy(), expected_determinant,
                rtol=2e-6, atol=2e-6
            )

            slogdet_function = self.required_export(
                runtime, "cnp_ahk_linalg_slogdet_v2"
            )
            slogdet_function.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_int,
            ]
            slogdet_function.restype = ctypes.c_int
            results = (ctypes.c_void_p * 2)()
            runtime.dll.cnp_clear_error()
            status = int(slogdet_function(source.pointer, results, 2))
            self.assertEqual(0, status)
            sign = stack.enter_context(
                runtime._owned_result(
                    results[0], "cnp_ahk_linalg_slogdet_v2:sign"
                )
            )
            logabsdet = stack.enter_context(
                runtime._owned_result(
                    results[1], "cnp_ahk_linalg_slogdet_v2:logabsdet"
                )
            )
            expected_sign, expected_logabsdet = np.linalg.slogdet(source_value)
            np.testing.assert_allclose(
                sign.to_numpy(), expected_sign, rtol=2e-6, atol=2e-6
            )
            np.testing.assert_allclose(
                logabsdet.to_numpy(), expected_logabsdet,
                rtol=2e-6, atol=2e-6
            )

            before_failure = runtime.retained_bytes
            insufficient = (ctypes.c_void_p * 1)(1)
            runtime.dll.cnp_clear_error()
            status = int(slogdet_function(source.pointer, insufficient, 1))
            self.assertEqual(-4, status)
            self.assertFalse(insufficient[0])
            self.assertEqual(
                "cnp_ahk_linalg_slogdet_v2",
                runtime.error_state().function,
            )
            self.assertEqual(before_failure, runtime.retained_bytes)

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_slogdet_double_outputs_delegate_without_scale_cutoff(
        self,
    ) -> None:
        source_value = np.diag([1e-310, -2e-310])
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            function = self.required_export(runtime, "cnp_slogdet")
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
            ]
            function.restype = ctypes.c_int
            sign = ctypes.c_double()
            logabsdet = ctypes.c_double()
            runtime.dll.cnp_clear_error()
            status = int(
                function(
                    source.pointer,
                    ctypes.byref(sign),
                    ctypes.byref(logabsdet),
                )
            )
            self.assertEqual(0, status)
            expected_sign, expected_logabsdet = np.linalg.slogdet(
                source_value
            )
            self.assertEqual(expected_sign, sign.value)
            self.assertAlmostEqual(expected_logabsdet, logabsdet.value, 12)
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_linalg_slogdet_array_delegates_without_underflow(
        self,
    ) -> None:
        source_value = np.diag([1e-310, -2e-310])
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = stack.enter_context(runtime.from_numpy(source_value))
            function = self.required_export(
                runtime, "cnp_linalg_slogdet"
            )
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_void_p
            runtime.dll.cnp_clear_error()
            result = stack.enter_context(
                runtime._owned_result(
                    function(source.pointer), "cnp_linalg_slogdet"
                )
            )
            expected_sign, expected_logabsdet = np.linalg.slogdet(
                source_value
            )
            np.testing.assert_allclose(
                result.to_numpy(),
                np.array([expected_sign, expected_logabsdet]),
                rtol=0.0,
                atol=1e-12,
            )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_legacy_slogdet_rejects_complex_and_batched_inputs_atomically(
        self,
    ) -> None:
        invalid_cases = (
            (
                np.eye(2, dtype=np.complex128),
                -3,
                "complex",
            ),
            (
                np.ones((2, 2, 2), dtype=np.float64),
                -4,
                "two-dimensional",
            ),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            scalar_function = self.required_export(runtime, "cnp_slogdet")
            scalar_function.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
            ]
            scalar_function.restype = ctypes.c_int
            packed_function = self.required_export(
                runtime, "cnp_linalg_slogdet"
            )
            packed_function.argtypes = [ctypes.c_void_p]
            packed_function.restype = ctypes.c_void_p

            for source_value, expected_status, expected_message in invalid_cases:
                source = stack.enter_context(runtime.from_numpy(source_value))
                before = runtime.retained_bytes

                sign = ctypes.c_double(123.0)
                logabsdet = ctypes.c_double(-456.0)
                runtime.dll.cnp_clear_error()
                status = int(
                    scalar_function(
                        source.pointer,
                        ctypes.byref(sign),
                        ctypes.byref(logabsdet),
                    )
                )
                self.assertEqual(expected_status, status)
                self.assertEqual(0.0, sign.value)
                self.assertEqual(0.0, logabsdet.value)
                self.assertFalse(np.signbit(sign.value))
                self.assertFalse(np.signbit(logabsdet.value))
                error = runtime.error_state()
                self.assertEqual(expected_status, error.status)
                self.assertEqual("cnp_slogdet", error.function)
                self.assertIn(expected_message, error.message.lower())
                self.assertEqual(before, runtime.retained_bytes)

                runtime.dll.cnp_clear_error()
                result = packed_function(source.pointer)
                self.assertFalse(result)
                error = runtime.error_state()
                self.assertEqual(expected_status, error.status)
                self.assertEqual("cnp_linalg_slogdet", error.function)
                self.assertIn(expected_message, error.message.lower())
                self.assertEqual(before, runtime.retained_bytes)

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_det_and_slogdet_v2_views_batches_scales_and_lifetimes(
        self,
    ) -> None:
        storage_value = np.array(
            [
                [[1 + 2j, 3 - 1j], [2 + 4j, 5 + 1j]],
                [[4 - 2j, 1 + 3j], [-2 + 1j, 6 - 4j]],
            ],
            dtype=np.complex128,
        )
        with (
            CnumpyRuntime(DLL) as runtime,
            ExitStack() as result_stack,
            ExitStack() as source_stack,
        ):
            baseline = runtime.retained_bytes
            storage = source_stack.enter_context(
                runtime.from_numpy(storage_value)
            )
            source = source_stack.enter_context(
                runtime.transpose(storage, (0, 2, 1))
            )
            determinant = result_stack.enter_context(
                self.det_result(runtime, source)
            )
            sign, logabsdet = self.slogdet_result(runtime, source)
            sign = result_stack.enter_context(sign)
            logabsdet = result_stack.enter_context(logabsdet)
            source_stack.close()
            expected_source = storage_value.transpose(0, 2, 1)
            expected_determinant = np.linalg.det(expected_source)
            expected_sign, expected_logabsdet = np.linalg.slogdet(
                expected_source
            )
            np.testing.assert_allclose(
                determinant.to_numpy(), expected_determinant,
                rtol=2e-12, atol=2e-12
            )
            np.testing.assert_allclose(
                sign.to_numpy(), expected_sign,
                rtol=2e-12, atol=2e-12
            )
            np.testing.assert_allclose(
                logabsdet.to_numpy(), expected_logabsdet,
                rtol=2e-12, atol=2e-12
            )
            result_stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

        real_storage_value = np.array(
            [
                [[2.0, -1.0, 4.0], [1.0, 2.0, 1.0]],
                [[3.0, 5.0, 2.0], [4.0, 3.0, 3.0]],
            ],
            dtype=np.float64,
        )
        expected_source = real_storage_value.transpose(2, 0, 1)
        contiguous_anti_source = real_storage_value.reshape(3, 2, 2)
        self.assertTrue(real_storage_value.flags.c_contiguous)
        self.assertFalse(expected_source.flags.c_contiguous)
        self.assertTrue(contiguous_anti_source.flags.c_contiguous)
        expected_determinant = np.asarray(np.linalg.det(expected_source))
        expected_sign, expected_logabsdet = np.linalg.slogdet(expected_source)
        expected_sign = np.asarray(expected_sign)
        expected_logabsdet = np.asarray(expected_logabsdet)
        anti_determinant = np.asarray(np.linalg.det(contiguous_anti_source))
        anti_sign, anti_logabsdet = np.linalg.slogdet(contiguous_anti_source)
        self.assertFalse(np.allclose(expected_determinant, anti_determinant))
        self.assertFalse(np.allclose(expected_sign, anti_sign))
        self.assertFalse(np.allclose(expected_logabsdet, anti_logabsdet))
        with (
            CnumpyRuntime(DLL) as runtime,
            ExitStack() as result_stack,
            ExitStack() as source_stack,
        ):
            baseline = runtime.retained_bytes
            storage = source_stack.enter_context(
                runtime.from_numpy(real_storage_value)
            )
            source = source_stack.enter_context(
                runtime.transpose(storage, (2, 0, 1))
            )
            self.assertFalse(source.c_contiguous)
            determinant = result_stack.enter_context(
                self.det_result(runtime, source)
            )
            sign, logabsdet = self.slogdet_result(runtime, source)
            sign = result_stack.enter_context(sign)
            logabsdet = result_stack.enter_context(logabsdet)
            source_stack.close()
            for result, expected in (
                (determinant, expected_determinant),
                (sign, expected_sign),
                (logabsdet, expected_logabsdet),
            ):
                self.assertEqual(expected.dtype, result.numpy_dtype)
                self.assertEqual(expected.shape, result.shape)
                np.testing.assert_allclose(
                    result.to_numpy(), expected, rtol=2e-12, atol=2e-12
                )
            result_stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

        scale_cases = (
            np.diag([1e-300, -2e-300]),
            np.diag([1e300, -2e300]),
            np.empty((2, 0, 0), dtype=np.float32),
            np.empty((0, 2, 2), dtype=np.complex64),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            for source_value in scale_cases:
                with self.subTest(shape=source_value.shape, scale=True):
                    source = stack.enter_context(
                        runtime.from_numpy(source_value)
                    )
                    determinant = stack.enter_context(
                        self.det_result(runtime, source)
                    )
                    sign, logabsdet = self.slogdet_result(runtime, source)
                    sign = stack.enter_context(sign)
                    logabsdet = stack.enter_context(logabsdet)
                    with np.errstate(all="ignore"):
                        expected_determinant = np.linalg.det(source_value)
                        expected_sign, expected_logabsdet = (
                            np.linalg.slogdet(source_value)
                        )
                    np.testing.assert_allclose(
                        determinant.to_numpy(), expected_determinant,
                        rtol=2e-6, atol=0.0, equal_nan=True
                    )
                    np.testing.assert_allclose(
                        sign.to_numpy(), expected_sign,
                        rtol=2e-6, atol=0.0, equal_nan=True
                    )
                    np.testing.assert_allclose(
                        logabsdet.to_numpy(), expected_logabsdet,
                        rtol=2e-6, atol=2e-6, equal_nan=True
                    )
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_slogdet_v2_validation_clears_outputs_without_retention(
        self,
    ) -> None:
        invalid_values = (
            (np.array([1.0, 2.0]), -4, "at least"),
            (np.ones((2, 3)), -4, "square"),
            (np.eye(2, dtype=np.float16), -3, "dtype"),
        )
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            function = self.required_export(
                runtime, "cnp_linalg_slogdet_v2"
            )
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
            ]
            function.restype = ctypes.c_int
            for source_value, expected_status, message in invalid_values:
                source = stack.enter_context(runtime.from_numpy(source_value))
                before = runtime.retained_bytes
                sign_slot = ctypes.c_void_p(1)
                logabsdet_slot = ctypes.c_void_p(2)
                runtime.dll.cnp_clear_error()
                status = int(
                    function(
                        source.pointer,
                        ctypes.byref(sign_slot),
                        ctypes.byref(logabsdet_slot),
                    )
                )
                self.assertEqual(expected_status, status)
                self.assertFalse(sign_slot.value)
                self.assertFalse(logabsdet_slot.value)
                error = runtime.error_state()
                self.assertEqual("cnp_linalg_slogdet_v2", error.function)
                self.assertIn(message, error.message.lower())
                self.assertEqual(before, runtime.retained_bytes)

            logabsdet_slot = ctypes.c_void_p(3)
            runtime.dll.cnp_clear_error()
            status = int(function(None, None, ctypes.byref(logabsdet_slot)))
            self.assertEqual(-1, status)
            self.assertFalse(logabsdet_slot.value)

            alias_slot = ctypes.c_void_p(4)
            runtime.dll.cnp_clear_error()
            status = int(
                function(
                    None,
                    ctypes.byref(alias_slot),
                    ctypes.byref(alias_slot),
                )
            )
            self.assertEqual(-1, status)
            self.assertFalse(alias_slot.value)
            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_cholesky_dtype_batch_and_empty_shapes_match_numpy_125(
        self,
    ) -> None:
        dtype_cases = (
            np.bool_,
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for dtype in dtype_cases:
                if dtype is np.bool_:
                    values = np.broadcast_to(
                        np.eye(2, dtype=np.bool_), (2, 2, 2)
                    ).copy()
                else:
                    values = np.asarray(
                        [[[4, 900], [1, 3]], [[9, 700], [3, 5]]],
                        dtype=dtype,
                    )
                if np.issubdtype(np.dtype(dtype), np.complexfloating):
                    values[0, 0, 0] += dtype(7j)
                    values[0, 1, 0] += dtype(2j)
                    values[1, 0, 0] += dtype(-11j)
                    values[1, 1, 0] += dtype(-1j)
                expected = np.linalg.cholesky(values)
                with self.subTest(dtype=np.dtype(dtype)):
                    with runtime.from_numpy(values) as source:
                        source_before = source.to_numpy().copy()
                        with self.cholesky_result(runtime, source) as result:
                            self.assertEqual(expected.dtype, result.numpy_dtype)
                            self.assertEqual(expected.shape, result.shape)
                            self.assertTrue(result.c_contiguous)
                            self.assertTrue(result.owns_data)
                            np.testing.assert_allclose(
                                result.to_numpy(), expected,
                                rtol=3e-6, atol=3e-6, equal_nan=True,
                            )
                        np.testing.assert_array_equal(
                            source.to_numpy(), source_before
                        )

            runtime.dll.cnp_array_zeros.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_int,
                ctypes.c_int,
            ]
            runtime.dll.cnp_array_zeros.restype = ctypes.c_void_p
            runtime.dll.cnp_array_set_int.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_int64,
            ]
            runtime.dll.cnp_array_set_int.restype = ctypes.c_int
            shape = (ctypes.c_int64 * 2)(2, 2)
            for dtype_number in (8, 9):
                with self.subTest(cnp_integer_dtype=dtype_number):
                    source = runtime._owned_result(
                        runtime.dll.cnp_array_zeros(
                            2, shape, dtype_number, 0
                        ),
                        "cnp_array_zeros:integer",
                    )
                    with source:
                        for diagonal in range(2):
                            indices = (ctypes.c_int64 * 2)(
                                diagonal, diagonal
                            )
                            self.assertEqual(
                                0,
                                runtime.dll.cnp_array_set_int(
                                    source.pointer, indices, 1
                                ),
                            )
                        with self.cholesky_result(runtime, source) as result:
                            self.assertEqual(np.dtype(np.float64), result.numpy_dtype)
                            np.testing.assert_array_equal(
                                np.eye(2, dtype=np.float64),
                                result.to_numpy(),
                            )

            for empty in (
                np.empty((0, 0), dtype=np.float64),
                np.empty((2, 0, 0), dtype=np.float32),
                np.empty((0, 2, 2), dtype=np.complex64),
            ):
                with self.subTest(shape=empty.shape):
                    expected = np.linalg.cholesky(empty)
                    with runtime.from_numpy(empty) as source:
                        with self.cholesky_result(runtime, source) as result:
                            self.assertEqual(expected.shape, result.shape)
                            self.assertEqual(expected.dtype, result.numpy_dtype)
                            self.assertTrue(result.c_contiguous)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_cholesky_uses_only_logical_lower_hermitian_triangle_and_views(
        self,
    ) -> None:
        logical = np.asarray(
            [[4 + 19j, np.nan + 70j, -800 + 90j],
             [1 + 2j, 6 - 13j, 500 - 60j],
             [0.5 - 1j, 1 + 0.25j, 8 + 17j]],
            dtype=np.complex128,
        )
        expected = np.linalg.cholesky(logical)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            slice_function = runtime.dll.cnp_array_slice
            slice_function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(_CnpSlice),
            ]
            slice_function.restype = ctypes.c_void_p

            with runtime.from_numpy(logical) as contiguous:
                with self.cholesky_result(runtime, contiguous) as result:
                    np.testing.assert_allclose(
                        result.to_numpy(), expected, rtol=2e-12, atol=2e-12
                    )
                    np.testing.assert_array_equal(
                        np.triu(result.to_numpy(), 1), 0
                    )
                    np.testing.assert_array_equal(
                        np.diag(result.to_numpy()).imag, 0
                    )

            with runtime.from_numpy(logical.T) as transpose_storage:
                with runtime.transpose(transpose_storage) as fortran_view:
                    self.assertTrue(fortran_view.f_contiguous)
                    with self.cholesky_result(runtime, fortran_view) as result:
                        np.testing.assert_allclose(
                            result.to_numpy(), expected,
                            rtol=2e-12, atol=2e-12,
                        )

            with runtime.from_numpy(logical[::-1, ::-1]) as reverse_storage:
                slices = (_CnpSlice * 2)(
                    _CnpSlice(0, 0, -1, False, False, True),
                    _CnpSlice(0, 0, -1, False, False, True),
                )
                reversed_view = runtime._owned_result(
                    slice_function(reverse_storage.pointer, 2, slices),
                    "cnp_array_slice",
                )
                try:
                    self.assertTrue(any(stride < 0 for stride in reversed_view.strides))
                    result = self.cholesky_result(runtime, reversed_view)
                    reversed_view.close()
                    with result:
                        np.testing.assert_allclose(
                            result.to_numpy(), expected,
                            rtol=2e-12, atol=2e-12,
                        )
                finally:
                    if not reversed_view._closed:
                        reversed_view.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_cholesky_nonfinite_and_finite_extremes_match_numpy_125(
        self,
    ) -> None:
        scale_cases = []
        for dtype in (np.float32, np.float64):
            finfo = np.finfo(dtype)
            scale_cases.extend(
                np.eye(2, dtype=dtype) * dtype(scale)
                for scale in (
                    np.nextafter(dtype(0), dtype(1)),
                    finfo.tiny,
                    finfo.max,
                )
            )
        for dtype in (np.complex64, np.complex128):
            real_dtype = np.float32 if dtype is np.complex64 else np.float64
            finfo = np.finfo(real_dtype)
            scale_cases.extend(
                np.eye(2, dtype=dtype) * real_dtype(scale)
                for scale in (
                    np.nextafter(real_dtype(0), real_dtype(1)),
                    finfo.tiny,
                    finfo.max,
                )
            )
        propagation_cases = (
            np.asarray([[np.inf, np.nan], [0.0, 1.0]]),
            np.asarray([[4.0, 0.0], [np.nan, 3.0]]),
            np.asarray(
                [[4 + 9j, np.nan + 2j], [np.nan + 1j, 3 - 8j]],
                dtype=np.complex128,
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for values in (*scale_cases, *propagation_cases):
                with self.subTest(dtype=values.dtype, values=repr(values)):
                    with np.errstate(all="ignore"):
                        expected = np.linalg.cholesky(values)
                    with runtime.from_numpy(values) as source:
                        with self.cholesky_result(runtime, source) as result:
                            self.assert_special_value_pattern(
                                result.to_numpy(), expected,
                                rtol=(3e-6 if values.dtype.itemsize <= 8 else 2e-12),
                            )
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_cholesky_contiguous_float64_large_matrix_semantics(
        self,
    ) -> None:
        def deterministic_matrix(n: int) -> np.ndarray:
            indices = np.arange(n * n, dtype=np.int64)
            values = (
                ((indices * 37 + 11) % 1009) / 1009.0 - 0.5
            ).reshape(n, n)
            return values @ values.T + n * np.eye(n, dtype=np.float64)

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for n in (16, 17, 32, 64, 128):
                values = deterministic_matrix(n)
                upper = np.triu_indices(n, 1)
                values[upper] = np.where(
                    (upper[0] + upper[1]) % 2 == 0,
                    np.nan,
                    np.inf,
                )
                expected = np.linalg.cholesky(values)
                with self.subTest(kind="well_conditioned", n=n):
                    with runtime.from_numpy(values) as source:
                        before = runtime.retained_bytes
                        with self.cholesky_result(runtime, source) as result:
                            actual = result.to_numpy()
                            self.assertEqual(np.dtype(np.float64), result.numpy_dtype)
                            self.assertEqual((n, n), result.shape)
                            self.assertTrue(result.c_contiguous)
                            np.testing.assert_array_equal(
                                np.triu(actual, 1), 0.0
                            )
                            np.testing.assert_allclose(
                                actual, expected,
                                rtol=2e-12, atol=2e-12,
                            )
                        self.assertEqual(before, runtime.retained_bytes)

            n = 32
            indices = np.arange(n * n, dtype=np.int64)
            basis = (
                ((indices * 37 + 11) % 1009) / 1009.0 - 0.5
            ).reshape(n, n)
            orthogonal, _ = np.linalg.qr(basis)
            eigenvalues = np.geomspace(1.0, 1e-12, n)
            ill_conditioned = (orthogonal * eigenvalues) @ orthogonal.T
            ill_conditioned = (
                ill_conditioned + ill_conditioned.T
            ) * 0.5
            expected = np.linalg.cholesky(ill_conditioned)
            with runtime.from_numpy(ill_conditioned) as source:
                before = runtime.retained_bytes
                with self.cholesky_result(runtime, source) as result:
                    actual = result.to_numpy()
                    self.assertEqual(np.dtype(np.float64), result.numpy_dtype)
                    self.assertEqual((n, n), result.shape)
                    np.testing.assert_array_equal(np.triu(actual, 1), 0.0)
                    np.testing.assert_allclose(
                        actual, expected, rtol=3e-5, atol=1e-11
                    )
                    np.testing.assert_allclose(
                        actual @ actual.T,
                        ill_conditioned,
                        rtol=2e-12,
                        atol=2e-12,
                    )
                self.assertEqual(before, runtime.retained_bytes)

            nonfinite_cases = []
            lower_nan = np.eye(32, dtype=np.float64)
            lower_nan[20, 16] = np.nan
            nonfinite_cases.append(("lower_nan", lower_nan))
            lower_positive_inf = np.eye(32, dtype=np.float64)
            lower_positive_inf[17, 17] = np.inf
            nonfinite_cases.append(("lower_positive_inf", lower_positive_inf))
            for name, values in nonfinite_cases:
                with np.errstate(all="ignore"):
                    expected = np.linalg.cholesky(values)
                with self.subTest(kind=name):
                    with runtime.from_numpy(values) as source:
                        before = runtime.retained_bytes
                        with self.cholesky_result(runtime, source) as result:
                            actual = result.to_numpy()
                            self.assertEqual(np.dtype(np.float64), result.numpy_dtype)
                            self.assertEqual((32, 32), result.shape)
                            np.testing.assert_array_equal(
                                np.triu(actual, 1), 0.0
                            )
                            self.assert_special_value_pattern(
                                actual, expected, rtol=2e-12
                            )
                        self.assertEqual(before, runtime.retained_bytes)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_cholesky_contiguous_float64_near_semidefinite_classification(
        self,
    ) -> None:
        success = np.eye(32, dtype=np.float64)
        success[17, 16] = 1.0
        success[16, 17] = np.nan
        success[17, 17] = np.nextafter(1.0, 2.0)
        expected = np.linalg.cholesky(success)

        failure = success.copy()
        failure[17, 17] = 1.0
        with self.assertRaises(np.linalg.LinAlgError):
            np.linalg.cholesky(failure)

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(success) as source:
                before = runtime.retained_bytes
                with self.cholesky_result(runtime, source) as result:
                    actual = result.to_numpy()
                    self.assertEqual(np.dtype(np.float64), result.numpy_dtype)
                    self.assertEqual((32, 32), result.shape)
                    np.testing.assert_array_equal(np.triu(actual, 1), 0.0)
                    np.testing.assert_allclose(
                        actual, expected, rtol=2e-12, atol=0.0
                    )
                self.assertEqual(before, runtime.retained_bytes)

            function = self.required_export(runtime, "cnp_linalg_cholesky")
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            function.restype = ctypes.c_int
            with runtime.from_numpy(failure) as source:
                before = runtime.retained_bytes
                slot = ctypes.c_void_p(1)
                runtime.dll.cnp_clear_error()
                status = int(function(source.pointer, ctypes.byref(slot)))
                self.assertEqual(-9, status)
                self.assertFalse(slot.value)
                error = runtime.error_state()
                self.assertEqual("cnp_linalg_cholesky", error.function)
                self.assertIn("positive definite", error.message.lower())
                self.assertEqual(before, runtime.retained_bytes)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_cholesky_contiguous_float64_multiterm_ulp_pivot_matches_view(
        self,
    ) -> None:
        rng = np.random.default_rng(20260803)
        lower_row = rng.normal(size=16)
        lower_row *= 10 ** rng.uniform(-8, 8)
        squared_norm = np.dot(lower_row, lower_row)

        values = np.eye(17, dtype=np.float64)
        values[-1, :-1] = lower_row
        values[-1, -1] = np.nextafter(squared_norm, np.inf)
        values[np.triu_indices(17, 1)] = np.nan
        expected = np.linalg.cholesky(values)
        self.assertEqual(2.0 ** -26, expected[-1, -1])

        failure = values.copy()
        failure[-1, -1] = np.nextafter(squared_norm, -np.inf)
        with self.assertRaises(np.linalg.LinAlgError):
            np.linalg.cholesky(failure)

        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            with runtime.from_numpy(values.T) as transpose_storage:
                with runtime.transpose(transpose_storage) as fortran_view:
                    before = runtime.retained_bytes
                    with self.cholesky_result(runtime, fortran_view) as result:
                        actual = result.to_numpy()
                        self.assertEqual(np.dtype(np.float64), result.numpy_dtype)
                        self.assertEqual((17, 17), result.shape)
                        self.assertEqual(expected[-1, -1], actual[-1, -1])
                        np.testing.assert_allclose(
                            actual, expected, rtol=2e-12, atol=0.0
                        )
                    self.assertEqual(before, runtime.retained_bytes)

            with runtime.from_numpy(values) as source:
                before = runtime.retained_bytes
                with self.cholesky_result(runtime, source) as result:
                    actual = result.to_numpy()
                    self.assertEqual(np.dtype(np.float64), result.numpy_dtype)
                    self.assertEqual((17, 17), result.shape)
                    self.assertEqual(expected[-1, -1], actual[-1, -1])
                    np.testing.assert_allclose(
                        actual, expected, rtol=2e-12, atol=0.0
                    )
                self.assertEqual(before, runtime.retained_bytes)

            function = self.required_export(runtime, "cnp_linalg_cholesky")
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            function.restype = ctypes.c_int
            with runtime.from_numpy(failure) as source:
                before = runtime.retained_bytes
                slot = ctypes.c_void_p(1)
                runtime.dll.cnp_clear_error()
                status = int(function(source.pointer, ctypes.byref(slot)))
                self.assertEqual(-9, status)
                self.assertFalse(slot.value)
                error = runtime.error_state()
                self.assertEqual("cnp_linalg_cholesky", error.function)
                self.assertIn("positive definite", error.message.lower())
                self.assertEqual(before, runtime.retained_bytes)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_cholesky_failures_are_precise_atomic_and_nonretaining(
        self,
    ) -> None:
        invalid_values = (
            (np.asarray(1.0), -4, "at least"),
            (np.asarray([1.0, 2.0]), -4, "at least"),
            (np.ones((2, 3)), -4, "square"),
            (np.eye(2, dtype=np.float16), -3, "dtype"),
            (np.asarray([[1.0, 0.0], [0.0, 0.0]]), -9, "positive definite"),
            (np.asarray([[1.0, 0.0], [0.0, -0.0]]), -9, "positive definite"),
            (np.asarray([[1.0, 0.0], [0.0, -np.inf]]), -9, "positive definite"),
            (
                np.asarray(
                    [[[1.0, 0.0], [0.0, 1.0]],
                     [[1.0, 0.0], [0.0, -1.0]]]
                ),
                -9,
                "positive definite",
            ),
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            function = self.required_export(runtime, "cnp_linalg_cholesky")
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            function.restype = ctypes.c_int
            for values, expected_status, expected_message in invalid_values:
                with self.subTest(shape=values.shape, dtype=values.dtype):
                    with runtime.from_numpy(values) as source:
                        before = runtime.retained_bytes
                        slot = ctypes.c_void_p(1)
                        runtime.dll.cnp_clear_error()
                        status = int(function(source.pointer, ctypes.byref(slot)))
                        self.assertEqual(expected_status, status)
                        self.assertFalse(slot.value)
                        error = runtime.error_state()
                        self.assertEqual("cnp_linalg_cholesky", error.function)
                        self.assertIn(expected_message, error.message.lower())
                        self.assertEqual(before, runtime.retained_bytes)

            string_shape = (ctypes.c_int64 * 2)(2, 2)
            runtime.dll.cnp_array_zeros.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_int,
                ctypes.c_int,
            ]
            runtime.dll.cnp_array_zeros.restype = ctypes.c_void_p
            string_source = runtime._owned_result(
                runtime.dll.cnp_array_zeros(2, string_shape, 19, 0),
                "cnp_array_zeros:string",
            )
            with string_source:
                slot = ctypes.c_void_p(2)
                status = int(function(string_source.pointer, ctypes.byref(slot)))
                self.assertEqual(-3, status)
                self.assertFalse(slot.value)
                self.assertEqual("cnp_linalg_cholesky", runtime.error_state().function)

            slot = ctypes.c_void_p(3)
            runtime.dll.cnp_clear_error()
            self.assertEqual(-1, int(function(None, ctypes.byref(slot))))
            self.assertFalse(slot.value)
            self.assertEqual("cnp_linalg_cholesky", runtime.error_state().function)
            runtime.dll.cnp_clear_error()
            self.assertEqual(-1, int(function(None, None)))
            self.assertEqual("cnp_linalg_cholesky", runtime.error_state().function)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_solve_singular_failure_is_explicit_atomic_and_nonretaining(
        self,
    ) -> None:
        singular_value = np.array(
            [[1.0, 2.0], [2.0, 4.0]], dtype=np.float64
        )
        rhs_value = np.array([1.0, 2.0], dtype=np.float64)
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            singular = stack.enter_context(
                runtime.from_numpy(singular_value)
            )
            rhs = stack.enter_context(runtime.from_numpy(rhs_value))
            function = self.required_export(runtime, "cnp_linalg_solve")
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            function.restype = ctypes.c_int

            result_slot = ctypes.c_void_p(1)
            before = runtime.retained_bytes
            runtime.dll.cnp_clear_error()
            status = int(
                function(
                    singular.pointer,
                    rhs.pointer,
                    ctypes.byref(result_slot),
                )
            )

            self.assertEqual(-9, status)
            self.assertFalse(result_slot.value)
            error = runtime.error_state()
            self.assertEqual("cnp_linalg_solve", error.function)
            self.assertIn("singular", error.message.lower())
            self.assertEqual(before, runtime.retained_bytes)

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
