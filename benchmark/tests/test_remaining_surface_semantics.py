from __future__ import annotations

import ctypes
from contextlib import ExitStack
from pathlib import Path
import unittest

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"


def _function(runtime: CnumpyRuntime, name: str, argtypes, restype):
    function = getattr(runtime.dll, name)
    function.argtypes = argtypes
    function.restype = restype
    return function


def _owned(runtime: CnumpyRuntime, pointer, origin: str):
    return runtime._owned_result(pointer, origin)


def _assert_array_equal(
    case: unittest.TestCase,
    actual,
    expected: np.ndarray,
) -> None:
    case.assertEqual(expected.shape, actual.shape)
    case.assertEqual(expected.dtype, actual.numpy_dtype)
    np.testing.assert_array_equal(expected, actual.to_numpy(), strict=True)


class ArraySurfaceSemanticsTests(unittest.TestCase):
    """NumPy 1.25 owners for the public array.c compatibility surface."""

    SYMBOLS = frozenset(
        {
            "cnp_arange",
            "cnp_array_copy",
            "cnp_array_empty",
            "cnp_array_flat_get",
            "cnp_array_flat_set",
            "cnp_array_free",
            "cnp_array_from_double_array",
            "cnp_array_from_float_array",
            "cnp_array_from_int_array",
            "cnp_array_from_scalar",
            "cnp_array_full",
            "cnp_array_get_double",
            "cnp_array_get_int",
            "cnp_array_incref",
            "cnp_array_new",
            "cnp_array_ones",
            "cnp_array_set_double",
            "cnp_array_set_int",
            "cnp_array_view",
            "cnp_array_zeros",
            "cnp_astype",
            "cnp_copy",
            "cnp_diag",
            "cnp_empty_like",
            "cnp_eye",
            "cnp_full_like",
            "cnp_geomspace",
            "cnp_identity",
            "cnp_linspace",
            "cnp_logspace",
            "cnp_ones_like",
            "cnp_tri",
            "cnp_zeros_like",
        }
    )

    _NUMERIC_DTYPES = (
        (1, np.dtype(np.bool_)),
        (2, np.dtype(np.int8)),
        (3, np.dtype(np.uint8)),
        (4, np.dtype(np.int16)),
        (5, np.dtype(np.uint16)),
        (6, np.dtype(np.int32)),
        (7, np.dtype(np.uint32)),
        (8, np.dtype(np.int64)),
        (9, np.dtype(np.uint64)),
        (12, np.dtype(np.float32)),
        (13, np.dtype(np.float64)),
        (15, np.dtype(np.complex64)),
        (16, np.dtype(np.complex128)),
        (24, np.dtype(np.float16)),
    )

    def test_creation_exports_match_numpy_values_dtypes_layouts_and_lifetimes(
        self,
    ) -> None:
        constructors = {
            "cnp_array_new": None,
            "cnp_array_empty": None,
            "cnp_array_zeros": 0.0,
            "cnp_array_ones": 1.0,
            "cnp_array_full": -3.5,
        }
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            for name, fill_value in constructors.items():
                if name == "cnp_array_full":
                    function = _function(
                        runtime,
                        name,
                        [
                            ctypes.c_int,
                            ctypes.POINTER(ctypes.c_int64),
                            ctypes.c_double,
                            ctypes.c_int,
                            ctypes.c_int,
                        ],
                        ctypes.c_void_p,
                    )
                else:
                    function = _function(
                        runtime,
                        name,
                        [
                            ctypes.c_int,
                            ctypes.POINTER(ctypes.c_int64),
                            ctypes.c_int,
                            ctypes.c_int,
                        ],
                        ctypes.c_void_p,
                    )
                for dtype_number, dtype in self._NUMERIC_DTYPES:
                    for shape, order_code, order in (
                        ((2, 3), 0, "C"),
                        ((2, 3), 1, "F"),
                        ((2, 0, 3), 0, "C"),
                        ((), 0, "C"),
                    ):
                        with self.subTest(
                            operation=name,
                            dtype=dtype.name,
                            shape=shape,
                            order=order,
                        ):
                            shape_buffer = (
                                (ctypes.c_int64 * len(shape))(*shape)
                                if shape
                                else None
                            )
                            runtime.dll.cnp_clear_error()
                            arguments = [
                                len(shape),
                                shape_buffer,
                            ]
                            if name == "cnp_array_full":
                                arguments.append(fill_value)
                            arguments.extend((dtype_number, order_code))
                            result = _owned(
                                runtime, function(*arguments), name
                            )
                            try:
                                self.assertEqual(shape, result.shape)
                                self.assertEqual(dtype, result.numpy_dtype)
                                if shape and all(shape):
                                    if order == "F":
                                        self.assertTrue(result.f_contiguous)
                                    else:
                                        self.assertTrue(result.c_contiguous)
                                if fill_value is not None:
                                    expected = np.full(
                                        shape, fill_value,
                                        dtype=dtype, order=order,
                                    )
                                    _assert_array_equal(self, result, expected)
                            finally:
                                result.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_typed_constructors_access_mutation_and_astype_match_numpy_125(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            constructors = (
                (
                    "cnp_array_from_int_array",
                    ctypes.c_int64,
                    np.asarray([-(2**62) + 3, 0, 2**62 + 5], dtype=np.int64),
                ),
                (
                    "cnp_array_from_float_array",
                    ctypes.c_float,
                    np.asarray([-1.25, 0.0, 3.5], dtype=np.float32),
                ),
                (
                    "cnp_array_from_double_array",
                    ctypes.c_double,
                    np.asarray([-1.25, 0.0, 3.5], dtype=np.float64),
                ),
            )
            native = {}
            for name, scalar_type, expected in constructors:
                function = _function(
                    runtime,
                    name,
                    [ctypes.POINTER(scalar_type), ctypes.c_int64],
                    ctypes.c_void_p,
                )
                values = (scalar_type * expected.size)(*expected.tolist())
                result = stack.enter_context(
                    _owned(runtime, function(values, expected.size), name)
                )
                native[name] = result
                _assert_array_equal(self, result, expected)

            from_scalar = _function(
                runtime,
                "cnp_array_from_scalar",
                [ctypes.c_double, ctypes.c_int],
                ctypes.c_void_p,
            )
            scalar = stack.enter_context(
                _owned(runtime, from_scalar(-2.5, 13), "cnp_array_from_scalar")
            )
            _assert_array_equal(
                self, scalar, np.asarray(-2.5, dtype=np.float64)
            )

            get_double = _function(
                runtime,
                "cnp_array_get_double",
                [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64)],
                ctypes.c_double,
            )
            get_int = _function(
                runtime,
                "cnp_array_get_int",
                [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64)],
                ctypes.c_int64,
            )
            set_double = _function(
                runtime,
                "cnp_array_set_double",
                [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.c_double,
                ],
                ctypes.c_int,
            )
            set_int = _function(
                runtime,
                "cnp_array_set_int",
                [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.c_int64,
                ],
                ctypes.c_int,
            )
            flat_get = _function(
                runtime,
                "cnp_array_flat_get",
                [ctypes.c_void_p, ctypes.c_int64],
                ctypes.c_double,
            )
            flat_set = _function(
                runtime,
                "cnp_array_flat_set",
                [ctypes.c_void_p, ctypes.c_int64, ctypes.c_double],
                ctypes.c_int,
            )
            float_values = native["cnp_array_from_double_array"]
            integer_values = native["cnp_array_from_int_array"]
            index = (ctypes.c_int64 * 1)(2)
            self.assertEqual(3.5, get_double(float_values.pointer, index))
            self.assertEqual(2**62 + 5, get_int(integer_values.pointer, index))
            self.assertEqual(0, set_double(float_values.pointer, index, 9.25))
            self.assertEqual(0, set_int(integer_values.pointer, index, 2**61 + 7))
            self.assertEqual(0, flat_set(float_values.pointer, 0, -7.5))
            self.assertEqual(-7.5, flat_get(float_values.pointer, 0))
            np.testing.assert_array_equal(
                [-7.5, 0.0, 9.25], float_values.to_numpy()
            )
            np.testing.assert_array_equal(
                [-(2**62) + 3, 0, 2**61 + 7], integer_values.to_numpy()
            )

            astype = _function(
                runtime,
                "cnp_astype",
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
                ctypes.c_void_p,
            )
            converted = stack.enter_context(
                _owned(
                    runtime,
                    astype(integer_values.pointer, 9, 4),
                    "cnp_astype",
                )
            )
            _assert_array_equal(
                self,
                converted,
                integer_values.to_numpy().astype(np.uint64, casting="unsafe"),
            )

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_copy_view_refcount_and_like_exports_own_exact_results(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            source = runtime.from_numpy(
                np.arange(12, dtype=np.float64).reshape(3, 4).T
            )
            transposed = runtime.transpose(source)
            expected = transposed.to_numpy().copy()
            copy = _function(
                runtime,
                "cnp_array_copy",
                [ctypes.c_void_p],
                ctypes.c_void_p,
            )
            copy_alias = _function(
                runtime,
                "cnp_copy",
                [ctypes.c_void_p],
                ctypes.c_void_p,
            )
            view = _function(
                runtime,
                "cnp_array_view",
                [ctypes.c_void_p],
                ctypes.c_void_p,
            )
            copied = stack.enter_context(
                _owned(runtime, copy(transposed.pointer), "cnp_array_copy")
            )
            copied_alias = stack.enter_context(
                _owned(runtime, copy_alias(transposed.pointer), "cnp_copy")
            )
            viewed = stack.enter_context(
                _owned(runtime, view(transposed.pointer), "cnp_array_view")
            )
            _assert_array_equal(self, copied, expected)
            _assert_array_equal(self, copied_alias, expected)
            transposed.close()
            source.close()
            _assert_array_equal(self, viewed, expected)

            for name, fill in (
                ("cnp_zeros_like", 0.0),
                ("cnp_ones_like", 1.0),
                ("cnp_full_like", -4.25),
                ("cnp_empty_like", None),
            ):
                if name == "cnp_full_like":
                    function = _function(
                        runtime,
                        name,
                        [ctypes.c_void_p, ctypes.c_double],
                        ctypes.c_void_p,
                    )
                    pointer = function(viewed.pointer, fill)
                else:
                    function = _function(
                        runtime, name, [ctypes.c_void_p], ctypes.c_void_p
                    )
                    pointer = function(viewed.pointer)
                result = stack.enter_context(_owned(runtime, pointer, name))
                self.assertEqual(expected.shape, result.shape)
                self.assertEqual(expected.dtype, result.numpy_dtype)
                if fill is not None:
                    _assert_array_equal(
                        self,
                        result,
                        np.full(expected.shape, fill, dtype=expected.dtype),
                    )

            array_new = _function(
                runtime,
                "cnp_array_new",
                [
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.c_int,
                    ctypes.c_int,
                ],
                ctypes.c_void_p,
            )
            incref = _function(
                runtime, "cnp_array_incref", [ctypes.c_void_p], None
            )
            decref = _function(
                runtime, "cnp_array_decref", [ctypes.c_void_p], None
            )
            free = _function(
                runtime, "cnp_array_free", [ctypes.c_void_p], None
            )
            shape = (ctypes.c_int64 * 1)(4)
            raw = array_new(1, shape, 13, 0)
            self.assertTrue(raw)
            incref(raw)
            decref(raw)
            self.assertEqual(4, runtime.dll.cnp_array_size(raw))
            decref(raw)
            raw = array_new(1, shape, 13, 0)
            self.assertTrue(raw)
            free(raw)

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_range_matrix_and_diag_exports_match_numpy_125(self) -> None:
        with CnumpyRuntime(DLL) as runtime, ExitStack() as stack:
            baseline = runtime.retained_bytes
            array_calls = (
                (
                    "cnp_arange",
                    [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int],
                    (5.0, -3.0, -1.25, 12),
                    np.arange(5.0, -3.0, -1.25, dtype=np.float32),
                ),
                (
                    "cnp_linspace",
                    [ctypes.c_double, ctypes.c_double, ctypes.c_int64, ctypes.c_bool, ctypes.c_int],
                    (-2.0, 3.0, 9, True, 24),
                    np.linspace(-2.0, 3.0, 9, endpoint=True, dtype=np.float16),
                ),
                (
                    "cnp_logspace",
                    [ctypes.c_double, ctypes.c_double, ctypes.c_int64, ctypes.c_bool, ctypes.c_double, ctypes.c_int],
                    (-2.0, 2.0, 7, False, 2.0, 13),
                    np.logspace(-2.0, 2.0, 7, endpoint=False, base=2.0, dtype=np.float64),
                ),
                (
                    "cnp_geomspace",
                    [ctypes.c_double, ctypes.c_double, ctypes.c_int64, ctypes.c_bool, ctypes.c_int],
                    (1.0, 256.0, 9, True, 13),
                    np.geomspace(1.0, 256.0, 9, endpoint=True, dtype=np.float64),
                ),
                (
                    "cnp_geomspace",
                    [ctypes.c_double, ctypes.c_double, ctypes.c_int64, ctypes.c_bool, ctypes.c_int],
                    (-1.0, -256.0, 9, True, 13),
                    np.geomspace(-1.0, -256.0, 9, endpoint=True, dtype=np.float64),
                ),
                (
                    "cnp_eye",
                    [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int],
                    (3, 5, -1, 6),
                    np.eye(3, 5, k=-1, dtype=np.int32),
                ),
                (
                    "cnp_identity",
                    [ctypes.c_int64, ctypes.c_int],
                    (4, 24),
                    np.identity(4, dtype=np.float16),
                ),
                (
                    "cnp_tri",
                    [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int],
                    (4, 6, 1, 3),
                    np.tri(4, 6, k=1, dtype=np.uint8),
                ),
            )
            for name, argtypes, arguments, expected in array_calls:
                function = _function(
                    runtime, name, argtypes, ctypes.c_void_p
                )
                result = stack.enter_context(
                    _owned(runtime, function(*arguments), name)
                )
                if expected.dtype.kind == "f":
                    self.assertEqual(expected.shape, result.shape)
                    self.assertEqual(expected.dtype, result.numpy_dtype)
                    np.testing.assert_allclose(
                        expected, result.to_numpy(), rtol=1e-6, atol=1e-7
                    )
                else:
                    _assert_array_equal(self, result, expected)

            diag = _function(
                runtime,
                "cnp_diag",
                [ctypes.c_void_p, ctypes.c_int64],
                ctypes.c_void_p,
            )
            vector_value = np.asarray([1, 2, 3], dtype=np.int64)
            vector = stack.enter_context(runtime.from_numpy(vector_value))
            matrix = stack.enter_context(
                _owned(runtime, diag(vector.pointer, -2), "cnp_diag")
            )
            _assert_array_equal(self, matrix, np.diag(vector_value, k=-2))
            extracted = stack.enter_context(
                _owned(runtime, diag(matrix.pointer, -2), "cnp_diag")
            )
            _assert_array_equal(self, extracted, np.diag(np.diag(vector_value, k=-2), k=-2))

            stack.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_invalid_array_requests_are_explicit_and_allocation_atomic(
        self,
    ) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            array_new = _function(
                runtime,
                "cnp_array_new",
                [ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_int, ctypes.c_int],
                ctypes.c_void_p,
            )
            for ndim, shape, dtype in ((-1, None, 13), (1, (ctypes.c_int64 * 1)(1), 0)):
                runtime.dll.cnp_clear_error()
                self.assertFalse(array_new(ndim, shape, dtype, 0))
                error = runtime.error_state()
                self.assertNotEqual(0, error.status)
                self.assertEqual("cnp_array_new", error.function)
                self.assertEqual(baseline, runtime.retained_bytes)

            null_array_functions = {
                "cnp_array_copy": _function(runtime, "cnp_array_copy", [ctypes.c_void_p], ctypes.c_void_p),
                "cnp_array_view": _function(runtime, "cnp_array_view", [ctypes.c_void_p], ctypes.c_void_p),
                "cnp_diag": _function(runtime, "cnp_diag", [ctypes.c_void_p, ctypes.c_int64], ctypes.c_void_p),
                "cnp_zeros_like": _function(runtime, "cnp_zeros_like", [ctypes.c_void_p], ctypes.c_void_p),
                "cnp_ones_like": _function(runtime, "cnp_ones_like", [ctypes.c_void_p], ctypes.c_void_p),
                "cnp_empty_like": _function(runtime, "cnp_empty_like", [ctypes.c_void_p], ctypes.c_void_p),
                "cnp_full_like": _function(runtime, "cnp_full_like", [ctypes.c_void_p, ctypes.c_double], ctypes.c_void_p),
                "cnp_astype": _function(runtime, "cnp_astype", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], ctypes.c_void_p),
            }
            for name, function in null_array_functions.items():
                with self.subTest(operation=name):
                    runtime.dll.cnp_clear_error()
                    pointer = function(None, -1.0) if name == "cnp_full_like" else (
                        function(None, 13, 4) if name == "cnp_astype" else (
                            function(None, 0) if name == "cnp_diag" else function(None)
                        )
                    )
                    self.assertFalse(pointer)
                    error = runtime.error_state()
                    self.assertNotEqual(0, error.status)
                    self.assertEqual(name, error.function)
                    self.assertEqual(baseline, runtime.retained_bytes)

            values = runtime.from_numpy(np.asarray([1.0, 2.0], dtype=np.float64))
            try:
                flat_get = _function(runtime, "cnp_array_flat_get", [ctypes.c_void_p, ctypes.c_int64], ctypes.c_double)
                flat_set = _function(runtime, "cnp_array_flat_set", [ctypes.c_void_p, ctypes.c_int64, ctypes.c_double], ctypes.c_int)
                runtime.dll.cnp_clear_error()
                self.assertTrue(np.isnan(flat_get(values.pointer, 2)))
                error = runtime.error_state()
                self.assertEqual(-6, error.status)
                self.assertEqual("cnp_array_flat_get", error.function)
                before = values.to_numpy().copy()
                runtime.dll.cnp_clear_error()
                self.assertEqual(-6, flat_set(values.pointer, -1, 9.0))
                np.testing.assert_array_equal(before, values.to_numpy())
                self.assertEqual("cnp_array_flat_set", runtime.error_state().function)
            finally:
                values.close()
            self.assertEqual(baseline, runtime.retained_bytes)


class DtypeDescriptorSurfaceTests(unittest.TestCase):
    """NumPy dtype descriptor mappings and cnumpy static-owner semantics."""

    class Descriptor(ctypes.Structure):
        _fields_ = [
            ("type_num", ctypes.c_int),
            ("elsize", ctypes.c_int),
            ("alignment", ctypes.c_int),
            ("kind", ctypes.c_char),
            ("byteorder", ctypes.c_char),
            ("name", ctypes.c_char * 32),
            ("refcount", ctypes.c_int),
        ]

    def test_descriptor_constructors_match_numpy_125_and_balance_references(
        self,
    ) -> None:
        cases = (
            (1, "?", "bool"),
            (2, "b", "int8"),
            (3, "B", "uint8"),
            (4, "h", "int16"),
            (5, "H", "uint16"),
            (6, "i", "int32"),
            (7, "I", "uint32"),
            (10, "q", "int64"),
            (11, "Q", "uint64"),
            (12, "f", "float32"),
            (13, "d", "float64"),
            (14, "g", "longdouble"),
            (15, "F", "complex64"),
            (16, "D", "complex128"),
            (17, "G", "clongdouble"),
            (18, "O", "object"),
            (19, "S", "bytes"),
            (20, "U", "str"),
            (21, "V", "void"),
            (22, "M", "datetime64"),
            (23, "m", "timedelta64"),
            (24, "e", "float16"),
        )
        descriptor_pointer = ctypes.POINTER(self.Descriptor)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            dtype_new = _function(
                runtime, "cnp_dtype_new", [ctypes.c_int], descriptor_pointer
            )
            from_char = _function(
                runtime,
                "cnp_dtype_from_char",
                [ctypes.c_char],
                descriptor_pointer,
            )
            from_string = _function(
                runtime,
                "cnp_dtype_from_string",
                [ctypes.c_char_p],
                descriptor_pointer,
            )
            incref = _function(
                runtime, "cnp_dtype_incref", [descriptor_pointer], None
            )
            decref = _function(
                runtime, "cnp_dtype_decref", [descriptor_pointer], None
            )

            for type_number, char_code, name in cases:
                with self.subTest(name=name):
                    expected = np.dtype(name)
                    by_number = dtype_new(type_number)
                    by_char = from_char(char_code.encode("ascii"))
                    by_name = from_string(name.encode("ascii"))
                    self.assertTrue(by_number)
                    self.assertTrue(by_char)
                    self.assertTrue(by_name)
                    self.assertEqual(type_number, by_number.contents.type_num)
                    self.assertEqual(type_number, by_char.contents.type_num)
                    self.assertEqual(type_number, by_name.contents.type_num)
                    self.assertEqual(expected.itemsize, by_number.contents.elsize)
                    self.assertEqual(
                        expected.kind.encode("ascii"), by_number.contents.kind
                    )
                    original = by_number.contents.refcount
                    incref(by_number)
                    self.assertEqual(original + 1, by_number.contents.refcount)
                    decref(by_number)
                    self.assertEqual(original, by_number.contents.refcount)
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_invalid_descriptor_requests_surface_exact_native_errors(
        self,
    ) -> None:
        descriptor_pointer = ctypes.POINTER(self.Descriptor)
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            calls = (
                (
                    "cnp_dtype_new",
                    _function(
                        runtime,
                        "cnp_dtype_new",
                        [ctypes.c_int],
                        descriptor_pointer,
                    ),
                    (0,),
                ),
                (
                    "cnp_dtype_from_char",
                    _function(
                        runtime,
                        "cnp_dtype_from_char",
                        [ctypes.c_char],
                        descriptor_pointer,
                    ),
                    (b"!",),
                ),
                (
                    "cnp_dtype_from_string",
                    _function(
                        runtime,
                        "cnp_dtype_from_string",
                        [ctypes.c_char_p],
                        descriptor_pointer,
                    ),
                    (b"not-a-dtype",),
                ),
            )
            for name, function, arguments in calls:
                with self.subTest(operation=name):
                    runtime.dll.cnp_clear_error()
                    self.assertFalse(function(*arguments))
                    error = runtime.error_state()
                    self.assertEqual(-3, error.status)
                    self.assertEqual(name, error.function)
                    self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
