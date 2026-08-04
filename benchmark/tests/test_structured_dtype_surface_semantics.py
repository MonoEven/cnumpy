from __future__ import annotations

import ctypes
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime
from compat.oracle import assert_array_equivalent


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"

CNP_INT = 6
CNP_DOUBLE = 13


class StructuredDtypeSurfaceTests(unittest.TestCase):
    @staticmethod
    def _array_function(
        runtime: CnumpyRuntime,
        name: str,
        argtypes: list[object],
    ) -> ctypes._CFuncPtr:
        function = getattr(runtime.dll, name)
        function.argtypes = argtypes
        function.restype = ctypes.c_void_p
        return function

    @staticmethod
    def _create_dtype(runtime: CnumpyRuntime) -> tuple[int, np.dtype]:
        names = (ctypes.c_char_p * 2)(b"identifier", b"score")
        types = (ctypes.c_int * 2)(CNP_INT, CNP_DOUBLE)
        function = runtime.dll.cnp_struct_dtype_create
        function.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        function.restype = ctypes.c_int
        runtime.dll.cnp_clear_error()
        dtype_id = function(names, types, 2)
        if dtype_id < 0:
            raise runtime.native_error("cnp_struct_dtype_create")
        return dtype_id, np.dtype(
            [("identifier", np.int32), ("score", np.float64)], align=False
        )

    def test_dtype_registry_matches_numpy_packed_field_metadata(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            dtype_id, expected = self._create_dtype(runtime)

            itemsize = runtime.dll.cnp_struct_dtype_itemsize
            itemsize.argtypes = [ctypes.c_int]
            itemsize.restype = ctypes.c_int64
            nfields = runtime.dll.cnp_struct_dtype_nfields
            nfields.argtypes = [ctypes.c_int]
            nfields.restype = ctypes.c_int
            field_name = runtime.dll.cnp_struct_dtype_field_name
            field_name.argtypes = [ctypes.c_int, ctypes.c_int]
            field_name.restype = ctypes.c_char_p
            field_offset = runtime.dll.cnp_struct_dtype_field_offset
            field_offset.argtypes = [ctypes.c_int, ctypes.c_int]
            field_offset.restype = ctypes.c_int64
            find_field = runtime.dll.cnp_struct_dtype_find_field
            find_field.argtypes = [ctypes.c_int, ctypes.c_char_p]
            find_field.restype = ctypes.c_int

            self.assertEqual(expected.itemsize, itemsize(dtype_id))
            self.assertEqual(2, nfields(dtype_id))
            for index, name in enumerate(expected.names or ()):
                self.assertEqual(name, field_name(dtype_id, index).decode())
                self.assertEqual(
                    expected.fields[name][1], field_offset(dtype_id, index)
                )
                self.assertEqual(index, find_field(dtype_id, name.encode()))
            self.assertEqual(b"identifier,score", self._names(runtime, dtype_id))
            self.assertEqual(baseline, runtime.retained_bytes)

    @staticmethod
    def _names(runtime: CnumpyRuntime, dtype_id: int) -> bytes:
        function = runtime.dll.cnp_recarray_names
        function.argtypes = [ctypes.c_int]
        function.restype = ctypes.c_char_p
        value = function(dtype_id)
        if value is None:
            raise runtime.native_error("cnp_recarray_names")
        return value

    def test_record_fields_records_views_and_lifetime_match_numpy(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            dtype_id, expected_dtype = self._create_dtype(runtime)
            baseline = runtime.retained_bytes
            shape = (ctypes.c_int64 * 2)(2, 2)
            new = self._array_function(
                runtime,
                "cnp_recarray_new",
                [ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_int],
            )
            get_field = self._array_function(
                runtime,
                "cnp_recarray_get_field",
                [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int],
            )
            set_field = runtime.dll.cnp_recarray_set_field
            set_field.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            set_field.restype = ctypes.c_int
            get_record = self._array_function(
                runtime,
                "cnp_recarray_get_record",
                [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int],
            )
            set_record = runtime.dll.cnp_recarray_set_record
            set_record.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            set_record.restype = ctypes.c_int

            records = runtime._owned_result(
                new(2, shape, dtype_id), "cnp_recarray_new"
            )
            self.assertEqual((2, 2), records.shape)
            self.assertEqual(expected_dtype.itemsize, records.itemsize)
            expected = np.zeros((2, 2), dtype=expected_dtype)
            identifiers = np.asarray([[1, 2], [3, 4]], dtype=np.int32)
            expected["identifier"] = identifiers
            expected["score"] = 2.5
            with ExitStack() as stack:
                identifier_values = stack.enter_context(
                    runtime.from_numpy(identifiers)
                )
                score_value = stack.enter_context(
                    runtime.from_numpy(np.asarray(2.5, dtype=np.float64))
                )
                self.assertEqual(
                    0,
                    set_field(
                        records.pointer,
                        b"identifier",
                        dtype_id,
                        identifier_values.pointer,
                    ),
                    runtime.error_state(),
                )
                self.assertEqual(
                    0,
                    set_field(
                        records.pointer,
                        b"score",
                        dtype_id,
                        score_value.pointer,
                    ),
                    runtime.error_state(),
                )

            replacement = np.asarray([99.0, 1.25], dtype=np.float64)
            with runtime.from_numpy(replacement) as replacement_values:
                self.assertEqual(
                    0,
                    set_record(
                        records.pointer,
                        2,
                        dtype_id,
                        replacement_values.pointer,
                    ),
                    runtime.error_state(),
                )
            expected.reshape(-1)[2] = (99, 1.25)
            with runtime._owned_result(
                get_record(records.pointer, 2, dtype_id),
                "cnp_recarray_get_record",
            ) as record:
                np.testing.assert_allclose(
                    record.to_numpy(), np.asarray([99.0, 1.25])
                )

            identifier_field = runtime._owned_result(
                get_field(records.pointer, b"identifier", dtype_id),
                "cnp_recarray_get_field",
            )
            score_field = runtime._owned_result(
                get_field(records.pointer, b"score", dtype_id),
                "cnp_recarray_get_field",
            )
            records.close()
            try:
                assert_array_equivalent(
                    self, identifier_field, expected["identifier"]
                )
                assert_array_equivalent(self, score_field, expected["score"])
            finally:
                score_field.close()
                identifier_field.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_view_getfield_setfield_and_newbyteorder_observables(self) -> None:
        encoded = np.asarray(
            [
                [0x0000000200000001, 0x0000000400000003],
                [0x0000000600000005, 0x0000000800000007],
            ],
            dtype=np.int64,
        )
        with CnumpyRuntime(DLL) as runtime:
            baseline = runtime.retained_bytes
            view_function = self._array_function(
                runtime, "cnp_view", [ctypes.c_void_p, ctypes.c_int]
            )
            getfield = self._array_function(
                runtime,
                "cnp_getfield",
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_int64],
            )
            setfield = runtime.dll.cnp_setfield
            setfield.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int64,
            ]
            setfield.restype = ctypes.c_int
            newbyteorder = self._array_function(
                runtime, "cnp_newbyteorder", [ctypes.c_void_p]
            )

            source = runtime.from_numpy(encoded)
            reinterpreted = runtime._owned_result(
                view_function(source.pointer, CNP_INT), "cnp_view"
            )
            expected_source = encoded.copy()
            expected_view = expected_source.view(np.int32)
            runtime.dll.cnp_array_flat_set(
                reinterpreted.pointer, 1, ctypes.c_double(17)
            )
            expected_view.reshape(-1)[1] = 17
            assert_array_equivalent(self, source, expected_source)
            source.close()
            try:
                assert_array_equivalent(self, reinterpreted, expected_view)
            finally:
                reinterpreted.close()

            source = runtime.from_numpy(encoded)
            high_words = runtime._owned_result(
                getfield(source.pointer, CNP_INT, 4), "cnp_getfield"
            )
            source.close()
            try:
                assert_array_equivalent(
                    self, high_words, encoded.getfield(np.int32, 4)
                )
            finally:
                high_words.close()

            expected_setfield = encoded.copy()
            expected_setfield.setfield(
                np.asarray(9, dtype=np.int32), np.int32, 0
            )
            with runtime.from_numpy(encoded) as destination, runtime.from_numpy(
                np.asarray(9, dtype=np.int32)
            ) as scalar:
                runtime.dll.cnp_clear_error()
                self.assertEqual(
                    0,
                    setfield(
                        destination.pointer, scalar.pointer, CNP_INT, 0
                    ),
                    runtime.error_state(),
                )
                assert_array_equivalent(self, destination, expected_setfield)

            byteorder_expected = encoded.newbyteorder().astype(np.int64)
            source = runtime.from_numpy(encoded)
            swapped = runtime._owned_result(
                newbyteorder(source.pointer), "cnp_newbyteorder"
            )
            source.close()
            try:
                assert_array_equivalent(self, swapped, byteorder_expected)
            finally:
                swapped.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_structured_failures_are_labeled_atomic_and_nonretaining(self) -> None:
        with CnumpyRuntime(DLL) as runtime:
            dtype_id, _ = self._create_dtype(runtime)
            baseline = runtime.retained_bytes
            invalid_names = (ctypes.c_char_p * 2)(b"same", b"same")
            types = (ctypes.c_int * 2)(CNP_INT, CNP_DOUBLE)
            create = runtime.dll.cnp_struct_dtype_create
            create.argtypes = [
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
            ]
            create.restype = ctypes.c_int
            runtime.dll.cnp_clear_error()
            self.assertLess(create(invalid_names, types, 2), 0)
            self.assertEqual(
                "cnp_struct_dtype_create", runtime.error_state().function
            )

            itemsize = runtime.dll.cnp_struct_dtype_itemsize
            itemsize.argtypes = [ctypes.c_int]
            itemsize.restype = ctypes.c_int64
            runtime.dll.cnp_clear_error()
            self.assertEqual(-1, itemsize(-1))
            self.assertEqual(
                "cnp_struct_dtype_itemsize", runtime.error_state().function
            )

            get_field = self._array_function(
                runtime,
                "cnp_recarray_get_field",
                [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int],
            )
            shape = (ctypes.c_int64 * 1)(2)
            new = self._array_function(
                runtime,
                "cnp_recarray_new",
                [ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_int],
            )
            with runtime._owned_result(
                new(1, shape, dtype_id), "cnp_recarray_new"
            ) as records:
                runtime.dll.cnp_clear_error()
                self.assertFalse(
                    get_field(records.pointer, b"missing", dtype_id)
                )
                self.assertEqual(
                    "cnp_recarray_get_field", runtime.error_state().function
                )

            source = runtime.from_numpy(np.ones(3, dtype=np.uint8))
            view_function = self._array_function(
                runtime, "cnp_view", [ctypes.c_void_p, ctypes.c_int]
            )
            try:
                runtime.dll.cnp_clear_error()
                self.assertFalse(view_function(source.pointer, CNP_INT))
                self.assertEqual("cnp_view", runtime.error_state().function)
            finally:
                source.close()
            self.assertEqual(baseline, runtime.retained_bytes)

    def test_recfromtxt_homogeneous_projection_and_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory, CnumpyRuntime(
            DLL
        ) as runtime:
            path = Path(directory) / "records.csv"
            path.write_text("identifier,score\n1,2.5\n3,4.75\n", encoding="utf-8")
            function = self._array_function(
                runtime,
                "cnp_recfromtxt",
                [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
            )
            baseline = runtime.retained_bytes
            with runtime._owned_result(
                function(
                    str(path).encode(), b",", 1, CNP_DOUBLE
                ),
                "cnp_recfromtxt",
            ) as actual:
                assert_array_equivalent(
                    self,
                    actual,
                    np.genfromtxt(
                        path, delimiter=",", skip_header=1, dtype=np.float64
                    ),
                )
            runtime.dll.cnp_clear_error()
            self.assertFalse(
                function(
                    str(path.with_name("missing.csv")).encode(),
                    b",",
                    0,
                    CNP_DOUBLE,
                )
            )
            self.assertEqual("cnp_recfromtxt", runtime.error_state().function)
            self.assertEqual(baseline, runtime.retained_bytes)


if __name__ == "__main__":
    unittest.main()
