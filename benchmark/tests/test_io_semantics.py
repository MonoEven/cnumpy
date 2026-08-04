from __future__ import annotations

import ctypes
import io
import struct
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path

import numpy as np

from compat.cnumpy_ctypes import CnumpyRuntime


ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT / "build" / "x64" / "Release" / "cnumpy_ahk.dll"


def write_raw_deflate_npz(
    path: Path, member_name: str, payload: bytes, compressed: bytes
) -> None:
    name = f"{member_name}.npy".encode("utf-8")
    crc = zlib.crc32(payload)
    local = struct.pack(
        "<IHHHHHIIIHH",
        0x04034B50,
        20,
        0x0800,
        8,
        0,
        0x0021,
        crc,
        len(compressed),
        len(payload),
        len(name),
        0,
    )
    central_offset = len(local) + len(name) + len(compressed)
    central = struct.pack(
        "<IHHHHHHIIIHHHHHII",
        0x02014B50,
        20,
        20,
        0x0800,
        8,
        0,
        0x0021,
        crc,
        len(compressed),
        len(payload),
        len(name),
        0,
        0,
        0,
        0,
        0,
        0,
    )
    eocd = struct.pack(
        "<IHHHHIIH",
        0x06054B50,
        0,
        0,
        1,
        1,
        len(central) + len(name),
        central_offset,
        0,
    )
    path.write_bytes(local + name + compressed + central + name + eocd)


class NpySemanticsTests(unittest.TestCase):
    def test_loads_numpy_fortran_order_file_in_logical_order(self) -> None:
        expected = np.asfortranarray(
            np.array([[1.25, 2.5, 3.75], [4.0, 5.5, 6.25]])
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fortran.npy"
            np.save(path, expected, allow_pickle=False)

            with CnumpyRuntime(DLL) as runtime:
                baseline = runtime.retained_bytes
                function = runtime.dll.cnp_load
                function.argtypes = [ctypes.c_char_p]
                function.restype = ctypes.c_void_p
                runtime.dll.cnp_clear_error()
                actual = runtime._owned_result(
                    function(str(path).encode("utf-8")), "cnp_load"
                )
                with actual:
                    self.assertEqual(expected.shape, actual.shape)
                    self.assertEqual(expected.dtype, actual.numpy_dtype)
                    np.testing.assert_array_equal(expected, actual.to_numpy())

                self.assertEqual(baseline, runtime.retained_bytes)

    def test_loads_big_endian_numpy_data_as_native_values(self) -> None:
        expected = np.array(
            [[1, -2, 0x1234], [-0x1234, 32767, -32768]], dtype=">i2"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "big-endian.npy"
            np.save(path, expected, allow_pickle=False)

            with CnumpyRuntime(DLL) as runtime:
                baseline = runtime.retained_bytes
                function = runtime.dll.cnp_load
                function.argtypes = [ctypes.c_char_p]
                function.restype = ctypes.c_void_p
                runtime.dll.cnp_clear_error()
                actual = runtime._owned_result(
                    function(str(path).encode("utf-8")), "cnp_load"
                )
                with actual:
                    self.assertEqual(np.dtype(np.int16), actual.numpy_dtype)
                    np.testing.assert_array_equal(
                        expected.astype(np.int16), actual.to_numpy()
                    )

                self.assertEqual(baseline, runtime.retained_bytes)

    def test_loads_big_endian_complex_components_without_reordering(self) -> None:
        expected = np.array(
            [1.25 + 2.5j, -3.75 + 0.125j, 8.0 - 16.5j], dtype=">c16"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "big-endian-complex.npy"
            np.save(path, expected, allow_pickle=False)

            with CnumpyRuntime(DLL) as runtime:
                baseline = runtime.retained_bytes
                function = runtime.dll.cnp_load
                function.argtypes = [ctypes.c_char_p]
                function.restype = ctypes.c_void_p
                runtime.dll.cnp_clear_error()
                actual = runtime._owned_result(
                    function(str(path).encode("utf-8")), "cnp_load"
                )
                with actual:
                    self.assertEqual(
                        np.dtype(np.complex128), actual.numpy_dtype
                    )
                    np.testing.assert_array_equal(
                        expected.astype(np.complex128), actual.to_numpy()
                    )

                self.assertEqual(baseline, runtime.retained_bytes)

    def test_loads_version_two_header_without_truncating_its_length(self) -> None:
        expected = np.array([1.25, -2.5, 7.75], dtype=np.float32)
        header_text = (
            "{'descr': '<f4', 'fortran_order': False, "
            "'shape': (3,), }"
        )
        header_length = 65536
        while (12 + header_length) % 64 != 0:
            header_length += 1
        header = (
            header_text
            + " " * (header_length - len(header_text) - 1)
            + "\n"
        ).encode("latin1")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "version-two-large-header.npy"
            with path.open("wb") as stream:
                stream.write(b"\x93NUMPY\x02\x00")
                stream.write(struct.pack("<I", len(header)))
                stream.write(header)
                stream.write(expected.tobytes(order="C"))
            np.testing.assert_array_equal(
                expected, np.load(path, max_header_size=len(header))
            )

            with CnumpyRuntime(DLL) as runtime:
                baseline = runtime.retained_bytes
                function = runtime.dll.cnp_load
                function.argtypes = [ctypes.c_char_p]
                function.restype = ctypes.c_void_p
                runtime.dll.cnp_clear_error()
                actual = runtime._owned_result(
                    function(str(path).encode("utf-8")), "cnp_load"
                )
                with actual:
                    self.assertEqual(expected.dtype, actual.numpy_dtype)
                    np.testing.assert_array_equal(expected, actual.to_numpy())

                self.assertEqual(baseline, runtime.retained_bytes)

    def test_saves_complex128_file_readable_by_numpy(self) -> None:
        expected = np.array(
            [[1.25 + 2.5j, -3.75 + 0.125j], [8.0 - 16.5j, 0.0 + 4.0j]],
            dtype=np.complex128,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "complex.npy"
            with CnumpyRuntime(DLL) as runtime:
                baseline = runtime.retained_bytes
                with runtime.from_numpy(expected) as source:
                    function = runtime.dll.cnp_save
                    function.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
                    function.restype = ctypes.c_int
                    runtime.dll.cnp_clear_error()
                    status = function(
                        str(path).encode("utf-8"), source.pointer
                    )
                    self.assertEqual(0, status, runtime.error_state())

                self.assertEqual(baseline, runtime.retained_bytes)
            actual = np.load(path, allow_pickle=False)
            self.assertEqual(expected.dtype, actual.dtype)
            np.testing.assert_array_equal(expected, actual)

    def test_preserves_fortran_order_when_saving_loaded_array(self) -> None:
        expected = np.asfortranarray(
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        )
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source-fortran.npy"
            destination_path = Path(directory) / "saved-fortran.npy"
            np.save(source_path, expected, allow_pickle=False)

            with CnumpyRuntime(DLL) as runtime:
                baseline = runtime.retained_bytes
                load = runtime.dll.cnp_load
                load.argtypes = [ctypes.c_char_p]
                load.restype = ctypes.c_void_p
                save = runtime.dll.cnp_save
                save.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
                save.restype = ctypes.c_int
                source = runtime._owned_result(
                    load(str(source_path).encode("utf-8")), "cnp_load"
                )
                with source:
                    status = save(
                        str(destination_path).encode("utf-8"), source.pointer
                    )
                    self.assertEqual(0, status, runtime.error_state())

                self.assertEqual(baseline, runtime.retained_bytes)

            actual = np.load(destination_path, allow_pickle=False)
            self.assertTrue(actual.flags.f_contiguous)
            np.testing.assert_array_equal(expected, actual)

    def test_supported_dtypes_round_trip_between_numpy_and_cnumpy(self) -> None:
        cases = (
            np.array([False, True, True, False], dtype=np.bool_),
            np.array([-128, -1, 0, 127], dtype=np.int8),
            np.array([0, 1, 128, 255], dtype=np.uint8),
            np.array([-32768, -1, 0, 32767], dtype=np.int16),
            np.array([0, 1, 32768, 65535], dtype=np.uint16),
            np.array([-(2**31), -1, 0, 2**31 - 1], dtype=np.int32),
            np.array([0, 1, 2**31, 2**32 - 1], dtype=np.uint32),
            np.array(
                [-(2**60) + 7, -1, 2**53 + 3, 2**60 + 9],
                dtype=np.int64,
            ),
            np.array([0, 2**53 + 5, 2**63 + 7, 2**64 - 1], dtype=np.uint64),
            np.array([-0.0, 1.25, -3.5, np.inf], dtype=np.float32),
            np.array([-0.0, 1.25, -3.5, np.inf], dtype=np.float64),
            np.array([1 + 2j, -3 + 0.25j, 0 - 0j, 8 - 16j], dtype=np.complex64),
            np.array(
                [1 + 2j, -3 + 0.25j, 0 - 0j, 8 - 16j],
                dtype=np.complex128,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for expected in cases:
                with self.subTest(dtype=expected.dtype):
                    source_path = Path(directory) / f"source-{expected.dtype}.npy"
                    saved_path = Path(directory) / f"saved-{expected.dtype}.npy"
                    np.save(source_path, expected, allow_pickle=False)

                    with CnumpyRuntime(DLL) as runtime:
                        baseline = runtime.retained_bytes
                        load = runtime.dll.cnp_load
                        load.argtypes = [ctypes.c_char_p]
                        load.restype = ctypes.c_void_p
                        save = runtime.dll.cnp_save
                        save.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
                        save.restype = ctypes.c_int
                        source = runtime._owned_result(
                            load(str(source_path).encode("utf-8")),
                            "cnp_load",
                        )
                        with source:
                            self.assertEqual(
                                expected.dtype, source.numpy_dtype
                            )
                            np.testing.assert_array_equal(
                                expected, source.to_numpy()
                            )
                            status = save(
                                str(saved_path).encode("utf-8"),
                                source.pointer,
                            )
                            self.assertEqual(
                                0, status, runtime.error_state()
                            )

                        self.assertEqual(baseline, runtime.retained_bytes)

                    actual = np.load(saved_path, allow_pickle=False)
                    self.assertEqual(expected.dtype, actual.dtype)
                    np.testing.assert_array_equal(expected, actual)

    def test_scalar_and_zero_length_shapes_round_trip(self) -> None:
        cases = (
            np.array(7.25, dtype=np.float64),
            np.empty((2, 0, 3), dtype=np.int16),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, expected in enumerate(cases):
                with self.subTest(shape=expected.shape):
                    source_path = Path(directory) / f"shape-source-{index}.npy"
                    saved_path = Path(directory) / f"shape-saved-{index}.npy"
                    np.save(source_path, expected, allow_pickle=False)

                    with CnumpyRuntime(DLL) as runtime:
                        baseline = runtime.retained_bytes
                        load = runtime.dll.cnp_load
                        load.argtypes = [ctypes.c_char_p]
                        load.restype = ctypes.c_void_p
                        save = runtime.dll.cnp_save
                        save.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
                        save.restype = ctypes.c_int
                        source = runtime._owned_result(
                            load(str(source_path).encode("utf-8")),
                            "cnp_load",
                        )
                        with source:
                            self.assertEqual(expected.shape, source.shape)
                            np.testing.assert_array_equal(
                                expected, source.to_numpy()
                            )
                            status = save(
                                str(saved_path).encode("utf-8"),
                                source.pointer,
                            )
                            self.assertEqual(
                                0, status, runtime.error_state()
                            )

                        self.assertEqual(baseline, runtime.retained_bytes)

                    np.testing.assert_array_equal(
                        expected, np.load(saved_path, allow_pickle=False)
                    )

    def test_rejects_unsupported_object_dtype_without_reading_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "object.npy"
            np.save(
                path,
                np.array([{"value": 1}, "text"], dtype=object),
                allow_pickle=True,
            )

            with CnumpyRuntime(DLL) as runtime:
                baseline = runtime.retained_bytes
                function = runtime.dll.cnp_load
                function.argtypes = [ctypes.c_char_p]
                function.restype = ctypes.c_void_p
                runtime.dll.cnp_clear_error()
                pointer = function(str(path).encode("utf-8"))
                unexpected_result = None
                if pointer:
                    unexpected_result = runtime._owned_result(
                        pointer, "cnp_load"
                    )
                    unexpected_result.close()
                self.assertIsNone(unexpected_result)
                error = runtime.error_state()
                self.assertEqual("cnp_load", error.function)
                self.assertEqual(-3, error.status)
                self.assertIn("unsupported dtype", error.message.lower())
                self.assertEqual(baseline, runtime.retained_bytes)

    def test_rejects_truncated_header_as_io_error_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated-header.npy"
            path.write_bytes(
                b"\x93NUMPY\x01\x00" + struct.pack("<H", 64) + b"{'descr': "
            )

            with CnumpyRuntime(DLL) as runtime:
                baseline = runtime.retained_bytes
                function = runtime.dll.cnp_load
                function.argtypes = [ctypes.c_char_p]
                function.restype = ctypes.c_void_p
                runtime.dll.cnp_clear_error()
                pointer = function(str(path).encode("utf-8"))
                if pointer:
                    runtime._owned_result(pointer, "cnp_load").close()
                self.assertFalse(pointer)
                error = runtime.error_state()
                self.assertEqual("cnp_load", error.function)
                self.assertEqual(-11, error.status)
                self.assertIn("header", error.message.lower())
                self.assertEqual(baseline, runtime.retained_bytes)


class NpzSemanticsTests(unittest.TestCase):
    def test_loads_numpy_stored_zip_with_multiple_named_arrays(self) -> None:
        expected = {
            "integers": np.array([2**60 + 3, -7, 11], dtype=np.int64),
            "matrix": np.array(
                [[1.25, 2.5], [3.75, 5.0]], dtype=np.float32
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stored.npz"
            np.savez(path, **expected)

            with CnumpyRuntime(DLL) as runtime:
                baseline = runtime.retained_bytes
                maximum = 4
                names = (ctypes.c_char * (maximum * 64))()
                pointers = (ctypes.c_void_p * maximum)()
                function = runtime.dll.cnp_loadz
                function.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_int,
                ]
                function.restype = ctypes.c_int
                runtime.dll.cnp_clear_error()
                count = function(
                    str(path).encode("utf-8"), names, pointers, maximum
                )
                loaded = []
                try:
                    for index in range(maximum):
                        if pointers[index]:
                            loaded.append(
                                runtime._owned_result(
                                    pointers[index], "cnp_loadz"
                                )
                            )
                    self.assertEqual(2, count, runtime.error_state())
                    actual_names = []
                    for index in range(count):
                        raw_name = bytes(
                            names[index * 64 : (index + 1) * 64]
                        )
                        actual_names.append(
                            raw_name.split(b"\0", 1)[0].decode("utf-8")
                        )
                    self.assertEqual(list(expected), actual_names)
                    for name, array in zip(actual_names, loaded):
                        np.testing.assert_array_equal(
                            expected[name], array.to_numpy()
                        )
                finally:
                    for array in loaded:
                        array.close()

                self.assertEqual(baseline, runtime.retained_bytes)

    def test_loads_numpy_deflated_zip_member(self) -> None:
        expected = np.arange(4096, dtype=np.int32).reshape(64, 64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compressed.npz"
            np.savez_compressed(path, values=expected)
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(
                    zipfile.ZIP_DEFLATED,
                    archive.getinfo("values.npy").compress_type,
                )

            with CnumpyRuntime(DLL) as runtime:
                baseline = runtime.retained_bytes
                names = (ctypes.c_char * 64)()
                pointers = (ctypes.c_void_p * 1)()
                function = runtime.dll.cnp_loadz
                function.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_int,
                ]
                function.restype = ctypes.c_int
                runtime.dll.cnp_clear_error()
                count = function(
                    str(path).encode("utf-8"), names, pointers, 1
                )
                loaded = None
                try:
                    if pointers[0]:
                        loaded = runtime._owned_result(
                            pointers[0], "cnp_loadz"
                        )
                    self.assertEqual(1, count, runtime.error_state())
                    self.assertEqual(
                        "values",
                        bytes(names).split(b"\0", 1)[0].decode("utf-8"),
                    )
                    self.assertIsNotNone(loaded)
                    np.testing.assert_array_equal(
                        expected, loaded.to_numpy()
                    )
                finally:
                    if loaded is not None:
                        loaded.close()

                self.assertEqual(baseline, runtime.retained_bytes)

    def test_loads_stored_and_fixed_huffman_deflate_blocks(self) -> None:
        expected = np.zeros((32, 32), dtype=np.int16)
        npy = io.BytesIO()
        np.save(npy, expected, allow_pickle=False)
        payload = npy.getvalue()
        compressors = {
            0: zlib.compressobj(level=0, wbits=-15),
            1: zlib.compressobj(
                level=9, wbits=-15, strategy=zlib.Z_FIXED
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            for expected_block_type, compressor in compressors.items():
                with self.subTest(block_type=expected_block_type):
                    compressed = compressor.compress(payload)
                    compressed += compressor.flush()
                    self.assertEqual(
                        expected_block_type, (compressed[0] >> 1) & 0x03
                    )
                    path = Path(directory) / (
                        f"deflate-{expected_block_type}.npz"
                    )
                    write_raw_deflate_npz(
                        path, "values", payload, compressed
                    )
                    with np.load(path, allow_pickle=False) as archive:
                        np.testing.assert_array_equal(
                            expected, archive["values"]
                        )

                    with CnumpyRuntime(DLL) as runtime:
                        baseline = runtime.retained_bytes
                        names = (ctypes.c_char * 64)()
                        pointers = (ctypes.c_void_p * 1)()
                        function = runtime.dll.cnp_loadz
                        function.argtypes = [
                            ctypes.c_char_p,
                            ctypes.c_void_p,
                            ctypes.POINTER(ctypes.c_void_p),
                            ctypes.c_int,
                        ]
                        function.restype = ctypes.c_int
                        runtime.dll.cnp_clear_error()
                        count = function(
                            str(path).encode("utf-8"),
                            names,
                            pointers,
                            1,
                        )
                        loaded = None
                        try:
                            if pointers[0]:
                                loaded = runtime._owned_result(
                                    pointers[0], "cnp_loadz"
                                )
                            self.assertEqual(
                                1, count, runtime.error_state()
                            )
                            self.assertIsNotNone(loaded)
                            np.testing.assert_array_equal(
                                expected, loaded.to_numpy()
                            )
                        finally:
                            if loaded is not None:
                                loaded.close()

                        self.assertEqual(
                            baseline, runtime.retained_bytes
                        )

    def test_rejects_member_with_crc_mismatch_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-crc.npz"
            np.savez(path, values=np.array([1, 2, 3], dtype=np.int32))
            with zipfile.ZipFile(path) as archive:
                info = archive.getinfo("values.npy")
                self.assertEqual(zipfile.ZIP_STORED, info.compress_type)
            archive_bytes = bytearray(path.read_bytes())
            name_length, extra_length = struct.unpack_from(
                "<HH", archive_bytes, info.header_offset + 26
            )
            data_offset = (
                info.header_offset + 30 + name_length + extra_length
            )
            archive_bytes[data_offset + info.compress_size - 1] ^= 0x01
            path.write_bytes(archive_bytes)
            with self.assertRaises(zipfile.BadZipFile):
                with np.load(path, allow_pickle=False) as archive:
                    archive["values"]

            with CnumpyRuntime(DLL) as runtime:
                baseline = runtime.retained_bytes
                names = (ctypes.c_char * 64)()
                pointers = (ctypes.c_void_p * 1)()
                function = runtime.dll.cnp_loadz
                function.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_int,
                ]
                function.restype = ctypes.c_int
                runtime.dll.cnp_clear_error()
                count = function(
                    str(path).encode("utf-8"), names, pointers, 1
                )
                if pointers[0]:
                    runtime._owned_result(
                        pointers[0], "cnp_loadz"
                    ).close()
                self.assertEqual(0, count)
                self.assertFalse(pointers[0])
                self.assertEqual(b"", bytes(names).split(b"\0", 1)[0])
                error = runtime.error_state()
                self.assertEqual("cnp_loadz", error.function)
                self.assertEqual(-11, error.status)
                self.assertIn("crc", error.message.lower())
                self.assertEqual(baseline, runtime.retained_bytes)

    def test_saves_named_arrays_as_numpy_readable_stored_zip(self) -> None:
        expected = {
            "integers": np.array(
                [-(2**60) + 7, 2**53 + 3, 2**60 + 9], dtype=np.int64
            ),
            "signal": np.array(
                [1.25 + 2.5j, -3.75 + 0.125j], dtype=np.complex128
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saved.npz"
            with CnumpyRuntime(DLL) as runtime:
                baseline = runtime.retained_bytes
                sources = [
                    runtime.from_numpy(array) for array in expected.values()
                ]
                try:
                    count = len(sources)
                    names = (ctypes.c_char_p * count)(
                        *(name.encode("utf-8") for name in expected)
                    )
                    pointers = (ctypes.c_void_p * count)(
                        *(source.pointer for source in sources)
                    )
                    function = runtime.dll.cnp_savez
                    function.argtypes = [
                        ctypes.c_char_p,
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_char_p),
                        ctypes.POINTER(ctypes.c_void_p),
                    ]
                    function.restype = ctypes.c_int
                    runtime.dll.cnp_clear_error()
                    status = function(
                        str(path).encode("utf-8"),
                        count,
                        names,
                        pointers,
                    )
                    self.assertEqual(0, status, runtime.error_state())
                finally:
                    for source in sources:
                        source.close()

                self.assertEqual(baseline, runtime.retained_bytes)

            with zipfile.ZipFile(path) as archive:
                self.assertEqual(
                    [f"{name}.npy" for name in expected],
                    archive.namelist(),
                )
                self.assertTrue(
                    all(
                        info.compress_type == zipfile.ZIP_STORED
                        for info in archive.infolist()
                    )
                )
            with np.load(path, allow_pickle=False) as actual:
                self.assertEqual(list(expected), actual.files)
                for name, array in expected.items():
                    self.assertEqual(array.dtype, actual[name].dtype)
                    np.testing.assert_array_equal(array, actual[name])

    def test_savez_auto_uses_numpy_positional_member_names(self) -> None:
        expected = (
            np.array([False, True, True], dtype=np.bool_),
            np.array([[1.25, -2.5], [3.75, 4.0]], dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auto.npz"
            with CnumpyRuntime(DLL) as runtime:
                baseline = runtime.retained_bytes
                sources = [runtime.from_numpy(array) for array in expected]
                try:
                    pointers = (ctypes.c_void_p * len(sources))(
                        *(source.pointer for source in sources)
                    )
                    function = runtime.dll.cnp_savez_auto
                    function.argtypes = [
                        ctypes.c_char_p,
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_void_p),
                    ]
                    function.restype = ctypes.c_int
                    runtime.dll.cnp_clear_error()
                    status = function(
                        str(path).encode("utf-8"),
                        len(sources),
                        pointers,
                    )
                    self.assertEqual(0, status, runtime.error_state())
                finally:
                    for source in sources:
                        source.close()

                self.assertEqual(baseline, runtime.retained_bytes)

            with np.load(path, allow_pickle=False) as actual:
                self.assertEqual(["arr_0", "arr_1"], actual.files)
                for index, array in enumerate(expected):
                    np.testing.assert_array_equal(
                        array, actual[f"arr_{index}"]
                    )

    def test_loadz_capacity_failure_releases_partial_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "too-many.npz"
            np.savez(
                path,
                first=np.array([1, 2, 3], dtype=np.int16),
                second=np.array([4.5, 6.75], dtype=np.float64),
            )

            with CnumpyRuntime(DLL) as runtime:
                baseline = runtime.retained_bytes
                names = (ctypes.c_char * 64)()
                pointers = (ctypes.c_void_p * 1)()
                function = runtime.dll.cnp_loadz
                function.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_int,
                ]
                function.restype = ctypes.c_int
                runtime.dll.cnp_clear_error()
                count = function(
                    str(path).encode("utf-8"), names, pointers, 1
                )
                if pointers[0]:
                    runtime._owned_result(
                        pointers[0], "cnp_loadz"
                    ).close()
                self.assertEqual(0, count)
                self.assertFalse(pointers[0])
                self.assertEqual(b"", bytes(names).split(b"\0", 1)[0])
                error = runtime.error_state()
                self.assertEqual("cnp_loadz", error.function)
                self.assertEqual(-4, error.status)
                self.assertIn("capacity", error.message.lower())
                self.assertEqual(baseline, runtime.retained_bytes)

    def test_reports_unsupported_zip_compression_method(self) -> None:
        expected = np.array([1.25, -3.5, 8.0], dtype=np.float64)
        npy = io.BytesIO()
        np.save(npy, expected, allow_pickle=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bzip2.npz"
            with zipfile.ZipFile(
                path, "w", compression=zipfile.ZIP_BZIP2
            ) as archive:
                archive.writestr("values.npy", npy.getvalue())
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(expected, archive["values"])

            with CnumpyRuntime(DLL) as runtime:
                baseline = runtime.retained_bytes
                names = (ctypes.c_char * 64)()
                pointers = (ctypes.c_void_p * 1)()
                function = runtime.dll.cnp_loadz
                function.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_int,
                ]
                function.restype = ctypes.c_int
                runtime.dll.cnp_clear_error()
                count = function(
                    str(path).encode("utf-8"), names, pointers, 1
                )
                self.assertEqual(0, count)
                self.assertFalse(pointers[0])
                error = runtime.error_state()
                self.assertEqual("cnp_loadz", error.function)
                self.assertEqual(-12, error.status)
                self.assertIn("method 12", error.message.lower())
                self.assertEqual(baseline, runtime.retained_bytes)

    def test_rejects_local_and_central_member_name_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "name-mismatch.npz"
            np.savez(path, values=np.array([2, 4, 8], dtype=np.uint16))
            with zipfile.ZipFile(path) as archive:
                info = archive.getinfo("values.npy")
            archive_bytes = bytearray(path.read_bytes())
            self.assertEqual(
                b"values.npy",
                bytes(
                    archive_bytes[
                        info.header_offset + 30 :
                        info.header_offset + 30 + len("values.npy")
                    ]
                ),
            )
            archive_bytes[info.header_offset + 30] = ord("x")
            path.write_bytes(archive_bytes)
            with self.assertRaises(zipfile.BadZipFile):
                with np.load(path, allow_pickle=False) as archive:
                    archive["values"]

            with CnumpyRuntime(DLL) as runtime:
                baseline = runtime.retained_bytes
                names = (ctypes.c_char * 64)()
                pointers = (ctypes.c_void_p * 1)()
                function = runtime.dll.cnp_loadz
                function.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_int,
                ]
                function.restype = ctypes.c_int
                runtime.dll.cnp_clear_error()
                count = function(
                    str(path).encode("utf-8"), names, pointers, 1
                )
                if pointers[0]:
                    runtime._owned_result(
                        pointers[0], "cnp_loadz"
                    ).close()
                self.assertEqual(0, count)
                self.assertFalse(pointers[0])
                error = runtime.error_state()
                self.assertEqual("cnp_loadz", error.function)
                self.assertEqual(-11, error.status)
                self.assertIn("name", error.message.lower())
                self.assertEqual(baseline, runtime.retained_bytes)


class RawFileSemanticsTests(unittest.TestCase):
    def test_fromfile_offset_past_end_reports_value_error_at_entry_point(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "offset-past-end.dat"
            path.write_bytes(b"1234")
            with CnumpyRuntime(DLL) as runtime:
                baseline = runtime.retained_bytes
                fromfile = runtime.dll.cnp_fromfile
                fromfile.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.c_int64,
                    ctypes.c_int64,
                ]
                fromfile.restype = ctypes.c_void_p
                runtime.dll.cnp_clear_error()
                self.assertFalse(
                    fromfile(str(path).encode("utf-8"), 3, -1, 5)
                )
                error = runtime.error_state()
                self.assertEqual(-4, error.status)
                self.assertEqual("cnp_fromfile", error.function)
                self.assertIn("offset", error.message.lower())
                self.assertEqual(baseline, runtime.retained_bytes)

    def test_fromfile_count_zero_returns_an_empty_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "count-zero.dat"
            path.write_bytes(bytes(range(16)))
            with CnumpyRuntime(DLL) as runtime:
                baseline = runtime.retained_bytes
                fromfile = runtime.dll.cnp_fromfile
                fromfile.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.c_int64,
                    ctypes.c_int64,
                ]
                fromfile.restype = ctypes.c_void_p
                runtime.dll.cnp_clear_error()
                result = runtime._owned_result(
                    fromfile(str(path).encode("utf-8"), 3, 0, 0),
                    "cnp_fromfile",
                )
                with result:
                    self.assertEqual((0,), result.shape)
                    self.assertEqual([], result.to_numpy().tolist())

                self.assertEqual(baseline, runtime.retained_bytes)

    def test_tofile_serializes_noncontiguous_view_in_logical_order(
        self,
    ) -> None:
        base_value = np.arange(12, dtype=np.int64).reshape(3, 4)
        expected = base_value.T
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "noncontiguous.dat"
            with CnumpyRuntime(DLL) as runtime:
                baseline = runtime.retained_bytes
                with runtime.from_numpy(base_value) as base:
                    with runtime.transpose(base, (1, 0)) as view:
                        self.assertFalse(view.c_contiguous)
                        tofile = runtime.dll.cnp_tofile
                        tofile.argtypes = [
                            ctypes.c_void_p,
                            ctypes.c_char_p,
                        ]
                        tofile.restype = ctypes.c_int
                        runtime.dll.cnp_clear_error()
                        self.assertEqual(
                            0,
                            tofile(
                                view.pointer,
                                str(path).encode("utf-8"),
                            ),
                            runtime.error_state(),
                        )

                self.assertEqual(baseline, runtime.retained_bytes)

            self.assertEqual(
                expected.tobytes(order="C"), path.read_bytes()
            )


class MemmapSemanticsTests(unittest.TestCase):
    def test_open_failure_reports_the_public_open_entry_point(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.dat"
            with CnumpyRuntime(DLL) as runtime:
                baseline = runtime.retained_bytes
                shape = (ctypes.c_int64 * 1)(1)
                open_memmap = runtime.dll.cnp_memmap_open
                open_memmap.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.c_int,
                    ctypes.c_int64,
                ]
                open_memmap.restype = ctypes.c_void_p
                runtime.dll.cnp_clear_error()
                self.assertFalse(
                    open_memmap(
                        str(path).encode("utf-8"), 1, shape, 6, 0
                    )
                )
                error = runtime.error_state()
                self.assertEqual(-11, error.status)
                self.assertEqual("cnp_memmap_open", error.function)
                self.assertIn("open", error.message.lower())
                self.assertEqual(baseline, runtime.retained_bytes)

    def test_open_is_read_only_and_reports_the_called_setter(self) -> None:
        expected = np.array([17, -29, 43], dtype=np.int32)
        offset = 5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "read-only.dat"
            path.write_bytes(b"prefix"[:offset] + expected.tobytes())

            with CnumpyRuntime(DLL) as runtime:
                baseline = runtime.retained_bytes
                shape = (ctypes.c_int64 * 1)(expected.size)
                open_memmap = runtime.dll.cnp_memmap_open
                open_memmap.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.c_int,
                    ctypes.c_int64,
                ]
                open_memmap.restype = ctypes.c_void_p
                runtime.dll.cnp_clear_error()
                mapped = runtime._owned_result(
                    open_memmap(
                        str(path).encode("utf-8"), 1, shape, 6, offset
                    ),
                    "cnp_memmap_open",
                )
                with mapped:
                    self.assertFalse(mapped.writeable)
                    np.testing.assert_array_equal(expected, mapped.to_numpy())

                    set_value = runtime.dll.cnp_array_set_int
                    set_value.argtypes = [
                        ctypes.c_void_p,
                        ctypes.POINTER(ctypes.c_int64),
                        ctypes.c_int64,
                    ]
                    set_value.restype = ctypes.c_int
                    coordinates = (ctypes.c_int64 * 1)(0)
                    runtime.dll.cnp_clear_error()
                    self.assertEqual(
                        -1,
                        set_value(mapped.pointer, coordinates, 999),
                    )
                    error = runtime.error_state()
                    self.assertEqual("cnp_array_set_int", error.function)
                    self.assertIn("not writeable", error.message.lower())
                    np.testing.assert_array_equal(expected, mapped.to_numpy())

                self.assertEqual(baseline, runtime.retained_bytes)

            self.assertEqual(
                b"prefix"[:offset] + expected.tobytes(), path.read_bytes()
            )

    def test_read_write_mode_persists_exact_int64_values(self) -> None:
        expected = np.array(
            [2**53 + 1, -(2**53) - 1, 2**62 + 3], dtype=np.int64
        )
        offset = 3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "read-write.dat"
            path.write_bytes(b"pre" + np.zeros_like(expected).tobytes())

            with CnumpyRuntime(DLL) as runtime:
                baseline = runtime.retained_bytes
                shape = (ctypes.c_int64 * 1)(expected.size)
                create = runtime.dll.cnp_memmap_create
                create.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int64,
                ]
                create.restype = ctypes.c_void_p
                runtime.dll.cnp_clear_error()
                mapped = runtime._owned_result(
                    create(
                        str(path).encode("utf-8"),
                        1,
                        shape,
                        8,
                        1,
                        offset,
                    ),
                    "cnp_memmap_create",
                )
                with mapped:
                    self.assertTrue(mapped.writeable)
                    set_value = runtime.dll.cnp_array_set_int
                    set_value.argtypes = [
                        ctypes.c_void_p,
                        ctypes.POINTER(ctypes.c_int64),
                        ctypes.c_int64,
                    ]
                    set_value.restype = ctypes.c_int
                    for index, value in enumerate(expected):
                        coordinates = (ctypes.c_int64 * 1)(index)
                        runtime.dll.cnp_clear_error()
                        self.assertEqual(
                            0,
                            set_value(
                                mapped.pointer, coordinates, int(value)
                            ),
                            runtime.error_state(),
                        )
                    np.testing.assert_array_equal(
                        expected, mapped.to_numpy()
                    )

                    flush = runtime.dll.cnp_memmap_flush
                    flush.argtypes = [ctypes.c_void_p]
                    flush.restype = ctypes.c_int
                    runtime.dll.cnp_clear_error()
                    self.assertEqual(
                        0, flush(mapped.pointer), runtime.error_state()
                    )

                self.assertEqual(baseline, runtime.retained_bytes)

            actual = np.memmap(
                path,
                dtype=expected.dtype,
                mode="r",
                offset=offset,
                shape=expected.shape,
            )
            try:
                np.testing.assert_array_equal(expected, actual)
            finally:
                del actual

    def test_copy_on_write_mode_never_persists_changes(self) -> None:
        original = np.array([3, 5, 7, 11], dtype=np.int32)
        changed = np.array([13, 17, 19, 23], dtype=np.int32)
        offset = 9
        prefix = b"copy-mode"[:offset]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "copy-on-write.dat"
            original_bytes = prefix + original.tobytes()
            path.write_bytes(original_bytes)

            with CnumpyRuntime(DLL) as runtime:
                baseline = runtime.retained_bytes
                shape = (ctypes.c_int64 * 1)(original.size)
                create = runtime.dll.cnp_memmap_create
                create.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int64,
                ]
                create.restype = ctypes.c_void_p
                runtime.dll.cnp_clear_error()
                mapped = runtime._owned_result(
                    create(
                        str(path).encode("utf-8"),
                        1,
                        shape,
                        6,
                        2,
                        offset,
                    ),
                    "cnp_memmap_create",
                )
                with mapped:
                    self.assertTrue(mapped.writeable)
                    set_value = runtime.dll.cnp_array_set_int
                    set_value.argtypes = [
                        ctypes.c_void_p,
                        ctypes.POINTER(ctypes.c_int64),
                        ctypes.c_int64,
                    ]
                    set_value.restype = ctypes.c_int
                    for index, value in enumerate(changed):
                        coordinates = (ctypes.c_int64 * 1)(index)
                        runtime.dll.cnp_clear_error()
                        self.assertEqual(
                            0,
                            set_value(
                                mapped.pointer, coordinates, int(value)
                            ),
                            runtime.error_state(),
                        )
                    np.testing.assert_array_equal(
                        changed, mapped.to_numpy()
                    )

                    flush = runtime.dll.cnp_memmap_flush
                    flush.argtypes = [ctypes.c_void_p]
                    flush.restype = ctypes.c_int
                    runtime.dll.cnp_clear_error()
                    self.assertEqual(
                        0, flush(mapped.pointer), runtime.error_state()
                    )

                self.assertEqual(baseline, runtime.retained_bytes)

            self.assertEqual(original_bytes, path.read_bytes())

    def test_view_keeps_mapping_alive_and_can_flush_after_root_close(
        self,
    ) -> None:
        expected = np.array([31, 37, 41, 43], dtype=np.int32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "view-lifetime.dat"
            with CnumpyRuntime(DLL) as runtime:
                baseline = runtime.retained_bytes
                shape = (ctypes.c_int64 * 1)(expected.size)
                create = runtime.dll.cnp_memmap_create
                create.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int64,
                ]
                create.restype = ctypes.c_void_p
                runtime.dll.cnp_clear_error()
                root_pointer = create(
                    str(path).encode("utf-8"), 1, shape, 6, 3, 0
                )
                self.assertTrue(root_pointer, runtime.error_state())

                close_root = runtime.dll.cnp_memmap_close
                close_root.argtypes = [ctypes.c_void_p]
                close_root.restype = None

                array_view = runtime.dll.cnp_array_view
                array_view.argtypes = [ctypes.c_void_p]
                array_view.restype = ctypes.c_void_p
                runtime.dll.cnp_clear_error()
                view_pointer = array_view(root_pointer)
                if not view_pointer:
                    close_root(root_pointer)
                    self.fail(str(runtime.error_state()))

                close_root(root_pointer)

                view = runtime._owned_result(
                    view_pointer, "cnp_array_view"
                )
                with view:
                    set_value = runtime.dll.cnp_array_set_int
                    set_value.argtypes = [
                        ctypes.c_void_p,
                        ctypes.POINTER(ctypes.c_int64),
                        ctypes.c_int64,
                    ]
                    set_value.restype = ctypes.c_int
                    for index, value in enumerate(expected):
                        coordinates = (ctypes.c_int64 * 1)(index)
                        runtime.dll.cnp_clear_error()
                        self.assertEqual(
                            0,
                            set_value(
                                view.pointer, coordinates, int(value)
                            ),
                            runtime.error_state(),
                        )
                    np.testing.assert_array_equal(expected, view.to_numpy())

                    flush = runtime.dll.cnp_memmap_flush
                    flush.argtypes = [ctypes.c_void_p]
                    flush.restype = ctypes.c_int
                    runtime.dll.cnp_clear_error()
                    self.assertEqual(
                        0, flush(view.pointer), runtime.error_state()
                    )

                self.assertEqual(baseline, runtime.retained_bytes)

            self.assertEqual(expected.tobytes(), path.read_bytes())

    def test_invalid_create_requests_are_explicit_and_leak_free(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            undersized = root / "undersized.dat"
            undersized.write_bytes(b"tiny")
            with CnumpyRuntime(DLL) as runtime:
                baseline = runtime.retained_bytes
                create = runtime.dll.cnp_memmap_create
                create.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int64,
                ]
                create.restype = ctypes.c_void_p
                valid_shape = (ctypes.c_int64 * 1)(2)
                negative_shape = (ctypes.c_int64 * 1)(-1)
                empty_shape = (ctypes.c_int64 * 1)(0)
                cases = (
                    (
                        "invalid-mode.dat",
                        valid_shape,
                        4,
                        0,
                        -1,
                        "mode",
                    ),
                    (
                        "negative-offset.dat",
                        valid_shape,
                        3,
                        -1,
                        -11,
                        "offset",
                    ),
                    (
                        "negative-shape.dat",
                        negative_shape,
                        3,
                        0,
                        -4,
                        "negative",
                    ),
                    (
                        "empty-shape.dat",
                        empty_shape,
                        3,
                        0,
                        -4,
                        "empty",
                    ),
                    (
                        "undersized.dat",
                        valid_shape,
                        0,
                        0,
                        -11,
                        "exceeds",
                    ),
                )
                for (
                    filename,
                    shape,
                    mode,
                    offset,
                    expected_status,
                    expected_message,
                ) in cases:
                    with self.subTest(filename=filename):
                        runtime.dll.cnp_clear_error()
                        self.assertFalse(
                            create(
                                str(root / filename).encode("utf-8"),
                                1,
                                shape,
                                6,
                                mode,
                                offset,
                            )
                        )
                        error = runtime.error_state()
                        self.assertEqual(expected_status, error.status)
                        self.assertEqual(
                            "cnp_memmap_create", error.function
                        )
                        self.assertIn(
                            expected_message, error.message.lower()
                        )
                        self.assertEqual(baseline, runtime.retained_bytes)

    def test_repeated_close_releases_every_windows_mapping_handle(
        self,
    ) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.argtypes = []
        get_current_process.restype = ctypes.c_void_p
        get_process_handle_count = kernel32.GetProcessHandleCount
        get_process_handle_count.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ulong),
        ]
        get_process_handle_count.restype = ctypes.c_int
        process = get_current_process()

        def handle_count() -> int:
            count = ctypes.c_ulong()
            self.assertTrue(
                get_process_handle_count(process, ctypes.byref(count)),
                ctypes.get_last_error(),
            )
            return int(count.value)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "handle-loop.dat"
            encoded_path = str(path).encode("utf-8")
            with CnumpyRuntime(DLL) as runtime:
                retained_baseline = runtime.retained_bytes
                handle_baseline = handle_count()
                shape = (ctypes.c_int64 * 1)(16)
                create = runtime.dll.cnp_memmap_create
                create.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int64,
                ]
                create.restype = ctypes.c_void_p
                close = runtime.dll.cnp_memmap_close
                close.argtypes = [ctypes.c_void_p]
                close.restype = None

                for iteration in range(64):
                    runtime.dll.cnp_clear_error()
                    pointer = create(
                        encoded_path, 1, shape, 6, 3, iteration % 13
                    )
                    self.assertTrue(pointer, runtime.error_state())
                    close(pointer)
                    self.assertEqual(0, runtime.error_state().status)

                self.assertEqual(retained_baseline, runtime.retained_bytes)
                self.assertEqual(handle_baseline, handle_count())

    def test_create_write_flush_and_array_release_own_the_mapping(self) -> None:
        expected = np.array(
            [[11, -22, 33], [44, -55, 66]], dtype=np.int32
        )
        offset = 7
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "created.dat"
            with CnumpyRuntime(DLL) as runtime:
                baseline = runtime.retained_bytes
                shape = (ctypes.c_int64 * 2)(*expected.shape)
                create = runtime.dll.cnp_memmap_create
                create.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int64,
                ]
                create.restype = ctypes.c_void_p
                runtime.dll.cnp_clear_error()
                mapped = runtime._owned_result(
                    create(
                        str(path).encode("utf-8"),
                        2,
                        shape,
                        6,
                        3,
                        offset,
                    ),
                    "cnp_memmap_create",
                )
                with mapped:
                    set_value = runtime.dll.cnp_array_set_int
                    set_value.argtypes = [
                        ctypes.c_void_p,
                        ctypes.POINTER(ctypes.c_int64),
                        ctypes.c_int64,
                    ]
                    set_value.restype = ctypes.c_int
                    for row in range(expected.shape[0]):
                        for column in range(expected.shape[1]):
                            coordinates = (ctypes.c_int64 * 2)(
                                row, column
                            )
                            runtime.dll.cnp_clear_error()
                            status = set_value(
                                mapped.pointer,
                                coordinates,
                                int(expected[row, column]),
                            )
                            self.assertEqual(
                                0, status, runtime.error_state()
                            )
                    np.testing.assert_array_equal(
                        expected, mapped.to_numpy()
                    )
                    flush = runtime.dll.cnp_memmap_flush
                    flush.argtypes = [ctypes.c_void_p]
                    flush.restype = ctypes.c_int
                    runtime.dll.cnp_clear_error()
                    self.assertEqual(
                        0,
                        flush(mapped.pointer),
                        runtime.error_state(),
                    )

                self.assertEqual(baseline, runtime.retained_bytes)

            self.assertEqual(offset + expected.nbytes, path.stat().st_size)
            self.assertEqual(b"\0" * offset, path.read_bytes()[:offset])
            actual = np.memmap(
                path,
                dtype=expected.dtype,
                mode="r",
                offset=offset,
                shape=expected.shape,
            )
            try:
                np.testing.assert_array_equal(expected, actual)
            finally:
                del actual


if __name__ == "__main__":
    unittest.main()
