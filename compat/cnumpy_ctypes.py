from __future__ import annotations

import ctypes
from itertools import product
import operator
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


CNP_MAXDIMS = 64
CNP_ORDER_C = 0
CNP_ARRAY_C_CONTIGUOUS = 0x0001
CNP_ARRAY_F_CONTIGUOUS = 0x0002
CNP_ARRAY_OWNDATA = 0x0004
CNP_ARRAY_WRITEABLE = 0x0400


_NUMPY_TO_CNP = {
    np.dtype(np.bool_): 1,
    np.dtype(np.int8): 2,
    np.dtype(np.uint8): 3,
    np.dtype(np.int16): 4,
    np.dtype(np.uint16): 5,
    np.dtype(np.int32): 6,
    np.dtype(np.uint32): 7,
    np.dtype(np.int64): 10,
    np.dtype(np.uint64): 11,
    np.dtype(np.float32): 12,
    np.dtype(np.float64): 13,
    np.dtype(np.complex64): 15,
    np.dtype(np.complex128): 16,
    np.dtype(np.float16): 24,
}

_CNP_TO_NUMPY = {
    1: np.dtype(np.bool_),
    2: np.dtype(np.int8),
    3: np.dtype(np.uint8),
    4: np.dtype(np.int16),
    5: np.dtype(np.uint16),
    6: np.dtype(np.int32),
    7: np.dtype(np.uint32),
    8: np.dtype(np.int64),
    9: np.dtype(np.uint64),
    10: np.dtype(np.int64),
    11: np.dtype(np.uint64),
    12: np.dtype(np.float32),
    13: np.dtype(np.float64),
    14: np.dtype(np.longdouble),
    15: np.dtype(np.complex64),
    16: np.dtype(np.complex128),
    17: np.dtype(np.clongdouble),
    22: np.dtype(np.int64),
    23: np.dtype(np.int64),
    24: np.dtype(np.float16),
}

_CNP_TO_CTYPES = {
    1: ctypes.c_int8,
    2: ctypes.c_int8,
    3: ctypes.c_uint8,
    4: ctypes.c_int16,
    5: ctypes.c_uint16,
    6: ctypes.c_int32,
    7: ctypes.c_uint32,
    8: ctypes.c_int64,
    9: ctypes.c_uint64,
    10: ctypes.c_int64,
    11: ctypes.c_uint64,
    12: ctypes.c_float,
    13: ctypes.c_double,
    14: ctypes.c_longdouble,
    22: ctypes.c_int64,
    23: ctypes.c_int64,
    24: ctypes.c_uint16,
}


class _CnpComplex64(ctypes.Structure):
    _fields_ = [("real", ctypes.c_float), ("imag", ctypes.c_float)]


class _CnpComplex128(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]


class _CnpComplexLongDouble(ctypes.Structure):
    _fields_ = [
        ("real", ctypes.c_longdouble),
        ("imag", ctypes.c_longdouble),
    ]


_CNP_TO_CTYPES[15] = _CnpComplex64
_CNP_TO_CTYPES[16] = _CnpComplex128
_CNP_TO_CTYPES[17] = _CnpComplexLongDouble


class _CnpErrorState(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int),
        ("message", ctypes.c_char * 256),
        ("function", ctypes.c_char * 64),
    ]


class CnumpyError(RuntimeError):
    def __init__(self, status: int, function: str, message: str):
        self.status = status
        self.function = function
        self.message = message
        super().__init__(f"{function} failed ({status}): {message}")


def _decode_c_string(value: bytes) -> str:
    return value.split(b"\0", 1)[0].decode("utf-8", errors="strict")


class CnumpyArray:
    """One owned CnpArray reference returned by a CnumpyRuntime."""

    def __init__(self, runtime: "CnumpyRuntime", pointer: int, origin: str):
        if not pointer:
            raise ValueError("CnumpyArray requires a non-null pointer")
        self._runtime = runtime
        self._pointer = ctypes.c_void_p(pointer)
        self._origin = origin
        self._closed = False
        runtime._register(self)

    def __enter__(self) -> "CnumpyArray":
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"array from {self._origin} is already closed")

    @property
    def pointer(self) -> ctypes.c_void_p:
        self._require_open()
        return self._pointer

    @property
    def ndim(self) -> int:
        self._require_open()
        return int(self._runtime.dll.cnp_array_ndim(self._pointer))

    @property
    def size(self) -> int:
        self._require_open()
        return int(self._runtime.dll.cnp_array_size(self._pointer))

    @property
    def dtype_number(self) -> int:
        self._require_open()
        return int(self._runtime.dll.cnp_array_dtype_num(self._pointer))

    @property
    def numpy_dtype(self) -> np.dtype:
        dtype_number = self.dtype_number
        try:
            return _CNP_TO_NUMPY[dtype_number]
        except KeyError as error:
            raise TypeError(
                f"ctypes adapter does not support CNP dtype {dtype_number}"
            ) from error

    @property
    def itemsize(self) -> int:
        self._require_open()
        return int(self._runtime.dll.cnp_array_itemsize(self._pointer))

    @property
    def shape(self) -> tuple[int, ...]:
        ndim = self.ndim
        if ndim == 0:
            return ()
        pointer = self._runtime.dll.cnp_array_shape(self._pointer)
        if not pointer:
            raise self._runtime.native_error("cnp_array_shape")
        return tuple(int(pointer[index]) for index in range(ndim))

    @property
    def strides(self) -> tuple[int, ...]:
        ndim = self.ndim
        if ndim == 0:
            return ()
        pointer = self._runtime.dll.cnp_array_strides(self._pointer)
        if not pointer:
            raise self._runtime.native_error("cnp_array_strides")
        return tuple(int(pointer[index]) for index in range(ndim))

    @property
    def flags(self) -> int:
        self._require_open()
        return int(self._runtime.dll.cnp_ahk_flags(self._pointer))

    @property
    def c_contiguous(self) -> bool:
        return bool(self.flags & CNP_ARRAY_C_CONTIGUOUS)

    @property
    def f_contiguous(self) -> bool:
        return bool(self.flags & CNP_ARRAY_F_CONTIGUOUS)

    @property
    def owns_data(self) -> bool:
        return bool(self.flags & CNP_ARRAY_OWNDATA)

    @property
    def writeable(self) -> bool:
        return bool(self.flags & CNP_ARRAY_WRITEABLE)

    def _coordinates(self) -> Iterator[tuple[int, ...]]:
        shape = self.shape
        if not shape:
            yield ()
            return
        yield from product(*(range(length) for length in shape))

    def values(self) -> tuple[object, ...]:
        self._require_open()
        if self.size == 0:
            return ()

        dtype_number = self.dtype_number
        try:
            scalar_type = _CNP_TO_CTYPES[dtype_number]
        except KeyError as error:
            raise TypeError(
                f"raw value reader does not support CNP dtype {dtype_number}"
            ) from error

        result: list[object] = []
        dummy_scalar_index = (ctypes.c_int64 * 1)(0)
        for coordinates in self._coordinates():
            if coordinates:
                indices = (ctypes.c_int64 * len(coordinates))(*coordinates)
            else:
                indices = dummy_scalar_index
            address = self._runtime.dll.cnp_array_at(self._pointer, indices)
            if not address:
                raise self._runtime.native_error("cnp_array_at")
            raw_value = ctypes.cast(
                address, ctypes.POINTER(scalar_type)
            ).contents
            if dtype_number in (15, 16, 17):
                result.append(complex(raw_value.real, raw_value.imag))
            elif dtype_number == 24:
                result.append(
                    np.asarray(
                        [raw_value.value], dtype=np.uint16
                    ).view(np.float16)[0]
                )
            elif dtype_number == 1:
                result.append(bool(raw_value.value))
            else:
                result.append(raw_value.value)
        return tuple(result)

    def to_numpy(self) -> np.ndarray:
        values = self.values()
        return np.asarray(values, dtype=self.numpy_dtype).reshape(self.shape)

    def close(self) -> None:
        self._require_open()
        self._runtime.dll.cnp_clear_error()
        self._runtime.dll.cnp_array_decref(self._pointer)
        state = self._runtime.error_state()
        self._closed = True
        self._runtime._unregister(self)
        self._pointer = ctypes.c_void_p()
        if state.status != 0:
            raise CnumpyError(state.status, state.function, state.message)


class CnumpyRuntime:
    def __init__(self, dll_path: Path):
        self.dll_path = Path(dll_path).resolve(strict=True)
        self.dll = ctypes.CDLL(str(self.dll_path))
        self._configure_bindings()
        self._arrays: set[CnumpyArray] = set()
        self._closed = False
        self.dll.cnp_clear_error()
        status = int(self.dll.cnp_init())
        if status != 0:
            raise self.native_error("cnp_init", status)
        self._memory_baseline = int(self.dll.cnp_get_allocated_memory())

    def __enter__(self) -> "CnumpyRuntime":
        if self._closed:
            raise RuntimeError("runtime is already closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _configure_bindings(self) -> None:
        dll = self.dll
        dll.cnp_init.argtypes = []
        dll.cnp_init.restype = ctypes.c_int
        dll.cnp_cleanup.argtypes = []
        dll.cnp_cleanup.restype = None
        dll.cnp_clear_error.argtypes = []
        dll.cnp_clear_error.restype = None
        dll.cnp_get_error.argtypes = [ctypes.POINTER(_CnpErrorState)]
        dll.cnp_get_error.restype = ctypes.c_int
        dll.cnp_get_allocated_memory.argtypes = []
        dll.cnp_get_allocated_memory.restype = ctypes.c_size_t

        dll.cnp_array_from_data.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int,
            ctypes.c_int,
        ]
        dll.cnp_array_from_data.restype = ctypes.c_void_p
        dll.cnp_array_decref.argtypes = [ctypes.c_void_p]
        dll.cnp_array_decref.restype = None
        dll.cnp_array_ndim.argtypes = [ctypes.c_void_p]
        dll.cnp_array_ndim.restype = ctypes.c_int
        dll.cnp_array_size.argtypes = [ctypes.c_void_p]
        dll.cnp_array_size.restype = ctypes.c_int64
        dll.cnp_array_dtype_num.argtypes = [ctypes.c_void_p]
        dll.cnp_array_dtype_num.restype = ctypes.c_int
        dll.cnp_array_itemsize.argtypes = [ctypes.c_void_p]
        dll.cnp_array_itemsize.restype = ctypes.c_int
        dll.cnp_array_shape.argtypes = [ctypes.c_void_p]
        dll.cnp_array_shape.restype = ctypes.POINTER(ctypes.c_int64)
        dll.cnp_array_strides.argtypes = [ctypes.c_void_p]
        dll.cnp_array_strides.restype = ctypes.POINTER(ctypes.c_int64)
        dll.cnp_array_at.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int64),
        ]
        dll.cnp_array_at.restype = ctypes.c_void_p
        dll.cnp_ahk_flags.argtypes = [ctypes.c_void_p]
        dll.cnp_ahk_flags.restype = ctypes.c_uint32

        dll.cnp_transpose.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        dll.cnp_transpose.restype = ctypes.c_void_p
        for minimum_ndim in (1, 2, 3):
            function = getattr(dll, f"cnp_atleast_{minimum_ndim}d")
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_void_p
        array_pointer = ctypes.POINTER(ctypes.c_void_p)
        dll.cnp_broadcast_arrays_v2.argtypes = [
            ctypes.c_int,
            array_pointer,
            array_pointer,
            ctypes.c_int,
        ]
        dll.cnp_broadcast_arrays_v2.restype = ctypes.c_int
        dll.cnp_meshgrid_v2.argtypes = [
            ctypes.c_int,
            array_pointer,
            ctypes.c_bool,
            ctypes.c_bool,
            ctypes.c_bool,
            array_pointer,
            ctypes.c_int,
        ]
        dll.cnp_meshgrid_v2.restype = ctypes.c_int

    @property
    def retained_bytes(self) -> int:
        if self._closed:
            raise RuntimeError("runtime is closed")
        return int(self.dll.cnp_get_allocated_memory())

    def _register(self, array: CnumpyArray) -> None:
        if self._closed:
            raise RuntimeError("cannot register an array on a closed runtime")
        self._arrays.add(array)

    def _unregister(self, array: CnumpyArray) -> None:
        self._arrays.remove(array)

    def error_state(self) -> CnumpyError:
        state = _CnpErrorState()
        status = int(self.dll.cnp_get_error(ctypes.byref(state)))
        return CnumpyError(
            status=status,
            function=_decode_c_string(bytes(state.function)),
            message=_decode_c_string(bytes(state.message)),
        )

    def native_error(
        self, requested_function: str, status: int | None = None
    ) -> CnumpyError:
        error = self.error_state()
        resolved_status = error.status if status is None else status
        function = error.function or requested_function
        message = error.message or "native call returned no result"
        return CnumpyError(resolved_status, function, message)

    def _owned_result(self, pointer: int | None, origin: str) -> CnumpyArray:
        if not pointer:
            raise self.native_error(origin)
        return CnumpyArray(self, int(pointer), origin)

    def from_numpy(self, value: np.ndarray) -> CnumpyArray:
        if self._closed:
            raise RuntimeError("runtime is closed")
        array = np.array(value, copy=True, order="C", subok=False)
        try:
            dtype_number = _NUMPY_TO_CNP[array.dtype]
        except KeyError as error:
            raise TypeError(f"unsupported NumPy dtype: {array.dtype}") from error

        shape_storage = (ctypes.c_int64 * max(1, array.ndim))(
            *(array.shape if array.ndim else (0,))
        )
        raw = array.tobytes(order="C")
        data_storage = ctypes.create_string_buffer(raw or b"\0")
        self.dll.cnp_clear_error()
        pointer = self.dll.cnp_array_from_data(
            ctypes.cast(data_storage, ctypes.c_void_p),
            array.ndim,
            shape_storage if array.ndim else None,
            dtype_number,
            CNP_ORDER_C,
        )
        return self._owned_result(pointer, "cnp_array_from_data")

    def transpose(
        self, source: CnumpyArray, axes: Iterable[int] | None = None
    ) -> CnumpyArray:
        source._require_open()
        axes_storage = None
        if axes is not None:
            axes_tuple = tuple(int(axis) for axis in axes)
            if len(axes_tuple) != source.ndim:
                raise ValueError(
                    f"transpose requires {source.ndim} axes, got {len(axes_tuple)}"
                )
            axes_storage = (ctypes.c_int * len(axes_tuple))(*axes_tuple)
        self.dll.cnp_clear_error()
        pointer = self.dll.cnp_transpose(source.pointer, axes_storage)
        return self._owned_result(pointer, "cnp_transpose")

    def eigh(
        self, source: CnumpyArray, uplo: str = "L"
    ) -> tuple[CnumpyArray, CnumpyArray]:
        source._require_open()
        if source._runtime is not self:
            raise ValueError("eigh source must belong to this runtime")
        if uplo not in ("L", "U"):
            raise ValueError("eigh UPLO must be 'L' or 'U'")

        function_name = "cnp_linalg_eigh_v2"
        function = getattr(self.dll, function_name)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_bool,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        function.restype = ctypes.c_int
        values_pointer = ctypes.c_void_p()
        vectors_pointer = ctypes.c_void_p()
        self.dll.cnp_clear_error()
        status = int(
            function(
                source.pointer,
                uplo == "U",
                ctypes.byref(values_pointer),
                ctypes.byref(vectors_pointer),
            )
        )
        if status != 0:
            partial = (values_pointer.value, vectors_pointer.value)
            for pointer in partial:
                if pointer:
                    self.dll.cnp_array_decref(pointer)
            if any(partial):
                raise RuntimeError(
                    f"{function_name} failed with live partial results"
                )
            raise self.native_error(function_name, status)
        if not values_pointer.value or not vectors_pointer.value:
            for pointer in (values_pointer.value, vectors_pointer.value):
                if pointer:
                    self.dll.cnp_array_decref(pointer)
            raise RuntimeError(
                f"{function_name} succeeded without both results"
            )
        return (
            self._owned_result(values_pointer.value, f"{function_name}:values"),
            self._owned_result(vectors_pointer.value, f"{function_name}:vectors"),
        )

    def eigvalsh(
        self, source: CnumpyArray, uplo: str = "L"
    ) -> CnumpyArray:
        source._require_open()
        if source._runtime is not self:
            raise ValueError("eigvalsh source must belong to this runtime")
        if uplo not in ("L", "U"):
            raise ValueError("eigvalsh UPLO must be 'L' or 'U'")

        function_name = "cnp_eigvalsh_v2"
        function = getattr(self.dll, function_name)
        function.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        function.restype = ctypes.c_void_p
        self.dll.cnp_clear_error()
        pointer = function(source.pointer, uplo == "U")
        return self._owned_result(pointer, function_name)

    def atleast(self, source: CnumpyArray, minimum_ndim: int) -> CnumpyArray:
        if minimum_ndim not in (1, 2, 3):
            raise ValueError("minimum_ndim must be 1, 2, or 3")
        source._require_open()
        if source._runtime is not self:
            raise ValueError("source array must belong to this runtime")
        function_name = f"cnp_atleast_{minimum_ndim}d"
        self.dll.cnp_clear_error()
        pointer = getattr(self.dll, function_name)(source.pointer)
        return self._owned_result(pointer, function_name)

    def _multi_array_call(
        self,
        function_name: str,
        sources: Iterable[CnumpyArray],
        *arguments: object,
    ) -> tuple[CnumpyArray, ...]:
        source_tuple = tuple(sources)
        for source in source_tuple:
            source._require_open()
            if source._runtime is not self:
                raise ValueError("all source arrays must belong to this runtime")
        count = len(source_tuple)
        source_storage = (
            (ctypes.c_void_p * count)(
                *(source.pointer.value for source in source_tuple)
            )
            if count
            else None
        )
        result_storage = (ctypes.c_void_p * count)() if count else None
        function = getattr(self.dll, function_name)
        self.dll.cnp_clear_error()
        status = int(
            function(
                count,
                source_storage,
                *arguments,
                result_storage,
                count,
            )
        )
        if status != 0:
            unexpected = (
                [pointer for pointer in result_storage if pointer]
                if result_storage is not None
                else []
            )
            if unexpected:
                raise RuntimeError(
                    f"{function_name} failed with live partial results: "
                    f"{unexpected}"
                )
            raise self.native_error(function_name, status)
        if count == 0:
            return ()
        assert result_storage is not None
        missing = [index for index, pointer in enumerate(result_storage) if not pointer]
        if missing:
            raise RuntimeError(
                f"{function_name} succeeded with null result slots: {missing}"
            )
        return tuple(
            self._owned_result(pointer, function_name)
            for pointer in result_storage
        )

    def broadcast_arrays(
        self, sources: Iterable[CnumpyArray]
    ) -> tuple[CnumpyArray, ...]:
        return self._multi_array_call(
            "cnp_broadcast_arrays_v2", sources
        )

    def meshgrid(
        self,
        sources: Iterable[CnumpyArray],
        *,
        indexing: str = "xy",
        sparse: bool = False,
        copy: bool = True,
    ) -> tuple[CnumpyArray, ...]:
        if indexing not in ("xy", "ij"):
            raise ValueError("meshgrid indexing must be 'xy' or 'ij'")
        return self._multi_array_call(
            "cnp_meshgrid_v2",
            sources,
            bool(sparse),
            indexing == "ij",
            bool(copy),
        )

    def _split_function(self, function_name: str):
        function = getattr(self.dll, function_name)
        array_pointer = ctypes.POINTER(ctypes.c_void_p)
        if function_name.endswith("_sections_v2"):
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                array_pointer,
                ctypes.c_int,
            ]
        elif function_name.endswith("_indices_v2"):
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_int,
                array_pointer,
                ctypes.c_int,
            ]
        else:
            raise ValueError(f"not a split v2 function: {function_name}")
        function.restype = ctypes.c_int
        return function

    def _split_result_call(
        self,
        function_name: str,
        source: CnumpyArray,
        indices_or_sections: int | Iterable[int],
        axis: int,
    ) -> tuple[CnumpyArray, ...]:
        source._require_open()
        if source._runtime is not self:
            raise ValueError("split source must belong to this runtime")

        function = self._split_function(function_name)
        self.dll.cnp_clear_error()
        if function_name.endswith("_sections_v2"):
            sections = int(indices_or_sections)
            result_count = max(sections, 0)
            result_storage = (
                (ctypes.c_void_p * result_count)() if result_count else None
            )
            status = int(
                function(
                    source.pointer,
                    sections,
                    int(axis),
                    result_storage,
                    result_count,
                )
            )
        else:
            indices = tuple(int(index) for index in indices_or_sections)
            index_storage = (
                (ctypes.c_int64 * len(indices))(*indices) if indices else None
            )
            result_count = len(indices) + 1
            result_storage = (ctypes.c_void_p * result_count)()
            status = int(
                function(
                    source.pointer,
                    len(indices),
                    index_storage,
                    int(axis),
                    result_storage,
                    result_count,
                )
            )

        if status != 0:
            partial = (
                [pointer for pointer in result_storage if pointer]
                if result_storage is not None
                else []
            )
            for pointer in partial:
                self.dll.cnp_array_decref(pointer)
            if partial:
                raise RuntimeError(
                    f"{function_name} failed with live partial results"
                )
            raise self.native_error(function_name, status)

        if result_storage is None:
            raise RuntimeError(
                f"{function_name} succeeded without result storage"
            )
        missing = [
            index for index, pointer in enumerate(result_storage) if not pointer
        ]
        if missing:
            for pointer in result_storage:
                if pointer:
                    self.dll.cnp_array_decref(pointer)
            raise RuntimeError(
                f"{function_name} succeeded with null result slots: {missing}"
            )
        return tuple(
            self._owned_result(pointer, function_name)
            for pointer in result_storage
        )

    def split(
        self,
        source: CnumpyArray,
        indices_or_sections: int | Iterable[int],
        axis: int = 0,
    ) -> tuple[CnumpyArray, ...]:
        suffix = (
            "sections" if isinstance(indices_or_sections, (int, np.integer))
            else "indices"
        )
        return self._split_result_call(
            f"cnp_split_{suffix}_v2", source, indices_or_sections, axis
        )

    def array_split(
        self,
        source: CnumpyArray,
        indices_or_sections: int | Iterable[int],
        axis: int = 0,
    ) -> tuple[CnumpyArray, ...]:
        suffix = (
            "sections" if isinstance(indices_or_sections, (int, np.integer))
            else "indices"
        )
        return self._split_result_call(
            f"cnp_array_split_{suffix}_v2",
            source,
            indices_or_sections,
            axis,
        )

    def legacy_split(
        self,
        function_name: str,
        source: CnumpyArray,
        indices_or_sections: int | Iterable[int],
        axis: int = 0,
    ) -> tuple[CnumpyArray, ...]:
        source._require_open()
        if source._runtime is not self:
            raise ValueError("split source must belong to this runtime")
        if isinstance(indices_or_sections, (int, np.integer)):
            output_count = int(indices_or_sections)
            indices = None
        else:
            boundaries = tuple(int(index) for index in indices_or_sections)
            output_count = len(boundaries) + 1
            indices = (
                (ctypes.c_int64 * len(boundaries))(*boundaries)
                if boundaries
                else None
            )
        result_count = max(output_count, 0)
        results = (
            (ctypes.c_void_p * result_count)() if result_count else None
        )
        function = getattr(self.dll, function_name)
        array_pointer = ctypes.POINTER(ctypes.c_void_p)
        if function_name == "cnp_split":
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_int,
                array_pointer,
            ]
            arguments = (
                source.pointer, output_count, indices, int(axis), results
            )
        elif function_name in ("cnp_hsplit", "cnp_vsplit", "cnp_dsplit"):
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int64),
                array_pointer,
            ]
            arguments = (source.pointer, output_count, indices, results)
        elif function_name == "cnp_array_split":
            if indices is not None:
                raise ValueError(
                    "legacy cnp_array_split cannot represent explicit indices"
                )
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                array_pointer,
            ]
            arguments = (source.pointer, output_count, int(axis), results)
        else:
            raise ValueError(f"not a legacy split function: {function_name}")
        function.restype = ctypes.c_int
        self.dll.cnp_clear_error()
        status = int(function(*arguments))
        if status != 0:
            partial = (
                [pointer for pointer in results if pointer]
                if results is not None
                else []
            )
            for pointer in partial:
                self.dll.cnp_array_decref(pointer)
            if partial:
                raise RuntimeError(
                    f"{function_name} failed with live partial results"
                )
            raise self.native_error(function_name, status)
        if results is None:
            raise RuntimeError(
                f"{function_name} succeeded without result storage"
            )
        return tuple(
            self._owned_result(pointer, function_name) for pointer in results
        )

    def _index_result_call(
        self,
        function_name: str,
        *arguments: object,
    ) -> CnumpyArray:
        function = getattr(self.dll, function_name)
        if function_name in ("cnp_take_v2", "cnp_take_along_axis_v2"):
            function.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_bool,
            ]
        elif function_name == "cnp_compress_v2":
            function.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_bool,
            ]
        elif function_name == "cnp_delete_v2":
            function.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_bool,
            ]
        elif function_name == "cnp_insert_v2":
            function.argtypes = [
                ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_bool,
            ]
        elif function_name == "cnp_insert_array_v2":
            function.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_bool,
            ]
        else:
            raise ValueError(f"not an indexing v2 function: {function_name}")
        function.restype = ctypes.c_void_p
        self.dll.cnp_clear_error()
        pointer = function(*arguments)
        return self._owned_result(pointer, function_name)

    def take(
        self, source: CnumpyArray, indices: CnumpyArray, axis: int | None = None
    ) -> CnumpyArray:
        return self._index_result_call(
            "cnp_take_v2", source.pointer, indices.pointer,
            0 if axis is None else int(axis), axis is None,
        )

    def take_along_axis(
        self, source: CnumpyArray, indices: CnumpyArray,
        axis: int | None,
    ) -> CnumpyArray:
        return self._index_result_call(
            "cnp_take_along_axis_v2", source.pointer, indices.pointer,
            0 if axis is None else int(axis), axis is None,
        )

    def compress(
        self, condition: CnumpyArray, source: CnumpyArray,
        axis: int | None = None,
    ) -> CnumpyArray:
        return self._index_result_call(
            "cnp_compress_v2", condition.pointer, source.pointer,
            0 if axis is None else int(axis), axis is None,
        )

    def delete(
        self, source: CnumpyArray, obj: CnumpyArray,
        axis: int | None = None,
    ) -> CnumpyArray:
        return self._index_result_call(
            "cnp_delete_v2", source.pointer, obj.pointer,
            0 if axis is None else int(axis), axis is None,
        )

    def insert(
        self, source: CnumpyArray, obj: int | CnumpyArray,
        values: CnumpyArray,
        axis: int | None = None,
    ) -> CnumpyArray:
        if isinstance(obj, CnumpyArray):
            function_name = "cnp_insert_array_v2"
            object_argument: object = obj.pointer
        elif isinstance(obj, int):
            function_name = "cnp_insert_v2"
            object_argument = int(obj)
        else:
            raise TypeError("insert obj must be an int or CnumpyArray")
        return self._index_result_call(
            function_name, source.pointer, object_argument, values.pointer,
            0 if axis is None else int(axis), axis is None,
        )

    def _partition_result_call(
        self,
        function_name: str,
        source: CnumpyArray,
        kth: int | Iterable[int],
        axis: int | None,
    ) -> CnumpyArray:
        source._require_open()
        if source._runtime is not self:
            raise ValueError("source array must belong to this runtime")
        if isinstance(kth, (int, np.integer)):
            kth_values = (int(kth),)
        else:
            kth_values = tuple(kth)
            if any(
                not isinstance(value, (int, np.integer))
                for value in kth_values
            ):
                raise TypeError("partition kth values must be integers")
            kth_values = tuple(int(value) for value in kth_values)
        kth_storage = (
            (ctypes.c_int64 * len(kth_values))(*kth_values)
            if kth_values
            else None
        )
        function = getattr(self.dll, function_name)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_bool,
        ]
        function.restype = ctypes.c_void_p
        self.dll.cnp_clear_error()
        pointer = function(
            source.pointer,
            kth_storage,
            len(kth_values),
            0 if axis is None else int(axis),
            axis is None,
        )
        return self._owned_result(pointer, function_name)

    def partition(
        self,
        source: CnumpyArray,
        kth: int | Iterable[int],
        axis: int | None = -1,
    ) -> CnumpyArray:
        return self._partition_result_call(
            "cnp_partition_v2", source, kth, axis
        )

    def argpartition(
        self,
        source: CnumpyArray,
        kth: int | Iterable[int],
        axis: int | None = -1,
    ) -> CnumpyArray:
        return self._partition_result_call(
            "cnp_argpartition_v2", source, kth, axis
        )

    def searchsorted(
        self,
        source: CnumpyArray,
        values: CnumpyArray,
        side: str = "left",
        sorter: CnumpyArray | None = None,
    ) -> CnumpyArray:
        for name, array in (("source", source), ("values", values)):
            if not isinstance(array, CnumpyArray):
                raise TypeError(f"searchsorted {name} must be a CnumpyArray")
            array._require_open()
            if array._runtime is not self:
                raise ValueError(
                    f"searchsorted {name} must belong to this runtime"
                )
        if not isinstance(side, str):
            raise TypeError("searchsorted side must be a string")
        if sorter is not None:
            if not isinstance(sorter, CnumpyArray):
                raise TypeError(
                    "searchsorted sorter must be a CnumpyArray or None"
                )
            sorter._require_open()
            if sorter._runtime is not self:
                raise ValueError(
                    "searchsorted sorter must belong to this runtime"
                )

        function = self.dll.cnp_searchsorted_v2
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
        ]
        function.restype = ctypes.c_void_p
        self.dll.cnp_clear_error()
        pointer = function(
            source.pointer,
            values.pointer,
            side.encode("utf-8"),
            None if sorter is None else sorter.pointer,
        )
        return self._owned_result(pointer, "cnp_searchsorted_v2")

    def digitize(
        self,
        x: CnumpyArray,
        bins: CnumpyArray,
        right: bool = False,
    ) -> CnumpyArray:
        for name, array in (("x", x), ("bins", bins)):
            if not isinstance(array, CnumpyArray):
                raise TypeError(f"digitize {name} must be a CnumpyArray")
            array._require_open()
            if array._runtime is not self:
                raise ValueError(
                    f"digitize {name} must belong to this runtime"
                )
        if not isinstance(right, bool):
            raise TypeError("digitize right must be a bool")

        function = self.dll.cnp_digitize
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_bool,
        ]
        function.restype = ctypes.c_void_p
        self.dll.cnp_clear_error()
        pointer = function(x.pointer, bins.pointer, right)
        return self._owned_result(pointer, "cnp_digitize")

    def lexsort(
        self,
        keys: Iterable[CnumpyArray],
        axis: int = -1,
    ) -> CnumpyArray:
        try:
            key_values = tuple(keys)
        except TypeError as error:
            raise TypeError("lexsort keys must be an iterable") from error
        for index, key in enumerate(key_values):
            if not isinstance(key, CnumpyArray):
                raise TypeError(
                    f"lexsort key {index} must be a CnumpyArray"
                )
            key._require_open()
            if key._runtime is not self:
                raise ValueError(
                    f"lexsort key {index} must belong to this runtime"
                )
        try:
            axis_value = operator.index(axis)
        except TypeError as error:
            raise TypeError(
                f"integer argument expected, got {type(axis).__name__}"
            ) from error
        if axis_value < -(2**31) or axis_value > 2**31 - 1:
            raise OverflowError("Python int too large to convert to C long")

        storage = (ctypes.c_void_p * len(key_values))(
            *(key.pointer for key in key_values)
        )
        function = self.dll.cnp_lexsort_v2
        function.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
        ]
        function.restype = ctypes.c_void_p
        self.dll.cnp_clear_error()
        pointer = function(len(key_values), storage, axis_value)
        return self._owned_result(pointer, "cnp_lexsort_v2")

    def _unary_sort_result(
        self, function_name: str, source: CnumpyArray
    ) -> CnumpyArray:
        if not isinstance(source, CnumpyArray):
            raise TypeError(f"{function_name} source must be a CnumpyArray")
        source._require_open()
        if source._runtime is not self:
            raise ValueError("source array must belong to this runtime")
        function = getattr(self.dll, function_name)
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_void_p
        self.dll.cnp_clear_error()
        pointer = function(source.pointer)
        return self._owned_result(pointer, function_name)

    def msort(self, source: CnumpyArray) -> CnumpyArray:
        return self._unary_sort_result("cnp_msort", source)

    def sort_complex(self, source: CnumpyArray) -> CnumpyArray:
        return self._unary_sort_result("cnp_sort_complex", source)

    def sum(
        self,
        source: CnumpyArray,
        axis: int | None = None,
        dtype_number: int = 0,
    ) -> CnumpyArray:
        function = self.dll.cnp_sum_v2
        function.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_bool, ctypes.c_int,
        ]
        function.restype = ctypes.c_void_p
        self.dll.cnp_clear_error()
        pointer = function(
            source.pointer,
            0 if axis is None else int(axis),
            axis is None,
            int(dtype_number),
        )
        return self._owned_result(pointer, "cnp_sum_v2")

    def prod(
        self,
        source: CnumpyArray,
        axis: int | None = None,
        dtype_number: int = 0,
    ) -> CnumpyArray:
        function = self.dll.cnp_prod_v2
        function.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_bool, ctypes.c_int,
        ]
        function.restype = ctypes.c_void_p
        self.dll.cnp_clear_error()
        pointer = function(
            source.pointer,
            0 if axis is None else int(axis),
            axis is None,
            int(dtype_number),
        )
        return self._owned_result(pointer, "cnp_prod_v2")

    def cumsum(
        self,
        source: CnumpyArray,
        axis: int | None = None,
        dtype_number: int = 0,
    ) -> CnumpyArray:
        function = self.dll.cnp_cumsum_v2
        function.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_bool, ctypes.c_int,
        ]
        function.restype = ctypes.c_void_p
        self.dll.cnp_clear_error()
        pointer = function(
            source.pointer,
            0 if axis is None else int(axis),
            axis is None,
            int(dtype_number),
        )
        return self._owned_result(pointer, "cnp_cumsum_v2")

    def cumprod(
        self,
        source: CnumpyArray,
        axis: int | None = None,
        dtype_number: int = 0,
    ) -> CnumpyArray:
        function = self.dll.cnp_cumprod_v2
        function.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_bool, ctypes.c_int,
        ]
        function.restype = ctypes.c_void_p
        self.dll.cnp_clear_error()
        pointer = function(
            source.pointer,
            0 if axis is None else int(axis),
            axis is None,
            int(dtype_number),
        )
        return self._owned_result(pointer, "cnp_cumprod_v2")

    def mean(
        self,
        source: CnumpyArray,
        axis: int | None = None,
        dtype_number: int = 0,
    ) -> CnumpyArray:
        function = self.dll.cnp_mean_v2
        function.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_bool, ctypes.c_int,
        ]
        function.restype = ctypes.c_void_p
        self.dll.cnp_clear_error()
        pointer = function(
            source.pointer,
            0 if axis is None else int(axis),
            axis is None,
            int(dtype_number),
        )
        return self._owned_result(pointer, "cnp_mean_v2")

    def average(
        self,
        source: CnumpyArray,
        axis: int | None,
        weights: CnumpyArray | None = None,
    ) -> CnumpyArray:
        source._require_open()
        if source._runtime is not self:
            raise ValueError("source array must belong to this runtime")
        if weights is not None:
            weights._require_open()
            if weights._runtime is not self:
                raise ValueError("weights array must belong to this runtime")
        function = self.dll.cnp_average_v2
        function.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_bool, ctypes.c_void_p,
        ]
        function.restype = ctypes.c_void_p
        self.dll.cnp_clear_error()
        pointer = function(
            source.pointer,
            0 if axis is None else int(axis),
            axis is None,
            None if weights is None else weights.pointer,
        )
        return self._owned_result(pointer, "cnp_average_v2")

    def variance(
        self,
        source: CnumpyArray,
        axis: int | None = None,
        ddof: int = 0,
        dtype_number: int = 0,
    ) -> CnumpyArray:
        function = self.dll.cnp_var_v2
        function.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_bool,
            ctypes.c_int, ctypes.c_int,
        ]
        function.restype = ctypes.c_void_p
        self.dll.cnp_clear_error()
        pointer = function(
            source.pointer,
            0 if axis is None else int(axis),
            axis is None,
            int(ddof),
            int(dtype_number),
        )
        return self._owned_result(pointer, "cnp_var_v2")

    def std(
        self,
        source: CnumpyArray,
        axis: int | None = None,
        ddof: int = 0,
        dtype_number: int = 0,
    ) -> CnumpyArray:
        function = self.dll.cnp_std_v2
        function.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_bool,
            ctypes.c_int, ctypes.c_int,
        ]
        function.restype = ctypes.c_void_p
        self.dll.cnp_clear_error()
        pointer = function(
            source.pointer,
            0 if axis is None else int(axis),
            axis is None,
            int(ddof),
            int(dtype_number),
        )
        return self._owned_result(pointer, "cnp_std_v2")

    def maximum(
        self, source: CnumpyArray, axis: int | None = None
    ) -> CnumpyArray:
        return self._axis_reduction("cnp_max_v2", source, axis)

    def minimum(
        self, source: CnumpyArray, axis: int | None = None
    ) -> CnumpyArray:
        return self._axis_reduction("cnp_min_v2", source, axis)

    def any(
        self, source: CnumpyArray, axis: int | None = None
    ) -> CnumpyArray:
        return self._axis_reduction("cnp_any_v2", source, axis)

    def all(
        self, source: CnumpyArray, axis: int | None = None
    ) -> CnumpyArray:
        return self._axis_reduction("cnp_all_v2", source, axis)

    def ptp(
        self, source: CnumpyArray, axis: int | None = None
    ) -> CnumpyArray:
        return self._axis_reduction("cnp_ptp_v2", source, axis)

    def argmax(
        self, source: CnumpyArray, axis: int | None = None
    ) -> CnumpyArray:
        return self._axis_reduction("cnp_argmax_v2", source, axis)

    def argmin(
        self, source: CnumpyArray, axis: int | None = None
    ) -> CnumpyArray:
        return self._axis_reduction("cnp_argmin_v2", source, axis)

    def _axis_reduction(
        self, function_name: str, source: CnumpyArray, axis: int | None
    ) -> CnumpyArray:
        function = getattr(self.dll, function_name)
        function.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_bool]
        function.restype = ctypes.c_void_p
        self.dll.cnp_clear_error()
        pointer = function(
            source.pointer,
            0 if axis is None else int(axis),
            axis is None,
        )
        return self._owned_result(pointer, function_name)

    def nansum(
        self, source: CnumpyArray, axis: int | None = None,
        dtype_number: int = 0,
    ) -> CnumpyArray:
        return self._nan_numeric_reduction(
            "cnp_nansum_v2", source, axis, dtype_number
        )

    def nanprod(
        self, source: CnumpyArray, axis: int | None = None,
        dtype_number: int = 0,
    ) -> CnumpyArray:
        return self._nan_numeric_reduction(
            "cnp_nanprod_v2", source, axis, dtype_number
        )

    def nanmean(
        self, source: CnumpyArray, axis: int | None = None,
        dtype_number: int = 0,
    ) -> CnumpyArray:
        return self._nan_numeric_reduction(
            "cnp_nanmean_v2", source, axis, dtype_number
        )

    def nanvar(
        self, source: CnumpyArray, axis: int | None = None,
        ddof: int = 0, dtype_number: int = 0,
    ) -> CnumpyArray:
        return self._nan_deviation_reduction(
            "cnp_nanvar_v2", source, axis, ddof, dtype_number
        )

    def nanstd(
        self, source: CnumpyArray, axis: int | None = None,
        ddof: int = 0, dtype_number: int = 0,
    ) -> CnumpyArray:
        return self._nan_deviation_reduction(
            "cnp_nanstd_v2", source, axis, ddof, dtype_number
        )

    def nanmax(
        self, source: CnumpyArray, axis: int | None = None
    ) -> CnumpyArray:
        return self._axis_reduction("cnp_nanmax_v2", source, axis)

    def nanmin(
        self, source: CnumpyArray, axis: int | None = None
    ) -> CnumpyArray:
        return self._axis_reduction("cnp_nanmin_v2", source, axis)

    def nancumsum(
        self, source: CnumpyArray, axis: int | None = None,
        dtype_number: int = 0,
    ) -> CnumpyArray:
        return self._nan_numeric_reduction(
            "cnp_nancumsum_v2", source, axis, dtype_number
        )

    def nancumprod(
        self, source: CnumpyArray, axis: int | None = None,
        dtype_number: int = 0,
    ) -> CnumpyArray:
        return self._nan_numeric_reduction(
            "cnp_nancumprod_v2", source, axis, dtype_number
        )

    def _nan_numeric_reduction(
        self, function_name: str, source: CnumpyArray,
        axis: int | None, dtype_number: int,
    ) -> CnumpyArray:
        function = getattr(self.dll, function_name)
        function.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_bool, ctypes.c_int,
        ]
        function.restype = ctypes.c_void_p
        self.dll.cnp_clear_error()
        pointer = function(
            source.pointer, 0 if axis is None else int(axis),
            axis is None, int(dtype_number),
        )
        return self._owned_result(pointer, function_name)

    def _nan_deviation_reduction(
        self, function_name: str, source: CnumpyArray,
        axis: int | None, ddof: int, dtype_number: int,
    ) -> CnumpyArray:
        function = getattr(self.dll, function_name)
        function.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_bool,
            ctypes.c_int, ctypes.c_int,
        ]
        function.restype = ctypes.c_void_p
        self.dll.cnp_clear_error()
        pointer = function(
            source.pointer, 0 if axis is None else int(axis),
            axis is None, int(ddof), int(dtype_number),
        )
        return self._owned_result(pointer, function_name)

    def nanargmax(
        self, source: CnumpyArray, axis: int | None = None
    ) -> CnumpyArray:
        return self._axis_reduction("cnp_nanargmax_v2", source, axis)

    def nanargmin(
        self, source: CnumpyArray, axis: int | None = None
    ) -> CnumpyArray:
        return self._axis_reduction("cnp_nanargmin_v2", source, axis)

    def median(
        self, source: CnumpyArray, axis: int | None = None
    ) -> CnumpyArray:
        return self._axis_reduction("cnp_median_v2", source, axis)

    def nanmedian(
        self, source: CnumpyArray, axis: int | None = None
    ) -> CnumpyArray:
        return self._axis_reduction("cnp_nanmedian_v2", source, axis)

    def percentile(
        self, source: CnumpyArray, q: float,
        axis: int | None = None,
    ) -> CnumpyArray:
        return self._percentile_reduction(
            "cnp_percentile_v2", source, q, axis
        )

    def nanpercentile(
        self, source: CnumpyArray, q: float,
        axis: int | None = None,
    ) -> CnumpyArray:
        return self._percentile_reduction(
            "cnp_nanpercentile_v2", source, q, axis
        )

    def quantile(
        self, source: CnumpyArray, q: float,
        axis: int | None = None,
    ) -> CnumpyArray:
        return self._percentile_reduction(
            "cnp_quantile_v2", source, q, axis
        )

    def nanquantile(
        self, source: CnumpyArray, q: float,
        axis: int | None = None,
    ) -> CnumpyArray:
        return self._percentile_reduction(
            "cnp_nanquantile_v2", source, q, axis
        )

    def _percentile_reduction(
        self, function_name: str, source: CnumpyArray, q: float,
        axis: int | None,
    ) -> CnumpyArray:
        function = getattr(self.dll, function_name)
        function.argtypes = [
            ctypes.c_void_p, ctypes.c_double,
            ctypes.c_int, ctypes.c_bool,
        ]
        function.restype = ctypes.c_void_p
        self.dll.cnp_clear_error()
        pointer = function(
            source.pointer, float(q),
            0 if axis is None else int(axis), axis is None,
        )
        return self._owned_result(pointer, function_name)

    def close(self) -> None:
        if self._closed:
            raise RuntimeError("runtime is already closed")
        live_origins = sorted(array._origin for array in self._arrays)
        retained = int(self.dll.cnp_get_allocated_memory())
        self.dll.cnp_cleanup()
        self._closed = True
        if live_origins:
            raise RuntimeError(f"runtime closed with live arrays: {live_origins}")
        if retained != self._memory_baseline:
            raise RuntimeError(
                "runtime retained native memory: "
                f"baseline={self._memory_baseline}, current={retained}"
            )
