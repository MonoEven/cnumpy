#Requires AutoHotkey v2.0

#Include <stdlib\ahktest>
#Include numpy.ahk

NumpyDllPath() {
    SplitPath A_LineFile, , &ahkDir
    return ahkDir "\..\build\x64\Release\cnumpy_ahk.dll"
}

; Explicitly load DLL via LoadLibraryW to ensure correct module is used.
; AHK's DllCall auto-load may resolve to a different module in some environments.
NumpyLoadDll() {
    dllPath := NumpyDllPath()
    DllCall("kernel32\LoadLibraryW", "Str", dllPath, "Ptr")
    return dllPath
}

NumpyAssertArrayNear(expected, actual, tolerance := 1e-10) {
    AhkTest.AssertEqual(expected.Length, actual.Length, "Array length mismatch")
    for index, expectedValue in expected
        AhkTest.AssertTrue(Abs(expectedValue - actual[index]) <= tolerance,
            "Index " index ": expected " expectedValue " got " actual[index])
}

NumpyAssertArrayEqual(expected, actual) {
    AhkTest.AssertEqual(expected.Length, actual.Length, "Array length mismatch")
    for index, value in expected
        AhkTest.AssertEqual(value, actual[index],
            "Array item " index " mismatch: expected <" value
            "> actual <" actual[index] ">")
}

NumpyCreateDoubleHandle(dllPath, values) {
    shape := Buffer(8, 0)
    NumPut("Int64", values.Length, shape, 0)
    data := Buffer(values.Length * 8, 0)
    for index, value in values
        NumPut("Double", value, data, (index - 1) * 8)
    handle := DllCall(dllPath "\cnp_ahk_from_doubles",
        "Int", 1, "Ptr", shape, "Ptr", data,
        "Int64", values.Length, "Ptr")
    AhkTest.AssertTrue(handle != 0, "float64 test handle must not be null")
    return handle
}

NumpyDoubleFromBits(high, low := 0) {
    bytes := Buffer(8, 0)
    NumPut("UInt", low, bytes, 0)
    NumPut("UInt", high, bytes, 4)
    return NumGet(bytes, 0, "Double")
}

NumpyFloatFromBits(bits) {
    bytes := Buffer(4, 0)
    NumPut("UInt", bits, bytes, 0)
    return NumGet(bytes, 0, "Float")
}

NumpySumCallbackLine(values) {
    total := 0.0
    for value in values
        total += value
    return total
}

class NumpySequenceCallback {
    Values := []
    Index := 0

    __New(values) {
        this.Values := values
    }

    Call(arguments*) {
        this.Index += 1
        return this.Values[this.Index]
    }
}

class NumpyPiecewiseOrderCallback {
    Calls := []

    Call(value) {
        this.Calls.Push(value)
        return 10 * this.Calls.Length + value
    }
}

class NumpyFailingNumericCallback {
    Call(arguments*) {
        throw Error("functional callback boom")
    }
}

class NumpyNegativeCountCallback {
    Calls := 0

    Call(arguments*) {
        this.Calls += 1
        throw Error("negative FromIter count reached the callback")
    }
}

class NumpyBulkCallbackProbe {
    static States := Map()
    static NextToken := 1

    static Register(state) {
        token := NumpyBulkCallbackProbe.NextToken
        NumpyBulkCallbackProbe.NextToken += 1
        NumpyBulkCallbackProbe.States[token] := state
        return token
    }

    static Release(token) {
        if NumpyBulkCallbackProbe.States.Has(token)
            NumpyBulkCallbackProbe.States.Delete(token)
    }
}

NumpyBulkUnaryProbe(
    valuesPointer, valueCount, resultsPointer, resultCapacity,
    producedPointer, userdata
) {
    state := NumpyBulkCallbackProbe.States[userdata]
    state.Calls += 1
    if state.Fail {
        NumPut("Int64", 0, producedPointer)
        return -13
    }
    state.Elements += valueCount
    loop valueCount {
        value := NumGet(valuesPointer, (A_Index - 1) * 8, "Double")
        NumPut("Double", value * 1.5 + 0.25,
            resultsPointer, (A_Index - 1) * 8)
    }
    NumPut("Int64", valueCount, producedPointer)
    return 0
}

NumpyBulkIteratorProbe(
    resultsPointer, resultCapacity, producedPointer, userdata
) {
    state := NumpyBulkCallbackProbe.States[userdata]
    state.Calls += 1
    remaining := state.Values.Length - state.Index
    count := Min(resultCapacity, remaining)
    loop count {
        state.Index += 1
        NumPut("Double", state.Values[state.Index],
            resultsPointer, (A_Index - 1) * 8)
    }
    NumPut("Int64", count, producedPointer)
    return 0
}

class NumpyFoundationTest
{
    static TestInitAndVersion() {
        dllPath := NumpyLoadDll()
        AhkTest.AssertTrue(FileExist(dllPath) != "", "DLL not found: " dllPath)

        handle := DllCall("kernel32\LoadLibraryW", "Str", dllPath, "Ptr")
        AhkTest.AssertTrue(handle != 0, "Failed to load DLL")

        status := DllCall(dllPath "\cnp_ahk_init", "Int")
        AhkTest.AssertEqual(0, status, "cnp_ahk_init should return 0")

        verPtr := DllCall(dllPath "\cnp_ahk_version", "Ptr")
        AhkTest.AssertTrue(verPtr != 0, "Version pointer should not be null")
        ver := StrGet(verPtr, "UTF-8")
        AhkTest.AssertTrue(StrLen(ver) > 0, "Version string should not be empty")

        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
        DllCall("kernel32\FreeLibrary", "Ptr", handle)
    }

    static TestNativeFunctionPointerCache() {
        first := Numpy.Proc("cnp_ahk_size")
        second := Numpy.Proc("cnp_ahk_size")
        AhkTest.AssertTrue(first != 0, "resolved function pointer must not be null")
        AhkTest.AssertEqual(first, second,
            "native function pointers must be cached by export name")

        rejected := false
        try Numpy.Proc("cnp_export_that_does_not_exist")
        catch Error as err {
            rejected := InStr(err.Message, "cnp_export_that_does_not_exist") > 0
        }
        AhkTest.AssertTrue(rejected,
            "missing native exports must fail with the requested name")
    }

    static TestNativeFunctionPointerCacheInvalidatesOnHandleChange() {
        originalHandle := Numpy.EnsureDllLoaded()
        Numpy.Proc("cnp_ahk_size")
        kernelHandle := DllCall("kernel32\GetModuleHandleW",
            "Str", "kernel32.dll", "Ptr")
        AhkTest.AssertTrue(kernelHandle != 0,
            "kernel32 module handle must be available")

        rejected := false
        try {
            Numpy.DllHandle := kernelHandle
            try Numpy.Proc("cnp_ahk_size")
            catch Error as err {
                rejected := InStr(err.Message, "cnp_ahk_size") > 0
            }
        } finally {
            Numpy.DllHandle := originalHandle
            Numpy.ProcCache := Map()
        }
        AhkTest.AssertTrue(rejected,
            "changing the DLL handle must invalidate cached export addresses")
    }

    static TestStaticAddPointerCacheTracksDllHandle() {
        left := Numpy.Array([1.0, 2.0])
        right := Numpy.Array([3.0, 4.0])
        destination := Numpy.Zeros([2])
        Numpy.Add(left, right, destination)
        originalHandle := Numpy.DllHandle
        kernelHandle := DllCall("kernel32\GetModuleHandleW",
            "Str", "kernel32.dll", "Ptr")

        rejected := false
        try {
            Numpy.DllHandle := kernelHandle
            try Numpy.Add(left, right, destination)
            catch Error as err {
                rejected := InStr(err.Message, "cnp_ahk_add") > 0
            }
        } finally {
            Numpy.DllHandle := originalHandle
            Numpy.ProcCache := Map()
        }
        AhkTest.AssertTrue(rejected,
            "Numpy.Add must not reuse pointers resolved from a different DLL handle")
    }

    static TestCreateFromDoubles() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")

        ; Create 2x3 array
        shapeBuf := Buffer(16, 0)
        NumPut("Int64", 2, shapeBuf, 0)
        NumPut("Int64", 3, shapeBuf, 8)

        dataBuf := Buffer(48, 0)
        loop 6
            NumPut("Double", A_Index * 1.0, dataBuf, (A_Index - 1) * 8)

        h := DllCall(dllPath "\cnp_ahk_from_doubles",
            "Int", 2, "Ptr", shapeBuf, "Ptr", dataBuf, "Int64", 6, "Ptr")
        AhkTest.AssertTrue(h != 0, "Handle should not be null")

        ndim := DllCall(dllPath "\cnp_ahk_ndim", "Ptr", h, "Int")
        AhkTest.AssertEqual(2, ndim)

        size := DllCall(dllPath "\cnp_ahk_size", "Ptr", h, "Int64")
        AhkTest.AssertEqual(6, size)

        ; Read back data
        outBuf := Buffer(48, 0)
        count := DllCall(dllPath "\cnp_ahk_get_doubles", "Ptr", h, "Ptr", outBuf, "Int64", 6, "Int64")
        AhkTest.AssertEqual(6, count)

        loop 6 {
            val := NumGet(outBuf, (A_Index - 1) * 8, "Double")
            AhkTest.AssertTrue(Abs(val - A_Index * 1.0) < 1e-10, "Element " A_Index " mismatch")
        }

        DllCall(dllPath "\cnp_ahk_free", "Ptr", h, "Int")
        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestZerosOnesFull() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")

        shapeBuf := Buffer(8, 0)
        NumPut("Int64", 5, shapeBuf, 0)

        ; Zeros
        h := DllCall(dllPath "\cnp_ahk_zeros", "Int", 1, "Ptr", shapeBuf, "Ptr")
        AhkTest.AssertTrue(h != 0)
        val := DllCall(dllPath "\cnp_ahk_get_item", "Ptr", h, "Int64", 0, "Double")
        AhkTest.AssertEqual(0.0, val)
        DllCall(dllPath "\cnp_ahk_free", "Ptr", h, "Int")

        ; Ones
        h := DllCall(dllPath "\cnp_ahk_ones", "Int", 1, "Ptr", shapeBuf, "Ptr")
        AhkTest.AssertTrue(h != 0)
        val := DllCall(dllPath "\cnp_ahk_get_item", "Ptr", h, "Int64", 2, "Double")
        AhkTest.AssertEqual(1.0, val)
        DllCall(dllPath "\cnp_ahk_free", "Ptr", h, "Int")

        ; Full
        h := DllCall(dllPath "\cnp_ahk_full", "Int", 1, "Ptr", shapeBuf, "Double", 3.14, "Ptr")
        AhkTest.AssertTrue(h != 0)
        val := DllCall(dllPath "\cnp_ahk_get_item", "Ptr", h, "Int64", 4, "Double")
        AhkTest.AssertTrue(Abs(val - 3.14) < 1e-10)
        DllCall(dllPath "\cnp_ahk_free", "Ptr", h, "Int")

        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestDemandZeroAllocationBoundaryAndRelease() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")
        memoryBefore := DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64")

        for testCase in [[8191, false], [8192, true], [1000000, true]] {
            count := testCase[1]
            expectVirtual := testCase[2]
            shapeBuf := Buffer(8, 0)
            NumPut("Int64", count, shapeBuf, 0)
            handle := DllCall(dllPath "\cnp_ahk_zeros",
                "Int", 1, "Ptr", shapeBuf, "Ptr")
            AhkTest.AssertTrue(handle != 0,
                "zeros handle should not be null for " count " elements")

            flags := DllCall(dllPath "\cnp_ahk_flags", "Ptr", handle, "UInt")
            hasVirtualOwner := (flags & 0x0800) != 0
            AhkTest.AssertEqual(expectVirtual ? 1 : 0, hasVirtualOwner ? 1 : 0,
                "virtual owner flag mismatch for " count " elements")

            for index in [0, Floor(count / 2), count - 1]
                AhkTest.AssertEqual(0.0,
                    DllCall(dllPath "\cnp_ahk_get_item",
                        "Ptr", handle, "Int64", index, "Double"),
                    "demand-zero value mismatch at " index)

            AhkTest.AssertEqual(0,
                DllCall(dllPath "\cnp_ahk_free", "Ptr", handle, "Int"),
                "zero array release should report success")
            AhkTest.AssertEqual(memoryBefore,
                DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64"),
                "zero array release must restore tracked memory")
        }

        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestArangeLinspace() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")

        ; arange(0, 5, 1) -> [0,1,2,3,4]
        h := DllCall(dllPath "\cnp_ahk_arange", "Double", 0.0, "Double", 5.0, "Double", 1.0, "Ptr")
        AhkTest.AssertTrue(h != 0)
        size := DllCall(dllPath "\cnp_ahk_size", "Ptr", h, "Int64")
        AhkTest.AssertEqual(5, size)
        val := DllCall(dllPath "\cnp_ahk_get_item", "Ptr", h, "Int64", 3, "Double")
        AhkTest.AssertEqual(3.0, val)
        DllCall(dllPath "\cnp_ahk_free", "Ptr", h, "Int")

        ; linspace(0, 1, 5) -> [0, 0.25, 0.5, 0.75, 1.0]
        h := DllCall(dllPath "\cnp_ahk_linspace", "Double", 0.0, "Double", 1.0, "Int64", 5, "Ptr")
        AhkTest.AssertTrue(h != 0)
        size := DllCall(dllPath "\cnp_ahk_size", "Ptr", h, "Int64")
        AhkTest.AssertEqual(5, size)
        val := DllCall(dllPath "\cnp_ahk_get_item", "Ptr", h, "Int64", 2, "Double")
        AhkTest.AssertTrue(Abs(val - 0.5) < 1e-10)
        DllCall(dllPath "\cnp_ahk_free", "Ptr", h, "Int")

        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestEyeIdentity() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")

        ; eye(3)
        h := DllCall(dllPath "\cnp_ahk_eye", "Int64", 3, "Int64", 0, "Int", 0, "Ptr")
        AhkTest.AssertTrue(h != 0)
        size := DllCall(dllPath "\cnp_ahk_size", "Ptr", h, "Int64")
        AhkTest.AssertEqual(9, size)
        v00 := DllCall(dllPath "\cnp_ahk_get_item", "Ptr", h, "Int64", 0, "Double")
        v01 := DllCall(dllPath "\cnp_ahk_get_item", "Ptr", h, "Int64", 1, "Double")
        v11 := DllCall(dllPath "\cnp_ahk_get_item", "Ptr", h, "Int64", 4, "Double")
        AhkTest.AssertEqual(1.0, v00)
        AhkTest.AssertEqual(0.0, v01)
        AhkTest.AssertEqual(1.0, v11)
        DllCall(dllPath "\cnp_ahk_free", "Ptr", h, "Int")

        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestBinaryOps() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")

        shapeBuf := Buffer(8, 0)
        NumPut("Int64", 3, shapeBuf, 0)

        ; a = [1, 2, 3]
        dataA := Buffer(24, 0)
        NumPut("Double", 1.0, dataA, 0)
        NumPut("Double", 2.0, dataA, 8)
        NumPut("Double", 3.0, dataA, 16)
        ha := DllCall(dllPath "\cnp_ahk_from_doubles", "Int", 1, "Ptr", shapeBuf, "Ptr", dataA, "Int64", 3, "Ptr")

        ; b = [4, 5, 6]
        dataB := Buffer(24, 0)
        NumPut("Double", 4.0, dataB, 0)
        NumPut("Double", 5.0, dataB, 8)
        NumPut("Double", 6.0, dataB, 16)
        hb := DllCall(dllPath "\cnp_ahk_from_doubles", "Int", 1, "Ptr", shapeBuf, "Ptr", dataB, "Int64", 3, "Ptr")

        ; add -> [5, 7, 9]
        hr := DllCall(dllPath "\cnp_ahk_add", "Ptr", ha, "Ptr", hb, "Ptr")
        AhkTest.AssertTrue(hr != 0)
        outBuf := Buffer(24, 0)
        DllCall(dllPath "\cnp_ahk_get_doubles", "Ptr", hr, "Ptr", outBuf, "Int64", 3, "Int64")
        AhkTest.AssertEqual(5.0, NumGet(outBuf, 0, "Double"))
        AhkTest.AssertEqual(7.0, NumGet(outBuf, 8, "Double"))
        AhkTest.AssertEqual(9.0, NumGet(outBuf, 16, "Double"))
        DllCall(dllPath "\cnp_ahk_free", "Ptr", hr, "Int")

        ; multiply -> [4, 10, 18]
        hr := DllCall(dllPath "\cnp_ahk_multiply", "Ptr", ha, "Ptr", hb, "Ptr")
        AhkTest.AssertTrue(hr != 0)
        DllCall(dllPath "\cnp_ahk_get_doubles", "Ptr", hr, "Ptr", outBuf, "Int64", 3, "Int64")
        AhkTest.AssertEqual(4.0, NumGet(outBuf, 0, "Double"))
        AhkTest.AssertEqual(10.0, NumGet(outBuf, 8, "Double"))
        AhkTest.AssertEqual(18.0, NumGet(outBuf, 16, "Double"))
        DllCall(dllPath "\cnp_ahk_free", "Ptr", hr, "Int")

        ; subtract -> [-3, -3, -3]
        hr := DllCall(dllPath "\cnp_ahk_subtract", "Ptr", ha, "Ptr", hb, "Ptr")
        AhkTest.AssertTrue(hr != 0)
        DllCall(dllPath "\cnp_ahk_get_doubles", "Ptr", hr, "Ptr", outBuf, "Int64", 3, "Int64")
        AhkTest.AssertEqual(-3.0, NumGet(outBuf, 0, "Double"))
        DllCall(dllPath "\cnp_ahk_free", "Ptr", hr, "Int")

        DllCall(dllPath "\cnp_ahk_free", "Ptr", ha, "Int")
        DllCall(dllPath "\cnp_ahk_free", "Ptr", hb, "Int")
        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestSimdArangeAndLinspaceEdges() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")

        descending := DllCall(dllPath "\cnp_arange",
            "Double", 5.0, "Double", -1.0, "Double", -1.5,
            "Int", Numpy.DT_FLOAT64, "Ptr")
        AhkTest.AssertTrue(descending != 0,
            "descending float64 arange must succeed")
        values := Buffer(4 * 8, 0)
        count := DllCall(dllPath "\cnp_ahk_get_doubles",
            "Ptr", descending, "Ptr", values, "Int64", 4, "Int64")
        AhkTest.AssertEqual(4, count)
        for index, expected in [5.0, 3.5, 2.0, 0.5]
            AhkTest.AssertTrue(
                Abs(NumGet(values, (index - 1) * 8, "Double") - expected) < 1e-12,
                "descending arange value " index " mismatch")
        DllCall(dllPath "\cnp_ahk_free", "Ptr", descending, "Int")

        noEndpoint := DllCall(dllPath "\cnp_linspace",
            "Double", -2.0, "Double", 3.0, "Int64", 8,
            "Int", false, "Int", Numpy.DT_FLOAT64, "Ptr")
        AhkTest.AssertTrue(noEndpoint != 0,
            "float64 linspace without endpoint must succeed")
        values := Buffer(8 * 8, 0)
        count := DllCall(dllPath "\cnp_ahk_get_doubles",
            "Ptr", noEndpoint, "Ptr", values, "Int64", 8, "Int64")
        AhkTest.AssertEqual(8, count)
        loop 8 {
            expected := -2.0 + (A_Index - 1) * 0.625
            AhkTest.AssertTrue(
                Abs(NumGet(values, (A_Index - 1) * 8, "Double") - expected) < 1e-12,
                "linspace value " A_Index " mismatch")
        }
        DllCall(dllPath "\cnp_ahk_free", "Ptr", noEndpoint, "Int")

        exactEndpoint := DllCall(dllPath "\cnp_linspace",
            "Double", 0.0, "Double", 1.0, "Int64", 10,
            "Int", true, "Int", Numpy.DT_FLOAT64, "Ptr")
        last := DllCall(dllPath "\cnp_ahk_get_item",
            "Ptr", exactEndpoint, "Int64", 9, "Double")
        AhkTest.AssertEqual(1.0, last,
            "endpoint linspace must preserve the exact stop value")
        DllCall(dllPath "\cnp_ahk_free", "Ptr", exactEndpoint, "Int")

        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestBitwiseNotCoreExport() {
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.IntArray([0, 1, -2])
        handle := DllCall(Numpy.RequireDllPath() "\cnp_bitwise_not",
            "Ptr", source.Handle, "Ptr")
        result := Numpy.WrapHandle(handle, "cnp_bitwise_not")

        NumpyAssertArrayNear([-1, -2, 1], result.ToArray())

        result := 0
        source := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "bitwise-not export test must release all native memory")
    }

    static TestUnaryOps() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")

        shapeBuf := Buffer(8, 0)
        NumPut("Int64", 3, shapeBuf, 0)

        ; a = [1, 4, 9]
        dataA := Buffer(24, 0)
        NumPut("Double", 1.0, dataA, 0)
        NumPut("Double", 4.0, dataA, 8)
        NumPut("Double", 9.0, dataA, 16)
        ha := DllCall(dllPath "\cnp_ahk_from_doubles", "Int", 1, "Ptr", shapeBuf, "Ptr", dataA, "Int64", 3, "Ptr")

        ; sqrt -> [1, 2, 3]
        hr := DllCall(dllPath "\cnp_ahk_sqrt", "Ptr", ha, "Ptr")
        AhkTest.AssertTrue(hr != 0)
        outBuf := Buffer(24, 0)
        DllCall(dllPath "\cnp_ahk_get_doubles", "Ptr", hr, "Ptr", outBuf, "Int64", 3, "Int64")
        AhkTest.AssertTrue(Abs(NumGet(outBuf, 0, "Double") - 1.0) < 1e-10)
        AhkTest.AssertTrue(Abs(NumGet(outBuf, 8, "Double") - 2.0) < 1e-10)
        AhkTest.AssertTrue(Abs(NumGet(outBuf, 16, "Double") - 3.0) < 1e-10)
        DllCall(dllPath "\cnp_ahk_free", "Ptr", hr, "Int")

        ; negative -> [-1, -4, -9]
        hr := DllCall(dllPath "\cnp_ahk_negative", "Ptr", ha, "Ptr")
        AhkTest.AssertTrue(hr != 0)
        DllCall(dllPath "\cnp_ahk_get_doubles", "Ptr", hr, "Ptr", outBuf, "Int64", 3, "Int64")
        AhkTest.AssertEqual(-1.0, NumGet(outBuf, 0, "Double"))
        AhkTest.AssertEqual(-4.0, NumGet(outBuf, 8, "Double"))
        DllCall(dllPath "\cnp_ahk_free", "Ptr", hr, "Int")

        DllCall(dllPath "\cnp_ahk_free", "Ptr", ha, "Int")
        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestReductions() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")

        shapeBuf := Buffer(8, 0)
        NumPut("Int64", 4, shapeBuf, 0)

        ; a = [1, 2, 3, 4]
        dataA := Buffer(32, 0)
        NumPut("Double", 1.0, dataA, 0)
        NumPut("Double", 2.0, dataA, 8)
        NumPut("Double", 3.0, dataA, 16)
        NumPut("Double", 4.0, dataA, 24)
        ha := DllCall(dllPath "\cnp_ahk_from_doubles", "Int", 1, "Ptr", shapeBuf, "Ptr", dataA, "Int64", 4, "Ptr")

        ; sum = 10
        val := DllCall(dllPath "\cnp_ahk_sum", "Ptr", ha, "Int", -1, "Double")
        AhkTest.AssertEqual(10.0, val)

        ; mean = 2.5
        val := DllCall(dllPath "\cnp_ahk_mean", "Ptr", ha, "Int", -1, "Double")
        AhkTest.AssertEqual(2.5, val)

        ; max = 4
        val := DllCall(dllPath "\cnp_ahk_max", "Ptr", ha, "Int", -1, "Double")
        AhkTest.AssertEqual(4.0, val)

        ; min = 1
        val := DllCall(dllPath "\cnp_ahk_min", "Ptr", ha, "Int", -1, "Double")
        AhkTest.AssertEqual(1.0, val)

        ; argmax = 3
        idx := DllCall(dllPath "\cnp_ahk_argmax", "Ptr", ha, "Int", -1, "Int64")
        AhkTest.AssertEqual(3, idx)

        ; argmin = 0
        idx := DllCall(dllPath "\cnp_ahk_argmin", "Ptr", ha, "Int", -1, "Int64")
        AhkTest.AssertEqual(0, idx)

        ; prod = 24
        val := DllCall(dllPath "\cnp_ahk_prod", "Ptr", ha, "Int", -1, "Double")
        AhkTest.AssertEqual(24.0, val)

        DllCall(dllPath "\cnp_ahk_free", "Ptr", ha, "Int")
        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestContiguousFloat64ArgmaxAndCumsum() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")
        memoryBefore := DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64")

        for testCase in [
            [[1.0, 5.0, 5.0, -2.0, 4.0], 1, [1.0, 6.0, 11.0, 9.0, 13.0]],
            [[1.0, 5.0, 5.0, -2.0], 1, [1.0, 6.0, 11.0, 9.0]]
        ] {
            values := testCase[1]
            expectedArgmax := testCase[2]
            expectedCumsum := testCase[3]
            count := values.Length
            shapeBuf := Buffer(8, 0)
            NumPut("Int64", count, shapeBuf, 0)
            dataBuf := Buffer(count * 8, 0)
            for index, value in values
                NumPut("Double", value, dataBuf, (index - 1) * 8)

            sourceHandle := DllCall(dllPath "\cnp_ahk_from_doubles",
                "Int", 1, "Ptr", shapeBuf, "Ptr", dataBuf,
                "Int64", count, "Ptr")
            AhkTest.AssertTrue(sourceHandle != 0, "source handle should not be null")

            actualArgmax := DllCall(dllPath "\cnp_ahk_argmax",
                "Ptr", sourceHandle, "Int", -1, "Int64")
            AhkTest.AssertEqual(expectedArgmax, actualArgmax,
                "argmax must keep the lowest index for equal maxima")

            resultHandle := DllCall(dllPath "\cnp_ahk_cumsum",
                "Ptr", sourceHandle, "Int", -1, "Ptr")
            AhkTest.AssertTrue(resultHandle != 0, "cumsum handle should not be null")
            outputBuf := Buffer(count * 8, 0)
            copied := DllCall(dllPath "\cnp_ahk_get_doubles",
                "Ptr", resultHandle, "Ptr", outputBuf,
                "Int64", count, "Int64")
            AhkTest.AssertEqual(count, copied)
            for index, expectedValue in expectedCumsum {
                actualValue := NumGet(outputBuf, (index - 1) * 8, "Double")
                AhkTest.AssertTrue(Abs(expectedValue - actualValue) <= 1e-10,
                    "cumsum index " index " expected " expectedValue " got " actualValue)
            }

            AhkTest.AssertEqual(0,
                DllCall(dllPath "\cnp_ahk_free", "Ptr", resultHandle, "Int"))
            AhkTest.AssertEqual(0,
                DllCall(dllPath "\cnp_ahk_free", "Ptr", sourceHandle, "Int"))
        }

        specialShape := Buffer(8, 0)
        NumPut("Int64", 3, specialShape, 0)
        specialData := Buffer(24, 0)
        ; Quiet NaN, negative infinity, and a finite value encoded directly.
        NumPut("UInt", 0, specialData, 0)
        NumPut("UInt", 0x7ff80000, specialData, 4)
        NumPut("UInt", 0, specialData, 8)
        NumPut("UInt", 0xfff00000, specialData, 12)
        NumPut("Double", -5.0, specialData, 16)
        specialHandle := DllCall(dllPath "\cnp_ahk_from_doubles",
            "Int", 1, "Ptr", specialShape, "Ptr", specialData,
            "Int64", 3, "Ptr")
        AhkTest.AssertEqual(0,
            DllCall(dllPath "\cnp_ahk_argmax",
                "Ptr", specialHandle, "Int", -1, "Int64"),
            "argmax first NaN must win per NumPy 1.25")
        AhkTest.AssertEqual(0,
            DllCall(dllPath "\cnp_ahk_free", "Ptr", specialHandle, "Int"))

        memoryAfter := DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64")
        AhkTest.AssertEqual(memoryBefore, memoryAfter,
            "argmax/cumsum test must release all native memory")
        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestRuntimeSimdDispatchAndReductionEdges() {
        dllPath := NumpyLoadDll()
        AhkTest.AssertEqual(0, DllCall(dllPath "\cnp_ahk_init", "Int"))
        level := DllCall(dllPath "\cnp_ahk_simd_level", "Int")
        AhkTest.AssertTrue(level = 1 || level = 2,
            "SIMD level must be documented SSE2 or AVX2")
        memoryBefore := DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64")

        positiveInfinity := NumpyDoubleFromBits(0x7ff00000)
        negativeInfinity := NumpyDoubleFromBits(0xfff00000)
        quietNan := NumpyDoubleFromBits(0x7ff80000)
        handles := []

        single := NumpyCreateDoubleHandle(dllPath, [7.0])
        handles.Push(single)
        AhkTest.AssertEqual(7.0,
            DllCall(dllPath "\cnp_ahk_max", "Ptr", single, "Int", -1, "Double"))
        AhkTest.AssertEqual(7.0,
            DllCall(dllPath "\cnp_ahk_min", "Ptr", single, "Int", -1, "Double"))
        AhkTest.AssertEqual(0,
            DllCall(dllPath "\cnp_ahk_argmax", "Ptr", single, "Int", -1, "Int64"))

        odd := NumpyCreateDoubleHandle(dllPath, [1.0, 9.0, 3.0, 9.0, 2.0])
        handles.Push(odd)
        AhkTest.AssertEqual(9.0,
            DllCall(dllPath "\cnp_ahk_max", "Ptr", odd, "Int", -1, "Double"))
        AhkTest.AssertEqual(1.0,
            DllCall(dllPath "\cnp_ahk_min", "Ptr", odd, "Int", -1, "Double"))
        AhkTest.AssertEqual(1,
            DllCall(dllPath "\cnp_ahk_argmax", "Ptr", odd, "Int", -1, "Int64"),
            "argmax must keep the first duplicate maximum")
        AhkTest.AssertEqual(486.0,
            DllCall(dllPath "\cnp_ahk_prod", "Ptr", odd, "Int", -1, "Double"),
            "SIMD prod must handle an odd tail")

        even := NumpyCreateDoubleHandle(dllPath, [-4.0, 8.0, 8.0, 0.0])
        handles.Push(even)
        AhkTest.AssertEqual(8.0,
            DllCall(dllPath "\cnp_ahk_max", "Ptr", even, "Int", -1, "Double"))
        AhkTest.AssertEqual(-4.0,
            DllCall(dllPath "\cnp_ahk_min", "Ptr", even, "Int", -1, "Double"))
        AhkTest.AssertEqual(1,
            DllCall(dllPath "\cnp_ahk_argmax", "Ptr", even, "Int", -1, "Int64"))

        prodTail := NumpyCreateDoubleHandle(dllPath,
            [2.0, 0.5, -3.0, -1.0,
             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        handles.Push(prodTail)
        AhkTest.AssertEqual(3.0,
            DllCall(dllPath "\cnp_ahk_prod", "Ptr", prodTail,
                "Int", -1, "Double"),
            "SIMD prod must combine vector lanes and a scalar tail")

        infinities := NumpyCreateDoubleHandle(dllPath,
            [negativeInfinity, -2.0, 5.0, positiveInfinity])
        handles.Push(infinities)
        AhkTest.AssertEqual(positiveInfinity,
            DllCall(dllPath "\cnp_ahk_max", "Ptr", infinities, "Int", -1, "Double"))
        AhkTest.AssertEqual(negativeInfinity,
            DllCall(dllPath "\cnp_ahk_min", "Ptr", infinities, "Int", -1, "Double"))
        AhkTest.AssertEqual(3,
            DllCall(dllPath "\cnp_ahk_argmax", "Ptr", infinities, "Int", -1, "Int64"))
        positiveInfinityProduct := NumpyCreateDoubleHandle(dllPath,
            [negativeInfinity, -2.0])
        handles.Push(positiveInfinityProduct)
        AhkTest.AssertEqual(positiveInfinity,
            DllCall(dllPath "\cnp_ahk_prod", "Ptr", positiveInfinityProduct,
                "Int", -1, "Double"))

        nanAndFinite := NumpyCreateDoubleHandle(dllPath,
            [quietNan, negativeInfinity, -5.0, 12.0])
        handles.Push(nanAndFinite)
        AhkTest.AssertEqual(0,
            DllCall(dllPath "\cnp_ahk_argmax", "Ptr", nanAndFinite,
                "Int", -1, "Int64"),
            "argmax first NaN must win per NumPy 1.25")

        allNan := NumpyCreateDoubleHandle(dllPath, [quietNan, quietNan, quietNan])
        handles.Push(allNan)
        AhkTest.AssertEqual(0,
            DllCall(dllPath "\cnp_ahk_argmax", "Ptr", allNan, "Int", -1, "Int64"),
            "all-NaN argmax must preserve index zero")
        nanProduct := DllCall(dllPath "\cnp_ahk_prod",
            "Ptr", allNan, "Int", -1, "Double")
        AhkTest.AssertTrue(nanProduct != nanProduct,
            "SIMD prod must propagate NaN")

        allNegativeInfinity := NumpyCreateDoubleHandle(dllPath,
            [negativeInfinity, negativeInfinity, negativeInfinity, negativeInfinity])
        handles.Push(allNegativeInfinity)
        AhkTest.AssertEqual(0,
            DllCall(dllPath "\cnp_ahk_argmax", "Ptr", allNegativeInfinity,
                "Int", -1, "Int64"),
            "all-negative-infinity argmax must preserve index zero")

        while handles.Length > 0
            AhkTest.AssertEqual(0,
                DllCall(dllPath "\cnp_ahk_free", "Ptr", handles.Pop(), "Int"))
        AhkTest.AssertEqual(memoryBefore,
            DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64"),
            "SIMD dispatch test must release all native memory")
        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestContiguousFloat64NormVarianceAndStd() {
        dllPath := NumpyLoadDll()
        AhkTest.AssertEqual(0, DllCall(dllPath "\cnp_ahk_init", "Int"))
        memoryBefore := DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64")

        for count in [1, 5, 16, 19, 257] {
            values := []
            total := 0.0
            squaredTotal := 0.0
            loop count {
                value := Mod(A_Index * 17 + 3, 29) / 7.0 - 2.0
                values.Push(value)
                total += value
                squaredTotal += value * value
            }
            mean := total / count
            squaredDeviation := 0.0
            for value in values {
                difference := value - mean
                squaredDeviation += difference * difference
            }

            handle := NumpyCreateDoubleHandle(dllPath, values)
            actualNorm := DllCall(dllPath "\cnp_ahk_linalg_norm",
                "Ptr", handle, "Double")
            actualVar := DllCall(dllPath "\cnp_ahk_var",
                "Ptr", handle, "Int", -1, "Double")
            actualStd := DllCall(dllPath "\cnp_ahk_std",
                "Ptr", handle, "Int", -1, "Double")
            AhkTest.AssertTrue(Abs(actualNorm - Sqrt(squaredTotal)) <= 1e-10,
                "norm mismatch for length " count)
            AhkTest.AssertTrue(Abs(actualVar - squaredDeviation / count) <= 1e-10,
                "variance mismatch for length " count)
            AhkTest.AssertTrue(Abs(actualStd - Sqrt(squaredDeviation / count)) <= 1e-10,
                "standard deviation mismatch for length " count)
            AhkTest.AssertEqual(0,
                DllCall(dllPath "\cnp_ahk_free", "Ptr", handle, "Int"))
        }

        source := NumpyCreateDoubleHandle(dllPath,
            [100.0, 1.0, 2.0, 3.0, 4.0, 200.0])
        sliceSpec := Buffer(32, 0)
        NumPut("Int64", 1, sliceSpec, 0)
        NumPut("Int64", 5, sliceSpec, 8)
        NumPut("Int64", 1, sliceSpec, 16)
        NumPut("UChar", 1, sliceSpec, 24)
        NumPut("UChar", 1, sliceSpec, 25)
        NumPut("UChar", 1, sliceSpec, 26)
        offsetView := DllCall(dllPath "\cnp_array_slice",
            "Ptr", source, "Int", 1, "Ptr", sliceSpec, "Ptr")
        AhkTest.AssertTrue(offsetView != 0)
        AhkTest.AssertTrue(Abs(DllCall(dllPath "\cnp_ahk_linalg_norm",
            "Ptr", offsetView, "Double") - Sqrt(30.0)) <= 1e-10)
        AhkTest.AssertTrue(Abs(DllCall(dllPath "\cnp_ahk_var",
            "Ptr", offsetView, "Int", -1, "Double") - 1.25) <= 1e-10)
        AhkTest.AssertTrue(Abs(DllCall(dllPath "\cnp_ahk_std",
            "Ptr", offsetView, "Int", -1, "Double") - Sqrt(1.25)) <= 1e-10)
        AhkTest.AssertEqual(0,
            DllCall(dllPath "\cnp_ahk_free", "Ptr", offsetView, "Int"))
        AhkTest.AssertEqual(0,
            DllCall(dllPath "\cnp_ahk_free", "Ptr", source, "Int"))
        AhkTest.AssertEqual(memoryBefore,
            DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64"),
            "norm/variance/std test must release all native memory")
        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestPreallocatedIntoAndBatchApis() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")
        memoryBefore := DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64")
        shape := Buffer(8, 0)
        NumPut("Int64", 5, shape, 0)
        leftData := Buffer(40, 0)
        rightData := Buffer(40, 0)
        for index, value in [1.0, 4.0, 9.0, 16.0, 25.0]
            NumPut("Double", value, leftData, (index - 1) * 8)
        for index, value in [3.0, 5.0, 7.0, 9.0, 11.0]
            NumPut("Double", value, rightData, (index - 1) * 8)
        left := DllCall(dllPath "\cnp_ahk_from_doubles", "Int", 1, "Ptr", shape,
            "Ptr", leftData, "Int64", 5, "Ptr")
        right := DllCall(dllPath "\cnp_ahk_from_doubles", "Int", 1, "Ptr", shape,
            "Ptr", rightData, "Int64", 5, "Ptr")
        addOut := DllCall(dllPath "\cnp_ahk_zeros", "Int", 1, "Ptr", shape, "Ptr")
        sqrtOut := DllCall(dllPath "\cnp_ahk_zeros", "Int", 1, "Ptr", shape, "Ptr")
        cumsumOut := DllCall(dllPath "\cnp_ahk_zeros", "Int", 1, "Ptr", shape, "Ptr")
        memoryAfterSetup := DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64")

        AhkTest.AssertEqual(0, DllCall(dllPath "\cnp_ahk_add_into",
            "Ptr", left, "Ptr", right, "Ptr", addOut, "Int"))
        AhkTest.AssertEqual(0, DllCall(dllPath "\cnp_ahk_sqrt_into",
            "Ptr", addOut, "Ptr", sqrtOut, "Int"))
        AhkTest.AssertEqual(0, DllCall(dllPath "\cnp_ahk_cumsum_into",
            "Ptr", left, "Int", -1, "Ptr", cumsumOut, "Int"))
        addOutView := Numpy.NdArray.FromHandle(addOut)
        addOutView._owned := false
        sqrtOutView := Numpy.NdArray.FromHandle(sqrtOut)
        sqrtOutView._owned := false
        cumsumOutView := Numpy.NdArray.FromHandle(cumsumOut)
        cumsumOutView._owned := false
        NumpyAssertArrayNear([4.0, 9.0, 16.0, 25.0, 36.0], addOutView.ToArray())
        NumpyAssertArrayNear([2.0, 3.0, 4.0, 5.0, 6.0], sqrtOutView.ToArray())
        NumpyAssertArrayNear([1.0, 5.0, 14.0, 30.0, 55.0], cumsumOutView.ToArray())

        scalar := Buffer(8, 0)
        AhkTest.AssertEqual(0, DllCall(dllPath "\cnp_ahk_sum_into_scalar",
            "Ptr", sqrtOut, "Ptr", scalar, "Int"))
        AhkTest.AssertEqual(20.0, NumGet(scalar, 0, "Double"))

        loop 100 {
            AhkTest.AssertEqual(0, DllCall(dllPath "\cnp_ahk_add_into",
                "Ptr", left, "Ptr", right, "Ptr", addOut, "Int"))
        }
        AhkTest.AssertEqual(memoryAfterSetup,
            DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64"),
            "destination calls must not allocate native memory")

        commands := Buffer(120, 0)
        NumPut("UInt", 1, commands, 0)
        NumPut("Ptr", left, commands, 8)
        NumPut("Ptr", right, commands, 16)
        NumPut("Ptr", addOut, commands, 24)
        NumPut("UInt", 2, commands, 40)
        NumPut("Ptr", addOut, commands, 48)
        NumPut("Ptr", sqrtOut, commands, 64)
        NumPut("UInt", 4, commands, 80)
        NumPut("Ptr", sqrtOut, commands, 88)
        NumPut("Ptr", scalar.Ptr, commands, 104)
        NumPut("Int64", -1, commands, 112)
        failedIndex := Buffer(8, 0)
        AhkTest.AssertEqual(0, DllCall(dllPath "\cnp_ahk_execute_batch",
            "Ptr", commands, "Int64", 3, "Ptr", failedIndex, "Int"))
        AhkTest.AssertEqual(-1, NumGet(failedIndex, 0, "Int64"))
        AhkTest.AssertEqual(20.0, NumGet(scalar, 0, "Double"))

        AhkTest.AssertEqual(0, DllCall(dllPath "\cnp_ahk_sqrt_into",
            "Ptr", left, "Ptr", left, "Int"))
        leftView := Numpy.NdArray.FromHandle(left)
        leftView._owned := false
        NumpyAssertArrayNear([1.0, 2.0, 3.0, 4.0, 5.0], leftView.ToArray())

        for handle in [cumsumOut, sqrtOut, addOut, right, left]
            AhkTest.AssertEqual(0, DllCall(dllPath "\cnp_ahk_free", "Ptr", handle, "Int"))
        AhkTest.AssertEqual(memoryBefore,
            DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64"),
            "destination test must release all native memory")
        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestPreallocatedIntoRejectsInvalidInputsAndStopsBatch() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")
        memoryBefore := DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64")

        shape5 := Buffer(8, 0)
        NumPut("Int64", 5, shape5, 0)
        data5 := Buffer(40, 0)
        otherData5 := Buffer(40, 0)
        intData5 := Buffer(40, 0)
        for index, value in [1.0, 4.0, 9.0, 16.0, 25.0] {
            NumPut("Double", value, data5, (index - 1) * 8)
            NumPut("Double", value + 1.0, otherData5, (index - 1) * 8)
            NumPut("Int64", index, intData5, (index - 1) * 8)
        }
        left := DllCall(dllPath "\cnp_ahk_from_doubles",
            "Int", 1, "Ptr", shape5, "Ptr", data5, "Int64", 5, "Ptr")
        right := DllCall(dllPath "\cnp_ahk_from_doubles",
            "Int", 1, "Ptr", shape5, "Ptr", otherData5, "Int64", 5, "Ptr")
        validOut := DllCall(dllPath "\cnp_ahk_zeros",
            "Int", 1, "Ptr", shape5, "Ptr")
        intOut := DllCall(dllPath "\cnp_ahk_from_ints",
            "Int", 1, "Ptr", shape5, "Ptr", intData5, "Int64", 5, "Ptr")

        shape4 := Buffer(8, 0)
        NumPut("Int64", 4, shape4, 0)
        shortOut := DllCall(dllPath "\cnp_ahk_zeros",
            "Int", 1, "Ptr", shape4, "Ptr")

        matrixShape := Buffer(16, 0)
        NumPut("Int64", 2, matrixShape, 0)
        NumPut("Int64", 3, matrixShape, 8)
        matrixData := Buffer(48, 0)
        loop 6
            NumPut("Double", A_Index, matrixData, (A_Index - 1) * 8)
        matrix := DllCall(dllPath "\cnp_ahk_from_doubles",
            "Int", 2, "Ptr", matrixShape, "Ptr", matrixData, "Int64", 6, "Ptr")
        transposed := DllCall(dllPath "\cnp_ahk_transpose", "Ptr", matrix, "Ptr")
        transposedShape := Buffer(16, 0)
        NumPut("Int64", 3, transposedShape, 0)
        NumPut("Int64", 2, transposedShape, 8)
        matrixOut := DllCall(dllPath "\cnp_ahk_zeros",
            "Int", 2, "Ptr", transposedShape, "Ptr")

        AhkTest.AssertTrue(DllCall(dllPath "\cnp_ahk_add_into",
            "Ptr", left, "Ptr", right, "Ptr", shortOut, "Int") != 0,
            "add_into must reject a destination shape mismatch")
        AhkTest.AssertTrue(DllCall(dllPath "\cnp_ahk_sqrt_into",
            "Ptr", left, "Ptr", intOut, "Int") != 0,
            "sqrt_into must reject a non-float64 destination")
        AhkTest.AssertTrue(DllCall(dllPath "\cnp_ahk_sqrt_into",
            "Ptr", transposed, "Ptr", matrixOut, "Int") != 0,
            "sqrt_into must reject a non-contiguous source")
        AhkTest.AssertTrue(DllCall(dllPath "\cnp_ahk_cumsum_into",
            "Ptr", left, "Int", 1, "Ptr", validOut, "Int") != 0,
            "cumsum_into must reject an unsupported axis")

        commands := Buffer(80, 0)
        NumPut("UInt", 999, commands, 0)
        NumPut("UInt", 1, commands, 40)
        NumPut("Ptr", left, commands, 48)
        NumPut("Ptr", right, commands, 56)
        NumPut("Ptr", validOut, commands, 64)
        failedIndex := Buffer(8, 0)
        NumPut("Int64", -2, failedIndex, 0)
        AhkTest.AssertTrue(DllCall(dllPath "\cnp_ahk_execute_batch",
            "Ptr", commands, "Int64", 2, "Ptr", failedIndex, "Int") != 0,
            "unknown opcode must fail")
        AhkTest.AssertEqual(0, NumGet(failedIndex, 0, "Int64"),
            "batch must report the first failing command")
        AhkTest.AssertEqual(0.0,
            DllCall(dllPath "\cnp_ahk_get_item",
                "Ptr", validOut, "Int64", 0, "Double"),
            "batch must stop before the second command")

        NumPut("UInt", 1, commands, 0)
        NumPut("UInt", 1, commands, 4)
        NumPut("Ptr", left, commands, 8)
        NumPut("Ptr", right, commands, 16)
        NumPut("Ptr", validOut, commands, 24)
        AhkTest.AssertTrue(DllCall(dllPath "\cnp_ahk_execute_batch",
            "Ptr", commands, "Int64", 1, "Ptr", failedIndex, "Int") != 0,
            "nonzero reserved metadata must fail")
        AhkTest.AssertEqual(0, NumGet(failedIndex, 0, "Int64"))

        for handle in [matrixOut, transposed, matrix, shortOut,
                       intOut, validOut, right, left]
            AhkTest.AssertEqual(0, DllCall(dllPath "\cnp_ahk_free", "Ptr", handle, "Int"))
        AhkTest.AssertEqual(memoryBefore,
            DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64"),
            "invalid-input test must release all native memory")
        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestHighLevelIntoAndBatchWrapper() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := DllCall(Numpy.RequireDllPath()
            "\cnp_ahk_allocated_memory", "Int64")
        left := Numpy.Array([1.0, 4.0, 9.0, 16.0, 25.0])
        right := Numpy.Array([3.0, 5.0, 7.0, 9.0, 11.0])
        addOut := Numpy.Zeros([5])
        sqrtOut := Numpy.Zeros([5])
        cumsumOut := Numpy.Zeros([5])

        left.AddInto(right, addOut)
        addOut.SqrtInto(sqrtOut)
        left.CumsumInto(cumsumOut)
        NumpyAssertArrayNear([4.0, 9.0, 16.0, 25.0, 36.0], addOut.ToArray())
        NumpyAssertArrayNear([2.0, 3.0, 4.0, 5.0, 6.0], sqrtOut.ToArray())
        NumpyAssertArrayNear([1.0, 5.0, 14.0, 30.0, 55.0], cumsumOut.ToArray())
        AhkTest.AssertEqual(20.0, sqrtOut.SumIntoScalar())
        AhkTest.AssertEqual(20.0,
            Numpy.AddSqrtSumBatch(left, right, addOut, sqrtOut))

        cumsumOut := 0
        sqrtOut := 0
        addOut := 0
        right := 0
        left := 0
        AhkTest.AssertEqual(memoryBefore,
            DllCall(Numpy.RequireDllPath()
                "\cnp_ahk_allocated_memory", "Int64"),
            "high-level wrappers must release all native memory")
        Numpy.Cleanup()
    }

    static TestHighLevelScalarAndAxisReductions() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])

        scalarCases := [
            [source.Sum(), 21.0, "sum"],
            [source.Prod(), 720.0, "prod"],
            [source.Mean(), 3.5, "mean"],
            [source.Var(), 35.0 / 12.0, "var"],
            [source.Std(), Sqrt(35.0 / 12.0), "std"],
            [source.Max(), 6.0, "max"],
            [source.Min(), 1.0, "min"]
        ]
        for scalarCase in scalarCases {
            AhkTest.AssertEqual("Float", Type(scalarCase[1]),
                scalarCase[3] " without axis must return a scalar")
            AhkTest.AssertTrue(Abs(scalarCase[1] - scalarCase[2]) <= 1e-10,
                scalarCase[3] " scalar value mismatch")
        }

        axisCases := [
            [source.Sum(0), [5.0, 7.0, 9.0], "sum"],
            [source.Prod(0), [4.0, 10.0, 18.0], "prod"],
            [source.Mean(0), [2.5, 3.5, 4.5], "mean"],
            [source.Var(0), [2.25, 2.25, 2.25], "var"],
            [source.Std(0), [1.5, 1.5, 1.5], "std"],
            [source.Max(0), [4.0, 5.0, 6.0], "max"],
            [source.Min(0), [1.0, 2.0, 3.0], "min"]
        ]
        for axisCase in axisCases {
            axisResult := axisCase[1]
            AhkTest.AssertEqual("Numpy.NdArray", Type(axisResult),
                axisCase[3] " with axis must return NdArray")
            AhkTest.AssertTrue(axisResult._owned,
                axisCase[3] " axis result must own its native handle")
            NumpyAssertArrayNear([3], axisResult.Shape, 0)
            NumpyAssertArrayNear(axisCase[2], axisResult.ToArray())
        }

        lastAxisCases := [
            [source.Sum(-1), [6.0, 15.0], "sum"],
            [source.Prod(-1), [6.0, 120.0], "prod"],
            [source.Mean(-1), [2.0, 5.0], "mean"],
            [source.Var(-1, 1), [1.0, 1.0], "var/ddof"],
            [source.Std(-1, 1), [1.0, 1.0], "std/ddof"],
            [source.Max(-1), [3.0, 6.0], "max"],
            [source.Min(-1), [1.0, 4.0], "min"],
            [source.Argmax(-1), [2.0, 2.0], "argmax"],
            [source.Argmin(-1), [0.0, 0.0], "argmin"],
            [source.Any(-1), [1.0, 1.0], "any"],
            [source.All(-1), [1.0, 1.0], "all"],
            [source.Ptp(-1), [2.0, 2.0], "ptp"],
            [source.Median(-1), [2.0, 5.0], "median"],
            [source.Percentile(50.0, -1), [2.0, 5.0], "percentile"],
            [source.Quantile(0.5, -1), [2.0, 5.0], "quantile"],
            [source.Nansum(-1), [6.0, 15.0], "nansum"],
            [source.Nanprod(-1), [6.0, 120.0], "nanprod"],
            [source.Nanmean(-1), [2.0, 5.0], "nanmean"],
            [source.Nanvar(-1, 1), [1.0, 1.0], "nanvar/ddof"],
            [source.Nanstd(-1, 1), [1.0, 1.0], "nanstd/ddof"],
            [source.Nanmax(-1), [3.0, 6.0], "nanmax"],
            [source.Nanmin(-1), [1.0, 4.0], "nanmin"],
            [source.Nanargmax(-1), [2.0, 2.0], "nanargmax"],
            [source.Nanargmin(-1), [0.0, 0.0], "nanargmin"],
            [source.Nanmedian(-1), [2.0, 5.0], "nanmedian"],
            [source.Nanpercentile(50.0, -1), [2.0, 5.0], "nanpercentile"],
            [source.Nanquantile(0.5, -1), [2.0, 5.0], "nanquantile"]
        ]
        for lastAxisCase in lastAxisCases {
            AhkTest.AssertEqual("Numpy.NdArray", Type(lastAxisCase[1]),
                lastAxisCase[3] " with explicit -1 must return NdArray")
            NumpyAssertArrayNear([2], lastAxisCase[1].Shape, 0)
            NumpyAssertArrayNear(lastAxisCase[2], lastAxisCase[1].ToArray())
        }

        cumulativeCases := [
            [source.Cumsum(-1), [1.0, 3.0, 6.0, 4.0, 9.0, 15.0], "cumsum"],
            [source.Cumprod(-1), [1.0, 2.0, 6.0, 4.0, 20.0, 120.0], "cumprod"],
            [source.Nancumsum(-1), [1.0, 3.0, 6.0, 4.0, 9.0, 15.0], "nancumsum"],
            [source.Nancumprod(-1), [1.0, 2.0, 6.0, 4.0, 20.0, 120.0], "nancumprod"]
        ]
        for cumulativeCase in cumulativeCases {
            NumpyAssertArrayNear([2, 3], cumulativeCase[1].Shape, 0)
            NumpyAssertArrayNear(cumulativeCase[2], cumulativeCase[1].ToArray())
        }

        facadeLastAxis := Numpy.Sum(source, -1)
        AhkTest.AssertEqual("Numpy.NdArray", Type(facadeLastAxis),
            "static facade must preserve explicit -1")
        NumpyAssertArrayNear([6.0, 15.0], facadeLastAxis.ToArray())

        beforeError := Numpy.AllocatedMemory()
        rejectedInvalidAxis := false
        try source.Sum(2)
        catch Error as err
            rejectedInvalidAxis := InStr(err.Message, "axis 2") > 0
        AhkTest.AssertTrue(rejectedInvalidAxis,
            "invalid reduction axis must surface the native error")
        AhkTest.AssertEqual(beforeError, Numpy.AllocatedMemory(),
            "invalid reduction axis must not retain native memory")

        err := 0
        facadeLastAxis := 0
        cumulativeCase := 0
        cumulativeCases := 0
        lastAxisCase := 0
        lastAxisCases := 0
        axisResult := 0
        axisCase := 0
        axisCases := 0
        scalarCase := 0
        scalarCases := 0
        source := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "scalar/axis reduction wrappers must release all native memory")
        Numpy.Cleanup()
    }

    static TestBlockedTransposeCopyFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        for dimensions in [[3, 5], [33, 65], [512, 512]] {
            rows := dimensions[1]
            columns := dimensions[2]
            source := Numpy.Zeros([rows, columns])
            points := [
                [0, 0],
                [0, columns - 1],
                [rows - 1, 0],
                [rows - 1, columns - 1],
                [rows > 31 ? 31 : rows - 1, columns > 31 ? 31 : columns - 1],
                [rows > 32 ? 32 : rows - 1, columns > 32 ? 32 : columns - 1]
            ]
            expectedValues := []
            for pointIndex, point in points {
                row := point[1]
                column := point[2]
                value := pointIndex * 100000.0 + row * 1000.0 + column + 0.25
                source.SetItem(row * columns + column, value)
            }
            for point in points
                expectedValues.Push(source.GetItem(point[1] * columns + point[2]))

            fused := source.TransposeCopy()
            view := source.Transpose()
            contiguous := view.AsContiguousArray()
            for result in [fused, contiguous] {
                AhkTest.AssertEqual("Numpy.NdArray", Type(result))
                AhkTest.AssertTrue(result._owned)
                NumpyAssertArrayNear([columns, rows], result.Shape, 0)
                for pointIndex, point in points {
                    transposedIndex := point[2] * rows + point[1]
                    AhkTest.AssertEqual(expectedValues[pointIndex],
                        result.GetItem(transposedIndex),
                        rows "x" columns " transpose sample " pointIndex)
                }
            }

            source.SetItem(0, -999.0)
            AhkTest.AssertEqual(expectedValues[1], fused.GetItem(0),
                "TransposeCopy must return independent storage")

            result := 0
            contiguous := 0
            view := 0
            fused := 0
            source := 0
        }
        dimensions := 0
        points := 0
        expectedValues := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "blocked transpose copies must release all native memory")
        Numpy.Cleanup()
    }

    static TestPublicElementwiseDispatchEdges() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        for count in [1, 3, 4, 15, 16, 17, 1001] {
            leftValues := []
            rightValues := []
            sqrtValues := []
            expectedAdd := []
            expectedSubtract := []
            expectedMultiply := []
            expectedDivide := []
            expectedMaximum := []
            expectedNegative := []
            expectedAbsolute := []
            expectedSqrt := []
            expectedFloor := []
            loop count {
                leftValue := Mod(A_Index * 17, 41) - 20.25
                rightValue := Mod(A_Index * 13, 23) + 1.5
                sqrtValue := Mod(A_Index * 19, 37) + 0.25
                leftValues.Push(leftValue)
                rightValues.Push(rightValue)
                sqrtValues.Push(sqrtValue)
                expectedAdd.Push(leftValue + rightValue)
                expectedSubtract.Push(leftValue - rightValue)
                expectedMultiply.Push(leftValue * rightValue)
                expectedDivide.Push(leftValue / rightValue)
                expectedMaximum.Push(leftValue > rightValue ? leftValue : rightValue)
                expectedNegative.Push(-leftValue)
                expectedAbsolute.Push(Abs(leftValue))
                expectedSqrt.Push(Sqrt(sqrtValue))
                expectedFloor.Push(Floor(leftValue))
            }

            left := Numpy.Array(leftValues)
            right := Numpy.Array(rightValues)
            sqrtInput := Numpy.Array(sqrtValues)
            addResult := left.Add(right)
            subtractResult := left.Subtract(right)
            multiplyResult := left.Multiply(right)
            divideResult := left.Divide(right)
            maximumResult := left.Maximum(right)
            negativeResult := left.Negative()
            absoluteResult := left.Absolute()
            sqrtResult := sqrtInput.Sqrt()
            floorResult := left.Floor()
            addDestination := Numpy.Zeros([count])
            sqrtDestination := Numpy.Zeros([count])
            left.AddInto(right, addDestination)
            sqrtInput.SqrtInto(sqrtDestination)

            NumpyAssertArrayNear(expectedAdd, addResult.ToArray())
            NumpyAssertArrayNear(expectedSubtract, subtractResult.ToArray())
            NumpyAssertArrayNear(expectedMultiply, multiplyResult.ToArray())
            NumpyAssertArrayNear(expectedDivide, divideResult.ToArray())
            NumpyAssertArrayNear(expectedMaximum, maximumResult.ToArray())
            NumpyAssertArrayNear(expectedNegative, negativeResult.ToArray())
            NumpyAssertArrayNear(expectedAbsolute, absoluteResult.ToArray())
            NumpyAssertArrayNear(expectedSqrt, sqrtResult.ToArray())
            NumpyAssertArrayNear(expectedFloor, floorResult.ToArray())
            NumpyAssertArrayNear(expectedAdd, addDestination.ToArray())
            NumpyAssertArrayNear(expectedSqrt, sqrtDestination.ToArray())

            sqrtDestination := 0
            addDestination := 0
            floorResult := 0
            sqrtResult := 0
            absoluteResult := 0
            negativeResult := 0
            maximumResult := 0
            divideResult := 0
            multiplyResult := 0
            subtractResult := 0
            addResult := 0
            sqrtInput := 0
            right := 0
            left := 0
        }

        positiveInfinity := NumpyDoubleFromBits(0x7ff00000)
        negativeInfinity := NumpyDoubleFromBits(0xfff00000)
        negativeZero := NumpyDoubleFromBits(0x80000000)
        quietNan := NumpyDoubleFromBits(0x7ff80000)
        numerator := Numpy.Array([1.0, 1.0, -1.0, -1.0])
        denominator := Numpy.Array([0.0, negativeZero, 0.0, negativeZero])
        zeroDivision := numerator.Divide(denominator).ToArray()
        AhkTest.AssertEqual(positiveInfinity, zeroDivision[1])
        AhkTest.AssertEqual(negativeInfinity, zeroDivision[2])
        AhkTest.AssertEqual(negativeInfinity, zeroDivision[3])
        AhkTest.AssertEqual(positiveInfinity, zeroDivision[4])

        maximumLeft := Numpy.Array([quietNan, 2.0])
        maximumRight := Numpy.Array([1.0, quietNan])
        nanMaximum := maximumLeft.Maximum(maximumRight).ToArray()
        AhkTest.AssertTrue(nanMaximum[1] != nanMaximum[1])
        AhkTest.AssertTrue(nanMaximum[2] != nanMaximum[2])
        negativeSquareRoot := Numpy.Array([-1.0]).Sqrt().ToArray()[1]
        AhkTest.AssertTrue(negativeSquareRoot != negativeSquareRoot)

        rawSource := NumpyCreateDoubleHandle(Numpy.RequireDllPath(),
            [999.0, -1.25, 2.5, -3.75, 888.0])
        sliceSpec := Buffer(32, 0)
        NumPut("Int64", 1, sliceSpec, 0)
        NumPut("Int64", 4, sliceSpec, 8)
        NumPut("Int64", 1, sliceSpec, 16)
        NumPut("UChar", 1, sliceSpec, 24)
        NumPut("UChar", 1, sliceSpec, 25)
        NumPut("UChar", 1, sliceSpec, 26)
        offsetHandle := DllCall(Numpy.RequireDllPath() "\cnp_array_slice",
            "Ptr", rawSource, "Int", 1, "Ptr", sliceSpec, "Ptr")
        AhkTest.AssertTrue(offsetHandle != 0)
        rawArray := Numpy.NdArray.FromHandle(rawSource)
        offsetView := Numpy.NdArray.FromHandle(offsetHandle)
        rawArray := 0
        NumpyAssertArrayNear([1.25, 2.5, 3.75], offsetView.Absolute().ToArray())

        offsetView := 0
        maximumRight := 0
        maximumLeft := 0
        denominator := 0
        numerator := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "elementwise dispatch tests must release all native memory")
        Numpy.Cleanup()
    }

    static TestDispatchedGemmEdges() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        cases := [
            [1, 1, 1, false],
            [2, 3, 5, false],
            [7, 9, 6, false],
            [33, 35, 31, false],
            [65, 67, 63, false],
            [7, 7, 7, true]
        ]
        for dimensions in cases {
            rows := dimensions[1]
            innerSize := dimensions[2]
            columns := dimensions[3]
            duplicateInput := dimensions[4]
            leftValues := []
            rightValues := []
            loop rows * innerSize
                leftValues.Push(Mod(A_Index * 17 + 3, 29) / 11.0 - 1.0)
            if duplicateInput {
                rightValues := leftValues.Clone()
            } else {
                loop innerSize * columns
                    rightValues.Push(Mod(A_Index * 13 + 5, 31) / 9.0 - 1.25)
            }

            expected := []
            loop rows {
                row := A_Index - 1
                loop columns {
                    column := A_Index - 1
                    total := 0.0
                    loop innerSize {
                        inner := A_Index - 1
                        total += leftValues[row * innerSize + inner + 1]
                            * rightValues[inner * columns + column + 1]
                    }
                    expected.Push(total)
                }
            }

            left := Numpy.Array(leftValues, [rows, innerSize])
            if duplicateInput
                right := left
            else
                right := Numpy.Array(rightValues, [innerSize, columns])
            result := left.Matmul(right)
            NumpyAssertArrayNear([rows, columns], result.Shape, 0)
            NumpyAssertArrayNear(expected, result.ToArray(), 1e-9)

            result := 0
            right := 0
            left := 0
        }
        dimensions := 0
        cases := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "GEMM edge tests must release all native memory")
        Numpy.Cleanup()
    }

    static TestParallelGemmThreadControlsAndDeterminism() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        AhkTest.AssertEqual(0, Numpy.GetNumThreads(),
            "fresh GEMM thread configuration must be auto")

        rows := 257
        innerSize := 259
        columns := 255
        leftValues := []
        rightValues := []
        loop rows * innerSize
            leftValues.Push(Mod(A_Index * 17 + 3, 29) / 11.0 - 1.0)
        loop innerSize * columns
            rightValues.Push(Mod(A_Index * 13 + 5, 31) / 9.0 - 1.25)
        left := Numpy.Array(leftValues, [rows, innerSize])
        right := Numpy.Array(rightValues, [innerSize, columns])

        outputs := []
        for threadCount in [1, 2, 0] {
            Numpy.SetNumThreads(threadCount)
            AhkTest.AssertEqual(threadCount, Numpy.GetNumThreads(),
                "configured GEMM thread count must round-trip")
            outputs.Push(left.Matmul(right))
        }
        expected := outputs[1].ToArray()
        NumpyAssertArrayNear(expected, outputs[2].ToArray(), 0)
        NumpyAssertArrayNear(expected, outputs[3].ToArray(), 0)

        rejected := false
        try Numpy.SetNumThreads(-1)
        catch Error as err
            rejected := InStr(err.Message, "non-negative") > 0
        AhkTest.AssertTrue(rejected,
            "negative thread counts must expose the native argument error")
        AhkTest.AssertEqual(0, Numpy.GetNumThreads(),
            "rejected configuration must preserve the previous value")

        output := 0
        while outputs.Length > 0
            output := outputs.Pop()
        output := 0
        outputs := 0
        right := 0
        left := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "parallel GEMM tests must release all native memory")
        Numpy.Cleanup()
    }

    static TestFixedMetadataAbiAndLazyCache() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])

        metadata := Buffer(544, 0)
        status := DllCall(Numpy.RequireDllPath() "\cnp_ahk_get_metadata",
            "Ptr", source.Handle, "Ptr", metadata.Ptr,
            "UInt", metadata.Size, "Int")
        AhkTest.AssertEqual(0, status)
        AhkTest.AssertEqual(1, NumGet(metadata, 0, "UInt"))
        AhkTest.AssertEqual(544, NumGet(metadata, 4, "UInt"))
        AhkTest.AssertEqual(2, NumGet(metadata, 8, "Int"))
        AhkTest.AssertEqual(Numpy.DT_FLOAT64, NumGet(metadata, 12, "Int"))
        AhkTest.AssertEqual(8, NumGet(metadata, 16, "Int"))
        AhkTest.AssertTrue(NumGet(metadata, 20, "UInt") != 0)
        AhkTest.AssertEqual(6, NumGet(metadata, 24, "Int64"))
        AhkTest.AssertEqual(2, NumGet(metadata, 32, "Int64"))
        AhkTest.AssertEqual(3, NumGet(metadata, 40, "Int64"))

        tooSmall := Buffer(543, 0xA5)
        AhkTest.AssertTrue(DllCall(Numpy.RequireDllPath()
            "\cnp_ahk_get_metadata", "Ptr", source.Handle,
            "Ptr", tooSmall.Ptr, "UInt", tooSmall.Size, "Int") != 0,
            "undersized metadata buffers must fail explicitly")
        AhkTest.AssertEqual(0xA5, NumGet(tooSmall, 0, "UChar"),
            "failed metadata calls must not mutate the caller buffer")

        AhkTest.AssertEqual(2, source.Ndim)
        AhkTest.AssertEqual(6, source.Size)
        AhkTest.AssertEqual(Numpy.DT_FLOAT64, source.Dtype)
        AhkTest.AssertEqual(8, source.ItemSize)
        AhkTest.AssertTrue(source.Flags != 0)
        returnedShape := source.Shape
        returnedShape[1] := 99
        NumpyAssertArrayNear([2, 3], source.Shape, 0)
        AhkTest.AssertTrue(source._metadataLoaded,
            "metadata must remain cached after the first property access")

        source := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "metadata cache tests must release all native memory")
        Numpy.Cleanup()
    }

    static TestNumpyStyleStaticFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        np := Numpy
        a := np.array([1.0, 2.0, 3.0, 4.0], [2, 2])
        b := np.ones([2, 2])
        out := np.zeros([2, 2])

        AhkTest.AssertEqual(out.Handle, np.add(a, b, out).Handle)
        NumpyAssertArrayNear([2.0, 3.0, 4.0, 5.0], out.ToArray())
        root := np.sqrt(out)
        total := np.sum(root)
        AhkTest.AssertTrue(Abs(
            Sqrt(2.0) + Sqrt(3.0) + 2.0 + Sqrt(5.0) - total) <= 1e-12)
        columns := np.sum(root, 0)
        NumpyAssertArrayNear([Sqrt(2.0) + 2.0, Sqrt(3.0) + Sqrt(5.0)],
            columns.ToArray(), 1e-12)

        product := np.matmul(a, b)
        NumpyAssertArrayNear([3.0, 3.0, 7.0, 7.0], product.ToArray())
        flat := np.reshape(np.transpose(product), [4])
        NumpyAssertArrayNear([3.0, 7.0, 3.0, 7.0], flat.ToArray())
        joined := np.concatenate([a, b], 0)
        NumpyAssertArrayNear([4, 2], joined.Shape, 0)
        NumpyAssertArrayNear(
            [1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0],
            joined.ToArray())

        cumulativeOut := np.zeros([4])
        AhkTest.AssertEqual(cumulativeOut.Handle,
            np.cumsum(a, -1, cumulativeOut).Handle)
        NumpyAssertArrayNear([1.0, 3.0, 6.0, 10.0],
            cumulativeOut.ToArray())
        copied := np.ascontiguousarray(np.transpose(a))
        NumpyAssertArrayNear([1.0, 3.0, 2.0, 4.0], copied.ToArray())
        flattened := np.flatten(a)
        NumpyAssertArrayNear([1.0, 2.0, 3.0, 4.0], flattened.ToArray())
        AhkTest.AssertEqual(3, np.argmax(a))
        AhkTest.AssertTrue(Abs(24.0 - np.prod(a)) <= 1e-12)

        rejected := false
        try np.concatenate([a, 0], 0)
        catch TypeError
            rejected := true
        AhkTest.AssertTrue(rejected,
            "facade concatenate must reject non-NdArray inputs")

        flattened := 0
        copied := 0
        cumulativeOut := 0
        joined := 0
        flat := 0
        product := 0
        columns := 0
        root := 0
        out := 0
        b := 0
        a := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "NumPy-style facade tests must release all native memory")
        Numpy.Cleanup()
    }

    static TestReshapeTranspose() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")

        ; Create 2x3 array [1..6]
        shapeBuf := Buffer(16, 0)
        NumPut("Int64", 2, shapeBuf, 0)
        NumPut("Int64", 3, shapeBuf, 8)
        dataBuf := Buffer(48, 0)
        loop 6
            NumPut("Double", A_Index * 1.0, dataBuf, (A_Index - 1) * 8)
        h := DllCall(dllPath "\cnp_ahk_from_doubles", "Int", 2, "Ptr", shapeBuf, "Ptr", dataBuf, "Int64", 6, "Ptr")

        ; Reshape to 3x2
        newShape := Buffer(16, 0)
        NumPut("Int64", 3, newShape, 0)
        NumPut("Int64", 2, newShape, 8)
        hr := DllCall(dllPath "\cnp_ahk_reshape", "Ptr", h, "Int", 2, "Ptr", newShape, "Ptr")
        AhkTest.AssertTrue(hr != 0)
        ndim := DllCall(dllPath "\cnp_ahk_ndim", "Ptr", hr, "Int")
        AhkTest.AssertEqual(2, ndim)
        size := DllCall(dllPath "\cnp_ahk_size", "Ptr", hr, "Int64")
        AhkTest.AssertEqual(6, size)
        DllCall(dllPath "\cnp_ahk_free", "Ptr", hr, "Int")

        ; Transpose 2x3 -> 3x2
        hr := DllCall(dllPath "\cnp_ahk_transpose", "Ptr", h, "Ptr")
        AhkTest.AssertTrue(hr != 0)
        shapeOut := Buffer(16, 0)
        DllCall(dllPath "\cnp_ahk_shape", "Ptr", hr, "Ptr", shapeOut, "Int")
        AhkTest.AssertEqual(3, NumGet(shapeOut, 0, "Int64"))
        AhkTest.AssertEqual(2, NumGet(shapeOut, 8, "Int64"))
        DllCall(dllPath "\cnp_ahk_free", "Ptr", hr, "Int")

        ; Flatten
        hr := DllCall(dllPath "\cnp_ahk_flatten", "Ptr", h, "Ptr")
        AhkTest.AssertTrue(hr != 0)
        ndim := DllCall(dllPath "\cnp_ahk_ndim", "Ptr", hr, "Int")
        AhkTest.AssertEqual(1, ndim)
        size := DllCall(dllPath "\cnp_ahk_size", "Ptr", hr, "Int64")
        AhkTest.AssertEqual(6, size)
        DllCall(dllPath "\cnp_ahk_free", "Ptr", hr, "Int")

        DllCall(dllPath "\cnp_ahk_free", "Ptr", h, "Int")
        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestReshapeSharesStorageAndSurvivesSourceRelease() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")
        memoryBefore := DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64")

        sourceShape := Buffer(16, 0)
        NumPut("Int64", 2, sourceShape, 0)
        NumPut("Int64", 3, sourceShape, 8)
        sourceData := Buffer(48, 0)
        loop 6
            NumPut("Double", A_Index, sourceData, (A_Index - 1) * 8)
        sourceHandle := DllCall(dllPath "\cnp_ahk_from_doubles",
            "Int", 2, "Ptr", sourceShape, "Ptr", sourceData,
            "Int64", 6, "Ptr")

        viewShape := Buffer(16, 0)
        NumPut("Int64", 3, viewShape, 0)
        NumPut("Int64", 2, viewShape, 8)
        viewHandle := DllCall(dllPath "\cnp_ahk_reshape",
            "Ptr", sourceHandle, "Int", 2, "Ptr", viewShape, "Ptr")
        AhkTest.AssertTrue(viewHandle != 0)
        AhkTest.AssertEqual(1,
            DllCall(dllPath "\cnp_ahk_shares_data",
                "Ptr", sourceHandle, "Ptr", viewHandle, "Int"),
            "reshape must share the source allocation")

        sourceShapeAfter := Buffer(16, 0)
        DllCall(dllPath "\cnp_ahk_shape", "Ptr", sourceHandle,
            "Ptr", sourceShapeAfter, "Int")
        AhkTest.AssertEqual(2, NumGet(sourceShapeAfter, 0, "Int64"))
        AhkTest.AssertEqual(3, NumGet(sourceShapeAfter, 8, "Int64"))
        viewShapeAfter := Buffer(16, 0)
        DllCall(dllPath "\cnp_ahk_shape", "Ptr", viewHandle,
            "Ptr", viewShapeAfter, "Int")
        AhkTest.AssertEqual(3, NumGet(viewShapeAfter, 0, "Int64"))
        AhkTest.AssertEqual(2, NumGet(viewShapeAfter, 8, "Int64"))

        sourceFlags := DllCall(dllPath "\cnp_ahk_flags",
            "Ptr", sourceHandle, "UInt")
        viewFlags := DllCall(dllPath "\cnp_ahk_flags",
            "Ptr", viewHandle, "UInt")
        AhkTest.AssertTrue((sourceFlags & 0x0004) != 0)
        AhkTest.AssertEqual(0, viewFlags & (0x0004 | 0x0800),
            "reshape view must not own heap or virtual storage")

        AhkTest.AssertEqual(0,
            DllCall(dllPath "\cnp_ahk_free", "Ptr", sourceHandle, "Int"),
            "source release should defer storage destruction")
        AhkTest.AssertEqual(5.0,
            DllCall(dllPath "\cnp_ahk_get_item",
                "Ptr", viewHandle, "Int64", 4, "Double"))
        AhkTest.AssertEqual(0,
            DllCall(dllPath "\cnp_ahk_set_item",
                "Ptr", viewHandle, "Int64", 4, "Double", 42.0, "Int"))
        AhkTest.AssertEqual(42.0,
            DllCall(dllPath "\cnp_ahk_get_item",
                "Ptr", viewHandle, "Int64", 4, "Double"))
        AhkTest.AssertEqual(0,
            DllCall(dllPath "\cnp_ahk_free", "Ptr", viewHandle, "Int"))
        AhkTest.AssertEqual(memoryBefore,
            DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64"),
            "heap reshape view must release source storage last")

        virtualShape := Buffer(8, 0)
        NumPut("Int64", 1000000, virtualShape, 0)
        virtualSource := DllCall(dllPath "\cnp_ahk_zeros",
            "Int", 1, "Ptr", virtualShape, "Ptr")
        virtualViewShape := Buffer(16, 0)
        NumPut("Int64", 1000, virtualViewShape, 0)
        NumPut("Int64", 1000, virtualViewShape, 8)
        virtualView := DllCall(dllPath "\cnp_ahk_reshape",
            "Ptr", virtualSource, "Int", 2, "Ptr", virtualViewShape, "Ptr")
        AhkTest.AssertEqual(1,
            DllCall(dllPath "\cnp_ahk_shares_data",
                "Ptr", virtualSource, "Ptr", virtualView, "Int"))
        AhkTest.AssertTrue((DllCall(dllPath "\cnp_ahk_flags",
            "Ptr", virtualSource, "UInt") & 0x0800) != 0)
        AhkTest.AssertEqual(0, DllCall(dllPath "\cnp_ahk_flags",
            "Ptr", virtualView, "UInt") & (0x0004 | 0x0800))

        AhkTest.AssertEqual(0,
            DllCall(dllPath "\cnp_ahk_free", "Ptr", virtualSource, "Int"))
        AhkTest.AssertEqual(0.0,
            DllCall(dllPath "\cnp_ahk_get_item",
                "Ptr", virtualView, "Int64", 999999, "Double"))
        AhkTest.AssertEqual(0,
            DllCall(dllPath "\cnp_ahk_free", "Ptr", virtualView, "Int"))
        AhkTest.AssertEqual(memoryBefore,
            DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64"),
            "virtual reshape view must release source storage last")

        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestOffsetAsStridedSurvivesSourceFirstRelease() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")
        memoryBefore := DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64")

        sourceShape := Buffer(8, 0)
        NumPut("Int64", 6, sourceShape, 0)
        sourceData := Buffer(48, 0)
        for index, value in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
            NumPut("Double", value, sourceData, (index - 1) * 8)
        sourceHandle := DllCall(dllPath "\cnp_ahk_from_doubles",
            "Int", 1, "Ptr", sourceShape, "Ptr", sourceData,
            "Int64", 6, "Ptr")

        sliceSpec := Buffer(32, 0)
        NumPut("Int64", 1, sliceSpec, 0)
        NumPut("Int64", 6, sliceSpec, 8)
        NumPut("Int64", 2, sliceSpec, 16)
        NumPut("UChar", 1, sliceSpec, 24)
        NumPut("UChar", 1, sliceSpec, 25)
        NumPut("UChar", 1, sliceSpec, 26)
        sliceHandle := DllCall(dllPath "\cnp_array_slice",
            "Ptr", sourceHandle, "Int", 1, "Ptr", sliceSpec, "Ptr")
        AhkTest.AssertTrue(sliceHandle != 0)

        viewShape := Buffer(8, 0)
        viewStrides := Buffer(8, 0)
        NumPut("Int64", 2, viewShape, 0)
        NumPut("Int64", 16, viewStrides, 0)
        viewHandle := DllCall(dllPath "\cnp_as_strided",
            "Ptr", sliceHandle, "Int", 1, "Ptr", viewShape,
            "Ptr", viewStrides, "Ptr")
        AhkTest.AssertTrue(viewHandle != 0)
        memoryAfterConstruction := DllCall(
            dllPath "\cnp_ahk_allocated_memory", "Int64")

        AhkTest.AssertEqual(0,
            DllCall(dllPath "\cnp_ahk_free", "Ptr", sourceHandle, "Int"))
        AhkTest.AssertEqual(0,
            DllCall(dllPath "\cnp_ahk_free", "Ptr", sliceHandle, "Int"))
        AhkTest.AssertEqual(memoryAfterConstruction,
            DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64"),
            "as_strided must retain its immediate slice base")
        AhkTest.AssertEqual(20.0,
            DllCall(dllPath "\cnp_ahk_get_item",
                "Ptr", viewHandle, "Int64", 0, "Double"))
        AhkTest.AssertEqual(40.0,
            DllCall(dllPath "\cnp_ahk_get_item",
                "Ptr", viewHandle, "Int64", 1, "Double"))
        AhkTest.AssertEqual(0,
            DllCall(dllPath "\cnp_ahk_flags", "Ptr", viewHandle, "UInt")
                & (0x0004 | 0x0800))
        AhkTest.AssertEqual(0,
            DllCall(dllPath "\cnp_ahk_free", "Ptr", viewHandle, "Int"))
        AhkTest.AssertEqual(memoryBefore,
            DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64"))
        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestVirtualTransposeRetainsStorageWithoutOwnershipFlags() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")
        memoryBefore := DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64")
        shapeBuf := Buffer(8, 0)
        NumPut("Int64", 1000000, shapeBuf, 0)
        sourceHandle := DllCall(dllPath "\cnp_ahk_zeros",
            "Int", 1, "Ptr", shapeBuf, "Ptr")
        viewHandle := DllCall(dllPath "\cnp_ahk_transpose",
            "Ptr", sourceHandle, "Ptr")
        memoryAfterConstruction := DllCall(
            dllPath "\cnp_ahk_allocated_memory", "Int64")
        AhkTest.AssertEqual(0,
            DllCall(dllPath "\cnp_ahk_flags", "Ptr", viewHandle, "UInt")
                & (0x0004 | 0x0800))
        AhkTest.AssertEqual(0,
            DllCall(dllPath "\cnp_ahk_free", "Ptr", sourceHandle, "Int"))
        AhkTest.AssertEqual(memoryAfterConstruction,
            DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64"))
        AhkTest.AssertEqual(0.0,
            DllCall(dllPath "\cnp_ahk_get_item",
                "Ptr", viewHandle, "Int64", 999999, "Double"))
        AhkTest.AssertEqual(0,
            DllCall(dllPath "\cnp_ahk_free", "Ptr", viewHandle, "Int"))
        AhkTest.AssertEqual(memoryBefore,
            DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64"))
        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestTransposeRejectsDuplicateAxes() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")
        shapeBuf := Buffer(16, 0)
        NumPut("Int64", 2, shapeBuf, 0)
        NumPut("Int64", 3, shapeBuf, 8)
        sourceHandle := DllCall(dllPath "\cnp_ahk_zeros",
            "Int", 2, "Ptr", shapeBuf, "Ptr")
        axesBuf := Buffer(8, 0)
        NumPut("Int", 0, axesBuf, 0)
        NumPut("Int", 0, axesBuf, 4)
        invalidHandle := DllCall(dllPath "\cnp_transpose",
            "Ptr", sourceHandle, "Ptr", axesBuf, "Ptr")
        errorStatus := DllCall(dllPath "\cnp_get_error", "Ptr", 0, "Int")
        if invalidHandle
            DllCall(dllPath "\cnp_ahk_free", "Ptr", invalidHandle, "Int")
        AhkTest.AssertEqual(0, invalidHandle,
            "duplicate transpose axes must be rejected")
        AhkTest.AssertTrue(errorStatus != 0,
            "duplicate transpose axes must set an explicit error")
        AhkTest.AssertEqual(0,
            DllCall(dllPath "\cnp_ahk_free", "Ptr", sourceHandle, "Int"))
        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestLinalg() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")

        ; 2x2 identity matrix
        shapeBuf := Buffer(16, 0)
        NumPut("Int64", 2, shapeBuf, 0)
        NumPut("Int64", 2, shapeBuf, 8)
        dataBuf := Buffer(32, 0)
        NumPut("Double", 1.0, dataBuf, 0)
        NumPut("Double", 0.0, dataBuf, 8)
        NumPut("Double", 0.0, dataBuf, 16)
        NumPut("Double", 1.0, dataBuf, 24)
        h := DllCall(dllPath "\cnp_ahk_from_doubles", "Int", 2, "Ptr", shapeBuf, "Ptr", dataBuf, "Int64", 4, "Ptr")

        ; det(I) = 1
        det := DllCall(dllPath "\cnp_ahk_linalg_det", "Ptr", h, "Double")
        AhkTest.AssertTrue(Abs(det - 1.0) < 1e-10, "det(I) should be 1, got " det)

        ; inv(I) = I
        hi := DllCall(dllPath "\cnp_ahk_linalg_inv", "Ptr", h, "Ptr")
        AhkTest.AssertTrue(hi != 0)
        v := DllCall(dllPath "\cnp_ahk_get_item", "Ptr", hi, "Int64", 0, "Double")
        AhkTest.AssertTrue(Abs(v - 1.0) < 1e-10)
        DllCall(dllPath "\cnp_ahk_free", "Ptr", hi, "Int")

        DllCall(dllPath "\cnp_ahk_free", "Ptr", h, "Int")
        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestDotProduct() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")

        ; a = [1, 2, 3], b = [4, 5, 6]
        shapeBuf := Buffer(8, 0)
        NumPut("Int64", 3, shapeBuf, 0)
        dataA := Buffer(24, 0)
        NumPut("Double", 1.0, dataA, 0)
        NumPut("Double", 2.0, dataA, 8)
        NumPut("Double", 3.0, dataA, 16)
        ha := DllCall(dllPath "\cnp_ahk_from_doubles", "Int", 1, "Ptr", shapeBuf, "Ptr", dataA, "Int64", 3, "Ptr")

        dataB := Buffer(24, 0)
        NumPut("Double", 4.0, dataB, 0)
        NumPut("Double", 5.0, dataB, 8)
        NumPut("Double", 6.0, dataB, 16)
        hb := DllCall(dllPath "\cnp_ahk_from_doubles", "Int", 1, "Ptr", shapeBuf, "Ptr", dataB, "Int64", 3, "Ptr")

        ; dot([1,2,3], [4,5,6]) = 32
        hr := DllCall(dllPath "\cnp_ahk_dot", "Ptr", ha, "Ptr", hb, "Ptr")
        AhkTest.AssertTrue(hr != 0)
        val := DllCall(dllPath "\cnp_ahk_get_item", "Ptr", hr, "Int64", 0, "Double")
        AhkTest.AssertEqual(32.0, val)
        DllCall(dllPath "\cnp_ahk_free", "Ptr", hr, "Int")

        DllCall(dllPath "\cnp_ahk_free", "Ptr", ha, "Int")
        DllCall(dllPath "\cnp_ahk_free", "Ptr", hb, "Int")
        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestSort() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")

        shapeBuf := Buffer(8, 0)
        NumPut("Int64", 5, shapeBuf, 0)
        dataBuf := Buffer(40, 0)
        NumPut("Double", 3.0, dataBuf, 0)
        NumPut("Double", 1.0, dataBuf, 8)
        NumPut("Double", 4.0, dataBuf, 16)
        NumPut("Double", 1.0, dataBuf, 24)
        NumPut("Double", 5.0, dataBuf, 32)
        h := DllCall(dllPath "\cnp_ahk_from_doubles", "Int", 1, "Ptr", shapeBuf, "Ptr", dataBuf, "Int64", 5, "Ptr")

        hr := DllCall(dllPath "\cnp_ahk_sort", "Ptr", h, "Int", -1, "Ptr")
        AhkTest.AssertTrue(hr != 0)
        outBuf := Buffer(40, 0)
        DllCall(dllPath "\cnp_ahk_get_doubles", "Ptr", hr, "Ptr", outBuf, "Int64", 5, "Int64")
        AhkTest.AssertEqual(1.0, NumGet(outBuf, 0, "Double"))
        AhkTest.AssertEqual(1.0, NumGet(outBuf, 8, "Double"))
        AhkTest.AssertEqual(3.0, NumGet(outBuf, 16, "Double"))
        AhkTest.AssertEqual(4.0, NumGet(outBuf, 24, "Double"))
        AhkTest.AssertEqual(5.0, NumGet(outBuf, 32, "Double"))
        DllCall(dllPath "\cnp_ahk_free", "Ptr", hr, "Int")

        DllCall(dllPath "\cnp_ahk_free", "Ptr", h, "Int")
        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestSortUniqueAndSetFacadeV2() {
        Numpy.Init()
        memoryBefore := DllCall(
            Numpy.Proc("cnp_ahk_allocated_memory"), "Int64")
        source := Numpy.IntArray([3, 1, 2, 2, 5, 4], [2, 3])
        lastAxis := source.Sort(-1, "stable")
        flattened := source.Sort("none", "stable")
        heapIndices := source.Argsort(-1, "heapsort")
        quickSorted := Numpy.Sort(source, -1, "quicksort")
        mergeIndices := Numpy.Argsort(source, -1, "mergesort")
        NumpyAssertArrayNear([1, 2, 3, 2, 4, 5], lastAxis.ToArray(), 0)
        NumpyAssertArrayNear([2, 3], lastAxis.Shape, 0)
        NumpyAssertArrayNear([1, 2, 2, 3, 4, 5], flattened.ToArray(), 0)
        NumpyAssertArrayNear([6], flattened.Shape, 0)
        NumpyAssertArrayNear([1, 2, 0, 0, 2, 1], heapIndices.ToArray(), 0)
        NumpyAssertArrayNear([1, 2, 3, 2, 4, 5], quickSorted.ToArray(), 0)
        NumpyAssertArrayNear([1, 2, 0, 0, 2, 1], mergeIndices.ToArray(), 0)

        uniqueParts := Numpy.Unique(source, true, true, true)
        AhkTest.AssertEqual(4, uniqueParts.Length)
        NumpyAssertArrayNear([1, 2, 3, 4, 5], uniqueParts[1].ToArray(), 0)
        NumpyAssertArrayNear([1, 2, 0, 5, 4], uniqueParts[2].ToArray(), 0)
        NumpyAssertArrayNear([2, 0, 1, 1, 4, 3], uniqueParts[3].ToArray(), 0)
        NumpyAssertArrayNear([1, 2, 1, 1, 1], uniqueParts[4].ToArray(), 0)

        right := Numpy.IntArray([2, 4, 6])
        intersection := Numpy.Intersect1d(source, right)
        union := Numpy.Union1d(source, right)
        difference := Numpy.Setdiff1d(source, right)
        symmetric := Numpy.Setxor1d(source, right)
        membership := Numpy.In1d(source, right)
        shapedMembership := Numpy.Isin(source, right)
        NumpyAssertArrayNear([2, 4], intersection.ToArray(), 0)
        NumpyAssertArrayNear([1, 2, 3, 4, 5, 6], union.ToArray(), 0)
        NumpyAssertArrayNear([1, 3, 5], difference.ToArray(), 0)
        NumpyAssertArrayNear([1, 3, 5, 6], symmetric.ToArray(), 0)
        NumpyAssertArrayNear([0, 0, 1, 1, 0, 1], membership.ToArray(), 0)
        NumpyAssertArrayNear([2, 3], shapedMembership.Shape, 0)
        NumpyAssertArrayNear([0, 0, 1, 1, 0, 1], shapedMembership.ToArray(), 0)

        rejected := false
        try source.Sort(-1, "invalid")
        catch ValueError as err {
            rejected := InStr(err.Message, "invalid") > 0
        }
        AhkTest.AssertTrue(rejected, "invalid sort kind must be explicit")

        invalidAxisMessage := ""
        try source.Sort(-3, "stable")
        catch ValueError as err
            invalidAxisMessage := err.Message
        AhkTest.AssertEqual(
            "NdArray.Sort failed with status -5: axis -3 is out of bounds for array of dimension 2",
            invalidAxisMessage)

        source := 0
        NumpyAssertArrayNear([1, 2, 3, 2, 4, 5], lastAxis.ToArray(), 0)
        lastAxis := 0
        flattened := 0
        heapIndices := 0
        quickSorted := 0
        mergeIndices := 0
        uniqueParts := 0
        right := 0
        intersection := 0
        union := 0
        difference := 0
        symmetric := 0
        membership := 0
        shapedMembership := 0
        AhkTest.AssertEqual(memoryBefore, DllCall(
            Numpy.Proc("cnp_ahk_allocated_memory"), "Int64"))
    }

    static TestPartitionFacadeV2() {
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        sourceValues := [9, 1, 7, 2, 8, 4]
        source := Numpy.IntArray(sourceValues, [2, 3])
        axisPartition := 0
        axisIndices := 0
        flattened := 0
        flattenedSorted := 0
        unchanged := 0
        try {
            axisPartition := Numpy.Partition(source, 1)
            NumpyAssertArrayNear([2, 3], axisPartition.Shape, 0)
            values := axisPartition.ToArray()
            AhkTest.AssertEqual(7, values[2])
            AhkTest.AssertEqual(4, values[5])
            AhkTest.AssertTrue(
                values[1] <= values[2] && values[3] >= values[2],
                "first row must satisfy the kth partition invariant")
            AhkTest.AssertTrue(
                values[4] <= values[5] && values[6] >= values[5],
                "second row must satisfy the kth partition invariant")

            axisIndices := source.Argpartition([0, 2])
            NumpyAssertArrayNear([2, 3], axisIndices.Shape, 0)
            indices := axisIndices.ToArray()
            loop 2 {
                offset := (A_Index - 1) * 3
                sum := indices[offset + 1] + indices[offset + 2]
                    + indices[offset + 3]
                squares := indices[offset + 1] ** 2
                    + indices[offset + 2] ** 2
                    + indices[offset + 3] ** 2
                AhkTest.AssertEqual(3, sum)
                AhkTest.AssertEqual(5, squares)
                AhkTest.AssertEqual(
                    A_Index = 1 ? 1 : 2,
                    sourceValues[offset + indices[offset + 1] + 1])
                AhkTest.AssertEqual(
                    A_Index = 1 ? 9 : 8,
                    sourceValues[offset + indices[offset + 3] + 1])
            }

            flattened := source.Partition([1, 4], "none")
            NumpyAssertArrayNear([6], flattened.Shape, 0)
            flatValues := flattened.ToArray()
            AhkTest.AssertEqual(2, flatValues[2])
            AhkTest.AssertEqual(8, flatValues[5])
            flattenedSorted := flattened.Sort("none", "stable")
            NumpyAssertArrayNear(
                [1, 2, 4, 7, 8, 9], flattenedSorted.ToArray(), 0)

            unchanged := source.Partition([])
            NumpyAssertArrayNear(sourceValues, unchanged.ToArray(), 0)

            rejectedKth := false
            try source.Partition(3)
            catch ValueError as err
                rejectedKth := InStr(err.Message, "out of bounds") > 0
            AhkTest.AssertTrue(
                rejectedKth, "invalid kth must expose the native error")

            rejectedKind := false
            try source.Argpartition(1, -1, "stable")
            catch ValueError as err
                rejectedKind := InStr(err.Message, "introselect") > 0
            AhkTest.AssertTrue(
                rejectedKind, "invalid partition kind must be explicit")
        } finally {
            unchanged := 0
            flattenedSorted := 0
            flattened := 0
            axisIndices := 0
            axisPartition := 0
            source := 0
        }
        AhkTest.AssertEqual(
            memoryBefore, Numpy.AllocatedMemory(),
            "partition facade must release every native result")
        Numpy.Cleanup()
    }

    static TestSearchsortedFacadeV2() {
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.IntArray([40, 10, 30, 20])
        queries := Numpy.IntArray([10, 25, 40, 50], [2, 2])
        sorter := Numpy.IntArray([1, 3, 2, 0])
        left := 0
        right := 0
        try {
            left := Numpy.Searchsorted(
                source, queries, "left", sorter)
            right := source.Searchsorted(queries, "right", sorter)
            NumpyAssertArrayNear([2, 2], left.Shape, 0)
            NumpyAssertArrayNear([0, 2, 3, 4], left.ToArray(), 0)
            NumpyAssertArrayNear([1, 2, 4, 4], right.ToArray(), 0)

            rejectedSide := false
            try source.Searchsorted(queries, "middle")
            catch ValueError as err
                rejectedSide := InStr(err.Message, "left") > 0
                    && InStr(err.Message, "right") > 0
            AhkTest.AssertTrue(
                rejectedSide, "invalid search side must be explicit")
        } finally {
            right := 0
            left := 0
            sorter := 0
            queries := 0
            source := 0
        }
        AhkTest.AssertEqual(
            memoryBefore, Numpy.AllocatedMemory(),
            "searchsorted facade must release every native result")
        Numpy.Cleanup()
    }

    static TestDigitizeFacadeSemantics() {
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.IntArray([0, 1, 2, 4], [2, 2])
        increasingBins := Numpy.IntArray([1, 2, 3])
        decreasingBins := Numpy.IntArray([3, 2, 1])
        increasing := 0
        decreasing := 0
        try {
            increasing := Numpy.Digitize(source, increasingBins)
            decreasing := source.Digitize(decreasingBins, true)
            NumpyAssertArrayNear([2, 2], increasing.Shape, 0)
            NumpyAssertArrayNear(
                [0, 1, 2, 3], increasing.ToArray(), 0)
            NumpyAssertArrayNear(
                [3, 3, 2, 0], decreasing.ToArray(), 0)

            invalidBins := Numpy.IntArray([1, 3, 2])
            rejectedBins := false
            try source.Digitize(invalidBins)
            catch ValueError as err
                rejectedBins := InStr(err.Message, "monotonically") > 0
            invalidBins := 0
            AhkTest.AssertTrue(
                rejectedBins, "nonmonotonic bins must raise ValueError")
        } finally {
            decreasing := 0
            increasing := 0
            decreasingBins := 0
            increasingBins := 0
            source := 0
        }
        AhkTest.AssertEqual(
            memoryBefore, Numpy.AllocatedMemory(),
            "digitize facade must release every native result")
        Numpy.Cleanup()
    }

    static TestLexsortFacadeSemantics() {
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        secondary := Numpy.IntArray([2, 1, 2, 0, 2, 1], [2, 3])
        primary := Numpy.Array([1, 1, 0, 1, 0, 0], [2, 3])
        lastAxis := 0
        firstAxis := 0
        try {
            lastAxis := Numpy.Lexsort([secondary, primary])
            firstAxis := Numpy.Lexsort([secondary, primary], 0)
            NumpyAssertArrayNear([2, 3], lastAxis.Shape, 0)
            NumpyAssertArrayNear(
                [2, 1, 0, 2, 1, 0], lastAxis.ToArray(), 0)
            NumpyAssertArrayNear(
                [1, 1, 1, 0, 0, 0], firstAxis.ToArray(), 0)

            rejectedEmpty := false
            emptyMessage := ""
            try Numpy.Lexsort([])
            catch ValueError as err {
                emptyMessage := err.Message
                rejectedEmpty := InStr(err.Message, "at least one") > 0
                    && InStr(err.Message, "cnp_ahk_lexsort_v2") > 0
            }
            AhkTest.AssertTrue(
                rejectedEmpty,
                "empty lexsort keys must expose native error; actual: "
                    emptyMessage)

            rejectedItem := false
            try Numpy.Lexsort([secondary, 1])
            catch TypeError as err
                rejectedItem := InStr(err.Message, "item 2") > 0
            AhkTest.AssertTrue(
                rejectedItem, "lexsort key types must be explicit")

            rejectedAxis := false
            try Numpy.Lexsort([secondary, primary], 0.5)
            catch TypeError as err
                rejectedAxis := InStr(err.Message, "axis") > 0
            AhkTest.AssertTrue(
                rejectedAxis, "lexsort axis must be an Integer")
        } finally {
            firstAxis := 0
            lastAxis := 0
            primary := 0
            secondary := 0
        }
        AhkTest.AssertEqual(
            memoryBefore, Numpy.AllocatedMemory(),
            "lexsort facade must release every native result")
        Numpy.Cleanup()
    }

    static TestMsortAndSortComplexFacadeSemantics() {
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.IntArray([3, 1, 2, 0, 4, -1], [2, 3])
        msortResult := 0
        complexResult := 0
        realResult := 0
        imagResult := 0
        try {
            msortResult := Numpy.Msort(source)
            NumpyAssertArrayNear(
                [0, 1, -1, 3, 4, 2], msortResult.ToArray(), 0)
            AhkTest.AssertEqual(source.Dtype, msortResult.Dtype)

            complexResult := Numpy.SortComplex(source)
            AhkTest.AssertEqual(Numpy.DT_COMPLEX128, complexResult.Dtype)
            realResult := Numpy.Real(complexResult)
            imagResult := Numpy.Imag(complexResult)
            NumpyAssertArrayNear(
                [1, 2, 3, -1, 0, 4], realResult.ToArray(), 0)
            NumpyAssertArrayNear(
                [0, 0, 0, 0, 0, 0], imagResult.ToArray(), 0)
            NumpyAssertArrayNear(
                [3, 1, 2, 0, 4, -1], source.ToArray(), 0)

            rejectedMsort := false
            try Numpy.Msort(1)
            catch TypeError as err
                rejectedMsort := InStr(err.Message, "NdArray") > 0
            AhkTest.AssertTrue(
                rejectedMsort, "Msort must reject non-array sources")

            rejectedComplex := false
            try Numpy.SortComplex(1)
            catch TypeError as err
                rejectedComplex := InStr(err.Message, "NdArray") > 0
            AhkTest.AssertTrue(
                rejectedComplex,
                "SortComplex must reject non-array sources")
        } finally {
            imagResult := 0
            realResult := 0
            complexResult := 0
            msortResult := 0
            source := 0
        }
        AhkTest.AssertEqual(
            memoryBefore, Numpy.AllocatedMemory(),
            "msort/sort_complex facade must release every native result")
        Numpy.Cleanup()
    }

    static TestMiscAxisFacadeV2() {
        Numpy.Init()
        memoryBefore := DllCall(
            Numpy.Proc("cnp_ahk_allocated_memory"), "Int64")
        source := Numpy.Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])

        softmax := Numpy.Softmax(source, 1)
        logSoftmax := source.LogSoftmax(1)
        softmaxAxis0 := source.Softmax(0)
        logSoftmaxAlias := Numpy.Log_Softmax(source, 1)
        rowExpected := [
            0.09003057317038046,
            0.24472847105479764,
            0.6652409557748218
        ]
        NumpyAssertArrayNear([
            rowExpected[1], rowExpected[2], rowExpected[3],
            rowExpected[1], rowExpected[2], rowExpected[3]
        ], softmax.ToArray(), 1e-12)
        NumpyAssertArrayNear([
            -2.4076059644443806,
            -1.4076059644443804,
            -0.4076059644443804,
            -2.4076059644443806,
            -1.4076059644443804,
            -0.4076059644443804
        ], logSoftmax.ToArray(), 1e-12)
        NumpyAssertArrayNear([
            0.04742587317756679,
            0.04742587317756679,
            0.04742587317756679,
            0.9525741268224334,
            0.9525741268224334,
            0.9525741268224334
        ], softmaxAxis0.ToArray(), 1e-12)
        NumpyAssertArrayNear(
            logSoftmax.ToArray(), logSoftmaxAlias.ToArray(), 0)

        integratedDx := Numpy.Trapz(source, , 0.5, 1)
        x := Numpy.Array([0.0, 1.0, 3.0])
        integratedX := source.Trapz(x, 1.0, 1)
        integratedAxis0 := source.Trapz(, 2.0, 0)
        NumpyAssertArrayNear([2.0, 5.0], integratedDx.ToArray(), 1e-12)
        NumpyAssertArrayNear([6.5, 15.5], integratedX.ToArray(), 1e-12)
        NumpyAssertArrayNear(
            [5.0, 7.0, 9.0], integratedAxis0.ToArray(), 1e-12)

        bits := Numpy.IntArray([1, 0, 1, 0, 1, 0, 1, 0, 1])
        packedBig := Numpy.Packbits(bits, , "big")
        packedLittle := bits.Packbits(, "little")
        NumpyAssertArrayNear([170, 128], packedBig.ToArray(), 0)
        NumpyAssertArrayNear([85, 1], packedLittle.ToArray(), 0)

        bitMatrix := Numpy.IntArray([
            1, 0, 1, 0, 1, 0,
            1, 1, 0, 0, 0, 0
        ], [2, 6])
        packedAxisBig := bitMatrix.Packbits(1, "big")
        packedAxisLittle := Numpy.Packbits(bitMatrix, 1, "little")
        NumpyAssertArrayNear([168, 192], packedAxisBig.ToArray(), 0)
        NumpyAssertArrayNear([21, 3], packedAxisLittle.ToArray(), 0)
        NumpyAssertArrayNear([2, 1], packedAxisBig.Shape, 0)

        bytes := Numpy.Array([170, 128], , Numpy.DT_UINT8)
        unpacked := Numpy.Unpackbits(bytes, , 9, "big")
        NumpyAssertArrayNear(
            [1, 0, 1, 0, 1, 0, 1, 0, 1], unpacked.ToArray(), 0)

        byteMatrix := Numpy.Array([170, 128, 1, 255], [2, 2], Numpy.DT_UINT8)
        unpackedAxis := byteMatrix.Unpackbits(1, 9, "big")
        unpackedZero := Numpy.Unpackbits(byteMatrix, 1, 0, "little")
        unpackedTrimmed := byteMatrix.Unpackbits(1, -1, "big")
        unpackedPadded := byteMatrix.Unpackbits(1, 17, "little")
        NumpyAssertArrayNear([2, 9], unpackedAxis.Shape, 0)
        NumpyAssertArrayNear([2, 0], unpackedZero.Shape, 0)
        NumpyAssertArrayNear([2, 15], unpackedTrimmed.Shape, 0)
        NumpyAssertArrayNear([2, 17], unpackedPadded.Shape, 0)
        NumpyAssertArrayNear([], unpackedZero.ToArray(), 0)
        paddedValues := unpackedPadded.ToArray()
        AhkTest.AssertEqual(0, paddedValues[17])
        AhkTest.AssertEqual(0, paddedValues[34])

        rejected := false
        try Numpy.Packbits(bits, , "middle")
        catch ValueError as err {
            rejected := InStr(err.Message, "bitorder") > 0
        }
        AhkTest.AssertTrue(rejected, "invalid bitorder must be explicit")

        rejectedAxis := false
        try source.Softmax(2)
        catch ValueError as err {
            rejectedAxis := InStr(err.Message, "status -5") > 0
                && InStr(err.Message, "axis 2 is out of bounds") > 0
        }
        AhkTest.AssertTrue(
            rejectedAxis, "invalid softmax axis must preserve native status")

        rejectedCount := false
        try byteMatrix.Unpackbits(1, -17, "big")
        catch Error as err {
            rejectedCount := InStr(err.Message, "status -1") > 0
                && InStr(err.Message, "larger than number") > 0
        }
        AhkTest.AssertTrue(
            rejectedCount, "invalid unpack count must remain explicit")

        rejectedX := false
        try Numpy.Trapz(source, 1)
        catch TypeError as err
            rejectedX := InStr(err.Message, "x must be an NdArray") > 0
        AhkTest.AssertTrue(rejectedX, "Trapz x type must be explicit")

        rejectedCountType := false
        try Numpy.Unpackbits(byteMatrix, 1, 1.5, "big")
        catch TypeError as err
            rejectedCountType := InStr(err.Message, "count") > 0
        AhkTest.AssertTrue(
            rejectedCountType, "Unpackbits count type must be explicit")

        source := 0
        NumpyAssertArrayNear([
            rowExpected[1], rowExpected[2], rowExpected[3],
            rowExpected[1], rowExpected[2], rowExpected[3]
        ], softmax.ToArray(), 1e-12)
        softmax := 0
        logSoftmax := 0
        softmaxAxis0 := 0
        logSoftmaxAlias := 0
        integratedDx := 0
        x := 0
        integratedX := 0
        integratedAxis0 := 0
        bits := 0
        packedBig := 0
        packedLittle := 0
        bitMatrix := 0
        packedAxisBig := 0
        packedAxisLittle := 0
        bytes := 0
        unpacked := 0
        byteMatrix := 0
        unpackedAxis := 0
        unpackedZero := 0
        unpackedTrimmed := 0
        unpackedPadded := 0
        AhkTest.AssertEqual(memoryBefore, DllCall(
            Numpy.Proc("cnp_ahk_allocated_memory"), "Int64"))
    }

    static TestEinsumFacadeV2() {
        Numpy.Init()
        memoryBefore := DllCall(
            Numpy.Proc("cnp_ahk_allocated_memory"), "Int64")
        left := Numpy.Array([1.0, 2.0, 3.0, 4.0], [2, 2])
        right := Numpy.Array([5.0, 6.0, 7.0, 8.0], [2, 2])

        product := Numpy.Einsum("ij,jk->ik", left, right)
        diagonal := Numpy.Einsum("ii->i", left)
        NumpyAssertArrayNear(
            [19.0, 22.0, 43.0, 50.0], product.ToArray(), 1e-12)
        NumpyAssertArrayNear([1.0, 4.0], diagonal.ToArray(), 1e-12)

        rejected := false
        try Numpy.Einsum("ij->ik", left)
        catch Error as err {
            rejected := InStr(err.Message, "output") > 0
        }
        AhkTest.AssertTrue(rejected, "einsum errors must remain explicit")

        left := 0
        right := 0
        product := 0
        diagonal := 0
        AhkTest.AssertEqual(memoryBefore, DllCall(
            Numpy.Proc("cnp_ahk_allocated_memory"), "Int64"))
    }

    static TestGeneralEigFacadeV2() {
        Numpy.Init()
        memoryBefore := DllCall(
            Numpy.Proc("cnp_ahk_allocated_memory"), "Int64")
        source := Numpy.Array([0.0, -1.0, 1.0, 0.0], [2, 2])
        rectangular := Numpy.Array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
        quietNan := NumpyDoubleFromBits(0x7ff80000)
        nonfinite := Numpy.Array([quietNan, 0.0, 0.0, 1.0], [2, 2])

        results := Numpy.Eig(source)
        AhkTest.AssertEqual("Array", Type(results))
        AhkTest.AssertEqual(2, results.Length)
        eigenvalues := results[1]
        eigenvectors := results[2]
        AhkTest.AssertEqual(Numpy.DT_COMPLEX128, eigenvalues.Dtype)
        AhkTest.AssertEqual(Numpy.DT_COMPLEX128, eigenvectors.Dtype)
        NumpyAssertArrayNear([2], eigenvalues.Shape, 0)
        NumpyAssertArrayNear([2, 2], eigenvectors.Shape, 0)

        rejected := false
        try Numpy.Eig(rectangular)
        catch ValueError as err {
            rejected := InStr(err.Message, "status -4") > 0
                && InStr(err.Message, "square") > 0
        }
        AhkTest.AssertTrue(rejected, "eig errors must remain explicit")

        rejectedConvergence := false
        beforeFailure := Numpy.AllocatedMemory()
        try Numpy.Eig(nonfinite)
        catch Error as err {
            rejectedConvergence := InStr(err.Message, "status -10") > 0
                && InStr(err.Message, "NaN") > 0
        }
        AhkTest.AssertTrue(
            rejectedConvergence,
            "eig convergence errors must remain explicit through the facade")
        AhkTest.AssertEqual(
            beforeFailure, Numpy.AllocatedMemory(),
            "failed eig must release every partial result")

        source := 0
        rectangular := 0
        nonfinite := 0
        results := 0
        eigenvalues := 0
        eigenvectors := 0
        AhkTest.AssertEqual(memoryBefore, DllCall(
            Numpy.Proc("cnp_ahk_allocated_memory"), "Int64"))
    }

    static TestEigvalsFacadeV2() {
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        np := Numpy
        source := np.array([0.0, -1.0, 1.0, 0.0], [2, 2])
        rectangular := np.ones([2, 3])
        realSpectrum := np.array([2.0, 1.0, 0.0, 3.0], [2, 2])

        values := np.linalg.eigvals(source)
        AhkTest.AssertEqual(Numpy.DT_COMPLEX128, values.Dtype)
        NumpyAssertArrayNear([2], values.Shape, 0)
        source := 0
        NumpyAssertArrayNear([2], values.Shape, 0)

        rejected := false
        beforeFailure := Numpy.AllocatedMemory()
        try np.linalg.eigvals(rectangular)
        catch ValueError as err {
            rejected := InStr(err.Message, "Numpy.Eigvals") > 0
                && InStr(err.Message, "square") > 0
        }
        AhkTest.AssertTrue(rejected,
            "eigvals facade must preserve the native shape error")
        AhkTest.AssertEqual(beforeFailure, Numpy.AllocatedMemory(),
            "failed eigvals must not retain native memory")

        loop 16 {
            temporary := np.linalg.eigvals(realSpectrum)
            temporary := 0
        }
        values := 0
        rectangular := 0
        realSpectrum := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "eigvals facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestSymmetricEigFacadeV2() {
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.Array([1.0, 99.0, 2.0, 3.0], [2, 2])
        rectangular := Numpy.Ones([2, 3])
        lower := 0
        upper := 0
        lowerValues := 0
        upperValues := 0
        try {
            lower := Numpy.Eigh(source)
            AhkTest.AssertEqual("Array", Type(lower))
            AhkTest.AssertEqual(2, lower.Length)
            lowerValues := source.Eigvalsh()
            NumpyAssertArrayNear(
                [-0.2360679774997897, 4.23606797749979],
                lower[1].ToArray(), 1e-11)
            NumpyAssertArrayNear(
                lower[1].ToArray(), lowerValues.ToArray(), 1e-11)
            NumpyAssertArrayNear([2], lower[1].Shape, 0)
            NumpyAssertArrayNear([2, 2], lower[2].Shape, 0)

            upper := source.Eigh("U")
            upperValues := Numpy.Eigvalsh(source, "U")
            NumpyAssertArrayNear(
                [-97.00505037623081, 101.00505037623081],
                upper[1].ToArray(), 1e-10)
            NumpyAssertArrayNear(
                upper[1].ToArray(), upperValues.ToArray(), 1e-10)

            rejectedUplo := false
            try Numpy.Eigh(source, "X")
            catch ValueError as err
                rejectedUplo := InStr(err.Message, "UPLO") > 0
            AhkTest.AssertTrue(
                rejectedUplo, "Eigh must reject UPLO outside L/U")

            rejectedSource := false
            try Numpy.Eigvalsh(1)
            catch TypeError as err
                rejectedSource := InStr(err.Message, "NdArray") > 0
            AhkTest.AssertTrue(
                rejectedSource, "Eigvalsh must reject non-array sources")

            beforeFailure := Numpy.AllocatedMemory()
            rejectedShape := false
            try Numpy.Eigh(rectangular)
            catch ValueError as err {
                rejectedShape := InStr(err.Message, "square") > 0
                    && InStr(err.Message, "status -4") > 0
            }
            AhkTest.AssertTrue(
                rejectedShape, "Eigh must expose native shape errors")
            AhkTest.AssertEqual(
                beforeFailure, Numpy.AllocatedMemory(),
                "failed Eigh must not retain partial outputs")
        } finally {
            upperValues := 0
            lowerValues := 0
            upper := 0
            lower := 0
            rectangular := 0
            source := 0
        }
        AhkTest.AssertEqual(
            memoryBefore, Numpy.AllocatedMemory(),
            "Eigh/Eigvalsh facade must release every result")
        Numpy.Cleanup()
    }

    static TestLinalgSpectralDelegatesV2() {
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        general := Numpy.Array([0.0, -1.0, 1.0, 0.0], [2, 2])
        symmetric := Numpy.Array([2.0, 1.0, 1.0, 3.0], [2, 2])
        rectangular := Numpy.Ones([2, 3])
        eig := 0
        eigh := 0
        eigvalsh := 0
        svd := 0
        singularValues := 0
        try {
            eig := Numpy.Linalg.Eig(general)
            eigh := Numpy.Linalg.Eigh(symmetric)
            eigvalsh := Numpy.Linalg.Eigvalsh(symmetric)
            svd := Numpy.Linalg.Svd(rectangular, false)
            singularValues := Numpy.Linalg.Svd(rectangular, false, false)
            AhkTest.AssertEqual(2, eig.Length)
            AhkTest.AssertEqual(2, eigh.Length)
            NumpyAssertArrayNear(
                eigh[1].ToArray(), eigvalsh.ToArray(), 1e-12)
            AhkTest.AssertEqual(3, svd.Length)
            NumpyAssertArrayNear([2], singularValues.Shape, 0)

            rejectedShape := false
            try Numpy.Linalg.Eig(rectangular)
            catch ValueError as caughtShapeError
                rejectedShape := InStr(caughtShapeError.Message, "square") > 0
            AhkTest.AssertTrue(rejectedShape,
                "Linalg.Eig must expose the native shape error")

            rejectedSource := false
            try Numpy.Linalg.Eigh(1)
            catch TypeError as caughtSourceError
                rejectedSource := InStr(caughtSourceError.Message, "NdArray") > 0
            AhkTest.AssertTrue(rejectedSource,
                "Linalg.Eigh must preserve public source validation")
        } finally {
            singularValues := 0
            svd := 0
            eigvalsh := 0
            eigh := 0
            eig := 0
            rectangular := 0
            symmetric := 0
            general := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "Linalg spectral delegates must release every native result")
        Numpy.Cleanup()
    }

    static TestDetAndSlogdetFacadeV2() {
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.Array(
            [1.0, 2.0, 3.0, 5.0, 2.0, 1.0, 4.0, 3.0],
            [2, 2, 2])
        rectangular := Numpy.Ones([2, 3])
        determinant := 0
        linalgDeterminant := 0
        instanceDeterminant := 0
        slogdet := 0
        linalgSlogdet := 0
        instanceSlogdet := 0
        try {
            determinant := Numpy.Det(source)
            linalgDeterminant := Numpy.Linalg.Det(source)
            instanceDeterminant := source.Det()
            AhkTest.AssertTrue(determinant is Numpy.NdArray)
            NumpyAssertArrayNear([2], determinant.Shape, 0)
            NumpyAssertArrayNear([-1.0, 2.0], determinant.ToArray(), 1e-12)
            NumpyAssertArrayNear(
                determinant.ToArray(), linalgDeterminant.ToArray(), 0)
            NumpyAssertArrayNear(
                determinant.ToArray(), instanceDeterminant.ToArray(), 0)

            slogdet := Numpy.Slogdet(source)
            linalgSlogdet := Numpy.Linalg.Slogdet(source)
            instanceSlogdet := source.Slogdet()
            AhkTest.AssertEqual("Array", Type(slogdet))
            AhkTest.AssertEqual(2, slogdet.Length)
            NumpyAssertArrayNear([-1.0, 1.0], slogdet[1].ToArray(), 1e-12)
            NumpyAssertArrayNear([0.0, Ln(2.0)], slogdet[2].ToArray(), 1e-12)
            NumpyAssertArrayNear(
                slogdet[1].ToArray(), linalgSlogdet[1].ToArray(), 0)
            NumpyAssertArrayNear(
                slogdet[2].ToArray(), instanceSlogdet[2].ToArray(), 0)

            rejectedSource := false
            try Numpy.Det(1)
            catch TypeError as err
                rejectedSource := InStr(err.Message, "NdArray") > 0
            AhkTest.AssertTrue(
                rejectedSource, "Det must reject non-array sources")

            beforeFailure := Numpy.AllocatedMemory()
            rejectedShape := false
            try Numpy.Slogdet(rectangular)
            catch ValueError as err {
                rejectedShape := InStr(err.Message, "square") > 0
                    && InStr(err.Message, "status -4") > 0
            }
            AhkTest.AssertTrue(
                rejectedShape, "Slogdet must expose native shape errors")
            AhkTest.AssertEqual(
                beforeFailure, Numpy.AllocatedMemory(),
                "failed Slogdet must not retain partial results")
        } finally {
            instanceSlogdet := 0
            linalgSlogdet := 0
            slogdet := 0
            instanceDeterminant := 0
            linalgDeterminant := 0
            determinant := 0
            rectangular := 0
            source := 0
        }
        AhkTest.AssertEqual(
            memoryBefore, Numpy.AllocatedMemory(),
            "Det/Slogdet facade must release every result")
        Numpy.Cleanup()
    }

    static TestCholeskyFacadeV2() {
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.Array([
            4.0, 999.0, 2.0, 5.0,
            9.0, -999.0, 3.0, 10.0
        ], [2, 2, 2])
        integerSource := Numpy.IntArray([4, 99, 2, 5], [2, 2])
        singular := Numpy.Array([1.0, 8.0, 0.0, 0.0], [2, 2])
        rectangular := Numpy.Ones([2, 3])
        direct := 0
        moduleResult := 0
        instanceResult := 0
        integerResult := 0
        try {
            direct := Numpy.Cholesky(source)
            moduleResult := Numpy.Linalg.Cholesky(source)
            instanceResult := source.Cholesky()
            AhkTest.AssertEqual(Numpy.DT_FLOAT64, direct.Dtype)
            NumpyAssertArrayNear([2, 2, 2], direct.Shape, 0)
            NumpyAssertArrayNear(
                [2.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 3.0],
                direct.ToArray(), 1e-12)
            NumpyAssertArrayNear(
                direct.ToArray(), moduleResult.ToArray(), 0)
            NumpyAssertArrayNear(
                direct.ToArray(), instanceResult.ToArray(), 0)

            integerResult := Numpy.Cholesky(integerSource)
            AhkTest.AssertEqual(Numpy.DT_FLOAT64, integerResult.Dtype)
            NumpyAssertArrayNear([2.0, 0.0, 1.0, 2.0],
                integerResult.ToArray(), 1e-12)

            rejectedSource := false
            try Numpy.Cholesky(1)
            catch TypeError as err
                rejectedSource := InStr(err.Message, "NdArray") > 0
            AhkTest.AssertTrue(rejectedSource,
                "Cholesky must reject non-array sources")

            beforeFailure := Numpy.AllocatedMemory()
            rejectedShape := false
            try Numpy.Linalg.Cholesky(rectangular)
            catch ValueError as err
                rejectedShape := InStr(err.Message, "square") > 0
                    && InStr(err.Message, "status -4") > 0
            AhkTest.AssertTrue(rejectedShape,
                "Cholesky must transparently expose native shape errors")

            rejectedSingular := false
            try singular.Cholesky()
            catch Error as err
                rejectedSingular := InStr(err.Message, "positive definite") > 0
                    && InStr(err.Message, "status -9") > 0
            AhkTest.AssertTrue(rejectedSingular,
                "Cholesky must transparently expose native LinAlg errors")
            AhkTest.AssertEqual(beforeFailure, Numpy.AllocatedMemory(),
                "failed Cholesky calls must not retain partial results")
        } finally {
            integerResult := 0
            instanceResult := 0
            moduleResult := 0
            direct := 0
            rectangular := 0
            singular := 0
            integerSource := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "Cholesky facade must release every result")
        Numpy.Cleanup()
    }

    static TestSvdFacadeV2() {
        Numpy.Init()
        memoryBefore := DllCall(
            Numpy.Proc("cnp_ahk_allocated_memory"), "Int64")
        source := Numpy.Array([3.0, 1.0, 0.0, 2.0, 1.0, -1.0], [3, 2])

        complete := Numpy.Svd(source)
        AhkTest.AssertEqual("Array", Type(complete))
        AhkTest.AssertEqual(3, complete.Length)
        NumpyAssertArrayNear([3, 3], complete[1].Shape, 0)
        NumpyAssertArrayNear([2], complete[2].Shape, 0)
        NumpyAssertArrayNear([2, 2], complete[3].Shape, 0)
        NumpyAssertArrayNear(
            [3.2906575520321457, 2.274109248750774],
            complete[2].ToArray(), 1e-12)

        reduced := Numpy.Svd(source, false)
        NumpyAssertArrayNear([3, 2], reduced[1].Shape, 0)
        NumpyAssertArrayNear([2], reduced[2].Shape, 0)
        NumpyAssertArrayNear([2, 2], reduced[3].Shape, 0)

        singularValues := Numpy.Svd(source, true, false)
        AhkTest.AssertTrue(singularValues is Numpy.NdArray)
        NumpyAssertArrayNear(
            [3.2906575520321457, 2.274109248750774],
            singularValues.ToArray(), 1e-12)

        rejected := false
        try Numpy.Svd(source, true, true, true)
        catch ValueError as err {
            rejected := InStr(err.Message, "status -4") > 0
                && InStr(err.Message, "square") > 0
        }
        AhkTest.AssertTrue(rejected, "SVD errors must remain explicit")

        source := 0
        complete := 0
        reduced := 0
        singularValues := 0
        AhkTest.AssertEqual(memoryBefore, DllCall(
            Numpy.Proc("cnp_ahk_allocated_memory"), "Int64"))
    }

    static TestTask9SolveLstsqAndCondFacadeV2() {
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        batchA := Numpy.Array([
            3.0, 1.0, 1.0, 2.0,
            4.0, -1.0, 2.0, 3.0
        ], [2, 2, 2])
        batchB := Numpy.Array([9.0, 8.0, 7.0, 4.0], [2, 2])
        leastSquaresA := Numpy.Array([
            1.0, 0.0,
            1.0, 1.0,
            1.0, 2.0
        ], [3, 2])
        leastSquaresB := Numpy.Array([1.0, 2.0, 2.0])
        conditionSource := Numpy.Array([3.0, 1.0, 1.0, 2.0], [2, 2])
        solved := 0
        tuple := 0
        try {
            solved := Numpy.Linalg.Solve(batchA, batchB)
            NumpyAssertArrayNear([2, 2], solved.Shape, 0)
            NumpyAssertArrayNear(
                [2.0, 3.0, 1.7857142857142858, 0.14285714285714285],
                solved.ToArray(), 1e-12)

            tuple := Numpy.Linalg.Lstsq(leastSquaresA, leastSquaresB)
            AhkTest.AssertEqual("Array", Type(tuple))
            AhkTest.AssertEqual(4, tuple.Length)
            NumpyAssertArrayNear([1.1666666666666667, 0.5],
                tuple[1].ToArray(), 1e-11)
            NumpyAssertArrayNear([0.16666666666666666],
                tuple[2].ToArray(), 1e-11)
            AhkTest.AssertEqual(2, tuple[3])
            NumpyAssertArrayNear([2.6762431989952593, 0.9152717300515846],
                tuple[4].ToArray(), 1e-11)

            condition := Numpy.Linalg.Cond(conditionSource)
            AhkTest.AssertTrue(
                Abs(condition - 2.618033988749896) <= 1e-11,
                "Linalg.Cond must unwrap a scalar condition result")

            beforeFailure := Numpy.AllocatedMemory()
            invalid := Numpy.Array([1.0, 2.0])
            rejected := false
            try Numpy.Linalg.Lstsq(leastSquaresA, invalid)
            catch ValueError as err
                rejected := InStr(err.Message, "row count") > 0
                    && InStr(err.Message, "status -4") > 0
            invalid := 0
            AhkTest.AssertTrue(rejected,
                "Lstsq must expose the exact native shape error")
            AhkTest.AssertEqual(beforeFailure, Numpy.AllocatedMemory(),
                "failed Lstsq must not retain partial native results")
        } finally {
            condition := 0
            tuple := 0
            solved := 0
            conditionSource := 0
            leastSquaresB := 0
            leastSquaresA := 0
            batchB := 0
            batchA := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "Task9 solve/lstsq/cond facade must release every result")
        Numpy.Cleanup()
    }

    static TestLargeFloat64ArgsortIsAStablePermutation() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")
        memoryBefore := DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64")

        count := 10001
        shapeBuf := Buffer(8, 0)
        NumPut("Int64", count, shapeBuf, 0)
        dataBuf := Buffer(count * 8, 0)
        loop count {
            zeroIndex := A_Index - 1
            key := Mod(Mod(zeroIndex * 48271 + 17, 65521), 32)
            NumPut("Double", key, dataBuf, zeroIndex * 8)
        }

        sourceHandle := DllCall(dllPath "\cnp_ahk_from_doubles",
            "Int", 1, "Ptr", shapeBuf, "Ptr", dataBuf,
            "Int64", count, "Ptr")
        AhkTest.AssertTrue(sourceHandle != 0)
        resultHandle := DllCall(dllPath "\cnp_ahk_argsort_v2",
            "Ptr", sourceHandle,
            "Int", 0, "Int", 1, "Int", 3, "Ptr")
        AhkTest.AssertTrue(resultHandle != 0)

        outputBuf := Buffer(count * 8, 0)
        copied := DllCall(dllPath "\cnp_ahk_get_doubles",
            "Ptr", resultHandle, "Ptr", outputBuf,
            "Int64", count, "Int64")
        AhkTest.AssertEqual(count, copied)

        seen := Buffer(count, 0)
        previousKey := -1.0
        previousIndex := -1
        loop count {
            index := Round(NumGet(outputBuf, (A_Index - 1) * 8, "Double"))
            AhkTest.AssertTrue(index >= 0 && index < count,
                "argsort index out of bounds: " index)
            AhkTest.AssertEqual(0, NumGet(seen, index, "UChar"),
                "argsort index repeated: " index)
            NumPut("UChar", 1, seen, index)

            key := NumGet(dataBuf, index * 8, "Double")
            AhkTest.AssertTrue(key >= previousKey,
                "argsort keys must be nondecreasing")
            if key == previousKey
                AhkTest.AssertTrue(index > previousIndex,
                    "equal keys must preserve increasing source indices")
            previousKey := key
            previousIndex := index
        }

        AhkTest.AssertEqual(0,
            DllCall(dllPath "\cnp_ahk_free", "Ptr", resultHandle, "Int"))
        AhkTest.AssertEqual(0,
            DllCall(dllPath "\cnp_ahk_free", "Ptr", sourceHandle, "Int"))
        memoryAfter := DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64")
        AhkTest.AssertEqual(memoryBefore, memoryAfter,
            "large argsort test must release all native memory")
        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestRandom() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")

        DllCall(dllPath "\cnp_ahk_random_seed", "Int", 42)

        shapeBuf := Buffer(8, 0)
        NumPut("Int64", 100, shapeBuf, 0)

        ; random values in [0, 1)
        h := DllCall(dllPath "\cnp_ahk_random_random", "Int", 1, "Ptr", shapeBuf, "Ptr")
        AhkTest.AssertTrue(h != 0)
        size := DllCall(dllPath "\cnp_ahk_size", "Ptr", h, "Int64")
        AhkTest.AssertEqual(100, size)

        ; Check all values in [0, 1)
        outBuf := Buffer(800, 0)
        DllCall(dllPath "\cnp_ahk_get_doubles", "Ptr", h, "Ptr", outBuf, "Int64", 100, "Int64")
        allInRange := true
        loop 100 {
            v := NumGet(outBuf, (A_Index - 1) * 8, "Double")
            if v < 0.0 || v >= 1.0 {
                allInRange := false
                break
            }
        }
        AhkTest.AssertTrue(allInRange, "All random values should be in [0, 1)")
        DllCall(dllPath "\cnp_ahk_free", "Ptr", h, "Int")

        ; normal distribution
        h := DllCall(dllPath "\cnp_ahk_random_normal", "Int", 1, "Ptr", shapeBuf, "Double", 0.0, "Double", 1.0, "Ptr")
        AhkTest.AssertTrue(h != 0)
        mean := DllCall(dllPath "\cnp_ahk_mean", "Ptr", h, "Int", -1, "Double")
        AhkTest.AssertTrue(Abs(mean) < 0.5, "Normal mean should be near 0, got " mean)
        DllCall(dllPath "\cnp_ahk_free", "Ptr", h, "Int")

        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestWeightedRandomChoiceFacadeV2() {
        population := Numpy.IntArray([10, 20, 30])
        probabilities := Numpy.Array([0.0, 0.0, 1.0])

        result := Numpy.Random.Choice(
            population, [2, 2], true, probabilities)

        NumpyAssertArrayNear([2, 2], result.Shape, 0)
        NumpyAssertArrayNear([30, 30, 30, 30], result.ToArray(), 0)
    }

    static TestRandomChoiceIntegerPopulationFacadeV2() {
        result := Numpy.Random.Choice(4, 3, true, [0.0, 0.0, 0.0, 1.0])

        NumpyAssertArrayNear([3], result.Shape, 0)
        NumpyAssertArrayNear([3, 3, 3], result.ToArray(), 0)
    }

    static TestRandomChoiceSeedScalarAndErrorFacadeV2() {
        memoryBefore := DllCall(
            Numpy.Proc("cnp_ahk_allocated_memory"), "Int64")
        population := Numpy.IntArray([0, 1, 2, 3])

        Numpy.Random.Seed(0x100000001)
        first := Numpy.Random.Choice(population, 16)
        firstValues := first.ToArray()
        first := 0

        Numpy.Random.Seed(0x100000001)
        second := Numpy.Random.Choice(population, 16)
        NumpyAssertArrayNear(firstValues, second.ToArray(), 0)
        second := 0

        Numpy.Random.Seed(1)
        lowSeed := Numpy.Random.Choice(population, 16)
        highBitsObserved := false
        for index, value in lowSeed.ToArray() {
            if value != firstValues[index] {
                highBitsObserved := true
                break
            }
        }
        AhkTest.AssertTrue(highBitsObserved,
            "64-bit seeds must not be truncated to their low 32 bits")
        lowSeed := 0

        scalar := Numpy.Random.Choice(population)
        AhkTest.AssertEqual(0, scalar.Ndim)
        AhkTest.AssertEqual(1, scalar.Size)
        scalar := 0

        rejected := false
        try Numpy.Random.Choice(population, 5, false)
        catch Error as err {
            rejected := InStr(err.Message, "larger sample") > 0
                && InStr(err.Message, "without replacement") > 0
        }
        AhkTest.AssertTrue(rejected,
            "infeasible choice errors must remain explicit")

        population := 0
        AhkTest.AssertEqual(memoryBefore, DllCall(
            Numpy.Proc("cnp_ahk_allocated_memory"), "Int64"))
    }

    static TestRandomChoiceProbabilityErrorsAndTemporaryLifetimeV2() {
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        result := Numpy.Random.Choice(
            3, [2, 2], true, [0.0, 1.0, 0.0])
        NumpyAssertArrayNear([2, 2], result.Shape, 0)
        NumpyAssertArrayNear([1, 1, 1, 1], result.ToArray(), 0)
        result := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "temporary integer population and probability Array must release after success")

        quietNan := NumpyDoubleFromBits(0x7ff80000)
        positiveInfinity := NumpyDoubleFromBits(0x7ff00000)
        cases := [
            [[0.5, -0.1, 0.6], "negative"],
            [[0.5, quietNan, 0.5], "finite"],
            [[0.5, positiveInfinity, 0.5], "finite"],
            [[0.0, 0.0, 0.0], "all zero"],
            [[0.5, 0.5], "length"]
        ]
        for entry in cases {
            beforeFailure := Numpy.AllocatedMemory()
            rejected := false
            try Numpy.Random.Choice(3, 4, true, entry[1])
            catch Error as err {
                rejected := InStr(err.Message, entry[2]) > 0
                    && InStr(err.Message, "Numpy.Random.Choice") > 0
            }
            AhkTest.AssertTrue(rejected,
                "choice must translate probability error containing: " entry[2])
            AhkTest.AssertEqual(beforeFailure, Numpy.AllocatedMemory(),
                "failed temporary probability conversion must release for: " entry[2])
        }

        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "random choice temporary facade arrays must retain zero bytes")
        Numpy.Cleanup()
    }

    static TestRandomPermutationShuffleFacadeV2() {
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        source := Numpy.IntArray([0, 1, 2, 3, 4, 5, 6, 7], [4, 2])
        replaySource := Numpy.IntArray([0, 1, 2, 3, 4, 5, 6, 7], [4, 2])
        Numpy.Random.Seed(0x123456789abcdef0)
        Numpy.Random.Shuffle(source)
        first := source.ToArray()
        Numpy.Random.Seed(0x123456789abcdef0)
        Numpy.Random.Shuffle(replaySource)
        NumpyAssertArrayNear(first, replaySource.ToArray(), 0)
        NumpyAssertArrayNear([4, 2], source.Shape, 0)

        seen := Map()
        loop 4 {
            left := first[(A_Index - 1) * 2 + 1]
            right := first[(A_Index - 1) * 2 + 2]
            AhkTest.AssertEqual(left + 1, right,
                "shuffle must preserve complete first-axis rows")
            AhkTest.AssertTrue(!seen.Has(left),
                "shuffle must not duplicate a first-axis row")
            seen[left] := true
        }

        original := Numpy.IntArray([10, 11, 20, 21, 30, 31], [3, 2])
        Numpy.Random.Seed(20260804)
        permutation := Numpy.Random.Permutation(original)
        NumpyAssertArrayNear([10, 11, 20, 21, 30, 31], original.ToArray(), 0)
        permutationValues := permutation.ToArray()
        permutationRows := Map()
        loop 3 {
            left := permutationValues[(A_Index - 1) * 2 + 1]
            right := permutationValues[(A_Index - 1) * 2 + 2]
            AhkTest.AssertEqual(left + 1, right,
                "permutation must preserve complete first-axis rows")
            permutationRows[Round(left)] := true
        }
        for value in [10, 20, 30]
            AhkTest.AssertTrue(permutationRows.Has(value),
                "permutation must contain source row " value
                "; actual=" permutationValues[1] "," permutationValues[2]
                "," permutationValues[3] "," permutationValues[4]
                "," permutationValues[5] "," permutationValues[6])

        scalarPopulation := Numpy.IntArray([7])
        scalar := Numpy.Random.Choice(scalarPopulation)
        shuffleRejected := false
        try Numpy.Random.Shuffle(scalar)
        catch Error as err
            shuffleRejected := InStr(err.Message, "at least 1-dimensional") > 0
        AhkTest.AssertTrue(shuffleRejected,
            "shuffle must surface a precise 0-D error")

        permutationRejected := false
        try Numpy.Random.Permutation(scalar)
        catch Error as err
            permutationRejected := InStr(err.Message, "at least 1-dimensional") > 0
        AhkTest.AssertTrue(permutationRejected,
            "permutation must surface a precise 0-D error")

        scalar := 0
        scalarPopulation := 0
        permutation := 0
        original := 0
        replaySource := 0
        source := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "shuffle/permutation facade must retain zero bytes")
        Numpy.Cleanup()
    }

    static TestConcatenate() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")

        shapeBuf := Buffer(8, 0)
        NumPut("Int64", 3, shapeBuf, 0)

        dataA := Buffer(24, 0)
        NumPut("Double", 1.0, dataA, 0)
        NumPut("Double", 2.0, dataA, 8)
        NumPut("Double", 3.0, dataA, 16)
        ha := DllCall(dllPath "\cnp_ahk_from_doubles", "Int", 1, "Ptr", shapeBuf, "Ptr", dataA, "Int64", 3, "Ptr")

        dataB := Buffer(24, 0)
        NumPut("Double", 4.0, dataB, 0)
        NumPut("Double", 5.0, dataB, 8)
        NumPut("Double", 6.0, dataB, 16)
        hb := DllCall(dllPath "\cnp_ahk_from_doubles", "Int", 1, "Ptr", shapeBuf, "Ptr", dataB, "Int64", 3, "Ptr")

        ; concatenate -> [1,2,3,4,5,6]
        hr := DllCall(dllPath "\cnp_ahk_concatenate", "Ptr", ha, "Ptr", hb, "Int", 0, "Ptr")
        AhkTest.AssertTrue(hr != 0)
        size := DllCall(dllPath "\cnp_ahk_size", "Ptr", hr, "Int64")
        AhkTest.AssertEqual(6, size)
        outBuf := Buffer(48, 0)
        DllCall(dllPath "\cnp_ahk_get_doubles", "Ptr", hr, "Ptr", outBuf, "Int64", 6, "Int64")
        AhkTest.AssertEqual(1.0, NumGet(outBuf, 0, "Double"))
        AhkTest.AssertEqual(6.0, NumGet(outBuf, 40, "Double"))
        DllCall(dllPath "\cnp_ahk_free", "Ptr", hr, "Int")

        DllCall(dllPath "\cnp_ahk_free", "Ptr", ha, "Int")
        DllCall(dllPath "\cnp_ahk_free", "Ptr", hb, "Int")
        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestContiguousAndStridedShapeCopies() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")
        memoryBefore := DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64")

        sourceShape := Buffer(16, 0)
        NumPut("Int64", 2, sourceShape, 0)
        NumPut("Int64", 3, sourceShape, 8)
        sourceData := Buffer(48, 0)
        loop 6
            NumPut("Double", A_Index, sourceData, (A_Index - 1) * 8)
        sourceHandle := DllCall(dllPath "\cnp_ahk_from_doubles",
            "Int", 2, "Ptr", sourceShape, "Ptr", sourceData,
            "Int64", 6, "Ptr")

        flatHandle := DllCall(dllPath "\cnp_ahk_flatten", "Ptr", sourceHandle, "Ptr")
        flatOutput := Buffer(48, 0)
        DllCall(dllPath "\cnp_ahk_get_doubles",
            "Ptr", flatHandle, "Ptr", flatOutput, "Int64", 6, "Int64")
        loop 6
            AhkTest.AssertEqual(A_Index,
                NumGet(flatOutput, (A_Index - 1) * 8, "Double"),
                "contiguous flatten element " A_Index)

        transposeHandle := DllCall(dllPath "\cnp_ahk_transpose",
            "Ptr", sourceHandle, "Ptr")
        stridedFlatHandle := DllCall(dllPath "\cnp_ahk_flatten",
            "Ptr", transposeHandle, "Ptr")
        stridedOutput := Buffer(48, 0)
        DllCall(dllPath "\cnp_ahk_get_doubles",
            "Ptr", stridedFlatHandle, "Ptr", stridedOutput,
            "Int64", 6, "Int64")
        expectedTranspose := [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        for index, expectedValue in expectedTranspose
            AhkTest.AssertEqual(expectedValue,
                NumGet(stridedOutput, (index - 1) * 8, "Double"),
                "strided flatten element " index)

        leftShape := Buffer(16, 0)
        NumPut("Int64", 2, leftShape, 0)
        NumPut("Int64", 2, leftShape, 8)
        leftData := Buffer(32, 0)
        for index, value in [1.0, 2.0, 3.0, 4.0]
            NumPut("Double", value, leftData, (index - 1) * 8)
        leftHandle := DllCall(dllPath "\cnp_ahk_from_doubles",
            "Int", 2, "Ptr", leftShape, "Ptr", leftData,
            "Int64", 4, "Ptr")

        rightShape := Buffer(16, 0)
        NumPut("Int64", 2, rightShape, 0)
        NumPut("Int64", 3, rightShape, 8)
        rightData := Buffer(48, 0)
        for index, value in [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            NumPut("Double", value, rightData, (index - 1) * 8)
        rightHandle := DllCall(dllPath "\cnp_ahk_from_doubles",
            "Int", 2, "Ptr", rightShape, "Ptr", rightData,
            "Int64", 6, "Ptr")

        concatHandle := DllCall(dllPath "\cnp_ahk_concatenate",
            "Ptr", leftHandle, "Ptr", rightHandle, "Int", 1, "Ptr")
        AhkTest.AssertTrue(concatHandle != 0)
        concatShape := Buffer(16, 0)
        DllCall(dllPath "\cnp_ahk_shape", "Ptr", concatHandle,
            "Ptr", concatShape, "Int")
        AhkTest.AssertEqual(2, NumGet(concatShape, 0, "Int64"))
        AhkTest.AssertEqual(5, NumGet(concatShape, 8, "Int64"))
        concatOutput := Buffer(80, 0)
        DllCall(dllPath "\cnp_ahk_get_doubles",
            "Ptr", concatHandle, "Ptr", concatOutput,
            "Int64", 10, "Int64")
        expectedConcat := [1.0, 2.0, 5.0, 6.0, 7.0,
                           3.0, 4.0, 8.0, 9.0, 10.0]
        for index, expectedValue in expectedConcat
            AhkTest.AssertEqual(expectedValue,
                NumGet(concatOutput, (index - 1) * 8, "Double"),
                "axis-1 concatenate element " index)

        for handle in [concatHandle, rightHandle, leftHandle,
                       stridedFlatHandle, transposeHandle, flatHandle, sourceHandle]
            AhkTest.AssertEqual(0,
                DllCall(dllPath "\cnp_ahk_free", "Ptr", handle, "Int"))

        memoryAfter := DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64")
        AhkTest.AssertEqual(memoryBefore, memoryAfter,
            "shape copy tests must release all native memory")
        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestNumpyStyleMultiArrayFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        np := Numpy

        column := np.array([1.0, 2.0], [2, 1])
        row := np.IntArray([10, 20, 30])
        broadcasted := np.broadcast_arrays([column, row])
        broadcastColumn := broadcasted[1]
        broadcastRow := broadcasted[2]
        NumpyAssertArrayNear([2, 3], broadcastColumn.Shape, 0)
        NumpyAssertArrayNear(
            [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            broadcastColumn.ToArray())
        NumpyAssertArrayNear(
            [10, 20, 30, 10, 20, 30], broadcastRow.ToArray())

        x := np.array([1.5, 2.5])
        y := np.IntArray([10, 20, 30])
        grids := np.meshgrid([x, y], "xy", false, false)
        gridX := grids[1]
        gridY := grids[2]
        NumpyAssertArrayNear([3, 2], gridX.Shape, 0)
        NumpyAssertArrayNear(
            [1.5, 2.5, 1.5, 2.5, 1.5, 2.5], gridX.ToArray())
        NumpyAssertArrayNear(
            [10, 10, 20, 20, 30, 30], gridY.ToArray())

        x := 0
        y := 0
        NumpyAssertArrayNear(
            [1.5, 2.5, 1.5, 2.5, 1.5, 2.5], gridX.ToArray())
        NumpyAssertArrayNear(
            [10, 10, 20, 20, 30, 30], gridY.ToArray())

        rejected := false
        try np.meshgrid([], "invalid")
        catch ValueError as err
            rejected := InStr(err.Message, "indexing") > 0
        AhkTest.AssertTrue(rejected,
            "meshgrid must reject indexing values other than xy and ij")

        gridY := 0
        gridX := 0
        grids := 0
        broadcastRow := 0
        broadcastColumn := 0
        broadcasted := 0
        row := 0
        column := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "multi-array facade must release sources and views")
        Numpy.Cleanup()
    }

    static TestNumpyStyleSplitFamilyFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        np := Numpy

        source := np.IntArray([
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11
        ], [3, 4])

        equal := np.split(source, 2, 1)
        AhkTest.AssertEqual(2, equal.Length)
        NumpyAssertArrayNear([3, 2], equal[1].Shape, 0)
        NumpyAssertArrayNear([0, 1, 4, 5, 8, 9], equal[1].ToArray(), 0)
        NumpyAssertArrayNear([2, 3, 6, 7, 10, 11], equal[2].ToArray(), 0)

        indexed := np.split(source, [3, 1], 1)
        NumpyAssertArrayNear([3, 3], indexed[1].Shape, 0)
        NumpyAssertArrayNear([3, 0], indexed[2].Shape, 0)
        NumpyAssertArrayNear([3, 3], indexed[3].Shape, 0)
        NumpyAssertArrayNear([1, 2, 3, 5, 6, 7, 9, 10, 11],
            indexed[3].ToArray(), 0)

        uneven := np.array_split(source, 3, -1)
        NumpyAssertArrayNear([3, 2], uneven[1].Shape, 0)
        NumpyAssertArrayNear([3, 1], uneven[2].Shape, 0)
        NumpyAssertArrayNear([3, 1], uneven[3].Shape, 0)

        horizontal := np.hsplit(source, 2)
        vertical := np.vsplit(source, 3)
        NumpyAssertArrayNear([3, 2], horizontal[1].Shape, 0)
        NumpyAssertArrayNear([1, 4], vertical[1].Shape, 0)

        line := np.IntArray([0, 1, 2, 3, 4, 5])
        lineParts := np.hsplit(line, [2, 4])
        NumpyAssertArrayNear([2], lineParts[1].Shape, 0)
        NumpyAssertArrayNear([2, 3], lineParts[2].ToArray(), 0)

        cube := np.IntArray([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
        ], [2, 3, 4])
        depth := np.dsplit(cube, 2)
        NumpyAssertArrayNear([2, 3, 2], depth[1].Shape, 0)

        source := 0
        NumpyAssertArrayNear([0, 1, 4, 5, 8, 9], equal[1].ToArray(), 0)

        rejectedUneven := false
        try np.split(np.IntArray([0, 1, 2, 3]), 3)
        catch ValueError as err
            rejectedUneven := InStr(err.Message, "equal division") > 0
        AhkTest.AssertTrue(rejectedUneven,
            "split must reject an unequal integer section count")

        rejectedRank := false
        try np.vsplit(np.IntArray([0, 1, 2, 3]), 2)
        catch ValueError as err
            rejectedRank := InStr(err.Message, "2 or more dimensions") > 0
        AhkTest.AssertTrue(rejectedRank,
            "vsplit must reject arrays with fewer than two dimensions")

        depth := 0
        cube := 0
        lineParts := 0
        line := 0
        vertical := 0
        horizontal := 0
        uneven := 0
        indexed := 0
        equal := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "split-family facade must release sources and views")
        Numpy.Cleanup()
    }

    static TestNumpyStyleIndexMutationFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        np := Numpy
        source := np.IntArray([0, 1, 2, 3, 4, 5], [2, 3])

        taken := np.take(source, [2, 0], 1)
        NumpyAssertArrayNear([2, 2], taken.Shape, 0)
        NumpyAssertArrayNear([2, 0, 5, 3], taken.ToArray(), 0)

        flatTaken := np.take(source, [-1, 0])
        NumpyAssertArrayNear([5, 0], flatTaken.ToArray(), 0)

        alongIndices := np.IntArray([2, 0, 1, 1], [2, 2])
        along := np.take_along_axis(source, alongIndices, 1)
        NumpyAssertArrayNear([2, 2], along.Shape, 0)
        NumpyAssertArrayNear([2, 0, 4, 4], along.ToArray(), 0)

        flatAlongIndices := np.IntArray([5, 0])
        flatAlong := np.take_along_axis(source, flatAlongIndices)
        NumpyAssertArrayNear([2], flatAlong.Shape, 0)
        NumpyAssertArrayNear([5, 0], flatAlong.ToArray(), 0)

        compressed := np.compress([true, false, true], source, 1)
        NumpyAssertArrayNear([2, 2], compressed.Shape, 0)
        NumpyAssertArrayNear([0, 2, 3, 5], compressed.ToArray(), 0)

        deleted := np.delete(source, [2, 0], 1)
        NumpyAssertArrayNear([2, 1], deleted.Shape, 0)
        NumpyAssertArrayNear([1, 4], deleted.ToArray(), 0)

        inserted := np.insert(source, 1, np.IntArray([9, 8]), 1)
        NumpyAssertArrayNear([2, 4], inserted.Shape, 0)
        NumpyAssertArrayNear([0, 9, 1, 2, 3, 8, 4, 5],
            inserted.ToArray(), 0)

        arrayInserted := np.insert(
            source, np.IntArray([1]), np.IntArray([9, 8]), 1)
        NumpyAssertArrayNear([2, 5], arrayInserted.Shape, 0)
        NumpyAssertArrayNear([0, 9, 8, 1, 2, 3, 9, 8, 4, 5],
            arrayInserted.ToArray(), 0)

        boolMask := np.Array([1, 0, 1], [3], 1)
        boolDeleted := np.delete(source, boolMask, 1)
        NumpyAssertArrayNear([2, 1], boolDeleted.Shape, 0)
        NumpyAssertArrayNear([1, 4], boolDeleted.ToArray(), 0)

        boolDeleted := 0
        boolMask := 0
        arrayInserted := 0
        inserted := 0
        deleted := 0
        compressed := 0
        flatAlong := 0
        flatAlongIndices := 0
        along := 0
        alongIndices := 0
        flatTaken := 0
        taken := 0
        source := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "indexing facade must release arrays and temporary indices")
        Numpy.Cleanup()
    }

    static TestNpyNpzIoFacadeV2() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        np := Numpy
        testDirectory := A_ScriptDir "\..\build\test-results"
        DirCreate testDirectory
        npyPath := testDirectory "\ahk-io-roundtrip.npy"
        npzPath := testDirectory "\ahk-io-roundtrip.npz"
        if FileExist(npyPath)
            FileDelete npyPath
        if FileExist(npzPath)
            FileDelete npzPath

        source := 0
        loaded := 0
        second := 0
        archive := 0
        try {
            source := np.Array([1.25, -2.5, 3.75, 8.0], [2, 2])
            np.save(npyPath, source)
            loaded := np.load(npyPath)
            NumpyAssertArrayNear([2, 2], loaded.Shape, 0)
            NumpyAssertArrayNear(source.ToArray(), loaded.ToArray(), 0)

            second := np.Array([10.5, 20.25, -4.0])
            np.savez(
                npzPath,
                [source, second],
                ["matrix", "vector"])
            archive := np.loadz(npzPath, 2)
            AhkTest.AssertTrue(archive is Map,
                "loadz must return a name-to-array Map")
            AhkTest.AssertTrue(
                archive.Has("matrix") && archive.Has("vector"),
                "loadz must preserve NPZ member names")
            NumpyAssertArrayNear(
                source.ToArray(), archive["matrix"].ToArray(), 0)
            NumpyAssertArrayNear(
                second.ToArray(), archive["vector"].ToArray(), 0)

            rejectedCapacity := false
            try np.loadz(npzPath, 1)
            catch ValueError as err
                rejectedCapacity := InStr(err.Message, "capacity") > 0
            AhkTest.AssertTrue(rejectedCapacity,
                "loadz capacity errors must remain explicit")

            rejectedMissing := false
            try np.load(testDirectory "\missing-cnumpy-file.npy")
            catch Error as err
                rejectedMissing := InStr(err.Message, "Cannot open file") > 0
            AhkTest.AssertTrue(rejectedMissing,
                "load must preserve the native file-open error")
        } finally {
            archive := 0
            second := 0
            loaded := 0
            source := 0
            if FileExist(npyPath)
                FileDelete npyPath
            if FileExist(npzPath)
                FileDelete npzPath
        }

        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "I/O facade must release loaded and source arrays")
        Numpy.Cleanup()
    }

    static TestRawFileFacadeV2() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        np := Numpy
        testDirectory := A_ScriptDir "\..\build\test-results"
        DirCreate testDirectory
        path := testDirectory "\ahk-raw-file.bin"
        missingPath := testDirectory "\ahk-raw-file-missing.bin"
        if FileExist(path)
            FileDelete path
        if FileExist(missingPath)
            FileDelete missingPath

        source := np.IntArray([1, 2, 3, 4, 5, 6], [2, 3])
        view := np.transpose(source)
        view.tofile(path)
        loaded := np.fromfile(path, source.Dtype)
        NumpyAssertArrayNear([1, 4, 2, 5, 3, 6], loaded.ToArray(), 0)
        partial := np.fromfile(path, source.Dtype, 3, source.ItemSize)
        NumpyAssertArrayNear([4, 2, 5], partial.ToArray(), 0)

        source := 0
        view := 0
        NumpyAssertArrayNear([1, 4, 2, 5, 3, 6], loaded.ToArray(), 0)

        rejected := false
        beforeFailure := Numpy.AllocatedMemory()
        try np.fromfile(missingPath, Numpy.DT_INT64)
        catch Error as err {
            rejected := InStr(err.Message, "Numpy.FromFile") > 0
                && InStr(err.Message, "Cannot open file") > 0
        }
        AhkTest.AssertTrue(rejected,
            "fromfile facade must expose the native open error")
        AhkTest.AssertEqual(beforeFailure, Numpy.AllocatedMemory(),
            "failed fromfile must not retain native memory")

        loaded := 0
        partial := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "raw-file facade must release sources, views, and results")
        if FileExist(path)
            FileDelete path
        Numpy.Cleanup()
    }

    static TestMemmapFacadeFlushAndRelease() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        np := Numpy
        testDirectory := A_ScriptDir "\..\build\test-results"
        DirCreate testDirectory
        path := testDirectory "\ahk-memmap-roundtrip.dat"
        if FileExist(path)
            FileDelete path

        mapped := 0
        reopened := 0
        try {
            mapped := np.memmap(
                path, np.DT_FLOAT64, "w+", 5, [2, 2])
            mapped.SetData([1.25, -2.5, 3.75, 8.0])
            mapped.Flush()
            mapped := 0

            AhkTest.AssertEqual(37, FileGetSize(path),
                "w+ memmap must size the file through offset and data")
            reopened := np.memmap(
                path, np.DT_FLOAT64, "r", 5, [2, 2])
            NumpyAssertArrayNear(
                [1.25, -2.5, 3.75, 8.0], reopened.ToArray(), 0)
            reopened := 0

            FileDelete path
            AhkTest.AssertTrue(!FileExist(path),
                "releasing the final memmap must close the file")
        } finally {
            reopened := 0
            mapped := 0
            if FileExist(path)
                FileDelete path
        }

        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "memmap facade must release all native memory")
        Numpy.Cleanup()
    }

    static TestReadOnlyMemmapMutationRaisesNativeError() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        np := Numpy
        testDirectory := A_ScriptDir "\..\build\test-results"
        DirCreate testDirectory
        path := testDirectory "\ahk-memmap-readonly.dat"
        if FileExist(path)
            FileDelete path

        writable := 0
        readonly := 0
        try {
            writable := np.memmap(
                path, np.DT_FLOAT64, "w+", 0, [2])
            writable.SetData([6.25, -7.5])
            writable.Flush()
            writable := 0

            readonly := np.memmap(
                path, np.DT_FLOAT64, "r", 0, [2])
            rejected := false
            try readonly.SetItem(0, 99.0)
            catch Error as err
                rejected := InStr(err.Message, "not writeable") > 0
            AhkTest.AssertTrue(rejected,
                "read-only memmap mutation must expose the native error")

            bulkRejected := false
            try readonly.SetData([99.0, 100.0])
            catch Error as err
                bulkRejected := InStr(err.Message, "not writeable") > 0
            AhkTest.AssertTrue(bulkRejected,
                "read-only bulk mutation must expose the native error")
            NumpyAssertArrayNear([6.25, -7.5], readonly.ToArray(), 0)
        } finally {
            readonly := 0
            writable := 0
            if FileExist(path)
                FileDelete path
        }

        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "read-only memmap failure must not retain native memory")
        Numpy.Cleanup()
    }

    static TestCharSplitFacadeV2() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        np := Numpy

        split := np.char_split(
            ["alpha,beta,,gamma", "", ",leading,trailing,"], ",")
        AhkTest.AssertEqual(3, split.Length,
            "char_split must preserve the outer input count")
        AhkTest.AssertEqual(4, split[1].Length,
            "char_split must return every token")
        AhkTest.AssertEqual("alpha", split[1][1])
        AhkTest.AssertEqual("beta", split[1][2])
        AhkTest.AssertEqual("", split[1][3])
        AhkTest.AssertEqual("gamma", split[1][4])
        AhkTest.AssertEqual(1, split[2].Length)
        AhkTest.AssertEqual("", split[2][1])
        AhkTest.AssertEqual(4, split[3].Length)
        AhkTest.AssertEqual("", split[3][1])
        AhkTest.AssertEqual("leading", split[3][2])
        AhkTest.AssertEqual("trailing", split[3][3])
        AhkTest.AssertEqual("", split[3][4])

        whitespace := np.char_split(["  one`ttwo  three "], unset, 1)
        AhkTest.AssertEqual(2, whitespace[1].Length)
        AhkTest.AssertEqual("one", whitespace[1][1])
        AhkTest.AssertEqual("two  three ", whitespace[1][2])

        split := 0
        whitespace := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "char_split facade must release its native string result")
        Numpy.Cleanup()
    }

    static TestStringFunctionCallbacks() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        values := Numpy.Array([1.25, -2.5])
        calls := []
        formatter := (array) => (
            calls.Push(array.Size), "<custom " array.Size ">")

        try {
            Numpy.Set_String_Function(formatter, false)
            AhkTest.AssertEqual("<custom 2>", values.ToString())
            AhkTest.AssertEqual(1, calls.Length)
            AhkTest.AssertEqual(2, calls[1])

            Numpy.Set_String_Function((array) => "<repr " array.Ndim ">")
            AhkTest.AssertEqual("<repr 1>", values.Repr())
            AhkTest.AssertEqual("<custom 2>", values.ToString())

            Numpy.Set_String_Function(
                (array) => NumpyFoundationTest.ThrowingStringFunction(array),
                false)
            exposed := false
            try values.ToString()
            catch Error as err
                exposed := InStr(err.Message, "formatter exploded") > 0
            AhkTest.AssertTrue(exposed,
                "AHK string callback exceptions must remain visible")
        } finally {
            Numpy.Set_String_Function(, true)
            Numpy.Set_String_Function(, false)
            values := 0
        }

        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "string callbacks must not retain native strings or arrays")
        Numpy.Cleanup()
    }

    static ThrowingStringFunction(array) {
        throw Error("formatter exploded")
    }

    static TestStringRegexSafeEvalFacadeV2() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        parsed := Numpy.FromString(
            "1, 2, -3.5e1, 4", Numpy.DT_FLOAT64, -1, ",")
        NumpyAssertArrayNear([1.0, 2.0, -35.0, 4.0], parsed.ToArray(), 0)
        limited := Numpy.FromString(
            "255|0|17|9", Numpy.DT_UINT8, 3, "|")
        NumpyAssertArrayNear([255, 0, 17], limited.ToArray(), 0)

        text := "A=1.5 B=-2`nnoise`nA=3e2 B=4.25`nA=-7 B=9`n"
        pattern := "A=([+-]?[0-9.eE]+)\s+B=([+-]?[0-9.eE]+)"
        fields := Numpy.FromRegex(
            text, pattern, ["a", "b"],
            [Numpy.DT_FLOAT64, Numpy.DT_FLOAT32], 2)
        AhkTest.AssertTrue(fields is Map,
            "fromregex must return a field-name map")
        AhkTest.AssertEqual(2, fields.Count)
        firstField := fields["a"]
        secondField := fields["b"]
        fields := 0
        NumpyAssertArrayNear([1.5, 300.0], firstField.ToArray(), 0)
        NumpyAssertArrayNear([-2.0, 4.25], secondField.ToArray(), 0)

        AhkTest.AssertEqual(1.25, Numpy.SafeEval("((1.25))"))
        AhkTest.AssertEqual(255.0, Numpy.SafeEval("0xff"))

        parseRejected := false
        try Numpy.FromString("1,bad,3", Numpy.DT_FLOAT64, -1, ",")
        catch ValueError as err
            parseRejected := InStr(err.Message, "invalid numeric token") > 0
        AhkTest.AssertTrue(parseRejected,
            "fromstring must expose invalid tokens")

        regexRejected := false
        try Numpy.FromRegex("1", "(", ["value"], [Numpy.DT_FLOAT64])
        catch ValueError as err
            regexRejected := InStr(err.Message, "pattern") > 0
        AhkTest.AssertTrue(regexRejected,
            "fromregex must expose invalid patterns")

        evalRejected := false
        try Numpy.SafeEval("1 + 2")
        catch ValueError as err
            evalRejected := InStr(err.Message, "trailing") > 0
        AhkTest.AssertTrue(evalRejected,
            "safe_eval must reject arithmetic syntax like NumPy 1.25")

        secondField := 0
        firstField := 0
        limited := 0
        parsed := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "string/regex/parser facade must release every native allocation")
        Numpy.Cleanup()
    }

    static TestPolynomialBasisConversionFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        coefficients := Numpy.Array([1.0, 2.0, 3.0])
        rankTwo := Numpy.Array([1.0, 2.0, 3.0, 4.0], [2, 2])

        NumpyAssertArrayNear(
            [2.5, 2.0, 1.5], Numpy.Poly2Cheb(coefficients).ToArray())
        NumpyAssertArrayNear(
            [-2.0, 2.0, 6.0], Numpy.Cheb2Poly(coefficients).ToArray())
        NumpyAssertArrayNear(
            [2.0, 2.0, 2.0], Numpy.Poly2Leg(coefficients).ToArray())
        NumpyAssertArrayNear(
            [-0.5, 2.0, 4.5], Numpy.Leg2Poly(coefficients).ToArray())

        rejected := false
        try Numpy.Poly2Cheb(rankTwo)
        catch Error as err
            rejected := InStr(err.Message, "not 1-d") > 0
        AhkTest.AssertTrue(rejected,
            "polynomial facade must expose native rank errors")

        rankTwo := 0
        coefficients := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "polynomial conversions must release every native result")
        Numpy.Cleanup()
    }

    static TestPolynomialBasisMultiplicationFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        left := Numpy.Array([1.0, 2.0, 3.0])
        right := Numpy.Array([3.0, 2.0])
        rankTwo := Numpy.Array([1.0, 2.0, 3.0, 4.0], [2, 2])

        NumpyAssertArrayNear(
            [5.0, 11.0, 11.0, 3.0],
            Numpy.ChebMul(left, right).ToArray())
        NumpyAssertArrayNear(
            [4.333333333333333, 10.4, 11.666666666666666, 3.6],
            Numpy.LegMul(left, right).ToArray())
        NumpyAssertArrayNear(
            [11.0, 32.0, 13.0, 6.0],
            Numpy.HermMul(left, right).ToArray())
        NumpyAssertArrayNear(
            [7.0, 12.0, -7.0, 18.0],
            Numpy.LagMul(left, right).ToArray())

        rejected := false
        try Numpy.ChebMul(rankTwo, right)
        catch Error as err
            rejected := InStr(
                err.Message, "must be one-dimensional") > 0
        AhkTest.AssertTrue(rejected,
            "polynomial multiplication facade must expose native rank errors")

        rankTwo := 0
        right := 0
        left := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "polynomial multiplication must release every native result")
        Numpy.Cleanup()
    }

    static TestChebyshevCoefficientArithmeticFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        left := Numpy.Array([1.0, 2.0, 0.0])
        right := Numpy.Array([3.0, -2.0])
        rankTwo := Numpy.Array([1.0, 2.0, 3.0, 4.0], [2, 2])

        NumpyAssertArrayNear(
            [4.0], Numpy.ChebAdd(left, right).ToArray())
        NumpyAssertArrayNear(
            [-2.0, 4.0], Numpy.ChebSub(left, right).ToArray())

        rejected := false
        try Numpy.ChebAdd(rankTwo, right)
        catch Error as err
            rejected := InStr(
                err.Message, "must be one-dimensional") > 0
        AhkTest.AssertTrue(rejected,
            "Chebyshev arithmetic facade must expose native rank errors")

        rankTwo := 0
        right := 0
        left := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "Chebyshev arithmetic must release every native result")
        Numpy.Cleanup()
    }

    static TestPowerPolynomialAdditionFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        left := Numpy.Array([1.0, 2.0, 0.0])
        right := Numpy.Array([3.0, -2.0])
        rankTwo := Numpy.Array([1.0, 2.0, 3.0, 4.0], [2, 2])

        NumpyAssertArrayNear(
            [1.0, 5.0, -2.0], Numpy.PolyAdd(left, right).ToArray())
        NumpyAssertArrayNear(
            [1.0, -1.0, 2.0], Numpy.PolySub(left, right).ToArray())

        rejected := false
        try Numpy.PolySub(rankTwo, right)
        catch Error as err
            rejected := InStr(
                err.Message, "must be one-dimensional") > 0
        AhkTest.AssertTrue(rejected,
            "power polynomial facade must expose native rank errors")

        rankTwo := 0
        right := 0
        left := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "power polynomial addition must release every native result")
        Numpy.Cleanup()
    }

    static TestPowerPolynomialMultiplicationFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        left := Numpy.Array([1.0, 2.0])
        right := Numpy.Array([3.0, 2.0])
        rankTwo := Numpy.Array([1.0, 2.0, 3.0, 4.0], [2, 2])

        NumpyAssertArrayNear(
            [3.0, 8.0, 4.0], Numpy.PolyMul(left, right).ToArray())

        rejected := false
        try Numpy.PolyMul(rankTwo, right)
        catch Error as err
            rejected := InStr(
                err.Message, "must be one-dimensional") > 0
        AhkTest.AssertTrue(rejected,
            "power polynomial multiplication must expose native rank errors")

        rankTwo := 0
        right := 0
        left := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "power polynomial multiplication must release native results")
        Numpy.Cleanup()
    }

    static TestPowerPolynomialDivisionFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        dividend := Numpy.Array([3.0, 5.0, 2.0])
        divisor := Numpy.Array([2.0, 1.0])
        rankTwo := Numpy.Array([1.0, 2.0, 3.0, 4.0], [2, 2])

        results := Numpy.PolyDiv(dividend, divisor)
        AhkTest.AssertEqual(2, results.Length,
            "power polynomial division must return quotient and remainder")
        NumpyAssertArrayNear([1.5, 1.75], results[1].ToArray())
        NumpyAssertArrayNear([0.25], results[2].ToArray())

        rejected := false
        try Numpy.PolyDiv(rankTwo, divisor)
        catch Error as err
            rejected := InStr(
                err.Message, "must be one-dimensional") > 0
        AhkTest.AssertTrue(rejected,
            "power polynomial division must expose native rank errors")

        results := 0
        rankTwo := 0
        divisor := 0
        dividend := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "power polynomial division must release both native results")
        Numpy.Cleanup()
    }

    static TestPowerPolynomialEvaluationFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        coefficients := Numpy.Array([3.0, 0.0, 1.0])
        points := Numpy.Array([2.0, 5.0])
        rankTwo := Numpy.Array([1.0, 2.0, 3.0, 4.0], [2, 2])

        NumpyAssertArrayNear(
            [13.0, 76.0], Numpy.PolyVal(coefficients, points).ToArray())

        rejected := false
        try Numpy.PolyVal(rankTwo, points)
        catch Error as err
            rejected := InStr(
                err.Message, "must be one-dimensional") > 0
        AhkTest.AssertTrue(rejected,
            "power polynomial evaluation must expose native rank errors")

        rankTwo := 0
        points := 0
        coefficients := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "power polynomial evaluation must release native results")
        Numpy.Cleanup()
    }

    static TestPolynomialBasisEvaluationFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        coefficients := Numpy.Array([1.0, 2.0, 3.0])
        points := Numpy.Array([0.0, 1.0, 2.0])
        empty := Numpy.Zeros([0])

        NumpyAssertArrayNear(
            [-2.0, 6.0, 26.0],
            Numpy.ChebVal(points, coefficients).ToArray(), 1e-12)
        NumpyAssertArrayNear(
            [-0.5, 6.0, 21.5],
            Numpy.LegVal(points, coefficients).ToArray(), 1e-12)
        NumpyAssertArrayNear(
            [-5.0, 11.0, 51.0],
            Numpy.HermVal(points, coefficients).ToArray(), 1e-12)
        NumpyAssertArrayNear(
            [6.0, -0.5, -4.0],
            Numpy.LagVal(points, coefficients).ToArray(), 1e-12)

        rejected := false
        try Numpy.ChebVal(points, empty)
        catch Error as err
            rejected := InStr(err.Message, "coefficient axis is empty") > 0
        AhkTest.AssertTrue(rejected,
            "basis evaluation must expose empty coefficient errors")

        empty := 0
        points := 0
        coefficients := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "basis evaluation must release native results")
        Numpy.Cleanup()
    }

    static TestPowerPolynomialFitFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        x := Numpy.Array([0.0, 1.0, 2.0, 3.0])
        y := Numpy.Array([1.0, 2.0, 5.0, 10.0])
        rankTwoX := Numpy.Array([0.0, 1.0, 2.0, 3.0], [2, 2])

        NumpyAssertArrayNear(
            [1.0, 0.0, 1.0], Numpy.PolyFit(x, y, 2).ToArray(), 1e-10)

        rejected := false
        try Numpy.PolyFit(rankTwoX, y, 2)
        catch Error as err
            rejected := InStr(err.Message, "must be one-dimensional") > 0
        AhkTest.AssertTrue(rejected,
            "power polynomial fit must expose native rank errors")

        rankTwoX := 0
        y := 0
        x := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "power polynomial fit must release native results")
        Numpy.Cleanup()
    }

    static TestPolynomialBasisFitFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        x := Numpy.Array([0.0, 1.0, 2.0, 3.0])
        y := Numpy.Array([1.0, 2.0, 5.0, 10.0])
        rankTwoX := Numpy.Array([0.0, 1.0, 2.0, 3.0], [2, 2])

        NumpyAssertArrayNear(
            [1.5, 0.0, 0.5], Numpy.ChebFit(x, y, 2).ToArray(), 1e-10)
        NumpyAssertArrayNear(
            [4.0 / 3.0, 0.0, 2.0 / 3.0],
            Numpy.LegFit(x, y, 2).ToArray(), 1e-10)
        NumpyAssertArrayNear(
            [1.5, 0.0, 0.25], Numpy.HermFit(x, y, 2).ToArray(), 1e-10)
        NumpyAssertArrayNear(
            [3.0, -4.0, 2.0], Numpy.LagFit(x, y, 2).ToArray(), 1e-10)

        rejected := false
        try Numpy.ChebFit(rankTwoX, y, 2)
        catch Error as err
            rejected := InStr(err.Message, "must be one-dimensional") > 0
        AhkTest.AssertTrue(rejected,
            "basis fit must expose native rank errors")

        rankTwoX := 0
        y := 0
        x := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "basis fit must release native results")
        Numpy.Cleanup()
    }

    static TestPowerPolynomialConstructionFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        roots := Numpy.Array([1.0, 2.0, 3.0])
        matrix := Numpy.Array([1.0, 2.0, 3.0, 4.0], [2, 2])
        nonSquare := Numpy.Array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])

        NumpyAssertArrayNear(
            [1.0, -6.0, 11.0, -6.0], Numpy.Poly(roots).ToArray())
        NumpyAssertArrayNear(
            [1.0, -5.0, -2.0], Numpy.Poly(matrix).ToArray(), 1e-12)

        rejected := false
        try Numpy.Poly(nonSquare)
        catch Error as err
            rejected := InStr(
                err.Message, "non-empty square 2d array") > 0
        AhkTest.AssertTrue(rejected,
            "power polynomial construction must expose native shape errors")

        nonSquare := 0
        matrix := 0
        roots := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "power polynomial construction must release native results")
        Numpy.Cleanup()
    }

    static TestPowerPolynomialFromRootsFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        roots := Numpy.Array([1.0, 2.0, 3.0])
        rankTwo := Numpy.Array([1.0, 2.0], [1, 2])

        NumpyAssertArrayNear(
            [-6.0, 11.0, -6.0, 1.0],
            Numpy.PolyFromRoots(roots).ToArray())

        rejected := false
        try Numpy.PolyFromRoots(rankTwo)
        catch Error as err
            rejected := InStr(err.Message, "must be one-dimensional") > 0
        AhkTest.AssertTrue(rejected,
            "polyfromroots must expose native rank errors")

        rankTwo := 0
        roots := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "polyfromroots must release native results")
        Numpy.Cleanup()
    }

    static TestPowerPolynomialRootsFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        coefficients := Numpy.Array([2.0, -3.0, 1.0])
        rankTwo := Numpy.Array([1.0, 2.0], [1, 2])

        NumpyAssertArrayNear(
            [1.0, 2.0],
            Numpy.PolyRoots(coefficients).ToArray(),
            1e-12)

        rejected := false
        try Numpy.PolyRoots(rankTwo)
        catch Error as err
            rejected := InStr(err.Message, "Coefficient array is not 1-d") > 0
        AhkTest.AssertTrue(rejected,
            "polyroots must expose native rank errors")

        rankTwo := 0
        coefficients := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "polyroots must release native results")
        Numpy.Cleanup()
    }

    static TestLegacyPowerPolynomialRootsFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        coefficients := Numpy.Array([1.0, -3.0, 2.0])
        rankTwo := Numpy.Array([1.0, 2.0], [1, 2])

        NumpyAssertArrayNear(
            [2.0, 1.0],
            Numpy.Roots(coefficients).ToArray(),
            1e-12)

        rejected := false
        try Numpy.Roots(rankTwo)
        catch Error as err
            rejected := InStr(err.Message, "Input must be a rank-1 array") > 0
        AhkTest.AssertTrue(rejected,
            "roots must expose native rank errors")

        rankTwo := 0
        coefficients := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "roots must release native results")
        Numpy.Cleanup()
    }

    static TestChebyshevPointFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        NumpyAssertArrayNear(
            [-0.8660254037844386, 0.0, 0.8660254037844386],
            Numpy.ChebPts1(3).ToArray(), 1e-14)
        NumpyAssertArrayNear(
            [-1.0, 0.0, 1.0],
            Numpy.ChebPts2(3).ToArray(), 1e-14)

        rejected := false
        try Numpy.ChebPts2(1)
        catch Error as err
            rejected := InStr(err.Message, "npts must be >= 2") > 0
        AhkTest.AssertTrue(rejected,
            "Chebyshev point facade must expose invalid counts")

        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "Chebyshev point results must release every native allocation")
        Numpy.Cleanup()
    }

    static TestPolynomialCalculusFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        coefficients := Numpy.Array([1.0, 2.0, 3.0])
        constants := Numpy.Array([2.0])

        NumpyAssertArrayNear(
            [2.0, 2.0], Numpy.PolyDer(coefficients).ToArray())
        NumpyAssertArrayNear(
            [1.0 / 3.0, 1.0, 3.0, 2.0],
            Numpy.PolyInt(coefficients, 1, constants).ToArray())
        NumpyAssertArrayNear(
            [2.0, 12.0], Numpy.ChebDer(coefficients).ToArray())
        NumpyAssertArrayNear(
            [0.5, -0.5, 0.5, 0.5],
            Numpy.ChebInt(coefficients).ToArray())
        NumpyAssertArrayNear(
            [2.0, 9.0], Numpy.LegDer(coefficients).ToArray())
        NumpyAssertArrayNear(
            [4.0, 12.0], Numpy.HermDer(coefficients).ToArray())
        NumpyAssertArrayNear(
            [-5.0, -3.0], Numpy.LagDer(coefficients).ToArray())

        rejected := false
        try Numpy.ChebDer(coefficients, -1)
        catch Error as err
            rejected := InStr(err.Message, "order must be non-negative") > 0
        AhkTest.AssertTrue(rejected,
            "polynomial calculus facade must expose native order errors")

        constants := 0
        coefficients := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "polynomial calculus facade must release every native result")
        Numpy.Cleanup()
    }

    static TestSpecialFunctionFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        values := Numpy.Array([-1.0, 0.0, 1.0])
        inverseValues := Numpy.Array([-0.5, 0.0, 0.5])
        probabilities := Numpy.Array([0.25, 0.5, 0.75])

        NumpyAssertArrayNear(
            [-0.8427007929497149, 0.0, 0.8427007929497149],
            Numpy.Erf(values).ToArray(), 1e-14)
        NumpyAssertArrayNear(
            [1.842700792949715, 1.0, 0.1572992070502851],
            Numpy.Erfc(values).ToArray(), 1e-14)
        NumpyAssertArrayNear(
            [-0.4769362762044699, 0.0, 0.4769362762044699],
            Numpy.ErfInv(inverseValues).ToArray(), 1e-14)
        NumpyAssertArrayNear(
            [0.2689414213699951, 0.5, 0.7310585786300049],
            Numpy.Expit(values).ToArray(), 1e-14)
        NumpyAssertArrayNear(
            [-1.0986122886681098, 0.0, 1.0986122886681098],
            Numpy.Logit(probabilities).ToArray(), 1e-14)

        probabilities := 0
        inverseValues := 0
        values := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "special function facade must release every native result")
        Numpy.Cleanup()
    }

    static TestGammaFunctionFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        values := Numpy.Array([0.5, 1.0, 5.0])
        floatValues := Numpy.Array([0.5, 1.0, 5.0], , Numpy.DT_FLOAT32)
        left := Numpy.Array([0.5, 2.0], [2, 1])
        right := Numpy.Array([1.0, 2.0, 3.0], [1, 3])

        gammaResult := Numpy.Gamma(values)
        gammaFloatResult := Numpy.Gamma(floatValues)
        gammaLnResult := Numpy.GammaLn(values)
        betaResult := Numpy.Beta(left, right)
        NumpyAssertArrayNear(
            [1.772453850905516, 1.0, 24.0],
            gammaResult.ToArray(), 1e-13)
        AhkTest.AssertEqual(Numpy.DT_FLOAT32, gammaFloatResult.Dtype)
        NumpyAssertArrayNear(
            [0.5723649429247001, 0.0, 3.178053830347946],
            gammaLnResult.ToArray(), 1e-13)
        NumpyAssertArrayNear([2, 3], betaResult.Shape, 0)
        NumpyAssertArrayNear(
            [2.0, 4.0 / 3.0, 16.0 / 15.0, 0.5, 1.0 / 6.0, 1.0 / 12.0],
            betaResult.ToArray(), 1e-13)

        betaResult := 0
        gammaLnResult := 0
        gammaFloatResult := 0
        gammaResult := 0
        right := 0
        left := 0
        floatValues := 0
        values := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "gamma facade must release every native result")
        Numpy.Cleanup()
    }

    static TestCombinatorialFunctionFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        values := Numpy.Array([-1.0, 0.0, 0.5, 5.0])
        integerValues := Numpy.Array([0, 5, 20], , Numpy.DT_INT64)
        left := Numpy.Array([5.0, 5.5], [2, 1])
        right := Numpy.Array([2.0, 2.5], [1, 2])

        exactFactorial := Numpy.Factorial(integerValues, true)
        NumpyAssertArrayNear(
            [0.0, 1.0, 0.886226925452758, 120.0],
            Numpy.Factorial(values).ToArray(), 1e-13)
        AhkTest.AssertEqual(Numpy.DT_LONGLONG, exactFactorial.Dtype)
        NumpyAssertArrayNear(
            [1, 120, 2432902008176640000],
            exactFactorial.ToArray(), 0)
        NumpyAssertArrayNear(
            [10.0, 10.86497744840672, 12.375, 14.4375],
            Numpy.Comb(left, right).ToArray(), 1e-11)
        NumpyAssertArrayNear(
            [20.0, 36.1081333470564, 24.75, 47.980879635840736],
            Numpy.Perm(left, right).ToArray(), 1e-11)

        exactFactorial := 0
        right := 0
        left := 0
        integerValues := 0
        values := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "combinatorial facade must release every native result")
        Numpy.Cleanup()
    }

    static TestBesselFunctionFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        values := Numpy.Array([-5.0, -1.0, -0.0, 0.0, 1.0, 5.0])

        i0Values := Numpy.I0(values)
        j0Values := Numpy.J0(values)
        j1Values := Numpy.J1(values)
        NumpyAssertArrayNear(
            [27.239871823604442, 1.2660658777520082,
             1.0, 1.0, 1.2660658777520082, 27.239871823604442],
            i0Values.ToArray(), 4e-14)
        NumpyAssertArrayNear(
            [-0.1775967713143383, 0.7651976865579665,
             1.0, 1.0, 0.7651976865579665, -0.1775967713143383],
            j0Values.ToArray(), 4e-14)
        NumpyAssertArrayNear(
            [0.3275791375914653, -0.44005058574493355,
             -0.0, 0.0, 0.44005058574493355, -0.3275791375914653],
            j1Values.ToArray(), 4e-14)

        j1Values := 0
        j0Values := 0
        i0Values := 0
        values := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "Bessel facade must release every native result")
        Numpy.Cleanup()
    }

    static TestHalfBitConversionFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        AhkTest.AssertEqual(0x3C00, Numpy.FloatToHalf(1.0))
        AhkTest.AssertEqual(0x8000, Numpy.FloatToHalf(-0.0))
        AhkTest.AssertEqual(0x7BFF, Numpy.FloatToHalf(65504.0))
        AhkTest.AssertEqual(0x7C00, Numpy.FloatToHalf(65520.0))
        AhkTest.AssertEqual(1.0, Numpy.HalfToFloat(0x3C00))
        AhkTest.AssertEqual(2.0 ** -24, Numpy.HalfToFloat(0x0001))

        signalingNan := Buffer(8, 0)
        NumPut("Double", Numpy.HalfToFloat(0x7C01), signalingNan, 0)
        AhkTest.AssertEqual(0, NumGet(signalingNan, 0, "UInt"))
        AhkTest.AssertEqual(0x7FF00400,
            NumGet(signalingNan, 4, "UInt"))

        rejected := false
        try Numpy.HalfToFloat(0x10000)
        catch ValueError as err
            rejected := InStr(err.Message, "16-bit") > 0
        AhkTest.AssertTrue(rejected,
            "HalfToFloat must reject values outside 16-bit storage")

        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "half conversion facade must not retain native memory")
        Numpy.Cleanup()
    }

    static TestFinfoFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        AhkTest.AssertEqual(2.0 ** -10,
            Numpy.FinfoEps(Numpy.DT_FLOAT16))
        AhkTest.AssertEqual(65504.0,
            Numpy.FinfoMax(Numpy.DT_FLOAT16))
        AhkTest.AssertEqual(-65504.0,
            Numpy.FinfoMin(Numpy.DT_FLOAT16))
        AhkTest.AssertEqual(2.0 ** -23,
            Numpy.FinfoEps(Numpy.DT_COMPLEX64))
        AhkTest.AssertEqual(-1.7976931348623157e308,
            Numpy.FinfoMin(Numpy.DT_FLOAT64))

        rejected := false
        try Numpy.FinfoMax(Numpy.DT_INT32)
        catch TypeError as err
            rejected := InStr(err.Message, "inexact dtype") > 0
        AhkTest.AssertTrue(rejected,
            "Finfo facade must expose invalid dtype errors")
        AhkTest.AssertEqual(2.0 ** -52,
            Numpy.FinfoEps(Numpy.DT_FLOAT64),
            "valid finfo call must not inherit a prior native error")

        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "finfo facade must not retain native memory")
        Numpy.Cleanup()
    }

    static TestTypenameFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        AhkTest.AssertEqual("bool", Numpy.Typename(Numpy.DT_BOOL))
        AhkTest.AssertEqual("signed char",
            Numpy.Typename(Numpy.DT_INT8))
        AhkTest.AssertEqual("unsigned long long integer",
            Numpy.Typename(Numpy.DT_ULONGLONG))
        AhkTest.AssertEqual("double precision",
            Numpy.Typename(Numpy.DT_FLOAT64))
        AhkTest.AssertEqual("complex long double precision",
            Numpy.Typename(Numpy.DT_COMPLEX_LONGDOUBLE))
        AhkTest.AssertEqual("object",
            Numpy.Typename(Numpy.DT_OBJECT))

        rejected := false
        try Numpy.Typename(Numpy.DT_FLOAT16)
        catch NumpyKeyError as err
            rejected := InStr(err.Message, "NumPy typename") > 0
        AhkTest.AssertTrue(rejected,
            "Typename facade must preserve unsupported typecode errors")
        AhkTest.AssertEqual("single precision",
            Numpy.Typename(Numpy.DT_FLOAT32),
            "valid typename call must not inherit a prior native error")

        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "typename facade must not retain native memory")
        Numpy.Cleanup()
    }

    static TestIinfoFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        AhkTest.AssertEqual(-128, Numpy.IinfoMin(Numpy.DT_INT8))
        AhkTest.AssertEqual(127, Numpy.IinfoMax(Numpy.DT_INT8))
        AhkTest.AssertEqual(4294967295,
            Numpy.IinfoMax(Numpy.DT_UINT32))
        AhkTest.AssertEqual(-9223372036854775808,
            Numpy.IinfoMin(Numpy.DT_INT64))
        AhkTest.AssertEqual(9223372036854775807,
            Numpy.IinfoMax(Numpy.DT_INT64))

        unsignedMaximum := Numpy.IinfoMax(Numpy.DT_UINT64)
        AhkTest.AssertEqual("String", Type(unsignedMaximum))
        AhkTest.AssertEqual("18446744073709551615", unsignedMaximum)

        rejected := false
        try Numpy.IinfoMax(Numpy.DT_FLOAT64)
        catch TypeError as err
            rejected := InStr(err.Message, "integer dtype") > 0
        AhkTest.AssertTrue(rejected,
            "Iinfo facade must expose invalid dtype errors")
        AhkTest.AssertEqual(65535, Numpy.IinfoMax(Numpy.DT_UINT16),
            "valid iinfo call must not inherit a prior native error")

        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "iinfo facade must not retain native memory")
        Numpy.Cleanup()
    }

    static TestDtypeMetadataFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        AhkTest.AssertEqual(1, Numpy.DtypeItemsize(Numpy.DT_BOOL))
        AhkTest.AssertEqual(8,
            Numpy.DtypeItemsize(Numpy.DT_LONGDOUBLE))
        AhkTest.AssertEqual(16,
            Numpy.DtypeItemsize(Numpy.DT_COMPLEX_LONGDOUBLE))
        AhkTest.AssertEqual(8,
            Numpy.DtypeItemsize(Numpy.DT_DATETIME))
        AhkTest.AssertEqual("u", Numpy.DtypeKind(Numpy.DT_UINT64))
        AhkTest.AssertEqual("M", Numpy.DtypeKind(Numpy.DT_DATETIME))
        AhkTest.AssertEqual("m", Numpy.DtypeKind(Numpy.DT_TIMEDELTA))
        AhkTest.AssertEqual(Numpy.DT_FLOAT64, Numpy.DtypeNew(Numpy.DT_FLOAT64))
        AhkTest.AssertEqual(Numpy.DT_FLOAT64, Numpy.DtypeFromChar("d"))
        AhkTest.AssertEqual(Numpy.DT_COMPLEX64,
            Numpy.DtypeFromString("complex64"))

        rejected := false
        try Numpy.DtypeKind(0)
        catch TypeError as err
            rejected := InStr(err.Message, "valid CNP_TYPE") > 0
        AhkTest.AssertTrue(rejected,
            "dtype metadata facade must expose invalid dtype errors")
        rejected := false
        try Numpy.DtypeFromChar("!")
        catch TypeError as err
            rejected := InStr(err.Message, "dtype character") > 0
        AhkTest.AssertTrue(rejected,
            "dtype character facade must expose invalid descriptor errors")
        AhkTest.AssertEqual(4, Numpy.DtypeItemsize(Numpy.DT_FLOAT32),
            "valid dtype metadata must not inherit a prior native error")

        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "dtype metadata facade must not retain native memory")
        Numpy.Cleanup()
    }

    static TestPromotionFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        AhkTest.AssertEqual(Numpy.DT_INT16,
            Numpy.PromoteTypes(Numpy.DT_INT8, Numpy.DT_UINT8))
        AhkTest.AssertEqual(Numpy.DT_TIMEDELTA,
            Numpy.PromoteTypes(Numpy.DT_BOOL, Numpy.DT_TIMEDELTA))

        int8Values := Numpy.Array([0], [1], Numpy.DT_INT8)
        uint8Values := Numpy.Array([0], [1], Numpy.DT_UINT8)
        float16Values := Numpy.Array([0], [1], Numpy.DT_FLOAT16)
        AhkTest.AssertEqual(Numpy.DT_FLOAT16,
            Numpy.ResultType([int8Values, uint8Values, float16Values]))

        largeScalar := Numpy.Array([65536], [1], Numpy.DT_LONGLONG)
        float32Values := Numpy.Array([0], [1], Numpy.DT_FLOAT32)
        AhkTest.AssertEqual(Numpy.DT_FLOAT64,
            Numpy.ResultType([largeScalar, float32Values]))

        rejectedPromotion := false
        try Numpy.PromoteTypes(Numpy.DT_BOOL, Numpy.DT_DATETIME)
        catch TypeError as err
            rejectedPromotion := InStr(err.Message, "common dtype") > 0
        AhkTest.AssertTrue(rejectedPromotion,
            "PromoteTypes must expose incompatible dtype errors")

        rejectedEmpty := false
        try Numpy.ResultType([])
        catch Error as err
            rejectedEmpty := InStr(err.Message, "at least one array") > 0
        AhkTest.AssertTrue(rejectedEmpty,
            "ResultType must expose empty input errors")

        int8Values := 0
        uint8Values := 0
        float16Values := 0
        largeScalar := 0
        float32Values := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "promotion facade must not retain native memory")
        Numpy.Cleanup()
    }

    static TestCastingFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        AhkTest.AssertTrue(
            Numpy.CanCast(Numpy.DT_INT64, Numpy.DT_FLOAT64, "safe"),
            "int64 must cast safely to float64")
        AhkTest.AssertTrue(
            Numpy.CanCast(Numpy.DT_INT64, Numpy.DT_LONGLONG, "no"),
            "equivalent int64 aliases must satisfy no casting")
        AhkTest.AssertTrue(
            Numpy.CanCast(Numpy.DT_LONGLONG, Numpy.DT_INT64, "no"),
            "equivalent int64 aliases must satisfy no casting in both directions")
        AhkTest.AssertFalse(
            Numpy.CanCast(Numpy.DT_UNICODE, Numpy.DT_STRING, "safe"),
            "unicode must not cast safely to bytes")

        rejectedCasting := false
        try Numpy.CanCast(Numpy.DT_INT8, Numpy.DT_INT16, "invalid")
        catch ValueError as err
            rejectedCasting := InStr(err.Message, "valid casting") > 0
        AhkTest.AssertTrue(rejectedCasting,
            "CanCast must reject unknown casting names explicitly")

        rejectedDtype := false
        try Numpy.CanCast(0, Numpy.DT_INT16)
        catch TypeError as err
            rejectedDtype := InStr(err.Message, "valid CNP_TYPE") > 0
        AhkTest.AssertTrue(rejectedDtype,
            "CanCast must expose invalid native dtype errors")
        AhkTest.AssertTrue(
            Numpy.CanCast(Numpy.DT_INT8, Numpy.DT_INT16),
            "valid casting must not inherit a prior native error")

        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "casting facade must not retain native memory")
        Numpy.Cleanup()
    }

    static TestCopytoFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        destinationOwner := Numpy.Zeros([3, 2], Numpy.DT_INT64)
        destination := destinationOwner.Transpose()
        source := Numpy.IntArray([10, 20, 30])
        returned := Numpy.CopyTo(destination, source)
        AhkTest.AssertEqual(destination.Handle, returned.Handle,
            "CopyTo must return its destination")
        NumpyAssertArrayNear(
            [10, 20, 30, 10, 20, 30], destination.ToArray(), 0)

        exactSource := Numpy.IntArray(
            [9007199254740993, -9007199254740995])
        exactDestination := Numpy.Zeros([2], Numpy.DT_INT64)
        Numpy.CopyTo(exactDestination, exactSource)
        index := Buffer(8, 0)
        AhkTest.AssertEqual(9007199254740993,
            DllCall(Numpy.Proc("cnp_array_get_int"),
                "Ptr", exactDestination.Handle, "Ptr", index, "Int64"),
            "CopyTo must preserve int64 values above 2**53")
        NumPut("Int64", 1, index, 0)
        AhkTest.AssertEqual(-9007199254740995,
            DllCall(Numpy.Proc("cnp_array_get_int"),
                "Ptr", exactDestination.Handle, "Ptr", index, "Int64"),
            "CopyTo must preserve negative int64 values below -2**53")

        fractional := Numpy.Array([1.9, -2.1])
        integers := Numpy.Zeros([2], Numpy.DT_INT64)
        Numpy.CopyTo(integers, fractional, "unsafe")
        NumPut("Int64", 0, index, 0)
        AhkTest.AssertEqual(1,
            DllCall(Numpy.Proc("cnp_array_get_int"),
                "Ptr", integers.Handle, "Ptr", index, "Int64"))
        NumPut("Int64", 1, index, 0)
        AhkTest.AssertEqual(-2,
            DllCall(Numpy.Proc("cnp_array_get_int"),
                "Ptr", integers.Handle, "Ptr", index, "Int64"))

        protected := Numpy.Array([7, 8], , Numpy.DT_INT8)
        narrowingSource := Numpy.Array([9, 10], , Numpy.DT_INT16)
        rejectedSafeCast := false
        try Numpy.CopyTo(protected, narrowingSource, "safe")
        catch TypeError as err
            rejectedSafeCast := InStr(err.Message, "cannot cast") > 0
        AhkTest.AssertTrue(rejectedSafeCast,
            "CopyTo must expose rejected casts")
        NumpyAssertArrayNear([7, 8], protected.ToArray(), 0)

        rejectedCasting := false
        try Numpy.CopyTo(protected, narrowingSource, "invalid")
        catch ValueError as err
            rejectedCasting := InStr(err.Message, "valid casting") > 0
        AhkTest.AssertTrue(rejectedCasting,
            "CopyTo must reject unknown casting names explicitly")
        NumpyAssertArrayNear([7, 8], protected.ToArray(), 0)

        narrowingSource := 0
        protected := 0
        integers := 0
        fractional := 0
        exactDestination := 0
        exactSource := 0
        returned := 0
        source := 0
        destination := 0
        destinationOwner := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "CopyTo facade must not retain native memory")
        Numpy.Cleanup()
    }

    static TestWhereFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        condition := Numpy.Array([1, 0], [2, 1], Numpy.DT_INT8)
        left := Numpy.IntArray(
            [9007199254740993, 1152921504606846981,
             4611686018427387913], [1, 3])
        right := Numpy.IntArray([-1, -2], [2, 1])
        selected := Numpy.Where(condition, left, right)
        NumpyAssertArrayNear([2, 3], selected.Shape, 0)
        expected := [
            9007199254740993, 1152921504606846981,
            4611686018427387913, -2, -2, -2
        ]
        coordinates := Buffer(16, 0)
        loop 6 {
            NumPut("Int64", (A_Index - 1) // 3, coordinates, 0)
            NumPut("Int64", Mod(A_Index - 1, 3), coordinates, 8)
            AhkTest.AssertEqual(expected[A_Index],
                DllCall(Numpy.Proc("cnp_array_get_int"),
                    "Ptr", selected.Handle,
                    "Ptr", coordinates.Ptr, "Int64"),
                "Where must preserve selected int64 value " A_Index)
        }

        indexCondition := Numpy.Array(
            [0, 1, 2, 0, 3, 0], [2, 3], Numpy.DT_INT16)
        whereIndices := Numpy.Where(indexCondition)
        AhkTest.AssertEqual(2, whereIndices.Length)
        axis0 := whereIndices[1]
        axis1 := whereIndices[2]
        NumpyAssertArrayNear([0, 0, 1], axis0.ToArray(), 0)
        NumpyAssertArrayNear([1, 2, 1], axis1.ToArray(), 0)

        emptyCondition := Numpy.Zeros([2, 0, 3], Numpy.DT_BOOL)
        emptyIndices := Numpy.Where(emptyCondition)
        AhkTest.AssertEqual(3, emptyIndices.Length)
        for item in emptyIndices
            AhkTest.AssertEqual(0, item.Size)

        incompatible := Numpy.Zeros([4], Numpy.DT_INT64)
        rejectedShape := false
        try Numpy.Where(condition, left, incompatible)
        catch ValueError as err
            rejectedShape := InStr(err.Message, "broadcast") > 0
        AhkTest.AssertTrue(rejectedShape,
            "Where must expose incompatible broadcast shapes")

        rejectedPair := false
        try Numpy.Where(condition, left)
        catch ValueError as err
            rejectedPair := InStr(err.Message, "together") > 0
        AhkTest.AssertTrue(rejectedPair,
            "Where must require x and y together")

        incompatible := 0
        emptyIndices := 0
        emptyCondition := 0
        axis1 := 0
        axis0 := 0
        whereIndices := 0
        indexCondition := 0
        selected := 0
        right := 0
        left := 0
        condition := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "Where facade must not retain native memory")
        Numpy.Cleanup()
    }

    static TestMatrixPowerFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        large := Numpy.IntArray(
            [134217729, 0, 0, -134217731], [2, 2])
        squared := Numpy.Linalg.MatrixPower(large, 2)
        AhkTest.AssertEqual(Numpy.DT_LONGLONG, squared.Dtype)
        coordinates := Buffer(16, 0)
        expected := [
            18014398777917441, 0,
            0, 18014399314788361
        ]
        loop 4 {
            NumPut("Int64", (A_Index - 1) // 2, coordinates, 0)
            NumPut("Int64", Mod(A_Index - 1, 2), coordinates, 8)
            AhkTest.AssertEqual(expected[A_Index],
                DllCall(Numpy.Proc("cnp_array_get_int"),
                    "Ptr", squared.Handle,
                    "Ptr", coordinates.Ptr, "Int64"),
                "MatrixPower must preserve exact int64 result " A_Index)
        }

        identity := Numpy.Linalg.MatrixPower(large, 0)
        AhkTest.AssertEqual(Numpy.DT_LONGLONG, identity.Dtype)
        NumpyAssertArrayNear([1, 0, 0, 1], identity.ToArray(), 0)

        floating := Numpy.Array([1.0, 2.0, 3.0, 5.0], [2, 2])
        inverse := Numpy.Linalg.MatrixPower(floating, -1)
        NumpyAssertArrayNear([-5.0, 2.0, 3.0, -1.0],
            inverse.ToArray(), 1e-10)

        batched := Numpy.IntArray(
            [1, 2, 3, 5, 2, 1, 1, 1], [2, 2, 2])
        batchedIdentity := Numpy.Linalg.MatrixPower(batched, 0)
        NumpyAssertArrayNear([2, 2, 2], batchedIdentity.Shape, 0)
        NumpyAssertArrayNear(
            [1, 0, 0, 1, 1, 0, 0, 1],
            batchedIdentity.ToArray(), 0)

        nonsquare := Numpy.Zeros([2, 3], Numpy.DT_FLOAT64)
        rejectedShape := false
        try Numpy.Linalg.MatrixPower(nonsquare, 2)
        catch ValueError as err
            rejectedShape := InStr(err.Message, "square") > 0
        AhkTest.AssertTrue(rejectedShape,
            "MatrixPower must expose non-square inputs")

        nonsquare := 0
        batchedIdentity := 0
        batched := 0
        inverse := 0
        floating := 0
        identity := 0
        squared := 0
        large := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "MatrixPower facade must not retain native memory")
        Numpy.Cleanup()
    }

    static TestMatrixRankFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        fullRank := Numpy.Array([1.0, 0.0, 0.0, 1.0], [2, 2])
        rank := Numpy.Linalg.MatrixRank(fullRank)
        AhkTest.AssertEqual("Integer", Type(rank),
            "MatrixRank must unwrap scalar results")
        AhkTest.AssertEqual(2, rank)

        batched := Numpy.Array([
            1.0, 0.0, 0.0, 1.0,
            1.0, 2.0, 2.0, 4.0
        ], [2, 2, 2])
        ranks := Numpy.Linalg.MatrixRank(batched)
        AhkTest.AssertTrue(ranks is Numpy.NdArray,
            "MatrixRank must preserve batched array results")
        NumpyAssertArrayNear([2], ranks.Shape, 0)
        NumpyAssertArrayNear([2, 1], ranks.ToArray(), 0)

        toleranceRank := Numpy.Linalg.MatrixRank(fullRank, 1.5)
        AhkTest.AssertEqual(0, toleranceRank,
            "MatrixRank scalar tolerance must remain scalar")

        tolerances := Numpy.Array([0.0, 6.0])
        broadcastRanks := Numpy.Linalg.MatrixRank(batched, tolerances)
        NumpyAssertArrayNear([2, 0], broadcastRanks.ToArray(), 0)

        asymmetric := Numpy.Array([0.0, 0.0, 1.0, 0.0], [2, 2])
        hermitianRank := Numpy.Linalg.MatrixRank(asymmetric, 0.0, true)
        AhkTest.AssertEqual(2, hermitianRank,
            "MatrixRank must forward the hermitian option")

        rejectedToleranceType := false
        try Numpy.Linalg.MatrixRank(fullRank, "invalid")
        catch TypeError as err
            rejectedToleranceType := InStr(err.Message, "tolerance") > 0
        AhkTest.AssertTrue(rejectedToleranceType,
            "MatrixRank must reject nonnumeric tolerance values")

        rejectedHermitian := false
        try Numpy.Linalg.MatrixRank(fullRank, 0.0, 2)
        catch ValueError as err
            rejectedHermitian := InStr(err.Message, "hermitian") > 0
        AhkTest.AssertTrue(rejectedHermitian,
            "MatrixRank must reject non-Boolean hermitian integers")

        incompatibleTolerance := Numpy.Array([0.0, 1.0, 2.0])
        rejectedBroadcast := false
        try Numpy.Linalg.MatrixRank(batched, incompatibleTolerance)
        catch ValueError as err
            rejectedBroadcast := InStr(err.Message, "broadcast") > 0
        AhkTest.AssertTrue(rejectedBroadcast,
            "MatrixRank must expose native tolerance broadcast errors")

        incompatibleTolerance := 0
        asymmetric := 0
        broadcastRanks := 0
        tolerances := 0
        ranks := 0
        batched := 0
        fullRank := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "MatrixRank facade must not retain native memory")
        Numpy.Cleanup()
    }

    static TestSpacingFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        negativeZero := NumpyDoubleFromBits(0x80000000)
        values := Numpy.Array([-1.0, negativeZero, 0.0, 1.0])
        spaced := Numpy.Spacing(values)
        actual := spaced.ToArray()
        AhkTest.AssertEqual(NumpyDoubleFromBits(0xbcb00000), actual[1])
        AhkTest.AssertEqual(NumpyDoubleFromBits(0, 1), actual[2])
        AhkTest.AssertEqual(NumpyDoubleFromBits(0, 1), actual[3])
        AhkTest.AssertEqual(NumpyDoubleFromBits(0x3cb00000), actual[4])

        floatSource := Numpy.Array([1.0], , Numpy.DT_FLOAT32)
        floatSpacing := Numpy.Spacing(floatSource)
        AhkTest.AssertEqual(Numpy.DT_FLOAT32, floatSpacing.Dtype)
        AhkTest.AssertEqual(2.0 ** -23, floatSpacing.GetItem(0))

        complexSource := Numpy.Zeros([1], Numpy.DT_COMPLEX64)
        rejected := false
        try Numpy.Spacing(complexSource)
        catch TypeError as err
            rejected := InStr(err.Message, "real numeric dtype") > 0
        AhkTest.AssertTrue(rejected,
            "Spacing facade must expose unsupported dtype errors")

        values := 0
        spaced := 0
        floatSource := 0
        floatSpacing := 0
        complexSource := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "spacing facade must not retain native memory")
        Numpy.Cleanup()
    }

    static TestNextafterFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        left := Numpy.Array([-1.0, 1.0], [2, 1], Numpy.DT_FLOAT32)
        right := Numpy.Array([-2.0, 1.0, 2.0], [3], Numpy.DT_FLOAT32)
        result := Numpy.Nextafter(left, right)
        instanceResult := left.Nextafter(right)
        expected := [
            NumpyFloatFromBits(0xbf800001),
            NumpyFloatFromBits(0xbf7fffff),
            NumpyFloatFromBits(0xbf7fffff),
            NumpyFloatFromBits(0x3f7fffff),
            1.0,
            NumpyFloatFromBits(0x3f800001)
        ]
        NumpyAssertArrayNear([2, 3], result.Shape, 0)
        AhkTest.AssertEqual(Numpy.DT_FLOAT32, result.Dtype)
        NumpyAssertArrayNear(expected, result.ToArray(), 0)
        NumpyAssertArrayNear(expected, instanceResult.ToArray(), 0)

        complexSource := Numpy.Zeros([1], Numpy.DT_COMPLEX64)
        rejected := false
        try Numpy.Nextafter(complexSource, right)
        catch TypeError as err
            rejected := InStr(err.Message, "real numeric dtype") > 0
        AhkTest.AssertTrue(rejected,
            "Nextafter facade must expose unsupported dtype errors")

        left := 0
        right := 0
        result := 0
        instanceResult := 0
        complexSource := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "nextafter facade must not retain native memory")
        Numpy.Cleanup()
    }

    static TestCopysignFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        negativeZero := NumpyDoubleFromBits(0x80000000)
        left := Numpy.Array([-1.0, 1.0], [2, 1], Numpy.DT_FLOAT32)
        signs := Numpy.Array([-1.0, 0.0, negativeZero],
            [3], Numpy.DT_FLOAT32)
        result := Numpy.Copysign(left, signs)
        instanceResult := left.Copysign(signs)
        expected := [-1.0, 1.0, -1.0, -1.0, 1.0, -1.0]
        NumpyAssertArrayNear([2, 3], result.Shape, 0)
        AhkTest.AssertEqual(Numpy.DT_FLOAT32, result.Dtype)
        NumpyAssertArrayNear(expected, result.ToArray(), 0)
        NumpyAssertArrayNear(expected, instanceResult.ToArray(), 0)

        complexSource := Numpy.Zeros([1], Numpy.DT_COMPLEX64)
        rejected := false
        try Numpy.Copysign(complexSource, signs)
        catch TypeError as err
            rejected := InStr(err.Message, "real numeric dtype") > 0
        AhkTest.AssertTrue(rejected,
            "Copysign facade must expose unsupported dtype errors")

        left := 0
        signs := 0
        result := 0
        instanceResult := 0
        complexSource := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "copysign facade must not retain native memory")
        Numpy.Cleanup()
    }

    static TestFrexpFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        source := Numpy.Array([-8.0, -0.0, 0.0, 6.5],
            [2, 2], Numpy.DT_FLOAT32)
        outputs := Numpy.Frexp(source)
        instanceOutputs := source.Frexp()
        AhkTest.AssertEqual(2, outputs.Length)
        AhkTest.AssertEqual(Numpy.DT_FLOAT32, outputs[1].Dtype)
        AhkTest.AssertEqual(Numpy.DT_INT32, outputs[2].Dtype)
        NumpyAssertArrayNear([2, 2], outputs[1].Shape, 0)
        NumpyAssertArrayNear([-0.5, -0.0, 0.0, 0.8125],
            outputs[1].ToArray(), 0)
        NumpyAssertArrayNear([4, 0, 0, 3], outputs[2].ToArray(), 0)
        NumpyAssertArrayNear(outputs[1].ToArray(),
            instanceOutputs[1].ToArray(), 0)
        NumpyAssertArrayNear(outputs[2].ToArray(),
            instanceOutputs[2].ToArray(), 0)

        complexSource := Numpy.Zeros([1], Numpy.DT_COMPLEX64)
        rejected := false
        try Numpy.Frexp(complexSource)
        catch TypeError as err
            rejected := InStr(err.Message, "real numeric dtype") > 0
        AhkTest.AssertTrue(rejected,
            "Frexp facade must expose unsupported dtype errors")

        source := 0
        outputs := 0
        instanceOutputs := 0
        complexSource := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "frexp facade must not retain native memory")
        Numpy.Cleanup()
    }

    static TestModfFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        source := Numpy.Array([-1.5, -0.0, 0.0, 6.75],
            [2, 2], Numpy.DT_FLOAT32)
        outputs := Numpy.Modf(source)
        instanceOutputs := source.Modf()
        AhkTest.AssertEqual(2, outputs.Length)
        AhkTest.AssertEqual(Numpy.DT_FLOAT32, outputs[1].Dtype)
        AhkTest.AssertEqual(Numpy.DT_FLOAT32, outputs[2].Dtype)
        NumpyAssertArrayNear([2, 2], outputs[1].Shape, 0)
        NumpyAssertArrayNear([-0.5, -0.0, 0.0, 0.75],
            outputs[1].ToArray(), 0)
        NumpyAssertArrayNear([-1.0, -0.0, 0.0, 6.0],
            outputs[2].ToArray(), 0)
        NumpyAssertArrayNear(outputs[1].ToArray(),
            instanceOutputs[1].ToArray(), 0)
        NumpyAssertArrayNear(outputs[2].ToArray(),
            instanceOutputs[2].ToArray(), 0)

        complexSource := Numpy.Zeros([1], Numpy.DT_COMPLEX64)
        rejected := false
        try Numpy.Modf(complexSource)
        catch TypeError as err
            rejected := InStr(err.Message, "real numeric dtype") > 0
        AhkTest.AssertTrue(rejected,
            "Modf facade must expose unsupported dtype errors")

        source := 0
        outputs := 0
        instanceOutputs := 0
        complexSource := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "modf facade must not retain native memory")
        Numpy.Cleanup()
    }

    static TestLdexpFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        values := Numpy.Array([0.5, -1.0], [2, 1], Numpy.DT_FLOAT32)
        exponents := Numpy.Array([-1, 0, 1], [3], Numpy.DT_INT32)
        result := Numpy.Ldexp(values, exponents)
        instanceResult := values.Ldexp(exponents)
        expected := [0.25, 0.5, 1.0, -0.5, -1.0, -2.0]
        NumpyAssertArrayNear([2, 3], result.Shape, 0)
        AhkTest.AssertEqual(Numpy.DT_FLOAT32, result.Dtype)
        NumpyAssertArrayNear(expected, result.ToArray(), 0)
        NumpyAssertArrayNear(expected, instanceResult.ToArray(), 0)

        invalidExponents := Numpy.Array([1.0], , Numpy.DT_FLOAT32)
        rejected := false
        try Numpy.Ldexp(values, invalidExponents)
        catch TypeError as err
            rejected := InStr(err.Message, "integer exponent dtype") > 0
        AhkTest.AssertTrue(rejected,
            "Ldexp facade must expose exponent dtype errors")

        values := 0
        exponents := 0
        result := 0
        instanceResult := 0
        invalidExponents := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "ldexp facade must not retain native memory")
        Numpy.Cleanup()
    }

    static TestDigammaZetaFunctionFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        digammaInput := Numpy.Array([-0.5, 0.5, 1.0, 2.0, 10.0])
        floatInput := Numpy.Array([0.5, 1.0], , Numpy.DT_FLOAT32)
        zetaInput := Numpy.Array([-2.0, -1.0, 0.0, 2.0, 3.0])

        digammaValues := Numpy.Digamma(digammaInput)
        digammaFloatValues := Numpy.Digamma(floatInput)
        zetaValues := Numpy.Zeta(zetaInput)
        NumpyAssertArrayNear(
            [0.03648997397857652, -1.9635100260214235,
             -0.5772156649015329, 0.42278433509846713,
             2.251752589066721],
            digammaValues.ToArray(), 6e-14)
        AhkTest.AssertEqual(
            Numpy.DT_FLOAT32, digammaFloatValues.Dtype)
        NumpyAssertArrayNear(
            [0.0, -1.0 / 12.0, -0.5,
             1.6449340668482264, 1.2020569031595942],
            zetaValues.ToArray(), 6e-14)

        zetaValues := 0
        digammaFloatValues := 0
        digammaValues := 0
        zetaInput := 0
        floatInput := 0
        digammaInput := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "digamma/zeta facade must release every native result")
        Numpy.Cleanup()
    }

    static TestAllcloseCopy() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")

        shapeBuf := Buffer(8, 0)
        NumPut("Int64", 3, shapeBuf, 0)
        dataBuf := Buffer(24, 0)
        NumPut("Double", 1.0, dataBuf, 0)
        NumPut("Double", 2.0, dataBuf, 8)
        NumPut("Double", 3.0, dataBuf, 16)
        ha := DllCall(dllPath "\cnp_ahk_from_doubles", "Int", 1, "Ptr", shapeBuf, "Ptr", dataBuf, "Int64", 3, "Ptr")

        ; Copy
        hc := DllCall(dllPath "\cnp_ahk_copy", "Ptr", ha, "Ptr")
        AhkTest.AssertTrue(hc != 0)

        ; allclose(a, copy) should be true
        result := DllCall(dllPath "\cnp_ahk_allclose", "Ptr", ha, "Ptr", hc, "Double", 1e-5, "Double", 1e-8, "Int")
        AhkTest.AssertEqual(1, result)

        ; Modify copy, allclose should fail
        DllCall(dllPath "\cnp_ahk_set_item", "Ptr", hc, "Int64", 0, "Double", 99.0, "Int")
        result := DllCall(dllPath "\cnp_ahk_allclose", "Ptr", ha, "Ptr", hc, "Double", 1e-5, "Double", 1e-8, "Int")
        AhkTest.AssertEqual(0, result)

        DllCall(dllPath "\cnp_ahk_free", "Ptr", ha, "Int")
        DllCall(dllPath "\cnp_ahk_free", "Ptr", hc, "Int")
        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestAllcloseFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        quietNan := NumpyDoubleFromBits(0x7ff80000)
        left := Numpy.Array([1.0, 1.0], [2, 1])
        right := Numpy.Array([1.0, 1.0, 1.0], [1, 3])
        AhkTest.AssertTrue(Numpy.Allclose(left, right),
            "Numpy.Allclose must apply NumPy broadcasting")

        nanLeft := Numpy.Array([quietNan])
        nanRight := Numpy.Array([quietNan])
        AhkTest.AssertTrue(!Numpy.Allclose(nanLeft, nanRight),
            "Numpy.Allclose must default equal_nan to false")
        AhkTest.AssertTrue(Numpy.Allclose(nanLeft, nanRight, 1e-5, 1e-8, true),
            "Numpy.Allclose must expose equal_nan")

        rejected := false
        incompatible := Numpy.Array([1.0, 2.0])
        try Numpy.Allclose(incompatible, right)
        catch ValueError as err {
            rejected := InStr(err.Message, "broadcast") > 0
        }
        AhkTest.AssertTrue(rejected,
            "Numpy.Allclose must surface native broadcast errors")

        loop 64
            AhkTest.AssertTrue(Numpy.Allclose(left, right))
        incompatible := 0
        nanRight := 0
        nanLeft := 0
        right := 0
        left := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "Numpy.Allclose must not retain native memory")
        Numpy.Cleanup()
    }

    static TestAssertionSurfaceFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        quietNan := NumpyDoubleFromBits(0x7ff80000)
        exactLeft := Numpy.Array([quietNan, 7.0])
        exactRight := Numpy.Array([quietNan, 7.0])
        exactMismatch := Numpy.Array([quietNan, 8.0])
        AhkTest.AssertTrue(Numpy.AssertArrayEqual(exactLeft, exactRight),
            "AssertArrayEqual must follow NumPy NaN equality")
        AhkTest.AssertTrue(!Numpy.AssertArrayEqual(exactLeft, exactMismatch),
            "AssertArrayEqual must report a value mismatch")

        almostLeft := Numpy.Array([0.0, quietNan])
        almostPass := Numpy.Array([0.149, quietNan])
        almostFail := Numpy.Array([0.151, quietNan])
        AhkTest.AssertTrue(
            Numpy.AssertArrayAlmostEqual(almostLeft, almostPass, 1),
            "AssertArrayAlmostEqual must use NumPy's 1.5 ULP decimal bound")
        AhkTest.AssertTrue(
            !Numpy.AssertArrayAlmostEqual(almostLeft, almostFail, 1),
            "AssertArrayAlmostEqual must reject values beyond the bound")

        closeRight := Numpy.Array([1.0000001, quietNan])
        closeFail := Numpy.Array([1.01, quietNan])
        closeLeft := Numpy.Array([1.0, quietNan])
        AhkTest.AssertTrue(Numpy.AssertAllclose(closeLeft, closeRight,
            1e-5, 1e-8), "AssertAllclose must default equal_nan to true")
        AhkTest.AssertTrue(!Numpy.AssertAllclose(closeLeft, closeFail,
            1e-5, 1e-8), "AssertAllclose must report tolerance mismatches")

        rejected := false
        try Numpy.AssertArrayEqual(exactLeft, 0)
        catch TypeError as err {
            rejected := InStr(err.Message, "NdArray") > 0
        }
        AhkTest.AssertTrue(rejected,
            "assertion facades must reject non-array inputs explicitly")

        loop 64 {
            AhkTest.AssertTrue(Numpy.AssertArrayEqual(exactLeft, exactRight))
            AhkTest.AssertTrue(
                Numpy.AssertArrayAlmostEqual(almostLeft, almostPass, 1))
            AhkTest.AssertTrue(Numpy.AssertAllclose(closeLeft, closeRight,
                1e-5, 1e-8))
        }
        closeLeft := 0
        closeFail := 0
        closeRight := 0
        almostFail := 0
        almostPass := 0
        almostLeft := 0
        exactMismatch := 0
        exactRight := 0
        exactLeft := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "assertion facades must not retain native memory")
        Numpy.Cleanup()
    }

    static TestRuntimeUtilitySurfaceFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        AhkTest.AssertEqual(24, Numpy.MinScalarType(1.0))
        AhkTest.AssertEqual(12, Numpy.MinScalarType(65000.0))
        integer := Numpy.Array([1], [1], 2)
        float32 := Numpy.Array([1], [1], 12)
        AhkTest.AssertEqual(13, Numpy.CommonType(integer, float32),
            "CommonType must apply NumPy's integer-to-double rule")

        originalErrors := Numpy.GetErr()
        originalPrint := Numpy.GetPrintOptions()
        originalBuffer := Numpy.GetBufsize()
        try {
            previousErrors := Numpy.SetErr(
                "raise", "ignore", "warn", "call")
            AhkTest.AssertEqual(originalErrors["divide"],
                previousErrors["divide"])
            currentErrors := Numpy.GetErr()
            AhkTest.AssertEqual("raise", currentErrors["divide"])
            AhkTest.AssertEqual("ignore", currentErrors["over"])
            AhkTest.AssertEqual("warn", currentErrors["under"])
            AhkTest.AssertEqual("call", currentErrors["invalid"])

            rejectedMode := false
            try Numpy.SetErr("explode")
            catch ValueError as err
                rejectedMode := InStr(err.Message, "mode") > 0
            AhkTest.AssertTrue(rejectedMode,
                "SetErr must reject unsupported modes explicitly")

            previousPrint := Numpy.SetPrintOptions(5, 12, 2, 91, true)
            AhkTest.AssertEqual(originalPrint["precision"],
                previousPrint["precision"])
            currentPrint := Numpy.GetPrintOptions()
            AhkTest.AssertEqual(5, currentPrint["precision"])
            AhkTest.AssertEqual(12, currentPrint["threshold"])
            AhkTest.AssertEqual(2, currentPrint["edgeitems"])
            AhkTest.AssertEqual(91, currentPrint["linewidth"])
            AhkTest.AssertTrue(currentPrint["suppress"])

            rejectedPrint := false
            try Numpy.SetPrintOptions(-2)
            catch ValueError as err
                rejectedPrint := InStr(err.Message, "print options") > 0
            AhkTest.AssertTrue(rejectedPrint,
                "SetPrintOptions must preserve native validation")

            AhkTest.AssertEqual(originalBuffer, Numpy.SetBufsize(16))
            AhkTest.AssertEqual(16, Numpy.GetBufsize())
            rejectedBuffer := false
            try Numpy.SetBufsize(17)
            catch ValueError as err
                rejectedBuffer := InStr(err.Message, "multiple of 16") > 0
            AhkTest.AssertTrue(rejectedBuffer,
                "SetBufsize must preserve NumPy's alignment error")

            AhkTest.AssertEqual("1.25", Numpy.FormatFloat(1.25, 2, false))
            AhkTest.AssertEqual("1.25e+00",
                Numpy.FormatFloat(1.25, 2, true))

            loop 64 {
                ignored := Numpy.GetErr()
                ignored := Numpy.GetPrintOptions()
                ignored := Numpy.FormatFloat(-0.0, 2, false)
            }
        } finally {
            Numpy.SetErr(
                originalErrors["divide"], originalErrors["over"],
                originalErrors["under"], originalErrors["invalid"])
            Numpy.SetPrintOptions(
                originalPrint["precision"], originalPrint["threshold"],
                originalPrint["edgeitems"], originalPrint["linewidth"],
                originalPrint["suppress"])
            Numpy.SetBufsize(originalBuffer)
        }
        float32 := 0
        integer := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "runtime utility facades must not retain native memory")
        Numpy.Cleanup()
    }

    static TestEmathSurfaceFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        negative := Numpy.Array([-4.0, 0.0, 9.0], [3], Numpy.DT_FLOAT32)
        sqrtResult := Numpy.Emath.Sqrt(negative)
        AhkTest.AssertEqual(Numpy.DT_COMPLEX64, sqrtResult.Dtype)
        sqrtReal := Numpy.Real(sqrtResult)
        sqrtImag := Numpy.Imag(sqrtResult)
        NumpyAssertArrayNear([0.0, 0.0, 3.0], sqrtReal.ToArray(), 1e-6)
        NumpyAssertArrayNear([2.0, 0.0, 0.0], sqrtImag.ToArray(), 1e-6)

        positive := Numpy.Array([1.0, 4.0, 9.0])
        positiveSqrt := Numpy.Emath.Sqrt(positive)
        AhkTest.AssertEqual(Numpy.DT_FLOAT64, positiveSqrt.Dtype)
        NumpyAssertArrayNear([1.0, 2.0, 3.0],
            positiveSqrt.ToArray(), 1e-12)

        logInput := Numpy.Array([-1.0, 8.0], [2], Numpy.DT_FLOAT32)
        logResult := Numpy.Emath.Log(logInput)
        log10Result := Numpy.Emath.Log10(logInput)
        log2Result := Numpy.Emath.Log2(logInput)
        arcsinResult := Numpy.Emath.Arcsin(logInput)
        arccosResult := Numpy.Emath.Arccos(logInput)
        arctanhResult := Numpy.Emath.Arctanh(logInput)
        for result in [logResult, log10Result, log2Result,
                arcsinResult, arccosResult, arctanhResult]
            AhkTest.AssertEqual(Numpy.DT_COMPLEX64, result.Dtype)

        base := Numpy.Array([-1.0, 4.0], [2, 1], Numpy.DT_FLOAT32)
        exponent := Numpy.Array([0.5, 2.0, -1.0], [1, 3], Numpy.DT_FLOAT32)
        powered := Numpy.Emath.Power(base, exponent)
        AhkTest.AssertEqual(Numpy.DT_COMPLEX64, powered.Dtype)
        NumpyAssertArrayNear([2, 3], powered.Shape, 0)
        poweredReal := Numpy.Real(powered)
        poweredImag := Numpy.Imag(powered)
        NumpyAssertArrayNear([0.0, 1.0, -1.0, 2.0, 16.0, 0.25],
            poweredReal.ToArray(), 2e-6)
        NumpyAssertArrayNear([1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            poweredImag.ToArray(), 2e-6)

        rejected := false
        try Numpy.Emath.Sqrt(0)
        catch TypeError as err
            rejected := InStr(err.Message, "NdArray") > 0
        AhkTest.AssertTrue(rejected,
            "Emath facade must reject non-array inputs explicitly")

        loop 64 {
            repeated := Numpy.Emath.Sqrt(negative)
            repeated := 0
        }
        poweredImag := 0
        poweredReal := 0
        powered := 0
        exponent := 0
        base := 0
        arctanhResult := 0
        arccosResult := 0
        arcsinResult := 0
        log2Result := 0
        log10Result := 0
        logResult := 0
        logInput := 0
        positiveSqrt := 0
        positive := 0
        sqrtImag := 0
        sqrtReal := 0
        sqrtResult := 0
        negative := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "Emath facade must release every native result")
        Numpy.Cleanup()
    }

    static TestEinsumAliasSurfaceFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        left := Numpy.Array([1, 2, 3, 4, 5, 6], [2, 3])
        square := Numpy.Array([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3])
        vector := Numpy.Array([1, 2, 3])
        otherVector := Numpy.Array([4, 5, 6])
        right := Numpy.Array([1, 2, 3, 4, 5, 6], [3, 2])

        diagonal := Numpy.EinsumDiag(square)
        transposed := Numpy.EinsumTranspose(left)
        outer := Numpy.EinsumOuter(vector, Numpy.Array([4, 5]))
        product := Numpy.EinsumMatmul(left, right)
        matrixVector := Numpy.EinsumMatvec(left, vector)
        NumpyAssertArrayNear([1, 5, 9], diagonal.ToArray(), 0)
        NumpyAssertArrayNear(
            [1, 4, 2, 5, 3, 6], transposed.ToArray(), 0)
        NumpyAssertArrayNear(
            [4, 5, 8, 10, 12, 15], outer.ToArray(), 0)
        NumpyAssertArrayNear([22, 28, 49, 64], product.ToArray(), 0)
        NumpyAssertArrayNear([14, 32], matrixVector.ToArray(), 0)
        AhkTest.AssertEqual(32.0, Numpy.EinsumDot(vector, otherVector))
        AhkTest.AssertEqual(15.0, Numpy.EinsumTrace(square))
        AhkTest.AssertEqual(21.0, Numpy.EinsumSum(left))

        rejectedShape := false
        incompatible := Numpy.Ones([2, 4])
        try Numpy.EinsumMatmul(left, incompatible)
        catch ValueError as err {
            rejectedShape := InStr(err.Message, "Numpy.EinsumMatmul") > 0
                && InStr(err.Message, "broadcast") > 0
        }
        AhkTest.AssertTrue(rejectedShape,
            "EinsumMatmul must preserve its public label and native shape error")

        rejectedComplexScalar := false
        complexValue := Numpy.Zeros([1], Numpy.DT_COMPLEX128)
        try Numpy.EinsumDot(complexValue, complexValue)
        catch TypeError as err {
            rejectedComplexScalar := InStr(err.Message, "Numpy.EinsumDot") > 0
                && InStr(err.Message, "legacy ABI") > 0
        }
        AhkTest.AssertTrue(rejectedComplexScalar,
            "EinsumDot must expose its complex scalar ABI limitation")

        rejectedType := false
        try Numpy.EinsumDiag(0)
        catch TypeError as err
            rejectedType := InStr(err.Message, "NdArray") > 0
        AhkTest.AssertTrue(rejectedType,
            "einsum alias facades must reject non-array inputs explicitly")

        loop 64 {
            temporary := Numpy.EinsumMatvec(left, vector)
            temporary := 0
            AhkTest.AssertEqual(32.0,
                Numpy.EinsumDot(vector, otherVector))
        }
        complexValue := 0
        incompatible := 0
        matrixVector := 0
        product := 0
        outer := 0
        transposed := 0
        diagonal := 0
        right := 0
        otherVector := 0
        vector := 0
        square := 0
        left := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "einsum alias facades must release every native result")
        Numpy.Cleanup()
    }

    static TestProductLinalgSurfaceFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        left := Numpy.Array([1, 2, 3, 4, 5, 6], [2, 3])
        right := Numpy.Array([1, 2, 3, 4, 5, 6], [3, 2])
        vector := Numpy.Array([1, 2, 3])
        otherVector := Numpy.Array([4, 5, 6])

        dot := Numpy.Dot(left, right)
        dotGeneral := Numpy.DotGeneral(left, right)
        matmul := Numpy.Matmul(left, right)
        inner := Numpy.Inner(left, vector)
        outer := Numpy.Outer(vector, Numpy.Array([4, 5]))
        cross := Numpy.Cross(
            Numpy.Array([1, 0, 0]), Numpy.Array([0, 1, 0]))
        kron := Numpy.Kron(
            Numpy.Array([1, 2, 3, 4], [2, 2]),
            Numpy.Array([0, 5, 6, 7], [2, 2]))
        tensordot := Numpy.Tensordot(left, right, 1, 1)
        tensordotDefault := Numpy.TensordotDefault(left, right, 1)
        dot1d := Numpy.Dot1D(vector, otherVector)
        square := Numpy.Array([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3])
        trace := Numpy.Trace(square)
        multiDot := Numpy.MultiDot([
            left,
            right,
            Numpy.Array([1, 2], [2, 1]),
        ])

        for product in [dot, dotGeneral, matmul, tensordot, tensordotDefault]
            NumpyAssertArrayNear([22, 28, 49, 64], product.ToArray(), 0)
        NumpyAssertArrayNear([14, 32], inner.ToArray(), 0)
        NumpyAssertArrayNear([4, 5, 8, 10, 12, 15], outer.ToArray(), 0)
        NumpyAssertArrayNear([0, 0, 1], cross.ToArray(), 0)
        NumpyAssertArrayNear([
            0, 5, 0, 10,
            6, 7, 12, 14,
            0, 15, 0, 20,
            18, 21, 24, 28,
        ], kron.ToArray(), 0)
        AhkTest.AssertEqual(0, dot1d.Ndim)
        NumpyAssertArrayNear([32], dot1d.ToArray(), 0)
        AhkTest.AssertEqual(0, trace.Ndim)
        NumpyAssertArrayNear([15], trace.ToArray(), 0)
        AhkTest.AssertEqual(15.0, Numpy.TraceExt(square))
        AhkTest.AssertEqual(32.0, Numpy.Vdot(vector, otherVector))
        NumpyAssertArrayNear([2, 1], multiDot.Shape, 0)
        NumpyAssertArrayNear([78, 177], multiDot.ToArray(), 0)

        rejectedAxes := false
        try Numpy.Tensordot(left, right, 1, 2)
        catch ValueError as err
            rejectedAxes := InStr(err.Message, "Numpy.Tensordot") > 0
                && InStr(err.Message, "must be equal") > 0
        AhkTest.AssertTrue(rejectedAxes,
            "Tensordot must expose its represented-axis contract")

        rejectedDot1d := false
        try Numpy.Dot1D(left, right)
        catch ValueError as err
            rejectedDot1d := InStr(err.Message, "Numpy.Dot1D") > 0
        AhkTest.AssertTrue(rejectedDot1d,
            "Dot1D must expose labeled rank errors")

        rejectedVdotComplex := false
        complexValue := Numpy.Zeros([2, 2], Numpy.DT_COMPLEX128)
        try Numpy.Vdot(complexValue, complexValue)
        catch TypeError as err
            rejectedVdotComplex := InStr(err.Message, "Numpy.Vdot") > 0
                && InStr(err.Message, "complex") > 0
        AhkTest.AssertTrue(rejectedVdotComplex,
            "Vdot must expose its complex scalar ABI limitation")

        rejectedTraceComplex := false
        try Numpy.TraceExt(complexValue)
        catch TypeError as err
            rejectedTraceComplex := InStr(err.Message, "Numpy.TraceExt") > 0
                && InStr(err.Message, "complex") > 0
        AhkTest.AssertTrue(rejectedTraceComplex,
            "TraceExt must expose its complex scalar ABI limitation")

        rejectedMultiDot := false
        try Numpy.MultiDot([left])
        catch ValueError as err
            rejectedMultiDot := InStr(err.Message, "Numpy.MultiDot") > 0
        AhkTest.AssertTrue(rejectedMultiDot,
            "MultiDot must expose its minimum-input error")

        activeMemory := Numpy.AllocatedMemory()
        loop 32 {
            temporary := Numpy.Dot(left, right)
            temporary := Numpy.DotGeneral(left, right)
            temporary := Numpy.Matmul(left, right)
            temporary := Numpy.Inner(left, vector)
            temporary := Numpy.Outer(vector, otherVector)
            temporary := Numpy.Cross(
                Numpy.Array([1, 0, 0]), Numpy.Array([0, 1, 0]))
            temporary := Numpy.Kron(left, right)
            temporary := Numpy.Tensordot(left, right, 1, 1)
            temporary := Numpy.TensordotDefault(left, right, 1)
            temporary := Numpy.Dot1D(vector, otherVector)
            temporary := Numpy.Trace(square)
            AhkTest.AssertEqual(15.0, Numpy.TraceExt(square))
            AhkTest.AssertEqual(32.0, Numpy.Vdot(vector, otherVector))
            temporary := Numpy.MultiDot([left, right])
            temporary := 0
        }
        AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
            "product facade repeated calls must release native results")

        complexValue := 0
        multiDot := 0
        trace := 0
        square := 0
        dot1d := 0
        tensordotDefault := 0
        tensordot := 0
        kron := 0
        cross := 0
        outer := 0
        inner := 0
        matmul := 0
        dotGeneral := 0
        dot := 0
        otherVector := 0
        vector := 0
        right := 0
        left := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "product facade must restore retained native bytes")
        Numpy.Cleanup()
    }

    static TestLinalgStatisticsSurfaceFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        matrix := Numpy.Array([4, 7, 2, 6], [2, 2])
        inverse := Numpy.Linalg.Inv(matrix)
        pseudoinverse := Numpy.Linalg.Pinv(matrix, 1e-12)
        pseudoinverseAlias := Numpy.Pinv(matrix, 1e-12)
        defaultNorm := Numpy.Linalg.Norm(matrix)
        rowNorms := Numpy.Linalg.Norm(matrix, "1", 1)
        qr := Numpy.Linalg.Qr(matrix)
        reconstructed := Numpy.Matmul(qr[1], qr[2])

        NumpyAssertArrayNear([0.6, -0.7, -0.2, 0.4],
            inverse.ToArray(), 1e-12)
        NumpyAssertArrayNear(inverse.ToArray(),
            pseudoinverse.ToArray(), 1e-12)
        NumpyAssertArrayNear(inverse.ToArray(),
            pseudoinverseAlias.ToArray(), 1e-12)
        AhkTest.AssertEqual(0, defaultNorm.Ndim)
        NumpyAssertArrayNear([Sqrt(105.0)], defaultNorm.ToArray(), 1e-12)
        NumpyAssertArrayNear([11.0, 8.0], rowNorms.ToArray(), 1e-12)
        AhkTest.AssertEqual(13.0, Numpy.Linalg.NormExt(matrix, 1.0, -1))
        NumpyAssertArrayNear(matrix.ToArray(), reconstructed.ToArray(), 1e-12)

        tensor := Numpy.Reshape(Numpy.Identity(4), [2, 2, 2, 2])
        tensorInverse := Numpy.Linalg.Tensorinv(tensor, 2)
        rhs := Numpy.Array([1, 2, 3, 4], [2, 2])
        tensorSolution := Numpy.Linalg.Tensorsolve(tensor, rhs)
        axesSolution := Numpy.Linalg.Tensorsolve(tensor, rhs, [0])
        NumpyAssertArrayNear(tensor.ToArray(),
            tensorInverse.ToArray(), 1e-12)
        NumpyAssertArrayNear(rhs.ToArray(), tensorSolution.ToArray(), 1e-12)
        NumpyAssertArrayNear(rhs.ToArray(), axesSolution.ToArray(), 1e-12)

        observations := Numpy.Array([1, 2, 3, 2, 4, 6], [2, 3])
        other := Numpy.Array([3, 6, 9])
        correlation := Numpy.Corrcoef(observations, other)
        covariance := Numpy.Cov(observations)
        NumpyAssertArrayNear([
            1, 1, 1,
            1, 1, 1,
            1, 1, 1,
        ], correlation.ToArray(), 1e-12)
        NumpyAssertArrayNear([1, 2, 2, 4], covariance.ToArray(), 1e-12)

        rejectedNorm := false
        try Numpy.Linalg.Norm(matrix, "invalid", 0)
        catch ValueError as err
            rejectedNorm := InStr(err.Message, "Numpy.Linalg.Norm") > 0
        AhkTest.AssertTrue(rejectedNorm,
            "Norm must expose labeled order errors")

        rejectedQr := false
        try Numpy.Linalg.Qr(other)
        catch ValueError as err
            rejectedQr := InStr(err.Message, "Numpy.Linalg.Qr") > 0
        AhkTest.AssertTrue(rejectedQr,
            "Qr must expose labeled rank errors")

        rejectedCorrelation := false
        short := Numpy.Array([1, 2])
        try Numpy.Corrcoef(observations, short)
        catch ValueError as err
            rejectedCorrelation := InStr(err.Message, "Numpy.Corrcoef") > 0
        AhkTest.AssertTrue(rejectedCorrelation,
            "Corrcoef must reject mismatched observation counts")

        rejectedRowvar := false
        try Numpy.Cov(observations, observations, 2, 1)
        catch ValueError as err
            rejectedRowvar := InStr(err.Message, "Numpy.Cov") > 0
        AhkTest.AssertTrue(rejectedRowvar,
            "Cov must reject non-Boolean rowvar values")

        activeMemory := Numpy.AllocatedMemory()
        loop 24 {
            temporary := Numpy.Linalg.Inv(matrix)
            temporary := Numpy.Linalg.Pinv(matrix, 1e-12)
            temporary := Numpy.Pinv(matrix, 1e-12)
            temporary := Numpy.Linalg.Norm(matrix)
            temporary := Numpy.Linalg.Norm(matrix, "1", 1)
            pair := Numpy.Linalg.Qr(matrix)
            pair := 0
            temporary := Numpy.Linalg.Tensorinv(tensor, 2)
            temporary := Numpy.Linalg.Tensorsolve(tensor, rhs)
            temporary := Numpy.Linalg.Tensorsolve(tensor, rhs, [0])
            temporary := Numpy.Corrcoef(observations, other)
            temporary := Numpy.Cov(observations)
            temporary := 0
            AhkTest.AssertEqual(13.0,
                Numpy.Linalg.NormExt(matrix, 1.0, -1))
        }
        AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
            "linalg/statistics repeated calls must release native results")

        short := 0
        covariance := 0
        correlation := 0
        other := 0
        observations := 0
        axesSolution := 0
        tensorSolution := 0
        rhs := 0
        tensorInverse := 0
        tensor := 0
        reconstructed := 0
        qr := 0
        rowNorms := 0
        defaultNorm := 0
        pseudoinverseAlias := 0
        pseudoinverse := 0
        inverse := 0
        matrix := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "linalg/statistics facade must restore retained native bytes")
        Numpy.Cleanup()
    }

    static TestRandomCoreSurfaceFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        Numpy.Random.Seed(917)
        random := Numpy.Random.Random([2, 3])
        uniform := Numpy.Random.Uniform([2, 3], -2.0, 3.0)
        normal := Numpy.Random.Normal([2, 3], 1.0, 2.0)
        standard := Numpy.Random.StandardNormal([2, 3])
        integers := Numpy.Random.Integers([2, 3], 2, 5)
        randint := Numpy.Random.Randint([2, 3], 2, 5)
        binomial := Numpy.Random.Binomial([2, 3], 7, 0.3)
        poisson := Numpy.Random.Poisson([2, 3], 3.0)
        exponential := Numpy.Random.Exponential([2, 3], 2.0)
        gamma := Numpy.Random.Gamma([2, 3], 2.5, 1.5)
        beta := Numpy.Random.Beta([2, 3], 2.0, 5.0)
        for result in [random, uniform, normal, standard, integers,
                randint, binomial, poisson, exponential, gamma, beta]
            NumpyAssertArrayNear([2, 3], result.Shape, 0)
        for result in [random, uniform, normal, standard,
                exponential, gamma, beta]
            AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
        for result in [integers, randint, binomial, poisson]
            AhkTest.AssertEqual(Numpy.DT_INT32, result.Dtype)

        for value in random.ToArray()
            AhkTest.AssertTrue(value >= 0.0 && value < 1.0)
        for value in integers.ToArray()
            AhkTest.AssertTrue(value >= 2 && value <= 5)
        for value in randint.ToArray()
            AhkTest.AssertTrue(value >= 2 && value < 5)

        Numpy.Random.Seed(1234)
        replayFirst := Numpy.Random.Random([8])
        Numpy.Random.Seed(1234)
        replaySecond := Numpy.Random.Random([8])
        NumpyAssertArrayNear(
            replayFirst.ToArray(), replaySecond.ToArray(), 0)

        rejectedParameter := false
        try Numpy.Random.Normal([2], 0.0, -1.0)
        catch ValueError as err {
            rejectedParameter := InStr(err.Message, "Numpy.Random.Normal") > 0
                && InStr(err.Message, "scale") > 0
        }
        AhkTest.AssertTrue(rejectedParameter,
            "random facades must preserve native parameter failures")

        rejectedShape := false
        try Numpy.Random.Random(2)
        catch TypeError as err
            rejectedShape := InStr(err.Message, "shape") > 0
        AhkTest.AssertTrue(rejectedShape,
            "random facades must reject non-array shapes explicitly")

        activeMemory := Numpy.AllocatedMemory()
        loop 64 {
            temporary := Numpy.Random.Gamma([4], 0.5, 2.0)
            temporary := 0
        }
        AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
            "random facade results must be independently released")

        replaySecond := 0
        replayFirst := 0
        beta := 0
        gamma := 0
        exponential := 0
        poisson := 0
        binomial := 0
        randint := 0
        integers := 0
        standard := 0
        normal := 0
        uniform := 0
        random := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "random core facades must release every native result")
        Numpy.Cleanup()
    }

    static TestRandomExtendedSurfaceFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        Numpy.Random.Seed(2026)
        results := [
            Numpy.Random.Logseries([2, 3], 0.4),
            Numpy.Random.NegativeBinomial([2, 3], 2.5, 0.4),
            Numpy.Random.Pareto([2, 3], 3.0),
            Numpy.Random.Power([2, 3], 2.0),
            Numpy.Random.Rayleigh([2, 3], 1.5),
            Numpy.Random.StandardCauchy([2, 3]),
            Numpy.Random.StandardT([2, 3], 5.0),
            Numpy.Random.Triangular([2, 3], -2.0, 0.5, 4.0),
            Numpy.Random.Vonmises([2, 3], 0.5, 2.0),
            Numpy.Random.NoncentralChisquare([2, 3], 3.0, 1.5),
            Numpy.Random.NoncentralF([2, 3], 4.0, 6.0, 1.5),
            Numpy.Random.F([2, 3], 4.0, 6.0),
            Numpy.Random.Laplace([2, 3], 0.5, 1.5),
            Numpy.Random.Logistic([2, 3], 0.5, 1.5),
            Numpy.Random.Gumbel([2, 3], 0.5, 1.5),
            Numpy.Random.Weibull([2, 3], 2.0)
        ]
        for result in results
            NumpyAssertArrayNear([2, 3], result.Shape, 0)
        AhkTest.AssertEqual(Numpy.DT_INT32, results[1].Dtype)
        AhkTest.AssertEqual(Numpy.DT_INT32, results[2].Dtype)
        loop results.Length - 2
            AhkTest.AssertEqual(Numpy.DT_FLOAT64, results[A_Index + 2].Dtype)

        dirichlet := Numpy.Random.Dirichlet([1.0, 2.0, 3.0], 4)
        multinomial := Numpy.Random.Multinomial(7, [0.2, 0.3, 0.5], 4)
        NumpyAssertArrayNear([4, 3], dirichlet.Shape, 0)
        NumpyAssertArrayNear([4, 3], multinomial.Shape, 0)
        AhkTest.AssertEqual(Numpy.DT_FLOAT64, dirichlet.Dtype)
        AhkTest.AssertEqual(Numpy.DT_INT32, multinomial.Dtype)
        dirichletValues := dirichlet.ToArray()
        multinomialValues := multinomial.ToArray()
        loop 4 {
            offset := (A_Index - 1) * 3
            AhkTest.AssertTrue(Abs(
                dirichletValues[offset + 1]
                + dirichletValues[offset + 2]
                + dirichletValues[offset + 3] - 1.0) <= 1e-12)
            AhkTest.AssertEqual(7,
                multinomialValues[offset + 1]
                + multinomialValues[offset + 2]
                + multinomialValues[offset + 3])
        }

        Numpy.Random.Seed(88)
        replayFirst := Numpy.Random.StandardT([8], 5.0)
        Numpy.Random.Seed(88)
        replaySecond := Numpy.Random.StandardT([8], 5.0)
        NumpyAssertArrayNear(
            replayFirst.ToArray(), replaySecond.ToArray(), 0)

        rejected := false
        try Numpy.Random.Triangular([2], 1.0, 0.5, 0.0)
        catch ValueError as err {
            rejected := InStr(err.Message, "Numpy.Random.Triangular") > 0
                && InStr(err.Message, "left") > 0
        }
        AhkTest.AssertTrue(rejected,
            "extended random facades must preserve native parameter errors")

        rejectedValues := false
        try Numpy.Random.Dirichlet("bad", 2)
        catch TypeError as err
            rejectedValues := InStr(err.Message, "non-empty Array") > 0
        AhkTest.AssertTrue(rejectedValues,
            "Dirichlet must reject invalid alpha containers explicitly")

        activeMemory := Numpy.AllocatedMemory()
        loop 64 {
            temporary := Numpy.Random.Dirichlet([1.0, 2.0], 3)
            temporary := 0
        }
        AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
            "extended random results must be independently released")

        replaySecond := 0
        replayFirst := 0
        multinomial := 0
        dirichlet := 0
        results := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "extended random facades must release every native result")
        Numpy.Cleanup()
    }

    static TestRandomSpecialSurfaceFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        Numpy.Random.Seed(404)
        chisquare := Numpy.Random.Chisquare([2, 3], 3.0)
        geometric := Numpy.Random.Geometric([2, 3], 0.4)
        zipf := Numpy.Random.Zipf([2, 3], 2.5)
        wald := Numpy.Random.Wald([2, 3], 2.0, 3.0)
        hypergeometric := Numpy.Random.Hypergeometric([2, 3], 7, 5, 4)
        for result in [chisquare, geometric, zipf, wald, hypergeometric]
            NumpyAssertArrayNear([2, 3], result.Shape, 0)
        AhkTest.AssertEqual(Numpy.DT_FLOAT64, chisquare.Dtype)
        AhkTest.AssertEqual(Numpy.DT_INT32, geometric.Dtype)
        AhkTest.AssertEqual(Numpy.DT_INT32, zipf.Dtype)
        AhkTest.AssertEqual(Numpy.DT_FLOAT64, wald.Dtype)
        AhkTest.AssertEqual(Numpy.DT_INT32, hypergeometric.Dtype)

        mean := Numpy.Array([1.0, -1.0], [2], Numpy.DT_FLOAT32)
        covariance := Numpy.Array(
            [2.0, 0.5, 0.5, 1.0], [2, 2], Numpy.DT_FLOAT32)
        multivariate := Numpy.Random.MultivariateNormal(
            mean, covariance, 4)
        NumpyAssertArrayNear([4, 2], multivariate.Shape, 0)
        AhkTest.AssertEqual(Numpy.DT_FLOAT64, multivariate.Dtype)

        Numpy.Random.Seed(123)
        bytesFirst := Numpy.Random.Bytes(16)
        Numpy.Random.Seed(123)
        bytesSecond := Numpy.Random.Bytes(16)
        AhkTest.AssertEqual(16, bytesFirst.Size)
        AhkTest.AssertEqual(16, bytesSecond.Size)
        loop 16
            AhkTest.AssertEqual(
                NumGet(bytesFirst, A_Index - 1, "UChar"),
                NumGet(bytesSecond, A_Index - 1, "UChar"))
        emptyBytes := Numpy.Random.Bytes(0)
        AhkTest.AssertEqual(0, emptyBytes.Size)

        rejected := false
        try Numpy.Random.Hypergeometric([2], 2, 3, 6)
        catch ValueError as err {
            rejected := InStr(
                err.Message, "Numpy.Random.Hypergeometric") > 0
                && InStr(err.Message, "sample") > 0
        }
        AhkTest.AssertTrue(rejected,
            "special random facades must preserve native domain errors")

        rejectedBytes := false
        try Numpy.Random.Bytes(-1)
        catch ValueError as err
            rejectedBytes := InStr(err.Message, "Numpy.Random.Bytes") > 0
        AhkTest.AssertTrue(rejectedBytes,
            "Random.Bytes must expose negative-length errors")

        activeMemory := Numpy.AllocatedMemory()
        loop 64 {
            temporary := Numpy.Random.Bytes(31)
            temporary := 0
            temporaryArray := Numpy.Random.Chisquare([4], 2.0)
            temporaryArray := 0
        }
        AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
            "special random facades must release arrays and raw byte owners")

        multivariate := 0
        covariance := 0
        mean := 0
        hypergeometric := 0
        wald := 0
        zipf := 0
        geometric := 0
        chisquare := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "special random facades must release every native allocation")
        Numpy.Cleanup()
    }

    static TestAngleFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        negativeZero := NumpyDoubleFromBits(0x80000000)
        source := Numpy.Array([1.0, -1.0, negativeZero])

        radians := Numpy.Angle(source)
        NumpyAssertArrayNear([0.0, 3.141592653589793, 3.141592653589793],
            radians.ToArray(), 1e-15)
        degrees := Numpy.Angle(source, true)
        NumpyAssertArrayNear([0.0, 180.0, 180.0], degrees.ToArray(), 0.0)

        rejected := false
        try Numpy.Angle(0)
        catch TypeError as err {
            rejected := InStr(err.Message, "NdArray") > 0
        }
        AhkTest.AssertTrue(rejected,
            "Numpy.Angle must reject non-array inputs explicitly")

        loop 64 {
            temporary := Numpy.Angle(source)
            temporary := 0
        }
        degrees := 0
        radians := 0
        source := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "Numpy.Angle must release all native results")
        Numpy.Cleanup()
    }

    static TestRealImagFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        realSource := Numpy.Array([1.0, -2.0, 3.5])
        realValues := Numpy.Real(realSource)
        zeroImag := Numpy.Imag(realSource)
        AhkTest.AssertEqual(Numpy.DT_FLOAT64, realValues.Dtype)
        AhkTest.AssertEqual(Numpy.DT_FLOAT64, zeroImag.Dtype)
        NumpyAssertArrayNear([1.0, -2.0, 3.5], realValues.ToArray(), 0.0)
        NumpyAssertArrayNear([0.0, 0.0, 0.0], zeroImag.ToArray(), 0.0)

        complexSource := Numpy.Zeros([3], Numpy.DT_COMPLEX128)
        index := Buffer(8, 0)
        pairs := [[1.0, -2.0], [-3.0, 4.0], [5.5, -6.5]]
        for zeroBased, pair in pairs {
            NumPut("Int64", zeroBased - 1, index, 0)
            address := DllCall(Numpy.Proc("cnp_array_at"),
                "Ptr", complexSource.Handle, "Ptr", index.Ptr, "Ptr")
            AhkTest.AssertTrue(address != 0,
                "complex source element address must exist")
            NumPut("Double", pair[1], address, 0)
            NumPut("Double", pair[2], address, 8)
        }
        complexReal := Numpy.Real(complexSource)
        complexImag := Numpy.Imag(complexSource)
        complexSource := 0
        AhkTest.AssertEqual(Numpy.DT_FLOAT64, complexReal.Dtype)
        AhkTest.AssertEqual(Numpy.DT_FLOAT64, complexImag.Dtype)
        NumpyAssertArrayNear([1.0, -3.0, 5.5],
            complexReal.ToArray(), 0.0)
        NumpyAssertArrayNear([-2.0, 4.0, -6.5],
            complexImag.ToArray(), 0.0)

        rejectedReal := false
        try Numpy.Real(0)
        catch TypeError as err
            rejectedReal := InStr(err.Message, "NdArray") > 0
        AhkTest.AssertTrue(rejectedReal,
            "Numpy.Real must reject non-array inputs explicitly")
        rejectedImag := false
        try Numpy.Imag(0)
        catch TypeError as err
            rejectedImag := InStr(err.Message, "NdArray") > 0
        AhkTest.AssertTrue(rejectedImag,
            "Numpy.Imag must reject non-array inputs explicitly")

        loop 64 {
            temporaryReal := Numpy.Real(realSource)
            temporaryImag := Numpy.Imag(realSource)
            temporaryReal := 0
            temporaryImag := 0
        }
        complexReal := 0
        complexImag := 0
        zeroImag := 0
        realValues := 0
        realSource := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "Numpy.Real/Imag must release arrays and component views")
        Numpy.Cleanup()
    }

    static TestRealIfCloseFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        closeSource := Numpy.Zeros([3], Numpy.DT_COMPLEX128)
        index := Buffer(8, 0)
        pairs := [[1.0, 1.0e-16], [-2.0, -2.0e-16], [3.5, 0.0]]
        for zeroBased, pair in pairs {
            NumPut("Int64", zeroBased - 1, index, 0)
            address := DllCall(Numpy.Proc("cnp_array_at"),
                "Ptr", closeSource.Handle, "Ptr", index.Ptr, "Ptr")
            AhkTest.AssertTrue(address != 0,
                "real_if_close source element address must exist")
            NumPut("Double", pair[1], address, 0)
            NumPut("Double", pair[2], address, 8)
        }
        closeResult := Numpy.RealIfClose(closeSource, 100)
        AhkTest.AssertEqual(Numpy.DT_FLOAT64, closeResult.Dtype)
        closeSource := 0
        NumpyAssertArrayNear([1.0, -2.0, 3.5],
            closeResult.ToArray(), 0.0)

        noncloseSource := Numpy.Zeros([1], Numpy.DT_COMPLEX128)
        NumPut("Int64", 0, index, 0)
        noncloseAddress := DllCall(Numpy.Proc("cnp_array_at"),
            "Ptr", noncloseSource.Handle, "Ptr", index.Ptr, "Ptr")
        NumPut("Double", 4.0, noncloseAddress, 0)
        NumPut("Double", 1.0, noncloseAddress, 8)
        noncloseResult := Numpy.RealIfClose(noncloseSource)
        AhkTest.AssertEqual(noncloseSource.Handle, noncloseResult.Handle,
            "nonclose complex input must retain the same native array")
        AhkTest.AssertEqual(Numpy.DT_COMPLEX128, noncloseResult.Dtype)

        zeroTolSource := Numpy.Zeros([1], Numpy.DT_COMPLEX128)
        zeroTolResult := Numpy.RealIfClose(zeroTolSource, 0)
        AhkTest.AssertEqual(zeroTolSource.Handle, zeroTolResult.Handle,
            "tol zero must use absolute strict-less semantics")
        AhkTest.AssertEqual(Numpy.DT_COMPLEX128, zeroTolResult.Dtype)

        realSource := Numpy.Array([7.0, -8.0])
        realResult := Numpy.RealIfClose(realSource)
        AhkTest.AssertEqual(realSource.Handle, realResult.Handle,
            "real input must retain the same native array")
        NumpyAssertArrayNear([7.0, -8.0], realResult.ToArray(), 0.0)

        rejected := false
        try Numpy.RealIfClose(0)
        catch TypeError as err
            rejected := InStr(err.Message, "NdArray") > 0
        AhkTest.AssertTrue(rejected,
            "Numpy.RealIfClose must reject non-array inputs explicitly")

        loop 64 {
            temporary := Numpy.RealIfClose(realSource)
            temporary := 0
        }
        realResult := 0
        realSource := 0
        zeroTolResult := 0
        zeroTolSource := 0
        noncloseResult := 0
        noncloseSource := 0
        closeResult := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "Numpy.RealIfClose must release aliases and component views")
        Numpy.Cleanup()
    }

    static TestConvolveCorrelateFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        left := Numpy.IntArray([1, 2])
        right := Numpy.IntArray([10, 20, 30])
        matrix := Numpy.IntArray([1, 2], [1, 2])
        convolution := 0
        convolutionSame := 0
        convolutionValid := 0
        correlation := 0
        correlationSame := 0
        correlationFull := 0
        swappedCorrelation := 0
        complexLeft := 0
        complexRight := 0
        complexConvolution := 0
        complexCorrelation := 0
        temporary := 0

        try {
            convolution := Numpy.Convolve(left, right)
            convolutionSame := Numpy.Convolve(left, right, "SAME")
            convolutionValid := Numpy.Convolve(left, right, "valid")
            NumpyAssertArrayNear(
                [10, 40, 70, 60], convolution.ToArray(), 0)
            NumpyAssertArrayNear(
                [10, 40, 70], convolutionSame.ToArray(), 0)
            NumpyAssertArrayNear(
                [40, 70], convolutionValid.ToArray(), 0)
            AhkTest.AssertEqual(Numpy.DT_LONGLONG, convolution.Dtype)

            correlation := Numpy.Correlate(left, right)
            correlationSame := Numpy.Correlate(left, right, "same")
            correlationFull := Numpy.Correlate(left, right, "full")
            swappedCorrelation := Numpy.Correlate(right, left, "full")
            NumpyAssertArrayNear([80, 50], correlation.ToArray(), 0)
            NumpyAssertArrayNear(
                [80, 50, 20], correlationSame.ToArray(), 0)
            NumpyAssertArrayNear(
                [30, 80, 50, 20], correlationFull.ToArray(), 0)
            NumpyAssertArrayNear(
                [20, 50, 80, 30], swappedCorrelation.ToArray(), 0)

            complexLeft := Numpy.Zeros([2], Numpy.DT_COMPLEX128)
            complexRight := Numpy.Zeros([3], Numpy.DT_COMPLEX128)
            index := Buffer(8, 0)
            for source, pairs in Map(
                complexLeft, [[1.0, 2.0], [3.0, 4.0]],
                complexRight, [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]) {
                for oneBased, pair in pairs {
                    NumPut("Int64", oneBased - 1, index, 0)
                    address := DllCall(Numpy.Proc("cnp_array_at"),
                        "Ptr", source.Handle, "Ptr", index.Ptr, "Ptr")
                    AhkTest.AssertTrue(address != 0,
                        "complex signal element address must exist")
                    NumPut("Double", pair[1], address, 0)
                    NumPut("Double", pair[2], address, 8)
                }
            }
            complexConvolution := Numpy.Convolve(
                complexLeft, complexRight)
            complexCorrelation := Numpy.Correlate(
                complexLeft, complexRight, "full")
            complexLeft := 0
            complexRight := 0
            expectedConvolution := [
                [-7.0, 16.0], [-18.0, 60.0],
                [-22.0, 80.0], [-13.0, 66.0]]
            expectedCorrelation := [
                [29.0, 8.0], [90.0, 12.0],
                [70.0, 8.0], [39.0, 2.0]]
            for result, pairs in Map(
                complexConvolution, expectedConvolution,
                complexCorrelation, expectedCorrelation) {
                for oneBased, pair in pairs {
                    NumPut("Int64", oneBased - 1, index, 0)
                    address := DllCall(Numpy.Proc("cnp_array_at"),
                        "Ptr", result.Handle, "Ptr", index.Ptr, "Ptr")
                    AhkTest.AssertTrue(address != 0,
                        "complex result element address must exist")
                    AhkTest.AssertEqual(
                        pair[1], NumGet(address, 0, "Double"))
                    AhkTest.AssertEqual(
                        pair[2], NumGet(address, 8, "Double"))
                }
            }

            rejected := false
            try Numpy.Convolve(0, right)
            catch TypeError as err
                rejected := InStr(err.Message, "NdArray") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Convolve must reject non-array inputs")

            rejected := false
            try Numpy.Correlate(left, right, 0)
            catch TypeError as err
                rejected := InStr(err.Message, "mode") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Correlate must reject non-string modes")

            rejected := false
            try Numpy.Convolve(left, right, "unexpected")
            catch ValueError as err
                rejected := InStr(err.Message, "mode") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Convolve must reject unknown mode names")

            rejected := false
            try Numpy.Convolve(matrix, right)
            catch Error as err
                rejected := InStr(err.Message, "one-dimensional") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Convolve must surface native rank errors")

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Convolve(left, right)
                temporary := 0
                temporary := Numpy.Correlate(left, right)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "convolve/correlate calls must release every result")
        } finally {
            err := 0
            temporary := 0
            complexCorrelation := 0
            complexConvolution := 0
            complexRight := 0
            complexLeft := 0
            swappedCorrelation := 0
            correlationFull := 0
            correlationSame := 0
            correlation := 0
            convolutionValid := 0
            convolutionSame := 0
            convolution := 0
            matrix := 0
            right := 0
            left := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "convolve/correlate facade must release operands and results")
        Numpy.Cleanup()
    }

    static TestClip() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")

        shapeBuf := Buffer(8, 0)
        NumPut("Int64", 5, shapeBuf, 0)
        dataBuf := Buffer(40, 0)
        NumPut("Double", -2.0, dataBuf, 0)
        NumPut("Double", -1.0, dataBuf, 8)
        NumPut("Double", 0.0, dataBuf, 16)
        NumPut("Double", 1.0, dataBuf, 24)
        NumPut("Double", 2.0, dataBuf, 32)
        h := DllCall(dllPath "\cnp_ahk_from_doubles", "Int", 1, "Ptr", shapeBuf, "Ptr", dataBuf, "Int64", 5, "Ptr")

        ; clip to [-1, 1]
        hr := DllCall(dllPath "\cnp_ahk_clip", "Ptr", h, "Double", -1.0, "Double", 1.0, "Ptr")
        AhkTest.AssertTrue(hr != 0)
        outBuf := Buffer(40, 0)
        DllCall(dllPath "\cnp_ahk_get_doubles", "Ptr", hr, "Ptr", outBuf, "Int64", 5, "Int64")
        AhkTest.AssertEqual(-1.0, NumGet(outBuf, 0, "Double"))
        AhkTest.AssertEqual(-1.0, NumGet(outBuf, 8, "Double"))
        AhkTest.AssertEqual(0.0, NumGet(outBuf, 16, "Double"))
        AhkTest.AssertEqual(1.0, NumGet(outBuf, 24, "Double"))
        AhkTest.AssertEqual(1.0, NumGet(outBuf, 32, "Double"))
        DllCall(dllPath "\cnp_ahk_free", "Ptr", hr, "Int")

        DllCall(dllPath "\cnp_ahk_free", "Ptr", h, "Int")
        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }

    static TestHighLevelClipArrayBounds() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.IntArray([-9, 1, 12, 3, 15, 7], [2, 3])
        minimum := Numpy.IntArray([0, 2, 4])
        maximum := Numpy.IntArray([5, 8], [2, 1])
        arrayResult := 0
        scalarResult := 0
        upperOnlyResult := 0

        try {
            arrayResult := Numpy.Clip(source, minimum, maximum)
            NumpyAssertArrayNear([2, 3], arrayResult.Shape, 0)
            NumpyAssertArrayNear(
                [0, 2, 5, 3, 8, 7], arrayResult.ToArray(), 0)
            AhkTest.AssertEqual(Numpy.DT_LONGLONG, arrayResult.Dtype)

            scalarResult := source.Clip(0, 6)
            NumpyAssertArrayNear(
                [0, 1, 6, 3, 6, 6], scalarResult.ToArray(), 0)
            AhkTest.AssertEqual(Numpy.DT_LONGLONG, scalarResult.Dtype)

            upperOnlyResult := source.Clip(, 4)
            NumpyAssertArrayNear(
                [-9, 1, 4, 3, 4, 4], upperOnlyResult.ToArray(), 0)

            beforeError := Numpy.AllocatedMemory()
            rejected := false
            try Numpy.Clip(source)
            catch Error as err
                rejected := InStr(err.Message, "a_min") > 0
                    && InStr(err.Message, "a_max") > 0
            AhkTest.AssertTrue(rejected,
                "clip without either bound must expose the native error")
            AhkTest.AssertEqual(beforeError, Numpy.AllocatedMemory(),
                "failed clip must not retain temporary scalar arrays")
        } finally {
            err := 0
            upperOnlyResult := 0
            scalarResult := 0
            arrayResult := 0
            maximum := 0
            minimum := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "clip facade must release arrays and temporary scalar bounds")
        Numpy.Cleanup()
    }

    static TestAroundFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.Array([2.5, 3.5, -2.5, -3.5, -0.0])
        integers := Numpy.IntArray([15, 25, -15, -25])
        rounded := 0
        tens := 0
        roundAlias := 0
        tensAlias := 0

        try {
            rounded := Numpy.Around(source)
            NumpyAssertArrayNear(
                [2.0, 4.0, -2.0, -4.0, -0.0],
                rounded.ToArray(), 0)
            AhkTest.AssertEqual(Numpy.DT_FLOAT64, rounded.Dtype)

            tens := integers.Around(-1)
            NumpyAssertArrayNear(
                [20, 20, -20, -20], tens.ToArray(), 0)
            AhkTest.AssertEqual(Numpy.DT_LONGLONG, tens.Dtype)

            roundAlias := Numpy.Round(source)
            NumpyAssertArrayNear(
                rounded.ToArray(), roundAlias.ToArray(), 0)
            AhkTest.AssertEqual(Numpy.DT_FLOAT64, roundAlias.Dtype)

            tensAlias := integers.Round(-1)
            NumpyAssertArrayNear(tens.ToArray(), tensAlias.ToArray(), 0)
            AhkTest.AssertEqual(Numpy.DT_LONGLONG, tensAlias.Dtype)

            rejected := false
            try Numpy.Around(source, 1.5)
            catch TypeError as err
                rejected := InStr(err.Message, "decimals") > 0
            AhkTest.AssertTrue(rejected,
                "around must reject non-integer decimals explicitly")
        } finally {
            tensAlias := 0
            roundAlias := 0
            tens := 0
            rounded := 0
            integers := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "around facade must release every result")
        Numpy.Cleanup()
    }

    static TestUnaryRoundingFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.Array([-2.5, -1.5, -0.5, -0.0,
            0.0, 0.5, 1.5, 2.5])
        rintResult := 0
        fixResult := 0
        truncResult := 0
        floorResult := 0
        ceilResult := 0

        try {
            rintResult := Numpy.Rint(source)
            NumpyAssertArrayNear(
                [-2, -2, -0.0, -0.0, 0, 0, 2, 2],
                rintResult.ToArray(), 0)
            AhkTest.AssertEqual(Numpy.DT_FLOAT64, rintResult.Dtype)

            fixResult := source.Fix()
            truncResult := Numpy.Trunc(source)
            NumpyAssertArrayNear(
                [-2, -1, -0.0, -0.0, 0, 0, 1, 2],
                fixResult.ToArray(), 0)
            NumpyAssertArrayNear(
                fixResult.ToArray(), truncResult.ToArray(), 0)

            floorResult := source.Floor()
            NumpyAssertArrayNear(
                [-3, -2, -1, -0.0, 0, 0, 1, 2],
                floorResult.ToArray(), 0)

            ceilResult := Numpy.Ceil(source)
            NumpyAssertArrayNear(
                [-2, -1, -0.0, -0.0, 0, 1, 2, 3],
                ceilResult.ToArray(), 0)
        } finally {
            ceilResult := 0
            floorResult := 0
            truncResult := 0
            fixResult := 0
            rintResult := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "unary rounding facade must release every result")
        Numpy.Cleanup()
    }

    static TestAbsoluteFamilyFacade() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.Array([-2.5, -0.0, 0.0, 2.5])
        integers := Numpy.IntArray([-2, -1, 0, 1, 2])
        absoluteResult := 0
        fabsResult := 0
        fabsAlias := 0

        try {
            absoluteResult := Numpy.Absolute(source)
            NumpyAssertArrayNear(
                [2.5, 0.0, 0.0, 2.5], absoluteResult.ToArray(), 0)
            AhkTest.AssertEqual(Numpy.DT_FLOAT64, absoluteResult.Dtype)

            fabsResult := Numpy.Fabs(integers)
            NumpyAssertArrayNear(
                [2, 1, 0, 1, 2], fabsResult.ToArray(), 0)
            AhkTest.AssertEqual(Numpy.DT_FLOAT64, fabsResult.Dtype)

            fabsAlias := source.Fabs()
            NumpyAssertArrayNear(
                absoluteResult.ToArray(), fabsAlias.ToArray(), 0)
        } finally {
            fabsAlias := 0
            fabsResult := 0
            absoluteResult := 0
            integers := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "absolute family facade must release every result")
        Numpy.Cleanup()
    }

    static TestAddFacadeBroadcastingAndLifecycle() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        left := Numpy.IntArray([1, 2], [2, 1])
        right := Numpy.Array(
            [0.5, -1.5, 2.0], [1, 3], Numpy.DT_FLOAT32)
        invalid := Numpy.Array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])
        output := Numpy.Zeros([2, 3], Numpy.DT_FLOAT32)
        integerOutput := Numpy.IntArray(
            [11, 12, 13, 14, 15, 16], [2, 3])
        staticResult := 0
        instanceResult := 0
        returnedOutput := 0
        temporary := 0

        try {
            staticResult := Numpy.Add(left, right)
            instanceResult := left.Add(right)
            expected := [1.5, -0.5, 3.0, 2.5, 0.5, 4.0]
            NumpyAssertArrayNear([2, 3], staticResult.Shape, 0)
            NumpyAssertArrayNear(expected, staticResult.ToArray(), 0)
            NumpyAssertArrayNear(expected, instanceResult.ToArray(), 0)
            AhkTest.AssertEqual(Numpy.DT_FLOAT64, staticResult.Dtype)
            AhkTest.AssertEqual(Numpy.DT_FLOAT64, instanceResult.Dtype)

            returnedOutput := Numpy.Add(left, right, output)
            AhkTest.AssertEqual(output.Handle, returnedOutput.Handle,
                "Numpy.Add out must return the supplied destination")
            NumpyAssertArrayNear(expected, output.ToArray(), 0)
            AhkTest.AssertEqual(Numpy.DT_FLOAT32, output.Dtype)

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Add(left, right)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated allocating Add calls must release every result")
            loop 64
                Numpy.Add(left, right, output)
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated Add out calls must not allocate native memory")

            beforeError := Numpy.AllocatedMemory()
            rejected := false
            try Numpy.Add(left, invalid)
            catch Error as err
                rejected := InStr(err.Message, "cannot be broadcast together") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Add must expose the native broadcast error")
            AhkTest.AssertEqual(beforeError, Numpy.AllocatedMemory(),
                "failed Numpy.Add must not retain native memory")

            rejected := false
            try Numpy.Add(left, right, integerOutput)
            catch Error as err
                rejected := InStr(err.Message, "same_kind") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Add out must expose the native casting error")
            NumpyAssertArrayNear(
                [11, 12, 13, 14, 15, 16], integerOutput.ToArray(), 0)
        } finally {
            err := 0
            temporary := 0
            returnedOutput := 0
            instanceResult := 0
            staticResult := 0
            integerOutput := 0
            output := 0
            invalid := 0
            right := 0
            left := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "Add facade must release all operands and results")
        Numpy.Cleanup()
    }

    static TestSubtractMultiplyFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        left := Numpy.IntArray([7, -3], [2, 1])
        right := Numpy.IntArray([2, 5, -4], [1, 3])
        subtractStatic := 0
        subtractInstance := 0
        multiplyStatic := 0
        multiplyInstance := 0
        temporary := 0

        try {
            subtractStatic := Numpy.Subtract(left, right)
            subtractInstance := left.Subtract(right)
            multiplyStatic := Numpy.Multiply(left, right)
            multiplyInstance := left.Multiply(right)
            expectedSubtract := [5, 2, 11, -5, -8, 1]
            expectedMultiply := [14, 35, -28, -6, -15, 12]
            for result in [subtractStatic, subtractInstance] {
                NumpyAssertArrayNear([2, 3], result.Shape, 0)
                NumpyAssertArrayNear(expectedSubtract, result.ToArray(), 0)
                AhkTest.AssertEqual(Numpy.DT_LONGLONG, result.Dtype)
            }
            for result in [multiplyStatic, multiplyInstance] {
                NumpyAssertArrayNear([2, 3], result.Shape, 0)
                NumpyAssertArrayNear(expectedMultiply, result.ToArray(), 0)
                AhkTest.AssertEqual(Numpy.DT_LONGLONG, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Subtract(left, right)
                temporary := Numpy.Multiply(left, right)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated subtract/multiply calls must release results")

            rejected := false
            try Numpy.Subtract(1, right)
            catch TypeError as err
                rejected := InStr(err.Message, "left input") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Subtract must reject a non-array left input")

            rejected := false
            try Numpy.Multiply(left, 1)
            catch TypeError as err
                rejected := InStr(err.Message, "right input") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Multiply must reject a non-array right input")

            left := 0
            right := 0
            NumpyAssertArrayNear(
                expectedSubtract, subtractStatic.ToArray(), 0)
            NumpyAssertArrayNear(
                expectedMultiply, multiplyStatic.ToArray(), 0)
        } finally {
            err := 0
            temporary := 0
            multiplyInstance := 0
            multiplyStatic := 0
            subtractInstance := 0
            subtractStatic := 0
            right := 0
            left := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "subtract/multiply facade must release operands and results")
        Numpy.Cleanup()
    }

    static TestDivideTrueDivideFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        left := Numpy.IntArray([7, -3], [2, 1])
        right := Numpy.IntArray([2, 5, -4], [1, 3])
        divideStatic := 0
        divideInstance := 0
        trueDivideStatic := 0
        trueDivideInstance := 0
        temporary := 0
        numerator := 0
        denominator := 0
        zeroDivision := 0

        try {
            divideStatic := Numpy.Divide(left, right)
            divideInstance := left.Divide(right)
            trueDivideStatic := Numpy.TrueDivide(left, right)
            trueDivideInstance := left.TrueDivide(right)
            expected := [3.5, 1.4, -1.75, -1.5, -0.6, 0.75]
            for result in [
                divideStatic,
                divideInstance,
                trueDivideStatic,
                trueDivideInstance
            ] {
                NumpyAssertArrayNear([2, 3], result.Shape, 0)
                NumpyAssertArrayNear(expected, result.ToArray(), 0)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            positiveInfinity := NumpyDoubleFromBits(0x7ff00000)
            negativeInfinity := NumpyDoubleFromBits(0xfff00000)
            negativeZero := NumpyDoubleFromBits(0x80000000)
            numerator := Numpy.Array([1.0, 1.0, -1.0, -1.0])
            denominator := Numpy.Array([0.0, negativeZero, 0.0, negativeZero])
            zeroDivision := Numpy.TrueDivide(
                numerator, denominator).ToArray()
            AhkTest.AssertEqual(positiveInfinity, zeroDivision[1])
            AhkTest.AssertEqual(negativeInfinity, zeroDivision[2])
            AhkTest.AssertEqual(negativeInfinity, zeroDivision[3])
            AhkTest.AssertEqual(positiveInfinity, zeroDivision[4])

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Divide(left, right)
                temporary := Numpy.TrueDivide(left, right)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated divide calls must release results")

            rejected := false
            try Numpy.Divide(1, right)
            catch TypeError as err
                rejected := InStr(err.Message, "left input") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Divide must reject a non-array left input")

            rejected := false
            try Numpy.TrueDivide(left, 1)
            catch TypeError as err
                rejected := InStr(err.Message, "right input") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.TrueDivide must reject a non-array right input")

            left := 0
            right := 0
            NumpyAssertArrayNear(expected, divideStatic.ToArray(), 0)
            NumpyAssertArrayNear(expected, trueDivideStatic.ToArray(), 0)
        } finally {
            err := 0
            zeroDivision := 0
            denominator := 0
            numerator := 0
            temporary := 0
            trueDivideInstance := 0
            trueDivideStatic := 0
            divideInstance := 0
            divideStatic := 0
            right := 0
            left := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "divide facades must release operands and results")
        Numpy.Cleanup()
    }

    static TestFloorDivideFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        left := Numpy.IntArray([-7, 7], [2, 1])
        right := Numpy.IntArray([3, -3, 2], [1, 3])
        staticResult := 0
        instanceResult := 0
        temporary := 0

        try {
            staticResult := Numpy.FloorDivide(left, right)
            instanceResult := left.FloorDivide(right)
            expected := [-3, 2, -4, 2, -3, 3]
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear([2, 3], result.Shape, 0)
                NumpyAssertArrayNear(expected, result.ToArray(), 0)
                AhkTest.AssertEqual(Numpy.DT_LONGLONG, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.FloorDivide(left, right)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated floor divide calls must release results")

            rejected := false
            try Numpy.FloorDivide(1, right)
            catch TypeError as err
                rejected := InStr(err.Message, "left input") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.FloorDivide must reject a non-array left input")

            left := 0
            right := 0
            NumpyAssertArrayNear(expected, staticResult.ToArray(), 0)
        } finally {
            err := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            right := 0
            left := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "floor divide facade must release operands and results")
        Numpy.Cleanup()
    }

    static TestModRemainderFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        left := Numpy.IntArray([-7, 7], [2, 1])
        right := Numpy.IntArray([3, -3, 2], [1, 3])
        modStatic := 0
        modInstance := 0
        remainderStatic := 0
        remainderInstance := 0
        temporary := 0

        try {
            modStatic := Numpy.Mod(left, right)
            modInstance := left.Mod(right)
            remainderStatic := Numpy.Remainder(left, right)
            remainderInstance := left.Remainder(right)
            expected := [2, -1, 1, 1, -2, 1]
            for result in [
                modStatic,
                modInstance,
                remainderStatic,
                remainderInstance
            ] {
                NumpyAssertArrayNear([2, 3], result.Shape, 0)
                NumpyAssertArrayNear(expected, result.ToArray(), 0)
                AhkTest.AssertEqual(Numpy.DT_LONGLONG, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Mod(left, right)
                temporary := Numpy.Remainder(left, right)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated mod/remainder calls must release results")

            rejected := false
            try Numpy.Mod(1, right)
            catch TypeError as err
                rejected := InStr(err.Message, "left input") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Mod must reject a non-array left input")

            rejected := false
            try Numpy.Remainder(left, 1)
            catch TypeError as err
                rejected := InStr(err.Message, "right input") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Remainder must reject a non-array right input")

            left := 0
            right := 0
            NumpyAssertArrayNear(expected, modStatic.ToArray(), 0)
            NumpyAssertArrayNear(expected, remainderStatic.ToArray(), 0)
        } finally {
            err := 0
            temporary := 0
            remainderInstance := 0
            remainderStatic := 0
            modInstance := 0
            modStatic := 0
            right := 0
            left := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "mod/remainder facade must release operands and results")
        Numpy.Cleanup()
    }

    static TestFmodFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        left := Numpy.IntArray([-7, 7], [2, 1])
        right := Numpy.IntArray([3, -3, 2], [1, 3])
        staticResult := 0
        instanceResult := 0
        temporary := 0

        try {
            staticResult := Numpy.Fmod(left, right)
            instanceResult := left.Fmod(right)
            expected := [-1, -1, -1, 1, 1, 1]
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear([2, 3], result.Shape, 0)
                NumpyAssertArrayNear(expected, result.ToArray(), 0)
                AhkTest.AssertEqual(Numpy.DT_LONGLONG, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Fmod(left, right)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated fmod calls must release results")

            rejected := false
            try Numpy.Fmod(1, right)
            catch TypeError as err
                rejected := InStr(err.Message, "left input") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Fmod must reject a non-array left input")

            rejected := false
            try Numpy.Fmod(left, 1)
            catch TypeError as err
                rejected := InStr(err.Message, "right input") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Fmod must reject a non-array right input")

            left := 0
            right := 0
            NumpyAssertArrayNear(expected, staticResult.ToArray(), 0)
        } finally {
            err := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            right := 0
            left := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "fmod facade must release operands and results")
        Numpy.Cleanup()
    }

    static TestNegativePositiveFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.IntArray([-7, 0, 7], [3])
        negativeStatic := 0
        negativeInstance := 0
        positiveStatic := 0
        positiveInstance := 0
        temporary := 0
        boolSource := 0

        try {
            negativeStatic := Numpy.Negative(source)
            negativeInstance := source.Negative()
            positiveStatic := Numpy.Positive(source)
            positiveInstance := source.Positive()
            for result in [negativeStatic, negativeInstance] {
                NumpyAssertArrayNear([7, 0, -7], result.ToArray(), 0)
                AhkTest.AssertEqual(Numpy.DT_LONGLONG, result.Dtype)
            }
            for result in [positiveStatic, positiveInstance] {
                NumpyAssertArrayNear([-7, 0, 7], result.ToArray(), 0)
                AhkTest.AssertEqual(Numpy.DT_LONGLONG, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Negative(source)
                temporary := Numpy.Positive(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated negative/positive calls must release results")

            rejected := false
            try Numpy.Negative(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Negative must reject a non-array source")

            rejected := false
            try Numpy.Positive(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Positive must reject a non-array source")

            boolSource := Numpy.Zeros([2], Numpy.DT_BOOL)
            rejected := false
            try Numpy.Negative(boolSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Negative") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Negative must preserve the native dtype TypeError")

            rejected := false
            try Numpy.Positive(boolSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Positive") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Positive must preserve the native dtype TypeError")

            source := 0
            NumpyAssertArrayNear([7, 0, -7], negativeStatic.ToArray(), 0)
            NumpyAssertArrayNear([-7, 0, 7], positiveStatic.ToArray(), 0)
        } finally {
            err := 0
            boolSource := 0
            temporary := 0
            positiveInstance := 0
            positiveStatic := 0
            negativeInstance := 0
            negativeStatic := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "negative/positive facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestSignFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.IntArray([-7, 0, 7], [3])
        staticResult := 0
        instanceResult := 0
        temporary := 0
        boolSource := 0

        try {
            staticResult := Numpy.Sign(source)
            instanceResult := source.Sign()
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear([-1, 0, 1], result.ToArray(), 0)
                AhkTest.AssertEqual(Numpy.DT_LONGLONG, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Sign(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated sign calls must release results")

            rejected := false
            try Numpy.Sign(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Sign must reject a non-array source")

            boolSource := Numpy.Zeros([2], Numpy.DT_BOOL)
            rejected := false
            try Numpy.Sign(boolSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Sign") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Sign must preserve the native dtype TypeError")

            source := 0
            NumpyAssertArrayNear([-1, 0, 1], staticResult.ToArray(), 0)
        } finally {
            err := 0
            boolSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "sign facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestReciprocalFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.IntArray([-2, -1, 1, 2], [4])
        staticResult := 0
        instanceResult := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Reciprocal(source)
            instanceResult := source.Reciprocal()
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear([0, -1, 1, 0], result.ToArray(), 0)
                AhkTest.AssertEqual(Numpy.DT_LONGLONG, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Reciprocal(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated reciprocal calls must release results")

            rejected := false
            try Numpy.Reciprocal(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Reciprocal must reject a non-array source")

            timedeltaSource := Numpy.Zeros([2], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Reciprocal(timedeltaSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Reciprocal") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Reciprocal must preserve the native dtype TypeError")

            source := 0
            NumpyAssertArrayNear([0, -1, 1, 0], staticResult.ToArray(), 0)
        } finally {
            err := 0
            timedeltaSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "reciprocal facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestSquareFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.IntArray([-3, -1, 0, 1, 3], [5])
        staticResult := 0
        instanceResult := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Square(source)
            instanceResult := source.Square()
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear([9, 1, 0, 1, 9],
                    result.ToArray(), 0)
                AhkTest.AssertEqual(Numpy.DT_LONGLONG, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Square(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated square calls must release results")

            rejected := false
            try Numpy.Square(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Square must reject a non-array source")

            timedeltaSource := Numpy.Zeros([2], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Square(timedeltaSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Square") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Square must preserve the native dtype TypeError")

            source := 0
            NumpyAssertArrayNear([9, 1, 0, 1, 9],
                staticResult.ToArray(), 0)
        } finally {
            err := 0
            timedeltaSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "square facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestSqrtFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.IntArray([0, 1, 4, 9, 16], [5])
        staticResult := 0
        instanceResult := 0
        intoSource := 0
        destination := 0
        returned := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Sqrt(source)
            instanceResult := source.Sqrt()
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear([0, 1, 2, 3, 4],
                    result.ToArray(), 0)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            intoSource := Numpy.Array([1.0, 4.0, 9.0, 16.0, 25.0])
            destination := Numpy.Zeros([5])
            returned := Numpy.Sqrt(intoSource, destination)
            AhkTest.AssertEqual(destination.Handle, returned.Handle,
                "Numpy.Sqrt out must return the destination")
            NumpyAssertArrayNear([1, 2, 3, 4, 5],
                destination.ToArray(), 0)

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Sqrt(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated sqrt calls must release results")

            rejected := false
            try Numpy.Sqrt(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Sqrt must reject a non-array source")

            timedeltaSource := Numpy.Zeros([2], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Sqrt(timedeltaSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Sqrt") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Sqrt must preserve the native dtype TypeError")

            source := 0
            NumpyAssertArrayNear([0, 1, 2, 3, 4],
                staticResult.ToArray(), 0)
        } finally {
            err := 0
            timedeltaSource := 0
            temporary := 0
            returned := 0
            destination := 0
            intoSource := 0
            instanceResult := 0
            staticResult := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "sqrt facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestCbrtFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.IntArray([-8, -1, 0, 1, 8], [5])
        staticResult := 0
        instanceResult := 0
        temporary := 0
        complexSource := 0

        try {
            staticResult := Numpy.Cbrt(source)
            instanceResult := source.Cbrt()
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear([-2, -1, 0, 1, 2],
                    result.ToArray(), 0)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Cbrt(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated cbrt calls must release results")

            rejected := false
            try Numpy.Cbrt(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Cbrt must reject a non-array source")

            complexSource := Numpy.Zeros([2], Numpy.DT_COMPLEX64)
            rejected := false
            try Numpy.Cbrt(complexSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Cbrt") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Cbrt must preserve the native complex TypeError")

            source := 0
            NumpyAssertArrayNear([-2, -1, 0, 1, 2],
                staticResult.ToArray(), 0)
        } finally {
            err := 0
            complexSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "cbrt facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestConjugateFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.IntArray([-3, -1, 0, 1, 3], [5])
        boolSource := Numpy.Array([0, 1], [2], Numpy.DT_BOOL)
        complexSource := Numpy.Zeros([2], Numpy.DT_COMPLEX64)
        timedeltaSource := Numpy.Zeros([2], Numpy.DT_TIMEDELTA)
        staticConj := 0
        staticConjugate := 0
        instanceConj := 0
        instanceConjugate := 0
        boolResult := 0
        complexResult := 0
        temporary := 0

        try {
            staticConj := Numpy.Conj(source)
            staticConjugate := Numpy.Conjugate(source)
            instanceConj := source.Conj()
            instanceConjugate := source.Conjugate()
            for result in [
                staticConj,
                staticConjugate,
                instanceConj,
                instanceConjugate
            ] {
                NumpyAssertArrayNear([-3, -1, 0, 1, 3],
                    result.ToArray(), 0)
                AhkTest.AssertEqual(source.Dtype, result.Dtype)
            }

            boolResult := Numpy.Conjugate(boolSource)
            NumpyAssertArrayNear([0, 1], boolResult.ToArray(), 0)
            AhkTest.AssertEqual(Numpy.DT_INT8, boolResult.Dtype)

            complexResult := Numpy.Conj(complexSource)
            AhkTest.AssertEqual(Numpy.DT_COMPLEX64, complexResult.Dtype)
            AhkTest.AssertEqual(2, complexResult.Size)

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Conj(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated conjugate calls must release results")

            rejected := false
            try Numpy.Conjugate(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Conjugate must reject a non-array source")

            rejected := false
            try Numpy.Conj(timedeltaSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Conj") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Conj must preserve the native dtype TypeError")

            source := 0
            NumpyAssertArrayNear([-3, -1, 0, 1, 3],
                staticConj.ToArray(), 0)
        } finally {
            err := 0
            temporary := 0
            complexResult := 0
            boolResult := 0
            instanceConjugate := 0
            instanceConj := 0
            staticConjugate := 0
            staticConj := 0
            timedeltaSource := 0
            complexSource := 0
            boolSource := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "conjugate facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestCosFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.IntArray([-3, -1, 0, 1, 3], [5])
        staticResult := 0
        instanceResult := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Cos(source)
            instanceResult := source.Cos()
            expected := [Cos(-3.0), Cos(-1.0), 1.0, Cos(1.0), Cos(3.0)]
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear(expected, result.ToArray(), 1e-12)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Cos(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated cos calls must release results")

            rejected := false
            try Numpy.Cos(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Cos must reject a non-array source")

            timedeltaSource := Numpy.Zeros([2], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Cos(timedeltaSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Cos") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Cos must preserve the native dtype TypeError")

            source := 0
            NumpyAssertArrayNear(expected, staticResult.ToArray(), 1e-12)
        } finally {
            err := 0
            timedeltaSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "cos facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestSinFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.IntArray([-3, -1, 0, 1, 3], [5])
        staticResult := 0
        instanceResult := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Sin(source)
            instanceResult := source.Sin()
            expected := [Sin(-3.0), Sin(-1.0), 0.0, Sin(1.0), Sin(3.0)]
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear(expected, result.ToArray(), 1e-12)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Sin(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated sin calls must release results")

            rejected := false
            try Numpy.Sin(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Sin must reject a non-array source")

            timedeltaSource := Numpy.Zeros([2], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Sin(timedeltaSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Sin") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Sin must preserve the native dtype TypeError")

            source := 0
            NumpyAssertArrayNear(expected, staticResult.ToArray(), 1e-12)
        } finally {
            err := 0
            timedeltaSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "sin facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestSinhFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.IntArray([-3, -1, 0, 1, 3], [5])
        staticResult := 0
        instanceResult := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Sinh(source)
            instanceResult := source.Sinh()
            expected := [
                (Exp(-3.0) - Exp(3.0)) / 2.0,
                (Exp(-1.0) - Exp(1.0)) / 2.0,
                0.0,
                (Exp(1.0) - Exp(-1.0)) / 2.0,
                (Exp(3.0) - Exp(-3.0)) / 2.0
            ]
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear(expected, result.ToArray(), 1e-12)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Sinh(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated sinh calls must release results")

            rejected := false
            try Numpy.Sinh(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Sinh must reject a non-array source")

            timedeltaSource := Numpy.Zeros([2], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Sinh(timedeltaSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Sinh") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Sinh must preserve the native dtype TypeError")

            source := 0
            NumpyAssertArrayNear(expected, staticResult.ToArray(), 1e-12)
        } finally {
            err := 0
            timedeltaSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "sinh facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestCoshFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.IntArray([-3, -1, 0, 1, 3], [5])
        staticResult := 0
        instanceResult := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Cosh(source)
            instanceResult := source.Cosh()
            expected := [
                (Exp(-3.0) + Exp(3.0)) / 2.0,
                (Exp(-1.0) + Exp(1.0)) / 2.0,
                1.0,
                (Exp(1.0) + Exp(-1.0)) / 2.0,
                (Exp(3.0) + Exp(-3.0)) / 2.0
            ]
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear(expected, result.ToArray(), 1e-12)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Cosh(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated cosh calls must release results")

            rejected := false
            try Numpy.Cosh(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Cosh must reject a non-array source")

            timedeltaSource := Numpy.Zeros([2], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Cosh(timedeltaSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Cosh") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Cosh must preserve the native dtype TypeError")

            source := 0
            NumpyAssertArrayNear(expected, staticResult.ToArray(), 1e-12)
        } finally {
            err := 0
            timedeltaSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "cosh facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestTanhFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.IntArray([-3, -1, 0, 1, 3], [5])
        staticResult := 0
        instanceResult := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Tanh(source)
            instanceResult := source.Tanh()
            expected := [
                (Exp(-6.0) - 1.0) / (Exp(-6.0) + 1.0),
                (Exp(-2.0) - 1.0) / (Exp(-2.0) + 1.0),
                0.0,
                (Exp(2.0) - 1.0) / (Exp(2.0) + 1.0),
                (Exp(6.0) - 1.0) / (Exp(6.0) + 1.0)
            ]
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear(expected, result.ToArray(), 1e-12)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Tanh(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated tanh calls must release results")

            rejected := false
            try Numpy.Tanh(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Tanh must reject a non-array source")

            timedeltaSource := Numpy.Zeros([2], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Tanh(timedeltaSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Tanh") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Tanh must preserve the native dtype TypeError")

            source := 0
            NumpyAssertArrayNear(expected, staticResult.ToArray(), 1e-12)
        } finally {
            err := 0
            timedeltaSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "tanh facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestArcsinhFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.IntArray([-3, -1, 0, 1, 3], [5])
        staticResult := 0
        instanceResult := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Arcsinh(source)
            instanceResult := source.Arcsinh()
            expected := [
                Ln(-3.0 + Sqrt(10.0)),
                Ln(-1.0 + Sqrt(2.0)),
                0.0,
                Ln(1.0 + Sqrt(2.0)),
                Ln(3.0 + Sqrt(10.0))
            ]
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear(expected, result.ToArray(), 1e-12)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Arcsinh(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated arcsinh calls must release results")

            rejected := false
            try Numpy.Arcsinh(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Arcsinh must reject a non-array source")

            timedeltaSource := Numpy.Zeros([2], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Arcsinh(timedeltaSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Arcsinh") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Arcsinh must preserve the native dtype TypeError")

            source := 0
            NumpyAssertArrayNear(expected, staticResult.ToArray(), 1e-12)
        } finally {
            err := 0
            timedeltaSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "arcsinh facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestArccoshFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.IntArray([1, 2, 3, 5], [4])
        staticResult := 0
        instanceResult := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Arccosh(source)
            instanceResult := source.Arccosh()
            expected := [
                0.0,
                Ln(2.0 + Sqrt(3.0)),
                Ln(3.0 + Sqrt(8.0)),
                Ln(5.0 + Sqrt(24.0))
            ]
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear(expected, result.ToArray(), 1e-12)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Arccosh(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated arccosh calls must release results")

            rejected := false
            try Numpy.Arccosh(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Arccosh must reject a non-array source")

            timedeltaSource := Numpy.Zeros([2], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Arccosh(timedeltaSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Arccosh") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Arccosh must preserve the native dtype TypeError")

            source := 0
            NumpyAssertArrayNear(expected, staticResult.ToArray(), 1e-12)
        } finally {
            err := 0
            timedeltaSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "arccosh facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestArctanhFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.Linspace(-0.75, 0.75, 5)
        staticResult := 0
        instanceResult := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Arctanh(source)
            instanceResult := source.Arctanh()
            expected := []
            for value in source.ToArray()
                expected.Push(0.5 * Ln((1.0 + value) / (1.0 - value)))
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear(expected, result.ToArray(), 1e-12)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Arctanh(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated arctanh calls must release results")

            rejected := false
            try Numpy.Arctanh(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Arctanh must reject a non-array source")

            timedeltaSource := Numpy.Zeros([2], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Arctanh(timedeltaSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Arctanh") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Arctanh must preserve the native dtype TypeError")

            source := 0
            NumpyAssertArrayNear(expected, staticResult.ToArray(), 1e-12)
        } finally {
            timedeltaSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "arctanh facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestExpFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.Linspace(-3.0, 3.0, 7)
        staticResult := 0
        instanceResult := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Exp(source)
            instanceResult := source.Exp()
            expected := []
            for value in source.ToArray()
                expected.Push(Exp(value))
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear(expected, result.ToArray(), 1e-12)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Exp(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated exp calls must release results")

            rejected := false
            try Numpy.Exp(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Exp must reject a non-array source")

            timedeltaSource := Numpy.Zeros([2], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Exp(timedeltaSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Exp") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Exp must preserve the native dtype TypeError")

            source := 0
            NumpyAssertArrayNear(expected, staticResult.ToArray(), 1e-12)
        } finally {
            timedeltaSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "exp facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestExp2FacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.Linspace(-3.0, 3.0, 7)
        staticResult := 0
        instanceResult := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Exp2(source)
            instanceResult := source.Exp2()
            expected := []
            for value in source.ToArray()
                expected.Push(2.0 ** value)
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear(expected, result.ToArray(), 1e-12)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Exp2(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated exp2 calls must release results")

            rejected := false
            try Numpy.Exp2(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Exp2 must reject a non-array source")

            timedeltaSource := Numpy.Zeros([2], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Exp2(timedeltaSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Exp2") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Exp2 must preserve the native dtype TypeError")

            source := 0
            NumpyAssertArrayNear(expected, staticResult.ToArray(), 1e-12)
        } finally {
            timedeltaSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "exp2 facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestExpm1FacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.Linspace(-3.0, 3.0, 7)
        staticResult := 0
        instanceResult := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Expm1(source)
            instanceResult := source.Expm1()
            expected := []
            for value in source.ToArray()
                expected.Push(Exp(value) - 1.0)
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear(expected, result.ToArray(), 1e-12)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Expm1(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated expm1 calls must release results")

            rejected := false
            try Numpy.Expm1(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Expm1 must reject a non-array source")

            timedeltaSource := Numpy.Zeros([2], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Expm1(timedeltaSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Expm1") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Expm1 must preserve the native dtype TypeError")

            source := 0
            NumpyAssertArrayNear(expected, staticResult.ToArray(), 1e-12)
        } finally {
            timedeltaSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "expm1 facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestLogFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.Linspace(0.25, 4.0, 7)
        staticResult := 0
        instanceResult := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Log(source)
            instanceResult := source.Log()
            expected := []
            for value in source.ToArray()
                expected.Push(Ln(value))
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear(expected, result.ToArray(), 1e-12)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Log(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated log calls must release results")

            rejected := false
            try Numpy.Log(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Log must reject a non-array source")

            timedeltaSource := Numpy.Zeros([2], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Log(timedeltaSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Log") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Log must preserve the native dtype TypeError")

            source := 0
            NumpyAssertArrayNear(expected, staticResult.ToArray(), 1e-12)
        } finally {
            timedeltaSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "log facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestLog2FacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.Array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
        staticResult := 0
        instanceResult := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Log2(source)
            instanceResult := source.Log2()
            expected := [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear(expected, result.ToArray(), 0)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Log2(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated log2 calls must release results")

            rejected := false
            try Numpy.Log2(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Log2 must reject a non-array source")

            timedeltaSource := Numpy.Zeros([2], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Log2(timedeltaSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Log2") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Log2 must preserve the native dtype TypeError")

            source := 0
            NumpyAssertArrayNear(expected, staticResult.ToArray(), 0)
        } finally {
            err := 0
            timedeltaSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "log2 facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestLog10FacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.Array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
        staticResult := 0
        instanceResult := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Log10(source)
            instanceResult := source.Log10()
            expected := [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear(expected, result.ToArray(), 0)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Log10(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated log10 calls must release results")

            rejected := false
            try Numpy.Log10(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Log10 must reject a non-array source")

            timedeltaSource := Numpy.Zeros([2], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Log10(timedeltaSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Log10") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Log10 must preserve the native dtype TypeError")

            source := 0
            NumpyAssertArrayNear(expected, staticResult.ToArray(), 0)
        } finally {
            err := 0
            timedeltaSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "log10 facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestLog1pFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.Array([-0.75, -0.5, 0.0, 1.0, 3.0, 7.0])
        staticResult := 0
        instanceResult := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Log1p(source)
            instanceResult := source.Log1p()
            expected := [-1.3862943611198906, -0.6931471805599453,
                0.0, 0.6931471805599453, 1.3862943611198906,
                2.0794415416798357]
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear(expected, result.ToArray(), 1e-12)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Log1p(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated log1p calls must release results")

            rejected := false
            try Numpy.Log1p(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Log1p must reject a non-array source")

            timedeltaSource := Numpy.Zeros([2], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Log1p(timedeltaSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Log1p") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Log1p must preserve the native dtype TypeError")

            source := 0
            NumpyAssertArrayNear(expected, staticResult.ToArray(), 1e-12)
        } finally {
            err := 0
            timedeltaSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "log1p facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestLogaddexpFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        left := Numpy.Array([-1000.0, -3.0, 0.0, 1.0, 3.0, 1000.0])
        right := Numpy.Array([1000.0, 4.0, 0.0, -1.0, -3.0, -1000.0])
        staticResult := 0
        instanceResult := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Logaddexp(left, right)
            instanceResult := left.Logaddexp(right)
            expected := [1000.0, 4.000911466453774, 0.6931471805599453,
                1.1269280110429725, 3.00247568513773, 1000.0]
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear(expected, result.ToArray(), 1e-12)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Logaddexp(left, right)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated logaddexp calls must release results")

            rejected := false
            try Numpy.Logaddexp(1, right)
            catch TypeError as err
                rejected := InStr(err.Message, "left") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Logaddexp must reject a non-array left input")

            rejected := false
            try Numpy.Logaddexp(left, 1)
            catch TypeError as err
                rejected := InStr(err.Message, "right") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Logaddexp must reject a non-array right input")

            timedeltaSource := Numpy.Zeros([2], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Logaddexp(timedeltaSource, right)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Logaddexp") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Logaddexp must preserve the native dtype TypeError")

            left := 0
            right := 0
            NumpyAssertArrayNear(expected, staticResult.ToArray(), 1e-12)
        } finally {
            err := 0
            timedeltaSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            right := 0
            left := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "logaddexp facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestLogaddexp2FacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        left := Numpy.Array([-1000.0, -3.0, 0.0, 1.0, 3.0, 1000.0])
        right := Numpy.Array([1000.0, 4.0, 0.0, -1.0, -3.0, -1000.0])
        staticResult := 0
        instanceResult := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Logaddexp2(left, right)
            instanceResult := left.Logaddexp2(right)
            expected := [1000.0, 4.011227255423254, 1.0,
                1.3219280948873624, 3.0223678130284544, 1000.0]
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear(expected, result.ToArray(), 1e-12)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Logaddexp2(left, right)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated logaddexp2 calls must release results")

            rejected := false
            try Numpy.Logaddexp2(1, right)
            catch TypeError as err
                rejected := InStr(err.Message, "left") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Logaddexp2 must reject a non-array left input")

            rejected := false
            try Numpy.Logaddexp2(left, 1)
            catch TypeError as err
                rejected := InStr(err.Message, "right") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Logaddexp2 must reject a non-array right input")

            timedeltaSource := Numpy.Zeros([2], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Logaddexp2(timedeltaSource, right)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Logaddexp2") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Logaddexp2 must preserve the native dtype TypeError")

            left := 0
            right := 0
            NumpyAssertArrayNear(expected, staticResult.ToArray(), 1e-12)
        } finally {
            err := 0
            timedeltaSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            right := 0
            left := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "logaddexp2 facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestComparisonFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        left := Numpy.IntArray([
            9007199254740992, 9007199254740993,
            -9007199254740993, 9223372036854775807])
        right := Numpy.IntArray([
            9007199254740993, 9007199254740993,
            -9007199254740992, 9223372036854775806])
        results := []
        temporary := 0
        datetimeSource := 0
        incompatibleLeft := 0
        incompatibleRight := 0

        try {
            results.Push([
                Numpy.Equal(left, right), left.Equal(right),
                [0, 1, 0, 0]])
            results.Push([
                Numpy.NotEqual(left, right), left.NotEqual(right),
                [1, 0, 1, 1]])
            results.Push([
                Numpy.Less(left, right), left.Less(right),
                [1, 0, 1, 0]])
            results.Push([
                Numpy.LessEqual(left, right), left.LessEqual(right),
                [1, 1, 1, 0]])
            results.Push([
                Numpy.Greater(left, right), left.Greater(right),
                [0, 0, 0, 1]])
            results.Push([
                Numpy.GreaterEqual(left, right), left.GreaterEqual(right),
                [0, 1, 0, 1]])

            for comparisonCase in results {
                for result in [comparisonCase[1], comparisonCase[2]] {
                    NumpyAssertArrayNear(
                        comparisonCase[3], result.ToArray(), 0)
                    AhkTest.AssertEqual(Numpy.DT_BOOL, result.Dtype)
                }
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Less(left, right)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated comparison calls must release results")

            rejected := false
            try Numpy.Equal(1, right)
            catch TypeError as err
                rejected := InStr(err.Message, "left") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Equal must reject a non-array left input")

            rejected := false
            try Numpy.Equal(left, 1)
            catch TypeError as err
                rejected := InStr(err.Message, "right") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Equal must reject a non-array right input")

            datetimeSource := Numpy.Zeros([4], Numpy.DT_DATETIME)
            rejected := false
            try Numpy.Greater(datetimeSource, right)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Greater") > 0
                    && InStr(err.Message, "do not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Greater must preserve the native dtype TypeError")

            incompatibleLeft := Numpy.Zeros([2, 2])
            incompatibleRight := Numpy.Zeros([3])
            rejected := false
            try Numpy.LessEqual(incompatibleLeft, incompatibleRight)
            catch ValueError as err
                rejected := InStr(err.Message, "Numpy.LessEqual") > 0
                    && InStr(err.Message, "broadcast") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.LessEqual must preserve the native broadcast error")

            left := 0
            right := 0
            NumpyAssertArrayNear(
                [0, 1, 0, 0], results[1][1].ToArray(), 0)
        } finally {
            err := 0
            incompatibleRight := 0
            incompatibleLeft := 0
            datetimeSource := 0
            temporary := 0
            results := 0
            right := 0
            left := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "comparison facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestExtremaFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        left := Numpy.IntArray([
            9007199254740992, 9007199254740993,
            -9007199254740993, 9223372036854775807])
        right := Numpy.IntArray([
            9007199254740993, 9007199254740992,
            -9007199254740992, 9223372036854775806])
        integerResults := []
        nanLeft := 0
        nanRight := 0
        nanResults := []
        temporary := 0
        timedeltaSource := 0
        incompatibleLeft := 0
        incompatibleRight := 0

        try {
            integerResults.Push([
                Numpy.Maximum(left, right), left.Maximum(right),
                [9007199254740993, 9007199254740993,
                    -9007199254740992, 9223372036854775807]])
            integerResults.Push([
                Numpy.Minimum(left, right), left.Minimum(right),
                [9007199254740992, 9007199254740992,
                    -9007199254740993, 9223372036854775806]])
            for extremaCase in integerResults {
                for result in [extremaCase[1], extremaCase[2]] {
                    NumpyAssertArrayNear(extremaCase[3], result.ToArray(), 0)
                    AhkTest.AssertEqual(Numpy.DT_LONGLONG, result.Dtype)
                }
            }

            quietNan := NumpyDoubleFromBits(0x7ff80000)
            nanLeft := Numpy.Array([quietNan, 2.0, quietNan, -3.0])
            nanRight := Numpy.Array([1.0, quietNan, quietNan, -4.0])
            nanResults.Push([
                Numpy.Maximum(nanLeft, nanRight),
                nanLeft.Maximum(nanRight), "maximum"])
            nanResults.Push([
                Numpy.Minimum(nanLeft, nanRight),
                nanLeft.Minimum(nanRight), "minimum"])
            nanResults.Push([
                Numpy.Fmax(nanLeft, nanRight),
                nanLeft.Fmax(nanRight), "fmax"])
            nanResults.Push([
                Numpy.Fmin(nanLeft, nanRight),
                nanLeft.Fmin(nanRight), "fmin"])
            for extremaCase in nanResults {
                for result in [extremaCase[1], extremaCase[2]] {
                    values := result.ToArray()
                    AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
                    if extremaCase[3] = "maximum"
                        AhkTest.AssertTrue(values[1] != values[1]
                            && values[2] != values[2]
                            && values[3] != values[3]
                            && values[4] = -3.0)
                    else if extremaCase[3] = "minimum"
                        AhkTest.AssertTrue(values[1] != values[1]
                            && values[2] != values[2]
                            && values[3] != values[3]
                            && values[4] = -4.0)
                    else if extremaCase[3] = "fmax"
                        AhkTest.AssertTrue(values[1] = 1.0
                            && values[2] = 2.0
                            && values[3] != values[3]
                            && values[4] = -3.0)
                    else
                        AhkTest.AssertTrue(values[1] = 1.0
                            && values[2] = 2.0
                            && values[3] != values[3]
                            && values[4] = -4.0)
                }
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Fmin(nanLeft, nanRight)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated extrema calls must release results")

            rejected := false
            try Numpy.Maximum(1, right)
            catch TypeError as err
                rejected := InStr(err.Message, "left") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Maximum must reject a non-array left input")

            rejected := false
            try Numpy.Fmax(left, 1)
            catch TypeError as err
                rejected := InStr(err.Message, "right") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Fmax must reject a non-array right input")

            timedeltaSource := Numpy.Zeros([4], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Fmin(timedeltaSource, nanRight)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Fmin") > 0
                    && InStr(err.Message, "do not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Fmin must preserve the native dtype TypeError")

            incompatibleLeft := Numpy.Zeros([2, 2])
            incompatibleRight := Numpy.Zeros([3])
            rejected := false
            try Numpy.Minimum(incompatibleLeft, incompatibleRight)
            catch ValueError as err
                rejected := InStr(err.Message, "Numpy.Minimum") > 0
                    && InStr(err.Message, "broadcast") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Minimum must preserve the native broadcast error")

            left := 0
            right := 0
            NumpyAssertArrayNear([
                9007199254740993, 9007199254740993,
                -9007199254740992, 9223372036854775807],
                integerResults[1][1].ToArray(), 0)
        } finally {
            err := 0
            incompatibleRight := 0
            incompatibleLeft := 0
            timedeltaSource := 0
            temporary := 0
            nanResults := 0
            nanRight := 0
            nanLeft := 0
            integerResults := 0
            right := 0
            left := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "extrema facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestLogicalFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        left := Numpy.IntArray([0, 1, -2, 0])
        right := Numpy.IntArray([1, 0, 3, 0])
        results := []
        temporary := 0

        try {
            results.Push([
                Numpy.LogicalAnd(left, right), left.LogicalAnd(right),
                [0, 0, 1, 0]])
            results.Push([
                Numpy.LogicalOr(left, right), left.LogicalOr(right),
                [1, 1, 1, 0]])
            results.Push([
                Numpy.LogicalXor(left, right), left.LogicalXor(right),
                [1, 1, 0, 0]])
            results.Push([
                Numpy.LogicalNot(left), left.LogicalNot(),
                [1, 0, 0, 1]])
            for logicalCase in results {
                for result in [logicalCase[1], logicalCase[2]] {
                    AhkTest.AssertEqual(Numpy.DT_BOOL, result.Dtype)
                    NumpyAssertArrayNear(logicalCase[3], result.ToArray(), 0)
                }
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.LogicalXor(left, right)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated logical calls must release results")

            rejected := false
            try Numpy.LogicalAnd(1, right)
            catch TypeError as err
                rejected := InStr(err.Message, "left") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.LogicalAnd must reject a non-array left input")

            rejected := false
            try Numpy.LogicalNot(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.LogicalNot must reject a non-array source")
        } finally {
            err := 0
            temporary := 0
            results := 0
            right := 0
            left := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "logical facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestBitwiseFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        left := Numpy.IntArray([0, 1, -2, 7])
        right := Numpy.IntArray([1, 2, 3, 65])
        results := []
        temporary := 0

        try {
            results.Push([
                Numpy.BitwiseAnd(left, right), left.BitwiseAnd(right),
                [0, 0, 2, 1]])
            results.Push([
                Numpy.BitwiseOr(left, right), left.BitwiseOr(right),
                [1, 3, -1, 71]])
            results.Push([
                Numpy.BitwiseXor(left, right), left.BitwiseXor(right),
                [1, 3, -3, 70]])
            results.Push([
                Numpy.LeftShift(left, right), left.LeftShift(right),
                [0, 4, -16, 0]])
            results.Push([
                Numpy.RightShift(left, right), left.RightShift(right),
                [0, 0, -1, 0]])
            results.Push([
                Numpy.Invert(left), left.Invert(),
                [-1, -2, 1, -8]])
            results.Push([
                Numpy.BitwiseNot(left), left.BitwiseNot(),
                [-1, -2, 1, -8]])
            for bitwiseCase in results {
                for result in [bitwiseCase[1], bitwiseCase[2]] {
                    AhkTest.AssertEqual(left.Dtype, result.Dtype)
                    NumpyAssertArrayNear(bitwiseCase[3], result.ToArray(), 0)
                }
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.BitwiseXor(left, right)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated bitwise calls must release results")

            rejected := false
            try Numpy.BitwiseAnd(1, right)
            catch TypeError as err
                rejected := InStr(err.Message, "left") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.BitwiseAnd must reject a non-array left input")

            rejected := false
            try Numpy.Invert(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Invert must reject a non-array source")
        } finally {
            err := 0
            temporary := 0
            results := 0
            right := 0
            left := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "bitwise facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestTanFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.IntArray([-3, -1, 0, 1, 3], [5])
        staticResult := 0
        instanceResult := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Tan(source)
            instanceResult := source.Tan()
            expected := [Tan(-3.0), Tan(-1.0), 0.0, Tan(1.0), Tan(3.0)]
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear(expected, result.ToArray(), 1e-12)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Tan(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated tan calls must release results")

            rejected := false
            try Numpy.Tan(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Tan must reject a non-array source")

            timedeltaSource := Numpy.Zeros([2], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Tan(timedeltaSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Tan") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Tan must preserve the native dtype TypeError")

            source := 0
            NumpyAssertArrayNear(expected, staticResult.ToArray(), 1e-12)
        } finally {
            err := 0
            timedeltaSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "tan facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestArcsinFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.IntArray([-1, 0, 1], [3])
        staticResult := 0
        instanceResult := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Arcsin(source)
            instanceResult := source.Arcsin()
            expected := [ASin(-1.0), 0.0, ASin(1.0)]
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear(expected, result.ToArray(), 1e-12)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Arcsin(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated arcsin calls must release results")

            rejected := false
            try Numpy.Arcsin(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Arcsin must reject a non-array source")

            timedeltaSource := Numpy.Zeros([2], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Arcsin(timedeltaSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Arcsin") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Arcsin must preserve the native dtype TypeError")

            source := 0
            NumpyAssertArrayNear(expected, staticResult.ToArray(), 1e-12)
        } finally {
            err := 0
            timedeltaSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "arcsin facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestArccosFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.IntArray([-1, 0, 1], [3])
        staticResult := 0
        instanceResult := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Arccos(source)
            instanceResult := source.Arccos()
            expected := [ACos(-1.0), ACos(0.0), ACos(1.0)]
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear(expected, result.ToArray(), 1e-12)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Arccos(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated arccos calls must release results")

            rejected := false
            try Numpy.Arccos(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Arccos must reject a non-array source")

            timedeltaSource := Numpy.Zeros([2], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Arccos(timedeltaSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Arccos") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Arccos must preserve the native dtype TypeError")

            source := 0
            NumpyAssertArrayNear(expected, staticResult.ToArray(), 1e-12)
        } finally {
            err := 0
            timedeltaSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "arccos facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestArctanFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.IntArray([-1, 0, 1], [3])
        staticResult := 0
        instanceResult := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Arctan(source)
            instanceResult := source.Arctan()
            expected := [ATan(-1.0), 0.0, ATan(1.0)]
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear(expected, result.ToArray(), 1e-12)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Arctan(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated arctan calls must release results")

            rejected := false
            try Numpy.Arctan(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Arctan must reject a non-array source")

            timedeltaSource := Numpy.Zeros([2], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Arctan(timedeltaSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Arctan") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Arctan must preserve the native dtype TypeError")

            source := 0
            NumpyAssertArrayNear(expected, staticResult.ToArray(), 1e-12)
        } finally {
            err := 0
            timedeltaSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "arctan facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestArctan2FacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        y := Numpy.IntArray([-1, 0, 1], [3])
        x := Numpy.IntArray([1, 1, 1], [3])
        staticResult := 0
        instanceResult := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Arctan2(y, x)
            instanceResult := y.Arctan2(x)
            expected := [ATan(-1.0), 0.0, ATan(1.0)]
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear(expected, result.ToArray(), 1e-12)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Arctan2(y, x)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated arctan2 calls must release results")

            rejected := false
            try Numpy.Arctan2(1, x)
            catch TypeError as err
                rejected := InStr(err.Message, "y") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Arctan2 must reject a non-array y input")

            rejected := false
            try Numpy.Arctan2(y, 1)
            catch TypeError as err
                rejected := InStr(err.Message, "x") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Arctan2 must reject a non-array x input")

            timedeltaSource := Numpy.Zeros([3], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Arctan2(timedeltaSource, x)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Arctan2") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Arctan2 must preserve the native dtype TypeError")

            y := 0
            x := 0
            NumpyAssertArrayNear(expected, staticResult.ToArray(), 1e-12)
        } finally {
            err := 0
            timedeltaSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            x := 0
            y := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "arctan2 facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestHypotFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        x := Numpy.IntArray([3, 5, 8], [3])
        y := Numpy.IntArray([4, 12, 15], [3])
        staticResult := 0
        instanceResult := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Hypot(x, y)
            instanceResult := x.Hypot(y)
            expected := [5.0, 13.0, 17.0]
            for result in [staticResult, instanceResult] {
                NumpyAssertArrayNear(expected, result.ToArray(), 1e-12)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Hypot(x, y)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated hypot calls must release results")

            rejected := false
            try Numpy.Hypot(1, y)
            catch TypeError as err
                rejected := InStr(err.Message, "x") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Hypot must reject a non-array x input")

            rejected := false
            try Numpy.Hypot(x, 1)
            catch TypeError as err
                rejected := InStr(err.Message, "y") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Hypot must reject a non-array y input")

            timedeltaSource := Numpy.Zeros([3], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Hypot(timedeltaSource, y)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Hypot") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Hypot must preserve the native dtype TypeError")

            x := 0
            y := 0
            NumpyAssertArrayNear(expected, staticResult.ToArray(), 1e-12)
        } finally {
            err := 0
            timedeltaSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            y := 0
            x := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "hypot facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestAngleConversionFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        pi := ACos(-1.0)
        degreeSource := Numpy.Array([0.0, 90.0, 180.0])
        radianSource := Numpy.Array([0.0, pi / 2.0, pi])
        degreesStatic := 0
        degreesInstance := 0
        rad2degStatic := 0
        rad2degInstance := 0
        radiansStatic := 0
        radiansInstance := 0
        deg2radStatic := 0
        deg2radInstance := 0
        temporary := 0
        timedeltaSource := 0

        try {
            degreesStatic := Numpy.Degrees(radianSource)
            degreesInstance := radianSource.Degrees()
            rad2degStatic := Numpy.Rad2deg(radianSource)
            rad2degInstance := radianSource.Rad2deg()
            for result in [degreesStatic, degreesInstance,
                    rad2degStatic, rad2degInstance] {
                NumpyAssertArrayNear([0.0, 90.0, 180.0],
                    result.ToArray(), 1e-12)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            radiansStatic := Numpy.Radians(degreeSource)
            radiansInstance := degreeSource.Radians()
            deg2radStatic := Numpy.Deg2rad(degreeSource)
            deg2radInstance := degreeSource.Deg2rad()
            for result in [radiansStatic, radiansInstance,
                    deg2radStatic, deg2radInstance] {
                NumpyAssertArrayNear([0.0, pi / 2.0, pi],
                    result.ToArray(), 1e-12)
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Degrees(radianSource)
                temporary := 0
                temporary := Numpy.Radians(degreeSource)
                temporary := 0
                temporary := Numpy.Rad2deg(radianSource)
                temporary := 0
                temporary := Numpy.Deg2rad(degreeSource)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated angle conversions must release results")

            rejected := false
            try Numpy.Degrees(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Degrees must reject a non-array source")

            timedeltaSource := Numpy.Zeros([3], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Deg2rad(timedeltaSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Deg2rad") > 0
                    && InStr(err.Message, "does not support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Deg2rad must preserve the native dtype TypeError")

            degreeSource := 0
            radianSource := 0
            NumpyAssertArrayNear([0.0, 90.0, 180.0],
                degreesStatic.ToArray(), 1e-12)
        } finally {
            err := 0
            timedeltaSource := 0
            temporary := 0
            deg2radInstance := 0
            deg2radStatic := 0
            radiansInstance := 0
            radiansStatic := 0
            rad2degInstance := 0
            rad2degStatic := 0
            degreesInstance := 0
            degreesStatic := 0
            radianSource := 0
            degreeSource := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "angle conversion facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestPowerFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        bases := Numpy.IntArray([2, -3, 4, 4], [4])
        exponents := Numpy.IntArray([3, 2, 0, 1], [4])
        negativeExponents := Numpy.IntArray([-1, -2, -1, 0], [4])
        staticResult := 0
        instanceResult := 0
        floatStatic := 0
        floatInstance := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Power(bases, exponents)
            instanceResult := bases.Power(exponents)
            for result in [staticResult, instanceResult] {
                AhkTest.AssertEqual(Numpy.DT_LONGLONG, result.Dtype)
                NumpyAssertArrayNear([8, 9, 1, 4], result.ToArray(), 0)
            }

            floatStatic := Numpy.FloatPower(bases, negativeExponents)
            floatInstance := bases.FloatPower(negativeExponents)
            for result in [floatStatic, floatInstance] {
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
                NumpyAssertArrayNear([0.5, 1.0 / 9.0, 0.25, 1.0],
                    result.ToArray(), 1e-12)
            }

            rejected := false
            try Numpy.Power(bases, negativeExponents)
            catch Error as err
                rejected := InStr(err.Message,
                    "Integers to negative integer powers are not allowed") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Power must expose negative integer exponent errors")

            rejected := false
            try Numpy.Power(1, exponents)
            catch TypeError as err
                rejected := InStr(err.Message, "left") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Power must reject a non-array left input")

            rejected := false
            try Numpy.FloatPower(bases, 1)
            catch TypeError as err
                rejected := InStr(err.Message, "right") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.FloatPower must reject a non-array right input")

            timedeltaSource := Numpy.Zeros([4], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.FloatPower(timedeltaSource, exponents)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.FloatPower") > 0
                    && InStr(err.Message, "support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.FloatPower must preserve native dtype errors")

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Power(bases, exponents)
                temporary := 0
                temporary := Numpy.FloatPower(bases, negativeExponents)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated power calls must release results")

            bases := 0
            exponents := 0
            negativeExponents := 0
            NumpyAssertArrayNear([8, 9, 1, 4], staticResult.ToArray(), 0)
        } finally {
            err := 0
            timedeltaSource := 0
            temporary := 0
            floatInstance := 0
            floatStatic := 0
            instanceResult := 0
            staticResult := 0
            negativeExponents := 0
            exponents := 0
            bases := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "power facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestHeavisideFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        left := Numpy.Array([-2.0, 0.0, 3.0, 0.0], [2, 2])
        right := Numpy.Array([0.25, -0.0], [2, 1])
        staticResult := 0
        instanceResult := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResult := Numpy.Heaviside(left, right)
            instanceResult := left.Heaviside(right)
            for result in [staticResult, instanceResult] {
                AhkTest.AssertEqual(Numpy.DT_FLOAT64, result.Dtype)
                NumpyAssertArrayNear([2, 2], result.Shape, 0)
                NumpyAssertArrayNear([0.0, 0.25, 1.0, -0.0],
                    result.ToArray(), 0)
            }

            rejected := false
            try Numpy.Heaviside(1, right)
            catch TypeError as err
                rejected := InStr(err.Message, "left") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Heaviside must reject a non-array left input")

            rejected := false
            try Numpy.Heaviside(left, 1)
            catch TypeError as err
                rejected := InStr(err.Message, "right") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Heaviside must reject a non-array right input")

            timedeltaSource := Numpy.Zeros([4], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Heaviside(timedeltaSource, right)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Heaviside") > 0
                    && InStr(err.Message, "support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Heaviside must preserve native dtype errors")

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Heaviside(left, right)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated heaviside calls must release results")

            left := 0
            right := 0
            NumpyAssertArrayNear([0.0, 0.25, 1.0, -0.0],
                staticResult.ToArray(), 0)
        } finally {
            err := 0
            timedeltaSource := 0
            temporary := 0
            instanceResult := 0
            staticResult := 0
            right := 0
            left := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "heaviside facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestGcdLcmFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        left := Numpy.IntArray([12, -18, 0, 7], [4])
        right := Numpy.IntArray([8, 24, 5, -3], [4])
        gcdStatic := 0
        gcdInstance := 0
        lcmStatic := 0
        lcmInstance := 0
        temporary := 0
        timedeltaSource := 0

        try {
            gcdStatic := Numpy.Gcd(left, right)
            gcdInstance := left.Gcd(right)
            lcmStatic := Numpy.Lcm(left, right)
            lcmInstance := left.Lcm(right)
            for result in [gcdStatic, gcdInstance] {
                AhkTest.AssertEqual(Numpy.DT_LONGLONG, result.Dtype)
                NumpyAssertArrayNear([4, 6, 5, 1], result.ToArray(), 0)
            }
            for result in [lcmStatic, lcmInstance] {
                AhkTest.AssertEqual(Numpy.DT_LONGLONG, result.Dtype)
                NumpyAssertArrayNear([24, 72, 0, 21], result.ToArray(), 0)
            }

            rejected := false
            try Numpy.Gcd(1, right)
            catch TypeError as err
                rejected := InStr(err.Message, "left") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Gcd must reject a non-array left input")

            rejected := false
            try Numpy.Lcm(left, 1)
            catch TypeError as err
                rejected := InStr(err.Message, "right") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Lcm must reject a non-array right input")

            timedeltaSource := Numpy.Zeros([4], Numpy.DT_TIMEDELTA)
            for method in ["Gcd", "Lcm"] {
                rejected := false
                try Numpy.%method%(timedeltaSource, right)
                catch TypeError as err
                    rejected := InStr(err.Message, "Numpy." method) > 0
                        && InStr(err.Message, "support") > 0
                AhkTest.AssertTrue(rejected,
                    "Numpy." method " must preserve native dtype errors")
            }

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Gcd(left, right)
                temporary := 0
                temporary := Numpy.Lcm(left, right)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated gcd/lcm calls must release results")

            left := 0
            right := 0
            NumpyAssertArrayNear([4, 6, 5, 1], gcdStatic.ToArray(), 0)
            NumpyAssertArrayNear([24, 72, 0, 21], lcmStatic.ToArray(), 0)
        } finally {
            err := 0
            timedeltaSource := 0
            temporary := 0
            lcmInstance := 0
            lcmStatic := 0
            gcdInstance := 0
            gcdStatic := 0
            right := 0
            left := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "gcd/lcm facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestDivmodFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        left := Numpy.IntArray([-7, 7], [2, 1])
        right := Numpy.IntArray([3, -3, 2], [1, 3])
        staticResults := 0
        instanceResults := 0
        temporary := 0
        timedeltaSource := 0

        try {
            staticResults := Numpy.Divmod(left, right)
            instanceResults := left.Divmod(right)
            for results in [staticResults, instanceResults] {
                AhkTest.AssertEqual(2, results.Length,
                    "divmod must return quotient and remainder")
                quotient := results[1]
                remainder := results[2]
                AhkTest.AssertEqual(Numpy.DT_LONGLONG, quotient.Dtype)
                AhkTest.AssertEqual(Numpy.DT_LONGLONG, remainder.Dtype)
                NumpyAssertArrayNear([2, 3], quotient.Shape, 0)
                NumpyAssertArrayNear([2, 3], remainder.Shape, 0)
                NumpyAssertArrayNear([-3, 2, -4, 2, -3, 3],
                    quotient.ToArray(), 0)
                NumpyAssertArrayNear([2, -1, 1, 1, -2, 1],
                    remainder.ToArray(), 0)
            }

            rejected := false
            try Numpy.Divmod(1, right)
            catch TypeError as err
                rejected := InStr(err.Message, "left") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Divmod must reject a non-array left input")

            rejected := false
            try left.Divmod(1)
            catch TypeError as err
                rejected := InStr(err.Message, "right") > 0
            AhkTest.AssertTrue(rejected,
                "NdArray.Divmod must reject a non-array right input")

            timedeltaSource := Numpy.Zeros([3], Numpy.DT_TIMEDELTA)
            rejected := false
            try Numpy.Divmod(timedeltaSource, right)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Divmod") > 0
                    && InStr(err.Message, "support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Divmod must preserve the native dtype TypeError")

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Divmod(left, right)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated divmod calls must release both results")

            left := 0
            right := 0
            NumpyAssertArrayNear([-3, 2, -4, 2, -3, 3],
                staticResults[1].ToArray(), 0)
            NumpyAssertArrayNear([2, -1, 1, 1, -2, 1],
                staticResults[2].ToArray(), 0)
        } finally {
            err := 0
            remainder := 0
            quotient := 0
            timedeltaSource := 0
            temporary := 0
            instanceResults := 0
            staticResults := 0
            right := 0
            left := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "divmod facade must release sources and both results")
        Numpy.Cleanup()
    }

    static TestArrayPredicateFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        positiveInfinity := NumpyDoubleFromBits(0x7ff00000)
        negativeInfinity := NumpyDoubleFromBits(0xfff00000)
        negativeZero := NumpyDoubleFromBits(0x80000000)
        quietNan := NumpyDoubleFromBits(0x7ff80000)
        source := Numpy.Array([
            negativeInfinity, negativeZero, 0.0, positiveInfinity, quietNan
        ])
        results := []
        temporary := 0
        complexSource := 0

        try {
            results.Push(Numpy.IsNan(source))
            results.Push(source.IsNan())
            results.Push(Numpy.IsInf(source))
            results.Push(source.IsInf())
            results.Push(Numpy.IsFinite(source))
            results.Push(source.IsFinite())
            results.Push(Numpy.Signbit(source))
            results.Push(source.Signbit())

            for index, result in results
                AhkTest.AssertEqual(Numpy.DT_BOOL, result.Dtype,
                    "array predicate result " index " must be bool")
            for index in [1, 2]
                NumpyAssertArrayNear([0, 0, 0, 0, 1],
                    results[index].ToArray(), 0)
            for index in [3, 4]
                NumpyAssertArrayNear([1, 0, 0, 1, 0],
                    results[index].ToArray(), 0)
            for index in [5, 6]
                NumpyAssertArrayNear([0, 1, 1, 0, 0],
                    results[index].ToArray(), 0)
            for index in [7, 8]
                NumpyAssertArrayNear([1, 1, 0, 0, 0],
                    results[index].ToArray(), 0)

            lowercase := Numpy.isnan(source)
            NumpyAssertArrayNear([0, 0, 0, 0, 1],
                lowercase.ToArray(), 0)
            lowercase := 0

            rejected := false
            try Numpy.IsFinite(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.IsFinite must reject a non-array source")

            complexSource := Numpy.Zeros([2], Numpy.DT_COMPLEX128)
            rejected := false
            try Numpy.Signbit(complexSource)
            catch TypeError as err
                rejected := InStr(err.Message, "Numpy.Signbit") > 0
                    && InStr(err.Message, "support") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Signbit must preserve native complex dtype errors")

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.IsNan(source)
                temporary := 0
                temporary := Numpy.IsInf(source)
                temporary := 0
                temporary := Numpy.IsFinite(source)
                temporary := 0
                temporary := Numpy.Signbit(source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated array predicate calls must release results")

            source := 0
            NumpyAssertArrayNear([0, 0, 0, 0, 1],
                results[1].ToArray(), 0)
        } finally {
            err := 0
            lowercase := 0
            complexSource := 0
            temporary := 0
            results := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "array predicate facade must release sources and results")
        Numpy.Cleanup()
    }

    static TestObjectKindPredicateFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        realSource := Numpy.Zeros([2, 3], Numpy.DT_FLOAT64)
        complexSource := Numpy.Zeros([2, 3], Numpy.DT_COMPLEX64)
        longComplexSource := Numpy.Zeros(
            [2], Numpy.DT_COMPLEX_LONGDOUBLE)
        emptyComplexSource := Numpy.Zeros(
            [0], Numpy.DT_COMPLEX128)
        scalarHandle := DllCall(Numpy.Proc("cnp_array_new"),
            "Int", 0, "Ptr", 0, "Int", Numpy.DT_COMPLEX128,
            "Int", 0, "Ptr")
        scalarComplexSource := Numpy.WrapHandle(
            scalarHandle, "object kind scalar fixture")

        try {
            AhkTest.AssertFalse(Numpy.IsComplexObj(realSource),
                "real arrays must not be complex objects")
            AhkTest.AssertTrue(Numpy.IsRealObj(realSource),
                "real arrays must be real objects")
            AhkTest.AssertTrue(Numpy.IsComplexObj(complexSource),
                "complex64 arrays must be complex objects")
            AhkTest.AssertFalse(Numpy.IsRealObj(complexSource),
                "complex64 arrays must not be real objects")
            AhkTest.AssertTrue(longComplexSource.IsComplexObj(),
                "complex long double arrays must be complex objects")
            AhkTest.AssertFalse(longComplexSource.IsRealObj(),
                "complex long double arrays must not be real objects")
            AhkTest.AssertTrue(Numpy.iscomplexobj(emptyComplexSource),
                "empty complex arrays must retain their dtype kind")
            AhkTest.AssertFalse(Numpy.isrealobj(scalarComplexSource),
                "zero-dimensional complex arrays must retain their dtype kind")

            rejected := false
            try Numpy.IsComplexObj(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.IsComplexObj must reject a non-array source")

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                AhkTest.AssertTrue(Numpy.IsComplexObj(complexSource),
                    "repeated complex object query must remain true")
                AhkTest.AssertFalse(Numpy.IsRealObj(complexSource),
                    "repeated real object query must remain false")
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "object kind predicates must not retain memory")
        } finally {
            err := 0
            scalarComplexSource := 0
            emptyComplexSource := 0
            longComplexSource := 0
            complexSource := 0
            realSource := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "object kind predicate facade must release all sources")
        Numpy.Cleanup()
    }

    static TestIsScalarFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        scalarHandle := DllCall(Numpy.Proc("cnp_array_new"),
            "Int", 0, "Ptr", 0, "Int", Numpy.DT_FLOAT64,
            "Int", 0, "Ptr")
        scalarArray := Numpy.WrapHandle(
            scalarHandle, "isscalar scalar fixture")
        oneElement := Numpy.Zeros([1], Numpy.DT_FLOAT64)
        emptyArray := Numpy.Zeros([0], Numpy.DT_FLOAT64)

        try {
            AhkTest.AssertTrue(Numpy.IsScalar(42),
                "Integer must be a scalar")
            AhkTest.AssertTrue(Numpy.IsScalar(1.5),
                "Float must be a scalar")
            AhkTest.AssertTrue(Numpy.isscalar("value"),
                "String must be a scalar")
            AhkTest.AssertTrue(Numpy.IsScalar(Buffer(1)),
                "Buffer must match NumPy memoryview scalar behavior")
            AhkTest.AssertFalse(Numpy.IsScalar([1]),
                "AHK Array must not be a scalar")
            AhkTest.AssertFalse(Numpy.IsScalar(Map("value", 1)),
                "AHK Map must not be a scalar")
            AhkTest.AssertFalse(Numpy.IsScalar({value: 1}),
                "AHK object must not be a scalar")
            AhkTest.AssertFalse(Numpy.IsScalar(scalarArray),
                "zero-dimensional ndarray must not be a scalar")
            AhkTest.AssertFalse(Numpy.IsScalar(oneElement),
                "single-element ndarray must not be a scalar")
            AhkTest.AssertFalse(Numpy.IsScalar(emptyArray),
                "empty ndarray must not be a scalar")
            AhkTest.AssertFalse(scalarArray.IsScalar(),
                "NdArray instance IsScalar must be false")

            activeMemory := Numpy.AllocatedMemory()
            loop 64
                AhkTest.AssertFalse(Numpy.IsScalar(oneElement),
                    "repeated ndarray scalar query must remain false")
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "isscalar facade must not retain native memory")
        } finally {
            emptyArray := 0
            oneElement := 0
            scalarArray := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "isscalar facade must release all arrays")
        Numpy.Cleanup()
    }

    static TestArrayMetadataFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        scalarHandle := DllCall(Numpy.Proc("cnp_array_new"),
            "Int", 0, "Ptr", 0, "Int", Numpy.DT_FLOAT64,
            "Int", 0, "Ptr")
        scalar := Numpy.WrapHandle(
            scalarHandle, "array metadata scalar fixture")
        empty := Numpy.Zeros([0], Numpy.DT_FLOAT64)
        cMatrix := Numpy.Zeros([2, 3], Numpy.DT_FLOAT64)

        matrixShape := Buffer(16, 0)
        NumPut("Int64", 2, matrixShape, 0)
        NumPut("Int64", 3, matrixShape, 8)
        fHandle := DllCall(Numpy.Proc("cnp_array_new"),
            "Int", 2, "Ptr", matrixShape.Ptr,
            "Int", Numpy.DT_FLOAT64, "Int", 1, "Ptr")
        fMatrix := Numpy.WrapHandle(
            fHandle, "array metadata Fortran fixture")

        singletonShape := Buffer(16, 0)
        NumPut("Int64", 1, singletonShape, 0)
        NumPut("Int64", 3, singletonShape, 8)
        singletonHandle := DllCall(Numpy.Proc("cnp_array_new"),
            "Int", 2, "Ptr", singletonShape.Ptr,
            "Int", Numpy.DT_FLOAT64, "Int", 1, "Ptr")
        singleton := Numpy.WrapHandle(
            singletonHandle, "array metadata singleton fixture")

        transposeHandle := DllCall(Numpy.Proc("cnp_transpose"),
            "Ptr", cMatrix.Handle, "Ptr", 0, "Ptr")
        transpose := Numpy.WrapHandle(
            transposeHandle, "array metadata transpose fixture")

        try {
            AhkTest.AssertEqual(8, scalar.Nbytes,
                "scalar nbytes must match NumPy")
            AhkTest.AssertTrue(scalar.CContiguous,
                "scalar must be C contiguous")
            AhkTest.AssertTrue(scalar.FContiguous,
                "scalar must be F contiguous")

            AhkTest.AssertEqual(0, empty.nbytes,
                "empty array nbytes must be zero")
            AhkTest.AssertTrue(empty.CContiguous,
                "empty array must be C contiguous")
            AhkTest.AssertTrue(empty.FContiguous,
                "empty array must be F contiguous")

            AhkTest.AssertEqual(48, cMatrix.Nbytes,
                "C matrix nbytes must match NumPy")
            AhkTest.AssertTrue(cMatrix.CContiguous,
                "C matrix must be C contiguous")
            AhkTest.AssertFalse(cMatrix.FContiguous,
                "C matrix must not be F contiguous")

            AhkTest.AssertEqual(48, fMatrix.Nbytes,
                "F matrix nbytes must match NumPy")
            AhkTest.AssertFalse(fMatrix.CContiguous,
                "F matrix must not be C contiguous")
            AhkTest.AssertTrue(fMatrix.FContiguous,
                "F matrix must be F contiguous")

            AhkTest.AssertEqual(24, singleton.Nbytes,
                "singleton-axis nbytes must match NumPy")
            AhkTest.AssertTrue(singleton.CContiguous,
                "singleton-axis matrix must be C contiguous")
            AhkTest.AssertTrue(singleton.FContiguous,
                "singleton-axis matrix must be F contiguous")

            AhkTest.AssertEqual(48, transpose.Nbytes,
                "transpose nbytes must match NumPy")
            AhkTest.AssertFalse(transpose.CContiguous,
                "transpose must not be C contiguous")
            AhkTest.AssertTrue(transpose.FContiguous,
                "transpose must be F contiguous")

            AhkTest.AssertEqual(
                (cMatrix.Flags & Numpy.ARRAY_C_CONTIGUOUS) != 0,
                cMatrix.CContiguous,
                "CContiguous must reflect cached flags")
            AhkTest.AssertEqual(
                (cMatrix.Flags & Numpy.ARRAY_F_CONTIGUOUS) != 0,
                cMatrix.FContiguous,
                "FContiguous must reflect cached flags")

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                checksum := cMatrix.Nbytes
                    + (cMatrix.CContiguous ? 1 : 0)
                    + (cMatrix.FContiguous ? 1 : 0)
                AhkTest.AssertEqual(49, checksum,
                    "repeated metadata query result")
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "cached metadata properties must not retain native memory")

            invalid := Numpy.NdArray()
            caught := false
            try ignored := invalid.Nbytes
            catch Error as err {
                caught := true
                AhkTest.AssertTrue(
                    InStr(err.Message, "array handle", true) != 0,
                    "invalid metadata handle error text")
            }
            AhkTest.AssertTrue(caught,
                "invalid metadata handle must surface an exception")
        } finally {
            transpose := 0
            singleton := 0
            fMatrix := 0
            cMatrix := 0
            empty := 0
            scalar := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "array metadata facade must release all arrays")
        Numpy.Cleanup()
    }

    static TestAtleastNdFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        scalarHandle := DllCall(Numpy.Proc("cnp_array_from_scalar"),
            "Double", 7.5, "Int", Numpy.DT_FLOAT64, "Ptr")
        scalar := Numpy.WrapHandle(
            scalarHandle, "atleast scalar fixture")
        vector := Numpy.Array([1.0, 2.0, 3.0, 4.0])
        matrix := Numpy.Array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
        scalar1d := 0
        scalar2d := 0
        scalar3d := 0
        vector2d := 0
        vector3d := 0
        matrix3d := 0
        sameRank := 0
        lifetimeSource := 0
        lifetimeView := 0
        temporary := 0

        try {
            scalar1d := Numpy.Atleast_1d(scalar)
            scalar2d := Numpy.Atleast2d(scalar)
            scalar3d := scalar.Atleast3d()
            NumpyAssertArrayNear([1], scalar1d.Shape, 0)
            NumpyAssertArrayNear([1, 1], scalar2d.Shape, 0)
            NumpyAssertArrayNear([1, 1, 1], scalar3d.Shape, 0)
            NumpyAssertArrayNear([7.5], scalar1d.ToArray(), 0)
            AhkTest.AssertEqual(1, DllCall(
                Numpy.Proc("cnp_ahk_shares_data"),
                "Ptr", scalar.Handle, "Ptr", scalar3d.Handle, "Int"),
                "scalar atleast result must share source storage")

            vector2d := Numpy.Atleast_2d(vector)
            vector3d := vector.Atleast3d()
            matrix3d := Numpy.Atleast_3d(matrix)
            NumpyAssertArrayNear([1, 4], vector2d.Shape, 0)
            NumpyAssertArrayNear([1, 4, 1], vector3d.Shape, 0)
            NumpyAssertArrayNear([2, 3, 1], matrix3d.Shape, 0)
            NumpyAssertArrayNear([1.0, 2.0, 3.0, 4.0],
                vector3d.ToArray(), 0)
            AhkTest.AssertEqual(0,
                vector2d.Flags & (0x0004 | 0x0800),
                "expanded atleast result must not own storage")

            sameRank := vector.Atleast1d()
            AhkTest.AssertEqual(vector.Handle, sameRank.Handle,
                "unchanged-rank result must retain the same native array")
            sameRank := 0
            NumpyAssertArrayNear([1.0, 2.0, 3.0, 4.0],
                vector.ToArray(), 0)

            lifetimeSource := Numpy.Array([8.0, 9.0, 10.0])
            lifetimeView := Numpy.Atleast_3d(lifetimeSource)
            lifetimeSource := 0
            NumpyAssertArrayNear([1, 3, 1], lifetimeView.Shape, 0)
            NumpyAssertArrayNear([8.0, 9.0, 10.0],
                lifetimeView.ToArray(), 0)

            rejected := false
            try Numpy.Atleast1d(1)
            catch TypeError as err
                rejected := InStr(err.Message, "source") > 0
            AhkTest.AssertTrue(rejected,
                "Numpy.Atleast1d must reject a non-array source")

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Atleast_2d(vector)
                temporary := 0
                temporary := vector.Atleast3d()
                temporary := 0
                temporary := vector.Atleast1d()
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated atleast facade calls must release references")
        } finally {
            err := 0
            temporary := 0
            lifetimeView := 0
            lifetimeSource := 0
            sameRank := 0
            matrix3d := 0
            vector3d := 0
            vector2d := 0
            scalar3d := 0
            scalar2d := 0
            scalar1d := 0
            matrix := 0
            vector := 0
            scalar := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "atleast facade must release all arrays and views")
        Numpy.Cleanup()
    }

    static TestAverageFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.Array(
            [1.0, 4.0, 9.0, 2.0, 8.0, 18.0], [2, 3])
        axisWeights := Numpy.Array([1.0, 2.0, 1.0])
        fullWeights := Numpy.Array(
            [1.0, 2.0, 1.0, 3.0, 1.0, 2.0], [2, 3])
        zeroWeights := Numpy.Zeros([3])
        axisResult := 0
        fullResult := 0
        lastAxisResult := 0
        temporary := 0

        try {
            AhkTest.AssertTrue(
                Abs(Numpy.Average(source) - 7.0) <= 1e-12,
                "Numpy.Average omitted axis value")
            AhkTest.AssertTrue(
                Abs(source.Average(, fullWeights) - 6.8) <= 1e-12,
                "NdArray.Average omitted axis with weights")

            axisResult := Numpy.Average(source, 1, axisWeights)
            AhkTest.AssertEqual("Numpy.NdArray", Type(axisResult),
                "explicit average axis must return an NdArray")
            NumpyAssertArrayNear([2], axisResult.Shape, 0)
            NumpyAssertArrayNear([4.5, 9.0], axisResult.ToArray(), 1e-12)

            fullResult := source.Average(1, fullWeights)
            NumpyAssertArrayNear(
                [4.5, 50.0 / 6.0], fullResult.ToArray(), 1e-12)

            lastAxisResult := source.Average(-1)
            AhkTest.AssertEqual("Numpy.NdArray", Type(lastAxisResult),
                "explicit -1 average must remain distinct from omitted axis")
            NumpyAssertArrayNear(
                [14.0 / 3.0, 28.0 / 3.0],
                lastAxisResult.ToArray(), 1e-12)

            rejectedWeights := false
            try Numpy.Average(source, 1, 1)
            catch TypeError as err
                rejectedWeights := InStr(err.Message, "weights") > 0
            AhkTest.AssertTrue(rejectedWeights,
                "Numpy.Average must reject non-array weights")

            activeMemory := Numpy.AllocatedMemory()
            failed := false
            try source.Average(1, zeroWeights)
            catch Error as err {
                failed := InStr(err.Message, "weights sum to zero") > 0
                    && InStr(err.Message, "cnp_ahk_average_v2") > 0
            }
            AhkTest.AssertTrue(failed,
                "zero-weight average must expose the native bridge error")
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "failed average must not retain native memory")

            loop 64 {
                temporary := source.Average(1, axisWeights)
                temporary := 0
                ignored := source.Average(, fullWeights)
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated average facade calls must release results")
        } finally {
            err := 0
            temporary := 0
            lastAxisResult := 0
            fullResult := 0
            axisResult := 0
            zeroWeights := 0
            fullWeights := 0
            axisWeights := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "average facade must release all arrays")
        Numpy.Cleanup()
    }

    static TestArraySurfaceFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        empty := 0
        zeros := 0
        ones := 0
        full := 0
        arange := 0
        linspace := 0
        logspace := 0
        geomspace := 0
        eye := 0
        identity := 0
        tri := 0
        vector := 0
        diagonal := 0
        source := 0
        copied := 0
        viewed := 0
        converted := 0
        zerosLike := 0
        onesLike := 0
        emptyLike := 0
        fullLike := 0
        temporary := 0

        try {
            empty := Numpy.Empty([2, 3], Numpy.DT_FLOAT32)
            zeros := Numpy.Zeros([2, 2], Numpy.DT_INT32)
            ones := Numpy.Ones([2, 2], Numpy.DT_INT16)
            full := Numpy.Full([3], -4.0, Numpy.DT_INT64)
            AhkTest.AssertEqual(Numpy.DT_FLOAT32, empty.Dtype)
            NumpyAssertArrayNear([2, 3], empty.Shape, 0)
            NumpyAssertArrayNear([0, 0, 0, 0], zeros.ToArray(), 0)
            NumpyAssertArrayNear([1, 1, 1, 1], ones.ToArray(), 0)
            NumpyAssertArrayNear([-4, -4, -4], full.ToArray(), 0)

            arange := Numpy.Arange(5.0, -3.0, -2.0, Numpy.DT_INT64)
            linspace := Numpy.Linspace(
                0.0, 1.0, 4, false, Numpy.DT_FLOAT32)
            logspace := Numpy.Logspace(
                -1.0, 1.0, 3, true, 10.0, Numpy.DT_FLOAT64)
            geomspace := Numpy.Geomspace(
                -1.0, -16.0, 3, true, Numpy.DT_FLOAT64)
            NumpyAssertArrayNear([5, 3, 1, -1], arange.ToArray(), 0)
            NumpyAssertArrayNear([0.0, 0.25, 0.5, 0.75],
                linspace.ToArray(), 1e-7)
            NumpyAssertArrayNear([0.1, 1.0, 10.0],
                logspace.ToArray(), 1e-12)
            NumpyAssertArrayNear([-1.0, -4.0, -16.0],
                geomspace.ToArray(), 1e-12)

            eye := Numpy.Eye(2, 3, 1, Numpy.DT_INT32)
            identity := Numpy.Identity(2, Numpy.DT_INT16)
            tri := Numpy.Tri(2, 3, 0, Numpy.DT_UINT8)
            NumpyAssertArrayNear([0, 1, 0, 0, 0, 1], eye.ToArray(), 0)
            NumpyAssertArrayNear([1, 0, 0, 1], identity.ToArray(), 0)
            NumpyAssertArrayNear([1, 0, 0, 1, 1, 0], tri.ToArray(), 0)

            vector := Numpy.IntArray([1, 2, 3])
            diagonal := Numpy.Diag(vector, -1)
            AhkTest.AssertEqual(vector.Dtype, diagonal.Dtype)
            NumpyAssertArrayNear([4, 4], diagonal.Shape, 0)
            NumpyAssertArrayNear([
                0, 0, 0, 0,
                1, 0, 0, 0,
                0, 2, 0, 0,
                0, 0, 3, 0
            ], diagonal.ToArray(), 0)

            source := Numpy.Array([1.5, 2.5, 3.5, 4.5], [2, 2])
            copied := source.Copy()
            viewed := source.View()
            converted := source.Astype(Numpy.DT_UINT64)
            zerosLike := Numpy.ZerosLike(source)
            onesLike := Numpy.OnesLike(source)
            emptyLike := Numpy.EmptyLike(source)
            fullLike := Numpy.FullLike(source, -2.25)
            NumpyAssertArrayNear([1.5, 2.5, 3.5, 4.5], copied.ToArray())
            NumpyAssertArrayNear([1, 2, 3, 4], converted.ToArray(), 0)
            NumpyAssertArrayNear([0, 0, 0, 0], zerosLike.ToArray(), 0)
            NumpyAssertArrayNear([1, 1, 1, 1], onesLike.ToArray(), 0)
            NumpyAssertArrayNear([-2.25, -2.25, -2.25, -2.25],
                fullLike.ToArray(), 0)
            AhkTest.AssertEqual(source.Shape.Length, emptyLike.Shape.Length)

            source[2] := 9.25
            AhkTest.AssertEqual(9.25, source[2])
            rejected := false
            try ignored := source.GetItem(99)
            catch ValueError as err
                rejected := InStr(err.Message, "out of bounds") > 0
            AhkTest.AssertTrue(rejected,
                "NdArray.GetItem must expose the native index error")

            source := 0
            NumpyAssertArrayNear([1.5, 9.25, 3.5, 4.5], viewed.ToArray())

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := viewed.Copy()
                temporary := 0
                temporary := viewed.View()
                temporary := 0
                temporary := Numpy.OnesLike(viewed)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated array surface calls must release native results")
        } finally {
            temporary := 0
            fullLike := 0
            emptyLike := 0
            onesLike := 0
            zerosLike := 0
            converted := 0
            viewed := 0
            copied := 0
            source := 0
            diagonal := 0
            vector := 0
            tri := 0
            identity := 0
            eye := 0
            geomspace := 0
            logspace := 0
            linspace := 0
            arange := 0
            full := 0
            ones := 0
            zeros := 0
            empty := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "array surface facade must release every owned result")
        Numpy.Cleanup()
    }

    static TestShapeSurfaceFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.IntArray([0, 1, 2, 3, 4, 5], [2, 3])
        vector := Numpy.IntArray([1, 2, 3])
        results := []
        broadcast := 0
        temporary := 0
        try {
            results.Push(source.Ravel())
            results.Push(Numpy.Swapaxes(source, 0, 1))
            results.Push(Numpy.Moveaxis(source, 0, 1))
            results.Push(Numpy.ExpandDims(source, -1))
            results.Push(Numpy.Tile(source, [2, 1]))
            results.Push(Numpy.Repeat(source, 2, 0))
            results.Push(Numpy.Flip(source, 1))
            results.Push(Numpy.Rot90(source, 1, 0, 1))
            results.Push(Numpy.Roll(source, -1, 1))
            results.Push(Numpy.Pad(source, 1, -7))
            results.Push(Numpy.Slice(source, 0, 3, 2, 1))
            results.Push(Numpy.Stack([source, source], 1))
            results.Push(Numpy.Vstack([source, source]))
            results.Push(Numpy.Hstack([source, source]))
            results.Push(Numpy.Dstack([source, source]))
            results.Push(Numpy.ColumnStack([vector, vector]))
            results.Push(Numpy.Append(source, vector))
            results.Push(Numpy.RowStack([source, source]))

            NumpyAssertArrayNear([0, 1, 2, 3, 4, 5], results[1].ToArray(), 0)
            NumpyAssertArrayNear([0, 3, 1, 4, 2, 5], results[2].ToArray(), 0)
            NumpyAssertArrayNear([0, 3, 1, 4, 2, 5], results[3].ToArray(), 0)
            NumpyAssertArrayNear([0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5],
                results[6].ToArray(), 0)
            NumpyAssertArrayNear([2, 5, 1, 4, 0, 3], results[8].ToArray(), 0)
            NumpyAssertArrayNear([0, 2, 3, 5], results[11].ToArray(), 0)
            NumpyAssertArrayNear([1, 1, 2, 2, 3, 3],
                results[16].ToArray(), 0)
            NumpyAssertArrayNear([0, 1, 2, 3, 4, 5, 1, 2, 3],
                results[17].ToArray(), 0)
            NumpyAssertArrayNear([
                0, 1, 2, 3, 4, 5,
                0, 1, 2, 3, 4, 5
            ], results[18].ToArray(), 0)

            target := [2, 3]
            broadcast := Numpy.BroadcastTo(vector, target)
            vector := 0
            NumpyAssertArrayNear([1, 2, 3, 1, 2, 3],
                broadcast.ToArray(), 0)

            rejected := false
            try Numpy.Moveaxis(source, 4, 0)
            catch ValueError as err
                rejected := InStr(err.Message, "out of bounds") > 0
            AhkTest.AssertTrue(rejected,
                "shape facade must expose invalid axis errors")

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Flip(source, 1)
                temporary := 0
                temporary := Numpy.Repeat(source, 2, 0)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated shape facade calls must release results")
        } finally {
            temporary := 0
            broadcast := 0
            results := 0
            vector := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "shape facade must release all arrays and views")
        Numpy.Cleanup()
    }

    static TestMaskedArrayFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.Array([1.0, 2.0, 3.0, 4.0])
        mask := Numpy.Array([0, 1, 0, 1], [4], Numpy.DT_BOOL)
        masked := 0
        dataView := 0
        maskView := 0
        filled := 0
        compressed := 0
        replacement := 0
        constructorSource := 0
        condition := 0
        constructors := []
        temporary := 0
        try {
            masked := Numpy.Masked(source, mask, -9.0)
            AhkTest.AssertEqual("Numpy.MaskedArray", Type(masked))
            source := 0
            mask := 0

            dataView := masked.Data
            maskView := masked.Mask
            filled := masked.Filled()
            compressed := masked.Compressed()
            NumpyAssertArrayNear([1, 2, 3, 4], dataView.ToArray(), 0)
            NumpyAssertArrayNear([0, 1, 0, 1], maskView.ToArray(), 0)
            NumpyAssertArrayNear([1, -9, 3, -9], filled.ToArray(), 0)
            NumpyAssertArrayNear([1, 3], compressed.ToArray(), 0)
            AhkTest.AssertEqual(2, masked.Count())
            AhkTest.AssertTrue(Abs(masked.Sum() - 4.0) <= 1e-12)
            AhkTest.AssertTrue(Abs(masked.Mean() - 2.0) <= 1e-12)
            AhkTest.AssertTrue(Abs(masked.Std() - 1.0) <= 1e-12)
            AhkTest.AssertTrue(Abs(masked.Min() - 1.0) <= 1e-12)
            AhkTest.AssertTrue(Abs(masked.Max() - 3.0) <= 1e-12)

            replacement := Numpy.Array([1, 0, 1, 0], [4], Numpy.DT_BOOL)
            masked.SetMask(replacement)
            replacement := 0
            compressed := masked.Compressed()
            NumpyAssertArrayNear([2, 4], compressed.ToArray(), 0)
            masked := 0
            NumpyAssertArrayNear([1, 2, 3, 4], dataView.ToArray(), 0)
            NumpyAssertArrayNear([0, 1, 0, 1], maskView.ToArray(), 0)

            constructorSource := Numpy.Array([-2.0, 0.0, 3.0])
            condition := Numpy.Array([0, 1, 0], [3], Numpy.DT_BOOL)
            constructors.Push(Numpy.MaskedWhere(condition, constructorSource))
            constructors.Push(Numpy.MaskedInvalid(constructorSource))
            constructors.Push(Numpy.MaskedGreater(constructorSource, 1.0))
            constructors.Push(Numpy.MaskedLess(constructorSource, -1.0))
            constructors.Push(Numpy.MaskedEqual(constructorSource, 0.0))
            constructors.Push(Numpy.MaskedNotEqual(constructorSource, 0.0))
            constructors.Push(Numpy.MaskedInside(constructorSource, -1.0, 2.0))
            constructors.Push(Numpy.MaskedOutside(constructorSource, -1.0, 2.0))
            expectedMasks := [
                [0, 1, 0], [0, 0, 0], [0, 0, 1], [1, 0, 0],
                [0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1]
            ]
            for index, owner in constructors {
                temporary := owner.Mask
                NumpyAssertArrayNear(
                    expectedMasks[index], temporary.ToArray(), 0)
                temporary := 0
            }

            wrongType := Numpy.IntArray([0, 1, 0])
            activeMemory := Numpy.AllocatedMemory()
            rejected := false
            try Numpy.Masked(constructorSource, wrongType)
            catch TypeError as err
                rejected := InStr(err.Message, "Mask dtype must be bool") > 0
            AhkTest.AssertTrue(rejected,
                "masked facade must expose native mask dtype errors")
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "failed masked creation must not retain native memory")

            wrongShape := Numpy.Array([1, 0], [2], Numpy.DT_BOOL)
            activeMemory := Numpy.AllocatedMemory()
            rejected := false
            try Numpy.MaskedWhere(wrongShape, constructorSource)
            catch ValueError as err
                rejected := InStr(err.Message, "Condition shape") > 0
            AhkTest.AssertTrue(rejected,
                "masked facade must expose native condition shape errors")
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "failed masked_where must not retain native memory")
            wrongShape := 0
            wrongType := 0

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.MaskedGreater(constructorSource, 0.0)
                ignored := temporary.Mask
                temporary := 0
            }
            ignored := 0
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated masked facade calls must release owners and views")
        } finally {
            temporary := 0
            constructors := 0
            condition := 0
            constructorSource := 0
            replacement := 0
            compressed := 0
            filled := 0
            maskView := 0
            dataView := 0
            masked := 0
            mask := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "masked facade must release all native owners and views")
        Numpy.Cleanup()
    }

    static TestMatrixCompatibilityFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := 0
        matrix := 0
        repeated := 0
        eye := 0
        ones := 0
        zeros := 0
        randomUniform := 0
        randomUniformAgain := 0
        randomNormal := 0
        temporary := 0
        try {
            source := Numpy.IntArray([1, 2, 3])
            matrix := Numpy.Mat(source)
            repeated := Numpy.MatlibRepmat(source, 2, 2)
            source := 0

            NumpyAssertArrayNear([1, 3], matrix.Shape, 0)
            NumpyAssertArrayNear([1, 2, 3], matrix.ToArray(), 0)
            NumpyAssertArrayNear([2, 6], repeated.Shape, 0)
            NumpyAssertArrayNear([
                1, 2, 3, 1, 2, 3,
                1, 2, 3, 1, 2, 3
            ], repeated.ToArray(), 0)

            eye := Numpy.MatlibEye(3, 5, -1)
            ones := Numpy.MatlibOnes(2, 3)
            zeros := Numpy.MatlibZeros(2, 3)
            NumpyAssertArrayNear([
                0, 0, 0, 0, 0,
                1, 0, 0, 0, 0,
                0, 1, 0, 0, 0
            ], eye.ToArray(), 0)
            NumpyAssertArrayNear([1, 1, 1, 1, 1, 1], ones.ToArray(), 0)
            NumpyAssertArrayNear([0, 0, 0, 0, 0, 0], zeros.ToArray(), 0)

            Numpy.Random.Seed(20260804)
            randomUniform := Numpy.MatlibRand(4, 5)
            Numpy.Random.Seed(20260804)
            randomUniformAgain := Numpy.MatlibRand(4, 5)
            randomNormal := Numpy.MatlibRandn(3, 7)
            NumpyAssertArrayNear([4, 5], randomUniform.Shape, 0)
            NumpyAssertArrayNear([3, 7], randomNormal.Shape, 0)
            NumpyAssertArrayNear(
                randomUniform.ToArray(), randomUniformAgain.ToArray(), 0)
            for value in randomUniform.ToArray()
                AhkTest.AssertTrue(value >= 0.0 && value < 1.0,
                    "matlib rand facade values must lie in [0, 1)")

            activeMemory := Numpy.AllocatedMemory()
            rejected := false
            try temporary := Numpy.MatlibOnes(-1, 3)
            catch ValueError as err
                rejected := InStr(err.Message, "must not be negative") > 0
            temporary := 0
            AhkTest.AssertTrue(rejected,
                "matrix facade must expose invalid dimension errors")
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "failed matrix construction must not retain native memory")

            rejected := false
            try temporary := Numpy.MatlibRand(-1, 3)
            catch ValueError as err
                rejected := InStr(err.Message, "cnp_matlib_rand") > 0
            temporary := 0
            AhkTest.AssertTrue(rejected,
                "matlib random facade must expose labeled dimension errors")
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "failed matlib random construction must not retain memory")

            loop 64 {
                temporary := Numpy.Mat(matrix)
                temporary := 0
                temporary := Numpy.MatlibRepmat(matrix, 2, 3)
                temporary := 0
                temporary := Numpy.MatlibRand(2, 3)
                temporary := 0
                temporary := Numpy.MatlibRandn(2, 3)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated matrix facade calls must release native results")
        } finally {
            temporary := 0
            randomNormal := 0
            randomUniformAgain := 0
            randomUniform := 0
            zeros := 0
            ones := 0
            eye := 0
            repeated := 0
            matrix := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "matrix facade must release every native result")
        Numpy.Cleanup()
    }

    static TestArrayRelationSurfaceFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := 0
        leftRight := 0
        upDown := 0
        left := 0
        right := 0
        close := 0
        wideA := 0
        wideB := 0
        matrix := 0
        row := 0
        storage := 0
        even := 0
        odd := 0
        independent := 0
        scalar := 0
        temporary := 0
        try {
            source := Numpy.IntArray([0, 1, 2, 3, 4, 5], [2, 3])
            leftRight := Numpy.Fliplr(source)
            upDown := Numpy.Flipud(source)
            source := 0
            NumpyAssertArrayNear([2, 1, 0, 5, 4, 3],
                leftRight.ToArray(), 0)
            NumpyAssertArrayNear([3, 4, 5, 0, 1, 2], upDown.ToArray(), 0)

            left := Numpy.Array([1.0, 0.0], [2, 1])
            right := Numpy.Array([1.0, 1.0000001, 4.0], [3])
            close := Numpy.IsClose(left, right, 1e-5, 1e-8)
            NumpyAssertArrayNear([2, 3], close.Shape, 0)
            NumpyAssertArrayNear([1, 1, 0, 0, 0, 0], close.ToArray(), 0)

            wideA := Numpy.IntArray([2 ** 60 + 1, 2 ** 60 + 3])
            wideB := Numpy.IntArray([2 ** 60 + 1, 2 ** 60 + 5])
            AhkTest.AssertTrue(!Numpy.ArrayEqual(wideA, wideB),
                "array_equal must preserve distinct int64 values")
            matrix := Numpy.IntArray([1, 2, 3, 1, 2, 3], [2, 3])
            row := Numpy.IntArray([1, 2, 3])
            AhkTest.AssertTrue(Numpy.ArrayEquiv(matrix, row),
                "array_equiv must compare broadcast values")

            storage := Numpy.IntArray([0, 1, 2, 3, 4, 5, 6, 7])
            even := Numpy.Slice(storage, 0, 8, 2)
            odd := Numpy.Slice(storage, 1, 8, 2)
            storage := 0
            independent := Numpy.IntArray([0, 2, 4, 6])
            AhkTest.AssertTrue(!Numpy.SharesMemory(even, odd),
                "interleaved views do not share element bytes")
            AhkTest.AssertTrue(Numpy.MayShareMemory(even, odd),
                "interleaved view byte bounds overlap")
            AhkTest.AssertTrue(!Numpy.SharesMemory(even, independent),
                "independent arrays must not share memory")
            bounds := Numpy.ByteBounds(even)
            AhkTest.AssertEqual(56, bounds[2] - bounds[1],
                "byte bounds must be high-exclusive across a strided view")

            scalar := Numpy.IntArray([1], [])
            activeMemory := Numpy.AllocatedMemory()
            rejected := false
            try temporary := Numpy.Flipud(scalar)
            catch ValueError as err
                rejected := InStr(err.Message, "at least one dimension") > 0
            temporary := 0
            AhkTest.AssertTrue(rejected,
                "flipud facade must expose dimensionality errors")
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "failed array relation call must not retain native memory")
            scalar := 0

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.IsClose(left, right)
                temporary := 0
                ignored := Numpy.ArrayEquiv(matrix, row)
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated relation calls must release native results")
        } finally {
            temporary := 0
            scalar := 0
            independent := 0
            odd := 0
            even := 0
            storage := 0
            row := 0
            matrix := 0
            wideB := 0
            wideA := 0
            close := 0
            right := 0
            left := 0
            upDown := 0
            leftRight := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "array relation facade must release every native result")
        Numpy.Cleanup()
    }

    static TestArrayLayoutSurfaceFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        owner := 0
        source := 0
        contiguous := 0
        fortran := 0
        required := 0
        resized := 0
        finite := 0
        invalid := 0
        temporary := 0
        try {
            owner := Numpy.IntArray([1, 2, 3, 4, 5, 6], [2, 3])
            source := Numpy.Transpose(owner)
            contiguous := Numpy.AsContiguousArray(source)
            fortran := Numpy.AsFortranArray(source)
            required := Numpy.Require(source, Numpy.DT_FLOAT64, true)
            resized := Numpy.Resize(source, [2, 5])
            finite := Numpy.AsArrayChkFinite(source, Numpy.DT_FLOAT64)
            source := 0
            owner := 0

            NumpyAssertArrayNear([1, 4, 2, 5, 3, 6],
                contiguous.ToArray(), 0)
            NumpyAssertArrayNear([1, 4, 2, 5, 3, 6], fortran.ToArray(), 0)
            NumpyAssertArrayNear([1, 4, 2, 5, 3, 6], required.ToArray(), 0)
            NumpyAssertArrayNear([1, 4, 2, 5, 3, 6, 1, 4, 2, 5],
                resized.ToArray(), 0)
            NumpyAssertArrayNear([1, 4, 2, 5, 3, 6], finite.ToArray(), 0)
            AhkTest.AssertTrue(contiguous.CContiguous)
            AhkTest.AssertTrue(fortran.FContiguous)
            AhkTest.AssertTrue(required.CContiguous)

            invalid := Numpy.Array([1.0, 1.7976931348623157e308 * 2.0])
            activeMemory := Numpy.AllocatedMemory()
            rejected := false
            try temporary := Numpy.AsArrayChkFinite(invalid)
            catch ValueError as err
                rejected := InStr(err.Message, "infinities or NaNs") > 0
            temporary := 0
            AhkTest.AssertTrue(rejected,
                "asarray_chkfinite facade must expose nonfinite input")
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "failed finite check must not retain native memory")

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Resize(contiguous, [3, 4])
                temporary := 0
                temporary := Numpy.AsFortranArray(contiguous)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated layout calls must release native results")
        } finally {
            temporary := 0
            invalid := 0
            finite := 0
            resized := 0
            required := 0
            fortran := 0
            contiguous := 0
            source := 0
            owner := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "array layout facade must release every native result")
        Numpy.Cleanup()
    }

    static TestIndexSurfaceFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := 0
        mask := 0
        indices := 0
        condition := 0
        alternative := 0
        item := 0
        selected := 0
        fancy := 0
        chosen := 0
        stacked := 0
        locations := 0
        flat := 0
        countLast := 0
        countAxis0Keep := 0
        countAllKeep := 0
        invalidMask := 0
        temporary := 0
        try {
            source := Numpy.IntArray([1, 0, 3, 0, 5, 0], [2, 3])
            mask := Numpy.Array([1, 0, 1, 0, 1, 0], [2, 3], Numpy.DT_BOOL)
            indices := Numpy.IntArray([-1, 0])
            condition := Numpy.Array([1, 0], [2, 1], Numpy.DT_BOOL)
            alternative := Numpy.IntArray([-1, -2, -3], [1, 3])

            item := Numpy.ArrayGetItem(source, [-1, 1])
            selected := Numpy.BooleanIndex(source, mask)
            fancy := Numpy.FancyIndex(source, indices, 1)
            chosen := Numpy.ArrayWhere(condition, source, alternative)
            stacked := Numpy.ArrayNonzero(source)
            locations := Numpy.ArgWhere(source)
            flat := Numpy.FlatNonzero(source)
            count := Numpy.CountNonzero(source)
            countLast := Numpy.CountNonzero(source, -1)
            countAxis0Keep := Numpy.CountNonzero(source, 0, true)
            countAllKeep := Numpy.CountNonzero(source, , true)

            AhkTest.AssertEqual(5, item.ToArray()[1])
            NumpyAssertArrayNear([1, 3, 5], selected.ToArray(), 0)
            NumpyAssertArrayNear([3, 1, 0, 0], fancy.ToArray(), 0)
            NumpyAssertArrayNear([1, 0, 3, -1, -2, -3],
                chosen.ToArray(), 0)
            NumpyAssertArrayNear([0, 0, 1, 0, 2, 1],
                stacked.ToArray(), 0)
            NumpyAssertArrayNear([0, 0, 0, 2, 1, 1],
                locations.ToArray(), 0)
            NumpyAssertArrayNear([0, 2, 4], flat.ToArray(), 0)
            AhkTest.AssertEqual(3, count)
            NumpyAssertArrayNear([2, 1], countLast.ToArray(), 0)
            NumpyAssertArrayNear([1, 1, 1],
                countAxis0Keep.ToArray(), 0)
            NumpyAssertArrayNear([3], countAllKeep.ToArray(), 0)
            NumpyAssertArrayNear([2], countLast.Shape, 0)
            NumpyAssertArrayNear([1, 3], countAxis0Keep.Shape, 0)
            NumpyAssertArrayNear([1, 1], countAllKeep.Shape, 0)

            source := 0
            mask := 0
            indices := 0
            condition := 0
            alternative := 0
            NumpyAssertArrayNear([1, 0, 3, -1, -2, -3],
                chosen.ToArray(), 0)

            invalidMask := Numpy.IntArray([1, 0, 1, 0, 1, 0], [2, 3])
            activeMemory := Numpy.AllocatedMemory()
            rejected := false
            try temporary := Numpy.BooleanIndex(chosen, invalidMask)
            catch TypeError as err
                rejected := InStr(err.Message, "mask dtype must be bool") > 0
            temporary := 0
            AhkTest.AssertTrue(rejected,
                "boolean index facade must expose mask dtype errors")
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "failed index operation must not retain native memory")

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.ArgWhere(chosen)
                temporary := 0
                temporary := Numpy.FlatNonzero(chosen)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated index calls must release native results")
        } finally {
            temporary := 0
            countAllKeep := 0
            countAxis0Keep := 0
            countLast := 0
            invalidMask := 0
            flat := 0
            locations := 0
            stacked := 0
            chosen := 0
            fancy := 0
            selected := 0
            item := 0
            alternative := 0
            condition := 0
            indices := 0
            mask := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "index facade must release every native result")
        Numpy.Cleanup()
    }

    static TestDiscreteUtilitySurfaceFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        values := 0
        weights := 0
        counts := 0
        weighted := 0
        wide := 0
        differences := 0
        flatIndices := 0
        coordinates := 0
        unraveled := 0
        raveled := 0
        upper := 0
        lower := 0
        sincInput := 0
        sincResult := 0
        negative := 0
        temporary := 0
        try {
            values := Numpy.IntArray([0, 2, 1, 2, 2])
            weights := Numpy.Array([0.5, 1.0, 1.5, -0.25, 2.0])
            counts := Numpy.Bincount(values, , 5)
            weighted := Numpy.Bincount(values, weights)
            NumpyAssertArrayNear([1, 1, 3, 0, 0], counts.ToArray(), 0)
            NumpyAssertArrayNear([0.5, 1.5, 2.75],
                weighted.ToArray(), 0)
            AhkTest.AssertEqual(Numpy.DT_LONGLONG, counts.Dtype)
            AhkTest.AssertEqual(Numpy.DT_FLOAT64, weighted.Dtype)

            wide := Numpy.IntArray([
                2 ** 60 + 1, 2 ** 60 + 3, 2 ** 60 + 8])
            differences := Numpy.Ediff1d(wide)
            NumpyAssertArrayNear([2, 5], differences.ToArray(), 0)
            AhkTest.AssertEqual(wide.Dtype, differences.Dtype)

            flatIndices := Numpy.IntArray([0, 5, 23])
            coordinates := Numpy.IntArray([
                0, 0, 1,
                0, 1, 2,
                0, 1, 3
            ], [3, 3])
            unraveled := Numpy.UnravelIndex(flatIndices, [2, 3, 4])
            raveled := Numpy.RavelMultiIndex(coordinates, [2, 3, 4])
            NumpyAssertArrayNear(coordinates.ToArray(),
                unraveled.ToArray(), 0)
            NumpyAssertArrayNear([0, 5, 23], raveled.ToArray(), 0)

            upper := Numpy.TriuIndices(3, 0, 4)
            lower := Numpy.TrilIndices(3, 0, 4)
            NumpyAssertArrayNear([
                0, 0, 0, 0, 1, 1, 1, 2, 2,
                0, 1, 2, 3, 1, 2, 3, 2, 3
            ], upper.ToArray(), 0)
            NumpyAssertArrayNear([
                0, 1, 1, 2, 2, 2,
                0, 0, 1, 0, 1, 2
            ], lower.ToArray(), 0)
            AhkTest.AssertEqual(Numpy.DT_INT32, upper.Dtype)
            AhkTest.AssertEqual(Numpy.DT_INT32, lower.Dtype)

            sincInput := Numpy.Array([-1.0, 0.0, 0.5], ,
                Numpy.DT_FLOAT32)
            sincResult := Numpy.Sinc(sincInput)
            NumpyAssertArrayNear([
                -2.7827534e-8, 1.0, 0.63661975
            ], sincResult.ToArray(), 2e-6)
            AhkTest.AssertEqual(Numpy.DT_FLOAT32, sincResult.Dtype)

            values := 0
            weights := 0
            wide := 0
            flatIndices := 0
            coordinates := 0
            sincInput := 0
            NumpyAssertArrayNear([2, 5], differences.ToArray(), 0)
            NumpyAssertArrayNear([0, 5, 23], raveled.ToArray(), 0)

            negative := Numpy.IntArray([-1, 2])
            activeMemory := Numpy.AllocatedMemory()
            rejected := false
            try temporary := Numpy.Bincount(negative)
            catch ValueError as err
                rejected := InStr(err.Message, "negative") > 0
            temporary := 0
            AhkTest.AssertTrue(rejected,
                "bincount facade must expose native negative-value errors")
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "failed discrete utility call must not retain native memory")

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Sinc(sincResult)
                temporary := 0
                temporary := Numpy.TriuIndices(9, -1, 7)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated discrete utility calls must release results")
        } finally {
            temporary := 0
            negative := 0
            sincResult := 0
            sincInput := 0
            lower := 0
            upper := 0
            raveled := 0
            unraveled := 0
            coordinates := 0
            flatIndices := 0
            differences := 0
            wide := 0
            weighted := 0
            counts := 0
            weights := 0
            values := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "discrete utility facade must release every native result")
        Numpy.Cleanup()
    }

    static TestStructuralUtilitySurfaceFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        cube := 0
        lower := 0
        upper := 0
        wide := 0
        diagonalMatrix := 0
        diagonalSource := 0
        diagonalView := 0
        selectedSource := 0
        condition := 0
        selected := 0
        nonzeroSource := 0
        locations := 0
        trimSource := 0
        trimmed := 0
        fillTarget := 0
        temporary := 0
        try {
            cube := Numpy.IntArray([
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
            ], [2, 3, 4])
            lower := Numpy.Tril(cube, -1)
            upper := Numpy.Triu(cube, 1)
            NumpyAssertArrayNear([
                0, 0, 0, 0, 4, 0, 0, 0, 8, 9, 0, 0,
                0, 0, 0, 0, 16, 0, 0, 0, 20, 21, 0, 0
            ], lower.ToArray(), 0)
            NumpyAssertArrayNear([
                0, 1, 2, 3, 0, 0, 6, 7, 0, 0, 0, 11,
                0, 13, 14, 15, 0, 0, 18, 19, 0, 0, 0, 23
            ], upper.ToArray(), 0)

            wide := Numpy.IntArray([2 ** 60 + 1, 2 ** 60 + 3], [1, 2])
            diagonalMatrix := Numpy.Diagflat(wide, -2)
            AhkTest.AssertEqual(Numpy.DT_LONGLONG, diagonalMatrix.Dtype)
            NumpyAssertArrayNear([
                0, 0, 0, 0,
                0, 0, 0, 0,
                2 ** 60 + 1, 0, 0, 0,
                0, 2 ** 60 + 3, 0, 0
            ], diagonalMatrix.ToArray(), 0)

            diagonalSource := Numpy.IntArray([
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
            ], [2, 3, 4])
            diagonalView := Numpy.Diagonal(diagonalSource, 1, 0, 2)
            NumpyAssertArrayNear([1, 14, 5, 18, 9, 22],
                diagonalView.ToArray(), 0)
            NumpyAssertArrayNear([3, 2], diagonalView.Shape, 0)

            selectedSource := Numpy.IntArray([
                2 ** 60 + 1, 0, 2 ** 60 + 3, 0])
            condition := Numpy.Array([1, 0, 1], , Numpy.DT_BOOL)
            selected := Numpy.Extract(condition, selectedSource)
            NumpyAssertArrayNear([2 ** 60 + 1, 2 ** 60 + 3],
                selected.ToArray(), 0)

            nonzeroSource := Numpy.IntArray([0, 2, 3, 0], [2, 2])
            locations := Numpy.Nonzero(nonzeroSource)
            NumpyAssertArrayNear([0, 1, 1, 0], locations.ToArray(), 0)

            trimSource := Numpy.IntArray([0, 0, 7, 0])
            trimmed := Numpy.TrimZeros(trimSource)
            NumpyAssertArrayNear([7], trimmed.ToArray(), 0)

            fillTarget := Numpy.IntArray([
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
            ], [3, 4])
            Numpy.FillDiagonal(fillTarget, 7.25)
            NumpyAssertArrayNear([
                7, 1, 2, 3, 4, 7, 6, 7, 8, 9, 7, 11
            ], fillTarget.ToArray(), 0)

            cube := 0
            wide := 0
            diagonalSource := 0
            selectedSource := 0
            condition := 0
            nonzeroSource := 0
            trimSource := 0
            NumpyAssertArrayNear([1, 14, 5, 18, 9, 22],
                diagonalView.ToArray(), 0)
            NumpyAssertArrayNear([7], trimmed.ToArray(), 0)

            activeMemory := Numpy.AllocatedMemory()
            rejected := false
            try temporary := Numpy.Diagonal(fillTarget, 0, 1, 1)
            catch ValueError as err
                rejected := InStr(err.Message,
                    "axis1 and axis2 cannot be the same") > 0
            temporary := 0
            AhkTest.AssertTrue(rejected,
                "diagonal facade must expose native axis errors")
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "failed structural utility call must not retain memory")

            loop 64 {
                temporary := Numpy.Triu(fillTarget, -1)
                temporary := 0
                temporary := Numpy.TrimZeros(trimmed)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated structural utility calls must release results")
        } finally {
            temporary := 0
            fillTarget := 0
            trimmed := 0
            trimSource := 0
            locations := 0
            nonzeroSource := 0
            selected := 0
            condition := 0
            selectedSource := 0
            diagonalView := 0
            diagonalSource := 0
            diagonalMatrix := 0
            wide := 0
            upper := 0
            lower := 0
            cube := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "structural utility facade must release every native result")
        Numpy.Cleanup()
    }

    static TestStrideViewSurfaceFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := 0
        strided := 0
        windowSource := 0
        windows := 0
        cube := 0
        rolled := 0
        temporary := 0
        try {
            source := Numpy.IntArray([
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
            ], [3, 4])
            strided := Numpy.AsStrided(source, [2, 2], [32, 16])
            NumpyAssertArrayNear([0, 2, 4, 6], strided.ToArray(), 0)
            NumpyAssertArrayNear([32, 16], strided.Strides, 0)

            windowSource := Numpy.IntArray([0, 1, 2, 3, 4])
            windows := Numpy.SlidingWindowView(windowSource, 3, 0)
            NumpyAssertArrayNear([3, 3], windows.Shape, 0)
            NumpyAssertArrayNear([
                0, 1, 2, 1, 2, 3, 2, 3, 4
            ], windows.ToArray(), 0)

            cube := Numpy.IntArray([
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
            ], [2, 3, 4])
            rolled := Numpy.Rollaxis(cube, 2, 1)
            NumpyAssertArrayNear([2, 4, 3], rolled.Shape, 0)
            NumpyAssertArrayNear([
                0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11,
                12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23
            ], rolled.ToArray(), 0)

            source := 0
            windowSource := 0
            cube := 0
            NumpyAssertArrayNear([0, 2, 4, 6], strided.ToArray(), 0)
            NumpyAssertArrayNear([
                0, 1, 2, 1, 2, 3, 2, 3, 4
            ], windows.ToArray(), 0)

            activeMemory := Numpy.AllocatedMemory()
            rejected := false
            try temporary := Numpy.SlidingWindowView(windows, 4, 1)
            catch ValueError as err
                rejected := InStr(err.Message, "window size exceeds") > 0
            temporary := 0
            AhkTest.AssertTrue(rejected,
                "sliding window facade must expose native size errors")
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "failed stride view call must not retain native memory")

            loop 64 {
                temporary := Numpy.Rollaxis(rolled, 2, 0)
                temporary := 0
                temporary := Numpy.AsStrided(strided, [2, 2], [32, 16])
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated stride view calls must release results")
        } finally {
            temporary := 0
            rolled := 0
            cube := 0
            windows := 0
            windowSource := 0
            strided := 0
            source := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "stride view facade must release every native view")
        Numpy.Cleanup()
    }

    static TestWindowMutationFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        destination := 0
        mask := 0
        values := 0
        putmaskValues := 0
        indices := 0
        putValues := 0
        chooseIndices := 0
        choiceA := 0
        choiceB := 0
        chosen := 0
        floatIndices := 0
        unsafeValues := 0
        invalidChoiceIndices := 0
        temporary := 0
        try {
            destination := Numpy.IntArray([1, 2, 3, 4])
            mask := Numpy.Array([1, 0, 1, 1], [4], Numpy.DT_BOOL)
            values := Numpy.IntArray([9, 8])
            Numpy.Place(destination, mask, values)
            NumpyAssertArrayNear([9, 2, 8, 9], destination.ToArray(), 0)

            indices := Numpy.IntArray([-1, 4])
            putValues := Numpy.IntArray([7, 6])
            Numpy.Put(destination, indices, putValues, "wrap")
            NumpyAssertArrayNear([6, 2, 8, 7], destination.ToArray(), 0)

            putmaskValues := Numpy.IntArray([5, 4])
            Numpy.PutMask(destination, mask, putmaskValues)
            NumpyAssertArrayNear([5, 2, 5, 4], destination.ToArray(), 0)

            chooseIndices := Numpy.IntArray([0, 1], [2, 1])
            choiceA := Numpy.IntArray([10, 11, 12])
            choiceB := Numpy.IntArray([-4])
            chosen := Numpy.Choose(chooseIndices, [choiceA, choiceB])
            NumpyAssertArrayNear([2, 3], chosen.Shape, 0)
            NumpyAssertArrayNear([10, 11, 12, -4, -4, -4],
                chosen.ToArray(), 0)

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Choose(
                    chooseIndices, [choiceA, choiceB])
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated choose calls must release native results")

            chooseIndices := 0
            choiceA := 0
            choiceB := 0
            NumpyAssertArrayNear([10, 11, 12, -4, -4, -4],
                chosen.ToArray(), 0)

            unchanged := destination.ToArray()
            activeMemory := Numpy.AllocatedMemory()
            rejected := false
            try Numpy.Put(destination, indices, putValues, "bad")
            catch ValueError as err
                rejected := InStr(err.Message, "mode must be") > 0
            AhkTest.AssertTrue(rejected,
                "put facade must expose invalid mode errors")
            NumpyAssertArrayNear(unchanged, destination.ToArray(), 0)
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "failed put must not retain native memory")

            floatIndices := Numpy.Array([0.0, 1.0])
            activeMemory := Numpy.AllocatedMemory()
            rejected := false
            try Numpy.Put(destination, floatIndices, putValues, "wrap")
            catch TypeError as err
                rejected := InStr(err.Message, "safely cast") > 0
            AhkTest.AssertTrue(rejected,
                "put facade must expose invalid index dtype errors")
            NumpyAssertArrayNear(unchanged, destination.ToArray(), 0)
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "invalid put dtype must not retain native memory")

            unsafeValues := Numpy.Array([1.5])
            activeMemory := Numpy.AllocatedMemory()
            rejected := false
            try Numpy.PutMask(destination, mask, unsafeValues)
            catch TypeError as err
                rejected := InStr(err.Message, "safely cast") > 0
            AhkTest.AssertTrue(rejected,
                "putmask facade must expose unsafe value dtype errors")
            NumpyAssertArrayNear(unchanged, destination.ToArray(), 0)
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "failed putmask must not retain native memory")

            invalidChoiceIndices := Numpy.IntArray([-1])
            activeMemory := Numpy.AllocatedMemory()
            rejected := false
            try temporary := Numpy.Choose(
                invalidChoiceIndices, [values, putValues])
            catch ValueError as err
                rejected := InStr(err.Message, "outside") > 0
            temporary := 0
            AhkTest.AssertTrue(rejected,
                "choose facade must expose invalid choice indices")
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "failed choose must not retain native memory")
        } finally {
            temporary := 0
            invalidChoiceIndices := 0
            unsafeValues := 0
            floatIndices := 0
            chosen := 0
            choiceB := 0
            choiceA := 0
            chooseIndices := 0
            putValues := 0
            indices := 0
            putmaskValues := 0
            values := 0
            mask := 0
            destination := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "mutation facade must release every native array")
        Numpy.Cleanup()
    }

    static TestWindowFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        empty := 0
        bartlett := 0
        blackman := 0
        hamming := 0
        hanning := 0
        kaiser := 0
        temporary := 0
        try {
            empty := Numpy.Bartlett(-3)
            AhkTest.AssertEqual(0, empty.Size)
            bartlett := Numpy.Bartlett(5)
            blackman := Numpy.Blackman(5)
            hamming := Numpy.Hamming(5)
            hanning := Numpy.Hanning(5)
            kaiser := Numpy.Kaiser(5, 5.0)
            NumpyAssertArrayNear([0, 0.5, 1, 0.5, 0],
                bartlett.ToArray(), 1e-15)
            NumpyAssertArrayNear([0, 0.34, 1, 0.34, 0],
                blackman.ToArray(), 2e-15)
            NumpyAssertArrayNear([0.08, 0.54, 1, 0.54, 0.08],
                hamming.ToArray(), 2e-15)
            NumpyAssertArrayNear([0, 0.5, 1, 0.5, 0],
                hanning.ToArray(), 2e-15)
            NumpyAssertArrayNear([
                0.0367108922712867, 0.5528517696991338, 1,
                0.5528517696991338, 0.0367108922712867
            ], kaiser.ToArray(), 2e-13)

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.Kaiser(33, 8.0)
                temporary := 0
                temporary := Numpy.Hanning(33)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "repeated window facade calls must release results")
        } finally {
            temporary := 0
            kaiser := 0
            hanning := 0
            hamming := 0
            blackman := 0
            bartlett := 0
            empty := 0
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "window facade must release every native result")
        Numpy.Cleanup()
    }

    static TestLegacyFinancialFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        rate := 0.025
        periods := 24
        payment := -125.0
        present := 2500.0
        future := Numpy.Fv(rate, periods, payment, present, 1)
        AhkTest.AssertTrue(
            Abs(Numpy.Pv(rate, periods, payment, future, 1) - present) <= 1e-9)
        AhkTest.AssertTrue(
            Abs(Numpy.Pmt(rate, periods, present, future, 1) - payment) <= 1e-9)
        AhkTest.AssertTrue(
            Abs(Numpy.Nper(rate, payment, present, future, 1) - periods) <= 1e-8)
        AhkTest.AssertTrue(
            Abs(Numpy.Rate(periods, payment, present, future, 1) - rate) <= 1e-9)

        cashflows := [-100.0, 20.0, 30.0, 40.0, 50.0]
        npv := Numpy.Npv(0.08, cashflows)
        AhkTest.AssertTrue(Abs(npv - 12.743465408201466) <= 1e-9)
        irr := Numpy.Irr(cashflows)
        residual := 0.0
        for index, value in cashflows
            residual += value / ((1.0 + irr) ** (index - 1))
        AhkTest.AssertTrue(Abs(residual) <= 1e-9)

        rejected := false
        try Numpy.Pmt(0.0, 0, 1.0, 1.0)
        catch ValueError as err
            rejected := InStr(err.Message, "nper must be nonzero") > 0
        AhkTest.AssertTrue(rejected,
            "financial facade must expose native domain errors")

        activeMemory := Numpy.AllocatedMemory()
        loop 64 {
            ignored := Numpy.Fv(rate, periods, payment, present)
            ignored := Numpy.Npv(0.08, cashflows)
        }
        AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
            "financial facade scalar calls must not retain native memory")
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "financial facade must leave native memory unchanged")
        Numpy.Cleanup()
    }

    static TestDatetimeSurfaceFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        date := Numpy.DatetimeFromDate(2024, 2, 29, "D")
        AhkTest.AssertEqual(19782, date)
        timestamp := Numpy.DatetimeFromTime(
            2024, 2, 29, 12, 34, 56, "s")
        AhkTest.AssertEqual(1709210096, timestamp)
        parsed := Numpy.DatetimeFromString(
            "2024-02-29T12:34:56.123", "ms")
        AhkTest.AssertEqual(1709210096123, parsed)
        NumpyAssertArrayEqual(
            [2024, 2, 29], Numpy.DatetimeToDate(parsed, "ms"))
        NumpyAssertArrayEqual(
            [12, 34, 56], Numpy.DatetimeToTime(parsed, "ms"))
        AhkTest.AssertEqual(
            "2024-02-29T12:34:56.123",
            Numpy.DatetimeToString(parsed, "ms"))
        AhkTest.AssertEqual("ms", Numpy.DatetimeUnitName("ms"))
        AhkTest.AssertEqual(7, Numpy.Timedelta64(7, "ms"))
        AhkTest.AssertEqual(parsed + 3, Numpy.DatetimeAdd(parsed, 3, "ms"))
        AhkTest.AssertEqual(
            3, Numpy.DatetimeSubtract(parsed, parsed - 3, "ms"))
        AhkTest.AssertEqual(-1, Numpy.DatetimeCompare(parsed - 1, parsed))
        AhkTest.AssertEqual(0, Numpy.DatetimeCompare(parsed, parsed))
        AhkTest.AssertEqual(1, Numpy.DatetimeCompare(parsed + 1, parsed))

        AhkTest.AssertTrue(Numpy.IsBusday(0))
        AhkTest.AssertTrue(!Numpy.IsBusday(2))
        AhkTest.AssertEqual(5, Numpy.BusdayCount(0, 7))
        AhkTest.AssertEqual(4, Numpy.BusdayOffset(0, 2))

        nat := Numpy.DatetimeFromString("NaT", "ms")
        datetimeArray := Numpy.DatetimeArray(
            [0, 1, -1, nat], [2, 2], "ms")
        AhkTest.AssertEqual(Numpy.DT_DATETIME, datetimeArray.Dtype)
        NumpyAssertArrayEqual(
            ["1970-01-01T00:00:00.000",
             "1970-01-01T00:00:00.001",
             "1969-12-31T23:59:59.999", "NaT"],
            Numpy.DatetimeAsString(datetimeArray, "ms"))

        range := Numpy.ArangeDatetime(0, 7, 2, "ms")
        NumpyAssertArrayEqual(
            ["1970-01-01T00:00:00.000",
             "1970-01-01T00:00:00.002",
             "1970-01-01T00:00:00.004",
             "1970-01-01T00:00:00.006"],
            Numpy.DatetimeAsString(range, "ms"))

        nowSeconds := Numpy.DatetimeNow("s")
        AhkTest.AssertTrue(nowSeconds > 1700000000,
            "datetime now must expose a real UTC epoch timestamp")

        rejected := false
        try Numpy.DatetimeFromDate(2023, 2, 29, "D")
        catch ValueError as err
            rejected := InStr(err.Message, "cnp_datetime64_from_date") > 0
        AhkTest.AssertTrue(rejected,
            "datetime facade must expose labeled invalid-date errors")

        rejected := false
        try Numpy.BusdayOffset(2, 0)
        catch ValueError as err
            rejected := InStr(err.Message, "cnp_busday_offset") > 0
                && InStr(err.Message, "business day") > 0
        AhkTest.AssertTrue(rejected,
            "business-day facade must expose labeled weekend errors")

        rejected := false
        try Numpy.ArangeDatetime(0, 7, 0, "ms")
        catch ValueError as err
            rejected := InStr(err.Message, "cnp_arange_datetime") > 0
                && InStr(err.Message, "nonzero") > 0
        AhkTest.AssertTrue(rejected,
            "datetime range facade must expose labeled zero-step errors")

        activeMemory := Numpy.AllocatedMemory()
        loop 64 {
            ignored := Numpy.DatetimeToString(parsed, "ms")
            ignored := Numpy.DatetimeAsString(datetimeArray, "ms")
        }
        AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
            "datetime facade repeated formatting must release strings")

        range := 0
        datetimeArray := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "datetime facade must restore retained native bytes")
        Numpy.Cleanup()
    }

    static TestIteratorSurfaceFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])

        enumerated := Numpy.Ndenumerate(source)
        AhkTest.AssertEqual(6, enumerated.Length)
        expectedCoordinates := [[0, 0], [0, 1], [0, 2],
            [1, 0], [1, 1], [1, 2]]
        loop 6 {
            NumpyAssertArrayEqual(
                expectedCoordinates[A_Index], enumerated[A_Index][1])
            AhkTest.AssertEqual(A_Index + 0.0, enumerated[A_Index][2])
        }

        indices := Numpy.Ndindex([2, 3])
        AhkTest.AssertEqual(6, indices.Length)
        loop 6
            NumpyAssertArrayEqual(expectedCoordinates[A_Index], indices[A_Index])
        AhkTest.AssertEqual(0, Numpy.Ndindex([2, 0, 3]).Length)

        rejected := false
        try Numpy.Ndindex([2, -1])
        catch ValueError as err
            rejected := InStr(err.Message, "cnp_ndindex_next") > 0
        AhkTest.AssertTrue(rejected,
            "ndindex facade must expose labeled negative-shape errors")

        activeMemory := Numpy.AllocatedMemory()
        loop 64 {
            ignored := Numpy.Ndenumerate(source)
            ignored := Numpy.Ndindex([2, 3])
        }
        AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
            "iterator facade calls must not retain native memory")
        source := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "iterator facade must restore retained native bytes")
        Numpy.Cleanup()
    }

    static TestBufferRepresentationFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := Numpy.IntArray([1, -2, 300, -400], [2, 2])

        bytes := Numpy.ToBytes(source)
        AhkTest.AssertEqual(32, bytes.Size)
        AhkTest.AssertEqual(1, NumGet(bytes, 0, "Int64"))
        AhkTest.AssertEqual(-2, NumGet(bytes, 8, "Int64"))
        AhkTest.AssertEqual(300, NumGet(bytes, 16, "Int64"))
        AhkTest.AssertEqual(-400, NumGet(bytes, 24, "Int64"))
        NumpyAssertArrayEqual(
            [1.0, -2.0, 300.0, -400.0], Numpy.ToList(source))

        raw := Buffer(8, 0)
        NumPut("Short", 1, raw, 0)
        NumPut("Short", -2, raw, 2)
        NumPut("Short", 300, raw, 4)
        NumPut("Short", -400, raw, 6)
        copied := Numpy.FromBuffer(raw, Numpy.DT_INT16)
        NumpyAssertArrayEqual([1.0, -2.0, 300.0, -400.0], copied.ToArray())
        NumPut("Short", 99, raw, 0)
        NumpyAssertArrayEqual([1.0, -2.0, 300.0, -400.0], copied.ToArray())

        AhkTest.AssertEqual("00000101", Numpy.BaseRepr(5, 2, 5))
        AhkTest.AssertEqual("-8000000000000000",
            Numpy.BaseRepr(-9223372036854775807 - 1, 16))
        AhkTest.AssertEqual("-101", Numpy.BinaryRepr(-5))
        AhkTest.AssertEqual("11111011", Numpy.BinaryRepr(-5, 8))

        rejected := false
        try Numpy.FromBuffer(Buffer(3, 0), Numpy.DT_INT16)
        catch ValueError as err
            rejected := InStr(err.Message, "cnp_frombuffer") > 0
        AhkTest.AssertTrue(rejected,
            "frombuffer facade must expose labeled element-size errors")

        rejected := false
        try Numpy.BaseRepr(5, 1)
        catch ValueError as err
            rejected := InStr(err.Message, "cnp_base_repr") > 0
        AhkTest.AssertTrue(rejected,
            "base representation facade must expose labeled base errors")

        activeMemory := Numpy.AllocatedMemory()
        loop 64 {
            ignored := Numpy.ToBytes(source)
            ignored := Numpy.ToList(source)
            ignored := Numpy.BinaryRepr(-123456789, 64)
        }
        AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
            "buffer facade calls must release native result buffers")
        copied := 0
        source := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "buffer facade must restore retained native bytes")
        Numpy.Cleanup()
    }

    static TestGridAssemblyFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        NumpyAssertArrayEqual(
            [5, 3, 1],
            Numpy.BroadcastShapes([[5, 1, 1], [3, 1], [1]]))
        broadcastLeft := Numpy.Zeros([3, 1])
        broadcastRight := Numpy.Zeros([1, 4])
        AhkTest.AssertTrue(Numpy.CanBroadcast(broadcastLeft, broadcastRight))
        incompatible := Numpy.Zeros([2, 3])
        AhkTest.AssertTrue(!Numpy.CanBroadcast(incompatible, broadcastRight))

        indices := Numpy.Indices([2, 3])
        NumpyAssertArrayEqual([2, 2, 3], indices.Shape)
        NumpyAssertArrayEqual(
            [0, 0, 0, 1, 1, 1, 0, 1, 2, 0, 1, 2],
            indices.ToArray())

        dense := Numpy.Mgrid([0, 1], [3, 6], [1, 2])
        AhkTest.AssertEqual(2, dense.Length)
        NumpyAssertArrayEqual([3, 3], dense[1].Shape)
        NumpyAssertArrayEqual(
            [0, 0, 0, 1, 1, 1, 2, 2, 2], dense[1].ToArray())
        NumpyAssertArrayEqual(
            [1, 3, 5, 1, 3, 5, 1, 3, 5], dense[2].ToArray())

        sparse := Numpy.Ogrid([3, 5], [0, -1], [-1, -2])
        NumpyAssertArrayEqual([3, 1], sparse[1].Shape)
        NumpyAssertArrayEqual([3, 2, 1], sparse[1].ToArray())
        NumpyAssertArrayEqual([1, 3], sparse[2].Shape)
        NumpyAssertArrayEqual([5, 3, 1], sparse[2].ToArray())

        blockA := Numpy.Array([1.0, 2.0], [2, 1])
        blockB := Numpy.Array([3.0, 4.0, 5.0, 6.0], [2, 2])
        blockC := Numpy.Array([7.0], [1, 1])
        blockD := Numpy.Array([8.0, 9.0], [1, 2])
        assembled := Numpy.Block([[blockA, blockB], [blockC, blockD]])
        matrix := Numpy.Bmat([[blockA, blockB], [blockC, blockD]])
        expected := [1.0, 3.0, 4.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        NumpyAssertArrayEqual(expected, assembled.ToArray())
        NumpyAssertArrayEqual(expected, matrix.ToArray())

        mesh := Numpy.Meshgrid(
            [Numpy.Array([1.0, 2.0]), Numpy.Array([10.0, 20.0, 30.0])])
        AhkTest.AssertEqual(2, mesh.Length)
        NumpyAssertArrayEqual(
            [1.0, 2.0, 1.0, 2.0, 1.0, 2.0], mesh[1].ToArray())

        rejected := false
        try Numpy.Mgrid([0, 0], [3, 3], [1, 0])
        catch ValueError as err
            rejected := InStr(err.Message, "cnp_mgrid") > 0
        AhkTest.AssertTrue(rejected,
            "mgrid facade must expose labeled zero-step errors")

        activeMemory := Numpy.AllocatedMemory()
        loop 64 {
            ignored := Numpy.BroadcastShapes([[3, 1], [1, 4]])
            ignored := Numpy.Indices([2, 3])
            ignored := Numpy.Mgrid([0, 0], [2, 3], [1, 1])
        }
        ignored := 0
        AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
            "grid facade repeated calls must release native results")

        mesh := 0
        matrix := 0
        assembled := 0
        blockD := 0
        blockC := 0
        blockB := 0
        blockA := 0
        sparse := 0
        dense := 0
        indices := 0
        incompatible := 0
        broadcastRight := 0
        broadcastLeft := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "grid facade must restore retained native bytes")
        Numpy.Cleanup()
    }

    static TestLegacyNumericUtilityFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        integers := Numpy.IntArray([-1, -3, 0, 255])
        counts := integers.BitwiseCount()
        AhkTest.AssertEqual(Numpy.DT_UINT8, counts.Dtype)
        NumpyAssertArrayEqual([1, 2, 0, 8], counts.ToArray())

        values := Numpy.Array([1.5, -2.25, 3.75, 8.5], [2, 2])
        AhkTest.AssertEqual(8.5, values.Item(-1))
        AhkTest.AssertEqual(1.5, Numpy.Item(values, -4))

        invalidItemRejected := false
        try values.Item(4)
        catch ValueError as err
            invalidItemRejected := InStr(err.Message, "cnp_item") > 0
        AhkTest.AssertTrue(invalidItemRejected,
            "item facade must expose labeled native index errors")

        roots := Numpy.Zeros([4])
        rootSource := Numpy.Array([0.0, 1.0, 4.0, 9.0])
        rootSource.SqrtInto(roots)
        NumpyAssertArrayEqual([0.0, 1.0, 2.0, 3.0], roots.ToArray())

        activeMemory := Numpy.AllocatedMemory()
        loop 64 {
            temporary := integers.BitwiseCount()
            temporary := 0
            ignored := values.Item(-1)
        }
        AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
            "numeric utility facade calls must release native results")

        rootSource := 0
        roots := 0
        values := 0
        counts := 0
        integers := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "numeric utility facade must restore retained native bytes")
        Numpy.Cleanup()
    }

    static TestCalculusSurfaceFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        source := Numpy.Array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0], [2, 3])
        differences := source.Diff(1, 1)
        gradients := Numpy.Gradient(source, 1)
        phases := Numpy.Array([0.0, 4.0, 8.0, 1.0, -3.0, -7.0], [2, 3])
        unwrapped := phases.Unwrap()

        NumpyAssertArrayEqual([2, 2], differences.Shape)
        NumpyAssertArrayEqual([1.0, 3.0, 7.0, 9.0], differences.ToArray())
        NumpyAssertArrayEqual([2, 3], gradients.Shape)
        NumpyAssertArrayEqual(
            [1.0, 2.0, 3.0, 7.0, 8.0, 9.0], gradients.ToArray())
        NumpyAssertArrayNear([
            0.0, -2.2831853071795862, -4.5663706143591725,
            1.0, 3.2831853071795862, 5.5663706143591725
        ], unwrapped.ToArray(), 1e-12)

        rejected := false
        try source.Diff(1, 2)
        catch ValueError as err
            rejected := InStr(err.Message, "cnp_diff") > 0
        AhkTest.AssertTrue(rejected,
            "diff facade must expose labeled native axis errors")

        activeMemory := Numpy.AllocatedMemory()
        loop 64 {
            temporary := source.Diff(2, 1)
            temporary := 0
            temporary := source.Gradient(1)
            temporary := 0
            temporary := phases.Unwrap()
            temporary := 0
        }
        AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
            "calculus facade calls must release native results")

        unwrapped := 0
        phases := 0
        gradients := 0
        differences := 0
        source := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "calculus facade must restore retained native bytes")
        Numpy.Cleanup()
    }

    static TestSamplingUtilitySurfaceFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        source := Numpy.Array([-2.0, -1.0, 0.0, 1.0, 2.0])
        histogram := source.Histogram(4)
        NumpyAssertArrayEqual([1, 1, 1, 2], histogram.ToArray())

        x := Numpy.Array([5.0, 5.0])
        y := Numpy.Array([-2.0, -2.0])
        histogram2d := x.Histogram2d(y, 3)
        NumpyAssertArrayEqual([3, 3], histogram2d.Shape)
        NumpyAssertArrayEqual([
            0.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 0.0
        ], histogram2d.ToArray())

        query := Numpy.Array([-1.0, 0.5, 1.5, 3.5], [2, 2])
        xp := Numpy.Array([0.0, 1.0, 2.0, 3.0])
        fp := Numpy.Array([0.0, 1.0, 4.0, 9.0])
        interpolated := query.Interp(xp, fp)
        bounded := query.InterpNd(xp, fp, 0.0, 0.0)
        NumpyAssertArrayEqual([2, 2], interpolated.Shape)
        NumpyAssertArrayNear([0.0, 0.5, 2.5, 9.0],
            interpolated.ToArray(), 1e-12)
        NumpyAssertArrayNear([0.0, 0.5, 2.5, 0.0],
            bounded.ToArray(), 1e-12)

        quietNan := NumpyDoubleFromBits(0x7ff80000)
        positiveInfinity := NumpyDoubleFromBits(0x7ff00000)
        negativeInfinity := NumpyDoubleFromBits(0xfff00000)
        special := Numpy.Array([
            quietNan, positiveInfinity, negativeInfinity, -0.0, 3.0])
        finite := special.NanToNum(7.0, 11.0, -13.0)
        NumpyAssertArrayEqual(
            [7.0, 11.0, -13.0, -0.0, 3.0], finite.ToArray())

        powers := Numpy.IntArray([1, 2, -3])
        vandermonde := powers.Vander()
        emptyVandermonde := powers.Vander(0)
        NumpyAssertArrayEqual([3, 3], vandermonde.Shape)
        NumpyAssertArrayEqual([
            1, 1, 1,
            4, 2, 1,
            9, -3, 1
        ], vandermonde.ToArray())
        NumpyAssertArrayEqual([3, 0], emptyVandermonde.Shape)
        NumpyAssertArrayEqual([], emptyVandermonde.ToArray())

        rejected := false
        try temporary := source.Histogram(0)
        catch ValueError as err
            rejected := InStr(err.Message, "cnp_histogram") > 0
        temporary := 0
        AhkTest.AssertTrue(rejected,
            "histogram facade must expose labeled bin errors")

        activeMemory := Numpy.AllocatedMemory()
        loop 64 {
            temporary := source.Histogram(4)
            temporary := 0
            temporary := x.Histogram2d(y, 3)
            temporary := 0
            temporary := query.Interp(xp, fp)
            temporary := 0
            temporary := query.InterpNd(xp, fp, 0.0, 0.0)
            temporary := 0
            temporary := special.NanToNum(7.0, 11.0, -13.0)
            temporary := 0
            temporary := powers.Vander(3)
            temporary := 0
        }
        AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
            "sampling utility facade calls must release native results")

        emptyVandermonde := 0
        vandermonde := 0
        powers := 0
        finite := 0
        special := 0
        bounded := 0
        interpolated := 0
        fp := 0
        xp := 0
        query := 0
        histogram2d := 0
        y := 0
        x := 0
        histogram := 0
        source := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "sampling utility facade must restore retained native bytes")
        Numpy.Cleanup()
    }

    static TestFftSurfaceFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()

        source := Numpy.Array([0.0, 1.0, 2.0, 3.0])
        spectrum := Numpy.Fft.Fft(source)
        spectrumReal := Numpy.Real(spectrum)
        spectrumImag := Numpy.Imag(spectrum)
        inverse := Numpy.Fft.Ifft(spectrum)
        inverseReal := Numpy.Real(inverse)
        realSpectrum := Numpy.Fft.Rfft(source)
        realInverse := Numpy.Fft.Irfft(realSpectrum, 4)
        NumpyAssertArrayNear([6.0, -2.0, -2.0, -2.0],
            spectrumReal.ToArray(), 1e-12)
        NumpyAssertArrayNear([0.0, 2.0, 0.0, -2.0],
            spectrumImag.ToArray(), 1e-12)
        NumpyAssertArrayNear([0.0, 1.0, 2.0, 3.0],
            inverseReal.ToArray(), 1e-12)
        NumpyAssertArrayNear([0.0, 1.0, 2.0, 3.0],
            realInverse.ToArray(), 1e-12)

        matrix := Numpy.Array([1.0, 2.0, 3.0, 4.0], [2, 2])
        fft2 := Numpy.Fft.Fft2(matrix)
        fft2Real := Numpy.Real(fft2)
        ifft2 := Numpy.Fft.Ifft2(fft2)
        ifft2Real := Numpy.Real(ifft2)
        fftn := Numpy.Fft.Fftn(matrix, [-2, -1])
        ifftn := Numpy.Fft.Ifftn(fftn, [-2, -1])
        ifftnReal := Numpy.Real(ifftn)
        rfftn := Numpy.Fft.Rfftn(matrix, [-2, -1])
        irfftn := Numpy.Fft.Irfftn(rfftn, [-2, -1], [2, 2])
        NumpyAssertArrayNear([10.0, -2.0, -4.0, 0.0],
            fft2Real.ToArray(), 1e-12)
        NumpyAssertArrayNear(matrix.ToArray(), ifft2Real.ToArray(), 1e-12)
        NumpyAssertArrayNear(matrix.ToArray(), ifftnReal.ToArray(), 1e-12)
        NumpyAssertArrayNear(matrix.ToArray(), irfftn.ToArray(), 1e-12)

        frequencies := Numpy.Fft.Fftfreq(4, 0.5)
        realFrequencies := Numpy.Fft.Rfftfreq(4, 0.5)
        shiftSource := Numpy.IntArray([0, 1, 2, 3, 4, 5], [2, 3])
        shifted := Numpy.Fft.Fftshift(shiftSource)
        unshifted := Numpy.Fft.Ifftshift(shifted)
        NumpyAssertArrayNear([0.0, 0.5, -1.0, -0.5],
            frequencies.ToArray(), 0)
        NumpyAssertArrayNear([0.0, 0.5, 1.0],
            realFrequencies.ToArray(), 0)
        NumpyAssertArrayNear([5, 3, 4, 2, 0, 1], shifted.ToArray(), 0)
        NumpyAssertArrayNear(shiftSource.ToArray(), unshifted.ToArray(), 0)

        inverseHermitian := Numpy.Fft.Ihfft(source)
        hermitianRoundTrip := Numpy.Fft.Hfft(inverseHermitian, 4)
        NumpyAssertArrayNear(source.ToArray(),
            hermitianRoundTrip.ToArray(), 1e-12)

        rejected := false
        try temporary := Numpy.Fft.Fft(source, 0)
        catch ValueError as err
            rejected := InStr(err.Message, "cnp_fft") > 0
        temporary := 0
        AhkTest.AssertTrue(rejected,
            "FFT facade must expose labeled invalid-length errors")

        activeMemory := Numpy.AllocatedMemory()
        loop 32 {
            temporary := Numpy.Fft.Fft(source)
            temporary := 0
            temporary := Numpy.Fft.Ifft(spectrum)
            temporary := 0
            temporary := Numpy.Fft.Rfft(source)
            temporary := 0
            temporary := Numpy.Fft.Irfft(realSpectrum, 4)
            temporary := 0
            temporary := Numpy.Fft.Fft2(matrix)
            temporary := 0
            temporary := Numpy.Fft.Ifft2(fft2)
            temporary := 0
            temporary := Numpy.Fft.Fftn(matrix, [-2, -1])
            temporary := 0
            temporary := Numpy.Fft.Ifftn(fftn, [-2, -1])
            temporary := 0
            temporary := Numpy.Fft.Rfftn(matrix, [-2, -1])
            temporary := 0
            temporary := Numpy.Fft.Irfftn(rfftn, [-2, -1], [2, 2])
            temporary := 0
            temporary := Numpy.Fft.Hfft(inverseHermitian, 4)
            temporary := 0
            temporary := Numpy.Fft.Ihfft(source)
            temporary := 0
        }
        AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
            "FFT facade calls must release native results")

        hermitianRoundTrip := 0
        inverseHermitian := 0
        unshifted := 0
        shifted := 0
        shiftSource := 0
        realFrequencies := 0
        frequencies := 0
        irfftn := 0
        rfftn := 0
        ifftnReal := 0
        ifftn := 0
        fftn := 0
        ifft2Real := 0
        ifft2 := 0
        fft2Real := 0
        fft2 := 0
        matrix := 0
        realInverse := 0
        realSpectrum := 0
        inverseReal := 0
        inverse := 0
        spectrumImag := 0
        spectrumReal := 0
        spectrum := 0
        source := 0
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "FFT facade must restore retained native bytes")
        Numpy.Cleanup()
    }

    static TestCharacterSurfaceFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        values := [" Ab ", "xy", "", "-3"]

        NumpyAssertArrayEqual(
            [" Ab !", "xy!", "!", "-3!"],
            Numpy.Char_Add(values, ["!"]))
        NumpyAssertArrayEqual(
            [" Ab  Ab ", "xyxy", "", "-3-3"],
            Numpy.Char_Multiply(values, [2]))
        NumpyAssertArrayEqual(
            [" AB ", "XY", "", "-3"], Numpy.Char_Upper(values))
        NumpyAssertArrayEqual(
            [" ab ", "xy", "", "-3"], Numpy.Char_Lower(values))
        NumpyAssertArrayEqual(
            ["Ab", "xy", "", "-3"], Numpy.Char_Strip(values))
        NumpyAssertArrayEqual(
            ["b ", "xy", "", "-3"], Numpy.Char_Lstrip(values, " A"))
        NumpyAssertArrayEqual(
            [" A", "xy", "", "-3"], Numpy.Char_Rstrip(values, " b"))
        NumpyAssertArrayEqual(
            [" Ab", ".xy", "...", ".-3"],
            Numpy.Char_Center(values, 3, "."))
        NumpyAssertArrayEqual(
            [" A", "xy", "..", "-3"], Numpy.Char_Ljust(values, 2, "."))
        NumpyAssertArrayEqual(
            [" A", "xy", "..", "-3"], Numpy.Char_Rjust(values, 2, "."))
        NumpyAssertArrayEqual(
            [" A", "xy", "00", "-3"], Numpy.Char_Zfill(values, 2))
        NumpyAssertArrayEqual(
            [4, 2, 0, 2], Numpy.Char_Strlen(values))
        NumpyAssertArrayEqual(
            [1, 0, 0, 0], Numpy.Char_Count(values, "A"))
        NumpyAssertArrayEqual(
            [1, -1, -1, -1], Numpy.Char_Find(values, "A"))
        NumpyAssertArrayEqual(
            ["X XAb ", "XxXy", "X", "X-X3"],
            Numpy.Char_Replace(values, "", "X", 2))
        NumpyAssertArrayEqual(
            ["a-b", "", "X-Y-Z"],
            Numpy.Char_Join(["ab", "", "XYZ"], "-"))

        rejected := false
        try Numpy.Char_Add(["a", "b"], ["1", "2", "3"])
        catch ValueError as err
            rejected := InStr(err.Message, "cnp_char_add") > 0
                && InStr(err.Message, "broadcast") > 0
        AhkTest.AssertTrue(rejected,
            "character facade must expose labeled broadcast errors")

        activeMemory := Numpy.AllocatedMemory()
        loop 64 {
            ignored := Numpy.Char_Upper(values)
            ignored := Numpy.Char_Replace(values, "", "X")
            ignored := Numpy.Char_Strlen(values)
        }
        AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
            "character facade repeated calls must release native results")
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "character facade must restore retained native bytes")
        Numpy.Cleanup()
    }

    static TestTextIoSurfaceFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        testDirectory := A_ScriptDir "\..\build\test-results"
        DirCreate testDirectory
        savedPath := testDirectory "\ahk-text-io-saved.csv"
        missingPath := testDirectory "\ahk-text-io-missing.csv"
        if FileExist(savedPath)
            FileDelete savedPath
        if FileExist(missingPath)
            FileDelete missingPath

        source := 0
        loaded := 0
        generated := 0
        temporary := 0
        try {
            source := Numpy.IntArray([1, 2, 3, 4], [2, 2])
            AhkTest.AssertEqual(
                "[[1, 2],`n [3, 4]]",
                Numpy.Array2String(source, 128))
            AhkTest.AssertEqual(
                "[[1.00, 2.00],`n [3.00, 4.00]]",
                Numpy.ArrayToString(source, "%.2f"))
            AhkTest.AssertEqual(
                "1;2`n3;4", Numpy.ArrayToCsv(source, 128, ";"))

            Numpy.Savetxt(savedPath, source, ",", "%.6g")
            loaded := Numpy.Loadtxt(savedPath, ",", Numpy.DT_INT64)
            NumpyAssertArrayEqual([2, 2], loaded.Shape)
            NumpyAssertArrayEqual([1, 2, 3, 4], loaded.ToArray())

            file := FileOpen(missingPath, "w", "UTF-8-RAW")
            file.Write("first,second,third`n1,,3`n4,bad,6`n")
            file.Close()
            generated := Numpy.Genfromtxt(
                missingPath, ",", 1, 2, Numpy.DT_FLOAT64)
            generatedValues := generated.ToArray()
            NumpyAssertArrayEqual([2, 3], generated.Shape)
            AhkTest.AssertEqual(1.0, generatedValues[1])
            AhkTest.AssertTrue(
                generatedValues[2] != generatedValues[2],
                "genfromtxt facade must preserve missing NaN")
            AhkTest.AssertEqual(3.0, generatedValues[3])
            AhkTest.AssertEqual(4.0, generatedValues[4])
            AhkTest.AssertTrue(
                generatedValues[5] != generatedValues[5],
                "genfromtxt facade must map invalid numeric text to NaN")
            AhkTest.AssertEqual(6.0, generatedValues[6])

            stdoutHandle := DllCall(
                "kernel32\GetStdHandle", "Int", -11, "Ptr")
            stdoutAvailable := stdoutHandle != 0 && stdoutHandle != -1

            printSucceeded := false
            printRejected := false
            try {
                Numpy.ArrayPrint(source)
                printSucceeded := true
            } catch Error as err {
                printRejected := InStr(
                    err.Message, "cnp_array_print") > 0
                    && InStr(err.Message, "stdout") > 0
            }
            AhkTest.AssertTrue(
                stdoutAvailable ? printSucceeded : printRejected,
                stdoutAvailable
                    ? "inherited stdout must accept real array_print output"
                    : "GUI AHK must expose the real array_print stdout error")

            dispSucceeded := false
            dispRejected := false
            try {
                Numpy.Disp("AHK text facade output")
                dispSucceeded := true
            } catch Error as err {
                dispRejected := InStr(err.Message, "cnp_disp") > 0
                    && InStr(err.Message, "stdout") > 0
            }
            AhkTest.AssertTrue(
                stdoutAvailable ? dispSucceeded : dispRejected,
                stdoutAvailable
                    ? "inherited stdout must accept real disp output"
                    : "GUI AHK must expose the real disp stdout error")

            bufferRejected := false
            try Numpy.ArrayToCsv(source, 4, ",")
            catch ValueError as err
                bufferRejected := InStr(
                    err.Message, "cnp_array_to_csv") > 0
            AhkTest.AssertTrue(bufferRejected,
                "CSV facade must expose explicit buffer capacity errors")

            rowLimitRejected := false
            try temporary := Numpy.Genfromtxt(
                missingPath, ",", 1, 0, Numpy.DT_FLOAT64)
            catch ValueError as err
                rowLimitRejected := InStr(
                    err.Message, "cnp_genfromtxt") > 0
            temporary := 0
            AhkTest.AssertTrue(rowLimitRejected,
                "genfromtxt facade must expose labeled row-limit errors")

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                ignored := Numpy.ArrayToString(source, "%g")
                ignored := Numpy.Array2String(source, 128)
                ignored := Numpy.ArrayToCsv(source, 128, ",")
                temporary := Numpy.Loadtxt(
                    savedPath, ",", Numpy.DT_INT64)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "text facade repeated calls must release native owners")
        } finally {
            temporary := 0
            generated := 0
            loaded := 0
            source := 0
            if FileExist(savedPath)
                FileDelete savedPath
            if FileExist(missingPath)
                FileDelete missingPath
        }
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "text facade must restore retained native bytes")
        Numpy.Cleanup()
    }

    static TestStructuredDtypeSurfaceFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        testDirectory := A_ScriptDir "\..\build\test-results"
        DirCreate testDirectory
        path := testDirectory "\ahk-structured-records.csv"
        missingPath := testDirectory "\ahk-structured-missing.csv"
        if FileExist(path)
            FileDelete path
        if FileExist(missingPath)
            FileDelete missingPath

        csv := FileOpen(path, "w", "UTF-8-RAW")
        csv.Write("identifier,score`n1,2.5`n3,4.75`n")
        csv.Close()

        source := 0
        reinterpreted := 0
        highWords := 0
        destination := 0
        replacement := 0
        swapped := 0
        records := 0
        identifiers := 0
        score := 0
        recordValues := 0
        record := 0
        identifierField := 0
        scoreField := 0
        loaded := 0
        temporary := 0
        try {
            source := Numpy.IntArray([4294967298])
            reinterpreted := Numpy.ReinterpretView(
                source, Numpy.DT_INT32)
            source := 0
            NumpyAssertArrayEqual([2, 1], reinterpreted.ToArray())

            source := Numpy.IntArray([4294967298])
            highWords := Numpy.GetField(source, Numpy.DT_INT32, 4)
            source := 0
            NumpyAssertArrayEqual([1], highWords.ToArray())

            destination := Numpy.IntArray([4294967298])
            replacement := Numpy.IntArray([7], [])
            Numpy.SetField(
                destination, replacement, Numpy.DT_INT32, 0)
            NumpyAssertArrayEqual([4294967303], destination.ToArray())

            source := Numpy.IntArray([0x0102030405060708])
            swapped := Numpy.NewByteOrder(source)
            source := 0
            NumpyAssertArrayEqual(
                [0x0807060504030201], swapped.ToArray())

            dtypeId := Numpy.StructDtypeCreate(
                ["identifier", "score"],
                [Numpy.DT_INT32, Numpy.DT_FLOAT64])
            AhkTest.AssertEqual(12, Numpy.StructDtypeItemsize(dtypeId))
            AhkTest.AssertEqual(2, Numpy.StructDtypeNfields(dtypeId))
            AhkTest.AssertEqual(
                "identifier", Numpy.StructDtypeFieldName(dtypeId, 0))
            AhkTest.AssertEqual(
                "score", Numpy.StructDtypeFieldName(dtypeId, 1))
            AhkTest.AssertEqual(
                4, Numpy.StructDtypeFieldOffset(dtypeId, 1))
            AhkTest.AssertEqual(
                1, Numpy.StructDtypeFindField(dtypeId, "score"))
            AhkTest.AssertEqual(
                "identifier,score", Numpy.RecarrayNames(dtypeId))

            records := Numpy.RecarrayNew([2, 2], dtypeId)
            identifiers := Numpy.IntArray([1, 2, 3, 4], [2, 2])
            score := Numpy.Array([2.5], [])
            Numpy.RecarraySetField(
                records, "identifier", dtypeId, identifiers)
            Numpy.RecarraySetField(records, "score", dtypeId, score)
            recordValues := Numpy.Array([99.0, 1.25])
            Numpy.RecarraySetRecord(records, 2, dtypeId, recordValues)
            record := Numpy.RecarrayGetRecord(records, 2, dtypeId)
            NumpyAssertArrayEqual([99.0, 1.25], record.ToArray())
            identifierField := Numpy.RecarrayGetField(
                records, "identifier", dtypeId)
            scoreField := Numpy.RecarrayGetField(
                records, "score", dtypeId)
            records := 0
            NumpyAssertArrayEqual(
                [1, 2, 99, 4], identifierField.ToArray())
            NumpyAssertArrayEqual(
                [2.5, 2.5, 1.25, 2.5], scoreField.ToArray())

            loaded := Numpy.Recfromtxt(
                path, ",", 1, Numpy.DT_FLOAT64)
            NumpyAssertArrayEqual([2, 2], loaded.Shape)
            NumpyAssertArrayEqual(
                [1.0, 2.5, 3.0, 4.75], loaded.ToArray())

            duplicateRejected := false
            try Numpy.StructDtypeCreate(
                ["same", "same"],
                [Numpy.DT_INT32, Numpy.DT_FLOAT64])
            catch ValueError as err
                duplicateRejected := InStr(
                    err.Message, "cnp_struct_dtype_create") > 0
            AhkTest.AssertTrue(duplicateRejected,
                "structured dtype duplicate-name error must remain labeled")

            missingRejected := false
            try temporary := Numpy.RecarrayGetField(
                identifierField, "missing", dtypeId)
            catch Error as err
                missingRejected := InStr(
                    err.Message, "cnp_recarray_get_field") > 0
            temporary := 0
            AhkTest.AssertTrue(missingRejected,
                "record field errors must remain labeled")

            fileRejected := false
            try temporary := Numpy.Recfromtxt(
                missingPath, ",", 0, Numpy.DT_FLOAT64)
            catch Error as err
                fileRejected := InStr(err.Message, "cnp_recfromtxt") > 0
            temporary := 0
            AhkTest.AssertTrue(fileRejected,
                "recfromtxt file-open errors must remain labeled")

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.RecarrayNew([2, 2], dtypeId)
                Numpy.RecarraySetField(
                    temporary, "identifier", dtypeId, identifiers)
                projected := Numpy.RecarrayGetField(
                    temporary, "identifier", dtypeId)
                temporary := 0
                projected := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "structured repeated calls must release native owners")
        } finally {
            temporary := 0
            loaded := 0
            scoreField := 0
            identifierField := 0
            record := 0
            recordValues := 0
            score := 0
            identifiers := 0
            records := 0
            swapped := 0
            replacement := 0
            destination := 0
            highWords := 0
            reinterpreted := 0
            source := 0
            if FileExist(path)
                FileDelete path
        }

        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "structured facade must restore retained native bytes")
        Numpy.Cleanup()
    }

    static TestFunctionalCallbackSurfaceFacadeSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := 0
        applied := 0
        appliedOver := 0
        generated := 0
        iterated := 0
        pyfuncResult := 0
        vectorized := 0
        conditionOne := 0
        conditionTwo := 0
        choiceOne := 0
        choiceTwo := 0
        selected := 0
        pieceSource := 0
        piecewise := 0
        destination := 0
        indices := 0
        values := 0
        badIndices := 0
        temporary := 0
        try {
            source := Numpy.Array([1, 2, 3, 4, 5, 6], [2, 3])
            applied := Numpy.ApplyAlongAxis(
                NumpySumCallbackLine, 0, source)
            NumpyAssertArrayEqual([3], applied.Shape)
            NumpyAssertArrayEqual([5.0, 7.0, 9.0], applied.ToArray())

            source := Numpy.Array([
                1, 2, 3, 4, 5, 6, 7, 8,
                9, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21, 22, 23, 24
            ], [2, 3, 4])
            appliedOver := Numpy.ApplyOverAxes(
                NumpySumCallbackLine, source, [0, 2])
            NumpyAssertArrayEqual([1, 3, 1], appliedOver.Shape)
            NumpyAssertArrayEqual([68.0, 100.0, 132.0], appliedOver.ToArray())

            generated := Numpy.FromFunction(
                (row, column) => row * 10 + column + 0.5, [2, 3])
            NumpyAssertArrayEqual([2, 3], generated.Shape)
            NumpyAssertArrayEqual(
                [0.5, 1.5, 2.5, 10.5, 11.5, 12.5],
                generated.ToArray())

            iterator := NumpySequenceCallback([1.9, -2.2, 7.0, 8.0])
            iterated := Numpy.FromIter(
                iterator, 4, Numpy.DT_INT16)
            AhkTest.AssertEqual(4, iterator.Index)
            NumpyAssertArrayEqual([1, -2, 7, 8], iterated.ToArray())

            negativeIterator := NumpyNegativeCountCallback()
            negativeCountRejected := false
            try temporary := Numpy.FromIter(
                negativeIterator, -1, Numpy.DT_FLOAT64)
            catch ValueError as err
                negativeCountRejected := InStr(
                    err.Message, "count must be non-negative") > 0
            temporary := 0
            AhkTest.AssertTrue(negativeCountRejected,
                "FromIter must reject a negative count before registration")
            AhkTest.AssertEqual(0, negativeIterator.Calls,
                "negative FromIter count must not invoke the callback")
            AhkTest.AssertEqual(0, Numpy.CallbackContexts.Count,
                "negative FromIter count must not register a callback context")

            source := Numpy.Array([1.0, 2.0, 3.0])
            pyfuncResult := Numpy.FromPyFunc(
                value => value * 2 + 1, source)
            vectorized := Numpy.Vectorize(
                value => value * 2 + 1, source)
            NumpyAssertArrayEqual([3.0, 5.0, 7.0], pyfuncResult.ToArray())
            NumpyAssertArrayEqual([3.0, 5.0, 7.0], vectorized.ToArray())

            conditionOne := Numpy.IntArray([1, 0], [2, 1])
            conditionTwo := Numpy.IntArray([0, 1, 1], [1, 3])
            choiceOne := Numpy.IntArray([1, 2, 3], [1, 3])
            choiceTwo := Numpy.Array([10.5, 20.5], [2, 1])
            selected := Numpy.Select(
                [conditionOne, conditionTwo],
                [choiceOne, choiceTwo], -1.5)
            NumpyAssertArrayEqual([2, 3], selected.Shape)
            NumpyAssertArrayEqual(
                [1.0, 2.0, 3.0, -1.5, 20.5, 20.5],
                selected.ToArray())

            pieceSource := Numpy.IntArray([0, 1, 2])
            conditionOne := Numpy.IntArray([1, 1, 0])
            conditionTwo := Numpy.IntArray([0, 1, 1])
            pieceCallback := NumpyPiecewiseOrderCallback()
            piecewise := Numpy.Piecewise(
                pieceSource, [conditionOne, conditionTwo], pieceCallback)
            NumpyAssertArrayEqual([10, 31, 42], piecewise.ToArray())
            NumpyAssertArrayEqual(
                [0.0, 1.0, 1.0, 2.0], pieceCallback.Calls)

            destination := Numpy.IntArray([
                0, 4, 8, 1, 5, 9,
                2, 6, 10, 3, 7, 11
            ], [4, 3])
            indices := Numpy.IntArray([
                0, -1, 1, 0, 2, 1, 0, 2
            ], [4, 2])
            values := Numpy.Array([100.9, 200.1])
            Numpy.PutAlongAxis(destination, indices, values, -1)
            NumpyAssertArrayEqual([
                100, 4, 200, 200, 100, 9,
                2, 200, 100, 100, 7, 200
            ], destination.ToArray())

            beforeFailure := destination.ToArray()
            badIndices := Numpy.IntArray([
                0, 3, 1, 0, 2, 1, 0, 2
            ], [4, 2])
            indexRejected := false
            try Numpy.PutAlongAxis(
                destination, badIndices, values, 1)
            catch Error as err
                indexRejected := InStr(
                    err.Message, "cnp_put_along_axis") > 0
            AhkTest.AssertTrue(indexRejected,
                "put_along_axis must expose its native index error")
            NumpyAssertArrayEqual(beforeFailure, destination.ToArray())

            callbackRejected := false
            try temporary := Numpy.Vectorize(
                NumpyFailingNumericCallback(), source)
            catch Error as err
                callbackRejected := err.Message = "functional callback boom"
            temporary := 0
            AhkTest.AssertTrue(callbackRejected,
                "callback exceptions must be rethrown without translation")
            AhkTest.AssertEqual(0, Numpy.CallbackContexts.Count,
                "callback contexts must be removed after exceptions")

            activeMemory := Numpy.AllocatedMemory()
            loop 64 {
                temporary := Numpy.FromPyFunc(
                    value => value + 1, source)
                temporary := 0
            }
            AhkTest.AssertEqual(activeMemory, Numpy.AllocatedMemory(),
                "functional callback repeats must release native results")
        } finally {
            temporary := 0
            badIndices := 0
            values := 0
            indices := 0
            destination := 0
            piecewise := 0
            pieceSource := 0
            selected := 0
            choiceTwo := 0
            choiceOne := 0
            conditionTwo := 0
            conditionOne := 0
            vectorized := 0
            pyfuncResult := 0
            iterated := 0
            generated := 0
            appliedOver := 0
            applied := 0
            source := 0
        }
        AhkTest.AssertEqual(0, Numpy.CallbackContexts.Count,
            "functional callback contexts must all be released")
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "functional callback facade must restore retained native bytes")
        Numpy.Cleanup()
    }

    static TestBulkCallbackV2AbiSemantics() {
        Numpy.DllPath := NumpyDllPath()
        Numpy.Init()
        memoryBefore := Numpy.AllocatedMemory()
        source := 0
        result := 0
        empty := 0
        emptyResult := 0
        iterated := 0
        unaryNative := 0
        iteratorNative := 0
        successToken := 0
        failureToken := 0
        iteratorToken := 0
        try {
            values := []
            loop 600
                values.Push(A_Index - 1.0)
            source := Numpy.Array(values, [20, 30])

            successState := {Calls: 0, Elements: 0, Fail: false}
            successToken := NumpyBulkCallbackProbe.Register(successState)
            unaryNative := CallbackCreate(NumpyBulkUnaryProbe, "C", 6)
            handle := DllCall(Numpy.Proc("cnp_ahk_vectorize_v2"),
                "Ptr", unaryNative, "Ptr", successToken,
                "Ptr", source.Handle, "Ptr")
            result := Numpy.WrapHandle(
                handle, "Numpy bulk callback probe (cnp_ahk_vectorize_v2)")
            AhkTest.AssertEqual(3, successState.Calls,
                "600 elements must cross the native boundary in three batches")
            AhkTest.AssertEqual(600, successState.Elements,
                "batch callback element count must remain exact")
            resultValues := result.ToArray()
            AhkTest.AssertEqual(0.25, resultValues[1])
            AhkTest.AssertEqual(898.75, resultValues[600])

            empty := Numpy.Zeros([0])
            callsBeforeEmpty := successState.Calls
            handle := DllCall(Numpy.Proc("cnp_ahk_frompyfunc_v2"),
                "Ptr", unaryNative, "Ptr", successToken,
                "Ptr", empty.Handle, "Ptr")
            emptyResult := Numpy.WrapHandle(
                handle, "Numpy empty bulk callback probe")
            AhkTest.AssertEqual(0, emptyResult.Size)
            AhkTest.AssertEqual(callsBeforeEmpty, successState.Calls,
                "empty input must not invoke the callback")

            iteratorState := {
                Calls: 0,
                Index: 0,
                Values: [1.5, -2.25, 7.0]
            }
            iteratorToken := NumpyBulkCallbackProbe.Register(iteratorState)
            iteratorNative := CallbackCreate(NumpyBulkIteratorProbe, "C", 4)
            handle := DllCall(Numpy.Proc("cnp_ahk_fromiter_v2"),
                "Ptr", iteratorNative, "Ptr", iteratorToken,
                "Int64", -1, "Int", Numpy.DT_FLOAT64, "Ptr")
            iterated := Numpy.WrapHandle(
                handle, "Numpy partial bulk iterator probe")
            NumpyAssertArrayEqual(iteratorState.Values, iterated.ToArray())
            AhkTest.AssertEqual(1, iteratorState.Calls,
                "partial production must terminate unknown-count iteration")

            failureState := {Calls: 0, Elements: 0, Fail: true}
            failureToken := NumpyBulkCallbackProbe.Register(failureState)
            rejected := false
            try {
                handle := DllCall(Numpy.Proc("cnp_ahk_vectorize_v2"),
                    "Ptr", unaryNative, "Ptr", failureToken,
                    "Ptr", source.Handle, "Ptr")
                result := Numpy.WrapHandle(
                    handle,
                    "bulk failure probe (cnp_ahk_vectorize_v2)")
            } catch Error as err {
                rejected := InStr(err.Message, "cnp_ahk_vectorize_v2") > 0
            }
            AhkTest.AssertTrue(rejected,
                "bulk callback status must remain a visible native failure")
            AhkTest.AssertEqual(1, failureState.Calls,
                "bulk callback failure must not retry through a scalar ABI")
        } finally {
            iterated := 0
            emptyResult := 0
            empty := 0
            result := 0
            source := 0
            if iteratorNative
                CallbackFree(iteratorNative)
            if unaryNative
                CallbackFree(unaryNative)
            if iteratorToken
                NumpyBulkCallbackProbe.Release(iteratorToken)
            if failureToken
                NumpyBulkCallbackProbe.Release(failureToken)
            if successToken
                NumpyBulkCallbackProbe.Release(successToken)
        }
        AhkTest.AssertEqual(0, NumpyBulkCallbackProbe.States.Count,
            "bulk probe registrations must all be released")
        AhkTest.AssertEqual(memoryBefore, Numpy.AllocatedMemory(),
            "bulk callback ABI must restore retained native bytes")
        Numpy.Cleanup()
    }

    static TestToString() {
        dllPath := NumpyLoadDll()
        DllCall(dllPath "\cnp_ahk_init", "Int")

        shapeBuf := Buffer(8, 0)
        NumPut("Int64", 3, shapeBuf, 0)
        dataBuf := Buffer(24, 0)
        NumPut("Double", 1.5, dataBuf, 0)
        NumPut("Double", 2.5, dataBuf, 8)
        NumPut("Double", 3.5, dataBuf, 16)
        h := DllCall(dllPath "\cnp_ahk_from_doubles", "Int", 1, "Ptr", shapeBuf, "Ptr", dataBuf, "Int64", 3, "Ptr")

        strBuf := Buffer(4096, 0)
        DllCall(dllPath "\cnp_ahk_to_string", "Ptr", h, "Ptr", strBuf, "Int64", 4096, "Int")
        str := StrGet(strBuf, "UTF-8")
        AhkTest.AssertTrue(StrLen(str) > 0, "String representation should not be empty")
        AhkTest.AssertTrue(InStr(str, "1.5") > 0, "Should contain 1.5")

        DllCall(dllPath "\cnp_ahk_free", "Ptr", h, "Int")
        DllCall(dllPath "\cnp_ahk_cleanup", "Int")
    }
}

testReportDirectory := A_ScriptDir "\..\build\test-results"
DirCreate testReportDirectory
AhkTest.SetOutputFile(testReportDirectory "\numpy.test.log")
AhkTest.Collect(NumpyFoundationTest)
suiteResult := AhkTest.Run()
ExitApp suiteResult.ExitCode
