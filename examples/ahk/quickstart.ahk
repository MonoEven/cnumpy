#Requires AutoHotkey v2.0
#Include ..\..\ahk\numpy.ahk

JoinValues(values) {
    text := ""
    for index, value in values
        text .= (index = 1 ? "" : ", ") value
    return text
}

PrintArray(label, array) {
    FileAppend(label " shape=[" JoinValues(array.Shape) "] values=["
        JoinValues(array.ToArray()) "]`n", "*")
}

Main() {
    Numpy.DllPath := A_ScriptDir "\..\..\build\x64\Release\cnumpy_ahk.dll"
    Numpy.Init()
    baseline := Numpy.AllocatedMemory()

    source := 0
    offsets := 0
    shifted := 0
    rowSums := 0
    invalid := 0
    retained := -1
    try {
        source := Numpy.Array([1, 2, 3, 4, 5, 6], [2, 3])
        offsets := Numpy.Array([10, 20, 30], [1, 3])
        shifted := Numpy.Add(source, offsets)
        rowSums := Numpy.Sum(shifted, 1)

        FileAppend("cnumpy " Numpy.Version() "`n", "*")
        FileAppend("dtype_id=" shifted.Dtype
            " c_contiguous=" shifted.CContiguous "`n", "*")
        PrintArray("shifted", shifted)
        PrintArray("row_sums", rowSums)

        nativeMessage := ""
        try invalid := source.Reshape([4, 2])
        catch Error as err
            nativeMessage := err.Message
        if nativeMessage = ""
            throw Error("incompatible reshape did not fail")
        FileAppend("expected_native_error=" nativeMessage "`n", "*")
    } finally {
        invalid := 0
        rowSums := 0
        shifted := 0
        offsets := 0
        source := 0
        retained := Numpy.AllocatedMemory()
        Numpy.Cleanup()
    }

    if retained != baseline
        throw Error("retained native bytes: baseline=" baseline
            " final=" retained)
    FileAppend("retained_bytes=" (retained - baseline) "`n", "*")
}

try Main()
catch Error as err {
    FileAppend("ERROR: " err.Message "`n", "**")
    ExitApp 1
}
ExitApp 0
