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

SumLine(values) {
    total := 0.0
    for value in values
        total += value
    return total
}

class SequenceIterator {
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

class FailingCallback {
    Call(arguments*) {
        throw Error("callback example failure")
    }
}

Main() {
    Numpy.DllPath := A_ScriptDir "\..\..\build\x64\Release\cnumpy_ahk.dll"
    Numpy.Init()
    baseline := Numpy.AllocatedMemory()

    source := 0
    applied := 0
    generated := 0
    iterated := 0
    vectorized := 0
    conditionOne := 0
    conditionTwo := 0
    choiceOne := 0
    choiceTwo := 0
    selected := 0
    temporary := 0
    retained := -1
    try {
        source := Numpy.Array([1, 2, 3, 4, 5, 6], [2, 3])
        applied := Numpy.ApplyAlongAxis(SumLine, 0, source)
        generated := Numpy.FromFunction(
            (row, column) => row * 10 + column + 0.5, [2, 3])

        iterator := SequenceIterator([1.9, -2.2, 7.0, 8.0])
        iterated := Numpy.FromIter(iterator, 4, Numpy.DT_INT16)
        vectorized := Numpy.Vectorize(value => value * 2 + 1, source)

        conditionOne := Numpy.IntArray([1, 0, 0])
        conditionTwo := Numpy.IntArray([0, 1, 0])
        choiceOne := Numpy.Array([10, 10, 10])
        choiceTwo := Numpy.Array([20, 20, 20])
        selected := Numpy.Select(
            [conditionOne, conditionTwo], [choiceOne, choiceTwo], -1.0)

        PrintArray("apply_along_axis", applied)
        PrintArray("fromfunction", generated)
        PrintArray("fromiter_int16", iterated)
        PrintArray("vectorize", vectorized)
        PrintArray("select", selected)

        callbackMessage := ""
        try temporary := Numpy.Vectorize(FailingCallback(), source)
        catch Error as err
            callbackMessage := err.Message
        temporary := 0
        if callbackMessage != "callback example failure"
            throw Error("callback exception identity was not preserved: "
                callbackMessage)
        if Numpy.CallbackContexts.Count != 0
            throw Error("callback context remained registered after failure")
        FileAppend("expected_callback_error=" callbackMessage "`n", "*")
    } finally {
        temporary := 0
        selected := 0
        choiceTwo := 0
        choiceOne := 0
        conditionTwo := 0
        conditionOne := 0
        vectorized := 0
        iterated := 0
        generated := 0
        applied := 0
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
