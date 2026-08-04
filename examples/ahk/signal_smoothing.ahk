#Requires AutoHotkey v2.0
#Include ..\..\ahk\numpy.ahk

EnsureParentDirectory(path) {
    SplitPath(path, , &directory)
    if directory != ""
        DirCreate(directory)
}

Main() {
    inputPath := A_Args.Length >= 1
        ? A_Args[1] : A_ScriptDir "\..\data\sensor_readings.csv"
    outputPath := A_Args.Length >= 2
        ? A_Args[2] : A_ScriptDir "\..\..\build\examples\signal_smoothing.csv"

    Numpy.DllPath := A_ScriptDir "\..\..\build\x64\Release\cnumpy_ahk.dll"
    Numpy.Init()
    baseline := Numpy.AllocatedMemory()

    loaded := 0
    signal := 0
    kernel := 0
    movingAverage := 0
    alignedSignal := 0
    residual := 0
    absoluteResidual := 0
    indices := 0
    report := 0
    retained := -1
    try {
        loaded := Numpy.Loadtxt(inputPath, ",", Numpy.DT_FLOAT64)
        signal := loaded.Ravel()
        if signal.Size < 3
            throw ValueError("signal input requires at least three samples")

        kernel := Numpy.Full([3], 1.0 / 3.0, Numpy.DT_FLOAT64)
        movingAverage := Numpy.Convolve(signal, kernel, "valid")
        alignedSignal := Numpy.Slice(signal, 1, signal.Size - 1)
        residual := Numpy.Subtract(alignedSignal, movingAverage)
        absoluteResidual := Numpy.Absolute(residual)

        peakOffset := absoluteResidual.Argmax()
        peakIndex := peakOffset + 1
        indices := Numpy.Arange(1, signal.Size - 1, 1, Numpy.DT_FLOAT64)
        report := Numpy.ColumnStack(
            [indices, alignedSignal, movingAverage, absoluteResidual])

        EnsureParentDirectory(outputPath)
        Numpy.Savetxt(outputPath, report, ",", "%.6f")

        FileAppend("samples=" signal.Size "`n", "*")
        FileAppend("smoothed_samples=" movingAverage.Size "`n", "*")
        FileAppend("spike_index=" Format("{:.0f}", peakIndex) "`n", "*")
        FileAppend("spike_observed=" Format("{:.2f}",
            alignedSignal.GetItem(peakOffset)) "`n", "*")
        FileAppend("spike_moving_average=" Format("{:.2f}",
            movingAverage.GetItem(peakOffset)) "`n", "*")
        FileAppend("spike_residual=" Format("{:.2f}",
            absoluteResidual.GetItem(peakOffset)) "`n", "*")
        FileAppend("output=" outputPath "`n", "*")
    } finally {
        report := 0
        indices := 0
        absoluteResidual := 0
        residual := 0
        alignedSignal := 0
        movingAverage := 0
        kernel := 0
        signal := 0
        loaded := 0
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
