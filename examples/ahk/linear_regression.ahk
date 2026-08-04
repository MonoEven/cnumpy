#Requires AutoHotkey v2.0
#Include ..\..\ahk\numpy.ahk

EnsureParentDirectory(path) {
    SplitPath(path, , &directory)
    if directory != ""
        DirCreate(directory)
}

Main() {
    inputPath := A_Args.Length >= 1
        ? A_Args[1] : A_ScriptDir "\..\data\regression_points.csv"
    outputPath := A_Args.Length >= 2
        ? A_Args[2]
        : A_ScriptDir "\..\..\build\examples\linear_regression_predictions.csv"

    Numpy.DllPath := A_ScriptDir "\..\..\build\x64\Release\cnumpy_ahk.dll"
    Numpy.Init()
    baseline := Numpy.AllocatedMemory()

    data := 0
    x := 0
    observed := 0
    interceptColumn := 0
    design := 0
    fit := 0
    coefficients := 0
    residualSums := 0
    singularValues := 0
    predictions := 0
    errors := 0
    squaredErrors := 0
    meanVector := 0
    centered := 0
    squaredCentered := 0
    report := 0
    retained := -1
    try {
        data := Numpy.Loadtxt(inputPath, ",", Numpy.DT_FLOAT64)
        if data.Ndim != 2 || data.Shape[2] != 2
            throw ValueError("regression input must have two numeric columns")

        x := Numpy.Slice(data, 0, 1, 1, 1).Ravel()
        observed := Numpy.Slice(data, 1, 2, 1, 1).Ravel()
        interceptColumn := Numpy.Ones([x.Size], Numpy.DT_FLOAT64)
        design := Numpy.ColumnStack([interceptColumn, x])

        fit := Numpy.Lstsq(design, observed)
        coefficients := fit[1]
        residualSums := fit[2]
        rank := fit[3]
        singularValues := fit[4]
        predictions := Numpy.Matmul(design, coefficients)

        errors := Numpy.Subtract(observed, predictions)
        squaredErrors := Numpy.Multiply(errors, errors)
        meanVector := Numpy.FullLike(observed, observed.Mean())
        centered := Numpy.Subtract(observed, meanVector)
        squaredCentered := Numpy.Multiply(centered, centered)
        rSquared := 1.0 - squaredErrors.Sum() / squaredCentered.Sum()

        report := Numpy.ColumnStack([x, observed, predictions])
        EnsureParentDirectory(outputPath)
        Numpy.Savetxt(outputPath, report, ",", "%.6f")

        FileAppend("observations=" data.Shape[1] "`n", "*")
        FileAppend("intercept=" Format("{:.6f}", coefficients.GetItem(0)) "`n", "*")
        FileAppend("slope=" Format("{:.6f}", coefficients.GetItem(1)) "`n", "*")
        FileAppend("rank=" rank "`n", "*")
        FileAppend("residual_sum=" Format("{:.6f}", residualSums.GetItem(0)) "`n", "*")
        FileAppend("r_squared=" Format("{:.6f}", rSquared) "`n", "*")
        FileAppend("output=" outputPath "`n", "*")
    } finally {
        report := 0
        squaredCentered := 0
        centered := 0
        meanVector := 0
        squaredErrors := 0
        errors := 0
        predictions := 0
        singularValues := 0
        residualSums := 0
        coefficients := 0
        fit := 0
        design := 0
        interceptColumn := 0
        observed := 0
        x := 0
        data := 0
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
