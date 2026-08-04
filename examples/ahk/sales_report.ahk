#Requires AutoHotkey v2.0
#Include ..\..\ahk\numpy.ahk

EnsureParentDirectory(path) {
    SplitPath(path, , &directory)
    if directory != ""
        DirCreate(directory)
}

Main() {
    inputPath := A_Args.Length >= 1
        ? A_Args[1] : A_ScriptDir "\..\data\sales.csv"
    outputPath := A_Args.Length >= 2
        ? A_Args[2] : A_ScriptDir "\..\..\build\examples\sales_report.csv"

    Numpy.DllPath := A_ScriptDir "\..\..\build\x64\Release\cnumpy_ahk.dll"
    Numpy.Init()
    baseline := Numpy.AllocatedMemory()

    data := 0
    orderIds := 0
    quantities := 0
    unitPrices := 0
    unitCosts := 0
    unitProfit := 0
    revenue := 0
    profit := 0
    report := 0
    retained := -1
    try {
        data := Numpy.Loadtxt(inputPath, ",", Numpy.DT_FLOAT64)
        if data.Ndim != 2 || data.Shape[2] != 4
            throw ValueError("sales input must have four numeric columns")

        orderIds := Numpy.Slice(data, 0, 1, 1, 1).Ravel()
        quantities := Numpy.Slice(data, 1, 2, 1, 1).Ravel()
        unitPrices := Numpy.Slice(data, 2, 3, 1, 1).Ravel()
        unitCosts := Numpy.Slice(data, 3, 4, 1, 1).Ravel()

        revenue := Numpy.Multiply(quantities, unitPrices)
        unitProfit := Numpy.Subtract(unitPrices, unitCosts)
        profit := Numpy.Multiply(quantities, unitProfit)
        report := Numpy.ColumnStack([orderIds, revenue, profit])

        totalRevenue := revenue.Sum()
        totalProfit := profit.Sum()
        profitMargin := totalProfit / totalRevenue * 100.0
        bestIndex := profit.Argmax()
        bestOrderId := orderIds.GetItem(bestIndex)

        EnsureParentDirectory(outputPath)
        Numpy.Savetxt(outputPath, report, ",", "%.2f")

        FileAppend("orders=" data.Shape[1] "`n", "*")
        FileAppend("total_revenue=" Format("{:.2f}", totalRevenue) "`n", "*")
        FileAppend("total_profit=" Format("{:.2f}", totalProfit) "`n", "*")
        FileAppend("profit_margin=" Format("{:.2f}", profitMargin) "%`n", "*")
        FileAppend("best_order_id=" Format("{:.0f}", bestOrderId) "`n", "*")
        FileAppend("output=" outputPath "`n", "*")
    } finally {
        report := 0
        profit := 0
        revenue := 0
        unitProfit := 0
        unitCosts := 0
        unitPrices := 0
        quantities := 0
        orderIds := 0
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
