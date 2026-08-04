#Requires AutoHotkey v2.0
#SingleInstance Force
#Include ahk\numpy.ahk

EnsureParentDirectory(path) {
    SplitPath(path, , &directory)
    if directory != ""
        DirCreate(directory)
}

Main(headless) {
    argumentOffset := headless ? 1 : 0
    inputPath := A_Args.Length >= argumentOffset + 1
        ? A_Args[argumentOffset + 1]
        : A_ScriptDir "\examples\data\sales.csv"
    outputPath := A_Args.Length >= argumentOffset + 2
        ? A_Args[argumentOffset + 2]
        : A_ScriptDir "\build\examples\forum_sales_report.csv"

    Numpy.DllPath := A_ScriptDir "\build\x64\Release\cnumpy_ahk.dll"
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
    summary := ""
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

        summary := "orders=" data.Shape[1] "`n"
            . "total_revenue=" Format("{:.2f}", totalRevenue) "`n"
            . "total_profit=" Format("{:.2f}", totalProfit) "`n"
            . "profit_margin=" Format("{:.2f}", profitMargin) "%`n"
            . "best_order_id=" Format("{:.0f}", bestOrderId) "`n"
            . "output=" outputPath "`n"
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
    summary .= "retained_bytes=" (retained - baseline) "`n"

    if headless
        FileAppend(summary, "*")
    else
        MsgBox(summary, "cnumpy sales report")
}

headless := A_Args.Length >= 1 && A_Args[1] = "--headless"
try Main(headless)
catch Error as err {
    message := "ERROR: " err.Message
    if headless
        FileAppend(message "`n", "**")
    else
        MsgBox(message, "cnumpy sales report", "Iconx")
    ExitApp 1
}
ExitApp 0
