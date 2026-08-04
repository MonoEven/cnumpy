#Requires AutoHotkey v2.0
#SingleInstance Off
#Include ..\ahk\numpy.ahk

FormatError(err) {
    text := err.Message
    if err.What != ""
        text .= " [" err.What "]"
    if err.Extra != ""
        text .= " (" err.Extra ")"
    if err.File != ""
        text .= " at " err.File ":" err.Line
    if err.Stack != ""
        text .= "`n" err.Stack
    return text
}

BenchmarkUnaryScalarCallback(valuePointer, userdata, resultPointer) {
    invocations := NumGet(userdata, 0, "Int64") + 1
    logicalElements := NumGet(userdata, 8, "Int64") + 1
    NumPut("Int64", invocations, userdata, 0)
    NumPut("Int64", logicalElements, userdata, 8)
    value := NumGet(valuePointer, 0, "Double")
    NumPut("Double", value * 1.5 + 0.25, resultPointer, 0)
    return 0
}

BenchmarkUnaryBatchCallback(
    valuesPointer, valueCount, resultsPointer,
    resultCapacity, producedPointer, userdata
) {
    if resultCapacity != valueCount
        return -13
    invocations := NumGet(userdata, 0, "Int64") + 1
    logicalElements := NumGet(userdata, 8, "Int64") + valueCount
    NumPut("Int64", invocations, userdata, 0)
    NumPut("Int64", logicalElements, userdata, 8)
    loop valueCount {
        offset := (A_Index - 1) * 8
        value := NumGet(valuesPointer, offset, "Double")
        NumPut("Double", value * 1.5 + 0.25, resultsPointer, offset)
    }
    NumPut("Int64", valueCount, producedPointer, 0)
    return 0
}

BenchmarkLineScalarCallback(
    linePointer, lineLength, userdata, resultPointer
) {
    NumPut("Int64", NumGet(userdata, 0, "Int64") + 1, userdata, 0)
    NumPut("Int64", NumGet(userdata, 8, "Int64") + 1, userdata, 8)
    total := BenchmarkLineSum(linePointer, lineLength, 0)
    NumPut("Double", total, resultPointer, 0)
    return 0
}

BenchmarkLineSum(linePointer, lineLength, lineOffset) {
    total := 0.0
    loop lineLength
        total += NumGet(
            linePointer, (lineOffset + A_Index - 1) * 8, "Double")
    return total
}

BenchmarkLineBatchCallback(
    linesPointer, lineCount, lineLength, resultsPointer,
    resultCapacity, producedPointer, userdata
) {
    if resultCapacity != lineCount
        return -13
    NumPut("Int64", NumGet(userdata, 0, "Int64") + 1, userdata, 0)
    NumPut("Int64", NumGet(userdata, 8, "Int64") + lineCount,
        userdata, 8)
    loop lineCount {
        lineIndex := A_Index - 1
        total := BenchmarkLineSum(
            linesPointer, lineLength, lineIndex * lineLength)
        NumPut("Double", total, resultsPointer, lineIndex * 8)
    }
    NumPut("Int64", lineCount, producedPointer, 0)
    return 0
}

BenchmarkCoordinateScalarCallback(
    coordinatesPointer, ndim, userdata, resultPointer
) {
    NumPut("Int64", NumGet(userdata, 0, "Int64") + 1, userdata, 0)
    NumPut("Int64", NumGet(userdata, 8, "Int64") + 1, userdata, 8)
    value := 0.0
    loop ndim
        value += A_Index * NumGet(
            coordinatesPointer, (A_Index - 1) * 8, "Int64")
    NumPut("Double", value, resultPointer, 0)
    return 0
}

BenchmarkCoordinateBatchCallback(
    coordinatesPointer, pointCount, ndim, resultsPointer,
    resultCapacity, producedPointer, userdata
) {
    if resultCapacity != pointCount
        return -13
    NumPut("Int64", NumGet(userdata, 0, "Int64") + 1, userdata, 0)
    NumPut("Int64", NumGet(userdata, 8, "Int64") + pointCount,
        userdata, 8)
    loop pointCount {
        coordinateOffset := (A_Index - 1) * ndim
        value := 0.0
        loop ndim
            value += A_Index * NumGet(
                coordinatesPointer,
                (coordinateOffset + A_Index - 1) * 8, "Int64")
        NumPut("Double", value, resultsPointer, (A_Index - 1) * 8)
    }
    NumPut("Int64", pointCount, producedPointer, 0)
    return 0
}

BenchmarkIteratorScalarCallback(userdata, resultPointer) {
    NumPut("Int64", NumGet(userdata, 0, "Int64") + 1, userdata, 0)
    NumPut("Int64", NumGet(userdata, 8, "Int64") + 1, userdata, 8)
    index := NumGet(userdata, 16, "Int64")
    NumPut("Double", index * 0.25 - 3.0, resultPointer, 0)
    NumPut("Int64", index + 1, userdata, 16)
    return 0
}

BenchmarkIteratorBatchCallback(
    resultsPointer, resultCapacity, producedPointer, userdata
) {
    NumPut("Int64", NumGet(userdata, 0, "Int64") + 1, userdata, 0)
    NumPut("Int64", NumGet(userdata, 8, "Int64") + resultCapacity,
        userdata, 8)
    first := NumGet(userdata, 16, "Int64")
    loop resultCapacity
        NumPut("Double", (first + A_Index - 1) * 0.25 - 3.0,
            resultsPointer, (A_Index - 1) * 8)
    NumPut("Int64", first + resultCapacity, userdata, 16)
    NumPut("Int64", resultCapacity, producedPointer, 0)
    return 0
}

ParseArgs(args) {
    if !(args is Array)
        throw TypeError("ParseArgs requires an Array")

    parsed := Map(
        "warmups", 5,
        "samples", 15,
        "target_sample_ms", 20.0,
        "seed", 12345
    )
    optionKeys := Map(
        "--jobs", "jobs",
        "--output", "output",
        "--dll", "dll",
        "--warmups", "warmups",
        "--samples", "samples",
        "--target-sample-ms", "target_sample_ms",
        "--seed", "seed"
    )
    seen := Map()
    index := 1
    while index <= args.Length {
        option := args[index]
        if !optionKeys.Has(option)
            throw ValueError("unknown option: " option)
        if seen.Has(option)
            throw ValueError("duplicate option: " option)
        if index = args.Length || RegExMatch(args[index + 1], "^--")
            throw ValueError("option requires a value: " option)
        seen[option] := true
        key := optionKeys[option]
        value := args[index + 1]
        switch option {
            case "--warmups":
                parsed[key] := ParseStrictUnsigned(value, "--warmups", true)
            case "--samples":
                sampleCount := ParseStrictUnsigned(value, "--samples", false)
                if Mod(sampleCount, 2) = 0
                    throw ValueError("--samples must be a positive odd integer")
                parsed[key] := sampleCount
            case "--target-sample-ms":
                parsed[key] := ParseStrictPositiveNumber(value, "--target-sample-ms")
            case "--seed":
                seed := ParseStrictUnsigned(value, "--seed", true)
                if seed > 2147483647
                    throw ValueError("--seed must be in the signed 32-bit range [0, 2147483647]")
                parsed[key] := seed
            default:
                parsed[key] := value
        }
        index += 2
    }

    for option in ["--jobs", "--output", "--dll"] {
        if !seen.Has(option)
            throw ValueError("required option missing: " option)
    }
    return parsed
}

ParseStrictUnsigned(text, context, allowZero) {
    if Type(text) != "String" || !RegExMatch(text, "^(0|[1-9][0-9]*)$")
        throw ValueError(context " must be a strict decimal integer")
    value := text + 0
    if Type(value) != "Integer"
        throw ValueError(context " is outside the signed 64-bit integer range")
    if !allowZero && value = 0
        throw ValueError(context " must be positive")
    return value
}

ParseStrictSignedInteger(text, context) {
    if Type(text) != "String" || !RegExMatch(text, "^-?(0|[1-9][0-9]*)$")
        throw ValueError(context " must be a strict decimal integer")
    value := text + 0
    if Type(value) != "Integer"
        throw ValueError(context " is outside the signed 64-bit integer range")
    return value
}

ParseStrictPositiveNumber(text, context) {
    pattern := "^[+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$"
    if Type(text) != "String" || !RegExMatch(text, pattern)
        throw ValueError(context " must be a finite positive number")
    value := text + 0.0
    if !IsFiniteNumber(value) || value <= 0
        throw ValueError(context " must be a finite positive number")
    return value
}

IsFiniteNumber(value) {
    if Type(value) != "Integer" && Type(value) != "Float"
        return false
    return DllCall("msvcrt\_finite", "Double", value, "Int") != 0
}

ParseJobs(path) {
    if Type(path) != "String" || path = ""
        throw ValueError("jobs path must be a non-empty string")
    text := FileRead(path, "UTF-8")
    if InStr(text, "`r")
        throw ValueError("jobs TSV must use LF-only line endings")
    if text = ""
        throw ValueError("jobs TSV is empty")

    lines := StrSplit(text, "`n")
    if lines[lines.Length] = ""
        lines.Pop()
    expectedHeader := "id`tcategory`toperation`tdtype`tsize`trows`tcols`taxis"
    if lines.Length = 0 || lines[1] != expectedHeader
        throw ValueError("jobs TSV header must be exactly: " expectedHeader)

    cases := []
    ids := Map()
    loop lines.Length - 1 {
        rowNumber := A_Index + 1
        fields := StrSplit(lines[rowNumber], "`t")
        if fields.Length != 8
            throw ValueError("jobs TSV row " rowNumber " must have exactly 8 fields")
        for fieldIndex, field in fields {
            if field = "" || InStr(field, Chr(34)) || InStr(field, "`r") || InStr(field, "`n")
                throw ValueError("jobs TSV row " rowNumber " field " fieldIndex " is empty or contains a forbidden character")
        }

        caseData := Map(
            "id", fields[1],
            "category", fields[2],
            "operation", fields[3],
            "dtype", fields[4],
            "size", ParseStrictUnsigned(fields[5], "jobs TSV row " rowNumber " size", true),
            "rows", ParseStrictUnsigned(fields[6], "jobs TSV row " rowNumber " rows", true),
            "cols", ParseStrictUnsigned(fields[7], "jobs TSV row " rowNumber " cols", true),
            "axis", ParseStrictSignedInteger(fields[8], "jobs TSV row " rowNumber " axis")
        )
        if ids.Has(caseData["id"])
            throw ValueError("duplicate jobs TSV id: " caseData["id"])
        ValidateJobCase(caseData)
        ids[caseData["id"]] := true
        cases.Push(caseData)
    }
    if cases.Length = 0
        throw ValueError("jobs TSV must contain at least one benchmark case")
    return cases
}

OperationContract(operation) {
    switch operation {
        case "zeros", "ones", "arange", "random", "linspace":
            return ["creation", "vector", -1]
        case "choice_weighted":
            return ["random", "vector", -1]
        case "sin", "cos", "exp", "expm1", "sqrt", "log", "log2", "log10", "log1p", "absolute", "floor", "tanh", "angle", "real", "imag", "real_if_close":
            return ["unary", "vector", -1]
        case "add", "subtract", "multiply", "divide", "divmod", "power", "float_power", "heaviside", "maximum", "minimum", "fmax", "fmin", "logaddexp", "logaddexp2":
            return ["binary", "vector", -1]
        case "logical_and", "logical_or", "logical_xor", "logical_not", "isnan", "isinf", "isfinite", "signbit", "iscomplexobj", "isrealobj", "isscalar":
            return ["logical", "vector", -1]
        case "bitwise_and", "bitwise_or", "bitwise_xor", "invert", "left_shift", "right_shift":
            return ["bitwise", "vector", -1]
        case "gcd", "lcm":
            return ["integer", "vector", -1]
        case "convolve", "correlate":
            return ["signal", "vector", -1]
        case "allclose", "equal":
            return ["comparison", "vector", -1]
        case "sum", "mean", "average", "std", "max", "min", "argmax", "cumsum", "prod":
            return ["reduction", "vector", -1]
        case "sum_axis_last", "cumsum_axis_last":
            return ["reduction", "matrix", 1]
        case "softmax", "log_softmax":
            return ["misc_axis", "vector", -1]
        case "softmax_axis_last", "log_softmax_axis_last":
            return ["misc_axis", "matrix", 1]
        case "softmax_axis0_strided", "log_softmax_axis0_strided":
            return ["misc_axis", "matrix", 0]
        case "trapz":
            return ["misc_axis", "vector", -1]
        case "trapz_axis_last":
            return ["misc_axis", "matrix", 1]
        case "trapz_axis0_strided":
            return ["misc_axis", "matrix", 0]
        case "packbits":
            return ["misc_axis", "vector", -1]
        case "packbits_axis_last":
            return ["misc_axis", "matrix", 1]
        case "packbits_axis0_strided":
            return ["misc_axis", "matrix", 0]
        case "unpackbits":
            return ["misc_axis", "vector", -1]
        case "unpackbits_axis_last":
            return ["misc_axis", "matrix", 1]
        case "unpackbits_axis0_strided":
            return ["misc_axis", "matrix", 0]
        case "matmul", "dot", "det", "inv", "norm", "solve", "cholesky",
                "einsum", "eig", "svd", "lstsq":
            return ["linalg", "matrix", -1]
        case "sort", "argsort",
                "sort_mergesort", "argsort_mergesort",
                "sort_heapsort", "argsort_heapsort",
                "sort_stable", "argsort_stable",
                "sort_stable_nan", "argsort_stable_nan",
                "partition", "argpartition",
                "partition_nan", "argpartition_nan",
                "searchsorted", "searchsorted_right",
                "digitize", "digitize_decreasing", "lexsort":
            return ["sorting", "vector", -1]
        case "msort", "sort_complex":
            return ["sorting", "vector", -1]
        case "unique_duplicates", "unique_nan",
                "intersect1d_duplicates", "union1d_duplicates",
                "setdiff1d_duplicates", "setxor1d_duplicates",
                "in1d_duplicates", "isin_duplicates":
            return ["set", "vector", -1]
        case "copy", "reshape", "flatten",
                "atleast_1d", "atleast_2d", "atleast_3d":
            return ["shape", "vector", -1]
        case "take", "compress":
            return ["indexing", "vector", -1]
        case "take_axis0_block", "take_axis0_strided",
                "compress_axis0_block", "compress_axis0_strided":
            return ["indexing", "matrix", 0]
        case "add_into", "sqrt_into", "cumsum_into":
            return ["preallocated", "vector", -1]
        case "pipeline_separate", "pipeline_batch":
            return ["pipeline", "vector", -1]
        case "transpose_copy":
            return ["shape", "matrix", -1]
        case "concatenate":
            return ["shape", "matrix", 0]
        case "fft":
            return ["fft", "fft", -1]
        case "callback_apply_along_axis_legacy",
                "callback_apply_along_axis_v2":
            return ["functional_callback", "matrix", 1]
        case "callback_fromfunction_legacy", "callback_fromfunction_v2",
                "callback_fromiter_legacy", "callback_fromiter_v2",
                "callback_vectorize_legacy", "callback_vectorize_v2":
            return ["functional_callback", "vector", -1]
        case "property_call", "property_cached", "nbytes_cached",
                "c_contiguous_cached", "f_contiguous_cached",
                "static_add_call":
            return ["bridge", "bridge", -1]
        default:
            throw ValueError("unknown benchmark operation: " operation)
    }
}

SortingOperation(operation) {
    switch operation {
        case "sort":
            return ["sort", 0, "sorting"]
        case "argsort":
            return ["argsort", 0, "sorting"]
        case "sort_mergesort":
            return ["sort", 1, "sorting"]
        case "argsort_mergesort":
            return ["argsort", 1, "sorting"]
        case "sort_heapsort":
            return ["sort", 2, "sorting"]
        case "argsort_heapsort":
            return ["argsort", 2, "sorting"]
        case "sort_stable":
            return ["sort", 3, "sorting"]
        case "argsort_stable":
            return ["argsort", 3, "sorting"]
        case "sort_stable_nan":
            return ["sort", 3, "sorting_nan"]
        case "argsort_stable_nan":
            return ["argsort", 3, "sorting_nan"]
        case "partition":
            return ["partition", 0, "sorting"]
        case "argpartition":
            return ["argpartition", 0, "sorting"]
        case "partition_nan":
            return ["partition", 0, "sorting_nan"]
        case "argpartition_nan":
            return ["argpartition", 0, "sorting_nan"]
        case "searchsorted":
            return ["searchsorted", 0, "searchsorted"]
        case "searchsorted_right":
            return ["searchsorted", 1, "searchsorted"]
        case "digitize":
            return ["digitize", 0, "digitize"]
        case "digitize_decreasing":
            return ["digitize", 1, "digitize"]
        case "lexsort":
            return ["lexsort", 0, "sorting"]
        case "msort":
            return ["msort", 0, "sorting"]
        case "sort_complex":
            return ["sort_complex", 0, "sorting"]
        default:
            throw ValueError("unknown sorting benchmark operation: " operation)
    }
}

ValidateJobCase(caseData) {
    for key in ["id", "category", "operation", "dtype", "size", "rows", "cols", "axis"] {
        if !caseData.Has(key)
            throw ValueError("job case is missing field: " key)
    }
    operation := caseData["operation"]
    contract := OperationContract(operation)
    expectedDtype := InStr(operation, "packbits", true) != 0
        ? "u8" : ((contract[1] = "bitwise"
            || contract[1] = "integer") ? "i64" : "f64")
    if caseData["dtype"] != expectedDtype
        throw ValueError("job dtype must be " expectedDtype ": " caseData["id"])
    if caseData["category"] != contract[1]
        throw ValueError("job category does not match operation contract: " caseData["id"])
    if caseData["axis"] != contract[3]
        throw ValueError("job axis does not match operation contract: " caseData["id"])

    kind := contract[2]
    size := caseData["size"]
    rows := caseData["rows"]
    cols := caseData["cols"]
    if kind = "bridge" {
        if caseData["id"] != "bridge/" operation
                || size != 1 || rows != 0 || cols != 0
            throw ValueError("bridge job declaration is not canonical")
        return
    }
    if kind = "vector" || kind = "fft" {
        if size <= 0 || rows != 0 || cols != 0
            throw ValueError("vector/fft job dimensions are not canonical: " caseData["id"])
        expectedId := operation "/" expectedDtype "/" size
        if caseData["id"] != expectedId
            throw ValueError("vector/fft job id is not canonical: " caseData["id"])
        return
    }
    if rows <= 0 || cols <= 0 || rows != cols
        throw ValueError("matrix job must be non-empty and square: " caseData["id"])
    matrixSize := rows * cols
    if Type(matrixSize) != "Integer" || size != matrixSize
        throw ValueError("matrix job size must equal rows*cols: " caseData["id"])
    expectedId := operation "/" expectedDtype "/" rows "x" cols
    if caseData["axis"] != -1
        expectedId .= "/axis" caseData["axis"]
    if caseData["id"] != expectedId
        throw ValueError("matrix job id is not canonical: " caseData["id"])
}

QueryTimerFrequency() {
    counterBytes := Buffer(8, 0)
    if !DllCall("kernel32\QueryPerformanceFrequency", "Ptr", counterBytes.Ptr, "Int")
        throw OSError(A_LastError, "QueryPerformanceFrequency failed")
    frequency := NumGet(counterBytes, 0, "Int64")
    if frequency <= 0
        throw Error("QueryPerformanceFrequency returned a non-positive frequency")
    return frequency
}

QueryTimerCounter() {
    counterBytes := Buffer(8, 0)
    if !DllCall("kernel32\QueryPerformanceCounter", "Ptr", counterBytes.Ptr, "Int")
        throw OSError(A_LastError, "QueryPerformanceCounter failed")
    return NumGet(counterBytes, 0, "Int64")
}

TimeOperation(callback, warmups, sampleCount, targetSampleNs, frequency := 0) {
    if Type(warmups) != "Integer" || warmups < 0
        throw ValueError("warmups must be a non-negative integer")
    if Type(sampleCount) != "Integer" || sampleCount <= 0 || Mod(sampleCount, 2) = 0
        throw ValueError("sampleCount must be a positive odd integer")
    if !IsFiniteNumber(targetSampleNs) || targetSampleNs <= 0
        throw ValueError("targetSampleNs must be a finite positive number")
    if frequency = 0
        frequency := QueryTimerFrequency()
    if Type(frequency) != "Integer" || frequency <= 0
        throw ValueError("timer frequency must be a positive integer")

    loop warmups
        callback.Call()

    innerLoops := 1
    loop {
        elapsedNs := TimeBatch(callback, innerLoops, frequency)
        if elapsedNs >= targetSampleNs
            break
        innerLoops *= 2
    }

    samples := []
    loop sampleCount {
        elapsedNs := TimeBatch(callback, innerLoops, frequency)
        sample := elapsedNs / innerLoops
        if !IsFiniteNumber(sample) || sample <= 0
            throw Error("benchmark sample must be a finite positive ns/op value")
        samples.Push(sample)
    }
    return Map("inner_loops", innerLoops, "samples_ns", samples)
}

TimeBatch(callback, innerLoops, frequency) {
    started := QueryTimerCounter()
    loop innerLoops
        callback.Call()
    elapsedTicks := QueryTimerCounter() - started
    if elapsedTicks <= 0
        throw Error("QueryPerformanceCounter returned a non-positive elapsed duration")
    elapsedNs := elapsedTicks * 1000000000.0 / frequency
    if !IsFiniteNumber(elapsedNs) || elapsedNs <= 0
        throw Error("QueryPerformanceCounter produced an invalid duration")
    return elapsedNs
}

JsonEscape(value) {
    if Type(value) != "String"
        throw TypeError("JsonEscape requires a String")
    output := ""
    loop parse value {
        character := A_LoopField
        code := Ord(character)
        switch character {
            case Chr(34):
                output .= "\" Chr(34)
            case "\":
                output .= "\\"
            case "`b":
                output .= "\b"
            case "`f":
                output .= "\f"
            case "`n":
                output .= "\n"
            case "`r":
                output .= "\r"
            case "`t":
                output .= "\t"
            default:
                output .= code < 0x20 ? "\u" Format("{:04X}", code) : character
        }
    }
    return output
}

JsonString(value) {
    return Chr(34) JsonEscape(value) Chr(34)
}

JsonNumber(value) {
    if Type(value) = "Integer"
        return value ""
    if Type(value) != "Float" || !IsFiniteNumber(value)
        throw ValueError("JSON number must be finite")
    return Format("{:.17g}", value)
}

JsonBoolean(value) {
    return value ? "true" : "false"
}

JsonNumberArray(values) {
    parts := []
    for value in values
        parts.Push(JsonNumber(value))
    return "[" JoinStrings(parts, ",") "]"
}

JsonStringArray(values) {
    parts := []
    for value in values
        parts.Push(JsonString(value))
    return "[" JoinStrings(parts, ",") "]"
}

JoinStrings(values, separator) {
    output := ""
    for index, value in values
        output .= (index = 1 ? "" : separator) value
    return output
}

CreateExclusiveAtomicTemp(path) {
    if Type(path) != "String" || path = ""
        throw ValueError("output path must be a non-empty string")
    SplitPath(path, &fileName, &directory)
    if directory = ""
        directory := A_WorkingDir

    guidBytes := Buffer(16, 0)
    guidStatus := DllCall("ole32\CoCreateGuid", "Ptr", guidBytes.Ptr, "Int")
    if guidStatus != 0
        throw Error("CoCreateGuid failed with HRESULT " Format("0x{:08X}", guidStatus & 0xFFFFFFFF))
    token := ""
    loop 16
        token .= Format("{:02X}", NumGet(guidBytes, A_Index - 1, "UChar"))
    temporary := directory "\." fileName "." token ".tmp"

    fileHandle := DllCall("kernel32\CreateFileW",
        "Str", temporary,
        "UInt", 0x40000000,
        "UInt", 0,
        "Ptr", 0,
        "UInt", 1,
        "UInt", 0x80,
        "Ptr", 0,
        "Ptr")
    if fileHandle = -1
        throw OSError(A_LastError, "CreateFileW CREATE_NEW failed for atomic JSON temp: " temporary)

    stream := 0
    try {
        stream := FileOpen(fileHandle, "h", "UTF-8-RAW")
        if !IsObject(stream)
            throw Error("FileOpen failed to wrap exclusive atomic JSON temp handle")
    } catch Error as err {
        if !DllCall("kernel32\CloseHandle", "Ptr", fileHandle, "Int")
            throw OSError(A_LastError, "CloseHandle failed after FileOpen wrapper failure")
        if FileExist(temporary)
            FileDelete(temporary)
        throw err
    }
    return Map("path", temporary, "handle", fileHandle, "stream", stream)
}

CloseExclusiveAtomicTemp(tempFile, context) {
    tempFile["stream"] := 0
    if !DllCall("kernel32\CloseHandle", "Ptr", tempFile["handle"], "Int")
        throw OSError(A_LastError, "CloseHandle failed for " context)
    tempFile["handle"] := -1
}

WriteJsonAtomic(path, jsonText) {
    if Type(path) != "String" || path = ""
        throw ValueError("output path must be a non-empty string")
    if Type(jsonText) != "String"
        throw TypeError("JSON document must be a String")
    temporary := ""
    tempFile := 0
    stream := 0
    try {
        tempFile := CreateExclusiveAtomicTemp(path)
        temporary := tempFile["path"]
        stream := tempFile["stream"]
        stream.Write(jsonText "`n")
        if !DllCall("kernel32\FlushFileBuffers", "Ptr", stream.Handle, "Int")
            throw OSError(A_LastError, "FlushFileBuffers failed for temporary JSON file")
        stream := 0
        CloseExclusiveAtomicTemp(tempFile, "temporary JSON file")
        flags := 0x1 | 0x8
        if !DllCall("kernel32\MoveFileExW", "Str", temporary, "Str", path, "UInt", flags, "Int")
            throw OSError(A_LastError, "MoveFileExW failed while publishing JSON")
    } catch Error as err {
        stream := 0
        if IsObject(tempFile) && tempFile["handle"] != -1
            CloseExclusiveAtomicTemp(tempFile, "failed temporary JSON publication")
        if FileExist(temporary)
            FileDelete(temporary)
        throw err
    }
}

MakeShapeBuffer(dimensions) {
    if !(dimensions is Array) || dimensions.Length = 0
        throw ValueError("array shape must contain at least one dimension")
    shapeBytes := Buffer(dimensions.Length * 8, 0)
    for index, dimension in dimensions {
        if Type(dimension) != "Integer" || dimension <= 0
            throw ValueError("array shape dimensions must be positive integers")
        NumPut("Int64", dimension, shapeBytes, (index - 1) * 8)
    }
    return shapeBytes
}

VectorElement(kind, index) {
    switch kind {
        case "general":
            return Mod(index * 37 + 11, 1009) / 1009.0 + 0.01
        case "binary":
            return Mod(index * 53 + 19, 1013) / 1013.0 + 0.02
        case "sorting":
            return Mod(Mod(index * 48271 + 17, 65521), 4096) * 1.0
        case "set":
            return Mod(index, 256) * 1.0
        case "product":
            return 1.0 + (Mod(index * 37 + 11, 1009) - 504) * 1e-9
        case "absolute":
            return Mod(index * 37 + 11, 1009) / 1009.0 + 0.01 - 0.5
        case "floor":
            return (Mod(index * 37 + 11, 1009) / 1009.0 + 0.01) * 100.0
        default:
            throw ValueError("unknown deterministic vector kind: " kind)
    }
}

BitwiseElement(kind, index) {
    switch kind {
        case "left":
            return Mod(index * 37 + 11, 4096) - 2048
        case "right":
            return Mod(index * 53 + 19, 4096) - 2048
        case "shift":
            return Mod(index * 5 + 3, 8)
        default:
            throw ValueError("unknown deterministic bitwise vector kind: " kind)
    }
}

CreateIntVectorHandle(dllPath, size, kind) {
    if Type(size) != "Integer" || size <= 0
        throw ValueError("integer input vector size must be positive")
    dataBytes := Buffer(size * 8, 0)
    loop size {
        index := A_Index - 1
        NumPut("Int64", BitwiseElement(kind, index), dataBytes, index * 8)
    }
    shapeBytes := MakeShapeBuffer([size])
    handle := DllCall(dllPath "\cnp_ahk_from_ints",
        "Int", 1, "Ptr", shapeBytes.Ptr,
        "Ptr", dataBytes.Ptr, "Int64", size, "Ptr")
    if handle = 0
        throw Error("cnp_ahk_from_ints returned null for " kind " bitwise input")
    return handle
}

CreateVectorHandle(dllPath, size, kind, dimensions := 0) {
    if Type(size) != "Integer" || size <= 0
        throw ValueError("input vector size must be positive")
    dataBytes := Buffer(size * 8, 0)
    loop size {
        index := A_Index - 1
        if kind = "predicate" {
            offset := index * 8
            switch Mod(index, 8) {
                case 0: NumPut("UInt64", 0x7FF8000000000000, dataBytes, offset)
                case 1: NumPut("UInt64", 0x7FF0000000000000, dataBytes, offset)
                case 2: NumPut("UInt64", 0xFFF0000000000000, dataBytes, offset)
                case 3: NumPut("UInt64", 0x0000000000000000, dataBytes, offset)
                case 4: NumPut("UInt64", 0x8000000000000000, dataBytes, offset)
                case 5: NumPut("Double", 1.25, dataBytes, offset)
                case 6: NumPut("Double", -2.5, dataBytes, offset)
                case 7: NumPut("Double", 3.0, dataBytes, offset)
            }
        } else if (kind = "sorting_nan" || kind = "set_nan")
                && Mod(index, 2) = 0
            NumPut("UInt64", 0x7FF8000000000000, dataBytes, index * 8)
        else
            NumPut("Double", VectorElement(
                kind = "sorting_nan" ? "sorting"
                    : (kind = "set_nan" ? "set" : kind), index),
                dataBytes, index * 8)
    }
    if !IsObject(dimensions)
        dimensions := [size]
    shapeBytes := MakeShapeBuffer(dimensions)
    handle := DllCall(dllPath "\cnp_ahk_from_doubles",
        "Int", dimensions.Length, "Ptr", shapeBytes.Ptr,
        "Ptr", dataBytes.Ptr, "Int64", size, "Ptr")
    if handle = 0
        throw Error("cnp_ahk_from_doubles returned null for " kind " input")
    return handle
}

CreateSetReferenceHandle(dllPath) {
    size := 128
    dataBytes := Buffer(size * 8, 0)
    loop size {
        index := A_Index - 1
        value := index < 64 ? index * 2.0 : 256.0 + index - 64
        NumPut("Double", value, dataBytes, index * 8)
    }
    shapeBytes := MakeShapeBuffer([size])
    handle := DllCall(dllPath "\cnp_ahk_from_doubles",
        "Int", 1, "Ptr", shapeBytes.Ptr,
        "Ptr", dataBytes.Ptr, "Int64", size, "Ptr")
    if handle = 0
        throw Error("cnp_ahk_from_doubles returned null for set reference input")
    return handle
}

CreateSearchsortedVectorHandle(
    dllPath, size, asSource, descending := false
) {
    if Type(size) != "Integer" || size <= 0
        throw ValueError("searchsorted vector size must be positive")
    dataBytes := Buffer(size * 8, 0)
    loop size {
        index := A_Index - 1
        value := asSource
            ? (descending ? (size - index - 1) * 2.0 : index * 2.0)
            : Mod(index * 48271 + 17, 2 * size + 1) * 1.0
        NumPut("Double", value, dataBytes, index * 8)
    }
    shapeBytes := MakeShapeBuffer([size])
    handle := DllCall(dllPath "\cnp_ahk_from_doubles",
        "Int", 1, "Ptr", shapeBytes.Ptr,
        "Ptr", dataBytes.Ptr, "Int64", size, "Ptr")
    if handle = 0
        throw Error("cnp_ahk_from_doubles returned null for searchsorted input")
    return handle
}

CreateIndexingSelectorHandle(dllPath, size, asCondition) {
    count := asCondition ? size : Ceil(size / 2)
    dataBytes := Buffer(count * 8, 0)
    loop count {
        index := A_Index - 1
        value := asCondition ? (Mod(index, 2) = 0 ? 1 : 0) : index * 2
        NumPut("Int64", value, dataBytes, index * 8)
    }
    shapeBytes := MakeShapeBuffer([count])
    handle := DllCall(dllPath "\cnp_ahk_from_ints",
        "Int", 1, "Ptr", shapeBytes.Ptr,
        "Ptr", dataBytes.Ptr, "Int64", count, "Ptr")
    if handle = 0
        throw Error("cnp_ahk_from_ints returned null for indexing selector")
    return handle
}

CreateMatrixHandle(dllPath, rows, cols) {
    count := rows * cols
    dataBytes := Buffer(count * 8, 0)
    loop count {
        index := A_Index - 1
        value := VectorElement("general", index) * 0.001
        if Floor(index / cols) = Mod(index, cols)
            value += 2.0
        NumPut("Double", value, dataBytes, index * 8)
    }
    shapeBytes := MakeShapeBuffer([rows, cols])
    handle := DllCall(dllPath "\cnp_ahk_from_doubles",
        "Int", 2, "Ptr", shapeBytes.Ptr,
        "Ptr", dataBytes.Ptr, "Int64", count, "Ptr")
    if handle = 0
        throw Error("cnp_ahk_from_doubles returned null for matrix input")
    return handle
}

Task9BaseMatrixElement(size, row, column) {
    index := row * size + column
    value := VectorElement("general", index) * 0.001
    if row = column
        value += 2.0
    return value
}

CreateTask9RhsHandle(dllPath, size) {
    dataBytes := Buffer(size * 8, 0)
    loop size {
        row := A_Index - 1
        value := 0.0
        loop size {
            column := A_Index - 1
            value += Task9BaseMatrixElement(size, row, column)
                * VectorElement("binary", column)
        }
        NumPut("Double", value, dataBytes, row * 8)
    }
    shapeBytes := MakeShapeBuffer([size])
    handle := DllCall(dllPath "\cnp_ahk_from_doubles",
        "Int", 1, "Ptr", shapeBytes.Ptr,
        "Ptr", dataBytes.Ptr, "Int64", size, "Ptr")
    if handle = 0
        throw Error("cnp_ahk_from_doubles returned null for Task9 rhs")
    return handle
}

CreateTask9EinsumMatrixHandle(dllPath, size) {
    dataBytes := Buffer(size * size * 8, 0)
    loop size {
        index := A_Index - 1
        diagonal := 1.0 + VectorElement("general", index)
        NumPut("Double", diagonal,
            dataBytes, (index * size + index) * 8)
    }
    shapeBytes := MakeShapeBuffer([size, size])
    handle := DllCall(dllPath "\cnp_ahk_from_doubles",
        "Int", 2, "Ptr", shapeBytes.Ptr,
        "Ptr", dataBytes.Ptr, "Int64", size * size, "Ptr")
    if handle = 0
        throw Error(
            "cnp_ahk_from_doubles returned null for Task9 einsum matrix")
    return handle
}

Task9EigMatrixElement(size, row, column) {
    if row > column
        return 0.0
    if row = column
        return 2.0 + row / size
    upperIndex := row * (2 * size - row - 1) // 2
        + column - row - 1
    return (Mod(upperIndex * 37 + 11, 1009) + 1.0) * 1e-6
}

CreateTask9EigMatrixHandle(dllPath, size) {
    count := size * size
    dataBytes := Buffer(count * 8, 0)
    loop count {
        index := A_Index - 1
        row := Floor(index / size)
        column := Mod(index, size)
        NumPut("Double", Task9EigMatrixElement(size, row, column),
            dataBytes, index * 8)
    }
    shapeBytes := MakeShapeBuffer([size, size])
    handle := DllCall(dllPath "\cnp_ahk_from_doubles",
        "Int", 2, "Ptr", shapeBytes.Ptr,
        "Ptr", dataBytes.Ptr, "Int64", count, "Ptr")
    if handle = 0
        throw Error(
            "cnp_ahk_from_doubles returned null for Task9 eig matrix")
    return handle
}

Task9LstsqMatrixElement(size, row, column) {
    return (row = column ? 2.0 : 0.0) + 1.0 / size
}

CreateTask9LstsqMatrixHandle(dllPath, size) {
    count := size * size
    dataBytes := Buffer(count * 8, 0)
    loop count {
        index := A_Index - 1
        NumPut("Double", Task9LstsqMatrixElement(
            size, Floor(index / size), Mod(index, size)),
            dataBytes, index * 8)
    }
    shapeBytes := MakeShapeBuffer([size, size])
    handle := DllCall(dllPath "\cnp_ahk_from_doubles",
        "Int", 2, "Ptr", shapeBytes.Ptr,
        "Ptr", dataBytes.Ptr, "Int64", count, "Ptr")
    if handle = 0
        throw Error(
            "cnp_ahk_from_doubles returned null for Task9 lstsq matrix")
    return handle
}

CreateTask9LstsqRhsHandle(dllPath, size) {
    total := 0.0
    loop size
        total += VectorElement("binary", A_Index - 1)
    common := total / size
    dataBytes := Buffer(size * 8, 0)
    loop size {
        index := A_Index - 1
        NumPut("Double",
            2.0 * VectorElement("binary", index) + common,
            dataBytes, index * 8)
    }
    shapeBytes := MakeShapeBuffer([size])
    handle := DllCall(dllPath "\cnp_ahk_from_doubles",
        "Int", 1, "Ptr", shapeBytes.Ptr,
        "Ptr", dataBytes.Ptr, "Int64", size, "Ptr")
    if handle = 0
        throw Error(
            "cnp_ahk_from_doubles returned null for Task9 lstsq rhs")
    return handle
}

Task9RequireShape(dllPath, handle, expectedShape, context) {
    ndim := DllCall(dllPath "\cnp_ahk_ndim", "Ptr", handle, "Int")
    if ndim != expectedShape.Length
        throw Error(context " ndim " ndim " does not match expected "
            expectedShape.Length)
    expectedSize := 1
    if expectedShape.Length > 0 {
        shapeBytes := Buffer(expectedShape.Length * 8, 0)
        copied := DllCall(dllPath "\cnp_ahk_shape",
            "Ptr", handle, "Ptr", shapeBytes.Ptr, "Int")
        if copied != expectedShape.Length
            throw Error(context " shape copy failed")
        for index, expectedDimension in expectedShape {
            actualDimension := NumGet(
                shapeBytes, (index - 1) * 8, "Int64")
            if actualDimension != expectedDimension
                throw Error(context " shape dimension " index " is "
                    actualDimension ", expected " expectedDimension)
            expectedSize *= expectedDimension
        }
    }
    actualSize := DllCall(
        dllPath "\cnp_ahk_size", "Ptr", handle, "Int64")
    if actualSize != expectedSize
        throw Error(context " size " actualSize " does not match expected "
            expectedSize)
}

Task9CopyDoubles(dllPath, handle, size, context) {
    values := Buffer(size * 8, 0)
    copied := DllCall(dllPath "\cnp_ahk_get_doubles",
        "Ptr", handle, "Ptr", values.Ptr, "Int64", size, "Int64")
    if copied != size
        throw Error(context " copied " copied " doubles, expected " size)
    return values
}

Task9RequireNear(actual, expected, context,
        relativeTolerance := 1e-8, absoluteTolerance := 1e-10) {
    if !IsFiniteNumber(actual) || !IsFiniteNumber(expected)
            || Abs(actual - expected) > absoluteTolerance
                + relativeTolerance * Abs(expected)
        throw Error(context ": expected=" expected ", actual=" actual)
}

ValidateTask9BaseMatrixHandle(dllPath, handle, size, context) {
    Task9RequireShape(dllPath, handle, [size, size], context)
    values := Task9CopyDoubles(dllPath, handle, size * size, context)
    loop size * size {
        index := A_Index - 1
        Task9RequireNear(
            NumGet(values, index * 8, "Double"),
            Task9BaseMatrixElement(
                size, Floor(index / size), Mod(index, size)),
            context " flat index " index, 0.0, 0.0)
    }
}

ValidateTask9RhsHandle(dllPath, handle, size, context) {
    Task9RequireShape(dllPath, handle, [size], context)
    values := Task9CopyDoubles(dllPath, handle, size, context)
    loop size {
        row := A_Index - 1
        expected := 0.0
        loop size {
            column := A_Index - 1
            expected += Task9BaseMatrixElement(size, row, column)
                * VectorElement("binary", column)
        }
        Task9RequireNear(NumGet(values, row * 8, "Double"), expected,
            context " row " row, 0.0, 0.0)
    }
}

ValidateTask9EinsumHandle(dllPath, result, source, size) {
    Task9RequireShape(dllPath, result, [size, size],
        "Task9 einsum result")
    values := Task9CopyDoubles(
        dllPath, result, size * size, "Task9 einsum result")
    loop size * size {
        index := A_Index - 1
        row := Floor(index / size)
        column := Mod(index, size)
        expected := row = column
            ? (1.0 + VectorElement("general", row)) ** 2 : 0.0
        Task9RequireNear(NumGet(values, index * 8, "Double"), expected,
            "Task9 einsum complete output " index)
    }
    sourceValues := Task9CopyDoubles(
        dllPath, source, size * size, "Task9 einsum source")
    loop size * size {
        index := A_Index - 1
        row := Floor(index / size)
        column := Mod(index, size)
        expected := row = column
            ? 1.0 + VectorElement("general", row) : 0.0
        Task9RequireNear(
            NumGet(sourceValues, index * 8, "Double"), expected,
            "Task9 einsum source " index, 0.0, 0.0)
    }
}

ValidateTask9SolveHandle(dllPath, result, matrix, rhs, size) {
    Task9RequireShape(dllPath, result, [size], "Task9 solve result")
    values := Task9CopyDoubles(
        dllPath, result, size, "Task9 solve result")
    loop size {
        index := A_Index - 1
        Task9RequireNear(NumGet(values, index * 8, "Double"),
            VectorElement("binary", index),
            "Task9 solve complete output " index)
    }
    ValidateTask9BaseMatrixHandle(
        dllPath, matrix, size, "Task9 solve matrix source")
    ValidateTask9RhsHandle(dllPath, rhs, size, "Task9 solve rhs source")
}

ValidateTask9EigHandles(
        dllPath, eigenvalues, eigenvectors, source, size) {
    Task9RequireShape(
        dllPath, eigenvalues, [size], "Task9 eig eigenvalues")
    Task9RequireShape(
        dllPath, eigenvectors, [size, size], "Task9 eig eigenvectors")
    valueBytes := Task9CopyDoubles(
        dllPath, eigenvalues, size, "Task9 eig eigenvalues")
    vectorBytes := Task9CopyDoubles(
        dllPath, eigenvectors, size * size, "Task9 eig eigenvectors")
    loop size {
        column := A_Index - 1
        lambda := NumGet(valueBytes, column * 8, "Double")
        norm := 0.0
        loop size {
            row := A_Index - 1
            vectorValue := NumGet(
                vectorBytes, (row * size + column) * 8, "Double")
            norm += vectorValue * vectorValue
            left := 0.0
            loop size {
                inner := A_Index - 1
                left += Task9EigMatrixElement(size, row, inner)
                    * NumGet(vectorBytes,
                        (inner * size + column) * 8, "Double")
            }
            Task9RequireNear(left, vectorValue * lambda,
                "Task9 eig decomposition row " row " column " column,
                1e-7, 1e-9)
        }
        Task9RequireNear(norm, 1.0,
            "Task9 eig vector norm " column, 1e-7, 1e-9)
    }
    sourceBytes := Task9CopyDoubles(
        dllPath, source, size * size, "Task9 eig source")
    loop size * size {
        index := A_Index - 1
        Task9RequireNear(NumGet(sourceBytes, index * 8, "Double"),
            Task9EigMatrixElement(
                size, Floor(index / size), Mod(index, size)),
            "Task9 eig source " index, 0.0, 0.0)
    }
    sorted := DllCall(Numpy.Proc("cnp_ahk_sort_v2"),
        "Ptr", eigenvalues, "Int", 0, "Int", 0, "Int", 3, "Ptr")
    if sorted = 0
        throw Error("Task9 eig could not sort validation eigenvalues")
    try {
        sortedBytes := Task9CopyDoubles(
            dllPath, sorted, size, "Task9 eig sorted eigenvalues")
        loop size {
            index := A_Index - 1
            Task9RequireNear(NumGet(sortedBytes, index * 8, "Double"),
                2.0 + index / size,
                "Task9 eig complete eigenvalue " index)
        }
    } catch Error as validationError {
        FreeArrayHandle(dllPath, sorted,
            "Task9 eig failed sorted validation values")
        throw validationError
    }
    return sorted
}

ValidateTask9SvdHandles(dllPath, left, singular, right, source, size) {
    Task9RequireShape(dllPath, left, [size, size], "Task9 SVD U")
    Task9RequireShape(dllPath, singular, [size], "Task9 SVD singular")
    Task9RequireShape(dllPath, right, [size, size], "Task9 SVD Vh")
    leftBytes := Task9CopyDoubles(dllPath, left, size * size, "Task9 SVD U")
    singularBytes := Task9CopyDoubles(
        dllPath, singular, size, "Task9 SVD singular")
    rightBytes := Task9CopyDoubles(
        dllPath, right, size * size, "Task9 SVD Vh")
    loop size {
        index := A_Index - 1
        current := NumGet(singularBytes, index * 8, "Double")
        if current < 0.0
            throw Error("Task9 SVD singular value is negative at " index)
        if index > 0 {
            previous := NumGet(singularBytes, (index - 1) * 8, "Double")
            if current > previous
                throw Error("Task9 SVD singular values are not descending")
        }
    }
    loop size {
        row := A_Index - 1
        loop size {
            column := A_Index - 1
            reconstructed := 0.0
            leftGram := 0.0
            rightGram := 0.0
            loop size {
                inner := A_Index - 1
                reconstructed += NumGet(leftBytes,
                    (row * size + inner) * 8, "Double")
                    * NumGet(singularBytes, inner * 8, "Double")
                    * NumGet(rightBytes,
                        (inner * size + column) * 8, "Double")
                leftGram += NumGet(leftBytes,
                    (inner * size + row) * 8, "Double")
                    * NumGet(leftBytes,
                        (inner * size + column) * 8, "Double")
                rightGram += NumGet(rightBytes,
                    (row * size + inner) * 8, "Double")
                    * NumGet(rightBytes,
                        (column * size + inner) * 8, "Double")
            }
            Task9RequireNear(reconstructed,
                Task9BaseMatrixElement(size, row, column),
                "Task9 SVD reconstruction " row "," column,
                1e-7, 1e-9)
            expectedIdentity := row = column ? 1.0 : 0.0
            Task9RequireNear(leftGram, expectedIdentity,
                "Task9 SVD U orthogonality " row "," column,
                1e-7, 1e-9)
            Task9RequireNear(rightGram, expectedIdentity,
                "Task9 SVD Vh orthogonality " row "," column,
                1e-7, 1e-9)
        }
    }
    ValidateTask9BaseMatrixHandle(
        dllPath, source, size, "Task9 SVD matrix source")
}

ValidateTask9LstsqHandles(
        dllPath, solution, residuals, rank, singular,
        matrix, rhs, size) {
    Task9RequireShape(dllPath, solution, [size], "Task9 lstsq solution")
    Task9RequireShape(dllPath, residuals, [0], "Task9 lstsq residuals")
    Task9RequireShape(dllPath, rank, [], "Task9 lstsq rank")
    Task9RequireShape(dllPath, singular, [size], "Task9 lstsq singular")
    solutionBytes := Task9CopyDoubles(
        dllPath, solution, size, "Task9 lstsq solution")
    singularBytes := Task9CopyDoubles(
        dllPath, singular, size, "Task9 lstsq singular")
    loop size {
        index := A_Index - 1
        Task9RequireNear(NumGet(solutionBytes, index * 8, "Double"),
            VectorElement("binary", index),
            "Task9 lstsq complete solution " index, 1e-7, 1e-9)
        current := NumGet(singularBytes, index * 8, "Double")
        if current < 0.0
            throw Error("Task9 lstsq singular value is negative at " index)
        if index > 0 {
            previous := NumGet(singularBytes, (index - 1) * 8, "Double")
            if current > previous
                throw Error(
                    "Task9 lstsq singular values are not descending")
        }
        Task9RequireNear(current, index = 0 ? 3.0 : 2.0,
            "Task9 lstsq complete singular value " index,
            1e-8, 1e-10)
    }
    rankValue := DllCall(dllPath "\cnp_ahk_get_item",
        "Ptr", rank, "Int64", 0, "Double")
    Task9RequireNear(rankValue, size, "Task9 lstsq rank", 0.0, 0.0)
    Task9RequireShape(
        dllPath, matrix, [size, size], "Task9 lstsq matrix source")
    matrixBytes := Task9CopyDoubles(
        dllPath, matrix, size * size, "Task9 lstsq matrix source")
    loop size * size {
        index := A_Index - 1
        Task9RequireNear(NumGet(matrixBytes, index * 8, "Double"),
            Task9LstsqMatrixElement(
                size, Floor(index / size), Mod(index, size)),
            "Task9 lstsq matrix source " index, 0.0, 0.0)
    }
    total := 0.0
    loop size
        total += VectorElement("binary", A_Index - 1)
    common := total / size
    Task9RequireShape(
        dllPath, rhs, [size], "Task9 lstsq rhs source")
    rhsBytes := Task9CopyDoubles(
        dllPath, rhs, size, "Task9 lstsq rhs source")
    loop size {
        index := A_Index - 1
        Task9RequireNear(NumGet(rhsBytes, index * 8, "Double"),
            2.0 * VectorElement("binary", index) + common,
            "Task9 lstsq rhs source " index, 0.0, 0.0)
    }
}

CreateBitpackHandle(dllPath, size, dimensions := 0, kind := "pack") {
    if Type(size) != "Integer" || size <= 0
        throw ValueError("bit-packing input size must be positive")
    if !IsObject(dimensions)
        dimensions := [size]
    dataBytes := Buffer(size * 8, 0)
    loop size {
        index := A_Index - 1
        value := kind = "pack" ? (Mod(index * 5 + 3, 2) != 0 ? 1 : 0)
            : (kind = "unpack" ? Mod(index * 73 + 19, 256)
                : throw ValueError(
                    "unknown bit-packing input kind: " kind))
        NumPut("Int64", value,
            dataBytes, index * 8)
    }
    shapeBytes := MakeShapeBuffer(dimensions)
    integerOwner := DllCall(dllPath "\cnp_ahk_from_ints",
        "Int", dimensions.Length, "Ptr", shapeBytes.Ptr,
        "Ptr", dataBytes.Ptr, "Int64", size, "Ptr")
    if integerOwner = 0
        throw Error("cnp_ahk_from_ints returned null for bit-packing input")
    try {
        handle := DllCall(Numpy.Proc("cnp_ahk_astype"),
            "Ptr", integerOwner, "Int", 3, "Ptr")
        if handle = 0
            throw Error("cnp_ahk_astype returned null for uint8 input")
        return handle
    } finally {
        FreeArrayHandle(
            dllPath, integerOwner, "bit-packing int64 staging input")
    }
}

CreateSymmetricIndexingMatrixHandle(dllPath, rows, cols) {
    if rows != cols
        throw ValueError("symmetric indexing matrix must be square")
    count := rows * cols
    dataBytes := Buffer(count * 8, 0)
    loop count {
        index := A_Index - 1
        row := Floor(index / cols)
        column := Mod(index, cols)
        symmetricIndex := Min(row, column) * cols + Max(row, column)
        NumPut("Double", VectorElement("general", symmetricIndex),
            dataBytes, index * 8)
    }
    shapeBytes := MakeShapeBuffer([rows, cols])
    handle := DllCall(dllPath "\cnp_ahk_from_doubles",
        "Int", 2, "Ptr", shapeBytes.Ptr,
        "Ptr", dataBytes.Ptr, "Int64", count, "Ptr")
    if handle = 0
        throw Error("cnp_ahk_from_doubles returned null for symmetric indexing input")
    return handle
}

CreateZeroVectorHandle(dllPath, size) {
    shapeBytes := MakeShapeBuffer([size])
    handle := DllCall(dllPath "\cnp_ahk_zeros",
        "Int", 1, "Ptr", shapeBytes.Ptr, "Ptr")
    if handle = 0
        throw Error("cnp_ahk_zeros returned null for destination")
    return handle
}

CreateWeightedChoiceHandle(dllPath, probabilities := false) {
    count := 257
    dataBytes := Buffer(count * 8, 0)
    loop count {
        index := A_Index - 1
        value := probabilities ? (index = 193 ? 1.0 : 0.0) : index * 1.0
        NumPut("Double", value, dataBytes, index * 8)
    }
    shapeBytes := MakeShapeBuffer([count])
    handle := DllCall(dllPath "\cnp_ahk_from_doubles",
        "Int", 1, "Ptr", shapeBytes.Ptr,
        "Ptr", dataBytes.Ptr, "Int64", count, "Ptr")
    if handle = 0
        throw Error("weighted choice input creation returned null")
    return handle
}

FreeArrayHandle(dllPath, handle, context) {
    if handle = 0
        throw Error(context " attempted to free a null handle")
    status := DllCall(dllPath "\cnp_ahk_free", "Ptr", handle, "Int")
    if status != 0
        throw Error(context " cnp_ahk_free returned status " status)
}

class PreparedCase {
    __New(caseData, dllPath, seed) {
        this.Case := caseData
        this.DllPath := dllPath
        this.Seed := seed
        this.OwnedHandles := []
        this.ShapeBytes := 0
        this.TargetShapeBytes := 0
        this.DestinationHandle := 0
        this.ScalarBytes := 0
        this.CommandBytes := 0
        this.FailedIndexBytes := 0
        this.ResultHandles := 0
        this.ResultHandleCount := 0
        this.PartitionKthBytes := 0
        this.HandleBytes := 0
        this.IndexingSourceHandle := 0
        this.IndexingSelectorHandle := 0
        this.AxisSourceHandle := 0
        this.CallbackNative := 0
        this.CallbackState := 0
        this.WrapperInputs := []
        this.RawInvoke := ObjBindMethod(this, "InvokeOperation")

        operation := caseData["operation"]
        size := caseData["size"]
        rows := caseData["rows"]
        cols := caseData["cols"]
        this.SortingContract := (
            caseData["category"] = "sorting"
            ? SortingOperation(operation)
            : 0
        )
        this.ReturnsStatus := IsIntoOperation(operation)
        this.ReturnsHandlePair := operation = "divmod"
        this.ResultHandleCount := operation = "eig" ? 2
            : (operation = "svd" ? 3 : (operation = "lstsq" ? 4 : 0))
        this.ReturnsResultHandles := this.ResultHandleCount > 0
        this.ReturnsHandle := !IsScalarOperation(operation)
            && !this.ReturnsStatus && !this.ReturnsHandlePair
            && !this.ReturnsResultHandles
        this.LogicalDtype := (caseData["category"] = "bitwise"
            || caseData["category"] = "integer") ? "i64" : ((operation = "equal"
            || caseData["category"] = "logical"
            || operation = "c_contiguous_cached"
            || operation = "f_contiguous_cached") ? "bool" : ((operation = "argmax"
            || (IsObject(this.SortingContract) && this.SortingContract[1] = "argsort")
            || operation = "searchsorted"
            || operation = "searchsorted_right"
            || operation = "digitize"
            || operation = "digitize_decreasing"
            || operation = "lexsort"
            || operation = "allclose"
            || caseData["category"] = "bridge") ? "i64" : "f64"))
        if operation = "in1d_duplicates" || operation = "isin_duplicates"
            this.LogicalDtype := "bool"
        if InStr(operation, "packbits", true) != 0
            this.LogicalDtype := "u8"
        this.ValidationMode := operation = "random" ? "shape"
            : ((operation = "sort_stable_nan" || operation = "unique_nan")
                ? "numeric_nan" : "numeric")
        this.ExpectedShape := ExpectedOperationShape(operation, size, rows, cols)
    }

    AddOwned(handle, context) {
        if handle = 0
            throw Error(context " returned a null input handle")
        this.OwnedHandles.Push(handle)
        return handle
    }

    Setup() {
        operation := this.Case["operation"]
        size := this.Case["size"]
        rows := this.Case["rows"]
        cols := this.Case["cols"]
        dllPath := this.DllPath

        switch operation {
            case "zeros", "ones", "random":
                this.ShapeBytes := MakeShapeBuffer([size])
            case "choice_weighted":
                this.ShapeBytes := MakeShapeBuffer([size])
                this.AddOwned(CreateWeightedChoiceHandle(dllPath),
                    "weighted choice population")
                this.AddOwned(CreateWeightedChoiceHandle(dllPath, true),
                    "weighted choice probabilities")
            case "arange", "linspace":
                return
            case "sin", "cos", "exp", "expm1", "sqrt", "log", "log2", "log10", "log1p", "tanh", "real", "imag":
                this.AddOwned(CreateVectorHandle(dllPath, size, "general"), operation " input")
            case "absolute":
                this.AddOwned(CreateVectorHandle(dllPath, size, "absolute"), operation " input")
            case "angle":
                this.AddOwned(CreateVectorHandle(dllPath, size, "absolute"), operation " input")
            case "real_if_close":
                this.ShapeBytes := MakeShapeBuffer([size])
                this.AddOwned(DllCall(Numpy.Proc("cnp_ahk_create"),
                    "Int", 1, "Ptr", this.ShapeBytes.Ptr,
                    "Int", 16, "Ptr"), operation " complex128 input")
            case "floor":
                this.AddOwned(CreateVectorHandle(dllPath, size, "floor"), operation " input")
            case "add", "subtract", "multiply", "divide", "divmod", "power", "float_power", "heaviside", "maximum", "minimum", "fmax", "fmin", "logaddexp", "logaddexp2", "equal":
                this.AddOwned(CreateVectorHandle(dllPath, size, "general"), operation " left input")
                this.AddOwned(CreateVectorHandle(dllPath, size, "binary"), operation " right input")
                if operation = "divmod"
                    this.ResultHandles := Buffer(2 * A_PtrSize, 0)
            case "logical_and", "logical_or", "logical_xor":
                this.AddOwned(CreateVectorHandle(dllPath, size, "general"), operation " left input")
                this.AddOwned(CreateVectorHandle(dllPath, size, "binary"), operation " right input")
            case "logical_not":
                this.AddOwned(CreateVectorHandle(dllPath, size, "general"), operation " input")
            case "isnan", "isinf", "isfinite", "signbit":
                this.AddOwned(CreateVectorHandle(dllPath, size, "predicate"), operation " input")
            case "iscomplexobj", "isrealobj", "isscalar":
                this.AddOwned(CreateVectorHandle(
                    dllPath, size, "general"), operation " input")
            case "bitwise_and", "bitwise_or", "bitwise_xor":
                this.AddOwned(CreateIntVectorHandle(dllPath, size, "left"), operation " left input")
                this.AddOwned(CreateIntVectorHandle(dllPath, size, "right"), operation " right input")
            case "left_shift", "right_shift":
                this.AddOwned(CreateIntVectorHandle(dllPath, size, "left"), operation " left input")
                this.AddOwned(CreateIntVectorHandle(dllPath, size, "shift"), operation " right input")
            case "invert":
                this.AddOwned(CreateIntVectorHandle(dllPath, size, "left"), operation " input")
            case "gcd", "lcm":
                this.AddOwned(CreateIntVectorHandle(dllPath, size, "left"), operation " left input")
                this.AddOwned(CreateIntVectorHandle(dllPath, size, "right"), operation " right input")
            case "convolve", "correlate":
                this.AddOwned(CreateVectorHandle(
                    dllPath, size, "general"), operation " data input")
                this.AddOwned(CreateVectorHandle(
                    dllPath, Min(size, 8), "binary"),
                    operation " kernel input")
            case "allclose":
                this.AddOwned(CreateVectorHandle(dllPath, size, "general"), operation " left input")
                this.AddOwned(CreateVectorHandle(dllPath, size, "general"), operation " right input")
            case "sum", "mean", "std", "max", "min", "argmax", "cumsum":
                this.AddOwned(CreateVectorHandle(dllPath, size, "general"), operation " input")
            case "average":
                this.AddOwned(CreateVectorHandle(
                    dllPath, size, "general"), operation " input")
                this.AddOwned(CreateVectorHandle(
                    dllPath, size, "binary"), operation " weights")
            case "prod":
                this.AddOwned(CreateVectorHandle(dllPath, size, "product"), operation " input")
            case "softmax", "log_softmax":
                this.AxisSourceHandle := this.AddOwned(
                    CreateVectorHandle(dllPath, size, "general"),
                    operation " input")
            case "softmax_axis_last", "softmax_axis0_strided",
                    "log_softmax_axis_last", "log_softmax_axis0_strided":
                owner := this.AddOwned(CreateMatrixHandle(
                    dllPath, rows, cols), operation " source owner")
                this.AxisSourceHandle := owner
                if InStr(operation, "axis0_strided", true) != 0 {
                    this.AxisSourceHandle := this.AddOwned(
                        DllCall(Numpy.Proc("cnp_ahk_transpose"),
                            "Ptr", owner, "Ptr"),
                        operation " transposed source")
                }
            case "trapz":
                this.AxisSourceHandle := this.AddOwned(
                    CreateVectorHandle(dllPath, size, "general"),
                    operation " input")
            case "trapz_axis_last", "trapz_axis0_strided":
                owner := this.AddOwned(CreateMatrixHandle(
                    dllPath, rows, cols), operation " source owner")
                this.AxisSourceHandle := owner
                if operation = "trapz_axis0_strided" {
                    this.AxisSourceHandle := this.AddOwned(
                        DllCall(Numpy.Proc("cnp_ahk_transpose"),
                            "Ptr", owner, "Ptr"),
                        operation " transposed source")
                }
            case "packbits":
                this.AxisSourceHandle := this.AddOwned(
                    CreateBitpackHandle(dllPath, size), operation " input")
            case "packbits_axis_last", "packbits_axis0_strided":
                owner := this.AddOwned(CreateBitpackHandle(
                    dllPath, size, [rows, cols]), operation " source owner")
                this.AxisSourceHandle := owner
                if operation = "packbits_axis0_strided" {
                    this.AxisSourceHandle := this.AddOwned(
                        DllCall(Numpy.Proc("cnp_ahk_transpose"),
                            "Ptr", owner, "Ptr"),
                        operation " transposed source")
                }
            case "unpackbits":
                this.AxisSourceHandle := this.AddOwned(
                    CreateBitpackHandle(
                        dllPath, size, 0, "unpack"), operation " input")
            case "unpackbits_axis_last", "unpackbits_axis0_strided":
                owner := this.AddOwned(CreateBitpackHandle(
                    dllPath, size, [rows, cols], "unpack"),
                    operation " source owner")
                this.AxisSourceHandle := owner
                if operation = "unpackbits_axis0_strided" {
                    this.AxisSourceHandle := this.AddOwned(
                        DllCall(Numpy.Proc("cnp_ahk_transpose"),
                            "Ptr", owner, "Ptr"),
                        operation " transposed source")
                }
            case "sort", "argsort",
                    "sort_mergesort", "argsort_mergesort",
                    "sort_heapsort", "argsort_heapsort",
                    "sort_stable", "argsort_stable",
                    "sort_stable_nan", "argsort_stable_nan",
                    "partition", "argpartition",
                    "partition_nan", "argpartition_nan",
                    "msort", "sort_complex":
                this.AddOwned(CreateVectorHandle(
                    dllPath, size, this.SortingContract[3]), operation " input")
                if this.SortingContract[1] = "partition"
                        || this.SortingContract[1] = "argpartition" {
                    this.PartitionKthBytes := Buffer(8, 0)
                    NumPut("Int64", Floor(size / 2), this.PartitionKthBytes, 0)
                }
            case "unique_duplicates", "unique_nan":
                this.AddOwned(CreateVectorHandle(
                    dllPath, size,
                    operation = "unique_nan" ? "set_nan" : "set"),
                    operation " input")
            case "intersect1d_duplicates", "union1d_duplicates",
                    "setdiff1d_duplicates", "setxor1d_duplicates",
                    "in1d_duplicates", "isin_duplicates":
                this.AddOwned(CreateVectorHandle(
                    dllPath, size, "set"), operation " left input")
                this.AddOwned(CreateSetReferenceHandle(dllPath),
                    operation " right input")
            case "searchsorted", "searchsorted_right":
                this.AddOwned(CreateSearchsortedVectorHandle(
                    dllPath, size, true), operation " source")
                this.AddOwned(CreateSearchsortedVectorHandle(
                    dllPath, size, false), operation " values")
            case "digitize", "digitize_decreasing":
                this.AddOwned(CreateSearchsortedVectorHandle(
                    dllPath, size, false), operation " values")
                this.AddOwned(CreateSearchsortedVectorHandle(
                    dllPath, size, true,
                    operation = "digitize_decreasing"),
                    operation " bins")
            case "lexsort":
                secondary := this.AddOwned(CreateVectorHandle(
                    dllPath, size, "sorting"), operation " secondary key")
                primary := this.AddOwned(CreateVectorHandle(
                    dllPath, size, "binary"), operation " primary key")
                this.HandleBytes := Buffer(2 * A_PtrSize, 0)
                NumPut("Ptr", secondary, this.HandleBytes, 0)
                NumPut("Ptr", primary, this.HandleBytes, A_PtrSize)
            case "copy", "reshape",
                    "atleast_1d", "atleast_2d", "atleast_3d":
                this.AddOwned(CreateVectorHandle(dllPath, size, "general"), operation " input")
                if operation = "reshape"
                    this.TargetShapeBytes := MakeShapeBuffer([size, 1])
            case "flatten":
                this.AddOwned(CreateVectorHandle(dllPath, size, "general", [1, size]), operation " input")
            case "take", "compress":
                this.AddOwned(CreateVectorHandle(
                    dllPath, size, "general"), operation " source")
                this.AddOwned(CreateIndexingSelectorHandle(
                    dllPath, size, operation = "compress"),
                    operation " selector")
            case "take_axis0_block", "take_axis0_strided",
                    "compress_axis0_block", "compress_axis0_strided":
                owner := this.AddOwned(CreateSymmetricIndexingMatrixHandle(
                    dllPath, rows, cols), operation " source owner")
                this.IndexingSourceHandle := owner
                if SubStr(operation, -7) = "strided" {
                    this.IndexingSourceHandle := this.AddOwned(
                        DllCall(Numpy.Proc("cnp_ahk_transpose"),
                            "Ptr", owner, "Ptr"),
                        operation " transposed source")
                }
                this.IndexingSelectorHandle := this.AddOwned(
                    CreateIndexingSelectorHandle(
                        dllPath, rows, SubStr(operation, 1, 8) = "compress"),
                    operation " selector")
            case "add_into":
                this.AddOwned(CreateVectorHandle(dllPath, size, "general"), operation " left input")
                this.AddOwned(CreateVectorHandle(dllPath, size, "binary"), operation " right input")
                this.DestinationHandle := this.AddOwned(
                    CreateZeroVectorHandle(dllPath, size), operation " destination")
            case "sqrt_into", "cumsum_into":
                this.AddOwned(CreateVectorHandle(dllPath, size, "general"), operation " input")
                this.DestinationHandle := this.AddOwned(
                    CreateZeroVectorHandle(dllPath, size), operation " destination")
            case "pipeline_separate", "pipeline_batch":
                left := this.AddOwned(
                    CreateVectorHandle(dllPath, size, "general"), operation " left input")
                right := this.AddOwned(
                    CreateVectorHandle(dllPath, size, "binary"), operation " right input")
                addOut := this.AddOwned(
                    CreateZeroVectorHandle(dllPath, size), operation " add destination")
                sqrtOut := this.AddOwned(
                    CreateZeroVectorHandle(dllPath, size), operation " sqrt destination")
                this.ScalarBytes := Buffer(8, 0)
                if operation = "pipeline_batch" {
                    this.CommandBytes := Buffer(120, 0)
                    this.FailedIndexBytes := Buffer(8, 0)
                    NumPut("Int64", -1, this.FailedIndexBytes, 0)
                    NumPut("UInt", 1, this.CommandBytes, 0)
                    NumPut("Ptr", left, this.CommandBytes, 8)
                    NumPut("Ptr", right, this.CommandBytes, 16)
                    NumPut("Ptr", addOut, this.CommandBytes, 24)
                    NumPut("UInt", 2, this.CommandBytes, 40)
                    NumPut("Ptr", addOut, this.CommandBytes, 48)
                    NumPut("Ptr", sqrtOut, this.CommandBytes, 64)
                    NumPut("UInt", 4, this.CommandBytes, 80)
                    NumPut("Ptr", sqrtOut, this.CommandBytes, 88)
                    NumPut("Ptr", this.ScalarBytes.Ptr, this.CommandBytes, 104)
                    NumPut("Int64", -1, this.CommandBytes, 112)
                }
            case "matmul", "dot", "det", "inv", "norm", "cholesky", "transpose_copy", "concatenate":
                this.AddOwned(CreateMatrixHandle(dllPath, rows, cols), operation " matrix input")
            case "sum_axis_last", "cumsum_axis_last":
                this.AddOwned(CreateMatrixHandle(dllPath, rows, cols), operation " matrix input")
            case "solve":
                this.AddOwned(CreateMatrixHandle(dllPath, rows, cols), operation " matrix input")
                this.AddOwned(CreateTask9RhsHandle(dllPath, rows), operation " rhs input")
            case "einsum":
                operand := this.AddOwned(
                    CreateTask9EinsumMatrixHandle(dllPath, rows),
                    operation " matrix input")
                this.HandleBytes := Buffer(2 * A_PtrSize, 0)
                NumPut("Ptr", operand, this.HandleBytes, 0)
                NumPut("Ptr", operand, this.HandleBytes, A_PtrSize)
            case "eig":
                this.AddOwned(CreateTask9EigMatrixHandle(dllPath, rows),
                    operation " matrix input")
                this.ResultHandles := Buffer(2 * A_PtrSize, 0)
            case "svd":
                this.AddOwned(CreateMatrixHandle(dllPath, rows, cols),
                    operation " matrix input")
                this.ResultHandles := Buffer(3 * A_PtrSize, 0)
            case "lstsq":
                this.AddOwned(CreateTask9LstsqMatrixHandle(dllPath, rows),
                    operation " matrix input")
                this.AddOwned(CreateTask9LstsqRhsHandle(dllPath, rows),
                    operation " rhs input")
                this.ResultHandles := Buffer(4 * A_PtrSize, 0)
            case "fft":
                this.AddOwned(CreateVectorHandle(dllPath, size, "general"), operation " input")
            case "callback_apply_along_axis_legacy",
                    "callback_apply_along_axis_v2",
                    "callback_fromfunction_legacy",
                    "callback_fromfunction_v2",
                    "callback_fromiter_legacy", "callback_fromiter_v2",
                    "callback_vectorize_legacy", "callback_vectorize_v2":
                isBulk := SubStr(operation, -2) = "v2"
                this.CallbackState := Buffer(24, 0)
                if InStr(operation, "apply_along_axis", true) != 0 {
                    this.AddOwned(CreateVectorHandle(
                        dllPath, size, "general", [rows, cols]),
                        operation " input")
                    callbackFunction := isBulk
                        ? BenchmarkLineBatchCallback
                        : BenchmarkLineScalarCallback
                    parameterCount := isBulk ? 7 : 4
                } else if InStr(operation, "fromfunction", true) != 0 {
                    this.ShapeBytes := MakeShapeBuffer([size])
                    callbackFunction := isBulk
                        ? BenchmarkCoordinateBatchCallback
                        : BenchmarkCoordinateScalarCallback
                    parameterCount := isBulk ? 7 : 4
                } else if InStr(operation, "fromiter", true) != 0 {
                    callbackFunction := isBulk
                        ? BenchmarkIteratorBatchCallback
                        : BenchmarkIteratorScalarCallback
                    parameterCount := isBulk ? 4 : 2
                } else {
                    this.AddOwned(CreateVectorHandle(
                        dllPath, size, "general"), operation " input")
                    callbackFunction := isBulk
                        ? BenchmarkUnaryBatchCallback
                        : BenchmarkUnaryScalarCallback
                    parameterCount := isBulk ? 6 : 3
                }
                this.CallbackNative := CallbackCreate(
                    callbackFunction, "C", parameterCount)
                if this.CallbackNative = 0
                    throw Error(operation " CallbackCreate returned null")
            case "property_call":
                this.AddOwned(CreateVectorHandle(dllPath, 1, "general"), operation " owner")
            case "property_cached", "nbytes_cached",
                    "c_contiguous_cached", "f_contiguous_cached":
                owner := Numpy.NdArray.FromHandle(
                    CreateVectorHandle(dllPath, 1, "general"))
                this.WrapperInputs.Push(owner)
                ignoredSize := owner.Size
            case "static_add_call":
                this.WrapperInputs.Push(Numpy.NdArray.FromHandle(
                    CreateVectorHandle(dllPath, 1, "general")))
                this.WrapperInputs.Push(Numpy.NdArray.FromHandle(
                    CreateVectorHandle(dllPath, 1, "binary")))
                this.WrapperInputs.Push(Numpy.NdArray.FromHandle(
                    CreateZeroVectorHandle(dllPath, 1)))
                ignoredSize := this.WrapperInputs[3].Size
            default:
                throw ValueError("unknown benchmark operation during setup: " operation)
        }
    }

    InvokeOperation() {
        operation := this.Case["operation"]
        size := this.Case["size"]
        dllPath := this.DllPath

        switch operation {
            case "zeros", "ones":
                return DllCall(Numpy.Proc("cnp_ahk_" operation), "Int", 1, "Ptr", this.ShapeBytes.Ptr, "Ptr")
            case "arange":
                return DllCall(Numpy.Proc("cnp_ahk_arange"), "Double", 0.0, "Double", size * 1.0, "Double", 1.0, "Ptr")
            case "random":
                return DllCall(Numpy.Proc("cnp_ahk_random_random"), "Int", 1, "Ptr", this.ShapeBytes.Ptr, "Ptr")
            case "choice_weighted":
                return DllCall(Numpy.Proc("cnp_ahk_random_choice_v2"),
                    "Ptr", this.OwnedHandles[1],
                    "Int", 1, "Ptr", this.ShapeBytes.Ptr,
                    "Int", 0, "Int", 1,
                    "Ptr", this.OwnedHandles[2], "Ptr")
            case "linspace":
                return DllCall(Numpy.Proc("cnp_ahk_linspace"), "Double", 0.0, "Double", 1.0, "Int64", size, "Ptr")
            case "sin", "cos", "exp", "expm1", "sqrt", "log", "log2", "log10", "log1p", "absolute", "floor", "tanh":
                return DllCall(Numpy.Proc("cnp_ahk_" operation), "Ptr", this.OwnedHandles[1], "Ptr")
            case "angle":
                return DllCall(Numpy.Proc("cnp_ahk_angle"),
                    "Ptr", this.OwnedHandles[1], "Int", 0, "Ptr")
            case "real", "imag":
                return DllCall(Numpy.Proc("cnp_ahk_" operation),
                    "Ptr", this.OwnedHandles[1], "Ptr")
            case "real_if_close":
                return DllCall(Numpy.Proc("cnp_ahk_real_if_close"),
                    "Ptr", this.OwnedHandles[1], "Double", 100.0, "Ptr")
            case "add", "subtract", "multiply", "divide", "power", "float_power", "heaviside", "maximum", "minimum", "fmax", "fmin", "logaddexp", "logaddexp2", "equal":
                return DllCall(Numpy.Proc("cnp_ahk_" operation), "Ptr", this.OwnedHandles[1], "Ptr", this.OwnedHandles[2], "Ptr")
            case "divmod":
                status := DllCall(Numpy.Proc("cnp_ahk_divmod"),
                    "Ptr", this.OwnedHandles[1],
                    "Ptr", this.OwnedHandles[2],
                    "Ptr", this.ResultHandles.Ptr, "Int", 2, "Int")
                this.RequireStatus(status, "divmod")
                return this.ResultHandles
            case "logical_and", "logical_or", "logical_xor":
                return DllCall(Numpy.Proc("cnp_ahk_" operation),
                    "Ptr", this.OwnedHandles[1],
                    "Ptr", this.OwnedHandles[2], "Ptr")
            case "logical_not":
                return DllCall(Numpy.Proc("cnp_ahk_logical_not"),
                    "Ptr", this.OwnedHandles[1], "Ptr")
            case "isnan", "isinf", "isfinite", "signbit":
                return DllCall(Numpy.Proc("cnp_ahk_" operation),
                    "Ptr", this.OwnedHandles[1], "Ptr")
            case "iscomplexobj", "isrealobj", "isscalar":
                return DllCall(Numpy.Proc("cnp_ahk_" operation),
                    "Ptr", this.OwnedHandles[1], "Int")
            case "bitwise_and", "bitwise_or", "bitwise_xor", "left_shift", "right_shift":
                return DllCall(Numpy.Proc("cnp_ahk_" operation),
                    "Ptr", this.OwnedHandles[1],
                    "Ptr", this.OwnedHandles[2], "Ptr")
            case "invert":
                return DllCall(Numpy.Proc("cnp_ahk_invert"),
                    "Ptr", this.OwnedHandles[1], "Ptr")
            case "gcd", "lcm":
                return DllCall(Numpy.Proc("cnp_ahk_" operation),
                    "Ptr", this.OwnedHandles[1],
                    "Ptr", this.OwnedHandles[2], "Ptr")
            case "convolve", "correlate":
                return DllCall(Numpy.Proc("cnp_ahk_" operation),
                    "Ptr", this.OwnedHandles[1],
                    "Ptr", this.OwnedHandles[2], "Int", 1, "Ptr")
            case "allclose":
                return DllCall(Numpy.Proc("cnp_ahk_allclose"),
                    "Ptr", this.OwnedHandles[1], "Ptr", this.OwnedHandles[2],
                    "Double", 1e-5, "Double", 1e-8, "Int")
            case "sum", "mean", "std", "max", "min", "prod":
                return DllCall(Numpy.Proc("cnp_ahk_" operation), "Ptr", this.OwnedHandles[1], "Int", -1, "Double")
            case "average":
                result := DllCall(Numpy.Proc("cnp_ahk_average_v2"),
                    "Ptr", this.OwnedHandles[1],
                    "Int", 0, "Int", 1,
                    "Ptr", this.OwnedHandles[2], "Ptr")
                if result = 0 {
                    messagePointer := DllCall(
                        this.DllPath "\cnp_get_error_message", "Ptr")
                    detail := messagePointer
                        ? StrGet(messagePointer, "UTF-8") : ""
                    throw Error("average returned a null result: " detail)
                }
                try
                    return DllCall(Numpy.Proc("cnp_ahk_get_item"),
                        "Ptr", result, "Int64", 0, "Double")
                finally
                    FreeArrayHandle(
                        this.DllPath, result, "average timed result")
            case "argmax":
                return DllCall(Numpy.Proc("cnp_ahk_argmax"), "Ptr", this.OwnedHandles[1], "Int", -1, "Int64")
            case "cumsum":
                return DllCall(Numpy.Proc("cnp_ahk_cumsum"), "Ptr", this.OwnedHandles[1], "Int", -1, "Ptr")
            case "sum_axis_last":
                return DllCall(Numpy.Proc("cnp_ahk_sum_v2"),
                    "Ptr", this.OwnedHandles[1], "Int", this.Case["axis"],
                    "Int", 0, "Ptr")
            case "cumsum_axis_last":
                return DllCall(Numpy.Proc("cnp_ahk_cumsum_v2"),
                    "Ptr", this.OwnedHandles[1], "Int", this.Case["axis"],
                    "Int", 0, "Ptr")
            case "softmax", "softmax_axis_last", "softmax_axis0_strided",
                    "log_softmax", "log_softmax_axis_last",
                    "log_softmax_axis0_strided":
                functionName := InStr(operation, "log_softmax", true) = 1
                    ? "cnp_ahk_log_softmax" : "cnp_ahk_softmax"
                return DllCall(Numpy.Proc(functionName),
                    "Ptr", this.AxisSourceHandle,
                    "Int", this.Case["axis"], "Ptr")
            case "trapz", "trapz_axis_last", "trapz_axis0_strided":
                return DllCall(Numpy.Proc("cnp_ahk_trapz"),
                    "Ptr", this.AxisSourceHandle, "Ptr", 0,
                    "Double", 0.25, "Int", this.Case["axis"], "Ptr")
            case "packbits", "packbits_axis_last",
                    "packbits_axis0_strided":
                return DllCall(Numpy.Proc("cnp_ahk_packbits_v2"),
                    "Ptr", this.AxisSourceHandle,
                    "Int", this.Case["axis"], "Int", 0,
                    "Int", 0, "Ptr")
            case "unpackbits", "unpackbits_axis_last",
                    "unpackbits_axis0_strided":
                return DllCall(Numpy.Proc("cnp_ahk_unpackbits_v2"),
                    "Ptr", this.AxisSourceHandle,
                    "Int", this.Case["axis"], "Int", 0,
                    "Int64", 0, "Int", 1, "Int", 0, "Ptr")
            case "matmul", "dot":
                return DllCall(Numpy.Proc("cnp_ahk_" operation), "Ptr", this.OwnedHandles[1], "Ptr", this.OwnedHandles[1], "Ptr")
            case "det":
                return DllCall(Numpy.Proc("cnp_ahk_linalg_det"), "Ptr", this.OwnedHandles[1], "Double")
            case "inv":
                return DllCall(Numpy.Proc("cnp_ahk_linalg_inv"), "Ptr", this.OwnedHandles[1], "Ptr")
            case "norm":
                return DllCall(Numpy.Proc("cnp_ahk_linalg_norm"), "Ptr", this.OwnedHandles[1], "Double")
            case "solve":
                return DllCall(Numpy.Proc("cnp_ahk_linalg_solve"), "Ptr", this.OwnedHandles[1], "Ptr", this.OwnedHandles[2], "Ptr")
            case "cholesky":
                return DllCall(Numpy.Proc("cnp_ahk_linalg_cholesky"),
                    "Ptr", this.OwnedHandles[1], "Ptr")
            case "einsum":
                return DllCall(Numpy.Proc("cnp_ahk_einsum"),
                    "AStr", "ik,kj->ij", "Int", 2,
                    "Ptr", this.HandleBytes.Ptr, "Ptr")
            case "eig":
                return DllCall(Numpy.Proc("cnp_ahk_linalg_eig"),
                    "Ptr", this.OwnedHandles[1],
                    "Ptr", this.ResultHandles.Ptr, "Int", 2, "Int")
            case "svd":
                return DllCall(Numpy.Proc("cnp_ahk_linalg_svd_v2"),
                    "Ptr", this.OwnedHandles[1],
                    "Int", 0, "Int", 1, "Int", 0,
                    "Ptr", this.ResultHandles.Ptr, "Int", 3, "Int")
            case "lstsq":
                return DllCall(Numpy.Proc("cnp_ahk_linalg_lstsq_v2"),
                    "Ptr", this.OwnedHandles[1],
                    "Ptr", this.OwnedHandles[2],
                    "Double", 0.0, "Int", 1,
                    "Ptr", this.ResultHandles.Ptr, "Int", 4, "Int")
            case "sort", "argsort",
                    "sort_mergesort", "argsort_mergesort",
                    "sort_heapsort", "argsort_heapsort",
                    "sort_stable", "argsort_stable",
                    "sort_stable_nan", "argsort_stable_nan":
                return DllCall(Numpy.Proc(
                    "cnp_ahk_" this.SortingContract[1] "_v2"),
                    "Ptr", this.OwnedHandles[1],
                    "Int", 0, "Int", 1, "Int", this.SortingContract[2],
                    "Ptr")
            case "partition", "argpartition",
                    "partition_nan", "argpartition_nan":
                return DllCall(Numpy.Proc(
                    "cnp_ahk_" this.SortingContract[1] "_v2"),
                    "Ptr", this.OwnedHandles[1],
                    "Ptr", this.PartitionKthBytes.Ptr, "Int", 1,
                    "Int", 0, "Int", 1, "Ptr")
            case "unique_duplicates", "unique_nan":
                return DllCall(this.DllPath "\cnp_unique",
                    "Ptr", this.OwnedHandles[1],
                    "Int", 0, "Int", 0, "Int", 0, "Ptr")
            case "intersect1d_duplicates":
                return DllCall(Numpy.Proc("cnp_ahk_intersect1d"),
                    "Ptr", this.OwnedHandles[1],
                    "Ptr", this.OwnedHandles[2], "Int", 0, "Ptr")
            case "union1d_duplicates":
                return DllCall(Numpy.Proc("cnp_ahk_union1d"),
                    "Ptr", this.OwnedHandles[1],
                    "Ptr", this.OwnedHandles[2], "Ptr")
            case "setdiff1d_duplicates":
                return DllCall(Numpy.Proc("cnp_ahk_setdiff1d"),
                    "Ptr", this.OwnedHandles[1],
                    "Ptr", this.OwnedHandles[2], "Int", 0, "Ptr")
            case "setxor1d_duplicates":
                return DllCall(Numpy.Proc("cnp_ahk_setxor1d"),
                    "Ptr", this.OwnedHandles[1],
                    "Ptr", this.OwnedHandles[2], "Int", 0, "Ptr")
            case "in1d_duplicates":
                return DllCall(Numpy.Proc("cnp_ahk_in1d"),
                    "Ptr", this.OwnedHandles[1],
                    "Ptr", this.OwnedHandles[2],
                    "Int", 0, "Int", 0, "Ptr")
            case "isin_duplicates":
                return DllCall(Numpy.Proc("cnp_ahk_isin"),
                    "Ptr", this.OwnedHandles[1],
                    "Ptr", this.OwnedHandles[2],
                    "Int", 0, "Int", 0, "Ptr")
            case "searchsorted", "searchsorted_right":
                return DllCall(Numpy.Proc("cnp_ahk_searchsorted_v2"),
                    "Ptr", this.OwnedHandles[1],
                    "Ptr", this.OwnedHandles[2],
                    "AStr", operation = "searchsorted_right"
                        ? "right" : "left",
                    "Ptr", 0, "Ptr")
            case "digitize", "digitize_decreasing":
                return DllCall(Numpy.Proc("cnp_ahk_digitize"),
                    "Ptr", this.OwnedHandles[1],
                    "Ptr", this.OwnedHandles[2],
                    "Int", 0, "Ptr")
            case "lexsort":
                return DllCall(Numpy.Proc("cnp_ahk_lexsort_v2"),
                    "Ptr", this.HandleBytes.Ptr,
                    "Int", 2, "Int", -1, "Ptr")
            case "msort", "sort_complex":
                return DllCall(Numpy.Proc("cnp_ahk_" operation),
                    "Ptr", this.OwnedHandles[1], "Ptr")
            case "copy":
                return DllCall(Numpy.Proc("cnp_ahk_copy"), "Ptr", this.OwnedHandles[1], "Ptr")
            case "reshape":
                return DllCall(Numpy.Proc("cnp_ahk_reshape"), "Ptr", this.OwnedHandles[1], "Int", 2, "Ptr", this.TargetShapeBytes.Ptr, "Ptr")
            case "flatten":
                return DllCall(Numpy.Proc("cnp_ahk_flatten"), "Ptr", this.OwnedHandles[1], "Ptr")
            case "atleast_1d", "atleast_2d", "atleast_3d":
                return DllCall(Numpy.Proc("cnp_ahk_" operation),
                    "Ptr", this.OwnedHandles[1], "Ptr")
            case "take":
                return DllCall(Numpy.Proc("cnp_ahk_take_v2"),
                    "Ptr", this.OwnedHandles[1], "Ptr", this.OwnedHandles[2],
                    "Int", 0, "Int", 1, "Ptr")
            case "compress":
                return DllCall(Numpy.Proc("cnp_ahk_compress_v2"),
                    "Ptr", this.OwnedHandles[2], "Ptr", this.OwnedHandles[1],
                    "Int", 0, "Int", 1, "Ptr")
            case "take_axis0_block", "take_axis0_strided":
                result := DllCall(Numpy.Proc("cnp_ahk_take_v2"),
                    "Ptr", this.IndexingSourceHandle,
                    "Ptr", this.IndexingSelectorHandle,
                    "Int", 0, "Int", 0, "Ptr")
                if result = 0 {
                    status := DllCall(
                        this.DllPath "\cnp_get_error", "Ptr", 0, "Int")
                    messagePointer := DllCall(
                        this.DllPath "\cnp_get_error_message", "Ptr")
                    detail := messagePointer
                        ? StrGet(messagePointer, "UTF-8") : ""
                    throw Error(this.Case["operation"]
                        " returned null; cnp_take_v2 status=" status
                        (detail != "" ? ": " detail : ""))
                }
                return result
            case "compress_axis0_block", "compress_axis0_strided":
                return DllCall(Numpy.Proc("cnp_ahk_compress_v2"),
                    "Ptr", this.IndexingSelectorHandle,
                    "Ptr", this.IndexingSourceHandle,
                    "Int", 0, "Int", 0, "Ptr")
            case "add_into":
                return DllCall(Numpy.Proc("cnp_ahk_add_into"),
                    "Ptr", this.OwnedHandles[1], "Ptr", this.OwnedHandles[2],
                    "Ptr", this.DestinationHandle, "Int")
            case "sqrt_into":
                return DllCall(Numpy.Proc("cnp_ahk_sqrt_into"),
                    "Ptr", this.OwnedHandles[1],
                    "Ptr", this.DestinationHandle, "Int")
            case "cumsum_into":
                return DllCall(Numpy.Proc("cnp_ahk_cumsum_into"),
                    "Ptr", this.OwnedHandles[1], "Int", this.Case["axis"],
                    "Ptr", this.DestinationHandle, "Int")
            case "pipeline_separate":
                this.RequireStatus(DllCall(Numpy.Proc("cnp_ahk_add_into"),
                    "Ptr", this.OwnedHandles[1], "Ptr", this.OwnedHandles[2],
                    "Ptr", this.OwnedHandles[3], "Int"), "pipeline add")
                this.RequireStatus(DllCall(Numpy.Proc("cnp_ahk_sqrt_into"),
                    "Ptr", this.OwnedHandles[3], "Ptr", this.OwnedHandles[4], "Int"),
                    "pipeline sqrt")
                this.RequireStatus(DllCall(Numpy.Proc("cnp_ahk_sum_into_scalar"),
                    "Ptr", this.OwnedHandles[4], "Ptr", this.ScalarBytes.Ptr, "Int"),
                    "pipeline sum")
                return NumGet(this.ScalarBytes, 0, "Double")
            case "pipeline_batch":
                status := DllCall(Numpy.Proc("cnp_ahk_execute_batch"),
                    "Ptr", this.CommandBytes.Ptr, "Int64", 3,
                    "Ptr", this.FailedIndexBytes.Ptr, "Int")
                if status != 0 {
                    failedIndex := NumGet(this.FailedIndexBytes, 0, "Int64")
                    this.RequireStatus(status, "pipeline batch command " failedIndex)
                }
                return NumGet(this.ScalarBytes, 0, "Double")
            case "transpose_copy":
                return this.InvokeTransposeCopy()
            case "concatenate":
                return DllCall(Numpy.Proc("cnp_ahk_concatenate"), "Ptr", this.OwnedHandles[1], "Ptr", this.OwnedHandles[1], "Int", this.Case["axis"], "Ptr")
            case "fft":
                return DllCall(Numpy.Proc("cnp_ahk_fft"), "Ptr", this.OwnedHandles[1], "Ptr")
            case "callback_apply_along_axis_legacy":
                return DllCall(Numpy.Proc("cnp_ahk_apply_along_axis"),
                    "Ptr", this.CallbackNative,
                    "Ptr", this.CallbackState.Ptr,
                    "Int", this.Case["axis"],
                    "Ptr", this.OwnedHandles[1], "Ptr")
            case "callback_apply_along_axis_v2":
                return DllCall(Numpy.Proc("cnp_ahk_apply_along_axis_v2"),
                    "Ptr", this.CallbackNative,
                    "Ptr", this.CallbackState.Ptr,
                    "Int", this.Case["axis"],
                    "Ptr", this.OwnedHandles[1],
                    "Int", 0, "Ptr", 0, "Ptr")
            case "callback_fromfunction_legacy":
                return DllCall(Numpy.Proc("cnp_ahk_fromfunction"),
                    "Ptr", this.CallbackNative,
                    "Ptr", this.CallbackState.Ptr,
                    "Int", 1, "Ptr", this.ShapeBytes.Ptr, "Ptr")
            case "callback_fromfunction_v2":
                return DllCall(Numpy.Proc("cnp_ahk_fromfunction_v2"),
                    "Ptr", this.CallbackNative,
                    "Ptr", this.CallbackState.Ptr,
                    "Int", 1, "Ptr", this.ShapeBytes.Ptr, "Ptr")
            case "callback_fromiter_legacy":
                return DllCall(Numpy.Proc("cnp_ahk_fromiter"),
                    "Ptr", this.CallbackNative,
                    "Ptr", this.CallbackState.Ptr,
                    "Int64", size, "Int", Numpy.DT_FLOAT64, "Ptr")
            case "callback_fromiter_v2":
                return DllCall(Numpy.Proc("cnp_ahk_fromiter_v2"),
                    "Ptr", this.CallbackNative,
                    "Ptr", this.CallbackState.Ptr,
                    "Int64", size, "Int", Numpy.DT_FLOAT64, "Ptr")
            case "callback_vectorize_legacy":
                return DllCall(Numpy.Proc("cnp_ahk_vectorize"),
                    "Ptr", this.CallbackNative,
                    "Ptr", this.CallbackState.Ptr,
                    "Ptr", this.OwnedHandles[1], "Ptr")
            case "callback_vectorize_v2":
                return DllCall(Numpy.Proc("cnp_ahk_vectorize_v2"),
                    "Ptr", this.CallbackNative,
                    "Ptr", this.CallbackState.Ptr,
                    "Ptr", this.OwnedHandles[1], "Ptr")
            case "property_call":
                return DllCall(dllPath "\cnp_ahk_size", "Ptr", this.OwnedHandles[1], "Int64")
            case "property_cached":
                return this.WrapperInputs[1].Size
            case "nbytes_cached":
                return this.WrapperInputs[1].Nbytes
            case "c_contiguous_cached":
                return this.WrapperInputs[1].CContiguous
            case "f_contiguous_cached":
                return this.WrapperInputs[1].FContiguous
            case "static_add_call":
                Numpy.Add(this.WrapperInputs[1], this.WrapperInputs[2],
                    this.WrapperInputs[3])
                return this.WrapperInputs[3].Size
            default:
                throw ValueError("unknown benchmark operation during invocation: " operation)
        }
    }

    InvokeTransposeCopy() {
        return DllCall(Numpy.Proc("cnp_ahk_transpose_copy"),
            "Ptr", this.OwnedHandles[1], "Ptr")
    }

    RequireStatus(status, context) {
        if status = 0
            return
        messagePointer := DllCall(this.DllPath "\cnp_get_error_message", "Ptr")
        detail := messagePointer ? StrGet(messagePointer, "UTF-8") : ""
        message := context " returned status " status
        if detail != ""
            message .= ": " detail
        throw Error(message)
    }

    TimedInvoke() {
        result := this.RawInvoke.Call()
        if this.ReturnsResultHandles {
            this.RequireStatus(result, this.Case["operation"])
            this.FreeResultHandles(
                this.Case["operation"] " timed results")
            return
        }
        if this.ReturnsHandlePair {
            this.FreeHandlePair(result, this.Case["operation"] " timed results")
            return
        }
        if this.ReturnsStatus {
            this.RequireStatus(result, this.Case["operation"])
            return
        }
        if this.ReturnsHandle {
            if result = 0
                throw Error(this.Case["operation"] " returned a null timed handle")
            FreeArrayHandle(this.DllPath, result, this.Case["operation"] " timed result")
            return
        }
        if !IsFiniteNumber(result)
            throw Error(this.Case["operation"] " returned a non-finite timed scalar")
    }

    ValidationSignature() {
        result := this.RawInvoke.Call()
        if this.ReturnsResultHandles {
            try {
                this.RequireStatus(result,
                    this.Case["operation"] " validation")
                return this.Task9ResultValidationSignature()
            } finally
                this.FreeResultHandles(
                    this.Case["operation"] " validation results", true)
        }
        if this.ReturnsHandlePair {
            try {
                return HandlePairValidationSignature(
                    this.DllPath, result, this.ExpectedShape,
                    this.LogicalDtype, this.ValidationMode)
            } finally
                this.FreeHandlePair(
                    result, this.Case["operation"] " validation results")
        }
        if this.ReturnsStatus {
            this.RequireStatus(result, this.Case["operation"] " validation")
            if this.DestinationHandle = 0
                throw Error(this.Case["operation"] " has no validation destination")
            return HandleValidationSignature(
                this.DllPath, this.DestinationHandle, this.ExpectedShape,
                this.LogicalDtype, this.ValidationMode)
        }
        if !this.ReturnsHandle
            return ScalarValidationSignature(result, this.ExpectedShape, this.LogicalDtype, this.ValidationMode)
        if result = 0
            throw Error(this.Case["operation"] " returned a null validation handle")
        normalized := 0
        normalizedSource := 0
        try {
            validationHandle := result
            if this.Case["operation"] = "sort_complex" {
                resultDtype := DllCall(
                    this.DllPath "\cnp_ahk_dtype", "Ptr", result, "Int")
                if resultDtype != 16
                    throw Error(
                        "sort_complex validation result dtype is "
                        resultDtype ", expected complex128")
                normalized := DllCall(
                    this.DllPath "\cnp_real", "Ptr", result, "Ptr")
                if normalized = 0
                    throw Error(
                        "cnp_real returned null during sort_complex validation")
                normalizedSource := DllCall(
                    this.DllPath "\cnp_imag", "Ptr", result, "Ptr")
                if normalizedSource = 0
                    throw Error(
                        "cnp_imag returned null during sort_complex validation")
                imaginarySignature := HandleValidationSignature(
                    this.DllPath, normalizedSource, this.ExpectedShape,
                    this.LogicalDtype, this.ValidationMode)
                if imaginarySignature["sum"] != 0.0
                    throw Error(
                        "sort_complex real-input imaginary sum is not zero")
                for value in imaginarySignature["values"] {
                    if value != 0.0
                        throw Error(
                            "sort_complex real-input imaginary sample is not zero")
                }
                validationHandle := normalized
            } else if IsObject(this.SortingContract)
                    && (this.SortingContract[1] = "partition"
                    || this.SortingContract[1] = "argpartition") {
                kth := Floor(this.Case["size"] / 2)
                if this.SortingContract[1] = "partition"
                    ValidatePartitionHandle(
                        this.DllPath, result, this.OwnedHandles[1],
                        this.Case["size"], kth)
                else
                    ValidateArgpartitionHandle(
                        this.DllPath, result, this.Case["size"], kth,
                        this.SortingContract[3])
                normalizedSource := DllCall(
                    Numpy.Proc("cnp_ahk_sort_v2"),
                    "Ptr", this.OwnedHandles[1],
                    "Int", 0, "Int", 1, "Int", 3, "Ptr")
                if normalizedSource = 0
                    throw Error(
                        this.Case["operation"]
                        " could not sort its source for validation")
                validationHandle := normalizedSource
                if this.SortingContract[3] = "sorting_nan" {
                    normalized := DllCall(this.DllPath "\cnp_nan_to_num",
                        "Ptr", normalizedSource,
                        "Double", 8192.0,
                        "Double", 8193.0, "Double", -8193.0,
                        "Ptr")
                    if normalized = 0
                        throw Error(
                            this.Case["operation"]
                            " could not normalize NaNs for validation")
                    validationHandle := normalized
                }
            }
            if this.Case["operation"] = "fft" {
                signature := FftValidationSignature(
                    this.DllPath, validationHandle, this.ExpectedShape,
                    this.LogicalDtype, this.ValidationMode)
            } else if this.Case["operation"] = "choice_weighted" {
                ValidateWeightedChoiceHandle(
                    this.DllPath, validationHandle, this.Case)
                signature := HandleValidationSignature(
                    this.DllPath, validationHandle, this.ExpectedShape,
                    this.LogicalDtype, this.ValidationMode)
            } else if this.Case["operation"] = "trapz"
                    || this.Case["operation"] = "trapz_axis_last"
                    || this.Case["operation"] = "trapz_axis0_strided"
                signature := ValidateTrapzHandle(
                    this.DllPath, validationHandle, this.Case)
            else
                signature := HandleValidationSignature(
                    this.DllPath, validationHandle, this.ExpectedShape,
                    this.LogicalDtype, this.ValidationMode,
                    this.Case["operation"], this.Case["size"])
            if this.Case["operation"] = "einsum"
                ValidateTask9EinsumHandle(
                    this.DllPath, validationHandle, this.OwnedHandles[1],
                    this.Case["rows"])
            if this.Case["operation"] = "solve"
                ValidateTask9SolveHandle(
                    this.DllPath, validationHandle,
                    this.OwnedHandles[1], this.OwnedHandles[2],
                    this.Case["rows"])
            if this.Case["operation"] = "packbits"
                    || this.Case["operation"] = "packbits_axis_last"
                    || this.Case["operation"] = "packbits_axis0_strided"
                ValidatePackbitsHandle(
                    this.DllPath, validationHandle, this.Case)
            if this.Case["operation"] = "unpackbits"
                    || this.Case["operation"] = "unpackbits_axis_last"
                    || this.Case["operation"] = "unpackbits_axis0_strided"
                ValidateUnpackbitsHandle(
                    this.DllPath, validationHandle, this.Case)
            if this.Case["operation"] = "softmax"
                    || this.Case["operation"] = "softmax_axis_last"
                    || this.Case["operation"] = "softmax_axis0_strided"
                    || this.Case["operation"] = "log_softmax"
                    || this.Case["operation"] = "log_softmax_axis_last"
                    || this.Case["operation"] = "log_softmax_axis0_strided"
                ValidateSoftmaxHandle(
                    this.DllPath, result, this.Case)
            if IsObject(this.SortingContract) && this.SortingContract[1] = "argsort"
                ValidateArgsortHandle(
                    this.DllPath, result, this.Case["size"],
                    this.SortingContract[3])
            if this.Case["category"] = "functional_callback" {
                ValidateFunctionalCallbackHandle(
                    this.DllPath, result, this.Case)
                callbackInvocations := NumGet(
                    this.CallbackState, 0, "Int64")
                logicalElements := NumGet(
                    this.CallbackState, 8, "Int64")
                expectedLogicalElements := InStr(
                    this.Case["operation"], "apply_along_axis", true) != 0
                    ? this.Case["rows"] : this.Case["size"]
                expectedInvocations :=
                    SubStr(this.Case["operation"], -2) = "v2"
                    ? Ceil(expectedLogicalElements / 256)
                    : expectedLogicalElements
                if callbackInvocations != expectedInvocations
                    throw Error(this.Case["operation"]
                        " callback invocations are " callbackInvocations
                        ", expected " expectedInvocations)
                if logicalElements != expectedLogicalElements
                    throw Error(this.Case["operation"]
                        " logical elements are " logicalElements
                        ", expected " expectedLogicalElements)
                signature["callback_invocations"] := callbackInvocations
                signature["logical_elements"] := logicalElements
            }
            return signature
        } finally {
            try {
                if normalized != 0
                    FreeArrayHandle(
                        this.DllPath, normalized,
                        this.Case["operation"] " normalized validation result")
            } finally {
                try {
                    if normalizedSource != 0
                        FreeArrayHandle(
                            this.DllPath, normalizedSource,
                            this.Case["operation"] " sorted validation source")
                } finally
                    FreeArrayHandle(
                        this.DllPath, result,
                        this.Case["operation"] " validation result")
            }
        }
    }

    FreeHandlePair(handles, context) {
        if !(handles is Buffer) || handles.Size != 2 * A_PtrSize
            throw Error(context " did not return a two-handle buffer")
        first := NumGet(handles, 0, "Ptr")
        second := NumGet(handles, A_PtrSize, "Ptr")
        if first = 0 || second = 0 {
            try {
                if first != 0
                    FreeArrayHandle(this.DllPath, first, context " first")
            } finally {
                if second != 0
                    FreeArrayHandle(this.DllPath, second, context " second")
                NumPut("Ptr", 0, handles, 0)
                NumPut("Ptr", 0, handles, A_PtrSize)
            }
            throw Error(context " contained a null result handle")
        }
        try {
            try
                FreeArrayHandle(this.DllPath, first, context " first")
            finally
                FreeArrayHandle(this.DllPath, second, context " second")
        } finally {
            NumPut("Ptr", 0, handles, 0)
            NumPut("Ptr", 0, handles, A_PtrSize)
        }
    }

    FreeResultHandles(context, allowEmpty := false) {
        if !(this.ResultHandles is Buffer)
                || this.ResultHandles.Size
                    != this.ResultHandleCount * A_PtrSize
            throw Error(context " has no exact result-handle buffer")
        firstError := 0
        found := 0
        loop this.ResultHandleCount {
            offset := (A_Index - 1) * A_PtrSize
            handle := NumGet(this.ResultHandles, offset, "Ptr")
            if handle != 0 {
                found += 1
                try
                    FreeArrayHandle(this.DllPath, handle,
                        context " handle " A_Index)
                catch Error as freeError {
                    if !IsObject(firstError)
                        firstError := freeError
                }
                NumPut("Ptr", 0, this.ResultHandles, offset)
            }
        }
        if found != this.ResultHandleCount && !allowEmpty
            throw Error(context " contained " found " handles, expected "
                this.ResultHandleCount)
        if IsObject(firstError)
            throw firstError
    }

    Task9ResultValidationSignature() {
        operation := this.Case["operation"]
        handles := []
        loop this.ResultHandleCount {
            handle := NumGet(
                this.ResultHandles, (A_Index - 1) * A_PtrSize, "Ptr")
            if handle = 0
                throw Error(operation " validation result " A_Index
                    " is null")
            handles.Push(handle)
        }
        switch operation {
            case "eig":
                normalized := ValidateTask9EigHandles(
                    this.DllPath, handles[1], handles[2],
                    this.OwnedHandles[1], this.Case["rows"])
                try
                    return HandleValidationSignature(
                        this.DllPath, normalized, [this.Case["rows"]],
                        "f64", "numeric")
                finally
                    FreeArrayHandle(this.DllPath, normalized,
                        "eig sorted validation values")
            case "svd":
                ValidateTask9SvdHandles(
                    this.DllPath, handles[1], handles[2], handles[3],
                    this.OwnedHandles[1], this.Case["rows"])
                return HandleValidationSignature(
                    this.DllPath, handles[2], [this.Case["rows"]],
                    "f64", "numeric")
            case "lstsq":
                ValidateTask9LstsqHandles(
                    this.DllPath, handles[1], handles[2], handles[3],
                    handles[4], this.OwnedHandles[1],
                    this.OwnedHandles[2], this.Case["rows"])
                return HandleValidationSignature(
                    this.DllPath, handles[1], [this.Case["rows"]],
                    "f64", "numeric")
            default:
                throw ValueError(
                    "unknown Task9 multi-result operation: " operation)
        }
    }

    ResetTimedState() {
        if this.Case["operation"] = "random"
            DllCall(this.DllPath "\cnp_ahk_random_seed", "Int", this.Seed)
        else if this.Case["operation"] = "choice_weighted"
            DllCall(Numpy.Proc("cnp_ahk_random_seed_v2"),
                "UInt64", this.Seed)
        else if this.Case["category"] = "functional_callback" {
            NumPut("Int64", 0, this.CallbackState, 0)
            NumPut("Int64", 0, this.CallbackState, 8)
            NumPut("Int64", 0, this.CallbackState, 16)
        }
    }

    Close() {
        firstError := 0
        this.WrapperInputs := []
        while this.OwnedHandles.Length > 0 {
            handle := this.OwnedHandles.Pop()
            try
                FreeArrayHandle(this.DllPath, handle, this.Case["operation"] " input")
            catch Error as err {
                if !IsObject(firstError)
                    firstError := err
            }
        }
        if this.CallbackNative != 0 {
            try
                CallbackFree(this.CallbackNative)
            catch Error as err {
                if !IsObject(firstError)
                    firstError := err
            }
            this.CallbackNative := 0
        }
        this.CallbackState := 0
        if IsObject(firstError)
            throw firstError
    }
}

PrepareCase(caseData, dllPath, seed := 12345) {
    prepared := PreparedCase(caseData, dllPath, seed)
    try {
        prepared.Setup()
        return prepared
    } catch Error as err {
        prepared.Close()
        throw err
    }
}

IsScalarOperation(operation) {
    return operation = "sum" || operation = "mean"
        || operation = "average" || operation = "std"
        || operation = "max" || operation = "min" || operation = "prod"
        || operation = "argmax" || operation = "det" || operation = "norm"
        || operation = "allclose"
        || operation = "property_call" || operation = "property_cached"
        || operation = "nbytes_cached"
        || operation = "c_contiguous_cached"
        || operation = "f_contiguous_cached"
        || operation = "static_add_call" || operation = "pipeline_separate"
        || operation = "pipeline_batch" || operation = "iscomplexobj"
        || operation = "isrealobj" || operation = "isscalar"
}

IsIntoOperation(operation) {
    return operation = "add_into" || operation = "sqrt_into"
        || operation = "cumsum_into"
}

ExpectedOperationShape(operation, size, rows, cols) {
    if IsScalarOperation(operation)
        return []
    switch operation {
        case "reshape":
            return [size, 1]
        case "atleast_2d":
            return [1, size]
        case "atleast_3d":
            return [1, size, 1]
        case "matmul", "dot", "inv", "cholesky", "einsum":
            return [rows, cols]
        case "solve", "eig", "svd", "lstsq":
            return [rows]
        case "transpose_copy":
            return [cols, rows]
        case "concatenate":
            return [rows * 2, cols]
        case "sum_axis_last":
            return [rows]
        case "callback_apply_along_axis_legacy",
                "callback_apply_along_axis_v2":
            return [rows]
        case "cumsum_axis_last":
            return [rows, cols]
        case "softmax_axis_last", "softmax_axis0_strided",
                "log_softmax_axis_last", "log_softmax_axis0_strided":
            return [rows, cols]
        case "trapz":
            return []
        case "trapz_axis_last":
            return [rows]
        case "trapz_axis0_strided":
            return [cols]
        case "packbits":
            return [Ceil(size / 8)]
        case "packbits_axis_last":
            return [rows, Ceil(cols / 8)]
        case "packbits_axis0_strided":
            return [Ceil(rows / 8), cols]
        case "unpackbits":
            return [size * 8]
        case "unpackbits_axis_last":
            return [rows, cols * 8]
        case "unpackbits_axis0_strided":
            return [rows * 8, cols]
        case "fft":
            return [size, 2]
        case "divmod":
            return [2, size]
        case "take", "compress":
            return [Ceil(size / 2)]
        case "take_axis0_block", "take_axis0_strided",
                "compress_axis0_block", "compress_axis0_strided":
            return [Ceil(rows / 2), cols]
        case "unique_duplicates":
            return [Min(size, 256)]
        case "unique_nan":
            return [Min(Floor(size / 2), 128) + 1]
        case "intersect1d_duplicates", "union1d_duplicates",
                "setdiff1d_duplicates", "setxor1d_duplicates":
            uniqueCount := Min(size, 256)
            intersectionCount := Ceil(Min(uniqueCount, 128) / 2)
            switch operation {
                case "intersect1d_duplicates": return [intersectionCount]
                case "union1d_duplicates":
                    return [uniqueCount + 128 - intersectionCount]
                case "setdiff1d_duplicates":
                    return [uniqueCount - intersectionCount]
                default:
                    return [uniqueCount + 128 - 2 * intersectionCount]
            }
        default:
            return [size]
    }
}

FftValidationSignature(
    dllPath, handle, expectedShape, logicalDtype, mode
) {
    if expectedShape.Length != 2 || expectedShape[2] != 2
        throw Error("FFT validation expected shape must be [size, 2]")
    if logicalDtype != "f64" || mode != "numeric"
        throw Error("FFT validation requires numeric f64 projection")
    resultDtype := DllCall(
        dllPath "\cnp_ahk_dtype", "Ptr", handle, "Int")
    if resultDtype != 16
        throw Error(
            "FFT validation result dtype is " resultDtype
            ", expected complex128")

    realHandle := 0
    imaginaryHandle := 0
    try {
        realHandle := DllCall(dllPath "\cnp_real", "Ptr", handle, "Ptr")
        if realHandle = 0
            throw Error("cnp_real returned null during FFT validation")
        imaginaryHandle := DllCall(
            dllPath "\cnp_imag", "Ptr", handle, "Ptr")
        if imaginaryHandle = 0
            throw Error("cnp_imag returned null during FFT validation")

        itemShape := [expectedShape[1]]
        realSignature := HandleValidationSignature(
            dllPath, realHandle, itemShape, logicalDtype, mode)
        imaginarySignature := HandleValidationSignature(
            dllPath, imaginaryHandle, itemShape, logicalDtype, mode)
        projectedSize := expectedShape[1] * 2
        sampleIndices := UniqueSampleIndices(projectedSize)
        values := []
        for flatIndex in sampleIndices {
            componentHandle := Mod(flatIndex, 2) = 0
                ? realHandle : imaginaryHandle
            sourceIndex := Floor(flatIndex / 2)
            value := DllCall(
                dllPath "\cnp_ahk_get_item",
                "Ptr", componentHandle, "Int64", sourceIndex, "Double")
            if !IsFiniteNumber(value)
                throw Error(
                    "FFT validation sample is non-finite at projected index "
                    flatIndex)
            values.Push(value)
        }
        aggregate := realSignature["sum"] + imaginarySignature["sum"]
        if !IsFiniteNumber(aggregate)
            throw Error("FFT validation projected sum is non-finite")
        return Map(
            "mode", mode,
            "shape", expectedShape.Clone(),
            "size", projectedSize,
            "logical_dtype", logicalDtype,
            "sample_indices", sampleIndices,
            "values", values,
            "sum", aggregate)
    } finally {
        if imaginaryHandle != 0
            FreeArrayHandle(
                dllPath, imaginaryHandle, "FFT imaginary validation result")
        if realHandle != 0
            FreeArrayHandle(dllPath, realHandle, "FFT real validation result")
    }
}

ValidateFunctionalCallbackHandle(dllPath, handle, caseData) {
    operation := caseData["operation"]
    isApply := InStr(operation, "apply_along_axis", true) != 0
    outputSize := isApply ? caseData["rows"] : caseData["size"]
    valueBytes := Buffer(outputSize * 8, 0)
    copied := DllCall(dllPath "\cnp_ahk_get_doubles",
        "Ptr", handle, "Ptr", valueBytes.Ptr,
        "Int64", outputSize, "Int64")
    if copied != outputSize
        throw Error(operation " copied " copied
            " callback values, expected " outputSize)
    loop outputSize {
        index := A_Index - 1
        if isApply {
            expected := 0.0
            loop caseData["cols"]
                expected += VectorElement(
                    "general", index * caseData["cols"] + A_Index - 1)
        } else if InStr(operation, "fromfunction", true) != 0
            expected := index * 1.0
        else if InStr(operation, "fromiter", true) != 0
            expected := index * 0.25 - 3.0
        else
            expected := VectorElement("general", index) * 1.5 + 0.25
        actual := NumGet(valueBytes, index * 8, "Double")
        if actual != expected
            throw Error(operation
                " callback result mismatch at flat index " index
                ": actual=" actual ", expected=" expected)
    }
    return true
}

HandleValidationSignature(
    dllPath, handle, expectedShape, logicalDtype, mode,
    nanOperation := "", nanInputSize := 0
) {
    ndim := DllCall(dllPath "\cnp_ahk_ndim", "Ptr", handle, "Int")
    if ndim < 0
        throw Error("cnp_ahk_ndim failed for validation handle")
    if ndim != expectedShape.Length
        throw Error("validation ndim " ndim " does not match expected " expectedShape.Length)
    if ndim = 0
        throw Error("array validation handle unexpectedly has zero dimensions")

    shapeBytes := Buffer(ndim * 8, 0)
    shapeStatus := DllCall(dllPath "\cnp_ahk_shape", "Ptr", handle, "Ptr", shapeBytes.Ptr, "Int")
    if shapeStatus != ndim
        throw Error("cnp_ahk_shape returned " shapeStatus ", expected " ndim)
    actualShape := []
    for index, expectedDimension in expectedShape {
        actualDimension := NumGet(shapeBytes, (index - 1) * 8, "Int64")
        if actualDimension != expectedDimension
            throw Error("validation shape dimension " index " is " actualDimension ", expected " expectedDimension)
        actualShape.Push(actualDimension)
    }

    size := DllCall(dllPath "\cnp_ahk_size", "Ptr", handle, "Int64")
    if size <= 0
        throw Error("cnp_ahk_size returned a non-positive validation size")
    expectedSize := 1
    for dimension in expectedShape
        expectedSize *= dimension
    if size != expectedSize
        throw Error("validation size " size " does not match expected " expectedSize)

    dtype := DllCall(dllPath "\cnp_ahk_dtype", "Ptr", handle, "Int")
    switch logicalDtype {
        case "f64": expectedDtype := 13
        case "i64": expectedDtype := 10
        case "u8": expectedDtype := 3
        case "bool": expectedDtype := 1
        default: throw ValueError("unknown validation logical dtype: " logicalDtype)
    }
    if dtype != expectedDtype
        throw Error("validation dtype enum " dtype " does not match expected " expectedDtype " for " logicalDtype)

    signature := Map("mode", mode, "shape", actualShape, "size", size, "logical_dtype", logicalDtype)
    if mode = "shape"
        return signature
    if mode = "numeric_nan" {
        if logicalDtype != "f64"
            throw ValueError("numeric_nan validation requires f64 dtype")
        valueBytes := Buffer(size * 8, 0)
        copied := DllCall(
            dllPath "\cnp_ahk_get_doubles",
            "Ptr", handle,
            "Ptr", valueBytes.Ptr,
            "Int64", size,
            "Int64"
        )
        if copied != size
            throw Error(
                "cnp_ahk_get_doubles copied " copied
                " numeric_nan values, expected " size)
        facts := ValidateDeterministicNanResultBuffer(
            valueBytes, size, nanOperation, nanInputSize)
        normalized := DllCall(dllPath "\cnp_nan_to_num",
            "Ptr", handle,
            "Double", 8192.0, "Double", 8193.0, "Double", -8193.0,
            "Ptr")
        if normalized = 0
            throw Error("cnp_nan_to_num returned null after raw NaN validation")
        try {
            signature := HandleValidationSignature(
                dllPath, normalized, expectedShape, logicalDtype, "numeric")
        } finally
            FreeArrayHandle(
                dllPath, normalized, nanOperation " normalized validation result")
        signature["mode"] := "numeric_nan"
        signature["nan_count"] := facts["nan_count"]
        signature["first_nan_index"] := facts["first_nan_index"]
        signature["trailing_nan"] := facts["trailing_nan"]
        signature["finite_members_exact"] := facts["finite_members_exact"]
        return signature
    }
    if mode != "numeric"
        throw ValueError("unknown validation mode: " mode)

    sampleIndices := UniqueSampleIndices(size)
    values := []
    for index in sampleIndices {
        value := DllCall(dllPath "\cnp_ahk_get_item", "Ptr", handle, "Int64", index, "Double")
        if !IsFiniteNumber(value)
            throw Error("validation sample is non-finite at flat index " index)
        values.Push(value)
    }
    aggregate := DllCall(dllPath "\cnp_ahk_sum", "Ptr", handle, "Int", -1, "Double")
    if !IsFiniteNumber(aggregate)
        throw Error("validation sum is non-finite")
    signature["sample_indices"] := sampleIndices
    signature["values"] := values
    signature["sum"] := aggregate
    return signature
}

ValidateWeightedChoiceFullBuffer(valueBytes, caseData) {
    size := caseData["size"]
    if !(valueBytes is Buffer) || valueBytes.Size != size * 8
        throw ValueError(
            "weighted choice full-output requires an exact float64 buffer")
    loop size {
        actual := NumGet(valueBytes, (A_Index - 1) * 8, "Double")
        if !IsFiniteNumber(actual) || actual != 193.0
            throw Error(
                "weighted choice full-output value at flat index "
                (A_Index - 1) " is " actual ", expected 193")
    }
    return true
}

ValidateWeightedChoiceHandle(dllPath, handle, caseData) {
    size := caseData["size"]
    valueBytes := Buffer(size * 8, 0)
    copied := DllCall(dllPath "\cnp_ahk_get_doubles",
        "Ptr", handle, "Ptr", valueBytes.Ptr,
        "Int64", size, "Int64")
    if copied != size
        throw Error("cnp_ahk_get_doubles copied " copied
            " weighted choice values, expected " size)
    return ValidateWeightedChoiceFullBuffer(valueBytes, caseData)
}

Task8MatrixElement(rows, cols, row, column) {
    index := row * cols + column
    value := VectorElement("general", index) * 0.001
    if row = column
        value += 2.0
    return value
}

Task8AxisSourceValue(caseData, row, column) {
    operation := caseData["operation"]
    if operation = "softmax" || operation = "log_softmax"
            || operation = "trapz"
        return VectorElement("general", column)
    if operation = "softmax_axis_last"
            || operation = "log_softmax_axis_last"
            || operation = "trapz_axis_last"
        return Task8MatrixElement(
            caseData["rows"], caseData["cols"], row, column)
    if operation = "softmax_axis0_strided"
            || operation = "log_softmax_axis0_strided"
            || operation = "trapz_axis0_strided"
        return Task8MatrixElement(
            caseData["rows"], caseData["cols"], column, row)
    throw ValueError(
        "Task8 full-output does not support operation: " operation)
}

ValidateSoftmaxFullBuffer(valueBytes, caseData) {
    size := caseData["size"]
    if !(valueBytes is Buffer) || valueBytes.Size != size * 8
        throw ValueError(
            "softmax full-output requires an exact float64 buffer")
    operation := caseData["operation"]
    isLog := InStr(operation, "log_softmax", true) = 1
    errorPrefix := isLog ? "log_softmax" : "softmax"
    rows := (operation = "softmax" || operation = "log_softmax")
        ? 1 : caseData["rows"]
    cols := (operation = "softmax" || operation = "log_softmax")
        ? size : caseData["cols"]
    axis := caseData["axis"]
    axisLength := axis = 0 ? rows : cols
    sliceCount := axis = 0 ? cols : rows
    loop sliceCount {
        slice := A_Index - 1
        maximum := 0.0
        loop axisLength {
            axisIndex := A_Index - 1
            row := axis = 0 ? axisIndex : slice
            column := axis = 0 ? slice : axisIndex
            sourceValue := Task8AxisSourceValue(
                caseData, row, column)
            if axisIndex = 0 || sourceValue > maximum
                maximum := sourceValue
        }
        denominator := 0.0
        loop axisLength {
            axisIndex := A_Index - 1
            row := axis = 0 ? axisIndex : slice
            column := axis = 0 ? slice : axisIndex
            sourceValue := Task8AxisSourceValue(
                caseData, row, column)
            denominator += DllCall(
                "msvcrt\exp", "Double", sourceValue - maximum, "Double")
        }
        loop axisLength {
            axisIndex := A_Index - 1
            row := axis = 0 ? axisIndex : slice
            column := axis = 0 ? slice : axisIndex
            flatIndex := row * cols + column
            actual := NumGet(valueBytes, flatIndex * 8, "Double")
            sourceValue := Task8AxisSourceValue(
                caseData, row, column)
            expected := isLog
                ? sourceValue - maximum - DllCall(
                    "msvcrt\log", "Double", denominator, "Double")
                : DllCall(
                    "msvcrt\exp", "Double", sourceValue - maximum, "Double")
                    / denominator
            tolerance := 1e-12 * Max(1.0, Abs(expected))
            if !IsFiniteNumber(actual) || Abs(actual - expected) > tolerance
                throw Error(
                    errorPrefix " full-output value at flat index " flatIndex
                    " is " actual ", expected " expected)
        }
    }
    return true
}

ValidateSoftmaxHandle(dllPath, handle, caseData) {
    size := caseData["size"]
    valueBytes := Buffer(size * 8, 0)
    copied := DllCall(dllPath "\cnp_ahk_get_doubles",
        "Ptr", handle, "Ptr", valueBytes.Ptr,
        "Int64", size, "Int64")
    if copied != size
        throw Error(
            "cnp_ahk_get_doubles copied " copied
            " softmax values, expected " size)
    return ValidateSoftmaxFullBuffer(valueBytes, caseData)
}

ValidateTrapzFullBuffer(valueBytes, caseData) {
    operation := caseData["operation"]
    outputSize := operation = "trapz" ? 1
        : (caseData["axis"] = 0 ? caseData["cols"] : caseData["rows"])
    if !(valueBytes is Buffer) || valueBytes.Size != outputSize * 8
        throw ValueError(
            "trapz full-output requires an exact float64 buffer")
    rows := operation = "trapz" ? 1 : caseData["rows"]
    cols := operation = "trapz" ? caseData["size"] : caseData["cols"]
    axis := caseData["axis"]
    axisLength := axis = 0 ? rows : cols
    loop outputSize {
        outputIndex := A_Index - 1
        integral := 0.0
        loop axisLength - 1 {
            axisIndex := A_Index - 1
            firstRow := axis = 0 ? axisIndex : outputIndex
            firstColumn := axis = 0 ? outputIndex : axisIndex
            secondRow := axis = 0 ? axisIndex + 1 : outputIndex
            secondColumn := axis = 0 ? outputIndex : axisIndex + 1
            first := Task8AxisSourceValue(
                caseData, firstRow, firstColumn)
            second := Task8AxisSourceValue(
                caseData, secondRow, secondColumn)
            integral += (first + second) * 0.5 * 0.25
        }
        actual := NumGet(valueBytes, outputIndex * 8, "Double")
        tolerance := 1e-12 * Max(1.0, Abs(integral))
        if !IsFiniteNumber(actual) || Abs(actual - integral) > tolerance
            throw Error(
                "trapz full-output value at flat index " outputIndex
                " is " actual ", expected " integral)
    }
    return true
}

ValidateTrapzHandle(dllPath, handle, caseData) {
    operation := caseData["operation"]
    outputSize := operation = "trapz" ? 1
        : (caseData["axis"] = 0 ? caseData["cols"] : caseData["rows"])
    valueBytes := Buffer(outputSize * 8, 0)
    copied := DllCall(dllPath "\cnp_ahk_get_doubles",
        "Ptr", handle, "Ptr", valueBytes.Ptr,
        "Int64", outputSize, "Int64")
    if copied != outputSize
        throw Error(
            "cnp_ahk_get_doubles copied " copied
            " trapz values, expected " outputSize)
    ValidateTrapzFullBuffer(valueBytes, caseData)
    if operation = "trapz"
        return ScalarValidationSignature(
            NumGet(valueBytes, 0, "Double"), [], "f64", "numeric")
    expectedShape := caseData["axis"] = 0
        ? [caseData["cols"]] : [caseData["rows"]]
    return HandleValidationSignature(
        dllPath, handle, expectedShape, "f64", "numeric")
}

Task8BitpackSourceValue(caseData, row, column) {
    operation := caseData["operation"]
    if operation = "packbits"
        index := column
    else if operation = "packbits_axis_last"
        index := row * caseData["cols"] + column
    else if operation = "packbits_axis0_strided"
        index := column * caseData["cols"] + row
    else
        throw ValueError(
            "packbits full-output does not support operation: " operation)
    return Mod(index * 5 + 3, 2) != 0 ? 1 : 0
}

ValidatePackbitsFullBuffer(valueBytes, caseData) {
    operation := caseData["operation"]
    if operation = "packbits" {
        outputRows := 1
        outputCols := Ceil(caseData["size"] / 8)
    } else if caseData["axis"] = 1 {
        outputRows := caseData["rows"]
        outputCols := Ceil(caseData["cols"] / 8)
    } else {
        outputRows := Ceil(caseData["rows"] / 8)
        outputCols := caseData["cols"]
    }
    outputSize := outputRows * outputCols
    if !(valueBytes is Buffer) || valueBytes.Size != outputSize * 8
        throw ValueError(
            "packbits full-output requires an exact double copy buffer")
    loop outputRows {
        outputRow := A_Index - 1
        loop outputCols {
            outputColumn := A_Index - 1
            expected := 0
            loop 8 {
                bitIndex := A_Index - 1
                if operation = "packbits" {
                    sourceIndex := outputColumn * 8 + bitIndex
                    present := sourceIndex < caseData["size"]
                    sourceRow := 0
                    sourceColumn := sourceIndex
                } else if caseData["axis"] = 1 {
                    sourceRow := outputRow
                    sourceColumn := outputColumn * 8 + bitIndex
                    present := sourceColumn < caseData["cols"]
                } else {
                    sourceRow := outputRow * 8 + bitIndex
                    sourceColumn := outputColumn
                    present := sourceRow < caseData["rows"]
                }
                if present && Task8BitpackSourceValue(
                        caseData, sourceRow, sourceColumn)
                    expected |= 1 << (7 - bitIndex)
            }
            flatIndex := outputRow * outputCols + outputColumn
            actual := NumGet(valueBytes, flatIndex * 8, "Double")
            if !IsFiniteNumber(actual) || actual != expected
                throw Error(
                    "packbits full-output byte at flat index " flatIndex
                    " is " actual ", expected " expected)
        }
    }
    return true
}

ValidatePackbitsHandle(dllPath, handle, caseData) {
    operation := caseData["operation"]
    outputSize := operation = "packbits" ? Ceil(caseData["size"] / 8)
        : (caseData["axis"] = 1
            ? caseData["rows"] * Ceil(caseData["cols"] / 8)
            : Ceil(caseData["rows"] / 8) * caseData["cols"])
    valueBytes := Buffer(outputSize * 8, 0)
    copied := DllCall(dllPath "\cnp_ahk_get_doubles",
        "Ptr", handle, "Ptr", valueBytes.Ptr,
        "Int64", outputSize, "Int64")
    if copied != outputSize
        throw Error(
            "cnp_ahk_get_doubles copied " copied
            " packbits bytes, expected " outputSize)
    return ValidatePackbitsFullBuffer(valueBytes, caseData)
}

Task8UnpackbitsSourceValue(caseData, row, column) {
    operation := caseData["operation"]
    if operation = "unpackbits"
        index := column
    else if operation = "unpackbits_axis_last"
        index := row * caseData["cols"] + column
    else if operation = "unpackbits_axis0_strided"
        index := column * caseData["cols"] + row
    else
        throw ValueError(
            "unpackbits full-output does not support operation: " operation)
    return Mod(index * 73 + 19, 256)
}

ValidateUnpackbitsFullBuffer(valueBytes, caseData) {
    operation := caseData["operation"]
    if operation = "unpackbits" {
        outputRows := 1
        outputCols := caseData["size"] * 8
    } else if caseData["axis"] = 1 {
        outputRows := caseData["rows"]
        outputCols := caseData["cols"] * 8
    } else {
        outputRows := caseData["rows"] * 8
        outputCols := caseData["cols"]
    }
    outputSize := outputRows * outputCols
    if !(valueBytes is Buffer) || valueBytes.Size != outputSize * 8
        throw ValueError(
            "unpackbits full-output requires an exact double copy buffer")
    loop outputRows {
        outputRow := A_Index - 1
        loop outputCols {
            outputColumn := A_Index - 1
            if operation = "unpackbits" {
                sourceRow := 0
                sourceColumn := Floor(outputColumn / 8)
                bitIndex := Mod(outputColumn, 8)
            } else if caseData["axis"] = 1 {
                sourceRow := outputRow
                sourceColumn := Floor(outputColumn / 8)
                bitIndex := Mod(outputColumn, 8)
            } else {
                sourceRow := Floor(outputRow / 8)
                sourceColumn := outputColumn
                bitIndex := Mod(outputRow, 8)
            }
            sourceValue := Task8UnpackbitsSourceValue(
                caseData, sourceRow, sourceColumn)
            expected := (sourceValue >> (7 - bitIndex)) & 1
            flatIndex := outputRow * outputCols + outputColumn
            actual := NumGet(valueBytes, flatIndex * 8, "Double")
            if !IsFiniteNumber(actual) || actual != expected
                throw Error(
                    "unpackbits full-output byte at flat index " flatIndex
                    " is " actual ", expected " expected)
        }
    }
    return true
}

ValidateUnpackbitsHandle(dllPath, handle, caseData) {
    operation := caseData["operation"]
    outputSize := operation = "unpackbits" ? caseData["size"] * 8
        : (caseData["axis"] = 1
            ? caseData["rows"] * caseData["cols"] * 8
            : caseData["rows"] * 8 * caseData["cols"])
    valueBytes := Buffer(outputSize * 8, 0)
    copied := DllCall(dllPath "\cnp_ahk_get_doubles",
        "Ptr", handle, "Ptr", valueBytes.Ptr,
        "Int64", outputSize, "Int64")
    if copied != outputSize
        throw Error(
            "cnp_ahk_get_doubles copied " copied
            " unpackbits bytes, expected " outputSize)
    return ValidateUnpackbitsFullBuffer(valueBytes, caseData)
}

ValidateDeterministicNanResultBuffer(
    valueBytes, resultSize, operation, inputSize
) {
    if !(valueBytes is Buffer) || valueBytes.Size != resultSize * 8
        throw ValueError("raw NaN validation requires an exact float64 buffer")
    if Type(inputSize) != "Integer" || inputSize <= 0
        throw ValueError("raw NaN validation input size must be positive")
    switch operation {
        case "sort_stable_nan": expectedNanCount := Ceil(inputSize / 2)
        case "unique_nan": expectedNanCount := 1
        default: throw ValueError(
            "raw NaN validation does not support operation: " operation)
    }

    nanCount := 0
    firstNanIndex := -1
    trailingNan := true
    seenNan := false
    loop resultSize {
        index := A_Index - 1
        bits := NumGet(valueBytes, index * 8, "UInt64")
        exponent := bits & 0x7FF0000000000000
        mantissa := bits & 0x000FFFFFFFFFFFFF
        if exponent = 0x7FF0000000000000 {
            if mantissa = 0
                throw Error(
                    "raw NaN validation found infinity at flat index " index)
            nanCount += 1
            if firstNanIndex = -1
                firstNanIndex := index
            seenNan := true
        } else if seenNan
            trailingNan := false
    }
    expectedFirstNanIndex := resultSize - expectedNanCount
    if nanCount != expectedNanCount
        throw Error(
            operation " raw NaN count is " nanCount
            ", expected " expectedNanCount)
    if firstNanIndex != expectedFirstNanIndex
        throw Error(
            operation " first NaN index is " firstNanIndex
            ", expected " expectedFirstNanIndex)
    if !trailingNan
        throw Error(operation " raw NaNs do not form the complete tail")

    finiteIndex := 0
    if operation = "sort_stable_nan" {
        counts := []
        loop 4096
            counts.Push(0)
        loop inputSize {
            sourceIndex := A_Index - 1
            if Mod(sourceIndex, 2) = 1 {
                value := Integer(VectorElement("sorting", sourceIndex))
                counts[value + 1] += 1
            }
        }
        for valueIndex, count in counts {
            loop count {
                actual := NumGet(valueBytes, finiteIndex * 8, "Double")
                expected := (valueIndex - 1) * 1.0
                if actual != expected
                    throw Error(
                        operation " finite member " finiteIndex " is " actual
                        ", expected " expected)
                finiteIndex += 1
            }
        }
    } else {
        seenValues := []
        loop 256
            seenValues.Push(false)
        loop inputSize {
            sourceIndex := A_Index - 1
            if Mod(sourceIndex, 2) = 1
                seenValues[Mod(sourceIndex, 256) + 1] := true
        }
        for valueIndex, present in seenValues {
            if present {
                actual := NumGet(valueBytes, finiteIndex * 8, "Double")
                expected := (valueIndex - 1) * 1.0
                if actual != expected
                    throw Error(
                        operation " finite member " finiteIndex " is " actual
                        ", expected " expected)
                finiteIndex += 1
            }
        }
    }
    if finiteIndex != expectedFirstNanIndex
        throw Error(
            operation " verified " finiteIndex " finite members, expected "
            expectedFirstNanIndex)
    return Map(
        "nan_count", nanCount,
        "first_nan_index", firstNanIndex,
        "trailing_nan", true,
        "finite_members_exact", true
    )
}

HandlePairValidationSignature(
    dllPath, handles, expectedShape, logicalDtype, mode
) {
    if !(handles is Buffer) || handles.Size != 2 * A_PtrSize
        throw Error("pair validation requires a two-handle buffer")
    if expectedShape.Length != 2 || expectedShape[1] != 2
        throw Error("pair validation expected shape must be [2, size]")
    itemShape := [expectedShape[2]]
    quotient := NumGet(handles, 0, "Ptr")
    remainder := NumGet(handles, A_PtrSize, "Ptr")
    if quotient = 0 || remainder = 0
        throw Error("pair validation received a null result handle")
    quotientSignature := HandleValidationSignature(
        dllPath, quotient, itemShape, logicalDtype, mode)
    remainderSignature := HandleValidationSignature(
        dllPath, remainder, itemShape, logicalDtype, mode)
    itemSize := expectedShape[2]
    return Map(
        "mode", mode,
        "shape", expectedShape.Clone(),
        "size", itemSize * 2,
        "logical_dtype", logicalDtype,
        "sample_indices", [0, itemSize, itemSize * 2 - 1],
        "values", [
            quotientSignature["values"][1],
            remainderSignature["values"][1],
            remainderSignature["values"][
                remainderSignature["values"].Length]
        ],
        "sum", quotientSignature["sum"] + remainderSignature["sum"]
    )
}

ValidateArgsortHandle(dllPath, handle, size, vectorKind := "sorting") {
    indexBytes := Buffer(size * 8, 0)
    copied := DllCall(
        dllPath "\cnp_ahk_get_doubles",
        "Ptr", handle,
        "Ptr", indexBytes.Ptr,
        "Int64", size,
        "Int64"
    )
    if copied != size
        throw Error("cnp_ahk_get_doubles copied " copied " argsort indices, expected " size)
    return ValidateArgsortIndexBuffer(indexBytes, size, vectorKind)
}

ValidateArgsortIndexBuffer(indexBytes, size, vectorKind := "sorting") {
    if !(indexBytes is Buffer) || indexBytes.Size != size * 8
        throw ValueError("argsort index buffer size does not match the requested result size")
    if vectorKind != "sorting" && vectorKind != "sorting_nan"
        throw ValueError("unknown argsort validation vector kind: " vectorKind)
    seen := Buffer(size, 0)
    previousKey := 0
    hasPreviousFinite := false
    seenNan := false
    loop size {
        rawIndex := NumGet(indexBytes, (A_Index - 1) * 8, "Double")
        if !IsFiniteNumber(rawIndex) || rawIndex != Floor(rawIndex)
            throw Error("argsort result contains a non-integer index at rank " (A_Index - 1))
        index := Floor(rawIndex)
        if index < 0 || index >= size
            throw Error("argsort result contains an out-of-bounds index: " index)
        if NumGet(seen, index, "UChar") != 0
            throw Error("argsort result contains duplicate index: " index)
        NumPut("UChar", 1, seen, index)

        isNan := vectorKind = "sorting_nan" && Mod(index, 2) = 0
        if isNan {
            seenNan := true
            continue
        }
        if seenNan
            throw Error("argsort result contains a finite key after a NaN key at rank " (A_Index - 1))
        key := VectorElement("sorting", index)
        if hasPreviousFinite && key < previousKey
            throw Error("argsort result keys are not nondecreasing at rank " (A_Index - 1))
        previousKey := key
        hasPreviousFinite := true
    }
    return true
}

ValidatePartitionHandle(dllPath, handle, sourceHandle, size, kth) {
    valueBytes := Buffer(size * 8, 0)
    copied := DllCall(
        dllPath "\cnp_ahk_get_doubles",
        "Ptr", handle,
        "Ptr", valueBytes.Ptr,
        "Int64", size,
        "Int64"
    )
    if copied != size
        throw Error(
            "cnp_ahk_get_doubles copied " copied
            " partition values, expected " size)
    ValidatePartitionValueBuffer(valueBytes, size, kth)

    sortedResult := 0
    sortedSource := 0
    normalizedResult := 0
    normalizedSource := 0
    try {
        sortedResult := DllCall(Numpy.Proc("cnp_ahk_sort_v2"),
            "Ptr", handle, "Int", 0, "Int", 1, "Int", 3, "Ptr")
        sortedSource := DllCall(Numpy.Proc("cnp_ahk_sort_v2"),
            "Ptr", sourceHandle, "Int", 0, "Int", 1, "Int", 3, "Ptr")
        if sortedResult = 0 || sortedSource = 0
            throw Error("partition permutation validation could not sort values")
        normalizedResult := DllCall(dllPath "\cnp_nan_to_num",
            "Ptr", sortedResult,
            "Double", 8192.0, "Double", 8193.0, "Double", -8193.0,
            "Ptr")
        normalizedSource := DllCall(dllPath "\cnp_nan_to_num",
            "Ptr", sortedSource,
            "Double", 8192.0, "Double", 8193.0, "Double", -8193.0,
            "Ptr")
        if normalizedResult = 0 || normalizedSource = 0
            throw Error("partition permutation validation could not normalize values")
        if !DllCall(dllPath "\cnp_array_equal",
                "Ptr", normalizedResult, "Ptr", normalizedSource, "Int")
            throw Error("partition values are not a permutation of the source")
    } finally {
        try {
            if normalizedSource != 0
                FreeArrayHandle(
                    dllPath, normalizedSource,
                    "partition normalized source validation")
        } finally {
            try {
                if normalizedResult != 0
                    FreeArrayHandle(
                        dllPath, normalizedResult,
                        "partition normalized result validation")
            } finally {
                try {
                    if sortedSource != 0
                        FreeArrayHandle(
                            dllPath, sortedSource,
                            "partition sorted source validation")
                } finally {
                    if sortedResult != 0
                        FreeArrayHandle(
                            dllPath, sortedResult,
                            "partition sorted result validation")
                }
            }
        }
    }
    return true
}

ValidatePartitionValueBuffer(valueBytes, size, kth) {
    if !(valueBytes is Buffer) || valueBytes.Size != size * 8
        throw ValueError(
            "partition value buffer size does not match the requested result size")
    if Type(kth) != "Integer" || kth < 0 || kth >= size
        throw ValueError("partition kth is out of bounds")
    pivot := NumGet(valueBytes, kth * 8, "Double")
    pivotIsNan := pivot != pivot
    loop kth {
        value := NumGet(valueBytes, (A_Index - 1) * 8, "Double")
        if !pivotIsNan && (value != value || value > pivot)
            throw Error(
                "partition contains a value above the kth pivot before kth")
    }
    loop size - kth - 1 {
        value := NumGet(valueBytes, (kth + A_Index) * 8, "Double")
        if pivotIsNan {
            if value = value
                throw Error(
                    "partition contains a finite value after a NaN pivot")
        } else if value = value && value < pivot
            throw Error(
                "partition contains a value below the kth pivot after kth")
    }
    return true
}

ValidateArgpartitionHandle(
    dllPath, handle, size, kth, vectorKind := "sorting"
) {
    indexBytes := Buffer(size * 8, 0)
    copied := DllCall(
        dllPath "\cnp_ahk_get_doubles",
        "Ptr", handle,
        "Ptr", indexBytes.Ptr,
        "Int64", size,
        "Int64"
    )
    if copied != size
        throw Error(
            "cnp_ahk_get_doubles copied " copied
            " argpartition indices, expected " size)
    return ValidateArgpartitionIndexBuffer(
        indexBytes, size, kth, vectorKind)
}

ValidateArgpartitionIndexBuffer(
    indexBytes, size, kth, vectorKind := "sorting"
) {
    if !(indexBytes is Buffer) || indexBytes.Size != size * 8
        throw ValueError(
            "argpartition index buffer size does not match the requested result size")
    if vectorKind != "sorting" && vectorKind != "sorting_nan"
        throw ValueError(
            "unknown argpartition validation vector kind: " vectorKind)
    seen := Buffer(size, 0)
    partitionedValues := Buffer(size * 8, 0)
    loop size {
        rawIndex := NumGet(indexBytes, (A_Index - 1) * 8, "Double")
        if !IsFiniteNumber(rawIndex) || rawIndex != Floor(rawIndex)
            throw Error(
                "argpartition result contains a non-integer index at rank "
                (A_Index - 1))
        index := Floor(rawIndex)
        if index < 0 || index >= size
            throw Error(
                "argpartition result contains an out-of-bounds index: " index)
        if NumGet(seen, index, "UChar") != 0
            throw Error(
                "argpartition result contains duplicate index: " index)
        NumPut("UChar", 1, seen, index)
        if vectorKind = "sorting_nan" && Mod(index, 2) = 0
            NumPut(
                "UInt64", 0x7FF8000000000000,
                partitionedValues, (A_Index - 1) * 8)
        else
            NumPut(
                "Double", VectorElement("sorting", index),
                partitionedValues, (A_Index - 1) * 8)
    }
    ValidatePartitionValueBuffer(partitionedValues, size, kth)
    return true
}

ScalarValidationSignature(value, expectedShape, logicalDtype, mode) {
    if expectedShape.Length != 0
        throw Error("scalar operation has a non-scalar expected shape")
    if mode != "numeric"
        throw ValueError("scalar validation only supports numeric mode")
    if !IsFiniteNumber(value)
        throw Error("validation scalar is non-finite")
    numericValue := value + 0.0
    return Map(
        "mode", mode,
        "shape", [],
        "size", 1,
        "logical_dtype", logicalDtype,
        "sample_indices", [0],
        "values", [numericValue],
        "sum", numericValue
    )
}

UniqueSampleIndices(size) {
    candidates := [0, Floor(size / 2), size - 1]
    result := []
    seen := Map()
    for index in candidates {
        if !seen.Has(index) {
            result.Push(index)
            seen[index] := true
        }
    }
    return result
}

RunBenchmarkCase(caseData, dllPath, warmups, sampleCount, targetSampleNs, frequency, seed) {
    before := DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64")
    if before < 0
        throw Error("cnp_ahk_allocated_memory returned a negative value before " caseData["id"])
    prepared := 0
    try {
        prepared := PrepareCase(caseData, dllPath, seed)
        prepared.ResetTimedState()
        validation := prepared.ValidationSignature()
        prepared.ResetTimedState()
        timing := TimeOperation(ObjBindMethod(prepared, "TimedInvoke"), warmups, sampleCount, targetSampleNs, frequency)
    } finally {
        if IsObject(prepared)
            prepared.Close()
    }
    after := DllCall(dllPath "\cnp_ahk_allocated_memory", "Int64")
    retained := after - before
    if retained != 0
        throw Error("retained memory for " caseData["id"] " is " retained " bytes")
    return Map(
        "case", caseData,
        "inner_loops", timing["inner_loops"],
        "samples_ns", timing["samples_ns"],
        "validation", validation,
        "retained_bytes", retained
    )
}

ValidationToJson(validation) {
    parts := [
        Chr(34) "mode" Chr(34) ":" JsonString(validation["mode"]),
        Chr(34) "shape" Chr(34) ":" JsonNumberArray(validation["shape"]),
        Chr(34) "size" Chr(34) ":" JsonNumber(validation["size"]),
        Chr(34) "logical_dtype" Chr(34) ":" JsonString(validation["logical_dtype"])
    ]
    if validation["mode"] = "numeric" || validation["mode"] = "numeric_nan" {
        parts.Push(Chr(34) "sample_indices" Chr(34) ":" JsonNumberArray(validation["sample_indices"]))
        parts.Push(Chr(34) "values" Chr(34) ":" JsonNumberArray(validation["values"]))
        parts.Push(Chr(34) "sum" Chr(34) ":" JsonNumber(validation["sum"]))
    }
    if validation["mode"] = "numeric_nan" {
        parts.Push(Chr(34) "nan_count" Chr(34) ":" JsonNumber(validation["nan_count"]))
        parts.Push(Chr(34) "first_nan_index" Chr(34) ":" JsonNumber(validation["first_nan_index"]))
        parts.Push(Chr(34) "trailing_nan" Chr(34) ":" JsonBoolean(validation["trailing_nan"]))
        parts.Push(Chr(34) "finite_members_exact" Chr(34) ":" JsonBoolean(validation["finite_members_exact"]))
    }
    if validation.Has("callback_invocations")
        parts.Push(Chr(34) "callback_invocations" Chr(34) ":"
            JsonNumber(validation["callback_invocations"]))
    if validation.Has("logical_elements")
        parts.Push(Chr(34) "logical_elements" Chr(34) ":"
            JsonNumber(validation["logical_elements"]))
    return "{" JoinStrings(parts, ",") "}"
}

BenchmarkCaseToJson(resultCase) {
    caseData := resultCase["case"]
    parts := [
        Chr(34) "id" Chr(34) ":" JsonString(caseData["id"]),
        Chr(34) "category" Chr(34) ":" JsonString(caseData["category"]),
        Chr(34) "operation" Chr(34) ":" JsonString(caseData["operation"]),
        Chr(34) "dtype" Chr(34) ":" JsonString(caseData["dtype"]),
        Chr(34) "size" Chr(34) ":" JsonNumber(caseData["size"]),
        Chr(34) "rows" Chr(34) ":" JsonNumber(caseData["rows"]),
        Chr(34) "cols" Chr(34) ":" JsonNumber(caseData["cols"]),
        Chr(34) "axis" Chr(34) ":" JsonNumber(caseData["axis"]),
        Chr(34) "inner_loops" Chr(34) ":" JsonNumber(resultCase["inner_loops"]),
        Chr(34) "samples_ns" Chr(34) ":" JsonNumberArray(resultCase["samples_ns"]),
        Chr(34) "validation" Chr(34) ":" ValidationToJson(resultCase["validation"]),
        Chr(34) "retained_bytes" Chr(34) ":" JsonNumber(resultCase["retained_bytes"])
    ]
    return "{" JoinStrings(parts, ",") "}"
}

BenchmarkDocumentToJson(arguments, dllVersion, frequency, targetSampleNs,
                        simdLevel, simdName, numThreads, resultCases) {
    metadataParts := [
        Chr(34) "ahk_version" Chr(34) ":" JsonString(A_AhkVersion),
        Chr(34) "dll_version" Chr(34) ":" JsonString(dllVersion),
        Chr(34) "dll_path" Chr(34) ":" JsonString(arguments["dll"]),
        Chr(34) "timer" Chr(34) ":" JsonString("QueryPerformanceCounter"),
        Chr(34) "timer_frequency" Chr(34) ":" JsonNumber(frequency),
        Chr(34) "warmups" Chr(34) ":" JsonNumber(arguments["warmups"]),
        Chr(34) "sample_count" Chr(34) ":" JsonNumber(arguments["samples"]),
        Chr(34) "target_sample_ns" Chr(34) ":" JsonNumber(targetSampleNs),
        Chr(34) "seed" Chr(34) ":" JsonNumber(arguments["seed"]),
        Chr(34) "simd_level" Chr(34) ":" JsonNumber(simdLevel),
        Chr(34) "simd_name" Chr(34) ":" JsonString(simdName),
        Chr(34) "num_threads" Chr(34) ":" JsonNumber(numThreads)
    ]
    caseParts := []
    for resultCase in resultCases
        caseParts.Push(BenchmarkCaseToJson(resultCase))
    documentParts := [
        Chr(34) "schema_version" Chr(34) ":1",
        Chr(34) "runtime" Chr(34) ":" JsonString("cnumpy"),
        Chr(34) "metadata" Chr(34) ":{" JoinStrings(metadataParts, ",") "}",
        Chr(34) "cases" Chr(34) ":[" JoinStrings(caseParts, ",") "]"
    ]
    return "{" JoinStrings(documentParts, ",") "}"
}

RequireRegularFile(path, label) {
    attributes := FileExist(path)
    if attributes = "" || InStr(attributes, "D")
        throw ValueError(label " does not exist as a regular file: " path)
}

ResolveCliPath(path) {
    if RegExMatch(path, "i)^(?:[A-Z]:[\\/]|\\\\)")
        return path
    return A_InitialWorkingDir "\" path
}

ConfigureBenchmarkThreads(dllPath) {
    configuredText := EnvGet("CNP_NUM_THREADS")
    requested := configuredText = ""
        ? 0 : ParseStrictUnsigned(
            configuredText, "CNP_NUM_THREADS", true)
    if requested > 2147483647
        throw ValueError(
            "CNP_NUM_THREADS must be in the signed 32-bit range [0, 2147483647]")
    status := DllCall(dllPath "\cnp_ahk_set_num_threads",
        "Int", requested, "Int")
    if status != 0 {
        messagePointer := DllCall(dllPath "\cnp_get_error_message", "Ptr")
        detail := messagePointer ? StrGet(messagePointer, "UTF-8") : ""
        message := "cnp_ahk_set_num_threads returned status " status
        if detail != ""
            message .= ": " detail
        throw Error(message)
    }
    configured := DllCall(dllPath "\cnp_ahk_get_num_threads", "Int")
    if configured != requested
        throw Error("cnp_ahk_get_num_threads returned " configured
            ", expected " requested)
    return configured
}

Main(args) {
    arguments := ParseArgs(args)
    arguments["jobs"] := ResolveCliPath(arguments["jobs"])
    arguments["output"] := ResolveCliPath(arguments["output"])
    arguments["dll"] := ResolveCliPath(arguments["dll"])
    RequireRegularFile(arguments["jobs"], "jobs file")
    RequireRegularFile(arguments["dll"], "DLL")
    cases := ParseJobs(arguments["jobs"])
    targetSampleNs := arguments["target_sample_ms"] * 1000000.0
    if !IsFiniteNumber(targetSampleNs) || targetSampleNs <= 0
        throw ValueError("target sample duration in nanoseconds must be finite and positive")

    moduleHandle := DllCall("kernel32\LoadLibraryW", "Str", arguments["dll"], "Ptr")
    if moduleHandle = 0
        throw OSError(A_LastError, "LoadLibraryW failed for " arguments["dll"])
    initAttempted := false
    try {
        initAttempted := true
        initStatus := DllCall(arguments["dll"] "\cnp_ahk_init", "Int")
        if initStatus != 0
            throw Error("cnp_ahk_init returned status " initStatus)
        Numpy.DllPath := arguments["dll"]
        Numpy.DllHandle := moduleHandle
        Numpy.Initialized := true
        numThreads := ConfigureBenchmarkThreads(arguments["dll"])
        simdLevel := DllCall(arguments["dll"] "\cnp_ahk_simd_level", "Int")
        if simdLevel = 1
            simdName := "sse2"
        else if simdLevel = 2
            simdName := "avx2"
        else
            throw Error("cnp_ahk_simd_level returned undocumented value " simdLevel)
        versionPointer := DllCall(arguments["dll"] "\cnp_ahk_version", "Ptr")
        if versionPointer = 0
            throw Error("cnp_ahk_version returned null")
        dllVersion := StrGet(versionPointer, "UTF-8")
        if dllVersion = ""
            throw Error("cnp_ahk_version returned an empty string")
        frequency := QueryTimerFrequency()

        FileAppend("cnumpy benchmark: " cases.Length " case(s)`n", "*")
        resultCases := []
        for index, caseData in cases {
            FileAppend("[" index "/" cases.Length "] " caseData["id"] "`n", "*")
            resultCases.Push(RunBenchmarkCase(caseData, arguments["dll"], arguments["warmups"], arguments["samples"], targetSampleNs, frequency, arguments["seed"]))
        }
        document := BenchmarkDocumentToJson(
            arguments, dllVersion, frequency, targetSampleNs,
            simdLevel, simdName, numThreads, resultCases)
        WriteJsonAtomic(arguments["output"], document)
        FileAppend("cnumpy benchmark: wrote " arguments["output"] "`n", "*")
    } finally {
        cleanupError := 0
        if initAttempted {
            try {
                cleanupStatus := DllCall(arguments["dll"] "\cnp_ahk_cleanup", "Int")
                if cleanupStatus != 0
                    cleanupError := Error("cnp_ahk_cleanup returned status " cleanupStatus)
            } catch Error as err {
                cleanupError := err
            }
        }
        Numpy.Initialized := false
        Numpy.ProcCache := Map()
        Numpy.ProcCacheHandle := 0
        Numpy.DllHandle := 0
        freeLibraryStatus := DllCall("kernel32\FreeLibrary", "Ptr", moduleHandle, "Int")
        if !freeLibraryStatus
            throw OSError(A_LastError, "FreeLibrary failed")
        if IsObject(cleanupError)
            throw cleanupError
    }
    return 0
}

if A_LineFile = A_ScriptFullPath {
    try {
        exitCode := Main(A_Args)
        ExitApp exitCode
    } catch Error as err {
        FileAppend("ERROR: " FormatError(err) "`n", "**")
        ExitApp 1
    }
}
