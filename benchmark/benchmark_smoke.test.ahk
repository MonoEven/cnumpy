#Requires AutoHotkey v2.0
#SingleInstance Off

#Include bench_cnumpy.ahk

AssertTrue(condition, message := "assertion failed") {
    if !condition
        throw Error(message)
}

AssertEqual(expected, actual, message := "values differ") {
    if expected != actual
        throw Error(message ": expected=" expected ", actual=" actual)
}

AssertThrows(callback, message := "expected an exception") {
    try {
        callback.Call()
    } catch Error {
        return
    }
    throw Error(message)
}

AssertThrowsContaining(callback, expectedText, message := "expected a matching exception") {
    try {
        callback.Call()
    } catch Error as err {
        if InStr(err.Message, expectedText, true) != 0
            return
        throw Error(message ": expected message containing=" expectedText ", actual=" err.Message)
    }
    throw Error(message ": no exception was thrown")
}

AssertSoftmaxFullBufferRejects(valueBytes, caseData) {
    try {
        ValidateSoftmaxFullBuffer(valueBytes, caseData)
    } catch Error as err {
        if InStr(err.Message, "softmax full-output", true) != 0
            return
        throw Error(
            "Task8 softmax unsampled interior corruption produced: "
            err.Message)
    }
    throw Error(
        "Task8 softmax unsampled interior corruption was accepted")
}

AssertTrapzFullBufferRejects(valueBytes, caseData) {
    try {
        ValidateTrapzFullBuffer(valueBytes, caseData)
    } catch Error as err {
        if InStr(err.Message, "trapz full-output", true) != 0
            return
        throw Error(
            "Task8 trapz unsampled interior corruption produced: "
            err.Message)
    }
    throw Error("Task8 trapz unsampled interior corruption was accepted")
}

AssertPackbitsFullBufferRejects(valueBytes, caseData) {
    try {
        ValidatePackbitsFullBuffer(valueBytes, caseData)
    } catch Error as err {
        if InStr(err.Message, "packbits full-output", true) != 0
            return
        throw Error(
            "Task8 packbits unsampled interior corruption produced: "
            err.Message)
    }
    throw Error("Task8 packbits unsampled interior corruption was accepted")
}

AssertUnpackbitsFullBufferRejects(valueBytes, caseData) {
    try {
        ValidateUnpackbitsFullBuffer(valueBytes, caseData)
    } catch Error as err {
        if InStr(err.Message, "unpackbits full-output", true) != 0
            return
        throw Error(
            "Task8 unpackbits unsampled interior corruption produced: "
            err.Message)
    }
    throw Error(
        "Task8 unpackbits unsampled interior corruption was accepted")
}

WriteTestFile(path, text) {
    stream := FileOpen(path, "w", "UTF-8-RAW")
    stream.Write(text)
    stream.Close()
}

TestParseArgs() {
    parsed := ParseArgs(["--jobs", "jobs.tsv", "--output", "out.json", "--dll", "cnumpy.dll"])
    AssertEqual(5, parsed["warmups"], "default warmups")
    AssertEqual(15, parsed["samples"], "default samples")
    AssertEqual(12345, parsed["seed"], "default seed")
    seeded := ParseArgs(["--jobs", "jobs.tsv", "--output", "out.json", "--dll", "cnumpy.dll", "--seed", "7"])
    AssertEqual(7, seeded["seed"], "explicit seed")
    AssertThrows((*) => ParseArgs(["--jobs", "x", "--output", "y", "--dll", "z", "--samples", "2"]), "even samples must fail")
    AssertThrows((*) => ParseArgs(["--jobs", "x", "--output", "y", "--dll", "z", "--warmups", "-1"]), "negative warmups must fail")
    AssertThrows((*) => ParseArgs(["--jobs", "x", "--output", "y", "--dll", "z", "--seed", "-1"]), "negative seed must fail")
    AssertThrows((*) => ParseArgs(["--jobs", "x", "--output", "y", "--dll", "z", "--seed", "2147483648"]), "seed outside C Int range must fail")
    AssertThrows((*) => ParseArgs(["--jobs", "x", "--output", "y", "--dll", "z", "--target-sample-ms", "nan"]), "NaN target must fail")
    AssertThrows((*) => ParseArgs(["--jobs", "x", "--jobs", "q", "--output", "y", "--dll", "z"]), "duplicate option must fail")
    AssertThrows((*) => ParseArgs(["--jobs", "x", "--output", "y", "--wat", "z", "--dll", "z"]), "unknown option must fail")
    AssertThrows((*) => ParseArgs(["--jobs", "--output", "y", "--dll", "z"]), "missing option value must fail")
}

TestParseJobs() {
    base := A_Temp "\cnumpy-ahk-parser-" DllCall("kernel32\GetCurrentProcessId", "UInt")
    header := "id`tcategory`toperation`tdtype`tsize`trows`tcols`taxis`n"
    valid := header "zeros/f64/4`tcreation`tzeros`tf64`t4`t0`t0`t-1`n"
    try {
        WriteTestFile(base "-valid.tsv", valid)
        jobs := ParseJobs(base "-valid.tsv")
        AssertEqual(1, jobs.Length, "valid parser case count")
        AssertEqual("zeros", jobs[1]["operation"], "valid parser operation")

        WriteTestFile(base "-empty.tsv", header)
        AssertThrowsContaining((*) => ParseJobs(base "-empty.tsv"), "at least one benchmark case", "header-only jobs must fail clearly")

        WriteTestFile(base "-header.tsv", "wrong`n")
        AssertThrows((*) => ParseJobs(base "-header.tsv"), "bad header must fail")

        WriteTestFile(base "-integer.tsv", header "zeros/f64/4`tcreation`tzeros`tf64`t4x`t0`t0`t-1`n")
        AssertThrows((*) => ParseJobs(base "-integer.tsv"), "bad integer must fail")

        WriteTestFile(base "-duplicate.tsv", valid "zeros/f64/4`tcreation`tzeros`tf64`t4`t0`t0`t-1`n")
        AssertThrows((*) => ParseJobs(base "-duplicate.tsv"), "duplicate id must fail")

        WriteTestFile(base "-semantic.tsv", header "zeros/f64/4`tunary`tzeros`tf64`t4`t0`t0`t-1`n")
        AssertThrows((*) => ParseJobs(base "-semantic.tsv"), "semantic mismatch must fail")

        WriteTestFile(base "-crlf.tsv", StrReplace(valid, "`n", "`r`n"))
        AssertThrows((*) => ParseJobs(base "-crlf.tsv"), "CRLF must fail")
    } finally {
        loop files base "-*.tsv"
            FileDelete(A_LoopFileFullPath)
    }
}

global gTimingCalls := 0

TimingCallback() {
    global gTimingCalls
    gTimingCalls += 1
    return gTimingCalls
}

TestTimer() {
    timing := TimeOperation(TimingCallback, 0, 3, 1000.0)
    AssertEqual(3, timing["samples_ns"].Length, "timer sample count")
    AssertTrue(timing["inner_loops"] > 0, "timer inner loops")
    for sample in timing["samples_ns"]
        AssertTrue(IsFiniteNumber(sample) && sample > 0, "timer sample must be positive and finite")
}

MakeDoubleBuffer(values) {
    bytes := Buffer(values.Length * 8, 0)
    for index, value in values
        NumPut("Double", value, bytes, (index - 1) * 8)
    return bytes
}

MakeNanResultBuffer(finiteValues, nanCount) {
    bytes := Buffer((finiteValues.Length + nanCount) * 8, 0)
    for index, value in finiteValues
        NumPut("Double", value, bytes, (index - 1) * 8)
    loop nanCount
        NumPut("UInt64", 0x7FF8000000000000,
            bytes, (finiteValues.Length + A_Index - 1) * 8)
    return bytes
}

TestRawNanSemanticValidation() {
    first := VectorElement("sorting", 1)
    second := VectorElement("sorting", 3)
    finiteSort := [Min(first, second), Max(first, second)]
    sortFacts := ValidateDeterministicNanResultBuffer(
        MakeNanResultBuffer(finiteSort, 2), 4, "sort_stable_nan", 4)
    AssertEqual(2, sortFacts["nan_count"], "sort raw NaN count")
    AssertEqual(2, sortFacts["first_nan_index"], "sort first raw NaN")
    AssertTrue(sortFacts["trailing_nan"], "sort NaNs must trail")
    AssertTrue(sortFacts["finite_members_exact"],
        "sort finite members must be fully verified")

    wrongFinite := [finiteSort[1] + 1.0, finiteSort[2]]
    AssertThrowsContaining(
        (*) => ValidateDeterministicNanResultBuffer(
            MakeNanResultBuffer(wrongFinite, 2), 4,
            "sort_stable_nan", 4),
        "finite member",
        "wrong sort finite member must fail")
    AssertThrowsContaining(
        (*) => ValidateDeterministicNanResultBuffer(
            MakeDoubleBuffer([finiteSort[1], finiteSort[2], 8192.0, 8192.0]),
            4, "sort_stable_nan", 4),
        "NaN count",
        "finite 8192 sort counterfeit must fail")

    uniqueFacts := ValidateDeterministicNanResultBuffer(
        MakeNanResultBuffer([1.0, 3.0], 1), 3, "unique_nan", 4)
    AssertEqual(1, uniqueFacts["nan_count"], "unique raw NaN count")
    AssertEqual(2, uniqueFacts["first_nan_index"], "unique first raw NaN")
    AssertTrue(uniqueFacts["finite_members_exact"],
        "unique finite members must be fully verified")
}

TestArgsortSemanticValidation() {
    valid := MakeDoubleBuffer([0, 3, 2, 1])
    AssertTrue(ValidateArgsortIndexBuffer(valid, 4), "valid argsort permutation")

    duplicate := MakeDoubleBuffer([0, 3, 2, 2])
    AssertThrowsContaining(
        (*) => ValidateArgsortIndexBuffer(duplicate, 4),
        "duplicate",
        "duplicate argsort index must fail"
    )

    wrongOrder := MakeDoubleBuffer([1, 2, 3, 0])
    AssertThrowsContaining(
        (*) => ValidateArgsortIndexBuffer(wrongOrder, 4),
        "nondecreasing",
        "wrong argsort key order must fail"
    )

    nanValid := MakeDoubleBuffer([3, 1, 0, 2])
    AssertTrue(
        ValidateArgsortIndexBuffer(nanValid, 4, "sorting_nan"),
        "valid NaN argsort permutation"
    )
    nanBeforeFinite := MakeDoubleBuffer([0, 3, 1, 2])
    AssertThrowsContaining(
        (*) => ValidateArgsortIndexBuffer(nanBeforeFinite, 4, "sorting_nan"),
        "NaN",
        "finite argsort keys after NaN must fail"
    )
}

TestPartitionSemanticValidation() {
    validValues := MakeDoubleBuffer([17, 1500, 2366, 3136])
    AssertTrue(
        ValidatePartitionValueBuffer(validValues, 4, 2),
        "valid partition values")
    invalidValues := MakeDoubleBuffer([3136, 1500, 2366, 17])
    AssertThrowsContaining(
        (*) => ValidatePartitionValueBuffer(invalidValues, 4, 2),
        "partition",
        "invalid partition values must fail")

    validIndices := MakeDoubleBuffer([0, 3, 2, 1])
    AssertTrue(
        ValidateArgpartitionIndexBuffer(validIndices, 4, 2),
        "valid argpartition permutation")
    duplicate := MakeDoubleBuffer([0, 3, 2, 2])
    AssertThrowsContaining(
        (*) => ValidateArgpartitionIndexBuffer(duplicate, 4, 2),
        "duplicate",
        "duplicate argpartition index must fail")
    wrongPartition := MakeDoubleBuffer([1, 3, 2, 0])
    AssertThrowsContaining(
        (*) => ValidateArgpartitionIndexBuffer(wrongPartition, 4, 2),
        "partition",
        "wrong argpartition order must fail")
}

TestJsonAndAtomicWrite() {
    escaped := JsonEscape(Chr(34) "\" "`n" Chr(1))
    AssertEqual("\" Chr(34) "\\\n\u0001", escaped, "JSON escaping")
    nan := DllCall("msvcrt\sqrt", "Double", -1.0, "Double")
    AssertThrows((*) => JsonNumber(nan), "non-finite JSON number must fail")

    output := A_Temp "\cnumpy-ahk-atomic-" DllCall("kernel32\GetCurrentProcessId", "UInt") ".json"
    badOutput := A_Temp "\cnumpy-ahk-missing-" A_TickCount "\out.json"
    firstTemp := 0
    secondTemp := 0
    try {
        firstTemp := CreateExclusiveAtomicTemp(output)
        secondTemp := CreateExclusiveAtomicTemp(output)
        SplitPath(output, &outputName, &outputDirectory)
        SplitPath(firstTemp["path"], , &firstDirectory)
        SplitPath(secondTemp["path"], , &secondDirectory)
        AssertEqual(outputDirectory, firstDirectory, "first atomic temp directory")
        AssertEqual(outputDirectory, secondDirectory, "second atomic temp directory")
        AssertTrue(firstTemp["path"] != secondTemp["path"], "exclusive atomic temp paths must differ")
        AssertEqual(firstTemp["handle"], firstTemp["stream"].Handle, "first FileOpen wrapper handle")
        AssertEqual(secondTemp["handle"], secondTemp["stream"].Handle, "second FileOpen wrapper handle")
        AssertTrue(firstTemp["handle"] != secondTemp["handle"], "simultaneous exclusive atomic temp handles must differ")
        CloseExclusiveAtomicTemp(firstTemp, "first test temporary file")
        FileDelete(firstTemp["path"])
        CloseExclusiveAtomicTemp(secondTemp, "second test temporary file")
        FileDelete(secondTemp["path"])

        WriteJsonAtomic(output, "{" Chr(34) "ok" Chr(34) ":1}")
        WriteJsonAtomic(output, "{" Chr(34) "ok" Chr(34) ":2}")
        AssertEqual("{" Chr(34) "ok" Chr(34) ":2}`n", FileRead(output, "UTF-8"), "consecutive atomic JSON contents")
        loop files outputDirectory "\." outputName ".*.tmp"
            throw Error("atomic JSON publication left temporary file: " A_LoopFileFullPath)
        AssertThrows((*) => WriteJsonAtomic(badOutput, "{}"), "invalid output directory must fail")
    } finally {
        for tempFile in [firstTemp, secondTemp] {
            if IsObject(tempFile) {
                if tempFile["handle"] != -1
                    CloseExclusiveAtomicTemp(tempFile, "test temporary cleanup")
                if FileExist(tempFile["path"])
                    FileDelete(tempFile["path"])
            }
        }
        if FileExist(output)
            FileDelete(output)
    }
}

QuoteCommandArgument(value) {
    return Chr(34) StrReplace(value, Chr(34), "\" Chr(34)) Chr(34)
}

CountSubstring(haystack, needle) {
    count := 0
    position := 1
    while found := InStr(haystack, needle, true, position) {
        count += 1
        position := found + StrLen(needle)
    }
    return count
}

BenchmarkPaths() {
    SplitPath(A_LineFile, , &benchmarkDirectory)
    repositoryDirectory := benchmarkDirectory "\.."
    return Map(
        "benchmark", benchmarkDirectory,
        "repository", repositoryDirectory,
        "worker", benchmarkDirectory "\bench_cnumpy.ahk",
        "fixture", benchmarkDirectory "\tests\fixtures\focus-small.tsv",
        "dll", repositoryDirectory "\build\x64\Release\cnumpy_ahk.dll"
    )
}

RunWorkerForTest(jobsPath, sampleCount, label) {
    paths := BenchmarkPaths()
    token := DllCall("kernel32\GetCurrentProcessId", "UInt") "-" A_TickCount "-" label
    output := A_Temp "\cnumpy-ahk-" token ".json"
    stdoutPath := A_Temp "\cnumpy-ahk-" token ".stdout.txt"
    stderrPath := A_Temp "\cnumpy-ahk-" token ".stderr.txt"
    command := (
        QuoteCommandArgument(A_AhkPath)
        " /ErrorStdOut=UTF-8 " QuoteCommandArgument(paths["worker"])
        " --jobs " QuoteCommandArgument(jobsPath)
        " --output " QuoteCommandArgument(output)
        " --dll " QuoteCommandArgument(paths["dll"])
        " --warmups 0 --samples " sampleCount " --target-sample-ms 0.1"
    )
    shellCommand := (
        QuoteCommandArgument(A_ComSpec) " /D /S /C " Chr(34)
        command " 1>" QuoteCommandArgument(stdoutPath) " 2>" QuoteCommandArgument(stderrPath) Chr(34)
    )
    try {
        exitCode := RunWait(shellCommand, paths["repository"], "Hide")
        if exitCode != 0 {
            standardOutput := FileExist(stdoutPath) ? FileRead(stdoutPath, "UTF-8") : ""
            standardError := FileExist(stderrPath) ? FileRead(stderrPath, "UTF-8") : ""
            throw Error(label " worker exited " exitCode "; stdout=" standardOutput "; stderr=" standardError)
        }
        if !FileExist(output)
            throw Error(label " worker exited zero without creating output")
        return FileRead(output, "UTF-8")
    } finally {
        for path in [output, stdoutPath, stderrPath] {
            if FileExist(path)
                FileDelete(path)
        }
    }
}

TestEightCaseWorker() {
    paths := BenchmarkPaths()
    document := RunWorkerForTest(paths["fixture"], 3, "eight")
    ids := [
        "bridge/property_call",
        "zeros/f64/8",
        "argmax/f64/8",
        "cumsum/f64/8",
        "argsort/f64/8",
        "flatten/f64/8",
        "reshape/f64/8",
        "concatenate/f64/2x2/axis0"
    ]
    for id in ids
        AssertTrue(InStr(document, Chr(34) "id" Chr(34) ":" JsonString(id), true) != 0, "missing 8-case id " id)
    AssertEqual(8, CountSubstring(document, Chr(34) "retained_bytes" Chr(34) ":0"), "8-case retained byte count")
    AssertEqual(8, CountSubstring(document, Chr(34) "samples_ns" Chr(34) ":["), "8-case sample array count")
}

TestExpm1Worker() {
    jobsPath := A_Temp "\cnumpy-ahk-expm1-" DllCall("kernel32\GetCurrentProcessId", "UInt") ".tsv"
    try {
        text := "id`tcategory`toperation`tdtype`tsize`trows`tcols`taxis`n"
        text .= "expm1/f64/8`tunary`texpm1`tf64`t8`t0`t0`t-1`n"
        WriteTestFile(jobsPath, text)
        parsed := ParseJobs(jobsPath)
        AssertEqual(1, parsed.Length, "expm1 job count")
        document := RunWorkerForTest(jobsPath, 3, "expm1")
        AssertTrue(
            InStr(document, Chr(34) "id" Chr(34) ":" JsonString("expm1/f64/8"), true) != 0,
            "missing expm1 case id"
        )
        AssertEqual(1, CountSubstring(document, Chr(34) "retained_bytes" Chr(34) ":0"), "expm1 retained byte count")
        AssertEqual(1, CountSubstring(document, Chr(34) "samples_ns" Chr(34) ":["), "expm1 sample array count")
    } finally {
        if FileExist(jobsPath)
            FileDelete(jobsPath)
    }
}

TestCholeskyWorker() {
    jobsPath := A_Temp "\cnumpy-ahk-cholesky-" DllCall(
        "kernel32\GetCurrentProcessId", "UInt") ".tsv"
    try {
        text := "id`tcategory`toperation`tdtype`tsize`trows`tcols`taxis`n"
        text .= "cholesky/f64/4x4`tlinalg`tcholesky`tf64`t16`t4`t4`t-1`n"
        WriteTestFile(jobsPath, text)
        parsed := ParseJobs(jobsPath)
        AssertEqual(1, parsed.Length, "cholesky job count")
        document := RunWorkerForTest(jobsPath, 3, "cholesky")
        AssertTrue(InStr(document,
            Chr(34) "id" Chr(34) ":" JsonString("cholesky/f64/4x4"),
            true) != 0, "missing cholesky case id")
        AssertEqual(1, CountSubstring(document,
            Chr(34) "retained_bytes" Chr(34) ":0"),
            "cholesky retained byte count")
        AssertEqual(1, CountSubstring(document,
            Chr(34) "samples_ns" Chr(34) ":["),
            "cholesky sample array count")
    } finally {
        if FileExist(jobsPath)
            FileDelete(jobsPath)
    }
}

TestFunctionalCallbackWorkers() {
    jobsPath := A_Temp "\cnumpy-ahk-functional-callback-"
        DllCall("kernel32\GetCurrentProcessId", "UInt") ".tsv"
    try {
        text := "id`tcategory`toperation`tdtype`tsize`trows`tcols`taxis`n"
        operations := [
            "callback_apply_along_axis_legacy",
            "callback_apply_along_axis_v2",
            "callback_fromfunction_legacy",
            "callback_fromfunction_v2",
            "callback_fromiter_legacy",
            "callback_fromiter_v2",
            "callback_vectorize_legacy",
            "callback_vectorize_v2"
        ]
        for operation in operations {
            if InStr(operation, "apply_along_axis", true) != 0
                text .= operation "/f64/17x17/axis1"
                    . "`tfunctional_callback`t" operation
                    . "`tf64`t289`t17`t17`t1`n"
            else
                text .= operation "/f64/513`tfunctional_callback`t"
                    . operation "`tf64`t513`t0`t0`t-1`n"
        }
        WriteTestFile(jobsPath, text)
        parsed := ParseJobs(jobsPath)
        AssertEqual(8, parsed.Length,
            "functional callback benchmark job count")
        document := RunWorkerForTest(
            jobsPath, 3, "functional-callback")
        AssertEqual(8, CountSubstring(document,
            Chr(34) "retained_bytes" Chr(34) ":0"),
            "functional callback retained byte count")
        AssertEqual(3, CountSubstring(document,
            Chr(34) "callback_invocations" Chr(34) ":513,"
            . Chr(34) "logical_elements" Chr(34) ":513"),
            "legacy vector callback evidence")
        AssertEqual(3, CountSubstring(document,
            Chr(34) "callback_invocations" Chr(34) ":3,"
            . Chr(34) "logical_elements" Chr(34) ":513"),
            "bulk vector callback evidence")
        AssertEqual(1, CountSubstring(document,
            Chr(34) "callback_invocations" Chr(34) ":17,"
            . Chr(34) "logical_elements" Chr(34) ":17"),
            "legacy line callback evidence")
        AssertEqual(1, CountSubstring(document,
            Chr(34) "callback_invocations" Chr(34) ":1,"
            . Chr(34) "logical_elements" Chr(34) ":17"),
            "bulk line callback evidence")
    } finally {
        if FileExist(jobsPath)
            FileDelete(jobsPath)
    }
}

TestTask9LinalgWorkers() {
    jobsPath := A_Temp "\cnumpy-ahk-task9-linalg-" DllCall(
        "kernel32\GetCurrentProcessId", "UInt") ".tsv"
    try {
        text := "id`tcategory`toperation`tdtype`tsize`trows`tcols`taxis`n"
        for operation in ["einsum", "eig", "svd", "solve", "lstsq"]
            text .= operation "/f64/4x4`tlinalg`t" operation
                . "`tf64`t16`t4`t4`t-1`n"
        WriteTestFile(jobsPath, text)
        parsed := ParseJobs(jobsPath)
        AssertEqual(5, parsed.Length, "Task9 linalg job count")
        document := RunWorkerForTest(jobsPath, 3, "task9-linalg")
        AssertEqual(5, CountSubstring(document,
            Chr(34) "retained_bytes" Chr(34) ":0"),
            "Task9 linalg retained byte count")
        for caseData in parsed
            AssertTrue(InStr(document,
                Chr(34) "id" Chr(34) ":" JsonString(caseData["id"]),
                true) != 0, "missing Task9 linalg id " caseData["id"])
    } finally {
        if FileExist(jobsPath)
            FileDelete(jobsPath)
    }
}

TestTask9LinalgCounterfeitRejection() {
    paths := BenchmarkPaths()
    moduleHandle := 0
    initAttempted := false
    try {
        moduleHandle := DllCall(
            "kernel32\LoadLibraryW", "Str", paths["dll"], "Ptr")
        if moduleHandle = 0
            throw OSError(A_LastError, "Task9 counterfeit LoadLibraryW failed")
        initAttempted := true
        initStatus := DllCall(paths["dll"] "\cnp_ahk_init", "Int")
        if initStatus != 0
            throw Error("Task9 counterfeit cnp_ahk_init returned " initStatus)
        Numpy.DllPath := paths["dll"]
        Numpy.DllHandle := moduleHandle
        Numpy.Initialized := true
        before := DllCall(
            paths["dll"] "\cnp_ahk_allocated_memory", "Int64")

        for operation in ["einsum", "eig", "svd", "solve", "lstsq"] {
            caseData := Map(
                "id", operation "/f64/4x4",
                "category", "linalg",
                "operation", operation,
                "dtype", "f64",
                "size", 16,
                "rows", 4,
                "cols", 4,
                "axis", -1)
            prepared := 0
            result := 0
            try {
                prepared := PrepareCase(caseData, paths["dll"], 12345)
                result := prepared.RawInvoke.Call()
                if prepared.ReturnsResultHandles {
                    prepared.RequireStatus(result,
                        operation " counterfeit setup")
                    corruptHandle := NumGet(
                        prepared.ResultHandles,
                        (operation = "eig" ? A_PtrSize : 0), "Ptr")
                    corruptIndex := operation = "eig" ? 6 : 1
                    current := DllCall(paths["dll"] "\cnp_ahk_get_item",
                        "Ptr", corruptHandle,
                        "Int64", corruptIndex, "Double")
                    status := DllCall(paths["dll"] "\cnp_ahk_set_item",
                        "Ptr", corruptHandle, "Int64", corruptIndex,
                        "Double", current + 1.0, "Int")
                    AssertEqual(0, status,
                        operation " counterfeit mutation status")
                    AssertThrowsContaining(
                        (*) => prepared.Task9ResultValidationSignature(),
                        "Task9", operation " counterfeit must be rejected")
                } else {
                    if result = 0
                        throw Error(operation " counterfeit setup returned null")
                    corruptIndex := operation = "einsum" ? 5 : 1
                    current := DllCall(paths["dll"] "\cnp_ahk_get_item",
                        "Ptr", result, "Int64", corruptIndex, "Double")
                    status := DllCall(paths["dll"] "\cnp_ahk_set_item",
                        "Ptr", result, "Int64", corruptIndex,
                        "Double", current + 1.0, "Int")
                    AssertEqual(0, status,
                        operation " counterfeit mutation status")
                    if operation = "einsum"
                        callback := (*) => ValidateTask9EinsumHandle(
                            paths["dll"], result,
                            prepared.OwnedHandles[1], 4)
                    else
                        callback := (*) => ValidateTask9SolveHandle(
                            paths["dll"], result,
                            prepared.OwnedHandles[1],
                            prepared.OwnedHandles[2], 4)
                    AssertThrowsContaining(callback, "Task9",
                        operation " counterfeit must be rejected")
                }
            } finally {
                if IsObject(prepared) {
                    if prepared.ReturnsResultHandles
                        prepared.FreeResultHandles(
                            operation " counterfeit results", true)
                    else if result != 0
                        FreeArrayHandle(paths["dll"], result,
                            operation " counterfeit result")
                    prepared.Close()
                }
            }
        }
        after := DllCall(
            paths["dll"] "\cnp_ahk_allocated_memory", "Int64")
        AssertEqual(before, after,
            "Task9 counterfeit setup/validation retained bytes")
    } finally {
        if initAttempted {
            cleanupStatus := DllCall(paths["dll"] "\cnp_ahk_cleanup", "Int")
            if cleanupStatus != 0
                throw Error(
                    "Task9 counterfeit cnp_ahk_cleanup returned " cleanupStatus)
        }
        Numpy.Initialized := false
        Numpy.ProcCache := Map()
        Numpy.ProcCacheHandle := 0
        Numpy.DllHandle := 0
        if moduleHandle != 0 && !DllCall(
                "kernel32\FreeLibrary", "Ptr", moduleHandle, "Int")
            throw OSError(
                A_LastError, "Task9 counterfeit FreeLibrary failed")
    }
}

TestAxis0IndexingBlockWorkers() {
    jobsPath := A_Temp "\cnumpy-ahk-index-axis0-" DllCall(
        "kernel32\GetCurrentProcessId", "UInt") ".tsv"
    moduleHandle := 0
    initAttempted := false
    try {
        text := "id`tcategory`toperation`tdtype`tsize`trows`tcols`taxis`n"
        operations := [
            "take_axis0_block",
            "take_axis0_strided",
            "compress_axis0_block",
            "compress_axis0_strided"
        ]
        for operation in operations {
            text .= (
                operation "/f64/4x4/axis0`tindexing`t" operation
                "`tf64`t16`t4`t4`t0`n"
            )
        }
        WriteTestFile(jobsPath, text)
        parsed := ParseJobs(jobsPath)
        AssertEqual(4, parsed.Length, "axis0 indexing job count")

        paths := BenchmarkPaths()
        moduleHandle := DllCall(
            "kernel32\LoadLibraryW", "Str", paths["dll"], "Ptr")
        if moduleHandle = 0
            throw OSError(A_LastError, "axis0 smoke LoadLibraryW failed")
        initAttempted := true
        initStatus := DllCall(paths["dll"] "\cnp_ahk_init", "Int")
        if initStatus != 0
            throw Error("axis0 smoke cnp_ahk_init returned " initStatus)
        Numpy.DllPath := paths["dll"]
        Numpy.DllHandle := moduleHandle
        Numpy.Initialized := true
        before := DllCall(paths["dll"] "\cnp_ahk_allocated_memory", "Int64")
        signatures := []
        for caseData in parsed {
            prepared := 0
            try {
                prepared := PrepareCase(caseData, paths["dll"], 12345)
                flags := DllCall(paths["dll"] "\cnp_ahk_flags",
                    "Ptr", prepared.IndexingSourceHandle, "UInt")
                isCContiguous :=
                    (flags & Numpy.ARRAY_C_CONTIGUOUS) != 0
                isFContiguous :=
                    (flags & Numpy.ARRAY_F_CONTIGUOUS) != 0
                if SubStr(caseData["operation"], -7) = "strided" {
                    AssertTrue(!isCContiguous,
                        caseData["operation"] " source must not be C-contiguous")
                    AssertTrue(isFContiguous,
                        caseData["operation"] " source must be F-contiguous")
                } else {
                    AssertTrue(isCContiguous,
                        caseData["operation"] " source must be C-contiguous")
                }
                signatures.Push(ValidationToJson(prepared.ValidationSignature()))
            } finally {
                if IsObject(prepared)
                    prepared.Close()
            }
        }
        after := DllCall(paths["dll"] "\cnp_ahk_allocated_memory", "Int64")
        AssertEqual(before, after, "axis0 indexing setup/cleanup retained bytes")
        for index in [2, 3, 4]
            AssertEqual(signatures[1], signatures[index],
                "axis0 indexing logical validation equality")

        document := RunWorkerForTest(jobsPath, 3, "index-axis0")
        AssertEqual(4, CountSubstring(document,
            Chr(34) "retained_bytes" Chr(34) ":0"),
            "axis0 indexing retained byte count")
        for caseData in parsed
            AssertTrue(InStr(document,
                Chr(34) "id" Chr(34) ":" JsonString(caseData["id"]),
                true) != 0, "missing axis0 indexing id " caseData["id"])
    } finally {
        if FileExist(jobsPath)
            FileDelete(jobsPath)
        if initAttempted {
            cleanupStatus := DllCall(
                paths["dll"] "\cnp_ahk_cleanup", "Int")
            if cleanupStatus != 0
                throw Error("axis0 smoke cnp_ahk_cleanup returned " cleanupStatus)
        }
        Numpy.Initialized := false
        Numpy.ProcCache := Map()
        Numpy.ProcCacheHandle := 0
        Numpy.DllHandle := 0
        if moduleHandle != 0 && !DllCall(
                "kernel32\FreeLibrary", "Ptr", moduleHandle, "Int")
            throw OSError(A_LastError, "axis0 smoke FreeLibrary failed")
    }
}

TestTask8SoftmaxWorkers() {
    jobsPath := A_Temp "\cnumpy-ahk-task8-softmax-" DllCall(
        "kernel32\GetCurrentProcessId", "UInt") ".tsv"
    moduleHandle := 0
    initAttempted := false
    try {
        text := "id`tcategory`toperation`tdtype`tsize`trows`tcols`taxis`n"
        text .= "softmax/f64/8`tmisc_axis`tsoftmax`tf64`t8`t0`t0`t-1`n"
        text .= (
            "softmax_axis_last/f64/4x4/axis1`tmisc_axis`t"
            "softmax_axis_last`tf64`t16`t4`t4`t1`n"
        )
        text .= (
            "softmax_axis0_strided/f64/4x4/axis0`tmisc_axis`t"
            "softmax_axis0_strided`tf64`t16`t4`t4`t0`n"
        )
        text .= (
            "log_softmax/f64/8`tmisc_axis`tlog_softmax`tf64`t8`t0`t0`t-1`n"
        )
        text .= (
            "log_softmax_axis_last/f64/4x4/axis1`tmisc_axis`t"
            "log_softmax_axis_last`tf64`t16`t4`t4`t1`n"
        )
        text .= (
            "log_softmax_axis0_strided/f64/4x4/axis0`tmisc_axis`t"
            "log_softmax_axis0_strided`tf64`t16`t4`t4`t0`n"
        )
        text .= "trapz/f64/8`tmisc_axis`ttrapz`tf64`t8`t0`t0`t-1`n"
        text .= (
            "trapz_axis_last/f64/4x4/axis1`tmisc_axis`t"
            "trapz_axis_last`tf64`t16`t4`t4`t1`n"
        )
        text .= (
            "trapz_axis0_strided/f64/4x4/axis0`tmisc_axis`t"
            "trapz_axis0_strided`tf64`t16`t4`t4`t0`n"
        )
        text .= "packbits/u8/16`tmisc_axis`tpackbits`tu8`t16`t0`t0`t-1`n"
        text .= (
            "packbits_axis_last/u8/8x8/axis1`tmisc_axis`t"
            "packbits_axis_last`tu8`t64`t8`t8`t1`n"
        )
        text .= (
            "packbits_axis0_strided/u8/8x8/axis0`tmisc_axis`t"
            "packbits_axis0_strided`tu8`t64`t8`t8`t0`n"
        )
        text .= "unpackbits/u8/8`tmisc_axis`tunpackbits`tu8`t8`t0`t0`t-1`n"
        text .= (
            "unpackbits_axis_last/u8/8x8/axis1`tmisc_axis`t"
            "unpackbits_axis_last`tu8`t64`t8`t8`t1`n"
        )
        text .= (
            "unpackbits_axis0_strided/u8/8x8/axis0`tmisc_axis`t"
            "unpackbits_axis0_strided`tu8`t64`t8`t8`t0`n"
        )
        WriteTestFile(jobsPath, text)
        parsed := ParseJobs(jobsPath)
        AssertEqual(15, parsed.Length, "Task8 misc-axis job count")

        paths := BenchmarkPaths()
        moduleHandle := DllCall(
            "kernel32\LoadLibraryW", "Str", paths["dll"], "Ptr")
        if moduleHandle = 0
            throw OSError(A_LastError, "Task8 softmax LoadLibraryW failed")
        initAttempted := true
        initStatus := DllCall(paths["dll"] "\cnp_ahk_init", "Int")
        if initStatus != 0
            throw Error("Task8 softmax cnp_ahk_init returned " initStatus)
        Numpy.DllPath := paths["dll"]
        Numpy.DllHandle := moduleHandle
        Numpy.Initialized := true

        before := DllCall(
            paths["dll"] "\cnp_ahk_allocated_memory", "Int64")
        for caseData in parsed {
            prepared := 0
            try {
                prepared := PrepareCase(caseData, paths["dll"], 12345)
                flags := DllCall(paths["dll"] "\cnp_ahk_flags",
                    "Ptr", prepared.AxisSourceHandle, "UInt")
                if InStr(caseData["operation"], "axis0_strided", true) != 0 {
                    AssertTrue((flags & Numpy.ARRAY_C_CONTIGUOUS) = 0,
                        "Task8 axis0 source must not be C-contiguous")
                    AssertTrue((flags & Numpy.ARRAY_F_CONTIGUOUS) != 0,
                        "Task8 axis0 source must be F-contiguous")
                }
                signature := prepared.ValidationSignature()
                operation := caseData["operation"]
                expectedValidationSize := operation = "trapz" ? 1
                    : (InStr(operation, "trapz_axis", true) = 1
                        ? (caseData["axis"] = 0
                            ? caseData["cols"] : caseData["rows"])
                        : (operation = "packbits"
                            ? Ceil(caseData["size"] / 8)
                            : (InStr(operation, "packbits_axis", true) = 1
                                ? (caseData["axis"] = 1
                                    ? caseData["rows"]
                                        * Ceil(caseData["cols"] / 8)
                                    : Ceil(caseData["rows"] / 8)
                                        * caseData["cols"])
                                : (InStr(operation, "unpackbits", true) = 1
                                    ? caseData["size"] * 8
                                    : caseData["size"]))))
                AssertEqual(expectedValidationSize, signature["size"],
                    caseData["operation"] " validation size")
                if caseData["operation"] = "softmax"
                        || caseData["operation"] = "log_softmax" {
                    rawResult := prepared.RawInvoke.Call()
                    try {
                        rawValues := Buffer(caseData["size"] * 8, 0)
                        copied := DllCall(
                            paths["dll"] "\cnp_ahk_get_doubles",
                            "Ptr", rawResult, "Ptr", rawValues.Ptr,
                            "Int64", caseData["size"], "Int64")
                        AssertEqual(caseData["size"], copied,
                            "Task8 softmax raw result copy count")
                        AssertTrue(
                            ValidateSoftmaxFullBuffer(rawValues, caseData),
                            "Task8 softmax valid raw buffer")
                        NumPut("Double",
                            NumGet(rawValues, 8, "Double") + 1e-6,
                            rawValues, 8)
                        AssertSoftmaxFullBufferRejects(
                            rawValues, caseData)
                    } finally {
                        FreeArrayHandle(paths["dll"], rawResult,
                            "Task8 softmax raw validation result")
                    }
                }
                if caseData["operation"] = "trapz_axis_last" {
                    rawResult := prepared.RawInvoke.Call()
                    try {
                        rawValues := Buffer(caseData["rows"] * 8, 0)
                        copied := DllCall(
                            paths["dll"] "\cnp_ahk_get_doubles",
                            "Ptr", rawResult, "Ptr", rawValues.Ptr,
                            "Int64", caseData["rows"], "Int64")
                        AssertEqual(caseData["rows"], copied,
                            "Task8 trapz raw result copy count")
                        AssertTrue(
                            ValidateTrapzFullBuffer(rawValues, caseData),
                            "Task8 trapz valid raw buffer")
                        NumPut("Double",
                            NumGet(rawValues, 8, "Double") + 1e-6,
                            rawValues, 8)
                        AssertTrapzFullBufferRejects(rawValues, caseData)
                    } finally {
                        FreeArrayHandle(paths["dll"], rawResult,
                            "Task8 trapz raw validation result")
                    }
                }
                if caseData["operation"] = "packbits_axis_last" {
                    rawResult := prepared.RawInvoke.Call()
                    outputSize := caseData["rows"]
                        * Ceil(caseData["cols"] / 8)
                    try {
                        rawValues := Buffer(outputSize * 8, 0)
                        copied := DllCall(
                            paths["dll"] "\cnp_ahk_get_doubles",
                            "Ptr", rawResult, "Ptr", rawValues.Ptr,
                            "Int64", outputSize, "Int64")
                        AssertEqual(outputSize, copied,
                            "Task8 packbits raw result copy count")
                        AssertTrue(
                            ValidatePackbitsFullBuffer(rawValues, caseData),
                            "Task8 packbits valid raw buffer")
                        NumPut("Double",
                            NumGet(rawValues, 8, "Double") + 1.0,
                            rawValues, 8)
                        AssertPackbitsFullBufferRejects(rawValues, caseData)
                    } finally {
                        FreeArrayHandle(paths["dll"], rawResult,
                            "Task8 packbits raw validation result")
                    }
                }
                if caseData["operation"] = "unpackbits_axis_last" {
                    rawResult := prepared.RawInvoke.Call()
                    outputSize := caseData["size"] * 8
                    try {
                        rawValues := Buffer(outputSize * 8, 0)
                        copied := DllCall(
                            paths["dll"] "\cnp_ahk_get_doubles",
                            "Ptr", rawResult, "Ptr", rawValues.Ptr,
                            "Int64", outputSize, "Int64")
                        AssertEqual(outputSize, copied,
                            "Task8 unpackbits raw result copy count")
                        AssertTrue(
                            ValidateUnpackbitsFullBuffer(
                                rawValues, caseData),
                            "Task8 unpackbits valid raw buffer")
                        NumPut("Double",
                            1.0 - NumGet(rawValues, 8, "Double"),
                            rawValues, 8)
                        AssertUnpackbitsFullBufferRejects(
                            rawValues, caseData)
                    } finally {
                        FreeArrayHandle(paths["dll"], rawResult,
                            "Task8 unpackbits raw validation result")
                    }
                }
            } finally {
                if IsObject(prepared)
                    prepared.Close()
            }
        }
        after := DllCall(
            paths["dll"] "\cnp_ahk_allocated_memory", "Int64")
        AssertEqual(before, after,
            "Task8 softmax setup/validation/cleanup retained bytes")

        document := RunWorkerForTest(jobsPath, 3, "task8-softmax")
        AssertEqual(15, CountSubstring(document,
            Chr(34) "retained_bytes" Chr(34) ":0"),
            "Task8 misc-axis retained byte count")
        for caseData in parsed
            AssertTrue(InStr(document,
                Chr(34) "id" Chr(34) ":" JsonString(caseData["id"]),
                true) != 0, "missing Task8 softmax id " caseData["id"])
    } finally {
        if FileExist(jobsPath)
            FileDelete(jobsPath)
        if initAttempted {
            cleanupStatus := DllCall(
                paths["dll"] "\cnp_ahk_cleanup", "Int")
            if cleanupStatus != 0
                throw Error(
                    "Task8 softmax cnp_ahk_cleanup returned " cleanupStatus)
        }
        Numpy.Initialized := false
        Numpy.ProcCache := Map()
        Numpy.ProcCacheHandle := 0
        Numpy.DllHandle := 0
        if moduleHandle != 0 && !DllCall(
                "kernel32\FreeLibrary", "Ptr", moduleHandle, "Int")
            throw OSError(
                A_LastError, "Task8 softmax FreeLibrary failed")
    }
}

TestWeightedChoiceWorker() {
    caseData := Map(
        "id", "choice_weighted/f64/8",
        "category", "random",
        "operation", "choice_weighted",
        "dtype", "f64",
        "size", 8,
        "rows", 0,
        "cols", 0,
        "axis", -1
    )
    completeOutput := Buffer(8 * 8, 0)
    loop 8
        NumPut("Double", 193.0, completeOutput, (A_Index - 1) * 8)
    AssertTrue(ValidateWeightedChoiceFullBuffer(completeOutput, caseData),
        "weighted choice must validate every deterministic sample")
    NumPut("Double", 192.0, completeOutput, 8)
    AssertThrowsContaining(
        (*) => ValidateWeightedChoiceFullBuffer(completeOutput, caseData),
        "weighted choice full-output",
        "weighted choice validation must reject an unsampled counterfeit")

    jobsPath := A_Temp "\cnumpy-ahk-choice-" DllCall(
        "kernel32\GetCurrentProcessId", "UInt") ".tsv"
    try {
        WriteTestFile(jobsPath,
            "id`tcategory`toperation`tdtype`tsize`trows`tcols`taxis`n"
            "choice_weighted/f64/8`trandom`tchoice_weighted`tf64`t8`t0`t0`t-1`n")
        document := RunWorkerForTest(jobsPath, 3, "weighted-choice")
        AssertTrue(InStr(document,
            Chr(34) "id" Chr(34) ":" JsonString(
                "choice_weighted/f64/8"), true) != 0,
            "weighted choice worker document must include its case id")
        AssertEqual(1, CountSubstring(document,
            Chr(34) "retained_bytes" Chr(34) ":0"),
            "weighted choice worker must retain zero bytes")
    } finally {
        if FileExist(jobsPath)
            FileDelete(jobsPath)
    }
}

TestFftComplexBenchmarkContract() {
    paths := BenchmarkPaths()
    moduleHandle := 0
    initAttempted := false
    prepared := 0
    rawResult := 0
    try {
        moduleHandle := DllCall(
            "kernel32\LoadLibraryW", "Str", paths["dll"], "Ptr")
        if moduleHandle = 0
            throw OSError(A_LastError, "FFT smoke LoadLibraryW failed")
        initAttempted := true
        initStatus := DllCall(paths["dll"] "\cnp_ahk_init", "Int")
        if initStatus != 0
            throw Error("FFT smoke cnp_ahk_init returned " initStatus)
        Numpy.DllPath := paths["dll"]
        Numpy.DllHandle := moduleHandle
        Numpy.Initialized := true

        before := DllCall(
            paths["dll"] "\cnp_ahk_allocated_memory", "Int64")
        caseData := Map(
            "id", "fft/f64/8",
            "category", "fft",
            "operation", "fft",
            "dtype", "f64",
            "size", 8,
            "rows", 0,
            "cols", 0,
            "axis", -1)
        prepared := PrepareCase(caseData, paths["dll"], 12345)

        rawResult := prepared.RawInvoke.Call()
        if rawResult = 0
            throw Error("FFT raw timed invocation returned null")
        AssertEqual(1, DllCall(paths["dll"] "\cnp_ahk_ndim",
            "Ptr", rawResult, "Int"),
            "FFT timed result must remain a native one-dimensional array")
        AssertEqual(16, DllCall(paths["dll"] "\cnp_ahk_dtype",
            "Ptr", rawResult, "Int"),
            "FFT timed result must remain native complex128")
        rawShape := Buffer(8, 0)
        AssertEqual(1, DllCall(paths["dll"] "\cnp_ahk_shape",
            "Ptr", rawResult, "Ptr", rawShape.Ptr, "Int"),
            "FFT timed result shape read")
        AssertEqual(8, NumGet(rawShape, 0, "Int64"),
            "FFT timed result length")
        FreeArrayHandle(paths["dll"], rawResult, "FFT raw timed result")
        rawResult := 0

        signature := prepared.ValidationSignature()
        AssertEqual(2, signature["shape"].Length,
            "FFT validation projection rank")
        AssertEqual(8, signature["shape"][1],
            "FFT validation projection row count")
        AssertEqual(2, signature["shape"][2],
            "FFT validation projection real/imag columns")
        AssertEqual(16, signature["size"],
            "FFT validation projection value count")
        AssertEqual("f64", signature["logical_dtype"],
            "FFT validation projection logical dtype")

        prepared.TimedInvoke()
        prepared.Close()
        prepared := 0
        after := DllCall(
            paths["dll"] "\cnp_ahk_allocated_memory", "Int64")
        AssertEqual(before, after,
            "FFT raw/validation/timed paths retained bytes")
    } finally {
        if rawResult != 0
            FreeArrayHandle(paths["dll"], rawResult,
                "FFT raw timed result cleanup")
        if IsObject(prepared)
            prepared.Close()
        if initAttempted {
            cleanupStatus := DllCall(paths["dll"] "\cnp_ahk_cleanup", "Int")
            if cleanupStatus != 0
                throw Error("FFT smoke cnp_ahk_cleanup returned " cleanupStatus)
        }
        Numpy.Initialized := false
        Numpy.ProcCache := Map()
        Numpy.ProcCacheHandle := 0
        Numpy.DllHandle := 0
        if moduleHandle != 0 && !DllCall(
                "kernel32\FreeLibrary", "Ptr", moduleHandle, "Int")
            throw OSError(A_LastError, "FFT smoke FreeLibrary failed")
    }
}

CanonicalOneHundredEightJobs() {
    text := "id`tcategory`toperation`tdtype`tsize`trows`tcols`taxis`n"
    groups := [
        ["creation", ["zeros", "ones", "arange", "random", "linspace"]],
        ["unary", ["sin", "cos", "exp", "expm1", "sqrt", "log", "absolute", "floor", "tanh", "angle", "real", "imag", "real_if_close"]],
        ["binary", ["add", "subtract", "multiply", "divide", "divmod", "power", "float_power", "heaviside", "maximum", "minimum", "fmax", "fmin"]],
        ["logical", [
            "logical_and", "logical_or", "logical_xor", "logical_not",
            "isnan", "isinf", "isfinite", "signbit",
            "iscomplexobj", "isrealobj", "isscalar"
        ]],
        ["bitwise", [
            "bitwise_and", "bitwise_or", "bitwise_xor", "invert",
            "left_shift", "right_shift"
        ]],
        ["integer", ["gcd", "lcm"]],
        ["signal", ["convolve", "correlate"]],
        ["comparison", ["allclose"]],
        ["reduction", ["sum", "mean", "average", "std", "max", "min", "argmax", "cumsum", "prod"]],
        ["sorting", [
            "sort", "argsort",
            "sort_mergesort", "argsort_mergesort",
            "sort_heapsort", "argsort_heapsort",
            "sort_stable", "argsort_stable",
            "sort_stable_nan", "argsort_stable_nan",
            "partition", "argpartition",
            "partition_nan", "argpartition_nan",
            "searchsorted", "searchsorted_right",
            "digitize", "digitize_decreasing", "lexsort",
            "msort", "sort_complex"
        ]],
        ["shape", [
            "copy", "reshape", "flatten",
            "atleast_1d", "atleast_2d", "atleast_3d"
        ]],
        ["indexing", ["take", "compress"]],
        ["preallocated", ["add_into", "sqrt_into", "cumsum_into"]],
        ["pipeline", ["pipeline_separate", "pipeline_batch"]]
    ]
    for group in groups {
        category := group[1]
        for operation in group[2] {
            dtype := (category = "bitwise" || category = "integer")
                ? "i64" : "f64"
            text .= operation "/" dtype "/8`t" category "`t" operation "`t" dtype "`t8`t0`t0`t-1`n"
        }
    }
    for operation in ["matmul", "dot", "det", "inv", "norm", "solve"]
        text .= operation "/f64/2x2`tlinalg`t" operation "`tf64`t4`t2`t2`t-1`n"
    text .= "transpose_copy/f64/2x2`tshape`ttranspose_copy`tf64`t4`t2`t2`t-1`n"
    text .= "concatenate/f64/2x2/axis0`tshape`tconcatenate`tf64`t4`t2`t2`t0`n"
    text .= "fft/f64/8`tfft`tfft`tf64`t8`t0`t0`t-1`n"
    text .= "bridge/property_call`tbridge`tproperty_call`tf64`t1`t0`t0`t-1`n"
    text .= "bridge/nbytes_cached`tbridge`tnbytes_cached`tf64`t1`t0`t0`t-1`n"
    text .= "bridge/c_contiguous_cached`tbridge`tc_contiguous_cached`tf64`t1`t0`t0`t-1`n"
    text .= "bridge/f_contiguous_cached`tbridge`tf_contiguous_cached`tf64`t1`t0`t0`t-1`n"
    return text
}

TestOneHundredEightCaseWorker() {
    jobsPath := A_Temp "\cnumpy-ahk-108-" DllCall("kernel32\GetCurrentProcessId", "UInt") ".tsv"
    try {
        WriteTestFile(jobsPath, CanonicalOneHundredEightJobs())
        parsed := ParseJobs(jobsPath)
        AssertEqual(108, parsed.Length, "canonical 108 job count")
        document := RunWorkerForTest(jobsPath, 1, "onehundredeight")
        AssertEqual(108, CountSubstring(document, Chr(34) "retained_bytes" Chr(34) ":0"), "108-case retained byte count")
        AssertEqual(108, CountSubstring(document, Chr(34) "samples_ns" Chr(34) ":["), "108-case sample array count")
        for caseData in parsed
            AssertTrue(InStr(document, Chr(34) "id" Chr(34) ":" JsonString(caseData["id"]), true) != 0, "missing 108-case id " caseData["id"])
    } finally {
        if FileExist(jobsPath)
            FileDelete(jobsPath)
    }
}

RunUnitTests() {
    TestParseArgs()
    TestParseJobs()
    TestTimer()
    TestArgsortSemanticValidation()
    TestPartitionSemanticValidation()
    TestRawNanSemanticValidation()
    TestJsonAndAtomicWrite()
    TestEightCaseWorker()
    TestExpm1Worker()
    TestCholeskyWorker()
    TestFunctionalCallbackWorkers()
    TestTask9LinalgWorkers()
    TestTask9LinalgCounterfeitRejection()
    TestAxis0IndexingBlockWorkers()
    TestTask8SoftmaxWorkers()
    TestWeightedChoiceWorker()
    TestFftComplexBenchmarkContract()
    TestOneHundredEightCaseWorker()
}

try {
    RunUnitTests()
    ExitApp 0
} catch Error as err {
    FileAppend("TEST FAILURE: " FormatError(err) "`n", "**")
    ExitApp 1
}
