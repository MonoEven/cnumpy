#Requires AutoHotkey v2.0
#Include ..\ahk\numpy.ahk

; Verification harness for the tutorial examples. Prints deterministic
; lines to stdout; compare against the outputs published in tutorial/*.html.

Out(label, text) {
    FileAppend(label "=" text "`n", "*")
}

Main() {
    Numpy.DllPath := A_ScriptDir "\..\build\x64\Release\cnumpy_ahk.dll"
    Numpy.Init()
    baseline := Numpy.AllocatedMemory()

    try {
        Out("version", Numpy.Version())

        ; --- ch2: an example ---
        a := Numpy.Arange(0, 15).Reshape([3, 5])
        Out("ch2_tostring", a.ToString())
        Out("ch2_shape", "[" a.Shape[1] ", " a.Shape[2] "]")
        Out("ch2_ndim", a.Ndim)
        Out("ch2_typename", Numpy.Typename(a.Dtype))
        Out("ch2_itemsize", a.ItemSize)
        Out("ch2_size", a.Size)
        a := 0

        ; --- ch2: creation ---
        matrix := Numpy.Array([1, 2, 3, 4, 5, 6], [2, 3])
        ints := Numpy.IntArray([1, 2, 3])
        booleans := Numpy.Array([1, 0, 1], [3], Numpy.DT_BOOL)
        Out("ch2_matrix", matrix.ToString())
        Out("ch2_ints", ints.ToString())
        Out("ch2_bools", booleans.ToString())
        r := Numpy.Arange(10, 30, 5)
        x := Numpy.Linspace(0, 2, 9)
        Out("ch2_arange", r.ToString())
        Out("ch2_linspace", x.ToString())
        matrix := 0
        ints := 0
        booleans := 0
        r := 0
        x := 0

        ; --- ch2: printing ---
        c := Numpy.Arange(0, 24).Reshape([2, 3, 4])
        Out("ch2_3d", c.ToString())
        previous := Numpy.SetPrintOptions(, 6)
        big := Numpy.Arange(0, 10000)
        Out("ch2_summarized", Numpy.Array2String(big, 4096))
        Numpy.SetPrintOptions(, previous["threshold"])
        c := 0
        big := 0

        ; --- ch2: operations ---
        a := Numpy.Array([20, 30, 40, 50])
        b := Numpy.Arange(0, 4)
        diff := Numpy.Subtract(a, b)
        squares := Numpy.Multiply(b, b)
        tenSin := Numpy.Multiply(Numpy.Full([1], 10.0), Numpy.Sin(a))
        mask := Numpy.Less(a, Numpy.Full([1], 35.0))
        Out("ch2_sub", diff.ToString())
        Out("ch2_squares", squares.ToString())
        Out("ch2_tensin", tenSin.ToString())
        Out("ch2_mask", mask.ToString())
        diff := 0
        squares := 0
        tenSin := 0
        mask := 0
        a := 0
        b := 0

        A_ := Numpy.Array([1, 1, 0, 1], [2, 2])
        B_ := Numpy.Array([2, 0, 3, 4], [2, 2])
        ew := Numpy.Multiply(A_, B_)
        pr := A_.Matmul(B_)
        Out("ch2_elementwise", ew.ToString())
        Out("ch2_matmul", pr.ToString())
        ew := 0
        pr := 0
        A_ := 0
        B_ := 0

        rg := Numpy.Arange(0, 6).Reshape([2, 3])
        Out("ch2_sum_all", rg.Sum())
        colSums := rg.Sum(0)
        rowSums := rg.Sum(1)
        running := rg.Cumsum(1)
        Out("ch2_sum_axis0", colSums.ToString())
        Out("ch2_sum_axis1", rowSums.ToString())
        Out("ch2_cumsum", running.ToString())
        colSums := 0
        rowSums := 0
        running := 0
        rg := 0

        ; --- ch2: ufuncs ---
        b2 := Numpy.Arange(0, 3)
        e := b2.Exp()
        s := b2.Sqrt()
        c2 := Numpy.Array([2, -1, 4])
        add := Numpy.Add(b2, c2)
        Out("ch2_exp", e.ToString())
        Out("ch2_sqrt", s.ToString())
        Out("ch2_add", add.ToString())
        e := 0
        s := 0
        add := 0
        c2 := 0

        dest := Numpy.Empty([3])
        c2 := Numpy.Array([2, -1, 4])
        Numpy.Add(b2, c2, dest)
        Out("ch2_add_into", dest.ToString())
        dest := 0
        c2 := 0
        b2 := 0

        ; --- ch2: indexing ---
        cubes := Numpy.Power(Numpy.Arange(0, 10), Numpy.Full([1], 3.0))
        Out("ch2_cubes", cubes.ToString())
        Out("ch2_getitem2", cubes.GetItem(2))
        Out("ch2_bracket3", cubes[3])
        part := Numpy.Slice(cubes, 2, 5)
        Out("ch2_slice", part.ToString())
        everyOther := Numpy.Slice(cubes, 0, 10, 2)
        Out("ch2_step", everyOther.ToString())
        rev := Numpy.Flip(cubes)
        Out("ch2_flip", rev.ToString())
        part := 0
        everyOther := 0
        rev := 0
        cubes := 0

        bfun := Numpy.FromFunction((x2, y2) => 10 * x2 + y2, [5, 4])
        Out("ch2_fromfunction", bfun.ToString())
        point := Numpy.ArrayGetItem(bfun, [2, 3])
        Out("ch2_point", point.ToArray()[1])
        lastRow := Numpy.ArrayGetItem(bfun, [-1, 1])
        Out("ch2_neg", lastRow.ToArray()[1])
        rows := Numpy.Slice(bfun, 1, 3)
        Out("ch2_rows", rows.ToString())
        column := Numpy.Slice(bfun, 1, 2, 1, 1)
        Out("ch2_column", column.ToString())
        point := 0
        lastRow := 0
        rows := 0
        column := 0
        bfun := 0

        small := Numpy.Array([1, 2, 3, 4], [2, 2])
        text := ""
        for value in small.ToArray()
            text .= value " "
        Out("ch2_iter_flat", text)
        lines := ""
        for pair in Numpy.Ndenumerate(small)
            lines .= "(" pair[1][1] ", " pair[1][2] ") -> " pair[2] "; "
        Out("ch2_ndenumerate", lines)
        small := 0

        ; --- ch3: shape ---
        a3 := Numpy.Array([2, 8, 0, 6, 4, 5, 1, 1, 8, 9, 3, 6], [3, 4])
        rv := a3.Ravel()
        rs := a3.Reshape([6, 2])
        tp := a3.Transpose()
        fl := a3.Flatten()
        Out("ch3_ravel", rv.ToString())
        Out("ch3_reshape", rs.ToString())
        Out("ch3_transpose", tp.ToString())
        Out("ch3_flatten", fl.ToString())
        auto := a3.Reshape([2, -1])
        Out("ch3_auto", auto.ToString())
        rv := 0
        rs := 0
        tp := 0
        fl := 0
        auto := 0
        a3 := 0

        src := Numpy.Array([1, 2, 3, 4, 5, 6], [2, 3])
        msg := ""
        try
            invalid := src.Reshape([4, 2])
        catch Error as err
            msg := err.Message
        Out("ch3_error", msg)
        src := 0

        sa := Numpy.Array([9, 7, 5, 2], [2, 2])
        sb := Numpy.Array([1, 9, 5, 1], [2, 2])
        vs := Numpy.Vstack([sa, sb])
        hs := Numpy.Hstack([sa, sb])
        Out("ch3_vstack", vs.ToString())
        Out("ch3_hstack", hs.ToString())
        cx := Numpy.Array([4.0, 2.0])
        cy := Numpy.Array([3.0, 8.0])
        cs := Numpy.ColumnStack([cx, cy])
        Out("ch3_colstack", cs.ToString())
        vs := 0
        hs := 0
        cs := 0
        cx := 0
        cy := 0
        sa := 0
        sb := 0

        sp := Numpy.Arange(0, 12).Reshape([2, 6])
        Out("ch3_splitsrc", sp.ToString())
        parts := Numpy.Hsplit(sp, 3)
        Out("ch3_hsplit1", parts[1].ToString())
        Out("ch3_hsplit2", parts[2].ToString())
        for i, p in parts
            parts[i] := 0
        parts := 0
        uneven := Numpy.Hsplit(sp, [3, 4])
        Out("ch3_uneven1", uneven[1].ToString())
        Out("ch3_uneven2", uneven[2].ToString())
        Out("ch3_uneven3", uneven[3].ToString())
        for i, p in uneven
            uneven[i] := 0
        uneven := 0
        sp := 0

        ; --- ch4: views ---
        va := Numpy.Arange(0, 12).Reshape([3, 4])
        vv := va.View()
        vr := va.Reshape([6, 2])
        vt := va.Transpose()
        vc := va.Copy()
        Out("ch4_share_view", Numpy.SharesMemory(va, vv))
        Out("ch4_share_reshape", Numpy.SharesMemory(va, vr))
        Out("ch4_share_transpose", Numpy.SharesMemory(va, vt))
        Out("ch4_share_copy", Numpy.SharesMemory(va, vc))
        va.SetItem(0, 99)
        Out("ch4_write_through", vr.GetItem(0))
        vv := 0
        vr := 0
        vt := 0
        vc := 0
        va := 0

        ; --- ch5: broadcasting ---
        source := Numpy.Array([1, 2, 3, 4, 5, 6], [2, 3])
        offsets := Numpy.Array([10, 20, 30], [1, 3])
        shifted := Numpy.Add(source, offsets)
        Out("ch5_shifted", shifted.ToString())
        doubled := Numpy.Multiply(source, Numpy.Full([1], 2.0))
        Out("ch5_doubled", doubled.ToString())
        Out("ch5_can", Numpy.CanBroadcast(source, offsets))
        rshape := Numpy.BroadcastShapes([[3, 1], [1, 4]])
        Out("ch5_shapes", "[" rshape[1] ", " rshape[2] "]")
        shifted := 0
        doubled := 0
        offsets := 0
        source := 0

        ; --- ch6: advanced indexing ---
        sq := Numpy.Multiply(Numpy.Arange(0, 12), Numpy.Arange(0, 12))
        Out("ch6_squares", sq.ToString())
        picked := Numpy.Take(sq, [1, 1, 3, 8, 5])
        Out("ch6_take", picked.ToString())
        picked := 0
        sq := 0

        ma := Numpy.Arange(0, 12).Reshape([3, 4])
        mask6 := Numpy.Greater(ma, Numpy.Full([1], 4.0))
        Out("ch6_mask", mask6.ToString())
        sel := Numpy.BooleanIndex(ma, mask6)
        Out("ch6_boolindex", sel.ToString())
        sel := 0
        mask6 := 0
        ma := 0

        wx := Numpy.Arange(0, 6)
        isBig := Numpy.GreaterEqual(wx, Numpy.Full([1], 3.0))
        capped := Numpy.Where(isBig, Numpy.Full([6], 3.0), wx)
        Out("ch6_where", capped.ToString())
        capped := 0
        isBig := 0
        wx := 0

        grid := Numpy.Array([1, 0, 0, 0, 2, 0], [2, 3])
        nz := Numpy.NotEqual(grid, Numpy.Zeros([1]))
        locations := Numpy.Where(nz)
        Out("ch6_where_idx", locations[1].ToString() "  " locations[2].ToString())
        for i, p in locations
            locations[i] := 0
        locations := 0
        nz := 0
        grid := 0

        s6 := Numpy.IntArray([3, 1, 2, 2, 5, 4], [2, 3])
        so1 := s6.Sort(-1, "stable")
        so2 := s6.Sort("none", "stable")
        so3 := s6.Argsort(-1, "heapsort")
        Out("ch6_sort_row", so1.ToString())
        Out("ch6_sort_flat", so2.ToString())
        Out("ch6_argsort", so3.ToString())
        uparts := Numpy.Unique(s6, true, true, true)
        Out("ch6_unique", uparts[1].ToString())
        for i, p in uparts
            uparts[i] := 0
        uparts := 0
        so1 := 0
        so2 := 0
        so3 := 0
        s6 := 0

        ; --- ch7: linalg ---
        la := Numpy.Array([1.0, 2.0, 3.0, 4.0], [2, 2])
        lt := la.Transpose()
        li := Numpy.Linalg.Inv(la)
        Out("ch7_transpose", lt.ToString())
        Out("ch7_inv", li.ToString())
        eye := Numpy.Eye(2)
        Out("ch7_trace", Numpy.TraceExt(eye))
        det := la.Det()
        Out("ch7_det", det.GetItem(0))
        ly := Numpy.Array([5.0, 7.0], [2, 1])
        lx := Numpy.Solve(la, ly)
        Out("ch7_solve", lx.ToString())
        lt := 0
        li := 0
        eye := 0
        det := 0
        lx := 0
        ly := 0
        la := 0

        ; --- ch8: random ---
        Numpy.Random.Seed(42)
        u := Numpy.Random.Random([2, 3])
        Out("ch8_ushape", "[" u.Shape[1] ", " u.Shape[2] "]")
        n8 := Numpy.Random.Normal([1000], 2.0, 0.5)
        Out("ch8_mean", Format("{:.2f}", n8.Mean()))
        deck := Numpy.Arange(0, 10)
        shuffled := Numpy.Random.Permutation(deck)
        Numpy.Random.Shuffle(deck)
        choice := Numpy.Random.Choice(deck, 3, false)
        Out("ch8_choice_size", choice.Size)
        u := 0
        n8 := 0
        deck := 0
        shuffled := 0
        choice := 0

        ; --- ch9: callbacks ---
        grid9 := Numpy.FromFunction((x9, y9) => 10 * x9 + y9, [2, 3])
        Out("ch9_fromfunction", grid9.ToString())
        src9 := Numpy.Array([1.0, 2.0, 3.0])
        trip := Numpy.Vectorize(v9 => v9 * 2 + 1, src9)
        Out("ch9_vectorize", trip.ToString())
        m9 := Numpy.Array([1, 2, 3, 4, 5, 6], [2, 3])
        colT := Numpy.ApplyAlongAxis(SumLine, 0, m9)
        Out("ch9_applyalong", colT.ToString())
        sq9 := Numpy.FromIter(MakeCounter(), 5)
        Out("ch9_fromiter", sq9.ToString())
        grid9 := 0
        src9 := 0
        trip := 0
        m9 := 0
        colT := 0
        sq9 := 0

        Numpy.Random.Seed(7)
        data9 := Numpy.Random.Normal([10000], 2.0, 0.5)
        counts := Numpy.Histogram(data9, 10)
        Out("ch9_hist_sum", counts.Sum())
        counts := 0
        data9 := 0

        ; --- ch10: lifecycle ---
        s10 := Numpy.Arange(0, 1024)
        r10 := Numpy.Sqrt(s10)
        Out("ch10_mean", Format("{:.2f}", r10.Mean()))
        r10 := 0
        s10 := 0
    } finally {
        retained := Numpy.AllocatedMemory()
        Numpy.Cleanup()
    }

    Out("retained_bytes", retained - baseline)
}

SumLine(values) {
    total := 0.0
    for value in values
        total += value
    return total
}

MakeCounter() {
    i := 0
    return (*) => (i += 1, i * i)
}

try Main()
catch Error as err {
    FileAppend("ERROR: " err.Message " | what=" err.What
        " | line=" err.Line " | extra=" (err.HasProp("Extra") ? err.Extra : "") "`n", "**")
    ExitApp 1
}
ExitApp 0
