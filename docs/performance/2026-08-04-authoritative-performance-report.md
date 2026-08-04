# cnumpy authoritative performance report / cnumpy 权威性能测试报告

**Date / 日期:** 2026-08-04

**Reference / 参考实现:** NumPy 1.25.0

**Artifact / 被测制品:** cnumpy 1.21.0 x64, SHA-256
`9b15dc7fc2a19fbb63ae919e6f3429cbd33443a04f588223c62773d6bd67e21b`

This is an authoritative, host-scoped engineering report: every headline value
is traceable to a public row-level CSV, an identified binary, a recorded protocol,
and passing behavior/lifecycle gates.

It is not an independent certification or a claim about unmeasured machines.

本报告是“限定主机范围的权威工程报告”：每个核心数字都能追溯到公开逐项 CSV、唯一二进制、固定协议及已通过的行为/生命周期门禁。
它不是第三方认证，也不把单机结果外推到未测平台。

## Executive verdict / 执行结论

The result does not support a blanket claim that cnumpy is faster than NumPy.
In the primary run, cnumpy won 201 of 459 cases, while the geometric-mean
`cnumpy/NumPy` ratio was 1.526201.

The order-control run produced 94 wins and a 2.375207 ratio.

结果不支持“cnumpy 全面快于 NumPy”的笼统结论。主运行中 cnumpy 赢得 459 项中的 201 项，
但 `cnumpy/NumPy` 比值的几何平均为 1.526201；反向顺序对照中仅赢 94 项，几何平均为 2.375207。

cnumpy's clearest strength is dense linear algebra. Its largest absolute primary
win was 512×512 solve: 79.154 ms versus 687.562 ms.

Sorting, duplicate-heavy set operations, shape/view calls, and the AHK/DLL
bridge offer the largest remaining optimization space.

cnumpy 最明确的优势是稠密线性代数。主运行中最大绝对优势是 512×512 `solve`：79.154 ms 对 687.562 ms。
排序、重复值密集集合运算、形状/视图调用及 AHK/DLL 桥接仍有最大优化空间。

The host was a Balanced-power Lenovo laptop running a Windows Insider build.
CV and order effects were high. The data is suitable for bottleneck discovery
and host-local decisions, but not for a cross-platform leaderboard.

主机是采用“平衡”电源方案、运行 Windows Insider 版本的联想笔记本。CV 和顺序效应均较高。
这些数据适合定位瓶颈和本机决策，不适合制作跨平台排行榜。

## Evidence bundle / 证据包

- [Environment and artifact record](evidence/2026-08-04-environment.md)
- [Primary: full profile, NumPy first](evidence/2026-08-04-full-numpy-first.csv)
- [Order control: full profile, cnumpy first](evidence/2026-08-04-full-cnumpy-first.csv)
- [Stability diagnostic: 25 samples, 50 ms batches](evidence/2026-08-04-noise-diagnostic.csv)
- [NumPy 1.25 compatibility statement](../compatibility/numpy-1.25.md)

The three CSVs contain exact machine-readable medians, ratios, winners, CVs,
p95 values, categories, shapes, and semantic-qualification IDs for all 459 cases.
Their SHA-256 values are recorded in the environment document.

三份 CSV 包含全部 459 项的精确机器可读中位数、比值、胜者、CV、p95、类别、形状和语义验证 ID；
文件 SHA-256 记录在环境文档中。

## Shared numerical evidence / 共享数值证据

Lower `cnumpy/NumPy` ratios are better for cnumpy. A ratio of 0.5 means half the
NumPy time; 2.0 means twice the NumPy time. The category aggregate is the
geometric mean of per-case ratios. No unrelated operation times are summed.

`cnumpy/NumPy` 比值越低越有利于 cnumpy。0.5 表示耗时为 NumPy 的一半，2.0 表示耗时为两倍。
类别汇总采用逐项比值的几何平均，不对无关操作的耗时求和。

### Run summary / 运行汇总

| Role / 角色 | Order / 顺序 | Protocol | Cases | Geomean ratio | cnumpy wins | NumPy wins | NumPy CV >5% | cnumpy CV >5% |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Primary / 主运行 | NumPy → cnumpy | 5 warmups, 15×20 ms | 459 | 1.526201 | 201 | 258 | 391 | 449 |
| Order control / 顺序对照 | cnumpy → NumPy | 5 warmups, 15×20 ms | 459 | 2.375207 | 94 | 365 | 447 | 440 |
| Diagnostic / 诊断 | cnumpy → NumPy | 5 warmups, 25×50 ms | 459 | 2.573790 | 93 | 366 | 367 | 440 |

The diagnostic ratio and winner count are not headline results. That run changes
the sampling protocol and exists only to test whether longer measurement reduces
CV. It must not be pooled with either 15-sample run.

诊断运行的比值和胜负数不是主结论。该运行改变了采样协议，仅用于判断延长测量能否降低 CV，
不得与两次 15 样本运行合并。

### Ratio distribution / 比值分布

| `cnumpy/NumPy` bucket | Meaning / 含义 | Primary | Order control |
|---|---|---:|---:|
| `<0.50` | cnumpy at least 2× faster / cnumpy 至少快 2 倍 | 64 | 30 |
| `0.50–<0.80` | cnumpy 1.25–2× faster / cnumpy 快 1.25–2 倍 | 94 | 31 |
| `0.80–<1.00` | cnumpy up to 1.25× faster / cnumpy 快至多 1.25 倍 | 43 | 33 |
| `1.00–<1.25` | NumPy up to 1.25× faster / NumPy 快至多 1.25 倍 | 33 | 67 |
| `1.25–<2.00` | NumPy 1.25–2× faster / NumPy 快 1.25–2 倍 | 59 | 78 |
| `≥2.00` | NumPy at least 2× faster / NumPy 至少快 2 倍 | 166 | 220 |

Primary ratio percentiles were p5 0.246760, p25 0.669696, median 1.232053,
p75 2.877266, and p95 18.112488. The wide distribution makes one overall ratio
insufficient for workload selection.

主运行比值分位数为 p5 0.246760、p25 0.669696、中位数 1.232053、p75 2.877266、p95 18.112488。
分布跨度很大，单一总体比值不足以指导具体负载选择。

### Category geometric means / 类别几何平均

| Category | Cases | Primary ratio | Primary C/N wins | Control ratio | Control C/N wins |
|---|---:|---:|---:|---:|---:|
| binary | 56 | 1.405846 | 30 / 26 | 2.311178 | 10 / 46 |
| bitwise | 24 | 3.255763 | 2 / 22 | 4.107762 | 0 / 24 |
| bridge | 6 | 54.297522 | 0 / 6 | 63.613262 | 0 / 6 |
| comparison | 8 | 0.885519 | 5 / 3 | 1.153375 | 4 / 4 |
| creation | 20 | 1.404471 | 9 / 11 | 2.179817 | 6 / 14 |
| fft | 4 | 1.812969 | 0 / 4 | 3.371419 | 0 / 4 |
| indexing | 12 | 1.883630 | 2 / 10 | 2.323159 | 2 / 10 |
| integer | 8 | 1.194105 | 2 / 6 | 1.287449 | 0 / 8 |
| linalg | 30 | 0.318404 | 22 / 8 | 0.518267 | 17 / 13 |
| logical | 44 | 3.299960 | 9 / 35 | 4.848893 | 6 / 38 |
| misc_axis | 20 | 1.043838 | 11 / 9 | 1.785043 | 3 / 17 |
| pipeline | 8 | 0.452426 | 8 / 0 | 1.143671 | 3 / 5 |
| preallocated | 12 | 0.688326 | 8 / 4 | 1.278969 | 4 / 8 |
| random | 1 | 0.945659 | 1 / 0 | 0.795131 | 1 / 0 |
| reduction | 42 | 1.052319 | 20 / 22 | 2.054389 | 4 / 38 |
| set | 16 | 1.739174 | 4 / 12 | 2.195255 | 4 / 12 |
| shape | 30 | 8.055573 | 5 / 25 | 12.034057 | 3 / 27 |
| signal | 8 | 1.091478 | 5 / 3 | 1.776035 | 1 / 7 |
| sorting | 46 | 1.024456 | 26 / 20 | 1.246152 | 17 / 29 |
| unary | 64 | 1.407863 | 32 / 32 | 2.565834 | 9 / 55 |

`C/N wins` means cnumpy wins / NumPy wins. Exact unrounded row values remain in
the linked CSVs. The `random` category has one weighted-choice case and should
not be read as a broad random-number result.

`C/N wins` 表示 cnumpy 胜项数 / NumPy 胜项数。未舍入的逐项精确值见所链接 CSV。
`random` 类别只有一个加权选择用例，不能解读为对全部随机数功能的结论。

### Largest absolute primary deficits / 主运行最大绝对劣势

| Case | NumPy ms | cnumpy ms | Ratio | cnumpy excess ms |
|---|---:|---:|---:|---:|
| `argsort_heapsort/f64/1000000` | 521.8318 | 903.8927 | 1.732153 | 382.0609 |
| `argsort/f64/1000000` | 143.0373 | 332.5575 | 2.324970 | 189.5202 |
| `sort_heapsort/f64/1000000` | 222.6998 | 381.2786 | 1.712074 | 158.5788 |
| `intersect1d_duplicates/f64/1000000` | 49.6727 | 146.2966 | 2.945211 | 96.6239 |
| `setdiff1d_duplicates/f64/1000000` | 52.2677 | 139.7003 | 2.672785 | 87.4326 |
| `union1d_duplicates/f64/1000000` | 53.3691 | 135.7707 | 2.543995 | 82.4016 |
| `unique_duplicates/f64/1000000` | 51.3380 | 132.6735 | 2.584314 | 81.3355 |
| `setxor1d_duplicates/f64/1000000` | 49.4003 | 130.4680 | 2.641037 | 81.0677 |
| `sort/f64/1000000` | 90.0852 | 166.3574 | 1.846667 | 76.2722 |
| `sort_complex/f64/1000000` | 101.3766 | 176.2396 | 1.738464 | 74.8630 |

### Largest absolute primary wins / 主运行最大绝对优势

| Case | NumPy ms | cnumpy ms | Ratio | cnumpy saved ms |
|---|---:|---:|---:|---:|
| `solve/f64/512x512` | 687.5615 | 79.1539 | 0.115123 | 608.4076 |
| `det/f64/512x512` | 675.6012 | 278.2669 | 0.411880 | 397.3343 |
| `lexsort/f64/1000000` | 369.7889 | 96.8019 | 0.261776 | 272.9870 |
| `inv/f64/512x512` | 730.7580 | 467.0185 | 0.639088 | 263.7395 |
| `searchsorted/f64/1000000` | 492.4563 | 320.7893 | 0.651407 | 171.6670 |
| `det/f64/128x128` | 127.7970 | 2.2109 | 0.017300 | 125.5861 |
| `searchsorted_right/f64/1000000` | 432.4060 | 313.9861 | 0.726137 | 118.4199 |
| `solve/f64/128x128` | 117.3148 | 1.4928 | 0.012725 | 115.8220 |
| `svd/f64/128x128` | 146.8549 | 31.7043 | 0.215889 | 115.1506 |
| `digitize/f64/1000000` | 448.2131 | 339.5402 | 0.757542 | 108.6729 |

These are discovery rankings, not guaranteed speedups. At least nine of the first ten
rows in each table had at least one runtime CV above 5% in the primary run.
Use the exact CSV CV columns before making a deployment decision.

这些是瓶颈发现排序，不是保证的加速比。两表各自前十项中都有九项在主运行中至少一个运行时 CV 超过 5%。
部署决策前必须查看 CSV 中的精确 CV 列。

### Highest ratios and their absolute cost / 最高比值及其绝对成本

| Case | Ratio | cnumpy excess |
|---|---:|---:|
| `bridge/property_call` | 173.538944 | 23.320 µs |
| `bridge/property_cached` | 82.378667 | 11.102 µs |
| `bridge/nbytes_cached` | 57.313070 | 8.255 µs |
| `real/f64/1000` | 55.116983 | 18.552 µs |
| `real/f64/10000` | 52.342677 | 17.767 µs |
| `reshape/f64/10000` | 48.344773 | 18.714 µs |
| `real/f64/100000` | 47.706858 | 16.676 µs |
| `real/f64/1000000` | 46.805144 | 16.168 µs |
| `reshape/f64/1000` | 43.455618 | 18.258 µs |
| `reshape/f64/1000000` | 43.307239 | 16.567 µs |

These very high ratios compare tens of microseconds with NumPy metadata/view
operations measured below one microsecond. They matter in tight call loops, but
they are not the largest end-to-end latency costs. Absolute time takes priority.

这些极高比值来自“几十微秒”与“低于一微秒的 NumPy 元数据/视图操作”的比较。
它们会影响高频小调用，但不是端到端耗时最大的项目，优化排序应优先看绝对时间。

### Full-only 512×512 cubic cases / 仅完整配置包含的 512×512 三次复杂度用例

| Case | Primary NumPy/cnumpy ms | Primary ratio | Control NumPy/cnumpy ms | Control ratio |
|---|---:|---:|---:|---:|
| `cholesky/f64/512x512` | 18.6869 / 12.9449 | 0.692725 | 18.3900 / 7.7848 | 0.423315 |
| `det/f64/512x512` | 675.6012 / 278.2669 | 0.411880 | 339.7396 / 293.3616 | 0.863490 |
| `inv/f64/512x512` | 730.7580 / 467.0185 | 0.639088 | 406.5545 / 453.1200 | 1.114537 |
| `solve/f64/512x512` | 687.5615 / 79.1539 | 0.115123 | 479.7294 / 89.9943 | 0.187594 |

`inv` changed winner under order reversal. The other three remained cnumpy wins,
but their CVs and median shifts still prevent a machine-independent claim.

`inv` 在反向顺序下改变了胜者。其余三项仍由 cnumpy 获胜，但 CV 和中位数偏移仍不足以支持跨机器结论。

### Order sensitivity and stability / 顺序敏感性与稳定性

Symmetric order change is defined as `max(primary, control) / min(primary,
control) - 1` for the same runtime and case.

同一运行时、同一用例的对称顺序变化定义为
`max(primary, control) / min(primary, control) - 1`。

| Runtime | Cases >10% | Cases >25% | Cases >50% | Median symmetric change | Maximum change |
|---|---:|---:|---:|---:|---:|
| NumPy | 450 | 428 | 323 | 72.368% | 609.865% |
| cnumpy | 250 | 120 | 59 | 11.463% | 178.377% |

| Runtime | Control CV >5% | Diagnostic CV >5% | Median CV before | Median CV after | Cases improved |
|---|---:|---:|---:|---:|---:|
| NumPy | 447 | 367 | 10.229% | 6.867% | 354 / 459 |
| cnumpy | 440 | 440 | 10.005% | 8.973% | 273 / 459 |

The longer diagnostic reduced median CV for both runtimes, but did not bring the
suite below the 5% stability threshold. All 459 cases required diagnostic rerun.

This is evidence of host phase, scheduling, power, or thermal sensitivity—not a
reason to hide or average away the variation.

延长诊断降低了两种运行时的 CV 中位数，但整套用例仍未收敛到 5% 稳定性阈值以内；全部 459 项都触发了复测。
这说明主机阶段、调度、电源或热状态敏感，不能通过隐藏或混合平均来消除。

---

## 中文完整报告

### 1. 测试目标与比较边界

本报告比较两个真实公开路径：Python 调用 NumPy 1.25.0 公共 API，以及 AutoHotkey v2 通过 `DllCall` 调用 cnumpy 公共接口。
它比较的是用户实际承担的边界成本，而不是孤立 C 内核。

cnumpy 计时包含 AHK 调用、参数封送、C 执行、结果数组分配和释放。输入构造与语义验证在计时区外；
显式 `out` 用例复用预分配结果，其余返回数组在每次计时调用内创建并释放。

Python 使用 `perf_counter_ns`，AHK 使用 `QueryPerformanceCounter`。每项先执行 5 次预热，再自动校准内部循环，
以 20 ms 为主运行目标批次并采集 15 个样本。随机种子固定为 12345。

主运行先测 NumPy、后测 cnumpy；顺序对照相反。任一运行时 CV 超过 5% 的用例按预定规则使用 25 样本、50 ms 批次复测。
由于噪声并集覆盖 459 项，诊断包含全套用例。

### 2. 结果解释

总体几何平均比值从主运行的 1.526201 变为顺序对照的 2.375207，cnumpy 胜项从 201 降至 94。
这不是小幅扰动，而是足以改变产品级结论的环境效应。

线性代数是最有希望的类别：主运行几何平均 0.318404，顺序对照 0.518267。
但 512×512 `inv` 在对照中由 cnumpy 胜转为 NumPy 胜，说明即使强项也必须按具体操作和环境复测。

`pipeline` 和 `preallocated` 在主运行表现良好，分别为 0.452426 和 0.688326；对照中变为 1.143671 和 1.278969。
这提示批处理和预分配方向合理，但当前主机证据不足以量化稳定收益。

桥接类别比值高达 54.297522，但最大单项额外成本只有 23.320 µs。对高频细粒度 AHK 循环，它会累积；
对几十到几百毫秒的大型计算，排序和集合算法的绝对差值更重要。

### 3. 可执行的性能优化优先级

| Priority | Evidence | Recommended engineering focus | Acceptance gate |
|---:|---|---|---|
| P0 | 1M `argsort_heapsort` +382.061 ms；`argsort` +189.520 ms | 分离比较、索引置换和输出分配成本；对堆排序和默认 argsort 分别剖析 | 精确排列/稳定性/NaN 异常测试 + 双顺序全量性能门禁 |
| P0 | 1M 重复值集合操作 +71–97 ms | 剖析排序、去重、合并和中间分配次数，优先减少全量临时缓冲 | 重复值、NaN、空数组、所有权和 `retained_bytes=0` |
| P1 | `sort`/`sort_complex` +75–76 ms | 检查内核选择、数据搬运和复数结果构造 | 排序顺序、dtype、异常和完整差分结果 |
| P1 | 1M `compress` +13.020 ms | 剖析掩码扫描、计数、结果分配和写入遍数 | 连续/跨步轴、空结果、错误轴和生命周期 |
| P1 | 8–23 µs 细粒度桥接税 | 扩展批量/预分配公共路径，减少 AHK 循环中的跨 DLL 次数 | 不绕过公共 ABI，不弱化错误与所有权语义 |
| P2 | `shape` 8.06×；`real`/`reshape` 比值高 | 在契约允许时复用元数据和零拷贝视图，避免重复包装 | 视图别名、源对象释放顺序和异常完全一致 |
| Protect | 大型 `solve`/`det`/`lexsort` 绝对优势 | 先建立独立性能基线，再修改热点，防止强项回退 | 两种顺序、性能电源方案、独立进程重复运行 |

建议先用采样剖析器把 P0 的时间拆到比较器、索引搬运、分配器和内核，再决定实现。
本报告不把相关性当作根因，也不建议通过减少验证、跳过释放或缓存错误结果换取数字。

性能仍有显著提升空间。仅前十个绝对劣势就集中在排序和集合两条执行链，说明优化可以聚焦，而不必对全部 752 个声明平均用力。

### 4. 行为、异常与生命周期门禁

性能数字只在行为门禁通过后才有效。Python 完整套件通过 1,677 项测试；AHK 套件通过 206 个测试方法；
AHK 基准冒烟和真实基准编排均以退出码 0 完成。

所有 459 项在每种运行时都先验证确定性结果。115 项携带任务级精确语义验证 ID：Task 6 为 42 项、Task 7 为 40 项、
Task 8 为 20 项、Task 9 为 12 项、Task 10 为 1 项。

其余 344 项的 `semantic_qualification=N/A` 表示未纳入上述任务级精确版本门禁，不表示跳过运行时结果验证。
兼容范围、故意差异和 752/752 声明所有权见兼容性声明。

三次运行共发布 1,377 条 cnumpy 记录，全部 `retained_bytes=0`。这验证了基准调用完成后的原生分配器回到基线，
并覆盖返回数组在计时循环内创建和释放的路径。

错误不会被替换为空数组、模拟结果或“成功”占位值。无效轴、dtype、形状、空输入、NaN/Inf、奇异矩阵、概率和释放顺序由差分及异常测试直接检查。

测试保留了真实警告：行为套件出现 `RuntimeWarning: invalid value encountered in subtract`；三次 NumPy 基准均记录
`DeprecationWarning: msort is deprecated`。警告没有被吞掉，也未改变退出状态。

审计期间还保留了一次被拒绝的环境失败：无限定的 `python` 命令解析到 Python 3.8.10 / NumPy 1.24.4，
实际运行 1,236 项，产生 20 个失败和 251 个错误。

错误包括缺少 `str.removeprefix`、`zip(strict=...)` 和 `numpy.exceptions`。根因是解释器版本不满足固定环境，
不是产品代码回归；该运行未进入任何性能结论。

同一源码使用显式 Python 3.10.11 / NumPy 1.25.0 后通过全部 1,677 项。复现命令因此固定 `$Python`，
不依赖 PATH 顺序，也没有增加自动降级或替代解释器。

### 5. 适用范围与限制

结果严格绑定到报告中的 DLL SHA-256、源码提交、Windows x64、Python/NumPy/SciPy/AHK 版本和该台 i7-12700H 主机。
更换任一关键组件后，应重新运行完整门禁。

该主机使用“平衡”电源方案和 Windows Insider 构建。未记录温度、有效频率、封装功耗、后台负载和交流供电状态，
也未固定 CPU 亲和性。因此不能把差异全部归因于库实现。

比较的是 Python/NumPy 公共路径与 AHK/cnumpy 公共路径，语言运行时、计时器和 ABI 不同。
桥接成本被故意保留；若目标是纯 C 内核吞吐，应另建只测 C ABI 的补充协议，不能从本报告中减去估算值。

每个用例在总体几何平均中权重相同，没有按真实业务调用频率加权。单项随机类别不代表完整 RNG；
Task 级验证 ID 也只覆盖登记范围，不能外推到相邻 API。

### 6. 工程决策建议

大型稠密线性代数负载可以优先评估 cnumpy，尤其是 `solve`；上线前应在目标机器上使用性能电源方案、随机化顺序和独立进程重复验证。

大量细粒度属性、视图或逐元素 AHK 调用应改为批处理或预分配接口。20 µs 级桥接税在单次调用中很小，
但在数十万次脚本循环中会成为主要成本。

以排序和重复值集合操作为主的工作负载应谨慎采用当前版本。先用实际数据建立操作级基线，
再把 P0 优化结果与 NumPy 1.25.0 做双顺序差分和生命周期复验。

---

## Complete English report

### 1. Objective and comparison boundary

This report compares two real public paths: Python calling NumPy 1.25.0 APIs,
and AutoHotkey v2 calling the cnumpy public interface through `DllCall`. It
measures user-visible boundary cost, not an isolated C kernel.

cnumpy timing includes AHK invocation, marshalling, C execution, result-array
allocation, and release. Input construction and semantic validation are outside
the timed region.

Explicit `out` cases reuse their destination; other returned arrays are created
and released inside every timed call.

Python uses `perf_counter_ns`; AHK uses `QueryPerformanceCounter`. Each case gets
five warmups and calibrated inner loops, then 15 samples targeting 20 ms batches.
The deterministic seed is 12345.

The primary run measures NumPy before cnumpy; the order control reverses them.
The preset rule reruns every case where either runtime CV exceeds 5% using 25
samples and 50 ms batches. The noisy-case union covered all 459 cases.

### 2. Interpretation

The overall geometric-mean ratio moved from 1.526201 in the primary run to
2.375207 in the control, while cnumpy wins fell from 201 to 94. This is large
enough to alter a product-level conclusion.

Linear algebra is the strongest category: 0.318404 primary and 0.518267 control.
Yet 512×512 `inv` changed from a cnumpy win to a NumPy win under reversed order.
Even the strongest area requires operation- and host-specific confirmation.

`pipeline` and `preallocated` were strong in the primary run at 0.452426 and
0.688326, but changed to 1.143671 and 1.278969 in the control.

Batching and preallocation remain sound directions; the current host cannot
quantify a stable benefit.

The bridge category ratio was 54.297522, while its largest single absolute
excess was only 23.320 µs. It can dominate fine-grained AHK loops. For work in
the tens or hundreds of milliseconds, sorting and set-operation deltas matter
more.

### 3. Actionable optimization priorities

| Priority | Evidence | Recommended engineering focus | Acceptance gate |
|---:|---|---|---|
| P0 | 1M `argsort_heapsort` +382.061 ms; `argsort` +189.520 ms | Separate comparison, index permutation, and output-allocation cost; profile heap and default paths independently | Exact permutation/stability/NaN errors plus both-order full performance gate |
| P0 | 1M duplicate-heavy set operations +71–97 ms | Profile sort, deduplication, merge, and intermediate allocation passes; reduce full-size temporaries first | Duplicates, NaNs, empties, ownership, and `retained_bytes=0` |
| P1 | `sort`/`sort_complex` +75–76 ms | Inspect kernel selection, data movement, and complex result construction | Ordering, dtype, exception, and full differential output |
| P1 | 1M `compress` +13.020 ms | Profile mask scan, count, result allocation, and write passes | Contiguous/strided axes, empty results, invalid axes, and lifetime |
| P1 | 8–23 µs fine-grained bridge tax | Extend bulk/preallocated public paths to reduce cross-DLL calls in AHK loops | Do not bypass the public ABI or weaken error/ownership semantics |
| P2 | `shape` 8.06×; high `real`/`reshape` ratios | Reuse metadata and zero-copy views where the contract permits; avoid repeated wrapper construction | View aliasing, source release order, and exact exceptions |
| Protect | Large `solve`/`det`/`lexsort` absolute wins | Establish an independent baseline before hotspot changes | Both orders, performance power mode, repeated independent processes |

Profile the P0 chains into comparison, index movement, allocation, and kernels
before changing implementation. The report does not treat correlation as root
cause, and it does not recommend skipping validation, release, or error paths.

There is material optimization headroom. The first ten absolute deficits are
concentrated in sorting and set pipelines, so work can be focused instead of
spread uniformly across all 752 declarations.

### 4. Behavior, exception, and lifecycle gates

Performance evidence is valid only after behavior gates pass. The complete
Python suite passed 1,677 tests; the AHK suite passed 206 test methods. The AHK
benchmark smoke and real benchmark orchestration completed with exit code 0.

Both runtimes validate deterministic results before timing all 459 cases. Of
these, 115 carry exact task-level semantic qualification IDs: 42 for Task 6,
40 for Task 7, 20 for Task 8, 12 for Task 9, and one for Task 10.

The other 344 rows have `semantic_qualification=N/A`. This means they are not in
those task-specific exact-version gates; it does not mean runtime result
validation was skipped.

See the compatibility statement for intentional differences and ownership of
all 752 declarations.

The three runs publish 1,377 cnumpy records, all with `retained_bytes=0`. This
checks that the native allocator returns to baseline after each benchmark case,
including paths that create and release returned arrays inside timed loops.

Failures are not replaced with empty arrays, simulated results, or success
placeholders. Differential and exception tests directly cover invalid axes,
dtypes, shapes, empties, NaN/Inf, singular matrices, probabilities, and release
ordering.

Real warnings remain visible. The behavior suite emitted
`RuntimeWarning: invalid value encountered in subtract`; all three NumPy runs
recorded `DeprecationWarning: msort is deprecated`.

Neither warning was swallowed or rewritten as success.

The audit also retains one rejected environment failure. An unqualified
`python` command resolved to Python 3.8.10 / NumPy 1.24.4 and ran 1,236 tests,
producing 20 failures and 251 errors.

Representative errors were missing `str.removeprefix`, `zip(strict=...)`, and
`numpy.exceptions`.

The confirmed root cause was an interpreter outside the pinned environment, not
a product-code regression. No result from that run is used in a performance
conclusion.

The same source passed all 1,677 tests with explicit Python 3.10.11 / NumPy
1.25.0. Reproduction therefore pins `$Python` and does not rely on PATH order,
automatic downgrade, or substitute execution.

### 5. Scope and limitations

Results are bound to the reported DLL SHA-256, source commit, Windows x64 host,
and Python/NumPy/SciPy/AHK versions. Replacing any material component requires
the full gate again.

The host used the Balanced power plan and a Windows Insider build. Temperature,
effective frequency, package power, background load, and AC state were not
recorded; CPU affinity was not pinned.

Implementation alone cannot explain all observed variation.

The comparison spans Python/NumPy and AHK/cnumpy public paths, with different
language runtimes, timers, and ABIs. Bridge cost is deliberately retained.

A pure C-kernel question needs a separate C-ABI protocol; estimated overhead
must not be subtracted from this report.

Every case has equal weight in the geometric mean; no production call-frequency
weights are applied. The one random case does not represent the whole RNG API,
and task qualification IDs do not extend beyond their registered operations.

### 6. Engineering decisions

Large dense-linear-algebra workloads should evaluate cnumpy first, especially
`solve`. Before deployment, repeat on the target machine with a performance
power plan, randomized order, and independent processes.

Fine-grained property, view, or element-wise AHK loops should use bulk or
preallocated interfaces. A roughly 20 µs bridge tax is small once, but dominant
across hundreds of thousands of script calls.

Sorting- and duplicate-heavy set workloads should treat the current version
cautiously. Establish an operation-level baseline with production data, then
re-run both-order differential and lifecycle gates after P0 optimization.

## Reproduction / 复现

Use the exact runtime versions and qualified DLL identified in the environment
record. From the repository root in PowerShell:

使用环境记录中的精确运行时版本和已验证 DLL。在仓库根目录运行：

```powershell
$Python = 'F:\Python\Python310\python.exe'
$env:CNP_NUM_THREADS = '0'

& $Python benchmark\benchmark.py --profile full `
  --warmups 5 --samples 15 --target-sample-ms 20 --seed 12345 `
  --runtime-order numpy,cnumpy `
  --output-root benchmark\runs-authoritative-report-numpy-first

& $Python benchmark\benchmark.py --profile full `
  --warmups 5 --samples 15 --target-sample-ms 20 --seed 12345 `
  --runtime-order cnumpy,numpy `
  --output-root benchmark\runs-authoritative-report-cnumpy-first

& $Python benchmark\benchmark.py --profile full `
  --warmups 5 --samples 25 --target-sample-ms 50 --seed 12345 `
  --runtime-order cnumpy,numpy `
  --output-root benchmark\runs-authoritative-report-noise
```

Run behavior and lifecycle gates before interpreting performance:

解释性能结果前，先运行行为和生命周期门禁：

```powershell
$Ahk = 'C:\Program Files\AutoHotkey\v2\AutoHotkey64.exe'
$Python = 'F:\Python\Python310\python.exe'

& $Python -B -W error::ResourceWarning -m unittest discover -s benchmark\tests -v
& $Ahk /ErrorStdOut=UTF-8 ahk\numpy.test.ahk
& $Ahk /ErrorStdOut=UTF-8 benchmark\benchmark_smoke.test.ahk
& $Python benchmark\benchmark.py --profile focus --size-scale smoke `
  --warmups 1 --samples 3 --target-sample-ms 1
```

Verify the binary and published evidence with `Get-FileHash -Algorithm SHA256`.
A valid reproduction must keep complete failed-run directories and stderr logs;
it must not publish partial comparison CSVs as successful runs.

使用 `Get-FileHash -Algorithm SHA256` 验证二进制和公开证据。有效复现必须保留完整失败目录和 stderr 日志，
不得把部分完成的 comparison CSV 发布为成功运行。
