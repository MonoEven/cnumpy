# cnumpy performance evidence environment / cnumpy 性能证据环境

This record identifies the host, software, binary, protocol, and published
evidence used by the 2026-08-04 bilingual performance report. It is descriptive,
not a claim that the measurements generalize to other hosts.

本记录标识 2026-08-04 中英文性能报告所用的主机、软件、二进制、测试协议和公开证据。
它是可审计的环境说明，不代表结果可直接推广到其他主机。

## Host / 主机

| Field / 字段 | Recorded value / 记录值 |
|---|---|
| Manufacturer and model / 厂商与型号 | Lenovo 82TF |
| CPU | 12th Gen Intel Core i7-12700H |
| Physical/logical processors / 物理核/逻辑处理器 | 14 / 20 |
| Reported maximum clock / 报告最大频率 | 2300 MHz |
| Physical memory / 物理内存 | 16,979,283,968 bytes |
| OS | Microsoft Windows 11 Pro Insider Preview, 64-bit |
| OS version/build / 系统版本 | 10.0.29617 / 29617 |
| Process architecture / 进程架构 | AMD64, Python 64-bit |
| Active power scheme / 活动电源方案 | Balanced (`381b4222-f694-41f0-9685-ff5bb260df2e`) |
| Logical processors recorded by runner / 执行器记录的逻辑处理器 | 20 |

CPU temperature, effective clock, package power, background-process load, AC/battery
state, and fan mode were not captured. No processor affinity or priority override was
applied.

These omissions matter because the observed timings were order-sensitive.

未采集 CPU 温度、有效频率、封装功耗、后台负载、交流/电池状态和风扇模式，也未固定处理器亲和性或提升进程优先级。
鉴于结果存在明显顺序敏感性，这些未采集项会影响解释。

## Software and timers / 软件与计时器

| Component / 组件 | Version or setting / 版本或设置 |
|---|---|
| cnumpy DLL | `1.21.0-cnumpy` |
| AutoHotkey | `2.1-alpha.30`, x64 |
| Python | `3.10.11`, x64 |
| NumPy oracle | `1.25.0` |
| SciPy oracle | `1.12.0` |
| Python timer | `time.perf_counter_ns`, recorded resolution 100 ns |
| AutoHotkey timer | `QueryPerformanceCounter`, frequency 10,000,000 Hz |
| Native SIMD selected at runtime / 原生运行时 SIMD | AVX2 (`simd_level=2`) |
| Native thread setting / 原生线程设置 | `CNP_NUM_THREADS=0` (automatic private thread pool) |
| Source commit under test / 被测源码提交 | `7344e303998a321a8f0f05f4cab73c6405b2e7c6` |

The orchestrator used an existing qualified Release DLL and recorded
`build.requested=false`; it did not rebuild during any measured run.

执行器使用既有、已验证的 Release DLL，并记录 `build.requested=false`；三次测量期间均未重新构建。

## Binary identity and build metadata / 二进制身份与构建元数据

| Field / 字段 | Value / 值 |
|---|---|
| File | `build/x64/Release/cnumpy_ahk.dll` |
| Size | 1,183,232 bytes |
| SHA-256 | `9b15dc7fc2a19fbb63ae919e6f3429cbd33443a04f588223c62773d6bd67e21b` |
| Recorded modification time | `2026-08-04T08:08:26.779180+00:00` |
| Configuration/platform | Release / x64 |
| MSVC optimization | `MaxSpeed`, function-level linking, intrinsics enabled |
| Floating-point model | `Fast` |
| Language mode | C, conformance mode enabled |
| Compiler warning level | Level 3; SDL checks enabled |

All three run manifests contain the same DLL size, timestamp, and SHA-256.

三份运行清单中的 DLL 大小、时间戳和 SHA-256 完全一致。

## Run identities and protocol / 运行身份与协议

| Role / 角色 | Run ID | Runtime order / 运行顺序 | Warmups | Samples | Target batch | Seed | Wall time |
|---|---|---|---:|---:|---:|---:|---:|
| Primary / 主运行 | `20260804T103912.897969Z` | NumPy → cnumpy | 5 | 15 | 20 ms | 12345 | 1,029.573429 s |
| Order control / 顺序对照 | `20260804T105712.384345Z` | cnumpy → NumPy | 5 | 15 | 20 ms | 12345 | 907.955302 s |
| Stability diagnostic / 稳定性诊断 | `20260804T111326.272846Z` | cnumpy → NumPy | 5 | 25 | 50 ms | 12345 | 2,296.132728 s |

Every run used the native-size `full` profile with 459 unique cases. Setup and
untimed semantic validation precede warmup and measurement. Calibrated batching
targets the duration shown above.

AHK `DllCall`, marshalling, native execution, returned-array allocation, and
release remain inside the cnumpy timing boundary.

每次运行都使用原生规模的 `full` 配置，共 459 个唯一用例。输入准备和非计时语义验证先于预热和测量；
校准批处理以表中时长为目标。AHK `DllCall`、封送、原生执行、返回数组分配和释放均保留在 cnumpy 计时边界内。

The diagnostic is not a third headline comparison. It was triggered because the
union of cases with either runtime CV above 5% in the two 15-sample runs covered
all 459 cases.

诊断运行不是第三组主排名。由于两次 15 样本运行中“任一运行时 CV 超过 5%”的用例并集覆盖全部 459 项，
因此按预定规则对全部用例执行 25 样本、50 ms 批次复测。

## Published evidence identity / 公开证据身份

| Evidence / 证据 | Rows | SHA-256 |
|---|---:|---|
| `2026-08-04-full-numpy-first.csv` | 459 | `255f3caeeb1bb5a2dca637cc4f8a8668e9465b893664f01a04d991cd4056e514` |
| `2026-08-04-full-cnumpy-first.csv` | 459 | `79beef5497d7248ee13ddb49b0821d9fdfc2cbea9676eb6ca25fccc8dfc3ea42` |
| `2026-08-04-noise-diagnostic.csv` | 459 | `ef4164b131be3458f1766f0866a74a94f74bdff8231ab21fc4277bebec7ac98a` |

Each CSV contains 459 unique IDs. Each corresponding `cnumpy.json` contains 459
native cases, and every case records `retained_bytes=0`.

The full JSON, jobs, stdout/stderr, environment manifests, and generated Markdown
remain in the ignored local run directories named by the run IDs.

每份 CSV 都包含 459 个唯一 ID；对应的 `cnumpy.json` 各含 459 个原生用例，且每项均记录
`retained_bytes=0`。完整 JSON、任务清单、标准输出/错误、环境清单和生成报告保留在按运行 ID 命名的本地忽略目录中。

## Visible warnings / 可见警告

All three NumPy benchmark runs retained this warning in `numpy.stderr.log`:

```text
DeprecationWarning: msort is deprecated, use np.sort(a, axis=0) instead
```

三次 NumPy 基准都在 `numpy.stderr.log` 中保留上述警告。警告未被过滤、吞掉或改写为成功信息。

## Rejected verification environment / 被拒绝的验证环境

An audit command using unqualified `python` resolved to
`F:\Python\Python38\python.exe` (Python 3.8.10, NumPy 1.24.4), not the qualified
Python 3.10.11 runtime.

It ran 1,236 tests and reported 20 failures plus 251 errors. Those failures
remain disclosed and are excluded from benchmark evidence.

一次使用无限定 `python` 的审计命令命中了 `F:\Python\Python38\python.exe`
（Python 3.8.10、NumPy 1.24.4），而不是已验证的 Python 3.10.11。该运行执行 1,236 项，
报告 20 个失败和 251 个错误；失败信息予以公开，但不纳入基准证据。

The same source then passed 1,677 tests under the explicit qualified interpreter
`F:\Python\Python310\python.exe`. No product source change occurred between the
rejected and accepted verification runs.

同一源码随后在显式已验证解释器 `F:\Python\Python310\python.exe` 下通过 1,677 项；
被拒绝与通过的两次运行之间没有产品源码变更。
