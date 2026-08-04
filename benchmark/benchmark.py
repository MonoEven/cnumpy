"""Orchestrate reproducible NumPy versus AHK/cnumpy benchmark runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import locale
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.report import (
    attach_baseline,
    compare_results,
    expand_cases,
    format_case_id,
    load_baseline_csv,
    load_catalog,
    load_worker_json,
    normalize_worker_result,
    render_comparison_csv,
    render_markdown,
    require_semantic_qualification_numpy_version,
    semantic_qualification_declarations,
    validate_job_case,
    write_jobs_tsv,
)


_MSBUILD_NAMESPACE = "http://schemas.microsoft.com/developer/msbuild/2003"
_MSBUILD_NS = {"msbuild": _MSBUILD_NAMESPACE}
_RELEASE_X64_CONDITION = re.compile(
    r"^\s*'\$\(Configuration\)\|\$\(Platform\)'\s*==\s*'Release\|x64'\s*$",
    re.IGNORECASE,
)


def parse_runtime_order(value: str) -> tuple[str, str]:
    """Parse an exact two-runtime execution order."""
    if value not in {"numpy,cnumpy", "cnumpy,numpy"}:
        raise ValueError(
            "runtime order must be exactly 'numpy,cnumpy' or 'cnumpy,numpy'"
        )
    first, second = value.split(",")
    return first, second


def resolve_explicit_or_candidates(
    explicit: str | Path | None,
    candidates: Iterable[str | Path],
    label: str,
) -> Path:
    """Resolve a required executable/file without replacing a bad explicit path."""
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"explicit {label} is not a regular file: {path}")
        return path
    examined: list[str] = []
    for candidate in candidates:
        path = Path(candidate).expanduser().resolve()
        examined.append(str(path))
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"could not discover {label}; checked: {', '.join(examined) or '(none)'}"
    )


def select_cases(
    cases: Sequence[dict[str, object]],
    case_ids: set[str],
    categories: set[str],
) -> list[dict[str, object]]:
    """Apply exact case-ID and category filters while preserving catalog order."""
    available_ids = {case.get("id") for case in cases}
    missing_ids = case_ids - available_ids
    if missing_ids:
        raise ValueError(
            "no benchmark cases can match unknown case filters: "
            f"{sorted(missing_ids)!r}"
        )
    selected = [
        case
        for case in cases
        if (not case_ids or case.get("id") in case_ids)
        and (not categories or case.get("category") in categories)
    ]
    if not selected:
        raise ValueError("no benchmark cases matched the requested filters")
    return selected


def scale_cases_for_smoke(
    cases: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Collapse a profile to one canonical small case per operation."""
    scaled_by_id: dict[str, dict[str, object]] = {}
    for source in cases:
        case = dict(source)
        operation = case.get("operation")
        dtype = case.get("dtype")
        axis = case.get("axis")
        if not isinstance(operation, str) or not isinstance(dtype, str) or type(axis) is not int:
            raise ValueError("smoke scaling requires canonical expanded cases")
        if case.get("category") == "bridge":
            size, rows, cols = 1, 0, 0
        elif type(case.get("rows")) is int and type(case.get("cols")) is int and case["rows"] > 0 and case["cols"] > 0:
            rows, cols = 2, 2
            size = rows * cols
        else:
            size, rows, cols = 8, 0, 0
        case.update({"size": size, "rows": rows, "cols": cols})
        case["id"] = format_case_id(
            operation,
            dtype,
            size=size,
            rows=rows,
            cols=cols,
            axis=axis,
        )
        validate_job_case(case)
        scaled_by_id.setdefault(case["id"], case)
    if not scaled_by_id:
        raise ValueError("smoke scaling requires at least one benchmark case")
    return [scaled_by_id[case_id] for case_id in sorted(scaled_by_id)]


def create_run_directory(output_root: str | Path, run_id: str) -> Path:
    """Create one exclusive run directory; collisions remain visible."""
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be one non-empty path component")
    destination = Path(output_root).expanduser().resolve() / run_id
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def atomic_write_text(path: str | Path, text: str) -> None:
    """Durably replace a UTF-8 text file from a same-directory temporary."""
    if not isinstance(text, str):
        raise TypeError("atomic text document must be a string")
    destination = Path(path)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def run_logged_command(
    command: Sequence[str | Path],
    cwd: str | Path,
    stdout_log: str | Path,
    stderr_log: str | Path,
    *,
    encoding: str,
) -> subprocess.CompletedProcess[str]:
    """Run a child, streaming both pipes while preserving complete UTF-8 logs."""
    if not command:
        raise ValueError("command must not be empty")
    normalized_command = [str(argument) for argument in command]
    if not isinstance(encoding, str) or not encoding:
        raise ValueError("child process encoding must be a non-empty string")
    with (
        Path(stdout_log).open("w", encoding="utf-8", newline="\n") as stdout_file,
        Path(stderr_log).open("w", encoding="utf-8", newline="\n") as stderr_file,
    ):
        process = subprocess.Popen(
            normalized_command,
            cwd=Path(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding=encoding,
            errors="strict",
            bufsize=1,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        pump_errors: list[BaseException] = []

        def pump(source: TextIO, log: TextIO, terminal: TextIO) -> None:
            try:
                for line in source:
                    log.write(line)
                    log.flush()
                    terminal.write(line)
                    terminal.flush()
            except BaseException as error:
                pump_errors.append(error)
                process.kill()

        stdout_thread = threading.Thread(
            target=pump,
            args=(process.stdout, stdout_file, sys.stdout),
            name="benchmark-stdout",
        )
        stderr_thread = threading.Thread(
            target=pump,
            args=(process.stderr, stderr_file, sys.stderr),
            name="benchmark-stderr",
        )
        stdout_thread.start()
        stderr_thread.start()
        return_code = process.wait()
        stdout_thread.join()
        stderr_thread.join()
        process.stdout.close()
        process.stderr.close()
    if pump_errors:
        raise pump_errors[0]
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, normalized_command)
    return subprocess.CompletedProcess(normalized_command, return_code)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 identity of a regular file."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"SHA-256 source is not a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_release_compiler_settings(project_file: str | Path) -> dict[str, str | None]:
    """Read effective Release|x64 ClCompile settings from a vcxproj."""
    project = ET.parse(project_file).getroot()
    settings: dict[str, str | None] = {}
    for group in project.findall("msbuild:ItemDefinitionGroup", _MSBUILD_NS):
        condition = group.attrib.get("Condition", "")
        if _RELEASE_X64_CONDITION.fullmatch(condition) is None:
            continue
        compiler = group.find("msbuild:ClCompile", _MSBUILD_NS)
        if compiler is None:
            continue
        for setting in compiler:
            name = setting.tag.split("}", 1)[-1]
            settings[name] = setting.text
    if not settings:
        raise ValueError(f"vcxproj has no Release|x64 compiler settings: {project_file}")
    return settings


def collect_environment(
    *,
    run_id: str,
    started_at_utc: str,
    profile: str,
    size_scale: str,
    case_ids: Sequence[str],
    categories: Sequence[str],
    warmups: int,
    samples: int,
    target_sample_ms: float,
    seed: int,
    runtime_order: tuple[str, str],
    build_requested: bool,
    project_root: str | Path,
    project_file: str | Path,
    python_executable: str | Path,
    ahk_executable: str | Path,
    msbuild_executable: str | Path | None,
    dll_path: str | Path,
    numpy_metadata: dict[str, object],
    cnumpy_metadata: dict[str, object],
    commands: dict[str, Sequence[str | Path]],
) -> dict[str, object]:
    """Capture the complete host, artifact, protocol, and command identity."""
    require_semantic_qualification_numpy_version(numpy_metadata)
    root = Path(project_root).resolve()
    project = Path(project_file).resolve()
    python = Path(python_executable).resolve()
    ahk = Path(ahk_executable).resolve()
    msbuild = Path(msbuild_executable).resolve() if msbuild_executable is not None else None
    dll = Path(dll_path).resolve()
    dll_stat = dll.stat()
    compiler_settings = read_release_compiler_settings(project)
    return {
        "schema_version": 2,
        "run_id": run_id,
        "started_at_utc": started_at_utc,
        "profile": profile,
        "size_scale": size_scale,
        "selection": {
            "case_ids": list(case_ids),
            "categories": list(categories),
        },
        "protocol": {
            "warmups": warmups,
            "sample_count": samples,
            "target_sample_ms": target_sample_ms,
            "seed": seed,
            "headline_includes_dllcall_overhead": True,
        },
        "runtime_order": list(runtime_order),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform_processor": platform.processor(),
            "processor_identifier": os.environ.get("PROCESSOR_IDENTIFIER"),
            "logical_processor_count": os.cpu_count(),
            "python_process_architecture": platform.architecture()[0],
        },
        "paths": {
            "project_root": str(root),
            "project": str(project),
            "python": str(python),
            "ahk": str(ahk),
            "msbuild": str(msbuild) if msbuild is not None else None,
            "dll": str(dll),
        },
        "artifacts": {
            "dll": {
                "sha256": sha256_file(dll),
                "size_bytes": dll_stat.st_size,
                "modified_at_utc": datetime.fromtimestamp(
                    dll_stat.st_mtime,
                    timezone.utc,
                ).isoformat(),
            }
        },
        "semantic_qualifications": semantic_qualification_declarations(),
        "runtimes": {
            "numpy": copy_dict(numpy_metadata),
            "cnumpy": copy_dict(cnumpy_metadata),
        },
        "commands": {
            name: [str(argument) for argument in command]
            for name, command in commands.items()
        },
        "build": {
            "requested": build_requested,
            "configuration": "Release",
            "platform": "x64",
            "compiler_settings": compiler_settings,
        },
    }


def copy_dict(value: dict[str, object]) -> dict[str, object]:
    """Copy metadata dictionaries so environment capture never aliases workers."""
    return {key: item for key, item in value.items()}


def discover_ahk_candidates(project_root: str | Path) -> list[Path]:
    """Return registry AHK candidates followed by the documented installation."""
    import winreg

    candidates: list[Path] = []
    access_modes = (winreg.KEY_READ | winreg.KEY_WOW64_64KEY, winreg.KEY_READ | winreg.KEY_WOW64_32KEY)
    for access in access_modes:
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\AutoHotkey",
                0,
                access,
            ) as key:
                install_dir, value_type = winreg.QueryValueEx(key, "InstallDir")
        except FileNotFoundError:
            continue
        if value_type not in {winreg.REG_SZ, winreg.REG_EXPAND_SZ}:
            raise ValueError("AutoHotkey InstallDir registry value must be a string")
        if not isinstance(install_dir, str) or not install_dir:
            raise ValueError("AutoHotkey InstallDir registry value must be non-empty")
        expanded = Path(os.path.expandvars(install_dir))
        candidates.extend((expanded / "AutoHotkey64.exe", expanded / "v2" / "AutoHotkey64.exe"))

    root = Path(project_root).resolve()
    if len(root.parents) < 4:
        raise ValueError(f"project path has no documented AutoHotkey installation parent: {root}")
    candidates.append(root.parents[3] / "AutoHotkey64.exe")
    return _unique_paths(candidates)


def discover_msbuild_candidates() -> list[Path]:
    """Return MSBuild candidates from vswhere and known VS 2022 editions."""
    candidates: list[Path] = []
    program_files_x86 = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    vswhere = program_files_x86 / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.is_file():
        completed = subprocess.run(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.Component.MSBuild",
                "-property",
                "installationPath",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding=locale.getpreferredencoding(False),
            errors="strict",
        )
        installation = completed.stdout.strip()
        if not installation:
            raise RuntimeError("vswhere returned an empty Visual Studio installationPath")
        candidates.append(Path(installation) / "MSBuild" / "Current" / "Bin" / "MSBuild.exe")

    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    for edition in ("Community", "Professional", "Enterprise", "BuildTools"):
        candidates.append(
            program_files
            / "Microsoft Visual Studio"
            / "2022"
            / edition
            / "MSBuild"
            / "Current"
            / "Bin"
            / "MSBuild.exe"
        )
    return _unique_paths(candidates)


def resolve_msbuild_executable(
    explicit: str | Path | None,
    *,
    build_requested: bool,
) -> Path | None:
    """Resolve MSBuild only when explicitly named or required for a build."""
    if explicit is not None:
        return resolve_explicit_or_candidates(explicit, [], "MSBuild executable")
    if not build_requested:
        return None
    return resolve_explicit_or_candidates(
        None,
        discover_msbuild_candidates(),
        "MSBuild executable",
    )


def write_json_atomic(path: str | Path, document: object) -> None:
    """Serialize strict JSON and publish it atomically."""
    text = json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    atomic_write_text(path, text)


def python_worker_prefix(executable: str | Path) -> list[str]:
    """Return a worker prefix without changing the installation's encoding mode."""
    return [str(Path(executable).resolve()), "-B"]


def run_benchmark(arguments: argparse.Namespace) -> Path:
    """Execute one complete benchmark and return its successful run directory."""
    project_root = Path(__file__).resolve().parents[1]
    benchmark_dir = project_root / "benchmark"
    project_file = _resolve_project_file(arguments.project_file, project_root)
    python_executable = resolve_explicit_or_candidates(
        arguments.python,
        [Path(sys.executable)],
        "Python executable",
    )
    ahk_executable = resolve_explicit_or_candidates(
        arguments.ahk,
        discover_ahk_candidates(project_root),
        "AHK executable",
    )
    msbuild_executable = resolve_msbuild_executable(
        arguments.msbuild,
        build_requested=arguments.build,
    )
    default_dll = project_root / "build" / "x64" / "Release" / "cnumpy_ahk.dll"
    if arguments.dll is not None:
        dll_path = resolve_explicit_or_candidates(arguments.dll, [], "cnumpy DLL")
    else:
        dll_path = default_dll.resolve()

    catalog = load_catalog(benchmark_dir / "cases.json")
    cases = expand_cases(catalog, arguments.profile)
    cases = select_cases(cases, set(arguments.case_ids), set(arguments.categories))
    if arguments.size_scale == "smoke":
        cases = scale_cases_for_smoke(cases)

    runtime_order = parse_runtime_order(arguments.runtime_order)
    baseline_path: Path | None = None
    if arguments.baseline is not None:
        baseline_path = resolve_explicit_or_candidates(
            arguments.baseline,
            [],
            "baseline comparison CSV",
        )

    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%S.%fZ")
    output_root = (
        Path(arguments.output_root).resolve()
        if arguments.output_root is not None
        else benchmark_dir / "results"
    )
    run_directory = create_run_directory(output_root, run_id)
    jobs_path = run_directory / "jobs.tsv"
    with jobs_path.open("w", encoding="utf-8", newline="") as jobs_file:
        write_jobs_tsv(cases, jobs_file)

    commands: dict[str, list[str]] = {}
    if arguments.build:
        assert msbuild_executable is not None
        build_command = [
            str(msbuild_executable),
            str(project_file),
            "/t:Rebuild",
            "/p:Configuration=Release",
            "/p:Platform=x64",
            "/m",
            "/nologo",
        ]
        commands["build"] = build_command
        print(f"Building Release|x64 with {msbuild_executable}")
        run_logged_command(
            build_command,
            project_root,
            run_directory / "build.stdout.log",
            run_directory / "build.stderr.log",
            encoding=locale.getpreferredencoding(False),
        )
    if not dll_path.is_file():
        raise FileNotFoundError(f"cnumpy DLL is not a regular file after build step: {dll_path}")

    worker_outputs = {
        "numpy": run_directory / "numpy.json",
        "cnumpy": run_directory / "cnumpy.json",
    }
    common_worker_arguments = [
        "--jobs",
        str(jobs_path),
        "--warmups",
        str(arguments.warmups),
        "--samples",
        str(arguments.samples),
        "--target-sample-ms",
        _format_cli_number(arguments.target_sample_ms),
        "--seed",
        str(arguments.seed),
    ]
    commands["numpy"] = [
        *python_worker_prefix(python_executable),
        str(benchmark_dir / "bench_numpy.py"),
        "--output",
        str(worker_outputs["numpy"]),
        *common_worker_arguments,
    ]
    commands["cnumpy"] = [
        str(ahk_executable),
        "/ErrorStdOut=UTF-8",
        str(benchmark_dir / "bench_cnumpy.ahk"),
        "--output",
        str(worker_outputs["cnumpy"]),
        "--dll",
        str(dll_path),
        *common_worker_arguments,
    ]

    case_ids = [str(case["id"]) for case in cases]
    raw_documents: dict[str, dict[str, object]] = {}
    for runtime in runtime_order:
        print(f"Running {runtime}: {len(cases)} case(s)")
        run_logged_command(
            commands[runtime],
            project_root,
            run_directory / f"{runtime}.stdout.log",
            run_directory / f"{runtime}.stderr.log",
            encoding=(
                "utf-8"
                if runtime == "cnumpy"
                else locale.getpreferredencoding(False)
            ),
        )
        raw_documents[runtime] = load_worker_json(
            worker_outputs[runtime],
            expected_runtime=runtime,
            expected_case_ids=case_ids,
        )
        _validate_worker_protocol(raw_documents[runtime], arguments, runtime)

    rows = compare_results(raw_documents["numpy"], raw_documents["cnumpy"])
    normalized_documents = {
        runtime: normalize_worker_result(
            raw_documents[runtime],
            expected_runtime=runtime,
            expected_case_ids=case_ids,
        )
        for runtime in ("numpy", "cnumpy")
    }
    for runtime, document in normalized_documents.items():
        write_json_atomic(worker_outputs[runtime], document)

    if baseline_path is not None:
        rows = attach_baseline(rows, load_baseline_csv(baseline_path))

    environment = collect_environment(
        run_id=run_id,
        started_at_utc=started.isoformat(),
        profile=arguments.profile,
        size_scale=arguments.size_scale,
        case_ids=case_ids,
        categories=sorted(set(arguments.categories)),
        warmups=arguments.warmups,
        samples=arguments.samples,
        target_sample_ms=arguments.target_sample_ms,
        seed=arguments.seed,
        runtime_order=runtime_order,
        build_requested=arguments.build,
        project_root=project_root,
        project_file=project_file,
        python_executable=python_executable,
        ahk_executable=ahk_executable,
        msbuild_executable=msbuild_executable,
        dll_path=dll_path,
        numpy_metadata=raw_documents["numpy"]["metadata"],
        cnumpy_metadata=raw_documents["cnumpy"]["metadata"],
        commands=commands,
    )
    environment["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    if baseline_path is not None:
        environment["baseline"] = str(baseline_path)

    write_json_atomic(run_directory / "environment.json", environment)
    atomic_write_text(run_directory / "comparison.csv", render_comparison_csv(rows))
    atomic_write_text(
        run_directory / "comparison.md",
        render_markdown(rows, environment=environment),
    )
    _print_terminal_summary(rows, run_directory)
    return run_directory


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("focus", "standard", "full"), default="focus")
    parser.add_argument("--build", action="store_true", help="rebuild Release|x64 before timing")
    parser.add_argument("--warmups", type=_non_negative_integer, default=5)
    parser.add_argument("--samples", type=_positive_odd_integer, default=15)
    parser.add_argument("--target-sample-ms", type=_positive_finite_number, default=20.0)
    parser.add_argument("--seed", type=_seed_integer, default=12345)
    parser.add_argument("--size-scale", choices=("native", "smoke"), default="native")
    parser.add_argument("--case", dest="case_ids", action="append", default=[])
    parser.add_argument("--category", dest="categories", action="append", default=[])
    parser.add_argument("--runtime-order", default="numpy,cnumpy")
    parser.add_argument("--python", type=Path)
    parser.add_argument("--ahk", type=Path)
    parser.add_argument("--dll", type=Path)
    parser.add_argument("--msbuild", type=Path)
    parser.add_argument("--project-file", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--baseline", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one benchmark from CLI arguments."""
    arguments = _argument_parser().parse_args(argv)
    try:
        arguments.runtime_order = ",".join(parse_runtime_order(arguments.runtime_order))
    except ValueError as error:
        _argument_parser().error(str(error))
    run_benchmark(arguments)
    return 0


def _resolve_project_file(explicit: Path | None, project_root: Path) -> Path:
    return resolve_explicit_or_candidates(
        explicit,
        [project_root / "src" / "cnumpy_ahk.vcxproj"],
        "vcxproj",
    )


def _validate_worker_protocol(
    document: dict[str, object],
    arguments: argparse.Namespace,
    runtime: str,
) -> None:
    metadata = document["metadata"]
    assert isinstance(metadata, dict)
    expected = {
        "warmups": arguments.warmups,
        "sample_count": arguments.samples,
        "target_sample_ns": arguments.target_sample_ms * 1_000_000,
        "seed": arguments.seed,
    }
    for field, expected_value in expected.items():
        actual = metadata.get(field)
        if actual != expected_value:
            raise ValueError(
                f"{runtime} metadata {field!r} is {actual!r}, expected {expected_value!r}"
            )


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _non_negative_integer(argument: str) -> int:
    try:
        value = int(argument)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from error
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def _positive_odd_integer(argument: str) -> int:
    try:
        value = int(argument)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive odd integer") from error
    if value <= 0 or value % 2 == 0:
        raise argparse.ArgumentTypeError("must be a positive odd integer")
    return value


def _positive_finite_number(argument: str) -> float:
    try:
        value = float(argument)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a finite positive number") from error
    if not math_is_finite_positive(value):
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def _seed_integer(argument: str) -> int:
    value = _non_negative_integer(argument)
    if value > 2_147_483_647:
        raise argparse.ArgumentTypeError("must be in [0, 2147483647]")
    return value


def math_is_finite_positive(value: float) -> bool:
    import math

    return math.isfinite(value) and value > 0


def _format_cli_number(value: float) -> str:
    return format(value, ".17g")


def _print_terminal_summary(rows: Sequence[dict[str, object]], run_directory: Path) -> None:
    ranked = sorted(rows, key=lambda row: float(row["cnumpy_over_numpy"]), reverse=True)
    print("\nTop cnumpy/NumPy ratios:")
    for row in ranked[: min(10, len(ranked))]:
        print(f"  {float(row['cnumpy_over_numpy']):8.3f}x  {row['id']}")
    print(f"\nReport: {run_directory / 'comparison.md'}")


if __name__ == "__main__":
    raise SystemExit(main())
