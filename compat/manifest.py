from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


_COMMENTS = re.compile(r"/\*.*?\*/|//[^\r\n]*", re.DOTALL)
_DECLARATION = re.compile(
    r"\bCNP_API\b"
    r"(?P<return_type>[^;]*?)"
    r"\bCNP_CALL\s+"
    r"(?P<symbol>cnp_[A-Za-z0-9_]+)\s*"
    r"\((?P<parameters>[^;]*?)\)\s*;",
    re.DOTALL,
)


@dataclass(frozen=True)
class PublicDeclaration:
    symbol: str
    return_type: str
    parameters: str


@dataclass(frozen=True)
class ManifestExport:
    symbol: str
    family: str
    reference: str | None
    status: str
    result: str
    tests: tuple[str, ...]


@dataclass(frozen=True)
class CompatibilityManifest:
    schema_version: int
    declaration_count: int
    python_version: str
    numpy_version: str
    exports: tuple[ManifestExport, ...]


def _collapse_whitespace(value: str) -> str:
    return " ".join(value.split())


_C_TYPE_WORDS = {
    "bool",
    "char",
    "const",
    "double",
    "float",
    "int",
    "long",
    "short",
    "signed",
    "size_t",
    "unsigned",
    "void",
}


def _canonical_abi_text(value: str) -> str:
    """Remove parameter names while retaining C type and ABI tokens."""

    def keep_or_remove(match: re.Match[str]) -> str:
        token = match.group(0)
        if (
            token in _C_TYPE_WORDS
            or token.endswith("_t")
            or token.startswith("Cnp")
            or token.startswith("CNP_")
        ):
            return token
        return ""

    without_names = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", keep_or_remove, value)
    return re.sub(r"\s+", "", without_names)


def parse_public_declarations(header: str) -> tuple[PublicDeclaration, ...]:
    """Parse exported C declarations from the public header.

    This deliberately parses only the stable `CNP_API ... CNP_CALL ...;`
    declaration shape used by cnumpy. Function bodies, macros without a
    semicolon, comments, and internal declarations are excluded.
    """

    without_comments = _COMMENTS.sub(" ", header)
    declarations: dict[str, PublicDeclaration] = {}
    for match in _DECLARATION.finditer(without_comments):
        declaration = PublicDeclaration(
            symbol=match.group("symbol"),
            return_type=_collapse_whitespace(match.group("return_type")),
            parameters=_collapse_whitespace(match.group("parameters")),
        )
        existing = declarations.get(declaration.symbol)
        if existing is not None and (
            existing.return_type != declaration.return_type
            or _canonical_abi_text(existing.parameters)
            != _canonical_abi_text(declaration.parameters)
        ):
            raise ValueError(
                "conflicting public declarations for "
                f"{declaration.symbol}: {existing!r} != {declaration!r}"
            )
        declarations[declaration.symbol] = declaration
    return tuple(declarations.values())


def _required_string(mapping: dict[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{key} must be a nonempty string")
    return value


def load_manifest(path: Path) -> CompatibilityManifest:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("manifest root must be an object")

    reference = document.get("reference")
    if not isinstance(reference, dict):
        raise ValueError("manifest.reference must be an object")

    raw_groups = document.get("groups")
    if not isinstance(raw_groups, list):
        raise ValueError("manifest.groups must be an array")

    header_path = path.parent.parent / "include" / "cnumpy" / "cnumpy.h"
    declarations_by_symbol = {
        item.symbol: item
        for item in parse_public_declarations(
            header_path.read_text(encoding="utf-8")
        )
    }

    exports_by_symbol: dict[str, ManifestExport] = {}
    symbol_order: list[str] = []
    for group_index, group in enumerate(raw_groups):
        context = f"groups[{group_index}]"
        if not isinstance(group, dict):
            raise ValueError(f"{context} must be an object")
        family = _required_string(group, "family", context)
        status = _required_string(group, "status", context)
        result = _required_string(group, "result", context)
        override = group.get("override", False)
        if not isinstance(override, bool):
            raise ValueError(f"{context}.override must be a boolean")

        raw_tests = group.get("tests")
        if (
            not isinstance(raw_tests, list)
            or not raw_tests
            or any(not isinstance(item, str) or not item for item in raw_tests)
        ):
            raise ValueError(f"{context}.tests must contain nonempty strings")
        tests = tuple(raw_tests)

        raw_reference = group.get("reference")
        if raw_reference is not None and not isinstance(raw_reference, str):
            raise ValueError(f"{context}.reference must be a string or null")

        raw_symbols = group.get("symbols")
        if isinstance(raw_symbols, str):
            raw_symbols = raw_symbols.split()
        if not isinstance(raw_symbols, list) or not raw_symbols:
            raise ValueError(
                f"{context}.symbols must be a nonempty array or symbol string"
            )
        for symbol_index, symbol in enumerate(raw_symbols):
            if not isinstance(symbol, str) or not symbol:
                raise ValueError(
                    f"{context}.symbols[{symbol_index}] must be a nonempty string"
                )
            operation = symbol.removeprefix("cnp_")
            resolved_result = result
            if result == "auto":
                declaration = declarations_by_symbol.get(symbol)
                if declaration is None:
                    raise ValueError(f"{context} contains unknown symbol {symbol}")
                if declaration.return_type == "CnpArray*":
                    resolved_result = "array"
                elif declaration.return_type == "CNP_STATUS":
                    resolved_result = "status"
                elif declaration.return_type == "void":
                    resolved_result = "void"
                else:
                    resolved_result = "scalar"
            resolved_reference = (
                raw_reference.replace("{operation}", operation)
                if raw_reference is not None
                else None
            )
            if symbol in exports_by_symbol and not override:
                raise ValueError(
                    f"{context} duplicates {symbol} without override=true"
                )
            if symbol not in exports_by_symbol:
                symbol_order.append(symbol)
            exports_by_symbol[symbol] = ManifestExport(
                symbol=symbol,
                family=family,
                reference=resolved_reference,
                status=status,
                result=resolved_result,
                tests=tests,
            )

    return CompatibilityManifest(
        schema_version=int(document["schema_version"]),
        declaration_count=int(document["declaration_count"]),
        python_version=_required_string(reference, "python", "reference"),
        numpy_version=_required_string(reference, "numpy", "reference"),
        exports=tuple(exports_by_symbol[symbol] for symbol in symbol_order),
    )
