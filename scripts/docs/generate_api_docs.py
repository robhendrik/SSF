from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

CANONICAL_MODULES = [
    "src/ssf/hybrid_strategy_analytics.py",
    "src/ssf/hybrid_strategy_fixtures.py",
    "src/ssf/hybrid_strategy_procedural.py",
    "src/ssf/hybrid_strategy_spec.py",
    "src/ssf/strategy_cache.py",
    "src/ssf/strategy_evaluation.py",
    "src/ssf/strategy_optimizer.py",
    "src/ssf/strategy_search.py",
]

CANONICAL_SCRIPTS = [
    "scripts/compare/plot_analytics_family_curves.py",
    "scripts/generate/generate_a_strategy_fixtures.py",
    "scripts/optimize/run_beam_strategy_optimizer.py",
    "scripts/optimize/run_exhaustive_strategy_optimizer.py",
    "scripts/validate/compare_strategy_evaluation_backends.py",
    "scripts/validate/rebuild_evaluation_cache.py",
    "scripts/validate/validate_strategy_cache_behavior.py",
]

MODULE_ROLE_MAP = {
    "hybrid_strategy_spec": "Canonical strategy schema, validation, naming, and structural key ownership.",
    "hybrid_strategy_fixtures": "Dense fixture construction and serialization from canonical strategy specs.",
    "hybrid_strategy_procedural": "Procedural strategy execution contract equivalent to dense fixture semantics.",
    "hybrid_strategy_analytics": "Closed-form analytical strategy scoring backend.",
    "strategy_evaluation": "Backend-agnostic evaluator dispatcher for analytical, exhaustive, and Monte Carlo modes.",
    "strategy_cache": "Evaluation cache schema, lookup filters, replacement policy, and persistence.",
    "strategy_search": "Legal strategy enumeration and exhaustive ranking helper layer.",
    "strategy_optimizer": "Reusable exhaustive and beam optimizer implementations over canonical specs.",
}


@dataclass(frozen=True)
class NamedItem:
    name: str
    purpose: str


@dataclass(frozen=True)
class CliOption:
    option: str
    type_name: str
    default: str
    required: str
    choices: str
    description: str


def _read_tree(path: Path) -> tuple[ast.Module, str]:
    source = path.read_text(encoding="utf-8-sig")
    return ast.parse(source), source


def _first_sentence(text: str | None) -> str:
    if not text:
        return ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _is_dataclass(class_def: ast.ClassDef) -> bool:
    for dec in class_def.decorator_list:
        node = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(node, ast.Name) and node.id == "dataclass":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "dataclass":
            return True
    return False


def _ast_to_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _ast_to_name(node.value)
        if base:
            return f"{base}.{node.attr}"
        return node.attr
    return ast.unparse(node)


def _literal_eval(node: ast.AST, constants: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in constants:
            return constants[node.id]
        raise ValueError(f"Unknown name: {node.id}")
    if isinstance(node, ast.List):
        return [_literal_eval(elt, constants) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_literal_eval(elt, constants) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _literal_eval(k, constants): _literal_eval(v, constants)
            for k, v in zip(node.keys, node.values)
            if k is not None
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _literal_eval(node.operand, constants)
        if isinstance(value, (int, float)):
            return -value
    raise ValueError("Non-literal")


def _display_value(node: ast.AST | None, constants: dict[str, Any]) -> str:
    if node is None:
        return ""
    try:
        value = _literal_eval(node, constants)
    except ValueError:
        return ast.unparse(node)

    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return str(value)
    if isinstance(value, tuple):
        return ", ".join(str(v) for v in value)
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    if isinstance(value, dict):
        keys = sorted(value.keys(), key=str)
        return ", ".join(f"{k}={value[k]}" for k in keys)
    return str(value)


def _collect_constants(module: ast.Module) -> dict[str, Any]:
    constants: dict[str, Any] = {}
    for node in module.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if not name.isupper():
                continue
            try:
                constants[name] = _literal_eval(node.value, constants)
            except ValueError:
                continue
    return constants


def _extract_docstring_section(doc: str, headings: set[str]) -> list[str]:
    lines = doc.splitlines()
    lowered = {h.lower() for h in headings}

    def is_heading(idx: int) -> bool:
        line = lines[idx].strip()
        if not line:
            return False
        if line.startswith("#"):
            text = line.lstrip("#").strip().lower()
            return text in lowered
        text = line.lower()
        if text in lowered:
            return True
        if idx + 1 < len(lines):
            underline = lines[idx + 1].strip()
            if underline and all(ch == "-" for ch in underline) and text in lowered:
                return True
        return False

    start = -1
    i = 0
    while i < len(lines):
        if is_heading(i):
            start = i + 1
            if i + 1 < len(lines):
                underline = lines[i + 1].strip()
                if underline and all(ch == "-" for ch in underline):
                    start = i + 2
            break
        i += 1

    if start < 0:
        return []

    out: list[str] = []
    for j in range(start, len(lines)):
        line = lines[j].rstrip()
        stripped = line.strip()
        if not stripped:
            if out:
                out.append("")
            continue
        if stripped.startswith("#"):
            break
        if j + 1 < len(lines):
            underline = lines[j + 1].strip()
            if underline and all(ch == "-" for ch in underline):
                break
        out.append(stripped)

    while out and out[-1] == "":
        out.pop()
    return out


def _extract_design_invariants(module_doc: str) -> list[str]:
    if not module_doc:
        return []
    primary = _extract_docstring_section(module_doc, {"Design invariants", "Invariants"})
    if primary:
        return primary
    assumptions = _extract_docstring_section(module_doc, {"Vertical assumption note", "Assumptions"})
    return assumptions


def _normalize_invariant_lines(lines: list[str]) -> list[str]:
    if not lines:
        return []

    paragraphs: list[str] = []
    current: list[str] = []

    for line in lines:
        text = line.strip()
        if not text:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if text.startswith("- "):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            paragraphs.append(text[2:].strip())
            continue
        if not current and paragraphs:
            paragraphs[-1] = f"{paragraphs[-1]} {text}".strip()
            continue
        current.append(text)

    if current:
        paragraphs.append(" ".join(current))

    return paragraphs


def _extract_module_items(module: ast.Module) -> tuple[list[NamedItem], list[NamedItem]]:
    dataclasses: list[NamedItem] = []
    functions: list[NamedItem] = []

    for node in module.body:
        if isinstance(node, ast.ClassDef) and _is_public(node.name) and _is_dataclass(node):
            dataclasses.append(
                NamedItem(name=node.name, purpose=_first_sentence(ast.get_docstring(node)))
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(node.name):
            functions.append(
                NamedItem(name=node.name, purpose=_first_sentence(ast.get_docstring(node)))
            )

    dataclasses = sorted(dataclasses, key=lambda item: item.name)
    functions = sorted(functions, key=lambda item: item.name)
    return dataclasses, functions


def _extract_canonical_imports(module: ast.Module) -> list[str]:
    imported: set[str] = set()
    for node in module.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            module_name = node.module
            leaf = module_name.split(".")[-1]
            if leaf in MODULE_ROLE_MAP:
                imported.add(leaf)
            if module_name.startswith("."):
                continue
            if module_name.startswith("ssf."):
                leaf = module_name.split(".")[-1]
                if leaf in MODULE_ROLE_MAP:
                    imported.add(leaf)
    return sorted(imported)


def _find_parse_args_function(module: ast.Module) -> ast.FunctionDef | None:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_parse_args":
            return node
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                    if sub.func.attr == "add_argument":
                        return node
    return None


def _extract_parser_description(parse_fn: ast.FunctionDef, constants: dict[str, Any]) -> str:
    for node in ast.walk(parse_fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "ArgumentParser":
                for kw in node.keywords:
                    if kw.arg == "description":
                        return _display_value(kw.value, constants)
    return ""


def _parse_cli_options(parse_fn: ast.FunctionDef, constants: dict[str, Any]) -> list[CliOption]:
    options: list[CliOption] = []

    for node in ast.walk(parse_fn):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue

        flags = [
            _display_value(arg, constants)
            for arg in node.args
            if isinstance(arg, (ast.Constant, ast.Name, ast.UnaryOp, ast.List, ast.Tuple, ast.Dict, ast.Attribute, ast.Call, ast.BinOp))
        ]
        flags = [flag for flag in flags if flag]
        if not flags:
            continue

        kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}

        action = _display_value(kwargs.get("action"), constants)
        type_name = ""
        if "type" in kwargs:
            type_name = _ast_to_name(kwargs["type"])
        elif action in {"store_true", "store_false"}:
            type_name = "flag"

        if "default" in kwargs:
            default = _display_value(kwargs["default"], constants)
        elif action == "store_true":
            default = "False"
        elif action == "store_false":
            default = "True"
        else:
            default = ""

        required = _display_value(kwargs.get("required"), constants) if "required" in kwargs else "False"
        choices = _display_value(kwargs.get("choices"), constants) if "choices" in kwargs else ""
        description = _display_value(kwargs.get("help"), constants) if "help" in kwargs else ""

        option = ", ".join(flags)
        options.append(
            CliOption(
                option=option,
                type_name=type_name,
                default=default,
                required=required,
                choices=choices,
                description=description,
            )
        )

    def sort_key(item: CliOption) -> tuple[str, str]:
        split_flags = [part.strip() for part in item.option.split(",")]
        long_flags = [flag for flag in split_flags if flag.startswith("--")]
        primary = long_flags[0] if long_flags else split_flags[0]
        return primary, item.option

    dedup: dict[str, CliOption] = {}
    for item in options:
        dedup[item.option] = item

    return sorted(dedup.values(), key=sort_key)


def _md_table(items: list[NamedItem]) -> list[str]:
    if not items:
        return ["| Name | Purpose |", "|---|---|", "| (none) |  |"]
    lines = ["| Name | Purpose |", "|---|---|"]
    for item in items:
        purpose = item.purpose.replace("|", "\\|") if item.purpose else ""
        lines.append(f"| {item.name} | {purpose} |")
    return lines


def _md_cli_table(options: list[CliOption]) -> list[str]:
    if not options:
        return ["| Option | Type | Default | Required | Choices | Description |", "|---|---|---|---|---|---|", "| (none) |  |  |  |  |  |"]
    lines = [
        "| Option | Type | Default | Required | Choices | Description |",
        "|---|---|---|---|---|---|",
    ]
    for opt in options:
        row = [
            opt.option,
            opt.type_name,
            opt.default,
            opt.required,
            opt.choices,
            opt.description,
        ]
        escaped = [col.replace("|", "\\|") for col in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return lines


def _module_doc_path(module_stem: str) -> Path:
    return REPO_ROOT / "docs" / "api" / "src" / f"{module_stem}.md"


def _script_doc_path(script_stem: str) -> Path:
    return REPO_ROOT / "docs" / "api" / "scripts" / f"{script_stem}.md"


def _generate_module_doc(module_rel: str) -> str:
    module_path = REPO_ROOT / module_rel
    tree, _ = _read_tree(module_path)
    module_stem = module_path.stem

    module_doc = ast.get_docstring(tree) or ""
    purpose = _first_sentence(module_doc)
    dataclasses, functions = _extract_module_items(tree)
    invariants = _normalize_invariant_lines(_extract_design_invariants(module_doc))
    see_also = _extract_canonical_imports(tree)

    lines: list[str] = []
    lines.append(f"# {module_stem}")
    lines.append("")
    lines.append("<!-- Generated by scripts/docs/generate_api_docs.py. Do not edit manually. -->")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(purpose if purpose else "No module docstring summary available.")
    lines.append("")
    lines.append("## Role in canonical architecture")
    lines.append("")
    lines.append(MODULE_ROLE_MAP.get(module_stem, "Canonical module in SSF architecture."))
    lines.append("")
    lines.append("## Key dataclasses")
    lines.append("")
    lines.extend(_md_table(dataclasses))
    lines.append("")
    lines.append("## Key functions")
    lines.append("")
    lines.extend(_md_table(functions))
    lines.append("")
    lines.append("## Design invariants")
    lines.append("")
    if invariants:
        for item in invariants:
            if item:
                lines.append(f"- {item}")
            else:
                lines.append("")
    else:
        lines.append("- No explicit design invariant section found in module docstring.")
    lines.append("")
    lines.append("## See also")
    lines.append("")
    if see_also:
        for name in see_also:
            lines.append(f"- [{name}]({name}.md)")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _usage_example(script_rel: str, options: list[CliOption]) -> str:
    mode_opt = None
    for opt in options:
        if "--mode" in opt.option:
            mode_opt = opt
            break

    if mode_opt and mode_opt.choices:
        choices = [c for c in mode_opt.choices.replace(",", " ").split() if c]
        mode_value = "analytical" if "analytical" in choices else choices[0]
        return f"python {script_rel} --mode {mode_value}"

    return f"python {script_rel} --help"


def _generate_script_doc(script_rel: str) -> str:
    script_path = REPO_ROOT / script_rel
    tree, _ = _read_tree(script_path)
    constants = _collect_constants(tree)

    module_doc = ast.get_docstring(tree) or ""
    parse_fn = _find_parse_args_function(tree)
    parser_description = _extract_parser_description(parse_fn, constants) if parse_fn else ""
    purpose = _first_sentence(module_doc) or parser_description or "No module or parser description available."

    options = _parse_cli_options(parse_fn, constants) if parse_fn else []
    see_also = _extract_canonical_imports(tree)

    lines: list[str] = []
    lines.append(f"# {script_path.name}")
    lines.append("")
    lines.append("<!-- Generated by scripts/docs/generate_api_docs.py. Do not edit manually. -->")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(purpose)
    lines.append("")
    lines.append("## CLI usage")
    lines.append("")
    lines.append("```bash")
    lines.append(_usage_example(script_rel, options))
    lines.append("```")
    lines.append("")
    lines.append("## CLI options")
    lines.append("")
    lines.extend(_md_cli_table(options))
    lines.append("")
    lines.append("## See also")
    lines.append("")
    if see_also:
        for name in see_also:
            lines.append(f"- [{name}](../src/{name}.md)")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = content.replace("\r\n", "\n")
    path.write_text(normalized, encoding="utf-8", newline="\n")


def generate() -> None:
    module_paths = sorted(CANONICAL_MODULES)
    script_paths = sorted(CANONICAL_SCRIPTS)

    for rel in module_paths:
        module_stem = Path(rel).stem
        _write_text(_module_doc_path(module_stem), _generate_module_doc(rel))

    for rel in script_paths:
        script_stem = Path(rel).stem
        _write_text(_script_doc_path(script_stem), _generate_script_doc(rel))


if __name__ == "__main__":
    generate()
