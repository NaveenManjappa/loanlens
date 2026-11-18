"""Guards for the layer rules in README "Project structure"."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "loanlens"
FRAMEWORKS = ("langchain", "langchain_core", "langchain_openai", "langgraph", "fastapi", "openai")


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_services_and_domain_have_no_framework_imports():
    for layer in ("services", "domain"):
        for path in (SRC / layer).glob("*.py"):
            for module in imported_modules(path):
                assert module.split(".")[0] not in FRAMEWORKS, f"{path.name} imports {module}"


def test_inner_layers_do_not_import_outer_ones():
    outer = {
        "services": ("loanlens.api", "loanlens.graph", "loanlens.agents", "loanlens.tools"),
        "domain": ("loanlens.",),
        "integrations": ("loanlens.api", "loanlens.graph", "loanlens.agents", "loanlens.tools"),
        "tools": ("loanlens.api", "loanlens.graph", "loanlens.agents"),
    }
    for layer, forbidden in outer.items():
        for path in (SRC / layer).glob("*.py"):
            for module in imported_modules(path):
                assert not module.startswith(forbidden), f"{layer}/{path.name} imports {module}"
