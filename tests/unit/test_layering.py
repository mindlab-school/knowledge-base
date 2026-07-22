"""Import-direction guard between the ``search`` and ``ingestion`` packages.

The ``Embedder`` abstraction lives in the neutral :mod:`kb.embeddings` module and
the facts write-service in :mod:`kb.services.facts`, so neither ``kb.search`` nor
``kb.ingestion`` may import the other. This test parses the source of every module
in both packages with :mod:`ast` (no import side effects, so heavy optional deps
stay untouched) and fails on any cross-package import — including type-only ones
guarded by ``TYPE_CHECKING``.
"""

from __future__ import annotations

import ast
from pathlib import Path

_KB_ROOT = Path(__file__).resolve().parents[2] / "src" / "kb"
_SEARCH_DIR = _KB_ROOT / "search"
_INGESTION_DIR = _KB_ROOT / "ingestion"


def _module_name(path: Path) -> str:
    """Fully-qualified module name for a source file under ``src``."""
    rel = path.relative_to(_KB_ROOT.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve(package: str, level: int, module: str | None) -> str:
    """Resolve a possibly-relative import target to an absolute module path."""
    if level == 0:
        return module or ""
    base_parts = package.split(".")
    if level > 1:
        base_parts = base_parts[: -(level - 1)]
    base = ".".join(base_parts)
    return f"{base}.{module}" if module else base


def _imported_targets(path: Path) -> set[str]:
    """Absolute module targets imported by ``path`` (handles relative imports)."""
    module = _module_name(path)
    package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve(package, node.level, node.module)
            targets.add(base)
            # Also record ``base.<name>`` so the package-import form
            # (``from kb import ingestion``) is caught, not only ``from
            # kb.ingestion import x``.
            targets.update(f"{base}.{alias.name}" if base else alias.name for alias in node.names)
    return targets


def _cross_package_imports(package_dir: Path, forbidden: str) -> list[str]:
    """Return ``"<module> imports <target>"`` for each forbidden cross-import."""
    files = sorted(package_dir.rglob("*.py"))
    assert files, f"no source files found under {package_dir}"
    violations: list[str] = []
    for path in files:
        for target in _imported_targets(path):
            if target == forbidden or target.startswith(f"{forbidden}."):
                violations.append(f"{_module_name(path)} imports {target}")
    return violations


def test_search_does_not_import_ingestion() -> None:
    violations = _cross_package_imports(_SEARCH_DIR, "kb.ingestion")
    assert not violations, "search must not depend on ingestion: " + "; ".join(violations)


def test_ingestion_does_not_import_search() -> None:
    violations = _cross_package_imports(_INGESTION_DIR, "kb.search")
    assert not violations, "ingestion must not depend on search: " + "; ".join(violations)
