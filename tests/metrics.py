from __future__ import annotations

import ast
from pathlib import Path


# ============================================================
# BLOCK 1: LOC (excluding docstrings, comment-only lines, blanks)
# ============================================================
def loc_without_docstrings(path: Path) -> int:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)

    # Collect docstring ranges (lineno..end_lineno) from module + functions + classes
    doc_ranges: set[tuple[int, int]] = set()

    def add_doc_range(node: ast.AST) -> None:
        doc = ast.get_docstring(node, clean=False)
        if not doc:
            return

        # A docstring is always the first statement in the body, if present
        body = getattr(node, "body", None)
        if not body:
            return

        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(
            getattr(first, "value", None), (ast.Str, ast.Constant)
        ):
            if hasattr(first, "lineno") and hasattr(first, "end_lineno") and first.end_lineno:
                doc_ranges.add((first.lineno, first.end_lineno))

    # Module-level docstring
    add_doc_range(tree)

    # Function and class docstrings
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add_doc_range(node)

    def is_in_docstring(lno: int) -> bool:
        return any(start <= lno <= end for start, end in doc_ranges)

    count = 0
    for i, line in enumerate(src.splitlines(), start=1):
        s = line.strip()
        if not s:               # Skip blank lines
            continue
        if s.startswith("#"):   # Skip comment-only lines
            continue
        if is_in_docstring(i):  # Skip docstring lines
            continue
        count += 1

    return count


# ============================================================
# BLOCK 2: Max nesting depth (AST block nodes)
# ============================================================
BLOCK_NODES = (
    ast.If,
    ast.For,
    ast.While,
    ast.With,
    ast.Try,
    ast.Match,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)


class NestingVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.max_depth = 0
        self.cur_depth = 0

    def generic_visit(self, node):
        is_block = isinstance(node, BLOCK_NODES)
        if is_block:
            # Entering a new block increases current nesting depth
            self.cur_depth += 1
            self.max_depth = max(self.max_depth, self.cur_depth)

        super().generic_visit(node)

        if is_block:
            # Leaving a block decreases current nesting depth
            self.cur_depth -= 1


def max_nesting_depth(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    v = NestingVisitor()
    v.visit(tree)
    # Subtract 1 so the file/module level is not counted as depth 1
    return max(0, v.max_depth - 1)


# ============================================================
# BLOCK 3: Helpers for average function length
#   - collect docstring ranges
#   - compute excluded line numbers (blank/comment/docstring)
# ============================================================
def _docstring_ranges(tree: ast.AST) -> set[tuple[int, int]]:
    """Return (start,end) line ranges for module/class/function docstrings."""
    doc_ranges: set[tuple[int, int]] = set()

    def add_doc_range(node: ast.AST) -> None:
        doc = ast.get_docstring(node, clean=False)
        if not doc:
            return

        body = getattr(node, "body", None)
        if not body:
            return

        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(
            getattr(first, "value", None), (ast.Str, ast.Constant)
        ):
            if hasattr(first, "lineno") and hasattr(first, "end_lineno") and first.end_lineno:
                doc_ranges.add((first.lineno, first.end_lineno))

    add_doc_range(tree)  # module docstring
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add_doc_range(node)

    return doc_ranges


def _excluded_lines(src: str, doc_ranges: set[tuple[int, int]]) -> set[int]:
    """Lines to exclude: blank, comment-only, and docstring lines."""
    def is_in_docstring(lno: int) -> bool:
        return any(start <= lno <= end for start, end in doc_ranges)

    excluded: set[int] = set()
    for i, line in enumerate(src.splitlines(), start=1):
        s = line.strip()
        if not s:              # blank
            excluded.add(i)
            continue
        if s.startswith("#"):  # comment-only line
            excluded.add(i)
            continue
        if is_in_docstring(i):  # docstring line
            excluded.add(i)
            continue

    return excluded


# ============================================================
# BLOCK 4: Average function length (LOC) per file
#   - counts LOC per function (excluding blank/comments/docstrings)
#   - averages over all functions in the file
# ============================================================
def avg_function_length(path: Path, *, top_level_only: bool = False) -> float:
    """
    Average function length (LOC) excluding blank lines, comment-only lines, and docstring lines.

    If top_level_only=True, only counts module-level functions (no methods, no nested defs).
    """
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)

    doc_ranges = _docstring_ranges(tree)
    excluded = _excluded_lines(src, doc_ranges)

    lengths: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", None)
            if not end:
                continue

            if top_level_only and getattr(node, "col_offset", 0) != 0:
                continue

            start = node.lineno
            loc = sum(1 for lno in range(start, end + 1) if lno not in excluded)
            lengths.append(loc)

    return (sum(lengths) / len(lengths)) if lengths else 0.0


# ============================================================
# BLOCK 5: CLI entrypoint / comparison output
# ============================================================
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    files = [
        ROOT / "src" / "imperative_version.py",
        ROOT / "src" / "functional_version.py",
    ]

    for f in files:
        print(f"{f.name}:")
        print(f"  LOC (no docstrings/comments/blank) = {loc_without_docstrings(f)}")
        print(f"  Avg function length (LOC)          = {avg_function_length(f):.2f}")
        print(f"  Max nesting depth (AST blocks)     = {max_nesting_depth(f)}")


