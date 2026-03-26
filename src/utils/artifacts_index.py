from __future__ import annotations

import html
from pathlib import Path


_EXCLUDED_NAMES = {".DS_Store"}
_EXCLUDED_SUFFIXES = {".zip"}


def _iter_entries(run_dir: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    for path in sorted(run_dir.rglob("*")):
        if path.name in _EXCLUDED_NAMES:
            continue
        if path.suffix.lower() in _EXCLUDED_SUFFIXES:
            continue
        rel = path.relative_to(run_dir)
        if rel.parts and rel.parts[0] == "artifacts":
            continue
        entries.append((str(rel).replace("\\", "/"), path))
    return entries


def _build_tree(items: list[tuple[str, Path]]) -> dict[str, dict]:
    root: dict[str, dict] = {}
    for rel, abs_path in items:
        parts = rel.split("/")
        cursor = root
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = {"__file__": abs_path, "__rel__": rel}
    return root


def _render_tree(tree: dict[str, dict], depth: int = 0) -> str:
    indent = "  " * depth
    lines = [f"{indent}<ul class='tree'>"]
    for name in sorted(tree.keys(), key=lambda item: (1 if "__file__" in tree[item] else 0, item.lower())):
        node = tree[name]
        safe_name = html.escape(name)
        if "__file__" in node:
            rel_href = "../" + str(node["__rel__"])
            lines.append(
                f"{indent}  <li class='file'><a href='{html.escape(rel_href)}'>{safe_name}</a></li>"
            )
        else:
            lines.append(f"{indent}  <li class='folder'><details open><summary>{safe_name}</summary>")
            lines.append(_render_tree(node, depth + 2))
            lines.append(f"{indent}  </details></li>")
    lines.append(f"{indent}</ul>")
    return "\n".join(lines)


def write_artifacts_index(run_dir: Path) -> Path:
    entries = _iter_entries(run_dir)
    tree = _build_tree(entries)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    html_lines = [
        "<!doctype html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Artifacts browser</title>",
        "<link rel='stylesheet' href='../../../reports/static/styles.css'>",
        "<style>",
        "ul.tree{list-style:none;margin:.5rem 0;padding-left:1rem}",
        "ul.tree ul.tree{padding-left:1.1rem}",
        "summary{font-weight:600;cursor:pointer}",
        "li.file,li.folder{margin:.25rem 0}",
        "</style>",
        "</head><body><main>",
        "<section class='card'>",
        "<h1 style='margin:.2rem 0'>Artifacts browser</h1>",
        "<p class='muted' style='margin:0'>Tap any file to open it directly in your browser.</p>",
        "<p style='margin:.75rem 0 0 0;display:flex;gap:8px;flex-wrap:wrap'>",
        "<a class='button-link' href='../report/index.html'>Open report</a>",
        "<a class='button-link secondary' href='../artifacts.zip'>Download .zip</a>",
        "</p>",
        "</section>",
        "<section class='card'>",
        _render_tree(tree),
        "</section>",
        "</main></body></html>",
    ]
    output = artifacts_dir / "index.html"
    output.write_text("\n".join(html_lines), encoding="utf-8")
    return output
