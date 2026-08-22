from pathlib import Path


def test_public_files_contain_no_private_context_or_local_paths():
    root = Path(__file__).resolve().parents[1]
    scan_roots = [root / "README.md", root / "docs", root / "configs", root / "src"]
    forbidden = (
        "UN" + "SW",
        "COMP" + "9517",
        "C:" + "\\Users\\",
        "/Users/",
        "group project specification",
    )
    violations = []
    for scan_root in scan_roots:
        files = [scan_root] if scan_root.is_file() else list(scan_root.rglob("*"))
        for path in files:
            if not path.is_file() or path.suffix not in {"", ".md", ".py", ".yaml", ".toml"}:
                continue
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token.lower() in text.lower():
                    violations.append(f"{path.relative_to(root)} contains {token!r}")
    assert not violations, "\n".join(violations)

