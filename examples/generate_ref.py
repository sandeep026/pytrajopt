from pathlib import Path
import importlib.util

root = Path(__file__).parent
search_dir = root  # generate_ref.py is inside examples/pytrajopt/examples

references = []

for i, file in enumerate(sorted(search_dir.glob("*.py"))):
    if file.name == Path(__file__).name:
        continue

    spec = importlib.util.spec_from_file_location(f"meta_module_{i}", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    meta = getattr(module, "meta", None)
    if callable(meta):
        filename = file.name
        references.append((filename, meta()))

lines = ["# References", ""]

for i, (filename, citation) in enumerate(references, 1):
    lines.append(f"{i}. [{filename}]({filename}) — {citation}")

markdown = "\n".join(lines)

(root / "references.md").write_text(markdown, encoding="utf-8")
print(markdown)