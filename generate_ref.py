from pathlib import Path
import importlib.util
import os
import re
import sys
from urllib.parse import quote


ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / "examples"
README = ROOT / "README.md"

BEGIN = "<!-- BEGIN GENERATED REFERENCES -->"
END = "<!-- END GENERATED REFERENCES -->"


def collect_references():
    references = []
    index = 0

    for folder, dirs, files in os.walk(EXAMPLES, followlinks=False):
        # Avoid recursive links and Python cache directories.
        dirs[:] = sorted(
            d for d in dirs
            if d != "__pycache__"
            and not (Path(folder) / d).is_symlink()
        )

        for filename in sorted(files):
            file = Path(folder) / filename

            if file.suffix != ".py" or file.name == "__init__.py":
                continue

            index += 1
            module_name = f"reference_meta_{index}"

            try:
                spec = importlib.util.spec_from_file_location(module_name, file)
                if spec is None or spec.loader is None:
                    continue

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                meta = getattr(module, "meta", None)

                if callable(meta):
                    citation = str(meta()).strip()
                    relative_path = file.relative_to(ROOT).as_posix()
                    references.append((relative_path, citation))

            except Exception as exc:
                print(f"Skipping {file}: {exc}")

    return references


def generate_markdown(references):
    lines = [BEGIN]

    if not references:
        lines.append("No references found.")
    else:
        for i, (filepath, citation) in enumerate(references, 1):
            link = quote(filepath, safe="/")
            filename = Path(filepath).name
            citation = " ".join(citation.split())

            lines.append(f"{i}. [{filename}]({link}) — {citation}")

    lines.append(END)
    return "\n".join(lines)


def update_readme(markdown):
    content = README.read_text(encoding="utf-8")

    # Replace an existing References section, if present.
    heading = re.search(
        r"(?m)^(#{1,6})[ \t]+References[ \t]*$",
        content,
        re.IGNORECASE
    )

    if heading:
        level = len(heading.group(1))
        remainder = content[heading.end():]

        # Find the next heading at the same or a higher level.
        next_heading = None

        for match in re.finditer(r"(?m)^(#{1,6})[ \t]+.*$", remainder):
            if len(match.group(1)) <= level:
                next_heading = match
                break

        end = (
            heading.end() + next_heading.start()
            if next_heading
            else len(content)
        )

        replacement = (
            heading.group(0)
            + "\n\n"
            + markdown
            + "\n\n"
        )

        content = content[:heading.start()] + replacement + content[end:].lstrip("\n")

    else:
        # Add a References section if one does not exist.
        content = content.rstrip() + "\n\n## References\n\n" + markdown + "\n"

    README.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    references = collect_references()
    markdown = generate_markdown(references)
    update_readme(markdown)

    print(f"Updated {README}")
    print(f"Collected {len(references)} references.")