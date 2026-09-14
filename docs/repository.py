"""Generate or check the committed Markdown mirror using a clean build."""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DOCS = Path(__file__).resolve().parent
REFERENCE = DOCS / "reference"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="quantkit-docs-") as temporary:
        output = Path(temporary) / "markdown"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sphinx",
                "-b",
                "markdown",
                "-W",
                "--keep-going",
                "-E",
                "-a",
                "-d",
                str(Path(temporary) / "doctrees"),
                str(DOCS / "source"),
                str(output),
            ],
            check=True,
        )
        generated = {
            p.relative_to(output): p.read_bytes().rstrip() + b"\n"
            for p in output.rglob("*.md")
        }
        existing = {
            p.relative_to(REFERENCE): p.read_bytes()
            for p in REFERENCE.rglob("*")
            if p.is_file()
        }
        changed = sorted(
            p
            for p in generated.keys() | existing.keys()
            if generated.get(p) != existing.get(p)
        )
        if args.check:
            if changed:
                print("Documentation is stale; run: uv run make -C docs repo")
                for path in changed:
                    print(f"  {path}")
                return 1
            print("Repository documentation is up to date.")
        else:
            # This directory contains generated files only.
            if REFERENCE.exists():
                shutil.rmtree(REFERENCE)
            for path, content in generated.items():
                target = REFERENCE / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            print(f"Generated {len(generated)} Markdown files in {REFERENCE}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
