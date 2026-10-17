"""Does a change need CI's GPU job? Prints ``code=true`` or ``code=false``.

The deliver job pulls data, may rebuild the index, scores answers and builds
the images: a change to the documentation alone needs none of it. A change
counts as docs only when every file it touches matches ``DOCS``:
the top-level Markdown files, and ``docs/`` except ``docs/metrics/`` (the
drift baseline there is baked into the API image, and the metrics are stage
outputs). The prompts are Markdown too, but they live in ``src/`` and count as
code.

Run as ``python3 scripts/ci_scope.py BASE HEAD >> "$GITHUB_OUTPUT"``; with no
usable base (a manual run, a new branch), it answers ``code=true``. Standard
library only, so GitHub's runner needs no install.
"""

import re
import subprocess
import sys

DOCS = re.compile(
    r"^(?:[^/]+\.md"  # README.md, CONTRIBUTING.md, CHANGELOG.md
    r"|docs/(?!metrics/).+)$"  # guides, reports, images
)
NO_BASE = {"", "0" * 40}


def needs_deliver(paths: list[str]) -> bool:
    """Whether any of ``paths`` is more than documentation."""
    return not paths or any(not DOCS.match(p) for p in paths)


def changed(base: str, head: str) -> list[str] | None:
    """The files changed between ``base`` and ``head``; None if git can't tell."""
    if base in NO_BASE:
        return None
    try:
        out = subprocess.run(
            ["git", "diff", "--name-only", base, head],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return None
    return [line for line in out.stdout.splitlines() if line]


def main() -> None:
    base, head = (sys.argv[1:3] + ["", ""])[:2]
    paths = changed(base, head)
    code = paths is None or needs_deliver(paths)
    sys.stdout.write(f"code={'true' if code else 'false'}\n")
    sys.stderr.write(
        f"{'unknown' if paths is None else len(paths)} files changed; "
        f"{'the GPU job runs' if code else 'docs only, the GPU job is skipped'}\n"
    )


if __name__ == "__main__":
    main()
