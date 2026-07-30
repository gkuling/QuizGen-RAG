'''
Self-enforcing ASCII gate: every .py file in this repo (outside .venv and
.idea) must decode as pure ASCII. This is what makes the project's "pure
ASCII" rule a checked invariant instead of a style suggestion.
'''

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
EXCLUDED_DIR_NAMES = {".venv", ".idea", ".git"}


def _iter_python_files() -> list[Path]:
    '''Every *.py file under the repo root, excluding the venv/IDE dirs.'''
    files = []
    for path in REPO_ROOT.rglob("*.py"):
        if EXCLUDED_DIR_NAMES.isdisjoint(path.parts):
            files.append(path)
    return files


def test_repo_has_python_files_to_check() -> None:
    # Guards against the walk silently finding nothing (e.g. wrong root),
    # which would make the ASCII test below vacuously pass.
    assert len(_iter_python_files()) > 0


def test_all_python_files_are_pure_ascii() -> None:
    violations: list[str] = []

    for path in _iter_python_files():
        raw_bytes = path.read_bytes()
        try:
            text = raw_bytes.decode("ascii")
        except UnicodeDecodeError:
            text = raw_bytes.decode("utf-8", errors="replace")
            for line_number, line in enumerate(text.splitlines(), start=1):
                for column, char in enumerate(line):
                    if ord(char) > 127:
                        violations.append(
                            f"{path.relative_to(REPO_ROOT)}:{line_number}:{column}: "
                            f"non-ASCII character {char!r} (U+{ord(char):04X})"
                        )
            continue

    assert not violations, "Non-ASCII characters found:\n" + "\n".join(violations)
