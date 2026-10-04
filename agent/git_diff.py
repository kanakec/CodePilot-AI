from __future__ import annotations

import subprocess
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

GENERATED_PROJECT = (
    PROJECT_ROOT / "generated_project"
)


def _run_git(
    args: list[str],
) -> tuple[bool, str]:
    """
    Run Git inside generated_project.
    """

    try:
        result = subprocess.run(
            ["git", *args],
            cwd=GENERATED_PROJECT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    except FileNotFoundError:
        return (
            False,
            "Git executable was not found.",
        )

    output = (
        result.stdout.strip()
        or result.stderr.strip()
    )

    return (
        result.returncode == 0,
        output,
    )


def repository_exists() -> bool:
    """
    Check whether generated_project has its own
    Git repository.
    """

    return (
        GENERATED_PROJECT / ".git"
    ).exists()


def get_diff(
    staged: bool = False,
) -> str:
    """
    Return the current Git diff.
    """

    if not repository_exists():
        return ""

    command = [
        "diff"
    ]

    if staged:
        command.append(
            "--cached"
        )

    success, output = _run_git(
        command
    )

    if not success:
        return ""

    return output


def get_diff_stat() -> str:
    """
    Return a compact summary of Git changes.
    """

    if not repository_exists():
        return ""

    success, output = _run_git(
        [
            "diff",
            "--stat",
        ]
    )

    if not success:
        return ""

    return output


def get_changed_files() -> list[str]:
    """
    Return files changed in the working tree.

    Uses Git's porcelain output and extracts the filename
    without manually assuming status-column spacing.
    """

    if not repository_exists():
        return []

    success, output = _run_git(
        [
            "status",
            "--porcelain=v1",
            "-z",
        ]
    )

    if not success or not output:
        return []

    files = []

    # -z separates entries with NUL characters.
    entries = output.split("\0")

    for entry in entries:

        if not entry:
            continue

        # Git porcelain format:
        #
        # XY filename
        #
        # For normal files, remove exactly the two
        # status characters and surrounding whitespace.
        if len(entry) >= 3:
            filepath = entry[2:].strip()
        else:
            filepath = entry.strip()

        # Handle rename/copy notation:
        #
        # R  old.py -> new.py
        #
        if " -> " in filepath:
            filepath = filepath.rsplit(
                " -> ",
                1
            )[1].strip()

        if filepath:
            files.append(
                filepath
            )

    return files


def has_changes() -> bool:
    """
    Check whether generated_project has
    uncommitted changes.
    """

    return bool(
        get_changed_files()
    )


def get_last_commit_diff() -> str:
    """
    Return the complete diff introduced by
    the latest commit.
    """

    if not repository_exists():
        return ""

    success, output = _run_git(
        [
            "show",
            "--format=",
            "--stat",
            "HEAD",
        ]
    )

    if not success:
        return ""

    return output


def get_file_diff(
    filepath: str,
) -> str:
    """
    Return the current diff for one specific file.
    """

    if not repository_exists():
        return ""

    success, output = _run_git(
        [
            "diff",
            "--",
            filepath,
        ]
    )

    if not success:
        return ""

    return output


def build_diff_context(
    max_chars: int = 12000,
) -> str:
    """
    Build a compact context block that can be supplied
    to the coding agent.
    """

    changed_files = (
        get_changed_files()
    )

    if not changed_files:
        return (
            "No uncommitted Git changes "
            "were detected."
        )

    diff = get_diff()

    if not diff:
        return (
            "Changed files detected, but "
            "no textual Git diff is available."
        )

    if len(diff) > max_chars:
        diff = (
            diff[:max_chars]
            + "\n\n[DIFF TRUNCATED]"
        )

    return (
        "CURRENT GIT CHANGES\n"
        "===================\n\n"
        "Changed files:\n"
        + "\n".join(
            f"- {filepath}"
            for filepath in changed_files
        )
        + "\n\n"
        "Diff:\n"
        "-----\n"
        f"{diff}"
    )


def print_diff_summary() -> None:
    """
    Display Git change information.
    """

    print()
    print("=" * 64)
    print("CodePilot-Ai Git Diff")
    print("=" * 64)

    if not repository_exists():
        print()
        print(
            "No generated_project Git repository found."
        )
        return

    changed_files = (
        get_changed_files()
    )

    print()

    if changed_files:
        print(
            "Changed files:"
        )

        for filepath in changed_files:
            print(
                f"  - {filepath}"
            )

    else:
        print(
            "No uncommitted changes."
        )

    print()

    stat = get_diff_stat()

    if stat:
        print(
            "Diff summary:"
        )
        print(stat)

    print()

    diff = get_diff()

    if diff:
        print(
            "Diff:"
        )
        print(diff)

    print()
    print("=" * 64)


if __name__ == "__main__":
    print_diff_summary()