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
    Run a Git command strictly inside generated_project.
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
            "Git executable was not found. "
            "Install Git and make sure it is available in PATH.",
        )

    output = (
        result.stdout.strip()
        or result.stderr.strip()
    )

    return (
        result.returncode == 0,
        output,
    )


def is_git_repository() -> bool:
    """
    Check whether generated_project itself contains
    a Git repository.

    Important:
    Git normally searches parent directories. Therefore
    git rev-parse alone cannot be used here.
    """

    git_directory = (
        GENERATED_PROJECT / ".git"
    )

    return git_directory.exists()


def initialize_repository() -> tuple[bool, str]:
    """
    Initialize a Git repository specifically inside
    generated_project.
    """

    GENERATED_PROJECT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if is_git_repository():
        return (
            True,
            "Git repository already exists inside generated_project.",
        )

    success, output = _run_git(
        ["init"]
    )

    if not success:
        return (
            False,
            output or "Git initialization failed.",
        )

    # Verify that Git actually created the repository
    # in generated_project rather than relying on a parent
    # repository.
    if not is_git_repository():
        return (
            False,
            "Git initialization completed, but "
            "generated_project/.git was not created.",
        )

    return (
        True,
        "Git repository initialized inside generated_project.",
    )


def get_status() -> tuple[bool, str]:
    """
    Get Git status for generated_project.
    """

    if not is_git_repository():
        return (
            False,
            "generated_project is not a Git repository.",
        )

    return _run_git(
        [
            "status",
            "--short",
        ]
    )


def get_changed_files() -> list[str]:
    """
    Return files changed or untracked inside
    generated_project.
    """

    success, output = get_status()

    if not success or not output:
        return []

    files = []

    for line in output.splitlines():

        line = line.strip()

        if not line:
            continue

        # Git status --short format:
        #
        # ?? app.js
        # M  index.html
        #
        # Remove the two-character status prefix.

        if len(line) > 3:
            filepath = line[3:].strip()
        else:
            filepath = line

        if filepath:
            files.append(filepath)

    return files


def has_changes() -> bool:
    """
    Check whether generated_project has Git changes.
    """

    return bool(
        get_changed_files()
    )


def create_commit(
    message: str,
) -> tuple[bool, str]:
    """
    Stage and commit generated-project changes.

    This function NEVER intentionally commits the
    parent CodePilot-Ai repository.
    """

    initialized, init_message = (
        initialize_repository()
    )

    if not initialized:
        return (
            False,
            init_message,
        )

    if not has_changes():
        return (
            True,
            "No Git changes to commit.",
        )

    success, output = _run_git(
        ["add", "."]
    )

    if not success:
        return (
            False,
            output or "Git add failed.",
        )

    success, output = _run_git(
        [
            "commit",
            "-m",
            message,
        ]
    )

    if not success:
        return (
            False,
            output or "Git commit failed.",
        )

    return (
        True,
        output or "Git commit created.",
    )


def get_latest_commit() -> str:
    """
    Return the latest commit inside generated_project.
    """

    if not is_git_repository():
        return ""

    success, output = _run_git(
        [
            "log",
            "-1",
            "--oneline",
        ]
    )

    if not success:
        return ""

    return output


def git_snapshot() -> dict:
    """
    Return the current Git state of generated_project.
    """

    repository = is_git_repository()

    changed_files = (
        get_changed_files()
        if repository
        else []
    )

    latest_commit = (
        get_latest_commit()
        if repository
        else ""
    )

    return {
        "repository_initialized": repository,
        "repository_path": str(
            GENERATED_PROJECT / ".git"
        ),
        "changed_files": changed_files,
        "latest_commit": latest_commit,
    }


if __name__ == "__main__":

    print()
    print("=" * 64)
    print("CodePilot-Ai Git Manager Test")
    print("=" * 64)

    print()

    print(
        f"[GIT] Target repository:"
    )

    print(
        f"[GIT] {GENERATED_PROJECT}"
    )

    print()

    success, message = (
        initialize_repository()
    )

    print(
        "[GIT] Initialize: "
        f"{'PASS' if success else 'FAIL'}"
    )

    print(
        f"[GIT] {message}"
    )

    print()

    print(
        "[GIT] Repository exists:"
        f" {is_git_repository()}"
    )

    print()

    changed_files = (
        get_changed_files()
    )

    print(
        "[GIT] Changed files:"
    )

    if changed_files:

        for filepath in changed_files:
            print(
                f"  - {filepath}"
            )

    else:
        print(
            "  No changes."
        )

    print()

    latest = get_latest_commit()

    if latest:
        print(
            f"[GIT] Latest commit: {latest}"
        )
    else:
        print(
            "[GIT] No commits yet."
        )

    print()

    print(
        "=" * 64
    )