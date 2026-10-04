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
    return (
        GENERATED_PROJECT / ".git"
    ).exists()


def create_checkpoint() -> tuple[bool, str]:
    """
    Capture the current HEAD commit.

    The returned commit hash represents the exact Git state
    that existed before the next coding operation.
    """

    if not repository_exists():
        return (
            False,
            "generated_project is not a Git repository.",
        )

    success, commit = _run_git(
        [
            "rev-parse",
            "HEAD",
        ]
    )

    if not success:
        return (
            False,
            commit or "Unable to determine current commit.",
        )

    return (
        True,
        commit,
    )


def restore_working_tree() -> tuple[bool, str]:
    """
    Restore tracked files to the current HEAD.

    IMPORTANT:
    This does NOT run `git clean`.
    Untracked files are intentionally preserved.
    """

    if not repository_exists():
        return (
            False,
            "generated_project is not a Git repository.",
        )

    success, output = _run_git(
        [
            "reset",
            "--hard",
            "HEAD",
        ]
    )

    if not success:
        return (
            False,
            output or "Git working-tree restore failed.",
        )

    return (
        True,
        (
            "Tracked project files restored to "
            "the current HEAD."
        ),
    )


def rollback_to_commit(
    commit_hash: str,
) -> tuple[bool, str]:
    """
    Roll the generated project back to a specific commit.

    Tracked files are restored to the requested commit.

    Untracked files are preserved.

    The target commit remains in Git history.
    """

    if not repository_exists():
        return (
            False,
            "generated_project is not a Git repository.",
        )

    commit_hash = (
        commit_hash.strip()
    )

    if not commit_hash:
        return (
            False,
            "A commit hash is required.",
        )

    # --------------------------------------------------------
    # VERIFY COMMIT EXISTS
    # --------------------------------------------------------

    success, output = _run_git(
        [
            "cat-file",
            "-e",
            f"{commit_hash}^{{commit}}",
        ]
    )

    if not success:
        return (
            False,
            (
                "Rollback target does not exist: "
                f"{commit_hash}"
            ),
        )

    # --------------------------------------------------------
    # ROLLBACK
    # --------------------------------------------------------

    success, output = _run_git(
        [
            "reset",
            "--hard",
            commit_hash,
        ]
    )

    if not success:
        return (
            False,
            output or "Git rollback failed.",
        )

    return (
        True,
        (
            "Project rolled back to commit "
            f"{commit_hash}. "
            "Untracked files were preserved."
        ),
    )


def get_current_commit() -> str:
    """
    Return the current HEAD commit.
    """

    if not repository_exists():
        return ""

    success, output = _run_git(
        [
            "rev-parse",
            "HEAD",
        ]
    )

    if not success:
        return ""

    return output


def get_recent_commits(
    limit: int = 5,
) -> list[str]:
    """
    Return recent commits in compact form.
    """

    if not repository_exists():
        return []

    if limit <= 0:
        return []

    success, output = _run_git(
        [
            "log",
            f"-{limit}",
            "--oneline",
        ]
    )

    if not success or not output:
        return []

    return [
        line.strip()
        for line in output.splitlines()
        if line.strip()
    ]


def checkpoint_exists(
    commit_hash: str,
) -> bool:
    """
    Check whether a checkpoint commit still exists.
    """

    if not repository_exists():
        return False

    commit_hash = (
        commit_hash.strip()
    )

    if not commit_hash:
        return False

    success, _ = _run_git(
        [
            "cat-file",
            "-e",
            f"{commit_hash}^{{commit}}",
        ]
    )

    return success


if __name__ == "__main__":

    print()
    print("=" * 64)
    print("CodePilot-Ai Git Recovery Test")
    print("=" * 64)

    if not repository_exists():

        print()
        print(
            "No generated_project Git repository found."
        )

    else:

        current = get_current_commit()

        print()
        print(
            f"Current commit: {current}"
        )

        print()
        print(
            "Recent commits:"
        )

        commits = get_recent_commits()

        for commit in commits:
            print(
                f"  - {commit}"
            )

        print()
        print(
            "Creating checkpoint..."
        )

        success, checkpoint = (
            create_checkpoint()
        )

        print(
            f"Checkpoint: "
            f"{'PASS' if success else 'FAIL'}"
        )

        print(
            checkpoint
        )

        if success:

            print()

            print(
                "Checkpoint exists: "
                f"{checkpoint_exists(checkpoint)}"
            )

    print()
    print("=" * 64)