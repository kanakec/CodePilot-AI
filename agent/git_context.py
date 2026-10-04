from __future__ import annotations

from agent.git_diff import (
    build_diff_context,
    get_changed_files,
)
from agent.git_manager import (
    get_latest_commit,
)


def get_coder_git_context() -> dict:
    """
    Collect Git information that can be supplied to the Coder.
    """

    changed_files = get_changed_files()

    context = build_diff_context()

    latest_commit = get_latest_commit()

    return {
        "git_context": context,
        "git_changed_files": changed_files,
        "git_latest_commit": latest_commit,
    }


def format_coder_git_context(
    git_context: str | None,
) -> str:
    """
    Convert Git context into a safe prompt section.
    """

    if not git_context:
        return (
            "GIT CONTEXT\n"
            "===========\n"
            "No Git change information is available."
        )

    return (
        "GIT CONTEXT\n"
        "===========\n\n"
        "Use the following Git information when deciding "
        "whether an existing file should be modified rather "
        "than regenerated.\n\n"
        f"{git_context}"
    )


if __name__ == "__main__":

    print()
    print("=" * 64)
    print("CodePilot-Ai Git Context Test")
    print("=" * 64)

    context = get_coder_git_context()

    print()
    print(
        "Latest commit:"
    )

    print(
        context["git_latest_commit"]
        or "No commit found."
    )

    print()

    print(
        "Changed files:"
    )

    if context["git_changed_files"]:

        for filepath in context[
            "git_changed_files"
        ]:
            print(
                f"  - {filepath}"
            )

    else:

        print(
            "  No uncommitted changes."
        )

    print()

    print(
        format_coder_git_context(
            context["git_context"]
        )
    )

    print()
    print(
        "=" * 64
    )