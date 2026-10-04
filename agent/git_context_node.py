from __future__ import annotations

from agent.git_context import get_coder_git_context
from agent.states import CoderState


def git_context_node(
    state: CoderState,
) -> CoderState:
    """
    Collect Git information before the Coder runs.

    This gives the Coder awareness of:
    - existing uncommitted changes
    - changed files
    - latest commit
    - current Git diff
    """

    print()
    print("=" * 64)
    print("[GIT CONTEXT] Inspecting existing project changes")
    print("=" * 64)

    try:
        context = get_coder_git_context()

        git_context = context.get(
            "git_context",
            "",
        )

        changed_files = context.get(
            "git_changed_files",
            [],
        )

        latest_commit = context.get(
            "git_latest_commit",
            "",
        )

        print()

        if latest_commit:
            print(
                f"[GIT CONTEXT] Latest commit: "
                f"{latest_commit}"
            )
        else:
            print(
                "[GIT CONTEXT] No Git commit found."
            )

        print()

        if changed_files:
            print(
                "[GIT CONTEXT] Changed files:"
            )

            for filepath in changed_files:
                print(
                    f"  - {filepath}"
                )

        else:
            print(
                "[GIT CONTEXT] No uncommitted changes."
            )

        print()

        return CoderState(
            **{
                **state.model_dump(),
                "git_context": git_context,
                "git_changed_files": changed_files,
                "git_latest_commit": latest_commit,
            }
        )

    except Exception as exc:

        print()
        print(
            "[GIT CONTEXT] Warning: "
            f"{type(exc).__name__}: {exc}"
        )

        return CoderState(
            **{
                **state.model_dump(),
                "git_context": (
                    "Git context could not be loaded. "
                    "Continue without Git diff information."
                ),
                "git_changed_files": [],
                "git_latest_commit": None,
            }
        )


if __name__ == "__main__":

    print()
    print("=" * 64)
    print("CodePilot-Ai Git Context Node Test")
    print("=" * 64)

    test_state = CoderState(
        user_prompt="Build a Weather Dashboard"
    )

    result = git_context_node(
        test_state
    )

    print()
    print(
        "Git context:"
    )

    print(
        result.git_context
    )

    print()
    print(
        "Changed files:"
    )

    print(
        result.git_changed_files
    )

    print()
    print(
        "Latest commit:"
    )

    print(
        result.git_latest_commit
    )