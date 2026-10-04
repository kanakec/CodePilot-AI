from __future__ import annotations

import json

from agent.memory import load_memory


def show_history() -> None:
    """
    Display persistent CodePilot-Ai project history.
    """

    data = load_memory()

    projects = data.get(
        "projects",
        {},
    )

    print()
    print("=" * 70)
    print("CodePilot-Ai Project History")
    print("=" * 70)

    if not projects:
        print()
        print("No project history found.")
        return

    for index, (project_key, memory) in enumerate(
        projects.items(),
        start=1,
    ):

        print()
        print(
            f"[PROJECT {index}]"
        )
        print("-" * 70)

        project_type = memory.get(
            "project_type",
            "unknown",
        )

        files = memory.get(
            "files",
            [],
        )

        executor_passed = memory.get(
            "executor_passed",
            False,
        )

        executor_attempts = memory.get(
            "executor_attempts",
            0,
        )

        review_attempts = memory.get(
            "review_attempts",
            0,
        )

        git_commit = memory.get(
            "git_commit",
            "",
        )

        last_validation = memory.get(
            "last_validation_result",
            "",
        )

        last_updated = memory.get(
            "last_updated",
            "",
        )

        print(
            f"Type: {project_type}"
        )

        print(
            f"Last updated: {last_updated}"
        )

        print(
            f"Executor passed: "
            f"{'YES' if executor_passed else 'NO'}"
        )

        print(
            f"Executor attempts: "
            f"{executor_attempts}"
        )

        print(
            f"Review attempts: "
            f"{review_attempts}"
        )

        if files:

            print()
            print("Generated files:")

            for filepath in files:
                print(
                    f"  - {filepath}"
                )

        if git_commit:

            print()
            print(
                f"Git commit: {git_commit}"
            )

        if last_validation:

            print()
            print(
                f"Validation: "
                f"{last_validation}"
            )

        print()
        print(
            "Project request:"
        )

        print(
            project_key
        )

    print()
    print("=" * 70)


def show_raw_memory() -> None:
    """
    Display the complete persistent memory as JSON.
    """

    data = load_memory()

    print()
    print(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":

    show_history()