from agent.executor import execute_project
from agent.git_recovery import (
    checkpoint_exists,
    rollback_to_commit,
)
from agent.states import CoderState


def executor_node(
    state: CoderState,
) -> CoderState:
    """
    Execute and test the generated project.

    Flow:

        Coder
          ↓
        Executor
          ↓
        PASS -> Finish
          ↓
        FAIL
          ↓
        Reviewer
          ↓
        Coder Fix
          ↓
        Executor

    If the maximum number of execution attempts is reached,
    the project is restored to the exact Git checkpoint that
    existed before the coding attempt.
    """

    print()
    print("=" * 64)
    print("[EXECUTOR] Automated project testing")
    print("=" * 64)

    try:

        result = execute_project()

    except Exception as exc:

        error_message = (
            "Executor crashed while testing the project: "
            f"{type(exc).__name__}: {exc}"
        )

        print()
        print("[EXECUTOR] FAIL")
        print(
            f"[EXECUTOR] ERROR: {error_message}"
        )

        feedback = (
            "AUTOMATED EXECUTOR ERROR:\n"
            f"- {error_message}\n\n"
            "The coder must inspect the generated project "
            "and fix the underlying problem."
        )

        return CoderState(
            **{
                **state.model_dump(),
                "last_validation_result": (
                    "Executor crashed while "
                    "testing the project."
                ),
                "review_feedback": feedback,
                "review_fix_required": True,
                "executor_passed": False,
                "executor_attempts": (
                    state.executor_attempts + 1
                ),
            }
        )

    passed = bool(
        result.get(
            "passed",
            False,
        )
    )

    summary = result.get(
        "summary",
        "Automated execution completed.",
    )

    errors = result.get(
        "errors",
        [],
    )

    warnings = result.get(
        "warnings",
        [],
    )

    if not isinstance(
        errors,
        list,
    ):
        errors = [
            str(errors)
        ]

    if not isinstance(
        warnings,
        list,
    ):
        warnings = [
            str(warnings)
        ]

    feedback_parts = []

    if errors:

        feedback_parts.append(
            "AUTOMATED EXECUTOR ERRORS:"
        )

        for error in errors:

            feedback_parts.append(
                f"- {error}"
            )

    if warnings:

        feedback_parts.append(
            "\nAUTOMATED EXECUTOR WARNINGS:"
        )

        for warning in warnings:

            feedback_parts.append(
                f"- {warning}"
            )

    feedback_parts.append(
        f"\nEXECUTOR SUMMARY:\n{summary}"
    )

    feedback = "\n".join(
        feedback_parts
    )

    # ========================================================
    # PASS
    # ========================================================

    if passed:

        print()
        print("[EXECUTOR] PASS")
        print(
            f"[EXECUTOR] {summary}"
        )

        if warnings:

            print()

            for warning in warnings:

                print(
                    f"[EXECUTOR] WARNING: {warning}"
                )

        print()
        print(
            "[EXECUTOR] Project passed automated testing."
        )

        return CoderState(
            **{
                **state.model_dump(),
                "last_validation_result": summary,
                "review_feedback": "",
                "review_fix_required": False,
                "executor_passed": True,
                "executor_attempts": (
                    state.executor_attempts + 1
                ),
            }
        )

    # ========================================================
    # FAIL
    # ========================================================

    print()
    print("[EXECUTOR] FAIL")

    if errors:

        for error in errors:

            print(
                f"[EXECUTOR] ERROR: {error}"
            )

    if warnings:

        for warning in warnings:

            print(
                f"[EXECUTOR] WARNING: {warning}"
            )

    print()
    print(
        "[EXECUTOR] Project failed automated testing."
    )

    new_attempts = (
        state.executor_attempts + 1
    )

    # ========================================================
    # MAXIMUM ATTEMPTS
    # ========================================================

    if new_attempts >= 3:

        print()
        print(
            "[EXECUTOR] Maximum execution attempts reached."
        )

        print(
            "[EXECUTOR] Attempting Git recovery..."
        )

        checkpoint = (
            state.git_checkpoint
        )

        if checkpoint:

            print()
            print(
                "[EXECUTOR] Recovery checkpoint:"
            )

            print(
                f"[EXECUTOR] {checkpoint}"
            )

            if checkpoint_exists(
                checkpoint
            ):

                success, rollback_message = (
                    rollback_to_commit(
                        checkpoint
                    )
                )

                if success:

                    print()
                    print(
                        "[EXECUTOR] Git recovery completed."
                    )

                    print(
                        f"[EXECUTOR] "
                        f"{rollback_message}"
                    )

                else:

                    print()
                    print(
                        "[EXECUTOR] Git recovery failed."
                    )

                    print(
                        f"[EXECUTOR] "
                        f"{rollback_message}"
                    )

            else:

                print()
                print(
                    "[EXECUTOR] Recovery checkpoint "
                    "no longer exists."
                )

        else:

            print()
            print(
                "[EXECUTOR] No pre-coding Git checkpoint "
                "available."
            )

        return CoderState(
            **{
                **state.model_dump(),
                "last_validation_result": (
                    f"{summary}\n"
                    "Maximum execution attempts reached."
                ),
                "review_feedback": feedback,
                "review_fix_required": False,
                "executor_passed": False,
                "executor_attempts": new_attempts,
            }
        )

    # ========================================================
    # SEND FAILURE TO REVIEWER
    # ========================================================

    print()
    print(
        "[EXECUTOR] Sending detected errors to reviewer."
    )

    return CoderState(
        **{
            **state.model_dump(),
            "last_validation_result": summary,
            "review_feedback": feedback,
            "review_fix_required": True,
            "executor_passed": False,
            "executor_attempts": new_attempts,
        }
    )


if __name__ == "__main__":

    print()
    print(
        "CodePilot-Ai Executor Node Test"
    )

    print(
        "=" * 64
    )

    test_state = CoderState(
        user_prompt="executor test"
    )

    final_state = executor_node(
        test_state
    )

    print()
    print(
        "=" * 64
    )

    print(
        "Executor state"
    )

    print(
        "=" * 64
    )

    print(
        final_state.model_dump()
    )