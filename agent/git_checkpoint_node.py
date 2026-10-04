from __future__ import annotations

from agent.git_recovery import create_checkpoint
from agent.states import CoderState


def git_checkpoint_node(
    state: CoderState,
) -> CoderState:
    """
    Capture the exact Git HEAD before the Coder modifies
    the existing project.
    """

    print()
    print("=" * 64)
    print("[GIT CHECKPOINT] Saving pre-coding state")
    print("=" * 64)

    try:
        success, checkpoint = create_checkpoint()

        if success:
            print()
            print(
                "[GIT CHECKPOINT] PASS"
            )

            print(
                f"[GIT CHECKPOINT] {checkpoint}"
            )

            return CoderState(
                **{
                    **state.model_dump(),
                    "git_checkpoint": checkpoint,
                }
            )

        print()
        print(
            "[GIT CHECKPOINT] No checkpoint created."
        )

        print(
            f"[GIT CHECKPOINT] {checkpoint}"
        )

        return CoderState(
            **{
                **state.model_dump(),
                "git_checkpoint": None,
            }
        )

    except Exception as exc:

        print()
        print(
            "[GIT CHECKPOINT] Warning: "
            f"{type(exc).__name__}: {exc}"
        )

        return CoderState(
            **{
                **state.model_dump(),
                "git_checkpoint": None,
            }
        )


if __name__ == "__main__":

    print()
    print("=" * 64)
    print("CodePilot-Ai Git Checkpoint Test")
    print("=" * 64)

    state = CoderState(
        user_prompt="Git checkpoint test"
    )

    result = git_checkpoint_node(
        state
    )

    print()
    print(
        f"Checkpoint: "
        f"{result.git_checkpoint}"
    )

    print()
    print("=" * 64)