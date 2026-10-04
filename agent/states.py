from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class File(BaseModel):
    path: str = Field(
        description="Path to the file to be created or modified."
    )

    purpose: str = Field(
        description="Purpose of the file."
    )


class Plan(BaseModel):
    name: str = Field(
        description="Name of the application."
    )

    description: str = Field(
        description="One-line description of the application."
    )

    techstack: str = Field(
        description="Technology stack used by the application."
    )

    features: list[str] = Field(
        description="Features the application should contain."
    )

    files: list[File] = Field(
        description="Files required by the application."
    )


class ImplementationTask(BaseModel):
    filepath: str = Field(
        description="Path of the file to implement or modify."
    )

    task_description: str = Field(
        description="Detailed implementation instructions."
    )


class TaskPlan(BaseModel):
    implementation_steps: list[ImplementationTask] = Field(
        description="Ordered implementation tasks."
    )

    model_config = ConfigDict(
        extra="allow"
    )


class ReviewResult(BaseModel):
    approved: bool = Field(
        description="Whether the generated project passes the review."
    )

    issues: list[str] = Field(
        default_factory=list,
        description="Problems discovered during review."
    )

    suggestions: list[str] = Field(
        default_factory=list,
        description="Suggested improvements."
    )

    summary: str = Field(
        description="Short summary of the review."
    )


class CoderState(BaseModel):
    user_prompt: str = Field(
        description="Original project request from the user."
    )

    plan: Optional[Plan] = Field(
        default=None,
        description="High-level application plan."
    )

    task_plan: Optional[TaskPlan] = Field(
        default=None,
        description="Detailed implementation plan."
    )

    current_step_idx: int = Field(
        default=0,
        description="Current implementation task index."
    )

    current_file_content: Optional[str] = Field(
        default=None,
        description="Content of the file currently being edited."
    )

    last_validation_result: Optional[str] = Field(
        default=None,
        description="Result of the latest validation."
    )

    validation_attempts: int = Field(
        default=0,
        description="Number of validation attempts."
    )

    files_modified: list[str] = Field(
        default_factory=list,
        description="Files modified by the coder."
    )

    review_attempts: int = Field(
        default=0,
        description="Number of reviewer passes."
    )

    review_feedback: Optional[str] = Field(
        default=None,
        description="Feedback from the reviewer that the coder must address."
    )

    review_fix_required: bool = Field(
        default=False,
        description="Whether the reviewer has requested fixes."
    )

    executor_passed: bool = Field(
        default=False,
        description="Whether automated project execution passed."
    )

    executor_attempts: int = Field(
        default=0,
        description="Number of automated execution attempts."
    )

    # ========================================================
    # GIT-AWARE CODING
    # ========================================================

    git_context: Optional[str] = Field(
        default=None,
        description=(
            "Git change context containing changed files and "
            "diff information available to the coder."
        ),
    )

    git_changed_files: list[str] = Field(
        default_factory=list,
        description=(
            "Files currently changed in the generated project."
        ),
    )

    git_latest_commit: Optional[str] = Field(
        default=None,
        description=(
            "Latest Git commit associated with the generated project."
        ),
    )

    git_checkpoint: Optional[str] = Field(
        default=None,
        description=(
            "Git commit recorded immediately before the "
            "current coding attempt. Used for safe rollback."
        ),
    )