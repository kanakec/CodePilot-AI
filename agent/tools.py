import pathlib
import subprocess
from typing import Optional

from langchain_core.tools import tool


PROJECT_ROOT = pathlib.Path.cwd() / "generated_project"


def safe_path_for_project(path: str) -> pathlib.Path:
    """Ensure paths stay inside generated_project."""

    if not path:
        raise ValueError("Path cannot be empty.")

    project_root = PROJECT_ROOT.resolve()
    target = (project_root / path).resolve()

    if target != project_root and project_root not in target.parents:
        raise ValueError(
            f"Attempt to access path outside project root: {path}"
        )

    return target


@tool
def write_file(path: str, content: str) -> str:
    """Create or overwrite a file inside generated_project."""

    target = safe_path_for_project(path)

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target.write_text(
        content,
        encoding="utf-8",
    )

    return f"SUCCESS: Wrote file {target.relative_to(PROJECT_ROOT)}"


@tool
def read_file(path: str) -> str:
    """Read a file from generated_project."""

    target = safe_path_for_project(path)

    if not target.exists():
        return f"FILE_NOT_FOUND: {path}"

    if not target.is_file():
        return f"NOT_A_FILE: {path}"

    return target.read_text(
        encoding="utf-8"
    )


@tool
def get_current_directory() -> str:
    """Return the generated project directory."""

    PROJECT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    return str(PROJECT_ROOT.resolve())


@tool
def list_files(directory: str = ".") -> str:
    """List files inside generated_project."""

    target = safe_path_for_project(directory)

    if not target.exists():
        return f"DIRECTORY_NOT_FOUND: {directory}"

    if not target.is_dir():
        return f"NOT_A_DIRECTORY: {directory}"

    files = []

    for file in target.rglob("*"):
        if file.is_file():
            files.append(
                str(
                    file.relative_to(PROJECT_ROOT)
                )
            )

    if not files:
        return "No files found."

    return "\n".join(sorted(files))


@tool
def run_project_command(
    command: str,
    cwd: Optional[str] = None,
    timeout: int = 30,
) -> str:
    """
    Run a validation command inside generated_project.

    Returns stdout, stderr and exit code so the coder
    agent can inspect failures and fix the project.
    """

    if cwd:
        working_directory = safe_path_for_project(cwd)
    else:
        working_directory = PROJECT_ROOT.resolve()

    working_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(working_directory),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = result.stdout.strip()
        error = result.stderr.strip()

        if result.returncode == 0:
            return (
                "SUCCESS\n"
                f"Exit code: {result.returncode}\n"
                f"STDOUT:\n{output}\n"
            )

        return (
            "FAIL\n"
            f"Exit code: {result.returncode}\n"
            f"STDOUT:\n{output}\n"
            f"STDERR:\n{error}\n"
        )

    except subprocess.TimeoutExpired:
        return (
            f"ERROR: Command timed out after "
            f"{timeout} seconds."
        )

    except Exception as exc:
        return (
            f"ERROR: Could not execute command: {exc}"
        )


def init_project_root() -> str:
    """Create generated_project if it doesn't exist."""

    PROJECT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    return str(PROJECT_ROOT.resolve())