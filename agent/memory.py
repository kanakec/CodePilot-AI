from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

MEMORY_DIR = (
    PROJECT_ROOT / ".codepilot"
)

MEMORY_FILE = (
    MEMORY_DIR / "project_memory.json"
)


# ============================================================
# PROJECT IDENTITY
# ============================================================

def detect_project_identity(
    user_prompt: str,
) -> str:
    """
    Convert differently worded requests into a stable
    project identity.

    Examples:

        "Build a Weather Dashboard"
        "Create a modern weather app"
        "Build a weather dashboard using HTML CSS JS"

    all become:

        weather
    """

    text = (
        user_prompt
        .strip()
        .lower()
    )

    # Normalize punctuation.
    text = re.sub(
        r"[^a-z0-9\s-]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    # --------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------

    weather_patterns = [
        r"\bweather\b",
        r"\bweather\s+dashboard\b",
        r"\bweather\s+app\b",
        r"\bweather\s+application\b",
        r"\bweather\s+website\b",
        r"\bweather\s+forecast\b",
        r"\bweather\s+tracker\b",
    ]

    if any(
        re.search(
            pattern,
            text,
        )
        for pattern in weather_patterns
    ):
        return "weather"

    # --------------------------------------------------------
    # TODO
    # --------------------------------------------------------

    todo_patterns = [
        r"\btodo\b",
        r"\bto-do\b",
        r"\btask\s+manager\b",
        r"\btask\s+management\b",
        r"\btask\s+list\b",
        r"\bproductivity\s+app\b",
    ]

    if any(
        re.search(
            pattern,
            text,
        )
        for pattern in todo_patterns
    ):
        return "todo"

    # --------------------------------------------------------
    # EXPENSE
    # --------------------------------------------------------

    expense_patterns = [
        r"\bexpense\b",
        r"\bexpenses\b",
        r"\bexpense\s+tracker\b",
        r"\bexpense\s+tracking\b",
        r"\bexpense\s+manager\b",
        r"\bexpense\s+management\b",
        r"\bpersonal\s+finance\b",
        r"\bspending\s+tracker\b",
    ]

    if any(
        re.search(
            pattern,
            text,
        )
        for pattern in expense_patterns
    ):
        return "expense"

    # --------------------------------------------------------
    # E-COMMERCE
    # --------------------------------------------------------

    ecommerce_patterns = [
        r"\becommerce\b",
        r"\be-commerce\b",
        r"\bshopping\s+app\b",
        r"\bonline\s+store\b",
        r"\bshopping\s+website\b",
        r"\bstorefront\b",
    ]

    if any(
        re.search(
            pattern,
            text,
        )
        for pattern in ecommerce_patterns
    ):
        return "ecommerce"

    # --------------------------------------------------------
    # PORTFOLIO
    # --------------------------------------------------------

    portfolio_patterns = [
        r"\bportfolio\b",
        r"\bportfolio\s+website\b",
        r"\bpersonal\s+website\b",
        r"\bdeveloper\s+portfolio\b",
    ]

    if any(
        re.search(
            pattern,
            text,
        )
        for pattern in portfolio_patterns
    ):
        return "portfolio"

    # --------------------------------------------------------
    # CHAT APPLICATION
    # --------------------------------------------------------

    chat_patterns = [
        r"\bchat\s+app\b",
        r"\bchat\s+application\b",
        r"\bmessaging\s+app\b",
        r"\bmessaging\s+application\b",
        r"\breal[-\s]?time\s+chat\b",
    ]

    if any(
        re.search(
            pattern,
            text,
        )
        for pattern in chat_patterns
    ):
        return "chat"

    # --------------------------------------------------------
    # GENERIC FALLBACK
    # --------------------------------------------------------

    # Use normalized first meaningful words so that
    # completely unknown projects still get stable identities.

    stop_words = {
        "build",
        "create",
        "make",
        "develop",
        "design",
        "developing",
        "a",
        "an",
        "the",
        "using",
        "with",
        "for",
        "and",
        "application",
        "app",
        "website",
        "project",
        "modern",
    }

    words = [
        word
        for word in text.split()
        if word not in stop_words
    ]

    if not words:
        return "generic"

    return "-".join(
        words[:5]
    )


def project_key(
    user_prompt: str,
) -> str:
    """
    Return the stable semantic identity of a project.
    """

    return detect_project_identity(
        user_prompt
    )


# ============================================================
# MEMORY FILE
# ============================================================

def _ensure_memory_file() -> None:
    """
    Create the persistent memory directory/file if necessary.
    """

    MEMORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not MEMORY_FILE.exists():

        MEMORY_FILE.write_text(
            json.dumps(
                {
                    "projects": {}
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def load_memory() -> dict:
    """
    Load persistent CodePilot-Ai memory.
    """

    _ensure_memory_file()

    try:

        return json.loads(
            MEMORY_FILE.read_text(
                encoding="utf-8"
            )
        )

    except (
        json.JSONDecodeError,
        OSError,
    ):

        return {
            "projects": {}
        }


def save_memory(
    data: dict,
) -> None:
    """
    Persist CodePilot-Ai memory.
    """

    _ensure_memory_file()

    MEMORY_FILE.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# PROJECT MEMORY
# ============================================================

def get_project_memory(
    user_prompt: str,
) -> dict:
    """
    Retrieve memory using semantic project identity.
    """

    data = load_memory()

    key = project_key(
        user_prompt
    )

    return (
        data
        .get("projects", {})
        .get(key, {})
    )


def update_project_memory(
    user_prompt: str,
    **updates,
) -> dict:
    """
    Update persistent memory for a project.
    """

    data = load_memory()

    key = project_key(
        user_prompt
    )

    projects = data.setdefault(
        "projects",
        {},
    )

    existing = projects.get(
        key,
        {},
    )

    existing.update(
        updates
    )

    existing[
        "project_id"
    ] = key

    existing[
        "last_updated"
    ] = datetime.now().isoformat()

    projects[key] = existing

    save_memory(
        data
    )

    return existing


# ============================================================
# EXECUTION HISTORY
# ============================================================

def add_execution_history(
    user_prompt: str,
    passed: bool,
    summary: str,
) -> dict:
    """
    Store an executor result.
    """

    memory = get_project_memory(
        user_prompt
    )

    history = memory.get(
        "execution_history",
        [],
    )

    history.append(
        {
            "timestamp": (
                datetime.now().isoformat()
            ),
            "passed": bool(
                passed
            ),
            "summary": str(
                summary
            ),
        }
    )

    # Keep only the latest 10 executions.
    history = history[-10:]

    return update_project_memory(
        user_prompt,
        execution_history=history,
    )


# ============================================================
# REVIEW HISTORY
# ============================================================

def add_review_history(
    user_prompt: str,
    approved: bool,
    summary: str,
    issues: list[str] | None = None,
) -> dict:
    """
    Store reviewer results.
    """

    memory = get_project_memory(
        user_prompt
    )

    history = memory.get(
        "review_history",
        [],
    )

    history.append(
        {
            "timestamp": (
                datetime.now().isoformat()
            ),
            "approved": bool(
                approved
            ),
            "summary": str(
                summary
            ),
            "issues": (
                issues or []
            ),
        }
    )

    # Keep only the latest 10 reviews.
    history = history[-10:]

    return update_project_memory(
        user_prompt,
        review_history=history,
    )


# ============================================================
# PROJECT SNAPSHOT
# ============================================================

def add_project_snapshot(
    user_prompt: str,
    files: list[str],
    project_type: str = "generic",
) -> dict:
    """
    Store the current project snapshot.
    """

    return update_project_memory(
        user_prompt,
        project_type=project_type,
        files=files,
    )


# ============================================================
# MEMORY SUMMARY
# ============================================================

def get_memory_summary(
    user_prompt: str,
) -> dict:
    """
    Return a compact summary useful to the agent.
    """

    memory = get_project_memory(
        user_prompt
    )

    if not memory:
        return {}

    return {
        "project_id": memory.get(
            "project_id",
            project_key(
                user_prompt
            ),
        ),
        "project_type": memory.get(
            "project_type",
            "generic",
        ),
        "files": memory.get(
            "files",
            [],
        ),
        "executor_passed": memory.get(
            "executor_passed",
            False,
        ),
        "executor_attempts": memory.get(
            "executor_attempts",
            0,
        ),
        "review_attempts": memory.get(
            "review_attempts",
            0,
        ),
        "git_commit": memory.get(
            "git_commit",
            "",
        ),
        "last_validation_result": memory.get(
            "last_validation_result",
            "",
        ),
    }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print(
        "CodePilot-Ai Semantic Memory Test"
    )
    print("=" * 70)

    prompts = [
        "Build a Weather Dashboard",
        "Create a modern weather app",
        "Build a weather dashboard using HTML CSS and JavaScript",
        "Develop a weather forecast website",
        "Create a todo task manager",
        "Build a personal expense tracker",
    ]

    print()

    for prompt in prompts:

        identity = detect_project_identity(
            prompt
        )

        print(
            f"Prompt:   {prompt}"
        )

        print(
            f"Identity: {identity}"
        )

        print()

    # --------------------------------------------------------
    # Test shared weather memory
    # --------------------------------------------------------

    first_weather_prompt = (
        "Build a Weather Dashboard"
    )

    second_weather_prompt = (
        "Create a modern weather app"
    )

    update_project_memory(
        first_weather_prompt,
        project_type="weather",
        files=[
            "index.html",
            "styles.css",
            "app.js",
        ],
    )

    retrieved = get_project_memory(
        second_weather_prompt
    )

    print(
        "=" * 70
    )

    print(
        "Shared Weather Memory Test"
    )

    print(
        "=" * 70
    )

    print()

    print(
        json.dumps(
            retrieved,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()

    if retrieved:

        print(
            "RESULT: PASS"
        )

        print(
            "Different weather prompts "
            "resolve to the same project memory."
        )

    else:

        print(
            "RESULT: FAIL"
        )

    print()