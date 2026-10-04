from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path("generated_project").resolve()


# ============================================================
# RESULT HELPERS
# ============================================================

def _result(
    passed: bool,
    summary: str,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict:

    return {
        "passed": passed,
        "summary": summary,
        "errors": errors or [],
        "warnings": warnings or [],
    }


# ============================================================
# PROJECT FILE HELPERS
# ============================================================

def project_files() -> list[str]:

    if not PROJECT_ROOT.exists():
        return []

    files = []

    for path in PROJECT_ROOT.rglob("*"):

        if not path.is_file():
            continue

        relative = path.relative_to(
            PROJECT_ROOT
        ).as_posix()

        if relative.startswith(".codepilot"):
            continue

        if "__pycache__" in relative:
            continue

        files.append(relative)

    return sorted(files)


def read_project_file(
    filepath: str,
) -> str:

    path = PROJECT_ROOT / filepath

    if not path.exists():
        return ""

    try:

        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )

    except Exception:

        return ""


# ============================================================
# REQUIRED FILE CHECK
# ============================================================

def check_required_files(
    errors: list[str],
) -> None:

    files = project_files()

    if not files:

        errors.append(
            "generated_project does not contain "
            "any generated files."
        )

        return

    required_web_files = {
        "index.html",
        "styles.css",
        "app.js",
    }

    existing = set(files)

    missing = (
        required_web_files - existing
    )

    for filepath in sorted(missing):

        errors.append(
            f"Required file is missing: {filepath}"
        )


# ============================================================
# HTML CHECK
# ============================================================

def check_html(
    filepath: str,
    html: str,
    errors: list[str],
    warnings: list[str],
) -> None:

    if not html.strip():

        errors.append(
            f"{filepath} is empty."
        )

        return

    # Case-insensitive DOCTYPE check.
    if "<!doctype html>" not in html.lower():

        warnings.append(
            f"{filepath} does not contain "
            "a standard DOCTYPE declaration."
        )

    if "<html" not in html.lower():

        errors.append(
            f"{filepath} does not contain "
            "an <html> element."
        )

    if "<body" not in html.lower():

        errors.append(
            f"{filepath} does not contain "
            "a <body> element."
        )


# ============================================================
# CSS CHECK
# ============================================================

def check_css(
    filepath: str,
    css: str,
    errors: list[str],
    warnings: list[str],
) -> None:

    if not css.strip():

        errors.append(
            f"{filepath} is empty."
        )

        return

    opening = css.count("{")
    closing = css.count("}")

    if opening != closing:

        errors.append(
            f"{filepath} has unbalanced CSS "
            "curly braces."
        )

    if "@media" not in css.lower():

        warnings.append(
            f"{filepath} does not contain an "
            "@media responsive rule."
        )


# ============================================================
# JAVASCRIPT SYNTAX CHECK
# ============================================================

def _check_with_node(
    filepath: str,
    content: str,
) -> tuple[bool, str]:

    node = shutil.which("node")

    if not node:

        return (
            False,
            "Node.js is not installed.",
        )

    temp_path = None

    try:

        # ----------------------------------------------------
        # IMPORTANT WINDOWS ENCODING FIX
        #
        # Do NOT send JavaScript through subprocess stdin.
        #
        # Windows may try to encode Unicode characters such
        # as:
        #
        #   ☀
        #   🌤
        #   ☁
        #   🌧
        #
        # using the system "charmap", which causes:
        #
        # UnicodeEncodeError
        #
        # Instead write the JavaScript as UTF-8 to a temporary
        # .js file and ask Node to syntax-check that file.
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".js",
            delete=False,
        ) as temp_file:

            temp_file.write(content)

            temp_path = temp_file.name

        process = subprocess.run(
            [
                node,
                "--check",
                temp_path,
            ],
            capture_output=True,
            timeout=20,
        )

        if process.returncode == 0:

            return (
                True,
                "",
            )

        # Decode Node output safely.
        stderr = (
            process.stderr.decode(
                "utf-8",
                errors="replace",
            )
            if process.stderr
            else ""
        )

        stdout = (
            process.stdout.decode(
                "utf-8",
                errors="replace",
            )
            if process.stdout
            else ""
        )

        output = (
            stderr
            or stdout
            or "Unknown Node.js syntax error."
        )

        return (
            False,
            output.strip(),
        )

    except subprocess.TimeoutExpired:

        return (
            False,
            "Node.js syntax check timed out.",
        )

    except Exception as exc:

        return (
            False,
            f"{type(exc).__name__}: {exc}",
        )

    finally:

        if temp_path:

            try:

                Path(temp_path).unlink(
                    missing_ok=True
                )

            except Exception:
                pass


def _balanced_javascript(
    content: str,
) -> tuple[bool, str]:

    pairs = {
        "{": "}",
        "[": "]",
        "(": ")",
    }

    stack = []

    in_single = False
    in_double = False
    in_template = False

    escape = False

    for char in content:

        if escape:

            escape = False
            continue

        if char == "\\" and (
            in_single
            or in_double
            or in_template
        ):

            escape = True
            continue

        if char == "'" and not (
            in_double
            or in_template
        ):

            in_single = not in_single
            continue

        if char == '"' and not (
            in_single
            or in_template
        ):

            in_double = not in_double
            continue

        if char == "`" and not (
            in_single
            or in_double
        ):

            in_template = not in_template
            continue

        if (
            in_single
            or in_double
            or in_template
        ):

            continue

        if char in pairs:

            stack.append(char)

        elif char in "}]":

            if not stack:

                return (
                    False,
                    f"Unexpected closing character: {char}",
                )

            opening = stack.pop()

            if pairs[opening] != char:

                return (
                    False,
                    f"Mismatched brackets: "
                    f"{opening} -> {char}",
                )

    if in_single:

        return (
            False,
            "Unterminated single-quoted string.",
        )

    if in_double:

        return (
            False,
            "Unterminated double-quoted string.",
        )

    if in_template:

        return (
            False,
            "Unterminated template string.",
        )

    if stack:

        return (
            False,
            "Unclosed JavaScript bracket.",
        )

    return (
        True,
        "",
    )


def check_javascript_syntax(
    filepath: str,
    content: str,
    errors: list[str],
) -> None:

    if not content.strip():

        errors.append(
            f"{filepath} is empty."
        )

        return

    node = shutil.which("node")

    if node:

        passed, message = (
            _check_with_node(
                filepath,
                content,
            )
        )

        if not passed:

            errors.append(
                f"{filepath} has JavaScript "
                f"syntax errors:\n{message}"
            )

        return

    # Fallback when Node.js is unavailable.
    passed, message = (
        _balanced_javascript(
            content
        )
    )

    if not passed:

        errors.append(
            f"{filepath} failed JavaScript "
            f"balance checking: {message}"
        )


# ============================================================
# PLACEHOLDER DETECTION
# ============================================================

def detect_placeholder_code(
    filepath: str,
    content: str,
    errors: list[str],
) -> None:

    if not content.strip():
        return

    patterns = [
        r"\bTODO\s*:\s*implement\b",
        r"\bTODO\s*:\s*add\b",
        r"\bIMPLEMENT\s+HERE\b",
        r"\bYOUR\s+CODE\s+HERE\b",
        r"\bPLACEHOLDER\s+IMPLEMENTATION\b",
        r"\bCOMING\s+SOON\b",
    ]

    for pattern in patterns:

        if re.search(
            pattern,
            content,
            flags=re.IGNORECASE,
        ):

            errors.append(
                f"{filepath} contains "
                "placeholder implementation text."
            )

            return


# ============================================================
# PROJECT TYPE DETECTION
# ============================================================

def detect_project_type(
    files: dict[str, str],
) -> str:

    combined = "\n".join(
        files.values()
    ).lower()

    filenames = " ".join(
        files.keys()
    ).lower()

    text = (
        combined
        + "\n"
        + filenames
    )

    weather_signals = [
        "weather",
        "open-meteo",
        "temperature",
        "humidity",
        "windspeed",
        "wind speed",
        "weatherform",
        "cityinput",
    ]

    weather_score = sum(
        1
        for signal in weather_signals
        if signal in text
    )

    if weather_score >= 2:

        return "weather"

    expense_analytics_signals = [
        "expense analytics",
        "expense dashboard",
        "average expense",
        "highest expense",
        "filter expenses",
        "spending analytics",
    ]

    if any(
        signal in text
        for signal in expense_analytics_signals
    ):

        return "expense_analytics"

    expense_signals = [
        "expense tracker",
        "expense tracking",
        "expense manager",
        "expense management",
    ]

    if any(
        signal in text
        for signal in expense_signals
    ):

        return "expense"

    todo_signals = [
        "todo",
        "to-do",
        "task manager",
        "task management",
    ]

    if any(
        signal in text
        for signal in todo_signals
    ):

        return "todo"

    return "generic"


# ============================================================
# WEATHER CHECKS
# ============================================================

def check_weather_project(
    files: dict[str, str],
    errors: list[str],
    warnings: list[str],
) -> None:

    html = files.get(
        "index.html",
        "",
    )

    css = files.get(
        "styles.css",
        "",
    )

    js = files.get(
        "app.js",
        "",
    )

    combined = (
        html
        + "\n"
        + css
        + "\n"
        + js
    ).lower()

    # Search input.
    if not (
        "cityinput" in combined
        or 'type="search"' in combined
        or 'type="text"' in html.lower()
    ):

        errors.append(
            "Weather Dashboard does not "
            "appear to contain a city search input."
        )

    # Search action.
    if not (
        "search" in combined
        or "submit" in combined
        or "addeventlistener" in js.lower()
    ):

        errors.append(
            "Weather Dashboard does not "
            "appear to implement a search action."
        )

    # API request.
    if not (
        "fetch(" in js.lower()
        or "xmlhttprequest" in js.lower()
    ):

        errors.append(
            "Weather Dashboard does not "
            "appear to call a weather API."
        )

    # API provider.
    if not (
        "open-meteo" in combined
        or "api.openweathermap" in combined
        or "weatherapi" in combined
        or "weather.gov" in combined
    ):

        warnings.append(
            "Weather Dashboard does not contain "
            "a recognizable weather API provider."
        )

    # Required weather information.
    required_weather_terms = {
        "temperature": [
            "temperature",
            "temp",
        ],
        "humidity": [
            "humidity",
        ],
        "wind": [
            "wind",
        ],
        "condition": [
            "condition",
            "weathercode",
            "weather code",
            "description",
        ],
    }

    for label, terms in (
        required_weather_terms.items()
    ):

        if not any(
            term in combined
            for term in terms
        ):

            errors.append(
                "Weather Dashboard does not "
                f"appear to display {label}."
            )

    # Loading state.
    loading_patterns = [
        "loading",
        "setstatus(",
        "status.textcontent",
        "statusmessage",
        "searchbutton.disabled",
        ".disabled",
    ]

    has_loading_logic = any(
        pattern in combined
        for pattern in loading_patterns
    )

    if not has_loading_logic:

        errors.append(
            "Weather Dashboard does not "
            "appear to implement a loading state."
        )

    # Error state.
    error_patterns = [
        "catch(",
        "catch (",
        "error",
        "failed",
        "unable",
        "try {",
        "try{",
    ]

    if not any(
        pattern in js.lower()
        for pattern in error_patterns
    ):

        errors.append(
            "Weather Dashboard does not "
            "appear to implement an error state."
        )

    # Responsive CSS.
    if "@media" not in css.lower():

        warnings.append(
            "Weather Dashboard does not contain "
            "a CSS @media responsive rule."
        )


# ============================================================
# EXPENSE ANALYTICS CHECKS
# ============================================================

def check_expense_analytics_project(
    files: dict[str, str],
    errors: list[str],
    warnings: list[str],
) -> None:

    combined = "\n".join(
        files.values()
    ).lower()

    if "expense" not in combined:

        errors.append(
            "Expense Analytics project "
            "does not appear to handle expenses."
        )

    if "analytics" not in combined:

        errors.append(
            "Expense Analytics project "
            "does not appear to implement analytics."
        )

    filter_terms = [
        "filter",
        "select",
        "category",
        "date",
    ]

    if not any(
        term in combined
        for term in filter_terms
    ):

        warnings.append(
            "Expense Analytics project does not "
            "appear to contain filtering functionality."
        )


# ============================================================
# EXPENSE CHECKS
# ============================================================

def check_expense_project(
    files: dict[str, str],
    errors: list[str],
    warnings: list[str],
) -> None:

    combined = "\n".join(
        files.values()
    ).lower()

    if "expense" not in combined:

        errors.append(
            "Expense Tracker does not "
            "appear to implement expense functionality."
        )

    if not (
        "add" in combined
        or "submit" in combined
        or "save" in combined
    ):

        warnings.append(
            "Expense Tracker does not clearly "
            "expose an add/save action."
        )


# ============================================================
# TODO CHECKS
# ============================================================

def check_todo_project(
    files: dict[str, str],
    errors: list[str],
    warnings: list[str],
) -> None:

    combined = "\n".join(
        files.values()
    ).lower()

    if not (
        "todo" in combined
        or "task" in combined
    ):

        errors.append(
            "Todo project does not appear "
            "to implement task functionality."
        )

    if not (
        "add" in combined
        or "submit" in combined
    ):

        warnings.append(
            "Todo project does not clearly "
            "expose an add-task action."
        )


# ============================================================
# PROJECT-SPECIFIC CHECKS
# ============================================================

def run_project_specific_checks(
    files: dict[str, str],
    errors: list[str],
    warnings: list[str],
) -> None:

    project_type = detect_project_type(
        files
    )

    print(
        f"[EXECUTOR] Detected project type: "
        f"{project_type}"
    )

    if project_type == "weather":

        check_weather_project(
            files,
            errors,
            warnings,
        )

    elif project_type == "expense_analytics":

        check_expense_analytics_project(
            files,
            errors,
            warnings,
        )

    elif project_type == "expense":

        check_expense_project(
            files,
            errors,
            warnings,
        )

    elif project_type == "todo":

        check_todo_project(
            files,
            errors,
            warnings,
        )


# ============================================================
# MAIN EXECUTION FUNCTION
# ============================================================

def execute_project() -> dict:

    print()
    print("=" * 64)
    print("CodePilot-Ai Execute/Test Module")
    print("=" * 64)

    print(
        "[EXECUTOR] Starting generated-project tests..."
    )

    errors: list[str] = []

    warnings: list[str] = []

    # --------------------------------------------------------
    # PROJECT ROOT
    # --------------------------------------------------------

    if not PROJECT_ROOT.exists():

        return _result(
            False,
            "generated_project directory does not exist.",
            [
                "generated_project directory does not exist."
            ],
            [],
        )

    # --------------------------------------------------------
    # REQUIRED FILES
    # --------------------------------------------------------

    check_required_files(
        errors
    )

    if errors:

        print(
            "[EXECUTOR] Required-file checks failed."
        )

        for error in errors:

            print(
                f"[EXECUTOR] ERROR: {error}"
            )

        return _result(
            False,
            "Generated project is missing required files.",
            errors,
            warnings,
        )

    # --------------------------------------------------------
    # LOAD FILES
    # --------------------------------------------------------

    files: dict[str, str] = {}

    for filepath in project_files():

        files[filepath] = (
            read_project_file(
                filepath
            )
        )

    # --------------------------------------------------------
    # FILE-LEVEL VALIDATION
    # --------------------------------------------------------

    for filepath, content in files.items():

        suffix = (
            Path(filepath)
            .suffix
            .lower()
        )

        if suffix == ".html":

            check_html(
                filepath,
                content,
                errors,
                warnings,
            )

        elif suffix == ".css":

            check_css(
                filepath,
                content,
                errors,
                warnings,
            )

        elif suffix in {
            ".js",
            ".mjs",
            ".cjs",
        }:

            check_javascript_syntax(
                filepath,
                content,
                errors,
            )

        detect_placeholder_code(
            filepath,
            content,
            errors,
        )

    # --------------------------------------------------------
    # PROJECT-SPECIFIC VALIDATION
    # --------------------------------------------------------

    run_project_specific_checks(
        files,
        errors,
        warnings,
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    passed = len(errors) == 0

    if passed:

        summary = (
            "Generated project passed "
            "all automated execution checks."
        )

        print()
        print(
            "[EXECUTOR] Automated checks passed."
        )

    else:

        summary = (
            "Generated project failed "
            "automated execution checks."
        )

        print()
        print(
            "[EXECUTOR] Automated checks failed."
        )

    for error in errors:

        print(
            f"[EXECUTOR] ERROR: {error}"
        )

    for warning in warnings:

        print(
            f"[EXECUTOR] WARNING: {warning}"
        )

    result = _result(
        passed=passed,
        summary=summary,
        errors=errors,
        warnings=warnings,
    )

    print()
    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    return result


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    result = execute_project()

    print()
    print("=" * 64)
    print("Executor test completed.")
    print("=" * 64)

    if result["passed"]:

        print(
            "STATUS: PASS"
        )

    else:

        print(
            "STATUS: FAIL"
        )