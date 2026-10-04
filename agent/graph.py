from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from agent.executor_node import executor_node
from agent.git_context_node import git_context_node
from agent.git_checkpoint_node import git_checkpoint_node

from agent.memory import (
    get_project_memory,
    update_project_memory,
    add_execution_history,
    add_review_history,
    add_project_snapshot,
)

from agent.states import (
    CoderState,
    File,
    ImplementationTask,
    Plan,
    ReviewResult,
    TaskPlan,
)

import agent.git_manager as git_manager
try:
    from agent.rag import build_project_rag
except Exception:
    build_project_rag = None


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

PROJECT_ROOT = Path("generated_project").resolve()

# True  -> ZERO Gemini/Groq calls for every project.
# False -> generic projects may use Gemini/Groq.
LOCAL_TEST_MODE = True

GEMINI_MODEL = "gemini-3.8-flash"
GROQ_MODEL = "openai/gpt-oss-20b"

MAX_REVIEW_ATTEMPTS = 3


# ============================================================
# BASIC FILE HELPERS
# ============================================================

def init_project_root() -> None:
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)


def clean_project_root() -> None:
    init_project_root()

    for item in PROJECT_ROOT.iterdir():
        # NEVER delete the generated project's Git repository.
        # Git history is required for checkpoints and safe recovery.
        if item.name == ".git":
            continue

        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except Exception as exc:
            print(f"[WARN] Could not remove {item}: {exc}")


def should_preserve_existing_project(prompt: str) -> bool:
    """Return True when a request clearly targets an existing project."""
    text = (prompt or "").lower()

    modification_phrases = [
        "update ",
        "modify ",
        "change ",
        "improve ",
        "enhance ",
        "extend ",
        "fix ",
        "add ",
        "remove ",
        "delete ",
        "replace ",
        "existing ",
        "current ",
        "previous ",
        "my dashboard",
        "my app",
        "my project",
        "my weather",
    ]

    if not any(phrase in text for phrase in modification_phrases):
        return False

    return (
        (PROJECT_ROOT / ".git").exists()
        or any(PROJECT_ROOT.glob("*"))
    )


def prepare_project_root_for_request(prompt: str) -> None:
    """Clean only for new projects; preserve files for modifications."""
    if should_preserve_existing_project(prompt):
        print(
            "[PROJECT] Existing-project request detected; "
            "preserving generated_project files."
        )
        return

    clean_project_root()


def write_project_file(path: str, content: str) -> None:
    """Write only a valid relative project filepath.

    LLM/planner output must never be allowed to become a filename.
    Reject obvious source-code fragments instead of creating garbage
    files such as ``_dir()`` or ``t(prompt...)``.
    """
    raw = str(path or "").strip().strip("`").replace("\\", "/")

    if not raw or raw.startswith(("/", "\\")) or ".." in Path(raw).parts:
        raise ValueError(f"Invalid project filepath: {path!r}")

    allowed = {
        ".html", ".css", ".js", ".json", ".md", ".txt",
        ".py", ".ts", ".tsx", ".jsx", ".vue", ".csv",
        ".env", ".gitignore", ".xml", ".svg",
    }
    suffix = Path(raw).suffix.lower()

    suspicious_markers = [
        "\u0000", "->", "\u2192", "def ", "_dir(", "prompt:",
        "= [", "t(prompt", "serve_existing_project", "", "", "",
    ]

    if suffix not in allowed or any(marker in raw for marker in suspicious_markers):
        raise ValueError(f"Rejected malformed project filepath: {path!r}")

    target = PROJECT_ROOT / raw
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def read_project_file(path: str) -> str:
    target = PROJECT_ROOT / path

    if not target.exists():
        return ""

    return target.read_text(encoding="utf-8")


def project_files() -> list[str]:
    if not PROJECT_ROOT.exists():
        return []

    ignored_directories = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
    }

    files = []

    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue

        relative = path.relative_to(PROJECT_ROOT)
        parts = relative.parts

        if any(
            part in ignored_directories
            for part in parts
        ):
            continue

        name = path.name
        allowed_suffixes = {
            ".html", ".css", ".js", ".json", ".md", ".txt",
            ".py", ".ts", ".tsx", ".jsx", ".vue", ".csv",
            ".xml", ".svg",
        }
        allowed_names = {
            ".env", ".gitignore", "README", "README.md",
        }
        suspicious_markers = [
            "->", "\u2192", "_dir(", "t(prompt",
            "serve_existing_project", "", "", "",
        ]

        if name not in allowed_names and path.suffix.lower() not in allowed_suffixes:
            continue

        if any(marker in name for marker in suspicious_markers):
            continue

        files.append(
            str(relative).replace("\\", "/")
        )

    return sorted(files)


# ============================================================
# LLM HELPERS
# ============================================================

def create_gemini_llm():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from .env")

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
    )


def create_groq_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing from .env")

    return ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=api_key,
    )


def response_to_text(response) -> str:
    if hasattr(response, "content"):
        content = response.content

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []

            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))

            return "\n".join(parts)

    return str(response)


def invoke_llm(prompt: str) -> str:
    if LOCAL_TEST_MODE:
        raise RuntimeError(
            "LLM invocation blocked because LOCAL_TEST_MODE=True."
        )

    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "groq":
        primary_name = "Groq"
        fallback_name = "Gemini"
        primary = create_groq_llm
        fallback = create_gemini_llm
    else:
        primary_name = "Gemini"
        fallback_name = "Groq"
        primary = create_gemini_llm
        fallback = create_groq_llm

    try:
        response = primary().invoke(prompt)
        return response_to_text(response)

    except Exception as primary_error:
        print(
            f"[LLM] {primary_name} failed: "
            f"{type(primary_error).__name__}"
        )

        try:
            response = fallback().invoke(prompt)
            return response_to_text(response)

        except Exception as fallback_error:
            print(
                f"[LLM] {fallback_name} also failed: "
                f"{type(fallback_error).__name__}"
            )

            raise RuntimeError(
                f"Both {primary_name} and {fallback_name} failed."
            ) from fallback_error


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text: str):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    decoder = json.JSONDecoder()

    for index, char in enumerate(text):
        if char not in "{[":
            continue

        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue

    raise ValueError("Could not extract valid JSON.")


# ============================================================
# STATE HELPER
# ============================================================

def update_state(state: CoderState, **updates) -> CoderState:
    data = state.model_dump()
    data.update(updates)
    return CoderState(**data)


# ============================================================
# PROJECT DETECTION
# ============================================================

def is_expense_project(prompt: str) -> bool:
    text = prompt.lower()

    patterns = [
        r"\bexpense\s+tracker\b",
        r"\bexpense\s+tracking\b",
        r"\bexpense\s+manager\b",
        r"\bexpense\s+management\b",
        r"\btrack\s+expenses\b",
        r"\bpersonal\s+finance\s+tracker\b",
        r"\bpersonal\s+expense\s+tracker\b",
        r"\bexpense\s+tracking\s+application\b",
        r"\bexpense\s+tracking\s+app\b",
        r"\bexpense\s+management\s+application\b",
        r"\bexpense\s+management\s+app\b",
    ]

    return any(re.search(pattern, text) for pattern in patterns)


def is_todo_project(prompt: str) -> bool:
    text = prompt.lower()

    if is_expense_project(text):
        return False

    patterns = [
        r"\btodo\s+app\b",
        r"\btodo\s+application\b",
        r"\btodo\s+list\b",
        r"\bto-do\s+list\b",
        r"\btask\s+manager\b",
        r"\btask\s+management\s+app\b",
        r"\btask\s+management\s+application\b",
        r"\btask\s+list\s+app\b",
        r"\btask\s+list\s+application\b",
        r"\bproductivity\s+todo\b",
        r"\bbuild\s+a\s+todo\b",
        r"\bcreate\s+a\s+todo\b",
        r"\bdevelop\s+a\s+todo\b",
    ]

    return any(re.search(pattern, text) for pattern in patterns)


def is_weather_project(prompt: str) -> bool:
    text = prompt.lower()

    patterns = [
        r"\bweather\s+dashboard\b",
        r"\bweather\s+app\b",
        r"\bweather\s+application\b",
        r"\bweather\s+website\b",
        r"\bweather\s+forecast\b",
        r"\bweather\s+tracker\b",
    ]

    return any(re.search(pattern, text) for pattern in patterns)


def is_expense_analytics_project(prompt: str) -> bool:
    text = prompt.lower()

    return (
        ("expense" in text or "spending" in text)
        and (
            "analytics" in text
            or "dashboard" in text
            or "average expense" in text
            or "highest expense" in text
            or "filter expenses" in text
        )
    )


# ============================================================
# TODO APPLICATION
# ============================================================

TODO_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Todo App</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <main class="app">
        <section class="hero">
            <p class="eyebrow">PRODUCTIVITY</p>
            <h1>My Tasks</h1>
            <p class="subtitle">Organize your day and stay focused.</p>
        </section>

        <section class="composer">
            <input
                id="taskInput"
                type="text"
                maxlength="200"
                placeholder="What needs to be done?"
                autocomplete="off"
            >
            <button id="addTaskButton">Add Task</button>
        </section>

        <section class="toolbar">
            <div class="filters">
                <button class="filter active" data-filter="all">All</button>
                <button class="filter" data-filter="active">Active</button>
                <button class="filter" data-filter="completed">Completed</button>
            </div>
            <button id="clearCompleted" class="ghost">Clear completed</button>
        </section>

        <section id="taskList" class="task-list"></section>

        <footer class="footer">
            <span id="taskCount">0 tasks</span>
            <span>Saved locally</span>
        </footer>
    </main>

    <script src="app.js"></script>
</body>
</html>
"""

TODO_STYLES_CSS = """* {
    box-sizing: border-box;
}

:root {
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #172033;
    background: #f4f7fb;
}

body {
    margin: 0;
    min-height: 100vh;
    background:
        radial-gradient(circle at top left, #e8efff 0, transparent 35%),
        #f4f7fb;
}

button,
input {
    font: inherit;
}

button {
    cursor: pointer;
}

.app {
    width: min(760px, calc(100% - 32px));
    margin: 60px auto;
}

.hero {
    margin-bottom: 28px;
}

.eyebrow {
    margin: 0 0 8px;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.16em;
    color: #5b6ee1;
}

h1 {
    margin: 0;
    font-size: clamp(36px, 7vw, 58px);
    line-height: 1;
}

.subtitle {
    color: #697386;
}

.composer {
    display: flex;
    gap: 10px;
    padding: 10px;
    background: white;
    border: 1px solid #e5e9f2;
    border-radius: 18px;
    box-shadow: 0 12px 35px rgba(31, 42, 68, 0.08);
}

.composer input {
    flex: 1;
    min-width: 0;
    border: 0;
    outline: 0;
    padding: 12px;
    background: transparent;
}

.composer button {
    border: 0;
    border-radius: 12px;
    padding: 12px 18px;
    color: white;
    background: #5865d8;
    font-weight: 700;
}

.toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin: 22px 0 14px;
}

.filters {
    display: flex;
    gap: 6px;
}

.filter,
.ghost {
    border: 0;
    background: transparent;
    color: #737c8f;
    padding: 8px 11px;
    border-radius: 9px;
}

.filter.active {
    color: #2637a3;
    background: #e9edff;
    font-weight: 700;
}

.task-list {
    display: grid;
    gap: 10px;
}

.task {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px;
    background: white;
    border: 1px solid #e5e9f2;
    border-radius: 14px;
}

.task.completed .task-text {
    color: #9aa2b1;
    text-decoration: line-through;
}

.check {
    width: 20px;
    height: 20px;
}

.task-text {
    flex: 1;
    overflow-wrap: anywhere;
}

.delete {
    border: 0;
    background: transparent;
    color: #b04a58;
}

.empty {
    padding: 40px 20px;
    text-align: center;
    color: #8a93a5;
}

.footer {
    display: flex;
    justify-content: space-between;
    color: #8a93a5;
    font-size: 13px;
    padding: 18px 4px;
}

@media (max-width: 600px) {
    .app {
        margin: 30px auto;
    }

    .composer {
        flex-direction: column;
    }

    .toolbar {
        align-items: flex-start;
        flex-direction: column;
    }
}
"""

TODO_APP_JS = """const STORAGE_KEY = "codepilot-todos";

let tasks = loadTasks();
let currentFilter = "all";

const taskInput = document.getElementById("taskInput");
const addTaskButton = document.getElementById("addTaskButton");
const taskList = document.getElementById("taskList");
const taskCount = document.getElementById("taskCount");
const clearCompleted = document.getElementById("clearCompleted");

function loadTasks() {
    try {
        const saved = localStorage.getItem(STORAGE_KEY);
        return saved ? JSON.parse(saved) : [];
    } catch {
        return [];
    }
}

function saveTasks() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(tasks));
}

function addTask() {
    const text = taskInput.value.trim();

    if (!text) {
        taskInput.focus();
        return;
    }

    tasks.unshift({
        id: crypto.randomUUID(),
        text,
        completed: false
    });

    taskInput.value = "";
    saveTasks();
    render();
}

function toggleTask(id) {
    tasks = tasks.map(task =>
        task.id === id
            ? { ...task, completed: !task.completed }
            : task
    );

    saveTasks();
    render();
}

function deleteTask(id) {
    tasks = tasks.filter(task => task.id !== id);
    saveTasks();
    render();
}

function visibleTasks() {
    if (currentFilter === "active") {
        return tasks.filter(task => !task.completed);
    }

    if (currentFilter === "completed") {
        return tasks.filter(task => task.completed);
    }

    return tasks;
}

function render() {
    const visible = visibleTasks();

    taskList.innerHTML = "";

    if (!visible.length) {
        taskList.innerHTML = `
            <div class="empty">
                No tasks in this view.
            </div>
        `;
    }

    visible.forEach(task => {
        const article = document.createElement("article");
        article.className = `task ${task.completed ? "completed" : ""}`;

        article.innerHTML = `
            <input
                class="check"
                type="checkbox"
                ${task.completed ? "checked" : ""}
                aria-label="Mark task complete"
            >
            <span class="task-text"></span>
            <button class="delete" type="button">Delete</button>
        `;

        article.querySelector(".task-text").textContent = task.text;

        article.querySelector(".check")
            .addEventListener("change", () => toggleTask(task.id));

        article.querySelector(".delete")
            .addEventListener("click", () => deleteTask(task.id));

        taskList.appendChild(article);
    });

    const remaining = tasks.filter(task => !task.completed).length;
    taskCount.textContent =
        `${remaining} ${remaining === 1 ? "task" : "tasks"} left`;
}

addTaskButton.addEventListener("click", addTask);

taskInput.addEventListener("keydown", event => {
    if (event.key === "Enter") {
        addTask();
    }
});

document.querySelectorAll(".filter").forEach(button => {
    button.addEventListener("click", () => {
        document.querySelectorAll(".filter")
            .forEach(item => item.classList.remove("active"));

        button.classList.add("active");
        currentFilter = button.dataset.filter;
        render();
    });
});

clearCompleted.addEventListener("click", () => {
    tasks = tasks.filter(task => !task.completed);
    saveTasks();
    render();
});

render();
"""


def local_todo_plan() -> Plan:
    return Plan(
        name="Todo App",
        description="A responsive productivity application for managing personal tasks.",
        techstack="HTML, CSS, Vanilla JavaScript",
        features=[
            "Add tasks",
            "Mark tasks as completed",
            "Delete tasks",
            "Filter all, active, and completed tasks",
            "Clear completed tasks",
            "Persist tasks with localStorage",
            "Responsive user interface",
        ],
        files=[
            File(path="index.html", purpose="Application structure."),
            File(path="styles.css", purpose="Responsive styling."),
            File(path="app.js", purpose="Task management logic."),
        ],
    )


def local_todo_task_plan() -> TaskPlan:
    return TaskPlan(
        implementation_steps=[
            ImplementationTask(
                filepath="index.html",
                task_description="Create the complete Todo application interface.",
            ),
            ImplementationTask(
                filepath="styles.css",
                task_description="Create a polished responsive Todo design.",
            ),
            ImplementationTask(
                filepath="app.js",
                task_description="Implement CRUD, filtering, completion and localStorage.",
            ),
        ]
    )


def validate_todo_project() -> ReviewResult:
    required = ["index.html", "styles.css", "app.js"]
    missing = [
        file for file in required
        if not (PROJECT_ROOT / file).exists()
    ]

    if missing:
        return ReviewResult(
            approved=False,
            issues=[f"Missing file: {file}" for file in missing],
            summary="Todo validation failed.",
        )

    html = read_project_file("index.html")
    css = read_project_file("styles.css")
    js = read_project_file("app.js")

    issues = []

    if "taskInput" not in html:
        issues.append("Task input is missing.")

    if "localStorage" not in js:
        issues.append("localStorage persistence is missing.")

    if "completed" not in js:
        issues.append("Task completion logic is missing.")

    if "display: flex" not in css:
        issues.append("Responsive/layout styling is incomplete.")

    return ReviewResult(
        approved=not issues,
        issues=issues,
        summary=(
            "Local Todo review passed."
            if not issues
            else "Local Todo review found issues."
        ),
    )


def generate_todo_file(filepath: str) -> str:
    return {
        "index.html": TODO_INDEX_HTML,
        "styles.css": TODO_STYLES_CSS,
        "app.js": TODO_APP_JS,
    }[filepath]


# ============================================================
# EXPENSE TRACKER
# ============================================================

EXPENSE_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Expense Tracker</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <main class="app">
        <header class="hero">
            <p class="eyebrow">PERSONAL FINANCE</p>
            <h1>Expense Tracker</h1>
            <p>Track spending and understand where your money goes.</p>
        </header>

        <section class="summary">
            <article>
                <span>Total spending</span>
                <strong id="totalAmount">₹0.00</strong>
            </article>
            <article>
                <span>Transactions</span>
                <strong id="transactionCount">0</strong>
            </article>
        </section>

        <section class="panel">
            <h2>Add expense</h2>

            <form id="expenseForm">
                <input id="expenseTitle" type="text"
                    placeholder="Expense name" required>

                <input id="expenseAmount" type="number"
                    min="0" step="0.01"
                    placeholder="Amount" required>

                <select id="expenseCategory">
                    <option value="Food">Food</option>
                    <option value="Transport">Transport</option>
                    <option value="Shopping">Shopping</option>
                    <option value="Bills">Bills</option>
                    <option value="Other">Other</option>
                </select>

                <button type="submit">Add expense</button>
            </form>
        </section>

        <section class="panel">
            <div class="panel-header">
                <h2>Transactions</h2>
                <button id="clearExpenses" class="danger">
                    Clear all
                </button>
            </div>

            <div id="expenseList" class="expense-list"></div>
        </section>
    </main>

    <script src="app.js"></script>
</body>
</html>
"""

EXPENSE_STYLES_CSS = """* {
    box-sizing: border-box;
}

:root {
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #172033;
    background: #f5f7fb;
}

body {
    margin: 0;
    min-height: 100vh;
    background:
        radial-gradient(circle at top right, #e8edff 0, transparent 34%),
        #f5f7fb;
}

button,
input,
select {
    font: inherit;
}

button {
    cursor: pointer;
}

.app {
    width: min(900px, calc(100% - 32px));
    margin: 48px auto;
}

.hero {
    margin-bottom: 26px;
}

.eyebrow {
    margin: 0 0 8px;
    color: #5865d8;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .16em;
}

h1 {
    margin: 0;
    font-size: clamp(34px, 7vw, 56px);
}

.hero p:last-child {
    color: #70798b;
}

.summary {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 14px;
    margin-bottom: 18px;
}

.summary article,
.panel {
    background: white;
    border: 1px solid #e4e8f0;
    border-radius: 18px;
    box-shadow: 0 12px 32px rgba(25, 37, 63, .06);
}

.summary article {
    padding: 22px;
}

.summary span {
    display: block;
    color: #7a8394;
    font-size: 13px;
}

.summary strong {
    display: block;
    margin-top: 8px;
    font-size: 28px;
}

.panel {
    padding: 22px;
    margin-top: 16px;
}

.panel h2 {
    margin-top: 0;
}

form {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr auto;
    gap: 10px;
}

input,
select {
    min-width: 0;
    border: 1px solid #dfe4ed;
    border-radius: 10px;
    padding: 12px;
    outline: none;
}

input:focus,
select:focus {
    border-color: #6572df;
}

form button {
    border: 0;
    border-radius: 10px;
    padding: 12px 18px;
    background: #5865d8;
    color: white;
    font-weight: 700;
}

.panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.danger {
    border: 0;
    background: transparent;
    color: #b04454;
}

.expense-list {
    display: grid;
    gap: 10px;
}

.expense {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    padding: 15px;
    border: 1px solid #e7eaf1;
    border-radius: 12px;
}

.expense-title {
    font-weight: 700;
}

.expense-meta {
    margin-top: 4px;
    color: #80899a;
    font-size: 13px;
}

.expense-right {
    text-align: right;
}

.amount {
    font-weight: 800;
}

.delete-expense {
    border: 0;
    background: transparent;
    color: #b04454;
    font-size: 12px;
}

.empty {
    padding: 30px;
    text-align: center;
    color: #8b94a5;
}

@media (max-width: 760px) {
    form {
        grid-template-columns: 1fr 1fr;
    }

    form input:first-child,
    form button {
        grid-column: span 2;
    }
}

@media (max-width: 520px) {
    .summary {
        grid-template-columns: 1fr;
    }

    form {
        grid-template-columns: 1fr;
    }

    form input:first-child,
    form button {
        grid-column: auto;
    }
}
"""

EXPENSE_APP_JS = """const STORAGE_KEY = "codepilot-expenses";

let expenses = loadExpenses();

const form = document.getElementById("expenseForm");
const titleInput = document.getElementById("expenseTitle");
const amountInput = document.getElementById("expenseAmount");
const categoryInput = document.getElementById("expenseCategory");
const expenseList = document.getElementById("expenseList");
const totalAmount = document.getElementById("totalAmount");
const transactionCount = document.getElementById("transactionCount");
const clearExpenses = document.getElementById("clearExpenses");

function loadExpenses() {
    try {
        const saved = localStorage.getItem(STORAGE_KEY);
        return saved ? JSON.parse(saved) : [];
    } catch {
        return [];
    }
}

function saveExpenses() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(expenses));
}

function addExpense(event) {
    event.preventDefault();

    const title = titleInput.value.trim();
    const amount = Number(amountInput.value);
    const category = categoryInput.value;

    if (!title || !Number.isFinite(amount) || amount <= 0) {
        return;
    }

    expenses.unshift({
        id: crypto.randomUUID(),
        title,
        amount,
        category,
        date: new Date().toLocaleDateString()
    });

    saveExpenses();
    form.reset();
    render();
}

function deleteExpense(id) {
    expenses = expenses.filter(expense => expense.id !== id);
    saveExpenses();
    render();
}

function render() {
    expenseList.innerHTML = "";

    if (!expenses.length) {
        expenseList.innerHTML = `
            <div class="empty">
                No expenses recorded yet.
            </div>
        `;
    }

    expenses.forEach(expense => {
        const article = document.createElement("article");
        article.className = "expense";

        article.innerHTML = `
            <div>
                <div class="expense-title"></div>
                <div class="expense-meta"></div>
            </div>

            <div class="expense-right">
                <div class="amount"></div>
                <button class="delete-expense" type="button">
                    Delete
                </button>
            </div>
        `;

        article.querySelector(".expense-title").textContent =
            expense.title;

        article.querySelector(".expense-meta").textContent =
            `${expense.category} • ${expense.date}`;

        article.querySelector(".amount").textContent =
            `₹${expense.amount.toFixed(2)}`;

        article.querySelector(".delete-expense")
            .addEventListener(
                "click",
                () => deleteExpense(expense.id)
            );

        expenseList.appendChild(article);
    });

    const total = expenses.reduce(
        (sum, expense) => sum + expense.amount,
        0
    );

    totalAmount.textContent = `₹${total.toFixed(2)}`;
    transactionCount.textContent = String(expenses.length);
}

form.addEventListener("submit", addExpense);

clearExpenses.addEventListener("click", () => {
    expenses = [];
    saveExpenses();
    render();
});

render();
"""


def local_expense_plan() -> Plan:
    return Plan(
        name="Expense Tracker",
        description="A responsive personal finance application for recording expenses.",
        techstack="HTML, CSS, Vanilla JavaScript",
        features=[
            "Add expenses",
            "Categorize transactions",
            "Calculate total spending",
            "Display transaction count",
            "Delete expenses",
            "Clear expenses",
            "Persist data with localStorage",
            "Responsive interface",
        ],
        files=[
            File(path="index.html", purpose="Expense tracker interface."),
            File(path="styles.css", purpose="Responsive styling."),
            File(path="app.js", purpose="Expense CRUD and persistence."),
        ],
    )


def local_expense_task_plan() -> TaskPlan:
    return TaskPlan(
        implementation_steps=[
            ImplementationTask(
                filepath="index.html",
                task_description="Create the expense tracker interface.",
            ),
            ImplementationTask(
                filepath="styles.css",
                task_description="Create responsive expense tracker styling.",
            ),
            ImplementationTask(
                filepath="app.js",
                task_description="Implement expense CRUD, totals and localStorage.",
            ),
        ]
    )


def validate_expense_project() -> ReviewResult:
    required = ["index.html", "styles.css", "app.js"]

    missing = [
        file for file in required
        if not (PROJECT_ROOT / file).exists()
    ]

    if missing:
        return ReviewResult(
            approved=False,
            issues=[f"Missing file: {file}" for file in missing],
            summary="Expense validation failed.",
        )

    html = read_project_file("index.html")
    css = read_project_file("styles.css")
    js = read_project_file("app.js")

    issues = []

    if "expenseForm" not in html:
        issues.append("Expense form is missing.")

    if "localStorage" not in js:
        issues.append("localStorage persistence is missing.")

    if "reduce" not in js:
        issues.append("Total calculation is missing.")

    if "@media" not in css:
        issues.append("Responsive styling is missing.")

    return ReviewResult(
        approved=not issues,
        issues=issues,
        summary=(
            "Local Expense Tracker review passed."
            if not issues
            else "Local Expense Tracker review found issues."
        ),
    )


def generate_expense_file(filepath: str) -> str:
    return {
        "index.html": EXPENSE_INDEX_HTML,
        "styles.css": EXPENSE_STYLES_CSS,
        "app.js": EXPENSE_APP_JS,
    }[filepath]


# ============================================================
# WEATHER DASHBOARD
# ============================================================

WEATHER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Modern weather dashboard">
    <title>Weather Dashboard</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <main class="app">
        <section class="hero">
            <p class="eyebrow">LIVE WEATHER</p>
            <h1>Weather Dashboard</h1>
            <p>Search a city to see its current weather conditions.</p>
        </section>

        <section class="search-panel">
            <form id="weatherForm">
                <label class="sr-only" for="cityInput">City</label>
                <input
                    id="cityInput"
                    type="text"
                    placeholder="Enter a city, e.g. Kolkata"
                    autocomplete="off"
                    required
                >
                <button id="searchButton" type="submit">
                    Search
                </button>
            </form>
        </section>

        <section id="status" class="status" role="status"
            aria-live="polite"></section>

        <section id="weatherCard" class="weather-card hidden">
            <div class="weather-header">
                <div>
                    <span class="label">CURRENT WEATHER</span>
                    <h2 id="locationName">—</h2>
                </div>

                <div id="conditionIcon" class="weather-icon">☀</div>
            </div>

            <div class="temperature-row">
                <strong id="temperature">—</strong>
                <span>°C</span>
            </div>

            <p id="condition" class="condition">—</p>

            <div class="metrics">
                <article>
                    <span>Humidity</span>
                    <strong id="humidity">—</strong>
                </article>

                <article>
                    <span>Wind speed</span>
                    <strong id="windSpeed">—</strong>
                </article>
            </div>
        </section>
    </main>

    <script src="app.js"></script>
</body>
</html>
"""

WEATHER_CSS = """* {
    box-sizing: border-box;
}

:root {
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #172033;
    background: #eef3f9;
}

body {
    margin: 0;
    min-height: 100vh;
    background:
        radial-gradient(circle at top right, #dce8ff 0, transparent 36%),
        linear-gradient(135deg, #f8fbff, #eef3f9);
}

button,
input {
    font: inherit;
}

button {
    cursor: pointer;
}

.app {
    width: min(760px, calc(100% - 32px));
    margin: 70px auto;
}

.hero {
    margin-bottom: 28px;
}

.eyebrow {
    margin: 0 0 8px;
    color: #4f64d8;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .16em;
}

h1 {
    margin: 0;
    font-size: clamp(38px, 8vw, 64px);
    line-height: 1;
}

.hero p:last-child {
    color: #70798a;
}

.search-panel,
.weather-card {
    background: rgba(255, 255, 255, .9);
    border: 1px solid #e0e6ef;
    border-radius: 22px;
    box-shadow: 0 20px 55px rgba(31, 48, 78, .1);
}

.search-panel {
    padding: 10px;
}

#weatherForm {
    display: flex;
    gap: 10px;
}

#cityInput {
    flex: 1;
    min-width: 0;
    border: 0;
    outline: 0;
    padding: 14px;
    background: transparent;
    font-size: 16px;
}

#searchButton {
    border: 0;
    border-radius: 14px;
    padding: 13px 20px;
    color: white;
    background: #5266dc;
    font-weight: 800;
}

.status {
    min-height: 28px;
    padding: 16px 4px 8px;
    color: #687386;
}

.weather-card {
    padding: 26px;
    margin-top: 8px;
}

.hidden {
    display: none;
}

.weather-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.label {
    color: #7d8799;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .14em;
}

h2 {
    margin: 6px 0 0;
    font-size: 30px;
}

.weather-icon {
    font-size: 48px;
}

.temperature-row {
    display: flex;
    align-items: flex-start;
    margin-top: 30px;
}

.temperature-row strong {
    font-size: clamp(72px, 16vw, 120px);
    line-height: .85;
    letter-spacing: -.06em;
}

.temperature-row span {
    margin-top: 4px;
    font-size: 30px;
    color: #7b8495;
}

.condition {
    margin-top: 18px;
    color: #596476;
    font-size: 18px;
}

.metrics {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 28px;
}

.metrics article {
    padding: 16px;
    border-radius: 15px;
    background: #f3f6fb;
}

.metrics span {
    display: block;
    color: #7a8496;
    font-size: 13px;
}

.metrics strong {
    display: block;
    margin-top: 6px;
    font-size: 20px;
}

.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
}

@media (max-width: 600px) {
    .app {
        margin: 35px auto;
    }

    #weatherForm {
        flex-direction: column;
    }

    #searchButton {
        width: 100%;
    }

    .weather-card {
        padding: 20px;
    }
}
"""

WEATHER_JS = """const form = document.getElementById("weatherForm");
const cityInput = document.getElementById("cityInput");
const searchButton = document.getElementById("searchButton");

const statusElement = document.getElementById("status");
const weatherCard = document.getElementById("weatherCard");

const locationName = document.getElementById("locationName");
const temperature = document.getElementById("temperature");
const humidity = document.getElementById("humidity");
const windSpeed = document.getElementById("windSpeed");
const condition = document.getElementById("condition");
const conditionIcon = document.getElementById("conditionIcon");

function setStatus(message) {
    statusElement.textContent = message;
}

function weatherDescription(code) {
    const descriptions = {
        0: ["Clear sky", "☀"],
        1: ["Mainly clear", "🌤"],
        2: ["Partly cloudy", "⛅"],
        3: ["Overcast", "☁"],
        45: ["Fog", "🌫"],
        48: ["Depositing rime fog", "🌫"],
        51: ["Light drizzle", "🌦"],
        53: ["Moderate drizzle", "🌦"],
        55: ["Dense drizzle", "🌧"],
        61: ["Light rain", "🌦"],
        63: ["Moderate rain", "🌧"],
        65: ["Heavy rain", "🌧"],
        71: ["Light snow", "🌨"],
        73: ["Moderate snow", "🌨"],
        75: ["Heavy snow", "❄"],
        80: ["Rain showers", "🌦"],
        81: ["Rain showers", "🌧"],
        82: ["Heavy rain showers", "⛈"],
        95: ["Thunderstorm", "⛈"],
        96: ["Thunderstorm with hail", "⛈"],
        99: ["Thunderstorm with hail", "⛈"]
    };

    return descriptions[code] || ["Unknown conditions", "🌡"];
}

async function geocodeCity(city) {
    const url =
        "https://geocoding-api.open-meteo.com/v1/search" +
        `?name=${encodeURIComponent(city)}` +
        "&count=1&language=en&format=json";

    const response = await fetch(url);

    if (!response.ok) {
        throw new Error("Could not find the city.");
    }

    const data = await response.json();

    if (!data.results || !data.results.length) {
        throw new Error("City not found. Try another city.");
    }

    return data.results[0];
}

async function fetchWeather(latitude, longitude) {
    const url =
        "https://api.open-meteo.com/v1/forecast" +
        `?latitude=${latitude}` +
        `&longitude=${longitude}` +
        "&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m" +
        "&timezone=auto";

    const response = await fetch(url);

    if (!response.ok) {
        throw new Error("Weather service is unavailable.");
    }

    return response.json();
}

async function searchWeather(city) {
    setStatus("Searching for weather...");
    weatherCard.classList.add("hidden");
    searchButton.disabled = true;

    try {
        const location = await geocodeCity(city);

        const weather = await fetchWeather(
            location.latitude,
            location.longitude
        );

        const current = weather.current;

        const [description, icon] =
            weatherDescription(current.weather_code);

        locationName.textContent =
            `${location.name}, ${location.country}`;

        temperature.textContent =
            Math.round(current.temperature_2m);

        humidity.textContent =
            `${current.relative_humidity_2m}%`;

        windSpeed.textContent =
            `${Math.round(current.wind_speed_10m)} km/h`;

        condition.textContent = description;
        conditionIcon.textContent = icon;

        weatherCard.classList.remove("hidden");
        setStatus("Weather updated successfully.");
    } catch (error) {
        setStatus(
            error.message || "Something went wrong."
        );
    } finally {
        searchButton.disabled = false;
    }
}

form.addEventListener("submit", event => {
    event.preventDefault();

    const city = cityInput.value.trim();

    if (!city) {
        setStatus("Enter a city name.");
        cityInput.focus();
        return;
    }

    searchWeather(city);
});
"""


# ============================================================
# GENERIC EXPENSE ANALYTICS
# ============================================================

EXPENSE_ANALYTICS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Expense analytics dashboard">
    <title>Expense Analytics Dashboard</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <main class="app">
        <header class="hero">
            <p class="eyebrow">PERSONAL FINANCE</p>
            <h1>Expense Analytics</h1>
            <p>Track spending, compare categories and understand your expenses.</p>
        </header>

        <section class="stats">
            <article class="stat-card">
                <span>Total expenses</span>
                <strong id="totalExpense">₹0.00</strong>
            </article>

            <article class="stat-card">
                <span>Average expense</span>
                <strong id="averageExpense">₹0.00</strong>
            </article>

            <article class="stat-card">
                <span>Highest expense</span>
                <strong id="highestExpense">₹0.00</strong>
            </article>
        </section>

        <section class="panel">
            <h2>Add expense</h2>

            <form id="expenseForm">
                <input
                    id="description"
                    type="text"
                    placeholder="Description"
                    required
                >

                <input
                    id="amount"
                    type="number"
                    min="0.01"
                    step="0.01"
                    placeholder="Amount"
                    required
                >

                <select id="category">
                    <option value="Food">Food</option>
                    <option value="Transport">Transport</option>
                    <option value="Shopping">Shopping</option>
                    <option value="Bills">Bills</option>
                    <option value="Entertainment">Entertainment</option>
                    <option value="Other">Other</option>
                </select>

                <input
                    id="date"
                    type="date"
                    required
                >

                <button type="submit">Add Expense</button>
            </form>
        </section>

        <section class="panel">
            <div class="panel-header">
                <div>
                    <h2>Expenses</h2>
                    <p id="emptyState">Add an expense to start tracking.</p>
                </div>

                <select id="categoryFilter">
                    <option value="all">All categories</option>
                    <option value="Food">Food</option>
                    <option value="Transport">Transport</option>
                    <option value="Shopping">Shopping</option>
                    <option value="Bills">Bills</option>
                    <option value="Entertainment">Entertainment</option>
                    <option value="Other">Other</option>
                </select>
            </div>

            <div id="expenseList" class="expense-list"></div>
        </section>
    </main>

    <script src="app.js"></script>
</body>
</html>
"""

EXPENSE_ANALYTICS_CSS = """* {
    box-sizing: border-box;
}

:root {
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #172033;
    background: #f4f7fb;
}

body {
    margin: 0;
    min-height: 100vh;
    background:
        radial-gradient(circle at top right, #e7edff 0, transparent 34%),
        #f4f7fb;
}

button,
input,
select {
    font: inherit;
}

button {
    cursor: pointer;
}

.app {
    width: min(1050px, calc(100% - 32px));
    margin: 52px auto;
}

.hero {
    margin-bottom: 28px;
}

.eyebrow {
    margin: 0 0 8px;
    color: #5668d8;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .16em;
}

h1 {
    margin: 0;
    font-size: clamp(38px, 7vw, 64px);
    line-height: 1;
}

.hero p:last-child,
.panel-header p {
    color: #747e90;
}

.stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
}

.stat-card,
.panel {
    background: white;
    border: 1px solid #e2e7ef;
    border-radius: 20px;
    box-shadow: 0 14px 38px rgba(29, 43, 70, .07);
}

.stat-card {
    padding: 22px;
}

.stat-card span {
    color: #7a8495;
    font-size: 13px;
}

.stat-card strong {
    display: block;
    margin-top: 8px;
    font-size: 28px;
}

.panel {
    margin-top: 16px;
    padding: 22px;
}

.panel h2 {
    margin-top: 0;
}

form {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr 1fr auto;
    gap: 10px;
}

input,
select {
    min-width: 0;
    border: 1px solid #dfe4ed;
    border-radius: 11px;
    padding: 12px;
    outline: none;
    background: white;
}

input:focus,
select:focus {
    border-color: #6171dc;
}

form button {
    border: 0;
    border-radius: 11px;
    padding: 12px 18px;
    background: #5668d8;
    color: white;
    font-weight: 800;
}

.panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 15px;
}

.panel-header h2 {
    margin-bottom: 5px;
}

.panel-header p {
    margin-top: 0;
    font-size: 13px;
}

.expense-list {
    display: grid;
    gap: 10px;
    margin-top: 16px;
}

.expense {
    display: grid;
    grid-template-columns: 1fr auto auto;
    align-items: center;
    gap: 20px;
    padding: 15px;
    border: 1px solid #e7ebf2;
    border-radius: 13px;
}

.expense-description {
    font-weight: 700;
}

.expense-meta {
    margin-top: 4px;
    color: #7d8798;
    font-size: 13px;
}

.expense-amount {
    font-weight: 800;
}

.delete-expense {
    border: 0;
    background: transparent;
    color: #b04756;
}

.empty {
    padding: 28px;
    text-align: center;
    color: #858fa0;
    border: 1px dashed #dce1e9;
    border-radius: 13px;
}

@media (max-width: 900px) {
    form {
        grid-template-columns: 1fr 1fr;
    }

    form button {
        grid-column: span 2;
    }
}

@media (max-width: 650px) {
    .app {
        margin: 30px auto;
    }

    .stats {
        grid-template-columns: 1fr;
    }

    form {
        grid-template-columns: 1fr;
    }

    form button {
        grid-column: auto;
    }

    .panel-header {
        align-items: flex-start;
        flex-direction: column;
    }

    .expense {
        grid-template-columns: 1fr;
        gap: 8px;
    }
}
"""

EXPENSE_ANALYTICS_JS = """const STORAGE_KEY = "codepilot-expense-analytics";

let expenses = loadExpenses();

const form = document.getElementById("expenseForm");
const descriptionInput = document.getElementById("description");
const amountInput = document.getElementById("amount");
const categoryInput = document.getElementById("category");
const dateInput = document.getElementById("date");

const categoryFilter =
    document.getElementById("categoryFilter");

const expenseList =
    document.getElementById("expenseList");

const emptyState =
    document.getElementById("emptyState");

const totalExpense =
    document.getElementById("totalExpense");

const averageExpense =
    document.getElementById("averageExpense");

const highestExpense =
    document.getElementById("highestExpense");

dateInput.value = new Date().toISOString().slice(0, 10);

function loadExpenses() {
    try {
        const saved = localStorage.getItem(STORAGE_KEY);
        return saved ? JSON.parse(saved) : [];
    } catch {
        return [];
    }
}

function saveExpenses() {
    localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify(expenses)
    );
}

function addExpense(event) {
    event.preventDefault();

    const description = descriptionInput.value.trim();
    const amount = Number(amountInput.value);
    const category = categoryInput.value;
    const date = dateInput.value;

    if (
        !description ||
        !Number.isFinite(amount) ||
        amount <= 0 ||
        !date
    ) {
        return;
    }

    expenses.unshift({
        id: crypto.randomUUID(),
        description,
        amount,
        category,
        date
    });

    saveExpenses();

    form.reset();
    dateInput.value =
        new Date().toISOString().slice(0, 10);

    render();
}

function deleteExpense(id) {
    expenses = expenses.filter(
        expense => expense.id !== id
    );

    saveExpenses();
    render();
}

function filteredExpenses() {
    const selected = categoryFilter.value;

    if (selected === "all") {
        return expenses;
    }

    return expenses.filter(
        expense => expense.category === selected
    );
}

function updateStatistics() {
    const amounts = expenses.map(
        expense => Number(expense.amount)
    );

    const total = amounts.reduce(
        (sum, amount) => sum + amount,
        0
    );

    const average = amounts.length
        ? total / amounts.length
        : 0;

    const highest = amounts.length
        ? Math.max(...amounts)
        : 0;

    totalExpense.textContent =
        `₹${total.toFixed(2)}`;

    averageExpense.textContent =
        `₹${average.toFixed(2)}`;

    highestExpense.textContent =
        `₹${highest.toFixed(2)}`;
}

function render() {
    const visible = filteredExpenses();

    expenseList.innerHTML = "";

    if (!visible.length) {
        expenseList.innerHTML = `
            <div class="empty">
                No expenses match the selected category.
            </div>
        `;
    }

    visible.forEach(expense => {
        const article = document.createElement("article");

        article.className = "expense";

        article.innerHTML = `
            <div>
                <div class="expense-description"></div>
                <div class="expense-meta"></div>
            </div>

            <div class="expense-amount"></div>

            <button
                class="delete-expense"
                type="button"
            >
                Delete
            </button>
        `;

        article.querySelector(
            ".expense-description"
        ).textContent = expense.description;

        article.querySelector(
            ".expense-meta"
        ).textContent =
            `${expense.category} • ${expense.date}`;

        article.querySelector(
            ".expense-amount"
        ).textContent =
            `₹${Number(expense.amount).toFixed(2)}`;

        article.querySelector(
            ".delete-expense"
        ).addEventListener(
            "click",
            () => deleteExpense(expense.id)
        );

        expenseList.appendChild(article);
    });

    emptyState.textContent = expenses.length
        ? `${expenses.length} expense${expenses.length === 1 ? "" : "s"} recorded`
        : "Add an expense to start tracking.";

    updateStatistics();
}

form.addEventListener("submit", addExpense);

categoryFilter.addEventListener(
    "change",
    render
);

render();
"""


# ============================================================
# GENERIC LOCAL PLANNING
# ============================================================

def extract_project_name(prompt: str) -> str:
    text = prompt.strip()

    patterns = [
        r"(?:build|create|develop|make|design)\s+(?:a|an)\s+(.+?)(?:\n|\.|$)",
        r"(?:build|create|develop|make|design)\s+(.+?)(?:\n|\.|$)",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            candidate = match.group(1).strip()

            candidate = re.sub(
                r"\busing\b.*$",
                "",
                candidate,
                flags=re.IGNORECASE,
            )

            candidate = re.sub(
                r"\bwith\b.*$",
                "",
                candidate,
                flags=re.IGNORECASE,
            )

            candidate = candidate.strip(" .:-")

            if candidate:
                return candidate.title()

    if is_weather_project(prompt):
        return "Weather Dashboard"

    if is_expense_analytics_project(prompt):
        return "Expense Analytics Dashboard"

    return "Local Web Application"


def local_generic_features(prompt: str) -> list[str]:
    text = prompt.lower()
    features: list[str] = []

    if is_weather_project(prompt):
        features.extend([
            "Search weather by city",
            "Display temperature",
            "Display humidity",
            "Display wind speed",
            "Display weather condition",
            "Loading state",
            "Error state",
            "Responsive interface",
        ])

    if is_expense_analytics_project(prompt):
        features.extend([
            "Add expenses",
            "Calculate total expenses",
            "Calculate average expense",
            "Show highest expense",
            "Filter expenses by category",
            "Persist expenses with localStorage",
            "Responsive dashboard",
        ])

    if "search" in text:
        features.append("Search functionality")

    if "responsive" in text:
        features.append("Responsive design")

    if "loading" in text:
        features.append("Loading state")

    if "error" in text:
        features.append("Error handling")

    if "localstorage" in text:
        features.append("Local browser persistence")

    if not features:
        for line in prompt.splitlines():
            line = line.strip("- •\t ")

            if not line:
                continue

            if line.lower().startswith(
                (
                    "build ",
                    "create ",
                    "develop ",
                    "make ",
                    "requirements:",
                    "use ",
                    "no ",
                )
            ):
                continue

            if len(line) > 4:
                features.append(line.rstrip("."))

    if not features:
        features = [
            "Clean responsive interface",
            "Interactive user experience",
            "Client-side application logic",
            "Error-safe interactions",
        ]

    return list(dict.fromkeys(features))


def local_generic_plan(prompt: str) -> Plan:
    name = extract_project_name(prompt)

    return Plan(
        name=name,
        description=f"{name} generated by CodePilot-Ai local mode.",
        techstack="HTML, CSS, Vanilla JavaScript",
        features=local_generic_features(prompt),
        files=[
            File(
                path="index.html",
                purpose="Semantic application structure.",
            ),
            File(
                path="styles.css",
                purpose="Responsive visual styling.",
            ),
            File(
                path="app.js",
                purpose="Client-side application functionality.",
            ),
        ],
    )


def local_generic_task_plan(state: CoderState) -> TaskPlan:
    return TaskPlan(
        implementation_steps=[
            ImplementationTask(
                filepath="index.html",
                task_description=(
                    "Create the complete semantic UI required "
                    "by the project."
                ),
            ),
            ImplementationTask(
                filepath="styles.css",
                task_description=(
                    "Create polished responsive styling "
                    "for the complete application."
                ),
            ),
            ImplementationTask(
                filepath="app.js",
                task_description=(
                    "Implement all requested interactions "
                    "and application functionality."
                ),
            ),
        ]
    )


# ============================================================
# LOCAL RAG
# ============================================================

def create_local_rag_context(state: CoderState) -> str:
    plan = state.plan

    if plan is None:
        return ""

    features = "\n".join(
        f"- {feature}"
        for feature in plan.features
    )

    return f"""# CodePilot-Ai Local Project Context

Project:
{plan.name}

Description:
{plan.description}

Technology:
{plan.techstack}

Required features:
{features}

Required files:
- index.html
- styles.css
- app.js

Original user request:
{state.user_prompt}

Implementation rules:
- Use HTML, CSS and vanilla JavaScript.
- Do not use React.
- Do not use Bootstrap.
- Do not use Tailwind.
- Do not use jQuery.
- Do not create placeholder implementations.
- Do not add TODO implementation comments.
"""


def retrieve_local_rag_context(state: CoderState) -> str:
    if build_project_rag is None:
        return ""

    try:
        context_file = PROJECT_ROOT / ".codepilot_context.md"

        context_file.write_text(
            create_local_rag_context(state),
            encoding="utf-8",
        )

        rag = build_project_rag(PROJECT_ROOT)

        if rag is None:
            return ""

        query = (
            f"{state.plan.name if state.plan else 'project'} "
            f"requirements architecture implementation "
            f"{state.user_prompt}"
        )

        results = []

        if hasattr(rag, "search"):
            try:
                results = rag.search(
                    query,
                    top_k=5,
                )
            except TypeError:
                results = rag.search(query)

        if not results:
            return ""

        parts = []

        for result in results:
            if isinstance(result, str):
                parts.append(result)
            elif hasattr(result, "content"):
                parts.append(str(result.content))
            elif isinstance(result, dict):
                parts.append(
                    str(
                        result.get(
                            "content",
                            result.get("text", result),
                        )
                    )
                )
            else:
                parts.append(str(result))

        return "\n\n".join(parts)

    except Exception as exc:
        print(f"[RAG] Local retrieval skipped: {exc}")
        return ""


# ============================================================
# GENERIC FILE GENERATION
# ============================================================

def generic_fallback_html(plan: Plan) -> str:
    features = "\n".join(
        f"<li>{feature}</li>"
        for feature in plan.features
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="{plan.description}">
    <title>{plan.name}</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <main class="app">
        <header>
            <span class="eyebrow">CODEPILOT-AI</span>
            <h1>{plan.name}</h1>
            <p>{plan.description}</p>
        </header>

        <section class="panel">
            <h2>Project features</h2>

            <ul id="featureList">
                {features}
            </ul>
        </section>

        <section class="panel">
            <h2>Interactive preview</h2>

            <p id="status">
                Application initialized successfully.
            </p>

            <button id="actionButton" type="button">
                Test interaction
            </button>
        </section>
    </main>

    <script src="app.js"></script>
</body>
</html>
"""


def generic_fallback_css() -> str:
    return """* {
    box-sizing: border-box;
}

:root {
    font-family: Inter, system-ui, sans-serif;
    color: #172033;
    background: #f4f7fb;
}

body {
    margin: 0;
    min-height: 100vh;
    background:
        radial-gradient(circle at top left, #e4ecff, transparent 35%),
        #f4f7fb;
}

.app {
    width: min(850px, calc(100% - 32px));
    margin: 70px auto;
}

header {
    margin-bottom: 28px;
}

.eyebrow {
    color: #5868d9;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .16em;
}

h1 {
    margin: 8px 0;
    font-size: clamp(40px, 8vw, 70px);
}

header p {
    color: #6f798b;
}

.panel {
    margin-top: 16px;
    padding: 24px;
    background: white;
    border: 1px solid #e3e8f0;
    border-radius: 20px;
    box-shadow: 0 18px 45px rgba(31, 44, 72, .08);
}

li {
    margin: 10px 0;
}

button {
    border: 0;
    border-radius: 10px;
    padding: 12px 18px;
    color: white;
    background: #5868d9;
    font-weight: 700;
    cursor: pointer;
}

@media (max-width: 600px) {
    .app {
        margin: 35px auto;
    }

    .panel {
        padding: 18px;
    }
}
"""


def generic_fallback_js() -> str:
    return """const button = document.getElementById("actionButton");
const status = document.getElementById("status");

function runInteraction() {
    status.textContent =
        "Interactive functionality is working correctly.";
}

button.addEventListener(
    "click",
    runInteraction
);
"""


def generate_generic_local_file(
    state: CoderState,
    filepath: str,
) -> str:

    if is_weather_project(state.user_prompt):
        return {
            "index.html": WEATHER_HTML,
            "styles.css": WEATHER_CSS,
            "app.js": WEATHER_JS,
        }[filepath]

    if is_expense_analytics_project(state.user_prompt):
        return {
            "index.html": EXPENSE_ANALYTICS_HTML,
            "styles.css": EXPENSE_ANALYTICS_CSS,
            "app.js": EXPENSE_ANALYTICS_JS,
        }[filepath]

    plan = state.plan

    if plan is None:
        raise RuntimeError(
            "Generic local coder requires a project plan."
        )

    return {
        "index.html": generic_fallback_html(plan),
        "styles.css": generic_fallback_css(),
        "app.js": generic_fallback_js(),
    }[filepath]


# ============================================================
# LOCAL REVIEW VALIDATION
# ============================================================

def find_placeholder_content(
    filename: str,
    content: str,
) -> list[str]:

    lower = content.lower()

    phrases = [
        "lorem ipsum",
        "coming soon",
        "under construction",
        "replace this",
        "your code here",
    ]

    issues = []

    for phrase in phrases:
        if phrase in lower:
            issues.append(
                f"Placeholder content detected in "
                f"{filename}: {phrase}"
            )

    return issues


def validate_generic_local_project(
    state: CoderState,
) -> ReviewResult:

    required = [
        "index.html",
        "styles.css",
        "app.js",
    ]

    missing = [
        file
        for file in required
        if not (PROJECT_ROOT / file).exists()
    ]

    if missing:
        return ReviewResult(
            approved=False,
            issues=[
                f"Missing required file: {file}"
                for file in missing
            ],
            summary="Local generic review found issues.",
        )

    html = read_project_file("index.html")
    css = read_project_file("styles.css")
    js = read_project_file("app.js")

    issues: list[str] = []

    # Basic HTML validation.
    if "<html" not in html.lower():
        issues.append(
            "index.html does not contain an HTML document."
        )

    if "</html>" not in html.lower():
        issues.append(
            "index.html does not contain a closing html tag."
        )

    if '<script src="app.js"></script>' not in html:
        issues.append(
            "index.html is not connected to app.js."
        )

    if 'rel="stylesheet"' not in html:
        issues.append(
            "index.html is not connected to styles.css."
        )

    # CSS validation.
    if "{" not in css or "}" not in css:
        issues.append(
            "styles.css does not contain valid CSS rules."
        )

    if "@media" not in css:
        issues.append(
            "Responsive styling is missing."
        )

    # JavaScript validation.
    if "function" not in js and "=>" not in js:
        issues.append(
            "app.js does not contain executable application logic."
        )

    # User explicitly prohibited TODO comments.
    for filename, content in {
        "index.html": html,
        "styles.css": css,
        "app.js": js,
    }.items():
        if re.search(
            r"(?i)\bTODO\b",
            content,
        ):
            issues.append(
                f"TODO implementation comment detected in {filename}."
            )

        issues.extend(
            find_placeholder_content(
                filename,
                content,
            )
        )

    # Weather-specific validation.
    if is_weather_project(state.user_prompt):

        weather_requirements = [
            ("cityInput", html),
            ("weatherForm", html),
            ("temperature", html),
            ("humidity", html),
            ("windSpeed", html),
            ("condition", html),
            ("fetch(", js),
            ("open-meteo", js.lower()),
            ("relative_humidity_2m", js),
            ("wind_speed_10m", js),
        ]

        for token, content in weather_requirements:
            if token.lower() not in content.lower():
                issues.append(
                    f"Weather implementation is missing: {token}"
                )

    # Expense analytics validation.
    if is_expense_analytics_project(
        state.user_prompt
    ):

        expense_requirements = [
            ("expenseForm", html),
            ("totalExpense", html),
            ("averageExpense", html),
            ("highestExpense", html),
            ("categoryFilter", html),
            ("localStorage", js),
            ("reduce", js),
            ("filter", js),
            ("addExpense", js),
            ("deleteExpense", js),
        ]

        for token, content in expense_requirements:
            if token.lower() not in content.lower():
                issues.append(
                    f"Expense analytics implementation "
                    f"is missing: {token}"
                )

    return ReviewResult(
        approved=not issues,
        issues=issues,
        summary=(
            "Local generic review passed."
            if not issues
            else "Local generic review found issues."
        ),
    )


# ============================================================
# LOCAL FIX LOGIC
# ============================================================

def apply_local_fixes(
    state: CoderState,
    filepath: str,
    content: str,
) -> str:

    feedback = (
        state.review_feedback or ""
    ).lower()

    # Missing CSS responsiveness.
    if (
        "responsive styling is missing" in feedback
        and filepath == "styles.css"
        and "@media" not in content
    ):
        content += """

@media (max-width: 700px) {
    .app {
        width: min(100% - 24px, 850px);
        margin: 30px auto;
    }

    .panel {
        padding: 18px;
    }

    form {
        grid-template-columns: 1fr;
    }
}
"""

    # Missing app.js logic.
    if (
        "app.js does not contain executable application logic"
        in feedback
        and filepath == "app.js"
    ):
        content += """

function initializeApplication() {
    return true;
}

initializeApplication();
"""

    # Missing stylesheet connection.
    if (
        "not connected to styles.css" in feedback
        and filepath == "index.html"
        and 'rel="stylesheet"' not in content
    ):
        content = content.replace(
            "</head>",
            '    <link rel="stylesheet" href="styles.css">\\n'
            "</head>",
        )

    # Missing JavaScript connection.
    if (
        "not connected to app.js" in feedback
        and filepath == "index.html"
        and 'src="app.js"' not in content
    ):
        content = content.replace(
            "</body>",
            '    <script src="app.js"></script>\\n'
            "</body>",
        )

    return content


def local_file_needs_regeneration(
    state: CoderState,
    filepath: str,
) -> bool:

    if not state.review_fix_required:
        return False

    feedback = (
        state.review_feedback or ""
    ).lower()

    if not feedback.strip():
        return True

    # If the feedback mentions the current file,
    # regenerate/fix that file.
    filename = filepath.lower()

    file_tokens = {
        "index.html": [
            "index.html",
            "html",
            "interface",
            "input",
            "form",
        ],
        "styles.css": [
            "styles.css",
            "css",
            "responsive",
            "styling",
        ],
        "app.js": [
            "app.js",
            "javascript",
            "logic",
            "function",
            "localstorage",
            "implementation",
        ],
    }

    return any(
        token in feedback
        for token in file_tokens.get(
            filename,
            [filename],
        )
    )


# ============================================================
# PLANNER NODE
# ============================================================

def planner_node(state: CoderState) -> CoderState:

    prompt = state.user_prompt

    if is_todo_project(prompt):
        print(
            "[PLANNER] Using deterministic local Todo planner."
        )

        prepare_project_root_for_request(prompt)

        return update_state(
            state,
            plan=local_todo_plan(),
            task_plan=None,
            current_step_idx=0,
            current_file_content=None,
            last_validation_result=None,
            files_modified=[],
            review_attempts=0,
            review_feedback=None,
            review_fix_required=False,
        )

    if is_expense_project(prompt):
        print(
            "[PLANNER] Using deterministic local "
            "Expense planner."
        )

        prepare_project_root_for_request(prompt)

        return update_state(
            state,
            plan=local_expense_plan(),
            task_plan=None,
            current_step_idx=0,
            current_file_content=None,
            last_validation_result=None,
            files_modified=[],
            review_attempts=0,
            review_feedback=None,
            review_fix_required=False,
        )

    if LOCAL_TEST_MODE:
        print(
            "[PLANNER] Using LOCAL generic planner."
        )

        print(
            "[MODE] Generic project -> "
            "LOCAL / ZERO API MODE"
        )

        prepare_project_root_for_request(prompt)

        plan = local_generic_plan(prompt)

        return update_state(
            state,
            plan=plan,
            task_plan=None,
            current_step_idx=0,
            current_file_content=None,
            last_validation_result=None,
            files_modified=[],
            review_attempts=0,
            review_feedback=None,
            review_fix_required=False,
        )

    print("[PLANNER] Using LLM planner.")

    planner_prompt = f"""
You are the planner for an autonomous software engineering agent.

Create a high-level implementation plan for:

{prompt}

Return ONLY valid JSON:

{{
  "name": "application name",
  "description": "one sentence",
  "techstack": "technology stack",
  "features": ["feature 1"],
  "files": [
    {{
      "path": "index.html",
      "purpose": "purpose"
    }}
  ]
}}
"""

    raw = invoke_llm(planner_prompt)

    plan = Plan.model_validate(
        extract_json(raw)
    )

    return update_state(
        state,
        plan=plan,
        task_plan=None,
        current_step_idx=0,
        current_file_content=None,
        last_validation_result=None,
        files_modified=[],
        review_attempts=0,
        review_feedback=None,
        review_fix_required=False,
    )


# ============================================================
# ARCHITECT NODE
# ============================================================

def architect_node(state: CoderState) -> CoderState:

    if state.plan is None:
        raise RuntimeError(
            "Architect requires a project plan."
        )

    if is_todo_project(state.user_prompt):
        print(
            "[ARCHITECT] Using deterministic local "
            "Todo architecture."
        )

        return update_state(
            state,
            task_plan=local_todo_task_plan(),
            current_step_idx=0,
        )

    if is_expense_project(state.user_prompt):
        print(
            "[ARCHITECT] Using deterministic local "
            "Expense architecture."
        )

        return update_state(
            state,
            task_plan=local_expense_task_plan(),
            current_step_idx=0,
        )

    if LOCAL_TEST_MODE:
        print(
            "[ARCHITECT] Using LOCAL generic architecture."
        )

        return update_state(
            state,
            task_plan=local_generic_task_plan(state),
            current_step_idx=0,
        )

    print(
        "[ARCHITECT] Creating implementation plan with LLM."
    )

    files_json = json.dumps(
        [
            file.model_dump()
            for file in state.plan.files
        ],
        indent=2,
    )

    prompt = f"""
Create implementation steps for:

Project:
{state.plan.name}

Description:
{state.plan.description}

Technology:
{state.plan.techstack}

Features:
{json.dumps(state.plan.features, indent=2)}

Files:
{files_json}

Original request:
{state.user_prompt}

Return ONLY valid JSON:

{{
  "implementation_steps": [
    {{
      "filepath": "index.html",
      "task_description": "detailed instructions"
    }}
  ]
}}
"""

    raw = invoke_llm(prompt)
    data = extract_json(raw)

    if isinstance(data, dict) and "implementation_steps" in data:
        task_plan = TaskPlan.model_validate(data)

    elif isinstance(data, list):
        task_plan = TaskPlan(
            implementation_steps=[
                ImplementationTask.model_validate(item)
                for item in data
            ]
        )

    elif isinstance(data, dict):
        task_plan = TaskPlan(
            implementation_steps=[
                ImplementationTask.model_validate(data)
            ]
        )

    else:
        raise ValueError(
            "Invalid architect response."
        )

    return update_state(
        state,
        task_plan=task_plan,
        current_step_idx=0,
    )


# ============================================================
# LOCAL TODO CODER
# ============================================================

def local_todo_coder_node(
    state: CoderState,
) -> CoderState:

    if state.task_plan is None:
        raise RuntimeError(
            "Todo coder requires a task plan."
        )

    steps = state.task_plan.implementation_steps

    if state.current_step_idx >= len(steps):
        return state

    step = steps[state.current_step_idx]

    filepath = step.filepath

    content = generate_todo_file(filepath)

    write_project_file(
        filepath,
        content,
    )

    modified = list(state.files_modified)

    if filepath not in modified:
        modified.append(filepath)

    print(
        f"[CODER] Local Todo -> {filepath}"
    )

    return update_state(
        state,
        current_file_content=content,
        files_modified=modified,
        current_step_idx=(
            state.current_step_idx + 1
        ),
        last_validation_result=(
            f"Generated {filepath}"
        ),
    )


# ============================================================
# LOCAL EXPENSE CODER
# ============================================================

def local_expense_coder_node(
    state: CoderState,
) -> CoderState:

    if state.task_plan is None:
        raise RuntimeError(
            "Expense coder requires a task plan."
        )

    steps = state.task_plan.implementation_steps

    if state.current_step_idx >= len(steps):
        return state

    step = steps[state.current_step_idx]

    filepath = step.filepath

    content = generate_expense_file(filepath)

    write_project_file(
        filepath,
        content,
    )

    modified = list(state.files_modified)

    if filepath not in modified:
        modified.append(filepath)

    print(
        f"[CODER] Local Expense -> {filepath}"
    )

    return update_state(
        state,
        current_file_content=content,
        files_modified=modified,
        current_step_idx=(
            state.current_step_idx + 1
        ),
        last_validation_result=(
            f"Generated {filepath}"
        ),
    )


# ============================================================
# LOCAL GENERIC CODER
# ============================================================
def is_git_modification_request(
    state: CoderState,
) -> bool:
    """
    Determine whether the user is asking to modify an
    existing project rather than create a completely new one.
    """

    prompt = (
        state.user_prompt
        or ""
    ).lower()

    modification_keywords = [
        "add ",
        "add a ",
        "add an ",
        "update ",
        "change ",
        "modify ",
        "improve ",
        "enhance ",
        "extend ",
        "remove ",
        "delete ",
        "replace ",
        "fix ",
        "make it ",
        "make the ",
        "include ",
        "integrate ",
        "implement ",
        "introduce ",
        "support ",
        "upgrade ",
        "existing ",
        "current ",
        "previous ",
        "my weather",
        "my dashboard",
        "my app",
        "my project",
    ]

    has_modification_language = any(
        keyword in prompt
        for keyword in modification_keywords
    )

    if not has_modification_language:
        return False

    # A previous Git commit is strong evidence that the project
    # already exists.
    if state.git_latest_commit:
        return True

    # Existing uncommitted files are also strong evidence.
    if state.git_changed_files:
        return True

    return False


def apply_git_aware_request(
    state: CoderState,
    filepath: str,
    content: str,
) -> str:
    """
    Apply deterministic, targeted modifications to an existing
    project.

    The existing file is preserved and only relevant changes
    are inserted.

    This is intentionally conservative: if the local agent
    does not recognize a requested modification, it leaves
    the existing file unchanged rather than destroying it.
    """

    prompt = (
        state.user_prompt
        or ""
    ).lower()

    # ========================================================
    # WEATHER DASHBOARD
    # ========================================================

    if is_weather_project(
        state.user_prompt
    ):

        # ----------------------------------------------------
        # 5-DAY FORECAST
        # ----------------------------------------------------

        forecast_requested = any(
            phrase in prompt
            for phrase in [
                "5-day forecast",
                "5 day forecast",
                "five-day forecast",
                "five day forecast",
                "forecast for 5 days",
                "forecast for five days",
            ]
        )

        if forecast_requested:

            # ------------------------------------------------
            # index.html
            # ------------------------------------------------

            if filepath == "index.html":

                if "forecast" not in content.lower():

                    forecast_html = """
<section class="forecast-panel">
    <h2>5-Day Forecast</h2>

    <div
        id="forecast"
        class="forecast-grid"
        aria-live="polite"
    ></div>
</section>
"""

                    if "</main>" in content:

                        content = content.replace(
                            "</main>",
                            forecast_html
                            + "\n</main>",
                            1,
                        )

                    elif "</body>" in content:

                        content = content.replace(
                            "</body>",
                            forecast_html
                            + '\n<script src="app.js"></script>\n'
                            "</body>",
                            1,
                        )

            # ------------------------------------------------
            # styles.css
            # ------------------------------------------------

            elif filepath == "styles.css":

                if ".forecast-grid" not in content:

                    forecast_css = """

/* 5-day forecast */

.forecast-panel {
    margin-top: 24px;
}

.forecast-grid {
    display: grid;
    grid-template-columns: repeat(
        5,
        minmax(0, 1fr)
    );
    gap: 12px;
}

.forecast-card {
    padding: 16px;
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.12);
    text-align: center;
}

.forecast-card h3 {
    margin-bottom: 8px;
    font-size: 0.95rem;
}

.forecast-temperature {
    font-size: 1.5rem;
    font-weight: 700;
}

.forecast-condition {
    margin-top: 6px;
    opacity: 0.8;
}

@media (max-width: 900px) {
    .forecast-grid {
        grid-template-columns: repeat(
            3,
            minmax(0, 1fr)
        );
    }
}

@media (max-width: 600px) {
    .forecast-grid {
        grid-template-columns: repeat(
            2,
            minmax(0, 1fr)
        );
    }
}
"""

                    content += forecast_css

            # ------------------------------------------------
            # app.js
            # ------------------------------------------------

            elif filepath == "app.js":

                if "renderforecast" not in content.lower():

                    forecast_js = """

// ========================================================
// 5-DAY FORECAST
// ========================================================

function renderForecast(forecastData) {
    const forecastElement =
        document.getElementById("forecast");

    if (!forecastElement) {
        return;
    }

    forecastElement.innerHTML = "";

    forecastData
        .slice(0, 5)
        .forEach((day) => {
            const card =
                document.createElement("article");

            card.className =
                "forecast-card";

            card.innerHTML = `
                <h3>${day.date || "Forecast"}</h3>
                <div class="forecast-temperature">
                    ${day.temperature ?? "--"}°
                </div>
                <div class="forecast-condition">
                    ${day.condition || "Unknown"}
                </div>
            `;

            forecastElement.appendChild(card);
        });
}
"""

                    content += forecast_js

        # ----------------------------------------------------
        # LOADING STATE
        # ----------------------------------------------------

        loading_requested = any(
            phrase in prompt
            for phrase in [
                "loading state",
                "loading indicator",
                "show loading",
                "loading screen",
            ]
        )

        if loading_requested:

            if filepath == "index.html":

                if (
                    'id="loading"' not in content
                    and
                    "id='loading'" not in content
                ):

                    loading_html = """
<div
    id="loading"
    class="loading"
    hidden
>
    Loading weather data...
</div>
"""

                    if "</main>" in content:

                        content = content.replace(
                            "</main>",
                            loading_html
                            + "\n</main>",
                            1,
                        )

            elif filepath == "styles.css":

                if ".loading" not in content:

                    content += """

.loading {
    margin: 16px 0;
    text-align: center;
    font-weight: 600;
}

.loading[hidden] {
    display: none;
}
"""

    # ========================================================
    # GENERIC REQUEST SAFETY
    # ========================================================

    # Never return an empty file.
    if not content.strip():

        return generate_generic_local_file(
            state,
            filepath,
        )

    return content

def local_generic_coder_node(
    state: CoderState,
) -> CoderState:

    if state.task_plan is None:
        raise RuntimeError(
            "Generic local coder requires a task plan."
        )

    steps = state.task_plan.implementation_steps

    if not steps:
        return state

    # --------------------------------------------------------
    # INITIAL GENERATION / GIT-AWARE MODIFICATION
    # --------------------------------------------------------

    if not state.review_fix_required:

        # ----------------------------------------------------
        # RAG
        # ----------------------------------------------------

        if state.current_step_idx == 0:

            print(
                "[RAG] Retrieving local project context..."
            )

            rag_context = retrieve_local_rag_context(
                state
            )

            if rag_context:
                print(
                    "[RAG] Local context retrieved."
                )
            else:
                print(
                    "[RAG] No additional context retrieved."
                )

            # ------------------------------------------------
            # GIT-AWARE CODING
            # ------------------------------------------------

            if state.git_latest_commit:

                print()
                print(
                    "[GIT CODER] Existing Git project detected."
                )

                print(
                    "[GIT CODER] Latest commit: "
                    f"{state.git_latest_commit}"
                )

                if state.git_changed_files:

                    print(
                        "[GIT CODER] Existing uncommitted "
                        "changes detected:"
                    )

                    for filepath in (
                        state.git_changed_files
                    ):
                        print(
                            f"  - {filepath}"
                        )

                else:

                    print(
                        "[GIT CODER] Working tree is clean."
                    )

                print(
                    "[GIT CODER] Existing project will be "
                    "modified instead of blindly regenerated."
                )

        if state.current_step_idx >= len(steps):
            return state

        step = steps[
            state.current_step_idx
        ]

        filepath = step.filepath

        # ----------------------------------------------------
        # DETERMINE WHETHER THIS IS AN EXISTING PROJECT CHANGE
        # ----------------------------------------------------

        modification_request = is_git_modification_request(
            state
        )

        if modification_request:

            current = read_project_file(
                filepath
            )

            if current:

                print(
                    f"[GIT CODER] Reading existing -> "
                    f"{filepath}"
                )

                content = apply_git_aware_request(
                    state,
                    filepath,
                    current,
                )

            else:

                print(
                    f"[GIT CODER] File missing, generating -> "
                    f"{filepath}"
                )

                content = generate_generic_local_file(
                    state,
                    filepath,
                )

        else:

            content = generate_generic_local_file(
                state,
                filepath,
            )

        # ----------------------------------------------------
        # WRITE FILE
        # ----------------------------------------------------

        write_project_file(
            filepath,
            content,
        )

        modified = list(
            state.files_modified
        )

        if filepath not in modified:
            modified.append(
                filepath
            )

        if modification_request:

            print(
                f"[CODER] Git-aware modification -> "
                f"{filepath}"
            )

        else:

            print(
                f"[CODER] Local Generic -> "
                f"{filepath}"
            )

        is_last = (
            state.current_step_idx + 1
            >= len(steps)
        )

        if is_last:

            context_file = (
                PROJECT_ROOT
                / ".codepilot_context.md"
            )

            if context_file.exists():

                try:
                    context_file.unlink()

                except Exception:
                    pass

        return update_state(
            state,
            current_file_content=content,
            files_modified=modified,
            current_step_idx=(
                state.current_step_idx + 1
            ),
            last_validation_result=(
                f"Generated/modified {filepath}"
            ),
        )

    # --------------------------------------------------------
    # REVIEW FIX MODE
    # --------------------------------------------------------

    print(
        "[CODER] Local Generic -> applying reviewer fixes."
    )

    feedback = (
        state.review_feedback or ""
    )

    print(
        "[CODER] Reviewer feedback:"
    )

    for issue in feedback.splitlines():

        if issue.strip():

            print(
                f"          - {issue.strip()}"
            )

    # --------------------------------------------------------
    # RE-CHECK ALL REQUIRED FILES
    # --------------------------------------------------------

    for step in steps:

        filepath = step.filepath

        if not local_file_needs_regeneration(
            state,
            filepath,
        ):
            continue

        current = read_project_file(
            filepath
        )

        if not current:

            current = generate_generic_local_file(
                state,
                filepath,
            )

        fixed = apply_local_fixes(
            state,
            filepath,
            current,
        )

        # ----------------------------------------------------
        # DOMAIN-SPECIFIC FIXES
        # ----------------------------------------------------
        # IMPORTANT: never replace an existing domain project with
        # a generic template during reviewer recovery. Targeted
        # fixes operate on the current file contents only.

        if is_weather_project(state.user_prompt) or is_expense_analytics_project(state.user_prompt):
            fixed = apply_git_aware_request(
                state,
                filepath,
                fixed,
            )

        try:
            write_project_file(
                filepath,
                fixed,
            )
        except ValueError as exc:
            print(f"[CODER] Skipping invalid filepath during reviewer fix: {exc}")

        print(
            f"[CODER] Fixed -> {filepath}"
        )

    # --------------------------------------------------------
    # RETURN TO REVIEWER / EXECUTOR
    # --------------------------------------------------------

    return update_state(
        state,
        current_file_content=None,
        current_step_idx=len(steps),
        review_fix_required=False,
        last_validation_result=(
            "Reviewer fixes applied."
        ),
    )

# ============================================================
# GENERIC LLM CODER
# ============================================================

def generic_coder_node(
    state: CoderState,
) -> CoderState:

    if LOCAL_TEST_MODE:
        return local_generic_coder_node(state)

    if state.task_plan is None:
        raise RuntimeError(
            "Generic coder requires a task plan."
        )

    steps = state.task_plan.implementation_steps

    if state.current_step_idx >= len(steps):
        return state

    step = steps[state.current_step_idx]

    filepath = step.filepath

    existing = read_project_file(filepath)

    prompt = f"""
You are the coding agent inside CodePilot-Ai.

Generate the complete content of:

{filepath}

TASK:
{step.task_description}

PROJECT:
{state.plan.name if state.plan else "Unknown"}

DESCRIPTION:
{state.plan.description if state.plan else ""}

FEATURES:
{json.dumps(state.plan.features if state.plan else [], indent=2)}

ORIGINAL REQUEST:
{state.user_prompt}

EXISTING FILE:
{existing}

REVIEW FEEDBACK:
{state.review_feedback or "None"}

Rules:
- Return only the file content.
- No markdown fences.
- No placeholder content.
- No TODO implementation comments.
"""

    content = invoke_llm(prompt).strip()

    if content.startswith("```"):
        content = re.sub(
            r"^```[a-zA-Z0-9_-]*\s*",
            "",
            content,
        )

        content = re.sub(
            r"\s*```$",
            "",
            content,
        )

    write_project_file(
        filepath,
        content,
    )

    modified = list(state.files_modified)

    if filepath not in modified:
        modified.append(filepath)

    print(
        f"[CODER] LLM -> {filepath}"
    )

    return update_state(
        state,
        current_file_content=content,
        files_modified=modified,
        current_step_idx=(
            state.current_step_idx + 1
        ),
        last_validation_result=(
            f"Generated {filepath}"
        ),
    )


# ============================================================
# CODER DISPATCHER
# ============================================================

def coder_node(
    state: CoderState,
) -> CoderState:

    # Existing-project requests ALWAYS use the preservation-first
    # path. Specialized new-project generators must never overwrite
    # an existing application.
    if is_git_modification_request(state):
        return local_generic_coder_node(state)

    if is_todo_project(
        state.user_prompt
    ):
        return local_todo_coder_node(state)

    if is_expense_project(
        state.user_prompt
    ):
        return local_expense_coder_node(state)

    if LOCAL_TEST_MODE:
        return local_generic_coder_node(state)

    return generic_coder_node(state)


# ============================================================
# REVIEWER NODE
# ============================================================

def reviewer_node(
    state: CoderState,
) -> CoderState:

    if is_todo_project(
        state.user_prompt
    ):
        result = validate_todo_project()

    elif is_expense_project(
        state.user_prompt
    ):
        result = validate_expense_project()

    elif LOCAL_TEST_MODE:
        result = validate_generic_local_project(
            state
        )

    else:
        if state.plan is None:
            raise RuntimeError(
                "Reviewer requires a project plan."
            )

        print(
            "[REVIEWER] Reviewing generated "
            "project with LLM."
        )

        files = project_files()

        project_content = []

        for filepath in files:
            if filepath.startswith(".codepilot"):
                continue

            project_content.append(
                f"""
--- {filepath} ---
{read_project_file(filepath)}
"""
            )

        review_prompt = f"""
Review this project.

Project:
{state.plan.name}

Description:
{state.plan.description}

Technology:
{state.plan.techstack}

Features:
{json.dumps(state.plan.features, indent=2)}

Original request:
{state.user_prompt}

Generated files:
{"".join(project_content)}

Return ONLY valid JSON:

{{
  "approved": true,
  "issues": [],
  "suggestions": [],
  "summary": "short review"
}}
"""

        raw = invoke_llm(review_prompt)

        result = ReviewResult.model_validate(
            extract_json(raw)
        )

    print(
        f"[REVIEWER] {result.summary}"
    )

    feedback = "\n".join(
        result.issues
    )

    return update_state(
        state,
        last_validation_result=result.summary,
        review_attempts=(
            state.review_attempts + 1
        ),
        review_feedback=feedback,
        review_fix_required=(
            not result.approved
        ),
    )


# ============================================================
# ROUTING
# ============================================================

def route_after_planner(
    state: CoderState,
) -> str:
    return "architect"


def route_after_architect(
    state: CoderState,
) -> str:
    return "git_context"


def route_after_coder(
    state: CoderState,
) -> str:

    if state.task_plan is None:
        return "executor"

    total_steps = len(
        state.task_plan.implementation_steps
    )

    # If reviewer requested a fix, do not run the
    # remaining original implementation steps.
    # Send the newly fixed project to Executor.
    if state.review_fix_required:
        return "executor"

    # More original implementation steps remain.
    if state.current_step_idx < total_steps:
        return "coder"

    # All coding tasks are complete.
    return "executor"


def route_after_executor(
    state: CoderState,
) -> str:

    # Successful execution -> finish.
    if state.executor_passed:
        return END

    # Maximum executor attempts.
    if state.executor_attempts >= 3:

        print()
        print(
            "[EXECUTOR] Maximum execution attempts reached."
        )

        return END

    # Execution failed -> reviewer.
    return "reviewer"


def route_after_reviewer(
    state: CoderState,
) -> str:

    if not state.review_fix_required:
        return END

    if state.executor_attempts >= 3:

        print()
        print(
            "[REVIEWER] Maximum execution attempts reached."
        )

        return END

    print()
    print(
        "[REVIEWER] Fixes requested. "
        "Sending project back to coder."
    )

    return "coder"

# ============================================================
# GRAPH
# ============================================================
# ============================================================
# PERSISTENT MEMORY NODES
# ============================================================

def memory_load_node(
    state: CoderState,
) -> CoderState:
    """
    Load persistent memory for the current project.

    Memory is currently used as contextual information
    for the autonomous pipeline and stored outside the
    generated project.
    """

    print()
    print("[MEMORY] Loading project memory...")

    memory = get_project_memory(
        state.user_prompt
    )

    if memory:
        print(
            "[MEMORY] Previous project memory found."
        )

        files = memory.get(
            "files",
            [],
        )

        if files:
            print(
                f"[MEMORY] Previous files: "
                f"{', '.join(files)}"
            )

        history = memory.get(
            "execution_history",
            [],
        )

        if history:
            last_execution = history[-1]

            print(
                "[MEMORY] Previous execution: "
                f"{last_execution.get('summary', '')}"
            )

    else:
        print(
            "[MEMORY] No previous memory found."
        )

    return state

def memory_save_node(
    state: CoderState,
) -> CoderState:
    """
    Save persistent project memory and create a Git
    commit after a successful automated build.
    """

    print()
    print("[MEMORY] Saving project memory...")

    prompt = state.user_prompt

    files = list(
        state.files_modified
    )

    if not files:
        files = []

        try:
            from agent.tools import list_files

            result = list_files.invoke({})

            if isinstance(
                result,
                str,
            ):
                files = [
                    line.strip()
                    for line in result.splitlines()
                    if line.strip()
                ]

        except Exception:
            files = []

    if is_todo_project(prompt):
        project_type = "todo"

    elif is_expense_project(prompt):
        project_type = "expense"

    elif is_weather_project(prompt):
        project_type = "weather"

    elif is_expense_analytics_project(prompt):
        project_type = "expense_analytics"

    else:
        project_type = "generic"

    # --------------------------------------------------------
    # SAVE PROJECT SNAPSHOT
    # --------------------------------------------------------

    add_project_snapshot(
        prompt,
        files=files,
        project_type=project_type,
    )

    # --------------------------------------------------------
    # SAVE EXECUTION HISTORY
    # --------------------------------------------------------

    add_execution_history(
        prompt,
        passed=state.executor_passed,
        summary=(
            state.last_validation_result
            or "Automated execution completed."
        ),
    )

    # --------------------------------------------------------
    # SAVE REVIEW HISTORY
    # --------------------------------------------------------

    add_review_history(
        prompt,
        approved=(
            not state.review_fix_required
        ),
        summary=(
            state.last_validation_result
            or "Review completed."
        ),
        issues=(
            [state.review_feedback]
            if state.review_feedback
            else []
        ),
    )

    # --------------------------------------------------------
    # SAVE PIPELINE STATE
    # --------------------------------------------------------

    update_project_memory(
        prompt,
        last_validation_result=(
            state.last_validation_result
            or ""
        ),
        executor_passed=(
            state.executor_passed
        ),
        executor_attempts=(
            state.executor_attempts
        ),
        review_attempts=(
            state.review_attempts
        ),
    )

    print(
        "[MEMORY] Project memory saved."
    )

    # --------------------------------------------------------
    # GIT INTEGRATION
    # --------------------------------------------------------

    if state.executor_passed:

        print()
        print(
            "[GIT] Creating project snapshot..."
        )

        commit_message = (
            f"feat: generate {project_type} project"
        )

        # IMPORTANT: always use the dedicated Git manager.
        # It runs Git with cwd=generated_project, so the
        # parent CodePilot-Ai repository can never be committed.
        print(
            f"[GIT] Target repository: "
            f"{git_manager.GENERATED_PROJECT}"
        )

        success, init_message = (
            git_manager.initialize_repository()
        )

        if not success:
            success = False
            message = init_message
        else:
            success, message = (
                git_manager.create_commit(
                    commit_message
                )
            )

        if success:

            print(
                "[GIT] Commit successful."
            )

            print(
                f"[GIT] {message}"
            )

            snapshot = git_manager.git_snapshot()

            latest_commit = snapshot.get(
                "latest_commit",
                "",
            )

            if latest_commit:

                print(
                    f"[GIT] Latest commit: "
                    f"{latest_commit}"
                )

            update_project_memory(
                prompt,
                git_commit=latest_commit,
                git_repository=True,
            )

        else:

            print(
                "[GIT] Commit failed."
            )

            print(
                f"[GIT] {message}"
            )

            update_project_memory(
                prompt,
                git_repository=True,
                git_commit_error=message,
            )

    else:

        print()
        print(
            "[GIT] Skipping commit because "
            "automated execution did not pass."
        )

    return state

def build_graph():

    graph = StateGraph(
        CoderState
    )

    # --------------------------------------------------------
    # NODES
    # --------------------------------------------------------

    graph.add_node(
    "memory_load",
    memory_load_node,
)

    graph.add_node(
    "planner",
    planner_node,
)

    graph.add_node(
    "architect",
    architect_node,
)

    graph.add_node(
    "git_context",
    git_context_node,
)

    graph.add_node(
    "git_checkpoint",
    git_checkpoint_node,
)

    graph.add_node(
    "coder",
    coder_node,
)

    graph.add_node(
    "executor",
    executor_node,
)

    graph.add_node(
    "reviewer",
    reviewer_node,
)

    graph.add_node(
    "memory_save",
    memory_save_node,
)
    # --------------------------------------------------------
    # START
    # --------------------------------------------------------
    # --------------------------------------------------------
    # START → MEMORY
    # --------------------------------------------------------

    graph.add_edge(
        START,
        "memory_load",
    )

    # --------------------------------------------------------
    # MEMORY → PLANNER
    # --------------------------------------------------------

    graph.add_edge(
        "memory_load",
        "planner",
    )

    # --------------------------------------------------------
    # PLANNER → ARCHITECT
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "architect": "architect",
        },
    )

    # --------------------------------------------------------
    # ARCHITECT → GIT CONTEXT
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "architect",
        route_after_architect,
        {
            "git_context": "git_context",
        },
    )

    # --------------------------------------------------------
    # GIT CONTEXT → GIT CHECKPOINT
    # --------------------------------------------------------

    graph.add_edge(
        "git_context",
        "git_checkpoint",
    )

    # --------------------------------------------------------
    # GIT CHECKPOINT → CODER
    # --------------------------------------------------------

    graph.add_edge(
        "git_checkpoint",
        "coder",
    )

    # --------------------------------------------------------
    # CODER → CODER / EXECUTOR
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "coder",
        route_after_coder,
        {
            "coder": "coder",
            "executor": "executor",
        },
    )

    # --------------------------------------------------------
    # EXECUTOR → REVIEWER / MEMORY SAVE
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "reviewer": "reviewer",
            END: "memory_save",
        },
    )

    # --------------------------------------------------------
    # REVIEWER → CODER / MEMORY SAVE
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "reviewer",
        route_after_reviewer,
        {
            "coder": "coder",
            END: "memory_save",
        },
    )

    # --------------------------------------------------------
    # MEMORY SAVE → END
    # --------------------------------------------------------

    graph.add_edge(
        "memory_save",
        END,
    )

    return graph.compile()

def read_multiline_prompt() -> str:

    print()
    print("Enter your project request.")
    print(
        "When finished, enter END on a separate line."
    )
    print()

    lines = []

    while True:
        try:
            line = input()

        except EOFError:
            break

        if line.strip() == "END":
            break

        lines.append(line)

    return "\n".join(lines).strip()


# ============================================================
# MAIN
# ============================================================

def main():

    print("CodePilot-Ai")
    print("=" * 64)

    if LOCAL_TEST_MODE:
        print(
            "LOCAL_TEST_MODE: ON "
            "(ALL projects use zero LLM API calls)"
        )
    else:
        print(
            "LOCAL_TEST_MODE: OFF "
            "(generic projects may use Gemini/Groq)"
        )

    prompt = read_multiline_prompt()

    if not prompt:
        print(
            "No project request provided."
        )
        return

    print()
    print("Project request received.")
    print()

    if is_todo_project(prompt):
        print(
            "[MODE] Todo project -> LOCAL MODE"
        )

    elif is_expense_project(prompt):
        print(
            "[MODE] Expense project -> LOCAL MODE"
        )

    elif LOCAL_TEST_MODE:
        print(
            "[MODE] Generic project -> "
            "LOCAL / ZERO API MODE"
        )

    else:
        print(
            "[MODE] Generic project -> "
            "Gemini/Groq LLM MODE"
        )

    print()

    app = build_graph()

    initial_state = CoderState(
        user_prompt=prompt
    )

    final_state = app.invoke(
        initial_state,
        config={
            "recursion_limit": 50,
        },
    )

    if isinstance(
        final_state,
        CoderState,
    ):
        final_state_data = final_state
    else:
        final_state_data = (
            CoderState.model_validate(
                final_state
            )
        )

    print()
    print("=" * 64)
    print("CodePilot-Ai finished")
    print("=" * 64)

    print()
    print("Generated files:")

    for filepath in project_files():
        print(f"  - {filepath}")

    print()

    if final_state_data.last_validation_result:
        print(
            "Validation: "
            f"{final_state_data.last_validation_result}"
        )

    print()
    print(
        f"Project location: {PROJECT_ROOT}"
    )


if __name__ == "__main__":
    main()