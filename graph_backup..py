from dotenv import load_dotenv
import json
import re

from langchain.globals import set_verbose, set_debug
from langchain_groq.chat_models import ChatGroq
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

from agent.prompts import (
    planner_prompt,
    coder_system_prompt,
    reviewer_system_prompt,
)

from agent.states import (
    Plan,
    TaskPlan,
    CoderState,
    ReviewResult,
)

from agent.tools import (
    write_file,
    read_file,
    get_current_directory,
    list_files,
    run_project_command,
    init_project_root,
)


# ============================================================
# INITIALIZATION
# ============================================================

load_dotenv()

init_project_root()

set_debug(True)
set_verbose(True)

llm = ChatGroq(
    model="openai/gpt-oss-20b"
)


# ============================================================
# JSON HELPER
# ============================================================

def extract_json(text: str):
    """
    Extract a JSON object from the model response.
    Handles plain JSON and JSON wrapped in markdown.
    """

    if not text:
        raise ValueError(
            "Model returned an empty response."
        )

    text = text.strip()

    # Remove markdown code fences
    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```\s*",
        "",
        text
    )

    # Try direct JSON parsing first
    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    # Find JSON object inside extra text
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            f"Could not find JSON object in model response:\n{text}"
        )

    json_text = text[start:end + 1]

    try:
        return json.loads(json_text)

    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON returned by model:\n{text}"
        ) from e


# ============================================================
# PLANNER
# ============================================================

def planner_agent(state: dict) -> dict:

    user_prompt = state["user_prompt"]

    print("\n====================================")
    print("PLANNER")
    print("====================================")

    response = llm.with_structured_output(
        Plan
    ).invoke(
        planner_prompt(user_prompt)
    )

    if response is None:
        raise ValueError(
            "Planner did not return a valid response."
        )

    print("\n=== PLAN ===")
    print(
        response.model_dump_json(
            indent=2
        )
    )

    # IMPORTANT:
    # Preserve user_prompt in the state so that
    # Architect, Coder and Reviewer can access it.
    return {
        "user_prompt": user_prompt,
        "plan": response,
    }


# ============================================================
# ARCHITECT
# ============================================================

def architect_agent(state: dict) -> dict:

    plan: Plan = state["plan"]

    user_prompt = state["user_prompt"]

    print("\n====================================")
    print("ARCHITECT")
    print("====================================")

    architect_prompt_text = f"""
You are the software architect.

Create an implementation plan for the project below.

USER REQUEST:
{user_prompt}

PROJECT PLAN:
{plan.model_dump_json(indent=2)}

Return ONLY valid JSON.

The JSON MUST have exactly this structure:

{{
  "implementation_steps": [
    {{
      "filepath": "index.html",
      "task_description": "Detailed implementation instructions"
    }}
  ]
}}

Rules:

1. Return JSON only.
2. Do not use markdown.
3. Do not use ```json.
4. Every implementation step must contain:
   - filepath
   - task_description
5. Include every file required by the project.
6. Tasks must be ordered logically.
7. Do not include explanations outside JSON.
8. Do not create unnecessary files.
9. Make sure every requested feature has an implementation task.
"""

    response = llm.invoke(
        architect_prompt_text
    )

    raw_content = response.content

    print("\n=== RAW ARCHITECT RESPONSE ===")
    print(raw_content)

    data = extract_json(
        raw_content
    )

    task_plan = TaskPlan.model_validate(
        data
    )

    print("\n=== TASK PLAN ===")

    print(
        task_plan.model_dump_json(
            indent=2
        )
    )

    # IMPORTANT:
    # Preserve user_prompt.
    return {
        "user_prompt": user_prompt,
        "task_plan": task_plan,
    }


# ============================================================
# CODER EXECUTOR
# ============================================================

def create_coder_executor():

    coder_tools = [
        read_file,
        write_file,
        list_files,
        get_current_directory,
        run_project_command,
    ]

    return create_react_agent(
        llm,
        coder_tools
    )


# ============================================================
# CODER
# ============================================================

def coder_agent(state: dict) -> dict:

    coder_state: CoderState = state.get(
        "coder_state"
    )

    # --------------------------------------------------------
    # Initialize coder state
    # --------------------------------------------------------

    if coder_state is None:

        coder_state = CoderState(
            task_plan=state["task_plan"],
            current_step_idx=0,
        )

    steps = (
        coder_state
        .task_plan
        .implementation_steps
    )

    user_prompt = state.get(
        "user_prompt",
        ""
    )

    # --------------------------------------------------------
    # REVIEW FIX MODE
    # --------------------------------------------------------

    if coder_state.review_fix_required:

        print("\n====================================")
        print("CODER - REVIEW FIX MODE")
        print("====================================")

        review_feedback = (
            coder_state.review_feedback
            or state.get(
                "review_feedback",
                ""
            )
        )

        print("\nReviewer feedback:")
        print(review_feedback)

        project_files = list_files.run(
            "."
        )

        fix_prompt = f"""
You are the CODER agent in REVIEW-FIX MODE.

USER REQUEST:
{user_prompt}

REVIEWER FEEDBACK:
{review_feedback}

CURRENT PROJECT FILES:
{project_files}

The project is inside generated_project/.

Your job is to FIX the actual project.

IMPORTANT:

1. Read the relevant files first.
2. Identify the exact cause of every issue.
3. Inspect related files when necessary.
4. Use write_file() to modify the actual files.
5. Do not merely explain the solution.
6. Do not return code without writing it.
7. Do not create unrelated functionality.
8. Preserve all requested functionality.
9. Keep HTML, CSS and JavaScript references consistent.
10. Remove placeholder or unrelated code.
11. Validate the project after changes.
12. Fix validation errors.
13. Read the files again after writing them.

The task is complete ONLY after the fixes
have been written to generated_project/.
"""

        coder_executor = (
            create_coder_executor()
        )

        response = coder_executor.invoke(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": coder_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": fix_prompt,
                    },
                ]
            }
        )

        print("\n=== REVIEW FIX RESULT ===")
        print(response)

        coder_state.last_validation_result = str(
            response
        )

        coder_state.validation_attempts += 1

        # Clear review-fix flag
        coder_state.review_fix_required = False

        coder_state.review_feedback = None

        return {
            "user_prompt": user_prompt,
            "coder_state": coder_state,
            "review_feedback": "",
        }

    # --------------------------------------------------------
    # ALL TASKS COMPLETE
    # --------------------------------------------------------

    if coder_state.current_step_idx >= len(steps):

        print("\n====================================")
        print("CODER FINISHED")
        print("====================================")

        return {
            "user_prompt": user_prompt,
            "coder_state": coder_state,
        }

    # --------------------------------------------------------
    # CURRENT TASK
    # --------------------------------------------------------

    current_task = steps[
        coder_state.current_step_idx
    ]

    print("\n====================================")

    print(
        f"CODER TASK "
        f"{coder_state.current_step_idx + 1}/"
        f"{len(steps)}"
    )

    print("====================================")

    print(
        f"File: {current_task.filepath}"
    )

    print(
        f"Task: {current_task.task_description}"
    )

    # --------------------------------------------------------
    # Read existing file
    # --------------------------------------------------------

    existing_content = read_file.run(
        current_task.filepath
    )

    print("\nExisting file:")
    print(existing_content)

    # --------------------------------------------------------
    # Coder prompt
    # --------------------------------------------------------

    coder_user_prompt = f"""
You are implementing ONE engineering task.

USER REQUEST:
{user_prompt}

CURRENT TASK:
{current_task.task_description}

TARGET FILE:
{current_task.filepath}

EXISTING CONTENT:
{existing_content}

IMPORTANT RULES:

1. Inspect relevant project files first.
2. Implement the COMPLETE task.
3. Use write_file() to actually save the implementation.
4. Do NOT merely describe the code.
5. Do NOT return code without writing it.
6. Preserve compatibility with other project files.
7. Check HTML/CSS/JavaScript references carefully.
8. Make IDs, classes and function names consistent.
9. Do not introduce unrelated functionality.
10. Do not leave placeholder implementations.
11. Use run_project_command() when practical.
12. If validation fails, fix the actual files.
13. Read the file again after writing it.
14. Make sure the final file contains real implementation.

MANDATORY WORKFLOW:

INSPECT
->
IMPLEMENT
->
WRITE WITH write_file()
->
READ/VERIFY
->
VALIDATE
->
FIX IF NEEDED
->
VALIDATE AGAIN

The task is NOT complete until the target file
has actually been written to generated_project/.
"""

    # --------------------------------------------------------
    # Run coder
    # --------------------------------------------------------

    coder_executor = (
        create_coder_executor()
    )

    response = coder_executor.invoke(
        {
            "messages": [
                {
                    "role": "system",
                    "content": coder_system_prompt(),
                },
                {
                    "role": "user",
                    "content": coder_user_prompt,
                },
            ]
        }
    )

    # --------------------------------------------------------
    # Verify file
    # --------------------------------------------------------

    written_content = read_file.run(
        current_task.filepath
    )

    # Check whether file actually exists.
    if (
        not written_content.strip()
        or written_content.startswith(
            "FILE_NOT_FOUND:"
        )
    ):

        print(
            f"\nWARNING: Coder did not write "
            f"{current_task.filepath}"
        )

        retry_prompt = f"""
The previous coder attempt failed to write:

{current_task.filepath}

You MUST now implement this task by actually
using the write_file tool.

USER REQUEST:
{user_prompt}

TASK:
{current_task.task_description}

TARGET FILE:
{current_task.filepath}

You MUST:

1. Read relevant files.
2. Implement the complete functionality.
3. Call write_file() with the complete file content.
4. Call read_file() afterward.
5. Verify the file contains the implementation.
6. Do not merely explain the implementation.
7. Do not finish until the actual file exists.

Do not skip the write_file tool.
"""

        retry_response = coder_executor.invoke(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": coder_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": retry_prompt,
                    },
                ]
            }
        )

        response = retry_response

        written_content = read_file.run(
            current_task.filepath
        )

    # --------------------------------------------------------
    # Final file verification
    # --------------------------------------------------------

    if (
        not written_content.strip()
        or written_content.startswith(
            "FILE_NOT_FOUND:"
        )
    ):

        raise RuntimeError(
            f"Coder failed to write required file: "
            f"{current_task.filepath}"
        )

    print("\n=== CODER RESULT ===")
    print(response)

    # --------------------------------------------------------
    # Update coder state
    # --------------------------------------------------------

    coder_state.last_validation_result = str(
        response
    )

    coder_state.current_file_content = (
        written_content
    )

    coder_state.validation_attempts += 1

    if (
        current_task.filepath
        not in coder_state.files_modified
    ):

        coder_state.files_modified.append(
            current_task.filepath
        )

    coder_state.current_step_idx += 1

    return {
        "user_prompt": user_prompt,
        "coder_state": coder_state,
    }


# ============================================================
# REVIEWER
# ============================================================

def reviewer_agent(state: dict) -> dict:

    coder_state: CoderState = (
        state["coder_state"]
    )

    user_prompt = state.get(
        "user_prompt",
        ""
    )

    print("\n====================================")
    print("REVIEWER STARTED")
    print("====================================")

    files = list_files.run(
        "."
    )

    review_prompt = f"""
Review the ACTUAL generated software project.

USER REQUEST:
{user_prompt}

PROJECT FILES:
{files}

The project is located inside generated_project/.

IMPORTANT:

You MUST inspect the actual file contents using read_file().

Do not assume files are implemented just because
they appear in the task plan.

Check:

1. Are all requested features implemented?
2. Does HTML reference the correct CSS and JS?
3. Does JavaScript implement the requested behavior?
4. Are there placeholder files?
5. Is there unrelated code?
6. Do JavaScript functions use correct DOM IDs?
7. Are CSS classes consistent with HTML?
8. Are there obvious JavaScript errors?
9. Does localStorage work if requested?
10. Are add/edit/delete/complete operations implemented?
11. Are search and filters implemented if requested?
12. Are all files integrated correctly?
13. Is the project responsive?
14. Is the project usable?
15. Does the project match the user's original request?

CRITICAL:

If the user requested a Todo app and any file contains
calculator logic, that is a failure.

If a required file is only a placeholder, that is a failure.

Do NOT modify files.

Return your review in plain text with:

APPROVED: YES or NO

ISSUES:
- issue 1
- issue 2

SUGGESTIONS:
- suggestion 1

SUMMARY:
short summary
"""

    reviewer_tools = [
        read_file,
        list_files,
        get_current_directory,
        run_project_command,
    ]

    reviewer_executor = (
        create_react_agent(
            llm,
            reviewer_tools
        )
    )

    response = reviewer_executor.invoke(
        {
            "messages": [
                {
                    "role": "system",
                    "content": reviewer_system_prompt(),
                },
                {
                    "role": "user",
                    "content": review_prompt,
                },
            ]
        }
    )

    print("\n=== REVIEWER RESULT ===")
    print(response)

    messages = response.get(
        "messages",
        []
    )

    final_message = (
        messages[-1].content
        if messages
        else str(response)
    )

    coder_state.review_attempts += 1

    # --------------------------------------------------------
    # Convert reviewer response to structured JSON
    # --------------------------------------------------------

    structured_review_prompt = f"""
Convert this software review into JSON.

REVIEW:

{final_message}

Return ONLY valid JSON.

Use exactly this structure:

{{
  "approved": true,
  "issues": [],
  "suggestions": [],
  "summary": "short summary"
}}

Rules:

- approved must be true or false
- issues must be an array of strings
- suggestions must be an array of strings
- summary must be a string
- JSON only
- no markdown
"""

    review_response = llm.invoke(
        structured_review_prompt
    )

    review_data = extract_json(
        review_response.content
    )

    review_result = (
        ReviewResult.model_validate(
            review_data
        )
    )

    print("\n=== STRUCTURED REVIEW ===")

    print(
        review_result.model_dump_json(
            indent=2
        )
    )

    # --------------------------------------------------------
    # REVIEW PASSED
    # --------------------------------------------------------

    if review_result.approved:

        print("\n====================================")
        print("REVIEW PASSED")
        print("====================================")

        return {
            "user_prompt": user_prompt,
            "review": review_result,
            "status": "DONE",
            "coder_state": coder_state,
        }

    # --------------------------------------------------------
    # REVIEW FAILED
    # --------------------------------------------------------

    print("\n====================================")
    print("REVIEW FAILED")
    print("====================================")

    feedback = "\n".join(
        review_result.issues
    )

    if not feedback:
        feedback = review_result.summary

    coder_state.review_feedback = (
        feedback
    )

    coder_state.review_fix_required = True

    return {
        "user_prompt": user_prompt,
        "review": review_result,
        "status": "REVIEW_FAILED",
        "review_feedback": feedback,
        "coder_state": coder_state,
    }


# ============================================================
# LANGGRAPH
# ============================================================

graph = StateGraph(
    dict
)


# ------------------------------------------------------------
# Nodes
# ------------------------------------------------------------

graph.add_node(
    "planner",
    planner_agent
)

graph.add_node(
    "architect",
    architect_agent
)

graph.add_node(
    "coder",
    coder_agent
)

graph.add_node(
    "reviewer",
    reviewer_agent
)


# ------------------------------------------------------------
# Planner -> Architect
# ------------------------------------------------------------

graph.add_edge(
    "planner",
    "architect"
)


# ------------------------------------------------------------
# Architect -> Coder
# ------------------------------------------------------------

graph.add_edge(
    "architect",
    "coder"
)


# ============================================================
# ROUTING AFTER CODER
# ============================================================

def route_after_coder(
    state: dict
) -> str:

    coder_state: CoderState = (
        state["coder_state"]
    )

    steps = (
        coder_state
        .task_plan
        .implementation_steps
    )

    # If reviewer requested fixes,
    # coder must perform them.
    if coder_state.review_fix_required:

        return "coder"

    # More implementation tasks remain.
    if (
        coder_state.current_step_idx
        < len(steps)
    ):

        return "coder"

    # All implementation tasks finished.
    return "reviewer"


graph.add_conditional_edges(
    "coder",
    route_after_coder,
    {
        "coder": "coder",
        "reviewer": "reviewer",
    }
)


# ============================================================
# ROUTING AFTER REVIEWER
# ============================================================

def route_after_reviewer(
    state: dict
) -> str:

    if state.get(
        "status"
    ) == "DONE":

        return "END"

    return "coder"


graph.add_conditional_edges(
    "reviewer",
    route_after_reviewer,
    {
        "END": END,
        "coder": "coder",
    }
)


# ============================================================
# ENTRY POINT
# ============================================================

graph.set_entry_point(
    "planner"
)


# ============================================================
# COMPILE
# ============================================================

agent = graph.compile()


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    print("\n====================================")
    print("          CODEPILOT-AI")
    print("====================================\n")

    user_prompt = input(
        "Enter your project prompt: "
    ).strip()

    if not user_prompt:

        print(
            "No project prompt provided."
        )

        raise SystemExit(1)

    result = agent.invoke(
        {
            "user_prompt": user_prompt
        },
        {
            "recursion_limit": 100
        }
    )

    print("\n====================================")
    print("       PROJECT GENERATION DONE")
    print("====================================")

    print("\nFinal State:")

    print(result)