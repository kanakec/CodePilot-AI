# CodePilot-Ai

**CodePilot-Ai** is an autonomous software engineering agent built with **Python and LangGraph**. It converts natural-language software requirements into implementation tasks, modifies project files, executes automated checks, reviews the result against the requirements, and iteratively corrects issues when necessary.

The system supports both **new project generation** and **focused modifications to existing projects** while maintaining project context and Git history.

---

## Overview

CodePilot-Ai follows a multi-stage software engineering workflow:

```text
User Requirement
       ↓
Memory
       ↓
Planner
       ↓
Architect
       ↓
RAG / Project Context
       ↓
Git Context
       ↓
Git Checkpoint
       ↓
Coder
       ↓
Executor
       ↓
Reviewer
       ↓
   ┌───┴───┐
   │       │
 Pass    Fix Required
   │       ↓
   │     Coder
   │       ↓
   │    Executor
   │       ↓
   │    Reviewer
   │
   ↓
Memory
   ↓
Git Commit
Key Features
Requirement Planning

The Planner analyzes the user's natural-language requirement and creates a structured project plan containing:

Application name
Application description
Technology stack
Required features
Required project files
Software Architecture

The Architect converts the high-level plan into concrete implementation tasks.

Each task identifies:

File to implement or modify
Detailed implementation requirements
Ordered implementation steps
File-by-File Coding

The Coder executes the implementation plan and writes changes directly into project files.

The workflow tracks:

Current implementation step
Current file
Modified files
Validation results
Review feedback
Execution attempts
Existing Project Modification

CodePilot-Ai is designed to work with existing projects as well as new projects.

For existing-project requests, the workflow uses the current project state and Git context to make focused changes instead of unnecessarily replacing the complete application.

RAG-Based Project Context

The agent includes a retrieval component that provides relevant project context to the coding workflow.

This helps the coder work with the existing implementation rather than relying only on the original user request.

Git-Aware Development

CodePilot-Ai integrates Git into the coding workflow.

The system can track:

Changed files
Current Git status
Latest commit
Project Git repository
Git diffs
Coding checkpoints
Recovery points

The generated project maintains its own Git repository:
CodePilot-Ai/
└── generated_project/
    └── .git/
Git Checkpoints and Recovery

Before a coding attempt, the workflow can record a Git checkpoint.

This provides a recovery point if an implementation causes an unwanted change.

The project includes Git-aware recovery functionality for safer automated modifications.

Automated Execution

After implementation, the Executor runs automated project checks.

The workflow records:

Execution status
Errors
Warnings
Execution attempts
Validation results
Reviewer-Driven Self-Correction

The Reviewer performs a requirement-level review after automated execution.

If the implementation does not satisfy the requirements, the workflow sends the project back to the Coder:
Executor
   ↓
Reviewer
   ↓
Issues Found
   ↓
Coder
   ↓
Executor
   ↓
Reviewer
The correction loop is bounded by a maximum number of attempts.

Persistent Memory

CodePilot-Ai includes persistent memory functionality for retaining relevant project information across runs.

The workflow contains dedicated memory loading and saving stages:
Memory Load
     ↓
Agent Workflow
     ↓
Memory Save
Project History

The system includes project history functionality for tracking previous project interactions and development activity.

Semantic Project Identity

The project includes semantic project identity handling so that existing project context can be associated with the appropriate development workflow.

Architecture

The main workflow is implemented using LangGraph.

The current graph contains the following major nodes:
START
  ↓
memory_load
  ↓
planner
  ↓
architect
  ↓
git_context
  ↓
git_checkpoint
  ↓
coder
  ↓
executor
  ↓
reviewer
  ↓
memory_save
  ↓
END
The workflow also contains conditional routing between the Coder, Executor, and Reviewer.

When the Reviewer requests corrections, the workflow routes the project back to the Coder for another implementation attempt.

Technology Stack
Core
Python
LangGraph
Pydantic
AI / LLM
Gemini
Groq
LLM-based planning and coding workflow
Development Infrastructure
Git
RAG / retrieval-based project context
Persistent project memory
Automated execution and validation
Project Environment
uv
Python virtual environment
pyproject.toml
Project Structure
CodePilot-Ai/
│
├── agent/
│   ├── graph.py
│   ├── states.py
│   ├── prompts.py
│   ├── tools.py
│   ├── executor.py
│   ├── executor_node.py
│   ├── rag.py
│   ├── memory.py
│   ├── history.py
│   ├── git_manager.py
│   ├── git_diff.py
│   ├── git_context.py
│   ├── git_context_node.py
│   ├── git_checkpoint_node.py
│   └── git_recovery.py
│
├── generated_project/
│   ├── .git/
│   └── ...
│
├── pre_generated_project_calculator/
│   └── ...
│
├── pre_generated_project_todo_app/
│   └── ...
│
├── resources/
│   └── ...
│
├── .codepilot
├── .env
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md
Core Components
agent/graph.py

Contains the main LangGraph workflow and orchestration logic.

It connects the different stages of the autonomous development process.

agent/states.py

Defines structured state models used throughout the workflow, including:

Project plans
Implementation tasks
Review results
Validation state
Git context
Git checkpoints
Execution state
agent/rag.py

Provides retrieval-based project context for the coding workflow.

agent/executor.py

Handles automated execution and validation of generated projects.

agent/executor_node.py

Integrates execution into the LangGraph workflow.

agent/memory.py

Handles persistent project memory.

agent/history.py

Tracks project-level development history.

agent/git_manager.py

Provides Git repository operations such as:

Repository initialization
Status checking
Changed-file detection
Commit creation
Latest commit retrieval
Git snapshots
agent/git_diff.py

Provides Git change and diff information to the development workflow.

agent/git_context.py

Builds Git-aware context for the coder.

agent/git_context_node.py

Integrates Git context into the LangGraph workflow.

agent/git_checkpoint_node.py

Creates Git checkpoints before coding operations.

agent/git_recovery.py

Provides safe recovery functionality using Git checkpoints.

Installation
Prerequisites

Make sure the following are installed:

Python 3.11+
Git
uv
Clone the repository
git clone https://github.com/kanakec/CodePilot-AI.git
cd CodePilot-AI
Create the virtual environment
uv venv
Activate the environment
Windows PowerShell
.venv\Scripts\Activate.ps1
Linux / macOS
source .venv/bin/activate
Install dependencies
uv sync
Environment Variables

Create a .env file in the project root.

Add the API credentials required by the configured LLM providers.

Example:

GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key

Never commit API keys or other credentials to Git.

Running the Application

The main application can be started with:

uv run python main.py
Running the Development Workflow

The LangGraph workflow can also be executed directly with:

uv run python -m agent.graph
Running Individual Components
Executor
uv run python -m agent.executor
Executor Node
uv run python -m agent.executor_node
Memory
uv run python -m agent.memory
Git Manager
uv run python -m agent.git_manager
Project History
uv run python -m agent.history
Git Diff
uv run python -m agent.git_diff
Git Context
uv run python -m agent.git_context
Git Context Node
uv run python -m agent.git_context_node
Git Recovery
uv run python -m agent.git_recovery
Git Checkpoint Node
uv run python -m agent.git_checkpoint_node
Example Workflow

A user can provide a requirement such as:

Update the existing Todo application by adding task priorities
with Low, Medium, and High levels while preserving all existing
functionality.

CodePilot-Ai processes the request through:

Requirement
    ↓
Planner
    ↓
Architect
    ↓
Existing Project Context
    ↓
Git Context
    ↓
Git Checkpoint
    ↓
Coder
    ↓
Executor
    ↓
Reviewer

If the reviewer identifies an incomplete requirement:

Reviewer
    ↓
Fix Required
    ↓
Coder
    ↓
Executor
    ↓
Reviewer

The workflow continues until the implementation passes the configured validation and review criteria or the maximum retry limit is reached.

Existing Project Safety

When modifying an existing application, the workflow is designed to preserve unrelated functionality.

The coding process can use:

Existing file contents
Project context
Git status
Git diffs
Git checkpoints
Reviewer feedback

This enables targeted modifications instead of treating every request as a brand-new application.

Git Integration

The generated project is maintained as a separate Git repository:

CodePilot-Ai/
│
└── generated_project/
    └── .git/

Useful commands for inspecting the generated project:

git -C ./generated_project status
git -C ./generated_project log --oneline
git -C ./generated_project diff

This separation allows the generated application's development history to be tracked independently from the agent itself.

Validation and Review

CodePilot-Ai uses two complementary validation stages.

Automated Execution

The Executor checks whether the generated project can successfully execute according to the configured validation process.

Requirement-Level Review

The Reviewer checks whether the implementation actually satisfies the requested requirements.

This distinction is important because a project can execute successfully while still missing a requested feature.

The intended completion condition is:

Implementation
      ↓
Automated Execution
      ↓
Requirement Review
      ↓
Approved
Local Test Mode

The project includes a local test mode for deterministic development and validation.

When local test mode is enabled, the workflow can execute configured local checks without requiring LLM API calls for that test run.

This is useful for testing the orchestration and project-validation infrastructure independently from external model availability.

Design Goals

CodePilot-Ai is designed around several software engineering principles:

Structured planning instead of direct code generation
File-level implementation tasks
Existing-project awareness
Automated execution
Requirement-level review
Iterative self-correction
Git-aware development
Checkpoint-based recovery
Persistent project memory
Focused modifications instead of unnecessary rewrites
Development Workflow

A typical development cycle looks like:

1. Receive requirement
2. Load relevant memory
3. Create project plan
4. Design implementation tasks
5. Retrieve project context
6. Inspect Git state
7. Create checkpoint
8. Implement changes
9. Execute automated checks
10. Review requirements
11. Correct identified issues
12. Re-execute
13. Re-review
14. Save project memory
15. Commit successful changes
Limitations

CodePilot-Ai is an active development project.

The quality of generated implementations depends on:

The clarity of the user's requirements
Available project context
Configured models
Existing project structure
Automated validation coverage
Reviewer feedback

The system should therefore be treated as an engineering automation workflow rather than a replacement for human code review.

Future Improvements

Potential areas for further development include:

More comprehensive automated test generation
Improved code-quality analysis
Better dependency management
More advanced project-level RAG
Multi-language project support
More robust rollback strategies
Improved execution sandboxing
Expanded test-case generation
Better observability and execution tracing
More granular agent performance metrics
Author

Kanak Bajaj

GitHub: https://github.com/kanakec

Project: https://github.com/kanakec/CodePilot-AI
