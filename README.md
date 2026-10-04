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

---
## Key Features

### Requirement Planning

The Planner analyzes the user's natural-language requirement and creates a structured project plan containing:

- Application name
- Application description
- Technology stack
- Required features
- Required project files

### Software Architecture

The Architect converts the high-level plan into concrete implementation tasks.

Each task identifies:

- File to implement or modify
- Detailed implementation requirements
- Ordered implementation steps

### File-by-File Coding

The Coder executes the implementation plan and writes changes directly into project files.

The workflow tracks:

- Current implementation step
- Current file
- Modified files
- Validation results
- Review feedback
- Execution attempts

### Existing Project Modification

CodePilot-Ai is designed to work with existing projects as well as new projects.

For existing-project requests, the workflow uses the current project state and Git context to make focused changes instead of unnecessarily replacing the complete application.

### RAG-Based Project Context

The agent includes a retrieval component that provides relevant project context to the coding workflow.

This helps the coder work with the existing implementation rather than relying only on the original user request.

### Git-Aware Development

CodePilot-Ai integrates Git into the coding workflow.

The system can track:

- Changed files
- Current Git status
- Latest commit
- Project Git repository
- Git diffs
- Coding checkpoints
- Recovery points

The generated project maintains its own Git repository:

```text
CodePilot-Ai/
└── generated_project/
    └── .git/

---
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

---
## Architecture

The main workflow is implemented using **LangGraph**.

The current graph contains the following major nodes:

```text
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

---
## Technology Stack

### Core

- Python
- LangGraph
- Pydantic

### AI / LLM

- Gemini
- Groq
- LLM-based planning and coding workflow

### Development Infrastructure

- Git
- RAG / retrieval-based project context
- Persistent project memory
- Automated execution and validation

### Project Environment

- `uv`
- Python virtual environment
- `pyproject.toml`

---
## Project Structure

```text
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

---
## How It Works

CodePilot-Ai processes a software requirement through multiple stages.

### 1. Requirement Analysis

The user provides a natural-language software requirement.

The Planner converts the requirement into a structured application plan.

### 2. Architecture Planning

The Architect converts the application plan into ordered implementation tasks.

Each task specifies the file to modify and the required implementation.

### 3. Project Context Retrieval

The RAG and Git Context stages inspect the existing project and provide relevant implementation context to the Coder.

This is especially important when modifying an existing application.

### 4. Git Checkpoint

Before coding, the workflow records the current Git state so that the implementation has a recovery point.

### 5. Code Generation

The Coder implements the assigned tasks directly in the project.

The workflow tracks the current implementation step and modified files.

### 6. Automated Execution

The Executor runs automated checks against the generated or modified project.

```text
Coder
  ↓
Executor
  ↓
Execution Result
```

### 7. Requirement-Level Review

The Reviewer evaluates the implementation against the original requirements.

```text
Execution
   ↓
Reviewer
   ↓
Requirements Satisfied?
   ├── Yes → Memory Save
   └── No  → Coder Fix
                ↓
             Executor
                ↓
             Reviewer
```

### 8. Completion

Once the implementation passes execution and requirement-level review, the workflow saves relevant memory and completes the run.

---
## Existing Project Modification

CodePilot-Ai supports modifying existing applications without unnecessarily regenerating the entire project.

For an existing-project request, the workflow:

1. Inspects the current implementation.
2. Retrieves relevant project context.
3. Detects the current Git state and changed files.
4. Creates a Git checkpoint before coding.
5. Identifies the files that need to be modified.
6. Applies focused implementation changes.
7. Runs automated execution checks.
8. Reviews the final implementation against the requirements.
9. Sends the project back to the Coder if corrections are required.

The goal is to preserve unrelated functionality, existing project structure, and Git history while implementing only the requested changes.

```text
Existing Project
       ↓
Inspect Current Implementation
       ↓
RAG + Git Context
       ↓
Git Checkpoint
       ↓
Focused Modification
       ↓
Automated Execution
       ↓
Requirement Review
       ↓
Pass → Complete
       │
       └── Fix Required → Coder → Executor → Reviewer
```

This allows CodePilot-Ai to operate as a development agent for both **new applications** and **existing-project maintenance**.

---
## Installation

### Clone the Repository

```bash
git clone https://github.com/kanakec/CodePilot-AI.git
cd CodePilot-AI
```

### Install Dependencies

Using `uv`:

```bash
uv sync
```

### Activate the Virtual Environment

```bash
.venv\Scripts\activate
```

---
## Environment Variables

Create a `.env` file in the project root and configure the required API credentials:

```env
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

Never commit API keys or other secrets to Git.

---
## Running CodePilot-Ai

Run the main agent workflow with:

```bash
uv run python -m agent.graph
```

The workflow processes the requirement through planning, architecture, project-context retrieval, Git checkpointing, coding, execution, and review.

For individual components:

```bash
uv run python -m agent.executor
uv run python -m agent.executor_node
uv run python -m agent.memory
uv run python -m agent.git_manager
uv run python -m agent.git_context
uv run python -m agent.git_context_node
uv run python -m agent.git_recovery
uv run python -m agent.git_checkpoint_node
```

---
## Example Workflow

A typical existing-project modification can be requested using a natural-language requirement such as:

```text
Update the existing Todo application.

Add Low, Medium, and High task priorities.
Add priority selection, display, filtering, styling,
and localStorage persistence while preserving all
existing functionality.
```

CodePilot-Ai then processes the request through:

```text
Requirement
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
Pass → Complete
    │
    └── Fix Required → Coder → Executor → Reviewer
```

The project is only considered complete after automated execution and requirement-level review.

---
## Author

**Kanak Bajaj**

- GitHub: https://github.com/kanakec
- LinkedIn: https://linkedin.com/in/kanak-bajaj-b73705289
- Project: https://github.com/kanakec/CodePilot-AI

---
