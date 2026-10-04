# CodePilot-Ai

**CodePilot-Ai** is an autonomous software engineering agent built with **Python and LangGraph**. It converts natural-language software requirements into implementation tasks, modifies project files, executes automated checks, reviews the result against the requirements, and iteratively corrects issues when necessary.

The system is designed to support both **new project generation** and **focused modifications to existing projects** while maintaining project context and Git history.

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
