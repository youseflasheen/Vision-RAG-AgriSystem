# Implementation Rules

## Core Principles

### 1. One-by-One Development
- **Never implement multiple labs simultaneously**
- Complete each lab fully before moving to the next
- Dependencies must be resolved first (e.g., Lab 3 before Lab 5)

### 2. Explain Before Code
Before writing any code, I will:
1. State the **logical thinking** process
2. Identify which files will be created/modified
3. Explain the **purpose** of each component
4. Wait for your confirmation

### 3. Code Clarity Requirements
- Every function must have a clear, single responsibility
- Variable names must be self-documenting (no abbreviations)
- Comment complex logic inline
- I must understand 100% before I proceed to next component

### 4. No Hallucination Rule
- Only implement what is explicitly specified
- When uncertain, ask for clarification
- Do not invent features or edge cases not mentioned
- Base all decisions on ARCHITECTURE.md and user input

### 5. No Over-Specification Rule
- Do not add unnecessary helper functions
- Do not create generic utilities "just in case"
- Avoid duplicate logic across labs
- Each lab must have its own isolated implementation

## File Structure Rules

```
Before writing code, always check:
1. Is this file defined in STRUCTURE.json?
2. Does this match ARCHITECTURE.md?
3. Is this part of the current lab only?
```

## Lab Execution Order

1. **Lab 3** → Data Collection & Preprocessing (foundational)
2. **Lab 5** → ML Model (depends on Lab 3)
3. **Lab 6** → GAN (depends on Lab 3)
4. **Lab 1** → Chatbot (depends on Lab 2, Lab 4)
5. **Lab 2** → Prompts (foundational for Lab 1)
6. **Lab 4** → RAG (depends on Lab 3)
7. **Lab 7** → Simulation (depends on Lab 5)
8. **Lab 8** → Dashboard (depends on all labs)
9. **Lab 9** → DSS (depends on all labs)

## Commit Protocol

Each modification must update PROGRESS.md with:
- Date
- Lab number
- File modified
- Action taken
- Status (In Progress → Complete)

## Verification Checklist

Before considering any lab complete:
- [ ] Code runs without errors
- [ ] Output matches ARCHITECTURE.md specification
- [ ] All functions have type hints and docstrings
- [ ] PROGRESS.md is updated
- [ ] You approve the implementation