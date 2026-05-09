# Project Context

## Agent Brief
You are working on a connected pest detection and decision-support system.
Treat this as one integrated product, not as separate lab exercises.

## Mission
Build a complete AI system for pest detection with vision, retrieval, LLM reasoning, simulation, visualization, and decision support.
The final system must be connected to a UI and support a full user workflow, not only model training or deployment.

## Required Deliverables
- Python source files for the project
- Screen recording of the running system
- Presentation slides
- Documentation

## Documentation Requirements
The documentation must include:
- Abstract
- Introduction
- Related Work
- Dataset section
- Implementation section
- Diagrams and visualizations wherever useful
- Screens from the running system

## System Overview
The project is a single connected pipeline:
User input and data sources flow into preprocessing, model inference, retrieval, simulation, dashboard visualization, and decision support.
The major labs must work together as one system:
- Lab 1: Domain Q&A chatbot with LLM
- Lab 2: Prompt engineering for domain analysis
- Lab 3: Data collection and preprocessing
- Lab 4: Retrieval Augmented Generation with datasets
- Lab 5: Predictive ML model
- Lab 6: Synthetic data generation with GANs
- Lab 7: Scenario simulation
- Lab 8: Visualization dashboard
- Lab 9: Integrated decision-support system

## Technical Direction
- Use Python as the implementation language
- Use PyTorch or compatible ML tooling for vision and generation tasks
- Use HuggingFace-style NLP components where needed
- Use a vector database approach such as FAISS for retrieval
- Keep the UI integrated with the AI services so the system is demonstrable end to end

## Architecture Rules
- Follow the project architecture in ARCHITECTURE.md
- Follow the coding conventions in CLAUDE.md
- Follow the implementation order and constraints in RULES.md
- Keep file and module organization aligned with STRUCTURE.json
- Do not invent unrelated components or features
- Do not split the system into disconnected mini-projects

## Working Rules For Any Agent
1. Start from the most relevant existing file or component.
2. Read the architecture and rules before making changes.
3. Make the smallest change that preserves the connected system design.
4. Prefer clarity, traceability, and maintainability over cleverness.
5. Update PROGRESS.md after each meaningful modification.
6. Keep explanations grounded in the actual project files.

## Current Implementation Priority
1. Establish the project structure.
2. Implement Lab 3 first because it is the data foundation.
3. Build Labs 5 and 6 on top of the processed data.
4. Add Labs 2, 4, 1, 7, 8, and 9 in dependency order.

## Quality Expectations
- The final product should be complete, connected, and runnable
- The dashboard must present the system outputs clearly
- The decision support layer must combine model, retrieval, chatbot, and simulation outputs
- The documentation and presentation should explain the full system, not just one component

## Reference Files
- ARCHITECTURE.md: system design and data flow
- CLAUDE.md: coding standards and development rules
- RULES.md: implementation order and constraints
- STRUCTURE.json: project file map
- PROGRESS.md: modification log

<environment_details>
Current time: 2026-05-07T13:40:41+03:00
</environment_details>