# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Build a production-grade multi-modal churn prediction system that combines:
- **Time series data**: User behavioral sequences (login frequency, feature usage, session duration over time)
- **Text data**: Support tickets, feedback, chat logs

The model predicts customer churn probability by learning from both temporal patterns and semantic signals.

## Data Description

### Time Series Component
- Sequential user activity logs with timestamps
- Features: engagement metrics, transaction history, usage patterns
- Variable-length sequences per user

### Text Component
- Customer support interactions
- User feedback and reviews
- Communication history

### Target
- Binary classification: churned (1) vs retained (0)
- Prediction horizon: 30/60/90 days (configurable)

## Development Rules

### Step-by-Step Development
- **Never generate complete modules in one shot.** Build incrementally.
- Implement one function at a time, test it, then proceed.
- Ask for confirmation before moving to the next component.
- If a task requires multiple files, create them one by one with explicit approval.

### My Role (Human)
- Define requirements and priorities
- Review and approve each implementation step
- Provide domain context and data specifics
- Make architectural decisions

### Your Role (Claude)
- Propose implementation approaches before coding
- Write code only after approach is approved
- Explain trade-offs for design decisions
- Flag potential issues or edge cases proactively

## Engineering Standards

### Code Quality
- Type hints on all functions
- Docstrings with Args, Returns, Raises
- No magic numbers—use constants or config
- Maximum function length: 30 lines

### Project Structure
```
churn-prediction/
├── src/
│   ├── data/          # Data loading, preprocessing
│   ├── features/      # Feature engineering
│   ├── models/        # Model architectures
│   ├── training/      # Training loops, callbacks
│   └── evaluation/    # Metrics, analysis
├── configs/           # YAML configuration files
├── notebooks/         # Exploration only, no production code
├── tests/             # Unit and integration tests
└── scripts/           # CLI entry points
```

### Modularity
- Each component must be independently testable
- Use dependency injection over hard-coded dependencies
- Separate data processing, model definition, and training logic
- Configuration-driven behavior (no hardcoded hyperparameters)

### Version Control
- Atomic commits with descriptive messages
- No generated files or data in commits
- Keep notebooks minimal and cleaned before commit

## Commands

*To be added as project develops:*
- Environment setup
- Data preprocessing pipeline
- Training commands
- Evaluation scripts
- Test execution
