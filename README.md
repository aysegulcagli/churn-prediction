
# Multi-Modal Churn Prediction

**Predicting customer churn by fusing behavioral time series and natural language signals — a production-grade deep learning system.**

---

## Problem Statement

Customer churn costs businesses billions annually. Identifying at-risk customers before they leave enables proactive retention strategies that directly impact revenue.

Traditional churn models rely on a single data source — typically tabular features or aggregated statistics. This approach misses critical signals:

- **Time series patterns**: Gradual disengagement (fewer logins, shorter sessions) often precedes churn, but aggregated metrics flatten these trends.
- **Text signals**: Support tickets and feedback contain explicit frustration indicators that structured data cannot capture.

**This project solves both problems** by combining an LSTM-based time series encoder with a Transformer-based text encoder, fusing them into a unified prediction model. The result: a system that learns from *what users do* and *what users say*.

---

## Solution Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT DATA                              │
├─────────────────────────────┬───────────────────────────────────┤
│   Time Series (30 days)     │      Text (Support Tickets)       │
│   [logins, sessions, ...]   │      "App keeps crashing..."      │
└──────────────┬──────────────┴──────────────────┬────────────────┘
               │                                  │
               ▼                                  ▼
┌──────────────────────────┐       ┌──────────────────────────────┐
│   TimeSeriesEncoder      │       │       TextEncoder            │
│   (Multi-layer LSTM)     │       │   (Pretrained BERT)          │
│                          │       │                              │
│   Input: (B, T, F)       │       │   Input: (B, seq_len)        │
│   Output: (B, 128)       │       │   Output: (B, 768)           │
└──────────────┬───────────┘       └──────────────┬───────────────┘
               │                                  │
               └──────────────┬───────────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │        FusionModel           │
               │   Concat → Linear → ReLU     │
               │   → Dropout → Linear         │
               │                              │
               │   Output: Churn Probability  │
               └──────────────────────────────┘
```

---

## Model Comparison & Ablation Study

This project includes multiple model variants to evaluate the contribution of each data modality and modeling assumption.

| Model | Uses Time Series | Uses Text | Purpose |
|------|------------------|-----------|---------|
| Tabular Baseline | Aggregated | No | Performance floor |
| Time-Series-Only | Sequential | No | Measure temporal signal |
| Fusion Model | Sequential | Yes | Full multi-modal model |

### Results (Pending)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|------|----------|-----------|--------|----|---------|--------|
| Tabular Baseline | — | — | — | — | — | — |
| Time-Series-Only | — | — | — | — | — | — |
| Fusion Model | — | — | — | — | — | — |

> Results are intentionally omitted until experiments are conducted on real data.

---

## Design Decisions & Trade-offs

### Why Multiple Baselines?

Complex models are only valuable if they outperform simpler alternatives. This project includes three model variants—not to inflate the repository, but to enable honest evaluation.

- **Tabular Baseline** answers: “Do we even need deep learning?”
- **Time-Series-Only** answers: “Does temporal modeling justify the added complexity?”
- **Fusion Model** answers: “Does text provide signal beyond behavioral data?”

Without baselines, a reported ROC-AUC lacks context.

> Understanding **why** a model works matters more than simply proving that it works.

---

### Why Ablation Over Benchmark Chasing?

Rather than reporting a single best-performing model, this project prioritizes ablation to understand the contribution of each component.

If the Fusion model only marginally outperforms the Time-Series-Only variant, the additional complexity of text modeling may not be justified in production. If a simple tabular baseline performs competitively, the entire modeling approach deserves reconsideration.

---

### Why a Unified Training Interface?

All model variants share the same Dataset, Trainer, and evaluation pipeline. This was a deliberate engineering decision:

1. **Consistency** – Differences in results reflect model architecture, not pipeline variance.
2. **Maintainability** – New baselines require only a model file, not a parallel infrastructure.
3. **Realism** – Production systems rarely maintain separate pipelines per model variant.

---

### Why Model Text Separately?

Instead of manual text feature engineering, this project uses a learned text encoder.

This allows:
- Richer semantic representations
- Task-specific feature learning
- Modular upgrades without pipeline changes

---

### Why No Results Yet?

Reporting metrics without real data is misleading. Placeholder plots demonstrate the evaluation pipeline—not real-world performance.

When real data becomes available, this repository supports rigorous experimentation and reproducible results without architectural changes.

---

## Project Structure

```
churn-prediction/
├── configs/
│   └── base.yaml              # All hyperparameters in one place
├── scripts/
│   ├── train.py               # CLI entry point for training
│   ├── evaluate.py            # CLI entry point for evaluation
│   └── plot_metrics.py        # Generate visual reports
├── src/
│   ├── data/
│   │   └── dataset.py         # PyTorch Dataset implementation
│   ├── models/
│   │   ├── time_series_encoder.py
│   │   ├── text_encoder.py
│   │   └── fusion_model.py
│   ├── training/
│   │   └── trainer.py         # Training loop abstraction
│   ├── evaluation/
│   │   └── metrics.py         # Sklearn-based metrics
│   └── utils/
│       └── config.py          # YAML config loader
├── tests/
│   ├── test_dataset.py
│   ├── test_time_series_encoder.py
│   ├── test_text_encoder.py
│   ├── test_fusion_model.py
│   ├── test_training_loop.py
│   └── test_metrics.py
├── docs/
│   └── figures/               # Visual reports for README
├── pyproject.toml             # Project metadata and dependencies
└── CLAUDE.md                  # Development guidelines
```

---

## How to Run

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Generate visual reports
python scripts/plot_metrics.py
```

---

## What This Project Demonstrates

This repository is designed to showcase:

- **End-to-end ML system design** — from data loading to evaluation
- **Multi-modal deep learning** — fusing heterogeneous data sources
- **Production code quality** — type hints, docstrings, modular architecture
- **Testing discipline** — unit tests, integration tests, CI-ready structure
- **Clean engineering** — configuration-driven, no hardcoded values, separation of concerns

Built as a portfolio project to demonstrate applied ML engineering skills.

---

## Future Work

- Early stopping and model checkpointing
- Learning rate scheduling (cosine, warmup)
- Attention-based fusion (cross-modal attention)
- Hyperparameter tuning with Optuna
- MLflow experiment tracking
- Docker containerization for deployment
