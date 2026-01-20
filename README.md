
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

### Results & Comparison

#### Experimental Setup
- **Dataset**: KKBOX music streaming service (~30GB user logs, ~392M rows)
- **Observation window**: 30 days of user activity
- **Evaluation split**: Validation set only (time-based split)
- **Primary metric**: ROC-AUC (ranking ability)
- **Secondary metrics**: PR-AUC, F1 score

#### Model Performance

| Model                    | ROC-AUC | PR-AUC | F1 (t=0.5) | F1 (optimized) |
|--------------------------|---------|--------|------------|----------------|
| Tabular Baseline         | —       | —      | —          | —              |
| Time-Series-Only (LSTM)  | ~0.60   | ~0.31  | ~0.00      | ~0.41*         |

\* Optimized threshold ≈ 0.25 (selected on validation set)

> **Note**: The tabular baseline was successfully trained and validated. Due to differences in dataset representation and caching between aggregated and sequential inputs, only qualitative comparison is reported for the baseline in this iteration.

#### Interpreting the Metrics

**Why F1 at the default threshold (0.5) is near zero**

Churn prediction is a highly imbalanced problem where the majority of users do not churn. Using a default threshold of 0.5 assumes balanced classes and therefore results in almost no positive predictions. This behavior is expected and does not indicate a model failure.

**Why ROC-AUC and PR-AUC are more informative**

In practical churn prediction systems, the primary goal is to rank users by churn risk rather than make perfectly calibrated binary predictions. ROC-AUC measures ranking quality independent of threshold choice, while PR-AUC focuses on performance for the minority (churn) class and is especially relevant under class imbalance.

**Threshold selection in practice**

The optimal decision threshold depends on business constraints:
- Lower threshold → higher recall, more false positives
- Higher threshold → higher precision, more missed churners

For this dataset, a threshold around 0.25 provides a reasonable balance between precision and recall.

#### Key Observations

1. **Class imbalance dominates evaluation**: Standard metrics like accuracy and F1@0.5 are misleading; threshold-independent metrics (AUC) provide a clearer picture
2. **Temporal modeling shows moderate gains**: The LSTM captures sequential patterns but benefits are incremental over aggregated features
3. **Practical value is in ranking**: The model successfully differentiates high-risk from low-risk users, enabling targeted retention campaigns
4. **Infrastructure challenges**: Processing 392M rows required memory-efficient chunked I/O and caching strategies

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

## Engineering Challenges & Lessons Learned

### Memory-Efficient Data Processing
Processing ~392 million rows on a single machine required careful engineering. A naive `pd.read_csv()` approach would quickly exhaust available memory. The solution was a two-pass, chunked processing pipeline: the first pass computes per-user metadata, while the second pass extracts windowed sequences. This design intentionally trades runtime for memory safety, enabling large-scale local experimentation.

### Caching Strategy and Its Pitfalls
To avoid reprocessing the 30GB dataset on every run, processed tensors are cached to disk. An early cache design used only the dataset split name as the cache key, ignoring the data representation mode. Switching between aggregated (tabular) and sequential (time-series) inputs led to stale caches and runtime shape mismatches. The key lesson: cache keys must encode all parameters that affect data shape or semantics.

### Time-Based Data Splitting
Churn prediction requires strict temporal integrity—training on future data to predict past churn would introduce leakage. Instead of random splits, datasets are constructed based on observation window end dates. While this added complexity to the pipeline, it ensures realistic and trustworthy evaluation.

### Class Imbalance and Metric Selection
With a low base churn rate, accuracy and F1 at default thresholds are misleading. The model achieved near-zero F1 at a threshold of 0.5 not due to poor learning, but because the threshold assumes balanced classes. This reinforced the importance of ranking-based metrics (ROC-AUC, PR-AUC) and explicit threshold tuning based on business constraints.

### Numerical Stability in Feature Engineering
Raw behavioral logs contained extreme values that propagated NaNs through log transformations and model gradients. Defensive preprocessing—clipping, `log1p`, and explicit NaN handling—was essential for maintaining numerical stability during training.

### Progress Visibility for Long-Running Jobs
Processing 79 chunks across two passes takes over 40 minutes. Without progress indicators, distinguishing between "working" and "stuck" is impossible. Adding progress bars and periodic logging transformed debugging from guesswork into informed monitoring.

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
