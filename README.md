# Multi-Modal Churn Prediction

A production-grade machine learning system for predicting customer churn by combining time series behavioral data with natural language signals from customer interactions.

## Project Overview

This repository implements an end-to-end churn prediction pipeline that fuses two distinct data modalities:

- **Time Series Data**: Sequential user behavior patterns (login frequency, feature usage, session duration)
- **Text Data**: Customer support tickets, feedback, and communication logs

The system is designed with production readiness in mind, emphasizing modularity, testability, and clean engineering practices.

## Modeling Approach

### Time Series Encoder

An LSTM-based encoder processes variable-length behavioral sequences into fixed-size representations. The encoder captures temporal dependencies in user activity patterns that often precede churn events.

**Architecture**: Multi-layer LSTM with configurable hidden dimensions and dropout regularization. The final hidden state serves as the sequence representation.

### Text Encoder

A Transformer-based encoder (BERT) extracts semantic features from customer text data. The encoder uses the `[CLS]` token representation as a summary of the text content.

**Architecture**: Pretrained BERT with configurable layer freezing for transfer learning efficiency.

### Fusion Model

A concatenation-based fusion layer combines the time series and text embeddings, followed by fully connected layers that output churn probability.

```
[Time Series Embedding] + [Text Embedding] → Linear → ReLU → Dropout → Linear → Logits
```

## Project Structure

```
churn-prediction/
├── configs/
│   └── base.yaml              # Hyperparameters and configuration
├── scripts/
│   ├── train.py               # Training entry point
│   └── evaluate.py            # Evaluation entry point
├── src/
│   ├── data/
│   │   └── dataset.py         # PyTorch Dataset for multi-modal data
│   ├── models/
│   │   ├── time_series_encoder.py
│   │   ├── text_encoder.py
│   │   └── fusion_model.py
│   ├── training/
│   │   └── trainer.py         # Training loop orchestration
│   ├── evaluation/
│   │   └── metrics.py         # Evaluation metrics
│   └── utils/
│       └── config.py          # Configuration loading
├── tests/
│   ├── test_dataset.py
│   ├── test_time_series_encoder.py
│   ├── test_text_encoder.py
│   ├── test_fusion_model.py
│   ├── test_training_loop.py
│   └── test_metrics.py
├── pyproject.toml
└── CLAUDE.md
```

## Evaluation Metrics

The evaluation module provides standard classification metrics appropriate for imbalanced churn datasets:

| Metric | Description |
|--------|-------------|
| Accuracy | Overall classification correctness |
| Precision | Proportion of predicted churners who actually churned |
| Recall | Proportion of actual churners correctly identified |
| F1 Score | Harmonic mean of precision and recall |
| ROC-AUC | Area under the ROC curve |
| PR-AUC | Area under the precision-recall curve |

All metrics are implemented using scikit-learn with proper handling of edge cases.

## Testing Strategy

The project employs a multi-level testing approach:

**Unit Tests**
- Individual component validation (encoders, dataset, metrics)
- Shape verification for tensor operations
- Edge case handling (mismatched dimensions, empty inputs)

**Integration Tests**
- End-to-end training loop smoke tests
- Gradient flow verification through the full model
- Mock encoders for isolated testing without heavy dependencies

**Test Execution**
```bash
pytest tests/ -v
```

## Training Loop Overview

The `Trainer` class orchestrates the training process:

1. **Epoch Training**: Iterates over batches, computes forward pass, backpropagates gradients
2. **Validation**: Evaluates on held-out data with gradients disabled
3. **History Tracking**: Records train/validation loss for monitoring

The training loop is designed for extensibility, with placeholder methods for checkpointing and early stopping.

## Engineering Principles

This project adheres to production ML engineering standards:

- **Type Hints**: Full type annotations on all functions and methods
- **Docstrings**: Comprehensive documentation with Args, Returns, and Raises
- **Modularity**: Each component is independently testable and replaceable
- **Configuration-Driven**: All hyperparameters externalized to YAML configs
- **No Magic Numbers**: Constants and parameters are explicitly named
- **Clean Separation**: Data loading, model definition, training logic, and evaluation are decoupled

## Future Work

- Early stopping based on validation loss
- Model checkpointing and resumption
- Learning rate scheduling
- Attention-based fusion mechanisms
- Hyperparameter tuning integration
