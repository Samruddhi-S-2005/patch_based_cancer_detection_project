"""
Objective 2 Package: EfficientNet-B0 Binary Tumor Detection & Anomaly Analysis.
"""

from src.objective_2.dataset import (
    HistopathPatchDataset,
    create_dataloaders,
    get_eval_transforms,
    get_train_transforms,
)
from src.objective_2.model import EfficientNetB0TumorClassifier
from src.objective_2.metrics import (
    compute_metrics,
    plot_training_curves,
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion_matrix,
    plot_anomaly_and_risk_distribution,
)
from src.objective_2.anomaly_and_risk import (
    compute_anomaly_scores,
    categorize_risk,
    build_predictions_dataframe,
    generate_risk_summary,
    DISCLAIMER_TEXT,
)

__all__ = [
    "HistopathPatchDataset",
    "create_dataloaders",
    "get_train_transforms",
    "get_eval_transforms",
    "EfficientNetB0TumorClassifier",
    "compute_metrics",
    "plot_training_curves",
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_confusion_matrix",
    "plot_anomaly_and_risk_distribution",
    "compute_anomaly_scores",
    "categorize_risk",
    "build_predictions_dataframe",
    "generate_risk_summary",
    "DISCLAIMER_TEXT",
]
