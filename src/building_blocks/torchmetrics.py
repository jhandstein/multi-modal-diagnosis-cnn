import torch
from typing import Literal
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryAUROC,
)
from torchmetrics.regression import (
    R2Score,
    MeanAbsoluteError,
    MeanSquaredError,
)


class BaseMetrics:
    def __init__(self, phase: Literal["train", "val", "test"]):
        self.phase = phase
        self.metrics = {}
        self._initialize_metrics()

    def _initialize_metrics(self):
        """Initialize metrics - to be implemented by subclasses."""
        raise NotImplementedError

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor) -> dict:
        return self.compute_metrics(y, y_hat)

    def compute_metrics(self, y: torch.Tensor, y_hat: torch.Tensor) -> dict:
        results = {}
        for name, metric in self.metrics.items():
            results[f"{self.phase}_{name}"] = round(float(metric(y_hat, y)), 4)
        return results


class ClassificationMetrics(BaseMetrics):
    def _initialize_metrics(self):
        self.metrics = {
            "accuracy": BinaryAccuracy(),
            "f1": BinaryF1Score(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
            "auc": BinaryAUROC(),
        }


class RegressionMetrics(BaseMetrics):
    def _initialize_metrics(self):
        self.metrics = {
            "r2": R2Score(),
            "mae": MeanAbsoluteError(),
            "mse": MeanSquaredError(),
            "rmse": MeanSquaredError(squared=False),
        }


class MetricsFactory:
    @staticmethod
    def create_metrics(
        task: Literal["classification", "regression"],
        phase: Literal["train", "val", "test"],
    ) -> BaseMetrics:
        if task == "classification":
            return ClassificationMetrics(phase)
        elif task == "regression":
            return RegressionMetrics(phase)
        else:
            raise ValueError(f"Task type {task} not supported.")