import torch
import numpy as np
from typing import Literal
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, precision_score, 
                           recall_score, roc_auc_score, mean_squared_error, r2_score, root_mean_squared_error)

class BaseMetrics:
    def __init__(self, phase: Literal["train", "val", "test"]):
        self.phase = phase
    
    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor) -> dict:
        # Shared numpy conversion
        y = self._tensor_to_numpy(y)
        y_hat = self._tensor_to_numpy(y_hat)
        return self.compute_metrics(y, y_hat)
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    def compute_metrics(self, y: np.ndarray, y_hat: np.ndarray) -> dict:
        raise NotImplementedError

class ClassificationMetrics(BaseMetrics):
    def compute_metrics(self, y: np.ndarray, y_hat: np.ndarray) -> dict:
        y_hat_thresh = (y_hat > 0.5).astype(int)
        metrics = {
            f"{self.phase}_accuracy": accuracy_score(y, y_hat_thresh),
            f"{self.phase}_f1": f1_score(y, y_hat_thresh),
            f"{self.phase}_precision": precision_score(y, y_hat_thresh, zero_division=0),
            f"{self.phase}_recall": recall_score(y, y_hat_thresh),
            f"{self.phase}_auc": roc_auc_score(y, y_hat) # use raw probabilities for AUC
        }
        return {k: round(v, 4) for k, v in metrics.items()}

class RegressionMetrics(BaseMetrics):
    def compute_metrics(self, y: np.ndarray, y_hat: np.ndarray) -> dict:
        metrics = {
            f"{self.phase}_r2": r2_score(y, y_hat),
            f"{self.phase}_msa": mean_absolute_error(y, y_hat),
            f"{self.phase}_mse": mean_squared_error(y, y_hat),
            f"{self.phase}_rmse": root_mean_squared_error(y, y_hat),
        }
        return {k: round(v, 4) for k, v in metrics.items()}

class MetricsFactory:
    @staticmethod
    def create_metrics(task: Literal["classification", "regression"], 
                      phase: Literal["train", "val", "test"]) -> BaseMetrics:
        # TODO: Add not implemented for other tasks
        if task == "classification":
            return ClassificationMetrics(phase)
        return RegressionMetrics(phase)