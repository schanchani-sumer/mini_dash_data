"""
Brier Skill Score metric for binary and multiclass classification.

This module provides a TorchMetrics-based implementation of Brier Skill Score
for evaluating probabilistic predictions.
"""

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.regression import R2Score


class BrierSkillScore(Metric):
    """
    Brier Skill Score metric for multiclass classification problems.

    Decomposes multiclass problem into M binary classifications and computes:
    BSS = 1 - sum_over_M(SS_res_i) / sum_over_M(SS_res_i / (1 - R²_i))

    Where:
    - SS_res_i: Sum of squared residuals for class i
    - R²_i: Coefficient of determination for class i
    """

    def __init__(self, num_classes: int, eps: float = 1e-8, **kwargs) -> None:  # noqa: ANN003
        """
        Args:
            num_classes: Number of classes in the multiclass problem
            eps: Small constant to avoid division by zero
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.eps = eps

        if num_classes == 1:
            # For binary classification, just use R2Score
            self.r2_metric = R2Score(**kwargs)
        else:
            # For multiclass, we need to track statistics for each binary decomposition
            self.r2_metrics = torch.nn.ModuleList([R2Score(**kwargs) for _ in range(num_classes)])

            # Add states for accumulating sum of squared residuals
            self.add_state("residuals_squared_sum", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update metric states with predictions and targets.

        Args:
            preds: Predicted probabilities - shape (N, M) for multiclass or (N,) for binary
            target: True class labels - shape (N,) with values in [0, M-1] for multiclass or [0, 1] for binary
        """
        # Handle binary classification case (num_classes=1)
        if self.num_classes == 1:
            if preds.dim() == 1:
                y_pred_binary = preds
            elif preds.dim() == 2 and preds.size(1) == 1:
                y_pred_binary = preds.squeeze(1)
            else:
                raise ValueError("For binary classification, preds must have shape (N,) or (N, 1)")

            y_true_binary = target.float()
            self.r2_metric.update(y_pred_binary, y_true_binary)

        else:
            # Multiclass case
            batch_size = preds.size(0)

            if preds.dim() != 2 or preds.size(1) != self.num_classes:
                raise ValueError(f"For multiclass, preds must have shape (N, {self.num_classes})")

            if target.dim() != 1 or target.size(0) != batch_size:
                raise ValueError("target must have shape (N,) matching preds batch size")

            # Convert targets to one-hot encoding
            targets_onehot = torch.zeros_like(preds)
            targets_onehot.scatter_(1, target.unsqueeze(1).long(), 1)

            # Update each binary R2 metric and accumulate residuals
            for class_idx in range(self.num_classes):
                y_true_binary = targets_onehot[:, class_idx]
                y_pred_binary = preds[:, class_idx]

                # Update R2 metric for this class
                self.r2_metrics[class_idx].update(y_pred_binary, y_true_binary)

                # Accumulate sum of squared residuals
                residuals = y_true_binary - y_pred_binary
                self.residuals_squared_sum[class_idx] += torch.sum(residuals**2)

    def compute(self) -> Tensor:
        """
        Compute the final Brier Skill Score from accumulated states.

        Returns:
            Brier Skill Score as a scalar tensor (R² for binary classification)
        """
        # Special case: binary classification (num_classes=1) - return R²
        if self.num_classes == 1:
            return self.r2_metric.compute()

        # Multiclass case: compute BSS using torchmetrics R2Score for each class
        total_ss_res = torch.tensor(
            0.0, device=self.residuals_squared_sum.device, dtype=self.residuals_squared_sum.dtype
        )
        total_normalized_ss_res = torch.tensor(
            0.0, device=self.residuals_squared_sum.device, dtype=self.residuals_squared_sum.dtype
        )

        # Process each class
        for class_idx in range(self.num_classes):
            # Get sum of squared residuals for this class
            ss_res = self.residuals_squared_sum[class_idx]

            # Get R² from torchmetrics
            r_squared = self.r2_metrics[class_idx].compute()

            # Accumulate terms for BSS calculation
            total_ss_res += ss_res
            total_normalized_ss_res += ss_res / (1 - r_squared + self.eps)

        # Compute Brier Skill Score
        bss = 1 - (total_ss_res / (total_normalized_ss_res + self.eps))

        return bss

    def reset(self) -> None:
        """
        Reset all metric states.
        """
        if self.num_classes == 1:
            self.r2_metric.reset()
        else:
            for r2_metric in self.r2_metrics:
                r2_metric.reset()
            self.residuals_squared_sum.zero_()
