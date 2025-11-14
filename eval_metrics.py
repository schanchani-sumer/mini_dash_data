"""
Standalone evaluation metrics module for computing R2 and Brier Skill Score.

This module provides metric calculation functions without package dependencies.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import torch
from sklearn.metrics import r2_score

from brier_skill_score import BrierSkillScore


def calculate_r2(values1: List[float], values2: List[float]) -> float:
    """Calculate R² score between two continuous value lists."""
    if len(values1) < 2 or len(values2) < 2 or len(values1) != len(values2):
        return np.nan
    try:
        return float(r2_score(values1, values2))
    except Exception:
        return np.nan


def calculate_brier_skill_score(
    y_true: List,
    y_pred: List,
    is_binary: bool,
    baseline_freq: Optional[float | Dict[str, float]] = None,
    y_pred_probs: Optional[List[Dict]] = None,
) -> float:
    """
    Calculate Brier Skill Score using torch-based implementation.

    For binary: BSS = R²
    For multiclass: BSS = 1 - sum(SS_res_i) / sum(SS_res_i / (1 - R²_i))

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (used only if y_pred_probs is None)
        is_binary: Whether this is a binary classification
        baseline_freq: Baseline frequency (currently unused, kept for API compatibility)
        y_pred_probs: Optional probability distributions (dict for each prediction)
                     For binary: {'true': 0.7, 'false': 0.3}
                     For categorical: {'class1': 0.5, 'class2': 0.3, 'class3': 0.2}

    Returns:
        Brier Skill Score
    """
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
        return np.nan

    try:
        # Determine number of classes
        num_classes = 1 if is_binary else len(set(y_true))

        # Create BSS metric
        bss_metric = BrierSkillScore(num_classes=num_classes)

        if is_binary:
            # Convert targets to torch tensors
            targets = torch.tensor([1 if val else 0 for val in y_true], dtype=torch.long)

            # Use probabilities if available, otherwise fall back to hard predictions
            if y_pred_probs is not None:
                # Extract probability of positive class (True)
                preds = torch.tensor([
                    float(prob_dict.get('true', prob_dict.get(True, 1.0 if pred else 0.0)))
                    for prob_dict, pred in zip(y_pred_probs, y_pred)
                ], dtype=torch.float32)
            else:
                # Fall back to hard predictions (0 or 1)
                preds = torch.tensor([1.0 if val else 0.0 for val in y_pred], dtype=torch.float32)
        else:
            # For categorical, encode labels to integers
            # Convert all to strings to handle mixed types (e.g., 1, 2, 3, "4+")
            y_true_str = [str(val) for val in y_true]
            y_pred_str = [str(val) for val in y_pred]
            unique_labels = sorted(set(y_true_str))
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

            targets = torch.tensor([label_to_idx[val] for val in y_true_str], dtype=torch.long)

            # Use probabilities if available, otherwise fall back to hard predictions
            if y_pred_probs is not None:
                # Build probability matrix from probability dictionaries
                preds = torch.zeros((len(y_pred_str), num_classes), dtype=torch.float32)
                for i, prob_dict in enumerate(y_pred_probs):
                    for label, prob in prob_dict.items():
                        label_str = str(label)
                        if label_str in label_to_idx:
                            preds[i, label_to_idx[label_str]] = float(prob)
            else:
                # Fall back to one-hot encoding (hard predictions)
                preds = torch.zeros((len(y_pred_str), num_classes), dtype=torch.float32)
                for i, val in enumerate(y_pred_str):
                    if val in label_to_idx:
                        preds[i, label_to_idx[val]] = 1.0

        # Update and compute
        bss_metric.update(preds, targets)
        bss_value = bss_metric.compute()

        return float(bss_value.item())
    except Exception as e:
        # For debugging, you might want to log the exception
        print(f"Error calculating BSS: {e}")
        import traceback
        traceback.print_exc()
        return np.nan


def compute_baseline_frequencies(gt_values: List, attr_type: str) -> Optional[float | Dict]:
    """
    Compute baseline class frequencies for BSS calculation.

    Args:
        gt_values: List of ground truth values
        attr_type: "boolean" or "categorical"

    Returns:
        Float for boolean (positive class frequency), Dict for categorical (class -> freq)
    """
    if len(gt_values) == 0:
        return None

    if attr_type == "boolean":
        return sum(1 for v in gt_values if v) / len(gt_values)
    elif attr_type == "categorical":
        from collections import Counter
        class_counts = Counter(gt_values)
        total = len(gt_values)
        return {cls: count / total for cls, count in class_counts.items()}

    return None


def determine_attribute_type(attr_name: str, gt_value: Any, cv_value: Any) -> str:
    """
    Determine if an attribute is boolean, categorical, or continuous.

    Args:
        attr_name: Attribute name
        gt_value: Ground truth value
        cv_value: CV predicted value

    Returns:
        "boolean", "categorical", or "continuous"
    """
    # Check if boolean FIRST (since bool is a subclass of int in Python)
    if isinstance(gt_value, bool) or isinstance(cv_value, bool):
        return "boolean"

    # Check if boolean-like strings
    if gt_value in [True, False, "true", "false"] and cv_value in [True, False, "true", "false"]:
        return "boolean"

    # Check if continuous (numeric, but not bool)
    if isinstance(gt_value, (int, float)) and isinstance(cv_value, (int, float)):
        return "continuous"

    # Otherwise categorical
    return "categorical"
