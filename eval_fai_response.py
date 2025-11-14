"""
FAI API response evaluation - compare GT vs CV predictions.

This module evaluates FAI API predictions by comparing ground truth (GT) and
computer vision (CV) responses for various football attributes.

Applies applicability filtering based on fai_target_metadata to only evaluate
predictions that are applicable for the given play/player context.
"""

import json
import warnings
import yaml
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

import numpy as np
import polars as pl
import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.preprocessing import LabelBinarizer

# Import BSS from our custom implementation
from .brier_skill_score import BrierSkillScore

# Suppress F-score warnings for rare events
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Import applicability filtering
try:
    from .fai_metadata import (
        get_applicable_frame_attributes,
        get_applicable_player_frame_attributes
    )
except ImportError:
    # Fallback if import fails
    def get_applicable_frame_attributes(play_type, catalog="sbx"):
        return set()  # Return empty set to include all attributes
    def get_applicable_player_frame_attributes(role, player_attrs, catalog="sbx"):
        return set()  # Return empty set to include all attributes


# ============================================================================
# METRIC CALCULATION FUNCTIONS
# ============================================================================


def calculate_accuracy(values1: List, values2: List) -> float:
    """Calculate accuracy between two value lists."""
    if len(values1) == 0 or len(values2) == 0 or len(values1) != len(values2):
        return np.nan
    return float(accuracy_score(values1, values2))


def calculate_f1(values1: List, values2: List, is_binary: bool) -> float:
    """Calculate F1 score between two value lists."""
    if len(values1) == 0 or len(values2) == 0 or len(values1) != len(values2):
        return np.nan
    average = None if is_binary else "weighted"
    try:
        return float(f1_score(values1, values2, average=average, zero_division=0))
    except Exception:
        return np.nan


def calculate_brier_skill_score(
    y_true: List,
    y_pred: List,
    is_binary: bool,
    baseline_freq: Optional[float | Dict[str, float]] = None,
) -> float:
    """
    Calculate Brier Skill Score using torch-based implementation.

    For binary: BSS = R²
    For multiclass: BSS = 1 - sum(SS_res_i) / sum(SS_res_i / (1 - R²_i))

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        is_binary: Whether this is a binary classification
        baseline_freq: Baseline frequency (currently unused, kept for API compatibility)

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
            # Convert to torch tensors
            preds = torch.tensor([1.0 if val else 0.0 for val in y_pred], dtype=torch.float32)
            targets = torch.tensor([1 if val else 0 for val in y_true], dtype=torch.long)
        else:
            # For categorical, encode labels to integers
            unique_labels = sorted(set(y_true))
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

            targets = torch.tensor([label_to_idx[val] for val in y_true], dtype=torch.long)
            # For predictions, use one-hot encoding (hard predictions)
            preds = torch.zeros((len(y_pred), num_classes), dtype=torch.float32)
            for i, val in enumerate(y_pred):
                if val in label_to_idx:
                    preds[i, label_to_idx[val]] = 1.0

        # Update and compute
        bss_metric.update(preds, targets)
        bss_value = bss_metric.compute()

        return float(bss_value.item())
    except Exception as e:
        # For debugging, you might want to log the exception
        return np.nan


def calculate_r2(values1: List[float], values2: List[float]) -> float:
    """Calculate R² score between two continuous value lists."""
    if len(values1) < 2 or len(values2) < 2 or len(values1) != len(values2):
        return np.nan
    try:
        return float(r2_score(values1, values2))
    except Exception:
        return np.nan


def calculate_relative_tolerance_accuracy(
    y_true: List[float],
    y_pred: List[float],
    relative_tolerance: float = 0.10
) -> float:
    """
    Calculate percentage of predictions within relative tolerance of true values.

    Uses adaptive tolerance that scales with the magnitude of each true value:
    - tolerance = max(|true_value| * relative_tolerance, min_threshold)
    - min_threshold is 10th percentile of non-zero absolute values

    Args:
        y_true: Ground truth continuous values
        y_pred: Predicted continuous values
        relative_tolerance: Fraction of true value to use as tolerance (default: 0.10 for 10%)

    Returns:
        Proportion of predictions within adaptive tolerance (0-1 scale)
    """
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
        return np.nan

    try:
        # Compute minimum threshold from label distribution (10th percentile of non-zero values)
        abs_values = np.abs(y_true)
        non_zero_values = abs_values[abs_values > 0]
        min_threshold = np.quantile(non_zero_values, 0.1) if len(non_zero_values) > 0 else 0.1

        # Apply relative tolerance with minimum threshold for each value
        tolerances = np.maximum(abs_values * relative_tolerance, min_threshold)
        errors = np.abs(np.array(y_true) - np.array(y_pred))
        within_tolerance = (errors <= tolerances).mean()

        return float(within_tolerance)
    except Exception:
        return np.nan


def calculate_null_accuracy(values1: List, values2: List) -> float:
    """
    Calculate accuracy of null predictions (matching nulledness between sources).

    Returns the proportion of cases where both are null OR both are not null.
    """
    if len(values1) == 0 or len(values2) == 0 or len(values1) != len(values2):
        return np.nan

    matching = sum(1 for v1, v2 in zip(values1, values2) if (v1 is None) == (v2 is None))
    return float(matching / len(values1))


def calculate_kl_divergence(
    gt_probs_list: List[Dict[str, float]],
    cv_probs_list: List[Dict[str, float]],
    epsilon: float = 1e-10
) -> float:
    """
    Calculate KL divergence between ground truth and CV probability distributions.

    Only applicable to boolean and categorical attributes with probability distributions.

    KL(GT || CV) = sum(p_gt * log(p_gt / p_cv)) where p_gt is GT probability

    Args:
        gt_probs_list: List of GT probability dictionaries (one per prediction)
        cv_probs_list: List of CV probability dictionaries (one per prediction)
        epsilon: Small constant to avoid log(0)

    Returns:
        Average KL divergence across all predictions
    """
    if len(gt_probs_list) == 0 or len(cv_probs_list) == 0:
        return np.nan

    if len(gt_probs_list) != len(cv_probs_list):
        return np.nan

    kl_values = []

    for gt_probs, cv_probs in zip(gt_probs_list, cv_probs_list):
        if not gt_probs or not cv_probs:
            continue

        # Get all possible classes from both distributions
        all_classes = set(gt_probs.keys()).union(set(cv_probs.keys()))

        kl_sum = 0.0
        for cls in all_classes:
            p_gt = gt_probs.get(cls, 0.0)
            p_cv = cv_probs.get(cls, 0.0)

            # Add epsilon to avoid log(0)
            p_gt = max(p_gt, epsilon)
            p_cv = max(p_cv, epsilon)

            # KL divergence: p * log(p / q)
            kl_sum += p_gt * np.log(p_gt / p_cv)

        kl_values.append(kl_sum)

    if len(kl_values) == 0:
        return np.nan

    return float(np.mean(kl_values))


def safe_round(value: float | None, decimals: int = 3) -> float | None:
    """Safely round a value, returning None if value is None or NaN."""
    if value is None or np.isnan(value):
        return None
    return round(value, decimals)


# ============================================================================
# PLAYER_FRAME APPLICABILITY RULES
# ============================================================================


def load_player_frame_applicability_rules() -> Dict[str, Set[str]]:
    """
    Load player_frame applicability rules from YAML config.

    Maps each player_frame attribute to the set of applicable roles.
    Based on comments in frame_targets_reference.yaml.

    Returns:
        Dict mapping attribute name -> set of applicable roles
    """
    # Hard-coded mapping based on frame_targets_reference.yaml comments
    # This could be loaded from a separate config file if needed
    return {
        # General (no applicability - all roles)
        "grade_numerical": set(),  # Empty set means all roles

        # QB/skill (PASS, PASS ROUTE, RUN)
        "interception": {"PASS", "PASS ROUTE", "RUN"},
        "touchdown": {"PASS", "PASS ROUTE", "RUN"},

        # Rushing (RUN)
        "rush_yards": {"RUN"},
        "rush_success": {"RUN"},

        # All defense (RUN DEFENSE, COVERAGE, PASS RUSH)
        "tackle": {"RUN DEFENSE", "COVERAGE", "PASS RUSH"},
        "first_contact": {"RUN DEFENSE", "COVERAGE", "PASS RUSH"},
        "stop": {"RUN DEFENSE", "COVERAGE", "PASS RUSH"},

        # Pass defense (COVERAGE, PASS RUSH)
        "interception_caught": {"COVERAGE", "PASS RUSH"},

        # Pass block (PASS BLOCK)
        "sack_allowed": {"PASS BLOCK"},
        "pressure_allowed": {"PASS BLOCK"},

        # Pass rush (PASS RUSH)
        "sack": {"PASS RUSH"},
        "pressure": {"PASS RUSH"},

        # Coverage (COVERAGE)
        "primary_coverage": {"COVERAGE"},
        "pbu": {"COVERAGE"},

        # Route running (PASS ROUTE)
        "target": {"PASS ROUTE"},
        "route": {"PASS ROUTE"},

        # Reception - requires both role AND custom condition (handled separately)
        "reception": {"PASS ROUTE"},  # + target == true
        "rec_yards": {"PASS ROUTE"},  # + reception == true
        "yards_after_catch": {"PASS ROUTE"},  # + reception == true
    }


def extract_player_roles_from_player_play(
    player_play_preds: Dict[str, Dict[str, Any]]
) -> Dict[str, str]:
    """
    Extract role information from player_play predictions.

    Args:
        player_play_preds: Dict of track_id -> {attribute -> value}

    Returns:
        Dict of track_id -> role
    """
    player_roles = {}
    for track_id, attrs in player_play_preds.items():
        if "role" in attrs:
            player_roles[track_id] = attrs["role"]
    return player_roles


def is_player_frame_prediction_applicable(
    attr_name: str,
    role: str,
    player_play_attrs: Dict[str, Any],
    applicability_rules: Dict[str, Set[str]]
) -> bool:
    """
    Check if a player_frame prediction is applicable given the player's role.

    Args:
        attr_name: Player_frame attribute name (e.g., "rush_yards")
        role: Player's role from player_play level (e.g., "RUN", "COVERAGE")
        player_play_attrs: All player_play attributes for this player
        applicability_rules: Mapping of attribute -> set of applicable roles

    Returns:
        True if prediction is applicable, False otherwise
    """
    # Get applicable roles for this attribute
    applicable_roles = applicability_rules.get(attr_name, None)

    # If attribute not in rules, assume it's applicable to all
    if applicable_roles is None:
        return True

    # Empty set means applicable to all roles
    if len(applicable_roles) == 0:
        return True

    # Check role-based applicability
    if role not in applicable_roles:
        return False

    # Additional custom conditions for reception-related attributes
    if attr_name == "reception":
        # Requires target == true
        target = player_play_attrs.get("target")
        return target is True

    if attr_name in ["rec_yards", "yards_after_catch"]:
        # Requires reception == true
        reception = player_play_attrs.get("reception")
        return reception is True

    return True


# ============================================================================
# FAI RESPONSE PARSING
# ============================================================================


def load_fai_responses(output_dir: Path) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Load GT and CV FAI response JSON files from output directory.

    Args:
        output_dir: Path to output directory containing fai_response_gt.json and fai_response_cv.json

    Returns:
        Tuple of (gt_response, cv_response) dictionaries, or (None, None) if files not found
    """
    gt_path = output_dir / "fai_response_gt.json"
    cv_path = output_dir / "fai_response_cv.json"

    if not gt_path.exists() or not cv_path.exists():
        print(f"  ⚠ FAI response files not found in {output_dir}")
        return None, None

    try:
        with open(gt_path, 'r') as f:
            gt_response = json.load(f)
        with open(cv_path, 'r') as f:
            cv_response = json.load(f)
        return gt_response, cv_response
    except Exception as e:
        print(f"  ✗ Error loading FAI responses: {e}")
        return None, None


def parse_play_predictions(response: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """
    Parse play-level predictions from FAI response.

    Args:
        response: FAI API response dictionary

    Returns:
        Tuple of (predictions dict, probabilities dict):
            - predictions: attribute name -> predicted value (argmax for bool/cat, value for continuous)
            - probabilities: attribute name -> probability distribution dict (only for bool/cat)
    """
    if not response or "predictions" not in response:
        return {}, {}

    play_preds = response["predictions"].get("play", [])
    if not play_preds or len(play_preds) == 0:
        return {}, {}

    # Take first play prediction (should only be one)
    play_pred = play_preds[0]

    parsed = {}
    probabilities = {}

    for attr, value in play_pred.items():
        if isinstance(value, dict):
            # Boolean or categorical prediction
            if "predicted_class" in value:
                parsed[attr] = value["predicted_class"]
                # Extract probabilities if available
                if "probabilities" in value:
                    probabilities[attr] = value["probabilities"]
        else:
            # Continuous prediction (e.g., yards_gained, ep, epa)
            parsed[attr] = value

    return parsed, probabilities


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


# ============================================================================
# FAI PREDICTION EVALUATION
# ============================================================================


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
        class_counts = Counter(gt_values)
        total = len(gt_values)
        return {cls: count / total for cls, count in class_counts.items()}

    return None


def evaluate_play_predictions(
    gt_response: Dict[str, Any],
    cv_response: Dict[str, Any],
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Evaluate play-level FAI predictions by comparing GT vs CV responses.

    Calculates accuracy, BSS, R², KL divergence, and MAE as per the database schema.

    Args:
        gt_response: Ground truth FAI API response
        cv_response: Computer vision FAI API response
        output_dir: Optional directory to save detailed results

    Returns:
        Dictionary containing:
            - metrics: Dict of attribute -> detailed metrics
            - summary: Overall summary statistics with BSS, R², KL divergence
    """
    print("\n" + "="*20 + " FAI PREDICTION EVALUATION " + "="*20)

    # Parse play-level predictions (with probabilities)
    gt_preds, gt_probs = parse_play_predictions(gt_response)
    cv_preds, cv_probs = parse_play_predictions(cv_response)

    if not gt_preds or not cv_preds:
        print("  ✗ No predictions found in responses")
        return {"error": "No predictions found"}

    print(f"  GT predictions: {len(gt_preds)} attributes")
    print(f"  CV predictions: {len(cv_preds)} attributes")

    # Find common attributes
    common_attrs = set(gt_preds.keys()).intersection(set(cv_preds.keys()))
    print(f"  Common attributes: {len(common_attrs)}")

    if not common_attrs:
        print("  ✗ No common attributes found")
        return {"error": "No common attributes"}

    # Collect per-attribute metrics
    results = {}

    # Aggregators for summary statistics
    boolean_data = {"matches": [], "gt_probs": [], "cv_probs": []}
    categorical_data = {"matches": [], "gt_probs": [], "cv_probs": []}
    continuous_data = {"errors": [], "relative_errors": [], "gt_values": [], "cv_values": []}

    for attr in sorted(common_attrs):
        gt_val = gt_preds[attr]
        cv_val = cv_preds[attr]

        # Skip if either is None
        if gt_val is None or cv_val is None:
            continue

        # Determine attribute type
        attr_type = determine_attribute_type(attr, gt_val, cv_val)

        # Calculate metrics based on type
        attr_metrics = {"type": attr_type, "gt_value": gt_val, "cv_value": cv_val}

        if attr_type == "continuous":
            # For continuous: calculate error, relative error
            error = abs(gt_val - cv_val)
            relative_error = abs(error / gt_val) if gt_val != 0 else (0 if error == 0 else np.inf)

            attr_metrics["absolute_error"] = safe_round(error, 3)
            attr_metrics["relative_error"] = safe_round(relative_error, 3)
            attr_metrics["match"] = error < abs(gt_val * 0.10)  # Within 10% tolerance

            continuous_data["errors"].append(error)
            continuous_data["relative_errors"].append(relative_error if not np.isinf(relative_error) else np.nan)
            continuous_data["gt_values"].append(gt_val)
            continuous_data["cv_values"].append(cv_val)

        elif attr_type in ["boolean", "categorical"]:
            # For boolean/categorical: check match and extract probabilities for KL divergence
            match = (gt_val == cv_val)
            attr_metrics["match"] = match

            # Calculate KL divergence if probabilities available
            gt_prob_dist = gt_probs.get(attr, {})
            cv_prob_dist = cv_probs.get(attr, {})

            if gt_prob_dist and cv_prob_dist:
                kl_div = calculate_kl_divergence([gt_prob_dist], [cv_prob_dist])
                attr_metrics["kl_divergence"] = safe_round(kl_div, 4)

            if attr_type == "boolean":
                boolean_data["matches"].append(match)
                if gt_prob_dist:
                    boolean_data["gt_probs"].append(gt_prob_dist)
                if cv_prob_dist:
                    boolean_data["cv_probs"].append(cv_prob_dist)
            else:
                categorical_data["matches"].append(match)
                if gt_prob_dist:
                    categorical_data["gt_probs"].append(gt_prob_dist)
                if cv_prob_dist:
                    categorical_data["cv_probs"].append(cv_prob_dist)

        results[attr] = attr_metrics

    # Compute summary statistics
    summary = {
        "total_attributes": len(results),
        "boolean_count": len(boolean_data["matches"]),
        "categorical_count": len(categorical_data["matches"]),
        "continuous_count": len(continuous_data["errors"]),
    }

    # Boolean metrics: accuracy, BSS, KL divergence
    if boolean_data["matches"]:
        boolean_accuracy = sum(boolean_data["matches"]) / len(boolean_data["matches"])
        summary["boolean_accuracy"] = safe_round(boolean_accuracy, 3)

        # Calculate KL divergence for boolean
        if boolean_data["gt_probs"] and boolean_data["cv_probs"]:
            boolean_kl = calculate_kl_divergence(boolean_data["gt_probs"], boolean_data["cv_probs"])
            summary["boolean_kl_divergence"] = safe_round(boolean_kl, 4)

    # Categorical metrics: accuracy, KL divergence
    if categorical_data["matches"]:
        categorical_accuracy = sum(categorical_data["matches"]) / len(categorical_data["matches"])
        summary["categorical_accuracy"] = safe_round(categorical_accuracy, 3)

        # Calculate KL divergence for categorical
        if categorical_data["gt_probs"] and categorical_data["cv_probs"]:
            categorical_kl = calculate_kl_divergence(categorical_data["gt_probs"], categorical_data["cv_probs"])
            summary["categorical_kl_divergence"] = safe_round(categorical_kl, 4)

    # Continuous metrics: MAE, MRE, R²
    if continuous_data["errors"]:
        mean_abs_error = np.mean(continuous_data["errors"])
        mean_rel_error = np.mean([e for e in continuous_data["relative_errors"] if not np.isnan(e)])
        summary["continuous_mean_absolute_error"] = safe_round(mean_abs_error, 3)
        summary["continuous_mean_relative_error"] = safe_round(mean_rel_error, 3)

        # Calculate R² for continuous (only makes sense if we have more than 1 prediction)
        if len(continuous_data["gt_values"]) > 1:
            r2 = calculate_r2(continuous_data["gt_values"], continuous_data["cv_values"])
            summary["continuous_r2"] = safe_round(r2, 3)

    # Overall accuracy (for boolean + categorical only)
    total_bool_cat_matches = sum(boolean_data["matches"]) + sum(categorical_data["matches"])
    total_bool_cat = len(boolean_data["matches"]) + len(categorical_data["matches"])
    if total_bool_cat > 0:
        summary["overall_accuracy"] = safe_round(total_bool_cat_matches / total_bool_cat, 3)

    # Print summary
    print("\n  Summary:")
    print(f"    Total attributes evaluated: {summary['total_attributes']}")
    if "boolean_accuracy" in summary:
        print(f"    Boolean accuracy: {summary['boolean_accuracy']} ({summary['boolean_count']} attrs)")
        if "boolean_kl_divergence" in summary:
            print(f"    Boolean KL divergence: {summary['boolean_kl_divergence']}")
    if "categorical_accuracy" in summary:
        print(f"    Categorical accuracy: {summary['categorical_accuracy']} ({summary['categorical_count']} attrs)")
        if "categorical_kl_divergence" in summary:
            print(f"    Categorical KL divergence: {summary['categorical_kl_divergence']}")
    if "continuous_mean_absolute_error" in summary:
        print(f"    Continuous MAE: {summary['continuous_mean_absolute_error']} ({summary['continuous_count']} attrs)")
        print(f"    Continuous MRE: {summary['continuous_mean_relative_error']}")
        if "continuous_r2" in summary:
            print(f"    Continuous R²: {summary['continuous_r2']}")
    if "overall_accuracy" in summary:
        print(f"    Overall accuracy (bool+cat): {summary['overall_accuracy']}")

    # Save detailed results if output_dir provided
    if output_dir:
        output_path = Path(output_dir) / "fai_evaluation_metrics.json"
        with open(output_path, 'w') as f:
            json.dump({
                "summary": summary,
                "attribute_metrics": results,
                "boolean_data": boolean_data,
                "categorical_data": categorical_data,
                "continuous_data": continuous_data
            }, f, indent=2)
        print(f"\n  ✓ Detailed metrics saved to: {output_path}")

    print("=" * 70)

    return {
        "summary": summary,
        "metrics": results
    }


def parse_player_play_predictions(response: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Parse player_play predictions from FAI response.

    Args:
        response: FAI API response dictionary

    Returns:
        Tuple of (predictions dict, probabilities dict):
            - predictions: track_id -> {attribute -> predicted value}
            - probabilities: track_id -> {attribute -> probability distribution}
    """
    if not response or "predictions" not in response:
        return {}, {}

    player_preds = response["predictions"].get("player_play", [])
    if not player_preds:
        return {}, {}

    # Parse each player's predictions
    parsed = {}
    probabilities = {}
    for player in player_preds:
        track_id = player.get("track_id")
        if not track_id:
            continue

        player_attrs = {}
        player_probs = {}
        for attr, value in player.items():
            if attr == "track_id":
                continue

            if isinstance(value, dict):
                # Boolean or categorical prediction
                if "predicted_class" in value:
                    player_attrs[attr] = value["predicted_class"]
                    # Extract probabilities if available
                    if "probabilities" in value:
                        player_probs[attr] = value["probabilities"]
            else:
                # Continuous prediction
                player_attrs[attr] = value

        parsed[track_id] = player_attrs
        if player_probs:
            probabilities[track_id] = player_probs

    return parsed, probabilities


def parse_frame_predictions(response: Dict[str, Any]) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Dict[str, float]]]]:
    """
    Parse frame-level predictions from FAI response.

    Args:
        response: FAI API response dictionary

    Returns:
        Tuple of (predictions dict, probabilities dict):
            - predictions: frame_id -> {attribute -> predicted value}
            - probabilities: frame_id -> {attribute -> probability distribution}
    """
    if not response or "predictions" not in response:
        return {}, {}

    frame_preds = response["predictions"].get("frame", [])
    if not frame_preds:
        return {}, {}

    # Parse each frame's predictions
    parsed = {}
    probabilities = {}

    for frame in frame_preds:
        frame_id = frame.get("frame_id")
        if frame_id is None:  # frame_id can be 0
            continue

        frame_attrs = {}
        frame_probs = {}
        for attr, value in frame.items():
            if attr == "frame_id":
                continue

            if isinstance(value, dict):
                # Boolean or categorical prediction
                if "predicted_class" in value:
                    frame_attrs[attr] = value["predicted_class"]
                    # Extract probabilities if available
                    if "probabilities" in value:
                        frame_probs[attr] = value["probabilities"]
            else:
                # Continuous prediction
                frame_attrs[attr] = value

        parsed[frame_id] = frame_attrs
        if frame_probs:
            probabilities[frame_id] = frame_probs

    return parsed, probabilities


def parse_player_frame_predictions(response: Dict[str, Any]) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], Dict[Tuple[str, int], Dict[str, Dict[str, float]]]]:
    """
    Parse player_frame predictions from FAI response.

    Args:
        response: FAI API response dictionary

    Returns:
        Tuple of (predictions dict, probabilities dict):
            - predictions: (track_id, frame_id) -> {attribute -> predicted value}
            - probabilities: (track_id, frame_id) -> {attribute -> probability distribution}
    """
    if not response or "predictions" not in response:
        return {}, {}

    player_frame_preds = response["predictions"].get("player_frame", [])
    if not player_frame_preds:
        return {}, {}

    # Parse each player-frame's predictions
    parsed = {}
    probabilities = {}

    for pf in player_frame_preds:
        track_id = pf.get("track_id")
        frame_id = pf.get("frame_id")

        if track_id is None or frame_id is None:
            continue

        key = (track_id, frame_id)
        pf_attrs = {}
        pf_probs = {}

        for attr, value in pf.items():
            if attr in ["track_id", "frame_id"]:
                continue

            if isinstance(value, dict):
                # Boolean or categorical prediction
                if "predicted_class" in value:
                    pf_attrs[attr] = value["predicted_class"]
                    # Extract probabilities if available
                    if "probabilities" in value:
                        pf_probs[attr] = value["probabilities"]
            else:
                # Continuous prediction
                pf_attrs[attr] = value

        parsed[key] = pf_attrs
        if pf_probs:
            probabilities[key] = pf_probs

    return parsed, probabilities


def evaluate_player_play_predictions(
    gt_response: Dict[str, Any],
    cv_response: Dict[str, Any],
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Evaluate player_play predictions by comparing GT vs CV responses.

    Args:
        gt_response: Ground truth FAI API response
        cv_response: Computer vision FAI API response
        output_dir: Optional directory to save detailed results

    Returns:
        Dictionary containing:
            - per_player_metrics: Dict of track_id -> per-player metrics
            - per_attribute_metrics: Dict of attribute -> aggregated metrics
            - summary: Overall summary statistics
    """
    print("\n" + "="*20 + " PLAYER_PLAY PREDICTION EVALUATION " + "="*20)

    # Parse player_play predictions (with probabilities)
    gt_players, gt_player_probs = parse_player_play_predictions(gt_response)
    cv_players, cv_player_probs = parse_player_play_predictions(cv_response)

    if not gt_players or not cv_players:
        print("  ✗ No player_play predictions found in responses")
        return {"error": "No player_play predictions found"}

    print(f"  GT players: {len(gt_players)}")
    print(f"  CV players: {len(cv_players)}")

    # Find common track IDs
    common_tracks = set(gt_players.keys()).intersection(set(cv_players.keys()))
    print(f"  Common players: {len(common_tracks)}")

    if not common_tracks:
        print("  ✗ No common players found")
        return {"error": "No common players"}

    # Collect metrics per player and per attribute
    per_player_metrics = {}
    attribute_aggregator = {}  # attr -> {gt_values, cv_values, matches, errors, gt_probs, cv_probs}

    for track_id in sorted(common_tracks):
        gt_attrs = gt_players[track_id]
        cv_attrs = cv_players[track_id]
        gt_probs = gt_player_probs.get(track_id, {})
        cv_probs = cv_player_probs.get(track_id, {})

        # Find common attributes for this player
        common_attrs = set(gt_attrs.keys()).intersection(set(cv_attrs.keys()))

        player_metrics = {
            "total_attributes": 0,
            "boolean_correct": 0,
            "categorical_correct": 0,
            "continuous_errors": [],
            "attributes": {}
        }

        for attr in common_attrs:
            gt_val = gt_attrs[attr]
            cv_val = cv_attrs[attr]

            # Skip if either is None
            if gt_val is None or cv_val is None:
                continue

            # Initialize attribute aggregator
            if attr not in attribute_aggregator:
                attribute_aggregator[attr] = {
                    "gt_values": [],
                    "cv_values": [],
                    "matches": [],
                    "errors": [],
                    "gt_probs": [],
                    "cv_probs": []
                }

            # Determine attribute type
            attr_type = determine_attribute_type(attr, gt_val, cv_val)

            attr_metrics = {"type": attr_type, "gt_value": gt_val, "cv_value": cv_val}

            if attr_type == "continuous":
                error = abs(gt_val - cv_val)
                relative_error = abs(error / gt_val) if gt_val != 0 else (0 if error == 0 else np.inf)

                attr_metrics["absolute_error"] = safe_round(error, 3)
                attr_metrics["relative_error"] = safe_round(relative_error, 3)
                match = error < abs(gt_val * 0.10)
                attr_metrics["match"] = match

                player_metrics["continuous_errors"].append(error)
                attribute_aggregator[attr]["errors"].append(error)
                attribute_aggregator[attr]["gt_values"].append(gt_val)
                attribute_aggregator[attr]["cv_values"].append(cv_val)

            elif attr_type == "boolean":
                match = (gt_val == cv_val)
                attr_metrics["match"] = match
                if match:
                    player_metrics["boolean_correct"] += 1

                attribute_aggregator[attr]["matches"].append(match)
                attribute_aggregator[attr]["gt_values"].append(gt_val)
                attribute_aggregator[attr]["cv_values"].append(cv_val)

                # Collect probabilities for KL divergence
                if attr in gt_probs:
                    attribute_aggregator[attr]["gt_probs"].append(gt_probs[attr])
                if attr in cv_probs:
                    attribute_aggregator[attr]["cv_probs"].append(cv_probs[attr])

            elif attr_type == "categorical":
                match = (gt_val == cv_val)
                attr_metrics["match"] = match
                if match:
                    player_metrics["categorical_correct"] += 1

                attribute_aggregator[attr]["matches"].append(match)
                attribute_aggregator[attr]["gt_values"].append(gt_val)
                attribute_aggregator[attr]["cv_values"].append(cv_val)

                # Collect probabilities for KL divergence
                if attr in gt_probs:
                    attribute_aggregator[attr]["gt_probs"].append(gt_probs[attr])
                if attr in cv_probs:
                    attribute_aggregator[attr]["cv_probs"].append(cv_probs[attr])

            player_metrics["total_attributes"] += 1
            player_metrics["attributes"][attr] = attr_metrics

        per_player_metrics[track_id] = player_metrics

    # Aggregate per-attribute metrics
    per_attribute_metrics = {}
    boolean_attrs = []
    categorical_attrs = []
    continuous_attrs = []

    for attr, data in attribute_aggregator.items():
        attr_summary = {}

        if data["errors"]:  # Continuous
            mean_abs_error = np.mean(data["errors"])
            gt_vals = np.array(data["gt_values"])
            cv_vals = np.array(data["cv_values"])
            rel_errors = np.abs((gt_vals - cv_vals) / np.where(gt_vals != 0, gt_vals, 1))
            mean_rel_error = np.mean(rel_errors[np.isfinite(rel_errors)])

            # Calculate R² for continuous
            r2 = calculate_r2(data["gt_values"], data["cv_values"])

            attr_summary = {
                "type": "continuous",
                "count": len(data["errors"]),
                "mean_absolute_error": safe_round(mean_abs_error, 3),
                "mean_relative_error": safe_round(mean_rel_error, 3),
                "r2_score": safe_round(r2, 3)
            }
            continuous_attrs.append(attr)

        elif data["matches"]:  # Boolean or categorical
            accuracy = sum(data["matches"]) / len(data["matches"])

            # Determine if boolean or categorical
            unique_vals = set(data["gt_values"] + data["cv_values"])
            is_boolean = all(v in [True, False, "true", "false"] for v in unique_vals)

            attr_summary = {
                "type": "boolean" if is_boolean else "categorical",
                "count": len(data["matches"]),
                "accuracy": safe_round(accuracy, 3)
            }

            # Calculate KL divergence if probabilities available
            if data["gt_probs"] and data["cv_probs"]:
                kl_div = calculate_kl_divergence(data["gt_probs"], data["cv_probs"])
                attr_summary["kl_divergence"] = safe_round(kl_div, 4)

            # Calculate BSS if we have probabilities
            if data["gt_values"] and data["cv_values"]:
                attr_type = "boolean" if is_boolean else "categorical"
                baseline_freq = compute_baseline_frequencies(data["gt_values"], attr_type)
                bss = calculate_brier_skill_score(
                    data["gt_values"],
                    data["cv_values"],
                    is_binary=is_boolean,
                    baseline_freq=baseline_freq
                )
                attr_summary["brier_skill_score"] = safe_round(bss, 3)

            if is_boolean:
                boolean_attrs.append(attr)
            else:
                categorical_attrs.append(attr)

        per_attribute_metrics[attr] = attr_summary

    # Compute overall summary
    total_boolean = sum(per_attribute_metrics[a]["count"] for a in boolean_attrs)
    total_categorical = sum(per_attribute_metrics[a]["count"] for a in categorical_attrs)
    total_continuous = sum(per_attribute_metrics[a]["count"] for a in continuous_attrs)

    summary = {
        "total_players": len(common_tracks),
        "total_attributes": len(per_attribute_metrics),
        "boolean_attributes": len(boolean_attrs),
        "categorical_attributes": len(categorical_attrs),
        "continuous_attributes": len(continuous_attrs),
        "total_boolean_predictions": total_boolean,
        "total_categorical_predictions": total_categorical,
        "total_continuous_predictions": total_continuous
    }

    # Average boolean metrics: accuracy, KL divergence, BSS
    if boolean_attrs:
        bool_acc = np.mean([per_attribute_metrics[a]["accuracy"] for a in boolean_attrs])
        summary["boolean_accuracy"] = safe_round(bool_acc, 3)

        # Average KL divergence for boolean attributes
        bool_kl_values = [per_attribute_metrics[a]["kl_divergence"] for a in boolean_attrs if "kl_divergence" in per_attribute_metrics[a] and per_attribute_metrics[a]["kl_divergence"] is not None]
        if bool_kl_values:
            summary["boolean_kl_divergence"] = safe_round(np.mean(bool_kl_values), 4)

        # Average BSS for boolean attributes
        bool_bss_values = [per_attribute_metrics[a]["brier_skill_score"] for a in boolean_attrs if "brier_skill_score" in per_attribute_metrics[a] and per_attribute_metrics[a]["brier_skill_score"] is not None]
        if bool_bss_values:
            summary["boolean_bss"] = safe_round(np.mean(bool_bss_values), 3)

    # Average categorical metrics: accuracy, KL divergence, BSS
    if categorical_attrs:
        cat_acc = np.mean([per_attribute_metrics[a]["accuracy"] for a in categorical_attrs])
        summary["categorical_accuracy"] = safe_round(cat_acc, 3)

        # Average KL divergence for categorical attributes
        cat_kl_values = [per_attribute_metrics[a]["kl_divergence"] for a in categorical_attrs if "kl_divergence" in per_attribute_metrics[a] and per_attribute_metrics[a]["kl_divergence"] is not None]
        if cat_kl_values:
            summary["categorical_kl_divergence"] = safe_round(np.mean(cat_kl_values), 4)

        # Average BSS for categorical attributes
        cat_bss_values = [per_attribute_metrics[a]["brier_skill_score"] for a in categorical_attrs if "brier_skill_score" in per_attribute_metrics[a] and per_attribute_metrics[a]["brier_skill_score"] is not None]
        if cat_bss_values:
            summary["categorical_bss"] = safe_round(np.mean(cat_bss_values), 3)

    # Average continuous metrics: MAE, MRE, R²
    if continuous_attrs:
        cont_mae = np.mean([per_attribute_metrics[a]["mean_absolute_error"] for a in continuous_attrs])
        cont_mre = np.mean([per_attribute_metrics[a]["mean_relative_error"] for a in continuous_attrs if per_attribute_metrics[a]["mean_relative_error"] is not None])
        summary["continuous_mean_absolute_error"] = safe_round(cont_mae, 3)
        summary["continuous_mean_relative_error"] = safe_round(cont_mre, 3)

        # Average R² for continuous attributes
        cont_r2_values = [per_attribute_metrics[a]["r2_score"] for a in continuous_attrs if "r2_score" in per_attribute_metrics[a] and per_attribute_metrics[a]["r2_score"] is not None]
        if cont_r2_values:
            summary["continuous_r2"] = safe_round(np.mean(cont_r2_values), 3)

    # Overall accuracy (bool + cat)
    if boolean_attrs or categorical_attrs:
        all_bool_cat_attrs = boolean_attrs + categorical_attrs
        overall_acc = np.mean([per_attribute_metrics[a]["accuracy"] for a in all_bool_cat_attrs])
        summary["overall_accuracy"] = safe_round(overall_acc, 3)

    print(f"\n  Summary:")
    print(f"    Total players evaluated: {summary['total_players']}")
    print(f"    Total attributes: {summary['total_attributes']}")
    if "boolean_accuracy" in summary:
        print(f"    Boolean accuracy: {summary['boolean_accuracy']} ({summary['boolean_attributes']} attrs)")
        if "boolean_kl_divergence" in summary:
            print(f"    Boolean KL divergence: {summary['boolean_kl_divergence']}")
        if "boolean_bss" in summary:
            print(f"    Boolean BSS: {summary['boolean_bss']}")
    if "categorical_accuracy" in summary:
        print(f"    Categorical accuracy: {summary['categorical_accuracy']} ({summary['categorical_attributes']} attrs)")
        if "categorical_kl_divergence" in summary:
            print(f"    Categorical KL divergence: {summary['categorical_kl_divergence']}")
        if "categorical_bss" in summary:
            print(f"    Categorical BSS: {summary['categorical_bss']}")
    if "continuous_mean_absolute_error" in summary:
        print(f"    Continuous MAE: {summary['continuous_mean_absolute_error']} ({summary['continuous_attributes']} attrs)")
        print(f"    Continuous MRE: {summary['continuous_mean_relative_error']}")
        if "continuous_r2" in summary:
            print(f"    Continuous R²: {summary['continuous_r2']}")
    if "overall_accuracy" in summary:
        print(f"    Overall accuracy (bool+cat): {summary['overall_accuracy']}")

    # Save detailed results if output_dir provided
    if output_dir:
        output_path = Path(output_dir) / "fai_player_play_evaluation_metrics.json"
        with open(output_path, 'w') as f:
            json.dump({
                "summary": summary,
                "per_attribute_metrics": per_attribute_metrics,
                "per_player_metrics": per_player_metrics
            }, f, indent=2)
        print(f"\n  ✓ Detailed player_play metrics saved to: {output_path}")

    print("=" * 70)

    return {
        "summary": summary,
        "per_attribute_metrics": per_attribute_metrics,
        "per_player_metrics": per_player_metrics
    }


def evaluate_frame_predictions(
    gt_response: Dict[str, Any],
    cv_response: Dict[str, Any],
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Evaluate frame-level predictions by comparing GT vs CV responses.

    Args:
        gt_response: Ground truth FAI API response
        cv_response: Computer vision FAI API response
        output_dir: Optional directory to save detailed results

    Returns:
        Dictionary containing:
            - per_frame_metrics: Dict of frame_id -> per-frame metrics
            - per_attribute_metrics: Dict of attribute -> aggregated metrics across frames
            - summary: Overall summary statistics
    """
    print("\n" + "="*20 + " FRAME-LEVEL PREDICTION EVALUATION " + "="*20)

    # Parse frame predictions
    gt_frames, gt_probs = parse_frame_predictions(gt_response)
    cv_frames, cv_probs = parse_frame_predictions(cv_response)

    if not gt_frames or not cv_frames:
        print("  ✗ No frame predictions found in responses")
        return {"error": "No frame predictions found"}

    print(f"  GT frames: {len(gt_frames)}")
    print(f"  CV frames: {len(cv_frames)}")

    # Apply applicability filtering based on play context
    # Extract play_type from play-level predictions
    play_type = None
    if gt_response and "predictions" in gt_response:
        play_preds = gt_response["predictions"].get("play", [])
        if play_preds and len(play_preds) > 0:
            play_type_pred = play_preds[0].get("play_type_intent")
            if isinstance(play_type_pred, dict):
                play_type = play_type_pred.get("predicted_class")
            else:
                play_type = play_type_pred

    # Get applicable frame attributes for this play type
    applicable_attrs = get_applicable_frame_attributes(play_type, catalog="sbx")

    # Filter frames to only include applicable attributes
    if applicable_attrs:
        print(f"  Applying applicability filtering (play_type={play_type})...")
        filtered_gt_frames = {}
        filtered_cv_frames = {}

        for frame_id, attrs in gt_frames.items():
            filtered_gt_frames[frame_id] = {k: v for k, v in attrs.items() if k in applicable_attrs}

        for frame_id, attrs in cv_frames.items():
            filtered_cv_frames[frame_id] = {k: v for k, v in attrs.items() if k in applicable_attrs}

        gt_frames = filtered_gt_frames
        cv_frames = filtered_cv_frames
        print(f"  Kept {len(applicable_attrs)} applicable attributes")

    # Find common frame IDs
    common_frames = set(gt_frames.keys()).intersection(set(cv_frames.keys()))
    print(f"  Common frames: {len(common_frames)}")

    if not common_frames:
        print("  ✗ No common frames found")
        return {"error": "No common frames"}

    # Collect metrics per frame and per attribute
    per_frame_metrics = {}
    attribute_aggregator = {}  # attr -> list of (gt_val, cv_val, match/error)

    for frame_id in sorted(common_frames):
        gt_attrs = gt_frames[frame_id]
        cv_attrs = cv_frames[frame_id]

        # Find common attributes for this frame
        common_attrs = set(gt_attrs.keys()).intersection(set(cv_attrs.keys()))

        frame_metrics = {
            "total_attributes": 0,
            "boolean_correct": 0,
            "categorical_correct": 0,
            "continuous_errors": [],
            "attributes": {}
        }

        for attr in common_attrs:
            gt_val = gt_attrs[attr]
            cv_val = cv_attrs[attr]

            # Skip if either is None
            if gt_val is None or cv_val is None:
                continue

            # Initialize attribute aggregator
            if attr not in attribute_aggregator:
                attribute_aggregator[attr] = {
                    "gt_values": [],
                    "cv_values": [],
                    "matches": [],
                    "errors": []
                }

            # Determine attribute type
            attr_type = determine_attribute_type(attr, gt_val, cv_val)

            attr_metrics = {"type": attr_type, "gt_value": gt_val, "cv_value": cv_val}

            if attr_type == "continuous":
                error = abs(gt_val - cv_val)
                relative_error = abs(error / gt_val) if gt_val != 0 else (0 if error == 0 else np.inf)

                attr_metrics["absolute_error"] = safe_round(error, 3)
                attr_metrics["relative_error"] = safe_round(relative_error, 3)
                match = error < abs(gt_val * 0.10)
                attr_metrics["match"] = match

                frame_metrics["continuous_errors"].append(error)
                attribute_aggregator[attr]["errors"].append(error)
                attribute_aggregator[attr]["gt_values"].append(gt_val)
                attribute_aggregator[attr]["cv_values"].append(cv_val)

            elif attr_type == "boolean":
                match = (gt_val == cv_val)
                attr_metrics["match"] = match
                if match:
                    frame_metrics["boolean_correct"] += 1

                attribute_aggregator[attr]["matches"].append(match)
                attribute_aggregator[attr]["gt_values"].append(gt_val)
                attribute_aggregator[attr]["cv_values"].append(cv_val)

            elif attr_type == "categorical":
                match = (gt_val == cv_val)
                attr_metrics["match"] = match
                if match:
                    frame_metrics["categorical_correct"] += 1

                attribute_aggregator[attr]["matches"].append(match)
                attribute_aggregator[attr]["gt_values"].append(gt_val)
                attribute_aggregator[attr]["cv_values"].append(cv_val)

            frame_metrics["total_attributes"] += 1
            frame_metrics["attributes"][attr] = attr_metrics

        per_frame_metrics[frame_id] = frame_metrics

    # Aggregate per-attribute metrics
    per_attribute_metrics = {}
    boolean_attrs = []
    categorical_attrs = []
    continuous_attrs = []

    for attr, data in attribute_aggregator.items():
        attr_summary = {}

        if data["errors"]:  # Continuous
            mean_abs_error = np.mean(data["errors"])
            gt_vals = np.array(data["gt_values"])
            cv_vals = np.array(data["cv_values"])
            rel_errors = np.abs((gt_vals - cv_vals) / np.where(gt_vals != 0, gt_vals, 1))
            mean_rel_error = np.mean(rel_errors[np.isfinite(rel_errors)])

            # R² score for continuous
            r2 = calculate_r2(data["gt_values"], data["cv_values"])

            attr_summary = {
                "type": "continuous",
                "count": len(data["errors"]),
                "mean_absolute_error": safe_round(mean_abs_error, 3),
                "mean_relative_error": safe_round(mean_rel_error, 3),
                "r2_score": safe_round(r2, 3)
            }
            continuous_attrs.append(attr)

        elif data["matches"]:  # Boolean or categorical
            accuracy = sum(data["matches"]) / len(data["matches"])

            # Determine if boolean or categorical
            unique_vals = set(data["gt_values"] + data["cv_values"])
            is_boolean = all(v in [True, False, "true", "false"] for v in unique_vals)

            attr_summary = {
                "type": "boolean" if is_boolean else "categorical",
                "count": len(data["matches"]),
                "accuracy": safe_round(accuracy, 3)
            }

            if is_boolean:
                boolean_attrs.append(attr)
            else:
                categorical_attrs.append(attr)

        per_attribute_metrics[attr] = attr_summary

    # Compute overall summary
    total_boolean = sum(per_attribute_metrics[a]["count"] for a in boolean_attrs)
    total_categorical = sum(per_attribute_metrics[a]["count"] for a in categorical_attrs)
    total_continuous = sum(per_attribute_metrics[a]["count"] for a in continuous_attrs)

    summary = {
        "total_frames": len(common_frames),
        "total_attributes": len(per_attribute_metrics),
        "boolean_attributes": len(boolean_attrs),
        "categorical_attributes": len(categorical_attrs),
        "continuous_attributes": len(continuous_attrs),
        "total_boolean_predictions": total_boolean,
        "total_categorical_predictions": total_categorical,
        "total_continuous_predictions": total_continuous
    }

    # Average boolean accuracy
    if boolean_attrs:
        bool_acc = np.mean([per_attribute_metrics[a]["accuracy"] for a in boolean_attrs])
        summary["boolean_accuracy"] = safe_round(bool_acc, 3)

    # Average categorical accuracy
    if categorical_attrs:
        cat_acc = np.mean([per_attribute_metrics[a]["accuracy"] for a in categorical_attrs])
        summary["categorical_accuracy"] = safe_round(cat_acc, 3)

    # Average continuous metrics
    if continuous_attrs:
        cont_mae = np.mean([per_attribute_metrics[a]["mean_absolute_error"] for a in continuous_attrs])
        cont_mre = np.mean([per_attribute_metrics[a]["mean_relative_error"] for a in continuous_attrs if per_attribute_metrics[a]["mean_relative_error"] is not None])
        cont_r2 = np.mean([per_attribute_metrics[a]["r2_score"] for a in continuous_attrs if per_attribute_metrics[a]["r2_score"] is not None])
        summary["continuous_mean_absolute_error"] = safe_round(cont_mae, 3)
        summary["continuous_mean_relative_error"] = safe_round(cont_mre, 3)
        summary["continuous_r2"] = safe_round(cont_r2, 3)

    # Overall accuracy (bool + cat)
    if boolean_attrs or categorical_attrs:
        all_bool_cat_attrs = boolean_attrs + categorical_attrs
        overall_acc = np.mean([per_attribute_metrics[a]["accuracy"] for a in all_bool_cat_attrs])
        summary["overall_accuracy"] = safe_round(overall_acc, 3)

    print(f"\n  Summary:")
    print(f"    Total frames evaluated: {summary['total_frames']}")
    print(f"    Total attributes: {summary['total_attributes']}")
    if "boolean_accuracy" in summary:
        print(f"    Boolean accuracy: {summary['boolean_accuracy']} ({summary['boolean_attributes']} attrs)")
    if "categorical_accuracy" in summary:
        print(f"    Categorical accuracy: {summary['categorical_accuracy']} ({summary['categorical_attributes']} attrs)")
    if "continuous_mean_absolute_error" in summary:
        print(f"    Continuous MAE: {summary['continuous_mean_absolute_error']} ({summary['continuous_attributes']} attrs)")
        print(f"    Continuous MRE: {summary['continuous_mean_relative_error']}")
        print(f"    Continuous R²: {summary['continuous_r2']}")
    if "overall_accuracy" in summary:
        print(f"    Overall accuracy (bool+cat): {summary['overall_accuracy']}")

    # Save detailed results if output_dir provided
    if output_dir:
        output_path = Path(output_dir) / "fai_frame_evaluation_metrics.json"
        with open(output_path, 'w') as f:
            json.dump({
                "summary": summary,
                "per_attribute_metrics": per_attribute_metrics,
                "per_frame_metrics": per_frame_metrics
            }, f, indent=2)
        print(f"\n  ✓ Detailed frame metrics saved to: {output_path}")

    print("=" * 70)

    return {
        "summary": summary,
        "per_attribute_metrics": per_attribute_metrics,
        "per_frame_metrics": per_frame_metrics
    }


def evaluate_player_frame_predictions(
    gt_response: Dict[str, Any],
    cv_response: Dict[str, Any],
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Evaluate player_frame predictions by comparing GT vs CV responses with applicability filtering.

    Filters predictions based on player role from player_play level, following
    the applicability rules defined in frame_targets_reference.yaml.

    Args:
        gt_response: Ground truth FAI API response
        cv_response: Computer vision FAI API response
        output_dir: Optional directory to save detailed results

    Returns:
        Dictionary containing:
            - per_player_frame_metrics: Dict of (track_id, frame_id) -> metrics
            - per_attribute_metrics: Dict of attribute -> aggregated metrics
            - summary: Overall summary statistics
            - filtering_stats: Statistics about applicability filtering
    """
    print("\n" + "="*20 + " PLAYER_FRAME PREDICTION EVALUATION " + "="*20)

    # Parse player_play predictions to extract roles
    gt_player_play, _ = parse_player_play_predictions(gt_response)
    cv_player_play, _ = parse_player_play_predictions(cv_response)

    # Extract player roles (use GT roles as source of truth)
    player_roles = extract_player_roles_from_player_play(gt_player_play)

    print(f"  Extracted roles for {len(player_roles)} players")
    print(f"  Using metadata-based applicability filtering from fai_target_metadata")

    # Parse player_frame predictions (with probabilities)
    gt_pf, gt_pf_probs = parse_player_frame_predictions(gt_response)
    cv_pf, cv_pf_probs = parse_player_frame_predictions(cv_response)

    if not gt_pf or not cv_pf:
        print("  ✗ No player_frame predictions found in responses")
        return {"error": "No player_frame predictions found"}

    print(f"  GT player-frames: {len(gt_pf)}")
    print(f"  CV player-frames: {len(cv_pf)}")

    # Find common (track_id, frame_id) keys
    common_keys = set(gt_pf.keys()).intersection(set(cv_pf.keys()))
    print(f"  Common player-frames: {len(common_keys)}")

    if not common_keys:
        print("  ✗ No common player-frames found")
        return {"error": "No common player-frames"}

    # Collect metrics per player-frame and per attribute (with filtering)
    per_player_frame_metrics = {}
    attribute_aggregator = {}  # attr -> {gt_values, cv_values, matches, errors, gt_probs, cv_probs}

    # Filtering statistics
    total_predictions = 0
    filtered_predictions = 0
    missing_role_count = 0

    for track_id, frame_id in sorted(common_keys):
        gt_attrs = gt_pf[(track_id, frame_id)]
        cv_attrs = cv_pf[(track_id, frame_id)]
        gt_probs = gt_pf_probs.get((track_id, frame_id), {})
        cv_probs = cv_pf_probs.get((track_id, frame_id), {})

        # Get player's role from player_play
        role = player_roles.get(track_id)
        player_play_attrs = gt_player_play.get(track_id, {})

        if role is None:
            missing_role_count += 1
            # Skip if we don't have role information
            continue

        # Find common attributes for this player-frame
        common_attrs = set(gt_attrs.keys()).intersection(set(cv_attrs.keys()))

        # Get applicable attributes for this player based on role and context
        applicable_attrs = get_applicable_player_frame_attributes(role, player_play_attrs, catalog="sbx")

        pf_metrics = {
            "track_id": track_id,
            "frame_id": frame_id,
            "role": role,
            "total_attributes": 0,
            "filtered_attributes": 0,
            "evaluated_attributes": 0,
            "attributes": {}
        }

        for attr in common_attrs:
            total_predictions += 1

            # Check applicability using metadata-based filtering
            if applicable_attrs and attr not in applicable_attrs:
                filtered_predictions += 1
                pf_metrics["filtered_attributes"] += 1
                continue

            pf_metrics["evaluated_attributes"] += 1

            gt_val = gt_attrs[attr]
            cv_val = cv_attrs[attr]

            # Skip if either is None
            if gt_val is None or cv_val is None:
                continue

            # Initialize attribute aggregator
            if attr not in attribute_aggregator:
                attribute_aggregator[attr] = {
                    "gt_values": [],
                    "cv_values": [],
                    "matches": [],
                    "errors": [],
                    "gt_probs": [],
                    "cv_probs": []
                }

            # Determine attribute type
            attr_type = determine_attribute_type(attr, gt_val, cv_val)

            attr_metrics = {"type": attr_type, "gt_value": gt_val, "cv_value": cv_val}

            if attr_type == "continuous":
                error = abs(gt_val - cv_val)
                relative_error = abs(error / gt_val) if gt_val != 0 else (0 if error == 0 else np.inf)

                attr_metrics["absolute_error"] = safe_round(error, 3)
                attr_metrics["relative_error"] = safe_round(relative_error, 3)
                match = error < abs(gt_val * 0.10)
                attr_metrics["match"] = match

                attribute_aggregator[attr]["errors"].append(error)
                attribute_aggregator[attr]["gt_values"].append(gt_val)
                attribute_aggregator[attr]["cv_values"].append(cv_val)

            elif attr_type == "boolean":
                match = (gt_val == cv_val)
                attr_metrics["match"] = match

                attribute_aggregator[attr]["matches"].append(match)
                attribute_aggregator[attr]["gt_values"].append(gt_val)
                attribute_aggregator[attr]["cv_values"].append(cv_val)

                # Collect probabilities for KL divergence
                if attr in gt_probs:
                    attribute_aggregator[attr]["gt_probs"].append(gt_probs[attr])
                if attr in cv_probs:
                    attribute_aggregator[attr]["cv_probs"].append(cv_probs[attr])

            elif attr_type == "categorical":
                match = (gt_val == cv_val)
                attr_metrics["match"] = match

                attribute_aggregator[attr]["matches"].append(match)
                attribute_aggregator[attr]["gt_values"].append(gt_val)
                attribute_aggregator[attr]["cv_values"].append(cv_val)

                # Collect probabilities for KL divergence
                if attr in gt_probs:
                    attribute_aggregator[attr]["gt_probs"].append(gt_probs[attr])
                if attr in cv_probs:
                    attribute_aggregator[attr]["cv_probs"].append(cv_probs[attr])

            pf_metrics["total_attributes"] += 1
            pf_metrics["attributes"][attr] = attr_metrics

        per_player_frame_metrics[(track_id, frame_id)] = pf_metrics

    # Aggregate per-attribute metrics
    per_attribute_metrics = {}
    boolean_attrs = []
    categorical_attrs = []
    continuous_attrs = []

    for attr, data in attribute_aggregator.items():
        attr_summary = {}

        if data["errors"]:  # Continuous
            mean_abs_error = np.mean(data["errors"])
            gt_vals = np.array(data["gt_values"])
            cv_vals = np.array(data["cv_values"])
            rel_errors = np.abs((gt_vals - cv_vals) / np.where(gt_vals != 0, gt_vals, 1))
            mean_rel_error = np.mean(rel_errors[np.isfinite(rel_errors)])

            # Calculate R² for continuous
            r2 = calculate_r2(data["gt_values"], data["cv_values"])

            attr_summary = {
                "type": "continuous",
                "count": len(data["errors"]),
                "mean_absolute_error": safe_round(mean_abs_error, 3),
                "mean_relative_error": safe_round(mean_rel_error, 3),
                "r2_score": safe_round(r2, 3)
            }
            continuous_attrs.append(attr)

        elif data["matches"]:  # Boolean or categorical
            accuracy = sum(data["matches"]) / len(data["matches"])

            # Determine if boolean or categorical
            unique_vals = set(data["gt_values"] + data["cv_values"])
            is_boolean = all(v in [True, False, "true", "false"] for v in unique_vals)

            attr_summary = {
                "type": "boolean" if is_boolean else "categorical",
                "count": len(data["matches"]),
                "accuracy": safe_round(accuracy, 3)
            }

            # Calculate KL divergence if probabilities available
            if data["gt_probs"] and data["cv_probs"]:
                kl_div = calculate_kl_divergence(data["gt_probs"], data["cv_probs"])
                attr_summary["kl_divergence"] = safe_round(kl_div, 4)

            # Calculate BSS if we have values
            if data["gt_values"] and data["cv_values"]:
                attr_type = "boolean" if is_boolean else "categorical"
                baseline_freq = compute_baseline_frequencies(data["gt_values"], attr_type)
                bss = calculate_brier_skill_score(
                    data["gt_values"],
                    data["cv_values"],
                    is_binary=is_boolean,
                    baseline_freq=baseline_freq
                )
                attr_summary["brier_skill_score"] = safe_round(bss, 3)

            if is_boolean:
                boolean_attrs.append(attr)
            else:
                categorical_attrs.append(attr)

        per_attribute_metrics[attr] = attr_summary

    # Compute overall summary
    total_boolean = sum(per_attribute_metrics[a]["count"] for a in boolean_attrs)
    total_categorical = sum(per_attribute_metrics[a]["count"] for a in categorical_attrs)
    total_continuous = sum(per_attribute_metrics[a]["count"] for a in continuous_attrs)

    summary = {
        "total_player_frames": len(common_keys),
        "total_attributes": len(per_attribute_metrics),
        "boolean_attributes": len(boolean_attrs),
        "categorical_attributes": len(categorical_attrs),
        "continuous_attributes": len(continuous_attrs),
        "total_boolean_predictions": total_boolean,
        "total_categorical_predictions": total_categorical,
        "total_continuous_predictions": total_continuous
    }

    # Average boolean metrics: accuracy, KL divergence, BSS
    if boolean_attrs:
        bool_acc = np.mean([per_attribute_metrics[a]["accuracy"] for a in boolean_attrs])
        summary["boolean_accuracy"] = safe_round(bool_acc, 3)

        # Average KL divergence for boolean attributes
        bool_kl_values = [per_attribute_metrics[a]["kl_divergence"] for a in boolean_attrs if "kl_divergence" in per_attribute_metrics[a] and per_attribute_metrics[a]["kl_divergence"] is not None]
        if bool_kl_values:
            summary["boolean_kl_divergence"] = safe_round(np.mean(bool_kl_values), 4)

        # Average BSS for boolean attributes
        bool_bss_values = [per_attribute_metrics[a]["brier_skill_score"] for a in boolean_attrs if "brier_skill_score" in per_attribute_metrics[a] and per_attribute_metrics[a]["brier_skill_score"] is not None]
        if bool_bss_values:
            summary["boolean_bss"] = safe_round(np.mean(bool_bss_values), 3)

    # Average categorical metrics: accuracy, KL divergence, BSS
    if categorical_attrs:
        cat_acc = np.mean([per_attribute_metrics[a]["accuracy"] for a in categorical_attrs])
        summary["categorical_accuracy"] = safe_round(cat_acc, 3)

        # Average KL divergence for categorical attributes
        cat_kl_values = [per_attribute_metrics[a]["kl_divergence"] for a in categorical_attrs if "kl_divergence" in per_attribute_metrics[a] and per_attribute_metrics[a]["kl_divergence"] is not None]
        if cat_kl_values:
            summary["categorical_kl_divergence"] = safe_round(np.mean(cat_kl_values), 4)

        # Average BSS for categorical attributes
        cat_bss_values = [per_attribute_metrics[a]["brier_skill_score"] for a in categorical_attrs if "brier_skill_score" in per_attribute_metrics[a] and per_attribute_metrics[a]["brier_skill_score"] is not None]
        if cat_bss_values:
            summary["categorical_bss"] = safe_round(np.mean(cat_bss_values), 3)

    # Average continuous metrics: MAE, MRE, R²
    if continuous_attrs:
        cont_mae = np.mean([per_attribute_metrics[a]["mean_absolute_error"] for a in continuous_attrs])
        cont_mre = np.mean([per_attribute_metrics[a]["mean_relative_error"] for a in continuous_attrs if per_attribute_metrics[a]["mean_relative_error"] is not None])
        summary["continuous_mean_absolute_error"] = safe_round(cont_mae, 3)
        summary["continuous_mean_relative_error"] = safe_round(cont_mre, 3)

        # Average R² for continuous attributes
        cont_r2_values = [per_attribute_metrics[a]["r2_score"] for a in continuous_attrs if "r2_score" in per_attribute_metrics[a] and per_attribute_metrics[a]["r2_score"] is not None]
        if cont_r2_values:
            summary["continuous_r2"] = safe_round(np.mean(cont_r2_values), 3)

    # Overall accuracy (bool + cat)
    if boolean_attrs or categorical_attrs:
        all_bool_cat_attrs = boolean_attrs + categorical_attrs
        overall_acc = np.mean([per_attribute_metrics[a]["accuracy"] for a in all_bool_cat_attrs])
        summary["overall_accuracy"] = safe_round(overall_acc, 3)

    # Filtering statistics
    filtering_stats = {
        "total_predictions_before_filtering": total_predictions,
        "filtered_predictions": filtered_predictions,
        "evaluated_predictions": total_predictions - filtered_predictions,
        "filter_rate": safe_round(filtered_predictions / total_predictions if total_predictions > 0 else 0, 3),
        "missing_role_count": missing_role_count
    }

    print(f"\n  Filtering Statistics:")
    print(f"    Total predictions before filtering: {filtering_stats['total_predictions_before_filtering']}")
    print(f"    Filtered predictions (not applicable): {filtering_stats['filtered_predictions']}")
    print(f"    Evaluated predictions: {filtering_stats['evaluated_predictions']}")
    print(f"    Filter rate: {filtering_stats['filter_rate']}")
    if missing_role_count > 0:
        print(f"    ⚠ Missing role for {missing_role_count} players")

    print(f"\n  Summary:")
    print(f"    Total player-frames evaluated: {summary['total_player_frames']}")
    print(f"    Total attributes: {summary['total_attributes']}")
    if "boolean_accuracy" in summary:
        print(f"    Boolean accuracy: {summary['boolean_accuracy']} ({summary['boolean_attributes']} attrs)")
        if "boolean_kl_divergence" in summary:
            print(f"    Boolean KL divergence: {summary['boolean_kl_divergence']}")
        if "boolean_bss" in summary:
            print(f"    Boolean BSS: {summary['boolean_bss']}")
    if "categorical_accuracy" in summary:
        print(f"    Categorical accuracy: {summary['categorical_accuracy']} ({summary['categorical_attributes']} attrs)")
        if "categorical_kl_divergence" in summary:
            print(f"    Categorical KL divergence: {summary['categorical_kl_divergence']}")
        if "categorical_bss" in summary:
            print(f"    Categorical BSS: {summary['categorical_bss']}")
    if "continuous_mean_absolute_error" in summary:
        print(f"    Continuous MAE: {summary['continuous_mean_absolute_error']} ({summary['continuous_attributes']} attrs)")
        print(f"    Continuous MRE: {summary['continuous_mean_relative_error']}")
        if "continuous_r2" in summary:
            print(f"    Continuous R²: {summary['continuous_r2']}")
    if "overall_accuracy" in summary:
        print(f"    Overall accuracy (bool+cat): {summary['overall_accuracy']}")

    # Save detailed results if output_dir provided
    if output_dir:
        output_path = Path(output_dir) / "fai_player_frame_evaluation_metrics.json"
        with open(output_path, 'w') as f:
            # Convert tuple keys to strings for JSON serialization
            serializable_metrics = {
                f"{track_id}_{frame_id}": metrics
                for (track_id, frame_id), metrics in per_player_frame_metrics.items()
            }
            json.dump({
                "summary": summary,
                "filtering_stats": filtering_stats,
                "per_attribute_metrics": per_attribute_metrics,
                "per_player_frame_metrics": serializable_metrics
            }, f, indent=2)
        print(f"\n  ✓ Detailed player_frame metrics saved to: {output_path}")

    print("=" * 70)

    return {
        "summary": summary,
        "filtering_stats": filtering_stats,
        "per_attribute_metrics": per_attribute_metrics,
        "per_player_frame_metrics": per_player_frame_metrics
    }


# ============================================================================
# BATCH PREDICTIONS LOADING (imported from batch_loader.py)
# ============================================================================
# Note: These functions require Doppler prd config to access Databricks.
# Use the _subprocess versions for production to ensure correct Doppler config.

try:
    from .batch_loader import (
        load_batch_predictions,
        load_batch_player_play_predictions,
        load_batch_predictions_subprocess,
        load_batch_player_play_predictions_subprocess
    )
except ImportError:
    # Fallback if batch_loader is not available
    def load_batch_predictions(sumer_game_id: str, sumer_play_id: str, catalog: str = "prd") -> Optional[Dict[str, Any]]:
        print("  ⚠ batch_loader module not available")
        return None

    def load_batch_player_play_predictions(sumer_game_id: str, sumer_play_id: str, catalog: str = "prd") -> Optional[Dict[str, Any]]:
        print("  ⚠ batch_loader module not available")
        return None

    def load_batch_predictions_subprocess(sumer_game_id: str, sumer_play_id: str, catalog: str = "prd", doppler_config: str = "prd") -> Optional[Dict[str, Any]]:
        print("  ⚠ batch_loader module not available")
        return None

    def load_batch_player_play_predictions_subprocess(sumer_game_id: str, sumer_play_id: str, catalog: str = "prd", doppler_config: str = "prd") -> Optional[Dict[str, Any]]:
        print("  ⚠ batch_loader module not available")
        return None


def evaluate_three_way_comparison(
    gt_response: Dict[str, Any],
    cv_response: Dict[str, Any],
    batch_response: Dict[str, Any],
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Perform 3-way comparison: GT vs CV vs Batch predictions.

    This helps separate:
    - Tracking quality (CV vs GT with same model)
    - Model quality (Batch vs GT with better model)

    Args:
        gt_response: Ground truth FAI API response (bad model)
        cv_response: Computer vision FAI API response (bad model)
        batch_response: Batch predictions (good model)
        output_dir: Optional directory to save detailed results

    Returns:
        Dictionary containing:
            - cv_vs_gt: CV vs GT comparison (tracking quality)
            - batch_vs_gt: Batch vs GT comparison (model quality)
            - summary: Combined summary statistics
    """
    print("\n" + "="*20 + " 3-WAY FAI COMPARISON " + "="*20)

    # Evaluate CV vs GT (same bad model, different tracking)
    print("\n  [1/2] Evaluating CV vs GT (tracking quality)...")
    cv_vs_gt = evaluate_play_predictions(gt_response, cv_response, output_dir=None)

    # Evaluate Batch vs GT (good model vs bad model)
    print("\n  [2/2] Evaluating Batch vs GT (model quality)...")
    batch_vs_gt = evaluate_play_predictions(gt_response, batch_response, output_dir=None)

    # Build combined summary
    results = {
        "cv_vs_gt": cv_vs_gt,
        "batch_vs_gt": batch_vs_gt,
        "summary": {
            "cv_vs_gt_accuracy": cv_vs_gt.get("summary", {}).get("overall_accuracy"),
            "batch_vs_gt_accuracy": batch_vs_gt.get("summary", {}).get("overall_accuracy"),
            "tracking_impact": None,  # Will calculate below
            "model_impact": None,  # Will calculate below
        }
    }

    # Calculate impact metrics
    cv_acc = cv_vs_gt.get("summary", {}).get("overall_accuracy")
    batch_acc = batch_vs_gt.get("summary", {}).get("overall_accuracy")

    if cv_acc is not None and batch_acc is not None:
        # Tracking impact: How much does CV tracking hurt accuracy vs GT tracking?
        results["summary"]["tracking_impact"] = safe_round(1.0 - cv_acc, 3)

        # Model impact: How much does the bad model hurt accuracy vs good model?
        results["summary"]["model_impact"] = safe_round(1.0 - batch_acc, 3)

    # Print summary
    print("\n  " + "="*60)
    print("  3-WAY COMPARISON SUMMARY")
    print("  " + "="*60)
    print(f"  CV vs GT (tracking quality):    {cv_acc}")
    print(f"  Batch vs GT (model quality):    {batch_acc}")
    if results["summary"]["tracking_impact"] is not None:
        print(f"  Tracking error impact:          {results['summary']['tracking_impact']}")
    if results["summary"]["model_impact"] is not None:
        print(f"  Model error impact:             {results['summary']['model_impact']}")
    print("  " + "="*60)

    # Save detailed results if output_dir provided
    if output_dir:
        output_path = Path(output_dir) / "fai_three_way_comparison.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  ✓ 3-way comparison saved to: {output_path}")

    print("=" * 70)

    return results


def evaluate_play_predictions_from_files(output_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load FAI responses from files and evaluate play-level predictions.

    Args:
        output_dir: Directory containing fai_response_gt.json and fai_response_cv.json

    Returns:
        Evaluation results dictionary, or None if files not found
    """
    gt_response, cv_response = load_fai_responses(output_dir)

    if gt_response is None or cv_response is None:
        return None

    return evaluate_play_predictions(gt_response, cv_response, output_dir)
