"""
FastAPI server for computing evaluation metrics on play/player records.

Clean architecture with pluggable transport heads:
- HTTP transport (REST API)
- WebSocket transport (real-time streaming)

Core computation logic is transport-agnostic.
"""

import numpy as np
import csv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional

from eval_metrics import (
    calculate_r2,
    calculate_brier_skill_score,
    compute_baseline_frequencies,
    determine_attribute_type
)
from applicability import load_applicability_rules, is_applicable

# Import transport heads
from transport_heads.http import router as http_router, PlayRecord
from transport_heads.websocket import router as ws_router

# Load metadata once at startup
_metadata_cache: Optional[Dict[str, Dict[str, str]]] = None

def load_metadata() -> Dict[str, Dict[str, str]]:
    """Load metadata.csv and create lookup for attribute status and type"""
    global _metadata_cache
    if _metadata_cache is not None:
        return _metadata_cache

    metadata = {}
    try:
        with open('metadata.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                target_id = row.get('target_id')
                attribute = row.get('attribute')

                if target_id:
                    meta_entry = {
                        'status': row.get('status', ''),
                        'entity': row.get('entity', ''),
                        'target_type': row.get('target_type', '')
                    }

                    # Add entry for target_id (primary key)
                    metadata[target_id] = meta_entry

                    # Also add entry for attribute if different (some CSVs use attribute name)
                    if attribute and attribute != target_id:
                        metadata[attribute] = meta_entry
        _metadata_cache = metadata
        print(f"Loaded metadata for {len(metadata)} attributes")
    except Exception as e:
        print(f"Warning: Could not load metadata.csv: {e}")
        _metadata_cache = {}

    return _metadata_cache


# ============================================================================
# APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Eval Metrics API",
    version="1.0.0",
    description="Multi-transport API for computing evaluation metrics with HTTP and WebSocket support"
)

# Enable CORS for client-side requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include transport heads
app.include_router(http_router, tags=["HTTP"])
app.include_router(ws_router, tags=["WebSocket"])


# ============================================================================
# CORE COMPUTATION LOGIC (Transport-agnostic)
# ============================================================================


def compute_accuracy(records: List[PlayRecord], allowed_statuses: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Compute accuracy metrics from records by comparing predictions to labels.

    Compares three models (gt, cv, batch) against PFF labels.

    Args:
        records: List of play/player_play records
        allowed_statuses: List of allowed status values (e.g., ['GA', 'PREVIEW', 'BETA'])
                         If None, all statuses are allowed

    Returns:
        List of attribute metrics with accuracy scores for each model vs labels
    """
    if not records:
        return []

    # Load metadata for status lookup
    metadata = load_metadata()

    # Reorganize data by attribute
    attributes_data = {}

    for record in records:
        # Iterate through all data fields to find label fields
        for key, value in record.data.items():
            if key.startswith("label_") and not key.endswith("_probs"):
                attr = key[6:]  # Remove "label_" prefix

                # Skip metadata columns
                if attr in ['season', 'league', 'game_date_week']:
                    continue

                # Initialize attribute if not seen before
                if attr not in attributes_data:
                    attributes_data[attr] = {
                        "label": [],
                        "gt": [],
                        "cv": [],
                        "batch": []
                    }

                # Collect label, gt, cv, and batch values
                label_val = value
                gt_val = record.data.get(f"gt_{attr}")
                cv_val = record.data.get(f"cv_{attr}")
                batch_val = record.data.get(f"batch_{attr}")

                # Only include if all four are non-null
                if (label_val is not None and
                    gt_val is not None and
                    cv_val is not None and
                    batch_val is not None):
                    attributes_data[attr]["label"].append(label_val)
                    attributes_data[attr]["gt"].append(gt_val)
                    attributes_data[attr]["cv"].append(cv_val)
                    attributes_data[attr]["batch"].append(batch_val)

    # Compute accuracy metrics for each attribute
    results = []

    for attr, data in attributes_data.items():
        if len(data["label"]) == 0:
            continue

        # Check if attribute status is allowed
        attr_metadata = metadata.get(attr, {})
        status = attr_metadata.get('status', 'UNKNOWN')

        if allowed_statuses is not None and status not in allowed_statuses:
            continue

        label_vals = data["label"]
        gt_vals = data["gt"]
        cv_vals = data["cv"]
        batch_vals = data["batch"]

        # Get attribute type from metadata
        attr_type = attr_metadata.get('target_type', 'UNKNOWN')

        # Compute accuracy (agreement rate) for each model vs labels
        # For continuous variables, use 10% tolerance; for categorical/boolean use exact match
        # For yard-based metrics, use 1 yard absolute tolerance
        is_continuous = (attr_type.lower() == 'continuous')

        gt_accuracy = _calculate_agreement(label_vals, gt_vals, is_continuous=is_continuous, attribute_name=attr)
        cv_accuracy = _calculate_agreement(label_vals, cv_vals, is_continuous=is_continuous, attribute_name=attr)
        batch_accuracy = _calculate_agreement(label_vals, batch_vals, is_continuous=is_continuous, attribute_name=attr)

        results.append({
            "attribute": attr,
            "type": attr_type,
            "status": status,
            "gt_accuracy": round(gt_accuracy, 4) if gt_accuracy is not None and not np.isnan(gt_accuracy) else None,
            "cv_accuracy": round(cv_accuracy, 4) if cv_accuracy is not None and not np.isnan(cv_accuracy) else None,
            "batch_accuracy": round(batch_accuracy, 4) if batch_accuracy is not None and not np.isnan(batch_accuracy) else None,
            "n_instances": len(label_vals)
        })

    return results


def _calculate_agreement(labels: List[Any], predictions: List[Any], is_continuous: bool = False, attribute_name: str = "") -> float:
    """
    Calculate agreement rate between labels and predictions.

    For yard-based metrics: checks if prediction is within 1 yard absolute tolerance
    For other continuous variables: checks if prediction is within 10% of label
    For categorical/boolean: checks for exact match
    """
    if len(labels) != len(predictions) or len(labels) == 0:
        return None

    if is_continuous:
        # Check if this is a yard-based metric
        is_yard_metric = 'yard' in attribute_name.lower()

        matches = 0
        for label, pred in zip(labels, predictions):
            try:
                label_val = float(label)
                pred_val = float(pred)

                if is_yard_metric:
                    # For yard metrics, use absolute tolerance of 1 yard
                    if abs(pred_val - label_val) <= 1.0:
                        matches += 1
                elif abs(label_val) < 1e-6:
                    # If label is 0, require prediction to be very close (absolute tolerance)
                    if abs(pred_val) < 0.1:  # Within 0.1 of zero
                        matches += 1
                else:
                    # Check if within 10% relative tolerance
                    if abs(pred_val - label_val) / abs(label_val) <= 0.10:
                        matches += 1
            except (ValueError, TypeError):
                # Skip if can't convert to float
                continue

        return matches / len(labels)
    else:
        # Exact match for categorical/boolean
        matches = sum(1 for label, pred in zip(labels, predictions) if label == pred)
        return matches / len(labels)


def compute_metrics(records: List[PlayRecord], allowed_statuses: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Pure function to compute metrics from records.

    This function is transport-agnostic and can be called from HTTP, WebSocket, etc.
    Uses batch predictions to determine applicability for all records.

    For continuous attributes: Uses R² score
    For categorical/boolean: Uses TRUE Brier Skill Score with probability distributions

    Args:
        records: List of play/player_play records
        allowed_statuses: List of allowed status values (e.g., ['GA', 'PREVIEW', 'BETA'])
                         If None, all statuses are allowed

    Returns:
        List of attribute metrics with cv_xy_score, batch_xy_score, cv_batch_score, and status
        - cv_xy_score: CV Data vs NGS Data (both using XY API) - uses CV probabilities
        - batch_xy_score: NGS Data XY API vs Batch API - uses Batch probabilities
        - cv_batch_score: CV Data XY API vs NGS Data Batch API - uses CV probabilities
    """
    if not records:
        return []

    # Load metadata for status lookup
    metadata = load_metadata()

    # Load applicability rules from metadata
    try:
        applicability_rules = load_applicability_rules()
    except Exception as e:
        # If metadata loading fails, proceed without applicability filtering
        print(f"Warning: Could not load applicability rules: {e}")
        applicability_rules = {}

    # Reorganize data by attribute
    # Each attribute has gt_, cv_, and batch_ versions
    attributes_data = {}

    for record in records:
        # Extract all batch predictions for applicability checking
        batch_data = {}
        for key, value in record.data.items():
            if key.startswith("batch_") and not key.endswith("_probs"):
                batch_data[key[6:]] = value  # Remove "batch_" prefix

        # Iterate through all data fields
        for key, value in record.data.items():
            # Look for ground truth fields (gt_*)
            if key.startswith("gt_") and not key.endswith("_probs"):
                attr = key[3:]  # Remove "gt_" prefix

                # Check if applicable based on batch predictions
                if not is_applicable(attr, batch_data, applicability_rules):
                    continue

                # Initialize attribute if not seen before
                if attr not in attributes_data:
                    attributes_data[attr] = {
                        "gt": [],
                        "cv": [],
                        "batch": [],
                        "gt_probs": [],
                        "cv_probs": [],
                        "batch_probs": []
                    }

                # Collect gt, cv, and batch values
                gt_val = value
                cv_val = record.data.get(f"cv_{attr}")
                batch_val = record.data.get(f"batch_{attr}")

                # Collect probability distributions (if available)
                gt_probs = record.data.get(f"gt_{attr}_probs")
                cv_probs = record.data.get(f"cv_{attr}_probs")
                batch_probs = record.data.get(f"batch_{attr}_probs")

                # Only include if all three are non-null
                if gt_val is not None and cv_val is not None and batch_val is not None:
                    attributes_data[attr]["gt"].append(gt_val)
                    attributes_data[attr]["cv"].append(cv_val)
                    attributes_data[attr]["batch"].append(batch_val)
                    attributes_data[attr]["gt_probs"].append(gt_probs)
                    attributes_data[attr]["cv_probs"].append(cv_probs)
                    attributes_data[attr]["batch_probs"].append(batch_probs)

    # Compute metrics for each attribute
    results = []

    for attr, data in attributes_data.items():
        if len(data["gt"]) == 0:
            continue

        # Check if attribute status is allowed
        attr_metadata = metadata.get(attr, {})
        status = attr_metadata.get('status', 'UNKNOWN')

        if allowed_statuses is not None and status not in allowed_statuses:
            continue  # Skip this attribute if status not in allowed list

        gt_vals = data["gt"]
        cv_vals = data["cv"]
        batch_vals = data["batch"]
        gt_probs = data["gt_probs"]
        cv_probs = data["cv_probs"]
        batch_probs = data["batch_probs"]

        # Get attribute type from metadata, fallback to auto-detection
        attr_type = attr_metadata.get('target_type')
        if not attr_type:
            attr_type = determine_attribute_type(attr, gt_vals[0], cv_vals[0])

        # Compute metrics based on type
        if attr_type == "continuous":
            cv_score = calculate_r2(gt_vals, cv_vals)
            batch_score = calculate_r2(gt_vals, batch_vals)
            cv_batch_score = calculate_r2(batch_vals, cv_vals)  # CV XY vs NGS Batch
            metric_name = "r2"
        else:
            # Boolean or categorical - use BSS with probabilities
            is_binary = attr_type == "boolean"
            baseline_freq = compute_baseline_frequencies(gt_vals, attr_type)

            # CV vs GT: use CV probabilities
            cv_score = calculate_brier_skill_score(
                gt_vals, cv_vals, is_binary, baseline_freq,
                y_pred_probs=cv_probs if any(p is not None for p in cv_probs) else None
            )

            # Batch vs GT: use Batch probabilities (now available!)
            batch_score = calculate_brier_skill_score(
                gt_vals, batch_vals, is_binary, baseline_freq,
                y_pred_probs=batch_probs if any(p is not None for p in batch_probs) else None
            )

            # CV vs Batch: use CV probabilities (batch is "ground truth" in this comparison)
            baseline_freq_batch = compute_baseline_frequencies(batch_vals, attr_type)
            cv_batch_score = calculate_brier_skill_score(
                batch_vals, cv_vals, is_binary, baseline_freq_batch,
                y_pred_probs=cv_probs if any(p is not None for p in cv_probs) else None
            )
            metric_name = "bss"

        # Debug logging for missing metadata
        if status == 'UNKNOWN':
            print(f"Warning: No metadata found for attribute '{attr}'")

        results.append({
            "attribute": attr,
            "type": attr_type,
            "status": status,
            "metric_name": metric_name,
            "cv_xy_score": round(cv_score, 4) if cv_score is not None and not np.isnan(cv_score) else None,
            "batch_xy_score": round(batch_score, 4) if batch_score is not None and not np.isnan(batch_score) else None,
            "cv_batch_score": round(cv_batch_score, 4) if cv_batch_score is not None and not np.isnan(cv_batch_score) else None,
            "n_instances": len(gt_vals)
        })

    return results


# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
