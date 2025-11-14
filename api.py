"""
FastAPI server for computing evaluation metrics on play/player records.

Clean architecture with pluggable transport heads:
- HTTP transport (REST API)
- WebSocket transport (real-time streaming)

Core computation logic is transport-agnostic.
"""

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any

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


def compute_metrics(records: List[PlayRecord]) -> List[Dict[str, Any]]:
    """
    Pure function to compute metrics from records.

    This function is transport-agnostic and can be called from HTTP, WebSocket, etc.
    Uses batch predictions to determine applicability for all records.

    Args:
        records: List of play/player_play records

    Returns:
        List of attribute metrics with cv_xy_score and batch_xy_score
    """
    if not records:
        return []

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
                        "batch": []
                    }

                # Collect gt, cv, and batch values
                gt_val = value
                cv_val = record.data.get(f"cv_{attr}")
                batch_val = record.data.get(f"batch_{attr}")

                # Only include if all three are non-null
                if gt_val is not None and cv_val is not None and batch_val is not None:
                    attributes_data[attr]["gt"].append(gt_val)
                    attributes_data[attr]["cv"].append(cv_val)
                    attributes_data[attr]["batch"].append(batch_val)

    # Compute metrics for each attribute
    results = []

    for attr, data in attributes_data.items():
        if len(data["gt"]) == 0:
            continue

        gt_vals = data["gt"]
        cv_vals = data["cv"]
        batch_vals = data["batch"]

        # Determine attribute type
        attr_type = determine_attribute_type(attr, gt_vals[0], cv_vals[0])

        # Compute metrics based on type
        if attr_type == "continuous":
            cv_score = calculate_r2(gt_vals, cv_vals)
            batch_score = calculate_r2(gt_vals, batch_vals)
            metric_name = "r2"
        else:
            # Boolean or categorical - use BSS
            is_binary = attr_type == "boolean"
            baseline_freq = compute_baseline_frequencies(gt_vals, attr_type)
            cv_score = calculate_brier_skill_score(gt_vals, cv_vals, is_binary, baseline_freq)
            batch_score = calculate_brier_skill_score(gt_vals, batch_vals, is_binary, baseline_freq)
            metric_name = "bss"

        results.append({
            "attribute": attr,
            "type": attr_type,
            "metric_name": metric_name,
            "cv_xy_score": round(cv_score, 4) if cv_score is not None and not np.isnan(cv_score) else None,
            "batch_xy_score": round(batch_score, 4) if batch_score is not None and not np.isnan(batch_score) else None,
            "n_instances": len(gt_vals)
        })

    return results


# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
