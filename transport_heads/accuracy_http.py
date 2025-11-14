"""
HTTP transport head for Accuracy Metrics API.

Provides REST endpoints for computing accuracy metrics via HTTP POST requests.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any, Optional


# Request/Response models
class PlayRecord(BaseModel):
    """Individual record from CSV (play or player_play level)"""
    game_id: str
    play_id: str
    track_id: Optional[str] = None  # Only for player_play records
    data: Dict[str, Any]  # All columns as dict


class AccuracyRequest(BaseModel):
    """Request payload for accuracy metrics computation"""
    records: List[PlayRecord]
    entity: str = "play"  # "play" or "player_play"
    allowed_statuses: Optional[List[str]] = None  # Optional status filter


class AttributeAccuracy(BaseModel):
    """Accuracy metrics for a single attribute"""
    attribute: str
    type: str  # "continuous", "boolean", or "categorical"
    status: str
    gt_accuracy: Optional[float]
    cv_accuracy: Optional[float]
    batch_accuracy: Optional[float]
    n_instances: int


class AccuracyResponse(BaseModel):
    """Response with computed accuracy metrics"""
    total_records: int
    metrics: List[AttributeAccuracy]


# Create router
router = APIRouter()


@router.post("/compute-accuracy", response_model=AccuracyResponse)
def compute_accuracy_endpoint(request: AccuracyRequest):
    """
    Receive filtered records from client, compute and return accuracy metrics.

    The client loads the CSV, filters it based on user criteria, and sends
    the filtered records here for metric computation.

    Compares three models (gt, cv, batch) against PFF labels.
    """
    # Import here to avoid circular dependency
    from accuracy_api import compute_accuracy

    metrics = compute_accuracy(request.records, allowed_statuses=request.allowed_statuses)

    return AccuracyResponse(
        total_records=len(request.records),
        metrics=metrics
    )


@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "transport": "http", "api": "accuracy"}
