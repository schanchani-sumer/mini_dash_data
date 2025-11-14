"""
HTTP transport head for Eval Metrics API.

Provides REST endpoints for computing metrics via HTTP POST requests.
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


class MetricsRequest(BaseModel):
    """Request payload for metrics computation"""
    records: List[PlayRecord]
    entity: str = "play"  # "play" or "player_play"


class AttributeMetric(BaseModel):
    """Metrics for a single attribute"""
    attribute: str
    type: str  # "continuous", "boolean", or "categorical"
    metric_name: str  # "r2" or "bss"
    cv_xy_score: Optional[float]
    batch_xy_score: Optional[float]
    n_instances: int


class MetricsResponse(BaseModel):
    """Response with computed metrics"""
    total_records: int
    metrics: List[AttributeMetric]


# Create router
router = APIRouter()


@router.post("/compute-metrics", response_model=MetricsResponse)
def compute_metrics_endpoint(request: MetricsRequest):
    """
    Receive filtered records from client, compute and return metrics.

    The client loads the CSV, filters it based on user criteria, and sends
    the filtered records here for metric computation.
    """
    # Import here to avoid circular dependency
    from api import compute_metrics

    metrics = compute_metrics(request.records)

    return MetricsResponse(
        total_records=len(request.records),
        metrics=metrics
    )


@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "transport": "http"}
