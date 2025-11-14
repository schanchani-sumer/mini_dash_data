"""
WebSocket transport head for Eval Metrics API.

Provides real-time streaming computation via WebSocket connections.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any, Optional
import json
import csv
from threading import Lock


# Create router
router = APIRouter()

# Shared data store (loaded once, used by all connections)
_data_cache = {
    "play_records": None,
    "player_play_records": None,
    "loaded": False
}
_cache_lock = Lock()


def load_csv_from_disk(csv_path: str) -> List[Dict[str, Any]]:
    """Load CSV from disk and convert to records"""
    records = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert string values to appropriate types
            data = {}
            for key, value in row.items():
                if value == '':
                    data[key] = None
                elif value.lower() == 'true':
                    data[key] = True
                elif value.lower() == 'false':
                    data[key] = False
                else:
                    # Try to convert to float if possible
                    try:
                        data[key] = float(value)
                    except (ValueError, TypeError):
                        data[key] = value

            record = {
                "game_id": row.get("game_id"),
                "play_id": row.get("play_id"),
                "data": data
            }

            # Add track_id if present (for player_play records)
            if "track_id" in row:
                record["track_id"] = row.get("track_id")

            records.append(record)

    return records


def apply_filters(records: List[Dict], filters: Dict[str, Any]) -> List[Dict]:
    """Apply client-side filters to records"""
    filtered = records

    # Filter by completeness
    if filters.get('minCompleteness', 0) > 0:
        min_comp = filters['minCompleteness']
        filtered = [r for r in filtered
                   if (r['data'].get('completeness_0_to_end') or 0) >= min_comp]

    # Filter by tracked players (if present)
    if filters.get('minTrackedPlayers'):
        min_tracked = filters['minTrackedPlayers']
        filtered = [r for r in filtered
                   if (r['data'].get('tracked_players') or 0) >= min_tracked]

    # Filter by worst case yard error (max threshold)
    if filters.get('maxWorstCaseYardError'):
        max_error = filters['maxWorstCaseYardError']
        filtered = [r for r in filtered
                   if (r['data'].get('worst_case_avg_pos_error_yards') or float('inf')) <= max_error]

    # Filter by avg yard error (max threshold)
    if filters.get('maxAvgYardError'):
        max_error = filters['maxAvgYardError']
        filtered = [r for r in filtered
                   if (r['data'].get('overall_median_pos_error_yards') or float('inf')) <= max_error]

    return filtered


@router.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """
    WebSocket endpoint for real-time metrics computation.

    Client sends:
    {
        "records": [...],  # OR
        "entity": "play" or "player_play"
    }

    Server responds with:
    {
        "total_records": int,
        "metrics": [...]
    }

    Supports streaming multiple requests over the same connection.
    """
    await websocket.accept()

    try:
        while True:
            # Receive data from client
            data = await websocket.receive_json()

            # Import here to avoid circular dependency
            from api import compute_metrics
            from transport_heads.http import PlayRecord

            # Parse records
            records = [PlayRecord(**record) for record in data.get('records', [])]

            # Compute metrics (same function as HTTP!)
            metrics = compute_metrics(records)

            # Send response
            response = {
                "total_records": len(records),
                "metrics": metrics
            }

            await websocket.send_json(response)

    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "error": str(e),
                "total_records": 0,
                "metrics": []
            })
        except:
            pass


def get_cached_data():
    """Get cached CSV data, loading if necessary (thread-safe)"""
    with _cache_lock:
        if not _data_cache["loaded"]:
            print("Loading CSV data from disk (first connection)...")
            _data_cache["play_records"] = load_csv_from_disk("play_records.csv")
            _data_cache["player_play_records"] = load_csv_from_disk("player_play_records.csv")
            _data_cache["loaded"] = True
            print(f"Loaded {len(_data_cache['play_records'])} play records, "
                  f"{len(_data_cache['player_play_records'])} player_play records")
        else:
            print("Using cached CSV data (subsequent connection)")

    return _data_cache["play_records"], _data_cache["player_play_records"]


@router.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """
    Dashboard WebSocket endpoint - accepts filter criteria, returns metrics.

    Server loads CSV from disk (once, cached), applies filters, and computes metrics.
    Supports multiple concurrent connections efficiently.

    Client sends:
    {
        "action": "compute",
        "entity": "play" or "player_play",
        "filters": {
            "minCompleteness": 0.0-1.0,
            "minTrackedPlayers": int,
            ... other filters
        }
    }

    Server responds:
    {
        "action": "result",
        "entity": "play" or "player_play",
        "total_records": int,
        "filtered_records": int,
        "metrics": [...]
    }
    """
    await websocket.accept()

    # Get cached data (shared across all connections)
    try:
        play_records, player_play_records = get_cached_data()
    except Exception as e:
        await websocket.send_json({
            "error": f"Failed to load CSV data: {str(e)}",
            "action": "error"
        })
        await websocket.close()
        return

    try:
        while True:
            # Receive filter criteria from client
            data = await websocket.receive_json()

            action = data.get('action', 'compute')
            entity = data.get('entity', 'play')
            filters = data.get('filters', {})

            if action == 'compute':
                # Import here to avoid circular dependency
                from api import compute_metrics
                from transport_heads.http import PlayRecord

                # Select appropriate dataset
                records = play_records if entity == 'play' else player_play_records

                # Apply filters
                filtered_records = apply_filters(records, filters)

                # Convert to PlayRecord objects
                record_objects = [PlayRecord(**record) for record in filtered_records]

                # Compute metrics
                metrics = compute_metrics(record_objects)

                # Send response
                response = {
                    "action": "result",
                    "entity": entity,
                    "total_records": len(records),
                    "filtered_records": len(filtered_records),
                    "metrics": metrics
                }

                await websocket.send_json(response)

            elif action == 'ping':
                await websocket.send_json({"action": "pong"})

    except WebSocketDisconnect:
        print("Dashboard WebSocket client disconnected")
    except Exception as e:
        print(f"Dashboard WebSocket error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({
                "error": str(e),
                "action": "error"
            })
        except:
            pass


@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Advanced WebSocket endpoint for streaming updates.

    Allows client to send records incrementally and receive
    progressive metric updates.
    """
    await websocket.accept()

    accumulated_records = []

    try:
        while True:
            # Receive new batch of records
            data = await websocket.receive_json()

            action = data.get('action', 'add')

            if action == 'add':
                # Import here to avoid circular dependency
                from transport_heads.http import PlayRecord

                # Add new records
                new_records = [PlayRecord(**record) for record in data.get('records', [])]
                accumulated_records.extend(new_records)

                # Compute metrics on accumulated records
                from api import compute_metrics
                metrics = compute_metrics(accumulated_records)

                # Send update
                await websocket.send_json({
                    "action": "update",
                    "total_records": len(accumulated_records),
                    "metrics": metrics
                })

            elif action == 'clear':
                # Clear accumulated records
                accumulated_records = []
                await websocket.send_json({
                    "action": "cleared",
                    "total_records": 0,
                    "metrics": []
                })

            elif action == 'compute':
                # Just compute without adding
                from api import compute_metrics
                metrics = compute_metrics(accumulated_records)

                await websocket.send_json({
                    "action": "result",
                    "total_records": len(accumulated_records),
                    "metrics": metrics
                })

    except WebSocketDisconnect:
        print("WebSocket streaming client disconnected")
    except Exception as e:
        print(f"WebSocket streaming error: {e}")
        try:
            await websocket.send_json({
                "error": str(e),
                "action": "error"
            })
        except:
            pass
