#!/usr/bin/env python3
"""
Compute agreement scores between model predictions and PFF labels.

This script calculates agreement metrics between:
- Model A (CV tracking) and PFF labels
- Model B (GT tracking) and PFF labels
- Model C (Batch API) and PFF labels

Usage:
    python src/compute_attribute_agreement_scores_from_eval_run.py <eval_run_id> [--output agreement.csv]
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import polars as pl
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn

from data_loaders.dbrks_polars import scan_table
from eval_metrics.eval_fai_response import parse_play_predictions, parse_player_play_predictions

console = Console()


def load_pff_labels(play_ids: List[str] = None, catalog: str = "prd", parquet_path: str = "playlist_plays_with_labels.parquet") -> pl.DataFrame:
    """
    Load PFF labels from parquet file.

    Args:
        play_ids: List of play IDs to filter to
        catalog: Not used, kept for API compatibility
        parquet_path: Path to the parquet file with PFF labels

    Returns:
        DataFrame with PFF labels indexed by sumer_play_id
    """
    console.print(f"\n[cyan]Loading PFF labels from {parquet_path}...[/cyan]")

    try:
        # Load from parquet file
        pff_df = pl.read_parquet(parquet_path)

        # Filter by play_ids if provided
        if play_ids and len(play_ids) > 0:
            pff_df = pff_df.filter(pl.col("sumer_play_id").is_in(play_ids))

        console.print(f"[green]✓ Loaded {len(pff_df)} PFF labels[/green]")
        return pff_df

    except Exception as e:
        console.print(f"[red]Error loading PFF labels: {e}[/red]")
        return pl.DataFrame()


def load_eval_run_predictions(eval_run_id: str, output_root: str = "outputs") -> List[Tuple[Dict, Dict, Dict]]:
    """
    Load all prediction results from an eval_run_id directory.

    Returns:
        List of tuples: (results_dict, metadata_dict, tracking_metrics_dict)
    """
    eval_run_path = Path(output_root) / eval_run_id

    if not eval_run_path.exists():
        console.print(f"[red]Error: Eval run directory not found: {eval_run_path}[/red]")
        return []

    all_results = []

    # Walk through game_id/play_id directories
    for game_dir in eval_run_path.iterdir():
        if not game_dir.is_dir():
            continue

        for play_dir in game_dir.iterdir():
            if not play_dir.is_dir():
                continue

            results_file = play_dir / "results.json"
            metadata_file = play_dir / "metadata.json"

            if results_file.exists() and metadata_file.exists():
                try:
                    import json

                    with open(results_file, "r") as f:
                        results = json.load(f)
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)

                    # Extract tracking metrics from core_metrics
                    tracking_metrics = {}
                    if "core_metrics" in results:
                        core_metrics = results["core_metrics"]
                        tracking_metrics = {
                            "tracked_players": core_metrics.get("missing_players", {}).get("tracked_players"),
                            "total_gt_tracks": core_metrics.get("missing_players", {}).get("total_gt_tracks"),
                            "completeness_0_to_10": core_metrics.get("completeness_by_frame", {}).get("frames_0_to_10"),
                            "completeness_0_to_30": core_metrics.get("completeness_by_frame", {}).get("frames_0_to_30"),
                            "completeness_0_to_end": core_metrics.get("completeness_by_frame", {}).get("frames_0_to_end"),
                            "reliable_players_0_to_10": core_metrics.get("reliably_tracked_players_by_frame", {}).get("frames_0_to_10", {}).get("count"),
                            "reliable_players_0_to_30": core_metrics.get("reliably_tracked_players_by_frame", {}).get("frames_0_to_30", {}).get("count"),
                            "reliable_players_0_to_end": core_metrics.get("reliably_tracked_players_by_frame", {}).get("frames_0_to_end", {}).get("count"),
                            "overall_median_pos_error_yards": core_metrics.get("overall_median_pos_error_yards"),
                            "worst_case_avg_pos_error_yards": core_metrics.get("worst_case_avg_pos_error_yards"),
                        }

                    all_results.append((results, metadata, tracking_metrics))
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to load {results_file}: {e}[/yellow]")

    return all_results


def load_batch_predictions(all_results: List[Tuple[Dict, Dict, Dict]], catalog: str = "prd") -> Dict[str, Dict]:
    """
    Pre-load all batch predictions (Model C) for the plays.

    Returns:
        Dict mapping play_id -> batch_predictions
    """
    console.print(f"\n[cyan]Pre-loading batch predictions (Model C)...[/cyan]")

    play_ids = [metadata["sumer_play_id"] for _, metadata, _ in all_results]

    try:
        lf = scan_table(table_name="fai_preds_plays", catalog_name=catalog, schema_name="football_ai")

        plays_df = lf.filter(pl.col("sumer_play_id").is_in(play_ids)).collect()

        console.print(f"  ✓ Loaded {len(plays_df)} batch predictions")

        # Index by play_id
        batch_cache = {}
        for row in plays_df.iter_rows(named=True):
            play_id = row["sumer_play_id"]
            batch_cache[play_id] = row

        return batch_cache

    except Exception as e:
        console.print(f"[yellow]Warning: Failed to load batch predictions: {e}[/yellow]")
        return {}


def load_target_metadata(csv_path: str = "fai_target_metadata.csv") -> pl.DataFrame:
    """
    Load target metadata from CSV file.

    Returns:
        DataFrame with metadata indexed by attribute
    """
    console.print(f"\n[cyan]Loading target metadata from {csv_path}...[/cyan]")

    try:
        metadata_df = pl.read_csv(csv_path)

        console.print(f"  ✓ Loaded {len(metadata_df)} target metadata records")
        return metadata_df

    except Exception as e:
        console.print(f"[yellow]Warning: Failed to load target metadata: {e}[/yellow]")
        return pl.DataFrame()


def calculate_pairwise_agreement(
    model_preds: Dict[str, any], pff_labels: Dict[str, any], attribute: str
) -> Dict[str, float]:
    """
    Calculate agreement between a model and PFF for a specific attribute.

    Args:
        model_preds: Model predictions {attribute: value}
        pff_labels: PFF labels {attribute: value}
        attribute: The attribute to compare

    Returns:
        Dict with agreement metrics
    """
    model_val = model_preds.get(attribute)
    pff_val = pff_labels.get(attribute)

    if model_val is None or pff_val is None:
        return {}

    # Calculate agreement
    agrees = model_val == pff_val

    return {"agrees": 1 if agrees else 0, "model_val": model_val, "pff_val": pff_val}


def calculate_hardness_function(
    model_a_val: any, model_b_val: any, model_c_val: any, pff_val: any
) -> bool:
    """
    Calculate hardness: True if models disagree OR any model disagrees with PFF.

    Args:
        model_a_val: Model A prediction
        model_b_val: Model B prediction
        model_c_val: Model C prediction
        pff_val: PFF label

    Returns:
        True if the instance is "hard", False otherwise
    """
    # Hard if models disagree with each other
    models_disagree = not (model_a_val == model_b_val == model_c_val)

    # Hard if any model disagrees with PFF
    any_disagrees_with_pff = (model_a_val != pff_val) or (model_b_val != pff_val) or (model_c_val != pff_val)

    return models_disagree or any_disagrees_with_pff


def aggregate_agreement_metrics(
    all_results: List[Tuple[Dict, Dict, Dict]], pff_df: pl.DataFrame, batch_cache: Dict, entity: str = "play"
) -> Dict[str, Dict]:
    """
    Aggregate agreement metrics across all plays.

    Args:
        all_results: List of (results, metadata, tracking_metrics) tuples
        pff_df: DataFrame with PFF labels
        batch_cache: Pre-loaded batch predictions
        entity: "play" or "player_play"

    Returns:
        Dict mapping attribute -> {
            "model_a_agreements": [...],
            "model_b_agreements": [...],
            "model_c_agreements": [...],
            "hardness": [...],
            "pff_values": [...],
            "tracking_metrics": [...],
            ...
        }
    """
    aggregated = defaultdict(
        lambda: {
            "model_a_agreements": [],
            "model_b_agreements": [],
            "model_c_agreements": [],
            "hardness": [],
            "pff_values": [],
            "model_a_values": [],
            "model_b_values": [],
            "model_c_values": [],
            "tracking_metrics": [],
        }
    )

    # Debug counters
    debug_counts = {
        "total_plays": 0,
        "missing_fai_preds": 0,
        "missing_batch": 0,
        "missing_pff": 0,
        "processed": 0,
    }

    # Build PFF column mapping once (target_id -> attribute)
    pff_column_mapping = _build_pff_column_mapping()
    console.print(f"[cyan]Loaded {len(pff_column_mapping)} target_id->attribute mappings[/cyan]")

    # Debug: save PFF columns to file
    if len(pff_df) > 0:
        pff_columns = sorted(pff_df.columns)
        console.print(f"[cyan]PFF table has {len(pff_columns)} columns[/cyan]")
        with open("pff_columns.txt", "w") as f:
            for col in pff_columns:
                f.write(f"{col}\n")
        console.print(f"[cyan]Saved PFF columns to pff_columns.txt[/cyan]")

    console.print(f"\n[cyan]Aggregating {entity}-level agreement metrics...[/cyan]")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("({task.completed}/{task.total})"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing plays", total=len(all_results))

        for results, metadata, tracking_metrics in all_results:
            debug_counts["total_plays"] += 1
            play_id = metadata.get("sumer_play_id")

            try:
                # Get FAI predictions for both models
                fai_preds = results.get("fai_predictions", {})

                if not fai_preds.get("gt_success") or not fai_preds.get("cv_success"):
                    debug_counts["missing_fai_preds"] += 1
                    progress.update(task, advance=1)
                    continue

                gt_response = fai_preds.get("gt_response")  # Model B (GT tracking)
                cv_response = fai_preds.get("cv_response")  # Model A (CV tracking)

                if not gt_response or not cv_response:
                    debug_counts["missing_fai_preds"] += 1
                    progress.update(task, advance=1)
                    continue

                # Parse predictions
                if entity == "play":
                    model_a_preds, _ = parse_play_predictions(cv_response)
                    model_b_preds, _ = parse_play_predictions(gt_response)
                else:  # player_play
                    model_a_preds, _ = parse_player_play_predictions(cv_response)
                    model_b_preds, _ = parse_player_play_predictions(gt_response)

                # Get Model C (Batch) predictions
                model_c_preds = {}
                if play_id in batch_cache:
                    batch_row = batch_cache[play_id]
                    # Convert batch row to predictions dict
                    model_c_preds = _extract_batch_attributes(batch_row, entity)
                else:
                    debug_counts["missing_batch"] += 1

                # Get PFF labels for this play
                pff_row = pff_df.filter(pl.col("sumer_play_id") == play_id)
                if len(pff_row) == 0:
                    debug_counts["missing_pff"] += 1
                    progress.update(task, advance=1)
                    continue

                pff_labels = _extract_pff_attributes(pff_row[0], entity, pff_column_mapping)

                # Debug first play to see what's happening
                if debug_counts["total_plays"] == 1:
                    console.print(f"\n[yellow]DEBUG - First play attribute counts:[/yellow]")
                    console.print(f"  Model A (CV): {len(model_a_preds)} attributes")
                    console.print(f"  Model B (GT): {len(model_b_preds)} attributes")
                    console.print(f"  Model C (Batch): {len(model_c_preds)} attributes")
                    console.print(f"  PFF labels: {len(pff_labels)} attributes")
                    console.print(f"  Model A sample: {list(model_a_preds.keys())[:10]}")
                    console.print(f"  PFF sample: {list(pff_labels.keys())[:10]}")

                # Find common attributes across all 4 sources
                common_attrs = (
                    set(model_a_preds.keys())
                    .intersection(set(model_b_preds.keys()))
                    .intersection(set(model_c_preds.keys()))
                    .intersection(set(pff_labels.keys()))
                )

                # Track if we processed at least one attribute for this play
                processed_any = False

                for attr in common_attrs:
                    model_a_val = model_a_preds[attr]
                    model_b_val = model_b_preds[attr]
                    model_c_val = model_c_preds[attr]
                    pff_val = pff_labels[attr]

                    # Skip if any value is None
                    if model_a_val is None or model_b_val is None or model_c_val is None or pff_val is None:
                        continue

                    processed_any = True

                    # Calculate agreements
                    aggregated[attr]["model_a_agreements"].append(1 if model_a_val == pff_val else 0)
                    aggregated[attr]["model_b_agreements"].append(1 if model_b_val == pff_val else 0)
                    aggregated[attr]["model_c_agreements"].append(1 if model_c_val == pff_val else 0)

                    # Calculate hardness
                    is_hard = calculate_hardness_function(model_a_val, model_b_val, model_c_val, pff_val)
                    aggregated[attr]["hardness"].append(is_hard)

                    # Store values
                    aggregated[attr]["pff_values"].append(pff_val)
                    aggregated[attr]["model_a_values"].append(model_a_val)
                    aggregated[attr]["model_b_values"].append(model_b_val)
                    aggregated[attr]["model_c_values"].append(model_c_val)

                    # Store tracking metrics for this instance
                    aggregated[attr]["tracking_metrics"].append(tracking_metrics)

                if processed_any:
                    debug_counts["processed"] += 1

                progress.update(task, advance=1)

            except Exception as e:
                console.print(f"[yellow]Warning: Error processing {play_id}: {e}[/yellow]")
                progress.update(task, advance=1)
                continue

    # Print debug statistics
    console.print("\n[cyan]Filtering Statistics:[/cyan]")
    console.print(f"  Total plays in eval_run: {debug_counts['total_plays']}")
    console.print(f"  Missing FAI predictions (GT/CV): {debug_counts['missing_fai_preds']}")
    console.print(f"  Missing batch predictions: {debug_counts['missing_batch']}")
    console.print(f"  Missing PFF labels: {debug_counts['missing_pff']}")
    console.print(f"  Plays with at least one valid attribute: {debug_counts['processed']}")

    return dict(aggregated)


def _extract_batch_attributes(batch_row: Dict, entity: str) -> Dict[str, any]:
    """Extract attributes from a batch prediction row."""
    # Comprehensive attribute mapping from compute_attribute_skill_scores_from_eval_run.py
    attrs = {}

    # Continuous attributes (use API names as keys)
    for db_col, api_name in [
        ("yards_gained", "yards_gained"),
        ("expected_points", "ep"),
        ("expected_points_added", "epa"),
        ("time_to_throw", "time_to_throw"),
        ("time_to_pressure", "time_to_pressure"),
        ("target_depth", "target_depth"),
        ("target_width", "target_width"),
        ("dropback_depth", "dropback_depth"),
    ]:
        if batch_row.get(db_col) is not None:
            attrs[api_name] = batch_row[db_col]

    # Boolean attributes (use API names as keys)
    for db_col, api_name in [
        ("first_down", "first_down"),
        ("touchdown", "touchdown"),
        ("expected_points_added_success", "epa_success"),
        ("trick_look", "trick_look"),
        ("trick_played", "trick_played"),
        ("screen", "screen"),
        ("play_action", "play_action"),
        ("run_pass_option", "rpo"),
        ("completion", "completion"),
        ("qb_pressured", "qb_pressured"),
        ("qb_left_pocket", "qb_left_pocket"),
        ("qb_scramble", "qb_scramble"),
        ("schemed_blitz", "schemed_blitz"),
        ("stunt", "stunt"),
        ("option_run", "option_run"),
        ("read_option_run", "read_option_run"),
        ("speed_option_run", "speed_option_run"),
        ("designed_qb_run", "designed_qb_run"),
        ("jet_sweep_run", "jet_sweep_run"),
        ("end_around_run", "end_around_run"),
        ("reverse_run", "reverse_run"),
        ("pitch_run", "pitch_run"),
        ("draw", "draw"),
        ("lead_run", "lead_run"),
        ("cross_lead_run", "cross_lead_run"),
        ("split_run", "split_run"),
        ("inverted_run", "inverted_run"),
    ]:
        if batch_row.get(db_col) is not None:
            attrs[api_name] = batch_row[db_col]

    # Categorical attributes (use API names as keys)
    for db_col, api_name in [
        ("formation_left", "formation_left"),
        ("formation_right", "formation_right"),
        ("qb_alignment", "qb_alignment"),
        ("middle_of_field_coverage_look", "mof_coverage_look"),
        ("middle_of_field_coverage_played", "mof_coverage_played"),
        ("run_pass_intent", "play_type_intent"),
        ("pass_outcome", "pass_outcome"),
        ("run_concept", "run_concept"),
        ("run_side", "run_side"),
        ("run_gap_intent", "run_gap_intent"),
        ("run_gap_intent_side", "run_gap_intent_side"),
        ("run_gap_outcome", "run_gap_outcome"),
        ("run_gap_outcome_side", "run_gap_outcome_side"),
        ("coverage_scheme", "coverage_scheme"),
        ("man_zone_coverage", "man_zone_coverage"),
        ("qb_scramble_side", "qb_scramble_side"),
    ]:
        if batch_row.get(db_col) is not None:
            attrs[api_name] = batch_row[db_col]

    return attrs


def _build_pff_column_mapping() -> Dict[str, str]:
    """
    Build mapping from target_id (PFF column names) to attribute (API names).
    Reads from fai_target_metadata.csv.

    Returns:
        Dict mapping target_id -> attribute
    """
    try:
        metadata_df = pl.read_csv("fai_target_metadata.csv")
        mapping = {}
        for row in metadata_df.iter_rows(named=True):
            target_id = row.get("target_id")
            attribute = row.get("attribute")
            if target_id and attribute:
                mapping[target_id] = attribute
        return mapping
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to load PFF column mapping: {e}[/yellow]")
        return {}


def _extract_pff_attributes(pff_row: pl.DataFrame, entity: str, column_mapping: Dict[str, str] = None) -> Dict[str, any]:
    """
    Extract attributes from a PFF label row.
    The fai_labels_plays_backup table already uses FAI attribute names as column names.

    Args:
        pff_row: Single row from fai_labels_plays_backup table
        entity: 'play' or 'player_play'
        column_mapping: Not used, kept for API compatibility

    Returns:
        Dict mapping attribute names to values
    """
    attrs = {}

    # Convert row to dict
    row_dict = pff_row.to_dicts()[0] if hasattr(pff_row, "to_dicts") else dict(pff_row)

    # Metadata columns to skip
    skip_columns = {
        'sumer_play_id', 'sumer_game_id', 'season', 'week', 'league', 'game_date_week'
    }

    # Iterate through all columns and extract attribute values
    for col_name, value in row_dict.items():
        # Skip metadata columns
        if col_name in skip_columns:
            continue

        # Rename pass_depth to target_depth for consistency with FAI API
        if col_name == 'pass_depth':
            col_name = 'target_depth'
        elif col_name == 'pass_width':
            col_name = 'target_width'
        elif col_name == 'time_to_pass':
            col_name = 'time_to_throw'
        elif col_name == 'pressure':
            col_name = 'qb_pressured'

        # Only include non-null values
        if value is not None:
            attrs[col_name] = value

    return attrs


def compute_agreement_summary(aggregated: Dict[str, Dict], metadata_df: pl.DataFrame = None) -> List[Dict]:
    """
    Compute summary agreement statistics per attribute.

    Args:
        aggregated: Aggregated agreement data per attribute
        metadata_df: Target metadata with pre-computed scores

    Returns:
        List of dicts with agreement statistics
    """
    import numpy as np

    results = []

    console.print("\n[cyan]Computing agreement statistics...[/cyan]")

    # Create a lookup dict for metadata by attribute
    metadata_lookup = {}
    if metadata_df is not None and len(metadata_df) > 0:
        for row in metadata_df.iter_rows(named=True):
            attribute = row.get("attribute")
            if attribute:
                metadata_lookup[attribute] = row

    for attr, data in aggregated.items():
        n = len(data["model_a_agreements"])

        if n == 0:
            continue

        # Overall agreement rates
        A_a_p = sum(data["model_a_agreements"]) / n  # Model A vs PFF
        A_b_p = sum(data["model_b_agreements"]) / n  # Model B vs PFF
        A_c_p = sum(data["model_c_agreements"]) / n  # Model C vs PFF

        # Agreement on hard instances
        hard_indices = [i for i, h in enumerate(data["hardness"]) if h]
        easy_indices = [i for i, h in enumerate(data["hardness"]) if not h]

        A_a_p_hard = (
            sum(data["model_a_agreements"][i] for i in hard_indices) / len(hard_indices) if hard_indices else 0
        )
        A_b_p_hard = (
            sum(data["model_b_agreements"][i] for i in hard_indices) / len(hard_indices) if hard_indices else 0
        )
        A_c_p_hard = (
            sum(data["model_c_agreements"][i] for i in hard_indices) / len(hard_indices) if hard_indices else 0
        )

        # Agreement on easy instances
        A_a_p_easy = (
            sum(data["model_a_agreements"][i] for i in easy_indices) / len(easy_indices) if easy_indices else 0
        )
        A_b_p_easy = (
            sum(data["model_b_agreements"][i] for i in easy_indices) / len(easy_indices) if easy_indices else 0
        )
        A_c_p_easy = (
            sum(data["model_c_agreements"][i] for i in easy_indices) / len(easy_indices) if easy_indices else 0
        )

        # Compute average tracking metrics
        tracking_metrics_list = data["tracking_metrics"]
        avg_tracking_metrics = {}

        if tracking_metrics_list:
            metrics_keys = [
                "tracked_players",
                "total_gt_tracks",
                "completeness_0_to_10",
                "completeness_0_to_30",
                "completeness_0_to_end",
                "reliable_players_0_to_10",
                "reliable_players_0_to_30",
                "reliable_players_0_to_end",
                "overall_median_pos_error_yards",
                "worst_case_avg_pos_error_yards",
            ]

            for key in metrics_keys:
                values = [m.get(key) for m in tracking_metrics_list if m.get(key) is not None]
                if values:
                    avg_tracking_metrics[f"avg_{key}"] = round(np.mean(values), 4)

        # Get metadata scores for this attribute (if available)
        batch_metadata = {}
        if attr in metadata_lookup:
            meta = metadata_lookup[attr]
            batch_metadata = {
                "batch_full_internal_agreement_score": meta.get("internal_agreement_score"),
                "batch_full_pff_agreement_score": meta.get("pff_agreement_score"),
                "batch_full_fai_agreement_score": meta.get("fai_agreement_score"),
                "batch_full_accuracy_score": meta.get("accuracy_score"),
                "batch_full_skill_score": meta.get("skill_score"),
                "batch_full_status": meta.get("status"),
            }

        results.append(
            {
                "attribute": attr,
                "n_instances": n,
                "n_hard": len(hard_indices),
                "n_easy": len(easy_indices),
                "A_model_a_pff": round(A_a_p, 4),
                "A_model_b_pff": round(A_b_p, 4),
                "A_model_c_pff": round(A_c_p, 4),
                "A_model_a_pff_hard": round(A_a_p_hard, 4),
                "A_model_b_pff_hard": round(A_b_p_hard, 4),
                "A_model_c_pff_hard": round(A_c_p_hard, 4),
                "A_model_a_pff_easy": round(A_a_p_easy, 4),
                "A_model_b_pff_easy": round(A_b_p_easy, 4),
                "A_model_c_pff_easy": round(A_c_p_easy, 4),
                **avg_tracking_metrics,
                **batch_metadata,
            }
        )

    return results


def save_to_csv(summary_results: List[Dict], output_file: str):
    """Save agreement summary to CSV file."""
    import csv

    console.print(f"\n[cyan]Saving results to {output_file}...[/cyan]")

    if not summary_results:
        console.print("[yellow]No results to save![/yellow]")
        return

    # Collect all possible fieldnames from all results
    all_fieldnames = set()
    for row in summary_results:
        all_fieldnames.update(row.keys())

    # Sort fieldnames for consistent output: base fields, batch metadata, then tracking metrics
    base_fields = [
        "attribute",
        "n_instances",
        "n_hard",
        "n_easy",
        "A_model_a_pff",
        "A_model_b_pff",
        "A_model_c_pff",
        "A_model_a_pff_hard",
        "A_model_b_pff_hard",
        "A_model_c_pff_hard",
        "A_model_a_pff_easy",
        "A_model_b_pff_easy",
        "A_model_c_pff_easy",
    ]
    batch_metadata_fields = [
        "batch_full_internal_agreement_score",
        "batch_full_pff_agreement_score",
        "batch_full_fai_agreement_score",
        "batch_full_accuracy_score",
        "batch_full_skill_score",
        "batch_full_status",
    ]
    tracking_fields = sorted([f for f in all_fieldnames if f.startswith("avg_")])
    fieldnames = base_fields + [f for f in batch_metadata_fields if f in all_fieldnames] + tracking_fields

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_results)

    console.print(f"[green]✓ Results saved to {output_file}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Compute agreement scores between models and PFF labels")
    parser.add_argument("eval_run_id", help="The eval_run_id to analyze (UUID)")
    parser.add_argument("--output", default="agreement_scores.csv", help="Output CSV file path")
    parser.add_argument("--output-root", default="outputs", help="Root directory containing eval runs")
    parser.add_argument("--catalog", default="prd", help="Databricks catalog to use")

    args = parser.parse_args()

    console.rule(f"[bold white]AGREEMENT ANALYSIS FOR {args.eval_run_id}[/bold white]")

    # Load eval run predictions first to get play IDs
    console.print(f"\n[cyan]Loading predictions from {args.output_root}/{args.eval_run_id}/...[/cyan]")
    all_results = load_eval_run_predictions(args.eval_run_id, args.output_root)

    if not all_results:
        console.print("[red]No results found![/red]")
        return

    console.print(f"[green]✓ Loaded {len(all_results)} plays[/green]")

    # Extract play IDs from results
    play_ids = [metadata.get("sumer_play_id") for _, metadata, _ in all_results]

    # Load PFF labels only for these plays (much faster!)
    pff_df = load_pff_labels(play_ids=play_ids, catalog=args.catalog)
    if len(pff_df) == 0:
        console.print("[red]Failed to load PFF labels. Exiting.[/red]")
        return

    # Pre-load batch predictions (Model C)
    batch_cache = load_batch_predictions(all_results, catalog=args.catalog)

    # Load target metadata with pre-computed scores (from CSV)
    metadata_df = load_target_metadata(csv_path="fai_target_metadata.csv")

    # Aggregate agreement metrics (play-level only for now)
    console.rule("[bold]Play-Level Agreement Analysis[/bold]")
    aggregated = aggregate_agreement_metrics(all_results, pff_df, batch_cache, entity="play")

    # Compute summary statistics
    summary_results = compute_agreement_summary(aggregated, metadata_df=metadata_df)
    console.print(f"[green]✓ Computed agreement for {len(summary_results)} attributes[/green]")

    # Save to CSV
    console.rule("[bold]Saving Results[/bold]")
    save_to_csv(summary_results, args.output)

    console.rule("[bold green]COMPLETE[/bold green]")


if __name__ == "__main__":
    main()
