"""
Export utilities for RLHF datasets.
Supports JSON and CSV formats with SHA-256 hashing for REPPO compatibility.

RLHF relevance: Proper data export ensures datasets can be:
1. Verified for integrity (SHA-256 hashes)
2. Loaded by various training frameworks
3. Published on platforms like REPPO
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config import Config
from data.schema import DatasetManifest, RLHFDataPoint


def generate_json_content(data_points: "list[RLHFDataPoint]") -> str:
    """
    Generate JSON content as a string (for browser downloads).

    Args:
        data_points: List of RLHF data points to export

    Returns:
        JSON string content

    RLHF relevance: JSON is the standard format for RLHF/DPO datasets.
    Each record includes SHA-256 hash for data integrity verification.
    """
    if not data_points:
        raise ValueError("No data points to export")

    records = [dp.model_dump() for dp in data_points]
    return json.dumps(records, indent=2, ensure_ascii=False)


def generate_manifest_content(data_points: "list[RLHFDataPoint]") -> str:
    """Generate manifest JSON content as a string."""
    if not data_points:
        raise ValueError("No data points to export")

    domains = list(set(dp.metadata.domain for dp in data_points))
    task_types = list(set(dp.metadata.task_type for dp in data_points))

    complexity_dist = {}
    for dp in data_points:
        complexity_dist[dp.metadata.complexity] = complexity_dist.get(
            dp.metadata.complexity, 0
        ) + 1

    avg_chosen = sum(dp.metadata.chosen_score.total for dp in data_points) / len(
        data_points
    )
    avg_rejected = sum(dp.metadata.rejected_score.total for dp in data_points) / len(
        data_points
    )

    manifest = DatasetManifest(
        total_records=len(data_points),
        domains=domains,
        task_types=task_types,
        complexity_distribution=complexity_dist,
        avg_chosen_score=round(avg_chosen, 4),
        avg_rejected_score=round(avg_rejected, 4),
    )

    return json.dumps(manifest.model_dump(), indent=2)


def export_to_json(
    data_points: "list[RLHFDataPoint]",
    filepath: "str | None" = None,
    include_manifest: bool = True,
) -> "tuple[str, str | None]":
    """
    Export data points to JSON format (file-based, for local use).

    Args:
        data_points: List of RLHF data points to export
        filepath: Output path (auto-generated if None)
        include_manifest: Whether to create a separate manifest file

    Returns:
        Tuple of (data_filepath, manifest_filepath or None)

    RLHF relevance: JSON is the standard format for RLHF/DPO datasets.
    Each record includes SHA-256 hash for data integrity verification.
    """
    if not data_points:
        raise ValueError("No data points to export")

    # Generate filepath if not provided
    if filepath is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filepath = Config.get_output_path(f"rlhf_dataset_{timestamp}.json")

    # Write main data file
    content = generate_json_content(data_points)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    manifest_path = None
    if include_manifest:
        manifest_path = _create_manifest(data_points, filepath)

    return filepath, manifest_path


def _flatten_data_points(data_points: "list[RLHFDataPoint]") -> "list[dict]":
    """Flatten nested data point structure for CSV/tabular formats."""
    rows = []
    for dp in data_points:
        row = {
            "id": dp.id,
            "prompt": dp.prompt,
            "chosen": dp.chosen,
            "rejected": dp.rejected,
            "task_type": dp.metadata.task_type,
            "domain": dp.metadata.domain,
            "complexity": dp.metadata.complexity,
            "chosen_efficiency": dp.metadata.chosen_score.efficiency,
            "chosen_clarity": dp.metadata.chosen_score.clarity,
            "chosen_total": dp.metadata.chosen_score.total,
            "rejected_efficiency": dp.metadata.rejected_score.efficiency,
            "rejected_clarity": dp.metadata.rejected_score.clarity,
            "rejected_total": dp.metadata.rejected_score.total,
            "sha256": dp.metadata.sha256,
            "generated_at": dp.metadata.generated_at,
        }
        rows.append(row)
    return rows


def generate_csv_content(data_points: "list[RLHFDataPoint]") -> str:
    """
    Generate CSV content as a string (for browser downloads).

    Args:
        data_points: List of RLHF data points to export

    Returns:
        CSV string content

    RLHF relevance: CSV format enables spreadsheet analysis and
    compatibility with tools that don't support nested JSON.
    """
    if not data_points:
        raise ValueError("No data points to export")

    rows = _flatten_data_points(data_points)
    df = pd.DataFrame(rows)
    return df.to_csv(index=False, encoding="utf-8")


def export_to_csv(
    data_points: "list[RLHFDataPoint]",
    filepath: "str | None" = None,
) -> str:
    """
    Export data points to CSV format (file-based, for local use).

    Args:
        data_points: List of RLHF data points to export
        filepath: Output path (auto-generated if None)

    Returns:
        Output filepath

    RLHF relevance: CSV format enables spreadsheet analysis and
    compatibility with tools that don't support nested JSON.
    Note: Metadata is flattened into columns.
    """
    if not data_points:
        raise ValueError("No data points to export")

    if filepath is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filepath = Config.get_output_path(f"rlhf_dataset_{timestamp}.csv")

    content = generate_csv_content(data_points)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filepath


def generate_jsonl_content(data_points: "list[RLHFDataPoint]") -> str:
    """
    Generate JSONL content as a string (for browser downloads).

    This format is compatible with:
    - trl library for DPO training
    - HuggingFace datasets.load_dataset()

    RLHF relevance: Direct compatibility with popular training libraries.
    """
    if not data_points:
        raise ValueError("No data points to export")

    lines = []
    for dp in data_points:
        record = {
            "prompt": dp.prompt,
            "chosen": dp.chosen,
            "rejected": dp.rejected,
            "domain": dp.metadata.domain,
            "task_type": dp.metadata.task_type,
        }
        lines.append(json.dumps(record, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def export_to_huggingface_format(
    data_points: "list[RLHFDataPoint]",
    filepath: "str | None" = None,
) -> str:
    """
    Export in HuggingFace datasets format (file-based, for local use).

    This format is compatible with:
    - trl library for DPO training
    - HuggingFace datasets.load_dataset()

    RLHF relevance: Direct compatibility with popular training libraries.
    """
    if not data_points:
        raise ValueError("No data points to export")

    if filepath is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filepath = Config.get_output_path(f"rlhf_dataset_{timestamp}.jsonl")

    content = generate_jsonl_content(data_points)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filepath


def _create_manifest(
    data_points: "list[RLHFDataPoint]",
    data_filepath: str,
) -> str:
    """Create manifest file with dataset statistics."""
    # Calculate statistics
    domains = list(set(dp.metadata.domain for dp in data_points))
    task_types = list(set(dp.metadata.task_type for dp in data_points))

    complexity_dist = {}
    for dp in data_points:
        complexity_dist[dp.metadata.complexity] = complexity_dist.get(
            dp.metadata.complexity, 0
        ) + 1

    avg_chosen = sum(dp.metadata.chosen_score.total for dp in data_points) / len(
        data_points
    )
    avg_rejected = sum(dp.metadata.rejected_score.total for dp in data_points) / len(
        data_points
    )

    manifest = DatasetManifest(
        total_records=len(data_points),
        domains=domains,
        task_types=task_types,
        complexity_distribution=complexity_dist,
        avg_chosen_score=round(avg_chosen, 4),
        avg_rejected_score=round(avg_rejected, 4),
    )

    # Write manifest next to data file
    manifest_path = data_filepath.replace(".json", "_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(), f, indent=2)

    return manifest_path


def load_dataset(filepath: str) -> "list[RLHFDataPoint]":
    """
    Load a previously exported dataset.

    Args:
        filepath: Path to JSON dataset file

    Returns:
        List of RLHFDataPoint objects

    RLHF relevance: Enables resuming data generation or
    combining multiple datasets.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        records = json.load(f)

    return [RLHFDataPoint.model_validate(r) for r in records]


def verify_dataset_integrity(filepath: str) -> "tuple[bool, list[str]]":
    """
    Verify dataset integrity by checking SHA-256 hashes.

    Args:
        filepath: Path to JSON dataset file

    Returns:
        Tuple of (all_valid, list of invalid record IDs)

    RLHF relevance: Data integrity is crucial for reproducible training.
    REPPO requires hash verification for published datasets.
    """
    import hashlib

    data_points = load_dataset(filepath)
    invalid_ids = []

    for dp in data_points:
        # Recompute hash
        content = f"{dp.prompt}|{dp.chosen}|{dp.rejected}"
        expected_hash = hashlib.sha256(content.encode()).hexdigest()

        if dp.metadata.sha256 != expected_hash:
            invalid_ids.append(dp.id)

    return len(invalid_ids) == 0, invalid_ids


def get_dataset_stats(data_points: "list[RLHFDataPoint]") -> "dict[str, Any]":
    """
    Calculate comprehensive statistics for a dataset.

    Returns dict with:
    - total_records
    - domain_distribution
    - task_type_distribution
    - complexity_distribution
    - score_statistics (mean, std, min, max)
    - preference_gap (avg difference between chosen and rejected)

    RLHF relevance: Statistics help evaluate dataset quality and balance.
    """
    import statistics

    if not data_points:
        return {"total_records": 0}

    chosen_scores = [dp.metadata.chosen_score.total for dp in data_points]
    rejected_scores = [dp.metadata.rejected_score.total for dp in data_points]
    gaps = [c - r for c, r in zip(chosen_scores, rejected_scores)]

    # Distribution counts
    domain_dist = {}
    task_dist = {}
    complexity_dist = {}

    for dp in data_points:
        domain_dist[dp.metadata.domain] = domain_dist.get(dp.metadata.domain, 0) + 1
        task_dist[dp.metadata.task_type] = task_dist.get(dp.metadata.task_type, 0) + 1
        complexity_dist[dp.metadata.complexity] = (
            complexity_dist.get(dp.metadata.complexity, 0) + 1
        )

    return {
        "total_records": len(data_points),
        "domain_distribution": domain_dist,
        "task_type_distribution": task_dist,
        "complexity_distribution": complexity_dist,
        "chosen_score_stats": {
            "mean": round(statistics.mean(chosen_scores), 4),
            "std": round(statistics.stdev(chosen_scores), 4) if len(chosen_scores) > 1 else 0,
            "min": round(min(chosen_scores), 4),
            "max": round(max(chosen_scores), 4),
        },
        "rejected_score_stats": {
            "mean": round(statistics.mean(rejected_scores), 4),
            "std": round(statistics.stdev(rejected_scores), 4) if len(rejected_scores) > 1 else 0,
            "min": round(min(rejected_scores), 4),
            "max": round(max(rejected_scores), 4),
        },
        "preference_gap": {
            "mean": round(statistics.mean(gaps), 4),
            "std": round(statistics.stdev(gaps), 4) if len(gaps) > 1 else 0,
            "min": round(min(gaps), 4),
            "max": round(max(gaps), 4),
        },
    }
