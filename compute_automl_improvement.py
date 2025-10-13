#!/usr/bin/env python3
"""
Compute AutoML improvement statistics from a function bank.

This script analyzes a function bank JSON file to determine:
- How many entries were AutoML optimized
- The average improvement in metrics for optimized entries
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def load_function_bank(run_dir: Path) -> List[Dict]:
    """Load the function bank JSON file from a run directory."""
    func_bank_path = run_dir / "preprocessing_func_bank.json"

    if not func_bank_path.exists():
        raise FileNotFoundError(f"Function bank not found at {func_bank_path}")

    with open(func_bank_path, 'r') as f:
        return json.load(f)


def compute_improvements(func_bank: List[Dict]) -> Tuple[int, Dict[str, List[Tuple[float, float, int]]]]:
    """
    Compute improvements for AutoML optimized entries.

    Returns:
        - Number of AutoML optimized entries
        - Dictionary mapping metric names to list of (improvement, original_value, optimized_index) tuples
    """
    optimized_count = 0
    improvements_by_metric = {}

    for idx, entry in enumerate(func_bank):
        if not entry.get('automl_optimized', False):
            continue

        optimized_count += 1

        # Get source index
        source_idx = entry.get('automl_source_index')
        if source_idx is None:
            print(f"Warning: AutoML optimized entry missing automl_source_index")
            continue

        # Validate source index
        if source_idx < 0 or source_idx >= len(func_bank):
            print(f"Warning: Invalid source index {source_idx}")
            continue

        source_entry = func_bank[source_idx]

        # Verify source entry is superseded
        if not source_entry.get('automl_superseded', False):
            print(f"Warning: Source entry at index {source_idx} not marked as superseded")

        # Get metrics
        optimized_metrics = entry.get('overall_metrics', {})
        source_metrics = source_entry.get('overall_metrics', {})

        # Compute improvement for each metric
        for metric_name in optimized_metrics.keys():
            if metric_name not in source_metrics:
                print(f"Warning: Metric {metric_name} not found in source entry")
                continue

            original_value = source_metrics[metric_name]
            improvement = optimized_metrics[metric_name] - original_value

            if metric_name not in improvements_by_metric:
                improvements_by_metric[metric_name] = []

            improvements_by_metric[metric_name].append((improvement, original_value, source_idx, idx))

    return optimized_count, improvements_by_metric


def print_statistics(total_entries: int, optimized_count: int,
                    improvements_by_metric: Dict[str, List[Tuple[float, float]]]):
    """Print improvement statistics."""
    print(f"Total entries in function bank: {total_entries}")
    print(f"AutoML optimized entries: {optimized_count}")
    print()

    if optimized_count == 0:
        print("No AutoML optimized entries found.")
        return

    print("Improvement Statistics:")
    print("-" * 60)

    for metric_name, improvement_tuples in improvements_by_metric.items():
        if not improvement_tuples:
            continue

        # Extract improvements and calculate percentages
        improvements = [imp for imp, _, _, _ in improvement_tuples]
        pct_improvements = [(imp / orig * 100) if orig != 0 else 0
                           for imp, orig, _, _ in improvement_tuples]

        avg_improvement = sum(improvements) / len(improvements)
        avg_pct_improvement = sum(pct_improvements) / len(pct_improvements)
        min_improvement = min(improvements)
        max_improvement = max(improvements)
        min_pct_improvement = min(pct_improvements)
        max_pct_improvement = max(pct_improvements)

        print(f"\nMetric: {metric_name}")
        print(f"  Average improvement: {avg_improvement:.6f} ({avg_pct_improvement:+.2f}%)")
        print(f"  Min improvement:     {min_improvement:.6f} ({min_pct_improvement:+.2f}%)")
        print(f"  Max improvement:     {max_improvement:.6f} ({max_pct_improvement:+.2f}%)")
        print(f"  Improvements: {len(improvements)} entries")

        # Show individual improvements
        print(f"\n  Individual improvements:")
        for i, (imp, orig, source_idx, new_idx) in enumerate(improvement_tuples, 1):
            pct = (imp / orig * 100) if orig != 0 else 0
            print(f"    {i:2d}. [idx: {source_idx}->{new_idx}]: {imp:+.6f} ({pct:+.2f}%)")


def main():
    if len(sys.argv) < 2:
        print("Usage: compute_automl_improvement.py <run_directory>")
        print()
        print("Example:")
        print("  compute_automl_improvement.py automl_runs_diff_seed/spot_detection/latest")
        sys.exit(1)

    run_dir = Path(sys.argv[1])

    if not run_dir.exists():
        print(f"Error: Directory {run_dir} does not exist")
        sys.exit(1)

    try:
        func_bank = load_function_bank(run_dir)
        optimized_count, improvements_by_metric = compute_improvements(func_bank)
        print_statistics(len(func_bank), optimized_count, improvements_by_metric)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
