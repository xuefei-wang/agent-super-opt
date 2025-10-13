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


def compute_improvements(func_bank: List[Dict], sampling_function) -> Tuple[int, List[Tuple[float, float, int, int]]]:
    """
    Compute improvements for AutoML optimized entries using the sampling function.

    Args:
        func_bank: List of function bank entries
        sampling_function: Function to extract the primary metric from an entry

    Returns:
        - Number of AutoML optimized entries
        - List of (improvement, original_value, source_idx, optimized_idx) tuples
    """
    optimized_count = 0
    improvements = []

    for idx, entry in enumerate(func_bank):
        if not entry.get('automl_optimized', False):
            continue

        optimized_count += 1

        # Get source index
        source_idx = entry.get('automl_source_index')
        if source_idx is None:
            print(f"Warning: AutoML optimized entry at index {idx} missing automl_source_index")
            continue

        # Validate source index
        if source_idx < 0 or source_idx >= len(func_bank):
            print(f"Warning: Invalid source index {source_idx} for entry {idx}")
            continue

        source_entry = func_bank[source_idx]

        # Verify source entry is superseded
        if not source_entry.get('automl_superseded', False):
            print(f"Warning: Source entry at index {source_idx} not marked as superseded")

        # Use sampling function to extract primary metric values
        try:
            optimized_value = sampling_function(entry)
            original_value = sampling_function(source_entry)

            if optimized_value is None or original_value is None:
                print(f"Warning: Could not extract metric values for entry {idx}")
                continue

            improvement = optimized_value - original_value
            improvements.append((improvement, original_value, source_idx, idx))

        except (KeyError, TypeError, AttributeError) as e:
            print(f"Warning: Error computing improvement for entry {idx}: {e}")
            continue

    return optimized_count, improvements


def print_statistics(total_entries: int, optimized_count: int,
                    improvements: List[Tuple[float, float, int, int]], experiment_name: str):
    """Print improvement statistics."""
    print(f"Total entries in function bank: {total_entries}")
    print(f"AutoML optimized entries: {optimized_count}")
    print()

    if optimized_count == 0:
        print("No AutoML optimized entries found.")
        return

    if not improvements:
        print("No improvements could be computed.")
        return

    print(f"Primary Metric Improvement Statistics ({experiment_name}):")
    print("-" * 60)

    # Extract improvements and calculate percentages
    improvement_values = [imp for imp, _, _, _ in improvements]
    pct_improvements = [(imp / orig * 100) if orig != 0 else 0
                       for imp, orig, _, _ in improvements]

    # Calculate effective improvements (only count positive improvements)
    effective_improvements = [max(0, imp) for imp in improvement_values]
    effective_pct_improvements = [(max(0, imp) / orig * 100) if orig != 0 else 0
                                  for imp, orig, _, _ in improvements]

    avg_improvement = sum(improvement_values) / len(improvement_values)
    avg_pct_improvement = sum(pct_improvements) / len(pct_improvements)
    avg_effective_improvement = sum(effective_improvements) / len(effective_improvements)
    avg_effective_pct = sum(effective_pct_improvements) / len(effective_pct_improvements)

    min_improvement = min(improvement_values)
    max_improvement = max(improvement_values)
    min_pct_improvement = min(pct_improvements)
    max_pct_improvement = max(pct_improvements)

    # Count positive vs negative improvements
    positive_count = sum(1 for imp in improvement_values if imp > 0)
    negative_count = sum(1 for imp in improvement_values if imp < 0)
    zero_count = sum(1 for imp in improvement_values if imp == 0)

    print(f"\n  Average improvement:           {avg_improvement:.6f} ({avg_pct_improvement:+.2f}%)")
    print(f"  Average effective improvement: {avg_effective_improvement:.6f} ({avg_effective_pct:+.2f}%)")
    print(f"  Min improvement:               {min_improvement:.6f} ({min_pct_improvement:+.2f}%)")
    print(f"  Max improvement:               {max_improvement:.6f} ({max_pct_improvement:+.2f}%)")
    print(f"  Total improvements:            {len(improvements)} entries")
    print(f"    Positive: {positive_count}, Negative: {negative_count}, Zero: {zero_count}")

    # Show individual improvements
    print(f"\n  Individual improvements:")
    for i, (imp, orig, source_idx, new_idx) in enumerate(improvements, 1):
        pct = (imp / orig * 100) if orig != 0 else 0
        optimized_value = orig + imp
        print(f"    {i:2d}. [idx: {source_idx}->{new_idx}]: {orig:.6f} -> {optimized_value:.6f} (improvement: {imp:+.6f}, {pct:+.2f}%)")


def get_sampling_function(experiment_name: str):
    """Get the sampling function based on experiment name (copied from main.py)."""
    if experiment_name == "spot_detection":
        return lambda x: x['overall_metrics']['f1_score']
    elif experiment_name == "cellpose_segmentation":
        return lambda x: x['overall_metrics']['average_precision']
    elif experiment_name == "medSAM_segmentation":
        return lambda x: x['overall_metrics']['dsc_metric'] + x['overall_metrics']['nsd_metric']
    else:
        raise ValueError(f"Unknown experiment name: {experiment_name}")


def main():
    if len(sys.argv) < 3:
        print("Usage: compute_automl_improvement.py <run_directory> <experiment_name>")
        print()
        print("Arguments:")
        print("  run_directory:    Path to the run directory")
        print("  experiment_name:  One of: spot_detection, cellpose_segmentation, medSAM_segmentation")
        print()
        print("Example:")
        print("  compute_automl_improvement.py automl_runs_diff_seed/spot_detection/latest spot_detection")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    experiment_name = sys.argv[2]

    # Validate experiment name
    valid_experiments = ["spot_detection", "cellpose_segmentation", "medSAM_segmentation"]
    if experiment_name not in valid_experiments:
        print(f"Error: experiment_name must be one of {valid_experiments}")
        sys.exit(1)

    if not run_dir.exists():
        print(f"Error: Directory {run_dir} does not exist")
        sys.exit(1)

    try:
        # Get the sampling function for this experiment
        sampling_function = get_sampling_function(experiment_name)

        # Load function bank and compute improvements
        func_bank = load_function_bank(run_dir)
        optimized_count, improvements = compute_improvements(func_bank, sampling_function)

        # Print statistics
        print_statistics(len(func_bank), optimized_count, improvements, experiment_name)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
