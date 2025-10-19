#!/usr/bin/env python3
"""
Compute AutoML improvement statistics from a function bank.

This script analyzes a function bank JSON file to determine:
- How many entries were AutoML optimized
- The average improvement in metrics for optimized entries
"""

import json
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def load_function_bank(run_dir: Path) -> List[Dict]:
    """Load the function bank JSON file from a run directory."""
    func_bank_path = run_dir / "preprocessing_func_bank.json"

    if not func_bank_path.exists():
        raise FileNotFoundError(f"Function bank not found at {func_bank_path}")

    with open(func_bank_path, 'r') as f:
        return json.load(f)


def compute_improvements(func_bank: List[Dict], sampling_function) -> Tuple[int, List[Tuple[float, float, int, int, bool]]]:
    """
    Compute improvements for AutoML optimized entries using the sampling function.

    Args:
        func_bank: List of function bank entries
        sampling_function: Function to extract the primary metric from an entry

    Returns:
        - Number of AutoML optimization attempts
        - List of (improvement, original_value, source_idx, optimized_idx, is_successful) tuples
    """
    optimized_count = 0
    improvements = []

    for idx, entry in enumerate(func_bank):
        # Skip entries that were never optimized (no automl_optimized key)
        if 'automl_optimized' not in entry:
            continue

        optimized_count += 1
        is_successful = entry['automl_optimized']

        # Get source index
        source_idx = entry.get('automl_source_index')
        if source_idx is None:
            print(f"Warning: AutoML entry at index {idx} missing automl_source_index")
            continue

        # Validate source index
        if source_idx < 0 or source_idx >= len(func_bank):
            print(f"Warning: Invalid source index {source_idx} for entry {idx}")
            continue

        source_entry = func_bank[source_idx]

        # Verify flags are correctly set
        if is_successful and not source_entry.get('automl_superseded', False):
            print(f"Warning: Source entry at index {source_idx} not marked as superseded for successful optimization")
        elif not is_successful and source_entry.get('automl_superseded', False):
            print(f"Warning: Source entry at index {source_idx} incorrectly marked as superseded for failed optimization")

        # Use sampling function to extract primary metric values
        try:
            optimized_value = sampling_function(entry)
            original_value = sampling_function(source_entry)

            if optimized_value is None or original_value is None:
                print(f"Warning: Could not extract metric values for entry {idx}")
                continue

            improvement = optimized_value - original_value
            improvements.append((improvement, original_value, source_idx, idx, is_successful))

        except (KeyError, TypeError, AttributeError) as e:
            print(f"Warning: Error computing improvement for entry {idx}: {e}")
            continue

    return optimized_count, improvements


def print_statistics(total_entries: int, optimized_count: int,
                    improvements: List[Tuple[float, float, int, int, bool]],
                    experiment_name: str):
    """Print improvement statistics."""
    if not improvements:
        print(f"Total entries in function bank: {total_entries}")
        print("No AutoML optimization attempts found.")
        return

    # Count successful vs failed
    successful_count = sum(1 for _, _, _, _, is_successful in improvements if is_successful)
    failed_count = sum(1 for _, _, _, _, is_successful in improvements if not is_successful)

    print(f"Total entries in function bank: {total_entries}")
    print(f"AutoML optimization summary:")
    print(f"  Successful optimizations (automl_optimized=True):  {successful_count}")
    print(f"  Failed optimizations (automl_optimized=False):     {failed_count}")
    print(f"  Total optimization attempts:                       {optimized_count}")
    print()

    # Unified statistics (all optimization attempts)
    print(f"Primary Metric Improvement Statistics ({experiment_name}):")
    print("=" * 60)

    # Extract improvements and calculate percentages
    improvement_values = [imp for imp, _, _, _, _ in improvements]
    pct_improvements = [(imp / orig * 100) if orig != 0 else 0
                       for imp, orig, _, _, _ in improvements]

    # Calculate effective improvements (only count positive improvements)
    effective_improvements = [max(0.0, imp) for imp in improvement_values]
    effective_pct_improvements = [(max(0.0, imp) / orig * 100) if orig != 0 else 0
                                  for imp, orig, _, _, _ in improvements]

    avg_improvement = np.mean(improvement_values)
    avg_pct_improvement = np.mean(pct_improvements)
    avg_effective_improvement = np.mean(effective_improvements)
    avg_effective_pct = np.mean(effective_pct_improvements)

    std_improvement = np.std(improvement_values)
    std_pct_improvement = np.std(pct_improvements)
    std_effective_improvement = np.std(effective_improvements)
    std_effective_pct = np.std(effective_pct_improvements)

    min_improvement = min(improvement_values)
    max_improvement = max(improvement_values)
    min_pct_improvement = min(pct_improvements)
    max_pct_improvement = max(pct_improvements)

    # Count positive vs negative improvements
    positive_count = sum(1 for imp in improvement_values if imp > 0)
    negative_count = sum(1 for imp in improvement_values if imp < 0)
    zero_count = sum(1 for imp in improvement_values if imp == 0)

    print(f"\n  Average improvement:           {avg_improvement:.6f} ± {std_improvement:.6f} ({avg_pct_improvement:+.2f}% ± {std_pct_improvement:.2f}%)")
    print(f"  Average effective improvement: {avg_effective_improvement:.6f} ± {std_effective_improvement:.6f} ({avg_effective_pct:+.2f}% ± {std_effective_pct:.2f}%)")
    print(f"  Min improvement:               {min_improvement:.6f} ({min_pct_improvement:+.2f}%)")
    print(f"  Max improvement:               {max_improvement:.6f} ({max_pct_improvement:+.2f}%)")
    print(f"  Total improvements:            {len(improvements)} entries")
    print(f"    Positive: {positive_count}, Negative: {negative_count}, Zero: {zero_count}")

    # Show individual improvements
    print(f"\n  Individual optimizations:")
    for i, (imp, orig, source_idx, new_idx, is_successful) in enumerate(improvements, 1):
        pct = (imp / orig * 100) if orig != 0 else 0
        optimized_value = orig + imp
        status = "✓" if is_successful else "✗"
        print(f"    {i:2d}. {status} [idx: {source_idx}->{new_idx}]: {orig:.6f} -> {optimized_value:.6f} (change: {imp:+.6f}, {pct:+.2f}%)")


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


def find_run_directories(base_dir: Path) -> List[Path]:
    """Find all directories containing preprocessing_func_bank.json"""
    run_dirs = []
    for item in base_dir.rglob("preprocessing_func_bank.json"):
        run_dirs.append(item.parent)
    return sorted(run_dirs)


def process_single_directory(run_dir: Path, experiment_name: str, print_stats: bool = True):
    """Process a single run directory and optionally print statistics

    Returns:
        Tuple of (total_entries, optimized_count, improvements) or None if error
    """
    try:
        # Get the sampling function for this experiment
        sampling_function = get_sampling_function(experiment_name)

        # Load function bank and compute improvements
        func_bank = load_function_bank(run_dir)
        optimized_count, improvements = compute_improvements(func_bank, sampling_function)

        # Print statistics if requested
        if print_stats:
            print_statistics(len(func_bank), optimized_count, improvements, experiment_name)

        return len(func_bank), optimized_count, improvements
    except Exception as e:
        print(f"Error processing {run_dir}: {e}")
        return None


def print_aggregate_statistics(all_results: List[Tuple[Path, int, int, List[Tuple[float, float, int, int, bool]]]],
                               experiment_name: str):
    """Print aggregate statistics across multiple run directories"""
    if not all_results:
        print("No results to aggregate.")
        return

    # Collect all improvements across all runs
    all_improvements = []
    total_runs = len(all_results)
    total_entries_sum = 0
    total_optimized_sum = 0

    for run_dir, total_entries, optimized_count, improvements in all_results:
        total_entries_sum += total_entries
        total_optimized_sum += optimized_count
        all_improvements.extend(improvements)

    if not all_improvements:
        print(f"Processed {total_runs} run directories")
        print("No AutoML optimization attempts found across any runs.")
        return

    # Count successful vs failed
    successful_count = sum(1 for _, _, _, _, is_successful in all_improvements if is_successful)
    failed_count = sum(1 for _, _, _, _, is_successful in all_improvements if not is_successful)

    print(f"\n{'=' * 80}")
    print(f"AGGREGATE STATISTICS ACROSS {total_runs} RUN DIRECTORIES")
    print(f"{'=' * 80}\n")

    print(f"Total entries across all runs:  {total_entries_sum}")
    print(f"AutoML optimization summary:")
    print(f"  Successful optimizations (automl_optimized=True):  {successful_count}")
    print(f"  Failed optimizations (automl_optimized=False):     {failed_count}")
    print(f"  Total optimization attempts:                       {total_optimized_sum}")
    print()

    # Unified statistics (all optimization attempts)
    print(f"Primary Metric Improvement Statistics ({experiment_name}):")
    print("=" * 60)

    # Extract improvements and calculate percentages
    improvement_values = [imp for imp, _, _, _, _ in all_improvements]
    pct_improvements = [(imp / orig * 100) if orig != 0 else 0
                       for imp, orig, _, _, _ in all_improvements]

    # Calculate effective improvements (only count positive improvements)
    effective_improvements = [max(0.0, imp) for imp in improvement_values]
    effective_pct_improvements = [(max(0.0, imp) / orig * 100) if orig != 0 else 0
                                  for imp, orig, _, _, _ in all_improvements]

    avg_improvement = np.mean(improvement_values)
    avg_pct_improvement = np.mean(pct_improvements)
    avg_effective_improvement = np.mean(effective_improvements)
    avg_effective_pct = np.mean(effective_pct_improvements)

    std_improvement = np.std(improvement_values)
    std_pct_improvement = np.std(pct_improvements)
    std_effective_improvement = np.std(effective_improvements)
    std_effective_pct = np.std(effective_pct_improvements)

    min_improvement = min(improvement_values)
    max_improvement = max(improvement_values)
    min_pct_improvement = min(pct_improvements)
    max_pct_improvement = max(pct_improvements)

    # Count positive vs negative improvements
    positive_count = sum(1 for imp in improvement_values if imp > 0)
    negative_count = sum(1 for imp in improvement_values if imp < 0)
    zero_count = sum(1 for imp in improvement_values if imp == 0)

    print(f"\n  Average improvement:           {avg_improvement:.6f} ± {std_improvement:.6f} ({avg_pct_improvement:+.2f}% ± {std_pct_improvement:.2f}%)")
    print(f"  Average effective improvement: {avg_effective_improvement:.6f} ± {std_effective_improvement:.6f} ({avg_effective_pct:+.2f}% ± {std_effective_pct:.2f}%)")
    print(f"  Min improvement:               {min_improvement:.6f} ({min_pct_improvement:+.2f}%)")
    print(f"  Max improvement:               {max_improvement:.6f} ({max_pct_improvement:+.2f}%)")
    print(f"  Total improvements:            {len(all_improvements)} entries")
    print(f"    Positive: {positive_count}, Negative: {negative_count}, Zero: {zero_count}")

    # Show per-directory breakdown
    print(f"\n  Per-directory breakdown:")
    for i, (run_dir, total_entries, optimized_count, improvements) in enumerate(all_results, 1):
        run_successful = sum(1 for _, _, _, _, is_successful in improvements if is_successful)
        run_failed = len(improvements) - run_successful
        print(f"    {i:2d}. {run_dir}")
        print(f"        Entries: {total_entries}, Optimizations: {optimized_count} (✓ {run_successful}, ✗ {run_failed})")


def main():
    parser = argparse.ArgumentParser(
        description="Compute AutoML improvement statistics from a function bank.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single run directory
  %(prog)s automl_runs/spot_detection/latest spot_detection

  # Analyze all run directories in current directory
  %(prog)s --all . spot_detection

  # Analyze all run directories in a specific directory
  %(prog)s --all automl_runs spot_detection
        """
    )

    parser.add_argument(
        "run_directory",
        type=str,
        help="Path to the run directory (or base directory when using --all)"
    )

    parser.add_argument(
        "experiment_name",
        type=str,
        choices=["spot_detection", "cellpose_segmentation", "medSAM_segmentation"],
        help="Experiment name"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all subdirectories containing preprocessing_func_bank.json"
    )

    args = parser.parse_args()

    run_dir = Path(args.run_directory)
    experiment_name = args.experiment_name

    if not run_dir.exists():
        print(f"Error: Directory {run_dir} does not exist")
        sys.exit(1)

    if args.all:
        # Process all run directories
        run_dirs = find_run_directories(run_dir)

        if not run_dirs:
            print(f"No run directories found in {run_dir}")
            sys.exit(1)

        print(f"Found {len(run_dirs)} run directories to process")
        print("=" * 80)

        # Collect results from all directories
        all_results = []
        for i, run_dir_path in enumerate(run_dirs, 1):
            print(f"Processing directory {i}/{len(run_dirs)}: {run_dir_path}...", end=" ")
            result = process_single_directory(run_dir_path, experiment_name, print_stats=False)
            if result is not None:
                all_results.append((run_dir_path, *result))
                print("✓")
            else:
                print("✗")

        # Print aggregate statistics
        print_aggregate_statistics(all_results, experiment_name)

    else:
        # Process single directory
        process_single_directory(run_dir, experiment_name)


if __name__ == "__main__":
    main()
