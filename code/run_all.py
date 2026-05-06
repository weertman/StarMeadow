"""
Run All Analyses

Master script to execute all analysis scripts in sequence.
Outputs are saved to the outputs/ directory in subfolders
named after each script.

Usage:
    python run_all.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "01_encounter_rate_analysis.py",
    "02_size_structure_analysis.py",
    "03_clustering_analysis.py",
    "04_temporal_analysis.py",
    "05_statistical_summary.py",
    "06_eelgrass_site_analysis.py",
    "07_size_clustering_analysis.py",
    "08_combined_clustering_analysis.py",
    "09_size_prediction_analysis.py",
    "10_eelgrass_size_relationship.py",
    "11_habitat_diversity_analysis.py",
    "12_eelgrass_basin_analysis.py",
    "13_static_maps.py",
    "14_interactive_maps.py",
    "15_shorezone_site_analysis.py",
    "16_shorezone_recovery_analysis.py",
    "15b_site_coverage_map.py",
    "19_probability_density_map.py",
    "20_publication_figures.py",
]

def main():
    code_dir = Path(__file__).parent
    
    print("=" * 70)
    print("STAR MEADOW: Pycnopodia helianthoides Analysis Pipeline")
    print("=" * 70)
    print()
    
    failures = []
    skipped = []

    for script in SCRIPTS:
        script_path = code_dir / script
        
        if not script_path.exists():
            print(f"⚠️  Skipping {script} (not found)")
            skipped.append(script)
            continue
        
        print(f"\n{'─' * 70}")
        print(f"▶ Running: {script}")
        print(f"{'─' * 70}")
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(code_dir)
        )
        
        if result.returncode != 0:
            print(f"❌ {script} failed with exit code {result.returncode}")
            failures.append((script, result.returncode))
        else:
            print(f"✅ {script} completed successfully")
    
    print()
    print("=" * 70)
    print("ALL ANALYSES COMPLETE")
    print("=" * 70)
    print()
    print("Outputs saved to: outputs/")
    print("  - 01_encounter_rate_analysis/")
    print("  - 02_size_structure_analysis/")
    print("  - 03_clustering_analysis/")
    print("  - 04_temporal_analysis/")
    print("  - 05_statistical_summary/")
    print("  - 06_eelgrass_site_analysis/")
    print("  - 07_size_clustering_analysis/")
    print("  - 08_combined_clustering_analysis/")
    print("  - 09_size_prediction_analysis/")
    print("  - 10_eelgrass_size_relationship/")
    print("  - 11_habitat_diversity_analysis/")
    print("  - 12_eelgrass_basin_analysis/")
    print("  - 13_static_maps/")
    print("  - 14_interactive_maps/")
    print("  - 15_shorezone_site_analysis/")
    print("  - 15b_site_coverage_map/")
    print("  - 16_shorezone_recovery_analysis/")
    print("  - 19_probability_density_map/")
    print("  - 20_publication_figures/")

    if skipped:
        print()
        print("Skipped scripts:")
        for script in skipped:
            print(f"  - {script}")

    if failures:
        print()
        print("FAILED SCRIPTS:")
        for script, returncode in failures:
            print(f"  - {script}: exit code {returncode}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

