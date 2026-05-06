"""
Shared utilities for the Star Meadow analysis project.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_FILE = DATA_DIR / "PycnoCountCLean_12_31_2025.csv"
LENGTH_FILE = DATA_DIR / "PycnoLengthCLean_12_31_2025.csv"


def get_output_dir(script_path: str) -> Path:
    """
    Returns outputs/<script_name>/, creating it if needed.
    
    Usage in any script:
        OUTPUT_DIR = get_output_dir(__file__)
    """
    script_name = Path(script_path).stem
    output_dir = OUTPUTS_DIR / script_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_data() -> pd.DataFrame:
    """
    Load and preprocess the Pycnopodia survey data.
    
    Returns:
        DataFrame with cleaned data and calculated encounter rate.
    """
    df = pd.read_csv(DATA_FILE)
    
    # Fix known data entry errors
    # "2/20/0204" should be "2/20/2024"
    df["Date"] = df["Date"].replace({"2/20/0204": "2/20/2024"})
    
    # Calculate encounter rate (per hour)
    df["Encounter.Rate.Hr"] = (df["Pycnopodia_count"] / df["Survey.Time"]) * 60
    
    # Parse date with error handling
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    
    # Extract year/month from parsed dates
    df["Year"] = df["Date"].dt.year
    df["Month_parsed"] = df["Date"].dt.month
    
    # Use original Month column if valid, otherwise use parsed month
    # Handle "#VALUE!" and other Excel errors in Month column
    # Also handle case where Month column doesn't exist (2025 data format)
    if "Month" in df.columns:
        df["Month_orig"] = pd.to_numeric(df["Month"], errors="coerce")
        df["Month"] = df["Month_parsed"].fillna(df["Month_orig"])
        df = df.drop(columns=["Month_parsed", "Month_orig"])
    else:
        df["Month"] = df["Month_parsed"]
        df = df.drop(columns=["Month_parsed"])
    
    # Map to seasons
    df["Season"] = df["Month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn"
    })
    
    # Report any rows with parsing issues
    bad_dates = df["Date"].isna().sum()
    if bad_dates > 0:
        print(f"  Warning: {bad_dates} rows with unparseable dates (set to NaT)")
    
    return df


def load_length_data() -> pd.DataFrame:
    """
    Load and preprocess the Pycnopodia length/size data.
    
    This dataset has one row per individual Pycnopodia observed,
    with Length(cm) measurements. Rows with "0" length represent
    survey transects with no observations.
    
    Returns:
        DataFrame with individual-level size measurements.
    """
    df = pd.read_csv(LENGTH_FILE)
    
    # Clean Length column - convert to numeric
    df["Length_cm"] = pd.to_numeric(df["Length(cm)"].astype(str).str.strip(), errors="coerce")
    
    # Fix date typos (same as count data)
    df["Date"] = df["Date"].replace({"2/20/0204": "2/20/2024"})
    
    # Parse date
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Season"] = df["Month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn"
    })
    
    # Add site-level eelgrass indicator
    sites_with_eelgrass = df.groupby("SiteName")["HabitatType"].apply(
        lambda x: (x == "Eelgrass").any()
    )
    df["Site_Has_Eelgrass"] = df["SiteName"].map(sites_with_eelgrass)
    
    return df


def load_length_data_individuals() -> pd.DataFrame:
    """
    Load only the rows with actual length measurements (non-zero).
    
    Returns:
        DataFrame with individual Pycnopodia measurements only.
    """
    df = load_length_data()
    
    # Filter to individuals with measurements
    df_individuals = df[df["Length_cm"] > 0].copy()
    
    print(f"  Loaded {len(df_individuals)} individual Pycnopodia measurements")
    print(f"  Size range: {df_individuals['Length_cm'].min():.0f} - {df_individuals['Length_cm'].max():.0f} cm")
    print(f"  Mean size: {df_individuals['Length_cm'].mean():.1f} cm")
    
    return df_individuals


def set_style():
    """Set consistent matplotlib/seaborn style for all figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
    })


def save_figure(fig, output_dir: Path, filename: str, formats: list = None):
    """
    Save figure in multiple formats.
    
    Args:
        fig: matplotlib figure object
        output_dir: directory to save to
        filename: base filename (without extension)
        formats: list of formats (default: ["png", "pdf"])
    """
    if formats is None:
        formats = ["png", "pdf"]
    
    for fmt in formats:
        filepath = output_dir / f"{filename}.{fmt}"
        fig.savefig(filepath, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {filepath}")

