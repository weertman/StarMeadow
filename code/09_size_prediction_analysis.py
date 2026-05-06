"""
Size Distribution Prediction from Habitat Composition

Predicts Pycnopodia size distribution characteristics from site-level
habitat composition using regression models.

Predictors (X): Habitat composition proportions per site
- Proportion of transects in each habitat type (Eelgrass, Soft Bottom, etc.)
- Site-level eelgrass presence indicator

Response variables (Y):
- Mean size
- Median size  
- Size distribution moments (std, skewness)
- Size bin proportions

Models:
- Linear Regression
- Random Forest Regression
- Multi-output regression for size bin proportions

Outputs:
- Model performance metrics
- Feature importance plots
- Predicted vs actual scatter plots
- Residual analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings

from utils import (
    get_output_dir, load_data, load_length_data_individuals,
    set_style, save_figure
)

OUTPUT_DIR = get_output_dir(__file__)

# Size bins
SIZE_BIN_WIDTH = 5
MAX_SIZE = 90
SIZE_BINS = list(range(0, MAX_SIZE + SIZE_BIN_WIDTH + 1, SIZE_BIN_WIDTH))
SIZE_LABELS = [f"{SIZE_BINS[i]}-{SIZE_BINS[i+1]}" for i in range(len(SIZE_BINS)-1)]


def prepare_site_data(df_count: pd.DataFrame, df_length: pd.DataFrame, min_individuals: int = 5):
    """
    Prepare site-level dataset with habitat features and size metrics.
    
    Returns:
        X: Habitat composition features (DataFrame)
        y_metrics: Size distribution metrics (DataFrame)
        site_info: Additional site information
    """
    # Habitat composition per site
    habitat_ratios = (
        df_count.groupby(["SiteName", "HabitatType"])
        .size()
        .reset_index(name="Count")
    )
    habitat_ratios["Ratio"] = habitat_ratios.groupby("SiteName")["Count"].transform(
        lambda x: x / x.sum()
    )
    
    habitat_matrix = habitat_ratios.pivot_table(
        index="SiteName",
        columns="HabitatType",
        values="Ratio",
        fill_value=0
    )
    
    # Site eelgrass indicator
    site_eelgrass = df_count.groupby("SiteName")["HabitatType"].apply(
        lambda x: (x == "Eelgrass").any()
    ).astype(int)
    
    # Size metrics per site
    site_size = df_length.groupby("SiteName")["Length_cm"].agg([
        ("mean_size", "mean"),
        ("median_size", "median"),
        ("std_size", "std"),
        ("min_size", "min"),
        ("max_size", "max"),
        ("n_individuals", "count")
    ])
    
    # Add skewness
    site_skew = df_length.groupby("SiteName")["Length_cm"].apply(
        lambda x: stats.skew(x) if len(x) >= 3 else np.nan
    )
    site_size["skewness"] = site_skew
    
    # Size bin proportions
    df_length = df_length.copy()
    df_length["Size_Bin"] = pd.cut(
        df_length["Length_cm"],
        bins=SIZE_BINS,
        labels=SIZE_LABELS,
        include_lowest=True,
        right=False
    )
    
    size_bin_props = df_length.groupby(["SiteName", "Size_Bin"], observed=False).size().unstack(fill_value=0)
    size_bin_props = size_bin_props.div(size_bin_props.sum(axis=1), axis=0)
    size_bin_props.columns = [f"Prop_{c}" for c in size_bin_props.columns]
    
    # Combine
    common_sites = (
        habitat_matrix.index
        .intersection(site_size.index)
        .intersection(site_eelgrass.index)
    )
    
    # Filter by minimum individuals
    valid_sites = site_size[site_size["n_individuals"] >= min_individuals].index
    common_sites = common_sites.intersection(valid_sites)
    
    # Prepare X (features)
    X = habitat_matrix.loc[common_sites].copy()
    X["Site_Has_Eelgrass"] = site_eelgrass.loc[common_sites]
    
    # Prepare y (targets)
    y_metrics = site_size.loc[common_sites].copy()
    y_bins = size_bin_props.loc[common_sites].copy()
    
    return X, y_metrics, y_bins, common_sites


def evaluate_single_target(X, y, target_name, models=None):
    """
    Evaluate models predicting a single target variable.
    Uses Leave-One-Out cross-validation for small sample sizes.
    """
    if models is None:
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
        }
    
    results = []
    predictions = {}
    
    # Drop NaN targets
    valid_idx = ~y.isna()
    X_valid = X[valid_idx]
    y_valid = y[valid_idx]
    
    if len(y_valid) < 5:
        print(f"    Skipping {target_name}: not enough valid samples ({len(y_valid)})")
        return None, None
    
    loo = LeaveOneOut()
    
    for name, model in models.items():
        try:
            # LOO cross-validation predictions
            y_pred = cross_val_predict(model, X_valid, y_valid, cv=loo)
            
            # Metrics
            r2 = r2_score(y_valid, y_pred)
            rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
            mae = mean_absolute_error(y_valid, y_pred)
            
            # Correlation
            corr, p_val = stats.pearsonr(y_valid, y_pred)
            
            results.append({
                "Target": target_name,
                "Model": name,
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae,
                "Correlation": corr,
                "p-value": p_val
            })
            
            predictions[name] = y_pred
            
        except Exception as e:
            print(f"    Error with {name}: {e}")
    
    return pd.DataFrame(results), predictions


def plot_feature_importance(X, y, target_name):
    """Plot feature importance from Random Forest."""
    # Drop NaN
    valid_idx = ~y.isna()
    X_valid = X[valid_idx]
    y_valid = y[valid_idx]
    
    if len(y_valid) < 5:
        return
    
    # Fit Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    rf.fit(X_valid, y_valid)
    
    # Feature importance
    importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    importance.plot(kind="barh", ax=ax, color="steelblue", edgecolor="black")
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Feature Importance for Predicting {target_name}")
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, f"feature_importance_{target_name.replace(' ', '_').lower()}")
    plt.close()


def plot_predictions(y_true, y_pred, target_name, model_name):
    """Scatter plot of predicted vs actual values."""
    fig, ax = plt.subplots(figsize=(7, 6))
    
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolor="black", s=60)
    
    # Perfect prediction line
    lims = [
        min(y_true.min(), y_pred.min()) - 1,
        max(y_true.max(), y_pred.max()) + 1
    ]
    ax.plot(lims, lims, "r--", alpha=0.8, label="Perfect prediction")
    
    # Regression line
    slope, intercept, r, p, se = stats.linregress(y_true, y_pred)
    x_line = np.linspace(lims[0], lims[1], 100)
    ax.plot(x_line, slope * x_line + intercept, "b-", alpha=0.6, 
            label=f"Fit: r={r:.3f}, p={p:.3f}")
    
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"Actual {target_name}")
    ax.set_ylabel(f"Predicted {target_name}")
    ax.set_title(f"{model_name}: Predicted vs Actual {target_name}")
    ax.legend()
    ax.set_aspect("equal")
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, f"pred_vs_actual_{target_name.replace(' ', '_').lower()}_{model_name.replace(' ', '_').lower()}")
    plt.close()


def plot_habitat_size_relationships(X, y_metrics):
    """Visualize relationships between habitat features and size metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Key habitat features
    habitat_features = [c for c in X.columns if c != "Site_Has_Eelgrass"][:5]
    
    for i, habitat in enumerate(habitat_features):
        ax = axes.flatten()[i]
        
        # Scatter with regression
        valid = ~y_metrics["mean_size"].isna()
        x = X.loc[valid, habitat]
        y = y_metrics.loc[valid, "mean_size"]
        
        ax.scatter(x, y, alpha=0.7, edgecolor="black", s=60)
        
        # Fit line if enough variation
        if x.std() > 0.01:
            slope, intercept, r, p, se = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, "r-", alpha=0.7)
            ax.set_title(f"{habitat}\nr={r:.3f}, p={p:.3f}")
        else:
            ax.set_title(f"{habitat}")
        
        ax.set_xlabel(f"Proportion {habitat}")
        ax.set_ylabel("Mean Size (cm)")
    
    # Last panel: eelgrass vs non-eelgrass
    ax = axes.flatten()[5]
    valid = ~y_metrics["mean_size"].isna()
    eelgrass_sites = X.loc[valid, "Site_Has_Eelgrass"] == 1
    
    data_plot = pd.DataFrame({
        "Mean Size": y_metrics.loc[valid, "mean_size"],
        "Site Type": eelgrass_sites.map({True: "Eelgrass\nSite", False: "Non-Eelgrass\nSite"})
    })
    
    sns.boxplot(data=data_plot, x="Site Type", y="Mean Size", ax=ax,
                hue="Site Type", palette={"Eelgrass\nSite": "#2ecc71", "Non-Eelgrass\nSite": "#e74c3c"})
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_xlabel("")
    ax.set_ylabel("Mean Size (cm)")
    ax.set_title("Mean Size by Site Eelgrass Presence")
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "habitat_size_relationships")
    plt.close()


def plot_multioutput_predictions(X, y_bins, min_prop=0.05):
    """
    Predict size bin proportions from habitat composition.
    """
    # Filter to bins with enough data
    mean_props = y_bins.mean()
    valid_bins = mean_props[mean_props >= min_prop].index.tolist()
    
    if len(valid_bins) < 2:
        print("  Not enough size bins with data for multi-output prediction")
        return
    
    y_subset = y_bins[valid_bins]
    
    # Multi-output Random Forest
    from sklearn.multioutput import MultiOutputRegressor
    
    rf = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    )
    
    # LOO cross-validation
    loo = LeaveOneOut()
    y_pred = cross_val_predict(rf, X, y_subset, cv=loo)
    y_pred_df = pd.DataFrame(y_pred, index=y_subset.index, columns=y_subset.columns)
    
    # Plot predicted vs actual for each bin
    n_bins = len(valid_bins)
    n_cols = min(4, n_bins)
    n_rows = (n_bins + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    for i, bin_name in enumerate(valid_bins):
        ax = axes[i]
        
        y_true = y_subset[bin_name]
        y_p = y_pred_df[bin_name]
        
        ax.scatter(y_true, y_p, alpha=0.7, edgecolor="black", s=40)
        
        lims = [0, max(y_true.max(), y_p.max()) + 0.05]
        ax.plot(lims, lims, "r--", alpha=0.6)
        
        r, p = stats.pearsonr(y_true, y_p)
        ax.set_title(f"{bin_name.replace('Prop_', '')} cm\nr={r:.2f}, p={p:.3f}", fontsize=9)
        ax.set_xlabel("Actual Proportion", fontsize=8)
        ax.set_ylabel("Predicted", fontsize=8)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    
    # Hide empty axes
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle("Predicting Size Bin Proportions from Habitat Composition\n(Leave-One-Out CV)", fontsize=11)
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "multioutput_size_bin_predictions")
    plt.close()


def generate_summary_report(all_results: pd.DataFrame):
    """Generate summary report of all model results."""
    if all_results is None or len(all_results) == 0:
        return
    
    # Best model per target
    best_models = all_results.loc[all_results.groupby("Target")["R2"].idxmax()]
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    
    print("\nBest model for each target (by R²):")
    print(best_models[["Target", "Model", "R2", "RMSE", "Correlation", "p-value"]].to_string(index=False))
    
    # Save full results
    results_path = OUTPUT_DIR / "model_results.csv"
    all_results.to_csv(results_path, index=False)
    print(f"\nSaved full results: {results_path}")
    
    return best_models


def main():
    print(f"\n{'='*60}")
    print("SIZE PREDICTION FROM HABITAT COMPOSITION")
    print(f"{'='*60}\n")
    
    set_style()
    warnings.filterwarnings("ignore")
    
    # Load data
    print("Loading data...")
    df_count = load_data()
    df_length = load_length_data_individuals()
    
    # Prepare site-level data
    print("\nPreparing site-level data...")
    X, y_metrics, y_bins, sites = prepare_site_data(df_count, df_length, min_individuals=5)
    
    print(f"  Sites with sufficient data: {len(sites)}")
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Features: {X.columns.tolist()}")
    
    print(f"\nOutput directory: {OUTPUT_DIR}\n")
    
    # Evaluate single-target predictions
    print("="*50)
    print("SINGLE-TARGET PREDICTION")
    print("="*50)
    
    all_results = []
    
    targets = {
        "Mean Size": y_metrics["mean_size"],
        "Median Size": y_metrics["median_size"],
        "Size Std": y_metrics["std_size"],
    }
    
    for target_name, y in targets.items():
        print(f"\n--- Predicting {target_name} ---")
        
        results, predictions = evaluate_single_target(X, y, target_name)
        
        if results is not None:
            all_results.append(results)
            print(results[["Model", "R2", "RMSE", "Correlation", "p-value"]].to_string(index=False))
            
            # Plot feature importance
            plot_feature_importance(X, y, target_name)
            
            # Plot best model predictions
            best_model = results.loc[results["R2"].idxmax(), "Model"]
            if best_model in predictions:
                valid_idx = ~y.isna()
                plot_predictions(y[valid_idx], predictions[best_model], target_name, best_model)
    
    # Combine results
    if all_results:
        all_results_df = pd.concat(all_results, ignore_index=True)
        generate_summary_report(all_results_df)
    
    # Habitat-size relationship plots
    print("\n" + "="*50)
    print("VISUALIZATIONS")
    print("="*50)
    
    print("\nGenerating habitat-size relationship plots...")
    plot_habitat_size_relationships(X, y_metrics)
    
    # Multi-output prediction
    print("\nGenerating multi-output size bin predictions...")
    plot_multioutput_predictions(X, y_bins)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()






