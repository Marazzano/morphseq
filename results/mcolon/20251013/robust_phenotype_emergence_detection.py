import pandas as pd
import os
import numpy as np
import seaborn as sns 
import plotly.express as px
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.cm as cm
from pathlib import Path


# Use the parent directory of this file for results
# results_dir = os.getcwd()
results_dir = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251013"
data_dir = os.path.join(results_dir, "data")
plot_dir = os.path.join(results_dir, "plots")

print(f"Results directory: {results_dir}")
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)


morphseq_root = os.environ.get('MORPHSEQ_REPO_ROOT')
morphseq_root = "/net/trapnell/vol1/home/mdcolon/proj/morphseq"
print(f"MORPHSEQ_REPO_ROOT: {morphseq_root}")
os.chdir(morphseq_root)

# Add morphseq root to Python path
import sys
sys.path.insert(0, morphseq_root)

from src.functions.embryo_df_performance_metrics import *
from src.functions.spline_morph_spline_metrics import *

# Import TZ experiments
WT_experiments = ["20230615","20230531", "20230525", "20250912"] 

b9d2_experiments = ["20250519","20250520"]

cep290_experiments = ["20250305", "20250416", "20250512", "20250515_part2", "20250519"]

tmem67_experiments = ["20250711"]

experiments = WT_experiments + b9d2_experiments + cep290_experiments + tmem67_experiments

build06_dir = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output"

# Load all experiments
dfs = []
for exp in experiments:
    try:
        file_path = f"{build06_dir}/df03_final_output_with_latents_{exp}.csv"
        df = pd.read_csv(file_path)
        df['source_experiment'] = exp
        print(df['genotype'].value_counts())
        dfs.append(df)
        print(f"Loaded {exp}: {len(df)} rows")
    except:
        print(f"Missing: {exp}")

# Combine all data
combined_df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal: {len(combined_df)} rows from {len(dfs)} experiments")



import numpy as np
import pandas as pd

def bin_by_embryo_time(
    df,
    time_col="predicted_stage_hpf",
    z_cols=None,
    bin_width=2.0,
    suffix="_binned"
):
    """
    Bin VAE embeddings by predicted time and embryo.

    Always averages embeddings per embryo_id × time_bin,
    keeping all non-latent metadata columns (e.g., genotype).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing 'embryo_id', 'predicted_stage_hpf', and latent columns.
    time_col : str
        Column name to bin by.
    z_cols : list or None
        Columns to average. If None, auto-detect those containing 'z_mu_b'.
    bin_width : float
        Width of time bins (same units as time_col, usually hours).
    suffix : str
        Suffix to append to averaged latent column names.

    Returns
    -------
    pd.DataFrame
        One row per (embryo_id, time_bin) containing averaged latent columns and preserved metadata.
    """

    df = df.copy()

    # detect latent columns
    if z_cols is None:
        z_cols = [c for c in df.columns if "z_mu_b" in c]
        if not z_cols:
            raise ValueError("No latent columns found matching pattern 'z_mu_b'.")

    # create time bins
    df["time_bin"] = (np.floor(df[time_col] / bin_width) * bin_width).astype(int)

    # average latent vectors per embryo × time_bin
    agg = (
        df.groupby(["embryo_id", "time_bin"], as_index=False)[z_cols]
        .mean()
    )

    # rename averaged latent columns
    agg.rename(columns={c: f"{c}{suffix}" for c in z_cols}, inplace=True)

    # merge back non-latent metadata (take first unique per embryo)
    # Exclude time_bin and time_col from meta_cols to avoid conflicts
    meta_cols = [c for c in df.columns if c not in z_cols + [time_col, "time_bin"]]
    meta_df = (
        df[meta_cols]
        .drop_duplicates(subset=["embryo_id"])
    )

    # merge metadata back in
    out = agg.merge(meta_df, on="embryo_id", how="left")

    # ensure sorting
    out = out.sort_values(["embryo_id", "time_bin"]).reset_index(drop=True)

    return out


def get_z_columns(df, z_cols=None, suffix="_binned"):
    
    """
    Identify latent (embedding) columns for analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (already binned by embryo/time).
    z_cols : list or None
        Optional explicit list. If None, automatically detect by suffix or 'z_mu_b' pattern.
    suffix : str
        Column suffix used in binning (default '_binned').

    Returns
    -------
    list
        Names of latent columns.
    """
    if z_cols is None:
        z_cols = [c for c in df.columns if c.endswith(suffix) or "z_mu_b" in c]
    if not z_cols:
        raise ValueError("No latent columns detected for analysis.")
    return z_cols

from itertools import combinations
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

# -- helper stats --

def energy_distance(X, Y):
    XY = cdist(X, Y).mean()
    XX = cdist(X, X).mean()
    YY = cdist(Y, Y).mean()
    return 2*XY - XX - YY

def energy_perm_test(X, Y, n_perm=500, rng=None):
    rng = np.random.default_rng(rng)
    obs = energy_distance(X, Y)
    Z = np.vstack([X, Y])
    nx = len(X)
    perm_stats = []
    for _ in range(n_perm):
        rng.shuffle(Z)
        perm_stats.append(energy_distance(Z[:nx], Z[nx:]))
    p = (np.sum(perm_stats >= obs) + 1) / (n_perm + 1)
    return obs, p

def hotellings_T2(X, Y):
    """
    Compute Hotelling's T-squared statistic with robust covariance estimation.
    Raises ValueError if data contains NaN or infinite values.
    """
    # Check for NaN or infinite values
    if not (np.isfinite(X).all() and np.isfinite(Y).all()):
        raise ValueError("Data contains NaN or infinite values")

    n, m = len(X), len(Y)
    mean_diff = X.mean(0) - Y.mean(0)
    Sx = LedoitWolf().fit(X).covariance_
    Sy = LedoitWolf().fit(Y).covariance_
    Sp = ((n-1)*Sx + (m-1)*Sy) / (n+m-2)
    invSp = np.linalg.pinv(Sp)
    return (n*m)/(n+m) * float(mean_diff @ invSp @ mean_diff)

# -- main analysis --

def run_distribution_tests(
    df_binned,
    group_col="genotype",
    time_col="time_bin",
    z_cols=None,
    tests=("energy", "hotelling"),
    n_perm=500,
    min_n=4,
    random_state=None
):
    """
    Run pairwise group comparisons per time bin.

    Parameters
    ----------
    df_binned : pd.DataFrame
        Output of bin_by_embryo_time().
    group_col : str
        Column specifying experimental group (e.g., genotype).
    time_col : str
        Column specifying time bins.
    z_cols : list or None
        Columns to analyze (auto-detected if None).
    tests : tuple
        Which tests to run.
    n_perm : int
        Number of permutations for nonparametric tests.
    min_n : int
        Minimum per-group sample size per bin.
    random_state : int or None
        RNG seed.

    Returns
    -------
    pd.DataFrame
        One row per (time_bin, group1, group2, test).
    """

    if z_cols is None:
        z_cols = get_z_columns(df_binned)

    rng = np.random.default_rng(random_state)
    results = []

    for time_val, df_t in df_binned.groupby(time_col):
        groups = sorted(df_t[group_col].dropna().unique())
        for g1, g2 in combinations(groups, 2):
            X = df_t.loc[df_t[group_col]==g1, z_cols].values
            Y = df_t.loc[df_t[group_col]==g2, z_cols].values
            if len(X) < min_n or len(Y) < min_n:
                continue

            rec = dict(time_bin=time_val, group1=g1, group2=g2)

            if "energy" in tests:
                stat, p = energy_perm_test(X, Y, n_perm=n_perm, rng=rng)
                rec.update(energy_stat=stat, energy_p=p)

            if "hotelling" in tests:
                try:
                    T2 = hotellings_T2(X, Y)
                    # permutation p-value for robustness
                    Z = np.vstack([X,Y])
                    nx = len(X)
                    perm_stats = []
                    for _ in range(n_perm):
                        rng.shuffle(Z)
                        perm_stats.append(hotellings_T2(Z[:nx], Z[nx:]))
                    perm_stats = np.array(perm_stats)
                    p = (np.sum(perm_stats >= T2)+1)/(n_perm+1)
                    rec.update(hotelling_T2=T2, hotelling_p=p)
                except np.linalg.LinAlgError:
                    rec.update(hotelling_T2=np.nan, hotelling_p=np.nan)

            results.append(rec)

    return pd.DataFrame(results)
    

from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

def summarize_test_results(results_df, test_col="energy_p", alpha=0.05, consecutive=2):
    """
    FDR-correct p-values across time bins and compute earliest
    consecutive-significant onset for each group pair.

    Returns
    -------
    summary : pd.DataFrame
        Columns: group1, group2, onset_bin, n_significant_bins, etc.
    """

    summaries = []
    for (g1, g2), sub in results_df.groupby(["group1", "group2"]):
        sub = sub.sort_values("time_bin")
        # FDR correction
        rej, p_corr, _, _ = multipletests(sub[test_col], alpha=alpha, method="fdr_bh")
        sub = sub.assign(p_corr=p_corr, reject=rej)

        # detect first consecutive-significant stretch
        consec = 0
        onset = None
        for t, sig in zip(sub["time_bin"], sub["reject"]):
            consec = consec + 1 if sig else 0
            if consec >= consecutive:
                onset = t - (consecutive-1)*2  # subtract bins if needed
                break

        summaries.append(dict(
            group1=g1,
            group2=g2,
            onset_bin=onset,
            n_sig_bins=rej.sum(),
            first_sig_bin=sub.loc[sub["reject"], "time_bin"].min() if rej.any() else None
        ))

    return pd.DataFrame(summaries)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_energy_distance_over_time(results_df, output_path=None):
    """
    Plot energy distance statistic over time for each pairwise comparison.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from run_distribution_tests()
    output_path : str or None
        Path to save figure. If None, displays interactively.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for (g1, g2), sub in results_df.groupby(['group1', 'group2']):
        sub = sub.sort_values('time_bin')
        label = f"{g1} vs {g2}"
        ax.plot(sub['time_bin'], sub['energy_stat'], marker='o', label=label, linewidth=2)

    ax.set_xlabel('Time (hpf)', fontsize=12)
    ax.set_ylabel('Energy Distance', fontsize=12)
    ax.set_title('Energy Distance Between Genotypes Over Development', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    return fig


def plot_pvalues_over_time(results_df, test_col='energy_p', alpha=0.05, output_path=None):
    """
    Plot p-values over time with significance threshold.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from run_distribution_tests()
    test_col : str
        Column containing p-values to plot
    alpha : float
        Significance threshold to mark on plot
    output_path : str or None
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for (g1, g2), sub in results_df.groupby(['group1', 'group2']):
        sub = sub.sort_values('time_bin')
        label = f"{g1} vs {g2}"
        ax.plot(sub['time_bin'], sub[test_col], marker='o', label=label, linewidth=2)

    # Add significance threshold line
    ax.axhline(y=alpha, color='red', linestyle='--', linewidth=2, label=f'α = {alpha}')

    ax.set_xlabel('Time (hpf)', fontsize=12)
    ax.set_ylabel('P-value', fontsize=12)
    ax.set_title('Statistical Significance of Genotype Differences Over Time', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    return fig


def plot_phenotype_onset_heatmap(summary_df, output_path=None):
    """
    Create heatmap showing phenotype emergence onset times.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output from summarize_test_results()
    output_path : str or None
        Path to save figure
    """
    # Create pivot table for heatmap
    pivot_data = summary_df.pivot_table(
        index='group1',
        columns='group2',
        values='onset_bin',
        aggfunc='first'
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(pivot_data.values, cmap='RdYlGn_r', aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(pivot_data.columns)))
    ax.set_yticks(np.arange(len(pivot_data.index)))
    ax.set_xticklabels(pivot_data.columns, rotation=45, ha='right')
    ax.set_yticklabels(pivot_data.index)

    # Add text annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            val = pivot_data.values[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{int(val)}',
                             ha="center", va="center", color="black", fontsize=12)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Onset Time (hpf)', rotation=270, labelpad=20, fontsize=12)

    ax.set_title('Phenotype Emergence Onset Times', fontsize=14, fontweight='bold')
    ax.set_xlabel('Group 2', fontsize=12)
    ax.set_ylabel('Group 1', fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    return fig


def plot_significance_timeline(results_df, test_col='energy_p', alpha=0.05, output_path=None):
    """
    Plot timeline showing which bins are significant for each comparison.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from run_distribution_tests()
    test_col : str
        Column containing p-values
    alpha : float
        Significance threshold
    output_path : str or None
        Path to save figure
    """
    from statsmodels.stats.multitest import multipletests

    fig, axes = plt.subplots(len(results_df.groupby(['group1', 'group2'])), 1,
                              figsize=(12, 3 * len(results_df.groupby(['group1', 'group2']))),
                              sharex=True)

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for idx, ((g1, g2), sub) in enumerate(results_df.groupby(['group1', 'group2'])):
        sub = sub.sort_values('time_bin')

        # FDR correction
        rej, p_corr, _, _ = multipletests(sub[test_col], alpha=alpha, method='fdr_bh')

        ax = axes[idx]

        # Plot raw p-values
        ax.plot(sub['time_bin'], sub[test_col], 'o-', color='steelblue', label='Raw p-value', linewidth=2)

        # Plot corrected p-values
        ax.plot(sub['time_bin'], p_corr, 's-', color='orange', label='FDR-corrected p-value', linewidth=2)

        # Mark significant bins
        sig_bins = sub.loc[rej, 'time_bin']
        if len(sig_bins) > 0:
            ax.scatter(sig_bins, [alpha/10]*len(sig_bins), color='red', s=100,
                      marker='v', label='Significant', zorder=5)

        ax.axhline(y=alpha, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_yscale('log')
        ax.set_ylabel('P-value', fontsize=10)
        ax.set_title(f'{g1} vs {g2}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('Time (hpf)', fontsize=12)
    fig.suptitle('Phenotype Emergence Timeline with FDR Correction',
                 fontsize=14, fontweight='bold', y=1.0)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    return fig


def plot_all_results(results_df, summary_df, plot_dir, test_col='energy_p', alpha=0.05):
    """
    Generate all plots and save to directory.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from run_distribution_tests()
    summary_df : pd.DataFrame
        Output from summarize_test_results()
    plot_dir : str
        Directory to save plots
    test_col : str
        P-value column to use
    alpha : float
        Significance threshold
    """
    os.makedirs(plot_dir, exist_ok=True)

    print("\nGenerating plots...")

    # 1. Energy distance over time
    plot_energy_distance_over_time(
        results_df,
        output_path=os.path.join(plot_dir, 'energy_distance_over_time.png')
    )

    # 2. P-values over time
    plot_pvalues_over_time(
        results_df,
        test_col=test_col,
        alpha=alpha,
        output_path=os.path.join(plot_dir, 'pvalues_over_time.png')
    )

    # 3. Onset heatmap (only if we have onset data)
    if not summary_df['onset_bin'].isna().all():
        plot_phenotype_onset_heatmap(
            summary_df,
            output_path=os.path.join(plot_dir, 'phenotype_onset_heatmap.png')
        )
    else:
        print("No onset data to plot heatmap")

    # 4. Significance timeline
    plot_significance_timeline(
        results_df,
        test_col=test_col,
        alpha=alpha,
        output_path=os.path.join(plot_dir, 'significance_timeline.png')
    )

    print(f"\nAll plots saved to: {plot_dir}")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

# Filter to CEP290 genotypes only
cep290_genotypes = ['cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous']
df_cep290 = combined_df[combined_df['genotype'].isin(cep290_genotypes)].copy()
print(f"\nFiltered to CEP290 genotypes: {len(df_cep290)} rows")
print(f"Genotype distribution:\n{df_cep290['genotype'].value_counts()}")

# 1️⃣ bin embeddings per embryo/time
print("\n=== DEBUGGING: Checking for NaNs in input data ===")
print(f"NaNs in df_cep290: {df_cep290.isna().sum().sum()}")
print(f"NaNs in predicted_stage_hpf: {df_cep290['predicted_stage_hpf'].isna().sum()}")

# Check latent columns for NaNs
z_cols_check = [c for c in df_cep290.columns if "z_mu_b" in c]
print(f"Number of latent columns: {len(z_cols_check)}")
for col in z_cols_check[:5]:  # Show first 5
    print(f"  {col}: {df_cep290[col].isna().sum()} NaNs")

df_binned = bin_by_embryo_time(df_cep290, time_col="predicted_stage_hpf")
print(f"\nBinned data: {len(df_binned)} rows")

# Check binned data for NaNs
print("\n=== DEBUGGING: Checking for NaNs in binned data ===")
binned_z_cols = [c for c in df_binned.columns if "_binned" in c]
print(f"Number of binned latent columns: {len(binned_z_cols)}")
nan_counts = df_binned[binned_z_cols].isna().sum()
if nan_counts.sum() > 0:
    print(f"Total NaNs in binned latent columns: {nan_counts.sum()}")
    print(f"Columns with NaNs:")
    for col in nan_counts[nan_counts > 0].index[:10]:  # Show first 10
        print(f"  {col}: {nan_counts[col]} NaNs")
else:
    print("No NaNs found in binned latent columns")

# Check for infinite values
inf_check = np.isinf(df_binned[binned_z_cols].values).sum()
print(f"Infinite values in binned data: {inf_check}")

# Drop rows with NaN in binned latent columns
if nan_counts.sum() > 0:
    print(f"\nDropping {df_binned[binned_z_cols].isna().any(axis=1).sum()} rows with NaNs in latent columns")
    df_binned = df_binned.dropna(subset=binned_z_cols)
    print(f"Remaining rows after dropping NaNs: {len(df_binned)}")

# TEST SYMMETRY FIRST with real data
print("\n=== TESTING ENERGY DISTANCE SYMMETRY WITH REAL DATA ===")
# Pick a time bin with both homo and wt
test_time_bin = df_binned['time_bin'].mode()[0]  # Most common time bin
test_df = df_binned[df_binned['time_bin'] == test_time_bin]

z_test_cols = [c for c in test_df.columns if "_binned" in c]
homo_data = test_df[test_df['genotype'] == 'cep290_homozygous'][z_test_cols].values
wt_data = test_df[test_df['genotype'] == 'cep290_wildtype'][z_test_cols].values

if len(homo_data) > 0 and len(wt_data) > 0:
    print(f"Time bin: {test_time_bin}")
    print(f"Homozygous samples: {len(homo_data)}")
    print(f"Wildtype samples: {len(wt_data)}")

    energy_homo_wt = energy_distance(homo_data, wt_data)
    energy_wt_homo = energy_distance(wt_data, homo_data)

    print(f"\nenergy_distance(homo, wt) = {energy_homo_wt:.10f}")
    print(f"energy_distance(wt, homo) = {energy_wt_homo:.10f}")
    print(f"Difference: {abs(energy_homo_wt - energy_wt_homo):.10e}")
    print(f"Symmetric: {np.isclose(energy_homo_wt, energy_wt_homo)}")
else:
    print("Not enough data to test symmetry")

# 2️⃣ run pairwise distribution tests
results_df = run_distribution_tests(df_binned, group_col="genotype")
print(f"\nTest results: {len(results_df)} comparisons")

# Debug: Check for duplicate comparisons
print("\n=== DEBUGGING: Checking for duplicate comparisons ===")
unique_pairs = results_df[['group1', 'group2']].drop_duplicates()
print(f"Unique pairs: {len(unique_pairs)}")
print(unique_pairs)

# Check if we have both directions of the same comparison
for _, row in unique_pairs.iterrows():
    g1, g2 = row['group1'], row['group2']
    reverse = results_df[(results_df['group1'] == g2) & (results_df['group2'] == g1)]
    if len(reverse) > 0:
        print(f"WARNING: Found both {g1} vs {g2} AND {g2} vs {g1}")

        # Check if energy stats match
        forward = results_df[(results_df['group1'] == g1) & (results_df['group2'] == g2)]
        print(f"  Forward direction: {len(forward)} comparisons")
        print(f"  Reverse direction: {len(reverse)} comparisons")

        # Compare energy stats at same time bins
        merged = forward.merge(reverse, on='time_bin', suffixes=('_fwd', '_rev'))
        if len(merged) > 0:
            energy_diff = (merged['energy_stat_fwd'] - merged['energy_stat_rev']).abs()
            print(f"  Max energy stat difference: {energy_diff.max():.6f}")
            print(f"  Mean energy stat difference: {energy_diff.mean():.6f}")

# 3️⃣ summarize onset
summary_df = summarize_test_results(results_df, test_col="energy_p")

print("\n=== PHENOTYPE EMERGENCE SUMMARY ===")
print(summary_df)

# 4️⃣ generate all plots
plot_all_results(results_df, summary_df, plot_dir, test_col='energy_p', alpha=0.05)

# Save results to CSV
results_output_path = os.path.join(data_dir, 'cep290_distribution_tests.csv')
summary_output_path = os.path.join(data_dir, 'cep290_phenotype_emergence_summary.csv')

results_df.to_csv(results_output_path, index=False)
summary_df.to_csv(summary_output_path, index=False)

print(f"\n=== RESULTS SAVED ===")
print(f"Full test results: {results_output_path}")
print(f"Summary: {summary_output_path}")
print(f"Plots: {plot_dir}")

# Print detailed summary
print("\n=== DETAILED SUMMARY ===")
for _, row in summary_df.iterrows():
    print(f"\n{row['group1']} vs {row['group2']}:")
    if pd.notna(row['onset_bin']):
        print(f"  Phenotype onset: {row['onset_bin']} hpf")
    else:
        print(f"  No consistent phenotype emergence detected")
    print(f"  First significant bin: {row['first_sig_bin']} hpf")
    print(f"  Total significant bins: {row['n_sig_bins']}")

print("\n=== ANALYSIS COMPLETE ===")