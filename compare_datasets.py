import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Optional plotting imports (lazy-loaded if --plots used)
try:
    import matplotlib
    matplotlib.use('Agg')  # headless safe
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # noqa: BLE001
    plt = None
    sns = None

# --------- KPI DEFINITIONS IMPLEMENTED ---------
# 1. Shape differences (rows, columns)
# 2. Label distribution comparison
# 3. Per-feature mean / std shift
# 4. Mean absolute & relative perturbation per feature
# 5. Sample-level perturbation norms (L0 fraction, L1, L2, Linf averages)
# 6. Correlation matrix shift (Frobenius norm + mean abs diff)
# 7. Jensen-Shannon divergence per feature distribution (hist-based)
# 8. Top features by relative mean shift
# 9. Global summary JSON per pair

EPS = 1e-12


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if 'Label' not in df.columns:
        raise ValueError(f"Label column missing in {path}")
    return df


def align_numeric(df_orig: pd.DataFrame, df_adv: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Intersect numeric feature columns excluding label
    num_orig = df_orig.drop(columns=['Label']).select_dtypes(include=[np.number])
    num_adv = df_adv.drop(columns=['Label']).select_dtypes(include=[np.number])
    common = list(sorted(set(num_orig.columns) & set(num_adv.columns)))
    if not common:
        raise ValueError("No overlapping numeric feature columns to compare.")
    return num_orig[common].reset_index(drop=True), num_adv[common].reset_index(drop=True)


def js_divergence_feature(x: np.ndarray, y: np.ndarray, bins: int = 30) -> float:
    # Build shared bins
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    if lo == hi:  # constant feature
        return 0.0
    hist_x, edges = np.histogram(x, bins=bins, range=(lo, hi), density=True)
    hist_y, _ = np.histogram(y, bins=edges, density=True)
    p = hist_x + EPS
    q = hist_y + EPS
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def correlation_shift(orig: pd.DataFrame, adv: pd.DataFrame) -> Dict[str, float]:
    corr_o = orig.corr().fillna(0.0).to_numpy()
    corr_a = adv.corr().fillna(0.0).to_numpy()
    diff = corr_o - corr_a
    frob = float(np.linalg.norm(diff))
    mean_abs = float(np.mean(np.abs(diff)))
    return {"corr_frobenius_diff": frob, "corr_mean_abs_diff": mean_abs}


def compute_kpis(df_orig: pd.DataFrame, df_adv: pd.DataFrame) -> Dict:
    labels_orig = df_orig['Label']
    labels_adv = df_adv['Label']

    Xo, Xa = align_numeric(df_orig, df_adv)

    if len(Xo) != len(Xa):
        # Align by truncation to min length
        n = min(len(Xo), len(Xa))
        Xo = Xo.iloc[:n]
        Xa = Xa.iloc[:n]
        labels_o_used = labels_orig.iloc[:n]
        labels_a_used = labels_adv.iloc[:n]
    else:
        labels_o_used = labels_orig
        labels_a_used = labels_adv

    delta = Xa.to_numpy() - Xo.to_numpy()
    abs_delta = np.abs(delta)

    # Sample-level norms
    l0_frac = (delta != 0).sum(axis=1) / delta.shape[1]
    l1 = np.sum(np.abs(delta), axis=1)
    l2 = np.sqrt(np.sum(delta ** 2, axis=1))
    linf = np.max(np.abs(delta), axis=1)

    feature_means_orig = Xo.mean()
    feature_means_adv = Xa.mean()
    feature_stds_orig = Xo.std(ddof=0)
    feature_stds_adv = Xa.std(ddof=0)

    mean_shift = feature_means_adv - feature_means_orig
    rel_mean_shift = mean_shift / feature_means_orig.replace(0, np.nan)

    std_shift = feature_stds_adv - feature_stds_orig

    js_divs = {col: js_divergence_feature(Xo[col].to_numpy(), Xa[col].to_numpy()) for col in Xo.columns}

    corr_stats = correlation_shift(Xo, Xa)

    # Label distribution
    label_counts_orig = labels_o_used.value_counts(normalize=True).to_dict()
    label_counts_adv = labels_a_used.value_counts(normalize=True).to_dict()

    per_feature = []
    for col in Xo.columns:
        per_feature.append({
            "feature": col,
            "mean_orig": float(feature_means_orig[col]),
            "mean_adv": float(feature_means_adv[col]),
            "mean_abs_delta": float(abs_delta[:, Xo.columns.get_loc(col)].mean()),
            "mean_rel_mean_shift": float(rel_mean_shift[col]) if not np.isnan(rel_mean_shift[col]) else None,
            "std_orig": float(feature_stds_orig[col]),
            "std_adv": float(feature_stds_adv[col]),
            "std_shift": float(std_shift[col]),
            "js_divergence": js_divs[col]
        })

    # Top 10 by absolute relative mean shift
    top_rel_shift = sorted(
        [p for p in per_feature if p["mean_rel_mean_shift"] is not None],
        key=lambda d: abs(d["mean_rel_mean_shift"]), reverse=True
    )[:10]

    kpis = {
        "shape": {
            "rows_orig": int(len(df_orig)),
            "rows_adv": int(len(df_adv)),
            "numeric_features_compared": int(len(Xo.columns))
        },
        "label_distribution": {
            "original": label_counts_orig,
            "adversarial": label_counts_adv
        },
        "perturbation_norms": {
            "mean_l0_fraction": float(l0_frac.mean()),
            "mean_l1": float(l1.mean()),
            "mean_l2": float(l2.mean()),
            "mean_linf": float(linf.mean())
        },
        "correlation_shift": corr_stats,
        "aggregate_feature_stats": {
            "avg_abs_mean_shift": float(np.abs(mean_shift).mean()),
            "avg_abs_std_shift": float(np.abs(std_shift).mean()),
            "avg_js_divergence": float(np.mean(list(js_divs.values())))
        },
        "top_relative_mean_shift_features": top_rel_shift,
        "per_feature": per_feature
    }
    # Attach lightweight extras for plotting reuse (avoid huge duplication)
    kpis['_internal'] = {
        'Xo_head': Xo.head(3).to_dict(orient='list'),  # tiny sample just for sanity
        'feature_order': list(Xo.columns)
    }
    return kpis


# ---------------- PLOTTING ---------------- #
def ensure_plot_libs():
    if plt is None or sns is None:
        raise RuntimeError("Plotting libraries not available. Install matplotlib seaborn.")


def _save_fig(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_mean_shift(Xo: pd.DataFrame, Xa: pd.DataFrame, out_path: str, top_n: int = 30):
    means_o = Xo.mean()
    means_a = Xa.mean()
    shift = (means_a - means_o).abs().sort_values(ascending=False)
    top_features = shift.head(top_n).index
    data = pd.DataFrame({
        'feature': top_features,
        'mean_orig': means_o[top_features].values,
        'mean_adv': means_a[top_features].values
    })
    data_melt = data.melt(id_vars='feature', var_name='type', value_name='mean')
    plt.figure(figsize=(10, max(4, len(top_features) * 0.25)))
    sns.barplot(data=data_melt, x='mean', y='feature', hue='type')
    plt.title('Top Feature Mean Comparison (|Δ|)')
    _save_fig(out_path)


def plot_js_divergence(Xo: pd.DataFrame, Xa: pd.DataFrame, out_path: str, top_n: int = 30):
    js_vals = {}
    for c in Xo.columns:
        js_vals[c] = js_divergence_feature(Xo[c].to_numpy(), Xa[c].to_numpy())
    ser = pd.Series(js_vals).sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(10, max(4, len(ser) * 0.25)))
    sns.barplot(x=ser.values, y=ser.index, color='firebrick')
    plt.xlabel('JS Divergence')
    plt.title('Top Features by JS Divergence')
    _save_fig(out_path)


def plot_perturbation_norms(Xo: pd.DataFrame, Xa: pd.DataFrame, out_dir: str):
    delta = Xa.to_numpy() - Xo.to_numpy()
    l2 = np.sqrt(np.sum(delta ** 2, axis=1))
    linf = np.max(np.abs(delta), axis=1)
    plt.figure(figsize=(6, 4))
    sns.histplot(l2, bins=40, kde=True)
    plt.title('Sample L2 Perturbation Distribution')
    _save_fig(os.path.join(out_dir, 'perturb_l2_hist.png'))
    plt.figure(figsize=(6, 4))
    sns.histplot(linf, bins=40, kde=True, color='darkorange')
    plt.title('Sample Linf Perturbation Distribution')
    _save_fig(os.path.join(out_dir, 'perturb_linf_hist.png'))


def plot_correlation_shift(Xo: pd.DataFrame, Xa: pd.DataFrame, out_dir: str, max_features: int = 25):
    # To keep plots legible, optionally truncate
    if Xo.shape[1] > max_features:
        cols = Xo.var().sort_values(ascending=False).head(max_features).index
        Xo_p = Xo[cols]
        Xa_p = Xa[cols]
    else:
        Xo_p, Xa_p = Xo, Xa
    corr_o = Xo_p.corr().fillna(0.0)
    corr_a = Xa_p.corr().fillna(0.0)
    diff = corr_o - corr_a
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_o, cmap='coolwarm', center=0, cbar_kws={'shrink': .6})
    plt.title('Original Correlation (subset)')
    _save_fig(os.path.join(out_dir, 'corr_original.png'))
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_a, cmap='coolwarm', center=0, cbar_kws={'shrink': .6})
    plt.title('Adversarial Correlation (subset)')
    _save_fig(os.path.join(out_dir, 'corr_adversarial.png'))
    plt.figure(figsize=(6, 5))
    sns.heatmap(diff, cmap='bwr', center=0, cbar_kws={'shrink': .6})
    plt.title('Correlation Difference (Orig - Adv)')
    _save_fig(os.path.join(out_dir, 'corr_difference.png'))


def generate_plots(df_orig: pd.DataFrame, df_adv: pd.DataFrame, out_dir: str, top_n: int = 30):
    ensure_plot_libs()
    os.makedirs(out_dir, exist_ok=True)
    Xo, Xa = align_numeric(df_orig, df_adv)
    plot_mean_shift(Xo, Xa, os.path.join(out_dir, 'feature_mean_shift.png'), top_n=top_n)
    plot_js_divergence(Xo, Xa, os.path.join(out_dir, 'feature_js_divergence.png'), top_n=top_n)
    plot_perturbation_norms(Xo, Xa, out_dir)
    plot_correlation_shift(Xo, Xa, out_dir)


def save_json(obj: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def compare_pair(orig_path: str, adv_path: str, out_dir: str, plots: bool = False, plots_top: int = 30, plots_root: str | None = None):
    print(f"Comparing:\n  Original: {orig_path}\n  Adversarial: {adv_path}")
    df_o = load_dataset(orig_path)
    df_a = load_dataset(adv_path)
    kpis = compute_kpis(df_o, df_a)
    base = os.path.splitext(os.path.basename(orig_path))[0]
    out_path = os.path.join(out_dir, f"metrics_{base}.json")
    save_json(kpis, out_path)
    print(f"Saved KPIs -> {out_path}")
    # Brief console summary
    norms = kpis['perturbation_norms']
    corr = kpis['correlation_shift']
    print(f"Mean L2: {norms['mean_l2']:.4f}, Mean Linf: {norms['mean_linf']:.4f}, CorrΔ Frobenius: {corr['corr_frobenius_diff']:.4f}")
    if plots:
        plot_dir = os.path.join(plots_root or out_dir, f"plots_{base}")
        try:
            generate_plots(df_o, df_a, plot_dir, top_n=plots_top)
            print(f"Plots saved -> {plot_dir}")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] Plot generation failed: {e}")


def auto_match_and_compare(orig_dir: str, adv_dir: str, out_dir: str, plots: bool = False, plots_top: int = 30, plots_root: str | None = None):
    adv_files = [f for f in os.listdir(adv_dir) if f.endswith('.csv')]
    if not adv_files:
        raise ValueError("No adversarial CSV files found.")
    for adv in adv_files:
        adv_path = os.path.join(adv_dir, adv)
        # Reconstruct original filename heuristic
        if adv.endswith('_fgsm.csv'):
            orig_candidate = adv.replace('_fgsm.csv', '_labeled.csv')
        else:
            # fallback: remove trailing '_fgsm'
            orig_candidate = adv.replace('_fgsm', '')
        orig_path = os.path.join(orig_dir, orig_candidate)
        if not os.path.exists(orig_path):
            print(f"[WARN] Original file not found for {adv}, expected {orig_candidate}. Skipping.")
            continue
        compare_pair(orig_path, adv_path, out_dir, plots=plots, plots_top=plots_top, plots_root=plots_root)


def parse_args():
    p = argparse.ArgumentParser(
        description="Simple comparison tool for original vs adversarial datasets.\n"
                    "Usage patterns:\n"
                    "  1) python compare_datasets.py  (defaults: --orig original_dataset --adv adversarial_dataset)\n"
                    "  2) python compare_datasets.py --orig original_dataset/benign_labeled.csv --adv adversarial_dataset/benign_fgsm.csv\n"
                    "  3) python compare_datasets.py --orig path/to/orig_dir --adv path/to/adv_dir --plots"
    )
    p.add_argument('--orig', default='original_dataset', help='Original CSV file OR directory (default: original_dataset)')
    p.add_argument('--adv', default='adversarial_dataset', help='Adversarial CSV file OR directory (default: adversarial_dataset)')
    p.add_argument('--out', default='comparison_reports', help='Output directory for metrics JSON files')
    p.add_argument('--plots', action='store_true', help='Generate graphical outputs (requires matplotlib & seaborn)')
    p.add_argument('--top', type=int, default=30, help='Top N features for mean/JS bar charts')
    p.add_argument('--plots-dir', default=None, help='Optional separate base directory for plots (defaults to --out)')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    orig_is_file = os.path.isfile(args.orig)
    adv_is_file = os.path.isfile(args.adv)
    orig_is_dir = os.path.isdir(args.orig)
    adv_is_dir = os.path.isdir(args.adv)

    if orig_is_file and adv_is_file:
        # Single pair
        compare_pair(args.orig, args.adv, args.out, plots=args.plots, plots_top=args.top, plots_root=args.plots_dir)
        return
    if orig_is_dir and adv_is_dir:
        # Batch mode
        auto_match_and_compare(args.orig, args.adv, args.out, plots=args.plots, plots_top=args.top, plots_root=args.plots_dir)
        return
    raise SystemExit("ERROR: --orig and --adv must both be files OR both be directories.")


if __name__ == '__main__':
    main()
