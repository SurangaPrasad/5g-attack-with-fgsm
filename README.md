# 5g-attack-with-fgsm

Generate adversarial (FGSM) variants of 5G traffic / attack CSV datasets and compare original vs adversarial distributions using quantitative KPIs.

## Contents

- `fgsm_convertor.py` – Iterates through every `*_labeled.csv` file in `original_dataset/`, trains a simple feed-forward classifier per file, applies FGSM, and writes perturbed copies as `*_fgsm.csv` into `adversarial_dataset/`.
- `compare_datasets.py` – Computes KPI metrics between original and adversarial CSVs (pairwise or bulk) and stores JSON reports.
- `original_dataset/` – Source labeled datasets (`*_labeled.csv`).
- `adversarial_dataset/` – Auto-created output folder for adversarial datasets.

## Environment

Python 3.9+ recommended.

Install core dependencies (adjust if you already have a virtualenv):

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch pandas numpy scikit-learn
```

## 1. Generate Adversarial Datasets

Ensure original CSVs are in `original_dataset/` with names like:

```
benign_labeled.csv
brute_force_attack_labeled.csv
ddos_attack_labeled.csv
gtp_encapsulation_labeled.csv
gtp_malformed_labeled.csv
intra_upf_ddos_attack_labeled.csv
```

Run FGSM generation:

```bash
python3 fgsm_convertor.py
```

Outputs (example):

```
adversarial_dataset/
	benign_fgsm.csv
	brute_force_attack_fgsm.csv
	ddos_attack_fgsm.csv
	gtp_encapsulation_fgsm.csv
	gtp_malformed_fgsm.csv
	intra_upf_ddos_attack_fgsm.csv
```

### FGSM Parameters

Edit inside `fgsm_convertor.py`:

- `EPSILON` – Perturbation strength (default 0.05 after MinMax scaling)
- `NUM_EPOCHS` – Epochs for the per-file model (default 5)
- `BATCH_SIZE`, `LEARNING_RATE`
- Uses a simple 2-layer MLP per file; each file trains independently.

## 2. Compare Original vs Adversarial KPIs

`compare_datasets.py` provides multiple quantitative indicators:

### KPIs Implemented

| Category                | KPI                                                                                    |
| ----------------------- | -------------------------------------------------------------------------------------- |
| Shape                   | Row counts (original vs adversarial), number of overlapping numeric features           |
| Labels                  | Normalized label distribution shift                                                    |
| Feature Stats           | Mean shift, relative mean shift, std shift per feature                                 |
| Perturbation Magnitude  | Mean L0 fraction (changed features), mean L1, mean L2, mean Linf norms                 |
| Distribution Divergence | Jensen–Shannon divergence per feature (hist-based)                                     |
| Correlation Structure   | Frobenius norm and mean absolute difference of correlation matrices                    |
| Ranking                 | Top 10 features by absolute relative mean shift                                        |
| Full Detail             | Per-feature JSON record including mean/std before/after, mean abs delta, JS divergence |

### Run Comparisons (Simplified CLI)

The script auto-detects whether you supply files or directories.

1. Batch mode (default directories):

```bash
python3 compare_datasets.py
```

Equivalent:

```bash
python3 compare_datasets.py --orig original_dataset --adv adversarial_dataset --out comparison_reports
```

2. Single pair:

```bash
python3 compare_datasets.py --orig original_dataset/benign_labeled.csv --adv adversarial_dataset/benign_fgsm.csv --out comparison_reports
```

3. With plots:

```bash
python3 compare_datasets.py --plots
```

4. Limit top features in charts & custom plots directory:

```bash
python3 compare_datasets.py --plots --top 20 --plots-dir comparison_plots
```

### Output

Creates JSON files like:

```
comparison_reports/
	metrics_benign_labeled.json
	metrics_ddos_attack_labeled.json
	...
```

Each JSON contains (abridged):

```jsonc
{
	"shape": { ... },
	"label_distribution": { ... },
	"perturbation_norms": { ... },
	"correlation_shift": { ... },
	"aggregate_feature_stats": { ... },
	"top_relative_mean_shift_features": [ ... ],
	"per_feature": [ { "feature": "Flow Duration", ... } ]
}
```

## 3. Interpreting Metrics

- Large `mean_linf` or `mean_l2` values suggest stronger perturbations—tune `EPSILON` accordingly.
- High JS divergence pinpoints features whose distributional shape changed substantially.
- Correlation Frobenius difference highlights structural shifts in multivariate relationships.
- Relative mean shift helps identify features with disproportionate scaling sensitivity (watch divide-by-zero cases—reported as null when original mean is 0).

## 4. Extending

Ideas:

- Add visual plots (hist overlays, PCA drift) via matplotlib / seaborn.
- Export per-feature table to CSV: adapt `compare_datasets.py` to write `per_feature.csv`.
- Add other attacks (PGD, CW) and reuse comparison pipeline.

## 5. Repro Tips

- Set `PYTHONHASHSEED` and torch seeds if you want deterministic per-file training.
- Train once and reuse a shared model if you only care about perturbations for a specific class set.

## 6. License

Add license information here if needed.

---

Generated artifacts: adversarial CSVs + JSON KPI reports (+ optional plots). Adjust parameters and re-run to study sensitivity.
