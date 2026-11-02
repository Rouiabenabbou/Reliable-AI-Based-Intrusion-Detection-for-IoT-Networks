# Reliable AI-Based Intrusion Detection for IoT Networks: code

Binary intrusion detection (benign vs attack) on IoT traffic. The code compares
Random Forest, XGBoost, MLP and CNN-LSTM with the proposed method (a deep
ensemble of 5 MLPs plus temperature scaling). It measures detection, false
alarms, calibration and uncertainty.

| File | What it does |
| --- | --- |
| `ids_reliability.py` | Loads data, trains all models, saves result tables and plots |
| `stats_tests.py` | Mean ± std tables over seeds, Wilcoxon tests with Holm correction and bootstrap CIs |

## Experiments

| Mode | Research question | What it measures |
| --- | --- | --- |
| `indomain` | RQ1 | F1, FPR, AUROC, AUPRC, ECE, Brier, NLL, risk–coverage. Also repeats the test at benign:attack = 1:1, 10:1, 100:1 |
| `cross` | RQ2 | Train on one NetFlow dataset, test on another. Includes how well uncertainty separates the two (`shift_auroc_*`) |
| `lofo` | RQ2 | Leave one attack family out of training. Reports how much of it is still detected and whether it gets higher uncertainty |

Every model appears twice in the results: raw (`xgb`) and temperature-scaled
(`xgb+TS`). The decision threshold is set on the validation split so that FPR is
1% (`--target-fpr`). Columns ending in `_at_thr` use that threshold, and the
other columns use 0.5.

## Running on Kaggle

1. Create a new notebook at kaggle.com. Under **Settings → Accelerator**, choose
   **GPU** (you need a verified phone number for this).
2. Click **Add data** and search for each dataset. Pick an upload with the CSV files:
   - CICIoT2023
   - Edge-IIoTset
   - TON_IoT (the network dataset)
   - NF-ToN-IoT-v2, NF-BoT-IoT-v2, NF-UNSW-NB15-v2 (for the cross-dataset test)
3. Upload the two `.py` files with **File → Upload input**, or paste each one into a
   cell that starts with `%%writefile ids_reliability.py`.
4. Find where the data is, and check the label column:

   ```python
   !ls -R /kaggle/input | head -50
   import pandas as pd, glob
   f = glob.glob("/kaggle/input/**/*.csv", recursive=True)[0]
   print(f); print(pd.read_csv(f, nrows=3).columns.tolist())
   ```

   If the folder or the label column differs from `DATASETS` at the top of
   `ids_reliability.py`, edit that entry or pass `--data-dir /kaggle/input/<folder>`.

5. Do a quick test run first (1 seed, 2% of the data, 2 fast models):

   ```python
   !python ids_reliability.py --mode indomain --dataset ciciot2023 --seeds 0 --sample-frac 0.02 --models rf xgb
   ```

6. Full runs:

   ```python
   # RQ1: one run per dataset
   !python ids_reliability.py --mode indomain --dataset ciciot2023 --seeds 0 1 2 3 4
   !python ids_reliability.py --mode indomain --dataset edgeiiot   --seeds 0 1 2 3 4 --sample-frac 1.0
   !python ids_reliability.py --mode indomain --dataset toniot     --seeds 0 1 2 3 4 --sample-frac 1.0

   # RQ2: cross-dataset, train on each NetFlow dataset and test on the other two
   !python ids_reliability.py --mode cross --dataset nf_toniot --targets nf_botiot nf_unsw
   !python ids_reliability.py --mode cross --dataset nf_botiot --targets nf_toniot nf_unsw
   !python ids_reliability.py --mode cross --dataset nf_unsw   --targets nf_toniot nf_botiot

   # RQ2: unseen attack families
   !python ids_reliability.py --mode lofo --dataset ciciot2023 --seeds 0 1 2 3 4

   # Tables and statistics
   !python stats_tests.py --results results
   ```

7. Download the `results/` folder from the notebook's **Output** panel. The
   `tables/summary_*.csv` and `tables/wilcoxon_*.csv` files are what to share for
   the analysis step.

Kaggle stops a session after about 12 hours. Results are saved after every seed,
so if a run stops, rerun it with the remaining seeds and a different `--out` folder.

## Notes for the paper

- **Sampling:** `--sample-frac 0.1` loads 10% of each CSV, capped at 1M rows. The
  sample changes with the seed. Report the fraction you used.
- **Split:** stratified 60% train, 10% calibration (temperature), 10% validation
  (early stopping, threshold), 20% test.
- **Preprocessing:** identifier columns (IPs, ports, timestamps) are removed, along
  with non-numeric columns, inf/NaN rows and exact duplicates. The script prints the
  counts, so report them. log1p is applied to heavy-tailed non-negative columns,
  then standardisation fitted on the training split only.
- **CNN-LSTM:** treats one flow's feature vector as a 1-D sequence, not as a
  sequence of flows. State this in the Method section.
- **Statistics:** with 5 seeds, a two-sided Wilcoxon test on one dataset cannot
  give p < 0.05 (the minimum is 0.0625). `stats_tests.py` also writes "pooled" rows
  that pair across datasets (for example 15 pairs). Report the bootstrap CI as well.
- **Seeds** control the data sample, the split, and model initialisation.
- **Not included yet:** adversarial and noise tests (RQ3), and multi-class detection.

## Requirements

Kaggle already has everything. Elsewhere: `pip install -r requirements.txt`.
