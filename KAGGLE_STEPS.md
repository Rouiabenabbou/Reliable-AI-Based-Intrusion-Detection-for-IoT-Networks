# All steps to run the experiments on Kaggle

Run each notebook cell with **Shift + Enter**. Do the parts in order.

## Part A: one-time setup

1. **Put the code on GitHub.** On your repo page, click **Add file → Upload files**. Drag in `ids_reliability.py`, `stats_tests.py`, `README.md` and `requirements.txt`, then click **Commit changes**.
2. **Verify your phone number on Kaggle.** Go to kaggle.com, open your profile, then **Settings → Phone verification**. The GPU and internet won't work without this.
3. **Create a notebook.** Click **Create → New Notebook**.
4. **Turn on GPU and internet.** In the right-hand panel under **Settings**, set **Accelerator** to **GPU T4 x2** (or P100) and set **Internet** to **On**.

## Part B: add the datasets

5. In the right-hand panel, click **Add Input**. Search for each dataset below and click **+** on a version that contains CSV files:
   - CICIoT2023
   - Edge-IIoTset
   - TON_IoT (the network dataset, sometimes called "Train_Test_Network")
   - NF-ToN-IoT-v2
   - NF-BoT-IoT-v2
   - NF-UNSW-NB15-v2

   If you can't find one of them, skip it and tell Claude which one is missing.

## Part C: get the code and test it

6. **Cell 1: get the code.**
   ```
   !git clone https://github.com/Rouiabenabbou/Reliable-AI-Based-Intrusion-Detection-for-IoT-Networks.git
   %cd Reliable-AI-Based-Intrusion-Detection-for-IoT-Networks
   ```
7. **Cell 2: see where the data is.**
   ```
   !ls /kaggle/input
   ```
8. **Cell 3: check the column names of each dataset.**
   ```
   import glob, pandas as pd
   for f in sorted(glob.glob("/kaggle/input/*/**/*.csv", recursive=True))[:10]:
       print(f)
       print(pd.read_csv(f, nrows=2).columns.tolist()[-5:], "\n")
   ```
9. **Cell 4: quick test (a few minutes).**
   ```
   !python ids_reliability.py --mode indomain --dataset ciciot2023 --seeds 0 --sample-frac 0.02 --models rf xgb
   ```
10. **Send Claude a screenshot of the output of cells 2, 3 and 4.** Do this even if everything worked. If a folder or column name differs, Claude will fix it here. Don't start Part D until the test works.

    If a dataset is in a folder the code doesn't find, add `--data-dir /kaggle/input/<folder name>` to its command.

## Part D: full runs (several hours)

11. **RQ1: one dataset at a time.** Each line is its own cell.
    ```
    !python ids_reliability.py --mode indomain --dataset ciciot2023 --seeds 0 1 2 3 4
    ```
    ```
    !python ids_reliability.py --mode indomain --dataset edgeiiot --seeds 0 1 2 3 4 --sample-frac 1.0
    ```
    ```
    !python ids_reliability.py --mode indomain --dataset toniot --seeds 0 1 2 3 4 --sample-frac 1.0
    ```
12. **RQ2: cross-dataset tests.**
    ```
    !python ids_reliability.py --mode cross --dataset nf_toniot --targets nf_botiot nf_unsw
    ```
    ```
    !python ids_reliability.py --mode cross --dataset nf_botiot --targets nf_toniot nf_unsw
    ```
    ```
    !python ids_reliability.py --mode cross --dataset nf_unsw --targets nf_toniot nf_botiot
    ```
13. **RQ2: unseen attack families.**
    ```
    !python ids_reliability.py --mode lofo --dataset ciciot2023 --seeds 0 1 2 3 4
    ```
14. **Make the tables and statistics.**
    ```
    !python stats_tests.py --results results
    ```

**Tips**
- Kaggle stops a session after about 12 hours, or sooner if the browser tab sits idle. To run while you're away, click **Save Version → Save & Run All (Commit)**. It then runs in the background, and the results appear under the notebook's **Output** tab.
- Results are saved after every seed. If a run stops partway, rerun it with only the missing seeds and add `--out results_part2` so the finished results aren't overwritten.
- If you see `CUDA out of memory` or the notebook restarts, add `--sample-frac 0.05` to that command.

## Part E: send the results

15. **Download the results.** In the right-hand panel, open **Output** (`/kaggle/working`), find the `results` folder, and download it. Or zip it first with `!zip -r results.zip results`.
16. **Send Claude these files:**
    - everything in `results/tables/` (the `summary_*.csv` and `wilcoxon_*.csv` files)
    - the plots in `results/plots/`
    - any notes on what went wrong or looked strange

Claude will then help you interpret the results (step 5) and draft the paper sections (step 6).
