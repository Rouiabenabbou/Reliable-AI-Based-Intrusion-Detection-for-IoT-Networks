"""
Reliable AI-Based Intrusion Detection for IoT Networks: experiment code.

What this script does
---------------------
Binary intrusion detection (benign = 0, attack = 1) with five models:
    rf        Random Forest
    xgb       XGBoost
    mlp       single MLP
    cnnlstm   CNN-LSTM (treats the feature vector as a 1-D sequence)
    ens       proposed: deep ensemble of 5 MLPs + temperature scaling

Every model is evaluated twice: raw probabilities, and after temperature
scaling fitted on a separate calibration split ("+TS" rows).

Experiments (pick one with --mode):
    indomain    train and test on the same dataset              (RQ1)
    baserate    in-domain models, test set resampled to
                benign:attack = 1:1, 10:1, 100:1                (RQ1, run inside indomain)
    cross       train on dataset A, test on dataset B           (RQ2)
    lofo        leave one attack family out of training         (RQ2)

Outputs go to --out (default ./results):
    results_<mode>.csv         one row per (dataset, model, seed[, ...])
    plots/*.png                reliability diagrams, risk-coverage curves

Run stats_tests.py afterwards for mean +/- std tables and Wilcoxon tests.

Example (in a Kaggle notebook cell):
    !python ids_reliability.py --mode indomain --dataset ciciot2023 --seeds 0 1 2 3 4
"""

import argparse
import glob
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")  # no screen needed; plots are saved to files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-7

# ---------------------------------------------------------------------------
# 1. Dataset configuration
# ---------------------------------------------------------------------------
# IMPORTANT: Kaggle dataset folders and column names differ between uploads.
# After adding a dataset in Kaggle, run   !ls /kaggle/input   and check the
# path and the label column, then edit the entry below (or pass --data-dir).
#
#   path        glob pattern for the CSV files
#   label       column holding the class label
#   benign      value(s) in `label` that mean benign traffic
#   family      column used for leave-one-family-out (None = derive from label)
#   drop        identifier columns removed so models can't "cheat"
DATASETS = {
    "ciciot2023": {
        "path": "/kaggle/input/*ciciot*2023*/**/*.csv",
        "label": "label",
        "benign": ["BenignTraffic", "Benign", "BENIGN"],
        "family": None,  # derived from the attack name, see attack_family()
        "drop": [],
    },
    "edgeiiot": {
        "path": "/kaggle/input/*edge*iiot*/**/*.csv",
        "label": "Attack_type",
        "benign": ["Normal"],
        "family": "Attack_type",
        "drop": ["Attack_label", "frame.time", "ip.src_host", "ip.dst_host",
                 "arp.src.proto_ipv4", "arp.dst.proto_ipv4", "tcp.srcport",
                 "tcp.dstport", "udp.port", "http.file_data",
                 "http.request.full_uri", "http.request.uri.query",
                 "icmp.transmit_timestamp", "tcp.options", "tcp.payload",
                 "mqtt.msg"],
    },
    "toniot": {
        "path": "/kaggle/input/*ton*iot*/**/*.csv",
        "label": "type",
        "benign": ["normal"],
        "family": "type",
        "drop": ["label", "ts", "src_ip", "dst_ip", "src_port", "dst_port"],
    },
    # NetFlow v2 versions share one feature set, so they are used for the
    # cross-dataset tests (RQ2).
    "nf_toniot": {
        "path": "/kaggle/input/*nf*ton*iot*v2*/**/*.csv",
        "label": "Attack",
        "benign": ["Benign"],
        "family": "Attack",
        "drop": ["Label", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT",
                 "L4_DST_PORT", "FLOW_START_MILLISECONDS",
                 "FLOW_END_MILLISECONDS"],
    },
    "nf_botiot": {
        "path": "/kaggle/input/*nf*bot*iot*v2*/**/*.csv",
        "label": "Attack",
        "benign": ["Benign"],
        "family": "Attack",
        "drop": ["Label", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT",
                 "L4_DST_PORT", "FLOW_START_MILLISECONDS",
                 "FLOW_END_MILLISECONDS"],
    },
    "nf_unsw": {
        "path": "/kaggle/input/*nf*unsw*v2*/**/*.csv",
        "label": "Attack",
        "benign": ["Benign"],
        "family": "Attack",
        "drop": ["Label", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT",
                 "L4_DST_PORT", "FLOW_START_MILLISECONDS",
                 "FLOW_END_MILLISECONDS"],
    },
}


def attack_family(name):
    """Map a CICIoT2023 attack name to one of its 7 families (+ Benign)."""
    n = str(name)
    for prefix in ["DDoS", "DoS", "Mirai", "Recon"]:
        if n.startswith(prefix):
            return prefix
    if "Spoofing" in n:
        return "Spoofing"
    if n == "VulnerabilityScan":
        return "Recon"
    if n == "DictionaryBruteForce":
        return "BruteForce"
    if n in ("SqlInjection", "XSS", "Uploading_Attack", "CommandInjection",
             "Backdoor_Malware", "BrowserHijacking"):
        return "Web"
    return n


def set_seed(seed):
    """Fix every source of randomness we control."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 2. Loading and preprocessing
# ---------------------------------------------------------------------------
def load_dataset(name, sample_frac, seed, data_dir=None, max_rows=None):
    """Read all CSVs of a dataset, sampling `sample_frac` of each file.

    Returns X (DataFrame, numeric only), y (0/1 array), family (str array).
    """
    cfg = DATASETS[name]
    pattern = os.path.join(data_dir, "**", "*.csv") if data_dir else cfg["path"]
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found for '{name}' with pattern {pattern}. "
            "Run !ls /kaggle/input and pass --data-dir.")

    parts = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=seed)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)

    if cfg["label"] not in df.columns:
        raise KeyError(f"Label column '{cfg['label']}' not in {name}. "
                       f"Columns are: {list(df.columns)[:50]}")

    labels = df[cfg["label"]].astype(str).str.strip()
    y = (~labels.isin(cfg["benign"])).astype(int).to_numpy()
    if cfg["family"] is None:
        family = labels.map(attack_family).to_numpy()
    else:
        family = df[cfg["family"]].astype(str).str.strip().to_numpy()
    family = np.where(y == 0, "Benign", family)

    # Drop label columns, identifiers, and anything non-numeric.
    drop = set(cfg["drop"]) | {cfg["label"]}
    if cfg["family"]:
        drop.add(cfg["family"])
    X = df.drop(columns=[c for c in drop if c in df.columns])
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        print(f"[{name}] dropping non-numeric columns: {non_numeric}")
        X = X.drop(columns=non_numeric)

    # inf -> NaN -> drop rows; report how many were removed.
    X = X.replace([np.inf, -np.inf], np.nan)
    keep = X.notna().all(axis=1).to_numpy().copy()
    # Exact duplicate rows (features + label) are removed.
    dup = pd.concat([X, pd.Series(y, name="__y", index=X.index)], axis=1).duplicated().to_numpy()
    keep &= ~dup
    print(f"[{name}] rows loaded={len(X)}, removed NaN/inf={int((~X.notna().all(axis=1)).sum())}, "
          f"removed duplicates={int(dup.sum())}, attack share={y[keep].mean():.3f}")
    X = X[keep].astype(np.float32).reset_index(drop=True)
    return X, y[keep], family[keep]


def split_data(X, y, family, seed):
    """Stratified 60/10/10/20 split: train / calibration / validation / test."""
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
    cal, rest = train_test_split(rest, test_size=0.75, stratify=y[rest], random_state=seed)
    val, te = train_test_split(rest, test_size=2 / 3, stratify=y[rest], random_state=seed)
    return {"train": tr, "cal": cal, "val": val, "test": te}


class Preprocessor:
    """log1p on non-negative heavy-tailed columns, then standardise.

    Fitted on the training split only, then applied to every other split
    (and to other datasets in the cross-dataset test).
    """

    def fit(self, X):
        self.columns = list(X.columns)
        nonneg = (X.min(axis=0) >= 0).to_numpy()
        skewed = (X.max(axis=0) > 100).to_numpy()
        self.log_cols = [c for c, a, b in zip(self.columns, nonneg, skewed) if a and b]
        Xt = self._log(X)
        # drop constant columns (they carry no information)
        self.keep = Xt.std(axis=0).to_numpy() > 0
        self.scaler = StandardScaler().fit(Xt.to_numpy()[:, self.keep])
        return self

    def _log(self, X):
        X = X[self.columns].copy()
        for c in self.log_cols:
            X[c] = np.log1p(X[c].clip(lower=0))
        return X

    def transform(self, X):
        Xt = self._log(X).to_numpy()[:, self.keep]
        Xt = self.scaler.transform(Xt)
        return np.clip(Xt, -20, 20).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Models
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, d, hidden=(256, 128, 64), dropout=0.2):
        super().__init__()
        layers, prev = [], d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # one logit per flow


class CNNLSTM(nn.Module):
    """1-D conv over the feature vector, then an LSTM.

    Note for the paper: the features of one flow are treated as a sequence.
    This is common in the IDS literature but has no real temporal meaning;
    say so in the Method section.
    """

    def __init__(self, d, channels=32, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2))
        self.lstm = nn.LSTM(channels, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.conv(x.unsqueeze(1))          # (B, C, d/2)
        out, _ = self.lstm(h.transpose(1, 2))  # (B, d/2, hidden)
        return self.head(out[:, -1]).squeeze(-1)


def train_torch(model, Xtr, ytr, Xval, yval, seed, epochs=20, batch=1024, lr=1e-3, patience=3):
    """Train with class-weighted BCE and early stopping on validation loss."""
    set_seed(seed)
    model = model.to(DEVICE)
    pos_weight = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=g)
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            xb, yb = Xtr_t[b].to(DEVICE), ytr_t[b].to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        vl = loss_fn(torch.from_numpy(predict_logits(model, Xval)).to(DEVICE),
                     torch.from_numpy(yval.astype(np.float32)).to(DEVICE)).item()
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_logits(model, X, batch=8192):
    model.eval()
    out = [model(torch.from_numpy(X[i:i + batch]).to(DEVICE)).cpu().numpy()
           for i in range(0, len(X), batch)]
    return np.concatenate(out)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def to_logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Fitted:
    """Wraps any trained model so it returns logits for X.

    For an ensemble, `member_logits` returns one row per member, which is
    used for the epistemic (mutual information) uncertainty.
    """

    def __init__(self, kind, models):
        self.kind, self.models = kind, models

    def member_logits(self, X):
        if self.kind == "sk":
            return np.stack([to_logit(m.predict_proba(X)[:, 1]) for m in self.models])
        return np.stack([predict_logits(m, X) for m in self.models])

    def logits(self, X):
        ml = self.member_logits(X)
        if len(ml) == 1:
            return ml[0]
        # average probabilities (standard for deep ensembles), back to logit
        return to_logit(sigmoid(ml).mean(axis=0))


def train_model(name, Xtr, ytr, Xval, yval, seed):
    d = Xtr.shape[1]
    if name == "rf":
        m = RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced",
                                   min_samples_leaf=2, random_state=seed)
        return Fitted("sk", [m.fit(Xtr, ytr)])
    if name == "xgb":
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        m = XGBClassifier(n_estimators=1000, learning_rate=0.1, max_depth=8,
                          subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                          tree_method="hist", device=DEVICE, early_stopping_rounds=30,
                          eval_metric="logloss", random_state=seed)
        m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        return Fitted("sk", [m])
    if name == "mlp":
        return Fitted("torch", [train_torch(MLP(d), Xtr, ytr, Xval, yval, seed)])
    if name == "cnnlstm":
        return Fitted("torch", [train_torch(CNNLSTM(d), Xtr, ytr, Xval, yval, seed)])
    if name == "ens":
        # 5 MLPs, each with its own seed (different init and batch order)
        members = [train_torch(MLP(d), Xtr, ytr, Xval, yval, seed * 100 + k) for k in range(5)]
        return Fitted("torch", members)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# 4. Calibration: temperature scaling
# ---------------------------------------------------------------------------
def fit_temperature(logits, y):
    """Find T > 0 minimising NLL of sigmoid(logits / T) on the calibration split."""
    z = torch.tensor(logits, dtype=torch.float64)
    t = torch.tensor(y, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(z / log_t.exp(), t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


# ---------------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------------
def ece_score(y, p, bins=15):
    """Expected calibration error for the predicted class, equal-width bins."""
    conf = np.where(p >= 0.5, p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y)
    edges = np.linspace(0.5, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def threshold_at_fpr(y, p, target_fpr):
    """Smallest threshold whose FPR on (y, p) is <= target_fpr."""
    benign = np.sort(p[y == 0])[::-1]
    k = int(np.floor(target_fpr * len(benign)))
    return float(benign[k]) if k < len(benign) else 0.0


def tpr_at_fpr(y, p, fpr_target):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(fpr_target, fpr, tpr))


def risk_coverage(y, p, unc):
    """Error rate on the most-confident fraction of flows, for each coverage."""
    order = np.argsort(unc)  # most certain first
    err = ((p[order] >= 0.5).astype(int) != y[order]).astype(float)
    cov = np.arange(1, len(y) + 1) / len(y)
    risk = np.cumsum(err) / np.arange(1, len(y) + 1)
    return cov, risk


def evaluate(y, p, thr, unc):
    """All metrics for one set of predictions."""
    pred05 = (p >= 0.5).astype(int)
    predt = (p > thr).astype(int)  # strict, so ties at the threshold count as benign
    benign = y == 0
    res = {
        "n": len(y), "attack_share": float(y.mean()),
        "f1": f1_score(y, pred05, zero_division=0),
        "fpr": float(pred05[benign].mean()) if benign.any() else np.nan,
        "f1_at_thr": f1_score(y, predt, zero_division=0),
        "fpr_at_thr": float(predt[benign].mean()) if benign.any() else np.nan,
        "precision_at_thr": float(y[predt == 1].mean()) if predt.any() else np.nan,
        "ece": ece_score(y, p),
        "brier": brier_score_loss(y, p),
        "nll": float(-np.mean(y * np.log(np.clip(p, EPS, 1)) + (1 - y) * np.log(np.clip(1 - p, EPS, 1)))),
    }
    if 0 < y.mean() < 1:
        res["auroc"] = roc_auc_score(y, p)
        res["auprc"] = average_precision_score(y, p)
        res["tpr_at_fpr_1e-2"] = tpr_at_fpr(y, p, 1e-2)
        res["tpr_at_fpr_1e-3"] = tpr_at_fpr(y, p, 1e-3)
    cov, risk = risk_coverage(y, p, unc)
    res["aurc"] = float(np.trapezoid(risk, cov)) if hasattr(np, "trapezoid") else float(np.trapz(risk, cov))
    for c in (0.9, 0.95):
        res[f"risk_at_cov_{c}"] = float(risk[int(c * len(y)) - 1])
    return res


def resample_base_rate(y, ratio, seed):
    """Indices of a test subset with benign:attack = ratio:1 (keeps all benign if possible)."""
    rng = np.random.default_rng(seed)
    ben, att = np.where(y == 0)[0], np.where(y == 1)[0]
    n_att = min(len(att), len(ben) // ratio)
    n_ben = n_att * ratio
    if n_att == 0:
        return None
    return np.concatenate([rng.choice(ben, n_ben, replace=False),
                           rng.choice(att, n_att, replace=False)])


# ---------------------------------------------------------------------------
# 6. Running one model and scoring it (raw and temperature-scaled)
# ---------------------------------------------------------------------------
def predict_all(model, X, T=1.0):
    """Return probability, total uncertainty (entropy) and, for the ensemble,
    epistemic uncertainty (mutual information)."""
    ml = model.member_logits(X) / T
    probs = sigmoid(ml)
    p = probs.mean(axis=0)
    total = entropy(p)
    mi = total - entropy(probs).mean(axis=0) if len(ml) > 1 else np.zeros_like(p)
    return p, total, mi


def score_model(model, name, Xs, ys, target_fpr, seed, extra_tests=None):
    """Score a trained model raw and with temperature scaling.

    Xs/ys: dicts with 'cal', 'val', 'test' arrays.
    extra_tests: dict of {test_name: (X, y)} for additional test sets
                 (e.g. another dataset in the cross-dataset experiment).
    Returns a list of result rows and the test predictions (for plots).
    """
    rows, preds = [], {}
    T_fit = fit_temperature(model.logits(Xs["cal"]), ys["cal"])
    for variant, T in [("raw", 1.0), ("TS", T_fit)]:
        label = name if variant == "raw" else f"{name}+TS"
        p_val, _, _ = predict_all(model, Xs["val"], T)
        thr = threshold_at_fpr(ys["val"], p_val, target_fpr)
        tests = {"test": (Xs["test"], ys["test"])}
        tests.update(extra_tests or {})
        for tname, (Xt, yt) in tests.items():
            p, unc, mi = predict_all(model, Xt, T)
            r = evaluate(yt, p, thr, unc)
            r.update(model=label, temperature=T, threshold=thr, test_set=tname, seed=seed)
            rows.append(r)
            preds[(label, tname)] = (yt, p, unc, mi)
    return rows, preds


# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------
def plot_reliability(preds, path, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    bins = np.linspace(0, 1, 11)
    for label, (y, p) in preds.items():
        idx = np.digitize(p, bins[1:-1])
        xs = [p[idx == b].mean() for b in range(10) if (idx == b).any()]
        ys_ = [y[idx == b].mean() for b in range(10) if (idx == b).any()]
        ax.plot(xs, ys_, "o-", ms=3, label=label)
    ax.set_xlabel("Predicted probability of attack")
    ax.set_ylabel("Observed fraction of attacks")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_risk_coverage(preds, path, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    for label, (y, p, unc) in preds.items():
        cov, risk = risk_coverage(y, p, unc)
        ax.plot(cov, risk, label=label)
    ax.set_xlabel("Coverage (share of flows the model decides on)")
    ax.set_ylabel("Error rate on covered flows")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Experiments
# ---------------------------------------------------------------------------
def prepare(name, args, seed):
    X, y, fam = load_dataset(name, args.sample_frac, seed, args.data_dir, args.max_rows)
    sp = split_data(X, y, fam, seed)
    pre = Preprocessor().fit(X.iloc[sp["train"]])
    Xs = {k: pre.transform(X.iloc[v]) for k, v in sp.items()}
    ys = {k: y[v] for k, v in sp.items()}
    fams = {k: fam[v] for k, v in sp.items()}
    return X, y, fam, sp, pre, Xs, ys, fams


def run_indomain(args):
    rows = []
    for seed in args.seeds:
        set_seed(seed)
        _, _, _, _, _, Xs, ys, _ = prepare(args.dataset, args, seed)
        rel, rc = {}, {}
        for m in args.models:
            t0 = time.time()
            model = train_model(m, Xs["train"], ys["train"], Xs["val"], ys["val"], seed)
            train_s = time.time() - t0
            r, preds = score_model(model, m, Xs, ys, args.target_fpr, seed)
            t0 = time.time()
            model.logits(Xs["test"][:1000])
            infer_ms = (time.time() - t0) * 1000
            for row in r:
                row.update(dataset=args.dataset, train_seconds=train_s,
                           infer_ms_per_1000=infer_ms, base_rate="natural")
            rows += r
            # Base-rate experiment (RQ1): same models, resampled test sets
            for ratio in (1, 10, 100):
                idx = resample_base_rate(ys["test"], ratio, seed)
                if idx is None:
                    continue
                for (label, tname), (yt, p, unc, _) in preds.items():
                    thr = next(x["threshold"] for x in r if x["model"] == label)
                    br = evaluate(yt[idx], p[idx], thr, unc[idx])
                    br.update(model=label, test_set="test", seed=seed, dataset=args.dataset,
                              base_rate=f"{ratio}:1")
                    rows.append(br)
            for (label, _), (yt, p, unc, _) in preds.items():
                rel[label] = (yt, p)
                rc[label] = (yt, p, unc)
            print(f"seed {seed} {m}: done in {train_s:.0f}s")
        if seed == args.seeds[0]:
            plot_reliability(rel, f"{args.out}/plots/reliability_{args.dataset}_seed{seed}.png",
                             f"Reliability, {args.dataset}, seed {seed}")
            plot_risk_coverage(rc, f"{args.out}/plots/riskcov_{args.dataset}_seed{seed}.png",
                               f"Risk-coverage, {args.dataset}, seed {seed}")
        save(rows, args, "indomain")


def run_cross(args):
    """Train on --dataset, test on each --target dataset (same feature schema)."""
    rows = []
    for seed in args.seeds:
        set_seed(seed)
        Xsrc, _, _, sp, pre, Xs, ys, _ = prepare(args.dataset, args, seed)
        extra = {}
        tdirs = args.target_dirs or [None] * len(args.targets)
        for tgt, tdir in zip(args.targets, tdirs):
            Xt, yt, _ = load_dataset(tgt, args.sample_frac, seed, tdir, args.max_rows)
            missing = [c for c in pre.columns if c not in Xt.columns]
            if missing:
                raise KeyError(f"{tgt} lacks columns {missing}; cross-dataset needs one schema.")
            # use a test-sized random subset of the target dataset
            rng = np.random.default_rng(seed)
            n = min(len(yt), len(ys["test"]))
            idx = rng.choice(len(yt), n, replace=False)
            extra[tgt] = (pre.transform(Xt.iloc[idx]), yt[idx])
        for m in args.models:
            model = train_model(m, Xs["train"], ys["train"], Xs["val"], ys["val"], seed)
            r, preds = score_model(model, m, Xs, ys, args.target_fpr, seed, extra)
            # Does uncertainty flag shifted traffic? AUROC of uncertainty for
            # source-test (0) vs target (1) flows.
            for row in r:
                row.update(source=args.dataset)
                if row["test_set"] != "test":
                    _, _, u_in, mi_in = preds[(row["model"], "test")]
                    _, _, u_out, mi_out = preds[(row["model"], row["test_set"])]
                    lab = np.r_[np.zeros(len(u_in)), np.ones(len(u_out))]
                    row["shift_auroc_entropy"] = roc_auc_score(lab, np.r_[u_in, u_out])
                    if m == "ens":
                        row["shift_auroc_mi"] = roc_auc_score(lab, np.r_[mi_in, mi_out])
            rows += r
            print(f"seed {seed} {m}: done")
        save(rows, args, "cross")


def run_lofo(args):
    """Leave one attack family out of train/cal/val; test on it."""
    rows = []
    for seed in args.seeds:
        set_seed(seed)
        X, y, fam, sp, _, _, _, _ = prepare(args.dataset, args, seed)
        families = [f for f in np.unique(fam) if f != "Benign"]
        held = args.holdout or families
        for h in held:
            keep = {k: v[fam[v] != h] for k, v in sp.items() if k != "test"}
            pre = Preprocessor().fit(X.iloc[keep["train"]])
            Xs = {k: pre.transform(X.iloc[v]) for k, v in keep.items()}
            ys = {k: y[v] for k, v in keep.items()}
            te_seen = sp["test"][fam[sp["test"]] != h]
            te_unseen = sp["test"][fam[sp["test"]] == h]
            Xs["test"], ys["test"] = pre.transform(X.iloc[te_seen]), y[te_seen]
            extra = {"unseen_family": (pre.transform(X.iloc[te_unseen]), y[te_unseen])}
            for m in args.models:
                model = train_model(m, Xs["train"], ys["train"], Xs["val"], ys["val"], seed)
                r, preds = score_model(model, m, Xs, ys, args.target_fpr, seed, extra)
                for row in r:
                    row.update(dataset=args.dataset, held_out=h)
                    if row["test_set"] == "unseen_family":
                        yt, p, u, mi = preds[(row["model"], "unseen_family")]
                        row["detection_rate_unseen"] = float((p > row["threshold"]).mean())
                        # Are unseen attacks more uncertain than seen-test flows?
                        _, _, u_in, mi_in = preds[(row["model"], "test")]
                        lab = np.r_[np.zeros(len(u_in)), np.ones(len(u))]
                        row["unseen_auroc_entropy"] = roc_auc_score(lab, np.r_[u_in, u])
                        if m == "ens":
                            row["unseen_auroc_mi"] = roc_auc_score(lab, np.r_[mi_in, mi])
                rows += r
                print(f"seed {seed} held-out {h} {m}: done")
        save(rows, args, "lofo")


def save(rows, args, mode):
    """Write results after every seed so a crash doesn't lose finished work."""
    tag = args.dataset
    path = f"{args.out}/results_{mode}_{tag}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"saved {len(rows)} rows -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["indomain", "cross", "lofo"], required=True)
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--targets", nargs="+", default=[], help="cross mode: datasets to test on")
    ap.add_argument("--target-dirs", nargs="+", default=None, help="cross mode: folders for --targets, same order")
    ap.add_argument("--holdout", nargs="+", default=None, help="lofo mode: families to hold out (default all)")
    ap.add_argument("--models", nargs="+", default=["rf", "xgb", "mlp", "cnnlstm", "ens"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--sample-frac", type=float, default=0.1, help="share of each CSV to load")
    ap.add_argument("--max-rows", type=int, default=1_000_000, help="cap on rows after sampling")
    ap.add_argument("--target-fpr", type=float, default=0.01, help="FPR used to pick the decision threshold")
    ap.add_argument("--data-dir", default=None, help="override the dataset folder")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(f"{args.out}/plots", exist_ok=True)
    with open(f"{args.out}/config_{args.mode}_{args.dataset}.json", "w") as f:
        json.dump(vars(args) | {"device": DEVICE}, f, indent=2)
    print(f"device: {DEVICE}")
    {"indomain": run_indomain, "cross": run_cross, "lofo": run_lofo}[args.mode](args)


if __name__ == "__main__":
    main()
