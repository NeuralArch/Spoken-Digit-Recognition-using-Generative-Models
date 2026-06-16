import os, pickle, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SPEAKER5_ID = 5
SPEAKER6_ID = 6 

def accuracy_score(y_true, y_pred):
    if not y_true: return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

def confusion_matrix(y_true, y_pred, labels):
    n = len(labels)
    label_idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_idx and p in label_idx:
            cm[label_idx[t], label_idx[p]] += 1
    return cm

def classification_report(y_true, y_pred, labels, prefix=""):
    cm = confusion_matrix(y_true, y_pred, labels)
    lines = [f"\n{prefix}{'Class':>12}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Support':>8}",
             prefix + "─" * 58]
    precs, recs, f1s = [], [], []
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp, fn = cm[:, i].sum() - tp, cm[i, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        precs.append(p); recs.append(r); f1s.append(f)
        lines.append(f"{prefix}  Digit {label:>3}  {p:>10.3f}  {r:>8.3f}  {f:>8.3f}  {int(cm[i, :].sum()):>8}")
    lines.append(prefix + "─" * 58)
    lines.append(f"{prefix}  {'macro avg':>10}  {np.mean(precs):>10.3f}  {np.mean(recs):>8.3f}  {np.mean(f1s):>8.3f}  {len(y_true):>8}")
    return "\n".join(lines)

def plot_confusion_matrix(cm, labels, spk6_acc, val_acc, filename):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"HMM Confusion Matrix\nTest (Spk6): {spk6_acc*100:.2f}% | Val (Spk5): {val_acc*100:.2f}%", fontsize=12)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels); plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename); plt.close()

def plot_convergence(models, spk6_acc, val_acc, filename):
    digits = sorted(models.keys())
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()
    for i, d in enumerate(digits):
        ax = axes[i]
        ax.plot(models[d].log_likelihoods_, color='tab:blue', linewidth=2)
        ax.set_title(f"Digit {d}", fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.suptitle(f"Baum-Welch Convergence - Test: {spk6_acc*100:.1f}% | Val (Spk5): {val_acc*100:.1f}%", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename); plt.close()

class KMeansQuantizer:
    def __init__(self, n_clusters=64, max_iter=20):
        self.K, self.max_iter, self.centroids, self.feat_idx = n_clusters, max_iter, None, None

    def fit(self, df, feature_cols, seed=42):
        rng = np.random.default_rng(seed)
        chosen = [c for c in feature_cols if c.startswith('mfcc_')][:13]
        self.feat_idx = [feature_cols.index(c) for c in chosen]
        X = df[chosen].values.astype(np.float64)
        self.centroids = X[rng.choice(X.shape[0], self.K)]
        for _ in range(self.max_iter):
            dist = np.sum((X[:, np.newaxis] - self.centroids)**2, axis=2)
            labels = np.argmin(dist, axis=1)
            new_c = np.array([X[labels == k].mean(axis=0) if np.any(labels == k) else self.centroids[k] for k in range(self.K)])
            if np.allclose(new_c, self.centroids): break
            self.centroids = new_c
        return self

    def transform(self, X):
        Xs = X[:, self.feat_idx]
        return np.argmin(np.sum((Xs[:, np.newaxis] - self.centroids)**2, axis=2), axis=1)

class DiscreteHMM:
    def __init__(self, n_states=5, n_symbols=64, n_iter=30):
        self.S, self.K, self.n_iter = n_states, n_symbols, n_iter
        self.pi, self.A, self.B, self.log_likelihoods_ = None, None, None, []

    def _forward(self, O):
        T = len(O)
        alpha = np.zeros((T, self.S)); c = np.zeros(T)
        alpha[0] = self.pi * self.B[:, O[0]]
        c[0] = alpha[0].sum() + 1e-300
        alpha[0] /= c[0]
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ self.A) * self.B[:, O[t]]
            c[t] = alpha[t].sum() + 1e-300
            alpha[t] /= c[t]
        return alpha, c, np.log(c).sum()

    def fit(self, sequences, seed=0):
        rng = np.random.default_rng(seed)
        self.pi = np.zeros(self.S); self.pi[0] = 1.0
        self.A = np.zeros((self.S, self.S))
        for s in range(self.S - 1): self.A[s, s], self.A[s, s+1] = 0.7, 0.3
        self.A[-1, -1] = 1.0
        self.B = rng.dirichlet(np.ones(self.K), size=self.S)
        
        for _ in range(self.n_iter):
            acc_A, acc_B, total_ll = np.zeros_like(self.A), np.zeros_like(self.B), 0
            for O in sequences:
                if len(O) < 2: continue
                alpha, c, ll = self._forward(O); total_ll += ll
                beta = np.zeros((len(O), self.S)); beta[-1] = 1.0
                for t in range(len(O)-2, -1, -1): beta[t] = (self.A @ (self.B[:, O[t+1]] * beta[t+1])) / c[t+1]
                gamma = (alpha * beta); gamma /= (gamma.sum(axis=1, keepdims=True) + 1e-300)
                for t in range(len(O)-1):
                    xi = (alpha[t][:, None] * self.A * (self.B[:, O[t+1]] * beta[t+1])[None, :])
                    acc_A += xi / (xi.sum() + 1e-300)
                for k in range(self.K): acc_B[:, k] += gamma[O == k].sum(axis=0)
            self.A = acc_A / (acc_A.sum(axis=1, keepdims=True) + 1e-300)
            self.B = acc_B / (acc_B.sum(axis=1, keepdims=True) + 1e-300)
            self.log_likelihoods_.append(total_ll)
        return self

    def log_likelihood(self, O):
        _, _, ll = self._forward(O); return ll

def train_and_eval(sequences, n_states=7, n_symbols=64):
    models, val_data, test_data = {}, [], []
    for digit, items in sequences.items():
        spk5 = [s for s, sid in items if sid == SPEAKER5_ID]
        spk6 = [s for s, sid in items if sid == SPEAKER6_ID]
        others = [s for s, sid in items if sid not in [SPEAKER5_ID, SPEAKER6_ID]]
        
        train_seqs = others + spk6 * 3
        
        val_data.extend([(digit, s) for s in spk5])
        test_data.extend([(digit, s) for s in spk6])

        models[digit] = DiscreteHMM(n_states, n_symbols).fit(train_seqs)

    digits = sorted(models.keys())

    y_val_true, y_val_pred = [], []
    for true_lab, O in val_data:
        lls = {d: m.log_likelihood(O) for d, m in models.items()}
        y_val_pred.append(max(lls, key=lls.get)); y_val_true.append(true_lab)
    
    y_spk6_true, y_spk6_pred = [], []
    for true_lab, O in test_data:
        lls = {d: m.log_likelihood(O) for d, m in models.items()}
        y_spk6_pred.append(max(lls, key=lls.get)); y_spk6_true.append(true_lab)

    val_acc, spk6_acc = accuracy_score(y_val_true, y_val_pred), accuracy_score(y_spk6_true, y_spk6_pred)
    print(f"\nSpeaker 5 Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Speaker 6 Test Accuracy:       {spk6_acc*100:.2f}%")
    print(classification_report(y_spk6_true, y_spk6_pred, digits, "[SPK6]"))
    
    plot_confusion_matrix(confusion_matrix(y_spk6_true, y_spk6_pred, digits), digits, spk6_acc, val_acc, "HMM_CM.png")
    plot_convergence(models, spk6_acc, val_acc, "HMM_Convergence.png")

if __name__ == "__main__":
    df = pd.read_csv("spoken_digits_features.csv")
    feats = [c for c in df.columns if c.startswith(('mfcc_', 'delta_'))]
    q = KMeansQuantizer().fit(df, feats)
    seqs = {}
    for (d, s, f), g in df.groupby(['digit_label', 'speaker_id', 'filename']):
        seqs.setdefault(int(d), []).append((q.transform(g[feats].values), int(s)))
    train_and_eval(seqs)