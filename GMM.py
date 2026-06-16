import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def accuracy_score(y_true, y_pred):
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if len(y_true) > 0 else 0

def confusion_matrix(y_true, y_pred, labels):
    n = len(labels)
    label_idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_idx and p in label_idx:
            cm[label_idx[t], label_idx[p]] += 1
    return cm

def classification_report(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels)
    lines = [f"\n{'Class':>12}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Support':>8}", "─" * 58]
    precs, recs, f1s = [], [], []
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp, fn = cm[:, i].sum() - tp, cm[i, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        precs.append(p); recs.append(r); f1s.append(f)
        lines.append(f"  Digit {label:>3}  {p:>10.3f}  {r:>8.3f}  {f:>8.3f}  {int(cm[i, :].sum()):>8}")
    lines.append("─" * 58)
    lines.append(f"  {'macro avg':>10}  {np.mean(precs):>10.3f}  {np.mean(recs):>8.3f}  {np.mean(f1s):>8.3f}  {len(y_true):>8}")
    return "\n".join(lines)

def plot_styled_confusion_matrix(cm, labels, test_acc, val_acc, filename):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"GMM CM | Test (Spk6): {test_acc*100:.2f}% | Val (Spk5): {val_acc*100:.2f}%", fontsize=12)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels); plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout(); plt.savefig(filename); plt.close()

def plot_styled_convergence(models, test_acc, val_acc, filename):
    digits = sorted(models.keys()); fig, axes = plt.subplots(2, 5, figsize=(20, 10)); axes = axes.flatten()
    for i, d in enumerate(digits):
        axes[i].plot(models[d].log_likelihoods_, color='tab:blue')
        axes[i].set_title(f"Digit {d}")
    plt.suptitle(f"EM Convergence | Test: {test_acc*100:.2f}% | Val: {val_acc*100:.2f}%", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(filename); plt.close()


class GaussianMixture:
    def __init__(self, n_components=8, n_iter=100, tol=1e-4, reg_covar=1e-6):
        self.n_components, self.n_iter, self.tol, self.reg_covar = n_components, n_iter, tol, reg_covar

    def _init_params(self, X):
        rng = np.random.default_rng(42); N, D = X.shape
        self.weights_ = np.full(self.n_components, 1.0 / self.n_components)
        indices = rng.choice(N, self.n_components, replace=False)
        self.means_ = X[indices].copy()
        self.covariances_ = np.tile(X.var(axis=0) + self.reg_covar, (self.n_components, 1))

    def _log_gaussian(self, X):
        N, D = X.shape; log_p = np.empty((N, self.n_components))
        for k in range(self.n_components):
            s, m = self.covariances_[k], self.means_[k]
            log_p[:, k] = -0.5 * (D * np.log(2*np.pi) + np.sum(np.log(s)) + np.sum((X - m)**2 / s, axis=1))
        return log_p

    def fit(self, X):
        self._init_params(X); self.log_likelihoods_ = []; prev_ll = -np.inf
        for _ in range(self.n_iter):
            lw = self._log_gaussian(X) + np.log(self.weights_ + 1e-300)
            lmax = lw.max(axis=1, keepdims=True)
            lnorm = lmax.squeeze() + np.log(np.exp(lw - lmax).sum(axis=1))
            resp = np.exp(lw - lnorm[:, None])
            total_ll = lnorm.sum(); self.log_likelihoods_.append(total_ll)
            nk = resp.sum(axis=0) + 1e-300; self.weights_ = nk / X.shape[0]
            self.means_ = (resp.T @ X) / nk[:, None]
            for k in range(self.n_components):
                self.covariances_[k] = (resp[:, k, None] * (X - self.means_[k])**2).sum(axis=0) / nk[k] + self.reg_covar
            if abs(total_ll - prev_ll) < self.tol: break
            prev_ll = total_ll
        return self

    def score(self, X):
        lw = self._log_gaussian(X) + np.log(self.weights_ + 1e-300)
        lmax = lw.max(axis=1, keepdims=True)
        return np.sum(lmax.squeeze() + np.log(np.exp(lw - lmax).sum(axis=1)))

def main():
    df = pd.read_csv('spoken_digits_features.csv')
    feats = [c for c in df.columns if c.startswith(('mfcc_', 'delta_', 'delta2_'))]

    sequences = {}
    for (d, spk, f), g in df.groupby(['digit_label', 'speaker_id', 'filename']):
        sequences.setdefault(int(d), []).append({'data': g[feats].values.astype(np.float64), 'spk': int(spk)})

    models, val_data, test_data = {}, [], []
    digits = sorted(sequences.keys())

    for d in digits:
        train_list = [item['data'] for item in sequences[d] if item['spk'] not in [5, 6]]
        val_list   = [item['data'] for item in sequences[d] if item['spk'] == 5]
        test_list  = [item['data'] for item in sequences[d] if item['spk'] == 6]
        
        val_data.extend([(d, x) for x in val_list])
        test_data.extend([(d, x) for x in test_list])

        if train_list:
            models[d] = GaussianMixture(n_components=8).fit(np.vstack(train_list))

    y_val_true, y_val_pred = [], []
    for t_lab, X in val_data:
        y_val_true.append(t_lab)
        y_val_pred.append(max(models.keys(), key=lambda d: models[d].score(X)))

    y_test_true, y_test_pred = [], []
    for t_lab, X in test_data:
        y_test_true.append(t_lab)
        y_test_pred.append(max(models.keys(), key=lambda d: models[d].score(X)))

    v_acc = accuracy_score(y_val_true, y_val_pred)
    t_acc = accuracy_score(y_test_true, y_test_pred)
    print(f"\nSpeaker 5 Validation Accuracy: {v_acc*100:.2f}%")
    print(f"Speaker 6 Test Accuracy:       {t_acc*100:.2f}%")
    print(classification_report(y_test_true, y_test_pred, labels=digits))
    plot_styled_confusion_matrix(confusion_matrix(y_test_true, y_test_pred, digits), digits, t_acc, v_acc, "GMM_CM.png")
    plot_styled_convergence(models, t_acc, v_acc, "GMM_Convergence.png")

if __name__ == '__main__':
    main()