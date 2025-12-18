import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_fscore_support
from .config import CFG
from .data_prep import load_and_clean_df
from .dataset import best_multilabel_split, make_tfds

def main():
    df = load_and_clean_df()
    _, val_df = best_multilabel_split(df)
    val_ds = make_tfds(val_df, augmenter=None, training=False, cache_name="val_eval")

    model_path = CFG.OUT_DIR / "best_retina_multilabel.keras"
    model = tf.keras.models.load_model(model_path, compile=False)

    y_true = val_df[CFG.CLASSES].values.astype(np.float32)
    y_pred = model.predict(val_ds, verbose=1)

    # Per-class AUC
    auc_scores = {}
    print("==== Per-class AUC ====")
    for i, c in enumerate(CFG.CLASSES):
        if len(np.unique(y_true[:, i])) < 2:
            print(f"{c}: not enough positives/negatives in val")
            continue
        sc = roc_auc_score(y_true[:, i], y_pred[:, i])
        auc_scores[c] = sc
        print(f"{c}: {sc:.4f}")
    if auc_scores:
        print("Mean AUC:", float(np.mean(list(auc_scores.values()))))

    # ROC plot
    CFG.OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    for i, c in enumerate(CFG.CLASSES):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        plt.plot(fpr, tpr, label=f"{c} (AUC={auc(fpr,tpr):.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves per disease (multi-label)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_roc = CFG.OUT_DIR / "roc_curves.png"
    plt.savefig(out_roc, dpi=150)
    plt.show()
    print("Saved:", out_roc)

    # Threshold metrics
    thr = 0.5
    y_hat = (y_pred >= thr).astype(int)
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_hat, average=None, zero_division=0)
    print(f"==== Per-class metrics @thr={thr} ====")
    for i, c in enumerate(CFG.CLASSES):
        print(f"{c}: P={prec[i]:.3f} R={rec[i]:.3f} F1={f1[i]:.3f} support(pos)={int(y_true[:,i].sum())}")
    print("Macro F1:", float(np.mean(f1)))

if __name__ == "__main__":
    main()
