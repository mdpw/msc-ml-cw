import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc, f1_score,
                             precision_score, recall_score, confusion_matrix, roc_curve)
import matplotlib.pyplot as plt

def eval_at_threshold(y_true, y_prob, thresh=0.5):
    y_pred = (y_prob >= thresh).astype(int)
    return {
        'F1': f1_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Accuracy': (y_true == y_pred).mean(),
        'AUC': roc_auc_score(y_true, y_prob)
    }

def best_f1_threshold(y_true, y_prob):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2*prec*rec/(prec+rec+1e-9)
    idx = f1.argmax()
    # thr has len-1 vs prec/rec; handle boundary
    chosen = thr[idx-1] if idx > 0 and idx-1 < len(thr) else 0.5
    return float(max(0.0, min(1.0, chosen))), float(f1.max()), float(auc(rec, prec))

def plot_and_save_curves(y_true, y_prob, title_prefix, out_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title_prefix} ROC')
    plt.grid(True)
    roc_path = f'{out_dir}/{title_prefix}_ROC.png'
    plt.savefig(roc_path, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{title_prefix} Precision-Recall')
    plt.grid(True)
    pr_path = f'{out_dir}/{title_prefix}_PR.png'
    plt.savefig(pr_path, bbox_inches='tight')
    plt.close()
    return roc_path, pr_path

def plot_and_save_confusion(y_true, y_pred, title_prefix, out_dir):
    import itertools
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(f'{title_prefix} Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    classes = ['No','Yes']
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    cm_path = f'{out_dir}/{title_prefix}_Confusion.png'
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    return cm_path
