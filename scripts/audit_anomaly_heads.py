import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score, accuracy_score
import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

pred_dir = Path('result/objective_2/anomaly_model/predictions')
files = {
    'Ductal (BACH)': ('bach_ductal_test_predictions.csv', 0.4038),
    'Density (NuCLS)': ('nucls_density_test_predictions.csv', 0.5315),
    'Necrosis (BCSS)': ('bcss_necrosis_test_predictions.csv', 0.2715),
    'Atypia (MITOS - Quarantined)': ('mitos_atypia_test_predictions.csv', 0.0500)
}

print("=== RECALCULATING ANOMALY HEAD METRICS ===")
for name, (fname, tau) in files.items():
    p = pred_dir / fname
    print(f"\n--- {name} --- ({fname})")
    df = pd.read_csv(p)
    y_true = df['true_label'].values
    y_prob = df['anomaly_probability'].values
    y_pred = (y_prob >= tau).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    
    print(f"  Total Samples: {len(df)} (Class 0: {(y_true==0).sum()}, Class 1: {(y_true==1).sum()})")
    print(f"  Configured Threshold tau: {tau:.4f}")
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  Accuracy:         {acc*100:.2f}%")
    print(f"  Sensitivity/Rec:  {sens*100:.2f}%")
    print(f"  Specificity:      {spec*100:.2f}%")
    print(f"  Precision/PPV:    {prec*100:.2f}%")
    print(f"  NPV:              {npv*100:.2f}%")
    print(f"  F1-Score:         {f1:.4f}")
    print(f"  ROC-AUC:          {roc:.4f}")
    print(f"  PR-AUC:           {pr:.4f}")

# Compare with anomaly_test_metrics.json
metrics_file = Path('result/objective_2/anomaly_model/metrics/anomaly_test_metrics.json')
if metrics_file.exists():
    with open(metrics_file, 'r', encoding='utf-8') as f:
        saved_anom = json.load(f)
    print("\n=== SAVED anomaly_test_metrics.json ===")
    for h, d in saved_anom.items():
        print(f"Head {h}: ROC-AUC={d.get('roc_auc')}, PR-AUC={d.get('pr_auc')}, Sensitivity={d.get('sensitivity')}, Specificity={d.get('specificity')}, Threshold={d.get('calibrated_threshold')}")
