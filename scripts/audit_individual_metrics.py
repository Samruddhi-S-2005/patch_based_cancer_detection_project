import json
from pathlib import Path

files = {
    'EfficientNet': Path('result/objective_2/metrics/test_metrics.json'),
    'ResNet': Path('result/objective_2/metrics/resnet/test_metrics.json'),
    'DenseNet': Path('result/objective_2/metrics/densenet/test_metrics.json'),
    'VGG': Path('result/objective_2/metrics/vgg/test_metrics.json'),
}

for name, p in files.items():
    print(f'=== {name} === ({p})')
    if p.exists():
        with open(p) as f:
            d = json.load(f)
        total = d.get('tn', 0) + d.get('fp', 0) + d.get('fn', 0) + d.get('tp', 0)
        print(f'  Total test samples evaluated: {total}')
        print(f"  TN={d.get('tn')}, FP={d.get('fp')}, FN={d.get('fn')}, TP={d.get('tp')}")
        print(f"  Acc={d.get('test_accuracy', 0):.4f}, Recall={d.get('test_recall', 0):.4f}, Spec={d.get('test_specificity', 0):.4f}, Prec={d.get('test_precision', 0):.4f}")
        print(f"  ROC-AUC={d.get('test_roc_auc', 0):.4f}, PR-AUC={d.get('test_pr_auc', 0):.4f}")
    else:
        print('  FILE MISSING!')
