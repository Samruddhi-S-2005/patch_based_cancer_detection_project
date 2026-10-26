import pandas as pd
import json

df_bench = pd.read_csv('result/objective_2/metrics/master_test_benchmark_comparison.csv')
print('=== master_test_benchmark_comparison.csv ===')
print(df_bench.to_string())

with open('result/objective_2/metrics/master_test_benchmark_comparison.json') as f:
    d = json.load(f)

print('\n=== Models in JSON ===')
for m, vals in d.items():
    print(f"{m}: Params={vals.get('parameters')}, Acc={vals.get('test_accuracy')}, Recall={vals.get('test_recall')}, ROC-AUC={vals.get('test_roc_auc')}")
