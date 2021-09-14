import numpy as np

with open('output.log', 'r') as f:
    log = f.readlines()

log = filter(lambda p: p.startswith('History:'), log)
log = map(lambda p: eval(p.split('History:')[1]), log)
log = list(log)

datasets = [
    'android-backup-extractor-20140630',
    'AoI30',
    'areca-7.4.7',
    'freeplane-1.3.12',
    'grinder-3.6',
    'jedit',
    'jexcelapi_2_6_12',
    'junit-4.10',
    'pmd-5.2.0',
    'weka'
]

for i, data in enumerate(datasets):
    print(data)
    print('-' * len(data))
    results = log[i::10]

    recalls = [x['val_rec'][-1] for x in results]
    precisions = [x['val_prec'][-1] for x in results]
    aucs = [x['val_auc'][-1] for x in results]
    f1s = [2 * p * r / (p + r) for p, r in zip(precisions, recalls)]

    print('Recall:', np.mean(recalls))
    print('Precision:', np.mean(precisions))
    print('AUC:', np.mean(aucs))
    print('F1:', np.mean(f1s))
    print()
