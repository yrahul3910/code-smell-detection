import numpy as np
from raise_utils.interpret.sk import Rx


with open('slurm-235127.out', 'r') as f:
	lines = f.readlines()

auc_lines = [x for x in lines if x.startswith('AUC')]
f1_lines = [x for x in lines if x.startswith('F1')]
prec_lines = [x for x in lines if x.startswith('prec')]
rec_lines = [x for x in lines if x.startswith('rec')]


projects = ['areca-7.4.7','freeplane-1.3.12','jedit','junit-4.10','pmd-5.2.0','weka','android-backup-extractor-20140630','grinder-3.6','AoI30','jexcelapi_2_6_12']

auc_us = [eval(x.split(':')[1]) for x in auc_lines[::2]]
auc_sota = [eval(x.split(':')[1]) for x in auc_lines[1::2]]

f1_us = [eval(x.split(':')[1]) for x in f1_lines[::2]]
f1_sota = [eval(x.split(':')[1]) for x in f1_lines[1::2]]

prec_us = [eval(x.split(':')[1]) for x in prec_lines[::2]]
prec_sota = [eval(x.split(':')[1]) for x in prec_lines[1::2]]

rec_us = [eval(x.split(':')[1]) for x in rec_lines[::2]]
rec_sota = [eval(x.split(':')[1]) for x in rec_lines[1::2]]

auc_us = np.array(auc_us).reshape(20, len(projects))
auc_sota = np.array(auc_sota).reshape(20, len(projects))

f1_us = np.array(f1_us).reshape(20, len(projects))
f1_sota = np.array(f1_sota).reshape(20, len(projects))

prec_us = np.array(prec_us).reshape(20, len(projects))
prec_sota = np.array(prec_sota).reshape(20, len(projects))

rec_us = np.array(rec_us).reshape(20, len(projects))
rec_sota = np.array(rec_sota).reshape(20, len(projects))

for i, project in enumerate(projects):
	print(project)
	print('-' * len(project))

	print('AUC:')
	d = {
		'us': auc_us[:,i],
		'sota': auc_sota[:,i]
	}
	Rx.show(Rx.sk(Rx.data(**d)))
	
	print()
	print('F1:')
	d = {
		'us': f1_us[:,i],
		'sota': f1_sota[:,i]
	}
	Rx.show(Rx.sk(Rx.data(**d)))

	print()
	print('prec:')
	d = {
		'us': prec_us[:,i],
		'sota': prec_sota[:,i]
	}
	Rx.show(Rx.sk(Rx.data(**d)))

	print()
	print('recall:')
	d = {
		'us': rec_us[:,i],
		'sota': rec_sota[:,i]
	}
	Rx.show(Rx.sk(Rx.data(**d)))
	print()
