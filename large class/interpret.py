import numpy as np
import sys
from raise_utils.interpret.sk import Rx


projects = ['areca-7.4.7','freeplane-1.3.12','jedit','junit-4.10','pmd-5.2.0','weka','android-backup-extractor-20140630','grinder-3.6','AoI30','jexcelapi_2_6_12']

d = {}
for filename in sys.argv[1:]:
	d[filename] = {}
	print('=' * 20, filename, '=' * 20)
	with open(filename, 'r') as f:
		lines = f.readlines()

	perf_lines = [eval(x.split(':')[1]) for x in lines if x.startswith('Perf')]
	print(len(perf_lines), len(perf_lines[0]))
	perf_lines = np.array(perf_lines).reshape((20, len(projects), 5))

	metrics = ['loss', 'acc', 'auc', 'rec', 'prec']

	for i, project in enumerate(projects):
		d[filename][project] = {}
		print(project)
		print('-' * len(project))

		for j, metric in enumerate(metrics):
			# ignore first two
			if j < 2:
				continue
			
			print(metric, '-', end=' ')
			result = perf_lines[:,i,j]
			print(round(np.median(result), 3), ' | ', end='')
			d[filename][project][metric] = result
		
		print()
		print()

print('=' * 80)
for project in projects:
	print('=' * 20, project, '=' * 20)
	for metric in metrics[2:]:
		print(metric)
		d_ = {k: v[project][metric] for k, v in d.items()}
		Rx.show(Rx.sk(Rx.data(**d_)))

