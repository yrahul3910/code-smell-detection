from raise_utils.interpret import ScottKnott
import numpy as np
import sys


projects = ['android-backup-extractor-20140630',"AoI30","areca-7.4.7","freeplane-1.3.12","grinder-3.6","jedit","jexcelapi_2_6_12","junit-4.10","pmd-5.2.0","weka"]
results = {}

for i, filename in enumerate(sys.argv[1:]):
	results[filename] = {}
	with open(filename, 'r') as f:
		lines = f.readlines()

	lines = filter(lambda p: p.startswith('Performance'), lines)
	lines = map(lambda p: eval(p.split('Performance:')[1]), lines)
	lines = list(lines)

	auc = [x['auc'][-1] for x in lines]
	prec = [x['prec'][-1] for x in lines]
	rec = [x['rec'][-1] for x in lines]

	auc = np.array(auc).reshape(20, 10)
	prec = np.array(prec).reshape(20, 10)
	rec = np.array(rec).reshape(20, 10)

	for i, project in enumerate(projects):
		results[filename][project] = {}
		results[filename][project]['auc'] = auc[:,i]
		results[filename][project]['prec'] = prec[:,i]
		results[filename][project]['rec'] = rec[:,i]
		results[filename][project]['f1'] = [2*p*r/(p+r) for p, r in zip(prec[:,i], rec[:,i])]

		print(project)
		print('=' * len(project))
		print('auc:', np.median(auc[:,i]))
		print('prec:', np.median(prec[:,i]))
		print('rec:', np.median(rec[:,i]))

		p = np.median(prec[:,i])
		r = np.median(rec[:,i])
		f1 = 2 * p * r / (p + r)
		print('f1:', f1)
		print()

for project in projects:
	print(project)
	print('=' * len(project))
	for metric in ['prec', 'rec', 'f1', 'auc']:
		print(metric)
		print('-' * len(metric))
		d = {k: results[k][project][metric] for k in sys.argv[1:]}
		sk = ScottKnott(d)
		sk.pprint()
		print()
	print()
