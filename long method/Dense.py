import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import AUC, Precision, Recall
from sklearn import preprocessing,metrics
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics
from sklearn import svm
from raise_utils.data import Data
from raise_utils.transforms import Transform
from hyperopt import hp, tpe, fmin
import pickle
import joblib
import time

MODEL_NUMBER=1
SUBSET_SIZE=0.7

tr=[]
trp=[]


class Model:
	def __init__(self, name=''):
		self.name = name

	def _run(self, args):
		n_units = int(args['n_units'])
		n_layers = int(args['n_layers'])

		X_train, X_test, y_train, y_test = train_test_split(self.X, self.y)

		clf = Sequential()
		clf.add(Dense(16, input_shape=(11,), activation='relu'))
		for _ in range(n_layers-1):
			clf.add(Dense(16, activation='relu'))
		clf.add(Dense(1, activation='sigmoid'))

		clf.compile(optimizer='adam',
				loss='binary_crossentropy',
				metrics=['accuracy', AUC(name='auc'), Recall(name='recall'), Precision(name='precision')])		

		clf.fit(X_train, y_train, epochs=10, batch_size=128, verbose=0)
		preds = clf.predict_classes(X_test)
		return 1. - metrics.f1_score(y_test, preds)
	
	def fit(self, X, y):
		self.X = X
		self.y = y
		space = hp.choice('params', [{
			'n_units': hp.uniform('n_units', 5, 20), 
			'n_layers': hp.uniform('n_layers', 2, 7)
		}])
		best = fmin(self._run, space, algo=tpe.suggest, max_evals=20)
		self.best = best
	
	def predict(self, X):
		n_units = int(self.best['n_units'])
		n_layers = int(self.best['n_layers'])

		clf = Sequential()
		clf.add(Dense(16, input_shape=(11,), activation='relu'))
		for _ in range(n_layers-1):
			clf.add(Dense(16, activation='relu'))
		clf.add(Dense(1, activation='sigmoid'))

		clf.compile(optimizer='adam',
				loss='binary_crossentropy',
				metrics=['accuracy', AUC(name='auc'), Recall(name='recall'), Precision(name='precision')])		

		clf.fit(self.X, self.y, epochs=10, batch_size=128, verbose=0)

		return clf.predict_classes(X)
	
	def predict_proba(self, X):
		n_units = int(self.best['n_units'])
		n_layers = int(self.best['n_layers'])

		clf = Sequential()
		clf.add(Dense(16, input_shape=(11,), activation='relu'))
		for _ in range(n_layers-1):
			clf.add(Dense(16, activation='relu'))
		clf.add(Dense(1, activation='sigmoid'))

		clf.compile(optimizer='adam',
				loss='binary_crossentropy',
				metrics=['accuracy', AUC(name='auc'), Recall(name='recall'), Precision(name='precision')])		

		clf.fit(self.X, self.y, epochs=10, batch_size=128, verbose=0)
		return clf.predict_proba(X)



def train(projectName):
	print(projectName)
	
	with open('../Database/data_semi.pickle', 'rb') as f:
		df = pickle.load(f)
	
	df.columns = [x[0] for x in df.columns]
	df=df[df.projectname==projectName]

	num = int(len(df) * SUBSET_SIZE)
	models=[]
	for i in range(MODEL_NUMBER):
		print("training ",i+1,"th model")

		data=shuffle(df)

		x=data.iloc[:num,5:16]
		y=np.array(data['Label'][:num])

		print('imbalance:', sum(y) / len(y))

		_data = Data(x, None, y, None)
		transform = Transform('wfo')
		transform.apply(_data)
		_data.y_train = 1 - _data.y_train
		transform.apply(_data)
		_data.y_train = 1 - _data.y_train
		#transform.apply(_data)

		transform = Transform('smote')
		transform.apply(_data)

		x = _data.x_train
		y = _data.y_train

		#x=preprocessing.scale(x,axis=1,with_mean=True,with_std=True,copy=True)

		#clf=MLPClassifier(hidden_layer_sizes=(16,16,16), max_iter=200,)
		clf = Model()
		#clf=RandomForestClassifier(n_estimators=100)

		clf_sota = MLPClassifier(hidden_layer_sizes=(16,8,4), max_iter=200,)
		clf_sota.fit(x, y)

		clf.fit(x, y)
		preds = clf.predict(data.iloc[num:,5:16])
		y_test = data['Label'][num:]
		print(f'AUC: {metrics.roc_auc_score(y_test, preds)}')
		print(f'F1: {metrics.f1_score(y_test, preds)}')
		print(f'prec: {metrics.precision_score(y_test, preds)}')
		print(f'recall: {metrics.recall_score(y_test, preds)}')

		print('-' * 60)
		preds = clf_sota.predict(data.iloc[num:,5:16])
		print(f'AUC_sota: {metrics.roc_auc_score(y_test, preds)}')
		print(f'F1_sota: {metrics.f1_score(y_test, preds)}')
		print(f'prec_sota: {metrics.precision_score(y_test, preds)}')
		print(f'rec_sota: {metrics.recall_score(y_test, preds)}')
		

		models.append(clf)

	return models

def eval(tp,tn,fp,fn):
	print("tp : ",tp)
	print("tn : ",tn)
	print("fp : ",fp)
	print("fn : ",fn)
	if tp==0 or tn==0 or fp==0 or fn==0:
		return 1
	P=tp*1.0/(tp+fp)
	R=tp*1.0/(tp+fn)
	print("Precision : ",P)
	print("Recall : ",R)
	print("F1 : ",2*P*R/(P+R))

	a=tp+fp
	b=tp+fn
	c=tn+fp
	d=tn+fn
	print("MCC : ",(tp*tn-fp*fn)/((a*b*c*d)**0.5))
	
	return 2*P*R/(P+R)

def test(models,projectName):
	print(projectName)
	with open('../Database/data_semi.pickle', 'rb') as f:
		df = pickle.load(f)
	
	df.columns = [x[0] for x in df.columns]

	df=df[(df.projectname==projectName)]
	predicts=[]
	predicts_proba=[]
	for i in range(MODEL_NUMBER):
		clf=models[i]
		x=df.iloc[:,5:16]
		
		#x=preprocessing.scale(x,axis=1,with_mean=True,with_std=True,copy=True)

		predict=clf.predict(x)
		predict_proba=clf.predict_proba(x)

		predicts.append(predict)
		predicts_proba.append(predict_proba)
	result=[]
	for i in range(len(predicts[0])):
		total=0
		for j in range(MODEL_NUMBER):
			total=total+predicts[j][i]
		if total>=3:
			result.append(1)
		else:
			result.append(0)

	rp=[]
	for i in range(len(predicts_proba[0])):
		total=0
		for j in range(MODEL_NUMBER):
			total=total+predicts_proba[j][i][1]
		rp.append(total/MODEL_NUMBER)
		trp.append(total/MODEL_NUMBER)

	y=np.array(df['Label'])
	print('*'*80)
	print("AUC : ",metrics.roc_auc_score(y,rp))
	print('*'*80)
	tp,tn,fp,fn=0,0,0,0

	for i in range(len(y)):
		tr.append(y[i])
		if result[i]==y[i]:
			if result[i]==0:
				tn=tn+1
			else:
				tp=tp+1
		else:
			if result[i]==0:
				fn=fn+1
			else:
				fp=fp+1
	
	return tp,tn,fp,fn

def load_models(projectName):
	models=[]

	for i in range(MODEL_NUMBER):
		#clf=joblib.load('D:/Longmethod/model/4677/'+projectName+"_"+str(i)+'.joblib')
		clf=joblib.load("./Model/"+projectName+"_"+str(i)+".joblib")

		models.append(clf)

	return models


projects = ['areca-7.4.7','freeplane-1.3.12','jedit','junit-4.10','pmd-5.2.0','weka','android-backup-extractor-20140630','grinder-3.6','AoI30','jexcelapi_2_6_12']

ttp,ttn,tfp,tfn=0,0,0,0
for _ in range(20):
	for i in range(len(projects)):

		print("------------------------------------")
		#ss=time.time()
		models=train(projects[i])
		#print('#####################',time.time()-ss)
		#models=load_models(projects[i])
		#ss=time.time()
		#tp,tn,fp,fn=test(models,projects[i])
		#print(time.time()-ss)
		#ttp=ttp+tp
		#ttn=ttn+tn
		#tfp=tfp+fp
		#tfn=tfn+fn
		#eval(tp,tn,fp,fn)
	print("------------------------------------")

