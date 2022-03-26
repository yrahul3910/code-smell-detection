from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Conv1D, MaxPooling1D, Masking, LSTM
from tensorflow.keras.metrics import AUC, Recall, Precision
from gensim.models.callbacks import CallbackAny2Vec
from raise_utils.data import Data
from raise_utils.transforms import Transform
from sklearn import metrics
from sklearn.utils import shuffle
from tensorflow import keras
import matplotlib.pyplot as plt
import preprocess
import pickle
import time
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"



class LossLogger(CallbackAny2Vec):
	'''Output loss at each epoch'''

	def __init__(self):
		self.epoch = 1
		self.losses = []

	def on_epoch_begin(self, model):
		print(f'Epoch: {self.epoch}', end='\t', flush=True)

	def on_epoch_end(self, model):
		loss = model.get_latest_training_loss()
		self.losses.append(loss)
		print(f'  Loss: {loss}')
		self.epoch += 1


class EpochSaver(CallbackAny2Vec):
	'''Callback to save model after each epoch.'''

	def __init__(self, path_prefix):
		self.path_prefix = path_prefix
		self.epoch = 0

	def on_epoch_end(self, model):
		output_path = './models/{}_epoch{}.model'.format(
			self.path_prefix, self.epoch)
		model.save(output_path)

		self.epoch += 1


total_y_pre=[]
total_y_test=[]

EMBEDDING_DIM=100
MAX_SEQUENCE_LENGTH = 50
MAX_JACCARD_LENGTH = 30
INC_BATCH_SIZE = 80000

W2V_MODEL_DIR = '../../../se-language-models/models/_epoch4.model'
TRAIN_SET_DIR = '../Database/'

tokenizer = preprocess.get_tokenizer()
all_word_index = tokenizer.word_index
embedding_matrix = preprocess.get_embedding_matrix(all_word_index, W2V_MODEL_DIR, dim=EMBEDDING_DIM)

MODEL_NUMBER=1
SUBSETSIZE=0.8

def getModels():
		models=[]
		for i in range(MODEL_NUMBER):
				method_a = Input(shape=(MAX_SEQUENCE_LENGTH,EMBEDDING_DIM),name='method_a')
				metric_a = Input(shape=(12,),name='metric_a')

				#masking_layer = Masking(mask_value=0,input_shape=(MAX_SEQUENCE_LENGTH,EMBEDDING_DIM))
				#lstm_share = LSTM(2,activation='sigmoid')
				#embedding_a = masking_layer(method_a)
				#lstm_a = lstm_share(embedding_a)
				#dense_share2 = Dense(12,activation='relu')
				#mtrdense_a= dense_share2(metric_a)

				flat = Flatten()(method_a)
				dense_method1 = Dense(5000, activation='relu')(flat)
				dense_method2 = Dense(5000, activation='relu')(dense_method1)
				dense_method3 = Dense(5000, activation='relu')(dense_method2)

				dense_metric1 = Dense(12, activation='relu')(metric_a)
				dense_metric2 = Dense(12, activation='relu')(dense_metric1)
				dense_metric3 = Dense(12, activation='relu')(dense_metric2)

				m_j_merged_a = keras.layers.concatenate([dense_method3, dense_metric3], axis=-1)

				dense1_a = Dense(4, activation='relu')(m_j_merged_a)
				total_output = Dense(1,activation='sigmoid',name='output')(dense1_a)

				model = Model(inputs=[method_a,metric_a],outputs=total_output)
				model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['binary_accuracy', AUC(name='auc'), Recall(name='recall'), Precision(name='precision')])
				models.append(model)

		return models

def getData(projectname):
		(x_train, y_train), (x_test, y_test) = preprocess.get_xy(projectname,tokenizer=tokenizer, mn_maxlen=MAX_SEQUENCE_LENGTH,embedding_matrix=embedding_matrix)
		#x_test, y_test = preprocess.get_xy_test(projectname,tokenizer=tokenizer, maxlen=MAX_SEQUENCE_LENGTH,embedding_matrix=embedding_matrix)
		return x_train,y_train,x_test,y_test

def train(projectname,models,x_train,y_train,x_test,y_test):
		for k in range(MODEL_NUMBER):
				#mn0,mn1,metrics0,metrics1=[],[],[],[]
				#print("Training ",k,"th model...")
				#for i in range(y_train.shape[0]):
				#		 if y_train[i]==0:
				#				 mn0.append(x_train[0][i])
				#				 metrics0.append(x_train[1][i])
				#		 else:
				#				 mn1.append(x_train[0][i])
				#				 metrics1.append(x_train[1][i])

				#size=(int)(len(mn1)*SUBSETSIZE)
				#temp_set=[]
				#mn0=np.array(mn0)
				#mn1=np.array(mn1)
				#metrics0=np.array(metrics0)
				#metrics1=np.array(metrics1)

				#indices0=np.arange(mn0.shape[0])
				#indices1=np.arange(mn1.shape[0])

				#np.random.shuffle(indices0)
				#np.random.shuffle(indices1)
				#indices0=shuffle(indices0)
				#indices1=shuffle(indices1)

				#mn0=mn0[indices0[:size]]
				#mn1=mn1[indices1[:size]]
				#metrics0=metrics0[indices0[:size]]
				#metrics1=metrics1[indices1[:size]]
				#temp_set=[]
				#for i in range(size):
				#		 temp_set.append([mn0[i],metrics0[i],0])
				#		 temp_set.append([mn1[i],metrics1[i],1])

				#np.random.shuffle(temp_set)
				
				y=y_train
				mn=x_train[0]
				metrics=x_train[1]
				#for i in range(len(temp_set)):
				#		 mn.append(temp_set[i][0])
				#		 metrics.append(temp_set[i][1])
				#		 y.append(temp_set[i][2])

				mn=np.array(mn)
				mn = mn.reshape((mn.shape[0], mn.shape[1] * mn.shape[2]))
				metrics=np.array(metrics)

				x=[mn,metrics]
				y=np.array(y)

				print(mn.shape, metrics.shape, y.shape)

				transform = Transform('wfo')

				data1 = Data(mn, None, y, None)
				data2 = Data(metrics, None, y, None)

				 
				# Apply fuzzy sampling; now, 0 is minority class
				transform.apply(data1)
				transform.apply(data2)

				# Make 1 the minority class again
				#data1.y_train = 1 - data1.y_train
				#data2.y_train = 1 - data2.y_train

				# Fuzzy sample and smote
				#transform = Transform('wfo')
				transform.apply(data1)
				transform.apply(data2)

				# Correct the labels
				#data1.y_train = 1 - data1.y_train
				#data2.y_train = 1 - data2.y_train

				transform = Transform('smote')
				transform.apply(data1)
				transform.apply(data2)


				x = [data1.x_train.reshape(data1.x_train.shape[0], 50, 100), data2.x_train]
				y = data1.y_train

				models[k].fit(x, y, epochs=3, validation_data=(x_test,y_test),batch_size=128,verbose=1)
				
				print('Performance:', models[k].evaluate(x_test, y_test))
		return models

def test(models,x_test,y_test, projectname):
		#mn0,mn1,metrics0,metrics1=[],[],[],[]
		#for i in range(y_test.shape[0]):
		#		 if y_test[i]==0:
		#				 mn0.append(x_test[0][i])
		#				 metrics0.append(x_test[1][i])
		#		 else:
		#				 mn1.append(x_test[0][i])
		#				 metrics1.append(x_test[1][i])

		#size=(int)(len(mn0)*.7)

		#temp_set=[]
		#mn0=np.array(mn0)
		#mn1=np.array(mn1)
		#metrics0=np.array(metrics0)
		#metrics1=np.array(metrics1)

		#indices0=np.arange(mn0.shape[0])
	   # indices1=np.arange(mn1.shape[0])

		#np.random.shuffle(indices0)
		#np.random.shuffle(indices1)


		#mn1=mn1[indices1[:size]]
		#metrics1=metrics1[indices1[:size]]

		#temp_set=[]
		#for i in range(mn0.shape[0]):
		#		 temp_set.append([mn0[i],metrics0[i],0])
		#for i in range(len(mn1)):
		#		 temp_set.append([mn1[i],metrics1[i],1])

		#np.random.shuffle(temp_set)
		#print('---',len(temp_set),'---')
		#y=[]
		#mn=[]
		#metrics=[]
		#for i in range(len(temp_set)):
		#		 mn.append(temp_set[i][0])
		#		 metrics.append(temp_set[i][1])
		#		 y.append(temp_set[i][2])

		#mn=np.array(mn)
		#metrics=np.array(metrics)

		print(y_test.shape, x_train[0].shape)
		print(models[0].predict(x_train).shape)
		return eval((models[0].predict(x_train) > 0.5).astype('int32'), y_test)

		#predict=[]
		#for i in range(MODEL_NUMBER):
		#		 predict.append(models[i].predict(x))
		#y_pre=[]
		#for i in range(y.shape[0]):
		#		 t=0.0
		#		 for j in range(MODEL_NUMBER):
		#				 t+=predict[j][i]
		#		 y_pre.append(t/MODEL_NUMBER)
		#return eval(y_pre,y)


projects=['android-backup-extractor-20140630',"AoI30","areca-7.4.7","freeplane-1.3.12","grinder-3.6","jedit","jexcelapi_2_6_12","junit-4.10","pmd-5.2.0","weka"]


for _ in range(20):
	print(f'Run #{_}:')
	for i in range(len(projects)):
		print("Build Models")
		models=getModels()
		print("Get Data")
		x_train,y_train,x_test,y_test=getData(projects[i])
		print(x_test[0].shape, y_test.shape)
		print("Start Training")
		models=train(projects[i],models,x_train,y_train,x_test,y_test)
		print('*'*80)
		print(projects[i])
		#f1=test(models,x_test,y_test, projects[i])



print('*'*80)
print("Final")
eval(total_y_pre,total_y_test)


print("Done")
