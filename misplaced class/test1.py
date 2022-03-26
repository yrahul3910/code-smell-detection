import numpy as np
import pandas as pd
import preprocessing
from gensim import models
from gensim.models import word2vec
from gensim.models.callbacks import CallbackAny2Vec
import time


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
        output_path = './models/{}_epoch{}.model'.format(self.path_prefix, self.epoch)
        model.save(output_path)

        self.epoch += 1


projects=['android-backup-extractor-20140630',"AoI30","areca-7.4.7","freeplane-1.3.12","grinder-3.6","jedit","jexcelapi_2_6_12","junit-4.10","pmd-5.2.0","weka"]
df=pd.read_csv('./input.csv')
df_classes=pd.read_csv('./test_class.csv')
df_items=pd.read_csv('./test_items.csv')

print('Building Tokenizer...')
tokenizer=preprocessing.get_tokenizer(df)

score=np.zeros(shape=(0))
label=np.zeros(shape=(0))
target_correct=0


for _ in range(20):
	for i in range(len(projects)):#len(projects)
		project=projects[i]
		print('*'*80)
		print(project)
		ss=time.time()
		models=preprocessing.train(df[df.projectname==project],tokenizer,project)
		print('###########################',time.time()-ss)

