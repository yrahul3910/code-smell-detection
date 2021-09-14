# -*- coding: utf-8 -*-
import sys
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.word2vec import PathLineSentences
from keras.models import model_from_json
from gensim.models import word2vec
import random
from sklearn.utils import shuffle
from keras.models import Model
from keras.layers import merge
from keras.models import Sequential
from keras.layers import Conv1D, Embedding
from keras.layers import Dense, Flatten
from keras.utils.np_utils import to_categorical
from keras.metrics import AUC, Precision, Recall 
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from raise_utils.transforms import Transform
from raise_utils.data import Data
import pickle
import numpy as np
import time
import os
np.random.seed(1337)



class TunedModel():
    def __init__(self, X, y, embedding_layer, data_name, ratio, min_class):
        self.n_dense = 3
        self.best_auc = 0.
        self.best_history = None

        self.X = X
        self.y = np.argmax(y, axis=1).squeeze()

        with open(data_name + '_test.pkl', 'rb') as f:
            d = pickle.load(f)
            X_names = d['X_names']
            X_dis = d['X_dis']
            y1 = d['y']

        cur_X_names = self.X[0]
        cur_X_names = np.append(cur_X_names, X_names)
        cur_X_dis = self.X[1]
        cur_X_dis = np.append(cur_X_dis, X_dis)

        self.X = [cur_X_names, cur_X_dis]
        self.y = np.append(self.y, y1)

        X_names_train, X_names_test, X_dis_train, X_dis_test, y_train, y_test = train_test_split(X[0], X[1], y)

        np.random.seed(42)
        random.seed(42)
        transform = Transform('wfo')
        y_train = y_train.argmax(axis=1)
        y2_train = y_train.copy()
        y2_test = y_test.copy()

        X_dis_train = X_dis_train.squeeze()
        data1 = Data(X_names_train, X_names_test, y_train, y_test)
        data2 = Data(X_dis_train, X_dis_test, y2_train, y2_test)

        # Apply fuzzy sampling; now, 0 is minority class
        transform.apply(data1)
        transform.apply(data2)

        # Make 1 the minority class again
        data1.y_train = 1 - data1.y_train
        data2.y_train = 1 - data2.y_train

        # Fuzzy sample and smote
        transform = Transform('wfo')
        transform.apply(data1)
        transform.apply(data2)
        
        # Correct the labels
        data1.y_train = 1 - data1.y_train
        data2.y_train = 1 - data2.y_train

        transform = Transform('smote')
        transform.apply(data1)
        transform.apply(data2)

        self.X = [data1.x_train, data2.x_train]
        self.y = to_categorical(data1.y_train)

        self.X_val = [data1.x_test, data2.x_test]
        self.y_val = data1.y_test.squeeze()

        self.embedding_layer = embedding_layer
        self.data_name = data_name
        self.ratio = ratio
        self.min_class = min_class
    
    def fit(self):
        for _ in range(3, 4):
            print('-' * 50)
            print(self.data_name, '- Trying', _, 'layers')
            print('-' * 50)
            self.n_dense = _
            cur_auc, history = self.build_and_train()
            if cur_auc > self.best_auc:
                self.best_auc = cur_auc
                self.best_history = history

        print('Best AUC:', self.best_auc)
        print('History:', self.best_history.history)
        

    def build_and_train(self):
        model_left = Sequential()
        model_left.add(self.embedding_layer)
        model_left.add(Flatten())
        model_left.add(Dense(128, name='left-dense-1',
                             activation='relu', input_shape=(15,)))

        for i in range(self.n_dense - 1):
            model_left.add(Dense(128, name=f'left-dense-{2+i}', activation='relu'))

        model_right = Sequential()
        model_right.add(
            Dense(128, name='right-dense-1', activation='relu', input_shape=(2,)))

        for i in range(self.n_dense - 1):
            model_right.add(
                Dense(128, name=f'dense-right-{2+i}', activation='relu'))

        output = merge.Concatenate()(
            [model_left.output, model_right.output])

        output = Dense(128, activation='relu')(output)
        output = Dense(2, activation='sigmoid')(output)

        input_left = model_left.input
        input_right = model_right.input

        model = Model([input_left, input_right], output)

        model.compile(loss='binary_crossentropy',
                      optimizer='Adam', metrics=['accuracy', AUC(name='auc'), Recall(name='rec'), Precision(name='prec')])

        history = model.fit(self.X, self.y, epochs=3, validation_data=(self.X_val, self.y_val), verbose=1, batch_size=256)
        json_string = model.to_json()
        open("./Models/"+self.data_name+"_0.json",'w').write(json_string)
        model.save_weights("./Models/"+self.data_name+'_0.h5')
        return history.history['val_auc'][-1], history



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


MAX_SEQUENCE_LENGTH = 15


def nfromm(m, n, unique=True):
    """
    ä»[0, m)ä¸­äº§çnä¸ªéæºæ°
    :param m:
    :param n:
    :param unique:
    :return:
    """
    if unique:
        box = [i for i in range(m)]
        out = []
        for i in range(n):
            index = random.randint(0, m - i - 1)

            # å°éä¸­çåç´ æå¥çè¾åºç»æåè¡¨ä¸­
            out.append(box[index])

            # åç´ äº¤æ¢ï¼å°éä¸­çåç´ æ¢å°æåï¼ç¶åå¨åé¢çåç´ ä¸­ç»§ç»­è¿è¡éæºéæ©ã
            box[index], box[m - i - 1] = box[m - i - 1], box[index]
        return out
    else:
        # åè®¸éå¤
        out = []
        for _ in range(n):
            out.append(random.randint(0, m - 1))
        return out


def getsubset(x, y):
    t1 = [[], []]
    t2 = [[], []]
    t3 = [[], []]

    for i in range(len(y)):
        if y[i][0] == 1:
            t1[0].append(x[0][i])
            t2[0].append(x[1][i])
            t3[0].append(y[i])
        else:
            t1[1].append(x[0][i])
            t2[1].append(x[1][i])
            t3[1].append(y[i])

    num = (int)(len(t1[1]))

    index = nfromm(len(t1[1]), num)

    tt1 = []
    tt2 = []
    tt3 = []

    for i in range(len(index)):
        tt1.append(t1[1][index[i]])
        tt2.append(t2[1][index[i]])
        tt3.append(t3[1][index[i]])

    index = nfromm(len(t1[0]), num)

    for i in range(len(index)):
        tt1.append(t1[0][index[i]])
        tt2.append(t2[0][index[i]])
        tt3.append(t3[0][index[i]])

    tt3 = np.array(tt3)
    tt2 = np.array(tt2)
    t = [np.array(tt1), tt2.reshape(tt2.shape[:-1]).astype(float)]
    print('Input shapes are', t[0].shape, 'and', t[1].shape)

    return t, tt3


projects = ['android-backup-extractor-20140630', "AoI30", "areca-7.4.7", "freeplane-1.3.12",
            "grinder-3.6", "jedit", "jexcelapi_2_6_12", "junit-4.10", "pmd-5.2.0", "weka"]
# projects=["areca-7.4.7","freeplane-1.3.12","jedit","jexcelapi_2_6_12","junit-4.10","pmd-5.2.0","weka"]
basepath = "../"
ff1 = 0.0

model_num = 1


def train():  # select optional model
    for kk in range(len(projects)):  # len(projects)
        ss = time.time()

        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print(projects[kk])
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

        distances = []  # TrainSet
        labels = []  # 0/1
        texts = []  # ClassNameAndMethodName
        MAX_SEQUENCE_LENGTH = 15
        EMBEDDING_DIM = 100  # Dimension of word vector

        embedding_model = word2vec.Word2Vec.load(
            "../../../se-language-models/models/_epoch4.model")

        with open(basepath+"Data/"+projects[kk]+"/train_Distances.txt", 'r') as file_to_read:
            # with open("D:/data/7#Fold/train-weka"+"/train_distances.txt",'r') as file_to_read:
            for line in file_to_read.readlines():
                values = line.split()
                distance = values[:2]
                distances.append(distance)
                label = values[2:]
                labels.append(label)

        with open(basepath+"Data/"+projects[kk]+"/train_Names.txt", 'r') as file_to_read:
            # with open("D:/data/7#Fold/train-weka"+"/train_names.txt",'r') as file_to_read:
            for line in file_to_read.readlines():
                texts.append(line)

        print('Found %s train_distances.' % len(distances))

        tokenizer = Tokenizer(num_words=None)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        word_index = tokenizer.word_index
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        distances = np.asarray(distances)
        #labels = to_categorical(np.asarray(labels))
        labels = np.array(labels, dtype=int)

        print('Shape of train_data tensor:', data.shape)
        print('Shape of train_label tensor:', labels.shape)

        x_train = []
        x_train_names = data
        x_train_dis = distances
        x_train_dis = np.expand_dims(x_train_dis, axis=2)

        x_train.append(x_train_names)
        x_train.append(np.array(x_train_dis))
        y_train = np.array(labels)

        # Perform oversampling
        if sum(y_train) / len(y_train) < 0.5:
            min_class = 1
        else:
            min_class = 0

        over_idx = np.where(y_train == min_class)[0]
        ratio = np.squeeze(len(y_train) / sum(y_train) - 1)
        print('Oversampling ratio is', ratio)

        #x_train_dis = x_train_dis.tolist()
        #x_train_names = x_train_names.tolist()
        #y_train = y_train.tolist()

        #for i in range(int(ratio)):
        #    for idx in over_idx:
        #        x_train_dis.append(x_train_dis[idx])
        #        x_train_names.append(x_train_names[idx])
        #        y_train.append(y_train[idx])

        x_train_dis = np.asarray(x_train_dis)
        x_train_dis = np.expand_dims(x_train_dis, axis=2)
        x_train_names = np.asarray(x_train_names)
        x_train = [x_train_names, x_train_dis]
        y_train = to_categorical(y_train)

        for index in range(model_num):
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print(projects[kk], '---', index+1)
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("start time: "+time.strftime("%Y/%m/%d  %H:%M:%S"))

            x_train, y_train = getsubset(x_train, y_train)

            nb_words = len(word_index)
            embedding_matrix = np.zeros((nb_words+1, EMBEDDING_DIM))
            for word, i in word_index.items():
                if word not in embedding_model.wv:
                    continue
                embedding_vector = embedding_model.wv[word]
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

            embedding_layer = Embedding(nb_words + 1,
                                        EMBEDDING_DIM,
                                        input_length=MAX_SEQUENCE_LENGTH,
                                        weights=[embedding_matrix],
                                        trainable=False)

            print('Training model.')

            model = TunedModel(x_train, y_train, embedding_layer, projects[kk], ratio, min_class)
            model.fit()
            #json_string = model.to_json()
            #open("./Models/"+projects[kk]+"_"+str(index)+".json",'w').write(json_string)
            #model.save_weights("./Models/"+projects[kk]+"_"+str(index)+'.h5')
        print('########################', time.time()-ss)

for i in range(20):
    train()
