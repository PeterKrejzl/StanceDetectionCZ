# -*- coding: utf-8 -*-
import multiprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


import os
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout
from keras.utils.np_utils import to_categorical

import codecs


np.random.seed(1337)


# set parameters:
vocab_dim = 300
maxlen = 100
n_iterations = 10
n_exposures = 2
window_size = 7
batch_size = 512
n_epoch = 50
input_length = 100
cpu_count = 16


K_FOLD_VALIDATION_K = 3


idnes_corpus = 'data/noentity.txt'
'''
je správné lít rum do kafe, nebo kafe do rumu...
Ani jedno
dost fašizující názor
Rum nepiji a kafe také ne.
jasně, samovar a vodka z vás je cítit...
Ano, takhle mluví pražská natokavárna!
Rum do žaludku a kafe na hnůj!
...
'''


training_data_filename = 'data/koureni.txt'
'''
800	ZÁKAZ KOUŘENÍ V RESTAURACÍCH	Už aby to konečně prošlo!  Konečně se vrátíme mezi civilizované státy, které zákaz kouření mají už dávno schválený a nezhroutili se z toho. U nás menšina = kuřáci radí většině = nekuřáci, kam mají a nemají do restaurace chodit. Poněkud postavené na hlavu, jeliž pravda.	PRO
801	ZÁKAZ KOUŘENÍ V RESTAURACÍCH	"K čemu všemu nás tady to přiblblé vedení státu (říká si to ""vláda"") ještě donutí?"	PROTI
802	ZÁKAZ KOUŘENÍ V RESTAURACÍCH	Vcera kuraci nedostali v restauraci popelnik se slovy ze uz si maji pomalu zvykat a jit si zapalit na cerstvy vzduch  ;-D   R^   ani jeden z nich neodesel domu ci do vedlejsi restaurace  8-o	PRO
806	ZÁKAZ KOUŘENÍ V RESTAURACÍCH	Máte dojem, že si v ČR mafie vydělají málo?	NIC
...
'''


'''
need to convert training data to 6 different sets (trainin - testing - neu - for - against)
'''


model_name = 'w2vmodel/w2v.model'


training_data = pd.read_csv(training_data_filename, delimiter='\t', header=None, names=['ID', 'TOPIC', 'TEXT', 'STANCE'])
training_data = training_data.drop(labels={'TOPIC', 'ID'}, axis=1)


global_f1s_official = []
global_f1s_extended = []


def tokenize_data(text):
    text = [doc.lower().split(' ') for doc in text]
    return text

def import_idnes_corpus(filename):
    data = []
    f = codecs.open(filename, encoding='utf-8')
    for line in f:
        data.append(line.lower())
    f.close()
    return data[0:5]


def create_dictionaries(train, test, model):
    print('Creating dictionaries')
    gensim_dict = Dictionary()
    print(type(model))
    print(type(gensim_dict))
    gensim_dict.doc2bow(model.vocab.keys(), allow_update=True)
    w2indx = {v: k+1 for k, v in gensim_dict.items()}
    w2vec = {word: model[word] for word in w2indx.keys()}

    def parse_dataset(data):
        for key in data.keys():
            txt = data[key].lower().replace('\n','').split()
            new_txt = []
            for word in txt:
                try:
                    new_txt.append(w2indx[word])
                except:
                    new_txt.append(0)
            data[key] = new_txt
        return data
    train = parse_dataset(train)
    test = parse_dataset(test)
    return w2indx, w2vec, train, test


def calculate_F1 (predictions, y_test):
    '''
        PRO: 0+         y = 0   [1, 0, 0]
        PROTI: 10k+     y = 1   [0, 1, 0]
        NIC: 20k+       y = 2   [0, 0, 1]
    '''
    merged = zip(predictions, y_test)
    accuracy_oks = 0
    total_train_favs = 0
    total_train_ags = 0
    total_train_nones = 0
    total_test_favs = 0
    total_test_ags = 0
    total_test_nones = 0
    tp_favs = 0
    tp_ags = 0
    tp_nones = 0

    accuracy_oks = 0
    accuracy_total_records = len(merged)

    for row in merged:
        if np.argmax(row[0]) == np.argmax(row[1]):
            accuracy_oks +=1

        if row[1][0] == 1.:
            total_train_favs += 1
            if np.argmax(row[0]) == 0:
                tp_favs += 1
        elif row[1][1] == 1.:
            total_train_ags += 1
            if np.argmax(row[0]) == 1:
                tp_ags += 1
        elif row[1][2] == 1.:
            total_train_nones += 1
            if np.argmax(row[0]) == 2:
                tp_nones += 1

        #print('Row = %s, argmax = %s' % (row[0], np.argmax(row[0])))
        if np.argmax(row[0]) == 0:
            total_test_favs += 1
        elif np.argmax(row[0]) == 1:
            total_test_ags += 1
        elif np.argmax(row[0]) == 2:
            total_test_nones += 1

        if total_test_favs > 0:
            precision_favs = tp_favs / float(total_test_favs)
        else:
            precision_favs = 0

        if total_test_ags > 0:
            precision_ags = tp_ags / float(total_test_ags)
        else:
            precision_ags = 0

        if total_test_nones > 0:
            precision_nones = tp_nones / float(total_test_nones)
        else:
            precision_nones = 0

        if total_train_favs > 0:
            recall_favs = tp_favs / float(total_train_favs)
        else:
            recall_favs = 0

        if total_train_ags > 0:
            recall_ags = tp_ags / float(total_train_ags)
        else:
            recall_ags = 0

        if total_train_nones > 0:
            recall_nones = tp_nones / float(total_train_nones)
        else:
            recall_nones = 0

    if (precision_favs + recall_favs) > 0:
        f1_favs = (2 * (precision_favs * recall_favs)) / float(precision_favs + recall_favs)
    else:
        f1_favs = 0

    if (precision_ags + recall_ags) > 0:
        f1_ags = (2 * (precision_ags * recall_ags)) / float(precision_ags + recall_ags)
    else:
        f1_ags = 0

    if (precision_nones + recall_nones) > 0:
        f1_nones = (2 * (precision_nones * recall_nones)) / float(precision_nones + recall_nones)
    else:
        f1_nones = 0

    macro_F1 = (f1_ags + f1_favs + f1_nones) / 3.0

    print('accuracy_oks = %s, total_train_favs = %s, total_train_ags = %s, total_train_nones = %s' \
          % (accuracy_oks, total_train_favs, total_train_ags, total_train_nones))
    print('total_test_favs = %s, total_test_ags = %s, total_test_nones = %s' % (
    total_test_favs, total_test_ags, total_test_nones))
    print('tp_favs = %s, tp_ags = %s, tp_nones = %s' % (tp_favs, tp_ags, tp_nones))

    print('Macro F1 = %s' % macro_F1)

    return macro_F1





def calculate_official_F1 (predictions, y_test):
    merged = zip(predictions, y_test)
    total_rows = 0


    accuracy_oks = 0

    total_train_favs = 0
    total_train_ags = 0
    total_train_nones = 0

    total_test_favs = 0
    total_test_ags = 0
    total_test_nones = 0

    tp_favs = 0
    tp_ags = 0
    tp_nones = 0



    for row in merged:
        #print(row)
        #print('Type1 = %s, type2 = %s, type3 = %s' % (type(row), type(row[0]), type(row[1])))
        total_rows += 1
        #in predictions find the index with the highest score
        if np.argmax(row[0]) == np.argmax(row[1]):
            #print('Match found')
            accuracy_oks +=1

        if row[1][0] == 1.:
            total_train_favs += 1
            if np.argmax(row[0]) == 0:
                tp_favs += 1
        elif row[1][1] == 1.:
            total_train_ags += 1
            if np.argmax(row[0]) == 1:
                tp_ags += 1
        elif row[1][2] == 1.:
            total_train_nones += 1
            if np.argmax(row[0]) == 2:
                tp_nones += 1

        #print('Row = %s, argmax = %s' % (row[0], np.argmax(row[0])))
        if np.argmax(row[0]) == 0:
            total_test_favs += 1
        elif np.argmax(row[0]) == 1:
            total_test_ags += 1
        elif np.argmax(row[0]) == 2:
            total_test_nones += 1

        if total_test_favs > 0:
            precision_favs = tp_favs / float(total_test_favs)
        else:
            precision_favs = 0

        if total_test_ags > 0:
            precision_ags = tp_ags / float(total_test_ags)
        else:
            precision_ags = 0

        if total_test_nones > 0:
            precision_nones = tp_nones / float(total_test_nones)
        else:
            precision_nones = 0

        if total_train_favs > 0:
            recall_favs = tp_favs / float(total_train_favs)
        else:
            recall_favs = 0

        if total_train_ags > 0:
            recall_ags = tp_ags / float(total_train_ags)
        else:
            recall_ags = 0

        if total_train_nones > 0:
            recall_nones = tp_nones / float(total_train_nones)
        else:
            recall_nones = 0

        if (precision_favs + recall_favs) > 0:
            f1_favs = (2 * (precision_favs * recall_favs)) / float(precision_favs + recall_favs)
        else:
            f1_favs = 0

        if (precision_ags + recall_ags) > 0:
            f1_ags = (2 * (precision_ags * recall_ags)) / float(precision_ags + recall_ags)
        else:
            f1_ags = 0




    macro_F1 = (f1_ags + f1_favs) / 2.0


    print('accuracy_oks = %s, total_train_favs = %s, total_train_ags = %s, total_train_nones = %s' \
          % (accuracy_oks, total_train_favs, total_train_ags, total_train_nones))
    print('total_test_favs = %s, total_test_ags = %s, total_test_nones = %s' % (total_test_favs, total_test_ags, total_test_nones))
    print('tp_favs = %s, tp_ags = %s, tp_nones = %s' % (tp_favs, tp_ags, tp_nones))


    print('Macro F1 = %s' % macro_F1)

    return macro_F1





idnes_data = import_idnes_corpus(idnes_corpus)
data_for_w2v = list(training_data['TEXT'].drop_duplicates().values)
data_for_w2v = [w.decode('utf-8') for w in data_for_w2v]
data_for_w2v = tokenize_data(data_for_w2v)
data_for_w2v += idnes_data


print('Training data for w2v size = %s' % len(data_for_w2v))
model = Word2Vec(size=vocab_dim, min_count=n_exposures, window=window_size, workers=8, iter=n_iterations)
model.build_vocab(data_for_w2v)
model.train(data_for_w2v)
print('Model trained')

model.save(model_name)
print('Model saved')





kf = KFold(n_splits=K_FOLD_VALIDATION_K, )
for train_index, test_index in kf.split(training_data):
    X_train, X_test = training_data.ix[train_index], training_data.ix[test_index]

    train = {}
    test = {}

    for index, row in X_train.iterrows():
        #print('Index = %s, row = %s' % (index, row['STANCE']))
        text = row['TEXT'].lower().decode('utf-8')

        if row['STANCE'] == 'PRO':
            train[index] = text
        elif row['STANCE'] == 'PROTI':
            train[index + 10000] = text
        elif row['STANCE'] == 'NIC':
            train[index + 20000] = text



    for index, row in X_test.iterrows():
        text = row['TEXT'].lower().decode('utf-8')

        if row['STANCE'] == 'PRO':
            test[index] = text
        elif row['STANCE'] == 'PROTI':
            test[index + 10000] = text
        elif row['STANCE'] == 'NIC':
            test[index + 20000] = text

    #print('Train shape = %s, test shape = %s' % (len(train), len(test)))


    #combined = train.values() + test.values()
    #combined += idnes_data

    #print('Combined data lenght = %s' % len(combined))

    #combined = tokenize_data(combined)


    '''
    print('Training w2v')
    model = Word2Vec(size=vocab_dim, min_count=n_exposures, window=window_size, workers=8, iter=n_iterations)
    model.build_vocab(combined)
    model.train(combined)
    '''


    print('Transform data')
    index_dict, word_vectors, train, test = create_dictionaries(train, test, model)

    print('Keras embeddings')
    n_symbols = len(index_dict) + 1  # 0 is masking
    embedding_weights = np.zeros((n_symbols, vocab_dim))
    for word, index in index_dict.items():
        embedding_weights[index, :] = word_vectors[word]


    #print(embedding_weights)

    print('Creating dataset')
    y_train = []
    y_test = []

    X_training = train.values()
    X_testing = test.values()

    for key in train.keys():
        if key >= 10000 and key < 20000:
            y_train.append(1)
        elif key >= 20000:
            y_train.append(2)
        else:
            y_train.append(0)

    for key in test.keys():
        if key >= 10000 and key < 20000:
            y_test.append(1)
        elif key >= 20000:
            y_test.append(2)
        else:
            y_test.append(0)

    '''
        PRO: 0+         y = 0
        PROTI: 10k+     y = 1
        NIC: 20k+       y = 2
    '''

    print('Pad sequences (samples x time)')
    X_training = sequence.pad_sequences(X_training, maxlen=maxlen)
    X_testing = sequence.pad_sequences(X_testing, maxlen=maxlen)

    print('X_train shape:', X_training.shape)
    print('X_test shape:', X_testing.shape)

    print('Convert labels to numpy sets')
    y_train = np.array(y_train)
    y_test = np.array(y_test)


    y_train = to_categorical(y_train, nb_classes=3)
    y_test = to_categorical(y_test, nb_classes=3)




    print('Keras model')
    model_keras = Sequential()
    model_keras.add(Embedding(output_dim=vocab_dim, input_dim=n_symbols, mask_zero=True, weights=[embedding_weights],
                        input_length=input_length))
    model_keras.add(LSTM(vocab_dim))
    model_keras.add(Dropout(0.3))
    model_keras.add(Dense(500, activation='relu'))
    model_keras.add(Dropout(0.3))
    model_keras.add(Dense(3, activation='softmax'))
    model_keras.summary()

    print('Compiling the model')
    model_keras.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    print('Train')
    model_keras.fit(X_training, y_train, batch_size=batch_size, nb_epoch=2, validation_data=(X_testing, y_test), shuffle=True)

    print('Evaluate')
    score = model_keras.evaluate(X_testing, y_test, batch_size=batch_size)

    print('Test score = ', score[0])
    print('Test accuracy = ', score[1])

    predictions = model_keras.predict(X_testing, batch_size=batch_size)


    global_f1s_extended.append(calculate_F1(predictions, y_test))
    global_f1s_official.append(calculate_official_F1(predictions, y_test))




print('Official metric\n**************************')
print('Partial F1s = %s' % (global_f1s_official))
print('AVG F1 = %s, MAX F1 = %s, MIN F1 = %s' % (np.mean(global_f1s_official), max(global_f1s_official), min(global_f1s_official)))


print('Extended metric\n**************************')
print('Partial F1s = %s' % (global_f1s_extended))
print('AVG F1 = %s, MAX F1 = %s, MIN F1 = %s' % (np.mean(global_f1s_extended), max(global_f1s_extended), min(global_f1s_extended)))





quit()










'''

data_locations = {'data_zeman/NEU_test.txt': 'TEST_NEUTRAL',
                  'data_zeman/NEU_train.txt': 'TRAIN_NEUTRAL',
                  'data_zeman/PRO_test.txt': 'TEST_FAVOR',
                  'data_zeman/PRO_train.txt': 'TRAIN_FAVOR',
                  'data_zeman/PROTI_test.txt': 'TEST_AGAINST',
                  'data_zeman/PROTI_train.txt': 'TRAIN_AGAINST'
                  }



idnes_corpus = 'data/noentity.txt'



def import_idnes_corpus(filename):
    data = []
    f = codecs.open(filename, encoding='utf-8')
    for line in f:
        data.append(line.lower())
    f.close()
    return data



def import_idnes_corpus2 (filename):
    with open(filename) as fpath:
        data = fpath.readlines()
    return data




def import_data (datasets):
    train = {}
    test = {}
    for k, v in datasets.items():
        data = []
        fpath = codecs.open(k, encoding='utf-8')
        for line in fpath:
            data.append(line.lower())
        #with open(k) as fpath:
        #    data = fpath.readlines()
        for val, each_line in enumerate(data):
            each_line = each_line.strip()
            if v.endswith('AGAINST') and v.startswith('TRAIN'):
                train[val] = each_line
            elif v.endswith('AGAINST') and v.startswith('TEST'):
                test[val] = each_line
            elif v.endswith('FAVOR') and v.startswith('TRAIN'):
                train[10000 + val] = each_line
            elif v.endswith('FAVOR') and v.startswith('TEST'):
                test[10000 + val] = each_line
            elif v.endswith('NEUTRAL') and v.startswith('TRAIN'):
                train[20000 + val] = each_line
            elif v.endswith('NEUTRAL') and v.startswith('TEST'):
                test[20000 + val] = each_line
    return train, test




def tokenize_data(text):
    text = [doc.lower().split(' ') for doc in text]
    return text




def create_dictionaries(train, test, model):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.vocab.keys(), allow_update=True)
    w2indx = {v: k+1 for k, v in gensim_dict.items()}
    w2vec = {word: model[word] for word in w2indx.keys()}

    def parse_dataset(data):
        for key in data.keys():
            txt = data[key].lower().replace('\n','').split()
            new_txt = []
            for word in txt:
                try:
                    new_txt.append(w2indx[word])
                except:
                    new_txt.append(0)
            data[key] = new_txt
        return data
    train = parse_dataset(train)
    test = parse_dataset(test)
    return w2indx, w2vec, train, test




print('Loading data')
train, test = import_data(data_locations)
combined = train.values() + test.values()

#idnes_data = import_idnes_corpus(idnes_corpus)


#combined += idnes_data
print('Combined shape = %s' % len(combined))

print('Tokenizing')
combined = tokenize_data(combined)


print('Training w2v')
model = Word2Vec(size=vocab_dim, min_count=n_exposures, window=window_size, workers=8, iter=n_iterations)
model.build_vocab(combined)
model.train(combined)
#model.save('idnes_zeman_300_iter_10_window_7')





print('Transform data')
index_dict, word_vectors, train, test = create_dictionaries(train, test, model)


print('Keras embeddings')
n_symbols = len(index_dict) + 1 #0 is masking
embedding_weights = np.zeros((n_symbols, vocab_dim))
for word, index in index_dict.items():
    embedding_weights[index, :] = word_vectors[word]



print('Creating dataset')
'''
'''
AGAINST 0 - 9999
FAVOR 10000+
NEUTRAL 20000+
'''
'''
X_train = train.values()
#y_train = [1 if value > 10000  else 0 for value in train.keys()]
y_train = []
for key in train.keys():
    #print(key)
    if key >= 10000 and key < 20000:
        y_train.append(1)
    elif key >= 20000:
        y_train.append(2)
    else:
        y_train.append(0)


X_test = test.values()
y_test = []
for key in test.keys():
    #print(key)
    if key >= 10000 and key < 20000:
        y_test.append(1)
    elif key >= 20000:
        y_test.append(2)
    else:
        y_test.append(0)


print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


print('Convert labels to numpy sets')
y_train = np.array(y_train)
y_test = np.array(y_test)


from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train, nb_classes=3)
y_test = to_categorical(y_test, nb_classes=3)

#n_symbols += 1

print('Keras model')
model = Sequential()
model.add(Embedding(output_dim=vocab_dim, input_dim=n_symbols, mask_zero=True, weights=[embedding_weights], input_length=input_length))
model.add(LSTM(vocab_dim))
model.add(Dropout(0.3))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))
model.summary()

print('Compiling the model')
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

print('Train')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=5, validation_data=(X_test, y_test), shuffle=True)

print('Evaluate')
score = model.evaluate(X_test, y_test, batch_size=batch_size)

print('Test score = ', score[0])
print('Test accuracy = ', score[1])


predictions = model.predict(X_test, batch_size=batch_size)

print(predictions)
print(y_test)


'''