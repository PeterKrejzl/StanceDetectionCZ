# -*- coding: utf-8 -*-
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
import codecs

filename = 'data_for_w2v/zeman+koureni+idnes.txt'
model_name = 'w2vmodel/w2v_300_min_2_win_7_iter_10.model'


def import_corpus(filename):
    data = []
    f = codecs.open(filename, encoding='utf-8')
    for line in f:
        data.append(line.lower())
    f.close()
    return data


def tokenize_data(text):
    text = [doc.lower().split(' ') for doc in text]
    return text


data = import_corpus(filename)
data = tokenize_data(data)


print('Training data for w2v size = %s' % len(data))
model = Word2Vec(size=300, min_count=2, window=7, workers=8, iter=10)
model.build_vocab(data)
model.train(data)
print('Model trained')

model.save(model_name)
