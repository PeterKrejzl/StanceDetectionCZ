import logging
import pandas as pd
import numpy as np
import sys
import os
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle

import Task6_raw_processing
import Task6_analysis
import Task6_file_links


K_FOLD_VALIDATION_K = 10
MAX_FEATURES_BASIC = 1000



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)





def generate_features_basic(train_data=[], test_data=[], max_features_basic=5000, ngram_range_basic=(1, 2),
                            idnes_comments=True, multiple_vocabs=None):
    logging.log(logging.INFO, '\tpreparing basic set of features')

    vectorizer = TfidfVectorizer(analyzer='word', tokenizer=None, preprocessor=None, stop_words=None,
                                 max_features=max_features_basic, ngram_range=ngram_range_basic)
    vectorizer2 = CountVectorizer(vocabulary=tfidf_vocab)

    if multiple_vocabs == None:
        train_data_features = vectorizer.fit_transform(train_data)
        train_data_features = train_data_features.toarray()
        vocab = vectorizer.get_feature_names()

        if len(test_data) > 0:
            test_data_features = vectorizer.transform(test_data)
            test_data_features = test_data_features.toarray()

        return (train_data_features, test_data_features, vocab)

    if multiple_vocabs == 'idnes_only':
        train_data_features = vectorizer2.fit_transform(train_data).toarray()

        if len(test_data) > 0:
            test_data_features = vectorizer2.fit_transform(test_data).toarray()

        return (train_data_features, test_data_features, tfidf_vocab)

    if multiple_vocabs == 'both':
        train_data_features = vectorizer.fit_transform(train_data).toarray()
        train_data_features2 = vectorizer2.fit_transform(train_data).toarray()

        train_data_features = np.hstack((train_data_features, train_data_features2))

        if len(test_data) > 0:
            test_data_features = vectorizer.transform(test_data).toarray()
            test_data_features2 = vectorizer2.fit_transform(test_data).toarray()

            test_data_features = np.hstack((test_data_features, test_data_features2))

        return (train_data_features, test_data_features, tfidf_vocab)


'''********************************************************************************************************************

                                                MAIN FUNCTION

*********************************************************************************************************************'''


def main():

    system_parameters = {
        'RunMode': 'k_fold',  # training_only, k_fold, trump

        # 'Topic' :                           'climate', #abortion, atheism, climate, feminism, hillary
        'Topic': 'zeman',  # 'koureni'

        'GenerateStatistics': False,
        'GenerateStatisticsCZ': False,

        'Features_hashtags': False,
        'Features_initial_unigrams': True,
        'Features_initial_bigrams': False,
        'Features_initial_trigrams': False,
        'Features_tweet_sizes': True,
        'Features_tweet_sentiment': False,  # do not use
        'Features_tweet_POS': False,
        'Features_tweet_POS_percentage': False,
        'GI_categories': False,
    }



    '''
    system_parameters = {
        'AdditionalData': 'ignore',  # ignore, load

        'Topics': 'all_together',  # all_together, separate
        'GenerateOutputFile': False,  # True, False

        'UseOverrides': False,

        'Features_SentiWord': False,  # TODO: features senti word
        'Features_capitals_percentage': False,  # TODO : features capitals percentage
        'Features_heurestics': False,  # TODO: features heuristics
        'Features_bible_reference': False,  # READY
        'Features_dsd': False,  # !!!!! need to change dictionary,
        'Features_JRC': False,
        'Features_dsd_additional': False,  # dictionary created from additionaly downloaded data
        'Features_pos_neg_dict': True,
        'Features_w2v': False
    }
    '''




    if system_parameters['GenerateStatistics']:
        Task6_analysis.analyze_training_data(Task6_file_links.training_data_filename)

    if system_parameters['GenerateStatisticsCZ']:
        Task6_analysis.analyze_training_data_cz(Task6_file_links.training_data_filename)





    raw_data = Task6_raw_processing.load_from_file(Task6_file_links.training_data_filename)

    _, tail = os.path.split(Task6_file_links.training_data_filename)

    logging.log(logging.INFO, 'loading training data from ' + tail)
    logging.log(logging.INFO, 'Basic stats = ' + str(Task6_analysis.data_stats(raw_data)))




    if system_parameters['RunMode'] == 'k_fold':
        logging.log(logging.INFO, 'running k-fold validation (' + str(K_FOLD_VALIDATION_K) + ' folds)')
        F1s = []
        F1bs = []
        for i in range(K_FOLD_VALIDATION_K):
            logging.log(logging.INFO, '\t' + str(i + 1) + '. iteration')

            index = len(raw_data) / 100.0 * 10.0
            test_data_raw = raw_data[int(i * index): int((i * index) + index - 1)]

            if i > 0:
                train_data_raw = pd.concat([raw_data[:int((i * index)) - 1], raw_data[int((i * index) + index):]],
                                           axis=0)
            else:
                train_data_raw = raw_data[int((i * index) + index):]

            # test and train using same set
            if K_FOLD_VALIDATION_K == 1:
                train_data_raw = raw_data
                test_data_raw = raw_data


            cleaned_rows_training = Task6_raw_processing.preprocess_raw_data(train_data_raw)
            cleaned_rows_testing = Task6_raw_processing.preprocess_raw_data(test_data_raw)

            train_data_features, test_data_features, vocabulary = generate_features_basic(cleaned_rows_training, \
                                                                                          cleaned_rows_testing, \
                                                                                          max_features_basic=MAX_FEATURES_BASIC,\
                                                                                          ngram_range_basic=(1, 1))



if __name__ == "__main__":
    main()
