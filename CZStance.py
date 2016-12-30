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
                            tfidf_vocab=None, idnes_comments=True, multiple_vocabs=None):
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


def generate_all_features(train_data_features, test_data_features,
                          train_data_raw, test_data_raw,
                          cleaned_rows_training,
                          cleaned_rows_testing,
                          system_parameters):
    return [], []


'''
def generate_all_features(train_data_features, test_data_features,
                          train_data_raw, test_data_raw,
                          cleaned_rows_training,
                          cleaned_rows_testing,
                          system_parameters):



    if system_parameters['Features_hashtags']:
        logging.log(logging.INFO, '\t\tadding features hashtags')
        hashtags_training = Task6_raw_processing.preprocess_raw_data_hashtags(train_data_raw)
        hashtags_testing  = Task6_raw_processing.preprocess_raw_data_hashtags(test_data_raw)

        train_data_features, test_data_features = generate_features_hashtags(train_data_features, \
                                                                             test_data_features, \
                                                                             hashtags_training, \
                                                                             hashtags_testing, \
                                                                             max_features_hashtags = 50, \
                                                                             ngram_range_hashtags=(1,2)
                                                                                 )


    if system_parameters['Features_initial_unigrams']:
        logging.log(logging.INFO, '\t\tadding features initial unigrams')
        initial_unigrams_training = Task6_raw_processing.preprocess_raw_data_initial_unigrams(cleaned_rows_training)
        initial_unigrams_testing  = Task6_raw_processing.preprocess_raw_data_initial_unigrams(cleaned_rows_testing)

        train_data_features, test_data_features = generate_features_initial_unigrams(train_data_features,
                                                                                     test_data_features,
                                                                                     initial_unigrams_training,
                                                                                     initial_unigrams_testing
                                                                                    )
    if system_parameters['Features_initial_bigrams']:
        logging.log(logging.INFO, '\t\tadding initial bigrams')
        initial_bigrams_training = Task6_raw_processing.preprocess_raw_data_initial_bigrams(cleaned_rows_training)
        initial_bigrams_testing  = Task6_raw_processing.preprocess_raw_data_initial_bigrams(cleaned_rows_testing)

        train_data_features, test_data_features = generate_features_initial_bigrams(train_data_features,
                                                                                    test_data_features,
                                                                                    initial_bigrams_training,
                                                                                    initial_bigrams_testing
                                                                                    )
    if system_parameters['Features_initial_trigrams']:
        logging.log(logging.INFO, '\t\tadding initial trigrams')
        initial_trigrams_training = Task6_raw_processing.preprocess_raw_data_initial_trigrams(cleaned_rows_training)
        initial_trigrams_testing  = Task6_raw_processing.preprocess_raw_data_initial_trigrams(cleaned_rows_testing)

        train_data_features, test_data_features = generate_features_initial_trigrams(train_data_features,
                                                                                     test_data_features,
                                                                                     initial_trigrams_training,
                                                                                     initial_trigrams_testing
                                                                                     )
    if system_parameters['Features_tweet_sentiment']:
        logging.log(logging.INFO, '\t\tadding tweet sentiment')
        tweet_sentiment_training = Task6_raw_processing.preprocess_raw_data_sentiment(cleaned_rows_training)
        tweet_sentiment_testing  = Task6_raw_processing.preprocess_raw_data_sentiment(cleaned_rows_testing)

        train_data_features, test_data_features = generate_features_sentiment(train_data_features,
                                                                              test_data_features,
                                                                              tweet_sentiment_training,
                                                                              tweet_sentiment_testing
                                                                              )
    if system_parameters['Features_tweet_POS']:
        logging.log(logging.INFO, '\t\tadding tweet POS')
        tweet_pos_training = Task6_raw_processing.preprocess_raw_data_POS(cleaned_rows_training)
        tweet_pos_testing  = Task6_raw_processing.preprocess_raw_data_POS(cleaned_rows_testing)

        train_data_features, test_data_features = generate_features_pos(train_data_features,
                                                                        test_data_features,
                                                                        tweet_pos_training,
                                                                        tweet_pos_testing
                                                                    )

    if system_parameters['Features_tweet_POS_percentage']:
        logging.log(logging.INFO, '\t\tadding features tweet POS percentage')
        tweet_pos_percentage_training = Task6_raw_processing.preprocess_raw_data_POS_percentage(cleaned_rows_training)
        tweet_pos_percentage_testing = Task6_raw_processing.preprocess_raw_data_POS_percentage(cleaned_rows_testing)


        train_data_features, test_data_features = generate_features_pos_percentage(train_data_features,
                                                                                    test_data_features,
                                                                                    tweet_pos_percentage_training,
                                                                                    tweet_pos_percentage_testing
                                                                                    )




    if system_parameters['GI_categories']:
        logging.log(logging.INFO, '\t\tadding features GI categories')
        gi_cats_training = Task6_raw_processing.preprocess_raw_data_GI(cleaned_rows_training)
        gi_cats_testing = Task6_raw_processing.preprocess_raw_data_GI(cleaned_rows_testing)

        train_data_features, test_data_features = generate_features_gi(train_data_features,
                                                                       test_data_features,
                                                                       gi_cats_training,
                                                                       gi_cats_testing
                                                                    )

    if system_parameters['Features_SentiWord']:
        logging.log(logging.INFO, '\t\tadding features sentiword')
        sentiwords_training = Task6_raw_processing.preprocess_raw_data_sentiwords(cleaned_rows_training)
        sentiwords_testing = Task6_raw_processing.preprocess_raw_data_sentiwords(cleaned_rows_testing)

        train_data_features, test_data_features = generate_features_sentiwordnet(train_data_features,
                                                                                 test_data_features,
                                                                                 sentiwords_training,
                                                                                 sentiwords_testing)

    if system_parameters['Features_w2v']:
        logging.log(logging.INFO, '\t\tadding w2v features')
        w2v_training = Task6_raw_processing.get_w2v_features(cleaned_rows_training)
        w2v_testing = Task6_raw_processing.get_w2v_features(cleaned_rows_testing)

        train_data_features, test_data_features = generate_features_w2v (train_data_features,
                                                                         test_data_features,
                                                                         w2v_training,
                                                                         w2v_testing)

    if system_parameters['Features_bible_reference']:
        logging.log(logging.INFO, '\t\tadding features bible reference')
        bible_training = Task6_raw_processing.preprocess_raw_data_bible_reference(train_data_raw)
        bible_testing = Task6_raw_processing.preprocess_raw_data_bible_reference(test_data_raw)


        #pprint(train_data_features.shape)
        #pprint(test_data_features.shape)

        train_data_features, test_data_features = generate_features_bible_reference(train_data_features,
                                                                                    test_data_features,
                                                                                    bible_training,
                                                                                    bible_testing)

        #pprint(train_data_features.shape)
        #pprint(test_data_features.shape)
        #pprint(bible_training)
        #quit()
    if system_parameters['Features_pos_neg_dict']:
        logging.log(logging.INFO,'\t\tadding features based on cs_n and cs_p dictionaries')

        cs_n_p_training = Task6_raw_processing.preprocess_raw_data_cs_n_p(cleaned_rows_training)
        cs_n_p_testing = Task6_raw_processing.preprocess_raw_data_cs_n_p(cleaned_rows_testing)

        train_data_features, test_data_features = generate_features_cs_n_p(train_data_features,
                                                                           test_data_features,
                                                                           cs_n_p_training,
                                                                           cs_n_p_testing)


    if system_parameters['Features_JRC']:
        logging.log(logging.INFO, '\t\tadding features jrc dictionaries')
        jrc_training1 = Task6_raw_processing.preprocess_raw_data_jrc(cleaned_rows_training, Task6_file_links.jrc_p, Task6_file_links.jrc_hp)
        jrc_testing1 = Task6_raw_processing.preprocess_raw_data_jrc(cleaned_rows_testing, Task6_file_links.jrc_p, Task6_file_links.jrc_hp)

        jrc_training2 = Task6_raw_processing.preprocess_raw_data_jrc(cleaned_rows_training, Task6_file_links.jrc_n, Task6_file_links.jrc_hn)
        jrc_testing2 = Task6_raw_processing.preprocess_raw_data_jrc(cleaned_rows_testing, Task6_file_links.jrc_n, Task6_file_links.jrc_hn)



        train_data_features, test_data_features = generate_features_jrc(train_data_features,
                                                                        test_data_features,
                                                                        jrc_training1,
                                                                        jrc_testing1)
        train_data_features, test_data_features = generate_features_jrc(train_data_features,
                                                                        test_data_features,
                                                                        jrc_training2,
                                                                        jrc_testing2)




    if system_parameters['Features_dsd']:
        logging.log(logging.INFO, '\t\tadding features dsd')

        fav_dict = Task6_file_links.dsd_atheism_favor
        ags_dict = Task6_file_links.dsd_atheism_against

        if system_parameters['Topic'] == 'abortion':
            #fav_dict = Task6_file_links.dsd_abortion_favor
            #ags_dict = Task6_file_links.dsd_abortion_against
            fav_dict = Task6_file_links.dsd2_abortion_favor
            ags_dict = Task6_file_links.dsd2_abortion_against

        elif system_parameters['Topic'] == 'atheism':
            fav_dict = Task6_file_links.dsd_atheism_favor
            ags_dict = Task6_file_links.dsd_atheism_against
            #fav_dict = Task6_file_links.dsd2_atheism_favor
            #ags_dict = Task6_file_links.dsd2_atheism_against

        elif system_parameters['Topic'] == 'climate':
            #fav_dict = Task6_file_links.dsd_climate_favor
            #ags_dict = Task6_file_links.dsd_climate_against
            fav_dict = Task6_file_links.dsd2_climate_favor
            ags_dict = Task6_file_links.dsd2_climate_against

        elif system_parameters['Topic'] == 'feminism':
            #fav_dict = Task6_file_links.dsd_feminism_favor
            #ags_dict = Task6_file_links.dsd_feminism_against
            fav_dict = Task6_file_links.dsd2_feminism_favor
            ags_dict = Task6_file_links.dsd2_feminism_against
        elif system_parameters['Topic'] == 'hillary':
            #fav_dict = Task6_file_links.dsd_hillary_favor
            #ags_dict = Task6_file_links.dsd_hillary_against
            fav_dict = Task6_file_links.dsd2_hillary_favor
            ags_dict = Task6_file_links.dsd2_hillary_against

        if system_parameters['RunMode'] == 'trump':
            fav_dict = Task6_file_links.dsd2_trump_favor
            ags_dict = Task6_file_links.dsd2_trump_against

        logging.log(logging.INFO, '\t\tusing dictionary from ' + str(fav_dict))
        logging.log(logging.INFO, '\t\tusing dictionary from ' + str(ags_dict))



        dsd_training1 = Task6_raw_processing.preprocess_raw_data_dsd2(train_data_raw, fav_dict, ags_dict)
        dsd_testing1 = Task6_raw_processing.preprocess_raw_data_dsd2(test_data_raw, fav_dict, ags_dict)

        dsd_training2 = Task6_raw_processing.preprocess_raw_data_dsd(cleaned_rows_training, fav_dict, ags_dict)
        dsd_testing2  = Task6_raw_processing.preprocess_raw_data_dsd(cleaned_rows_testing,  fav_dict, ags_dict)



        #DSDs are analyzed from raw data as well as from cleaned rows
        dsd_training = []
        dsd_testing = []

        for i in range(len(dsd_training1)):
            dsd_training.append([dsd_training1[i][0] + dsd_training2[i][0], dsd_training1[i][1] + dsd_training2[i][1]])


        for i in range(len(dsd_testing1)):
            dsd_testing.append([dsd_testing1[i][0] + dsd_testing2[i][0], dsd_testing1[i][1] + dsd_testing2[i][1]])


        train_data_features, test_data_features = generate_features_dsd(train_data_features,
                                                                        test_data_features,
                                                                        dsd_training,
                                                                        dsd_testing)






    if system_parameters['Features_dsd_additional']:
        logging.log(logging.INFO, '\t\tadding features dsd additional dictionary')

        fav_dict_add = None
        ags_dict_add = None

        if system_parameters['Topic'] == 'abortion':
            #fav_dict_add = Task6_file_links.dsd_abortion_favor_additional
            #ags_dict_add = Task6_file_links.dsd_abortion_against_additional
            fav_dict_add = Task6_file_links.dsd2_abortion_favor_additional
            ags_dict_add = Task6_file_links.dsd2_abortion_against_additional
        elif system_parameters['Topic'] == 'hillary':
            #fav_dict_add = None
            #ags_dict_add = Task6_file_links.dsd_hillary_against_additional
            fav_dict_add = Task6_file_links.dsd2_hillary_favor_additional
            ags_dict_add = Task6_file_links.dsd2_hillary_against_additional
        else:
            fav_dict_add = None
            ags_dict_add = None



        logging.log(logging.INFO, '\t\tusing dictionary from ' + str(fav_dict_add))
        logging.log(logging.INFO, '\t\tusing dictionary from ' + str(ags_dict_add))

        dsd_additional_data_training = Task6_raw_processing.preprocess_raw_data_dsd_additional(cleaned_rows_training,
                                                                                               fav_dict_add,
                                                                                               ags_dict_add)

        dsd_additional_data_testing = Task6_raw_processing.preprocess_raw_data_dsd_additional(cleaned_rows_testing,
                                                                                              fav_dict_add,
                                                                                              ags_dict_add)


        train_data_features, test_data_features = generate_features_dsd(train_data_features,
                                                                        test_data_features,
                                                                        dsd_additional_data_training,
                                                                        dsd_additional_data_testing)




    if system_parameters['Features_tweet_sizes']:
        logging.log(logging.INFO, '\tadding tweet sizes features')
        tweet_sizes_training = Task6_raw_processing.preprocess_raw_data_tweet_sizes(cleaned_rows_training)
        tweet_sizes_testing = Task6_raw_processing.preprocess_raw_data_tweet_sizes(cleaned_rows_testing)


        train_data_features, test_data_features = generate_features_tweet_sizes(train_data_features,
                                                                                test_data_features,
                                                                                tweet_sizes_training,
                                                                                tweet_sizes_testing)





    return(train_data_features, test_data_features)




'''



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


            tfidf_freqs = pickle.load(open(Task6_file_links.pickle_tfidf, 'r'))

            train_data_features, test_data_features, vocabulary = generate_features_basic(train_data= cleaned_rows_training, \
                                                                                          test_data= cleaned_rows_testing, \
                                                                                          max_features_basic=MAX_FEATURES_BASIC,\
                                                                                          ngram_range_basic=(1, 1),\
                                                                                          idnes_comments=True,
                                                                                          multiple_vocabs=None,\
                                                                                          tfidf_vocab=tfidf_freqs)


            train_data_features, test_data_features = generate_all_features(train_data_features,\
                                                                            test_data_features,\
                                                                            train_data_raw, \
                                                                            test_data_raw, \
                                                                            cleaned_rows_training, \
                                                                            cleaned_rows_testing,\
                                                                            system_parameters)


            gnb = LogisticRegression(C=1.0, class_weight='balanced', n_jobs=-1)
            gnb.fit(train_data_features, train_data_raw['Stance'])
            result = gnb.predict(test_data_features)




if __name__ == "__main__":
    main()
