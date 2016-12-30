import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)



training_data_filename = 'data/koureni.txt'


stop_words = 'stopwords/czech_stop_words_out.txt'


cs_n = 'slovniky/cs_n_out.def'
cs_p = 'lovniky/cs_p_out.def'



idnes_comments = "idnes/noentity_out.txt"
pickle_tfidf = 'pickle/vocab_freqs.pickle'


