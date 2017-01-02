import logging
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
from pprint import pprint
import random
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import Task6_file_links
import string
from collections import Counter



excluded_tags = ('SemST', 'semst')

#trump only
#excluded_tags = ('SemST', 'semst', 'hilary', 'hillary', 'clinton')


USER_NAME = ' NAME '
stances = ('NONE', 'AGAINST', 'FAVOR', )
BAD_WORD = ' BAD_WORD '
LINK = ' URL '
IMG_LINK = ' IMGURL '
EXCLAMATION = ' MULTIPLEEXCLAMATIONS '
QMARK = ' MULTIPLEQUESTIONMARKS '
NUMBER = 'NUMBER'
BIBLE = ' BIBLE '



'''
    load DSDs
'''
def load_dsd(filename):
    raw = pd.read_csv(filename, header=None,quoting=3,delimiter='\t',names=['WORD'])
       
    return(raw)


def load_jrc(filename):
    raw = pd.read_csv(filename, header=None, quoting=3, names=['WORD'])
    return(raw)

#pprint(load_jrc(Task6_file_links.jrc_hn))
#pprint(load_jrc(Task6_file_links.jrc_hp))
#pprint(load_jrc(Task6_file_links.jrc_p))
#pprint(load_jrc(Task6_file_links.jrc_n))


def load_heuristics(filename):
    #logging.log(logging.INFO, '\tloading heuristic definitions')
    raw = pd.read_csv(filename, header=0,quoting=3,delimiter='\t')
    #pprint(raw.head(10))
    #quit()
    
    return(raw)


#load_heuristics(Task6_file_links.heuristics)



'''
    load sentiwordnet
'''
def load_sentiwordnet(filename):
    senti_words = {}
    
    logging.log(logging.INFO, '\tloading senti wordnet')
    raw = pd.read_csv(Task6_file_links.senti_wordnet, header=0, delimiter=',')
    
    for i in raw.index:
        pos_score = raw['PosScore'].ix[i]
        neg_score = raw['NegScore'].ix[i]
        synset_terms = raw['SynsetTerms'].ix[i]
        if pos_score > 0.5 or neg_score > 0.5:
            terms = synset_terms.split()
            terms_parsed = [w[0:w.index('#')].replace('_', ' ') for w in terms]
            
            for t in terms_parsed:
                senti_words[t] = [pos_score, neg_score]
        
    
    return(senti_words)





#r = load_sentiwordnet(Task6_file_links.senti_wordnet)
#pprint(len(r))
#quit()








'''
    GI load
    example: ABOLISH: ['Strong', 'Hostile', 'Negativ']
    
'''
def load_GI_data(filename):
    #raw2 = pd.read_csv(Task6_file_links.gi_input_file, header=0, delimiter="\t", quoting=3)
    raw2 = pd.read_csv(filename, header=0, delimiter="\t", quoting=3)
    gi = {}
    
    '''
        Entry    Source    Positiv    Negativ    Pstv    Affil    Ngtv    Hostile    Strong    Power    Weak    Submit    Active    Passive    Pleasur    
        Pain    Feel    Arousal    EMOT    Virtue    Vice    Ovrst    Undrst    Academ    Doctrin    Econ@    
        Exch    ECON    Exprsv    Legal    Milit    Polit@    POLIT    Relig    Role    COLL    Work    Ritual    SocRel    Race    Kin@    MALE    
        Female    Nonadlt    HU    ANI    PLACE    Social    Region    Route    Aquatic    Land    Sky    Object    Tool    Food    Vehicle    BldgPt    
        ComnObj    NatObj    BodyPt    ComForm    COM    Say    Need    Goal    Try    Means    Persist    Complet    Fail    NatrPro    Begin    Vary    
        Increas    Decreas    Finish    Stay    Rise    Exert    Fetch    Travel    Fall    Think    Know    Causal    Ought    Perceiv    Compare    
        Eval@    EVAL    Solve    Abs@    ABS    Quality    Quan    NUMB    ORD    CARD    FREQ    DIST    Time@    TIME    Space    POS    DIM    
        Rel    COLOR    Self    Our    You    Name    Yes    No    Negate    Intrj    IAV    DAV    SV    IPadj    IndAdj    PowGain    PowLoss    PowEnds    
        PowAren    PowCon    PowCoop    PowAuPt    PowPt    PowDoct    PowAuth    PowOth    PowTot    RcEthic    RcRelig    RcGain    RcLoss    RcEnds    RcTot    
        RspGain    RspLoss    RspOth    RspTot    AffGain    AffLoss    AffPt    AffOth    AffTot    WltPt    WltTran    WltOth    WltTot    WlbGain    WlbLoss    
        WlbPhys    WlbPsyc    WlbPt    WlbTot    EnlGain    EnlLoss    EnlEnds    EnlPt    EnlOth    EnlTot    SklAsth    SklPt    SklOth    SklTOT    
        TrnGain    TrnLoss    TranLw    MeansLw    EndsLw    ArenaLw    PtLw    Nation    Anomie    NegAff    PosAff    SureLw    If    NotLw    TimeSpc    
        FormLw    Othrtags    Defined
    ''' 
    columns = ['Positiv', 'Negativ', 'Hostile', 'Strong', 'Pleasur', 'Pain']
    
    i = 0
    for i in raw2.index:
    
        categries = []
        #for col in raw2.columns:
        for col in columns:
            #print('%s = %s, %s' % (col, raw2[col].ix[i], type(raw2[col].ix[i])))
            if col != 'Entry' and col != 'Source' and raw2['Entry'].ix[i] != 'A':
                if type(raw2[col].ix[i]) is str:
                    categries.append(col)
                    
                    
        #check if key is already there (e.g. ABOUT#2
        #entry = raw2['Entry'].ix[i].split('#')[0]
        en    = raw2['Entry'].ix[i].split('#')[0]
        
        if len(categries) > 0:
            if en in gi:
                gi[en] = gi[en] + categries
            else:
                gi[en] = categries
            
            gi[en] = list(set(gi[en]))
                
            
    
        '''
        if i > 200:    
            for k in gi:
                #print(k)
                #print(gi[k])
                print('%s: %s' % (k, str(gi[k])))
            quit()
        '''
    
    return(gi)



'''
    load data from original training file
'''
def load_from_file(filename):
    #raw = pd.read_csv(filename, header=0, delimiter="\t", quoting=3, index_col=['ID'])
    #raw = pd.read_csv(filename, header=0, delimiter="\t", quoting=3)
    
    raw = pd.read_csv(Task6_file_links.training_data_filename, header=None, delimiter="\t", quoting=3, names=['ID', 'Target', 'Tweet', 'Stance'])

    raw.loc[raw['Stance'] == 'PROTI','Stance'] = 'AGAINST'
    raw.loc[raw['Stance'] == 'PRO','Stance'] = 'FAVOR'
    raw.loc[raw['Stance'] == 'NIC','Stance'] = 'NONE'


    
    return(raw)

def load_trump (filename, filter_nas = True):
    #raw = pd.read_csv(filename, header=0, delimiter='\t', quoting=3)
    raw = pd.read_csv(filename,  header=None,delimiter='\t', quoting=3)
    raw.columns = ['ID', 'Tweet']

    #load_trump(Task6_file_links.trump_data_filename)
    
    #print(raw.head(15))
    return (raw)


#ID    Target    Tweet    Stance
def load_trump_oficial (filename, filter_nas = True):
    raw = pd.read_csv(filename, header=0,delimiter='\t', quoting=3)
    
    if filter_nas:
        raw = raw[(raw.Tweet != 'Not Available')]

    
    #pprint(raw.head(5))
    #quit()
    return(raw)
    

#load_trum_oficial(Task6_file_links.official_test_data_subtaskB, True)
#quit()


def load_additional_data_labelled (filename):
    raw = pd.read_csv(filename, header=0, delimiter='\t', quoting=3)
    #pprint(raw.head(10))
    return (raw)

#x = load_additional_data_labelled(Task6_file_links.manually_labelled_sample)


def load_official_test_data(filename):
    raw = pd.read_csv(filename, header=0, delimiter='\t', quoting=3)
    #pprint(raw.head(5))
    return(raw[['ID','Target','Tweet','Stance']])





# TODO is_new_tweet_similar_to_existing
def is_new_tweet_similar_to_existing(tweet_text, existing_tweets):
    return(True)

def load_additional_data (filename, target, stance):
    '''
    hillary: {'benghazi': 'AGAINST', 'lol': 'AGAINST', 'stophillary2016': 'AGAINST'}
    climate: {'climate': 'FAVOR', 'mission': 'FAVOR', 'peace': 'NONE', 'tip': 'FAVOR'}
    feminist: {'feminists': 'AGAINST', 'spankafeminist': 'AGAINST'}
    abortion: {'alllivesmatter': 'AGAINST', 'ccot': 'AGAINST', 'prolifeyouth': 'AGAINST'}
    atheism: {'freethinker': 'FAVOR', 'islam': 'AGAINST'}
    '''
    '''
    {'benghazi': 'AGAINST', 'catholic': 'AGAINST', 'ccot': 'AGAINST', 'climate': 'FAVOR', 'feminists': 'AGAINST',
     'freethinker': 'FAVOR', 'mission': 'FAVOR', 'prolifeyouth': 'AGAINST', 'rapeculture': 'FAVOR',
     'spankafeminist': 'AGAINST', 'stophillary2016': 'AGAINST', 'teamjesus': 'AGAINST', 'tip': 'FAVOR'}
    '''
    
    #benghazi_hillary_clinton = '/Users/peterkrejzl/Dropbox/PHD/SemEval2016/OldTweets/GetOldTweets-master/benghazi_hillary_clinton.csv'
    #stophillary2016 = '/Users/peterkrejzl/Dropbox/PHD/SemEval2016/OldTweets/GetOldTweets-master/stophillary2016.csv'
    #support_hillary_max_1000 = '/Users/peterkrejzl/Dropbox/PHD/SemEval2016/OldTweets/GetOldTweets-master/support_hillary_max_1000.csv'
    #abortion_alllivesmatter = '/Users/peterkrejzl/Dropbox/PHD/SemEval2016/OldTweets/GetOldTweets-master/abortion_alllivesmatter.csv'
    #abortion_prolifeyouth = '/Users/peterkrejzl/Dropbox/PHD/SemEval2016/OldTweets/GetOldTweets-master/abortion_prolifeyouth.csv'
    #alllivesmatter = '/Users/peterkrejzl/Dropbox/PHD/SemEval2016/OldTweets/GetOldTweets-master/alllivesmatter.csv'
    #atheism_islam_freethinker = '/Users/peterkrejzl/Dropbox/PHD/SemEval2016/OldTweets/GetOldTweets-master/atheism_islam_freethinker.csv'
    #islam_freethinker = '/Users/peterkrejzl/Dropbox/PHD/SemEval2016/OldTweets/GetOldTweets-master/islam_freethinker.csv'
    #climate_mission = '/Users/peterkrejzl/Dropbox/PHD/SemEval2016/OldTweets/GetOldTweets-master/climate_mission.csv'
    #pankafeminist = '/Users/peterkrejzl/Dropbox/PHD/SemEval2016/OldTweets/GetOldTweets-master/spankafeminist.csv'
     
     
    #raw = pd.read_csv(filename,  header=0,delimiter=';', quoting=3)
    #pprint(raw.head(10))
    docs = [line.rstrip('\r\n') for line in open(filename, 'r') if line.rstrip('\r\n')[0] != 'username']
    docs = docs[1:]
    processed_rows = []
    
    for doc in docs:
        positions = [x for x,v in enumerate(doc) if v == ';']
        if len(positions) > 4:
            row = doc[:positions[3]+1].replace(';','\t') + doc[positions[3]+1:]
        else:
            row = doc.replace(';','\t')
            
        #if is_new_tweet_similar_to_existing(row, raw):
        processed_rows.append(row.split('\t')[4])
            
    additional_rec = pd.DataFrame({'Target' : target, 'Stance' : stance, 'Tweet' : processed_rows})
    
    #pprint(additional_rec.head(10))
    #print(additional_rec.shape)
      
    return(additional_rec)


'''
load_additional_data(Task6_file_links.spankafeminist, 'Feminism', 'AGAINST')
load_additional_data(Task6_file_links.benghazi_hillary_clinton, 'Hillary', 'AGAINST')
load_additional_data(Task6_file_links.stophillary2016, 'Hillary', 'AGAINST')

load_additional_data(Task6_file_links.support_hillary_max_1000, 'Hillary', 'FAVOR')

load_additional_data(Task6_file_links.abortion_alllivesmatter, 'Abortion', 'AGAINST')
load_additional_data(Task6_file_links.abortion_prolifeyouth, 'Abortion', 'AGAINST')
load_additional_data(Task6_file_links.alllivesmatter, 'Abortion', 'AGAINST')

load_additional_data(Task6_file_links.atheism_islam_freethinker, 'Atheism', 'AGAINST')
load_additional_data(Task6_file_links.islam_freethinker, 'Atheism', 'AGAINST')

load_additional_data(Task6_file_links.climate_mission, 'Climate', 'AGAINST')

quit()
'''


'''
    load bad words from following places
'''
'''
def load_bad_words():

    docs = [line.rstrip('\r\n') for line in open(Task6_file_links.bad_words, 'r')]
    docs2 = [line.rstrip('\r\n') for line in open(Task6_file_links.bad_words2, 'r')]
    docs3 = [line.rstrip('\r\n') for line in open(Task6_file_links.bad_words3, 'r')]

    docs_all = docs + docs2 + docs3

    bw = set(docs_all)
    return (bw)
 
 
 
 
bw = load_bad_words()
cached_stop_words = stopwords.words("english")
gi = load_GI_data(Task6_file_links.gi_input_file)


'''
cached_stop_words = []
#stop_w_file = open("/Users/peterkrejzl/Dropbox/PHD/SemEval2016/CzClassifierSourceCode/stopwords/czech_stop_words_out.txt", "r")
stop_w_file = open(Task6_file_links.stop_words, 'r')

lines = stop_w_file.readlines()
for i in lines:
    #print(i)
    #print(i.rstrip())
    cached_stop_words.append(i.rstrip())
#print(lines)
#print(cached_stop_words)
#quit()
#cached_stop_words = 
'''
text_file = open("filename.dat", "r")
lines = text_file.readlines()
print lines
print len(lines)
text_file.close()
'''


 
def extract_hashtags(row):
    tags = re.findall(r"#(\w+)", row.lower())
    return(' '.join(t for t in tags if t not in excluded_tags))

 
#Task6_raw_processing.generate_output_file(result, test_data_raw, 'out.txt')
            
            
def generate_output_file_training_only(result, test_data_raw, filename):
    output_file = open(filename, 'w')
    output_file.write('ID    Target    Tweet    Stance\n')
    
    
    output = {}

    j = 0
    for i in test_data_raw.index:
        id = test_data_raw['ID'].ix[i]
        target = test_data_raw['Target'].ix[i]
        tweet = test_data_raw['Tweet'].ix[i]
        res_stance = result[j]
        output[id] = target + '\t' + tweet + '\t' + res_stance

        j+=1      
        

    x = 0
    for key in sorted(output):
        output_file.write("" + str(key) + "\t" + str(output[key]) + "\n")
        x +=1
    
            
    output_file.close()
    return()    
      
def generate_output_file_trump(result, test_data_raw, filename):
    output_file = open(filename, 'w')
    #output_file.write('ID    Tweet    Stance\n')
    output_file.write('ID\tTarget\tTweet\tStance\n')
    
    
    output = {}
    
    j = 0
    for i in test_data_raw.index:
        id = test_data_raw['ID'].ix[i]
        tweet = test_data_raw['Tweet'].ix[i]
        res_stance = result[j]
        output[id] = tweet + '\t' + res_stance
        j += 1
        
    x = 0
    for key in sorted(output):
        output_file.write("" + str(key) + '\tDonald Trump\t' + str(output[key]) + "\n")
        x += 1
        
    output_file.close()
    return()

      
            
#def generate_output_file (output_data, filename):
def generate_output_file(result, test_data_raw, filename):
    '''
        header =     ID    Target    Tweet    Stance
        rows =        1    Hillary Clinton    @tedcruz And, #HandOverTheServer... FAVOR
    '''
    output_file = open(filename, 'w')
    output_file.write('ID    Target    Tweet    Stance\n')
    
    if len(result) == 0:
        output_file.write("1    Hillary Clinton    @tedcruz And, #HandOverTheServer she wiped clean + 30k deleted emails, explains dereliction of duty/lies re #Benghazi,etc #tcot #SemST    AGAINST\n")
        output_file.write("2    Hillary Clinton    Hillary is our best choice if we truly want to continue being a progressive nation. #Ohio #SemST    FAVOR\n")
        output_file.write("3    Hillary Clinton    @TheView I think our country is ready for a female pres, it can't ever be Hillary #SemST    AGAINST\n")
        output_file.write("4    Hillary Clinton    I just gave an unhealthy amount of my hard-earned money away to the big gov't & untrustworthy IRS. #WhyImNotVotingForHillary #SemST    NONE\n")
        output_file.write("4    Hillary Clinton    I just gave an unhealthy amount of my hard-earned money away to the big gov't & untrustworthy IRS. #WhyImNotVotingForHillary #SemST    FAVOR\n")
    
        output_file.close()
        return()
    
   
   
    output = {}

    j = 0
    #pprint(test_data_raw.head(10))
    #pprint(result.head(10))
    #quit()
    
    for i in test_data_raw.index:
        id = test_data_raw['ID'].ix[i]
        target = test_data_raw['Target'].ix[i]
        tweet = test_data_raw['Tweet'].ix[i]
        stance = test_data_raw['Stance'].ix[i]
        res_stance = result[j]
        output[id] = target + '\t' + tweet + '\t' + stance + '\t' + res_stance
        
        
        j+=1

    
    x = 0
    for key in sorted(output):
        output_file.write("" + str(key) + "\t" + str(output[key]) + "\n")
        x +=1
        
 
    output_file.close()
    return()
    
    
def preprocess_raw_data_hashtags(raw_data):
    hashtags = []
    
    for row in raw_data['Tweet']:
        hashtags.append(extract_hashtags(row))
        
    return(hashtags)

# TODO: calculate sentiment 
def calculate_sentiment(row):
    return([random.random()])

'''
    return array of POS tags for given tweet
'''
def get_POS(string):
    tokens = word_tokenize(string)
    pos = nltk.pos_tag(tokens)

    pos_array = []
    for p in pos:
        pos_array.append(p[1])
        
    return(' '.join(pos_array))


def get_GI(string):
    categories = []
    
    
    #print('S = %s' % string)
    #print(gi.keys())
    for w in string.split():
        if w.upper() in gi:
            #print('\t%s = %s' % (w.upper(), gi[w.upper()]))
            for cat in gi[w.upper()]:
                categories.append(cat)
                
    
    #if len(categories) > 0:
        #print(string)
        #print(categories)
        #print(list(set(categories)))
        #quit()
    #return(categories)
    #print(list(set(categories)))
    return(' '.join(list(set(categories))))

    
def preprocess_raw_data(raw_data):
    #stop_words = set(stopwords.words('english'))
    #stop_words = set(stopwords.words('czech'))
    
    #pprint(stop_words)
    #exit()
    
    clean_tweets = []
    
    for row in raw_data['Tweet']:
        #cleaned_row = preprocess_one_row_raw_data(row, stop_words)
        #print(row)
        #quit()
        cleaned_row = preprocess_one_row_raw_data(row)
        clean_tweets.append(cleaned_row)
    
    
    
    return(clean_tweets)



def preprocess_raw_data_dsd2(rows, positive_dsd, negative_dsd):
    results = []

    load_positive = load_dsd(positive_dsd)
    load_negative = load_dsd(negative_dsd) 
    
    load_negative = set(load_negative['WORD'])
    load_positive = set(load_positive['WORD'])
    
    for row in rows['Tweet']:
        pos = neg = 0
        for word in row.split():
            if word in load_positive:
                pos += 1
            if word in load_negative:
                neg += 1

        results.append([pos, neg])
        
    return(results)
    
    
def preprocess_raw_data_dsd(rows, positive_dsd, negative_dsd):
    results = []

    load_positive = load_dsd(positive_dsd)
    load_negative = load_dsd(negative_dsd)
    
    load_positive = [w.lower() for w in load_positive['WORD']]
    load_negative = [w.lower() for w in load_negative['WORD']]
    
    #add same with hashtags
    load_positive2 = []
    load_negative2 = []
    
    for w in load_positive:
        load_positive2.append('#' + w)
        load_positive2.append(w)
        
    for w in load_negative:
        load_negative2.append('#' + w)
        load_negative2.append(w)
        
    load_negative2 = set(load_negative2)
    load_positive2 = set(load_positive2)
    

    
    
    for row in rows:#['Tweet']:
        #print('processing row = %s' % row)
        #If you regularly base your thoughts
        
        pos = neg = 0
        
        for word in row.split():
            if word in load_positive2:
                pos +=1
            if word in load_negative2:
                neg +=1
                    
        results.append([pos, neg])

    
    return(results)


def preprocess_raw_data_dsd_additional (rows, positive_dsd, negative_dsd):
    results = []
    load_positive2 = []
    load_negative2 = []
    
    if positive_dsd != None:
        load_positive = load_dsd(positive_dsd)
        load_positive = [w.lower() for w in load_positive['WORD']]
        for w in load_positive:
            load_positive2.append('#' + w)
            load_positive2.append(w)
        load_positive2 = set(load_positive2)
            
    if negative_dsd != None:
        load_negative = load_dsd(negative_dsd)
        load_negative = [w.lower() for w in load_negative['WORD']]
        for w in load_negative:
            load_negative2.append('#' + w)
            load_negative2.append(w)
        load_negative2 = set(load_negative2)
            
    for row in rows:
        pos = neg = 0
        
        for word in row.split():
            if word in load_positive2:
                pos += 1
            if word in load_negative2:
                neg += 1
        results.append([pos,neg])
    
    return(results)
    


def preprocess_raw_data_jrc(rows, jrc_dict_filename1, jrc_dict_filename2):
    dict1 = load_jrc(jrc_dict_filename1)
    dict2 = load_jrc(jrc_dict_filename2)
    
    #pprint(dict1)
    dict1b = set([w.lower() for w in dict1['WORD']])
    dict2b = set([w.lower() for w in dict2['WORD']])
    

    
    counts = []
    
    for row in rows:
        cnt = 0
        for word in row.split():
            if word in dict1b or word in dict2b:
                cnt += 1
        counts.append([cnt])
    return (counts)


    



def preprocess_raw_data_bible_reference (rows):
    
    contains_reference = []
    #logging.log(logging.INFO, 'processing row for bible reference')
    #pprint(rows)
    
    for row in rows['Tweet']:
       
            
        if re.search(r"(\d+):(\d+)", row) != None:
            contains_reference.append([1])
            #print('reference found')
        else:
            contains_reference.append([0])
            #print('reference not found')
    #quit()
    return(contains_reference)
                                       


def preprocess_raw_data_initial_unigrams(cleaned_rows):
    initial_unigrams = []
    
    for row in cleaned_rows:
        ln = len(row.split())
        if ln >= 1:
            initial_unigrams.append(' '.join(row.split()[0:1]))
        else:
            initial_unigrams.append('')
    return(initial_unigrams)
        

def preprocess_raw_data_initial_bigrams(cleaned_rows):
    initial_bigrams = []
    
    for row in cleaned_rows:
        ln = len(row.split())
        if ln >= 2:
            initial_bigrams.append(' '.join(row.split()[0:2]))
        else:
            initial_bigrams.append('')
    
    return(initial_bigrams)


def preprocess_raw_data_initial_trigrams(cleaned_rows):
    initial_trigrams = []
    
    for row in cleaned_rows:
        ln = len(row.split())
        if ln >= 3:
            initial_trigrams.append(' '.join(row.split()[0:3]))
        else:
            initial_trigrams.append('')

    return(initial_trigrams)


def get_percentage_of_all_caps(row):
    all_caps = 0
    num_of_words = len(row.split())
    
    for word in row.split():
        if word.isupper():
            all_caps += 1
            
    percentage = int(all_caps / (num_of_words / 100.0))
    
    return(percentage)


def preprocess_raw_data_all_caps(rows):
    percentages = []
    
    for row in rows:
        cnt = get_percentage_of_all_caps(row)
        percentages.append(cnt)
        
    return(percentages)



def preprocess_raw_data_tweet_sizes(cleaned_rows):
    tweet_sizes = []
    
    for row in cleaned_rows:
        ln = len(row.split())
        tweet_sizes.append([ln])
        
    tweet_sizes = np.array(tweet_sizes)
        
    return(tweet_sizes)

def preprocess_raw_data_sentiment(cleaned_rows):
    sentiments = []
    
    for row in cleaned_rows:
        sentiments.append(calculate_sentiment(row))

    sentiments = np.array(sentiments)
    return(sentiments)


def preprocess_raw_data_cs_n_p (rows):

    neg = []
    pos = []
    res = []
    #load dictionaries
    with open(Task6_file_links.cs_n) as neg_f:
        for line in neg_f:
            if len(line) > 0:
                neg.append(line.rstrip())
    
    with open(Task6_file_links.cs_p) as pos_f:
        for line in pos_f:
            if len(line) > 0:
                pos.append(line.rstrip())
        

    for row in rows:
        cs_n = cs_p = 0
        for word in row.split():
            if word in pos:
                cs_p += 1
            if word in neg:
                cs_n += 1
        res.append([cs_p, cs_n])
    
    return(res)


def get_senti_stats(row, dict):
    #print(row)
    
    pos_cnt = neg_cnt = 0
    
    for word in row.split():
        #check if the word exists
        if dict.has_key(word):
            #print(word)
            #print(dict[word])
            if dict[word][0] > dict[word][1]:
                #print('POS')
                pos_cnt +=1
            else:
                neg_cnt +=1
        
    
    #print([pos_cnt, neg_cnt])
    #quit()       
    return([pos_cnt, neg_cnt])

# TODO: preprocess_raw_data_sentiwords
def preprocess_raw_data_sentiwords(cleaned_rows):
    senti_stats = []
    
    dict = load_sentiwordnet(Task6_file_links.senti_wordnet)
    
    for row in cleaned_rows:
        # TODO:
        senti_stats.append(get_senti_stats(row, dict))
        
    
    return(senti_stats)



def preprocess_raw_data_POS(cleaned_rows):
    POS = []
    
    for row in cleaned_rows:
        POS.append(get_POS(row))
        
    return(POS)



'''
    nouns, verbs, adjectives, adverbs
    NN nouns
    VB verbs
    JJ adjective
    RB adverb
'''
def preprocess_raw_data_POS_percentage(cleaned_rows, nouns = True, verbs = True, adjectives = True, adverbs = True):
    POS_percentage = []
    items_to_check = []
    
    if nouns:
        items_to_check.append('NN')
    if verbs:
        items_to_check.append('VB')
    if adjectives:
        items_to_check.append('JJ')
    if adverbs:
        items_to_check.append('RB')


    '''
        e.g. all types of verbs VBS, VBG,... are converted into VB
    '''    
    for row in cleaned_rows:
        pos = get_POS(row).split()
        pos_len = len(pos)
        cnt = Counter([item[0:2] for item in pos])
        
        vector = [int(cnt[item] / (pos_len / 100.0)) for item in items_to_check]
        #print(vector)
        POS_percentage.append(vector)
            
    return(POS_percentage)







#ABOLISH: ['Strong', 'Hostile', 'Negativ']
def preprocess_raw_data_GI(cleaned_rows):
    GI_categories = []
    
    for row in cleaned_rows:
        GI_categories.append(get_GI(row))

    return(GI_categories)



def preprocess_raw_data_for_overrides(rows, topic):
    overrides = [] #output
    
    ovrs = {}
    #abortion, atheism, climate, feminism, hillary
    
    if topic == 'hillary':
        favs_dsd = load_dsd(Task6_file_links.dsd2_hillary_favor_override)
        ags_dsd = load_dsd(Task6_file_links.dsd2_hillary_against_override)
        logging.log(logging.INFO, '\t\toverride dictionary from ' + Task6_file_links.dsd2_hillary_favor_override + ' loaded')
        logging.log(logging.INFO, '\t\toverride dictionary from ' + Task6_file_links.dsd2_hillary_against_override + ' loaded')
    elif topic == 'abortion':
        favs_dsd = load_dsd(Task6_file_links.dsd2_abortion_favor_override)
        ags_dsd = load_dsd(Task6_file_links.dsd2_abortion_against_override)
        logging.log(logging.INFO, '\t\toverride dictionary from ' + Task6_file_links.dsd2_abortion_favor_override + ' loaded')
        logging.log(logging.INFO, '\t\toverride dictionary from ' + Task6_file_links.dsd2_abortion_against_override + ' loaded')
    elif topic == 'atheism':
        favs_dsd = load_dsd(Task6_file_links.dsd2_atheism_favor_override)
        ags_dsd = load_dsd(Task6_file_links.dsd2_atheism_against_override)
        logging.log(logging.INFO, '\t\toverride dictionary from ' + Task6_file_links.dsd2_atheism_favor_override + ' loaded')
        logging.log(logging.INFO, '\t\toverride dictionary from ' + Task6_file_links.dsd2_atheism_against_override + ' loaded')
    elif topic == 'feminism':
        favs_dsd = load_dsd(Task6_file_links.dsd2_feminism_favor_override)
        ags_dsd = load_dsd(Task6_file_links.dsd2_feminism_against_override)
        logging.log(logging.INFO, '\t\toverride dictionary from ' + Task6_file_links.dsd2_feminism_favor_override + ' loaded')
        logging.log(logging.INFO, '\t\toverride dictionary from ' + Task6_file_links.dsd2_feminism_against_override + ' loaded')
    elif topic == 'climate':
        favs_dsd = load_dsd(Task6_file_links.dsd2_climate_favor_override)
        ags_dsd = load_dsd(Task6_file_links.dsd2_climate_against_override)
        logging.log(logging.INFO, '\t\toverride dictionary from ' + Task6_file_links.dsd2_climate_favor_override + ' loaded')
        logging.log(logging.INFO, '\t\toverride dictionary from ' + Task6_file_links.dsd2_climate_against_override + ' loaded')
    elif topic == 'trump':
        favs_dsd = load_dsd(Task6_file_links.dsd2_trump_favor_override)
        ags_dsd = load_dsd(Task6_file_links.dsd2_trump_against_override)
        logging.log(logging.INFO, '\t\toverride dictionary from ' + Task6_file_links.dsd2_trump_favor_override + ' loaded')
        logging.log(logging.INFO, '\t\toverride dictionary from ' + Task6_file_links.dsd2_trump_against_override + ' loaded')
       

    #quit()

    #load into dictionary
    for w in favs_dsd['WORD']:
        ovrs[w] = 'FAVOR'
    for w in ags_dsd['WORD']:
        ovrs[w] = 'AGAINST'
     
    #pprint(favs_dsd)
    #quit()   

    #process raw data
    for row in rows['Tweet']:
        
        #for this row check all
        favs = nones = ags = 0
        catched = 0
        for k in ovrs:
            for word in row.split():
                
                if k.lower() == word.lower() or '#' + k.lower() == word.lower():
                    logging.log(logging.INFO, '\tTweet = ' + row)
                    logging.log(logging.INFO, '\t\tcatched ' + k)
                    catched = 1
                    if ovrs[k] == 'FAVOR':
                        favs += 1
                    elif ovrs[k] == 'NONE':
                        nones +=1
                    elif ovrs[k] == 'AGAINST':
                        ags +=1
             
        if catched:
            logging.log(logging.INFO, '\t\tFavorites = ' + str(favs) + 'x, againsts = ' + str(ags) + 'x, nones = ' + str(nones) +'x')
        #now choose biggest one
        if favs > 0 or ags > 0 or nones > 0:    
            if favs > ags and favs > nones:
                overrides.append('FAVOR')
                logging.log(logging.INFO, '\t\toverride to FAVOR')
            elif ags > favs and ags > nones:
                overrides.append('AGAINST')
                logging.log(logging.INFO, '\t\toverride to AGAINST')
            elif nones > favs and nones > ags:
                overrides.append('NONE')
                logging.log(logging.INFO, '\t\toverride to NONE')
            else:
                overrides.append(None)
        else:
            overrides.append(None)
        
        #if row == '@realDonaldTrump  I like Mexicans who come to US legally.  #IStandWithTrump #SemST':
       
        if row == "No you see Donald Trump was saying Mexican immigrants are all *rappists* like with the hip hop y'all  #Trump #SemST":
            print(row)
            print('favs = %d, ags = %d, nones = %d' % (favs, ags, nones))
            #quit()
        
    return(overrides)


def process_overrides(result, test_data_raw, topic):
    logging.log(logging.INFO, '\tapplying overrides')
    overrides = preprocess_raw_data_for_overrides(test_data_raw, topic)
    
    for i in range(len(overrides)):
        if overrides[i] != None:
            logging.log(logging.INFO, '\t\tOVERRIDE ' + result[i] + ' -> ' + overrides[i])
        if overrides[i] != None and overrides[i] != result[i]:
        #if overrides[i] != None:
            
            logging.log(logging.INFO,  '\t\toverride stance  ' + result[i] + ' with ' + overrides[i])
            
            result[i] = overrides[i]
    
    return(result)  

# TODO: preprocess one row
def preprocess_one_row_raw_data(row, stopwords = None, preprocessing=True, trump=False):
    #print(row)
    if preprocessing:
        
        #remove RT
        '''
            if the tweet starts with RT
            RT @GreenLivingGB
        '''
        if row.startswith('RT'):
            row = row[2:]
        
        
        row = re.sub(r"(?:\http|https?\://)\S+", LINK, row)
        row = re.sub(r"(\!)\1{2,}", EXCLAMATION, row)
        row = re.sub(r"(\?)\1{2,}", QMARK, row)
        
        if trump:
            row = row.replace('hillary', '')
            row = row.replace('hilary', '')
            row = row.replace('clinton', '')
        
        
        
        
        #pic.twitter.com/yGse4h5LSJ
        splt = row.split()
        splt_w = []
        for s in splt:
            if s.startswith('pic.twitter.com'):
                splt_w.append(IMG_LINK)
            else:
                splt_w.append(s)
        row = ' '.join(splt_w)
        
        
        #print('preprocessing %s' + row)
        
        #hardcoding bible regex for now
        row = re.sub(r"\b(\d+):(\d+)\b", BIBLE, row)
        '''
        if re.search(r"(\d+):(\d+)", row):
            print(row)
            row = re.sub(r"\b(\d+):(\d+)\b", BIBLE, row)
            print(row)
            quit()
        '''
        '''
        #heuristics
        heur = load_heuristics(Task6_file_links.heuristics)
        for i in heur.index:
            name = heur['NAME'].ix[i]
            regex = heur['REGEX'].ix[i]
            #row = re.sub(regex, name, row)
            
            patt = re.compile(regex)
            row = re.sub(patt, name, row)
        '''

        
        #letters_only = re.sub("[^a-zA-Z#@]", " ", row) 
        letters_only = row
        
        words = letters_only.lower()
        words = re.sub(LINK.lower(), LINK, words)
        words = re.sub(QMARK.lower(), QMARK, words)
        words = re.sub(EXCLAMATION.lower(), EXCLAMATION, words)
        words = re.sub(IMG_LINK.lower(), IMG_LINK, words)
        words = words.replace('#semst', '')
        

            
        
        #convert user name to NAME
        words = re.sub(r'\B\@\w+', USER_NAME, words)
        

        #caps?
        
    
        
        #remove stop words
        
       # lmn = WordNetLemmatizer()
        #meaningful_words = [lmn.lemmatize(w) for w in words.split() if not w in cached_stop_words]
        #meaningful_words = [lmn.lemmatize(w) for w in words.split()]
        #meaningful_words = [w for w in words.split()] #if not w in stop_words]
        meaningful_words = [w for w in words.split() if not w in cached_stop_words]
    
        '''
        meaningful_words2 = []
        for w in meaningful_words:
            if w in bw:
                meaningful_words2.append(BAD_WORD)
            else:
                meaningful_words2.append(w)
        '''
        
        #print(' '.join(meaningful_words))
        #quit()
        
        
        return(' '.join(meaningful_words))
    else:
        #pprint(filter(lambda x: x in string.printable, row))
        return(filter(lambda x: x in string.printable, row))

def prepare_data_for_similarity(training_data, additional_data, output_file):
    compare_file = open(output_file, 'w')
    
    train_data_raw = pd.read_csv(training_data, header=0, delimiter="\t", quoting=3)
    #train_data_raw = train_data_raw['Tweet']
    
    add_data = load_additional_data (additional_data, 'Hillary', 'AGAINST')
    #add_data = add_data['Tweet']
    #pprint(train_data_raw)
    #pprint(add_data.head(5))
    
    
    for i in train_data_raw.index:
        for j in add_data.index:
            sentence1 = train_data_raw['Tweet'].ix[i]
            sentence2 = add_data['Tweet'].ix[j]
            
            compare_file.write(sentence1 + '\t' + sentence2 + '\n')
    
    
    compare_file.close();
    return(0)

def export_additional_tweets(filename, outfilename):
    #load_additional_data (filename, target, stance):
    out = load_additional_data(filename, 'Hillary', 'AGAINST')
    pprint(out['Tweet'].head(5))
    outf = open(outfilename, 'w')
    for i in out.index:
        outf.write('%s\n' % out['Tweet'].ix[i])
        
    outf.close     
def load_additional_data_filtered(filename):
    raw = pd.read_csv(filename, header=None, delimiter='\t', quoting=3, names = ['Similarity', 'Tweet', 'Stance', 'SourceTweet'])
    #pprint(raw.head(10))
    '''
    Similarity    Tweet    Stance    SourceTweet
    '''
    #ignoring stance for now
    #return(raw[['Tweet']])
    return(raw[['Tweet', 'Stance']])




def validate_trump():
    raw = pd.read_csv('/Users/peterkrejzl/Dropbox/PHD/SemEval2016/SourceCode2/trump_output.txt', header=0, delimiter="\t", quoting=3)
    raw_labelled = pd.read_csv('/Users/peterkrejzl/Dropbox/PHD/SemEval2016/_official_test_data/SemEval2016-Task6-testdata/Trump_manually_labelled.txt', header=0, delimiter="\t", quoting=3)
    #pprint(raw_labelled.head(5))
    #pprint(raw.head(5))
    
    to_check = zip(raw['Stance'], raw_labelled['Stance'])
    #pprint(to_check)
    
    total = 0
    ok = 0
    
    for i in range(len(to_check)):
        if to_check[i][1] != 'UNKNOWN':
            total += 1
            if to_check[i][1] == to_check[i][0]:
                ok += 1


    print('Trump - %d / %d' % (ok, total))

    
#validate_trump()
#quit()



def get_w2v_features(cleaned_rows_training):
    for row in cleaned_rows_training:
        print(row)
        #gaspadin zeman mel tote urychlen udel se sebo akor vzit rusk ambasad azyl u pritel kgb maleh velk vladimir
        quit()


