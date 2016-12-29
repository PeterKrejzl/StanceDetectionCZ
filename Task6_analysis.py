import logging
import random
from pprint import pprint
import re

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)



'''

    Analysis of the training data.
    Will produce hashtags that are worth of to use in additional twitter queries 
    because they have a strong polarity towards "AGAINST" or "FAVOR" stance.

'''



'''
    displays simple statistics of input data about counts of particular stances
'''
def data_stats(raw):
    stats = {'AGAINST' : 0, 'FAVOR' : 0, 'NONE' : 0}
    
    for stance in raw['Stance']:
        if stance == 'AGAINST':
            stats['AGAINST'] += 1
        elif stance == 'NONE':
            stats['NONE'] += 1
        elif stance == 'FAVOR':
            stats['FAVOR'] += 1
            
    return(stats)


'''
 normal precision / recall calculated against all three stances - FAVOR/AGAINST/NEUTRAL
 and not just FAVOR/AGAINST as in the official SemEval2016 metric
'''
def compute_f1_extended (result, test_data_raw, display_partial=True):
    combo = zip(test_data_raw['Stance'], result)
    
    
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
    accuracy_total_records = len(combo)
    
    for row in combo:
        if row[0] == row[1]:
            accuracy_oks += 1
        if row[0] == 'FAVOR':
            total_train_favs += 1
            if row[1] == 'FAVOR':
                tp_favs += 1
        elif row[0] == 'AGAINST':
            total_train_ags += 1
            if row[1] == 'AGAINST':
                tp_ags += 1
        elif row[0] == 'NONE':
            total_train_nones += 1
            if row[1] == 'NONE':
                tp_nones += 1
        
        if row[1] == 'FAVOR':
            total_test_favs += 1
        elif row[1] == 'AGAINST':
            total_test_ags += 1
        elif row[1] == 'NONE':
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
    
     
    if display_partial:
        logging.log(logging.INFO, '\t\tFAV P = ' + str(precision_favs) + ', R = ' + str(recall_favs) + ', F1 = ' + str(f1_favs))
        #print('FAV P = %f, R = %f, F1 = %f' % (precision_favs, recall_favs, f1_favs))
        logging.log(logging.INFO, '\t\ttr = ' + str(total_train_favs) + ', tp = ' + str(tp_favs) + ', test = ' + str(total_test_favs))
        #print('tr = %d, tp = %d, test = %d' % (total_train_favs, tp_favs, total_test_favs))
        logging.log(logging.INFO, '\t\tAGs P = ' + str(precision_ags) + ', R = ' + str(recall_ags) + ', F1 = ' + str(f1_ags))
        #print('AGs P = %f, R = %f, F1 = %f' % (precision_ags, recall_ags, f1_ags))
        
        
        logging.log(logging.INFO, '\t\ttr = ' + str(total_train_ags) + ', tp = ' + str(tp_ags) + ', test = ' + str(total_test_ags))
        #print('tr = %d, tp = %d, test = %d' % (total_train_ags, tp_ags, total_test_ags))
        
        logging.log(logging.INFO, '\t\tNONEs P = ' + str(precision_nones) + ', R = ' + str(recall_nones))
        #print('NONEs P = %f, R = %f' % (precision_nones, recall_nones))
        logging.log(logging.INFO, '\t\ttr = ' + str(total_train_nones) + ', tp = ' + str(tp_nones) + ', test = ' + str(total_test_nones))
        #print('tr = %d, tp = %d, test = %d' % (total_train_nones, tp_nones, total_test_nones))
        
        #print('avg F1 = %f' % (macro_F1))
        logging.log(logging.INFO, '\t\tAVG F1 = ' + str(macro_F1))
        
        logging.log(logging.INFO, '\t\tAccuracy = ' + str(accuracy_oks) + '/' + str(accuracy_total_records) + '(' + str(accuracy_oks / float(accuracy_total_records)) + ')')
        
        
    return (macro_F1)
    
    


'''
    official SemEval metric
'''
def compute_f1(result, test_raw_data, display_partial=True):  
    if len(result) == 0:
        return(random.random())
    
    combo = zip(test_raw_data['Stance'], result)
    
    

    total_train_favs = 0
    total_train_ags = 0
    total_train_nones = 0
    
    total_test_favs = 0
    total_test_ags = 0
    total_test_nones = 0
    
    tp_favs = 0
    tp_ags = 0
    tp_nones = 0
    
    
    #used for my internal check
    accuracy_oks = 0
    accuracy_total_records = len(combo)
    
    

    for row in combo:
        
        #accuracy
        if row[0] == row[1]:
            accuracy_oks += 1
        
        if row[0] == 'FAVOR':
            total_train_favs += 1
            if row[1] == 'FAVOR':
                tp_favs += 1
        elif row[0] == 'AGAINST':
            total_train_ags += 1
            if row[1] == 'AGAINST':
                tp_ags += 1
        elif row[0] == 'NONE':
            total_train_nones += 1
            if row[1] == 'NONE':
                tp_nones += 1
            
        if row[1] == 'FAVOR':
            total_test_favs += 1
        elif row[1] == 'AGAINST':
            total_test_ags += 1
        elif row[1] == 'NONE':
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
    

    
    if display_partial:
        logging.log(logging.INFO, '\t\tFAV P = ' + str(precision_favs) + ', R = ' + str(recall_favs) + ', F1 = ' + str(f1_favs))
        #print('FAV P = %f, R = %f, F1 = %f' % (precision_favs, recall_favs, f1_favs))
        logging.log(logging.INFO, '\t\ttr = ' + str(total_train_favs) + ', tp = ' + str(tp_favs) + ', test = ' + str(total_test_favs))
        #print('tr = %d, tp = %d, test = %d' % (total_train_favs, tp_favs, total_test_favs))
        logging.log(logging.INFO, '\t\tAGs P = ' + str(precision_ags) + ', R = ' + str(recall_ags) + ', F1 = ' + str(f1_ags))
        #print('AGs P = %f, R = %f, F1 = %f' % (precision_ags, recall_ags, f1_ags))
        
        
        logging.log(logging.INFO, '\t\ttr = ' + str(total_train_ags) + ', tp = ' + str(tp_ags) + ', test = ' + str(total_test_ags))
        #print('tr = %d, tp = %d, test = %d' % (total_train_ags, tp_ags, total_test_ags))
        
        logging.log(logging.INFO, '\t\tNONEs P = ' + str(precision_nones) + ', R = ' + str(recall_nones))
        #print('NONEs P = %f, R = %f' % (precision_nones, recall_nones))
        logging.log(logging.INFO, '\t\ttr = ' + str(total_train_nones) + ', tp = ' + str(tp_nones) + ', test = ' + str(total_test_nones))
        #print('tr = %d, tp = %d, test = %d' % (total_train_nones, tp_nones, total_test_nones))
        
        #print('avg F1 = %f' % (macro_F1))
        logging.log(logging.INFO, '\t\tAVG F1 = ' + str(macro_F1))
        
        logging.log(logging.INFO, '\t\tAccuracy = ' + str(accuracy_oks) + '/' + str(accuracy_total_records) + '(' + str(accuracy_oks / float(accuracy_total_records)) + ')')
        
        


    return (macro_F1)
    
    
def get_hashtags(docs, excluded, stances):
    data = {}
    topic = docs[1].split('\t')[1]
    
    for line in docs:
        fields = line.split('\t')
        tags = re.findall(r"#(\w+)", fields[2].lower())
        stance = fields[3]
        
        for tag in tags:
            #print(tag)
            if tag not in excluded:
                '''
                    one hot encoding :) NONE is index 0, AGAINST index = 1, FAVOR index = 2
                    other fields will be used later for some aggregations
                '''
                if data.has_key(tag) == False:
                    if stance == 'NONE':
                        data[tag] = [1,0,0,0,0,0,0]
                    elif stance == 'AGAINST':
                        data[tag] = [0,1,0,0,0,0,0]
                    elif stance == 'FAVOR':
                        data[tag] = [0,0,1,0,0,0,0]
                else:
                    if stance == 'NONE':
                        data[tag][0] += 1
                    elif stance == 'AGAINST':
                        data[tag][1] += 1
                    elif stance == 'FAVOR':
                        data[tag][2] += 1

    return(data, topic)



'''
    same function as get_hashtags but with Czech Stance words
'''
def get_hashtags_cz (docs, excluded, stances):
    data= {}
    topic = docs[1].split('\t')[1]
    
    for line in docs:
        if len(line) > 1:
            #print(line)
            fields = line.split('\t')
            #print(fields)
            tags = fields[2].lower().split()
            stance = fields[3]
            
            for tag in tags:
                if tag not in excluded:
                    if data.has_key(tag) == False:
                        if stance == 'NIC':
                            data[tag] = [1,0,0,0,0,0,0]
                        elif stance == 'PROTI':
                            data[tag] = [0,1,0,0,0,0,0]
                        elif stance == 'PRO':
                            data[tag] = [0,0,1,0,0,0,0]
                    else:
                        if stance == 'NIC':
                            data[tag][0] += 1
                        elif stance == 'PROTI':
                            data[tag][1] += 1
                        elif stance == 'PRO':
                            data[tag][2] += 1
    return (data, topic)
        


# TODO: get_words
def get_words():
    return([])



def calculate_hashtags(hashtags):
    for h in hashtags:
        #sum
        hashtags[h][3] = hashtags[h][0] + hashtags[h][1] + hashtags[h][2]
        #none_ratio
        hashtags[h][4] = hashtags[h][0] / float(hashtags[h][3])
        #against ratio
        hashtags[h][5] = hashtags[h][1] / float(hashtags[h][3])
        #favor ratio
        hashtags[h][6] = hashtags[h][2] / float(hashtags[h][3])
        
    return(hashtags)


    
def get_topn_hashtags(hashtags, sum_min, ratio_min):
    topn_hashtags = {}
    for h in hashtags:
        if hashtags[h][3] >= sum_min and (hashtags[h][4] >= ratio_min or hashtags[h][5] >= ratio_min or hashtags[h][6] >= ratio_min):
            topn_hashtags[h] = hashtags[h]
    
    return(topn_hashtags)


def get_tag_with_stance(topn_hashtags, ratio_min):
    topn_stances = {}
    for h in topn_hashtags:
        #print(h)
        if topn_hashtags[h][4] >= ratio_min:
            topn_stances[h] = 'NONE'
        elif topn_hashtags[h][5] >= ratio_min:
            topn_stances[h] = 'AGAINST'
        elif topn_hashtags[h][6] >= ratio_min:
            topn_stances[h] = 'FAVOR'
    
    return(topn_stances)

def analyze_training_data(filename):
    
    print('starting analysis')
    
    docs = [line.rstrip('\r\n') for line in open(filename, 'r')]
    data = {}
    
    stances = ('NONE', 'AGAINST', 'FAVOR', )
    excluded_tags = ('SemST', 'semst')
    
    #hashtags, topic = get_hashtags(docs, excluded_tags, stances)
    #hashtags = calculate_hashtags(hashtags)
    #topn_hashtags = get_topn_hashtags(hashtags, 10, 0.9)
    #topn_hashtags_with_stances = get_tag_with_stance(topn_hashtags, 0.9)

    
    pprint('********************   HASHTAGS   ********************   ')
    #pprint(topn_hashtags)
    #pprint(topn_hashtags_with_stances)

    print('done analysis')
    quit()

def analyze_training_data_cz(filename):
    print('starting analysis')
    
    docs = [line.rstrip('\r\n') for line in open(filename, 'r')]
    data = {}
    
    stances = ('NIC', 'PROTI', 'PRO', )
    excluded_tags = ('SemST', 'semst')
    
    tags, topic = get_hashtags_cz(docs, excluded_tags, stances)
    tags = calculate_hashtags(tags)
    topn_hashtags = get_topn_hashtags(tags, 6, 0.9)
    topn_hashtags_with_stances = get_tag_with_stance(topn_hashtags, 0.5)
    #print(topn_hashtags)
    #print(topn_hashtags_with_stances)
    
    #quit()

