import time
import csv
import pickle
import operator
import datetime
import os

def prepare(dataset):
    with open(dataset, "r") as f:
        reader = csv.DictReader(f, delimiter=',')
        sess_clicks, sess_date = {}, {}
        ctr, curid = 0, -1
        curdate = None
        
        for data in reader:
            sessid = data['session_id']
            
            if curdate and not curid == sessid:
                sess_date[curid] = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
                
            curid = sessid
            item = data['item_id']
            curdate = data['timestamp']

            if sessid in sess_clicks:
                sess_clicks[sessid] += [item]
            else:
                sess_clicks[sessid] = [item]
                
            ctr += 1
            
        date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
        sess_date[curid] = date
    
    # Filter out length 1 sessions
    for s in list(sess_clicks):
        if len(sess_clicks[s]) == 1:
            del sess_clicks[s]
            del sess_date[s]
            
    # Count number of times each item appears
    iid_counts = {}
    for s in sess_clicks:
        seq = sess_clicks[s]
        for iid in seq:
            if iid in iid_counts:
                iid_counts[iid] += 1
            else:
                iid_counts[iid] = 1
                

    sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

    length = len(sess_clicks)
    for s in list(sess_clicks):
        curseq = sess_clicks[s]
        filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
        if len(filseq) < 2:
            del sess_clicks[s]
            del sess_date[s]
        else:
            sess_clicks[s] = filseq

    # Split out test set based on dates
    dates = list(sess_date.items())
    maxdate = dates[0][1]

    for _, date in dates:
        if maxdate < date:
            maxdate = date

    # 7 days for test
    splitdate = splitdate = maxdate - 86400 * 1  # the number of seconds for a day：86400

    tra_sess = filter(lambda x: x[1] < splitdate, dates)
    tes_sess = filter(lambda x: x[1] > splitdate, dates)

    # Sort sessions by date
    tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
    tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
    print(len(tra_sess))    # 186670    # 7966257
    print(len(tes_sess))    # 15979     # 15324
    
    return tra_sess, tes_sess, sess_clicks
    
    
def get_train_dataset(train_sess, sess_clicks, item_dict):
    train_ids, train_seqs, train_dates = [], [], []
    item_ctr = 1
    
    for s, date in train_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
    print(item_ctr)     # 43098, 37484
    return train_ids, train_dates, train_seqs  
    
    
def get_test_dataset(test_sess, sess_clicks, item_dict):
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in test_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs


def process_sequences(iseqs, idates):
    out_seqs, out_dates = [], []
    labs = []
    ids = []
    
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]
            
    return out_seqs, out_dates, labs, ids


def preprocess():
    
    train_sess, test_sess, sess_clicks = prepare('yoochoose/yoochoose-clicks.dat')
    item_dict = {}
    
    train_ids, train_dates, train_seqs = get_train_dataset(train_sess, sess_clicks, item_dict)
    test_ids, test_dates, test_seqs = get_test_dataset(test_sess, sess_clicks, item_dict)

    tr_seqs, tr_dates, tr_labs, tr_ids = process_sequences(train_seqs, train_dates)
    te_seqs, te_dates, te_labs, te_ids = process_sequences(test_seqs, test_dates)
    
    tra = (tr_seqs, tr_labs)
    tes = (te_seqs, te_labs)
    
    n_sequences = 0

    for seq in train_seqs:
        n_sequences += len(seq)
        
    for seq in test_seqs:
        n_sequences += len(seq)
        
    print('avg length: ', n_sequences/(len(train_seqs) + len(test_seqs) * 1.0))
    
    if not os.path.exists('yoochoose1_4'):
        os.makedirs('yoochoose1_4')
    if not os.path.exists('yoochoose1_64'):
        os.makedirs('yoochoose1_64')
        
    pickle.dump(tes, open('yoochoose1_4/test.txt', 'wb'))
    pickle.dump(tes, open('yoochoose1_64/test.txt', 'wb'))

    split4, split64 = int(len(tr_seqs) / 4), int(len(tr_seqs) / 64)

    tra4, tra64 = (tr_seqs[-split4:], tr_labs[-split4:]), (tr_seqs[-split64:], tr_labs[-split64:])
    seq4, seq64 = train_seqs[tr_ids[-split4]:], train_seqs[tr_ids[-split64]:]

    pickle.dump(tra4, open('yoochoose1_4/train.txt', 'wb'))
    pickle.dump(seq4, open('yoochoose1_4/all_train_seq.txt', 'wb'))

    pickle.dump(tra64, open('yoochoose1_64/train.txt', 'wb'))
    pickle.dump(seq64, open('yoochoose1_64/all_train_seq.txt', 'wb'))
