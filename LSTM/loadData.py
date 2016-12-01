import random, operator
import  numpy as np
from theano import config

def build_vocab(trainPath,testPath):
    code, vocab = int(0), {}
    vocab['UNKNOWN'] = code
    code += 1
    for line in open(trainPath):
        items = line.strip().split(' ')
        for word in items:
            if len(word) <= 0:
                continue
            if not word in vocab:
                vocab[word] = code
                code += 1
    for line in open(testPath):
        items = line.strip().split(' ')
        for word in items:
            if len(word) <= 0:
                continue
            if not word in vocab:
                vocab[word] = code
                code += 1
    return vocab

def load_vectors(emPath,dim):
    vectors = {}
    for line in open(emPath):
        items = line.strip().split(' ')
        if len(items[0]) <= 0:
            continue
        vec = []
        for i in range(1, dim+1):
            vec.append(float(items[i]))
        vectors[items[0]] = vec
    return vectors

def load_word_embeddings(vocab, dim):
    vectors = load_vectors()
    embeddings = [] #brute initialization
    for i in range(0, len(vocab)):
        vec =np.random.uniform(-0.25,0.25,dim)
        embeddings.append(vec)
    for word, code in vocab.items():
        if word in vectors:
            embeddings[code] = vectors[word]
    return np.array(embeddings, dtype='float32')

def load_word_idx(vocab,path):
    wordList =[]
    wordItems = []
    for line in open(path):
        items = line.strip().split(' ')
        for word in items:
            wordItems.append(vocab[word])
        wordList.append(wordItems)
    return wordList

def prepare_data(seqs, labels, maxlen=None):

    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples)).astype('int32')
    x_mask = np.zeros((maxlen, n_samples)).astype(config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels
def load_data(path="imdb.pkl", n_words=100000, valid_portion=0.1, maxlen=None,
              sort_by_len=True):
    import cPickle
    f =open(path,'rb')
    train_set = cPickle.load(f)
    test_set = cPickle.load(f)
    f.close()

    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

    # split training set into validation set
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.random.permutation(n_samples)
    n_train =int (np.round(n_samples ))
    # n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    # valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    # valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y)
    # valid_set = (valid_set_x, valid_set_y)

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    test_set_x, test_set_y = test_set
    # valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    train_set_x = remove_unk(train_set_x)
    # valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        # sorted_index = len_argsort(valid_set_x)
        # valid_set_x = [valid_set_x[i] for i in sorted_index]
        # valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    # valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)

    return train, test

# def load_data(trainList, vocab, batch_size):
#     train_1, train_2, train_3 = [], [], []
#     mask_1, mask_2, mask_3 = [], [], []
#     counter = 0
#     while True:
#         pos = trainList[random.randint(0, len(trainList)-1)]
#         neg = trainList[random.randint(0, len(trainList)-1)]
#         if pos[2].startswith('<a>') or pos[3].startswith('<a>') or neg[3].startswith('<a>'):
#             #print 'empty string ......'
#             continue
#         x, m = encode_sent(vocab, pos[2], 100)
#         train_1.append(x)
#         mask_1.append(m)
#         x, m = encode_sent(vocab, pos[3], 100)
#         train_2.append(x)
#         mask_2.append(m)
#         x, m = encode_sent(vocab, neg[3], 100)
#         train_3.append(x)
#         mask_3.append(m)
#         counter += 1
#         if counter >= batch_size:
#             break
#     return np.transpose(np.array(train_1, dtype=config.floatX)), np.transpose(np.array(train_2, dtype=config.floatX)), np.transpose(np.array(train_3, dtype=config.floatX)), np.transpose(np.array(mask_1, dtype=config.floatX)) , np.transpose(np.array(mask_2, dtype=config.floatX)), np.transpose(np.array(mask_3, dtype=config.floatX))
#
# def load_data_val(testList, vocab, index, batch_size):
#     x1, x2, x3, m1, m2, m3 = [], [], [], [], [], []
#     for i in range(0, batch_size):
#         true_index = index + i
#         if true_index >= len(testList):
#             true_index = len(testList) - 1
#         items = testList[true_index]
#         x, m = encode_sent(vocab, items[2], 100)
#         x1.append(x)
#         m1.append(m)
#         x, m = encode_sent(vocab, items[3], 100)
#         x2.append(x)
#         m2.append(m)
#         x, m = encode_sent(vocab, items[3], 100)
#         x3.append(x)
#         m3.append(m)
#     return np.transpose(np.array(x1, dtype=config.floatX)), np.transpose(np.array(x2, dtype=config.floatX)), np.transpose(np.array(x3, dtype=config.floatX)), np.transpose(np.array(m1, dtype=config.floatX)) , np.transpose(np.array(m2, dtype=config.floatX)), np.transpose(np.array(m3, dtype=config.floatX))
#
# def validation(validate_model, testList, vocab, batch_size):
#     index, score_list = int(0), []
#     while True:
#         x1, x2, x3, m1, m2, m3 = load_data_val(testList, vocab, index, batch_size)
#         batch_scores, nouse = validate_model(x1, x2, x3, m1, m2, m3)
#         for score in batch_scores:
#             score_list.append(score)
#         index += batch_size
#         if index >= len(testList):
#             break
#         print 'Evaluation ' + str(index)
#     sdict, index = {}, int(0)
#     for items in testList:
#         qid = items[1].split(':')[1]
#         if not qid in sdict:
#             sdict[qid] = []
#         sdict[qid].append((score_list[index], items[0]))
#         index += 1
#     lev0, lev1 = float(0), float(0)
#     of = open('/QAcorpus/acc.lstm', 'a')
#     for qid, cases in sdict.items():
#         cases.sort(key=operator.itemgetter(0), reverse=True)
#         score, flag = cases[0]
#         if flag == '1':
#             lev1 += 1
#         if flag == '0':
#             lev0 += 1
#     for s in score_list:
#         of.write(str(s) + '\n')
#     of.write('lev1:' + str(lev1) + '\n')
#     of.write('lev0:' + str(lev0) + '\n')
#     print 'lev1:' + str(lev1)
#     print 'lev0:' + str(lev0)
#     of.close()

if __name__=='__main__':
    vocab =build_vocab('QAcorpus/Ftrain.cnn','QAcorpus/Ftest.cnn')
    with open('vocab.txt','w')as f:
        for i,word in vocab.items():
            f.write(str(i)+' '+str(word)+'\n')
    wordlists =load_word_idx(vocab,'QAcorpus/Ftrain.cnn')
    with open('wordidx.txt','w')as f:
        for line in wordlists:
            for item in line:
                f.write(str(item)+' ')
            f.write('\n')
