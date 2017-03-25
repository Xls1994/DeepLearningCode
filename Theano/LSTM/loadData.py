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

def load_word_embeddings(vocab, path,dim):
    vectors = load_vectors(path,dim)
    embeddings = [] #brute initialization
    for i in range(0, len(vocab)):
        vec =np.random.uniform(-0.25,0.25,dim)
        embeddings.append(vec)
    for word, code in vocab.items():
        if word in vectors:
            embeddings[code] = vectors[word]
    return np.array(embeddings, dtype='float32')

def load_word_idx(vocab,path,save='None'):
    wordList =[]
    maxlen =0
    for line in open(path):
        items = line.strip().split(' ')
        linelen =len(items)
        if linelen>maxlen:
            maxlen=linelen
        wordItems = []
        for word in items:
            wordItems.append(vocab[word])
        wordList.append(wordItems)
    if save!='None':
        with open(path+'_wordidx.txt','w')as f:
            for line in wordList:
                for item in line:
                    f.write(str(item)+' ')
                f.write('\n')
    print 'maxlen: ',maxlen
    return wordList

def loadLabel(path):
    labels =[]
    for line in open(path):
        line =line.strip()
        if line=='0':
            labels.append(0)
        else:
            labels.append(1)
    return labels
def dumpFiles(filePath,dimension):
    import  json
    import cPickle
    lstmJson = open(filePath, "r")
    inputInfo = json.load(lstmJson)
    lstmJson.close()
    trainPath =inputInfo["TraiContext"]
    testPath =inputInfo["TestContext"]
    trainLabelPath =inputInfo["TraiLabel"]
    testLabelPath =inputInfo["TestLabel"]
    emPath =inputInfo["WordVector"]
    outputPath=inputInfo["pklPath"]
    output_file =open(outputPath,'w')

    vocab =build_vocab(trainPath,testPath)
    embeddings =load_word_embeddings(vocab,emPath,dimension)
    print len(vocab)
    trainlabel =loadLabel(trainLabelPath)
    testlabel =loadLabel(testLabelPath)
    trainIdx =load_word_idx(vocab,trainPath)
    testIdx =load_word_idx(vocab,testPath)
    train_data = [trainIdx, trainlabel]
    test_data = [testIdx, testlabel]
    embeddings_data =[embeddings]
    cPickle.dump(train_data, output_file)
    cPickle.dump(test_data, output_file)
    cPickle.dump(embeddings_data,output_file)
    output_file.close()


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

def load_data_embeddings(path="imdb.pkl", n_words=100000, valid_portion=0.1, maxlen=None,
              sort_by_len=False):
    import cPickle
    f =open(path,'r')
    train_set = cPickle.load(f)
    test_set = cPickle.load(f)
    embeddings =cPickle.load(f)
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

    return train, test,embeddings[0]

def load_data(path="imdb.pkl", n_words=100000, valid_portion=0.1, maxlen=None,
              sort_by_len=False):
    import cPickle
    f =open(path,'r')
    train_set = cPickle.load(f)
    test_set = cPickle.load(f)
    # embeddings =cPickle.load(f)
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

if __name__=='__main__':
    # path ="corpus/mr_FscopeContexts.pkl"
    jsonPath ='Word.json'
    dumpFiles(jsonPath,240)
    import cPickle
    # f =open(path,'r')
    # train_set = cPickle.load(f)
    # test_set = cPickle.load(f)
    # embeddings =cPickle.load(f)
    # f.close()
    # np.savetxt('label.txt',train_set[1],fmt="%.2f",delimiter=' ')
    # np.savetxt('file.txt',train_set[0],fmt='%i',delimiter=' ')
    # # print max(train_set[0])
    # # print(max(test_set[1]))
    # # print( len(embeddings))
    # # print type(embeddings[0])
    pass

