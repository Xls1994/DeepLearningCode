#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
from __future__ import print_function


def process():
    import util as c
    import numpy as np
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils.np_utils import to_categorical

    np.random.seed(1337)  # for reproducibility

    ######################################
    # 加载数据
    ######################################

    MAX_SEQUENCE_LENGTH = 10  # 每个新闻文本最多保留80个词
    MAX_NB_WORDS = 22353  # 字典大小
    EMBEDDING_DIM = 100  # 词向量的维度

    embedding_file = '../data/Embedding/word2vec_100_dim.txt'
    F_LABEL_DIR = r'../data/Labels/fscope/'
    L_LABEL_DIR = r'../data/Labels/lscope/'
    DATA_DIR = r'../data/corpus/'

    '''
    数据预处理
    1、 遍历语料文件下的所有文件
    '''
    print('Processing text dataset')

    X_Ftrain = c.read_file(DATA_DIR + 'Ftrain.context')
    X_Ftest = c.read_file(DATA_DIR + 'Ftest.context')
    X_Ltrain = c.read_file(DATA_DIR + 'Ltrain.context')
    X_Ltest = c.read_file(DATA_DIR + 'Ltest.context')

    Y_Ftest = c.read_int_file(F_LABEL_DIR + 'Ftestlabel.txt')
    Y_Ftrain = c.read_int_file(F_LABEL_DIR + 'Ftrainlabel.txt')
    Y_Ltest = c.read_int_file(L_LABEL_DIR + 'Ltestlabel.txt')
    Y_Ltrain = c.read_int_file(L_LABEL_DIR + 'Ltrainlabel.txt')

    print('Found %s texts.' % len(X_Ftrain))  # Found ? texts.
    print('Found %s labels.' % len(Y_Ftrain))  # Found ? labels.

    '''
    2、之后，我们可以新闻样本转化为神经网络训练所用的张量。
    所用到的Keras库是keras.preprocessing.text.Tokenizer和keras.preprocessing.sequence.pad_sequences
    '''
    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(X_Ftrain)
    sequences = tokenizer.texts_to_sequences(X_Ftrain)

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)  # 填充序列，不足的补0

    # 将类别向量(从0到nb_classes的整数向量)映射为二值类别矩阵, 用于应用到以categorical_crossentropy为目标函数的模型中.
    labels = to_categorical(np.asarray(Y_Ftrain), 2)
    (x_train, y_train) = data, labels


    # 测试用
    tokenizer_test = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(X_Ftest)
    sequences = tokenizer.texts_to_sequences(X_Ftest)

    data_t = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)  # 填充序列，不足的补0
    labels_t = to_categorical(np.asarray(Y_Ftest), 2)
    (x_test, y_test) = data_t, labels_t

    print('Preparing embedding matrix.')

    '''
    3、接下来，我们从embedding文件中解析出每个词和它所对应的词向量，并用字典的方式存储embeddings_index
    '''
    print('Indexing word vectors.')

    embeddings_index = {}
    f = open(embedding_file)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    '''
    4、此时，我们可以根据得到的字典生成上文所定义的词向量矩阵
    '''

    import numpy as np

    content = []
    with open(DATA_DIR + 'Ltrain.context') as f:
        for line in f:
            content.append(line)
    with open(DATA_DIR + 'Ltest.context') as f:
        for line in f:
            content.append(line)

    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(content)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.uniform(-1, 1, size=100)  # 生成长度为100维，在[-1,1)之间平均分布的随机数组

    return (x_train, y_train), (x_test, y_test), embedding_matrix, nb_words