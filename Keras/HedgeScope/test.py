#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
from __future__ import print_function
import numpy as np
from process_data import make_idx_data_cv

np.random.seed(1337)

#####################################
# 加载数据
#####################################

# embedding_file = r'data/Embedding/word2vec_100_dim.txt'
# F_LABEL_DIR = r'data/Labels/fscope/'
# L_LABEL_DIR = r'data/Labels/lscope/'
# DATA_DIR = r'data/corpus/'
#
# MAX_SEQUENCE_LENGTH = 20  # 每个句子最多保留10个词
# MAX_NB_WORDS = 9870L  # 字典大小 22353
# EMBEDDING_DIM = 100  # 词向量的维度
# VALIDATION_SPLIT = 0.2  # 训练集:验证集 = 1:4
# BATCH_SIZE = 128
# EPOCH = 2  # 迭代次数
# LR = 0.01  # 学习率

# '''
# 数据预处理
# 1、 遍历语料文件下的所有文件
# '''
# print('Processing text dataset')
#
# X_Ftrain = c.read_file(DATA_DIR + 'Ftrain.context')
# X_Ftest = c.read_file(DATA_DIR + 'Ftest.context')
# X_Ltrain = c.read_file(DATA_DIR + 'Ltrain.context')
# X_Ltest = c.read_file(DATA_DIR + 'Ltest.context')
#
# Y_Ftest = c.read_int_file(F_LABEL_DIR + 'Ftestlabel.txt')
# Y_Ftrain = c.read_int_file(F_LABEL_DIR + 'Ftrainlabel.txt')
# Y_Ltest = c.read_int_file(L_LABEL_DIR + 'Ltestlabel.txt')
# Y_Ltrain = c.read_int_file(L_LABEL_DIR + 'Ltrainlabel.txt')
#
# print('Found %s texts.' % len(X_Ltrain))  # Found 47477 texts.
# print('Found %s labels.' % len(Y_Ltrain))  # Found 47477 labels.
#
# '''
# 2、之后，我们可以新闻样本转化为神经网络训练所用的张量。
# '''
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils.np_utils import to_categorical
#
# print('Vectorizing sequence data...')
#
# tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)  # nb_words 处理的最大单词数量
# tokenizer.fit_on_texts(X_Ltrain)
# sequences = tokenizer.texts_to_sequences(X_Ltrain)
#
# data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)  # 填充序列，不足的补0
# labels = to_categorical(np.asarray(Y_Ltrain), 2)
# (x_train, y_train) = data, labels
#
# '''测试用'''
# tokenizer_test = Tokenizer(nb_words=MAX_NB_WORDS)
# tokenizer_test.fit_on_texts(X_Ltest)
# sequences = tokenizer_test.texts_to_sequences(X_Ltest)
#
# data_t = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)  # 填充序列，不足的补0
# labels_t = to_categorical(np.asarray(Y_Ltest), 2)
# (x_test, y_test) = data_t, labels_t
#
#
# '''
# 3、接下来，我们从embedding文件中解析出每个词和它所对应的词向量，并用字典的方式存储embeddings_index
# '''
# print('Indexing word vectors.')
#
# embeddings_index = {}
# f = open(embedding_file)
# for line in f:
# values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()
#
# print('Found %s word vectors.' % len(embeddings_index))
#
# '''
# 4、此时，我们可以根据得到的字典生成上文所定义的词向量矩阵
# '''
# content = []
# for i in range(len(X_Ltrain)):
#     content.append(X_Ltrain[i])
# for i in range(len(X_Ltest)):
#     content.append(X_Ltrain[i])
# # with open(DATA_DIR + 'Ltrain.context') as f:
# #     for line in f:
# #         content.append(line.strip())
# # with open(DATA_DIR + 'Ltest.context') as f:
# #     for line in f:
# #         content.append(line.strip())
#
# tokenizer1 = Tokenizer(nb_words=MAX_NB_WORDS)
# tokenizer1.fit_on_texts(content)
# sequences = tokenizer1.texts_to_sequences(content)
#
# word_index = tokenizer1.word_index  # word_index 是一个字典（键值对）
# print('Found %s unique tokens...' % len(word_index))  # 9526
#
# nb_words = min(MAX_NB_WORDS, len(word_index))
# embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
# for word, i in word_index.items():  # i从1开始,按出现次数来排序:。，的 blank
#     if i > MAX_NB_WORDS:
#         continue
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector
#     else:
#         embedding_matrix[i] = np.random.uniform(-0.25, 0.25, size=100)  # 生成长度为100维，在[-1,1)之间平均分布的随机数组


from keras.models import load_model
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
import cPickle


def loadData(path):
	x = cPickle.load(open(path, "rb"))
	revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
	print(len(word_idx_map))
	print(len(vocab))
	datasets = make_idx_data_cv(revs, word_idx_map, 1, max_l=10, k=100, filter_h=1)
	img_h = len(datasets[0][0]) - 1
	test_set_x = datasets[1][:, :img_h]
	test_set_y = np.asarray(datasets[1][:, -1], "int32")
	train_set_x = datasets[0][:, :img_h]
	train_set_y = np.asarray(datasets[0][:, -1], "int32")
	print(np.shape(train_set_x))
	print('load data...')
	print(np.shape(W))
	print(type(W))
	return (train_set_x, train_set_y), (test_set_x, test_set_y), W

def preditFval(predictions, test_label):
	num = len(predictions)
	with open(RESULT_FILE, 'w') as f:
		for i in range(num):
			if predictions[i][1] > predictions[i][0]:
				predict = +1
			else:
				predict = -1
			f.write(str(predictions[i][0]) + ' ' + str(predictions[i][1]) + '\n')
		# f.write(str(predict) + str(predictions[i]) + '\n')

	TP = len([1 for i in range(num) if
			  predictions[i][1] > predictions[i][0] and (test_label[i] == np.asarray([0, 1])).all()])
	FP = len([1 for i in range(num) if
			  predictions[i][1] > predictions[i][0] and (test_label[i] == np.asarray([1, 0])).all()])
	FN = len([1 for i in range(num) if
			  predictions[i][1] < predictions[i][0] and (test_label[i] == np.asarray([0, 1])).all()])
	TN = len([1 for i in range(num) if
			  predictions[i][1] < predictions[i][0] and (test_label[i] == np.asarray([1, 0])).all()])

	precision = recall = Fscore = 0, 0, 0
	try:
		precision = TP / (float)(TP + FP)  # ZeroDivisionError: float division by zero
		recall = TP / (float)(TP + FN)
		Fscore = (2 * precision * recall) / (precision + recall)
	except ZeroDivisionError as exc:
		print(exc.message)
	finally:
		print('Wether match? ', (TP + FP + FN + TN) == num)
		print(TP, FP, FN, TN)  # 0 0 1875 9803

	print(">> Report the result ...")
	print("-1 --> ", len([1 for i in range(num) if predictions[i][1] < predictions[i][0]]))
	print("+1 --> ", len([1 for i in range(num) if predictions[i][1] > predictions[i][0]]))
	print("TP=", TP, "  FP=", FP, " FN=", FN, " TN=", TN)
	print("precision= ", precision)
	print("recall= ", recall)
	print("Fscore= ", Fscore)

if __name__ == '__main__':

	RESULT_FILE = 'L_predict_result.txt'
	DATA_PATH = 'data/mr_Lscope.p'
	MAX_NB_WORDS = 200000
	MAX_SEQUENCE_LENGTH = 10  # maxlen

	(X_train, y_train), (X_test, y_test), word_embedding_matrix = loadData(DATA_PATH)
	print(len(X_train), 'train sequences')
	print(len(X_test), 'test sequences')

	nb_words = min(MAX_NB_WORDS, len(word_embedding_matrix))

	#   5-Prepare training data tensors
	print('Pad sequences (samples x time)')
	X_train = sequence.pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
	X_test = sequence.pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
	print('X_train shape:', X_train.shape)
	print('X_test shape:', X_test.shape)
	train_label = to_categorical(y_train, 2)
	test_label = to_categorical(y_test, 2)

	model = load_model('BiLSTM_maxpool4_fweights.h5')
	# history = model.history # 'Sequential' object has no attribute 'history'

	predictions = model.predict(X_test)
	preditFval(predictions, test_label)
