# -*- encoding: utf-8 -*-


from __future__ import print_function
import numpy as np
np.random.seed(1337)

import cPickle  # import pickle as cPickle
import model

from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import load_model
from process_data import make_idx_data_cv
import time

MAX_NB_WORDS = 200000  # 10650L    9870L	max_features
MAX_SEQUENCE_LENGTH = 10  # maxlen
EMBEDDING_DIM = 100
MODEL_WEIGHTS_FILE = 'weights.h5'
model_path = 'LSTM_3_fweights.h5'
RESULT_FILE = 'LSTM_3_F_predict_result.txt'
DATA_PATH = 'data/mr_Fscope.p'

# Training
VALIDATION_SPLIT = 0.2
NB_EPOCHS = 20
BATCH_SIZE = 64

# Convolution
FILTER_LENGTH = [2, 3, 5]
NB_FILTER = 100
POOL_LENGTH = 7


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
	#   4-Prepare word embedding matrix
	print('\nLoading data...')

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
	print('Build model...')

	# 建模
	# model = model.buildCNN(MAX_SEQUENCE_LENGTH, nb_words,
	# 					   word_embedding_matrix, FILTER_LENGTH, NB_FILTER)
	# model = model.buildLstmCnn(MAX_SEQUENCE_LENGTH, nb_words,
	# 					   word_embedding_matrix, FILTER_LENGTH, NB_FILTER)
	# model = model.buildCnnLSTM(MAX_SEQUENCE_LENGTH, nb_words,
	# 						   word_embedding_matrix, NB_FILTER)
	# model = model.buildLstmPool(nb_words, word_embedding_matrix ,MAX_SEQUENCE_LENGTH)
	model = model.LSTM3(nb_words, word_embedding_matrix, MAX_SEQUENCE_LENGTH)
	# model = model.BiLSTM(nb_words, word_embedding_matrix, MAX_SEQUENCE_LENGTH)
	# model = model.BiLstmPool(nb_words, word_embedding_matrix, MAX_SEQUENCE_LENGTH, POOL_LENGTH)

	model.compile(loss='categorical_crossentropy', optimizer='adagrad',  # adam
				  metrics=['accuracy'])
	model.summary()  # 打印出模型概况
	callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE,
								 monitor='val_acc', save_best_only=True)]

	t0 = time.time()
	history = model.fit(X_train, train_label,
						batch_size=BATCH_SIZE,
						verbose=1,
						validation_split=VALIDATION_SPLIT, # (X_test, test_label)
						callbacks=callbacks,
						nb_epoch=NB_EPOCHS)
	t1 = time.time()
	print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

	# 将模型和权重保存到指定路径
	model.save(model_path)
	# 加载权重到当前模型
	# model = load_model(model_path)

	# Print best validation accuracy and epoch in valid_set
	max_val_acc, idx = max((val, idx) for (idx, val) in enumerate(history.history['val_acc']))
	print('Maximum accuracy at epoch', '{:d}'.format(idx + 1), '=', '{:.4f}'.format(max_val_acc))

	# plot the result
	import matplotlib.pyplot as plt

	plt.figure()
	plt.plot(history.epoch, history.history['acc'], label="acc")
	plt.plot(history.epoch, history.history['val_acc'], label="val_acc")
	plt.scatter(history.epoch, history.history['acc'], marker='*')
	plt.scatter(history.epoch, history.history['val_acc'])
	plt.legend(loc='lower right')
	plt.show()

	plt.figure()
	plt.plot(history.epoch, history.history['loss'], label="loss")
	plt.plot(history.epoch, history.history['val_loss'], label="val_loss")
	plt.scatter(history.epoch, history.history['loss'], marker='*')
	plt.scatter(history.epoch, history.history['val_loss'], marker='*')
	plt.legend(loc='lower right')
	plt.show()

	score, acc = model.evaluate(X_test, test_label, batch_size=BATCH_SIZE)
	print('Test score:', score)
	print('Test accuracy:', acc)

	predictions = model.predict(X_test)
	preditFval(predictions, test_label)
