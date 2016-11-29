import data
import numpy as np
batch_size=100
seqs, i2w, w2i, data_xy = data.word_sequence("/data/toy.txt", batch_size)
dim_x = len(w2i)
dim_y = len(w2i)
num_sents = data_xy[0][3]
print "#features = ", dim_x, "#labels = ", dim_y
print  np.shape(seqs)
print np.shape(data_xy[0])
print np.shape(data_xy[0][1])