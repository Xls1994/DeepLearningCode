import numpy as np
def batch_iter(data, batch_size, num_epochs):
  """
  Generates a batch iterator for a dataset.
  """
  data = np.array(data)
  data_size = len(data)
  num_batches_per_epoch = int(len(data)/batch_size) + 1
  for epoch in range(num_epochs):
    # Shuffle the data at each epoch
    shuffle_indices = np.random.permutation(np.arange(data_size))
    shuffled_data = data[shuffle_indices]
    for batch_num in range(num_batches_per_epoch):
      start_index = batch_num * batch_size
      end_index = min((batch_num + 1) * batch_size, data_size)
      yield shuffled_data[start_index:end_index]

def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], inputs[excerpt],targets[excerpt]

X = np.arange(25).reshape((5,5))
y = np.asarray([[1],[2],[2],[3],[4]],dtype='int32')
z =np.asarray([1,2])
batch_size =4
a =zip(X,y)
if X.shape[0] % batch_size > 0:
    extra_data_num = batch_size - X.shape[0] % batch_size
    train_set = np.random.permutation(a)
    extra_data = train_set[:extra_data_num]
    new_data = np.append(a, extra_data, axis=0)
else:
    new_data =X
new_data = np.random.permutation(new_data)

print new_data
print len(new_data)
print np.shape(new_data)
a,b =zip(*new_data)
print type(a)
print np.shape(np.asarray(b))
# a =zip(X,y)
# print a
# print type(a)
# print len(a)
# batches =batch_iter(a,3,1)
# print batches
# for batch in batches:
#     x_batch, y_batch = zip(*batch)
#     print x_batch,'...',len(x_batch)
