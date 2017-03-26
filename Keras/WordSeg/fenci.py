# -*- coding:utf-8 -*-
import re
import numpy as np
import pandas as pd

# action='Predict'
action='Predict'

s = open('msr_train.txt').read().decode('gbk')
s = s.split('\r\n')
print 'load train set'
def clean(s): #处理一下数据，有些不规范的地方
    if u'“/s' not in s:
        return s.replace(u' ”/s', '')
    elif u'”/s' not in s:
        return s.replace(u'“/s ', '')
    elif u'‘/s' not in s:
        return s.replace(u' ’/s', '')
    elif u'’/s' not in s:
        return s.replace(u'‘/s ', '')
    else:
        return s

s = u''.join(map(clean, s))
s = re.split(u'[，。！？、]/[bems]', s)

data = [] #train set
label = []
def get_xy(s):
    s = re.findall('(.)/(.)', s)
    if s:
        s = np.array(s)
        return list(s[:,0]), list(s[:,1])

for i in s:
    x = get_xy(i)
    if x:
        data.append(x[0])
        label.append(x[1])
maxlen=32
d = pd.DataFrame(index=range(len(data)))
d['data'] = data
d['label'] = label
d = d[d['data'].apply(len) <= maxlen]
d.index = range(len(d))
tag = pd.Series({'s':0, 'b':1, 'm':2, 'e':3, 'x':4})


chars = [] #统计所有词和编号
for i in data:
    chars.extend(i)

chars = pd.Series(chars).value_counts()
chars[:] = range(1, len(chars)+1)

#生成实验数据
from keras.utils import np_utils
d['x'] = d['data'].apply(lambda x: np.array(list(chars[x])+[0]*(maxlen-len(x))))
d['y'] = d['label'].apply(lambda x: np.array(map(lambda y:np_utils.to_categorical(y,5), tag[x].values.reshape((-1,1)))+[np.array([[0,0,0,0,1]])]*(maxlen-len(x))))

#LSTM 模型 seq2seq
word_size = 128
maxlen = 32
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional
from keras.models import Model
 
sequence = Input(shape=(maxlen,), dtype='int32')
embedded = Embedding(len(chars)+1, word_size, input_length=maxlen, mask_zero=True)(sequence)
blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)
output = TimeDistributed(Dense(5, activation='softmax'))(blstm)
model = Model(input=sequence, output=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from keras.utils.visualize_util import plot

plot(model, to_file='model.png',show_shapes=True)
batch_size = 1024
weight_file ='model_weights.hdf5'
import os
if action=='Predict' and os.path.exists(weight_file):
    model.load_weights(weight_file)
else:
    history = model.fit(np.array(list(d['x'])), np.array(list(d['y'])).reshape((-1,maxlen,5)), batch_size=batch_size, nb_epoch=10)
    model.save_weights(weight_file)
    jsonfile =model.to_json()
    import json
    with open('model.json','w') as f:
        json.dump(jsonfile,f)
        print 'dump json file...'

#viterbi 转移概率，现在使用的是等概率
zy = {'be':0.5, 
      'bm':0.5, 
      'eb':0.5, 
      'es':0.5, 
      'me':0.5, 
      'mm':0.5,
      'sb':0.5, 
      'ss':0.5
     }
 
zy = {i:np.log(zy[i]) for i in zy.keys()}
 
def viterbi(nodes):
    paths = {'b':nodes[0]['b'], 's':nodes[0]['s']}
    for l in range(1,len(nodes)):
        paths_ = paths.copy()
        paths = {}
        for i in nodes[l].keys():
            nows = {}
            for j in paths_.keys():
                if j[-1]+i in zy.keys():
                    nows[j+i]= paths_[j]+nodes[l][i]+zy[j[-1]+i]
            k = np.argmax(nows.values())
            paths[nows.keys()[k]] = nows.values()[k]
    return paths.keys()[np.argmax(paths.values())]
 
def simple_cut(s):
    if s:
        r = model.predict(np.array([list(chars[list(s)].fillna(0).astype(int))+[0]*(maxlen-len(s))]), verbose=False)[0][:len(s)]
        r = np.log(r)
        nodes = [dict(zip(['s','b','m','e'], i[:4])) for i in r]
        t = viterbi(nodes)
        words = []
        for i in range(len(s)):
            if t[i] in ['s', 'b']:
                words.append(s[i])
            else:
                words[-1] += s[i]
        return words
    else:
        return []
 
not_cuts = re.compile(u'([\da-zA-Z ]+)|[。，、？！\.\?,!]')
def cut_word(s):
    result = []
    j = 0
    for i in not_cuts.finditer(s):
        result.extend(simple_cut(s[j:i.start()]))
        result.append(s[i.start():i.end()])
        j = i.end()
    result.extend(simple_cut(s[j:]))
    return result
if action=='Predict':
    print 'predict the model...'
    result =cut_word(u'我是一名小学生')
    print result
    print type(result[0])
    with open('result.txt','w')as f:
        for w in result:
            f.write(w.decode('gbk')+" ")
        f.write('\n')
