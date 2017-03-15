__author__ = 'yangyl'
#LSTM for named entity recognition
import numpy as np
max_doc_len =30
max_word_len=15
def pos(tag):
    onehot =np.zeros(5)
    if tag=='NN' or tag=='NNS':
        onehot[0]=1
    elif tag=='FW':
        onehot[1]=1
    elif tag=='NNP' or tag=='NNPS':
        onehot[2]=1
    elif 'VB' in tag:
        onehot[3]=1
    else:
        onehot[4]=1
    return onehot
def chunk(tag):
    onehot = np.zeros(5)
    if 'NP' in tag:
        onehot[0] = 1
    elif 'VP' in tag:
        onehot[1] = 1
    elif 'PP' in tag:
        onehot[2] = 1
    elif tag == 'O':
        onehot[3] = 1
    else:
        onehot[4] = 1
    return onehot
def captial(word):
    if ord(word[0])>='A' and ord(word[0])<='Z':
        return np.asarray([1])
    else:
        return np.asarray([0])
def sentence_to_num(sentence,dicts={}):
    num_array=[]
    values =[]
    for s in sentence:
        for w in s:
            value =dicts.get(w)
            values.append(value)
        num_array.append(values)
        values=[]
    return np.asarray(num_array)

def loadfile(filename):
    word =[]
    sentence =[]
    sentence_tag=[]
    tag =[]
    sentence_len=0
    for line in open(filename):
        temp =line.strip().split()
        if line in ['\n','\r\n']:
            for _ in range(max_doc_len-sentence_len):
                tag.append(np.asarray([0,0,0,0,0]))
                word.append('EOS')
            sentence.append(word)
            sentence_tag.append(np.asarray(tag))
            word=[]
            tag=[]
            sentence_len=0
        else:
            if sentence_len>=max_doc_len:
                continue
            sentence_len+=1
            t =temp[3]
            if t.endswith('O'):
                tag.append(np.asarray([1, 0, 0, 0, 0]))
            elif t.endswith('PER'):
                tag.append(np.asarray([0, 1, 0, 0, 0]))
            elif t.endswith('LOC'):
                tag.append(np.asarray([0, 0, 1, 0, 0]))
            elif t.endswith('ORG'):
                tag.append(np.asarray([0, 0, 0, 1, 0]))
            elif t.endswith('MISC'):
                tag.append(np.asarray([0, 0, 0, 0, 1]))
            else:
                print("error in input"+str(t))
            word.append(temp[0])
    # with open('sentence.txt','w')as f:
    #     for s in sentence:
    #         f.write(str(s)+'\n')
    print len(sentence_tag)
    print len(sentence_tag[0])
    worddict ={}
    worddict['EOS']=0
    index =1
    for s in sentence:
        for w in s:
            if w not in worddict:
                worddict[w]=index
                index+=1
    # with open('sentag.txt','w')as f:
    #     for key,value in worddict.items():
    #         f.write(str(key)+" "+str(value)+'\n')
    sentenceNum =sentence_to_num(sentence,worddict)
    # np.savetxt('ff.txt',sentenceNum,fmt='%.1i',delimiter=' ')
    return sentenceNum,sentence_tag
def buildModel():
    from keras.layers import LSTM,Dense,Embedding,Input
    from keras.layers import TimeDistributed,Bidirectional
    from keras.models import Model
    sequence =Input(shape=(30,),dtype='int32')
    em =Embedding(23015,100,mask_zero=True)(sequence)
    out =Bidirectional(LSTM(50,return_sequences=True),merge_mode='sum')(em)
    out =TimeDistributed(Dense(5,activation='softmax'))(out)
    model =Model(input=sequence,output=out)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adagrad',
                  metrics=['accuracy'])
    model.summary()
    from keras.utils.visualize_util import plot
    plot(model,to_file='model.png')

    return model

def f1(prediction,target,num_classes,max_doc_len): # not tensors but result values

    tp=np.asarray([0]*(num_classes+2))
    fp=np.asarray([0]*(num_classes+2))
    fn=np.asarray([0]*(num_classes+2))

    # target = np.argmax(target, 2)
    # prediction = np.argmax(prediction, 2)


    for i in range(len(target)):
        for j in range(max_doc_len):
            if target[i][j] == prediction[i][j]:
                tp[target[i][j]] += 1
            else:
                fp[target[i][j]] += 1
                fn[prediction[i][j]] += 1

    NON_NAMED_ENTITY = 0
    for i in range(num_classes):
        if i != NON_NAMED_ENTITY:
            tp[5] += tp[i]
            fp[5] += fp[i]
            fn[5] += fn[i]
        else:
            tp[6] += tp[i]
            fp[6] += fp[i]
            fn[6] += fn[i]

    precision = []
    recall = []
    fscore = []
    for i in range(num_classes+2):
        precision.append(tp[i]*1.0/(tp[i]+fp[i]))
        recall.append(tp[i]*1.0/(tp[i]+ fn[i]))
        fscore.append(2.0*precision[i]*recall[i]/(precision[i]+recall[i]))

    print("precision = " ,precision)
    print("recall = " ,recall)
    print("f1score = " ,fscore)
    efs = fscore[5]
    print("Entity fscore :", efs )
    del precision
    del recall
    del fscore
    return efs



if __name__=='__main__':
    Mode='train'
    f ='ner/eng.train'
    train,label =loadfile(f)
    # label =np.asarray(label).astype(int).reshape(len(label),-1)
    label =np.asarray(label).astype(int)
    model =buildModel()
    if Mode=='train':
    # model.load_weights('weights.hdf5')
        model.fit(train,label)
        model.save_weights('weights.hdf5')
    elif Mode=='test':
        model.load_weights('weights.hdf5')
        pred =model.predict(train)
        pred =np.argmax(pred,axis=2)
        np.savetxt('result.txt',pred,fmt='%.1i',delimiter=' ')
    # test =np.asarray([[15,16, 17 ,18, 19 ,20 ,21 ,22 ,23, 4 ,24 ,6, 25 ,6, 26, 8 ,9 ,27, 28, 29, 30 ,31, 32, 33 ,34, 35, 36 ,6 ,37, 10]])
    # h=model.predict(test)
    # print(type(h))
    # print (np.shape(h))
    # arr =np.argmax(h,axis=2)
    # print arr
    # print arr.shape
    # print arr[test>4]
