import cPickle
import numpy as np
def loadFeatures(trainfilePath,testfilePath, wordembeddingPath, dictPath, dim):
    keysdict ={}
    value =0

    wordem =open(wordembeddingPath,'w')
    dict_w =open(dictPath,'w')
    for filepath in [trainfilePath,testfilePath]:
        with open(filepath, 'r') as f:
            for line in f:
                feaArray =line.strip().split("@@")
                if keysdict.has_key(feaArray[1])==False:
                    keysdict[feaArray[1]]=value
                    value +=1

    for items,val in keysdict.items():
        embeddings =np.random.uniform(-0.25, 0.25, dim)
        print 'items ',items,'val ',val,'embedddings ',embeddings
        wolist =list(embeddings)
        dict_w.write(items+'\n')
        l =''
        for w in wolist:
            l =str(w)+" "+l
        wordem.write(l.strip()+"\n")
    wordem.close()
    dict_w.close()

def makeidx_map(jsonfilepath):
    import json
    indexJson = open(jsonfilepath, "r")
    inputInfo = json.load(indexJson)
    indexJson.close()
    dictPath =inputInfo["dictPath"]
    trainPath=inputInfo["trainFea"]
    testPath =inputInfo["testFea"]
    fadicts={}
    newTri =trainPath.split('.')[0]+'new.txt'
    newTes =testPath.split('.')[0]+'new.txt'
    trainfile =open(newTri,'w')
    testfile=open(newTes,'w')
    trainData=[]
    testData=[]
    val =0
    with open(dictPath,'r')as f:
        for line in f:
            arg =line.strip()
            fadicts[arg]=val
            val +=1
    for items in fadicts.items():
        print items
    with open(trainPath,'r')as ftrain:
        for line in ftrain:
            arg =line.strip().split("@@")
            trainfile.write(str(fadicts.get(arg[1]))
                            +"\n")
            trainData.append([fadicts.get(arg[1])])
    with open(testPath,'r')as ftest:
        for line in ftest:
            arg =line.strip().split("@@")
            testfile.write(str(fadicts.get(arg[1]))
                            +"\n")
            testData.append([fadicts.get(arg[1])])
    trainfile.close()
    testfile.close()
    return trainData,testData








if __name__=="__main__":
    from cnn import loadData

    from keras.models import Sequential
    from keras.layers import Embedding,Dense,Activation,Flatten
    from keras.utils import np_utils
    import theano.sandbox.cuda


    def use_gpu(gpu_id):
        if gpu_id > -1:
            theano.sandbox.cuda.use("gpu" + str(gpu_id))


    # use_gpu(0)  # -1:cpu; 0,1,2,..: gpu
    trainfilePath ='corpus/crosssentence/ctd/train.feature'
    testfilePath="corpus/crosssentence/ctd/test.feature"
    wordembeddingPath ='corpus/crosssentence/ctd/ctdembeddings.txt'
    dictPath='corpus/crosssentence/ctd/dicts.txt'
    loadFeatures(trainfilePath,testfilePath,wordembeddingPath,dictPath,10)


    # xx =np.loadtxt('corpus/ctdembeddings.txt',delimiter=' ',dtype='float32')
    # print np.shape(xx)
    # (X_train, y_train), (X_test, y_test), WordEm = loadData(path='corpus/wordseq/mr.p')
    # y_train=np_utils.to_categorical(y_train,2)
    # y_test=np_utils.to_categorical(y_test,2)
    # jsonfile ='test.json'
    # train,test=makeidx_map(jsonfile)
    # print np.shape(train)
    # print type(train)
    # model=Sequential()
    # model.add(Embedding(4, 10, input_length=1))
    # model.add(Dense(10,input_dim=1))
    #
    # model.add(Dense(2))
    # model.add(Flatten())
    # model.add(Activation("softmax"))
    # model.summary()
    # model.compile(optimizer="adadelta",loss="categorical_crossentropy")
    # model.fit(train,y_train,batch_size=32)




