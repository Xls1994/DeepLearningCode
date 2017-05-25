import numpy as np 
import re 
import csv
class Data(object):
    
    def __init__(self,
                 data_source,
                 alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
                 l0 = 1014,
                 batch_size = 128,
                 n_classes = 4):
        
        self.alphabet = alphabet
        # len:69
        self.alphabet_size = len(self.alphabet)
        self.dict = {}
        self.no_of_classes = n_classes
        for i, c in enumerate(self.alphabet):
            self.dict[c] = i + 1
        
        self.length = l0
        self.batch_size = batch_size
        self.data_source = data_source

    def loadData(self):
        data = []
        with open(self.data_source, 'rb') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            for row in rdr:

                txt = ""
                for s in row[1:]:
                    txt = txt + " " + re.sub("^\s*(.-)\s*$", "%1", s).replace("\\n", "\n")
                txt =txt.strip()

                data.append ((int(row[0]), txt))

        self.data = np.array(data)
        self.shuffled_data = self.data
        
    def shuffleData(self):
        np.random.seed(235)
        data_size = len(self.data)
        
        shuffle_indices = np.random.permutation(np.arange(data_size))
        self.shuffled_data = self.data[shuffle_indices]
        
    def getBatch(self, batch_num=0):
        data_size = len(self.data)
        start_index = batch_num * self.batch_size
        end_index = min((batch_num + 1) * self.batch_size, data_size)
        return self.shuffled_data[start_index:end_index]
        
    def getBatchToIndices(self, batch_num = 0):
        data_size = len(self.data)
        start_index = batch_num * self.batch_size
        end_index = data_size if self.batch_size == 0 else min((batch_num + 1) * self.batch_size, data_size)
        batch_texts = self.shuffled_data[start_index:end_index]
        batch_indices = []
        one_hot = np.eye(self.no_of_classes, dtype='int64')
        classes = []
        for c, s in batch_texts:
            batch_indices.append(self.strToIndexs(s))
            c = int(c) - 1
            classes.append(one_hot[c])
        return np.asarray(batch_indices, dtype='int64'), np.asarray(classes)


    def getAllData(self):
        data_size = len(self.data)
        start_index = 0
        end_index = data_size 
        batch_texts = self.data[start_index:end_index]
        batch_indices = []
        one_hot = np.eye(self.no_of_classes, dtype='int64')
        classes = []
        for c, s in batch_texts:
            batch_indices.append(self.strToIndexs(s))
            c = int(c) - 1
            classes.append(one_hot[c])
        return np.asarray(batch_indices, dtype='int64'), np.asarray(classes)

        
    def strToIndexs(self, s):
   
        s = s.lower()
        m = len(s)
        n = min(m, self.length)
        str2idx = np.zeros(self.length, dtype='int64')

        for i in range(1, n+1):
            c = s[-i]
            if c in self.dict:
                str2idx[i-1] = self.dict[c]


        return str2idx

    def getLength(self):
        return len(self.data)

    def saveVec(self,path):

        with open(path, "w") as fo:
            for i in range(self.getLength()):
                c = self.data[i][0]
                txt = self.data[i][1]
                vec = ",".join(map(str, data.strToIndexs(txt)))

                fo.write("{}\t{}\n".format(c, vec))


if __name__ == '__main__':
    data = Data('data/train.csv')
    data.loadData()
    # print data.data
    # print data.data[0][1]
    # print data.data[1][1]
    # print data.alphabet
    # print data.alphabet_size
    # data.saveVec('vec.vec')

        
    

        
            
        
        
        
