# -*- encoding: utf-8 -*-
'''
autuor: yangyl
@site: 
@software: PyCharm Community Edition
@file: w2v_proc.py
@time: 2017/8/10 21:33
'''
import numpy as np
class Writer():
    def __init__(self,file_path):
        self.file_path=file_path

    def init(self,path):
        self.writer=open(path,'w')
    def close(self):
        self.writer.close()

    def extract_entity_id(self):
        vec={}
        for files in self.file_path:
            self.reader=open(files,'r')
            for line in self.reader:
                if line.find("Entity:")!=-1:
                    entity=line.strip().split("\t")
                    if entity[3] not in vec:
                        vec[entity[3]]=entity[5]
                        self.writer.write(entity[3]+"@@"+entity[5]+"@@"+entity[4]+"\n")

    def check_id(self,file_path,id_path):
        vec ={}
        with open(file_path,'r')as f:
            for line in f:
                line =line.strip()
                entity =line.split(' ')
                wordem =' '.join(entity[1:])
                if entity[0] not in vec:
                    word =entity[0]
                    word =word.split("_")[0]
                    vec[word]=wordem
        with open(id_path,'r')as f:
            for line in f:
                line =line.strip()
                word =line.split("@@")[1]
                if vec.get(word):
                    self.writer.write(word+" "+vec.get(word)+"\n")
                else:
                    em =np.random.uniform(-0.25,0.25,100)
                    self.writer.write(word+" ")
                    for i in em.tolist():
                        self.writer.write(str(i)+" ")
                    self.writer.write("\n")
                    print word
def load_w2v(file_path,output_path):
    # 去除
    writer =open(output_path,'w')

    with open(file_path,'r')as f:
        for line in f:
            line =line.strip()
            if line.find("_Chemical")!=-1 or line.find("_Disease")!=-1:
                writer.write(line+"\n")
    writer.close()
    pass


if __name__ == '__main__':

    # load_w2v('w2v/vec_cbow_100.vec','w2v/output.vec')
    wr =Writer(['corpus/CDR_TrainSentence.txt','corpus/CDR_TestSentence.txt'])
    wr.init('w2v/entity_embedding.vec')
    # wr.extract_entity_id()
    wr.check_id('w2v/output.vec','id_fiter.txt')

    wr.close()
    pass