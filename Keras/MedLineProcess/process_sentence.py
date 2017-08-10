# -*- encoding: utf-8 -*-
'''
autuor: yangyl
@site: 
@software: PyCharm Community Edition
@file: process_sentence.py
@time: 2017/8/10 15:20
从已经标注好单个句子的语料中抽取CDR关系的样例
'''
from collections import namedtuple
Entity=namedtuple("Entity",'docId start end name type id')
# Relation=namedtuple("Relation",'docId type cstart cend cid dstart dend did polar')
Sentence =namedtuple("Sentence",'text')
class Relation(object):
    def __init__(self,docId,type,cstart,cend,cid,dstart,dend,did,polar,cIndex,dIndex):
        self.docId =docId
        self.type=type
        self.cstart =cstart
        self.cend=cend
        self.cid=cid
        self.dstart=dstart
        self.dend =dend
        self.did=did
        self.polar=polar
        self.cIndex=cIndex
        self.dIndex=dIndex
    def print_relation(self):
        print "CID:{0} DID {1} cIndex {2} dIndex {3}".format(self.cid,self.did,self.cIndex,self.dIndex)
class SenReader(object):
    def __init__(self, sentence_path, gdep_path):
        self.stream = open(sentence_path, 'r')
        self.gdep_stream=open(gdep_path,'r')
    def NextDoc(self):
        entities = []
        relations =[]
        gdeps =[]
        text=None
        for line in self.stream:
            line = line.strip()
            if line == "":
                break

            tempDoc = self.trySen(line)
            if tempDoc:
                text=tempDoc
            entity = self.tryEntity(line)

            if entity:
                entities.append(entity)

            relation =self.tryRelation(line)
            if relation:
                relations.append(relation)
        for line in self.gdep_stream:
            line =line.strip()
            if line =="":
                break
            temp =line.split("\t")
            gdeps.append(temp[1])
        return text, entities,relations,gdeps

    def Close(self):
        self.stream.close()
        self.gdep_stream.close()

    def trySen(self, s):
        index =s.find("Sentence:")
        if index != -1:
            text=s[index+10: ]
            return Sentence(text)
        else:
            return None
    def tryEntity(self, s):
        items = s.split("\t")
        if len(items) == 6:
            items[0]=items[0].replace("Entity: ","")

            return Entity(items[0], int(items[1]), int(items[2]), items[3], items[4], items[5])
        else:
            return None
    def tryRelation(self,s):
        items =s.split("\t")
        if len(items)==9:
            items[0]=items[0].replace("Relation: ","")
            return Relation(items[0],items[1],
                            int(items[2]),int(items[3]),
                            items[4],int(items[5]),int(items[6]),
                            items[7],items[8],0,0)
        else:
            return  None

def add_offset(sentence,gdep_phrase):
    pos =0
    position=[]
    for i in gdep_phrase:
        token=i
        pos =sentence.find(token,pos)
        if pos==-1:
            print token
            print "error"
            print sentence
        begin_off =pos
        end_off =pos+len(token)
        position.append((token,begin_off,end_off,end_off-1))
        pos=end_off
    return position

def extract_word_seq(relation,gdep,trainIns):
    #extract win=3 word


        chIndex =relation.cIndex
        dsIndex =relation.dIndex
        trainIns.write(relation.polar+'|')
        if chIndex>dsIndex:
            chIndex,dsIndex =dsIndex,chIndex
        if chIndex==0:
            while chIndex<=dsIndex:
                trainIns.write(gdep[chIndex]+"|")
                chIndex+=1
        elif chIndex==1:
            chIndex =chIndex-1
            while chIndex<=dsIndex:
                trainIns.write(gdep[chIndex]+"|")
                chIndex+=1
        elif chIndex==2:
            chIndex =chIndex-2
            while chIndex<=dsIndex:
                trainIns.write(gdep[chIndex]+"|")
                chIndex+=1
        else:
            chIndex =chIndex-3
            while chIndex<=dsIndex:
                trainIns.write(gdep[chIndex]+"|")
                chIndex+=1
        senLen=len(gdep)
        if dsIndex<senLen-3:
            trainIns.write(gdep[dsIndex+1]+"|"
                           +gdep[dsIndex+2]+"|"
                           +gdep[dsIndex+3]+"|")
        elif dsIndex==senLen-3:
            trainIns.write(gdep[dsIndex+1]+"|"
                           +gdep[dsIndex+2]+"|"
                           )
        elif dsIndex==senLen-2:
            trainIns.write(gdep[dsIndex+1]+"|")
        trainIns.write("\n")

def extract_word_index(relation,stream):
    chIndex = relation.cIndex
    dsIndex = relation.dIndex
    stream.write(str(chIndex)+" "+str(dsIndex)+"\n")



if __name__ == '__main__':
    rawCorpus = 'corpus/CDR_TrainSentence.txt'
    gdepCorpus = 'corpus/train.gdep'
    cleanCorpus ='corpus/train.clean'
    cleanCorpusStream = open(cleanCorpus, 'w')
    indexStream=open('corpus/index.txt','w')
    senReader = SenReader(rawCorpus,gdepCorpus)
    iters_num = 1
    while True:

        text, entities, relations,gdeps = senReader.NextDoc()
        if not text:
            break
        position=add_offset(text.text,gdeps)
        # print relations

        # print position
        for relation in relations:
            chemBegin =relation.cstart
            chemEnd=relation.cend
            diseaseBegin=relation.dstart
            diseaseEnd =relation.dend

            chemflag =False
            diseaseflag=False
            for index,pos in enumerate(position):
                begin=pos[1]
                end =pos[2]

                if begin==chemBegin or end ==chemEnd:
                    relation.cIndex=index
                    chemflag=True
                if begin==diseaseBegin or end ==diseaseEnd:
                    relation.dIndex =index
                    diseaseflag=True
                if diseaseflag and chemflag:
                    break

            if not(chemflag and diseaseflag):
                print relation
        for relation in relations:
            relation.print_relation()
        # print relations
            extract_word_seq(relation,gdeps,cleanCorpusStream)
            extract_word_index(relation,indexStream)
    senReader.Close()
    cleanCorpusStream.close()
    indexStream.close()



    pass