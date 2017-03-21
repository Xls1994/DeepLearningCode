import numpy as np
import pandas as pd
import codecs
# dataFrame =pd.read_excel('testData.xls')
def evalSpearman(path):
    dataFrame =pd.read_excel(path)
    print  dataFrame.head()
    print  dataFrame.corr('spearman')
    x =dataFrame[['score','similarity']]
    x.to_excel('spearman.xls')
    print x

def cosSim(A,B):
    num =A.dot(B)
    denom =np.linalg.norm(A) *np.linalg.norm(B)
    cos =num/denom
    sim =0.5 + 0.5 * cos
    return sim

def euclideanSim(A,B):
    dist = np.linalg.norm(A - B)
    sim =1.0/ (1.0+dist)
    return sim
# f =codecs.open('cilin.txt','r',encoding='gbk')
# l =codecs.open('cc','w',encoding='utf-8')
# for line in f:
#     print line
#     l.write(line)
# f.close()
# l.close()
if __name__=='__main__':
    # path ='result.xlsx'
    # evalSpearman(path)
    A =np.asarray([1,2,3],dtype='int32')
    B = np.asarray([1,3,1],dtype='int32')
    sim =euclideanSim(A,B)
    sim2 =cosSim(A,B)
    print sim,sim2
    print np.dot(A,B)