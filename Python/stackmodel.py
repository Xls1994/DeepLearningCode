# -*- coding:utf-8 -*-
'''
author: yangyl
@version: v1.0
@license: Apache Licence
@site:  
@software: PyCharm Community Edition 
@file: stackmodel.py 
@time: 2017/4/30 19:46
model average
'''
import matplotlib.pyplot as plt
import matplotlib.pylab as pyl
import numpy as np
from sklearn import datasets
from sklearn.model_selection import StratifiedShuffleSplit,train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

##所有的分类器模型,采用默认参数
classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    SVC(probability=True),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(),

    ]
dataset =datasets.load_iris()
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)#生成十组训练集和测试集，每组测试集为1/10
x = dataset.data
y = dataset.target
def signleModel(x,y):
    X_train,X_test,y_train,y_test =train_test_split(x,y,
                                                        random_state=35,
                                                        test_size=0.2)
    x1_test =np.zeros((X_test.shape[0],len(classifiers)))
    accuracy = np.zeros(len(classifiers))#每个模型的准确率

    for train_index, test_index in sss.split(X_train, y_train):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf_num = 0
        for clf in classifiers:
            clf_name = clf.__class__.__name__
            clf.fit(x_train, y_train)
            accuracy[clf_num] += (y_test == clf.predict(x_test)).mean()#该模型的准确率，十次平均
            clf_num += 1
    x1_test =x1_test/10
    accuracy = accuracy / 10
    # plt.bar(np.arange(len(classifiers)), accuracy, width=0.5, color='b')
    # plt.xlabel('Alog')
    # plt.ylabel('Accuracy')
    # plt.xticks(np.arange(len(classifiers)) + 0.25,
    #            ('KNN', 'DT', 'RF', 'SVC', 'AdaB', 'GBC', 'GNB',
    #             'LDA', 'QDA', 'LR', 'xgb'))
    # plt.show()

def bagging():
    X_train,X_test,y_train,y_test =train_test_split(x,y,
                                                        random_state=35,
                                                        test_size=0.2)
    x1_test =np.zeros((X_test.shape[0],len(classifiers)))#存储第一层测试集的输出结果
    accuracy = np.zeros(len(classifiers))#每个模型的准确率
    for train_index, test_index in sss.split(X_train, y_train):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf_num = 0
        for clf in classifiers:
            clf_name = clf.__class__.__name__
            clf.fit(x_train, y_train)
            x1_test[:, clf_num] += clf.predict(X_test)#直接对测试集进行预测，总共有十次，进行平均
            accuracy[clf_num] += (y_test == clf.predict(x_test)).mean()#该模型的准确率，十次平均
            clf_num += 1
        
    x1_test = x1_test / 10
    accuracy = accuracy / 10
    plt.bar(np.arange(len(classifiers)), accuracy, width=0.5, color='b')
    plt.xlabel('Alog')  
    plt.ylabel('Accuracy')  
    plt.xticks(np.arange(len(classifiers)) + 0.25, 
               ['KNN', 'DT', 'RF', 'SVC', 'AdaB', 'GBC', 'GNB',
                'LDA', 'QDA', 'LR', 'xgb'])
    
    pyl.pcolor(np.corrcoef(x1_test.T), cmap = 'Blues')
    pyl.colorbar() 
    pyl.xticks(np.arange(0.5, 11.5),
               ['KNN', 'DT', 'RF', 'SVC', 'AdaB', 'GBC', 'GNB','LDA', 'QDA', 'LR', 'xgb'])
    pyl.yticks(np.arange(0.5, 11.5),
               ['KNN', 'DT', 'RF', 'SVC', 'AdaB', 'GBC', 'GNB','LDA', 'QDA', 'LR', 'xgb'])
    pyl.show()
    import pandas as pd
    index = [0, 1, 2, 5, 9]
    linear_prediction = x1_test[:, index].mean(axis=1)
    print np.shape(linear_prediction)
    # linear_prediction[linear_prediction >= 0.5] = 1
    # linear_prediction[linear_prediction < 0.5] = 0
    # StackingSubmission = pd.DataFrame({
    #                             'type': linear_prediction.astype(int) })
    # linear_prediction.to_csv("linear_prediction.csv", index=False)

def stacking():
    X_train,X_test,Y_train,Y_test =train_test_split(x,y,
                                                        random_state=35,
                                                        test_size=0.2)
    x1_test =np.zeros((X_test.shape[0],len(classifiers)))#存储第一层测试集的输出结果
    x1_train =np.zeros((X_train.shape[0],len(classifiers)))
    print 'x1.shape',np.shape(x1_train)
    print 'y....',np.shape(Y_train)
    accuracy = np.zeros(len(classifiers))#每个模型的准确率
    for train_index, test_index in sss.split(X_train, Y_train):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf_num = 0
        for clf in classifiers:
            clf_name = clf.__class__.__name__
            clf.fit(x_train, y_train)
            x1_train[test_index,clf_num]=clf.predict(x_test)#下层模型的训练集输入是上层模型对于对应测试集的预测输出
            x1_test[:, clf_num] += clf.predict(X_test)#直接对测试集进行预测，总共有十次，进行平均
            accuracy[clf_num] += (y_test == x1_train[test_index,clf_num]).mean()#该模型的准确率，十次平均
            clf_num += 1


    print np.shape(x1_train)
    print np.shape(y_train)
    x2_train,x2_test,y2_train,y2_test =train_test_split(x1_train,Y_train,test_size=0.1)
    lr =LogisticRegression()
    lr.fit(x2_train,y2_train)
    print lr.predict(x1_test)
    print Y_test

    # plt.bar(np.arange(len(classifiers)), accuracy, width=0.5, color='b')
    # plt.xlabel('Alog')
    # plt.ylabel('Accuracy')
    # plt.xticks(np.arange(len(classifiers)) + 0.25,
    #            ['KNN', 'DT', 'RF', 'SVC', 'AdaB', 'GBC', 'GNB',
    #             'LDA', 'QDA', 'LR', 'xgb'])
    #
    # pyl.pcolor(np.corrcoef(x1_test.T), cmap = 'Blues')
    # pyl.colorbar()
    # pyl.xticks(np.arange(0.5, 11.5),
    #            ['KNN', 'DT', 'RF', 'SVC', 'AdaB', 'GBC', 'GNB','LDA', 'QDA', 'LR', 'xgb'])
    # pyl.yticks(np.arange(0.5, 11.5),
    #            ['KNN', 'DT', 'RF', 'SVC', 'AdaB', 'GBC', 'GNB','LDA', 'QDA', 'LR', 'xgb'])
    # pyl.show()
bagging()



if __name__ == "__main__":
    pass
