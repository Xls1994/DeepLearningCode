import  numpy as np
np.random.seed(1566)




if __name__=='__main__':
    batch_size =2
    train_x =np.arange(9).reshape((3,3))
    train_y =np.asarray([1,0,1])
    train_y_new =np.reshape(train_y,(len(train_y),1))
    print( type(train_y_new))
    print train_y_new.shape

    trainFile =np.concatenate((train_x,train_y_new),axis=1)
    print trainFile
    if np.shape(train_x)[0] % batch_size > 0:
        extra_data_num = batch_size - np.shape(train_x)[0] % batch_size
        print extra_data_num
        train_set = np.random.permutation(trainFile)
        extra_data = train_set[:extra_data_num,:]
        new_data=np.append(trainFile,extra_data,axis=0)
        x =new_data[...,:-1]
        y =new_data[:,-1]
        print 'xxx',x
        print x.shape
        print 'yyyyy',y
        ff=new_data.tolist()
        print(type(ff))
        print 'ffff',ff
    else:
        print 'pass'
