from numpy import *
# y =np.asarray([np.arange(3),np.arange(3),np.arange(3)])
# print (np.dtype('f2'))
# print (y.shape)
# t =np.dtype([('name',np.str_,40),('price',np.float32)])
# item =np.array([('butter',23)],dtype=t)
# print(item)
def test():
    b =arange(24)
    x =reshape(b,(2,3,4))
    print x
    print x.shape
    c =x.flatten()
    print c
    print x

def testStack():
    a =arange(9).reshape(3,3)
    b=2*a
    c =dstack((a,b))
    d =concatenate((a,b),axis=1)
    print c.shape
def ditt():
    a =array([[1,2],[4,6]])
    c =diff(a,axis=1)
    print c
    print c.shape
if __name__=='__main__':
    # testStack()
    ditt()
    pass