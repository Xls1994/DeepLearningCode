import  keras
from keras.utils.np_utils import to_categorical
from kim_cnn import loadData
from keras.models import model_from_json
import  numpy as np

if __name__=='__main__':
    modelPath ='model.json'
    weightsPath ='weights.hdf5'
    dataPath ='mr_Fscope.p'
    _,test ,W=loadData(dataPath)
    test_set_x,test_set_y =test
    testlabel =to_categorical(test_set_y)
    model =model_from_json(modelPath)
    model.load_weights()
    y_predict=model.predict(test_set_x)
    np.savetxt('result11'+'.txt',y_predict,fmt='%.4f',delimiter=' ')
