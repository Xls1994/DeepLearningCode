# 多窗口卷积cnn
```python
def buildCNN(*args):
    nb_filter =50
    main_inputs = Input(shape=(maxlen,), dtype='int32', name='main_input')
    inputs = Embedding(max_features, embedding_size, input_length=maxlen, weights=[WordEM])(main_inputs)
    # x =Dropout(0.25)(inputs)
    convs = []
    filter_sizes = (1,2, 3, 4)
    for fsz in filter_sizes:
        conv = Conv1D(filters=nb_filter,
                             kernel_size=fsz,
                             strides=1,
                             padding='valid',
                             activation='relu',
                             kernel_regularizer=l2(l=0.01)
                             )(inputs)
        pool_size=maxlen-fsz-1
        pool = MaxPooling1D(pool_size=pool_size)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)
    out = Concatenate(axis=1)(convs)
    out =Dense(100,activation='relu')(out)
    out =Dropout(0.5)(out)
    out =Dense(50,activation='relu')(out)
    predict = Dense(2, activation='softmax',kernel_regularizer=l2(0.01))(out)
    model = Model(inputs=main_inputs, outputs=predict)
    return model
```
