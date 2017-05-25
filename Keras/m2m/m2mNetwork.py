# -*- coding: utf-8 -*-
'''
author:yangyl

'''
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate
from keras.layers import LSTM

class m2mNetwork():
    def __init__(self,story_maxlen,query_maxlen,vocab_size):
        self.input_sequence = Input((story_maxlen,))
        self.question = Input((query_maxlen,))
        self.vocab_size =vocab_size
        self.query_maxlen =query_maxlen
        self.story_maxlen=story_maxlen
    def encoders_m(self, inputs):
        input_encoder_m = Sequential()
        input_encoder_m.add(Embedding(input_dim=self.vocab_size,
                                      output_dim=64))
        input_encoder_m.add(Dropout(0.3))
        encode_m = input_encoder_m(inputs)
        return encode_m

    def encoders_c(self, inputs):
        input_encoder_c = Sequential()
        input_encoder_c.add(Embedding(input_dim=self.vocab_size,
                                      output_dim=self.query_maxlen))
        input_encoder_c.add(Dropout(0.3))
        encoder_c = input_encoder_c(inputs)
        return encoder_c

    def encoders_question(self, inputs):
        question_encoder = Sequential()
        question_encoder.add(Embedding(input_dim=self.vocab_size,
                                       output_dim=64,
                                       input_length=self.query_maxlen))
        question_encoder.add(Dropout(0.3))
        quest_en = question_encoder(inputs)
        return quest_en

    def build(self):
        input_encoded_m = self.encoders_m(self.input_sequence)
        input_encoded_c = self.encoders_c(self.input_sequence)
        question_encoded = self.encoders_question(self.question)
        match = dot([input_encoded_m, question_encoded], axes=(2, 2))
        match = Activation('softmax')(match)

        # add the match matrix with the second input vector sequence
        response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
        response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

        # concatenate the match matrix with the question vector sequence
        answer = concatenate([response, question_encoded])

        # the original paper uses a matrix multiplication for this reduction step.
        # we choose to use a RNN instead.
        answer = LSTM(32)(answer)  # (samples, 32)

        # one regularization layer -- more would probably be needed.
        answer = Dropout(0.3)(answer)
        answer = Dense(self.vocab_size)(answer)  # (samples, vocab_size)
        # we output a probability distribution over the vocabulary
        answer = Activation('softmax')(answer)

        # build the final model
        model = Model([self.input_sequence, self.question], answer)
        self.model = model
        return model
    def summary(self):
        return self.model.summary()

if __name__ == '__main__':
    # test the class
    m2m =m2mNetwork(30,20,10)
    m2m.build()
    m2m.summary()