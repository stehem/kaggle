#!/usr/bin/python3

from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.models import Model
from keras.layers import Dropout, SpatialDropout1D
from sklearn import metrics
from keras.layers import Bidirectional, GlobalMaxPool1D, CuDNNGRU, GlobalAveragePooling1D, concatenate


import embeddings


# Embedding and GRU


#

sequence_input = Input(shape=(embeddings.MAX_LENGTH,), dtype='int32')
layer = Embedding(embeddings.VOCAB_SIZE, 200,input_shape=(embeddings.MAX_LENGTH,))(sequence_input)
layer = SpatialDropout1D(0.2)(layer)

layer = Bidirectional(CuDNNGRU(150, return_sequences=True))(layer)
layer = GlobalAveragePooling1D()(layer)

preds = Dense(6, activation='sigmoid')(layer)
model = Model(sequence_input, preds)


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

EPOCHS=2

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=1, mode='auto')


history = model.fit(embeddings.train_padded_docs, y=embeddings.ytrain, epochs=EPOCHS, verbose=1, validation_data = (embeddings.valid_padded_docs,embeddings.yvalid), callbacks=[early_stopping])


embeddings.print_roc_auc(model, embeddings.valid_padded_docs, embeddings.yvalid)
embeddings.show_model_history(history)


embeddings.generate_submission(model)
 #id,toxic,severe_toxic,obscene,threat,insult,identity_hate
