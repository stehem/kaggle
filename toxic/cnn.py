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
from sklearn import metrics
from keras.layers import Bidirectional, GlobalMaxPool1D


import embeddings


embedding_layer = Embedding(embeddings.VOCAB_SIZE, embeddings.EMBEDDINGS_SIZE, weights=[embeddings.get_embedding_matrix()], input_length=embeddings.MAX_LENGTH, trainable=False)

#
convs = []
filter_sizes = [2,3,4,5]

sequence_input = Input(shape=(MAX_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

for fsz in filter_sizes:
    l_conv = Conv1D(nb_filter=99,filter_length=fsz,activation='relu')(embedded_sequences)
    l_pool = MaxPooling1D(5)(l_conv)
    l_conv2 = Conv1D(nb_filter=99,filter_length=fsz,activation='relu')(l_pool)
    l_pool2 = MaxPooling1D(5)(l_conv2)
    drop = Dropout(0.3)(l_pool2)
    convs.append(drop)




l_merge = Merge(mode='concat', concat_axis=1)(convs)
l_flat = Flatten()(l_merge)
l_dense = Dense(32, activation='relu')(l_flat)
preds = Dense(6, activation='sigmoid')(l_dense)
model = Model(sequence_input, preds)
#

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=1, mode='auto')

EPOCHS=100
history = model.fit(embeddings.train_padded_docs, y=embeddings.ytrain, epochs=EPOCHS, verbose=1, validation_data = (embeddings.valid_padded_docs,embeddings.yvalid), callbacks=[early_stopping])



embeddings.print_roc_auc(model, embeddings.valid_padded_docs, embeddings.yvalid)
embeddings.show_model_history(history)





