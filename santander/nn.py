import numpy as np
import pandas as pd
import csv
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras import backend as K
from sklearn import preprocessing 
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping


train = pd.read_csv('train.csv')

submit = pd.read_csv('test.csv')
x_submit = submit.iloc[:,1:]
ids = submit.iloc[:,0]


x = train.iloc[:,2:]
y = np.log1p(train.iloc[:,1].values)


def remove_zero_columns(x, submit):
    cols = []
    for col in x.columns:
        if col != 'ID' and col != 'target':
            if x[col].std() == 0: 
                cols.append(col)
    x.drop(cols, axis=1, inplace=True)
    submit.drop(cols, axis=1, inplace=True)
    return x, submit


x, x_submit = remove_zero_columns(x, x_submit)
x = preprocessing.StandardScaler().fit_transform(x)
x_submit = preprocessing.StandardScaler().fit_transform(x_submit)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

model = Sequential()
model.add(Dense(10, kernel_initializer="he_normal", input_dim=4735))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(5, kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(1))

adam = Adam(lr=0.003)
model.compile(loss=root_mean_squared_error, optimizer=adam)

callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=1, mode='auto')]
model.fit(x_train, y_train, epochs=200, batch_size=32, validation_data=(x_test, y_test), callbacks=callbacks)



preds = pd.DataFrame(np.expm1(model.predict(x_submit)))
preds.insert(0, "ID", ids.values)
preds.columns = ["ID","target"]

preds.to_csv("submit.csv", index = False)


