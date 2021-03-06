#!/usr/bin/python3

import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import nltk
import re


from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from sklearn import metrics

KAGGLE_HOME = os.environ['KAGGLE_HOME']
MAX_LENGTH = 100
MAX_FEATURES = 30000
EMBEDDINGS_SIZE = 300

STEMMER = nltk.stem.SnowballStemmer('english')

fixed_count = 0


def build_misspellings_dict():
    dict_ = {}
    with open("%s/%s" % (KAGGLE_HOME, "misspellings3"), encoding='utf8') as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
        for line in lines:
            split = line.split("->")
            if len(split) == 2:
                mis = split[0]
                correct = split[1]
                dict_[mis] = correct
    return dict_


MISSPELLINGS_DICT = build_misspellings_dict()


def fix_word(word):
    global fixed_count
    if word in MISSPELLINGS_DICT:
        fixed_count += 1
        return MISSPELLINGS_DICT[word]
    return word


def generate_embeddings_index():
    embeddings_index = {}
    with open("%s/%s" % (KAGGLE_HOME, "crawl-300d-2M.vec"), encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
            except Exception as e:
                print(e)
                print(values[1:])
            embeddings_index[word] = coefs
    return embeddings_index


def dump_embeddings_index():
    with open("%s/%s" % (KAGGLE_HOME, "embeddings_index.pickle"), "wb") as pickle_out:
        pickle.dump(generate_embeddings_index(), pickle_out)


def load_embeddings_index():
    with open("%s/%s" % (KAGGLE_HOME, "embeddings_index.pickle"), "rb") as pickle_in:
        return pickle.load(pickle_in)


EMBEDDINGS_INDEX = load_embeddings_index()


def keep_word(word):
    return word not in stop_words and word.isalpha()
    #return word.isalpha()


def stem_word(word):
    return STEMMER.stem(word)


def format_word(word):
    word = re.sub(r'\b\w{1,2}\b', '', word)
    return fix_word(word)


def filter_sequence(sequence):
    return [format_word(word) for word in sequence if keep_word(word)]


def sequence_to_text(sequence):
    return " ".join(sequence)


def clean_text(text):
    return sequence_to_text(filter_sequence(text_to_word_sequence(text)))


def get_processed_train_valid_test():
    train_csv = pd.read_csv(
        "%s/%s" %
        (KAGGLE_HOME,
         "train.csv"),
        sep=',',
        encoding='utf8')
    test_csv = pd.read_csv(
        "%s/%s" %
        (KAGGLE_HOME,
         "test.csv"),
        sep=',',
        encoding='utf8')
    x = train_csv.comment_text
    y = train_csv.iloc[:, 2:]
    xtest = test_csv.comment_text
    #xtrain, xvalid, ytrain, yvalid = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y[[
    xtrain, xvalid, ytrain, yvalid = train_test_split(x, y, test_size=0.05, random_state=42)
    xtrain = xtrain.apply(lambda text: clean_text(text))
    xvalid = xvalid.apply(lambda text: clean_text(text))
    xtest = xtest.apply(lambda text: clean_text(text))
    return xtrain, xvalid, xtest, ytrain, yvalid


xtrain, xvalid, xtest, ytrain, yvalid = get_processed_train_valid_test()

print("fixed : ", fixed_count)


def get_tokenizer(docs):
    t = Tokenizer(lower=True, num_words=MAX_FEATURES)
    t.fit_on_texts(docs)
    return t


docs = list(xtrain) + list(xvalid) + list(xtest)
tokenizer = get_tokenizer(docs)


def get_encoded_xtrain_xvalid_xtest():
    train_encoded_docs = tokenizer.texts_to_sequences(xtrain)
    train_padded_docs = pad_sequences(
        train_encoded_docs,
        maxlen=MAX_LENGTH,
        padding='post')
    valid_encoded_docs = tokenizer.texts_to_sequences(xvalid)
    valid_padded_docs = pad_sequences(
        valid_encoded_docs,
        maxlen=MAX_LENGTH,
        padding='post')
    test_encoded_docs = tokenizer.texts_to_sequences(xtest)
    test_padded_docs = pad_sequences(
        test_encoded_docs,
        maxlen=MAX_LENGTH,
        padding='post')
    return train_padded_docs, valid_padded_docs, test_padded_docs


train_padded_docs, valid_padded_docs, test_padded_docs = get_encoded_xtrain_xvalid_xtest()


def get_embedding_matrix():
    embedding_matrix = zeros((MAX_FEATURES, EMBEDDINGS_SIZE))
    for word, i in list(tokenizer.word_index.items()):
        if i >= MAX_FEATURES:
            continue
        embedding_vector = EMBEDDINGS_INDEX.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def show_model_history(his):
    plt.plot(his.history['loss'])
    plt.plot(his.history['val_loss'])  # RAISE ERROR
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def generate_submission(model):
    test_csv = pd.read_csv(
        "%s/%s" %
        (KAGGLE_HOME,
         "test.csv"),
        sep=',',
        encoding='utf8')
    predictions = model.predict(test_padded_docs)
    submission = pd.DataFrame({
        'id': test_csv['id'],
        'toxic': predictions[:, 0],
        'severe_toxic': predictions[:, 1],
        'obscene': predictions[:, 2],
        'threat': predictions[:, 3],
        'insult': predictions[:, 4],
        'identity_hate': predictions[:, 5],
    })
    submission.to_csv("%s/%s" % (KAGGLE_HOME, "submit.csv"), index=False)


def print_roc_auc(model, xvalid, yvalid):
    validpreds = model.predict(xvalid)
    roc = metrics.roc_auc_score(yvalid, validpreds)
    print("ROC:", roc)
