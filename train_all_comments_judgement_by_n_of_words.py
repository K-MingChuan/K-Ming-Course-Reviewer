# -*- coding: utf-8 -*-
"""## 引入套件"""

import numpy as np
import json
from collections import defaultdict
import jieba
from keras.layers import *
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.sequence import pad_sequences

from TestCallBacks import ShowTestAccuracyEachEpoch
import pickle

"""# 預設資料"""

# change the vector's max length
max_seq_len = 150

"""## 預處理"""


def readJudgementsFromFile():
    with open("data/judgements.json", 'r', encoding='utf-8') as f:
        judgement = json.load(f)
    return judgement


def build_word_dataset(judgement):
    word_db = []
    for comment in judgement:
        words = jieba.cut(comment, cut_all=True)
        for word in words:
            if len(word.strip()) != 0:
                word_db.append(word)
    return [word for word in set(word_db)]


def get_comment_and_judgement(judgement):
    comments = []
    judges = []
    for com, ju in judgement.items():
        comments.append(com)
        judges.append(ju)
    return comments, judges


def words2Vector(comments, word_dataset):
    word_vectors = []
    for comment in comments:
        jieba_words = jieba.cut(comment)
        words = [word for word in jieba_words
                 if len(word.strip()) != 0]

        vectors = defaultdict(float)
        for word in word_dataset:
            vectors[word] += words.count(word)
        word_vectors.append([w for w in vectors.values()])
    return word_vectors


def build_up_word_index_dict(word_dataset):
    assert isinstance(word_dataset, list), 'The word list should be a list! Unordered collections are not accepted!'
    return dict((word, index) for index, word in enumerate(word_dataset))


def comment_to_indices(comment, word_to_index):
    indices = []
    size = len(word_to_index)
    words = jieba.cut(comment)
    # print('Comment: {}\n Terms: {}'.format(comment, ','.join(words)))
    for word in words:
        if word in word_to_index:
            indices.append(word_to_index[word])
        else:
            indices.append(size)  # other
    return indices


def comment_to_one_hot(comment, word_dataset):
    jieba_words = jieba.cut(comment)
    jieba_words = [word for word in jieba_words]
    vectors = defaultdict(float)

    for word in word_dataset:
        # vectors[word] += jieba_words.count(word)
        vectors[word] = 1
    word_vector = [w for w in vectors.values()]

    return word_vector


def judge_to_one_hot(judgements):
    return [[1, 0] if j == 0 else [0, 1] for j in judgements]


def build_baseline_model(word_count):
    model = Sequential()
    model.add(Embedding(word_count, 600, input_length=max_seq_len))
    model.add(Bidirectional(LSTM(1500, return_sequences=True, dropout=0.3, recurrent_dropout=0.1)
                            , input_shape=(max_seq_len, 400)))
    model.add(Bidirectional(LSTM(1500, dropout=0.3, recurrent_dropout=0.1)))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_and_save_model(model, train_data, train_labels,
                         validation_data, validation_labels,
                         test_data, test_labels,
                         epoch=20, batch_size=128):
    filename = "judgemental_model_weights_e{}_b{}".format(epoch, batch_size)
    checkpoint = ModelCheckpoint(filename + ".hdf5")
    early_stopping = EarlyStopping(monitor='val_acc', mode='max', patience=2)
    test_callback = ShowTestAccuracyEachEpoch(test_data, test_labels)
    callbacks_list = [checkpoint, test_callback, early_stopping]
    history = model.fit(train_data, train_labels, validation_data=(validation_data, validation_labels),
                        epochs=epoch, batch_size=batch_size, callbacks=callbacks_list)
    with open(filename + ".pickle", 'wb') as fw:
        pickle.dump(history.history, fw)


def test_model(comments):
    while True:
        comment = input("評論: ")

        comment = comment_to_indices(comment, word_index_dict)

        comment = pad_sequences([comment], maxlen=100, dtype='float32')

        # comment = np.swapaxes(comment, 1, 2)

        x = np.array(comment)
        y = model.predict(x)
        d1 = round(y[0][0])
        print(d1)

        d2 = round(y[0][1])
        print(d2)

        score = '有' if d1 == 1 and d2 == 0 else '沒有'

        print("批判性: ", score)


if __name__ == '__main__':
    comment_judgements = readJudgementsFromFile()
    comments, judgements = get_comment_and_judgement(comment_judgements)
    word_dataset = build_word_dataset(comment_judgements)
    word_index_dict = build_up_word_index_dict(word_dataset)

    comments = [comment_to_indices(comment, word_index_dict) for comment in comments]

    judgements = np.array(judge_to_one_hot(judgements))

    # -----Preparing the training and testing data-----
    trainAmount = int(len(comments) * 0.6)
    data = pad_sequences(comments, maxlen=max_seq_len, dtype='float32')

    random_mask = np.arange(len(data))
    np.random.shuffle(random_mask)
    data = data[random_mask]
    judgements = judgements[random_mask]

    train_data = data[:trainAmount]
    train_labels = judgements[:trainAmount]
    test_data = data[trainAmount:]
    test_labels = judgements[trainAmount:]
    validation_data = test_data[:200]
    validation_labels = test_labels[:200]

    print("Train Data's shape: ", train_data.shape)
    print("Train Labels' shape: ", train_labels.shape)
    print("Test Data's shape: ", test_data.shape)
    print("Test Labels' shape: ", test_labels.shape, "\n")

    # -----Start training-----
    batch_size = 32
    epoch = 30
    model = build_baseline_model(len(word_index_dict) + 1)  # since the first value was not usable
    model.summary()
    train_and_save_model(model, train_data, train_labels, validation_data, validation_labels,
                         test_data, test_labels,
                         epoch=epoch, batch_size=batch_size)
    loss, accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size)
    print(loss)
    print(accuracy)
