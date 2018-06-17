import json
import numpy as np
from keras.callbacks import ModelCheckpoint

from keras.layers import *
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

from jiebas import jieba_utils
from word2vec import word_vector_utils

commentFileName = 'data/useful_comments_words_new'
wordRepositoryFileName = 'data/comments_words_repository'

word_vector_model = word_vector_utils.get_word2vec_model()
word_vector_size = word_vector_utils.get_word_vector_size()


def readWordRepository():
    with open(wordRepositoryFileName + '.txt', 'r', encoding='utf-8') as f:
        index = 0
        words = dict()
        for line in f.readlines():
            words[line] = index + 1
    return words


def readfile(filename):
    with open(filename + '.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def writeEachUsefulCommentsWords(comment_word_list, rating_list):
    with open(commentFileName + '.txt', 'w+', encoding='utf-8') as f:
        for i in range(len(comment_word_list)):
            line = str(rating_list[i]) + ' '
            for cw in comment_word_list[i]:
                line += cw + ' '
            f.write(line + '\n')


def readEachUsefulCommentsWordsAndRatings():
    with open(commentFileName + '.txt', 'r', encoding='utf-8') as f:
        words = list()
        ras = list()
        for line in f.readlines():
            tokens = line.strip().split(' ')
            ras.append(int(tokens[0])/5)
            words.append([t for t in tokens[1:]])
        return words, ras


def createJudgementalCommentsWordsAndRatings():
    # rs, cws = readEachUsefulCommentsWordsAndRatings()
    # if rs:
    #     return rs, cws

    ratings_new = readfile('data/ratings_new')
    cws = list()
    rs = list()
    for c, rating in ratings_new.items():
        if rating != -1:
            jws = jieba_utils.cut(c)
            cws.append(jws)
            rs.append(rating)
    writeEachUsefulCommentsWords(cws, rs)
    return cws, rs


def getWordVectors(comment):
    words = list()
    for w in comment:
        if w in word_vector_model.wv:
            words.append(word_vector_model.wv[w])
    return words


def comment_to_data(word, word_index_dict):
    data = np.zeros(len(word_index_dict))
    if word in word_index_dict:
        index = word_index_dict[word]
        data[index] = 1
        data = np.append(data, 0)
    else:
        data = np.append(data, 1)
    return list(data)


def build_baseline_model():
    model = Sequential()
    model.add(LSTM(128, input_shape=(max_len, 2405), return_sequences=True))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


def train_and_save_model(model, data, labels, epoch=20, batch_size=128):
    filepath = "model_weights_e{}_b{}.hdf5".format(epoch, batch_size)
    checkpoint = ModelCheckpoint(filepath)
    callbacks_list = [checkpoint]
    model.fit(data, labels, epochs=epoch, batch_size=batch_size, callbacks=callbacks_list)


commentWords_list, labels = createJudgementalCommentsWordsAndRatings()

wordsVector = list()
# for comment in commentWords:
#     wordsVector.append(getWordVectors(comment))

words = readWordRepository()
for commentWords in commentWords_list:
    data = [comment_to_data(word, words) for word in commentWords]
    wordsVector.append(data)
    # print(len(data))


max_len = 50
trainAmount = int(len(commentWords_list) * 0.7)
inputData = pad_sequences(wordsVector, maxlen=max_len, dtype='float32')
# inputData = wordsVector
train_data = np.array(inputData[0:trainAmount])
train_labels = np.array(labels[0:trainAmount])
test_data = np.array(inputData[trainAmount:])
test_labels = np.array(labels[trainAmount:])


print("Train Data's shape: ", train_data.shape)
print("Train Labels' shape: ", train_labels.shape)
print("Test Data's shape: ", test_data.shape)
print("Test Labels' shape: ", test_labels.shape)

batch_size = 1
epoch = 35
model = build_baseline_model()
train_and_save_model(model, train_data, train_labels)
model.summary()
score = model.evaluate(test_data, test_labels, batch_size=batch_size)
print(score)

while True:
    comment = input("評論: ")
    commentWords = jieba_utils.cut(comment)
    print(commentWords)
    # wordVectors = getWordVectors(words)
    wordVectors = [comment_to_data(word, words) for word in commentWords]

    comment = pad_sequences([wordVectors], maxlen=max_len, dtype='float32')
    x = np.array(comment)
    y = model.predict(x)[0][0]*5
    print("評分: ", round(y))
