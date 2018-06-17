import json

import numpy as np
from jiebas import jieba_utils
from keras.layers import Dense, Activation, Softmax
from keras.models import Sequential


def num_to_one_hot(num):
    a = [0] * 6  # 0~5
    a[num] = 1
    return a


def build_up_word_set(comments):
    wordset = set()
    for comment in comments:
        words = jieba_utils.cut(comment, cut_all=True)
        wordset.update(words)
    return wordset


def build_up_word_index_dict(wordset):
    words = list(wordset)
    word_index_dict = {}
    for i in range(len(words)):
        word_index_dict[words[i]] = i
    return word_index_dict


def comment_to_data(comment, word_index_dict):
    data = np.zeros(len(word_index_dict))
    words = jieba_utils.cut(comment, cut_all=True)
    for word in words:
        if word in word_index_dict:
            index = word_index_dict[word]
            data[index] += 1
    return data


def build_baseline_model(word_count):
    model = Sequential()
    model.add(Dense(50, input_dim=word_count))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(6))
    model.add(Softmax())

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def torture_model(model, word_index_dict):
    while True:
        comment = input("評論: ")
        x = np.array([comment_to_data(comment, word_index_dict)])
        y = model.predict(x)[0]
        print(y)
        print("評分: ", np.argmax(y))


with open('data/reviews.json', 'r', encoding='utf-8') as f:
    reviews = json.load(f)

comments = reviews.keys()
data = []

wordset = build_up_word_set(comments)
word_index_dict = build_up_word_index_dict(wordset)

for comment in comments:
    data.append(comment_to_data(comment, word_index_dict))

data = np.array(data)
labels = np.array(list([num_to_one_hot(rating) for rating in reviews.values()]))

count = len(data)
train_data = data[: int(count*0.7), :]
train_labels = labels[: int(count*0.7)]
test_data = data[int(count*0.7):, :]
test_labels = labels[int(count*0.7):]

print("Train Data's shape: ", train_data.shape)
print("Train Labels' shape: ", train_labels.shape)
print("Test Data's shape: ", test_data.shape)
print("Test Labels' shape: ", test_labels.shape)


batch_size = 50
epoch = 30
best_score = -1
best_model = None
for i in range(5):
    print("Round ", i)
    model = build_baseline_model(len(wordset))
    model.fit(train_data, train_labels, epochs=epoch, batch_size=batch_size)
    model.summary()
    score = model.evaluate(test_data, test_labels, batch_size=batch_size)
    if score[1] > best_score:
        best_score = score[1]
        best_model = model
    print("Score: ", score)

print("Best score: ", best_score)
torture_model(best_model, word_index_dict)