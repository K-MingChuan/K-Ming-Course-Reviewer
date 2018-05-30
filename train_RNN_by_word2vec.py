import json

from keras.layers import *
from keras.models import Sequential

from jiebas import jieba_utils
from word2vec import word_vector_utils

with open('data/reviews.json', 'r', encoding='utf-8') as fr:
    all_reviews = json.load(fr)

assert isinstance(all_reviews, dict)

word_vector_model = word_vector_utils.get_word2vec_model()
word_vector_size = word_vector_utils.get_word_vector_size()


def num_to_one_hot(num):
    a = [0] * 6
    a[num] = 1
    return a


def get_sum_word_vector(sentence):
    words = jieba_utils.cut(sentence, cut_all=True)
    word_vector_sum = np.zeros(word_vector_size)
    for word in words:
        if word in word_vector_model.wv:
            word_vector_sum += word_vector_model.wv[word]
    return word_vector_sum


def build_baseline_model():
    model = Sequential()
    model.add(Dense(50, input_dim=word_vector_size))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(6))
    model.add(Softmax())

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def torture_model(model):
    while True:
        comment = input("評論: ")
        x = np.array([get_sum_word_vector(comment)])
        y = model.predict(x)[0]
        print(y)
        print("評分: ", np.argmax(y))


data = []
labels = []
for comment, rating in all_reviews.items():
    data.append(get_sum_word_vector(comment))
    labels.append(num_to_one_hot(rating))
    if len(data) % 100 == 0:
        print("100 data finished...")

print("Data already prepared.")

count = len(data)
data = np.array(data)
labels = np.array(labels)

train_data = data[: int(count * 0.7), :]
train_labels = labels[: int(count * 0.7)]
test_data = data[int(count * 0.7):, :]
test_labels = labels[int(count * 0.7):]

print("Train Data's shape: ", train_data.shape)
print("Train Labels' shape: ", train_labels.shape)
print("Test Data's shape: ", test_data.shape)
print("Test Labels' shape: ", test_labels.shape)

batch_size = 50
epoch = 30
best_score = -1
best_model = None
for i in range(10):
    print("Round ", i)
    model = build_baseline_model()
    model.fit(train_data, train_labels, epochs=epoch, batch_size=batch_size)
    model.summary()
    score = model.evaluate(test_data, test_labels, batch_size=batch_size)
    if score[1] > best_score:
        best_score = score[1]
        best_model = model
    print("Score: ", score)

torture_model(best_model)
