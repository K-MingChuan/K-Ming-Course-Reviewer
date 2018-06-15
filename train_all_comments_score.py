import comment_preprocessing
from keras.layers import *
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from word2vec import word_vector_utils
from keras.preprocessing.sequence import pad_sequences
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
max_len = 200
commentFileName = 'data/ratings_new'
word_vector_size = word_vector_utils.get_word_vector_size()


def build_baseline_model():
    model = Sequential()
    model.add(LSTM(3000, input_shape=(max_len, 2557), return_sequences=True))
    model.add(LSTM(1280))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


def train_and_save_model(model, data, labels, epoch=20, batch_size=128):
    filepath = "model_weights_e{}_b{}.hdf5".format(epoch, batch_size)
    checkpoint = ModelCheckpoint(filepath)
    callbacks_list = [checkpoint]
    model.fit(data, labels, epochs=epoch, batch_size=batch_size, callbacks=callbacks_list)


def save_comments_words_set(comments_words_set):
    with open('comments_words_set.txt', 'w+', encoding='utf-8') as f:
        for word in comments_words_set:
            f.write(word + '\n')


if __name__ == '__main__':
    comments, ratings = comment_preprocessing.get_useful_comment_and_rating(commentFileName)

    comments_words_set = comment_preprocessing.build_up_word_set(comments)
    word_index_dict = comment_preprocessing.build_up_word_index_dict(comments_words_set)

    comments = [comment_preprocessing.comment_to_one_hot(comment, word_index_dict) for comment in comments]

    trainAmount = int(len(comments) * 0.7)
    comments = pad_sequences(comments, maxlen=max_len, dtype='float32')
    train_data = np.array(comments[0:trainAmount])
    train_labels = np.array(ratings[0:trainAmount])
    test_data = np.array(comments[trainAmount:])
    test_labels = np.array(ratings[trainAmount:])

    print("Train Data's shape: ", train_data.shape)
    print("Train Labels' shape: ", train_labels.shape)
    print("Test Data's shape: ", test_data.shape)
    print("Test Labels' shape: ", test_labels.shape)

    batch_size = 1
    epoch = 10
    model = build_baseline_model()
    train_and_save_model(model, train_data, train_labels, epoch, batch_size)
    model.summary()
    score = model.evaluate(test_data, test_labels, batch_size=batch_size)
    print(score)
