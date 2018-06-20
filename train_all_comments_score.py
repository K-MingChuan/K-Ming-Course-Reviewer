import comment_preprocessing
from keras.layers import *
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from word2vec import word_vector_utils
from keras.preprocessing.sequence import pad_sequences
import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
max_seq_len = 100
commentFileName = 'data/ratings_new'
word_vector_size = word_vector_utils.get_word_vector_size()


def build_baseline_model(word_count):
    model = Sequential()
    model.add(Embedding(word_count, 400, input_length=max_seq_len))
    model.add(LSTM(400, input_shape=(max_seq_len, 400), return_sequences=True, dropout=0.3))
    model.add(LSTM(400, input_shape=(max_seq_len, 400), dropout=0.3))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


def train_and_save_model(model, data, labels, epoch=20, batch_size=128):
    file_name = "n_of_words_800_2lstm_e{}_b{}".format(epoch, batch_size)
    checkpoint = ModelCheckpoint(file_name + '.hdf5')
    callbacks_list = [checkpoint]
    history = model.fit(data, labels, epochs=epoch, batch_size=batch_size, callbacks=callbacks_list)
    with open(file_name + '.pickle', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


def save_comments_words_set(comments_words_set):
    with open('comments_words_set.txt', 'w+', encoding='utf-8') as f:
        for word in comments_words_set:
            f.write(word + '\n')


if __name__ == '__main__':
    comments, ratings = comment_preprocessing.get_judgemental_comments_and_rating(commentFileName)

    comments_word_list = comment_preprocessing.build_up_word_list(comments,
                                                                  word_list_file_name='word_list.txt')
    word_index_dict = comment_preprocessing.build_up_word_index_dict(comments_word_list)


    data = [comment_preprocessing.comment_to_indices(comment, word_index_dict) for comment in comments]

    max_word_count = len(max(data, key=lambda seq: len(seq)))
    print('Max words count: ', max_word_count, ' Your max_seq_len param is set: ', max_seq_len)

    # shuffle the comments and rating
    data = pad_sequences(data, maxlen=max_seq_len, dtype='float32')
    ratings = np.array(ratings)
    random_mask = np.arange(len(data))
    np.random.shuffle(random_mask)
    data = data[random_mask]
    ratings = ratings[random_mask]

    trainAmount = int(len(data) * 0.7)
    train_data = data[0:trainAmount]
    train_labels = ratings[0:trainAmount]
    test_data = data[trainAmount:]
    test_labels = ratings[trainAmount:]

    print("Train Data's shape: ", train_data.shape)
    print("Train Labels' shape: ", train_labels.shape)
    print("Test Data's shape: ", test_data.shape)
    print("Test Labels' shape: ", test_labels.shape)

    batch_size = 32
    epoch = 50
    model = build_baseline_model(len(word_index_dict) + 1)  # word count +1 for others
    model.summary()
    train_and_save_model(model, train_data, train_labels, epoch, batch_size)
    loss, accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size)
    print(accuracy)
