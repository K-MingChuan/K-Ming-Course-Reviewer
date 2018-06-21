import comment_preprocessing
import pickle
from word2vec import word_vector_utils
from keras.layers import *
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences

commentFileName = 'data/ratings_new'
max_seq_len = 100
word_vector_size = word_vector_utils.get_word_vector_size()


def build_baseline_model(feature_size):
    model = Sequential()
    model.add(LSTM(200, input_shape=(max_seq_len, feature_size), return_sequences=True))
    model.add(LSTM(200))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


def train_and_save_model(model, data, labels, epoch=20, batch_size=128):
    file_name = "word_vector_lstm_e{}_b{}".format(epoch, batch_size)
    checkpoint = ModelCheckpoint(file_name + '.hdf5')
    callbacks_list = [checkpoint]
    history = model.fit(data, labels, epochs=epoch, batch_size=batch_size, callbacks=callbacks_list)
    with open(file_name + '.pickle', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


if __name__ == '__main__':
    comments, ratings = comment_preprocessing.get_judgemental_comments_and_rating(commentFileName)

    comments_word_vector = [comment_preprocessing.comment_to_word_vectors(comment) for comment in comments]

    data = pad_sequences(comments_word_vector, maxlen=max_seq_len, dtype='float32')
    ratings = np.array(ratings)
    trainAmount = int(len(data) * 0.7)
    train_data = data[0:trainAmount]
    train_labels = ratings[0:trainAmount]
    test_data = data[trainAmount:]
    test_labels = ratings[trainAmount:]

    print("Train Data's shape: ", train_data.shape)
    print("Train Labels' shape: ", train_labels.shape)
    print("Test Data's shape: ", test_data.shape)
    print("Test Labels' shape: ", test_labels.shape)

    batch_size = 100
    epoch = 1
    model = build_baseline_model(train_data.shape[2])
    model.summary()
    train_and_save_model(model, train_data, train_labels, epoch, batch_size)
    loss, accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size)
    print(accuracy)
