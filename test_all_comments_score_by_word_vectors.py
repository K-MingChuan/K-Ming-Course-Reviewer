import comment_preprocessing
from keras.models import load_model
from keras.layers import *
from keras.preprocessing.sequence import pad_sequences

max_seq_len = 100
model_file_name = ''
model = load_model(model_file_name)

while True:
    comment = input("評論: ")

    comments_word_vector = [comment_preprocessing.comment_to_word_vectors(c) for c in comment]

    comment = pad_sequences(comments_word_vector, maxlen=max_seq_len, dtype='float32')

    x = np.array(comment)
    y = model.predict(x)[0][0]
    print("評分: ", round(y))