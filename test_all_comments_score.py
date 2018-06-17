import train_all_comments_score
import comment_preprocessing
from keras.models import load_model
from keras.layers import *
from keras.preprocessing.sequence import pad_sequences

max_len = 200
commentFileName = 'data/ratings_new'
model_file_name = 'model_weights_e30_b32.hdf5'


comments, ratings = comment_preprocessing.get_useful_comment_and_rating(commentFileName)
model = load_model(model_file_name)

while True:
    comment = input("評論: ")
    comments_word_list = comment_preprocessing.build_up_word_list(comments,
                                                                  word_list_file_name='word_list.txt',
                                                                  output_file_name='word_list.txt')

    word_index_dict = comment_preprocessing.build_up_word_index_dict(comments_word_list)
    comment = comment_preprocessing.comment_to_one_hot(comment, word_index_dict)

    comment = pad_sequences([comment], maxlen=max_len, dtype='float32')
    x = np.array(comment)
    y = model.predict(x)[0][0]
    print("評分: ", round(y))