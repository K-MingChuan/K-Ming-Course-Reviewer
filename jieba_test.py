from jiebas import jieba_utils
from word2vec import word_vector_utils

word_vector_model = word_vector_utils.get_word2vec_model()


def test(comment):
    sentence_words = jieba_utils.cut(comment)
    print(sentence_words)
    words = list()
    for word in sentence_words:
        if word in word_vector_model.wv:
            words.append(word_vector_model.wv[word])
        else:
            print(word, ' 沒有在模組中')
