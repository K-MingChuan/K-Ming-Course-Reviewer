import json
import numpy as np
from jiebas import jieba_utils
from word2vec import word_vector_utils

word_vector_model = word_vector_utils.get_word2vec_model()


def read_file(filename):
    with open(filename + '.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def comment_to_word_vectors(comment):
    words = list()
    sentence_words = jieba_utils.cut(comment)
    for word in sentence_words:
        if word in word_vector_model.wv:
            words.append(word_vector_model.wv[word])
    return words


def comment_to_n_of_words(comment):
    return jieba_utils.cut(comment, cut_all=True)


def word_to_vectors(word):
    if word in word_vector_model.wv:
        return word_vector_model.wv[word]


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


def comment_to_one_hot(comment, word_index_dict):
    data = list()
    words = jieba_utils.cut(comment, cut_all=True)
    for word in words:
        data_one_hot = np.zeros(len(word_index_dict))
        if word in word_index_dict:
            index = word_index_dict[word]
            data_one_hot[index] = 1
            data_one_hot = np.append(data_one_hot, 0)
        else:
            data_one_hot = np.append(data_one_hot, 1)
        data.append(data_one_hot.tolist())
    return data


def get_useful_comment_and_rating(commentFileName):
    comments_json = read_file(commentFileName)
    comments = list()
    ratings = list()
    for comment, rating in comments_json.items():
        if rating != -1:
            comments.append(comment)
            ratings.append(rating)
    return comments, ratings
