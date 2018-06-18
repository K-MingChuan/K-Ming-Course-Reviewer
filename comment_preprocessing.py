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


def build_up_word_list(comments, word_list_file_name=None, output_file_name=None):
    if word_list_file_name:
        print('Loading word list from file ' + word_list_file_name)
        with open(word_list_file_name, 'r', encoding='utf-8') as fr:
            word_list = [line.strip() for line in fr.readlines() if len(line.strip()) != 0]
    else:
        wordset = set()
        print('No word list file specified, creating new word list.')
        for comment in comments:
            words = jieba_utils.cut(comment, cut_all=True)
            wordset.update(words)
        word_list = list(wordset)

    if output_file_name:
        with open(output_file_name, 'w+', encoding='utf-8') as fw:
            for word in word_list:
                fw.write(word + '\n')
            print('Word list saved to ' + output_file_name + '.')

    return word_list


def build_up_word_index_dict(word_list):
    assert isinstance(word_list, list), 'The word list should be a list! Unordered collections are not accepted!'
    return dict((word, index) for index, word in enumerate(word_list))


def comment_to_one_hot(comment, word_index_dict):
    data = list()
    words = jieba_utils.cut(comment, cut_all=True)
    for word in words:
        data_one_hot = np.zeros(len(word_index_dict) + 1)
        if word in word_index_dict:
            index = word_index_dict[word]
            data_one_hot[index] = 1
        else:
            data_one_hot[-1] = 1
        data.append(data_one_hot.tolist())
    return data


def get_judgemental_comments_and_rating(commentFileName):
    comments_json = read_file(commentFileName)
    comments = [comment for comment, rating in comments_json.items() if rating != -1]
    ratings = [rating for comment, rating in comments_json.items() if rating != -1]
    return comments, ratings
