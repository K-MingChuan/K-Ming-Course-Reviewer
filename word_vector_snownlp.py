from hanziconv import HanziConv
from snownlp import SnowNLP
import json


def readfile(filename):
    with open(filename + '.json', 'r', encoding='utf-8') as f:
        return json.load(f)


ratings_new = readfile('D:/GitHub/K-Ming-Course-Reviewer/data/ratings_new')
comment = {s: r for s, r in ratings_new.items() if r != -1}
print(ratings_new)
print(comment)

print(HanziConv.toSimplified('繁簡轉換器'))
print(HanziConv.toTraditional('繁简转换器'))
###############################


for sentence, rating in comment.items():
    s = SnowNLP(HanziConv.toSimplified(sentence))  # first translate to simplified Chinese
    print(s.words)  # tokenize into words
    print([tag for tag in s.tags])  # [('这个', 'r'), ('产品', 'n'), ('很', 'd'), ('糟', 'a')]
    print(s.sentiments*5, rating)  # sentiment analysis result [0, 1]
    # print(s.pinyin)  # 轉成拼音
    # s = SnowNLP(u'「繁體字」「繁體中文」的叫法在臺灣亦很常見。')
    # print(s.han)  # 轉成簡體字
