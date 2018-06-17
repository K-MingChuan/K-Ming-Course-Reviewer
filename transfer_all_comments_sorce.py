import json


def readfile(filename):
    with open(filename + '.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def writefile(filename, contents):
    with open(filename + '.json', 'w+', encoding='utf-8') as fw:
        fw.write(json.dumps(contents, indent=4, ensure_ascii=False))


judgements = readfile('data/judgements')
ratings = readfile('data/ratings')

for judgement_key, judgement_value in judgements.items():
    if judgement_value == 0:
        ratings[judgement_key] = -1

writefile('data/ratings_new', ratings)
