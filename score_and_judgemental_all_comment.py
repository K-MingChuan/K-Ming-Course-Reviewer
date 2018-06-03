import json

from word_preprocessing import load_all_comments

comments = set([cm.strip() for cm in load_all_comments()])

# reviews expected to be a dict with { comment text: rating to the class }
REVIEWS_FILE_NAME = 'data/ratings.json'
JUDGEMENT_FILE_NAME = 'data/judgements.json'

with open(REVIEWS_FILE_NAME, 'r+', encoding='utf-8') as fr1:
    reviews = json.load(fr1)

with open(JUDGEMENT_FILE_NAME, 'r+', encoding='utf-8') as fr2:
    judgements = json.load(fr2)

comments = [cm for cm in comments
           if cm not in reviews]  # get all unlabeled comments
print("目前已分完: " + str(len(reviews)) + " 筆留言。")
print("剩下" + str(len(comments)) + "筆留言。")

for i in range(len(comments)):
    while True:
        try:
            rating = int(input("(" + str(i) + ") [" + comments[i] + "], 請評分(0~5): "))
            judge = int(input("(" + str(i) + ") 請給予批判性(有:1, 沒有:0): "))
            
            if 0 <= judge <= 1 and 0 <= rating <= 5:
                break
        except:
            print("Error!")

    reviews[comments[i]] = rating
    judgements[comments[i]] = judge

    with open(REVIEWS_FILE_NAME, 'w+', encoding='utf-8') as fw1:
        fw1.write(json.dumps(reviews, indent=4, ensure_ascii=False))

    with open(JUDGEMENT_FILE_NAME, 'w+', encoding='utf-8') as fw2:
        fw2.write(json.dumps(judgements, indent=4, ensure_ascii=False))

    print()


