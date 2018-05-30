import json

from word_preprocessing import load_all_comments

comments = set([cm.strip() for cm in load_all_comments()])

# reviews expected to be a dict with { comment text: rating to the class }
REVIEWS_FILE_NAME = 'data/reviews.json'

with open(REVIEWS_FILE_NAME, 'r+', encoding='utf-8') as fr:
    reviews = json.load(fr)

comments = [cm for cm in comments
           if cm not in reviews]  # get all unlabeled comments
print("目前已分完: " + str(len(reviews)) + " 筆留言。")
print("剩下" + str(len(comments)) + "筆留言。")

for i in range(len(comments)):
    while True:
        try:
            rating = int(input("(" + str(i) + ") [" + comments[i] + "], 請評分(0~5): "))
            if 0 <= rating <= 5:
                break
        except:
            print("Error!")

    reviews[comments[i]] = rating

    with open(REVIEWS_FILE_NAME, 'w+', encoding='utf-8') as fw:
        json.dump(reviews, fw, ensure_ascii=False)


