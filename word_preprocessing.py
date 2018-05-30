import json
from bs4 import BeautifulSoup

ALL_COMMENTS = 'data/all_comments.txt'
OLD_COMMENTS = 'data/old_comments.txt'
SUANKHO_COURSES = 'data/suankho_courses.json'


def remove_html_tag(text):
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    return text


def load_courses():
    with open(SUANKHO_COURSES, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_old_comments():
    return load_comments_by_lines(OLD_COMMENTS)


def load_suankho_comments():
    with open(SUANKHO_COURSES, 'r', encoding='utf-8') as f:
        courses = json.load(f)
        comments = []
        for course in courses:
            for comment in course['comments']:
                text = remove_html_tag(comment['body'])
                comments.append(text)
        return comments


def save_comments_by_lines(comments, filename):
    with open(filename, 'w+', encoding='utf-8') as fw:
        fw.writelines(comments)


def load_comments_by_lines(filename):
    with open(filename, 'r+', encoding='utf-8') as fw:
        return [c for c in fw.readlines()]


def load_all_comments():
    return load_comments_by_lines(ALL_COMMENTS)


def start_processing():
    comments = set([l for l in load_comments_by_lines(ALL_COMMENTS) if len(l) != 0])
    print("Count: ", len(comments))


if __name__ == '__main__':
    start_processing()
