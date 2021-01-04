from nltk.corpus import stopwords
import pandas as pd
import re
import os


# def process_text(filename):
#     with open(filename, 'r', encoding='utf-8') as file:
#         text = []
#         for line in file.readlines():
#             tokens = clean_line(line)
#             for token in tokens:
#                 if (token != '') & (token not in text):
#                     text.append(token)
#
#         return text


def clean_line(line):
    tokens = line.split(" ")
    tokens = map(lambda x: x.lower(), tokens)
    tokens = map(lambda x: re.sub("[^a-zA-Z]+", "", x), tokens)
    tokens = filter(lambda x: x != '', tokens)
    tokens = filter(lambda x: x not in stopwords.words('english'), tokens)
    tokens = list(tokens)
    return tokens


def process_corpus_csv(csvfile):
    df = pd.read_csv(csvfile)
    corpus = set()
    for index, row in df.iterrows():
        text = row[5]
        corpus.update(clean_line(text))

    return corpus


def process_text_csv(csvfile):
    df = pd.read_csv(csvfile)
    texts = []
    for index, row in df.iterrows():
        text = row[5]
        texts.append(clean_line(text))

    return texts


def process_directory(directory):
    texts = []

    for file_name in os.listdir(directory):
        text = set()
        with open(directory + "/" + file_name) as file:
            for line in file.readlines():
                text.update(clean_line(line))
        texts.append(list(text))

    return texts


def process_corpus(reviews):
    corpus = set()
    for review in reviews:
        corpus.update(review.text)
    return corpus


def load_lexicon_words(filename):
    words = []
    polarity = []
    with open(filename) as file:
        for line in file.readlines():
            words.append(line.split(" ")[0])
            polarity.append(float(line.split(" ")[1]))

    return words, polarity









