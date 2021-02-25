from common.constants import REVIEW_TOKENS_PATH
from text_processing.yelp_utils import read_reviews
from nltk.corpus import sentiwordnet as swn


def main():
    reviews = read_reviews('../' + REVIEW_TOKENS_PATH)
    words = set()
    for review in reviews:
        words.update(review.text)

    for word in words:
        for synset in swn.senti_synsets(word):
            if (synset.pos_score() == synset.neg_score()) & (synset.pos_score() != 0) & (synset.obj_score() != 0):
                print(synset)


if __name__ == "__main__":
    main()
