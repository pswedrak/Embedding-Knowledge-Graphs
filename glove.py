from common.constants import REVIEW_TOKENS_PATH, GLOVE_CORPUS
from glove.glove import prepare_corpus_for_glove


def main():
    prepare_corpus_for_glove(REVIEW_TOKENS_PATH, GLOVE_CORPUS)


if __name__ == "__main__":
    main()
