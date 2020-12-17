from sentiment_analysis.simon import compute_simon_vector
from text_processing.text_utils import process_text
import time


def main():
    start = time.time()
    input_tokens = process_text("texts/review_short.txt")
    simon_vector = compute_simon_vector(input_tokens, 10)
    end = time.time()
    #print(end - start)
    print(simon_vector)


if __name__ == "__main__":
    main()
