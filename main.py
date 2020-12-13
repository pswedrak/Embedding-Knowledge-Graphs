from sentiment_analysis.simon import compute_simon_vector
from text_processing.text_utils import process_text


def main():
    input_tokens = process_text("texts/review_short.txt")
    simon_vector = compute_simon_vector(input_tokens, 50)


if __name__ == "__main__":
    main()
