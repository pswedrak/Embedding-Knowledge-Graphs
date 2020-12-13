from nltk.corpus import stopwords


def process_text(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = []
        for line in file.readlines():
            tokens = line.split(" ")
            tokens = map(lambda x: x.lower(), tokens)
            tokens = map(lambda x: x.replace(".", ""), tokens)
            tokens = map(lambda x: x.replace(",", ""), tokens)
            tokens = map(lambda x: x.replace("\n", ""), tokens)
            tokens = map(lambda x: x.replace("’", ""), tokens)
            tokens = filter(lambda x: x.isdigit() is False, tokens)
            tokens = filter(lambda x: x not in stopwords.words('english'), tokens)
            tokens = list(tokens)
            for token in tokens:
                if (token != '') & (token not in text):
                    text.append(token)

        return text


