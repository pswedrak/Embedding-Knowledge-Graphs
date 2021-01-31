def prepare_corpus_for_glove(corpusfile, outputfile):
    output = open(outputfile, 'w')
    with open(corpusfile) as file:
        tokens = file.readline()
        while len(tokens) > 0:
            tokens = tokens[1:-2]
            tokens = tokens.split(", ")
            tokens = list(map(lambda x: x[1:-1], tokens))
            for token in tokens:
                output.write(token)
                output.write(' ')
            output.write('\n')
            file.readline()
            tokens = file.readline()
    output.close()
