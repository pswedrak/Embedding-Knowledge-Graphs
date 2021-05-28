SIMON_MODEL_TRAIN_DISSYMMETRY = 'review_simon_train_dis.txt'
SIMON_MODEL_TEST_DISSYMETRY = 'review_simon_test_dis.txt'
SIMON_MODEL_TRAIN_DISSYMMETRY_CORRECTED = 'review_simon_train_dis_corrected.txt'
SIMON_MODEL_TEST_DISSYMETRY_CORRECTED = 'review_simon_test_dis_corrected.txt'


def convert_results(source, target):
    vectors = []
    with open(source) as source_file:
        for line in source_file.readlines():
            vector = []
            tokens = line.split(", ")
            vector.append(tokens[0][1:])
            for token in tokens[1:50]:
                vector.append(token)
            for token in tokens[100:149]:
                vector.append(token)
            vector.append(tokens[149][:-2])
            vectors.append(vector)
    store_vectors(target, vectors)


def store_vectors(filepath, vectors):
    with open(filepath, 'w') as file:
        for vector in vectors:
            file.write(str(list(map(lambda x: float(x), vector))))
            file.write('\n')


def main():
    convert_results(SIMON_MODEL_TRAIN_DISSYMMETRY, SIMON_MODEL_TRAIN_DISSYMMETRY_CORRECTED)
    # convert_results(SIMON_MODEL_TEST_DISSYMETRY, SIMON_MODEL_TEST_DISSYMETRY_CORRECTED)


if __name__ == "__main__":
    main()
