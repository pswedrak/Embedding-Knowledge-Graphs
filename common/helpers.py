def store_vectors(filepath, vectors):
    with open(filepath, 'w') as file:
        for vector in vectors:
            file.write(str(vector))
            file.write('\n')


def store_vector(filepath, vector):
    with open(filepath, 'a') as file:
        file.write(str(vector))
        file.write('\n')
