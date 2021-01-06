from gensim.models.doc2vec import TaggedDocument, Doc2Vec


def generate_doc2vec(reviews):
    embeddings = []
    documents = [TaggedDocument(doc.text, [i]) for i, doc in enumerate(reviews)]
    doc2vec_model = Doc2Vec(documents, vector_size=100, window=3, min_count=1)

    for review in reviews:
        embeddings.append(doc2vec_model.infer_vector(review.text).tolist())

    return embeddings
