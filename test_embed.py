from embeddings import embed_one

test_vector = embed_one("test sentence")
print(len(test_vector))  # should print 768