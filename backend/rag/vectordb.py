import faiss
import numpy as np
import pickle

class VectorDB:

    def __init__(self, dimension):

        self.index = faiss.IndexFlatL2(dimension)

        self.text_chunks = []

    def add_embeddings(self, embeddings, chunks):

        self.index.add(
            np.array(embeddings).astype("float32")
        )

        self.text_chunks.extend(chunks)

    def save(self):

        faiss.write_index(
            self.index,
            "vectorstore/faiss_index.bin"
        )

        with open(
            "vectorstore/chunks.pkl",
            "wb"
        ) as f:

            pickle.dump(
                self.text_chunks,
                f
            )