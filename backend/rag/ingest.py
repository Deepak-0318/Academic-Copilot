import os

from tools.pdf_tool import PDFTool
from rag.chunker import Chunker
from rag.embedder import Embedder
from rag.vectordb import VectorDB

DATA_FOLDER = "data"


def main():

    all_chunks = []

    # STEP 1 — READ ALL PDFs

    for subject in os.listdir(DATA_FOLDER):

        subject_path = os.path.join(
            DATA_FOLDER,
            subject
        )

        if os.path.isdir(subject_path):

            for file in os.listdir(subject_path):

                if file.endswith(".pdf"):

                    pdf_path = os.path.join(
                        subject_path,
                        file
                    )

                    print(f"Processing: {pdf_path}")

                    # Extract Text
                    text = PDFTool.extract_text(
                        pdf_path
                    )

                    # Chunk Text
                    chunks = Chunker.split_text(
                        text
                    )

                    all_chunks.extend(chunks)

    # STEP 2 — EMBEDDINGS

    embedder = Embedder()

    embeddings = embedder.generate_embeddings(
        all_chunks
    )

    # STEP 3 — STORE IN VECTOR DB

    dimension = embeddings.shape[1]

    vectordb = VectorDB(dimension)

    vectordb.add_embeddings(
        embeddings,
        all_chunks
    )

    vectordb.save()

    print("Knowledge Base Created Successfully")


if __name__ == "__main__":
    main()
