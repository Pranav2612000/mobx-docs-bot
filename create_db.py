from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_PATH = "data/docs"

def main():
    print("Starting DB Generation...")

    documents = load_documents()
    print("Scanned Files: ", [x.metadata["source"] for x in documents])

    chunks = split_text(documents)
    print("Sample Chunk: ")
    print(chunks[7].page_content)
    print(chunks[7].metadata)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md", recursive=True)
    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks;

if __name__ == "__main__":
    main()
