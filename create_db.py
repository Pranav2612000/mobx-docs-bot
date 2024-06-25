from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai
import os
import shutil

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']
openai.model = "text-embedding-3-small"
print(os.environ['OPENAI_API_KEY'])

DATA_PATH = "data/docs"
CHROMA_PATH = "chroma"

def main():
    print("Starting DB Generation...")

    documents = load_documents()
    print("Scanned Files: ", [x.metadata["source"] for x in documents])

    chunks = split_text(documents)
    print("Sample Chunk: ")
    print(chunks[7].page_content)
    print(chunks[7].metadata)

    save(chunks)
    print("DB Generation Complete.")

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

def save(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.");

if __name__ == "__main__":
    main()
