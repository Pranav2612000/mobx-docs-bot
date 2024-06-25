from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader

DATA_PATH = "data/docs"

def main():
    print("Starting DB Generation...")
    documents = load_documents()
    print("Scanned Files: ", [x.metadata["source"] for x in documents])

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md", recursive=True)
    documents = loader.load()
    return documents

if __name__ == "__main__":
    main()
