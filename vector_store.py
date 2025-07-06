# vector_store.py

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()

pdf_path = "Kanski’s clinical ophthalmology _ a systematic approach.pdf"
persist_directory = os.path.join("C:\\VisionBuddy-LangGraph_RAG", "chroma_db")
collection_name = "vectorstore"

def vectorstore_exists(path: str) -> bool:
    expected_files = ["chroma-collections.parquet", "chroma-embeddings.parquet", "index"]
    return all(os.path.exists(os.path.join(path, f)) for f in expected_files)

def build_vectorstore():
    print("✅ Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"✅ Loaded PDF: {len(pages)} pages")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(pages)

    print("🆕 Creating new Chroma vector store...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )

    # Chroma >=0.4 auto-saves, no need for persist()
    print("✅ Vector store saved!")

if __name__ == "__main__":
    if vectorstore_exists(persist_directory):
        print("✅ Vector store already exists. No need to recreate.")
    else:
        build_vectorstore()


