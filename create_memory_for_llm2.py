import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Set path to the 'data/' folder
DATA_PATH = "data/"

# Function to load PDFs with metadata (filename as source)
def load_pdf_files_with_metadata(data_path):
    all_documents = []
    book_names = []  # List to keep track of the loaded books

    # Iterate over all files in the data folder
    for file_name in os.listdir(data_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(data_path, file_name)
            loader = PyPDFium2Loader(file_path)
            try:
                # Load documents from the PDF
                docs = loader.load()
                # Add book name (filename) as metadata
                for doc in docs:
                    doc.metadata["source"] = os.path.splitext(file_name)[0]
                # Add the documents to the list
                all_documents.extend(docs)
                # Keep track of the books
                book_names.append(file_name)
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
    
    # Print how many books were loaded
    print(f"✅ Loaded {len(book_names)} books:")
    for book in book_names:
        print(f" - {book}")
    
    return all_documents

# Step 2: Create Chunks from the loaded documents
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Step 3: Get the embedding model
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

# Step 4: Create and Save Vector Store
def create_and_save_vector_store(text_chunks, embedding_model):
    DB_FAISS_PATH = "vectorstore/db_faiss"
    # Create the FAISS vector store from the text chunks
    db = FAISS.from_documents(text_chunks, embedding_model)
    # Save the FAISS vector store to the specified path
    db.save_local(DB_FAISS_PATH)
    print(f"✅ FAISS Vector Store saved at {DB_FAISS_PATH}")

# Main process to load PDFs, create chunks, and store embeddings
def main():
    # Load documents from the data folder
    documents = load_pdf_files_with_metadata(DATA_PATH)
    
    # Create text chunks
    text_chunks = create_chunks(extracted_data=documents)
    print(f"✅ Created {len(text_chunks)} chunks from the loaded PDFs.")

    # Get embedding model
    embedding_model = get_embedding_model()

    # Create and save the vector store
    create_and_save_vector_store(text_chunks, embedding_model)

if __name__ == "__main__":
    main()
