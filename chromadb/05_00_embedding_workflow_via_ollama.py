import os

import ollama
import chromadb
from chromadb.utils.embedding_functions import ollama_embedding_function
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

# 找到 ./03_openai-embedding_functions.py中创建的collection

# 创建embedding_functions
ollama_ef = ollama_embedding_function.OllamaEmbeddingFunction(
    url= f"{OLLAMA_HOST}/api/embeddings",
    model_name=OLLAMA_MODEL,
)

ollama_client = ollama.Client(host=OLLAMA_HOST)


# 创建chromadb客户端

client = chromadb.PersistentClient(path="./db/database")

# 从database中加载所有的data
collection = client.get_or_create_collection("my_story_chunks", embedding_function=ollama_ef)

def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), "r") as f:
                documents.append({"id": filename, "text": f.read()})
    return documents


# Split text into chunks
def chop_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

def calculate_embeddings(text):
    print("==== Calculating embeddings ====")
    response = ollama_client.embed(
        model=OLLAMA_MODEL,
        input=text
    )
    return response.embeddings

directory_path = "../materials"
documents = load_documents_from_directory(directory_path)

print(f"Loaded {len(documents)} documents")

chunked_documents = []
for doc in documents:
    chunks = chop_text(doc["text"])
    print(f"==== Splitting doc {doc["id"]} into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append(
            {"id": f"{doc["id"]}_{i}", "text": chunk, "embeddings": calculate_embeddings(chunk)}
        )

for doc in chunked_documents:
    print("==== Inserting chunks into db ====")
    collection.upsert(ids=[doc["id"]], embeddings=list(doc["embeddings"]), documents=[doc["text"]])





