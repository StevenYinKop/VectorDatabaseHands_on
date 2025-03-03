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

def query_documents(question, n_results = 2):
    embedding = ollama_client.embed(model=OLLAMA_MODEL, input=question)
    results = collection.query(query_embeddings=list(embedding.embeddings), n_results=n_results)
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    return relevant_chunks

print(query_documents("Tell me about AI replacing TV writers strike"))
