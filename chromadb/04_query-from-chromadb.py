import chromadb
from chromadb.utils.embedding_functions import ollama_embedding_function
# 找到 ./03_openai-embedding_functions.py中创建的collection


# 创建embedding_functions
ollama_ef = ollama_embedding_function.OllamaEmbeddingFunction(
    url="http://192.168.1.103:11434/api/embeddings",
    model_name="mxbai-embed-large",
)

# 创建chromadb客户端

client = chromadb.PersistentClient(path="./db/database")

# 从database中加载所有的data
collection = client.get_collection("my_story", embedding_function=ollama_ef)

# 执行查询
# 创建query
queries = [
    "Tell me something about Google Pixel",
    "Which tools help with microservices development?",
    "What database is good for scalable web applications?",
    "Which technologies are used for front-end development?",
    "How to manage infrastructure using code?"
]

for query in queries:
    results = collection.query(query_texts=[query], n_results=3)
    print(f"Query: {query}")
    print("Results:", results)
    print("=" * 50)
