import os

import chromadb
from openai import OpenAI
from dotenv import load_dotenv

from chromadb.utils.embedding_functions import ollama_embedding_function



# 创建embedding_functions
ollama_ef = ollama_embedding_function.OllamaEmbeddingFunction(
    url="http://192.168.1.101:11434/api/embeddings",
    model_name="mxbai-embed-large",
)

# 创建chromadb客户端

client = chromadb.PersistentClient(path="./db/database")
# 从chromadb的客户端中查询/创建collection，并且指定相应的embedding_functions

collection = client.get_or_create_collection("my_story", embedding_function=ollama_ef)

# 构建需要传入的documents集合。
for filename in os.listdir(f"{os.getcwd()}/../materials/"):
    filepath = f"{os.getcwd()}/../materials/{filename}"
    try:
        with open(filepath, "r",  encoding="utf-8") as f:
            text = f.read()
            # 将构建好的文档传入集合中并且保存至数据库
            print(f"Prepare file: {filename}, document size: {len(text)}")
            collection.upsert(ids=[filename], documents=[text])
            print(f"Insert {filename} Into Database Complete.")
    except Exception as e:
        print(f"Error processing {filename}: {e}")
