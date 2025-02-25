import chromadb
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2



ef = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])

chroma_client = chromadb.Client()

collection_name = "first_collection_test"

collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=ef)

collection.add(
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ],
    ids=["id1", "id2"]
)

results = collection.query(
    query_texts=["This is a query document about hawaii"], # Chroma will embed this for you
    n_results=2 # how many results to return
)
print(results)

