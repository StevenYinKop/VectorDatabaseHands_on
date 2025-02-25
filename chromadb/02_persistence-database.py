import chromadb
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2



ef = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])

chroma_client = chromadb.PersistentClient(path="./db/database")

collection_name = "tech_collection_test"

collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=ef)


# 添加 20 条技术相关的数据
documents = [
    "Python is a versatile programming language commonly used for web development, data science, and automation.",
    "Java is a powerful object-oriented language widely used in enterprise applications and Android development.",
    "JavaScript is a high-level, interpreted programming language primarily used for building interactive web applications.",
    "C++ is a compiled language often used in game development, high-performance applications, and system programming.",
    "Go is a statically typed language developed by Google, known for its concurrency support and efficiency in backend systems.",
    "Rust is a systems programming language focused on safety, speed, and memory management without garbage collection.",
    "Spring Boot is a Java framework that simplifies microservice development by providing auto-configuration and built-in tools.",
    "Django is a high-level Python web framework that promotes rapid development and clean, pragmatic design.",
    "React is a popular JavaScript library for building user interfaces, developed and maintained by Facebook.",
    "Angular is a TypeScript-based web application framework developed by Google, providing a full MVC structure.",
    "Docker is a containerization tool that allows developers to package applications and dependencies into lightweight containers.",
    "Kubernetes is an open-source container orchestration platform designed to automate the deployment and scaling of applications.",
    "Redis is an in-memory key-value database often used for caching, session management, and real-time analytics.",
    "MongoDB is a NoSQL database that stores data in flexible JSON-like documents, commonly used in modern web applications.",
    "PostgreSQL is an advanced, open-source relational database known for its strong ACID compliance and extensibility.",
    "Git is a distributed version control system that helps developers track changes in source code and collaborate efficiently.",
    "Jenkins is an open-source automation server used for continuous integration and continuous delivery (CI/CD).",
    "Terraform is an infrastructure-as-code (IaC) tool that enables the automated provisioning of cloud resources.",
    "GraphQL is a query language for APIs that allows clients to request only the data they need, improving efficiency.",
    "RESTful APIs follow REST principles and use HTTP methods to enable communication between clients and servers."
]

ids = [f"id{i}" for i in range(1, 21)]

# 添加数据到集合
collection.add(documents=documents, ids=ids)

# 执行查询
queries = [
    "What is a good programming language for data science?",  # 可能匹配 "Python"
    "Which tools help with microservices development?",  # 可能匹配 "Spring Boot", "Docker", "Kubernetes"
    "What database is good for scalable web applications?",  # 可能匹配 "MongoDB", "PostgreSQL"
    "Which technologies are used for front-end development?",  # 可能匹配 "React", "Angular", "JavaScript"
    "How to manage infrastructure using code?"  # 可能匹配 "Terraform", "Kubernetes"
]

for query in queries:
    results = collection.query(query_texts=[query], n_results=3)
    print(f"Query: {query}")
    print("Results:", results)
    print("=" * 50)
