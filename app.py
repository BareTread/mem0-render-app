from mem0 import Memory
from fastapi import FastAPI
import os
import chromadb

# Setup ChromaDB with persistent storage
chroma_client = chromadb.PersistentClient(path="/data/chroma")

app = FastAPI()

# Configure Mem0 with ChromaDB
config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "client": chroma_client,
            "collection_name": "mem0"
        }
    }
}

m = Memory(config=config)

@app.get("/")
def hello():
    return {"message": "Mem0 + ChromaDB running!", "status": "healthy"}

@app.post("/add")
def add_memory(text: str):
    result = m.add(text, user_id="default")
    return {"result": result}

@app.get("/search")
def search_memories(query: str):
    results = m.search(query, user_id="default")
    return {"results": results}

@app.get("/memories")
def get_all_memories():
    results = m.get_all(user_id="default")
    return {"memories": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))