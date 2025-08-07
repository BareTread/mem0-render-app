from mem0 import Memory
from fastapi import FastAPI
import os
import chromadb

# Setup ChromaDB with persistent storage (use /tmp for Render)
chroma_client = chromadb.PersistentClient(path="/tmp/chroma")

app = FastAPI()

# Configure Mem0 with ChromaDB and environment variables
config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "client": chroma_client,
            "collection_name": "mem0"
        }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0.2,
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    }
}

m = Memory.from_config(config)

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

@app.get("/visualize")
def visualize_memories():
    """Show memory relationships and structure"""
    memories = m.get_all(user_id="default")
    
    # Create a simple visualization structure
    visualization = {
        "total_memories": len(memories),
        "memory_structure": "vector_store",
        "backend": "ChromaDB",
        "graph_enabled": False,
        "memories_with_embeddings": []
    }
    
    for memory in memories:
        visualization["memories_with_embeddings"].append({
            "id": memory.get("id"),
            "text": memory.get("memory"),
            "created": memory.get("created_at"),
            "metadata": memory.get("metadata", {}),
            "hash": memory.get("hash"),
            "note": "Stored as vector embedding, not graph node"
        })
    
    visualization["explanation"] = (
        "This mem0 instance uses ChromaDB (vector store), not a graph database. "
        "Memories are stored as embeddings for similarity search, not as connected nodes. "
        "To get graph functionality, you'd need to configure mem0 with Neo4j or similar."
    )
    
    return visualization

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))