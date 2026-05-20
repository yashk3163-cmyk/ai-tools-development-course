# Module 11: Local LLMs with Ollama — Privacy-First AI

> **Estimated Reading Time:** 3 hours  
> **Critical for handling confidential client data that must not leave your infrastructure.**  
> **Outcome:** Run powerful open-source LLMs locally with zero API costs and complete data privacy.

---

## 11.1 Why Local LLMs Matter for Professional Practice

Every time you send client data to the OpenAI or Anthropic API, that data travels across the internet to servers in the United States and is processed by a foreign company's infrastructure. For most general tasks, this is acceptable. But consider:

- **Client confidentiality (Bar Council Rules):** Advocate-client privilege may be compromised if sensitive communications are processed by third-party servers
- **Chartered Accountants Act:** Client financial data is confidential professional information
- **DPDP Act 2023:** Personal data of Indian citizens has specific processing requirements
- **Assessment proceedings:** Documents related to ongoing assessments are often privileged
- **Cost at scale:** Processing thousands of documents through cloud APIs becomes expensive

Local LLMs — models running entirely on your own machine — solve all of these concerns. The data never leaves your computer. Inference is completely free (after hardware). And modern open-source models are surprisingly capable.

---

```bash
# Installation (Run in terminal)
# Mac/Linux:
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download installer from https://ollama.ai/download

# Download models
ollama pull llama3.2          # 2GB - Fast, good quality
ollama pull llama3.3:70b      # 40GB - Near GPT-4 quality (needs 16GB+ RAM)
ollama pull nomic-embed-text  # 274MB - Local embeddings for RAG
ollama pull mistral           # 4GB - Excellent code and analysis

# List installed models
ollama list

# Run interactively
ollama run llama3.2
```

```python
import requests
import json
from openai import OpenAI  # Ollama has OpenAI-compatible API
from dotenv import load_dotenv

load_dotenv()

# Method 1: Ollama REST API directly
def local_chat_raw(prompt: str, model: str = "llama3.2") -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120
    )
    return response.json()["response"]

# Method 2: OpenAI-compatible API (recommended)
local_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Ollama doesn't need a real key
)

def local_chat(messages: list, model: str = "llama3.2") -> str:
    response = local_client.chat.completions.create(
        model=model, messages=messages, temperature=0.3
    )
    return response.choices[0].message.content

# Test local inference
result = local_chat([
    {"role": "system", "content": "You are an expert Indian CA."},
    {"role": "user", "content": "What is Section 44AB?"}
])
print("Local LLM response:", result)

# ----- Local RAG with Private Data -----
import chromadb
from chromadb.utils import embedding_functions

def build_private_rag(documents: list[dict]) -> chromadb.Collection:
    """Build a RAG system using only local, private components."""
    client_chroma = chromadb.PersistentClient(path="./private_rag")
    
    # Use Ollama's local embedding model (nomic-embed-text)
    local_ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text"
    )
    
    collection = client_chroma.get_or_create_collection(
        name="private_client_files",
        embedding_function=local_ef
    )
    
    collection.add(
        documents=[d["content"] for d in documents],
        ids=[d["id"] for d in documents],
        metadatas=[d.get("metadata", {}) for d in documents]
    )
    return collection

def private_qa(collection, question: str, model: str = "llama3.2") -> str:
    """100% private question answering: local embeddings + local LLM."""
    results = collection.query(query_texts=[question], n_results=3,
                               include=["documents"])
    context = "\n\n".join(results['documents'][0])
    
    return local_chat([
        {"role": "system", "content": "Answer questions based ONLY on provided context. Be precise."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ], model=model)

# Model comparison helper
def compare_models(query: str, models: list[str] = ["llama3.2", "mistral"]) -> dict:
    results = {}
    for model in models:
        try:
            import time
            start = time.time()
            resp = local_chat([{"role": "user", "content": query}], model=model)
            elapsed = time.time() - start
            results[model] = {"response": resp, "time_seconds": round(elapsed, 2)}
        except Exception as e:
            results[model] = {"error": str(e)}
    return results
```

## 11.2 Model Selection Guide for Local Deployment

| Model | RAM Needed | Speed | Quality | Best For |
|-------|-----------|-------|---------|----------|
| llama3.2:1b | 1GB | Very fast | Basic | Simple queries, testing |
| llama3.2:3b | 2GB | Fast | Good | Most tasks |
| mistral:7b | 4GB | Fast | Very good | Code, analysis |
| llama3.3:70b | 40GB | Slow | Excellent | Complex reasoning |
| nomic-embed-text | 274MB | Fast | Excellent | Embeddings only |
