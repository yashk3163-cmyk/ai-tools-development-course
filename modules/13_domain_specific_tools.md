# Module 13: Building Production Domain-Specific AI Tools

> **Estimated Reading Time:** 4–5 hours  
> **This module brings everything together into a complete production architecture.**  
> **Outcome:** Build a fully production-ready, maintainable AI tool with all professional-grade features.

---

## 13.1 Production Architecture for Professional AI Tools

A production AI tool for professional practice has requirements far beyond a proof-of-concept script:

- **Reliability:** Must handle API failures, rate limits, and unexpected inputs without crashing
- **Security:** Client data must be handled safely; API keys must be protected
- **Cost management:** Token usage must be logged, monitored, and optimized
- **Auditability:** All AI outputs must be traceable to source documents
- **Performance:** Caching for repeated queries; async for parallel processing
- **Maintainability:** Prompts, models, and parameters configurable without code changes

---

```python
from openai import OpenAI
from anthropic import Anthropic
import chromadb
from chromadb.utils import embedding_functions
import json, os, hashlib, time, logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional
from functools import wraps
import sqlite3

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ===== 1. Configuration Management =====
class Config:
    PRIMARY_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4.1-mini")
    FALLBACK_MODEL = "gpt-4.1-mini"
    EMBEDDING_MODEL = "text-embedding-3-small"
    MAX_RETRIES = 3
    CACHE_TTL_SECONDS = 3600
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
    TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))
    VECTOR_DB_PATH = "./ca_knowledge_base"
    USAGE_DB_PATH = "./usage_log.db"

# ===== 2. Usage Logging and Cost Tracking =====
class UsageLogger:
    def __init__(self, db_path: str = Config.USAGE_DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS api_calls (
            id INTEGER PRIMARY KEY,
            timestamp TEXT, model TEXT,
            prompt_tokens INTEGER, completion_tokens INTEGER,
            total_tokens INTEGER, estimated_cost_usd REAL,
            query_hash TEXT, session_id TEXT
        )""")
        self.conn.commit()
    
    COST_PER_1M = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4.1": (2.00, 8.00),
                   "claude-3-5-sonnet-20241022": (3.00, 15.00)}
    
    def log(self, model: str, prompt_tokens: int, completion_tokens: int,
            query_hash: str = "", session_id: str = ""):
        rates = self.COST_PER_1M.get(model, (1.0, 3.0))
        cost = (prompt_tokens * rates[0] + completion_tokens * rates[1]) / 1_000_000
        self.conn.execute(
            "INSERT INTO api_calls VALUES (NULL,?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(), model, prompt_tokens, completion_tokens,
             prompt_tokens + completion_tokens, round(cost, 8), query_hash, session_id)
        )
        self.conn.commit()
        return cost
    
    def get_summary(self) -> dict:
        rows = self.conn.execute(
            "SELECT COUNT(*), SUM(total_tokens), SUM(estimated_cost_usd) FROM api_calls"
        ).fetchone()
        return {"total_calls": rows[0], "total_tokens": rows[1],
                "total_cost_usd": round(rows[2] or 0, 4)}

# ===== 3. Response Caching =====
class QueryCache:
    def __init__(self, ttl: int = Config.CACHE_TTL_SECONDS):
        self.ttl = ttl
        self._cache = {}
    
    def _key(self, query: str, model: str) -> str:
        return hashlib.md5(f"{query}::{model}".encode()).hexdigest()
    
    def get(self, query: str, model: str):
        key = self._key(query, model)
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['response']
        return None
    
    def set(self, query: str, model: str, response: str):
        self._cache[self._key(query, model)] = {
            'response': response, 'timestamp': time.time()
        }

# ===== 4. Complete CA AI Tool =====
class CAPracticeAI:
    """Complete production AI tool for a CA firm."""
    
    def __init__(self):
        self.openai = OpenAI()
        self.anthropic = Anthropic()
        self.logger = UsageLogger()
        self.cache = QueryCache()
        self._init_vector_db()
    
    def _init_vector_db(self):
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=Config.EMBEDDING_MODEL
        )
        client_chroma = chromadb.PersistentClient(path=Config.VECTOR_DB_PATH)
        self.collection = client_chroma.get_or_create_collection(
            name="ca_knowledge", embedding_function=ef
        )
    
    def _retry_api_call(self, func, *args, **kwargs):
        for attempt in range(Config.MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == Config.MAX_RETRIES - 1: raise
                wait = (2 ** attempt) + 0.5
                logger.warning(f"API error (attempt {attempt+1}): {e}. Retrying in {wait:.1f}s")
                time.sleep(wait)
    
    def add_document(self, text: str, doc_id: str, metadata: dict = None):
        """Add a document to the knowledge base."""
        self.collection.add(
            documents=[text], ids=[doc_id],
            metadatas=[metadata or {"added": datetime.now().isoformat()}]
        )
        logger.info(f"Added document '{doc_id}' to knowledge base")
    
    def query(self, question: str, session_id: str = "default",
              use_cache: bool = True) -> dict:
        """Complete query pipeline with RAG, caching, and logging."""
        
        # Check cache
        cached = self.cache.get(question, Config.PRIMARY_MODEL) if use_cache else None
        if cached:
            logger.info("Cache hit")
            return {"answer": cached, "source": "cache", "cost_usd": 0}
        
        # Retrieve context
        results = self.collection.query(
            query_texts=[question], n_results=4,
            include=["documents", "metadatas", "distances"]
        )
        context_parts = []
        for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0],
                                    results['distances'][0]):
            if 1 - dist > 0.5:  # Only include relevant results
                context_parts.append(f"[{meta.get('source', 'KB')}]\n{doc}")
        context = "\n\n---\n\n".join(context_parts)
        
        # Build prompt
        system = """You are TaxAdvisorAI for a CA firm in Indore.
        Answer ONLY from provided context. Cite section numbers.
        Always end with: ⚠️ Verify with latest CBDT/CBIC circulars."""
        user = f"""KNOWLEDGE BASE CONTEXT:\n{context}\n\n---\n\nQUESTION: {question}"""
        
        # API call with retry
        response = self._retry_api_call(
            self.openai.chat.completions.create,
            model=Config.PRIMARY_MODEL,
            messages=[{"role": "system", "content": system},
                       {"role": "user", "content": user}],
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        
        answer = response.choices[0].message.content
        cost = self.logger.log(
            Config.PRIMARY_MODEL, response.usage.prompt_tokens,
            response.usage.completion_tokens, session_id=session_id
        )
        self.cache.set(question, Config.PRIMARY_MODEL, answer)
        
        return {"answer": answer, "sources": [m.get('source', 'KB') for m in results['metadatas'][0]],
                "tokens_used": response.usage.total_tokens, "cost_usd": cost}
    
    def get_usage_summary(self) -> dict:
        return self.logger.get_summary()

# Usage
if __name__ == "__main__":
    ai = CAPracticeAI()
    ai.add_document(
        "Section 44AB requires tax audit for business turnover > Rs 1 crore.",
        "sec_44ab", {"source": "IT Act", "section": "44AB"}
    )
    result = ai.query("Does my Rs 1.5 crore business need tax audit?")
    print(result['answer'])
    print("Cost:", result['cost_usd'])
    print("Usage summary:", ai.get_usage_summary())
