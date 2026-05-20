# Module 8: Retrieval-Augmented Generation (RAG) — Building AI That Knows Your Documents

> **Estimated Reading Time:** 5–6 hours  
> **This is the most practically important module for professional AI applications.**  
> **Outcome:** Build a complete, production-grade RAG pipeline that can answer questions from any document collection with grounded, verifiable answers.

---

## 8.1 The Core Problem RAG Solves

Consider the problem of building an AI assistant for a CA firm in Indore. The AI needs to:

1. Answer questions about specific client files (which change daily)
2. Reference the firm's internal SOPs and checklists
3. Apply specific ITAT judgments and HC decisions the firm has collected
4. Know the details of ongoing assessments and their unique facts
5. Understand the firm's fee structures and client agreements

None of this information is in GPT-4's training data. Even if some of it were, you could not trust the model to recall specific case details accurately without hallucinating. The context window (128,000 tokens for GPT-4.1) is too small to load all documents simultaneously, and doing so would be prohibitively expensive. Even if you could load everything, the "lost in the middle" problem means the model would miss details buried in the middle of a giant context.

**RAG is the solution.** Instead of teaching the model your documents (fine-tuning, expensive and slow to update) or loading all documents into context (expensive and limited), RAG retrieves only the most relevant passages at query time and injects them into the context. The architecture is:

```
User Query → Retrieve relevant passages → Augment context → Generate grounded answer
```

This approach is:
- **Accurate:** Answers are grounded in retrieved documents, dramatically reducing hallucination
- **Scalable:** Works with millions of documents (search is fast)
- **Updatable:** Add new documents without retraining
- **Verifiable:** You can show users which sources were used
- **Cost-efficient:** Only relevant content is sent to the LLM

RAG has become the dominant architecture for enterprise AI applications, professional tools, and any system where knowledge must be precise and up-to-date.

---

> ### 📌 SIDENOTE: Fine-tuning vs. RAG — When to Use Which?
>
> *A critical architectural decision that many developers get wrong.*
>
> **Fine-tuning** modifies the model's weights to encode new knowledge or behaviors. It is like giving an employee a formal education in your domain.
>
> **RAG** retrieves external documents at inference time. It is like giving an employee a reference library they can look up during a task.
>
> **Use Fine-tuning when:**
> - You need to change how the model communicates (tone, format, style)
> - You need to teach a new skill or behavior (not facts)
> - You have 1,000+ high-quality examples of inputs and desired outputs
> - The knowledge is static and rarely changes
>
> **Use RAG when:**
> - You need factual accuracy about specific documents
> - Your knowledge base changes frequently (new cases, updated provisions)
> - You need source attribution ("this answer comes from page 4 of X")
> - You need to scale to millions of documents
> - You want to update knowledge without retraining
>
> **The answer for most professional applications:** RAG first. Fine-tune only if RAG doesn't achieve your quality goals, and only for behavioral changes (style, format), not factual updates.

---

## 8.2 The RAG Pipeline: Five Stages

A complete RAG pipeline has five sequential stages:

### Stage 1: Document Ingestion
Load raw documents from various sources (PDFs, Word files, URLs, databases) and prepare them for processing. This stage handles file reading and initial cleaning.

### Stage 2: Chunking
Split documents into smaller, semantically coherent pieces. This is the most underappreciated step in RAG — chunking quality has a larger impact on RAG quality than almost any other factor.

### Stage 3: Embedding
Convert each chunk into a vector (embedding) that represents its meaning. This was covered in Module 7.

### Stage 4: Retrieval
Given a user query, embed the query and find the most similar chunks using vector similarity search.

### Stage 5: Generation
Inject the retrieved chunks into the LLM context and generate a grounded response.

---

## 8.3 Document Loading

```python
import os
from pathlib import Path
from openai import OpenAI
import fitz  # PyMuPDF
from docx import Document as DocxDocument
import requests
from bs4 import BeautifulSoup

client = OpenAI()

def load_pdf(file_path: str) -> str:
    """Extract text from PDF, including complex layouts."""
    doc = fitz.open(file_path)
    full_text = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            full_text.append(f"--- Page {page_num + 1} ---\n{text}")
    return "\n".join(full_text)

def load_docx(file_path: str) -> str:
    """Extract text from Word document."""
    doc = DocxDocument(file_path)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)

def load_url(url: str) -> str:
    """Extract clean text from a URL."""
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')
    for element in soup(['script', 'style', 'nav', 'footer', 'header']):
        element.decompose()
    return soup.get_text(separator='\n', strip=True)

def load_text_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def auto_load_document(file_path: str) -> dict:
    """Automatically load any supported document type."""
    path = Path(file_path)
    loaders = {
        '.pdf': load_pdf,
        '.docx': load_docx,
        '.doc': load_docx,
        '.txt': load_text_file,
        '.md': load_text_file,
    }
    
    if path.suffix.lower() not in loaders:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    
    content = loaders[path.suffix.lower()](file_path)
    return {
        "filename": path.name,
        "file_path": file_path,
        "content": content,
        "file_type": path.suffix.lower(),
        "size_chars": len(content)
    }
```

---

## 8.4 Chunking Strategies — The Most Critical Step

Chunking — splitting documents into passages for indexing — is where most RAG systems fail. The core tension:

- **Too small:** Each chunk lacks context. "The penalty is Rs. 10,000" is useless without knowing *which* penalty.
- **Too large:** Each chunk contains too many topics. The embedding averages across all of them, reducing precision.

The goal is **semantically coherent** chunks: each chunk should discuss one idea thoroughly, with enough context to be understood in isolation.

```python
from typing import Optional

def chunk_by_fixed_size(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> list[dict]:
    """Simple fixed-size chunking with overlap."""
    tokens_approx = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(tokens_approx), step):
        chunk_tokens = tokens_approx[i:i + chunk_size]
        if len(chunk_tokens) < 50:  # Skip tiny trailing chunks
            continue
        chunks.append({
            "text": " ".join(chunk_tokens),
            "chunk_index": len(chunks),
            "start_token": i,
            "end_token": i + len(chunk_tokens),
        })
    return chunks

def chunk_by_paragraphs(
    text: str,
    max_chunk_size: int = 800,
    overlap_paragraphs: int = 1
) -> list[dict]:
    """Semantic chunking by paragraphs (much better for legal text)."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk = []
    current_size = 0
    
    for i, para in enumerate(paragraphs):
        para_size = len(para.split())
        if current_size + para_size > max_chunk_size and current_chunk:
            chunks.append({
                "text": "\n\n".join(current_chunk),
                "chunk_index": len(chunks),
                "paragraph_start": i - len(current_chunk),
                "paragraph_end": i - 1,
            })
            current_chunk = current_chunk[-overlap_paragraphs:]  # Keep overlap
            current_size = sum(len(p.split()) for p in current_chunk)
        
        current_chunk.append(para)
        current_size += para_size
    
    if current_chunk:
        chunks.append({
            "text": "\n\n".join(current_chunk),
            "chunk_index": len(chunks),
        })
    return chunks

def chunk_legal_document(
    text: str,
    section_markers: list[str] = None
) -> list[dict]:
    """Legal-specific chunking: split at section boundaries."""
    import re
    if section_markers is None:
        section_markers = [
            r'^Section \d+',
            r'^\d+\.',
            r'^WHEREAS',
            r'^NOW THEREFORE',
            r'^ARTICLE \d+',
            r'^CLAUSE \d+',
        ]
    
    combined_pattern = '|'.join(f'({m})' for m in section_markers)
    sections = re.split(combined_pattern, text, flags=re.MULTILINE)
    sections = [s.strip() for s in sections if s and s.strip()]
    
    chunks = []
    for i, section in enumerate(sections):
        if len(section.split()) > 50:  # Skip very short sections
            chunks.append({
                "text": section,
                "chunk_index": len(chunks),
                "section_number": i,
                "is_legal_section": True,
            })
    return chunks
```

---

## 8.5 Building the Complete RAG Pipeline

```python
import chromadb
from chromadb.utils import embedding_functions
import json
from datetime import datetime

class RAGPipeline:
    """
    Production-grade RAG pipeline for document question-answering.
    """
    
    def __init__(
        self,
        collection_name: str,
        db_path: str = "./rag_index",
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4.1-mini"
    ):
        self.llm_client = OpenAI()
        self.llm_model = llm_model
        self.collection_name = collection_name
        
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=embedding_model
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef,
            metadata={"created": str(datetime.now()), "embedding_model": embedding_model}
        )
    
    def ingest_document(
        self,
        file_path: str,
        chunk_strategy: str = "paragraphs",
        metadata_override: dict = None
    ) -> int:
        """Load, chunk, and index a document. Returns number of chunks added."""
        doc = auto_load_document(file_path)
        
        if chunk_strategy == "paragraphs":
            chunks = chunk_by_paragraphs(doc['content'])
        elif chunk_strategy == "legal":
            chunks = chunk_legal_document(doc['content'])
        else:
            chunks = chunk_by_fixed_size(doc['content'])
        
        base_metadata = {
            "source_file": doc['filename'],
            "file_path": file_path,
            "file_type": doc['file_type'],
            "ingested_at": str(datetime.now()),
            "total_chunks": len(chunks),
        }
        if metadata_override:
            base_metadata.update(metadata_override)
        
        texts = [c['text'] for c in chunks]
        ids = [f"{doc['filename']}_chunk_{c['chunk_index']}" for c in chunks]
        metadatas = [{**base_metadata, **{k: v for k, v in c.items() if k != 'text'}} 
                     for c in chunks]
        
        self.collection.add(documents=texts, ids=ids, metadatas=metadatas)
        print(f"Indexed '{doc['filename']}': {len(chunks)} chunks")
        return len(chunks)
    
    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        where_filter: dict = None
    ) -> list[dict]:
        """Retrieve most relevant chunks for a query."""
        kwargs = {
            "query_texts": [query],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"]
        }
        if where_filter:
            kwargs["where"] = where_filter
        
        results = self.collection.query(**kwargs)
        
        retrieved = []
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            retrieved.append({
                "text": doc,
                "source": meta.get('source_file', 'Unknown'),
                "similarity": round(1 - dist, 4),
                "metadata": meta
            })
        return retrieved
    
    def ask(
        self,
        question: str,
        n_results: int = 5,
        system_role: str = None,
        where_filter: dict = None,
        show_sources: bool = True
    ) -> dict:
        """Full RAG pipeline: retrieve + generate."""
        
        retrieved_chunks = self.retrieve(question, n_results, where_filter)
        
        if not retrieved_chunks:
            return {"answer": "No relevant documents found in the knowledge base.", 
                    "sources": [], "retrieved_chunks": []}
        
        # Build context block
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"[SOURCE {i}: {chunk['source']} | Relevance: {chunk['similarity']:.3f}]\n"
                f"{chunk['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)
        
        # Build the augmented prompt
        augmented_prompt = f"""Use ONLY the following sources to answer the question.
If the answer is not in the sources, say "I cannot find this in the provided documents."
Always cite which source number(s) your answer is based on.

SOURCES:
{context}

---

QUESTION: {question}

ANSWER (cite source numbers like [SOURCE 1], [SOURCE 2]):"""        
        
        if system_role is None:
            system_role = "You are a precise AI assistant. Answer only based on provided sources. Never fabricate information."
        
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": augmented_prompt}
            ],
            temperature=0.0,  # Deterministic for factual queries
        )
        
        answer = response.choices[0].message.content
        sources = [c['source'] for c in retrieved_chunks if c['similarity'] > 0.6]
        
        return {
            "question": question,
            "answer": answer,
            "sources": list(set(sources)),
            "retrieved_chunks": retrieved_chunks,
            "tokens_used": response.usage.total_tokens,
        }
    
    def get_stats(self) -> dict:
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
        }


# Usage Example: CA Practice Knowledge Base
if __name__ == "__main__":
    rag = RAGPipeline(
        collection_name="ca_knowledge_base",
        db_path="./ca_rag"
    )
    
    # Ingest documents
    # rag.ingest_document("gst_handbook.pdf", chunk_strategy="legal", metadata_override={"subject": "GST"})
    # rag.ingest_document("income_tax_circular_2024.pdf", chunk_strategy="paragraphs")
    
    # For demo, add text directly
    rag.collection.add(
        documents=[
            "Section 44AB of Income Tax Act requires mandatory tax audit. Threshold is Rs 1 crore for business and Rs 50 lakhs for professions. CA must issue Form 3CB and 3CD report.",
            "GSTR-3B must be filed by 20th of the following month for monthly filers. Late fee is Rs 50 per day (Rs 20 for nil return). Maximum late fee cap is Rs 10,000.",
            "Section 148 notice for reassessment must be issued within 3 years from end of assessment year for escaped income up to Rs 50 lakhs, and 10 years for higher amounts.",
        ],
        ids=["sec44ab", "gstr3b", "sec148"],
        metadatas=[
            {"category": "income_tax", "source_file": "IT_Act.txt"},
            {"category": "gst", "source_file": "GST_Rules.txt"},
            {"category": "income_tax", "source_file": "IT_Act.txt"},
        ]
    )
    
    # Ask questions
    queries = [
        "What is the deadline for GSTR-3B filing?",
        "When can a Section 148 notice be issued for Rs 30 lakh escaped income?",
        "Do I need a tax audit for my Rs 80 lakh consulting income?",
    ]
    
    for q in queries:
        result = rag.ask(q)
        print(f"\nQ: {q}")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print(f"Tokens: {result['tokens_used']}")
```

---

## 8.6 Advanced RAG: Hybrid Search

Pure semantic search (vectors only) has blind spots. Technical identifiers like "Section 44AB," "GSTR-3B," "PAN ABCDE1234F" don't tokenize into semantically meaningful vectors — they're arbitrary identifiers. Hybrid search combines:

- **Dense search:** Vector similarity (semantic meaning)
- **Sparse search:** BM25/TF-IDF keyword matching (exact term matching)

```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRAG:
    """Combines semantic + keyword search for higher accuracy."""
    
    def __init__(self, documents: list[str], embeddings_client, alpha: float = 0.5):
        self.documents = documents
        self.alpha = alpha  # Weight: 0=keyword only, 1=semantic only, 0.5=equal
        
        # Build BM25 index
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        
        # Build embeddings
        response = embeddings_client.embeddings.create(
            input=documents, model="text-embedding-3-small"
        )
        self.embeddings = np.array([r.embedding for r in response.data], dtype=np.float32)
    
    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        # BM25 scores
        bm25_scores = self.bm25.get_scores(query.lower().split())
        
        # Semantic scores
        from openai import OpenAI
        client_embed = OpenAI()
        q_emb = np.array(client_embed.embeddings.create(
            input=[query], model="text-embedding-3-small"
        ).data[0].embedding, dtype=np.float32)
        semantic_scores = np.dot(self.embeddings, q_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb)
        )
        
        # Normalize and combine
        def normalize(scores):
            mn, mx = scores.min(), scores.max()
            return (scores - mn) / (mx - mn + 1e-10)
        
        hybrid_scores = (
            self.alpha * normalize(semantic_scores) +
            (1 - self.alpha) * normalize(bm25_scores)
        )
        
        top_k = np.argsort(hybrid_scores)[::-1][:k]
        return [(self.documents[i], hybrid_scores[i]) for i in top_k]
```

---

## Module 8 Summary

RAG is the architecture that makes AI practically useful for professional knowledge work. By:
1. Ingesting and chunking documents intelligently
2. Embedding chunks into a vector database
3. Retrieving relevant chunks at query time
4. Augmenting the LLM context with retrieved evidence

...you get factual, verifiable, source-attributed AI answers over any document collection.

## Module 8 Exercises

1. Build a RAG system over 10–20 PDF documents from your practice (GST circulars, tax provisions, client agreements)
2. Compare paragraph-level vs. section-level chunking on the same query — which gives better answers?
3. Implement the hybrid search class and test it on queries with specific section numbers
4. Add a conversation history feature to the `RAGPipeline.ask()` method
