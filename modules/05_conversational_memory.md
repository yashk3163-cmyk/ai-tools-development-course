# Module 5: Conversational Memory — Making AI Remember

> **Estimated Reading Time:** 3–4 hours  
> **Outcome:** Implement five types of memory for AI applications: buffer, window, summary, entity, and persistent memory with database storage.

---

## 5.1 Why Memory Is Non-Trivial for AI Applications

Every API call to an LLM is completely stateless. The model has no background process running between your calls, no persistent state, no memory of previous interactions unless you explicitly include that history in the current call. This is fundamentally different from databases, web servers, or any other software system you've likely worked with.

This statelessness has both advantages and challenges. The advantage is simplicity and scalability: each call is independent, can be load-balanced across servers, and does not require session management at the infrastructure level. The challenge is that building conversational AI — where the AI should remember what was discussed earlier — requires you to explicitly manage and inject the relevant history into every API call.

Furthermore, you cannot simply accumulate the entire conversation history forever. Context windows have limits. Longer contexts cost more per API call. And as conversations grow longer, the "lost in the middle" problem means earlier important context may be underweighted by the model. Memory management is therefore not just about remembering, but about intelligently selecting *what* to remember and *how* to represent it efficiently.

---

```python
from openai import OpenAI
import json, os
from pathlib import Path
from collections import deque
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# ===== Memory Type 1: Buffer Memory (Full History) =====
class BufferMemory:
    """Keeps all conversation history. Simple, but grows indefinitely."""
    def __init__(self, system_prompt: str):
        self.messages = [{"role": "system", "content": system_prompt}]
    
    def add_user(self, text: str): self.messages.append({"role": "user", "content": text})
    def add_assistant(self, text: str): self.messages.append({"role": "assistant", "content": text})
    def get_messages(self): return self.messages
    def token_count(self): return sum(len(m['content'].split()) * 1.3 for m in self.messages)

# ===== Memory Type 2: Window Memory (Last N turns) =====
class WindowMemory:
    """Keeps only the last k conversation turns. Bounded memory."""
    def __init__(self, system_prompt: str, k: int = 6):
        self.system = {"role": "system", "content": system_prompt}
        self.k = k
        self.window = deque(maxlen=k * 2)  # k turns = 2k messages
    
    def add_user(self, text: str): self.window.append({"role": "user", "content": text})
    def add_assistant(self, text: str): self.window.append({"role": "assistant", "content": text})
    def get_messages(self): return [self.system] + list(self.window)

# ===== Memory Type 3: Summary Memory =====
class SummaryMemory:
    """Summarizes old history to save tokens while preserving key info."""
    def __init__(self, system_prompt: str, summarize_after: int = 10):
        self.system_prompt = system_prompt
        self.summarize_after = summarize_after
        self.summary = ""
        self.recent = []
    
    def add_user(self, text: str): self.recent.append({"role": "user", "content": text})
    def add_assistant(self, text: str):
        self.recent.append({"role": "assistant", "content": text})
        if len(self.recent) >= self.summarize_after:
            self._summarize()
    
    def _summarize(self):
        conv_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in self.recent)
        r = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content":
                f"Summarize this conversation in 3-5 sentences, preserving key facts and decisions:\n\n{conv_text}"}],
            temperature=0.0, max_tokens=200
        )
        new_summary = r.choices[0].message.content
        self.summary = f"{self.summary}\n{new_summary}".strip() if self.summary else new_summary
        self.recent = []
    
    def get_messages(self):
        system = self.system_prompt
        if self.summary:
            system += f"\n\nConversation history summary:\n{self.summary}"
        return [{"role": "system", "content": system}] + self.recent

# ===== Memory Type 4: Entity Memory =====
class EntityMemory:
    """Extracts and remembers specific entities: clients, cases, amounts."""
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.entities = {}
        self.recent = []
    
    def _extract_entities(self, user_msg: str):
        r = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content":
                f"Extract entities from this message as JSON: {{\"persons\": [], \"organizations\": [], \"amounts\": [], \"case_numbers\": [], \"sections\": []}}\n\nMessage: {user_msg}"}],
            response_format={"type": "json_object"}, temperature=0.0
        )
        extracted = json.loads(r.choices[0].message.content)
        for k, v in extracted.items():
            if v:
                self.entities[k] = list(set(self.entities.get(k, []) + v))
    
    def add_user(self, text: str):
        self._extract_entities(text)
        self.recent.append({"role": "user", "content": text})
    
    def add_assistant(self, text: str):
        self.recent.append({"role": "assistant", "content": text})
    
    def get_messages(self):
        system = self.system_prompt
        if self.entities:
            system += f"\n\nKnown entities from this conversation:\n{json.dumps(self.entities, indent=2)}"
        return [{"role": "system", "content": system}] + self.recent[-10:]

# ===== Memory Type 5: Persistent Memory (SQLite) =====
import sqlite3
from datetime import datetime

class PersistentMemory:
    """Saves conversation history to disk. Survives program restarts."""
    def __init__(self, db_path: str, session_id: str, system_prompt: str):
        self.session_id = session_id
        self.system_prompt = system_prompt
        self.conn = sqlite3.connect(db_path)
        self._create_table()
    
    def _create_table(self):
        self.conn.execute("""CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""")
        self.conn.commit()
    
    def add_user(self, text: str):
        self.conn.execute("INSERT INTO messages (session_id, role, content) VALUES (?,?,?)",
                          (self.session_id, "user", text))
        self.conn.commit()
    
    def add_assistant(self, text: str):
        self.conn.execute("INSERT INTO messages (session_id, role, content) VALUES (?,?,?)",
                          (self.session_id, "assistant", text))
        self.conn.commit()
    
    def get_messages(self, last_n: int = 20):
        rows = self.conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (self.session_id, last_n)
        ).fetchall()
        history = [{"role": r, "content": c} for r, c in reversed(rows)]
        return [{"role": "system", "content": self.system_prompt}] + history

# ===== Unified Chat Function =====
def chat_with_memory(memory_obj, user_input: str) -> str:
    memory_obj.add_user(user_input)
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=memory_obj.get_messages(),
        temperature=0.3
    )
    reply = response.choices[0].message.content
    memory_obj.add_assistant(reply)
    return reply

# Test all memory types
if __name__ == "__main__":
    sp = "You are a CA assistant. Remember client details throughout our conversation."
    mem = EntityMemory(sp)
    
    r1 = chat_with_memory(mem, "My client Mr. Sharma runs a trading business with Rs 1.5 crore turnover.")
    r2 = chat_with_memory(mem, "Does he need a tax audit?")
    r3 = chat_with_memory(mem, "What form does the CA need to submit?")
    
    print(r2)  # Should know it's about Mr. Sharma
    print(r3)  # Should know about the audit context
    print("Entities tracked:", mem.entities)
```

## Module 5 Summary

| Memory Type | Best For | Limitation |
|-------------|----------|------------|
| Buffer | Short conversations, testing | Grows indefinitely |
| Window | Production chatbots | Loses early context |
| Summary | Long conversations | Summary may lose details |
| Entity | Client/case tracking | Only captures named entities |
| Persistent | Multi-session applications | Requires database setup |
