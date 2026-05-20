# Module 3: The OpenAI API — A Complete Deep Dive

> **Estimated Reading Time:** 3–4 hours  
> **Outcome:** Master every aspect of the OpenAI Chat Completions API including streaming, structured outputs, token counting, error handling, and cost management.

---

## 3.1 Understanding the Chat Completions API Architecture

The OpenAI API is built around a simple but powerful abstraction: the **chat completion**. Every interaction with the model is framed as a conversation — a sequence of messages from different "roles" — and the model generates the next message in that conversation.

This conversation-centric design is not just a UX choice. It reflects a fundamental truth about how LLMs work: context is everything. The model's output at any point is a function of everything that came before it in the conversation. By structuring interactions as conversations with explicit roles (system, user, assistant), the API gives you precise control over what context the model sees.

The three message roles:

**System:** Instructions, persona, and constraints. The system message is the "constitution" of your interaction. It sets the model's role, behavior, tone, scope, and output format. The system message is read by the model before any user messages, and its instructions persist throughout the conversation. This is where your carefully engineered system prompt (from Module 4) lives.

**User:** The human's input. In single-turn applications, this is the query. In multi-turn conversations, this alternates with assistant messages to build the conversation history.

**Assistant:** The model's previous responses. When building a multi-turn conversation, you include previous assistant responses in the message list so the model has context of what it already said.

---

> ### 📌 SIDENOTE: Why Does the API Use HTTP REST? (Networking Background)
>
> *For Python developers new to APIs:*
>
> The OpenAI API is a **REST API** — Representational State Transfer. This means:
> - Your Python code sends HTTP requests (like a web browser loading a page) to OpenAI's servers
> - The API endpoint is a URL: `https://api.openai.com/v1/chat/completions`
> - Your request contains JSON data (the messages, model name, parameters)
> - OpenAI's server processes the request, runs the model, and returns JSON data
> - Your Python code reads the JSON response
>
> The `openai` Python library wraps all of this HTTP machinery in a clean Python interface. When you call `client.chat.completions.create(...)`, behind the scenes it:
> 1. Serializes your Python objects to JSON
> 2. Sends an HTTP POST request with your API key as authentication
> 3. Receives the HTTP response
> 4. Deserializes the JSON into Python objects you can work with
>
> **Why this matters:** If the API call fails, you might get HTTP error codes: 401 (bad API key), 429 (rate limit exceeded), 500 (server error). Understanding these helps you write better error handling.

---

```python
from openai import OpenAI
from dotenv import load_dotenv
import json, time, os
from typing import Optional

load_dotenv()
client = OpenAI()

# ----- Basic Chat Completion -----
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": "You are an expert Indian CA."},
        {"role": "user", "content": "What is the due date for GSTR-9?"}
    ],
    temperature=0.3,
    max_tokens=500,
)
print(response.choices[0].message.content)
print(f"Tokens: {response.usage.prompt_tokens} in + {response.usage.completion_tokens} out")

# ----- Structured JSON Output -----
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": "Extract tax information as JSON."},
        {"role": "user", "content": "Analyze: Rs 5L income, 18% GST on consulting services"}
    ],
    response_format={"type": "json_object"},
    temperature=0.0,
)
data = json.loads(response.choices[0].message.content)
print(json.dumps(data, indent=2))

# ----- Streaming -----
def stream_response(query: str):
    """Stream the response token by token for real-time output."""
    with client.chat.completions.stream(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": query}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print()

stream_response("Explain Section 44AB in 3 sentences.")

# ----- Multi-turn Conversation -----
conversation = [
    {"role": "system", "content": "You are a tax expert for Indian professionals."}
]

def chat(user_input: str) -> str:
    conversation.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=conversation,
        temperature=0.3
    )
    reply = response.choices[0].message.content
    conversation.append({"role": "assistant", "content": reply})
    return reply

print(chat("What turnover triggers mandatory audit?"))
print(chat("What if it's a professional firm?"))  # Remembers prior context

# ----- Retry with Exponential Backoff -----
import time
def robust_api_call(messages, model="gpt-4.1-mini", max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = (2 ** attempt) + 0.5
            print(f"Error: {e}. Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)

# ----- Token Counting Before API Call -----
import tiktoken

def count_tokens(messages: list, model: str = "gpt-4.1-mini") -> int:
    enc = tiktoken.encoding_for_model(model)
    total = 0
    for msg in messages:
        total += 4  # message overhead
        total += len(enc.encode(msg.get("content", "")))
    return total + 2  # reply priming

msgs = [{"role": "user", "content": "Explain Section 44AB in detail."}]
print(f"This call will use ~{count_tokens(msgs)} input tokens")
print(f"Estimated cost: ${count_tokens(msgs) * 0.40 / 1_000_000:.6f}")
```

## Module 3 Summary
- The API is conversation-based with system/user/assistant roles
- JSON mode forces structured, parseable outputs
- Streaming improves UX for long-form generation
- Always count tokens before large calls to manage costs
- Use exponential backoff for robust error handling
