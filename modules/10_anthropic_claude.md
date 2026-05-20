# Module 10: Anthropic Claude — A Different Kind of AI

> **Estimated Reading Time:** 3 hours  
> **Outcome:** Master the Anthropic API, understand when to prefer Claude over GPT, and build hybrid workflows.

---

## 10.1 Claude's Unique Position in the AI Landscape

Anthropic was founded in 2021 by former OpenAI researchers, including Dario Amodei, explicitly focused on AI safety research. This origin shapes Claude in fundamental ways: the models are trained with a technique called **Constitutional AI (CAI)** that embeds values and ethical guidelines directly into the training process, rather than relying solely on RLHF.

Claude's key differentiators:
- **Long-form reasoning:** Exceptional at maintaining coherent analysis over very long outputs
- **Instruction following:** Highly accurate at following complex, multi-part instructions
- **Nuanced writing:** More natural, less formulaic prose than GPT models
- **200K context window:** Ideal for analyzing entire legal documents or codebases at once
- **Less prone to certain hallucinations:** More likely to say "I don't know" than fabricate

---

```python
import anthropic
from dotenv import load_dotenv

load_dotenv()
claude = anthropic.Anthropic()

# ----- Basic Claude API Call -----
response = claude.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1000,
    messages=[{
        "role": "user",
        "content": "Analyze this arbitration clause for enforceability in India: 'All disputes shall be settled by arbitration in Singapore under ICC Rules.'"
    }]
)
print(response.content[0].text)

# ----- System Prompt with Claude -----
response = claude.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=2000,
    system="""You are a Senior Indian Advocate specializing in commercial arbitration and international trade law.
    Always cite specific sections of the Arbitration and Conciliation Act 1996.
    Be thorough and practical in your analysis.""",
    messages=[{"role": "user", "content": "What is the current legal status of enforcement of foreign arbitral awards in India?"}]
)
print(response.content[0].text)

# ----- Claude for Long Document Analysis (200K context) -----
def analyze_long_document(document_text: str, analysis_request: str) -> str:
    """Use Claude's 200K context window for full document analysis."""
    if len(document_text) > 150000:  # Approx 150K chars
        print(f"Warning: Document is {len(document_text)} chars. May approach context limit.")
    
    response = claude.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": f"""DOCUMENT:\n{document_text}\n\n---\n\nANALYSIS REQUEST: {analysis_request}"""
        }]
    )
    return response.content[0].text

# ----- Streaming with Claude -----
with claude.messages.stream(
    model="claude-3-5-haiku-20241022",  # Fast + cheap
    max_tokens=500,
    messages=[{"role": "user", "content": "Summarize Section 34 of Arbitration Act."}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
print()

# ----- Hybrid Workflow: Claude for drafting, GPT for extraction -----
from openai import OpenAI
openai_client = OpenAI()

def hybrid_document_workflow(facts: dict) -> dict:
    """Use Claude for high-quality drafting, GPT for structured extraction."""
    
    # Step 1: Claude drafts the legal opinion (better prose quality)
    draft = claude.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        system="You are a Senior Advocate. Write formal, precise legal opinions.",
        messages=[{"role": "user", "content":
            f"Write a legal opinion on: {json.dumps(facts)}"}]
    ).content[0].text
    
    # Step 2: GPT extracts structured data from the opinion
    import json
    structured = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content":
            f"Extract from this legal opinion as JSON: {{opinion_summary, risk_level, applicable_sections, recommendations}}\n\n{draft}"}],
        response_format={"type": "json_object"},
        temperature=0.0
    ).choices[0].message.content
    
    return {
        "full_opinion": draft,
        "structured_summary": json.loads(structured)
    }
```

## Claude vs GPT-4: When to Use Which

| Scenario | Prefer Claude | Prefer GPT |
|----------|--------------|------------|
| Full contract analysis | ✓ (200K context) | Context may truncate |
| Long-form opinion writing | ✓ Better prose | Acceptable |
| Structured data extraction | Use either | ✓ Slightly more reliable JSON |
| Function/tool calling | Good | ✓ More mature tooling |
| Speed-sensitive tasks | Haiku is fast | Mini is also fast |
| Cost-sensitive at scale | Haiku cheapest | Mini cheapest |
