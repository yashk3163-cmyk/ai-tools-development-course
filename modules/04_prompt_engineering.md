# Module 4: Prompt Engineering — The Art and Science of Communicating with AI

> **Estimated Reading Time:** 4–5 hours  
> **This is the highest-leverage module in the course.** Mastery here multiplies the effectiveness of every other module.  
> **Outcome:** You will design reliable, production-grade prompts for any professional task, understand failure modes, and apply the 12 major prompting techniques with confidence.

---

## 4.1 What Prompt Engineering Actually Is

Prompt engineering is the practice of crafting the input text to a language model in ways that reliably produce desired outputs. The word "engineering" is deliberate: this is not guesswork or trial-and-error tweaking. Done well, it is a disciplined, systematic process of designing, testing, iterating, and documenting inputs that produce reliable, predictable outputs across a wide range of inputs.

To understand why prompt engineering is so powerful, you need to revisit a concept from Module 1: LLMs are trained on an enormous corpus of text that contains examples of virtually every kind of human communication. Embedded in that training data are millions of examples of: expert lawyers writing legal briefs, accountants analyzing financial documents, engineers debugging code, doctors explaining diagnoses, teachers explaining difficult concepts, and much more. When you craft your prompt, you are not "instructing" the model in the traditional programming sense — you are **activating** relevant patterns in its training data.

A prompt like "What is GST?" activates general explanatory patterns from articles, textbooks, and introductory guides. A prompt like "You are a Senior Chartered Accountant writing to a GSTIN-registered supplier. Explain the impact of the 2024 IGST amendment on your invoice format using formal accounting language, referencing CGST Rules 46 and 48" activates a completely different, much more specific set of patterns. You are, in effect, navigating the model toward a specific region of its learned capabilities.

This insight — that prompts *activate* patterns rather than *program* behavior — fundamentally changes how you approach prompt design. You are not writing code that the model executes; you are writing text that guides the model to retrieve and apply the most relevant capabilities it learned during training. The craft of prompt engineering is understanding what kinds of text patterns in your prompt will cause the model to access the capabilities you need.

---

> ### 📌 SIDENOTE: Why Does the Same Prompt Sometimes Give Different Answers?
>
> *This confuses many new AI developers. Here's the deep explanation.*
>
> Recall from Module 1 that LLMs generate text by sampling from a probability distribution over their vocabulary. Even with temperature=0 (which makes the model deterministic in practice), the relationship between prompt and output is not a mathematical function in the traditional sense.
>
> Multiple factors create apparent variability:
>
> **1. System-level non-determinism:** Even with temperature=0, GPU floating-point arithmetic can produce slightly different results across different hardware configurations, batch sizes, or software versions — because floating-point operations on GPUs are not perfectly deterministic.
>
> **2. Semantic sensitivity:** LLMs are *extremely* sensitive to small changes in prompt wording. Adding a comma, changing "explain" to "describe," or reordering clauses in your instruction can shift the probability distribution enough to produce meaningfully different outputs. This is not a flaw — it reflects the model's sensitivity to the statistical patterns in natural language.
>
> **3. The training distribution:** The model's behavior on any input is fundamentally a reflection of what kinds of text followed similar inputs in the training data. Inputs that are very common in training data (like asking about basic Python syntax) produce consistent outputs. Inputs that are rare or unusual in the training data produce more variable outputs.
>
> **Practical implication:** Always test your prompts with multiple runs. A single test that produces a good result does not mean the prompt is reliable — you need to see consistent, high-quality outputs across varied inputs before trusting a prompt in production.

---

## 4.2 The Anatomy of a Perfect Prompt

A well-engineered prompt has identifiable components. Understanding each component and its purpose allows you to design prompts systematically rather than hoping that "magic words" work.

### Component 1: Role and Persona

The role assignment tells the model which region of its learned capabilities to activate. When you say "You are a Senior Advocate with 20 years of experience in ITAT proceedings," you are not lying to the model — you are providing a contextual frame that causes it to draw on the legal discourse patterns in its training data rather than, say, the conversational patterns from Reddit comments.

Role prompts work because in the training data, text written by domain experts has distinctive vocabulary, sentence structures, levels of detail, use of citations, and expression of uncertainty. By activating the "Senior Advocate" frame, you encourage the model to produce text that statistically resembles how senior advocates write — which is more likely to be accurate, professionally formatted, and appropriately detailed.

Effective role definitions include:
- **Specific domain and sub-domain:** Not "legal expert" but "ITAT tax litigation specialist"
- **Experience level and context:** "Senior partner at a Big Four CA firm"
- **Audience awareness:** "...writing to a client who is a business owner, not a lawyer"
- **Constraints:** "You only answer questions within your expertise; for others, you refer to appropriate specialists"

### Component 2: Task Description

The task description is the core of your prompt — what you want the model to do. Effective task descriptions share several characteristics:

**Specificity over generality:** "Analyze this contract" is general. "Identify all clauses that limit the service provider's liability, assess whether each limit is commercially reasonable for a Rs. 50 lakh annual contract, and flag any clauses that may be unenforceable under Indian Contract Act Section 74" is specific. The specific version activates more focused, relevant capabilities.

**Action verbs that match the output:** Different verbs activate different response patterns:
- "Summarize" → condensed overview
- "Analyze" → structured examination of components
- "Compare" → side-by-side evaluation
- "Draft" → original text creation
- "Extract" → structured data retrieval from text
- "Evaluate" → judgment with criteria
- "Classify" → categorical assignment

**Explicit scope boundaries:** Tell the model what NOT to do as much as what TO do. "Do not include information that is not in the provided document" is often as important as the positive instruction.

### Component 3: Context and Background

Context primes the model with information it needs to interpret your task correctly. The question "Is this clause enforceable?" means something different when the context is "This is a consumer contract in India" versus "This is an international commercial arbitration agreement between two multinational corporations."

Context includes: jurisdiction, the relationship between parties, the purpose of the document, relevant precedents or preferences, and any constraints on the response. The more relevant context you provide, the less the model needs to make assumptions — and every assumption the model makes is a potential source of irrelevance or error.

### Component 4: Examples (Few-Shot)

Examples are one of the most powerful tools in your prompt engineering toolkit. Providing concrete examples of the input-output relationship you want bypasses ambiguity in natural language instructions. Instead of writing a paragraph describing exactly what format you want, show the model one or two examples in that format and it will generalize the pattern.

Few-shot examples are particularly powerful for: specialized output formats, domain-specific classification tasks, tone and style calibration, and complex reasoning patterns. You will learn the mechanics of few-shot prompting in detail in Section 4.4.

### Component 5: Output Format Specification

Explicit output format instructions are often the difference between a usable response and one that requires significant post-processing. If you need the output in JSON for a downstream system, say so explicitly. If you need it in a specific structure for a legal brief, provide that structure. If you need a numbered list, specify that.

Output format specifications should include:
- **Structure:** "Respond in JSON with the following keys: ..."
- **Length:** "Your response should be exactly 3 paragraphs" or "Limit your answer to 200 words"
- **Style:** "Use bullet points for all lists. Use formal legal language."
- **Completeness:** "Include all relevant sections even if they are not applicable (mark them N/A)"
- **Citation format:** "After each claim, cite the specific section or rule you are relying on"

### Component 6: Constraints and Guardrails

Constraints prevent the model from doing things you don’t want. They are especially important in production systems where user inputs are unpredictable. Constraints include:

- **Scope limits:** "Only answer questions about Indian income tax. For other topics, respond: 'This is outside my scope.'"
- **Uncertainty expression:** "If you are not confident in an answer, explicitly say so and recommend verification."
- **Format prohibitions:** "Never use markdown headers. Respond in plain text only."
- **Safety constraints:** "Do not provide specific tax advice. Provide general educational information only."

---

## 4.3 Zero-Shot Prompting

Zero-shot prompting is the simplest form: you give the model a task with no examples, relying entirely on its pre-trained capabilities. Despite its simplicity, zero-shot prompting is remarkably effective for tasks that are well-represented in the training data.

Zero-shot works well when:
- The task is clearly expressible in natural language
- The task is common enough to be well-represented in training data
- The output format is standard (prose, JSON, lists)
- Precision and specialized formatting are not critical

Zero-shot is insufficient when:
- You need a highly specific output format unique to your domain
- The task involves specialized jargon or reasoning patterns unusual in the training data
- You need consistent, highly reliable outputs across many inputs
- The task requires multi-step reasoning chains

```python
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# Minimal zero-shot
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the penalty for late GST filing?"}
    ],
    temperature=0.0
)
print("Zero-shot:", response.choices[0].message.content)

# Enhanced zero-shot with role and constraints
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {
            "role": "system",
            "content": """You are an expert Indian Chartered Accountant and GST practitioner.
            
            When answering:
            - Always cite the specific Rule/Section under CGST Act
            - State the current penalty rates with their caps
            - Mention if the provision has been amended recently
            - End with a practical tip
            - Keep your answer under 300 words
            - If uncertain about recent changes, explicitly note this"""
        },
        {"role": "user", "content": "What is the penalty for late GST return filing under GSTR-3B?"}
    ],
    temperature=0.0
)
print("Enhanced zero-shot:", response.choices[0].message.content)
```

---

## 4.4 Few-Shot Prompting

Few-shot prompting provides the model with examples ("shots") of the input-output pairs you want, allowing it to generalize the pattern to new inputs. It is one of the most powerful and consistently effective prompting techniques because it eliminates ambiguity: instead of describing what you want in natural language (which is inherently ambiguous), you show the model exactly what you want.

The term "few-shot" refers to providing 2–10 examples. "One-shot" provides a single example. "Zero-shot" provides none. Research has consistently shown that even a single well-chosen example significantly improves performance on specialized tasks, and diminishing returns typically set in after 5–7 examples.

**Choosing your examples:** The quality of your few-shot examples matters more than their quantity. Good examples:
- Cover the range of input types you expect
- Demonstrate the exact output format you want
- Show how to handle edge cases and ambiguous inputs
- Are representative of real use cases, not toy examples

**The order of examples matters:** Research shows that models are influenced by the order of few-shot examples, tending to produce outputs that more closely resemble the last example than the first. Put your most representative example last.

```python
# Few-shot for specialized legal clause classification
few_shot_prompt = """Classify the following legal clause by type and risk level.
Respond ONLY in the format: TYPE | RISK | REASON

Example 1:
Clause: "Party A shall indemnify Party B against all losses arising from Party A's negligence."
Answer: INDEMNITY | MEDIUM | Standard mutual indemnity clause; balanced exposure.

Example 2:
Clause: "The agreement shall be automatically renewed for successive one-year terms unless
written notice is given 90 days before expiry."
Answer: AUTO-RENEWAL | HIGH | Long notice period may cause inadvertent renewal; silent rollover risk.

Example 3:
Clause: "This agreement shall be governed by the laws of England and Wales."
Answer: GOVERNING LAW | LOW | Standard governing law clause; note foreign jurisdiction implications.

Now classify this clause:
Clause: "The service provider's liability under this agreement shall not exceed Rs. 5,000
regardless of the nature or extent of the loss."
Answer:"""

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": "You are an expert Indian contract lawyer."},
        {"role": "user", "content": few_shot_prompt}
    ],
    temperature=0.0
)
print(response.choices[0].message.content)
# Expected: LIMITATION OF LIABILITY | HIGH | Cap of Rs. 5,000 is commercially
# unreasonably low; likely unenforceable under ICA Section 73/74.
```

---

## 4.5 Chain-of-Thought (CoT) Prompting

Chain-of-thought prompting is one of the most significant advances in prompt engineering, introduced in a Google research paper in 2022. The core insight is that LLMs produce significantly more accurate answers to complex reasoning tasks when they are instructed (or shown examples of being instructed) to reason step by step before arriving at a final answer.

The mechanism works because of how the model generates text autoregressively. When the model writes out its reasoning process token by token, each reasoning step becomes part of the context that influences subsequent steps. This serial reasoning process forces the model to be self-consistent — if it makes an incorrect intermediate claim, subsequent reasoning may catch and correct it. In contrast, when you ask for a direct answer, the model must "jump" to the conclusion without the corrective benefit of the reasoning chain.

Research has shown that CoT prompting dramatically improves performance specifically on: arithmetic problems, logical reasoning, multi-step planning, legal analysis, financial calculations, and any task requiring several sequential inferences. It has little effect on simple factual retrieval tasks.

**Two ways to invoke CoT:**

1. **Zero-shot CoT:** Simply append "Let's think step by step" to your question. This surprisingly simple phrase activates step-by-step reasoning without any examples.

2. **Few-shot CoT:** Provide examples where the reasoning chain is shown explicitly, teaching the model the specific type of step-by-step reasoning you want.

```python
# Problem that requires multi-step reasoning
complex_query = """A CA firm billed Rs. 18 lakhs in FY 2024-25 for services.
They have Input Tax Credit of Rs. 85,000.
GST rate on professional services is 18%.

What is the net GST payable to the government?"""

# Without CoT - may make arithmetic errors
response_no_cot = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "user", "content": f"{complex_query}\n\nAnswer directly:"}
    ],
    temperature=0.0
)

# With Zero-Shot CoT
response_with_cot = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {
            "role": "system",
            "content": "You are an expert CA. Show all calculations clearly."
        },
        {
            "role": "user",
            "content": f"{complex_query}\n\nThink through this step by step, showing each calculation:"
        }
    ],
    temperature=0.0
)

print("WITHOUT CoT:", response_no_cot.choices[0].message.content)
print("\nWITH CoT:", response_with_cot.choices[0].message.content)

# Few-Shot CoT for legal analysis
few_shot_cot = """Analyze legal issues step by step.

Example Question: Is an oral contract to sell agricultural land valid in India?

Step 1 - Identify the transaction type: This is a sale of immovable property (agricultural land).
Step 2 - Check applicable law: Section 54 of Transfer of Property Act 1882 governs sale of immovable property.
Step 3 - Check formal requirements: Section 54 requires a registered instrument for immovable property worth over Rs. 100.
Step 4 - Apply to facts: Agricultural land is almost certainly worth over Rs. 100.
Step 5 - Conclusion: An oral contract for sale of agricultural land is NOT enforceable. A registered sale deed is mandatory.

Conclusion: The oral contract is void for want of registration under Section 54 TPA.

Now analyze: Is an oral agreement to provide legal services for Rs. 5 lakhs valid in India?"""

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": few_shot_cot}],
    temperature=0.0
)
print(response.choices[0].message.content)
```

---

## 4.6 Self-Consistency Prompting

Self-consistency is a technique that generates multiple reasoning paths for the same problem and selects the most consistent answer through majority voting. It is particularly valuable for high-stakes calculations and legal determinations where a single model run might make an error that a majority of runs would not.

The intuition is: if you ask the same question 5 times with slight temperature variation and 4 out of 5 answers agree, that answer is more trustworthy than a single run. This mirrors how a CA would double-check a critical calculation by computing it twice using different methods.

```python
from collections import Counter
import re

def self_consistency_answer(
    question: str,
    n_samples: int = 5,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.6
) -> dict:
    """Generate multiple reasoning chains and return majority-vote answer."""
    
    responses = []
    final_answers = []
    
    for i in range(n_samples):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Solve step by step. Write FINAL ANSWER: [answer] on the last line."
                },
                {"role": "user", "content": question}
            ],
            temperature=temperature
        )
        
        text = response.choices[0].message.content
        responses.append(text)
        
        # Extract final answer
        lines = text.strip().split('\n')
        for line in reversed(lines):
            if 'FINAL ANSWER:' in line.upper():
                answer = line.split(':', 1)[1].strip()
                final_answers.append(answer)
                break
    
    answer_counts = Counter(final_answers)
    most_common, count = answer_counts.most_common(1)[0]
    confidence = count / n_samples
    
    return {
        "answer": most_common,
        "confidence": f"{confidence:.0%} ({count}/{n_samples} agree)",
        "all_answers": dict(answer_counts),
        "reasoning_samples": responses
    }

result = self_consistency_answer(
    """An advocate charges Rs. 8 lakhs for services.
    GST rate is 18%. He has ITC of Rs. 40,000.
    He deposits Rs. 20,000 in advance tax.
    What is the net GST payable after ITC?""",
    n_samples=5
)
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
```

---

## 4.7 The ReAct Pattern: Reason + Act

ReAct (Reasoning + Acting) is a prompting pattern that instructs the model to alternate between reasoning about what to do ("Thought:") and deciding on an action ("Action:"), then observing the result ("Observation:"). This pattern is the conceptual foundation of all AI agents (covered in depth in Module 9).

ReAct is powerful because it makes the model's planning process explicit and iterative. Rather than jumping directly to an answer, the model lays out its reasoning strategy, identifies what information it needs, and builds up its answer piece by piece. For complex, multi-step professional tasks, this structured deliberation consistently produces higher quality results than direct answering.

```python
react_template = """You are a research assistant for a CA firm.
Solve problems by thinking and planning in steps.

For each step, use this format:
- Thought: [What do I need to figure out?]
- Action: [What information do I need / what calculation should I do?]
- Observation: [What does the calculation/research tell me?]

Repeat as needed, then:
- Final Answer: [Complete, actionable answer]

---

Question: A client has received an income tax notice under Section 148A(b) for AY 2021-22, 
claiming Rs. 12 lakhs of income escaped assessment. What are the key steps to respond,
what is the legal deadline, and what documents should be prepared?"""

response = client.chat.completions.create(
    model="gpt-4.1",  # Use more capable model for complex planning
    messages=[
        {"role": "system", "content": "You are an expert Indian tax advocate."},
        {"role": "user", "content": react_template}
    ],
    temperature=0.1
)
print(response.choices[0].message.content)
```

---

## 4.8 Prompt Templates for Production Systems

In production AI tools, prompts are never static strings. They are dynamic templates that inject: user input, retrieved context (RAG), conversation history, current date/time, and domain-specific parameters. Managing these templates systematically is critical for maintainability.

A well-designed prompt template system allows non-programmers (your clients, colleagues) to modify the AI's behavior by editing template files, without touching code. It also makes prompts version-controllable, testable, and shareable.

```python
from string import Template
import yaml
from pathlib import Path

# Method 1: Python Template strings (simple)
contract_template = Template("""
ROLE: You are a $role with $experience years of experience.

JURISDICTION: $jurisdiction
APPLICABLE LAW: $applicable_law

DOCUMENT TYPE: $document_type

DOCUMENT EXCERPT:
$document_text

TASK: $task

OUTPUT FORMAT:
- Issue: What is the legal issue?
- Applicable Provision: Which section applies?
- Analysis: 2-3 paragraph analysis
- Risk Level: HIGH/MEDIUM/LOW with justification
- Recommendation: Specific, actionable advice
""")

# Fill the template
prompt = contract_template.substitute(
    role="Senior Advocate",
    experience="20",
    jurisdiction="Indore, Madhya Pradesh",
    applicable_law="Indian Contract Act 1872, CGST Act 2017",
    document_type="Service Agreement",
    document_text="...clause 12: All disputes shall be resolved by binding arbitration...",
    task="Identify enforceability issues with the dispute resolution clause"
)

# Method 2: Load prompts from YAML files (production approach)
def load_prompt_template(template_name: str, **kwargs) -> str:
    """Load a prompt template from YAML and fill with keyword arguments."""
    template_path = Path(f"prompts/{template_name}.yaml")
    with open(template_path) as f:
        template_data = yaml.safe_load(f)
    
    system = template_data['system'].format(**kwargs)
    user = template_data['user'].format(**kwargs)
    return system, user

# Example YAML file (prompts/contract_analysis.yaml):
# system: |
#   You are a {role} specializing in {domain}.
#   Jurisdiction: {jurisdiction}
# user: |
#   Analyze this {document_type}:
#   {document_text}
#   Focus on: {analysis_focus}
```

---

## 4.9 System Prompt Architecture for Professional Tools

The system prompt is the most powerful lever in an AI application. For professional-grade tools, it should be carefully engineered with these layers:

```python
# A production system prompt for a CA's AI Assistant
CA_SYSTEM_PROMPT = """
=== IDENTITY ===
You are TaxAdvisorAI, an AI assistant deployed within a Chartered Accountant's practice 
in Indore, Madhya Pradesh. You assist qualified CAs with research, analysis, and drafting.

=== SCOPE ===
You ONLY engage with topics related to:
1. Indian Income Tax Act 1961 and 2025
2. CGST/SGST/IGST Acts and Rules
3. FEMA and allied regulations (basic queries)
4. Indian Companies Act 2013 (basic queries)
5. ITAT and High Court/Supreme Court judgments on tax matters
6. Arbitration and Conciliation Act 1996 (basic queries)

For ANY query outside this scope, respond:
"I'm specialized in Indian tax and business law. Please consult the appropriate expert for [topic]."

=== OUTPUT STANDARDS ===
- Always cite the specific Section/Rule/Notification number
- Show all tax calculations with each step labeled
- Use Indian number format (lakhs, crores) for monetary amounts
- For opinions: state the basis, the analysis, and the conclusion separately
- For complex matters: always recommend professional verification of final advice

=== UNCERTAINTY PROTOCOL ===
When uncertain about recent amendments or circulars:
- Explicitly state the date of your knowledge
- Say "Please verify against the latest CBDT/CBIC circular"
- Do NOT fabricate provision numbers or case citations

=== DISCLAIMER ===
End every substantive response with:
"\u26a0️ AI-generated guidance. Verify with current CBDT/CBIC circulars before acting."
"""

# Create a specialized assistant using this system prompt
def create_specialized_assistant(system_prompt: str):
    history = [{"role": "system", "content": system_prompt}]
    
    def ask(question: str) -> str:
        history.append({"role": "user", "content": question})
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=history,
            temperature=0.1
        )
        answer = response.choices[0].message.content
        history.append({"role": "assistant", "content": answer})
        return answer
    
    return ask

ca_ai = create_specialized_assistant(CA_SYSTEM_PROMPT)
print(ca_ai("What is the current penalty under Section 271B for non-audit?"))
print(ca_ai("Can you recommend a good restaurant in Indore?"))  # Should be politely declined
```

---

## 4.10 Advanced Techniques: Prompt Chaining, Tree-of-Thought, and Meta-Prompting

### Prompt Chaining
For complex multi-stage tasks, break the work into a sequence of prompts where the output of each feeds into the next. This "pipeline" approach produces more reliable results than a single complex prompt.

```python
def legal_analysis_pipeline(document_text: str) -> dict:
    """Multi-stage pipeline: extract → classify → analyze → recommend."""
    
    # Stage 1: Extract key facts
    facts = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{
            "role": "user",
            "content": f"Extract ONLY the following facts from this document as JSON: "
                       f"{{parties, dates, monetary_amounts, key_obligations, deadline_dates}}\n\n{document_text}"
        }],
        response_format={"type": "json_object"},
        temperature=0.0
    ).choices[0].message.content
    
    # Stage 2: Classify the document and issues
    classification = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{
            "role": "user",
            "content": f"Based on these extracted facts: {facts}\n\n"
                       f"Classify: (1) document type, (2) risk level HIGH/MED/LOW, "
                       f"(3) top 3 legal issues. Respond as JSON."
        }],
        response_format={"type": "json_object"},
        temperature=0.0
    ).choices[0].message.content
    
    # Stage 3: Deep analysis using both previous stages
    analysis = client.chat.completions.create(
        model="gpt-4.1",  # More powerful model for synthesis
        messages=[{
            "role": "system",
            "content": "You are a Senior Indian legal advisor. Be thorough and cite specific provisions."
        }, {
            "role": "user",
            "content": f"Facts: {facts}\nClassification: {classification}\n\n"
                       f"Original document: {document_text}\n\n"
                       f"Provide a complete legal analysis with risk assessment and recommendations."
        }],
        temperature=0.2
    ).choices[0].message.content
    
    import json
    return {
        "extracted_facts": json.loads(facts),
        "classification": json.loads(classification),
        "analysis": analysis
    }
```

### Tree-of-Thought Prompting
Tree-of-Thought (ToT) extends CoT by generating multiple reasoning paths simultaneously and evaluating which path is most promising before continuing. It is particularly useful for problems with multiple solution strategies.

```python
tree_of_thought_prompt = """
I need to solve the following problem using multiple approaches, then select the best.

Problem: A private limited company wants to distribute Rs. 50 lakhs to its shareholders.
What is the most tax-efficient method?

--- Approach 1: Dividend Distribution ---
Think through this approach step by step...
[Analyze: corporate dividend tax, shareholder's tax liability, DDT implications]
Rating for this approach: [1-10 with reasoning]

--- Approach 2: Buy-back of Shares ---
Think through this approach step by step...
[Analyze: Section 115QA tax on buyback, capital gains implications, procedures]
Rating for this approach: [1-10 with reasoning]

--- Approach 3: Deductible Salary/Bonus ---
Think through this approach step by step...
[Analyze: deductibility, ESIC/PF implications, income tax for recipients]
Rating for this approach: [1-10 with reasoning]

--- SYNTHESIS ---
Based on all three approaches, the optimal strategy is: [final recommendation]
Because: [reasoning combining best elements]
"""
```

---

## Module 4 Summary

You have now covered the complete toolkit of prompt engineering:

| Technique | When to Use | Key Benefit |
|-----------|------------|-------------|
| Zero-shot | Simple, well-known tasks | Fast, no examples needed |
| Few-shot | Specialized format/classification | Pattern generalization |
| Chain-of-Thought | Math, multi-step reasoning | Error correction in reasoning |
| Self-Consistency | High-stakes calculations | Confidence through consensus |
| ReAct | Research, multi-tool tasks | Structured deliberation |
| Prompt Chaining | Complex, multi-stage tasks | Modular, reliable pipelines |
| Tree-of-Thought | Multiple solution paths | Optimal strategy selection |
| System Prompts | Production AI tools | Consistent, bounded behavior |

## Module 4 Exercises

1. Take a task you regularly perform (e.g., drafting a response to a client query) and design a complete system prompt with all 6 components.
2. Implement few-shot classification for document types you encounter in your practice.
3. Use self-consistency to verify a complex GST or income tax calculation.
4. Build a prompt chain for a multi-step professional analysis task.
