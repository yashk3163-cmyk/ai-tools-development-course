# Module 12: Fine-Tuning LLMs — Teaching AI Your Way

> **Estimated Reading Time:** 3–4 hours  
> **Outcome:** Understand when and how to fine-tune, prepare training data, run a fine-tuning job, and evaluate the result.

---

## 12.1 When Fine-Tuning Is and Is NOT the Answer

Fine-tuning is one of the most misunderstood concepts in practical AI development. Many practitioners attempt to fine-tune when RAG would work better, and waste significant time and money in the process.

**Fine-tuning changes the model's weights** to make certain behaviors, styles, or response patterns more likely. It is NOT a knowledge-insertion mechanism. If you want the model to know about a specific case file, RAG is correct. If you want the model to consistently respond in a specific format, style, or tone, fine-tuning is appropriate.

**Correct use cases for fine-tuning:**
- Teach the model a very specific response format (e.g., always produce a specific legal brief structure)
- Adapt the model's tone and style to your firm's communication standards
- Teach specialized reasoning patterns in your domain
- Reduce prompting overhead when the same complex instructions would be needed in every call
- Improve performance on a narrow, well-defined task where you have hundreds of examples

**Incorrect use cases (use RAG instead):**
- Teaching factual knowledge (case details, client information, recent judgments)
- Knowledge that changes frequently
- When you have fewer than 50 high-quality examples

---

```python
import json
from openai import OpenAI
from dotenv import load_dotenv
import time

load_dotenv()
client = OpenAI()

# ===== STEP 1: Prepare Training Data =====
# Format: list of {messages: [{role, content}, ...]} objects
# Each example shows the model how to respond to a specific type of input

training_examples = [
    {
        "messages": [
            {"role": "system", "content": "You are TaxAdvisorAI for Sharma & Associates CA Firm, Indore."},
            {"role": "user", "content": "My turnover is Rs 1.5 crore. Do I need audit?"},
            {"role": "assistant", "content": "ASSESSMENT: Tax Audit Required\n\nAPPLICABLE SECTION: Section 44AB, Income Tax Act 1961\n\nANALYSIS: Your business turnover of Rs 1.5 crore exceeds the Rs 1 crore threshold under Section 44AB(a). Tax audit is mandatory.\n\nEXCEPTION: If cash receipts and payments are each below 5% of total receipts/payments, the threshold increases to Rs 10 crore under proviso to Section 44AB.\n\nACTION REQUIRED: Engage a Chartered Accountant for audit. Form 3CD due by 30 September.\n\n⚠️ Verify with current CBDT guidelines before acting."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are TaxAdvisorAI for Sharma & Associates CA Firm, Indore."},
            {"role": "user", "content": "What is the late fee for GSTR-3B?"},
            {"role": "assistant", "content": "ASSESSMENT: Late Filing Penalty\n\nAPPLICABLE SECTION: Section 47, CGST Act 2017\n\nANALYSIS: Late fee applies per day of delay:\n- Rs 50/day (Rs 25 CGST + Rs 25 SGST) for returns with tax liability\n- Rs 20/day (Rs 10 CGST + Rs 10 SGST) for nil returns\n\nMAXIMUM CAP: Late fee capped at Rs 10,000 (Rs 5,000 CGST + Rs 5,000 SGST)\n\nACTION REQUIRED: File return immediately to minimize penalty.\n\n⚠️ Verify with current CBIC circulars before acting."}
        ]
    },
]

# Save training data as JSONL
def save_training_data(examples: list, filename: str):
    with open(filename, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
    print(f"Saved {len(examples)} training examples to {filename}")

save_training_data(training_examples, "training_data.jsonl")

# ===== STEP 2: Validate Training Data =====
def validate_training_data(filename: str):
    errors = []
    with open(filename) as f:
        for i, line in enumerate(f):
            try:
                example = json.loads(line)
                messages = example.get('messages', [])
                roles = [m['role'] for m in messages]
                if 'system' not in roles: errors.append(f"Line {i+1}: Missing system message")
                if 'assistant' not in roles: errors.append(f"Line {i+1}: Missing assistant message")
                if 'user' not in roles: errors.append(f"Line {i+1}: Missing user message")
            except json.JSONDecodeError as e:
                errors.append(f"Line {i+1}: Invalid JSON: {e}")
    
    if errors:
        print("VALIDATION ERRORS:"); [print(f"  - {e}") for e in errors]
    else:
        print("Validation passed! Training data is valid.")
    return len(errors) == 0

validate_training_data("training_data.jsonl")

# ===== STEP 3: Upload and Start Fine-Tuning =====
def run_fine_tuning_job(training_file: str, model: str = "gpt-4.1-mini-2025-04-14",
                        n_epochs: int = 3) -> str:
    """Upload training file and create fine-tuning job."""
    print("Uploading training file...")
    with open(training_file, 'rb') as f:
        file_response = client.files.create(file=f, purpose='fine-tune')
    file_id = file_response.id
    print(f"File uploaded: {file_id}")
    
    print("Starting fine-tuning job...")
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=model,
        hyperparameters={"n_epochs": n_epochs}
    )
    print(f"Fine-tuning job created: {job.id}")
    return job.id

def monitor_fine_tuning(job_id: str):
    """Monitor fine-tuning progress."""
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"Status: {job.status} | Model: {job.fine_tuned_model or 'In training...'}") 
        if job.status in ['succeeded', 'failed', 'cancelled']:
            return job
        time.sleep(60)

# ===== STEP 4: Use Your Fine-Tuned Model =====
def use_fine_tuned_model(model_id: str, query: str) -> str:
    """Call your custom fine-tuned model."""
    response = client.chat.completions.create(
        model=model_id,  # e.g., "ft:gpt-4.1-mini-2025-04-14:sharma-ca-firm::AbCd1234"
        messages=[
            {"role": "system", "content": "You are TaxAdvisorAI for Sharma & Associates CA Firm, Indore."},
            {"role": "user", "content": query}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content
```
