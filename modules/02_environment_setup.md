# Module 2: Setting Up Your AI Development Environment

> **Estimated Time:** 2–3 hours (theory + hands-on setup)  
> **Outcome:** A fully configured, professional-grade AI development environment with working API connections

---

## 2.1 Why Environment Setup Is Not Trivial for AI Development

If you have written Python before, you have probably installed packages with `pip install` and moved on. For AI development, this approach quickly becomes problematic. The ecosystem of AI libraries is evolving at an extraordinary pace — LangChain releases updates multiple times per week, OpenAI's SDK changes its API surface regularly, and incompatibilities between package versions are common.

Without disciplined environment management, you will encounter situations where updating one project's dependencies breaks another, where a tutorial that worked last month now fails because a library changed its interface, and where you spend more time fighting dependency conflicts than building AI applications. The investment of five minutes setting up a proper virtual environment saves hours of debugging later.

Beyond dependencies, AI development introduces a critical security consideration that standard programming tutorials rarely emphasize: **API keys**. These keys provide direct access to services billed to your account. A single exposed key in a public GitHub repository can result in thousands of dollars of unauthorized charges within hours — there are automated bots that scan GitHub commits for API keys and exploit them immediately. The security practices in this module are not optional.

---

> ### 📌 SIDENOTE: What Is a Virtual Environment and Why Does It Matter?
>
> *Essential reading if you've only used pip install globally before.*
>
> When you install Python, you get a single global Python installation. When you run `pip install langchain`, it installs LangChain into that global installation, accessible to all Python projects on your system.
>
> **The problem:** Project A needs `langchain==0.1.0` and Project B needs `langchain==0.3.0`. These are incompatible. You cannot have both installed in the same global Python.
>
> **The solution — Virtual Environments:** A virtual environment is an isolated copy of Python with its own set of installed packages. Each project gets its own environment:
> ```
> project_a/
>   venv/              <- isolated Python with langchain 0.1.0
>   main.py
>
> project_b/
>   venv/              <- isolated Python with langchain 0.3.0
>   main.py
> ```
> When you "activate" an environment, your terminal's `python` and `pip` commands point to that isolated installation. Packages installed go only into that environment.
>
> **Think of it like:** Each project is a separate office with its own set of tools. You don't mix tools between offices.

---

## 2.2 Complete Environment Setup

```bash
# Step 1: Verify Python version (need 3.10+)
python --version
# Should show Python 3.10.x or higher

# Step 2: Create your course virtual environment
python -m venv ai_course_env

# Step 3: Activate it
# On Windows:
ai_course_env\Scripts\activate
# On macOS/Linux:
source ai_course_env/bin/activate

# Your terminal prompt now shows: (ai_course_env)

# Step 4: Upgrade pip first
pip install --upgrade pip

# Step 5: Install all course dependencies
pip install openai anthropic
pip install langchain langchain-openai langchain-community langchain-core
pip install chromadb faiss-cpu sentence-transformers
pip install python-dotenv tiktoken pydantic
pip install requests httpx rich
pip install jupyter notebook

# Step 6: Save your environment for reproducibility
pip freeze > requirements.txt

# Step 7: Verify key installations
python -c "import openai; print('OpenAI:', openai.__version__)"
python -c "import langchain; print('LangChain:', langchain.__version__)"
python -c "import chromadb; print('ChromaDB:', chromadb.__version__)"
```

## 2.3 API Key Management — The Secure Way

```bash
# Create .env file (NEVER commit this to git)
touch .env

# Create .gitignore to protect it
echo ".env" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
```

```ini
# .env file contents:
OPENAI_API_KEY=sk-proj-your-actual-key-here
ANTHROPIC_API_KEY=sk-ant-your-actual-key-here
LANGCHAIN_API_KEY=your-langsmith-key-here
```

```python
# test_setup.py - Run this to verify everything works
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Say: Setup successful!"}],
    max_tokens=20
)

print(response.choices[0].message.content)
print(f"Tokens used: {response.usage.total_tokens}")
print("\n Environment setup is COMPLETE!")
```

## 2.4 Understanding API Costs

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Best For |
|-------|----------------------|------------------------|----------|
| gpt-4.1-mini | $0.40 | $1.60 | Most tasks, cost-sensitive apps |
| gpt-4.1 | $2.00 | $8.00 | Complex reasoning, critical tasks |
| gpt-4o | $2.50 | $10.00 | Multimodal (images + text) |
| claude-3-5-sonnet | $3.00 | $15.00 | Long documents, nuanced writing |
| text-embedding-3-small | $0.02 | N/A | RAG embeddings |

A 10-page document ≈ 5,000 tokens. Analyzing it with gpt-4.1-mini costs approximately $0.002–0.004.

---

## Module 2 Exercises

1. Set up your virtual environment and install all packages
2. Create your `.env` file with your OpenAI key
3. Run `test_setup.py` and confirm "Setup successful"
4. Check your OpenAI usage dashboard after running the test
