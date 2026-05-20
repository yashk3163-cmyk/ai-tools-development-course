# 🧠 Mastering AI Tools & Development
### *A Complete Practical Course: From Python Developer to AI Engineer*

> **Author:** AI Course — Expert-Level Curriculum  
> **Prerequisites:** Basic Python only (variables, functions, loops, pip)  
> **Duration:** ~150–180 hours of study and hands-on practice  
> **Outcome:** Build production-grade AI tools across any domain — legal, finance, medical, education, automation

---

## 📚 Course Structure

| # | Module | Core Topics | Difficulty |
|---|--------|------------|------------|
| 01 | [Understanding AI & LLMs](modules/01_understanding_ai_llms.md) | What is AI, Neural Networks, Transformers, Tokenization, The Ecosystem | 🟢 Beginner |
| 02 | [Dev Environment Setup](modules/02_environment_setup.md) | Virtual Envs, API Keys, First API Call, Cost Management | 🟢 Beginner |
| 03 | [OpenAI API in Depth](modules/03_openai_api_deep_dive.md) | Chat Completions, JSON Mode, Streaming, Token Counting | 🟡 Intermediate |
| 04 | [Prompt Engineering](modules/04_prompt_engineering.md) | Zero-Shot, Few-Shot, CoT, ReAct, Self-Consistency, Templates | 🟡 Intermediate |
| 05 | [Conversational Memory](modules/05_conversational_memory.md) | Buffer, Window, Summary, Entity, Persistent Memory | 🟡 Intermediate |
| 06 | [LangChain Framework](modules/06_langchain_framework.md) | LCEL, Chains, Output Parsers, Document Loaders | 🟡 Intermediate |
| 07 | [Embeddings & Vector Databases](modules/07_embeddings_vector_databases.md) | What are Embeddings, ChromaDB, FAISS, Pinecone | 🔴 Advanced |
| 08 | [Retrieval-Augmented Generation](modules/08_rag_systems.md) | RAG Architecture, Chunking, Hybrid Search, Re-ranking | 🔴 Advanced |
| 09 | [AI Agents & Tool Use](modules/09_ai_agents_tool_use.md) | Function Calling, Agent Loop, OpenAI Agents SDK, LangGraph | 🔴 Advanced |
| 10 | [Anthropic Claude](modules/10_anthropic_claude.md) | Claude API, Long Context, Hybrid Workflows | 🟡 Intermediate |
| 11 | [Local LLMs with Ollama](modules/11_local_llms_ollama.md) | Ollama Setup, Private RAG, Local Embeddings | 🟡 Intermediate |
| 12 | [Fine-Tuning LLMs](modules/12_fine_tuning_llms.md) | When to Fine-tune, SFT, DPO, Data Preparation | 🔴 Advanced |
| 13 | [Domain-Specific AI Tools](modules/13_domain_specific_tools.md) | Production Architecture, Caching, Retry Logic, Cost Control | 🔴 Advanced |
| 14 | [Multi-Modal AI](modules/14_multimodal_ai.md) | Vision (GPT-4o), Whisper Audio, PDF Processing | 🔴 Advanced |
| 15 | [Capstone Projects](modules/15_capstone_projects.md) | Full Legal AI Suite, Research Assistant, Automation Agents | 🔴 Advanced |

---

## 🚀 Quick Start

```bash
# 1. Clone this repository
git clone https://github.com/yashk3163-cmyk/ai-tools-development-course.git
cd ai-tools-development-course

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Set up your API keys
cp .env.example .env
# Edit .env and add your API keys

# 5. Start with Module 1
cd modules
# Open 01_understanding_ai_llms.md and follow along
```

---

## 🗂️ Repository Structure

```
ai-tools-development-course/
├── README.md                    # This file — course overview
├── requirements.txt             # All Python dependencies
├── .env.example                 # Template for API keys
├── modules/                     # Course modules (theory + code)
│   ├── 01_understanding_ai_llms.md
│   ├── 02_environment_setup.md
│   ├── 03_openai_api_deep_dive.md
│   ├── 04_prompt_engineering.md
│   ├── 05_conversational_memory.md
│   ├── 06_langchain_framework.md
│   ├── 07_embeddings_vector_databases.md
│   ├── 08_rag_systems.md
│   ├── 09_ai_agents_tool_use.md
│   ├── 10_anthropic_claude.md
│   ├── 11_local_llms_ollama.md
│   ├── 12_fine_tuning_llms.md
│   ├── 13_domain_specific_tools.md
│   ├── 14_multimodal_ai.md
│   └── 15_capstone_projects.md
├── code/                        # Standalone runnable Python files
│   ├── module_03_openai_basics.py
│   ├── module_04_prompting.py
│   ├── module_05_memory.py
│   ├── module_06_langchain.py
│   ├── module_07_embeddings.py
│   ├── module_08_rag.py
│   ├── module_09_agents.py
│   ├── module_10_claude.py
│   ├── module_11_ollama.py
│   ├── module_12_finetuning.py
│   ├── module_13_production_tool.py
│   ├── module_14_multimodal.py
│   └── module_15_capstone.py
├── projects/                    # Capstone project templates
│   ├── legal_ai_suite/
│   └── research_assistant/
└── appendix/                    # Reference material
    ├── model_selection_guide.md
    ├── prompt_patterns_library.md
    └── python_libraries_reference.md
```

---

## 🔑 API Keys You'll Need

| Service | Get Key From | Free Tier? | Used In |
|---------|-------------|-----------|--------|
| OpenAI | [platform.openai.com](https://platform.openai.com) | $5 credit | Modules 2–9, 12–14 |
| Anthropic | [console.anthropic.com](https://console.anthropic.com) | $5 credit | Module 10 |
| LangSmith | [smith.langchain.com](https://smith.langchain.com) | Yes | Module 6 |
| Ollama | Local install | Free | Module 11 |

---

## 💡 Learning Philosophy

This course follows a **"Concept First, Code Second"** philosophy:

1. **Read the theory** (7–10 pages per module) — understand *why* before *how*
2. **Study the code** — read line by line, don't just run it
3. **Modify and experiment** — change parameters, break things, fix them
4. **Build your own** — each module has an exercise to build a domain-specific tool
5. **Connect concepts** — each module references underlying concepts with detailed sidenotes

---

## 📋 Prerequisites Check

Before starting, ensure you are comfortable with:
- [ ] Python functions, classes, and modules
- [ ] `pip install` and virtual environments
- [ ] Reading and writing files in Python
- [ ] Basic `dict`, `list`, `str` operations
- [ ] `import` statements and using third-party libraries

You do **NOT** need: machine learning math, statistics, linear algebra, deep learning

---

*Built with ❤️ for Indian legal and financial professionals learning AI development*
