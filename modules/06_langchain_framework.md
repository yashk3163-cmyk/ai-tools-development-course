# Module 6: The LangChain Framework — Building AI Applications Faster

> **Estimated Reading Time:** 4–5 hours  
> **Outcome:** Use LangChain's LCEL (LangChain Expression Language) to build composable, production-ready AI pipelines faster than with raw API calls.

---

## 6.1 Why LangChain Exists

As you built the components in Modules 3–5, you noticed repetitive patterns: loading prompts, calling the LLM, parsing the output, passing results to the next step. LangChain emerged as a framework to standardize these patterns into reusable, composable components.

The core abstraction in modern LangChain is the **chain** — a composable, executable unit that takes an input, performs operations (often involving an LLM), and produces an output. Chains are built using LCEL (LangChain Expression Language), which uses the pipe operator `|` to compose components in a readable, left-to-right data flow.

```
prompt | llm | output_parser
```

This reads: "pass input to the prompt template, then to the LLM, then through the parser."

---

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import json

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
llm_precise = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)

# ===== Basic LCEL Chain =====
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Indian CA. Be concise and cite specific sections."),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"question": "What is Section 44AB?"})
print(result)

# ===== JSON Output Chain =====
json_prompt = ChatPromptTemplate.from_template("""
Extract tax information from this text as JSON with keys:
taxpayer_name, turnover_amount, tax_type, applicable_section

Text: {text}

Respond with valid JSON only.
""")

json_chain = json_prompt | llm_precise | JsonOutputParser()
result = json_chain.invoke({
    "text": "Mr. Agarwal runs a trading firm with Rs 1.2 crore turnover under Section 44AB."
})
print(result)

# ===== Document Analysis Chain =====
analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a legal document analyst for an Indian CA firm.
    Provide structured analysis.
    Format: Issue | Section | Risk | Recommendation"""),
    ("human", "Analyze this document clause:\n\n{clause}")
])

analysis_chain = analysis_prompt | llm | StrOutputParser()
clause = "The penalty for delayed payment shall be 24% per annum compounded monthly."
print(analysis_chain.invoke({"clause": clause}))

# ===== Sequential Pipeline (Multi-Step) =====
# Step 1: Extract key issues
extract_prompt = ChatPromptTemplate.from_template(
    "List the top 3 legal issues in this document as JSON array of strings:\n{document}"
)

# Step 2: Analyze each issue
analyze_prompt = ChatPromptTemplate.from_template(
    """For these legal issues: {issues}
    Provide a risk assessment for an Indian business context. Be specific about applicable laws."""
)

pipeline = (
    {"issues": extract_prompt | llm | StrOutputParser(),
     "document": RunnablePassthrough()}
    | analyze_prompt | llm | StrOutputParser()
)

doc = "This agreement auto-renews. Disputes go to London arbitration. Liability capped at Rs 1000."
result = pipeline.invoke({"document": doc})
print(result)

# ===== Conversational Chain with Memory =====
conv_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a CA assistant. Remember context across the conversation."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

memory = ConversationBufferWindowMemory(return_messages=True, k=5)

conv_chain = conv_prompt | llm | StrOutputParser()

def chat(user_input: str) -> str:
    hist = memory.load_memory_variables({})
    response = conv_chain.invoke({
        "input": user_input,
        "history": hist.get("history", [])
    })
    memory.save_context({"input": user_input}, {"output": response})
    return response

print(chat("My client has Rs 80 lakh consulting income."))
print(chat("Does he need a tax audit?"))  # Remembers context
print(chat("What is the deadline for the audit report?"))

# ===== Document Loader + Analysis Chain =====
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

def build_langchain_rag(pdf_path: str) -> RetrievalQA:
    """Build a LangChain RAG pipeline from a PDF."""
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(
        chunks, OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory="./lc_rag"
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_precise,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type="stuff",
        return_source_documents=True
    )
    return qa_chain
```

## Module 6 Summary
- LCEL `|` pipe operator composes LangChain components cleanly
- ChatPromptTemplate manages prompt formatting with variables
- Output parsers convert LLM text to structured Python objects
- LangChain's memory classes handle conversation context
- Built-in RAG chains simplify retrieval-augmented generation
