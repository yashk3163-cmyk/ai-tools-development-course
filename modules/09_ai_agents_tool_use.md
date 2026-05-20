# Module 9: AI Agents and Tool Use — Building AI That Can Take Action

> **Estimated Reading Time:** 5–6 hours  
> **This module is the gateway to truly autonomous AI applications.**  
> **Outcome:** Build complete AI agents that can use tools, plan multi-step tasks, handle errors, and complete real-world professional workflows autonomously.

---

## 9.1 What Is an AI Agent? The Deep Conceptual Foundation

The term "AI agent" is used loosely in popular discourse, but it has a precise technical meaning in the context of LLM-based systems. An AI agent is a system where an LLM drives the decision-making process in a loop: it reasons about a goal, decides on an action, executes that action in the world, observes the result, and then reasons again based on what it observed.

Compare this to the simpler AI systems you've built so far:

**Non-agentic pipeline (Modules 3–8):** User input → Process → LLM call → Output. The LLM is called once, at a fixed point in a predetermined workflow. You, the programmer, decide in advance exactly what the LLM will be given and exactly what will be done with its output. The LLM is a component in your pipeline, not the controller of it.

**Agentic system (this module):** User goal → [LLM decides what to do] → [Execute action] → [Observe result] → [LLM decides what to do next] → ... → Goal achieved. The LLM is now the controller. It decides which tools to call, in what order, and what to do with the results. The programmer defines the available tools and the stopping conditions, but the LLM determines the path to the goal.

This shift from LLM-as-component to LLM-as-controller is profound. It enables automation of open-ended tasks that cannot be decomposed into a predetermined sequence of steps in advance. A tax research agent might: search case law, read relevant judgments, identify applicable sections, cross-reference with current circulars, draft an opinion, check for recent amendments, and revise the opinion — all without the programmer having specified this sequence in advance.

The capabilities this enables for professional practice are transformative: agents can autonomously research complex tax questions, draft documents from templates, verify calculations across multiple standards, compile client reports, schedule follow-up tasks, and flag inconsistencies — all in response to a single high-level instruction.

---

> ### 📌 SIDENOTE: The ReAct Framework — The Cognitive Architecture of Agents
>
> *Understanding this will help you design better agents and debug them when they fail.*
>
> The dominant framework for LLM agents is **ReAct** (Reasoning + Acting), introduced in a 2022 research paper. In ReAct, the agent operates in a loop with four phases:
>
> 1. **Thought:** The agent reasons about its current state and what it needs to do next. ("I need to find the applicable GST rate for software services. I should search for SAC code 998314.")
>
> 2. **Action:** The agent selects a tool and specifies the arguments. ("search_gst_database(query='SAC code 998314 GST rate')")
>
> 3. **Observation:** The agent receives the tool result. ("SAC 998314: Information technology services, GST rate 18%.")
>
> 4. **Repeat or Conclude:** If the goal is achieved, produce the final answer. Otherwise, return to Thought.
>
> This loop continues until the agent either achieves the goal, exhausts its tool calls, or decides it cannot complete the task.
>
> **Why the loop matters:** Each observation enriches the agent's context with new information. The agent's reasoning in the next Thought step is better-informed than the previous step. This iterative refinement allows the agent to navigate complex, multi-step tasks that no single LLM call could handle.
>
> **The failure modes of the loop:** Agents can get stuck in reasoning loops (thinking endlessly without acting), can take unnecessary actions (calling tools redundantly), or can misinterpret observations (drawing wrong conclusions from tool results). Good agent design includes safeguards against these failures.

---

## 9.2 Function Calling: The Mechanism Behind Tool Use

Function calling (also called "tool use") is the technical mechanism that allows LLMs to request the execution of external functions. Instead of just generating text, the LLM can generate a structured specification of a function call: which function to call and with what arguments.

Here's how it works technically:

1. **Tool Definition:** You provide the LLM with a list of available tools, each described by: name, description (plain English), and parameter schema (JSON Schema format).

2. **LLM Decision:** When generating a response, if the LLM determines that a tool call is needed, instead of generating a text response, it generates a JSON object specifying the tool and arguments.

3. **Your Code Executes:** Your code receives this tool call specification, executes the actual function (database query, web request, calculation, etc.), and captures the result.

4. **Result Injection:** You add the tool result to the conversation history and call the LLM again, now with the tool result in context.

5. **Continuation:** The LLM reads the tool result and either calls another tool or generates a final text response.

The crucial insight is that the LLM does NOT directly execute code. It generates a structured request; your code performs the actual execution. This keeps the LLM sandboxed — you control exactly what capabilities it can access.

```python
from openai import OpenAI
from dotenv import load_dotenv
import json
import math
from datetime import datetime

load_dotenv()
client = OpenAI()

# Step 1: Define your tools (functions the agent can use)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_gst",
            "description": "Calculate GST amount and total invoice value given base amount and GST rate",
            "parameters": {
                "type": "object",
                "properties": {
                    "base_amount": {
                        "type": "number",
                        "description": "Invoice amount before GST in Indian Rupees"
                    },
                    "gst_rate": {
                        "type": "number",
                        "description": "GST rate as percentage (e.g., 18 for 18%)"
                    },
                    "transaction_type": {
                        "type": "string",
                        "enum": ["IGST", "CGST+SGST"],
                        "description": "IGST for interstate, CGST+SGST for intrastate"
                    }
                },
                "required": ["base_amount", "gst_rate", "transaction_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_gst_rate",
            "description": "Look up the applicable GST rate for a service or product by SAC/HSN code or description",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_description": {
                        "type": "string",
                        "description": "Description of the service or product"
                    }
                },
                "required": ["item_description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_filing_deadline",
            "description": "Check the next filing deadline for a GST return type",
            "parameters": {
                "type": "object",
                "properties": {
                    "return_type": {
                        "type": "string",
                        "enum": ["GSTR-1", "GSTR-3B", "GSTR-9", "GSTR-9C"],
                        "description": "Type of GST return"
                    }
                },
                "required": ["return_type"]
            }
        }
    }
]

# Step 2: Implement the actual functions
def calculate_gst(base_amount: float, gst_rate: float, transaction_type: str) -> dict:
    gst_amount = base_amount * gst_rate / 100
    result = {"base_amount": base_amount, "gst_rate": gst_rate,
               "transaction_type": transaction_type,
               "total_invoice_value": base_amount + gst_amount}
    if transaction_type == "IGST":
        result["igst_amount"] = gst_amount
    else:
        result["cgst_amount"] = gst_amount / 2
        result["sgst_amount"] = gst_amount / 2
    return result

def lookup_gst_rate(item_description: str) -> dict:
    """Simplified GST rate lookup (in production, query a real database)."""
    rates_db = {
        "legal services": {"rate": 18, "sac": "998212", "note": "Professional legal services"},
        "ca services": {"rate": 18, "sac": "998211", "note": "Accounting and audit services"},
        "software services": {"rate": 18, "sac": "998314", "note": "IT software services"},
        "food": {"rate": 5, "sac": "996331", "note": "Restaurant services"},
        "construction": {"rate": 18, "sac": "995411", "note": "General construction"},
        "medical": {"rate": 0, "sac": "999311", "note": "Hospital services (exempt)"},
    }
    desc_lower = item_description.lower()
    for key, data in rates_db.items():
        if any(word in desc_lower for word in key.split()):
            return {"item": item_description, **data}
    return {"item": item_description, "rate": "unknown",
            "note": "Rate not found in simplified database. Please check GST rate schedule."}

def check_filing_deadline(return_type: str) -> dict:
    today = datetime.now()
    deadlines = {
        "GSTR-1": f"11th of {(today.replace(day=1)).strftime('%B %Y')} (next month for previous month)",
        "GSTR-3B": f"20th of {(today.replace(day=1)).strftime('%B %Y')} (next month for previous month)",
        "GSTR-9": "31st December (annual return)",
        "GSTR-9C": "31st December (audit return)"
    }
    return {"return_type": return_type, "deadline": deadlines.get(return_type, "Unknown"),
            "current_date": today.strftime("%d %B %Y")}

# Map function names to actual functions
FUNCTION_MAP = {
    "calculate_gst": calculate_gst,
    "lookup_gst_rate": lookup_gst_rate,
    "check_filing_deadline": check_filing_deadline,
}

# Step 3: The Agent Loop
def run_agent(user_query: str, max_iterations: int = 10) -> str:
    """Run the agent loop until completion or max iterations."""
    
    messages = [
        {"role": "system", "content": """You are an expert GST assistant for an Indian CA firm.
        Use the available tools to answer questions accurately.
        Always calculate exact amounts when asked, and look up rates before calculating.
        Show your work clearly."""},
        {"role": "user", "content": user_query}
    ]
    
    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        
        choice = response.choices[0]
        
        # If model gives a final text response (no tool calls)
        if choice.finish_reason == "stop":
            return choice.message.content
        
        # If model wants to call tools
        if choice.finish_reason == "tool_calls":
            # Add assistant's tool call request to history
            messages.append(choice.message)
            
            # Execute each requested tool call
            for tool_call in choice.message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                print(f"   [Tool] {func_name}({func_args})")
                
                if func_name in FUNCTION_MAP:
                    result = FUNCTION_MAP[func_name](**func_args)
                else:
                    result = {"error": f"Unknown function: {func_name}"}
                
                # Add tool result to conversation history
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
    
    return "Agent reached maximum iterations without completing the task."

# Test the agent
if __name__ == "__main__":
    queries = [
        "My client is a CA firm in Indore. They billed Rs. 2,50,000 for audit services to a client in Mumbai. What is the GST amount and the total invoice value?",
        "What are the deadlines for GSTR-1 and GSTR-3B this month?",
        "I need to bill Rs. 5 lakhs for software development to a client in the same state. What GST applies and what is the final invoice?",
    ]
    
    for q in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {q}")
        print(f"{'='*60}")
        result = run_agent(q)
        print(f"\nANSWER:\n{result}")
```

---

## 9.3 Building a Robust Agent with Error Handling

```python
from pydantic import BaseModel
from typing import Optional, Callable, Any
import traceback
from rich.console import Console
from rich.panel import Panel

console = Console()

class Tool(BaseModel):
    """A tool that an agent can use."""
    name: str
    description: str
    func: Callable
    parameters_schema: dict
    
    class Config:
        arbitrary_types_allowed = True
    
    def to_openai_format(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema
            }
        }
    
    def execute(self, **kwargs) -> str:
        try:
            result = self.func(**kwargs)
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


class LLMAgent:
    """
    A robust, production-grade LLM Agent with:
    - Tool use
    - Conversation history
    - Error handling and retry logic
    - Verbose logging
    - Max iteration safeguards
    """
    
    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: list[Tool],
        model: str = "gpt-4.1-mini",
        max_iterations: int = 15,
        verbose: bool = True
    ):
        self.name = name
        self.model = model
        self.tools = {t.name: t for t in tools}
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.messages = [{"role": "system", "content": system_prompt}]
        self.iteration_count = 0
    
    def _log(self, message: str, style: str = "cyan"):
        if self.verbose:
            console.print(f"[{style}][{self.name}][/] {message}")
    
    def run(self, user_query: str) -> str:
        self.messages.append({"role": "user", "content": user_query})
        self._log(f"Starting task: {user_query[:100]}...", "bold blue")
        
        for i in range(self.max_iterations):
            self.iteration_count += 1
            self._log(f"Iteration {i+1}/{self.max_iterations}")
            
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=[t.to_openai_format() for t in self.tools.values()],
                    tool_choice="auto"
                )
            except Exception as e:
                self._log(f"API error: {e}", "red")
                return f"Error communicating with AI: {str(e)}"
            
            choice = response.choices[0]
            
            if choice.finish_reason == "stop":
                answer = choice.message.content
                self._log(f"Task complete after {i+1} iterations", "bold green")
                return answer
            
            if choice.finish_reason == "tool_calls":
                self.messages.append(choice.message)
                
                for tool_call in choice.message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    self._log(f"Calling tool: {tool_name}({tool_args})", "yellow")
                    
                    if tool_name in self.tools:
                        result = self.tools[tool_name].execute(**tool_args)
                    else:
                        result = json.dumps({"error": f"Tool '{tool_name}' not found."})
                    
                    self._log(f"Tool result: {result[:200]}", "dim")
                    
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
        
        return "Maximum iterations reached. Task may be incomplete."
```

---

## 9.4 A Complete Legal Research Agent

```python
import requests

# Legal research tools
def search_income_tax_provisions(query: str, section: str = None) -> dict:
    """Simulate search of Income Tax Act provisions."""
    # In production: query a real legal database API
    provisions = {
        "148": {"title": "Issue of notice where income has escaped assessment",
                 "key_points": ["Time limit: 3 years for < Rs 50L, 10 years for more",
                                  "Mandatory approval required from PCIT/CIT",
                                  "Section 148A inquiry must precede notice"]},
        "271B": {"title": "Failure to get accounts audited",
                  "key_points": ["Penalty: 0.5% of turnover, maximum Rs 1.5 lakh",
                                   "Penalty not leviable if reasonable cause shown"]},
        "44AB": {"title": "Audit of accounts of certain persons",
                  "key_points": ["Business: Rs 1 crore threshold (Rs 10 crore if cash < 5%)",
                                   "Profession: Rs 50 lakhs threshold",
                                   "Report: Form 3CD due 30 September"]},
    }
    sec_num = section or query.split()[-1] if query else None
    if sec_num and sec_num in provisions:
        return {"section": sec_num, **provisions[sec_num]}
    return {"result": f"Provisions relevant to: {query}", 
            "note": "Full database search required for complete results"}

def calculate_income_tax(taxable_income: float, taxpayer_type: str = "individual") -> dict:
    """Calculate income tax for AY 2025-26 new regime."""
    slabs = [
        (300000, 0, "Up to 3L"),
        (600000, 0.05, "3L-6L"),
        (900000, 0.10, "6L-9L"),
        (1200000, 0.15, "9L-12L"),
        (1500000, 0.20, "12L-15L"),
        (float('inf'), 0.30, "Above 15L")
    ]
    
    tax = 0
    prev_limit = 0
    breakdown = []
    
    for limit, rate, label in slabs:
        if taxable_income <= prev_limit:
            break
        taxable_in_slab = min(taxable_income, limit) - prev_limit
        tax_in_slab = taxable_in_slab * rate
        if tax_in_slab > 0:
            breakdown.append({"slab": label, "amount": taxable_in_slab, 
                               "rate": f"{rate*100:.0f}%", "tax": tax_in_slab})
        tax += tax_in_slab
        prev_limit = limit
    
    cess = tax * 0.04
    return {
        "taxable_income": taxable_income,
        "base_tax": tax,
        "health_education_cess": cess,
        "total_tax": tax + cess,
        "effective_rate": f"{(tax+cess)/taxable_income*100:.2f}%" if taxable_income > 0 else "0%",
        "breakdown": breakdown,
        "regime": "New Tax Regime (AY 2025-26)"
    }

# Build the legal research agent
legal_tools = [
    Tool(
        name="search_income_tax_provisions",
        description="Search Income Tax Act provisions by section number or topic",
        func=search_income_tax_provisions,
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query or topic"},
                "section": {"type": "string", "description": "Specific section number (optional)"}
            },
            "required": ["query"]
        }
    ),
    Tool(
        name="calculate_income_tax",
        description="Calculate income tax liability for AY 2025-26 under new regime",
        func=calculate_income_tax,
        parameters_schema={
            "type": "object",
            "properties": {
                "taxable_income": {"type": "number", "description": "Total taxable income in Rs."},
                "taxpayer_type": {"type": "string", "enum": ["individual", "huf", "firm", "company"]}
            },
            "required": ["taxable_income"]
        }
    )
]

tax_agent = LLMAgent(
    name="TaxResearchAgent",
    system_prompt="""You are an expert Indian tax research AI assistant for a CA practice.
    Use tools to look up provisions and calculate taxes precisely.
    Always show step-by-step reasoning and cite specific sections.""",
    tools=legal_tools,
    model="gpt-4.1",
    verbose=True
)

# Run a complex multi-tool query
result = tax_agent.run(
    """My client has taxable income of Rs. 18 lakhs for AY 2025-26.
    He also failed to get his accounts audited under Section 44AB.
    Please: (1) Calculate his tax liability, (2) Tell me about the Section 44AB requirements,
    (3) Calculate the penalty under Section 271B, and give me a complete summary."""
)
print(result)
```

---

## Module 9 Summary

You have built:
- A fundamental understanding of what makes a system "agentic" vs. "pipeline-based"
- A complete function-calling implementation
- A robust, production-grade `LLMAgent` class with error handling and logging
- A domain-specific legal and tax research agent

Agents become truly powerful in Module 13 (Domain Tools) where you'll combine RAG + agents + conversation memory into a complete professional tool.

## Module 9 Exercises

1. Add a `search_gst_rate(description)` tool to the tax agent
2. Add a `get_itat_judgments(topic)` tool that searches a knowledge base
3. Implement an agent that can analyze a client's complete tax situation (income + deductions + penalties) in one query
4. Add conversation memory to the `LLMAgent` class so it remembers the client profile across multiple queries
