import os

from dotenv import load_dotenv
from typing import Literal
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from email_utils import send_market_report_email
import datetime

load_dotenv(override=True)

# Define Global State
class MarketState(TypedDict):
    messages: list
    task_type: str            # Routing determination: tender / competitor / gov / marketing / draft
    raw_data: list            # Collected raw data
    normalized_data: str      # Standardized data
    analysis_report: str      # Analysis and recommendations report
    draft_content: str        # Drafted content
    review_feedback: dict     # Human review feedback

model = ChatOllama(
    model="qwen2.5:7b",
    base_url="http://localhost:11434"
)
tavily_tool = TavilySearchResults(max_results=3)

# Define nodes

def supervisor_router_node(state: MarketState):
    """Supervisor / Router Node: Classify user intent into specific task types."""
    messages = state["messages"]

    system_prompt = SystemMessage(
        content="You are an expert AI supervisor for a market intelligence platform. "
                "Analyze the user's latest request and classify it into EXACTLY ONE of the following categories: "
                "'tender', 'competitor', 'gov', 'marketing', or 'draft'.\n"
                "- Use 'tender' if it relates to government procurement, bids, or tenders.\n"
                "- Use 'competitor' if it relates to rival companies, products, or competitors.\n"
                "- Use 'gov' if it relates to government policies, grants, or tax incentives.\n"
                "- Use 'marketing' if it relates to web analytics, traffic, or digital marketing.\n"
                "- Use 'draft' if the user wants to write or draft content.\n"
                "Return ONLY the category keyword."
    )

    response = model.invoke([system_prompt] + messages)
    content = response.content.strip().lower()

    if "tender" in content:
        task = "tender"
    elif "competitor" in content:
        task = "competitor"
    elif "gov" in content:
        task = "gov"
    elif "marketing" in content:
        task = "marketing"
    elif "draft" in content:
        task = "draft"
    else:
        task = "general"

    print(f"[Supervisor Routing] Intent classified as: [{task}]")
    return {"task_type": task}


def real_search_node(state: MarketState):
    """Search Node: Dynamic query generation and Tavily tool execution."""
    task_type = state.get("task_type", "general")

    current_year = datetime.datetime.now().year
    query_map = {
        "tender": f"Australia government disability funding braille equipment tenders {current_year}",
        "competitor": f"blind and low vision support services Australia market intelligence {current_year}",
        "gov": f"Australia government funding disability services braille literacy grants {current_year}",
        "marketing": f"accessibility awareness campaigns blind community engagement Australia {current_year}",
        "general": f"Australia braille literacy accessibility assistive technology trends {current_year}"
    }

    query = query_map.get(task_type, query_map["general"])
    print(f"Executing search node | Task: [{task_type}], Query: {query}")

    # Invoke Tavily tool
    search_results = tavily_tool.invoke({"query": query})

    # Format output
    formatted = [f"URL: {res.get('url')}\nContent: {res.get('content')}" for res in search_results]

    return {
        "raw_data": ["\n\n".join(formatted)]
    }

def ingest_and_normalize_node(state: MarketState):
    """Ingest & Normalize Node: Merge and clean raw data."""
    raws = state.get("raw_data", [])
    combined = "\n".join(raws)
    normalized = f"--- NORMALIZED MARKET INTELLIGENCE ---\n{combined}"
    return {"normalized_data": normalized}

def analyze_opportunities_node(state: MarketState):
    """Analyze opportunities and detect market trends."""
    data = state.get("normalized_data", "")
    current_year = datetime.datetime.now().year

    system_prompt = SystemMessage(
        content=(
            f"You are a senior accessibility and non-profit market research analyst specializing in the "
            f"blind and low-vision community in Australia. Based on the provided intelligence data for {current_year}, write a "
            "comprehensive market trend and opportunity analysis report focusing on **braille literacy, "
            "accessibility services, assistive technology, and community advocacy in Australia**. "
            "Formatting requirements:\n"
            "1. Output ONLY the clean report text. Do not wrap it in extra JSON format or outer brackets.\n"
            "2. Use standard Markdown headings (#, ##) and ensure every section addresses the Australian blind/low-vision sector."
        )
    )
    response = model.invoke([system_prompt, HumanMessage(content=data)])
    return {"analysis_report": response.content}

def generate_recommendations_node(state: MarketState):
    """Generate strategic business recommendations."""
    current_report = state.get("analysis_report", "")
    system_prompt = SystemMessage(
        content="Review the analysis report below and generate a set of actionable, strategic business recommendations "
            "for our executive team focusing on the Australian blind and low-vision community.\n"
            "STRICT FORMATTING RULES to prevent empty or broken lines:\n"
            "1. Each recommendation MUST follow this exact structure:\n"
            "   ### X. **[Action Title]**\n"
            "   - **Action**: [Clear description of what to do]\n"
            "   - **Expected Impact**: [Why this helps the organization]\n"
            "2. DO NOT output floating titles without an 'Action' and 'Expected Impact' bullet point underneath.\n"
            "3. Write in professional English."
    )
    response = model.invoke([system_prompt, HumanMessage(content=current_report)])
    return {"analysis_report": f"{current_report}\n\n### Actionable Recommendations\n{response.content}"}

def content_drafting_node(state: MarketState):
    """Draft content based on user instructions."""
    system_prompt = SystemMessage(
        content="You are a professional content creator for B2B tech markets. "
                "Draft high-quality, engaging business content based on the user's request in English."
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"draft_content": response.content}

def human_review_node(state: MarketState):
    """Human Review / Approval Node: Core HITL mechanism."""
    target_content = state.get("analysis_report") or state.get("draft_content") or "No content available for review."

    # Trigger interrupt for human review
    user_action = interrupt({
        "status": "pending_review",
        "content": target_content
    })
    return {"review_feedback": user_action}

def deliver_node(state: MarketState):
    """Delivery Node: Extract report and deliver it to user with E-mail."""
    final_report = (
        state.get("analysis_report")
        or state.get("draft_content")
        or "No report content generated."
    )

    receiver_email = os.getenv("DEFAULT_RECEIVER_EMAIL", os.getenv("SENDER_EMAIL"))
    subject = "Weekly Market Analysis Report"

    print(f"--- Delivering to: {receiver_email} ---")

    success = send_market_report_email(
        receiver_email=receiver_email,
        subject=subject,
        markdown_content=final_report
    )

    delivery_status = "Successfully generated, approved, and delivered via email." if success else "Generated and approved, but email delivery failed."

    return {
        "messages": [AIMessage(content=delivery_status)]
    }


# router function
def route_from_supervisor(state: MarketState) -> Literal["tender", "competitor", "gov", "marketing", "draft"]:
    task = state.get("task_type")
    if task in ["tender", "competitor", "gov", "marketing"]:
        return task
    elif task == "draft":
        return "draft"
    return "marketing"

def route_after_review(state: MarketState) -> Literal["generate_recommendations", END]:
    feedback = state.get("review_feedback", {})
    if isinstance(feedback, dict) and feedback.get("action") == "needs_changes":
        return "generate_recommendations"
    return END

# add nodes and edge
builder = StateGraph(MarketState)

# Register nodes
builder.add_node("supervisor", supervisor_router_node)
builder.add_node("real_search", real_search_node)
builder.add_node("ingest_normalize", ingest_and_normalize_node)
builder.add_node("analyze", analyze_opportunities_node)
builder.add_node("recommendations", generate_recommendations_node)
builder.add_node("content_drafting", content_drafting_node)
builder.add_node("human_review", human_review_node)
builder.add_node("deliver", deliver_node)

# Set edges
builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "tender": "real_search",
        "competitor": "real_search",
        "gov": "real_search",
        "marketing": "real_search",
        "draft": "content_drafting"
    }
)

builder.add_edge("real_search", "ingest_normalize")
builder.add_edge("ingest_normalize", "analyze")
builder.add_edge("analyze", "recommendations")
builder.add_edge("recommendations", "human_review")
builder.add_edge("content_drafting", "human_review")

builder.add_conditional_edges(
    "human_review",
    route_after_review,
    {
        "generate_recommendations": "recommendations",
        END: "deliver"
    }
)

builder.add_edge("deliver", END)

# Compile graph with checkpointer for interrupt support
DB_URI = os.getenv("DB_URI")

connection_pool = ConnectionPool(
    conninfo=DB_URI,
    max_size=20,
    kwargs={"autocommit": True, "prepare_threshold": None}
)

checkpointer = PostgresSaver(connection_pool)
checkpointer.setup()

graph = builder.compile(checkpointer=checkpointer)

print("Graph compiled successfully.")

from langgraph.types import Command

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "market_agent_interactive_1"}}

    initial_input = {
        "messages": [HumanMessage(content="Please search for the latest government policies and tax incentives for tech startups in 2026.")]
    }

    print("=" * 50)
    print("Starting workflow...")
    print("=" * 50)

    try:
        graph.invoke(initial_input, config=config)
    except Exception:
        pass

    current_state = graph.get_state(config)

    print("\n" + "=" * 50)
    print("Workflow Paused: Human Review Required")
    print("=" * 50)

    for task in current_state.tasks:
        for interrupt_obj in task.interrupts:
            payload = getattr(interrupt_obj, "value", interrupt_obj)
            if isinstance(payload, dict):
                print("\n[Content Summary]:\n", payload.get("content")[:500] + "\n... (truncated) ...")
                # Uncomment this to get full content, remember to comment above
                #print("\n[Full Content]:\n", payload.get("content"))

    # Wait for human input
    print("\n" + "-" * 40)
    user_decision = input("Approve delivery? (yes/no): ").strip().lower()
    print("-" * 40)

    if user_decision == 'yes':
        resume_payload = {"action": "approved", "comment": "Looks good!"}
        print("Status: Approved")
    else:
        resume_payload = {"action": "needs_changes", "comment": "Please revise the recommendations."}
        print("Status: Changes requested")

    final_output = graph.invoke(Command(resume=resume_payload), config=config)
    print("\nWorkflow completed successfully.")

    try:
        connection_pool.close()
    except Exception:
        pass