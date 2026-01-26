from typing import Annotated, Literal, Optional, List, Dict, Any
from typing_extensions import TypedDict
import operator

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from config.settings import get_model_config
from core.schemas import Context
from OpenMCS_Agent.tools.basic_tools import upload_sdk_doc, inspect_artifacts, generate_plugin_stub
from OpenMCS_Agent.tools.code_tools import create_file, execute_python_file
from tools.memory_tool import save_memory, read_memory, list_memories
from tools.rag_tool import (
    search_knowledge_base,
    add_to_knowledge_base,
    update_knowledge_base_from_files,
    rag_answer,
    create_temp_knowledge_base,
    search_temp_knowledge_base,
    search_web
)

from core.context_manager import set_active_context


SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor of the OpenMCS Agent team.
Your goal is to route the user's request to the most appropriate specialist worker.

The available workers are:
1. **Developer**: Specialized in writing OpenMCS extensions, coding plugins, and debugging.
2. **Support**: Specialized in explaining software usage, reading manuals, and providing instructions.
3. **Scientist**: Specialized in answering experimental scientific questions and general reasoning.

Rules:
- Analyze the user request and conversation history.
- **CRITICAL**: If the previous message was from a Worker (marked with [Developer], [Support], etc.) and it appears to answer the user's question, you **MUST** output 'FINISH'.
- Do NOT route back to the same worker immediately unless they explicitly request it or failed.
- If the conversation has reached a natural conclusion, output 'FINISH'.
- If the user asks to write code, debug, or create a plugin, route to 'Developer'.
- If the user asks how to use the software, where a button is, or for documentation, route to 'Support'.
- If the user asks about scientific principles, experiment design, or analysis, route to 'Scientist'.
- If you are unsure, default to 'FINISH' (let the user ask again).

IMPORTANT:
- Do NOT utilize any tools.
- Do NOT answer the user's request directly.
- ONLY output the JSON decision.
- Your output must be valid JSON with a single key 'next'.

Output your decision as a JSON object with a single key 'next' mapping to the worker name (Developer, Support, Scientist) or 'FINISH'.
"""

DEVELOPER_PROMPT = """You are the OpenMCS Extension Developer.
Your primary task is to implement new device plugins and maintain the codebase.
You have access to tools for file operations and code generation.
Always access the framework documentation and artifacts provided.

When writing code:
- Follow OpenMCS plugin architecture.
- Use Python 3 and standard typing.
- Ensure device safety.
"""

SUPPORT_PROMPT = """You are the OpenMCS Support Specialist.
Your task is to help users navigate and use the OpenMCS software.
You have access to the RAG system to look up manuals (HTML/PDF).
Explain clearly and referencing specific sections of the manuals if possible.
"""

SCIENTIST_PROMPT = """You are the Scientific Consultant.
Your task is to answer scientific questions related to microscopy and experiments.
You can use the knowledge base to find relevant papers or notes.
Reason step-by-step about physics and experimental setups.
"""

# --- State Definition ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    # Context removed to avoid serialization errors (msgpack)
    next: str

# --- Graph Construction ---

def get_chat_model_instance(config_name=None):
    cfg = get_model_config(config_name)
    kwargs = {
        "model": cfg["model_id"],
        "temperature": 0,
        "max_tokens": None,
        "timeout": None,
        "max_retries": 2,
        "api_key": cfg["api_key"],
        "model_provider": cfg["provider"]
    }
    if cfg.get("base_url"):
        kwargs["base_url"] = cfg["base_url"]
    return init_chat_model(**kwargs)

def build_multi_agent_graph(config_name=None):
    llm = get_chat_model_instance(config_name)

    # 1. Supervisor Agent
    
    def supervisor_node(state: AgentState):
        messages = state['messages']
        
        # Convert messages to a text summary for the Supervisor
        # this avoids 400 errors with structure-sensitive LLMs (invalid tool call history)
        # and lets the Supervisor focus on the high-level flow.
        conversation_text = ""
        for m in messages:
            if isinstance(m, HumanMessage):
                conversation_text += f"User: {m.content}\n\n"
            elif isinstance(m, AIMessage):
                # Only include actual checks, skip empty tool calls
                if m.content:
                    conversation_text += f"Assistant: {m.content}\n\n"
            elif isinstance(m, SystemMessage):
                # Skip system messages in the summary, Supervisor has its own prompt
                pass
                
        prompt = ChatPromptTemplate.from_messages([
            ("system", SUPERVISOR_SYSTEM_PROMPT),
            ("user", "Here is the conversation so far:\n\n{conversation}\n\nBased on this, who should act next? Return ONLY JSON."),
        ])
        
        chain = prompt | llm | JsonOutputParser()
        try:
            result = chain.invoke({"conversation": conversation_text})
            next_agent = result.get("next", "FINISH")
        except Exception as e:
            # Fallback
            print(f"Supervisor parsing failed: {e}. Defaulting to FINISH.")
            next_agent = "FINISH"
        
        return {"next": next_agent}

    # 2. Worker Agents using create_react_agent logic
    
    developer_tools = [
        inspect_artifacts, generate_plugin_stub, create_file, execute_python_file,
        search_knowledge_base, save_memory, read_memory, list_memories,
        create_temp_knowledge_base, search_temp_knowledge_base, search_web
    ]
    support_tools = [
        upload_sdk_doc, search_knowledge_base, rag_answer, read_memory, list_memories,
        create_temp_knowledge_base, search_temp_knowledge_base, search_web
    ]
    scientist_tools = [
        save_memory, read_memory, list_memories,
        search_knowledge_base, create_temp_knowledge_base, search_temp_knowledge_base, search_web
    ]
    
    def create_worker_node(name, system_prompt, tools):
        # Using create_react_agent. 
        # Note: We handle system_prompt manually via message injection to avoid version compatibility issues with 'state_modifier'.
        agent_executor = create_react_agent(llm, tools)
        
        def worker_node(state: AgentState, config):
            # Retrieve active context from config (thread-safe, non-serialized passing)
            ctx = config.get("configurable", {}).get("context")
            
            if ctx is None:
                # Try to recover context from metadata or logs if possible, or just log error
                print(f"CRITICAL: Context is None in worker_node '{name}'! Config keys: {list(config.keys())}")
            else:
                set_active_context(ctx)

            # Inject system prompt at the beginning of the context for this worker
            messages_with_system = [SystemMessage(content=system_prompt)] + state['messages']
            
            # Invoke the agent
            result = agent_executor.invoke({"messages": messages_with_system})
            
            # Extract new messages.
            # The result['messages'] contains: [SystemMessage, ...original_msgs..., ...new_msgs...]
            # We only want to return the ...new_msgs... to the graph state.
            input_len = len(messages_with_system)
            new_messages = result["messages"][input_len:]
            
            # Add agent identity tag to the final response
            if new_messages and isinstance(new_messages[-1], AIMessage):
                last_msg = new_messages[-1]
                # Prepend the agent name to the content for visibility
                tag = f"**[{name}]**\n\n"
                # Check if tag is already present (to avoid duplication on re-runs)
                if not str(last_msg.content).startswith(tag):
                    last_msg.content = f"{tag}{last_msg.content}"
            
            return {"messages": new_messages}
            
        return worker_node

    developer_node = create_worker_node("Developer", DEVELOPER_PROMPT, developer_tools)
    support_node = create_worker_node("Support", SUPPORT_PROMPT, support_tools)
    # Ensure context is carried over if not returned by nodes
    # (Typical LangGraph behavior, but being explicit doesn't hurt if we define reducers, 
    # but here TypedDict should ideally preserve keys. 
    # However, let's verify if 'context' is actually being stripped.)
    
    scientist_node = create_worker_node("Scientist", SCIENTIST_PROMPT, scientist_tools)

    # 3. Build Graph
    workflow = StateGraph(AgentState)
    
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("Developer", developer_node)
    workflow.add_node("Support", support_node)
    workflow.add_node("Scientist", scientist_node)

    # Edges
    workflow.add_edge(START, "Supervisor")
    
    def route_supervisor(state: AgentState):
        return state["next"]
        
    workflow.add_conditional_edges(
        "Supervisor",
        route_supervisor,
        {
            "Developer": "Developer",
            "Support": "Support",
            "Scientist": "Scientist",
            "FINISH": END
        }
    )
    # Return from workers back to Supervisor to check if more work is needed
    workflow.add_edge("Developer", "Supervisor")
    workflow.add_edge("Support", "Supervisor")
    workflow.add_edge("Scientist", "Supervisor")

    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)
