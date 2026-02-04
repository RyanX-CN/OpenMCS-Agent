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
from tools.basic_tools import upload_sdk_doc, inspect_artifacts, generate_plugin_stub
from tools.code_tools import create_file, execute_python_file, execute_in_process_code
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
1. **Developer**: Specialized in:
    - Writing OpenMCS extensions and device plugins.
    - **CONTROLLING DEVICES** (stages, cameras, etc.) and interacting with hardware. 
    - **EXECUTING CODE** in the live environment.
    - Debugging errors.
2. **Support**: Specialized in explaining software usage, reading manuals, and providing instructions. **DO NOT** route code execution or device control tasks here.
3. **Scientist**: Specialized in answering experimental scientific questions and general reasoning.

Rules:
- Analyze the user request and conversation history.
- **CRITICAL**: If the previous message was from a Worker (marked with [Developer], [Support], etc.) and it appears to answer the user's question, you **MUST** output 'FINISH'.
- Do NOT route back to the same worker immediately unless they explicitly request it or failed.
- If the conversation has reached a natural conclusion, output 'FINISH'.
- IF the user asks to control hardware, run code, or debug -> route to Developer.
- IF the user asks how to use the software GUI, where a button is, or for documentation -> route to Support.
- IF the user asks about scientific principles OR uploads an image for analysis -> route to Scientist.
- If you are unsure, default to 'FINISH' (let the user ask again).

IMPORTANT:
- Do NOT utilize any tools.
- Do NOT answer the user's request directly.
- ONLY output the JSON decision.
- Your output must be valid JSON with a single key 'next'.

Output your decision as a JSON object with a single key 'next' mapping to the worker name (Developer, Support, Scientist) or 'FINISH'.
"""

DEVELOPER_PROMPT = """You are the OpenMCS Extension Developer.
Your primary task is to implement new device plugins, maintain the codebase, and CONTROL DEVICES.
You have access to tools for file operations and code execution.

CRITICAL RULES FOR DEVICE CONTROL:
1. Check Worker Existence: Before controlling a device, verify if its Plugin Worker is already created/opened.
   - Use "OpenedPluginManager.get_opened_plugins()".
2. Check Connection: If the worker exists, check if the device is actually connected/opened (check "m_device_dict"). If not, call 'worker.open_device()'.
3. Execute: Only after ensuring the device is ready, execute the control command.
   - To control live hardware or access running plugins, you must use the "execute_in_process_code" tool. Do NOT try to run `main.py` or `execute_python_file` for this purpose, as that creates a separate process without access to the hardware.

   4. Simplicity: Do NOT write complex wrapper classes or try to "re-implement" the plugin logic in your script. Just find the existing instance and call its methods.

CRITICAL RULES FOR FILE OPERATIONS:
1. ASK PERMISSION: You MUST ask the user for explicit confirmation before creating or overwriting ANY local files (using `create_file`).
2. Exception: If the user explicitly asked you to "create a file named X", you may proceed without a second confirmation.

**Code Execution Environment:**
- You are running INSIDE the OpenMCS process when using `execute_in_process_code`.
- `ServiceManager`, `OpenedPluginManager`, `MCSPluginBase` are available in the global scope.
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
            # Handle structured content (text + image) to avoid dumping base64 to Supervisor
            content_display = ""
            if isinstance(m.content, list):
                for block in m.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content_display += block.get("text", "") + " "
                    elif isinstance(block, dict) and block.get("type") == "image_url":
                        content_display += "[Image Uploaded] "
            else:
                content_display = str(m.content)

            if isinstance(m, HumanMessage):
                conversation_text += f"User: {content_display}\n\n"
            elif isinstance(m, AIMessage):
                # Only include actual checks, skip empty tool calls
                if content_display:
                    conversation_text += f"Assistant: {content_display}\n\n"
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
        inspect_artifacts, generate_plugin_stub, create_file, execute_python_file, execute_in_process_code,
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
            messages_with_system = [SystemMessage(content=system_prompt)]
            
            # Filter and convert messages for the worker if needed
            # DeepSeek / Non-Vision Models specific fix:
            # If the model provider is known to NOT support vision, we must strip the image_url content blocks
            # Or convert them to a placeholder text.
            # Assuming 'llm' instance has some provider info, but it might be generic.
            # However, for robustness, we can inspect messages.
            
            for m in state['messages']:
                if isinstance(m, HumanMessage) and isinstance(m.content, list):
                    # Check if it has images
                    has_image = any(isinstance(block, dict) and block.get("type") == "image_url" for block in m.content)
                    if has_image:
                        # Create a safe copy of the message for models that might crash on image_url
                        # If the current model SUPPORTS vision (e.g. GPT-4o), we want to keep it.
                        # If it DOES NOT (e.g. DeepSeek-V3, legacy OpenAI), we MUST remove it.
                        # Since we don't dynamically check model capabilities here easily, 
                        # we can try to rely on the fact that if the user uploaded an image, they EXPECT vision.
                        
                        # BUT, the error 400 'unknown variant `image_url`' indicates the backend model 
                        # explicitely rejected the format. 
                        # We should try to fallback or sanitize if we suspect non-vision model.
                        
                        # Fix: DeepSeek API (and some others) strictly fail on 'image_url' if not supported.
                        # We will construct a new list without image_url if the model is deemed "text-only" 
                        # OR if we want to be safe, we can just pass it through and hope. 
                        # But here we clearly crashed.
                        
                        # Strategy: Check if the model name implies vision? 
                        # Better strategy: Catch the 400 error? No, graph execution is hard to catch mid-node easily without retry logic.
                        
                        # Robust Fix: If the message has images, but we are sending to 'Developer' or 'Support' (who might use text models),
                        # we should maybe strip it? 
                        # Actually, Scientist is the one likely receiving it. 
                        # If Scientist is using DeepSeek-V3 (which is text-only usually, unlike R1/Vis), it will crash.
                        
                        # Let's clean the message for the worker execution IF it fails?
                        # No, we must prevention.
                        
                        # We will convert the mix of text/image content into just text if it's a list
                        # UNLESS we are sure. But since we can't be sure, let's look at the error.
                        # "unknown variant image_url".
                        
                        # We will modify the message to simple text "[Image Uploaded]" for the prompt
                        # IF we are running on a non-vision model. 
                        # Since I can't easily change the model at runtime here without major refactor,
                        # I will sanitize the input for now to prevent the crash, assuming the user might be using a text model
                        # but still wants to try.
                        
                        # Wait, if the user wants Scientist to analyze image, they NEED a vision model.
                        # If they selected "DeepSeek" (text model), it will physically fail to send.
                        # So we MUST strip it to avoid crash, even if it means they can't see the image.
                        # The better fix is to use a Vision model (GPT-4o).
                        
                        # However, to fix the code crash:
                        # We check if the configured LLM is likely vision-capable? 
                        # Hard to say.
                        
                        # Safest approach for STABILITY: 
                        # If the model is NOT 'gpt-4o', 'claude-3-5', 'gemini-pro-vision' etc, we strip images.
                        # But config is global.
                        
                        # Let's try to pass it. If it crashes, it crashes. 
                        # But the user IS checking DeepSeek which IS crashing.
                        # I will add a sanitization step that keeps the text and replaces image with a placeholder
                        # IF the model name doesn't look like a vision model.
                        
                        model_name = getattr(llm, "model_name", "").lower()
                        # Keywords that suggest typical Vision-Language Models
                        vision_keywords = ["gpt-4", "claude-3", "gemini", "vision", "omni", "qwen", "vl", "image"]
                        is_vision = any(v in model_name for v in vision_keywords)
                        
                        if not is_vision:
                            # Strip images for non-vision models to prevent 400 Bad Request
                            text_only_blocks = [b for b in m.content if b.get("type") == "text"]
                            clean_text = " ".join([b.get("text", "") for b in text_only_blocks])
                            clean_text += "\n[Image Attachment Ignored: Model does not support vision]"
                            messages_with_system.append(HumanMessage(content=clean_text))
                        else:
                            messages_with_system.append(m)
                    else:
                         messages_with_system.append(m)
                else:
                    messages_with_system.append(m)
            
            # messages_with_system = [SystemMessage(content=system_prompt)] + state['messages'] (OLD)
            
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
    agent = workflow.compile(checkpointer=checkpointer)
    
    # graph_png = agent.get_graph(xray=True).draw_mermaid_png()
    # with open("agent_workflow.png", "wb") as f:
    #     f.write(graph_png)
    # print("Workflow graph saved to agent_workflow.png")
        
    return agent