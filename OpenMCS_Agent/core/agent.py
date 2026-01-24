from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage

from config.settings import get_model_config
from core.schemas import Context, ResponseFormat
from OpenMCS_Agent.core.multi_agent import build_multi_agent_graph

class MultiAgentWrapper:
    """Wrapper to make LangGraph output compatible with Main Window UI expecting ResponseFormat"""
    def __init__(self, graph):
        self.graph = graph

    def invoke(self, input, config=None, context=None, **kwargs):
        # Inject context into config for runtime access without serialization
        if config is None:
            config = {}
            
        if context:
             if "configurable" not in config:
                 config["configurable"] = {}
             config["configurable"]["context"] = context
        
        # Set a recursion limit to prevent infinite loops (user requested limit)
        config["recursion_limit"] = 20  # Reasonable limit for complex tasks
        
        # Invoke the graph
        result = self.graph.invoke(input, config=config, **kwargs)
        
        # Process result to match ResponseFormat
        messages = result.get("messages", [])
        if messages:
            last_msg = messages[-1]
            content = last_msg.content
        else:
            content = "No response generated."
            
        # Try to extract potential files or actions from context if available
        files = None
        
        return ResponseFormat(assistant_message=str(content), files=files)

def build_agent(config_name=None):
    """构建并返回 Agent 实例"""
    graph = build_multi_agent_graph(config_name)
    return MultiAgentWrapper(graph)
