from typing import Literal
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage

from config.settings import get_model_config
from core.schemas import Context, ResponseFormat
from core.single_agent import build_single_agent
from core.multi_agent import build_multi_agent_graph, get_chat_model_instance
from core.context_manager import set_active_context

class MultiAgentWrapper:
    """Wrapper to make LangGraph output compatible with Main Window UI expecting ResponseFormat"""
    def __init__(self, graph):
        self.graph = graph

    def invoke(self, input, config=None, context=None, **kwargs):
        if config is None:
            config = {}
            
        if context:
             if "configurable" not in config:
                 config["configurable"] = {}
             config["configurable"]["context"] = context
             # Crucial: Set the active context for tools running in this thread
             set_active_context(context)
        
        config["recursion_limit"] = 20  # Reasonable limit for complex tasks
        
        result = self.graph.invoke(input, config=config, **kwargs)
        messages = result.get("messages", [])
        if messages:
            last_msg = messages[-1]
            content = last_msg.content
        else:
            content = "No response generated."
            
        # Try to extract potential files or actions from context if available
        files = None
        
        return ResponseFormat(assistant_message=str(content), files=files)

class AgentFactory:
    """Factory to create different types of agents (Single vs Multi-Agent)"""
    @staticmethod
    def create_agent(mode: Literal["single", "multi"], config_name=None):
        if mode == "single":
            graph = build_single_agent(config_name)
        else:
            graph = build_multi_agent_graph(config_name)
        return MultiAgentWrapper(graph)

def build_agent(config_name=None, mode: str = "multi"):
    """
    Builds and returns an Agent instance.
    
    Args:
        config_name: Configuration profile name.
        mode: 'multi' for Supervisor-Multi-Agent, 'single' for Standalone ReAct Agent.
    """
    return AgentFactory.create_agent(mode, config_name)
