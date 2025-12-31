import sys
import os
# Add OpenMCS_chatGPT to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "OpenMCS_chatGPT")))

from core.agent import build_agent
from core.schemas import Context

def test_rag_memory():
    print("Building agent...")
    # Assuming default config or one that works. 
    # If api_keys.yaml is missing, this might fail or warn.
    # Ensure you have a valid config or mock it.
    agent = build_agent() 
    
    if not agent:
        print("Failed to build agent. Check configuration.")
        return

    config = {"configurable": {"thread_id": "test-thread-1"}}
    # We reuse context to ensure in-memory objects like VectorStore persist if not handled by checkpointer
    ctx = Context(operator_id="tester")

    print("\n--- Test 1: Save to Memory ---")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Please remember that my favorite color is blue."}]},
        config=config,
        context=ctx
    )
    print("Response:", response['structured_response'].assistant_message)

    print("\n--- Test 2: Read from Memory ---")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "What is my favorite color?"}]},
        config=config,
        context=ctx
    )
    print("Response:", response['structured_response'].assistant_message)

    print("\n--- Test 3: Add to Knowledge Base (RAG) ---")
    # We can use the tool directly or ask the agent to do it.
    # Let's ask the agent to upload a doc (which triggers RAG indexing)
    doc_content = "The flux capacitor requires 1.21 gigawatts of power."
    response = agent.invoke(
        {"messages": [{"role": "user", "content": f"Here is a technical spec: {doc_content}. Please save it as 'flux_spec'."}]},
        config=config,
        context=ctx
    )
    print("Response:", response['structured_response'].assistant_message)

    print("\n--- Test 4: Search Knowledge Base ---")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "How much power does the flux capacitor need?"}]},
        config=config,
        context=ctx
    )
    print("Response:", response['structured_response'].assistant_message)

if __name__ == "__main__":
    test_rag_memory()
