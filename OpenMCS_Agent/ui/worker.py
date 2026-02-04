from PyQt5.QtCore import QThread, pyqtSignal
# from langchain_core.tracers.stdout import ConsoleCallbackHandler # Unused and potentially slow import
# from core.agent import build_agent # Moved to run() to avoid heavy top-level imports

class AgentInitializeWorker(QThread):
    """Initializes the Agent in a background thread"""
    finished_signal = pyqtSignal(object, str) # agent_instance, provider_name
    error_signal = pyqtSignal(str)

    def __init__(self, provider_name, mode="multi"):
        super().__init__()
        self.provider_name = provider_name
        self.mode = mode

    def run(self):
        try:
            # Import here to avoid freezing UI during initial load
            from core.agent import build_agent
            agent = build_agent(self.provider_name, mode=self.mode)
            self.finished_signal.emit(agent, self.provider_name)
        except Exception as e:
            self.error_signal.emit(str(e))

class AgentWorker(QThread):
    """在后台运行 Agent 的 invoke 方法，避免阻塞 UI"""
    result_ready = pyqtSignal(object) # 发送 ResponseFormat 对象或错误信息

    def __init__(self, agent, user_input, config, context, images=None):
        super().__init__()
        self.agent = agent
        self.user_input = user_input
        self.config = config
        self.context = context
        self.images = images or []

    def run(self):
        import base64
        from langchain_core.messages import HumanMessage, AIMessage

        try:
            # 1. Prepare Configuration
            run_config = self.config.copy()
            if "configurable" not in run_config:
                run_config["configurable"] = {}
            run_config["configurable"]["context"] = self.context

            # 2. Prepare Content (Text + Images)
            structured_content = [{"type": "text", "text": self.user_input}]
            
            for img_path in self.images:
                try:
                    with open(img_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        
                    mime_type = "image/png"
                    if img_path.lower().endswith(('.jpg', '.jpeg')):
                        mime_type = "image/jpeg"
                        
                    structured_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{encoded_string}"}
                    })
                except Exception as e:
                    print(f"Failed to encode image {img_path}: {e}")

            # 3. Invoke Agent
            input_message = HumanMessage(content=structured_content)
            
            response = self.agent.invoke(
                {"messages": [input_message]},
                config=run_config
            )

            # 4. Extract Final Response
            if "messages" in response and len(response["messages"]) > 0:
                final_msg = response["messages"][-1]
                
                # Check if the last message is HumanMessage (Input).
                # This happens if the graph execution failed (e.g. Supervisor error) 
                # and returned the input state without adding any AI response.
                if isinstance(final_msg, HumanMessage):
                    result = "⚠️ Agent execution failed or stopped early (No response generated).\nPlease check the terminal logs for 'Supervisor parsing failed' or other errors."
                elif hasattr(final_msg, "content"):
                     result = final_msg.content
                else:
                     result = str(final_msg)
            else:
                 result = "No response generated."

            self.result_ready.emit(result)

        except Exception as e:
            err_msg = str(e)
            import traceback
            traceback.print_exc()
            
            if "insufficient tool messages" in err_msg or "tool_calls" in err_msg:
                self.result_ready.emit(
                    "⚠️ Session state corrupted. Please click the Reset button (↻) to start fresh."
                )
            else:
                self.result_ready.emit(f"Error: {err_msg}")
