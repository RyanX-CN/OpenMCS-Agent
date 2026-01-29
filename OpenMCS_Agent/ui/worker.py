from PyQt5.QtCore import QThread, pyqtSignal
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from core.agent import build_agent

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
            agent = build_agent(self.provider_name, mode=self.mode)
            self.finished_signal.emit(agent, self.provider_name)
        except Exception as e:
            self.error_signal.emit(str(e))

class AgentWorker(QThread):
    """在后台运行 Agent 的 invoke 方法，避免阻塞 UI"""
    result_ready = pyqtSignal(object) # 发送 ResponseFormat 对象或错误信息

    def __init__(self, agent, user_input, config, context):
        super().__init__()
        self.agent = agent
        self.user_input = user_input
        self.config = config
        self.context = context

    def run(self):
        # # Configure callbacks for command line logging to show trace of the chain
        # callbacks = self.config.get("callbacks", [])
        # if not any(isinstance(c, ConsoleCallbackHandler) for c in callbacks):
        #     callbacks.append(ConsoleCallbackHandler())
        # self.config["callbacks"] = callbacks

        try:
            # Attempt 1: Try passing content as a list of content blocks (OpenAI structured format)
            # The provider explicitly requested 'ChatCompletionRequestContentBlock'
            structured_content = [{"type": "text", "text": self.user_input}]
            
            response = self.agent.invoke(
                {"messages": [{"role": "user", "content": structured_content}]},
                config=self.config,
                context=self.context
            )
            self.result_ready.emit(response)

        except Exception as e:
            err_msg = str(e)
            
            # Check for corrupted history (dangling tool call)
            if "insufficient tool messages" in err_msg or "tool_calls" in err_msg:
                self.result_ready.emit(
                    "⚠️ Session state corrupted (dangling tool call). "
                    "Please click the Reset button (↻) in the toolbar to start fresh."
                )
                return

            print(f"Primary invocation failed: {err_msg}. Retrying with list of strings...")
            
            try:
                # Attempt 2: Fallback to list of strings (some strict validations might accept this)
                response = self.agent.invoke(
                    {"messages": [{"role": "user", "content": [self.user_input]}]},
                    config=self.config,
                    context=self.context
                )
                print(response)
                self.result_ready.emit(response)
                
            except Exception as e2:
                err_msg_2 = str(e2)
                if "insufficient tool messages" in err_msg_2:
                    self.result_ready.emit(
                        "⚠️ Session state corrupted. Please click Reset (↻) to clear history."
                    )
                else:
                    self.result_ready.emit(f"Error: {err_msg} | Fallback: {err_msg_2}")