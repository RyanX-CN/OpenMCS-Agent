from PyQt5.QtCore import QThread, pyqtSignal

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
        try:
            response = self.agent.invoke(
                {"messages": [{"role": "user", "content": self.user_input}]},
                config=self.config,
                context=self.context
            )
            structured_res = response.get('structured_response')
            self.result_ready.emit(structured_res)
        except Exception as e:
            self.result_ready.emit(f"Error: {str(e)}")