from langchain.tools import tool
import os
import sys
import subprocess

@tool
def create_file(filename: str, content: str) -> str:
    """Create a new file with the specified content. 
    filename: Path to the file. Can be relative to current working directory.
    content: The content to write to the file."""
    try:
        # Ensure directory exists
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"File '{filename}' created successfully."
    except Exception as e:
        return f"Error creating file '{filename}': {str(e)}"

@tool
def execute_python_file(filename: str) -> str:
    """Execute a Python file in the current environment and return the output.
    filename: Path to the python file to execute."""
    try:
        # Run using the current python executable
        result = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            text=True,
            timeout=120 # 2 minute timeout
        )
        output = f"Exit Code: {result.returncode}\n"
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        return output
    except Exception as e:
        return f"Error executing file '{filename}': {str(e)}"

@tool
def execute_in_process_code(code: str) -> str:
    """Execute Python code in the CURRENT running application process.
    Use this to control the OpenMCS application, access instantiated devices, or interact with the UI.
    Now supports REAL-TIME output in the hosting terminal.
    code: The python code to execute.
    """
    import io
    import sys
    import traceback
    
    # Try to import OpenMCS internals
    try:
        from utils.hooks import ServiceManager, OpenedPluginManager, InitConfig
        from utils.base import MCSPluginBase, MCSWidgetBase
    except ImportError:
        return "Error: Could not import OpenMCS utils. Are you running inside the OpenMCS application?"

    # Setup context
    local_scope = {}
    global_scope = {
        "ServiceManager": ServiceManager,
        "OpenedPluginManager": OpenedPluginManager,
        "InitConfig": InitConfig,
        "MCSPluginBase": MCSPluginBase,
        "MCSWidgetBase": MCSWidgetBase,
        "__name__": "__main__",
        "print": print
    }
    
    # Helper class to write to both capture buffer and real stdout/stderr
    class DualWriter:
        def __init__(self, original, capture):
            self.original = original
            self.capture = capture
        
        def write(self, text):
            self.original.write(text)
            # Flush frequently to ensure real-time terminal output
            if text.endswith('\n'):
                self.original.flush() 
            self.capture.write(text)
            
        def flush(self):
            self.original.flush()
            self.capture.flush()
    
    # Capture stdout/stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    # Redirect to DualWriter
    sys.stdout = DualWriter(old_stdout, stdout_capture)
    sys.stderr = DualWriter(old_stderr, stderr_capture)
    
    try:
        exec(code, global_scope, local_scope)
        output = stdout_capture.getvalue()
        error = stderr_capture.getvalue()
        
        result = "Execution Successful.\n"
        if output:
            result += f"Output:\n{output}\n"
        if error:
            result += f"Errors:\n{error}\n"
        return result
    except Exception:
        return f"Execution Failed:\n{traceback.format_exc()}\nOutput so far:\n{stdout_capture.getvalue()}"
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
