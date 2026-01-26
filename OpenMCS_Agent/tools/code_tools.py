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
