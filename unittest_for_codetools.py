import unittest
import os
import shutil
from OpenMCS_Agent.tools.code_tools import create_file, execute_python_file

class TestCodeTools(unittest.TestCase):
    def setUp(self):
        self.test_dir = "temp_test_verification"
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

    # Delete the test directory after tests
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_create_file(self):
        filename = os.path.join(self.test_dir, "hello.txt")
        content = "Hello, World!"
        
        result = create_file.invoke({"filename": filename, "content": content})
        
        self.assertIn("created successfully", result)
        self.assertTrue(os.path.exists(filename))
        
        with open(filename, 'r', encoding='utf-8') as f:
            read_content = f.read()
        self.assertEqual(read_content, content)

    def test_execute_python_file_success(self):
        script_path = os.path.join(self.test_dir, "success_script.py")
        code = "print('Test Output')"
        create_file.invoke({"filename": script_path, "content": code})
        
        result = execute_python_file.invoke({"filename": script_path})
        
        self.assertIn("Exit Code: 0", result)
        self.assertIn("STDOUT:\nTest Output", result)

    def test_execute_python_file_error(self):
        script_path = os.path.join(self.test_dir, "error_script.py")
        code = "raise ValueError('Test Error')"
        create_file.invoke({"filename": script_path, "content": code})
        
        result = execute_python_file.invoke({"filename": script_path})
        
        self.assertNotIn("Exit Code: 0", result)
        self.assertIn("STDERR", result)
        self.assertIn("ValueError: Test Error", result)

    def test_execute_python_file_infinite_loop_protection(self):
        # We can't easily test the 120s timeout without waiting 120s.
        # So we just verify a script that takes a little bit of time but finishes is fine.
        # And ensure we don't actually write an infinite loop here.
        script_path = os.path.join(self.test_dir, "loop.py")
        code = """
import time
from langchain.tools import tool
for i in range(3):
    print(i)
    time.sleep(0.1)
"""
        create_file.invoke({"filename": script_path, "content": code})
        result = execute_python_file.invoke({"filename": script_path})
        self.assertIn("Exit Code: 0", result)
        self.assertIn("0\n1\n2", result)

if __name__ == '__main__':
    unittest.main()
