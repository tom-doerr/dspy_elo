#!/usr/bin/env python3
"""
Script to run all isolated tests without using pytest.
"""

import os
import importlib.util
import sys
import traceback

def import_module_from_file(file_path):
    """Import a module from a file path."""
    module_name = os.path.basename(file_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def run_test_functions(module):
    """Run all test functions in a module."""
    test_functions = [name for name in dir(module) if name.startswith('test_')]
    for test_name in test_functions:
        test_function = getattr(module, test_name)
        if callable(test_function):
            print(f"Running {test_name}...")
            try:
                test_function()
                print(f"✓ {test_name} passed")
            except Exception as e:
                print(f"✗ {test_name} failed: {str(e)}")
                traceback.print_exc()
                return False
    return True

def main():
    """Run all isolated tests."""
    isolated_tests_dir = os.path.join(os.path.dirname(__file__), 'isolated_tests')
    test_files = [f for f in os.listdir(isolated_tests_dir) if f.startswith('test_') and f.endswith('.py')]
    
    total_tests = len(test_files)
    passed_tests = 0
    failed_tests = []
    
    print(f"Found {total_tests} test files")
    
    for test_file in test_files:
        file_path = os.path.join(isolated_tests_dir, test_file)
        print(f"\nRunning tests from {test_file}...")
        
        try:
            module = import_module_from_file(file_path)
            if run_test_functions(module):
                passed_tests += 1
            else:
                failed_tests.append(test_file)
        except Exception as e:
            print(f"Error importing {test_file}: {str(e)}")
            traceback.print_exc()
            failed_tests.append(test_file)
    
    print(f"\n{'=' * 50}")
    print(f"Test Results: {passed_tests}/{total_tests} test files passed")
    
    if failed_tests:
        print("\nFailed test files:")
        for test_file in failed_tests:
            print(f"  - {test_file}")
        return 1
    
    print("\nAll tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
