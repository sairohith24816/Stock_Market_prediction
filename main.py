import sys
import os
import subprocess

def main():
    """
    Wrapper script to run the main application logic located in python/main.py.
    Passes all command-line arguments to the inner script.
    """
    # Get the absolute path to the python/main.py script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, 'python', 'main.py')
    
    # Check if the script exists
    if not os.path.exists(script_path):
        print(f"Error: Could not find {script_path}")
        sys.exit(1)

    # Construct the command to run the script
    # We use sys.executable to ensure we use the same Python interpreter
    cmd = [sys.executable, script_path] + sys.argv[1:]
    
    try:
        # Run the script and wait for it to complete
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        sys.exit(1)

if __name__ == "__main__":
    main()
