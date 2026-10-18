import sys
import subprocess
from pathlib import Path

def run_all_dynamic_scripts():
    # 1. Scan the current directory for all .py files except this master script
    current_dir = Path(__file__).parent
    python_files = [f for f in current_dir.glob("*.py") if f.name != Path(__file__).name]
    
    print(f"Found {len(python_files)} scripts to execute.\n")
    
    # Use the current environment's Python executable path automatically
    python_exe = sys.executable

    for file_path in python_files:
        print(f"--- Executing Script: {file_path.name} ---")
        try:
            # 2. Spawn a separate terminal process for each file
            result = subprocess.run([python_exe, str(file_path)], capture_output=False, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Script {file_path.name} failed with exit code {e.returncode}")
        print()

if __name__ == "__main__":
    run_all_dynamic_scripts()
