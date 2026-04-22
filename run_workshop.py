#!/usr/bin/env python3
"""
Wrapper script to run the bash workshop script.
Allows users to execute: python run_workshop.py
Instead of: bash run_workshop.sh
"""

import subprocess
import sys
import os

def main():
    """Execute the bash workshop script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bash_script = os.path.join(script_dir, 'run_workshop.sh')
    
    if not os.path.exists(bash_script):
        print(f"❌ Error: {bash_script} not found")
        sys.exit(1)
    
    try:
        # Execute the bash script
        result = subprocess.run(['bash', bash_script], check=False)
        sys.exit(result.returncode)
    except FileNotFoundError:
        print("❌ Error: bash command not found. This script requires bash to be installed.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠ Workshop interrupted by user")
        sys.exit(130)

if __name__ == '__main__':
    main()
