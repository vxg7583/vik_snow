import subprocess
import pandas as pd
def run_script(script_name):
    try:
        subprocess.run(["python", script_name], check=True)
        print(f"Successfully executed {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {script_name}: {e}")
if __name__ == "__main__":
    run_script("data_transform.py")
    run_script("train.py")
    run_script("eval.py")
