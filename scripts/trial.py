import pandas as pd
import subprocess
import sys
from datetime import datetime
import os

def query_ollama(model_name: str, prompt: str) -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt.encode('utf-8'),
            capture_output=True,
            timeout=30
        )
        return result.stdout.decode('utf-8').strip()
    except Exception as e:
        return f"Error: {e}"

def main():
    if len(sys.argv) < 2:
        print("Usage: python trial.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]

    input_file = "data/trial.csv"
    output_file = f"results/trial_{model_name.replace(':','_')}.csv"

    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        sys.exit(1)

    df = pd.read_csv(input_file)

    results = []
    for question in df['question']:
        response = query_ollama(model_name, question)
        timestamp = datetime.now().isoformat()
        results.append({
            "question": question,
            "response": response,
            "model": model_name,
            "timestamp": timestamp
        })

    pd.DataFrame(results).to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
