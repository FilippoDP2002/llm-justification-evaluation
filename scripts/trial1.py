import pandas as pd
import subprocess
import sys
from datetime import datetime
import sqlite3
import os

def query_ollama(model_name: str, prompt: str) -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt.encode('utf-8'),
            capture_output=True,
            #timeout=60
        )
        return result.stdout.decode('utf-8').strip()
    except Exception as e:
        return f"Error: {e}"

def main():
    if len(sys.argv) < 2:
        print("Usage: python trial.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]

    with open("prompts/math_question_prompt.txt", "r", encoding="utf-8") as f:
        prompt_prefix = f.read().strip()

    db_path = "data/math_questions_trial.db"
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT id, problem FROM math_questions LIMIT 5", conn)
    conn.close()

    results = []
    for _, row in df.iterrows():
        full_prompt = f"{prompt_prefix}\n\n{row['problem']}"
        response = query_ollama(model_name, full_prompt)
        print('response')
        print(response)
        timestamp = datetime.now().isoformat()
        results.append({
            "id": row['id'],
            "question": row['problem'],
            "response": response,
            "model": model_name,
            "timestamp": timestamp
        })

    os.makedirs("results", exist_ok=True)
    output_path = f"results/trial_{model_name.replace(':','_')}.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
