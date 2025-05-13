import pandas as pd
import ollama
import sys
from datetime import datetime
import sqlite3
import os

def query_ollama(model_name: str, problem: str, sys_prompt:str) -> str:
    try:
        result = ollama.generate(model=model_name, prompt=problem, system=sys_prompt)
        return result['response']
    except Exception as e:
        return f"Error: {e}"

def main():
    if len(sys.argv) < 2:
        print("Usage: python trial.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]

    with open("../data/prompts/math_question_prompt.txt", "r", encoding="utf-8") as f:
        sys_prompt = f.read().strip()

    db_path = "../data/datasets/math_questions.db"
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT id, problem FROM math_questions LIMIT 1", conn)
    conn.close()

    results = []
    for _, row in df.iterrows():
        question_id, problem_text = row
        response = query_ollama(model_name, problem_text, sys_prompt)
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

    output_path = f"../data/generated_data/math_answers.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
