import os
import subprocess

os.chdir(os.path.expanduser("~"))

INPUT_DIR = "NLP_analysis"

EMBEDDING_MAP = {
    "math_analysis.csv": "MathBERT",
    "proofs_analysis.csv": "MathBERT",
    "critical_reasoning_analysis.csv": "SBERT",
    "essay_evaluation_analysis.csv": "SBERT",
    "reading_comprehension_analysis.csv": "SBERT"
}

with open("cosine_calculator_semantic/berts.txt", "r") as f:
    bert_models = [line.strip() for line in f if line.strip()]

for input_csv, embedding_type in EMBEDDING_MAP.items():
    if embedding_type == "MathBERT":
        embedding_model = bert_models[0] 
    else:
        embedding_model = bert_models[1]  

    print(f"Processing {input_csv} with model {embedding_model}")

    cmd = [
        "python", "cosine_calculator_semantic/cosine_similarity_calculator.py",
        "--input", os.path.join(INPUT_DIR, input_csv),
        "--embedding_model", embedding_model,
    ]

    subprocess.run(cmd, check=True)