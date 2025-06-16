import os
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# Parse the command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help='Path to input CSV file')
parser.add_argument('--embedding_model', required=True, help='Hugging Face model name')
args = parser.parse_args()

INPUT_CSV = args.input
EMBEDDING_MODEL_NAME = args.embedding_model
OUTPUT_CSV = os.path.splitext(INPUT_CSV)[0] + "_cosine.csv"
REASONING_MODELS = ["deepseek-r1:1.5b", "deepseek-r1:14b", "qwen2.5:1.5b", "qwen2.5:14b"]


# Load the embedding model and tokenizer correpsponding to the provided model name
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

#  Load the input csv file, containing the texts to compare
df = pd.read_csv(INPUT_CSV)
solution_candidates = [col for col in df.columns if col.lower() == "solution"]
if not solution_candidates:
    raise ValueError("No 'solution' column found (case-insensitive).")
SOLUTION_COL = solution_candidates[0]
similarity_data = {}

# Add the corresponding id columns and, for math and proofs, the problem type
id_col = "uuid" if "uuid" in df.columns else "QuestionID" if "QuestionID" in df.columns else None
if id_col:
    similarity_data[id_col] = df[id_col].reset_index(drop=True)
if ("math" in INPUT_CSV.lower() or "proofs" in INPUT_CSV.lower()) and "problem_type" in df.columns:
    similarity_data["problem_type"] = df["problem_type"].reset_index(drop=True)

# Function to chunk text into smaller parts for embedding: some text matches may exceed the model's max token limit
def chunk_text(text, max_tokens=512):
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]

def get_embedding(text_or_chunks):
    if isinstance(text_or_chunks, str):
        text_or_chunks = [text_or_chunks]

    embeddings = []
    for chunk in text_or_chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :]
        embeddings.append(emb)
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings.mean(dim=0, keepdim=True)

def safe_strip(x):
    if isinstance(x, str):
        return x.strip()
    else:
        return ""


# Main loop to calculate cosine similarity for each reasoning model answer against the solution
for model_name in tqdm(REASONING_MODELS, desc="Processing models"):
    cosine_values = []

    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"Processing row {i} for model {model_name}")
            
        reasoning_text = row.get(f"{model_name}_reasoning", "")
        solution_text = row.get(SOLUTION_COL, "")

        if not safe_strip(reasoning_text) or not safe_strip(solution_text):
            cosine_values.append(float("nan"))
            continue    

        try:
            emb_reasoning = get_embedding(chunk_text(reasoning_text))
            emb_solution = get_embedding(chunk_text(solution_text))

            if torch.isnan(emb_reasoning).any() or torch.isnan(emb_solution).any():
                print(f"NaN detected at row {i}, model {model_name}")
                sim = float("nan")
            else:
                sim = F.cosine_similarity(emb_reasoning, emb_solution).item()
        except Exception as e:
            print(f"Error at row {i}, model {model_name}: {e}")
            sim = float("nan")

        cosine_values.append(sim)

    similarity_data[f"{model_name}_cosine"] = cosine_values

    if f"{model_name}_time" in df.columns:
        similarity_data[f"{model_name}_time"] = df[f"{model_name}_time"].reset_index(drop=True)


output_df = pd.DataFrame(similarity_data)
output_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Done. Output saved to: {OUTPUT_CSV}")