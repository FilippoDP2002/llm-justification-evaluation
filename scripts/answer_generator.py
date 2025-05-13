
import subprocess
import sys
import argparse
import sqlite3
import os
import time # Added for timing
import csv # Added for CSV writing

def query_ollama(model_name: str, prompt: str):
    """
    Queries the Ollama model with the given prompt.

    Args:
        model_name: The name of the Ollama model to use.
        prompt: The prompt to send to the model.

    Returns:
        The model's response as a string, or an error message.
    """
    try:
        # Record start time
        start_time = time.time()
        
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt.encode('utf-8'),
            capture_output=True,
            # timeout=60 # Consider re-enabling or adjusting timeout if needed
        )
        
        # Record end time
        end_time = time.time()
        
        response_text = result.stdout.decode('utf-8').strip()
        
        # Calculate time taken in seconds
        time_taken = end_time - start_time
        
        # Return both response and time_taken
        return response_text, time_taken
    except Exception as e:
        return f"Error: {e}", 0 # Return 0 for time_taken in case of error

def main():
    """
    Main function to process math questions, query Ollama, and save results.
    """
    parser = argparse.ArgumentParser(
        description="Ask math questions to a local LLM via Ollama.",
        formatter_class=argparse.RawTextHelpFormatter # To preserve formatting in help text
    )

    parser.add_argument(
        "--model",
        required=True,
        help="The name of the Ollama model to use (e.g., 'llama3', 'mistral')."
    )
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="If enabled, the output file is cleared before execution."
    )

    args = parser.parse_args()

    model_name = args.model
    # Define paths (consider making these configurable or relative to script location)
    # Assuming the script is run from a directory where '../data/' path is valid.
    # Adjust these paths if your directory structure is different.
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Get directory of the script
    prompt_file_path = os.path.join(script_dir, "../data/prompts/math_question_prompt.txt")
    db_path = os.path.join(script_dir, "../data/datasets/math_questions.db")
    output_path = os.path.join(script_dir, "../data/generated_data/math_answers.csv")
    header = ["id", "question", "response", "model", "time_taken_seconds"]

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompt_prefix = f.read().strip()
    except FileNotFoundError:
        print(f"Prompt file not found: {prompt_file_path}")
        sys.exit(1)

    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        sys.exit(1)

    # Clear the output file and write the header
    if(args.reset_output): 
        try:
            with open(output_path, "w", newline='', encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)
                # Define header row
                csv_writer.writerow(header)
        except IOError as e:
            print(f"Error initializing output file {output_path}: {e}")
            sys.exit(1)

    conn = None # Initialize conn to None
    try:
        conn = sqlite3.connect(db_path)
        # Using pandas to read SQL is convenient, but for large datasets,
        # consider processing row by row directly from cursor.
        # For now, keeping it similar to original, but limiting to 1 for testing.
        # Remove 'LIMIT 1' for processing all questions.
        cursor = conn.cursor()
        cursor.execute("SELECT id, problem FROM math_questions LIMIT 1") # Removed LIMIT 1

        row = cursor.fetchone() # Fetch the first row
        while row:
            question_id, problem_text = row

            full_prompt = f"{prompt_prefix}\n\n{problem_text}"
            
            print(f"\nProcessing question ID: {question_id}...")
            response, time_taken = query_ollama(model_name, full_prompt)
            
            if "Error:" in response and time_taken == 0: # Check if query_ollama returned an error
                print(f"Failed to get response for question ID {question_id}: {response}")
            else:
                print(f"Response received in {time_taken:.2f} seconds.")
                # print(f"Response: {response[:100]}...") # Print a snippet of the response

            # Prepare data for CSV
            result_data = {
                "id": question_id,
                "question": problem_text,
                "response": response,
                "model": model_name,
                "time_taken_seconds": f"{time_taken:.1f}" # Format time_taken
            }

            # Append the new result to the CSV file
            try:
                with open(output_path, "a", newline='', encoding="utf-8") as csvfile:
                    csv_writer = csv.DictWriter(csvfile, fieldnames=header)
                    # No need to write header again, DictWriter can use fieldnames
                    csv_writer.writerow(result_data)
                print(f"Appended answer for question ID {question_id} to {output_path}")
            except IOError as e:
                print(f"Error writing to output file {output_path} for question ID {question_id}: {e}")
            
            row = cursor.fetchone() # Fetch the next row

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()

    print(f"\nProcessing complete. Results saved in {output_path}")

if __name__ == "__main__":
    main()
