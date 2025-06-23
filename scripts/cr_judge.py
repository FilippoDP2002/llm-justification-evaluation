import pandas as pd
import ollama
import json
import random
import os
from datetime import datetime
import time

class GMATJudge:
    def __init__(self, system_prompt_file, csv_file, judge_model="deepseek-r1:32b"):
        self.system_prompt_file = system_prompt_file
        self.csv_file = csv_file
        self.judge_model = judge_model
        self.system_prompt = self._load_system_prompt()
        self.data = self._load_csv_data()
        
        # Model names for mapping
        self.model_names = [
            "deepseek-r1:1.5b",
            "deepseek-r1:14b", 
            "qwen2.5:1.5b",
            "qwen2.5:14b"
        ]
        
        # Greek letters for randomization
        self.greek_letters = ["Alpha", "Beta", "Gamma", "Delta"]
        
        # Results storage
        self.judge_outputs = []
        self.evaluation_results = []
    
    def _load_system_prompt(self):
        """Load the system prompt from file."""
        try:
            with open(self.system_prompt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"System prompt file '{self.system_prompt_file}' not found.")
            return ""
    
    def _load_csv_data(self):
        """Load the CSV data containing GMAT questions and candidate answers."""
        try:
            return pd.read_csv(self.csv_file)
        except FileNotFoundError:
            print(f"CSV file '{self.csv_file}' not found.")
            return pd.DataFrame()
    
    def _create_random_mapping(self):
        """Create a random mapping between models and Greek letters."""
        shuffled_letters = self.greek_letters.copy()
        random.shuffle(shuffled_letters)
        
        mapping = {}
        reverse_mapping = {}
        
        for i, model in enumerate(self.model_names):
            letter = shuffled_letters[i]
            mapping[model] = letter
            reverse_mapping[letter] = model
        
        return mapping, reverse_mapping
    
    def _query_ollama(self, prompt, max_retries=3):
        """Query the Ollama model with retry logic."""
        for attempt in range(max_retries):
            try:
                response = ollama.generate(
                    model=self.judge_model,
                    prompt=prompt,
                    system=self.system_prompt,
                    options={
                        "temperature": 0.1,  # Low temperature for consistency
                        "top_p": 0.9
                    }
                )
                return response['response'].strip()
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"Failed to get response after {max_retries} attempts")
                    return ""
    
    def _format_question_for_judge(self, row, model_mapping):
        """Format the question and candidate answers for the judge."""
        question_id = row['QuestionID']
        official_solution = row['Solution']
        
        # Get reasoning and solutions for each model
        candidate_data = {}
        for model in self.model_names:
            reasoning_col = f"{model}_reasoning"
            solution_col = f"{model}_solution"
            
            reasoning = row.get(reasoning_col, "No reasoning provided")
            solution = row.get(solution_col, "No solution provided")
            
            letter = model_mapping[model]
            candidate_data[letter] = {
                'reasoning': reasoning,
                'solution': solution,
                'model': model
            }
        
        # Format the prompt
        prompt = f"""# Question
Question ID: {question_id}
{row.get('Answer', 'Question text not available')}

# Solution
{official_solution}

"""
        
        # Add candidate answers in Greek letter order
        for letter in self.greek_letters:
            if letter in candidate_data:
                prompt += f"# Candidate answer {letter}\n"
                prompt += f"Reasoning: {candidate_data[letter]['reasoning']}\n"
                prompt += f"Final Answer: {candidate_data[letter]['solution']}\n\n"
        
        return prompt, candidate_data
    
    def _parse_judge_response(self, response):
        """Parse the judge's ranking response."""
        lines = response.strip().split('\n')
        ranking = {}
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line and any(letter in line for letter in self.greek_letters):
                # Extract the Greek letter from the line
                for letter in self.greek_letters:
                    if letter in line:
                        ranking[i] = letter
                        break
        
        return ranking
    
    def evaluate_all_questions(self):
        """Evaluate all questions in the dataset."""
        total_questions = len(self.data)
        print(f"Starting evaluation of {total_questions} questions...")
        
        for idx, row in self.data.iterrows():
            print(f"Processing question {idx + 1}/{total_questions} (ID: {row['QuestionID']})")
            
            # Create random mapping for this question
            model_mapping, reverse_mapping = self._create_random_mapping()
            
            # Format the question for the judge
            prompt, candidate_data = self._format_question_for_judge(row, model_mapping)
            
            # Query the judge
            judge_response = self._query_ollama(prompt)
            
            if not judge_response:
                print(f"Failed to get response for question {row['QuestionID']}")
                continue
            
            # Parse the ranking
            ranking = self._parse_judge_response(judge_response)
            
            # Store the full judge output
            judge_output = {
                'question_id': row['QuestionID'],
                'model_mapping': model_mapping,
                'reverse_mapping': reverse_mapping,
                'judge_response': judge_response,
                'ranking': ranking,
                'timestamp': datetime.now().isoformat()
            }
            self.judge_outputs.append(judge_output)
            
            # Create evaluation result
            eval_result = {
                'QuestionID': row['QuestionID'],
                'Judge_Reasoning': judge_response,
            }
            
            # Add ranking information
            for position in range(1, 5):
                if position in ranking:
                    letter = ranking[position]
                    model = reverse_mapping.get(letter, "Unknown")
                    eval_result[f'Position_{position}_Model'] = model
                    eval_result[f'Position_{position}_Letter'] = letter
                else:
                    eval_result[f'Position_{position}_Model'] = "Not ranked"
                    eval_result[f'Position_{position}_Letter'] = "Not ranked"
            
            self.evaluation_results.append(eval_result)
            
            # Add small delay to avoid overwhelming the API
            time.sleep(0.5)
        
        print("Evaluation completed!")
    
    def save_results(self):
        """Save all results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save judge outputs as JSON
        judge_output_file = f"judge_outputs_{timestamp}.json"
        with open(judge_output_file, 'w', encoding='utf-8') as f:
            json.dump(self.judge_outputs, f, indent=2, ensure_ascii=False)
        print(f"Judge outputs saved to: {judge_output_file}")
        
        # Save evaluation results as CSV
        eval_results_file = f"evaluation_results_{timestamp}.csv"
        eval_df = pd.DataFrame(self.evaluation_results)
        eval_df.to_csv(eval_results_file, index=False)
        print(f"Evaluation results saved to: {eval_results_file}")
        
        return judge_output_file, eval_results_file
    
    def generate_summary_statistics(self):
        """Generate summary statistics of the evaluation."""
        if not self.evaluation_results:
            print("No evaluation results to summarize.")
            return
        
        eval_df = pd.DataFrame(self.evaluation_results)
        
        # Count wins for each model
        model_stats = {}
        for model in self.model_names:
            model_stats[model] = {
                'first_place': 0,
                'second_place': 0,
                'third_place': 0,
                'fourth_place': 0
            }
        
        for _, row in eval_df.iterrows():
            for position in range(1, 5):
                model_col = f'Position_{position}_Model'
                if model_col in row and row[model_col] in model_stats:
                    position_names = ['first_place', 'second_place', 'third_place', 'fourth_place']
                    model_stats[row[model_col]][position_names[position-1]] += 1
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        for model, stats in model_stats.items():
            print(f"\n{model}:")
            print(f"  1st place: {stats['first_place']}")
            print(f"  2nd place: {stats['second_place']}")
            print(f"  3rd place: {stats['third_place']}")
            print(f"  4th place: {stats['fourth_place']}")
        
        return model_stats

def main():
    # Configuration
    SYSTEM_PROMPT_FILE = "../data/prompts/judge_gmat_cr.txt"
    CSV_FILE = "../data/NLP_analysis/critical_reasoning_analysis.csv"  # Replace with your actual CSV file name
    JUDGE_MODEL = "gemma3:1b"
    
    # Check if files exist
    if not os.path.exists(SYSTEM_PROMPT_FILE):
        print(f"Error: System prompt file '{SYSTEM_PROMPT_FILE}' not found.")
        return
    
    if not os.path.exists(CSV_FILE):
        print(f"Error: CSV file '{CSV_FILE}' not found.")
        return
    
    # Initialize the judge
    judge = GMATJudge(
        system_prompt_file=SYSTEM_PROMPT_FILE,
        csv_file=CSV_FILE,
        judge_model=JUDGE_MODEL
    )
    
    # Run evaluation
    try:
        judge.evaluate_all_questions()
        judge.save_results()
        judge.generate_summary_statistics()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        if judge.evaluation_results:
            print("Saving partial results...")
            judge.save_results()
    except Exception as e:
        print(f"An error occurred: {e}")
        if judge.evaluation_results:
            print("Saving partial results...")
            judge.save_results()

if __name__ == "__main__":
    main()
