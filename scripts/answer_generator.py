#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A simple script to ask a question to a local LLM using Ollama.

Takes the model name and the question as command-line arguments.
Requires the 'ollama' Python library and a running Ollama instance.

Usage:
  python ask_ollama.py --model <model_name> --question "Your question here"

Example:
  python ask_ollama.py --model phi3 --question "What is the capital of France?"
"""

import argparse
import sys
import ollama

# --- Configuration ---
# Define the system prompt to guide the LLM's behavior
SYSTEM_PROMPT = "You are a helpful and concise AI assistant. Answer the user's question directly."
# --- End Configuration ---

def ask_llm(model_name: str, question: str, system_prompt: str):
    """
    Sends a question to the specified Ollama model and returns the answer.

    Args:
        model_name: The name of the Ollama model to use (e.g., 'llama3', 'mistral').
        question: The user's question string.
        system_prompt: The system prompt string to guide the model.

    Returns:
        The content of the LLM's response string, or None if an error occurs.
    """
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': question},
    ]

    try:
        print(f"--- Sending request to model: {model_name} ---")
        # Use stream=False for a single complete response
        response = ollama.chat(model=model_name, messages=messages, stream=False)
        # print(f"Raw response: {response}") # Uncomment for debugging
        return response['message']['content']
    except ollama.ResponseError as e:
        print(f"\nError communicating with Ollama model '{model_name}':")
        print(f"  Status Code: {e.status_code}")
        print(f"  Error: {e.error}")
        if "model not found" in e.error:
            print(f"  Suggestion: Make sure you have pulled the model using 'ollama pull {model_name}'")
        elif "connection refused" in e.error.lower():
             print(f"  Suggestion: Make sure the Ollama application or server is running.")
        return None
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        return None

def main():
    """
    Parses command-line arguments and orchestrates the LLM query.
    """
    parser = argparse.ArgumentParser(
        description="Ask a question to a local LLM via Ollama.",
        formatter_class=argparse.RawTextHelpFormatter # To preserve formatting in help text
    )
    parser.add_argument(
        "--model",
        required=True,
        help="The name of the Ollama model to use (e.g., 'llama3', 'mistral')."
    )
    parser.add_argument(
        "--question",
        required=True,
        help="The question you want to ask the LLM."
    )

    args = parser.parse_args()

    print(f"Asking Model: '{args.model}'")
    print(f"Question: '{args.question}'")
    print("-" * 20)

    answer = ask_llm(args.model, args.question, SYSTEM_PROMPT)

    if answer:
        print("\n--- LLM Answer ---")
        print(answer)
        print("-" * 20)
    else:
        print("\nFailed to get an answer from the LLM.")
        sys.exit(1) # Exit with error status if no answer

if __name__ == "__main__":
    main()
