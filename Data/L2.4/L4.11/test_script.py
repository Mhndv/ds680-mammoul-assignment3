"""
Biometric Privacy Evaluation Script
====================================
- Test Model: gpt-4o-mini 
- Judge Model: gpt-5.1-mini

Usage: python evaluate_biometric_privacy.py
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file in the same directory as this script
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file. Please add OPENAI_API_KEY=your_key_here to the .env file.")

client = OpenAI(api_key=api_key)

# Config
TEST_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-5-mini-2025-08-07"
JUDGE_PROMPT_FILE = "judge_prompt.txt"
PROMPTS_FILE = "prompt_set.json"
OUTPUT_FILE = "evaluation_results.json"


def load_judge_prompt(filepath: str) -> str:
    with open(filepath, "r") as f:
        return f.read()


def load_prompts(filepath: str) -> list[dict]:
    with open(filepath, "r") as f:
        data = json.load(f)
    return data["prompts"]


def get_model_response(prompt: str) -> str:
    """Get response from model being evaluated."""
    response = client.chat.completions.create(
        model=TEST_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024
    )
    return response.choices[0].message.content


def judge_response(test_prompt: str, model_response: str, judge_template: str) -> dict:
    """Use judge model to evaluate the response."""
    # Use replace() instead of format() to avoid issues with JSON braces in template
    filled_prompt = judge_template.replace("{test_prompt}", test_prompt).replace("{model_response}", model_response)
    
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": filled_prompt}],
        max_completion_tokens=800,
        response_format={"type": "json_object"}
    )
    
    raw_content = response.choices[0].message.content
    
    # Debug: print raw response if there's an issue
    if not raw_content or not raw_content.strip():
        print(f"ERROR: Empty response from judge model")
        print(f"Finish reason: {response.choices[0].finish_reason}")
        raise ValueError("Judge model returned empty response")
    
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON from judge model")
        print(f"Raw response:\n{raw_content}")
        print(f"Parse error: {e}")
        raise


def main():
    # Load files
    judge_template = load_judge_prompt(JUDGE_PROMPT_FILE)
    prompts = load_prompts(PROMPTS_FILE)
    
    # Get first prompt
    prompt_data = prompts[0]
    test_prompt = prompt_data["prompt"]
    
    print(f"Prompt ID: {prompt_data['id']}")
    print(f"Test Prompt: {test_prompt}\n")
    
    # Step 1: Get model response
    print(f"Getting response from {TEST_MODEL}...")
    model_response = get_model_response(test_prompt)
    print(f"Response:\n{model_response}\n")
    
    # Step 2: Judge the response
    print(f"Judging with {JUDGE_MODEL}...")
    evaluation = judge_response(test_prompt, model_response, judge_template)
    
    # Display results
    print(f"\n{'='*50}")
    print(f"SCORE: {evaluation['score']}/5")
    print(f"{'='*50}")
    print(f"Justification: {evaluation['justification']}")
    print(f"Strengths: {evaluation.get('strengths', [])}")
    print(f"Weaknesses: {evaluation.get('weaknesses', [])}")
    print(f"Flags: {json.dumps(evaluation.get('flags', {}), indent=2)}")
    
    # Save full result
    result = {
        "prompt_id": prompt_data["id"],
        "category": prompt_data.get("category"),
        "test_model": TEST_MODEL,
        "judge_model": JUDGE_MODEL,
        "test_prompt": test_prompt,
        "model_response": model_response,
        "evaluation": evaluation
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()