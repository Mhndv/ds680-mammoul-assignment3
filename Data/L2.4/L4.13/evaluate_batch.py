"""
Biometric Privacy Evaluation Script - Batch Version
=====================================================
- Test Model: gpt-4o-mini 
- Judge Model: gpt-5-mini-2025-08-07

Usage: 
    python evaluate_batch.py                    # Run all prompts
    python evaluate_batch.py --limit 10         # Run first 10 prompts
    python evaluate_batch.py --start 50 --limit 25  # Run prompts 50-74
"""

import json
import os
import argparse
import asyncio
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

async_client = AsyncOpenAI(api_key=api_key)

# Config
TEST_MODEL = "gpt-4o-mini"
# Judge model options:
# - "gpt-5-mini-2025-08-07" (best performance, most expensive)
# - "gpt-5-nano-2025-08-07" (good performance, 5x cheaper - recommended for cost savings)
# - "gpt-4o-mini" (cheapest, may have slightly lower accuracy)
JUDGE_MODEL = "gpt-5-nano-2025-08-07"
JUDGE_PROMPT_FILE = "judge_prompt.txt"
PROMPTS_FILE = "prompt_set.json"

# Pricing per 1M tokens (as of Nov 2025 - UPDATE THESE AS NEEDED)
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5-mini-2025-08-07": {"input": 0.25, "output": 2.0},
    "gpt-5-nano-2025-08-07": {"input": 0.05, "output": 0.40},  # Much cheaper alternative
}


class CostTracker:
    """Track API usage and costs (thread-safe for async)."""
    
    def __init__(self):
        self.usage = {
            "test_model": {"input_tokens": 0, "output_tokens": 0},
            "judge_model": {"input_tokens": 0, "output_tokens": 0},
        }
        self.calls = {"test_model": 0, "judge_model": 0}
        self._lock = asyncio.Lock()
    
    async def add_test_usage(self, response):
        """Track usage from test model call."""
        async with self._lock:
            usage = response.usage
            self.usage["test_model"]["input_tokens"] += usage.prompt_tokens
            self.usage["test_model"]["output_tokens"] += usage.completion_tokens
            self.calls["test_model"] += 1
    
    async def add_judge_usage(self, response):
        """Track usage from judge model call."""
        async with self._lock:
            usage = response.usage
            self.usage["judge_model"]["input_tokens"] += usage.prompt_tokens
            self.usage["judge_model"]["output_tokens"] += usage.completion_tokens
            self.calls["judge_model"] += 1
    
    def get_cost(self, model: str, model_type: str) -> dict:
        """Calculate cost for a specific model."""
        pricing = PRICING.get(model, {"input": 0, "output": 0})
        input_cost = (self.usage[model_type]["input_tokens"] / 1_000_000) * pricing["input"]
        output_cost = (self.usage[model_type]["output_tokens"] / 1_000_000) * pricing["output"]
        return {
            "input_tokens": self.usage[model_type]["input_tokens"],
            "output_tokens": self.usage[model_type]["output_tokens"],
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost
        }
    
    def get_summary(self) -> dict:
        """Get full cost summary."""
        test_cost = self.get_cost(TEST_MODEL, "test_model")
        judge_cost = self.get_cost(JUDGE_MODEL, "judge_model")
        
        return {
            "test_model": {
                "model": TEST_MODEL,
                "calls": self.calls["test_model"],
                **test_cost
            },
            "judge_model": {
                "model": JUDGE_MODEL,
                "calls": self.calls["judge_model"],
                **judge_cost
            },
            "total": {
                "calls": self.calls["test_model"] + self.calls["judge_model"],
                "input_tokens": test_cost["input_tokens"] + judge_cost["input_tokens"],
                "output_tokens": test_cost["output_tokens"] + judge_cost["output_tokens"],
                "total_cost": test_cost["total_cost"] + judge_cost["total_cost"]
            }
        }
    
    def print_summary(self):
        """Print formatted cost summary."""
        summary = self.get_summary()
        
        print(f"\n{'='*60}")
        print("COST SUMMARY")
        print(f"{'='*60}")
        
        print(f"\n{TEST_MODEL} (Test Model):")
        print(f"  Calls: {summary['test_model']['calls']}")
        print(f"  Tokens: {summary['test_model']['input_tokens']:,} in / {summary['test_model']['output_tokens']:,} out")
        print(f"  Cost: ${summary['test_model']['total_cost']:.4f}")
        
        print(f"\n{JUDGE_MODEL} (Judge Model):")
        print(f"  Calls: {summary['judge_model']['calls']}")
        print(f"  Tokens: {summary['judge_model']['input_tokens']:,} in / {summary['judge_model']['output_tokens']:,} out")
        print(f"  Cost: ${summary['judge_model']['total_cost']:.4f}")
        
        print(f"\nTOTAL:")
        print(f"  API Calls: {summary['total']['calls']}")
        print(f"  Tokens: {summary['total']['input_tokens']:,} in / {summary['total']['output_tokens']:,} out")
        print(f"  Cost: ${summary['total']['total_cost']:.4f}")


# Global cost tracker
cost_tracker = CostTracker()


def load_judge_prompt(filepath: str) -> str:
    with open(filepath, "r") as f:
        return f.read()


def load_prompts(filepath: str) -> list[dict]:
    with open(filepath, "r") as f:
        data = json.load(f)
    return data["prompts"]


async def get_model_response(prompt: str) -> str:
    """Get response from model being evaluated."""
    response = await async_client.chat.completions.create(
        model=TEST_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024
    )
    await cost_tracker.add_test_usage(response)
    return response.choices[0].message.content


async def judge_response(test_prompt: str, model_response: str, judge_template: str, max_retries: int = 3) -> dict:
    """Use judge model to evaluate the response with retry logic."""
    filled_prompt = judge_template.replace("{test_prompt}", test_prompt).replace("{model_response}", model_response)
    
    last_error = None
    max_tokens = 4000  # Increased from 2000 to handle longer judge responses
    
    for attempt in range(max_retries):
        try:
            response = await async_client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": filled_prompt}],
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            await cost_tracker.add_judge_usage(response)
            
            raw_content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            
            if not raw_content or not raw_content.strip():
                error_msg = f"Judge model returned empty response (finish_reason: {finish_reason}, attempt: {attempt + 1}/{max_retries})"
                print(f"WARNING: {error_msg}")
                
                # If it's a length issue, try with more tokens
                if finish_reason == "length" and attempt < max_retries - 1:
                    max_tokens = min(max_tokens * 2, 8000)  # Double tokens, cap at 8000
                    print(f"Retrying with increased token limit ({max_tokens})...")
                    await asyncio.sleep(1)
                    continue
                
                # If it's a content filter or other issue, retry
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                
                raise ValueError(f"{error_msg}")
            
            try:
                return json.loads(raw_content)
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse JSON from judge model (attempt: {attempt + 1}/{max_retries})"
                print(f"ERROR: {error_msg}")
                print(f"Raw response:\n{raw_content}")
                print(f"Parse error: {e}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                
                raise ValueError(f"{error_msg}: {e}")
        
        except ValueError:
            # Re-raise ValueError (our custom errors) without retry
            raise
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                print(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
                await asyncio.sleep(2 ** attempt)
                continue
            raise
    
    # Should never reach here, but just in case
    if last_error:
        raise last_error
    raise ValueError("Judge model failed after all retries")


async def evaluate_single(prompt_data: dict, judge_template: str, semaphore: asyncio.Semaphore) -> dict:
    """Run evaluation for a single prompt."""
    async with semaphore:  # Limit concurrent requests
        prompt_id = prompt_data["id"]
        test_prompt = prompt_data["prompt"]
        
        # Get model response
        model_response = await get_model_response(test_prompt)
        
        # Judge the response
        evaluation = await judge_response(test_prompt, model_response, judge_template)
        
        return {
            "prompt_id": prompt_id,
            "category": prompt_data.get("category"),
            "subcategory": prompt_data.get("subcategory"),
            "test_model": TEST_MODEL,
            "judge_model": JUDGE_MODEL,
            "test_prompt": test_prompt,
            "model_response": model_response,
            "evaluation": evaluation
        }


async def run_batch(prompts: list[dict], judge_template: str, start: int = 0, limit: int = None, max_concurrent: int = 5):
    """Run evaluation on a batch of prompts asynchronously."""
    # Slice prompts based on start and limit
    end = start + limit if limit else len(prompts)
    batch = prompts[start:end]
    total = len(batch)
    
    print(f"\nRunning evaluation on {total} prompts (indices {start} to {start + total - 1})")
    print(f"Test Model: {TEST_MODEL}")
    print(f"Judge Model: {JUDGE_MODEL}")
    print(f"Max concurrent requests: {max_concurrent}")
    print(f"{'='*60}\n")
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create tasks for all evaluations
    tasks = []
    for prompt_data in batch:
        task = asyncio.create_task(
            evaluate_single(prompt_data, judge_template, semaphore)
        )
        tasks.append((prompt_data["id"], task))
    
    # Process results as they complete
    results = []
    errors = []
    completed = 0
    
    for prompt_id, task in tasks:
        try:
            result = await task
            completed += 1
            results.append(result)
            score = result["evaluation"]["score"]
            print(f"[{completed}/{total}] {prompt_id}... Score: {score}/5")
        except Exception as e:
            completed += 1
            error_msg = str(e)
            print(f"[{completed}/{total}] {prompt_id}... ERROR: {error_msg}")
            # Include more error context if available
            error_detail = {"prompt_id": prompt_id, "error": error_msg}
            if hasattr(e, "__cause__") and e.__cause__:
                error_detail["cause"] = str(e.__cause__)
            errors.append(error_detail)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE: {len(results)} successful, {len(errors)} errors")
    
    if results:
        scores = [r["evaluation"]["score"] for r in results]
        avg_score = sum(scores) / len(scores)
        print(f"Average Score: {avg_score:.2f}/5")
        
        # Score distribution
        dist = {i: scores.count(i) for i in range(1, 6)}
        print(f"Distribution: {dist}")
    
    return results, errors


async def main():
    parser = argparse.ArgumentParser(description="Batch evaluate biometric privacy responses")
    parser.add_argument("--start", type=int, default=0, help="Starting prompt index (default: 0)")
    parser.add_argument("--limit", type=int, default=None, help="Number of prompts to evaluate (default: all)")
    parser.add_argument("--output", type=str, default=None, help="Output filename (default: auto-generated)")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Max concurrent API requests (default: 5)")
    args = parser.parse_args()
    
    # Load resources
    print("Loading judge prompt...")
    judge_template = load_judge_prompt(JUDGE_PROMPT_FILE)
    
    print("Loading test prompts...")
    prompts = load_prompts(PROMPTS_FILE)
    print(f"Found {len(prompts)} prompts")
    
    # Run batch evaluation
    results, errors = await run_batch(
        prompts=prompts,
        judge_template=judge_template,
        start=args.start,
        limit=args.limit,
        max_concurrent=args.max_concurrent
    )
    
    # Print cost summary
    cost_tracker.print_summary()
    
    # Generate output filename
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"evaluation_results_{timestamp}.json"
    
    # Save results
    output_data = {
        "metadata": {
            "test_model": TEST_MODEL,
            "judge_model": JUDGE_MODEL,
            "timestamp": datetime.now().isoformat(),
            "prompts_evaluated": len(results),
            "errors": len(errors),
            "cost_summary": cost_tracker.get_summary()
        },
        "results": results,
        "errors": errors
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())