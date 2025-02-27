import asyncio
import hashlib
import json
import os
import time
from asyncio import Semaphore
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
from tqdm import tqdm

from utils.report.result_manager import ResultManager
from evaluators.correctness_evaluator import grade_sample
from tavily import TavilyClient

# Concurrency settings
MAX_CONCURRENT_TASKS = 5  # Max concurrent questions per policy


tavily_api_key = os.getenv("TAVILY_API_KEY")

def get_data():
    """Load the SimpleQA test set."""
    df = pd.read_csv("simple_qa_test_set.csv")
    return df

def sample_questions(n: int = None):
    """Sample n questions from the test set."""
    df = get_data()
    # If n is None, return the complete dataset
    return df.sample(n) if n is not None else df

def generate_question_id(question: str) -> str:
    """Generate a unique ID for a question."""
    return hashlib.md5(question.encode()).hexdigest()

async def run_tavily_policy(
    question: str, 
    search_depth: str = "advanced", 
    include_answer: bool = True,
    max_results: int = 10,
    search_filter: Dict[str, Any] = None
) -> Tuple[str, None]:
    """Run tavily policy in a thread to avoid blocking."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        search_params = {
            "query": question,
            "search_depth": search_depth,
            "include_answer": include_answer,
            "max_results": max_results
        }
        
        # Add search filter if provided
        if search_filter:
            search_params.update(search_filter)
            
        result = await loop.run_in_executor(
            pool, 
            lambda: TavilyClient(api_key=tavily_api_key).search(**search_params)['answer']
        )
        return result, None

async def run_policy_async(
    question: str, 
    policy_type: str = "tavily_advanced",
    include_answer: bool = True,
    max_results: int = 10,
    search_filter: Dict[str, Any] = None
) -> Tuple[str, Optional[Any]]:
    """Async version of run_policy."""
    # Parse policy type to extract parameters
    parts = policy_type.split("_")
    
    if parts[0] != "tavily":
        raise ValueError(f"Unknown policy type: {policy_type}. Only 'tavily' is supported.")
    
    # Extract search depth from policy type
    search_depth = parts[1].split("=")[1] if "=" in parts[1] else parts[1]
    
    # Extract include_answer if specified in policy type
    if len(parts) > 2 and parts[2].startswith("ia="):
        include_answer_str = parts[2].split("=")[1]
        include_answer = include_answer_str.lower() == "true"
    
    return await run_tavily_policy(
        question, 
        search_depth=search_depth, 
        include_answer=include_answer,
        max_results=max_results,
        search_filter=search_filter
    )

def calculate_f_score(metrics: Dict[str, float]) -> float:
    """Calculate F score from metrics."""
    if (metrics["accuracy_given_attempted"] + metrics["is_correct"]) > 0:
        return (
            2 * metrics["accuracy_given_attempted"] * metrics["is_correct"]
            / (metrics["accuracy_given_attempted"] + metrics["is_correct"])
        )
    return 0.0

def calculate_metrics(results: list) -> Dict[str, float]:
    """Calculate aggregate metrics from results."""
    total = len(results)
    if not total:
        return {
            "is_correct": 0,
            "is_incorrect": 0,
            "is_not_attempted": 0,
            "is_given_attempted": 0,
            "accuracy_given_attempted": 0,
            "avg_latency": 0,
            "f_score": 0,
            "total_samples": 0
        }
    
    counts = {"A": 0, "B": 0, "C": 0}
    latencies = []
    
    for grade, latency in results:
        counts[grade] = counts.get(grade, 0) + 1
        latencies.append(latency)
    
    metrics = {
        "is_correct": counts["A"] / total,
        "is_incorrect": counts["B"] / total,
        "is_not_attempted": counts["C"] / total,
        "avg_latency": sum(latencies) / len(latencies),
        "total_samples": total
    }
    
    metrics["is_given_attempted"] = metrics["is_correct"] + metrics["is_incorrect"]
    metrics["accuracy_given_attempted"] = (
        metrics["is_correct"] / metrics["is_given_attempted"]
        if metrics["is_given_attempted"] > 0
        else 0
    )
    
    metrics["f_score"] = calculate_f_score(metrics)
    return metrics

def print_metrics(metrics: Dict[str, float], policy_name: str = None):
    """Print metrics in a formatted way."""
    if policy_name:
        print(f"\nMETRICS FOR {policy_name.upper()}")
    print("##################")
    print(f"Accuracy Given Attempted: {metrics['accuracy_given_attempted']:.3f}")
    print(f"F Score: {metrics['f_score']:.3f}")
    print(f"Average Latency: {metrics['avg_latency']:.2f} seconds")
    print(f"Correct: {metrics['is_correct']:.3f}")
    print(f"Incorrect: {metrics['is_incorrect']:.3f}")
    print(f"Not Attempted: {metrics['is_not_attempted']:.3f}")
    print(f"Total Samples: {metrics['total_samples']}")


async def evaluate_questions_async(
        df,
        search_depth: str = "advanced",
        include_answer: bool = True,
        max_results: int = 10,
        search_filter: Dict[str, Any] = None,
        run_dir: str = "results",
        column_mapping: Dict[str, str] = None,
        config_name: str = None
) -> List[Tuple[str, float]]:
    """Evaluate questions and return results.

    Args:
        df: DataFrame containing questions and answers
        search_depth: Tavily search depth ("advanced" or "basic")
        include_answer: Whether to include answer in Tavily response
        max_results: Maximum number of search results to return
        search_filter: Optional filters for the search
        run_dir: Directory to store results
        column_mapping: Mapping of required columns to dataset columns
                        e.g. {'question': 'problem', 'answer': 'correct_answer'}
    """
    sem = Semaphore(MAX_CONCURRENT_TASKS)
    results = []

    # Default column mapping if not provided
    if column_mapping is None:
        column_mapping = {
            'question': 'question',
            'answer': 'answer'
        }

    # Validate that required columns exist in the dataframe
    for required_col, dataset_col in column_mapping.items():
        if dataset_col not in df.columns:
            raise ValueError(
                f"Required column '{dataset_col}' not found in dataset. Available columns: {list(df.columns)}")


    # Initialize result manager with the run directory
    result_manager = ResultManager(config_name, run_dir=run_dir)

    # Load existing results if any
    if result_manager.results_file.exists():
        with open(result_manager.results_file, 'r') as f:
            existing_results = [json.loads(line) for line in f]
            results.extend((r['grade'], r['latency']) for r in existing_results)
            print(f"\nLoaded {len(existing_results)} existing results for {config_name}")

    async def process_question(q_id: str, problem: str, answer: str) -> Optional[Tuple[str, float]]:
        # Skip if already processed
        cache_key = f"{q_id}_{config_name}"
        if cache_key in result_manager.processed_indices:
            return None

        async with sem:
            retries = 3
            for attempt in range(retries):
                start_time = time.time()
                try:
                    # Call Tavily API with direct parameters
                    predicted_answer, _ = await asyncio.wait_for(
                        run_tavily_policy(
                            problem,
                            search_depth=search_depth,
                            include_answer=include_answer,
                            max_results=max_results,
                            search_filter=search_filter
                        ),
                        timeout=300
                    )
                    latency = time.time() - start_time
                    grade_letter = grade_sample(problem, answer, predicted_answer)

                    result = {
                        'question_id': q_id,
                        'config': config_name,
                        'grade': grade_letter,
                        'question': problem,
                        'predicted': predicted_answer,
                        'correct': answer,
                        'latency': latency,
                        'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                        'attempt': attempt + 1,
                        'search_depth': search_depth,
                        'include_answer': include_answer,
                        'max_results': max_results,
                        'search_filter': search_filter
                    }
                    result_manager.save_result(result)
                    result_manager.save_checkpoint(cache_key)

                    return (grade_letter, latency)

                except asyncio.TimeoutError:
                    print(f"Timeout on question {q_id}, attempt {attempt + 1}/{retries}")
                    if attempt == retries - 1:
                        error_msg = "Maximum retries exceeded due to timeouts"
                        break
                    await asyncio.sleep(5 * (attempt + 1))  # Exponential backoff

                except Exception as e:
                    print(f"Error on question {q_id}, attempt {attempt + 1}/{retries}: {str(e)}")
                    if attempt == retries - 1:
                        error_msg = f"Maximum retries exceeded: {str(e)}"
                        break
                    await asyncio.sleep(5 * (attempt + 1))  # Exponential backoff

            # If we get here, all retries failed
            result = {
                'question_id': q_id,
                'config': config_name,
                'grade': 'X',  # X for not attempted
                'question': problem,
                'predicted': f"ERROR: {error_msg}",
                'correct': answer,
                'latency': time.time() - start_time,
                'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                'attempt': attempt + 1,
                'search_depth': search_depth,
                'include_answer': include_answer,
                'max_results': max_results,
                'search_filter': search_filter
            }
            result_manager.save_result(result)
            result_manager.save_checkpoint(cache_key)

            return ('X', 0.0)  # X for not attempted

    # Process all questions
    tasks = []
    for _, row in df.iterrows():
        question = row[column_mapping['question']]
        answer = row[column_mapping['answer']]
        q_id = generate_question_id(question)
        tasks.append(process_question(q_id, question, answer))

    # Wait for all tasks to complete
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Evaluating {config_name}"):
        result = await task
        if result:
            results.append(result)

    return results

async def compare_policies_async(
    policy1: str, 
    policy2: str, 
    num_samples: int, 
    run_dir: str = "results",
    run_id: str = None,
    include_answer: bool = True,
    max_results: int = 10,
    search_filter: Dict[str, Any] = None
):
    """Compare two policies on the same set of questions."""
    questions_df = sample_questions(num_samples)
    
    print(f"\nEvaluating {policy1}...")
    results1, summary_path1 = await evaluate_questions_async(
        questions_df, 
        policy1, 
        run_dir=run_dir,
        run_id=run_id,
        include_answer=include_answer,
        max_results=max_results,
        search_filter=search_filter
    )
    
    print(f"\nEvaluating {policy2}...")
    results2, summary_path2 = await evaluate_questions_async(
        questions_df, 
        policy2, 
        run_dir=run_dir,
        run_id=run_id,
        include_answer=include_answer,
        max_results=max_results,
        search_filter=search_filter
    )
    
    metrics1 = calculate_metrics(results1)
    metrics2 = calculate_metrics(results2)
    
    print("\nComparison Results:")
    print("=" * 50)
    print(f"{policy1} vs {policy2}")
    print("-" * 50)
    print(f"Accuracy: {metrics1['accuracy_given_attempted']:.3f} vs {metrics2['accuracy_given_attempted']:.3f}")
    print(f"F Score: {metrics1['f_score']:.3f} vs {metrics2['f_score']:.3f}")
    print(f"Latency: {metrics1['avg_latency']:.2f}s vs {metrics2['avg_latency']:.2f}s")
    
    return {
        "policy1": {
            "name": policy1,
            "metrics": metrics1,
            "summary_path": summary_path1
        },
        "policy2": {
            "name": policy2,
            "metrics": metrics2,
            "summary_path": summary_path2
        }
    }

def analyze_results(results_file: Path):
    """Analyze saved results."""
    df = ResultManager.load_results(results_file)
    
    # Basic analysis
    grade_counts = df['grade'].value_counts()
    avg_latency = df['latency'].mean()
    
    print("\nResults Analysis")
    print("=" * 50)
    print(f"Total questions: {len(df)}")
    print("\nGrade Distribution:")
    for grade, count in grade_counts.items():
        print(f"Grade {grade}: {count} ({count/len(df)*100:.2f}%)")
    print(f"\nAverage latency: {avg_latency:.2f}s")
    
    return df