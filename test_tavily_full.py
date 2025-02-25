import asyncio
import hashlib
import json
import os
import sys
import time
import traceback
from asyncio import Semaphore
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from grader import grade_sample
from tavily import TavilyClient

# Load environment variables
load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Concurrency settings
MAX_CONCURRENT_TASKS = 5

def get_data():
    """Load the SimpleQA test set."""
    df = pd.read_csv("simple_qa_test_set.csv")
    return df

def sample_questions(n: int = None):
    """Sample n questions from the dataset."""
    df = get_data()
    return df.sample(n) if n is not None else df

def generate_question_id(question: str) -> str:
    """Generate a unique, deterministic ID for a question."""
    return hashlib.sha256(question.encode()).hexdigest()[:16]

async def run_tavily_policy(question: str, search_depth: str, include_answer: bool) -> Tuple[str, None]:
    """Run tavily policy in a thread to avoid blocking."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool, 
            lambda: TavilyClient(api_key=tavily_api_key).search(
                question, 
                search_depth=search_depth, 
                include_answer=include_answer
            )['answer']
        )
        return result, None

class ResultManager:
    """Manages saving and loading of evaluation results."""
    
    def __init__(self, policy_type: str, run_dir: str = "results"):
        self.policy_type = policy_type
        self.results_dir = Path(run_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Use policy type for file names
        self.results_file = self.results_dir / f"{policy_type}_results.jsonl"
        self.summary_file = self.results_dir / f"{policy_type}_summary.json"
        self.checkpoint_file = self.results_dir / f"{policy_type}_checkpoint.json"
        
        # Track processed indices
        self.processed_indices = self.load_checkpoint()
    
    def load_checkpoint(self) -> set:
        """Load checkpoint of processed indices if exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return set(json.load(f))
        return set()
    
    def save_checkpoint(self, cache_key: str):
        """Save checkpoint of processed cache_key."""
        self.processed_indices.add(cache_key)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(list(self.processed_indices), f)
    
    def save_result(self, result: dict):
        """Save individual result in JSONL format."""
        with open(self.results_file, 'a') as f:
            f.write(json.dumps(result) + '\n')
    
    def save_summary(self, metrics: dict):
        """Save summary metrics."""
        with open(self.summary_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    @staticmethod
    def load_results(results_file: Path) -> pd.DataFrame:
        """Load results from a JSONL file into a DataFrame."""
        results = []
        with open(results_file, 'r') as f:
            for line in f:
                results.append(json.loads(line))
        return pd.DataFrame(results)

def calculate_metrics(results: List[Tuple[str, float]]) -> Dict[str, float]:
    """Calculate metrics from results."""
    if not results:
        return {
            'is_correct': 0.0,
            'is_incorrect': 0.0,
            'is_not_attempted': 0.0,
            'is_given_attempted': 0.0,
            'accuracy_given_attempted': 0.0,
            'f_score': 0.0,
            'avg_latency': 0.0
        }
    
    total = len(results)
    grades = {'A': 0, 'B': 0, 'C': 0}
    total_latency = 0.0
    
    for grade, latency in results:
        grades[grade] += 1
        if grade != 'C':  # Only count latency for attempted questions
            total_latency += latency
    
    is_correct = grades['A'] / total
    is_incorrect = grades['B'] / total
    is_not_attempted = grades['C'] / total
    is_given_attempted = is_correct + is_incorrect
    
    # Avoid division by zero
    accuracy_given_attempted = (
        grades['A'] / (grades['A'] + grades['B']) 
        if (grades['A'] + grades['B']) > 0 
        else 0.0
    )
    
    # Calculate F-score
    if (accuracy_given_attempted + is_correct) > 0:
        f_score = (
            2 * accuracy_given_attempted * is_correct
            / (accuracy_given_attempted + is_correct)
        )
    else:
        f_score = 0.0
    
    # Calculate average latency for attempted questions
    avg_latency = (
        total_latency / (grades['A'] + grades['B'])
        if (grades['A'] + grades['B']) > 0
        else 0.0
    )
    
    return {
        'is_correct': is_correct,
        'is_incorrect': is_incorrect,
        'is_not_attempted': is_not_attempted,
        'is_given_attempted': is_given_attempted,
        'accuracy_given_attempted': accuracy_given_attempted,
        'f_score': f_score,
        'avg_latency': avg_latency,
        'total_samples': total,
        'correct_count': grades['A'],
        'incorrect_count': grades['B'],
        'not_attempted_count': grades['C']
    }

async def evaluate_questions_async(
    questions_df: pd.DataFrame, 
    search_depth: str, 
    include_answer: bool,
    run_dir: str = "results"
) -> List[Tuple[str, float]]:
    """Evaluate questions and return results."""
    sem = Semaphore(MAX_CONCURRENT_TASKS)
    results = []
    
    # Create a unique config name
    config_name = f"tavily_{search_depth}_{include_answer}"
    
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
                    predicted_answer, _ = await asyncio.wait_for(
                        run_tavily_policy(problem, search_depth, include_answer),
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
                        'include_answer': include_answer
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
                        error_msg = str(e)
                        break
                    await asyncio.sleep(5 * (attempt + 1))
            
            # If all retries failed, save error result
            error_result = {
                'question_id': q_id,
                'config': config_name,
                'grade': 'C',
                'question': problem,
                'predicted': f"ERROR: {error_msg}",
                'correct': answer,
                'latency': time.time() - start_time,
                'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                'error': error_msg,
                'attempts': retries,
                'search_depth': search_depth,
                'include_answer': include_answer
            }
            result_manager.save_result(error_result)
            result_manager.save_checkpoint(cache_key)
            return ("C", 0.0)

    # Filter out already processed questions
    question_ids = [generate_question_id(q) for q in questions_df["problem"]]
    remaining_questions = [
        (q_id, problem, answer)
        for q_id, problem, answer in zip(question_ids, questions_df["problem"], questions_df["answer"])
        if f"{q_id}_{config_name}" not in result_manager.processed_indices
    ]
    
    print(f"\nTotal questions: {len(questions_df)}")
    print(f"Already processed: {len(questions_df) - len(remaining_questions)}")
    print(f"Remaining to process: {len(remaining_questions)}")
    
    tasks = [
        process_question(q_id, problem, answer)
        for q_id, problem, answer in remaining_questions
    ]
    
    if tasks:  # Only process if there are remaining tasks
        with tqdm(total=len(tasks), desc=f"Evaluating {config_name}") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result is not None:  # Skip None results (already processed)
                    grade, latency = result
                    results.append((grade, latency))
                    pbar.set_postfix({"Last Grade": grade, "Latency": f"{latency:.2f}s"})
                pbar.update(1)
                
                # Periodically show summary and update summary file (every 1000 questions)
                if pbar.n % 1000 == 0 and pbar.n > 0:
                    current_metrics = calculate_metrics(results)
                    print(f"\nInterim metrics after {pbar.n} questions:")
                    print(f"Accuracy: {current_metrics['accuracy_given_attempted']:.3f}")
                    print(f"Average latency: {current_metrics['avg_latency']:.2f}s")
                    # Update summary file with current metrics
                    result_manager.save_summary(current_metrics)
    
    # Save summary metrics using all results (existing + new)
    metrics = calculate_metrics(results)
    result_manager.save_summary(metrics)
    
    return results

def print_metrics(metrics: Dict[str, float], config_name: str = None):
    """Print metrics in a formatted way."""
    if config_name:
        print(f"\nMETRICS FOR {config_name}")
    print("##################")
    print(f"Accuracy Given Attempted: {metrics['accuracy_given_attempted']:.3f}")
    print(f"F Score: {metrics['f_score']:.3f}")
    print(f"Average Latency: {metrics['avg_latency']:.2f} seconds")
    print(f"Correct: {metrics['is_correct']:.3f}")
    print(f"Incorrect: {metrics['is_incorrect']:.3f}")
    print(f"Not Attempted: {metrics['is_not_attempted']:.3f}")

def analyze_by_metadata(results_file, metadata_field):
    """Analyze results by a specific metadata field (topic or answer_type)."""
    # Load results
    with open(results_file, 'r') as f:
        results = [json.loads(line) for line in f]
    
    # Load original questions with metadata
    df = pd.read_csv("simple_qa_test_set.csv")
    
    # Extract metadata from the string representation
    df['metadata_dict'] = df['metadata'].apply(lambda x: eval(x))
    df[metadata_field] = df['metadata_dict'].apply(lambda x: x.get(metadata_field, 'Unknown'))
    
    # Create a mapping from question to metadata
    question_to_metadata = dict(zip(df['problem'], df[metadata_field]))
    
    # Group results by metadata field
    grouped_results = {}
    for result in results:
        question = result.get('question', '')
        metadata_value = question_to_metadata.get(question, 'Unknown')
        
        if metadata_value not in grouped_results:
            grouped_results[metadata_value] = []
        
        grouped_results[metadata_value].append(result)
    
    # Calculate metrics for each group
    metrics_by_group = {}
    for group, group_results in grouped_results.items():
        total = len(group_results)
        grades = {"A": 0, "B": 0, "C": 0}
        total_latency = 0
        
        for result in group_results:
            grade = result.get('grade', 'C')
            grades[grade] += 1
            latency = result.get('latency', 0)
            if grade != 'C':  # Only count latency for attempted questions
                total_latency += latency
        
        # Calculate metrics
        metrics = {
            'total_samples': total,
            'correct_count': grades['A'],
            'incorrect_count': grades['B'],
            'not_attempted_count': grades['C'],
            'is_correct': grades['A'] / total if total > 0 else 0,
            'is_incorrect': grades['B'] / total if total > 0 else 0,
            'is_not_attempted': grades['C'] / total if total > 0 else 0,
            'avg_latency': total_latency / (total - grades['C']) if (total - grades['C']) > 0 else 0
        }
        
        # Calculate additional metrics
        metrics['is_given_attempted'] = metrics['is_correct'] + metrics['is_incorrect']
        metrics['accuracy_given_attempted'] = (
            metrics['is_correct'] / metrics['is_given_attempted']
            if metrics['is_given_attempted'] > 0
            else 0
        )
        
        # Calculate F-score
        if (metrics['accuracy_given_attempted'] + metrics['is_correct']) > 0:
            metrics['f_score'] = (
                2 * metrics['accuracy_given_attempted'] * metrics['is_correct']
                / (metrics['accuracy_given_attempted'] + metrics['is_correct'])
            )
        else:
            metrics['f_score'] = 0
        
        metrics_by_group[group] = metrics
    
    return metrics_by_group

def print_metadata_analysis(metrics_by_group, metadata_field, file=None):
    """Print analysis by metadata field in a formatted table."""
    # Sort groups by f_score (descending)
    sorted_groups = sorted(metrics_by_group.items(), 
                          key=lambda x: x[1]['f_score'], 
                          reverse=True)
    
    # Print header
    header = f"\nPerformance Analysis by {metadata_field}"
    print(header, file=file)
    print("=" * len(header), file=file)
    
    # Table headers
    headers = ["Category", "Count", "F-score", "Accuracy", "Correct", "Incorrect", "Not Attempted", "Avg Latency"]
    widths = [25, 8, 10, 10, 10, 10, 15, 12]
    
    # Print header row
    header_row = ""
    for h, w in zip(headers, widths):
        header_row += f"{h:{w}}"
    print(header_row, file=file)
    print("-" * sum(widths), file=file)
    
    # Print each group's metrics
    for group, metrics in sorted_groups:
        row = f"{str(group)[:23]:{widths[0]}}"
        row += f"{metrics['total_samples']:{widths[1]}}"
        row += f"{metrics['f_score']:.3f}{'':{widths[2]-5}}"
        row += f"{metrics['accuracy_given_attempted']:.3f}{'':{widths[3]-5}}"
        row += f"{metrics['is_correct']:.3f}{'':{widths[4]-5}}"
        row += f"{metrics['is_incorrect']:.3f}{'':{widths[5]-5}}"
        row += f"{metrics['is_not_attempted']:.3f}{'':{widths[6]-5}}"
        row += f"{metrics['avg_latency']:.2f}s{'':{widths[7]-6}}"
        print(row, file=file)
    
    # Print overall stats
    print("-" * sum(widths), file=file)
    total_samples = sum(m['total_samples'] for _, m in metrics_by_group.items())
    print(f"Total samples: {total_samples}", file=file)

async def main():
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Test Tavily API with different configurations')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to test (default: all)')
    args = parser.parse_args()
    
    # Create a unique run directory with timestamp
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = f"results/run_{run_timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"Results will be saved to: {run_dir}")
    
    # Define all combinations to test
    combinations = [
        ("advanced", "advanced"),  # search_depth, include_answer
        ("advanced", "basic"),
        ("basic", "advanced"),
        ("basic", "basic")
    ]
    
    questions_df = sample_questions(args.num_samples)
    all_metrics = {}
    
    for search_depth, include_answer in combinations:
        config_name = f"tavily_{search_depth}_{include_answer}"
        print(f"\nEvaluating Tavily with search_depth='{search_depth}', include_answer='{include_answer}'")
        
        try:
            # Pass the run directory to evaluate_questions_async
            results = await evaluate_questions_async(questions_df, search_depth, include_answer, run_dir=run_dir)
            metrics = calculate_metrics(results)
            all_metrics[config_name] = metrics
            print_metrics(metrics, config_name)
            
            # Save metrics to the run directory
            with open(f"{run_dir}/{config_name}_summary.json", 'w') as f:
                json.dump(metrics, f, indent=2)
                
            # Analyze by topic and answer_type
            results_file = f"{run_dir}/{config_name}_results.jsonl"
            if os.path.exists(results_file):
                # Analyze by topic
                topic_metrics = analyze_by_metadata(results_file, 'topic')
                with open(f"{run_dir}/{config_name}_topic_analysis.txt", 'w') as f:
                    print_metadata_analysis(topic_metrics, 'Topic', file=f)
                print_metadata_analysis(topic_metrics, 'Topic')
                
                # Analyze by answer_type
                answer_type_metrics = analyze_by_metadata(results_file, 'answer_type')
                with open(f"{run_dir}/{config_name}_answer_type_analysis.txt", 'w') as f:
                    print_metadata_analysis(answer_type_metrics, 'Answer Type', file=f)
                print_metadata_analysis(answer_type_metrics, 'Answer Type')
                
        except Exception as e:
            print(f"Error evaluating {config_name}: {str(e)}")
            traceback.print_exc()
    
    # Save comparison summary to the run directory
    with open(f"{run_dir}/comparison_summary.txt", 'w') as f:
        f.write("COMPARISON SUMMARY\n")
        f.write("=" * 120 + "\n")
        
        # Create headers for the table with sd/ia format
        column_width = 20
        metric_width = 25
        
        # Format header row
        header_row = f"{'Metric':{metric_width}}"
        for sd in ["advanced", "basic"]:
            for ia in ["advanced", "basic"]:
                header_row += f" | {f'sd={sd}_ia={ia}':{column_width}}"
        
        f.write(f"{header_row}\n")
        f.write("-" * 120 + "\n")
        
        metrics_to_show = [
            ('f_score', 'F-score'),
            ('accuracy_given_attempted', 'Accuracy (attempted)'),
            ('is_correct', 'Correct rate'),
            ('is_incorrect', 'Incorrect rate'),
            ('is_not_attempted', 'Not attempted rate'),
            ('avg_latency', 'Average latency (sec)')
        ]
        
        for metric_key, metric_name in metrics_to_show:
            row = f"{metric_name:{metric_width}}"
            for sd in ["advanced", "basic"]:
                for ia in ["advanced", "basic"]:
                    config = f"tavily_{sd}_{ia}"
                    if config in all_metrics:
                        if metric_key == 'avg_latency':
                            value = f"{all_metrics[config][metric_key]:.2f}"
                        else:
                            value = f"{all_metrics[config][metric_key]:.3f}"
                    else:
                        value = "N/A"
                    row += f" | {value:{column_width}}"
            f.write(f"{row}\n")
    
    # Print comparison table to console
    print("\nCOMPARISON SUMMARY")
    print("=" * 120)
    
    # Print header row
    header_row = f"{'Metric':{metric_width}}"
    for sd in ["advanced", "basic"]:
        for ia in ["advanced", "basic"]:
            header_row += f" | {f'sd={sd}_ia={ia}':{column_width}}"
    
    print(header_row)
    print("-" * 120)
    
    for metric_key, metric_name in metrics_to_show:
        row = f"{metric_name:{metric_width}}"
        for sd in ["advanced", "basic"]:
            for ia in ["advanced", "basic"]:
                config = f"tavily_{sd}_{ia}"
                if config in all_metrics:
                    if metric_key == 'avg_latency':
                        value = f"{all_metrics[config][metric_key]:.2f}"
                    else:
                        value = f"{all_metrics[config][metric_key]:.3f}"
                else:
                    value = "N/A"
                row += f" | {value:{column_width}}"
        print(row)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Progress has been saved.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)