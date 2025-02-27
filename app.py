
from datetime import datetime
from typing import Dict, Any, Optional, List
import traceback
from flask import Flask, request, jsonify
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from evaluators import SearchRelevanceEvaluator, CorrectnessEvaluator
from utils.tavily_handler import TavilyHandler
from utils.dataset_manager import DatasetManager
from langsmith import Client
import asyncio
import threading
langsmith_client = Client()

app = Flask(__name__)

# Global lock to ensure only one evaluation runs at a time
evaluation_lock = threading.Lock()
current_evaluation = None

# Initialize the dataset manager
dataset_manager = DatasetManager()

class Environment(str, Enum):
    PROD = "prod"
    DEV = "dev"
    STAGING = "staging"

class SampleSize(str, Enum):
    TEST = "test"
    BASE = "reg"
    FULL = "base"

class EvaluationRequest(BaseModel):
    """Request model for evaluation API"""
    env: Environment = Field(
        Environment.PROD,
        description="Environment to run the evaluation in (prod/dev/staging)"
    )
    search_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Search parameters and filters for the evaluation"
    )
    dataset: List[str] = Field(
        ["OpenAI-SimpleQA"],
        description="List of Datasets identifier to use for evaluation"
    )
    sample_size: SampleSize = Field(
        SampleSize.TEST,
        description="Amount of dataset to process (base/full)"
    )
    description: Optional[str] = Field(
        None,
        description="Optional description of this evaluation run"
    )

    class Config:
        json_schema_extra = ({
            "example": {
                "env": "prod",
                "search_params": {
                    "depth": "advanced",
                    "include_answer": True,
                    "max_results": 10,
                    "domains": ["example.com"],
                    "time_range": "month"
                },
                "dataset": "qa_dataset_v1",
                "sample_size": "base",
                "description": "QA evaluation for product search improvements"
            }
        })


@app.route('/evaluate', methods=['POST'])
def evaluate():
    global current_evaluation

    # Check if an evaluation is already running
    if evaluation_lock.locked():
        return jsonify({
            "status": "error",
            "message": "An evaluation is already in progress",
            "current_job": current_evaluation
        }), 409

    try:
        # Parse and validate request using Pydantic
        data = request.json or {}
        eval_request = EvaluationRequest(**data)

        # Acquire the lock and start the evaluation in a background thread
        evaluation_lock.acquire()
        current_evaluation = {
            "search_params": eval_request.search_params,
            "dataset": eval_request.dataset,
            "sample_size": eval_request.sample_size,
            "start_time": datetime.now().isoformat(),
            "status": "running"
        }

        # Create async task and run it in the background
        async def run_async():
            await run_evaluation(
                eval_request.dataset,
                eval_request.sample_size,
                eval_request.description,
                eval_request.search_params,
                eval_request.env
            )

        threading.Thread(
            target=lambda: asyncio.run(run_async())
        ).start()

        return jsonify({
            "status": "success",
            "message": "Evaluation started",
            "job": current_evaluation
        })

    except Exception as e:
        if evaluation_lock.locked():
            evaluation_lock.release()
            current_evaluation = None
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/status', methods=['GET'])
def status():
    """Get the status of the current or last evaluation."""
    if current_evaluation:
        return jsonify({
            "status": "success",
            "evaluation": current_evaluation
        })
    else:
        return jsonify({
            "status": "success",
            "message": "No evaluation is currently running or has been run"
        })


@app.route('/datasets', methods=['GET'])
def list_datasets():
    """List all available datasets."""
    try:
        datasets = dataset_manager.list_datasets()
        dataset_info = []

        for name in datasets:
            config = dataset_manager.get_dataset_config(name)
            dataset_info.append({
                "name": config.name,
                "file_path": config.file_path,
                "description": config.description,
                "column_mapping": config.column_mapping,
                "metadata": config.metadata
            })

        return jsonify({
            "status": "success",
            "datasets": dataset_info
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


async def run_evaluation(
    datasets: List[str],
    sample_size: str,
    description: str,
    search_params: dict,
    env: Environment,
):
    """Run the evaluation in a background thread."""
    global current_evaluation
    metadata = {
        **search_params,
        "env": env
    }
    
    experiment_prefix = f"tavily-search-eval - {datetime.now().strftime('%Y-%m-%d %H:%M')} - env = {env.value}"
    
    # Initialize TavilyHandler with search params
    tavily_handler = TavilyHandler(
        env=env,
        search_params=search_params
    )

    # Initialize appropriate evaluators based on search_params
    if search_params.get("include_answer", False):
        evaluator = CorrectnessEvaluator()
        eval_function = evaluator.evaluate
    else:
        evaluator = SearchRelevanceEvaluator()
        eval_function = evaluator.evaluate

    try:
        for dataset in datasets:
            # Load the dataset and convert it to a list of dictionaries
            dataset_examples = list(dataset_manager.load_data(dataset, sample_size))
            
            # Define the async target function
            async def async_target(x):
                return await tavily_handler.search(x)

            await langsmith_client.aevaluate(
                async_target,
                data=dataset_examples,
                evaluators=[eval_function],  # Pass the evaluate method instead of the class instance
                description=description,
                metadata=metadata,
                experiment_prefix=experiment_prefix
            )
    finally:
        evaluation_lock.release()



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Start the Tavily Evaluator API server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind the server to')
    parser.add_argument('--debug', action='store_true', help='Run the server in debug mode')

    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=args.debug)