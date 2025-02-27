import pandas as pd
from typing import Dict, List, Optional, Any
from langsmith import Client
client = Client()

class DatasetManager:
    """Manages loading and accessing datasets from LangSmith."""
    
    def __init__(self):
        """Initialize the dataset manager with LangSmith client."""
        self.client = client
        self.dataset_names = ["OpenAI-SimpleQA", "WebSearch_QA_2025-02-09"]
        self._available_datasets = None

    def list_datasets(self) -> List[str]:
        """List available datasets.

        Returns:
            List of dataset names
        """
        if self._available_datasets is None:
            self._available_datasets = [dataset.name for dataset in self.client.list_datasets()]
        return self._available_datasets
    
    def load_data(
        self, 
        dataset_name: str, 
        split:str = "test" # Change later to "reg"
    ):
        """Load a dataset from LangSmith.
        
        Args:
            dataset_name: Name of the dataset in LangSmith
            sample_percentage: Percentage of the dataset to sample (0.001-100)

        """
        if dataset_name not in self.list_datasets():
            raise ValueError(
                f"Dataset '{dataset_name}' not found. "
                f"Available datasets: {self.list_datasets()}"
            )

        # Get dataset from LangSmith
        #if splits is not full use , splits=["input_split"]
        # Get examples from the dataset
        examples = self.client.list_examples(dataset_name=dataset_name, splits=[split])

        return examples
    
    def get_column_mapping(self, dataset_name: str) -> Dict[str, str]:
        """Get the column mapping for a dataset.
        
        Args:
            dataset_name: Name of the dataset in LangSmith
            
        Returns:
            Column mapping dictionary
        """
        # Since we're standardizing the columns in load_dataset,
        # we can return a fixed mapping
        return {
            "question": "question",
            "answer": "answer"
        }