from typing_extensions import Annotated
from langchain_openai import ChatOpenAI
from langsmith import traceable
from pydantic import BaseModel
from dataclasses import dataclass
from typing import Dict, Any, List
from dotenv import load_dotenv
from langsmith.evaluation.evaluator import EvaluationResult
load_dotenv()


class SearchRelevanceGrade(BaseModel):
    """Grade schema for search relevance evaluation"""
    explanation: Annotated[str, "Explain your reasoning for the score"]
    score: Annotated[int, "Relevance score (0-5) based on completeness"]


@dataclass
class SearchRelevanceConfig:
    """Configuration for search relevance evaluations"""
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0


class SearchRelevanceEvaluator:
    """Evaluator for assessing web search result relevance"""

    # Grade criteria instructions
    EVALUATION_PROMPT = """You are an expert search evaluator.

    You will be given:
    - A SEARCH QUERY
    - A set of SEARCH RESULTS (title, snippet, and URL)
    - A REFERENCE ANSWER (the ideal response to the query)

    ### Your task:
    Assign a relevance **score (0-5)** based on how well the search results provide the necessary information to answer the query.

    ### Scoring Criteria:
    - **5**: All necessary information is present to fully answer the query.
    - **4**: Most of the necessary information is available, but some minor details are missing.
    - **3**: Partial information is present, but key aspects of the query are missing.
    - **2**: Very little relevant information is present.
    - **1**: The search results are mostly irrelevant, with only slight relation to the query.
    - **0**: None of the search results contain relevant information.

    ### Guidelines:
    - Consider **title and snippet** when assessing relevance.
    - URLs may provide context but should not heavily influence the score.
    - Compare the search results against the **reference answer** to determine completeness.
    - Provide a step-by-step explanation of your reasoning.

    ### Example Format:
    Score: X
    Explanation: <detailed reasoning>
    """

    def __init__(self, config: SearchRelevanceConfig = SearchRelevanceConfig()):
        """Initialize the evaluator with configuration"""
        self.config = config
        self.llm = ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=512
        ).with_structured_output(
            SearchRelevanceGrade
        )

    async def evaluate(self, inputs: Dict[str, Any], outputs: Dict[str, Any], reference_outputs: Dict[str, Any]) -> \
            EvaluationResult:
        """Evaluate search result relevance for a given query.

        Args:
            inputs: Dictionary containing:
                - 'query': The search query.
            outputs: Dictionary containing:
                - 'search_results': The retrieved search results.
            reference_outputs: Dictionary containing:
                - 'reference_answer': The expected ideal answer.

        Returns:
            EvaluationResult: Structured evaluation result
        """
        # Extract search result summaries
        search_snippets = "\n\n".join([
            f"Title: {result['title']}\nContent Summary: {result['content']}"
            for result in outputs["tavily_response"]["results"]
        ])

        # Format the evaluation prompt
        evaluation_text = f"""SEARCH RESULTS:\n{search_snippets}\n\nSEARCH QUERY: {inputs['question']}\n\nREFERENCE ANSWER: {reference_outputs['answer']}"""

        # Run evaluation
        grade = self.llm.invoke([
            {"role": "system", "content": self.EVALUATION_PROMPT},
            {"role": "user", "content": evaluation_text}
        ])

        # Map scores to relevance labels
        relevance_map = {
            5: "HIGHLY_RELEVANT",
            4: "MOSTLY_RELEVANT",
            3: "PARTIALLY_RELEVANT",
            2: "SLIGHTLY_RELEVANT",
            1: "BARELY_RELEVANT",
            0: "NOT_RELEVANT"
        }

        # Normalize score to 0-1 range
        normalized_score = grade.score / 5.0

        return EvaluationResult(
            key="search_relevance",
            score=normalized_score,
            value=relevance_map[grade.score],
            comment=grade.explanation,
            evaluator_info={
                "name": self.evaluation_name,
                "description": self.evaluation_description
            },
            extra={
                "raw_score": grade.score,
                "relevance_level": relevance_map[grade.score]
            }
        )

    @property
    def evaluation_name(self) -> str:
        """Name of this evaluator"""
        return "search_relevance"

    @property
    def evaluation_description(self) -> str:
        """Description of what this evaluator does"""
        return "Evaluates whether web search results are relevant to the input query"
import asyncio

async def main():
    """Test the SearchRelevanceEvaluator with a sample query, search results, and reference output."""
    config = SearchRelevanceConfig()
    evaluator = SearchRelevanceEvaluator(config)

    inputs = {
        "query": "What is the capital of France?"
    }

    outputs = {
        "search_results": [
            {"title": "Paris - Capital of France", "snippet": "Paris is the capital city of France, known for the Eiffel Tower.", "url": "https://example.com/paris"},
            {"title": "Best cities to visit in Europe", "snippet": "Europe has many great cities, including Berlin, Madrid, and Rome.", "url": "https://example.com/europe-travel"},
            {"title": "Paris travel guide", "snippet": "Find out the best places to visit in Paris, the capital of France.", "url": "https://example.com/paris-travel"},
        ]
    }

    reference_outputs = {
        "reference_answer": "The capital of France is Paris."
    }

    result = await evaluator.evaluate(inputs, outputs, reference_outputs)
    print(f"Relevance Score: {result['score']}")
    print(f"Explanation: {result['explanation']}")

# if __name__ == "__main__":
#     asyncio.run(main())
