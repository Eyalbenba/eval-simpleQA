import re
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from dataclasses import dataclass
from typing import Annotated
from dotenv import load_dotenv
load_dotenv()
from langsmith.evaluation.evaluator import EvaluationResult

class CorrectnessGrade(BaseModel):
    """Schema for correctness evaluation result."""
    grade: Annotated[str, "A (CORRECT), B (INCORRECT), or C (NOT_ATTEMPTED)"]



@dataclass
class CorrectnessConfig:
    """Configuration for correctness evaluation."""
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0


class CorrectnessEvaluator:
    """Evaluator for assessing the correctness of predicted answers."""

    # Grading prompt template
    GRADER_TEMPLATE = """
Your job is to evaluate whether a predicted answer correctly answers the question compared to the reference (gold) answer.

### **Scoring Criteria**
- **CORRECT (A)**: The predicted answer fully contains the important information in the reference answer, with no contradictions.
- **INCORRECT (B)**: The predicted answer contains incorrect or misleading information.
- **NOT_ATTEMPTED (C)**: The predicted answer does not attempt to answer the question or is too vague.

### **Examples:**
#### ✅ CORRECT:
- **Question:** "Who is the president of the United States in 2024?"
- **Reference Answer:** "Joe Biden."
- **Predicted Answer:** "Joe Biden is the president of the United States in 2024."

#### ❌ INCORRECT:
- **Predicted Answer:** "Donald Trump is the president in 2024."

#### ⚠️ NOT_ATTEMPTED:
- **Predicted Answer:** "I don't know."

### **New Evaluation:**
**Question:** {question}  
**Reference Answer:** {reference_answer}  
**Predicted Answer:** {predicted_answer}

Return only the grade: **A (CORRECT), B (INCORRECT), or C (NOT_ATTEMPTED)**.
"""

    def __init__(self, config: CorrectnessConfig = CorrectnessConfig()):
        """Initialize the evaluator with configuration."""
        self.config = config
        self.llm = ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature
        ).with_structured_output(
            CorrectnessGrade
        )

    async def evaluate(self, inputs: Dict[str, Any], outputs: Dict[str, Any], reference_outputs: Dict[str, Any]) -> EvaluationResult:
        """Evaluate the correctness of a predicted answer.

        Args:
            inputs: Dictionary containing 'query'.
            outputs: Dictionary containing 'predicted_answer'.
            reference_outputs: Dictionary containing 'reference_answer'.

        Returns:
            EvaluationResult: Structured evaluation result
        """
        # Format grading prompt
        grader_prompt = self.GRADER_TEMPLATE.format(
            question=inputs["question"],
            reference_answer=reference_outputs["answer"],
            predicted_answer=outputs["answer"]
        )

        # Run evaluation
        grade_response = self.llm.invoke([
            {"role": "user", "content": grader_prompt}
        ])

        # Extract grade (A, B, or C)
        match = re.search(r"(A|B|C)", grade_response.grade)
        grade = match.group(0) if match else "C"

        # Map grades to labels
        grade_map = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}

        return EvaluationResult(
            key="correctness",
            score=1.0 if grade == "A" else 0.0,
            value=grade_map[grade],
            comment=f"Answer was {grade_map[grade].lower()}",
            evaluator_info={
                "name": self.evaluation_name,
                "description": self.evaluation_description
            },
            extra={
                "grade_letter": grade,
                "not_attempted": grade == "C",
                "status": "not_attempted" if grade == "C" else "attempted"
            }
        )

    @property
    def evaluation_name(self) -> str:
        """Name of this evaluator."""
        return "correctness_evaluator"

    @property
    def evaluation_description(self) -> str:
        """Description of what this evaluator does."""
        return "Evaluates the correctness of a predicted answer against a reference answer."
import asyncio

async def main():
    """Test the CorrectnessEvaluator with a sample query, reference answer, and predicted answer."""
    config = CorrectnessConfig()
    evaluator = CorrectnessEvaluator(config)

    inputs = {
        "query": "Who is the president of the United States in 2024?"
    }

    outputs = {
        "predicted_answer": "Joe Biden is the president of the United States in 2024."
    }

    reference_outputs = {
        "reference_answer": "Joe Biden."
    }

    result = await evaluator.evaluate(inputs, outputs, reference_outputs)
    print(f"Grade: {result['grade']}")
    print(f"Explanation: {result['explanation']}")

# if __name__ == "__main__":
#     asyncio.run(main())
