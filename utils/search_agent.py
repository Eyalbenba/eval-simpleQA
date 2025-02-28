from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from utils.tavily_handler import TavilyHandler
import asyncio
from pydantic import BaseModel , Field
import logging
import json
from datetime import datetime
import dotenv
dotenv.load_dotenv()

# Configure logging
def setup_logger(name: str) -> logging.Logger:
    """Setup a logger with JSON formatting to console."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create formatter for JSON output
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName
            }

            # Add extra fields if they exist
            if hasattr(record, "extra"):
                log_data.update(record.extra)

            return json.dumps(log_data)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JsonFormatter())
    logger.addHandler(console_handler)

    return logger

@dataclass
class DecomposedQuery(BaseModel):
    """Represents a decomposed sub-question with its context"""
    question: str = Field(description="The search-optimized sub-question")

class DecomposedQueries(BaseModel):
    """Collection of decomposed queries"""
    decomposed: bool = Field(description="Whether the question needed decomposition")
    questions: List[DecomposedQuery] = Field(description="List of decomposed sub-questions")

@dataclass
class SearchResult:
    """Represents extracted answer from search results"""
    query: str
    answer: str
    confidence: float
    sources: List[str]

class SearchAgent:
    """Agent that decomposes questions and orchestrates search and answer synthesis"""

    DECOMPOSE_PROMPT = """Break down complex questions into simple, search-engine-optimized sub-questions.
    For complex queries, create focused sub-questions that:
    - Use simple, direct language (15 words or less)
    - Include specific search terms
    - Are self-contained (no references to other questions)
    - Focus on single facts or concepts

    Example decomposition:
    Input: "What were the major impacts of the Industrial Revolution on urban development and public health?"

    Output sub-questions:
    1. "What changes did Industrial Revolution bring to cities?"
    2. "How did Industrial Revolution affect public health?"
    3. "What were living conditions in Industrial Revolution cities?"

    Question: {question}
    """

    ANSWER_SYNTHESIS_PROMPT = """You are an expert at synthesizing information from multiple sources.
    Review the following answers to sub-questions and create a final comprehensive answer.

    Original Question: {original_question}

    Sub-questions and their answers:
    {subquestion_answers}

    Provide:
    1. A synthesized final answer to the original question , keep it concise and answer only on what the question is asking.
    2. A confidence score (0-1) indicating how certain you are about the answer.

    Format as JSON with fields: answer , confidence
    """

    def __init__(
        self,
        tavily_handler: TavilyHandler,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.0
    ):
        """Initialize the SearchAgent.

        Args:
            tavily_handler: Initialized TavilyHandler instance
            llm_model: Model to use for question decomposition and synthesis
            temperature: Temperature for LLM calls
        """
        self.tavily = tavily_handler
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature
        )
        self.logger = setup_logger(f"search_agent.{id(self)}")

    async def _decompose_question(self, question: str) -> DecomposedQueries:
        """Decompose complex question into sub-questions using LLM."""
        self.logger.info("Decomposing question", extra={
            "question": question,
            "stage": "decomposition"
        })

        try:
            response = await self.llm.with_structured_output(DecomposedQueries).ainvoke([{
                "role": "system",
                "content": self.DECOMPOSE_PROMPT.format(question=question)
            }])

            if not response.decomposed:
                return DecomposedQueries(
                    decomposed=False,
                    questions=[DecomposedQuery(
                        question=question,
                        reasoning="Direct question requiring no decomposition",
                        order=1
                    )]
                )

            self.logger.info("Question successfully decomposed", extra={
                "question": question,
                "sub_questions": [q.question for q in response.questions],
                "stage": "decomposition"
            })
            return response

        except Exception as e:
            self.logger.error("Error decomposing question", extra={
                "question": question,
                "error": str(e),
                "stage": "decomposition"
            })
            raise

    async def _parallel_search(self, queries: List[DecomposedQuery]) -> Dict[str, Any]:
        """Execute searches for all sub-questions in parallel."""
        self.logger.info("Starting parallel searches", extra={
            "num_queries": len(queries),
            "queries": [q.question for q in queries],
            "stage": "search"
        })

        try:
            search_tasks = [
                self.tavily.search({"question": q.question})
                for q in queries
            ]
            results = await asyncio.gather(*search_tasks)

            self.logger.info("Parallel searches completed", extra={
                "num_results": len(results),
                "stage": "search"
            })
            return results

        except Exception as e:
            self.logger.error("Error during parallel searches", extra={
                "error": str(e),
                "stage": "search"
            })
            raise

    async def _extract_answers(
        self,
        queries: List[DecomposedQuery],
        search_results: List[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Extract relevant answers from search results for each sub-question."""
        self.logger.info("Starting answer extraction", extra={
            "num_queries": len(queries),
            "stage": "extraction"
        })

        answers = []
        try:
            for query, result in zip(queries, search_results):
                self.logger.debug("Processing query results", extra={
                    "query": query.question,
                    "num_results": len(result.get("tavily_response", {}).get("results", [])),
                    "stage": "extraction"
                })

                sources = [
                    f"{r['title']}: {r['content']}"
                    for r in result.get("tavily_response", {}).get("results", [])
                ]

                response = await self.llm.ainvoke([{
                    "role": "system",
                    "content": f"""Extract the most relevant answer to this question from the sources:
                    Question: {query.question}
                    
                    Sources:
                    {sources}
                    
                    Format response as JSON with fields:
                    - answer: extracted answer
                    - confidence: float 0-1
                    - supporting_sources: list of source indices used
                    """
                }])

                extracted = json.loads(response.content)
                answers.append(SearchResult(
                    query=query.question,
                    answer=extracted["answer"],
                    confidence=extracted["confidence"],
                    sources=[sources[i] for i in extracted["supporting_sources"]]
                ))

                self.logger.debug("Answer extracted", extra={
                    "query": query.question,
                    "confidence": extracted["confidence"],
                    "stage": "extraction"
                })

            self.logger.info("Answer extraction completed", extra={
                "num_answers": len(answers),
                "stage": "extraction"
            })
            return answers

        except Exception as e:
            self.logger.error("Error during answer extraction", extra={
                "error": str(e),
                "stage": "extraction"
            })
            raise

    async def _synthesize_final_answer(
        self,
        original_question: str,
        search_results: List[SearchResult]
    ) -> Dict[str, Any]:
        """Synthesize final answer from all sub-question results."""
        self.logger.info("Starting answer synthesis", extra={
            "original_question": original_question,
            "num_results": len(search_results),
            "stage": "synthesis"
        })

        try:
            subquestion_answers = "\n\n".join([
                f"Sub-question: {r.query}\nAnswer: {r.answer}\n"
                for r in search_results
            ])

            response = await self.llm.ainvoke([{
                "role": "system",
                "content": self.ANSWER_SYNTHESIS_PROMPT.format(
                    original_question=original_question,
                    subquestion_answers=subquestion_answers
                )
            }])

            final_answer = json.loads(response.content)
            self.logger.info("Answer synthesis completed", extra={
                "confidence": final_answer["confidence"],
                "stage": "synthesis"
            })
            return final_answer
            
        except Exception as e:
            self.logger.error("Error during answer synthesis", extra={
                "error": str(e),
                "stage": "synthesis"
            })
            raise

    async def search(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multi-step search and answer synthesis process.
        
        Args:
            inputs: Dictionary containing 'question' key
            
        Returns:
            Dictionary containing synthesized answer and supporting information
        """
        question = inputs["question"]
        
        # Step 1: Decompose question
        decomposed_queries = await self._decompose_question(question)
        
        # Step 2: Execute parallel searches
        search_results = await self._parallel_search(decomposed_queries.questions)
        
        # Step 3: Extract answers from search results
        extracted_answers = await self._extract_answers(decomposed_queries.questions, search_results)
        
        # Step 4: Synthesize final answer
        final_answer = await self._synthesize_final_answer(question, extracted_answers)
        
        return {
            "answer": final_answer["answer"],
            "confidence": final_answer["confidence"],
            "sub_questions": [
                {
                    "question": q.question,
                }
                for q in decomposed_queries.questions
            ],
            "search_results": search_results
        }


if __name__ == "__main__":
    # Example usage
    agent = SearchAgent(TavilyHandler())
    asyncio.run(agent.search({"question": "In which year did the former PM of J&K, Mehr Chand Mahajan, become a judge of the Lahore High Court?"}))
