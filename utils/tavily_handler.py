import asyncio
import aiohttp
import os
from typing import Dict, Any, Optional, Tuple

class TavilyHandler:
    """Handles interactions with the Tavily API."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        env: Optional[str] = "prod",
        search_params: Optional[Dict[str, Any]] = None
    ):
        """Initialize the TavilyHandler.
        
        Args:
            api_key: Tavily API key (defaults to TAVILY_API_KEY environment variable)
            env: Environment to use (prod/dev/staging)
            search_params: Default search parameters to use for all searches
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("Tavily API key not provided and TAVILY_API_KEY environment variable not set")
        
        self.BASE_URL = {
            'dev': 'http://localhost:8000',
            'staging': 'https://staging-api.tavily.com',
            'prod': 'https://api.tavily.com'
        }.get(env, 'https://api.tavily.com')

        # Store default search parameters
        self.search_params = search_params or {}
        self.decompose_query = False

    async def search(
        self,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a Tavily search using async HTTP request.
        
        Args:
            inputs: Dictionary containing input data, must include 'question' key
            
        Returns:
            Dictionary containing 'answer' and 'raw_response'
        """
        question = inputs["question"]
        headers = {
            'Content-Type': 'application/json',
        }

        # Construct request data
        data = {
            'query': question,
            'api_key': self.api_key,
            **self.search_params
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.BASE_URL}/search",
                    json=data,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        print(f"Error in Tavily search: HTTP {response.status}")
                        return {
                            "answer": "",
                            "raw_response": None
                        }

                    response_data = await response.json()
                    return {
                        "answer": response_data.get('answer', ''),
                        "tavily_response": response_data
                    }

        except Exception as e:
            print(f"Error in Tavily search: {str(e)}")
            return {
                "answer": "",
                "raw_response": None
            }