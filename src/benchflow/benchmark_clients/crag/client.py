import logging
from typing import Dict, Any
from benchflow import BenchClient

logger = logging.getLogger(__name__)

class CRAGBenchClient(BenchClient):
    def __init__(self, agent_url: str, max_retry: int = 1):
        super().__init__(agent_url, max_retry)

    def prepare_environment(self, state_update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
        state_update:
        {
            "question": str, # The question to be answered
            "context": List[Dict], # List of relevant documents
            "metadata": Dict[str, Any] # Optional metadata about the question
        }

        Returns:
        env_params:
        {
            "query": str, # The question
            "documents": List[Dict], # Formatted documents
            "task_type": str, # rag_qa for CRAG
            "metadata": Dict[str, Any] # Additional task information
        }
        """

        if "question" not in state_update:
            raise ValueError("state_update must contain 'question'")
        
        documents = state_update.get("context", [])
        formatted_docs = [
            {
                "content": doc.get("content", ""),
                "metadata": doc.get("metadata", {})
            }
            for doc in documents
        ]

        env_params = {
            "query": state_update["question"],
            "documents": formatted_docs,
            "task_type": "rag_qa",
            "metadata": state_update.get("metadata", {})
        }

        return env_params


    def parse_action(self, raw_action: str) -> Dict[str, Any]:
        """
        Args:
            raw_action: The response from the agent
            Expected format: Either a string answer or a JSON object with
            answer, sources, and optional reasoning
            
        Returns:
        {
            "answer": str,              # The agent's answer
            "sources": List[str],       # References to source documents used
            "reasoning": str,           # Optional explanation of reasoning
            "confidence": float,        # Optional confidence score
        }
        """
        
