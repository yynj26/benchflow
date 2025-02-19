import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, final

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class FeedbackRequest(BaseModel):
   env_info: Dict[str, Any] = None

class BaseAgent(ABC):
    """
    You need to extend this class to make your agent a server.
    So that it can communicate with the benchmark client.
    If you want to integrate your agent with BenchFlow, you need to implement the following methods:
    ```
    - call_api
    ```
    """
    def __init__(self):
        self.app = FastAPI()
        self.setup_routes()
        self.env_info = None  # Maintain compatibility with old agent

    @final
    def setup_routes(self):
        """
        Setup the routes for the agent.
        """
        @self.app.post("/action")
        async def take_action(feedback: FeedbackRequest):
            try:
                self.update_env_info(feedback.model_dump()['env_info'])
                action = self.call_api(feedback.model_dump()['env_info'])
                logger.info(f"[BaseAgent]: Got action from API: {action}")
                return {"action": action}
            except Exception as e:
                logger.error(f"[BaseAgent]: Error getting action: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/")
        async def root():
            return {"message": "Welcome to Benchmarkthing Agent API"}
        
    @final
    def update_env_info(self, env_info: Dict[str, Any]) -> None:
        """
        Update the environment information.
        """
        self.env_info = env_info

    @final
    def run_with_endpoint(self, host: str, port: int):
        """
        Run the agent server.
        """
        logger.info(f"Starting agent server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)

    @abstractmethod 
    def call_api(self, env_info: Dict[str, Any]) -> str:
        """
        You can get the request information from the env_info parameter.
        The env_info is a dictionary that contains the keys provided by the benchmark client.
        You need to refer to the benchmark documentation to get the keys.

        This method is called when the agent server receives a request from the benchmark client.
        You need to implement this method to make your agent work and return the action to the benchmark client.
        Your action could be a real action(e.g. click, scroll, etc) or just any prediction(e.g. code, text, etc) needed by the benchmark.
        """
        pass