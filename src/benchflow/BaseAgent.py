import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class FeedbackRequest(BaseModel):
   env_info: Dict[str, Any] = None

class BaseAgent(ABC):
    """Base agent class that handles state management and HTTP interface"""
    def __init__(self):
        self.app = FastAPI()
        self.setup_routes()
        self.env_info = None

    def setup_routes(self):
        @self.app.post("/action")
        async def get_action(feedback: FeedbackRequest):
            try:
                self.update_env_info(feedback.model_dump()['env_info'])
                action = self.call_api()
                logger.info(f"[BaseAgent]: Got action from API: {action}")
                return {"action": action}
            except Exception as e:
                logger.error(f"[BaseAgent]: Error getting action: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/")
        async def root():
            return {"message": "Welcome to Benchmarkthing Agent API"}

    @abstractmethod 
    def call_api(self) -> str:
        pass

    def update_env_info(self, env_info: Dict[str, Any]) -> None:
        self.env_info = env_info

    def run_with_endpoint(self, host: str, port: int):
        logger.info(f"Starting agent server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)