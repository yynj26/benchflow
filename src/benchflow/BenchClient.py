from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import requests
import logging
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

class BenchClient(ABC):
    
    def __init__(self, agent_url: str, max_retry: int = 1):
        self.agent_url = agent_url.rstrip('/')
        self.max_retry = max_retry
        logger.info(f"[{self.__class__.__name__}] Initialized with agent_url: {agent_url}")

    @abstractmethod
    def prepare_environment(self, state_update: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def parse_action(self, raw_action: str) -> Dict[str, Any]:
        pass

    def get_action(self, state_update: Dict[str, Any]) -> Dict[str, Any]:
        if state_update is None:
            raise ValueError("state_update cannot be None")
        
        env_data = self.prepare_environment(state_update)
        
        for attempt in range(self.max_retry):
            try:
                response = requests.post(
                    urljoin(self.agent_url, "action"),
                    json=env_data
                )
                response.raise_for_status()
                break
            except Exception as e:
                if attempt == self.max_retry - 1:
                    raise Exception(f"Failed to get action after {self.max_retry} attempts: {str(e)}")
                logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                continue

        try:
            raw_action = response.json()['action']
            logger.info(f"[{self.__class__.__name__}] Received action: {raw_action}")
            
            parsed_action = self.parse_action(raw_action)
            parsed_action["raw_prediction"] = raw_action
            
            return parsed_action
            
        except KeyError as e:
            raise ValueError(f"Invalid response format from agent: {str(e)}")
        except Exception as e:
            raise Exception(f"Error parsing action: {str(e)}")