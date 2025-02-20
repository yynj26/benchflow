import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, final
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)

class BenchClient(ABC):
    """
    The BenchClient is used to make the benchmark client so that it can communicate with the agent server.
    You need to extend this class in your benchmark entrypoint (e.g. run.py).
    You need to implement the following methods:
        - prepare_environment
        - parse_action
    """
    
    def __init__(self, agent_url: str, max_retry: int = 1):
        self.agent_url = agent_url.rstrip('/')
        self.max_retry = max_retry
        logger.info(f"[{self.__class__.__name__}] Initialized with agent_url: {agent_url}")

    @final
    def get_action(self, state_update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the action from the agent.
        """
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
    
    @abstractmethod
    def prepare_environment(self, state_update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input:
            state_update: Dict[str, Any]
            For example, 
                If your benchmark is a web agent benchmark, the state_update could be the observation from the web page.
                if your benchmark is a Q&A benchmark, the state_update could be the question.
                You should define the keys in the state_update.
                If your benchmark don't need to deal with the state_update, you can just return the raw state_update.
        Output:
            env_data: Dict[str, Any]
            The environment data to be sent to the agent.
            You need to specify the keys in the env_data that you want to send to the agent.
            And add the keys to the benchmark documentation. # To Be Done in benchflow v0.2.0
        """
        pass

    @abstractmethod
    def parse_action(self, raw_action: str) -> Dict[str, Any]:
        """
        Input:
            raw_action: str
            The raw action from the agent.

        You can specify the format of the raw_action in the benchmark documentation. # To Be Done in benchflow v0.2.0
        So that agent developers can know what to return.

        For example,
            you can specify the format of the raw_action as follows:
            ```
            "action_type": click
            "action_arguments": arguments
            ```
            so that you can use regex to parse the action_type and action_arguments from the raw_action.
            and return the parsed_action as follows:
            ```
            {
                "action_type": click,
                "action_arguments": arguments
            }
            ```

        Output:
            parsed_action: Dict[str, Any]
            The parsed action.
            You need to specify the keys in the parsed_action that you want to send to the benchmark.
            And add the keys to the benchmark documentation. # To Be Done in benchflow v0.2.0
        """
        pass
