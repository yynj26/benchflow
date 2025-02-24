import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, final
from urllib.parse import urljoin

import requests

from benchflow.schemas.InputData import InputData

logger = logging.getLogger(__name__)

class BenchClient(ABC):
    """
    The BenchClient is used to make the benchmark client so that it can communicate with the agent server.
    You need to extend this class in your benchmark entrypoint (e.g. run.py).
    You need to implement the following methods:
        - prepare_input
        - parse_response
    """
    
    def __init__(self, agent_url: str, max_retry: int = 1):
        self.agent_url = agent_url.rstrip('/')
        self.max_retry = max_retry
        logger.info(f"[{self.__class__.__name__}] Initialized with agent_url: {agent_url}")

    @final
    def get_response(self, raw_input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the response from the agent. You should use this method to get the response from the agent.
        """
        if raw_input_data is None:
            raise ValueError("raw_input_data cannot be None")
        
        input_data = self.prepare_input(raw_input_data)
        # old env_info will be deprecated, we use input_data instead
        if input_data.get("env_info") is not None:
            input_data = InputData(input_data=input_data["env_info"])
        else:
            input_data = InputData(input_data=input_data)
        
        for attempt in range(self.max_retry):
            try:
                response = requests.post(
                    urljoin(self.agent_url, "action"),
                    json=input_data.model_dump()
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
    def prepare_input(self, raw_input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input:
            raw_input_data: Dict[str, Any]
            For example, 
                If your benchmark is a web agent benchmark, the raw_input_data could be the observation from the web page.
                if your benchmark is a Q&A benchmark, the raw_input_data could be the question.
                You should define the keys in the raw_input_data.
                If your benchmark don't need to deal with the raw_input_data, you can just return the raw_input_data.
        Output:
            Dict[str, Any]
            The input data of the task to be sent to the agent.
            And add the keys to the benchmark documentation. # To Be Done in benchflow v0.2.0
        """
        pass

    @abstractmethod
    def parse_response(self, raw_response: str) -> Dict[str, Any]:
        """
        Input:
            raw_response: str
            The raw response from the agent.

        You can specify the format of the raw_action in the benchmark documentation. # To Be Done in benchflow v0.2.0
        So that agent developers can know what to return.

        For example,
            you can specify the format of the raw_response as follows:
            ```
            "action_type": click
            "action_arguments": arguments
            ```
            so that you can use regex to parse the response_type and response_arguments from the raw_response.
            and return the parsed_action as follows:
            ```
            {
                "action_type": click,
                "action_arguments": arguments
            }
            ```

        Output:
            parsed_response: Dict[str, Any]
            The parsed response.
            You need to specify the keys in the parsed_response that you want to send to the benchmark.
            And add the keys to the benchmark documentation. # To Be Done in benchflow v0.2.0
        """
        pass
