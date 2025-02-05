from typing import Dict, Any
from abc import ABC, abstractmethod

class BaseBench(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run_bench(self, agent_url: str, task_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_all_tasks(self):
        pass