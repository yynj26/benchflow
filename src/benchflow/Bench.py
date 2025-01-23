import sys
import base64
import time
import requests
from typing import List, Union, Dict, Any
import logging
from .BaseAgent import BaseAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Bench:
    def __init__(self, benchmark: Dict[str, Any]):
        self.benchmark_url = benchmark["benchmark_url"]
        self.benchmark_name = benchmark["benchmark_name"]
        self.resource_manager_url = "http://159.89.229.132:10000"
        self.running_tasks = {}
        self.results = {}
        
    def run(self, task_ids: Union[str|int, List[str|int]], agents: Union[BaseAgent, List[BaseAgent]], require_gpu: bool = False):
        if isinstance(task_ids, str|int):
            task_ids = [str(task_ids)]
        if isinstance(agents, BaseAgent):
            agents = [agents]

        results_ids = []
        for task_id in task_ids:
            task_id = str(task_id)
            for Baseagent in agents:
                results_ids.append(self._run_single_task(task_id, Baseagent, require_gpu))
        self.cleanup()
        return results_ids

    def get_results(self, run_ids: List[str]):
        return [self.results[run_id] for run_id in run_ids]

    def _run_single_task(self, task_id: str, Baseagent: BaseAgent, require_gpu: bool):
        logger.info(f"Starting task {task_id} on {Baseagent.__class__.__name__}")
        
        try:
            agent_code = self._get_agent_code(Baseagent)
            response = requests.post(
                f"{self.resource_manager_url}/deploy",
                json={
                    "agent_code": base64.b64encode(agent_code.encode()).decode(),
                    "require_gpu": require_gpu,
                    "benchmark_name": self.benchmark_name
                }
            )
            response.raise_for_status()
            logger.info(f"{Baseagent.__class__.__name__} deployed successfully on port {response.json()['port']}")
            
            deploy_info = response.json()
            host, port = deploy_info["host"], deploy_info["port"]
            agent_url = f"http://{host}:{port}"


            self.running_tasks[task_id] = {
                "host": host,
                "port": port,
                "start_time": time.time()
            }

            url = f"{self.benchmark_url}/api/v1/{self.benchmark_name}/evaluate"

            result = requests.post(
                url,
                json={
                    "task_id": task_id,
                    "agent_url": agent_url,
                    "params": {}
                }
            )

            logger.info(f"Task {task_id} on {Baseagent.__class__.__name__} finished")
            self.results[task_id] = result.json()
            return task_id
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {str(e)}")
            self.results[task_id] = "failed"
            return task_id
        
    def _get_agent_code(self, Baseagent: BaseAgent) -> str:
        agent_file = sys.modules[Baseagent.__class__.__module__].__file__
        with open(agent_file, 'r') as f:
            return f.read()

    def cleanup(self):
        for task_id, task_info in self.running_tasks.items():
            logger.info(f"Releasing task {task_id} on {task_info['host']}:{task_info['port']}")
            try:
                response = requests.post(
                    f"{self.resource_manager_url}/release",
                    json={
                        "host": task_info["host"],
                        "port": task_info["port"]
                    }
                )
                response.raise_for_status()
                logger.info(f"Release successfully for task {task_id} on {task_info['host']}:{task_info['port']}")
            except Exception as e:
                logger.error(f"Release request failed for task {task_id} on {task_info['host']}:{task_info['port']}: {str(e)}")