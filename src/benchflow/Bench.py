import signal
import sys
import base64
import time
import requests
from typing import List, Union
import logging
from .BaseAgent import BaseAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Bench:
    def __init__(self, benchmark_url: str):
        self.benchmark_url = benchmark_url
        self.resource_manager_url = "http://159.89.229.132:10000"
        self.running_tasks = {}
        self.results = {}

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def run(self, task_ids: Union[str, List[str]], agents: Union[BaseAgent, List[BaseAgent]], require_gpu: bool = False):
        if isinstance(task_ids, str):
            task_ids = [task_ids]
        if isinstance(agents, BaseAgent):
            agents = [agents]

        results = []
        for task_id in task_ids:
            for Baseagent in agents:
                results.append(self._run_single_task(task_id, Baseagent, require_gpu))
            
        return results if len(task_ids) > 1 or len(agents) > 1 else results[0]

    def _run_single_task(self, task_id: str, Baseagent: BaseAgent, require_gpu: bool):
        logger.info(f"Starting task {task_id}")
        
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
            
            deploy_info = response.json()
            host, port = deploy_info["host"], deploy_info["port"]
            agent_url = f"http://{host}:{port}"
            
            self.running_tasks[task_id] = {
                "host": host,
                "port": port,
                "start_time": time.time()
            }

            result = self.benchmark_server.evaluate(agent_url, task_id)
            self.results[task_id] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {str(e)}")
            if task_id in self.running_tasks:
                task_info = self.running_tasks[task_id]
                requests.post(
                    f"{self.resource_manager_url}/release",
                    json={"host": task_info["host"], "port": task_info["port"]}
                )
                del self.running_tasks[task_id]
            raise
            
    def _get_agent_code(self, Baseagent: BaseAgent) -> str:
        agent_file = sys.modules[Baseagent.__class__.__module__].__file__
        with open(agent_file, 'r') as f:
            return f.read()

    def cleanup(self):
            print("Cleaning up resources...")
            print(self.running_tasks.items())
            for task_id, task_info in self.running_tasks.items():
                print("send release request")
                try:
                    requests.post(
                        f"{self.resource_manager_url}/release",
                        json={
                            "host": task_info["host"],
                            "port": task_info["port"]
                        }
                    )
                except:
                    print("release request failed")
                    pass

    def __del__(self):
        self.cleanup()

    # def _signal_handler(self, signum, frame):
    #     self.cleanup()
    #     sys.exit(0)