from benchflow import BaseBench
from typing import Dict, Any
import docker
import os
import json

class WebCanvasBench(BaseBench):
    def __init__(self):
        super().__init__()
        self.image_name = "kirk2000/benchflow:webcanvas-v1"
        self.results_dir = os.path.abspath("./tmp/webcanvas/results")
        self.log_files_dir = os.path.abspath("./tmp/webcanvas/log_files")

    def run_bench(self, task_id: str, agent_url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        client = docker.from_env()
        self._validate_params(params)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.log_files_dir, exist_ok=True)

        try:
            container = client.containers.run(
                image=self.image_name,
                environment={
                    "AGENT_URL": agent_url,
                    "TEST_START_IDX": str(task_id),
                    "TEST_END_IDX": str(task_id),
                    "BROWSERBASE_API_KEY": params["browserbase_api_key"],
                    "GRAPHQL_USERNAME": params["graphql_username"],
                    "GRAPHQL_PASSWORD": params["graphql_password"],
                    "OPENAI_API_KEY": params["openai_api_key"],
                    "RESULTS_DIR": "/app/batch_tasks_results/example"
                },
                volumes={
                    self.results_dir: {
                        'bind': '/app/batch_tasks_results',
                        'mode': 'rw'
                    },
                    self.log_files_dir: {
                        'bind': '/app/LOGS',
                        'mode': 'rw'
                    }
                },
                remove=True,
                detach=True
            )
   
            output = ""
            for line in container.logs(stream=True):
                line_str = line.decode('utf-8')
                self.logger.info(line_str)
                output += line_str

            container.wait()

            result_file = os.path.join(self.results_dir, "example", "result", "result.json")
            if not os.path.exists(result_file):
                return {
                    "task_id": str(task_id),
                    "is_resolved": False,
                    "score": 0,
                    "message": {"error": f"No results found"}
                }

            results = self._get_results(result_file)

            return {
                "task_id": str(task_id),
                "is_resolved": results["task_success_rate"] > 0.99,
                "score": results["average_step_score_rate"],
                "message": '{' + ', '.join(f'{k}: {v}' for k,v in results.items()) + '}'
            }

        except docker.errors.ImageNotFound:
            return {
                "task_id": str(task_id),
                "is_resolved": False,
                "score": 0,
                "message": {"error": f"Image not found"}
            }
  
        except Exception as e:
            return {
                "task_id": str(task_id),
                "is_resolved": False,
                "score": 0,
                "message": {"error": f"{e}"}
            }

    def get_all_tasks(self, split: str) -> Dict[str, Any]:
        task_ids = [str(i) for i in range(103)]
        if split == "train":
            task_ids = [str(i) for i in range(20)]
        return {"task_ids": task_ids, "error_message": None}

    def cleanup(self):
        pass

    def _validate_params(self, params: Dict[str, Any]) -> bool:
        if not params.get("browserbase_api_key"):
            raise ValueError("Invalid parameters: browserbase_api_key is required")
        if not params.get("graphql_username"):
            raise ValueError("Invalid parameters: graphql_username is required")
        if not params.get("graphql_password"):
            raise ValueError("Invalid parameters: graphql_password is required")
        if not params.get("openai_api_key"):
            raise ValueError("Invalid parameters: openai_api_key is required")
        return True

    def _get_results(self, results_dir: str) -> Dict[str, Any]:
        with open(results_dir, 'r') as f:
            data = f.read().strip()
            data = data.replace('{', '{"').replace(': ', '": ').replace(', ', ', "')
            return json.loads(data)