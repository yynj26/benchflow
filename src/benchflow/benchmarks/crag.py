import os
import json
import subprocess
from typing import Any, Dict
from benchflow import BaseBench, BaseBenchConfig

class CRAGConfig(BaseBenchConfig):

    def __init__(self, params: Dict[str, Any], task_id: str):
        params.setdefault("BATCH_SIZE", 100)
        self.required_params = ["OPENAI_API_KEY", "EVALUATION_MODEL_NAME"] # for llm api call evaluation when exact match fails
        self.optional_params = ["BATCH_SIZE"]
        super().__init__(params)


class CRAGBench(BaseBench):
    def __init__(self):
        super().__init__()

    def get_config(self, params: Dict[str, Any], task_id: str) -> CRAGConfig:
        return CRAGConfig(params, task_id)
    
    def get_image_name(self) -> str:
        return "kirk2000/benchflow:crag-v1" # TODO: check if we need to push the image to docker hub
    
    def get_results_dir_in_container(self) -> str:
        return "/app/results"

    def get_log_files_dir_in_container(self) -> str:
        return "/app/logs"
       
    def get_result(self, task_id: str) -> Dict[str, Any]:
        result_file = os.path.join(self.results_dir, f"{task_id}_results.json")

        if not os.path.exists(result_file):
            return {
                "is_resolved": False,
                "score": 0,
                "message": {"error": "No results found"}
            }
        
        try:
            with open(result_file, "r") as f:
                results = json.load(f)

            is_resolved = False
            if results.get("score"):
                is_resolved = True

            return {
                "is_resolved": is_resolved,
                "score": results.get("score", 0),
                "message": {"details": "Task runs successfully."},
                "log": str(results)
            }
        
        except Exception as e:
            return {
                "is_resolved": False,
                "score": 0,
                "message": {"error": str(e)},
                "log": str(e),
            }
        
    def get_all_tasks(self, split: str) -> Dict[str, Any]:
        # Only one task for CRAG benchmark
        return {"task_ids": ["0"], "error_message": None}
        
    def cleanup(self):
        if os.path.exists(self.results_dir):
            self.logger.info(f"Removing {self.results_dir}")
            subprocess.run(['sudo', 'rm', '-rf', self.results_dir], check=True)
        if os.path.exists(self.log_files_dir):
            self.logger.info(f"Removing {self.log_files_dir}")
            subprocess.run(['sudo', 'rm', '-rf', self.log_files_dir], check=True)


