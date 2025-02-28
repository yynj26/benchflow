import os
import json
import glob
from typing import Any, Dict
from benchflow import BaseBench, BaseBenchConfig

class TauBenchConfig(BaseBenchConfig):
    def __init__(self, params: Dict[str, Any]):
        required_env = ["TEST_START_IDX", "TEST_END_IDX"]  
        optional_env = []  
        defaults = {
        "RESULTS_DIR": "/app/results"  
        }


class TauBenchBench(BaseBench):
    def __init__(self):
        super().__init__()

    def get_config(self, params: Dict[str, Any], task_id: str) -> BaseBenchConfig:
        params["TEST_START_IDX"] = task_id
        params["TEST_END_IDX"] = str(int(task_id) + 1)
        return TauBenchConfig(params)
    
    def get_image_name(self) -> str:
        return "taubench-benchflow:latest"
    
    def get_results_dir_in_container(self) -> str:
        return "/app/results"
    
    def get_log_files_dir_in_container(self) -> str:
        return "/app/log_files"
    
    def get_result(self, task_id: str) -> Dict[str, Any]:
        result_file = os.path.join(self.results_dir, f"{task_id}.json") 
        if not os.path.exists(result_file):
            return {"is_resolved": False, "score": 0, "message": {"error": "No results found"}}
        
        try:
            with open(result_file, 'r') as f:
                task_result = json.load(f)
                is_resolved = task_result.get("reward", 0) >= 1.0
                score = float(task_result.get("reward", 0))
                return {
                    "is_resolved": is_resolved,
                    "score": score,
                    "message": {"details": task_result.get("info", {})},
                    "log": json.dumps(task_result, indent=2)
                }
        except Exception as e:
            return {
                "is_resolved": False,
                "score": 0,
                "message": {"error": f"Failed to parse result file: {e}"},
                "log": f"Error parsing result file: {e}"
            }
    
    def get_all_tasks(self, split: str) -> Dict[str, Any]:
        if split == "train":
            env_task_counts = {
                "retail": 100,
                "airline": 80
            }
        else:  
            env_task_counts = {
                "retail": 50,
                "airline": 40
            }
        
        env = os.environ.get("ENV", "retail") 
        task_count = env_task_counts.get(env, 50)
        
        task_ids = [str(i) for i in range(task_count)]
        return {"task_ids": task_ids, "error_message": None}