import os
import subprocess
from typing import Any, Dict

import docker

from benchflow import BaseBench


class WebArenaBench(BaseBench):
    def __init__(self):
        super().__init__()
        self.image_name = "kirk2000/benchflow:webarena-v1"
        self.results_dir = os.path.abspath("./tmp/webarena/results")
        self.log_files_dir = os.path.abspath("./tmp/webarena/log_files")

    def run_bench(self, task_id: str, agent_url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        client = docker.from_env()
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.log_files_dir, exist_ok=True)

        try:
            container = client.containers.run(
                image=self.image_name,
                environment={
                    "AGENT_URL": agent_url,
                    "TEST_START_IDX": str(task_id),
                    "TEST_END_IDX": str(int(task_id) + 1),
                    "RESULTS_DIR": "/app/results"
                },
                volumes={
                    self.results_dir: {
                        'bind': '/app/results',
                        'mode': 'rw'
                    },
                    self.log_files_dir: {
                        'bind': '/app/log_files',
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

            result_file = os.path.join(self.results_dir)
            if not os.path.exists(result_file):
                return {
                    "task_id": str(task_id),
                    "is_resolved": False,
                    "score": 0,
                    "message": {"error": "No results found"}
                }
                
            results_data = self._get_results(task_id)

            results_lines = results_data.splitlines()
            is_resolved = False
            score = 0.0

            for line in results_lines:
                if "Average score:" in line:
                    score = float(line.split(":")[-1].strip())
                if "[Result]" in line:
                    is_resolved = "(PASS)" in line
                
            return {
                "task_id": str(task_id),
                "is_resolved": is_resolved,
                "score": score,
                "message": {"details": results_data}
            }
            
        except docker.errors.ImageNotFound:
            return {
                "task_id": str(task_id),
                "is_resolved": False,
                "score": 0,
                "message": {"error": "Image not found"}
            }
            
        except Exception as e:
            return {
                "task_id": str(task_id),
                "is_resolved": False,
                "score": 0,
                "message": {"error": f"{e}"}
            }

    def get_all_tasks(self, split: str) -> Dict[str, Any]:
        task_ids = [str(i) for i in range(812)]
        if split == "train":
            task_ids = [str(i) for i in range(200)]
        return {"task_ids": task_ids, "error_message": None}

    def cleanup(self):
        if os.path.exists(self.results_dir):
            self.logger.info(f"Removing {self.results_dir}")
            subprocess.run(['sudo', 'rm', '-rf', self.results_dir], check=True)
        if os.path.exists(self.log_files_dir):
            self.logger.info(f"Removing {self.log_files_dir}")
            subprocess.run(['sudo', 'rm', '-rf', self.log_files_dir], check=True)

    def _get_results(self, task_id: int) -> Dict[str, Any]:
        log_content = ""
        log_files_txt = os.path.join(self.results_dir, "log_files.txt")
        
        if os.path.exists(log_files_txt):
            try:
                with open(log_files_txt, 'r') as f:
                    for line in f:
                        log_path = line.strip()
                        full_log_path = os.path.join(os.path.dirname(self.results_dir), log_path)
                        if os.path.exists(full_log_path):
                            with open(full_log_path, 'r') as log_file:
                                log_content += log_file.read() + "\n"
            except Exception as e:
                print(f"Failed to read log file: {str(e)}")
        
            return log_content    
    
    
