# webarena_bench.py
import os
import subprocess
from typing import Any, Dict

from benchflow import BaseBench
from benchflow.schemas import BenchArgs, BenchmarkResult


class WebArenaBench(BaseBench):
    def __init__(self):
        super().__init__()

    def get_args(self, task_id: str) -> BenchArgs:
        """
        Return a WebArenaConfig instance that validates the input arguments.
        """
        arguments = {
            "required": [],
            "optional": [
                {"TEST_END_IDX": str(int(task_id) + 1)}
            ]
        }
        return BenchArgs(arguments)
    
    def get_image_name(self) -> str:
        """
        Return the Docker image name for running the WebArena benchmark.
        """
        return "kirk2000/benchflow:webarena-v1"
    
    def get_results_dir_in_container(self) -> str:
        """
        Return the directory inside the container where the benchmark results will be stored.
        """
        return "/app/results"
    
    def get_log_files_dir_in_container(self) -> str:
        """
        Return the directory inside the container where the log files will be stored.
        """
        return "/app/log_files"
    
    def get_result(self, task_id: str) -> BenchmarkResult:
        """
        Read and parse the benchmark result from the log files.
        
        This method expects a file named 'log_files.txt' in the results directory.
        It then reads the content of each log file listed in 'log_files.txt',
        aggregates the log output, and extracts the average score and pass status.
        """
        log_files_txt = os.path.join(self.results_dir, "log_files.txt")
        if not os.path.exists(log_files_txt):
            return BenchmarkResult(task_id=task_id, is_resolved=False, metrics={"score": 0},log={"error": "No results found"}, other={})
        
        log_content = ""
        try:
            with open(log_files_txt, 'r') as f:
                for line in f:
                    log_path = os.path.basename(line.strip())
                    # Assume the log file path is relative to the parent directory of results_dir
                    full_log_path = os.path.join(os.path.dirname(self.log_files_dir), str(task_id), log_path)
                    with open(full_log_path, 'r') as log_file:
                        log_content += log_file.read() + "\n"
        except Exception as e:
            return BenchmarkResult(task_id=task_id, is_resolved=False, metrics={"score": 0}, log={"error": f"Failed to read log files: {e}"}, other={})
        
        # Parse the log content to extract score and status
        is_resolved = False
        score = 0.0
        for line in log_content.splitlines():
            if "Average score:" in line:
                try:
                    score = float(line.split(":")[-1].strip())
                except ValueError:
                    score = 0.0
            if "[Result]" in line:
                if "(PASS)" in line:
                    is_resolved = True
                    
        return BenchmarkResult(task_id=task_id, is_resolved=is_resolved, metrics={"score": score}, log={"details": log_content}, other={})
    
    def get_all_tasks(self, split: str) -> Dict[str, Any]:
        """
        Return a dictionary with all task IDs and an optional error message.
        For 'train' split, return 200 tasks; otherwise, return 812 tasks.
        """
        if split == "train":
            task_ids = [str(i) for i in range(200)]
        else:
            task_ids = [str(i) for i in range(812)]
        return {"task_ids": task_ids, "error_message": None}
    
    def cleanup(self):
        """
        Clean up benchmark resources by removing the local results and log files directories.
        """
        if os.path.exists(self.results_dir):
            self.logger.info(f"Removing {self.results_dir}")
            subprocess.run(['sudo', 'rm', '-rf', self.results_dir], check=True)
        if os.path.exists(self.log_files_dir):
            self.logger.info(f"Removing {self.log_files_dir}")
            subprocess.run(['sudo', 'rm', '-rf', self.log_files_dir], check=True)
