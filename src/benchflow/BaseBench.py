import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from pydantic import ValidationError
from benchflow.schemas import BenchmarkResult, BenchConfig
from typing import Any, Dict, Optional, final

import docker


class ColoredFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(
           fmt='%(colored_level)s: -- %(name)s -- %(message)s',
           datefmt='%H:%M:%S'
       )

    def format(self, record):
        record.msg = " ".join(record.msg.strip().splitlines())
        return super().format(record).strip()

def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(ColoredFormatter())
            logger.addHandler(file_handler)

    return logger

class BaseBench(ABC):
    """
    Base class for all benchmarks. (Now you should name your benchmark class end with "Bench". To be deleted in benchflow v0.2.0)    
    If you want to integrate your benchmark with BenchFlow, you need to implement the following methods:
    ```
    - get_config
    - get_image_name
    - get_results_dir_in_container
    - get_log_files_dir_in_container
    - get_result
    - get_all_tasks
    ```
    Please open a PR to add your benchmark to the BenchFlow benchmarks.
    All you need to include in the PR is a script with the definition of the subclass of BaseBench and BaseBenchConfig.
    """
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.docker_client = docker.from_env()

    @final
    def run_bench(self, task_id: str, agent_url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the benchmark through docker.
        """
        config = self.get_config(params, task_id)
        config.model_validate(config)
        params = config.get_params()
        params.update({
            "AGENT_URL": agent_url,
            "TEST_START_IDX": str(task_id),
        })

        bench_name = self.__class__.__name__
        timestamp = str(time.time())
        self.results_dir = os.path.abspath(f"./tmp/{bench_name}/results/{timestamp}/{task_id}")
        self.log_files_dir = os.path.abspath(f"./tmp/{bench_name}/logs/{timestamp}/{task_id}")
        os.makedirs(self.results_dir, exist_ok=True)    
        os.makedirs(self.log_files_dir, exist_ok=True)

        try:
            container = self.docker_client.containers.run(
                image=self.get_image_name(),
                environment=params,
                volumes=self.get_volumes(),
                remove=True,
                detach=True
            )

            for line in container.logs(stream=True):
                line_str = line.decode('utf-8').strip()
                self.logger.info(line_str)

            container.wait()

            result = self.get_result(task_id)
            if not self.validate_result(result):
                return self.format_result(task_id, False, 0, {"error": "Benchmark result is invalid", "result": str(result)})
            print(result)
            return self.format_result(task_id, result["is_resolved"], result["score"], result["message"], result["log"])
        except docker.errors.ImageNotFound:
            return self.format_result(task_id, False, 0, {"error": "Image not found"})
        except Exception as e:
            self.logger.exception("Error during benchmark execution:")
            return self.format_result(task_id, False, 0, {"error": str(e)})

    @final
    def format_result(self, task_id: str, is_resolved: bool, score: float, message: Dict[str, Any], log: Optional[str] = None) -> Dict[str, Any]:
        """
        Format the result of the benchmark.
        
        The result should be a dictionary with the following keys:
        ``` 
        {
            "task_id": the id of the task,
            "is_resolved": a boolean value indicating if the task is resolved,
            "score": a float value indicating the score of the task,
            "message": a dictionary containing any information you want to show to the agent user,
            "log": a string containing the log of the task (trace, trajectory, etc)
        }
        ```
        """
        if log is None:
            log = message["error"]
        return {
            "task_id": task_id,
            "is_resolved": is_resolved,
            "score": score,
            "message": message,
            "log": log
        }

    @final
    def get_volumes(self) -> Dict[str, Dict[str, str]]:
        """
        Get the volumes of the benchmark.
        The volumes are used to store the results and log files of the benchmark.
        """
        return {
            f"{self.results_dir}": {
                'bind': f"{self.get_results_dir_in_container()}",
                'mode': 'rw'
            },
            f"{self.log_files_dir}": {
                'bind': f"{self.get_log_files_dir_in_container()}",
                'mode': 'rw'
            },
            "/var/run/docker.sock": {
                'bind': "/var/run/docker.sock",
                'mode': 'rw'
            }
        }
    
    @final
    def validate_result(self, result: Dict[str, Any]) -> bool:
        """
        Validate the result of the benchmark.
        """
        try:
            BenchmarkResult.model_validate(result)
            return True
        except ValidationError as e:
            self.logger.error(f"BenchmarkResult validation failed: {e}")
            return False
    
    def cleanup(self):
        """
        Clean up benchmark resources.
        It will be called when the benchmark is finished.
        Just leave it empty if you don't need to clean up anything.
        """
        pass
    
    @abstractmethod
    def get_config(self, params: Dict[str, Any], task_id: str) -> BenchConfig:
        """
        Benchmark need to deal with the END_IDX so that it can only run one task at a time
        task_id is the start index of the task. You can also make your benchmark a single 
        whole task. But if you want to run your benchmark in parallel, you need to split 
        your benchmark into multiple tasks.
        """
        pass

    @abstractmethod
    def get_image_name(self) -> str:
        """
        Return the image name you uploaded to the docker hub.
        """
        pass

    @abstractmethod
    def get_results_dir_in_container(self) -> str:
        """
        Return the directory in the container to store the results.
        """
        pass

    @abstractmethod
    def get_log_files_dir_in_container(self) -> str:
        """
        Return the directory in the container to store the log files (trace, trajectory, etc).
        """
        pass

    @abstractmethod
    def get_result(self, task_id: str) -> BenchmarkResult:
        """
        You should return the results in this function.
        
        Return a BenchmarkResult containing the benchmark results.

        The BenchmarkResult model has the following fields:
            - is_resolved (bool): Indicates whether the task is resolved.
            - message (dict): Contains additional information to be displayed to the agent user.
            - log (str): Contains the log output (e.g., trace, trajectory, etc).
            - metrics (dict): A dictionary of various metrics, where each metric can be of different types (e.g., bool, int, float, or str).
            - other (dict): Any extra fields or metadata relevant to the benchmark result.
        
        Please refer to the example in the definition of BenchmarkResult for the expected format.
        """
        pass

    @abstractmethod
    def get_all_tasks(self, split: str) -> Dict[str, Any]:
        """
        Return all task_ids and optional error messages.
        
        For example:
        ```
            { "task_ids": [...], "error_message": None }
        ```
        
        You can use index as the task_id if your benchmark doesn't have a meaningful field for task_id.
        
        For example:
        ```
            task_ids = list(str(i) for i in range(len(number_of_your_benchmark_tasks)))
        ```
        """
        pass
