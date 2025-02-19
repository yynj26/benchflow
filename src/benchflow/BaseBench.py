import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, final

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


class BaseBenchConfig:
    """
    Config Class for all benchmark configurations.
    You can specify the required and optional parameters 
    in the __init__ method in your subclass of BaseBench.
    
    For example:
    ```
        def __init__(self, params: Dict[str, Any]):
            params.setdefault("PARAM1", "DEFAULT1")
            params.setdefault("PARAM2", "DEFAULT2")
            params.setdefault("PARAM3", "DEFAULT3")
            params.setdefault("PARAM4", "DEFAULT4")
            self.required_params = ["PARAM1", "PARAM2"]
            self.optional_params = ["PARAM3", "PARAM4"]
            super().__init__(params)
    ```
            
    """
    required_params: List[str] = []
    optional_params: List[str] = []

    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def validate(self):
        """
        Validate the parameters passed to the benchmark.
        """
        missing = [
            key for key in self.required_params
            if key not in self.params or not self.params[key]
        ]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    @final
    def get_params(self) -> Dict[str, str]:
        """
        Get the parameters of the benchmark.
        The parameters are used to configure the benchmark.
        """
        params = {}
        for key in self.required_params + self.optional_params:
            value = self.params.get(key)
            if value is not None:
                params[key] = str(value)
        for key, value in self.defaults.items():
            if key not in params:
                params[key] = str(value)
        return params

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
        config.validate()

        bench_name = self.__class__.__name__
        timestamp = str(time.time())
        self.results_dir = os.path.abspath(f"./tmp/{bench_name}/results/{timestamp}/{task_id}")
        self.log_files_dir = os.path.abspath(f"./tmp/{bench_name}/logs/{timestamp}/{task_id}")
        os.makedirs(self.results_dir, exist_ok=True)    
        os.makedirs(self.log_files_dir, exist_ok=True)

        params = config.get_params()
        params.update({
            "AGENT_URL": agent_url,
            "TEST_START_IDX": str(task_id),
        })

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
        if result.get("is_resolved") is None:
            return False
        if result.get("score") is None:
            return False
        if result.get("message") is None:
            return False
        return True
    
    def cleanup(self):
        """
        Clean up benchmark resources.
        It will be called when the benchmark is finished.
        Just leave it empty if you don't need to clean up anything.
        """
        pass
    
    @abstractmethod
    def get_config(self, params: Dict[str, Any], task_id: str) -> BaseBenchConfig:
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
    def get_result(self, task_id: str) -> Dict[str, Any]:
        """
        You should return the results in this function.
        
        The result should be a dictionary with the following keys:
        ```
        { 
            "is_resolved": a boolean value indicating if the task is resolved, 
            "score": a float value indicating the score of the task, 
            "message": a dictionary containing any information you want to show to the agent user,
            "log": a string containing the log of the task (trace, trajectory, etc)
        }
        ```
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
