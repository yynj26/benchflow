import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from benchflow import BenchClient

LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE = f"{LOG_FOLDER}/log{datetime.now().strftime('%Y%m%d%H%M%S')}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler(LOG_FILE))
logger.addHandler(logging.StreamHandler())

class TauBenchClient(BenchClient):
    def __init__(self, agent_url: str, max_retry: int = 1):
        super().__init__(agent_url, max_retry)

    def prepare_environment(self, state_update: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "env_info": {
                "observation": state_update.get("observation", ""),
                "task": state_update.get("task", {}),
                "tools_info": state_update.get("tools_info", []),
                "wiki": state_update.get("wiki", "")
            }
        }

    def parse_action(self, raw_action: Dict[str, Any]) -> Dict[str, Any]:
        return raw_action

def run_task(env, task_index: int, agent_client: TauBenchClient, max_steps: int) -> Dict[str, Any]:
    print(f"Running task {task_index}")
    reset_response = env.reset(task_index=task_index)
    observation = reset_response.observation
    task = reset_response.info.task

    messages = []
    steps = 0
    done = False
    reward = 0

    while not done and steps < max_steps:
        task_data = task.dict() if hasattr(task, 'dict') else task.model_dump() if hasattr(task, 'model_dump') else task

        state_update = {
            "observation": observation,
            "task": task_data,
            "tools_info": env.tools_info,
            "wiki": env.wiki
        }

        action_data = agent_client.get_action(state_update)
        action = {
            "name": action_data.get("name"),
            "kwargs": action_data.get("kwargs", {})
        }

        messages.append({
            "role": "assistant",
            "content": json.dumps(action)
        })

        response = env.step(action)
        observation = response.observation
        reward = response.reward
        done = response.done

        messages.append({
            "role": "user",
            "content": observation
        })

        steps += 1

    return {
        "task_id": task_index,
        "reward": reward,
        "info": {"steps": steps, "done": done},
        "traj": messages,
        "trial": 0
    }

def main():
    agent_url = os.environ.get("AGENT_URL")
    start_idx = os.environ.get("TEST_START_IDX")
    end_idx = os.environ.get("TEST_END_IDX")

#in docker we clone the repository
    from tau_bench.envs import get_env

    env_name = os.environ.get("ENV", "retail")
    user_strategy = os.environ.get("USER_STRATEGY", "llm")
    user_model = os.environ.get("USER_MODEL", "gpt-4o")
    user_model_provider = os.environ.get("USER_MODEL_PROVIDER", "openai")
    task_split = os.environ.get("TASK_SPLIT", "test")
    log_dir = os.environ.get("LOG_DIR", "/app/results")
    max_steps = int(os.environ.get("MAX_STEPS", "30"))

    os.makedirs(log_dir, exist_ok=True)

    agent_client = TauBenchClient(agent_url=agent_url)

    end_index = int(end_idx) if end_idx else int(start_idx) + 1
    task_indices = list(range(int(start_idx), end_index))

    results = []
    for task_index in task_indices:
        isolated_env = get_env(
            env_name,
            user_strategy=user_strategy,
            user_model=user_model,
            user_provider=user_model_provider,
            task_split=task_split,
            task_index=task_index,
        )

        result = run_task(isolated_env, task_index, agent_client, max_steps)
        results.append(result)

        print(f"{'success' if result['reward'] == 1.0 else 'fail'} task_id={task_index}")

        # save individual task result 
        task_result_path = f"{log_dir}/{task_index}.json"
        with open(task_result_path, "w") as f:
            json.dump(result, f, indent=2)

    # save complete result
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_path = f"{log_dir}/results-{time_str}.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {result_path}")

if __name__ == "__main__":
    main()