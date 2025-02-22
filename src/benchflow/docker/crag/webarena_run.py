import argparse
import base64
import glob
import json
import logging
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import create_id_based_action, create_none_action, create_playwright_action, is_equivalent
from browser_env.auto_login import get_site_comb_from_filepath
from browser_env.env_config import URL_MAPPINGS
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from benchflow import BenchClient
from evaluation_harness import evaluator_router
import requests

LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    parser.add_argument(
        "--agent_url",
        default="http://0.0.0.0:9000",
        help="Path to the agent file"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the browser"
    )
    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["accessibility_tree", "html", "image"],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=720)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)

    parser.add_argument("--max_steps", type=int, default=30)

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="agents/prompts/state_action_agent.json",
    )
    parser.add_argument(
        "--parsing_failure_th",
        help="When concesecutive parsing failure exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When concesecutive repeating action exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )

    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=1920,
    )
    parser.add_argument(
        "--model_endpoint",
        help="huggingface model endpoint",
        type=str,
        default="",
    )

    # example config
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=1000)

    # logging related
    parser.add_argument("--result_dir", type=str, default="")
    args = parser.parse_args()

    # check the whether the action space is compatible with the observation space
    if (
        args.action_set_tag == "id_accessibility_tree"
        and args.observation_type != "accessibility_tree"
    ):
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )

    return args


def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to early stop"""

    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [
                action["action_type"] == ActionTypes.NONE
                for action in last_k_actions
            ]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all(
                [
                    is_equivalent(action, last_action)
                    for action in last_k_actions
                ]
            ):
                return True, f"Same action for {k} times"

    else:
        # check the action sequence
        if (
            sum([is_equivalent(action, last_action) for action in action_seq])
            >= k
        ):
            return True, f"Same typing action for {k} times"

    return False, ""

def test(
    args: argparse.Namespace,
    agent_url: str,
    config_file_list: list[str],
) -> None:
    scores = []
    max_steps = args.max_steps

    early_stop_thresholds = {
        "parsing_failure": args.parsing_failure_th,
        "repeating_action": args.repeating_action_failure_th,
    }

    env = ScriptBrowserEnv(
        headless=not args.render,
        slow_mo=args.slow_mo,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_size={
            "width": args.viewport_width,
            "height": args.viewport_height,
        },
        save_trace_enabled=args.save_trace_enabled,
        sleep_after_execution=args.sleep_after_execution,
    )

    # Initialize agent process
    agent = WebArenaClient(
        agent_url=agent_url,
        action_set_tag=args.action_set_tag,
        max_retry=args.max_retry
    )

    for config_file in config_file_list:
        try:
            render_helper = RenderHelper(
                config_file, args.result_dir, args.action_set_tag
            )

            with open(config_file) as f:
                _c = json.load(f)
                intent = _c["intent"]
                task_id = _c["task_id"]
                if _c["storage_state"]:
                    config_file = _handle_auth_state(_c, config_file)

            logger.info(f"[Config file]: {config_file}")
            logger.info(f"[Intent]: {intent}")

            logger.info("[Environment reset]")
            trajectory: Trajectory = []
            obs, info = env.reset(options={"config_file": config_file})
            state_info: StateInfo = {"observation": obs, "info": info}
            trajectory.append(state_info)

            meta_data = {"action_history": ["None"]}
            
            # main loop
            while True:
                early_stop_flag, stop_info = early_stop(
                    trajectory, max_steps, early_stop_thresholds
                )

                if early_stop_flag:
                    action = create_stop_action(f"Early stop: {stop_info}")
                else:
                    try:
                        # Get next action from agent
                        state_update = {
                            "trajectory": trajectory,
                            "meta_data": meta_data,
                            "intent": intent
                        }
                        action = agent.get_action(state_update)
                    except Exception as e:
                        logger.error(f"Error in get_action: {str(e)}")
                        action = create_stop_action(f"ERROR: {str(e)}")
                
                logger.info(f"action")
                trajectory.append(action)

                action_str = get_action_description(
                    action,
                    state_info["info"]["observation_metadata"],
                    action_set_tag=args.action_set_tag,
                    prompt_constructor=None
                )
                render_helper.render(
                    action, state_info, meta_data, args.render_screenshot
                )
                meta_data["action_history"].append(action_str)

                if action["action_type"] == ActionTypes.STOP:
                    break

                obs, _, terminated, _, info = env.step(action)
                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)

                if terminated:
                    trajectory.append(create_stop_action(""))
                    break

            # Evaluate results
            evaluator = evaluator_router(config_file)
            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=env.page,
                client=env.get_page_client(env.page),
            )

            scores.append(score)
            logger.info(f"[Result] ({'PASS' if score == 1 else 'FAIL'}) {config_file}")

            if args.save_trace_enabled:
                env.save_trace(
                    Path(args.result_dir) / "traces" / f"{task_id}.zip"
                )

        except Exception as e:
            logger.error(f"[Unhandled Error] {repr(e)}]")
            _handle_error(args.result_dir, config_file, e)

        render_helper.close()

    env.close()
    logger.info(f"Average score: {sum(scores) / len(scores)}")
    print(f"Average score: {sum(scores) / len(scores)}")


def _handle_auth_state(config: dict, config_file: str) -> str:
    cookie_file_name = os.path.basename(config["storage_state"])
    comb = get_site_comb_from_filepath(cookie_file_name)
    temp_dir = tempfile.mkdtemp()
    
    subprocess.run([
        "python",
        "browser_env/auto_login.py",
        "--auth_folder", temp_dir,
        "--site_list", *comb,
    ])
    
    config["storage_state"] = f"{temp_dir}/{cookie_file_name}"
    assert os.path.exists(config["storage_state"])
    
    new_config_file = f"{temp_dir}/{os.path.basename(config_file)}"
    with open(new_config_file, "w") as f:
        json.dump(config, f)
        
    return new_config_file

def _handle_error(result_dir: str, config_file: str, error: Exception) -> None:
    import traceback
    with open(Path(result_dir) / "error.txt", "a") as f:
        f.write(f"[Config file]: {config_file}\n")
        f.write(f"[Unhandled Error] {repr(error)}\n")
        f.write(traceback.format_exc())

def prepare(args: argparse.Namespace) -> None:
    # convert prompt python files to json
    from agent.prompts import to_json

    to_json.run()

    # prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = (
            f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
        )
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)

    # log the log file
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")


def get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    result_files = glob.glob(f"{result_dir}/*.html")
    task_ids = [
        os.path.basename(f).split(".")[0].split("_")[1] for f in result_files
    ]
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        if task_id not in task_ids:
            unfinished_configs.append(config_file)
    return unfinished_configs


def dump_config(args: argparse.Namespace) -> None:
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")

class WebArenaClient(BenchClient):
    def __init__(self, agent_url: str, action_set_tag: str, max_retry: int = 1):
        super().__init__(agent_url, max_retry)
        self.action_set_tag = action_set_tag
        self.url_mappings = {}

    def prepare_environment(self, state_update: Dict[str, Any]) -> Dict[str, Any]:
        trajectory = state_update["trajectory"]
        meta_data = state_update.get("meta_data", {})
        intent = state_update.get("intent", "")
        
        state_info = trajectory[-1]
        observation = state_info["observation"].copy()
        
        if isinstance(observation, dict) and "image" in observation:
            self._process_image(observation)
            
        page = state_info["info"]["page"]
        url = page.url
        mapped_url = self._map_url(url)
        
        return {"env_info": {
            "observation": observation,
            "url": mapped_url,
            "intent": intent,
            "previous_action": meta_data.get('action_history', ['None'])[-1]
        }}

    def parse_action(self, raw_action: Dict[str, Any]) -> Dict[str, Any]:
        if self.action_set_tag == "id_accessibility_tree":
            return create_id_based_action(raw_action)
        elif self.action_set_tag == "playwright":
            return create_playwright_action(raw_action)
        else:
            raise ValueError(f"Unknown action type {self.action_set_tag}")

    def _process_image(self, observation: Dict[str, Any]) -> None:
        image = observation["image"]
        if hasattr(image, '__array_interface__'):
            observation["image_shape"] = image.shape
            observation["image_dtype"] = str(image.dtype)
            image_bytes = image.tobytes()
            observation["image"] = base64.b64encode(image_bytes).decode('utf-8')

    def _map_url(self, url: str) -> str:
        for internal, real in self.url_mappings.items():
            if internal in url:
                return url.replace(internal, real)
        return url


if __name__ == "__main__":
    args = config()
    args.sleep_after_execution = 2.0
    prepare(args)

    test_file_list = []
    st_idx = args.test_start_idx
    ed_idx = args.test_end_idx
    for i in range(st_idx, ed_idx):
        test_file_list.append(f"config_files/{i}.json")
    if "debug" not in args.result_dir:
        test_file_list = get_unfinished(test_file_list, args.result_dir)

    if len(test_file_list) == 0:
        logger.info("No task left to run")
    else:
        print(f"Total {len(test_file_list)} tasks left")
        args.render = False
        args.render_screenshot = True
        args.save_trace_enabled = True

        args.current_viewport_only = True
        dump_config(args)

        test(
            args=args,
            agent_url=args.agent_url,
            config_file_list=test_file_list
        )