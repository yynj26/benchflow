# 🚀 BenchFlow: AI Benchmark Runtime

## 📦 Installation

### Requirements
- **Python 3.11+**
- Docker (for benchmark integration)

```bash
pip install benchflow
```

---

## 🤖 Agent Development Guide

### ➤ Step 1: Define Your Agent

```python
from benchflow import BaseAgent

class YourAgent(BaseAgent):
    def __init__(self):
        super().__init__()
    
    def call_api(self) -> str:
        """
        IMPLEMENTATION CONTRACT
        Process environment data and generate task solution
        
        Access:
        - self.env_info: dict containing benchmark-specific data
        
        Returns:
        str: Unified diff patch for code tasks
        """
        # Access task parameters
        instance_id = self.env_info['instance_id']
        # You can deal with the data provided in the `env_info` here and 
        # return your prediction(could be a patch action or anything else)

        # Example: return a patch for SWE-Bench
        return (
            "diff --git a/src/rules/L031.py b/src/rules/L031.py\n"
            "--- a/src/rules/L031.py\n"
            "+++ b/src/rules/L031.py\n"
            "@@ -211,7 +211,7 @@ def _lint_aliases_in_join(\n"
            "    violation_buff.append(\n"
            "        LintResult(\n"
            "            anchor=alias_info.alias_identifier_ref,\n"
            " -          description=\"Original message\",\n"
            " +          description=\"Updated message\",\n"
            "            fixes=fixes,\n"
            "        )\n"
            "    )"
        )
```

#### 🔑 Key Requirements
- ✅ Maintain agent logic in a single file
- ✅ Return predictions as formatted strings
- ✅ Access benchmark data through `self.env_info`

---

### ➤ Step 2: Test Your Agent

```python
from benchflow import load_benchmark
from your_agent import YourAgent

# Initialize benchmark
bench = load_benchmark("SWE-Bench")

# Configure agent
agent = YourAgent()

# Execution parameters
config = {
    "task_ids": ["astropy__astropy-12907"],
    "agents": agent,
    "install_sh_dir": "setup.sh",
    "requirements_dir": "requirements.txt",
    "api": {"OPENAI_API_KEY": "your_api_key_here"}
}

# Run evaluation
results = bench.run(**config)
```

#### ⚙️ Configuration Options
| Parameter | Type | Description |
|-----------|------|-------------|
| `task_ids` | List[str] | Benchmark tasks to evaluate |
| `install_sh_dir` | str | Environment setup script path |
| `requirements_dir` | str | Python dependencies file path |
| `api` | Dict[str, str] | Required API credentials |

---

## 🧪 Benchmark Integration Guide

### ➤ Step 1: Implement BenchClient

```python
from benchflow import BenchClient
from typing import Dict, Any

class YourClient(BenchClient):
    def prepare_environment(self, state_update: Dict) -> Dict:
        """Transform raw state to agent inputs"""
        return {
            "env_info": {
                "observation": state_update["trajectory"][-1],
                "intent": state_update.get("intent", "")
            }
        }
    
    def parse_action(self, raw_action: str) -> str:
        """Process agent responses"""
        parsed_action = raw_action # or you can do some post-processing here
        return parsed_action
```

#### 🔄 Processing Workflow
1. **Input Transformation**  
   `prepare_environment()` handles state conversion, you should wrap the raw state into a dict with key `env_info`
   
2. **Output Parsing**  
   `parse_action()` processes agent responses, you should return the results as a string

---

### ➤ Step 2: Create Benchmark Docker Image

```python
from benchflow import BaseBench
import docker

class YourBench(BaseBench):
    def __init__(self):
        super().__init__()
        # You should upload your benchmark image to a public registry and 
        # set the image_name to the image name
        self.image_name = "your_image_name"

    def run_bench(self, task_id, agent_url, params):
        # Run benchmark in Docker container
        container = docker.from_env().containers.run(
            image="your-bench-image",
            environment={
                    "AGENT_URL": agent_url,
                    "TEST_START_IDX": str(task_id),
                    "TEST_END_IDX": str(int(task_id) + 1),
                    "BROWSERBASE_API_KEY": params["browserbase_api_key"],
                    "GRAPHQL_USERNAME": params["graphql_username"],
                    "GRAPHQL_PASSWORD": params["graphql_password"],
                    "OPENAI_API_KEY": params["openai_api_key"],
                    "RESULTS_DIR": "/app/batch_tasks_results/example"
                },
            volumes={...},
            detach=True
        )
        
        # Process results
        return {
            "task_id": task_id,
            "score": calculate_score(),
            "is_resolved": success_flag,
            "message": {'log': 'logs of the task', 'error': 'some error message if the task is not resolved'}
        }
```

**Key Requirements**:
- Package your benchmark as Docker image
- Using ENV variables to pass the args to the benchmark
- Must return a dict with the following keys: `task_id`, `score`, `is_resolved`, `message`

#### 🐳 Docker Requirements
| ENV Variable | Purpose |
|--------------|---------|
| `AGENT_URL` | Agent service endpoint |
| `TASK_ID` | Task identifier |
| `API_KEYS` | JSON string of credentials |

---

## 📚 API Reference

### 🔧 BaseAgent Class

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `call_api` | `self.env_info: Dict` | `str` | Core task processing method |

### 🔌 BenchClient Class

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `prepare_environment` | `Dict` | `Dict` | State conversion |
| `parse_action` | `str` | `str` | Response processing |

### 🧪 BaseBench Class

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `run_bench` | `task_id: str`, `agent_url: str`, `params: Dict` | `Dict` | Benchmark execution |

---