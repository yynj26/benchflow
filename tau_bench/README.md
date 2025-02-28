# Tau-Bench Integration for BenchFlow:

---

## Docker Setup (in `/tau_bench`)
- Dockerfile: Configures the container with the correct ENTRYPOINT.
- entrypoint.py: Implements the evaluation entrypoint following the BenchClient interface.
- requirements.txt: Specifies the necessary dependencies.

---

## Benchmark Integration
- src/benchflow/benchmarks/taubench.py: 
Defines the Tau-Bench benchmark using the BaseBench interface.

---

## Demo Agent
- src/benchflow/agents/taubench_openai.py: 
Implements a demo agent based on the BaseAgent interface.

---

## Testing
- tests/test_taubench.py:
 Provides a script to test the demo agent on the new benchmark.

---

## Quick Start

### Build the Docker Image
```bash
docker build -t tau-bench -f tau_bench/Dockerfile .
```

### Run the Container
```bash
docker run tau-bench
```

### Run the Tests
```bash
python tests/test_taubench.py
```