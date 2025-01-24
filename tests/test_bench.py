from benchflow import load_benchmark
from agents import WebarenaAgent
import os

bench = load_benchmark(benchmark_name="webarena")

your_agents = WebarenaAgent()

run_ids = bench.run(
    task_ids=[0], 
    agents=your_agents,
    api={"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")}
)

results = bench.get_results(run_ids)
print(results)