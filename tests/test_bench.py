from benchflow import load_benchmark
from langbase import WebarenaAgent
import os

bench = load_benchmark(benchmark_name="webarena")

your_agents = WebarenaAgent()

run_ids = bench.run(
    task_ids=[0], 
    agents=your_agents,
    api={"LANGBASE_API_KEY": os.getenv("LANGBASE_API_KEY")}
)

results = bench.get_results(run_ids)
print(results)