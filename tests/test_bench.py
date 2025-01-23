from benchflow import load_benchmark
from langbase import WebarenaAgent

bench = load_benchmark(benchmark_name="webarena")

your_agents = WebarenaAgent()

results = bench.run(
    task_ids=["0"], 
    agents=your_agents
)

print(results)