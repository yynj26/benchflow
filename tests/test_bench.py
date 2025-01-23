from benchflow import load_benchmark
from langbase import WebarenaAgent

bench = load_benchmark(benchmark_name="webarena")

your_agents = WebarenaAgent()

run_ids = bench.run(
    task_ids=[0,1], 
    agents=your_agents
)

results = bench.get_results(run_ids)
print(results)