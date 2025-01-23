from benchflow import load_benchmark
from langbase import WebarenaAgent

bench = load_benchmark(benchmark_name="webarena")

your_agents = WebarenaAgent()

run_id = bench.run(
    task_ids=[1, 2, 3], 
    agents=your_agents
)

result = bench.get_result(run_id)