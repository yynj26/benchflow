from benchflow import load_benchmark
from swe_agent import SWEAgent
import os

bench = load_benchmark(benchmark_name="swebench")

your_agents = SWEAgent()

run_ids = bench.run(
    task_ids=["astropy__astropy-12907"],
    agents=your_agents,
    api={"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")},
    params={}
)

results = bench.get_results(run_ids)
print(results)