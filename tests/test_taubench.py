import os

from benchflow import load_benchmark
from benchflow.agents.taubench_openai import TauBenchAgent

bench = load_benchmark(benchmark_name="taubench", bf_token=os.getenv("BF_TOKEN"))

your_agent = TauBenchAgent()

run_ids = bench.run(
    task_ids=[0, 1, 2, 3],
    agents=your_agent,
    api={
        "provider": "openai", 
        "model": "gpt-4o-mini", 
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
    },
    requirements_txt="requirements.txt",
    params={}
)

# Get the results
results = bench.get_results(run_ids)