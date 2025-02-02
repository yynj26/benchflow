from benchflow import load_benchmark
from webcanvas_openai import WebcanvasAgent
import os

bench = load_benchmark(benchmark_name="webcanvas")

your_agents = WebcanvasAgent()

params = {
   "browserbase_api_key": os.environ.get("BROWSERBASE_API_KEY"),
   "graphql_username": os.environ.get("GRAPHQL_USERNAME"), 
   "graphql_password": os.environ.get("GRAPHQL_PASSWORD"),
   "openai_api_key": os.environ.get("OPENAI_API_KEY")
}

run_ids = bench.run(
    task_ids=[0],
    agents=your_agents,
    api={"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")},
    params=params
)

results = bench.get_results(run_ids)
print(results)