from .Bench import Bench

BENCHMARK_REGISTRY = {
    "cmu/webarena": "http://webarena-benchmark:8000",
    "cmu/mind2web": "http://mind2web-benchmark:8001",
    "other/benchmark": "http://other-benchmark:8002"
}

def load_benchmark(benchmark_name: str) -> Bench:
    """
    """
    if benchmark_name not in BENCHMARK_REGISTRY:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    benchmark_url = BENCHMARK_REGISTRY[benchmark_name]
    return Bench(benchmark_url)