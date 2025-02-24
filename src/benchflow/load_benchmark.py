from .Bench import Bench
from .benchmarks import __all__ as BENCHMARKS


def load_benchmark(benchmark_name: str, bf_token: str) -> Bench:
    """
    Load the benchmark. You need to get a bf_token on https://benchflow.ai.
    For example:
    ```
    from benchflow import load_benchmark
    bench = load_benchmark("webarena", "your_bf_token")
    ```
    """
    benchmarks = [
        (parts := benchmark.lower().rsplit("bench", 1))[0] + (parts[1] if len(parts) > 1 else "") # to be deleted in benchflow v0.2.0
        for benchmark in BENCHMARKS
    ]
    if benchmark_name not in benchmarks:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    return Bench(benchmark_name, bf_token)