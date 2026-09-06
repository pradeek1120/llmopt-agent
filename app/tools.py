from itertools import product
from .models import ExperimentConfig, ProjectConfig
from .benchmark import simulate_benchmark

def generate_candidates(project: ProjectConfig, limit: int = 12) -> list[ExperimentConfig]:
    candidates = [
        ExperimentConfig(
            precision=p,
            kv_cache_precision=k,
            batch_size=b,
        )
        for p, k, b in product(
            project.search_space.precisions,
            project.search_space.kv_cache_precisions,
            project.search_space.batch_sizes,
        )
    ]
    # Keep V1 bounded. V2 can use an adaptive search strategy.
    return candidates[:limit]

def benchmark_candidates(project: ProjectConfig, candidates):
    return [simulate_benchmark(project, c) for c in candidates]
