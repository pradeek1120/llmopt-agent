import pytest
from pydantic import ValidationError

from app.config import load_config
from app.benchmark import simulate_benchmark
from app.agents import EvaluationResearchAgent
from app.graph import build_graph
from app.models import ExperimentConfig, ProjectConfig
from app.tools import benchmark_candidates, generate_candidates

def test_config_loads():
    cfg = load_config("configs/example.json")
    assert cfg.model_name.startswith("Llama")
    assert cfg.hardware.vram_gb > 0


def test_candidate_generation_is_bounded_and_valid():
    cfg = load_config("configs/example.json")
    candidates = generate_candidates(cfg)

    assert len(candidates) == 12
    assert all(candidate.batch_size > 0 for candidate in candidates)
    assert all(candidate.precision in cfg.search_space.precisions for candidate in candidates)
    assert all(
        candidate.kv_cache_precision in cfg.search_space.kv_cache_precisions
        for candidate in candidates
    )

def test_fp8_uses_less_memory_than_fp16():
    cfg = load_config("configs/example.json")

    fp16 = simulate_benchmark(
        cfg,
        ExperimentConfig(
            precision="fp16",
            kv_cache_precision="fp16",
            batch_size=16,
        ),
    )
    fp8 = simulate_benchmark(
        cfg,
        ExperimentConfig(
            precision="fp8",
            kv_cache_precision="fp8",
            batch_size=16,
        ),
    )

    assert fp8.vram_gb < fp16.vram_gb
    assert fp8.throughput_tok_s > fp16.throughput_tok_s


def test_int8_uses_less_memory_than_fp16():
    cfg = load_config("configs/example.json")
    fp16 = simulate_benchmark(
        cfg, ExperimentConfig(precision="fp16", kv_cache_precision="fp16", batch_size=16)
    )
    int8 = simulate_benchmark(
        cfg, ExperimentConfig(precision="int8", kv_cache_precision="fp16", batch_size=16)
    )

    assert int8.vram_gb < fp16.vram_gb


def test_benchmark_result_has_expected_metrics():
    cfg = load_config("configs/example.json")
    result = simulate_benchmark(
        cfg, ExperimentConfig(precision="fp16", kv_cache_precision="fp16", batch_size=16)
    )

    assert result.config.precision == "fp16"
    assert result.ttft_ms > 0
    assert result.tpot_ms > 0
    assert result.throughput_tok_s > 0
    assert result.vram_gb > 0
    assert 0 < result.quality_score <= 1
    assert 0 < result.gpu_utilization_pct <= 100

def test_quality_constraint_is_enforced():
    cfg = load_config("configs/example.json")
    result = simulate_benchmark(
        cfg,
        ExperimentConfig(
            precision="int4",
            kv_cache_precision="fp16",
            batch_size=16,
        ),
    )
    assert result.quality_score < cfg.workload.target_quality
    assert result.passed_constraints is False


def test_kv_cache_fp8_reduces_memory():
    cfg = load_config("configs/example.json")
    fp16 = simulate_benchmark(
        cfg, ExperimentConfig(precision="fp16", kv_cache_precision="fp16", batch_size=16)
    )
    fp8 = simulate_benchmark(
        cfg, ExperimentConfig(precision="fp16", kv_cache_precision="fp8", batch_size=16)
    )

    assert fp8.vram_gb < fp16.vram_gb


def test_batch_size_changes_throughput_and_ttft():
    cfg = load_config("configs/example.json")
    small = simulate_benchmark(
        cfg, ExperimentConfig(precision="fp16", kv_cache_precision="fp16", batch_size=8)
    )
    large = simulate_benchmark(
        cfg, ExperimentConfig(precision="fp16", kv_cache_precision="fp16", batch_size=32)
    )

    assert large.throughput_tok_s > small.throughput_tok_s
    assert large.ttft_ms > small.ttft_ms


def test_evaluation_selects_a_valid_candidate():
    cfg = load_config("configs/example.json")
    candidates = generate_candidates(cfg)
    results = [result.model_dump() for result in benchmark_candidates(cfg, candidates)]

    selected, _, _ = EvaluationResearchAgent().run(cfg, {}, results)

    assert selected in results
    assert selected["passed_constraints"] is True
    assert selected["ttft_ms"] <= cfg.workload.target_ttft_ms
    assert selected["quality_score"] >= cfg.workload.target_quality


def test_graph_executes_end_to_end():
    cfg = load_config("configs/example.json")
    result = build_graph(cfg).invoke({})

    assert result["analysis"]["model"] == cfg.model_name
    assert result["candidates"]
    assert result["results"]
    assert result["selected"] in result["results"]
    assert result["selected"]["passed_constraints"] is True


def test_missing_model_name_is_rejected():
    cfg = load_config("configs/example.json").model_dump()
    del cfg["model_name"]

    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(cfg)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_name", ""),
        ("workload.prompt_tokens", -1),
        ("workload.concurrency", 0),
        ("workload.target_quality", 1.1),
        ("search_space.precisions", []),
        ("search_space.batch_sizes", [0]),
    ],
)
def test_invalid_project_config_is_rejected(field, value):
    cfg = load_config("configs/example.json").model_dump()
    target = cfg
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value

    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(cfg)


def test_invalid_experiment_batch_size_is_rejected():
    with pytest.raises(ValidationError):
        ExperimentConfig(precision="fp16", kv_cache_precision="fp16", batch_size=0)
