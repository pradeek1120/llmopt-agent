"""
V1 benchmark backend.

This is a deterministic simulator, NOT a real GPU benchmark.
The purpose is to make the agent architecture runnable before
connecting expensive/OS-specific inference engines.
"""

from .models import BenchmarkResult, ExperimentConfig, ProjectConfig

PRECISION_SPEED = {
    "fp16": 1.00,
    "bf16": 1.02,
    "fp8": 1.22,
    "int8": 1.16,
    "int4": 1.35,
}

PRECISION_MEMORY = {
    "fp16": 1.00,
    "bf16": 1.00,
    "fp8": 0.55,
    "int8": 0.52,
    "int4": 0.30,
}

KV_MEMORY = {
    "fp16": 1.00,
    "fp8": 0.55,
}

def simulate_benchmark(project: ProjectConfig, exp: ExperimentConfig) -> BenchmarkResult:
    w = project.workload

    speed = PRECISION_SPEED.get(exp.precision, 1.0)
    memory_factor = PRECISION_MEMORY.get(exp.precision, 1.0)
    kv_factor = KV_MEMORY.get(exp.kv_cache_precision, 1.0)

    # Higher concurrency/batch improves aggregate throughput until this simple
    # simulator's saturation point.
    batching_gain = min(1.0 + 0.018 * exp.batch_size, 1.65)
    concurrency_pressure = 1.0 + max(0, w.concurrency - exp.batch_size) * 0.008

    # Lower latency for faster precision, but larger batches can increase TTFT.
    ttft = 92.0 / speed + 0.22 * exp.batch_size + 5.0 * concurrency_pressure
    tpot = 23.0 / speed / min(1.0 + exp.batch_size / 64, 1.5)

    throughput = (1000.0 / tpot) * batching_gain

    base_vram = 18.0
    vram = base_vram * memory_factor
    kv_vram = 7.0 * kv_factor * min(w.concurrency / 32, 2.0)
    vram += kv_vram

    quality = {
        "fp16": 1.000,
        "bf16": 0.999,
        "fp8": 0.992,
        "int8": 0.986,
        "int4": 0.955,
    }.get(exp.precision, 0.95)

    if exp.kv_cache_precision == "fp8":
        quality -= 0.002

    gpu_util = min(98.0, 55.0 + exp.batch_size * 1.1 + (speed - 1.0) * 25)

    passed = ttft <= w.target_ttft_ms and quality >= w.target_quality

    return BenchmarkResult(
        config=exp,
        ttft_ms=round(ttft, 3),
        tpot_ms=round(tpot, 3),
        throughput_tok_s=round(throughput, 3),
        vram_gb=round(vram, 3),
        quality_score=round(quality, 4),
        gpu_utilization_pct=round(gpu_util, 2),
        passed_constraints=passed,
    )
