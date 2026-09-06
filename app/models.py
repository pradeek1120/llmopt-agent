from typing import Annotated, Any, TypedDict
from pydantic import BaseModel, Field

class Hardware(BaseModel):
    vendor: str
    gpu: str
    vram_gb: float
    memory_bandwidth_gbps: float

class Workload(BaseModel):
    prompt_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=0)
    concurrency: int = Field(gt=0)
    target_ttft_ms: float = Field(gt=0)
    target_quality: float = Field(gt=0, le=1)

class SearchSpace(BaseModel):
    precisions: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    kv_cache_precisions: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    batch_sizes: list[Annotated[int, Field(gt=0)]] = Field(min_length=1)

class ProjectConfig(BaseModel):
    model_name: str = Field(min_length=1)
    hardware: Hardware
    workload: Workload
    search_space: SearchSpace
    max_iterations: int = Field(default=2, ge=1, le=10)

class ExperimentConfig(BaseModel):
    precision: str
    kv_cache_precision: str
    batch_size: int = Field(gt=0)

class BenchmarkResult(BaseModel):
    config: ExperimentConfig
    ttft_ms: float
    tpot_ms: float
    throughput_tok_s: float
    vram_gb: float
    quality_score: float
    gpu_utilization_pct: float
    passed_constraints: bool

class OptimizationState(TypedDict, total=False):
    config: dict[str, Any]
    analysis: dict[str, Any]
    candidates: list[dict[str, Any]]
    results: list[dict[str, Any]]
    selected: dict[str, Any]
    iteration: int
    continue_loop: bool
    diagnosis: str
