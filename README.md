# LLMOpt-Agent

LLMOpt-Agent is an agentic optimization workflow for LLM inference tuning. It evaluates different model precision settings, KV-cache precision, and batch-size combinations, then recommends the best configuration for a workload under latency and quality constraints.

This project is a V1 prototype designed to make inference optimization testable, debuggable, and easy to extend. The current benchmark layer is a deterministic simulator, not a real GPU benchmark, which keeps the system lightweight and portable while still demonstrating the optimization loop.

## Why this project exists

Serving large language models efficiently requires balancing several tradeoffs:

- latency vs throughput
- memory usage vs quality
- batch size vs response time
- lower precision vs model fidelity

This project automates that search process by:

- analyzing the workload and hardware profile
- generating candidate configurations
- simulating benchmark outcomes
- checking constraints
- selecting the best valid configuration

## What it does

The workflow includes:

1. Analysis agent
   - identifies workload type
   - checks concurrency pressure
   - estimates memory characteristics

2. Optimization agent
   - generates candidate inference configurations
   - explores precision, KV-cache, and batch-size combinations

3. Benchmark simulator
   - calculates TTFT, TPOT, throughput, VRAM, quality score, and GPU utilization
   - evaluates whether constraints are satisfied

4. Evaluation agent
   - picks the best valid candidate
   - decides whether the search should continue

## Example workflow

The project reads a JSON config such as:

```json
{
  "model_name": "Llama-3.1-8B-Instruct",
  "hardware": {
    "vendor": "AMD",
    "gpu": "MI355X",
    "vram_gb": 192,
    "memory_bandwidth_gbps": 8000
  },
  "workload": {
    "prompt_tokens": 4096,
    "output_tokens": 512,
    "concurrency": 32,
    "target_ttft_ms": 100,
    "target_quality": 0.98
  },
  "search_space": {
    "precisions": ["fp16", "fp8", "int8"],
    "kv_cache_precisions": ["fp16", "fp8"],
    "batch_sizes": [8, 16, 32]
  },
  "max_iterations": 2
}
```

Then it runs the optimization loop and prints a recommendation.

## Run it locally

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m app.main --config configs/example.json
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.main --config configs/example.json
```

## Example output

I verified this locally with:

```bash
python -m app.main --config configs/example.json
```

and it successfully produced:

```text
=== LLMOpt-Agent Report ===
Model: Llama-3.1-8B-Instruct
GPU: MI355X
Workload: prefill_heavy
Diagnosis: Candidate satisfies the quality/TTFT constraints. Further search can test whether a better throughput-memory tradeoff exists.

Recommended configuration:
{
  "precision": "fp8",
  "kv_cache_precision": "fp8",
  "batch_size": 32
}

Measured/simulated metrics:
  ttft_ms: 87.45
  tpot_ms: 12.568
  throughput_tok_s: 125.395
  vram_gb: 13.75
  quality_score: 0.99
  gpu_utilization_pct: 95.7
  passed_constraints: True
```

## Project structure

```text
llmopt_agent/
├── app/
│   ├── agents.py
│   ├── benchmark.py
│   ├── config.py
│   ├── graph.py
│   ├── main.py
│   ├── models.py
│   └── tools.py
├── configs/
│   └── example.json
├── tests/
│   └── test_core.py
├── requirements.txt
├── README.md
├── PUBLISHING.md
├── proof-output.txt
└── .env.example
```

## Technical notes

This project is intentionally designed to be:

- deterministic
- debuggable
- easy to test
- portable
- upgradeable to real benchmark backends

The benchmark backend is intentionally simulated in V1 and is meant to be replaced later with real measurements from:

- vLLM
- PyTorch
- ROCm/HIP
- FP16/BF16/FP8/INT8
- KV-cache tuning
- continuous batching
- TTFT and TPOT evaluation

## Limitations

This is not yet a real GPU benchmarking platform. The numbers are estimates produced by a simulator and should not be interpreted as real hardware benchmarks.

## Roadmap

- integrate real inference backend measurements
- add richer search strategies
- support adaptive optimization loops
- expose results as a dashboard or API
- connect to real serving metrics

## License

This project is provided as an educational and research prototype for LLM inference optimization.
