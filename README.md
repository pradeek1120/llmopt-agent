# LLMOpt-Agent

An agentic LLM inference optimization prototype.

## What this first version does

It implements a small, debuggable optimization loop:

1. **Analysis Agent** analyzes model/workload/hardware metadata.
2. **Optimization Agent** proposes inference configurations.
3. **Benchmark Tool** evaluates configurations using a deterministic simulator.
4. **Evaluation/Research Agent** compares performance + quality and decides whether another experiment is useful.
5. A **LangGraph workflow** orchestrates the loop.

The benchmark is intentionally simulated in V1 so the project can run on an ordinary laptop. Later, replace `app/benchmark.py` with real vLLM/PyTorch/ROCm execution.

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
├── .env.example
├── requirements.txt
└── README.md
```

## Setup in VS Code

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then:

```bash
python -m app.main --config configs/example.json
```

Run tests:

```bash
pytest -q
```

## Optional LLM reasoning

The current agents are deterministic by default so the project works without an API key.

To enable an OpenAI-backed reasoning layer:

```bash
copy .env.example .env
```

or on Linux/macOS:

```bash
cp .env.example .env
```

Set:

```text
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o-mini
```

The core optimization loop does not depend on the LLM being available.

## Important V2 direction

Replace the simulator with real measurements:

- vLLM
- PyTorch
- ROCm/HIP
- FP16/BF16/FP8/INT8
- KV-cache experiments
- continuous batching
- TTFT
- TPOT
- throughput
- VRAM
- GPU utilization

Do not claim simulator numbers as real GPU benchmark results.
