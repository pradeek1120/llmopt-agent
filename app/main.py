from __future__ import annotations

import argparse
import json

from .config import load_config
from .graph import build_graph

def main():
    parser = argparse.ArgumentParser(description="LLMOpt-Agent V1")
    parser.add_argument("--config", default="configs/example.json")
    args = parser.parse_args()

    project = load_config(args.config)
    graph = build_graph(project)

    result = graph.invoke({})

    print("\n=== LLMOpt-Agent Report ===")
    print(f"Model: {project.model_name}")
    print(f"GPU: {project.hardware.gpu}")
    print(f"Workload: {result['analysis']['workload_type']}")
    print(f"Diagnosis: {result['diagnosis']}")

    selected = result["selected"]
    print("\nRecommended configuration:")
    print(json.dumps(selected["config"], indent=2))
    print("\nMeasured/simulated metrics:")
    for key in [
        "ttft_ms",
        "tpot_ms",
        "throughput_tok_s",
        "vram_gb",
        "quality_score",
        "gpu_utilization_pct",
        "passed_constraints",
    ]:
        print(f"  {key}: {selected[key]}")

if __name__ == "__main__":
    main()
